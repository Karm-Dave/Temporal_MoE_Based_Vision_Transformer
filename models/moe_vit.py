# temporal-moe-vit/models/moe_vit.py

import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoTokenizer

# Importing all the custom modular components
from .experts import GenericExpert, MotionExpert, TextureExpert, QA_AlignedExpert, FastChangeExpert
from .router import Router
from .attention import MultiHeadSelfAttention
from .prediction_head import PredictionHead

class MoEFeedForward(nn.Module):
    """
    The Heterogeneous Mixture-of-Experts Feed-Forward block.
    This module contains the router and the specialized experts, and handles the
    complex logic of dispatching tokens to their assigned specialists.
    """
    def __init__(self, config):
        super().__init__()
        self.router = Router(
            embed_dim=config.model.embed_dim,
            num_experts=config.model.moe.num_experts,
            top_k=config.model.moe.top_k,
        )
        
        # Instantiate the full list of heterogeneous experts as per the design
        # Inside MoEFeedForward.__init__ in models/moe_vit.py

        self.experts = nn.ModuleList([
            MotionExpert(config.model.embed_dim, **config.model.experts.motion),         # ID 0
            TextureExpert(config.model.embed_dim, **config.model.experts.texture),       # ID 1
            GenericExpert(config.model.embed_dim),                                       # ID 2 ("Background")
            QA_AlignedExpert(config.model.embed_dim),                                    # ID 3
            FastChangeExpert(config.model.embed_dim, **config.model.experts.fast_change),# ID 4
            GenericExpert(config.model.embed_dim),                                       # ID 5
            GenericExpert(config.model.embed_dim),                                       # ID 6
            GenericExpert(config.model.embed_dim),                                       # ID 7
        ])

        # Freeze the "Background" expert (ID 2) as per the architectural design
        for param in self.experts[2].parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, expert_kwargs: dict):
        batch_size, seq_len, embed_dim = x.shape
        x_flat = x.view(-1, embed_dim) # Flatten to [b*s, d] for efficient routing

        # 1. Get routing decisions from the state-of-the-art router
        top_k_indices, routing_weights, aux_loss = self.router(x_flat)
        
        # 2. Prepare the final output tensor and perform batched dispatch
        final_output = torch.zeros_like(x_flat)
        
        # Iterate over each expert to process tokens destined for it in a single batch
        for i, expert in enumerate(self.experts):
            # Find the flattened indices of tokens that have chosen this expert `i`
            token_indices, which_top_k_slot = (top_k_indices == i).nonzero(as_tuple=True)
            
            if token_indices.numel() > 0:
                # Retrieve the gating weights corresponding to these specific tokens/slots
                routing_weights_for_expert = routing_weights[token_indices, which_top_k_slot]
                
                # Prepare the specific kwargs needed for this expert type
                current_expert_kwargs = {}
                expert_type = type(expert)

                if expert_type is MotionExpert:
                    flow_vecs = expert_kwargs['flow_vectors'].view(-1, expert_kwargs['flow_vectors'].shape[-1])
                    current_expert_kwargs['flow_vectors'] = flow_vecs[token_indices]
                elif expert_type is TextureExpert:
                    raw_patches = expert_kwargs['raw_patches'].view(-1, *expert_kwargs['raw_patches'].shape[2:])
                    current_expert_kwargs['raw_patches'] = raw_patches[token_indices]
                elif expert_type is QA_AlignedExpert:
                    current_expert_kwargs['avg_question_embedding'] = expert_kwargs['avg_question_embedding']
                elif expert_type is FastChangeExpert:
                    frame_deltas = expert_kwargs['frame_deltas'].view(-1, expert_kwargs['frame_deltas'].shape[-1])
                    current_expert_kwargs['frame_deltas'] = frame_deltas[token_indices]
                
                # Process the selected tokens through the specialized expert
                expert_output = expert(x_flat[token_indices], **current_expert_kwargs)
                
                # Apply gating weights and efficiently add back to the final output tensor
                weighted_output = expert_output * routing_weights_for_expert.unsqueeze(-1)
                final_output.index_add_(0, token_indices, weighted_output)
                
        return final_output.view(batch_size, seq_len, embed_dim), aux_loss

class TemporalMoEBlock(nn.Module):
    """
    A complete Transformer block for our model, containing the self-attention
    sub-layer and the MoE Feed-Forward sub-layer, using a Pre-LN structure.
    """
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadSelfAttention(
            embed_dim=config.model.embed_dim,
            num_heads=config.model.num_heads,
        )
        self.ffn = MoEFeedForward(config)
        self.norm = nn.LayerNorm(config.model.embed_dim)

    def forward(self, x: torch.Tensor, expert_kwargs: dict):
        # The attention block handles its own pre-normalization and residual connection
        x = self.attention(x)
        
        # The FFN block uses a residual connection around it
        ffn_input = self.norm(x)
        ffn_output, aux_loss = self.ffn(self.norm(x), expert_kwargs)
        x = x + ffn_output
        
        return x, aux_loss

class TemporalMoEViT(nn.Module):
    """
    The complete, end-to-end Temporal Mixture-of-Experts Vision Transformer.
    This model assembles all the custom modules into the final research artifact.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # --- Embedding Layers ---
        self.tokenizer = AutoTokenizer.from_pretrained(config.data.text_tokenizer)
        self.text_embedding = nn.Embedding(self.tokenizer.vocab_size, config.model.embed_dim, padding_idx=self.tokenizer.pad_token_id)
        patch_input_channels = config.model.frames_per_video * 3
        self.patch_embedding = nn.Conv2d(patch_input_channels, config.model.embed_dim, kernel_size=config.model.video_patch_size, stride=config.model.video_patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.model.embed_dim))
        self.positional_embedding = nn.Parameter(torch.randn(1, config.model.max_seq_len, config.model.embed_dim))
        self.dropout = nn.Dropout(0.1)

        # --- Transformer Body ---
        self.layers = nn.ModuleList([TemporalMoEBlock(config) for _ in range(config.model.num_layers)])
        
        # --- Prediction Head ---
        self.head = PredictionHead(config.model.embed_dim, config.model.num_answer_classes)

    def _embed(self, video_frames, question_ids):
        batch_size = video_frames.shape[0]
        # Embed video
        video_reshaped = rearrange(video_frames, 'b t c h w -> b (t c) h w')
        video_patches = self.patch_embedding(video_reshaped)
        video_tokens = rearrange(video_patches, 'b d ph pw -> b (ph pw) d')
        # Embed text
        text_tokens = self.text_embedding(question_ids)
        # Prepend CLS token and add positional embeddings
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, text_tokens, video_tokens), dim=1)
        x = x + self.positional_embedding[:, :x.size(1), :]
        return self.dropout(x), text_tokens

    def forward(self, batch: dict):
        # 1. Unpack all required data from the batch dictionary
        video_frames, question_ids = batch['video'], batch['question_ids']
        
        # 2. Convert inputs to a unified sequence of embedded tokens
        x, text_tokens = self._embed(video_frames, question_ids)

        # 3. Prepare the expert_kwargs dictionary to be passed through the layers
        expert_kwargs = {
            'raw_patches': batch.get('raw_patches'),
            'flow_vectors': batch.get('flow_vectors'),
            'frame_deltas': batch.get('frame_deltas'),
            'avg_question_embedding': torch.mean(text_tokens, dim=1, keepdim=True),
        }

        # 4. Pass sequence through the stack of MoE Transformer blocks
        total_aux_loss = 0.0
        for layer in self.layers:
            x, aux_loss_from_layer = layer(x, expert_kwargs)
            total_aux_loss += aux_loss_from_layer

        # 5. Use the final [CLS] token representation for prediction
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        
        # 6. Return logits for the main task loss and the averaged auxiliary loss
        return logits, total_aux_loss / len(self.layers)