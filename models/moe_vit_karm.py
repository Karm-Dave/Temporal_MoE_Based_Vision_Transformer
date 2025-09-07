# temporal-moe-vit/models/moe_vit.py

import torch
import torch.nn as nn
from einops import rearrange

# Importing all the custom modular components
from .experts import GenericExpert, MotionExpert, TextureExpert, QA_AlignedExpert, FastChangeExpert
from .router import Router
# The following imports will use your custom files. Ensure their names match.
from .attention_karm import MultiHeadSelfAttention
from .prediction_head_karm import PredictionHead

class MoEFeedForward(nn.Module):
    # ... (init method is correct, no changes needed)
    def __init__(self, config):
        super().__init__()
        self.router = Router(
            embed_dim=config.model.embed_dim,
            num_experts=config.model.moe.num_experts,
            top_k=config.model.moe.top_k,
        )
        self.experts = nn.ModuleList([
            MotionExpert(config.model.embed_dim, **config.model.moe.experts.motion),
            TextureExpert(config.model.embed_dim, patch_size=config.model.video_patch_size),
            GenericExpert(config.model.embed_dim),
            QA_AlignedExpert(config.model.embed_dim),
            FastChangeExpert(config.model.embed_dim, **config.model.moe.experts.fast_change),
            GenericExpert(config.model.embed_dim),
            GenericExpert(config.model.embed_dim),
            GenericExpert(config.model.embed_dim),
        ])
        for param in self.experts[2].parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, expert_kwargs: dict):
        batch_size, seq_len, embed_dim = x.shape
        x_flat = x.view(-1, embed_dim)
        top_k_indices, routing_weights, aux_loss = self.router(x_flat)
        final_output = torch.zeros_like(x_flat)
        
        for i, expert in enumerate(self.experts):
            token_indices, which_top_k_slot = (top_k_indices == i).nonzero(as_tuple=True)
            if token_indices.numel() > 0:
                routing_weights_for_expert = routing_weights[token_indices, which_top_k_slot]
                current_expert_kwargs = {}
                expert_type = type(expert)
                
                # --- THIS IS THE KEY FIX ---
                # We need to get the flattened, expanded question embedding from kwargs
                # and then select the right tokens from it.
                if expert_type is QA_AlignedExpert:
                    # 'avg_question_embedding_flat' is now expected in the kwargs
                    avg_q_flat = expert_kwargs['avg_question_embedding_flat']
                    current_expert_kwargs['avg_question_embedding_for_expert'] = avg_q_flat[token_indices]
                # Other experts still need their data indexed correctly from the flattened sequence.
                elif expert_type is MotionExpert and expert_kwargs.get('flow_vectors') is not None:
                    current_expert_kwargs['flow_vectors'] = expert_kwargs['flow_vectors'].view(-1, expert_kwargs['flow_vectors'].shape[-1])[token_indices]
                elif expert_type is TextureExpert and expert_kwargs.get('raw_patches') is not None:
                    current_expert_kwargs['raw_patches'] = expert_kwargs['raw_patches'].view(-1, *expert_kwargs['raw_patches'].shape[2:])[token_indices]
                elif expert_type is FastChangeExpert and expert_kwargs.get('frame_deltas') is not None:
                    current_expert_kwargs['frame_deltas'] = expert_kwargs['frame_deltas'].view(-1, expert_kwargs['frame_deltas'].shape[-1])[token_indices]
                
                expert_output = expert(x_flat[token_indices], **current_expert_kwargs)
                weighted_output = expert_output * routing_weights_for_expert.unsqueeze(-1)
                final_output.index_add_(0, token_indices, weighted_output)
                
        return final_output.view(batch_size, seq_len, embed_dim), aux_loss

class TemporalMoEBlock(nn.Module):
    # ... (init method is correct)
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config.model.embed_dim, config.model.num_heads)
        self.ffn = MoEFeedForward(config)
        self.norm = nn.LayerNorm(config.model.embed_dim)

    def forward(self, x: torch.Tensor, expert_kwargs: dict):
        x = self.attention(x)
        ffn_input = self.norm(x)
        ffn_output, aux_loss = self.ffn(ffn_input, expert_kwargs)
        x = x + ffn_output
        return x, aux_loss

class TemporalMoEViT(nn.Module):
    # ... (init method is correct, using the one from the last successful run)
    def __init__(self, config):
        super().__init__()
        self.config = config
        if not hasattr(config, 'tokenizer_vocab_size'):
             raise ValueError("Config must contain 'tokenizer_vocab_size'.")
        self.text_embedding = nn.Embedding(config.tokenizer_vocab_size, config.model.embed_dim, padding_idx=config.pad_token_id)
        patch_input_channels = config.model.frames_per_video * 3
        self.patch_embedding = nn.Conv2d(patch_input_channels, config.model.embed_dim, kernel_size=config.model.video_patch_size, stride=config.model.video_patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.model.embed_dim))
        self.positional_embedding = nn.Parameter(torch.randn(1, config.model.max_seq_len, config.model.embed_dim))
        self.dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList([TemporalMoEBlock(config) for _ in range(config.model.num_layers)])
        self.head = PredictionHead(config.model.embed_dim, config.model.num_answer_classes)

    def _embed(self, video_frames, question_ids):
        # ... (_embed method is correct)
        batch_size = video_frames.shape[0]
        video_reshaped = rearrange(video_frames, 'b t c h w -> b (t c) h w')
        video_patches = self.patch_embedding(video_reshaped)
        video_tokens = rearrange(video_patches, 'b d ph pw -> b (ph pw) d')
        text_tokens = self.text_embedding(question_ids)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, text_tokens, video_tokens), dim=1)
        x = x + self.positional_embedding[:, :x.size(1), :]
        return self.dropout(x), text_tokens

    def forward(self, batch: dict):
        video_frames, question_ids = batch['video'], batch['question_ids']
        x, text_tokens = self._embed(video_frames, question_ids)
        
        # --- THIS IS THE KEY FIX: PRE-EXPAND AND FLATTEN ---
        # 1. Calculate the average question embedding per sample. Shape: [b, 1, d]
        avg_q_embedding = torch.mean(text_tokens, dim=1, keepdim=True)
        # 2. Expand it to match the full sequence length. Shape: [b, s, d]
        avg_q_embedding_expanded = avg_q_embedding.expand(-1, x.shape[1], -1)

        expert_kwargs = {
            'raw_patches': batch.get('raw_patches'),
            'flow_vectors': batch.get('flow_vectors'),
            'frame_deltas': batch.get('frame_deltas'),
            # 3. Add the flattened version to kwargs. Shape: [b*s, d]
            'avg_question_embedding_flat': avg_q_embedding_expanded.reshape(-1, self.config.model.embed_dim)
        }

        total_aux_loss = 0.0
        for layer in self.layers:
            x, aux_loss_from_layer = layer(x, expert_kwargs)
            total_aux_loss += aux_loss_from_layer
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        return logits, total_aux_loss / len(self.layers)