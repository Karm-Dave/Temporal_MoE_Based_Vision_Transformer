# temporal-moe-vit/models/base_vit.py

import torch
import torch.nn as nn
from einops import rearrange

# CORRECTED: Name now matches the one in train_dummy.py (BaseVit)
# The class StandardTransformerBlock also needs to be corrected to take the right argument.
class StandardTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim_multiply: int = 4, dropout: float = 0.1):
        super().__init__()
        ffn_dim = embed_dim * ffn_dim_multiply

        self.norm1 = nn.LayerNorm(embed_dim)
        # CORRECTION: Pass the correctly spelled 'embed_dim' argument.
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_output
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output
        return x

# Renamed to BaseVit to exactly match the call in train_dummy.py
class BaseVit(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # --- MODULE 1: EMBEDDINGS ---
        # CORRECTION: Using the corrected attribute name 'embed_dim'
        self.text_embedding = nn.Embedding(
            config.tokenizer_vocab_size,
            config.model.embed_dim,
            padding_idx=config.pad_token_id
        )

        patch_input_channels = config.model.frames_per_video * 3
        self.patch_embedding = nn.Conv2d(
            in_channels=patch_input_channels,
            out_channels=config.model.embed_dim,
            kernel_size=config.model.video_patch_size,
            stride=config.model.video_patch_size
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.model.embed_dim))
        self.positional_embedding = nn.Parameter(torch.randn(1, config.model.max_seq_len, config.model.embed_dim))
        self.dropout = nn.Dropout(0.1)

        # --- MODULE 2: TRANSFORMER BODY ---
        self.layers = nn.ModuleList(
            [StandardTransformerBlock(
                embed_dim=config.model.embed_dim,
                num_heads=config.model.num_heads,
             ) for _ in range(config.model.num_layers)]
        )

        # --- MODULE 3: PREDICTION HEAD ---
        self.head_norm = nn.LayerNorm(config.model.embed_dim)
        self.prediction_head = nn.Linear(config.model.embed_dim, config.model.num_answer_classes, bias=False)

    def _embed(self, video_frames, question_ids):
        batch_size = video_frames.shape[0]
        video_reshaped = rearrange(video_frames, 'b t c h w -> b (t c) h w')
        video_patches = self.patch_embedding(video_reshaped)
        video_tokens = rearrange(video_patches, 'b d ph pw -> b (ph pw) d')
        text_tokens = self.text_embedding(question_ids)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, text_tokens, video_tokens), dim=1)
        x = x + self.positional_embedding[:, :x.size(1), :]
        return self.dropout(x)
    
    def forward(self, batch: dict):
        video_frames = batch['video']
        question_ids = batch['question_ids']
        x = self._embed(video_frames, question_ids)
        for layer in self.layers:
            x = layer(x)
        cls_output = x[:, 0]
        logits = self.prediction_head(self.head_norm(cls_output))
        return logits