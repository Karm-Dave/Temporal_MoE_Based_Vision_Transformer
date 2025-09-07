# temporal-moe-vit/models/attention.py

import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    """
    A complete Multi-Head Self-Attention block using the Pre-LN architecture.

    This module includes Layer Normalization, the core attention mechanism, and
    the residual connection. It encapsulates the "information mixing" part of a
    standard Transformer layer.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Initializes the self-attention block.

        Args:
            embed_dim (int): The embedding dimension of the model.
            num_heads (int): The number of parallel attention heads.
            dropout (float): The dropout rate for the attention mechanism.
        """
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # We use the [batch, seq, dim] format
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input sequence through self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, seq, dim].

        Returns:
            torch.Tensor: Output tensor of the same shape.
        """
        # Apply layer normalization before passing to the attention mechanism
        x_norm = self.norm(x)
        
        # The MHA layer returns the attention output and optionally the attention weights
        attn_output, _ = self.attention(query=x_norm, key=x_norm, value=x_norm, need_weights=False)
        
        # Apply the residual connection. This is crucial for training deep models.
        x = x + attn_output
        return x