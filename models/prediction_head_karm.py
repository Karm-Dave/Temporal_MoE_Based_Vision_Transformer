# temporal-moe-vit/models/prediction_head.py

import torch
import torch.nn as nn

class PredictionHead(nn.Module):
    """
    A modular prediction head for classification tasks.

    This module takes the final representation of the [CLS] token, applies layer
    normalization for stability, and then projects it to the number of answer classes
    to produce the final logits.
    """
    def __init__(self, embed_dim: int, num_classes: int):
        """
        Initializes the prediction head.

        Args:
            embed_dim (int): The embedding dimension of the model (e.g., 768).
            num_classes (int): The number of possible output classes (answers).
        """
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.linear_head = nn.Linear(embed_dim, num_classes, bias=False)

    def forward(self, x):
        """
        Performs the final classification projection.

        Args:
            x (torch.Tensor): The final hidden state of the [CLS] token,
                            of shape [batch_size, embed_dim].

        Returns:
            torch.Tensor: The output logits of shape [batch_size, num_classes].
        """
        # Apply layer normalization for a more stable input to the final layer.
        x_norm = self.norm(x)
        # Project to the final number of classes.
        logits = self.linear_head(x_norm)
        return logits