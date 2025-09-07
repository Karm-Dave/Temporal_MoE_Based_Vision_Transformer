# temporal-moe-vit/train/losses.py

import torch
import torch.nn as nn

def calculate_total_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    auxiliary_loss: torch.Tensor,
    alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the total loss for training the Temporal-MoE-ViT model.
    """
    # 1. Calculate the primary Task Loss (Cross-Entropy)
    task_loss_fn = nn.CrossEntropyLoss()
    task_loss = task_loss_fn(logits, labels)

    # 2. Combine with the Auxiliary Loss
    total_loss = task_loss + alpha * auxiliary_loss

    # 3. --- THIS IS THE CRITICAL FIX ---
    # The previous version was likely missing this line, causing it to return None.
    return total_loss, task_loss