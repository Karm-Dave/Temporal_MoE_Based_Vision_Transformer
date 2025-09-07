import torch
import torch.nn as nn
from torch.nn import functional as F

class Router(nn.Module):
    def __init__(self,
                embed_dim: int,
                num_experts: int,
                top_k: int,
                z_loss_weight: float = 0.001,
                aux_loss_weight: float = 0.01):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Hyperparameters for the auxiliary losses
        self.z_loss_weight = z_loss_weight
        self.aux_loss_weight = aux_loss_weight

        # The main learnable layer
        self.gate_layer = nn.Linear(self.embed_dim, self.num_experts, bias = False)

        # Learnable parameter for gating noise
        # We multiply a random normal distribution by softplus(w_noise) to get learnable noise
        self.w_noise = nn.Parameter(torch.randn(embed_dim, num_experts) / 100)

    def forward(self, x: torch.Tensor):

        # Ensure we are working with a flattened sequence of tokens
        # Shape: [num_tokens, embed_dim], where num_tokens = batch_size * seq_len
        nums_tokens, _ = x.shape

        # 1. GET gate LOGITS & ADD NOISE (for training stability)
        # Shape: [num_tokens, num_experts]
        gate_logits = self.gate_layer(x)

        if self.training:
            # Add noise to the gate logits during training
            noise = (torch.randn_like(x) @ self.w_noise)
            gate_logits += noise

        # 2. CALCULATE Z-LOSS (for training stability)
        logsumexp_logits = torch.logsumexp(gate_logits, dim=1, keepdim=True)
        z_loss = (logsumexp_logits ** 2).mean() * self.z_loss_weight

        # 3. GET TOP-K EXPERTS
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1, sorted=False)
        
        # 4. CALCULATE LOAD BALANCING LOSS
        gate_probs = F.softmax(gate_logits, dim=-1, dtype=torch.float32)


        P = gate_probs.mean(dim=0)

        dispatch_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).float()
        D = dispatch_mask.sum(dim=(0, 1)) / nums_tokens

        load_balancing_loss = self.aux_loss_weight * (P -D).sum() * self.aux_loss_weight

        routing_weights = F.softmax(top_k_logits, dim=1, dtype=torch.float32).type_as(x)

        total_aux_loss = z_loss + load_balancing_loss

        return top_k_indices, routing_weights, total_aux_loss