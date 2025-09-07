# temporal-moe-vit/models/experts.py

import torch
import torch.nn as nn

class Expert(nn.Module):
    """
    Abstract Base Class for all expert modules.
    
    CORRECTION: The __init__ method should only take `embed_dim`, as the output
    dimension is always the same as the input dimension in this architecture.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, **kwargs):
        """
        The forward pass must be implemented by all child classes.
        """
        raise NotImplementedError("Each expert must implement its own forward method.")

# --- Expert Implementations ---

class GenericExpert(Expert):
    """
    Expert ID: 5, 6, 7 (and 2, before freezing)
    A standard Feed-Forward Network (FFN) that acts as a general-purpose processor.
    """
    def __init__(self, embed_dim: int, ffn_dim_multiply: int = 4, dropout: float = 0.1):
        super().__init__(embed_dim) # This call now correctly matches Expert.__init__
        ffn_dim = embed_dim * ffn_dim_multiply
        
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, **kwargs):
        """
        Processes the input tokens through a standard FFN.
        kwargs are ignored as this expert requires no specialized input.
        """
        return self.net(x)

class MotionExpert(Expert):
    """
    Expert ID: 0
    A specialist expert that receives additional optical flow information.
    """
    def __init__(self, embed_dim: int, flow_dim: int = 64, ffn_dim_multiply: int = 4, dropout: float = 0.1):
        super().__init__(embed_dim)
        input_dim = embed_dim + flow_dim
        ffn_dim = embed_dim * ffn_dim_multiply
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, **kwargs):
        flow_vectors = kwargs['flow_vectors']
        combined_input = torch.cat([x, flow_vectors], dim=-1)
        return self.net(combined_input)

class TextureExpert(Expert):
    """
    Expert ID: 1
    A specialist that operates directly on raw image patches using a CNN stem.
    """
    def __init__(self, embed_dim: int, patch_size: int = 16, channels: int = 3):
        super().__init__(embed_dim)
        self.patch_size = patch_size
        self.cnn_stem = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        cnn_output_dim = 32 * patch_size * patch_size
        self.ffn = nn.Sequential(
            nn.Linear(cnn_output_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, x: torch.Tensor, **kwargs):
        raw_patches = kwargs['raw_patches']
        cnn_features = self.cnn_stem(raw_patches)
        cnn_features_flat = cnn_features.view(cnn_features.size(0), -1)
        return self.ffn(cnn_features_flat)

class QA_AlignedExpert(Expert):
    """
    Expert ID: 3
    A specialist that is explicitly conditioned on the user's question.
    """
    def __init__(self, embed_dim: int, ffn_dim_multiply: int = 4, dropout: float = 0.1):
        super().__init__(embed_dim)
        input_dim = embed_dim * 2
        ffn_dim = embed_dim * ffn_dim_multiply
        self.net = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, **kwargs):
        avg_question_embedding_for_expert = kwargs['avg_question_embedding_for_expert']
        combined_input = torch.cat([x, avg_question_embedding_for_expert], dim=-1)
        return self.net(combined_input)

class FastChangeExpert(Expert):
    """
    Expert ID: 4
    A specialist that receives the difference between video frames as an additional input.
    """
    def __init__(self, embed_dim: int, delta_dim: int = 32, ffn_dim_multiply: int = 4, dropout: float = 0.1):
        super().__init__(embed_dim)
        input_dim = embed_dim + delta_dim
        ffn_dim = embed_dim * ffn_dim_multiply
        self.net = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, **kwargs):
        frame_deltas = kwargs['frame_deltas']
        combined_input = torch.cat([x, frame_deltas], dim=-1)
        return self.net(combined_input)