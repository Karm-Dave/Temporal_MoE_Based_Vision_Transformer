# train_dummy.py
# A single, self-contained script to test the full training pipeline,
# evaluate both models, and print a final comparison summary.

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- Step 1: Import all the modules YOU have built ---
from models.base_vit import BaseVit
from models.moe_vit_karm import TemporalMoEViT, MoEFeedForward
from train.trainer import Trainer
from train.losses import calculate_total_loss

# ----------------------------------------------------------------------------------
# --- Step 2: Hardcoded Configuration (Replaces the need for .yaml files) ---
# ----------------------------------------------------------------------------------
class DummyConfig:
    """A simple class to hold all hyperparameters."""
    def __init__(self):
        # Base Model Hyperparameters
        self.model = type('model', (), {})()
        self.model.embed_dim = 768
        self.model.num_layers = 4
        self.model.num_heads = 8
        self.model.video_patch_size = 16
        self.model.num_answer_classes = 50
        self.model.frames_per_video = 8
        
        self.model.text_seq_len = 128
        num_video_patches = (224 // self.model.video_patch_size) ** 2
        self.model.max_seq_len = 1 + self.model.text_seq_len + num_video_patches # 1 + 128 + 196 = 325

        # MoE Specific Hyperparameters
        self.model.moe = type('moe', (), {})()
        self.model.moe.num_experts = 8
        self.model.moe.top_k = 2
        self.model.moe.experts = type('experts', (), {})()
        self.model.moe.experts.motion = {'flow_dim': 64}
        self.model.moe.experts.fast_change = {'delta_dim': 32}
        self.model.moe.experts.texture = {'channels': 3}

        # Data & Training Hyperparameters
        self.data = type('data', (), {})()
        self.data.batch_size = 16 # Slightly larger batch for stability
        self.data.num_workers = 0

        self.training = type('training', (), {})()
        self.training.epochs = 1 # More epochs to see clear overfitting
        # CORRECTED: A more stable learning rate for small dummy data
        self.training.learning_rate = 3.0e-4
        self.training.weight_decay = 0.01
        self.training.loss_alpha = 0.01
        self.training.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer_vocab_size = 30522
        self.pad_token_id = 0

# ----------------------------------------------------------------------------------
# --- Step 3: Dummy Data Generation (Replaces the entire data/ directory) ---
# ----------------------------------------------------------------------------------
class DummyDataset(Dataset):
    def __init__(self, config, num_samples=256):
        self.config = config
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        cfg = self.config.model
        num_patches = (224 // cfg.video_patch_size) ** 2

        # --- THIS IS THE KEY FIX for the IndexError ---
        # The specialized tensors must be padded to the full sequence length,
        # so even text tokens have a corresponding (zero) vector.
        full_seq_len = cfg.max_seq_len
        
        # Create full-size tensors of zeros
        flow_full = torch.zeros(full_seq_len, cfg.moe.experts.motion['flow_dim'])
        deltas_full = torch.zeros(full_seq_len, cfg.moe.experts.fast_change['delta_dim'])
        patches_full = torch.zeros(full_seq_len, cfg.moe.experts.texture['channels'], cfg.video_patch_size, cfg.video_patch_size)
        
        # Fill in the parts corresponding to the video patches with random data
        video_start_index = 1 + cfg.text_seq_len
        video_end_index = video_start_index + num_patches
        
        flow_full[video_start_index:video_end_index] = torch.randn(num_patches, cfg.moe.experts.motion['flow_dim'])
        deltas_full[video_start_index:video_end_index] = torch.randn(num_patches, cfg.moe.experts.fast_change['delta_dim'])
        patches_full[video_start_index:video_end_index] = torch.randn(num_patches, cfg.moe.experts.texture['channels'], cfg.video_patch_size, cfg.video_patch_size)
        
        batch = {
            'video': torch.randn(cfg.frames_per_video, 3, 224, 224),
            'question_ids': torch.randint(0, self.config.tokenizer_vocab_size, (cfg.text_seq_len,)),
            'answer_label': torch.randint(0, cfg.num_answer_classes, (1,)).squeeze(),
            'raw_patches': patches_full,
            'flow_vectors': flow_full,
            'frame_deltas': deltas_full,
        }
        return batch

# ----------------------------------------------------------------------------------
# --- Step 4: Final Evaluation and Comparison Logic ---
# ----------------------------------------------------------------------------------
def get_parameter_count(model):
    """Calculates the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_active_params_moe(model):
    """
    Calculates the active parameters (a proxy for FLOPs) for our MoE model.
    It sums the non-expert params and the params of `top_k` experts.
    """
    total_non_expert_params = 0
    total_one_expert_params = 0

    for name, module in model.named_modules():
        if isinstance(module, MoEFeedForward):
            # The router is part of the non-expert cost
            total_non_expert_params += get_parameter_count(module.router)
            # We assume all experts have the same size for this calculation
            total_one_expert_params = get_parameter_count(module.experts[0])
        elif 'experts' not in name:
            total_non_expert_params += get_parameter_count(module)

    # Active params = (all non-expert params) + (top_k * size_of_one_expert) * num_layers
    active_params = total_non_expert_params + (model.config.model.moe.top_k * total_one_expert_params)
    return active_params


def evaluate_model(model, dataloader, device):
    """Runs a final evaluation loop to get accuracy."""
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['answer_label']
            
            output = model(batch)
            logits = output[0] if isinstance(output, tuple) else output
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# ----------------------------------------------------------------------------------
# --- Step 5: Main Execution Block ---
# ----------------------------------------------------------------------------------
if __name__ == '__main__':
    # A. Setup
    config = DummyConfig()
    dummy_train_loader = DataLoader(DummyDataset(config), batch_size=config.data.batch_size)
    dummy_test_loader = DataLoader(DummyDataset(config, num_samples=128), batch_size=config.data.batch_size) # A separate "test" set

    # B. --- Train the Baseline Model ---
    print("\n" + "="*50)
    print("--- 1. TRAINING BASELINE (BaseVit) ---")
    print("="*50)
    base_model = BaseVit(config)
    base_optimizer = torch.optim.AdamW(base_model.parameters(), lr=config.training.learning_rate)
    base_trainer = Trainer(config, base_model, base_optimizer, dummy_train_loader, val_loader=None)
    base_trainer.train()

    # C. --- Train the MoE Model ---
    print("\n" + "="*50)
    print("--- 2. TRAINING OUR MODEL (TemporalMoEViT) ---")
    print("="*50)
    moe_model = TemporalMoEViT(config)
    moe_optimizer = torch.optim.AdamW(moe_model.parameters(), lr=config.training.learning_rate)
    moe_trainer = Trainer(config, moe_model, moe_optimizer, dummy_train_loader, val_loader=None)
    moe_trainer.train()

    # D. --- Final Comparison ---
    print("\n" + "="*60)
    print("--- 3. FINAL EVALUATION & COMPARISON SUMMARY ---")
    print("="*60)

    # Evaluate both models on the "test" dummy data
    base_accuracy = evaluate_model(base_model, dummy_test_loader, config.training.device)
    moe_accuracy = evaluate_model(moe_model, dummy_test_loader, config.training.device)

    # Get parameter counts
    base_params = get_parameter_count(base_model)
    moe_total_params = get_parameter_count(moe_model)
    moe_active_params = get_active_params_moe(moe_model)
    
    print(f"\n{'Metric':<25} | {'BaseViT':<15} | {'TemporalMoEViT (Ours)':<25}")
    print("-"*70)
    print(f"{'Final Test Accuracy':<25} | {base_accuracy:<15.4f} | {moe_accuracy:<25.4f}")
    print(f"{'Total Trainable Params':<25} | {base_params/1e6:<15.2f}M | {moe_total_params/1e6:<25.2f}M")
    print(f"{'Active Params (Compute)':<25} | {base_params/1e6:<15.2f}M | {moe_active_params/1e6:<25.2f}M")
    print("-"*70)
    
    print("\n--- Dummy data workflow test complete. ---")