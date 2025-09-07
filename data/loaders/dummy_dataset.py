import torch
from torch.utils.data import Dataset

class DummyVideoTextDataset(Dataset):
    """
    A dummy dataset that generates random tensors with the correct shapes and keys.
    This is essential for debugging the entire model pipeline without needing real data.
    """
    def __init__(self, config, num_samples=128):
        self.config = config
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # All shapes are derived from the master config file
        cfg_model = self.config.model
        b, t, c, h, w = 1, cfg_model.frames_per_video, 3, 224, 224
        
        # Simulate all the data your model expects in its batch dictionary
        batch = {
            'video': torch.randn(t, c, h, w),
            'question_ids': torch.randint(0, self.config.tokenizer_vocab_size, (cfg_model.max_seq_len,)),
            
            # --- Specialized Expert Data ---
            'raw_patches': torch.randn(196, c, cfg_model.video_patch_size, cfg_model.video_patch_size), # 14*14 patches
            'flow_vectors': torch.randn(196, cfg_model.experts.motion.flow_dim),
            'frame_deltas': torch.randn(196, cfg_model.experts.fast_change.delta_dim),
            
            # Use a random integer for the label
            'answer_label': torch.randint(0, cfg_model.num_answer_classes, (1,)).squeeze()
        }
        return batch