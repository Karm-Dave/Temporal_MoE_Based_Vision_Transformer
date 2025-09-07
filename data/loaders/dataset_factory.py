
from torch.utils.data import DataLoader
from .loaders.dummy_dataset import DummyVideoTextDataset
from .loaders.video_text_dataset import VideoTextDataset

def create_dataloader(config, split, use_dummy_data=False):
    if use_dummy_data:
        print(">>> Using DUMMY dataset for pipeline testing.")
        # We need the vocab size for the dummy dataset tokenizer
        from transformers import AutoTokenizer
        config.tokenizer_vocab_size = AutoTokenizer.from_pretrained(config.data.text_tokenizer).vocab_size
        
        dataset = DummyVideoTextDataset(config, num_samples=config.data.batch_size * 4)
    else:
        # This will be used for the real training run
        dataset = VideoTextDataset(config, split=split)
    
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        shuffle=(split == 'train')
    )