# train/train.py
import yaml
import torch
from argparse import ArgumentParser
from dotmap import DotMap
from data.dataset_factory import get_and_split_data, create_dataloader
from models.base_vit import BaseVit
from models.moe_vit import TemporalMoEViT
from train.trainer import Trainer

if __name__ == '__main__':
    parser = ArgumentParser(description="Train a model on the MSVD dataset.")
    parser.add_argument('--config', type=str, default='config/training_msvd.yaml')
    parser.add_argument('--model_type', type=str, required=True, choices=['moe', 'base'])
    args = parser.parse_args()
    with open(args.config) as f: config = DotMap(yaml.safe_load(f))
    
    # --- THE FINAL, VICTORIOUS FIX for the TypeError ---
    # 1. We get all data splits and the final, correct vocab FIRST.
    print(f"\n--- Loading and Splitting Data for '{args.model_type.upper()}' ---")
    train_meta, val_meta, _, vocab = get_and_split_data(config)
    
    # 2. Now we create the dataloaders. The config will be updated inside them.
    train_loader = create_dataloader(config, 'train', train_meta, vocab)
    val_loader = create_dataloader(config, 'val', val_meta, vocab)

    # 3. ONLY NOW, with the config fully correct, do we initialize the model.
    print(f"\n--- Initializing {args.model_type.upper()} model ---")
    if args.model_type == 'base': model = BaseVit(config)
    else: model = TemporalMoEViT(config)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)

    # 4. Launch the trainer.
    trainer = Trainer(config, model, optimizer, train_loader, val_loader)
    trainer.train()
    print(f"\n--- Training for {args.model_type.upper()} finished successfully. ---")