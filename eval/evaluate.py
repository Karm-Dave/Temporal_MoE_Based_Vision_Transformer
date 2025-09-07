# eval/evaluate.py
import yaml
import torch
from argparse import ArgumentParser
from dotmap import DotMap
from data.dataset_factory import get_and_split_data, create_dataloader
from models.base_vit import BaseVit
from models.moe_vit import TemporalMoEViT
from train.trainer import Trainer

if __name__ == '__main__':
    parser = ArgumentParser(description="Evaluate a trained model checkpoint.")
    parser.add_argument('--config', type=str, default='config/training_msvd.yaml')
    parser.add_argument('--model_type', type=str, required=True, choices=['moe', 'base'])
    parser.add_argument('--checkpoint_path', type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f: config = DotMap(yaml.safe_load(f))
    
    # --- THE FINAL, VICTORIOUS FIX ---
    # 1. Get all data splits and the final, correct vocab FIRST.
    _, val_meta, _, vocab = get_and_split_data(config)
    
    # 2. Create the dataloader. The config will be updated.
    val_loader = create_dataloader(config, 'val', val_meta, vocab)
    
    # 3. ONLY NOW do we initialize the model.
    print(f"--- Initializing {args.model_type.upper()} model for evaluation ---")
    if args.model_type == 'base': model = BaseVit(config)
    else: model = TemporalMoEViT(config)
    
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=config.training.device))
    
    # 4. Launch the evaluation.
    evaluator = Trainer(config, model, optimizer=None, train_loader=None, val_loader=val_loader)
    print("\n--- Running Final Evaluation on Validation Set ---")
    _, val_accuracy = evaluator._run_epoch(epoch=0, is_training=False)
    
    print("\n" + "="*50)
    print("--- EVALUATION COMPLETE ---")
    print(f"Model Type:    {args.model_type.upper()}")
    print(f"Checkpoint:    {args.checkpoint_path}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print("="*50)