import torch
import torch.nn as nn
from .losses import calculate_total_loss

class Trainer:
    """
    The Trainer class encapsulates the entire training and validation loop,
    making the main training script clean and simple.
    """
    def __init__(self, config, model, optimizer, train_loader, val_loader):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.training.device

    def _run_epoch(self, epoch, is_training):
        """Helper function to run a single epoch of training or validation."""
        # Set the model to the correct mode (train or eval)
        self.model.train(is_training)
        
        # Determine which data loader to use
        dataloader = self.train_loader if is_training else self.val_loader
        
        # Initialize trackers for metrics
        total_task_loss = 0
        correct_predictions = 0
        total_samples = 0

        # Loop over the data
        for i, batch in enumerate(dataloader):
            # Move all tensors in the batch dictionary to the configured device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Gradients are only enabled during training
            with torch.set_grad_enabled(is_training):
                # --- FORWARD PASS ---
                model_output = self.model(batch)
                
                # The BaseViT returns only logits. The MoE-ViT returns (logits, aux_loss).
                # This logic handles both cases gracefully.
                if isinstance(model_output, tuple):
                    logits, aux_loss = model_output
                else:
                    logits, aux_loss = model_output, 0.0

                # --- LOSS CALCULATION ---
                labels = batch['answer_label']
                total_loss_batch, task_loss_batch = calculate_total_loss(
                    logits, labels, aux_loss, self.config.training.loss_alpha
                )

            # --- BACKWARD PASS & OPTIMIZATION ---
            if is_training:
                self.optimizer.zero_grad()
                total_loss_batch.backward()
                self.optimizer.step()

            # --- METRIC CALCULATION ---
            total_task_loss += task_loss_batch.item()
            preds = torch.argmax(logits, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

        # Calculate average metrics for the epoch
        avg_loss = total_task_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        return avg_loss, accuracy

    def train(self): # <<< CORRECTION IS HERE
        """
        The main training loop. This method no longer takes any arguments.
        It retrieves the number of epochs directly from the config object.
        """
        self.model.to(self.device)
        print(f"Starting training on {self.device}...")
        
        for epoch in range(self.config.training.epochs):
            train_loss, train_acc = self._run_epoch(epoch, is_training=True)
            
            mode = "Train"
            print(f"Epoch {epoch+1:02d}/{self.config.training.epochs:02d} | [{mode}] Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

            # Optionally, run a validation loop if a validation loader is provided
            if self.val_loader:
                val_loss, val_acc = self._run_epoch(epoch, is_training=False)
                mode = "Val"
                print(f"Epoch {epoch+1:02d}/{self.config.training.epochs:02d} | [{mode}]  Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")