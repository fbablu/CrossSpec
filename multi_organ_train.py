"""
Multi-Organ Cross-Species Training Script
Extends parent paper's training approach for multiple organs and species
"""

import os
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse

# Import simplified models
from simple_models import (
    SimpleUNet as Unet,
    SimplePSPNet as PSPNet,
    SimpleDeepLab as DeepLab,
)

# Import new multi-organ components
from multi_organ_config import *
from enhanced_loss_functions import MultiOrganCrossSpeciesLoss
from multi_organ_dataloader import MultiOrganDataset, multi_organ_collate_fn


class MultiOrganTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model based on architecture choice
        self.model = self._create_model()
        self.model.to(self.device)

        # Initialize loss function
        self.criterion = MultiOrganCrossSpeciesLoss(
            mode=args.training_mode,
            organ_type=args.organ_type,
            num_classes=args.num_classes,
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Create data loaders
        self.train_loader, self.val_loader = self._create_data_loaders()

        # Setup logging
        self.save_dir = f"logs/{args.training_mode}_{args.organ_type}_{args.model_type}"
        os.makedirs(self.save_dir, exist_ok=True)

    def _create_model(self):
        """Create model based on architecture choice"""
        if self.args.model_type == "unet":
            return Unet(num_classes=self.args.num_classes)
        elif self.args.model_type == "pspnet":
            return PSPNet(num_classes=self.args.num_classes)
        elif self.args.model_type == "deeplabv3":
            return DeepLab(num_classes=self.args.num_classes)
        else:
            raise ValueError(f"Unknown model type: {self.args.model_type}")

    def _create_optimizer(self):
        """Create optimizer with different settings for different training modes"""
        if self.args.training_mode == "separate":
            # Higher learning rate for separate training
            lr = 1e-3
        else:
            # Lower learning rate for cross-species training
            lr = 5e-4

        return optim.SGD(
            self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
        )

    def _create_data_loaders(self):
        """Create training and validation data loaders"""

        # Determine organs to include based on training mode
        if self.args.training_mode == "separate":
            organs = [self.args.organ_type]
        elif self.args.training_mode == "homologous":
            organs = [self.args.organ_type]  # Same organ, both species
        else:
            organs = ["kidney", "liver", "spleen"]  # All organs

        # Training dataset
        train_dataset = MultiOrganDataset(
            mode=self.args.training_mode,
            organs=organs,
            input_shape=[self.args.input_size, self.args.input_size],
            train=True,
        )

        # Validation dataset
        val_dataset = MultiOrganDataset(
            mode=self.args.training_mode,
            organs=organs,
            input_shape=[self.args.input_size, self.args.input_size],
            train=False,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=multi_organ_collate_fn,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=multi_organ_collate_fn,
        )

        return train_loader, val_loader

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["images"].to(self.device)
            labels = batch["labels"].to(self.device)
            organs = batch["organs"]
            species = batch["species"]

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)

            # Compute loss based on training mode
            if self.args.training_mode == "analogous":
                # Use organ information for analogous training
                source_organ = organs[0] if len(set(organs)) == 1 else None
                target_organ = organs[0] if len(set(organs)) == 1 else None
                loss = self.criterion(outputs, labels, source_organ, target_organ)
            else:
                loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch} - Average Training Loss: {avg_loss:.4f}")
        return avg_loss

    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["images"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch} - Average Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def train(self):
        """Main training loop"""
        print(f"Starting {self.args.training_mode} training for {self.args.organ_type}")
        print(f"Model: {self.args.model_type}, Epochs: {self.args.epochs}")

        best_val_loss = float("inf")

        for epoch in range(self.args.epochs):
            # Training
            train_loss = self.train_epoch(epoch)

            # Validation
            val_loss = self.validate_epoch(epoch)

            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, train_loss, val_loss, is_best=True)

            # Save regular checkpoint every 5 epochs
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, train_loss, val_loss, is_best=False)

    def save_checkpoint(self, epoch, train_loss, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "args": self.args,
        }

        if is_best:
            filename = os.path.join(self.save_dir, "best_model.pth")
        else:
            filename = os.path.join(self.save_dir, f"epoch_{epoch:03d}.pth")

        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Organ Cross-Species Training")

    # Training configuration
    parser.add_argument(
        "--training_mode",
        type=str,
        default="homologous",
        choices=["separate", "homologous", "analogous", "combined"],
    )
    parser.add_argument(
        "--organ_type",
        type=str,
        default="kidney",
        choices=["kidney", "liver", "spleen"],
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="unet",
        choices=["unet", "pspnet", "deeplabv3"],
    )

    # Model parameters
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--input_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)

    # System parameters
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    # Adjust num_classes based on training mode
    if args.training_mode == "separate":
        args.num_classes = 2  # background + organ
    else:
        args.num_classes = 7  # 6 organ-species combinations + background

    # Initialize trainer and start training
    trainer = MultiOrganTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
