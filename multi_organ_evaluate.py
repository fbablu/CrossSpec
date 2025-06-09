"""
Multi-Organ Cross-Species Evaluation Script
Evaluate performance across different training modes and organ types
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import argparse

from multi_organ_config import *
from multi_organ_dataloader import MultiOrganDataset, multi_organ_collate_fn
from simple_models import (
    SimpleUNet as Unet,
    SimplePSPNet as PSPNet,
    SimpleDeepLab as DeepLab,
)


class MultiOrganEvaluator:
    def __init__(self, model_path, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()

        # Create test data loader
        self.test_loader = self._create_test_loader()

        # Results storage
        self.results = {
            "organ_type": [],
            "species": [],
            "training_mode": [],
            "iou": [],
            "dice": [],
            "pixel_accuracy": [],
        }

    def _load_model(self, model_path):
        """Load trained model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)

        # Create model based on saved args
        saved_args = checkpoint["args"]
        if saved_args.model_type == "unet":
            model = Unet(
                num_classes=saved_args.num_classes,
                pretrained=False,
                backbone="resnet50",
            )
        elif saved_args.model_type == "pspnet":
            model = PSPNet(
                num_classes=saved_args.num_classes,
                backbone="resnet50",
                downsample_factor=8,
            )
        elif saved_args.model_type == "deeplabv3":
            model = DeepLab(
                num_classes=saved_args.num_classes,
                backbone="mobilenet",
                downsample_factor=8,
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        return model

    def _create_test_loader(self):
        """Create test data loader"""
        test_dataset = MultiOrganDataset(
            mode=self.args.training_mode,
            organs=["kidney", "liver", "spleen"],
            input_shape=[self.args.input_size, self.args.input_size],
            train=False,
        )

        return DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=multi_organ_collate_fn,
        )

    def calculate_metrics(self, pred, target, num_classes):
        """Calculate IoU, Dice, and Pixel Accuracy"""
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()

        ious = []
        dices = []

        for c in range(num_classes):
            pred_c = pred == c
            target_c = target == c

            intersection = np.logical_and(pred_c, target_c).sum()
            union = np.logical_or(pred_c, target_c).sum()

            if union > 0:
                iou = intersection / union
                dice = 2 * intersection / (pred_c.sum() + target_c.sum())
            else:
                iou = 1.0 if intersection == 0 else 0.0
                dice = 1.0 if intersection == 0 else 0.0

            ious.append(iou)
            dices.append(dice)

        pixel_acc = (pred == target).mean()

        return np.mean(ious), np.mean(dices), pixel_acc

    def evaluate(self):
        """Run evaluation on test set"""
        print(f"Evaluating {self.args.training_mode} model...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                images = batch["images"].to(self.device)
                labels = batch["labels"].to(self.device)
                organs = batch["organs"]
                species = batch["species"]

                # Forward pass
                outputs = self.model(images)
                predictions = torch.argmax(F.softmax(outputs, dim=1), dim=1)

                # Calculate metrics for each sample in batch
                for i in range(len(organs)):
                    pred = predictions[i]
                    target = labels[i]

                    iou, dice, pixel_acc = self.calculate_metrics(
                        pred, target, self.args.num_classes
                    )

                    # Store results
                    self.results["organ_type"].append(organs[i])
                    self.results["species"].append(species[i])
                    self.results["training_mode"].append(self.args.training_mode)
                    self.results["iou"].append(iou)
                    self.results["dice"].append(dice)
                    self.results["pixel_accuracy"].append(pixel_acc)

                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx}/{len(self.test_loader)} batches")

        return self._summarize_results()

    def _summarize_results(self):
        """Summarize evaluation results"""
        df = pd.DataFrame(self.results)

        # Overall performance
        overall_stats = {
            "mean_iou": df["iou"].mean(),
            "mean_dice": df["dice"].mean(),
            "mean_pixel_acc": df["pixel_accuracy"].mean(),
            "std_iou": df["iou"].std(),
            "std_dice": df["dice"].std(),
            "std_pixel_acc": df["pixel_accuracy"].std(),
        }

        # Per-organ performance
        organ_stats = (
            df.groupby("organ_type")
            .agg(
                {
                    "iou": ["mean", "std"],
                    "dice": ["mean", "std"],
                    "pixel_accuracy": ["mean", "std"],
                }
            )
            .round(4)
        )

        # Per-species performance
        species_stats = (
            df.groupby("species")
            .agg(
                {
                    "iou": ["mean", "std"],
                    "dice": ["mean", "std"],
                    "pixel_accuracy": ["mean", "std"],
                }
            )
            .round(4)
        )

        # Print results
        print("\n" + "=" * 60)
        print(f"EVALUATION RESULTS - {self.args.training_mode.upper()} MODE")
        print("=" * 60)

        print(f"\nOverall Performance:")
        print(
            f"Mean IoU: {overall_stats['mean_iou']:.4f} ± {overall_stats['std_iou']:.4f}"
        )
        print(
            f"Mean Dice: {overall_stats['mean_dice']:.4f} ± {overall_stats['std_dice']:.4f}"
        )
        print(
            f"Mean Pixel Acc: {overall_stats['mean_pixel_acc']:.4f} ± {overall_stats['std_pixel_acc']:.4f}"
        )

        print(f"\nPer-Organ Performance:")
        print(organ_stats)

        print(f"\nPer-Species Performance:")
        print(species_stats)

        # Save detailed results
        results_dir = f"results_{self.args.training_mode}"
        os.makedirs(results_dir, exist_ok=True)
        df.to_csv(os.path.join(results_dir, "detailed_results.csv"), index=False)

        return overall_stats, organ_stats, species_stats


def compare_training_modes(model_paths):
    """Compare results across different training modes"""
    comparison_data = []

    for mode, path in model_paths.items():
        if os.path.exists(path):
            args = argparse.Namespace(
                training_mode=mode,
                input_size=1024,
                num_classes=7 if mode != "separate" else 2,
            )

            evaluator = MultiOrganEvaluator(path, args)
            overall_stats, _, _ = evaluator.evaluate()

            comparison_data.append(
                {
                    "training_mode": mode,
                    "mean_iou": overall_stats["mean_iou"],
                    "mean_dice": overall_stats["mean_dice"],
                    "mean_pixel_acc": overall_stats["mean_pixel_acc"],
                }
            )

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + "=" * 60)
    print("TRAINING MODE COMPARISON")
    print("=" * 60)
    print(comparison_df.round(4))

    # Save comparison
    comparison_df.to_csv("training_mode_comparison.csv", index=False)

    return comparison_df


def main():
    parser = argparse.ArgumentParser(description="Multi-Organ Cross-Species Evaluation")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--training_mode",
        type=str,
        default="homologous",
        choices=["separate", "homologous", "analogous", "combined"],
    )
    parser.add_argument("--input_size", type=int, default=1024)
    parser.add_argument(
        "--compare_modes",
        action="store_true",
        help="Compare all training modes (requires multiple model paths)",
    )

    args = parser.parse_args()

    if args.compare_modes:
        # Compare multiple training modes
        model_paths = {
            "separate": "logs/separate_kidney_unet/best_model.pth",
            "homologous": "logs/homologous_kidney_unet/best_model.pth",
            "analogous": "logs/analogous_kidney_unet/best_model.pth",
            "combined": "logs/combined_kidney_unet/best_model.pth",
        }
        compare_training_modes(model_paths)
    else:
        # Evaluate single model
        args.num_classes = 7 if args.training_mode != "separate" else 2
        evaluator = MultiOrganEvaluator(args.model_path, args)
        evaluator.evaluate()


if __name__ == "__main__":
    main()
