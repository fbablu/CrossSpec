"""
Enhanced Loss Functions for Multi-Organ Cross-Species Training
Extends parent paper's approach with similarity-weighted analogous training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from multi_organ_config import ANALOGOUS_PAIRS, CLASS_WEIGHTS


class SimilarityWeightedFocalLoss(nn.Module):
    """Focal Loss with similarity weighting for analogous organ transfer"""

    def __init__(self, alpha=1, gamma=2, weight=None, size_average=True):
        super(SimilarityWeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, inputs, targets, similarity_weight=1.0):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # Apply similarity weighting
        focal_loss = focal_loss * similarity_weight

        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class DiceLoss(nn.Module):
    """Dice Loss from parent paper"""

    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), "predict & target shape do not match"
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class MultiOrganCrossSpeciesLoss(nn.Module):
    """Combined loss function for multi-organ cross-species training"""

    def __init__(self, mode="combined", organ_type="kidney", num_classes=7):
        super(MultiOrganCrossSpeciesLoss, self).__init__()
        self.mode = mode
        self.organ_type = organ_type
        self.num_classes = num_classes

        # Initialize loss components
        if mode == "separate":
            self.ce_loss = nn.CrossEntropyLoss(
                weight=torch.FloatTensor(CLASS_WEIGHTS[organ_type])
            )
            self.dice_loss = DiceLoss(2)  # Binary for separate training
        else:
            self.focal_loss = SimilarityWeightedFocalLoss(
                weight=torch.FloatTensor(CLASS_WEIGHTS["combined"])
            )
            self.dice_loss = DiceLoss(num_classes)

    def forward(self, inputs, targets, source_organ=None, target_organ=None):
        if self.mode == "separate":
            # Original parent paper approach for separate training
            ce_loss = self.ce_loss(inputs, targets)
            dice_loss = self.dice_loss(inputs, targets)
            return 0.5 * ce_loss + 0.75 * dice_loss

        elif self.mode == "homologous":
            # Same organ, different species (e.g., human kidney + mouse kidney)
            focal_loss = self.focal_loss(inputs, targets)
            dice_loss = self.dice_loss(inputs, targets)
            return 0.75 * focal_loss + 1.0 * dice_loss

        elif self.mode == "analogous":
            # Different organs with similarity weighting
            similarity_weight = self._get_similarity_weight(source_organ, target_organ)
            focal_loss = self.focal_loss(inputs, targets, similarity_weight)
            dice_loss = self.dice_loss(inputs, targets)
            return 0.75 * focal_loss + 1.0 * dice_loss + 0.5 * similarity_weight

        elif self.mode == "combined":
            # Full combined training (original + analogous)
            focal_loss = self.focal_loss(inputs, targets)
            dice_loss = self.dice_loss(inputs, targets)
            return 0.75 * focal_loss + 1.0 * dice_loss

    def _get_similarity_weight(self, source_organ, target_organ):
        """Get similarity weight from PCA analysis results"""
        organ_pair = (source_organ, target_organ)
        for pair_info in ANALOGOUS_PAIRS:
            if (pair_info[0], pair_info[1]) == organ_pair:
                return pair_info[2]  # Similarity score
        return 0.5  # Default weight for unknown pairs
