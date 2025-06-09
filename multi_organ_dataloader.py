"""
Multi-Organ Cross-Species Data Loader
Extends parent paper's dataloader for multiple organs and training modes
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from multi_organ_config import (
    ORGAN_CLASSES,
    DATASET_PATHS,
    LABEL_PATHS,
    HOMOLOGOUS_PAIRS,
    ANALOGOUS_PAIRS,
)


class MultiOrganDataset(Dataset):
    """Dataset for multi-organ cross-species training"""

    def __init__(
        self,
        mode="combined",
        organs=["kidney"],
        input_shape=[1024, 1024],
        train=True,
        transform=None,
    ):
        self.mode = mode
        self.organs = organs
        self.input_shape = input_shape
        self.train = train
        self.transform = transform

        self.images = []
        self.labels = []
        self.organ_types = []
        self.species_types = []

        self._load_data()

    def _load_data(self):
        """Load images and labels based on training mode"""

        if self.mode == "separate":
            # Load single organ-species combination
            self._load_separate_data()
        elif self.mode == "homologous":
            # Load same organs across species
            self._load_homologous_data()
        elif self.mode == "analogous":
            # Load analogous organ pairs
            self._load_analogous_data()
        elif self.mode == "combined":
            # Load all available data
            self._load_combined_data()

    def _load_separate_data(self):
        """Load data for separate training (single organ-species)"""
        for organ in self.organs:
            for species in ["human", "mouse"]:
                organ_key = f"{species}_{organ}"
                if organ_key.replace("_", " ") in DATASET_PATHS:
                    path = DATASET_PATHS[organ_key.replace("_", " ")]
                    self._load_images_from_path(path, organ, species)

    def _load_homologous_data(self):
        """Load data for homologous training (same organ, different species)"""
        for organ in self.organs:
            human_key = f"human {organ}"
            mouse_key = f"mouse {organ}"

            if human_key in DATASET_PATHS:
                self._load_images_from_path(DATASET_PATHS[human_key], organ, "human")
            if mouse_key in DATASET_PATHS:
                self._load_images_from_path(DATASET_PATHS[mouse_key], organ, "mouse")

    def _load_analogous_data(self):
        """Load data for analogous training (different organs, high similarity)"""
        for pair_info in ANALOGOUS_PAIRS:
            source_organ_species = pair_info[0]  # e.g., 'human_liver'
            target_organ_species = pair_info[1]  # e.g., 'mouse_kidney'
            similarity = pair_info[2]

            # Only load high similarity pairs (>0.9)
            if similarity > 0.9:
                source_parts = source_organ_species.split("_")
                target_parts = target_organ_species.split("_")

                source_key = f"{source_parts[0]} {source_parts[1]}"
                target_key = f"{target_parts[0]} {target_parts[1]}"

                if source_key in DATASET_PATHS:
                    self._load_images_from_path(
                        DATASET_PATHS[source_key], source_parts[1], source_parts[0]
                    )
                if target_key in DATASET_PATHS:
                    self._load_images_from_path(
                        DATASET_PATHS[target_key], target_parts[1], target_parts[0]
                    )

    def _load_combined_data(self):
        """Load all available organ data"""
        for path_key, path in DATASET_PATHS.items():
            parts = path_key.split(" ")
            species = parts[0]
            organ = parts[1] if len(parts) > 1 else "unknown"
            self._load_images_from_path(path, organ, species)

    def _load_images_from_path(self, path, organ, species):
        """Load images from a specific path"""
        if not os.path.exists(path):
            print(f"Warning: Path does not exist: {path}")
            return

        # Get corresponding label path using space format (not underscore)
        organ_key = f"{species} {organ}"
        label_path = LABEL_PATHS.get(organ_key)

        image_extensions = (".png", ".jpg", ".jpeg", ".tiff", ".tif")

        for filename in os.listdir(path):
            if filename.lower().endswith(image_extensions):
                img_path = os.path.join(path, filename)

                # Find corresponding label file
                label_file = None
                if label_path and os.path.exists(label_path):
                    # Try different extensions for label
                    base_name = os.path.splitext(filename)[0]
                    for ext in [".tif", ".png", ".jpg"]:
                        potential_label = os.path.join(label_path, base_name + ext)
                        if os.path.exists(potential_label):
                            label_file = potential_label
                            break

                self.images.append(img_path)
                self.labels.append(label_file)
                self.organ_types.append(organ)
                self.species_types.append(species)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Load image
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        image = image.resize(self.input_shape)
        image = np.array(image, dtype=np.float32)

        # Load label
        label_path = self.labels[index]
        if label_path and os.path.exists(label_path):
            label = Image.open(label_path).convert("L")
            label = label.resize(self.input_shape)
            label = np.array(label, dtype=np.uint8)
        else:
            # Create dummy label for organs without segmentation masks
            label = np.zeros(self.input_shape, dtype=np.uint8)

        # Convert to class indices based on mode
        organ = self.organ_types[index]
        species = self.species_types[index]

        if self.mode == "separate":
            # Binary: background=0, organ=1
            label = (label > 0).astype(np.uint8)
        else:
            # Multi-class: convert to organ-species specific classes
            if organ in ORGAN_CLASSES and species in ORGAN_CLASSES[organ]:
                organ_class = ORGAN_CLASSES[organ][species]
                label = np.where(label > 0, organ_class, ORGAN_CLASSES["background"])

        # Apply transforms
        if self.transform:
            pass

        # Normalize image
        image = image / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array(
            [0.229, 0.224, 0.225]
        )
        image = image.transpose(2, 0, 1)  # HWC to CHW

        return {
            "image": torch.FloatTensor(image),
            "label": torch.LongTensor(label),
            "organ": organ,
            "species": species,
        }


def multi_organ_collate_fn(batch):
    """Custom collate function for multi-organ data"""
    images = torch.stack([item["image"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    organs = [item["organ"] for item in batch]
    species = [item["species"] for item in batch]

    return {"images": images, "labels": labels, "organs": organs, "species": species}
