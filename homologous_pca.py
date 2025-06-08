"""
Homologous PCA Analysis
Analyzes same organ types across human and mouse species to demonstrate
structural similarity (like kidney-to-kidney comparison from parent paper).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image


class HomologousPCAAnalyzer:
    def __init__(self, data_root):
        self.data_root = data_root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pre-trained ResNet50 (same as parent paper)
        try:
            from torchvision.models import ResNet50_Weights

            self.feature_extractor = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        except ImportError:
            # Fallback for older torchvision versions
            self.feature_extractor = resnet50(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(
            *list(self.feature_extractor.children())[:-1]
        )
        self.feature_extractor.eval().to(self.device)

        # Image preprocessing
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Discover overlapping organs
        self.overlapping_organs = self._find_overlapping_organs()

    def _find_overlapping_organs(self):
        """Find organs that exist in both human and mouse datasets"""
        human_organs = set()
        mouse_organs = set()

        try:
            for folder in os.listdir(self.data_root):
                folder_path = os.path.join(self.data_root, folder)
                if os.path.isdir(folder_path):
                    if folder.startswith("human "):
                        organ = folder.replace("human ", "")
                        # Try different possible paths for tissue images
                        possible_paths = [
                            os.path.join(folder_path, "tissue images"),
                            os.path.join(folder_path, "images"),
                            folder_path,  # Sometimes images are directly in the organ folder
                        ]

                        for tissue_path in possible_paths:
                            if os.path.exists(tissue_path):
                                # Check for image files
                                image_files = [
                                    f
                                    for f in os.listdir(tissue_path)
                                    if f.lower().endswith(
                                        (".png", ".jpg", ".jpeg", ".tiff", ".tif")
                                    )
                                ]
                                if image_files:
                                    human_organs.add(organ)
                                    print(
                                        f"Found {len(image_files)} images in {tissue_path}"
                                    )
                                    break

                    elif folder.startswith("mouse "):
                        organ = folder.replace("mouse ", "")
                        # Try different possible paths for tissue images
                        possible_paths = [
                            os.path.join(folder_path, "tissue images"),
                            os.path.join(folder_path, "images"),
                            folder_path,  # Sometimes images are directly in the organ folder
                        ]

                        for tissue_path in possible_paths:
                            if os.path.exists(tissue_path):
                                # Check for image files
                                image_files = [
                                    f
                                    for f in os.listdir(tissue_path)
                                    if f.lower().endswith(
                                        (".png", ".jpg", ".jpeg", ".tiff", ".tif")
                                    )
                                ]
                                if image_files:
                                    mouse_organs.add(organ)
                                    print(
                                        f"Found {len(image_files)} images in {tissue_path}"
                                    )
                                    break
        except Exception as e:
            print(f"Error scanning organs: {e}")

        overlapping = list(human_organs.intersection(mouse_organs))

        print("=" * 60)
        print("HOMOLOGOUS ORGAN ANALYSIS")
        print("=" * 60)
        print(f"Human organs: {sorted(human_organs)}")
        print(f"Mouse organs: {sorted(mouse_organs)}")
        print(f"Overlapping organs: {sorted(overlapping)}")
        print("=" * 60)

        return overlapping

    def _find_tissue_images_path(self, species, organ):
        """Find the correct path for tissue images for a given species and organ"""
        base_folder = os.path.join(self.data_root, f"{species} {organ}")

        if not os.path.exists(base_folder):
            return None

        # Try different possible paths
        possible_paths = [
            os.path.join(base_folder, "tissue images"),
            os.path.join(base_folder, "images"),
            base_folder,  # Sometimes images are directly in the organ folder
        ]

        for path in possible_paths:
            if os.path.exists(path):
                # Check for image files
                image_files = [
                    f
                    for f in os.listdir(path)
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".tif"))
                ]
                if image_files:
                    return path

        return None

    def _extract_features_from_image(self, image_path):
        """Extract ResNet50 features from single image"""
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.feature_extractor(image_tensor)
                features = features.view(features.size(0), -1)

            return features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def collect_homologous_features(self):
        """Collect features from overlapping organs only"""
        all_features = []
        labels = []

        # Count images per organ
        print("\nIMAGE COUNTS PER ORGAN:")
        for organ in self.overlapping_organs:
            human_path = self._find_tissue_images_path("human", organ)
            mouse_path = self._find_tissue_images_path("mouse", organ)

            human_count = 0
            mouse_count = 0

            if human_path:
                human_count = len(
                    [
                        f
                        for f in os.listdir(human_path)
                        if f.lower().endswith(
                            (".png", ".jpg", ".jpeg", ".tiff", ".tif")
                        )
                    ]
                )

            if mouse_path:
                mouse_count = len(
                    [
                        f
                        for f in os.listdir(mouse_path)
                        if f.lower().endswith(
                            (".png", ".jpg", ".jpeg", ".tiff", ".tif")
                        )
                    ]
                )

            print(
                f"{organ}: {human_count} human + {mouse_count} mouse = {human_count + mouse_count} total"
            )

        # Extract features
        for organ in self.overlapping_organs:
            # Human data
            human_folder = self._find_tissue_images_path("human", organ)
            if human_folder:
                for img_file in os.listdir(human_folder):
                    if img_file.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".tiff", ".tif")
                    ):
                        img_path = os.path.join(human_folder, img_file)
                        features = self._extract_features_from_image(img_path)
                        if features is not None:
                            all_features.append(features)
                            labels.append(f"Human {organ.title()}")

            # Mouse data
            mouse_folder = self._find_tissue_images_path("mouse", organ)
            if mouse_folder:
                for img_file in os.listdir(mouse_folder):
                    if img_file.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".tiff", ".tif")
                    ):
                        img_path = os.path.join(mouse_folder, img_file)
                        features = self._extract_features_from_image(img_path)
                        if features is not None:
                            all_features.append(features)
                            labels.append(f"Mouse {organ.title()}")

        return np.array(all_features), labels

    def perform_pca_analysis(self, features, labels, n_components=2):
        """Perform PCA analysis"""
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        pca = PCA(n_components=n_components)
        features_pca = pca.fit_transform(features_scaled)

        print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
        print(f"Total Explained Variance: {sum(pca.explained_variance_ratio_):.3f}")

        return features_pca, pca

    def plot_homologous_distribution(self, features_pca, labels, save_path=None):
        """Create the homologous feature distribution plot (like parent paper Figure 1)"""
        plt.figure(figsize=(12, 8))

        # Define colors for different organ-species combinations
        organ_colors = {
            "kidney": ("#8C0000", "#E97171"),  # Red shades
            "liver": ("#1A0090", "#7FDBDA"),  # Blue shades
            "spleen": ("#004E00", "#58F358"),  # Green shades
            "heart": ("#8B4513", "#DEB887"),  # Brown shades
            "brain": ("#FFD700", "#FFFF99"),  # Yellow shades
            "lung": ("#4682B4", "#87CEEB"),  # Steel blue shades
        }

        scatter_handles = []
        ellipse_handles = []

        # Plot each organ-species combination
        for organ in self.overlapping_organs:
            human_label = f"Human {organ.title()}"
            mouse_label = f"Mouse {organ.title()}"

            human_mask = np.array(labels) == human_label
            mouse_mask = np.array(labels) == mouse_label

            colors = organ_colors.get(organ, ("#999999", "#CCCCCC"))

            if np.any(human_mask):
                scatter = plt.scatter(
                    features_pca[human_mask, 0],
                    features_pca[human_mask, 1],
                    c=colors[0],
                    label=human_label,
                    alpha=0.7,
                    s=50,
                    marker="o",
                )
                scatter_handles.append(scatter)

            if np.any(mouse_mask):
                scatter = plt.scatter(
                    features_pca[mouse_mask, 0],
                    features_pca[mouse_mask, 1],
                    c=colors[1],
                    label=mouse_label,
                    alpha=0.7,
                    s=50,
                    marker="^",
                )
                scatter_handles.append(scatter)

            # Add overlap ellipse for homologous organs
            if np.any(human_mask) and np.any(mouse_mask):
                combined_data = features_pca[human_mask | mouse_mask]

                # Draw confidence ellipse
                mean = np.mean(combined_data, axis=0)
                cov = np.cov(combined_data.T)
                eigenvals, eigenvecs = np.linalg.eigh(cov)

                angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                width, height = 2 * np.sqrt(eigenvals) * 2  # 2 sigma

                ellipse = Ellipse(
                    mean,
                    width,
                    height,
                    angle=angle,
                    fill=False,
                    edgecolor=colors[0],
                    linewidth=2.5,
                    alpha=0.9,
                )
                plt.gca().add_patch(ellipse)

                # Legend ellipse
                legend_ellipse = Ellipse(
                    (0, 0),
                    1,
                    1,
                    angle=0,
                    fill=False,
                    edgecolor=colors[0],
                    linewidth=2.5,
                    alpha=0.9,
                    label=f"{organ.title()} Overlap",
                )
                ellipse_handles.append(legend_ellipse)

        plt.xlabel("PCA Component 1", fontsize=12)
        plt.ylabel("PCA Component 2", fontsize=12)
        plt.title(
            "Homologous Feature Distribution\nCross-Species Comparison of Same Organ Types",
            fontsize=14,
        )

        # Create legend
        all_handles = scatter_handles + ellipse_handles
        plt.legend(handles=all_handles, bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def run_homologous_analysis(self, save_plot=True):
        """Run complete homologous structure analysis"""
        print("Starting homologous PCA analysis...")

        if not self.overlapping_organs:
            print("No overlapping organs found!")
            return None, None, None

        # Collect features
        print("Extracting features from homologous organs...")
        features, labels = self.collect_homologous_features()
        print(f"Collected {len(features)} images across {len(set(labels))} categories")

        # Perform PCA
        print("Performing PCA analysis...")
        features_pca, pca = self.perform_pca_analysis(features, labels)

        # Create plot
        print("Creating homologous visualization...")
        save_path = "homologous_feature_distribution.png" if save_plot else None
        self.plot_homologous_distribution(features_pca, labels, save_path)

        # Print summary
        self._print_summary(labels)

        return features_pca, labels, pca

    def _print_summary(self, labels):
        """Print analysis summary"""
        print("\n" + "=" * 50)
        print("HOMOLOGOUS ANALYSIS SUMMARY")
        print("=" * 50)

        for organ in self.overlapping_organs:
            human_count = sum(1 for l in labels if l == f"Human {organ.title()}")
            mouse_count = sum(1 for l in labels if l == f"Mouse {organ.title()}")
            print(f"{organ.title()}: {human_count} human, {mouse_count} mouse images")

        print(f"\nTotal images analyzed: {len(labels)}")
        print(f"Overlapping organ types: {len(self.overlapping_organs)}")


if __name__ == "__main__":
    # Example usage
    data_root = "/Users/fardeenb/Documents/Projects/CrossSpec/data"

    analyzer = HomologousPCAAnalyzer(data_root)
    features_pca, labels, pca = analyzer.run_homologous_analysis()
