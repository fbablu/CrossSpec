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
import seaborn as sns


class CrossSpeciesFeatureAnalyzer:
    def __init__(self, data_root):
        self.data_root = data_root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pre-trained ResNet50 (same as parent paper)
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

        # Will discover all overlapping organs first
        self.overlapping_organs, self.human_organs, self.mouse_organs = (
            self.scan_available_organs()
        )

    def scan_available_organs(self):
        """First step: discover all available organs in both human and mouse"""
        human_organs = set()
        mouse_organs = set()

        for folder in os.listdir(self.data_root):
            if os.path.isdir(os.path.join(self.data_root, folder)):
                if folder.startswith("human "):
                    organ = folder.replace("human ", "")
                    # Check if has tissue images
                    tissue_path = os.path.join(self.data_root, folder, "tissue images")
                    if os.path.exists(tissue_path) and os.listdir(tissue_path):
                        human_organs.add(organ)
                elif folder.startswith("mouse "):
                    organ = folder.replace("mouse ", "")
                    tissue_path = os.path.join(self.data_root, folder, "tissue images")
                    if os.path.exists(tissue_path) and os.listdir(tissue_path):
                        mouse_organs.add(organ)

        overlapping = list(human_organs.intersection(mouse_organs))
        human_only = human_organs - mouse_organs
        mouse_only = mouse_organs - human_organs

        print("=" * 60)
        print("ORGAN AVAILABILITY ANALYSIS")
        print("=" * 60)
        print(f"Human organs ({len(human_organs)}): {sorted(human_organs)}")
        print(f"Mouse organs ({len(mouse_organs)}): {sorted(mouse_organs)}")
        print(f"\nOVERLAPPING organs ({len(overlapping)}): {sorted(overlapping)}")
        print(f"Human-only organs ({len(human_only)}): {sorted(human_only)}")
        print(f"Mouse-only organs ({len(mouse_only)}): {sorted(mouse_only)}")
        print("=" * 60)

        return overlapping, human_organs, mouse_organs

    def extract_features_from_image(self, image_path):
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

    def count_images_per_organ(self, organs):
        """Count available images for each organ"""
        image_counts = {}

        for organ in organs:
            human_path = os.path.join(self.data_root, f"human {organ}", "tissue images")
            mouse_path = os.path.join(self.data_root, f"mouse {organ}", "tissue images")

            human_count = (
                len([f for f in os.listdir(human_path) if f.endswith(".png")])
                if os.path.exists(human_path)
                else 0
            )
            mouse_count = (
                len([f for f in os.listdir(mouse_path) if f.endswith(".png")])
                if os.path.exists(mouse_path)
                else 0
            )

            image_counts[organ] = {
                "human": human_count,
                "mouse": mouse_count,
                "total": human_count + mouse_count,
            }

        return image_counts
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

    def collect_all_features(self):
        """Collect features from ALL overlapping organs"""
        all_features = []
        labels = []

        # Print summary before processing
        image_counts = self.count_images_per_organ(self.overlapping_organs)
        print("\nIMAGE COUNTS PER ORGAN:")
        for organ, counts in image_counts.items():
            print(
                f"{organ}: {counts['human']} human + {counts['mouse']} mouse = {counts['total']} total"
            )

        for organ in self.overlapping_organs:
            # Human data
            human_folder = os.path.join(
                self.data_root, f"human {organ}", "tissue images"
            )
            if os.path.exists(human_folder):
                for img_file in os.listdir(human_folder):
                    if img_file.endswith(".png"):
                        img_path = os.path.join(human_folder, img_file)
                        features = self.extract_features_from_image(img_path)
                        if features is not None:
                            all_features.append(features)
                            labels.append(f"Human {organ.title()}")

            # Mouse data
            mouse_folder = os.path.join(
                self.data_root, f"mouse {organ}", "tissue images"
            )
            if os.path.exists(mouse_folder):
                for img_file in os.listdir(mouse_folder):
                    if img_file.endswith(".png"):
                        img_path = os.path.join(mouse_folder, img_file)
                        features = self.extract_features_from_image(img_path)
                        if features is not None:
                            all_features.append(features)
                            labels.append(f"Mouse {organ.title()}")

        return np.array(all_features), labels

    def perform_pca_analysis(self, features, labels, n_components=2):
        """Perform PCA analysis and return transformed features"""
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Apply PCA
        pca = PCA(n_components=n_components)
        features_pca = pca.fit_transform(features_scaled)

        print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
        print(f"Total Explained Variance: {sum(pca.explained_variance_ratio_):.3f}")

        return features_pca, pca

    def plot_feature_distribution(self, features_pca, labels, save_path=None):
        """Create the feature distribution plot similar to parent paper"""
        plt.figure(figsize=(12, 8))

        # Define colors for different organ-species combinations
        colors = {
            "Human Kidney": "#8C0000",  # Red
            "Mouse Kidney": "#E97171",  # Light Red
            "Human Liver": "#1A0090",  # Teal
            "Mouse Liver": "#7FDBDA",  # Light Teal
            # 'Human Muscle': '#45B7D1',    # Blue
            # 'Mouse Muscle': '#7CC7E8',    # Light Blue
            "Human Spleen": "#004E00",  # Green
            "Mouse Spleen": "#58F358",  # Light Green
        }

        scatter_handles = []

        # Plot each organ-species combination
        for label in set(labels):
            mask = np.array(labels) == label
            scatter = plt.scatter(
                features_pca[mask, 0],
                features_pca[mask, 1],
                c=colors.get(label, "#999999"),
                label=label,
                alpha=0.7,
                s=50,
            )
            scatter_handles.append(scatter)

        organ_colors = {
            # Homologous organs
            "kidney": "red",
            "liver": "purple",
            "spleen": "orange",
            # Analogous organs
            "heart": "green",
            "brain": "yellow",
            "lung": "blue",
        }

        ellipse_handles = []

        # Add overlap visualization
        for organ in self.overlapping_organs:
            human_mask = np.array(labels) == f"Human {organ.title()}"
            mouse_mask = np.array(labels) == f"Mouse {organ.title()}"

            color = organ_colors.get(organ, "black")

            if np.any(human_mask) and np.any(mouse_mask):
                # Combine human and mouse data for this organ
                combined_data = features_pca[human_mask | mouse_mask]

                # Draw confidence ellipse
                mean = np.mean(combined_data, axis=0)
                cov = np.cov(combined_data.T)
                eigenvals, eigenvecs = np.linalg.eigh(cov)

                # Draw ellipse
                angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                width, height = 2 * np.sqrt(eigenvals) * 2  # 2 sigma

                ellipse = Ellipse(
                    mean,
                    width,
                    height,
                    angle=angle,
                    fill=False,
                    edgecolor=color,
                    linewidth=2.0,
                    alpha=0.9,
                )
                plt.gca().add_patch(ellipse)

                # For legend — create a dummy Ellipse handle with label
                legend_ellipse = Ellipse(
                    (0, 0),
                    1,
                    1,
                    angle=0,
                    fill=False,
                    edgecolor=color,
                    linewidth=2.0,
                    alpha=0.9,
                    label=f"{organ.title()} Overlap",
                )
                ellipse_handles.append(legend_ellipse)

        plt.xlabel("PCA Component 1", fontsize=12)
        plt.ylabel("PCA Component 2", fontsize=12)
        plt.title(
            "Feature Distribution of Human and Mouse Pathology Images\nAcross Multiple Organs",
            fontsize=14,
        )
        plt.legend(
            handles=scatter_handles + ellipse_handles,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def run_full_analysis(self, save_plot=True):
        """Run complete cross-species feature analysis"""
        print("Starting cross-species feature analysis...")

        # Collect features
        print("Extracting features from all images...")
        features, labels = self.collect_all_features()
        print(f"Collected {len(features)} images across {len(set(labels))} categories")

        # Perform PCA
        print("Performing PCA analysis...")
        features_pca, pca = self.perform_pca_analysis(features, labels)

        # Create plot
        print("Creating visualization...")
        save_path = "cross_species_feature_distribution.png" if save_plot else None
        self.plot_feature_distribution(features_pca, labels, save_path)

        # Print summary statistics
        self.print_analysis_summary(labels)

        return features_pca, labels, pca

    def print_analysis_summary(self, labels):
        """Print summary of the analysis"""
        print("\n" + "=" * 50)
        print("CROSS-SPECIES ANALYSIS SUMMARY")
        print("=" * 50)

        for organ in self.overlapping_organs:
            human_count = sum(1 for l in labels if l == f"Human {organ.title()}")
            mouse_count = sum(1 for l in labels if l == f"Mouse {organ.title()}")
            print(f"{organ.title()}: {human_count} human, {mouse_count} mouse images")

        print(f"\nTotal images analyzed: {len(labels)}")
        print(f"Overlapping organ types: {len(self.overlapping_organs)}")
        print(f"Total human organs available: {len(self.human_organs)}")
        print(f"Total mouse organs available: {len(self.mouse_organs)}")


# Usage
if __name__ == "__main__":
    # Update this path to your NuInsSeg data directory
    data_root = "/Users/fardeenb/Documents/Projects/CrossSpec/data"

    analyzer = CrossSpeciesFeatureAnalyzer(data_root)
    features_pca, labels, pca = analyzer.run_full_analysis()
