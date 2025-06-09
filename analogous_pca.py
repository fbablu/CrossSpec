"""
Analogous PCA Analysis
Analyzes similarities between different organ types across species to find
potential analogous structures (e.g., human kidney vs mouse liver).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image


class AnalogousPCAAnalyzer:
    def __init__(self, data_root):
        self.data_root = data_root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pre-trained ResNet50
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

        # Discover all available organs
        self.human_organs, self.mouse_organs = self._find_all_organs()

    def _find_all_organs(self):
        """Find all available organs in both human and mouse datasets"""
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
                                    break
        except Exception as e:
            print(f"Error scanning organs: {e}")

        print("=" * 60)
        print("ANALOGOUS STRUCTURE ANALYSIS")
        print("=" * 60)
        print(f"Human organs ({len(human_organs)}): {sorted(human_organs)}")
        print(f"Mouse organs ({len(mouse_organs)}): {sorted(mouse_organs)}")
        print("=" * 60)

        return human_organs, mouse_organs

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

    def collect_all_organ_features(self):
        """Collect features from ALL organs (not just overlapping)"""
        all_features = []
        labels = []

        # Count images per organ
        print("\nIMAGE COUNTS PER ORGAN:")

        # Get all human organs
        for organ in sorted(self.human_organs):
            human_folder = self._find_tissue_images_path("human", organ)
            if human_folder:
                count = 0
                for img_file in os.listdir(human_folder):
                    if img_file.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".tiff", ".tif")
                    ):
                        img_path = os.path.join(human_folder, img_file)
                        features = self._extract_features_from_image(img_path)
                        if features is not None:
                            all_features.append(features)
                            labels.append(f"Human_{organ}")
                            count += 1
                print(f"Human {organ}: {count} images")

        # Get all mouse organs
        for organ in sorted(self.mouse_organs):
            mouse_folder = self._find_tissue_images_path("mouse", organ)
            if mouse_folder:
                count = 0
                for img_file in os.listdir(mouse_folder):
                    if img_file.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".tiff", ".tif")
                    ):
                        img_path = os.path.join(mouse_folder, img_file)
                        features = self._extract_features_from_image(img_path)
                        if features is not None:
                            all_features.append(features)
                            labels.append(f"Mouse_{organ}")
                            count += 1
                print(f"Mouse {organ}: {count} images")

        return np.array(all_features), labels

    def compute_cross_species_similarity_matrix(self, features, labels):
        """Compute similarity between all human and mouse organ pairs"""
        # Group features by organ-species
        organ_features = {}
        for i, label in enumerate(labels):
            if label not in organ_features:
                organ_features[label] = []
            organ_features[label].append(features[i])

        # Compute mean features for each organ-species
        organ_means = {}
        for organ, feat_list in organ_features.items():
            organ_means[organ] = np.mean(feat_list, axis=0)

        # Create similarity matrix between human and mouse organs
        human_organs = sorted([k for k in organ_means.keys() if k.startswith("Human_")])
        mouse_organs = sorted([k for k in organ_means.keys() if k.startswith("Mouse_")])

        similarity_matrix = np.zeros((len(human_organs), len(mouse_organs)))

        for i, h_organ in enumerate(human_organs):
            for j, m_organ in enumerate(mouse_organs):
                sim = cosine_similarity([organ_means[h_organ]], [organ_means[m_organ]])[
                    0
                ][0]
                similarity_matrix[i, j] = sim

        return similarity_matrix, human_organs, mouse_organs

    """
    def plot_analogous_pca(self, features, labels):
        # Create PCA plot for ALL organs with cross-species highlighting
        # Standardize and apply PCA
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)

        plt.figure(figsize=(15, 10))

        # Create distinct colors for each organ type
        all_organ_types = list(self.human_organs.union(self.mouse_organs))
        colors = plt.cm.tab20(np.linspace(0, 1, len(all_organ_types)))

        color_map = {}
        for i, organ in enumerate(all_organ_types):
            color_map[organ] = colors[i]

        # Plot points
        for label in set(labels):
            mask = np.array(labels) == label
            species, organ = label.split("_", 1)
            marker = "o" if species == "Human" else "^"
            size = 80 if species == "Human" else 60
            alpha = 0.8 if species == "Human" else 0.6

            plt.scatter(
                features_pca[mask, 0],
                features_pca[mask, 1],
                c=[color_map[organ]],
                label=label.replace("_", " "),
                alpha=alpha,
                s=size,
                marker=marker,
                edgecolors="black",
                linewidth=0.5,
            )

        plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.3f})")
        plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]:.3f})")
        plt.title(
            "Analogous Structure Analysis\nCross-Species Comparison of All Organ Types"
        )

        # Create legend with better organization
        handles, labels_legend = plt.gca().get_legend_handles_labels()
        # Sort legend by species then organ
        human_items = [
            (h, l) for h, l in zip(handles, labels_legend) if l.startswith("Human")
        ]
        mouse_items = [
            (h, l) for h, l in zip(handles, labels_legend) if l.startswith("Mouse")
        ]

        sorted_handles = [h for h, l in human_items] + [h for h, l in mouse_items]
        sorted_labels = [l for h, l in human_items] + [l for h, l in mouse_items]

        plt.legend(
            sorted_handles,
            sorted_labels,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            ncol=2,
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return features_pca, pca
  """

    def plot_analogous_pca_with_ellipses(self, features, labels):
        """Create analogous PCA plot with similarity ellipses for top pairs"""
        # Standardize and apply PCA
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)

        plt.figure(figsize=(15, 10))

        # Top analogous pairs
        top_analogous_pairs = [
            ("Human_liver", "Mouse_kidney", 0.9413),
            ("Human_placenta", "Mouse_kidney", 0.9267),
            ("Human_salivory_gland", "Mouse_kidney", 0.9222),
            ("Human_kidney", "Mouse_spleen", 0.9274),
        ]

        # Create distinct colors for each organ type
        all_organ_types = list(self.human_organs.union(self.mouse_organs))
        colors = plt.cm.tab20(np.linspace(0, 1, len(all_organ_types)))
        color_map = {}
        for i, organ in enumerate(all_organ_types):
            color_map[organ] = colors[i]

        # Plot points
        for label in set(labels):
            mask = np.array(labels) == label
            species, organ = label.split("_", 1)
            marker = "o" if species == "Human" else "^"
            size = 80 if species == "Human" else 60
            alpha = 0.8 if species == "Human" else 0.6

            plt.scatter(
                features_pca[mask, 0],
                features_pca[mask, 1],
                c=[color_map[organ]],
                label=label.replace("_", " "),
                alpha=alpha,
                s=size,
                marker=marker,
                edgecolors="black",
                linewidth=0.5,
            )

        # Add ellipses for top analogous pairs
        ellipse_colors = ["red", "blue", "green", "purple", "orange"]

        for i, (human_organ, mouse_organ, similarity) in enumerate(top_analogous_pairs):
            human_mask = np.array(labels) == human_organ
            mouse_mask = np.array(labels) == mouse_organ

            if np.any(human_mask) and np.any(mouse_mask):
                # Combine data from both organs
                combined_data = features_pca[human_mask | mouse_mask]

                if len(combined_data) > 2:  # Need at least 3 points for ellipse
                    # Draw confidence ellipse
                    mean = np.mean(combined_data, axis=0)
                    cov = np.cov(combined_data.T)
                    eigenvals, eigenvecs = np.linalg.eigh(cov)

                    # Handle negative eigenvalues
                    eigenvals = np.abs(eigenvals)

                    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                    width, height = 2 * np.sqrt(eigenvals) * 2.0  # 2 sigma

                    from matplotlib.patches import Ellipse

                    ellipse = Ellipse(
                        mean,
                        width,
                        height,
                        angle=angle,
                        fill=False,
                        edgecolor=ellipse_colors[i % len(ellipse_colors)],
                        linewidth=3,
                        alpha=0.9,
                        linestyle="--",
                    )
                    plt.gca().add_patch(ellipse)

                    # Add similarity score annotation
                    plt.annotate(
                        f"Sim: {similarity:.3f}",
                        xy=mean,
                        xytext=(10, 10),
                        textcoords="offset points",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor=ellipse_colors[i % len(ellipse_colors)],
                            alpha=0.3,
                        ),
                        fontsize=10,
                        fontweight="bold",
                    )

        plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.3f})")
        plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]:.3f})")
        plt.title(
            "Analogous Structure Analysis with Similarity Ellipses\nDashed ellipses show top cross-organ similarities"
        )

        # Create legend
        handles, labels_legend = plt.gca().get_legend_handles_labels()
        plt.legend(
            handles, labels_legend, bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return features_pca, pca

    def plot_similarity_heatmap(
        self, similarity_matrix, human_organs, mouse_organs, save_path=None
    ):
        """Plot heatmap of cross-species organ similarities"""
        plt.figure(figsize=(12, 8))

        # Clean organ names for display
        h_names = [name.replace("Human_", "") for name in human_organs]
        m_names = [name.replace("Mouse_", "") for name in mouse_organs]

        # Create heatmap
        sns.heatmap(
            similarity_matrix,
            xticklabels=m_names,
            yticklabels=h_names,
            annot=True,
            cmap="RdYlBu_r",
            center=0.7,  # Center around typical similarity values
            fmt=".3f",
            cbar_kws={"label": "Cosine Similarity"},
        )

        plt.title(
            "Cross-Species Analogous Structure Similarity Matrix\n(ResNet50 Feature Similarity)",
            fontsize=14,
        )
        plt.xlabel("Mouse Organs", fontsize=12)
        plt.ylabel("Human Organs", fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def find_top_analogous_pairs(
        self, similarity_matrix, human_organs, mouse_organs, top_k=10
    ):
        """Find most similar cross-species organ pairs"""
        pairs = []

        for i, h_organ in enumerate(human_organs):
            for j, m_organ in enumerate(mouse_organs):
                sim = similarity_matrix[i, j]
                h_clean = h_organ.replace("Human_", "")
                m_clean = m_organ.replace("Mouse_", "")
                pairs.append((h_clean, m_clean, sim))

        # Sort by similarity
        pairs.sort(key=lambda x: x[2], reverse=True)

        print("\n" + "=" * 70)
        print("TOP ANALOGOUS STRUCTURE PAIRS")
        print("=" * 70)
        print(f"{'Rank':<5} {'Human Organ':<20} {'Mouse Organ':<20} {'Similarity':<15}")
        print("-" * 70)

        for i, (h_org, m_org, sim) in enumerate(pairs[:top_k]):
            print(f"{i+1:<5} {h_org:<20} {m_org:<20} {sim:.4f}")

        return pairs[:top_k]

    def run_analogous_analysis(self, save_plots=True):
        """Run complete analogous structure analysis"""
        print("Starting analogous PCA analysis...")

        # Collect features
        print("Extracting features from ALL organs...")
        features, labels = self.collect_all_organ_features()

        print(
            f"\nCollected {len(features)} images from {len(set(labels))} organ-species combinations"
        )

        # PCA visualization
        print("\nCreating analogous structure PCA plot...")
        features_pca, pca = self.plot_analogous_pca_with(features, labels)

        # Similarity analysis
        print("\nComputing cross-species similarity matrix...")
        similarity_matrix, human_organs, mouse_organs = (
            self.compute_cross_species_similarity_matrix(features, labels)
        )

        # Heatmap
        print("Creating similarity heatmap...")
        heatmap_path = "analogous_similarity_heatmap.png" if save_plots else None
        self.plot_similarity_heatmap(
            similarity_matrix, human_organs, mouse_organs, heatmap_path
        )

        # Top pairs
        top_pairs = self.find_top_analogous_pairs(
            similarity_matrix, human_organs, mouse_organs
        )

        # Summary
        self._print_summary(len(features), len(set(labels)), top_pairs[:5])

        return features_pca, similarity_matrix, top_pairs

    def _print_summary(self, total_images, total_categories, top_5_pairs):
        """Print analysis summary"""
        print("\n" + "=" * 50)
        print("ANALOGOUS ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Total images analyzed: {total_images}")
        print(f"Total organ-species combinations: {total_categories}")
        print(f"Human organ types: {len(self.human_organs)}")
        print(f"Mouse organ types: {len(self.mouse_organs)}")

        print("\nTop 5 most analogous pairs:")
        for i, (h_org, m_org, sim) in enumerate(top_5_pairs):
            print(f"  {i+1}. Human {h_org} ↔ Mouse {m_org} (similarity: {sim:.4f})")


if __name__ == "__main__":
    # Example usage
    data_root = "/Users/fardeenb/Documents/Projects/CrossSpec/data"

    analyzer = AnalogousPCAAnalyzer(data_root)
    features_pca, similarity_matrix, top_pairs = analyzer.run_analogous_analysis()
