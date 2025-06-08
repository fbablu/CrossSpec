"""
CrossSpec: Cross-Species AI Tool for Improved Tissue Analysis
Main entry point for running homologous and analogous PCA analyses.
"""

import os
import sys
import shutil
import argparse
import kagglehub
from pathlib import Path

# Import our analysis modules
from homologous_pca import HomologousPCAAnalyzer
from analogous_pca import AnalogousPCAAnalyzer


def explore_dataset_structure(data_path, max_depth=3):
    """Recursively explore dataset structure to understand the organization"""
    print("\n" + "=" * 60)
    print("EXPLORING DATASET STRUCTURE")
    print("=" * 60)

    def explore_recursive(path, depth=0, max_depth=3):
        if depth > max_depth:
            return

        try:
            items = sorted(os.listdir(path))
            for item in items[:10]:  # Limit to first 10 items per directory
                item_path = os.path.join(path, item)
                indent = "  " * depth

                if os.path.isdir(item_path):
                    print(f"{indent}📁 {item}/")
                    if depth < max_depth:
                        explore_recursive(item_path, depth + 1, max_depth)
                else:
                    # Show file extension and count if many files
                    print(f"{indent}📄 {item}")

            if len(items) > 10:
                print(f"{indent}... and {len(items) - 10} more items")

        except PermissionError:
            print(f"{indent}(Permission denied)")
        except Exception as e:
            print(f"{indent}(Error: {e})")

    explore_recursive(data_path, 0, max_depth)


def find_actual_data_directory(base_path):
    """Find the directory that contains the actual organ folders"""
    print(f"\nSearching for organ data in: {base_path}")

    def search_for_organs(path, max_depth=3, current_depth=0):
        if current_depth > max_depth:
            return None

        try:
            items = os.listdir(path)

            # Look for patterns like "human kidney", "mouse liver", etc.
            organ_folders = [
                item
                for item in items
                if (item.startswith("human ") or item.startswith("mouse "))
                and os.path.isdir(os.path.join(path, item))
            ]

            if organ_folders:
                print(f"✓ Found {len(organ_folders)} organ folders in: {path}")
                print(f"  Examples: {organ_folders[:5]}")
                return path

            # If not found, search in subdirectories
            for item in items:
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    result = search_for_organs(item_path, max_depth, current_depth + 1)
                    if result:
                        return result

            return None

        except (PermissionError, OSError) as e:
            print(f"Cannot access {path}: {e}")
            return None

    return search_for_organs(base_path)


def download_dataset():
    """Download the NuInsSeg dataset from Kaggle"""
    print("=" * 60)
    print("DOWNLOADING NUINSSEG DATASET")
    print("=" * 60)

    try:
        # Download the dataset
        path = kagglehub.dataset_download("ipateam/nuinsseg")
        target_dir = "/Users/fardeenb/Documents/Projects/CrossSpec/data"
        os.makedirs(target_dir, exist_ok=True)

        # Move to our project directory
        new_path = os.path.join(target_dir, os.path.basename(path))
        if os.path.exists(new_path):
            shutil.rmtree(new_path)  # Remove existing
        shutil.move(path, new_path)
        print(f"Dataset downloaded and moved to: {new_path}")

        # Explore the structure
        explore_dataset_structure(new_path, max_depth=2)

        # Find the actual data directory
        actual_data_path = find_actual_data_directory(new_path)

        if actual_data_path:
            print(f"\n✓ Dataset ready at: {actual_data_path}")
            return actual_data_path
        else:
            print("❌ Could not find organ data directories")
            return None

    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("Please ensure you have kaggle credentials configured.")
        print("See: https://github.com/Kaggle/kaggle-api#api-credentials")
        return None


def run_homologous_analysis(data_path):
    """Run homologous PCA analysis"""
    print("\n" + "=" * 60)
    print("RUNNING HOMOLOGOUS ANALYSIS")
    print("=" * 60)

    try:
        analyzer = HomologousPCAAnalyzer(data_path)
        features_pca, labels, pca = analyzer.run_homologous_analysis(save_plot=True)

        if features_pca is not None:
            print("✓ Homologous analysis completed successfully!")
            print("✓ Plot saved as 'homologous_feature_distribution.png'")
            return True
        else:
            print("❌ No overlapping organs found for homologous analysis")
            return False

    except Exception as e:
        print(f"❌ Error in homologous analysis: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_analogous_analysis(data_path):
    """Run analogous PCA analysis"""
    print("\n" + "=" * 60)
    print("RUNNING ANALOGOUS ANALYSIS")
    print("=" * 60)

    try:
        analyzer = AnalogousPCAAnalyzer(data_path)
        features_pca, similarity_matrix, top_pairs = analyzer.run_analogous_analysis(
            save_plots=True
        )

        print("✓ Analogous analysis completed successfully!")
        print("✓ Plot saved as 'analogous_similarity_heatmap.png'")
        return True

    except Exception as e:
        print(f"❌ Error in analogous analysis: {e}")
        import traceback

        traceback.print_exc()
        return False


def validate_data_path(data_path):
    """Validate that the data path contains the expected structure"""
    if not os.path.exists(data_path):
        return False, f"Path does not exist: {data_path}"

    # Look for organ folders
    try:
        items = os.listdir(data_path)
        organ_folders = [
            item
            for item in items
            if (item.startswith("human ") or item.startswith("mouse "))
            and os.path.isdir(os.path.join(data_path, item))
        ]

        if not organ_folders:
            # Try to find the correct subdirectory
            actual_path = find_actual_data_directory(data_path)
            if actual_path and actual_path != data_path:
                return True, actual_path
            else:
                return False, f"No organ folders found in {data_path}"

        return True, data_path

    except Exception as e:
        return False, f"Error accessing path: {e}"


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="CrossSpec: Cross-Species AI Tool for Tissue Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --download                    # Download dataset only
  python main.py --data /path/to/data --homo   # Run homologous analysis
  python main.py --data /path/to/data --ana    # Run analogous analysis
  python main.py --data /path/to/data --both   # Run both analyses
  python main.py --auto                        # Download dataset and run both analyses
  python main.py --explore /path/to/data       # Just explore dataset structure
        """,
    )

    parser.add_argument("--data", type=str, help="Path to NuInsSeg dataset directory")
    parser.add_argument(
        "--download", action="store_true", help="Download dataset from Kaggle"
    )
    parser.add_argument(
        "--homo", action="store_true", help="Run homologous analysis only"
    )
    parser.add_argument(
        "--ana", action="store_true", help="Run analogous analysis only"
    )
    parser.add_argument("--both", action="store_true", help="Run both analyses")
    parser.add_argument(
        "--auto", action="store_true", help="Download dataset and run both analyses"
    )
    parser.add_argument(
        "--explore", type=str, help="Just explore the structure of a dataset directory"
    )

    args = parser.parse_args()

    # Print header
    print("=" * 60)
    print("CrossSpec: Cross-Species AI Tool")
    print("Enhanced Layer Segmentation in Kidney Pathology")
    print("=" * 60)

    # Handle explore option
    if args.explore:
        explore_dataset_structure(args.explore, max_depth=3)
        actual_path = find_actual_data_directory(args.explore)
        if actual_path:
            print(f"\n✓ Recommended data path: {actual_path}")
        sys.exit(0)

    data_path = args.data

    # Handle download
    if args.download or args.auto:
        data_path = download_dataset()
        if not data_path:
            sys.exit(1)

    # Validate data path
    if not data_path:
        print("❌ No data path provided. Use --data or --download option.")
        parser.print_help()
        sys.exit(1)

    # Validate and potentially correct the data path
    is_valid, corrected_path = validate_data_path(data_path)
    if not is_valid:
        print(f"❌ {corrected_path}")
        sys.exit(1)

    if corrected_path != data_path:
        print(f"📍 Using corrected data path: {corrected_path}")
        data_path = corrected_path
    else:
        print(f"📍 Using data path: {data_path}")

    # Determine which analyses to run
    run_homo = args.homo or args.both or args.auto
    run_ana = args.ana or args.both or args.auto

    if not (run_homo or run_ana):
        print("❌ No analysis specified. Use --homo, --ana, --both, or --auto")
        parser.print_help()
        sys.exit(1)

    # Run analyses
    results = []

    if run_homo:
        success = run_homologous_analysis(data_path)
        results.append(("Homologous", success))

    if run_ana:
        success = run_analogous_analysis(data_path)
        results.append(("Analogous", success))

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    for analysis_type, success in results:
        status = "✓ COMPLETED" if success else "❌ FAILED"
        print(f"{analysis_type} Analysis: {status}")

    # Check if any files were created
    output_files = [
        "homologous_feature_distribution.png",
        "analogous_similarity_heatmap.png",
    ]

    created_files = [f for f in output_files if os.path.exists(f)]
    if created_files:
        print(f"\nOutput files created:")
        for file in created_files:
            print(f"  📄 {file}")

    print("=" * 60)


if __name__ == "__main__":
    main()
