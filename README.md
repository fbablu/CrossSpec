# CrossSpec

A cross-species AI tool for improved kidney layer segmentation.

# Notes

Found .pth file for `Swin-UNet/pretrained_ckpt/swin_tiny_patch4_window7_224.pth` - [Source](https://github.com/HuCaoFighting/Swin-Unet?tab=readme-ov-file).

For the others below, the parent paper has code to setup and train them.

- [ ] PSPNet/model_data/pspnet_mobilenetv2.pth
- [ ] PSPNet/model_data/pspnet_resnet50.pth
- [ ] UNet/model_data/unet_resnet_voc.pth
- [ ] UNet/model_data/unet_vgg_voc.pth



===========================
# CrossSpec

A cross-species AI tool for improved layer segmentation and tissue analysis.

## Overview

CrossSpec implements feature analysis techniques to demonstrate the structural similarity between human and mouse tissue pathology images, supporting the use of cross-species data for enhanced model training in medical image segmentation.

This project reproduces and extends the analysis from the paper _"Cross-Species Data Integration for Enhanced Layer Segmentation in Kidney Pathology"_ by analyzing feature distributions across different organ types and species.

## Features

- **Homologous Analysis**: Compare same organ types across species (e.g., human kidney vs mouse kidney)
- **Analogous Analysis**: Find structural similarities between different organ types across species
- **Automated PCA Visualization**: Generate publication-quality plots showing feature distributions
- **Similarity Matrix**: Compute and visualize cross-species organ similarities

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/fbablu/CrossSpec.git
cd CrossSpec

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset and Run Analysis

```bash
# Download dataset and run both analyses automatically
python main.py --auto

# Or run specific analyses
python main.py --download                    # Download dataset only
python main.py --data /path/to/data --homo   # Run homologous analysis
python main.py --data /path/to/data --ana    # Run analogous analysis
python main.py --data /path/to/data --both   # Run both analyses
```

## Project Structure

```
CrossSpec/
├── main.py                    # Main entry point
├── homologous_pca.py         # Homologous structure analysis
├── analogous_pca.py          # Analogous structure analysis
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── LICENSE                   # License file
└── data/                     # Dataset directory (auto-created)
```

## Analysis Types

### Homologous Analysis (`homologous_pca.py`)

Analyzes the same organ types across different species to demonstrate structural similarity:

- Compares human kidney vs mouse kidney
- Compares human liver vs mouse liver
- Creates PCA plots with overlap ellipses
- Validates cross-species training feasibility

**Output**: `homologous_feature_distribution.png`

### Analogous Analysis (`analogous_pca.py`)

Discovers structural similarities between different organ types across species:

- Compares all human organs vs all mouse organs
- Generates similarity heatmaps
- Identifies top analogous structure pairs
- Provides insights for transfer learning

**Output**: `analogous_similarity_heatmap.png`

## Usage Examples

### Command Line Interface

```bash
# Quick start - download and analyze everything
python main.py --auto

# Use existing dataset
python main.py --data ./data --both

# Run only homologous analysis
python main.py --data ./data --homo

# Run only analogous analysis
python main.py --data ./data --ana
```

### Programmatic Usage

```python
from homologous_pca import HomologousPCAAnalyzer
from analogous_pca import AnalogousPCAAnalyzer

# Homologous analysis
homo_analyzer = HomologousPCAAnalyzer(data_path)
features_pca, labels, pca = homo_analyzer.run_homologous_analysis()

# Analogous analysis
ana_analyzer = AnalogousPCAAnalyzer(data_path)
features_pca, similarity_matrix, top_pairs = ana_analyzer.run_analogous_analysis()
```

## Dataset

This project uses the [NuInsSeg dataset](https://www.kaggle.com/datasets/ipateam/nuinsseg) from Kaggle, which contains histopathological images of various organs from both human and mouse sources.

## Results

The analysis generates:

1. **PCA Visualizations**: Show feature clustering and cross-species overlap
2. **Similarity Matrices**: Quantify structural similarities between organs
3. **Statistical Summaries**: Report variance explained and top analogous pairs

## Technical Details

- **Feature Extraction**: ResNet50 pre-trained on ImageNet
- **Dimensionality Reduction**: PCA with standardized features
- **Similarity Metric**: Cosine similarity between mean organ features
- **Visualization**: Matplotlib with confidence ellipses and color coding

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@article{crossspec2025,
  title={},
  author={},
}
```





# First training results: 
Ok cool, I got this result :
(venv) (base) fardeenb@Fardeen-MacBook-Pro CrossSpec % python multi_organ_train.py --training_mode homologous --organ_type kidney --model_type unet --epochs 1 --batch_size 1
Starting homologous training for kidney
Model: unet, Epochs: 1
Epoch 0, Batch 0/51, Loss: 2.2484
Epoch 0, Batch 10/51, Loss: 2.0292
Epoch 0, Batch 20/51, Loss: 2.3345
Epoch 0, Batch 30/51, Loss: 1.9786
Epoch 0, Batch 40/51, Loss: 2.1243
Epoch 0, Batch 50/51, Loss: 1.9414
Epoch 0 - Average Training Loss: 2.1629
Epoch 0 - Average Validation Loss: 2.0581
Checkpoint saved: logs/homologous_kidney_unet/best_model.pth
Checkpoint saved: logs/homologous_kidney_unet/epoch_000.pth
