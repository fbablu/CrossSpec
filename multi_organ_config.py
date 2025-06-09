"""
Multi-Organ Cross-Species Training Configuration
Based on PCA similarity analysis results
"""

import numpy as np

# Organ class mappings based on PCA results
ORGAN_CLASSES = {
    "kidney": {"human": 0, "mouse": 1},
    "liver": {"human": 2, "mouse": 3},
    "spleen": {"human": 4, "mouse": 5},
    "background": 6,
}

# Combined training: 7 classes total (6 organ-species + background)
NUM_CLASSES_COMBINED = 7
NUM_CLASSES_SEPARATE = 2  # background + target organ

# Homologous pairs (same organ, different species)
HOMOLOGOUS_PAIRS = [
    ("human_kidney", "mouse_kidney"),
    ("human_liver", "mouse_liver"),
    ("human_spleen", "mouse_spleen"),
]

# Analogous pairs (different organs, high similarity from PCA)
ANALOGOUS_PAIRS = [
    ("human_liver", "mouse_kidney", 0.9413),
    ("human_placenta", "mouse_kidney", 0.9267),
    ("human_salivary_gland", "mouse_kidney", 0.9222),
    ("human_kidney", "mouse_spleen", 0.9274),
]


############ WHAT DOES THIS MEAN BY IMAGE COUNTS? IS THIS FROM PARENT PAPER? HELPME
# Class weights for imbalanced data from image counts
CLASS_WEIGHTS = {
    "combined": np.array([1.0, 2.5, 1.0, 1.0, 1.5, 4.0, 1.0]),  # Based on your counts
    "kidney": np.array([1.0, 2.5]),  # Human:11, Mouse:40
    "liver": np.array([1.0, 1.1]),  # Human:40, Mouse:36
    "spleen": np.array([1.0, 4.8]),  # Human:34, Mouse:7
}

# Training modes
TRAINING_MODES = {
    "separate": "Train each organ-species separately",
    "homologous": "Train same organs across species together",
    "analogous": "Train different organs with similarity weighting",
    "combined": "Train all organs together with full cross-species data",
}

# Dataset paths
DATASET_PATHS = {
    "human kidney": "/Users/fardeenb/Documents/Projects/CrossSpec/data/human kidney/tissue images",
    "mouse kidney": "/Users/fardeenb/Documents/Projects/CrossSpec/data/mouse kidney/tissue images",
    "human liver": "/Users/fardeenb/Documents/Projects/CrossSpec/data/human liver/tissue images",
    "mouse liver": "/Users/fardeenb/Documents/Projects/CrossSpec/data/mouse liver/tissue images",
    "human spleen": "/Users/fardeenb/Documents/Projects/CrossSpec/data/human spleen/tissue images",
    "mouse spleen": "/Users/fardeenb/Documents/Projects/CrossSpec/data/mouse spleen/tissue images",
}

# Label paths
LABEL_PATHS = {
    "human kidney": "/Users/fardeenb/Documents/Projects/CrossSpec/data/human kidney/label masks",
    "mouse kidney": "/Users/fardeenb/Documents/Projects/CrossSpec/data/mouse kidney/label masks",
    "human liver": "/Users/fardeenb/Documents/Projects/CrossSpec/data/human liver/label masks",
    "mouse liver": "/Users/fardeenb/Documents/Projects/CrossSpec/data/mouse liver/label masks",
    "human spleen": "/Users/fardeenb/Documents/Projects/CrossSpec/data/human spleen/label masks",
    "mouse spleen": "/Users/fardeenb/Documents/Projects/CrossSpec/data/mouse spleen/label masks",
}

# Loss function hyperparameters (from parent paper)
LOSS_PARAMS = {
    "separate": {"lambda1": 0.5, "lambda2": 0.75},  # CE + Dice
    "combined": {"lambda3": 0.75, "lambda4": 1.0},  # Focal + Dice
    "analogous": {"lambda5": 0.5},  # Additional similarity weight
}
