import kagglehub

# downloading this dataset: https://www.kaggle.com/datasets/ipateam/nuinsseg
path = kagglehub.dataset_download("ipateam/nuinsseg")

print("Path to dataset files:", path)