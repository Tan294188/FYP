import pandas as pd

splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}

# Load training dataset
df_train = pd.read_parquet("hf://datasets/Andyrasika/banking-marketing/" + splits["train"])

# Load test dataset
df_test = pd.read_parquet("hf://datasets/Andyrasika/banking-marketing/" + splits["test"])

# Concatenate training and test datasets
df_combined = pd.concat([df_train, df_test])

# Save combined dataset to CSV
df_combined.to_csv("banking-marketing_combined.csv", index=False)