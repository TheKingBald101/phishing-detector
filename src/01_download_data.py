import pandas as pd
import requests
import os 

print("DOWNLOAD PHISHING DATASET")
os.makedirs('../data', exist_ok=True)

url = "https://raw.githubusercontent.com/GregaVrbancic/Phishing-Dataset/master/dataset_full.csv"
df = pd.read_csv(url)
df.to_csv('../data/phishing_dataset.csv', index=False)

print(f"Dataset saved: {len(df)} emails")
print(f"Columns: {list(df.columns)}")
print("\nFirst three rows")
print(df.head(3))
