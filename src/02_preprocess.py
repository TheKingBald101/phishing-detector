import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import re
import os
import joblib
print("PREPROCESSING DATA")
df = pd.read_csv('../data/phishing_dataset.csv')
print(f"Dataset shape: {df.shape}")
print(f"Aviable columns: {list(df.columns)}")
print("\nFirst three rows:")
print(df.head(3))
if 'URL' in df.columns:
    url_col = 'URL'
elif 'url' in df.columns:
    url_col = 'url'
elif len(df.columns) > 0:
    url_col = df.columns[0] 
else:
    raise ValueError("No hay columnas en el dataset")

if 'label' in df.columns:
    label_col = 'label'
elif 'Label' in df.columns:
    label_col = 'Label'
elif 'target' in df.columns:
    label_col = 'target'
else:
    label_col = df.columns[-1]

print(f"\n🔍 Usando: {url_col} para features, {label_col} para labels")

df = df.dropna(subset=[url_col, label_col])
print(f"Limpio: {df.shape}")

def extract_features(text):
    t = str(text)
    return [
        len(t),           
        t.count('.'),     
        t.count('/'),     
        t.count('-'),     
        t.count('?'),     
        t.count('='),     
        1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', t) else 0  # IP
    ]

print("Extrayendo features...")
X = np.array([extract_features(row[url_col]) for _, row in df.iterrows()])
y = df[label_col].values.astype(int)
print(f"Features shape: {X.shape}")
print(f"Labels: {np.bincount(y)} - Phishing: {np.mean(y)*100:.1f}%")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
os.makedirs('../data', exist_ok=True)
np.save('../data/X_train.npy', X_train)
np.save('../data/X_test.npy', X_test)
np.save('../data/y_train.npy', y_train)
np.save('../data/y_test.npy', y_test)
joblib.dump(scaler, '../data/scaler.pkl')
print("PREPROCESSING COMPELTE")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")