import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import re
import os
import joblib

print("🔄 02. PREPROCESANDO DATOS")

df = pd.read_csv('../data/phishing_dataset.csv')
df = df.dropna()

def extract_features(text):
    text = str(text)
    return [
        len(text),
        text.count('.'),
        text.count('/'),
        text.count('-'),
        text.count('?'),
        text.count('='),
        1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', text) else 0
    ]

X = np.array([extract_features(row['URL']) for _, row in df.iterrows()])
y = df['label'].values

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
print(f"✅ Features: {X.shape}, Train: {X_train.shape}")
print(f"✅ Balance: Phishing {np.sum(y)}/{len(y)} ({np.mean(y)*100:.1f}%)")
