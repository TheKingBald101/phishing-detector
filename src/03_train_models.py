import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("TRAININIG DOBLE LAYER")

X_train = np.load('../data/X_train.npy')
X_test = np.load('../data/X_test.npy')
y_train = np.load('../data/y_train.npy')
y_test = np.load('../data/y_test.npy')

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

joblib.dump(rf, '../data/rf_phishing_model.pkl')
joblib.dump(lr, '../data/lr_spam_model.pkl')

print(f"RF Phishing: {rf_acc:.3f}")
print(f"LR Spam: {lr_acc:.3f}")
print("\nReport RF:")
print(classification_report(y_test, rf_pred))