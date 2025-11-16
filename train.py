# train.py
import joblib
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os

def main():
    # Load Olivetti faces
    data = fetch_olivetti_faces()
    X = data.data  # (n_samples, n_features)
    y = data.target

    # Train-test split 70:30
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Train DecisionTreeClassifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")

    # Save model as savedmodel.pth (joblib) in root
    os.makedirs("model", exist_ok=True)
    joblib.dump({'model': clf, 'acc': acc}, "model/savedmodel.pth")
    print("Saved model to model/savedmodel.pth")

    # Also save a small metadata file for later use
    with open("model/metadata.txt", "w") as f:
        f.write(f"accuracy:{acc:.4f}\n")
        f.write("model:DecisionTreeClassifier\n")

if __name__ == "__main__":
    main()

