# test.py
import joblib
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    data = fetch_olivetti_faces()
    X = data.data
    y = data.target

    # Same split parameters as train.py to reconstruct test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    model_data = joblib.load("model/savedmodel.pth")
    clf = model_data['model']
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Loaded model test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()

