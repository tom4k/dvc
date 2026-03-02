from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import joblib
import os
import json
import pandas as pd

def main():
    data = pd.read_csv("data/data.csv")
    X = data.drop("sales", axis=1)
    y = data["sales"]
  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Lasso(alpha=0.2, selection="random")
    model.fit(X_train, y_train)

    # Save model
    os.makedirs("artifacts", exist_ok=True)
    model_path = os.path.join("artifacts", "model.pkl")
    joblib.dump(model, model_path)

    # Save a tiny metrics file
    acc = model.score(X_test, y_test)
    metrics = {"accuracy": float(acc)}
    with open(os.path.join("artifacts", "metrics.json"), "w") as f:
        json.dump(metrics, f)

    print(f"Saved model to {model_path}")
    print(f"Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()