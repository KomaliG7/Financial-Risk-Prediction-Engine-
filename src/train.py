import argparse
import numpy as np
import xgboost as xgb
import joblib
import os

def main(processed_path, model_path):
    print(f"Loading processed data from {processed_path}...")
    data = np.load(processed_path)
    X_train, y_train = data["X"], data["y"]

    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✅ Model saved successfully to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost risk prediction model.")
    parser.add_argument("--processed", required=True, help="Path to processed training data (.npz)")
    parser.add_argument("--out", required=True, help="Output path for trained model")
    args = parser.parse_args()
    main(args.processed, args.out)
