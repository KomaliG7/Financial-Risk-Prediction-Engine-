import argparse
import joblib
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
import seaborn as sns
import os

# -----------------------------
# Evaluation + SHAP Visualization
# -----------------------------
def evaluate_model(model, X_test, y_test):
    print("\n📈 Generating predictions ...")
    y_pred = model.predict(X_test)

    # Evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print("\n✅ Evaluation Metrics:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    # -----------------------------
    # Plot Confusion Matrix
    # -----------------------------
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    os.makedirs("reports/figures", exist_ok=True)
    plt.savefig("reports/figures/confusion_matrix.png", bbox_inches="tight")
    plt.close()

    # -----------------------------
    # Plot ROC Curve
    # -----------------------------
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig("reports/figures/roc_curve.png", bbox_inches="tight")
    plt.close()

    # -----------------------------
    # Plot Feature Importances
    # -----------------------------
    print("\n📊 Plotting feature importances ...")
    try:
        feature_importances = model.feature_importances_
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(feature_importances)), feature_importances)
        plt.title("Feature Importances")
        plt.xlabel("Feature Index")
        plt.ylabel("Importance")
        plt.savefig("reports/figures/feature_importance.png", bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"⚠️ Could not plot feature importances: {e}")

    # -----------------------------
    # SHAP Explanation
    # -----------------------------
    print("\n🧠 Computing SHAP values (using safe SHAP Explainer)...")
    try:
        # Take a sample for faster SHAP computation
        sample_size = min(1000, len(X_test))
        X_sample = X_test[:sample_size]

        # Use SHAP Explainer safely
        explainer = shap.Explainer(model.predict, X_sample)
        shap_values = explainer(X_sample)

        shap.summary_plot(shap_values, X_sample, show=False)
        plt.savefig("reports/figures/shap_summary.png", bbox_inches="tight")
        plt.close()
        print("✅ SHAP summary plot saved to reports/figures/shap_summary.png")

    except Exception as e:
        print(f"⚠️ SHAP explanation skipped due to error: {e}")

# -----------------------------
# Main Function
# -----------------------------
def main(model_path, test_path):
    print(f"\n🔍 Loading model from {model_path} ...")
    model = joblib.load(model_path)

    print(f"📂 Loading test data from {test_path} ...")
    arr = np.load(test_path)
    X_test, y_test = arr["X"], arr["y"]

    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model file (.pkl)")
    parser.add_argument("--test", required=True, help="Path to processed test data (.npz)")
    args = parser.parse_args()
    main(args.model, args.test)
