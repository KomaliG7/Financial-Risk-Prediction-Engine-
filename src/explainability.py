# src/explainability.py
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

def explain(model_path='models/xgb_baseline.pkl', processed='data/processed/train.npz', max_display=20):
    clf = joblib.load(model_path)
    arr = np.load(processed)
    X_train = arr['X_train']
    # Use TreeExplainer for XGBoost
    explainer = shap.TreeExplainer(clf)
    # For speed, sample subset
    sample = X_train if X_train.shape[0] <= 5000 else X_train[np.random.choice(X_train.shape[0], 5000, replace=False)]
    shap_vals = explainer.shap_values(sample)
    # summary plot
    shap.summary_plot(shap_vals, sample, show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig('models/shap_summary.png', bbox_inches='tight')
    # Dependence plot for top feature index 0 (replace with actual index after inspecting)
    try:
        shap.dependence_plot(0, shap_vals, sample, show=False)
        plt.savefig('models/shap_dependence_0.png', bbox_inches='tight')
    except Exception as e:
        print("Dependence plot error:", e)

if __name__ == "__main__":
    explain()
