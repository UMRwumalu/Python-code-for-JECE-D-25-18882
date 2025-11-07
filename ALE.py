# ale_tox1_full_fixed.py
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import shap
from alibi.explainers import ALE

# ===== Draw ALE graph function =====
def plot_ale(feature_values_scaled, ale_values, feature_name, mu, sigma, target_name):
    """
    feature_values_scaled: Standardized Z-score values for features
    ale_values: The cumulative effect of ALE interpreter output
    mu, sigma: The mean and standard deviation of this feature are used for inverse normalization
    """
    # Horizontal axis inverse standardization
    x_original = feature_values_scaled * sigma + mu

    plt.figure(figsize=(6, 4))
    plt.plot(x_original, ale_values, color='royalblue', lw=2, label='ALE')

    # Horizontal and vertical coordinates and title
    plt.xlabel(f"{feature_name} (original scale)", fontname="Times New Roman", fontweight="bold")
    plt.ylabel("Accumulated Local Effect", fontname="Times New Roman", fontweight="bold")
    plt.title(f"ALE for Fish(96h LC50) - {feature_name}", fontname="Times New Roman", fontweight="bold")

    # Coordinate scale font
    plt.xticks(fontname="Times New Roman", fontweight="bold")
    plt.yticks(fontname="Times New Roman", fontweight="bold")

    # Legend
    plt.legend(loc="upper left", prop={'family':'Times New Roman', 'weight':'bold'})

    plt.tight_layout()
    plt.show()


# ===== main program =====
def main():
    # Loading models and data
    data = joblib.load(
        r"./rf_models_mipi_all.pkl"
    )
    trained_models = data['models']
    X_train_all = data['X_train']

    # tox1 Model and Training Set
    model = trained_models['tox1']
    X_train = X_train_all['tox1']

    # Standardized training set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    mu = scaler.mean_
    sigma = scaler.scale_
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)

    # Top 12 features of SHAP ranking
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_scaled_df)
    shap_importance = np.abs(shap_values).mean(axis=0)
    feature_names = np.array(X_train.columns)
    idx_sorted = np.argsort(shap_importance)[::-1][:12]
    top_features = feature_names[idx_sorted]

    print("\n--- tox1 (fishLC50) Top 12 important features ---")
    for i, feat in enumerate(top_features, 1):
        print(f"{i:02d}. {feat}")

    # ===== convert to numpy to avoid KeyError =====
    X_train_scaled_np = X_train_scaled_df.values

    # ===== Generate ALE image =====
    for feat in top_features:
        feat_idx = list(X_train.columns).index(feat)
        mu_feat = mu[feat_idx]
        sigma_feat = sigma[feat_idx]

        # Using the Alibi ALE interpreter (new version interface)
        ale_exp = ALE(model.predict, feature_names=X_train.columns.tolist())
        explanation = ale_exp.explain(X_train_scaled_np, features=[feat_idx])

        ale_values = explanation.ale_values[0].ravel()
        feature_values_scaled = explanation.feature_values[0]

        print(f"Display ALE image: {feat}")
        plot_ale(feature_values_scaled, ale_values, feat, mu_feat, sigma_feat, target_name="Fish(96h LC50)")

    print("\n✅ tox1 The first 12 feature ALE images have all been displayed！")


if __name__ == "__main__":
    main()
