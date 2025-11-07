# pdp_ice_tox1_full.py
import joblib
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Use the back end of the displayable window.
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay
import shap

# ===== Function of drawing PDP+ICE diagram =====
def plot_pdp_ice_inverse_times_new_roman(model, X_train_scaled, feature, target_name, feature_mu, feature_sigma):
    """
    Draw PDP+ICE diagram (abscissa inverse standardization), and the text is Times New Roman.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    display = PartialDependenceDisplay.from_estimator(
        model,
        X_train_scaled,
        features=[feature],
        kind="both",   # PDP + ICE
        ax=ax
    )

    # Horizontal axis inverse standardization
    xticks = ax.get_xticks()
    xticks_original = xticks * feature_sigma + feature_mu
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.2f}" for x in xticks_original], fontname="Times New Roman", fontweight="bold")

    # Horizontal and vertical axis labels, titles
    ax.set_xlabel(f"{feature} (original scale)", fontname="Times New Roman", fontweight="bold")
    ax.set_ylabel("Predicted value", fontname="Times New Roman", fontweight="bold")
    ax.set_title(f"PDP + ICE for Fish(96h LC50) - {feature}",
                 fontname="Times New Roman", fontweight="bold")

    # Scale font
    for label in ax.get_xticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontweight("bold")

    # Legend font
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontname("Times New Roman")
            text.set_fontweight("bold")

    plt.tight_layout()
    plt.show()

# ===== main function =====
def main():
    # Loading models and data
    data = joblib.load(
        r"./rf_models_mipi_all.pkl"
    )
    trained_models = data['models']
    X_train_all = data['X_train']

    # Fetch TOX1 data
    model = trained_models['tox1']
    X_train = X_train_all['tox1']

    # ===== Standardized training set =====
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    feature_mu = scaler.mean_
    feature_sigma = scaler.scale_

    # Restore the standardized training set to a DataFrame and maintain column names
    import pandas as pd
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)

    # ===== Calculate the top 12 SHAP features =====
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_scaled_df)
    shap_importance = np.abs(shap_values).mean(axis=0)
    feature_names = np.array(X_train.columns)
    idx_sorted = np.argsort(shap_importance)[::-1][:12]
    top_features = feature_names[idx_sorted]

    print("\n--- tox1 (fishLC50) The top 12 important features ---")
    for i, feat in enumerate(top_features, 1):
        print(f"{i:02d}. {feat}")



    # ===== Generate PDP+ICE diagram =====
    for i, feat in enumerate(top_features, 1):
        feat_idx = list(X_train.columns).index(feat)
        mu = feature_mu[feat_idx]
        sigma = feature_sigma[feat_idx]
        print(f"Displaying PDP+ICE: {i:02d} - {feat}")
        plot_pdp_ice_inverse_times_new_roman(
            model,
            X_train_scaled_df,
            feat,
            target_name="tox1 (fishLC50)",
            feature_mu=mu,
            feature_sigma=sigma
        )

    print("\n✅ All 12 features of tox1's PDP+ICE maps have been displayed！")

if __name__ == "__main__":
    main()
