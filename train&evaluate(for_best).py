# multi_rf_mipi_with_cas.py
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression
from data import load_and_split_data
from model import get_models
import shap
import os

# ---------- helper function ----------
def rmse_func(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Ten fold discount on training set CV+testing set metrics"""
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Ten fold cross validation of training set metrics
    y_pred_cv = cross_val_predict(model, X_train, y_train, cv=kf, n_jobs=-1)
    cv_metrics = {
        "R2": r2_score(y_train, y_pred_cv),
        "RMSE": rmse_func(y_train, y_pred_cv),
        "MAE": mean_absolute_error(y_train, y_pred_cv)
    }

    # Test set metrics
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    test_metrics = {
        "R2": r2_score(y_test, y_pred_test),
        "RMSE": rmse_func(y_test, y_pred_test),
        "MAE": mean_absolute_error(y_test, y_pred_test)
    }

    return cv_metrics, test_metrics

# ---------- main function----------
def main():
    start_time = time.time()

    # ---------- load data ----------
    file_path = r"./Supplementary materials.xlsx"
    df, train_ids, test_ids, _ = load_and_split_data(file_path)

    train_df = df[df["ID"].isin(train_ids)].reset_index(drop=True)
    test_df  = df[df["ID"].isin(test_ids)].reset_index(drop=True)

    target_cols = ["tox1","tox2","tox3","tox4","tox5","tox6"]
    feature_cols = [c for c in df.columns if c not in ["ID","SMILES"]+target_cols]
    X_train_full, X_test_full = train_df[feature_cols], test_df[feature_cols]

    models = get_models()

    # ---------- Multi output variable training and feature selection ----------
    trained_models = {}           # Store the trained model for each output variable
    selected_features_dict = {}   # Store the reserved features of each output variable
    X_train_all = {}              # Store the training set feature matrix for each output variable
    X_test_all  = {}              # Store the feature matrix of each output variable test set
    y_train_all = {}              # Store training labels for each output variable
    y_test_all  = {}              # Store test labels for each output variable
    train_cas_all = {}            # Store training set CAS
    test_cas_all  = {}            # Store test set CAS

    for target in target_cols:
        print("="*50)
        print(f">>> Output variables: {target}")

        y_train, y_test = train_df[target], test_df[target]
        base_model = models[target]  # Call the best hyperparameter RF in model

        # ------------------- MIPI feature selection -------------------
        print("\n--- MIPI feature selection ---")
        mi_scores = mutual_info_regression(X_train_full, y_train, random_state=42)
        mi_scores_series = pd.Series(mi_scores, index=X_train_full.columns)
        selected_columns = mi_scores_series[mi_scores_series >= 0.1].index.tolist()
        print(f"Retain the number of features: {len(selected_columns)}")

        X_train_mipi = X_train_full[selected_columns]
        X_test_mipi  = X_test_full[selected_columns]

        # ---------- evaluation model ----------
        cv_metrics, test_metrics = evaluate_model(base_model, X_train_mipi, y_train, X_test_mipi, y_test)
        print(f"Ten fold discount on CV index for training set | R²: {cv_metrics['R2']:.4f} | RMSE: {cv_metrics['RMSE']:.4f} | MAE: {cv_metrics['MAE']:.4f}")
        print(f"Test set metrics       | R²: {test_metrics['R2']:.4f} | RMSE: {test_metrics['RMSE']:.4f} | MAE: {test_metrics['MAE']:.4f}")

        # ---------- Save the trained model ----------
        base_model.fit(X_train_mipi, y_train)
        trained_models[target] = base_model
        selected_features_dict[target] = selected_columns
        X_train_all[target] = X_train_mipi
        X_test_all[target]  = X_test_mipi
        y_train_all[target] = y_train
        y_test_all[target]  = y_test

        # ---------- Save training and testing sets CAS ----------
        train_cas_all[target] = train_df["ID"].tolist()
        test_cas_all[target]  = test_df["ID"].tolist()

    elapsed = time.time() - start_time
    print(f"\nTotal time taken: {elapsed:.2f} second")


if __name__ == "__main__":
    main()
