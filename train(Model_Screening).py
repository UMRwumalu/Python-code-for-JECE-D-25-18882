# print_metrics_all_targets_rawSD.py
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
    make_scorer,
)

from data import load_and_split_data
from model import get_models


def rmse_func(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    # ================== Load data ==================
    file_path = r"./Supplementary materials.xlsx"
    df, train_ids, test_ids, cluster_info = load_and_split_data(file_path)

    # ================== Build training/testing set ==================
    train_df = df[df["ID"].isin(train_ids)].reset_index(drop=True)
    test_df  = df[df["ID"].isin(test_ids)].reset_index(drop=True)

    # ================== target column ==================
    target_cols = ["tox1","tox2","tox3","tox4","tox5","tox6"]
    feature_cols = [c for c in df.columns if c not in ["ID", "SMILES"] + target_cols]

    X_train, X_test = train_df[feature_cols], test_df[feature_cols]

    models = get_models()

    # ================== Loop each target ==================
    for target in target_cols:
        y_train, y_test = train_df[target], test_df[target]

        # ---------- OBS（original SD） ----------
        obs_train_sd = np.std(y_train, ddof=1)
        obs_test_sd = np.std(y_test, ddof=1)
        print("\n" + "="*60)
        print(f"target indicator: {target}")
        print("="*60)
        print(f"training set OBS SD = {obs_train_sd:.4f}")
        print(f"test set OBS SD = {obs_test_sd:.4f}")

        for name, model in models.items():
            print(f"\nmodel: {name}")

            # ---------- Ten fold cross validation ----------
            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            scoring = {
                "R2": make_scorer(r2_score),
                "RMSE": make_scorer(rmse_func, greater_is_better=False),
                "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
                "EV": make_scorer(explained_variance_score),
            }

            cv = cross_validate(model, X_train, y_train, cv=kf, scoring=scoring,
                                n_jobs=-1, return_train_score=False)

            # ---------- Training Set 10% off CV SD (Original SD) ----------
            y_train_pred_cv = cross_val_predict(model, X_train, y_train, cv=kf, n_jobs=-1)
            sd_train_cv = np.std(y_train_pred_cv, ddof=1)  # Directly use the original standard deviation

            print("Training set CV index (ten fold average):")
            print(f"  R2  = {np.mean(cv['test_R2']):.4f}")
            print(f"  RMSE = {-np.mean(cv['test_RMSE']):.4f}")
            print(f"  MAE  = {-np.mean(cv['test_MAE']):.4f}")
            print(f"  EV   = {np.mean(cv['test_EV']):.4f}")
            print(f"  SD   = {sd_train_cv:.4f} (original)")

            # ---------- Test Set SD (Original SD) ----------
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            sd_test = np.std(y_pred_test, ddof=1)  # Directly use the original standard deviation

            print("Test set metrics:")
            print(f"  R2  = {r2_score(y_test, y_pred_test):.4f}")
            print(f"  RMSE = {rmse_func(y_test, y_pred_test):.4f}")
            print(f"  MAE  = {mean_absolute_error(y_test, y_pred_test):.4f}")
            print(f"  EV   = {explained_variance_score(y_test, y_pred_test):.4f}")
            print(f"  SD   = {sd_test:.4f} (original)")


if __name__ == "__main__":
    main()
