z# train.py
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
    make_scorer,
)
from sklearn.ensemble import RandomForestRegressor

from data import load_and_split_data
from model import get_models  

def rmse_func(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_model(model, X_train, y_train, X_test, y_test):

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scoring = {
        "R2": make_scorer(r2_score),
        "RMSE": make_scorer(rmse_func, greater_is_better=False),
        "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
    }

    cv = cross_validate(model, X_train, y_train, cv=kf, scoring=scoring, n_jobs=-1, return_train_score=False)

    cv_means = {
        "R2": np.mean(cv["test_R2"]),
        "RMSE": -np.mean(cv["test_RMSE"]),
        "MAE": -np.mean(cv["test_MAE"]),
    }

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_metrics = {
        "R2": r2_score(y_test, y_pred),
        "RMSE": rmse_func(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
    }

    return cv_means, test_metrics

def main():
    # ================== load data ==================
    file_path = r"./Supplementary materials.xlsx"
    df, train_ids, test_ids, cluster_info = load_and_split_data(file_path)

    train_df = df[df["ID"].isin(train_ids)].reset_index(drop=True)
    test_df  = df[df["ID"].isin(test_ids)].reset_index(drop=True)

    target_cols = ["tox1", "tox2", "tox3", "tox4", "tox5", "tox6"]
    feature_cols = [c for c in df.columns if c not in ["ID", "SMILES"] + target_cols]
    X_train, X_test = train_df[feature_cols], test_df[feature_cols]

    # Check if the training/testing sets overlap
    overlap_ids = set(train_ids) & set(test_ids)
    if overlap_ids:
        raise ValueError(f"overlapping samples between the training and testing setsID: {overlap_ids}")
    else:
        print("The division check between the training and testing sets has been passed, and there is no overlap in the samples")

    base_model = get_models()["RandomForest"]["model"]
    param_grid = get_models()["RandomForest"]["params"]

    for target in target_cols:
        print("=" * 50)
        print(f">>> Output variables: {target}")
        y_train_target, y_test_target = train_df[target], test_df[target]

        # ================== Grid search parameter tuning ==================
        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=10,
            scoring='r2',
            n_jobs=-1
        )
        grid.fit(X_train, y_train_target)
        best_model = grid.best_estimator_
        print(f"Optimal hyperparameters: {grid.best_params_}")

        # ================== Ten fold cross validation evaluation training set ==================
        cv_means, test_metrics = evaluate_model(best_model, X_train, y_train_target, X_test, y_test_target)

        # Print training set with ten fold cross validation metrics
        print("Cross validation (10% off) average metrics:")
        print(f"  R² : {cv_means['R2']:.4f}")
        print(f"  RMSE: {cv_means['RMSE']:.4f}")
        print(f"  MAE : {cv_means['MAE']:.4f}")

        # Print test set metrics
        print("Test set metrics:")
        print(f"  R² : {test_metrics['R2']:.4f}")
        print(f"  RMSE: {test_metrics['RMSE']:.4f}")
        print(f"  MAE : {test_metrics['MAE']:.4f}")

if __name__ == "__main__":
    main()

