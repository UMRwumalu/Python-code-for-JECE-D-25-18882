# model.py
from sklearn.ensemble import RandomForestRegressor

def get_models():
    """
Return a dictionary containing the RandomForest model and its large-scale parameter space for random search of RandomizedSearchCV
    """
    models = {
        "RandomForest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200, 300, 400],
                "max_depth": [4, 8, 12, 16, None],
                "min_samples_split": [1, 2, 3, 4],
                "min_samples_leaf": [1, 2, 3, 4],
                "bootstrap": [True, False],
            }
        }
    }
    return models
