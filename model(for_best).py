# model.py
from sklearn.ensemble import RandomForestRegressor

def get_models():
    """
    Return a dictionary containing the best RandomForestRegressor model corresponding to each output variable
    """

    models = {
        "tox1": RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=2,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        "tox2": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ),
        "tox3": RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ),
        "tox4": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ),
        "tox5": RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=2,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        "tox6": RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ),
    }

    return models
