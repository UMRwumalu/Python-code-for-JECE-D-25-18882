# model.py
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor

def get_models():
    """
    Function: Return a dictionary containing 6 candidate models (naked running mode, default parameters)
    """
    models = {
        "XGB": XGBRegressor(verbosity=0, random_state=42),  # XGBoost
        "AdB": AdaBoostRegressor(random_state=42),   # AdB
        "RF": RandomForestRegressor(random_state=42),  # Random Forest
        "MLP": MLPRegressor(hidden_layer_sizes=(50,50),random_state=42),  # multilayer perceptron
        "KNN": KNeighborsRegressor(),  # k-nearest neighbors
        "RR": Ridge()  # Ridge Regression
    }
    return models

# ========== Test ==========
if __name__ == "__main__":
    models = get_models()
    print("Number of candidate models:", len(models))
    print("Model List:", list(models.keys()))
