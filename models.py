# models.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import logging

logger = logging.getLogger(__name__)

def get_model_configs():
    """Defines and returns a list of model configurations to be trained."""
    model_configs = []

    # Configuration for Logistic Regression
    model_configs.append({
        "name": "LogisticRegression",
        "class": LogisticRegression,
        "params": {
            "solver": "lbfgs",
            "max_iter": 10000,
            "random_state": 8888,
            "class_weight": "balanced", # Useful for imbalanced datasets
            "penalty": "l2",
            "C": 0.1 # Regularization parameter
        }
    })

    # Configuration for Random Forest Classifier
    model_configs.append({
        "name": "RandomForest",
        "class": RandomForestClassifier,
        "params": {
            "n_estimators": 30, # Number of trees
            "max_depth": 3 # Maximum depth of trees
        }
    })

    logger.info(f"Defined {len(model_configs)} model configurations.")
    return model_configs