import os
import logging
import traceback
import mlflow.sklearn
import bentoml
from bentoml.io import PandasDataFrame
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables and set tracking URI
try:
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"].replace("mlflow+", "")
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI set to: {tracking_uri}")
except Exception as e:
    logger.error(f"Error setting MLflow tracking URI: {e}")
    logger.error(traceback.format_exc())
    raise

# Function to verify model registration and stage
def verify_model_registration(model_name, stage):
    client = MlflowClient()
    try:
        model_versions = client.search_model_versions(f"name='{model_name}'")
        for version in model_versions:
            if stage in version.aliases or version.current_stage == stage:
                logger.info(f"Model '{model_name}' found in stage/alias '{stage}' with version {version.version}")
                return True
        logger.error(f"No versions of model '{model_name}' found in stage/alias '{stage}'")
        return False
    except Exception as e:
        logger.error(f"Error verifying model registration for '{model_name}': {e}")
        return False

# Define model URIs
logistic_regression_model_uri = "models:/tracking-wine-logisticregression/Staging"
random_forest_model_uri = "models:/tracking-wine-randomforest/Staging"

# Model names and stage
model_name_lr = "tracking-wine-logisticregression"
model_name_rf = "tracking-wine-randomforest"
stage = "Staging"

# Verify model registration before loading
if not verify_model_registration(model_name_lr, stage):
    raise Exception(f"Model '{model_name_lr}' not found in stage/alias '{stage}'")
if not verify_model_registration(model_name_rf, stage):
    raise Exception(f"Model '{model_name_rf}' not found in stage/alias '{stage}'")

# Load the MLflow models
try:
    logistic_regression_model = mlflow.sklearn.load_model(logistic_regression_model_uri)
    random_forest_model = mlflow.sklearn.load_model(random_forest_model_uri)
    feature_names_lr = logistic_regression_model.feature_names_in_ if hasattr(logistic_regression_model, 'feature_names_in_') else None
    feature_names_rf = random_forest_model.feature_names_in_ if hasattr(random_forest_model, 'feature_names_in_') else None
    logger.info("MLflow models loaded successfully.")
except Exception as e:
    logger.error(f"Error loading MLflow models: {e}")
    logger.error(traceback.format_exc())
    raise

# Define a BentoML service
@bentoml.service(
    resources={"cpu": "2", "memory": "2Gi"},
    traffic={"timeout": 30},
)
class WineQualityService:
    @bentoml.api(input=PandasDataFrame(), output="numpy.ndarray")
    def predict_logistic_regression(self, df: pd.DataFrame) -> np.ndarray:
        try:
            if feature_names_lr and list(df.columns) != list(feature_names_lr):
                logger.warning("Input DataFrame columns do not match model feature names. Reordering...")
                df = df.reindex(columns=feature_names_lr)
            return logistic_regression_model.predict(df)
        except Exception as e:
            logger.error(f"Error predicting with Logistic Regression model: {e}")
            logger.error(traceback.format_exc())
            raise

    @bentoml.api(input=PandasDataFrame(), output="numpy.ndarray")
    def predict_random_forest(self, df: pd.DataFrame) -> np.ndarray:
        try:
            if feature_names_rf and list(df.columns) != list(feature_names_rf):
                logger.warning("Input DataFrame columns do not match model feature names. Reordering...")
                df = df.reindex(columns=feature_names_rf)
            return random_forest_model.predict(df)
        except Exception as e:
            logger.error(f"Error predicting with Random Forest model: {e}")
            logger.error(traceback.format_exc())
            raise