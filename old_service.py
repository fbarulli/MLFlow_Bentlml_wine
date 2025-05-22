import os
import logging
import traceback
import warnings
import mlflow.sklearn
import bentoml
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables and set tracking URI
try:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        logger.error("MLFLOW_TRACKING_URI environment variable is not set!")
        logger.error("Make sure your .env file exists and contains MLFLOW_TRACKING_URI")
        raise ValueError("MLFLOW_TRACKING_URI not found in environment variables")

    # Clean the URI (remove mlflow+ prefix if present)
    clean_uri = tracking_uri.replace("mlflow+", "")
    mlflow.set_tracking_uri(clean_uri)
    logger.info(f"MLflow tracking URI set to: {clean_uri}")

    # Set authentication if available
    username = os.environ.get("MLFLOW_TRACKING_USERNAME")
    password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
    token = os.environ.get("MLFLOW_TRACKING_TOKEN")

    if username and password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password
        logger.info(f"MLflow authentication set for user: {username}")
    elif token:
        os.environ["MLFLOW_TRACKING_TOKEN"] = token
        logger.info("MLflow token authentication set")

except Exception as e:
    logger.error(f"Error setting MLflow tracking URI: {e}")
    logger.error(traceback.format_exc())
    raise

def safe_load_model(model_uri, model_name):
    """
    Safely load a model with version compatibility handling
    """
    try:
        logger.info(f"Attempting to load {model_name} from {model_uri}")

        # Try loading with warnings suppressed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = mlflow.sklearn.load_model(model_uri)

        logger.info(f"✅ Successfully loaded {model_name}")
        return model, None

    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Failed to load {model_name}: {error_msg}")

        # Check if it's a version compatibility issue
        if "incompatible dtype" in error_msg or "version" in error_msg.lower():
            logger.warning(f"Version compatibility issue detected for {model_name}")
            return None, f"Version compatibility issue: {error_msg}"
        else:
            return None, f"Loading error: {error_msg}"

def get_model_uri(model_name, preferred_stage="Staging"):
    """
    Get the model URI, preferring the specified stage but falling back to latest version.
    """
    client = MlflowClient()

    try:
        model_versions = client.search_model_versions(f"name='{model_name}'")

        if not model_versions:
            raise Exception(f"No versions found for model '{model_name}'")

        # Check for preferred stage/alias
        for version in model_versions:
            if preferred_stage in version.aliases or version.current_stage == preferred_stage:
                logger.info(f"Found model '{model_name}' in {preferred_stage}: version {version.version}")
                return f"models:/{model_name}/{preferred_stage}", version.version

        # Fallback to latest version
        logger.warning(f"Model '{model_name}' not found in {preferred_stage}. Using latest version.")

        # Sort by version number (assuming they are numeric)
        try:
            sorted_versions = sorted(model_versions, key=lambda x: int(x.version), reverse=True)
        except ValueError:
            # If versions are not numeric, sort by creation timestamp
            sorted_versions = sorted(model_versions, key=lambda x: x.creation_timestamp, reverse=True)

        latest_version = sorted_versions[0]
        logger.info(f"Using latest version {latest_version.version} for model '{model_name}')")
        return f"models:/{model_name}/{latest_version.version}", latest_version.version

    except Exception as e:
        logger.error(f"Error getting model URI for '{model_name}': {e}")
        raise

def safe_load_model(model_uri, model_name):
    """
    Safely load a model with version compatibility handling
    """
    try:
        logger.info(f"Attempting to load {model_name} from {model_uri}")

        # Try loading with warnings suppressed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = mlflow.sklearn.load_model(model_uri)

        logger.info(f"✅ Successfully loaded {model_name}")
        return model, None

    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Failed to load {model_name}: {error_msg}")

        # Check if it's a version compatibility issue
        if "incompatible dtype" in error_msg or "version" in error_msg.lower():
            logger.warning(f"Version compatibility issue detected for {model_name}")
            return None, f"Version compatibility issue: {error_msg}"
        else:
            return None, f"Loading error: {error_msg}"

def get_model_uri(model_name, preferred_stage="Staging"):
    """
    Get the model URI, preferring the specified stage but falling back to latest version.
    """
    client = MlflowClient()

    try:
        model_versions = client.search_model_versions(f"name='{model_name}'")

        if not model_versions:
            raise Exception(f"No versions found for model '{model_name}'")

        # Check for preferred stage/alias
        for version in model_versions:
            if preferred_stage in version.aliases or version.current_stage == preferred_stage:
                logger.info(f"Found model '{model_name}' in {preferred_stage}: version {version.version}")
                return f"models:/{model_name}/{preferred_stage}", version.version

        # Fallback to latest version
        logger.warning(f"Model '{model_name}' not found in {preferred_stage}. Using latest version.")

        # Sort by version number (assuming they are numeric)
        try:
            sorted_versions = sorted(model_versions, key=lambda x: int(x.version), reverse=True)
        except ValueError:
            # If versions are not numeric, sort by creation timestamp
            sorted_versions = sorted(model_versions, key=lambda x: x.creation_timestamp, reverse=True)

        latest_version = sorted_versions[0]
        logger.info(f"Using latest version {latest_version.version} for model '{model_name}')")
        return f"models:/{model_name}/{latest_version.version}", latest_version.version

    except Exception as e:
        logger.error(f"Error getting model URI for '{model_name}': {e}")
        raise

# Model names
model_name_lr = "tracking-wine-logisticregression"
model_name_rf = "tracking-wine-randomforest"
stage = "Staging"

# Get model URIs (with fallback to latest version)
try:
    logistic_regression_model_uri, lr_version = get_model_uri(model_name_lr, stage)
    random_forest_model_uri, rf_version = get_model_uri(model_name_rf, stage)

    logger.info(f"Using Logistic Regression model URI: {logistic_regression_model_uri}")
    logger.info(f"Using Random Forest model URI: {random_forest_model_uri}")

except Exception as e:
    logger.error(f"Failed to get model URIs: {e}")
    raise

# Try to load the MLflow models with error handling
logistic_regression_model = None
random_forest_model = None
lr_error = None
rf_error = None

logger.info("Loading models with version compatibility handling...")

# Load Logistic Regression model
logistic_regression_model, lr_error = safe_load_model(logistic_regression_model_uri, "Logistic Regression")

# Load Random Forest model
random_forest_model, rf_error = safe_load_model(random_forest_model_uri, "Random Forest")

# Get feature names if models loaded successfully
feature_names_lr = None
feature_names_rf = None

if logistic_regression_model:
    feature_names_lr = getattr(logistic_regression_model, 'feature_names_in_', None)
    logger.info(f"LR model features: {feature_names_lr}")

if random_forest_model:
    feature_names_rf = getattr(random_forest_model, 'feature_names_in_', None)
    logger.info(f"RF model features: {feature_names_rf}")


# Check if at least one model loaded successfully
if not logistic_regression_model and not random_forest_model:
    logger.error("❌ No models could be loaded successfully!")
    logger.error(f"LR Error: {lr_error}")
    logger.error(f"RF Error: {rf_error}")
    logger.error("\nPossible solutions:")
    logger.error("1. Downgrade scikit-learn: pip install scikit-learn==1.2.2")
    logger.error("2. Retrain models with current scikit-learn version")
    logger.error("3. Use MLflow model serving instead of direct sklearn loading")
    raise RuntimeError("Cannot load any models due to version compatibility issues")

logger.info(f"Model loading summary:")
logger.info(f"✅ Logistic Regression: {'Loaded' if logistic_regression_model else 'Failed'}")
logger.info(f"✅ Random Forest: {'Loaded' if random_forest_model else 'Failed'}")

# Define a BentoML service
@bentoml.service(
    resources={"cpu": "2", "memory": "2Gi"},
    traffic={"timeout": 30},
)
class WineQualityService:

    def __init__(self):
        # Store model info and availability
        self.lr_version = lr_version if logistic_regression_model else None
        self.rf_version = rf_version if random_forest_model else None
        self.lr_available = logistic_regression_model is not None
        self.rf_available = random_forest_model is not None
        self.lr_error = lr_error
        self.rf_error = rf_error

        logger.info(f"WineQualityService initialized")
        logger.info(f"LR available: {self.lr_available} (v{self.lr_version})")
        logger.info(f"RF available: {self.rf_available} (v{self.rf_version})")

    # Use pandas.DataFrame and numpy.ndarray type hints
    @bentoml.api()
    def predict_logistic_regression(self, df: pd.DataFrame) -> np.ndarray:
        """Predict wine quality using Logistic Regression model."""
        if not self.lr_available:
            raise RuntimeError(f"Logistic Regression model not available. Error: {self.lr_error}")

        try:
            logger.info(f"Received prediction request with shape: {df.shape}")

            # Validate and reorder columns if needed
            if feature_names_lr is not None:
                if list(df.columns) != list(feature_names_lr):
                    logger.warning("Reordering input columns to match model expectations")
                    missing_features = set(feature_names_lr) - set(df.columns)
                    if missing_features:
                        raise ValueError(f"Missing required features: {missing_features}")
                    df = df.reindex(columns=feature_names_lr)

            predictions = logistic_regression_model.predict(df)
            logger.info(f"Generated {len(predictions)} predictions using LR model v{self.lr_version}")
            return predictions

        except Exception as e:
            logger.error(f"Error predicting with Logistic Regression model: {e}")
            raise

    # Use pandas.DataFrame and numpy.ndarray type hints
    @bentoml.api()
    def predict_random_forest(self, df: pd.DataFrame) -> np.ndarray:
        """Predict wine quality using Random Forest model."""
        if not self.rf_available:
            raise RuntimeError(f"Random Forest model not available. Error: {self.rf_error}")

        try:
            logger.info(f"Received prediction request with shape: {df.shape}")

            # Validate and reorder columns if needed
            if feature_names_rf is not None:
                if list(df.columns) != list(feature_names_rf):
                    logger.warning("Reordering input columns to match model expectations")
                    missing_features = set(feature_names_rf) - set(df.columns)
                    if missing_features:
                        raise ValueError(f"Missing required features: {missing_features}")
                    df = df.reindex(columns=feature_names_rf)

            predictions = random_forest_model.predict(df)
            logger.info(f"Generated {len(predictions)} predictions using RF model v{self.rf_version}")
            return predictions

        except Exception as e:
            logger.error(f"Error predicting with Random Forest model: {e}")
            raise

    # Use pandas.DataFrame input and dict return type hint (BentoML handles dict -> JSON)
    @bentoml.api()
    def predict_best_available(self, df: pd.DataFrame) -> dict:
        """Get predictions from the best available model."""
        try:
            # Prefer logistic regression if available, fallback to random forest
            if self.lr_available:
                # Pass df as pd.DataFrame
                predictions = self.predict_logistic_regression(df.copy())
                model_used = "logistic_regression"
                version_used = self.lr_version
            elif self.rf_available:
                # Pass df as pd.DataFrame
                predictions = self.predict_random_forest(df.copy())
                model_used = "random_forest"
                version_used = self.rf_version
            else:
                raise RuntimeError("No models are available for prediction")

            # predictions is a numpy.ndarray, convert to list for JSON
            return {
                "predictions": predictions.tolist(),
                "model_used": model_used,
                "model_version": version_used,
                "available_models": {
                    "logistic_regression": self.lr_available,
                    "random_forest": self.rf_available
                }
            }
        except Exception as e:
            logger.error(f"Error in predict_best_available: {e}")
            raise

    @bentoml.api()
    def health_check(self) -> dict:
        """Health check endpoint to see model availability."""
        return {
            "status": "healthy",
            "models": {
                "logistic_regression": {
                    "available": self.lr_available,
                    "version": self.lr_version,
                    "error": self.lr_error if not self.lr_available else None
                },
                "random_forest": {
                    "available": self.rf_available,
                    "version": self.rf_version,
                    "error": self.rf_error if not self.rf_available else None
                }
            }
        }