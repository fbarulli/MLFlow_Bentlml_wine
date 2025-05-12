# mlflow_tracking/bentoml/service.py
import bentoml
from bentoml.io import JSON
from pydantic import BaseModel, Field
import pandas as pd
import logging
import traceback
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WineInput(BaseModel):
    data: list[list[float]] = Field(..., min_items=1)
    columns: list[str] = Field(
        default=[
            "fixed acidity", "volatile acidity", "citric acidity", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
            "pH", "sulphates", "alcohol", "Id"
        ],
        min_items=12,
        max_items=12
    )

    class Config:
        schema_extra = {
            "example": {
                "data": [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0]],
                "columns": [
                    "fixed acidity", "volatile acidity", "citric acidity", "residual sugar",
                    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                    "pH", "sulphates", "alcohol", "Id"
                ]
            }
        }

# Define the BentoML service. The name must match the service name used when serving.
svc = bentoml.Service("wine_quality_service")

# --- Load the model runner using the tag provided by the entrypoint ---
# This happens when the service definition is loaded/built by bentoml.
# The entrypoint.sh script is responsible for setting this environment variable.
BENTOML_MODEL_TAG = os.environ.get("BENTOML_SERVE_MODEL_TAG")

if not BENTOML_MODEL_TAG:
    # This indicates a configuration error in the entrypoint or deployment setup.
    logger.error("BENTOML_SERVE_MODEL_TAG environment variable not set. Cannot load model.")
    # Listing models here might help debugging, but the primary issue is the missing env var.
    logger.info("Available models in BentoML store:")
    try:
        available_models = bentoml.models.list()
        if available_models:
            for model in available_models:
                logger.info(f"  - {model.tag}")
        else:
             logger.info("  (No models found)")
    except Exception as list_err:
         logger.error(f"Error listing models: {str(list_err)}")

    # Raise an error here during service definition loading.
    # BentoML will catch this and the container startup will fail.
    raise ValueError("BENTOML_SERVE_MODEL_TAG not set. Please ensure the entrypoint script sets this.")
else:
    logger.info(f"Loading BentoML model with tag: {BENTOML_MODEL_TAG}")
    try:
        # Load the model specified by the tag
        model_to_serve = bentoml.models.get(BENTOML_MODEL_TAG)
        logger.info(f"Successfully loaded model: {model_to_serve.tag}")
    except bentoml.exceptions.NotFound:
        logger.error(f"BentoML model with tag '{BENTOML_MODEL_TAG}' not found in the local store.")
        logger.info("Available models in BentoML store:")
        try:
            available_models = bentoml.models.list()
            if available_models:
                for model in available_models:
                    logger.info(f"  - {model.tag}")
            else:
                logger.info("  (No models found)")
        except Exception as list_err:
            logger.error(f"Error listing models: {str(list_err)}")
        # Raise error during service loading if the specified model isn't found
        raise ValueError(f"BentoML model '{BENTOML_MODEL_TAG}' not found.")
    except Exception as e:
        logger.error(f"Error loading BentoML model with tag '{BENTOML_MODEL_TAG}': {str(e)}")
        logger.error(traceback.format_exc())
        raise # Re-raise other exceptions during loading


# Create the runner from the loaded model
# This prepares the model for inference, potentially loading it into memory/GPU
wine_runner = model_to_serve.to_runner()

# Add the runner to the BentoML service
# This makes the runner available to API functions
svc.add_runner(wine_runner)


@svc.api(input=JSON(pydantic_model=WineInput), output=JSON())
async def predict(input_data: WineInput):
    """
    Prediction API endpoint.
    Receives WineInput data, performs inference using the loaded model runner,
    and returns the predicted wine quality labels.
    """
    try:
        # Convert the input Pydantic model data to a pandas DataFrame
        # Ensure columns match the training data's expected feature order
        input_df = pd.DataFrame(input_data.data, columns=input_data.columns)
        
        # Use the runner to perform inference.
        # The runner is added to the service and managed by the BentoML runtime.
        # async_run is used because predict is an async api function
        predictions = await wine_runner.predict.async_run(input_df)

        logger.info(f"Generated {len(predictions)} predictions.")

        # Convert predicted labels (0, 1, 2) back to quality names ("low", "medium", "high")
        # This mapping should ideally be stored with the model or derived from training data.
        # For simplicity here, it's hardcoded.
        # Ensure the predicted values are integers before mapping
        quality_map = {0: "low", 1: "medium", 2: "high"}
        # Use a default value like "unknown" if a prediction falls outside 0, 1, 2
        predicted_labels = [quality_map.get(int(p), "unknown") for p in predictions] 

        # Return the predictions as a JSON object
        return {"predictions": predicted_labels}
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        logger.error(traceback.format_exc())
        # Re-raise the exception. BentoML will handle uncaught exceptions by
        # returning a 500 Internal Server Error response.
        raise