# pipeline.py
import logging
import traceback
import mlflow
import os

# Import functions from our new modules
import config # Imports config.py
import data
import models
import training
import mlflow_logging

# Configure basic logging for the application
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Main pipeline function to load data, train and evaluate models,
    and log results to MLflow.
    """
    logger.info("Starting ML pipeline...")
    try:
        # --- Setup Environment and MLflow ---
        # 1. Initialize DagsHub MLflow integration (calls dagshub.init)
        config.init_dagshub_mlflow() # <-- Call init here

        # 2. Setup general environment and set MLflow URI (mostly redundant now, but safe)
        output_dir = config.setup_environment()
        logger.info(f"Using output directory: {output_dir}")

        # --- Data Loading and Preprocessing ---
        df = data.load_data()
        X_train, X_test, y_train, y_test, quality_order, wine_feature_names = data.preprocess_data(df)

        # Basic validation after preprocessing
        if len(wine_feature_names) != X_train.shape[1]:
             logger.error(f"Feature name length ({len(wine_feature_names)}) does not match X_train features ({X_train.shape[1]})")
             raise AssertionError("Feature name mismatch")
        logger.info(f"Feature names ({len(wine_feature_names)}) match X_train shape ({X_train.shape[1]}).")

        # --- Model Configuration ---
        model_configs = models.get_model_configs()

        # --- MLflow Experiment Run ---
        # Set the MLflow experiment name - this should now work after init_dagshub_mlflow
        mlflow.set_experiment("Wine_Quality_Classification")

        # Start the main MLflow run
        active_run = mlflow.active_run()
        if active_run:
             logger.info(f"Using existing active MLflow run: {active_run.info.run_id}")
        else:
             logger.info("No active MLflow run found. Starting a new parent run.")
             active_run = mlflow.start_run(run_name="Wine_Model_Training_Run")
             logger.info(f"Started parent MLflow run: {active_run.info.run_id}")

        with active_run: # Use the active run context
            # Log parameters relevant to the overall run
            mlflow.log_param("dataset", "kaggle_wine_quality_classification_downloaded")
            mlflow.log_param("train_test_split_ratio", 0.8)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("quality_order", quality_order)
            mlflow.log_param("feature_names", wine_feature_names)

            # --- Train, Evaluate, and Log Each Model ---
            if not model_configs:
                 logger.warning("No model configurations found. Skipping training.")

            for config_item in model_configs:
                model_name = config_item["name"]

                logger.info(f"Processing model: {model_name}")
                try:
                    with mlflow.start_run(run_name=model_name, nested=True):
                         logger.info(f"Started nested MLflow run: {mlflow.active_run().info.run_id}")

                         model, report_dict_model = training.train_and_evaluate_model(
                             config_item,
                             X_train,
                             X_test,
                             y_train,
                             y_test,
                             quality_order
                         )

                         mlflow_logging.log_model_run(
                             model=model,
                             model_name=model_name,
                             model_params=config_item["params"],
                             report_dict=report_dict_model,
                             X_train=X_train,
                             X_test=X_test,
                             y_test=y_test,
                             wine_feature_names=wine_feature_names,
                             quality_order=quality_order,
                             output_dir=output_dir
                         )
                         logger.info(f"Completed nested MLflow run for {model_name}.")

                except Exception as model_run_err:
                    logger.error(f"An error occurred during the nested run for {model_name}: {str(model_run_err)}")
                    logger.error(traceback.format_exc())
                    if mlflow.active_run() and mlflow.active_run().info.run_name == model_name:
                         try:
                             mlflow.set_tag("status", "Failed")
                             mlflow.log_param("error_message", f"Nested run failed: {str(model_run_err)}")
                         except Exception as log_fail_err:
                             logger.error(f"Failed to log failure status for nested run: {log_fail_err}")
                    logger.warning(f"Skipping remaining processing for {model_name} due to error.")


        logger.info("Completed parent MLflow run.")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        logger.error(traceback.format_exc())
        if mlflow.active_run() and mlflow.active_run().info.run_uuid == active_run.info.run_uuid:
             logger.info(f"Logging failure status to parent MLflow run {active_run.info.run_id}")
             try:
                 mlflow.set_tag("status", "Pipeline Failed")
                 mlflow.log_param("pipeline_error_message", str(e))
             except Exception as log_err:
                  logger.error(f"Failed to log pipeline failure status: {log_err}")

        raise


if __name__ == "__main__":
    main()