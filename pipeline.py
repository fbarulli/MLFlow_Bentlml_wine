# pipeline.py
import logging
import traceback
import mlflow
import os

# Import functions from our new modules
import config # Imports config.py (contains init_dagshub_mlflow, setup_environment)
import data
import models
import training
import mlflow_logging

# Configure basic logging for the application
# This will catch logs from all imported modules unless they override it
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__) # Logger for this specific module


def main():
    """
    Main pipeline function to load data, train and evaluate models,
    and log results to MLflow.
    """
    logger.info("Starting ML pipeline...")
    # No need to assign active_run to None initially
    try:
        # --- Setup Environment and MLflow ---
        config.init_dagshub_mlflow()
        output_dir = config.setup_environment()
        logger.info(f"Using output directory: {output_dir}")

        # --- Data Loading and Preprocessing ---
        logger.info("Loading and preprocessing data...")
        df = data.load_data()
        X_train, X_test, y_train, y_test, quality_order, wine_feature_names = data.preprocess_data(df)
        logger.info(f"Data loaded and preprocessed. Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        if len(wine_feature_names) != X_train.shape[1]:
             logger.error(f"Feature name length ({len(wine_feature_names)}) does not match X_train features ({X_train.shape[1]})")
             raise AssertionError("Feature name mismatch")
        logger.info(f"Feature names ({len(wine_feature_names)}) match X_train shape ({X_train.shape[1]}).")

        # --- Model Configuration ---
        logger.info("Getting model configurations...")
        model_configs = models.get_model_configs()
        logger.info(f"Found {len(model_configs)} model configurations.")

        # --- MLflow Experiment Run Management ---
        experiment_name = "Wine_Quality_Classification"
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow Experiment set to: '{experiment_name}'")

        # Start the main MLflow run (Parent run)
        # The 'with' statement ensures the run stays active for the duration of the block
        # and automatically ends it (or logs error status) upon exiting the block.
        logger.info("Starting parent MLflow run.")
        # Use a descriptive run name
        with mlflow.start_run(run_name="Wine_Model_Training_Pipeline_Run") as parent_run:
            logger.info(f"Parent MLflow run started: {parent_run.info.run_id}")
            logger.info(f"🏃 View run at: {parent_run.info.artifact_uri}") # Note: artifact_uri is usually the link
            logger.info(f"🧪 View experiment at: {mlflow.get_tracking_uri()}#/experiments/{parent_run.info.experiment_id}")


            # --- Log Global Parameters for the Parent Run ---
            logger.info("Logging global parameters to parent run...")
            mlflow.log_param("dataset", "kaggle_wine_quality_classification_downloaded")
            mlflow.log_param("train_test_split_ratio", 0.8)
            mlflow.log_param("random_state", 42)
            # Log lists as JSON strings or convert to params if short
            # mlflow.log_param("quality_order", quality_order) # Cannot log list directly
            # mlflow.log_param("feature_names", wine_feature_names) # Cannot log list directly
            # Alternative: log as artifact
            # try:
            #      with open(os.path.join(output_dir, "quality_order.json"), "w") as f: json.dump(quality_order, f)
            #      mlflow.log_artifact(os.path.join(output_dir, "quality_order.json"))
            #      with open(os.path.join(output_dir, "feature_names.json"), "w") as f: json.dump(wine_feature_names, f)
            #      mlflow.log_artifact(os.path.join(output_dir, "feature_names.json"))
            # except Exception as log_artifact_err:
            #      logger.warning(f"Failed to log list parameters as artifacts: {log_artifact_err}")

            logger.info("Global parameters logged to parent run.")

            # --- Train, Evaluate, and Log Each Model (Inside the Parent Run Context) ---
            if not model_configs:
                 logger.warning("No model configurations found. Skipping training.")

            for config_item in model_configs:
                model_name = config_item.get("name", "UnknownModel")

                logger.info(f"Processing model: {model_name}")
                # Use a try/except block around each model's processing
                # so failure of one model doesn't stop the entire pipeline
                try:
                    nested_run = None # Initialize nested_run for try/except scope
                    # Start a nested MLflow run for each model
                    # This is inside the 'with parent_run:' block, so it's correctly nested.
                    with mlflow.start_run(run_name=model_name, nested=True) as nested_run:
                         logger.info(f"Started nested MLflow run: {nested_run.info.run_id} for {model_name}")

                         # Call the training and evaluation logic from training.py
                         model, report_dict_model = training.train_and_evaluate_model(
                             config_item,
                             X_train,
                             X_test,
                             y_train,
                             y_test,
                             quality_order
                         )

                         # Call the logging logic from mlflow_logging.py
                         # This function will make the actual mlflow.log_* calls while inside this 'with nested_run:' block
                         mlflow_logging.log_model_run(
                             model=model,
                             model_name=model_name,
                             model_params=config_item.get("params", {}), # Pass parameters safely
                             report_dict=report_dict_model,
                             X_train=X_train, # Pass data for potential artifact logging (e.g., feature importance calc in log_model_run)
                             X_test=X_test,
                             y_test=y_test,
                             wine_feature_names=wine_feature_names,
                             quality_order=quality_order,
                             output_dir=output_dir # Pass output directory for temporary artifact storage
                         )
                         logger.info(f"Completed nested MLflow run: {nested_run.info.run_id} for {model_name}.")

                except Exception as model_run_err:
                    # This catches errors specifically during one model's training/logging
                    logger.error(f"An error occurred during the nested run for {model_name}: {str(model_run_err)}")
                    print(f"An error occurred during the nested run for {model_name}: {str(model_run_err)}")
                    logger.error(traceback.format_exc())
                    print("Exception caught in pipeline.py!")
                    # Attempt to log failure to the *current* active run (which should be the failed nested one)
                    if nested_run and mlflow.active_run() and mlflow.active_run().info.run_id == nested_run.info.run_id:
                         logger.warning(f"Attempting to log failure status to nested run {nested_run.info.run_id}.")
                         try:
                             mlflow.set_tag("status", "Failed") # Set a tag indicating failure
                             # Log error message as a parameter for visibility in MLflow UI
                             mlflow.log_param("error_message", f"Nested run failed: {str(model_run_err)}")
                             logger.warning(f"Logged failure status for nested run {nested_run.info.run_id}.")
                         except Exception as log_fail_err:
                             logger.error(f"Failed to log failure status for nested run: {log_fail_err}")
                    else:
                         # Fallback if nested_run wasn't even successfully started or assigned
                         logger.warning(f"Error occurred processing {model_name}, but could not log failure to a specific nested run.")
                    logger.warning(f"Skipping remaining processing for {model_name} due to error.")
                    raise model_run_err
                    # Continue the loop to try the next model

            # --- End of loop over model_configs ---

        # --- End of 'with parent_run:' block ---
        logger.info(f"Completed parent MLflow run: {parent_run.info.run_id}")
        logger.info("Main pipeline logic finished successfully.")


    except Exception as e:
        # This catches any exceptions raised within the main() function that weren't specifically caught
        # and handled within the model loop (e.g., errors during data loading, setup, or after the loop).
        # Because the parent run is managed by a 'with' statement, if an exception occurs
        # *within* the 'with' block, the run will automatically be marked as failed.
        # This outer except block is primarily for errors *before* the parent run starts
        # or specific errors that might escape the nested try/excepts unexpectedly.
        logger.error(f"Pipeline execution failed due to an error: {str(e)}")
        logger.error(traceback.format_exc())

        # If an error happens *after* the parent run starts, the 'with parent_run:'
        # should handle logging failure to the parent run automatically.
        # This check might be redundant but could help if 'active_run' somehow isn't the parent run
        current_active_run = mlflow.active_run()
        if current_active_run:
             logger.warning(f"Exception occurred while MLflow run {current_active_run.info.run_id} was active.")
             # The 'with' statement handling the parent run should mark it failed.
             # We don't need to log failure *again* here typically.

        # Re-raise the exception so the calling script/process knows execution failed.
        raise


if __name__ == "__main__":
    main()