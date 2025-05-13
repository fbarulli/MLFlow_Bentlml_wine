# mlflow_logging.py
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import pandas as pd
import logging
import traceback
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report # Needed for parsing report_dict structure

logger = logging.getLogger(__name__)

def get_pip_requirements(req_file_path):
    """Reads pip requirements from a file."""
    if not os.path.exists(req_file_path):
        logger.error(f"Requirements file not found at: {req_file_path}")
        # Do not raise error here, let the logging function decide if it's critical
        return None
    try:
        with open(req_file_path, 'r') as f:
            # Filter out empty lines and comments
            requirements = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        return requirements
    except Exception as e:
        logger.error(f"Error reading requirements file {req_file_path}: {e}")
        logger.error(traceback.format_exc())
        return None


def save_predictions(model, X_test, y_test, wine_feature_names, model_name, output_dir):
    """Saves predictions to a CSV file and logs it as an MLflow artifact."""
    try:
        predictions = model.predict(X_test)
        if isinstance(X_test, pd.DataFrame):
            result_df = X_test.copy()
        elif isinstance(X_test, np.ndarray):
             # Assuming X_test is a numpy array if not a DataFrame
             # Ensure feature names count matches array columns
             if wine_feature_names and len(wine_feature_names) == X_test.shape[1]:
                 result_df = pd.DataFrame(X_test, columns=wine_feature_names)
             else:
                 logger.warning("Feature names not provided or do not match X_test shape. Creating DataFrame without column names.")
                 result_df = pd.DataFrame(X_test)
        else:
             logger.error("X_test is not a DataFrame or NumPy array. Cannot save predictions.")
             return # Exit function if input format is unsupported


        result_df["actual_class"] = y_test
        result_df["predicted_class"] = predictions

        csv_filename = f"{model_name.lower()}_predictions.csv"
        csv_dir = os.path.join(output_dir, "predictions")
        csv_path = os.path.join(csv_dir, csv_filename)
        os.makedirs(csv_dir, exist_ok=True)

        result_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path="predictions")
        logger.info(f"Predictions saved to '{csv_path}' and logged as artifact for {model_name}")
        # Optional: Remove local file after logging if space is a concern
        # os.remove(csv_path)
        # os.rmdir(csv_dir) # Will fail if other files exist

    except Exception as e:
        logger.error(f"Failed to save and log predictions for {model_name}: {e}")
        logger.error(traceback.format_exc())
        # Do not re-raise, artifact logging failure shouldn't stop the run


def save_model_plots(model, model_name, wine_feature_names, quality_order, output_dir):
    """Generates and saves model-specific plots (e.g., coefficients, feature importance) and logs them as MLflow artifacts."""
    try:
        plot_output_dir = os.path.join(output_dir, "plots")
        os.makedirs(plot_output_dir, exist_ok=True)

        if model_name == "LogisticRegression":
            if hasattr(model, 'coef_') and model.coef_ is not None:
                coefs_per_class = model.coef_
                num_classes_in_model = coefs_per_class.shape[0]
                # Map class indices to quality names, handling potential mismatches
                class_labels = [quality_order[i] if i < len(quality_order) else f"Class {i}" for i in range(num_classes_in_model)]


                for i in range(num_classes_in_model):
                    # Ensure feature names match coefficient count for plotting
                    if wine_feature_names and len(wine_feature_names) == coefs_per_class.shape[1]:
                        plt.figure(figsize=(10, 6))
                        plt.bar(wine_feature_names, coefs_per_class[i])
                        plt.title(f"{model_name} Coefficients ({class_labels[i]})")
                        plt.xlabel("Features")
                        plt.ylabel("Coefficient Value")
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()

                        safe_class_label = class_labels[i].replace(' ', '_').replace('-', '_').lower()
                        plot_filename = f"logistic_coefficients_{safe_class_label}.png"
                        plot_path = os.path.join(plot_output_dir, plot_filename)
                        plt.savefig(plot_path)
                        mlflow.log_artifact(plot_path, artifact_path="plots")
                        plt.close() # Close the figure to free memory
                        logger.info(f"Plot saved to '{plot_path}' and logged for {model_name}")
                    else:
                         logger.warning(f"Feature name count ({len(wine_feature_names) if wine_feature_names else 0}) does not match coefficients count ({coefs_per_class.shape[1]}) for {model_name}, class {i}. Cannot plot coefficients.")

            else:
                 logger.warning(f"LogisticRegression model does not have coef_ attribute or it is None. Cannot plot coefficients.")


        elif model_name == "RandomForest":
            if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
                # Ensure feature names match importance count for plotting
                if wine_feature_names and len(wine_feature_names) == len(model.feature_importances_):
                    plt.figure(figsize=(10, 6))
                    importances = model.feature_importances_
                    sorted_idx = np.argsort(importances)[::-1]
                    plt.bar(np.array(wine_feature_names)[sorted_idx], importances[sorted_idx])
                    plt.title(f"{model_name} Feature Importance")
                    plt.xlabel("Features")
                    plt.ylabel("Importance")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()

                    plot_filename = "random_forest_feature_importance.png"
                    plot_path = os.path.join(plot_output_dir, plot_filename)
                    plt.savefig(plot_path)
                    mlflow.log_artifact(plot_path, artifact_path="plots")
                    plt.close() # Close the figure
                    logger.info(f"Plot saved to '{plot_path}' and logged for {model_name}")
                else:
                    logger.warning(f"Feature name count ({len(wine_feature_names) if wine_feature_names else 0}) does not match importance count ({len(model.feature_importances_)}) for {model_name}. Cannot plot feature importance.")

            else:
                logger.warning(f"{model_name} does not have feature_importances_ attribute or it is None.")

        else:
            logger.warning(f"Plotting logic not implemented for model type: {model_name}")

        # Optional: Clean up local plot files if space is a concern
        # import shutil
        # shutil.rmtree(plot_output_dir, ignore_errors=True)


    except Exception as e:
        logger.error(f"Failed to generate and log plots for {model_name}: {e}")
        logger.error(traceback.format_exc())
        # Do not re-raise, plotting failure shouldn't necessarily stop the run


from mlflow.tracking import MlflowClient
import mlflow.exceptions
import logging

logger = logging.getLogger(__name__)

def transition_model_to_staging(registered_model_name):
    """Transitions the latest version of a registered model to the 'Staging' alias."""
    try:
        client = MlflowClient()
        logger.info(f"Attempting to assign 'Staging' alias to latest version of '{registered_model_name}'")

        # Get the latest version
        latest_versions = client.search_model_versions(
            f"name='{registered_model_name}'",
            order_by=["version_number DESC"],
            max_results=1
        )

        if not latest_versions:
            logger.warning(f"No version found for registered model '{registered_model_name}'. Cannot set alias.")
            return

        latest_version = latest_versions[0]
        latest_version_number = latest_version.version
        logger.info(f"Latest version found is {latest_version_number} (Run ID: {latest_version.run_id}).")

        # Ensure version is a string for the API call
        latest_version_number_str = str(latest_version_number)

        # Check if there is an existing 'Staging' alias
        try:
            current_staging_version = client.get_model_version_by_alias(
                name=registered_model_name,
                alias="Staging"
            )
            if current_staging_version.version != latest_version_number_str:
                logger.info(f"Removing 'Staging' alias from version {current_staging_version.version}")
                client.delete_model_version_alias(
                    name=registered_model_name,
                    alias="Staging",
                    version=current_staging_version.version
                )
        except mlflow.exceptions.MlflowException as e:
            if "RESOURCE_DOES_NOT_EXIST" in str(e):
                logger.info(f"No existing 'Staging' alias found for {registered_model_name}.")
            else:
                logger.warning(f"Error checking existing 'Staging' alias: {e}")

        # Set the alias on the latest version
        client.set_registered_model_alias(
            name=registered_model_name,
            alias="Staging",
            version=latest_version_number_str
        )

        logger.info(f"Registered model '{registered_model_name}' version {latest_version_number_str} successfully aliased as 'Staging'.")

    except Exception as e:
        logger.error(f"Failed to set 'Staging' alias for model '{registered_model_name}': {e}")



def log_model_run(model, model_name, model_params, report_dict, X_train, X_test, y_test, wine_feature_names, quality_order, output_dir):
    """
    Logs model metrics, parameters, the model artifact, predictions, and plots
    to the current active MLflow run.
    """
    try:
        logger.info(f"Starting MLflow logging for {model_name}...")

        # Log metrics from the classification report
        try:
            if not isinstance(report_dict, dict) or not report_dict:
                 logger.warning(f"No valid report_dict provided for {model_name}. Skipping metric logging.")
            else:
                 metrics = {}
                 # Populate metrics dictionary (similar to original logic)
                 if 'accuracy' in report_dict: metrics["accuracy"] = report_dict["accuracy"]
                 if 'weighted avg' in report_dict and isinstance(report_dict['weighted avg'], dict):
                     metrics["weighted_avg_precision"] = report_dict["weighted avg"].get("precision")
                     metrics["weighted_avg_recall"] = report_dict["weighted avg"].get("recall")
                     metrics["weighted_avg_f1-score"] = report_dict["weighted avg"].get("f1-score")
                 if 'macro avg' in report_dict and isinstance(report_dict['macro avg'], dict):
                     metrics["macro_avg_precision"] = report_dict["macro avg"].get("precision")
                     metrics["macro_avg_recall"] = report_dict["macro avg"].get("recall")
                     metrics["macro_avg_f1-score"] = report_dict["macro avg"].get("f1-score")

                 # Log per-class metrics
                 for i, label_name in enumerate(quality_order):
                     label_key_float = str(float(i))
                     label_key_int = str(int(i))
                     label_key = None
                     if label_key_float in report_dict and isinstance(report_dict[label_key_float], dict):
                          label_key = label_key_float
                     elif label_key_int in report_dict and isinstance(report_dict[label_key_int], dict):
                          label_key = label_key_int

                     if label_key:
                         metrics[f"precision_{label_name}"] = report_dict[label_key].get("precision")
                         metrics[f"recall_{label_name}"] = report_dict[label_key].get("recall")
                         metrics[f"f1-score_{label_name}"] = report_dict[label_key].get("f1-score")
                         metrics[f"support_{label_name}"] = report_dict[label_key].get("support")


                 metrics = {k: v for k, v in metrics.items() if v is not None} # Filter out None values

                 if metrics:
                     mlflow.log_metrics(metrics)
                     logger.info(f"Metrics logged for {model_name}: {metrics}")


        except Exception as metric_err:
            logger.error(f"Failed to log metrics for {model_name}: {metric_err}")
            logger.error(traceback.format_exc())
            # Do not re-raise


        # Log model parameters
        if model_params:
             mlflow.log_params(model_params)
             logger.info(f"Parameters logged for {model_name}: {model_params}")


        mlflow.set_tag("Training Info", f"{model_name} model for Wine")
        logger.info(f"Tag 'Training Info' set for {model_name}.")

        # Infer signature and create input example
        signature = None
        input_example = None
        if hasattr(model, 'predict') and isinstance(X_train, (pd.DataFrame, np.ndarray)) and X_train.shape[0] > 0:
            try:
                # Use a small sample for input example and signature inference
                sample_size = min(100, X_train.shape[0]) if X_train.shape[0] > 1 else X_train.shape[0]
                if sample_size > 0:
                    # Use pd.DataFrame for sampling even if X_train is ndarray for consistency
                    input_example_df = pd.DataFrame(X_train).sample(sample_size, random_state=42) if sample_size > 1 else pd.DataFrame(X_train).head(1)
                    if not input_example_df.empty:
                        # Ensure the model's predict method can handle the sample (e.g., column names if needed)
                        # Pass column names if available, otherwise let infer_signature handle it
                        input_example_data_for_predict = input_example_df
                        if wine_feature_names and isinstance(X_train, pd.DataFrame): # Only if X_train was DF originally
                             input_example_data_for_predict.columns = wine_feature_names

                        signature = infer_signature(input_example_data_for_predict, model.predict(input_example_data_for_predict.copy()))
                        input_example = input_example_df.head(min(5, sample_size)) # Log only a few examples
                        logger.info(f"Signature inferred successfully for {model_name}.")
                    else:
                        logger.warning(f"Sampled input data is empty for signature/input example inference for {model_name}.")
                elif X_train.shape[0] == 0:
                    logger.warning(f"X_train is empty. Cannot infer signature or create input example for {model_name}.")
            except Exception as sig_err:
                logger.error(f"Failed to infer signature or create input example for {model_name}: {sig_err}")
                logger.error(traceback.format_exc())
                # Do not re-raise


        registered_model_name = f"tracking-wine-{model_name.lower()}"

        # Get pip requirements for logging with the model
        container_req_path = "/app/requirements.txt" # Path inside the container
        pip_requirements = get_pip_requirements(container_req_path)
        if pip_requirements is not None:
             logger.info(f"Read {len(pip_requirements)} requirements from {container_req_path} for model logging.")
        else:
             logger.warning("Could not read pip requirements. Model may be logged without specifying dependencies explicitly.")


        # Log the model artifact and register it
        try:
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=f"{model_name.lower()}_model",
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name, # This requires Model Registry
                pip_requirements=pip_requirements # Use requirements if successfully loaded
            )
            logger.info(f"Model {model_name} logged to MLflow run artifact '{model_info.artifact_path}' and registered as '{registered_model_name}'.")

            # Transition the model to staging using the registered model name
            transition_model_to_staging(registered_model_name)

        except Exception as e:
            logger.error(f"Failed to log model {model_name} to registry: {e}")
            logger.error(traceback.format_exc())
            # Re-raise if logging/registering the model is critical for your workflow
            raise


        # Save predictions and plots (if these functions are defined and desired)
        try:
             save_predictions(model, X_test, y_test, wine_feature_names, model_name, output_dir)
        except Exception as e:
             logger.error(f"Error during save_predictions for {model_name}: {e}")
             logger.error(traceback.format_exc())


        try:
             save_model_plots(model, model_name, wine_feature_names, quality_order, output_dir)
        except Exception as e:
             logger.error(f"Error during save_model_plots for {model_name}: {e}")
             logger.error(traceback.format_exc())


        logger.info(f"MLflow logging completed for {model_name}.")

    except Exception as e:
        # Catch any exception in the overall log_model_run process
        logger.error(f"Overall MLflow logging process failed for {model_name}: {e}")
        logger.error(traceback.format_exc())
        # Re-raise if logging is a critical part of the training run
        raise