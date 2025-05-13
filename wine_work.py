# mlflow_tracking/mlflow/wine_work.py
import kagglehub
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import pandas as pd
import logging
import traceback
import os
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

import dagshub  # <--- ADD THIS IMPORT

# --- DagsHub MLflow Integration Initialization ---
# This initializes DagsHub's MLflow integration.
# It will automatically use MLFLOW_TRACKING_URI and MLFLOW_TRACKING_TOKEN
# from the environment if they are set.
try:
    # Get repo owner and name from the MLFLOW_TRACKING_URI environment variable
    dagshub_uri = os.getenv("MLFLOW_TRACKING_URI")
    repo_owner = None
    repo_name = None
    if dagshub_uri and "dagshub.com" in dagshub_uri:
         # Expected format: mlflow+https://dagshub.com/<owner>/<repo_name>.mlflow
         # Split by '/'
         parts = dagshub_uri.split('/')
         if len(parts) >= 5:
              repo_owner = parts[3]
              # The repo name part includes '.mlflow'. Need to remove it.
              repo_name_mlflow_suffix = parts[4]
              if repo_name_mlflow_suffix.endswith(".mlflow"):
                  repo_name = repo_name_mlflow_suffix[:-len(".mlflow")]
              else:
                  # Handle unexpected URI format if necessary, or just log a warning
                  logger.warning(f"MLFLOW_TRACKING_URI '{dagshub_uri}' has unexpected format (repo part missing .mlflow suffix).")
                  repo_owner = None # Invalidate if format is truly wrong

    if repo_owner and repo_name:
        logger.info(f"Initializing DagsHub for MLflow: repo_owner={repo_owner}, repo_name={repo_name}")
        # The mlflow=True flag tells dagshub.init to configure MLflow
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        logger.info("DagsHub MLflow integration initialized.")
    else:
        # Log a warning if the URI wasn't in the expected format to extract owner/repo
        logger.warning(f"Could not parse repo owner/name from MLFLOW_TRACKING_URI '{dagshub_uri}' for dagshub.init. Proceeding, but MLflow may not be configured correctly.")

except Exception as e:
    # Log any exceptions during dagshub.init but don't necessarily stop the script
    # immediately, as MLflow might still work with just environment variables.
    # However, based on previous errors, it seems dagshub.init is crucial.
    logger.error(f"Error initializing DagsHub MLflow integration: {str(e)}")
    logger.error(traceback.format_exc())
    # Decide whether to re-raise or just warn. Re-raising might be safer
    # if dagshub.init is truly required for model registry. Let's re-raise for now.
    raise # Re-raise the exception if dagshub.init fails


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add or ensure necessary imports are here (copied from your existing script)
# Assuming these were already present or you've added them based on previous errors/needs
# from sklearn.metrics import classification_report # Needed for the report function
# from pydantic import BaseModel, Field # Not needed in wine_work.py, remove if present


def get_pip_requirements(req_file_path):
    if not os.path.exists(req_file_path):
        logger.error(f"Requirements file not found at: {req_file_path}")
        raise FileNotFoundError(f"Requirements file not found at: {req_file_path}")
    with open(req_file_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    return requirements

def setup_environment():
    output_dir = os.getenv("CONTAINER_APP_OUTPUT_DIR")
    if not output_dir:
         logger.error("CONTAINER_APP_OUTPUT_DIR environment variable is not set.")
         raise ValueError("CONTAINER_APP_OUTPUT_DIR environment variable is not set.")

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Outputs folder created at {output_dir}")

    # This call to mlflow.set_tracking_uri is potentially redundant if dagshub.init(mlflow=True)
    # handles it, but keeping it ensures the URI is explicitly set from the environment.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
         logger.error("MLFLOW_TRACKING_URI environment variable is not set.")
         raise ValueError("MLFLOW_TRACKING_URI environment variable is not set.")
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

    mlflow.sklearn.autolog(log_models=False)
    logger.info("MLflow Scikit-learn autologging enabled (excluding model logging).")

    return output_dir

def load_data():
    file_path = "wine_quality_classification.csv"
    repo_name = "sahideseker/wine-quality-classification"
    try:
        logger.info(f"Attempting to download dataset '{file_path}' from Kaggle repo '{repo_name}' using kagglehub")
        # Use the new load_dataset method if kagglehub is recent enough, or old one if needed
        # Assuming the old one works based on previous logs, but noting the deprecation warning.
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            repo_name,
            file_path
        )
        logger.info("Dataset downloaded successfully using kagglehub.")
        logger.info(f"Dataset shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to download dataset using kagglehub: {e}")
        logger.error(traceback.format_exc())
        raise

def preprocess_data(df):
    quality_order = ["low", "medium", "high"]
    if 'quality_label' not in df.columns:
        logger.error("Column 'quality_label' not found in the dataset.")
        raise ValueError("Missing target column 'quality_label'")

    encoder = OrdinalEncoder(
        categories=[quality_order],
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )
    try:
        y_encoded = encoder.fit_transform(df[['quality_label']]).ravel()
        logger.info("Target encoded successfully.")
    except Exception as e:
        logger.error(f"Failed to encode target: {e}")
        logger.error(traceback.format_exc())
        raise

    X = df.drop(columns=["quality_label"])
    y = y_encoded
    try:
        class_counts = np.bincount(y.astype(int))
        min_samples_split = min(class_counts)
        if min_samples_split < 2:
             logger.warning(f"Minimum samples in a class for stratification is {min_samples_split}. Stratification may fail if test_size is too large.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        logger.info(f"Data split successfully. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test, quality_order, X.columns.tolist()
    except Exception as e:
        logger.error(f"Failed to split data: {e}")
        logger.error(traceback.format_exc())
        raise

def get_model_configs():
    model_configs = []

    model_configs.append({
        "name": "LogisticRegression",
        "class": LogisticRegression,
        "params": {
            "solver": "lbfgs",
            "max_iter": 10000,
            "random_state": 8888,
            "class_weight": "balanced",
            "penalty": "l2",
            "C": 0.1
        }
    })

    model_configs.append({
        "name": "RandomForest",
        "class": RandomForestClassifier,
        "params": {
            "n_estimators": 30,
            "max_depth": 3
        }
    })

    return model_configs

def save_predictions(model, X_test, y_test, wine_feature_names, model_name, output_dir):
    try:
        predictions = model.predict(X_test)
        if isinstance(X_test, pd.DataFrame):
            result_df = X_test.copy()
        else:
             result_df = pd.DataFrame(X_test, columns=wine_feature_names)

        result_df["actual_class"] = y_test
        result_df["predicted_class"] = predictions

        csv_filename = f"{model_name.lower()}_predictions.csv"
        csv_path = os.path.join(output_dir, "predictions", csv_filename)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        result_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path="predictions")
        logger.info(f"Predictions saved to '{csv_path}' and logged as artifact for {model_name}")
        return csv_path
    except Exception as e:
        logger.error(f"Failed to save predictions for {model_name}: {e}")
        logger.error(traceback.format_exc())
        raise

def save_model_plots(model, model_name, wine_feature_names, quality_order, output_dir):
    try:
        plot_output_dir = os.path.join(output_dir, "plots")
        os.makedirs(plot_output_dir, exist_ok=True)

        # Import classification_report here or at the top if needed elsewhere
        # from sklearn.metrics import classification_report
        # Assuming it's not strictly needed for plotting itself based on the function content

        if model_name == "LogisticRegression":
            if hasattr(model, 'coef_') and model.coef_ is not None:
                coefs_per_class = model.coef_
                num_classes_in_model = coefs_per_class.shape[0]
                # Handle potential mismatch between model classes and quality_order length
                if num_classes_in_model != len(quality_order):
                    logger.warning(f"LogisticRegression coef_ shape {coefs_per_class.shape} doesn't match expected {len(quality_order)} quality order labels. Plotting based on available coefficients.")
                    class_labels = [quality_order[i] if i < len(quality_order) else f"Class {i}" for i in range(num_classes_in_model)]
                else:
                    class_labels = quality_order


                for i in range(num_classes_in_model):
                    # Ensure feature names match coefficient count
                    if len(wine_feature_names) != coefs_per_class.shape[1]:
                         logger.error(f"Feature name count ({len(wine_feature_names)}) does not match coefficients count ({coefs_per_class.shape[1]}) for {model_name}, class {i}. Cannot plot.")
                         continue # Skip plotting for this class if mismatch

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
                    plt.close()
                    logger.info(f"Plot saved to '{plot_path}' and logged for {model_name}")
            else:
                 logger.warning(f"LogisticRegression model does not have coef_ attribute or it is None. Cannot plot coefficients.")


        elif model_name == "RandomForest":
            if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
                # Ensure feature names match importance count
                if len(wine_feature_names) != len(model.feature_importances_):
                     logger.error(f"Feature name count ({len(wine_feature_names)}) does not match importance count ({len(model.feature_importances_)}) for {model_name}. Cannot plot.")
                else:
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
                    plt.close()
                    logger.info(f"Plot saved to '{plot_path}' and logged for {model_name}")
            else:
                logger.warning(f"{model_name} does not have feature_importances_ attribute or it is None.")

        else:
            logger.warning(f"Plotting logic not implemented for model type: {model_name}")

    except Exception as e:
        logger.error(f"Failed to log plots for {model_name}: {e}")
        logger.error(traceback.format_exc())
        # Do not re-raise, plotting failure shouldn't necessarily stop the run


def transition_model_to_staging(registered_model_name):
    try:
        client = MlflowClient()
        logger.info(f"Attempting to assign 'Staging' alias to latest version of '{registered_model_name}'")

        # Use search_model_versions ordered by version descending to get latest
        latest_versions = client.search_model_versions(f"name='{registered_model_name}'", order_by=["version DESC"], max_results=1)

        if not latest_versions:
            logger.warning(f"No version found for registered model '{registered_model_name}'. Cannot set alias.")
            return

        latest_version = latest_versions[0]
        latest_version_number = latest_version.version
        logger.info(f"Latest version found is {latest_version_number} (Run ID: {latest_version.run_id}).")

        # Optional: Remove alias from previous versions if desired.
        # Note: DagsHub/MLflow allows multiple versions to have the same alias.
        # If you want *only* the latest to have 'Staging', you would first
        # remove the alias from any version that currently has it.
        # This part is commented out as managing aliases can vary based on workflow.
        # try:
        #     versions_with_current_alias = client.search_model_versions(f"name='{registered_model_name}' and aliases='Staging'")
        #     for version_with_alias in versions_with_current_alias:
        #         if version_with_alias.version != latest_version_number:
        #              logger.info(f"Removing 'Staging' alias from version {version_with_alias.version}")
        #              # Removing an alias requires specifying the version to remove from
        #              client.set_registered_model_alias(name=registered_model_name, alias="Staging", version=version_with_alias.version, is_dangling=True) # is_dangling=True effectively removes
        # except Exception as remove_alias_err:
        #      logger.warning(f"Could not remove existing 'Staging' alias: {remove_alias_err}")


        # Set the alias on the latest version
        client.set_registered_model_alias(
            name=registered_model_name,
            alias="Staging",
            version=latest_version_number
        )

        logger.info(f"Registered model '{registered_model_name}' version {latest_version_number} successfully aliased as 'Staging'.")

    except Exception as e:
        logger.error(f"Failed to set 'Staging' alias for model '{registered_model_name}': {e}")
        logger.error(traceback.format_exc())
        # Do not re-raise, alias transition failure shouldn't stop the entire run


def log_model_to_mlflow(model, model_name, params, report_dict, X_train, X_test, y_test, wine_feature_names, quality_order, output_dir):
    try:
        logger.info(f"Starting MLflow logging for {model_name}...")

        try:
            # Import classification_report here if used only in this block
            from sklearn.metrics import classification_report
            # Re-calculate report if report_dict wasn't computed outside
            # Assuming report_dict is already computed and passed in correctly
            # For safety, let's ensure report_dict is usable
            if not isinstance(report_dict, dict):
                 logger.warning("report_dict is not a dictionary. Skipping metric logging.")
                 metrics = {} # Empty metrics if input is bad
            else:
                 metrics = {}
                 # Populate metrics dictionary similar to your original code
                 if 'accuracy' in report_dict: metrics["accuracy"] = report_dict["accuracy"]
                 if 'weighted avg' in report_dict and isinstance(report_dict['weighted avg'], dict):
                     metrics["weighted_avg_precision"] = report_dict["weighted avg"].get("precision")
                     metrics["weighted_avg_recall"] = report_dict["weighted avg"].get("recall")
                     metrics["weighted_avg_f1-score"] = report_dict["weighted avg"].get("f1-score")
                 if 'macro avg' in report_dict and isinstance(report_dict['macro avg'], dict):
                     metrics["macro_avg_precision"] = report_dict["macro avg"].get("precision")
                     metrics["macro_avg_recall"] = report_dict["macro avg"].get("recall")
                     metrics["macro_avg_f1-score"] = report_dict["macro avg"].get("f1-score")

                 for i, label_name in enumerate(quality_order):
                     label_key_float = str(float(i))
                     label_key_int = str(int(i))
                     label_key = None # Find the correct key (float or int string)
                     if label_key_float in report_dict and isinstance(report_dict[label_key_float], dict):
                          label_key = label_key_float
                     elif label_key_int in report_dict and isinstance(report_dict[label_key_int], dict):
                          label_key = label_key_int

                     if label_key:
                         metrics[f"precision_{label_name}"] = report_dict[label_key].get("precision")
                         metrics[f"recall_{label_name}"] = report_dict[label_key].get("recall")
                         metrics[f"f1-score_{label_name}"] = report_dict[label_key].get("f1-score")
                         metrics[f"support_{label_name}"] = report_dict[label_key].get("support")
                     # else: logger.warning(f"Class {i} ({label_name}) not found in classification report dictionary.")


            metrics = {k: v for k, v in metrics.items() if v is not None}

            if metrics:
                mlflow.log_metrics(metrics)
                logger.info(f"Metrics logged for {model_name}: {metrics}")
            else:
                logger.warning(f"No valid metrics found in report_dict for {model_name} to log.")

        except Exception as metric_err:
            logger.error(f"Failed to log metrics for {model_name}: {metric_err}")
            logger.error(traceback.format_exc())
            # Do not re-raise, metric logging failure shouldn't stop the run

        mlflow.set_tag("Training Info", f"{model_name} model for Wine")
        logger.info(f"Tag 'Training Info' set for {model_name}.")

        signature = None
        input_example = None
        if hasattr(model, 'predict'):
:start_line:406
-------
            try:
                # Ensure X_train is suitable for sampling and prediction
                if isinstance(X_train, pd.DataFrame) and not X_train.empty:
                    # Limit sample size to prevent issues with very large dataframes
                    sample_size = min(100, X_train.shape[0]) if X_train.shape[0] > 1 else X_train.shape[0]
                    if sample_size > 0:
                        input_example_data = X_train.sample(sample_size, random_state=42)
                        if not input_example_data.empty:
                            # Predict on a copy to avoid potential issues with inplace operations by the model
                            signature = infer_signature(input_example_data, model.predict(input_example_data.copy()))
                            # Log only a few examples, not the whole sample
                            input_example = input_example_data.head(min(5, sample_size))
                            logger.info(f"Signature inferred successfully for {model_name}.")
                        else:
                            logger.warning(f"Sampled input data is empty for signature inference for {model_name}.")
                    elif X_train.shape[0] == 0:
                        logger.warning(f"X_train is empty. Cannot infer signature or create input example for {model_name}.")
                    else:  # X_train.shape[0] == 1, sample_size is 1
                        input_example_data = X_train.head(1)
                        signature = infer_signature(input_example_data, model.predict(input_example_data.copy()))
                        input_example = input_example_data.head(1)
                        logger.info(f"Signature inferred successfully for {model_name} (single example).")
                else:
                    logger.warning(f"X_train is not a DataFrame or is empty. Cannot infer signature for {model_name}.")
            except Exception as sig_err:
                logger.error(f"Failed to infer signature for {model_name}: {sig_err}")
                logger.error(traceback.format_exc())
                # Do not re-raise, signature failure shouldn't stop the run


        registered_model_name = f"tracking-wine-{model_name.lower()}"

        # Get pip requirements for logging with the model
        container_req_path = "/app/requirements.txt" # Path inside the container
        pip_requirements = None # Initialize to None
        try:
             pip_requirements = get_pip_requirements(container_req_path)
             logger.info(f"Read {len(pip_requirements)} requirements from {container_req_path}")
        except Exception as req_err:
             logger.error(f"Failed to read pip requirements from {container_req_path}: {req_err}")
             logger.error(traceback.format_exc())
             # Do not re-raise, model logging can often proceed without explicit pip_requirements


        try:
            # Log the model artifact and register it
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
            raise # Re-raise this error


        # Save predictions and plots (if these functions are defined and desired)
        try:
             save_predictions(model, X_test, y_test, wine_feature_names, model_name, output_dir)
        except Exception as e:
             logger.error(f"Error during save_predictions for {model_name}: {e}")
             logger.error(traceback.format_exc()) # Log but continue


        try:
             save_model_plots(model, model_name, wine_feature_names, quality_order, output_dir)
        except Exception as e:
             logger.error(f"Error during save_model_plots for {model_name}: {e}")
             logger.error(traceback.format_exc()) # Log but continue


        logger.info(f"MLflow logging completed for {model_name}.")

    except Exception as e:
        # Catch any exception in the overall log_model_to_mlflow process
        logger.error(f"Overall MLflow logging process failed for {model_name}: {e}")
        logger.error(traceback.format_exc())
        # Re-raise if logging is a critical part of the training run
        raise # Re-raise


def main():
    logger.info("Starting ML pipeline...")
    try:
        output_dir = setup_environment()

        with mlflow.start_run(run_name="Wine_Model_Training_Run"):
            output_dir = setup_environment()
            X_train, X_test, y_train, y_test, quality_order, wine_feature_names = load_data_and_preprocess()
            model_configs = get_model_configs()

            for config in model_configs:
                model_name = config["name"]
                model_class = config["class"]
                params = config["params"]

                logger.info(f"Training {model_name} model...")
                model = model_class(**params)
                model.fit(X_train, y_train)
                logger.info(f"{model_name} model trained.")

                try:
                    # Import classification_report here if used only in this block
                    from sklearn.metrics import classification_report
                    y_pred = model.predict(X_test)
                    report = classification_report(y_test, y_pred, target_names=quality_order, output_dict=True, zero_division=0)
                    logger.info(f"{model_name} Classification Report:\n{classification_report(y_test, y_pred, target_names=quality_order, zero_division=0)}")
                except Exception as report_err:
                    logger.error(f"Failed to generate classification report for {model_name}: {report_err}")
                    logger.error(traceback.format_exc())
                    report = {} # Empty report if generation fails

                with mlflow.start_run(run_name=model_name, nested=True):
                    model_name = config["name"]
                    log_model_to_mlflow(model, model_name, params, report, X_train, X_test, y_test, wine_feature_names, quality_order, output_dir)

    except Exception as e:
        logger.error(f"ML pipeline failed: {e}")
        logger.error(traceback.format_exc())
        try:
            logger.info(f"Logging failure status to MLflow run {mlflow.active_run().info.run_id}")
            mlflow.log_param("status", "failed")
            mlflow.log_metric("status", 0)
        except:
            pass # best effort logging
    finally:
        mlflow.end_run()
        logger.info("ML pipeline completed.")

if __name__ == "__main__":
    main()