# mlflow_tracking/mlflow/wine_work.py
import kagglehub
from kagglehub import KaggleDatasetAdapter
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import logging
import os
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

        if model_name == "LogisticRegression":
            if hasattr(model, 'coef_') and model.coef_ is not None:
                coefs_per_class = model.coef_
                num_classes_in_model = coefs_per_class.shape[0]
                if num_classes_in_model != len(quality_order):
                    logger.warning(f"LogisticRegression coef_ shape {coefs_per_class.shape} doesn't match expected {len(quality_order)} quality order labels. Plotting based on available coefficients.")
                    class_labels = [quality_order[i] if i < len(quality_order) else f"Class {i}" for i in range(num_classes_in_model)]
                else:
                    class_labels = quality_order

                for i in range(num_classes_in_model):
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
        raise

def transition_model_to_staging(registered_model_name):
    try:
        client = MlflowClient()
        logger.info(f"Attempting to assign 'Staging' alias to latest version of '{registered_model_name}'")

        latest_versions = client.search_model_versions(f"name='{registered_model_name}'")

        if not latest_versions:
            logger.warning(f"No 'latest' version found for registered model '{registered_model_name}'. Cannot set alias.")
            return

        latest_version = latest_versions[0]
        latest_version_number = latest_version.version
        logger.info(f"Latest version found is {latest_version_number} (Run ID: {latest_version.run_id}).")

        try:
            versions_with_current_alias = client.search_model_versions(f"name='{registered_model_name}' and aliases='Staging'")
            for version_with_alias in versions_with_current_alias:
                if version_with_alias.version != latest_version_number:
                     logger.info(f"Removing 'Staging' alias from version {version_with_alias.version}")
                     client.set_registered_model_alias(
                         name=registered_model_name,
                         alias="Staging",
                         version="1"
                     )
        except Exception as remove_alias_err:
             logger.warning(f"Could not remove existing 'Staging' alias: {remove_alias_err}")

        client.set_registered_model_alias(
            name=registered_model_name,
            alias="Staging",
            version=latest_version_number
        )

        logger.info(f"Registered model '{registered_model_name}' version {latest_version_number} successfully aliased as 'Staging'.")

    except Exception as e:
        logger.error(f"Failed to set 'Staging' alias for model '{registered_model_name}': {e}")
        logger.error(traceback.format_exc())

def log_model_to_mlflow(model, model_name, params, report_dict, X_train, X_test, y_test, wine_feature_names, quality_order, output_dir):
    try:
        logger.info(f"Starting MLflow logging for {model_name}...")

        try:
            metrics = {}
            if 'accuracy' in report_dict: metrics["accuracy"] = report_dict["accuracy"]
            if 'weighted avg' in report_dict:
                metrics["weighted_avg_precision"] = report_dict["weighted avg"].get("precision")
                metrics["weighted_avg_recall"] = report_dict["weighted avg"].get("recall")
                metrics["weighted_avg_f1-score"] = report_dict["weighted avg"].get("f1-score")
            if 'macro avg' in report_dict:
                metrics["macro_avg_precision"] = report_dict["macro avg"].get("precision")
                metrics["macro_avg_recall"] = report_dict["macro avg"].get("recall")
                metrics["macro_avg_f1-score"] = report_dict["macro avg"].get("f1-score")

            for i, label_name in enumerate(quality_order):
                 label_key_float = str(float(i))
                 label_key_int = str(int(i))
                 if label_key_float in report_dict:
                      label_key = label_key_float
                 elif label_key_int in report_dict:
                      label_key = label_key_int
                 else:
                      continue

                 metrics[f"precision_{label_name}"] = report_dict[label_key].get("precision")
                 metrics[f"recall_{label_name}"] = report_dict[label_key].get("recall")
                 metrics[f"f1-score_{label_name}"] = report_dict[label_key].get("f1-score")
                 metrics[f"support_{label_name}"] = report_dict[label_key].get("support")

            metrics = {k: v for k, v in metrics.items() if v is not None}

            if metrics:
                mlflow.log_metrics(metrics)
                logger.info(f"Metrics logged for {model_name}: {metrics}")
            else:
                logger.warning(f"No metrics found in report_dict for {model_name} to log.")

        except Exception as metric_err:
            logger.error(f"Failed to log metrics for {model_name}: {metric_err}")
            logger.error(traceback.format_exc())

        mlflow.set_tag("Training Info", f"{model_name} model for Wine")
        logger.info(f"Tag 'Training Info' set for {model_name}.")

        signature = None
        input_example = None
        if hasattr(model, 'predict'):
            try:
                if isinstance(X_train, pd.DataFrame) and not X_train.empty:
                     input_example_data = X_train.sample(min(100, X_train.shape[0]), random_state=42) if X_train.shape[0] > 1 else X_train.head(1)
                     if not input_example_data.empty:
                        signature = infer_signature(input_example_data, model.predict(input_example_data))
                        input_example = input_example_data.head(5) if not input_example_data.empty else None
                        logger.info(f"Signature inferred successfully for {model_name}.")
                     else:
                         logger.warning(f"Sampled input data is empty for signature inference for {model_name}.")
                else:
                    logger.warning(f"X_train is not a DataFrame or is empty. Cannot infer signature for {model_name}.")

            except Exception as sig_err:
                logger.error(f"Failed to infer signature for {model_name}: {sig_err}")
                logger.error(traceback.format_exc())

        registered_model_name = f"tracking-wine-{model_name.lower()}"

        container_req_path = "/app/requirements.txt"
        try:
             pip_requirements = get_pip_requirements(container_req_path)
             logger.info(f"Read {len(pip_requirements)} requirements from {container_req_path}")
        except Exception as req_err:
             logger.error(f"Failed to read pip requirements: {req_err}")
             logger.error(traceback.format_exc())
             raise

        try:
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=f"{model_name.lower()}_model",
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name,
                pip_requirements=pip_requirements
            )
            logger.info(f"Model {model_name} logged to MLflow run artifact '{model_info.artifact_path}' and registered as '{registered_model_name}'.")

            transition_model_to_staging(registered_model_name)

        except Exception as e:
            logger.error(f"Failed to log model {model_name} to registry: {e}")
            logger.error(traceback.format_exc())
            raise

    
        save_predictions(model, X_test, y_test, wine_feature_names, model_name, output_dir)
    
        save_model_plots(model, model_name, wine_feature_names, quality_order, output_dir)

        logger.info(f"MLflow logging completed for {model_name}.")

    except Exception as e:
        logger.error(f"Overall MLflow logging process failed for {model_name}: {e}")
        logger.error(traceback.format_exc())
        raise

def main():
    logger.info("Starting ML pipeline...")
    try:
        output_dir = setup_environment()
        logger.info(f"Using output directory: {output_dir}")

        df = load_data()

        X_train, X_test, y_train, y_test, quality_order, wine_feature_names = preprocess_data(df)

        if len(wine_feature_names) != X_train.shape[1]:
             logger.error(f"Feature name length ({len(wine_feature_names)}) does not match X_train features ({X_train.shape[1]})")
             raise AssertionError("Feature name mismatch")
        logger.info(f"Feature names ({len(wine_feature_names)}) match X_train shape ({X_train.shape[1]}).")

        model_configs = get_model_configs()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Re-created output directory: {output_dir}")

        mlflow.set_experiment("Wine_Quality_Classification")

        with mlflow.start_run(run_name="Wine_Model_Training_Run"):
            logger.info(f"Started parent MLflow run: {mlflow.active_run().info.run_id}")

            mlflow.log_param("dataset", "kaggle_wine_quality_classification_downloaded")
            mlflow.log_param("train_test_split_ratio", 0.8)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("quality_order", quality_order)

            for config in model_configs:
                model_name = config["name"]
                model_class = config["class"]
                model_params = config["params"]

                logger.info(f"Starting nested MLflow run for {model_name}...")
                with mlflow.start_run(run_name=model_name, nested=True):
                    logger.info(f"Started nested MLflow run: {mlflow.active_run().info.run_id}")

                    mlflow.log_param("model_type", model_name)

                    logger.info(f"Instantiating and training {model_name}...")
                    model = model_class(**model_params)
                    model.fit(X_train, y_train)
                    logger.info(f"{model_name} training complete.")

                    logger.info(f"Evaluating {model_name}...")
                    y_pred = model.predict(X_test)
                    unique_labels = sorted(list(np.unique(y_test)))
                    target_names = [quality_order[int(i)] for i in unique_labels]
                    report_dict_model = classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names, output_dict=True)
                    logger.info(f"{model_name} evaluation complete.")

                    log_model_to_mlflow(
                        model=model,
                        model_name=model_name,
                        params=model_params,
                        report_dict=report_dict_model,
                        X_train=X_train,
                        X_test=X_test,
                        y_test=y_test,
                        wine_feature_names=wine_feature_names,
                        quality_order=quality_order,
                        output_dir=output_dir
                    )
                    logger.info(f"Completed nested MLflow run for {model_name}.")

            logger.info("Completed parent MLflow run.")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        logger.error(traceback.format_exc())
        if mlflow.active_run():
             mlflow.set_tag("status", "Failed")
             mlflow.log_param("error_message", str(e))
        raise

if __name__ == "__main__":
    main()