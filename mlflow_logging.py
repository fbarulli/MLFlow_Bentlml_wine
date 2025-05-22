# mlflow_logging.py
import logging
import mlflow
import mlflow.sklearn
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Import necessary modules
import mlflow.tracking.client
from mlflow.entities.model_registry import ModelVersion
import traceback # <--- ADDED IMPORT







print(mlflow.__version__)

logger = logging.getLogger(__name__)


def _log_parameters(model_params, model=None):
    """Logs model parameters to the current active MLflow run."""
    try:
        logger.info("Logging parameters...")
        mlflow.log_params(model_params)
        if model is not None:
            mlflow.log_param("model_class", type(model).__name__)
        logger.info("Parameters logged.")
    except Exception as e:
        # Log exception type and message for debugging
        logger.warning(f"Failed to log parameters: {type(e).__name__} - {e}")
        # Do not re-raise


def _log_metrics(report_dict, quality_order):
    """Logs evaluation metrics from a classification report dictionary."""
    try:
        logger.info("Logging metrics...")
        if 'accuracy' in report_dict:
             mlflow.log_metric("overall_accuracy", report_dict.get('accuracy'))
             logger.info(f"  Logged overall_accuracy: {report_dict.get('accuracy'):.4f}")

        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in report_dict:
                 for metric, value in report_dict[avg_type].items():
                     if metric != 'support': # Avoid logging 'support' from averages as metrics
                         mlflow.log_metric(f"{avg_type.replace(' ', '_')}_{metric}", value)
                         logger.info(f"  Logged {avg_type.replace(' ', '_')}_{metric}: {value:.4f}")

        # Optional: Log per-class metrics
        # for class_label in quality_order:
        #     class_str = str(class_label) # Ensure key matches report_dict
        #     if class_str in report_dict:
        #          for metric in ['precision', 'recall', 'f1-score']:
        #              if metric in report_dict[class_str]:
        #                  mlflow.log_metric(f"class_{class_str}_{metric}", report_dict[class_str][metric])

        logger.info("Metrics logged.")

    except Exception as e:
        # Log exception type and message
        logger.warning(f"Failed to log metrics: {type(e).__name__} - {e}")
        # Do not re-raise


def _log_and_register_model(model, model_name, X_train=None):
    """
    Logs the model artifact and registers it in the MLflow Model Registry.
    Includes model signature and input example when available.
    """
    registered_model_name = f"tracking-wine-{model_name.lower().replace(' ', '-')}"
    artifact_path = model_name.lower().replace(' ', '_')
    logger.info(f"Logging model artifact to '{artifact_path}' and attempting registration as '{registered_model_name}'...")
    model_info = None
    
    try:
        # Create model signature and input example if X_train is provided
        signature = None
        input_example = None
        
        if X_train is not None:
            try:
                from mlflow.models.signature import infer_signature
                
                # Get a single row as example (avoid large input examples)
                example_X = X_train.iloc[:1] if hasattr(X_train, 'iloc') else X_train[:1]
                
                # Try to infer the model signature from the model and input data
                # For the output schema, we'll predict on the example
                example_y_pred = model.predict(example_X)
                
                # Infer the signature from input and output
                signature = infer_signature(example_X, example_y_pred)
                
                # Use the same example as input_example
                input_example = example_X
                
                logger.info("Successfully created model signature and input example")
            except Exception as sig_err:
                logger.warning(f"Failed to create model signature: {type(sig_err).__name__} - {sig_err}")
                logger.warning(traceback.format_exc())
                # Continue without signature if it fails
        
        # Log the model with signature and example if available
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example
        )
        
        if hasattr(model_info, 'model_uri'):
            logger.info(f"Model '{model_name}' logged successfully to {model_info.model_uri}")
            
        if isinstance(model_info, ModelVersion) and hasattr(model_info, 'version') and hasattr(model_info, 'name'):
            logger.info(f"Model '{model_name}' registered as '{model_info.name}' version {model_info.version}.")
        elif model_info is not None:
            # For newer MLflow versions that return ModelInfo instead of ModelVersion
            logger.info(f"Model '{model_name}' logged successfully. Check MLflow UI for registration details.")
            
            # Try to extract registration info from ModelInfo if possible
            if hasattr(model_info, 'run_id') and hasattr(model_info, 'model_uri'):
                logger.info(f"Run ID: {model_info.run_id}, Model URI: {model_info.model_uri}")
                
                # Try to extract the version from the model_uri if possible
                # Format is usually runs:/run_id/artifacts/model_path
                try:
                    client = mlflow.tracking.client.MlflowClient()
                    versions = client.search_model_versions(f"name='{registered_model_name}'")
                    # Sort by creation timestamp to get the latest
                    versions.sort(key=lambda x: x.creation_timestamp, reverse=True)
                    if versions:
                        latest_version = versions[0].version
                        logger.info(f"Latest model version for '{registered_model_name}': {latest_version}")
                        # Create a minimal ModelVersion-like object with name and version attributes
                        class MinimalModelVersion:
                            def __init__(self, name, version):
                                self.name = name
                                self.version = version
                        
                        # Return this object so _set_staging_alias can use it
                        return MinimalModelVersion(registered_model_name, latest_version)
                except Exception as e:
                    logger.warning(f"Failed to extract version details: {type(e).__name__} - {e}")
        else:
            logger.warning("mlflow.sklearn.log_model returned None. Registration likely failed.")
            
        return model_info
    except Exception as e:
        logger.error(f"Failed to log or register model '{model_name}' as '{registered_model_name}': {type(e).__name__} - {e}")
        logger.error(traceback.format_exc())
        return None


def _set_staging_alias(registered_model_name, model_version_info=None):
    """
    Attempts to assign the 'Staging' alias to a specific or the latest version of a registered model.
    Prefers using the version from model_version_info if provided, falls back to searching for the numerically latest.
    """
    alias_name = "Staging"
    logger.info(f"Attempting to set staging alias for model {registered_model_name}")

    if not registered_model_name:
        logger.warning(f"No registered model name provided for alias assignment. Skipping alias assignment.")
        return

    latest_version_number_str = None # Use string as version numbers are often strings
    if model_version_info is not None and hasattr(model_version_info, 'version'):
         # Use the version from the object returned by log_model if available
         latest_version_number_str = str(model_version_info.version)
         logger.info(f"Using version {latest_version_number_str} obtained from model_version_info for alias assignment.")
    else:
        # Fallback: Search for the numerically latest version
        logger.warning(f"Model version info not available directly for alias assignment. Searching for numerically latest version for '{registered_model_name}'.")
        try:
            client = mlflow.tracking.client.MlflowClient()
            all_versions = client.search_model_versions(f"name='{registered_model_name}'", order_by=["version_number DESC"], max_results=1)
            if all_versions:
                # Get the numerically highest version number (latest)
                latest_version_number_str = all_versions[0].version
                logger.info(f"Found numerically latest version {latest_version_number_str} for '{registered_model_name}'.")
            else:
                 logger.warning(f"No versions found for registered model '{registered_model_name}'. Cannot assign alias.")
        except Exception as search_err:
             logger.warning(f"Failed to search for model versions for '{registered_model_name}': {type(search_err).__name__} - {search_err}")
             logger.warning(traceback.format_exc()) # Log traceback for search failure


    if latest_version_number_str is not None:
        try:
            logger.info(f"Attempting to assign '{alias_name}' alias to version {latest_version_number_str} of '{registered_model_name}'...")
            client = mlflow.tracking.client.MlflowClient() # Ensure client is instantiated if needed

            # This is the CORRECT method to move/set an alias in MLflow 2.x+
            client.set_registered_model_alias(registered_model_name, alias_name, latest_version_number_str)
            logger.info(f"Successfully assigned '{alias_name}' alias to version {latest_version_number_str} for model '{registered_model_name}'.")

            # Optional: Log which version has the alias to the current run
            mlflow.log_param(f"{alias_name}_model_{registered_model_name.replace('tracking-wine-', '')}_version", latest_version_number_str)

        except Exception as e:
            # Log exception type and message
            logger.error(f"Failed to set '{alias_name}' alias for model '{registered_model_name}': {type(e).__name__} - {e}")
            logger.error(traceback.format_exc()) # Log traceback for alias setting failure
            # Do not re-raise, alias assignment failure shouldn't stop logging
        else:
            logger.info(f"Successfully set alias '{alias_name}' for model '{registered_model_name}' version {latest_version_number_str}")


    else:
         logger.warning(f"No version number determined for model '{registered_model_name}'. Skipping alias assignment.")


def _log_predictions_artifact(model, X_test, y_test, model_name, output_dir):
    """Logs test set predictions as a CSV artifact."""
    try:
        logger.info(f"Logging predictions artifact for {model_name}...")
        # Predict on the test set to get predictions
        y_pred = model.predict(X_test)
        predictions_filename = f"{model_name.lower().replace(' ', '_')}_predictions.csv"
        predictions_path = os.path.join(output_dir, predictions_filename)

        # Create a DataFrame to save predictions with actual values for comparison
        predictions_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred
        })
        predictions_df.to_csv(predictions_path, index=False)

        # Log artifact to the current active run
        mlflow.log_artifact(predictions_path, artifact_path="predictions") # Organize artifacts
        logger.info(f"Predictions saved to '{predictions_path}' and logged as artifact: predictions/{predictions_filename}")

    except Exception as e:
         logger.warning(f"Failed to log predictions artifact for {model_name}: {type(e).__name__} - {e}")
         # Do not re-raise


def _log_coefficient_plot(model, wine_feature_names, quality_order, model_name, output_dir):
    """Logs coefficient plot artifact for models with a 'coef_' attribute."""
    if not hasattr(model, 'coef_') or not wine_feature_names:
        # logger.debug(f"Model {model_name} does not have 'coef_' attribute or feature names missing. Skipping coefficient plot.")
        return # Skip if not applicable

    logger.info(f"Attempting to log coefficients plot for {model_name}...")
    try:
        coefficients = model.coef_ # Shape (n_classes, n_features) for multi-class or (1, n_features) for binary/linear
        # Ensure coefficients is a list of lists/arrays even for binary case
        # Note: .coef_ is shape (n_classes, n_features) for multiclass, (n_features,) for binary/linear
        if len(coefficients.shape) == 1:
             coefficients = [coefficients]
             # Adjust quality_names for binary if needed - assuming multi-class mapping based on previous logs
             # If you have a binary model and need a plot, you might need separate logic or adapt
             # This plot is primarily designed for multi-class logistic regression as seen previously
             logger.warning(f"Coefficient plot logic is primarily for multi-class. Skipping default plot for {model_name} with coef_ shape {model.coef_.shape}.")
             return # Skip the loop if binary/linear

        # Map quality integers back to names for plot labels
        quality_names = [str(q) for q in quality_order] # e.g., ['3', '4', '5', '6', '7', '8']

        for i, class_coefs in enumerate(coefficients):
             # Skip plotting if the class index 'i' is outside the quality_names range
             if i >= len(quality_names):
                 logger.warning(f"Class index {i} is out of bounds for quality names ({len(quality_names)}). Skipping plot for this class.")
                 continue

             plt.figure(figsize=(10, 6))
             coef_df = pd.DataFrame({'Feature': wine_feature_names, 'Coefficient': class_coefs})
             coef_df = coef_df.sort_values(by='Coefficient', ascending=False) # Sort by importance
             
             # Updated seaborn barplot call to avoid deprecation warning
             sns.barplot(
                 x='Coefficient', 
                 y='Feature', 
                 data=coef_df, 
                 hue='Feature',    # Use Feature as the hue variable  
                 palette='viridis', # Keep the color palette
                 legend=False      # Don't show the legend since it would be redundant
             )
             
             plt.title(f'Feature Coefficients for Quality {quality_names[i]} ({model_name})')
             plt.xlabel('Coefficient Value')
             plt.ylabel('Feature')
             plt.tight_layout() # Adjust layout

             # Determine quality group name for filename
             quality_group_name = "quality_" + quality_names[i]

             plot_filename = f"{model_name.lower().replace(' ', '_')}_{quality_group_name}_coefficients.png"
             plot_path = os.path.join(output_dir, plot_filename)

             plt.savefig(plot_path)
             mlflow.log_artifact(plot_path, artifact_path="plots") # Organize plots
             logger.info(f"Logged coefficients plot artifact: plots/{plot_filename}")
             plt.close() # Close figure to free memory

    except Exception as e:
         # Log exception type and message
         logger.warning(f"Failed to log coefficients plot for {model_name}: {type(e).__name__} - {e}")
         logger.warning(traceback.format_exc()) # Log traceback for plotting failure


def _log_feature_importance_plot(model, wine_feature_names, model_name, output_dir):
    """Logs feature importance plot artifact for models with a 'feature_importances_' attribute."""
    if not hasattr(model, 'feature_importances_') or not wine_feature_names:
        # logger.debug(f"Model {model_name} does not have 'feature_importances_' attribute or feature names missing. Skipping feature importance plot.")
        return # Skip if not applicable

    logger.info(f"Attempting to log feature importances plot for {model_name}...")
    try:
        importances = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': wine_feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', hue='Feature', legend=False)
        plt.title(f'Feature Importances for {model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()

        plot_filename = f"{model_name.lower().replace(' ', '_')}_feature_importances.png"
        plot_path = os.path.join(output_dir, plot_filename)

        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path, artifact_path="plots") # Organize plots
        logger.info(f"Logged feature importances plot artifact: plots/{plot_filename}")
        plt.close() # Close figure

    except Exception as e:
         # Log exception type and message
         logger.warning(f"Failed to log feature importances plot for {model_name}: {type(e).__name__} - {e}")
         logger.warning(traceback.format_exc()) # Log traceback for plotting failure
         # Do not re-raise


def _log_confusion_matrix_plot(model, X_test, y_test, quality_order, model_name, output_dir):
    """Logs confusion matrix plot artifact."""
    logger.info(f"Attempting to log confusion matrix plot for {model_name}...")
    try:
        y_pred = model.predict(X_test)
        
        # Check what values actually exist in y_test and y_pred
        actual_classes = sorted(set(y_test))
        logger.info(f"Actual unique classes in y_test: {actual_classes}")
        
        # Use the actual values from the dataset instead of relying on quality_order
        cm = confusion_matrix(y_test, y_pred)
        
        # Create a DataFrame with the actual class labels
        cm_df = pd.DataFrame(cm, index=actual_classes, columns=actual_classes)

        plt.figure(figsize=(len(actual_classes)*1.2, len(actual_classes)*1.2))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {model_name}')
        plt.xlabel('Predicted Quality')
        plt.ylabel('Actual Quality')
        plt.tight_layout() # Adjust layout

        cm_plot_filename = f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        cm_plot_path = os.path.join(output_dir, cm_plot_filename)

        plt.savefig(cm_plot_path)
        mlflow.log_artifact(cm_plot_path, artifact_path="plots") # Organize plots
        logger.info(f"Logged confusion matrix plot artifact: plots/{cm_plot_filename}")
        plt.close() # Close the plot figure to free memory
    except Exception as e:
        logger.warning(f"Could not generate/log confusion matrix plot for {model_name}: {type(e).__name__} - {e}")
        logger.warning(traceback.format_exc())


def log_model_run(model, model_name, model_params, report_dict, X_train, X_test, y_test, wine_feature_names, quality_order, output_dir):
    """Orchestrates logging model parameters, metrics, and artifacts to the current active MLflow run."""
    if not mlflow.active_run():
        logger.error(f"log_model_run called outside of an active MLflow run for model '{model_name}'. Skipping logging.")
        return
    current_run_id = mlflow.active_run().info.run_id
    logger.info(f"Starting MLflow logging orchestration for model '{model_name}' in run {current_run_id}...")
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Ensured temporary artifact directory exists: {output_dir}")
        _log_parameters(model_params, model)
        _log_metrics(report_dict, quality_order)
        
        # Pass X_train to _log_and_register_model
        model_info = _log_and_register_model(model, model_name, X_train)
        
        # Updated logic for determining the registered model name
        registered_model_name = None
        if isinstance(model_info, ModelVersion) or (hasattr(model_info, 'name') and hasattr(model_info, 'version')):
            registered_model_name = model_info.name
        else:
            # Fallback for when model_info doesn't have the expected attributes
            registered_model_name = f"tracking-wine-{model_name.lower().replace(' ', '-')}"
            logger.info(f"Using fallback registered model name: {registered_model_name}")
        
        if registered_model_name:
            _set_staging_alias(registered_model_name, model_info)
        else:
            logger.warning(f"Model registration failed or skipped for {model_name}. Skipping alias assignment.")
        
        _log_predictions_artifact(model, X_test, y_test, model_name, output_dir)
        _log_confusion_matrix_plot(model, X_test, y_test, quality_order, model_name, output_dir)
        _log_coefficient_plot(model, wine_feature_names, quality_order, model_name, output_dir)
        _log_feature_importance_plot(model, wine_feature_names, model_name, output_dir)
        logger.info(f"MLflow logging orchestration complete for model '{model_name}'.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during MLflow logging orchestration for '{model_name}': {type(e).__name__} - {e}")
        logger.error(traceback.format_exc())
        raise

