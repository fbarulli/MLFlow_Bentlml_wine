# training.py
import logging
import traceback
import numpy as np
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)

def train_and_evaluate_model(model_config, X_train, X_test, y_train, y_test, quality_order):
    """
    Trains a model based on configuration and evaluates it.

    Args:
        model_config (dict): Dictionary containing 'name', 'class', and 'params'.
        X_train (pd.DataFrame or np.ndarray): Training features.
        X_test (pd.DataFrame or np.ndarray): Testing features.
        y_train (np.ndarray): Training labels (encoded integers).
        y_test (np.ndarray): Testing labels (encoded integers).
        quality_order (list): List of original string labels (e.g., ["low", "medium", "high"]).

    Returns:
        tuple: (trained_model, report_dict)
    """
    model_name = model_config["name"]
    model_class = model_config["class"]
    model_params = model_config["params"]

    logger.info(f"Instantiating and training {model_name}...")
    try:
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        logger.info(f"{model_name} training complete.")
    except Exception as e:
        logger.error(f"Failed to train model {model_name}: {e}")
        print(f"Failed to train model {model_name}: {e}")
        logger.error(traceback.format_exc())
        raise # Training failure is critical

    logger.info(f"Evaluating {model_name}...")
    report_dict = {} # Initialize report_dict
    try:
        y_pred = model.predict(X_test)

        # Ensure all unique labels from y_test are included in target_names mapping
        unique_labels_in_test = sorted(list(np.unique(y_test)))
        # Map numerical labels back to quality names using quality_order
        # Add a check in case encoded labels are outside the range of quality_order indices
        target_names_for_report = [quality_order[int(i)] if int(i) < len(quality_order) else f"Unknown_{int(i)}" for i in unique_labels_in_test]

        # Generate classification report
        report_dict = classification_report(
            y_test,
            y_pred,
            labels=unique_labels_in_test, # Explicitly pass labels found in y_test
            target_names=target_names_for_report,
            output_dict=True,
            zero_division=0 # Handle cases with no samples in a class gracefully
        )
        logger.info(f"{model_name} evaluation complete.")
    except Exception as e:
        logger.error(f"Failed to evaluate model {model_name}: {e}")
        logger.error(traceback.format_exc())
        # Evaluation failure is less critical than training, log warning and return empty report
        logger.warning("Evaluation failed, returning empty report dictionary.")
        report_dict = {} # Return empty dict if evaluation fails

    return model, report_dict