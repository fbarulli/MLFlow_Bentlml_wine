# data.py
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import logging
import traceback

logger = logging.getLogger(__name__)

def load_data():
    """Loads the wine quality dataset from KaggleHub."""
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
    """Encodes the target and splits the data into training and testing sets."""
    quality_order = ["low", "medium", "high"]
    if 'quality_label' not in df.columns:
        logger.error("Column 'quality_label' not found in the dataset.")
        raise ValueError("Missing target column 'quality_label'")

    # Ordinally encode the target variable
    encoder = OrdinalEncoder(
        categories=[quality_order],
        handle_unknown='use_encoded_value', # Map unknown values to -1 (or raise error if preferred)
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

    # Split data into training and testing sets with stratification
    try:
        # Check class counts to avoid issues with stratification on very small classes
        class_counts = np.bincount(y.astype(int))
        min_samples_split = min(class_counts) if class_counts.size > 0 else 0
        if min_samples_split > 0 and min_samples_split < 2:
             logger.warning(f"Minimum samples in a class for stratification is {min_samples_split}. Stratification may fail if test_size is too large.")
        elif class_counts.size > 0 and min_samples_split == 0:
             logger.error("One or more classes have zero samples. Stratification is not possible.")
             raise ValueError("Cannot stratify with zero samples in a class.")
        elif class_counts.size == 0:
             logger.warning("Target variable y is empty. Skipping stratification.")


        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            # Only stratify if there are classes to stratify by and samples exist
            stratify=y if class_counts.size > 0 and min_samples_split > 0 else None
        )
        logger.info(f"Data split successfully. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test, quality_order, X.columns.tolist()
    except Exception as e:
        logger.error(f"Failed to split data: {e}")
        logger.error(traceback.format_exc())
        raise