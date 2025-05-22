# Wine Quality Classification Project

This project trains and evaluates machine learning models to classify the quality of wine based on various features. It uses MLflow for experiment tracking and DagsHub for collaboration and version control.

## File Breakdown


*   **`.env`**: Stores environment variables, such as MLflow tracking URI and credentials. **Input:** Credentials and URI.
*   **`build-train.sh`**: A shell script that builds the training Docker image.
*   **`config.py`**: Initializes DagsHub and sets up the environment, including setting the MLflow tracking URI and creating credential files. **Input:** Environment variables. **Output:** Configured MLflow and DagsHub integrations.
*   **`data.py`**: Loads the wine quality dataset from KaggleHub, preprocesses it by encoding the target variable, and splits the data into training and testing sets. **Input:** None (downloads data from KaggleHub). **Output:** Training and testing sets, quality order, and feature names.
*   **`debug_logistic_regression.py`**: A script for debugging the Logistic Regression model.
*   **`Dockerfile.bentoml`**: Dockerfile for building a BentoML service.
*   **`Dockerfile.train`**: Dockerfile for building the training environment.
*   **`mlflow_logging.py`**: Contains functions for logging model parameters, metrics, and artifacts to MLflow. **Input:** Trained model, model name, model parameters, classification report, and data. **Output:** MLflow logs.
*   **`models.py`**: Defines and returns a list of model configurations to be trained, including Logistic Regression and Random Forest Classifier. **Output:** A list of model configurations.
*   **`new_test.sh`**: A shell script for running tests.
*   **`output.log`**: Log file for the pipeline output.
*   **`pipeline.py`**: The main script that orchestrates the entire machine learning pipeline. It loads data, preprocesses it, gets model configurations, trains and evaluates models, and logs the results to MLflow. **Input:** None (relies on other modules and configurations). **Output:** Trained models and MLflow logs.
*   **`pipetest.sh`**: A shell script for testing the pipeline.
*   **`README.md`**: This file, providing an overview of the project.
*   **`requirements.txt`**: Lists the Python dependencies required for the project.
*   **`run-train.sh`**: A shell script that runs the training pipeline.
*   **`service.py`**: Defines the BentoML service for serving the trained model.
*   **`temp.sh`**: A temporary shell script.
*   **`training.py`**: Trains and evaluates a model based on a given configuration. **Input:** Model configuration, training data, and testing data. **Output:** Trained model and classification report.
*   **`wine_work.py`**: A script for working with the wine quality data.