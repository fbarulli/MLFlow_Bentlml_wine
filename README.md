# Wine Quality Classification Project

This project trains and evaluates ML models to classify wine quality using MLflow for experiment tracking and DagsHub for collaboration.

## File Breakdown

*   **.env**:
    *   Stores MLflow credentials and URI for accessing the MLflow server and DagsHub.
    *   **Input**: MLflow credentials and URI.
*   **bentofile.yaml**:
    *   BentoML configuration: service name, labels, files, Python dependencies, service class, and requirements file.
    *   **Input**: Service name, labels, include patterns, Python requirements.
*   **build-train.sh**:
    *   Builds the training Docker image using `Dockerfile.train`.
    *   **Input**: Env variables from `.env`.
*   **build.sh**:
    *   Builds the BentoML service using `bentoml build` and configurations from `bentofile.yaml`. Uses instructions in `Dockerfile.bentoml`.
    *   **Input**: Environment variables from `.env`, configurations from `bentofile.yaml`.
*   **check_models.py**:
    *   Checks available models and their stages in the MLflow Model Registry.
    *   **Input**: MLflow tracking URI, model names.
*   **check_sklearn_compatibility.py**:
    *   Checks scikit-learn version compatibility with MLflow models.
    *   **Input**: MLflow tracking URI, model names.
*   **config.py**:
    *   Initializes DagsHub, sets up the environment, and configures MLflow tracking URI.
    *   **Input**: Environment variables.
    *   **Output**: Configured MLflow and DagsHub integrations.
*   **data.py**:
    *   Loads and preprocesses the wine quality dataset from KaggleHub, splitting it into training and testing sets.
    *   **Input**: None (downloads data from KaggleHub).
    *   **Output**: Training/testing sets, quality order, feature names.
*   **debug_logistic_regression.py**:
    *   Debugging script for the Logistic Regression model (potentially incomplete).
*   **Dockerfile.bentoml**:
    *   Dockerfile for building a BentoML service.
    *   **Input**: `requirements.txt`, `service.py`.
*   **Dockerfile.train**:
    *   Dockerfile for building the training environment.
*   **mlflow_logging.py**:
    *   Logs model parameters, metrics, and artifacts to MLflow.
    *   **Input**: Trained model, model name, parameters, classification report, data.
    *   **Output**: MLflow logs.
*   **models.py**:
    *   Defines model configurations (Logistic Regression, Random Forest).
    *   **Output**: List of model configurations.
*   **new_test.sh**:
    *   Runs tests by executing `pipeline.py` inside a Docker container.
    *   **Input**: Environment variables from `.env`.
*   **output.log**:
    *   Log file for the pipeline output (training and evaluation results).
*   **pipeline.py**:
    *   Orchestrates the ML pipeline: loads data, preprocesses, trains/evaluates models, and logs to MLflow.
    *   **Input**: None (relies on other modules/configurations).
    *   **Output**: Trained models, MLflow logs.
*   **problems.md**:
    *   Documents problems encountered during the project and their solutions.
*   **requirements.txt**:
    *   Lists Python dependencies for the project.
*   **service_fixed.py**:
*   **service.py**:
    *   Defines the BentoML service, loads models from MLflow, and defines API endpoints for predictions.
    *   **Input**: MLflow models.
    *   **Output**: BentoML service.
*   **train_container.log**:
    *   Log file for the training container output.
*   **training.py**:
    *   Trains and evaluates a model based on a given configuration, logging results to MLflow.
    *   **Input**: Model configuration, training data, testing data.
    *   **Output**: Trained model, classification report.
*   **wine_work.py**:
    *   Script for working with wine quality data (data exploration/preprocessing).