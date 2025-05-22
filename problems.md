# Problems Encountered

- `bentoml: command not found`: The `bentoml` command was not found, indicating that BentoML was not installed in the environment.
  *Fix:* BentoML was already installed.
- `TypeError: PythonOptions.__init__() got an unexpected keyword argument 'env'`: The `env` keyword is not a valid option within the `python` section of the `bentofile.yaml` file.
  *Fix:* Removed the `env` section from `bentofile.yaml` and used a `build.sh` script to set environment variables.
- `mlflow.tracking.registry.UnsupportedModelRegistryStoreURIException`: The MLflow tracking URI is not supported for model registry operations. The URI scheme `mlflow+https` is not supported.
  *Fix:* Removed the `mlflow+` prefix from the `MLFLOW_TRACKING_URI` environment variable in `service.py` and set the tracking URI using `mlflow.set_tracking_uri()` at the beginning of the file.
- `NameError: name 'os' is not defined`: The `os` module was not imported in `service.py`.
  *Fix:* Added `import os` to the `service.py` file.
- `TypeError: load_model() got an unexpected keyword argument 'tracking_uri'`: The `tracking_uri` argument is not supported by the `mlflow.sklearn.load_model()` function in this version of MLflow.
  *Fix:* Removed the `tracking_uri` argument from the `mlflow.sklearn.load_model()` function calls in `service.py`.
- `mlflow.exceptions.MlflowException: No versions of model with name 'tracking-wine-logisticregression' and stage 'Staging' found`: The models were not found in the MLflow registry when building the BentoML service.
  *Attempted Fix:* Explicitly set the tracking URI using `mlflow.set_tracking_uri(tracking_uri)` before loading the models in `service.py`. Moved the environment variable loading and tracking URI setting logic to the beginning of the `service.py` file.
  *Attempted Fix:* Explicitly create an `MlflowClient` in `service.py` and use it to load the models. Removed the `mlflow_client` argument from the `mlflow.sklearn.load_model()` function calls.
- `line 72: ---: No such file or directory`: The `docker run` command in `run-train.sh` had incorrect syntax.
  *Fix:* Corrected the syntax of the `docker run` command in `run-train.sh`.