import os
import logging
import traceback
import mlflow
import dagshub
from dagshub.auth import get_token, save_token

logger = logging.getLogger(__name__)

def init_dagshub_mlflow():
    """
    Initializes DagsHub MLflow integration from environment variables.
    Requires MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, and MLFLOW_TRACKING_TOKEN to be set.
    Parses repo owner and name from the URI for dagshub.init.
    """
    try:
        dagshub_uri = os.getenv("MLFLOW_TRACKING_URI")
        username = os.getenv("MLFLOW_TRACKING_USERNAME")
        token = os.getenv("MLFLOW_TRACKING_TOKEN")  # Get the token from environment
        
        # Make sure token is set in all supported environment variables
        if token:
            os.environ["DAGSHUB_TOKEN"] = token
            os.environ["MLFLOW_TRACKING_PASSWORD"] = token
            # Manually save the token to avoid OAuth flow
            save_token(token)
            logger.info("Token has been set in environment and saved to local configuration.")
        else:
            logger.warning("No token found in environment variables, authentication may fail.")
        
        if username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = username
            os.environ["DAGSHUB_USERNAME"] = username
        
        # Add stripping of potential quotes
        if dagshub_uri:
             dagshub_uri = dagshub_uri.strip('"').strip("'")

        repo_owner = None
        repo_name = None

        logger.info(f"Attempting to parse repo owner/name from MLFLOW_TRACKING_URI: {dagshub_uri}")

        if dagshub_uri and "dagshub.com" in dagshub_uri:
             parts = dagshub_uri.split('/')
             if len(parts) >= 5:
                repo_owner = parts[3]
                repo_name_part = parts[4]
                if repo_name_part.endswith(".mlflow"):
                    repo_name = repo_name_part[:-len(".mlflow")]
                else:
                    repo_name = repo_name_part

                logger.info(f"Parsed owner: {repo_owner}, Final repo name: {repo_name}")
             else:
                 logger.warning(f"MLFLOW_TRACKING_URI '{dagshub_uri}' does not have enough parts after splitting by '/'. Cannot parse owner/repo reliably.")
        else:
             logger.warning(f"MLFLOW_TRACKING_URI '{dagshub_uri}' does not contain 'dagshub.com' or is empty.")

        if repo_owner and repo_name:
            # Pre-check if a token is already available in dagshub configuration
            stored_token = get_token()
            if stored_token is None and token:
                logger.info("No stored token found but we have an environment token. Saving it...")
                save_token(token)
            
            logger.info(f"Initializing DagsHub for MLflow with parsed details...")
            
            # Try to set up credentials file as an additional authentication method
            try:
                dagshub_dir = os.path.expanduser("~/.dagshub")
                os.makedirs(dagshub_dir, exist_ok=True)
                cred_file = os.path.join(dagshub_dir, "credentials")
                if not os.path.exists(cred_file) and username and token:
                    with open(cred_file, "w") as f:
                        f.write(f"login: {username}\npassword: {token}")
                    os.chmod(cred_file, 0o600)
                    logger.info("Created DagsHub credentials file as fallback authentication method.")
            except Exception as cred_err:
                logger.warning(f"Failed to create credentials file: {str(cred_err)}")
            
            # Initialize DagsHub MLflow integration
            dagshub.init(
                repo_owner=repo_owner, 
                repo_name=repo_name, 
                mlflow=True
            )
            logger.info("DagsHub MLflow integration initialized successfully.")
        else:
            # If parsing failed or owner/repo are missing, log error and raise
            logger.error(f"Could not determine valid repo owner/name from MLFLOW_TRACKING_URI '{dagshub_uri}' for dagshub.init. Ensure URI is in the format like 'mlflow+https://dagshub.com/<owner>/<repo>.mlflow'.")
            raise ValueError(f"Failed to parse repo owner/name from URI: {dagshub_uri}")

    except Exception as e:
        logger.error(f"Error initializing DagsHub MLflow integration: {str(e)}")
        logger.error(traceback.format_exc())
        # Re-raise as this initialization seems necessary for registry
        raise


def setup_environment():
    """
    Sets up the environment, including the output directory and MLflow tracking URI.
    Assumes dagshub.init_dagshub_mlflow has been called prior if needed.
    """
    output_dir = os.getenv("CONTAINER_APP_OUTPUT_DIR")
    if not output_dir:
         logger.error("CONTAINER_APP_OUTPUT_DIR environment variable is not set.")
         raise ValueError("CONTAINER_APP_OUTPUT_DIR environment variable is not set.")

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Outputs folder created at {output_dir}")

    # Get MLflow Tracking URI from environment
    # This redundant call to set_tracking_uri is okay, confirms URI is picked up
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        tracking_uri = tracking_uri.strip('"').strip("'")

    if not tracking_uri:
         logger.error("MLFLOW_TRACKING_URI environment variable is not set (or is empty after stripping quotes).")
         raise ValueError("MLFLOW_TRACKING_URI environment variable is not set.")

    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

    mlflow.sklearn.autolog(log_models=False)
    logger.info("MLflow Scikit-learn autologging enabled (excluding model logging).")

    return output_dir