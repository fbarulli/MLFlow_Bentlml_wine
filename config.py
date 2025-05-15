# config.py
import os
import logging
import traceback
import mlflow
import dagshub
import dagshub.auth
import json
import pathlib
import stat # Import the stat module for chmod constants

# Set up logger for the config module
# Ensure this logger is correctly configured in pipeline.py's basicConfig
logger = logging.getLogger(__name__)

def init_dagshub_mlflow():
    """
    Initializes DagsHub MLflow integration from environment variables with explicit auth handling.
    Requires MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, and MLFLOW_TRACKING_TOKEN/PASSWORD to be set.
    Includes debugging steps for token verification inside the container.
    Creates .dagshub/ and .netrc files for DagsHub library and DVC/Git command-line authentication.
    """
    logger.info("Starting DagsHub/MLflow initialization...")
    try:
        # --- Get credentials from environment variables ---
        # os.getenv reads from the container's environment, populated by docker run --env-file
        dagshub_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip('"').strip("'")
        username = os.getenv("MLFLOW_TRACKING_USERNAME", "").strip('"').strip("'")

        # --- DEBUGGING: Print raw token values from environment ---
        # These show exactly what os.getenv is returning before any processing.
        # Useful for debugging issues with the .env file or --env-file parsing.
        raw_token_mlflow = os.getenv("MLFLOW_TRACKING_TOKEN")
        raw_token_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
        logger.info(f"DEBUG: Raw MLFLOW_TRACKING_TOKEN: '{raw_token_mlflow}' (Type: {type(raw_token_mlflow)})")
        logger.info(f"DEBUG: Raw MLFLOW_TRACKING_PASSWORD: '{raw_token_password}' (Type: {type(raw_token_password)})")
        # --- END DEBUGGING ---

        # Prefer MLFLOW_TRACKING_TOKEN, fallback to MLFLOW_TRACKING_PASSWORD
        token = raw_token_mlflow or raw_token_password

        # --- Process the token string ---
        # Clean the token string by removing quotes and whitespace.
        # This fixes issues caused by comments on the same line in .env files.
        if token:
            # MORE ROBUST STRIPPING: Remove all occurrences of single/double quotes
            # Then strip any leading/trailing whitespace (including spaces from comments if they were included)
            token = token.replace('"', '').replace("'", "").strip()

            # --- DEBUGGING: Print processed token value ---
            # This is the exact string value being used for authentication attempts below.
            # Compare this to the token string that worked with 'curl'.
            logger.info(f"DEBUG: Processed token used for auth: '{token}' (Length: {len(token)})")
            if len(token) > 10: # Avoid printing excessively long tokens, show start/end
                 logger.info(f"DEBUG: Processed token (partial): '{token[:5]}...{token[-5:]}'")
            # WARNING: Printing the full token like this is a security risk in production logs.
            # Remove these DEBUG lines once authentication is working.
            # --- END DEBUGGING ---
        else:
             logger.error("MLFLOW_TRACKING_TOKEN or MLFLOW_TRACKING_PASSWORD environment variable is not set (value is None or empty).")
             # Raise an error immediately if no token is found, as authentication will fail.
             raise ValueError("DagsHub/MLflow token is not set in environment variables.")

        # Check if username is also present (token alone might not be enough for all steps)
        if not username:
             logger.error("MLFLOW_TRACKING_USERNAME environment variable is not set (value is empty).")
             raise ValueError("DagsHub/MLflow username is not set in environment variables.")

        # Also check if tracking URI is set
        if not dagshub_uri:
             logger.error("MLFLOW_TRACKING_URI environment variable is not set (or is empty).")
             raise ValueError("MLFLOW_TRACKING_URI is not set.")


        # Set various environment variables that might be used by dagshub or mlflow libs internally
        # Ensure these are set in the Python process environment for subprocesses or libraries
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token # Use the processed token
        os.environ["DAGSHUB_TOKEN"] = token
        os.environ["DAGSHUB_USER_TOKEN"] = token
        os.environ["DAGSHUB_USERNAME"] = username
        logger.info("Set MLFLOW/DAGSHUB environment variables in Python process.")


        # --- Attempt to create credential files (~/.dagshub and ~/.netrc) ---
        # This helps command-line tools like DVC/Git authenticate via HTTPS non-interactively.
        try:
            home_dir = os.path.expanduser("~") # In Docker, this is usually /root
            dagshub_dir = os.path.join(home_dir, ".dagshub")
            os.makedirs(dagshub_dir, exist_ok=True)
            logger.info(f"Ensured directory exists: {dagshub_dir}")

            # Create credentials file (used by dagshub library)
            cred_file = os.path.join(dagshub_dir, "credentials")
            with open(cred_file, "w") as f:
                f.write(f"login: {username}\npassword: {token}")
            # Set restrictive permissions (owner read/write)
            os.chmod(cred_file, stat.S_IRUSR | stat.S_IWUSR) # 0o600
            logger.info(f"Created credentials file at {cred_file}")

            # Create token cache file (used by dagshub library)
            token_file = os.path.join(dagshub_dir, "token_cache.json")
            token_data = {
                "access_token": token,
                "token_type": "Bearer", # Use "Bearer" token type
                "username": username
            }
            with open(token_file, "w") as f:
                json.dump(token_data, f)
            # Set restrictive permissions (owner read/write)
            os.chmod(token_file, stat.S_IRUSR | stat.S_IWUSR) # 0o600
            logger.info(f"Created token cache at {token_file}")

            # Also try the .session approach (might be used by older dagshub versions)
            session_file = os.path.join(dagshub_dir, ".session")
            with open(session_file, "w") as f:
                f.write(token)
            # Set restrictive permissions (owner read/write)
            os.chmod(session_file, stat.S_IRUSR | stat.S_IWUSR) # 0o600
            logger.info(f"Created session file at {session_file}")

            # Create .netrc file for DVC/Git HTTPS authentication
            netrc_path = os.path.expanduser("~/.netrc")
            dagshub_hostname = "dagshub.com" # Standard hostname for DagsHub remotes

            logger.info(f"Attempting to create .netrc file at {netrc_path} for DVC/Git auth.")
            # Use 'w' mode to overwrite any existing .netrc for a clean state
            with open(netrc_path, "w") as f:
                # Standard format for machine login password
                f.write(f"machine {dagshub_hostname}\n")
                f.write(f"login {username}\n")
                f.write(f"password {token}\n")

            # Set restrictive permissions (owner read/write only). ESSENTIAL for .netrc security.
            os.chmod(netrc_path, stat.S_IRUSR | stat.S_IWUSR) # 0o600
            logger.info(f"Successfully created and secured .netrc file at {netrc_path}")

        except Exception as e:
            # Log the error but don't necessarily fail setup just yet, as env vars might be enough
            # If DVC/Git via HTTPS is critical and fails here, you might want to raise.
            logger.warning(f"Error creating credential files (~/.dagshub, ~/.netrc): {str(e)}")


        # Parse repository information from URI for dagshub.init
        repo_owner = None
        repo_name = None

        logger.info(f"Attempting to parse repo owner/name from MLFLOW_TRACKING_URI: {dagshub_uri}")

        if dagshub_uri and "dagshub.com" in dagshub_uri:
            # Robust parsing by finding "dagshub.com" index
            try:
                # Remove scheme and mlflow+ prefix, split by /, filter empty parts
                # This handles http://, https://, and mlflow+https://
                uri_path_parts = [p for p in dagshub_uri.replace("mlflow+", "").replace("https://", "").replace("http://", "").split('/') if p]
                # Find the index of the hostname
                dagshub_index = -1
                for i, part in enumerate(uri_path_parts):
                     if part == "dagshub.com":
                          dagshub_index = i
                          break

                # The owner should be the part after the hostname, and repo after owner
                if dagshub_index != -1 and len(uri_path_parts) > dagshub_index + 2:
                    repo_owner = uri_path_parts[dagshub_index + 1]
                    repo_name_part = uri_path_parts[dagshub_index + 2]
                    # Remove .mlflow suffix if present
                    if repo_name_part.endswith(".mlflow"):
                        repo_name = repo_name_part[:-len(".mlflow")]
                    else:
                        repo_name = repo_name_part
                    logger.info(f"Parsed owner: {repo_owner}, Final repo name: {repo_name}")
                else:
                    logger.warning(f"URI '{dagshub_uri}' does not have enough parts after 'dagshub.com'. Parts found: {uri_path_parts}")
            except Exception as parse_error:
                 logger.warning(f"Error parsing URI '{dagshub_uri}': {str(parse_error)}")

        else:
            logger.warning(f"MLFLOW_TRACKING_URI '{dagshub_uri}' does not contain 'dagshub.com' or is empty/invalid. Cannot parse repo details for explicit dagshub.init.")

        # --- Attempt direct authentication using dagshub library methods ---
        # This step often triggers an API request to validate the token.
        # If this fails, the token is likely bad or there's a connectivity issue.
        # Based on your logs, this step is now succeeding with the cleaned token.
        try:
            logger.info("Attempting direct authentication using dagshub.auth methods...")
            # Clear any cached auth states the library might hold before explicit init
            # dagshub.auth.INTEGRATIONS.clear() # This attribute not found in the installed dagshub version, commented out

            # Try to use direct authentication methods if they exist
            if hasattr(dagshub.auth, 'add_app_token'):
                logger.info("Using add_app_token method...")
                # This call often makes an API request to validate the token
                dagshub.auth.add_app_token(token)
                logger.info("dagshub.auth.add_app_token succeeded.") # If you see this, this specific call worked.

            elif hasattr(dagshub.auth, 'save_token'): # save_token might be older or an alternative
                 logger.info("Using save_token method...")
                 dagshub.auth.save_token(token)
                 logger.info("dagshub.auth.save_token succeeded.") # If you see this, this specific call worked.
            else:
                 logger.warning("Neither add_app_token nor save_token method found in dagshub.auth module.")

        except Exception as auth_err:
            # Log the specific authentication error details
            logger.error(f"Direct dagshub.auth attempt failed:")
            logger.error(f"Exception Type: {type(auth_err)}")
            try:
                logger.error(f"Exception Args: {auth_err.args}")
            except:
                 logger.error("Could not retrieve exception arguments.")
            try:
                 # Attempt string conversion, handle the case where __str__ might fail
                 err_str = str(auth_err)
                 logger.error(f"Exception String: {err_str}")
            except Exception as inner_e:
                 logger.error(f"Could not get exception string: {inner_e}")

            # Re-raise the specific authentication error so the main block catches it.
            # If this error occurs, authentication failed, so the rest of the script likely can't proceed.
            raise auth_err


        # --- Configure MLflow tracking URI ---
        # This was checked earlier, but setting it again ensures MLflow library is configured.
        if dagshub_uri:
            mlflow.set_tracking_uri(dagshub_uri)
            logger.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
        else:
             # This case should ideally be handled by init_dagshub_mlflow, but kept as safeguard
             logger.error("MLFLOW_TRACKING_URI environment variable is not set (or is empty after stripping quotes).")
             # This is a critical error, should have been caught earlier, but re-raise if somehow missed
             raise ValueError("MLFLOW_TRACKING_URI is not set.")


        # --- Initialize DagsHub library (might interact with repo/DVC) ---
        # This is often needed for dagshub:// DVC remotes or library-specific features.
        # It should be called AFTER setting MLflow tracking URI and creating .netrc/.dagshub files.
        try:
            if repo_owner and repo_name:
                logger.info(f"Initializing DagsHub integration for {repo_owner}/{repo_name} using explicit init...")
                # Use the repo_owner/repo_name parsed from the URI for explicit init
                dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
                logger.info("DagsHub MLflow integration initialized successfully (explicit).")
            else:
                logger.warning("Could not determine valid repo owner/name from URI. Attempting implicit DagsHub init...")
                # Attempt implicit init as a fallback, relies on env vars/config set earlier
                dagshub.init(mlflow=True)
                logger.info("DagsHub implicit init successful.")

        except Exception as init_err:
             logger.error(f"DagsHub init failed: {str(init_err)}")
             logger.error(traceback.format_exc())
             # If dagshub.init fails, it's a critical error for DVC/MLflow integration
             raise init_err # Re-raise the initialization error


    except Exception as e:
        # This catches any exceptions raised within this function that weren't specifically caught
        # and re-raised earlier (like the initial credential check failure, or re-raised auth/init errors)
        logger.error(f"Critical error during DagsHub/MLflow initialization: {str(e)}")
        logger.error(traceback.format_exc())
        # Re-raise the exception so the calling script/process knows init failed
        raise # Re-raise the original exception


def setup_environment():
    """
    Sets up the environment, including the output directory and MLflow tracking URI.
    Assumes dagshub.init_dagshub_mlflow has been called prior if needed.
    """
    logger.info("Setting up environment...")
    output_dir = os.getenv("CONTAINER_APP_OUTPUT_DIR")
    if not output_dir:
         logger.error("CONTAINER_APP_OUTPUT_DIR environment variable is not set.")
         raise ValueError("CONTAINER_APP_OUTPUT_DIR environment variable is not set.")

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Outputs folder created at {output_dir}")

    # Get MLflow Tracking URI from environment again to be sure, and set it
    # This redundant call to set_tracking_uri is okay, confirms URI is picked up
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        tracking_uri = tracking_uri.strip('"').strip("'")

    if not tracking_uri:
         # This case should ideally be handled by init_dagshub_mlflow, but kept as safeguard
         logger.error("MLFLOW_TRACKING_URI environment variable is not set (or is empty after stripping quotes).")
         raise ValueError("MLFLOW_TRACKING_URI environment variable is not set.")

    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI confirmed and set to: {mlflow.get_tracking_uri()}")

    # Enable Scikit-learn autologging
    # Note: autolog() should generally be called *before* you train your model
    # If your pipeline.py trains the model *after* calling setup_environment, this is correct.
    # mlflow.sklearn.autolog(log_models=False) # log_models=False to avoid logging large models directly
    logger.info("MLflow Scikit-learn autologging disabled.")

    return output_dir

# Add other config-related functions here if needed