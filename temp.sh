#!/bin/bash

# --- Source Environment Variables ---
# Load variables from .env file
if [ -f .env ]; then
    source .env
else
    echo "Error: .env file not found!"
    exit 1
fi

# --- Configuration (Loaded from .env) ---
# TRAIN_IMAGE_NAME is sourced from .env
# MLFLOW_TRACKING_URI is sourced from .env
# MLFLOW_TRACKING_TOKEN is sourced from .env
# CONTAINER_APP_OUTPUT_DIR is sourced from .env

# Define a temporary container name for this interactive session
DEBUG_CONTAINER_NAME="debug-train-env-$(date +%Y%m%d%H%M%S)"


echo "--- Launching Interactive Debugging Container ---"
echo "Image: '$TRAIN_IMAGE_NAME'"
echo "Container Name: '$DEBUG_CONTAINER_NAME'"
echo "Environment variables will be passed from .env"
echo "You will be dropped into a bash shell inside."


# Check if the image exists before trying to run it
if ! docker image inspect "$TRAIN_IMAGE_NAME" >/dev/null 2>&2; then
    echo "Error: Docker image '$TRAIN_IMAGE_NAME' not found."
    echo "Please run ./build-train.sh first to build the image."
    exit 1
fi

# Run interactively (-it), remove on exit (--rm), set name and environment variables, run bash
docker run \
  -it --rm \
  --name "$DEBUG_CONTAINER_NAME" \
  -e MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
  -e MLFLOW_TRACKING_TOKEN="$MLFLOW_TRACKING_TOKEN" \
  -e CONTAINER_APP_OUTPUT_DIR="$CONTAINER_APP_OUTPUT_DIR" \
  "$TRAIN_IMAGE_NAME" \
  /bin/bash # Command to run: launch a bash shell

echo "--- Interactive session ended ---"
# The script exits here after the bash session inside the container finishes (when you type 'exit')