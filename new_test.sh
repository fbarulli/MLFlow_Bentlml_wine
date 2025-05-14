#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Source Environment Variables ---
# Load variables from .env file
# We need them in the host script to pass to Docker run options/name, and also for explicit -e flags
if [ -f .env ]; then
    # Use `set -a` to automatically export variables read by `source`
    # This makes them available to the rest of the script and commands like docker build/run
    set -a
    source .env
    set +a
else
    echo "Error: .env file not found!"
    exit 1
fi

# --- Configuration (Loaded from .env) ---
# TRAIN_IMAGE_NAME is sourced from .env
# MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_TOKEN/PASSWORD are also sourced

# Use a temporary container name for this direct run
DIRECT_CONTAINER_NAME="pipeline-direct-run-$(date +%Y%m%d%H%M%S)"

# Need TRAIN_IMAGE_NAME from .env for the docker run command
if [ -z "$TRAIN_IMAGE_NAME" ]; then
    echo "Error: TRAIN_IMAGE_NAME not set in .env file."
    exit 1
fi

# Need MLflow credentials from .env for explicit passing
if [ -z "$MLFLOW_TRACKING_USERNAME" ] || [ -z "$MLFLOW_TRACKING_TOKEN" ] && [ -z "$MLFLOW_TRACKING_PASSWORD" ]; then
    echo "Error: MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_TOKEN/PASSWORD must be set in .env file."
    exit 1
fi

# Use TOKEN if available, otherwise PASSWORD
AUTH_TOKEN="$MLFLOW_TRACKING_TOKEN"
if [ -z "$AUTH_TOKEN" ]; then
    AUTH_TOKEN="$MLFLOW_TRACKING_PASSWORD"
fi


# --- Cleanup Function ---
# This function stops and removes the specific container instance
cleanup (){
    echo "--- Cleaning up ---"
    echo "Attempting to stop and remove container '$DIRECT_CONTAINER_NAME' if running or exited..."
    # Stop the container first to ensure it's not running when we try to remove it
    docker stop "$DIRECT_CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm -f "$DIRECT_CONTAINER_NAME" >/dev/null 2>&2 || true
    echo "Cleanup finished."
    echo "-------------------"
}

# --- Trap for Exiting ---
# This ensures the cleanup function runs if the script exits normally or is interrupted (like with Ctrl+C)
trap cleanup EXIT INT TERM


echo "--- Running Pipeline Directly (Interactive) ---"
echo "Running container '$DIRECT_CONTAINER_NAME' from image '$TRAIN_IMAGE_NAME'"
echo "You will see output directly. Press Ctrl+C to stop the script and container."


# Check if the image exists before trying to run it
if ! docker image inspect "$TRAIN_IMAGE_NAME" >/dev/null 2>&2; then
    echo "Error: Docker image '$TRAIN_IMAGE_NAME' not found."
    echo "Please run ./build-train.sh first."
    exit 1
fi

# Ensure a container with the same name isn't running (less likely with timestamp, but safer)
if docker container inspect "$DIRECT_CONTAINER_NAME" >/dev/null 2>&2; then
    echo "Warning: Container '$DIRECT_CONTAINER_NAME' already exists. Removing it."
    # Use the cleanup function to remove it safely
    cleanup
fi


# --- Direct docker run command (Interactive) ---
# -it: interactive and allocate a pseudo-TTY (allows seeing output live)
# --rm: remove on exit (important for cleanup)
# --name: unique name for this container run
# --env-file: Read environment variables directly from the .env file on the host (keeps other variables)
# -e: Explicitly set MLflow credentials as environment variables for the container
docker run \
  -it --rm \
  --name "$DIRECT_CONTAINER_NAME" \
  --env-file .env \
  -e MLFLOW_TRACKING_USERNAME="$MLFLOW_TRACKING_USERNAME" \
  -e MLFLOW_TRACKING_PASSWORD="$AUTH_TOKEN" \
  "$TRAIN_IMAGE_NAME" \
  python pipeline.py # <-- Run the main script directly

# The script will block here until the container exits or is stopped (e.g., via Ctrl+C)

echo "--- Container exited ---"
# The trap will run the cleanup function automatically because the script is exiting.