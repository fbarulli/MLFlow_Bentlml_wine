#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Source Environment Variables ---
if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo "Error: .env file not found!"
    exit 1
fi

if [ -z "$TRAIN_CONTAINER_NAME_PREFIX" ]; then
    echo "Error: TRAIN_CONTAINER_NAME_PREFIX not set in .env file."
    exit 1
fi
TRAIN_CONTAINER_NAME="${TRAIN_CONTAINER_NAME_PREFIX}-$(date +%Y%m%d%H%M%S)"

if [ -z "$TRAIN_IMAGE_NAME" ]; then
    echo "Error: TRAIN_IMAGE_NAME not set in .env file."
    exit 1
fi

if [ -z "$TRAIN_LOG_FILE" ]; then
    echo "Error: TRAIN_LOG_FILE not set in .env file."
    exit 1
fi


# --- Cleanup Function ---
cleanup (){
    echo "--- Cleaning up ---"
    echo "Attempting to stop and remove container '$TRAIN_CONTAINER_NAME' if running or exited..."
    docker stop "$TRAIN_CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm -f "$TRAIN_CONTAINER_NAME" >/dev/null 2>&2 || true
    echo "Cleanup finished."
    echo "-------------------"
}

# --- Trap for Exiting ---
trap cleanup EXIT INT TERM

# --- Prepare for Run ---
echo "--- Preparing for Run ---"
if [ -f "$TRAIN_LOG_FILE" ]; then
    echo "Deleting previous log file: $TRAIN_LOG_FILE"
    rm "$TRAIN_LOG_FILE"
fi
echo "Log file ready."
echo "-------------------------"

# --- Run Step ---
echo "--- Running Container ---"
echo "Running container '$TRAIN_CONTAINER_NAME' from image '$TRAIN_IMAGE_NAME'"

if ! docker image inspect "$TRAIN_IMAGE_NAME" >/dev/null 2>&2; then
    echo "Error: Docker image '$TRAIN_IMAGE_NAME' not found."
    echo "Please run ./build-train.sh first."
    exit 1
fi

if docker container inspect "$TRAIN_CONTAINER_NAME" >/dev/null 2>&2; then
    echo "Warning: Container '$TRAIN_CONTAINER_NAME' already exists. Removing it."
    docker rm -f "$TRAIN_CONTAINER_NAME" >/dev/null 2>&2 || true
fi

echo "Starting container '$TRAIN_CONTAINER_NAME' in detached mode (-d) with TTY (-t)..."
# Add -t flag here
docker run \
  -d -t \ # <--- Add -t here
  --name "$TRAIN_CONTAINER_NAME" \
  --env-file .env \
  -e PYTHONUNBUFFERED=1 \
  "$TRAIN_IMAGE_NAME" \
  python pipeline.py

RUN_STATUS=$?

if [ $RUN_STATUS -ne 0 ]; then
    echo "Failed to start detached container (Docker exit code $RUN_STATUS). Exiting."
    exit 1
fi

echo "Container started. Monitoring logs for authorization link or completion..."

# --- Monitor Logs for Authorization Link ---
# Use docker logs -f to stream logs
# This will block until the container exits.
# We need to capture the output live and look for the link.
# Reverting to a simpler log streaming that just prints, hoping the link appears.
docker logs -f "$TRAIN_CONTAINER_NAME" 2>&1

# If the container exits, the `docker logs -f` command above will finish.
# At this point, the script continues.

# --- Wait for Container to Exit (Redundant after docker logs -f) ---
# This wait command is actually not needed after docker logs -f, as -f blocks until exit.
# However, keeping it is harmless.
echo "Log monitoring finished. Waiting for container '$TRAIN_CONTAINER_NAME' to complete (if not already stopped)..."
# We already waited for the container to exit via `docker logs -f`.
# The exit code is captured automatically by docker logs -f.
# We can get the exit code from `docker inspect` if needed here, but the script should rely
# on the container exiting to finish the `docker logs -f` block.
# Let's remove the explicit docker wait here for simplicity and rely on the implicit wait of docker logs -f.

# --- Removed explicit docker wait ---
# CONTAINER_EXIT_CODE=$(docker wait "$TRAIN_CONTAINER_NAME")
# WAIT_STATUS=$?
# if [ $WAIT_STATUS -ne 0 ]; then echo "Error waiting..."; exit 1; fi
# echo "Container exited with code: $CONTAINER_EXIT_CODE"
# --- End Removed ---

# Instead of docker wait, get the exit code *after* docker logs -f completes
# Use docker inspect to get the exit code of the container that just finished
# Note: This requires the container *not* to be auto-removed immediately.
# We are already *not* using --rm, so this should work.
CONTAINER_EXIT_CODE=$(docker inspect "$TRAIN_CONTAINER_NAME" --format '{{.State.ExitCode}}')

echo "Container exited with code: $CONTAINER_EXIT_CODE"


# --- Capture Full Logs ---
echo "Capturing full logs to '$TRAIN_LOG_FILE'..."
docker logs "$TRAIN_CONTAINER_NAME" > "$TRAIN_LOG_FILE" 2>&1
LOG_CAPTURE_STATUS=$?

if [ $LOG_CAPTURE_STATUS -ne 0 ]; then
    echo "Warning: Failed to capture logs from container '$TRAIN_CONTAINER_NAME'."
fi


# --- Final Cleanup and Status ---
echo "--- Finalizing ---"
# Check the container's exit code to determine job success
if [ "$CONTAINER_EXIT_CODE" -ne 0 ]; then
    echo "Training job failed inside the container (exit code $CONTAINER_EXIT_CODE)."
    echo "Check '$TRAIN_LOG_FILE' for details."
    exit 1
else
    echo "Training job completed successfully."
    echo "Logs saved to '$TRAIN_LOG_FILE'."
    exit 0
fi