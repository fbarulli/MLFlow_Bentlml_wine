#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Source Environment Variables ---
# Load variables from .env file
if [ -f .env ]; then
    # Using 'set -a' to automatically export variables is good practice if
    # the script needs to pass these variables to child processes directly,
    # but docker run --env-file is used elsewhere, so just 'source' is fine here.
    source .env
else
    echo "Error: .env file not found!"
    exit 1
fi

# --- Configuration (Now loaded from .env) ---
# TRAIN_IMAGE_NAME is now sourced from .env
if [ -z "$TRAIN_IMAGE_NAME" ]; then
    echo "Error: TRAIN_IMAGE_NAME not set in .env file."
    exit 1
fi

# --- Prune Build Cache (Optional Cleanup) ---
# Cleans up unused build cache data before starting a new build.
# Adding || true prevents the script from exiting if pruning fails (e.g., no cache to prune).
echo "--- Pruning Docker Build Cache ---"
# Using docker builder prune specifically targets build cache
docker system prune -f || true
echo "Build cache pruning finished."
echo "--------------------------------"


# --- Build Step ---
echo "--- Building Image ---"
# Added --no-cache flag to force a fresh build without using cache layers.
# This ensures the latest code changes in COPY steps are definitely included.
# Remove --no-cache for faster builds once the cache issue is resolved.
echo "Building Docker image '$TRAIN_IMAGE_NAME' using Dockerfile.train with --no-cache"
# Add --progress=plain to see each step if needed for debugging cached steps
docker build --no-cache -t "$TRAIN_IMAGE_NAME" -f Dockerfile.train .
BUILD_STATUS=$?

if [ $BUILD_STATUS -ne 0 ]; then
    echo "Docker build failed. Check build logs above. Exiting."
    # The set -e should handle this, but explicit exit is clearer after checking status.
    exit 1
fi

echo "Build successful. Image '$TRAIN_IMAGE_NAME' created."
echo "--------------------"

# Script exits here on success (status 0)