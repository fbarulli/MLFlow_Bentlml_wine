#!/bin/bash

# --- Source Environment Variables ---
if [ -f .env ]; then
    source .env
else
    echo "Error: .env file not found!"
    exit 1
fi

# --- Configuration (Now loaded from .env) ---
# TRAIN_IMAGE_NAME is now sourced from .env

# --- Build Step ---
echo "--- Building Image ---"
echo "Building Docker image '$TRAIN_IMAGE_NAME' using Dockerfile.train"
docker build -t "$TRAIN_IMAGE_NAME" -f Dockerfile.train .
BUILD_STATUS=$?

if [ $BUILD_STATUS -ne 0 ]; then
    echo "Docker build failed. Check build logs above. Exiting."
    exit 1
fi
echo "Build successful. Image '$TRAIN_IMAGE_NAME' created."
echo "--------------------"