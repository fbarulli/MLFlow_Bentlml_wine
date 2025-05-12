#!/bin/bash




# run this when Dockerfile.train or requirements.txt changes





TRAIN_IMAGE_NAME="wine-trainer"




echo " -------- Build Image--------"
echo "Building '$TRAIN_IMAGE_NAME' using Dockerfile.train"
docker build -t "$TRAIN_IMAGE_NAME" -f Dockerfile.train . #-f specifies dockerfile
BUILD_STATUS=$?

if [$BUILD_STATUS -ne 0]; then
    echo "Docker build failed"
fi
echo " Great Success"
echo "-----------------"
