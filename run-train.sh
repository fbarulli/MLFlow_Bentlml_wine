#!/bin/bash

TRAIN_IMAGE_NAME="wine-trainer"
TRAIN_CONTAINER_NAME="wine-trainer-running-$(date +%Y%m%d%H%M%S)"
LOG_FILE="train_container.log"

MLFLOW_TRACKING_URI="https://dagshub.com/fbarulli/MLFlow_backend"
MLFLOW_TRACKING_TOKEN="1de275ede522e8bd56e558a81ecd32a803b7ba64" 

CONTAINER_APP_OUTPUT_DIR="/app/outputs"
