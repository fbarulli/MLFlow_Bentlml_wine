# MLflow Tracking Server Setup with Docker Compose

This repository provides a containerized setup for a self-hosted MLflow tracking server using Docker Compose. It includes a database, artifact storage, and a reverse proxy to create a complete, persistent environment for logging ML experiments.

## File Breakdown

*   **`.dockerignore`**: Specifies files and directories to exclude when building Docker images, helping keep images small and build contexts clean.
*   **`.gitignore`**: Specifies intentionally untracked files that Git should ignore (e.g., local environment files, build artifacts).
*   **`docker-compose.yml`**: Defines and orchestrates the multi-container MLflow tracking server application, including the MLflow service, PostgreSQL database, MinIO artifact store, and Nginx reverse proxy.
*   **`mlflow/`**: Directory containing files specifically for building the custom Docker image for the MLflow tracking server application.
*   **`mlflow/Dockerfile`**: Instructions for building the Docker image for the MLflow tracking server service. It sets up the environment and copies necessary files.
*   **`mlflow/requirements.txt`**: Lists the Python dependencies required to run the MLflow tracking server application within its container.
*   **`nginx/`**: Directory containing configuration files for the Nginx reverse proxy service.
*   **`nginx/nginx.conf`**: Configuration file for the Nginx reverse proxy, setting up how external traffic is routed, typically directing requests to the MLflow UI and API.
*   **`src/`**: Directory containing example source code to interact with the MLflow server.
*   **`src/run.py`**: An example Python script demonstrating how to initialize an MLflow client and log parameters, metrics, and potentially artifacts to the tracking server defined by the Docker Compose setup.