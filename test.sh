#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Building the Docker image..."
docker-compose build

echo "Starting the Docker container with Docker Compose..."
docker-compose up -d

echo "Running tests (if any)..."
# Add commands to run your tests here. For example, if using pytest:
# docker-compose exec bookscanner pytest

echo "Docker Compose setup is running on http://localhost:8502"
echo "Press Ctrl+C to stop the Docker Compose setup."

# Keep the script running until interrupted
trap "docker-compose down" INT
wait
