#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define variables
IMAGE_NAME="yourusername/bookscanner:latest"

echo "Building the Docker image..."
docker build -t $IMAGE_NAME .

echo "Pushing the Docker image to Docker Hub..."
docker push $IMAGE_NAME

echo "Deploying the Docker stack to Swarm..."
docker stack deploy -c stack.yml bookscanner_stack

echo "Deployment is complete."
