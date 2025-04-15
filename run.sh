#!/bin/bash

if [ -f ".env" ]; then
  source .env
else
  echo "Error: .env file not found."
  exit 1
fi

if [[ "$CUDA_AVAILAIBLE" == "Yes" ]]; then
  echo "GPU detected. Running docker-compose with GPU support..."
  docker compose -f docker-compose-gpu.yml up
else
  echo "No GPU detected. Running docker-compose without GPU support..."
  docker compose -f docker-compose-cpu.yml up
fi