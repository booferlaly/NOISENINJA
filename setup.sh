#!/bin/bash

# Exit on error
set -e

echo "Setting up development environment..."

# Detect OS
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    IS_WINDOWS=true
else
    IS_WINDOWS=false
fi

# Create and activate virtual environment
if [ "$IS_WINDOWS" = true ]; then
    python -m venv .venv
    source .venv/Scripts/activate
else
    python3 -m venv .venv
    source .venv/bin/activate
fi

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Set environment variables
echo "Configuring environment variables..."
if [ "$IS_WINDOWS" = true ]; then
    setx PYTHONPATH "%PYTHONPATH%;%CD%"
    setx TF_CPP_MIN_LOG_LEVEL "2"
    setx TF_FORCE_GPU_ALLOW_GROWTH "true"
else
    export PYTHONPATH="${PYTHONPATH}:${PWD}"
    export TF_CPP_MIN_LOG_LEVEL=2
    export TF_FORCE_GPU_ALLOW_GROWTH=true
fi

# Create necessary directories
echo "Creating project directories..."
mkdir -p models
mkdir -p collected_data
mkdir -p logs

# Make scripts executable (Unix-like systems only)
if [ "$IS_WINDOWS" = false ]; then
    chmod +x .devcontainer/healthcheck.sh
    chmod +x .devcontainer/setup.sh
fi

# Run health check
echo "Running health check..."
if [ "$IS_WINDOWS" = true ]; then
    bash .devcontainer/healthcheck.sh
else
    .devcontainer/healthcheck.sh
fi

echo "Setup completed successfully!" 