#!/bin/bash

echo "Setting up the ML pipeline..."

# Install dependencies
pip install -r requirements.txt

# Set Python path
export PYTHONPATH=$(pwd)

# Run Data Ingestion
echo "Running main programme..."
python src/components/data_ingestion.py

echo "Pipeline execution completed!"
