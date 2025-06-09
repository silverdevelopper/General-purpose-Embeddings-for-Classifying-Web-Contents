#!/bin/bash

# train.sh: Train a model for a given dataset and embedding type.

# Exit immediately if any command fails
set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 CONFIG_FILE"
    exit 1
fi

CONFIG_FILE=$1   # Path to the experiment config YAML file

echo "Starting Experiment: ${CONFIG_FILE}..."

# Run each command separately to ensure set -e works
bash scripts/generate_embeddings.sh --modes train "$CONFIG_FILE"
bash scripts/generate_embeddings.sh --modes encode "$CONFIG_FILE"

# Check if dataset.val exists in the YAML file using yq
if yq '.dataset.val' "$CONFIG_FILE" | tr -d '"' | grep -qv 'null'; then
    echo "Found validation dataset, generating val embeddings..."
    bash scripts/generate_embeddings.sh --modes val "$CONFIG_FILE"
fi

bash scripts/train.sh "$CONFIG_FILE"
bash scripts/evaluate.sh "$CONFIG_FILE"

echo "Training complete for experiment: ${CONFIG_FILE}."