#!/bin/bash

# train.sh: Train a model for a given dataset and embedding type.

# Exit immediately if any command fails
set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0  CONFIG_FILE"
    exit 1
fi


CONFIG_FILE=$1   # Path to the experiment config YAML file


echo "Starting Evaluating for experiment: ${CONFIG_FILE}..."
# Set CONFIG and type for running train_model.py
python src/models/test_model.py \
  --config $CONFIG_FILE 

echo "Testing complete for experiment: ${CONFIG_FILE}."