#!/bin/bash

# preprocess.sh: Preprocess a raw dataset for a specified dataset name.

# Exit immediately if any command fails
set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 DATASET_NAME"
    exit 1
fi

DATASET_NAME=$1  # Name of the dataset (e.g., "dataset1")
RAW_DATASET_PATH="data/raw/${DATASET_NAME}.csv"         # Path to raw dataset
PROCESSED_DATASET_PATH="data/processed/${DATASET_NAME}.csv" # Path to save preprocessed dataset
TEXT_COLUMN_NAME="text"                                # Column containing text data

echo "Starting preprocessing for ${DATASET_NAME}..."
python data/preprocess.py \
  --input_path $RAW_DATASET_PATH \
  --output_path $PROCESSED_DATASET_PATH \
  --text_column $TEXT_COLUMN_NAME

echo "Preprocessed dataset saved at $PROCESSED_DATASET_PATH"