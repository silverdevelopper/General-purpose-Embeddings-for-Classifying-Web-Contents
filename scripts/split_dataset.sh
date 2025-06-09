#!/bin/bash

# split_dataset.sh: Split the preprocessed dataset for a given dataset name into train/test sets.

# Exit immediately if any command fails
set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 DATASET_NAME"
    exit 1
fi

DATASET_NAME=$1  # Name of the dataset (e.g., "dataset1")
PROCESSED_DATASET_PATH="data/embeddings/${DATASET_NAME}.csv"            # Path to embedding dataset
TRAIN_PATH="data/splits/${DATASET_NAME}_embeddings_train.csv"           # Path to save train dataset
TEST_PATH="data/splits/${DATASET_NAME}_embeddings_test.csv"             # Path to save test dataset
LABEL_COLUMN_NAME="category"                                            # Column containing target labels
TEST_SIZE=0.2                                                           # Proportion of test data (20%)
RANDOM_STATE=4112                                                       # Seed for reproducibility

echo "Splitting dataset: ${DATASET_NAME}..."
python data/split_dataset.py \
  --input_path $PROCESSED_DATASET_PATH \
  --output_train_path $TRAIN_PATH \
  --output_test_path $TEST_PATH \
  --label_column $LABEL_COLUMN_NAME \
  --test_size $TEST_SIZE \
  --random_state $RANDOM_STATE

echo "Train dataset saved at $TRAIN_PATH"
echo "Test dataset saved at $TEST_PATH"