#!/bin/bash

# generate_embeddings.sh: Generate embeddings for a specified dataset using different embedding methods.

# Exit immediately if any command fails
set -e

# Check if a dataset name is provided
if [ "$#" -lt 3 ]; then
    echo "Usage: --mode $0 train|encode  CONFIG_PATHS $2 [experiment_1.yaml] [experiment_2.yaml] [experiment_3.yaml] ..."
    exit 1
fi

if [ "$1" != "--mode" ] && [ "$2" != "train" ] && [ "$2" != "encode" ] && [ "$2" != "val" ]; then
    echo "Invalid mode. Usage: $0 --mode train|encode CONFIG_PATHS  [experiment_1.yaml] [experiment_2.yaml] [experiment_3.yaml] ..."
    exit 1
fi

MODE=$2
DATASET_SETTINGS_PATH=".dataset.train"

if [ "$2" == "encode" ]; then
    DATASET_SETTINGS_PATH=".dataset.test"
fi

if [ "$2" == "val" ]; then
    echo "Generating validation embeddings..."
    DATASET_SETTINGS_PATH=".dataset.val"
fi


shift
shift
EMBEDDINGS_DIR="data/embeddings"   # Path to save embeddings
MODEL_DIR="models/embeddings"      # Path to save embedding models
export TOKENIZERS_PARALLELISM=true  # Enable parallel tokenization
# Ensure the embeddings directory exists
mkdir -p "$EMBEDDINGS_DIR"
mkdir -p "$MODEL_DIR"

# Process each embedding type specified in the arguments
for option in "$@"; do
    CONFIG_PATH=$option
    type=$(yq '.embeddings.type' "$CONFIG_PATH" | tr -d '"')
    DATASET=$(yq $DATASET_SETTINGS_PATH $CONFIG_PATH  | tr -d '"')
    MODEL=$(yq '.dataset.name' $CONFIG_PATH  | tr -d '"')
    echo "Dataset:" $DATASET
    EXPERIMENT=$(yq '.name' $CONFIG_PATH | tr -d '"')
    echo "Embedding Type:" "$type"
    case "--"$type in
        --doc2vec)
            echo "Generating Doc2Vec embeddings for dataset ${DATASET}..."
            DOC2VEC_MODEL_PATH="${MODEL_DIR}/doc2vec_${MODEL}.model"  # Path to save Doc2Vec model
            DOC2VEC_OUTPUT="${EMBEDDINGS_DIR}/doc2vec_${DATASET}_embeddings.csv"
            python src/embeddings/doc2vec_embedding.py \
                --output_path "$DOC2VEC_OUTPUT" \
                --model_save_path "$DOC2VEC_MODEL_PATH"\
                --config "$CONFIG_PATH"\
                --mode $MODE
            echo "Doc2Vec embeddings saved at $DOC2VEC_OUTPUT"
            ;;
        --fasttext)
            echo "Generating FastText embeddings for dataset ${DATASET}..."
            FASTTEXT_MODEL_PATH="${MODEL_DIR}/fasttext_${MODEL}.model" # Path to save FastText model
            FASTTEXT_OUTPUT="${EMBEDDINGS_DIR}/fasttext_${DATASET}_embeddings.csv"
            python src/embeddings/fasttext_embedding.py \
                --output_path "$FASTTEXT_OUTPUT" \
                --model_save_path "$FASTTEXT_MODEL_PATH"\
                --config "$CONFIG_PATH"\
                --mode $MODE
            echo "FastText embeddings saved at $FASTTEXT_OUTPUT"
            ;;
        --hf)
            echo "Generating HuggingFace embeddings for dataset ${DATASET}..."
            HF_OUTPUT="${EMBEDDINGS_DIR}/hf_${DATASET}_embeddings.csv"
            python src/embeddings/hf_embedding.py \
                --output_path "$HF_OUTPUT"\
                --config "$CONFIG_PATH"\
                --mode $MODE
            echo "HuggingFace embeddings saved at $HF_OUTPUT"
            ;;
        --cde)
            CDE_OUTPUT="${EMBEDDINGS_DIR}/cde_${DATASET}_embeddings.csv"
            echo "Generating HuggingFace embeddings for dataset ${DATASET}..."
            python src/embeddings/cde_small_v1_embedding.py \
                --output_path "$CDE_OUTPUT"\
                --config "$CONFIG_PATH"\
                --mode $MODE
            echo "HuggingFace embeddings saved at $CDE_OUTPUT"
            ;;
        --tfidf)
            echo "Generating TF-IDF embeddings for dataset ${DATASET}..."
            TFIDF_OUTPUT="${EMBEDDINGS_DIR}/tfidf_${DATASET}_embeddings.csv"
            TFIDF_MODEL_PATH="${MODEL_DIR}/tfidf_${MODEL}.model"   
            python src/embeddings/tfidf_embedding.py \
                --output_path "$TFIDF_OUTPUT" \
                --model_save_path "$TFIDF_MODEL_PATH"\
                --config "$CONFIG_PATH"\
                --mode $MODE
            echo "TF-IDF embeddings saved at $TFIDF_OUTPUT, model saved at $TFIDF_MODEL_PATH"
            ;;
        --tfidf_cf)
            TFIDFCF_MODEL_SETTINGS_PATH="${MODEL_DIR}/tfidf_cf_${MODEL}.json" # Path to save TF-IDF-CF model
            TFIDFCF_MODEL_PATH="${MODEL_DIR}/tfidf_cf_${MODEL}.model" # Path to save TF-IDF-CF model
            TFIDF_CF_OUTPUT="${EMBEDDINGS_DIR}/tfidf_cf_${DATASET}_embeddings.csv"
            echo "Generating TF-IDF-CF embeddings for dataset ${DATASET}..."
            python src/embeddings/tfidf_cf_embedding.py \
                --output_path "$TFIDF_CF_OUTPUT" \
                --model_save_path "$TFIDFCF_MODEL_PATH"\
                --setting_save_path "$TFIDFCF_MODEL_SETTINGS_PATH"\
                --config "$CONFIG_PATH"\
                --mode $MODE
                echo "TF-IDF-CF embeddings saved at $TFIDF_CF_OUTPUT"
            ;;
        *)
            echo "Unknown option: $type"
            exit 1
            ;;
    esac
done


echo "Embedding generation completed for dataset ${DATASET}."