#!/bin/bash

# Define variables
CHECKPOINT_DIR="tangle_ttf1_binary_6_0.01_5e-05_5e-07_12_25_class_head_new"
DATASET_CSV="../prov-gigapath/dataset_csv/ttf1/"
OUTPUT_DIR="slide_embeds_ttf1"
STUDY="ttf1"
MODEL_NAME="model.pt"
TILE_EMBED_DIR="data/histostainalign_embeddings"
SCRIPT="pipeline/02_generate_slide_embeddings.py"

# Run the Python script
python "$SCRIPT" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --dataset_csv "$DATASET_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --study "$STUDY" \
    --model_name "$MODEL_NAME" \
    --tile_embed_dir "$TILE_EMBED_DIR"
