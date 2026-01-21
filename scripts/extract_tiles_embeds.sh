#!/bin/bash

# Define variables
SCRIPT="pipeline/00_extract_tile_embeds.py"
PATCH_DIR="C:/Users/manon/Desktop/data/patches_bloc_60/"
OUT_DIR="data/ttf1_embeddings"

# Run the Python script
python "$SCRIPT" \
    --patch_dir "$PATCH_DIR" \
    --out_dir "$OUT_DIR" \
