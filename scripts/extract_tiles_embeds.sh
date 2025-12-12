#!/bin/bash

# Define variables

SCRIPT="pipeline/00_extract_tile_embeds.py"
PATCH_DIR="C:/Users/manon/Desktop/data/25P18936/HES"
OUT_DIR="data/histostainalign_embeddings"

# Run the Python script
python "$SCRIPT" \
    --patch_dir "$PATCH_DIR" \ 
    --out_dir "$OUT_DIR" \
