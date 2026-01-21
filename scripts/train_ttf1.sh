#!/bin/bash

# Define variables
DATA_DIR="data/histostainalign_embeddings"
INFO_CSV="C:/Users/manon/Documents/histo/data/datasets/info_csv.csv"
STUDY="pttf153"
DATASET_CSV="C:/Users/manon/Documents/histo/data/datasets"
N_TOKENS=6

# Set CUDA devices and run the Python script
CUDA_VISIBLE_DEVICES=0,1,2,3 python pipeline/01_train_model_with_classification_head.py \
--data_dir "$DATA_DIR" \
--info_csv "$INFO_CSV" \
--study "$STUDY" \
--dataset_csv "$DATASET_CSV" \
--n_tokens "$N_TOKENS" \
--gpu_devices 2 \
--batch_size 12 \
--num_workers 2
