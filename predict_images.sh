#!/bin/bash

# Directory where input images are stored
DIR=/home/ghosalk/Pictures/sample_ava/

# Model Template Name. Choose one from Graph_Models.py
MODEL_NAME=GAT_x3_GATP_MH

# Path to saved weights
SAVED_WEIGHTS=/media/nas/02_Data/Aesthetics/MTL_BACKUP/models/MTL/Exp-GAT_3_GATP_04_40AM_November_21_2020.model_best

python3 predict_images.py --dir $DIR --model_name $MODEL_NAME --start_from $SAVED_WEIGHTS
