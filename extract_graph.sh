#!/bin/bash

#This is a default ID. No need to change
ID=AIAG_Extraction

#This is the id of the particular run. Name as you wish
EXPERIMENT_ID=Extraction

#Path to the CSV file containing AVA IDs
DB=meta/A2P2_FULL_Corrected.CSV

#Path to AVA images.
DATAPATH=/path/to/images/

#Directory to store the features.
SAVE_FEAT=dump/

#Feature File Name
FEAT_FILE_NAME=INC_RN_V2.h5

#Backbone. Currently supports Inc-ResNet-v2 only. Adding new backbones is trivial.
BASE_MODEL=inceptionresnetv2

#Saved feature data precision
FP=16

#Number of images to extract features from. Use -1 if all images are to be used. Use a smaller value for debugging.
PILOT=1000

 CUDA_VISIBLE_DEVICES=1 python3 -W ignore extract_graph.py --id $ID --db $DB --datapath $DATAPATH --pretrained --exp_id $EXPERIMENT_ID --feature_file_name $FEAT_FILE_NAME\
  --base_model $BASE_MODEL \
  --data_precision $FP \
  --save_feat $SAVE_FEAT --pilot $PILOT --n_workers 4

