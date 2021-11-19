#!/bin/bash

#This is a default ID. No need to change
ID=AADB_AIAG_Extraction

#This is the id of the particular run. Name as you wish
EXPERIMENT_ID=Extraction

#Path to the CSV file containing AVA IDs
#DB=meta/A2P2_FULL_Corrected.CSV
DB=meta/AADB_dataset.csv
#Path to AVA images.
#DATAPATH=/media/koustav/Naihati/Dataset/AVADataSet/
#DATAPATH=/home/ghosalk/Datasets/AVADataSet/
DATAPATH=/data2/AADB/images/datasetImages_originalSize/
#Directory to store the features.
#SAVE_FEAT=/media/koustav/Naihati/Dataset/AVA/Features_8_5x5/
SAVE_FEAT=/data2/ghosalk/AADB/Features_8_5x5

#Feature File Name
FEAT_FILE_NAME=32-bit_INC_RN_V2.h5

#Backbone. Currently supports Inc-ResNet-v2 only. Adding new backbones is trivial.
BASE_MODEL=inceptionresnetv2

#Saved feature data precision
FP=32

#Number of images to extract features from. Use -1 if all images are to be used. Use a smaller value for debugging.
PILOT=-1

 CUDA_VISIBLE_DEVICES=0 python3 -W ignore extract_graph.py --id $ID --db $DB --datapath $DATAPATH --pretrained --exp_id $EXPERIMENT_ID --feature_file_name $FEAT_FILE_NAME\
  --base_model $BASE_MODEL \
  --data_precision $FP \
  --save_feat $SAVE_FEAT --pilot $PILOT --n_workers 4

