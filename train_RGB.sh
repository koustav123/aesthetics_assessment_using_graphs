#!/bin/bash

#This is a default ID. No need to change
ID=AADB_AIAG_RGB

#This is the id of the particular run. Name as you wish
EXPERIMENT_ID=ResNet18

# Model name. Check Graph_Models.py for other model names.
MODEL=Bug_Test

#Path to the CSV file containing AVA IDs
#DB=meta/A2P2_FULL_Corrected.CSV
DB=meta/AADB_dataset.csv

#Path to AVA images. Note, that images are not used here directly. Instead, this is used to check for missing files in the dataloader.
#DATAPATH=/home/ghosalk/Datasets/AVADataSet/
DATAPATH=/data2/AADB/images/datasetImages_originalSize/
#Path to save the trained models
SAVE=/data2/ghosalk/AVA/models/

#Path to save tensorboard files
SAVE_VISUALS=/data2/ghosalk/AVA/visuals/debug/

#Path to load the feature graph from
#FEATURE_PATH=/data2/ghosalk/AVA/Features_8_5x5/INC_RN_V2.h5
#FEATURE_PATH=/data2/ghosalk/AADB/Features_8_5x5/32-bit_INC_RN_V2.h5
#Initial Learning Rate
LR=1e-3

#Batch size for training and validation
BATCH_SIZE=8
VAL_BATCH_SIZE=32

#Validate after every N epochs
VAL_AFTER=101

OPTIMIZER=ADAM

#Training data precision
FP=32

#Number of images to extract features from. Use -1 if all images are to be used. Use a smaller value for debugging.
PILOT=100

#Number of workers to use in multiprocessing dataloader
WORKERS=4

# Backbone. Currently supports Inc-ResNet-v2 only. Adding new backbones is trivial.
# Note, backbone is not used here but this is to adjust GNN input parameters accordingly
BASE_MODEL=resnet18

#Loss Weights. We tried both EMD and MSE loss. The default is MSE
W_MSE=1
W_EMD=0

#Number of output layers
A2_D=1

 CUDA_VISIBLE_DEVICES=2 python3 -W ignore train.py --id $ID --exp_id $EXPERIMENT_ID --db $DB --datapath $DATAPATH --save $SAVE --save_visuals $SAVE_VISUALS \
  --base_model $BASE_MODEL  --A2_D $A2_D --model_name $MODEL \
  --lr $LR --batch_size $BATCH_SIZE --batch_size_test $VAL_BATCH_SIZE --optimizer $OPTIMIZER --data_precision $FP --val_after_every $VAL_AFTER --n_workers $WORKERS \
  --w_emd $W_EMD --w_mse $W_MSE \
  --pilot $PILOT --num_epochs 100 --aug S
#################################################################################################