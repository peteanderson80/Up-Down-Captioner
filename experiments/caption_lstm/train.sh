#!/bin/bash

GPU_ID=0,1
BASE_DIR=experiments/
DATA_DIR=data/coco_splits/
LOG_DIR=logs/
SNAPSHOT_DIR=snapshots/
NET_NAME=caption_lstm
OUT_DIR=outputs/
VOCAB_FILE=train_vocab.txt
MAX_IT=60000
SCST_MAX_IT=1000

mkdir -p ${LOG_DIR}${NET_NAME}
mkdir -p ${SNAPSHOT_DIR}${NET_NAME}
mkdir -p ${OUT_DIR}${NET_NAME}
   
python -u external/caffe/python/train.py     --solver ${BASE_DIR}${NET_NAME}/solver.prototxt     --gpus ${GPU_ID//,/ }     > ${LOG_DIR}${NET_NAME}/solver.log 2<&1 
    
# Decode the cross entropy trained model
python ./scripts/beam_decode.py   --gpu ${GPU_ID:0:1}       --model ${BASE_DIR}${NET_NAME}/decoder.prototxt     --weights=${SNAPSHOT_DIR}${NET_NAME}/lstm_iter_${MAX_IT}.caffemodel.h5     --vocab ${DATA_DIR}${VOCAB_FILE}     --outfile ${OUT_DIR}/${NET_NAME}/iter_${MAX_IT}.json 

# Self-critical sequence training
python -u external/caffe/python/train.py     --solver ${BASE_DIR}${NET_NAME}/scst_solver.prototxt     --gpus ${GPU_ID//,/ }     --weights=${SNAPSHOT_DIR}${NET_NAME}/lstm_iter_${MAX_IT}.caffemodel.h5     > ${LOG_DIR}${NET_NAME}/scst.log 2<&1
    
# Decode the finished model
python ./scripts/beam_decode.py   --gpu ${GPU_ID:0:1}       --model ${BASE_DIR}${NET_NAME}/decoder.prototxt     --weights=${SNAPSHOT_DIR}${NET_NAME}/lstm_scst_iter_${SCST_MAX_IT}.caffemodel.h5     --vocab ${DATA_DIR}${VOCAB_FILE}     --outfile ${OUT_DIR}/${NET_NAME}/scst_iter_${SCST_MAX_IT}.json


