#!/bin/bash

# Exit script instantly if any error is encountered
set -e

function downloadFileFromS3() {
    pipenv run aws s3 cp $1 $2
}

function downloadDirFromS3() {
    pipenv run aws s3 cp $1 $2 --recursive
}

function uploadFileToS3() {
    pipenv run aws s3 cp $1 $2 
}

#Sync folder to s3
function syncToS3() {
    pipenv run aws s3 sync $1 $2 --delete
}

function syncToS3Backround() {
    while true
    do
        syncToS3 $1 $2
        sleep 30
    done
}

LOCAL_TRAIN_SCRIPT="train_model.py"
LOCAL_SCRIPT_DIR=.
downloadDirFromS3 $SCRIPT_DIR $LOCAL_SCRIPT_DIR

LOCAL_TRAINING_DATA="training_data.csv"
downloadFileFromS3 $TRAINING_DATA $LOCAL_TRAINING_DATA

LOCAL_TEST_DATA="test_data.csv"
downloadFileFromS3 $TEST_DATA $LOCAL_TEST_DATA

LOCAL_MODEL_DIR="model/"
mkdir $LOCAL_MODEL_DIR

LOCAL_EXPORT_PATH="export/"
mkdir $LOCAL_EXPORT_PATH

# Download existing model checkpoints if it exists
downloadDirFromS3 $MODEL_DIR $LOCAL_MODEL_DIR

# Run the training
pipenv run python ${LOCAL_TRAIN_SCRIPT} \
    --model_dir=${LOCAL_MODEL_DIR} \
    --export_model=${LOCAL_EXPORT_PATH} \
    --train_data=${LOCAL_TRAINING_DATA} \
    --test_data=${LOCAL_TEST_DATA}

syncToS3 $LOCAL_MODEL_DIR $MODEL_DIR

# Sync the model directory and the export directory 
# containing the saved model
syncToS3 $LOCAL_MODEL_DIR $MODEL_DIR
syncToS3 $LOCAL_EXPORT_PATH $EXPORT_PATH