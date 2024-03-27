#!/bin/bash

pip install git+https://github.com/huggingface/trl.git  # install TRL from the main branch to use the ORPOTrainer
pip install bitsandbytes accelerate
pip install ninja packaging
MAX_JOBS=6 pip install flash-attn --no-build-isolation --upgrade  # flash-attn speeds up the training on compatible GPUs
pip install wandb

apt-get update
apt-get install jq -y

upload_log_to_github_gist() {
    echo "Uploading log file to GitHub Gist..."
    GIST_JSON=$(jq -n --arg content "$(cat $LOGFILE)" \
        '{public: false, files: {"logfile.log": {content: $content}}}')
    if [ $? -ne 0 ]; then
        echo "Failed to create JSON payload for Gist."
        #exit 1
    fi

    GIST_RESPONSE=$(curl -s -H "Authorization: token $GH_TOKEN" \
        -H "Accept: application/vnd.github.v3+json" \
        -d "$GIST_JSON" \
        "https://api.github.com/gists")
    if [ $? -ne 0 ]; then
        echo "Curl request to upload Gist failed."
        #exit 1
    fi

    GIST_URL=$(echo $GIST_RESPONSE | jq -r '.files["logfile.log"].raw_url')
    if [[ "$GIST_URL" != "null" ]]; then
        echo "Log file uploaded to Gist: $GIST_URL"
    else
        echo "Failed to upload log file to Gist."
        #exit 1
    fi
}

LOGFILE="logfile.log"
echo "Starting log..." > $LOGFILE
exec > >(tee -a "$LOGFILE") 2>&1

python orpo_run.py

echo $WANDB_TOKEN
echo $WANDB_PROJECT
echo $MODEL_ID
echo $DATASET
echo $LEARNING_RATE
echo $EPOCH 

echo "finished main script..."

upload_log_to_github_gist
echo "Log uploaded."
