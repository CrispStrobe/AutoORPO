#!/bin/bash

pip install datasets accelerate wandb transformers bitsandbytes sentencepiece

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

git clone https://github.com/xfactlab/orpo.git
cd orpo
pip install -r requirements.txt

LOGFILE="logfile.log"
echo "Starting log..." > $LOGFILE
exec > >(tee -a "$LOGFILE") 2>&1

sed -i 's/num_processes: 2/num_processes: 1/' ./src/accelerate/fsdp.yaml
sed -i 's/--num_proc", default=8/--num_proc", default=1/' ./src/args.py
wandb login $WANDB_TOKEN
wandb init -p $WANDB_PROJECT

echo "starting main script..."
accelerate launch --config_file ./src/accelerate/fsdp.yaml main.py \
    --lr_scheduler_type inverse_sqrt \
    --alpha 0.1 \
    --torch_compile False \
    --warmup_steps 100 \
    --model_name $MODEL_ID \
    --data_name $DATASET \
    --lr $LEARNING_RATE \
    --num_train_epochs $EPOCH \
    --prompt_max_length 512 \
    --response_max_length 2048 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_proc 1 \
    --flash_attention_2 
echo "finished main script..."
cd $OUTPUT
cd */
huggingface-cli login --token $TOKEN
huggingface-cli upload $NEW_MODEL . .

upload_log_to_github_gist
echo "Log uploaded."
