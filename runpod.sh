pip install datasets accelerate wandb transformers bitsandbytes sentencepiece
git clone https://github.com/xfactlab/orpo.git
cd orpo
sed -i 's/num_processes: 2/num_processes: 1/' ./src/accelerate/fsdp.yaml
sed -i 's/--num_proc", default=8/--num_proc", default=1/' ./src/args.py
wandb login $WANDB_TOKEN
wandb init -p $WANDB_PROJECT
echo "starting main script..."
accelerate launch --config_file ./src/accelerate/fsdp.yaml main.py \
    --lr $LEARNING_RATE \
    --warmup_steps 100 \
    --model_name $MODEL_ID \
    --data_name $DATASET \
    --num_train_epochs $EPOCH \
    --prompt_max_length 128 \
    --response_max_length 2048 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_proc 1
echo "finished main script..."
cd $OUTPUT
cd */
huggingface-cli login --token $TOKEN
huggingface-cli upload $NEW_MODEL . .
