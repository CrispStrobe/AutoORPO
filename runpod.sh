#!/bin/bash

export REQ_TXT_PATH="$(pwd)/requirements.txt"
cat << EOF > requirements.txt
accelerate
aiohttp
aiosignal
appdirs==1.4.4
asttokens
attrs
bitsandbytes==0.43.0
Bottleneck
Brotli
cachetools==5.3.3
certifi
cffi
charset-normalizer
click
comm
datasets
debugpy
decorator
dill
docker-pycreds
einops==0.7.0
exceptiongroup
executing
filelock
flash-attn==2.5.6
frozenlist
fsspec==2023.4.0
gitdb
GitPython
gmpy2
huggingface-hub
idna
importlib_metadata
ipykernel
ipython
jedi
Jinja2==3.1.2
jupyter_client
jupyter_core
MarkupSafe
matplotlib-inline
mkl-fft
mkl-random
mkl-service==2.4.0
mpmath
multidict
multiprocess
nest_asyncio
networkx==3.2.1
ninja==1.11.1.1
numexpr
numpy
nvidia-cublas-cu11==11.11.3.6
nvidia-cuda-cupti-cu11==11.8.87
nvidia-cuda-nvrtc-cu11==11.8.89
nvidia-cuda-runtime-cu11==11.8.89
nvidia-cudnn-cu11==8.7.0.84
nvidia-cufft-cu11==10.9.0.58
nvidia-curand-cu11==10.3.0.86
nvidia-cusolver-cu11==11.4.1.48
nvidia-cusparse-cu11==11.7.5.86
nvidia-ml-py==12.535.133
nvidia-nccl-cu11==2.19.3
nvidia-nvtx-cu11==11.8.86
nvitop==1.3.2
packaging
pandas
parso
pathtools
pexpect
pickleshare
pillow==10.2.0
platformdirs
prompt-toolkit
protobuf==3.20.3
psutil
ptyprocess
pure-eval
pyarrow
pyarrow-hotfix
pycparser
Pygments
PySocks
python-dateutil
pytz
PyYAML
pyzmq
regex
requests
safetensors
sentry-sdk
setproctitle
six
smmap
stack-data
sympy
termcolor==2.4.0
tokenizers
torch==2.2.1+cu118
torchaudio==2.2.1+cu118
torchvision==0.17.1+cu118
tornado
tqdm
traitlets
transformers
triton==2.2.0
typing_extensions==4.8.0
tzdata
urllib3
wandb
wcwidth
xxhash
yarl
zipp
EOF

pip install datasets accelerate wandb transformers bitsandbytes sentencepiece
pip install flash-attn
pip install -r $REQ_TXT_PATH
apt-get update
apt-get install jq

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
