import os
from huggingface_hub import login
from trl import ORPOConfig, ORPOTrainer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import multiprocessing
import wandb

# Define required environment variables
required_env_vars = ["HF_TOKEN", "WANDB_TOKEN", "WANDB_PROJECT", "MODEL_ID", "DATASET", "NEW_MODEL", "LEARNING_RATE", "EPOCH"]
missing_vars = []

# Check if required environment variables are missing or empty
for var in required_env_vars:
    if var not in os.environ or not os.environ[var]:
        missing_vars.append(var)

# If any required variables are missing, print an error message and exit
if missing_vars:
    print("The following required environment variables are not set or empty:", ", ".join(missing_vars))

# Retrieve environment variables
HF_TOKEN = os.environ['HF_TOKEN']
WANDB_TOKEN = os.environ['WANDB_TOKEN']
WANDB_PROJECT = os.environ['WANDB_PROJECT']
MODEL_ID = os.environ['MODEL_ID']
DATASET = os.environ['DATASET']
NEW_MODEL = os.environ['NEW_MODEL']
LEARNING_RATE = os.environ['LEARNING_RATE']
EPOCH = os.environ['EPOCH']

# Login to Hugging Face Hub
login(token=HF_TOKEN)

# Convert EPOCH to an integer
num_train_epochs = int(EPOCH) if EPOCH else 3  # Default to 3 if EPOCH is not set

# Define ORPOConfig
cfg = ORPOConfig(
    output_dir='./out',
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=20,
    bf16=True,
    tf32=True,
    learning_rate=float(LEARNING_RATE) if LEARNING_RATE else 5e-5,
    warmup_ratio=0.1,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    max_prompt_length=512,
    remove_unused_columns=False,
    max_length=1024,
    beta=0.1,
    save_total_limit=3,
    save_strategy="epoch",
    push_to_hub=True,
    report_to=['wandb'],
    hub_model_id=NEW_MODEL,
)

# Load model and tokenizer
model_id = MODEL_ID
tokenizer_id = MODEL_ID
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

# Set the pad token if it's not already set. We probably cannbot disable padding altogether without changing the ORPOTrainer code.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Assuming '</s>' is the eos_token

# Ensure the padding side is consistent with your configuration
#tokenizer.padding_side = 'right' # usually this should be parsed from tokenizer_config.json sufficiently
#tokenizer.padding_side = 'left'  # Make sure this is intended as per your tokenizer configuration

# Load dataset
ds = load_dataset(DATASET)

# Preprocess dataset
def process(row):
    row["prompt"] = tokenizer.apply_chat_template(row["prompt"], tokenize=False)
    row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
    row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
    return row

ds = ds.map(
    process,
    num_proc=multiprocessing.cpu_count(),
    load_from_cache_file=False,
)
train_dataset = ds["train"]
eval_dataset = ds["test"]

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# Initialize Weights & Biases
loggedin = wandb.login(key=WANDB_TOKEN)
run = wandb.init(project=WANDB_PROJECT)

# Initialize ORPOTrainer and start training
orpo_trainer = ORPOTrainer(
    model=model,
    args=cfg,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)
orpo_trainer.train()
orpo_trainer.push_to_hub()
