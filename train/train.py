import os
import json
import argparse
import random
from datasets import load_dataset
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoConfig,
    Trainer,
    set_seed,
    Seq2SeqTrainingArguments,
)
from utils import TokenDataset, train_val_split
from model import T5_CustomTrainer, T5ForMultiTaskLearning, custom_data_collator
import hydra
import deepspeed

# Constants
MAX_LEN = 512
MAX_GRAD_NORM = 10

@hydra.main(version_base=None, config_path="config", config_name="finetune_roberta")
def main(args):
    # Set device and seed
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    os.environ["WANDB_DISABLED"] = "true"
    
    # Create directories for saving model and logs
    os.makedirs(os.path.dirname(args.save_dir), exist_ok=True)
    print(f"Using device: {device}")

    # Load train and validation data
    with open(args.train_data_path, 'r') as f:
        train_data = json.load(f)
    with open(args.val_data_path, 'r') as f:
        val_data = json.load(f)
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_text, train_label, val_text, val_label = train_val_split(train_data, val_data)
    
    # Prepare datasets
    train_dataset = TokenDataset(train_text, train_label, MAX_LEN, tokenizer=tokenizer, model_name=args.model_name)
    val_dataset = TokenDataset(val_text, val_label, MAX_LEN, tokenizer=tokenizer, model_name=args.model_name)
    print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")

    # Training setup
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    max_steps = int(args.num_epochs * len(train_dataset)) // (batch_size * gradient_accumulation_steps * num_devices)
    print(f"Max steps: {max_steps}")
    
    # Model and Training Arguments setup
    if args.model_name.startswith('FacebookAI'):
        # Configure and initialize Roberta model
        config = AutoConfig.from_pretrained(args.model_name)
        model = AutoModelForTokenClassification.from_pretrained(args.model_name, config=config)
        training_args = transformers.TrainingArguments(
            output_dir=args.save_dir,
            evaluation_strategy=args.evaluation_strategy,
            logging_strategy="epoch",
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.num_epochs,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            save_steps=max_steps,
            save_strategy=args.save_strategy,
            eval_accumulation_steps=args.eval_accumulation_steps,
            optim='paged_adamw_32bit',
            seed=args.seed,
        )

        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=training_args,
            data_collator=custom_data_collator,
        )
        
        # Training and Saving
        trainer.train()
        trainer.save_model(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
    
    elif args.model_family == 't5':
        # Configure and initialize T5 model
        model = T5ForMultiTaskLearning(args.check_model_name)
        training_args = Seq2SeqTrainingArguments(
            output_dir=args.save_dir,
            evaluation_strategy=args.evaluation_strategy,
            logging_strategy="epoch",
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.num_epochs,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            save_steps=max_steps,
            save_strategy="no",
            eval_accumulation_steps=args.eval_accumulation_steps,
            optim='paged_adamw_32bit',
            seed=args.seed,
        )

        trainer = T5_CustomTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=training_args,
            data_collator=custom_data_collator,
        )
        
        # Training and Saving
        model.config.use_cache = False
        trainer.train()
        model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)

if __name__ == "__main__":
    main()
