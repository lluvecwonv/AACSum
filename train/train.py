import os
import json
import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)
from utils import TokenDataset, train_val_split, custom_data_collator
import hydra


MAX_LEN = 512


@hydra.main(version_base=None, config_path="config", config_name="finetune_roberta")
def main(args):
    """
    Main function for fine-tuning a RoBERTa model for token classification.
    """
    # Set device and seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    os.environ["WANDB_DISABLED"] = "true"
    os.makedirs(os.path.dirname(args.save_dir), exist_ok=True)

    print(f"Using device: {device}")

    # Load train and validation data
    with open(args.train_data_path, 'r') as f:
        train_data = json.load(f)
    with open(args.val_data_path, 'r') as f:
        val_data = json.load(f)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_text, train_label, val_text, val_label = train_val_split(train_data, val_data)

    # Prepare datasets
    train_dataset = TokenDataset(train_text, train_label, MAX_LEN, tokenizer=tokenizer, model_name=args.model_name)
    val_dataset = TokenDataset(val_text, val_label, MAX_LEN, tokenizer=tokenizer, model_name=args.model_name)
    print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")

    # Calculate maximum steps
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    max_steps = int(args.num_epochs * len(train_dataset)) // (batch_size * gradient_accumulation_steps * num_devices)
    print(f"Max steps: {max_steps}")

    # Configure model and training arguments
    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, config=config)

    training_args = TrainingArguments(
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

    # Initialize Trainer
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
    print(f"Model and tokenizer saved to {args.save_dir}")

# -----------------------------------
# Entry Point
# -----------------------------------
if __name__ == "__main__":
    main()
