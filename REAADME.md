# Fine-Tuning a RoBERTa Model for Compression Data Learning 🧠
---
This script fine-tunes a RoBERTa model specifically for compression data learning tasks. It utilizes the transformers library to train a model that effectively processes and compresses token-level information, making it ideal for applications requiring structured compression and token-level accuracy.

## Key Features 🚀
Compression Data Learning: Tailored for learning compression tasks that require understanding and retaining essential token-level information.
Customizable Training: Easily adaptable with hydra configuration for various datasets and training setups.
Efficient Training:
Supports gradient accumulation for handling large batches.
Utilizes the lightweight paged_adamw_32bit optimizer.
Reproducibility: Ensures deterministic results by fixing random seeds.
Dataset Handling: Includes utilities for splitting datasets and preparing tokenized input.

```bash
master_port=18767
lr=1e-5
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$master_port train.py --config-name=finetune.yaml  batch_size=4 gradient_accumulation_steps=1 lr=${lr}
```

# Text Compression and Aspect Merging Pipeline
---
This repository provides a comprehensive pipeline for performing text compression, aspect merging, and summarization using state-of-the-art language models and APIs. The system is designed to handle paired review data and generate concise summaries by compressing text, assigning aspects, merging similar aspects, and summarizing the results.

## Features
Text Compression: Compress long texts into concise representations using fine-tuned transformer models.
Aspect Assignment: Assign aspects to compressed texts for better organization and understanding.
Aspect Merging: Cluster and merge similar aspects from paired review data.
Summarization: Generate summaries based on merged aspects.

# General Command Structure

```bash
python main.py \
    --data_path "<path_to_input_data>" \
    --model_family "<pretrained_model_family>" \
    --model "<path_to_finetuned_model>" \
    --result_path "<path_to_save_results>" \
    --prompt "<path_to_prompt_template>" \
    --api_key "<your_openai_api_key>" \
    --seed 1212 \
    --assigner_model "gpt-4o-mini" \
    --assigner_prompt "<path_to_assigner_prompt_template>" \
    [--comp_text] \
    [--merge]
```

Examples

## 1. Aspect Merging Only (--merge)
If you only want to perform aspect merging on pre-compressed text data:

```bash
python main.py \
    --data_path "/path/to/data" \
    --model_family "FacebookAI/xlm-roberta-large" \
    --model "/path/to/finetuned_model" \
    --result_path "/path/to/save/merged_results" \
    --prompt "/path/to/summarization_template.txt" \
    --api_key "<your_openai_api_key>" \
    --seed 1212 \
    --assigner_model "gpt-4o-mini" \
    --assigner_prompt "/path/to/assigner_prompt_template.txt" \
    --merge
```

## 2. Text Compression Only (--comp_text)
If you want to compress the text data without merging aspects:

``` bash
python main.py \
    --data_path "/path/to/data" \
    --model_family "FacebookAI/xlm-roberta-large" \
    --model "/path/to/finetuned_model" \
    --result_path "/path/to/save/compressed_results" \
    --prompt "/path/to/summarization_template.txt" \
    --api_key "<your_openai_api_key>" \
    --seed 1212 \
    --assigner_model "gpt-4o-mini" \
    --assigner_prompt "/path/to/assigner_prompt_template.txt" \
    --comp_text
```

## 3. Full Pipeline (--merge --comp_text)
To perform both text compression and aspect merging:

```bash
python main.py \
    --data_path "/path/to/data" \
    --model_family "FacebookAI/xlm-roberta-large" \
    --model "/path/to/finetuned_model" \
    --result_path "/path/to/save/merged_compressed_results" \
    --prompt "/path/to/summarization_template.txt" \
    --api_key "<your_openai_api_key>" \
    --seed 1212 \
    --assigner_model "gpt-4o-mini" \
    --assigner_prompt "/path/to/assigner_prompt_template.txt" \
    --merge \
    --comp_text
```


# Evaluation Script for COCOTRIP and AMASUM Datasets 📝
---
This repository contains an evaluation script designed to assess the performance of generated summaries on two datasets: COCOTRIP and AMASUM. The evaluation involves comparing generated results against a predefined benchmark using metrics like Rouge, BERTScore, and CASPR.

## Key Features 🚀
Dataset Support: COCOTRIP and AMASUM datasets.
Metrics:
Rouge: Measures overlap between generated and benchmark summaries.
BERTScore: Evaluates semantic similarity.
CASPR: Assesses comparative aspect similarity and polarity recognition.
Handles Multiple File Structures: Automatically detects and processes files for evaluation.

## Example Commands

### Evaluate on COCOTRIP

```bash
CUDA_VISIBLE_DEVICES=1 python main.py \
  --dataset cocotrip \
  --bench_folder_path "/path/to/benchmark/cocotrip" \
  --result_folder_path "/path/to/results/cocotrip" \
  --save_path "/path/to/save/cocotrip_results"
```

### Evaluate on AMASUM

```bash
CUDA_VISIBLE_DEVICES=1 python main.py \
  --dataset amasum \
  --bench_folder_path "/path/to/benchmark/amasum" \
  --result_folder_path "/path/to/results/amasum" \
  --save_path "/path/to/save/amasum_results"
```