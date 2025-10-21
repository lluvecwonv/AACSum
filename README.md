# Abstractive Aspect-Based Comparative Summarization
This repository contains the official code and datasets for the paper:
**"Abstractive Aspect-Based Comparative Summarization"**  
> Hyeon Jin, Chaewon Yoon, Yurim Oh, Hyun-Je Song  
> *Presented at the ACM Web Conference 2025 (WWW Companion '25), Sydney, Australia* 
## Overview
Comparative summarization aims to generate summaries highlighting the key similarities and differences between two entities. However, existing methods often fail to provide aspect-specific insights that are crucial for user decision-making.  
This work introduces **Abstractive Aspect-Based Comparative Summarization**, which:
- Identifies aspects of entities from a set of two reviews.
- Generates abstractive contrastive and common summaries for each aspect.
- Leverages Large Language Models (LLMs) for generating high-quality summaries.
<table align="center">
  <tr>
    <td align="center"> 
      <img src="assets/figure1.jpeg" alt="Teaser Figure" style="width: 700px;"/> 
      <br>
      <em style="font-size: 18px;">  <strong style="font-size: 18px;">Figure 1: Overview of the proposed method 
    </td>
  </tr>
</table>
</div>

### Features:
- **Aspect-Based Summarization**: Summaries are structured around specific aspects such as "Staff", "Parking", "Price", etc.
- **Contrastive & Common Summaries**: Captures both **differences** and **similarities** between entities.
- **LLM-Powered**: Uses goal-driven clustering and hierarchical aspect merging to generate high-quality abstractive summaries.
- **New Benchmark Datasets**: Two datasets, **CoCoCom** (hotels) and **AmaCom** (Amazon products), designed for comparative summarization.
---
## Datasets
| Dataset  | # Pairs | Avg. # Reviews | Avg. # Aspects | Avg. Summary Length |
|----------|--------|---------------|---------------|---------------------|
| **CoCoCom** | 48     | 7.8           | 7.75          | 325.72              |
| **AmaCom**  | 646    | 77.78         | 11.56         | 368.45              |
- **CoCoCom**: Human-annotated comparative summaries from **TripAdvisor** hotel reviews.
- **AmaCom**: Summaries derived from **Amazon product categories** (Electronics, Home & Kitchen, Tools & Home Improvement).
---
## Installation
### Prerequisites:
- Python 3.8+
- OpenAI API key
- Required Python packages:
  ```bash
  pip install openai scikit-learn numpy
  ```

## Usage
### 1. Aspect-Based Summarization
Run the main script to generate aspect-based summaries:

**Basic usage:**
```bash
python main.py \
  --data_path data/amasum/amasum_bench_dataset.json \
  --result_path results/ \
  --api_key YOUR_OPENAI_API_KEY \
  --prompt template/sum.txt \
  --assigner_prompt template/gpt_assigner.txt
```

**Advanced usage (with custom models):**
```bash
python main.py \
  --data_path data/amasum/amasum_bench_dataset.json \
  --result_path results/ \
  --api_key YOUR_OPENAI_API_KEY \
  --prompt template/sum.txt \
  --assigner_prompt template/gpt_assigner.txt \
  --assigner_model gpt-4o \
  --summarizer_model gpt-4o \
  --embedding_model text-embedding-3-large \
  --seed 1220
```

**Arguments:**

*Required:*
- `--data_path`: Path to input JSON file with review pairs
- `--result_path`: Directory to save output summaries
- `--api_key`: Your OpenAI API key
- `--prompt`: Path to summarization prompt template
- `--assigner_prompt`: Path to aspect assignment prompt template

*Optional:*
- `--assigner_model`: GPT model for aspect assignment (default: gpt-4o-mini)
- `--summarizer_model`: GPT model for summarization (default: gpt-4o-mini)
- `--embedding_model`: OpenAI embedding model (default: text-embedding-3-small)
- `--seed`: Random seed for reproducibility (default: 1220)

**Supported Models:**
- GPT models: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`
- Embedding models: `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`

**Cost Tracking:**
The script automatically tracks API usage and saves a cost summary to `{result_path}/{file_name}/api_cost_summary.json`

### 2. Evaluation
Evaluate the summarization performance:
```bash
python eval.py \
  --dataset cocotrip \
  --predictions results/summaries.json \
  --references data/gold_summaries.json \
  --save_folder_path results/evaluation/
```

**Arguments:**
- `--dataset`: Dataset name (choices: cocotrip, amasum)
- `--predictions`: Path to generated summaries JSON
- `--references`: Path to gold standard summaries JSON
- `--save_folder_path`: Directory to save evaluation results
---
## Citation
If you use this work, please cite:
```bibtex
@inproceedings{Jin2025AACSum,
  author    = {Hyeon Jin and Chaewon Yoon and Yurim Oh and Hyun-Je Song},
  title     = {Abstractive Aspect-Based Comparative Summarization},
  booktitle = {Companion Proceedings of the ACM Web Conference 2025},
  year      = {2025},
}
```
