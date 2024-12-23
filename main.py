import os
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig, set_seed
from merge import AspectMerge, parse_assigned_list
from openai import OpenAI
from comp import AspectCompressor
from utils import read_json_files_in_directory, group_files_by_category, get_json_file_combinations

# -----------------------------------
# Initialization Functions
# -----------------------------------

def initialize_openai_client(api_key: str) -> OpenAI:
    """Initialize OpenAI API client."""
    return OpenAI(api_key=api_key)

def load_txt(file_path: str) -> str:
    """Load the content of a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# -----------------------------------
# Core Functionalities
# -----------------------------------

def compress_text(model_family: str, text: str, model, tokenizer) -> dict:
    """Compress the given text using AspectCompressor."""
    compressor = AspectCompressor(
        model_family=model_family,
        model=model,
        tokenizer=tokenizer,
        max_seq_len=512,
        device='cuda',
        max_len=64
    )
    compressor.load_model()
    compressed_output = compressor.compress_aspect(text, rate=0.85)
    return {
        "compressed_prompt_list": compressed_output.get("compressed_prompt_list"),
        "compressed_prompt": compressed_output.get("compressed_prompt"),
        "compressed_tokens": compressed_output.get("compressed_tokens"),
        "context_aspects": compressed_output.get("context_aspects"),
        "origin_tokens": compressed_output.get("origin_tokens"),
        "ratio": compressed_output.get("ratio"),
        "rate": compressed_output.get("rate"),
        "saving": compressed_output.get("saving"),
    }

def assign_aspects(compressed_prompt: list, tokenizer, assigner_model: str, assigner_template: str, client) -> str:
    """Assign aspects using OpenAI API."""
    prompt_template = load_txt(assigner_template)
    formatted_prompt = prompt_template.replace('{corpus}', " ".join(compressed_prompt))
    response = client.chat.completions.create(
        model=assigner_model,
        messages=[
            {"role": "system", "content": "You are an AI system specialized in analyzing product reviews."},
            {"role": "user", "content": formatted_prompt},
        ],
        max_tokens=16000,
        seed=1220
    )
    return response.choices[0].message.content.strip()

def merge_aspects(data: dict, merger: AspectMerge, tokenizer, assigner_model: str, assigner_template: str, client):
    """Perform aspect merging for paired review data."""
    for index, entry in data.items():
        entity_a_content = entry['entity_a']['content']
        entity_b_content = entry['entity_b']['content']

        # Assign aspects
        assigned_list_a = assign_aspects(entity_a_content, tokenizer, assigner_model, assigner_template, client)
        assigned_list_b = assign_aspects(entity_b_content, tokenizer, assigner_model, assigner_template, client)

        # Parse assigned lists and merge
        parsed_a = parse_assigned_list(assigned_list_a)
        parsed_b = parse_assigned_list(assigned_list_b)
        merged = merger.merge_aspects(parsed_a, parsed_b)
        data[index]['merged_aspects'] = merged

# -----------------------------------
# Summarization
# -----------------------------------

def generate_summary(merged_aspects: dict, prompt_template: str, client, model_name: str) -> str:
    """Generate a summary based on merged aspects."""
    formatted_prompt = prompt_template.replace('{input}', json.dumps(merged_aspects))
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an AI model tasked with comparing and summarizing reviews."},
            {"role": "user", "content": formatted_prompt},
        ],
        max_tokens=16000,
    )
    return response.choices[0].message.content.strip()

# -----------------------------------
# Main Pipeline
# -----------------------------------

def run_pipeline(args):
    """Run the complete pipeline for text compression, aspect assignment, and summarization."""
    # Initialize components
    client = initialize_openai_client(args.api_key)
    tokenizer = AutoTokenizer.from_pretrained(args.model_family)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, 
        config=AutoConfig.from_pretrained(args.model_family)
    )
    merger = AspectMerge(client)

    # Load data
    if os.path.isfile(args.data_path):
        with open(args.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        raise FileNotFoundError(f"Data path {args.data_path} does not exist.")

    # Text compression and aspect merging
    if args.comp_text:
        print("Performing text compression...")
        for index, entry in data.items():
            entry['entity_a']['compressed'] = compress_text(args.model_family, entry['entity_a']['content'], model, tokenizer)
            entry['entity_b']['compressed'] = compress_text(args.model_family, entry['entity_b']['content'], model, tokenizer)

    if args.merge:
        print("Merging aspects...")
        merge_aspects(data, merger, tokenizer, args.assigner_model, args.assigner_prompt, client)

    # Generate summaries
    with open(args.prompt, 'r', encoding='utf-8') as prompt_file:
        prompt_template = prompt_file.read()

    print("Generating summaries...")
    for index, entry in data.items():
        entry['summary'] = generate_summary(entry['merged_aspects'], prompt_template, client, args.assigner_model)

    # Save results
    os.makedirs(args.result_path, exist_ok=True)
    result_file = os.path.join(args.result_path, "results.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Results saved to {result_file}")

# -----------------------------------
# Entry Point
# -----------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to input data.")
    parser.add_argument("--model_family", type=str, required=True, help="Pre-trained transformer model family.")
    parser.add_argument("--model", type=str, required=True, help="Path to the fine-tuned transformer model.")
    parser.add_argument("--result_path", type=str, required=True, help="Directory to save results.")
    parser.add_argument("--prompt", type=str, required=True, help="Path to the summarization prompt template.")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key.")
    parser.add_argument("--seed", type=int, default=1212, help="Random seed for reproducibility.")
    parser.add_argument("--assigner_model", type=str, default="gpt-4o-mini", help="Model name for aspect assignment.")
    parser.add_argument("--assigner_prompt", type=str, required=True, help="Path to aspect assignment prompt template.")
    parser.add_argument("--comp_text", action="store_true", help="Enable text compression.")
    parser.add_argument("--merge", action="store_true", help="Enable aspect merging.")
    
    args = parser.parse_args()
    set_seed(args.seed)
    run_pipeline(args)
