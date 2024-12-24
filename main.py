import os
import json
import argparse
from merge import AspectMerge, parse_assigned_list
from openai import OpenAI
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig, set_seed
from comp import AspectCompressor
from utils import read_json_files_in_directory, group_files_by_category, get_json_file_combinations
from tqdm import tqdm

# Initialize OpenAI client
def initialize_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

# Load text file
def load_txt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Text compression function
def compress_text(model_family: str, text: str, model, tokenizer) -> Dict:
    compressor = AspectCompressor(model_family=model_family, model=model, tokenizer=tokenizer, max_seq_len=512, device='cuda', max_len=64)
    compressor.load_model()
    compressed_output = compressor.compress_aspect(text, rate=0.85)
    return compressed_output

# Assignment function
def assign_aspects(compressed_prompt: List[str], tokenizer, assigner_model: str, assigner_template: str, client) -> str:
    prompt_template = load_txt(assigner_template)
    formatted_prompt = prompt_template.replace('{corpus}', compressed_prompt)

    response = client.chat.completions.create(
        model=assigner_model,
        messages=[
            {"role": "system", "content": "You are an AI system specialized in analyzing product reviews."},
            {"role": "user", "content": formatted_prompt},
        ],
        max_tokens=16000,
    )
    return response.choices[0].message.content.strip()

# Process coco dataset
def process_coco_data(data_path, model_family, model, tokenizer, client, args):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for index, entry in data.items():
        print(f"Processing index: {index}")

        entity_a_content = entry['entity_a']['content']
        entity_b_content = entry['entity_b']['content']

        # Compress text
        compressed_a = compress_text(model_family, entity_a_content, model, tokenizer)
        compressed_b = compress_text(model_family, entity_b_content, model, tokenizer)

        data[index]['entity_a']['compressed'] = compressed_a
        data[index]['entity_b']['compressed'] = compressed_b

        # Assign aspects
        data[index]['entity_a']['assigned_list'] = assign_aspects(compressed_a['compressed_prompt'], tokenizer, args.assigner_model, args.assigner_prompt, client)
        data[index]['entity_b']['assigned_list'] = assign_aspects(compressed_b['compressed_prompt'], tokenizer, args.assigner_model, args.assigner_prompt, client)

        # Merge aspects
        merger = AspectMerge(client)
        entity_a_parsed = parse_assigned_list(data[index]['entity_a']['assigned_list'])
        entity_b_parsed = parse_assigned_list(data[index]['entity_b']['assigned_list'])
        data[index]['merged_aspects'] = merger.merge_aspects(entity_a_parsed, entity_b_parsed)

    return data

# Process AMASUM dataset
def process_amasum_data(data_path, model_family, model, tokenizer, client, args):
    json_files = read_json_files_in_directory(data_path)
    category_files = group_files_by_category(json_files, data_path)
    json_combinations = get_json_file_combinations(category_files)

    data = {}
    for category, file_pairs in json_combinations.items():
        for json_file1, json_file2 in tqdm(file_pairs):
            with open(json_file1, 'r', encoding='utf-8') as f1, open(json_file2, 'r', encoding='utf-8') as f2:
                review1 = json.load(f1)
                review2 = json.load(f2)

                entity_a_content = review1['text']
                entity_b_content = review2['text']

                # Compress text
                compressed_a = compress_text(model_family, entity_a_content, model, tokenizer)
                compressed_b = compress_text(model_family, entity_b_content, model, tokenizer)

                index_key = f"{os.path.basename(json_file1)}_{os.path.basename(json_file2)}"
                data[index_key] = {
                    'entity_a': {
                        'content': entity_a_content,
                        'compressed': compressed_a,
                        'assigned_list': assign_aspects(compressed_a['compressed_prompt'], tokenizer, args.assigner_model, args.assigner_prompt, client)
                    },
                    'entity_b': {
                        'content': entity_b_content,
                        'compressed': compressed_b,
                        'assigned_list': assign_aspects(compressed_b['compressed_prompt'], tokenizer, args.assigner_model, args.assigner_prompt, client)
                    }
                }

                # Merge aspects
                merger = AspectMerge(client)
                entity_a_parsed = parse_assigned_list(data[index_key]['entity_a']['assigned_list'])
                entity_b_parsed = parse_assigned_list(data[index_key]['entity_b']['assigned_list'])
                data[index_key]['merged_aspects'] = merger.merge_aspects(entity_a_parsed, entity_b_parsed)

    return data

# Main pipeline function
def run_pipeline(args):
    client = initialize_openai_client(args.api_key)
    tokenizer = AutoTokenizer.from_pretrained(args.model_family)
    model = AutoModelForTokenClassification.from_pretrained(args.model, config=AutoConfig.from_pretrained(args.model_family))

    if "coco" in args.data_path:
        data = process_coco_data(args.data_path, args.model_family, model, tokenizer, client, args)
    elif "AMASUM" in args.data_path:
        data = process_amasum_data(args.data_path, args.model_family, model, tokenizer, client, args)

    # Save results
    save_path = os.path.join(args.result_path, "processed_data.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Results saved to: {save_path}")

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_family", type=str, default="FacebookAI/xlm-roberta-large")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--assigner_prompt", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--assigner_model", type=str, default="gpt-4o-mini")

    args = parser.parse_args()
    set_seed(args.seed)
    run_pipeline(args)
