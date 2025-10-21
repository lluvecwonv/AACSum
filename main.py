import os
import json
import argparse
from merge import AspectMerge, parse_assigned_list
from openai import OpenAI
from typing import List, Dict, Tuple



# Initialize OpenAI client
def initialize_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

# Load text file
def load_txt(file_path: str) -> str:
    with open(file_path, 'r') as f:
        return f.read()


# Assignment function
def assign(compressed_prompt: List[str], assigner_name: str, assigner_template: str, client) -> str:
    # Load the template for assignment
    prompt_template = load_txt(assigner_template)
    formatted_prompt = prompt_template.replace('{corpus}', compressed_prompt)

    max_output_tokens = 16000
    response = client.chat.completions.create(
        model=assigner_name,
        messages=[
            {"role": "system", "content": "You are an AI system specialized in analyzing product reviews."},
            {"role": "user", "content": formatted_prompt},
        ],
        seed=1220,
        max_tokens=max_output_tokens,
    )

    response_text = response.choices[0].message.content.strip()
    print(f"Assigned list: {response_text}")
    return response_text

# Integrated execution function
def run_pipeline(args):
    # Initialize OpenAI client
    client = initialize_openai_client(args.api_key)

    # Load the input data
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"📊 Processing {len(data)} data pairs...")
    print("Merging aspects...")
    merger = AspectMerge(client, embedding_model=args.embedding_model)
    for index, entry in data.items():
        print(f"Processing data index: {index}")

        # Directly pass content for assignment
        entity_a_content = ' '.join(entry['Reviews_info']['entity_a_review'])
        entity_b_content = ' '.join(entry['Reviews_info']['entity_b_review'])

        # Perform assignment
        assigned_list_a = assign(
            entity_a_content,
            args.assigner_model,
            args.assigner_prompt,
            client
        )
        assigned_list_b = assign(
            entity_b_content,
            args.assigner_model,
            args.assigner_prompt,
            client
        )

        # Initialize entity_a and entity_b if they don't exist
        if 'entity_a' not in data[str(index)]:
            data[str(index)]['entity_a'] = {}
        if 'entity_b' not in data[str(index)]:
            data[str(index)]['entity_b'] = {}

        # Save assigned results
        data[str(index)]['entity_a']['assigned_list'] = assigned_list_a
        data[str(index)]['entity_b']['assigned_list'] = assigned_list_b

        # Parse the assigned lists
        entity_a_parsed = parse_assigned_list(assigned_list_a)
        entity_b_parsed = parse_assigned_list(assigned_list_b)
        
        merged_aspect_map = merger.merge_aspects(entity_a_parsed, entity_b_parsed)
        data[str(index)]['merged_aspects'] = merged_aspect_map
        
        print(f"Merged aspects for index {index}: {data[str(index)]['merged_aspects']}")

    # Summarization
    with open(args.prompt, 'r') as prompt_file:
        prompt_template = prompt_file.read()
        
        file_name = os.path.basename(args.data_path).split('.')[0]
        save_path = os.path.join(args.result_path, file_name, f'{file_name}.json')
        save_dir = os.path.dirname(save_path)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    for index, entry in data.items():
        save_path2 = os.path.join(args.result_path, file_name, f'aspect_review{[index]}_new.json')
        print('save_path2:', save_path2)
        formatted_prompt = prompt_template.replace('{input}', json.dumps(entry['merged_aspects']))
        max_output_tokens = 16000
        response = client.chat.completions.create(
            model=args.summarizer_model,
            messages=[
                {"role": "system", "content": "You are an AI model tasked with comparing and summarizing reviews based on extracted aspects from two product reviews."},
                {"role": "user", "content": formatted_prompt},
            ],
            seed=args.seed,
            max_tokens=max_output_tokens,
        )

        summary = response.choices[0].message.content.strip()
        print(f"Summary for index {index}: {summary}")
        entry['summary'] = summary
        
        # Remove unnecessary ```json text from summary
        summary = summary.replace('```json', '')
        summary = summary.replace('```', '').strip()

        try:
            # Attempt to convert summary to JSON format
            json_data = json.loads(summary)
        except json.JSONDecodeError as e:
            # If JSON conversion fails, output error message and save as text
            print(f"Error during model processing: {e}")
            json_data = {"summary_text": summary}

        # Save results
        with open(save_path2, 'w') as f:
            json.dump(json_data, f, indent=4)
        print(f"Results saved to: {save_path2}")

    print(f"\n✅ All processing complete! Results saved to: {args.result_path}")

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to input JSON file with review pairs")
    parser.add_argument("--result_path", type=str, required=True, help="Directory to save output summaries")
    parser.add_argument("--prompt", type=str, required=True, help="Path to summarization prompt template")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--assigner_model", type=str, default="gpt-4o-mini", help="GPT model for aspect assignment (default: gpt-4o-mini)")
    parser.add_argument("--summarizer_model", type=str, default="gpt-4o-mini", help="GPT model for summarization (default: gpt-4o-mini)")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small", help="OpenAI embedding model (default: text-embedding-3-small)")
    parser.add_argument("--assigner_prompt", type=str, required=True, help="Path to aspect assignment prompt template")
    parser.add_argument("--seed", type=int, default=1220, help="Random seed for reproducibility")

    args = parser.parse_args()
    run_pipeline(args)
