import os
import json
import time
import datetime
import os
import copy
import random
from collections import Counter
from typing import List, Tuple
import json
from tqdm import tqdm
from comp import AspectCompressor
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
import numpy as np
from openai import OpenAI
import tiktoken
import transformers
import torch


def load_txt(file_path):
    with open(file_path, 'r') as f:
        data = f.read()
    return data


api_key = 'user-api-key'
client = OpenAI(api_key=api_key)

      
def get_token_length(
        text: str,
        tokenizer,
        add_special_tokens: bool = True,
        use_oai_tokenizer: bool = False,
    ):
        if use_oai_tokenizer:
            oai_tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
            return len(oai_tokenizer.encode(text))
        else:
            return len(
                tokenizer(text, add_special_tokens=add_special_tokens).input_ids
            )
    
        
def text_comasp(model_family:str, text : str, model: str, tokenizer: str):
    compasp = AspectCompressor(
        model_family = model_family,
        model=model,
        tokenizer= tokenizer,
        max_seq_len=512, #건들지마셈 
        device='cuda',
        max_len=64   # Whether to use llmlingua-2
        )
    
    compasp.load_model()
    compressed_output = compasp.compress_aspect(text,rate=0.45)
    compressed_list_prompt = compressed_output.get("compressed_prompt_list") 
    compressed_prompt = compressed_output.get("compressed_prompt")  
    compressed_tokens = compressed_output.get("compressed_tokens")  
    aspect = compressed_output.get("context_aspects")  
    origin_tokens = compressed_output.get("origin_tokens")  
    ratio = compressed_output.get("ratio")  
    rate = compressed_output.get("rate") 
    saving = compressed_output.get("saving")  
    
    #print(compressed_output)
    return compressed_list_prompt, compressed_prompt,compressed_tokens, aspect , origin_tokens, ratio, rate, saving

def format_template(template: str, candidate_clusters: List[str], new_sentences: List[str]):
    candidate_cluster_str = ', '.join(candidate_clusters) 
    new_text_str = '\n'.join([f"{i+1}. {sentence}" for i, sentence in enumerate(new_sentences)])  
    formatted_output = template.format(candidate_cluster=candidate_cluster_str, new_text=new_text_str)
    return formatted_output


def assign(
    compressed_prompt: List[str],
    tokenizer,  
    assigner_name : str,
    assigner_template: str,
):
    
    prompt_template = load_txt(assigner_template)
    
    formatted_prompt = prompt_template.replace('{corpus}', compressed_prompt)
    #print(formatted_prompt)
        
    context_window = 128000

    input_token = get_token_length(formatted_prompt, tokenizer, add_special_tokens=True, use_oai_tokenizer=False)
    print(f"Input token length: {input_token}")
    """
    max_output_tokens = 16384 
    if  context_window - input_token > max_output_tokens:
        max_output_tokens =  max_output_tokens
    """
    max_output_tokens = 16000

    #print(f"Max output tokens: {max_output_tokens}")
    
    response = client.chat.completions.create(
        model=assigner_name,
        messages=[
            {"role": "system", "content": "You are an AI system specialized in analyzing product reviews.  following specific guidelines."},
            {"role": "user", "content": formatted_prompt},
        ],
        seed=1220,
        max_tokens=max_output_tokens,
    )
    
    assign = response.choices[0].message.content.strip()  
    cluster_token = get_token_length(assign, tokenizer, add_special_tokens=True, use_oai_tokenizer=False)
    
    print(f"Cluster token length: {cluster_token}")
    
    return assign

def remove_duplicates(phrases):
    
    words = set()  # Use a set to avoid duplicates
    for phrase in phrases:  # Iterate over the list of phrases
        if isinstance(phrase, str):  # Ensure it's a string before splitting
            words.update(phrase.split())  # Split each phrase and add the words to the set
        else:
            print(f"Skipping non-string phrase: {phrase}")
    return list(words)  # Convert back to a list if needed

    
def run(
    model_family: str,
    problem: str,
    tokenizer: str,
    comps_model: str,
    comps_tokenizer: str,
    comps_dir : str,
    random_seed: int = 0,
    # assigner arguments
    assigner_model: str = "gpt-4o-mini",
    assigner_name : str = "gpt-4o-mini",
    assigner_tokenizer: str = "gpt-4o-mini",
    asssinger_prompt: str = "/root/AACS/templet/gpt_assigner.txt",

):
    random.seed(random_seed)
    np.random.seed(random_seed)
    

    
    n = len(problem)

    all_descriptions = []
    all_text_descriptions_matching = np.zeros((n, 0), dtype=np.int64)
    unselected_text_indicators = np.ones(n, dtype=bool)
    num_iterations = 0

    #compressor
    
    _problem = copy.deepcopy(problem)
    #print(_problem.texts)
    print(f"len of texts : {len(_problem)}")

    # select the text instances that are not covered by any of the explanations, but still make sure that the number of text instances is not too small
    _problem = [
        problem[i] for i in range(n) if unselected_text_indicators[i]
    ]       


    compressed_list_prompt, compressed_prompt,compressed_tokens, aspect , origin_tokens, ratio, rate, saving = text_comasp(
            model_family=model_family,
            text=_problem,
            model=comps_model,
            tokenizer=comps_tokenizer)
    
    
    _problem = compressed_prompt
    
    assigner_template_ = asssinger_prompt # Replace "your_assigner_template_value" with the actual value
    
    assigned_list = assign(
            _problem,
            tokenizer,
            assigner_name,
            assigner_model,
            assigner_tokenizer,
            assigner_template_)
    #print('assigned_list:', assigned_list)
    
    compressed_data = {
        "compressed_text": compressed_list_prompt,
        "original_tokens": origin_tokens,
        "compressed_tokens":compressed_tokens,
        "ratio": ratio,
        "rate": rate,
        "saving": saving,
        "assigned_list": assigned_list, 
    }

    return compressed_data

def read_json_files_in_directory(directory):

    json_files = []

    def explore_directory(current_path):
        with os.scandir(current_path) as it:
            for entry in it:
                if entry.is_dir():
                    explore_directory(entry.path) 
                elif entry.is_file() and entry.name.endswith('.json'):
                    json_files.append(entry.path)

    explore_directory(directory)
    return json_files


if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/root/AACS/coco/test_raw_reviews_content.json",
    )
    parser.add_argument("--model_family", type=str, default="roberta")
    parser.add_argument("--exp_dir", type=str, default="/root/AACS/experiments/gpt4")
    parser.add_argument("--subsample", type=int, default=0)
    parser.add_argument("--chunk_text_to_words", type=int, default=None)
    parser.add_argument("--turn_off_approval_before_running", action="store_true")
    parser.add_argument("--with_labels", action="store_true")

    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--min_cluster_fraction", type=float, default=0.0)
    parser.add_argument("--max_cluster_fraction", type=float, default=0.4)
    parser.add_argument("--model", type=str, default="/aspect_sum/scr/xlm_roberta_models")
    parser.add_argument("--comp_dir", type=str, default="/home/work/chaewo/aspect_sum/templet")
    parser.add_argument("--assigner_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--assigner_prompt", type=str, default="/root/AACS/templet/gpt_assigner.txt")
    args = parser.parse_args()
    
    args.exp_dir += f"/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    os.makedirs(args.exp_dir, exist_ok=True)

    with open(os.path.join(args.exp_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
 
        
    config = AutoConfig.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model, config=config)
    tokenizer = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-large') # FacebookAI/xlm-roberta-large
    print('mdoel', model)
    all_compasp_outputs = []

    with open(args.data_path, 'r', encoding='utf-8') as f:
        print(f"Loading data from {args.data_path}")
        data = json.load(f)
        
        entry_count = len(data.keys())  
        print(f"Entry count: {entry_count}")
        
        for index in range(len(data)):
            entity_a_content = data[str(index)]['entity_a']['content']
            entity_b_content = data[str(index)]['entity_b']['content']
            
            category_path = os.path.relpath(os.path.dirname(args.data_path), start=args.data_path)

            # 저장할 파일 경로 설정
            exp_subdir = os.path.join(args.exp_dir, category_path)
            save_path = os.path.join(exp_subdir, os.path.basename(args.data_path))
            print(f"Save path: {save_path}")

            # 디렉토리 생성
            if not os.path.exists(exp_subdir):
                os.makedirs(exp_subdir)

            all_compressed_outputs_A = []
                #print(reviw.texts)  
            compressed_output_A = run(
                model_family=args.model_family,
                problem=entity_a_content,
                tokenizer=tokenizer,
                random_seed=args.random_seed,
                comps_model=model,
                comps_tokenizer=tokenizer,
                comps_dir=args.comp_dir,
                assigner_model=args.assigner_model,
                assigner_name=args.assigner_model,
                asssinger_prompt=args.assigner_prompt,
            ) 
            all_compressed_outputs_A = {
                "compressed_text": compressed_output_A["compressed_text"],
                "original_tokens": compressed_output_A["original_tokens"],
                "compressed_tokens": compressed_output_A["compressed_tokens"],
                "ratio": compressed_output_A["ratio"],
                "rate": compressed_output_A["rate"],
                "saving": compressed_output_A["saving"],
                "assigned_list": compressed_output_A["assigned_list"]
            }
            
            all_compressed_outputs_B = []
            compressed_output_B = run(
                model_family=args.model_family,
                problem=entity_b_content,
                tokenizer=tokenizer,
                random_seed=args.random_seed,
                comps_model=model,
                comps_tokenizer=tokenizer,
                comps_dir=args.comp_dir,
                assigner_name=args.assigner_model,
                asssinger_prompt=args.assigner_prompt,
            )  


            all_compressed_outputs_B = {
                "compressed_text": compressed_output_B["compressed_text"],
                "original_tokens": compressed_output_B["original_tokens"],
                "compressed_tokens": compressed_output_B["compressed_tokens"],
                "ratio": compressed_output_B["ratio"],
                "rate": compressed_output_B["rate"],
                "saving": compressed_output_B["saving"],
                "assigned_list": compressed_output_B["assigned_list"]
            }
                            

            dict = {
                str(index): {
                    "entity_a": all_compressed_outputs_A,
                    "entity_b": all_compressed_outputs_B
                }
            }


            if os.path.exists(save_path):
                with open(save_path, 'r') as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = []
            else:
                existing_data = []


            existing_data.append(dict)


            with open(save_path, 'w') as json_file:
                json.dump(existing_data, json_file, indent=4)
            all_compressed_outputs_A.clear()
            all_compressed_outputs_B.clear()

    
    