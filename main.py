import os
import json
import argparse
from merge import AspectMerge,parse_assigned_list
from openai import OpenAI
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig, set_seed
from comp import AspectCompressor


# Initialize OpenAI client
def initialize_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

# Load text file
def load_txt(file_path: str) -> str:
    with open(file_path, 'r') as f:
        return f.read()

# Text compression function
def comp_text(model_family: str, text: str, model, tokenizer) -> Dict:
    compasp = AspectCompressor(model_family=model_family, model=model, tokenizer=tokenizer, max_seq_len=512, device='cuda', max_len=64)
    compasp.load_model()
    compressed_output = compasp.compress_aspect(text, rate=0.45)
    
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

# Assignment function
def assign(compressed_prompt: List[str], tokenizer, assigner_name: str, assigner_template: str, client) -> str:
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
    response = response.choices[0].message.content.strip()
    print(f"Assigned list: {response}")
    return response

# Integrated execution function
def run_pipeline(args):
    client = initialize_openai_client(args.api_key)
    tokenizer = AutoTokenizer.from_pretrained(args.model_family)
    model = AutoModelForTokenClassification.from_pretrained(args.model, config=AutoConfig.from_pretrained(args.model_family))

    # Load data
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # print(data)
    
    # Check if comp_text execution is enabled
    if args.comp_text and args.merge == False:
        for index, entry in data.items():
            entity_a_content = entry['entity_a']['content']
            entity_b_content = entry['entity_b']['content']

            # Perform text compression
            compressed_a = comp_text(args.model_family, entity_a_content, model, tokenizer)
            compressed_b = comp_text(args.model_family, entity_b_content, model, tokenizer)

            data[str(index)]['entity_a']['compressed'] = compressed_a
            data[str(index)]['entity_b']['compressed'] = compressed_b

            # Perform assignment
            data[str(index)]['entity_a']['assigned_list'] = assign(compressed_a['compressed_prompt'], tokenizer, args.assigner_model, args.assigner_prompt, client)
            data[str(index)]['entity_b']['assigned_list'] = assign(compressed_b['compressed_prompt'], tokenizer, args.assigner_model, args.assigner_prompt, client)
            
             # Parse the assigned lists for entity_a and entity_b
            entity_a_parsed = parse_assigned_list(entry['entity_a']['assigned_list'])
            entity_b_parsed = parse_assigned_list(entry['entity_b']['assigned_list'])
        
            all_aspects = set(entity_a_parsed.keys()).union(entity_b_parsed.keys())
            data[str(index)]['merged_aspects'] = {
                aspect: (
                    entity_a_parsed.get(aspect, []), #해당키가 없으면 빈 리스트로 반환 
                    entity_b_parsed.get(aspect, [])
                )
                for aspect in all_aspects
            }
           
            print(f"Merged aspects for index {index}: {data[str(index)]['merged_aspects']}")


    # Check if aspect merging is enabled
  # Check if aspect merging is enabled
    if args.merge and args.comp_text:
        print("Merging aspects...")
        merger = AspectMerge(client)
        for index, entry in data.items():
            entity_a_content = entry['entity_a']['content']
            entity_b_content = entry['entity_b']['content']

            # Perform text compression
            compressed_a = comp_text(args.model_family, entity_a_content, model, tokenizer)
            compressed_b = comp_text(args.model_family, entity_b_content, model, tokenizer)

            data[str(index)]['entity_a']['compressed'] = compressed_a
            data[str(index)]['entity_b']['compressed'] = compressed_b

            # Perform assignment
            data[str(index)]['entity_a']['assigned_list'] = assign(compressed_a['compressed_prompt'], tokenizer, args.assigner_model, args.assigner_prompt, client)
            data[str(index)]['entity_b']['assigned_list'] = assign(compressed_b['compressed_prompt'], tokenizer, args.assigner_model, args.assigner_prompt, client)
            
            # Parse the assigned lists for entity_a and entity_b
            entity_a_parsed = parse_assigned_list(entry['entity_a']['assigned_list'])
            entity_b_parsed = parse_assigned_list(entry['entity_b']['assigned_list'])

            # Merge the parsed aspects
            merged_aspect_map = merger.merge_aspects(entity_a_parsed, entity_b_parsed)

            # Store the merged aspects back in the data structure
            data[str(index)]['merged_aspects'] = merged_aspect_map
            
    elif args.merge and args.comp_text == False:
        print("Merging aspects...")
        merger = AspectMerge(client)
        for index, entry in data.items():
            print(f"Processing data index: {index}")

            # `entity_a`와 `entity_b`의 content를 바로 할당 함수에 전달
            entity_a_content = ' '.join(entry['entity_a']['content'])
            entity_b_content = ' '.join(entry['entity_b']['content'])

            # 할당 작업 수행
            assigned_list_a = assign(entity_a_content, tokenizer, args.assigner_model, args.assigner_prompt, client)
            assigned_list_b = assign(entity_b_content, tokenizer, args.assigner_model, args.assigner_prompt, client)

            # 할당 결과 저장
            data[str(index)]['entity_a']['assigned_list'] = assigned_list_a
            data[str(index)]['entity_b']['assigned_list'] = assigned_list_b
            
            entity_a_parsed = parse_assigned_list(entry['entity_a']['assigned_list'])
            entity_b_parsed = parse_assigned_list(entry['entity_b']['assigned_list'])
            
            merged_aspect_map = merger.merge_aspects(entity_a_parsed, entity_b_parsed)
            data[str(index)]['merged_aspects'] = merged_aspect_map

            # print(f"Entity A parsed: {entity_a_parsed}")
            # print(f"Entity B parsed: {entity_b_parsed}")
                
           # Directly create merged_aspects in the desired tuple format
            # all_aspects = set(entity_a_parsed.keys()).union(entity_b_parsed.keys())
            # data[str(index)]['merged_aspects'] = {
            #     aspect: (
            #         entity_a_parsed.get(aspect, []), 
            #         entity_b_parsed.get(aspect, [])
            #     )
            #     for aspect in all_aspects
            # }
           
            print(f"Merged aspects for index {index}: {data[str(index)]['merged_aspects']}")



    # Summarization
    with open(args.prompt, 'r') as prompt_file:
        prompt_template = prompt_file.read()
        
        file_name = os.path.basename(args.data_path).split('.')[0]
        save_path = os.path.join(args.result_path,file_name,f'{file_name}.json')
        save_dir = os.path.dirname(save_path)
        
        if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        # print(f"save_path: {save_path}")

    for index, entry in data.items():
        save_path2 = os.path.join(args.result_path,file_name,f'aspect_review{[index]}_new.json')
        print('save_path2:',save_path2)
        formatted_prompt = prompt_template.replace('{input}', json.dumps(entry['merged_aspects']))
        max_output_tokens = 16000
        response = client.chat.completions.create(
            model='gpt-4o-mini',
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
        
        # summary에서 불필요한 ```json 등의 텍스트 제거
        summary = summary.replace('```json', '')
        summary = summary.replace('```', '').strip()

        try:
            # JSON 형식으로 summary를 변환 시도
            json_data = json.loads(summary)
        except json.JSONDecodeError as e:
            # JSON 변환에 실패한 경우, 오류 메시지 출력하고 텍스트 그대로 저장
            print(f"Error during model processing: {e}")
            json_data = {"summary_text": summary}

        # Save results
        # os.makedirs(args.result_path, exist_ok=True)
        with open(save_path2, 'w') as f:
            json.dump(json_data, f, indent=4)
        print(f"Results saved to: {save_path2}")

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/root/AACS/coco/test_raw_reviews_content.json')
    parser.add_argument("--model_family", type=str, default="FacebookAI/xlm-roberta-large")
    parser.add_argument("--model", type=str, default="/root/AACS/models/xlm_roberta_large_reviews/checkpoint-700")
    parser.add_argument("--result_path", type=str, default='/root/AACS/sumarization/coco_comp_sum')
    parser.add_argument("--prompt", type=str, default='/root/AACS/templet/sum.txt')
    parser.add_argument("--api_key", type=str,required=True)
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--assigner_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--assigner_prompt", type=str, default="/root/AACS/templet/gpt_assigner.txt")
    parser.add_argument("--comp_text", action="store_true", default=False, help="Enable text compression (default: True)")
    parser.add_argument("--merge", action="store_true", default=False, help="Enable aspect merging (default: False)")

    args = parser.parse_args()
    set_seed(args.seed)
    run_pipeline(args)
