import itertools
from openai import OpenAI
from typing import List, Dict, Any, Tuple
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from collections import defaultdict
from transformers import set_seed
 
from tqdm import tqdm


api_key = 'user-api-key'
client = OpenAI(api_key=api_key)
    

class AspectMerge:
    def __init__(self, api_key: str):
                self.client = OpenAI(api_key=api_key)

    def merge_aspects(self, aspects_A: Dict[str, List[str]], aspects_B: Dict[str, List[str]]) -> Dict[str, Tuple[List[str], List[str]]]:
        
        all_aspects = list(aspects_A.keys()) + list(aspects_B.keys())
        merged_aspect_map = self.cluster_similar_aspects(all_aspects)

        remapped_A = self.remap_values(merged_aspect_map, aspects_A)
        remapped_B = self.remap_values(merged_aspect_map, aspects_B)
        final_merged = {
            aspect: (remapped_A.get(aspect, []), remapped_B.get(aspect, []))
            for aspect in merged_aspect_map.values()
        }
        return final_merged

    def cluster_similar_aspects(self, aspects: List[str], similarity_threshold: float = 0.65) -> Dict[str, str]:
        if not aspects:
            return {}

        aspect_map = {}
        embeddings = [self.get_gpt_embedding(attr) for attr in aspects]
        vectors = np.array(embeddings)
        cosine_matrix = cosine_similarity(vectors)

        for i, aspect in enumerate(aspects):
            if aspect not in aspect_map:
                cluster = [aspect]
                for j, other_asp in enumerate(aspects):
                    if i != j and other_asp not in aspect_map:
                        similarity = cosine_matrix[i][j]
                        if similarity >= similarity_threshold:
                            cluster.append(other_asp)
                cluster_center = self.select_cluster_center(cluster)
                for attr in cluster:
                    aspect_map[attr] = cluster_center

        return aspect_map

    def select_cluster_center(self, cluster: List[str]) -> str:
        if not cluster:
            return ""
        embeddings = [self.get_gpt_embedding(asp) for asp in cluster]
        vectors = np.array(embeddings)
        cosine_matrix = cosine_similarity(vectors)
        avg_similarity = cosine_matrix.mean(axis=1)
        best_index = avg_similarity.argmax()
        return cluster[best_index]

    def remap_values(self, merged_aspect_map: Dict[str, str], aspects: Dict[str, List[str]]) -> Dict[str, List[str]]:
        remapped_values = defaultdict(list)

        for original_asp, new_asp in merged_aspect_map.items():
            new_asp_key = tuple(new_asp) if isinstance(new_asp, list) else new_asp
            current_values = aspects.get(original_asp, [])
            remapped_values[new_asp_key].extend(current_values)
        return remapped_values


    def get_gpt_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(input=text, model="text-embedding-3-small")
        embedding_data = response.data[0].embeddin
        return embedding_data

def merge_dicts(dicts: List[Dict[str, List[str]]]) -> Dict[str, List[str]]:
    merged_dict = defaultdict(list)
    for d in dicts:
        for key, value in d.items():
            merged_dict[key].extend(value)
    return dict(merged_dict)


def group_files_by_category(json_files, data_path):
    grouped_files = defaultdict(list)
    for json_file in json_files:
        category_path = os.path.relpath(os.path.dirname(json_file), start=data_path)
        grouped_files[category_path].append(json_file)
    return grouped_files

def read_json_files_in_directory(data_path):
    json_files = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def get_json_file_combinations(category_files):
    category_file_combinations = defaultdict(list)
    for category, files in category_files.items():
        file_combinations = list(itertools.combinations(files, 2))
        category_file_combinations[category].extend(file_combinations)
    return category_file_combinations


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default='/root/AACS/coco/test_raw_reviews_content.json',
    )
    parser.add_argument(
        '--result_path', type=str, default='/aspect_sum/sumarization/coco_asepct'
    )
    parser.add_argument( 
        '--prompt', type=str, default='/aspect_sum/templet/sum.txt'
    )
    parser.add_argument(
        '--seed', type=int, default=1212
    )

    args = parser.parse_args()
        
    set_seed(args.seed)

    args = parser.parse_args()
    os.makedirs(args.result_path, exist_ok=True)  # 저장파일 생성

    with open(args.data_path, 'r', encoding='utf-8') as f:
        print(f"Loading data from {args.data_path}")
        data = json.load(f)
    file_name = os.path.basename(args.data_path).split('.')[0]
    save_path = os.path.join(args.result_path,file_name,f'{file_name}.json')
    save_dir = os.path.dirname(save_path)
    
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    print(f"save_path: {save_path}")
    
    for index in range(len(data)):
            print(f"Processing data index: {index}")    
            save_path2 = os.path.join(args.result_path,file_name,f'aspect_review{[index]}_new.json')
            print(save_path)
            entity_a_assigned_list = data[str(index)]['entity_a']['assigned_list']
            entity_b_assigned_list = data[str(index)]['entity_b']['assigned_list']
            
            aspect_dict1 = defaultdict(list)
            aspect_dict2 = defaultdict(list)
            if entity_a_assigned_list:
                for line in entity_a_assigned_list.split('\n'):
                    if '->' in line:
                        sentence, aspect = line.split('->', 1)  
                        aspect = aspect.strip() 
                    # aspect_dict1[aspect.strip()].append(sentence.strip())
                        if aspect.lower() != '[None]' and not aspect.isdigit():
                                aspect_dict1[aspect].append(sentence.strip())
                # print(aspect_dict1)
            if entity_b_assigned_list:
                for line in entity_b_assigned_list.split('\n'):
                    if '->' in line:
                        sentence, aspect = line.split('->', 1) 
                        aspect = aspect.strip() 
                        if aspect.lower() != '[None]' and not aspect.isdigit():
                            aspect_dict2[aspect].append(sentence.strip())
                            # print(aspect_dict2)
            aspect_dict1.pop('[None]', None)
            aspect_dict2.pop('[None]', None)
            
            data[str(index)]['entity_a']['assigned_list'] = {aspect: sentences for aspect, sentences in aspect_dict1.items()}
            data[str(index)]['entity_b']['assigned_list'] = {aspect: sentences for aspect, sentences in aspect_dict2.items()}

        
            aspect_reviews1 = [data[str(index)]['entity_a']['assigned_list']]
            aspect_reviews2 = [data[str(index)]['entity_b']['assigned_list']]
    
            def merge_dicts(dicts: list) -> dict:
                merged_dict = defaultdict(list)
                for d in dicts:
                    for key, value in d.items():
                        merged_dict[key].extend(value)
                return dict(merged_dict)
            if isinstance(aspect_reviews1, list) and all(isinstance(d, dict) for d in aspect_reviews1):
                aspect_reviews1 = merge_dicts(aspect_reviews1)

            if isinstance(aspect_reviews2, list) and all(isinstance(d, dict) for d in aspect_reviews2):
                aspect_reviews2 = merge_dicts(aspect_reviews2)

            lm_asp_merge = AspectMerge(api_key=api_key)
            
            print("\nMerging aspects from both reviews:")
            
            merged_aspect_map = lm_asp_merge.merge_aspects(aspect_reviews1, aspect_reviews2)
            
            print("="*70)
            
            summary_output = []   
            aspect_dict = {}
            for aspect, (a_reviews, b_reviews) in merged_aspect_map.items():
                aspect_dict[aspect] = {
                    "A review_list": a_reviews,
                    "B review_list": b_reviews
                }
                
            print(aspect_dict)

            with open(args.prompt, 'r') as prompt_file:
                prompt_template = prompt_file.read()


            formatted_prompt = prompt_template.replace('{input}', json.dumps(aspect_dict))


            max_output_tokens = 16000
            response = client.chat.completions.create(
                model= 'gpt-4o-mini',
                messages=[
                    {"role": "system", "content": "You are an AI model tasked with comparing and summarizing reviews based on extracted aspects from two product reviews. The input consists of dictionaries where the keys are aspect names and the values are lists of sentences from reviews corresponding to each aspect. Your job is to write summaries for each aspect and clearly compare the similarities and differences between the two products based on these aspects."},
                    {"role": "user", "content": formatted_prompt},
                ],
                seed=1212,
                max_tokens=max_output_tokens,
            )


            summary = response.choices[0].message.content.strip() 

            summary = summary.replace('```json', '')
            summary = summary.replace('```', '').strip()


            try:
                # Attempt to load the summary as JSON
                json_data = json.loads(summary)
                
                
            except json.JSONDecodeError as e:
                # Handle JSON decode error
                print(f"Error during model processing: {e}")
                json_data = summary 
                
            # Save the updated dictionary back to the file
            with open(save_path2, 'w') as f:
                json.dump( json_data, f, indent=2)

    