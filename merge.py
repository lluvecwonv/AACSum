import datetime
import random
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Union
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# Aspect parsing function
def parse_assigned_list(assigned_list: Union[str, None]) -> Dict[str, List[str]]:
    if assigned_list is None:
        print("Warning: assigned_list is None, returning empty dictionary.")
        return defaultdict(list)
    
    # 이후 코드 동일
    aspect_dict = defaultdict(list)
    for line in assigned_list.split('\n'):
        # print(f"Line: {line}")
        if '->' in line:
            sentence, aspect = line.split('->', 1)
            aspect = aspect.strip()
            sentence = sentence.strip()
            if aspect.lower() != '[none]' and not aspect.isdigit():
                aspect_dict[aspect].append(sentence)
    
    # '[None]' 키가 있는 경우 제거
    aspect_dict.pop('[None]', None)
    return aspect_dict

# Aspect merging class
class AspectMerge:
    def __init__(self, client):
        self.client = client

    def merge_aspects(self, aspects_A: Dict[str, List[str]], aspects_B: Dict[str, List[str]]) -> Dict[str, Tuple[List[str], List[str]]]:
        """Merge aspects from two aspect dictionaries."""
        all_aspects = list(set(aspects_A.keys()).union(aspects_B.keys()))
        merged_aspect_map = self.cluster_similar_aspects(all_aspects)
        
        remapped_A = self.remap_values(merged_aspect_map, aspects_A)
        remapped_B = self.remap_values(merged_aspect_map, aspects_B)
        
        return {aspect: (remapped_A.get(aspect, []), remapped_B.get(aspect, [])) for aspect in merged_aspect_map.values()}


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
                        if cosine_matrix[i][j] >= similarity_threshold:
                            cluster.append(other_asp)
                cluster_center = self.select_cluster_center(cluster)
                for attr in cluster:
                    aspect_map[attr] = cluster_center

        return aspect_map

    def select_cluster_center(self, cluster: List[str]) -> str:
        embeddings = [self.get_gpt_embedding(asp) for asp in cluster]
        vectors = np.array(embeddings)
        cosine_matrix = cosine_similarity(vectors)
        avg_similarity = cosine_matrix.mean(axis=1)
        return cluster[avg_similarity.argmax()]

    def remap_values(self, merged_aspect_map: Dict[str, str], aspects: Dict[str, List[str]]) -> Dict[str, List[str]]:
        remapped_values = defaultdict(list)
        for original_asp, new_asp in merged_aspect_map.items():
            remapped_values[new_asp].extend(aspects.get(original_asp, []))
        return remapped_values

    def get_gpt_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(input=text, model="text-embedding-3-small")
        return response.data[0].embedding
