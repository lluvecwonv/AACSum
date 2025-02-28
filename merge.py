import datetime
import random
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Union
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# -----------------------------------
# Aspect Parsing Function
# -----------------------------------

def parse_assigned_list(assigned_list: Union[str, None]) -> Dict[str, List[str]]:
    """
    Parses a list of assigned aspects and returns a dictionary.

    Parameters:
    - assigned_list (str or None): String containing assigned aspects in 'sentence -> aspect' format.

    Returns:
    - Dict[str, List[str]]: A dictionary mapping aspects to lists of sentences.
    """
    if assigned_list is None:
        print("Warning: assigned_list is None, returning empty dictionary.")
        return defaultdict(list)

    aspect_dict = defaultdict(list)
    for line in assigned_list.split('\n'):
        if '->' in line:
            sentence, aspect = line.split('->', 1)
            aspect = aspect.strip()
            sentence = sentence.strip()
            if aspect.lower() != '[none]' and not aspect.isdigit():
                aspect_dict[aspect].append(sentence)

    # Remove '[None]' key if present
    aspect_dict.pop('[None]', None)
    return aspect_dict


# -----------------------------------
# Aspect Merging Class
# -----------------------------------

class AspectMerge:
    def __init__(self, client):
        """
        Initializes the AspectMerge class.

        Parameters:
        - client: An API client (e.g., OpenAI client) for embeddings.
        """
        self.client = client

    def merge_aspects(self, aspects_A: Dict[str, List[str]], aspects_B: Dict[str, List[str]]) -> Dict[str, Tuple[List[str], List[str]]]:
        """
        Merges aspects from two dictionaries into clusters.

        Parameters:
        - aspects_A (Dict[str, List[str]]): First aspect dictionary.
        - aspects_B (Dict[str, List[str]]): Second aspect dictionary.

        Returns:
        - Dict[str, Tuple[List[str], List[str]]]: Merged dictionary mapping cluster centers to aspect lists from both inputs.
        """
        # Combine all aspects and cluster similar ones
        all_aspects = list(set(aspects_A.keys()).union(aspects_B.keys()))
        merged_aspect_map = self.cluster_similar_aspects(all_aspects)

        # Remap values in both dictionaries
        remapped_A = self.remap_values(merged_aspect_map, aspects_A)
        remapped_B = self.remap_values(merged_aspect_map, aspects_B)

        # Create merged aspect map
        return {
            aspect: (remapped_A.get(aspect, []), remapped_B.get(aspect, []))
            for aspect in merged_aspect_map.values()
        }

    def cluster_similar_aspects(self, aspects: List[str], similarity_threshold: float = 0.65) -> Dict[str, str]:
        """
        Clusters similar aspects based on cosine similarity of embeddings.

        Parameters:
        - aspects (List[str]): List of aspects to cluster.
        - similarity_threshold (float): Cosine similarity threshold for clustering.

        Returns:
        - Dict[str, str]: Mapping of original aspects to cluster centers.
        """
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
        """
        Selects the best representative aspect (cluster center) from a cluster.

        Parameters:
        - cluster (List[str]): List of aspects in the cluster.

        Returns:
        - str: The cluster center (most representative aspect).
        """
        embeddings = [self.get_gpt_embedding(asp) for asp in cluster]
        vectors = np.array(embeddings)
        cosine_matrix = cosine_similarity(vectors)
        avg_similarity = cosine_matrix.mean(axis=1)
        return cluster[avg_similarity.argmax()]

    def remap_values(self, merged_aspect_map: Dict[str, str], aspects: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Remaps aspects from original to their respective cluster centers.

        Parameters:
        - merged_aspect_map (Dict[str, str]): Mapping of original aspects to cluster centers.
        - aspects (Dict[str, List[str]]): Original aspect dictionary.

        Returns:
        - Dict[str, List[str]]: Dictionary with remapped aspects.
        """
        remapped_values = defaultdict(list)
        for original_asp, new_asp in merged_aspect_map.items():
            remapped_values[new_asp].extend(aspects.get(original_asp, []))
        return remapped_values

    def get_gpt_embedding(self, text: str) -> List[float]:
        """
        Retrieves the embedding vector for a given text using GPT embeddings.

        Parameters:
        - text (str): The input text for which to get the embedding.

        Returns:
        - List[float]: The embedding vector.
        """
        response = self.client.embeddings.create(input=text, model="text-embedding-3-small")
        return response.data[0].embedding
