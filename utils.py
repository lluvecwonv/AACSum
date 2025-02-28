import os
import json
import itertools
import string
import random
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq, AutoTokenizer, RobertaForMaskedLM

# -----------------------------------
# Dataset for Token Classification
# -----------------------------------

class TokenClfDataset(Dataset):
    def __init__(self, texts, max_len=512, tokenizer=None, model_name="xlm-roberta-large"):
        """
        Initializes the Token Classification Dataset.

        Parameters:
        - texts (list): A list of text inputs.
        - max_len (int): Maximum length of tokens per text.
        - tokenizer (object): Tokenizer to tokenize text.
        - model_name (str): The model name (e.g., 'xlm-roberta-large').
        """
        self.texts = [item for sublist in texts for item in sublist] if any(isinstance(i, list) for i in texts) else texts
        self.len = len(texts)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.model_name = model_name

        # Define special tokens for different models
        if "bert-base-multilingual-cased" in model_name:
            self.cls_token = "[CLS]"
            self.sep_token = "[SEP]"
            self.unk_token = "[UNK]"
            self.pad_token = "[PAD]"
            self.mask_token = "[MASK]"
        elif "xlm-roberta-large" in model_name:
            self.bos_token = "<s>"
            self.eos_token = "</s>"
            self.sep_token = "</s>"
            self.cls_token = "<s>"
            self.unk_token = "<unk>"
            self.pad_token = "<pad>"
            self.mask_token = "<mask>"
        else:
            raise NotImplementedError("Model not supported.")

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset.

        Parameters:
        - index (int): Index of the data point.

        Returns:
        - dict: A dictionary containing token IDs and attention masks.
        """
        text = self.texts[index]
        tokenized_text = self.tokenizer.tokenize(text)

        # Add special tokens
        tokenized_text = [self.cls_token] + tokenized_text + [self.sep_token]

        # Truncate or pad the tokens
        if len(tokenized_text) > self.max_len:
            tokenized_text = tokenized_text[:self.max_len]
        else:
            tokenized_text += [self.pad_token] * (self.max_len - len(tokenized_text))

        # Create attention mask
        attn_mask = [1 if tok != self.pad_token else 0 for tok in tokenized_text]

        # Convert tokens to IDs
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(attn_mask, dtype=torch.long),
        }

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
        - int: The number of data points.
        """
        return self.len

# -----------------------------------
# Token Utility Functions
# -----------------------------------

def is_begin_of_new_word(token, model_name, force_tokens, token_map):
    """
    Determines whether a token is the beginning of a new word.

    Parameters:
    - token (str): The token to check.
    - model_name (str): The model name.
    - force_tokens (set): Set of tokens to always consider as new words.
    - token_map (dict): A mapping of added tokens.

    Returns:
    - bool: True if the token starts a new word, False otherwise.
    """
    if "bert-base-multilingual-cased" in model_name:
        return token.lstrip("##") in force_tokens or not token.startswith("##")
    elif "roberta" in model_name:
        return token.startswith("▁") or token in string.punctuation or token in force_tokens
    else:
        raise NotImplementedError("Model not supported.")

def get_pure_token(token, model_name):
    """
    Removes special characters from the token.

    Parameters:
    - token (str): The token to process.
    - model_name (str): The model name.

    Returns:
    - str: The pure token.
    """
    if "bert-base-multilingual-cased" in model_name:
        return token.lstrip("##")
    elif "roberta" in model_name:
        return token.lstrip("▁")
    else:
        raise NotImplementedError("Model not supported.")

def replace_added_token(token, token_map):
    """
    Replaces added tokens in the input with original tokens.

    Parameters:
    - token (str): The token to replace.
    - token_map (dict): Mapping of original to added tokens.

    Returns:
    - str: The token with replacements applied.
    """
    for ori_token, new_token in token_map.items():
        token = token.replace(new_token, ori_token)
    return token

# -----------------------------------
# File Utility Functions
# -----------------------------------

def load_txt(file_path):
    """
    Loads the content of a text file.

    Parameters:
    - file_path (str): Path to the text file.

    Returns:
    - str: The content of the file.
    """
    with open(file_path, 'r') as f:
        return f.read()

def group_files_by_category(json_files, data_path):
    """
    Groups files by their category (relative directory path).

    Parameters:
    - json_files (list): List of JSON file paths.
    - data_path (str): Root directory containing the files.

    Returns:
    - dict: Dictionary grouping files by category.
    """
    grouped_files = defaultdict(list)
    for json_file in json_files:
        category_path = os.path.relpath(os.path.dirname(json_file), start=data_path)
        grouped_files[category_path].append(json_file)
    return grouped_files

def get_json_file_combinations(category_files):
    """
    Generates all combinations of two JSON files within the same category.

    Parameters:
    - category_files (dict): Dictionary of files grouped by category.

    Returns:
    - dict: Dictionary of file combinations grouped by category.
    """
    category_file_combinations = defaultdict(list)
    for category, files in category_files.items():
        file_combinations = list(itertools.combinations(files, 2))
        category_file_combinations[category].extend(file_combinations)
    return category_file_combinations

def read_json_files_in_directory(data_path):
    """
    Reads all JSON file paths in a given directory.

    Parameters:
    - data_path (str): Directory path to search for JSON files.

    Returns:
    - list: List of JSON file paths.
    """
    json_files = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files
