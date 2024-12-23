from time import sleep
from openai import OpenAI
import tiktoken
import os 
from transformers import DataCollatorForSeq2Seq, AutoTokenizer, RobertaForMaskedLM
import random

import spacy
import torch
from torch.utils.data import Dataset, DataLoader


def load_model_and_tokenizer(model_name_or_path, chat_completion=False):
    
    api_key = 'your_api'
    client = OpenAI(api_key = api_key)
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    return tokenizer


def train_val_split(train_data, val_data):
    """
    Splits training and validation data into text and labels.

    Parameters:
    - train_data (dict): Training data with 'origin' and 'labels' keys.
    - val_data (dict): Validation data with 'origin' and 'labels' keys.

    Returns:
    - Tuple of lists: (train_text, train_label, val_text, val_label).
    """
    # Process training data
    origin = train_data["origin"]
    labels = train_data["labels"]
    assert len(origin) == len(labels), "Mismatch in training data lengths."

    text_label = list(zip(origin, labels))
    random.shuffle(text_label)
    train_text, train_label = zip(*text_label)

    # Process validation data
    origin_val = val_data["origin"]
    labels_val = val_data["labels"]
    assert len(origin_val) == len(labels_val), "Mismatch in validation data lengths."

    val_text_label = list(zip(origin_val, labels_val))
    val_text, val_label = zip(*val_text_label)

    return list(train_text), list(train_label), list(val_text), list(val_label)



class TokenDataset(Dataset):
    def __init__(self, texts, labels=None, max_len=512, tokenizer=None, model_name="bert-base-multilingual-cased"):
        """
        Initializes a dataset for token classification tasks.

        Parameters:
        - texts (list): List of input texts.
        - labels (list or None): List of label sequences corresponding to texts.
        - max_len (int): Maximum token length for inputs.
        - tokenizer: Tokenizer to preprocess texts.
        - model_name (str): Model name for handling special tokens.
        """
        self.texts = texts
        self.labels = labels
        self.len = len(texts)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.model_name = model_name

        # Define special tokens based on model type
        if "bert-base-multilingual-cased" in model_name:
            self.cls_token = "[CLS]"
            self.sep_token = "[SEP]"
            self.pad_token = "[PAD]"
        elif "roberta" in model_name:
            self.cls_token = "<s>"
            self.sep_token = "</s>"
            self.pad_token = "<pad>"
        else:
            raise NotImplementedError(f"Model {model_name} is not supported.")
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.pad_token)

        # Load SpaCy for tokenization
        self.nlp = spacy.load("en_core_web_sm")

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return self.len

    def __getitem__(self, index):
        """
        Retrieves a sample and its corresponding labels (if available).

        Parameters:
        - index (int): Index of the sample.

        Returns:
        - Tuple: (input_ids, attention_mask, labels)
        """
        text = self.texts[index]

        if self.labels:
            labels = self.labels[index][:]
            tokenized_text, labels = self.tokenize_and_preserve_labels(text, labels, self.tokenizer)
            assert len(tokenized_text) == len(labels), "Tokenized text and labels length mismatch."
            labels = [False] + labels + [False]  # Add False for special tokens
        else:
            tokenized_text = self.tokenizer.tokenize(text)
            labels = []

        # Add special tokens
        tokenized_text = [self.cls_token] + tokenized_text + [self.sep_token]

        # Truncate or pad tokenized text and labels
        if len(tokenized_text) > self.max_len:
            tokenized_text = tokenized_text[:self.max_len]
            if labels:
                labels = labels[:self.max_len]
        else:
            tokenized_text += [self.pad_token] * (self.max_len - len(tokenized_text))
            if labels:
                labels += [False] * (self.max_len - len(labels))

        # Convert tokens to IDs and create attention mask
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        attention_mask = [1 if tok != self.pad_token else 0 for tok in tokenized_text]

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long) if labels else torch.tensor([], dtype=torch.long),
        )

    def tokenize_and_preserve_labels(self, text, text_labels, tokenizer):
        """
        Tokenizes text and aligns the labels with the tokens.

        Parameters:
        - text (str): Input text.
        - text_labels (list): Labels corresponding to the input text.
        - tokenizer: Tokenizer to preprocess text.

        Returns:
        - Tuple: (tokenized_text, aligned_labels)
        """
        tokenized_text, labels = [], []
        for word, label in zip(self.split_string(text), text_labels):
            tokenized_word = tokenizer.tokenize(word)
            tokenized_text.extend(tokenized_word)
            labels.extend([label] * len(tokenized_word))
        return tokenized_text, labels

    def split_string(self, input_string, ignore_tokens=set([","])):
        """
        Splits a string into words using SpaCy while ignoring specified tokens.

        Parameters:
        - input_string (str): The input string to split.
        - ignore_tokens (set): Set of tokens to ignore during splitting.

        Returns:
        - list: List of words after splitting.
        """
        doc = self.nlp(input_string)
        return [word.lemma_ for word in doc if word.lemma_ not in ignore_tokens]
