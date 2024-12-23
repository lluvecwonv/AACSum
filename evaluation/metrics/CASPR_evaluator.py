import pandas as pd
from transformers import pipeline
import numpy as np
import transformers
import torch
import random

# Set fixed seed for reproducibility
FIXED_SEED = 42
torch.manual_seed(FIXED_SEED)
torch.cuda.manual_seed(FIXED_SEED)
torch.cuda.manual_seed_all(FIXED_SEED)
random.seed(FIXED_SEED)
transformers.set_seed(FIXED_SEED)

# Function to split complex text into sentences
def decompose_sentences(text):
    """
    Decompose text into individual sentences.
    """
    sentences = text.split('.')
    return [sentence.strip() for sentence in sentences if sentence.strip()]

# Load NLI model
nli_model = pipeline("text-classification", model="roberta-large-mnli", device=0)

# Evaluate logical relationship between two sentences
def nli_score(sentence1, sentence2):
    """
    Compute the logical relationship (entailment, contradiction, neutral)
    between two sentences using an NLI model.
    """
    result1 = nli_model(f"{sentence1} entails {sentence2}")
    result2 = nli_model(f"{sentence2} entails {sentence1}")
    
    label1 = result1[0]['label']
    label2 = result2[0]['label']
    
    print(f"Comparing: '{sentence1}' <-> '{sentence2}'")
    print(f" -> Result 1: {label1}, Result 2: {label2}")
    
    if label1 == 'CONTRADICTION' or label2 == 'CONTRADICTION':
        return 'contradiction'
    elif label1 == 'ENTAILMENT' or label2 == 'ENTAILMENT':
        return 'entailment'
    else:
        return 'neutral'

# Compute CASPR score between two summaries
def calculate_caspr(summary1, summary2):
    """
    Compute CASPR (Comparative Aspect Similarity and Polarity Recognition) score
    between two summaries.
    """
    scores = []
    summaries = [(summary1, summary2), (summary2, summary1)]  # Compare both directions
    
    for s1, s2 in summaries:
        sentences1 = decompose_sentences(s1)
        sentences2 = decompose_sentences(s2)
        
        for sentence1 in sentences1:
            contradictions = 0
            entailments = 0
            neutrals = 0
            
            for sentence2 in sentences2:
                relationship = nli_score(sentence1, sentence2)
                if relationship == 'contradiction':
                    contradictions += 1
                elif relationship == 'entailment':
                    entailments += 1
                else:
                    neutrals += 1
            
            # Determine score for the sentence pair
            if contradictions == 0 and entailments == 0:
                scores.append(1)  # All neutral
            elif contradictions > entailments:
                scores.append(1)  # Contradictions outweigh entailments
            else:
                scores.append(-1)  # Entailments outweigh contradictions
    
    if not scores:
        return 0  # Return 0 if no valid scores were computed
    
    print(f"Calculated Scores: {scores}")
    
    # Normalize CASPR score to the range [0, 100]
    caspr_score = (sum(scores) / len(scores) + 1) * 50
    return caspr_score
