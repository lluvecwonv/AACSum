# kernel : myenv(python 3.11.9)
from rouge import Rouge
import transformers
import torch
import random

FIXED_SEED = 42
torch.manual_seed(FIXED_SEED)
torch.cuda.manual_seed(FIXED_SEED)
torch.cuda.manual_seed_all(FIXED_SEED)
random.seed(FIXED_SEED)

def eval_rouge(hypothesis, reference):
    rouge = Rouge(stats=["f"])
    scores = rouge.get_scores(hypothesis, reference)
    
    print("F1 Scores: ", scores)
    
    return scores