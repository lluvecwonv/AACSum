# GPT Embedding + Cosine similarity
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import torch
import random

FIXED_SEED = 42
torch.manual_seed(FIXED_SEED)
torch.cuda.manual_seed(FIXED_SEED)
torch.cuda.manual_seed_all(FIXED_SEED)
random.seed(FIXED_SEED)

load_dotenv()
SERVICE_KEY = os.getenv('OPEN_AI_KEY')

def get_gpt_embedding(text):
      client = OpenAI(api_key=SERVICE_KEY)
      response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
      embedding = response.data[0].embedding

      return embedding

def aspect_similar(result_aspect, bench_aspect):   
  # Get embeddings for both aspects
  bench_embedding = get_gpt_embedding(bench_aspect)
  result_embedding = get_gpt_embedding(result_aspect)

  # Compute cosine similarity between the embeddings
  similarity_score = cosine_similarity([bench_embedding], [result_embedding])[0][0]
  
  return similarity_score