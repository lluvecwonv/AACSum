from metrics.rouge_evaluator import eval_rouge
from metrics.bert_evaluator import eval_bertScore
from metrics.CASPR_evaluator import calculate_caspr

import os
import sys
import pandas as pd
import transformers
import torch
import random
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()
SERVICE_KEY = os.getenv('OPEN_AI_KEY')

# Set fixed seed for reproducibility
FIXED_SEED = 42
torch.manual_seed(FIXED_SEED)
torch.cuda.manual_seed(FIXED_SEED)
torch.cuda.manual_seed_all(FIXED_SEED)
random.seed(FIXED_SEED)
transformers.set_seed(FIXED_SEED)

class Evaluation:
    """
    Evaluation class to compute scores for benchmarks and result summaries.
    """
    def __init__(self, result_json, benchmark_json, file_number):
        self.result_json = result_json
        self.benchmark_json = benchmark_json
        self.file_number = file_number
        self.result_embeddings = {}
        self.benchmark_embeddings = {}

    def update_data(self, result_json, benchmark_json, number):
        """Update the data for evaluation."""
        self.result_json = result_json
        self.benchmark_json = benchmark_json
        self.file_number = number
        self.precompute_embeddings()

    def precompute_embeddings(self):
        """Precompute embeddings for result and benchmark aspects."""
        result_aspects = self.result_json.keys()
        benchmark_aspects = self.benchmark_json.keys()
        print("[INFO] Precomputing embeddings...")
        self.result_embeddings = self.get_batch_embeddings(list(result_aspects))
        self.benchmark_embeddings = self.get_batch_embeddings(list(benchmark_aspects))

    def get_batch_embeddings(self, texts):
        """Compute embeddings for a batch of texts."""
        client = OpenAI(api_key=SERVICE_KEY)
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        embeddings = {text: response.data[i].embedding for i, text in enumerate(texts)}
        return embeddings

    def aspect_similar(self, result_aspect, bench_aspect):
        """Calculate similarity between two aspects."""
        result_embedding = self.result_embeddings.get(result_aspect)
        bench_embedding = self.benchmark_embeddings.get(bench_aspect)

        if not result_embedding or not bench_embedding:
            raise ValueError(f"[ERROR] Missing embeddings for '{result_aspect}' or '{bench_aspect}'")
        
        return cosine_similarity([result_embedding], [bench_embedding])[0][0]

    def calculate_cont_aspects_CASPR(self, total_caspr_score, caspr_count):
        """Calculate CASPR score for 'cont' summaries."""
        for aspect, details in self.result_json.items():
            cont = details.get('cont', {})
            entity_a = cont.get('entity_a', "").strip()
            entity_b = cont.get('entity_b', "").strip()

            if entity_a and entity_b:
                print(f"\nProcessing CASPR for aspect: '{aspect}'")
                caspr_score = calculate_caspr(entity_a, entity_b)
                total_caspr_score += caspr_score
                caspr_count += 1
            else:
                print(f"[INFO] Missing 'Entity A' or 'Entity B' for aspect '{aspect}'")
        return total_caspr_score, caspr_count

    def extract_and_concatenate_text(self, data):
        """Concatenate all text values in a JSON object."""
        result_text = []

        def recursive_extract(obj):
            if isinstance(obj, dict):
                for value in obj.values():
                    recursive_extract(value)
            elif isinstance(obj, str):
                result_text.append(value if value.endswith('.') else value + '.')

        recursive_extract(data)
        return " ".join(result_text)

    def calculate_rouge_BS_without_aspect(self):
        """Calculate Rouge and BertScore without aspect consideration."""
        rst_text = self.extract_and_concatenate_text(self.result_json)
        ben_text = self.extract_and_concatenate_text(self.benchmark_json)

        rouge_score = eval_rouge(rst_text, ben_text)[0]
        r1_score = rouge_score["rouge-1"]["f"]
        rl_score = rouge_score["rouge-l"]["f"]
        bs_score = eval_bertScore([rst_text], [ben_text])

        print(f"Without aspect scores - Rouge-1: {r1_score}, Rouge-L: {rl_score}, BertScore: {bs_score}")
        return r1_score, rl_score, bs_score

    def calculate_rouge_and_bert(self, benchmark_text, result_text, rouge_1, rouge_L, bert_benchList, bert_resultList):
        """Calculate Rouge and BertScore for specific texts."""
        if not benchmark_text or not result_text:
            bert_benchList.append(benchmark_text)
            bert_resultList.append(result_text)
            return rouge_1, rouge_L, bert_benchList, bert_resultList
        
        bert_benchList.append(benchmark_text)
        bert_resultList.append(result_text)

        rouge_score = eval_rouge(result_text, benchmark_text)[0]
        rouge_1 += rouge_score["rouge-1"]["f"]
        rouge_L += rouge_score["rouge-l"]["f"]

        return rouge_1, rouge_L, bert_benchList, bert_resultList

    def eval_max_aspect_similar(self):
        """
        Perform full evaluation including Rouge, BertScore, CASPR, and Aspect Accuracy.
        """
        rouge_1, rouge_L, BS_wo_asp, aspect_matches, aspect_total = 0, 0, 0, 0, 0
        total_caspr_score, caspr_count = 0, 0
        bert_benchList, bert_resultList = [], []

        # CASPR evaluation
        total_caspr_score, caspr_count = self.calculate_cont_aspects_CASPR(total_caspr_score, caspr_count)

        # Without aspect evaluation
        rouge_1_wo_asp, rouge_L_wo_asp, BS_wo_asp = self.calculate_rouge_BS_without_aspect()

        for bench_aspect, bench_details in self.benchmark_json.items():
            best_match, highest_similarity = None, -1

            for result_aspect, result_details in self.result_json.items():
                similarity = self.aspect_similar(result_aspect, bench_aspect)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = result_aspect

            if highest_similarity >= 0.4:
                bench_comm = bench_details.get("comm", "")
                bench_cont = bench_details.get("cont", {})
                result_details = self.result_json.get(best_match, {})
                result_comm = result_details.get("comm", "")
                rouge_1, rouge_L, bert_benchList, bert_resultList = self.calculate_rouge_and_bert(
                    bench_comm, result_comm, rouge_1, rouge_L, bert_benchList, bert_resultList
                )

        bert_score = eval_bertScore(bert_resultList, bert_benchList)
        return {
            "total_comparisons": len(bert_benchList),
            "rouge_1": rouge_1,
            "rouge_L": rouge_L,
            "bert_score": bert_score,
            "caspr_score": total_caspr_score / caspr_count if caspr_count > 0 else 0,
        }
