from utils.rouge_evaluator import eval_rouge
from utils.bert_evaluator import eval_bertScore
from utils.CASPR_evaluator import calculate_caspr

import os
import sys
import transformers
import torch
import random
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()
SERVICE_KEY = os.getenv('OPEN_AI_KEY')

# Set fixed random seed
FIXED_SEED = 42
torch.manual_seed(FIXED_SEED)
torch.cuda.manual_seed(FIXED_SEED)
torch.cuda.manual_seed_all(FIXED_SEED)
random.seed(FIXED_SEED)
transformers.set_seed(FIXED_SEED)

class Evaluation:
    def __init__(self, result_json, benchmark_json, file_number):
        """
        Initialize the evaluation with result and benchmark data.
        """
        self.result_json = result_json
        self.benchmark_json = benchmark_json
        self.file_number = file_number

    def update_data(self, result_json, benchmark_json, number):
        """
        Update the data for evaluation.
        """
        self.result_json = result_json
        self.benchmark_json = benchmark_json
        self.file_number = number
        self.result_embeddings = {}
        self.benchmark_embeddings = {}

        self.precompute_embeddings()

    def precompute_embeddings(self):
        """
        Precompute text embeddings for aspect matching.
        """
        result_aspects = [item["aspect"] for item in self.result_json]
        benchmark_aspects = [item["aspect"] for item in self.benchmark_json]

        print("[INFO] Precomputing embeddings for result and benchmark aspects...")
        self.result_embeddings = self.get_batch_embeddings(result_aspects)
        self.benchmark_embeddings = self.get_batch_embeddings(benchmark_aspects)

    def get_batch_embeddings(self, texts):
        """
        Compute embeddings for a batch of texts.
        """
        client = OpenAI(api_key=SERVICE_KEY)
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return {text: response.data[i].embedding for i, text in enumerate(texts)}

    def aspect_similar(self, result_aspect, bench_aspect):
        """
        Compute cosine similarity between precomputed aspect embeddings.
        """
        result_embedding = self.result_embeddings.get(result_aspect)
        bench_embedding = self.benchmark_embeddings.get(bench_aspect)

        if result_embedding is None or bench_embedding is None:
            raise ValueError(f"[ERROR] Missing embeddings for {result_aspect} or {bench_aspect}")

        return cosine_similarity([result_embedding], [bench_embedding])[0][0]

    def calculate_cont_aspects_CASPR(self, total_caspr_score, caspr_count):
        """
        Calculate CASPR scores for content summaries.
        """
        for aspect_data in self.result_json:
            aspect = aspect_data["aspect"]
            cont = aspect_data.get("cont", {})
            entity_a = cont.get("entity_a", "").strip()
            entity_b = cont.get("entity_b", "").strip()

            if entity_a and entity_b:
                print(f"\nProcessing CASPR for aspect: '{aspect}'")
                caspr_score = calculate_caspr(entity_a, entity_b)
                total_caspr_score += caspr_score
                caspr_count += 1
            else:
                print(f"[INFO] Aspect '{aspect}' is missing 'Entity A' or 'Entity B'. Skipping CASPR.")

        return total_caspr_score, caspr_count

    def calculate_rouge_and_bert(self, benchmark_text, result_text, rouge_1, rouge_L, bert_benchList, bert_resultList):
        """
        Compute Rouge and BERTScore for a given text pair.
        """
        if not benchmark_text or not result_text:
            bert_benchList.append(benchmark_text)
            bert_resultList.append(result_text)
            return rouge_1, rouge_L, bert_benchList, bert_resultList

        bert_benchList.append(benchmark_text)
        bert_resultList.append(result_text)

        rougeScore = eval_rouge(result_text, benchmark_text)[0]
        rouge_1 += rougeScore["rouge-1"]["f"]
        rouge_L += rougeScore["rouge-l"]["f"]

        return rouge_1, rouge_L, bert_benchList, bert_resultList

    def evaluate_text_pairs(self, benchmark_text, result_text, text_type, bench_aspect, rouge_1, rouge_L, bert_benchList, bert_resultList):
        """
        Evaluate Rouge and BERTScore for a given aspect.
        """
        print(f"Evaluating {text_type} for aspect '{bench_aspect}'...")

        rouge_1, rouge_L, bert_benchList, bert_resultList = self.calculate_rouge_and_bert(
            benchmark_text, result_text, rouge_1, rouge_L, bert_benchList, bert_resultList
        )

        print(f"Finished evaluating Rouge and BERTScore {text_type} for aspect '{bench_aspect}'.\n")
        return rouge_1, rouge_L, bert_benchList, bert_resultList

    def evaluate_aspect_correctness(self, flag, bench_comm, result_comm, bench_entity_a, bench_entity_b, result_entity_a, result_entity_b, correctness_matches, correctness_total):
        """
        Evaluate aspect correctness by comparing benchmark and result structures.
        """
        if flag:
            is_structure_matched = (
                bool(bench_comm) == bool(result_comm)
                and bool(bench_entity_a) == bool(result_entity_a)
                and bool(bench_entity_b) == bool(result_entity_b)
            )
            if is_structure_matched:
                correctness_matches += 1
            correctness_total += 1
        else:
            correctness_total += 1

        return correctness_matches, correctness_total

    def calculate_rouge_BS_without_aspect(self):
        """
        Compute Rouge and BERTScore without considering aspects.
        """
        rst_text = " ".join([item["comm"] + " " + item["cont"]["entity_a"] + " " + item["cont"]["entity_b"] for item in self.result_json])
        ben_text = " ".join([item["comm"] + " " + item["cont"]["entity_a"] + " " + item["cont"]["entity_b"] for item in self.benchmark_json])

        rougeScore = eval_rouge(rst_text, ben_text)[0]
        bs_score = eval_bertScore([rst_text], [ben_text])

        return rougeScore["rouge-1"]["f"], rougeScore["rouge-l"]["f"], bs_score

    def eval_max_aspect_similar(self):
        """
        Perform aspect matching and evaluation.
        """
        rouge_1, rouge_L, bertScore = 0, 0, 0
        rouge_1_wo_asp, rouge_L_wo_asp, BS_wo_asp = 0, 0, 0
        aspect_correctness_matches, aspect_correctness_total = 0, 0
        caspr_count, total_caspr_score = 0, 0
        bert_benchList, bert_resultList = [], []

        print("=" * 50)
        print(f"Processing file {self.file_number} ... \n")
        print("=" * 50)

        caspr_total, caspr_count = self.calculate_cont_aspects_CASPR(total_caspr_score, caspr_count)
        rouge_1_wo_asp, rouge_L_wo_asp, BS_wo_asp = self.calculate_rouge_BS_without_aspect()

        for benchmark_entry in self.benchmark_json:
            benchmark_aspect = benchmark_entry["aspect"]
            best_match, highest_similarity, best_result_entry = None, -1, {}

            for result_entry in self.result_json:
                similarity = self.aspect_similar(result_entry["aspect"], benchmark_aspect)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = result_entry["aspect"]
                    best_result_entry = result_entry

            print(f"\nBest match for '{benchmark_aspect}' is '{best_match}' with similarity: {highest_similarity}")

            if highest_similarity >= 0.4:
                rouge_1, rouge_L, bert_benchList, bert_resultList = self.evaluate_text_pairs(
                    benchmark_entry["comm"], best_result_entry["comm"], "comm",
                    benchmark_aspect, rouge_1, rouge_L, bert_benchList, bert_resultList
                )

                aspect_correctness_matches, aspect_correctness_total = self.evaluate_aspect_correctness(
                    True, benchmark_entry["comm"], best_result_entry["comm"],
                    benchmark_entry["cont"]["entity_a"], benchmark_entry["cont"]["entity_b"],
                    best_result_entry["cont"]["entity_a"], best_result_entry["cont"]["entity_b"],
                    aspect_correctness_matches, aspect_correctness_total
                )

        return len(bert_benchList), caspr_count, caspr_total, aspect_correctness_matches, aspect_correctness_total, rouge_1, rouge_L, rouge_1_wo_asp, rouge_L_wo_asp, bertScore, BS_wo_asp