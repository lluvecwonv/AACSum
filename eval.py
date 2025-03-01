import os
import json
import pandas as pd
import argparse
import torch
import random
import time
from tqdm import tqdm
from eval import Evaluation

# Set a fixed random seed for reproducibility
FIXED_SEED = 42
torch.manual_seed(FIXED_SEED)
torch.cuda.manual_seed(FIXED_SEED)
torch.cuda.manual_seed_all(FIXED_SEED)
random.seed(FIXED_SEED)

def set_args():
    """ Define command-line arguments """
    parser = argparse.ArgumentParser(description="Evaluation Script for COCOTRIP and AMASUM")
    parser.add_argument("--dataset", type=str, required=True, choices=["cocotrip", "amasum"], help="Dataset to evaluate")
    parser.add_argument("--references", type=str, required=True, help="Path to the benchmark JSON file")
    parser.add_argument("--predictions", type=str, required=True, help="Path to the result JSON file")
    parser.add_argument("--save_folder_path", type=str, help="Path to save evaluation results")
    return parser.parse_args()

def load_json(file_path):
    """ Load a JSON file """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_matching_index(bench_data, result_entry):
    """ 
    Find a matching index in the benchmark data based on entity IDs for AMASUM.
    The order of entity_a and entity_b does not matter.
    """
    result_ids = {result_entry["Reviews_info"]["entity_a_id"], result_entry["Reviews_info"]["entity_b_id"]}
    
    for index, bench_entry in bench_data.items():
        bench_ids = {bench_entry["Reviews_info"]["entity_a_id"], bench_entry["Reviews_info"]["entity_b_id"]}
        
        if result_ids == bench_ids:
            return index  # Return matching index
    
    return None  # Return None if no match is found

def process_evaluation(dataset, bench_data, result_data):
    """ Perform evaluation by comparing benchmark and result data """
    evaluator = Evaluation(None, None, None)

    total_metrics = {
        "num_of_compare": 0,
        "num_CASPR": 0,
        "num_aspect_accuracy_match": 0,
        "CASPR_score": 0,
        "aspect_Accuracy": 0,
        "rouge_1_aspect": 0,
        "rouge_L_aspect": 0,
        "rouge_1_without_aspect": 0,
        "rouge_L_without_aspect": 0,
        "BertScore_aspect": 0,
        "BertScore_without_aspect": 0,
    }

    for result_index, result_entry in tqdm(result_data.items()):
        if dataset == "amasum":
            # AMASUM: Compare based on entity ID
            bench_index = find_matching_index(bench_data, result_entry)
        else:
            # COCOTRIP: Compare based on index
            bench_index = result_index if result_index in bench_data else None

        if bench_index is None:
            print(f"⚠️ No matching benchmark entry found for result index {result_index} (entity_a: {result_entry['Reviews_info']['entity_a_id']}, entity_b: {result_entry['Reviews_info']['entity_b_id']})")
            continue

        # Extract only the `Summarization` section for comparison
        benchmark_entry = bench_data[bench_index].get("Summarization", [])
        result_summarization = result_entry.get("Summarization", [])

        evaluator.update_data(result_summarization, benchmark_entry, bench_index)

        metrics = evaluator.eval_max_aspect_similar()

        total_metrics["num_of_compare"] += metrics[0]
        total_metrics["num_CASPR"] += metrics[1]
        total_metrics["num_aspect_accuracy_match"] += metrics[3]
        total_metrics["CASPR_score"] += metrics[2]
        total_metrics["aspect_Accuracy"] += metrics[4]
        total_metrics["rouge_1_aspect"] += metrics[5]
        total_metrics["rouge_L_aspect"] += metrics[6]
        total_metrics["rouge_1_without_aspect"] += metrics[7]
        total_metrics["rouge_L_without_aspect"] += metrics[8]
        total_metrics["BertScore_aspect"] += metrics[9]
        total_metrics["BertScore_without_aspect"] += metrics[10]

        print(f"✅ Processed result index {result_index} (matched with benchmark index {bench_index})")

    return total_metrics

def save_results(dataset, metrics, save_folder_path):
    """ Save evaluation results to a CSV file """
    num_files = 18 if dataset == "cocotrip" else 646  

    data = {
        "dataset": [dataset],
        "aspect Accuracy": [metrics["num_aspect_accuracy_match"] / metrics["aspect_Accuracy"] if metrics["aspect_Accuracy"] else 0],
        "rouge_1_aspect": [metrics["rouge_1_aspect"] / metrics["num_of_compare"] if metrics["num_of_compare"] else 0],
        "rouge_L_aspect": [metrics["rouge_L_aspect"] / metrics["num_of_compare"] if metrics["num_of_compare"] else 0],
        "rouge_1_without_aspect": [metrics["rouge_1_without_aspect"] / num_files],
        "rouge_L_without_aspect": [metrics["rouge_L_without_aspect"] / num_files],
        "bertScore_aspect": [metrics["BertScore_aspect"] / metrics["num_of_compare"] if metrics["num_of_compare"] else 0],
        "bertScore_without_aspect": [metrics["BertScore_without_aspect"] / num_files],
        "CASPR": [metrics["CASPR_score"] / metrics["num_CASPR"] if metrics["num_CASPR"] else 0]
    }

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(save_folder_path, f"final_result_{dataset}.csv"), index=False)
    print(f"✅ Results saved to {save_folder_path}")

def main():
    args = set_args()
    start_time = time.time()

    print("🔹 Loading Benchmark and Result JSON files...")
    bench_data = load_json(args.references)
    result_data = load_json(args.predictions)

    print("🔹 Evaluating results...")
    metrics = process_evaluation(args.dataset, bench_data, result_data)

    print("🔹 Saving results...")
    save_results(args.dataset, metrics, args.save_folder_path)

    print(f"⏳ Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
