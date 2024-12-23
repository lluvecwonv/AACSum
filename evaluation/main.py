import os
import json
import pandas as pd
import argparse
import torch
import random
import time
from tqdm import tqdm
from evaluation_for_one import Evaluation

# Set a fixed seed for reproducibility
FIXED_SEED = 42
torch.manual_seed(FIXED_SEED)
torch.cuda.manual_seed(FIXED_SEED)
torch.cuda.manual_seed_all(FIXED_SEED)
random.seed(FIXED_SEED)


# -----------------------
# Argument Parsing
# -----------------------
def set_args():
    parser = argparse.ArgumentParser(description="Evaluation script for COCOTRIP and AMASUM datasets.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to evaluate: cocotrip or amasum")
    parser.add_argument("--bench_folder_path", type=str, required=True, help="Path to the Benchmark folder")
    parser.add_argument("--result_folder_path", type=str, required=True, help="Path to the folder containing results")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save evaluation results")
    return parser.parse_args()


# -----------------------
# Helper Functions
# -----------------------
def find_aspect_files(folder_path, result_files=True):
    """
    Finds aspect-related JSON files in the given folder.

    Args:
        folder_path (str): Folder path to search.
        result_files (bool): Whether to find result files or benchmark files.

    Returns:
        list: List of file paths matching the criteria.
    """
    aspect_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if result_files and file.endswith("_result.json"):
                aspect_files.append(os.path.join(root, file))
            elif not result_files and file.startswith("BM") and file.endswith(".json"):
                aspect_files.append(os.path.join(root, file))
    return aspect_files


# -----------------------
# File Processing
# -----------------------
def process_files(dataset, result_folder, bench_folder, save_path):
    """
    Processes result and benchmark files for evaluation.

    Args:
        dataset (str): Dataset name (e.g., cocotrip or amasum).
        result_folder (str): Path to result folder.
        bench_folder (str): Path to benchmark folder.
        save_path (str): Path to save results.

    Returns:
        pd.DataFrame: DataFrame containing evaluation metrics.
    """
    total_metrics = {
        "num_compare": 0,
        "num_CASPR": 0,
        "num_aspect_accuracy_match": 0,
        "rouge_1_aspect": 0,
        "rouge_L_aspect": 0,
        "rouge_1_without_aspect": 0,
        "rouge_L_without_aspect": 0,
        "bertScore_aspect": 0,
        "bertScore_without_aspect": 0,
        "CASPR_score": 0,
        "aspect_accuracy": 0,
    }

    start_time = time.time()
    eval = Evaluation(None, None, None)

    # Process files based on dataset
    num_of_files = 18 if dataset == "cocotrip" else 646
    high_folders = (
        ["Electronics", "Home & Kitchen", "Tools & Home Improvement"]
        if dataset == "amasum"
        else []
    )

    if dataset == "cocotrip":
        for result_file in tqdm(os.listdir(result_folder)):
            if "aspect_review[" in result_file and result_file.endswith("_new.json"):
                number = result_file.split("[")[1].split("]")[0]
                result_path = os.path.join(result_folder, result_file)
                bench_path = os.path.join(bench_folder, f"aspect_review[{number}]_new.json")

                if os.path.exists(result_path) and os.path.exists(bench_path):
                    metrics = evaluate_files(eval, result_path, bench_path, number)
                    update_metrics(total_metrics, metrics)
                else:
                    print(f"Missing files for number {number}")
    else:  # For AMASUM
        for high_folder in high_folders:
            result_files = find_aspect_files(os.path.join(result_folder, high_folder))
            bench_files = find_aspect_files(os.path.join(bench_folder, high_folder), result_files=False)

            for result_path in tqdm(result_files):
                ids = extract_ids_from_filename(result_path)
                if not ids:
                    print(f"Invalid file name: {result_path}")
                    continue

                id1, id2 = ids
                bench_path = find_matching_bench_file(bench_files, id1, id2)

                if bench_path:
                    metrics = evaluate_files(eval, result_path, bench_path, f"{id1}_{id2}")
                    update_metrics(total_metrics, metrics)
                else:
                    print(f"Missing benchmark file for IDs {id1}, {id2}")

    # Final results and DataFrame creation
    metrics_data = calculate_final_metrics(total_metrics, num_of_files)
    print("Evaluation complete!")
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")
    return pd.DataFrame(metrics_data, index=[0])


def evaluate_files(eval, result_path, bench_path, identifier):
    """
    Evaluates a pair of result and benchmark files.

    Args:
        eval (Evaluation): Evaluation object.
        result_path (str): Path to result file.
        bench_path (str): Path to benchmark file.
        identifier (str): Identifier for the evaluation.

    Returns:
        dict: Metrics from the evaluation.
    """
    with open(result_path, "r", encoding="utf-8") as rf, open(bench_path, "r", encoding="utf-8") as bf:
        result_data = json.load(rf)
        benchmark_data = json.load(bf)

    eval.update_data(result_data, benchmark_data, identifier)
    return eval.eval_max_aspect_similar()


def update_metrics(total_metrics, metrics):
    """
    Updates total metrics with the results from a single evaluation.

    Args:
        total_metrics (dict): Accumulated metrics.
        metrics (tuple): Metrics from a single evaluation.
    """
    (
        num_compare,
        num_CASPR,
        CASPR_score,
        num_aspect_acc_match,
        aspect_accuracy,
        rouge_1_aspect,
        rouge_L_aspect,
        rouge_1_without_aspect,
        rouge_L_without_aspect,
        bertScore_aspect,
        bertScore_without_aspect,
    ) = metrics

    total_metrics["num_compare"] += num_compare
    total_metrics["num_CASPR"] += num_CASPR
    total_metrics["num_aspect_accuracy_match"] += num_aspect_acc_match
    total_metrics["CASPR_score"] += CASPR_score
    total_metrics["aspect_accuracy"] += aspect_accuracy
    total_metrics["rouge_1_aspect"] += rouge_1_aspect
    total_metrics["rouge_L_aspect"] += rouge_L_aspect
    total_metrics["rouge_1_without_aspect"] += rouge_1_without_aspect
    total_metrics["rouge_L_without_aspect"] += rouge_L_without_aspect
    total_metrics["bertScore_aspect"] += bertScore_aspect
    total_metrics["bertScore_without_aspect"] += bertScore_without_aspect


def calculate_final_metrics(total_metrics, num_of_files):
    """
    Calculates final averaged metrics from accumulated totals.

    Args:
        total_metrics (dict): Accumulated metrics.
        num_of_files (int): Total number of files.

    Returns:
        dict: Final averaged metrics.
    """
    return {
        "aspect Accuracy": total_metrics["num_aspect_accuracy_match"] / total_metrics["aspect_accuracy"]
        if total_metrics["aspect_accuracy"] != 0
        else 0,
        "rouge_1_aspect": total_metrics["rouge_1_aspect"] / total_metrics["num_compare"]
        if total_metrics["num_compare"] != 0
        else 0,
        "rouge_L_aspect": total_metrics["rouge_L_aspect"] / total_metrics["num_compare"]
        if total_metrics["num_compare"] != 0
        else 0,
        "rouge_1_without_aspect": total_metrics["rouge_1_without_aspect"] / num_of_files,
        "rouge_L_without_aspect": total_metrics["rouge_L_without_aspect"] / num_of_files,
        "bertScore_aspect": total_metrics["bertScore_aspect"] / total_metrics["num_compare"]
        if total_metrics["num_compare"] != 0
        else 0,
        "bertScore_without_aspect": total_metrics["bertScore_without_aspect"] / num_of_files,
        "CASPR": total_metrics["CASPR_score"] / total_metrics["num_CASPR"]
        if total_metrics["num_CASPR"] != 0
        else 0,
    }


def extract_ids_from_filename(filename):
    """Extracts IDs from the filename."""
    filename = os.path.basename(filename).replace("_result.json", "")
    ids = filename.split("_")
    return ids if len(ids) == 2 else None


def find_matching_bench_file(bench_files, id1, id2):
    """Finds the matching benchmark file for given IDs."""
    for bf in bench_files:
        if bf.endswith(f"BM_{id1}_{id2}.json") or bf.endswith(f"BM_{id2}_{id1}.json"):
            return bf
    return None


# -----------------------
# Main Function
# -----------------------
def main():
    args = set_args()
    final_df = process_files(args.dataset, args.result_folder_path, args.bench_folder_path, args.save_path)
    output_path = os.path.join(args.save_path, "final_result_average.csv")
    final_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
