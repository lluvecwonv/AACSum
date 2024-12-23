# kernel : myenv(python 3.11.9)
from bert_score import score
from transformers import AutoTokenizer, AutoModel
import transformers
import torch
import random

FIXED_SEED = 42
torch.manual_seed(FIXED_SEED)
torch.cuda.manual_seed(FIXED_SEED)
torch.cuda.manual_seed_all(FIXED_SEED)
random.seed(FIXED_SEED)
transformers.set_seed(FIXED_SEED)

def eval_bertScore(result_summary_list, benchmark_summary_list):
    # 원하는 모델 로드
    model_type = "microsoft/deberta-xlarge-mnli"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    
    # BERTScore 함수에서 모델 타입을 직접 설정
    P, R, F1 = score(result_summary_list, benchmark_summary_list, lang="en", model_type=model_type, verbose=True, device = device)
    
    # bertScore = F1.mean().item()
    # F1 리스트를 파이썬 리스트로 변환
    f1_scores = F1.tolist()
    
    # F1 점수들의 합계를 계산
    f1_sum = sum(f1_scores)
    print("bertScore - F1 Score:", f1_sum)

    return f1_sum