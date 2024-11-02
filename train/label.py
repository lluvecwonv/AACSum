import argparse
import json
import logging  
import os 
from collections import defaultdict
import spacy
from datasets import load_dataset
from tqdm import tqdm
import torch
import os

parser = argparse.ArgumentParser(description="annotate token")

parser.add_argument(
    "--dataset_name",
    help="dataset_name",
    default='lluvecwonv/reviews_compasp_val'
)


parser.add_argument(
    "--window_size",
    help="window size",
    type=int,
    default=400,
)

parser.add_argument(
    "--save_path",
    help="path to save results",
    default="/root/aspect_sum/scr/annotation/reviwes/label_val.json",
)

parser.add_argument(
    "--verbose",
    help="print debug info",
    action=argparse.BooleanOptionalAction,
    default=False,
)

args = parser.parse_args()

os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
logging.basicConfig(
    filename = f"{os.path.dirname(args.save_path)}/log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
#모듈의 basicconfig함수를 사용하면 로깅에 대한 기본 설정을 간단히 설정할 수 있다.
#filename은 로그 파일의 이름을 설정한다.
#level은 로깅의 수준을 설정한다. 에러만 기록할것인가? 디버그까지 기록할것인가?
#로그 메세지 형식 지정 
logger = logging.getLogger()

nlp = spacy.load("en_core_web_sm")


def is_equal(token1,token2):
    return token1.lower() == token2.lower()


origins, comps, aspects= [], [], []

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

filepath = 'path_to_your_json_file.json'
dataset = load_dataset(args.dataset_name, split="train")
#compressed 된 데이터를 불러온다.
  
for i, sample in enumerate(dataset):
    origins.append(sample["origin_review"])
    comps.append(sample["compress_text"])
    aspects.append(sample["aspects"])


res ={}
res_pt = defaultdict(list)
#빈리스트 []를 기본값으로 생성하여 딕셔너리를 생성
  
num_sample = 0
compression_rate_avg = 0
find_rate_avg = 0
variation_rate_avg = 0
matching_rate_avg = 0
hitting_rate_avg = 0
alignment_gap_avg = 0


def split_string(input_string, ignore_tokens=set([",", ".", "!", "?"])):
    doc = nlp(input_string)
    word_list = []
    for word in doc:
        if word.text not in ignore_tokens:
            word_list.append(word.text.lower())
    return word_list

#def split_string(input_string, ignore_tokens=set([","])):
#    doc = nlp(input_string)
#    word_list = []
#    for word in doc:
#        if word.lemma_ not in ignore_tokens:
#            word_list.append(word.lemma_)
#    return word_list


def is_equal(token1,token2):
    return token1.lower() == token2.lower()

for chunk_idx,(origin,comp,aspect) in tqdm(enumerate(zip(origins,comps,aspects))):
    num_sample += 1
    origin_tokens = split_string(origin)
    comp_tokens = split_string(comp)
    #리스트 변환
    origin_tokens_set = set(origin_tokens)
    for origin_token in  origin_tokens:
        origin_tokens_set.add(origin_token.lower())
    
    num_find = 0
    prev_idx = 0
    back_cnt = 0
    num_origin_tokens = len(origin_tokens)
    labels = [False]*num_origin_tokens
    
    for comp_token in comp_tokens:
        print("Current comp_token:", comp_token) 
        flag = False
        #comp_token이 origin_tokens_set에 있거나 origin_tokens_set의 소문자가 있으면 num_find를 1증가시킨다.
        if comp_token in origin_tokens_set or comp_token.lower() in origin_tokens_set:
            num_find +=1
        for i in range(args.window_size):
            #Look forward
            token_idx = min(prev_idx+i, num_origin_tokens-1)
            #마지막인덱스를 초과하지않도록 설정
            if is_equal(origin_tokens[token_idx],comp_token)and not labels[token_idx]:  
                labels[token_idx] = True
                if token_idx - prev_idx > args.window_size//2:
                    #ex )token_idx =20, prev_idx = 10, args.window_size = 10
                    #20-10 = 10  > 10//2 = 5 
                    prev_idx += args.window_size//2
                    #prev_idx = 15
                    print(prev_idx)
                    #token_idx와 prev_idx의 차이가 args.window_size//2보다 크면 prev_idx를 args.window_size//2만큼 증가시킨다.
                else:
                    prev_idx = token_idx
                    print(prev_idx)
                if args.verbose:
                    print(
                        comp_token,
                        token_idx,
                        prev_idx,
                        origin_tokens[token_idx - 1 : token_idx + 2],
                    )
                flag = True
                break
    retrieval_tokens = []
    for idx, origin_token in enumerate(origin_tokens):
        if labels[idx]: #labels[idx]가 True이면
            retrieval_tokens.append(origin_token)
    retrieval = " ".join(retrieval_tokens)

    comp_rate = len(comp_tokens)/len(origin_tokens)
    #압축되어있는 토큰의 개수/원본 토큰의 개수 
    if len(comp_tokens)>0:
        find_rate = num_find/len(comp_tokens)
        #압축된 토큰 중에 찾에 찾은 토큰비율 
    else:
        find_rate = 0.0
    variation_rate = 1 - find_rate
    #찾지못한 토큰 비율
    hitting_rate = num_find/len(origin_tokens)
    #원본 토큰 중에서 찾은 토큰 비율
    matching_rate = sum(labels)/len(labels)
    #레이블의 평균값을 나타낸는 값 
    alignment_gap = hitting_rate - matching_rate
    #원본토큰중에서 찾은 토큰의 비율과 레이블의 평균간의 차이를 나타내는 값
    #차이가 크면 레이블이 잘못된것이다.
    
    compression_rate_avg += comp_rate
    find_rate_avg += find_rate
    variation_rate_avg += variation_rate
    hitting_rate_avg += hitting_rate
    matching_rate_avg += matching_rate
    alignment_gap_avg += alignment_gap
    
    if alignment_gap > 0.1:
        print(origin)
        print("-" * 50)
        print(comp)
        print("-" * 50)
        print(retrieval)
        print("-" * 50)
        print(origin_tokens)
        print("-" * 50)
        print(comp_tokens)
        print("-" * 50)
        print(retrieval_tokens)
        print("=" * 50)

        print(
            f"comp rate: {comp_rate}, variation_rate: {variation_rate}, alignment_gap: {alignment_gap}"
        )
                
    res[chunk_idx] = {
        "labels": labels,
        "origin": origin,
        "comp": comp,
        "retrieval": retrieval,
        "origin_tokens": origin_tokens,
        "comp_rate": comp_rate,
        "variation_rate": variation_rate,
        "hitting_rate": hitting_rate,
        "matching_rate": matching_rate,
        "alignment_gap": alignment_gap,
        "aspects": aspect,
    }
    
    res_pt["labels"].append(labels)
    res_pt["origin"].append(origin)
    res_pt["comp"].append(comp)
    res_pt["retrieval"].append(retrieval)
    res_pt["origin_tokens"].append(origin_tokens)
    res_pt["comp_rate"].append(comp_rate)
    res_pt["variation_rate"].append(variation_rate)
    res_pt["hitting_rate"].append(hitting_rate)
    res_pt["matching_rate"].append(matching_rate)
    res_pt["alignment_gap"].append(alignment_gap)
    res_pt["aspects"].append(aspect)

    #주기적인 저장
    # JSON 파일을 이어서 저장하기
    if int(chunk_idx) % 1000 == 0:
        if os.path.exists(args.save_path):
            with open(args.save_path, "r") as f:
                res_existing = json.load(f)
        else:
            res_existing = {}
        
        # 기존 리스트에 새로운 결과 추가
        res_existing.update(res)
        
        # 새로운 리스트를 파일에 저장
        json.dump(res_existing, open(args.save_path, "w"), indent=4)
        
        # PyTorch 텐서 저장하기
        torch.save(res_pt, args.save_path.replace(".json", ".pt"))

    
json.dump(res, open(args.save_path, "w"), indent=4)
torch.save(res_pt, args.save_path.replace(".json", ".pt"))
 
compression_rate_avg = compression_rate_avg / num_sample
find_rate_avg = find_rate_avg / num_sample
variation_rate_avg = variation_rate_avg / num_sample
matching_rate_avg = matching_rate_avg / num_sample
hitting_rate_avg = hitting_rate_avg / num_sample
alignment_gap_avg = alignment_gap_avg / num_sample

print_info = f"window size: {args.window_size}, comp rate: {compression_rate_avg}, hitting_rate: {hitting_rate_avg}, retrieval rate: {matching_rate_avg}"
print(print_info)
logger.info(print_info)

output_file_path = f"{args.dataset_name}.txt"
output_dir = os.path.dirname(output_file_path)

os.makedirs(output_dir, exist_ok=True)

with open(output_file_path, "w") as file:
    file.write(print_info)
    
    
#comp_rate: 압축된 텍스트가 원본 텍스트에 비해 얼마나 축소되었는지.
#find_rate: 압축된 텍스트에서 검색된 토큰의 비율.
#variation_rate: 압축된 텍스트에서 검색되지 않은 토큰의 비율.
#hitting_rate: 원본 텍스트에서 검색된 토큰의 비율.
#matching_rate: 레이블이 1인 경우의 비율.
#alignment_gap: 검색된 토큰 비율과 레이블 비율 간의 차이, 레이블의 신뢰성을 평가하는 지표.
#0에 가까울수록 레이블이 올바르게 설정됨 