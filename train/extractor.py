
import argparse
import os
import random
import time
import json
import torch
import copy
import tiktoken


from tqdm import tqdm

parser= argparse.ArgumentParser(description ="compress any prompt and extraction")

parser.add_argument("--extractor", help="compress and extract method", default="gpt4")
parser.add_argument("--model_name", help="llm used to compress and extraction", default="gpt-4")
parser.add_argument(
"--save_key", help="the key to load the text to analysis", default="prompt" 
)

parser.add_argument(
    "--data_path",
    help="instruction file for compressing and extracting",
    default="/root/aspect_sum/scr/data/hotel", 
)

parser.add_argument(
    "--load_prompt_from",
    help="instruction file for compressing and extracting",
    default="/root/aspect_sum/scr/train/instruction.json", 
)

parser.add_argument("--chunk_size", type=int, default=3500)
parser.add_argument("--compress_prompt_id", type=int, default=0)
parser.add_argument("--aspect_prompt_id", type=int, default=1)
parser.add_argument("--instruct_prompt_id", type=int, default=2)

parser.add_argument(
    "--save_path",
    help ="path to save results", 
    default="/root/aspect_sum/scr/data/result/extraction_hotel.json",
)  

parser.add_argument("--ratio", help="compression rate", type=float, default=0.5)
parser.add_argument("--n_target_token", help='number of target tokens', type=int, default=-1)

args = parser.parse_args()


os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
#exist_ok=True 이미 디렉토리가 존재해도 오류 발생하지 않음

def read_txt_file(file_path):
    data_files = []
    
    def explore_folder(folder_path):
        with os.scandir(folder_path) as it:
            #os.scandir()는 지정된 폴더의 파일 및 하위 폴더 목록을 검색합니다.
            for entry in it:
                if entry.is_dir():
                    explore_folder(entry.path)
                elif entry.is_file() and entry.name.endswith(".txt"):
                    data_files.append(entry.path)
                    
    explore_folder(file_path)       
    return data_files

def process_data_file(data_files):
    datas = []
    for data_file in data_files:
        marker ="####"
        with open(data_file, "r",encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if marker in line:
                    data = line.split(marker)[0].strip()
                    datas.append(data)
                    #line.strip(marker) : 문자열을 marker 기준으로 분할
                    #[0] : 첫번째 문자열만 가져옴
    return datas
    
datas = process_data_file(read_txt_file(args.data_path))
print(len(datas))
print(type(datas))


if args.extractor == "gpt4":
    from Gpt4_extractor import Promptextractor
    prompt = json.load(open(args.load_prompt_from))
    #compress_prompt_id = 0
    #aspect_prompt_id = 1   
    com_system_prompt = prompt[str(args.compress_prompt_id)]["system_prompt"]
    com_user_prompt = prompt[str(args.compress_prompt_id)]["user_prompt"]
    
    asp_system_prompt = prompt[str(args.aspect_prompt_id)]["system_prompt"]
    asp_user_prompt = prompt[str(args.aspect_prompt_id)]["user_prompt"]

    instruct_prompt = prompt[str(args.instruct_prompt_id)]["user_prompt"]
    
    extractor = Promptextractor(
        model_name=args.model_name, com_system_prompt=com_system_prompt, com_user_prompt=com_user_prompt,
        asp_system_prompt=asp_system_prompt, asp_user_prompt=asp_user_prompt, instruct_prompt=instruct_prompt
    )

elif args.extract == "llmlingua" or args.extract == "longllmlingua":
    from llmlingua import PromptCompressor
    
    extractor = Promptextractor() 
else:
    raise NotImplementedError()


result = {}
results_list = []
total_time = 0

if os.path.exists(args.save_path):
    results = json.load((open(args.save_path)))
    
tokenizer = tiktoken.encoding_for_model("gpt-4")

    
idx= 0
start = 0
chunk_list = []
total_chunk = []

messages, len_mesages = extractor.query_template()
n = len_mesages

for idx in tqdm(range(len(datas))):
    
    data = datas[idx]
    data_ = copy.deepcopy(data)

    t = time.time()

    extract_list = []

    if args.extractor == "gpt4":
        ext = extractor.compress(data_ )
    ext = "".join(ext)
    total_time += time.time()-t
    ext_ = copy.deepcopy(ext)
    
    result ={} #딕셔너리 초기화 
    if(
        not (args.extractor == "llmlingua" or args.extractor == "longllmlingua")
        and len(ext)>0
    ):
        
        result['prompt_list'] = data_ 
        result['extractor_prompt'] = ext_

    entry ={str(idx): result}

    try:
        with open(args.save_path, "r+", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)  # 기존 데이터를 읽어들임
            except json.JSONDecodeError:
                existing_data = {}  # 파일이 비어있거나 잘못된 JSON 형식일 경우 초기화

            existing_data.update(entry)  # 새로운 데이터를 기존 데이터에 추가

            f.seek(0)  # 파일의 시작으로 이동
            f.truncate()  # 파일 내용을 지움
            json.dump(existing_data, f, indent=4, ensure_ascii=False)  # 업데이트된 데이터를 파일에 다시 씀
    except FileNotFoundError:
        # 파일이 존재하지 않을 경우 새 파일을 생성하여 초기화
        with open(args.save_path, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=4, ensure_ascii=False)
            
            
        
print(args.save_path, total_time)

