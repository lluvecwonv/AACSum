from time import sleep
from openai import OpenAI
import tiktoken
import os 
from transformers import DataCollatorForSeq2Seq, AutoTokenizer, RobertaForMaskedLM
import random

import spacy
import torch
from torch.utils.data import Dataset, DataLoader


def load_model_and_tokenizer(model_name_or_path, chat_completion=False):
    
    api_key = 'your_api'
    client = OpenAI(api_key = api_key)
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    return tokenizer

def train_val_split(train_data, val_data):
    # train_data 처리
    origin = train_data["origin"]  # 'origin' 리스트에 텍스트가 저장됨
    labels = train_data["labels"]  # 'labels' 리스트에 라벨 정보가 저장됨


    # 데이터 길이 확인 (모든 리스트가 동일한 길이를 가져야 함)
    assert len(origin) == len(labels)

    # 텍스트, 라벨, 측면을 하나의 리스트로 묶고 셔플
    text_label = list(zip(origin, labels))
    random.shuffle(text_label)

    # 각 요소를 다시 분리
    train_text = [text for text, label in text_label]
    train_label = [label for text, label in text_label]


    # val_data 처리 (val_data도 동일한 구조를 가정)
    origin_val = val_data["origin"]
    labels_val = val_data["labels"]

    assert len(origin_val) == len(labels_val)

    val_text_label = list(zip(origin_val, labels_val))

    val_text = [text for text, label in val_text_label]
    val_label = [label for text, label in val_text_label]

    return train_text, train_label, val_text, val_label
        

        


class TokenDataset(Dataset):
    def __init__(
      self,
      texts,
      labels=None,
      max_len=512,
      tokenizer=None,
      model_name="bert-base-multilingual-cased",  
    ):
    
        self.len = len(texts)
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = labels
        self.model_name = model_name
        
        if "bert-base-multilingual-cased" in model_name:
            self.cls_token = "[CLS]"
            self.sep_token = "[SEP]"
            self.unk_token = "[UNK]"
            self.pad_token = "[PAD]"
            self.mask_token = "[MASK]"
            self.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.pad_token)
            
        elif "roberta" in model_name:
            self.bos_token = "<s>"
            self.eos_token = "</s>"
            self.sep_token = "</s>"
            self.cls_token = "<s>"
            self.unk_token = "<unk>"
            self.pad_token = "<pad>"
            self.mask_token = "<mask>"
            self.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.pad_token)
        
        elif "t5" in model_name:
            # T5 모델 설정
            self.cls_token = None  # T5는 CLS 토큰이 없음
            self.sep_token = None  # T5는 SEP 토큰이 없음
            self.pad_token = "<pad>"  # T5의 패드 토큰
            self.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.pad_token)
            # 데이터 콜레이터는 시퀀스 생성 작업을 위한 것임
            self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)

        else:
            raise NotImplementedError()

        
        self.nlp = spacy.load("en_core_web_sm")
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        text = self.texts[index]
   

        
        # 레이블이 있을 경우
        if self.labels is not None:
            labels = self.labels[index][:]
            tokenized_text, labels = self.tokenize_and_preserve_labels(
                text, labels, self.tokenizer
            )
            assert len(tokenized_text) == len(labels)
            labels.insert(0, False)  # 첫 번째 False 추가
            labels.append(False)     # 마지막 False 추가
            
        else:
            tokenized_text = self.tokenizer.tokenize(text)
            labels = []

        # [CLS] 토큰과 [SEP] 토큰 추가
        if  "t5" in self.model_name:
            tokenized_text = tokenized_text
        else:
            tokenized_text = [self.cls_token] + tokenized_text + [self.sep_token]

        # tokenized_text가 max_len보다 긴 경우 자르기
        if len(tokenized_text) > self.max_len:
            tokenized_text = tokenized_text[:self.max_len]
            if self.labels is not None:
                labels = labels[:self.max_len]
        else:
            tokenized_text = tokenized_text + [self.pad_token] * (self.max_len - len(tokenized_text))
            if self.labels is not None:
                labels = labels + [False] * (self.max_len - len(labels))

        # Attention mask 생성
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        attn_mask = [1 if tok != self.pad_token else 0 for tok in tokenized_text]

        ids_list=[]
        mask_list = []
        target_list = []
        
        ids_list.append(ids)
        mask_list.append(attn_mask)
        target_list.append(labels)
  
        ids = torch.tensor(ids, dtype=torch.long)
        mask = torch.tensor(attn_mask, dtype=torch.long)
        target = torch.tensor(labels, dtype=torch.long)


        return   torch.tensor(ids_list, dtype=torch.long).squeeze(),\
        torch.tensor(mask_list, dtype=torch.long).squeeze(),\
        torch.tensor(target_list, dtype=torch.long).squeeze()



    def split_string(self, input_string, ignore_tokens=set([","])):
        doc = self.nlp(input_string)
        word_list = []
        for word in doc:
            if word.lemma_ not in ignore_tokens:
                word_list.append(word.lemma_)
        return word_list
    
    
    def tokenize_and_preserve_labels(self, text, text_labels, tokenizer):
        tokenizer_text = []
        labels = []
        
        for word, label in zip(self.split_string(text), text_labels):
            tokenized_word = tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)
             
            tokenizer_text.extend(tokenized_word)
            labels.extend([label] * n_subwords)
            
        return tokenizer_text, labels   
            