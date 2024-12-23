import copy
import re
import numpy as np
from regex import F
import tiktoken
from typing import List
import torch
from utils import is_begin_of_new_word
from utils import TokenClfDataset, get_pure_token, replace_added_token
from torch.utils.data import DataLoader
import string
import torch.nn.functional as F
from transformers import AutoTokenizer, XLMRobertaForTokenClassification, XLMRobertaForCausalLM, GenerationConfig, AutoConfig

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize


class AspectCompressor:
    def __init__(self, model_family ,tokenizer, max_seq_len, model,device,max_len): 
        self.model_family = model_family
        self.tokenizer = tokenizer
        self.chunk_end_tokens = ["<eos>", ".", "?", "!"]
        self.model = model
        self.max_seq_len = max_seq_len
        self.device = device
        self.max_batch_siz = 8
        self.oai_tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        self.max_len = max_len
        
    
        self.special_tokens = self.tokenizer.all_special_tokens
      
    def get_token_length(
            self,
            text: str,
            tokenizer,
            add_special_tokens: bool = True,
            use_oai_tokenizer: bool = False,
        ):
            if use_oai_tokenizer:
                oai_tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
                return len(oai_tokenizer.encode(text))
            else:
                return len(
                    tokenizer(text, add_special_tokens=add_special_tokens).input_ids
                )
        
    def __chunk_context(
        self, 
        origin_text, 
        chunk_end_tokens):
        
        max_len = self.max_seq_len - 2
        origin_list = []
        origin_tokens = self.tokenizer.tokenize(origin_text)
        n = len(origin_tokens)
        st = 0
        while st < n:
            if st + max_len > n - 1:
                chunk = self.tokenizer.convert_tokens_to_string(origin_tokens[st:n])
                origin_list.append(chunk)
                break
            else:
                ed = st + max_len
                for j in range(0, ed - st):
                    if origin_tokens[ed - j] in chunk_end_tokens:
                        ed = ed - j
                        break
                chunk = self.tokenizer.convert_tokens_to_string(
                    origin_tokens[st : ed + 1]
                )
                origin_list.append(chunk)
                st = ed + 1
        return origin_list

    def load_model(self):  
        
        self.model.to(self.device)


    
    def __call__(self, *args, **kwargs):
        return self.compress_aspect(*args, **kwargs)

    def compress_aspect(
        self,  
        text,
        target_token: int = -1,
        rate: float = 0.5,
        target_context: int = -1, 
        context_level_rate: float = 1.0, 
        context_level_target_token: int = -1, 
        token_to_word: str = "mean",
        force_tokens: List[str] = [],
        force_reserve_digit: bool = False,
        force_context_ids: List[int] = [],
        use_token_level_filter: bool = True,
        drop_consecutive: bool = False,
        return_word_label: bool = False
    ):
        token_map = {}

        for i, t in enumerate(force_tokens):
            if len(self.tokenizer.tokenize(t)) != 1:
                token_map[t] = self.added_tokens[i]
        
        chunk_end_tokens = []
        
        chunk_end_tokens = copy.deepcopy(chunk_end_tokens)
        for c in chunk_end_tokens:
            if c in token_map:
                chunk_end_tokens.append(token_map[c])
        chunk_end_tokens = set(chunk_end_tokens)
        
        
        if type(text) == str:
            text = [text]
            
        context_ = copy.deepcopy(text)
        
        n_original_token = 0
        context_chunked = []
        
        for i in range(len(context_)):
            n_original_token += self.get_token_length(
                context_[i], self.tokenizer, use_oai_tokenizer=True  
            )
            context_chunked.append(
                self.__chunk_context(context_[i], self.chunk_end_tokens)  
            ) 
    
        if (
            target_context <= 0
            and context_level_rate >= 1.0
            and context_level_target_token <= 0
        ):
            if target_token < 0 and rate < 1.0:
                context_level_rate = (
                    (rate + 1.0) / 2 
                )  # 0.3
            if target_token >= 0:
                context_level_target_token = (  
                    target_token * 2 
                ) 
            
        if target_context >= 0: 
            context_level_rate = min(target_context / len(text), 1.0)
        
        if context_level_target_token >= 0: 
            context_level_rate = min(
                context_level_target_token / n_original_token, 1.
            )

        # context_decoder_words
        context_probs, context_words,context_decoder_words = self.__get_context_prob(
                context_chunked, 
                token_to_word=token_to_word,
                force_tokens=force_tokens,
                token_map=token_map,
                force_reserve_digit=force_reserve_digit,
                max_batch_size=8,
        )
        
        threshold = np.percentile(
                context_probs, int(100 * (1 - context_level_rate))
            )
        print(f'threshold: {threshold}')
      
        reserved_context = []
        context_label = [False] * len(context_probs) 
        for i, p in enumerate(context_probs):
            if p >= threshold or (
                    force_context_ids is not None and i in force_context_ids
                ):
                reserved_context.append(context_chunked[i]) 
                context_label[i] = True
        n_reserved_token = 0
        for chunks in reserved_context:
                for c in chunks:
                    n_reserved_token += self.get_token_length(c,tokenizer=self.tokenizer, use_oai_tokenizer=True)
        if target_token >= 0:
                rate = min(target_token / n_reserved_token, 1.0)
                
        if use_token_level_filter:
                compressed_context, word_list, word_label_list = self.__compress(
                    reserved_context,
                    reduce_rate=max(0, 1 - rate),
                    token_to_word=token_to_word,
                    force_tokens=force_tokens,
                    token_map=token_map,
                    force_reserve_digit=force_reserve_digit,
                    drop_consecutive=drop_consecutive,
                    max_batch_size=8,
                )
        else:
                compressed_context, word_list, word_label_list = self.__compress(
                    reserved_context,
                    reduce_rate=0,
                    token_to_word=token_to_word,
                    force_tokens=force_tokens,
                    token_map=token_map,
                    force_reserve_digit=force_reserve_digit,
                    drop_consecutive=drop_consecutive,
                    max_batch_size=8,
                )
                
        n_compressed_token = 0 
    
        
        
        for c in compressed_context:
            n_compressed_token += self.get_token_length(c, self.tokenizer,use_oai_tokenizer=True)
        saving = (n_original_token - n_compressed_token) * 0.06 / 1000
 
        ratio = (
            1 if n_compressed_token == 0 else n_original_token / n_compressed_token
        ) 
        res = {
            "compressed_prompt": "\n\n".join(compressed_context),
            "compressed_prompt_list": compressed_context,
            "origin_tokens": n_original_token,
            "compressed_tokens": n_compressed_token,
            "ratio": f"{ratio:.1f}x",
            "rate": f"{1 / ratio * 100:.1f}%",
            "saving": f", Saving ${saving:.1f} in GPT-4.",
        }
        return res

        
                
    def compress_aspect(
        self,
        text,
        target_token: int = -1,
        rate: float = 0.5,
        target_context: int = -1, 
        context_level_rate: float = 1.0,  
        context_level_target_token: int = -1,  
        token_to_word: str = "mean",
        force_tokens: List[str] = [],
        force_reserve_digit: bool = False,
        force_context_ids: List[int] = [],
        use_token_level_filter: bool = True,
        drop_consecutive: bool = False,
        return_word_label: bool = False
    ):
        token_map = {}

        for i, t in enumerate(force_tokens):
            if len(self.tokenizer.tokenize(t)) != 1:
                token_map[t] = self.added_tokens[i]
        
        chunk_end_tokens = []
        
        chunk_end_tokens = copy.deepcopy(chunk_end_tokens)
        for c in chunk_end_tokens:
            if c in token_map:
                chunk_end_tokens.append(token_map[c])
        chunk_end_tokens = set(chunk_end_tokens)
        
        
        if type(text) == str:
            text = [text]
            
        context_ = copy.deepcopy(text)
        
        n_original_token = 0
        context_chunked = []
        
        for i in range(len(context_)):
            n_original_token += self.get_token_length(
                context_[i], self.tokenizer, use_oai_tokenizer=True
            )
            context_chunked.append(
                self.__chunk_context(context_[i], self.chunk_end_tokens) 
            ) 
            
        if (
            target_context <= 0
            and context_level_rate >= 1.0
            and context_level_target_token <= 0
        ):
            if target_token < 0 and rate < 1.0:
                context_level_rate = (
                    (rate + 1.0) / 2 
                )  # 0.3
            if target_token >= 0:
                context_level_target_token = ( 
                    target_token * 2 
                ) 
            
        if target_context >= 0: 
            context_level_rate = min(target_context / len(text), 1.0)
        
        if context_level_target_token >= 0: 
            context_level_rate = min(
                context_level_target_token / n_original_token, 1.
            )
        
        
        context_probs, context_words = self.__get_context_prob(
                context_chunked, 
                token_to_word=token_to_word,
                force_tokens=force_tokens,
                token_map=token_map,
                force_reserve_digit=force_reserve_digit,
                max_batch_size=8,
        )
        
        threshold = np.percentile(
                context_probs, int(100 * (1 - context_level_rate))
            )
        print(f'threshold: {threshold}')
  
        reserved_context = [] 
        context_label = [False] * len(context_probs) 
        for i, p in enumerate(context_probs):
            if p >= threshold or (
                    force_context_ids is not None and i in force_context_ids
                ):
                reserved_context.append(context_chunked[i]) 
                context_label[i] = True 
        n_reserved_token = 0
        for chunks in reserved_context:
                for c in chunks:
                    n_reserved_token += self.get_token_length(c,tokenizer=self.tokenizer, use_oai_tokenizer=True)
        if target_token >= 0:
                rate = min(target_token / n_reserved_token, 1.0)
                
        if use_token_level_filter: 
                compressed_context, word_list, word_label_list = self.__compress(
                    reserved_context,
                    reduce_rate=max(0, 1 - rate),
                    token_to_word=token_to_word,
                    force_tokens=force_tokens,
                    token_map=token_map,
                    force_reserve_digit=force_reserve_digit,
                    drop_consecutive=drop_consecutive,
                    max_batch_size=8,
                )
        else:
                compressed_context, word_list, word_label_list = self.__compress(
                    reserved_context,
                    reduce_rate=0,
                    token_to_word=token_to_word,
                    force_tokens=force_tokens,
                    token_map=token_map,
                    force_reserve_digit=force_reserve_digit,
                    drop_consecutive=drop_consecutive,
                    max_batch_size=8,
                )
                
        n_compressed_token = 0 
    
        
        for c in compressed_context:
            n_compressed_token += self.get_token_length(c, self.tokenizer,use_oai_tokenizer=True)
        saving = (n_original_token - n_compressed_token) * 0.06 / 1000
        ratio = (
            1 if n_compressed_token == 0 else n_original_token / n_compressed_token
        ) 
        res = {
            "compressed_prompt": "\n\n".join(compressed_context),
            "compressed_prompt_list": compressed_context,
            "origin_tokens": n_original_token,
            "compressed_tokens": n_compressed_token,
            "ratio": f"{ratio:.1f}x",
            "rate": f"{1 / ratio * 100:.1f}%",
            "saving": f", Saving ${saving:.1f} in GPT-4.",
        }
        return res

        
                
                  
    def __get_context_prob(
            self,
            context_list: list,
            token_to_word="mean",
            force_tokens: List[str] = [],
            token_map: dict = {},
            force_reserve_digit: bool = False,
            max_batch_size: int = 4,
        ):

            chunk_list = []
            for chunks in context_list:
                for c in chunks:
                    sentences = sent_tokenize(c)
                    chunk_list.append(sentences)
                    

            dataset = TokenClfDataset(
                chunk_list, tokenizer=self.tokenizer, max_len=self.max_seq_len
            )
            dataloader = DataLoader(
                dataset, batch_size=max_batch_size, shuffle=False, drop_last=False
            )

            encoder_probs = []  
            encoder_words = []  
            decoder_words = []  
    
            with torch.no_grad():
                for batch in dataloader:
                    ids = batch["ids"].to(self.device, dtype=torch.long)
                    mask = batch["mask"].to(self.device, dtype=torch.long) == 1
                    
                    outputs = self.model(input_ids=ids, attention_mask=mask)
                    encoder_logits = outputs.logits

                    encoder_probs_tensor = F.softmax(encoder_logits, dim=-1) 
    
                    for j in range(ids.shape[0]):
                        encoder_active_probs = torch.masked_select(encoder_probs_tensor[j, :, 1], mask[j])
                        encoder_active_ids = torch.masked_select(ids[j], mask[j])

                        encoder_tokens = self.tokenizer.convert_ids_to_tokens(encoder_active_ids.squeeze().tolist())

                        encoder_token_probs = [prob for prob in encoder_active_probs.cpu().numpy()]

                        encoder_words_chunk, valid_token_probs, valid_token_probs_no_force = self.__merge_token_to_word(
                            encoder_tokens, encoder_token_probs, force_tokens=force_tokens, token_map=token_map, force_reserve_digit=force_reserve_digit
                        )

                        word_probs_no_force = self.__token_prob_to_word_prob(
                            valid_token_probs_no_force, convert_mode=token_to_word
                        )

                        if "roberta" in self.model_family:
                            for i in range(len(encoder_words_chunk)):
                                encoder_words_chunk[i] = encoder_words_chunk[i].lstrip("▁")                    
        

                        encoder_words.append(encoder_words_chunk) 
                        encoder_probs.append(word_probs_no_force)  
                        

    
            prev_idx = 0 
            context_encoder_probs = []
            context_encoder_words = []
            #context_decoder_words = []
            for chunk_list in context_list:
                n_chunk = len(chunk_list)
                context_encoder_probs.append([])
                context_encoder_words.append([])
                for i in range(n_chunk):
                    context_encoder_probs[-1].extend(encoder_probs[prev_idx + i]) 
                    context_encoder_words[-1].extend(encoder_words[prev_idx + i]) 
                prev_idx = prev_idx + n_chunk 
            context_probs = [sum(probs) / len(probs) for probs in context_encoder_probs]
       
            return context_probs, context_encoder_words
    def __merge_token_to_word(
        self, tokens, token_probs, force_tokens, token_map, force_reserve_digit
    ):
        
        words = [] 
        word_probs = [] 
        word_probs_no_force = []

        for token, prob in zip(tokens, token_probs):
            if token in self.special_tokens:
                continue
            # add a new word
            elif is_begin_of_new_word(token, self.model_family, force_tokens, token_map):
                pure_token = get_pure_token(token, self.model_family) 
                prob_no_force = prob
                if pure_token in force_tokens or pure_token in set(token_map.values()):
                    prob = 1.0
                token = replace_added_token(token, token_map)
                words.append(token)
                word_probs.append(
                    [
                        1.0
                        if force_reserve_digit and bool(re.search(r"\d", token))
                        else prob
                    ]
                )
                word_probs_no_force.append([prob_no_force])
            # concatenate with previous token
            else:
                pure_token = get_pure_token(token, self.model_family)
                words[-1] += pure_token
                word_probs[-1].append(
                    1.0
                    if force_reserve_digit and bool(re.search(r"\d", token))
                    else prob
                )
                word_probs_no_force[-1].append(prob_no_force)

        return words, word_probs, word_probs_no_force
    
    def __token_prob_to_word_prob(self, token_probs, convert_mode="mean"):
            if convert_mode == "mean":
                word_probs = [sum(p) / len(p) for p in token_probs]
            elif convert_mode == "first":
                word_probs = [p[0] for p in token_probs]
            else:
                raise NotImplementedError()

            return word_probs
        
        
                
    def __compress(
        self,
        context_list: list, 
        reduce_rate: float = 0.5,
        token_to_word: str = "mean", 
        force_tokens: List[str] = [], 
        token_map: dict = {}, 
        force_reserve_digit: bool = False,
        drop_consecutive: bool = False, 
        max_batch_size: int = 8
    ):
        def split_string_to_words(input_string):
            pattern = r'\b\w+\b|[<>=/!@#$%^&*()?":{}|\\`~;_+-]'
            result = re.findall(pattern, input_string)
            return result

        if reduce_rate <= 0:
            words, word_labels = [], []
            for i in range(len(context_list)):
                chunk_list = context_list[i] 
                chunk_words = [] 
                chunk_word_labels = []
                for j in range(len(chunk_list)):
                    # replace to original token
                    for ori_token, new_token in token_map.items():
                        chunk_list[j] = chunk_list[j].replace(new_token, ori_token)
                    ws = split_string_to_words(chunk_list[j])
                    chunk_words.extend(ws)
                    chunk_word_labels.extend([1 for _ in range(len(ws))])
                context_list[i] = "".join(chunk_list)
                words.append(chunk_words)
                word_labels.append(chunk_word_labels)
            return context_list, words, word_labels

        chunk_list = []
        for chunks in context_list:
            for c in chunks:
                chunk_list.append(c)

        dataset = TokenClfDataset(
            chunk_list, tokenizer=self.tokenizer, max_len=self.max_seq_len
        )
        dataloader = DataLoader(
            dataset, batch_size=max_batch_size, shuffle=False, drop_last=False
        )

        compressed_chunk_list = []
        word_list = []
        word_label_list = []
        with torch.no_grad():
            for batch in dataloader:
                ids = batch["ids"].to(self.device, dtype=torch.long)
                mask = batch["mask"].to(self.device, dtype=torch.long) == 1

                outputs = self.model(input_ids=ids, attention_mask=mask)
                encoder_logits = outputs.logits
                encoder_probs_tensor = F.softmax(encoder_logits, dim=-1)  
                
                #print(decoder_logits)
                for j in range(ids.shape[0]):
                    encoder_active_probs = torch.masked_select(encoder_probs_tensor[j, :, 1], mask[j])
                    encoder_active_ids = torch.masked_select(ids[j], mask[j])

                    encoder_tokens = self.tokenizer.convert_ids_to_tokens(encoder_active_ids.squeeze().tolist())

                    encoder_token_probs = [prob for prob in encoder_active_probs.cpu().numpy()]

                    words, valid_token_probs, _ = self.__merge_token_to_word(
                        tokens=encoder_tokens,
                        token_probs=encoder_token_probs,
                        force_tokens=force_tokens,
                        token_map=token_map,
                        force_reserve_digit=force_reserve_digit,
                    )
                    word_probs = self.__token_prob_to_word_prob(
                        valid_token_probs, convert_mode=token_to_word
                    )

                    if drop_consecutive:
                        threshold = np.percentile(word_probs, int(100 * reduce_rate))
                        is_token_between = False
                        prev = None
                        for i, (word, word_prob) in enumerate(zip(words, word_probs)):
                            if word in force_tokens:
                                if is_token_between:
                                    is_token_between = False
                                elif not is_token_between and word == prev:
                                    word_probs[i] = 0.0
                                prev = word
                            else:
                                is_token_between |= word_prob > threshold

                    new_token_probs = []
                    for word, word_prob in zip(words, word_probs):
                        num_token = len(self.oai_tokenizer.encode(word))
                        new_token_probs.extend([word_prob for _ in range(num_token)])
                    threshold = np.percentile(
                        new_token_probs, int(100 * reduce_rate + 1)
                    )

                    keep_words = []
                    word_labels = []
                    assert len(words) == len(word_probs)
                    for word, word_prob in zip(words, word_probs):
                        if word_prob > threshold or (
                            threshold == 1.0 and word_prob == threshold
                        ):
                            if (
                                drop_consecutive
                                and word in force_tokens
                                and len(keep_words) > 0
                                and keep_words[-1] == word
                            ):
                                word_labels.append(0)
                            else:
                                keep_words.append(word)
                                word_labels.append(1)
                        else:
                            word_labels.append(0)
                    keep_str = self.tokenizer.convert_tokens_to_string(keep_words)
                    if "xlm-roberta-large" in self.model_family:
                        for i in range(len(words)):
                            words[i] = words[i].lstrip("▁")

                    compressed_chunk_list.append(keep_str)
                    word_list.append(words[:])
                    word_label_list.append(word_labels[:])

        compressed_context_list = []
        original_word_list = []
        original_word_label_list = []
        prev_idx = 0
        for chunk_list in context_list:
            n_chunk = len(chunk_list)
            compressed_context_list.append(
                "".join(compressed_chunk_list[prev_idx : prev_idx + n_chunk])
            )
            original_word_list.append([])
            original_word_label_list.append([])
            for i in range(n_chunk):
                original_word_list[-1].extend(word_list[prev_idx + i])
                original_word_label_list[-1].extend(word_label_list[prev_idx + i])
            prev_idx = prev_idx + n_chunk

        return compressed_context_list, original_word_list, original_word_label_list
    