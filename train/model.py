from transformers import RobertaPreTrainedModel, AutoConfig, XLMRobertaForTokenClassification, XLMRobertaForCausalLM
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaLMHead
import torch
from torch.nn import CrossEntropyLoss,Linear
import  torch
import torch.nn as nn
import copy
import numpy as np  
from sklearn.metrics import accuracy_score
from transformers import AutoConfig, T5ForTokenClassification, AutoModelForSeq2SeqLM,T5ForConditionalGeneration, T5Config,Seq2SeqTrainer
import transformers
import torch.nn as nn
from transformers import T5ForConditionalGeneration, AutoConfig,T5PreTrainedModel
from transformers import Trainer

import torch
import torch.nn as nn
from transformers import XLMRobertaModel, GPT2LMHeadModel, GPT2Tokenizer


MAX_LEN = 512
MAX_GRAD_NORM = 10


"""
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids, labels, attention_mask, aspects = inputs
        # forward pass
        
        model = model.module if hasattr(model, 'module') else model 
        
        encoder_logits, decoder_logits, token_loss, aspect_loss = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels, 
            aspect_labels=aspects
        )
        
        # Compute the combined loss
        loss = 0.3*token_loss + 0.7*aspect_loss
        
        return (loss, encoder_logits, decoder_logits) if return_outputs else loss 
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask, aspects = inputs
        # forward passc
        with torch.no_grad():
            encoder_logits, decoder_logits, encoder_loss, decoder_loss = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels, 
                aspect_labels=aspects
            )
""" 

class RobertaGPT2 (RobertaPreTrainedModel):
    def __init__(self, encoder_model_name="xlm-roberta-base", decoder_model_name="gpt2",config=None):
        if config is None:
            config = AutoConfig.from_pretrained(encoder_model_name)
        config.output_attentions = True
        config.output_hidden_states = True
        config.is_decoder = True
        
        super(RobertaGPT2, self).__init__(config)
        self.num_labels = 2
        self.encoder = XLMRobertaForTokenClassification.from_pretrained(decoder_model_name, config=config)
        self.encoder.config.num_labels = self.num_labels


        # 디코더: GPT-2 (텍스트 생성)
        self.decoder = GPT2LMHeadModel.from_pretrained(decoder_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(decoder_model_name)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        
        # 필요한 특수 토큰 설정
        self.tokenizer.pad_token = self.tokenizer.eos_token  # padding 토큰을 eos로 설정
    
        self.text_generation_loss_fct = nn.CrossEntropyLoss()  # 텍스트 생성 손실 함수

    def forward(self, input_ids, attention_mask, decoder_input_ids=None, labels=None, aspect_labels=None):
        if decoder_input_ids is None:
            decoder_input_ids = input_ids
        # 나머지 로직
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        encoder_logits = encoder_outputs.logits
        encoder_loss = encoder_outputs.loss if labels is not None else None
        
        hidden_states = encoder_outputs.hidden_states
        last_hidden_states = hidden_states[-1] 

        # GPT-2 디코더에 인코더의 출력을 입력으로 사용하여 텍스트 생성
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, encoder_hidden_states=last_hidden_states, attention_mask=attention_mask, labels=aspect_labels)
        
        decoder_logits = decoder_outputs.logits
        decoder_loss = decoder_outputs.loss if labels is not None else None

        return encoder_logits, decoder_logits, encoder_loss, decoder_loss



class Roberta_CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        #print(input_ids.shape)  # 출력값이 (batch_size, seq_length)이어야 합니다.
        
        input_ids, labels, attention_mask, aspects = inputs
        # forward pass
        
        model = model.module if hasattr(model, 'module') else model 
        
        encoder_logits, decoder_logits, token_loss, aspect_loss = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels, 
            aspect_labels=aspects
        )
        
        print(f"token_loss: {token_loss}")
        print(f"aspect_loss: {aspect_loss}")
        # Compute the combined loss
        loss = 0.3*token_loss + 0.7*aspect_loss
        #aspect_loass 가중치 추가 5배정도 
        
        return (loss, encoder_logits, decoder_logits) if return_outputs else loss 
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask, aspects = inputs
        # forward passc
        with torch.no_grad():
            encoder_logits, decoder_logits, encoder_loss, decoder_loss = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels, 
                aspect_labels=aspects
            )
            
        loss = encoder_loss + decoder_loss if prediction_loss_only else None
        
        # Dynamically determine the number of labels
        num_labels = encoder_logits.size(-1)
        active_logits = encoder_logits.view(-1, num_labels)
        flattened_targets = labels.view(-1)
        flattened_predictions = torch.argmax(active_logits, axis=1)
        
        active_accuracy = attention_mask.view(-1) == 1
        targets = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
    

        return (loss, (predictions, targets), (labels, aspects)) if not prediction_loss_only else (loss, None, None)


class RobertaForMultitask(RobertaPreTrainedModel):
    def __init__(self, model_name, config=None):
        if config is None:
            config = AutoConfig.from_pretrained(model_name)
        #config.output_attentions = True
        #config.output_hidden_states = True
        #config.is_decoder = True
        
        super(RobertaForMultitask, self).__init__(config)
        self.num_labels = 2
        self.encoder = XLMRobertaForTokenClassification.from_pretrained(model_name, config=config)
        self.encoder.config.num_labels = self.num_labels
        
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Decoder for causal language modeling
        self.decoder = XLMRobertaForCausalLM.from_pretrained(model_name, config=config)
        self.lm_head = XLMRobertaLMHead(config)
        
    def forward(self, input_ids, attention_mask=None, labels=None, aspect_labels=None, return_dict=True):
        # Encoder forward pass
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=return_dict)
        
        # Extract logits and loss from encoder outputs
        sequence_output = encoder_outputs[0]

        sequence_output = self.dropout(sequence_output)
        encoder_logits = self.classifier(sequence_output)
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(encoder_logits.device)
            loss_fct = CrossEntropyLoss()
            encoder_loss = loss_fct(encoder_logits.view(-1, self.num_labels), labels.view(-1))
        
        # Extract the last hidden states
        hidden_states = encoder_outputs.hidden_states
        last_hidden_states = hidden_states[-1]  # The final layer's hidden states from the encoder
        
        # Directly pass the last hidden states to the decoder
        decoder_outputs = self.decoder(inputs_embeds=last_hidden_states, attention_mask=attention_mask, labels=aspect_labels, return_dict=return_dict)
        
        decoder_sequence_output = decoder_outputs[0]
        decoder_logits = self.lm_head(decoder_sequence_output)

        decoder_loss = None
        if aspect_labels is not None:
            # move labels to correct device to enable model parallelism
            labels = aspect_labels.to(decoder_logits.device)
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_decoder_logits = decoder_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            decoder_loss = loss_fct(decoder_logits.view(-1, self.config.vocab_size), labels.view(-1))
    

        return encoder_logits, decoder_logits, encoder_loss, decoder_loss




from transformers import T5ForConditionalGeneration, AutoConfig, T5PreTrainedModel
import torch.nn as nn
from torch.nn import CrossEntropyLoss

class T5ForMultiTaskLearning(T5PreTrainedModel):
    def __init__(self, model_name, config=None, num_labels=2):
        # config가 없다면, 모델을 사전 학습된 설정으로부터 로드하고 num_labels를 설정합니다.
        if config is None:
            config = AutoConfig.from_pretrained(model_name)
            config.output_hidden_states = True
            config.num_labels = num_labels  # num_labels를 config에 설정

        super(T5ForMultiTaskLearning, self).__init__(config)

        # T5ForConditionalGeneration 모델 로드 (인코더와 디코더 모두 포함됨)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)

        # 인코더 위에 분류(클래시피케이션) 헤드 추가
        self.dropout = nn.Dropout(config.dropout_rate if hasattr(config, 'dropout_rate') else 0.1)
        self.classifier = nn.Linear(config.d_model, config.num_labels)

        # lm_head는 T5ForConditionalGeneration 모델의 디코더 출력을 처리
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None, aspect_labels=None, return_dict=True):
        # T5 모델의 인코더 부분 forward (디코더는 호출하지 않음)
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict
        )

        hidden_states = encoder_outputs[0]
        hidden_states = self.dropout(hidden_states)

        # 인코더 출력에 분류 헤드 적용
        encoder_logits = self.classifier(hidden_states)

        # 디코더 forward
        decoder_outputs = self.model.decoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            return_dict=return_dict
        )

        sequence_output = decoder_outputs[0]
        decoder_logits = self.lm_head(sequence_output)

        # 인코더 loss 계산 (라벨이 주어졌을 때만)
        encoder_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            encoder_loss = loss_fct(encoder_logits.view(-1, self.config.num_labels), labels.view(-1))

        # 디코더 loss 계산 (aspect_labels이 주어졌을 때만)
        decoder_loss = None
        if aspect_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)  # -100은 무시
            decoder_loss = loss_fct(decoder_logits.view(-1, self.config.vocab_size), aspect_labels.view(-1))

        return encoder_logits, decoder_logits, encoder_loss, decoder_loss
    
    def generate_text(self, input_ids, attention_mask=None):
        # 텍스트 생성
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)
        
        # 토크나이저를 사용하여 ID를 텍스트로 디코딩
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return generated_text



class T5_CustomTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids, labels, attention_mask, aspects = inputs
        
        # 모델에서 예측 결과 및 손실 계산
        model = model.module if hasattr(model, 'module') else model

        # forward pass
        encoder_logits, decoder_logits, token_loss, aspect_loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,             # 토큰 분류 작업 레이블
            aspect_labels=aspects      # 이 부분에서 오타 수정: 'aspects_labels' -> 'aspect_labels'
        )

        loss = 0.3 * token_loss + 0.7 * (aspect_loss)
        #rint(f"token_loss: {token_loss}")
        #print(f"aspect_loss: {aspect_loss}")
        #print(f"loss: {loss}")
        return (loss, (encoder_logits, decoder_logits)) if return_outputs else loss

    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask, aspects = inputs

        model = model.module if hasattr(model, 'module') else model
        with torch.no_grad():
            encoder_logits, decoder_logits, encoder_loss, decoder_loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                labels=labels,  # 디코더 라벨 (aspect 추출용)
                aspect_labels=aspects  # 인코더 토큰 분류용 라벨
            )

        loss = encoder_loss + decoder_loss if prediction_loss_only else None

        # 토큰 분류 예측값 계산 (인코더)
        num_labels = encoder_logits.size(-1)
        active_logits = encoder_logits.contiguous().view(-1, num_labels)  # Apply .contiguous() before view
        flattened_targets = labels.contiguous().view(-1)  # Apply .contiguous() before view
        flattened_predictions = torch.argmax(active_logits, axis=1)

        # attention_mask를 사용해 유효한 토큰만 선택 (토큰 분류 정확도 계산)
        active_accuracy = attention_mask.view(-1) == 1
        targets = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        decoder_predictions = torch.argmax(decoder_logits, dim=-1)

        return (loss, (predictions, targets), (decoder_predictions, labels)) if not prediction_loss_only else (loss, None, None)


def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    attention_mask = [s[1] for s in samples]
    labels = [s[2] for s in samples]
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'labels': torch.stack(labels)
    }
