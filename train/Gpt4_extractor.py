from time import sleep
from utils import load_model_and_tokenizer
from openai import OpenAI

api_key = 'your_api'
client = OpenAI(api_key = api_key)

SLEEP_TIME_SUCCESS = 10
SLEEP_TIME_FAILED = 62

class Promptextractor: 
    def __init__(
        self,
        model_name, 
        com_user_prompt,
        asp_user_prompt,
        com_system_prompt=None,
        asp_system_prompt=None,
        instruct_prompt=None,
        temperature=0.3,
        top_p =1.0,
        n_max_token = 32700,
    ):
        
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        
        self.com_user_prompt = com_user_prompt
        self.com_system_prompt = com_system_prompt  
        
        self.asp_user_prompt = asp_user_prompt
        self.asp_system_prompt = asp_system_prompt

        self.instru_prompt = instruct_prompt
        self.model = model_name
        self.tokenizer = load_model_and_tokenizer(
            self.model_name, chat_completion=True
        )
        self.n_max_token = n_max_token  
    
    def query_template(self, text=None, n_max_new_token=4096):
        #compress
        if self.com_user_prompt and self.asp_user_prompt: 
            #user prompt가 있고 text_to_compress가 포함되어 있으면
            prompt = []
            #format 함수를 사용하여 text_to_compress를 text로 대체s
            prompt1 = self.com_user_prompt 
            prompt2 = self.asp_user_prompt
            instruct = self.instru_prompt.format(text_to_compress=text)
            instruct_no = self.instru_prompt 
            prompt.append({"role": "system", "content": prompt1})
            prompt.append({"role": "system", "content": prompt2})
            prompt.append({"role": "user", "content": instruct})
            
            len_prompt1 = len(self.tokenizer.encode(prompt1))
            len_prompt2 = len(self.tokenizer.encode(prompt2))
            len_prompt3 = len(self.tokenizer.encode(instruct_no))
            len_prompt = len_prompt1 + len_prompt2 + len_prompt3
        else:
            prompt = text
            
        len_sys_prompt = 0
        if self.com_system_prompt and self.asp_system_prompt:
           #system prompt가 있으면
            messages = []
            messages.append({"role": "system", "content": self.com_system_prompt})
            messages.append({"role": "system", "content": self.asp_system_prompt})
            
            len_com_sys_prompt = len(self.tokenizer.encode(self.com_system_prompt))
            len_asp_sys_prompt = len(self.tokenizer.encode(self.asp_system_prompt))
            len_sys_prompt = len_com_sys_prompt + len_asp_sys_prompt
            
        token_ids = self.tokenizer.encode(str(prompt))
        
        if len(token_ids) > (self.n_max_token - n_max_new_token - len_sys_prompt):
            half = int((self.n_max_token - n_max_new_token - len_sys_prompt) / 2) - 1
            prompt = self.tokenizer.decode(token_ids[:half] ) + self.tokenizer.decode(
                token_ids[-half:]
            )
        messages.extend(prompt)
        
        len_messages = len_prompt + len_sys_prompt
        
        return messages,len_messages
    
    def compress(self, text, n_max_new_token=4096):
        messages,len_messages = self.query_template(text,n_max_new_token)
        ext = None
        while ext is None:
            try:
                response = client.chat.completions.create(
                    model = self.model,
                    messages=messages,
                    temperature = self.temperature,
                    top_p =self.top_p,
                    max_tokens =n_max_new_token,
                )
                ext = response.choices[0].message.content
            except Exception as e:
                print(e)
                sleep(SLEEP_TIME_FAILED)
            
            return ext
                
                
        
        
        
        
            
        