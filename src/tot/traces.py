import sqlite3
import hashlib
import time 
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import OpenAI, AsyncOpenAI
import sqlite3
import json
import hashlib
from dataclasses import dataclass
from typing import List
import pickle as pkl

import openai 
import asyncio


@dataclass
class Trace:
    messages: List[dict]
    tools: List[dict]
    response: dict
    start_time: float
    end_time: float
    elapsed_time: float
    openai_processing_time: float
    input_tokens: int
    output_tokens: int
    parallel: bool

class CachedOpenAI:
    def __init__(self, name, use_cache = True, verbose = False):
        self.name = name
        self.use_cache = use_cache
        self.verbose = verbose 
        
    def __enter__(self) -> 'CachedAPICaller':
        self.caller = CachedAPICaller(self.name, self.use_cache, verbose = self.verbose)
        return self.caller
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.caller.close()

class CachedAPICaller:
    def __init__(self, name, use_cache = True, verbose = False, async_mode = False):
        if async_mode:
            self.client = AsyncOpenAI()
        else: 
            self.client = OpenAI()
            
        self.conn = sqlite3.connect('cache.db')
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS cache (
            hash TEXT PRIMARY KEY,
            elapsed_time REAL, 
            openai_processing_time REAL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            response TEXT
        )''')
            
        self.conn.commit()
        self.traces: list[Trace] = []
        self.name: str = name 
        self.use_cache: bool = use_cache
        self.verbose: bool = verbose
        self.query_count = {}
        
    def display_cache(self):
        c = self.conn.cursor()
        c.execute("SELECT * FROM cache")
        self.conn.commit()
        for row in c.fetchall():
            print(row)
    
    def close(self):
        self.conn.close()
        with open(f"{self.name}_traces.pkl", "wb") as f:
            pkl.dump(self.traces, f)
        print(f"Traces saved to {self.name}_traces.pkl")

    def hash_messages(self, messages, tools = None):
        inputs = {'messages': messages, 'tools': tools}
        messages_str = json.dumps(inputs, sort_keys=True)
        message_hash = hashlib.sha256(messages_str.encode('utf-8')).hexdigest()
        return message_hash
    
    def serialize_chat_completion_response(self, response):
        '''
        ChatCompletionMessage(content='This is a test. How can I assist you today?', refusal=None, role='assistant', function_call=None, tool_calls=None)
        '''
        print('response type', type(response))
        return json.dumps(response.model_dump())
    
    def deserialize_chat_completion_response(self, response: str):
        response_dict = json.loads(response)
        return openai.types.chat.chat_completion.ChatCompletion.model_construct(**response_dict)
    
    async def chat_completion_request_async(self, 
                                messages, 
                                tools = None, 
                                tool_choices = None, 
                                model = "gpt-3.5-turbo",
                                n = 1,
                                **kwargs) -> openai.types.chat.chat_completion.ChatCompletion:
        message_hash = self.hash_messages(messages, tools)
        query_count = self.query_count[message_hash] = self.query_count.get(message_hash, 0) + n
        c = self.conn.cursor()
        
        if self.verbose:
            print('query count', query_count)
            print('n', n)
        
        result = None
        c.execute("SELECT * FROM cache WHERE hash = ?", (message_hash,))
        result = c.fetchone()
        
        start_time = time.perf_counter()
        
        suc = False
        prior_choices = []
        n_new_query = n
        if result: # fetched from cache
            if self.verbose:
                print("Cache hit")
                
            hash, elapsed_time, openai_processing_time, input_tokens, output_tokens, response = result
            
            await asyncio.sleep(elapsed_time)
            
            response = self.deserialize_chat_completion_response(response)
            
            prior_choices = response.choices

        n_new_query = query_count - len(prior_choices) if self.use_cache else n
        
        if n_new_query <= 0: suc = True
        else: 
            if self.verbose:
                print("Cache miss")
            for _ in range(1):
                # try:
                t = time.perf_counter()
                response = await self.client.chat.completions.with_raw_response.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choices,
                    n=n_new_query,
                    **kwargs
                )
                elapsed_time = (time.perf_counter() - t)
                openai_processing_time = response.headers.get('openai-processing-ms')
                input_tokens = response.headers.get('prompt_tokens')
                output_tokens = response.headers.get('completion_tokens')
                
                response = response.parse()
                response.choices = prior_choices + response.choices
                    
                # except Exception as e:
                #     print("Unable to generate ChatCompletion response")
                #     print(f"Exception: {e}")
                #     continue 
                            
                c.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?, ?, ?)", 
                            (message_hash, 
                            elapsed_time, 
                            openai_processing_time, 
                            input_tokens, 
                            output_tokens, 
                            self.serialize_chat_completion_response(response)))
                self.conn.commit()
                suc = True
                break 
                # except Exception as e:
                #     print("Unable to generate ChatCompletion response")
                #     print(f"Exception: {e}")
        if not suc: 
            raise Exception("Unable to generate ChatCompletion response")
        
        if self.verbose: 
            print("response", response)
        
        # assert len(response.choices) >= query_count
        response.choices = response.choices[max(query_count - n, 0): min(query_count, len(response.choices))]
        print('len(response.choices)', len(response.choices))
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) * 1000
        self.traces.append(Trace(
            messages, 
            tools, 
            response, 
            start_time, 
            end_time, 
            elapsed_time, 
            float(openai_processing_time), 
            input_tokens, output_tokens, True))
        
        return response
    
    def chat_completion_request(self, 
                                messages, 
                                tools = None, 
                                tool_choices = None, 
                                model = "gpt-3.5-turbo",
                                n = 1,
                                **kwargs) -> openai.types.chat.chat_completion.ChatCompletion:
        message_hash = self.hash_messages(messages, tools)
        query_count = self.query_count[message_hash] = self.query_count.get(message_hash, 0) + n
        c = self.conn.cursor()
        
        result = None
        c.execute("SELECT * FROM cache WHERE hash = ?", (message_hash,))
        result = c.fetchone()
        
        start_time = time.perf_counter()
        
        suc = False
        prior_choices = []
        n_new_query = n
        if result: # fetched from cache
            if self.verbose:
                print("Cache hit")
                
            hash, elapsed_time, openai_processing_time, input_tokens, output_tokens, response = result
            
            time.sleep(elapsed_time)
            
            response = self.deserialize_chat_completion_response(response)
            
            prior_choices = response.choices

        n_new_query = query_count - len(prior_choices) if self.use_cache else n
        
        if n_new_query <= 0: suc = True
        else: 
            if self.verbose:
                print("Cache miss")
            for _ in range(2):
                try:
                    t = time.perf_counter()
                    response = asyncio.run(self.client.chat.completions.with_raw_response.create(
                        model=model,
                        messages=messages,
                        tools=tools,
                        tool_choice=tool_choices,
                        n=n_new_query,
                        **kwargs
                    ))
                    elapsed_time = (time.perf_counter() - t)
                    openai_processing_time = response.headers.get('openai-processing-ms')
                    input_tokens = response.headers.get('prompt_tokens')
                    output_tokens = response.headers.get('completion_tokens')
                    
                    response = response.parse()
                    response.choices = prior_choices + response.choices
                    
                except Exception as e:
                    print("Unable to generate ChatCompletion response")
                    print(f"Exception: {e}")
                    continue 
                            
                c.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?, ?, ?)", 
                            (message_hash, 
                            elapsed_time, 
                            openai_processing_time, 
                            input_tokens, 
                            output_tokens, 
                            self.serialize_chat_completion_response(response)))
                self.conn.commit()
                suc = True
                break 
                # except Exception as e:
                #     print("Unable to generate ChatCompletion response")
                #     print(f"Exception: {e}")
        if not suc: 
            raise Exception("Unable to generate ChatCompletion response")
        
        if self.verbose: 
            print("response", response)
        
        assert len(response.choices) >= query_count
        response.choices = response.choices[query_count - n: query_count]
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) * 1000
        self.traces.append(Trace(
            messages, 
            tools, 
            response, 
            start_time, 
            end_time, 
            elapsed_time, 
            float(openai_processing_time), 
            input_tokens, output_tokens, False))
        
        return response

caller: CachedAPICaller = None

def init_caller(name, verbose = False, async_mode = False, use_cache = True):
    global caller
    caller = CachedAPICaller(name, verbose = verbose, async_mode = async_mode, use_cache=use_cache)
    return caller


VERBOSE = True

if __name__ == '__main__':
    caller = CachedAPICaller("cached_caller", verbose = True)
    messages = [{
        "role": "user",
        "content": "Say this is a test",
    }]
    response = caller.chat_completion_request(messages)
    
    print(response)
    
    response = caller.chat_completion_request(messages)
    
    print(response)