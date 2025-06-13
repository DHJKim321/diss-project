import torch, transformers
import time as tm
import os, sys
import pandas as pd
from tqdm import tqdm
from mistralai import Mistral

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from transformers import AutoTokenizer, AutoModelForCausalLM

def load_llama_model(model_path):
    start_ = tm.time()
    print(f"Using path: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    pipeline_ = transformers.pipeline(
        "text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16,
        device_map="auto", do_sample=False, return_full_text=False
    )
    print(f'Took {(tm.time()-start_)/60} minutes to load {model}')
    return pipeline_

def load_mistral_model(model_name, TOKEN, cache_path=None):
    start_ = tm.time()
    if cache_path:
        local_model_path = f"{cache_path}/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/3ad372fc79158a2148299e3318516c786aeded6c"
        print(f'Using cache path: {local_model_path}')
        tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(local_model_path, local_files_only=True)
    else:
        print('No cache path provided, using default model.')
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=TOKEN)
        model = AutoModelForCausalLM.from_pretrained(model_name, token=TOKEN)

    pipeline_ = transformers.pipeline(
        "text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16,
        device_map="auto", do_sample=False, return_full_text=False
    )
    print(f'Took {(tm.time()-start_)/60} minutes to load {model_name}')
    return pipeline_


def process_batch_of_prompts( pipeline, instruction, prompts, **kwargs):
    start_ = tm.time()
    full_prompts = [ instruction.format( task_content= prompt) for prompt in prompts]
    kwargs.update({
        'batch_size': 1, 'pad_token_id': pipeline.tokenizer.eos_token_id, 'eos_token_id': pipeline.tokenizer.eos_token_id
    })
    outputs = pipeline(full_prompts, **kwargs)
    outputs = [out[0]["generated_text"] for out in outputs]
    # print(f'Took {(tm.time() - start_)} seconds')
    return outputs

def batch_process( pipeline, instruction, df, new_col, num_posts, file_name, data_path= None, source_col= 'text'):
    if not (new_col in df.columns):
        df[ new_col] = None
    left_to_process = df.loc[ pd.isnull( df[ new_col])].index
    n_to_process = len( left_to_process)
    range_top = n_to_process // num_posts + 1
    print(f'Need to process {n_to_process} posts out of a {len(df)} dataframe.\n----')

    for i in tqdm(range(range_top), desc="Processing batches", unit="batch"):
        idx_ = left_to_process[ i* num_posts: (i+1)* num_posts]
        if len( idx_) == 0:
            print(f'Batch {i} and no more to process!')
            break
        out = process_batch_of_prompts( pipeline, instruction, df.loc[ idx_, source_col].tolist())
        out = [x.replace("\n", "") for x in out]
        df.loc[ idx_, new_col] = out
    if data_path is not None:
        path = data_path + "evaluated_" + file_name
        df.to_csv(path, index= False)
         
    return df 

def load_mistral_client(api_key):
    start_ = tm.time()
    client = Mistral(api_key=api_key)
    print(f'Took {(tm.time() - start_) / 60} minutes to load Mistral client.')
    return client

def mistral_process_batch_of_prompts(client, model_name, instruction_template, system_prompt, prompts):
    """
    prompts: list of Reddit posts (strings)
    """
    start_ = tm.time()
    outputs = []

    for prompt in prompts:
        user_prompt = instruction_template.format(
            task_content=prompt
        )

        response = client.chat.complete(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        outputs.append(response.choices[0].message.content.strip())
    return outputs

def mistral_batch_process(client, model_name, instruction_template, system_prompt, df, new_col, num_posts,
                          data_path=None, source_col='selftext'):
    if not (new_col in df.columns):
        df[new_col] = None

    left_to_process = df.loc[pd.isnull(df[new_col])].index
    n_to_process = len(left_to_process)
    range_top = n_to_process // num_posts + 1
    print(f'Need to process {n_to_process} posts out of a {len(df)} dataframe.\n----')

    for i in tqdm(range(range_top), desc="Processing batches", unit="batch"):
        idx_ = left_to_process[i * num_posts: (i + 1) * num_posts]
        if len(idx_) == 0:
            break

        prompts = df.loc[idx_, source_col].tolist()
        outputs = mistral_process_batch_of_prompts(
            client, model_name, instruction_template, 
            system_prompt,
            prompts, 
        )

        df.loc[idx_, new_col] = outputs

        # if data_path is not None:
        #     df.to_csv(data_path, index=False)
        #     print(f'--- Saved intermediate results ---')
    df.to_csv(data_path,index=False) if data_path is not None else None
    print(f'--- Finished processing batches ---')
    print(f'--- Saved final results to {data_path} ---') if data_path is not None else None
    return df