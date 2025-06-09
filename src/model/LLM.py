import torch, transformers
import time as tm
import os, sys
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from transformers import AutoTokenizer


def load_model( model_name, TOKEN):
    start_ = tm.time()
    tokenizer = AutoTokenizer.from_pretrained( model_name, padding_side="left", token= TOKEN)
    pipeline_ = transformers.pipeline( 
        "text-generation", model= model_name, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map="auto",
        do_sample=False, return_full_text=False, token= TOKEN
    )
    print(f'Took {(tm.time()-start_)/60} minutes to load {model_name}')
    return pipeline_

def process_batch_of_prompts( pipeline, instruction, prompts, max_tokens, **kwargs):
    start_ = tm.time()
    full_prompts = [ instruction.format( task_content= prompt) for prompt in prompts]
    kwargs.update({
        'batch_size': 1, 'max_new_tokens': max_tokens, 'pad_token_id': pipeline.tokenizer.eos_token_id, 'eos_token_id': pipeline.tokenizer.eos_token_id
    })
    outputs = pipeline(full_prompts, **kwargs)
    outputs = [out[0]["generated_text"] for out in outputs]
    print(f'Took {(tm.time() - start_) / 60} minutes to go through {len(prompts)} posts.')
    return outputs

def batch_process( pipeline, instruction, df, new_col, num_posts, max_tokens= 8, data_path= None, source_col= 'selftext'):
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
        out = process_batch_of_prompts( pipeline, instruction, df.loc[ idx_, source_col].tolist(), max_tokens)
        df.loc[ idx_, new_col] = out
        if data_path is not None:
            df.to_csv( data_path, index= False)
            print(f'--- Saved intermediate results ---') 
         
    return df 