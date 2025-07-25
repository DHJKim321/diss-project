import pandas as pd
import numpy as np
import torch
import pickle
from tqdm.notebook import tqdm
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from transformers import MarianMTModel, MarianTokenizer

def load_translation_model(model_name):
    tok = MarianTokenizer.from_pretrained(model_name)
    mod = MarianMTModel.from_pretrained(model_name).to(device)
    mod.eval()
    return tok, mod

# English <-> Russian
tok_en2ru, mod_en2ru = load_translation_model("Helsinki-NLP/opus-mt-en-ru")
tok_ru2en, mod_ru2en = load_translation_model("Helsinki-NLP/opus-mt-ru-en")

# English <-> German
tok_en2de, mod_en2de = load_translation_model("Helsinki-NLP/opus-mt-en-de")
tok_de2en, mod_de2en = load_translation_model("Helsinki-NLP/opus-mt-de-en")

@torch.no_grad()
def hf_translate(texts, tok, mod, sampling=True, temperature=0.9):
    inputs = tok(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = mod.generate(
        **inputs,
        num_beams=1,
        do_sample=sampling,
        temperature=temperature,
        top_k=50,
        max_length=128
    )
    return tok.batch_decode(outputs, skip_special_tokens=True)

train_df = pd.read_csv('agnews_train.csv', header=None)
print(train_df.head())
train_labels = [v-1 for v in train_df[0]]
train_text = [v for v in train_df[2]]

def train_val_split(labels, n_labeled_per_class, n_labels, seed=0):
    np.random.seed(seed)
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(n_labels):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:n_labeled_per_class + 10000])
        val_idxs.extend(idxs[-3000:])

    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)
    return train_labeled_idxs, train_unlabeled_idxs, val_idxs

train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(train_labels, 500, 10)
idxs = train_unlabeled_idxs

def translate_ru(start, end, file_name):
    trans_result = {}
    for i in tqdm(range(start, end)):
        text = train_text[idxs[i]]
        # en -> ru -> en
        ru = hf_translate([text], tok_en2ru, mod_en2ru, sampling=True, temperature=0.9)[0]
        back = hf_translate([ru], tok_ru2en, mod_ru2en, sampling=True, temperature=0.9)[0]
        trans_result[idxs[i]] = back
        if i % 500 == 0:
            with open(file_name, 'wb') as f:
                pickle.dump(trans_result, f)
    with open(file_name, 'wb') as f:
        pickle.dump(trans_result, f)

def translate_de(start, end, file_name):
    trans_result = {}
    for i in tqdm(range(start, end)):
        text = train_text[idxs[i]]
        # en -> de -> en
        de = hf_translate([text], tok_en2de, mod_en2de, sampling=True, temperature=0.9)[0]
        back = hf_translate([de], tok_de2en, mod_de2en, sampling=True, temperature=0.9)[0]
        trans_result[idxs[i]] = back
        if i % 500 == 0:
            with open(file_name, 'wb') as f:
                pickle.dump(trans_result, f)
    with open(file_name, 'wb') as f:
        pickle.dump(trans_result, f)
translate_de(0, 100000, 'de_1.pkl')
translate_ru(0, 100000, 'ru_1.pkl')
