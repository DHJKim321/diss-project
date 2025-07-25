"""
Utility functions for the 'Augment' function in DivideMix.
"""

import os, sys
import regex as re
import random
from nltk.corpus import wordnet as wn, stopwords
from nltk import pos_tag
_STOP = set(stopwords.words("english"))
_tok_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_PENN_TO_WN = {'N': wn.NOUN, 'V': wn.VERB, 'J': wn.ADJ, 'R': wn.ADV}
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch, random

def mask_augment(input_ids, attention_mask, tokenizer, p=0.2):
        """
        Apply masking augmentation to the input_ids and attention_mask.
        """
        valid_pos = (attention_mask == 1).clone()
        valid_pos[0] = False  # CLS
        if valid_pos.sum() > 2:
                valid_pos[valid_pos.nonzero()[-1]] = False  # SEP

        # Random mask
        rand = torch.rand(input_ids.shape)
        mask = (rand < p) & valid_pos.bool()

        # Apply mask
        input_ids = input_ids.clone()
        input_ids[mask] = tokenizer.mask_token_id
        return input_ids, attention_mask

def _ids_to_tokens(input_ids, tokenizer, attention_mask):
    valid = attention_mask.bool().cpu()
    return tokenizer.convert_ids_to_tokens(input_ids[valid].tolist())

def _tokens_to_ids(tokens, tokenizer, seq_len, device):
    enc = tokenizer(tokens,
                    is_split_into_words=True,
                    padding='max_length',
                    truncation=True,
                    max_length=seq_len,
                    return_tensors='pt')
    return enc['input_ids'][0].to(device), enc['attention_mask'][0].to(device)

def _is_special(tok, tokenizer):
    return tok in {tokenizer.cls_token, tokenizer.sep_token,
                   tokenizer.pad_token, tokenizer.mask_token}

def _find_synonyms(word, pos=None):
    syns = set()
    synsets = wn.synsets(word, pos=pos) if pos else wn.synsets(word)
    for syn in synsets:
        for lemma in syn.lemmas():
            w = lemma.name().replace('_', ' ').lower()
            if w != word.lower():
                syns.add(w)
    return [s for s in syns if _tok_re.fullmatch(s)]

def _join_tokens(tokens):
    out, prev = [], ''
    for curr in tokens:
        if prev and prev[-1].isalnum() and curr.isalnum():
            out.append(' ')
        out.append(curr)
        prev = curr
    return ''.join(out)

def _preserve_case(src, repl):
    """Capitalize replacement if src was capitalized."""
    return repl.capitalize() if src and src[0].isupper() else repl

def delete_augment(input_ids, attention_mask, tokenizer, p=0.2):
    device, seq_len = input_ids.device, input_ids.size(0)
    tokens = _ids_to_tokens(input_ids, tokenizer, attention_mask)

    kept = []
    for t in tokens:
        if _is_special(t, tokenizer):
            kept.append(t)
        else:
            if random.random() > p:
                kept.append(t)

    if len(kept) <= 2:
        kept.append(tokens[1])

    new_ids, new_mask = _tokens_to_ids(kept, tokenizer, seq_len, device)
    return new_ids, new_mask

def synonym_augment(text, alpha=0.1, keep_pos=('N','V','J','R')):
    tokens = _tok_re.findall(text)
    word_positions = [(i, tok) for i, tok in enumerate(tokens) if tok.isalpha() and tok.lower() not in _STOP]
    if not word_positions:
        return text

    _, word_list = zip(*word_positions)
    tags = pos_tag(list(word_list))

    eligible = []
    for (i, tok), (_, tag) in zip(word_positions, tags):
        if tag and tag[0] in keep_pos:
            eligible.append((i, tok, tag[0]))

    # No eligible texts
    if not eligible:
        return text

    k = max(1, int(alpha * len(tokens))) # Per the paper
    chosen = random.sample(eligible, min(k, len(eligible)))

    for idx, orig_tok, penn_first in chosen:
        wn_pos = _PENN_TO_WN.get(penn_first)
        syns = _find_synonyms(orig_tok, wn_pos)
        if syns:
            rep = random.choice(syns)
            tokens[idx] = _preserve_case(orig_tok, rep)

    return _join_tokens(tokens)

PUNCTS = [".", ",", "!", "?", ";", ":"]

def aeda_augment(text, n=2):
    words = text.split()
    for _ in range(n):
        pos = random.randint(0, len(words))
        mark = random.choice(PUNCTS)
        words.insert(pos, mark)
    aug = " ".join(words)
    return aug

def augment_token(input_ids, attention_mask, tokenizer):
        if random.random() < 0.5:
            return mask_augment(input_ids, attention_mask, tokenizer)
        else:
            return delete_augment(input_ids, attention_mask, tokenizer)
        
def augment_text(text):
    temp = synonym_augment(text)
    return aeda_augment(temp)

def augment(text, input_ids, attention_mask, tokenizer):
    if random.random() < 1/3:
        text = augment_text(text)
        return text, None, None
    elif random.random() < 2/3:
        input_ids, attention_mask = augment_token(input_ids, attention_mask, tokenizer)
        return None, input_ids, attention_mask
    else:
        input_ids, attention_mask = mask_augment(input_ids, attention_mask, tokenizer)
        return None, input_ids, attention_mask