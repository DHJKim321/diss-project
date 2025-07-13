"""
Utility functions for the 'Augment' function in DivideMix.
"""

import os, sys
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

def augment(input_ids, attention_mask, tokenizer):
        if random.random() < 0.5:
                return mask_augment(input_ids, attention_mask, tokenizer)
        else:
                return delete_augment(input_ids, attention_mask, tokenizer)