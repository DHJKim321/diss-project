"""
Utility functions for the 'Augment' function in DivideMix.
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch

def mask_augment(input_ids, attention_mask, tokenizer, p=0.15):
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

# Ideas:
# Insert/Delete/Substitute random tokens in input_ids
# Randomly shuffle a subset of tokens in input_ids
