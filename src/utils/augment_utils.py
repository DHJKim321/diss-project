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
        padding_length = int(attention_mask.sum().item())
        # Only get valid token IDs (i.e., disregard first and last tokens - CLS and SEP)
        valid_input_ids = input_ids[attention_mask == 0][1:-1]

        # Uniformly distributed between [0, 1]
        mask = torch.rand(valid_input_ids.shape) < p
        # Turn mask to same shape as input_ids
        mask = torch.cat([torch.tensor([False]), mask, torch.tensor([False])])
        # Add padding to the mask
        mask = torch.cat([mask, torch.tensor([True] * padding_length)])
        # mask = torch.tensor([False]) + mask + torch.tensor([False]) + torch.tensor([False] * (padding_length - len(mask)))
        print(mask.shape, input_ids.shape)
        assert mask.shape == input_ids.shape

        attention_mask[mask] = 1
        input_ids[mask] = tokenizer.mask_token_id
        return input_ids, attention_mask

# Ideas:
# Insert/Delete/Substitute random tokens in input_ids
# Randomly shuffle a subset of tokens in input_ids
