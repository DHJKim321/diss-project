"""
Utility functions for the 'Augment' function in DivideMix.
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch

def mask_augment(self, input_ids, attention_mask, p=0.15):
        """
        Apply masking augmentation to the input_ids and attention_mask.
        """
        input_ids = input_ids.clone()
        attention_mask = attention_mask.clone()

        # For each input, randomly select tokens to mask with probability p
        for i in range(input_ids.size(0)):
            valid_input_ids = input_ids[i][attention_mask[i] == 1]
            mask_indices = (torch.rand(valid_input_ids.size(0)) < p) & (attention_mask[i] == 1)
            input_ids[i][mask_indices] = self.tokenizer.mask_token_id
        return input_ids, attention_mask

# Ideas:
# Insert/Delete/Substitute random tokens in input_ids
# Randomly shuffle a subset of tokens in input_ids
