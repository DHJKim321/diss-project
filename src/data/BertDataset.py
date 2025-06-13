from torch.utils.data import Dataset
import pandas as pd
import torch

class BertDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = str(self.data.iloc[idx, 'text'])
        label = int(self.data.iloc[idx, 'label'])
        encoding = self.tokenizer(
            text,
            # padding='max_length',
            # truncation=True,
            # max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }