import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from torch.utils.data import Dataset
from src.utils.augment_utils import augment, augment_text

class DivideMixDataset(Dataset):
    
    def __init__(self, data, tokenizer, mode, preds=[], probs=[], max_length=256, augmentation='mask'):
        self.tokenizer = tokenizer
        self.mode = mode
        self.preds = preds
        self.probs = probs
        self.max_length = max_length

        text = data['text'].tolist()
        labels = data['label'].tolist()
        if self.mode == 'test':
            self.text = text
            self.labels = labels
            print(f"Test data size: {len(self.text)}")
            return
        if self.mode == 'all':
            self.text = text
            self.labels = labels
        elif self.mode == 'labelled':
            preds_idx = preds.nonzero()[0] # Samples with cluster membership probability > p_threshold
            self.text = [text[i] for i in preds_idx]
            self.labels = [labels[i] for i in preds_idx]
            self.probability = [probs[i] for i in preds_idx] # These would all be > p_threshold
            print(f"{self.mode} data size: {len(self.text)}")
        elif self.mode == 'unlabelled':
            preds_idx = (1-preds).nonzero()[0] # Samples with cluster membership probability <= p_threshold
            text = data['text'].tolist()
            self.text = [text[i] for i in preds_idx]
            print(f"{self.mode} data size: {len(self.text)}")
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Choose from 'all', 'labelled', or 'unlabelled'.")
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        """
        N.B. (mode == 'labelled' and mode == 'unlabelled'):
            We return two inputs and augment them (M = 2 in the Pseudocode).
        """
        if self.mode == 'all':
            encoding = self.tokenizer(
                self.text[index],
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(self.labels[index], dtype=torch.long),
                'index': index
            }
        elif self.mode == 'labelled':
            encoding1 = self.tokenizer(
                augment_text(self.text[index]),
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            encoding2 = self.tokenizer(
                augment_text(self.text[index]),
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids_1, attention_mask_1 = encoding1['input_ids'].squeeze(0), encoding1['attention_mask'].squeeze(0)
            # input_ids_1, attention_mask_1 = augment(input_ids_1.clone(), attention_mask_1.clone(), self.tokenizer)
            input_ids_2, attention_mask_2 = encoding2['input_ids'].squeeze(0), encoding2['attention_mask'].squeeze(0)
            # input_ids_2, attention_mask_2 = augment(input_ids_2.clone(), attention_mask_2.clone(), self.tokenizer)
            return {
                'input_ids_1': input_ids_1,
                'attention_mask_1': attention_mask_1,
                'input_ids_2': input_ids_2,
                'attention_mask_2': attention_mask_2,
                'labels': self.labels[index],
                'probability': self.probability[index]
            }
        elif self.mode == 'unlabelled':
            encoding1 = self.tokenizer(
                augment_text(self.text[index]),
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            encoding2 = self.tokenizer(
                augment_text(self.text[index]),
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids_1, attention_mask_1 = encoding1['input_ids'].squeeze(0), encoding1['attention_mask'].squeeze(0)
            # input_ids_1, attention_mask_1 = augment(input_ids_1.clone(), attention_mask_1.clone(), self.tokenizer)
            input_ids_2, attention_mask_2 = encoding2['input_ids'].squeeze(0), encoding2['attention_mask'].squeeze(0)
            # input_ids_2, attention_mask_2 = augment(input_ids_2.clone(), attention_mask_2.clone(), self.tokenizer)
            return {
                'input_ids_1': input_ids_1,
                'attention_mask_1':attention_mask_1,
                'input_ids_2': input_ids_2,
                'attention_mask_2': attention_mask_2
            }
        elif self.mode == 'test':
            encoding = self.tokenizer(
                self.text[index],
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(self.labels[index], dtype=torch.long)
            }