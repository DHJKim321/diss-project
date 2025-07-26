import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from torch.utils.data import Dataset
from src.utils.augment_utils import augment_text
import pickle

class Translator:
    def __init__(self, path):
        # Pre-processed German data
        with open(path + '/de_1.pkl', 'rb') as f:
            self.de = pickle.load(f)
        # Pre-processed Russian data
        with open(path + '/ru_1.pkl', 'rb') as f:
            self.ru = pickle.load(f)

    def __call__(self, ori, idx):
        out1 = self.de[idx]
        out2 = self.ru[idx]
        return out1, out2, ori

class MixTextDataset(Dataset):
    def __init__(self, data, tokenizer, mode, pickle_path, max_length=512):
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
        self.text = data['text'].tolist()
        self.labels = data['label'].tolist()
        self.translator = Translator(pickle_path)

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        if self.mode == 'labelled':
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
        elif self.mode == 'unlabelled':
            # Original
            encoding = self.tokenizer(
                self.text[index],
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            # Augmented
            augmented_text1 = self.de[index]
            encoding_augmented1 = self.tokenizer(
                augmented_text1,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            augmented_text2 = self.ru[index]
            encoding_augmented2 = self.tokenizer(
                augmented_text2,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids_orig': encoding['input_ids'].squeeze(0),
                'attention_mask_orig': encoding['attention_mask'].squeeze(0),
                'input_ids_aug_1': encoding_augmented1['input_ids'].squeeze(0),
                'attention_mask_aug_1': encoding_augmented1['attention_mask'].squeeze(0),
                'input_ids_aug_2': encoding_augmented2['input_ids'].squeeze(0),
                'attention_mask_aug_2': encoding_augmented2['attention_mask'].squeeze(0)
            }
        elif self.mode == 'val':
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
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")