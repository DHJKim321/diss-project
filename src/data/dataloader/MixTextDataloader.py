import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.dataset.MixTextDataset import MixTextDataset
from torch.utils.data import DataLoader

class MixTextDataLoader():
    def __init__(self, batch_size, tokenizer, num_workers=4):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer

    def run(self, data, mode):
        if mode == 'labelled':
            labelled_dataset = MixTextDataset(
                data=data,
                tokenizer=self.tokenizer,
                mode='labelled',
            )
            loader = DataLoader(
                labelled_dataset,
                batch_size=self.batch_size//2,
                num_workers=self.num_workers,
                shuffle=True
            )
            return loader
        elif mode == 'unlabelled':
            unlabelled_dataset = MixTextDataset(
                data=data,
                tokenizer=self.tokenizer,
                mode='unlabelled',
            )
            loader = DataLoader(
                unlabelled_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True
            )
            return loader
        elif mode == 'val':
            val_dataset = MixTextDataset(
                data=data,
                tokenizer=self.tokenizer,
                mode='val'
            )
            loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False
            )
            return loader
        elif mode == 'test':
            test_dataset = MixTextDataset(
                data=data,
                tokenizer=self.tokenizer,
                mode='test'
            )
            loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False
            )
            return loader
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose from 'labelled', 'unlabelled', 'val', or 'test'.")