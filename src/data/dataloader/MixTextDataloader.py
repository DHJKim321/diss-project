import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.dataset.MixTextDataset import MixTextDataset
from torch.utils.data import DataLoader

class MixTextDataLoader():
    def __init__(self, batch_size_x, batch_size_u, tokenizer, pickle_path, num_workers=4):
        self.batch_size_x = batch_size_x
        self.batch_size_u = batch_size_u
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.pickle_path = pickle_path

    def run(self, data, mode, indices=None):
        if mode == 'labelled':
            labelled_dataset = MixTextDataset(
                data=data,
                tokenizer=self.tokenizer,
                mode='labelled',
                pickle_path=self.pickle_path
            )
            loader = DataLoader(
                labelled_dataset,
                batch_size=self.batch_size_x,
                num_workers=self.num_workers,
                shuffle=True
            )
            return loader
        elif mode == 'unlabelled':
            unlabelled_dataset = MixTextDataset(
                data=data,
                indices=indices,
                tokenizer=self.tokenizer,
                mode='unlabelled',
                pickle_path=self.pickle_path
            )
            loader = DataLoader(
                unlabelled_dataset,
                batch_size=self.batch_size_u,
                num_workers=self.num_workers,
                shuffle=True
            )
            return loader
        elif mode == 'val':
            val_dataset = MixTextDataset(
                data=data,
                tokenizer=self.tokenizer,
                mode='val',
                pickle_path=self.pickle_path
            )
            loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size_u,
                num_workers=self.num_workers,
                shuffle=False
            )
            return loader
        elif mode == 'test':
            test_dataset = MixTextDataset(
                data=data,
                tokenizer=self.tokenizer,
                mode='test',
                pickle_path=self.pickle_path
            )
            loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size_u,
                num_workers=self.num_workers,
                shuffle=False
            )
            return loader
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose from 'labelled', 'unlabelled', 'val', or 'test'.")