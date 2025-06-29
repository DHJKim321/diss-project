import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.DivideMixDataset import DivideMixDataset
from torch.utils.data import DataLoader

class DivideMixDataloader():
    def __init__(self, batch_size, tokenizer, num_workers=4):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer

    def run(self, data, mode, preds=[], probs=[], augmentation='mask'):
        if mode == 'warmup':
            all_dataset = DivideMixDataset(
                data=data,
                tokenizer=self.tokenizer,
                mode='all',
                preds=preds,
                probs=probs
            )
            loader = DataLoader(
                all_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True
            )
            return loader
        elif mode == 'train':
            labelled_dataset = DivideMixDataset(
                data=data,
                tokenizer=self.tokenizer,
                mode='labelled',
                preds=preds,
                probs=probs,
                augmentation=augmentation
            )
            unlabelled_dataset = DivideMixDataset(
                data=data,
                tokenizer=self.tokenizer,
                mode='unlabelled',
                preds=preds,
                augmentation=augmentation
            )
            labelled_loader = DataLoader(
                labelled_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True
            )
            unlabelled_loader = DataLoader(
                unlabelled_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True
            )
            return labelled_loader, unlabelled_loader
        elif mode == 'eval_train':
            eval_dataset = DivideMixDataset(
                data=data,
                tokenizer=self.tokenizer,
                mode='all',
            )
            loader = DataLoader(
                eval_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False
            )
            return loader
        elif mode == 'test':
            test_dataset = DivideMixDataset(
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
            raise ValueError(f"Invalid mode: {mode}. Choose from 'warmup', 'train', 'eval_train', or 'test'.")