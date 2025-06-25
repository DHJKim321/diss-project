import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from DivideMixDataset import DivideMixDataset
from torch.utils.data import DataLoader

class DivideMixDataloader():
    def __init__(self, batch_size, collator, num_workers=4):
        self.batch_size = batch_size
        self.collator = collator
        self.num_workers = num_workers

    def run(self, data, mode, tokenizer, preds=[], probs=[]):
        if mode == 'warmup':
            all_dataset = DivideMixDataset(
                data=data,
                tokenizer=tokenizer,
                mode='all',
                preds=preds,
                probs=probs
            )
            loader = DataLoader(
                all_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collator,
                num_workers=self.num_workers,
                shuffle=True
            )
            return loader
        elif mode == 'train':
            labelled_dataset = DivideMixDataset(
                data=data,
                tokenizer=tokenizer,
                mode='labelled',
                preds=preds,
                probs=probs
            )
            unlabelled_dataset = DivideMixDataset(
                data=data,
                tokenizer=tokenizer,
                mode='unlabelled',
                preds=preds
            )
            labelled_loader = DataLoader(
                labelled_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collator,
                num_workers=self.num_workers,
                shuffle=True
            )
            unlabelled_loader = DataLoader(
                unlabelled_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collator,
                num_workers=self.num_workers,
                shuffle=True
            )
            return labelled_loader, unlabelled_loader
        elif mode == 'eval':
            eval_dataset = DivideMixDataset(
                data=data,
                tokenizer=tokenizer,
                mode='all',
            )
            loader = DataLoader(
                eval_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collator,
                num_workers=self.num_workers,
                shuffle=False
            )
            return loader
        elif mode == 'test':
            test_dataset = DivideMixDataset(
                data=data,
                tokenizer=tokenizer,
                mode='test'
            )
            loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collator,
                num_workers=self.num_workers,
                shuffle=False
            )
            return loader
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose from 'warmup', 'train', 'eval', or 'test'.")