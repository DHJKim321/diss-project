import argparse, os, random, math, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
import torch.nn.functional as F
from src.utils.data_utils import *
from torch.utils.data import Dataset, DataLoader
from transformers import (BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# --------------------- Repro ---------------------
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# --------------------- Noise ---------------------
def inject_symmetric_noise(df, label_col, noise_ratio, num_classes, rng):
    """
    Flip each label with probability noise_ratio to a *different* random class.
    Returns: noisy_df, flip_mask (True where label was flipped)
    """
    noisy = df.copy()
    flips = rng.rand(len(df)) < noise_ratio
    orig  = df[label_col].to_numpy()
    new   = orig.copy()
    # draw from 0..num_classes-2, then shift to skip orig
    r = rng.randint(0, num_classes - 1, size=flips.sum())
    new_vals = (r + (r >= orig[flips])).astype(int)
    new[flips] = new_vals
    noisy[label_col] = new
    return noisy, flips

# --------------------- Dataset ---------------------
class TxtDataset(Dataset):
    def __init__(self, df, tokenizer, text_col, label_col, max_len=128):
        self.texts  = df[text_col].tolist()
        self.labels = df[label_col].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        item["index"]  = torch.tensor(idx, dtype=torch.long)
        return item

def collate(batch):
    out = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch])
    return out

# --------------------- Train / Eval ---------------------
def run_epoch(model, loader, optimizer=None, scheduler=None, device='cuda'):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, total = 0.0, 0
    preds, gts = [], []
    for batch in tqdm(loader, disable=not train_mode):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        token_type_ids = batch.get('token_type_ids')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        with torch.set_grad_enabled(train_mode):
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels)
            loss = out.loss
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler: scheduler.step()

        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        preds.extend(out.logits.argmax(1).detach().cpu().tolist())
        gts.extend(labels.detach().cpu().tolist())

    return total_loss/total, accuracy_score(gts, preds), f1_score(gts, preds, average='weighted')

def per_sample_loss(model, loader, device='cuda'):
    model.eval()
    n = len(loader.dataset)
    losses = torch.zeros(n)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Per-sample CE"):
            idx = batch['index']
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            token_type_ids = batch.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            logits = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids).logits
            ce = F.cross_entropy(logits, labels, reduction='none')
            losses[idx] = ce.cpu()
    return losses.numpy()

# --------------------- Plot ---------------------
def plot_hist(losses, flip_mask, out_path, title):
    clean = losses[~flip_mask]
    noisy = losses[flip_mask]

    plt.figure(figsize=(8,5))
    bins = 60
    plt.hist(clean, bins=bins, alpha=0.6, label='loss on correct labels', color='green')
    plt.hist(noisy, bins=bins, alpha=0.6, label='loss on wrong labels',   color='red')
    plt.xlabel("loss"); plt.ylabel("frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved histogram to {out_path}")

# --------------------- Main ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--text_col', default='text')
    ap.add_argument('--label_col', default='label')
    ap.add_argument('--noise_ratio', type=float, default=0.7)
    ap.add_argument('--val_size', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--patience', type=int, default=2)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--max_len', type=int, default=128)
    ap.add_argument('--model_name', default='bert-base-uncased')
    ap.add_argument('--out_png', default='loss_hist.png')
    args = ap.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Load data ---
    if args.dataset == 'agnews':
        df = load_agnews_train("agnews_train.csv", "src/data/train")
    elif args.dataset == 'yahoo':
        df = load_yahoo_train("yahoo_train.csv", "src/data/train")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # --- Split BEFORE noise (paper used noisy val; toggle if you want clean) ---
    train_df, val_df = train_test_split(df, test_size=args.val_size,
                                        stratify=df[args.label_col], random_state=args.seed)
    rng = np.random.RandomState(args.seed)
    train_noisy, flip_mask = inject_symmetric_noise(train_df, args.label_col, args.noise_ratio, num_classes, rng)
    val_noisy, _           = inject_symmetric_noise(val_df,   args.label_col, args.noise_ratio, num_classes, rng)

    # --- Tokenizer & datasets ---
    tok = BertTokenizer.from_pretrained(args.model_name)
    d_train = TxtDataset(train_noisy, tok, args.text_col, args.label_col, args.max_len)
    d_val   = TxtDataset(val_noisy,   tok, args.text_col, args.label_col, args.max_len)
    d_eval_train = TxtDataset(train_noisy, tok, args.text_col, args.label_col, args.max_len)

    L_train = DataLoader(d_train, batch_size=args.batch_size, shuffle=True,  collate_fn=collate, num_workers=4)
    L_val   = DataLoader(d_val,   batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=4)
    L_eval_train = DataLoader(d_eval_train, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=4)

    # --- Model ---
    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=num_classes).to(device)

    # --- Optimizer & schedule ---
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = math.ceil(len(L_train)) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # --- Train with early stopping on noisy val ---
    best_val_acc, best_state, best_epoch = -1, None, -1
    bad = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch}")
        tr_loss, tr_acc, tr_f1 = run_epoch(model, L_train, optimizer, scheduler, device)
        val_loss, val_acc, val_f1 = run_epoch(model, L_val, optimizer=None, scheduler=None, device=device)
        print(f"  Train: loss {tr_loss:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f}")
        print(f"  Val  : loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping triggered.")
                break

    # --- Restore best ---
    assert best_state is not None, "No best state saved—check training loop."
    model.load_state_dict(best_state)
    model.to(device)
    print(f"Using checkpoint from epoch {best_epoch} (val acc {best_val_acc:.4f})")

    # --- Per-sample CE on TRAIN ---
    losses = per_sample_loss(model, L_eval_train, device=device)

    # --- Plot ---
    plot_hist(losses, flip_mask, args.out_png,
              f"Loss histogram at early stop (noise={args.noise_ratio})")


if __name__ == "__main__":
    main()