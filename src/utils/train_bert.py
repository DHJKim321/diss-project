import torch, numpy as np, os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, BertModel, DataCollatorWithPadding
from torch.optim import AdamW
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.data_utils import load_full_data
from src.data.BertDataset import BertDataset
from src.model.bert import Bert
from src.utils.eval_utils import evaluate_model

torch.manual_seed(42)
np.random.seed(42)

def train_one_run(
    csv_file:      str,
    csv_path:      str,
    bert_name:     str = "bert-base-uncased",
    batch_size:    int = 16,
    lr:            float = 2e-5,
    epochs:        int = 3,
    use_dropout:   bool = False,
    dropout_p:     float = 0.1,
    device:        str  = "cuda"
):
    print(f"Device: {device}")
    if device is None:
        print("No GPU available, exiting...")
        exit(1)

    # Data
    df = load_full_data(csv_file, csv_path)
    train_df, v_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"train={len(train_df)}  val={len(v_df)}")

    tokenizer = BertTokenizer.from_pretrained(bert_name)
    collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")

    train_ds = BertDataset(train_df, tokenizer)
    val_ds = BertDataset(v_df, tokenizer)

    train_loader = torch.utils.data.DataLoader(
                    train_ds, batch_size=batch_size,
                    shuffle=True,  collate_fn=collator)
    val_loader = torch.utils.data.DataLoader(
                    val_ds,   batch_size=batch_size,
                    shuffle=False, collate_fn=collator)

    # Model
    base = BertModel.from_pretrained(bert_name)
    model = Bert(base, use_dropout=use_dropout, dropout=dropout_p)
    model.to(device).train()
    optim = AdamW(model.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    num_batches = 0
    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        train_loss, train_preds, train_labels = 0, [], []
        for batch in tqdm(train_loader, desc=f"ep{epoch}-train"):
            optim.zero_grad()
            logits = model(batch["input_ids"].to(device),
                         batch["attention_mask"].to(device))
            loss_value = loss(logits, batch["labels"].to(device))
            loss_value.backward()
            optim.step()
            train_loss += loss_value.item()
            num_batches += 1

            train_preds.extend(torch.argmax(logits, 1).cpu())
            train_labels.extend(batch["labels"].cpu())

        train_loss /= num_batches
        train_f1 = f1_score(train_labels, train_preds)
        train_acc = accuracy_score(train_labels, train_preds)
        print(f"Train Loss: {train_loss:.4f}, F1: {train_f1:.4f}, Accuracy: {train_acc:.4f}")

        # ---- val ----
        model.eval(); val_loss, val_preds, val_labels = 0, [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="val"):
                logits = model(batch["input_ids"].to(device),
                               batch["attention_mask"].to(device))
                val_loss += loss(logits, batch["labels"].to(device)).item()
                val_preds.extend(torch.argmax(logits, 1).cpu())
                val_labels.extend(batch["labels"].cpu())
        val_f1  = f1_score(val_labels, val_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Validation Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Accuracy: {val_acc:.4f}")

        best_val_f1 = max(best_val_f1, val_f1)

    # ---------- return summary ----------
    return model.state_dict(), {
        "val_f1": best_val_f1,
        "val_acc": val_acc,
        "params": dict(
            bert_name=bert_name, batch_size=batch_size,
            lr=lr, epochs=epochs, dropout=use_dropout, p=dropout_p
        )
    }