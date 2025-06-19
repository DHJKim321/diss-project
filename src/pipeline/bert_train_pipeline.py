import os, sys, torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv
from src.utils.data_utils import load_full_data
from src.utils.eval_utils import evaluate_model
from src.data.BertDataset import BertDataset
from src.model.bert import Bert
from transformers import BertTokenizer, BertModel, DataCollatorWithPadding
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
torch.manual_seed(42)

if __name__ == "__main__":
    # ------------ Load environment variables ------------
    load_dotenv()
    train_file = os.getenv("TRAIN_FILE")
    train_data_path = os.getenv("TRAIN_DATA_PATH")
    batch_size = int(os.getenv("BATCH_SIZE"))
    bert_model = os.getenv("BERT_MODEL")
    learning_rate = float(os.getenv("LEARNING_RATE"))
    epochs = int(os.getenv("EPOCHS"))
    model_save_path = os.getenv("MODEL_SAVE_PATH")
    early_stopping = os.getenv("EARLY_STOPPING").lower() == "true"
    patience = int(os.getenv("PATIENCE"))
    use_dropout = os.getenv("USE_DROPOUT").lower() == "true"
    dropout = float(os.getenv("DROPOUT"))

    # ------------ Load Data and Tokenizer ------------
    tokenizer = BertTokenizer.from_pretrained(bert_model)

    full_data = load_full_data(train_file, train_data_path)
    train_data, val_data = train_test_split(full_data, test_size=0.2, random_state=42)
    print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}")
    train_dataset = BertDataset(train_data, tokenizer)
    val_dataset = BertDataset(val_data, tokenizer)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    # ------------ Load Model, Loss, and Device ------------
    bert = BertModel.from_pretrained(bert_model)
    model = Bert(bert, use_dropout=use_dropout, dropout=dropout)
    loss = CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else None
    print(f"Using device: {device}")
    if device is None:
        print("No GPU available, exiting...")
        exit(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.to_device(device)
    model.train()

    print(f"Model loaded and moved to {device}")
    print(f"Starting training with batch size {batch_size}...")
    print(f"Number of batches: {len(train_loader)}")

    # ------------ Training Loop ------------
    print("Starting training...")
    best_val_loss = float("inf")
    curr_patience = 0
    val_history  = []

    for epoch in range(epochs):
        train_preds = []
        train_labels = []
        # ------------ Training ------------
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        train_loss = 0
        num_batches = 0

        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss_value = loss(logits, labels)
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss_value.item()
            num_batches += 1

            train_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            train_labels.extend(labels.cpu().tolist())

        train_loss /= num_batches
        train_f1 = f1_score(train_labels, train_preds)
        train_acc = accuracy_score(train_labels, train_preds)
        print(f"Train Loss: {train_loss:.4f}, F1: {train_f1:.4f}, Accuracy: {train_acc:.4f}")

        # ------------ Validation ------------
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        print("Starting validation...")
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss_value = loss(logits, labels)

                val_loss += loss_value.item()
                val_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                val_labels.extend(labels.cpu().tolist())
            val_loss /= len(val_loader)
            val_history.append(val_loss)
            val_f1 = f1_score(val_labels, val_preds)
            val_acc = accuracy_score(val_labels, val_preds)
            print(f"Validation Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Accuracy: {val_acc:.4f}")

            # ------------ Early Stopping ------------
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    curr_patience = 0
                    print(f"New best validation loss: {best_val_loss:.4f}")
                else:
                    curr_patience += 1
                    print(f"No improvement in validation loss for {curr_patience} epochs")
                    if curr_patience >= patience:
                        print("Early stopping triggered")
                        break
        torch.cuda.empty_cache()

    # ------------ Evaluate Model ------------
    evaluate_model(val_preds, val_labels)

    # ------------ Save Model ------------
    model.save(model_save_path + "bert_model.pth")
    print(f"Model saved to {model_save_path}/bert_model.pth")