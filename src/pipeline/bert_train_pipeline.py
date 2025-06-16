import os, sys, torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv
from src.utils.data_utils import load_train_data
from src.data.BertDataset import BertDataset
from src.model.bert import Bert
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
torch.manual_seed(42)

if __name__ == "__main__":
    # ------------ Load environment variables ------------
    load_dotenv()
    train_file_path = os.getenv("TRAIN_FILE")
    batch_size = int(os.getenv("BATCH_SIZE"))
    bert_name = os.getenv("BERT_NAME")
    learning_rate = float(os.getenv("LEARNING_RATE"))
    epochs = int(os.getenv("EPOCHS"))
    model_save_path = os.getenv("MODEL_SAVE_PATH")
    early_stopping = os.getenv("EARLY_STOPPING").lower() == "true"
    patience = int(os.getenv("PATIENCE"))
    use_dropout = os.getenv("USE_DROPOUT").lower() == "true"
    dropout = float(os.getenv("DROPOUT"))

    # ------------ Load Data and Tokenizer ------------
    train_data = load_train_data(train_file_path)
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    dataset = BertDataset(train_data, tokenizer)
    print(f"Dataset length: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ------------ Load Model, Loss, and Device ------------
    bert = BertForSequenceClassification.from_pretrained(bert_name, num_labels=2)
    model = Bert(bert, use_dropout=use_dropout, dropout=dropout)
    loss = CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else None
    print(f"Using device: {device}")
    if device is None:
        print("No GPU avialable, exiting...")
        exit(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.to_device(device)
    model.train()

    losses = []

    print(f"Model loaded and moved to {device}")
    print(f"Starting training with batch size {batch_size}...")
    print(f"Number of batches: {len(dataloader)}")

    # ------------ Start Training ------------
    epoch_losses = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        total_loss = 0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss_value = loss(logits, labels)
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            total_loss += loss_value.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        if early_stopping and epoch > patience:
            if avg_loss > epoch_losses[-patience - 1]:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

