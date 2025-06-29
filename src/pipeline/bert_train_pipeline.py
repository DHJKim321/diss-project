import os, sys, torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv
from src.utils.data_utils import load_full_data, check_embedding_existence
from src.data.BertDataset import BertDataset
from src.model.bert import Bert
from src.modules.GMMLabelCorrector import GMMLabelCorrector
from transformers import BertTokenizer, BertModel, DataCollatorWithPadding
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import KMeans
import random
import numpy as np
torch.manual_seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)

if __name__ == "__main__":
    # ------------ Load environment variables ------------
    load_dotenv()
    # ---- Data Paths ----
    train_file = os.getenv("TRAIN_FILE")
    train_data_path = os.getenv("TRAIN_DATA_PATH")
    embedding_full_path = os.getenv("EMBEDDING_FULL_PATH")
    model_save_path = os.getenv("MODEL_SAVE_PATH")
    # ---- Model Variables ----
    batch_size = int(os.getenv("BATCH_SIZE"))
    bert_model = os.getenv("BERT_MODEL")
    learning_rate = float(os.getenv("LEARNING_RATE"))
    epochs = int(os.getenv("EPOCHS"))
    use_dropout = os.getenv("USE_DROPOUT").lower() == "true"
    dropout = float(os.getenv("DROPOUT"))
    head_type = os.getenv("HEAD_TYPE").lower()
    # ---- Label Denoising ----
    denoise_labels = os.getenv("DENOISE_LABELS").lower() == "true"
    denoise_type = os.getenv("DENOISE_TYPE").lower()
    gmm_threshold = float(os.getenv("GMM_THRESHOLD"))
    reducer_type = os.getenv("REDUCER_TYPE").lower()

    # ------------ Load Data and Tokenizer ------------
    tokenizer = BertTokenizer.from_pretrained(bert_model)

    full_data = load_full_data(train_file, train_data_path)
    train_data = full_data
    print(f"Loaded {len(train_data)} training samples.")

    # ------------ (Optional) Denoise Noisy Labels ------------
    if not check_embedding_existence(embedding_full_path):
        print("Embedding file does not exist. Please run src/pipeline/bert_create_embeddings.py first.")
        exit(1)
    if denoise_labels:
        print(f"Denoising labels with type: {denoise_type}")
        print("Loading embeddings for denoising...")
        train_embeddings = torch.from_numpy(np.load(embedding_full_path))
        if denoise_type == "gmm":
            print("Denoising labels")
            gmm = GMMLabelCorrector(train_embeddings, reducer_type, n_components=2, covariance_type='full')
            train_data['denoised_label'] = gmm.threshold_predict(train_embeddings, threshold=gmm_threshold)
            train_data = train_data[train_data['denoised_label'] != -1]  # Remove uncertain predictions
            train_data['denoised_label'] = train_data['denoised_label'].apply(lambda x: 1 if x == 0 else 0) # Version 2 - align GMM clusters with labels
            # print(f"Removed {len(train_data[train_data['denoised_label'] == -1])} uncertain labels.")
        elif denoise_type == 'kmeans':
            print("Denoising labels with KMeans")
            kmeans = KMeans(n_clusters=2, random_state=42)
            kmeans.fit(train_embeddings)
            train_data['denoised_label'] = kmeans.labels_
        print("Labels denoised.")
            
        train_data.to_csv(f"{train_data_path}/{denoise_type}_denoised_{train_file}", index=False)
        train_data.drop(columns=['label'], inplace=True)
        train_data.rename(columns={'denoised_label': 'label'}, inplace=True)

    # ------------ Create Data Loaders ------------
    train_dataset = BertDataset(train_data, tokenizer)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

    # ------------ Load Model, Loss, and Device ------------
    bert = BertModel.from_pretrained(bert_model)
    model = Bert(bert, head_type=head_type, use_dropout=use_dropout, dropout=dropout)
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
        torch.cuda.empty_cache()

    # ------------ Save Model ------------
    model_save_path += "bert_model.pth"
    if denoise_labels:
        model_save_path = model_save_path.replace(".pth", f"_{reducer_type}_{denoise_type}_denoised.pth")
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}/bert_model.pth")