import os, sys, torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv
from src.utils.data_utils import *
from src.utils.eval_utils import evaluate_model, save_evaluation, add_predictions_to_data
from src.data.BertDataset import BertDataset
from src.model.bert import Bert
from src.modules.GMMLabelCorrector import GMMLabelCorrector
from transformers import BertTokenizer, DataCollatorWithPadding
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
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
    # train_file = os.getenv("TRAIN_FILE")
    train_file = "expanded_full_v2.csv"
    train_data_path = os.getenv("TRAIN_DATA_PATH")
    test_file = os.getenv("TEST_FILE")
    test_data_path = os.getenv("TEST_DATA_PATH")
    data_save_path = os.getenv("DATA_SAVE_PATH")
    embedding_full_path = os.getenv("EMBEDDING_FULL_PATH")
    model_save_path = os.getenv("MODEL_SAVE_PATH")
    # ---- Model Variables ----
    batch_size = int(os.getenv("BATCH_SIZE"))
    bert_model = os.getenv("BERT_MODEL")
    # learning_rate = float(os.getenv("LEARNING_RATE"))
    learning_rate = 2e-5
    # epochs = int(os.getenv("EPOCHS"))
    epochs = 5
    use_dropout = os.getenv("USE_DROPOUT").lower() == "true"
    dropout = float(os.getenv("DROPOUT"))
    # head_type = os.getenv("HEAD_TYPE").lower()
    head_type = "lstm"
    # ---- Label Denoising ----
    # denoise_labels = os.getenv("DENOISE_LABELS").lower() == "true"
    denoise_labels = False
    denoise_type = os.getenv("DENOISE_TYPE").lower()
    gmm_threshold = float(os.getenv("GMM_THRESHOLD"))
    reducer_type = os.getenv("REDUCER_TYPE").lower()
    noise_ratio = float(os.getenv("NOISE_RATIO"))
    # use_imdb = os.getenv("USE_IMDB").lower() == "true"
    use_imdb = False

    # ------------ Load Data and Tokenizer ------------
    tokenizer = BertTokenizer.from_pretrained(bert_model)

    if use_imdb:
        print("Using IMDB dataset for training.")
        imdb_data = load_imdb_data(train_file, train_data_path)
        train_data, test_data = train_test_split(imdb_data, test_size=0.2, random_state=42)
        train_data = inject_symmetric_noise(train_data, noise_ratio=noise_ratio)
        test_file = train_file # IMDB dataset does not have a separate test file
    else:
        print("Using ShaPe dataset for training.")
        train_data = load_full_data(train_file, train_data_path)
        test_data = load_test_data(test_file, test_data_path)
    print(f"Loaded {len(train_data)} training samples.")

    # ------------ (Optional) Denoise Noisy Labels ------------
    if denoise_labels:
        if not check_embedding_existence(embedding_full_path):
            print("Embedding file does not exist. Please run src/pipeline/bert_create_embeddings.py first.")
            exit(1)
        print(f"Denoising labels with type: {denoise_type}")
        print("Loading embeddings for denoising...")
        train_embeddings = torch.from_numpy(np.load(embedding_full_path))
        if denoise_type == "gmm":
            print("Denoising labels")
            gmm = GMMLabelCorrector(train_embeddings, reducer_type, n_components=2, covariance_type='full')
            print(f"Reduced embeddings shape: {gmm.reduced_embeddings.shape}")
            train_embeddings = gmm.reduced_embeddings
            train_labels = train_data['label'].values
            train_data['denoised_label'] = gmm.threshold_predict(train_embeddings, train_labels, threshold=gmm_threshold)
            mapping = gmm.get_label_mapping(train_data)
            train_data['denoised_label'] = train_data['denoised_label'].map(mapping)
            print(f"Label mapping: {mapping}")
        elif denoise_type == 'kmeans':
            print("Denoising labels with KMeans")
            kmeans = KMeans(n_clusters=2, random_state=42)
            kmeans.fit(train_embeddings)
            train_data['denoised_label'] = kmeans.labels_
        print("Labels denoised.")
            
        # train_data.to_csv(f"{train_data_path}/{denoise_type}_denoised_{train_file}", index=False)
        train_data.drop(columns=['label'], inplace=True)
        train_data.rename(columns={'denoised_label': 'label'}, inplace=True)

    # ------------ Create Data Loaders ------------
    train_dataset = BertDataset(train_data, tokenizer)
    test_dataset = BertDataset(test_data, tokenizer)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    # ------------ Load Model, Loss, and Device ------------
    print(f"Loading BERT model from {bert_model} with head type {head_type}")
    model = Bert(bert_model, head_type=head_type, use_dropout=use_dropout, dropout=dropout)
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

        # ------------ Validation ------------
        print(f"Testing model on test data at epoch {epoch + 1}")
        test_preds = []
        test_labels = []
        test_loss = 0
        num_test_batches = 0
        with torch.no_grad():
            model.eval()
            for batch in tqdm(test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss_value = loss(logits, labels)

                test_loss += loss_value.item()
                num_test_batches += 1

                test_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                test_labels.extend(labels.cpu().tolist())
        print(f"Test Loss: {test_loss / num_test_batches:.4f}")
        print(f"Test Accuracy: {accuracy_score(test_labels, test_preds):.4f}")
        print(f"Test F1 Score: {f1_score(test_labels, test_preds, average='macro'):.4f}")

    # ------------ Save Evaluations ------------
    evaluations = evaluate_model(test_preds, test_labels)
    if head_type:
        bert_model += f"_{head_type}"
    if denoise_labels:
        bert_model += f"_{reducer_type}_{denoise_type}_denoised"
    save_evaluation(evaluations, test_file, data_save_path, model_name=bert_model)
    add_predictions_to_data(test_data, test_file, data_save_path, test_preds, model_name=bert_model)

    # ------------ Save Model ------------
    model_save_path += f"bert_model_{head_type}.pth"
    if use_imdb:
        model_save_path = model_save_path.replace(".pth", "_imdb.pth")
    if head_type:
        model_save_path = model_save_path.replace(".pth", f"_{head_type}.pth")
    if denoise_labels:
        model_save_path = model_save_path.replace(".pth", f"_{reducer_type}_{denoise_type}_denoised.pth")
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")