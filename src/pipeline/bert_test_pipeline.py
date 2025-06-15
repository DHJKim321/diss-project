import os
import sys
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from src.utils.data_utils import load_test_data
from src.data.BertDataset import BertDataset
from src.model.bert import Bert
from transformers import BertTokenizer
from torch.utils.data import DataLoader

torch.manual_seed(42)

if __name__ == "__main__":
    load_dotenv()

    # ------------ Load environment variables ------------
    test_file_path  = os.getenv("TEST_FILE")
    model_path = os.getenv("MODEL_SAVE_PATH")
    batch_size = int(os.getenv("BATCH_SIZE"))
    bert_name = os.getenv("BERT_NAME")
    max_length = int(os.getenv("MAX_LENGTH"))

    # ------------ Load Data and Tokenizer ------------
    test_data   = load_test_data(test_file_path)
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    dataset   = BertDataset(test_data, tokenizer, max_length=max_length)
    dataloader   = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ------------ Load Model and Device ------------
    device = 'cuda' if torch.cuda.is_available() else None
    if device is None:
        print("No GPU available, exiting...")
        exit(1)
    print(f"Using device: {device}")
    print(f"Loading model from {model_path}")
    model = Bert.load(model_path)
    model.to_device(device)
    model.eval()

    preds, labels = [], []

    # ------------ Start Inference ------------
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids,
                           attention_mask=attention_mask).logits

            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            labels.extend(labels.cpu().tolist())

    print(classification_report(labels, preds, digits=4))
