import os
import sys
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dotenv import load_dotenv
from src.utils.data_utils import load_test_data
from src.utils.eval_utils import evaluate_model, save_evaluation
from src.data.BertDataset import BertDataset
from src.model.bert import Bert
from transformers import BertTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

torch.manual_seed(42)

if __name__ == "__main__":
    load_dotenv()

    # ------------ Load environment variables ------------
    test_file = os.getenv("TEST_FILE")
    test_data_path = os.getenv("TEST_DATA_PATH")
    model_path = os.getenv("MODEL_SAVE_PATH")
    data_save_path = os.getenv("DATA_SAVE_PATH")
    batch_size = int(os.getenv("BATCH_SIZE"))
    bert_model = os.getenv("BERT_MODEL")

    # ------------ Load Data and Tokenizer ------------
    test_data = load_test_data(test_file, test_data_path)
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    dataset = BertDataset(test_data, tokenizer)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    # ------------ Load Model and Device ------------
    device = 'cuda' if torch.cuda.is_available() else None
    if device is None:
        print("No GPU available, exiting...")
        exit(1)
    print(f"Using device: {device}")
    print(f"Loading model from {model_path}")
    model = Bert.load(model_path + "bert_model.pth")
    model.to_device(device)
    model.eval()

    preds, all_labels = [], []

    # ------------ Start Inference ------------
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids,
                           attention_mask=attention_mask)

            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # ------------ Evaluate Model ------------
    evaluations = evaluate_model(preds, all_labels)
    save_evaluation(evaluations, test_file, data_save_path, model_name=bert_model)