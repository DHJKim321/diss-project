import os
import sys
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dotenv import load_dotenv
from src.utils.data_utils import load_test_data, load_imdb_data
from src.utils.eval_utils import evaluate_model, save_evaluation, add_predictions_to_data
from src.data.BertDataset import BertDataset
from src.model.bert import Bert
from transformers import BertTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

torch.manual_seed(42)

if __name__ == "__main__":
    load_dotenv()

    # ------------ Load environment variables ------------
    train_file = os.getenv("TRAIN_FILE")
    test_file = os.getenv("TEST_FILE")
    test_data_path = os.getenv("TEST_DATA_PATH")
    model_path = os.getenv("MODEL_SAVE_PATH")
    data_save_path = os.getenv("DATA_SAVE_PATH")
    batch_size = int(os.getenv("BATCH_SIZE"))
    bert_model = os.getenv("BERT_MODEL")
    denoise_labels = os.getenv("DENOISE_LABELS").lower() == "true"
    denoise_type = os.getenv("DENOISE_TYPE").lower()
    reducer_type = os.getenv("REDUCER_TYPE").lower()
    head_type = os.getenv("HEAD_TYPE").lower()
    use_imdb = os.getenv("USE_IMDB").lower() == "true"
    noise_ratio = float(os.getenv("NOISE_RATIO"))

    # ------------ Load Data and Tokenizer ------------
    if use_imdb:
        print("Using IMDB dataset for testing.")
        imdb_data = load_imdb_data(test_file, test_data_path)
        _, test_data = train_test_split(imdb_data, test_size=0.2, random_state=42)
        test_file = train_file # IMDB dataset does not have a separate test file
    else:
        print("Using ShaPe dataset for testing.")
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
    model_save_path = model_path + "bert_model.pth"
    if head_type:
        model_save_path = model_save_path.replace(".pth", f"_{head_type}.pth")
    if denoise_labels:
        model_save_path = model_save_path.replace(".pth", f"_{reducer_type}_{denoise_type}_denoised.pth")
    print(f"Loading model from {model_save_path}")
    model = Bert.load(model_save_path, device, head_type)
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
    if denoise_labels:
        bert_model += f"_{head_type}_{reducer_type}_{denoise_type}_denoised"
    save_evaluation(evaluations, test_file, data_save_path, model_name=bert_model)
    add_predictions_to_data(test_data, test_file, data_save_path, preds, model_name=bert_model)