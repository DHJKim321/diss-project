import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.model.bert import BertModel

from dotenv import load_dotenv

if __name__ == "__main__":
    # ------------ Load environment variables ------------
    load_dotenv()
    # ---- Data Paths ----
    model_save_path = os.getenv("MODEL_SAVE_PATH")
    bert_model = os.getenv("BERT_MODEL")

    # ---- Runtime Variables ----
    batch_size = int(os.getenv("BATCH_SIZE"))
    learning_rate = float(os.getenv("LEARNING_RATE"))
    epochs = int(os.getenv("EPOCHS"))
    use_dropout = os.getenv("USE_DROPOUT").lower() == "true"
    dropout = float(os.getenv("DROPOUT"))

    # ------------ Load Model ------------
    print(f"Loading BERT model from {bert_model}")
    model = BertModel(bert_model)

