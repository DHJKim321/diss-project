import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.model.bert import BertModel
import torch

from dotenv import load_dotenv

import random
torch.manual_seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)

if __name__ == "__main__":
    # ------------ Load environment variables ------------
    load_dotenv()

    # ---- Data Paths ----
    model_save_path = os.getenv("MODEL_SAVE_PATH")
    bert_model = os.getenv("BERT_MODEL")
    # ---- Runtime Variables ----
    batch_size = int(os.getenv("BATCH_SIZE"))
    learning_rate = float(os.getenv("LEARNING_RATE"))
    warmup_epochs = int(os.getenv("WARMUP_EPOCHS"))
    epochs = int(os.getenv("EPOCHS"))
    alpha = float(os.getenv("ALPHA"))
    lambda_u = float(os.getenv("LAMBDA_U"))
    p_threshold = float(os.getenv("P_THRESHOLD"))
    temperature = int(os.getenv("SHARPENING TEMPERATURE"))



    # ------------ Load Model ------------
    print(f"Loading BERT model from {bert_model}")
    model = BertModel(bert_model)

