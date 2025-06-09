import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import os
from dotenv import load_dotenv

from model.LLM import *
from src.utils.data_utils import load_test_data
from src.utils.api_utils import read_token
from src.prompts.templates import TEMPLATE_V1

if __name__ == "__main__":
    load_dotenv()

    model_path = os.getenv("LLM_PATH")
    huggingface_token_path = os.getenv("HUGGINGFACE_TOKEN")
    huggingface_token = read_token(huggingface_token_path)
    data_path = os.getenv("DATA_PATH")
    save_path = os.getenv("DATA_SAVE_PATH")

    data = load_test_data(data_path)                         

    pipeline = load_model(model_path, huggingface_token)

    data = batch_process(pipeline, TEMPLATE_V1, data, new_col='predictions', num_posts=10, max_tokens=1, data_path=save_path, source_col='text')
    print(data.head())

    