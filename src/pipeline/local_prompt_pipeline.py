import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dotenv import load_dotenv

from src.utils.llm_utils import *
from src.utils.data_utils import load_test_data
from src.prompts.templates import TEMPLATE_V1

if __name__ == "__main__":
    load_dotenv()

    model_path = os.getenv("LLM_PATH")
    data_path = os.getenv("DATA_PATH")
    save_path = os.getenv("DATA_SAVE_PATH")

    data = load_test_data(data_path)                         

    pipeline_ = load_llama_model(model_path)

    data = batch_process(pipeline_, TEMPLATE_V1, data, new_col='predictions', num_posts=10, data_path=save_path, source_col='text')
    print(data.head())