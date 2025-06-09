import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dotenv import load_dotenv

from model.LLM import *
from src.utils.data_utils import load_test_data
from src.utils.api_utils import read_token
from src.prompts.templates import TEMPLATE_V1, SYSTEM_PROMPT_V1

if __name__ == "__main__":
    load_dotenv()

    data_path = os.getenv("DATA_PATH")
    save_path = os.getenv("DATA_SAVE_PATH")
    mistral_key_path = os.getenv("MISTRAL_API_KEY")
    mistral_key = read_token(mistral_key_path)
    mistral_name = os.getenv("MISTRAL_MODEL_NAME")

    data = load_test_data(data_path)
    client = load_mistral_client(mistral_key)
    data = mistral_batch_process(
        client, mistral_name, TEMPLATE_V1, SYSTEM_PROMPT_V1, data, new_col='predictions', num_posts=1, data_path=save_path, source_col='text'
    )
    print(data.head())

    