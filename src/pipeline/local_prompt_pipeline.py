import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dotenv import load_dotenv

from src.utils.llm_utils import *
from src.utils.data_utils import load_test_data
from src.prompts.templates import TEMPLATE_V1
from src.utils.eval_utils import evaluate_model, save_evaluation

if __name__ == "__main__":
    load_dotenv()

    model_path = os.getenv("LLM_PATH")
    test_file = os.getenv("TEST_FILE")
    data_path = os.getenv("DATA_PATH")
    save_path = os.getenv("DATA_SAVE_PATH")

    data = load_test_data(test_file, data_path)                         

    pipeline_ = load_llama_model(model_path)

    updated_data = batch_process(pipeline_, TEMPLATE_V1, data, new_col='predictions', num_posts=4, test_file=test_file, data_path=save_path, source_col='text')

    predictions = updated_data['predictions'].tolist()
    labels = updated_data['label'].tolist()
    evaluations = evaluate_model(predictions, labels)
    save_evaluation(evaluations, test_file, save_path)