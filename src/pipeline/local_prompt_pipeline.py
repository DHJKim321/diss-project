import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dotenv import load_dotenv

from src.utils.llm_utils import *
from src.utils.data_utils import load_test_data
from src.prompts.templates import *
from src.utils.eval_utils import evaluate_model, save_evaluation, add_predictions_to_data

if __name__ == "__main__":
    load_dotenv()

    model_path = os.getenv("LLM_PATH")
    test_file = os.getenv("TEST_FILE")
    test_path = os.getenv("TEST_DATA_PATH")
    save_path = os.getenv("DATA_SAVE_PATH")

    data = load_test_data(test_file, test_path)                         

    pipeline_ = load_llama_model(model_path)

    for i, template in enumerate([TEMPLATE_V4]):
        print(f"Processing with template {i+1}")
        updated_data = batch_process(pipeline_, template, data, new_col=f'predictions_{i+1}', num_posts=4, test_file=test_file, data_path=save_path, source_col='text')

        predictions = updated_data[f'predictions_{i+1}'].tolist()
        labels = updated_data['label'].tolist()
        evaluations = evaluate_model(predictions, labels)
        save_evaluation(evaluations, test_file, save_path, f'mistral_template_{i+1}')
        add_predictions_to_data(updated_data, test_file, save_path, predictions, f'mistral_template_{i+1}', template=f"_{i+1}")