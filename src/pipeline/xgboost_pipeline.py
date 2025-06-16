import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dotenv import load_dotenv
from src.utils.data_utils import load_full_data, load_test_data
from src.utils.eval_utils import evaluate_model, save_evaluation
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # ------------ Load environment variables ------------
    load_dotenv()
    train_file = os.getenv("TRAIN_FILE")
    train_data_path = os.getenv("TRAIN_DATA_PATH")
    test_file = os.getenv("TEST_FILE")
    test_data_path = os.getenv("TEST_DATA_PATH")
    model_save_path = os.getenv("MODEL_SAVE_PATH")
    data_save_path = os.getenv("DATA_SAVE_PATH")
    use_random_forest = os.getenv("USE_RANDOM_FOREST").lower() == "true"
    
    # ------------ Load Data ------------
    data = load_full_data(train_file, train_data_path)
    test_data = load_test_data(test_file, test_data_path)
    X = data['text']
    y = data['class']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ------------ Initialize Model ------------
    if use_random_forest:
        model = XGBRFClassifier(random_state=42)
    else:
        model = XGBClassifier(random_state=42)
    
    # ------------ Train Model ------------
    model.fit(X_train, y_train)
    
    # ------------ Save Model ------------
    model.save_model(model_save_path)
    
    # ------------ Evaluate Model ------------
    y_pred = model.predict(X_val)
    print(f"Validation Classification Report:")
    evaluate_model(y_pred, y_val)

    # ------------ Test Model ------------
    test_X = test_data['text']
    test_y = test_data['label']
    test_preds = model.predict(test_X)
    evaluations = evaluate_model(test_preds, test_y)
    save_evaluation(evaluations, test_file, data_save_path, model_name="xgboost_rf" if use_random_forest else "xgboost")
    print(f"Test Classification Report:")