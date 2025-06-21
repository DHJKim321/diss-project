import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dotenv import load_dotenv
from src.utils.data_utils import load_full_data, load_test_data
from src.utils.eval_utils import evaluate_model, save_evaluation, add_predictions_to_data
from src.utils.text_utils import reddit_tokenizer
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
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
    max_features = int(os.getenv("MAX_FEATURES_TFIDF"))
    use_class_weights = os.getenv("USE_CLASS_WEIGHTS").lower() == "true"
    
    # ------------ Load Data ------------
    print("Loading data...")
    data = load_full_data(train_file, train_data_path)
    test_data = load_test_data(test_file, test_data_path)
    vectorizer = TfidfVectorizer(
        # tokenizer=reddit_tokenizer,
        ngram_range=(1, 2),
        max_features=max_features,
        stop_words='english',
        lowercase=True,
        strip_accents='unicode'
    )
    # Training Data
    X = vectorizer.fit_transform(data['text']).toarray()
    y = data['label'].values

    # Calculate scale_pos_weight for imbalanced classes
    weight = None
    if use_class_weights:
        class_counts = data['label'].value_counts()
        weight = class_counts[0] / class_counts[1]
        print(f"Class distribution: {class_counts.to_dict()}")
        print(f"Scale pos weight: {weight}")

    features = vectorizer.get_feature_names_out()

    print(f"Total features: {len(features)}")
    for i, feat in enumerate(features[:20]):
        print(f"{i + 1}. {feat}")

    # Test Data
    test_X = vectorizer.transform(test_data['text']).toarray()
    test_y = test_data['label'].values
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {test_X.shape}")

    # ------------ Initialize Model ------------
    print("Initializing model...")
    if use_random_forest:
        model = XGBRFClassifier(
            scale_pos_weight=weight,
            eval_metric='logloss', 
            random_state=42
        )
    else:
        model = XGBClassifier(
            scale_pos_weight=weight,
            eval_metric='logloss', 
            random_state=42
        )
    
    # ------------ Train Model ------------
    print("Training model...")
    model.fit(X_train, y_train)
    
    # ------------ Save Model ------------
    print(f"Saving model to {model_save_path}...")
    model.save_model(model_save_path + "/xgboost_rf.model" if use_random_forest else "/xgboost.model")
    
    # ------------ Evaluate Model ------------
    print("Evaluating model on validation set...")
    y_pred = model.predict(X_val)
    print("Validation Classification Report:")
    evaluate_model(y_pred, y_val)

    # ------------ Test Model ------------
    print("Evaluating model on test set...")
    test_preds = model.predict(test_X)
    print("Test Classification Report:")
    evaluations = evaluate_model(test_preds, test_y)
    save_evaluation(evaluations, test_file, data_save_path, model_name="xgboost_rf" if use_random_forest else "xgboost")
    add_predictions_to_data(test_data, test_file, test_preds, model_name="xgboost_rf" if use_random_forest else "xgboost")