import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dotenv import load_dotenv
from src.utils.data_utils import load_full_data, load_test_data
from src.utils.eval_utils import evaluate_model, save_evaluation, add_predictions_to_data
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

if __name__ == "__main__":
    # ------------ Load environment variables ------------
    load_dotenv()
    train_file = os.getenv("TRAIN_FILE")
    train_data_path = os.getenv("TRAIN_DATA_PATH")
    test_file = os.getenv("TEST_FILE")
    test_data_path = os.getenv("TEST_DATA_PATH")
    model_save_path = os.getenv("MODEL_SAVE_PATH")
    data_save_path = os.getenv("DATA_SAVE_PATH")
    max_features = int(os.getenv("MAX_FEATURES_TFIDF"))
    
    # ------------ Load Data ------------
    print("Loading data...")
    data = load_full_data(train_file, train_data_path)
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        stop_words='english',
        lowercase=True,
        strip_accents='unicode'
    )
    
    # Training Data
    X = vectorizer.fit_transform(data['text'])
    y = data['label'].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Class Weights
    classes = list(set(y))
    weights = compute_class_weight(class_weight='balanced', classes=np.array(classes), y=y)
    class_weight_dict = dict(zip(classes, weights))
    
    features = vectorizer.get_feature_names_out()
    
    print(f"Total features: {len(features)}")
    for i, feat in enumerate(features[:20]):
        print(f"{i + 1}. {feat}")
    
    # Test Data
    test_data = load_test_data(test_file, test_data_path)
    test_X = vectorizer.transform(test_data['text'])
    test_y = test_data['label'].values

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {test_X.shape}")
    
    # ------------ Initialize Model ------------
    model = SVC(kernel='linear', class_weight=class_weight_dict, probability=True)
    
    # ------------ Train Model ------------
    print("Training model...")
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # ------------ Evaluate Model ------------
    print("Evaluating model...")
    preds = model.predict(X_val)
    print("Validation Classification Report:")
    evaluations = evaluate_model(preds, y_val)
    
    # ------------ Test Model ------------
    print("Testing model...")
    test_preds = model.predict(test_X)
    print("Test Classification Report:")
    test_evaluations = evaluate_model(test_preds, test_y)
    save_evaluation(test_evaluations, test_file, data_save_path, "svm")
    add_predictions_to_data(test_data, test_file, test_preds, "svm")
