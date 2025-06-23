import pandas as pd
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def load_full_data(train_file, file_path):
    path = file_path + train_file
    print(f"Loading data from {path}")
    data = pd.read_csv(path)
    data.fillna('', inplace=True)
    data['text'] = data['title'].astype(str) + ' ' + data['selftext'].astype(str)
    # Only keep text and label columns
    data = data[['text', 'label']]
    return data

def load_test_data(test_file, file_path):
    path = file_path + test_file
    print(f"Loading test data from {path}")
    data = pd.read_csv(path)
    data.fillna('', inplace=True)
    data = data[['text', 'label']]
    return data

def check_embedding_existence(embedding_path):
    if not os.path.exists(embedding_path):
        print(f"Embedding file {embedding_path} does not exist.")
        return False
    else:
        print(f"Embedding file {embedding_path} exists.")
        return True