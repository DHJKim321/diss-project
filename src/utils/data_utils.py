import pandas as pd
import os, sys
import numpy as np
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
    
def load_imdb_data(file, file_path):
    path = file_path + file
    print(f"Loading IMDB data from {path}")
    data = pd.read_csv(path)
    data.fillna('', inplace=True)
    data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    data = data.rename(columns={'review': 'text', 'sentiment': 'label'})
    data = data[['text', 'label']]
    print(f"IMDB data loaded with {len(data)} samples")
    return data

def inject_symmetric_noise(data, noise_ratio=0.2):
    print(f"Injecting symmetric noise with ratio {noise_ratio}")
    num_samples = len(data)
    num_noisy_samples = int(num_samples * noise_ratio)
    noisy_indices = np.random.choice(num_samples, num_noisy_samples, replace=False)
    data.loc[noisy_indices, 'label'] = 1 - data.loc[noisy_indices, 'label']
    return data