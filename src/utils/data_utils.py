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

def load_and_prepare_data(csv_file):
    # Load the dataset
    data = pd.read_csv(csv_file)
    
    # Fill missing values with empty strings
    data.fillna('', inplace=True)
    
    # Replace subreddit with class label
    data['class'] = data['subreddit'].apply(lambda x: 0 if x.lower() == 'casualuk' else 1)
    
    # Drop the original subreddit column
    data.drop(columns=['subreddit'], inplace=True)
    
    # Concatenate title and selftext into a single text column
    data['text'] = data['title'] + ' ' + data['selftext']
    
    # Drop the original title and selftext columns
    data.drop(columns=['title', 'selftext'], inplace=True)
    
    return data