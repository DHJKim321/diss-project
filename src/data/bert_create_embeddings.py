import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from transformers import BertTokenizer, BertModel
import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

if __name__ == "__main__":
    df = pd.read_csv('src/data/train/expanded_full.csv')
    df.fillna('', inplace=True)
    bert = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = 'cuda'
    bert.to(device)
    data = (df['title'] + ' ' + df['selftext']).tolist()

    def encode_texts(texts):
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            print("Encoding texts...")
            outputs = bert(**inputs, verbose=True)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    encoded_data = encode_texts(data)
    print(f"Encoded data shape: {encoded_data.shape}")

    with open('train/bert_embeddings_expanded_full.npy', 'wb') as f:
        np.save(f, encoded_data)
    print("Embeddings saved to train/bert_embeddings_expanded_full.npy")
