import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def encode_texts(texts, model, device, batch_size=16):
    model.eval()
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i+batch_size]
        with torch.no_grad():
            batch_embeddings = model.encode(batch, device=device)
            embeddings.append(batch_embeddings)

    return np.concatenate(embeddings, axis=0)

if __name__ == "__main__":
    df = pd.read_csv('src/data/train/expanded_full.csv')
    df.fillna('', inplace=True)

    device = torch.device('mps')
    model = SentenceTransformer("all-MiniLM-L6-v2")

    data = (df['title'] + ' ' + df['selftext']).tolist()
    encoded_data = encode_texts(data, model, device)

    print(f"Encoded data shape: {encoded_data.shape}")
    np.save('src/data/train/SF_embeddings_expanded_full.npy', encoded_data)
    print("Embeddings saved to train/SF_embeddings_expanded_full.npy")
