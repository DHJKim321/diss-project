import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def encode_texts(texts, model, tokenizer, device, batch_size=16):
    model.eval()
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]
            embeddings.append(batch_embeddings.cpu().numpy())

    return np.concatenate(embeddings, axis=0)

if __name__ == "__main__":
    df = pd.read_csv('src/data/train/imdb.csv')
    df.fillna('', inplace=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    bert = BertModel.from_pretrained('bert-base-uncased').to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # data = (df['title'] + ' ' + df['selftext']).tolist()
    data = df['review'].tolist()
    encoded_data = encode_texts(data, bert, tokenizer, device)

    print(f"Encoded data shape: {encoded_data.shape}")
    np.save('src/data/train/bert_embeddings_imdb.npy', encoded_data)
    print("Embeddings saved to train/bert_embeddings_imdb.npy")
