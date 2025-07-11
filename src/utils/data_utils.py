import pandas as pd
import os, sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import matplotlib.pyplot as plt

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
    data = data.copy().reset_index(drop=True)
    num_samples = len(data)
    num_noisy_samples = int(num_samples * noise_ratio)
    noisy_indices = np.random.choice(num_samples, num_noisy_samples, replace=False)
    data.loc[noisy_indices, 'label'] = 1 - data.loc[noisy_indices, 'label']
    print(f"Injected noise into {num_noisy_samples} samples")
    return data

def save_loss_histogram(losses, epoch, model):
    plt.figure(figsize=(6, 4))
    plt.hist(losses, bins=100, color='skyblue', alpha=0.7, edgecolor='black')
    plt.title(f"Loss Distribution At Epoch {epoch} for Model {model}")
    plt.xlabel("Normalized Loss")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"src/data/images/loss/loss_distribution_epoch_{epoch}_model_{model}.png")
    plt.close()

def get_labels_injected_list(original, noisy):
    # Outputs a dictionary of {idx: isNoiseInjected}
    output = {}
    for idx, (orig_label, noisy_label) in enumerate(zip(original, noisy)):
        output[idx] = orig_label != noisy_label
    return output

def save_orig_noisy_loss_histogram(noisy_dict, loss, epoch, model):
    noisy_losses = np.array([loss[idx] for idx, is_noisy in noisy_dict.items() if is_noisy])
    clean_losses = np.array([loss[idx] for idx, is_noisy in noisy_dict.items() if not is_noisy])

    bins = np.linspace(0, 1, 100)

    plt.figure(figsize=(8, 5))

    plt.hist(clean_losses, bins=bins, density=True, color='royalblue', label='Clean', edgecolor='black', linewidth=0.3, histtype='stepfilled', alpha=1, zorder=1)

    plt.hist(noisy_losses, bins=bins, density=True, color='salmon', label='Noisy', edgecolor='black', linewidth=0.3, histtype='stepfilled', alpha=1, zorder=2)

    plt.xlabel("Normalized loss")
    plt.ylabel("Frequency")
    plt.title(f"Loss Distribution at Epoch {epoch} for Model {model}")
    plt.legend()
    plt.tight_layout()

    save_path = f"src/data/images/loss/orig_noisy_loss_distribution_epoch_{epoch}_model_{model}.png"
    plt.savefig(save_path)
    plt.close()