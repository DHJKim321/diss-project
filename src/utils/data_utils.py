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

def inject_symmetric_noise(data, noise_ratio, seed=42):
    rng = np.random.default_rng(seed)
    data = data.copy().reset_index(drop=True)

    num_samples = len(data)
    classes = np.unique(data['label'].values)
    num_classes = len(classes)
    print(f"Injecting symmetric noise with ratio {noise_ratio}")
    print(f"Detected {num_classes} classes: {classes.tolist()}")

    # Choose indices to corrupt
    num_noisy_samples = int(num_samples * noise_ratio)
    noisy_indices = rng.choice(num_samples, size=num_noisy_samples, replace=False)

    # Original labels for those indices
    original_labels = data.loc[noisy_indices, 'label'].values

    # For each original label, pick a random *different* label
    # Generate random labels uniformly, then fix collisions
    random_labels = rng.choice(classes, size=num_noisy_samples, replace=True)

    # Ensure new label != original label
    same_mask = random_labels == original_labels
    while np.any(same_mask):
        # regenerate only for positions where label matched original
        random_labels[same_mask] = rng.choice(classes, size=same_mask.sum(), replace=True)
        same_mask = random_labels == original_labels

    # Apply noisy labels
    data.loc[noisy_indices, 'label'] = random_labels

    print(f"Injected noise into {num_noisy_samples} samples.")
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
    return np.array([orig != new for orig, new in zip(original, noisy)], dtype=bool)

import numpy as np
import matplotlib.pyplot as plt
import os

def save_orig_noisy_loss_histogram(noisy_mask, raw_losses, epoch, model):
    """
    Save a histogram comparing clean vs noisy sample losses for a given epoch.

    Args:
        noisy_mask (array-like): Boolean array of shape [num_samples], True for noisy samples.
        raw_losses (array-like): Per-sample losses (unnormalized or normalized), shape [num_samples].
        epoch (int): Current epoch number.
        model (int): Model index (1 or 2) to label the plot.
        save_dir (str): Directory to save the plot.
    """
    raw_losses = np.array(raw_losses)
    noisy_mask = np.array(noisy_mask)
    clean_losses = raw_losses[~noisy_mask]
    noisy_losses = raw_losses[noisy_mask]

    plt.figure(figsize=(8, 5))
    bins = 100
    plt.hist(clean_losses, bins=bins, density=True, alpha=0.5, label="Clean", color="blue")
    plt.hist(noisy_losses, bins=bins, density=True, alpha=0.5, label="Noisy", color="red")
    plt.xlabel("Normalized loss")
    plt.ylabel("Empirical pdf")
    plt.title(f"Epoch {epoch}: Loss Distribution (Model {model})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"src/data/images/loss_no_lu/clean_noisy_loss_{epoch}_model_{model}.png")
    plt.close()

def load_yahoo_train(file, file_path):
    path = file_path + file
    print(f"Loading Yahoo train data from {path}")
    data = pd.read_csv(path, sep=',', header=0)
    data.fillna('', inplace=True)
    data['label'] = data['label'].astype(int)
    data['label'] -= 1
    data = data[['text', 'label']]
    print(f"Yahoo train data loaded with {len(data)} samples")
    # Take half the data for training with equal distribution
    data = data.groupby('label').apply(lambda x: x.sample(frac=0.03, random_state=42)).reset_index(drop=True)
    print(f"Reduced Yahoo train data to {len(data)} samples")
    return data

def load_yahoo_test(file, file_path):
    path = file_path + file
    print(f"Loading Yahoo tratestin data from {path}")
    data = pd.read_csv(path, sep=',', header=0)
    data.fillna('', inplace=True)
    data['label'] = data['label'].astype(int)
    data['label'] -= 1
    data = data[['text', 'label']]
    print(f"Yahoo test data loaded with {len(data)} samples")
    return data