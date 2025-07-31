import pandas as pd
import os, sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

    num_noisy_samples = int(num_samples * noise_ratio)
    noisy_indices = rng.choice(num_samples, size=num_noisy_samples, replace=False)
    original_labels = data.loc[noisy_indices, 'label'].values
    random_labels = rng.choice(classes, size=num_noisy_samples, replace=True)
    same_mask = random_labels == original_labels
    while np.any(same_mask):
        random_labels[same_mask] = rng.choice(classes, size=same_mask.sum(), replace=True)
        same_mask = random_labels == original_labels

    data.loc[noisy_indices, 'label'] = random_labels

    print(f"Injected noise into {num_noisy_samples} samples.")
    return data

def get_labels_injected_list(original, noisy):
    return np.array([orig != new for orig, new in zip(original, noisy)], dtype=bool)

def save_orig_noisy_loss_histogram(noisy_mask, raw_losses, epoch, model):
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
    plt.title(f"Epoch {epoch+1}: Loss Distribution (Model {model})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"src/data/images/loss_agnews_noise=0.7/clean_noisy_loss_{epoch+1}_model_{model}.png")
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
    data = data.groupby('label').apply(lambda x: x.sample(frac=1/30, random_state=42)).reset_index(drop=True)
    print(f"Reduced Yahoo train data to {len(data)} samples")
    return data

def load_yahoo_test(file, file_path):
    path = file_path + file
    print(f"Loading Yahoo test data from {path}")
    data = pd.read_csv(path, sep=',', header=0)
    data.fillna('', inplace=True)
    data['label'] = data['label'].astype(int)
    data['label'] -= 1
    data = data[['text', 'label']]
    print(f"Yahoo test data loaded with {len(data)} samples")
    return data

def load_agnews_train(file, file_path):
    path = file_path + file
    print(f"Loading AG News train data from {path}")
    data = pd.read_csv(path, sep=',', header=0)
    data.fillna('', inplace=True)
    data['text'] = data['Title'].astype(str) + ' ' + data['Description'].astype(str)
    data['label'] = data['Class Index'].astype(int) - 1
    data = data[['text', 'label']]
    print(f"AG News train data loaded with {len(data)} samples")
    # Take half the data for training with equal distribution
    data = data.groupby('label').apply(lambda x: x.sample(frac=1/3, random_state=42)).reset_index(drop=True)
    print(f"Reduced AG News train data to {len(data)} samples")
    return data

def load_agnews_test(file, file_path):
    path = file_path + file
    print(f"Loading AG News test data from {path}")
    data = pd.read_csv(path, sep=',', header=0)
    data.fillna('', inplace=True)
    data['text'] = data['Title'].astype(str) + ' ' + data['Description'].astype(str)
    data['label'] = data['Class Index'].astype(int) - 1
    data = data[['text', 'label']]
    print(f"AG News test data loaded with {len(data)} samples")
    return data

def load_data(train_file, train_data_path, dataset, noise_ratio, test_file=None, test_data_path=None):
    if dataset == 'imdb':
        imdb_data = load_imdb_data(train_file, train_data_path)
        train_data, test_data = train_test_split(imdb_data, test_size=0.2, random_state=42)
        original_labels = train_data['label'].values
        train_data = inject_symmetric_noise(train_data, noise_ratio=noise_ratio)
        noisy_mask = get_labels_injected_list(original_labels, train_data['label'].values)
        num_classes = 2
    elif dataset == 'yahoo':
        train_data = load_yahoo_train(train_file, train_data_path)
        test_data = load_yahoo_test(test_file, test_data_path)
        original_labels = train_data['label'].values
        train_data = inject_symmetric_noise(train_data, noise_ratio=noise_ratio)
        noisy_mask = get_labels_injected_list(original_labels, train_data['label'].values)
        num_classes = train_data['label'].nunique()
        print(f"Number of classes in Yahoo dataset: {num_classes}")
    elif dataset == 'agnews':
        train_data = load_agnews_train(train_file, train_data_path)
        test_data = load_agnews_test(test_file, test_data_path)
        original_labels = train_data['label'].values
        train_data = inject_symmetric_noise(train_data, noise_ratio=noise_ratio)
        noisy_mask = get_labels_injected_list(original_labels, train_data['label'].values)
        num_classes = train_data['label'].nunique()
        print(f"Number of classes in AGNews dataset: {num_classes}")
    else:
        print("Loading ShaPe Data")
        train_data = load_full_data(train_file, train_data_path)
        test_data = load_test_data(test_file, test_data_path)
        num_classes = 2
        noisy_mask = None
    return train_data, test_data, noisy_mask, num_classes

def load_mixtext_train(file, data_path):
    path = data_path + file
    print(f"Loading AG News train data from {path} (MixText)")
    data = pd.read_csv(path, sep=',', header=0)
    data.fillna('', inplace=True)
    data['text'] = data['Description'].astype(str)
    data['label'] = data['Class Index'].astype(int) - 1
    data = data[['text', 'label']]
    print(f"AG News train data loaded with {len(data)} samples")
    return data

def load_mixtext_test(file, data_path):
    path = data_path + file
    print(f"Loading AG News test data from {path} (MixText)")
    data = pd.read_csv(path, sep=',', header=0)
    data.fillna('', inplace=True)
    data['text'] = data['Description'].astype(str)
    data['label'] = data['Class Index'].astype(int) - 1
    data = data[['text', 'label']]
    print(f"AG News test data loaded with {len(data)} samples")
    return data

def train_val_split(labels, n_labeled_per_class, n_labels, seed=0):
    np.random.seed(seed)
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(n_labels):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_pool = np.concatenate((idxs[:500], idxs[5500:-2000]))
        train_labeled_idxs.extend(train_pool[:n_labeled_per_class])
        train_unlabeled_idxs.extend(
            idxs[500: 500 + 5000])
        val_idxs.extend(idxs[-2000:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs

def get_mixtext_data_idx(train_file, train_data_path, test_file, test_data_path, n_labelled_per_class=20):
    train_data = load_mixtext_train(train_file, train_data_path)
    test_data = load_mixtext_test(test_file, test_data_path)
    train_labelled_idxs, train_unlabelled_idxs, val_idxs = train_val_split(
        train_data['label'], n_labelled_per_class, len(train_data['label'].unique()), seed=0)
    num_classes = len(train_data['label'].unique())
    return train_data, test_data, train_labelled_idxs, train_unlabelled_idxs, val_idxs, num_classes