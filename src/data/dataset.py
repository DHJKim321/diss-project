from torch.utils.data import Dataset
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (pd.DataFrame): A DataFrame containing the dataset with 'text' and 'class' columns.
        """
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data['text'][idx], self.data['class'][idx]