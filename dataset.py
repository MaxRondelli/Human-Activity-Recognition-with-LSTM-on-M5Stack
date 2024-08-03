import torch
from torch.utils.data import Dataset
import numpy as np

# Import constants and functions from your data.py
from data import (
    X_train, X_test, y_train, y_test, 
    LABELS, n_steps, n_input, n_classes
)
class HARDataset(Dataset):
    def __init__(self, X, y, class_mapping):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y.reshape(-1))
        self.class_mapping = class_mapping

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data():
    # Create class mapping
    class_mapping = {label: i for i, label in enumerate(LABELS)}
    
    # Reshape X_train and X_test to match PyTorch expected input
    X_train_reshaped = X_train.transpose(0, 2, 1) 
    X_test_reshaped = X_test.transpose(0, 2, 1)
    aza
    # Create datasets
    train_dataset = HARDataset(X_train_reshaped, y_train, class_mapping)
    test_dataset = HARDataset(X_test_reshaped, y_test, class_mapping)
    
    return train_dataset, test_dataset, class_mapping