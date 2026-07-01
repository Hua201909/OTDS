import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch

class WaterQualityDataset(Dataset):
    def __init__(self, features, labels, seq_length):
        self.features = features
        self.labels = labels
        self.seq_length = seq_length
    def __len__(self):
        return len(self.features) - self.seq_length
    def __getitem__(self, idx):
        x = self.features[idx:idx+self.seq_length]
        y = self.labels[idx+self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def load_data(file_path, scaler_features=None, label_mean=None, label_std=None, is_train=True):
    data = pd.read_excel(file_path).dropna()
    features_df = data[['T', 'PH', 'C', 'NTU',
                        'TN']]
    labels_series = data['TN']
    features_np = features_df.values.astype(np.float32)
    labels_np = labels_series.values.astype(np.float32)

    if is_train:
        scaler_features = StandardScaler()
        features_scaled = scaler_features.fit_transform(features_np)
        label_mean = labels_np.mean()
        label_std = labels_np.std()
        if label_std == 0:
            label_std = 1
        labels_scaled = (labels_np - label_mean) / label_std
    else:
        features_scaled = scaler_features.transform(features_np)
        labels_scaled = (labels_np - label_mean) / label_std
    return features_scaled, labels_scaled, scaler_features, label_mean, label_std, data