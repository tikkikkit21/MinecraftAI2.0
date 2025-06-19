import torch
import torch.nn as nn
from torch.utils.data import Dataset

class PoseDataset(Dataset):
    def __init__(self, df, label_encoder):
        self.X = torch.tensor(df.drop(columns=['label']).values, dtype=torch.float32)
        labels = df['label'].values
        self.y = torch.tensor(label_encoder.transform(labels), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PoseClassifier(nn.Module):
    def __init__(self, input_dim=70, num_classes=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)
