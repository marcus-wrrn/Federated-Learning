import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from data_handling.datasets import HARSDataset


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class HARSNet(nn.Module):
    """Basic model for the HARS dataset, will be expanded upon later"""
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(561, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2024),
            nn.ReLU(),
            nn.Linear(2024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)
    

class HARSModel(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.network = HARSNet()
        self.device = device

        self.to(self.device)

    def fit(self, data_load: DataLoader):
        """Function used to train the HARS model on a dataset"""
        for feat, label in data_load:
            feat: Tensor = feat.to(self.device)
            label: Tensor = label.to(self.device)

            logits: Tensor = self.network(feat)

            ...