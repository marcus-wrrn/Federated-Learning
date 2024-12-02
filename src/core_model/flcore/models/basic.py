import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim import AdamW
from flcore.data_handling.datasets import HARSDataset


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
            nn.GELU(),
            nn.Linear(1024, 3000),
            nn.GELU(),
            nn.Linear(3000, 2000),
            nn.GELU(),
            nn.Linear(2000, 512),
            nn.GELU(),
            nn.Linear(512, 100),
            nn.GELU(),
            nn.Linear(100, 6)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)
    

class HARSModel(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.network = HARSNet()
        self.device = device

        self.to(self.device)

    def fit(self, data_load: DataLoader, optimizer: AdamW, train=True):
        """
        Function used to train the HARS model on a dataset

        Parameters:
        - data_load: DataLoader for the HARS Dataset
        - train: Boolean to determine backpropogation

        NOTE: use function with torch.no_grad when evaluating to save on device memory
        """
        criterion = nn.BCEWithLogitsLoss()

        total_loss = 0.0
        for i, (feat, label) in enumerate(data_load):
            if not i % 100 and i:
                print(f"Current loss: {total_loss / i}")

            feat: Tensor = feat.to(self.device)
            label: Tensor = label.to(self.device)

            logits: Tensor = self.network(feat)

            loss: Tensor = criterion(logits, label)

            if train: 
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()

        return total_loss / len(data_load)
    
    def forward(self, x: Tensor) -> Tensor:        
        return torch.sigmoid(self.network(x))