import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import AdamW
import io
import gzip
from logger import client_logger

class HARSNet(nn.Module):
    """Basic model for the HARS dataset"""
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
    """Full HARS Model: Contains functionallity for training and """
    def __init__(self, device: torch.device):
        super().__init__()
        self.network = HARSNet()
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.to(self.device)

    def fit(self, data_load: DataLoader, optimizer: AdamW, train=True,client_key=1):
        """
        Function used to train the HARS model on a dataset

        Parameters:
        - data_load: DataLoader for the HARS Dataset
        - train: Boolean to determine backpropogation

        NOTE: use function with torch.no_grad when evaluating to save on device memory
        """
        logger.info("Training client {}: ".format(client_key))
        total_loss = 0.0
        for i, (feat, label) in enumerate(data_load):
            if not i % 100 and i:
                logger.info("Current loss {}".format(total_loss/i))
                #print(f"Current loss: {total_loss / i}")

            feat: Tensor = feat.to(self.device)
            label: Tensor = label.to(self.device)

            logits: Tensor = self.network(feat)

            loss: Tensor = self.criterion(logits, label)

            if train: 
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()

        return total_loss / len(data_load)
    
    def forward(self, x: Tensor) -> Tensor:        
        return torch.sigmoid(self.network(x))
    

    # Methods for federated learning: 
    # TODO: Move to a parent class once we start working with other models
    def export_binary(self, compress=True) -> bytes:
        """
        Exports state dictionary into a compressed binary file
        """
        bytes_data = io.BytesIO()
        torch.save(self.state_dict(), bytes_data)

        # Set pointer to 0
        bytes_data.seek(0)

        if compress:
            return gzip.compress(bytes_data.getvalue())

        return bytes_data.getvalue()
    
    def import_binary(self, byte_data: bytes, decompress=True):
        """
        Imports a compressed state dictionary in binary format and loads it into the models current state dictionary
        """
        
        if decompress:
            byte_data = gzip.decompress(byte_data)
        byte_data = io.BytesIO(byte_data)
        byte_data.seek(0)

        state_dict = torch.load(byte_data, weights_only=True)
        self.load_state_dict(state_dict)

