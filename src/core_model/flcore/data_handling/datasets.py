import pandas as pd
from torch.utils.data import Dataset
from torch import Tensor
import torch

class HARSDataset(Dataset):
    def __init__(self, filepath: str) -> None:
        super().__init__()

        self.data = pd.read_csv(filepath)
        self.features = torch.tensor(self.data.values[:, :-2].astype(float)).float()

        # Map label strings to integer values
        self.labels = torch.tensor([self._map_label(label) for label in self.data.values[:, -1]])
        self.labels = torch.nn.functional.one_hot(self.labels, num_classes=6).float()
        self.user = self.data.values[:, -2]
        self.class_list = [0,1,2,3,4,5]
    
    def _map_label(self, label: str) -> int:
        label_mapping = {
            'WALKING_UPSTAIRS': 0,
            'WALKING_DOWNSTAIRS': 1,
            'STANDING': 2,
            'LAYING': 3,
            'WALKING': 4,
            'SITTING': 5
        }
        
        value = label_mapping.get(label, -1)
        assert value >= 0

        return value
    
    def __len__(self) -> int:
        return self.features.shape[0]
    
    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        features = self.features[index]
        labels = self.labels[index]

        return (features, labels)
