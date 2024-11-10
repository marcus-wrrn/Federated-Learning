from dataclasses import dataclass
import torch


@dataclass
class HARSConfig:
    """Config object which contains all metadata + hyperparameter information for training the HARS model"""
    train_path: str
    test_path: str
    epochs: str
    batch_size: str
    device: torch.device

