from dataclasses import dataclass
import torch
import logging
import os
from datetime import datetime
import json

@dataclass
class HARSConfig:
    """Config object which contains all metadata + hyperparameter information for training the HARS model"""
    train_path: str
    test_path: str
    save_path: str
    epochs: int
    batch_size: int
    learning_rate: float
    device: torch.device

    def to_json(self):
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "device": self.device.type
        }

class HARSLog:
    def __init__(self, config: HARSConfig, log_dir=None) -> None:
        if log_dir is None:
            now = datetime.now()
            log_dir = f"HARSModel{now.minute}{now.hour}{now.day}{now.month}{now.year}"
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.config = config
        self.log_file = os.path.join(self.log_dir, "loss_results.csv")
        self.model_filepath = os.path.join(self.log_dir, "model.pth")
        self.hyperparameter_log = os.path.join(self.log_dir, "config.json")

        self.train_loss = []
        self.val_loss = []
        self.loss = 0.0

    def update_results(self, loss: float, validation_loss: float):
        self.train_loss.append(loss)
        self.val_loss.append(validation_loss)
        self.loss = loss / len(self.train_loss)

    def save_log(self, model: torch.nn.Module):
        # Save model
        torch.save(model.state_dict(), self.model_filepath)

        with open(self.hyperparameter_log, "w") as fp:
            json.dump(self.config.to_json(), fp)

        # Save loss values
        with open(self.log_file, "w") as fp:
            fp.write("Train Loss, Validation Loss\n")
            for loss, vloss in zip(self.train_loss, self.val_loss):
                fp.write(f"{loss}, {vloss}\n")
