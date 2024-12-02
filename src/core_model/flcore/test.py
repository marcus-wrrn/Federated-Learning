import argparse
import os
import torch
from torch import Tensor
from models.basic import HARSModel
from data_handling.datasets import HARSDataset


def validate_model(model_path: str, data_path: str, device: torch.device):
    dataset = HARSDataset(data_path)
    model = HARSModel(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()

    num_correct = 0
    for (feat, ground_truth) in dataset:
        feat: Tensor = feat.to(device)
        ground_truth: int = ground_truth.argmax().item()

        pred_label: Tensor = model(feat)
        pred_label = pred_label.argmax().item()
        
        if pred_label == ground_truth:
            num_correct += 1
            
    print(f"Accuracy: {num_correct/len(dataset)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type=str, default="/home/marcuswrrn/Projects/Capstone/models/HARSModel151517112024")
    parser.add_argument("-data", type=str, default="/home/marcuswrrn/Projects/Capstone/data/test.csv")
    parser.add_argument("-cuda", type=str, help="Use Cuda: Y/n", default="Y")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda.lower() == 'y' else "cpu")

    validate_model(os.path.join(args.path, "model.pth"), args.data, device)
