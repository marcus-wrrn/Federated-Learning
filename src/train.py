import torch
from torch.utils.data import DataLoader
import argparse
from data_handling.loader_files import HARSConfig
from models.basic import HARSModel
from data_handling.datasets import HARSDataset



def main(config: HARSConfig):
    model = HARSModel(config.device)

    train_data = HARSDataset(config.train_path)
    test_data = HARSDataset(config.test_path)

    train_loader = DataLoader(train_data, config.batch_size, shuffle=True)

    model.fit(train_loader)
    ...






if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training pipeline for validation model")
    parser.add_argument("-train_path", type=str, default="/home/marcuswrrn/Projects/Capstone/model_training/data/train.csv", help="Filepath for train file")
    parser.add_argument("-test_path", type=str, default="/home/marcuswrrn/Projects/Capstone/model_training/data/test.csv", help="Filepath for test file")
    parser.add_argument("-e", type=int, default=5, help="Number of epochs")
    parser.add_argument("-b", type=int, default=10, help="Batch size")
    parser.add_argument("-cuda", type=str, default="y", help="Use cuda Y/n")

    args = parser.parse_args()

    # Initiatlize device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda.lower() == 'y' else "cpu")
    
    # Create config object
    cfg = HARSConfig(
        args.train_path, 
        args.test_path, 
        args.e, 
        args.b,
        device
    )

    main(cfg)

