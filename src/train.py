import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import argparse
from data_handling.loader_files import HARSConfig, HARSLog
from models.basic import HARSModel
from data_handling.datasets import HARSDataset



def main(config: HARSConfig):
    model = HARSModel(config.device)
    logger = HARSLog(config)

    train_data = HARSDataset(config.train_path)
    test_data = HARSDataset(config.test_path)

    train_loader = DataLoader(train_data, config.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = StepLR(optimizer, gamma=0.1, step_size=10)

    for epoch in range(config.epochs):
        print(f"Epoch: {epoch + 1}")
        train_loss = model.fit(train_loader, optimizer)
        with torch.no_grad():
            print("Validating")
            val_loss = model.fit(test_loader, optimizer, train=False)
            print(f"Validation: {val_loss}")
        logger.update_results(train_loss, val_loss)
        scheduler.step()


    logger.save_log(model)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training pipeline for validation model")
    parser.add_argument("-train_path", type=str, default="/home/marcuswrrn/Projects/Capstone/model_training/data/train.csv", help="Filepath for train file")
    parser.add_argument("-test_path", type=str, default="/home/marcuswrrn/Projects/Capstone/model_training/data/test.csv", help="Filepath for test file")
    parser.add_argument("-save", type=str, default=None)
    parser.add_argument("-e", type=int, default=5, help="Number of epochs")
    parser.add_argument("-b", type=int, default=100, help="Batch size")
    parser.add_argument("-lr", type=float, default=0.001, help="Learning Rate")
    parser.add_argument("-cuda", type=str, default="y", help="Use cuda Y/n")

    args = parser.parse_args()

    # Initiatlize device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda.lower() == 'y' else "cpu")
    
    # Create config object
    cfg = HARSConfig(
        train_path=args.train_path, 
        test_path=args.test_path, 
        save_path=args.save,
        epochs=args.e, 
        batch_size=args.b,
        learning_rate=args.lr,
        device=device
    )

    main(cfg)

