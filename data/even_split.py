import pandas as pd
import numpy as np
import argparse
import os

def split_data(train_path: str, out_dir: str, num_splits: int = 10):
    df = pd.read_csv(train_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle data
    grouped = df.groupby('Activity')
    
    # Initialize empty DataFrames for each split
    datasets = [pd.DataFrame(columns=df.columns) for _ in range(num_splits)]
    
    # Distribute data evenly across splits
    for _, group in grouped:
        splits = np.array_split(group, num_splits)
        for i in range(num_splits):
            datasets[i] = pd.concat([datasets[i], splits[i]])
    
    # Shuffle each dataset again
    datasets = [subset.sample(frac=1, random_state=i).reset_index(drop=True) for i, subset in enumerate(datasets)]
    
    # Save each dataset as a CSV file
    os.makedirs(out_dir, exist_ok=True)
    for i, subset in enumerate(datasets):
        path = os.path.join(out_dir, f"dataset_{i+1}.csv")
        subset.to_csv(path, index=False)
        print(f"Saved file to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./train.csv", help="Path to input CSV file")
    parser.add_argument("--out", type=str, default="./", help="Output directory")
    parser.add_argument("--splits", type=int, default=10, help="Number of splits")
    args = parser.parse_args()
    
    split_data(args.data, args.out, args.splits)
