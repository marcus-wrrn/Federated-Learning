import pandas as pd
import numpy as np
import argparse
import os

def split_data_normal(train_path: str, out_dir: str, num_splits: int = 10, std_dev: float = 0.8):
    df = pd.read_csv(train_path)
    total_samples = len(df)
    avg_samples_per_split = total_samples / num_splits
    std_dev = avg_samples_per_split * std_dev 
    
    # Generate the number of samples for each split following a normal distribution
    split_sizes = np.random.normal(loc=avg_samples_per_split, scale=std_dev, size=num_splits).astype(int)
    split_sizes = np.clip(split_sizes, 1, None)  # Ensure at least one sample per split
    
    # Adjust split sizes to ensure they sum to total_samples
    split_sizes = (split_sizes / split_sizes.sum() * total_samples).astype(int)
    diff = total_samples - split_sizes.sum()
    split_sizes[0] += diff  # Adjust first split to account for rounding errors
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle data
    
    os.makedirs(out_dir, exist_ok=True)
    start_idx = 0
    
    for i, size in enumerate(split_sizes):
        subset = df.iloc[start_idx:start_idx + size]
        start_idx += size
        path = os.path.join(out_dir, f"dataset_{i+1}.csv")
        subset.to_csv(path, index=False)
        print(f"Saved file to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./train.csv", help="Path to input CSV file")
    parser.add_argument("--out", type=str, default="./", help="Output directory")
    parser.add_argument("--splits", type=int, default=10, help="Number of splits")
    parser.add_argument("--std_dev", type=float, default=0.8, help="Standard Deviation")
    args = parser.parse_args()
    
    split_data_normal(args.data, args.out, args.splits, args.std_dev)