import pandas as pd
import numpy as np
import argparse
import os


def split_data(train_path: str, out_dir: str):
    df = pd.read_csv(train_path)
    grouped = df.groupby('subject')


    for i, (_, group) in enumerate(grouped):
        path = os.path.join(out_dir, f"dataset_{i}.csv")
        group.to_csv(path, index=False)
        print(f"Saved file to {path}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./train.csv")
    parser.add_argument("--out", type=str, help="Output directory", default="./")
    args = parser.parse_args()

    split_data(args.data, args.out)


