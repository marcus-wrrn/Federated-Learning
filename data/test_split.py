import pandas as pd
import numpy as np

df = pd.read_csv('/home/marcuswrrn/Projects/Federated-Learning/data/train.csv')

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Group by label
grouped = df.groupby('Activity')

# Split data while ensuring no data loss
datasets = [pd.DataFrame(columns=df.columns) for _ in range(10)]

# Distribute data evenly across datasets
for _, group in grouped:
    splits = np.array_split(group, 10)  # Evenly split into 10 parts (some might be slightly larger)
    for i in range(10):
        datasets[i] = pd.concat([datasets[i], splits[i]])

# Shuffle each dataset again
datasets = [subset.sample(frac=1, random_state=i).reset_index(drop=True) for i, subset in enumerate(datasets)]

# Save each dataset as a CSV file
for i, subset in enumerate(datasets):
    filename = f"dataset_{i+1}.csv"
    subset.to_csv(filename, index=False)
    print(f"Saved {filename}")