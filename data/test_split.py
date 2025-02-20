import pandas as pd

df = pd.read_csv('/home/marcuswrrn/Projects/Federated-Learning/data/train.csv')

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Group by label
grouped = df.groupby('Activity')

# Determine the minimum count for a balanced split
min_count_per_label = grouped.size().min()
samples_per_split = min_count_per_label // 10

# Create 10 balanced datasets
datasets = []
for i in range(10):
    subset = pd.concat([group.sample(n=samples_per_split, random_state=i) for _, group in grouped])
    datasets.append(subset.sample(frac=1, random_state=i).reset_index(drop=True))  # Shuffle each subset

# Example: Accessing the first dataset
print(datasets[0].head())

for i, subset in enumerate(datasets):
    filename = f"dataset_{i+1}.csv"
    subset.to_csv(filename, index=False)
    print(f"Saved {filename}")