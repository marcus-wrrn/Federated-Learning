import pandas as pd

df = pd.read_csv('Federated-Learning/data/train.csv')
for category, group in df.groupby('subject'):
    group.to_csv('{}.csv'.format(category), index=False)