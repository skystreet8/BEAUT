import pandas as pd
import torch
import random


df = pd.read_csv('../data/sequence_dataset_v3_substrate_pocket_aug.csv')
test_df = df.query('dataset_fold_1 == "test"').copy()
for i in range(1, 6):
    del test_df[f'dataset_fold_{i}']
pos_indexes = test_df.query('label == 1').index.values.tolist()
neg_indexes = test_df.query('label == 0').index.values.tolist()
random.shuffle(neg_indexes)
num_neg_samples = int(5 * len(pos_indexes))
selected_indexes = [pos_indexes + neg_indexes[i * num_neg_samples:(i + 1) * num_neg_samples] for i in range(5)]
sub_dfs = [test_df.loc[indexes].copy() for indexes in selected_indexes]
for i in range(1, 6):
    sub_dfs[i - 1].to_csv(f'../data/test_set_sample_{i}.csv', index=False)
embeddings = torch.load('../data/seq_embeddings_v3_substrate_pocket_aug.pt')
for i in range(5):
    data = {}
    headers = sub_dfs[i]['header'].values.tolist()
    for k in headers:
        if k in embeddings:
            data[k] = embeddings[k]
        else:
            print(k)
            raise KeyError
    torch.save(data, f'../data/test_set_sample_{i + 1}_embeddings.pt')
