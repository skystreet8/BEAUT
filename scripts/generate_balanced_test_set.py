import pandas as pd
import torch
import random

random.seed(1001)
df = pd.read_csv('../data/sequence_dataset_v3_substrate_pocket_aug.csv')
test_df = df.query('dataset_fold_1 == "test"').copy()
for i in range(1, 6):
    del test_df[f'dataset_fold_{i}']
neg_neg_blast = pd.read_csv('../data/test_set_neg_against_non_test_neg_blast.tsv', sep='\t',
                            names=['query', 'target', 'fident', 'alnlen', 'mismatch', 'gapopen', 'qstart', 'qend',
                                   'tstart', 'tend', 'e_value', 'bits'])
neg_neg_blast.sort_values('fident', ascending=False, inplace=True)
neg_neg_blast.drop_duplicates('query', inplace=True)

neg_neg_blast = neg_neg_blast.query('fident >= 30').copy()
high_id_hs = set(neg_neg_blast['query'].values.tolist())

pos_indexes = test_df.query('label == 1').index.values.tolist()
neg_indexes = test_df.query('label == 0 and header not in @high_id_hs').index.values.tolist()
random.shuffle(neg_indexes)
num_neg_samples = int(5 * len(pos_indexes))
selected_neg_indexes = random.sample(neg_indexes, num_neg_samples)
selected_indexes = pos_indexes + selected_neg_indexes
sub_df = test_df.loc[selected_indexes].copy()
sub_df.to_csv(f'../data/test_set_balanced.csv', index=False)
embeddings = torch.load('../data/seq_embeddings_v3_substrate_pocket_aug.pt')

data = {}
headers = sub_df['header'].values.tolist()
for k in headers:
    if k in embeddings:
        data[k] = embeddings[k]
    else:
        print(k)
        raise KeyError
torch.save(data, f'../data/test_set_balanced_embeddings.pt')
