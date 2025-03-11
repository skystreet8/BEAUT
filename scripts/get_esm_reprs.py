import torch
import pandas as pd
from tqdm import tqdm

pos_embeddings = torch.load('../data/positive_seqs_v2_substrate_pocket_aug_embeddings.pt')
neg_embeddings = torch.load('../data/negative_seqs_v2_embeddings.pt')

dataset = pd.read_csv('../data/sequence_dataset_v3_substrate_pocket_aug_eq_len_dist.csv')
headers = dataset['header'].values.tolist()
data = {}
for k in tqdm(headers):
    if k in pos_embeddings:
        data[k] = pos_embeddings[k]
    elif k in neg_embeddings:
        data[k] = neg_embeddings[k]
    else:
        print(k)
        raise KeyError
torch.save(data, '../data/seq_embeddings_v3_substrate_pocket_aug_eq_len_dist.pt')
