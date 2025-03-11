from argparse import ArgumentParser
import torch
import pandas as pd
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument('-m', '--model', required=True)
args = parser.parse_args()

pos_embeddings = torch.load('../data/positive_seqs_v2_substrate_pocket_aug_embeddings.pt')
neg_embeddings = torch.load('../data/negative_seqs_v2_embeddings.pt')

if args.model == 'base':
    dataset = pd.read_csv('../data/sequence_dataset_v3.csv')
elif args.model == 'aug':
    dataset = pd.read_csv('../data/sequence_dataset_v3_substrate_pocket_aug_eq_len_dist.csv')
else:
    raise NotImplementedError()
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
if args.model == 'base':
    torch.save(data, '../data/seq_embeddings_v3.pt')
elif args.model == 'aug':
    torch.save(data, '../data/seq_embeddings_v3_substrate_pocket_aug_eq_len_dist.pt')
