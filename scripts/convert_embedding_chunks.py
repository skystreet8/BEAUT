from argparse import ArgumentParser
import torch
import os
import re


parser = ArgumentParser()
parser.add_argument('-f', '--fasta',  type=str, required=True,
                    help='The name of your FASTA file which will be used for naming the embedding file.')
parser.add_argument('--multiple', action='store_true', default=False, help='Whether converting multiple sequences.')
args = parser.parse_args()
if args.multiple:
    datadir = '../data/'
else:
    datadir = '../data/case_embeddings/'
if args.multiple:
    all_chunks = {}
    files = [f for f in os.listdir(datadir) if re.match(r'embeddings_\d+\.pt', f)]
    for file in files:
        chunk = torch.load(os.path.join(datadir, file))
        chunk = {k: chunk[k]['mean_representations'][33] for k in chunk}
        for k in chunk:
            if k not in all_chunks:
                all_chunks[k] = chunk[k]
            else:
                raise KeyError(f'Why duplicated headers? Header "{k}"')
    torch.save(all_chunks, f'../data/{args.fasta}_embeddings.pt')
    for file in files:
        os.remove(os.path.join(datadir, file))
else:
    chunk = torch.load(os.path.join(datadir, 'embeddings_0.pt'))
    chunk = chunk[args.fasta]
    torch.save(chunk, os.path.join(datadir, f'{args.fasta}.pt'))
    os.remove(os.path.join(datadir, 'embeddings_0.pt'))
