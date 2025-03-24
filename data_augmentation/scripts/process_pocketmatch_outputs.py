import pandas as pd
from read_pmscore_utils import *
import os
from utils import *
from collections import Counter
from tqdm import tqdm


pos_headers, pos_seqs = ReadFastaFile('../../data/positive_seqs_v3_unique.fasta')
datadir = '../PocketMatch/results'
fasta_dir = '../data/non_BA_transformers'
orgs = ['A_Muc', 'B_Ang', 'B_Dor', 'C_You', 'E_Rec', 'R_Lac', 'V_Vad']
# organisms = ['B_Adolescentis', 'B_Xylanisolvens', 'C_Comes', 'C_M62_1', 'H_Filiformis', 'R_Gnavus', 'S_Infantarius']
dfs = [read_txt(os.path.join(datadir, f'{o}_default_against_pos.txt'), format_q='genbank') for o in tqdm(orgs)]
for df in dfs:
    df.sort_values('P-max_OP', ascending=False, inplace=True)
aug_headers = []
aug_seqs = []
default_qdfs = []
for i, df in tqdm(enumerate(dfs), total=len(dfs)):
    o = orgs[i]
    qdf = df.query('`P-max_OP` >= 0.7').copy()
    default_qdfs.append(qdf)
    counter = Counter(qdf['header'].values.tolist())
    this_aug_headers = [t[0] for t in counter.most_common()]
    headers, seqs = ReadFastaFile(os.path.join(fasta_dir, f'{o}_filtered.fasta'))
    header2seq = {k: v for k, v in zip(headers, seqs)}
    seq2header = {k: v for k, v in zip(seqs, headers)}
    assert len(header2seq) == len(seq2header)
    this_aug_seqs = [header2seq[h] for h in this_aug_headers]
    this_aug_seqs = list(set(this_aug_seqs) - set(pos_seqs))  # Remove existing sequences in positive seqs
    this_aug_headers = [seq2header[s] for s in this_aug_seqs]

    aug_headers.extend(this_aug_headers)
    aug_seqs.extend(this_aug_seqs)
default_qdf_merged = pd.concat(default_qdfs, ignore_index=True)
default_qdf_merged.to_csv('../data/non_BA_transformers_default_matched_pockets_against_pos.csv', index=False)

dfs = [read_txt(os.path.join(datadir, f'{o}_rescue_against_pos.txt'), format_q='genbank') for o in tqdm(orgs)]
rescue_qdfs = []
for df in dfs:
    df.sort_values('P-max_OP', ascending=False, inplace=True)
for i, df in tqdm(enumerate(dfs), total=len(dfs)):
    o = orgs[i]
    qdf = df.query('`P-max_OP` >= 0.7').copy()
    rescue_qdfs.append(qdf)
    counter = Counter(qdf['header'].values.tolist())
    this_aug_headers = [t[0] for t in counter.most_common()]
    this_aug_headers = list(set(this_aug_headers) - set(aug_headers))
    headers, seqs = ReadFastaFile(os.path.join(fasta_dir, f'{o}_filtered.fasta'))
    header2seq = {k: v for k, v in zip(headers, seqs)}
    seq2header = {k: v for k, v in zip(seqs, headers)}
    assert len(header2seq) == len(seq2header)
    this_aug_seqs = [header2seq[h] for h in this_aug_headers]
    this_aug_seqs = list(set(this_aug_seqs) - set(pos_seqs))  # Remove existing sequences in positive seqs
    this_aug_headers = [seq2header[s] for s in this_aug_seqs]

    aug_headers.extend(this_aug_headers)
    aug_seqs.extend(this_aug_seqs)
rescue_qdf_merged = pd.concat(rescue_qdfs, ignore_index=True)
rescue_qdf_merged.to_csv('../data/non_BA_transformers_rescue_matched_pockets_against_pos.csv', index=False)
SaveFastaFile('../data/substrate_pocket_sim_aug_v3_from_non_BA_transformers.fasta', aug_headers, aug_seqs)
