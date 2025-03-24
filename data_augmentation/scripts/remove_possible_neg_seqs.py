import pandas as pd
from utils import *


orgs = ['A_Muc', 'B_Ang', 'B_Dor', 'C_You', 'E_Rec', 'R_Lac', 'V_Vad']

for o in orgs:
    df = pd.read_csv(f'../data/non_BA_transformers/{o}_against_neg_seqs_v2_blast.tsv', sep='\t',
                     names=['query', 'target', 'fident', 'alnlen', 'mismatch', 'gapopen', 'qstart',
                            'qend', 'tstart', 'tend', 'e_value', 'bits']
                     )
    df = df.query('bits >= 50').copy()
    headers, seqs = ReformatFastaFile(f'../data/non_BA_transformers/{o}.faa')
    h2s = {h: s for h, s in zip(headers, seqs)}
    s2h = {s: h for h, s in zip(headers, seqs)}
    seqs = [s for s in seqs if 157 <= len(s) <= 1074]
    headers = [s2h[s] for s in seqs]
    remove_hs = set(df['query'].values.tolist())
    headers = list(set(headers) - remove_hs)
    seqs = [h2s[h] for h in headers]
    SaveFastaFile(f'../data/non_BA_transformers/{o}_filtered_neg_seqs.fasta', headers, seqs)
