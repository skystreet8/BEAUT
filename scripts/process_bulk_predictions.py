import pandas as pd
import pickle
from utils import ReadFastaFile, SaveFastaFile

result_file = '../data/PRJNA28331_aug/PRJNA28331_Genbank_results_BEAUT_aug.pkl'
seq2organism = pickle.load(open('../data/PRJNA28331_Genbank_seq2organism.pkl', 'rb'))
results = pickle.load(open(result_file, 'rb'))
headers, seqs = ReadFastaFile('../data/PRJNA28331_Genbank_filtered_proteins.fasta')
scores = [results[h][1] for h in headers]
organisms = [list(seq2organism[s]) for s in seqs]
for t in organisms:
    t.sort()
df = pd.DataFrame({'header': headers, 'seq': seqs, 'pred_score': scores, 'organism': organisms})
df.sort_values('pred_score', ascending=False, inplace=True)
pos_df = df.query('pred_score >= 0.5').copy()
print(f'{len(pos_df)} sequences were predicted to be positive.')
for i in range(1, int(len(pos_df) // 100000) + 2):
    sub_df = pos_df.iloc[(i - 1) * 100000:i * 100000]
    sub_df.to_csv(f'../data/PRJNA28331_aug/PRJNA28331_filtered_proteins_positive_results_aug_pt{i}.csv', index=False)
    sub_headers, sub_seqs = sub_df['header'].values.tolist(), sub_df['seq'].values.tolist()
    SaveFastaFile(f'../data/PRJNA28331_aug/PRJNA28331_filtered_proteins_positive_results_aug_pt{i}.fasta', sub_headers, sub_seqs)
