from argparse import ArgumentParser
import pandas as pd
import pickle
from utils import ReadFastaFile, SaveFastaFile


parser = ArgumentParser()
parser.add_argument('-m', '--model', required=True, help='Processing Base / Aug model predictions.')
args = parser.parse_args()
if args.model == 'aug':
    result_file = '../data/PRJNA28331_aug/PRJNA28331_results_ba_pred_DNN_aug.pkl'
    seq2organism = pickle.load(open('../data/PRJNA28331_seq2organism.pkl', 'rb'))
    results = pickle.load(open(result_file, 'rb'))
    headers, seqs = ReadFastaFile('../data/PRJNA28331_filtered_proteins.fasta')
    scores = [results[h] for h in headers]
    organisms = [list(seq2organism[s]) for s in seqs]
    for t in organisms:
        t.sort()
    df = pd.DataFrame({'header': headers, 'seq': seqs, 'pred_score': scores, 'organism': organisms})
    df.sort_values('pred_score', ascending=False, inplace=True)
    pos_df = df.query('pred_score >= 0.8').copy()
    print(f'{len(pos_df)} sequences have prediction scores >= 0.8.')
    for i in range(1, 11):
        sub_df = pos_df.iloc[(i - 1) * 100000:i * 100000]
        sub_df.to_csv(f'../data/PRJNA28331_aug/PRJNA28331_filtered_proteins_positive_results_aug_pt{i}.csv', index=False)
        sub_headers, sub_seqs = sub_df['header'].values.tolist(), sub_df['seq'].values.tolist()
        SaveFastaFile(f'../data/PRJNA28331_aug/PRJNA28331_filtered_proteins_positive_results_aug_pt{i}.fasta', sub_headers, sub_seqs)
elif args.model == 'base':
    result_file = '../data/PRJNA28331_base/PRJNA28331_results_ba_pred_DNN_base.pkl'
    seq2organism = pickle.load(open('../data/PRJNA28331_seq2organism.pkl', 'rb'))
    results = pickle.load(open(result_file, 'rb'))
    headers, seqs = ReadFastaFile('../data/PRJNA28331_filtered_proteins.fasta')
    scores = [results[h] for h in headers]
    organisms = [list(seq2organism[s]) for s in seqs]
    for t in organisms:
        t.sort()
    df = pd.DataFrame({'header': headers, 'seq': seqs, 'pred_score': scores, 'organism': organisms})
    df['length'] = df['seq'].apply(len)
    df = df.query('157 <= length <= 1074').copy()
    del df['length']
    print(f'{len(df)} sequences have lengths between 157 to 1074.')
    df.sort_values('pred_score', ascending=False, inplace=True)
    pos_df = df.query('pred_score >= 0.9').copy()
    print(f'{len(pos_df)} sequences have prediction scores >= 0.9.')
    pos_df.to_csv('../data/PRJNA28331_base/PRJNA28331_filtered_proteins_positive_results_base.csv', index=False)
    pos_headers, pos_seqs = pos_df['header'].values.tolist(), pos_df['seq'].values.tolist()
    SaveFastaFile('../data/PRJNA28331_base/PRJNA28331_filtered_proteins_positive_results_base.fasta', pos_headers,
                  pos_seqs)
else:
    raise NotImplementedError('Argument should be "base" or "aug".')