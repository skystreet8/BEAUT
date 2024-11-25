import pandas as pd
from argparse import ArgumentParser
from ec_utils import eval_ec, sort_ecs


parser = ArgumentParser()
parser.add_argument('-m', '--model', required=True)
args = parser.parse_args()
if args.model == 'base':
    ec_df = pd.read_csv('../data/PRJNA28331_base/PRJNA28331_filtered_proteins_positive_results_base_filtered_by_annotation_maxsep.csv',
                       names=['header', 'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5'])
    ec_df.fillna('-', inplace=True)
    for i in range(1, 6):
        ec_df[f'pred_{i}'] = ec_df[f'pred_{i}'].apply(eval_ec)

    df = pd.read_csv('../data/PRJNA28331_base/PRJNA28331_filtered_proteins_positive_results_base_filtered_by_annotation.csv')
    ec_predictions = {t[1]: sort_ecs([t[i][0] for i in range(2, 6) if t[i][0] != 'NONE']) for t in ec_df.itertuples()}
    df = df.assign(clean_ec=[ec_predictions[t[1]] for t in df.itertuples()])
    df.to_csv('../data/PRJNA28331_base/PRJNA28331_base_final.csv', index=False)
elif args.model == 'aug':
    ec_dfs = []
    for i in range(1, 11):
        ec_df = pd.read_csv(
            f'../data/PRJNA28331_aug/PRJNA28331_filtered_proteins_positive_results_aug_filtered_by_annotation_pt{i}_maxsep.csv',
            names=['header', 'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5'])
        ec_df.fillna('-', inplace=True)
        for i in range(1, 6):
            ec_df[f'pred_{i}'] = ec_df[f'pred_{i}'].apply(eval_ec)
        ec_dfs.append(ec_df)
    merged_ec_df = pd.concat(ec_dfs, ignore_index=True)
    dfs = []
    for i in range(1, 11):
        df = pd.read_csv(
            f'../data/PRJNA28331_aug/PRJNA28331_filtered_proteins_positive_results_aug_filtered_by_annotation_pt{i}.csv')
        dfs.append(df)
    merged_df = pd.concat(dfs, ignore_index=True)
    ec_predictions = {t[1]: sort_ecs([t[i][0] for i in range(2, 6) if t[i][0] != 'NONE']) for t in
                      merged_ec_df.itertuples()}
    merged_df = merged_df.assign(clean_ec=[ec_predictions[t[1]] for t in merged_df.itertuples()])
    merged_df.to_csv(
        '../data/PRJNA28331_aug/PRJNA28331_aug_final.csv',
        index=False)
else:
    raise NotImplementedError()
