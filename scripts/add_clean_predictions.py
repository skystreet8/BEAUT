import pandas as pd
from ec_utils import eval_ec, sort_ecs

ec_dfs = []
for i in range(1, 15):
    ec_df = pd.read_csv(
        f'../data/PRJNA28331_aug/PRJNA28331_filtered_proteins_positive_results_aug_filtered_by_annotation_pt{i}_maxsep.csv',
        names=['header', 'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5'])
    ec_df.fillna('-', inplace=True)
    for i in range(1, 6):
        ec_df[f'pred_{i}'] = ec_df[f'pred_{i}'].apply(eval_ec)
    ec_dfs.append(ec_df)
merged_ec_df = pd.concat(ec_dfs, ignore_index=True)
dfs = []
for i in range(1, 15):
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
