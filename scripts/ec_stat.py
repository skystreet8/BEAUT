import pickle
from ec_utils import sort_ecs_short, shorten_ec
import pandas as pd
from collections import defaultdict
from argparse import ArgumentParser


def shorten_ec_list(t):
    return [shorten_ec(s) for s in t]


parser = ArgumentParser()
parser.add_argument('-m', '--model', required=True)
args = parser.parse_args()

ec2names = pickle.load(open('../data/kegg_ec2names.pkl', 'rb'))
if args.model == 'aug':
    raw_df_aug = pd.read_csv('../data/PRJNA28331_aug/PRJNA28331_aug_final.csv')
    raw_df_aug['clean_ec_short'] = raw_df_aug['clean_ec'].apply(eval)
    for i in raw_df_aug.index:
        short_ecs = raw_df_aug.at[i, 'clean_ec_short']
        for j in range(len(short_ecs)):
            if short_ecs[j] == '3.3.1.1':
                raw_df_aug.at[i, 'clean_ec_short'][j] = '3.13.2.1'
            elif short_ecs[j] == '3.3.1.2':
                raw_df_aug.at[i, 'clean_ec_short'][j] = '3.13.2.2'
            elif short_ecs[j] == '3.3.1.3':
                raw_df_aug.at[i, 'clean_ec_short'][j] = '3.2.1.148'

    raw_df_aug['clean_ec_short'] = raw_df_aug['clean_ec_short'].apply(shorten_ec_list)
    all_ecs = set()
    for t in raw_df_aug.itertuples():
        for ec in t[26]:
            all_ecs.add(ec)
    all_ecs = list(all_ecs)
    all_ecs = sort_ecs_short(all_ecs)

    short_ec_counter = defaultdict(int)
    for ec in all_ecs:
        for t in raw_df_aug.itertuples():
            if ec in t[26]:
                short_ec_counter[ec] += 1
    names = []
    for ec in all_ecs:
        digits = ec.split('.')
        ec1 = digits[0]
        ec2 = '.'.join(digits[:2])
        names.append(', '.join([ec2names[ec1], ec2names[ec2], ec2names[ec]]))
    ec_stat_df_aug = pd.DataFrame({'EC': all_ecs, 'total': [short_ec_counter[ec] for ec in all_ecs], 'name': names})
    ec_stat_df_aug.to_csv('../data/PRJNA28331_aug/aug_results_ec_counts_total.csv', index=False)
elif args.model == 'base':
    raw_df_base = pd.read_csv('../data/PRJNA28331_base/PRJNA28331_base_final.csv')
    raw_df_base['clean_ec_short'] = raw_df_base['clean_ec'].apply(eval)
    for i in raw_df_base.index:
        short_ecs = raw_df_base.at[i, 'clean_ec_short']
        for j in range(len(short_ecs)):
            if short_ecs[j] == '3.3.1.1':
                raw_df_base.at[i, 'clean_ec_short'][j] = '3.13.2.1'
            elif short_ecs[j] == '3.3.1.2':
                raw_df_base.at[i, 'clean_ec_short'][j] = '3.13.2.2'
            elif short_ecs[j] == '3.3.1.3':
                raw_df_base.at[i, 'clean_ec_short'][j] = '3.2.1.148'

    raw_df_base['clean_ec_short'] = raw_df_base['clean_ec_short'].apply(shorten_ec_list)
    all_ecs = set()
    for t in raw_df_base.itertuples():
        for ec in t[26]:
            all_ecs.add(ec)
    all_ecs = list(all_ecs)
    all_ecs = sort_ecs_short(all_ecs)

    short_ec_counter = defaultdict(int)
    for ec in all_ecs:
        for t in raw_df_base.itertuples():
            if ec in t[26]:
                short_ec_counter[ec] += 1
    names = []
    for ec in all_ecs:
        digits = ec.split('.')
        ec1 = digits[0]
        ec2 = '.'.join(digits[:2])
        names.append(', '.join([ec2names[ec1], ec2names[ec2], ec2names[ec]]))
    ec_stat_df_base = pd.DataFrame(
        {'EC': all_ecs, 'total': [short_ec_counter[ec] for ec in all_ecs], 'name': names})
    ec_stat_df_base.to_csv('../data/PRJNA28331_base/base_results_ec_counts_total.csv', index=False)
else:
    raise NotImplementedError()
