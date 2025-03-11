import pickle
import pandas as pd
from ec_utils import shorten_ec
from collections import defaultdict

df = pd.read_csv('../data/PRJNA28331_aug/PRJNA28331_aug_final.csv')
df.set_index('header', inplace=True)
clu_60 = pickle.load(open('../data/PRJNA28331_aug/PRJNA28331_aug_alnscore60_full_clusters.pkl', 'rb'))

cluster_info = pd.DataFrame({'cluster_id':
                                 [f'cluster_{i}' for i in range(1, len(clu_60) + 1)],
                             'size':
                                 [len(t) for t in clu_60]
                             })
cluster_ec_info = []
for indexes in clu_60:
    qdf = df.loc[indexes].copy().sort_values('pred_score', ascending=False)
    ec_dict = defaultdict(int)
    for t in qdf.itertuples():
        this_ecs = [shorten_ec(s) for s in eval(t[-1])]
        this_ecs = set(this_ecs)
        for k in this_ecs:
            ec_dict[k] += 1
    ec_info = ''
    most_common_ecs = list(ec_dict.items())
    most_common_ecs.sort(key=lambda x: x[1], reverse=True)
    most_common_ecs = most_common_ecs[:3]
    for t in most_common_ecs:
        ec_info += f'EC:{t[0]}, {round(t[1] / len(qdf) * 100, 1)}%; '
    cluster_ec_info.append(ec_info[:-2])

top1_ec = []
top1_ec_ratio = []
for s in cluster_ec_info:
    if s == 'NONE':
        top1_ec.append('-')
        top1_ec_ratio.append('-')
    else:
        try:
            top1_ec.append(s.split('; ')[0].split(',')[0])
            top1_ec_ratio.append(s.split('; ')[0].split(', ')[1])
        except:
            print(s)
            raise

top2_ec = []
top2_ec_ratio = []
for s in cluster_ec_info:
    if s == 'NONE':
        top2_ec.append('-')
        top2_ec_ratio.append('-')
    else:
        strings = s.split('; ')
        if len(strings) == 1:
            top2_ec.append('-')
            top2_ec_ratio.append('-')
        else:
            top2_ec.append(strings[1].split(', ')[0])
            top2_ec_ratio.append(strings[1].split(', ')[1])

top3_ec = []
top3_ec_ratio = []
for s in cluster_ec_info:
    if s == 'NONE':
        top3_ec.append('-')
        top3_ec_ratio.append('-')
    else:
        strings = s.split('; ')
        if len(strings) < 3:
            top3_ec.append('-')
            top3_ec_ratio.append('-')
        else:
            top3_ec.append(strings[2].split(', ')[0])
            top3_ec_ratio.append(strings[2].split(', ')[1])

cluster_info = cluster_info.assign(top1_ec=top1_ec, top1_ec_ratio=top1_ec_ratio,
                                   top2_ec=top2_ec, top2_ec_ratio=top2_ec_ratio, top3_ec=top3_ec, top3_ec_ratio=top3_ec_ratio)
cluster_info.to_csv('../data/PRJNA28331_aug/PRJNA28331_aug_alnscore60_ssn_clusters_full/cluster_info.csv',
                    index=False)

header2clu = {}
for i, t in enumerate(clu_60):
    for h in t:
        header2clu[h] = i + 1
headers = []
for t in clu_60:
    headers.extend(t)

qdf = df.loc[headers].copy()
qdf['cluster'] = [header2clu[t[0]] for t in qdf.itertuples()]
qdf.sort_values(by=['cluster', 'pred_score'], ascending=[True, False], inplace=True)
qdf['cluster'] = qdf['cluster'].astype(str)
for i in qdf.index:
    qdf.at[i, 'cluster'] = f"cluster_{qdf.at[i, 'cluster']}"
cols = qdf.columns.values.tolist()
qdf = qdf[cols[:2] + ['cluster'] + cols[2:-1]]
qdf.reset_index(inplace=True)
qdf.to_csv('../data/PRJNA28331_aug/PRJNA28331_aug_alnscore60_ssn_clusters_full/cluster_results.csv',
           index=False)
