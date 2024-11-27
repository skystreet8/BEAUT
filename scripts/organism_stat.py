import pandas as pd
from argparse import ArgumentParser
from collections import defaultdict
from ec_utils import shorten_ec, sort_ecs_short


parser = ArgumentParser()
parser.add_argument('-m', '--model', required=True, help='Processing Base / Aug model predictions.')
args = parser.parse_args()
if args.model in {'base', 'aug'}:
    df = pd.read_csv(f'../data/PRJNA28331_{args.model}/PRJNA28331_{args.model}_final.csv')
    df['clean_ec'] = df['clean_ec'].apply(eval)
    organism_set = set()
    for t in df.itertuples():
        olist = eval(t[4])
        for o in olist:
            organism_set.add(o)

    organism2headers = defaultdict(list)
    for t in df.itertuples():
        olist = eval(t[4])
        for o in olist:
            organism2headers[o].append(t[1])
    assert len(organism_set) == len(organism2headers)
    organism2num_seqs = {k: len(organism2headers[k]) for k in organism2headers}

    org_count_df = pd.DataFrame(organism2num_seqs.items(), columns=['organism', 'num_positive_seqs'])
    org_count_df.sort_values('num_positive_seqs', ascending=False, inplace=True)
    org_count_df.to_csv(f'../data/PRJNA28331_{args.model}/PRJNA28331_{args.model}_num_positive_seqs_by_organism.csv',
                        index=False)

    headers2ecs = {t[1]: t[-1] for t in df.itertuples()}
    organism2ecs = defaultdict(list)
    for t in df.itertuples():
        olist = eval(t[4])
        shorten_ec_list = [shorten_ec(s) for s in t[-1]]
        for o in olist:
            organism2ecs[o].extend(shorten_ec_list)

    all_shortend_ecs = set()
    for o in organism2ecs:
        for s in organism2ecs[o]:
            all_shortend_ecs.add(s)
    all_shortend_ecs = list(all_shortend_ecs)
    all_shortend_ecs = sort_ecs_short(all_shortend_ecs)
    all_organisms = list(organism2ecs.keys())

    data = {'organism': all_organisms}
    for ec in all_shortend_ecs:
        data[ec] = [organism2ecs[o].count(ec) for o in all_organisms]
    org_ec_count_df = pd.DataFrame(data, columns=['organism'] + all_shortend_ecs)
    total_seqs = [organism2num_seqs[t[1]] for t in org_ec_count_df.itertuples()]
    org_ec_count_df = org_ec_count_df.assign(total_seqs=total_seqs)
    org_ec_count_df.sort_values('total_seqs', ascending=False, inplace=True)
    org_ec_count_df.to_csv(f'../data/PRJNA28331_{args.model}/PRJNA28331_{args.model}_seq_ec_counts_by_organism.csv',
                           index=False)
else:
    raise NotImplementedError()
