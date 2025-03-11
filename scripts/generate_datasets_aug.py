from collections import defaultdict
from copy import deepcopy
import networkx as nx
import random
import pandas as pd
from utils import ReadFastaFile


def get_num_neg_seqs_per_bin(seqs):
    lengths = [len(s) for s in seqs]
    num_neg_seqs_per_bin = [0 for _ in range(18)]
    for l in lengths:
        num_neg_seqs_per_bin[min(l // 50 - 3, len(num_neg_seqs_per_bin) - 1)] += 1
    num_neg_seqs_per_bin = [max(w, 1) for w in num_neg_seqs_per_bin]  # Must have 1 sequence per bin to ensure a smooth distribution
    num_neg_seqs_per_bin = [5 * w for w in num_neg_seqs_per_bin]
    return num_neg_seqs_per_bin


def flatten(l):
    res = []
    for t in l:
        res.extend(t)
    return res


def divide_k_fold(items, K):
    """
    E.g. input: [1, 2, 3, 4, 5, 6, 7, 8], K=3

    output: [[1, 4, 7], [2, 5, 8], [3, 6]]
    :param items: A list of items
    :param K: Number of folds
    :return: K sub-lists within one list
    """
    folds = [[] for _ in range(K)]
    for i, obj in enumerate(items):
        folds[i % K].append(obj)
    return folds


NUM_FOLDS = 5
TEST_RATIO = 0.1
pos_headers, pos_seqs = ReadFastaFile('../data/positive_seqs_v3_substrate_pocket_sim_aug_v3_unique.fasta') #
pos_h2s = {h: s for h, s in zip(pos_headers, pos_seqs)}
pos_s2h = {s: h for h, s in zip(pos_headers, pos_seqs)}
pos_h_v3, pos_s_v3 = ReadFastaFile('../data/positive_seqs_v3_unique.fasta')
neg_headers, neg_seqs = ReadFastaFile('../data/negative_seqs_v2.fasta')
neg_s2h = {s: h for h, s in zip(neg_headers, neg_seqs)}
# Divide positive sequences into clusters according to local pairwise alignments
pos_blast = pd.read_csv('../data/pos_seqs_v3_sub_pok_sim_aug_v3_uniq_self_blast.tsv', sep='\t', comment='#', #
                        names=['query', 'target', 'fident', 'alnlen', 'mismatch', 'gapopen', 'qstart', 'qend', 'tstart',
                               'tend', 'e_value', 'bits'])
seq_graph = nx.Graph()
edges = [(t[1], t[2]) for t in pos_blast.itertuples() if t[3] >= 30 and t[1] != t[2]]  # 10.1002/0471250953.bi0301s42
seq_graph.add_edges_from(edges)
nodes = list(seq_graph.nodes.keys())
for h in pos_headers:
    if h not in nodes:
        seq_graph.add_node(h)
components = list(nx.connected_components(seq_graph))
components.sort(key=lambda x: len(x), reverse=True)
components = [list(clu) for clu in components]  # 951 clusters

# Select test positive samples
num_test_pos_headers = int(round(len(pos_headers) * TEST_RATIO, 0))  # 238 test sequences
test_possible_clus = [deepcopy(components[i]) for i in range(len(components))]
keep_test_clus = [i for i in range(len(components)) if set(test_possible_clus[i]) & set(pos_h_v3) == set(test_possible_clus[i])]
test_pos_headers = []  # Test for positive seqs
for i in keep_test_clus:
    test_pos_headers.extend(test_possible_clus[i])
test_possible_clus = [deepcopy(test_possible_clus[i]) for i in range(len(components)) if i not in keep_test_clus]
random.shuffle(test_possible_clus)
test_pos_clus = []  # Indexes for clusters to be tested
flag = False
while True:
    for i in range(len(components) - len(keep_test_clus)):
        test_pos_headers.extend(test_possible_clus[i])
        test_pos_clus.append(i)
        if len(test_pos_headers) == num_test_pos_headers:
            flag = True
            break
    if flag:
        break
    else:
        test_pos_headers = []
        test_pos_clus = []
        random.shuffle(test_possible_clus)
print(len(test_pos_clus) + len(keep_test_clus))
print(len(test_pos_headers))

# Select positive & negative samples for train & validation sets
no_test_clus = list(set(range(len(components) - len(keep_test_clus))) - set(test_pos_clus))
no_test_pos_headers = []
for clu in no_test_clus:
    no_test_pos_headers.extend(test_possible_clus[clu])
no_test_pos_seqs = [pos_h2s[h] for h in no_test_pos_headers]
no_test_pos_seq_groups = [[] for _ in range(18)]
for s in no_test_pos_seqs:
    no_test_pos_seq_groups[min(len(s) // 50 - 3, len(no_test_pos_seq_groups) - 1)].append(s)
no_test_pos_header_groups = [[pos_s2h[s] for s in g] for g in no_test_pos_seq_groups]  # Train + Val for positive seqs
num_neg_seqs_per_bin = get_num_neg_seqs_per_bin(no_test_pos_seqs)
neg_seq_groups = [[] for _ in range(18)]
no_test_neg_seq_groups = []
for s in neg_seqs:
    gid = min([len(s) // 50 - 3, len(neg_seq_groups) - 1])  # Starts from 157, 50 per bin, last bin is 1000 to 1074
    neg_seq_groups[gid].append(s)
for i in range(len(neg_seq_groups)):
    N = num_neg_seqs_per_bin[i]
    if N < len(neg_seq_groups[i]):
        no_test_neg_seq_groups.append(random.sample(neg_seq_groups[i], N))
    else:
        no_test_neg_seq_groups.append(neg_seq_groups[i])
        print(f'Group {i + 1} of {len(neg_seq_groups[i])} negative samples less than {N}, fully taken.')
no_test_neg_header_groups = [[neg_s2h[s] for s in g] for g in no_test_neg_seq_groups]  # Train + Val for negative seqs
# Remaining negative samples are used for testing
test_neg_headers = list(set(neg_headers) - set(flatten(no_test_neg_header_groups)))  # Test for negative seqs

# Start shuffling
for g in no_test_pos_header_groups:
    random.shuffle(g)
for g in no_test_neg_header_groups:
    random.shuffle(g)
# Divide positive & negative seqs for the Train + Val sets into 5 folds within each length bin.
# Merge all sub-folds from each bin to generate one complete fold.
# This would ensure that the length distributions are consistent for all folds.
# If any sub-fold inside a bin is empty(e.g. 5 folds for 3 sequences), then some complete folds
# will lack sequences with specific lengths.
no_test_neg_header_groups_k_fold = [divide_k_fold(g, NUM_FOLDS) for g in no_test_neg_header_groups]
no_test_pos_header_groups_k_fold = [divide_k_fold(g, NUM_FOLDS) for g in no_test_pos_header_groups]
no_test_neg_headers = flatten([flatten(t) for t in no_test_neg_header_groups_k_fold])
no_test_pos_headers = flatten([flatten(t) for t in no_test_pos_header_groups_k_fold])
datasets = []
for i in range(NUM_FOLDS):
    datasets_pos = [[] for _ in range(len(no_test_pos_header_groups_k_fold))]
    datasets_neg = [[] for _ in range(len(no_test_neg_header_groups_k_fold))]
    for j in range(len(no_test_pos_header_groups_k_fold)):
        for ii in range(NUM_FOLDS):
            if ii == i:
                datasets_pos[j].append(['val' for _ in range(len(no_test_pos_header_groups_k_fold[j][ii]))])
                datasets_neg[j].append(['val' for _ in range(len(no_test_neg_header_groups_k_fold[j][ii]))])
            else:
                datasets_pos[j].append(['train' for _ in range(len(no_test_pos_header_groups_k_fold[j][ii]))])
                datasets_neg[j].append(['train' for _ in range(len(no_test_neg_header_groups_k_fold[j][ii]))])
    datasets.append(flatten([flatten(t) for t in datasets_pos]) + flatten([flatten(t) for t in datasets_neg]))

num_test_seqs = len(test_pos_headers) + len(test_neg_headers)
data = {'header': no_test_pos_headers + no_test_neg_headers + test_pos_headers + test_neg_headers,
        'label': [1 for _ in range(len(no_test_pos_headers))] + [0 for _ in range(len(no_test_neg_headers))] \
    + [1 for _ in range(len(test_pos_headers))] + [0 for _ in range(len(test_neg_headers))],
        'dataset_fold_1': datasets[0] + ['test' for _ in range(num_test_seqs)],
        'dataset_fold_2': datasets[1] + ['test' for _ in range(num_test_seqs)],
        'dataset_fold_3': datasets[2] + ['test' for _ in range(num_test_seqs)],
        'dataset_fold_4': datasets[3] + ['test' for _ in range(num_test_seqs)],
        'dataset_fold_5': datasets[4] + ['test' for _ in range(num_test_seqs)],
        }
df = pd.DataFrame(data=data)
df.to_csv(f'../data/sequence_dataset_v3_substrate_pocket_aug_eq_len_dist.csv', index=False) #
