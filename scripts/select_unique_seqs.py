from argparse import ArgumentParser
from utils import *
import pandas as pd
import networkx as nx
import random


parser = ArgumentParser()
parser.add_argument('-f', '--file', type=str, required=True)
args = parser.parse_args()
source = args.file
if source == 'pos_seqs_v3':
    pos_h, pos_s = ReadFastaFile('../data/positive_seqs_v3.fasta')
    h2s = {h: s for h, s in zip(pos_h, pos_s)}
    s2h = {s: h for h, s in zip(pos_h, pos_s)}
    pos_blast = pd.read_csv('../data/pos_seqs_v3_self_blast.tsv', sep='\t', comment='#', #
                            names=['query', 'target', 'fident', 'alnlen', 'mismatch', 'gapopen', 'qstart', 'qend', 'tstart',
                                   'tend', 'e_value', 'bits'])
    seq_graph = nx.Graph()
    edges = [(t[1], t[2]) for t in pos_blast.itertuples() if t[3] >= 90 and t[1] != t[2]]
    seq_graph.add_edges_from(edges)
    nodes = list(seq_graph.nodes.keys())
    for h in pos_h:
        if h not in nodes:
            seq_graph.add_node(h)
    components = list(nx.connected_components(seq_graph))
    components.sort(key=lambda x: len(x), reverse=True)
    remove_hs = []
    for clu in components:
        if len(clu) > 1:
            keep_h = random.choice(list(clu))
            remove_hs.extend(list(clu - {keep_h}))
    assert len(remove_hs) + len(components) == len(pos_h)
    pos_h = list(set(pos_h) - set(remove_hs))
    pos_s = [h2s[h] for h in pos_h]
    print(len(pos_h))
    SaveFastaFile('../data/positive_seqs_v3_unique.fasta', pos_h, pos_s)
elif source == 'pos_seqs_v3_sub_pok_sim_aug_v3':
    pos_h, pos_s = ReadFastaFile('../data/positive_seqs_v3_substrate_pocket_sim_aug_v3.fasta')
    h2s = {h: s for h, s in zip(pos_h, pos_s)}
    s2h = {s: h for h, s in zip(pos_h, pos_s)}
    pos_blast = pd.read_csv('../data/pos_seqs_v3_sub_pok_sim_aug_v3_self_blast.tsv', sep='\t', comment='#',  #
                            names=['query', 'target', 'fident', 'alnlen', 'mismatch', 'gapopen', 'qstart', 'qend',
                                   'tstart',
                                   'tend', 'e_value', 'bits'])
    seq_graph = nx.Graph()
    edges = [(t[1], t[2]) for t in pos_blast.itertuples() if t[3] >= 90 and t[1] != t[2]]
    seq_graph.add_edges_from(edges)
    nodes = list(seq_graph.nodes.keys())
    for h in pos_h:
        if h not in nodes:
            seq_graph.add_node(h)
    components = list(nx.connected_components(seq_graph))
    components.sort(key=lambda x: len(x), reverse=True)
    remove_hs = []
    for clu in components:
        if len(clu) > 1:
            keep_h = random.choice(list(clu))
            remove_hs.extend(list(clu - {keep_h}))
    assert len(remove_hs) + len(components) == len(pos_h)
    pos_h = list(set(pos_h) - set(remove_hs))
    pos_s = [h2s[h] for h in pos_h]
    print(len(pos_h))
    SaveFastaFile('../data/positive_seqs_v3_substrate_pocket_sim_aug_v3_unique.fasta', pos_h, pos_s)
else:
    raise NotImplementedError()
