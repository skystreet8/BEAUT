import pickle
import networkx as nx
from copy import deepcopy
import xml.parsers.expat
import time
import logging
logger = logging.getLogger('process_xgmml_graph')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class XGMMLParserHelper(object):
    def __init__(self):
        self._graph = nx.DiGraph()
        self._parser = xml.parsers.expat.ParserCreate()
        self._parser.StartElementHandler = self._start_element
        self._parser.EndElementHandler = self._end_element
        self._tagstack = list()

        self._network_att_el = dict()
        self._current_att_el = dict()
        self._current_list_att_el = list()
        self._current_obj = dict()

    def _start_element(self, tag, attr):
        self._tagstack.append(tag)

        if tag == 'graph':
            self._network_att_el = dict()

        if tag == 'node' or tag == 'edge':
            self._current_obj = dict(attr)

        if tag == 'att' and (self._tagstack[-2] == 'node' or
                             self._tagstack[-2] == 'edge'):
            if 'value' in attr:
                self._current_att_el = self._parse_att_el(self._current_att_el,
                                                          tag, attr)
            elif attr['type'] == 'list':
                self._current_list_name = attr['name']
                self._current_att_el[attr['name']] = list()

        if tag == 'att' and (self._tagstack[-2] == 'att'):
            self._current_list_att_el = dict(attr)
            if 'value' in attr:
                self._current_list_att_el = self._parse_att_el(
                    self._current_list_att_el, tag, attr)
                self._current_att_el[self._current_list_name].append(
                    self._current_list_att_el[attr['name']])

        if tag == 'att' and self._tagstack[-2] == 'graph':
            if 'value' in attr:
                self._network_att_el[attr['name']] = attr['value']

    def _parse_att_el(self, att_el, tag, attr):
        if 'value' in attr:
            if attr['type'] == 'string':
                att_el[attr['name']] = attr['value']
            elif attr['type'] == 'real':
                att_el[attr['name']] = float(attr['value'])
            elif attr['type'] == 'integer':
                att_el[attr['name']] = int(attr['value'])
            elif attr['type'] == 'boolean':
                att_el[attr['name']] = bool(attr['value'])
            else:
                raise NotImplementedError(attr['type'])

            return att_el

    def _end_element(self, tag):
        if tag == 'node':
            for k in remove_node_keys:
                if k in self._current_att_el:
                    del self._current_att_el[k]
            for k in keep_node_keys_repl:
                if k in self._current_att_el:
                    self._current_att_el[keep_node_keys_repl[k]] = deepcopy(self._current_att_el[k])
                    del self._current_att_el[k]
            if 'label' in self._current_obj:
                if 'label' in self._current_att_el:
                    self._current_att_el['@label'] = self._current_att_el['label']
                    del self._current_att_el['label']

                self._graph.add_node(self._current_obj['id'],
                                     label=self._current_obj['label'],
                                     **self._current_att_el)
            else:
                self._graph.add_node(self._current_obj['id'],
                                     **self._current_att_el)
            self._current_att_el = dict()
        elif tag == 'edge':
            self._current_att_el['fident'] = self._current_att_el['%id']
            del self._current_att_el['%id']
            self._graph.add_edge(self._current_obj['source'],
                                 self._current_obj['target'],
                                 **self._current_att_el)
            self._current_att_el = dict()

        self._tagstack.pop()

    def parseFile(self, file):
        self._parser.ParseFile(file)

    def graph(self):
        return self._graph

    def graph_attributes(self):
        return self._network_att_el


def XGMMLReader(graph_file):
    parser = XGMMLParserHelper()
    parser.parseFile(graph_file)
    return parser.graph()


ssn_name = 'alnscore60_full'
keep_node_keys_repl = {
    'Taxonomy ID': 'taxonomy_id',
    'UniProt Annotation Status': 'uniprot_annotation_status',
    'SwissProt Description': 'swissprot_description',
    'Sequence Length': 'length',
    'Sequence Status': 'sequence_status',
    'Number of IDs in Rep Node': 'num_members'
}
remove_node_keys = ['Sequence Source', 'Other IDs', 'Gene Name', 'NCBI IDs', 'List of IDs in Rep Node', 'Superkingdom', 'Kingdom', 'Phylum', 'Class',
                    'Order', 'Family',
                    'Genus', 'Species', 'PDB', 'TIGRFAMs', 'InterPro (Domain)', 'InterPro (Family)', 'InterPro (Homologous Superfamily)',
                    'InterPro (Other)', 'BRENDA ID', 'Cazy Name', 'GO Term', 'KEGG ID', 'PATRIC ID', 'STRING ID', 'HMP Body Site', 'HMP Oxygen',
                    'P01 gDNA', 'Rhea', 'AlphaFold', 'Sequence']

start_time = time.monotonic()
graph = XGMMLReader(
    open(f'../data/PRJNA28331_aug/PRJNA28331_aug_alnscore60_ssn_clusters_full/PRJNA28331_aug_{ssn_name}_ssn.xgmml',
         'rb')
)
end_time = time.monotonic()
logger.info(f'{round(end_time - start_time, 0)}s used for loading the graph.')
start_time = time.monotonic()
graph = graph.to_undirected()
end_time = time.monotonic()
logger.info(f'{round(end_time - start_time, 0)}s used for transforming the graph to undirected.')

logger.info(f'There are {graph.number_of_nodes()} nodes in the graph.')
logger.info(f'There are {graph.number_of_edges()} egdes in the graph.')
n_connected_components = len(list(nx.connected_components(graph)))
logger.info(f'There are {n_connected_components} clusters in the graph.')
cluster_sizes = [len(c) for c in nx.connected_components(graph)]
logger.info(f'Largest cluster: {max(cluster_sizes)}')
logger.info(f'There are {sum([n >= 3 for n in cluster_sizes])} clusters with >= 3 sequences.')
logger.info(f'There are {sum([n >= 5 for n in cluster_sizes])} clusters with >= 5 sequences.')
logger.info(f'There are {sum([n >= 10 for n in cluster_sizes])} clusters with >= 10 sequences.')
logger.info(f'There are {sum([n >= 50 for n in cluster_sizes])} clusters with >= 50 sequences.')
logger.info(f'There are {sum([n >= 100 for n in cluster_sizes])} clusters with >= 100 sequences.')
logger.info(f'There are {sum([n >= 500 for n in cluster_sizes])} clusters with >= 500 sequences.')
logger.info(f'There are {sum([n >= 1000 for n in cluster_sizes])} clusters with >= 1000 sequences.')
logger.info(f'There are {sum([n == 1 for n in cluster_sizes])} singletons.')

conn_comps = list(nx.connected_components(graph))
conn_comps.sort(key=lambda t: len(t), reverse=True)
conn_comps_to_save = []
for comp in conn_comps:
    conn_comps_to_save.append([])
    for n in comp:
        conn_comps_to_save[-1].extend(graph.nodes[n]['Description'])
with open(f'../data/PRJNA28331_aug/PRJNA28331_aug_{ssn_name}_clusters.pkl', 'wb') as f:
    pickle.dump(conn_comps_to_save, f)
f.close()
