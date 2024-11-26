import pandas as pd
import re
from functools import partial


def get_header(fname_string: str, format='pdb'):
    if format == 'pdb':
        return fname_string.split('.')[0].split('_')[0]
    elif format == 'uhgp':
        return '_'.join(fname_string.split('.')[0].split('_')[:2])
    elif format == 'genbank':
        if fname_string.startswith('WP_'):
            return '_'.join(fname_string.split('_')[:2])
        else:
            return fname_string.split('_')[0]


def read_txt(fname: str, format_q='pdb'):
    lines = open(fname).readlines()
    lines = [l.strip() for l in lines]
    names = ['query', 'target', 'P-min_OP', 'P-max_OP', 'A', 'B', 'C']
    data = [re.split(r'\s+', l) for l in lines]
    df = pd.DataFrame(data=data, columns=names)
    del df['P-min_OP'], df['A'], df['B'], df['C']
    df['header'] = df['query'].apply(partial(get_header, format=format_q))
    df['P-max_OP'] = df['P-max_OP'].astype(float)
    return df[['header', 'query', 'target', 'P-max_OP']].copy()

