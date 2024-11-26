from get_pocket_plddt_index import get_plddt_index, get_plddt_index_Calpha
from cavity_pdb_transform import FillPocketPDB
from get_pocket_vol import get_pocket_volume
from argparse import ArgumentParser
from collections import defaultdict
import os
import shutil
from tqdm import tqdm


def get_header(s: str):
    if s.startswith('WP_'):
        return '_'.join(s.split('_')[:2])
    else:
        return s.split('_')[0]


def get_index(s: str):
    if s.startswith('WP_'):
        return s.split('_')[3].split('.')[0]
    else:
        return s.split('_')[2].split('.')[0]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-o', '--organism', type=str, default='')
    parser.add_argument('-r', '--rescue', action='store_true', default=False)
    args = parser.parse_args()
    organism = args.organism
    if organism:
        pdb_dir = f'../data/BA_transformers/high_plddt_structs/{organism}_pdbs'
        if args.rescue:
            src_dir = f'../data/BA_transformers/high_plddt_pockets_rescue/{organism}'
            vol_tgt_dir = f'../data/BA_transformers/high_plddt_pockets_rescue_vol_filtered/{organism}'
            pm_tgt_dir = f'../data/BA_transformers/high_plddt_pockets_rescue_vol_filtered_pm/{organism}'
            id_tgt_dir = f'../data/BA_transformers/high_plddt_pockets_rescue_filtered/{organism}'
        else:
            src_dir = f'../data/BA_transformers/high_plddt_pockets/{organism}'
            vol_tgt_dir = f'../data/BA_transformers/high_plddt_pockets_vol_filtered/{organism}'
            pm_tgt_dir = f'../data/BA_transformers/high_plddt_pockets_vol_filtered_pm/{organism}'
            id_tgt_dir = f'../data/BA_transformers/high_plddt_pockets_filtered/{organism}'
        func = get_plddt_index_Calpha
    else:
        pdb_dir = '../data/positive_seq_v2_pdbs'
        if args.rescue:
            src_dir = f'../data/pos_v2_structs_high_plddt_pockets_rescue'
            vol_tgt_dir = f'../data/pos_v2_structs_high_plddt_pockets_rescue_vol_filtered'
            pm_tgt_dir = f'../data/pos_v2_structs_high_plddt_pockets_rescue_vol_filtered_pm'
            id_tgt_dir = f'../data/pos_v2_structs_high_plddt_pockets_rescue_filtered'
        else:
            src_dir = f'../data/pos_v2_structs_high_plddt_pockets'
            vol_tgt_dir = f'../data/pos_v2_structs_high_plddt_pockets_vol_filtered'
            pm_tgt_dir = f'../data/pos_v2_structs_high_plddt_pockets_vol_filtered_pm'
            id_tgt_dir = f'../data/pos_v2_structs_high_plddt_pockets_filtered'
        func = get_plddt_index
    if not os.path.exists(vol_tgt_dir):
        os.mkdir(vol_tgt_dir)
    if not os.path.exists(pm_tgt_dir):
        os.mkdir(pm_tgt_dir)
    if not os.path.exists(id_tgt_dir):
        os.mkdir(id_tgt_dir)
    header2pocket = defaultdict(list)
    pocket2vol = defaultdict(float)
    for f in tqdm(os.listdir(src_dir)):
        if '_cavity_' in f:
            h = get_header(f)
            index = get_index(f)
            header2pocket[h].append(f)
            pocket2vol[f] = get_pocket_volume(os.path.join(src_dir, f'{h}_vacant_{index}.pdb'))
    print(f'{len(pocket2vol)} pockets in total.')
    c = 0
    if args.organism:
        for f in pocket2vol:
            if 1000 <= pocket2vol[f] <= 5500:
                c += 1
                _ = shutil.copy(os.path.join(src_dir, f), vol_tgt_dir)
        print(f'{c} pockets are larger than 1000 Å^3 and smaller than 5500 Å^3.')
    else:
        for f in pocket2vol:
            if pocket2vol[f] >= 1000:
                c += 1
                _ = shutil.copy(os.path.join(src_dir, f), vol_tgt_dir)
        print(f'{c} pockets are larger than 1000 Å^3.')
    c = 0
    for f in tqdm(os.listdir(vol_tgt_dir)):
        h = get_header(f)
        index = get_index(f)
        lines = FillPocketPDB(os.path.join(vol_tgt_dir, f), os.path.join(pdb_dir, f'{h}.pdb'))
        with open(os.path.join(pm_tgt_dir, f'{h}_cavity_{index}_pm.pdb'), 'w') as file:
            for i, s in enumerate(lines):
                if i == len(lines) - 1:
                    file.write(f'{s}')
                else:
                    file.write(f'{s}\n')
    for f in tqdm(os.listdir(pm_tgt_dir)):
        if func(os.path.join(pm_tgt_dir, f)) >= 0.7:
            c += 1
            _ = shutil.copy(os.path.join(pm_tgt_dir, f), id_tgt_dir)
    if args.organism:
        print(f'{c} pockets have volumes between 1000 and 5500 Å^3 and have indexes >= 0.7.')
    else:
        print(f'{c} pockets are larger than 1000 Å^3 and have indexes >= 0.7.')
