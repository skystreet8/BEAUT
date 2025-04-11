import argparse
import biotite.structure.io as bsio
import os
import shutil
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--organism', type=str, required=True)
args = parser.parse_args()
organism = args.organism
if organism in ['B_Ado', 'B_Xyl', 'C_Com', 'C_M62_1', 'H_Fil', 'R_Gna', 'S_Inf']:
    src_dir = f'../data/BA_transformers/{organism}_pdbs'
    tgt_dir = f'../data/BA_transformers/high_plddt_structs/{organism}_pdbs'
    if not os.path.exists('../data/BA_transformers/high_plddt_structs'):
        os.mkdir('../data/BA_transformers/high_plddt_structs')
    if not os.path.exists(tgt_dir):
        os.mkdir(tgt_dir)
else:
    raise NotImplementedError()
fnames = os.listdir(src_dir)
keep_fnames = []
for f in tqdm(fnames):
    struct = bsio.load_structure(os.path.join(src_dir, f), extra_fields=["b_factor"])
    plddt = struct.b_factor.mean()
    if plddt >= 70:
        keep_fnames.append(f)
for f in keep_fnames:
    n = shutil.copy(os.path.join(src_dir, f), tgt_dir)