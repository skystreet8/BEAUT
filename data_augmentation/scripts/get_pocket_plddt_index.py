import biotite.structure.io as bsio


def get_plddt_index(pocket_file):
    struct = bsio.load_structure(pocket_file, extra_fields=["b_factor"])
    res_id2plddt = {}
    for k, v in zip(struct.res_id, struct.b_factor):
        res_id2plddt[k] = v
    plddts = list(res_id2plddt.values())
    return sum([v >= 90 for v in plddts]) / len(res_id2plddt)


def get_plddt_index_Calpha(pocket_file):
    struct = bsio.load_structure(pocket_file, extra_fields=["b_factor"])
    res_id2plddt = {}
    for k, atom_name, v in zip(struct.res_id, struct.atom_name, struct.b_factor):
        if atom_name == 'CA':
            res_id2plddt[k] = v
    plddts = list(res_id2plddt.values())
    return sum([v >= 80 for v in plddts]) / len(res_id2plddt)
