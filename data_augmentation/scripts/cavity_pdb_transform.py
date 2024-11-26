import re


def ReadPDBFile(file):
    lines = open(file).readlines()
    lines = [l for l in lines if l.startswith('ATOM')]
    lines = [s[:-1] for s in lines]
    for l in lines:
        if not len(l) == 80:
            raise ValueError
    new_lines = []
    for l in lines:
        fields = re.split(r'\s+', l)
        if fields[-2] == 'H':
            continue
        new_lines.append(l)
    return new_lines


def FillPocketPDB(pocket_pdbfile, pdbfile):
    pocket_lines = open(pocket_pdbfile).readlines()
    pocket_lines = [l[:-1] for l in pocket_lines]
    if pocket_lines[-1].startswith('ENDMDL'):
        pocket_lines.pop(-1)
    pocket_residue_nums = set([re.split(r'\s+', l)[5] for l in pocket_lines])
    pdb_lines = ReadPDBFile(pdbfile)
    new_lines = []
    for l in pdb_lines:
        fields = re.split(r'\s+', l)
        if fields[5] in pocket_residue_nums:
            new_lines.append(l)
    new_lines.sort(key=lambda l: int(re.split(r'\s+', l)[1]))
    return new_lines
