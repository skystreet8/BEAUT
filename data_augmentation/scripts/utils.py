import re


def ReadFastaFile(fname: str):
    lines = open(fname).readlines()
    assert len(lines) % 2 == 0
    lines = [l.strip() for l in lines]
    headers = [lines[i * 2][1:] for i in range(len(lines) // 2)]
    seqs = [lines[i * 2 + 1] for i in range(len(lines) // 2)]
    return headers, seqs


def SaveFastaFile(filepath, headers, seqs):
    with open(filepath, 'w') as f:
        count = 0
        for h, s in zip(headers, seqs):
            count += 1
            f.write(f'>{h}\n')
            if not count == len(headers):
                f.write(f'{s}\n')
            else:
                f.write(f'{s}')
    f.close()


def ReformatFastaFile(filepath, full_header=False):
    lines = open(filepath).readlines()
    lines = [l.strip() for l in lines]
    headers = []
    seqs = []
    next_seq = ''
    for l in lines:
        if l.startswith('>'):
            if headers:
                seqs.append(next_seq)
                next_seq = ''
            if not full_header:
                headers.append(l[1:].split(' ')[0])
            else:
                headers.append(l[1:])
        else:
            next_seq = next_seq + l
    seqs.append(next_seq)
    return headers, seqs


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
    for l in pdb_lines:
        if l in pocket_lines:
            continue
        else:
            fields = re.split(r'\s+', l)
            if fields[5] in pocket_residue_nums:
                pocket_lines.append(l)
    pocket_lines.sort(key=lambda l: int(re.split(r'\s+', l)[1]))
    return pocket_lines
