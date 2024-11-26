import re


def get_pocket_volume(fname: str):
    lines = open(fname).readlines()
    vol_line = lines[16].strip()
    fields = re.split(r'\s+', vol_line)
    if 'A^3' != fields[-1]:
        raise RuntimeError('Volume data not found!')
    return float(fields[-2])

