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
