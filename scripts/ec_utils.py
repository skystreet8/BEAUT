def eval_ec(s):
    if s == '-':
        return 'NONE', 0
    return str(s.split('/')[0].split(':')[1]), float(s.split('/')[1])


def sort_key(ec_string):
    digits = ec_string.split('.')
    if digits[3].startswith('n'):
        return int(digits[0]) * 10000000 + int(digits[1]) * 100000 + int(digits[2]) * 1000 + 999
    else:
        digits = [int(d) for d in digits]
        return digits[0] * 10000000 + digits[1] * 100000 + digits[2] * 1000 + digits[3]


def sort_ecs(ecs):
    ecs = list(ecs)
    ecs.sort(key=sort_key)
    return ecs


def sort_key_short(ec_string):
    digits = ec_string.split('.')
    if len(digits) == 2:
        return int(digits[0]) * 10000000 + int(digits[1]) * 100000
    elif len(digits) == 4:
        if digits[3].startswith('n'):
            return int(digits[0]) * 10000000 + int(digits[1]) * 100000 + int(digits[2]) * 1000 + 999
        else:
            digits = [int(d) for d in digits]
            return digits[0] * 10000000 + digits[1] * 100000 + digits[2] * 1000 + digits[3]
    else:
        digits = [int(d) for d in digits]
        return digits[0] * 10000000 + digits[1] * 100000 + digits[2] * 1000


def sort_ecs_short(ecs):
    ecs = list(ecs)
    ecs.sort(key=sort_key_short)
    return ecs


def shorten_ec(ec_string: str):
    digits = ec_string.split('.')
    return '.'.join(digits[:3])
