import os, sys, torch


def sizeof(obj, unit='m'):
    size = 0
    if type(obj) == dict:
        for key in obj.keys():
            size += sizeof(key)
            size += sizeof(obj[key])

    elif type(obj) in [list, tuple]:
        for element in obj:
            size += sizeof(element)

    else:
        size = sys.getsizeof(obj)

    return size


def ascii_encode(x):
    # str => int list 
    result = []
    for char in x:
        result.append(ord(char))
    return result


def ascii_decode(x):
    # int list => str
    result = []
    for number in x:
        result.append(chr(number))
    return ''.join(result)


def shift_sequence(sequence, offset):
    seq_len = sequence.size(1)
    if offset > 0:  # shift right
        sequence = torch.cat([sequence[:, seq_len - offset:], sequence[:, :seq_len - offset]], dim=-1)
    elif offset < 0:  # shift left
        sequence = torch.cat([sequence[:, -offset:], sequence[:, :-offset]], dim=-1)

    return sequence


def check_binary_matrix(matrix):
    records = []

    for i, row in enumerate(matrix):
        row_short = []
        pre_value = None
        cnt = 0
        for j, value in enumerate(row):
            if pre_value is not None and pre_value != value:
                row_short.append('{}*{}'.format(pre_value, cnt))
                cnt = 0
            cnt += 1
            pre_value = value

        row_short.append('{}*{}'.format(pre_value, cnt))

        records.append(row_short)

    return records


def is_int(number):
    try:
        int(number)
        return True
    except ValueError:
        return False


