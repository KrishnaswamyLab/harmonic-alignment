import numpy as np


def with_default(x, default):
    if x is None:
        return default
    else:
        return x


def check_and_transpose(ref, inData):
    if ref not in inData.shape:
        return False
    else:
        if inData.ndim < 2:
            inData = inData[:, np.newaxis]
        if inData.shape[0] == ref:
            return inData
        else:
            pos = inData.shape.index(ref)
            return np.swapaxes(inData, 0, pos)


def is_positive(x):
    return x > 0


def in_positive_range(x, y):
    return is_positive(x) and x < y
