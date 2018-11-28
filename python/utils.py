import numpy as np
import warnings



def _check_and_transpose(ref, inData):
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


def _ispos(x):
    return x > 0


def _inposrng(x, y):
    return _ispos(x) and x < y


