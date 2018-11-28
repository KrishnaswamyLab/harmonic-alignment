import numpy as np
from scipy.sparse.linalg import eigs as eigs
import warnings

def _check_and_transpose(ref, inData):
    if ref not in inData.shape:
        return False
    else:
        if inData.ndim<2:
            inData = inData[:,np.newaxis]
        if inData.shape[0] == ref:
            return inData
        else:
            pos = inData.shape.index(ref)
            return np.swapaxes(inData, 0,pos)


class DiffusionWavelets(object):
    def __init__(self, A, frames="tight", beta=None):
        self.N = A.shape[0]
        if isinstance(frames, str) and frames == "tight":
            if beta is None:
                warnings.warn("Tight frame requested and no spectral gap supplied. "
                              " Computing eigenvalues for spectral gap.")
                w, _ = eigs(A, 3)

                if (np.allclose(1, np.abs(w), atol=1e-8)):
                    raise ValueError('Disconnected Graph Detected')
                else:
                    beta = np.max(np.abs(w[1:]))

            self.beta = beta
            self.numframes = int(
                1 + np.ceil(np.log2((-1 / np.log2(self.beta)))))
        else:
            self.numframes = frames

        N = A.shape[0]
        self.T = 0.5 * (A + np.eye(N))
        W = [None for i in range(self.numframes)]

        W[0] = np.eye(N) - self.T
        curT = self.T
        for i in range(self.numframes):
            nextT = np.matmul(curT, curT)
            W[i] =  curT - nextT
            curT = nextT
        self.W = np.concatenate(W, axis=1)

    def __getitem__(self, scale):
        toret = self.W[:, scale * 2000:(scale + 1) * 2000]
        if toret.size>0:
            return toret
        else:
            raise KeyError("Attempted to slice a frame that does not exist.")
            
    def transform(self, x, scale=None):
        x = _check_and_transpose(self.N,x)
        return np.matmul(self.W.T, x)

    def power_density(self, x, p=2):
        x = _check_and_transpose(self.N,x)
        x_density = np.zeros((self.numframes, x.shape[1]))
        for i in range(0, self.numframes):
            x_density[i,:] = np.sum(np.matmul(self[i].T, x)**p,axis=0)**(1 / p)
        return x_density
