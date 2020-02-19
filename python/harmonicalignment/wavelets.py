import numpy as np
import warnings

from . import utils

from scipy.sparse.linalg import eigs as eigs
from abc import ABCMeta, abstractmethod
from functools import partial


class Wavelet(object):
    __metaclass__ = ABCMeta

    def __init__(self, N, numframes, transform_spectrum):
        self.numframes = numframes
        self.N = N
        self.spectral_representation = [None for x in range(numframes)]
        self.T = None
        self.W = None

    def __getitem__(self, j):

        if utils.in_positive_range(j, self.numframes):
            return self.W[:, j * self.N : (j + 1) * self.N]
        else:
            raise KeyError("Attempted to slice a frame that does not exist.")

    def evaluate_spectral(self, x=None, transform_spectrum="normalized"):
        if x is None:
            x = np.linspace(0, 2, 1000)
        output = np.zeros((self.numframes, x.shape[0]))
        for j in range(self.numframes):
            output[j, :] = self.evaluate_frame_spectral(j, x, transform_spectrum)
        return output

    @abstractmethod
    def _fouriermapping(self, x, domain, direction="forward"):
        raise NotImplementedError("Must override _fouriermapping")

    def evaluate_frame_spectral(self, j, x=None, transform_spectrum="normalized"):
        if utils.in_positive_range(j, self.numframes):
            x = self._fouriermapping(x, transform_spectrum, "forward")
            return self._spectral_representation[j](x)

    def transform(self, x, j=None):
        x = utils.check_and_transpose(self.N, x)
        if j is None:
            return np.matmul(self.W.T, x)
        else:
            return np.matmul(self[j].T, x)

    def power_density(self, x, p=2):
        x = utils.check_and_transpose(self.N, x)
        x_density = np.zeros((self.numframes, x.shape[1]))
        for j in range(self.numframes):
            x_density[j, :] = np.sum(np.matmul(self[j].T, x) ** p, axis=0) ** (1 / p)
        return x_density


class DiffusionWavelets(Wavelet):
    def __init__(self, A, frames="tight", beta=None, transform_spectrum="normalized"):
        if isinstance(frames, str) and frames == "tight":
            if beta is None:
                warnings.warn(
                    "Tight frame requested and no spectral gap supplied. "
                    " Computing eigenvalues for spectral gap."
                )
                w, _ = eigs(A, 3)

                if np.allclose(1, np.abs(w), atol=1e-8):
                    raise ValueError("Disconnected Graph Detected")
                else:
                    beta = np.max(np.abs(w[1:]))

            self.beta = beta
            numframes = int(1 + np.ceil(np.log2((-1 / np.log2(self.beta)))))
        else:
            numframes = frames

        N = A.shape[0]

        super().__init__(N, numframes, transform_spectrum)

        self.T = 0.5 * (A + np.eye(N))
        W = [None for i in range(self.numframes)]

        W[0] = np.eye(N) - self.T
        curT = self.T
        self._spectral_representation = [lambda x: 1 - x]
        genfunc = lambda x, j: -(x ** 2 ** (-1 + (j))) * (-1 + x ** 2 ** (-1 + j))
        for j in range(self.numframes):
            nextT = np.matmul(curT, curT)
            W[j] = curT - nextT
            curT = nextT
            self._spectral_representation.append(
                # this is the Fourier representation of the wavelets
                # excuse me while i MATLAB in your python
                partial(genfunc, j=float(j))
            )

        self.W = np.concatenate(W, axis=1)

    def _fouriermapping(self, x, domain, direction="forward"):
        # this holds the mapping back to the Fourier domain - the eigenvalues
        # of this operator are different.
        if domain == "normalized":
            if direction == "forward":
                if np.min(x) < 0 and np.allclose(np.max(np.abs(x)), 1, atol=1e-5):
                    warnings.warn(
                        "Requested a forward fourier mapping for points already in the range of this filter"
                    )
                    return x
                else:
                    return 1 - 0.5 * x
            else:
                return 2 - 2 * x
        # this one only works one direction until we figure out the logic.
        elif domain == "diffop":
            return 2 * (x) - 1
        elif domain == False or domain is None:
            return x
        else:
            raise NotImplementedError("This mapping is not implemented.")
