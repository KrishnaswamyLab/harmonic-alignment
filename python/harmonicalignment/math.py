from __future__ import division
import graphtools
import numpy as np
from scipy import sparse
from sklearn import decomposition
from sklearn.utils.extmath import randomized_svd


def itersine(x):
    return np.sin(0.5 * np.pi * (np.cos(np.pi * x))**2) * \
        ((x >= -0.5) & (x <= 0.5))


def randPCA(X, n_components=None, random_state=None):
    pca_op = decomposition.PCA(n_components, random_state=random_state)
    U, S, VT = pca_op._fit(X)
    return U, S, VT.T


def randSVD(X, n_components=None, random_state=None):
    if n_components is None:
        n_components = min(X.shape)
    U, S, VT = randomized_svd(X, n_components, random_state=random_state)
    return U, S, VT.T


def orthogonalize(X, random_state=None):
    # why PCA and not SVD?
    U, _, V = randPCA(X, random_state=random_state)
    X_orth = U.dot(V.T)
    return X_orth


def diffusionCoordinates(X, decay, knn, n_pca,
                         n_jobs=1, verbose=0, random_state=None):
    # diffusion maps with normalized Laplacian
    # n_pca = 0 corresponds to NO pca
    G = graphtools.Graph(X, knn=knn, decay=decay,
                         n_pca=n_pca, use_pygsp=True, thresh=0,
                         n_jobs=n_jobs, verbose=verbose,
                         random_state=random_state)
    n_samples = X.shape[0]
    W = G.W.tocoo()
    # W / (DD^T)
    W.data = W.data / (G.dw[W.row] * G.dw[W.col])
    # this is the anisotropic kernel
    nsqrtD = sparse.dia_matrix((np.array(np.sum(W, 0)) ** (-0.5), [0]),
                               W.shape)
    L = sparse.eye(n_samples) - nsqrtD.dot(W).dot(nsqrtD)
    U, S, _ = randSVD(L, random_state=random_state)
    # smallest to largest
    S_idx = np.argsort(S)
    U, S = U[:, S_idx], S[S_idx]
    #  trim trivial information
    U, S = U[:, 1:], S[1:]
    return U, S
