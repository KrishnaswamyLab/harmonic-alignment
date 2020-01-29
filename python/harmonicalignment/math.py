from __future__ import division
import graphtools
import numpy as np
from sklearn import decomposition
from sklearn.utils.extmath import randomized_svd
import tasklogger


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
    U, _, V = randSVD(X, random_state=random_state)
    X_orth = U.dot(V.T)
    return X_orth


def graphFourierBasis(G, n_eigenvectors=None):
    # diffusion maps with normalized Laplacian
    tasklogger.log_start("eigendecomposition")
    if n_eigenvectors is None:
        G.compute_fourier_basis()
    else:
        # temporary workaround until pygsp updates to pypi
        from scipy import sparse
        G._e, G._U = sparse.linalg.eigsh(G.L, n_eigenvectors, which='SM')
    tasklogger.log_complete("eigendecomposition")
    phi, lmbda = G.U, G.e
    # smallest to largest
    lmbda_idx = np.argsort(lmbda)
    phi, lmbda = phi[:, lmbda_idx], lmbda[lmbda_idx]
    #  trim trivial information
    phi, lmbda = phi[:, 1:], lmbda[1:]
    return phi, lmbda


def fourierBasis(X, decay, knn, n_pca, n_eigenvectors=None,
                         n_jobs=1, verbose=0, random_state=None):
    # diffusion maps with normalized Laplacian
    G = graphtools.Graph(X, knn=knn, decay=decay,
                         n_pca=n_pca, use_pygsp=True, thresh=1e-4,
                         anisotropy=1, lap_type='normalized',
                         n_jobs=n_jobs, verbose=verbose,
                         random_state=random_state)
    return graphFourierBasis(G, n_eigenvectors=n_eigenvectors)


def graphDiffusionCoordinates(G, n_eigenvectors=None):
    # diffusion maps with normalized affinity matrix
    tasklogger.log_start("eigendecomposition")
    if n_eigenvectors is None:
        A = graphtools.utils.to_array(G.diff_aff)
        phi, lmbda = np.linalg.eigh(A)
    else:
        A = sparse.csr_matrix(G.diff_aff)
        phi, lmbda = sparse.linalg.eigsh(A, k=n_eigenvectors, which='LM')
    tasklogger.log_complete("eigendecomposition")
    # largest to smallest
    lmbda_idx = np.argsort(lmbda)[::-1]
    phi, lmbda = phi[:, lmbda_idx], lmbda[lmbda_idx]
    # divide by sqrt degrees
    phi = phi / phi[:,0][:,None]
    assert np.all(phi[:,0] == 1)
    #  trim trivial information
    phi, lmbda = phi[:, 1:], lmbda[1:]
    return phi, lmbda


def diffusionCoordinates(X, decay, knn, n_pca, n_eigenvectors=None,
                         n_jobs=1, verbose=0, random_state=None):
    # diffusion maps with normalized Laplacian
    G = graphtools.Graph(X, knn=knn, decay=decay,
                         n_pca=n_pca, use_pygsp=False, thresh=1e-4,
                         anisotropy=1,
                         n_jobs=n_jobs, verbose=verbose,
                         random_state=random_state)
    return graphDiffusionCoordinates(G, n_eigenvectors=n_eigenvectors)


def diffusionMap(phi, lmbda, t=1):
    """Diffusion map from diffusion coordinates

    Parameters
    ----------
    phi : array-like
        Eigenvectors, sorted according to eigenvalues,
        excluding the trivial eigenvector
    lmbda : list-like
        Eigenvalues sorted from smallest to largest,
        excluding the trivial eigenvalue
    t : int, optional (default: 1)
        Diffusion power
    """
    return phi * np.exp(-t * lmbda)
