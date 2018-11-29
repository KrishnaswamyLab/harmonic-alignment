import numpy as np
import tasklogger

from . import utils, math


def itersine_wavelet(loc, scale, overlap):
    def itersine_wavelet_i(x):
        return math.itersine(x / scale - loc / overlap + 1 / 2) * np.sqrt(2 / overlap)
    return itersine_wavelet_i


def build_itersine_wavelets(lmbda, n_filters, overlap):
    lambda_max = max(lmbda)
    # maximum laplacian eigenvalue
    scale = lambda_max / (n_filters - overlap + 1) * (overlap)
    # response evaluation... this is the money
    lambda_filt = np.zeros((len(lmbda), n_filters))
    for i in range(n_filters):
        filter_i = itersine_wavelet(loc=i + 1, scale=scale, overlap=overlap)
        lambda_filt[:, i] = filter_i(lmbda)
    return lambda_filt


def evaluate_itersine_wavelets(X, phi_X, lambda_X, n_filters, overlap):
    # get fourier coefficients
    X_fourier = phi_X.T @ X
    # build wavelets
    wavelet_X = build_itersine_wavelets(lambda_X, n_filters, overlap=overlap)
    # wavelet_X is the filter evaluated over the eigenvalues.  So we
    # can pointwise multiply each wavelet_X / 2 by the fourier
    # coefficients
    # evaluate wavelets over data in the spectral domain
    wavelet_X_eval = np.conj(wavelet_X)[:, :, None] * X_fourier[:, None, :]
    return wavelet_X_eval


def correlate_wavelets(wavelet_X, wavelet_Y, n_filters):
    blocks = np.zeros((wavelet_X.shape[0], n_filters, wavelet_Y.shape[0]))
    for i in range(n_filters):  # for each filter, build a correlation
        blocks[:, i, :] = wavelet_X[:, i, :] @ wavelet_Y[:, i, :].T
    #  construct transformation matrix
    # sum wavelets up
    transform = np.sum(blocks, axis=1)
    return transform


def build_wavelet_transform(wavelet_X, wavelet_Y, n_filters):
    transform = correlate_wavelets(wavelet_X, wavelet_Y, n_filters)
    transform_orth = math.orthogonalize(transform)
    return transform_orth


def align(X, Y, n_filters, overlap=2, t=1,
          knn=5, decay=20, n_pca=100,
          n_jobs=1, verbose=False, random_state=None,
          knn_X=None, knn_Y=None, knn_XY=None,
          decay_X=None, decay_Y=None, decay_XY=None,
          n_pca_X=None, n_pca_Y=None, n_pca_XY=0):
    """Harmonic alignment

    Parameters
    ----------
    X : array-like, shape=[n_samples, n_features]
        Input dataset
    Y : array-like, shape=[m_samples, n_features]
        Input dataset
    n_filters : int
        Number of wavelets
    overlap : float, optional (default: 2)
        Amount of overlap between wavelets
    t : int, optional (default: 1)
        Amount of diffusion
    knn : int, optional (default: 5)
        Default number of nearest neighbors
    decay : float, optional (default: 20)
        Default value of alpha decay
    n_pca : int, optional (default: 100)
        Default number of principal components on which to build graph.
        If 0, no PCA is performed.
    n_jobs : int, optional (default: 1)
        Number of threads. -1 uses all available
    verbose : int or bool, optional (default: 0)
        Verbosity of logging output. 0 is silent, 1 is standard, 2 is debug.
    random_state : int or np.RandomState, optional (default: None)
        Sets the random seed
    knn_{X,Y,XY} : int, optional (default: None)
        If not None, overrides `knn`
    decay_{X,Y,XY} : int, optional (default: None)
        If not None, overrides `decay`
    n_pca_{X,Y,XY} : int, optional (default: None)
        If not None, overrides `n_pca`

    Returns
    -------
    XY : array-like, shape=[n_samples + m_samples, n_samples + m_samples - 1]
    """
    tasklogger.set_level(verbose)
    np.random.seed(random_state)
    tasklogger.log_start("Harmonic Alignment")
    # handle default values
    knn_X = utils.with_default(knn_X, knn)
    knn_Y = utils.with_default(knn_Y, knn)
    knn_XY = utils.with_default(knn_XY, knn)
    decay_X = utils.with_default(decay_X, decay)
    decay_Y = utils.with_default(decay_Y, decay)
    decay_XY = utils.with_default(decay_XY, decay)
    n_pca_X = utils.with_default(n_pca_X, n_pca)
    n_pca_Y = utils.with_default(n_pca_Y, n_pca)
    n_pca_XY = utils.with_default(n_pca_XY, n_pca)
    n_pca = None if n_pca == 0 else n_pca
    n_pca_X = None if n_pca_X == 0 else n_pca_X
    n_pca_Y = None if n_pca_Y == 0 else n_pca_Y
    n_pca_XY = None if n_pca_XY == 0 else n_pca_XY
    # normalized L with diffusion coordinates
    tasklogger.log_start("diffusion coordinates")
    phi_X, lambda_X = math.diffusionCoordinates(
        X, decay_X, knn_X, n_pca_X,
        n_jobs=n_jobs, verbose=verbose, random_state=random_state)
    phi_Y, lambda_Y = math.diffusionCoordinates(
        Y, decay_Y, knn_Y, n_pca_Y,
        n_jobs=n_jobs, verbose=verbose, random_state=random_state)
    tasklogger.log_complete("diffusion coordinates")
    # evaluate wavelets over data in the spectral domain
    tasklogger.log_start("wavelets")
    wavelet_X = evaluate_itersine_wavelets(
        X, phi_X, lambda_X, n_filters, overlap)
    wavelet_Y = evaluate_itersine_wavelets(
        Y, phi_Y, lambda_Y, n_filters, overlap)
    tasklogger.log_complete("wavelets")
    #  correlate the spectral domain wavelet coefficients.
    tasklogger.log_start("orthogonal transformation")
    transform = build_wavelet_transform(wavelet_X, wavelet_Y, n_filters)
    tasklogger.log_complete("orthogonal transformation")
    #  compute transformed data
    tasklogger.log_start("transformed data")
    # phi_X in span(phi_Y)
    phi_X_transform = phi_X @ transform
    #  phi_Y in span(phi_X)
    phi_Y_transform = phi_Y @ transform.T
    # what is E?
    E = np.vstack([np.hstack([phi_X, phi_X_transform]),
                   np.hstack([phi_Y_transform, phi_Y])])
    # weight by low passed eigenvalues
    E_weighted = E @ np.diag(np.exp(-t * np.concatenate([lambda_X, lambda_Y])))
    # build the joint diffusion map
    tasklogger.log_start("diffusion coordinates")
    phi_transform, lambda_transform = math.diffusionCoordinates(
        E_weighted, decay_XY, knn_XY, n_pca_XY,
        n_jobs=n_jobs, verbose=verbose, random_state=random_state)
    XY_combined = phi_transform @ np.diag(np.exp(-lambda_transform))
    tasklogger.log_complete("diffusion coordinates")
    tasklogger.log_complete("transformed data")
    tasklogger.log_complete("Harmonic Alignment")
    return XY_combined
