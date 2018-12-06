from __future__ import division
from builtins import super
import numpy as np
import tasklogger
import joblib

from . import utils, math, parallel


def itersine_wavelet(loc, scale, overlap):
    def itersine_wavelet_i(x):
        return math.itersine(x / scale - loc / overlap + 1 / 2) * np.sqrt(2 / overlap)
    return itersine_wavelet_i


def build_itersine_wavelet(filter_idx, lmbda, n_filters, overlap):
    lambda_max = max(lmbda)
    # maximum laplacian eigenvalue
    scale = lambda_max / (n_filters - overlap + 1) * (overlap)
    # response evaluation... this is the money
    filter_fn = itersine_wavelet(
        loc=filter_idx + 1, scale=scale, overlap=overlap)
    return filter_fn(lmbda)


def evaluate_itersine_wavelet(filter_idx, X, phi_X, lambda_X, n_filters, overlap):
    # get fourier coefficients
    X_fourier = phi_X.T.dot(X)
    # build wavelets
    wavelet_X = build_itersine_wavelet(
        filter_idx, lambda_X, n_filters, overlap=overlap)
    # wavelet_X is the filter evaluated over the eigenvalues.  So we
    # can pointwise multiply each wavelet_X / 2 by the fourier
    # coefficients
    # evaluate wavelets over data in the spectral domain
    wavelet_X_eval = np.conj(wavelet_X)[:, None] * X_fourier
    return wavelet_X_eval


def correlate_wavelet(filter_idx,
                      X, phi_X, lambda_X,
                      Y, phi_Y, lambda_Y,
                      n_filters, overlap, q):
    q.queue(evaluate_itersine_wavelet,
            filter_idx, X, phi_X, lambda_X,
            n_filters=n_filters,
            overlap=overlap)
    q.queue(evaluate_itersine_wavelet,
            filter_idx, Y, phi_Y, lambda_Y,
            n_filters=n_filters,
            overlap=overlap)
    wavelet_X, wavelet_Y = q.run()
    return wavelet_X.dot(wavelet_Y.T)


def correlate_wavelets(X, phi_X, lambda_X,
                       Y, phi_Y, lambda_Y,
                       n_filters, overlap,
                       q=None, n_jobs=1):
    if q is None:
        with parallel.ParallelQueue(n_jobs=n_jobs) as q:
            return correlate_wavelets(X, phi_X, lambda_X,
                                      Y, phi_Y, lambda_Y,
                                      n_filters, overlap, q=q)
    else:
        transform = np.zeros((phi_X.shape[1], phi_Y.shape[1]))
        for filter_idx in range(n_filters):
            # for each filter, build a correlation
            transform += correlate_wavelet(filter_idx,
                                           X, phi_X, lambda_X,
                                           Y, phi_Y, lambda_Y,
                                           n_filters, overlap, q)
        return transform


def build_wavelet_transform(X, phi_X, lambda_X,
                            Y, phi_Y, lambda_Y,
                            n_filters, overlap,
                            q=None, n_jobs=1):
    transform = correlate_wavelets(X, phi_X, lambda_X,
                                   Y, phi_Y, lambda_Y,
                                   n_filters, overlap,
                                   q=q, n_jobs=n_jobs)
    transform_orth = math.orthogonalize(transform)
    return transform_orth


def combine_eigenvectors(transform, phi_X, phi_Y, lambda_X, lambda_Y, t=1):
    # phi_X in span(phi_Y)
    phi_X_transform = phi_X.dot(transform)
    #  phi_Y in span(phi_X)
    phi_Y_transform = phi_Y.dot(transform.T)
    # what is E?
    E = np.vstack([np.hstack([phi_X, phi_X_transform]),
                   np.hstack([phi_Y_transform, phi_Y])])
    # weight by low passed eigenvalues
    E_weighted = E * np.exp(-t * np.concatenate([lambda_X, lambda_Y]))
    return E_weighted


class HarmonicAlignment(object):
    """Harmonic alignment

    Parameters
    ----------
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
    """

    def __init__(self, n_filters, overlap=2, t=1,
                 knn=5, decay=20, n_pca=100,
                 n_jobs=1, verbose=False, random_state=None,
                 knn_X=None, knn_Y=None, knn_XY=None,
                 decay_X=None, decay_Y=None, decay_XY=None,
                 n_pca_X=None, n_pca_Y=None, n_pca_XY=0):
        self.n_filters = n_filters
        self.overlap = overlap
        self.t = t
        self.n_jobs = joblib.effective_n_jobs(n_jobs=n_jobs)
        self.random_state = random_state
        self.verbose = verbose
        self.knn_X = utils.with_default(knn_X, knn)
        self.knn_Y = utils.with_default(knn_Y, knn)
        self.knn_XY = utils.with_default(knn_XY, knn)
        self.decay_X = utils.with_default(decay_X, decay)
        self.decay_Y = utils.with_default(decay_Y, decay)
        self.decay_XY = utils.with_default(decay_XY, decay)
        self.n_pca_X = utils.with_default(
            n_pca_X, n_pca) if n_pca_X != 0 else None
        self.n_pca_Y = utils.with_default(
            n_pca_Y, n_pca) if n_pca_Y != 0 else None
        self.n_pca_XY = utils.with_default(
            n_pca_XY, n_pca) if n_pca_XY != 0 else None
        tasklogger.set_level(self.verbose)
        super().__init__()

    def align(self, X, Y):
        """Harmonic alignment

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            Input dataset
        Y : array-like, shape=[m_samples, n_features]
            Input dataset

        Returns
        -------
        XY_aligned : array-like, shape=[n_samples + m_samples, n_samples + m_samples - 1]
        """
        tasklogger.log_start("Harmonic Alignment")
        np.random.seed(self.random_state)
        # normalized L with diffusion coordinates
        with parallel.ParallelQueue(n_jobs=min(2, self.n_jobs)) as q:
            tasklogger.log_start("diffusion coordinates")
            q.queue(math.diffusionCoordinates, X,
                    decay=self.decay_X,
                    knn=self.knn_X,
                    n_pca=self.n_pca_X,
                    n_jobs=max(self.n_jobs // 2, 1),
                    verbose=self.verbose,
                    random_state=self.random_state)
            q.queue(math.diffusionCoordinates, Y,
                    decay=self.decay_Y,
                    knn=self.knn_Y,
                    n_pca=self.n_pca_Y,
                    n_jobs=max(self.n_jobs // 2, 1),
                    verbose=self.verbose,
                    random_state=self.random_state)
            (phi_X, lambda_X), (phi_Y, lambda_Y) = q.run()
            tasklogger.log_complete("diffusion coordinates")
            # evaluate wavelets over data in the spectral domain
            tasklogger.log_start("wavelets")
            transform = build_wavelet_transform(
                X, phi_X, lambda_X,
                Y, phi_Y, lambda_Y,
                self.n_filters, self.overlap, q=q)
            tasklogger.log_complete("wavelets")
        #  compute transformed data
        tasklogger.log_start("transformed data")
        E = combine_eigenvectors(transform, phi_X, phi_Y,
                                 lambda_X, lambda_Y, t=self.t)
        # build the joint diffusion map
        tasklogger.log_start("diffusion coordinates")
        phi_transform, lambda_transform = math.diffusionCoordinates(
            E, self.decay_XY, self.knn_XY, self.n_pca_XY,
            n_jobs=self.n_jobs, verbose=self.verbose,
            random_state=self.random_state)
        XY_aligned = math.diffusionMap(phi_transform, lambda_transform)
        tasklogger.log_complete("diffusion coordinates")
        tasklogger.log_complete("transformed data")
        tasklogger.log_complete("Harmonic Alignment")
        return XY_aligned
