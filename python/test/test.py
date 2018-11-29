from __future__ import division
from sklearn import datasets, decomposition
from sklearn.utils.extmath import randomized_svd
from scipy import stats, sparse
import graphtools
import numpy as np
import harmonicalignment


def randPCA(X, n_components=None, random_state=None):
    pca_op = decomposition.PCA(n_components, random_state=random_state)
    U, S, VT = pca_op._fit(X)
    return U, S, VT.T


def randSVD(X, n_components=None, random_state=None):
    if n_components is None:
        n_components = min(X.shape)
    U, S, VT = randomized_svd(X, n_components, random_state=random_state)
    return U, S, VT.T


def diffusionCoordinates(X, decay, knn, n_pca, random_state=None):
    # diffusion maps with normalized Laplacian
    # n_pca = 0 corresponds to NO pca
    G = graphtools.Graph(X, knn=knn, decay=decay,
                         n_pca=n_pca, use_pygsp=True, thresh=0,
                         random_state=random_state)
    n_samples = X.shape[0]
    W = G.W.tocoo()
    # W / (DD^T)
    W.data = W.data / (G.dw[W.row] * G.dw[W.col])
    # this is the anisotropic kernel
    nsqrtD = sparse.dia_matrix((np.array(np.sum(W, 0)) ** (-0.5), [0]),
                               W.shape)
    L = sparse.eye(n_samples) - nsqrtD @ W @ nsqrtD
    U, S, _ = randSVD(L, random_state=random_state)
    # smallest to largest
    S_idx = np.argsort(S)
    U, S = U[:, S_idx], S[S_idx]
    #  trim trivial information
    U, S = U[:, 1:], S[1:]
    return U, S


def itersine(x):
    return np.sin(0.5 * np.pi * (np.cos(np.pi * x))**2) * \
        ((x >= -0.5) & (x <= 0.5))


def wavelet(loc, scale, overlap):
    def wavelet_i(x):
        return itersine(x / scale - loc / overlap + 1 / 2) * np.sqrt(2 / overlap)
    return wavelet_i


def build_wavelets(lmbda, n_filters, overlap):
    lambda_max = max(lmbda)
    # maximum laplacian eigenvalue
    scale = lambda_max / (n_filters - overlap + 1) * (overlap)
    # response evaluation... this is the money
    lambda_filt = np.zeros((len(lmbda), n_filters))
    for i in range(n_filters):
        filter_i = wavelet(loc=i + 1, scale=scale, overlap=overlap)
        lambda_filt[:, i] = filter_i(lmbda)
    return lambda_filt


def test_digits():
    random_state = 42
    np.random.seed(random_state)
    digits = datasets.load_digits()
    labels = digits['target']
    imgs = digits['data']

    n_samples = 100
    n_features = 64
    # kernel params
    knn_1 = 20
    decay_1 = 20
    pca_1 = 32
    knn_2 = knn_1
    decay_2 = decay_1
    pca_2 = pca_1
    # Z = transformed
    knn_transform = 10
    decay_transform = 10
    pca_transform = None
    # diffusion time for final embedding
    diffusion_t = 1

    pct = 0.8
    n_filters = 4

    random_rotation = stats.ortho_group.rvs(n_features)
    # random orthogonal rotation
    colReplace = np.random.choice(
        n_features, np.floor(pct * n_features).astype(int), replace=False)
    random_rotation[:, colReplace] = np.eye(n_features)[:, colReplace]
    #  sample two sets of digits from MNIST
    X1_idx = np.random.choice(len(labels), n_samples, replace=False)
    X2_idx = np.random.choice(len(labels), n_samples, replace=False)
    #  slice the digits
    X1 = imgs[X1_idx, :]
    X2 = imgs[X2_idx, :]
    #  transform X2
    X2_rotate = X2.dot(random_rotation.T)

    Z = harmonicalignment.align(
        X1, X2_rotate, n_filters, t=diffusion_t, overlap=2,
        verbose=0, random_state=random_state,
        knn_X=knn_1, knn_Y=knn_2, knn_XY=knn_transform,
        decay_X=decay_1, decay_Y=decay_2, decay_XY=decay_transform,
        n_pca_X=pca_1, n_pca_Y=pca_2, n_pca_XY=pca_transform)

    np.random.seed(random_state)
    U1, S1 = diffusionCoordinates(
        X1, decay_1, knn_1, pca_1, random_state=random_state)
    # normalized L with diffusion coordinates for sample 1
    U2, S2 = diffusionCoordinates(
        X2_rotate, decay_2, knn_2, pca_2, random_state=random_state)
    # ... sample 2
    #  get fourier coefficients
    X1_fourier = U1.T @ X1
    X2_rotate_fourier = U2.T @ X2_rotate
    #  build wavelets
    wavelet_1 = build_wavelets(S1, n_filters, 2)
    wavelet_2 = build_wavelets(S2, n_filters, 2)
    # wavelet_1 is the filter evaluated over the eigenvalues.  So we
    # can pointwise multiply each wavelet_1 / 2 by the fourier
    # coefficients
    #  evaluate wavelets over data in the spectral domain
    #  stolen from gspbox, i have no idea how the fuck this works
    wavelet_1_spectral = np.conj(wavelet_1)[:, :, None] * \
        X1_fourier[:, None, :]
    wavelet_2_spectral = np.conj(wavelet_2)[:, :, None] * \
        X2_rotate_fourier[:, None, :]
    #  correlate the spectral domain wavelet coefficients.
    blocks = np.zeros((wavelet_1_spectral.shape[0], n_filters,
                       wavelet_2_spectral.shape[0]))
    for i in range(n_filters):  # for each filter, build a correlation
        blocks[:, i, :] = wavelet_1_spectral[:, i, :] @ \
            wavelet_2_spectral[:, i, :].T

    #  construct transformation matrix
    transform = np.sum(blocks, axis=1)
    # sum wavelets up
    Ut, St, Vt = randPCA(transform, random_state=random_state)
    # this is random svd
    St = St[St > 0]
    # this is from earlier experiments where I was truncating by rank.
    #  We can probably remove this.
    transform_rank = len(St)
    Ut = Ut[:, :transform_rank]
    Vt = Vt[:, :transform_rank]
    transform_orth = Ut @ Vt.T
    # the orthogonal transformation matrix
    #  compute transformed data
    U1_transform = U1 @ transform_orth
    # U1 in span(U2)
    U2_transform = U2 @ transform_orth.T
    #  U2 in span(U1)
    E = np.vstack([np.hstack([U1, U1_transform]),
                   np.hstack([U2_transform, U2])])
    X = E @ np.diag(np.exp(-diffusion_t * np.concatenate([S1, S2])))
    U_transform, S_transform = diffusionCoordinates(
        X, decay_transform, knn_transform,
        pca_transform, random_state=random_state)
    Z2 = U_transform @ np.diag(np.exp(-S_transform))

    np.testing.assert_allclose(Z, Z2, atol=1e-12, rtol=1e-12)
