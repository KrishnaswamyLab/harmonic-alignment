from __future__ import division
from sklearn import datasets, decomposition
from sklearn.utils.extmath import randomized_svd
from scipy import stats, sparse
import graphtools
import numpy as np
import harmonicalignment
import unittest


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
    L = sparse.eye(n_samples) - nsqrtD.dot(W).dot(nsqrtD)
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


class TestDigits(unittest.TestCase):

    def setUp(self):
        self.random_state = 42
        np.random.seed(self.random_state)
        digits = datasets.load_digits()
        labels = digits['target']
        imgs = digits['data']

        self.n_samples = 100
        self.n_features = imgs.shape[1]
        # kernel params
        self.knn_1 = 20
        self.decay_1 = 20
        self.pca_1 = 32
        self.knn_2 = self.knn_1
        self.decay_2 = self.decay_1
        self.pca_2 = self.pca_1
        # Z = transformed
        self.knn_transform = 10
        self.decay_transform = 10
        self.pca_transform = None
        # diffusion time for final embedding
        self.diffusion_t = 1

        self.pct = 0.5
        self.n_filters = 4
        self.run_example(imgs, labels)

    def run_example(self, imgs, labels):
        self.generate_data(imgs, labels)
        self.align()

    def generate_data(self, imgs, labels):
        np.random.seed(self.random_state)
        random_rotation = stats.ortho_group.rvs(self.n_features)
        # random orthogonal rotation
        colReplace = np.random.choice(
            self.n_features, np.floor(self.pct * self.n_features).astype(int),
            replace=False)
        random_rotation[:, colReplace] = np.eye(self.n_features)[:, colReplace]
        #  sample two sets of digits from MNIST
        X1_idx = np.random.choice(len(labels), self.n_samples, replace=False)
        X2_idx = np.random.choice(len(labels), self.n_samples, replace=False)
        #  slice the digits
        self.X1 = imgs[X1_idx, :]
        X2 = imgs[X2_idx, :]
        #  transform X2
        self.X2_rotate = X2.dot(random_rotation.T)

    def align(self):
        np.random.seed(self.random_state)
        U1, S1 = diffusionCoordinates(
            self.X1, self.decay_1, self.knn_1, self.pca_1,
            random_state=self.random_state)
        U2, S2 = diffusionCoordinates(
            self.X2_rotate, self.decay_2, self.knn_2, self.pca_2,
            random_state=self.random_state)
        X1_fourier = U1.T.dot(self.X1)
        X2_rotate_fourier = U2.T.dot(self.X2_rotate)
        wavelet_1 = build_wavelets(S1, self.n_filters, 2)
        wavelet_2 = build_wavelets(S2, self.n_filters, 2)
        wavelet_1_spectral = np.conj(wavelet_1)[:, :, None] * \
            X1_fourier[:, None, :]
        wavelet_2_spectral = np.conj(wavelet_2)[:, :, None] * \
            X2_rotate_fourier[:, None, :]
        #  correlate the spectral domain wavelet coefficients.
        blocks = np.zeros((wavelet_1_spectral.shape[0], self.n_filters,
                           wavelet_2_spectral.shape[0]))
        for i in range(self.n_filters):  # for each filter, build a correlation
            blocks[:, i, :] = wavelet_1_spectral[
                :, i, :].dot(wavelet_2_spectral[:, i, :].T)

        #  construct transformation matrix
        transform = np.sum(blocks, axis=1)

        # the orthogonal transformation matrix
        Ut, _, Vt = randSVD(transform, random_state=self.random_state)
        transform_orth = Ut.dot(Vt.T)
        #  compute transformed data
        U1_transform = U1.dot(transform_orth)
        # U1 in span(U2)
        U2_transform = U2.dot(transform_orth.T)
        #  U2 in span(U1)
        E = np.vstack([np.hstack([U1, U1_transform]),
                       np.hstack([U2_transform, U2])])
        E_weighted = E.dot(np.diag(np.exp(
            -self.diffusion_t * np.concatenate([S1, S2]))))
        # store output
        self.U1 = U1
        self.U2 = U2
        self.S1 = S1
        self.S2 = S2
        self.wavelet_1_spectral = wavelet_1_spectral
        self.wavelet_2_spectral = wavelet_2_spectral
        self.transform = transform
        self.transform_orth = transform_orth
        self.E_weighted = E_weighted

    def test_harmonicalignment(self):
        Z = harmonicalignment.HarmonicAlignment(
            self.n_filters, t=self.diffusion_t, overlap=2,
            verbose=0, random_state=self.random_state,
            knn_X=self.knn_1, knn_Y=self.knn_2, knn_XY=self.knn_transform,
            decay_X=self.decay_1, decay_Y=self.decay_2,
            decay_XY=self.decay_transform, n_pca_X=self.pca_1, n_pca_Y=self.pca_2,
            n_pca_XY=self.pca_transform).align(self.X1, self.X2_rotate)
        assert Z.shape == (self.n_samples * 2, self.n_samples * 2 - 1)

    def test_diffusion_coords(self):
        np.testing.assert_allclose(
            self.S1, harmonicalignment.math.diffusionCoordinates(
                self.X1, knn=self.knn_1, decay=self.decay_1, n_pca=self.pca_1,
                random_state=self.random_state)[1], atol=1e-3, rtol=1e-2)

    def test_diffusion_map(self):
        np.testing.assert_equal(
            self.U1 * np.exp(-self.diffusion_t * self.S1),
            harmonicalignment.math.diffusionMap(self.U1, self.S1,
                                                t=self.diffusion_t))

    def test_wavelets(self):
        np.testing.assert_equal(
            self.wavelet_1_spectral,
            harmonicalignment.evaluate_itersine_wavelets(
                self.X1, self.U1, self.S1,
                self.n_filters, overlap=2))

    def test_correlate(self):
        np.testing.assert_equal(
            self.transform,
            harmonicalignment.correlate_wavelets(
                self.wavelet_1_spectral, self.wavelet_2_spectral,
                self.n_filters))

    def test_orthogonalize(self):
        np.testing.assert_allclose(
            self.transform_orth,
            harmonicalignment.math.orthogonalize(self.transform))

    def test_combine(self):
        np.testing.assert_equal(
            self.E_weighted,
            harmonicalignment.combine_eigenvectors(
                self.transform_orth, self.U1, self.U2,
                self.S1, self.S2, t=self.diffusion_t))
