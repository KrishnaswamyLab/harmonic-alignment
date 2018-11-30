# we use this script in the paper to make the corruption experiment.
from sklearn import datasets, neighbors
from scipy import stats
import numpy as np
import harmonicalignment
import harmonicalignment.math

from shutil import copyfileobj
from six.moves import urllib
from sklearn.datasets.base import get_data_home
import os


def knnclassifier(X, X_labels, Y, Y_labels, knn):
    knn_op = neighbors.KNeighborsClassifier(knn)
    knn_op.fit(X, X_labels)
    return knn_op.score(Y, Y_labels)


def fetch_mnist(data_home=None):
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    data_home = get_data_home(data_home=data_home)
    data_home = os.path.join(data_home, 'mldata')
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    mnist_save_path = os.path.join(data_home, "mnist-original.mat")
    if not os.path.exists(mnist_save_path):
        mnist_url = urllib.request.urlopen(mnist_alternative_url)
        with open(mnist_save_path, "wb") as matlab_file:
            copyfileobj(mnist_url, matlab_file)


np.random.seed(42)
fetch_mnist()
digits = datasets.fetch_mldata("MNIST original")
labels = digits['target']
imgs = digits['data']

n_samples = 1000
n_features = 784
n_iters = 1
n_percentages = 3
n_wavelets = 2

colreplace_probs = np.linspace(
    0, 1, n_percentages) if n_percentages > 1 else [1]
# scale of wavelets(eg n_filters) to use
wavelet_scales = [2, 8, 16, 64]
# kernel params
knn_1 = 20
decay_1 = 20
pca_1 = 100
knn_2 = knn_1
decay_2 = decay_1
pca_2 = pca_1
# Z = transformed
knn_transform = 10
decay_transform = 10
pca_transform = None
# diffusion time for final embedding
diffusion_t = 1
#
output = np.zeros((n_percentages, n_iters, n_wavelets, 2))
# store metrics in here
##

for p in range(n_percentages):
    # build random matrix and replace prct of columns with I
    pct = colreplace_probs[p]
    random_rotation = stats.ortho_group.rvs(n_features)
    # random orthogonal rotation
    colReplace = np.random.choice(
        n_features, np.floor(pct * n_features).astype(int), replace=False)
    random_rotation[:, colReplace] = np.eye(n_features)[:, colReplace]
    for iter_idx in range(n_iters):
        #  sample two sets of digits from MNIST
        X1_idx = np.random.choice(len(labels), n_samples, replace=False)
        X2_idx = np.random.choice(len(labels), n_samples, replace=False)
        #  slice the digits
        X1 = imgs[X1_idx, :]
        X2 = imgs[X2_idx, :]
        #  transform X2
        X2_rotate = X2 @ random_rotation.T
        X_combined = np.vstack([X1, X2_rotate])
        U_combined, S_combined = harmonicalignment.math.diffusionCoordinates(
            X_combined, decay_1, knn_1, pca_1)
        # this is for evaluating unaligned data.  You can also ploit this.
        #  slice the labels
        X1_labels = labels[X1_idx]
        X2_labels = labels[X2_idx]
        combined_labels = np.concatenate([X1_labels, X2_labels])
        #  run pca and classify
        DM_combined = U_combined @ np.diag(np.exp(-S_combined))
        beforeprct = knnclassifier(DM_combined[:n_samples, :], X1_labels,
                                   DM_combined[n_samples:, :], X2_labels, 5)
        for scale_idx in range(n_wavelets):
            n_filters = wavelet_scales[scale_idx]
            Z = harmonicalignment.HarmonicAlignment(
                n_filters, t=diffusion_t, overlap=2,
                verbose=1,
                knn_X=knn_1, knn_Y=knn_2, knn_XY=knn_transform,
                decay_X=decay_1, decay_Y=decay_2, decay_XY=decay_transform,
                n_pca_X=pca_1, n_pca_Y=pca_2,
                n_pca_XY=pca_transform).align(X1, X2_rotate)
            afterprct = knnclassifier(Z[:n_samples, :], X1_labels,
                                      Z[n_samples:, :], X2_labels, 5)
            output[p, iter_idx, scale_idx, 0] = beforeprct
            output[p, iter_idx, scale_idx, 1] = afterprct

print(output)
