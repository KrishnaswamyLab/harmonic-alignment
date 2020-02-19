from __future__ import division
from sklearn import datasets, decomposition
from sklearn.utils.testing import assert_raise_message
from scipy import stats, sparse
import graphtools
import numpy as np
import harmonicalignment
import unittest

import warnings

warnings.filterwarnings(
    "ignore",
    category=PendingDeprecationWarning,
    message="the matrix subclass is not the recommended way to represent "
    "matrices or deal with linear algebra ",
)


def test_dm_without_align(self):
    assert_raise_message(
        RuntimeError,
        "No alignment performed. " "Please call HarmonicAlignment.align() first.",
        harmonicalignment.HarmonicAlignment.diffusion_map,
    )
