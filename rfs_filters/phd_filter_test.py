import unittest
import numpy as np
from phd_filter import gauss_merge


class GaussianMixtureTest(unittest.TestCase):

    def test_gauss_merge(self):
        weights = np.array([0.4, 0.1, 0.5])
        means = np.array([
            [0, 0],
            [1, 0],
            [0, 5],
        ], dtype=np.float).T
        covs = np.dstack((0.5*np.eye(2), 0.25*np.eye(2), np.eye(2)))
        new_weights, new_means, new_covs = gauss_merge(weights, means, covs)
        pass


if __name__ == '__main__':
    unittest.main()
