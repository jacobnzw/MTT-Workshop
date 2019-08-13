import unittest
import numpy as np
from phd_filter import gauss_merge, ospa


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

        self.assertTrue(np.array_equal(new_weights, np.array([0.5, 0.5])))
        self.assertTrue(np.array_equal(new_means, np.array([[0.0, 5.0], [0.2, 0]]).T))
        self.assertTrue(np.array_equal(new_covs, np.stack((np.eye(2), 0.45*np.eye(2)))))

    def test_ospa(self):
        # setup finite sets
        x = np.array([[1, 0, 1, 0],
                      [0, 1, 1, 0]], dtype=np.float)
        y = np.array([[2, 0, 2, 0],
                      [0, 2, 2, 0]], dtype=np.float)
        print(ospa(x, y))


if __name__ == '__main__':
    unittest.main()
