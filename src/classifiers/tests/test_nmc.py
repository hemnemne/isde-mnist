import unittest
import numpy as np

from classifiers import NMC


class TestNMC(unittest.TestCase):

    def setUp(self):
        self.x = np.zeros(shape=(10, 2))
        self.y = np.zeros(shape=(10,))
        self.clf = NMC()

    def test_fit(self):
        self.clf.fit(self.x, self.y)
        out = self.clf.centroids
        self.assertIsNotNone(out)
        self.assertEqual(out.shape, (1, 2))
        self.assertTrue(out.sum() == 0)

    def test_predict(self):
        # Test for None
        self.assertRaises(ValueError, self.clf.predict, self.x)

        # Set the centroids for the 2nd Test:
        self.clf._centroids = np.zeros(shape=(np.unique(self.y).size, self.x.shape[1]))
        y_pred = self.clf.predict(self.x)
        self.assertEqual(self.y.shape, y_pred.shape)
