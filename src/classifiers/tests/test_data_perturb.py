import unittest
import numpy as np

from data_perturb import CDataPerturbRandom, CDataPerturbGaussian, CDataPerturb


class TestPerturb(unittest.TestCase):

    def setUp(self):
        self.random_perturb = CDataPerturbRandom(K=50)
        self.gaussian_perturb = CDataPerturbGaussian()

    def test_negative_normal(self):
        self.assertRaises(ValueError, CDataPerturbRandom, K=-50)

    def test_data_perturbation_normal(self):
        x = np.zeros((10,))
        x_compare = self.random_perturb.data_perturbation(x=x.copy())
        self.assertNotEqual(x.all(), x_compare.all())

    def test_data_perturbation_gaussian(self):
        x = np.zeros((10,))
        z = self.gaussian_perturb.data_perturbation(x=x)
        print(x, "\n", z)
        self.assertNotEqual(x.any(), z.any())

    def test_abstract_perturb(self):
        # todo implement testing the abstract method
        pass
