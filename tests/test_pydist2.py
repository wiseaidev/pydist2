import sys
import unittest

import numpy as np

from pydist2.distance import (Chebychev, ChiSquaredDistance, CityBlock,
                              Correlation, Cosine, CosineDistance,
                              EarthMoversDistance, Euclidean,
                              EuclideanDistance, Hamming, Jaccard, L1Distance,
                              Mahalanobis, Minkowski,
                              PairwiseDistanceDescriptor, SpearmanCorrelation,
                              SquaredEuclideanDistance, StandardizedEuclidean,
                              VectorsDistanceDescriptor)


class VectorsDistanceDescriptorTest(unittest.TestCase):
    def test_exception(self):
        with self.assertRaises(TypeError):
            VectorsDistanceDescriptor('Euclidean')


class PairwiseDistanceDescriptorTest(unittest.TestCase):
    def test_exception(self):
        with self.assertRaises(TypeError):
            PairwiseDistanceDescriptor('Euclidean')


class L1Test(unittest.TestCase):
    def test_metric(self):
        l1_distance = L1Distance()
        metric = l1_distance.metric
        self.assertEqual(metric, 'L1 Distance')
        l1_distance.metric = 'L1'
        self.assertEqual(l1_distance.metric, 'L1')

    def test_set_metric_exception(self):
        l1_distance = L1Distance()
        with self.assertRaises(TypeError):
            l1_distance.metric = 13

    def test_compute(self):
        l1_distance = L1Distance()
        x = np.array([
            [1, 2, 3],
            [7, 8, 9],
            [5, 6, 7]]
        )
        y = np.array([
            [10, 20, 30],
            [70, 80, 90],
            [50, 60, 70]]
        )
        ref = np.array([
            [54., 234., 174.],
            [36., 216., 156.],
            [42., 222., 162.]]
        )
        value = l1_distance.compute(x, y)
        self.assertTrue(np.allclose(value, ref))

    def test__repr__(self):
        l1_distance = L1Distance()
        ref = "{'metric': 'L1 Distance'}"
        value = repr(l1_distance)
        self.assertEqual(value, ref)


class SquaredEuclideanDistanceTest(unittest.TestCase):

    def test_metric(self):
        sqeuclidean_distance = SquaredEuclideanDistance('SEuclidean Distance')
        metric = sqeuclidean_distance.metric
        self.assertEqual(metric, 'SEuclidean Distance')
        sqeuclidean_distance.metric = 'Squared Euclidean Distance'
        self.assertEqual(sqeuclidean_distance.metric, 'Squared Euclidean Distance')

    def test_set_metric_exception(self):
        sqeuclidean_distance = SquaredEuclideanDistance()
        with self.assertRaises(TypeError):
            sqeuclidean_distance.metric = 13

    def test_compute(self):
        sqeuclidean_distance = SquaredEuclideanDistance()
        x = np.array([
            [1, 2, 3],
            [7, 8, 9],
            [5, 6, 7]]
        )
        y = np.array([
            [10, 20, 30],
            [70, 80, 90],
            [50, 60, 70]]
        )
        ref = np.array([
            [1134., 18414., 10254.],
            [594., 15714., 8274.],
            [750., 16590., 8910.]]
        )

        value = sqeuclidean_distance.compute(x, y)
        self.assertTrue(np.allclose(value, ref))

    def test__repr__(self):
        sqeuclidean_distance = SquaredEuclideanDistance()
        ref = "{'metric': 'Squared Euclidean Distance'}"
        value = repr(sqeuclidean_distance)
        self.assertEqual(value, ref)


class EuclideanDistanceTest(unittest.TestCase):

    def test_metric(self):
        euclidean_distance = EuclideanDistance('Euclidean Distance')
        metric = euclidean_distance.metric
        self.assertEqual(metric, 'Euclidean Distance')
        euclidean_distance.metric = 'EDistance'
        self.assertEqual(euclidean_distance.metric, 'EDistance')

    def test_set_metric_exception(self):
        euclidean_distance = EuclideanDistance()
        with self.assertRaises(TypeError):
            euclidean_distance.metric = 13

    def test_compute(self):
        euclidean_distance = EuclideanDistance()
        x = np.array([
            [1, 2, 3],
            [7, 8, 9],
            [5, 6, 7]]
        )
        y = np.array([
            [10, 20, 30],
            [70, 80, 90],
            [50, 60, 70]]
        )
        ref = np.array([
            [33.67491648, 135.69819453, 101.26203632],
            [24.37211521, 125.35549449, 90.96153033],
            [27.38612788, 128.80217389, 94.39279634]]
        )
        value = euclidean_distance.compute(x, y)
        self.assertTrue(np.allclose(value, ref))

    def test__repr__(self):
        euclidean_distance = EuclideanDistance()
        ref = "{'metric': 'Euclidean Distance'}"
        value = repr(euclidean_distance)
        self.assertEqual(value, ref)


class ChiSquaredDistanceTest(unittest.TestCase):

    def test_metric(self):
        chi_square_distance = ChiSquaredDistance('Chi-Square')
        metric = chi_square_distance.metric
        self.assertEqual(metric, 'Chi-Square')
        chi_square_distance.metric = 'ChiSquared Distance'
        self.assertEqual(chi_square_distance.metric, 'ChiSquared Distance')

    def test_set_metric_exception(self):
        chi_square_distance = ChiSquaredDistance()
        with self.assertRaises(TypeError):
            chi_square_distance.metric = 13

    def test_compute(self):
        chi_square_distance = ChiSquaredDistance()
        x = np.array([
            [1, 2, 3],
            [7, 8, 9],
            [5, 6, 7]]
        )
        y = np.array([
            [10, 20, 30],
            [70, 80, 90],
            [50, 60, 70]]
        )
        ref = np.array([
            [22.09091, 111.319275, 81.414825],
            [8.489981, 88.36364, 59.652283],
            [11.751213, 95.51418, 66.27273]]
        )
        value = chi_square_distance.compute(x, y)
        self.assertTrue(np.allclose(value, ref))

    def test__repr__(self):
        chi_square_distance = ChiSquaredDistance()
        ref = "{'metric': 'Chi-Squared Distance'}"
        value = repr(chi_square_distance)
        self.assertEqual(value, ref)


class CosineDistanceTest(unittest.TestCase):

    def test_metric(self):
        cosine_distance = CosineDistance('CosDistance')
        metric = cosine_distance.metric
        self.assertEqual(metric, 'CosDistance')
        cosine_distance.metric = 'Cosine Distance'
        self.assertEqual(cosine_distance.metric, 'Cosine Distance')

    def test_set_metric_exception(self):
        cosine_distance = CosineDistance()
        with self.assertRaises(TypeError):
            cosine_distance.metric = 13

    def test_compute(self):
        cosine_distance = CosineDistance()
        x = np.array([
            [1, 2, 3],
            [7, 8, 9],
            [5, 6, 7]]
        )
        y = np.array([
            [10, 20, 30],
            [70, 80, 90],
            [50, 60, 70]]
        )
        ref = np.array([
            [0.00000000e+00, 4.05880544e-02, 3.16703363e-02],
            [4.05880544e-02, 1.11022302e-16, 5.62482467e-04],
            [3.16703363e-02, 5.62482467e-04, 0.00000000e+00]]
        )
        value = cosine_distance.compute(x, y)
        self.assertTrue(np.allclose(value, ref))

    def test__repr__(self):
        cosine_distance = CosineDistance()
        ref = "{'metric': 'Cosine Distance'}"
        value = repr(cosine_distance)
        self.assertEqual(value, ref)


class EarthMoversDistanceTest(unittest.TestCase):

    def test_metric(self):
        earth_mover_distance = EarthMoversDistance('Earth Movers Distance')
        metric = earth_mover_distance.metric
        self.assertEqual(metric, 'Earth Movers Distance')
        earth_mover_distance.metric = 'Earth Movers'
        self.assertEqual(earth_mover_distance.metric, 'Earth Movers')

    def test_set_metric_exception(self):
        earth_mover_distance = EarthMoversDistance()
        with self.assertRaises(TypeError):
            earth_mover_distance.metric = 13

    def test_compute(self):
        earth_mover_distance = EarthMoversDistance()
        x = np.array([
            [1, 2, 3],
            [7, 8, 9],
            [5, 6, 7]]
        )
        y = np.array([
            [10, 20, 30],
            [70, 80, 90],
            [50, 60, 70]]
        )
        ref = np.array([
            [90., 450., 330.],
            [54., 414., 294.],
            [66., 426., 306.]]
        )
        value = earth_mover_distance.compute(x, y)
        self.assertTrue(np.allclose(value, ref))

    def test__repr__(self):
        earth_mover_distance = EarthMoversDistance()
        ref = str({'metric': "Earth Mover\'s Distance"})
        value = repr(earth_mover_distance)
        self.assertEqual(value, ref)


class EuclideanTest(unittest.TestCase):

    def test_metric(self):
        euclidean_distance = Euclidean('Euclidean')
        metric = euclidean_distance.metric
        self.assertEqual(metric, 'Euclidean')
        euclidean_distance.metric = 'Pairwise Euclidean'
        self.assertEqual(euclidean_distance.metric, 'Pairwise Euclidean')

    def test_set_metric_exception(self):
        euclidean_distance = Euclidean()
        with self.assertRaises(TypeError):
            euclidean_distance.metric = 13

    def test_compute(self):
        euclidean_distance = Euclidean()
        src = np.array([
            [1, 2, 3],
            [7, 8, 9],
            [5, 6, 7]]
        )
        ref = np.array(
            [10.39230485, 6.92820323, 3.46410162]
        )
        value = euclidean_distance.compute(src)
        self.assertTrue(np.allclose(value, ref))

    def test__repr__(self):
        euclidean_distance = Euclidean()
        ref = "{'metric': 'Pairwise Euclidean Distance'}"
        value = repr(euclidean_distance)
        self.assertEqual(value, ref)


class StandardizedEuclideanTest(unittest.TestCase):

    def test_metric(self):
        seuclidean_distance = StandardizedEuclidean('SEuclidean')
        metric = seuclidean_distance.metric
        self.assertEqual(metric, 'SEuclidean')
        seuclidean_distance.metric = 'Pairwise SEuclidean'
        self.assertEqual(seuclidean_distance.metric, 'Pairwise SEuclidean')

    def test_set_metric_exception(self):
        seuclidean_distance = StandardizedEuclidean()
        with self.assertRaises(TypeError):
            seuclidean_distance.metric = 13

    def test_compute(self):
        seuclidean_distance = StandardizedEuclidean()
        src = np.array([
            [1, 2, 3],
            [7, 8, 9],
            [5, 6, 7]]
        )
        ref = np.array(
            [3.40168026, 2.26778684, 1.13389342]
        )
        value = seuclidean_distance.compute(src)
        self.assertTrue(np.allclose(value, ref))

    def test__repr__(self):
        seuclidean_distance = StandardizedEuclidean()
        ref = "{'metric': 'Pairwise Standardized Euclidean Distance'}"
        value = repr(seuclidean_distance)
        self.assertEqual(value, ref)


class CityBlockTest(unittest.TestCase):

    def test_metric(self):
        city_block_distance = CityBlock('CityBlock Distance')
        metric = city_block_distance.metric
        self.assertEqual(metric, 'CityBlock Distance')
        city_block_distance.metric = 'City Block'
        self.assertEqual(city_block_distance.metric, 'City Block')

    def test_set_metric_exception(self):
        city_block_distance = CityBlock()
        with self.assertRaises(TypeError):
            city_block_distance.metric = 13

    def test_compute(self):
        city_block_distance = CityBlock()
        src = np.array([
            [1, 2, 3],
            [7, 8, 9],
            [5, 6, 7]]
        )
        ref = np.array(
            [18., 12., 6.]
        )
        value = city_block_distance.compute(src)
        self.assertTrue(np.allclose(value, ref))

    def test__repr__(self):
        city_block_distance = CityBlock()
        ref = "{'metric': 'City Block Distance'}"
        value = repr(city_block_distance)
        self.assertEqual(value, ref)


class MahalanobisTest(unittest.TestCase):

    def test_metric(self):
        mahalanobis_distance = Mahalanobis('Mahalanobis Distance')
        metric = mahalanobis_distance.metric
        self.assertEqual(metric, 'Mahalanobis Distance')
        mahalanobis_distance.metric = 'Mahalanobis'
        self.assertEqual(mahalanobis_distance.metric, 'Mahalanobis')

    def test_set_metric_exception(self):
        mahalanobis_distance = Mahalanobis()
        with self.assertRaises(TypeError):
            mahalanobis_distance.metric = 13

    def test_compute(self):
        mahalanobis_distance = Mahalanobis()
        src = np.array([
            [1, 2, 3],
            [7, 8, 9],
            [5, 6, 7],
            [10, 20, 30]]
        )
        ref = np.array(
            [2.40535118, 1.60356745, 2.40535118,
             0.80178373, 2.26778684, 2.01777813]
        )
        value = mahalanobis_distance.compute(src)
        self.assertTrue(np.allclose(value, ref))

    def test__repr__(self):
        mahalanobis_distance = Mahalanobis()
        ref = "{'metric': 'Mahalanobis Distance'}"
        value = repr(mahalanobis_distance)
        self.assertEqual(value, ref)


class MinkowskiTest(unittest.TestCase):

    def test_metric(self):
        minkowski_distance = Minkowski('Minkowski Distance')
        metric = minkowski_distance.metric
        self.assertEqual(metric, 'Minkowski Distance')
        minkowski_distance.metric = 'Minkowski'
        self.assertEqual(minkowski_distance.metric, 'Minkowski')

    def test_set_metric_exception(self):
        minkowski_distance = Minkowski()
        with self.assertRaises(TypeError):
            minkowski_distance.metric = 13

    def test_compute(self):
        minkowski_distance = Minkowski()
        src = np.array([
            [1, 2, 3],
            [7, 8, 9],
            [5, 6, 7]]
        )
        ref = np.array(
            [7.47438564, 4.98292376, 2.49146188]
        )
        value = minkowski_distance.compute(src, exp=5)
        self.assertTrue(np.allclose(value, ref))
        # exp=2 ---> euclidean distance
        ref = np.array(
            [10.39230485, 6.92820323, 3.46410162]
        )
        value = minkowski_distance.compute(src, exp=2)
        self.assertTrue(np.allclose(value, ref))
        # exp=1 ---> city block distance
        ref = np.array(
            [18., 12., 6.]
        )
        value = minkowski_distance.compute(src, exp=1)
        self.assertTrue(np.allclose(value, ref))

    def test__repr__(self):
        mahalanobis_distance = Minkowski()
        ref = "{'metric': 'Minkowski Distance'}"
        value = repr(mahalanobis_distance)
        self.assertEqual(value, ref)


class ChebychevTest(unittest.TestCase):

    def test_metric(self):
        chebychev_distance = Chebychev('Chebychev Distance')
        metric = chebychev_distance.metric
        self.assertEqual(metric, 'Chebychev Distance')
        chebychev_distance.metric = 'Chebychev'
        self.assertEqual(chebychev_distance.metric, 'Chebychev')

    def test_set_metric_exception(self):
        chebychev_distance = Chebychev()
        with self.assertRaises(TypeError):
            chebychev_distance.metric = 13

    def test_compute(self):
        chebychev_distance = Chebychev()
        src = np.array([
            [1, 2, 3],
            [7, 8, 9],
            [5, 6, 7],
            [10, 20, 30]]
        )
        ref = np.array(
            [6., 4., 27., 2., 21., 23.]
        )
        value = chebychev_distance.compute(src)
        self.assertTrue(np.allclose(value, ref))

    def test__repr__(self):
        chebychev_distance = Chebychev()
        ref = "{'metric': 'Chebychev Distance'}"
        value = repr(chebychev_distance)
        self.assertEqual(value, ref)


class CosineTest(unittest.TestCase):

    def test_metric(self):
        cosine_distance = Cosine('Cosine Distance')
        metric = cosine_distance.metric
        self.assertEqual(metric, 'Cosine Distance')
        cosine_distance.metric = 'Cosine'
        self.assertEqual(cosine_distance.metric, 'Cosine')

    def test_set_metric_exception(self):
        cosine_distance = Cosine()
        with self.assertRaises(TypeError):
            cosine_distance.metric = 13

    def test_compute(self):
        cosine_distance = Cosine()
        src = np.array([
            [1, 2, 3],
            [7, 8, 9],
            [5, 6, 7],
            [10, 20, 30]]
        )
        ref = np.array(
            [0.04058805, 0.03167034, 0., 0.00056248, 0.04058805, 0.03167034]
        )
        value = cosine_distance.compute(src)
        self.assertTrue(np.allclose(value, ref))

    def test__repr__(self):
        cosine_distance = Cosine()
        ref = "{'metric': 'Cosine Distance'}"
        value = repr(cosine_distance)
        self.assertEqual(value, ref)


class CorrelationTest(unittest.TestCase):

    def test_metric(self):
        correlation_distance = Correlation('Correlation Distance')
        metric = correlation_distance.metric
        self.assertEqual(metric, 'Correlation Distance')
        correlation_distance.metric = 'Correlation'
        self.assertEqual(correlation_distance.metric, 'Correlation')

    def test_set_metric_exception(self):
        correlation_distance = Correlation()
        with self.assertRaises(TypeError):
            correlation_distance.metric = 13

    def test_compute(self):
        correlation_distance = Correlation()
        src = np.array([
            [1, 2, 3],
            [7, 8, 9],
            [5, 6, 7],
            [10, 20, 30]]
        )
        ref = np.array(
            [2.22044605e-16, 2.22044605e-16, 2.22044605e-16,
             2.22044605e-16, 2.22044605e-16, 2.22044605e-16]
        )
        value = correlation_distance.compute(src)
        self.assertTrue(np.allclose(value, ref))

    def test__repr__(self):
        correlation_distance = Correlation()
        ref = "{'metric': 'Correlation Distance'}"
        value = repr(correlation_distance)
        self.assertEqual(value, ref)


class SpearmanTest(unittest.TestCase):

    def test_metric(self):
        spearman_distance = SpearmanCorrelation('Spearman Distance')
        metric = spearman_distance.metric
        self.assertEqual(metric, 'Spearman Distance')
        spearman_distance.metric = 'Spearman'
        self.assertEqual(spearman_distance.metric, 'Spearman')

    def test_set_metric_exception(self):
        spearman_distance = SpearmanCorrelation()
        with self.assertRaises(TypeError):
            spearman_distance.metric = 13

    def test_compute(self):
        spearman_distance = SpearmanCorrelation()
        src = np.array([
            [1, 2, 3],
            [7, 8, 9],
            [5, 6, 7],
            [10, 20, 30]]
        )
        ref = np.array(
            [2.22044605e-16, 2.22044605e-16, 2.22044605e-16,
             2.22044605e-16, 2.22044605e-16, 2.22044605e-16]
        )
        value = spearman_distance.compute(src)
        self.assertTrue(np.allclose(value, ref))

    def test__repr__(self):
        spearman_distance = SpearmanCorrelation()
        ref = "{'metric': 'Spearman Distance'}"
        value = repr(spearman_distance)
        self.assertEqual(value, ref)


class HammingTest(unittest.TestCase):

    def test_metric(self):
        hamming_distance = Hamming('Hamming Distance')
        metric = hamming_distance.metric
        self.assertEqual(metric, 'Hamming Distance')
        hamming_distance.metric = 'Hamming'
        self.assertEqual(hamming_distance.metric, 'Hamming')

    def test_set_metric_exception(self):
        hamming_distance = Hamming()
        with self.assertRaises(TypeError):
            hamming_distance.metric = 13

    def test_compute(self):
        hamming_distance = Hamming()
        src = np.array([
            [1, 0, 1],
            [0, 1, 1],
            [0, 0, 1],
            [1, 1, 1]]
        )
        ref = np.array(
            [0.66666667, 0.33333333, 0.33333333,
             0.33333333, 0.33333333, 0.66666667]
        )
        value = hamming_distance.compute(src)
        self.assertTrue(np.allclose(value, ref))

    def test__repr__(self):
        hamming_distance = Hamming()
        ref = "{'metric': 'Hamming Distance'}"
        value = repr(hamming_distance)
        self.assertEqual(value, ref)


class JaccardTest(unittest.TestCase):

    def test_metric(self):
        jaccard_distance = Jaccard('Jaccard Distance')
        metric = jaccard_distance.metric
        self.assertEqual(metric, 'Jaccard Distance')
        jaccard_distance.metric = 'Jaccard'
        self.assertEqual(jaccard_distance.metric, 'Jaccard')

    def test_set_metric_exception(self):
        jaccard_distance = Jaccard()
        with self.assertRaises(TypeError):
            jaccard_distance.metric = 13

    def test_compute(self):
        jaccard_distance = Jaccard()
        src = np.array([
            [1, 0, 1],
            [0, 1, 1],
            [0, 0, 1],
            [1, 1, 1]]
        )
        ref = np.array(
            [0.66666667, 0.5, 0.33333333,
             0.5, 0.33333333, 0.66666667]
        )
        value = jaccard_distance.compute(src)
        self.assertTrue(np.allclose(value, ref))

    def test__repr__(self):
        jaccard_distance = Jaccard()
        ref = "{'metric': 'Jaccard Distance'}"
        value = repr(jaccard_distance)
        self.assertEqual(value, ref)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(verbosity=2).run(suite)
