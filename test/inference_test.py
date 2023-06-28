import unittest
import gmm_inference.gmm
import pandas as pd
import sklearn.datasets
import jpt
import numpy as np


class InferenceTestCase(unittest.TestCase):

    model: gmm_inference.gmm.GaussianMixture
    data: pd.DataFrame

    @classmethod
    def setUpClass(cls) -> None:
        np.random.seed(69)
        dataset = sklearn.datasets.load_iris()
        df = pd.DataFrame(columns=dataset.feature_names, data=dataset.data)

        target = dataset.target.astype(object)
        for idx, target_name in enumerate(dataset.target_names):
            target[target == idx] = target_name

        df["plant"] = target
        del df["plant"]

        cls.data = df
        cls.model = gmm_inference.gmm.GaussianMixture(jpt.infer_from_dataframe(df, scale_numeric_types=False),
                                                      n_components=3)
        cls.model.fit(df)

    def test_cdf(self):
        cdf_values = np.array([self.model.cdf(point) for point in self.data.to_numpy()])
        self.assertTrue(np.all(cdf_values > 0))

    def test_cdf_full_infinity(self):
        point = np.full(len(self.data.columns), fill_value=np.inf)
        self.assertEqual(1., self.model.cdf(point))

    def test_cdf_partial_infinity(self):
        mean_point = self.data.mean().to_numpy()
        for idx, mean in enumerate(mean_point):
            point = np.full(len(self.data.columns), fill_value=np.inf)
            point[idx] = mean
            cdf_value = self.model.cdf(point)
            self.assertTrue(0 < cdf_value < 1)

    def test_infer_trivial(self):
        query = self.model.bind(dict())
        prob = self.model.infer(query)
        self.assertEqual(prob, 1)

    def test_infer(self):
        query = self.model.bind({"sepal length (cm)": [5, 6]})
        prob = self.model.infer(query)
        self.assertTrue(prob > 0)

    def test_infer_conditional(self):
        query = self.model.bind({"sepal length (cm)": [5, 6]})
        evidence = self.model.bind({"sepal width (cm)": [5, 6]})
        prob = self.model.infer(query, evidence)
        self.assertTrue(prob > 0)

    def test_infer_impossible(self):
        query = self.model.bind({"sepal length (cm)": [5, 6]})
        evidence = self.model.bind({"sepal length (cm)": [7, 8]})
        prob = self.model.infer(query, evidence)
        self.assertAlmostEqual(prob, 0)

    def test_infer_single(self):
        query = self.model.bind({"sepal length (cm)": [5, 5],
                                 "sepal width (cm)": [5, 6]})
        prob = self.model.infer(query)
        self.assertTrue(prob > 0)

    def test_posterior(self):
        query = self.model.bind({"sepal length (cm)": [5, 5],
                                 "sepal width (cm)": [5, 6]})

if __name__ == '__main__':
    unittest.main()
