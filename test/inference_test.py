import unittest

import jpt
import jpt.distributions.univariate
import numpy as np
import pandas as pd
import sklearn.datasets

import gmm_inference.gmm


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
        evidence = self.model.bind({"sepal length (cm)": [5, 5],
                                    "sepal width (cm)": 5})
        posterior = self.model.posterior(evidence=evidence)
        self.assertTrue(isinstance(posterior["sepal length (cm)"], jpt.distributions.univariate.Numeric))
        self.assertTrue(isinstance(posterior["sepal width (cm)"], jpt.distributions.univariate.Numeric))
        self.assertTrue(isinstance(posterior["petal length (cm)"], gmm_inference.gmm.GaussianMixture))
        self.assertTrue(isinstance(posterior["petal width (cm)"], gmm_inference.gmm.GaussianMixture))
        posterior["petal width (cm)"].expectation()
        posterior["petal width (cm)"].variance()

    def test_likelihood(self):
        likelihoods = self.model.likelihood(self.data.to_numpy())
        self.assertTrue(len(self.data), len(likelihoods))
        self.assertTrue(all(likelihoods > 0))

    def test_mpe(self):
        evidence = self.model.bind({"sepal length (cm)": [5, 5],
                                    "sepal width (cm)": [1, 6]})
        mpe, likelihood = self.model.mpe(evidence)
        self.assertEqual(len(mpe), 1)
        self.assertTrue(likelihood > 0)

    def test_conditional_gaussian(self):
        """Test the conditional gaussian using the example from https://www.youtube.com/watch?v=3DREBC6fr6g&t=3s """
        mean = np.array([1, 2, 0, 3])
        covariance = np.array([[4, 0, 1, 3],
                               [0, 4, 1, 1],
                               [1, 1, 3, 1],
                               [3, 1, 1, 9]])

        variables = [jpt.variables.NumericVariable(f"X{i}") for i in range(4)]

        model = gmm_inference.gmm.GaussianMixture(variables)

        evidence = model.bind({"X1": 2, "X3": 3})

        resulting_mean, resulting_covariance = model.conditional_multivariate_gaussian(evidence, mean, covariance)
        self.assertTrue(np.allclose(np.array([1, 0]), resulting_mean))

        covariance_result_from_exercise = 1 / 35 * np.array([[104, 26],
                                                             [26, 94]])
        self.assertTrue(np.allclose(covariance_result_from_exercise, resulting_covariance))

    def test_conditional_gmm(self):
        evidence = self.model.bind({"sepal length (cm)": [5, 5]})
        result = self.model.conditional_gmm(evidence)
        self.assertEqual(len(result.variables), 3)

    def test_expectation(self):
        result = self.model.expectation()
        self.assertTrue(np.allclose(np.array([*result.values()]), self.data.to_numpy().mean(axis=0)))
        evidence = self.model.bind({"sepal length (cm)": [5, 5]})
        result = self.model.expectation(evidence)

    def test_variance(self):
        result = self.model.variance()

        self.assertTrue(np.allclose(np.array([*result.values()]), self.data.to_numpy().var(axis=0), atol=0.001))
        evidence = self.model.bind({"sepal length (cm)": [5, 5]})

        result = self.model.variance(evidence)
        self.assertEqual(result["sepal length (cm)"], 0)

    def test_sample(self):
        samples = self.model.sample(amount=100)
        self.assertEqual(samples.shape, (100, len(self.data.columns)))
        self.assertTrue(np.all(self.model.likelihood(samples) > 0))

        evidence = self.model.bind({"sepal length (cm)": [5, 5]})
        samples = self.model.sample(evidence, 100)

        self.assertEqual(samples.shape, (100, len(self.data.columns)))
        self.assertTrue(np.all(samples[:, 0] == 5))
        self.assertTrue(np.all(self.model.likelihood(samples) > 0))

    def test_posterior_parameter_shape(self):
        posteriors = self.model.posterior()
        for index, (variable, dist) in enumerate(posteriors.items()):
            sklearn_model = sklearn.mixture.GaussianMixture(n_components=self.model.n_components)
            sklearn_model.fit(self.data.to_numpy()[:, index].reshape(-1, 1))
            self.assertEqual(sklearn_model.means_.shape, dist.means_.shape)
            self.assertEqual(sklearn_model.covariances_.shape, dist.covariances_.shape)

    def test_sample_1D(self):
        posteriors = self.model.posterior()
        for variable, dist in posteriors.items():
            samples = dist.sample(amount=100)
            self.assertTrue(all(self.model.likelihood(samples.reshape(-1, 1)) > 0))

    def test_plot_1D(self):
        posteriors = self.model.posterior()

        for variable, dist in posteriors.items():
            dist.plot().show()

    def test_plot_2D(self):
        evidence = self.model.bind({"sepal length (cm)": 5,
                                    "sepal width (cm)": 6})
        conditional_gmm = self.model.conditional_gmm(evidence)
        conditional_gmm.plot()


if __name__ == '__main__':
    unittest.main()
