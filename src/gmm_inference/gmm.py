import copy
import itertools
from typing import Union, Dict, Any, List, Optional, Tuple

import jpt.distributions.univariate
import jpt.trees
import numpy as np
import sklearn.mixture
from jpt.base.intervals import R, ContinuousSet, RealSet
from jpt.variables import Variable, VariableAssignment, VariableMap, LabelAssignment, ValueAssignment
from scipy.stats import multivariate_normal
import plotly.graph_objects as go


class GaussianMixture(sklearn.mixture.GaussianMixture):
    """Implementation of Gaussian Mixtures that are queryable for more than just likelihoods…"""

    def __init__(
            self,
            variables: List[Variable],
            n_components=1,
            *,
            covariance_type="full",
            tol=1e-3,
            reg_covar=1e-6,
            max_iter=100,
            n_init=1,
            init_params="kmeans",
            weights_init=None,
            means_init=None,
            precisions_init=None,
            random_state=None,
            warm_start=False,
            verbose=0,
            verbose_interval=10,
    ):
        super().__init__(n_components=n_components, covariance_type=covariance_type, tol=tol, reg_covar=reg_covar,
                         max_iter=max_iter, n_init=n_init, init_params=init_params, weights_init=weights_init,
                         means_init=means_init, precisions_init=precisions_init, random_state=random_state,
                         warm_start=warm_start, verbose=verbose, verbose_interval=verbose_interval)
        self.variables = variables

    def bind(self, query: Dict[str, Any]) -> LabelAssignment:
        """
        Convert a dictionary into a variable assignment.
        :param query: The dictionary to construct the query from
        :return: The usable variable assignment
        """
        return jpt.JPT(self.variables).bind(query)

    def fill_missing_variables(self, query: Optional[VariableAssignment] = None) -> VariableAssignment:
        """
        Fill all not specified variables with the Real line as value.
        :param query: The query to read from
        :return: The filled query
        """

        if query is None:
            return LabelAssignment(list({variable: R for variable in self.variables}.items()))

        result = LabelAssignment(variables=self.variables)

        for variable in self.variables:
            if variable.numeric:
                if variable not in query:
                    result[variable] = R
                else:
                    result[variable] = query[variable]
            else:
                raise ValueError(f"GMM inference not implemented for {type(variable)}.")

        return result

    def posterior(self,
                  variables: List[Variable or str] = None,
                  evidence: Dict[Union[Variable, str], Any] or VariableAssignment = None,
                  fail_on_unsatisfiability: bool = True) -> Optional[VariableMap]:
        """
        Calculate the independent posterior distributions given the evidence.
        :param variables: The variables to calculate the distributions on
        :param evidence: The evidence to apply
        :param fail_on_unsatisfiability: Rather to raise an error of the evidence is impossible or not.
        :return: A VariableMap mapping to dirac impulses or 1D Gaussian mixtures.
        """

        if evidence is None:
            evidence = LabelAssignment(variables=self.variables)

        if isinstance(evidence, LabelAssignment):
            evidence = evidence.value_assignment()

        if variables is None:
            variables = self.variables

        # only allow single points as evidence
        self.assert_point_description(evidence)

        # calculate conditional gmm
        conditional_gmm = self.conditional_gmm(evidence, fail_on_unsatisfiability)

        # initialize result
        result = VariableMap(variables=self.variables)

        # fill result
        for variable in variables:

            # with dirac impulses if its evidence
            if variable in evidence:
                result[variable] = jpt.distributions.univariate.Numeric().create_dirac_impulse(evidence[variable].lower)
                continue

            # and a 1D gaussian mixture if it's not evidence
            index_in_cg = conditional_gmm.variables.index(variable)

            distribution = GaussianMixture([variable])
            distribution.weights_ = conditional_gmm.weights_
            distribution.means_ = [mean[index_in_cg] for mean in conditional_gmm.means_]
            distribution.covariances_ = [covariance[index_in_cg, index_in_cg].reshape(1, -1)
                                         for covariance in conditional_gmm.covariances_]

            result[variable] = distribution

        return result

    def expectation(self, evidence: Dict[Union[Variable, str], Any] or VariableAssignment = None,
                    fail_on_unsatisfiability: bool = True) -> Optional[LabelAssignment]:
        """
        Calculate the conditional expectation
        :param evidence: The condition as a point description.
        :param fail_on_unsatisfiability: Rather to raise an exception on impossible evidence or not.
        :return: A LabelAssignment mapping from variable to its expectation.
        """

        if evidence is None:
            evidence = LabelAssignment(variables=self.variables)

        if isinstance(evidence, LabelAssignment):
            evidence = evidence.value_assignment()

        self.assert_point_description(evidence)

        conditional_gmm = self.conditional_gmm(evidence, fail_on_unsatisfiability)
        if conditional_gmm is None:
            return None

        result = ValueAssignment(variables=self.variables)

        for variable, value in evidence.items():
            result[variable] = value

        expectation = np.sum([weight * mean for weight, mean in zip(conditional_gmm.weights_, conditional_gmm.means_)],
                             axis=0)

        if len(self.variables) == 1:
            expectation = [expectation]

        for variable, expected_value in zip(conditional_gmm.variables, expectation):
            result[variable] = expected_value

        return result.label_assignment()

    def variance(self, evidence: Dict[Union[Variable, str], Any] or VariableAssignment = None,
                 fail_on_unsatisfiability: bool = True) -> Optional[LabelAssignment]:
        """
        Calculate the conditional variance of the GMM. For this it uses the formula given by
        https://stats.stackexchange.com/questions/16608/what-is-the-variance-of-the-weighted-mixture-of-two-gaussians
        :param evidence: The condition as a point description.
        :param fail_on_unsatisfiability: Rather to raise an exception on impossible evidence or not.
        :return: A LabelAssignment mapping from variable to its expectation.
        """

        if evidence is None:
            evidence = LabelAssignment(variables=self.variables)

        if isinstance(evidence, LabelAssignment):
            evidence = evidence.value_assignment()

        self.assert_point_description(evidence)

        conditional_gmm = self.conditional_gmm(evidence, fail_on_unsatisfiability)
        if conditional_gmm is None:
            return None

        result = ValueAssignment(variables=self.variables)

        for variable, value in evidence.items():
            result[variable] = 0

        average_variances = np.zeros(len(conditional_gmm.variables))
        average_square_mean = np.zeros(len(conditional_gmm.variables))
        square_average_mean = np.zeros(len(conditional_gmm.variables))

        for weight, mean, covariance in zip(conditional_gmm.weights_, conditional_gmm.means_,
                                            conditional_gmm.covariances_):
            average_variances += weight * np.diagonal(covariance)
            average_square_mean += weight * np.square(mean)
            square_average_mean += weight * mean

        square_average_mean = np.square(square_average_mean)

        variances = average_variances + average_square_mean - square_average_mean

        for variable, variance in zip(conditional_gmm.variables, variances):
            result[variable] = variance

        return result.label_assignment()

    def cdf(self, point: np.ndarray) -> float:
        """
        Evaluate the CDF of this GMM at the given point.
        :param point: The point to evaluate
        :return: The CDF value at point
        """
        result = 0.

        for weight, mean, covariance in zip(self.weights_, self.means_, self.covariances_):
            dist = multivariate_normal(mean, covariance)
            result += weight * dist.cdf(point)

        return result

    def infer(self,
              query: Union[Dict[Union[Variable, str], Any], VariableAssignment],
              evidence: Union[Dict[Union[Variable, str], Any], VariableAssignment] = None,
              fail_on_unsatisfiability: bool = True) -> float or None:
        """
        Calculate the conditional probability P(Q|E) where Q and E are both sets.
        :param query: The query description of a set as VariableAssignment
        :param evidence: The evidence description of a set as VariableAssignment
        :param fail_on_unsatisfiability:
        :return: P(Q|E)
        """
        query = self.fill_missing_variables(query)
        evidence = self.fill_missing_variables(evidence)

        if isinstance(query, LabelAssignment):
            query = query.value_assignment()

        if isinstance(evidence, LabelAssignment):
            evidence = evidence.value_assignment()

        # form the intersection of q and e
        query_and_evidence = ValueAssignment(variables=self.variables)
        for variable in self.variables:
            query_and_evidence[variable] = evidence[variable].intersection(query[variable])

        # calculate P(e)
        if evidence:
            p_evidence = self.probability(evidence)
        else:
            p_evidence = 1.

        # check if P(e) is > 0
        if p_evidence == 0:
            if fail_on_unsatisfiability:
                raise jpt.trees.Unsatisfiability()
            else:
                return None

        # calculate P(q, e)
        p_query_and_evidence = self.probability(query_and_evidence)

        # return conditional probability
        return p_query_and_evidence / p_evidence

    def probability(self, query: VariableAssignment) -> float:
        sets = self.disjoint_sets_from_assignment(query)
        result = 0.
        for query_ in itertools.product(*sets):
            upper = np.array([element.upper for element in query_])
            lower = np.array([element.lower for element in query_])
            result += self.cdf(upper) - self.cdf(lower)
        return result

    def disjoint_sets_from_assignment(self, assignment: VariableAssignment) -> List[List[ContinuousSet]]:
        """
        Transform a VariableAssignment into a list that contains the disjoint ContinuousSet of each dimension.
        This can be used to iterate over all disjoint combinations of those sets with `itertools.product` for example.
        :param assignment:
        :return: List containing lists that contain continuous sets that can be combined for the union of disjoint sets.
        """

        assignment = self.fill_missing_variables(assignment)
        if isinstance(assignment, LabelAssignment):
            assignment = assignment.value_assignment()

        sets = []

        for variable in self.variables:
            value = assignment[variable]
            if variable.numeric:
                if isinstance(value, ContinuousSet):
                    sets.append([value])
                elif isinstance(value, RealSet):
                    sets.append(value.intervals)
                else:
                    raise ValueError(f"Unknown type for numeric variable: {type(value)}")

        return sets

    def weight_encoding(self, query: VariableAssignment) -> np.ndarray:
        """
        Calculate the un-normalized probability P(sigma|X) where sigma are the weights of each distribution.
        :param query: The query to serve as X
        :return: The new distribution over Sigma.
        """
        sets = self.disjoint_sets_from_assignment(query)

        result = np.zeros(len(self.weights_))

        for idx, (weight, mean, covariance) in enumerate(zip(self.weights_, self.means_, self.covariances_)):
            dist = multivariate_normal(mean, covariance)

            current_mixture_probability = 0.
            for query_ in itertools.product(*sets):
                upper = np.array([element.upper for element in query_])
                lower = np.array([element.lower for element in query_])
                current_mixture_probability += dist.cdf(upper) - dist.cdf(lower)

            result[idx] = weight * current_mixture_probability

        return result

    def _closest_point_in_rectangle(self, rectangle: List[ContinuousSet], mean: np.ndarray) \
            -> np.ndarray:
        """
        Get the closest point to the mean that lie inside the given rectangle.
        For all dimensions the point in the rectangle that is closest to the mean is selected.

        :param rectangle: A list of continuous sets that describe a possible evidence area.
        :param mean: The mean of the multivariate gaussian.

        :return: A numpy array containing the MPE candidates
        """

        point = np.zeros(len(self.variables))
        for idx, (interval, mean_) in enumerate(zip(rectangle, mean)):

            # if the current dimensions mean is inside the interval, copy its value
            if interval.lower <= mean_ <= interval.upper:
                point[idx] = mean_

            # if the mean is smaller, take the lowest possible value
            elif mean_ < interval.lower:
                point[idx] = interval.lower

            # if the mean is bigger, take the highest possible value
            elif mean_ > interval.upper:
                point[idx] = interval.upper

            # this should never happen
            else:
                raise ValueError("Something weird happened.")

        return point

    def likelihood(self, queries: np.ndarray, weights: Optional[np.ndarray] = None):
        """
        Get the likelihoods of a list of worlds. The worlds must be fully assigned with
        single numbers (no intervals).

        :param queries: An array containing the worlds. The shape is (x, len(variables)).
        :param weights: Optional weight to use instead of the own weights
        :returns: A numpy array with shape (x, ) containing the likelihoods.
        """

        if weights is None:
            weights = self.weights_

        result = np.zeros(len(queries))

        for weight, mean, covariance in zip(weights, self.means_, self.covariances_):
            dist = multivariate_normal(mean, covariance)
            result += weight * dist.pdf(queries)

        return result

    def mpe(self, evidence: Union[Dict[Union[Variable, str], Any], VariableAssignment] = None,
            fail_on_unsatisfiability: bool = True) -> Optional[Tuple[List[LabelAssignment], float]]:
        """
        Get the most likely states given some evidence. This uses an approximation by seeking for points inside the
        specified sets that are closest to the mean.
        :param evidence: The evidence (set) to search in
        :param fail_on_unsatisfiability:
        :return: A list of LabelAssignment that maps each variable to its most likely value. The list contains multiple
            assignments if more than one solution is maximally likely (rare).
        """
        evidence = self.fill_missing_variables(evidence)

        if isinstance(evidence, LabelAssignment):
            evidence = evidence.value_assignment()

        # check if evidence is possible
        weight_encoding = self.weight_encoding(evidence)

        weight_sum = np.sum(weight_encoding)

        if weight_sum == 0:
            if fail_on_unsatisfiability:
                raise jpt.trees.Unsatisfiability()
            else:
                return None

        # normalize weights
        weight_encoding /= weight_sum

        # get all candidates from disjoint sets
        sets = self.disjoint_sets_from_assignment(evidence)
        candidate_points = [self._closest_point_in_rectangle(rectangle, mean) for rectangle in itertools.product(*sets)
                            for mean in self.means_]

        # make candidate points unique
        candidate_points = np.unique(np.array(candidate_points), axis=1)

        # rate by likelihood with new weights
        likelihoods = self.likelihood(candidate_points, weight_encoding)

        # get best solutions
        max_likelihood = np.max(likelihoods)
        results = [point for point, likelihood in zip(candidate_points, likelihoods) if likelihood == max_likelihood]

        return [ValueAssignment(
            list({variable: value for variable, value in zip(self.variables, point)}.items())).label_assignment()
                for point in results], max_likelihood

    def conditional_gmm(self, evidence: Optional[VariableAssignment] = None, fail_on_unsatisfiability: bool = True) -> \
            Optional['GaussianMixture']:
        """
        Compute the conditional GMM given some evidence. The evidence has to be a description of a (partial)
        point. For the new weights, the bayes theorem is used. The resulting GMM has the evidence variables removed.

        :param evidence: The evidence description.
        :param fail_on_unsatisfiability:
        :return: The conditional GMM
        """
        result = copy.deepcopy(self)

        if evidence is None or len(evidence) == 0:
            return result

        if isinstance(evidence, LabelAssignment):
            evidence = evidence.value_assignment()

        remaining_variables = [v for v in self.variables if v not in evidence]

        result.variables = remaining_variables

        # check if evidence is possible
        weight_encoding = self.weight_encoding(evidence)

        weight_sum = np.sum(weight_encoding)

        if weight_sum == 0:
            if fail_on_unsatisfiability:
                raise jpt.trees.Unsatisfiability()
            else:
                return None

        # normalize weights
        weight_encoding /= weight_sum

        # reset means and covariances of resulting GMM
        result.means_ = np.empty((len(self.means_), len(remaining_variables)))
        result.covariances_ = np.empty((len(self.covariances_), len(remaining_variables), len(remaining_variables)))

        # enumerate over new weights, old means and old covariances
        for index, (weight, mean, covariance) in enumerate(zip(weight_encoding, self.means_, self.covariances_)):
            # set the new weight
            result.weights_[index] = weight

            # get conditional multivariate distributions
            conditional_mean, conditional_covariance = self.conditional_multivariate_gaussian(evidence, mean,
                                                                                              covariance)
            # set conditional multivariate distributions
            result.means_[index] = conditional_mean
            result.covariances_[index] = conditional_covariance

        return result

    @staticmethod
    def assert_point_description(query: VariableAssignment):
        """
        Check that the query only contains a point description.
        :param query: A VariableAssignment
        """
        # check that the input has the required form
        for variable, value in query.items():

            if isinstance(value, RealSet):
                if len(value.intervals) > 1:
                    ValueError(f"Only single values are allowed for conditional a Gaussian. Got {value} instead.")
                value = value.intervals[0]

            if value.upper != value.lower and value != R:
                raise ValueError(f"Only single values are allowed for conditional a Gaussian. Got {value} instead.")

    def conditional_multivariate_gaussian(self, evidence: VariableAssignment, mean: np.ndarray,
                                          covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the conditional multivariate normal distribution P(Y|X) given an assignment of single values to some
        variables.
        This implementation is guided by this video https://www.youtube.com/watch?v=3DREBC6fr6g&t=3s
        :param evidence: The assignment of individual values to some variables
        :param mean: Mean of the gaussian
        :param covariance: Covariance of the gaussian
        :return: The conditional distribution
        """
        if isinstance(evidence, LabelAssignment):
            evidence = evidence.value_assignment()

        # check that the input has the required form
        GaussianMixture.assert_point_description(evidence)

        # assert plausibility of mean and covariance
        assert mean.shape[0] == covariance.shape[0] == covariance.shape[1]

        query_indices = np.array([index for index, variable in enumerate(self.variables) if variable not in evidence])
        evidence_indices = np.array([index for index, variable in enumerate(self.variables) if variable in evidence])

        # get evidence as vector for later use in new mean calculation
        evidence_vector = np.array([i.lower for i in evidence.values()])

        # construct mean_y
        mean_y = mean[query_indices]

        # add mean_x
        mean_x = mean[evidence_indices]

        # construct yy indices for covariance selection and add covariance_yy
        yy_indices = np.array(list(itertools.product(query_indices, query_indices)))
        covariance_yy = covariance[yy_indices[:, 0], yy_indices[:, 1]]. \
            reshape((len(query_indices), len(query_indices)))

        # construct xx indices for covariance selection and add covariance_xx
        xx_indices = np.array(list(itertools.product(evidence_indices, evidence_indices)))
        covariance_xx = covariance[xx_indices[:, 0], xx_indices[:, 1]]. \
            reshape((len(evidence_indices), len(evidence_indices)))

        # construct xy indices for covariance selection and add covariance_xy
        xy_indices = np.array(list(itertools.product(evidence_indices, query_indices)))
        covariance_xy = covariance[xy_indices[:, 0], xy_indices[:, 1]]. \
            reshape((len(evidence_indices), len(query_indices)))

        # construct yx indices for covariance selection and add covariance_yx
        yx_indices = np.array(list(itertools.product(query_indices, evidence_indices)))
        covariance_yx = covariance[yx_indices[:, 0], yx_indices[:, 1]]. \
            reshape((len(query_indices), len(evidence_indices)))

        # calculate inverse of xx
        covariance_xx_inverse = np.linalg.inv(covariance_xx)

        # calculate conditional mean
        conditional_mean = mean_y + (covariance_yx @ covariance_xx_inverse)[:, 0] * (evidence_vector - mean_x)

        # calculate conditional covariance
        conditional_covariance = covariance_yy - (covariance_yx @ covariance_xx_inverse @ covariance_xy)

        return conditional_mean, conditional_covariance

    def sample(self, evidence: Optional[VariableAssignment] = None, amount: int = 100) -> np.ndarray:
        """
        Sample from the GMM given the evidence.
        :param evidence: The evidence
        :param amount: The number of samples to draw.
        :return: A numpy array of shape (len(self.variables), amount) containing the samples
        """

        if evidence is None or len(evidence) == 0:
            samples, _ = super().sample(n_samples=amount)
            return samples

        if isinstance(evidence, LabelAssignment):
            evidence = evidence.value_assignment()

        GaussianMixture.assert_point_description(evidence)
        conditional_gmm = self.conditional_gmm(evidence)

        result = np.empty((amount, len(self.variables)))

        conditional_gmm_indices = [idx for idx, variable in enumerate(self.variables) if variable not in evidence]

        conditional_samples = conditional_gmm.sample(amount=amount)

        result[:, conditional_gmm_indices] = conditional_samples

        for index, variable in enumerate(self.variables):
            if variable in evidence:
                result[:, index] = evidence[variable].lower

        return result

    def plot(self, points_per_dimension: int = 1000, variance_scaling: float = 2.5) -> go.Figure:
        """
        Create a plotly figure that contains a visualization of the GMM in up to 2 dimensions.
        :return: Plotly Figure
        """

        # initialize result
        fig = go.Figure()

        if len(self.variables) == 1:
            variable = self.variables[0]
            variance = self.variance()[variable]
            expectation = self.expectation()[variable]

            leftmost = expectation - variance_scaling * variance
            rightmost = expectation + variance_scaling * variance

            points = np.linspace(leftmost, rightmost, points_per_dimension)
            likelihoods = self.likelihood(points.reshape(-1, 1))

            fig.add_trace(go.Scatter(x=points, y=likelihoods, name="Likelihood"))
            fig.update_layout(title=f"Probability Density Function of {variable.name}", showlegend=True)

        elif len(self.variables) == 2:
            pass

        else:
            raise NotImplementedError("Plotting is not supported for more than two dimensions.")
        return fig
