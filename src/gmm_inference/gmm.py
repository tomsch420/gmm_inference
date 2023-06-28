import itertools
from typing import Union, Dict, Any, List, Optional

import jpt.trees
from jpt.base.intervals import R, ContinuousSet, RealSet
import numpy as np
from scipy.stats import multivariate_normal
import sklearn.mixture
from jpt.variables import Variable, VariableAssignment, VariableMap, LabelAssignment, ValueAssignment


class GaussianMixture(sklearn.mixture.GaussianMixture):

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
        return jpt.JPT(self.variables).bind(query)

    def fill_missing_variables(self, query: Optional[VariableAssignment] = None) -> VariableAssignment:
        """
        Fill all not specified variables with the Real line as value.
        :param query: The query to read from
        :return: The filled query
        """

        if query is None:
            return LabelAssignment({variable: R for variable in self.variables}.items())

        for variable in self.variables:
            if variable.numeric:
                if variable not in query:
                    query[variable] = R
            else:
                raise ValueError(f"GMM inference not implemented for {type(variable)}.")

        return query

    def posterior(self,
                  variables: List[Variable or str] = None,
                  evidence: Dict[Union[Variable, str], Any] or VariableAssignment = None,
                  fail_on_unsatisfiability: bool = True) -> Optional[VariableMap]:

        # default variables to all variables
        if variables is None:
            variables = self.variables

        # make evidence usable
        evidence = self.fill_missing_variables(evidence)
        if isinstance(evidence, LabelAssignment):
            evidence = evidence.value_assignment()

        # check if evidence is possible
        p_evidence = self.probability(evidence)
        if p_evidence == 0:
            if fail_on_unsatisfiability:
                raise jpt.trees.Unsatisfiability()
            else:
                return None

        result = VariableMap(variables=variables)

        raise NotImplementedError("This has to be calculated still.")

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
        return p_query_and_evidence/p_evidence

    def probability(self, query: VariableAssignment) -> float:
        query = self.fill_missing_variables(query)
        if isinstance(query, LabelAssignment):
            query = query.value_assignment()

        sets = []

        for variable in self.variables:
            value = query[variable]
            if variable.numeric:
                if isinstance(value, ContinuousSet):
                    sets.append([value])
                elif isinstance(value, RealSet):
                    sets.append(value.intervals)
                else:
                    raise ValueError(f"Unknown type for numeric variable: {type(value)}")

        result = 0.
        for query_ in itertools.product(*sets):
            upper = np.array([element.upper for element in query_])
            lower = np.array([element.lower for element in query_])
            result += self.cdf(upper) - self.cdf(lower)
        return result

    def mpe(self, evidence: Union[Dict[Union[Variable, str], Any], VariableAssignment] = None,
            fail_on_unsatisfiability: bool = True) -> (List[LabelAssignment], float) or None:
        ...
