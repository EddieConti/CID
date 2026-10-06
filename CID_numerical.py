import numpy as np
from pandas.api.types import is_numeric_dtype

from CID_base import BaseCIDExplainer
from Dissimilarity_Measure import  _ecdf, _wasserstein
from Initializer import is_numerical_column


class CIDNumericalExplainer(BaseCIDExplainer):
    """
    Counterfactual-based explainer for the numerical features of a dataset.

    See `BaseCIDExplainer` for the common parameters. The explainer is modular:

    - cf_function:
        Counterfactual generator (see `BaseCIDExplainer`).

    - distr_approx:
        Approximates the distributions of a feature from the two counterfactual
        samples. It must accept two samples and return the two approximated
        distributions together with the evaluation points (or directly a scalar
        dissimilarity, for degenerate cases).

    - dist_function:
        Dissimilarity between the two approximated distributions. It must accept
        the two distributions and their evaluation points and return a scalar.

    Defaults: KDE distribution approximation and continuous Jaccard dissimilarity.
    """

    feature_kind = "numerical"

    def __init__(
        self,
        training_data,
        target_ft_name,
        model,
        features_names=None,
        cf_function="knn",
        cf_kwargs=None,
        distr_approx=_ecdf,
        dist_function=_wasserstein,
    ):
        self.distr_approx = distr_approx
        self.dist_function = dist_function

        super().__init__(training_data, target_ft_name, model, features_names, cf_function, cf_kwargs)

    @staticmethod
    def _matches_kind(series):
        return is_numerical_column(series)

    def _validate_features(self, features):
        for column in features:
            if not is_numeric_dtype(self.training_data[column]):
                raise TypeError(
                    f"Feature '{column}' is not numerical. "
                    "Please convert it or pass a different set of columns."
                )

    def _feature_importance(self, opposite_values, predicted_values):
        result = self.distr_approx(opposite_values, predicted_values)

        # Degenerate cases: the approximation already returns the dissimilarity
        if np.isscalar(result):
            return float(result)

        func_1, func_2, points = result

        return self.dist_function(func_1, func_2, points)

    def __str__(self):
        return (
            f"Explanation method based on {self.cf_function!r} counterfactual generator, "
            f"{self.distr_approx} distribution approximation and {self.dist_function} metric"
        )
