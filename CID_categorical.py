from CID_base import BaseCIDExplainer
from Dissimilarity_Measure import _jaccard_distance
from Initializer import is_numerical_column


class CIDCategoricalExplainer(BaseCIDExplainer):
    """
    Counterfactual-based explainer for the categorical features of a dataset.

    See `BaseCIDExplainer` for the common parameters. The explainer is modular:

    - cf_function:
        Counterfactual generator (see `BaseCIDExplainer`).

    - dist_function:
        Dissimilarity between two samples of categorical values. It must accept
        the two samples and return a scalar.

    The default dissimilarity is the Jaccard distance.

    Categorical features are the non-numeric columns (object, category, bool).
    Integer-coded categoricals must be passed explicitly through `features_names`.
    """

    feature_kind = "categorical"

    def __init__(
        self,
        training_data,
        target_ft_name,
        model,
        features_names=None,
        cf_function="knn",
        cf_kwargs=None,
        dist_function=_jaccard_distance,
    ):
        self.dist_function = dist_function

        super().__init__(training_data, target_ft_name, model, features_names, cf_function, cf_kwargs)

    @staticmethod
    def _matches_kind(series):
        return not is_numerical_column(series)

    def _feature_importance(self, opposite_values, predicted_values):
        return self.dist_function(opposite_values, predicted_values)

    def __str__(self):
        return (
            f"Explanation method based on {self.cf_function!r} counterfactual generator "
            f"and {self.dist_function} metric"
        )
