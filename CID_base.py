import os
import warnings
from abc import ABC, abstractmethod
from contextlib import redirect_stderr, redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from Initializer import is_numerical_column, make_generator


class BaseCIDExplainer(ABC):
    """
    Shared logic of the CID explainers.

    The dataset may mix numerical and categorical columns: each explainer
    explains only the features of its own type (detected from the dtypes, or
    given explicitly through `features_names`) and keeps the others fixed
    while generating counterfactuals.

    Subclasses only define how the importance of a single feature is computed
    from its values in the opposite-class and predicted-class counterfactuals.

    Parameters
    ----------
    training_data : pandas.DataFrame
        Full dataset, including the target column.
    target_ft_name : str
        Name of the target column. Every other column is a model input.
    model : fitted estimator
        Takes a DataFrame with all the non-target columns.
    features_names : list of str, optional
        Features to explain. By default, all the columns of the explainer's type.
    cf_function : {"knn", "random", "dice"} or callable
        Counterfactual generator. A callable must accept
        ``(instance_df, amount_of_cfs)`` and return two arrays
        ``(predicted_class_data, opposite_class_data)`` whose columns follow
        `features_names`.
    cf_kwargs : dict, optional
        Extra arguments for the generator when `cf_function` is a string.
    """

    feature_kind = None  # "numerical" or "categorical"

    def __init__(
        self,
        training_data,
        target_ft_name,
        model,
        features_names=None,
        cf_function="knn",
        cf_kwargs=None,
    ):
        self.training_data = training_data
        self.target_ft_name = target_ft_name
        self.model = model
        self.model_features = [c for c in training_data.columns if c != target_ft_name]

        self.features_names = self._resolve_features(features_names)
        self.categorical_columns = self._resolve_categorical_columns()
        self.cf_function = self._resolve_cf_function(cf_function, cf_kwargs or {})

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    @staticmethod
    @abstractmethod
    def _matches_kind(series):
        """Whether a column belongs to this explainer (used for auto-detection)."""

    def _validate_features(self, features):
        """Extra checks on explicitly given features."""

    @abstractmethod
    def _feature_importance(self, opposite_values, predicted_values):
        """Importance of one feature from its counterfactual values."""

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _resolve_features(self, features_names):
        if features_names is None:
            features = [c for c in self.model_features if self._matches_kind(self.training_data[c])]
            if not features:
                raise ValueError(
                    f"No {self.feature_kind} features found in the dataset. "
                    "Pass `features_names` explicitly if they are encoded with a different dtype."
                )
            return features

        features = list(features_names)

        missing = [c for c in features if c not in self.model_features]
        if missing:
            raise KeyError(f"Features not found among the model inputs: {missing}")

        self._validate_features(features)
        return features

    def _resolve_categorical_columns(self):
        """Columns the generators must treat as categorical."""
        categorical = [c for c in self.model_features if not is_numerical_column(self.training_data[c])]

        if self.feature_kind == "categorical":
            categorical += [c for c in self.features_names if c not in categorical]

        return categorical

    def _resolve_cf_function(self, cf_function, cf_kwargs):
        if callable(cf_function):
            return cf_function

        return make_generator(
            cf_function,
            model=self.model,
            training_data=self.training_data,
            target_ft_name=self.target_ft_name,
            features_names=self.features_names,
            categorical_features=self.categorical_columns,
            **cf_kwargs,
        )

    def __str__(self):
        return f"{type(self).__name__} using {self.cf_function!r} for {len(self.features_names)} features"

    def __repr__(self):
        return self.__str__()

    # ------------------------------------------------------------------
    # Explanations
    # ------------------------------------------------------------------

    def _prepare_instance(self, instance):
        x_df = instance.to_frame().T if not isinstance(instance, pd.DataFrame) else instance

        x_df = x_df.infer_objects()

        return x_df[self.model_features]

    def explain_instance(self, instance, amount_of_cfs=50):
        """
        Local importance of each explained feature for one instance.

        Returns
        -------
        numpy.ndarray
            One value per feature in `features_names`.
        """
        if not isinstance(amount_of_cfs, (int, np.integer)) or amount_of_cfs < 1:
            raise TypeError("The amount of counterfactuals must be a positive integer!")

        x_df = self._prepare_instance(instance)

        predicted_class_data, opposite_class_data = self.cf_function(x_df, amount_of_cfs)

        return np.array([
            self._feature_importance(opposite_class_data[:, i], predicted_class_data[:, i])
            for i in range(len(self.features_names))
        ])

    def global_explanation(self, n_instances=30, amount_of_cfs=50, variability=False, random_state=None):
        """
        Global explanation: average of the local explanations over a sample of
        training instances.

        Returns
        -------
        pandas.DataFrame
            Mean importance per feature (and std if `variability` is True).
        """
        if n_instances > len(self.training_data):
            raise ValueError(
                "The amount of instances exceeds the amount of data. Please lower the value of n_instances"
            )

        sample_set = self.training_data[self.model_features].sample(n=n_instances, random_state=random_state)

        importances = []
        skipped = 0
        last_error = None

        with tqdm(range(n_instances), desc="Explaining instances") as progress:
            with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
                for i in progress:
                    try:
                        importances.append(
                            self.explain_instance(sample_set.iloc[i:i + 1], amount_of_cfs=amount_of_cfs)
                        )
                    except ValueError as error:
                        # e.g. no opposite-class counterfactual reachable for this instance
                        skipped += 1
                        last_error = error

        if skipped:
            warnings.warn(
                f"{skipped} of {n_instances} instances were skipped. Last reason: {last_error}"
            )

        if not importances:
            raise ValueError(
                "No counterfactuals could be generated for any of the sampled instances. "
                "Try another cf_function, a larger n_samples, or fewer counterfactuals."
            )

        importances = np.array(importances)

        result = {"Feature Importance": importances.mean(axis=0)}

        if variability:
            result["Std"] = importances.std(axis=0)

        return pd.DataFrame(result, index=self.features_names)

    def visualize_feature_importances(self, feature_importances, barplot=True, output=False):
        """
        Display feature importance values.

        Parameters
        ----------
        feature_importances : numpy.ndarray
            Values returned by `explain_instance`.
        barplot : bool, default=True
            Whether to display a horizontal bar plot.
        output : bool, default=False
            Whether to return the feature importance DataFrame.
        """
        if not isinstance(feature_importances, np.ndarray):
            raise TypeError("feature_importances must be a NumPy array.")

        if len(feature_importances) != len(self.features_names):
            raise ValueError("feature_importances and features_names must have the same length.")

        df = pd.DataFrame({"Feature Importance": feature_importances}, index=self.features_names)

        if barplot:
            fig, ax = plt.subplots()
            ax.barh(self.features_names[::-1], feature_importances[::-1])
            ax.set_xlabel("Feature Importance")
            ax.set_ylabel("Feature")
            plt.tight_layout()
            plt.show()

        if output:
            return df
