import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def is_numerical_column(series):
    """True for numeric columns (booleans are treated as categorical)."""
    return is_numeric_dtype(series) and not is_bool_dtype(series)


class CounterfactualGenerator:
    """
    Base class of the counterfactual generators.

    A generator is a callable ``generator(instance, amount_of_cfs)`` that returns
    two arrays ``(predicted_class_data, opposite_class_data)`` containing only the
    columns in ``features_names`` (the features being explained). All the other
    features are kept fixed at the values of the instance, so the same generator
    works on datasets mixing numerical and categorical columns.

    Parameters
    ----------
    model : fitted estimator
        Must accept a DataFrame with all the columns of `training_data` except
        the target, and expose `predict`.
    training_data : pandas.DataFrame
        Dataset (it may contain the target column).
    target_ft_name : str
        Name of the target column.
    features_names : list of str
        Features that are varied / returned.
    categorical_features : list of str, optional
        Columns to be treated as categorical. By default, every non-numeric
        column (this is where you declare integer-coded categoricals).
    """

    def __init__(self, model, training_data, target_ft_name, features_names, categorical_features=None):
        self.model = model
        self.training_data = training_data
        self.target_ft_name = target_ft_name
        self.features_names = list(features_names)
        self.model_features = [c for c in training_data.columns if c != target_ft_name]

        missing = [c for c in self.features_names if c not in self.model_features]
        if missing:
            raise KeyError(f"Features not found in training_data: {missing}")

        if categorical_features is None:
            categorical_features = [c for c in self.model_features if not is_numerical_column(training_data[c])]

        self.categorical_features = list(categorical_features)
        self.numerical_features = [c for c in self.model_features if c not in self.categorical_features]

        # Explained features split by type
        self._varied_num = [c for c in self.features_names if c in self.numerical_features]
        self._varied_cat = [c for c in self.features_names if c in self.categorical_features]

    def __repr__(self):
        return f"{type(self).__name__}(features_names={self.features_names})"

    def __call__(self, instance, amount_of_cfs):
        raise NotImplementedError

    def _predicted_class(self, instance):
        return self.model.predict(instance)[0]

    def _split_by_class(self, candidates, predictions, predicted_class, amount_of_cfs, strict):
        """Select up to `amount_of_cfs` same-class / opposite-class rows (restricted to the explained features)."""
        same = predictions == predicted_class

        predicted_class_data = candidates.loc[same, self.features_names].to_numpy()[:amount_of_cfs]
        opposite_class_data = candidates.loc[~same, self.features_names].to_numpy()[:amount_of_cfs]

        for name, data in (("same-class", predicted_class_data), ("opposite-class", opposite_class_data)):
            if len(data) == 0 or (strict and len(data) < amount_of_cfs):
                raise ValueError(
                    f"Not enough {name} samples (got {len(data)}, needed {amount_of_cfs}). "
                    "Try a larger n_samples (RANDOM), a smaller amount_of_cfs, or another cf_function."
                )
            if len(data) < amount_of_cfs:
                warnings.warn(f"Only {len(data)} {name} samples found (requested {amount_of_cfs}).")

        return predicted_class_data, opposite_class_data


class RANDOM(CounterfactualGenerator):
    """
    Random perturbations of the explained features around the instance:
    numerical features ~ Normal(instance value, training std),
    categorical features ~ uniform over the categories seen in training.
    """

    def __init__(self, model, training_data, target_ft_name, features_names,
                 categorical_features=None, n_samples=1000, random_state=None):
        super().__init__(model, training_data, target_ft_name, features_names, categorical_features)

        self.n_samples = n_samples
        self.rng = np.random.default_rng(random_state)

        self._std = {c: training_data[c].std() for c in self._varied_num}
        self._categories = {c: np.asarray(training_data[c].dropna().unique()) for c in self._varied_cat}

    def __call__(self, instance, amount_of_cfs):

        predicted_class = self._predicted_class(instance)

        # n_samples copies of the instance; only the explained features get perturbed
        samples = instance.iloc[np.zeros(self.n_samples, dtype=int)].reset_index(drop=True)

        for c in self._varied_num:
            samples[c] = self.rng.normal(
                loc=float(instance[c].iloc[0]), scale=self._std[c], size=self.n_samples
            )

        for c in self._varied_cat:
            values = self.rng.choice(self._categories[c], size=self.n_samples)
            samples[c] = pd.Series(values).astype(self.training_data[c].dtype)

        predictions = np.asarray(self.model.predict(samples[self.model_features]))

        return self._split_by_class(samples, predictions, predicted_class, amount_of_cfs, strict=True)


class KNeighbors(CounterfactualGenerator):
    """
    Nearest training points of each class. Distance (computed on the explained
    features only) is Gower-like: range-normalised absolute difference for
    numerical features, 0/1 mismatch for categorical ones.
    """

    def __init__(self, model, training_data, target_ft_name, features_names, categorical_features=None):
        super().__init__(model, training_data, target_ft_name, features_names, categorical_features)

        self._num = training_data[self._varied_num].to_numpy(dtype=float)
        self._cat = training_data[self._varied_cat].to_numpy(dtype=object)

        ranges = np.ptp(self._num, axis=0) if self._varied_num else np.empty(0)
        ranges[ranges == 0] = 1.0
        self._range = ranges

        self._train_predictions = None  # computed lazily, once

    def _distances(self, instance):
        distances = np.zeros(len(self.training_data))

        if self._varied_num:
            x = instance[self._varied_num].to_numpy(dtype=float)[0]
            distances += (np.abs(self._num - x) / self._range).sum(axis=1)

        if self._varied_cat:
            x = instance[self._varied_cat].to_numpy(dtype=object)[0]
            distances += (self._cat != x).sum(axis=1)

        return distances / len(self.features_names)

    def __call__(self, instance, amount_of_cfs):

        if self._train_predictions is None:
            self._train_predictions = np.asarray(self.model.predict(self.training_data[self.model_features]))

        predicted_class = self._predicted_class(instance)

        order = np.argsort(self._distances(instance), kind="stable")
        candidates = self.training_data.iloc[order].reset_index(drop=True)
        predictions = self._train_predictions[order]

        return self._split_by_class(candidates, predictions, predicted_class, amount_of_cfs, strict=False)


class DICE(CounterfactualGenerator):
    """
    Counterfactuals generated with dice_ml, varying only the explained features.
    Binary classification (labels 0/1) is assumed.
    """

    def __init__(self, model, training_data, target_ft_name, features_names,
                 categorical_features=None, cf_generation_dice="random"):
        super().__init__(model, training_data, target_ft_name, features_names, categorical_features)

        import dice_ml  # imported here so the rest of the library works without it

        data = dice_ml.Data(
            dataframe=training_data[self.model_features + [target_ft_name]],
            continuous_features=self.numerical_features,
            outcome_name=target_ft_name,
        )
        dice_model = dice_ml.Model(model=model, backend="sklearn")

        self.exp = dice_ml.Dice(data, dice_model, method=cf_generation_dice)

    def _generate(self, instance, amount_of_cfs, desired_class):
        from raiutils.exceptions import UserConfigValidationException  # dependency of dice_ml

        try:
            cfs = self.exp.generate_counterfactuals(
                instance,
                total_CFs=amount_of_cfs,
                desired_class=int(desired_class),
                features_to_vary=self.features_names,
            )
        except UserConfigValidationException as error:
            raise ValueError(f"DICE found no counterfactuals of class {desired_class}: {error}") from error

        return cfs.cf_examples_list[0].final_cfs_df[self.features_names].values

    def __call__(self, instance, amount_of_cfs):

        predicted_class = int(self._predicted_class(instance))
        opposite_class = 1 - predicted_class

        predicted_class_data = self._generate(instance, amount_of_cfs, predicted_class)
        opposite_class_data = self._generate(instance, amount_of_cfs, opposite_class)

        return predicted_class_data, opposite_class_data


GENERATORS = {"random": RANDOM, "knn": KNeighbors, "dice": DICE}


def make_generator(name, **kwargs):
    """Build a generator from its short name ("random", "knn" or "dice")."""
    if name.lower() not in GENERATORS:
        raise ValueError(f"Unknown cf_function '{name}'. Choose from {list(GENERATORS)} or pass a callable.")

    return GENERATORS[name.lower()](**kwargs)
