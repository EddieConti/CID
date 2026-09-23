import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from contextlib import redirect_stdout, redirect_stderr

from pandas.api.types import is_numeric_dtype

# Internal functions
from Dissimilarity_Measure import _continuous_jaccard, _kde_approximation


class CIDNumericalExplainer:
    """
    Counterfactual-based explainer for numerical features.

    The explainer is modular and allows the user to provide custom
    implementations for three main components:

    - cf_function:
        Generates counterfactual samples for the predicted and opposite
        classes. It must accept an instance and the desired number of
        counterfactuals, and return two arrays containing the generated
        samples.

    - distr_approx:
        Approximates the distributions of a feature from the two
        counterfactual sets. It must accept two sets of samples and return
        the two approximated distributions together with the points at
        which they are evaluated.

    - dist_function:
        Computes the dissimilarity between the two approximated
        distributions. It must accept the two distributions and their
        evaluation points and return a scalar value.

    The default implementations are KDE-based distribution approximation
    and continuous Jaccard dissimilarity.
    """

    def __init__(
        self,
        training_data,
        features_names,
        model,
        cf_function,
        distr_approx=_kde_approximation,
        dist_function=_continuous_jaccard,
    ):

        self.training_data = training_data
        self.model = model
        self.cf_function = cf_function
        self.dist_function = dist_function
        self.distr_approx = distr_approx
        self.features_names = features_names

        for column in features_names:
            if not is_numeric_dtype(training_data[column]):
                raise TypeError(
                    f"Feature '{column}' is not numerical. "
                    "Please convert it or pass a different set of columns."
                )

    def explain_instance(
        self,
        instance,
        amount_of_cfs=50,
    ):

        if not isinstance(instance, pd.DataFrame):
            x_df = instance.to_frame().T
        else:
            x_df = instance

        # Check on amount of CFs
        if not isinstance(amount_of_cfs, int):
            raise TypeError(
                "The amount of counterfactuals must be an integer!"
            )

        predicted_class_data, opposite_class_data = self.cf_function(x_df,amount_of_cfs)

        feature_importances = []

        for feature_idx in range(len(self.features_names)):

            func_1, func_2, points = self.distr_approx(
                opposite_class_data[:, feature_idx],
                predicted_class_data[:, feature_idx]
            )

            distance = self.dist_function(func_1,func_2,points)

            feature_importances.append(distance)

        return np.array(feature_importances)


    def visualize_feature_importances(
        self,
        feature_importances,
        barplot=True,
        output=False,
    ):
        """
        Display feature importance values.

        Parameters
        ----------
        feature_importances : array-like
            Feature importance values returned by `explain_instance`.
        barplot : bool, default=False
            Whether to display a horizontal bar plot.
        output : bool, default=True
            Whether to return the feature importance DataFrame.

        Returns
        -------
        pandas.DataFrame, optional
            Feature importance values indexed by feature name.
        """
        if not isinstance(feature_importances, np.ndarray):
            raise TypeError("feature_importances must be a NumPy array.")

        if len(feature_importances) != len(self.features_names):
            raise ValueError("feature_importances and features_names must have the same length.")

        df = pd.DataFrame({"Feature Importance": feature_importances},index=self.features_names)

        if output:
            return df


        if barplot:
            fig, ax = plt.subplots()
            ax.barh(self.features_names,feature_importances)
            ax.set_xlabel("Feature Importance")
            ax.set_ylabel("Feature")
            plt.tight_layout()
            plt.show()



    def global_explanation(
        self,
        n_instances=30,
        amount_of_cfs=50,
        variability=False
    ):
        """
        Compute a global explanation by averaging local explanations
        over a sample of training instances.

        Parameters
        ----------
        n_instances : int, default=30
            Number of instances used for the global explanation.
        amount_of_cfs : int, default=50
            Number of counterfactuals generated per instance.

        Returns
        -------
        pandas.DataFrame
            Mean feature importance for each feature.
        """
        if n_instances > len(self.training_data):
            raise ValueError(
                f"The amount of instances exceeds the amount of data. Please lower the value of n_instances"
            )
        
        sample_set = self.training_data.drop(columns=self.target_ft_name)
        sample_set = sample_set.sample(n=n_instances)


        importances = np.zeros((n_instances, len(self.features_names)))

        with tqdm(range(n_instances),desc="Explaining instances") as progress:

            with open(os.devnull, "w") as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    for i in progress:
                        importances[i, :] = self.explain_instance(
                        sample_set.iloc[i:i+1],
                        amount_of_cfs=amount_of_cfs,
                        )
        global_importances = np.mean(importances, axis=0)

        if variability:
            stds = np.std(importances, axis=0)
            return pd.DataFrame({"Feature Importance": global_importances,"Std": stds,},
                    index=self.features_names,
                    )
        else:
            return pd.DataFrame({"Feature Importance": global_importances},
                                index=self.features_names,
                                )
            