import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from contextlib import redirect_stdout, redirect_stderr

from pandas.api.types import is_numeric_dtype

# Internal functions
from Dissimilarity_Measure import _continuous_jaccard,_ecdf,_kde_approximation,_wasserstein
from Initializer import _init_dice,_init_neighbors,_init_random
from Counterfactual_generators import _generate_dice,_generate_neighbors,_generate_random


class CIDNumericalExplainer:
    """
    CID explainer for numerical data.

    Given an instance, CID generates two sets of counterfactuals:
    one belonging to the predicted class and one belonging to the
    opposite class. For each feature, the distributions of the two
    counterfactual sets are compared using a KDE-based dissimilarity.

    Args
    ----------
    training_data : pandas.DataFrame
        Training dataset used by DiCE.
    model : sklearn-compatible model
        Binary classification model exposing `predict_proba`.
    target_ft_name : str, 
            Name of the target feature
    cf_method : str, default="dice"
        Counterfactual generation method. Possible alternatives neighbors, random
    distance_method : str, default="jaccard"
            Distance between two distributions. Possible alternatives wasserstein
    distribution_approx : str, default="pdf"
            Type of distribution computation from data. Possible alternatives ecdf
    cf_generation_dice : str, default="random"
        DiCE counterfactual generation strategy.
    kernel : str, default="gaussian"
        KDE kernel used to compare feature distributions.
    features_names : list of str, optional
        Names of the continuous features.
    kernel_bandwidth : float, optional
        KDE bandwidth for non-Gaussian kernels.
    """

    def __init__(
        self,
        training_data,
        model,
        target_ft_name,
        cf_method="dice",
        distance_method = "jaccard",
        distribution_approx = "pdf",
        cf_generation_dice="random",
        kernel="gaussian",
        features_names=None,
        kernel_bandwidth=None,
    ):
    
        self.training_data = training_data
        self.model = model
        self.target_ft_name = target_ft_name
        self.kernel = kernel
        self.kernel_bandwidth = kernel_bandwidth
        self.features_names = features_names

        # Checking if features are numerical
        for column in features_names:
            if not is_numeric_dtype(training_data[column]):
                raise TypeError(f"Feature '{column}' is not numerical. "
            "Please convert it or pass a different set of columns."
             )


        cf_initializers = {
            "dice": _init_dice,
            "neighbors": _init_neighbors,
            "random": _init_random,
        }

        available_methods = [key for key in cf_initializers.keys()]

        if cf_method not in available_methods:
            raise ValueError("The method is not available. Please use one among: ",available_methods)

        self.cf_initializer = cf_initializers[cf_method]

        self.cf_explainer = self.cf_initializer(model,
                                                training_data,
                                                features_names,
                                                target_ft_name,
                                                cf_generation_dice)
        self.cf_generators = {
            "dice": _generate_dice,
            "neighbors": _generate_neighbors,
            "random": _generate_random
        }

        self.cf_generator = self.cf_generators[cf_method]

        distances = {
                    "jaccard": _continuous_jaccard,
                    "wasserstein": _wasserstein,
                }
        self.distance = distances[distance_method]

        approximations = {
                    "pdf": _kde_approximation,
                    "ecdf": _ecdf,
                }
        self.approx = approximations[distribution_approx]
        



    def explain_instance(
        self,
        instance,
        amount_of_cfs=50,
    ):
        """
        Explain a single instance.

        Parameters
        ----------
        instance : pandas.dataframe
            Instance to explain.
        amount_of_cfs : int, default=50
            Number of counterfactuals generated for each class.

        Returns
        -------
        list
            Feature importance values in the same order as
            `features_names`.
        """

        if not isinstance(instance, pd.DataFrame):
            x_df = instance.to_frame().T
        else:
            x_df = instance

        # Check on amount of CFs
        if not type(amount_of_cfs) == int:
            raise TypeError("The amount of counterfactuals must be an intenger!")

        probabilities = self.model.predict_proba(x_df).squeeze()
        predicted_class = np.argmax(probabilities)

        predicted_class_data, opposite_class_data = self.cf_generator(
            self.model,
            self.cf_explainer,
            x_df,
            predicted_class,
            amount_of_cfs,
            self.training_data,
            self.features_names,
        )

        feature_importances = []


        for feature_idx in range(len(self.features_names)):
            
            func_1,funct_2,points = self.approx(opposite_class_data[:, feature_idx],
                                                predicted_class_data[:, feature_idx],
                                                kernel=self.kernel,
                                                bandwidth=self.kernel_bandwidth)


            distance = self.distance(func_1,funct_2,points)

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
            print("ciao")
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
            