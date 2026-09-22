import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity


def _kde_dissimilarity(
    set_1,
    set_2,
    kernel="gaussian",
    bandwidth=None,
):
    """
    Compute the dissimilarity between two one-dimensional distributions
    estimated through Kernel Density Estimation.

    The dissimilarity is based on the overlap between the two densities:

        D = 1 - integral(min(p1, p2)) / integral(max(p1, p2))

    Parameters
    ----------
    set_1 : array-like
        Samples from the first distribution.
    set_2 : array-like
        Samples from the second distribution.
    kernel : str, default="gaussian"
        Kernel used for KDE. Supported kernels are:
        "gaussian", "epanechnikov", and "exponential".
    bandwidth : float, optional
        Bandwidth used by non-Gaussian kernels.

    Returns
    -------
    float
        Dissimilarity value between 0 and 1.
    """

    set_1 = np.asarray(set_1).ravel()
    set_2 = np.asarray(set_2).ravel()

    # Check if we degenerate to Dirac's delta

    if np.allclose(set_1, set_1[0]) and np.allclose(set_2, set_2[0]) and set_1[0]!=set_2[0]:
        return 1.0 # two separate distributions

    if np.allclose(set_1, set_1[0]) and np.allclose(set_2, set_2[0]):
            return 0.0 #In this case coincide

    # One of the two degenerate -> CID is undefined.
    if np.allclose(set_1, set_1[0]) or np.allclose(set_2, set_2[0]):
        if np.allclose(set_1, set_1[0]) or np.allclose(set_2, set_2[0]):
            raise ValueError("Dissimilarity is undefined when one distribution is degenerate.")


    x_min = min(set_1.min(), set_2.min())
    x_max = max(set_1.max(), set_2.max())

    x = np.linspace(x_min, x_max, 1000)

    if kernel == "gaussian":
        kde_1 = gaussian_kde(set_1)
        kde_2 = gaussian_kde(set_2)

        density_1 = kde_1(x)
        density_2 = kde_2(x)

    elif kernel in ["epanechnikov", "exponential"]:
        if bandwidth is None:
            raise ValueError(
                f"Bandwidth must be specified when using the '{kernel}' kernel."
            )

        kde_1 = KernelDensity(kernel=kernel,bandwidth=bandwidth).fit(set_1.reshape(-1, 1))

        kde_2 = KernelDensity(kernel=kernel,bandwidth=bandwidth).fit(set_2.reshape(-1, 1))

        density_1 = np.exp(kde_1.score_samples(x.reshape(-1, 1)))
        density_2 = np.exp(kde_2.score_samples(x.reshape(-1, 1)))

    else:
        raise ValueError(
            f"Unsupported kernel: {kernel}. "
            "Choose from 'gaussian', 'epanechnikov', or 'exponential'."
        )

    dissimilarity = 1 - np.trapz(np.minimum(density_1, density_2), x.ravel()) / np.trapz(np.maximum(density_1, density_2), x.ravel())

    return np.round(dissimilarity,4)


def _jaccard_distance(set1, set2):
    """
    Compute the jaccard distance between two sets which is 1 - intersection/union

    Parameters
    ----------
    set_1 : array or list
        Samples from the first distribution.
    set_2 : array or list
        Samples from the second distribution.
    """
    if len(set(set1))==0 and len(set(set2))==0:
        raise ArithmeticError("Both sets are empty")
    
    intersection = set(set1).intersection(set(set2))
    union = set(set1).union(set(set2))

    return 1 - len(intersection)/len(union)

