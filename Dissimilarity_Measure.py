import numpy as np
from scipy.stats import ecdf, gaussian_kde
from sklearn.neighbors import KernelDensity

# np.trapz was removed in NumPy 2.0 (replaced by np.trapezoid)
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _as_1d(values):
    return np.asarray(values).ravel()


def _grid(set_1, set_2, n_points=1000):
    """Evenly spaced evaluation points covering both samples."""
    return np.linspace(min(set_1.min(), set_2.min()), max(set_1.max(), set_2.max()), n_points)


# ---------------------------------------------------------------------------
# Distribution approximations (numerical features)
#
# Contract: they receive two samples and return (func_1, func_2, points).
# They may also short-circuit by returning a scalar dissimilarity directly
# (used for the degenerate "Dirac delta" cases).
# ---------------------------------------------------------------------------

def _kde_approximation(set_1, set_2, kernel="gaussian", bandwidth=None):
    """
    PDFs estimated through Kernel Density Estimation.

    Parameters
    ----------
    set_1, set_2 : array-like
        Samples from the two distributions.
    kernel : str, default="gaussian"
        "gaussian", "epanechnikov" or "exponential".
    bandwidth : float, optional
        Required by the non-Gaussian kernels.

    Returns
    -------
    (density_1, density_2, points), or a float if both samples are constant.
    """
    set_1, set_2 = _as_1d(set_1), _as_1d(set_2)

    degenerate_1 = np.allclose(set_1, set_1[0])
    degenerate_2 = np.allclose(set_2, set_2[0])

    # Both are Dirac deltas: dissimilarity is 0 if they coincide, 1 otherwise
    if degenerate_1 and degenerate_2:
        return 0.0 if np.isclose(set_1[0], set_2[0]) else 1.0

    # Only one is a Dirac delta: a KDE cannot be built for it
    if degenerate_1 or degenerate_2:
        raise ValueError(
            "Dissimilarity is undefined when only one distribution is degenerate. "
            "Please use the ecdf."
        )

    x = _grid(set_1, set_2)

    if kernel == "gaussian":
        density_1 = gaussian_kde(set_1)(x)
        density_2 = gaussian_kde(set_2)(x)

    elif kernel in ("epanechnikov", "exponential"):
        if bandwidth is None:
            raise ValueError(f"Bandwidth must be specified when using the '{kernel}' kernel.")

        kde_1 = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(set_1.reshape(-1, 1))
        kde_2 = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(set_2.reshape(-1, 1))

        density_1 = np.exp(kde_1.score_samples(x.reshape(-1, 1)))
        density_2 = np.exp(kde_2.score_samples(x.reshape(-1, 1)))

    else:
        raise ValueError(
            f"Unsupported kernel: {kernel}. Choose from 'gaussian', 'epanechnikov', or 'exponential'."
        )

    return density_1, density_2, x


def _ecdf(set_1, set_2):
    """Empirical CDFs of the two samples, evaluated on a common grid."""
    set_1, set_2 = _as_1d(set_1), _as_1d(set_2)

    x = _grid(set_1, set_2)

    return ecdf(set_1).cdf.evaluate(x), ecdf(set_2).cdf.evaluate(x), x


class KDE:
    """Configurable (kernel, bandwidth) version of `_kde_approximation`."""

    def __init__(self, kernel="gaussian", bandwidth=None):
        self.kernel = kernel
        self.bandwidth = bandwidth

    def __call__(self, set_1, set_2):
        return _kde_approximation(set_1, set_2, self.kernel, self.bandwidth)


class ECDF:
    """Callable version of `_ecdf`."""

    def __call__(self, set_1, set_2):
        return _ecdf(set_1, set_2)


# ---------------------------------------------------------------------------
# Dissimilarities between approximated distributions (numerical features)
# ---------------------------------------------------------------------------

def _continuous_jaccard(func_1, func_2, points):
    """Dissimilarity between two distributions using a continuous Jaccard index."""
    points = np.asarray(points).ravel()
    dissimilarity = 1 - _trapz(np.minimum(func_1, func_2), points) / _trapz(np.maximum(func_1, func_2), points)

    return np.round(dissimilarity, 4)


def _wasserstein(func_1, func_2, points):
    """
    Wasserstein-1 distance. It is only correct when func_1 and func_2 are CDFs
    (i.e. when using the ecdf approximation).
    """
    return _trapz(np.abs(func_1 - func_2), np.asarray(points).ravel())


# ---------------------------------------------------------------------------
# Dissimilarity between samples (categorical features) -- ongoing
# ---------------------------------------------------------------------------

def _jaccard_distance(set1, set2):
    """
    Jaccard distance between the sets of values of two samples:
    1 - |intersection| / |union|.
    """
    set1, set2 = set(set1), set(set2)

    if not set1 and not set2:
        raise ArithmeticError("Both sets are empty")

    return 1 - len(set1 & set2) / len(set1 | set2)
