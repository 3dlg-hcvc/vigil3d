import numpy as np
from scipy.stats import norm


def two_proportion_z_test(size_1: int, size_2: int, score_1: float, score_2: float) -> tuple[float, float]:
    """Apply two proportion z-test to compare two proportions for statistical significance."""
    pooled_proportion = (size_1 * score_1 + size_2 * score_2) / (size_1 + size_2)
    standard_error = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (1 / size_1 + 1 / size_2))
    z = (score_1 - score_2) / standard_error
    return 2 * norm.sf(np.abs(z))
