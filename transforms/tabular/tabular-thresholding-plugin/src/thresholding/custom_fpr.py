import statistics

import numpy


def find_threshold(
        values: numpy.ndarray,
        false_positive_rate: float,
) -> float:
    """ Computes a threshold value using a user-specified false positive rate.

    We assume that the `negative_values` follow a single gaussian distribution.
     We estimate the mean and standard deviation of this distribution and
     compute a threshold such that the area to the right of the threshold is
     equal to the given `false_positive_rate`.

    Args:
        values: drawn from a single gaussian distribution.
        false_positive_rate: A user-defined tuning parameter.

    Returns:
        The computed threshold value.
    """
    if not (0 < false_positive_rate < 1):
        raise ValueError(f'`false_positive_rate` mut be in the range (0, 1). Got {false_positive_rate:.2e} instead.')

    mu = float(numpy.mean(values))
    sigma = float(numpy.std(values))

    distribution = statistics.NormalDist(mu, sigma)
    threshold = distribution.inv_cdf(1 - false_positive_rate)

    return threshold
