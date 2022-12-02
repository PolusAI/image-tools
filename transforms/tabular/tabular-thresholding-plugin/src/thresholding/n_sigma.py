import numpy


def find_threshold(values: numpy.ndarray, n: int = 4) -> float:
    """ Computes the threshold as `mu + n * sigma`.

    Args:
        values: 1d array of values over which tom compute the threshold.
        n: number of standard deviations to go away from the mean.

    Returns:
        The threshold value.
    """
    mu = numpy.mean(values)
    sigma = numpy.std(values)
    threshold = mu + n * sigma
    return float(threshold)
