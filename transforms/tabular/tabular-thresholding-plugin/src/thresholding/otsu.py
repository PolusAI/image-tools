import numpy


def find_threshold(
        values: numpy.ndarray,
        num_bins: int = 256,
        normalize_histogram: bool = False,
) -> float:
    """ Computes the otsu threshold for the given values.

    Args:
        values: 1d array of values
        num_bins: to use for a histogram
        normalize_histogram: Whether to normalize the histogram by max
         frequency.

    Returns:
        The calculated threshold value.
    """

    # Get the image histogram
    hist, bin_edges = numpy.histogram(values, bins=num_bins)

    # Get normalized histogram if it is required
    if normalize_histogram:
        hist = numpy.divide(hist.ravel(), hist.max(initial=0))

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = numpy.cumsum(hist)
    weight2 = numpy.cumsum(hist[::-1])[::-1]

    # Get the class means mu0(t)
    mean1 = numpy.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (numpy.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance
    index_of_max_var = numpy.argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_var]
    return float(threshold)
