"""Feature segmentation evaluation package."""
## Source: https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EPSILON = 1e-10


def _error(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Simple error."""
    return actual - predicted


def _percentage_error(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Percentage error.

    Note: result is NOT multiplied by 100.
    """
    return _error(actual, predicted) / (actual + EPSILON)


def _geometric_mean(
    a: np.ndarray,
    axis: Optional[int] = 0,
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """Geometric mean."""
    if not isinstance(a, np.ndarray):  # if not an ndarray object attempt to convert it
        log_a = np.log(np.array(a, dtype=dtype))
    elif dtype:  # Must change the default dtype allowing array type
        if isinstance(a, np.ma.MaskedArray):
            log_a = np.log(np.ma.asarray(a, dtype=dtype))
        else:
            log_a = np.log(np.asarray(a, dtype=dtype))
    else:
        log_a = np.log(a)
    return np.exp(log_a.mean(axis=axis))


def mse(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Mean Squared Error."""
    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Root Mean Squared Error."""
    return np.sqrt(mse(actual, predicted))


def nrmse(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Normalized Root Mean Squared Error."""
    return rmse(actual, predicted) / (actual.max() - actual.min())


def me(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Mean Error."""
    return np.mean(_error(actual, predicted))


def mae(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Mean Absolute Error."""
    return np.mean(np.abs(_error(actual, predicted)))


mad = mae  # Mean Absolute Deviation (it is the same as MAE)


def gmae(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Geometric Mean Absolute Error."""
    return _geometric_mean(np.abs(_error(actual, predicted)))


def mdae(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Median Absolute Error."""
    return np.median(np.abs(_error(actual, predicted)))


def mpe(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Mean Percentage Error."""
    return np.mean(_percentage_error(actual, predicted))


def mape(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Mean Absolute Percentage Error.

    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0
    Note: result is NOT multiplied by 100.
    """
    return np.mean(np.abs(_percentage_error(actual, predicted)))


def mdape(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Median Absolute Percentage Error.

    Note: result is NOT multiplied by 100.
    """
    return np.median(np.abs(_percentage_error(actual, predicted)))


def smape(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Symmetric Mean Absolute Percentage Error.

    Note: result is NOT multiplied by 100.
    """
    return np.mean(
        2.0
        * np.abs(actual - predicted)
        / ((np.abs(actual) + np.abs(predicted)) + EPSILON),
    )


def smdape(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Symmetric Median Absolute Percentage Error.

    Note: result is NOT multiplied by 100.
    """
    return np.median(
        2.0
        * np.abs(actual - predicted)
        / ((np.abs(actual) + np.abs(predicted)) + EPSILON),
    )


def maape(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Mean Arctangent Absolute Percentage Error.

    Note: result is NOT multiplied by 100.
    """
    return np.mean(np.arctan(np.abs((actual - predicted) / (actual + EPSILON))))


def std_ae(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Normalized Absolute Error."""
    __mae = mae(actual, predicted)
    return np.sqrt(
        np.sum(np.square(_error(actual, predicted) - __mae)) / (len(actual) - 1),
    )


def std_ape(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Normalized Absolute Percentage Error."""
    __mape = mape(actual, predicted)
    return np.sqrt(
        np.sum(np.square(_percentage_error(actual, predicted) - __mape))
        / (len(actual) - 1),
    )


def rmspe(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Root Mean Squared Percentage Error.

    Note: result is NOT multiplied by 100.
    """
    return np.sqrt(np.mean(np.square(_percentage_error(actual, predicted))))


def rmdspe(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Root Median Squared Percentage Error.

    Note: result is NOT multiplied by 100.
    """
    return np.sqrt(np.median(np.square(_percentage_error(actual, predicted))))


def inrse(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Integral Normalized Root Squared Error."""
    return np.sqrt(
        np.sum(np.square(_error(actual, predicted)))
        / np.sum(np.square(actual - np.mean(actual))),
    )


def rrse(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Root Relative Squared Error."""
    return np.sqrt(
        np.sum(np.square(actual - predicted))
        / np.sum(np.square(actual - np.mean(actual))),
    )


def rae(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Relative Absolute Error (aka Approximation Error)."""
    return np.sum(np.abs(actual - predicted)) / (
        np.sum(np.abs(actual - np.mean(actual))) + EPSILON
    )


def mda(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Mean Directional Accuracy."""
    return np.mean(
        (
            np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])
        ).astype(int),
    )


METRICS = {
    "mse": mse,
    "rmse": rmse,
    "nrmse": nrmse,
    "me": me,
    "mae": mae,
    "mad": mad,
    "gmae": gmae,
    "mdae": mdae,
    "mpe": mpe,
    "mape": mape,
    "mdape": mdape,
    "smape": smape,
    "smdape": smdape,
    "maape": maape,
    "std_ae": std_ae,
    "std_ape": std_ape,
    "rmspe": rmspe,
    "rmdspe": rmdspe,
    "inrse": inrse,
    "rrse": rrse,
    "rae": rae,
    "mda": mda,
}


def evaluate(actual: np.ndarray, predicted: np.ndarray, metrics: dict) -> dict:
    """Compute error metrics."""
    results = {}
    for name in metrics:
        try:
            results[name] = METRICS[name](actual, predicted)
        except ValueError as err:
            results[name] = np.nan
            logger.info(f"Unable to compute metric {name}: {err}")
    return results


def evaluate_all(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute all metrics."""
    return evaluate(actual, predicted, metrics=set(METRICS.keys()))  # type: ignore
