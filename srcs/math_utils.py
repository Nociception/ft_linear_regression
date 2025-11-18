"""
Utility functions for mathematical operations used in the linear regression training.

These functions are purely mathematical and contain no I/O logic. They are imported
by the training script and isolated here for clarity and testability.
"""


from typing import Tuple, List
import numpy as np


def zscore(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Standardize an array using z-score normalization.

    The transformation is:
        x_standardized = (x - mean_x) / std_x

    Args:
        x (np.ndarray): Input array to standardize.

    Returns:
        Tuple[np.ndarray, float, float]:
            - The standardized array.
            - The mean of the input.
            - The standard deviation of the input.

    Raises:
        ValueError: If the standard deviation is zero (cannot standardize).
    """
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma == 0.0:
        raise ValueError("Error: standard deviation is zero; cannot standardize.")
    return (x - mu) / sigma, mu, sigma


def cost_mse(pred: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the Mean Squared Error (MSE) between predictions and true targets.

    MSE = (1/n) * Σ (pred_i - y_i)²

    Args:
        pred (np.ndarray): Predicted values.
        y (np.ndarray): True target values.

    Returns:
        float: The MSE value as a Python float.
    """
    return float((1 / len(y)) * np.sum((pred - y) ** 2))


def predict_line(theta1: float, theta0: float, x: np.ndarray) -> np.ndarray:
    """
    Compute linear predictions using y = theta1 * x + theta0.

    Args:
        theta1 (float): Slope of the regression line.
        theta0 (float): Intercept of the regression line.
        x (np.ndarray): Input feature array.

    Returns:
        np.ndarray: Predicted values.
    """
    return theta1 * x + theta0


def unstandardize_params(
    theta1_stand: float,
    theta0_stand: float,
    mean_x: float,
    std_x: float,
    mean_y: float,
    std_y: float,
) -> Tuple[float, float]:
    """
    Convert standardized regression parameters back to raw-space parameters.

    Given standardized parameters (a*, b*) trained on z-scored data:
        theta1_raw = a* * (std_y / std_x)
        theta0_raw = b* * std_y + mean_y - theta1_raw * mean_x

    Args:
        theta1_stand (float): Standardized slope.
        theta0_stand (float): Standardized intercept.
        mean_x (float): Mean of the raw x-values.
        std_x (float): Standard deviation of the raw x-values.
        mean_y (float): Mean of the raw y-values.
        std_y (float): Standard deviation of the raw y-values.

    Returns:
        Tuple[float, float]: (theta1_raw, theta0_raw).
    """
    theta1_raw = theta1_stand * (std_y / std_x)
    theta0_raw = theta0_stand * std_y + mean_y - theta1_raw * mean_x
    return float(theta1_raw), float(theta0_raw)


def unstandardize_results(
    theta1_stand_seq: np.ndarray,
    theta0_stand_seq: np.ndarray,
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    mean_x: float,
    std_x: float,
    mean_y: float,
    std_y: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unstandardize sequences of standardized parameters and compute raw-space costs.

    For each pair (a*, b*) in the training sequence:
        - Convert to raw parameters (theta1_raw, theta0_raw).
        - Compute MSE on the raw data.
        - Accumulate the results.

    Args:
        theta1_stand_seq (np.ndarray): Sequence of standardized slopes.
        theta0_stand_seq (np.ndarray): Sequence of standardized intercepts.
        x_raw (np.ndarray): Raw input feature values.
        y_raw (np.ndarray): Raw target values.
        mean_x (float): Mean of x_raw.
        std_x (float): Standard deviation of x_raw.
        mean_y (float): Mean of y_raw.
        std_y (float): Standard deviation of y_raw.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - Sequence of unstandardized slopes.
            - Sequence of unstandardized intercepts.
            - Sequence of raw-space MSE values.
    """
    theta1_raw_seq: List[float] = []
    theta0_raw_seq: List[float] = []
    cost_raw_seq: List[float] = []
    for theta1_stand, theta0_stand in zip(theta1_stand_seq, theta0_stand_seq):
        theta1_unstand, theta0_unstand = unstandardize_params(
            theta1_stand, theta0_stand, mean_x, std_x, mean_y, std_y
        )
        theta1_raw_seq.append(theta1_unstand)
        theta0_raw_seq.append(theta0_unstand)
        cost_raw_seq.append(
            cost_mse(predict_line(theta1_unstand, theta0_unstand, x_raw), y_raw)
        )

    return np.array(theta1_raw_seq), np.array(theta0_raw_seq), np.array(cost_raw_seq)
