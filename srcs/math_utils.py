import numpy as np
from typing import Tuple, List


def zscore(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma == 0.0:
        raise SystemExit("Error: standard deviation is zero; cannot standardize.")
    return (x - mu) / sigma, mu, sigma


def cost_mse(pred: np.ndarray, y: np.ndarray) -> float:
    return float((1 / len(y)) * np.sum((pred - y) ** 2))


def predict_line(theta1: float, theta0: float, x: np.ndarray) -> np.ndarray:
    return theta1 * x + theta0


def unstandardize_params(
    theta1_stand: float,
    theta0_stand: float,
    mean_x: float,
    std_x: float,
    mean_y: float,
    std_y: float,
) -> Tuple[float, float]:
    """Map standardized params (theta1_stand, theta0_stand) to raw-space (theta1_raw, theta0_raw)."""
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
    """Unstandardize standardized parameters and compute costs on raw data."""
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
