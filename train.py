from __future__ import annotations
from srcs.env_check import ensure_env

ensure_env()
from typing import Dict, Tuple, List
import argparse
from pathlib import Path
import numpy as np
import polars as pl
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from srcs.math_utils import zscore, cost_mse, predict_line, unstandardize_results
from srcs.TrainAnimation import TrainAnimation


def read_csv_strict(path: Path) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """CSV Requirements:
    - Exactly two columns
    - All values numeric
    """
    if not path.exists():
        raise SystemExit(f"Error: CSV file not found: {path}")
    try:
        df = pl.read_csv(path)
    except Exception as e:
        raise SystemExit(f"Error: failed to read CSV '{path}': {e}")

    if len(df.columns) != 2:
        raise SystemExit("Error: CSV must contain exactly two columns.")

    col_names = df.columns
    x_col_name, y_col_name = col_names[0], col_names[1]

    n_rows = df.height
    threshold = 100_000
    data_type, dtype_name = (
        (pl.Float64, "Float64") if n_rows < threshold else (pl.Float32, "Float32")
    )
    print(f"Data conversion strategy: {dtype_name} (based on {n_rows} rows)")

    try:
        feature = df[x_col_name].cast(data_type).to_numpy()
        target = df[y_col_name].cast(data_type).to_numpy()
    except Exception:
        raise SystemExit(
            f"Error: both '{x_col_name}' and '{y_col_name}' columns must be numeric."
        )

    if feature.size == 0:
        raise SystemExit("Error: CSV contains no rows.")

    return feature, target, x_col_name, y_col_name


def grad_desc_stand(
    x_stand: np.ndarray,
    y_stand: np.ndarray,
    alpha: float = 0.01,
    n_iter: int = 1000,
    threshold: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Returns the sequences of (a_n, b_n) at each iteration and the standardized cost list."""
    m = float(len(x_stand))

    theta1_stand, theta0_stand = 0.0, 0.0

    theta1_stand_seq, theta0_stand_seq, cost_stand_seq = [], [], []

    count = 0
    while count < n_iter and (
        len(cost_stand_seq) < 2
        or abs(cost_stand_seq[-1] - cost_stand_seq[-2]) > threshold
    ):
        count += 1

        estimate_stand = predict_line(theta1_stand, theta0_stand, x_stand)
        cost_stand = cost_mse(estimate_stand, y_stand)
        cost_stand_seq.append(cost_stand)

        tmp_theta0 = alpha * np.sum(estimate_stand - y_stand) / m
        theta0_stand -= tmp_theta0
        theta0_stand_seq.append(theta0_stand)

        tmp_theta1 = alpha * np.sum((estimate_stand - y_stand) * x_stand) / m
        theta1_stand -= tmp_theta1
        theta1_stand_seq.append(theta1_stand)

    return np.array(theta1_stand_seq), np.array(theta0_stand_seq), cost_stand_seq


def metrics_raw(x: np.ndarray, y: np.ndarray, a: float, b: float) -> Dict[str, float]:
    """
    Calculates a set of common regression metrics to evaluate a linear model's performance.

    This function computes the Mean Squared Error (MSE), Mean Absolute Error (MAE),
    Root Mean Squared Error (RMSE), and the R-squared (R²) score for a simple linear
    regression model. It compares the predictions of the line defined by the slope `a`
    and intercept `b` against the true target values `y`.

    Args:
        x (np.ndarray): The input feature array used for making predictions.
        y (np.ndarray): The true target values against which the predictions are compared.
        a (float): The calculated slope (or weight) of the regression line.
        b (float): The calculated y-intercept (or bias) of the regression line.

    Returns:
        Dict[str, float]: A dictionary containing the following metrics:
            - "mse" (Mean Squared Error): The average of the squared differences between
              the predicted and actual values. It penalizes larger errors more heavily,
              making it sensitive to outliers.
            - "mae" (Mean Absolute Error): The average of the absolute differences between
              the predicted and actual values. It's more robust to outliers than MSE,
              as it doesn't square the errors.
            - "rmse" (Root Mean Squared Error): The square root of the MSE. It's a key metric
              because its units are the same as the target variable `y`, making it
              easier to interpret and compare with the data's scale.
            - "r2" (R-squared): The coefficient of determination. This value represents the
              proportion of the variance in the dependent variable that is predictable
              from the independent variable. A value close to 1.0 indicates that the model
              explains a large portion of the variance, while a value near 0.0 suggests
              the model explains very little.
    """
    pred = predict_line(a, b, x)
    mse = cost_mse(pred, y)
    mae = float(np.mean(np.abs(pred - y)))
    rmse = float(np.sqrt(mse))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - pred) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


def model_out_path(csv_path: Path) -> Path:
    if csv_path.name.lower() == "data.csv":
        return Path("model.txt")
    stem = csv_path.stem
    return Path(f"{stem}_model.txt")


def save_model(
    x_label: str,
    y_label: str,
    path: Path,
    theta0: float,
    theta1: float,
    extra: Dict[str, float],
) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"x_label={x_label}\n")
            f.write(f"y_label={y_label}\n")
            f.write(f"theta0={theta0}\n")
            f.write(f"theta1={theta1}\n")
            for k, v in extra.items():
                f.write(f"{k}={v}\n")
    except OSError as e:
        raise SystemExit(f"Error: failed to write model file '{path}': {e}")


def parsing_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a linear regression model on a CSV file."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("data.csv"),
        help="Path to the training data CSV file (default: data.csv).",
    )
    parser.add_argument(
        "--bonus",
        action="store_true",
        help="Enable bonus mode to visualize the training process.",
    )
    parser.add_argument(
        "--dalton-type",
        choices=["protanopia", "deuteranopia", "tritanopia"],
        help="Adjust colors for red-green-blue color deficiencies.",
    )

    try:
        args = parser.parse_args()
        if not args.file.is_absolute():
            args.file = Path.cwd() / args.file
        if not args.file.suffix.lower() == ".csv":
            raise SystemExit("Error: file must be a CSV file (with .csv extension).")
        if not args.file.exists():
            raise SystemExit(f"Error: file does not exist: {args.file}")
        if not args.file.is_file():
            raise SystemExit(f"Error: path is not a file: {args.file}")
        if not args.file.stat().st_size > 0:
            raise SystemExit(f"Error: file is empty: {args.file}")
    except Exception as e:
        raise SystemExit(f"Error: failed to parse command line arguments: {e}")

    return args


def main() -> None:
    args = parsing_cli_args()

    csv_path: Path = args.file
    bonus: bool = args.bonus
    dalton_type: bool = args.dalton_type

    print(f"Training file: {csv_path}")
    print(f"Bonus mode: {'ON' if bonus else 'OFF'}")

    x_raw, y_raw, x_label, y_label = read_csv_strict(csv_path)
    m = x_raw.size
    print(f"Loaded {m} rows.")

    x_stand, mean_x, std_x = zscore(x_raw)
    y_stand, mean_y, std_y = zscore(y_raw)
    print(
        f"Standardization: mean_x={mean_x:.6f}, sigma_x={std_x:.6f} | mean_y={mean_y:.6f}, sigma_y={std_y:.6f}"
    )

    alpha = 0.01
    n_iter = 600
    threshold = 1e-6
    print(f"Hyperparameters: learning_rate={alpha}, iterations={n_iter}")

    theta1_stand_seq, theta0_stand_seq, cost_stand_seq = grad_desc_stand(
        x_stand, y_stand, alpha=alpha, n_iter=n_iter, threshold=threshold
    )

    theta1_raw_seq, theta0_raw_seq, cost_raw_seq = unstandardize_results(
        theta1_stand_seq, theta0_stand_seq, x_raw, y_raw, mean_x, std_x, mean_y, std_y
    )

    theta1_raw = float(theta1_raw_seq[-1])
    theta0_raw = float(theta0_raw_seq[-1])

    print("Training completed.")
    print(f"Final (raw) parameters: theta0={theta0_raw:.10f}, theta1={theta1_raw:.10f}")
    print(f"Final raw cost (MSE): {cost_raw_seq[-1]:.10f}")

    extra = metrics_raw(x_raw, y_raw, theta1_raw, theta0_raw)
    out_path = model_out_path(csv_path)
    save_model(x_label, y_label, out_path, theta0_raw, theta1_raw, extra)
    print(f"Model saved to: {out_path.resolve()}")

    print("Model performance metrics available in the model file: model.txt")

    print("\nYou now can use the model for predictions.")
    print("\n\t./run.sh predict <mileage>\n")
    print("\n(Notes:")
    print("- <mileage> should be a numeric value.")
    print("- this file also accepts the --bonus flag for visualization.)\n")

    if bonus:
        print("Showing synchronized bonus animations… (will play once)")
        animation = TrainAnimation(
            x_raw,
            y_raw,
            x_stand,
            y_stand,
            theta1_stand_seq,
            theta0_stand_seq,
            cost_stand_seq,
            theta1_raw_seq,
            theta0_raw_seq,
            cost_raw_seq,
            x_label,
            y_label,
            dalton_type=dalton_type,
        )
        animation.run()
    else:
        print("Bonus mode is off. To enable, run with the --bonus flag:")
        print("\n\t./run.sh train --bonus\n")


if __name__ == "__main__":
    main()
