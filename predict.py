from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple

import polars as pl

from srcs.PredictionAnimator import PredictionAnimator

try:
    from srcs.math_utils import predict_line
except ImportError:
    print(
        "Error: 'srcs/math_utils.py' not found. Please ensure it's in the same directory."
    )
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict car price based on mileage.", add_help=False
    )

    parser.add_argument(
        "mileage", type=float, help="The mileage to predict the price for."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("model.txt"),
        help="Path to the model file (default: model.txt).",
    )
    parser.add_argument(
        "--bonus",
        action="store_true",
        help="Enable bonus mode to visualize the prediction on a graph.",
    )
    parser.add_argument(
        "--help",
        "-h",
        action="store_true",
        help="Show this help message and continue program execution.",
    )

    args = parser.parse_args()

    if args.help:
        parser.print_help()
        print("\nNote: The program will now proceed with execution as requested.")

    return args


def read_model_file(path: Path) -> Tuple[str, str, float, float]:
    """Read and parse the model file."""
    parsed_data: Dict[str, str] = {}
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("=")
                if len(parts) == 2:
                    parsed_data[parts[0].strip()] = parts[1].strip()

        x_label = parsed_data.get("x_label")
        y_label = parsed_data.get("y_label")
        theta0_str = parsed_data.get("theta0")
        theta1_str = parsed_data.get("theta1")

        if theta0_str is None or theta1_str is None:
            raise ValueError("Missing 'theta0' or 'theta1' in model file.")

        return x_label, y_label, float(theta0_str), float(theta1_str)

    except (IOError, ValueError, IndexError) as e:
        print(
            f"Error: Could not read or parse model file '{path}'. {e}", file=sys.stderr
        )
        sys.exit(1)


def main():
    args = parse_args()

    if args.mileage < 0:
        print("Error: mileage must be a positive number.", file=sys.stderr)
        sys.exit(1)

    model_path = args.model
    if not model_path.exists():
        print(f"Error: Model file '{model_path}' not found.", file=sys.stderr)
        print(
            "It is suggested to run the training program to create the model.txt file, "
            "with the explicit command: python3 train.py",
            file=sys.stderr,
        )
        sys.exit(1)

    x_label, y_label, theta0, theta1 = read_model_file(model_path)

    predicted_price = predict_line(theta1, theta0, args.mileage)
    print(
        f"The estimated price for a mileage of {int(args.mileage)} ({x_label}) is {predicted_price:.2f} ({y_label})."
    )

    if args.bonus:
        try:
            if args.model.name == "model.txt":
                data_path = Path("data.csv")
            else:
                data_stem = args.model.stem.replace("_model", "")
                data_path = args.model.parent / f"{data_stem}.csv"

            if not data_path.exists():
                print(
                    f"Warning: Data file '{data_path}' not found. Cannot plot bonus graph.",
                    file=sys.stderr,
                )
                return

            df_data = pl.read_csv(data_path)
            x_label, y_label = df_data.columns[0], df_data.columns[1]
            x_data = df_data.select(x_label).to_numpy().flatten()
            y_data = df_data.select(y_label).to_numpy().flatten()

            animator = PredictionAnimator(
                args.mileage,
                predicted_price,
                x_data,
                y_data,
                theta0,
                theta1,
                x_label,
                y_label,
            )
            animator.run()

        except pl.ColumnNotFoundError:
            print(
                f"Error: Columns not found in data file '{data_path}'.", file=sys.stderr
            )
            sys.exit(1)
        except Exception as e:
            print(f"Error during bonus visualization: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
