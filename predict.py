from __future__ import annotations
from srcs.env_check import ensure_env
ensure_env()
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple
import polars as pl
from srcs.PredictAnimation import PredictAnimation
from srcs.math_utils import predict_line
from srcs.utils import eprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict car price based on mileage.", add_help=False
    )

    parser.add_argument(
        "mileage", type=str, help="The mileage to predict the price for."
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
    parser.add_argument(
        "--dalton-type",
        choices=["protanopia", "deuteranopia", "tritanopia"],
        help="Adjust colors for red-green-blue color deficiencies."
    )

    args = parser.parse_args()

    if args.help:
        parser.print_help()
        print("\nNote: The program will now proceed with execution as requested.")

    return args


def read_model_file(path: Path) -> Tuple[str, str, float, float]:
    """Read and parse the model file."""
    parsed_data: Dict[str, str] = dict()
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
        eprint(f"Error: Could not read or parse model file '{path}'. {e}")
        sys.exit(1)


def main():
    args = parse_args()
    dalton_type: bool = args.dalton_type

    try:
        mileage_float = float(args.mileage)
        if mileage_float < 0:
            eprint("Error: mileage must be a positive number.")
            sys.exit(1)
    except ValueError:
        eprint("Error: mileage must be a valid number.")
        sys.exit(1)

    if mileage_float < 20_000:
        print("Warning: mileage is quite low, prediction may be inaccurate.")
        print(
            "Remind that the idea is to predict a price for a used car, not a new one.\n"
        )

    model_path = args.model
    if model_path.exists():
        model_read = True
        x_label, y_label, theta0, theta1 = read_model_file(model_path)
    else:
        model_read = False
        eprint(f"Error: Model file '{model_path}' not found.\n"
            "It is suggested to run the training program to create the model.txt file, "
            "with the explicit command:\n"
                "\n\tpython3 train.py\n\n"
            "Otherwise, parameters theta0 and theta1 are set to 0 by default.\n"
        )
        x_label, y_label, theta0, theta1 = "feature", "target", 0.0, 0.0
        

    predicted_target = max(0, predict_line(theta1, theta0, mileage_float))
    print(
        f"The estimated price for a mileage of {args.mileage} ({x_label}) is {predicted_target:.2f} ({y_label})."
    )

    if predicted_target > 0 and args.bonus:
        try:
            if args.model.name == "model.txt":
                data_path = Path("data.csv")
            else:
                data_stem = args.model.stem.replace("_model", "")
                data_path = args.model.parent / f"{data_stem}.csv"

            if not data_path.exists():
                eprint(f"Warning: Data file '{data_path}' not found. Cannot plot bonus graph.")
                return

            df_data = pl.read_csv(data_path)
            x_data = df_data.select(x_label).to_numpy().flatten()
            y_data = df_data.select(y_label).to_numpy().flatten()

            animator = PredictAnimation(
                mileage_float,
                predicted_target,
                x_data,
                y_data,
                theta0,
                theta1,
                x_label,
                y_label,
                dalton_type=dalton_type
            )
            animator.run()

        except pl.ColumnNotFoundError:
            eprint(f"Error: Columns not found in data file '{data_path}'.")
            sys.exit(1)
        except Exception as e:
            print(f"Error during bonus visualization: {e}")
            sys.exit(1)

    elif predicted_target == 0 and args.bonus:
        print("\nPrediction is zero, skipping bonus visualization.")

    elif model_read and not args.bonus:
        print("\nBonus mode not enabled. Type:"
              "\n\n\tpython3 predict.py <your_mileage> --bonus\n\n")


if __name__ == "__main__":
    main()
