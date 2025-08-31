import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from srcs.math_utils import predict_line


class PredictAnimation:
    def __init__(
        self,
        mileage: float,
        predicted_price: float,
        x_data: np.ndarray,
        y_data: np.ndarray,
        theta0: float,
        theta1: float,
        x_label: str,
        y_label: str,
    ):
        self.mileage = mileage
        self.predicted_price = predicted_price
        self.x_data = x_data
        self.y_data = y_data
        self.theta0 = theta0
        self.theta1 = theta1
        self.x_label = x_label
        self.y_label = y_label
        self.n_frames = 100

        self.fig, self.ax = plt.subplots(figsize=(10, 6))

        self.ax.scatter(
            self.x_data, self.y_data, s=16, alpha=0.6, label="Training Data"
        )
        x_line = np.array([np.min(self.x_data), np.max(self.x_data)])
        y_line = predict_line(self.theta1, self.theta0, x_line)
        self.ax.plot(x_line, y_line, color="red", label="Regression Line")

        (self.line_vert,) = self.ax.plot([], [], "--", color="green", lw=1)
        (self.line_horiz,) = self.ax.plot([], [], "--", color="green", lw=1)
        (self.pred_point,) = self.ax.plot(
            [], [], "o", color="green", markersize=10, zorder=5
        )

        self.mileage_text = self.ax.text(
            self.mileage,
            self.ax.get_ylim()[0]-500,
            f"{self.mileage:.0f}",
            ha="center",
            va="bottom",
            color="blue",
            fontsize=10,
            backgroundcolor="white",
        )

        self.text_pred = self.ax.text(
            0, self.predicted_price, "", ha="right", va="center", color="green"
        )

        self.ax.set_xlabel(f"{self.x_label.capitalize()}")
        self.ax.set_ylabel(f"{self.y_label.capitalize()}")

        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        self.ax.set_xlim(
            min(x_min, self.mileage - 0.1 * (x_max - x_min)),
            max(x_max, self.mileage + 0.1 * (x_max - x_min)),
        )
        self.ax.set_ylim(
            min(y_min, self.predicted_price - 0.1 * (y_max - y_min)),
            max(y_max, self.predicted_price + 0.1 * (y_max - y_min)),
        )

        self.ax.legend()
        self.ax.grid(True)
        self.ax.ticklabel_format(style="plain", useOffset=False)

    def _update(self, frame: int) -> Tuple:
        """Update function for the animation."""
        if frame < self.n_frames // 2:
            y_end = (frame / (self.n_frames // 2)) * self.predicted_price
            self.line_vert.set_data([self.mileage, self.mileage], [0, y_end])
            self.pred_point.set_data([self.mileage], [y_end])
            return (self.line_vert, self.pred_point, self.text_pred)
        else:
            y_coord = self.predicted_price
            x_end = self.mileage * (
                (self.n_frames - 1 - frame) / (self.n_frames // 2 - 1)
            )
            self.line_horiz.set_data([self.mileage, x_end], [y_coord, y_coord])

            self.line_vert.set_data(
                [self.mileage, self.mileage], [0, self.predicted_price]
            )
            self.pred_point.set_data([self.mileage], [self.predicted_price])

            if frame == self.n_frames - 1:
                self.text_pred.set_text(f"{self.predicted_price:.2f}")

            return (
                self.line_horiz,
                self.line_vert,
                self.pred_point,
                self.text_pred,
            )

    def run(self):
        anim = FuncAnimation(
            self.fig,
            self._update,
            frames=self.n_frames,
            interval=20,
            blit=False,
            repeat=False,
        )
        plt.show()