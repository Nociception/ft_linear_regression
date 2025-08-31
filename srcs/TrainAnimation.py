import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from typing import List, Tuple
from .math_utils import predict_line, cost_mse


class TrainAnimation:
    def __init__(
        self,
        x_raw: np.ndarray,
        y_raw: np.ndarray,
        x_stand: np.ndarray,
        y_stand: np.ndarray,
        theta1_stand_seq: np.ndarray,
        theta0_stand_seq: np.ndarray,
        cost_stand_seq: List[float],
        theta1_raw_seq: np.ndarray,
        theta0_raw_seq: np.ndarray,
        cost_raw_seq: List[float],
        x_label: str,
        y_label: str,
    ):
        self.x_raw = x_raw
        self.y_raw = y_raw
        self.x_stand = x_stand
        self.y_stand = y_stand
        self.theta1_stand_seq = theta1_stand_seq
        self.theta0_stand_seq = theta0_stand_seq
        self.cost_stand_seq = cost_stand_seq
        self.theta1_raw_seq = theta1_raw_seq
        self.theta0_raw_seq = theta0_raw_seq
        self.cost_raw_seq = cost_raw_seq
        self.x_label = x_label
        self.y_label = y_label
        self.n_frames = len(self.theta1_stand_seq)

        self.blue_color = "blue"
        self.orange_color = "orange"

        self.fig = plt.figure(figsize=(14, 6))

        ax_main_rect = [0.08, 0.1, 0.5, 0.8]
        ax_raw_rect = [0.6, 0.55, 0.35, 0.4]
        ax_norm_rect = [0.6, 0.05, 0.35, 0.4]

        self.ax_main = self.fig.add_axes(ax_main_rect)
        self.ax_main.set_xlabel(self.x_label, color=self.blue_color)
        self.ax_main.set_ylabel(self.y_label, color=self.blue_color)
        self.ax_main.tick_params(axis="x", colors=self.blue_color)
        self.ax_main.tick_params(axis="y", colors=self.blue_color)

        self.ax_main.scatter(
            self.x_raw, self.y_raw, s=16, alpha=0.9, label="data", color=self.blue_color
        )
        x_line = np.array([np.min(self.x_raw), np.max(self.x_raw)])
        (self.line_reg,) = self.ax_main.plot(
            x_line,
            predict_line(self.theta1_raw_seq[0], self.theta0_raw_seq[0], x_line),
            lw=2,
            label="model",
            color="red",
        )
        self.ax_main.legend(loc="best")

        self.ax_overlay = self.fig.add_axes(ax_main_rect, frameon=False)
        self.ax_overlay.patch.set_alpha(0.0)
        self.ax_overlay.xaxis.set_label_position("top")
        self.ax_overlay.xaxis.tick_top()
        self.ax_overlay.yaxis.set_label_position("right")
        self.ax_overlay.yaxis.tick_right()

        self.ax_overlay.ticklabel_format(style="plain", axis="y", useOffset=False)

        self.ax_overlay.set_xlabel("Iteration", color=self.orange_color)
        self.ax_overlay.set_ylabel("Cost (MSE) - RAW data", color=self.orange_color)

        self.ax_overlay.tick_params(axis="x", colors=self.orange_color)
        self.ax_overlay.tick_params(axis="y", colors=self.orange_color)
        self.ax_overlay.set_xlim(1, self.n_frames)
        cmin = float(np.min(self.cost_raw_seq))
        cmax = float(np.max(self.cost_raw_seq))
        cmarg = 0.05 * (cmax - cmin if cmax > cmin else max(cmax, 1.0))
        self.ax_overlay.set_ylim(cmin - cmarg, cmax + cmarg)
        self.iters = np.arange(1, self.n_frames + 1)
        self.cost_scatter = self.ax_overlay.scatter(
            [], [], s=4, color=self.orange_color
        )

        self.ax3d_raw = self.fig.add_axes(ax_raw_rect, projection="3d")
        self.ax3d_raw.set_title("Cost surface - RAW data")
        self.ax3d_raw.set_xlabel("theta1")
        self.ax3d_raw.set_ylabel("theta0")
        self.ax3d_raw.set_zlabel("Cost (MSE)")
        A_r, B_r, C_r = self.build_cost_surface(
            self.x_raw,
            self.y_raw,
            self.theta1_raw_seq,
            self.theta0_raw_seq,
            use_gd_path=True,
        )
        self.ax3d_raw.plot_surface(
            A_r, B_r, C_r, cmap="viridis", alpha=0.6, linewidth=0, antialiased=True
        )
        self.traj_r_x: List[float] = []
        self.traj_r_y: List[float] = []
        self.traj_r_z: List[float] = []
        self.traj_r_scatter = self.ax3d_raw.scatter(
            [], [], [], s=12, color=self.orange_color
        )

        self.ax3d_n = self.fig.add_axes(ax_norm_rect, projection="3d")
        self.ax3d_n.set_title("Cost surface - STANDARDIZED data")
        self.ax3d_n.set_xlabel("theta1")
        self.ax3d_n.set_ylabel("theta0")
        self.ax3d_n.set_zlabel("Cost (MSE)")
        A_n, B_n, C_n = self.build_cost_surface(
            self.x_stand,
            self.y_stand,
            self.theta1_stand_seq,
            self.theta0_stand_seq,
            use_gd_path=False,
        )
        self.ax3d_n.plot_surface(
            A_n, B_n, C_n, cmap="viridis", alpha=0.6, linewidth=0, antialiased=True
        )
        self.traj_n_x: List[float] = []
        self.traj_n_y: List[float] = []
        self.traj_n_z: List[float] = []
        self.traj_n_scatter = self.ax3d_n.scatter(
            [], [], [], s=12, color=self.orange_color
        )

    def build_cost_surface(
        self,
        X: np.ndarray,
        y: np.ndarray,
        a_path: np.ndarray,
        b_path: np.ndarray,
        grid_n: int = 50,
        use_gd_path: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build a (theta0, theta1, cost) surface covering the gradient descent path."""
        if use_gd_path:
            a_min, a_max = float(np.min(a_path)), float(np.max(a_path))
            b_min, b_max = float(np.min(b_path)), float(np.max(b_path))
            da = 0.2 * max(
                1e-12, (a_max - a_min) if a_max > a_min else max(abs(a_max), 1.0)
            )
            db = 0.2 * max(
                1e-12, (b_max - b_min) if b_max > b_min else max(abs(b_max), 1.0)
            )
            a_vals = np.linspace(a_min - da, a_max + da, grid_n)
            b_vals = np.linspace(b_min - db, b_max + db, grid_n)
        else:
            a_vals = np.linspace(-1.5, 1.5, grid_n)
            b_vals = np.linspace(-1.5, 1.5, grid_n)

        A, B = np.meshgrid(a_vals, b_vals)
        pred = (A[None, :, :] * X[:, None, None]) + B[None, :, :]
        err = pred - y[:, None, None]
        C = np.mean(err**2, axis=0)
        return A, B, C

    def update(self, frame: int):
        a_r = self.theta1_raw_seq[frame]
        b_r = self.theta0_raw_seq[frame]
        y_line = predict_line(
            a_r, b_r, np.array([np.min(self.x_raw), np.max(self.x_raw)])
        )
        self.line_reg.set_data([np.min(self.x_raw), np.max(self.x_raw)], y_line)

        xs = self.iters[: frame + 1]
        ys = np.array(self.cost_raw_seq[: frame + 1])
        self.cost_scatter.set_offsets(np.c_[xs, ys])

        t1 = float(self.theta1_raw_seq[frame])
        t0 = float(self.theta0_raw_seq[frame])
        cz = float(self.cost_raw_seq[frame])
        self.traj_r_x.append(t1)
        self.traj_r_y.append(t0)
        self.traj_r_z.append(cz)
        self.traj_r_scatter._offsets3d = (self.traj_r_x, self.traj_r_y, self.traj_r_z)
        if frame > 0:
            self.ax3d_raw.plot(
                [self.traj_r_x[frame - 1], t1],
                [self.traj_r_y[frame - 1], t0],
                [self.traj_r_z[frame - 1], cz],
                linestyle="--",
                linewidth=1,
                color=self.orange_color,
            )

        tn1 = float(self.theta1_stand_seq[frame])
        tn0 = float(self.theta0_stand_seq[frame])
        cz_n = float(cost_mse(predict_line(tn1, tn0, self.x_stand), self.y_stand))
        self.traj_n_x.append(tn1)
        self.traj_n_y.append(tn0)
        self.traj_n_z.append(cz_n)
        self.traj_n_scatter._offsets3d = (self.traj_n_x, self.traj_n_y, self.traj_n_z)
        if frame > 0:
            self.ax3d_n.plot(
                [self.traj_n_x[-2], self.traj_n_x[-1]],
                [self.traj_n_y[-2], self.traj_n_y[-1]],
                [self.traj_n_z[-2], self.traj_n_z[-1]],
                linestyle="--",
                linewidth=1,
                color=self.orange_color,
            )

        return (
            self.line_reg,
            self.cost_scatter,
            self.traj_r_scatter,
            self.traj_n_scatter,
        )

    def run(self):
        anim = FuncAnimation(
            fig=self.fig,
            func=self.update,
            frames=self.n_frames,
            interval=1,
            repeat=False,
            blit=False,
        )
        plt.show()
