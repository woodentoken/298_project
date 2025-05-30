import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import ipdb
from itertools import product
import scipy.integrate
import scipy

# define which meta values to plot
META_FILTER = {
    "meta_speed_mph": [13, 19],
    "meta_stance": ["up", "drop", "mix"],
    "meta_direction": ["EW", "WE"],
}


def plot_velocities_and_power():
    data = pl.read_csv("master_data_set.csv").with_columns(
        absolute_time=pl.col("absolute_time").cast(pl.Datetime),
    )

    combination = product(META_FILTER["meta_speed_mph"], META_FILTER["meta_stance"], META_FILTER["meta_direction"])

    for speed, stance, direction in combination:
        figure, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))
        subdata = data.filter(
            pl.col("meta_direction") == direction, pl.col("meta_speed_mph") == speed, pl.col("meta_stance") == stance
        )

        # calculat the open loop velocity from the acceleration data alone (will be ugly!)
        time = subdata["time"].to_numpy()
        axial_acceleration = subdata["Y (m/s^2)"].to_numpy()
        ol_velocity = scipy.integrate.cumulative_trapezoid(axial_acceleration, time, initial=0)

        # velocity plots
        axes[0].plot(
            time,
            subdata["Wind (m/s)"].to_numpy(),
            label=f"Wind Speed",
            color="blue",
        )
        axes[0].plot(
            time,
            subdata["enhanced_speed"].to_numpy(),
            label=f"Garmin Velocity",
            linewidth=2,
            color="orange",
        )
        axes[0].plot(
            time,
            subdata["Speed"].to_numpy(),
            label=f"Iphone Velocity",
            linestyle="--",
            color="red",
        )
        axes[0].plot(
            time,
            ol_velocity,
            label=f"Open Loop Velocity",
            linewidth=2,
            color="magenta",
        )

        # acceleration plot
        axes[1].axhline(0, color="black", linewidth=2, linestyle="--")
        axes[1].plot(
            time,
            subdata["Y (m/s^2)"].to_numpy(),
            label=f"Y Acceleration",
            color="orange",
        )

        # power plot
        axes[2].plot(
            time,
            subdata["power"].to_numpy(),
            label=f"Power",
            color="green",
        )

        axes[0].set_ylabel("Velocity (m/s)")
        axes[0].set_ylim(0, 50)
        axes[0].legend(loc="upper right")
        axes[0].grid()

        axes[1].set_ylabel("Acceleration (m/s²)")
        axes[1].legend(loc="upper left")
        axes[1].grid()

        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Power (W)")
        axes[2].set_ylim(0, 500)
        axes[2].legend(loc="upper right")
        axes[2].grid()

        figure.suptitle(f"SPEED: {speed} mph - STANCE: {stance} - DIRECTION: {direction}")
        plt.tight_layout()
        plt.savefig(f"_plots/{direction}_speed_{speed}_stance_{stance}.png")


if __name__ == "__main__":
    plot_velocities_and_power()
    print("Plots generated and saved in the '_plots' directory.")
