import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import ipdb
from itertools import product

# define which meta values to plot 
META_FILTER = {
    "meta_speed_mph": [13, 19],
    "meta_stance": ['up', 'drop', 'mix'],
    "meta_direction": ['EW', 'WE'],
}

def plot_velocities_and_power():
    data = pl.read_csv("master_data_set.csv").with_columns(
        absolute_time=pl.col("absolute_time").cast(pl.Datetime),
    )

    combination = product(META_FILTER["meta_speed_mph"],
                          META_FILTER["meta_stance"],
                          META_FILTER["meta_direction"])

    for speed, stance, direction in combination:
        figure, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
        subdata = data.filter(
            pl.col("meta_direction") == direction,
            pl.col("meta_speed_mph") == speed,
            pl.col("meta_stance") == stance
        )
                
        axes[0].plot(
            subdata["time"].to_numpy(),
            subdata["Wind (m/s)"].to_numpy(),
            label=f"Wind Speed ({direction})",
            color="blue",
        )
        axes[0].plot(
            subdata["time"].to_numpy(),
            subdata["enhanced_speed"].to_numpy(),
            label=f"Speed ({direction})",
            color="orange",
        )
        axes[1].plot(
            subdata["time"].to_numpy(),
            subdata["power"].to_numpy(),
            label=f"Power ({direction})",
            color="green",
        )

        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Velocity (m/s)")
        axes[0].set_ylim(0, 12)
        axes[0].legend()
        axes[0].grid()

        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Power (W)")
        axes[1].set_ylim(0, 500)
        axes[1].legend()
        axes[1].grid()

        figure.suptitle(f"SPEED: {speed} mph - STANCE: {stance} - DIRECTION: {direction}")
        plt.tight_layout()
        plt.savefig(f"plots/{direction}_speed_{speed}_stance_{stance}.png")



if __name__ == "__main__":
    plot_velocities_and_power()
    print("Plots generated and saved in the 'plots' directory.")
