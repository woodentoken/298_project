import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import ipdb


def main():
    data = pl.read_csv("master_data_set.csv").with_columns(
        time=pl.col("time").cast(pl.Datetime),
    )

    meta_filter = {
        "meta_speed_mph": 19,
        "meta_stance": "drop",
    }

    data = data.filter(
        pl.col("meta_speed_mph") == meta_filter["meta_speed_mph"],
        pl.col("meta_stance") == meta_filter["meta_stance"],
    )

    for direction in ["EW", "WE"]:
        figure, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
        subdata = data.filter(pl.col("meta_direction") == direction)
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

        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Velocity (m/s)")
        axes[0].legend()

        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Power (W)")

        figure.suptitle(f"SPEED: {meta_filter['meta_speed_mph']} mph - STANCE: {meta_filter['meta_stance']}")
        plt.tight_layout()
        plt.savefig(f"plots/{direction}_speed_{meta_filter['meta_speed_mph']}_stance_{meta_filter['meta_stance']}.png")


if __name__ == "__main__":
    main()
