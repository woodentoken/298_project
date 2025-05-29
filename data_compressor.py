import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from datetime import datetime, timedelta
import ipdb

COLUMN_LOOKUP = {
    "garmin": [
        "timestamp",
        "position_lat",
        "position_long",
        "enhanced_altitude",
        "enhanced_speed",
        "heart_rate",
        "cadence",
        "distance",
        "power",
    ],
    "wind": None,  # take all columns"
    "iphone": ["Date", "Speed", "X", "Y", "Z", "G"],
}
TIME_LOOKUP = {
    "garmin": "timestamp",
    "wind": "Time",
    "iphone": "Date",
}

MPH2MS = 0.44704  # Conversion factor from miles per hour to meters per second
GRAVITY = 9.81  # Acceleration due to gravity in m/s^2
KM2MS = 1000 / 3600  # Conversion factor from kilometers per hour to meters per second

class Dataset:
    def __init__(self, data):
        self.data = data

    def plot(self):
        plt.plot(self.data)
        plt.show()


def compress_data(folder_array):
    # instantiate empty dataframes for each type
    datasets = {"garmin": pl.DataFrame(), "wind": pl.DataFrame(), "iphone": pl.DataFrame()}

    for folder in folder_array:
        if os.path.exists(folder):
            print(f"Processing folder: {folder}")

            # get all csv files in the folder
            csv_files = [f for f in os.listdir(folder) if f.endswith(".csv") or f.endswith(".CSV")]

            # identify the type of the data source
            source_type = "garmin" if "garmin" in folder.lower() else "wind" if "wind" in folder.lower() else "iphone"

            print(f"Found {len(csv_files)} CSV files of type: {source_type}")

            # for each file, append to the dataframe according to its source type
            for file in csv_files:
                file_path = os.path.join(folder, file)

                # decide which columns to extract based on the source type
                raw_df = pl.read_csv(file_path, columns=COLUMN_LOOKUP.get(source_type, None))

                # the iphone data needs to be recast to remove the micro second information (apparently)
                if source_type == "iphone":
                    raw_df = raw_df.with_columns(
                        pl.col("Date").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S.%f"),
                    )

                # unify the time columns (rename them to "time") and make them datetime opjects
                df = raw_df.with_columns(
                    time=pl.col(TIME_LOOKUP[source_type]).cast(pl.Datetime),
                ).drop(TIME_LOOKUP[source_type])
                # outsource type specific conversions and recasting to another function
                df = handle_source_specifics(df, source_type, file)

                print(f"    Loaded {file}: {df.shape[0]} rows, {df.shape[1]} columns")
                # append the new file information to the correct source in the datasets dictionary
                datasets[source_type] = pl.concat([datasets[source_type], df])

        else:
            print(f"Folder {folder} does not exist.")

    # construct a joint dataset by joining on the "time" column. times which do not coincide are dropped entirely
    joint_set = (
        datasets["wind"]
        .join(
            datasets["iphone"],
            on="time",
            how="inner",
        )
        .join(datasets["garmin"], on="time", how="inner", suffix="xxx")
    )

    return joint_set


def handle_source_specifics(df, source_type, file):
    if source_type == "garmin":
        df = df.with_columns(
            time=pl.col("time")
            - timedelta(hours=1),  # the garmin data is in a different DST zone, need to augment by 1 hour
            enhanced_altitude = pl.col("enhanced_altitude").cast(pl.Float64), # altitude is in meters
            enhanced_speed = pl.col("enhanced_speed").cast(pl.Float64)*KM2MS, # convert km/h to m/s
        )

    # convert wind to m/s
    if source_type == "wind":
        df = df.with_columns(
            pl.col("WindMPH").mul(MPH2MS).alias("Wind (m/s)"),
        )

    meta = file.split("_")
    # convert accelerations to m/s^2
    if source_type == "iphone":
        df = df.with_columns(
            pl.col("X").cast(pl.Float64).mul(GRAVITY).alias("X (m/s^2)"),
            pl.col("Y").cast(pl.Float64).mul(GRAVITY).alias("Y (m/s^2)"),
            pl.col("Z").cast(pl.Float64).mul(GRAVITY).alias("Z (m/s^2)"),
            pl.col("G").cast(pl.Float64).mul(GRAVITY).alias("G (m/s^2)"),
            meta_stance=pl.lit(meta[1]),
            meta_speed_mph=pl.lit(meta[3]),
            meta_direction=pl.lit(meta[4].split(".")[0]),
        )

    return df


if __name__ == "__main__":
    folder_array = ["Garmin_CSV_files", "Wind_sensor_data", "Iphone_CSV_files"]
    joint_set = compress_data(folder_array)

    # TODO reorder data set so that time is the first column

    meta_filters = {"speed": ["13", "19"], "position": ["up", "drop", "mix"], "direction": ["EW", "WE"]}
    joint_set.write_csv("master_data_set.csv")
