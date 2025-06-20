import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from datetime import datetime, timedelta
import ipdb
from itertools import product

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

# which final columns to keep in the joint dataset
JOINT_KEEP = [
    "WindAD",
    "absolute_time",
    "Wind (m/s)",
    "Speed",
    "X (m/s^2)",
    "Y (m/s^2)",
    "Z (m/s^2)",
    "G (m/s^2)",
    "X",
    "Y",
    "Z",
    "G",
    "meta_stance",
    "meta_speed_mph",
    "meta_direction",
    "position_lat",
    "position_long",
    "heart_rate",
    "cadence",
    "distance",
    "power",
    "enhanced_speed",
    "enhanced_altitude",
]

MPH2MS = 0.44704  # Conversion factor from miles per hour to meters per second
GRAVITY = 9.81  # Acceleration due to gravity in m/s^2
KM2MS = 1000 / 3600  # Conversion factor from kilometers per hour to meters per second


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
                    absolute_time=pl.col(TIME_LOOKUP[source_type]).cast(pl.Datetime),
                ).drop(TIME_LOOKUP[source_type])

                # outsource type specific conversions and recasting to another function
                df = handle_source_specifics(df, source_type, file)

                print(f"    Loaded {file}: {df.shape[0]} rows, {df.shape[1]} columns")
                # append the new file information to the correct source in the datasets dictionary
                datasets[source_type] = pl.concat([datasets[source_type], df])

        else:
            print(f"Folder {folder} does not exist.")

    return construct_joint_set(datasets)


def construct_joint_set(datasets):
    # construct a joint dataset by joining on the "time" column. times which do not coincide are dropped entirely
    joint_set = (
        (
            datasets["wind"]
            .join(
                datasets["iphone"],
                on="absolute_time",
                how="inner",
            )
            .join(datasets["garmin"], on="absolute_time", how="inner", suffix="xxx")
        )
        .select(JOINT_KEEP)
        .sort("absolute_time")
    )

    # reorder columns to have "absolute_time" as the first column
    joint_set = joint_set.select(["absolute_time"] + [col for col in joint_set.columns if col != "absolute_time"])
    print(f"Joint dataset created with {joint_set.shape[0]} rows and {joint_set.shape[1]} columns.")

    return joint_set


def handle_source_specifics(df, source_type, file):
    if source_type == "garmin":
        df = df.with_columns(
            absolute_time=pl.col("absolute_time")
            - timedelta(hours=1),  # the garmin data is in a different DST zone, need to augment by 1 hour
            enhanced_altitude=pl.col("enhanced_altitude").cast(pl.Float64),  # altitude is in meters
            enhanced_speed=pl.col("enhanced_speed").cast(pl.Float64) * KM2MS,  # convert km/h to m/s
        )
        # if the power data is larger than 1000, set the value to 0 (this is an outlier)
        df = df.with_columns(
            pl.when(pl.col("power") > 1000).then(0).otherwise(pl.col("power")).alias("power"),
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


def compute_time_diff(df):
    # find all unique combinations of stance, speed, and direction
    stances = df["meta_stance"].unique().to_list()
    speeds = df["meta_speed_mph"].unique().to_list()
    directions = df["meta_direction"].unique().to_list()
    # create a cartesian product of the unique values
    combinations = list(product(stances, speeds, directions))

    # need to make a copy of the dataframe to avoid modifying the original one
    df_copy = deepcopy(df)

    differential_time_df = pl.DataFrame()
    for stance, speed, direction in combinations:
        df = df_copy.filter(
            (pl.col("meta_stance") == stance)
            & (pl.col("meta_speed_mph") == speed)
            & (pl.col("meta_direction") == direction)
        )
        # for each combination, compute the time difference from the first timestamp
        time_diff = df["absolute_time"] - df["absolute_time"].min()
        time_diff = time_diff.map_elements(lambda x: x.seconds, return_dtype=pl.Int64)
        # compute time in seconds from the first timestamp
        df = df.with_columns(
            time=time_diff,
        )
        # save the filtered dataframe to a CSV file
        differential_time_df = pl.concat([differential_time_df, df])

    # save the joint dataset to a CSV file
    # reorder columns to have "time" as the first column
    differential_time_df = differential_time_df.select(
        ["time"] + [col for col in differential_time_df.columns if col != "time"]
    )
    differential_time_df.write_csv("master_data_set.csv")
    print(
        f"Differential time dataset created with {differential_time_df.shape[0]} rows and {differential_time_df.shape[1]} columns."
    )

    print("Dataset saved as 'master_data_set.csv'.")


if __name__ == "__main__":
    folder_array = ["Garmin_CSV_files", "Wind_sensor_data", "Iphone_CSV_files"]
    joint_set = compress_data(folder_array)
    compute_time_diff(joint_set)

    print("Data compression complete.")
