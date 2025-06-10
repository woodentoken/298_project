import pandas as pd
from datetime import timedelta
import numpy as np
import matplotlib as mpl


freq_space = np.linspace(0.25, 2, 20)[::-1]  # Define frequency space for the colormap
# freq_space = np.array([2.0, 3.0, 4.0, 5.0])[::-1]  # Example frequency space in Hz
# freq_space = np.array([0.25, 0.75, 2.0])[::-1]  # Example frequency space in Hz

cmap = mpl.colormaps["magma"]

if len(freq_space) < 2:
    colors = "red"
    norm = mpl.colors.Normalize(vmin=freq_space[0]-0.5, vmax=freq_space[0]+0.5)
else:
    colors = cmap(freq_space / freq_space.max())  # Normalize to [0, 1] for colormap
    norm = mpl.colors.Normalize(vmin=freq_space.min(), vmax=freq_space.max())

def test_case_generator(log_number_str, test_date, stop_distance):
    target_speed = "14_6MPH"
    if log_number_str == "001":
        position = "upright"
        direction = "EW"
    elif log_number_str == "002":
        position = "upright"
        direction = "WE"
    elif log_number_str == "003":
        position = "drop"
        direction = "EW"
    elif log_number_str == "004":
        position = "drop"
        direction = "WE"
    elif log_number_str == "005":
        position = "mix"
        direction = "EW"
    elif log_number_str == "006":
        position = "mix"
        direction = "WE"
    else:
        raise ValueError("Invalid log number string. Must be '001', '002', etc.")

    test_case_string = f"Test_{test_date}_{target_speed}_{position}_{direction}_StopAT_{stop_distance}m"
    print(f"Generated test case string: {test_case_string}")
    return test_case_string


def round_datetime_to_100th(delta):
    # Total microseconds since the start of the second
    microseconds = delta.microsecond
    # 10,000 microseconds = 0.01 seconds
    rounded_us = round(microseconds / 10_000) * 10_000
    return delta.replace(microsecond=0) + timedelta(microseconds=rounded_us)


def load_upsampled_processed_df(log_number_str, test_date, stop_distance, cutoff=None):
    test_case_string = test_case_generator(log_number_str, test_date, stop_distance)
    if cutoff is None:
        file_path = f"Processed_data/Test_{test_date}/{test_case_string}/Combined_upsampled_data.csv"
    else:
        file_path = f"Processed_data/Test_{test_date}/{test_case_string}_{cutoff}/Combined_upsampled_data.csv"
    df = pd.read_csv(file_path)
    return df


def load_processed_dfs(log_number_str, test_date, stop_distance, cutoff=None):
    test_case_string = test_case_generator(log_number_str, test_date, stop_distance)
    file_path = f"Processed_data/Test_{test_date}/{test_case_string}"

    garmin_df = pd.read_csv(f"{file_path}/Garmin_data.csv")
    wind_df = pd.read_csv(f"{file_path}/Wind_speed_data.csv")
    if cutoff is not None:
        acceleration_df = pd.read_csv(f"{file_path}/Acceleration_data_{cutoff}.csv")
    else:
        acceleration_df = pd.read_csv(f"{file_path}/Acceleration_data.csv")

    return garmin_df, wind_df, acceleration_df


def rho_calculator(temperature, altitude):
    """Calculate air density based on temperature and altitude."""
    # Constants
    T0 = 288.15
    P0 = 101325
    R = 287.05

    P = P0 * (1 - (0.0065 * altitude) / T0) ** (9.81 / (0.0065 * R))

    T = temperature + 273.15
    rho = P / (R * T)
    return rho
