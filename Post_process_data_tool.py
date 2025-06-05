import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import os

def butter_lowpass_filter(data, cutoff=1.5, fs=100.0, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def upsampling_function(df_low, upsampling_freq):
    """
    df_low: DataFrame @ lower frequency with columns
    upsampling_freq: sampling frequency in Hz you want to upsample to
    """
    df_low_copy = df_low.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_low_copy['time']):
        df_low_copy['time'] = pd.to_datetime(df_low_copy['time'])
    df_low_copy = df_low_copy.sort_values('time').reset_index(drop=True)

    hz = float(upsampling_freq)
    period_ms = int(round(1000 / hz))  

    # Creat a new DateTimeIndex from the first to last timestamp at that frequency
    t_start = df_low_copy['time'].iloc[0]
    t_end   = df_low_copy['time'].iloc[-1]
    freq_str = f"{period_ms}L" 
    idx_new = pd.date_range(start=t_start, end=t_end, freq=freq_str)

    # Reindex df_low_copy onto this high‐rate index (introducing NaNs where no data existed)
    df_low_indexed = df_low_copy.set_index('time')
    df_reindexed   = df_low_indexed.reindex(idx_new)

    # Interpolate all numeric columns linearly in time
    df_interpolated = df_reindexed.interpolate(method='time')

    # Forward/backward fill any remaining (non-numeric) columns
    df_filled = df_interpolated.ffill().bfill()

    # Reset index so that 'time' becomes a column again
    df_upsampled = df_filled.reset_index().rename(columns={'index': 'time'})

    return df_upsampled


def process_garmin_data(log_number_str, test_date, stop_distance):
    """
    Process Garmin data from a CSV file and calculate gradient, fix the daylight saving time.
    Parameters:
    log_number_str (str): "001, 002, etc." 
    test_date (str)     : "MM_DD_YYYY"

    Returns:
    df: has time, velocity, distance, altitude, gradient, power， heartrate, cadence.
    """

    # Read the CSV file
    file_name = f"Raw_data\Test_{test_date}\Garmin_CSV_files\LOG_{log_number_str}.csv"
    df_OG = pd.read_csv(file_name)
    
    # Take needed columns
    df_needed = df_OG[['timestamp', 'heart_rate', 'cadence', 'distance', 'power', 'enhanced_speed', 'enhanced_altitude']]
    # Rename columns with units
    df_needed.columns = [
        'time', 
        'heart_rate (bpm)', 
        'cadence (rpm)', 
        'distance (m)', 
        'power (W)', 
        'velocity (m/s)', 
        'altitude (m)'
    ]
    df = df_needed.copy()

    # Fix the time format with daylight saving time
    df['time'] = pd.to_datetime(df['time']) - pd.Timedelta(hours=1)
    
    # Calculate the gradient in radians
    df['gradient (rad)'] = np.nan

    for i in range(1, len(df)):
        delta_dist = df.loc[i, 'distance (m)'] - df.loc[i - 1, 'distance (m)']
        if delta_dist > 0:
            delta_alt = df.loc[i, 'altitude (m)'] - df.loc[i - 1, 'altitude (m)']
            df.loc[i, 'gradient (rad)'] = np.arctan(delta_alt / delta_dist)

    # Calculate the estimated acceleration based on the change in velocity
    df['estimated_acceleration (m/s^2)'] = df['velocity (m/s)'].diff() / df['time'].diff().dt.total_seconds()
    df['estimated_acceleration (m/s^2)'].fillna(0, inplace=True)


    # Filter out rows where distance is greater than the stop distance
    df_filtered = df[df['distance (m)'] <= stop_distance].copy()
    df_cleaned = df_filtered.reset_index(drop=True)

    # Convert velocity from km/h to m/s
    df_cleaned['velocity (m/s)'] = df_cleaned['velocity (m/s)'] / 3.6

    # Check if any NaN values, and convert them to 0
    df_cleaned.fillna(0, inplace=True)

    start_time = df_cleaned['time'].iloc[0]
    stop_time = df_cleaned['time'].iloc[-1]

    return df_cleaned, start_time, stop_time

def process_acceleration_data(log_number_str, test_date, start_time, stop_time):
    """
    Process acceleration data from a CSV file, 
    Parameters:
    !!Note!!: start time and stop time are using from Garmin data.
    log_number_str (str): "001", "002", etc.
    test_date (str): "MM_DD_YYYY"

    Returns:
    df: DataFrame with raw, smoothed acceleration with flipped y-axis.
    """

    # Read the CSV file
    file_name = f"Raw_data/Test_{test_date}/Iphone_CSV_files/LOG_{log_number_str}.csv"
    df = pd.read_csv(file_name)

    # Select needed columns
    df_needed = df[['Date', 'X', 'Y', 'Z', 'G', 'Speed']]
    df_needed.columns = [
        'time',
        'acceleration_x (m/s^2)',
        'acceleration_y (m/s^2)',
        'acceleration_z (m/s^2)',
        'acceleration_G (m/s^2)',
        'velocity (m/s)'
    ]

    # Convert acceleration  to m/s^2
    for col in ['acceleration_x (m/s^2)', 'acceleration_y (m/s^2)', 'acceleration_z (m/s^2)', 'acceleration_G (m/s^2)']:
        df_needed[col] *= 9.80665

    df = df_needed.copy()

    # Subtract the bias from the first 100 samples
    # bias_x = df['acceleration_x (m/s^2)'].iloc[:100].mean()
    # bias_y = df['acceleration_y (m/s^2)'].iloc[:100].mean()
    # bias_z = df['acceleration_z (m/s^2)'].iloc[:100].mean()
    # df['acceleration_x (m/s^2)'] -= bias_x
    # df['acceleration_y (m/s^2)'] -= bias_y
    # df['acceleration_z (m/s^2)'] -= bias_z
    
    # Filter the DataFrame based on start and stop times
    df['time'] = pd.to_datetime(df['time'])
    df = df[(df['time'] >= start_time) & (df['time'] <= stop_time)].reset_index(drop=True)

    # Smoothing with butter lowpass filter
    df['acceleration_y_LOWPASS_filtered (m/s^2)'] = - butter_lowpass_filter(df['acceleration_y (m/s^2)'])

    return df

def process_wind_speed_data(log_number_str, test_date, start_time, stop_time):
    """
    Process wind speed data from a CSV file.
    Parameters:
    !!Note!!: start time and stop time are using from Garmin data.
    log_number_str (str): "001", "002", etc.
    test_date (str): "MM_DD_YYYY"

    Returns:
    df: DataFrame with wind speed data.
    """
    # Read the CSV file
    file_name = f"Raw_data/Test_{test_date}/Wind_sensor_CSV_files/LOG_{log_number_str}.csv"
    df_raw = pd.read_csv(file_name)

    # Keep only Time & WindMPS, rename columns
    df = df_raw[['Time', 'WindMPS']].copy()
    df.columns = ['time', 'wind_speed (m/s)']

    # Convert 'time' to datetime64 and sort
    df['time'] = pd.to_datetime(df['time'])

    # Fix the time format with 16 seconds offset (this clock is 16 seconds behind)
    df['time'] = df['time'] + pd.Timedelta(seconds=16)
    df = df.sort_values('time').reset_index(drop=True)

    # Trim to start_time, stop_time window
    df = df[(df['time'] >= start_time) & (df['time'] <= stop_time)].reset_index(drop=True)

    df['sec'] = df['time'].dt.floor('S')
    df_grouped = df.groupby('sec')['wind_speed (m/s)'].mean().reset_index()
    df_grouped.columns = ['time','wind_speed (m/s)']

    t0 = start_time.floor('S')
    t1 = stop_time.ceil('S')
    full_seconds = pd.DataFrame({'time': pd.date_range(start=t0, end=t1, freq='1S')})

    df_1hz = full_seconds.merge(df_grouped, on='time', how='left')

    df_1hz['wind_speed (m/s)'].fillna(method='ffill', inplace=True)
    df_1hz['wind_speed (m/s)'].fillna(method='bfill', inplace=True)

    return df_1hz

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

    return test_case_string

def combine_upsampled_dataframes(garmin_upsampled_df, wind_sensor_upsampled_df, acceleration_df):
    # Combine three upsampled DataFrames into one by the index, and replace time with the first DataFrame's time
    g = garmin_upsampled_df.copy().reset_index(drop=True)
    w = wind_sensor_upsampled_df.copy().reset_index(drop=True)
    a = acceleration_df.copy().reset_index(drop=True)

    # Clip all to the minimum length
    min_len = min(len(g), len(w), len(a))
    if len(g) != min_len or len(w) != min_len or len(a) != min_len:
        g = g.iloc[:min_len].reset_index(drop=True)
        w = w.iloc[:min_len].reset_index(drop=True)
        a = a.iloc[:min_len].reset_index(drop=True)

    combined = g.copy()
    combined['wind_speed (m/s)'] = w['wind_speed (m/s)']
    combined['acceleration_x (m/s^2)']           = a['acceleration_x (m/s^2)']
    combined['acceleration_y (m/s^2)']           = a['acceleration_y (m/s^2)']
    combined['acceleration_z (m/s^2)']           = a['acceleration_z (m/s^2)']
    combined['acceleration_G (m/s^2)']           = a['acceleration_G (m/s^2)']
    combined['acceleration_y_LOWPASS_filtered (m/s^2)'] = a['acceleration_y_LOWPASS_filtered (m/s^2)']

    # Make sure ‘time’ is a datetime column, and set as index
    combined['time'] = pd.to_datetime(combined['time'])
    combined.set_index('time', inplace=True)

    return combined

if __name__ == "__main__":
    test_date = "06_01_2025"
    stop_distance = 805 # meters - 0.5 mile x 1609.34 ~ 805 meters

    # Loop through LOG number from 001 to 006 and save to 
    for i in range(1, 2):
        log_number_str = f"{i:03}"  # Format as "001", "002", etc.
        test_case_string = test_case_generator(log_number_str, test_date, stop_distance)
        print(f"Generated Test Case: {test_case_string}")

        # Process the data
        garmin_df, start_time, stop_time = process_garmin_data(log_number_str, test_date, stop_distance)
        accel_df = process_acceleration_data(log_number_str, test_date, start_time, stop_time)
        wind_speed_df = process_wind_speed_data(log_number_str, test_date, start_time, stop_time)

        # Upsample Garmin data to match acceleration data
        garmin_df_upsampled = upsampling_function(garmin_df, 100)

        # Upsample wind speed data to match acceleration data
        wind_speed_df_upsampled = upsampling_function(wind_speed_df, 100)

        plt.figure(figsize=(12, 6))
        plt.plot(garmin_df_upsampled['time'].to_numpy(), garmin_df_upsampled['velocity (m/s)'].to_numpy(), label='Velocity (m/s)', color='green')
        plt.plot(garmin_df['time'].to_numpy(), garmin_df['velocity (m/s)'].to_numpy(), label='Original Velocity (m/s)', color='red', linestyle='--')
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(wind_speed_df_upsampled['time'].to_numpy(), wind_speed_df_upsampled['wind_speed (m/s)'].to_numpy(), label='Wind Speed (m/s)', color='blue')
        plt.plot(wind_speed_df['time'].to_numpy(), wind_speed_df['wind_speed (m/s)'].to_numpy(), label='Original Wind Speed (m/s)', color='orange', linestyle='--')
        plt.show()


        plt.figure(figsize=(12, 6))
        plt.plot(accel_df['time'].to_numpy(), accel_df['acceleration_y_LOWPASS_filtered (m/s^2)'].to_numpy(), label='Acceleration Y (m/s²)', color='blue')
        plt.plot(garmin_df_upsampled['time'].to_numpy(), garmin_df_upsampled['estimated_acceleration (m/s^2)'].to_numpy(), label='Estimated Acceleration (m/s²)', color='orange', linestyle='--')
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(accel_df['time'].to_numpy(), accel_df['acceleration_y_LOWPASS_filtered (m/s^2)'].to_numpy(), label='Filtered Acceleration Y (m/s²)', color='red')
        #plt.plot(accel_df['time'].to_numpy(), accel_df['acceleration_y (m/s^2)'].to_numpy(), label='Original Acceleration Y (m/s²)', color='orange', linestyle='--', alpha=0.5)
        plt.show()


        # Save the processed data to CSV files
        if not os.path.exists(f"Processed_data/Test_{test_date}/{test_case_string}"):
            os.makedirs(f"Processed_data/Test_{test_date}/{test_case_string}")
        garmin_df.to_csv(f"Processed_data/Test_{test_date}/{test_case_string}/Garmin_data.csv", index=False)
        accel_df.to_csv(f"Processed_data/Test_{test_date}/{test_case_string}/Acceleration_data.csv", index=False)
        wind_speed_df.to_csv(f"Processed_data/Test_{test_date}/{test_case_string}/Wind_speed_data.csv", index=False)

        combined_df = combine_upsampled_dataframes(garmin_df_upsampled, wind_speed_df_upsampled, accel_df)
        combined_df.to_csv(f"Processed_data/Test_{test_date}/{test_case_string}/Combined_upsampled_data.csv", index=True)

        print(f"Processed data for LOG_{log_number_str} saved successfully.")

# For plotting the data




    



    
