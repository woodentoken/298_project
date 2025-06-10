import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def file_generator(log_number_str, test_date, stop_distance):
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
    
    test_case_string = f"ukf_estimated_results_{log_number_str}_{test_date}_{stop_distance}m"
    print(f"Generated test case string: {test_case_string}")
    return test_case_string

def load_data(file_path):
    """
    Load data from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return 
    
if __name__ == "__main__":
    log_number_str = "001"
    test_date      = "06_01_2025"
    stop_distance  = 805
    test_case = file_generator(log_number_str, test_date, stop_distance)

    filename = f"UKF_estimated_results\{test_case}.csv"
    df = load_data(filename)
    
    # Information printing
    # Mean of CdA
    mean_cda = df["CdA_est"].mean()
    print(f"Mean CdA: {mean_cda:.4f} m²")
    # Mean of acceleration bias
    mean_accel_bias = df["accel_bias"].mean()
    print(f"Mean Acceleration Bias: {mean_accel_bias:.4f} m/s²")
    # RMSE of velocity
    rmse_velocity = np.sqrt(np.mean((df["velocity_measured"] - df["v_est"]) ** 2))
    print(f"RMSE of Velocity: {rmse_velocity:.4f} m/s")
    # RMSE of power
    rmse_power = np.sqrt(np.mean((df["power_measured"] - df["power_pred"]) ** 2))
    print(f"RMSE of Power: {rmse_power:.4f} W")

    # ──────────────────────────────────────────────────────────
    plot_dir = "plot"
    os.makedirs(plot_dir, exist_ok=True)

    fig, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    idx = df.index.to_numpy()  # fast integer index

    # Velocity
    axs[0].plot(idx, df["velocity_measured"].to_numpy(), label="Measured Velocity", color="black", linewidth=3)
    axs[0].plot(idx, df["v_est"].to_numpy(),             label="UKF Estimated Velocity", color="steelblue",  linewidth=3)
    axs[0].set_ylabel("Velocity (m/s)")
    axs[0].set_title("Estimated Velocity vs Measured")
    axs[0].legend()
    axs[0].grid(axis='x', which='both')

    # CdA
    axs[1].plot(idx, df["CdA_est"].to_numpy(), color="steelblue", linewidth=3, label="Estimated CdA")
    axs[1].axhline(df["CdA_est"].mean(), color="crimson", linestyle="--", linewidth= 3, label="Mean CdA")
    axs[1].set_ylabel("CdA (M²)")
    axs[1].legend()
    axs[1].grid(axis='x', which='both')

    # Acceleration bias
    axs[2].plot(idx, df["accel_bias"].to_numpy(),            label="UKF Estimated Bias", color="steelblue", linewidth=3)
    axs[2].plot(idx, df["acceleration_measured"].to_numpy(), label="Measured Acceleration", color="black",  linewidth=3)
    axs[2].set_title("Acceleration Bias Estimation")
    axs[2].set_ylabel("m/s²")
    axs[2].legend()
    axs[2].grid(axis='x', which='both')

    # Power
    axs[3].plot(idx, df["power_pred"].to_numpy(),     label="Predicted Power", color="steelblue", linewidth=3)
    axs[3].plot(idx, df["power_measured"].to_numpy(), label="Measured Power",  color="black", linewidth=3)
    axs[0].set_title("Estimated Power vs Measured")
    axs[3].set_ylabel("Power (W)")
    axs[3].set_xlabel("Time (s)")
    # Set x-ticks and labels so that x-axis shows time in seconds (idx/100)
    xticks = np.arange(0, 121 * 100, step=20 * 100)  # 0, 2000, 4000, ..., 12000
    xticks = xticks[xticks < len(idx)]  # Ensure ticks are within data range
    axs[3].set_xticks(xticks)
    axs[3].set_xticklabels((xticks / 100).astype(int))
    axs[3].legend()
    axs[3].grid(axis='x', which='both')

    plt.tight_layout()
    save_path = os.path.join(
        plot_dir, f"ukf_results_{log_number_str}_{test_date}_{stop_distance}m.png"
    )
    plt.savefig(save_path, dpi=1000)
    plt.show()
    plt.close(fig)
    print(f"Plot saved to {save_path}")
