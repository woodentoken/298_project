import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import (
    load_processed_dfs,
    load_upsampled_processed_df,
    rho_calculator,
    round_datetime_to_100th,
)

from utils import freq_space, colors, norm, cmap

DT = 0.01  # time step in seconds, used in the UKF prediction
MASS = 81.5  # mass of the cyclist + bike in kg
CRR = 0.004  # coefficient of rolling resistance
ETA_DRIVE = 0.97  # drive train efficiency
ALPHA = 0.003  # UKF alpha parameter
BETA = 2.3  # UKF beta parameter
KAPPA = 1.0  # UKF kappa parameter
INNOV_GATE_W = 1000  # innovation gate for power measurement in watts


def sigma_points(x, P, lam, jitter=1e-8):
    """Generate 2n+1 sigma points for state x ~ N(x,P)."""
    n = x.size
    try:
        S = np.linalg.cholesky((n + lam) * P)
    except np.linalg.LinAlgError:
        S = np.linalg.cholesky((n + lam) * (P + jitter * np.eye(n)))
    sigmas = np.zeros((2 * n + 1, n))
    sigmas[0] = x
    for i in range(n):
        sigmas[i + 1] = x + S[i]
        sigmas[n + i + 1] = x - S[i]
    return sigmas


def f_process(x, acceleration_measurement, accel_minus_bias=True):
    """state = [v, CdA, bias]"""
    v, cda, bias = x
    v_next = v + (acceleration_measurement - bias if accel_minus_bias else acceleration_measurement + bias) * DT
    return np.array([v_next, cda, bias])


def transform_unscented(sigmas, Wm, Wc, noise_cov):
    """Return unscented mean/cov of sigma set (+ additive noise)."""
    x_bar = np.dot(Wm, sigmas)
    Y = sigmas - x_bar
    P = Y.T @ np.diag(Wc) @ Y + noise_cov
    return x_bar, P


def h_measure(sigma, rho_air, mass, slope_rad, wind_speed, accel_meas, Crr, eta_drive):
    v, CdA, bias = sigma
    g = 9.80665
    v_air = wind_speed
    a_long = accel_meas - bias

    F_aero = 0.5 * rho_air * CdA * v_air**2
    F_grav = mass * g * np.sin(slope_rad)
    F_roll = mass * g * Crr * np.cos(slope_rad)
    F_acc = mass * a_long

    P_wheel = (F_aero + F_grav + F_roll + F_acc) * v
    P_crank = P_wheel / eta_drive
    return np.array([P_crank])


def ukf_prediction_measurement(
    x,
    P,
    accel_meas,
    slow_data,
    Q,
    R_crank,
    accel_minus_bias=True,
):
    power_measured = slow_data["power (W)"]
    wind_speed = slow_data["wind_speed (m/s)"]
    slope_rad = slow_data["gradient (rad)"]
    rho_air = rho_calculator(slow_data["temperature (C)"], slow_data["altitude (m)"])
    n = 3
    lam = ALPHA**2 * (n + KAPPA) - n
    Wm = np.full(2 * n + 1, 0.5 / (n + lam))
    Wc = Wm.copy()
    Wm[0] = lam / (n + lam)
    Wc[0] = Wm[0] + (1 - ALPHA**2 + BETA)

    #  prediction
    sigmas = sigma_points(x, P, lam)
    sig_f = np.array([f_process(s, accel_meas, accel_minus_bias) for s in sigmas])

    Q_eff = Q * DT
    x_pred, P_pred = transform_unscented(sig_f, Wm, Wc, Q_eff)

    #  measurement
    sig_h = np.zeros((2 * n + 1, 1))
    for i, s in enumerate(sig_f):
        sig_h[i, 0] = h_measure(s, rho_air, MASS, slope_rad, wind_speed, accel_meas, CRR, ETA_DRIVE)[0]

    y_pred, Pyy = transform_unscented(sig_h, Wm, Wc, np.array([[(ETA_DRIVE**2) * R_crank]]))

    diff_x = sig_f - x_pred  # (2n+1,3)
    diff_y = sig_h[:, 0] - y_pred  # (2n+1,)
    Pxy = (diff_x.T * Wc) @ diff_y.reshape(-1, 1)

    innov = power_measured - y_pred.item()

    if abs(innov) <= INNOV_GATE_W:
        K = Pxy / Pyy
        x_new = x_pred + K.flatten() * innov
        P_new = P_pred - K @ K.T * Pyy
    else:
        x_new, P_new = x_pred, P_pred

    # absolute physical bounds
    x_new[0] = np.clip(x_new[0], 0.0, 50.0)  # v
    x_new[1] = np.clip(x_new[1], 0.0, 1.2)  # CdA
    x_new[2] = np.clip(x_new[2], -9.81, 9.81)  # bias

    # ensure P stays symmetric / PSD
    P_new = 0.5 * (P_new + P_new.T)

    return x_new, P_new, y_pred.item()


def ukf_prediction(
    x,
    P,
    accel_meas,
    Q,
    accel_minus_bias=True,
):
    n = 3
    lam = ALPHA**2 * (n + KAPPA) - n
    Wm = np.full(2 * n + 1, 0.5 / (n + lam))
    Wc = Wm.copy()
    Wm[0] = lam / (n + lam)
    Wc[0] = Wm[0] + (1 - ALPHA**2 + BETA)

    #  prediction
    sigmas = sigma_points(x, P, lam)
    sig_f = np.array([f_process(s, accel_meas, accel_minus_bias) for s in sigmas])

    Q_eff = Q * DT
    x_pred, P_pred = transform_unscented(sig_f, Wm, Wc, Q_eff)

    return x_pred, P_pred


def estimate_signals(QR_ratio, meta):
    log_number_str = meta["log_number_str"]
    test_date = meta["test_date"]
    stop_distance = meta["stop_distance"]
    cutoff = meta["cutoff"] if "cutoff" in meta else 0.5

    #  load data
    garmin_df, wind_df, fast_df = load_processed_dfs(log_number_str, test_date, stop_distance, cutoff=cutoff)

    # join garmin and wind data on time
    slow_df = garmin_df.merge(wind_df, on="time", how="left")
    slow_df["time"] = pd.to_datetime(slow_df["time"])
    fast_df["time"] = pd.to_datetime(fast_df["time"]).apply(round_datetime_to_100th)

    #  initial state
    v0 = float(fast_df["velocity (m/s)"].iloc[0])
    x = np.array([v0, 0.0, 0.0])
    P = np.diag([0.01**2, 0.1**2, 3**2])

    #  run UKF
    if QR_ratio is not None:
        Q = 1 * np.identity(3)
        R = 90000

        print(f"QR_ratio: {1 / R}")

        fast_times, v_est, CdA_est, bias_est = [], [], [], []
        slow_times, power_pred_buf = [], []

        for fast_index, fast_time in enumerate(fast_df["time"]):
            if fast_index == 0:
                # Skip the first index as we cannot calculate dt
                continue
            slow_index = np.where(slow_df["time"] == fast_time)[0]

            accel_meas = fast_df["acceleration_y_LOWPASS_filtered (m/s^2)"][fast_index]
            # slow time not matching fast time
            if slow_index.size == 0:
                # print(f"Warning: No matching time found in slow_df for fast_df time {fast_time}. Skipping this index.")
                slow_index = None
                x, P = ukf_prediction(
                    x,
                    P,
                    accel_meas,
                    Q,
                )
                fast_times.append(fast_time)
                v_est.append(x[0])
                CdA_est.append(x[1])
                bias_est.append(x[2])
            else:
                slow_index = slow_index[0]
                x, P, P_crank_pred = ukf_prediction_measurement(
                    x,
                    P,
                    accel_meas,
                    slow_df.iloc[slow_index],
                    Q,
                    R,
                )
                fast_times.append(fast_time)
                v_est.append(x[0])
                CdA_est.append(x[1])
                bias_est.append(x[2])

                slow_times.append(slow_df["time"][slow_index])
                power_pred_buf.append(P_crank_pred)

        fast_estimate_df = pd.DataFrame(
            {
                "time": fast_times,
                "v_est": v_est,
                "CdA_est": CdA_est,
                "accel_bias": bias_est,
            }
        )
        slow_estimate_df = pd.DataFrame(
            {
                "time": slow_times,
                "power_pred": power_pred_buf,
            }
        )
    else:
        fast_estimate_df = pd.DataFrame(columns=["time", "v_est", "CdA_est", "accel_bias"])
        slow_estimate_df = pd.DataFrame(columns=["time", "power_pred"])

    return fast_df, slow_df, fast_estimate_df, slow_estimate_df


#  plotting
def measured_plotting(fast_df, slow_df, fig, axs):
    fast_time = fast_df["time"]
    slow_time = slow_df["time"]

    # speed
    axs[0].plot(fast_time, fast_df["velocity (m/s)"].to_numpy(), label="Measured Speed", color="black", linewidth=5)
    axs[0].set_ylabel("Velocity (m/s)")
    axs[0].set_title("UKF: Estimated Velocity vs Measured")
    axs[0].set_ylim(0, 10)
    axs[0].grid(True, axis='x', linestyle='--', color='gray', linewidth=0.7)
    # vertical grid lines for each 10 seconds

    # CdA
    axs[1].set_ylabel("CdA (m²)")
    axs[1].set_title("UKF: Estimated $C_dA$")
    axs[1].grid(True, axis='x', linestyle='--', color='gray', linewidth=0.7)


    # bias
    axs[2].plot(
        fast_time,
        fast_df["acceleration_y_LOWPASS_filtered (m/s^2)"],
        label="Measured Acceleration",
        color="black",
        linewidth=5,
    )
    axs[2].set_ylabel("m/s²")
    axs[2].set_title("UKF: Acceleration Bias Estimation")
    axs[2].set_ylim(0, 3)
    axs[2].grid(True, axis='x', linestyle='--', color='gray', linewidth=0.7)


    # power
    axs[3].plot(slow_time, slow_df["power (W)"], label="Measured Power", color="black", linewidth=5)
    axs[3].set_ylabel("Power (W)")
    axs[3].set_title("UKF: Predicted Power vs Measured")
    axs[3].set_ylim(0, 500)
    axs[3].grid(True, axis='x', linestyle='--', color='gray', linewidth=0.7)



def estimated_plotting(fast_estimated_df, slow_estimated_df, fig, axs, color="black"):
    fast_time = fast_estimated_df["time"]
    slow_time = slow_estimated_df["time"]

    # speed
    axs[0].plot(fast_time, fast_estimated_df["v_est"], color=color, linewidth=3)

    # CdA
    axs[1].plot(fast_time, fast_estimated_df["CdA_est"], color=color, linewidth=3)

    # bias
    axs[2].plot(fast_time, fast_estimated_df["accel_bias"], color=color, linewidth=3)

    # power
    axs[3].plot(slow_time, slow_estimated_df["power_pred"], color=color, linewidth=3)


def execute_ukf(log_number_str="001", test_date="06_01_2025", stop_distance=805):
    meta = {}
    meta["log_number_str"] = log_number_str
    meta["test_date"] = test_date
    meta["stop_distance"] = stop_distance

    plot_dir = "plot"
    os.makedirs(plot_dir, exist_ok=True)
    fig, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    # # calculate the estimated signals and plot them per QR ratio
    for num, freq in enumerate(freq_space):
        meta["cutoff"] = freq
        _, _, fast_est_df, slow_est_df = estimate_signals(1, meta)
        estimated_plotting(fast_est_df, slow_est_df, fig, axs, color=colors[num])

    # calculate the measured signals alone and plot them
    fast_df, slow_data, _, _ = estimate_signals(None, meta)
    measured_plotting(fast_df, slow_data, fig, axs)

    for ax in axs:
        ax.legend(loc="upper right")

    # finalize the plot and save it
    plt.tight_layout()
    # divider = make_axes_locatable(fig.gca())
    # cax = divider.append_axes("right", size="4%", pad=0.05)
    # plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, label="Cutoff Frequency (Hz)")
    # cax.set_yticks(freq_space)
    save_path = os.path.join(
        plot_dir, f"ukf_results_{meta['log_number_str']}_{meta['test_date']}_{meta['stop_distance']}m.png"
    )
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    execute_ukf()
