import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


def load_upsampled_processed_df(log_number_str, test_date, stop_distance, cutoff=None):
    test_case_string = test_case_generator(log_number_str, test_date, stop_distance)
    if cutoff is None:
        file_path = f"Processed_data/Test_{test_date}/{test_case_string}/Combined_upsampled_data.csv"
    else:
        file_path = f"Processed_data/Test_{test_date}/{test_case_string}_{cutoff}/Combined_upsampled_data.csv"
    df = pd.read_csv(file_path)
    return df


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


def f_process(x, acceleration_measurement, dt, accel_minus_bias=True):
    """state = [v, CdA, bias]"""
    v, cda, bias = x
    v_next = v + (acceleration_measurement - bias if accel_minus_bias else acceleration_measurement + bias) * dt
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


def ukf_cda_step(
    x,
    P,
    accel_meas,
    slope_rad,
    wind_speed,
    power_meas_crank,
    dt,
    mass,
    rho_air,
    Crr,
    Q,
    R_crank,
    alpha,
    beta,
    kappa,
    eta_drive,
    innov_gate_W=2000.0,
    accel_minus_bias=True,
    cdA_step_max=None,
):
    n = 3
    lam = alpha**2 * (n + kappa) - n
    Wm = np.full(2 * n + 1, 0.5 / (n + lam))
    Wc = Wm.copy()
    Wm[0] = lam / (n + lam)
    Wc[0] = Wm[0] + (1 - alpha**2 + beta)

    #  prediction
    sigmas = sigma_points(x, P, lam)
    sig_f = np.array([f_process(s, accel_meas, dt, accel_minus_bias) for s in sigmas])

    Q_eff = Q * dt
    x_pred, P_pred = transform_unscented(sig_f, Wm, Wc, Q_eff)

    #  measurement
    sig_h = np.zeros((2 * n + 1, 1))
    for i, s in enumerate(sig_f):
        sig_h[i, 0] = h_measure(s, rho_air, mass, slope_rad, wind_speed, accel_meas, Crr, eta_drive)[0]

    y_pred, Pyy = transform_unscented(sig_h, Wm, Wc, np.array([[(eta_drive**2) * R_crank]]))

    diff_x = sig_f - x_pred  # (2n+1,3)
    diff_y = sig_h[:, 0] - y_pred  # (2n+1,)
    Pxy = (diff_x.T * Wc) @ diff_y.reshape(-1, 1)

    innov = power_meas_crank - y_pred.item()

    if abs(innov) <= innov_gate_W:
        K = Pxy / Pyy
        x_new = x_pred + K.flatten() * innov
        P_new = P_pred - K @ K.T * Pyy
    else:
        x_new, P_new = x_pred, P_pred

    # absolute physical bounds
    x_new[0] = np.clip(x_new[0], 0.0, 30.0)  # v
    x_new[1] = np.clip(x_new[1], 0.0, 1.2)  # CdA
    x_new[2] = np.clip(x_new[2], -9.81, 9.81)  # bias

    # ensure P stays symmetric / PSD
    P_new = 0.5 * (P_new + P_new.T)

    return x_new, P_new, y_pred.item()

def estimate_signals(QR_ratio, meta):
    log_number_str = meta["log_number_str"]
    test_date = meta["test_date"]
    stop_distance = meta["stop_distance"]
    cutoff = meta["cutoff"] if "cutoff" in meta else 0.5

    #  load data
    df_measured = load_upsampled_processed_df(log_number_str, test_date, stop_distance, cutoff=cutoff)
    df_measured["time"] = pd.to_datetime(df_measured["time"])

    #  constants & UKF settings
    mass = 81.5
    Crr = 0.004
    eta_drive = 0.97

    alpha = 0.003
    beta = 2.3
    kappa = 1.0

    #  initial state
    v0 = float(df_measured["velocity (m/s)"].iloc[0])
    x = np.array([v0, 0.0, 0.0])
    P = np.diag([0.01**2, 0.1**2, 3**2])

    #  run UKF
    if QR_ratio is not None:
        Q = np.diag(
            [
                1,  # v
                0.1,  # CdA
                0.3,  # bias
            ]
        )
        # Q = 1 * np.identity(3)
        R = 90000

        print(f"QR_ratio: {1 / R}")

        times, v_est, CdA_est, bias_est, power_pred_buf = [], [], [], [], []

        for k in range(1, len(df_measured)):
            dt = (df_measured["time"].iat[k] - df_measured["time"].iat[k - 1]).total_seconds()
            if dt <= 0:
                continue
            
            accel_meas = df_measured["acceleration_y_LOWPASS_filtered (m/s^2)"].iloc[k]
            power_meas_crank = df_measured["power (W)"].iloc[k]
            wind_speed = df_measured["wind_speed (m/s)"].iloc[k]
            slope_rad = df_measured["gradient (rad)"].iloc[k]
            rho_air = rho_calculator(df_measured["temperature (C)"].iloc[k], df_measured["altitude (m)"].iloc[k])

            x, P, P_crank_pred = ukf_cda_step(
                x,
                P,
                accel_meas,
                slope_rad,
                wind_speed,
                power_meas_crank,
                dt,
                mass,
                rho_air,
                Crr,
                Q,
                R,
                alpha,
                beta,
                kappa,
                eta_drive=eta_drive,
            )
            # if k % 100 == 0:
            # print(
            #     f"t={times[-1]}  v_err={v_est[-1] - df_measured['velocity (m/s)'].iloc[k]:.2f}  "
            #     f"CdA={x[1]:.3f}  bias={x[2]:.2f}"
            #     f"  acceleration={accel_meas:.2f}  "
            # )
            times.append(df_measured["time"].iloc[k])
            v_est.append(x[0])
            CdA_est.append(x[1])
            bias_est.append(x[2])
            power_pred_buf.append(P_crank_pred)

        df_est = pd.DataFrame(
            {
                "time": times,
                "v_est": v_est,
                "CdA_est": CdA_est,
                "accel_bias": bias_est,
                "power_pred": power_pred_buf,
            }
        )
        print(df_est.head())
    else:
        df_est = pd.DataFrame(columns=["time", "v_est", "CdA_est", "accel_bias", "power_pred"])

    return df_measured, df_est


#  plotting
def measured_plotting(measured_df, fig, axs):
    time = measured_df["time"]

    # speed
    axs[0].plot(time, measured_df["velocity (m/s)"].to_numpy(), label="Measured Speed", color="black", linewidth=5)
    axs[0].set_ylabel("Velocity (m/s)")
    axs[0].set_title("UKF: Estimated Velocity vs Measured")

    # CdA
    axs[1].set_ylabel("CdA (m²)")
    axs[1].set_title("UKF: Estimated $C_dA$")

    # bias
    axs[2].plot(
        time, measured_df["acceleration_y_LOWPASS_filtered (m/s^2)"], label="Measured Acceleration", color="black", linewidth=5
    )
    # axs[2].plot(time, measured_df["estimated_acceleration (m/s^2)"], label="Estimated accel", color="black")
    axs[2].set_ylabel("m/s²")
    axs[2].set_title("UKF: Acceleration Bias Estimation")
    axs[2].set_ylim(0,3)

    # power
    axs[3].plot(time, measured_df["power (W)"], label="Measured Power", color="black", linewidth=5)
    axs[3].set_ylabel("Power (W)")
    axs[3].set_title("UKF: Predicted Power vs Measured")

    # airspeed
    axs[4].plot(time, measured_df["wind_speed (m/s)"], label="Wind Speed", color="black", linewidth=5)
    axs[4].set_xlabel("Time")
    axs[4].set_ylabel("Wind Speed (m/s)")
    axs[4].set_title("Wind Speed")


def estimated_plotting(estimated_df, fig, axs,  color="black"):
    time = estimated_df["time"]

    # speed
    axs[0].plot(time, estimated_df["v_est"], color=color, linewidth=3)

    # CdA
    axs[1].plot(time, estimated_df["CdA_est"], color=color, linewidth=3)
    # axs[1].axhline(y=np.mean(estimated_df["CdA_est"]), color="orange", linestyle="--", label="Mean CdA")

    # bias
    axs[2].plot(time, estimated_df["accel_bias"], color=color, linewidth=3)

    # power
    axs[3].plot(time, estimated_df["power_pred"], color=color, linewidth=3)



if __name__ == "__main__":
    meta = {}
    meta["log_number_str"] = "001"
    meta["test_date"] = "06_01_2025"
    meta["stop_distance"] = 805

    plot_dir = "plot"
    os.makedirs(plot_dir, exist_ok=True)
    fig, axs = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    # calculate the measured signals alone and plot them
    measured_df, _ = estimate_signals(None, meta)
    measured_plotting(measured_df, fig, axs)

    cmap = mpl.colormaps['plasma']
    freq_space = np.linspace(0.5, 4.5, 5)[::-1]  # Define frequency space for the colormap
    colors = cmap(freq_space/freq_space.max())  # Normalize to [0, 1] for colormap
    norm = mpl.colors.Normalize(vmin=freq_space.min(), vmax=freq_space.max())

    # # calculate the estimated signals and plot them per QR ratio
    QR_ratio = np.array([100000])  # Example scaling factors for Q and R
    for num, freq in enumerate(freq_space):
        meta["cutoff"] = freq
        meta["QR_ratio"] = QR_ratio[0]
        _, estimated_df = estimate_signals(1, meta)
        estimated_plotting(estimated_df, fig, axs, color=colors[num])

    for ax in axs:
        ax.legend(loc='upper right')

    plt.tight_layout()
    divider = make_axes_locatable(fig.gca())
    cax = divider.append_axes("right", size="4%", pad=0.05)
    plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, label='Cutoff Frequency (Hz)')
    cax.set_yticks(freq_space)
    save_path = os.path.join(
        plot_dir, f"ukf_results_{meta['log_number_str']}_{meta['test_date']}_{meta['stop_distance']}m.png"
    )
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Plot saved to {save_path}")

