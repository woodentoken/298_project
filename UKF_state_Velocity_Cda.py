import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

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

def load_upsampled_processed_df(log_number_str, test_date, stop_distance):
    test_case_string = test_case_generator(log_number_str, test_date, stop_distance)
    file_path = f"Processed_data/Test_{test_date}/{test_case_string}/Combined_upsampled_data.csv"
    df = pd.read_csv(file_path)
    return df

def sigma_points(x, P, lam):
    """
    Generate 2n+1 sigma points for an n‐dimensional state x with covariance P,
    """
    n = x.size
    S = np.linalg.cholesky((n + lam) * P)  
    sigmas = np.zeros((2 * n + 1, n))
    sigmas[0] = x

    for i in range(n):
        sigmas[i + 1]     = x + S[i]
        sigmas[n + i + 1] = x - S[i]
    return sigmas

def transform_unscented(sigmas, Wm, Wc, noise_cov):
    """
    Given 2n+1 propagated sigma points (shape (2n+1, n)), weight vectors Wm, Wc,
    and additive noise covariance, return (mean, covariance).
    """
    x_bar = np.dot(Wm, sigmas)                
    Y     = sigmas - x_bar                    
    P     = Y.T @ np.diag(Wc) @ Y + noise_cov 
    return x_bar, P

def f_process(x, u, dt):
    """
    state = [v, a, CdA]
    """
    v, CdA = x
    a = u  # directly use input acceleration
    v_next = v + a * dt
    return np.array([v_next, CdA])

def h_measure(sigma, rho_air, mass, slope_rad, wind_speed,
              accel_meas, Crr, eta_drive):
    """
    sigma = [v, CdA]
    accel_meas = externally measured a_long
    """
    v, CdA = sigma
    g = 9.80665
    v_air = wind_speed

    F_aero = 0.5 * rho_air * CdA * v_air**2
    F_grav = mass * g * np.sin(slope_rad)
    F_roll = mass * g * Crr * np.cos(slope_rad)
    F_acc  = mass * accel_meas   # input from sensors

    P_wheel = (F_aero + F_grav + F_roll + F_acc) * v
    P_crank = P_wheel / eta_drive

    return np.array([P_crank])



def ukf_cda_step(x, P,
                 accel_meas, slope_rad, wind_speed, power_meas_crank,
                 dt, mass, rho_air, Crr, Q, R_crank,
                 alpha, beta, kappa,
                 eta_drive,
                 innov_gate_W=2000.0,      
                 accel_minus_bias=True):   
    
    # UKF prediction 
    n   = 2
    lam = alpha**2 * (n + kappa) - n
    Wm  = np.full(2*n+1, 0.5/(n+lam))
    Wc = Wm.copy()
    Wm[0] = lam/(n+lam); Wc[0] = Wm[0] + (1-alpha**2+beta)
    sigmas = sigma_points(x, P, lam)
    sig_f  = np.array([f_process(s, accel_meas, dt) for s in sigmas])

    x_pred, P_pred = transform_unscented(sig_f, Wm, Wc, Q)

    #  Measurement update
    sig_h = np.zeros((2 * n + 1, 1))
    for i, s in enumerate(sig_f):
        sig_h[i, 0] = h_measure(s, rho_air, mass, slope_rad, wind_speed,
                                accel_meas, Crr, eta_drive)[0]


    y_pred, Pyy = transform_unscented(sig_h, Wm, Wc,
                                       np.array([[(eta_drive**2)*R_crank]]))
    diff_x = sig_f - x_pred           # (2n+1,3)
    diff_y = sig_h[:,0] - y_pred      # (2n+1,)
    Pxy    = (diff_x.T * Wc) @ diff_y.reshape(-1,1)

    # innovation & gate
    innov = power_meas_crank - y_pred.item()
    if abs(innov) > innov_gate_W:   
        x_new, P_new = x_pred, P_pred
        P_crank_pred = y_pred.item()
    else:
        K     = Pxy / Pyy
        x_new = x_pred + K.flatten()*innov
        P_new = P_pred - K@K.T*Pyy
        P_crank_pred = y_pred.item()

    # clamp state to physical bounds 
    x_new[0] = np.clip(x_new[0], 0.0, 30.0)     # v
    x_new[1] = np.clip(x_new[1], 0.05, 1.2)     # CdA
    return x_new, P_new, P_crank_pred


if __name__ == "__main__":
    log_number_str = "001"
    test_date       = "06_01_2025"
    stop_distance   = 805    # (0.5 mile -> 805 m)

    # Load df
    df = load_upsampled_processed_df(log_number_str, test_date, stop_distance)
    df["time"] = pd.to_datetime(df["time"])

    # Parameters
    # Physical constants
    mass      = 81.5            # rider+bike (kg)
    rho_air   = 1.20            # air density (kg/m³)
    Crr       = 0.003           # rolling‐resistance
    eta_drive = 1               # drivetrain efficiency

    # (UKF parameters)
    alpha = 1e-3    # spread of sigma points
    beta  = 2.0     # optimal for Gaussian
    kappa = 0.0     # secondary scaling parameter

    # Process‐noise covariance
    Q = np.diag([
        (0.05)**2,        # v
        (0.001)**2,       # CdA 
    ])

    R_crank = 10.0   

    # Initial state & covariance
    v0 = float(df["velocity (m/s)"].iloc[0])
    x  = np.array([v0, 0.30])    # [v, CdA, bias]
    P  = np.diag([0.005, 0.005])   # initial uncertainty

    # Initialize output lists
    times, v_est, CdA_est, bias_est, power_pred_buf = [], [], [], [], []

    # UKF loop
    for k in range(1, len(df)):
        # Get dt
        dt = (df["time"].iat[k] - df["time"].iat[k-1]).total_seconds()

        # Read measurements
        accel_meas       = df["estimated_acceleration (m/s^2)"].iloc[k]
        power_meas_crank = df["power (W)"].iloc[k]
        wind_speed       = df["wind_speed (m/s)"].iloc[k]
        slope_rad        = df["gradient (rad)"].iloc[k]

        # Run 1 step
        x, P, P_crank_pred = ukf_cda_step(
            x, P,
            accel_meas,
            slope_rad,
            wind_speed,
            power_meas_crank,
            dt, mass, rho_air, Crr,
            Q, R_crank,
            alpha, beta, kappa,
            eta_drive=eta_drive
        )

        times.append(df["time"].iloc[k])
        v_est.append(x[0])
        CdA_est.append(x[1])
        power_pred_buf.append(P_crank_pred)

    df_est = pd.DataFrame({
        "time":       times,
        "v_est":      v_est,
        "CdA_est":    CdA_est,
        "power_pred": power_pred_buf
    })
    print(df_est.head())

    # Create a single figure with 4 subplots and save to 'plot' folder

    # Ensure the plot directory exists
    plot_dir = "plot_no_bias"
    os.makedirs(plot_dir, exist_ok=True)

    fig, axs = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    # Speed subplot
    axs[0].plot(np.array(df["time"]), np.array(df["velocity (m/s)"]), label="Measured Speed")
    axs[0].plot(np.array(times), np.array(v_est), label="UKF Speed", color="red")
    axs[0].set_ylabel("Velocity (m/s)")
    axs[0].set_title("UKF: Estimated Velocity vs Measured")
    axs[0].legend()
    axs[0].grid(True)

    # CdA subplot
    axs[1].plot(np.array(times), np.array(CdA_est), color="purple")
    axs[1].set_ylabel("CdA (m²)")
    axs[1].set_title("UKF: Estimated $C_dA$")
    axs[1].grid(True)


    # Power subplot
    axs[2].plot(np.array(times), np.array(df["power (W)"].iloc[1:]), label="Measured Power", color="blue")
    axs[2].plot(np.array(times), np.array(power_pred_buf), label="Predicted Power", color="orange")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Power (W)")
    axs[2].set_title("UKF: Predicted Power vs Measured")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    save_path = os.path.join(plot_dir, f"ukf_results_{log_number_str}_{test_date}_{stop_distance}m.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Plot saved to {save_path}")
