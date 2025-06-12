import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize                     # Bayesian optimiser
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args


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

def load_processed_df(log_number_str, test_date, stop_distance):
    test_case_string = test_case_generator(log_number_str, test_date, stop_distance)
    upsampled_file_path = f"Processed_data/Test_{test_date}/{test_case_string}/Combined_upsampled_data.csv"
    garmin_file_path = f"Processed_data/Test_{test_date}/{test_case_string}/Garmin_data.csv"
    acceleration_file_path = f"Processed_data/Test_{test_date}/{test_case_string}/Acceleration_data.csv"
    wind_speed_file_path = f"Processed_data/Test_{test_date}/{test_case_string}/Wind_speed_data.csv"

    upsampled_df = pd.read_csv(upsampled_file_path)
    garmin_df = pd.read_csv(garmin_file_path)
    acceleration_df = pd.read_csv(acceleration_file_path)
    wind_speed_df = pd.read_csv(wind_speed_file_path)

    return upsampled_df, garmin_df, acceleration_df, wind_speed_df

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
        sigmas[i + 1]     = x + S[i]
        sigmas[n + i + 1] = x - S[i]
    return sigmas

def f_process(x, u, dt, accel_minus_bias=True):
    """state = [v, CdA, bias]"""
    v, cda, b = x
    v_next = v + (u - b if accel_minus_bias else u + b) * dt
    return np.array([v_next, cda, b])

def transform_unscented(sigmas, Wm, Wc, noise_cov):
    """Return unscented mean/cov of sigma set (+ additive noise)."""
    x_bar = np.dot(Wm, sigmas)
    Y     = sigmas - x_bar
    P     = Y.T @ np.diag(Wc) @ Y + noise_cov
    return x_bar, P

def h_measure(sigma, rho_air, mass, slope_rad,
              wind_speed, accel_meas, Crr, eta_drive):
    v, CdA, bias = sigma
    g = 9.80665
    v_air  = wind_speed               
    a_long = accel_meas - bias

    F_aero = 0.5 * rho_air * CdA * v_air**2
    F_grav = mass * g * np.sin(slope_rad)
    F_roll = mass * g * Crr * np.cos(slope_rad)
    F_acc  = mass * a_long

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
    x, P,
    accel_meas, slope_rad, wind_speed, power_meas_crank,
    dt, mass, rho_air, Crr, Q, R_crank,
    alpha, beta, kappa, eta_drive,
    innov_gate_W=2000.0,
    accel_minus_bias=True,
):
    n = 3
    lam = alpha**2 * (n + kappa) - n
    Wm  = np.full(2 * n + 1, 0.5 / (n + lam))
    Wc  = Wm.copy()
    Wm[0] = lam / (n + lam)
    Wc[0] = Wm[0] + (1 - alpha**2 + beta)

    #  prediction
    sigmas = sigma_points(x, P, lam)
    sig_f  = np.array([f_process(s, accel_meas, dt, accel_minus_bias) for s in sigmas])

    Q_eff  = Q * dt                        
    x_pred, P_pred = transform_unscented(sig_f, Wm, Wc, Q_eff)

    #  measurement
    sig_h = np.zeros((2 * n + 1, 1))
    for i, s in enumerate(sig_f):
        sig_h[i, 0] = h_measure(
            s, rho_air, mass, slope_rad,
            wind_speed, accel_meas, Crr, eta_drive
        )[0]

    y_pred, Pyy = transform_unscented(
        sig_h, Wm, Wc,
        np.array([[(eta_drive**2) * R_crank]])   
    )

    diff_x = sig_f - x_pred          # (2n+1,3)
    diff_y = sig_h[:, 0] - y_pred    # (2n+1,)
    Pxy    = (diff_x.T * Wc) @ diff_y.reshape(-1, 1)

    innov = power_meas_crank - y_pred.item()

    if abs(innov) <= innov_gate_W:
        K     = Pxy / Pyy
        x_new = x_pred + K.flatten() * innov
        P_new = P_pred - K @ K.T * Pyy
    else:                          
        x_new, P_new = x_pred, P_pred

    # absolute physical bounds
    x_new[0] = np.clip(x_new[0], 0.0, 30.0)     # v
    x_new[1] = np.clip(x_new[1], 0.0, 1.2)      # CdA
    x_new[2] = np.clip(x_new[2], -9.81, 9.81)   # bias

    # ensure P stays symmetric / PSD
    P_new = 0.5 * (P_new + P_new.T)

    return x_new, P_new, y_pred.item()

def build_objective(
    garmin_df,
    acceleration_df,
    wind_speed_df,
    mass,
    Crr,
    eta_drive
):
    fast_len = len(acceleration_df)
    slow_df  = garmin_df.merge(wind_speed_df, on="time", how="left")
    slow_len = len(slow_df)
    ratio    = fast_len // slow_len
    DT       = 0.01

    def single_run(params):
        alpha, beta, kappa, Q_v, Q_CdA, Q_bias, R_crank, x0_CdA, x0_bias = params

        # build noise & initials
        Q   = np.diag([Q_v, Q_CdA, Q_bias])
        R   = R_crank
        x   = np.array([float(acceleration_df["velocity (m/s)"].iat[0]), x0_CdA, x0_bias])
        P   = np.diag([0.2**2, 0.05**2, 0.5**2])

        v_est, v_true = [], []

        for fast_idx in range(1, fast_len):
            accel_meas = acceleration_df["acceleration_y_LOWPASS_filtered (m/s^2)"].iat[fast_idx]

            # 100 Hz predict
            x, P = ukf_prediction(x, P, accel_meas, Q, DT, alpha, beta, kappa)

            if fast_idx % ratio == 0:
                slow_idx = fast_idx // ratio
                if slow_idx < slow_len:
                    row = slow_df.iloc[slow_idx]
                    x, P, _ = ukf_prediction_measurement(
                        x, P,
                        accel_meas, DT, row,
                        Q, R,
                        mass, Crr, eta_drive,
                        alpha, beta, kappa
                    )
                    v_est.append(x[0])
                    v_true.append(slow_df["velocity (m/s)"].iat[slow_idx])
                    

        if not v_est:
            return 1e6
        return np.sqrt(mean_squared_error(v_true, v_est))
    

    @use_named_args(search_space)
    def objective(**kwargs):
        # preserve the order of search_space
        params = [kwargs[d.name] for d in search_space]
        return single_run(params)

    return objective


def optimise_ukf(
    garmin_df,
    acceleration_df,
    wind_speed_df,
    n_calls,
    random_state,
    mass,
    Crr,
    eta_drive
):
    obj = build_objective(
        garmin_df, acceleration_df, wind_speed_df,
        mass, Crr, eta_drive
    )

    result = gp_minimize(
        func              = obj,
        dimensions        = search_space,
        n_calls           = n_calls,
        n_initial_points  = min(15, n_calls//3),
        acq_func          = "EI",
        random_state      = random_state,
        verbose           = True
    )

    trials = pd.DataFrame(result.x_iters, columns=[d.name for d in search_space])
    trials["rmse_velocity"] = result.func_vals
    trials.sort_values("rmse_velocity", inplace=True)

    print("Best configuration:\n", trials.iloc[0])
    return result, trials

def ukf_prediction(x, P, accel_meas, Q, dt, alpha, beta, kappa):
    x_pred, P_pred, _ = ukf_cda_step(
        x, P,
        accel_meas,
        slope_rad=0.0, wind_speed=0.0, power_meas_crank=0.0,
        dt=dt, mass=0.0, rho_air=1.225, Crr=0.0,
        Q=Q, R_crank=0.0,
        alpha=alpha, beta=beta, kappa=kappa,
        eta_drive=1.0,
        innov_gate_W=-1.0
    )
    return x_pred, P_pred

def ukf_prediction_measurement(x, P, accel_meas, dt, slow_row,
                               Q, R_crank, mass, Crr, eta_drive,
                               alpha, beta, kappa):
    slope_rad  = slow_row["gradient (rad)"]
    wind_speed = slow_row["wind_speed (m/s)"]
    power_meas = slow_row["power (W)"]
    rho_air    = rho_calculator(slow_row["temperature (C)"], slow_row["altitude (m)"])
    x_upd, P_upd, P_pred = ukf_cda_step(
        x, P,
        accel_meas, slope_rad, wind_speed, power_meas,
        dt, mass, rho_air, Crr,
        Q, R_crank,
        alpha, beta, kappa,
        eta_drive
    )
    return x_upd, P_upd, P_pred


if __name__ == "__main__":
    log_number_str = "003"
    test_date      = "06_01_2025"
    stop_distance  = 805     

    #  load data
    upsampled_df, garmin_df, acceleration_df, wind_speed_df = load_processed_df(log_number_str, test_date, stop_distance)
    slow_df = garmin_df.merge(wind_speed_df, on="time", how="left")
    fast_df = acceleration_df.copy()

    # constants 
    mass = 81.5       # kg
    Crr  = 0.004      # rolling resistance coefficient
    eta_drive = 0.97  # drivetrain efficiency

    v0 = float(garmin_df["velocity (m/s)"].iloc[0])
    search_space = [
        Real(1e-6, 0.001,   name="alpha",   prior="log-uniform"),    # spread
        Real(1.999,  2.001,    name="beta"),                         # keep ~2
        Real(0.0,  0.0001,    name="kappa"),                         # often 0
        Real(0.0,  0.05,    name="Q_v"),                             # Q_v  (m s⁻¹)
        Real(0.001,  0.1,    name="Q_CdA"),                          # Q_CdA
        Real(0.5,  2,    name="Q_bias"),                             # Q_bias
        Real(5.0, 300.0,  name="R_crank"),                           # crank-P 
        Real(0.0,  0.3,    name="x0_CdA"),                           # initial CdA
        Real(0, 2.0,    name="x0_bias")                              # initial bias
    ]

    n_iterations = 200
    random_state = 2  # for reproducibility

    # run optimisation
    result, df_trials = optimise_ukf(
        garmin_df, acceleration_df, wind_speed_df, n_calls=n_iterations, random_state=random_state,
        mass=mass, Crr=Crr, eta_drive=eta_drive)
    
    output_folder = "Optimisation_results"
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"optimisation_results_{log_number_str}_{test_date}_{stop_distance}m.csv")
    df_trials.to_csv(output_file, index=False)

    print(f'Best parameters found:')
    print(df_trials.head(1))
    
    ALPHA = df_trials["alpha"].iloc[0]
    BETA  = df_trials["beta"].iloc[0]
    KAPPA = df_trials["kappa"].iloc[0]

    Q = np.diag([df_trials["Q_v"].iloc[0]**2,
                 df_trials["Q_CdA"].iloc[0]**2,
                df_trials["Q_bias"].iloc[0]**2])
    R_crank = df_trials["R_crank"].iloc[0]

    # constants
    fast_len = len(acceleration_df)
    slow_len = len(slow_df)
    ratio    = fast_len // slow_len   

    DT        = 0.01   

    # initial state
    v0 = float(garmin_df["velocity (m/s)"].iat[0])
    x  = np.array([v0, df_trials["x0_CdA"].iloc[0], df_trials["x0_bias"].iloc[0]])
    # initial covariance
    P  = np.diag([0.2**2, 0.05**2, 0.5**2])

    fast_times, v_est, CdA_est, bias_est = [], [], [], []
    slow_times, power_pred_buf            = [], []

    for fast_index, fast_time in enumerate(fast_df["time"]):
        if fast_index == 0:
            continue

        accel_meas = fast_df["acceleration_y_LOWPASS_filtered (m/s^2)"].iat[fast_index]

        # 100 Hz prediction step
        x, P = ukf_prediction(
            x, P,
            accel_meas,
            Q,
            DT,       # e.g. 0.01
            ALPHA,
            BETA,
            KAPPA
        )

        # Every ratio step, do 1 Hz update
        if fast_index % ratio == 0:
            slow_index = fast_index // ratio
            if slow_index < slow_len:
                row = slow_df.iloc[slow_index]

                x, P, P_crank_pred = ukf_prediction_measurement(
                    x, P,
                    accel_meas,
                    DT,
                    row,
                    Q,
                    R_crank,
                    mass,
                    Crr,
                    eta_drive,
                    ALPHA,
                    BETA,
                    KAPPA
                )

                slow_times.append(fast_time)
                power_pred_buf.append(P_crank_pred)

        fast_times.append(fast_time)
        v_est.append(x[0])
        CdA_est.append(x[1])
        bias_est.append(x[2])

    # build DataFrames
    fast_estimate_df = pd.DataFrame({
        "time":       fast_times,
        "v_est":      v_est,
        "CdA_est":    CdA_est,
        "accel_bias": bias_est,
    })

    slow_estimate_df = pd.DataFrame({
        "time":       slow_times,
        "power_pred": power_pred_buf,
    })

    # Ensure time columns are sorted and in datetime
    fast_estimate_df["time"] = pd.to_datetime(fast_estimate_df["time"])
    slow_estimate_df["time"] = pd.to_datetime(slow_estimate_df["time"])
    fast_estimate_df = fast_estimate_df.sort_values("time").reset_index(drop=True)
    slow_estimate_df = slow_estimate_df.sort_values("time").reset_index(drop=True)

    # Merge asof:
    merged_df = pd.merge_asof(
        fast_estimate_df,
        slow_estimate_df,
        on="time",
        direction="forward"
    )


    # Add measured acceleration from acceleration_df (already at fast rate)
    merged_df["acceleration_measured"] = acceleration_df["acceleration_y_LOWPASS_filtered (m/s^2)"].values[:len(merged_df)]

    # Add measured velocity from garmin_df (slow rate, forward-fill to fast rate)
    garmin_df_sorted = garmin_df.sort_values("time").reset_index(drop=True)
    # Ensure both DataFrames have 'time' as datetime64 dtype
    merged_df["time"] = pd.to_datetime(merged_df["time"])
    garmin_df_sorted["time"] = pd.to_datetime(garmin_df_sorted["time"])
    merged_df = pd.merge_asof(
        merged_df,
        garmin_df_sorted[["time", "velocity (m/s)", "power (W)"]],
        on="time"
    )
    merged_df.rename(columns={"velocity (m/s)": "velocity_measured", "power (W)": "power_measured"}, inplace=True)

    # fix the NaN with next value in power_pred
    merged_df["power_pred"] = merged_df["power_pred"].ffill()

    # Drop rows with any NaN values
    merged_df = merged_df.dropna()

    # check NahN values
    if merged_df.isnull().values.any():
        print("Warning: NaN values found in merged DataFrame.")

    print(merged_df)

    # Save the merged DataFrame
    output_results_folder = "UKF_estimated_results"
    os.makedirs(output_results_folder, exist_ok=True)

    out_fname = (f"ukf_estimated_results_{log_number_str}_"
                f"{test_date}_{stop_distance}m.csv")
    merged_df.to_csv(os.path.join(output_results_folder, out_fname), index=False)
    print(f"UKF results saved to {out_fname}")