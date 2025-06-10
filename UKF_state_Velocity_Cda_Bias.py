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
    cdA_step_max=None,    
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
    

    return x_new, P_new, y_pred.item

def build_objective(df, eta_drive, mass, Crr):
    
    # ρ calculator & ukf_cda_step must already be in scope
    def single_run(params):
        alpha, beta, kappa, Q_v, Q_CdA, Q_bias, R_crank, x0_v, x0_CdA, x0_bias= params
        Q  = np.diag([Q_v**2, Q_CdA**2, Q_bias**2])
        x  = np.array([x0_v, x0_CdA, x0_bias])
        # P  = np.diag([x0_v**2, x0_CdA**2, x0_bias**2]) 
        P = np.diag([0.2**2, 0.05**2, 0.5**2])  # initial uncertainty

        v_est = []  # store velocity estimates

        try:
            for k in range(1, len(df)):
                dt = (df["time"].iat[k] - df["time"].iat[k-1]).total_seconds()
                if dt <= 0:               # skip bad rows
                    continue

                accel_meas       = df["acceleration_y_LOWPASS_filtered (m/s^2)"].iat[k]
                power_meas_crank = df["power (W)"].iat[k]
                wind_speed       = df["wind_speed (m/s)"].iat[k]
                slope_rad        = df["gradient (rad)"].iat[k]
                
                rho_air = rho_calculator(
                    df["temperature (C)"].iat[k],
                    df["altitude (m)"].iat[k]
                )
                
                x, P, _ = ukf_cda_step(
                    x, P,
                    accel_meas, slope_rad, wind_speed, power_meas_crank,
                    dt, mass, rho_air, Crr,
                    Q, R_crank,
                    alpha, beta, kappa,
                    eta_drive=eta_drive
                )
                v_est.append(x[0])

            v_true = df["velocity (m/s)"].iloc[1:len(v_est)+1]
            rmse   = np.sqrt(mean_squared_error(v_true, v_est))
            print(f"RMSE: {rmse:.4f}  Params: {params}")
            return rmse

        except Exception:
            return 1e6 

    @use_named_args(search_space)
    def objective(**named_params):
        ordered = [named_params[var.name] for var in search_space]
        return single_run(ordered)
    
    return objective

def optimise_ukf(df, n_calls, random_state, mass, Crr, eta_drive):
    """Runs Bayesian optimisation and returns the result object & tidy DF."""
    
    objective = build_objective(df, eta_drive, mass, Crr)
    
    result = gp_minimize(
        func           = objective,
        dimensions     = search_space,
        n_calls        = n_calls,       # total evaluations
        n_initial_points = 15,          # random starts
        acq_func       = "EI",          # expected-improvement
        random_state   = random_state,
        verbose        = True
    )
    
    # pretty summary
    res_df = pd.DataFrame(result.x_iters, columns=[d.name for d in search_space])
    res_df["rmse_velocity"] = result.func_vals
    res_df.sort_values("rmse_velocity", inplace=True)
    
    print("\nBest configuration:")
    print(res_df.iloc[0])
    
    return result, res_df


if __name__ == "__main__":
    log_number_str = "005"
    test_date      = "06_01_2025"
    stop_distance  = 805     

    #  load data
    upsampled_df, garmin_df, acceleration_df, wind_speed_df = load_processed_df(log_number_str, test_date, stop_distance)


    v0 = float(df["velocity (m/s)"].iloc[0])
    search_space = [
        Real(1e-4, 0.1,   name="alpha",   prior="log-uniform"),    # spread
        Real(1.99,  2.01,    name="beta"),                         # keep ~2
        Real(0.0,  1.0,    name="kappa"),                          # often 0
        Real(0.3,  1.0,    name="Q_v"),                            # Q_v  (m s⁻¹)
        Real(0.01, 0.3,    name="Q_CdA"),                          # Q_CdA
        Real(0.1,  1.5,    name="Q_bias"),                         # Q_bias
        Real(15.0, 900.0,  name="R_crank"),                        # crank-P 
        Real(v0 - 0.01, v0 + 0.01,   name="x0_v"),                 # initial v
        Real(0.1,  0.6,    name="x0_CdA"),                         # initial CdA
        Real(0, 2.0,    name="x0_bias")                            # initial bias
    ]

    #  constants & UKF settings
    mass      = 81.5
    Crr       = 0.0037
    eta_drive = 0.98      

    random_state = 1  # for reproducibility

    n_iterations = 300
    res_opt, df_trials = optimise_ukf(df, n_iterations, random_state, mass, Crr, eta_drive)
    # Svave the optimisation results
    output_folder = "Optimisation_results"
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"optimisation_results_{log_number_str}_{test_date}_{stop_distance}m.csv")
    df_trials.to_csv(output_file, index=False)

    print(f'Best parameters found:')
    print(df_trials.head(1))


    alpha = df_trials["alpha"].iloc[0]
    beta  = df_trials["beta"].iloc[0]
    kappa = df_trials["kappa"].iloc[0]

    Q = np.diag([
        df_trials["Q_v"].iloc[0]**2,
        df_trials["Q_CdA"].iloc[0]**2,
        df_trials["Q_bias"].iloc[0]**2
    ])

    R_crank = df_trials["R_crank"].iloc[0]

    #  initial state
    x = np.array([v0, df_trials["x0_CdA"].iloc[0], df_trials["x0_bias"].iloc[0]])
    P = np.diag([df_trials["x0_v"].iloc[0]**2,
                 df_trials["x0_CdA"].iloc[0]**2,
                 df_trials["x0_bias"].iloc[0]**2])
    
    #  run UKF
    times, v_est, CdA_est, bias_est, power_pred_buf = [], [], [], [], []

    for k in range(1, len(df)):
        dt = (df["time"].iat[k] - df["time"].iat[k - 1]).total_seconds()
        if dt <= 0:
            continue                            

        accel_meas       = df["acceleration_y_LOWPASS_filtered (m/s^2)"].iloc[k]
        power_meas_crank = df["power (W)"].iloc[k]
        wind_speed       = df["wind_speed (m/s)"].iloc[k]
        slope_rad        = df["gradient (rad)"].iloc[k]
        rho_air          = rho_calculator(
            df["temperature (C)"].iloc[k],
            df["altitude (m)"].iloc[k]
        )

        x, P, P_crank_pred = ukf_cda_step(
            x, P,
            accel_meas, slope_rad, wind_speed, power_meas_crank,
            dt, mass, rho_air, Crr,
            Q, R_crank,
            alpha, beta, kappa,
            eta_drive=eta_drive
        )
        if k % 100 == 0:
            print(f"t={times[-1]}  v_err={v_est[-1]-df['velocity (m/s)'].iloc[k]:.2f}  "
                f"CdA={x[1]:.3f}  bias={x[2]:.2f}"
                f"  acceleration={accel_meas:.2f}  ")
        times.append(df["time"].iloc[k])
        v_est.append(x[0])
        CdA_est.append(x[1])
        bias_est.append(x[2])
        power_pred_buf.append(P_crank_pred)

    df_est = pd.DataFrame({
        "time":       times,
        "v_est":      v_est,
        "CdA_est":    CdA_est,
        "accel_bias": bias_est,
        "power_pred": power_pred_buf,
    })
    
    df_slice = df.iloc[1 : len(v_est) + 1].reset_index(drop=True)

    # append raw-sensor columns
    df_est["velocity (m/s)"]                          = df_slice["velocity (m/s)"]
    df_est["acceleration_y_LOWPASS_filtered (m/s^2)"] = df_slice["acceleration_y_LOWPASS_filtered (m/s^2)"]
    df_est["gradient (rad)"]                          = df_slice["gradient (rad)"]
    df_est["wind_speed (m/s)"]                        = df_slice["wind_speed (m/s)"]
    df_est["temperature (C)"]                         = df_slice["temperature (C)"]
    df_est["altitude (m)"]                            = df_slice["altitude (m)"]
    df_est["power (W)"]                               = df_slice["power (W)"]

    output_results_folder = "UKF_estimated_results"
    os.makedirs(output_results_folder, exist_ok=True)

    out_fname = (f"ukf_estimated_results_{log_number_str}_"
                f"{test_date}_{stop_distance}m.csv")
    df_est.to_csv(os.path.join(output_results_folder, out_fname), index=False)
    print(f"UKF results saved to {out_fname}")

    # #  plotting
    # plot_dir = "plot"
    # os.makedirs(plot_dir, exist_ok=True)

    # fig, axs = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    # # speed
    # axs[0].plot(df["time"].to_numpy(), df["velocity (m/s)"].to_numpy(), label="Measured Speed")
    # axs[0].plot(np.array(times), np.array(v_est), label="UKF Speed", color="red")
    # axs[0].set_ylabel("Velocity (m/s)")
    # axs[0].set_title("UKF: Estimated Velocity vs Measured")
    # axs[0].legend()
    # axs[0].grid(True)

    # # CdA
    # axs[1].plot(np.array(times), np.array(CdA_est), color="purple")
    # axs[1].axhline(y=np.mean(CdA_est), color='orange', linestyle='--', label='Mean CdA')
    # axs[1].set_ylabel("CdA (m²)")
    # axs[1].set_title("UKF: Estimated $C_dA$")
    # axs[1].grid(True)

    # # bias
    # axs[2].plot(times, df["acceleration_y_LOWPASS_filtered (m/s^2)"].iloc[1:].to_numpy(), 
    #             label="Measured Acceleration", color="orange")
    # axs[2].plot(times, df["estimated_acceleration (m/s^2)"].iloc[1:].to_numpy(), label="Estimated accel", color="black")
    # axs[2].plot(times, bias_est, label="Estimated Bias", color="green")
    # axs[2].set_ylabel("m/s²")
    # axs[2].set_title("UKF: Acceleration Bias Estimation")
    # axs[2].legend()
    # axs[2].grid(True)

    # # power
    # axs[3].plot(times, df["power (W)"].iloc[1:].to_numpy(), label="Measured Power",
    #             color="blue")
    # # axs[3].plot(times, np.array(power_pred_buf), label="Predicted Power", color="orange")
    # axs[3].set_xlabel("Time")
    # axs[3].set_ylabel("Power (W)")
    # axs[3].set_title("UKF: Predicted Power vs Measured")
    # axs[3].legend()
    # axs[3].grid(True)

    # # airspeed
    # axs[4].plot(times, df["wind_speed (m/s)"].iloc[1:].to_numpy(), label="Wind Speed", color="cyan")
    # axs[4].set_xlabel("Time")
    # axs[4].set_ylabel("Wind Speed (m/s)")
    # axs[4].set_title("Wind Speed")
    # axs[4].legend()
    # axs[4].grid(True)

    # plt.tight_layout()
    # save_path = os.path.join(
    #     plot_dir, f"ukf_results_{log_number_str}_{test_date}_{stop_distance}m.png"
    # )
    # plt.savefig(save_path)
    # plt.close(fig)
    # print(f"Plot saved to {save_path}")
