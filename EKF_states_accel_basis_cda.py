import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ipdb
import scipy


def main():
    df = load_data("master_data_set.csv", 829, 1024)
    print(df.head())

    # Constants & tuning
    mass = 82.6
    rho = 1.20
    Crr = 0.005
    Q = np.diag([0.25, 0.05])
    R = np.diag([20000])

    # Initial state
    length = len(df["time"])

    x = np.ndarray((length, 2, 1))  # [v, CdA, bias]
    P = np.ndarray((length, 2, 2))

    x[0, 0] = float(df["Speed"].iloc[0])  # initial velocity
    x[0, 1] = 0.0  # initial bias

    P[0, :, :] = np.diag([0.0, 0.0])  # initial covariance

    slope_rad = np.ndarray(length)
    slope_rad[0] = 0.0  # initial slope in radians

    # Output
    # CdA_est = np.array([x[1]])
    residuals = np.ndarray((length))  # Initialize residuals
    residuals[0] = 0.0  # Initial residual

    kalman_gains = np.ndarray((length, 2, 1))  # Initialize Kalman gains
    kalman_gains[0, :, :] = np.array([[0.0], [0.0]])  # Initial Kalman gain

    # v_est, CdA_est, bias_est = [df["time"][0]], [x[0]], [0], [x[1]]
    # for var in [v_est, CdA_est, bias_est]:
    #     var = np.array(var)
    dt = df["time"].iloc[1] - df["time"].iloc[0]
    # Time step and measurements
    accel_meas = df["Y (m/s^2)"]
    power_meas = df["power"]
    wind_speed = df["Wind (m/s)"]

    # Calculate slope in radians
    alt = df["enhanced_altitude"]
    dist = df["distance"]

    # EKF loop
    for k in range(1, length):
        # ipdb.set_trace()
        slope_rad[k] = np.arctan((alt[k] - alt[k - 1]) / max(dist[k] - dist[k - 1], 0.5))
        x[k], P[k], _, residuals[k], kalman_gains[k] = ekf_cda_step(
            slope_rad[k],
            wind_speed[k],
            dt,
            mass,
            rho,
            Crr,
            Q,
            R,
            x[k - 1],
            P[k - 1],
            accel_meas[k - 1],
            power_meas[k - 1],
        )

    ekf_df = pd.DataFrame(
        {
            "t": df["time"],
            "v_est": x[:, 0, 0],
            # "CdA_est": x[:, 1, 0],
            "accel_bias": x[:, 1, 0],
            "residuals": residuals,
            "kalman_gains_1": kalman_gains[:, 0, 0],
            "kalman_gains_2": kalman_gains[:, 1, 0],
            "slope_deg": slope_rad * 180 / np.pi,
            "wind_speed": wind_speed,
        })

    plotting(df, ekf_df)


def load_data(file_path, start_row, end_row):
    df = pd.read_csv(file_path)
    df_new = df.iloc[start_row - 2 : end_row - 1]  # Adjust for 0-indexing
    return df_new.reset_index(drop=True)


# EKF one-step:
# state = [v, accel_bias]
def ekf_cda_step(slope_rad, wind_speed, dt, mass, rho, Crr, Q, R, states, variance, input, output):
    """
    x           : [v, CdA, bias] State vector
    P           : 3×3 covariance
    accel_meas  : measured longitudinal accel (m/s²)
    slope_rad   : road pitch (rad)
    wind_speed  : air-relative wind speed (m/s)
    y           : power meter (W)
    dt          : time step (s)
    mass, rho   : constants
    Crr         : rolling-resistance coefficient
    Q           : 3×3 process noise
    R           : scalar power-meas noise variance
    """
    CdA_pred = 0.3
    g = 9.80665

    # PREDICTION
    A_prime = np.array([[1.0, 1.0], [0.0, 1.0]])
    B_prime = np.array([[dt], [0.0]])
    # noise is added linearly to both states
    E_prime = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    x_pred = A_prime @ states + B_prime * input
    P_pred = A_prime @ variance @ A_prime.T + E_prime @ Q @ E_prime.T

    # MEASUREMENT UPDATE
    F_aero = 0.5 * rho * CdA_pred * wind_speed**2
    F_grav = mass * g * np.sin(slope_rad)
    F_roll = mass * g * Crr * np.cos(slope_rad)
    power_pred = (x_pred[0] + x_pred[1]) * (F_aero + F_grav + F_roll + mass * input)

    # Measurement Jacobian H = dh/dx  (1×3)
    dh_dv = F_aero + F_grav + F_roll + mass * input
    # dh_dCdA = 0.5 * rho * x_pred[0] * wind_speed**2
    dh_dbias = mass * x_pred[1][0]
    print(f"dh_dv: {dh_dv}, dh_dbias: {dh_dbias}")
    # ipdb.set_trace()

    # linearize the measurement equation based on the state
    C_prime = np.array([dh_dv, 0]).reshape(1, 2)  # 1×3 Jacobian matrix
    # no input in measurment equation
    D_prime = np.array(
        [
            [0.0],  # dP/dv
        ]
    )
    # noise is added in the measurement equation
    F_prime = np.array(
        [
            [1.0],  # dP/dv
        ]
    )

    # ipdbset_trace()

    # 4) KALMAN GAIN & UPDATE
    L = P_pred @ C_prime.T * np.linalg.inv(C_prime @ P_pred @ C_prime.T + F_prime @ R @ F_prime.T)
    residual = output - power_pred

    x_est = x_pred + L * residual
    P_est = P_pred - (L @ C_prime @ P_pred)

    return x_est, P_est, power_pred, residual, L

    # CdA_est.append(x[1])
    # bias_est = np.array(bias_est).flatten()
    # t = np.array(t).flatten()
    # v_est = np.array(v_est).flatten()
    # ipdb.set_trace()

    # ol_velocity = scipy.integrate.cumulative_trapezoid(bias_est, t, initial=0)

    # Build and show results
    # df_est = pd.DataFrame(
    #     {
    #         "time": t,
    #         "v_est": v_est,
    #         # "CdA_est":    CdA_est,
    #         "accel_bias": bias_est,
    #     }
    # )
    # print(df_est.head())


def plotting(df, ekf_df):
    t = df["time"]

    figure, axes = plt.subplots(3, 1, figsize=(10, 12))

    # plt.figure(figsize=(10, 4))
    axes[0].plot(t, df["enhanced_speed"], label="Actual Speed")
    axes[0].plot(t, ekf_df["v_est"], label="EKF Speed", color="red")
    # axes[0].plot(np.array(t), ol_velocity, label="Bias Speed", color="red")
    axes[0].legend()
    axes[0].set_title("Velocity: EKF vs Measured")
    axes[0].grid()

    # plt.figure(figsize=(10,4))
    # plt.plot(np.array(times), np.array(CdA_est), color='purple')
    # plt.title("Estimated $C_dA$"); plt.ylabel("CdA (m²)"); plt.grid(); plt.show()

    # Plot acceleration input
    # plt.figure(figsize=(10, 4))
    axes[1].plot(t, df["Y (m/s^2)"], label="Measured Acceleration", color="orange")
    axes[1].plot(t, ekf_df["accel_bias"], label="Estimated Bias", color="green")
    axes[1].legend()
    axes[1].set_title("Acceleration Bias Estimation")
    axes[1].grid()

    axes[2].plot(t, ekf_df["residuals"], label="Residuals", color="black")
    # axes[2].plot(t, np.array(bias_est), label="Estimated Bias", color="green")
    axes[2].legend()
    axes[2].set_title("Residuals (y - h(x))")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid()
    plt.show()


if __name__ == "__main__":
    main()
