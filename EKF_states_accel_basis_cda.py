import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ipdb


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

    # v, acceleration_bias = x

    # Prediction
    # accel_true = accel_meas + bias
    # v_pred     = v + acceleration_bias + u * dt
    # CdA_pred   = CdA

    # x_pred = np.array([v_pred, acceleration_bias])
    # ipdb.set_trace()

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
    # ipdb.set_trace()
    x_pred = A_prime @ states + B_prime * input
    P_pred = A_prime @ variance @ A_prime.T + E_prime @ Q @ E_prime.T

    # MEASUREMENT UPDATE
    F_aero = 0.5 * rho * CdA_pred * wind_speed**2
    F_grav = mass * g * np.sin(slope_rad)
    F_roll = mass * g * Crr * np.cos(slope_rad)
    power_pred = x_pred[0] * (F_aero + F_grav + F_roll + mass * input)
    output = power_meas - power_pred

    # Measurement Jacobian H = dh/dx  (1×3)
    dh_dv = F_aero + F_grav + F_roll + mass * input
    # dh_dCdA = 0.5 * rho * x_pred[0] * wind_speed**2
    dh_dbias = mass * x_pred[1]

    # linearize the measurement equation based on the state
    C_prime = np.array([dh_dv, dh_dbias]).reshape(1, 2)  # 1×3 Jacobian matrix
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

    # 4) KALMAN GAIN & UPDATE
    # ipdb.set_trace()


    L = P_pred @ C_prime.T * np.linalg.inv(C_prime @ P_pred @ C_prime.T + F_prime @ R @ F_prime.T)
    residual = output - (C_prime @ x_pred + D_prime * input)

    x_est = x_pred + L @ residual
    P_est = P_pred - (L @ C_prime @ P_pred)

    return x_est, P_est, power_pred


if __name__ == "__main__":
    df = load_data("master_data_set.csv", 829, 1024)
    print(df.head())

    # Constants & tuning
    mass = 82.6
    rho = 1.20
    Crr = 0.005
    Q = np.diag([0.25, 0.05])
    R = np.diag([100000])

    # Initial state
    v0 = float(df["Speed"].iloc[0])
    # ipdb.set_trace()

    x = np.array([[v0], [2.0]])  # [v, CdA, bias]
    P = np.diag([0, 0])

    # Output
    # ipdb.set_trace()
    times, v_est, CdA_est, bias_est = [df["time"][0]], [x[0]], [0], [x[1]]
    dt = df["time"].iloc[1] - df["time"].iloc[0]

    # EKF loop
    for k in range(1, len(df)):
        # Time step and measurements
        # dt = df["time"].iloc[k]
        accel_meas = np.array([df["Y (m/s^2)"].iloc[k]])
        power_meas = np.array([df["power"].iloc[k]])
        wind_speed = np.array([df["Wind (m/s)"].iloc[k]])

        # Calculate slope in radians
        alt, alt0 = df["enhanced_altitude"].iloc[k], df["enhanced_altitude"].iloc[k - 1]
        dist, dist0 = df["distance"].iloc[k], df["distance"].iloc[k - 1]
        slope_rad = np.arctan((alt - alt0) / max(dist - dist0, 0.5))

        # import ipdb; ipdb.set_trace()

        x, P, _ = ekf_cda_step(
            slope_rad,
            wind_speed,
            dt,
            mass,
            rho,
            Crr,
            Q,
            R,
            x,
            P,
            accel_meas,
            power_meas,
        )

        times.append(df["time"].iloc[k])
        v_est.append(x[0])
        # CdA_est.append(x[1])
        bias_est.append(x[1])

    # Build and show results
    df_est = pd.DataFrame(
        {
            "time": times,
            "v_est": v_est,
            # "CdA_est":    CdA_est,
            "accel_bias": bias_est,
        }
    )
    print(df_est.head())

    figure, axes = plt.subplots(2, 1, figsize=(10, 8))

    # plt.figure(figsize=(10, 4))
    axes[0].plot(np.array(times), np.array(df["enhanced_speed"]), label="Actual Speed")
    axes[0].plot(np.array(times), np.array(v_est), label="EKF Speed", color="red")
    axes[0].legend()
    axes[0].set_title("Velocity: EKF vs Measured")
    axes[0].grid()

    # plt.figure(figsize=(10,4))
    # plt.plot(np.array(times), np.array(CdA_est), color='purple')
    # plt.title("Estimated $C_dA$"); plt.ylabel("CdA (m²)"); plt.grid(); plt.show()

    # Plot acceleration input
    # plt.figure(figsize=(10, 4))
    # ipdb.set_trace()
    axes[1].plot(np.array(times), np.array(df["Y (m/s^2)"].iloc[0:]), label="Measured Acceleration", color="orange")
    axes[1].plot(np.array(times), np.array(bias_est), label="Estimated Bias", color="green")
    axes[1].legend()
    axes[1].set_title("Acceleration Bias Estimation")
    axes[1].grid()
    # axes[1].show()

    # 

    plt.show()
