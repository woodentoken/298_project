import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ipdb
import scipy

# Constants & tuning
MASS = 82.6
RHO = 1.20
CRR = 0.005
G = 9.80665  # m/s², acceleration due to gravity
Q = np.diag([0.1, 0.1])
R = np.diag([100000])
USE_DERIVED_ACCELERATION = False  # Set to True to use derived acceleration, False to use measured acceleration

def main():
    if USE_DERIVED_ACCELERATION:
        print("Using derived acceleration for EKF, NOT the measured acceleration.")

    df = load_data("master_data_set.csv", 829, 1024)
    print(df.head())
    plot_power_constituents(df)

    # Initial state
    length = len(df["time"])

    x = np.ndarray((length, 2, 1))
    P = np.ndarray((length, 2, 2))

    x[0, 0] = float(df["Speed"].iloc[0])  # initial velocity
    x[0, 1] = 0.0  # initial bias

    P[0, :, :] = np.diag([0.0, 0.0])  # initial covariance

    # Output
    residuals1 = np.ndarray((length))  # Initialize residuals
    residuals1[0] = 0.0  # Initial residual

    residuals2 = np.ndarray((length))  # Initialize residuals
    residuals2[0] = 0.0  # Initial residual

    power_pred_true = np.ndarray((length))  # Initialize residuals
    power_pred_true[0] = 0.0  # Initial residual

    power_pred_est = np.ndarray((length))  # Initialize residuals
    power_pred_est[0] = 0.0  # Initial residual

    kalman_gains = np.ndarray((length, 2, 1))  # Initialize Kalman gains
    kalman_gains[0, :, :] = np.array([[0.0], [0.0]])  # Initial Kalman gain

    dt = df["time"].iloc[1] - df["time"].iloc[0]
    # Time step and measurements
    accel_meas = df["Y (m/s^2)"]
    vel_meas = df["enhanced_speed"]
    power_meas = df["power"]
    wind_speed = df["Wind (m/s)"]

    # Calculate slope in radians
    alt = df["enhanced_altitude"]
    dist = df["distance"]
    slope_rad = compute_slope(altitude=alt, distance=dist)
    df["derived_acceleration"] = df["Speed"].diff() / df["time"].diff()
    df["derived_acceleration"][0] = 0.0

    if USE_DERIVED_ACCELERATION:
        accel_meas = df["derived_acceleration"]
    else:  # Use the measured acceleration
        accel_meas = df["Y (m/s^2)"]

    # EKF loop
    for k in range(1, length):
        x[k], P[k], power_pred_true[k], power_pred_est[k], residuals1[k], residuals2[k], kalman_gains[k] = ekf_cda_step(
            slope_rad[k],
            wind_speed[k],
            dt,
            x[k - 1],
            P[k - 1],
            accel_meas[k - 1],
            power_meas[k - 1],
            vel_meas[k - 1],
            step=k,
        )

    ekf_df = pd.DataFrame(
        {
            "t": df["time"],
            "v_est": x[:, 0, 0],
            # "CdA_est": x[:, 1, 0],
            "accel_bias": x[:, 1, 0],
            "residuals_est": residuals2,
            "residuals_true": residuals1,
            "power_pred_true": power_pred_true,
            "power_pred_est": power_pred_est,
            "kalman_gains_1": kalman_gains[:, 0, 0],
            "kalman_gains_2": kalman_gains[:, 1, 0],
            "slope_deg": slope_rad * 180 / np.pi,
            "wind_speed": wind_speed,
        }
    )

    plotting(df, ekf_df)


def compute_slope(altitude, distance):
    """
    Compute the slope in radians given altitude and distance.
    """
    altitude[-1:] = 0  # Ensure the last altitude is not NaN
    if len(altitude) != len(distance):
        raise ValueError("Altitude and distance arrays must have the same length.")

    slope_rad = np.arctan(np.diff(altitude) / np.maximum(np.diff(distance), 0.5))
    slope_rad[-1:] = 0.0  # Ensure the last slope is not NaN
    return np.insert(slope_rad, 0, 0.0)  # Insert initial slope as 0


def load_data(file_path, start_row, end_row):
    df = pd.read_csv(file_path)
    df_new = df.iloc[start_row - 2 : end_row - 1]  # Adjust for 0-indexing
    return df_new.reset_index(drop=True)


def plot_power_constituents(df):
    slope_rad = compute_slope(df["enhanced_altitude"], df["distance"])  # Convert degrees to radians
    wind_speed = df["Wind (m/s)"]
    v_meas = df["enhanced_speed"]
    derived_acceleration = df["Speed"].diff() / df["time"].diff()
    derived_acceleration[0] = 0.0

    CdA_pred = 0.3  # Predicted drag area (m²)
    # Calculate forces
    F_aero = 0.5 * RHO * CdA_pred * wind_speed**2
    F_grav = MASS * G * np.sin(slope_rad)
    F_roll = MASS * G * CRR * np.cos(slope_rad)
    if USE_DERIVED_ACCELERATION:
        F_inertia = MASS * derived_acceleration  # Inertial force from acceleration
        accel_source = "derived"
    else:
        F_inertia = MASS * df["Y (m/s^2)"]
        accel_source = "measured"

    Total = F_aero + F_grav + F_roll + F_inertia
    Power = v_meas * Total

    figure, axes = plt.subplots(3, 1, figsize=(10, 12))
    axes[0].plot(df["time"], F_aero, label="Aero Drag Force")
    axes[0].plot(df["time"], F_grav, label="Gravitational Force")
    axes[0].plot(df["time"], F_roll, label="Rolling Resistance Force")
    axes[0].plot(df["time"], F_inertia, label=f"Inertial Force ({accel_source}) ")
    axes[0].plot(df["time"], Total, label="Total Force (predicted)", color="black")
    axes[0].set_title("Forces Acting on the Bicycle")
    axes[0].set_ylabel("Force (N)")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(df["time"], Power, label="Power (predicted)", color="black")
    axes[1].plot(df["time"], df["power"], label="Measured Power (power meter)", linestyle="--", color="black")
    axes[1].set_title("Power estimation from forces")
    axes[1].set_ylabel("Power (W)")
    axes[1].legend()
    axes[1].grid()

    axes[2].plot(df["time"], df["Y (m/s^2)"], label="Acceleration")
    axes[2].plot(
        df["time"], df["enhanced_speed"].diff() / df["time"].diff(), label="Estimated Acceleration", linestyle="--"
    )
    axes[2].set_title("Acceleration Input")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Acceleration (m/s²)")
    axes[2].legend()
    axes[2].grid()
    plt.show()


# EKF one-step:
def ekf_cda_step(slope_rad, wind_speed, dt, states, variance, input, output, v_meas, step):
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
    CdA_pred = 0.5

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
    F_aero = 0.5 * RHO * CdA_pred * wind_speed**2
    F_grav = MASS * G * np.sin(slope_rad)
    F_roll = MASS * G * CRR * np.cos(slope_rad)

    power_pred_true = (v_meas) * (F_aero + F_roll + F_grav + MASS * input)
    power_pred_est = (states[0]) * (F_aero + F_roll + F_grav + MASS * input)

    # Measurement Jacobian H = dh/dx  (1×3)
    dh_dv = F_aero + F_grav + F_roll + MASS * input
    # dh_dCdA = 0.5 * rho * x_pred[0] * wind_speed**2
    dh_dbias = MASS * x_pred[1][0]
    dh_du = MASS * x_pred[0][0]

    # linearize the measurement equation based on the state
    C_prime = np.array([dh_dv, 0]).reshape(1, 2)  # 1×3 Jacobian matrix
    D_prime = np.array([0.0])
    # noise is added in the measurement equation
    F_prime = np.array([1.0])

    # 4) KALMAN GAIN & UPDATE
    L = P_pred @ C_prime.T * np.linalg.inv(C_prime @ P_pred @ C_prime.T + F_prime @ R @ F_prime.T)
    # L = 0 * L  # | open loop prediction
    residual_true = output - power_pred_true
    residual_est = output - power_pred_est

    x_est = x_pred + L * residual_est
    P_est = P_pred - (L @ C_prime @ P_pred)

    return x_est, P_est, power_pred_true, power_pred_est, residual_true, residual_est, L


def plotting(df, ekf_df):
    t = df["time"]
    figure, axes = plt.subplots(3, 1, figsize=(10, 10))

    if USE_DERIVED_ACCELERATION:
        print("Using derived acceleration for plotting.")
        accel_source = "derived"
        accel_color = "magenta"
    else:
        print("Using measured acceleration for plotting.")
        accel_source = "measured"
        accel_color = "red"

    # plt.figure(figsize=(10, 4))
    axes[0].plot(t, df["enhanced_speed"], label="Actual Speed")
    axes[0].plot(t, ekf_df["v_est"], label=f"EKF Speed ({accel_source} acceleration)", color=accel_color)
    # axes[0].plot(np.array(t), ol_velocity, label="Bias Speed", color="red")
    axes[0].legend()
    axes[0].set_title("Velocity: EKF vs Measured")
    axes[0].grid()

    # Plot acceleration input
    # axes[1].plot(t, df["Y (m/s^2)"], label="Measured Acceleration", color="red")
    axes[1].plot(t, df["derived_acceleration"], label="Derived Acceleration", color="magenta")
    axes[1].plot(t, ekf_df["accel_bias"], label="Estimated Bias", color="green")
    axes[1].legend()
    axes[1].set_title("Acceleration Bias Estimation")
    axes[1].grid()

    # Plot residual
    axes[2].plot(t, ekf_df["power_pred_true"], label="Power (using measured velocity)", color="black")
    axes[2].plot(t, ekf_df["power_pred_est"], label="Estimated power (EKF)", color="red")
    axes[2].plot(t, df["power"], label="Power meter measurement", color="blue")
    axes[2].legend()
    axes[2].set_title("Residuals (y - h(x))")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid()
    plt.show()


if __name__ == "__main__":
    main()
