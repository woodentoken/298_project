import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_data(file_path, start_row, end_row):
    df = pd.read_csv(file_path)
    df_new = df.iloc[start_row-2:end_row-1]  # Adjust for 0-indexing
    return df_new

# EKF one-step: 
# state = [v, CdA, accel_bias] 
def ekf_cda_step(x, P, accel_meas, slope_rad, wind_speed, power_meas, dt, mass, rho, Crr, Q, R):
    """
    x           : [v, CdA, bias] State vector
    P           : 3×3 covariance
    accel_meas  : measured longitudinal accel (m/s²)
    slope_rad   : road pitch (rad)
    wind_speed  : air-relative wind speed (m/s)
    power_meas  : power meter (W)
    dt          : time step (s)
    mass, rho   : constants
    Crr         : rolling-resistance coefficient
    Q           : 3×3 process noise
    R           : scalar power-meas noise variance
    """
    g = 9.80665
    v, CdA, bias = x

    # Prediction
    accel_true = accel_meas + bias
    v_pred     = v + accel_true * dt
    CdA_pred   = CdA
    bias_pred  = bias     

    x_pred = np.array([v_pred, CdA_pred, bias_pred])

    # Jacobian F = df/dx (3×3)
    F = np.array([
        [1.0, 0.0, dt],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    P_pred = F @ P @ F.T + Q

    # Measurement update
    F_aero = 0.5 * rho * CdA_pred * wind_speed**2
    F_grav = mass * g * np.sin(slope_rad)
    F_roll = mass * g * Crr * np.cos(slope_rad)
    power_pred = v_pred * (F_aero + F_grav + F_roll + mass * accel_true)
    y = power_meas - power_pred

    # Measurement Jocobian H = dh/dx  (1×3)
    dPdv   = (F_aero + F_grav + F_roll + mass * accel_true)
    dPdCdA = 0.5 * rho * v_pred * wind_speed**2
    dPdb   = mass * v_pred
    H = np.array([[dPdv, dPdCdA, dPdb]])

    # 4) KALMAN GAIN & UPDATE
    S     = H @ P_pred @ H.T + R         
    K     = (P_pred @ H.T) / S           
    x_new = x_pred + (K.flatten() * y)
    P_new = (np.eye(3) - K @ H) @ P_pred

    return x_new, P_new, power_pred

if __name__ == "__main__":
    df = load_data("master_data_set.csv", 829, 1024)
    print(df.head())

    # Constants & tuning
    mass      = 82.6
    rho       = 1.20
    Crr       = 0.005
    Q         = np.diag([0.005, 0.005, 0.005])  
    R         = 50                               

    # Initial state
    v0   = float(df["Speed"].iloc[0])
    x    = np.array([v0, 0.30, 0.0])      # [v, CdA, bias]
    P    = np.diag([0, 0, 0])

    # Output
    times, v_est, CdA_est, bias_est = [], [], [], []

    # EKF loop
    for k in range(1, len(df)):
        # Time step and measurements
        dt         = df["time"].iat[k] - df["time"].iat[k-1]
        accel_meas = df["Y (m/s^2)"].iloc[k]
        power_meas = df["power"].iloc[k]
        wind_speed = df["Wind (m/s)"].iloc[k]

        # Calculate slope in radians
        alt, alt0     = df["enhanced_altitude"].iloc[k], df["enhanced_altitude"].iloc[k-1]
        dist, dist0   = df["distance"].iloc[k], df["distance"].iloc[k-1]
        slope_rad     = np.arctan((alt-alt0)/max(dist-dist0, 0.5))

        x, P, _ = ekf_cda_step(x, P,
                              accel_meas, slope_rad, wind_speed, power_meas,
                              dt, mass, rho, Crr, Q, R)

        times.append(df["absolute_time"].iloc[k])
        v_est.append(x[0])
        CdA_est.append(x[1])
        bias_est.append(x[2])

    # Build and show results
    df_est = pd.DataFrame({
        "time":       times,
        "v_est":      v_est,
        "CdA_est":    CdA_est,
        "accel_bias": bias_est
    })
    print(df_est.head())

    plt.figure(figsize=(10,4))
    plt.plot(np.array(df["absolute_time"]), np.array(df["enhanced_speed"]), label="Actual Speed")
    plt.plot(np.array(times), np.array(v_est), label="EKF Speed", color='red')
    plt.legend(); plt.title("Velocity: EKF vs Measured"); plt.grid(); plt.show()

    plt.figure(figsize=(10,4))
    plt.plot(np.array(times), np.array(CdA_est), color='purple')
    plt.title("Estimated $C_dA$"); plt.ylabel("CdA (m²)"); plt.grid(); plt.show()


    # Plot acceleration input
    plt.figure(figsize=(10,4))
    plt.plot(np.array(times), np.array(df["Y (m/s^2)"].iloc[1:]), label="Measured Acceleration", color='orange')
    plt.plot(np.array(times), np.array(bias_est), label="Estimated Bias", color='green')
    plt.legend(); plt.title("Acceleration Bias Estimation"); plt.grid(); plt.show()
