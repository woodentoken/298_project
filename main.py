from data_processing import post_process_data
from ukf_estimation import execute_ukf
from utils import freq_space, cmap, colors, norm

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
    # Example usage of the post_process_data function
    log_number_str = "001"
    test_date = "06_01_2025"
    stop_distance = 805  # meters - 0.5 mile x 1609.34 ~ 805 meters

    print(f"using frequency space: {freq_space}, edit utils.py to change it")

    # Post-process data
    post_process_data(test_date, stop_distance)

    # run the UKF with the specified parameters
    # will use the frequency space and colormap defined above
    execute_ukf(log_number_str=log_number_str, test_date=test_date, stop_distance=stop_distance)
