import numpy as np
import os
import matplotlib.pyplot as plt

def calculate_bcg_sn_with_iqr(path, start, end):
    """
    Calculate sliding window RMS of BCG signal from target file (index=48),
    filter outliers using IQR method, and compute median of remaining segments.

    Parameters:
        path (str): Directory containing BCG CSV files
        start (int): Start index of data extraction (inclusive)
        end (int): End index of data extraction (inclusive)

    Returns:
        float: Median value of filtered RMS segments

    Processing Steps:
        1. Locate target file (49th file in sorted directory list)
        2. Extract specified data segment from CSV file
        3. Compute sliding window RMS with fixed window size (200 samples)
        4. Normalize RMS values to [0, 1] range
        5. Remove outliers using IQR method (Q1-1.5IQR to Q3+1.5IQR)
        6. Calculate median of remaining valid segments
        7. Visualize distribution with box plot and statistical summary
    """
    # Get sorted list of CSV files in target directory
    csv_files = sorted([f for f in os.listdir(path) if f.endswith('.csv')])

    # Select 49th file (index=48) and construct full path
    target_file = csv_files[48]
    file_path = os.path.join(path, target_file)
    print(f"Processing file: {file_path}")

    # Read target data segment (column 1) from CSV file
    try:
        data = read_numpy_from_CSV_data(
            path=file_path,
            begin=start,
            end=end,
            column=1  # Second column (0-based index)
        )
    except Exception as e:
        raise RuntimeError(f"Data reading failed: {str(e)}")

    # Validate data length against window size
    window_size = 200
    data_length = len(data)
    if data_length < window_size:
        raise ValueError(
            f"Data length ({data_length}) is less than window size ({window_size}), "
            "cannot compute sliding window RMS"
        )

    # Calculate sliding window RMS values
    sn_values = []
    for i in range(data_length - window_size + 1):
        window_data = data[i:i + window_size]
        sum_square = np.sum(np.square(window_data))
        sn = np.sqrt(sum_square / window_size)  # RMS calculation
        sn_values.append(sn)

    # Normalize RMS values to [0, 1] range
    sn_array = np.array(sn_values)
    min_sn, max_sn = np.min(sn_array), np.max(sn_array)

    if max_sn - min_sn != 0:
        normalized_sn = (sn_array - min_sn) / (max_sn - min_sn)
    else:
        normalized_sn = np.zeros_like(sn_array)
        print("Warning: All RMS values are identical, normalization result is all zeros")

    # Outlier filtering using IQR method
    q1 = np.percentile(normalized_sn, 25)  # 1st quartile
    q3 = np.percentile(normalized_sn, 75)  # 3rd quartile
    iqr = q3 - q1  # Interquartile range
    lower_bound = q1 - 1.5 * iqr  # Lower bound for valid range
    upper_bound = q3 + 1.5 * iqr  # Upper bound for valid range

    # Filter valid segments within IQR range
    filtered_sn = normalized_sn[
        (normalized_sn >= lower_bound) & (normalized_sn <= upper_bound)
        ]

    # Calculate median of filtered segments
    filtered_median = np.median(filtered_sn)

    # Print statistical summary
    print(f"Original number of RMS segments: {len(normalized_sn)}")
    print(f"Number of outliers removed: {len(normalized_sn) - len(filtered_sn)}")
    print(f"IQR valid range: [{lower_bound:.4f}, {upper_bound:.4f}]")
    print(f"Median of filtered segments: {filtered_median:.4f}")

    # Visualize RMS distribution after filtering
    plt.figure(figsize=(12, 6))

    # Plot box plot with custom styling
    bp = plt.boxplot(
        normalized_sn,
        patch_artist=True,
        boxprops=dict(facecolor='#8ecae6', color='#219ebc'),
        capprops=dict(color='#023047'),
        whiskerprops=dict(color='#023047'),
        flierprops=dict(marker='o', color='#fb8500', alpha=0.5),
        medianprops=dict(color='#ffb703', linewidth=2)
    )

    # Highlight filtered median with reference line
    plt.axhline(
        y=filtered_median,
        color='#e63946',
        linestyle='--',
        label=f'Filtered median: {filtered_median:.4f}'
    )

    plt.title('BCG Sliding Window RMS Distribution (Post-IQR Filtering)', fontsize=12)
    plt.ylabel('Normalized RMS Value', fontsize=10)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    return filtered_median


def rms_filter(signal, window_size=200):
    """
    Calculate Root Mean Square (RMS) sequence using sliding symmetric window (loop implementation).

    Parameters:
        signal (np.ndarray): Input 1D signal array
        window_size (int): Number of samples in the analysis window (must be odd for symmetric window)

    Returns:
        np.ndarray: Normalized RMS sequence in range [0, 1]

    Processing Steps:
        1. Window configuration: Initialize symmetric analysis window
        2. Sliding computation: Iterate through signal with partial window handling
        3. Edge case management: Adaptive window adjustment at signal boundaries
        4. Energy normalization: Global max scaling to [0,1] range
    """
    rms_signal = np.zeros(signal.shape[0])
    for i in range(signal.shape[0]):
        # Dynamic window boundary calculations
        start = max(0, i - window_size // 2)
        end = min(signal.shape[0], i + window_size // 2)
        signal_mean = signal[start:end]
        # RMS energy calculations
        rms_signal[i] = np.sqrt(np.mean(signal_mean ** 2))
    # Global energy normalization
    return rms_signal / np.max(rms_signal)


def discard_near_threshold(rms, thresholds, discard_seconds=15, fs=125):
    """
    Discard RMS points near threshold-exceeding regions with specified duration.

    Parameters:
        rms (np.ndarray): Normalized RMS sequence in range [0, 1]
        thresholds (tuple): Lower and upper bounds (low, high) for valid RMS range
        discard_seconds (int): Time duration around threshold crossings to discard (default: 15)
        fs (int): Sampling frequency in Hz (default: 125)

    Returns:
        np.ndarray: Boolean mask indicating discarded RMS points (True = discarded)

    Processing Steps:
        1. Identifies areas where energy is exceeded (below low or above high).
        2. expand the exclusion window in both directions centered on the point of exceeding the limit.
        3. Adjust the window boundaries to prevent the index from crossing the boundaries
        4. merge all consecutive intervals to be excluded
    """
    low, high = thresholds
    exceed_low = rms < low
    exceed_high = rms > high
    exceed_mask = exceed_low | exceed_high
    discard_points = discard_seconds * fs

    # Initialize discard mask
    discard_mask = np.zeros_like(rms, dtype=bool)

    # Mark discard regions around threshold crossings
    for i in np.where(exceed_mask)[0]:
        start = max(0, i - discard_points)
        end = min(len(rms), i + discard_points + 1)
        discard_mask[start:end] = True

    return discard_mask

