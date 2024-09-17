import csv
import pandas as pd
import math as m
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import welch, butter, filtfilt

def gon2rad(gon):
    """
    Convert an angle from gons to radians.

    Parameters
    ----------
    gon : float
        Angle in gons.

    Returns
    -------
    float
        Angle in radians.
    """
    return gon * (m.pi / 200)

def process_interferometer_data(csv_paths, zenithal_angles, D_mm, delta_angle, omega_angle, ksi_angle, channel_mapping, max_time=None, min_time=None):
    """
    Process interferometer data from CSV files and apply mathematical corrections.

    Parameters
    ----------
    csv_paths : list of str
        List of paths to the CSV files containing interferometer data.
    zenithal_angles : list of float
        Zenithal angles for each interferometer in gons.
    D_mm : float
        Distance value in millimeters.
    delta_angle : float
        Correction angle delta in gons.
    omega_angle : float
        Correction angle omega in gons.
    ksi_angle : float
        Correction angle ksi in gons.
    channel_mapping : dict
        A dictionary defining the mapping from channels to horizontal distances.
    max_time : float, optional
        Maximum time in seconds to consider for the analysis.
    min_time : float, optional
        Minimum time in seconds to consider for the analysis (default is 7200).

    Returns
    -------
    pandas.DataFrame
        Processed DataFrame containing time, raw measurement, horizontal distance,
        and other corrected data.
    dict
        Dictionary containing the maximum values for time and each measurement.
    """
    # Initialize lists to store data
    times = []
    raw_measurements = []

    # Initialize variables to track maximum values
    max_time_val = float('-inf')
    max_measurement1 = float('-inf')
    max_measurement2 = float('-inf')
    max_measurement3 = float('-inf')

    if len(csv_paths) == 1:
        # Reading from a single file with all data combined
        with open(csv_paths[0], 'r') as file:
            reader = csv.reader(file, delimiter=';')
            headers = next(reader)  # Skip header

            for row in reader:
                # Extract time and raw measurement from each channel using the mapping
                time = float(row[0])
                measurement1 = float(row[channel_mapping['Horizontal Distance 1']])  # Use mapping for Horizontal Distance 1
                measurement2 = float(row[channel_mapping['Horizontal Distance 2']])  # Use mapping for Horizontal Distance 2
                measurement3 = float(row[channel_mapping['Horizontal Distance 3']])  # Use mapping for Horizontal Distance 3

                # Check if the time is within the specified range
                if (min_time is None or time >= min_time) and (max_time is None or time <= max_time):
                    # Update maximum values
                    max_time_val = max(max_time_val, time)
                    max_measurement1 = max(max_measurement1, measurement1)
                    max_measurement2 = max(max_measurement2, measurement2)
                    max_measurement3 = max(max_measurement3, measurement3)

                    times.append(time)
                    raw_measurements.append((measurement1, measurement2, measurement3))
    else:
        # Reading from three separate files
        with open(csv_paths[0], 'r') as file1, \
             open(csv_paths[1], 'r') as file2, \
             open(csv_paths[2], 'r') as file3:

            reader1 = csv.reader(file1, delimiter=';')
            reader2 = csv.reader(file2, delimiter=';')
            reader3 = csv.reader(file3, delimiter=';')

            header1 = next(reader1)  # Skip header
            header2 = next(reader2)
            header3 = next(reader3)

            for row1, row2, row3 in zip(reader1, reader2, reader3):
                # Extract time and raw measurement from each interferometer
                time1, time2, time3 = float(row1[0]), float(row2[0]), float(row3[0])
                measurement1 = float(row1[channel_mapping['Horizontal Distance 1']])
                measurement2 = float(row2[channel_mapping['Horizontal Distance 2']])
                measurement3 = float(row3[channel_mapping['Horizontal Distance 3']])

                # Check if the time is within the specified range
                if all((min_time is None or t >= min_time) and (max_time is None or t <= max_time) for t in (time1, time2, time3)):
                    # Update maximum values
                    max_time_val = max(max_time_val, time1, time2, time3)
                    max_measurement1 = max(max_measurement1, measurement1)
                    max_measurement2 = max(max_measurement2, measurement2)
                    max_measurement3 = max(max_measurement3, measurement3)

                    # Check if the timing is roughly the same for all interferometers
                    if abs(time1 - time2) < 0.000001 and abs(time1 - time3) < 0.000001:
                        times.append(time1)
                        raw_measurements.append((measurement1, measurement2, measurement3))

    # Conversion and correction parameters
    D = D_mm * 10**3
    omega = gon2rad(omega_angle)
    delta = gon2rad(delta_angle)
    ksi = gon2rad(ksi_angle)

    # Initialize lists for corrected data
    horizontal_distances = []
    deltas_x = []
    deltas_y = []
    phis = []
    baseline_measurement = [None, None, None]

    # Process each measurement
    for i, (measurement1, measurement2, measurement3) in enumerate(raw_measurements):
        # Apply mathematical corrections
        horizontal_distance1 = measurement1 * m.sin(gon2rad(zenithal_angles[0]))
        horizontal_distance2 = measurement2 * m.sin(gon2rad(zenithal_angles[1]))
        horizontal_distance3 = measurement3 * m.sin(gon2rad(zenithal_angles[2]))

        if None in baseline_measurement:
            baseline_measurement = (horizontal_distance1, horizontal_distance2, horizontal_distance3)

        # Calculate cartesian coordinates
        delta_x = ((horizontal_distance1 * m.sin(delta) - baseline_measurement[0] * m.sin(delta)) + 
                   (horizontal_distance2 * m.sin(m.tau - omega) - baseline_measurement[1] * m.sin(m.tau - omega))) / 2
        phi = ((horizontal_distance2 - baseline_measurement[1]) * m.sin(m.tau - omega) - 
               ((horizontal_distance1 - baseline_measurement[0]) * m.sin(delta))) / D
        delta_y = m.cos(ksi) * (horizontal_distance3 - baseline_measurement[2]) - (D / 2 - (D * m.cos(phi)) / 2)

        # Append data to lists
        horizontal_distances.append((horizontal_distance1 * 10**6, horizontal_distance2 * 10**6, horizontal_distance3 * 10**6))
        deltas_x.append(delta_x * 10**6)
        deltas_y.append(delta_y * -10**6)
        phis.append(phi * 10**6)

    # Create a DataFrame from the lists
    df = pd.DataFrame({
        'Time [s]': times,
        'Horizontal Distance 1 [um]': [h[0] for h in horizontal_distances],
        'Horizontal Distance 2 [um]': [h[1] for h in horizontal_distances],
        'Horizontal Distance 3 [um]': [h[2] for h in horizontal_distances],
        'Delta X [um]': deltas_x,
        'Delta Y [um]': deltas_y,
        'Phi [uRad]': phis
    })

    # Dictionary of maximum values
    max_values = {
        'Max Time': max_time_val,
        'Max Channel 1 - Position': max_measurement1,
        'Max Channel 2 - Position': max_measurement2,
        'Max Channel 3 - Position': max_measurement3
    }

    return df, max_values

def analyze_data(df):
    # Find maximums and minimums in X, Y, and Phi
    max_x = df['Delta X [um]'].max()
    min_x = df['Delta X [um]'].min()
    max_y = df['Delta Y [um]'].max()
    min_y = df['Delta Y [um]'].min()
    max_phi = df['Phi [uRad]'].max()
    min_phi = df['Phi [uRad]'].min()

    peak_to_valye_x = max_x - min_x
    peak_to_valye_y = max_y - min_y

    # Find corresponding times for maximums and minimums
    time_max_x = df.loc[df['Delta X [um]'].idxmax(), 'Time [s]']
    time_min_x = df.loc[df['Delta X [um]'].idxmin(), 'Time [s]']
    time_max_y = df.loc[df['Delta Y [um]'].idxmax(), 'Time [s]']
    time_min_y = df.loc[df['Delta Y [um]'].idxmin(), 'Time [s]']
    time_max_phi = df.loc[df['Phi [uRad]'].idxmax(), 'Time [s]']
    time_min_phi = df.loc[df['Phi [uRad]'].idxmin(), 'Time [s]']

    print("Maximums and Minimums:")
    print(f"Max Delta X [um]: {max_x} at Time: {time_max_x}")
    print(f"Min Delta X [um]: {min_x} at Time: {time_min_x}")
    print(f"Max Delta Y [um]: {max_y} at Time: {time_max_y}")
    print(f"Min Delta Y [um]: {min_y} at Time: {time_min_y}")
    print(f"Max Phi [uRad]: {max_phi} at Time: {time_max_phi}")
    print(f"Min Phi [uRad]: {min_phi} at Time: {time_min_phi}")
    print(f"Peak to valey X [um]: {peak_to_valye_x}")
    print(f"Peak to valey Y [um]: {peak_to_valye_y}")

    # Perform Fast Fourier Transform (FFT)
    x = df['Delta X [um]'].values
    y = df['Delta Y [um]'].values
    Fs = 610.351562
    # Complex values from x and y arrays
    Complex = [x[i] + 1j * y[i] for i in range(len(x))]

    # Sampling frequency (Fs) in Hz
    Fs = 610.35

    # Perform FFT
    FreqDist = np.fft.fft(Complex).real

    # Take only the positive half of the frequencies
    FreqDist = FreqDist[:len(FreqDist)//2]

    # Frequency axis
    FreqAxis = [i * 0.5 * Fs / len(FreqDist) for i in range(len(FreqDist))]

    #fft_x = np.fft.rfft(df['Delta X [um]'])
    #fft_y = np.fft.rfft(df['Delta Y [um]'])

    """     # Plot FFT results
    plt.figure(figsize=(10, 6))
    plt.plot(np.abs(fft_x), label='Delta X')
    plt.plot(np.abs(fft_y), label='Delta Y')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title('FFT of Delta X and Delta Y')
    plt.legend()
    plt.savefig('plot1_fft_delta_x_and_delta_y.png')
    plt.close() """

    # Plot the frequency distribution with a logarithmic scale on the y-axis
    plt.figure(figsize=(10, 6))
    plt.semilogy(FreqAxis[2:], np.abs(FreqDist)[2:], label='Absolute Frequency Distribution')  # Log scale on y-axis
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (log scale)')
    plt.title('Absolute Frequency Distribution with Logarithmic Y-Axis')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Fine grid
    plt.legend()
    plt.savefig('plot1_absolute_frequency_distribution_plot_logy.png')
    #plt.show()

    # Plot changes over time
    plt.figure(figsize=(10, 6))
    plt.plot(df['Time [s]'].values, df['Delta X [um]'].values, label='Delta X')
    plt.plot(df['Time [s]'].values, df['Delta Y [um]'].values, label='Delta Y')
    plt.xlabel('Time [s]')
    plt.ylabel('Change')
    plt.title('Changes Over Time')
    plt.legend()
    plt.savefig('plot2_changes_over_time.png')
    plt.close()

    # Bin the sizes of delta X and Y and plot their distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['Delta X [um]'].values, bins=1500, alpha=1, label='Delta X')#, range=(min_x/5, max_x/5))
    plt.hist(df['Delta Y [um]'].values, bins=1500, alpha=0.5, label='Delta Y')#, range=(min_y/5, max_y/5))
    plt.xlabel('Change')
    plt.ylabel('Frequency')
    plt.title('Distribution of Delta X and Y')
    plt.legend()
    plt.savefig('plot3_delta_distribution.png')
    plt.close()

def main_old():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = "Data"
    # Paths to interferometer CSV files
    interferometer1_csv = '01_Channel_1.csv'
    interferometer2_csv = '01_Channel_2.csv'
    interferometer3_csv = '01_Channel_3.csv'

    IFM_files = [os.path.join(current_dir, data_dir, interferometer1_csv), 
                 os.path.join(current_dir, data_dir, interferometer2_csv), 
                 os.path.join(current_dir, data_dir, interferometer3_csv)]

    channel_mapping = {
    'Horizontal Distance 1': 1,  # Channel 2 for Horizontal Distance 1
    'Horizontal Distance 2': 2,  # Channel 3 for Horizontal Distance 2
    'Horizontal Distance 3': 3   # Channel 1 for Horizontal Distance 3
    }

    # Zenith angles for each interferometer in gons
    IFM_zenithal_angles = [102.8495, 102.6638, 103.6139]

    # Angles for corrections in gons
    delta_angle = 100.3974
    omega_angle = -99.6872
    ksi_angle = -11.4865

    # Value of D in millimeters
    D_value = 173.588

    # Maximum time in seconds
    max_time = 2690

    # Process interferometer data
    dfs = process_interferometer_data(IFM_files,IFM_zenithal_angles, D_value, delta_angle, omega_angle, ksi_angle, max_time)

    analyze_data(dfs)


        # Process interferometer data
    df, max_values = process_interferometer_data(
    IFM_files,
    IFM_zenithal_angles,
    D_value,
    delta_angle,
    omega_angle,
    ksi_angle,
    channel_mapping,
    max_time=10000,
    min_time=9000  # Default of 2 hours, adjust if needed
    )

    analyze_data(df)

def compute_psd(data, sampling_rate, column_name):
    """
    Compute the Power Spectral Density (PSD) of a data column.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame containing the time series data.
    sampling_rate : float
        The sampling rate in Hz (samples per second).
    column_name : str
        The column name for which to compute the PSD.

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    Pxx : ndarray
        Power spectral density of the data.
    """
    # Extract the data series for PSD calculation
    signal = data[column_name].values

    # Compute the PSD using Welch's method
    f, Pxx = welch(signal, fs=sampling_rate, nperseg=1024)

    return f, Pxx

def plot_psd(f, Pxx, column_name):
    """
    Plot the Power Spectral Density (PSD).

    Parameters
    ----------
    f : ndarray
        Array of sample frequencies.
    Pxx : ndarray
        Power spectral density of the data.
    column_name : str
        The column name for which the PSD is plotted.
    """
    plt.figure(figsize=(8, 6))
    plt.semilogy(f, Pxx)  # Logarithmic scale for the y-axis
    plt.title(f"Power Spectral Density (PSD) of {column_name}")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectral Density [V^2/Hz]')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    #plt.show()

def plot_psd_multiple(f1, Pxx1, f2, Pxx2, label1, label2):
    """
    Plot the Power Spectral Density (PSD) for two data columns.

    Parameters
    ----------
    f1 : ndarray
        Array of sample frequencies for the first data series.
    Pxx1 : ndarray
        Power spectral density of the first data series.
    f2 : ndarray
        Array of sample frequencies for the second data series.
    Pxx2 : ndarray
        Power spectral density of the second data series.
    label1 : str
        Label for the first data series.
    label2 : str
        Label for the second data series.
    """
    plt.figure(figsize=(8, 6))
    plt.semilogy(f1, Pxx1, label=label1)  # Logarithmic scale for the y-axis
    plt.semilogy(f2, Pxx2, label=label2)  # Logarithmic scale for the y-axis
    plt.title(f"Power Spectral Density (PSD)")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectral Density [V^2/Hz]')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig('plot4_PSD_XY.png')
    #plt.show()

def plot_psd_combined(f_x, Pxx_x, f_y, Pxx_y):
    """
    Plot the Power Spectral Density (PSD) for Delta X and Delta Y on the same plot.

    Parameters
    ----------
    f_x : ndarray
        Array of sample frequencies for Delta X.
    Pxx_x : ndarray
        Power spectral density of Delta X.
    f_y : ndarray
        Array of sample frequencies for Delta Y.
    Pxx_y : ndarray
        Power spectral density of Delta Y.
    """
    plt.figure(figsize=(8, 6))
    plt.semilogy(f_x, Pxx_x, label='Filtered Delta X [um]')  # Logarithmic scale for y-axis
    plt.semilogy(f_y, Pxx_y, label='Filtered Delta Y [um]')  # Logarithmic scale for y-axis
    plt.title("Power Spectral Density (PSD) of Filtered Delta X and Delta Y")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectral Density [V^2/Hz]')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig('plot5_PSD_HighFilter.png')
    plt.show()

def plot_horizontal_distance(data, time_column, distance_column, title):
    """
    Plot the horizontal distance over time.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame containing the time series data.
    time_column : str
        The column name for the time data.
    distance_column : str
        The column name for the horizontal distance data.
    title : str
        Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data[time_column].to_numpy(), data[distance_column].to_numpy(), label=distance_column)
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Horizontal Distance [um]')
    plt.grid(True)
    plt.legend()
    plt.savefig('plot6_horizontal_dist3_raw.png')
    plt.show()

def main():
    # Paths to interferometer CSV files
    IFM_files = ['V:/Projekte/PETRA4/Pillar stability Tests/06Aug24 Instrument Stand Prototype 0 - LT_Arm_Seismo/Channels_300.csv']

    # Zenith angles for each interferometer in gons
    IFM_zenithal_angles = [100.4608, 104.6191, 104.6092]

    # Angles for corrections in gons
    delta_angle = 107.5968
    omega_angle = -98.1402
    ksi_angle = -3.6112

    # Value of D in millimeters
    D_value = 144.0975

    channel_mapping = {
    'Horizontal Distance 1': 2,  # Channel 2 for Horizontal Distance 1
    'Horizontal Distance 2': 3,  # Channel 3 for Horizontal Distance 2
    'Horizontal Distance 3': 1   # Channel 1 for Horizontal Distance 3
    }

    # Process interferometer data
    df, max_values = process_interferometer_data(
    IFM_files,
    IFM_zenithal_angles,
    D_value,
    delta_angle,
    omega_angle,
    ksi_angle,
    channel_mapping,
    max_time=15400,
    #min_time=9000  # Default of 2 hours, adjust if needed
    )

    analyze_data(df)
    #print("Processed DataFrame (first few rows):")
    #print(df.head(10))
    #print("\nMaximum Values:")
    #print(max_values)

    # Plot raw data for "Horizontal Distance 3"
    #plot_horizontal_distance(df, 'Time [s]', 'Horizontal Distance 3 [um]', 'Raw Horizontal Distance 3 (Channel 1)')

if __name__ == "__main__":
    main()
