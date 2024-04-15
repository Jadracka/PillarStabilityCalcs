import csv
import pandas as pd
import math as m
import matplotlib.pyplot as plt
import numpy as np
import os

# New comment

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

   
def process_interferometer_data(csv_paths, zenithal_angles, D_mm, delta_angle, omega_angle, ksi_angle, max_time=None):
    """
    Process interferometer data from CSV files and apply mathematical corrections.

    Parameters
    ----------
    csv_paths : list of str
        List of paths to the CSV files containing interferometer data.

    Returns
    -------
    pandas.DataFrame
        Processed DataFrame containing time, raw measurement, horizontal distance,
        and other corrected data.
    """
    # Initialize lists to store data
    times = []
    raw_measurements = []

    # Read all CSV files simultaneously
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
            measurement1, measurement2, measurement3 = float(row1[1]), float(row2[1]), float(row3[1])

            # Check if the timing is roughly the same for all interferometers
            if abs(time1 - time2) < 0.000001 and abs(time1 - time3) < 0.000001 and (max_time is None or time1 <= max_time):
                times.append(time1)
                raw_measurements.append((measurement1, measurement2, measurement3))

    D = D_mm * 10**3
    omega = gon2rad(omega_angle)
    delta = gon2rad(delta_angle)
    ksi = gon2rad(ksi_angle)


    # Apply mathematical corrections and calculate cartesian coordinates
    horizontal_distances = []
    deltas_x = []
    deltas_y = []
    phis = []
    prev_measurements = [None, None, None]

    for i, (measurement1, measurement2, measurement3) in enumerate(raw_measurements):
        # Apply mathematical corrections
        horizontal_distance1 = measurement1 * m.cos(gon2rad(zenithal_angles[0]))
        horizontal_distance2 = measurement2 * m.cos(gon2rad(zenithal_angles[1]))
        horizontal_distance3 = measurement3 * m.cos(gon2rad(zenithal_angles[2]))

        # Calculate cartesian coordinates
        if None not in prev_measurements:
            delta_x = ((horizontal_distance1 * m.sin(delta) - prev_measurements[0] * m.sin(delta)) + (horizontal_distance2 * m.sin(m.tau - omega) - prev_measurements[1] * m.sin(m.tau - omega)))/2
            phi = ((horizontal_distance2  - prev_measurements[1] ) * m.sin(m.tau - omega) - ((horizontal_distance1 - prev_measurements[0]) * m.sin(delta)))/D
            delta_y = m.sin(ksi) * (horizontal_distance3 - prev_measurements[2]) - (D/2 - (D * m.cos(phi))/2)
        else:
            delta_x = 0
            phi = 0
            delta_y = 0

        # Append data to lists
        horizontal_distances.append((horizontal_distance1 * 10**6, horizontal_distance2 * 10**6, horizontal_distance3 * 10**6))
        deltas_x.append(delta_x * 10**6)
        deltas_y.append(delta_y * 10**6)
        phis.append(phi * 10**6)

        # Update previous measurements
        prev_measurements = (horizontal_distance1, horizontal_distance2, horizontal_distance3)

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

    return df

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

    # Plot the frequency distribution
    plt.figure(figsize=(10, 6))
    plt.plot(FreqAxis, np.abs(FreqDist), label='Absolute Frequency Distribution')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Absolute Frequency Distribution')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Fine grid
    plt.legend()
    plt.savefig('plot1_absolute_frequency_distribution_plot.png')
    plt.show()

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
    plt.hist(df['Delta X [um]'].values, bins=1500, alpha=1, label='Delta X', range=(min_x/5, max_x/5))
    plt.hist(df['Delta Y [um]'].values, bins=1500, alpha=0.5, label='Delta Y', range=(min_y/5, max_y/5))
    plt.xlabel('Change')
    plt.ylabel('Frequency')
    plt.title('Distribution of Delta X and Y')
    plt.legend()
    plt.savefig('plot3_delta_distribution.png')
    plt.close()



def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = "Data"
    # Paths to interferometer CSV files
    interferometer1_csv = '01_Channel_1.csv'
    interferometer2_csv = '01_Channel_2.csv'
    interferometer3_csv = '01_Channel_3.csv'

    IFM_files = [os.path.join(current_dir, data_dir, interferometer1_csv), 
                 os.path.join(current_dir, data_dir, interferometer2_csv), 
                 os.path.join(current_dir, data_dir, interferometer3_csv)]

    # Zenith angles for each interferometer in gons
    IFM_zenithal_angles = [102.8495, 102.6638, 103.6139]

    # Angles for corrections in gons
    delta_angle = 100.3974
    omega_angle = -99.6872
    ksi_angle = -11.4865

    # Value of D in millimeters
    D_value = 173.588

    # Value of D in millimeters
    max_time = 2500

    # Process interferometer data
    dfs = process_interferometer_data(IFM_files,IFM_zenithal_angles, D_value, delta_angle, omega_angle, ksi_angle, max_time)

    analyze_data(dfs)

if __name__ == "__main__":
    main()
