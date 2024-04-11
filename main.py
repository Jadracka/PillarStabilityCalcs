import csv
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np

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
    return gon * (math.pi / 200)

def process_interferometer_data(csv_path, zenith_angle_gon, delta_gon, omega_gon, ksi_gon, D_mm):
    """
    Process interferometer data from a CSV file and apply corrections into the cartesian coordinate system.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing interferometer data.
    zenith_angle_gon : float
        Zenith angle of the interferometer in gons.
    delta_gon : float
        Delta angle in gons.
    omega_gon : float
        Omega angle in gons.
    ksi_gon : float
        Ksi angle in gons.
    D_mm : float
        Value of D in millimeters.

    Returns
    -------
    pandas.DataFrame
        Processed DataFrame containing time, raw measurement, horizontal distance, cartesian coordinates, and Phi angle.
    """
    # Initialize lists to store time, raw measurement, horizontal distance, cartesian coordinates, and Phi angle
    times = []
    raw_measurements = []
    horizontal_distances = []
    cartesian_x = []
    cartesian_y = []
    phi_angles = []

    # Initialize variables for previous measurement
    prev_measurement = None

    # Read CSV file line by line
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header
        for row in reader:
            # Extract time and raw measurement from the row
            time = float(row[0])
            raw_measurement = float(row[1])

            # Apply mathematical correction
            horizontal_distance = raw_measurement * math.cos(gon2rad(zenith_angle_gon))

            # Calculate cartesian coordinates
            if prev_measurement is not None:
                delta_x = (((horizontal_distance - prev_measurement) * math.sin(gon2rad(delta_gon))) +
                           ((horizontal_distance - prev_measurement) * math.sin(gon2rad(omega_gon)))) / 2
                phi = ((horizontal_distance - prev_measurement) * math.sin(gon2rad(omega_gon)) - (horizontal_distance - prev_measurement) * math.sin(gon2rad(delta_gon))) / D_mm
                delta_y = math.sin(gon2rad(ksi_gon)) * (horizontal_distance - prev_measurement) - (D_mm / 2 - (D_mm * math.cos(phi)) / 2)
                x_coord = cartesian_x[-1] + delta_x
                y_coord = cartesian_y[-1] + delta_y
            else:
                x_coord = 0  # Initial X coordinate
                y_coord = 0  # Initial Y coordinate

            # Append data to lists
            times.append(time)
            raw_measurements.append(raw_measurement)
            horizontal_distances.append(horizontal_distance)
            cartesian_x.append(x_coord)
            cartesian_y.append(y_coord)
            phi_angles.append(phi)

            # Update previous measurement
            prev_measurement = horizontal_distance

    # Create a DataFrame from the lists
    df = pd.DataFrame({
        'Time [s]': times,
        'Raw Measurement [mm]': raw_measurements,
        'Horizontal Distance [mm]': horizontal_distances,
        'Cartesian X [mm]': cartesian_x,
        'Cartesian Y [mm]': cartesian_y,
        'Phi [rad]': phi_angles
    })

    return df

def analyze_data(df):
    # Find maximums and minimums in X, Y, and Phi
    max_x = df['Cartesian X [mm]'].max()
    min_x = df['Cartesian X [mm]'].min()
    max_y = df['Cartesian Y [mm]'].max()
    min_y = df['Cartesian Y [mm]'].min()
    max_phi = df['Phi [rad]'].max()
    min_phi = df['Phi [rad]'].min()

    # Find corresponding times for maximums and minimums
    time_max_x = df.loc[df['Cartesian X [mm]'].idxmax(), 'Time [s]']
    time_min_x = df.loc[df['Cartesian X [mm]'].idxmin(), 'Time [s]']
    time_max_y = df.loc[df['Cartesian Y [mm]'].idxmax(), 'Time [s]']
    time_min_y = df.loc[df['Cartesian Y [mm]'].idxmin(), 'Time [s]']
    time_max_phi = df.loc[df['Phi [rad]'].idxmax(), 'Time [s]']
    time_min_phi = df.loc[df['Phi [rad]'].idxmin(), 'Time [s]']

    print("Maximums and Minimums:")
    print(f"Max X: {max_x} at Time: {time_max_x}")
    print(f"Min X: {min_x} at Time: {time_min_x}")
    print(f"Max Y: {max_y} at Time: {time_max_y}")
    print(f"Min Y: {min_y} at Time: {time_min_y}")
    print(f"Max Phi: {max_phi} at Time: {time_max_phi}")
    print(f"Min Phi: {min_phi} at Time: {time_min_phi}")

    # Perform Fast Fourier Transform (FFT)
    fft_x = np.fft.fft(df['Cartesian X [mm]'])
    fft_y = np.fft.fft(df['Cartesian Y [mm]'])
    fft_phi = np.fft.fft(df['Phi [rad]'])

    # Plot FFT results
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(np.abs(fft_x))
    plt.title('FFT of X')
    plt.subplot(3, 1, 2)
    plt.plot(np.abs(fft_y))
    plt.title('FFT of Y')
    plt.subplot(3, 1, 3)
    plt.plot(np.abs(fft_phi))
    plt.title('FFT of Phi')
    plt.tight_layout()
    plt.show()

    # Plot changes over time
    plt.figure(figsize=(10, 6))
    plt.plot(df['Time [s]'], df['Cartesian X [mm]'], label='Delta X')
    plt.plot(df['Time [s]'], df['Cartesian Y [mm]'], label='Delta Y')
    plt.plot(df['Time [s]'], df['Phi [rad]'], label='Phi')
    plt.xlabel('Time [s]')
    plt.ylabel('Change')
    plt.title('Changes Over Time')
    plt.legend()
    plt.show()

    # Bin the sizes of delta X and Y and plot their distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['Cartesian X [mm]'], bins=50, alpha=0.5, label='Delta X')
    plt.hist(df['Cartesian Y [mm]'], bins=50, alpha=0.5, label='Delta Y')
    plt.xlabel('Change')
    plt.ylabel('Frequency')
    plt.title('Distribution of Delta X and Y')
    plt.legend()
    plt.show()

def main():
    # Paths to interferometer CSV files
    interferometer1_csv = 'interferometer1_data.csv'
    interferometer2_csv = 'interferometer2_data.csv'
    interferometer3_csv = 'interferometer3_data.csv'

    # Zenith angles for each interferometer in gons
    IFM1_zenithal_angle = 102.8495
    IFM2_zenithal_angle = 102.6638
    IFM3_zenithal_angle = 103.6139

    # Angles for corrections in gons
    delta_angle = 100.3974
    omega_angle = 99.6872
    ksi_angle = -11.4865

    # Value of D in millimeters
    D_value = 173.588

    # Process and correct interferometer data
    df1 = process_interferometer_data(interferometer1_csv, IFM1_zenithal_angle, delta_angle, omega_angle, ksi_angle, D_value)
    df2 = process_interferometer_data(interferometer2_csv, IFM2_zenithal_angle, delta_angle, omega_angle, ksi_angle, D_value)
    df3 = process_interferometer_data(interferometer3_csv, IFM3_zenithal_angle, delta_angle, omega_angle, ksi_angle, D_value)

    # Perform further analysis on the corrected dataframes
    analyze_data(df1)
    analyze_data(df2)
    analyze_data(df3)


if __name__ == "__main__":
    main()
