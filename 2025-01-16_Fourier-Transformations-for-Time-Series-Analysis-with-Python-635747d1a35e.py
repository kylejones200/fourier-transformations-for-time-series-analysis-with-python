# Description: Short example for Fourier Transformations for Time Series Analysis with Python.



# Generate a time series with two frequencies

from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)

time = np.linspace(0, 10, 500)  # 500 time points
freq1, freq2 = 2, 5  # Frequencies in Hz
signal = np.sin(2 * np.pi * freq1 * time) + 0.5 * np.sin(2 * np.pi * freq2 * time)

# Plot the time series
plt.figure(figsize=(10, 4))
plt.plot(time, signal)
plt.title("Time Series (Combination of Two Sine Waves)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

# Apply Fourier Transform
fft_result = np.fft.fft(signal)

# Frequency axis
frequencies = np.fft.fftfreq(len(fft_result), d=(time[1] - time[0]))  

# Plot the magnitude spectrum
plt.figure(figsize=(10, 4))
plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_result)[:len(fft_result)//2])
plt.title("Frequency Domain Representation")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.show()

#from sktime.datasets import load_airline

# Load the Airline Passengers Dataset
y = load_airline()
y = y.values

# Create time array
time = np.arange(len(y))

# Apply Fourier Transform
fft_result = np.fft.fft(y)
frequencies = np.fft.fftfreq(len(fft_result), d=1)  # Assume monthly data (d=1)

# Plot the original time series
plt.figure(figsize=(10, 4))
plt.plot(y)
plt.title("Original Airline Passenger Data")
plt.xlabel("Time")
plt.ylabel("Passengers")
plt.show()

# Plot the frequency domain
plt.figure(figsize=(10, 4))
plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_result)[:len(fft_result)//2])
plt.title("Frequency Domain of Airline Passenger Data")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.savefig('time_series_passanger.png')
plt.show()

# Add random noise to the signal
noisy_signal = y + np.random.normal(0, 50, len(y))

# Apply Fourier Transform to noisy signal
fft_result_noisy = np.fft.fft(noisy_signal)

# Filter out high frequencies
fft_filtered = fft_result_noisy.copy()
threshold = 0.1  # Adjust this threshold as needed
fft_filtered[np.abs(frequencies) > threshold] = 0

# Inverse FFT to get the filtered signal
filtered_signal = np.fft.ifft(fft_filtered)

# Plot the original, noisy, and filtered signals
plt.figure(figsize=(10, 6))
plt.plot(time, noisy_signal, label="Noisy Signal", alpha=0.5)
plt.plot(time, filtered_signal.real, label="Filtered Signal", color='red')
plt.title("Noise Filtering with Fourier Transform")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.savefig('time_series_passanger_amp.png')
plt.show()


# Perform FFT with SciPy (uses FFTW internally for optimization)
fft_result = fft(signal)
ifft_result = ifft(fft_result)
