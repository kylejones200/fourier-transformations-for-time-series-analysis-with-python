#!/usr/bin/env python3
"""
Generated script to create Tufte-style visualizations
"""
import signalplot
import logging

logger = logging.getLogger(__name__)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set random seeds
np.random.seed(42)
try:
    import tensorflow as tf
    tf.random.set_seed(42)
except ImportError:
    tf = None
except Exception:
    tf = None

# Tufte-style configuration
signalplot.apply(font_family='serif')

images_dir = Path("images")
images_dir.mkdir(exist_ok=True)

# Update all savefig calls to use images_dir
original_savefig = plt.savefig

def savefig_tufte(filename, **kwargs):
    """Wrapper to save figures in images directory with Tufte style"""
    if not str(filename).startswith('/') and not str(filename).startswith('images/'):
        filename = images_dir / filename
    original_savefig(filename, **kwargs)
    logger.info(f"Saved: {filename}")

plt.savefig = savefig_tufte

# Code blocks from article

# Code block 1
import pywt


# Set style

# Load CO2 data
data_path = Path("../../geospatial/datasets/co2_OK.csv")
df = pd.read_csv(data_path)

co2_cols = [col for col in df.columns if col.isdigit()]

# Melt to long format
df_long = df.melt(
    id_vars=['State', 'MSN'],
    value_vars=co2_cols,
    var_name='Year',
    value_name='Value'
)

df_long['Year'] = pd.to_datetime(df_long['Year'], format='%Y')
df_long = df_long.sort_values('Year')

# Get total CO2 emissions
total_co2 = df_long[df_long['MSN'].str.contains('TOT|TCR', na=False)].copy()
total_co2 = total_co2.groupby('Year')['Value'].sum().reset_index()
total_co2 = total_co2[total_co2['Value'].notna() & (total_co2['Value'] > 0)]

ts = total_co2.set_index('Year')['Value']
ts = ts.interpolate(method='linear')  # Fill small gaps
ts = ts.sort_index()

logger.info(f"Time series length: {len(ts)}")
logger.info(f"Date range: {ts.index.min()} to {ts.index.max()}")
logger.info(f"Value range: {ts.min():.2f} to {ts.max():.2f} million metric tons")

# Visualize
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(ts.index, ts.values, linewidth=2, color='#1f77b4')
ax.set_title('Oklahoma CO2 Emissions (1960-2023)', fontsize=14, fontweight='bold')
ax.set_ylabel('CO2 Emissions (Million Metric Tons)', fontsize=11)
plt.tight_layout()
plt.savefig('co2_series.png', dpi=300, bbox_inches='tight')
plt.show()



# Code block 2
# Perform wavelet decomposition
wavelet = 'db4'  # Daubechies 4 wavelet - good balance of smoothness and localization
level = 5  # Decomposition level

coeffs = pywt.wavedec(ts.values, wavelet, level=level)
cA, cD = coeffs[0], coeffs[1:]  # Approximation and detail coefficients

logger.info(f"Wavelet: {wavelet}")
logger.info(f"Decomposition level: {level}")
logger.info(f"Approximation coefficients length: {len(cA)}")
logger.info(f"Number of detail levels: {len(cD)}")
for i, detail in enumerate(cD):
    logger.info(f"  Detail level {level-i}: {len(detail)} coefficients")

# Verify reconstruction
reconstructed = pywt.waverec(coeffs, wavelet)
reconstruction_error = np.mean(np.abs(ts.values - reconstructed[:len(ts)]))
logger.info(f"\nReconstruction error: {reconstruction_error:.6f}")



# Code block 3
# Visualize decomposition
fig, axes = plt.subplots(level + 2, 1, figsize=(14, 3*(level+2)))

axes[0].plot(ts.index, ts.values, 'b-', linewidth=2)
axes[0].set_title('Original Signal', fontweight='bold', fontsize=12)
axes[0].set_ylabel('CO2 Emissions', fontsize=10)
# Approximation (low-frequency trend)
# Upsample to match original length
cA_upsampled = pywt.upcoef('a', cA, wavelet, level=level, take=len(ts))
axes[1].plot(ts.index, cA_upsampled, 'r-', linewidth=2)
axes[1].set_title(f'Approximation (Level {level}) - Long-term Trend', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Trend', fontsize=10)
# Detail coefficients (high-frequency components)
for i, detail in enumerate(cD):
    detail_upsampled = pywt.upcoef('d', detail, wavelet, level=level-i, take=len(ts))
    axes[i + 2].plot(ts.index, detail_upsampled, 'g-', linewidth=1.5, alpha=0.8)
    axes[i + 2].set_title(f'Detail Level {level - i} - {2**(level-i)} year scale', 
                         fontweight='bold', fontsize=12)
    axes[i + 2].set_ylabel('Detail', fontsize=10)
plt.tight_layout()
plt.savefig('wavelet_decomposition.png', dpi=300, bbox_inches='tight')
plt.show()



# Code block 4
def wavelet_denoise(data, wavelet='db4', threshold_mode='soft', level=5):
    """
    Denoise signal using wavelet thresholding.
    
    Parameters:
    -----------
    data : array-like
        Input signal
    wavelet : str
        Wavelet type
    threshold_mode : str
        'soft' or 'hard' thresholding
    level : int
        Decomposition level
    
    Returns:
    --------
    denoised : ndarray
        Denoised signal
    """
    coeffs = pywt.wavedec(data, wavelet, level=level)
    
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Median absolute deviation
    threshold = sigma * np.sqrt(2 * np.log(len(data)))
    
    # Apply threshold to detail coefficients (keep approximation)
    coeffs_thresh = [coeffs[0]]  # Keep approximation
    for c in coeffs[1:]:  # Threshold detail coefficients
        coeffs_thresh.append(pywt.threshold(c, threshold, mode=threshold_mode))
    
    # Reconstruct
    denoised = pywt.waverec(coeffs_thresh, wavelet)
    return denoised[:len(data)]  # Trim to original length

denoised = wavelet_denoise(ts.values, wavelet='db4', threshold_mode='soft')

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(ts.index, ts.values, 'b-', alpha=0.7, label='Original', linewidth=1.5)
axes[0].plot(ts.index, denoised, 'r-', label='Denoised', linewidth=2)
axes[0].set_title('Wavelet Denoising: Original vs Denoised', fontsize=14, fontweight='bold')
axes[0].set_ylabel('CO2 Emissions', fontsize=11)
axes[0].legend(frameon=True, fancybox=True, shadow=True)
noise = ts.values - denoised
axes[1].plot(ts.index, noise, 'g-', linewidth=1.5)
axes[1].set_title('Removed Noise Component', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Noise', fontsize=11)
axes[1].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig('wavelet_denoising.png', dpi=300, bbox_inches='tight')
plt.show()

logger.info(f"Noise standard deviation: {np.std(noise):.4f}")
logger.info(f"Signal-to-noise ratio improvement: {np.std(ts.values)/np.std(noise):.2f}x")



# Code block 5
# Continuous Wavelet Transform for time-frequency analysis
scales = np.arange(1, 65)  # Different scales (1 to 64 years)
coefficients, frequencies = pywt.cwt(ts.values, scales, 'morl', 1.0)

fig, ax = plt.subplots(figsize=(14, 8))

im = ax.contourf(
    range(len(ts)), 
    scales, 
    np.abs(coefficients),
    levels=50,
    cmap='viridis',
    extend='both'
)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, label='Magnitude')

# Format x-axis with years
ax.set_xticks(np.arange(0, len(ts), 10))
ax.set_xticklabels([ts.index[i].year for i in range(0, len(ts), 10)], rotation=45)

ax.set_ylabel('Scale (Years)', fontsize=11)
ax.set_title('Continuous Wavelet Transform: Time-Frequency Representation', 
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('cwt_spectrogram.png', dpi=300, bbox_inches='tight')
plt.show()

# Find dominant scales
scale_power = np.sum(np.abs(coefficients), axis=1)
dominant_scale_idx = np.argmax(scale_power)
dominant_scale = scales[dominant_scale_idx]

logger.info(f"Dominant scale: {dominant_scale} years")
logger.info(f"Scale power: {scale_power[dominant_scale_idx]:.2f}")



# Code block 6
def detect_anomalies_wavelet(data, wavelet='db4', threshold_factor=3, level=5):
    """
    Detect anomalies using wavelet decomposition.
    
    Anomalies appear as large coefficients in detail levels.
    """
    coeffs = pywt.wavedec(data, wavelet, level=level)
    
    # Focus on detail coefficients (high-frequency anomalies)
    detail_coeffs = coeffs[1:]
    
    anomalies = np.zeros(len(data), dtype=bool)
    anomaly_scores = np.zeros(len(data))
    
    for i, detail in enumerate(detail_coeffs):
        if len(detail) > 0:
            # Use robust statistics (median, MAD)
            median_detail = np.median(np.abs(detail))
            mad = np.median(np.abs(np.abs(detail) - median_detail))
            threshold = median_detail + threshold_factor * (mad / 0.6745)
            
            # Find anomalies in this detail level
            detail_abs = np.abs(detail)
            anomaly_mask = detail_abs > threshold
            
            # Map back to original time indices
            scale_factor = len(data) // len(detail)
            for idx in np.where(anomaly_mask)[0]:
                start_idx = idx * scale_factor
                end_idx = min((idx + 1) * scale_factor, len(data))
                anomalies[start_idx:end_idx] = True
                anomaly_scores[start_idx:end_idx] += detail_abs[idx]
    
    return anomalies, anomaly_scores

# Detect anomalies
anomalies, scores = detect_anomalies_wavelet(ts.values, threshold_factor=2.5)

# Visualize
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Time series with anomalies
axes[0].plot(ts.index, ts.values, 'b-', linewidth=2, label='CO2 Emissions', alpha=0.7)
axes[0].scatter(ts.index[anomalies], ts.values[anomalies], 
          color='red', s=100, zorder=5, label='Anomalies', marker='x', linewidths=2)
axes[0].set_title('Anomaly Detection using Wavelets', fontsize=14, fontweight='bold')
axes[0].set_ylabel('CO2 Emissions', fontsize=11)
axes[0].legend(frameon=True, fancybox=True, shadow=True)
# Anomaly scores
axes[1].plot(ts.index, scores, 'g-', linewidth=1.5, alpha=0.7)
axes[1].axhline(np.percentile(scores[scores > 0], 95), color='red', 
                linestyle='--', linewidth=1, label='95th percentile threshold')
axes[1].set_title('Anomaly Scores', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Anomaly Score', fontsize=11)
axes[1].legend()
plt.tight_layout()
plt.savefig('wavelet_anomalies.png', dpi=300, bbox_inches='tight')
plt.show()

logger.info(f"Detected {anomalies.sum()} anomaly periods")
if anomalies.sum() > 0:
    anomaly_years = ts.index[anomalies].year.tolist()
    logger.info(f"Anomaly years: {anomaly_years[:10]}..." if len(anomaly_years) > 10 else f"Anomaly years: {anomaly_years}")



# Code block 7
from scipy.fft import fft, fftfreq

# Fourier Transform
fft_values = fft(ts.values)
fft_freq = fftfreq(len(ts), 1.0)  # Assuming yearly data

# Power spectral density
psd = np.abs(fft_values) ** 2

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Fourier spectrum
positive_freq_idx = fft_freq > 0
axes[0].plot(fft_freq[positive_freq_idx], psd[positive_freq_idx], 'b-', linewidth=2)
axes[0].set_title('Fourier Transform: Frequency Domain', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Frequency (cycles per year)', fontsize=11)
axes[0].set_ylabel('Power Spectral Density', fontsize=11)
# Find dominant frequencies
dominant_freq_idx = np.argmax(psd[positive_freq_idx])
dominant_freq = fft_freq[positive_freq_idx][dominant_freq_idx]
period = 1.0 / dominant_freq if dominant_freq > 0 else np.inf

logger.info(f"Dominant frequency: {dominant_freq:.4f} cycles/year")
logger.info(f"Corresponding period: {period:.2f} years")

# Wavelet scalogram (from CWT above)
im = axes[1].contourf(
    range(len(ts)), 
    scales, 
    np.abs(coefficients),
    levels=50,
    cmap='viridis',
    extend='both'
)
axes[1].set_title('Wavelet Transform: Time-Frequency Domain', fontsize=14, fontweight='bold')
axes[1].set_xticks(np.arange(0, len(ts), 10))
axes[1].set_xticklabels([ts.index[i].year for i in range(0, len(ts), 10)], rotation=45)
axes[1].set_ylabel('Scale (Years)', fontsize=11)
plt.colorbar(im, ax=axes[1], label='Magnitude')

plt.tight_layout()
plt.savefig('fourier_vs_wavelet.png', dpi=300, bbox_inches='tight')
plt.show()



# Code block 8
# Complete code for reproducibility
# All imports, data loading, wavelet analysis, and visualization
# See individual code blocks above for full implementation



logger.info("All images generated successfully!")
