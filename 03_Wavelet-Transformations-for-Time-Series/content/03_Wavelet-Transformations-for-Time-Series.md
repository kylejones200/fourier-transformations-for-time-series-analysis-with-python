# Wavelet Transformations for Time Series Analysis in Python: Beyond Fourier 

Fourier transforms excel at periodic signals, but wavelets are superior for non-stationary time series. We demonstrate wavelet decomposition, denoising, and anomaly detection on Oklahoma CO2 emissions data from 1960-2023.

### Wavelet Transformations for Time Series Analysis in Python: Beyond Fourier
Fourier transforms revolutionized signal processing by decomposing signals into frequency components. But they assume signals are stationary—patterns repeat indefinitely. Real-world time series are non-stationary: trends change, patterns shift, anomalies occur at specific times.

Wavelets solve this by providing localized time-frequency analysis. They capture both when and where patterns occur, making them ideal for non-stationary signals with abrupt changes, trends, and localized features.

We demonstrate wavelet analysis on Oklahoma CO2 emissions data spanning 1960-2023. This series exhibits non-stationary behavior: policy changes create sudden shifts, economic cycles cause fluctuations, and technological advances drive long-term trends.

### Why Wavelets Over Fourier?
Fourier transforms show what frequencies are present but not when they occur. Wavelets show both frequency and time information simultaneously. This makes wavelets superior for:

- Non-stationary signals Patterns that change over time
- Localized features Events that occur at specific times
- Multi-scale analysis Patterns at different time scales
- Denoising Removing noise while preserving important features

### Dataset: Oklahoma CO2 Emissions
We use CO2 emissions data that shows clear non-stationary behavior.

The dataset shows clear non-stationary behavior: emissions rise through the 1970s, decline in the 1980s, then rise again. Policy changes and economic cycles create localized patterns that Fourier transforms would miss.

In our experiment, the annual CO2 emissions series spans 64 years (1960–2023), with values ranging from 272.7 to 1,569.7 million metric tons. This long history with clear structural changes makes it a good stress test for wavelet methods.

### Wavelet Decomposition
Wavelet decomposition splits a signal into approximation (low-frequency trend) and detail (high-frequency components) coefficients.

The approximation captures the long-term trend, while detail coefficients capture fluctuations at different scales.

The decomposition reveals multi-scale patterns: long-term trends in approximation, short-term fluctuations in detail levels.

In our analysis, we used a Daubechies-4 (`db4`) wavelet with a 5-level decomposition. The approximation coefficients at level 5 condense the 64-point series down to just 8 coefficients, while the detail levels capture structure at progressively finer scales (from 8 up to 35 coefficients). The plot saved as `03_Wavelet-Transformations-for-Time-Series/images/wavelet_decomposition.png` shows the original CO2 series at the top and the approximation/detail components at each level underneath.

### Wavelet Denoising
Wavelet thresholding removes noise while preserving important signal features.

Soft thresholding shrinks coefficients toward zero, preserving smoothness. Hard thresholding sets small coefficients to zero, which can create artifacts.

On this CO2 series, wavelet denoising produced a reconstruction error effectively equal to 0 (down to numerical precision), meaning the denoised signal closely matches the original. The estimated noise standard deviation was ≈ 25.97, and the signal-to-noise ratio (SNR) improved by about 16.9× after denoising. The figure `03_Wavelet-Transformations-for-Time-Series/images/wavelet_denoising.png` overlays the raw and denoised series so you can see how short-term noise is smoothed while long-term trends remain intact.

### Continuous Wavelet Transform (CWT)
CWT provides a time-frequency representation showing how patterns evolve over time.

The scalogram shows when different frequency patterns occur. Bright regions indicate strong patterns at specific times and scales.

For this experiment, the dominant scale identified by the CWT corresponds to roughly 64 years, with a dominant frequency of 0.0156 cycles per year. This matches the overall length of the series and highlights that most of the energy is in very low-frequency, long-term trends. The scalogram in `03_Wavelet-Transformations-for-Time-Series/images/cwt_spectrogram.png` shows how energy is concentrated at these coarse scales but still exhibits localized bursts linked to specific policy or economic shifts.

### Anomaly Detection with Wavelets
Wavelets excel at detecting localized anomalies that statistical methods miss.

Wavelet-based anomaly detection identifies localized events that global statistics miss. It's particularly effective for detecting sudden changes or outliers.

Using wavelet detail coefficients and a robust threshold based on the estimated noise level, the method flagged 17 anomaly periods in the CO2 series. Many of these anomalies cluster in the mid-1960s and late 1970s–mid-1980s (for example, 1964–1967, 1978–1980, 1982, 1984–1985), which correspond to major shifts in economic activity and energy policy. The anomalies are visualized in `03_Wavelet-Transformations-for-Time-Series/images/wavelet_anomalies.png`, where unusually sharp local changes stand out from the background trend.

### Comparison with Fourier Transform
We compare wavelets with Fourier transforms to show their complementary strengths.

Fourier shows what frequencies exist globally. Wavelets show when and where frequencies occur, making them superior for non-stationary analysis.

In our CO2 example, the Fourier spectrum correctly identifies strong low-frequency content but cannot indicate when shifts in emissions occur. The side-by-side comparison in `03_Wavelet-Transformations-for-Time-Series/images/fourier_vs_wavelet.png` makes this clear: the Fourier plot compresses all activity into global frequency bands, while the wavelet-based view exposes how those bands intensify or fade at specific decades.

### Key Advantages of Wavelets
- Localized analysis Captures both frequency and time information simultaneously
- Multi-resolution Analyzes signal at different scales (short-term and long-term patterns)
- Non-stationary signals Handles signals with changing properties over time
- Denoising Effective noise removal while preserving important signal features
- Anomaly detection Identifies localized anomalies that global methods miss

### Practical Applications
- Signal denoising Remove noise from sensor data while preserving important features
- Feature extraction Extract time-frequency features for machine learning models
- Anomaly detection Identify unusual patterns or events in time series
- Trend analysis Separate long-term trends from short-term fluctuations
- Compression Efficient signal representation for storage and transmission

### When to Use Wavelets vs Fourier
Use Wavelets when:- Signals are non-stationary (patterns change over time)
- You need time-localized frequency information
- Detecting localized anomalies or events
- Denoising while preserving edges/transitions
- Multi-scale analysis is important

Use Fourier when:- Signals are stationary (patterns repeat)
- You only need frequency information
- Analyzing periodic components
- Fast computation is critical
- Global frequency analysis is sufficient

### Conclusion
Wavelets provide a powerful alternative to Fourier transforms for non-stationary time series. They offer localized time-frequency analysis essential for modern applications where patterns change over time. For CO2 emissions data with policy changes, economic cycles, and technological shifts, wavelets reveal patterns that Fourier transforms would miss.

The ability to decompose signals into multi-scale components, denoise effectively, and detect localized anomalies makes wavelets indispensable for real-world time series analysis.


