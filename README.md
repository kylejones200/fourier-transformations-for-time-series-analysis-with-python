# Fourier Transformations for Time Series Analysis with Python

This project demonstrates Fourier Transform analysis for time series data, including frequency domain analysis and noise filtering.

## Business context

Time series data often contains hidden periodic patterns that are difficult to identify through direct observation. Fourier Transformations provide a mathematical framework to decompose these complex signals into their fundamental frequency components, revealing underlying patterns and cycles that might otherwise remain undetected.

The Fourier Transform converts a time series from the time domain (data as observed over time) to the frequency domain (data as periodic signals). Essentially, it expresses a time series as a sum of sine and cosine waves with different frequencies and amplitudes.

Financial markets use Fourier analysis to detect trading cycles and seasonal patterns in asset prices. By decomposing price movements into their frequency components, analysts can distinguish between short-term fluctuations and longer-term trends, informing trading strategies and risk management decisions.

## Article

Medium article: [Fourier Transformations for Time Series Analysis](https://medium.com/@kylejones_47003/fourier-transformations-for-time-series-analysis-with-python-635747d1a35e)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Fourier transform functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files (if needed)
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Synthetic signal parameters (frequencies, amplitudes)
- Noise filtering threshold
- Which analyses to run

## Caveats

- The synthetic signal example demonstrates basic FFT on a combination of sine waves.
- The airline data example uses the sktime dataset loader.
- Noise filtering uses a simple threshold-based approach; more sophisticated methods are available.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).