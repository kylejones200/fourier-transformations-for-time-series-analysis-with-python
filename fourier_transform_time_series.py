import logging
from pathlib import Path

import numpy as np
from scipy.fft import fft, ifft
from src.core import (
    add_noise,
    compute_fft,
    filter_noise,
    generate_synthetic_signal,
    inverse_fft,
    load_airline_data,
    plot_frequency_domain,
    plot_noise_filtering,
    plot_time_series,
)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def run_synthetic_fft_demo(output_dir: Path) -> None:
    np.random.seed(42)
    time = np.linspace(0, 10, 500)
    signal = generate_synthetic_signal(time, freq1=2, freq2=5)
    plot_time_series(
        time,
        signal,
        output_dir / "synthetic_signal.png",
        "Time Series (Combination of Two Sine Waves)",
        plot=True,
    )
    sample_rate = time[1] - time[0]
    frequencies, fft_result = compute_fft(signal, sample_rate)
    plot_frequency_domain(
        frequencies,
        fft_result,
        output_dir / "synthetic_frequency_domain.png",
        "Frequency Domain Representation",
        plot=True,
    )
    fft_result_scipy = fft(signal)
    ifft(fft_result_scipy)


def run_airline_fft_demo(output_dir: Path) -> None:
    time, y = load_airline_data()
    plot_time_series(
        time,
        y,
        output_dir / "airline_data.png",
        "Original Airline Passenger Data",
        ylabel="Passengers",
        plot=True,
    )
    frequencies, fft_result = compute_fft(y, sample_rate=1)
    plot_frequency_domain(
        frequencies,
        fft_result,
        output_dir / "time_series_passanger.png",
        "Frequency Domain of Airline Passenger Data",
        plot=True,
    )
    noisy_signal = add_noise(y, noise_level=50, seed=42)
    frequencies, fft_result_noisy = compute_fft(noisy_signal, sample_rate=1)
    fft_filtered = filter_noise(fft_result_noisy, frequencies, threshold=0.1)
    filtered_signal = inverse_fft(fft_filtered)
    plot_noise_filtering(
        time,
        noisy_signal,
        filtered_signal,
        output_dir / "time_series_passanger_amp.png",
        plot=True,
    )


def main() -> None:
    configure_logging()
    output_dir = Path(".")
    run_synthetic_fft_demo(output_dir)
    run_airline_fft_demo(output_dir)


if __name__ == "__main__":
    main()
