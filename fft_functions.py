from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import math


@dataclass
class Waveform:
    """
    Gather meta information to generate a numpy array of a waveform.
    """
    frequency: int
    amplitude: int = 1
    # This will impact how many frequencies FFT analyzes.
    resolution: int = 200
    phase_shift: float = 0

    def get_wave(self):
        length = np.pi * 2 * self.frequency
        t = np.arange(0, length, length / self.resolution)
        return t, self.amplitude * np.cos(t + self.phase_shift)


def dft_only_sin(
    t: npt.NDArray[float], waveform: npt.NDArray[float]
) -> npt.NDArray[float]:
    N = len(waveform)
    harmonics = np.zeros(len(t))
    for k, j in enumerate(range(N)):
        potential_harmonic = 0
        for i, x in enumerate(waveform):
            n = i
            potential_harmonic += x * np.sin(np.pi * 2 * (k / N) * n)
        # print(f"harmonic {j} is {potential_harmonic}")
        # Dividing by N means normalizing the result.
        harmonics[j] = potential_harmonic / N

    return harmonics


def dft(
    t: npt.NDArray[float], waveform: npt.NDArray[float]
) -> npt.NDArray[complex]:
    N = len(waveform)
    harmonics = np.zeros(len(t), dtype=complex)
    for k, j in enumerate(range(N)):
        potential_harmonic = 0
        for i, x in enumerate(waveform):
            n = i
            potential_harmonic += x * complex(
                np.cos(2 * np.pi * (k / N) * n),
                -1 * np.sin(2 * np.pi * (k / N) * n),
            )
        # print(f"harmonic {j} is {potential_harmonic}")
        harmonics[j] = potential_harmonic

    return harmonics


def generate_window(
    t: npt.NDArray[float], window_fill: float, window_offset: float
) -> npt.NDArray[float]:
    """
    # windowing: https://numpy.org/doc/stable/reference/routines.window.html
    Generate a window function, that has the same length as the input t.
    This function isn't safer - you can generate a function that won't have a
    size that fits and it'll raise an exception.
    """
    window_fill_in_samples = math.floor(window_fill * len(t))
    window_offset_in_samples = math.floor(window_offset * len(t))

    window_with_padding = np.zeros(len(t))
    window = np.blackman(window_fill_in_samples)

    # Insert the offset window.
    window_with_padding[
        window_offset_in_samples : window_offset_in_samples
        + window_fill_in_samples
    ] = window
    return window_with_padding


