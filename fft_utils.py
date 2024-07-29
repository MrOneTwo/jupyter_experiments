from dataclasses import dataclass
from pathlib import Path
import numpy as np
import numpy.typing as npt
import math
import wave
import typing
import csv
import struct


@dataclass
class Waveform:
    """
    Gather meta information to generate a numpy array of a waveform.
    """
    frequency: int
    amplitude: int = 1
    # This will impact how many frequencies FFT analyzes.
    resolution: int = 256
    phase_shift: float = 0
    time: float = 1.0

    def get_wave(self):
        t = np.arange(0, self.time + self.time / self.resolution, self.time / self.resolution)
        return t, self.amplitude * np.cos(2 * np.pi * self.frequency * t + self.phase_shift)


def dft_only_sin(
    t: npt.NDArray[np.float32], waveform: npt.NDArray[np.float32]
) -> npt.NDArray[np.float64]:
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


def dft_naive(
    t: npt.NDArray[np.float32], waveform: npt.NDArray[np.float32]
) -> npt.NDArray[np.complex128]:
    N = len(waveform)
    print(f"running DFT on {N} samples")
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


def dft(samples: npt.NDArray[np.float32]):
    N = len(samples)

    # Lets precompute the array.
    X_k = np.zeros([N, N], dtype=complex)
    for k in range(N):
        for n in range(N):
            X_k[k][n] = complex(np.cos(2 * np.pi * (k / N)  * n), -1 * np.sin(2 * np.pi * (k / N) * n))

    return X_k.dot(samples)

def fft(x):
    """
    A recursive implementation of
    the 1D Cooley-Tukey FFT, the
    input should have a length of
    power of 2.
    """
    N = len(x)

    if N == 1:
        return x
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        # Prepare the factors for phase shifting the complex frequencies.
        # Phase shifting means "vector rotating".
        factor = np.exp(-2j*np.pi*np.arange(N)/ N)

        X = np.concatenate([X_even+factor[:int(N/2)]*X_odd,
                            X_even+factor[int(N/2):]*X_odd])
        return X


def bin_to_freq(sampling_freq: int, bin_index: int, samples_count: int) -> float:
    base_period = (sampling_freq / samples_count)
    freq = bin_index * base_period
    return freq


def generate_window(
    t: npt.NDArray[float],
    window_fill: float,
    window_offset: float,
    to_nearest_power_of_two: bool=False
) -> npt.NDArray[float]:
    """
    # windowing: https://numpy.org/doc/stable/reference/routines.window.html
    Generate a window function, that has the same length as the input t.
    This function isn't safer - you can generate a function that won't have a
    size that fits and it'll raise an exception.
    """
    window_fill_in_samples = math.floor(window_fill * len(t))
    window_offset_in_samples = math.floor(window_offset * len(t))

    if to_nearest_power_of_two:
        multiple_of_256 = window_fill_in_samples // 256
        window_fill_in_samples = multiple_of_256 * 256

    window_with_padding = np.zeros(len(t))
    window = np.blackman(window_fill_in_samples)

    # Insert the offset window.
    window_with_padding[
        window_offset_in_samples : window_offset_in_samples
        + window_fill_in_samples
    ] = window
    return window_with_padding


def generate_window_n(
    t: npt.NDArray[float],
    window_width_in_samples: int,
    window_offset: float
) -> npt.NDArray[float]:
    window_offset_in_samples = math.floor(window_offset * len(t))

    window_with_padding = np.zeros(len(t))
    window = np.blackman(window_width_in_samples)

    # Insert the offset window.
    window_with_padding[
        window_offset_in_samples : window_offset_in_samples
        + window_width_in_samples
    ] = window
    return window_with_padding


def params_from_file_name(filename: str) -> typing.List[int]:
    sample_rate = 32000
    bytes_per_sample = 2

    if "16k" in filename:
        sample_rate = 16000
    elif "32k" in filename:
        sample_rate = 32000
    elif "44k" in filename:
        sample_rate = 44100

    if "8bit" in filename:
        bytes_per_sample = 1
    elif "16bit" in filename:
        bytes_per_sample = 2
    elif "32bit" in filename:
        bytes_per_sample = 4

    return (sample_rate, bytes_per_sample)


def raw_data_to_wave(data: np.ndarray, output_name: str, sample_rate: int, bytes_per_sample):
    # The microphone that recorded the samples has a certain bit depth for each sample.
    # Convert to signed 16bit samples.
    # Signedss of PCM data: uint8, int16, int24, int32
    if bytes_per_sample == 1:
        normalize_factor = np.iinfo(np.uint8).max
    elif bytes_per_sample == 2:
        normalize_factor = np.iinfo(np.int16).max
    # elif bytes_per_sample == 3:
    #     normalize_factor = np.iinfo(np.int16).max
    elif bytes_per_sample == 4: # signed
        normalize_factor = np.iinfo(np.int32).max

    normalize = lambda x: x / normalize_factor
    _data_float = normalize(data)
    soundwave = data

    with wave.open(output_name, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(bytes_per_sample)
        f.setframerate(sample_rate)
        f.writeframes(soundwave.astype(np.uint16))


def filter_harmonics(harmonics: npt.NDArray[np.complex128], epsilon: float=0.001)-> typing.List[dict]:
    valid_harmonics = []
    for i, h in enumerate(harmonics):
        if abs(h) > epsilon or abs(h) < -1 * epsilon:
            valid_harmonics.append({"bin_idx": i, "value": h})

    return valid_harmonics


def sound_from_file(filepath: str) -> list[int, int, npt.NDArray[np.float32]]:
    if Path(filepath).suffix == ".wav":
        with wave.open(filepath, "r") as wf:
            _wav_data = wf.readframes(wf.getnframes())
        data_unpacked = np.frombuffer(_wav_data, dtype=np.int16)
        bytes_per_sample = wf.getsampwidth()
        sample_rate = wf.getframerate()
    elif Path(filepath).suffix == ".bin":
        # Lets make a wave file out of raw samples.
        data_unpacked = np.asarray(
            [
                d[0]
                for d in struct.iter_unpack("<h", Path(filepath).read_bytes())
            ]
        )
        sample_rate, bytes_per_sample = params_from_file_name(filepath)
        raw_data_to_wave(
            data_unpacked,
            str(Path(filepath).with_suffix(".wav")),
            sample_rate,
            bytes_per_sample,
        )
        _wav_data = Path(filepath).with_suffix(".wav").read_bytes()
    else:
        assert False, "yikes"

    return (bytes_per_sample, sample_rate, data_unpacked)


def sound_from_wav_file(filepath: str, chunk: int) -> list[int, int, npt.NDArray[np.float32]]:
    """
    Load a wav file, but expect a file with the same filename
    but .txt extension. The .txt file should be a labels file
    generated by Audacity.
    """
    assert Path(filepath).suffix == ".wav"

    wave_file = wave.open(filepath, "r")
    framerate = wave_file.getframerate()
    frames_count = wave_file.getnframes()
    bytes_per_sample = wave_file.getsampwidth()
    duration = frames_count / float(framerate)
    wave_file.close()

    # Expect a file with the same name, but .txt for extension,
    # in a Labels format, generated by Audacity.
    labels_filepath = Path(filepath).with_suffix(".txt")
    labels = []
    if labels_filepath.exists():
        with open(str(labels_filepath), newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter="\t")
            for row in reader:
                s_start, s_end, s_label = row
                labels.append(
                    {
                        "label": s_label,
                        "start": float(s_start),
                        "end": float(s_end),
                    }
                )
    else:
        assert (
            False
        ), f"The {Path(filepath).with_suffix('.txt')} doesn't exist!"

    wave_file = wave.open(filepath, "r")
    chunk_size_time = labels[chunk]["end"] - labels[chunk]["start"]
    chunk_size_frames = int(frames_count * (chunk_size_time / duration))
    wave_file.setpos(int(frames_count * (labels[chunk]["start"] / duration)))
    if bytes_per_sample == 4:
        data_unpacked = np.frombuffer(
            wave_file.readframes(chunk_size_frames), dtype=np.int32
        )
    elif bytes_per_sample == 2:
        data_unpacked = np.frombuffer(
            wave_file.readframes(chunk_size_frames), dtype=np.int16
        )
    elif bytes_per_sample == 1:
        data_unpacked = np.frombuffer(
            wave_file.readframes(chunk_size_frames), dtype=np.int8
        )
    wave_file.close()

    return (bytes_per_sample, framerate, data_unpacked)
