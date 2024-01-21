import marimo

__generated_with = "0.1.77"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    from pathlib import Path
    import struct

    import numpy as np
    import numpy.typing as npt
    import matplotlib.pyplot as plt
    import wave

    mo.md("# Fourier Transform")
    return Path, mo, np, npt, plt, struct, wave


@app.cell
def __(Path, mo, np, struct, wave):
    SAMPLES_FILE = "samples.bin"

    data_unpacked = np.asarray(
        [d[0] for d in struct.iter_unpack("<H", Path(SAMPLES_FILE).read_bytes())]
    )

    # The microphone that recorded the samples has a certain bit depth for each sample.
    # Convert to signed 16bit samples.
    _data_float = data_unpacked / 8192.0
    soundwave = _data_float * np.iinfo(np.int16).max

    SAMPLE_RATE = 32000
    BYTES_PER_SAMPLE = 2

    with wave.open(str(Path(SAMPLES_FILE).with_suffix(".wav")), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(BYTES_PER_SAMPLE)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(data_unpacked.astype(np.int16))

    with open(str(Path(SAMPLES_FILE).with_suffix(".wav")), "rb") as _p:
        mo.audio(src=_p)
    return (
        BYTES_PER_SAMPLE,
        SAMPLES_FILE,
        SAMPLE_RATE,
        data_unpacked,
        f,
        soundwave,
    )


@app.cell
def __(mo):
    mo.md("Loaded the 2 byte sample data.")
    return


@app.cell
def __(SAMPLE_RATE, data_unpacked, np, plt):
    data_float = list()
    normalize_data = lambda x: x/ 8192.0
    data_float = normalize_data(data_unpacked)

    _data_to_plot = data_float
    dt = 1.0 /SAMPLE_RATE
    _t = np.arange(0, len(_data_to_plot), 1)

    _fig, _axs = plt.subplots(len((_data_to_plot, [])))

    for _i, _data in enumerate((_data_to_plot, )):
        _axs[_i].set_ylim([_data.mean() - 0.05, _data.mean() + 0.05])
        _axs[_i].plot(_t, _data, linewidth=0.1)
        _axs[_i].set(xlabel='sample', ylabel='val', title='Soundwave plot')
        _axs[_i].grid(color='k', alpha=0.2, linestyle='-.', linewidth=0.5)

    _fig
    return data_float, dt, normalize_data


@app.cell
def __(mo):
    mo.md(r"""Here is the simplified DFT

    \[
    X_k = \sum_{n=0}^{N-1} x_n \sin(2\pi\frac{k}{N}n)
    \]

    N - is the total samples count.
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""You can control the samples count for the following waveforms.""")
    return


@app.cell
def __(mo, np):
    samples_count_slider = mo.ui.slider(
        start=1, stop=400, value=200, label="Samples count", debounce=True
    )
    phase_shift_slider = mo.ui.slider(
        start=0.0,
        stop=np.pi,
        step=0.1,
        value=1.0,
        label="Phase shift of the first waveform",
        debounce=True,
    )
    mo.vstack([mo.md(f"{samples_count_slider}"), mo.md(f"{phase_shift_slider}")])
    return phase_shift_slider, samples_count_slider


@app.cell
def __(np, npt, phase_shift_slider, plt, samples_count_slider):
    from dataclasses import dataclass
    from matplotlib.ticker import MultipleLocator, AutoMinorLocator


    @dataclass
    class Waveform:
        frequency: int
        amplitude: int = 1
        # This will impact how many frequencies FFT analyzes.
        resolution: int = 200
        phase_shift: float = 0

        def get_wave(self):
            length = np.pi * 2 * self.frequency
            t = np.arange(0, length, length / self.resolution)
            return t, self.amplitude * np.cos(t + self.phase_shift)


    _t, harmonic01 = Waveform(
        frequency=4,
        amplitude=1,
        phase_shift=phase_shift_slider.value,
        resolution=int(samples_count_slider.value),
    ).get_wave()
    _, harmonic02 = Waveform(
        frequency=12, amplitude=2, resolution=int(samples_count_slider.value)
    ).get_wave()
    # _, waveform03 = Waveform(cycles=5).get_wave()

    waveform = harmonic01 + harmonic02


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


    # Here we have the information split over real and imaginary part, depending
    # on if the harmonics resemble cos or sin more.
    harmonics = dft(_t, waveform)
    # Filter out the almost 0 values. It's especially important for computing the
    # phase shift, with arctan2. That's because two very small number, divided by
    # each other, will result in a legit value: 0.00000002/0.00000001 = 2.
    harmonics[:] = list(map(lambda c: c if abs(c) > 0.0001 else 0.0, harmonics))
    # abs for complex computes magnituted
    harmonics_mag = list(map(abs, harmonics))
    harmonics_phase = list(map(lambda c: np.arctan2(c.imag, c.real), harmonics))
    #print(harmonics)
    #print(harmonics_mag)
    #print(harmonics_phase)

    data_to_plot = (
        harmonic01,
        harmonic02,
        waveform,
        harmonics_mag,
        harmonics_phase,
    )
    _fig, _axs = plt.subplots(len(data_to_plot), figsize=(14, 14))

    for i, data in enumerate(data_to_plot[:-2]):
        _axs[i].set_ylim([data.min() - 0.4, data.max() + 0.4])
        _axs[i].plot(_t, data, linewidth=0.7, linestyle="solid", marker="o")
        # _axs[i].set(xlabel='sample', ylabel='val', title='Soundwave plot')
        _axs[i].grid(color="k", alpha=0.2, linestyle="-.", linewidth=0.5)

    _axs[-2].bar(np.arange(len(harmonics_mag)), harmonics_mag)
    _axs[-2].grid(color="k", alpha=0.2, linestyle="-.", linewidth=0.5)
    _axs[-2].xaxis.set_major_locator(MultipleLocator(10))
    _axs[-2].xaxis.set_minor_locator(MultipleLocator(5))

    _axs[-1].bar(np.arange(len(harmonics_mag)), harmonics_phase)
    _axs[-1].grid(color="k", alpha=0.2, linestyle="-.", linewidth=0.5)
    _axs[-1].xaxis.set_major_locator(MultipleLocator(10))
    _axs[-1].xaxis.set_minor_locator(MultipleLocator(5))

    _fig
    return (
        AutoMinorLocator,
        MultipleLocator,
        Waveform,
        data,
        data_to_plot,
        dataclass,
        dft,
        dft_only_sin,
        harmonic01,
        harmonic02,
        harmonics,
        harmonics_mag,
        harmonics_phase,
        i,
        waveform,
    )


@app.cell
def __(harmonics, mo):
    def filter_harmonics(harmonics):
        valid_harmonics = []
        for i, h in enumerate(harmonics):
            if abs(h) > 0.001 or abs(h) < -0.001:
                valid_harmonics.append((i, h))

        return valid_harmonics


    mo.md(
        f"Checked {len(harmonics)} possible harmonics: {filter_harmonics(harmonics)}"
    )
    return filter_harmonics,


@app.cell
def __(mo):
    mo.md("The spectrum graph is mirrored. That's because the samples of the originally sampled signal match with the sinusoid of those higher frequencies too. The frequencies in the middle is the Nyquist Rate.")
    return


if __name__ == "__main__":
    app.run()
