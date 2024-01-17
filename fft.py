import marimo

__generated_with = "0.1.77"
app = marimo.App()


@app.cell
def __(mo):
    mo.md("# Fourier Transform")
    return


@app.cell
def __():
    import marimo as mo
    from pathlib import Path
    import struct

    import numpy as np
    import matplotlib.pyplot as plt

    data_unpacked = np.asarray(
        [d[0] for d in struct.iter_unpack("<H", Path("samples.bin").read_bytes())]
    )
    return Path, data_unpacked, mo, np, plt, struct


@app.cell
def __(mo):
    mo.md("Loaded the 2 byte sample data.")
    return


@app.cell
def __(data_unpacked, np, plt):
    data_float = list()
    normalize_data = lambda x: x/ 8192.0
    data_float = normalize_data(data_unpacked)

    FREQUENCY = 32 * 1000
    BYTES_PER_SAMPLE = 2

    _data_to_plot = data_float
    dt = 1.0 /FREQUENCY
    _t = np.arange(0, len(_data_to_plot), 1)

    _fig, _axs = plt.subplots(len((_data_to_plot, [])))

    for _i, _data in enumerate((_data_to_plot, )):
        _axs[_i].set_ylim([_data.mean() - 0.05, _data.mean() + 0.05])
        _axs[_i].plot(_t, _data, linewidth=0.1)
        _axs[_i].set(xlabel='sample', ylabel='val', title='Soundwave plot')
        _axs[_i].grid(color='k', alpha=0.2, linestyle='-.', linewidth=0.5)

    _fig
    return BYTES_PER_SAMPLE, FREQUENCY, data_float, dt, normalize_data


@app.cell
def __(mo):
    mo.md(r"""Here is the simplified DFT

    \[
    X_k = \sum_{n=0}^{N-1} x_n \sin(2\pi\frac{k}{N}n)
    \]
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""You can control the samples count for the following waveforms.""")
    return


@app.cell
def __(mo):
    samples_count_slider = mo.ui.slider(start=1, stop=400, value=200, label="Samples count", debounce=True)
    mo.md(f"{samples_count_slider}")
    return samples_count_slider,


@app.cell
def __(np, numpy, plt, samples_count_slider):
    from dataclasses import dataclass
    from matplotlib.ticker import MultipleLocator, AutoMinorLocator


    @dataclass
    class Waveform:
        cycles: int
        amplitude: int = 1
        # This will impact how many frequencies FFT analyzes.
        resolution: int = 200

        def get_wave(self):
            length = np.pi * 2 * self.cycles
            t = np.arange(0, length, length / self.resolution)
            return t, self.amplitude * np.sin(t)


    _t, harmonic01 = Waveform(
        cycles=4, amplitude=1, resolution=int(samples_count_slider.value)
    ).get_wave()
    _, harmonic02 = Waveform(
        cycles=12, amplitude=2, resolution=int(samples_count_slider.value)
    ).get_wave()
    # _, waveform03 = Waveform(cycles=5).get_wave()

    waveform = harmonic01 + harmonic02


    def dft(t: numpy.ndarray, waveform: numpy.ndarray):
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


    harmonics = dft(_t, waveform)

    data_to_plot = (harmonic01, harmonic02, waveform, harmonics)
    _fig, _axs = plt.subplots(len(data_to_plot), figsize=(14, 14))

    for i, data in enumerate(data_to_plot[:-1]):
        _axs[i].set_ylim([data.min() - 0.4, data.max() + 0.4])
        _axs[i].plot(_t, data, linewidth=0.5)
        # _axs[i].set(xlabel='sample', ylabel='val', title='Soundwave plot')
        _axs[i].grid(color="k", alpha=0.2, linestyle="-.", linewidth=0.5)

    _axs[-1].bar(np.arange(len(harmonics)), harmonics)
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
        harmonic01,
        harmonic02,
        harmonics,
        i,
        waveform,
    )


@app.cell
def __(harmonics, mo):
    mo.md(f"Checked {len(harmonics)} possible harmonics")
    return


if __name__ == "__main__":
    app.run()
