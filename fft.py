import marimo

__generated_with = "0.1.77"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    from pathlib import Path
    import struct

    data_unpacked = np.asarray([d[0] for d in struct.iter_unpack("<H", Path("samples.bin").read_bytes())])
    return Path, data_unpacked, np, struct


@app.cell
def _(mo):
    mo.md(
        r"""
        Loaded the 2 byte sample data.

        Lets proceed to plotting the scaled data.
        """
    )
    return


@app.cell
def _(data_unpacked, np):
    import matplotlib.pyplot as plt

    data_float = []
    to_float = lambda x: x / 8192.0
    data_float = to_float(data_unpacked)

    FREQUENCY = 32 * 1000
    BYTES_PER_SAMPLE = 2

    data_to_plot = data_float

    dt = 1.0 / FREQUENCY
    t = np.arange(0, len(data_to_plot), 1)

    fig, axs = plt.subplots(len((data_to_plot, [])))

    for i, data in enumerate((data_to_plot, )):
        axs[i].set_ylim([data.mean() - 0.05, data.mean() + 0.05])
        axs[i].plot(t, data, linewidth=0.1)
        axs[i].set(xlabel='sample', ylabel='val', title='Soundwave plot')
        axs[i].grid(color='k', alpha=0.2, linestyle='-.', linewidth=0.5)
    return (
        BYTES_PER_SAMPLE,
        FREQUENCY,
        axs,
        data,
        data_float,
        data_to_plot,
        dt,
        fig,
        i,
        plt,
        t,
        to_float,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        Lets test some simple FFT case.
        """
    )
    return


@app.cell
def _():
    from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
    from dataclasses import dataclass

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

    t, harmonic01 = Waveform(cycles=4, amplitude=1).get_wave()
    _, harmonic02 = Waveform(cycles=12, amplitude=2).get_wave()
    #_, waveform03 = Waveform(cycles=5).get_wave()
    waveform = harmonic01 + harmonic02

    harmonics = np.zeros(len(t))
    N = len(waveform)
    for k, j in enumerate(range(N)):    
        potential_harmonic = 0
        for i, x in enumerate(waveform):
            n = i
            potential_harmonic += x * np.sin(np.pi * 2 * (k/N) * n)

        print(f"harmonic {j} is {potential_harmonic}")
        # Dividing by N means normalizing the result.
        harmonics[j] = potential_harmonic / N

    data_to_plot = (harmonic01, harmonic02, waveform, harmonics)
    fig, axs = plt.subplots(len(data_to_plot), figsize=(14, 14))

    for i, data in enumerate(data_to_plot[:-1]):
        axs[i].set_ylim([data.min() - 0.4, data.max() + 0.4])
        axs[i].plot(t, data, linewidth=0.5)
        #axs[i].set(xlabel='sample', ylabel='val', title='Soundwave plot')
        axs[i].grid(color='k', alpha=0.2, linestyle='-.', linewidth=0.5)

    axs[-1].bar(np.arange(len(harmonics)), harmonics)
    axs[-1].grid(color='k', alpha=0.2, linestyle='-.', linewidth=0.5)
    axs[-1].xaxis.set_major_locator(MultipleLocator(10))
    axs[-1].xaxis.set_minor_locator(MultipleLocator(5))
        
    return (
        AutoMinorLocator,
        MultipleLocator,
        N,
        Waveform,
        axs,
        data,
        data_to_plot,
        dataclass,
        fig,
        harmonic01,
        harmonic02,
        harmonics,
        i,
        j,
        k,
        n,
        np,
        plt,
        potential_harmonic,
        t,
        waveform,
        x,
    )


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    return mo,


if __name__ == "__main__":
    app.run()

