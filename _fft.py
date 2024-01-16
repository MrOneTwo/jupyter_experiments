import marimo

__generated_with = "0.1.77"
app = marimo.App()


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

    data_to_plot = data_float
    dt = 1.0 /FREQUENCY
    t = np.arange(0, len(data_to_plot), 1)

    fig, axs = plt.subplots(len((data_to_plot, [])))

    for i, data in enumerate((data_to_plot, )):
        axs[i].set_ylim([data.mean() - 0.05, data.mean() + 0.05])
        axs[i].plot(t, data, linewidth=0.1)
        axs[i].set(xlabel='sample', ylabel='val', title='Soundwave plot')
        axs[i].grid(color='k', alpha=0.2, linestyle='-.', linewidth=0.5)

    fig

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
        normalize_data,
        t,
    )


@app.cell
def __(mo):
    mo.md(r"""Here is the simplified DFT

    \[
    X_k = \sum_{n=0}^{N-1} x_n \sin(2\pi\frac{k}{N}n)
    \]
    """)
    return


if __name__ == "__main__":
    app.run()
