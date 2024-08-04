import marimo

__generated_with = "0.7.16"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    from pathlib import Path
    import struct
    import csv

    import numpy as np
    import numpy.typing as npt
    import math
    import matplotlib.pyplot as plt
    import matplotlib
    import wave
    import base64
    import pandas as pd
    import plotnine as p9

    import fft_utils as fftu

    import importlib

    importlib.reload(fftu)

    mo.md("# Fourier Transform")
    return (
        Path,
        base64,
        csv,
        fftu,
        importlib,
        math,
        matplotlib,
        mo,
        np,
        npt,
        p9,
        pd,
        plt,
        struct,
        wave,
    )


@app.cell
def __(plt):
    plt.style.use("ggplot")
    colors = []

    def pull_colors():
        colors = list()
        for style in sorted(plt.style.library):
            the_rc = plt.style.library[style]
            if "axes.prop_cycle" in the_rc:
                cols = the_rc["axes.prop_cycle"].by_key()["color"]
                colors = cols
                #print("%25s: %s" % (style, ", ".join(color for color in cols)))
            else:
                #print("%25s this style does not modify colors" % style)
                pass

        return colors

    colors = pull_colors()
    return colors, pull_colors


@app.cell
def __(mo):
    mo.md(
        r"""
        Here is the simplified DFT. The result will be a set of complex numbers. Think of those complex numbers like of a vector $A -j B$. $A$ is proportional to how much of the $\cos$ of that specific frequency is present in the final signal, with $B$ being proportional to how much $\sin$ is there.

        \[
        X_k = \sum_{n=0}^{N-1} x_n [\cos(2\pi\frac{n}{N}k) -j \sin(2\pi\frac{n}{N}k)]
        \]

        N - is the total samples count.

        Total contribution of a specific frequency is taken from $\sqrt{A^2 + B^2}$. The phase shift can be computed from $atan2(\frac{B}{A})$.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(r"""You can control the samples count for the following waveforms.""")
    return


@app.cell
def __(mo, np):
    samples_count_slider = mo.ui.slider(
        start=1, stop=512, value=512, label="Samples count", debounce=True
    )
    phase_shift_slider = mo.ui.slider(
        start=0.0,
        stop=np.pi,
        step=0.1,
        value=0.0,
        label="Phase shift of the first waveform",
        debounce=True,
    )
    mo.vstack([
               mo.md("Lets generate a waveform made out of two simple sinusoids."),
               mo.md("The spectrum of the signal is reflected. That's because of the negative frequencies the transform uses. It's normal to just ignore frequencies above the Nyquist frequency."),
               mo.md("Sliders below allow you to this case."),
               mo.md(f"{samples_count_slider}"),
               mo.md(f"{phase_shift_slider}"),
              ])
    return phase_shift_slider, samples_count_slider


@app.cell
def __(
    fftu,
    importlib,
    np,
    pd,
    phase_shift_slider,
    plt,
    samples_count_slider,
):
    from matplotlib.ticker import MultipleLocator, AutoMinorLocator

    importlib.reload(fftu)

    _sample_rate = int(samples_count_slider.value)

    _freq1 = 4
    _freq2 = 12

    time_base, harmonic01 = fftu.Waveform(
        frequency=_freq1,
        amplitude=1,
        phase_shift=phase_shift_slider.value,
        resolution=_sample_rate,
        time=2.0,
    ).get_wave()
    _, harmonic02 = fftu.Waveform(
        frequency=_freq2, amplitude=2, resolution=_sample_rate, time=2.0
    ).get_wave()

    waveform_pd = pd.DataFrame(
        {
            "t": time_base,
            "harmonic01": harmonic01,
            "harmonic02": harmonic02,
            "sum": harmonic01 + harmonic02,
        }
    )
    waveform = waveform_pd["sum"]


    # The added constant is there to show leakage.
    _window_width = 128 + 2
    # Offset won't affect leakage.
    _window_offset = 24

    _window = slice(_window_offset, _window_offset + _window_width)

    # Here we have the information split over real and imaginary part, depending
    # on if the harmonics resemble cos or sin more.
    # Use only 128 samples - basicly square windowing.
    harmonics = fftu.dft(waveform_pd["sum"][_window])

    # Compute normalized magnitude.
    harmonics_mag = abs(harmonics) / _window_width
    harmonics_phase = np.array(
        list(map(lambda c: np.arctan2(c.imag, c.real), harmonics))
    )

    _frequencies = [
        fftu.bin_to_freq(_sample_rate, k, _window_width)
        for k in range(len(harmonics_mag))
    ]


    _to_plot = [
        {
            "data": waveform_pd["harmonic01"],
            "x": time_base,
            "ylim": (
                waveform_pd["harmonic01"].min() - 0.4,
                waveform_pd["harmonic01"].max() + 0.4,
            ),
            "title": f"harmonic 1, freq: {_freq1}Hz",
            "draw_func": "plot",
            "xlabel": "[s]",
        },
        {
            "data": waveform_pd["harmonic02"],
            "x": time_base,
            "ylim": (
                waveform_pd["harmonic02"].min() - 0.4,
                waveform_pd["harmonic02"].max() + 0.4,
            ),
            "title": f"harmonic 2, freq: {_freq2}Hz",
            "draw_func": "plot",
            "xlabel": "[s]",
        },
        {
            "data": waveform_pd["sum"],
            "ylim": (
                waveform_pd["sum"].min() - 0.4,
                waveform_pd["sum"].max() + 0.4,
            ),
            "title": "combined waveform",
            "draw_func": "plot",
            "highlight": (_window_offset, _window_offset + _window_width),
            "xlabel": "[samples]",
        },
        {
            "data": waveform_pd["sum"][_window],
            "x": np.arange(_window_offset, _window_offset + _window_width, 1),
            "xticks": np.arange(_window_offset, _window_offset + _window_width, 4),
            "ylim": (
                waveform_pd["sum"].min() - 0.4,
                waveform_pd["sum"].max() + 0.4,
            ),
            "title": "sampled waveform",
            "xlabel": "[samples]",
            "draw_func": "plot",
            "draw_style": "o",
        },
        {
            # Since we're plotting only half of the results, we could multiply the magnitude
            # by 2, because part of the power is stored in that other, symmetric, half.
            "data": harmonics_mag[: (len(harmonics_mag) // 2) + 1],
            "x": _frequencies[: (len(harmonics_mag) // 2) + 1],
            "xticks": _frequencies[: (len(harmonics_mag) // 2) + 1 : 3],
            "xlabel": "[Hz]",
            "title": "harmonics magnitude - unique results",
            "draw_func": "bar",
        },
        {
            "data": harmonics_phase,
            "ylim": (harmonics_phase.min() - 0.4, harmonics_phase.max() + 0.4),
            # "yticks": {"minor": 0.5, "major": 1},
            "title": "harmonics phase",
            "xlabel": "bins",
            "draw_func": "bar",
        },
    ]

    _fig, _axs = plt.subplots(len(_to_plot), figsize=(14, 20))
    plt.subplots_adjust(hspace=1.2)

    for _ax, _data in zip(_axs, _to_plot):
        # Remove the keys that the set() function doesn't recognize.
        _y = _data.pop("data")
        _x = _data.pop("x", np.arange(len(_y)))
        _draw_func = _data.pop("draw_func")
        if "hlines" in _data:
            _hlines = _data.pop("hlines")
            _ax.axhline(_hlines, color="blue", linewidth=0.5)
        if "highlight" in _data:
            _highlight = _data.pop("highlight")
            _ax.axvspan(*_highlight, color="blue", alpha=0.15, label="window")

        if "draw_style" in _data:
            _draw_style = _data.pop("draw_style")
        else:
            _draw_style = None

        if "draw_col" in _data:
            _draw_col = _data.pop("draw_col")
        else:
            _draw_col = "black"

        # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set.html
        _ax.set(**_data)

        if _draw_func == "plot":
            if _draw_style:
                _ax.plot(
                    _x,
                    _y,
                    _draw_style,
                    markersize=1,
                    linewidth=0.5,
                    color=_draw_col,
                )
            else:
                _ax.plot(_x, _y, linewidth=0.5, color=_draw_col)
        elif _draw_func == "bar":
            _ax.bar(_x, _y, linewidth=0.5)
        elif _draw_func == "hist":
            _ax.hist(_y, bins=10)

        _ax.grid(color="k", alpha=0.2, linestyle="-.", linewidth=0.5)
        _ax.tick_params(axis="x", labelrotation=45)


    _fig
    return (
        AutoMinorLocator,
        MultipleLocator,
        harmonic01,
        harmonic02,
        harmonics,
        harmonics_mag,
        harmonics_phase,
        time_base,
        waveform,
        waveform_pd,
    )


@app.cell
def __(mo):
    mo.md("""You can see leakage in the graphs above. That's because the frequencies used in the DFT aren't exactly the same as frequencies present in the signal.""")
    return


@app.cell
def __(np, p9, waveform_pd):
    (
        p9.ggplot()
        + p9.geom_line(
            waveform_pd,
            p9.aes(
                "t",
                "harmonic02",
            ),
        )
        + p9.geom_line(waveform_pd, p9.aes("t", "harmonic01"))
        + p9.geom_point(waveform_pd, p9.aes("t", "harmonic02"))
        + p9.geom_point(waveform_pd, p9.aes("t", "harmonic01"))
        + p9.scale_x_continuous(
            expand=(0, np.pi / 2),
            breaks=(lambda x: np.arange(x[0], x[1], np.pi / 2)),
            minor_breaks=(lambda x: np.arange(x[0], x[1], np.pi / 4)),
        )
        + p9.theme(figure_size=(16, 4))
    )
    return


@app.cell
def __(mo):
    mo.vstack(
        [
            mo.md(
                "If the transform's input data isn't periodic you might see spectral leakage and aliasing effects."
            ),
            mo.md(
                "In order to force input data to be periodic, windowing is used."
            ),
            mo.md(
                "I reuse the waveform from the previous example. This time I'm windowing that waveform, to make that data periodic."
            ),
        ]
    )
    return


@app.cell
def __(MultipleLocator, fftu, np, plt, time_base, waveform):
    # Window out the input signal, to ensure a periodic input data.
    window = fftu.generate_window(time_base, 0.6, 0.3, False)
    # Create an array of bools.
    window_mask = window != 0

    harmonics_windowed = fftu.dft(
        (waveform * window)[window_mask]
    )
    windowed_harmonics_mag = list(map(abs, harmonics_windowed))
    windowed_harmonics_phase = list(map(lambda c: np.arctan2(c.imag, c.real), harmonics_windowed))


    _fig, _axs = plt.subplots(5, figsize=(10, 16))
    plt.subplots_adjust(hspace=0.4)

    _axs[0].set_ylim([waveform.min() - 0.4, waveform.max() + 0.4])
    _axs[0].plot(time_base, waveform, linewidth=0.7, linestyle="solid", marker="o", markersize=1)
    # _axs[0].set(xlabel='sample', ylabel='val', title='Soundwave plot')
    _axs[0].grid(color="k", alpha=0.2, linestyle="-.", linewidth=0.5)

    _axs[1].set_ylim([window.min() - 0.4, window.max() + 0.4])
    _axs[1].plot(time_base, window, linewidth=0.7, linestyle="solid", marker="o", markersize=1)
    # _axs[1].set(xlabel='sample', ylabel='val', title='Soundwave plot')
    _axs[1].grid(color="k", alpha=0.2, linestyle="-.", linewidth=0.5)

    _axs[2].plot(
        np.arange(len(waveform)), (waveform * window)
    )
    _axs[2].grid(color="k", alpha=0.2, linestyle="-.", linewidth=0.5)
    _axs[2].xaxis.set_major_locator(MultipleLocator(10))
    _axs[2].xaxis.set_minor_locator(MultipleLocator(5))

    _axs[3].bar(
        np.arange(len(windowed_harmonics_mag)), windowed_harmonics_mag
    )
    _axs[3].grid(color="k", alpha=0.2, linestyle="-.", linewidth=0.5)
    _axs[3].xaxis.set_major_locator(MultipleLocator(10))
    _axs[3].xaxis.set_minor_locator(MultipleLocator(5))

    _axs[4].bar(
        np.arange(len(windowed_harmonics_phase)), windowed_harmonics_phase
    )
    _axs[4].grid(color="k", alpha=0.2, linestyle="-.", linewidth=0.5)
    _axs[4].xaxis.set_major_locator(MultipleLocator(10))
    _axs[4].xaxis.set_minor_locator(MultipleLocator(5))

    _fig
    return (
        harmonics_windowed,
        window,
        window_mask,
        windowed_harmonics_mag,
        windowed_harmonics_phase,
    )


@app.cell
def __():
    #temp_data = pd.DataFrame({"t": time_base, "data": waveform})
    #print(temp_data)

    #(
        #ggplot(temp_data, aes("t", "data"))
        #+ geom_point()
        #+ theme(figure_size=(16, 8))
    #)
    return


@app.cell
def __(fftu, harmonics, mo):
    def result_to_table(data: dict) -> str:
        lines = []
        for element in data:
            line = f"{element.get('bin_idx'): >5} | {element.get('value')}"
            lines.append(line)

        return lines

    mo.vstack(
        [
            mo.md("Lets present the results in an easily readable table."),
            mo.md(
                f"""
         Checked {len(harmonics)} possible harmonics:
         """
            ),
            result_to_table(fftu.filter_harmonics(harmonics)),
        ]
    )
    return result_to_table,


@app.cell
def __(mo):
    mo.md("""The spectrum graph is mirrored. That's because the samples of the originally sampled signal match with the sinusoid of those higher frequencies too. The frequencies in the middle is the Nyquist Rate.""")
    return


@app.cell
def __(mo):
    mo.vstack([
        mo.md("## Analysis of recorded data"),
        mo.md("This time I'm reading data from a file, and compute its Fourier transform."),
    ])
    return


@app.cell
def __(fftu, mo):
    SAMPLES_FILE = "waver_abc_16k_16bit.wav"


    BYTES_PER_SAMPLE, SAMPLE_RATE, data_unpacked = fftu.sound_from_wav_file(
        SAMPLES_FILE, chunk=0
    )
    # BYTES_PER_SAMPLE, SAMPLE_RATE, data_unpacked = fftu.sound_from_file(SAMPLES_FILE)

    with open(SAMPLES_FILE, "rb") as _f:
        audio_player = mo.audio(src=_f)

    mo.vstack(
        [
            mo.md(f"Example sound file {SAMPLES_FILE}:"),
            audio_player,
        ]
    )
    return (
        BYTES_PER_SAMPLE,
        SAMPLES_FILE,
        SAMPLE_RATE,
        audio_player,
        data_unpacked,
    )


@app.cell
def __(BYTES_PER_SAMPLE, SAMPLE_RATE, fftu, mo):
    mo.vstack([mo.md(f"Loaded the {SAMPLE_RATE}Hz, {BYTES_PER_SAMPLE} byte sample data."),
               mo.md(f"The current cut off frequency is {fftu.bin_to_freq(SAMPLE_RATE, 24, 1024)} Hz")
              ])
    return


@app.cell
def __(BYTES_PER_SAMPLE, SAMPLE_RATE, data_unpacked, fftu, mo, np, plt):
    if BYTES_PER_SAMPLE == 4:
        normalize_factor = np.iinfo(np.int32).max
    elif BYTES_PER_SAMPLE == 2:
        normalize_factor = np.iinfo(np.int16).max
    elif BYTES_PER_SAMPLE == 1:
        normalize_factor = np.iinfo(np.int8).max
    normalize = lambda x: x / normalize_factor
    data_float = normalize(data_unpacked)

    _data_to_plot = data_float
    dt = 1.0 / SAMPLE_RATE
    _t = np.arange(0, len(_data_to_plot), 1)

    # Window out the input signal, to ensure a periodic input data.
    _window = fftu.generate_window_n(_t, 4096, 0.1)
    # Create an array of bools.
    _window_mask = _window != 0

    # Perform Fourier analysis on the windowed data.
    __harmonics_windowed = fftu.fft((data_float * _window)[_window_mask])
    _harmonics_windowed = __harmonics_windowed[: (len(__harmonics_windowed) // 2) + 1]
    _frequencies = [
        fftu.bin_to_freq(SAMPLE_RATE, k, 4096)
        for k in range(len(_harmonics_windowed))
    ]

    _windowed_harmonics_mag = list(map(abs, _harmonics_windowed))

    # TODO(michalc): filter_harmonics work with an array of complex numbers
    # not list of floats.
    frequency_filter_threshold = 20.0
    filtered_harmonics = fftu.filter_harmonics(
        _windowed_harmonics_mag, frequency_filter_threshold
    )

    print(filtered_harmonics)
    print(np.histogram(_windowed_harmonics_mag, bins=10))

    print(len(filtered_harmonics))

    _to_plot = [
        {
            "data": _data_to_plot,
            "ylim": (-1.1, 1.1),
            "title": "waveform",
            "draw_func": "plot",
            "xticks": [i * 2048 for i in range(10)],
            "xticklabels": [i * 2048 for i in range(10)],
        },
        {
            "data": _window,
            "ylim": (-0.1, 1.1),
            "title": "window",
            "draw_func": "plot",
            "xticks": [i * 2048 for i in range(10)],
            "xticklabels": [i * 2048 for i in range(10)],
        },
        {
            "data": (_window * _data_to_plot)[_window_mask],
            "x": np.arange(len(_window))[_window_mask],
            "ylim": (-1.1, 1.1),
            "title": "waveform windowed",
            "draw_func": "plot",
            "xlabel": "samples",
        },
        {
            "data": _windowed_harmonics_mag[: len(_windowed_harmonics_mag) // 2],
            "title": "histogram",
            "draw_func": "hist",
            "xlabel": "samples",
        },
        {
            "data": _windowed_harmonics_mag[
                filtered_harmonics[0]["bin_idx"] : filtered_harmonics[-1][
                    "bin_idx"
                ]
            ],
            "x": _frequencies[
                filtered_harmonics[0]["bin_idx"] : filtered_harmonics[-1][
                    "bin_idx"
                ]
            ],
            "title": "DFT",
            "draw_func": "bar",
            "xlabel": "samples",
            # Every Nth frequency.
            "xticks": [freq for freq in np.arange(1875.0, 1875.0 + 32 * 46.875, 46.875)],
            "xticklabels": ["{:.2f}".format(freq) for freq in np.arange(1875.0, 1875.0 + 32 * 46.875, 46.875)],
            "hlines": frequency_filter_threshold,
        },
    ]

    _fig, _axs = plt.subplots(len(_to_plot), figsize=(10, 16))

    plt.subplots_adjust(hspace=0.6)

    for _ax, _data in zip(_axs, _to_plot):
        # Remove the keys that the set() function doesn't recognize.
        _y = _data.pop("data")
        _x = _data.pop("x", np.arange(len(_y)))
        _draw_func = _data.pop("draw_func")
        if "hlines" in _data:
            _hlines = _data.pop("hlines")
            _ax.axhline(_hlines, color="blue", linewidth=0.5)
        if "highlight" in _data:
            _highlight = _data.pop("highlight")
            _ax.axvspan(*_highlight, color='blue', alpha=0.15, label='window')

        if "draw_style" in _data:
            _draw_style = _data.pop("draw_style")
        else:
            _draw_style = None

        if "draw_col" in _data:
            _draw_col = _data.pop("draw_col")
        else:
            _draw_col = "black"

        # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set.html
        _ax.set(**_data)

        if _draw_func == "plot":
            if _draw_style:
                _ax.plot(_x, _y, _draw_style, markersize=1, linewidth=0.5, color=_draw_col)
            else:
                _ax.plot(_x, _y, linewidth=0.5, color=_draw_col)
        elif _draw_func == "bar":
            _ax.bar(_x, _y, linewidth=0.5)
        elif _draw_func == "hist":
            _ax.hist(_y, bins=10)

        _ax.grid(color="k", alpha=0.2, linestyle="-.", linewidth=0.5)
        _ax.tick_params(axis='x', labelrotation=45)

    mo.vstack([_fig, mo.md("Fourier...")])
    return (
        __harmonics_windowed,
        data_float,
        dt,
        filtered_harmonics,
        frequency_filter_threshold,
        normalize,
        normalize_factor,
    )


@app.cell
def __(SAMPLE_RATE, fftu, filtered_harmonics, mo):
    _bins = [i['bin_idx'] for i in filtered_harmonics]

    mo.vstack([mo.md(f"{len(filtered_harmonics)} harmonics meet our criteria: {[fftu.bin_to_freq(SAMPLE_RATE, bin, 4096) for bin in _bins]}"),
               mo.md(f"...")
              ])
    return


if __name__ == "__main__":
    app.run()
