import marimo

__generated_with = "0.4.3"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    from pathlib import Path
    import struct

    import numpy as np
    import numpy.typing as npt
    import math
    import matplotlib.pyplot as plt
    import matplotlib
    import wave
    import base64
    import pandas as pd
    import plotnine as p9

    import fft_functions as fft

    import importlib

    importlib.reload(fft)

    mo.md("# Fourier Transform")
    return (
        Path,
        base64,
        fft,
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
    mo.md(r"""Here is the simplified DFT. The result will be a set of complex numbers. Think of those complex numbers like of a vector $A -j B$. $A$ is proportional to how much of the $\cos$ of that specific frequency is present in the final signal, with $B$ being proportional to how much $\sin$ is there.

    \[
    X_k = \sum_{n=0}^{N-1} x_n [\cos(2\pi\frac{n}{N}k) -j \sin(2\pi\frac{n}{N}k)]
    \]

    N - is the total samples count.

    Total contribution of a specific frequency is taken from $\sqrt{A^2 + B^2}$. The phase shift can be computed from $atan2(\frac{B}{A})$.
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
    mo.vstack([mo.md(f"{samples_count_slider}"),
               mo.md(f"{phase_shift_slider}"),
               mo.md("Notice we're using fully periodic signals here. That's what Fourier Transform expects. If one of the harmonics isn't periodic, the FT will behave strangely. That's why in practice we use windowing."),
               mo.md("The spectrum of the signal is reflected. That's because of the negative frequencies the transform uses. It's normal to just ignore frequencies above the Nyquist frequency.")
              ])
    return phase_shift_slider, samples_count_slider


@app.cell
def __(
    fft,
    importlib,
    np,
    pd,
    phase_shift_slider,
    plt,
    samples_count_slider,
):
    from matplotlib.ticker import MultipleLocator, AutoMinorLocator

    importlib.reload(fft)


    time_base, harmonic01 = fft.Waveform(
        frequency=4,
        amplitude=1,
        phase_shift=phase_shift_slider.value,
        resolution=int(samples_count_slider.value),
    ).get_wave()
    _, harmonic02 = fft.Waveform(
        frequency=12, amplitude=2, resolution=int(samples_count_slider.value)
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

    # Here we have the information split over real and imaginary part, depending
    # on if the harmonics resemble cos or sin more.
    harmonics = fft.dft(waveform_pd["sum"])
    # Filter out the almost 0 values. It's especially important for computing the
    # phase shift, with arctan2. That's because two very small number, divided by
    # each other, will result in a legit value: 0.00000002/0.00000001 = 2.
    harmonics[:] = list(map(lambda c: c if abs(c) > 0.0001 else 0.0, harmonics))
    # abs for complex computes magnituted
    harmonics_mag = abs(harmonics)
    harmonics_phase = np.array(
        list(map(lambda c: np.arctan2(c.imag, c.real), harmonics))
    )

    # ax.set_xticks(list(ax.get_xticks()) + extraticks)

    _to_plot = [
        {
            "data": waveform_pd["harmonic01"],
            "y_lim": (
                waveform_pd["harmonic01"].min() - 0.4,
                waveform_pd["harmonic01"].max() + 0.4,
            ),
            "y_ticks": {"minor": 0.5, "major": 1},
            "x_ticks": {"minor": np.pi * 2, "major": np.pi * 4},
            "title": "harmonic 1",
            "draw_func": "plot",
        },
        {
            "data": waveform_pd["harmonic02"],
            "y_lim": (
                waveform_pd["harmonic02"].min() - 0.4,
                waveform_pd["harmonic02"].max() + 0.4,
            ),
            "y_ticks": {"minor": 0.5, "major": 1},
            "x_ticks": {"minor": np.pi * 2, "major": np.pi * 4},
            "title": "harmonic 2",
            "draw_func": "plot",
        },
        {
            "data": waveform_pd["harmonic01"] + waveform_pd["harmonic02"],
            "y_lim": (
                waveform_pd["sum"].min() - 0.4,
                waveform_pd["sum"].max() + 0.4,
            ),
            "y_ticks": {"minor": 0.5, "major": 1},
            "x_ticks": {"minor": np.pi * 2, "major": np.pi * 4},
            "title": "combined waveform",
            "draw_func": "plot",
        },
        {
            "data": harmonics_mag,
            "y_lim": (harmonics_mag.min() - 20, harmonics_mag.max() + 20),
            "y_ticks": {"minor": 25, "major": 50},
            "x_ticks_extra": {},
            "title": "harmonics magnitude",
            "draw_func": "bar",
        },
        {
            "data": harmonics_phase,
            "y_lim": (harmonics_phase.min() - 0.4, harmonics_phase.max() + 0.4),
            "y_ticks": {"minor": 0.5, "major": 1},
            "title": "harmonics phase",
            "draw_func": "bar",
        },
    ]

    _fig, _axs = plt.subplots(len(_to_plot), figsize=(8, 16))
    plt.subplots_adjust(hspace=0.8)

    for _i, _data in enumerate(_to_plot):
        # vertical axis
        try:
            _axs[_i].set_ylim(_data["y_lim"])
        except KeyError:
            pass

        # horizontal axis
        _x = np.arange(len(_data["data"]))
        try:
            _x = _data["x"]
        except KeyError:
            pass

        # axes ticks
        try:
            _axs[_i].yaxis.set_major_locator(
                MultipleLocator(_data["y_ticks"]["major"])
            )
        except KeyError:
            pass
        try:
            _axs[_i].yaxis.set_minor_locator(
                MultipleLocator(_data["y_ticks"]["minor"])
            )
        except KeyError:
            pass
        try:
            _axs[_i].xaxis.set_major_locator(
                MultipleLocator(_data["x_ticks"]["major"])
            )
        except KeyError:
            pass
        try:
            _axs[_i].xaxis.set_minor_locator(
                MultipleLocator(_data["x_ticks"]["minor"])
            )
        except KeyError:
            pass

        # type of a plot
        try:
            if _data["draw_func"] == "plot":
                _axs[_i].plot(_x, _data["data"], linewidth=0.5)
            elif _data["draw_func"] == "bar":
                _axs[_i].bar(_x, _data["data"], linewidth=0.5)
            elif _data["draw_func"] == "hist":
                _axs[_i].hist(_data["data"], bins=10)
        except KeyError:
            pass

        _axs[_i].set(xlabel="sample", ylabel="val", title=_data["title"])
        _axs[_i].grid(color="k", alpha=0.2, linestyle="-.", linewidth=0.5)


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
    mo.md(r"""
    If the transform's input data isn't periodic you might see spectral leakage and aliasing effects.

    In order to force input data to be periodic, windowing is used.
    """)
    return


@app.cell
def __(MultipleLocator, fft, np, plt, time_base, waveform):
    # Window out the input signal, to ensure a periodic input data.
    window = fft.generate_window(time_base, 0.6, 0.3, False)
    # Create an array of bools.
    window_mask = window != 0

    harmonics_windowed = fft.dft(
        (waveform * window)[window_mask]
    )
    windowed_harmonics_mag = list(map(abs, harmonics_windowed))
    windowed_harmonics_phase = list(map(lambda c: np.arctan2(c.imag, c.real), harmonics_windowed))


    _fig, _axs = plt.subplots(5, figsize=(14, 14))

    _axs[0].set_ylim([waveform.min() - 0.4, waveform.max() + 0.4])
    _axs[0].plot(time_base, waveform, linewidth=0.7, linestyle="solid", marker="o")
    # _axs[0].set(xlabel='sample', ylabel='val', title='Soundwave plot')
    _axs[0].grid(color="k", alpha=0.2, linestyle="-.", linewidth=0.5)

    _axs[1].set_ylim([window.min() - 0.4, window.max() + 0.4])
    _axs[1].plot(time_base, window, linewidth=0.7, linestyle="solid", marker="o")
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
def __(fft, harmonics, mo):
    def result_to_table(data: dict) -> str:
        lines = []
        for element in data:
            line = f"{element.get('idx'): >5} | {element.get('value')}"
            lines.append(line)

        return lines

    mo.vstack(
        [
            mo.md(
                f"""
         Checked {len(harmonics)} possible harmonics:
         """
            ),
            result_to_table(fft.filter_harmonics(harmonics)),
        ]
    )
    return result_to_table,


@app.cell
def __(mo):
    mo.md("The spectrum graph is mirrored. That's because the samples of the originally sampled signal match with the sinusoid of those higher frequencies too. The frequencies in the middle is the Nyquist Rate.")
    return


@app.cell
def __(mo):
    mo.md("## Analysis of recorded data")
    return


@app.cell
def __(Path, base64, fft, mo, np, struct, wave):
    SAMPLES_FILE = "guitar_string_D.wav"


    if Path(SAMPLES_FILE).suffix == ".wav":
        with wave.open(SAMPLES_FILE, 'r') as wf:
            _wav_data = wf.readframes(wf.getnframes())
        _data_unpacked = np.frombuffer(_wav_data, dtype=np.int16)
        BYTES_PER_SAMPLE = wf.getsampwidth()
        SAMPLE_RATE = wf.getframerate()
    elif Path(SAMPLES_FILE).suffix == ".bin":
        # Lets make a wave file out of raw samples.
        _data_unpacked = np.asarray(
            [d[0] for d in struct.iter_unpack("<h", Path(SAMPLES_FILE).read_bytes())]
        )
        SAMPLE_RATE, BYTES_PER_SAMPLE = fft.params_from_file_name(SAMPLES_FILE)
        fft.raw_data_to_wave(_data_unpacked,
                             str(Path(SAMPLES_FILE).with_suffix(".wav")),
                             SAMPLE_RATE,
                             BYTES_PER_SAMPLE
                            )
        _wav_data = Path(SAMPLES_FILE).with_suffix(".wav").read_bytes()
    else:
        assert False, "yikes"

    data_unpacked = _data_unpacked

    # TODO(michalc): delete this, when https://github.com/marimo-team/marimo/issues/632 gets fixed
    _wav_base64 = base64.b64encode(_wav_data).decode("utf-8")

    # with open(str(Path(SAMPLES_FILE).with_suffix(".wav")), "rb") as _p:
    mo.vstack(
        [
            mo.md(f"Example sound file {SAMPLES_FILE}:"),
            #mo.audio(src="samples.wav"),
            # TODOD(michalc): delete this, when https://github.com/marimo-team/marimo/issues/632 gets fixed
            mo.Html(
                f"""
                <audio controls>
                    <source src="data:audio/wav;base64,{_wav_base64}" type="audio/wav">
                </audio>
                """
            )
        ]
    )
    return BYTES_PER_SAMPLE, SAMPLES_FILE, SAMPLE_RATE, data_unpacked, wf


@app.cell
def __(BYTES_PER_SAMPLE, SAMPLE_RATE, fft, mo):
    mo.vstack([mo.md(f"Loaded the {SAMPLE_RATE}Hz, {BYTES_PER_SAMPLE} byte sample data."),
               mo.md(f"The current cut off frequency is {fft.bin_to_freq(SAMPLE_RATE, 24, 1024)} Hz")
              ])
    return


@app.cell
def __(MultipleLocator, SAMPLE_RATE, data_unpacked, fft, np, plt):
    normalize_factor = np.iinfo(np.uint16).max
    normalize = lambda x: x / normalize_factor
    data_float = normalize(data_unpacked)

    _data_to_plot = data_float
    dt = 1.0 / SAMPLE_RATE
    _t = np.arange(0, len(_data_to_plot), 1)

    # Window out the input signal, to ensure a periodic input data.
    _window = fft.generate_window_n(_t, 1024, 0.1)
    # Create an array of bools.
    _window_mask = _window != 0

    _harmonics_windowed = fft.fft(
        (data_float * _window)[_window_mask]
    )
    _frequencies = [ fft.bin_to_freq(SAMPLE_RATE, k, 1024) for k in range(len(_harmonics_windowed)) ]
    _frequency_bin_of_interest_max = 24

    print(_harmonics_windowed)
    _windowed_harmonics_mag = list(map(abs, _harmonics_windowed))

    print(fft.filter_harmonics(_windowed_harmonics_mag, 1.2))
    print(np.histogram(_windowed_harmonics_mag, bins=10))

    _to_plot = [
        {"data": _data_to_plot,
         "y_lim": (-1.1, 1.1),
         "title": "waveform",
         "draw_func": "plot"
        },
        {"data": _window,
         "y_lim": (-0.1, 1.1),
         "title": "window",
         "draw_func": "plot"
        },
        {"data": (_window * _data_to_plot)[_window_mask],
         "x": np.arange(len(_window))[_window_mask],
         "y_lim": (-0.02, 0.02),
         "title": "waveform windowed",
         "draw_func": "plot",
        },
        {"data": _windowed_harmonics_mag[:len(_windowed_harmonics_mag)//2],
         "title": "histogram",
         "draw_func": "hist",
        },
        {"data": _windowed_harmonics_mag[:_frequency_bin_of_interest_max],
         "x": _frequencies[:_frequency_bin_of_interest_max],
         "x_ticks": {"major": _frequencies[:_frequency_bin_of_interest_max][1]},
         "title": "DFT",
         "draw_func": "bar",
        }
    ]

    _fig, _axs = plt.subplots(
        len(_to_plot),
        figsize=(10,16)
    )

    plt.subplots_adjust(hspace=0.8)

    for _i, _data in enumerate(_to_plot):
        # vertical axis
        try:
            _axs[_i].set_ylim(_data["y_lim"])
        except KeyError:
            pass

        # horizontal axis
        _x = np.arange(len(_data["data"]))
        try:
            _x = _data["x"]
        except KeyError:
            pass

        try:
            _axs[_i].xaxis.set_major_locator(
                MultipleLocator(_data["x_ticks"]["major"])
            )
        except KeyError:
            pass
        try:
            _axs[_i].xaxis.set_minor_locator(
                MultipleLocator(_data["x_ticks"]["minor"])
            )
        except KeyError:
            pass

        # type of a plot
        try:
            if _data["draw_func"] == "plot":
                _axs[_i].plot(_x, _data["data"], linewidth=0.5)
                _axs[_i].yaxis.set_major_locator(MultipleLocator(0.5))
                _axs[_i].yaxis.set_minor_locator(MultipleLocator(.25))
            elif _data["draw_func"] == "bar":
                _axs[_i].bar(_x, _data["data"], linewidth=0.5)
            elif _data["draw_func"] == "hist":
                _axs[_i].hist(_data["data"], bins=10)
        except KeyError:
            pass

        _axs[_i].set(xlabel="sample", ylabel="val", title=_data["title"])
        _axs[_i].grid(color="k", alpha=0.2, linestyle="-.", linewidth=0.5)

    _fig
    return data_float, dt, normalize, normalize_factor


if __name__ == "__main__":
    app.run()
