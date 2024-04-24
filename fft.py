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
    import wave
    import base64

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
        mo,
        np,
        npt,
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
    mo.vstack([mo.md(f"{samples_count_slider}"), mo.md(f"{phase_shift_slider}"), mo.md("Notice we're using fully periodic signals here. That's what Fourier Transform expects. If one of the harmonics isn't periodic, the FT will behave strangely. That's why in practice we use windowing.")])
    return phase_shift_slider, samples_count_slider


@app.cell
def __(fft, importlib, np, phase_shift_slider, plt, samples_count_slider):
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
    # _, waveform03 = Waveform(cycles=5).get_wave()

    waveform = harmonic01 + harmonic02


    # Here we have the information split over real and imaginary part, depending
    # on if the harmonics resemble cos or sin more.
    harmonics = fft.dft(time_base, waveform)
    # Filter out the almost 0 values. It's especially important for computing the
    # phase shift, with arctan2. That's because two very small number, divided by
    # each other, will result in a legit value: 0.00000002/0.00000001 = 2.
    harmonics[:] = list(map(lambda c: c if abs(c) > 0.0001 else 0.0, harmonics))
    # abs for complex computes magnituted
    harmonics_mag = list(map(abs, harmonics))
    harmonics_phase = list(map(lambda c: np.arctan2(c.imag, c.real), harmonics))


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
        _axs[i].plot(time_base, data, linewidth=0.7, linestyle="solid", marker="o")
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
        data,
        data_to_plot,
        harmonic01,
        harmonic02,
        harmonics,
        harmonics_mag,
        harmonics_phase,
        i,
        time_base,
        waveform,
    )


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
    window = fft.generate_window(time_base, 0.5, 0.3)
    # Create an array of bools.
    window_mask = window != 0

    harmonics_windowed = fft.dft(
        np.arange(len(waveform[window_mask])), (waveform * window)[window_mask]
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
def __(harmonics, mo):
    def filter_harmonics(harmonics):
        valid_harmonics = []
        for i, h in enumerate(harmonics):
            if abs(h) > 0.001 or abs(h) < -0.001:
                valid_harmonics.append({"idx": i, "value": h})

        return valid_harmonics


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
            result_to_table(filter_harmonics(harmonics)),
        ]
    )
    return filter_harmonics, result_to_table


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

    SAMPLE_RATE, BYTES_PER_SAMPLE = fft.params_from_file_name(SAMPLES_FILE)

    if Path(SAMPLES_FILE).suffix == ".wav":
        with wave.open(SAMPLES_FILE, 'r') as wf:
            _wav_data = wf.readframes(wf.getnframes())
        _data_unpacked = np.frombuffer(_wav_data, dtype=np.int16)
    elif Path(SAMPLES_FILE).suffix == ".bin":
        # Lets make a wave file out of raw samples.
        _data_unpacked = np.asarray(
            [d[0] for d in struct.iter_unpack("<h", Path(SAMPLES_FILE).read_bytes())]
        )
        fft.raw_data_to_wave(_data_unpacked,
                             str(Path(SAMPLES_FILE).with_suffix(".wav")),
                             SAMPLE_RATE,
                             BYTES_PER_SAMPLE
                            )
    else:
        assert False, "yikes"

    data_unpacked = _data_unpacked

    # TODO(michalc): delete this, when https://github.com/marimo-team/marimo/issues/632 gets fixed
    _wav_data = Path(SAMPLES_FILE).with_suffix(".wav").read_bytes()
    _wav_base64 = base64.b64encode(_wav_data).decode("utf-8")

    # with open(str(Path(SAMPLES_FILE).with_suffix(".wav")), "rb") as _p:
    mo.vstack(
        [
            mo.md("Example sound file:"),
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
def __(BYTES_PER_SAMPLE, SAMPLE_RATE, mo):
    mo.md(f"Loaded the {SAMPLE_RATE}Hz, {BYTES_PER_SAMPLE} byte sample data.")
    return


@app.cell
def __(MultipleLocator, SAMPLE_RATE, data_unpacked, np, plt):
    normalize_factor = np.iinfo(np.uint16).max
    normalize = lambda x: x / normalize_factor
    data_float = normalize(data_unpacked)

    _data_to_plot = data_float
    dt = 1.0 / SAMPLE_RATE
    _t = np.arange(0, len(_data_to_plot), 1)

    #_harmonics_windowed = fft.dft(
        #np.arange(len(_t)), (data_unpacked * window)[window_mask]
    #)

    _fig, _axs = plt.subplots(len((_data_to_plot, [])), figsize=(8,4))

    for _i, _data in enumerate((_data_to_plot,)):
        _axs[_i].set_ylim([-1.1, 1.1])
        _axs[_i].yaxis.set_major_locator(MultipleLocator(0.5))
        _axs[_i].yaxis.set_minor_locator(MultipleLocator(.25))
        _axs[_i].plot(_t, _data, linewidth=0.1)
        _axs[_i].set(xlabel="sample", ylabel="val", title="Soundwave plot")
        _axs[_i].grid(color="k", alpha=0.2, linestyle="-.", linewidth=0.5)

    _fig
    return data_float, dt, normalize, normalize_factor


if __name__ == "__main__":
    app.run()
