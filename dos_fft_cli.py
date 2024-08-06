import fft_utils as fftu
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import math
import ggwave as ggw


SAMPLES_FILE = "waver_abc_16k_16bit.wav"


def main():
    def plot_spectrum(window_pos: float):
        bytes_per_sample, sample_rate, data_unpacked = fftu.sound_from_wav_file(SAMPLES_FILE, chunk=0)

        print(f"Loaded file {SAMPLES_FILE}, bytes per sample {bytes_per_sample}, sampling rate {sample_rate}")

        match bytes_per_sample:
            case 4:
                normalize_factor = np.iinfo(np.int32).max
            case 2:
                normalize_factor = np.iinfo(np.int16).max
            case 1:
                normalize_factor = np.iinfo(np.int8).max
        normalize = lambda x: x / normalize_factor
        data_to_plot = normalize(data_unpacked)

        print(f"Normalizing with a factor of {normalize_factor}")

        dt = 1.0 / sample_rate
        t = np.arange(0, len(data_to_plot), 1)

        window_size = 1024

        window = fftu.generate_window_n(t, window_size, window_pos)
        window_mask = window != 0

        _harmonics = fftu.fft((data_to_plot * window)[window_mask])
        harmonics = _harmonics[: (len(_harmonics) // 2) + 1]

        frequencies = [
            fftu.bin_to_freq(sample_rate, k, window_size)
            for k in range(len(harmonics))
        ]

        harmonics_mag = np.absolute(harmonics)
        harmonics_power = np.square(np.absolute(harmonics))

        frequency_filter_threshold = 2.0
        filtered_harmonics = fftu.filter_harmonics(
            harmonics_mag, frequency_filter_threshold
        )

        # For example 16000 / 2048 == 7.8125 (7.8125 * 6 == 46.875)
        fft_step = sample_rate / window_size

        to_plot = [
            {
                "data": data_to_plot,
                "ylim": (-1.1, 1.1),
                "title": "waveform",
                "draw_func": "plot",
                "xticks": [i * 2048 for i in range(10)],
                "xticklabels": [i * 2048 for i in range(10)],
            },
            {
                "data": window,
                "ylim": (-0.1, 1.1),
                "title": "window",
                "draw_func": "plot",
                "xticks": [i * 2048 for i in range(10)],
                "xticklabels": [i * 2048 for i in range(10)],
            },
            {
                "data": (window * data_to_plot)[window_mask],
                "x": np.arange(len(window))[window_mask],
                "ylim": (-1.1, 1.1),
                "title": "waveform windowed",
                "draw_func": "plot",
                "xlabel": "samples",
            },
            {
                "data": harmonics_mag,
                "x": frequencies,
                "ylim": (0.0, 20.0),
                "xlim": (1800, 3400),
                "title": "DFT",
                "draw_func": "bar",
                "xlabel": "[Hz]",
                # Every Nth frequency.
                "xticks": [
                    freq for freq in
                    np.arange(ggw.GGWAVE_PROTO_BASE_FREQ, ggw.GGWAVE_PROTO_BASE_FREQ + 32 * ggw.GGWAVE_PROTO_DELTA_FREQ, ggw.GGWAVE_PROTO_DELTA_FREQ)
                ],
                "xticklabels": [
                    "{:.2f}".format(freq) for freq in
                    np.arange(ggw.GGWAVE_PROTO_BASE_FREQ, ggw.GGWAVE_PROTO_BASE_FREQ + 32 * ggw.GGWAVE_PROTO_DELTA_FREQ, ggw.GGWAVE_PROTO_DELTA_FREQ)
                ],
                "xticksminor": fft_step,
                "hlines": frequency_filter_threshold,
            },
            {
                "data": harmonics_power,
                "x": frequencies,
                #"ylim": (0.0, 100.0),
                "xlim": (1800, 3400),
                "title": "DFT - harmonics power",
                "draw_func": "bar",
                "xlabel": "[Hz]",
                # Every Nth frequency.
                "xticks": [
                    freq for freq in
                    np.arange(ggw.GGWAVE_PROTO_BASE_FREQ, ggw.GGWAVE_PROTO_BASE_FREQ + 32 * ggw.GGWAVE_PROTO_DELTA_FREQ, ggw.GGWAVE_PROTO_DELTA_FREQ)
                ],
                "xticklabels": [
                    "{:.2f}".format(freq) for freq in
                    np.arange(ggw.GGWAVE_PROTO_BASE_FREQ, ggw.GGWAVE_PROTO_BASE_FREQ + 32 * ggw.GGWAVE_PROTO_DELTA_FREQ, ggw.GGWAVE_PROTO_DELTA_FREQ)
                ],
                "xticksminor": fft_step,
                "hlines": frequency_filter_threshold,
            },
        ]

        upscale = 4
        fig, axs = plt.subplots(len(to_plot), figsize=(upscale * 10, upscale * 20))
        plt.subplots_adjust(hspace=0.8)

        for ax, data in zip(axs, to_plot):
            # Remove the keys that the set() function doesn't recognize.
            y = data.pop("data")
            x = data.pop("x", np.arange(len(y)))
            draw_func = data.pop("draw_func")
            if "hlines" in data:
                hlines = data.pop("hlines")
                ax.axhline(hlines, color="blue", linewidth=0.5)
            if "highlight" in data:
                highlight = data.pop("highlight")
                ax.axvspan(*highlight, color="blue", alpha=0.15, label="window")
            if "xticksminor" in data:
                xticksminor = data.pop("xticksminor")
                ax.xaxis.set_minor_locator(MultipleLocator(xticksminor))
                ax.tick_params(which='minor', length=6.0)
                ax.tick_params(which='major', length=15.0, width=2.0)

            if "draw_style" in data:
                draw_style = data.pop("draw_style")
            else:
                draw_style = None

            if "draw_col" in data:
                draw_col = data.pop("draw_col")
            else:
                draw_col = "black"

            # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set.html
            ax.set(**data)

            if draw_func == "plot":
                if draw_style:
                    ax.plot(
                        x,
                        y,
                        draw_style,
                        markersize=1,
                        linewidth=0.5,
                        color=draw_col,
                    )
                else:
                    ax.plot(x, y, linewidth=0.5, color=draw_col)
            elif draw_func == "bar":
                ax.bar(x, y, linewidth=0.5)
            elif _draw_func == "hist":
                ax.hist(y, bins=10)

            ax.grid(color="k", alpha=0.2, linestyle="-.", linewidth=0.5)
            ax.tick_params(axis="x", labelrotation=45)

        name = f"out_{'%03d' % int(window_pos * 100)}.png"
        plt.savefig(name)
        print(f"saving {name}")

    for i in np.arange(0, 0.8, 0.4):
        plot_spectrum(i)


if __name__ == "__main__":
    main()
