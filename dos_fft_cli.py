import fft_utils as fftu
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


SAMPLES_FILE = "waver_abc_44k_32bit.wav"


def main():
    def plot_spectrum(window_pos: float):
        bytes_per_sample, sample_rate, data_unpacked = fftu.sound_from_wav_file(SAMPLES_FILE, chunk=0)

        match bytes_per_sample:
            case 4:
                normalize_factor = np.iinfo(np.int32).max
            case 2:
                normalize_factor = np.iinfo(np.int16).max
            case 1:
                normalize_factor = np.iinfo(np.int8).max

        normalize = lambda x: x / normalize_factor
        data_to_plot = normalize(data_unpacked)
        dt = 1.0 / sample_rate
        t = np.arange(0, len(data_to_plot), 1)

        window_size = 2048

        window = fftu.generate_window_n(t, window_size, window_pos)
        window_mask = window != 0

        _harmonics = fftu.fft((data_to_plot * window)[window_mask])
        harmonics = _harmonics[:len(_harmonics) // 2]

        frequencies = [
            fftu.bin_to_freq(sample_rate, k, window_size)
            for k in range(len(harmonics))
        ]

        harmonics_mag = list(map(abs, harmonics))

        frequency_filter_threshold = 20.0
        filtered_harmonics = fftu.filter_harmonics(
            harmonics_mag, frequency_filter_threshold
        )

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
                "ylim": (0.0, 60.0),
                "title": "DFT",
                "draw_func": "bar",
                "xlabel": "samples",
                # Every Nth frequency.
                "xticks": [freq for freq in np.arange(1875.0, 1875.0 + 32 * 46.875, 46.875)],
                "xticklabels": ["{:.2f}".format(freq) for freq in np.arange(1875.0, 1875.0 + 32 * 46.875, 46.875)],
                "hlines": frequency_filter_threshold,
            }
        ]

        fig, axs = plt.subplots(len(to_plot), figsize=(10, 16))
        plt.subplots_adjust(hspace=0.4)

        for ax, data in zip(axs, to_plot):
            y = data.pop("data")
            x = data.pop("x", np.arange(len(y)))
            draw_func = data.pop("draw_func")
            if "hlines" in data:
                hlines = data.pop("hlines")
                ax.axhline(hlines, color="blue", linewidth=0.5)

            ax.set(**data)

            match draw_func:
                case "plot":
                    ax.plot(x, y, linewidth=0.5)
                case "bar":
                    ax.bar(x, y, linewidth=0.5)
                case "hist":
                    pass


            ax.grid(color="k", alpha=0.2, linestyle="-.", linewidth=0.5)
            ax.tick_params(axis='x', labelrotation=45)

        name = f"out_{'%03d' % int(window_pos * 100)}.png"
        plt.savefig(name)
        print(f"saving {name}")

    for i in np.arange(0, 0.7, 0.05):
        plot_spectrum(i)


if __name__ == "__main__":
    main()
