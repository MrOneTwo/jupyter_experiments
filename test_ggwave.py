import ggwave as ggw
import fft_utils as fftu


def test_delimiter_start():
    data = [
        12, 1, 2, 24, 18, 1, 0, 12,
        21, 1, 4, 15, 21, 1, 2, 15,
        8,  0, 1, 11, 10, 0, 1, 21,
        20, 1, 0, 18, 17, 0, 1, 18,
    ]

    assert ggw.is_delimiter_start(data) == True


def test_freq_to_bin():
    sample_rate = 16000
    window_size = 2048
    fft_step = sample_rate / window_size

    print(f"freq step {sample_rate / window_size}")

    bin_idx = fftu.freq_to_bin(sample_rate, fft_step, window_size)

    assert bin_idx == 1


def test_bin_to_freq():
    sample_rate = 16000
    window_size = 2048

    freq = fftu.bin_to_freq(sample_rate, 1, window_size)

    assert freq == 7.8125
