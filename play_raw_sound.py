import struct
import numpy as np
import simpleaudio as sa
from pathlib import Path
from config import *
import sys


FREQUENCY = 16000
BYTES_PER_SAMPLE = 2


def play_pcm(pcm_file: str, frequency: int, sample_size: int):
    _data = Path(pcm_file).read_bytes()

    bytes_per_sample = BYTES_PER_SAMPLE

    if bytes_per_sample == 1:
        normalize_factor = np.iinfo(np.uint8).max
        data_unpacked = np.asarray(struct.unpack(f"<{len(_data)}b", _data))
    elif bytes_per_sample == 2:
        normalize_factor = np.iinfo(np.int16).max
        data_unpacked = np.asarray(struct.unpack(f"<{len(_data)//2}h", _data))
    elif bytes_per_sample == 4:
        normalize_factor = np.iinfo(np.int32).max
        data_unpacked = np.asarray(struct.unpack(f"<{len(_data)//4}i", _data))

    samples_count = len(data_unpacked)

    normalize_factor = np.iinfo(np.int16).max
    to_float = lambda x: x / normalize_factor
    data_float = to_float(data_unpacked)

    seconds = samples_count / frequency

    print(f"samples: {samples_count}, (min {data_unpacked.min()},  max {data_unpacked.max()})")
    print(f"float: (min {data_float.min()}, max {data_float.max()}")
    print(f"length {seconds}s")

    t = np.linspace(0, seconds, samples_count, False)
    soundwave = data_float * normalize_factor
    # You can change int16 to uint16 and the audio result is different
    # but the way it plays seems to be the same.
    audio = soundwave.astype(np.int16)

    play_obj = sa.play_buffer(audio, 1, sample_size, frequency)
    play_obj.wait_done()

if __name__ == "__main__":
    play_pcm(sys.argv[1], FREQUENCY, BYTES_PER_SAMPLE)
