import struct
import numpy as np
import simpleaudio as sa
from pathlib import Path
from config import *

INPUT_FILE = SAMPLES_FILE

data_unpacked = np.asarray([d[0] for d in struct.iter_unpack("<H", INPUT_FILE.read_bytes())])

data_float = []

to_float = lambda x: x / 8192.0
data_float = to_float(data_unpacked)

samples_count = len(data_unpacked)

seconds = samples_count / FREQUENCY

print(f"Samples {samples_count}, length {seconds}s")

t = np.linspace(0, seconds, int(seconds * FREQUENCY), False)
soundwave = data_float * np.iinfo(np.int16).max
audio = soundwave.astype(np.int16)

play_obj = sa.play_buffer(audio, 1, BYTES_PER_SAMPLE, FREQUENCY)
play_obj.wait_done()
