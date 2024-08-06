GGWAVE_PROTO_BASE_FREQ = 1875.0
GGWAVE_PROTO_DELTA_FREQ = 46.875
GGWAVE_PROTO_DELIMITER_FREQS_COUNT = 32
GGWAVE_PROTO_DATA_FREQS_COUNT = 96


def is_delimiter_start(data: list) -> bool:
    assert len(data) == GGWAVE_PROTO_DELIMITER_FREQS_COUNT, (
        f"Expecting input list of length {GGWAVE_PROTO_DELIMITER_FREQS_COUNT}"
    )

    ret = True

    for i in range(0, 32, 2):
        if i % 2 == 0:
            # Previous value, even divided, is bigger than the next.
            ret = (data[i] / 3) > data[i + 1]
        elif i % 2 == 1:
            # Previous value, even divided, is smaller than the next.
            ret = (data[i] / 3) < data[i + 1]

        if ret == False:
            break

    return ret


def freqs_to_data(freqs: list, base_freq: float=1875.0, delta_freq: float=46.875):
    """
    ggwave transmits 6 4-bit chunks at the same time.
    Chunk 0 is between F0 and F0 + 15*dF.
    Chunk 1 is between F0 + 16*dF and F0 + 31*dF.
    Chunks end at F0 + 95*dF.
    The value encoded by a chunk is the N, in F0 + N*dF, masked by 0b1111.
    """
    data = bytearray()
    for freq in freqs:
        data.append(int(round((freq - base_freq) / delta_freq)))

    return data
