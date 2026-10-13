"""Stage 1: use the official writer as a black box to make controlled samples.

Each sample changes exactly one thing from the baseline, so any byte that
differs between baseline and sample is "about" that one thing.
"""
import os
import numpy as np
import pylibCZIrw.czi as czi

OUT = os.path.join(os.path.dirname(__file__), "samples")
os.makedirs(OUT, exist_ok=True)


def make(name, tiles, compression=None):
    """tiles: list of (array, location, plane) tuples."""
    path = os.path.join(OUT, name + ".czi")
    if os.path.exists(path):
        os.remove(path)
    with czi.create_czi(path, exist_ok=True, compression_options=compression) as w:
        for arr, loc, plane in tiles:
            w.write(arr, location=loc, plane=plane)
    return path


def img(h, w, dtype=np.uint8, fill=0):
    return np.full((h, w, 1), fill, dtype=dtype)


base = img(16, 16)

make("baseline", [(base, (0, 0), {"C": 0})])
make("width_32", [(img(16, 32), (0, 0), {"C": 0})])
make("height_32", [(img(32, 16), (0, 0), {"C": 0})])
make("fill_7", [(img(16, 16, fill=7), (0, 0), {"C": 0})])

one_px = img(16, 16)
one_px[3, 5, 0] = 200          # row 3, col 5
make("one_pixel", [(one_px, (0, 0), {"C": 0})])

make("uint16", [(img(16, 16, np.uint16, fill=0x1234), (0, 0), {"C": 0})])
make("location_100_50", [(base, (100, 50), {"C": 0})])
make("channel_1", [(base, (0, 0), {"C": 1})])
make("two_channels", [(base, (0, 0), {"C": 0}), (img(16, 16, fill=9), (0, 0), {"C": 1})])
make("z_3", [(base, (0, 0), {"C": 0, "Z": 3})])

for f in sorted(os.listdir(OUT)):
    print(f"{f:24s} {os.path.getsize(os.path.join(OUT, f)):8d} bytes")
