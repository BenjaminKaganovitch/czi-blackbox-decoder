"""Stage 4: the answer key.

Generate random files with the official writer, read them with both the
official reader and our parser, and require identical pixels. Disagreements
are saved so they can be studied, exactly like failing tests in the pipeline.
"""
import os
import sys
import json
import shutil
import numpy as np
import pylibCZIrw.czi as czi
from our_parser import CziFile

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
ALLOW_OVERLAP = "--overlap" in sys.argv
COMPRESSION = None
if "--compression" in sys.argv:
    COMPRESSION = sys.argv[sys.argv.index("--compression") + 1]
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "fuzz")
FAIL = os.path.join(HERE, "failures")
shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(WORK)
os.makedirs(FAIL, exist_ok=True)

TYPES = [(np.uint8, 1), (np.uint16, 1), (np.float32, 1), (np.uint8, 3), (np.uint16, 3)]
rng = np.random.default_rng(1234)


def random_tile(dtype, ch):
    h, w = rng.integers(1, 65, size=2)
    if dtype == np.float32:
        a = rng.standard_normal((h, w, ch)).astype(np.float32) * 1000
    else:
        a = rng.integers(0, np.iinfo(dtype).max, size=(h, w, ch), dtype=dtype)
    return a


def non_overlapping_location(placed, h, w):
    for _ in range(200):
        x, y = (int(v) for v in rng.integers(-50, 300, size=2))
        if all(x + w <= px or px + pw <= x or y + h <= py or py + ph <= y
               for px, py, pw, ph, _ in placed):
            return x, y
    return None


stats = {"files": 0, "planes": 0, "pass": 0, "fail": 0, "tiles": 0}
for i in range(N):
    dtype, ch = TYPES[rng.integers(len(TYPES))]
    path = os.path.join(WORK, f"f{i:04d}.czi")
    planes = {}
    with czi.create_czi(path, exist_ok=True, compression_options=COMPRESSION) as w:
        for c in range(int(rng.integers(1, 4))):
            for z in range(int(rng.integers(1, 3))):
                placed = []
                for _ in range(int(rng.integers(1, 5))):
                    a = random_tile(dtype, ch)
                    if ALLOW_OVERLAP:
                        loc = tuple(int(v) for v in rng.integers(-50, 120, size=2))
                    else:
                        loc = non_overlapping_location(placed, a.shape[0], a.shape[1])
                        if loc is None:
                            continue
                    placed.append((loc[0], loc[1], a.shape[1], a.shape[0], a))
                    w.write(a, location=loc, plane={"C": c, "Z": z})
                    stats["tiles"] += 1
                planes[(c, z)] = placed
    stats["files"] += 1

    ours = CziFile(path)
    with czi.open_czi(path) as r:
        for (c, z) in planes:
            stats["planes"] += 1
            mine, (x, y, w_, h_) = ours.read({"C": c, "Z": z})
            theirs = r.read(roi=(x, y, w_, h_), plane={"C": c, "Z": z})
            theirs = theirs.astype(mine.dtype, copy=False).reshape(mine.shape)
            ok = np.array_equal(mine, theirs, equal_nan=(mine.dtype == np.float32))
            if ok:
                stats["pass"] += 1
            else:
                stats["fail"] += 1
                diff = int((mine != theirs).any(axis=-1).sum())
                # Tie-breaker: we know what we wrote. Rebuild the plane from the
                # original arrays and see which reader matches the ground truth.
                truth = np.zeros_like(mine)
                for px, py, pw, ph, arr in planes[(c, z)]:
                    truth[py - y:py - y + ph, px - x:px - x + pw] = arr
                ours_ok = np.array_equal(mine, truth)
                theirs_ok = np.array_equal(theirs, truth)
                key = ("ours_right" if ours_ok and not theirs_ok else
                       "official_right" if theirs_ok and not ours_ok else "both_wrong")
                stats[key] = stats.get(key, 0) + 1
                shutil.copy(path, os.path.join(FAIL, os.path.basename(path)))
                print(f"MISMATCH {os.path.basename(path)} C={c} Z={z} "
                      f"dtype={np.dtype(dtype).name}x{ch} differing_px={diff} -> {key}")

print(json.dumps(stats, indent=1))
