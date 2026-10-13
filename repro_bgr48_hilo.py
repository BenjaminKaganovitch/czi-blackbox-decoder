"""Minimal reproduction: official reader returns wrong pixels for Bgr48 + HiLoByteUnpack.

Writes a 10x30 Bgr48 image with zstd1 + HiLoByteUnpack using pylibCZIrw,
reads it back with pylibCZIrw, and compares to the array that was written.
"""
import numpy as np
import pylibCZIrw.czi as czi

a = np.random.default_rng(3).integers(0, 65535, (10, 30, 3), dtype=np.uint16)
path = "repro_bgr48_hilo.czi"
with czi.create_czi(path, exist_ok=True,
                    compression_options="zstd1:ExplicitLevel=3;PreProcess=HiLoByteUnpack") as w:
    w.write(a, plane={"C": 0})
with czi.open_czi(path) as r:
    b = r.read(plane={"C": 0})

print("round trip exact:", np.array_equal(a, b))
flat_a, flat_b = a.reshape(-1), b.reshape(-1)
match = np.nonzero(flat_a == flat_b)[0]
print(f"correct 16-bit values: {len(match)} of {flat_a.size}; "
      f"they are positions {match.min()}..{match.max()} (= the first `width` values)"
      if len(match) else "no values correct")
