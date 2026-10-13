"""Stage 2: compare each sample against the baseline, byte by byte.

Prints (a) readable ASCII markers in the baseline with their offsets and
(b) for same-size samples, every differing byte range.
"""
import os
import re
import sys

S = os.path.join(os.path.dirname(__file__), "samples")


def load(n):
    with open(os.path.join(S, n + ".czi"), "rb") as f:
        return f.read()


base = load("baseline")

print("== ASCII markers in baseline (len >= 6) ==")
for m in re.finditer(rb"[\x20-\x7e]{6,}", base):
    s = m.group().decode()
    print(f"  0x{m.start():05x}  {s[:70]}")


def ranges(a, b):
    out, start = [], None
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            if start is None:
                start = i
        elif start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, min(len(a), len(b))))
    return out


names = sys.argv[1:] or ["fill_7", "one_pixel", "location_100_50",
                         "channel_1", "z_3", "width_32", "height_32", "uint16"]
for n in names:
    d = load(n)
    print(f"\n== baseline vs {n}  (sizes {len(base)} / {len(d)}) ==")
    rs = ranges(base, d)
    # skip long runs of noise like GUIDs / timestamps by summarizing
    for a, b in rs[:25]:
        print(f"  0x{a:05x}-0x{b:05x}  base={base[a:b][:16].hex(' ')}  new={d[a:b][:16].hex(' ')}")
    if len(rs) > 25:
        print(f"  ... {len(rs) - 25} more ranges")
