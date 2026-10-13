"""Stage 3: an independent CZI reader built only from what the diffs revealed.

No vendor code is used here: just struct + numpy. Every constant below is
annotated with the experiment that revealed it.

Inferred layout (all little-endian):

Segment header (32 bytes), found by the ASCII markers at segment starts:
    0x00  16 bytes  segment id, e.g. b"ZISRAWFILE", zero padded
    0x10  int64     allocated size of segment body
    0x18  int64     used size of segment body

File header body (starts 0x20):
    0x34  int64     directory segment position   (moved when tiles were added)
    0x3C  int64     metadata segment position
    (0x10..0x30 of the body are two per-file GUIDs: the "noise" in every diff)

Directory body:
    +0      int32   number of entries          (1 vs 2 in two_channels)
    +4      124 bytes reserved
    +128    entries, back to back

Directory entry ("DV" schema):
    +0   2 bytes  b"DV"
    +2   int32    pixel type  (0 Gray8, 1 Gray16, 2 Gray32Float, 3 Bgr24, 4 Bgr48)
    +6   int64    file position of the subblock segment
    +14  int32    file part
    +18  int32    compression (0 = none)
    +22  byte     pyramid type
    +23  5 bytes  spare
    +28  int32    dimension count
    +32  dimension entries, 20 bytes each:
         4-char name, int32 start, int32 size, float32 start coord, int32 stored size
         (location_100_50 flipped X start to 100 and Y start to 50;
          channel_1 flipped the C start; z_3 flipped the Z start)

Subblock body:
    +0   int32    metadata size
    +4   int32    attachment size
    +8   int64    data size
    +16  a copy of the directory entry
    fixed header area is 256 bytes, then metadata XML, then pixel data
    (one_pixel: row 3, col 5 landed at data offset 3*16+5 -> row-major)
"""
import struct
import numpy as np

PIXEL_TYPES = {
    0: (np.uint8, 1),       # Gray8
    1: (np.uint16, 1),      # Gray16
    2: (np.float32, 1),     # Gray32Float
    3: (np.uint8, 3),       # Bgr24
    4: (np.uint16, 3),      # Bgr48
}
DIM_ENTRY_SIZE = 20
DIR_ENTRY_FIXED = 32
SUBBLOCK_FIXED_AREA = 256


def _zstd(data, expected):
    import zstandard  # generic open-source decompressor, not vendor code
    return zstandard.ZstdDecompressor().decompress(data, max_output_size=expected)


class Tile:
    def __init__(self, pixel_type, file_pos, compression, dims):
        self.pixel_type = pixel_type
        self.file_pos = file_pos
        self.compression = compression
        self.dims = dims  # name -> (start, size, stored_size)

    def plane(self):
        return {k: v[0] for k, v in self.dims.items() if k not in ("X", "Y", "M")}


def _segment(buf, pos):
    seg_id = buf[pos:pos + 16].rstrip(b"\0").decode("ascii")
    alloc, used = struct.unpack_from("<qq", buf, pos + 16)
    return seg_id, pos + 32, used


def _dir_entry(buf, p):
    assert buf[p:p + 2] == b"DV", f"unknown entry schema {buf[p:p+2]!r}"
    pixel_type, file_pos, _part, compression = struct.unpack_from("<iqii", buf, p + 2)
    ndims = struct.unpack_from("<i", buf, p + 28)[0]
    dims = {}
    q = p + DIR_ENTRY_FIXED
    for _ in range(ndims):
        name = buf[q:q + 4].rstrip(b"\0").decode("ascii")
        start, size, _coord, stored = struct.unpack_from("<iifi", buf, q + 4)
        dims[name] = (start, size, stored)
        q += DIM_ENTRY_SIZE
    return Tile(pixel_type, file_pos, compression, dims), q - p


class CziFile:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.buf = f.read()
        seg_id, body, _ = _segment(self.buf, 0)
        assert seg_id == "ZISRAWFILE", "not a CZI file"
        dir_pos, meta_pos = struct.unpack_from("<qq", self.buf, body + 0x34)
        self.metadata_pos = meta_pos

        seg_id, body, _ = _segment(self.buf, dir_pos)
        assert seg_id == "ZISRAWDIRECTORY"
        count = struct.unpack_from("<i", self.buf, body)[0]
        p = body + 128
        self.tiles = []
        for _ in range(count):
            tile, n = _dir_entry(self.buf, p)
            self.tiles.append(tile)
            p += n

    def tile_pixels(self, tile):
        seg_id, body, _ = _segment(self.buf, tile.file_pos)
        assert seg_id == "ZISRAWSUBBLOCK"
        meta_size, _att, data_size = struct.unpack_from("<iiq", self.buf, body)
        data_off = body + SUBBLOCK_FIXED_AREA + meta_size
        payload = self.buf[data_off:data_off + data_size]
        dtype, ch = PIXEL_TYPES[tile.pixel_type]
        w = tile.dims["X"][2]
        h = tile.dims["Y"][2]
        n_bytes = h * w * ch * np.dtype(dtype).itemsize

        if tile.compression == 0:
            raw = payload
        elif tile.compression == 5:
            # Payload began with 28 b5 2f fd: the standard zstd frame magic.
            raw = _zstd(payload, n_bytes)
        elif tile.compression == 6:
            # Payload began 03 01 00 then zstd magic: a small header whose first
            # byte is its own length; the next byte pair is (chunk type 1, value).
            # Value 1 appeared only when we asked for "HiLoByteUnpack".
            hdr_len = payload[0]
            hilo = hdr_len >= 3 and payload[1] == 1 and payload[2] == 1
            raw = _zstd(payload[hdr_len:], n_bytes)
            if hilo and np.dtype(dtype).itemsize == 2:
                # all low bytes first, then all high bytes -> interleave back
                half = len(raw) // 2
                inter = bytearray(len(raw))
                inter[0::2] = raw[:half]
                inter[1::2] = raw[half:]
                raw = bytes(inter)
        else:
            raise NotImplementedError(f"compression {tile.compression} not decoded yet")
        arr = np.frombuffer(raw, dtype=dtype, count=h * w * ch)
        return arr.reshape(h, w, ch)

    def read(self, plane):
        """Compose all tiles on a plane into one image covering their bounding box."""
        tiles = [t for t in self.tiles
                 if all(t.dims.get(k, (0,))[0] == v for k, v in plane.items())]
        if not tiles:
            raise KeyError(f"no tiles on plane {plane}")
        x0 = min(t.dims["X"][0] for t in tiles)
        y0 = min(t.dims["Y"][0] for t in tiles)
        x1 = max(t.dims["X"][0] + t.dims["X"][1] for t in tiles)
        y1 = max(t.dims["Y"][0] + t.dims["Y"][1] for t in tiles)
        dtype, ch = PIXEL_TYPES[tiles[0].pixel_type]
        out = np.zeros((y1 - y0, x1 - x0, ch), dtype=dtype)
        for t in tiles:  # file order; later tiles drawn on top (to be verified)
            px = self.tile_pixels(t)
            xs, ys = t.dims["X"][0] - x0, t.dims["Y"][0] - y0
            out[ys:ys + px.shape[0], xs:xs + px.shape[1]] = px
        return out, (x0, y0, x1 - x0, y1 - y0)

    def planes(self):
        seen = []
        for t in self.tiles:
            p = t.plane()
            if p not in seen:
                seen.append(p)
        return seen
