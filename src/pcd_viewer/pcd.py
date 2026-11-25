"""Minimal reader for Point Cloud Data (PCD) files.

Supports ``DATA ascii``, ``binary`` and ``binary_compressed`` with arbitrary
FIELDS/SIZE/TYPE/COUNT combinations. Returns a numpy structured array so that
custom fields (e.g. ``reflectivity``) are preserved.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_TYPE_CHARS = {"F": "f", "I": "i", "U": "u"}


@dataclass
class PCDHeader:
    version: str = ".7"
    fields: list[str] = field(default_factory=list)
    size: list[int] = field(default_factory=list)
    type: list[str] = field(default_factory=list)
    count: list[int] = field(default_factory=list)
    width: int = 0
    height: int = 1
    viewpoint: list[float] = field(default_factory=lambda: [0, 0, 0, 1, 0, 0, 0])
    points: int = 0
    data: str = "ascii"

    def dtype(self) -> np.dtype:
        """Little-endian numpy dtype for one point record."""
        items = []
        for name, size, typ, count in zip(
            self.fields, self.size, self.type, self.count, strict=True
        ):
            if typ not in _TYPE_CHARS:
                raise ValueError(f"Unsupported PCD TYPE {typ!r} for field {name!r}")
            base = f"<{_TYPE_CHARS[typ]}{size}"
            items.append((name, base, (count,)) if count > 1 else (name, base))
        return np.dtype(items)


def _parse_header(raw: bytes) -> tuple[PCDHeader, int]:
    """Parse the header; return it together with the byte offset of the data section."""
    header = PCDHeader()
    offset = 0
    while True:
        end = raw.find(b"\n", offset)
        if end == -1:
            raise ValueError("Unexpected end of file while reading PCD header")
        line = raw[offset:end].decode("ascii", errors="replace").strip()
        offset = end + 1
        if not line or line.startswith("#"):
            continue

        key, *values = line.split()
        key = key.upper()
        match key:
            case "VERSION":
                header.version = values[0]
            case "FIELDS" | "COLUMNS":
                header.fields = values
            case "SIZE":
                header.size = [int(v) for v in values]
            case "TYPE":
                header.type = [v.upper() for v in values]
            case "COUNT":
                header.count = [int(v) for v in values]
            case "WIDTH":
                header.width = int(values[0])
            case "HEIGHT":
                header.height = int(values[0])
            case "VIEWPOINT":
                header.viewpoint = [float(v) for v in values]
            case "POINTS":
                header.points = int(values[0])
            case "DATA":
                header.data = values[0].lower()
                break

    if not header.count:
        header.count = [1] * len(header.fields)
    if not header.points:
        header.points = header.width * header.height
    return header, offset


def lzf_decompress(data: bytes, expected_size: int) -> bytes:
    """Pure-Python LZF decompressor (as used by PCL's binary_compressed format)."""
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        ctrl = data[i]
        i += 1
        if ctrl < 32:  # literal run of ctrl + 1 bytes
            length = ctrl + 1
            out += data[i : i + length]
            i += length
        else:  # back reference
            length = ctrl >> 5
            ref = len(out) - ((ctrl & 0x1F) << 8) - 1
            if length == 7:
                length += data[i]
                i += 1
            ref -= data[i]
            i += 1
            length += 2
            if ref < 0:
                raise ValueError("Corrupt LZF data: invalid back reference")
            for _ in range(length):  # byte-wise copy, references may overlap
                out.append(out[ref])
                ref += 1
    if len(out) != expected_size:
        raise ValueError(f"LZF size mismatch: got {len(out)}, expected {expected_size}")
    return bytes(out)


def _read_binary_compressed(body: bytes, header: PCDHeader, dtype: np.dtype) -> np.ndarray:
    compressed_size, uncompressed_size = struct.unpack("<II", body[:8])
    raw = lzf_decompress(body[8 : 8 + compressed_size], uncompressed_size)

    # Data is stored column-wise: all values of field 0, then field 1, ...
    result = np.empty(header.points, dtype=dtype)
    pos = 0
    for name in dtype.names:
        field_dtype = dtype.fields[name][0]
        nbytes = field_dtype.itemsize * header.points
        column = np.frombuffer(raw[pos : pos + nbytes], dtype=field_dtype.base)
        result[name] = column.reshape((header.points, *field_dtype.shape))
        pos += nbytes
    return result


def read_pcd(path: str | Path) -> tuple[PCDHeader, np.ndarray]:
    """Read a PCD file into ``(header, structured_array)``."""
    raw = Path(path).read_bytes()
    header, offset = _parse_header(raw)
    dtype = header.dtype()
    body = raw[offset:]

    match header.data:
        case "ascii":
            text = body.decode("ascii", errors="replace").split()
            values_per_point = sum(header.count)
            flat = np.array(text[: header.points * values_per_point], dtype=np.float64)
            flat = flat.reshape(header.points, values_per_point)
            data = np.empty(header.points, dtype=dtype)
            col = 0
            for name, count in zip(header.fields, header.count, strict=True):
                part = flat[:, col : col + count]
                data[name] = part[:, 0] if count == 1 else part
                col += count
        case "binary":
            data = np.frombuffer(body, dtype=dtype, count=header.points).copy()
        case "binary_compressed":
            data = _read_binary_compressed(body, header, dtype)
        case other:
            raise ValueError(f"Unsupported PCD DATA mode {other!r}")

    return header, data
