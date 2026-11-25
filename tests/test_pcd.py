import struct
from pathlib import Path

import numpy as np
import pytest

from pcd_viewer.pcd import lzf_decompress, read_pcd

HEADER = """VERSION .7
FIELDS x y z reflectivity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {n}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {n}
DATA {mode}
"""

POINTS = np.array(
    [(0.0, 1.0, 2.0, 10.0), (3.5, -4.25, 5.0, 20.0), (6.0, 7.0, -8.5, 30.0)],
    dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("reflectivity", "<f4")],
)


def _lzf_literal_only(data: bytes) -> bytes:
    """Valid LZF stream made only of literal runs (max 32 bytes each)."""
    out = bytearray()
    for i in range(0, len(data), 32):
        chunk = data[i : i + 32]
        out.append(len(chunk) - 1)
        out += chunk
    return bytes(out)


def _write(tmp_path: Path, mode: str, body: bytes) -> Path:
    path = tmp_path / f"cloud_{mode}.pcd"
    path.write_bytes(HEADER.format(n=len(POINTS), mode=mode).encode() + body)
    return path


def _assert_points(path: Path) -> None:
    header, data = read_pcd(path)
    assert header.fields == ["x", "y", "z", "reflectivity"]
    assert header.points == len(POINTS)
    for name in POINTS.dtype.names:
        np.testing.assert_allclose(data[name], POINTS[name])


def test_ascii(tmp_path: Path) -> None:
    body = "\n".join(" ".join(str(v) for v in p) for p in POINTS.tolist()) + "\n"
    _assert_points(_write(tmp_path, "ascii", body.encode()))


def test_binary(tmp_path: Path) -> None:
    _assert_points(_write(tmp_path, "binary", POINTS.tobytes()))


def test_binary_compressed(tmp_path: Path) -> None:
    columns = b"".join(POINTS[name].tobytes() for name in POINTS.dtype.names)
    compressed = _lzf_literal_only(columns)
    body = struct.pack("<II", len(compressed), len(columns)) + compressed
    _assert_points(_write(tmp_path, "binary_compressed", body))


def test_lzf_back_reference() -> None:
    # literal "abc" followed by a back reference: length 6, distance 3 -> "abcabcabc"
    stream = bytes([2]) + b"abc" + bytes([(4 << 5) | 0, 2])
    assert lzf_decompress(stream, 9) == b"abcabcabc"


def test_lzf_size_mismatch() -> None:
    with pytest.raises(ValueError, match="size mismatch"):
        lzf_decompress(bytes([2]) + b"abc", 4)


@pytest.mark.parametrize("name", sorted(Path(__file__).parent.parent.glob("*.pcd")))
def test_sample_files(name: Path) -> None:
    header, data = read_pcd(name)
    assert len(data) == header.points > 0
    assert np.isfinite(data["x"]).any()
