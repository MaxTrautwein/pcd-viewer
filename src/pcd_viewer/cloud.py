"""Toolkit-independent helpers: folder listing, cached loading, field stats, color ranges, crop."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np

from pcd_viewer.pcd import PCDHeader, read_pcd

SortMode = Literal["name", "mtime"]
RangeMode = Literal["percentile", "full"]
Limits = Sequence[tuple[float | None, float | None]]  # per axis (min, max); None = unbounded

_UNIQUE_CAP = 1000  # stop counting unique values beyond this (it's continuous data anyway)


def _natural_key(path: Path) -> list[int | str]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def list_pcd_files(folder: str | Path, sort: SortMode = "name") -> list[Path]:
    """All ``*.pcd`` files (case-insensitive) directly inside ``folder``."""
    files = [p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() == ".pcd"]
    if sort == "mtime":
        return sorted(files, key=lambda p: (p.stat().st_mtime, _natural_key(p)))
    return sorted(files, key=_natural_key)


@dataclass(frozen=True)
class PointCloud:
    path: Path
    header: PCDHeader
    data: np.ndarray  # structured array, all points
    valid: np.ndarray  # bool mask: points with finite x, y, z
    xyz: np.ndarray  # (n_valid, 3) float32

    @property
    def n_points(self) -> int:
        return len(self.data)

    @property
    def n_valid(self) -> int:
        return len(self.xyz)

    @property
    def file_size(self) -> int:
        return self.path.stat().st_size


def make_cloud(path: Path, header: PCDHeader, data: np.ndarray) -> PointCloud:
    missing = {"x", "y", "z"} - set(header.fields)
    if missing:
        raise ValueError(f"PCD file lacks coordinate fields: {', '.join(sorted(missing))}")
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    valid = np.all(np.isfinite(xyz), axis=1)
    return PointCloud(path=path, header=header, data=data, valid=valid, xyz=xyz[valid])


@lru_cache(maxsize=256)
def _load_cached(path: Path, mtime_ns: int, size: int) -> PointCloud:
    header, data = read_pcd(path)
    return make_cloud(path, header, data)


def load_cloud(path: str | Path) -> PointCloud:
    """Read a PCD file; results are cached until the file changes on disk."""
    path = Path(path).resolve()
    st = path.stat()
    return _load_cached(path, st.st_mtime_ns, st.st_size)


@dataclass(frozen=True)
class FieldStats:
    name: str
    type: str  # e.g. "F4" or "U2 x3"
    min: float
    max: float
    mean: float
    non_finite: int
    unique: int | None  # None = more than _UNIQUE_CAP distinct values

    @property
    def unique_text(self) -> str:
        return f">{_UNIQUE_CAP}" if self.unique is None else str(self.unique)


def field_stats(header: PCDHeader, data: np.ndarray) -> list[FieldStats]:
    """Per-field statistics over all points (multi-count fields use all components)."""
    result = []
    for name, size, typ, count in zip(
        header.fields, header.size, header.type, header.count, strict=True
    ):
        values = np.asarray(data[name], dtype=np.float64).ravel()
        finite = values[np.isfinite(values)]
        if finite.size:
            lo, hi, mean = float(finite.min()), float(finite.max()), float(finite.mean())
            uniq = np.unique(finite).size
        else:
            lo = hi = mean = float("nan")
            uniq = 0
        result.append(
            FieldStats(
                name=name,
                type=f"{typ}{size}" + (f" x{count}" if count > 1 else ""),
                min=lo,
                max=hi,
                mean=mean,
                non_finite=int(values.size - finite.size),
                unique=uniq if uniq <= _UNIQUE_CAP else None,
            )
        )
    return result


def scalars_for(cloud: PointCloud, field: str, component: int = 0) -> np.ndarray:
    """Values of ``field`` for the valid points (one component of multi-count fields)."""
    values = np.asarray(cloud.data[field], dtype=np.float32)
    if values.ndim > 1:
        values = values[:, component]
    return values[cloud.valid]


def color_range(
    values: np.ndarray, mode: RangeMode = "percentile", lo: float = 1, hi: float = 99
) -> tuple[float, float] | None:
    """Color limits for ``values``; ``None`` if there are no finite values."""
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None
    if mode == "percentile":
        low, high = np.percentile(finite, [lo, hi])
        if high > low:
            return float(low), float(high)
    low, high = float(finite.min()), float(finite.max())
    if high <= low:
        high = low + 1.0
    return low, high


def crop_mask(xyz: np.ndarray, limits: Limits) -> np.ndarray | None:
    """Bool mask of the points inside every set limit (edges included).

    ``limits[i]`` is the ``(min, max)`` pair for column ``i`` of ``xyz``; ``None`` = no limit.
    Returns ``None`` if no limit is set at all, so callers can skip the copy.
    """
    mask = None
    for axis, bounds in enumerate(limits):
        for bound, inside in zip(bounds, (np.greater_equal, np.less_equal), strict=True):
            if bound is None:
                continue
            keep = inside(xyz[:, axis], bound)
            mask = keep if mask is None else mask & keep
    return mask
