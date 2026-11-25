import os
from pathlib import Path

import numpy as np
import pytest

from pcd_viewer import cli
from pcd_viewer.cloud import (
    color_range,
    crop_mask,
    field_stats,
    list_pcd_files,
    load_cloud,
    make_cloud,
    scalars_for,
)
from pcd_viewer.pcd import PCDHeader

SAMPLES = Path(__file__).parent.parent

HEADER = """VERSION .7
FIELDS x y z ring normal
SIZE 4 4 4 2 4
TYPE F F F U F
COUNT 1 1 1 1 3
WIDTH {n}
HEIGHT 1
POINTS {n}
DATA ascii
"""
ROWS = [
    "0 0 0 1 1 2 3",
    "1 2 3 1 4 5 6",
    "nan 0 0 2 7 8 9",
    "4 5 6 3 nan 0 0",
]


def _write(path: Path, rows: list[str] = ROWS) -> Path:
    path.write_text(HEADER.format(n=len(rows)) + "\n".join(rows) + "\n")
    return path


def test_list_pcd_files_natural_sort_and_filter(tmp_path: Path) -> None:
    for name in ("frame10.pcd", "frame2.PCD", "frame1.pcd", "notes.txt"):
        (tmp_path / name).write_text("")
    (tmp_path / "sub.pcd").mkdir()
    names = [p.name for p in list_pcd_files(tmp_path)]
    assert names == ["frame1.pcd", "frame2.PCD", "frame10.pcd"]


def test_list_pcd_files_mtime(tmp_path: Path) -> None:
    for i, name in enumerate(("c.pcd", "a.pcd", "b.pcd")):
        p = tmp_path / name
        p.write_text("")
        os.utime(p, ns=(i * 10**9, i * 10**9))
    assert [p.name for p in list_pcd_files(tmp_path, sort="mtime")] == ["c.pcd", "a.pcd", "b.pcd"]


def test_load_cloud_valid_mask_and_cache(tmp_path: Path) -> None:
    path = _write(tmp_path / "a.pcd")
    cloud = load_cloud(path)
    assert cloud.n_points == 4
    assert cloud.n_valid == 3
    assert load_cloud(path) is cloud  # cached
    _write(path, ROWS[:2])
    os.utime(path, ns=(10**18, 10**18))
    assert load_cloud(path).n_points == 2  # file changed -> reloaded


def test_field_stats(tmp_path: Path) -> None:
    cloud = load_cloud(_write(tmp_path / "a.pcd"))
    stats = {s.name: s for s in field_stats(cloud.header, cloud.data)}
    assert list(stats) == ["x", "y", "z", "ring", "normal"]
    assert stats["x"].non_finite == 1
    assert stats["x"].max == 4
    assert stats["ring"].type == "U2"
    assert stats["ring"].unique == 3
    assert stats["normal"].type == "F4 x3"
    assert stats["normal"].non_finite == 1


def test_field_stats_many_unique_values() -> None:
    header = PCDHeader(fields=["x", "y", "z"], size=[4] * 3, type=["F"] * 3, count=[1] * 3)
    data = np.zeros(5000, dtype=header.dtype())
    data["x"] = np.arange(5000)
    stats = field_stats(header, data)
    assert stats[0].unique is None
    assert stats[0].unique_text == ">1000"
    assert stats[1].unique == 1


def test_scalars_for_multi_count(tmp_path: Path) -> None:
    cloud = load_cloud(_write(tmp_path / "a.pcd"))
    np.testing.assert_allclose(scalars_for(cloud, "normal"), [1, 4, np.nan])
    np.testing.assert_allclose(scalars_for(cloud, "normal", component=2), [3, 6, 0])
    np.testing.assert_allclose(scalars_for(cloud, "ring"), [1, 1, 3])


def test_make_cloud_requires_xyz() -> None:
    header = PCDHeader(fields=["x", "y"], size=[4, 4], type=["F", "F"], count=[1, 1])
    with pytest.raises(ValueError, match="z"):
        make_cloud(Path("x.pcd"), header, np.zeros(1, dtype=header.dtype()))


def test_color_range() -> None:
    values = np.array([*range(101), np.nan], dtype=np.float32)
    assert color_range(values, "full") == (0, 100)
    lo, hi = color_range(values, "percentile")
    assert lo == pytest.approx(1) and hi == pytest.approx(99)
    assert color_range(np.full(3, 5.0), "percentile") == (5, 6)  # constant: non-empty range
    assert color_range(np.array([np.nan])) is None


def test_crop_mask() -> None:
    xyz = np.array([[0, 0, 0], [1, 2, 3], [2, 4, 6], [3, 6, 9]], dtype=np.float32)
    assert crop_mask(xyz, [(None, None)] * 3) is None
    assert crop_mask(xyz, []) is None
    assert crop_mask(xyz, [(1, None), (None, None), (None, None)]).tolist() == [0, 1, 1, 1]
    assert crop_mask(xyz, [(None, None), (None, 4), (None, None)]).tolist() == [1, 1, 1, 0]
    # all set limits must hold; edges are inside
    limits = [(1, None), (None, 4), (None, None)]
    assert crop_mask(xyz, limits).tolist() == [0, 1, 1, 0]
    assert crop_mask(xyz, [(None, None), (None, None), (3, 6)]).tolist() == [0, 1, 1, 0]
    assert not crop_mask(xyz, [(2, 1), (None, None), (None, None)]).any()  # min > max


def test_cli_info_folder(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(tmp_path / "a.pcd")
    (tmp_path / "broken.pcd").write_text("garbage")
    cli.main(["--info", str(tmp_path)])
    out = capsys.readouterr().out
    assert "2 .pcd files" in out
    assert "a.pcd" in out and "x y z ring normal" in out
    assert "broken.pcd  ERROR" in out


def test_cli_info_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["--info", str(_write(tmp_path / "a.pcd"))])
    out = capsys.readouterr().out
    assert "ring" in out and "unique=3" in out


def test_gui_smoke(tmp_path: Path) -> None:
    """Build the window offscreen and step through the sample files."""
    if not list_pcd_files(SAMPLES):
        pytest.skip("no sample files")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("pyvistaqt")
    from qtpy import QtWidgets

    from pcd_viewer.app import ViewerWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = ViewerWindow(SAMPLES)
    try:
        files = list_pcd_files(SAMPLES)
        assert win.file_list.count() == len(files)
        assert win._cloud is not None and win._cloud.path == files[0].resolve()
        assert win.field_combo.currentData() == "reflectivity"
        assert win.stats_table.rowCount() == 4
        win.step(1)
        assert win._cloud.path == files[1].resolve()
        win.step(-5)
        assert win.file_list.currentRow() == 0
        win.step(-1, wrap=True)
        assert win.file_list.currentRow() == len(files) - 1

        win._lock_current_range()
        locked = (win.min_spin.value(), win.max_spin.value())
        win.step(-1)
        assert (win.min_spin.value(), win.max_spin.value()) == locked

        # crop: checked limits stay when switching files, unchecked ones follow the file
        cloud = win._cloud
        assert win._n_shown == cloud.n_valid
        x_max_check, x_max_spin = win.crop_checks[0][1], win.crop_spins[0][1]
        assert x_max_spin.value() == pytest.approx(cloud.xyz[:, 0].max(), abs=1e-3)
        x_mid = float(np.median(cloud.xyz[:, 0]))
        x_max_spin.setValue(x_mid)  # editing checks the box
        assert x_max_check.isChecked()
        assert 0 < win._n_shown < cloud.n_valid
        assert win._n_shown == int((cloud.xyz[:, 0] <= x_max_spin.value()).sum())
        win.step(1)
        cloud = win._cloud
        assert x_max_check.isChecked()
        assert x_max_spin.value() == pytest.approx(x_mid, abs=1e-3)
        assert win._n_shown == int((cloud.xyz[:, 0] <= x_max_spin.value()).sum())
        assert win.crop_spins[2][0].value() == pytest.approx(cloud.xyz[:, 2].min(), abs=1e-3)
        win.crop_checks[2][0].setChecked(True)
        win.crop_spins[2][0].setValue(1e6)  # nothing left
        assert win._n_shown == 0 and win._actor is None
        win._reset_crop()
        assert not any(c.isChecked() for checks in win.crop_checks for c in checks)
        assert win._n_shown == cloud.n_valid
        assert win.crop_spins[0][1].value() == pytest.approx(cloud.xyz[:, 0].max(), abs=1e-3)
        win.step(-1)

        win.filter_edit.setText(files[2].name[:6])
        assert win._cloud.path == files[2].resolve()
        win.filter_edit.setText("does-not-exist")
        assert win._cloud is None
        win.filter_edit.clear()

        win.hide_outside_check.setChecked(True)
        win.field_combo.setCurrentIndex(0)  # solid
        from pcd_viewer.app import VIEWS

        for name in VIEWS:
            win.set_view(name)
        win.view_iso()
        win.reset_camera()

        win.toggle_view("front", "back")
        front = win.plotter.camera_position
        win.toggle_view("front", "back")  # same key again -> opposite side
        back = win.plotter.camera_position
        assert win._last_view[0] == "back"
        np.testing.assert_allclose(
            np.subtract(back.position, back.focal_point),
            -np.subtract(front.position, front.focal_point),
            atol=1e-6,
        )
        win.toggle_view("front", "back")
        assert win._last_view[0] == "front"
        win.view_iso()  # camera moved -> next press goes to the base view, not the opposite
        win.toggle_view("top", "bottom")
        win.plotter.camera.azimuth = 30
        win.toggle_view("top", "bottom")
        assert win._last_view[0] == "top"
        win.grid_check.setChecked(True)
        win.parallel_check.setChecked(True)
        win.step(1)
    finally:
        win.close()
        app.processEvents()
