"""Qt application: folder browser + 3D view + field/coloring/camera panels."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from qtpy import QtCore, QtGui, QtWidgets

from pcd_viewer.cloud import (
    PointCloud,
    color_range,
    crop_mask,
    field_stats,
    list_pcd_files,
    load_cloud,
    scalars_for,
)

DEFAULT_COLOR_FIELDS = ("reflectivity", "intensity", "z")
COLORMAPS = ("viridis", "plasma", "inferno", "turbo", "jet", "gray", "coolwarm", "Spectral")
_SOLID = ""  # field-combo data for "no scalar coloring"
_RANGE_PERCENTILE, _RANGE_FULL, _RANGE_MANUAL = "percentile", "full", "manual"
_STATS_COLUMNS = ("Field", "Type", "Min", "Max", "Mean", "NaN/Inf", "Unique")

# name -> (pyvista view method, negative)
VIEWS = {
    "top": ("view_xy", False),
    "bottom": ("view_xy", True),
    "front": ("view_xz", False),
    "back": ("view_xz", True),
    "side": ("view_yz", False),
    "other side": ("view_yz", True),
}

SHORTCUT_HELP = """\
Right / PgDn      next file
Left / PgUp       previous file
Home / End        first / last file
Space             play / pause
1 / 2 / 3 / 4     top / front / side / isometric view
                  (press 1 / 2 / 3 again for bottom / back / other side)
Ctrl+1 / 2 / 3    bottom / back / other side
R                 reset camera
P                 toggle parallel projection
+ / -             point size
Ctrl+O            open folder
F5                reload folder
Ctrl+F            focus file filter
Ctrl+Q            quit

Mouse: left-drag rotate, right-drag / wheel zoom, middle-drag pan.
Click a row in the Fields table to color by that field.
Drop a folder or .pcd file onto the window to open it."""


def _fmt(value: float) -> str:
    return "–" if math.isnan(value) else f"{value:.6g}"


def _human_size(nbytes: int) -> str:
    size = float(nbytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{nbytes} B"


class ViewerWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        path: Path | None = None,
        color_by: str | None = None,
        point_size: float = 3.0,
        cmap: str = "viridis",
        full_range: bool = False,
    ) -> None:
        super().__init__()
        self.setWindowTitle("PCD Viewer")
        self.resize(1500, 900)
        self.setAcceptDrops(True)

        self._folder: Path | None = None
        self._files: list[Path] = []  # all files in the folder (sorted)
        self._visible: list[Path] = []  # after the filter
        self._cloud: PointCloud | None = None
        self._preferred_field = color_by  # None = automatic default
        self._camera_initialized = False
        self._last_clim: tuple[float, float] | None = None
        self._last_view: tuple[str, object] | None = None  # (view name, camera position)
        self._n_shown = 0  # points drawn after cropping / hiding

        # auto_update=False: no periodic re-render while idle; we request repaints ourselves
        self.plotter = QtInteractor(self, auto_update=False)
        self._actor = None  # current cloud actor
        self._actor_key: tuple | None = None  # settings the actor was built with
        self.setCentralWidget(self.plotter)
        self._setup_plotter()

        self._build_sidebar(point_size, cmap, full_range)
        self._build_shortcuts()

        self._play_timer = QtCore.QTimer(self)
        self._play_timer.timeout.connect(lambda: self.step(1, wrap=True))

        if path is not None:
            self.open_path(path)
        else:
            self.statusBar().showMessage("Open a folder with Ctrl+O or drop one onto the window.")

    # ------------------------------------------------------------------ setup

    def _setup_plotter(self) -> None:
        p = self.plotter
        p.set_background("black")
        p.show_axes()
        p.enable_trackball_style()
        # Drop pyvista's built-in keys that would conflict with the sidebar state
        # ('q' would close the render widget, +/- would desync the point size box).
        for key in ("q", "plus", "minus"):
            p.clear_events_for_key(key, raise_on_missing=False)

    def _build_sidebar(self, point_size: float, cmap: str, full_range: bool) -> None:
        dock = QtWidgets.QDockWidget("Controls", self)
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.addWidget(self._build_files_group(), stretch=3)
        layout.addWidget(self._build_fields_group(), stretch=2)
        layout.addWidget(self._build_color_group(point_size, cmap, full_range))
        layout.addWidget(self._build_crop_group())
        layout.addWidget(self._build_camera_group())

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(content.sizeHint().width() + 20)
        dock.setWidget(scroll)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _build_files_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Files")
        lay = QtWidgets.QVBoxLayout(box)

        row = QtWidgets.QHBoxLayout()
        open_btn = QtWidgets.QPushButton("Open folder…")
        open_btn.clicked.connect(self.choose_folder)
        reload_btn = QtWidgets.QPushButton("Reload")
        reload_btn.setToolTip("Re-scan the folder (F5)")
        reload_btn.clicked.connect(self.reload_folder)
        help_btn = QtWidgets.QPushButton("Shortcuts")
        help_btn.clicked.connect(self._show_help)
        row.addWidget(open_btn)
        row.addWidget(reload_btn)
        row.addWidget(help_btn)
        lay.addLayout(row)

        self.folder_label = QtWidgets.QLabel("No folder opened")
        self.folder_label.setWordWrap(True)
        self.folder_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        lay.addWidget(self.folder_label)

        row = QtWidgets.QHBoxLayout()
        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.setPlaceholderText("Filter file names…")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._apply_filter)
        self.sort_combo = QtWidgets.QComboBox()
        self.sort_combo.addItem("Name", "name")
        self.sort_combo.addItem("Modified", "mtime")
        self.sort_combo.currentIndexChanged.connect(self.reload_folder)
        row.addWidget(self.filter_edit, stretch=1)
        row.addWidget(QtWidgets.QLabel("Sort:"))
        row.addWidget(self.sort_combo)
        lay.addLayout(row)

        self.file_list = QtWidgets.QListWidget()
        self.file_list.setUniformItemSizes(True)
        self.file_list.currentRowChanged.connect(self._on_row_changed)
        lay.addWidget(self.file_list, stretch=1)

        row = QtWidgets.QHBoxLayout()
        prev_btn = QtWidgets.QPushButton("◀ Prev")
        prev_btn.clicked.connect(lambda: self.step(-1))
        self.play_btn = QtWidgets.QPushButton("▶ Play")
        self.play_btn.setCheckable(True)
        self.play_btn.toggled.connect(self._set_playing)
        next_btn = QtWidgets.QPushButton("Next ▶")
        next_btn.clicked.connect(lambda: self.step(1))
        self.fps_spin = QtWidgets.QDoubleSpinBox()
        self.fps_spin.setRange(0.2, 60)
        self.fps_spin.setValue(5)
        self.fps_spin.setSuffix(" fps")
        self.fps_spin.valueChanged.connect(self._update_play_interval)
        for w in (prev_btn, self.play_btn, next_btn, self.fps_spin):
            row.addWidget(w)
        lay.addLayout(row)
        return box

    def _build_fields_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Fields (current file)")
        lay = QtWidgets.QVBoxLayout(box)
        self.stats_table = QtWidgets.QTableWidget(0, len(_STATS_COLUMNS))
        self.stats_table.setHorizontalHeaderLabels(_STATS_COLUMNS)
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.stats_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.stats_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.stats_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        self.stats_table.cellClicked.connect(self._on_stats_clicked)
        self.stats_table.setToolTip("Click a row to color the cloud by that field")
        lay.addWidget(self.stats_table)

        self.header_label = QtWidgets.QLabel()
        self.header_label.setWordWrap(True)
        self.header_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        lay.addWidget(self.header_label)
        return box

    def _build_color_group(
        self, point_size: float, cmap: str, full_range: bool
    ) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Coloring")
        form = QtWidgets.QFormLayout(box)

        self.field_combo = QtWidgets.QComboBox()
        self.field_combo.currentIndexChanged.connect(self._on_field_changed)
        form.addRow("Color by:", self.field_combo)

        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.setEditable(True)  # any matplotlib colormap name works
        self.cmap_combo.addItems(COLORMAPS)
        self.cmap_combo.setCurrentText(cmap)
        self.cmap_combo.currentTextChanged.connect(self.redraw)
        form.addRow("Colormap:", self.cmap_combo)

        self.range_combo = QtWidgets.QComboBox()
        self.range_combo.addItem("Percentile 1–99 (per file)", _RANGE_PERCENTILE)
        self.range_combo.addItem("Full min–max (per file)", _RANGE_FULL)
        self.range_combo.addItem("Manual (locked across files)", _RANGE_MANUAL)
        self.range_combo.setCurrentIndex(1 if full_range else 0)
        self.range_combo.currentIndexChanged.connect(self._on_range_mode_changed)
        form.addRow("Range:", self.range_combo)

        row = QtWidgets.QHBoxLayout()
        self.min_spin = self._make_range_spin()
        self.max_spin = self._make_range_spin()
        lock_btn = QtWidgets.QPushButton("Lock current")
        lock_btn.setToolTip("Freeze the current color range so all files use the same scale")
        lock_btn.clicked.connect(self._lock_current_range)
        row.addWidget(self.min_spin)
        row.addWidget(QtWidgets.QLabel("–"))
        row.addWidget(self.max_spin)
        row.addWidget(lock_btn)
        form.addRow("Min / max:", row)

        self.hide_outside_check = QtWidgets.QCheckBox("Hide points outside range")
        self.hide_outside_check.toggled.connect(self.redraw)
        form.addRow(self.hide_outside_check)

        self.point_size_spin = QtWidgets.QDoubleSpinBox()
        self.point_size_spin.setRange(0.5, 30)
        self.point_size_spin.setSingleStep(0.5)
        self.point_size_spin.setValue(point_size)
        self.point_size_spin.valueChanged.connect(self.redraw)
        self.spheres_check = QtWidgets.QCheckBox("Spheres")
        self.spheres_check.setChecked(True)
        self.spheres_check.toggled.connect(self.redraw)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.point_size_spin)
        row.addWidget(self.spheres_check)
        form.addRow("Point size:", row)
        return box

    def _make_range_spin(self) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(-1e12, 1e12)
        spin.setDecimals(4)
        spin.setKeyboardTracking(False)
        spin.valueChanged.connect(self._on_manual_range_edited)
        return spin

    def _build_crop_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Crop (X / Y / Z limits)")
        box.setToolTip(
            "Only points inside every checked limit are shown. Checked limits stay when switching "
            "files; unchecked ones show the current file's extent."
        )
        lay = QtWidgets.QVBoxLayout(box)
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Min"), 0, 1)
        grid.addWidget(QtWidgets.QLabel("Max"), 0, 2)
        self.crop_checks: list[tuple[QtWidgets.QCheckBox, QtWidgets.QCheckBox]] = []
        self.crop_spins: list[tuple[QtWidgets.QDoubleSpinBox, QtWidgets.QDoubleSpinBox]] = []
        for axis, name in enumerate("XYZ"):
            grid.addWidget(QtWidgets.QLabel(name), axis + 1, 0)
            checks, spins = [], []
            for side, label in enumerate(("min", "max")):
                check = QtWidgets.QCheckBox()
                check.setToolTip(f"Limit {name} {label}")
                check.toggled.connect(self._on_crop_changed)
                spin = QtWidgets.QDoubleSpinBox()
                spin.setRange(-1e9, 1e9)
                spin.setDecimals(3)
                spin.setSingleStep(0.1)
                spin.setKeyboardTracking(False)
                spin.valueChanged.connect(lambda _v, c=check: self._on_crop_spin_edited(c))
                cell = QtWidgets.QHBoxLayout()
                cell.addWidget(check)
                cell.addWidget(spin, stretch=1)
                grid.addLayout(cell, axis + 1, side + 1)
                checks.append(check)
                spins.append(spin)
            self.crop_checks.append((checks[0], checks[1]))
            self.crop_spins.append((spins[0], spins[1]))
        lay.addLayout(grid)

        row = QtWidgets.QHBoxLayout()
        self.shown_label = QtWidgets.QLabel()
        reset_btn = QtWidgets.QPushButton("Reset")
        reset_btn.setToolTip("Remove all X / Y / Z limits")
        reset_btn.clicked.connect(self._reset_crop)
        row.addWidget(self.shown_label, stretch=1)
        row.addWidget(reset_btn)
        lay.addLayout(row)
        return box

    def _build_camera_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Camera")
        lay = QtWidgets.QVBoxLayout(box)
        grid = QtWidgets.QGridLayout()
        views = (
            ("Top (1)", lambda: self.set_view("top")),
            ("Front (2)", lambda: self.set_view("front")),
            ("Side (3)", lambda: self.set_view("side")),
            ("Bottom (Ctrl+1)", lambda: self.set_view("bottom")),
            ("Back (Ctrl+2)", lambda: self.set_view("back")),
            ("Other side (Ctrl+3)", lambda: self.set_view("other side")),
            ("Iso (4)", self.view_iso),
            ("Reset (R)", self.reset_camera),
        )
        for i, (label, slot) in enumerate(views):
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(slot)
            grid.addWidget(btn, i // 3, i % 3)
        grid.addWidget(QtWidgets.QLabel("Press 1/2/3 twice to flip sides"), 2, 2)
        lay.addLayout(grid)

        self.keep_camera_check = QtWidgets.QCheckBox("Keep camera when switching files")
        self.keep_camera_check.setChecked(True)
        self.parallel_check = QtWidgets.QCheckBox("Parallel projection (P)")
        self.parallel_check.toggled.connect(self._set_parallel)
        self.grid_check = QtWidgets.QCheckBox("Show bounds grid")
        self.grid_check.toggled.connect(self._update_grid)
        for w in (self.keep_camera_check, self.parallel_check, self.grid_check):
            lay.addWidget(w)
        return box

    def _build_shortcuts(self) -> None:
        bindings = {
            ("Right", "PgDown"): lambda: self.step(1),
            ("Left", "PgUp"): lambda: self.step(-1),
            ("Home",): lambda: self.go_to(0),
            ("End",): lambda: self.go_to(len(self._visible) - 1),
            ("Space",): self.play_btn.toggle,
            ("1",): lambda: self.toggle_view("top", "bottom"),
            ("2",): lambda: self.toggle_view("front", "back"),
            ("3",): lambda: self.toggle_view("side", "other side"),
            ("Ctrl+1",): lambda: self.set_view("bottom"),
            ("Ctrl+2",): lambda: self.set_view("back"),
            ("Ctrl+3",): lambda: self.set_view("other side"),
            ("4",): self.view_iso,
            ("R",): self.reset_camera,
            ("P",): self.parallel_check.toggle,
            ("+", "="): lambda: self.point_size_spin.stepBy(1),
            ("-",): lambda: self.point_size_spin.stepBy(-1),
            ("Ctrl+O",): self.choose_folder,
            ("F5",): self.reload_folder,
            ("Ctrl+F",): self._focus_filter,
            ("Ctrl+Q",): self.close,
        }
        for keys, slot in bindings.items():
            for key in keys:
                sc = QtGui.QShortcut(QtGui.QKeySequence(key), self)
                sc.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
                sc.activated.connect(slot)

    # ------------------------------------------------------------- folder/files

    def open_path(self, path: Path) -> None:
        """Open a folder, or a file's folder with that file selected."""
        path = Path(path).expanduser().resolve()
        if path.is_file():
            self.open_folder(path.parent, select=path)
        elif path.is_dir():
            self.open_folder(path)
        else:
            self.statusBar().showMessage(f"Not found: {path}")

    def open_folder(self, folder: Path, select: Path | None = None) -> None:
        self._folder = folder
        self.folder_label.setText(str(folder))
        self.setWindowTitle(f"PCD Viewer — {folder.name or folder}")
        self._camera_initialized = False
        self.reload_folder(select=select)

    def choose_folder(self) -> None:
        start = str(self._folder) if self._folder else str(Path.cwd())
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open PCD folder", start)
        if folder:
            self.open_folder(Path(folder))

    def reload_folder(self, *_args, select: Path | None = None) -> None:
        if self._folder is None:
            return
        current = select or (self._cloud.path if self._cloud else None)
        try:
            self._files = list_pcd_files(self._folder, sort=self.sort_combo.currentData())
        except OSError as exc:
            self._files = []
            self.statusBar().showMessage(f"Cannot read folder: {exc}")
        self._apply_filter(select=current)

    def _apply_filter(self, *_args, select: Path | None = None) -> None:
        current = select or (self._cloud.path if self._cloud else None)
        needle = self.filter_edit.text().strip().lower()
        self._visible = [p for p in self._files if needle in p.name.lower()]

        self.file_list.blockSignals(True)
        self.file_list.clear()
        self.file_list.addItems([p.name for p in self._visible])
        self.file_list.blockSignals(False)

        row = 0
        if current is not None:
            resolved = [p.resolve() for p in self._visible]
            if current.resolve() in resolved:
                row = resolved.index(current.resolve())
        if self._visible:
            self.file_list.setCurrentRow(row)
            # setCurrentRow does not emit if the row number is unchanged
            if self._cloud is None or self._cloud.path != self._visible[row].resolve():
                self._on_row_changed(row)
        else:
            self._show_cloud(None)
            total = len(self._files)
            self.statusBar().showMessage(
                f"No matching files ({total} .pcd files in folder)" if total else "No .pcd files"
            )

    def step(self, delta: int, wrap: bool = False) -> None:
        n = len(self._visible)
        if not n:
            return
        row = self.file_list.currentRow() + delta
        row = row % n if wrap else max(0, min(n - 1, row))
        self.go_to(row)

    def go_to(self, row: int) -> None:
        if 0 <= row < len(self._visible):
            self.file_list.setCurrentRow(row)

    def _on_row_changed(self, row: int) -> None:
        if not 0 <= row < len(self._visible):
            return
        path = self._visible[row]
        item = self.file_list.item(row)
        try:
            cloud = load_cloud(path)
        except Exception as exc:  # a broken file must not kill the browser
            item.setForeground(QtGui.QColor("tomato"))
            item.setToolTip(str(exc))
            self._show_cloud(None)
            self.statusBar().showMessage(f"{row + 1} / {len(self._visible)} · {path.name} · {exc}")
            return
        self._show_cloud(cloud)
        h = cloud.header
        self.statusBar().showMessage(
            f"{row + 1} / {len(self._visible)} · {path.name} · {cloud.n_points} pts "
            f"({cloud.n_valid} valid) · {h.data} · {_human_size(cloud.file_size)}"
        )

    # ------------------------------------------------------------ current cloud

    def _show_cloud(self, cloud: PointCloud | None) -> None:
        self._cloud = cloud
        self._update_fields(cloud)
        self._sync_crop_spins()
        self.redraw()
        if cloud is not None and (
            not self._camera_initialized or not self.keep_camera_check.isChecked()
        ):
            self.plotter.reset_camera(render=False)
            self._camera_initialized = True
        if self.grid_check.isChecked():
            self._update_grid()

    def _update_fields(self, cloud: PointCloud | None) -> None:
        fields = cloud.header.fields if cloud else []

        # Field combo: rebuild only when the field set changes; keep the chosen field.
        existing = [self.field_combo.itemData(i) for i in range(self.field_combo.count())]
        wanted = [_SOLID, *fields]
        if existing != wanted:
            self.field_combo.blockSignals(True)
            self.field_combo.clear()
            self.field_combo.addItem("(solid white)", _SOLID)
            for f in fields:
                self.field_combo.addItem(f, f)
            self.field_combo.blockSignals(False)
        self._select_field(self._choose_field(fields))

        # Stats table
        stats = field_stats(cloud.header, cloud.data) if cloud else []
        self.stats_table.setRowCount(len(stats))
        for r, s in enumerate(stats):
            cells = (s.name, s.type, _fmt(s.min), _fmt(s.max), _fmt(s.mean), s.non_finite)
            for c, value in enumerate((*cells, s.unique_text)):
                item = self.stats_table.item(r, c)
                if item is None:
                    item = QtWidgets.QTableWidgetItem()
                    if c >= 2:
                        item.setTextAlignment(
                            QtCore.Qt.AlignmentFlag.AlignRight
                            | QtCore.Qt.AlignmentFlag.AlignVCenter
                        )
                    self.stats_table.setItem(r, c, item)
                item.setText(str(value))
        self._highlight_stats_row()

        if cloud:
            h = cloud.header
            vp = " ".join(f"{v:g}" for v in h.viewpoint)
            self.header_label.setText(
                f"VERSION {h.version} · DATA {h.data} · WIDTH {h.width} × HEIGHT {h.height}"
                f" · VIEWPOINT {vp}"
            )
        else:
            self.header_label.setText("")

    def _choose_field(self, fields: list[str]) -> str:
        pref = self._preferred_field
        if pref is not None and (pref == _SOLID or pref in fields):
            return pref
        return next((f for f in DEFAULT_COLOR_FIELDS if f in fields), _SOLID)

    def _select_field(self, field: str) -> None:
        idx = self.field_combo.findData(field)
        self.field_combo.blockSignals(True)
        self.field_combo.setCurrentIndex(max(idx, 0))
        self.field_combo.blockSignals(False)

    def _current_field(self) -> str:
        return self.field_combo.currentData() or _SOLID

    def _highlight_stats_row(self) -> None:
        field = self._current_field()
        self.stats_table.blockSignals(True)
        self.stats_table.clearSelection()
        for r in range(self.stats_table.rowCount()):
            item = self.stats_table.item(r, 0)
            if item is not None and item.text() == field:
                self.stats_table.selectRow(r)
        self.stats_table.blockSignals(False)

    # ---------------------------------------------------------------- coloring

    def _on_field_changed(self, _index: int) -> None:
        self._preferred_field = self._current_field()
        self._highlight_stats_row()
        if self._range_mode() == _RANGE_MANUAL:
            # a manual range of another field is meaningless: start from this field's range
            self.range_combo.setCurrentIndex(0)
        self.redraw()

    def _on_stats_clicked(self, row: int, _col: int) -> None:
        item = self.stats_table.item(row, 0)
        if item is not None:
            idx = self.field_combo.findData(item.text())
            if idx >= 0:
                self.field_combo.setCurrentIndex(idx)

    def _range_mode(self) -> str:
        return self.range_combo.currentData()

    def _on_range_mode_changed(self, _index: int) -> None:
        if self._range_mode() == _RANGE_MANUAL and self._last_clim is not None:
            self._set_range_spins(self._last_clim)
        self.redraw()

    def _on_manual_range_edited(self, _value: float) -> None:
        if self._range_mode() != _RANGE_MANUAL:
            self.range_combo.blockSignals(True)
            self.range_combo.setCurrentIndex(self.range_combo.findData(_RANGE_MANUAL))
            self.range_combo.blockSignals(False)
        self.redraw()

    def _lock_current_range(self) -> None:
        if self._last_clim is not None:
            self._set_range_spins(self._last_clim)
        self.range_combo.setCurrentIndex(self.range_combo.findData(_RANGE_MANUAL))

    def _set_range_spins(self, clim: tuple[float, float]) -> None:
        for spin, value in zip((self.min_spin, self.max_spin), clim, strict=True):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def _clim_for(self, values: np.ndarray) -> tuple[float, float] | None:
        mode = self._range_mode()
        if mode == _RANGE_MANUAL:
            lo, hi = self.min_spin.value(), self.max_spin.value()
            return (lo, hi) if hi > lo else (lo, lo + 1e-6)
        return color_range(values, mode=mode)

    # ------------------------------------------------------------------- crop

    def _crop_limits(self) -> list[tuple[float | None, float | None]]:
        """Per axis (min, max) of the checked limits; ``None`` = no limit."""
        return [
            tuple(s.value() if c.isChecked() else None for c, s in zip(checks, spins, strict=True))
            for checks, spins in zip(self.crop_checks, self.crop_spins, strict=True)
        ]

    def _sync_crop_spins(self) -> None:
        """Show the current file's extent in the spin boxes of the unchecked limits."""
        cloud = self._cloud
        if cloud is None or cloud.n_valid == 0:
            return
        extent = (cloud.xyz.min(axis=0), cloud.xyz.max(axis=0))
        for axis, (checks, spins) in enumerate(zip(self.crop_checks, self.crop_spins, strict=True)):
            for side, (check, spin) in enumerate(zip(checks, spins, strict=True)):
                if not check.isChecked():
                    spin.blockSignals(True)
                    spin.setValue(float(extent[side][axis]))
                    spin.blockSignals(False)

    def _on_crop_spin_edited(self, check: QtWidgets.QCheckBox) -> None:
        if check.isChecked():
            self._on_crop_changed()
        else:
            check.setChecked(True)  # emits toggled -> _on_crop_changed

    def _on_crop_changed(self, *_args) -> None:
        self.redraw()
        if self.grid_check.isChecked():
            self._update_grid()

    def _reset_crop(self) -> None:
        for checks in self.crop_checks:
            for check in checks:
                check.blockSignals(True)
                check.setChecked(False)
                check.blockSignals(False)
        self._sync_crop_spins()
        self._on_crop_changed()

    def _set_shown(self, n: int) -> None:
        self._n_shown = n
        total = self._cloud.n_valid if self._cloud else 0
        self.shown_label.setText(f"Showing {n} / {total} points" if self._cloud else "")

    def request_render(self) -> None:
        """Schedule one repaint; several requests within one event-loop pass are merged."""
        QtWidgets.QWidget.update(self.plotter)  # not BasePlotter.update()

    def _clear_actor(self) -> None:
        if self._actor is not None:
            self.plotter.remove_actor(self._actor, render=False)
        for title in list(self.plotter.scalar_bars.keys()):
            self.plotter.remove_scalar_bar(title, render=False)
        self._actor = self._actor_key = None

    def redraw(self, *_args) -> None:
        """Draw the current cloud with the current coloring settings.

        The actor is only rebuilt when field / colormap / point style change; switching files
        just swaps the mapper's input data, which keeps file switching fast.
        """
        cloud = self._cloud
        if cloud is None or cloud.n_valid == 0:
            self._clear_actor()
            self._set_shown(0)
            self.request_render()
            return

        xyz = cloud.xyz
        crop = crop_mask(xyz, self._crop_limits())
        if crop is not None:
            xyz = xyz[crop]
        field = self._current_field()
        cmap = self.cmap_combo.currentText() or "viridis"
        clim = None
        if field == _SOLID:
            self._last_clim = None
            mesh = pv.PolyData(xyz)
        else:
            values = scalars_for(cloud, field)
            if crop is not None:
                values = values[crop]
            clim = self._clim_for(values)  # automatic ranges follow the cropped points
            self._last_clim = clim
            if self._range_mode() != _RANGE_MANUAL and clim is not None:
                self._set_range_spins(clim)  # show what the automatic range is
            if self.hide_outside_check.isChecked() and clim is not None:
                keep = (values >= clim[0]) & (values <= clim[1])
                xyz, values = xyz[keep], values[keep]
            mesh = pv.PolyData(xyz)
            mesh[field] = values

        self._set_shown(mesh.n_points)
        if mesh.n_points == 0:
            self._clear_actor()
            self.request_render()
            return

        key = (field, cmap, self.spheres_check.isChecked())
        if self._actor is not None and key == self._actor_key:
            mapper = self._actor.mapper
            mapper.SetInputData(mesh)
            if clim is not None:
                mapper.scalar_range = clim
                mapper.lookup_table.scalar_range = clim
        else:
            self._clear_actor()
            kwargs = {
                "render_points_as_spheres": key[2],
                "reset_camera": False,
                "render": False,
            }
            if field == _SOLID:
                self._actor = self.plotter.add_points(mesh, color="white", **kwargs)
            else:
                self._actor = self.plotter.add_points(
                    mesh,
                    scalars=field,
                    cmap=cmap,
                    clim=clim,
                    scalar_bar_args={
                        "title": field,
                        "color": "white",
                        "fmt": "%.4g",
                        "vertical": True,
                        "position_x": 0.9,
                        "position_y": 0.15,
                        "height": 0.7,
                        "width": 0.06,
                    },
                    **kwargs,
                )
            self._actor_key = key
        self._actor.prop.point_size = self.point_size_spin.value()
        self.request_render()

    # ------------------------------------------------------------------ camera

    def set_view(self, name: str) -> None:
        """Look at the cloud along an axis; see ``VIEWS`` for the names."""
        method, negative = VIEWS[name]
        getattr(self.plotter, method)(negative=negative)
        self._last_view = (name, self.plotter.camera_position)

    def toggle_view(self, name: str, opposite: str) -> None:
        """Go to ``name``; if the camera is still exactly there, flip to ``opposite``."""
        last = self._last_view
        at_name = last is not None and last[0] == name
        if at_name and last[1] == self.plotter.camera_position:
            self.set_view(opposite)
        else:
            self.set_view(name)

    def view_iso(self) -> None:
        self.plotter.view_isometric()

    def reset_camera(self) -> None:
        self.plotter.reset_camera()

    def _set_parallel(self, enabled: bool) -> None:
        if enabled:
            self.plotter.enable_parallel_projection()
        else:
            self.plotter.disable_parallel_projection()
        self.request_render()

    def _update_grid(self, *_args) -> None:
        p = self.plotter
        p.remove_bounds_axes()
        if self.grid_check.isChecked() and self._actor is not None:
            p.show_grid(color="gray", render=False)
        self.request_render()

    # ------------------------------------------------------------------- misc

    def _set_playing(self, playing: bool) -> None:
        self.play_btn.setText("⏸ Pause" if playing else "▶ Play")
        if playing:
            self._update_play_interval()
            self._play_timer.start()
        else:
            self._play_timer.stop()

    def _update_play_interval(self, *_args) -> None:
        self._play_timer.setInterval(int(1000 / self.fps_spin.value()))

    def _focus_filter(self) -> None:
        self.filter_edit.setFocus()
        self.filter_edit.selectAll()

    def _show_help(self) -> None:
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Shortcuts")
        box.setText(f"<pre>{SHORTCUT_HELP}</pre>")
        box.exec()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # noqa: N802 (Qt API)
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # noqa: N802 (Qt API)
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            self.open_path(Path(urls[0].toLocalFile()))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt API)
        self._play_timer.stop()
        self.plotter.close()
        super().closeEvent(event)


def run(path: Path | None = None, **kwargs) -> int:
    """Start the Qt event loop with a :class:`ViewerWindow`."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("PCD Viewer")
    window = ViewerWindow(path, **kwargs)
    window.show()
    return app.exec()
