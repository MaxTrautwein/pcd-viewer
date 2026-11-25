# pcd-viewer

A viewer and folder browser for [PCD](https://pointclouds.org/documentation/tutorials/pcd_file_format.html)
point cloud files. It uses a small numpy-based PCD reader that handles ascii, binary and
binary_compressed data and keeps custom fields such as `reflectivity`. The GUI is Qt with
[PyVista](https://pyvista.org) rendering ([pyvistaqt](https://qtdocs.pyvista.org)).

## Usage

```bash
uv sync
uv run pcd-viewer                              # empty window, then "Open folder…" (Ctrl+O)
uv run pcd-viewer path/to/folder               # browse all .pcd files in a folder
uv run pcd-viewer cloud.pcd                    # opens the file's folder with that file selected
uv run pcd-viewer folder --color-by z --cmap turbo -s 5
uv run pcd-viewer folder --full-range          # start with min–max instead of 1st–99th percentile
uv run pcd-viewer cloud.pcd --info             # print header + field stats (no GUI)
uv run pcd-viewer folder --info                # one summary line per file (no GUI)
```

You can also drop a folder or a `.pcd` file onto the window.

### Sidebar

- **Files**: filter by name, sort by name (natural order) or modified time, Prev/Next, and Play
  with an adjustable fps. Loaded files are cached, so going back is instant. A file that fails
  to parse is marked red and its error is shown in the status bar.
- **Fields**: every field in the current file with type, min, max, mean, NaN/Inf count and
  number of unique values, plus the raw header info. Click a row to color by that field.
- **Coloring**: field, colormap (the box is editable, so any matplotlib name works), and range mode:
  - percentile 1–99 per file;
  - full min–max per file;
  - manual. *Lock current* freezes the current range, so all files use the same color scale.

  You can also hide points outside the range, change the point size, and toggle spheres.
- **Crop**: an optional min and max for X, Y and Z, each with its own checkbox; only points inside
  every checked limit are shown. Checked limits stay when you switch files; unchecked ones show the
  current file's extent. *Reset* removes all limits.
- **Camera**: Top / Bottom, Front / Back, Side / Other side, Iso and Reset presets, parallel
  projection, bounds grid, and "Keep camera when switching files" (on by default).

### Shortcuts

| Key | Action |
| --- | --- |
| `→` / `PgDn`, `←` / `PgUp` | next / previous file |
| `Home` / `End` | first / last file |
| `Space` | play / pause |
| `1` `2` `3` `4` | top / front / side / isometric view |
| `1` `2` `3` again, or `Ctrl+1` `Ctrl+2` `Ctrl+3` | bottom / back / other side |
| `R` | reset camera |
| `P` | toggle parallel projection |
| `+` / `-` | point size |
| `Ctrl+O`, `F5`, `Ctrl+F` | open folder, reload folder, focus filter |
| `Ctrl+Q` | quit |

Mouse: left-drag rotates, right-drag / scroll zooms, middle-drag pans.

## Development

```bash
uv run ruff check .
uv run ruff format .
uv run pytest
```
