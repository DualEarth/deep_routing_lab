# src/drl/catchment.py

import matplotlib.colors as mcolors
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from pysheds.grid import Grid
from pysheds.sview import Raster, ViewFinder


class CatchmentDelineation:
    """Run catchment delineation on NPZ DEM samples and write plots."""

    D8_DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)

    D8_OFFSETS = {
        64: (-1, 0),
        128: (-1, 1),
        1: (0, 1),
        2: (1, 1),
        4: (1, 0),
        8: (1, -1),
        16: (0, -1),
        32: (-1, -1),
    }

    def __init__(
        self,
        npz_dir: str,
        out_dir: str,
        max_files: Optional[int] = None,
        dpi: Optional[int] = None,
    ) -> None:
        self.npz_dir = Path(npz_dir)
        self.out_dir = Path(out_dir)
        self.max_files = max_files
        self.dpi = dpi

    def run(self) -> None:
        """Process the configured NPZ files and save catchment outputs."""
        dpi_value = 150 if self.dpi is None else int(self.dpi)

        npz_files = discover_npz_files(self.npz_dir)
        if self.max_files is not None:
            npz_files = npz_files[: max(0, int(self.max_files))]

        self.out_dir.mkdir(parents=True, exist_ok=True)

        for npz_path in npz_files:
            sample_name = npz_path.stem
            sample_out = self.out_dir / sample_name
            sample_out.mkdir(parents=True, exist_ok=True)

            dem = load_dem(npz_path)
            _, _, fdir_arr, accumulation = compute_flow_products(dem)
            depression_labels = compute_depression_labels(dem)

            interior_catchments, border_catchments = delineate_catchments(
                fdir_arr=fdir_arr,
                depression_labels=depression_labels,
            )

            save_catchments_plot(
                dem=dem,
                interior_catchments=interior_catchments,
                border_catchments=border_catchments,
                out_path=sample_out / "catchments.png",
                sample_name=sample_name,
                dpi=dpi_value,
            )
            save_flow_accumulation_plot(
                accumulation=accumulation,
                interior_catchments=interior_catchments,
                border_catchments=border_catchments,
                out_path=sample_out / "flow_accumulation.png",
                sample_name=sample_name,
                dpi=dpi_value,
            )
            # Interactive HTML keeps the 3D view readable without the old static render.
            save_dem_3d_plotly(
                dem=dem,
                interior_catchments=interior_catchments,
                border_catchments=border_catchments,
                out_path=sample_out / "dem_3d_interactive.html",
                sample_name=sample_name,
            )

            num_interior = len(interior_catchments)
            num_border = len(border_catchments)
            print(
                "Saved {} interior and {} border catchments for {} -> {}".format(
                    num_interior, num_border, sample_name, sample_out
                )
            )

def discover_npz_files(npz_dir: Path) -> List[Path]:
    """Return sorted NPZ samples from the input directory."""
    if not npz_dir.exists():
        raise FileNotFoundError("NPZ directory not found: {}".format(npz_dir))

    files = sorted(npz_dir.rglob("sample_*.npz"))
    if not files:
        files = sorted(npz_dir.rglob("*.npz"))
    if not files:
        raise FileNotFoundError("No NPZ files found under {}".format(npz_dir))
    return files


def load_dem(npz_path: Path) -> np.ndarray:
    """Load and sanitize the DEM array from a sample NPZ file."""
    with np.load(npz_path) as data:
        if "dem" not in data:
            raise KeyError("File {} is missing key dem".format(npz_path))
        dem = np.asarray(data["dem"], dtype=np.float64)

    if dem.ndim != 2:
        raise ValueError("DEM in {} must be 2D, got {}".format(npz_path, dem.shape))

    finite = np.isfinite(dem)
    if not finite.any():
        raise ValueError("DEM in {} has no finite values".format(npz_path))
    if not finite.all():
        dem = dem.copy()
        dem[~finite] = np.nanmin(dem[finite])

    return dem


def letter_label(index: int) -> str:
    """Convert a zero-based index into an Excel-style catchment label."""
    if index < 0:
        raise ValueError("index must be >= 0")
    n = index + 1
    chars: List[str] = []
    while n > 0:
        n, rem = divmod(n - 1, 26)
        chars.append(chr(ord("A") + rem))
    return "".join(reversed(chars))


def compute_flow_products(dem: np.ndarray):
    """Compute conditioned DEM, flow direction, and flow accumulation."""
    viewfinder = ViewFinder(shape=dem.shape, nodata=np.nan)
    dem_raster = Raster(dem, viewfinder=viewfinder)
    grid = Grid(viewfinder=viewfinder)

    conditioned = grid.resolve_flats(grid.fill_pits(dem_raster))
    fdir = grid.flowdir(conditioned, routing="d8", dirmap=CatchmentDelineation.D8_DIRMAP)
    acc = grid.accumulation(fdir, routing="d8", dirmap=CatchmentDelineation.D8_DIRMAP)

    return grid, conditioned, np.asarray(fdir), np.asarray(acc, dtype=np.float64)


def compute_depression_labels(dem: np.ndarray) -> np.ndarray:
    """Label connected depression regions in the DEM."""
    viewfinder = ViewFinder(shape=dem.shape, nodata=np.nan)
    dem_raster = Raster(dem, viewfinder=viewfinder)
    grid = Grid(viewfinder=viewfinder)

    pit_filled = grid.fill_pits(dem_raster)
    depression_filled = grid.fill_depressions(pit_filled)
    depression_mask = np.asarray(depression_filled) > (dem + 1e-9)
    labels, _ = ndi.label(depression_mask, structure=np.ones((3, 3), dtype=int))
    return labels.astype(np.int32)


def delineate_catchments(
    fdir_arr: np.ndarray,
    depression_labels: Optional[np.ndarray] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Delineate interior sinks and distinct border outlets."""
    nrows, ncols = fdir_arr.shape
    unresolved = np.int64(np.iinfo(np.int64).min)
    terminal = np.full((nrows, ncols), unresolved, dtype=np.int64)

    sink_to_id: Dict[Tuple[int, int], int] = {}
    border_outlet_to_id: Dict[Tuple[int, int], int] = {}
    depression_anchor: Dict[int, Tuple[int, int]] = {}
    if depression_labels is not None and depression_labels.shape == fdir_arr.shape:
        dep_ids = np.unique(depression_labels)
        for dep_id in dep_ids.tolist():
            dep_id = int(dep_id)
            if dep_id <= 0:
                continue
            rows, cols = np.where(depression_labels == dep_id)
            if rows.size == 0:
                continue
            # Use one representative cell per depression so the whole basin maps to one sink.
            order = np.argmin(rows * ncols + cols)
            depression_anchor[dep_id] = (int(rows[order]), int(cols[order]))

    def is_edge(r: int, c: int) -> bool:
        return r == 0 or c == 0 or r == (nrows - 1) or c == (ncols - 1)

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < nrows and 0 <= c < ncols

    def get_sink_id(cell: Tuple[int, int]) -> int:
        if cell not in sink_to_id:
            sink_to_id[cell] = len(sink_to_id)
        return sink_to_id[cell]

    def get_border_id(cell: Tuple[int, int]) -> int:
        # Keep each outlet edge cell distinct so ridge-separated border drainage stays separate.
        if cell not in border_outlet_to_id:
            border_outlet_to_id[cell] = -(len(border_outlet_to_id) + 1)
        return border_outlet_to_id[cell]

    def resolve_terminal(start_r: int, start_c: int) -> int:
        path: List[Tuple[int, int]] = []
        seen_index: Dict[Tuple[int, int], int] = {}
        r, c = start_r, start_c

        while True:
            if not in_bounds(r, c):
                term = int(get_border_id((start_r, start_c)))
                break

            existing = terminal[r, c]
            if existing != unresolved:
                term = int(existing)
                break

            key = (r, c)
            if key in seen_index:
                cycle = path[seen_index[key]:]
                edge_cycle_cells = [cell for cell in cycle if is_edge(cell[0], cell[1])]
                if edge_cycle_cells:
                    # Border-touching cycles belong with edge drainage, not with interior sinks.
                    term = int(get_border_id(min(edge_cycle_cells)))
                else:
                    anchor = min(cycle)
                    term = int(get_sink_id(anchor))
                break

            seen_index[key] = len(path)
            path.append(key)

            code = int(fdir_arr[r, c]) if np.isfinite(fdir_arr[r, c]) else 0
            step = CatchmentDelineation.D8_OFFSETS.get(code)
            if step is None:
                if is_edge(r, c):
                    term = int(get_border_id((r, c)))
                else:
                    dep_id = int(depression_labels[r, c]) if depression_labels is not None else 0
                    if dep_id > 0 and dep_id in depression_anchor:
                        term = int(get_sink_id(depression_anchor[dep_id]))
                    else:
                        term = int(get_sink_id((r, c)))
                break

            nr = r + step[0]
            nc = c + step[1]
            if not in_bounds(nr, nc):
                term = int(get_border_id((r, c)))
                break

            r, c = nr, nc

        for pr, pc in path:
            terminal[pr, pc] = term
        return term

    for r in range(nrows):
        for c in range(ncols):
            if terminal[r, c] == unresolved:
                resolve_terminal(r, c)

    interior_catchments: List[Dict[str, object]] = []
    for sink_cell, sink_id in sink_to_id.items():
        mask = terminal == sink_id
        interior_catchments.append(
            {
                "sink": sink_cell,
                "mask": mask,
                "size": int(mask.sum()),
            }
        )

    interior_catchments.sort(key=lambda item: int(item["size"]), reverse=True)

    for idx, cat in enumerate(interior_catchments):
        cat["label"] = letter_label(idx) + "_in"

    border_catchments: List[Dict[str, object]] = []
    for sink_cell, border_id in border_outlet_to_id.items():
        mask = terminal == int(border_id)
        size = int(mask.sum())
        sink = sink_cell

        border_catchments.append(
            {
                "sink": sink,
                "mask": mask,
                "size": size,
            }
        )

    border_catchments.sort(key=lambda item: int(item["size"]), reverse=True)
    for idx, cat in enumerate(border_catchments):
        cat["label"] = letter_label(idx) + "_out"

    return interior_catchments, border_catchments

def save_catchments_plot(
    dem: np.ndarray,
    interior_catchments: List[Dict[str, object]],
    border_catchments: List[Dict[str, object]],
    out_path: Path,
    sample_name: str,
    dpi: int,
) -> None:
    """Save the greyscale DEM with colored catchment overlays."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))

    base = ax.imshow(dem, cmap="gray", interpolation="none")
    levels = np.linspace(np.nanmin(dem), np.nanmax(dem), 16)
    ax.contour(dem, levels=levels, colors="k", linewidths=0.45, alpha=0.35)

    cbar = plt.colorbar(base, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Elevation (greyscale)")

    all_catchments = interior_catchments + border_catchments
    cmap = plt.get_cmap("tab20", max(len(all_catchments), 1))
    for i, cat in enumerate(all_catchments):
        color = cmap(i)
        mask = cat["mask"]
        sink_r, sink_c = cat["sink"]
        label = cat["label"]

        shaded = np.ma.masked_where(~mask, mask.astype(float))
        ax.imshow(
            shaded,
            cmap=mcolors.ListedColormap([color]),
            alpha=0.40,
            interpolation="none",
        )
        ax.text(
            sink_c,
            sink_r,
            label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, boxstyle="round,pad=0.2"),
        )

    ax.set_title("{}: Interior (_in) & Border (_out) Catchments on DEM".format(sample_name))
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_flow_accumulation_plot(
    accumulation: np.ndarray,
    interior_catchments: List[Dict[str, object]],
    border_catchments: List[Dict[str, object]],
    out_path: Path,
    sample_name: str,
    dpi: int,
) -> None:
    """Save the flow accumulation plot with catchment boundaries."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    acc = np.asarray(accumulation, dtype=np.float64)
    acc[~np.isfinite(acc)] = 0.0
    acc_log = np.log1p(np.clip(acc, 0.0, None))

    fig, ax = plt.subplots(figsize=(8, 7))
    base = ax.imshow(acc_log, cmap="Blues", interpolation="none")
    cbar = plt.colorbar(base, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log(1 + flow accumulation) [white→blue]")

    all_catchments = interior_catchments + border_catchments
    cmap = plt.get_cmap("tab20", max(len(all_catchments), 1))
    for i, cat in enumerate(all_catchments):
        color = cmap(i)
        mask = cat["mask"]
        sink_r, sink_c = cat["sink"]
        label = cat["label"]

        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=1.6, alpha=0.95)
        ax.text(
            sink_c,
            sink_r,
            label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=color,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, boxstyle="round,pad=0.2"),
        )

    ax.set_title("{}: Interior (_in) & Border (_out) Catchment Borders on Flow Accumulation".format(sample_name))
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)




def save_dem_3d_plotly(
    dem: np.ndarray,
    interior_catchments: List[Dict[str, object]],
    border_catchments: List[Dict[str, object]],
    out_path: Path,
    sample_name: str,
) -> None:
    """Save an interactive Plotly 3D visualization of the DEM and catchments."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Skipping Plotly interactive output (plotly not installed): {}".format(out_path))
        return

    nrows, ncols = dem.shape
    x = np.arange(ncols)
    y = np.arange(nrows)

    z_min = float(np.nanmin(dem))
    z_max = float(np.nanmax(dem))
    z_pad = max((z_max - z_min) * 0.04, 1e-6)

    traces: List[object] = [
        go.Surface(
            z=dem,
            x=x,
            y=y,
            colorscale="Greys",
            showscale=True,
            opacity=0.7,
            colorbar=dict(title="Elevation"),
            name="DEM",
        )
    ]

    all_catchments = interior_catchments + border_catchments
    cmap = plt.get_cmap("tab20", max(len(all_catchments), 1))

    for i, cat in enumerate(all_catchments):
        mask = cat["mask"]
        sink_r, sink_c = cat["sink"]
        label = cat["label"]

        # Plot only the boundary cells to keep the 3D figure readable.
        boundary = mask & ~ndi.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
        rows, cols = np.where(boundary)
        if rows.size == 0:
            continue

        rgba = cmap(i)
        color = "rgb({},{},{})".format(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))

        traces.append(
            go.Scatter3d(
                x=cols,
                y=rows,
                z=dem[rows, cols] + z_pad,
                mode="markers",
                marker=dict(size=2.8, color=color, opacity=0.9),
                name=label,
                hovertemplate="{}<extra></extra>".format(label),
                showlegend=True,
            )
        )

        traces.append(
            go.Scatter3d(
                x=[sink_c],
                y=[sink_r],
                z=[float(dem[sink_r, sink_c]) + (2.5 * z_pad)],
                mode="text",
                text=[label],
                textfont=dict(color="black", size=10),
                name="{}_label".format(label),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="{}: 3D DEM with Catchments (Interactive)".format(sample_name),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Elevation",
            aspectmode="data",
            camera=dict(eye=dict(x=-1.6, y=-1.6, z=1.1)),
        ),
        template="plotly_white",
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def run_catchment_delineation(
    npz_dir: str,
    out_dir: str,
    max_files: Optional[int] = None,
    dpi: Optional[int] = None,
) -> None:
    """Run catchment delineation over NPZ samples."""
    CatchmentDelineation(
        npz_dir=npz_dir,
        out_dir=out_dir,
        max_files=max_files,
        dpi=dpi,
    ).run()


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the catchment script."""
    parser = argparse.ArgumentParser(
        description="Delineate interior sink catchments from DEM samples in NPZ files."
    )
    parser.add_argument("--npz_dir", type=str, default="dataset/npz", help="Input directory with NPZ samples")
    parser.add_argument("--out_dir", type=str, default="dataset/png/catchment", help="Output directory")
    parser.add_argument("--max_files", type=int, default=None, help="Optional cap on processed files")
    parser.add_argument("--dpi", type=int, default=None, help="PNG DPI override")
    return parser


def main() -> None:
    """Entry point for the catchment CLI."""
    args = _build_arg_parser().parse_args()
    run_catchment_delineation(
        npz_dir=args.npz_dir,
        out_dir=args.out_dir,
        max_files=args.max_files,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
