"""Notebook-friendly helpers for manual map/spectrum comparisons.

This module wraps common tasks used in collaborator-vs-local checks:
- robust map loading from local outputs and collaborator converted numpy files
- manual D_ell computation with ``get_power_spec``
- comparisons against saved ``mock_ps_pred`` products
- one-point distribution summary metrics

Example
-------
from ciber.mocks.notebook_manual_compare import (
    default_paths,
    load_ours_intensity,
    load_ours_gal_overdens,
    load_collab_maps,
    compute_dell_manual,
)

paths = default_paths()
I_ours, _ = load_ours_intensity(paths.out8, tm=1, zlo=0.4, zhi=0.6, tile_deg=8.0)
g_ours, _, _ = load_ours_gal_overdens(paths.out8, tm=1, zlo=0.4, zhi=0.6, tile_deg=8.0)
I_col, g_col, _, _ = load_collab_maps(paths.collab, tm=1, zlo=0.4, zhi=0.6)
sp_ours = compute_dell_manual(I_ours, g_ours, fov_deg=8.0)
sp_col = compute_dell_manual(I_col, g_col, fov_deg=10.0)
"""

from __future__ import annotations

import glob
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from ciber.core.powerspec_pipeline import get_power_spec


ZBINS: List[Tuple[float, float]] = [
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.0),
]


_DELL_NAME_RE = re.compile(
    r"^Dell_ciber_(?P<tm>\d+)_x_band_(?P<band>[A-Za-z0-9]+)_z_(?P<zlo>[0-9.]+)_(?P<zhi>[0-9.]+)\.pkl$"
)


@dataclass(frozen=True)
class ComparePaths:
    """Canonical paths used in local collaborator comparisons."""

    repo_root: Path
    out8: Path
    out10: Path
    collab: Path


def default_paths(repo_root: str | Path = "/Users/richardfeder/Documents/ciber") -> ComparePaths:
    """Return default repository/output paths for this workspace."""
    root = Path(repo_root)
    return ComparePaths(
        repo_root=root,
        out8=root / "data/jordan_mocks/v3_boxed_outputs/tiles_8p0deg",
        out10=root / "data/jordan_mocks/v3_boxed_outputs/tiles_10p0deg",
        collab=root / "data/jordan_mocks/v3/data_for_richard_pix_6.0/converted_numpy",
    )


def _pick_existing_file(exact: Path, fallback_glob: Path) -> Path:
    if exact.exists():
        return exact
    hits = sorted(glob.glob(str(fallback_glob)))
    if not hits:
        raise FileNotFoundError(
            f"No file found. exact='{exact}', fallback_glob='{fallback_glob}'"
        )
    return Path(hits[-1])


def load_ours_intensity(
    outbase: str | Path,
    tm: int,
    zlo: float,
    zhi: float,
    sample_tag: str = "hsc_i_lt_25.0_CIBERfidmask",
    tile_deg: float = 8.0,
) -> Tuple[np.ndarray, str]:
    """Load local intensity map for a given TM and redshift bin."""
    outbase = Path(outbase)
    exact = (
        outbase
        / "mock_maps"
        / "intensity"
        / f"TM{tm}"
        / (
            f"rlz1_TM{tm}_{sample_tag}_"
            f"zmin={zlo:.1f}_zmax={zhi:.1f}_"
            f"pred_tile000_{tile_deg:.1f}deg_intensity.npz"
        )
    )
    fallback = (
        outbase
        / "mock_maps"
        / "intensity"
        / f"TM{tm}"
        / f"*zmin={zlo:.1f}_zmax={zhi:.1f}*tile000_{tile_deg:.1f}deg_intensity.npz"
    )
    fpath = _pick_existing_file(exact, fallback)
    dat = np.load(fpath)
    return np.asarray(dat["ciber_map"]), str(fpath)


def load_ours_gal_overdens(
    outbase: str | Path,
    tm: int,
    zlo: float,
    zhi: float,
    sample_tag: str = "hsc_i_lt_25.0_CIBERfidmask",
    tile_deg: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load local galaxy counts map and return overdensity (delta_g)."""
    outbase = Path(outbase)
    exact = (
        outbase
        / "mock_maps"
        / "galaxy"
        / f"TM{tm}"
        / (
            f"rlz1_TM{tm}_{sample_tag}_"
            f"zmin={zlo:.1f}_zmax={zhi:.1f}_"
            f"tile000_{tile_deg:.1f}deg_galaxy.npz"
        )
    )
    fallback = (
        outbase
        / "mock_maps"
        / "galaxy"
        / f"TM{tm}"
        / f"*zmin={zlo:.1f}_zmax={zhi:.1f}*tile000_{tile_deg:.1f}deg_galaxy.npz"
    )
    fpath = _pick_existing_file(exact, fallback)
    dat = np.load(fpath)
    counts = np.asarray(dat["gal_counts"])
    mean_counts = float(np.mean(counts))
    if mean_counts <= 0:
        raise ValueError("Mean galaxy counts <= 0; cannot construct overdensity map")
    overdens = (counts - mean_counts) / mean_counts
    return overdens, counts, str(fpath)


def load_ours_mask(
    outbase: str | Path,
    tm: int,
    zlo: float,
    zhi: float,
    tile_deg: float = 8.0,
    sample_tag: str = "hsc_i_lt_25.0_CIBERfidmask",
    map_kind: str = "full",
) -> Tuple[np.ndarray | None, str]:
    """Load local intensity mask for a given TM/redshift bin.

    Parameters
    ----------
    outbase
        Tile output directory (e.g. tiles_8p0deg or tiles_10p0deg).
    tm, zlo, zhi
        TM index and redshift bin.
    tile_deg
        Tile size in degrees in filename tag.
    sample_tag
        Tracer/mask selection tag used by pred maps.
    map_kind
        One of:
        - "full": load rlz*_TM*_full_zmin=... intensity map (stores mask when map masking applied)
        - "pred": load rlz*_TM*_<sample>_..._pred intensity map (often has mask=None)

    Returns
    -------
    mask, path
        ``mask`` is a 2D array or ``None`` if no mask was stored in file.
    """
    outbase = Path(outbase)

    if map_kind not in {"full", "pred"}:
        raise ValueError("map_kind must be 'full' or 'pred'")

    if map_kind == "full":
        exact = (
            outbase
            / "mock_maps"
            / "intensity"
            / f"TM{tm}"
            / f"rlz1_TM{tm}_full_zmin={zlo:.1f}_zmax={zhi:.1f}_tile000_{tile_deg:.1f}deg_intensity.npz"
        )
        fallback = (
            outbase
            / "mock_maps"
            / "intensity"
            / f"TM{tm}"
            / f"*TM{tm}_full_zmin={zlo:.1f}_zmax={zhi:.1f}*tile000_{tile_deg:.1f}deg_intensity.npz"
        )
    else:
        exact = (
            outbase
            / "mock_maps"
            / "intensity"
            / f"TM{tm}"
            / (
                f"rlz1_TM{tm}_{sample_tag}_"
                f"zmin={zlo:.1f}_zmax={zhi:.1f}_"
                f"pred_tile000_{tile_deg:.1f}deg_intensity.npz"
            )
        )
        fallback = (
            outbase
            / "mock_maps"
            / "intensity"
            / f"TM{tm}"
            / f"*{sample_tag}*zmin={zlo:.1f}_zmax={zhi:.1f}*pred_tile000_{tile_deg:.1f}deg_intensity.npz"
        )

    fpath = _pick_existing_file(exact, fallback)
    dat = np.load(fpath, allow_pickle=True)

    if "mask" not in dat.files:
        return None, str(fpath)

    mask = dat["mask"]
    if mask is None:
        return None, str(fpath)

    # Some npz files can encode None as a 0-d object array.
    if isinstance(mask, np.ndarray) and mask.dtype == object and mask.size == 1:
        if mask.item() is None:
            return None, str(fpath)

    return np.asarray(mask), str(fpath)


def mask_fraction(mask: np.ndarray | None) -> float:
    """Compute unmasked pixel fraction from a mask array.

    Returns 1.0 when mask is None.
    """
    if mask is None:
        return 1.0
    m = np.asarray(mask)
    good = np.isfinite(m)
    if np.sum(good) == 0:
        return float("nan")
    return float(np.mean((m[good] > 0).astype(float)))


def load_collab_maps(
    collab_dir: str | Path,
    tm: int,
    zlo: float,
    zhi: float,
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """Load collaborator intensity and i-band overdensity-like map."""
    collab_dir = Path(collab_dir)
    f_int = collab_dir / f"img_flux_band_ciber_{tm}_z_{zlo:.1f}_{zhi:.1f}.npy"
    f_gal = collab_dir / f"img_num_band_i_z_{zlo:.1f}_{zhi:.1f}.npy"
    return np.load(f_int), np.load(f_gal), str(f_int), str(f_gal)


def _load_pickle_numpy_compat(fpath: str | Path):
    """Load pickle with compatibility aliases for historical numpy internals."""
    import numpy.core

    # Some collaborator pickles were produced with environments that reference
    # numpy._core. Add aliases so loading works across numpy versions.
    sys.modules.setdefault("numpy._core", numpy.core)
    sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)

    with open(fpath, "rb") as f:
        return pickle.load(f)


def list_collab_dell_files(
    collab_dir: str | Path,
    tm: int | None = None,
    band: str | None = None,
) -> List[str]:
    """List available collaborator Dell pickle files, optionally filtered."""
    collab_dir = Path(collab_dir)
    files = sorted(collab_dir.glob("Dell_*.pkl"))
    out: List[str] = []
    for fp in files:
        m = _DELL_NAME_RE.match(fp.name)
        if m is None:
            continue
        if tm is not None and int(m.group("tm")) != int(tm):
            continue
        if band is not None and m.group("band") != band:
            continue
        out.append(str(fp))
    return out


def load_collab_dell(
    collab_dir: str | Path,
    tm: int,
    band: str,
    zlo: float,
    zhi: float,
) -> Dict[str, object]:
    """Load collaborator precomputed D_ell spectrum from Dell pickle.

    For current collaborator products, the pickle payload is a tuple:
    ``(lb, Dell)``.
    """
    collab_dir = Path(collab_dir)
    fpath = collab_dir / f"Dell_ciber_{tm}_x_band_{band}_z_{zlo:.1f}_{zhi:.1f}.pkl"
    payload = _load_pickle_numpy_compat(fpath)

    if isinstance(payload, (tuple, list)) and len(payload) >= 2:
        lb = np.asarray(payload[0])
        dell = np.asarray(payload[1])
        out: Dict[str, object] = {
            "lb": lb,
            "Dell": dell,
            "path": str(fpath),
            "tm": int(tm),
            "band": str(band),
            "zlo": float(zlo),
            "zhi": float(zhi),
        }
        if len(payload) >= 3:
            out["extra"] = payload[2:]
        return out

    if isinstance(payload, dict):
        # Provide a consistent return shape if dict style appears in future files.
        keys_lower = {k.lower(): k for k in payload.keys()}
        lb_key = keys_lower.get("lb") or keys_lower.get("ell")
        dell_key = keys_lower.get("dell") or keys_lower.get("dl")
        if lb_key is None or dell_key is None:
            raise ValueError(
                f"Dell pickle dict missing lb/Dell-like keys: {list(payload.keys())}"
            )
        return {
            "lb": np.asarray(payload[lb_key]),
            "Dell": np.asarray(payload[dell_key]),
            "path": str(fpath),
            "tm": int(tm),
            "band": str(band),
            "zlo": float(zlo),
            "zhi": float(zhi),
            "raw": payload,
        }

    raise ValueError(
        f"Unrecognized Dell pickle payload type {type(payload)} in {fpath}"
    )


def compute_dell_manual(
    intensity_map: np.ndarray,
    gal_overdens_map: np.ndarray,
    fov_deg: float,
    nbins: int = 20,
) -> Dict[str, np.ndarray]:
    """Compute auto/cross D_ell from maps with manual mean-subtraction on intensity."""
    intensity_mean_sub = intensity_map - np.mean(intensity_map)
    pixsize_arcsec = fov_deg * 3600.0 / float(intensity_map.shape[0])

    lb, cl_i, dcl_i = get_power_spec(intensity_mean_sub, nbins=nbins, pixsize=pixsize_arcsec)
    _, cl_g, dcl_g = get_power_spec(gal_overdens_map, nbins=nbins, pixsize=pixsize_arcsec)
    _, cl_x, dcl_x = get_power_spec(
        intensity_mean_sub, map_b=gal_overdens_map, nbins=nbins, pixsize=pixsize_arcsec
    )

    prefac = lb * (lb + 1.0) / (2.0 * np.pi)
    return {
        "lb": lb,
        "DlI": prefac * cl_i,
        "eDlI": prefac * dcl_i,
        "Dlg": prefac * cl_g,
        "eDlg": prefac * dcl_g,
        "Dlx": prefac * cl_x,
        "eDlx": prefac * dcl_x,
    }


def find_pred_npz(
    outbase: str | Path,
    tm: int,
    zlo: float,
    zhi: float,
    tile_deg: float,
    sample_tag: str = "hsc_i_lt_25.0_CIBERfidmask",
) -> str:
    """Find saved prediction spectrum file for a TM/z-bin/tile."""
    outbase = Path(outbase)
    exact = (
        outbase
        / "mock_ps_pred"
        / f"TM{tm}"
        / "indiv"
        / (
            f"rlz1_TM{tm}_auto_cross_pred_{sample_tag}_"
            f"zmin={zlo:.1f}_zmax={zhi:.1f}_tile000_{tile_deg:.1f}deg.npz"
        )
    )
    fallback = (
        outbase
        / "mock_ps_pred"
        / f"TM{tm}"
        / "indiv"
        / f"*zmin={zlo:.1f}_zmax={zhi:.1f}*tile000_{tile_deg:.1f}deg.npz"
    )
    return str(_pick_existing_file(exact, fallback))


def median_ratio_over_ell(
    ell: np.ndarray,
    y_num: np.ndarray,
    ell_den: np.ndarray,
    y_den: np.ndarray,
    ell_min: float = 300.0,
    ell_max: float = 3000.0,
) -> float:
    """Compute median y_num/y_den over overlapping ell range with interpolation."""
    m = np.isfinite(ell) & np.isfinite(y_num) & (ell >= ell_min) & (ell <= ell_max)
    if np.sum(m) < 2:
        return float("nan")
    y_den_i = np.interp(ell[m], ell_den, y_den)
    good = np.isfinite(y_den_i) & (y_den_i > 0)
    if np.sum(good) == 0:
        return float("nan")
    return float(np.nanmedian(y_num[m][good] / y_den_i[good]))


def compare_manual_to_saved_cross(
    outbase: str | Path,
    tm: int,
    zlo: float,
    zhi: float,
    tile_deg: float,
    sample_tag: str = "hsc_i_lt_25.0_CIBERfidmask",
    nbins: int = 20,
) -> Dict[str, object]:
    """Run a manual local cross-spectrum and compare to saved prediction file."""
    intensity, fint = load_ours_intensity(outbase, tm, zlo, zhi, sample_tag=sample_tag, tile_deg=tile_deg)
    gal_overdens, _, fgal = load_ours_gal_overdens(
        outbase, tm, zlo, zhi, sample_tag=sample_tag, tile_deg=tile_deg
    )
    sp = compute_dell_manual(intensity, gal_overdens, fov_deg=tile_deg, nbins=nbins)

    pred_fpath = find_pred_npz(
        outbase, tm, zlo, zhi, tile_deg=tile_deg, sample_tag=sample_tag
    )
    pred = np.load(pred_fpath)
    lb_pred = np.asarray(pred["lb"])
    prefac = lb_pred * (lb_pred + 1.0) / (2.0 * np.pi)
    dlx_pred = prefac * np.asarray(pred["clx_comb"])

    med = median_ratio_over_ell(sp["lb"], sp["Dlx"], lb_pred, dlx_pred)
    return {
        "median_manual_over_saved_cross_300_3000": med,
        "manual": sp,
        "saved_cross": {"lb": lb_pred, "Dlx": dlx_pred},
        "intensity_path": fint,
        "galaxy_path": fgal,
        "pred_path": pred_fpath,
    }


def cdf_l1(map_a: np.ndarray, map_b: np.ndarray, bins: int = 500) -> float:
    """L1 distance between empirical CDFs on a shared histogram support."""
    a = np.asarray(map_a).ravel()
    b = np.asarray(map_b).ravel()
    lo = float(min(np.nanmin(a), np.nanmin(b)))
    hi = float(max(np.nanmax(a), np.nanmax(b)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("nan")

    h_a, edges = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    h_b, _ = np.histogram(b, bins=edges, density=True)
    dx = edges[1] - edges[0]
    cdf_a = np.cumsum(h_a) * dx
    cdf_b = np.cumsum(h_b) * dx
    return float(np.sum(np.abs(cdf_a - cdf_b)) * dx)


def summarize_onepoint(collab_map: np.ndarray, ours_map: np.ndarray) -> Dict[str, float]:
    """Return compact one-point comparison statistics."""
    c = np.asarray(collab_map).ravel()
    o = np.asarray(ours_map).ravel()
    return {
        "cdf_L1": cdf_l1(c, o),
        "collab_mean": float(np.nanmean(c)),
        "ours_mean": float(np.nanmean(o)),
        "collab_std": float(np.nanstd(c)),
        "ours_std": float(np.nanstd(o)),
        "collab_q01": float(np.nanpercentile(c, 1)),
        "ours_q01": float(np.nanpercentile(o, 1)),
        "collab_q10": float(np.nanpercentile(c, 10)),
        "ours_q10": float(np.nanpercentile(o, 10)),
        "collab_q50": float(np.nanpercentile(c, 50)),
        "ours_q50": float(np.nanpercentile(o, 50)),
        "collab_q90": float(np.nanpercentile(c, 90)),
        "ours_q90": float(np.nanpercentile(o, 90)),
        "collab_q99": float(np.nanpercentile(c, 99)),
        "ours_q99": float(np.nanpercentile(o, 99)),
    }
