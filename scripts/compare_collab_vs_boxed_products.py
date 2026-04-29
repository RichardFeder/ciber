#!/usr/bin/env python3
"""Compare collaborator mock products against local boxed-output products.

This script follows the collaborator parsing/power-spectrum conventions used in
`data/jordan_mocks/v3/data_for_richard_pix_6.0/generate_intermediate_products.py`:
- Loads collaborator pickles named like:
  - img_flux_band_ciber_{TM}_z_{zlo}_{zhi}.pkl
  - img_num_band_i_z_{zlo}_{zhi}.pkl
  - Dell_ciber_{TM}_x_band_i_z_{zlo}_{zhi}.pkl
- Uses powerbox.get_power(..., log_bins=1, bins=20, ignore_zero_mode=1)
  for auto/cross from maps.

Outputs comparison figures and a summary text file.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ciber.core.powerspec_pipeline import get_power_spec

try:
    import powerbox as pbox
    HAVE_POWERBOX = True
except Exception:
    HAVE_POWERBOX = False


@dataclass
class CaseSummary:
    tm: int
    zlo: float
    zhi: float
    ell_overlap_min: float
    ell_overlap_max: float
    cross_median_ratio_our_over_collab: float
    gal_auto_median_ratio_our_over_collab: float
    cross_ratio_at_ell1000: float
    gal_ratio_at_ell1000: float


def _load_pickle_array(path: Path) -> np.ndarray:
    # Compatibility shim: some collaborator pickles reference numpy._core
    # which may not exist in older numpy builds.
    try:
        import numpy.core as _np_core  # type: ignore
        sys.modules.setdefault("numpy._core", _np_core)
    except Exception:
        pass
    with open(path, "rb") as f:
        arr = pickle.load(f)
    return np.asarray(arr)


def _load_collab_cross(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import numpy.core as _np_core  # type: ignore
        sys.modules.setdefault("numpy._core", _np_core)
    except Exception:
        pass
    with open(path, "rb") as f:
        ell, dell = pickle.load(f)
    return np.asarray(ell), np.asarray(dell)


def _compute_powerbox_dell(
    map_a: np.ndarray,
    fov_deg: float,
    map_b: np.ndarray | None = None,
    bins: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    # Use local get_power_spec by request; this keeps collaborator parsing but
    # computes spectra via the same function as local pipeline products.
    pixsize_arcsec = fov_deg * 3600.0 / float(map_a.shape[0])
    if map_b is None:
        ell, cl, _ = get_power_spec(map_a, nbins=bins, pixsize=pixsize_arcsec)
    else:
        ell, cl, _ = get_power_spec(map_a, map_b=map_b, nbins=bins, pixsize=pixsize_arcsec)
    dell = ell * (ell + 1.0) * cl / (2.0 * np.pi)
    return np.asarray(ell), np.asarray(dell)


def _median_ratio_on_overlap(
    ell_ref: np.ndarray,
    y_ref: np.ndarray,
    ell_cmp: np.ndarray,
    y_cmp: np.ndarray,
    ell_min: float,
    ell_max: float,
) -> Tuple[float, float, float]:
    mask_ref = np.isfinite(ell_ref) & np.isfinite(y_ref) & (y_ref > 0)
    mask_cmp = np.isfinite(ell_cmp) & np.isfinite(y_cmp) & (y_cmp > 0)

    if not np.any(mask_ref) or not np.any(mask_cmp):
        return np.nan, np.nan, np.nan

    e_ref = ell_ref[mask_ref]
    y_ref = y_ref[mask_ref]
    e_cmp = ell_cmp[mask_cmp]
    y_cmp = y_cmp[mask_cmp]

    lo = max(np.min(e_ref), np.min(e_cmp), ell_min)
    hi = min(np.max(e_ref), np.max(e_cmp), ell_max)
    if not (hi > lo):
        return np.nan, np.nan, np.nan

    m = (e_ref >= lo) & (e_ref <= hi)
    if np.sum(m) < 2:
        return lo, hi, np.nan

    y_cmp_interp = np.interp(e_ref[m], e_cmp, y_cmp)
    ratio = y_ref[m] / y_cmp_interp
    return lo, hi, float(np.nanmedian(ratio))


def _ratio_at_ell(
    ell_ref: np.ndarray,
    y_ref: np.ndarray,
    ell_cmp: np.ndarray,
    y_cmp: np.ndarray,
    target_ell: float = 1000.0,
) -> float:
    m_ref = np.isfinite(ell_ref) & np.isfinite(y_ref) & (y_ref > 0)
    m_cmp = np.isfinite(ell_cmp) & np.isfinite(y_cmp) & (y_cmp > 0)
    if not np.any(m_ref) or not np.any(m_cmp):
        return np.nan

    e_ref = ell_ref[m_ref]
    y_ref = y_ref[m_ref]
    e_cmp = ell_cmp[m_cmp]
    y_cmp = y_cmp[m_cmp]

    if target_ell < max(np.min(e_ref), np.min(e_cmp)) or target_ell > min(np.max(e_ref), np.max(e_cmp)):
        return np.nan

    yr = np.interp(target_ell, e_ref, y_ref)
    yc = np.interp(target_ell, e_cmp, y_cmp)
    if yc <= 0:
        return np.nan
    return float(yr / yc)


def _plot_case(
    outdir: Path,
    tm: int,
    zlo: float,
    zhi: float,
    ell_cross_collab: np.ndarray,
    dell_cross_collab: np.ndarray,
    ell_cross_ours: np.ndarray,
    dell_cross_ours: np.ndarray,
    ell_gal_collab: np.ndarray,
    dell_gal_collab: np.ndarray,
    ell_gal_ours: np.ndarray,
    dell_gal_ours: np.ndarray,
    collab_flux: np.ndarray,
    ours_flux: np.ndarray,
    collab_gal: np.ndarray,
    ours_gal: np.ndarray,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))

    # Cross spectrum
    ax = axes[0]
    m1 = np.isfinite(dell_cross_collab) & (dell_cross_collab > 0)
    m2 = np.isfinite(dell_cross_ours) & (dell_cross_ours > 0)
    ax.loglog(ell_cross_collab[m1], dell_cross_collab[m1], "o-", ms=3, lw=1.4, label="collab 10x10")
    ax.loglog(ell_cross_ours[m2], dell_cross_ours[m2], "o-", ms=3, lw=1.4, label="ours 8x8")
    ax.set_title("CIBER x Galaxy")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$D_\ell$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # Galaxy auto
    ax = axes[1]
    m1 = np.isfinite(dell_gal_collab) & (dell_gal_collab > 0)
    m2 = np.isfinite(dell_gal_ours) & (dell_gal_ours > 0)
    ax.loglog(ell_gal_collab[m1], dell_gal_collab[m1], "o-", ms=3, lw=1.4, label="collab 10x10")
    ax.loglog(ell_gal_ours[m2], dell_gal_ours[m2], "o-", ms=3, lw=1.4, label="ours 8x8")
    ax.set_title("Galaxy Auto")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$D_\ell$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # One-point distributions
    ax = axes[2]
    cf = collab_flux[np.isfinite(collab_flux) & (collab_flux > 0)]
    of = ours_flux[np.isfinite(ours_flux) & (ours_flux > 0)]
    cg = collab_gal[np.isfinite(collab_gal)]
    og = ours_gal[np.isfinite(ours_gal)]

    if cf.size > 0:
        ax.hist(np.log10(cf), bins=120, density=True, histtype="step", lw=1.2, label="flux collab")
    if of.size > 0:
        ax.hist(np.log10(of), bins=120, density=True, histtype="step", lw=1.2, label="flux ours")
    if cg.size > 0:
        ax.hist(cg, bins=120, density=True, histtype="step", lw=1.2, label="gal collab")
    if og.size > 0:
        ax.hist(og, bins=120, density=True, histtype="step", lw=1.2, label="gal ours")

    ax.set_title("1-Point Distributions")
    ax.set_xlabel("value (flux shown as log10)")
    ax.set_ylabel("pdf")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(f"TM{tm}, z=[{zlo:.1f},{zhi:.1f}], tracer=hsc_i", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outdir / f"TM{tm}_hsc_i_z_{zlo:.1f}_{zhi:.1f}_comparison.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare collaborator mock products against local boxed outputs")
    p.add_argument(
        "--collab-dir",
        default="/Users/richardfeder/Documents/ciber/data/jordan_mocks/v3/data_for_richard_pix_6.0",
    )
    p.add_argument(
        "--our-base",
        default="/Users/richardfeder/Documents/ciber/data/jordan_mocks/v3_boxed_outputs/tiles_8p0deg",
    )
    p.add_argument(
        "--outdir",
        default="/Users/richardfeder/Documents/ciber/data/jordan_mocks/v3_boxed_outputs/tiles_8p0deg/diagnostics/collab_comparison_pix6",
    )
    p.add_argument("--rlz", type=int, default=1)
    p.add_argument("--tile-label", default="tile000_8.0deg")
    p.add_argument("--band-suffix", default="i", choices=["i", "W1"])
    p.add_argument("--our-sample-tag", default="hsc_i_lt_25.0_CIBERfidmask")
    p.add_argument("--ell-min", type=float, default=300.0)
    p.add_argument("--ell-max", type=float, default=3000.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    collab_dir = Path(args.collab_dir)
    our_base = Path(args.our_base)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Using get_power_spec for all map-based spectra; collaborator script is used for file parsing conventions.")

    zbins: Sequence[Tuple[float, float]] = [
        (0.0, 0.2),
        (0.2, 0.4),
        (0.4, 0.6),
        (0.6, 0.8),
        (0.8, 1.0),
    ]

    summaries: List[CaseSummary] = []

    for tm in [1, 2]:
        for zlo, zhi in zbins:
            # Collaborator products
            collab_flux_path = collab_dir / f"img_flux_band_ciber_{tm}_z_{zlo:.1f}_{zhi:.1f}.pkl"
            collab_num_path = collab_dir / f"img_num_band_{args.band_suffix}_z_{zlo:.1f}_{zhi:.1f}.pkl"
            collab_cross_path = collab_dir / f"Dell_ciber_{tm}_x_band_{args.band_suffix}_z_{zlo:.1f}_{zhi:.1f}.pkl"

            # Our products (HSC matching by default)
            our_flux_path = (
                our_base
                / "mock_maps"
                / "intensity"
                / f"TM{tm}"
                / f"rlz{args.rlz}_TM{tm}_{args.our_sample_tag}_zmin={zlo:.1f}_zmax={zhi:.1f}_pred_{args.tile_label}_intensity.npz"
            )
            our_gal_path = (
                our_base
                / "mock_maps"
                / "galaxy"
                / f"TM{tm}"
                / f"rlz{args.rlz}_TM{tm}_{args.our_sample_tag}_zmin={zlo:.1f}_zmax={zhi:.1f}_{args.tile_label}_galaxy.npz"
            )

            if not (collab_flux_path.exists() and collab_num_path.exists() and collab_cross_path.exists()):
                print(f"[skip] missing collaborator files for TM{tm} z=[{zlo:.1f},{zhi:.1f}]")
                continue
            if not (our_flux_path.exists() and our_gal_path.exists()):
                print(f"[skip] missing local files for TM{tm} z=[{zlo:.1f},{zhi:.1f}]")
                continue

            collab_flux = _load_pickle_array(collab_flux_path)
            collab_gal = _load_pickle_array(collab_num_path)
            ell_cross_collab, dell_cross_collab = _load_collab_cross(collab_cross_path)

            our_flux = np.load(our_flux_path)["ciber_map"]
            our_counts = np.load(our_gal_path)["gal_counts"]
            mean_counts = np.mean(our_counts)
            if mean_counts <= 0:
                print(f"[skip] zero-mean counts in local galaxy map for TM{tm} z=[{zlo:.1f},{zhi:.1f}]")
                continue
            our_gal = (our_counts - mean_counts) / mean_counts

            # Follow collaborator's convention for map-based power spectra.
            ell_cross_ours, dell_cross_ours = _compute_powerbox_dell(
                our_flux - np.mean(our_flux),
                fov_deg=8.0,
                map_b=our_gal,
                bins=20,
            )
            ell_gal_collab, dell_gal_collab = _compute_powerbox_dell(
                collab_gal,
                fov_deg=10.0,
                map_b=None,
                bins=20,
            )
            ell_gal_ours, dell_gal_ours = _compute_powerbox_dell(
                our_gal,
                fov_deg=8.0,
                map_b=None,
                bins=20,
            )

            lo_x, hi_x, cross_med_ratio = _median_ratio_on_overlap(
                ell_cross_ours,
                dell_cross_ours,
                ell_cross_collab,
                dell_cross_collab,
                args.ell_min,
                args.ell_max,
            )
            lo_g, hi_g, gal_med_ratio = _median_ratio_on_overlap(
                ell_gal_ours,
                dell_gal_ours,
                ell_gal_collab,
                dell_gal_collab,
                args.ell_min,
                args.ell_max,
            )
            ell_lo = np.nanmin([lo_x, lo_g]) if np.isfinite(lo_x) or np.isfinite(lo_g) else np.nan
            ell_hi = np.nanmax([hi_x, hi_g]) if np.isfinite(hi_x) or np.isfinite(hi_g) else np.nan

            cross_ratio_l1000 = _ratio_at_ell(
                ell_cross_ours, dell_cross_ours, ell_cross_collab, dell_cross_collab, target_ell=1000.0
            )
            gal_ratio_l1000 = _ratio_at_ell(
                ell_gal_ours, dell_gal_ours, ell_gal_collab, dell_gal_collab, target_ell=1000.0
            )

            summaries.append(
                CaseSummary(
                    tm=tm,
                    zlo=zlo,
                    zhi=zhi,
                    ell_overlap_min=float(ell_lo) if np.isfinite(ell_lo) else np.nan,
                    ell_overlap_max=float(ell_hi) if np.isfinite(ell_hi) else np.nan,
                    cross_median_ratio_our_over_collab=cross_med_ratio,
                    gal_auto_median_ratio_our_over_collab=gal_med_ratio,
                    cross_ratio_at_ell1000=cross_ratio_l1000,
                    gal_ratio_at_ell1000=gal_ratio_l1000,
                )
            )

            _plot_case(
                outdir=outdir,
                tm=tm,
                zlo=zlo,
                zhi=zhi,
                ell_cross_collab=ell_cross_collab,
                dell_cross_collab=dell_cross_collab,
                ell_cross_ours=ell_cross_ours,
                dell_cross_ours=dell_cross_ours,
                ell_gal_collab=ell_gal_collab,
                dell_gal_collab=dell_gal_collab,
                ell_gal_ours=ell_gal_ours,
                dell_gal_ours=dell_gal_ours,
                collab_flux=collab_flux,
                ours_flux=our_flux,
                collab_gal=collab_gal,
                ours_gal=our_gal,
            )

            print(
                f"TM{tm} z=[{zlo:.1f},{zhi:.1f}] "
                f"cross_med_ratio={cross_med_ratio:.3f}, gal_med_ratio={gal_med_ratio:.3f}, "
                f"cross_ratio_l1000={cross_ratio_l1000:.3f}, gal_ratio_l1000={gal_ratio_l1000:.3f}"
            )

    summary_path = outdir / "summary_hsc_i.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("TM zlo zhi ell_overlap_min ell_overlap_max cross_med_ratio_our_over_collab gal_med_ratio_our_over_collab cross_ratio_at_l1000 gal_ratio_at_l1000\n")
        for s in summaries:
            f.write(
                f"{s.tm} {s.zlo:.1f} {s.zhi:.1f} {s.ell_overlap_min:.1f} {s.ell_overlap_max:.1f} "
                f"{s.cross_median_ratio_our_over_collab:.6f} {s.gal_auto_median_ratio_our_over_collab:.6f} "
                f"{s.cross_ratio_at_ell1000:.6f} {s.gal_ratio_at_ell1000:.6f}\n"
            )

    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
