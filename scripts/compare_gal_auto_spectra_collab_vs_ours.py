#!/usr/bin/env python3
"""Compare galaxy auto spectra between collaborator and local boxed outputs.

Uses local get_power_spec for both datasets. Handles different map areas by
using each map's native fov and pixel scale when computing C_ell.
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ciber.core.powerspec_pipeline import get_power_spec


COLLAB_DIR = Path(
    "/Users/richardfeder/Documents/ciber/data/jordan_mocks/v3/data_for_richard_pix_6.0/converted_numpy"
)
OUR_BASE = Path(
    "/Users/richardfeder/Documents/ciber/data/jordan_mocks/v3_boxed_outputs/tiles_8p0deg"
)
OUTDIR = OUR_BASE / "diagnostics" / "collab_comparison_pix6"
OUTDIR.mkdir(parents=True, exist_ok=True)

ZBINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]


def load_collab_overdens_i(zlo: float, zhi: float) -> np.ndarray:
    p = COLLAB_DIR / f"img_num_band_i_z_{zlo:.1f}_{zhi:.1f}.npy"
    return np.load(p)


def load_our_overdens(tm: int, zlo: float, zhi: float) -> np.ndarray:
    p = (
        OUR_BASE
        / "mock_maps"
        / "galaxy"
        / f"TM{tm}"
        / f"rlz1_TM{tm}_hsc_i_lt_25.0_CIBERfidmask_zmin={zlo:.1f}_zmax={zhi:.1f}_tile000_8.0deg_galaxy.npz"
    )
    d = np.load(p)
    counts = np.asarray(d["gal_counts"])
    m = float(np.mean(counts))
    if m <= 0:
        return np.full_like(counts, np.nan)
    return (counts - m) / m


def compute_gal_auto_dell(overdens_map: np.ndarray, fov_deg: float, nbins: int = 20):
    pixsize_arcsec = fov_deg * 3600.0 / float(overdens_map.shape[0])
    lb, cl, clerr = get_power_spec(overdens_map, nbins=nbins, pixsize=pixsize_arcsec)
    pf = lb * (lb + 1.0) / (2.0 * np.pi)
    return lb, pf * cl, pf * clerr


def overlap_median_ratio(lb1, d1, lb2, d2, ell_min=300.0, ell_max=3000.0):
    m1 = np.isfinite(lb1) & np.isfinite(d1) & (d1 > 0)
    m2 = np.isfinite(lb2) & np.isfinite(d2) & (d2 > 0)
    if not np.any(m1) or not np.any(m2):
        return np.nan, np.nan, np.nan

    e1 = lb1[m1]
    y1 = d1[m1]
    e2 = lb2[m2]
    y2 = d2[m2]

    lo = max(float(np.min(e1)), float(np.min(e2)), ell_min)
    hi = min(float(np.max(e1)), float(np.max(e2)), ell_max)
    if hi <= lo:
        return lo, hi, np.nan

    m = (e1 >= lo) & (e1 <= hi)
    if np.sum(m) < 2:
        return lo, hi, np.nan

    y2i = np.interp(e1[m], e2, y2)
    ratio = y1[m] / y2i
    return lo, hi, float(np.nanmedian(ratio))


def plot_tm(tm: int):
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.9), sharey=True)
    rows = []

    for i, (zlo, zhi) in enumerate(ZBINS):
        ax = axes[i]

        collab = load_collab_overdens_i(zlo, zhi)
        ours = load_our_overdens(tm, zlo, zhi)

        lb_c, dell_c, derr_c = compute_gal_auto_dell(collab, fov_deg=10.0, nbins=20)
        lb_o, dell_o, derr_o = compute_gal_auto_dell(ours, fov_deg=8.0, nbins=20)

        m_c = np.isfinite(dell_c) & np.isfinite(derr_c) & (dell_c > 0)
        m_o = np.isfinite(dell_o) & np.isfinite(derr_o) & (dell_o > 0)

        ax.errorbar(lb_c[m_c], dell_c[m_c], yerr=derr_c[m_c], fmt="o-", ms=3, lw=1.2, capsize=2, label="collab 10x10")
        ax.errorbar(lb_o[m_o], dell_o[m_o], yerr=derr_o[m_o], fmt="o-", ms=3, lw=1.2, capsize=2, label="ours 8x8")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(f"z=[{zlo:.1f},{zhi:.1f}]")
        ax.set_xlabel(r"$\ell$")

        lo, hi, medr = overlap_median_ratio(lb_o, dell_o, lb_c, dell_c)
        rows.append((tm, zlo, zhi, lo, hi, medr))

    axes[0].set_ylabel(r"$D_\ell^{gg}$")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"TM{tm}: Galaxy Auto Comparison (hsc_i) using get_power_spec", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    fig_path = OUTDIR / f"TM{tm}_gal_auto_compare_getpowerspec_hsc_i.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    return fig_path, rows


def main():
    all_rows = []
    fig_paths = []

    for tm in [1, 2]:
        fp, rows = plot_tm(tm)
        fig_paths.append(fp)
        all_rows.extend(rows)

    txt = OUTDIR / "gal_auto_compare_getpowerspec_hsc_i.txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write("TM zlo zhi ell_overlap_min ell_overlap_max median_ratio_ours_over_collab\n")
        for tm, zlo, zhi, lo, hi, medr in all_rows:
            f.write(f"{tm} {zlo:.1f} {zhi:.1f} {lo:.1f} {hi:.1f} {medr:.6f}\n")

    print(f"wrote {txt}")
    for fp in fig_paths:
        print(f"wrote {fp}")


if __name__ == "__main__":
    main()
