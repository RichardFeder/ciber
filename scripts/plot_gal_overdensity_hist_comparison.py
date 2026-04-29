#!/usr/bin/env python3
"""Plot galaxy overdensity histogram comparisons in each redshift bin.

Compares collaborator i-band overdensity-like maps against local 8x8 boxed maps.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


COLLAB_CONVERTED = Path(
    "/Users/richardfeder/Documents/ciber/data/jordan_mocks/v3/data_for_richard_pix_6.0/converted_numpy"
)
OUR_BASE = Path(
    "/Users/richardfeder/Documents/ciber/data/jordan_mocks/v3_boxed_outputs/tiles_8p0deg"
)
OUTDIR = OUR_BASE / "diagnostics" / "collab_comparison_pix6"
OUTDIR.mkdir(parents=True, exist_ok=True)

ZBINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]


def load_collab_gal_i(zlo: float, zhi: float) -> np.ndarray:
    p = COLLAB_CONVERTED / f"img_num_band_i_z_{zlo:.1f}_{zhi:.1f}.npy"
    return np.load(p)


def load_our_gal_overdens(tm: int, zlo: float, zhi: float) -> np.ndarray:
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


def plot_tm(tm: int) -> Path:
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), sharey=True)

    for i, (zlo, zhi) in enumerate(ZBINS):
        ax = axes[i]
        c = load_collab_gal_i(zlo, zhi)
        o = load_our_gal_overdens(tm, zlo, zhi)

        c = c[np.isfinite(c)]
        o = o[np.isfinite(o)]

        # Focus on central + moderate tail where both distributions carry most weight.
        hi = np.nanpercentile(np.concatenate([c, o]), 99.8)
        lo = -1.0
        bins = np.linspace(lo, hi, 250)

        ax.hist(c, bins=bins, density=True, histtype="step", linewidth=1.3, label="collab 10x10")
        ax.hist(o, bins=bins, density=True, histtype="step", linewidth=1.3, label="ours 8x8")
        ax.axvline(-1.0, linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_yscale("log")
        ax.set_title(f"z=[{zlo:.1f},{zhi:.1f}]")
        ax.set_xlabel(r"$\delta_g$")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("pdf")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"TM{tm}: Galaxy Overdensity Histogram Comparison (hsc_i)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    outpath = OUTDIR / f"TM{tm}_gal_overdensity_hist_comparison_hsc_i.png"
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    return outpath


def main() -> None:
    outs = []
    for tm in [1, 2]:
        outs.append(plot_tm(tm))
    for p in outs:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
