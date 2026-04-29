#!/usr/bin/env python3
"""Compare one-point distributions between collaborator and local saved maps.

This starts from map products only (no power spectra):
- collaborator pickles in data/jordan_mocks/v3/data_for_richard_pix_6.0
- local boxed outputs in data/jordan_mocks/v3_boxed_outputs/tiles_8p0deg
"""

from pathlib import Path
import pickle
import numpy as np


COLLAB_DIR = Path("/Users/richardfeder/Documents/ciber/data/jordan_mocks/v3/data_for_richard_pix_6.0")
OUR_BASE = Path("/Users/richardfeder/Documents/ciber/data/jordan_mocks/v3_boxed_outputs/tiles_8p0deg")
OUTDIR = OUR_BASE / "diagnostics" / "collab_comparison_pix6"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTPATH = OUTDIR / "onepoint_map_comparison_hsc_i.txt"

ZBINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]


def load_collab_pickle(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        return np.asarray(pickle.load(f))


def load_our_flux(tm: int, zlo: float, zhi: float) -> np.ndarray:
    p = (
        OUR_BASE
        / "mock_maps"
        / "intensity"
        / f"TM{tm}"
        / f"rlz1_TM{tm}_hsc_i_lt_25.0_CIBERfidmask_zmin={zlo:.1f}_zmax={zhi:.1f}_pred_tile000_8.0deg_intensity.npz"
    )
    d = np.load(p)
    return np.asarray(d["ciber_map"])


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
        return np.full_like(counts, np.nan, dtype=float)
    return (counts - m) / m


def stats(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "q01": np.nan,
            "q10": np.nan,
            "q50": np.nan,
            "q90": np.nan,
            "q99": np.nan,
        }
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "q01": float(np.quantile(x, 0.01)),
        "q10": float(np.quantile(x, 0.10)),
        "q50": float(np.quantile(x, 0.50)),
        "q90": float(np.quantile(x, 0.90)),
        "q99": float(np.quantile(x, 0.99)),
    }


def cdf_l1(x: np.ndarray, y: np.ndarray, nbins: int = 1000) -> float:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return np.nan

    lo = min(float(np.min(x)), float(np.min(y)))
    hi = max(float(np.max(x)), float(np.max(y)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.nan

    edges = np.linspace(lo, hi, nbins + 1)
    hx, _ = np.histogram(x, bins=edges, density=True)
    hy, _ = np.histogram(y, bins=edges, density=True)
    dx = edges[1] - edges[0]
    cfx = np.cumsum(hx) * dx
    cfy = np.cumsum(hy) * dx
    return float(np.sum(np.abs(cfx - cfy)) * dx)


def fmt_line(tm: int, zlo: float, zhi: float, map_type: str, d: float, s1: dict, s2: dict) -> str:
    return (
        f"{tm} {zlo:.1f} {zhi:.1f} {map_type} {d:.6e} "
        f"{s1['mean']:.6e} {s2['mean']:.6e} "
        f"{s1['std']:.6e} {s2['std']:.6e} "
        f"{s1['q01']:.6e} {s2['q01']:.6e} "
        f"{s1['q10']:.6e} {s2['q10']:.6e} "
        f"{s1['q50']:.6e} {s2['q50']:.6e} "
        f"{s1['q90']:.6e} {s2['q90']:.6e} "
        f"{s1['q99']:.6e} {s2['q99']:.6e}"
    )


def main() -> None:
    lines = []
    lines.append("one-point comparison: collaborator(10x10,pix6) vs ours(8x8,tile000), hsc_i")
    lines.append(
        "TM zlo zhi map_type cdf_L1 collab_mean ours_mean collab_std ours_std "
        "collab_q01 ours_q01 collab_q10 ours_q10 collab_q50 ours_q50 collab_q90 ours_q90 collab_q99 ours_q99"
    )

    for tm in [1, 2]:
        for zlo, zhi in ZBINS:
            collab_flux_path = COLLAB_DIR / f"img_flux_band_ciber_{tm}_z_{zlo:.1f}_{zhi:.1f}.pkl"
            collab_gal_path = COLLAB_DIR / f"img_num_band_i_z_{zlo:.1f}_{zhi:.1f}.pkl"

            if collab_flux_path.exists():
                collab_flux = load_collab_pickle(collab_flux_path)
                our_flux = load_our_flux(tm, zlo, zhi)
                d = cdf_l1(collab_flux.ravel(), our_flux.ravel())
                lines.append(
                    fmt_line(tm, zlo, zhi, "intensity", d, stats(collab_flux), stats(our_flux))
                )

            if collab_gal_path.exists():
                collab_gal = load_collab_pickle(collab_gal_path)
                our_gal = load_our_gal_overdens(tm, zlo, zhi)
                d = cdf_l1(collab_gal.ravel(), our_gal.ravel())
                lines.append(
                    fmt_line(tm, zlo, zhi, "gal_overdens_i", d, stats(collab_gal), stats(our_gal))
                )

    OUTPATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {OUTPATH}")
    for ln in lines[:8]:
        print(ln)


if __name__ == "__main__":
    main()
