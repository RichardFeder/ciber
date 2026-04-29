#!/usr/bin/env python3
"""Analyze intensity-auto reconstruction from mock cross/galaxy spectra on 8x8 maps.

This script tests two reconstruction routes for each sample/TM:
1) Per-slice prediction:   C_ell^I,rec(z) = C_ell^x(z)^2 / C_ell^g(z), then sum over z-bins.
2) Direct z<1 prediction: build summed z<1 intensity/count maps, compute C_ell^x(z<1),
   C_ell^g(z<1), and use C_ell^I,rec(z<1) = C_ell^x(z<1)^2 / C_ell^g(z<1).

Both are compared to the true z<1 intensity auto from summed intensity maps.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ciber.core.powerspec_pipeline import get_power_spec
from ciber.core.powerspec_utils import compute_knox_errors_from_model


ZBINS: List[Tuple[float, float]] = [
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.0),
]

SAMPLES = {
    "hsc": "hsc_i_lt_25.0_CIBERfidmask",
    "sdss": "sdss_z_lt_22.0_CIBERfidmask",
}


@dataclass
class ReconstructionResult:
    lb: np.ndarray
    clI_true_zlt1: np.ndarray
    clI_true_zlt1_err: np.ndarray
    clI_pred_sum_slices: np.ndarray
    clI_pred_sum_slices_err: np.ndarray
    clI_pred_direct_zlt1: np.ndarray
    clI_pred_direct_zlt1_err: np.ndarray
    clI_true_sum_slices: np.ndarray
    clI_true_sum_slices_err: np.ndarray
    clx_zlt1: np.ndarray
    clx_zlt1_err: np.ndarray
    clg_zlt1: np.ndarray
    clg_zlt1_err: np.ndarray
    clx_sum_slices: np.ndarray
    clx_sum_slices_err: np.ndarray
    clg_sum_slices: np.ndarray
    clg_sum_slices_err: np.ndarray


@dataclass
class ShotSubtractionInfo:
    target_ell: float
    ell_used: float
    index_used: int
    shot_level: float
    shot_level_err: float


@dataclass
class RatioSummaryRow:
    tm: int
    sample: str
    n_ell: int
    direct_median: float
    direct_p16: float
    direct_p84: float
    direct_frac_consistent_unity: float
    direct_xg_weighted_median: float
    direct_rescaled_median: float
    direct_rescaled_p16: float
    direct_rescaled_p84: float
    direct_rescaled_frac_consistent_unity: float
    sum_median: float
    sum_p16: float
    sum_p84: float
    sum_frac_consistent_unity: float
    sum_xg_weighted_median: float
    sum_rescaled_median: float
    sum_rescaled_p16: float
    sum_rescaled_p84: float
    sum_rescaled_frac_consistent_unity: float


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=float)
    m = np.isfinite(num) & np.isfinite(den) & (den > 0)
    out[m] = num[m] / den[m]
    return out


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=float)
    m = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out[m] = num[m] / den[m]
    return out


def _safe_divide_by_scalar(num: np.ndarray, den: float) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=float)
    if not np.isfinite(den) or den == 0:
        return out
    m = np.isfinite(num)
    out[m] = num[m] / den
    return out


def _predict_intensity_auto_and_err(
    clx: np.ndarray,
    clx_err: np.ndarray,
    clg: np.ndarray,
    clg_err: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    cl_pred = _safe_ratio(clx ** 2, clg)
    cl_pred_err = np.full_like(cl_pred, np.nan, dtype=float)

    m = (
        np.isfinite(clx)
        & np.isfinite(clx_err)
        & np.isfinite(clg)
        & np.isfinite(clg_err)
        & (clg > 0)
    )
    if np.any(m):
        dfdx = 2.0 * clx[m] / clg[m]
        dfdg = -(clx[m] ** 2) / (clg[m] ** 2)
        cl_pred_err[m] = np.sqrt((dfdx * clx_err[m]) ** 2 + (dfdg * clg_err[m]) ** 2)
    return cl_pred, cl_pred_err


def _estimate_shot_noise_level(
    ell: np.ndarray,
    cl: np.ndarray,
    cl_err: Optional[np.ndarray],
    target_ell: float,
) -> ShotSubtractionInfo:
    idx = int(np.nanargmin(np.abs(ell - target_ell)))
    shot_level = float(cl[idx])
    shot_level_err = float(cl_err[idx]) if cl_err is not None else np.nan
    return ShotSubtractionInfo(
        target_ell=target_ell,
        ell_used=float(ell[idx]),
        index_used=idx,
        shot_level=shot_level,
        shot_level_err=shot_level_err,
    )


def _subtract_shot_noise(
    cl: np.ndarray,
    cl_err: np.ndarray,
    shot_info: ShotSubtractionInfo,
) -> Tuple[np.ndarray, np.ndarray]:
    cl_sub = cl - shot_info.shot_level
    cl_sub_err = np.sqrt(cl_err ** 2 + shot_info.shot_level_err ** 2)
    return cl_sub, cl_sub_err


def _ratio_with_uncertainty(
    num: np.ndarray,
    num_err: np.ndarray,
    den: np.ndarray,
    den_err: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    ratio = _safe_divide(num, den)
    ratio_err = np.full_like(ratio, np.nan, dtype=float)

    m = (
        np.isfinite(num)
        & np.isfinite(num_err)
        & np.isfinite(den)
        & np.isfinite(den_err)
        & (num != 0)
        & (den != 0)
        & np.isfinite(ratio)
    )
    if np.any(m):
        rel_num = num_err[m] / np.abs(num[m])
        rel_den = den_err[m] / np.abs(den[m])
        ratio_err[m] = np.abs(ratio[m]) * np.sqrt(rel_num ** 2 + rel_den ** 2)
    return ratio, ratio_err


def _estimate_bin_edges_from_centers(lb: np.ndarray) -> np.ndarray:
    edges = np.empty(lb.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (lb[:-1] + lb[1:])
    edges[0] = max(1.0, lb[0] - 0.5 * (lb[1] - lb[0]))
    edges[-1] = lb[-1] + 0.5 * (lb[-1] - lb[-2])
    return edges


def _estimate_mode_counts(lb: np.ndarray, map_side_deg: float) -> np.ndarray:
    if lb.size < 2:
        return np.ones_like(lb, dtype=float)

    edges = _estimate_bin_edges_from_centers(lb)
    delta_ell = np.diff(edges)
    area_sr = (map_side_deg * np.pi / 180.0) ** 2
    fsky = area_sr / (4.0 * np.pi)
    nmodes = (2.0 * lb + 1.0) * delta_ell * fsky
    return np.clip(nmodes, 1.0, None)


def _estimate_delta_ell_from_lb(lb: np.ndarray) -> np.ndarray:
    if lb.size < 2:
        return np.ones_like(lb, dtype=float)
    edges = _estimate_bin_edges_from_centers(lb)
    delta_ell = np.diff(edges)
    return np.clip(delta_ell, 1e-6, None)


def _estimate_fsky_from_map_side_deg(map_side_deg: float) -> float:
    return float(map_side_deg ** 2) / 41253.0


def _estimate_auto_err_modecount(
    lb: np.ndarray,
    cl_auto: np.ndarray,
    delta_ell: np.ndarray,
    fsky: float,
) -> np.ndarray:
    return compute_knox_errors_from_model(lb, np.abs(cl_auto), delta_ell, fsky, mode="auto")


def _estimate_cross_err_modecount(
    lb: np.ndarray,
    cl_auto_a: np.ndarray,
    cl_auto_b: np.ndarray,
    cl_cross: np.ndarray,
    delta_ell: np.ndarray,
    fsky: float,
) -> np.ndarray:
    cross_model = np.sqrt(np.clip(cl_auto_a * cl_auto_b + cl_cross ** 2, 0.0, None))
    return compute_knox_errors_from_model(lb, cross_model, delta_ell, fsky, mode="cross")


def _plot_power_with_errorbars(
    ell: np.ndarray,
    cl: np.ndarray,
    cl_err: np.ndarray,
    pf: np.ndarray,
    label: str,
    linewidth: float = 2.0,
    linestyle: str = "-",
    errorevery: int = 2,
) -> None:
    y = pf * cl
    yerr = pf * cl_err
    m = np.isfinite(ell) & np.isfinite(y) & np.isfinite(yerr) & (y > 0) & (yerr >= 0)
    if not np.any(m):
        return

    plt.errorbar(
        ell[m],
        y[m],
        yerr=yerr[m],
        fmt=linestyle,
        linewidth=linewidth,
        elinewidth=0.9,
        capsize=2,
        alpha=0.95,
        label=label,
        errorevery=errorevery,
    )


def _load_slice_spectra(
    base_dir: str,
    tm: int,
    sample_tag: str,
    zmin: float,
    zmax: float,
) -> Dict[str, np.ndarray]:
    fpath = os.path.join(
        base_dir,
        f"mock_ps_pred/TM{tm}/indiv/rlz1_TM{tm}_auto_cross_pred_{sample_tag}_zmin={zmin}_zmax={zmax}_tile000_8.0deg.npz",
    )
    dat = np.load(fpath)
    return {
        "lb": dat["lb"],
        "clI": dat["clI_comb"],
        "clI_err": dat["clI_err_comb"],
        "clg": dat["clg_comb"],
        "clg_err": dat["clg_err_comb"],
        "clx": dat["clx_comb"],
        "clx_err": dat["clx_err_comb"],
    }


def _load_slice_maps(
    base_dir: str,
    tm: int,
    sample_tag: str,
    zmin: float,
    zmax: float,
) -> Tuple[np.ndarray, np.ndarray]:
    i_fpath = os.path.join(
        base_dir,
        f"mock_maps/intensity/TM{tm}/rlz1_TM{tm}_{sample_tag}_zmin={zmin}_zmax={zmax}_pred_tile000_8.0deg_intensity.npz",
    )
    g_fpath = os.path.join(
        base_dir,
        f"mock_maps/galaxy/TM{tm}/rlz1_TM{tm}_{sample_tag}_zmin={zmin}_zmax={zmax}_tile000_8.0deg_galaxy.npz",
    )
    i_map = np.load(i_fpath)["ciber_map"]
    g_counts = np.load(g_fpath)["gal_counts"]
    return i_map, g_counts


def compute_reconstruction_for_case(
    base_dir: str,
    tm: int,
    sample_tag: str,
    uncertainty_model: str = "modecount",
) -> ReconstructionResult:
    slice_specs = []
    intensity_sum = None
    counts_sum = None

    for zmin, zmax in ZBINS:
        sp = _load_slice_spectra(base_dir, tm, sample_tag, zmin, zmax)
        slice_specs.append(sp)

        i_map, g_counts = _load_slice_maps(base_dir, tm, sample_tag, zmin, zmax)
        if intensity_sum is None:
            intensity_sum = np.array(i_map, copy=True)
            counts_sum = np.array(g_counts, copy=True)
        else:
            intensity_sum += i_map
            counts_sum += g_counts

    if intensity_sum is None or counts_sum is None:
        raise RuntimeError("No slice maps loaded")

    lb = slice_specs[0]["lb"]
    pixsize_arcsec = 8.0 * 3600.0 / float(intensity_sum.shape[0])
    map_side_deg = float(intensity_sum.shape[0]) * pixsize_arcsec / 3600.0
    delta_ell = _estimate_delta_ell_from_lb(lb)
    fsky = _estimate_fsky_from_map_side_deg(map_side_deg)

    # Route 1: slice-wise prediction then sum predicted components.
    clI_pred_slices = []
    clI_pred_slices_err = []
    clI_true_slices = []
    clI_true_slices_err = []
    clx_slices = []
    clx_slices_err = []
    clg_slices = []
    clg_slices_err = []
    for sp in slice_specs:
        if uncertainty_model == "modecount":
            clI_err_slice = _estimate_auto_err_modecount(lb, sp["clI"], delta_ell, fsky)
            clg_err_slice = _estimate_auto_err_modecount(lb, sp["clg"], delta_ell, fsky)
            clx_err_slice = _estimate_cross_err_modecount(lb, sp["clI"], sp["clg"], sp["clx"], delta_ell, fsky)
        else:
            clI_err_slice = sp["clI_err"]
            clg_err_slice = sp["clg_err"]
            clx_err_slice = sp["clx_err"]

        clI_pred_slice, clI_pred_slice_err = _predict_intensity_auto_and_err(
            sp["clx"], clx_err_slice, sp["clg"], clg_err_slice
        )
        clI_pred_slices.append(clI_pred_slice)
        clI_pred_slices_err.append(clI_pred_slice_err)
        clI_true_slices.append(sp["clI"])
        clI_true_slices_err.append(clI_err_slice)
        clx_slices.append(sp["clx"])
        clx_slices_err.append(clx_err_slice)
        clg_slices.append(sp["clg"])
        clg_slices_err.append(clg_err_slice)

    clI_pred_sum_slices = np.nansum(np.array(clI_pred_slices), axis=0)
    clI_pred_sum_slices_err = np.sqrt(np.nansum(np.array(clI_pred_slices_err) ** 2, axis=0))
    clI_true_sum_slices = np.nansum(np.array(clI_true_slices), axis=0)
    clI_true_sum_slices_err = np.sqrt(np.nansum(np.array(clI_true_slices_err) ** 2, axis=0))
    clx_sum_slices = np.nansum(np.array(clx_slices), axis=0)
    clx_sum_slices_err = np.sqrt(np.nansum(np.array(clx_slices_err) ** 2, axis=0))
    clg_sum_slices = np.nansum(np.array(clg_slices), axis=0)
    clg_sum_slices_err = np.sqrt(np.nansum(np.array(clg_slices_err) ** 2, axis=0))

    # Route 2 + ground truth for z<1 from summed maps.
    meansub_intensity = intensity_sum - np.mean(intensity_sum)
    gal_overdens = (counts_sum - np.mean(counts_sum)) / np.mean(counts_sum)

    nbins_ps = 26

    lb2, clI_true_zlt1, clI_true_zlt1_err = get_power_spec(meansub_intensity, nbins=nbins_ps, pixsize=pixsize_arcsec)
    _, clg_zlt1, clg_zlt1_err = get_power_spec(gal_overdens, nbins=nbins_ps, pixsize=pixsize_arcsec)
    _, clx_zlt1, clx_zlt1_err = get_power_spec(meansub_intensity, map_b=gal_overdens, nbins=nbins_ps, pixsize=pixsize_arcsec)

    if uncertainty_model == "modecount":
        delta_ell_zlt1 = _estimate_delta_ell_from_lb(lb2)
        clI_true_zlt1_err = _estimate_auto_err_modecount(lb2, clI_true_zlt1, delta_ell_zlt1, fsky)
        clg_zlt1_err = _estimate_auto_err_modecount(lb2, clg_zlt1, delta_ell_zlt1, fsky)
        clx_zlt1_err = _estimate_cross_err_modecount(
            lb2,
            clI_true_zlt1,
            clg_zlt1,
            clx_zlt1,
            delta_ell_zlt1,
            fsky,
        )

    # Align by index; in practice lb should match for same nbins/pixsize choices.
    if lb2.shape != lb.shape:
        n = min(lb.shape[0], lb2.shape[0])
        lb = lb[:n]
        clI_true_zlt1 = clI_true_zlt1[:n]
        clI_true_zlt1_err = clI_true_zlt1_err[:n]
        clg_zlt1 = clg_zlt1[:n]
        clg_zlt1_err = clg_zlt1_err[:n]
        clx_zlt1 = clx_zlt1[:n]
        clx_zlt1_err = clx_zlt1_err[:n]
        clI_pred_sum_slices = clI_pred_sum_slices[:n]
        clI_pred_sum_slices_err = clI_pred_sum_slices_err[:n]
        clI_true_sum_slices = clI_true_sum_slices[:n]
        clI_true_sum_slices_err = clI_true_sum_slices_err[:n]
        clx_sum_slices = clx_sum_slices[:n]
        clx_sum_slices_err = clx_sum_slices_err[:n]
        clg_sum_slices = clg_sum_slices[:n]
        clg_sum_slices_err = clg_sum_slices_err[:n]

    clI_pred_direct_zlt1, clI_pred_direct_zlt1_err = _predict_intensity_auto_and_err(
        clx_zlt1, clx_zlt1_err, clg_zlt1, clg_zlt1_err
    )

    return ReconstructionResult(
        lb=lb,
        clI_true_zlt1=clI_true_zlt1,
        clI_true_zlt1_err=clI_true_zlt1_err,
        clI_pred_sum_slices=clI_pred_sum_slices,
        clI_pred_sum_slices_err=clI_pred_sum_slices_err,
        clI_pred_direct_zlt1=clI_pred_direct_zlt1,
        clI_pred_direct_zlt1_err=clI_pred_direct_zlt1_err,
        clI_true_sum_slices=clI_true_sum_slices,
        clI_true_sum_slices_err=clI_true_sum_slices_err,
        clx_zlt1=clx_zlt1,
        clx_zlt1_err=clx_zlt1_err,
        clg_zlt1=clg_zlt1,
        clg_zlt1_err=clg_zlt1_err,
        clx_sum_slices=clx_sum_slices,
        clx_sum_slices_err=clx_sum_slices_err,
        clg_sum_slices=clg_sum_slices,
        clg_sum_slices_err=clg_sum_slices_err,
    )


def _plot_case(
    outdir: str,
    tm: int,
    sample_tag: str,
    res: ReconstructionResult,
    shot_ell: float,
) -> Tuple[ShotSubtractionInfo, ShotSubtractionInfo, ShotSubtractionInfo]:
    os.makedirs(outdir, exist_ok=True)
    ell = res.lb
    pf = ell * (ell + 1.0) / (2.0 * np.pi)

    shot_true = _estimate_shot_noise_level(ell, res.clI_true_zlt1, res.clI_true_zlt1_err, target_ell=shot_ell)
    shot_pred_direct = _estimate_shot_noise_level(
        ell, res.clI_pred_direct_zlt1, res.clI_pred_direct_zlt1_err, target_ell=shot_ell
    )
    shot_pred_sum = _estimate_shot_noise_level(
        ell, res.clI_pred_sum_slices, res.clI_pred_sum_slices_err, target_ell=shot_ell
    )

    clI_true_clust, clI_true_clust_err = _subtract_shot_noise(res.clI_true_zlt1, res.clI_true_zlt1_err, shot_true)
    clI_pred_direct_clust, clI_pred_direct_clust_err = _subtract_shot_noise(
        res.clI_pred_direct_zlt1, res.clI_pred_direct_zlt1_err, shot_pred_direct
    )
    clI_pred_sum_clust, clI_pred_sum_clust_err = _subtract_shot_noise(
        res.clI_pred_sum_slices, res.clI_pred_sum_slices_err, shot_pred_sum
    )

    plt.figure(figsize=(6.4, 5.2))
    _plot_power_with_errorbars(
        ell,
        clI_true_clust,
        clI_true_clust_err,
        pf,
        label="true z<1 intensity clustering",
        linewidth=2.2,
    )
    _plot_power_with_errorbars(
        ell,
        clI_pred_direct_clust,
        clI_pred_direct_clust_err,
        pf,
        label="pred clustering from direct ratio",
        linewidth=2.0,
    )
    _plot_power_with_errorbars(
        ell,
        clI_pred_sum_clust,
        clI_pred_sum_clust_err,
        pf,
        label="pred clustering from sum of slice preds",
        linewidth=2.0,
    )
    _plot_power_with_errorbars(
        ell,
        res.clI_true_sum_slices,
        res.clI_true_sum_slices_err,
        pf,
        label="sum of true slice autos",
        linewidth=1.7,
        linestyle="--",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$D_{\ell}$")
    plt.title(f"TM{tm}, {sample_tag}: clustering-only intensity-auto reconstruction")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"TM{tm}_{sample_tag}_reconstruction_comparison.png"), dpi=180)
    plt.close()

    # Ratio-to-truth panel
    clI_true_clust, clI_true_clust_err = _subtract_shot_noise(res.clI_true_zlt1, res.clI_true_zlt1_err, shot_true)
    clI_pred_direct_clust, clI_pred_direct_clust_err = _subtract_shot_noise(
        res.clI_pred_direct_zlt1, res.clI_pred_direct_zlt1_err, shot_pred_direct
    )
    clI_pred_sum_clust, clI_pred_sum_clust_err = _subtract_shot_noise(
        res.clI_pred_sum_slices, res.clI_pred_sum_slices_err, shot_pred_sum
    )

    r_direct, r_direct_err = _ratio_with_uncertainty(
        clI_pred_direct_clust,
        clI_pred_direct_clust_err,
        clI_true_clust,
        clI_true_clust_err,
    )
    r_sum, r_sum_err = _ratio_with_uncertainty(
        clI_pred_sum_clust,
        clI_pred_sum_clust_err,
        clI_true_clust,
        clI_true_clust_err,
    )

    plt.figure(figsize=(6.4, 4.4))
    plt.plot(ell, r_direct, label="direct z<1 ratio / true", linewidth=2.0)
    plt.fill_between(ell, r_direct - r_direct_err, r_direct + r_direct_err, alpha=0.2)
    plt.plot(ell, r_sum, label="sum(slice preds) / true", linewidth=2.0)
    plt.fill_between(ell, r_sum - r_sum_err, r_sum + r_sum_err, alpha=0.2)
    plt.axhline(1.0, color="k", linestyle="--", linewidth=1.0)
    plt.xscale("log")
    plt.ylim(0, 2.5)
    plt.xlabel(r"$\ell$")
    plt.ylabel("clustering reconstruction / clustering truth")
    plt.title(f"TM{tm}, {sample_tag}: clustering reconstruction bias")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"TM{tm}_{sample_tag}_reconstruction_ratio_to_truth.png"), dpi=180)
    plt.close()

    return shot_true, shot_pred_direct, shot_pred_sum


def _write_summary_text(
    outdir: str,
    summaries: List[RatioSummaryRow],
    shot_rows: List[Tuple[int, str, str, float, float, float]],
    ell_min: float,
    ell_max: float,
    shot_ell: float,
    uncertainty_model: str,
) -> None:
    outpath = os.path.join(outdir, "reconstruction_summary.txt")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(f"ell_range_for_summary = [{ell_min:.1f}, {ell_max:.1f}]\n")
        f.write(f"shot_noise_reference_ell = {shot_ell:.1f}\n")
        f.write(f"uncertainty_model = {uncertainty_model}\n")
        f.write("\n")
        f.write("Shot-noise subtraction levels (from C_ell at nearest ell to shot reference):\n")
        f.write("TM sample spectrum ell_used shot_level shot_err\n")
        for tm, sample, spectrum, ell_used, shot_level, shot_err in shot_rows:
            f.write(f"{tm} {sample} {spectrum} {ell_used:.1f} {shot_level:.6e} {shot_err:.6e}\n")

        f.write("\n")
        f.write(
            "TM sample n_ell "
            "direct_med direct_p16 direct_p84 direct_frac_unity_1sigma "
            "direct_xg_wmed direct_rescaled_med direct_rescaled_p16 direct_rescaled_p84 direct_rescaled_frac_unity_1sigma "
            "sum_med sum_p16 sum_p84 sum_frac_unity_1sigma "
            "sum_xg_wmed sum_rescaled_med sum_rescaled_p16 sum_rescaled_p84 sum_rescaled_frac_unity_1sigma\n"
        )
        for row in summaries:
            f.write(
                f"{row.tm} {row.sample} {row.n_ell} "
                f"{row.direct_median:.4f} {row.direct_p16:.4f} {row.direct_p84:.4f} {row.direct_frac_consistent_unity:.3f} "
                f"{row.direct_xg_weighted_median:.4f} {row.direct_rescaled_median:.4f} {row.direct_rescaled_p16:.4f} {row.direct_rescaled_p84:.4f} {row.direct_rescaled_frac_consistent_unity:.3f} "
                f"{row.sum_median:.4f} {row.sum_p16:.4f} {row.sum_p84:.4f} {row.sum_frac_consistent_unity:.3f} "
                f"{row.sum_xg_weighted_median:.4f} {row.sum_rescaled_median:.4f} {row.sum_rescaled_p16:.4f} {row.sum_rescaled_p84:.4f} {row.sum_rescaled_frac_consistent_unity:.3f}\n"
            )


def _compute_summary_stats(ratio: np.ndarray, ratio_err: np.ndarray) -> Tuple[float, float, float, float, int]:
    valid = np.isfinite(ratio)
    if not np.any(valid):
        return np.nan, np.nan, np.nan, np.nan, 0

    vals = ratio[valid]
    med = float(np.nanmedian(vals))
    p16, p84 = np.nanpercentile(vals, [16.0, 84.0])

    valid_err = valid & np.isfinite(ratio_err) & (ratio_err > 0)
    if np.any(valid_err):
        lo = ratio[valid_err] - ratio_err[valid_err]
        hi = ratio[valid_err] + ratio_err[valid_err]
        frac_unity = float(np.mean((lo <= 1.0) & (hi >= 1.0)))
    else:
        frac_unity = np.nan

    return med, float(p16), float(p84), frac_unity, int(np.sum(valid))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(m):
        return np.nan

    v = values[m]
    w = weights[m]
    order = np.argsort(v)
    v = v[order]
    w = w[order]

    csum = np.cumsum(w)
    cutoff = 0.5 * np.sum(w)
    idx = int(np.searchsorted(csum, cutoff, side="left"))
    idx = min(max(idx, 0), len(v) - 1)
    return float(v[idx])


def main() -> None:
    parser = argparse.ArgumentParser(description="Test intensity-auto reconstruction on 8x8 mock spectra")
    parser.add_argument(
        "--base-dir",
        default="/Users/richardfeder/Documents/ciber/data/jordan_mocks/v3_boxed_outputs/tiles_8p0deg",
        help="Base directory with mock_ps_pred and mock_maps",
    )
    parser.add_argument(
        "--outdir",
        default="/Users/richardfeder/Documents/ciber/data/jordan_mocks/v3_boxed_outputs/tiles_8p0deg/diagnostics/reconstruction_checks",
        help="Output directory for plots and summary",
    )
    parser.add_argument(
        "--ell-min",
        type=float,
        default=300.0,
        help="Minimum ell for summary metrics (matches predict_ciber_auto_vs_redshift default)",
    )
    parser.add_argument(
        "--ell-max",
        type=float,
        default=3000.0,
        help="Maximum ell for summary metrics (matches predict_ciber_auto_vs_redshift default)",
    )
    parser.add_argument(
        "--shot-ell",
        type=float,
        default=10000.0,
        help="Reference ell at which C_ell level is treated as shot noise and subtracted",
    )
    parser.add_argument(
        "--uncertainty-model",
        choices=["modecount", "bandpower"],
        default="modecount",
        help=(
            "Uncertainty model for errorbars/ratio errors: "
            "'modecount' uses core Knox utilities with measured C_ell as model and map fsky; "
            "'bandpower' uses per-bandpower errors returned by the spectrum estimator"
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    summary_rows: List[RatioSummaryRow] = []
    shot_rows: List[Tuple[int, str, str, float, float, float]] = []

    for tm in [1, 2]:
        for sample_key, sample_tag in SAMPLES.items():
            res = compute_reconstruction_for_case(
                args.base_dir,
                tm,
                sample_tag,
                uncertainty_model=args.uncertainty_model,
            )
            shot_true, shot_pred_direct, shot_pred_sum = _plot_case(
                args.outdir,
                tm,
                sample_tag,
                res,
                shot_ell=args.shot_ell,
            )

            shot_rows.extend(
                [
                    (tm, sample_key, "true_zlt1", shot_true.ell_used, shot_true.shot_level, shot_true.shot_level_err),
                    (
                        tm,
                        sample_key,
                        "pred_direct_zlt1",
                        shot_pred_direct.ell_used,
                        shot_pred_direct.shot_level,
                        shot_pred_direct.shot_level_err,
                    ),
                    (
                        tm,
                        sample_key,
                        "pred_sum_slices",
                        shot_pred_sum.ell_used,
                        shot_pred_sum.shot_level,
                        shot_pred_sum.shot_level_err,
                    ),
                ]
            )

            ell_mask = (res.lb > args.ell_min) & (res.lb < args.ell_max)
            if not np.any(ell_mask):
                raise RuntimeError(
                    f"No ell bins found in requested summary range [{args.ell_min}, {args.ell_max}]"
                )

            cl_true_sub, cl_true_sub_err = _subtract_shot_noise(res.clI_true_zlt1, res.clI_true_zlt1_err, shot_true)
            cl_direct_sub, cl_direct_sub_err = _subtract_shot_noise(
                res.clI_pred_direct_zlt1, res.clI_pred_direct_zlt1_err, shot_pred_direct
            )
            cl_sum_sub, cl_sum_sub_err = _subtract_shot_noise(
                res.clI_pred_sum_slices, res.clI_pred_sum_slices_err, shot_pred_sum
            )

            ratio_direct, ratio_direct_err = _ratio_with_uncertainty(
                cl_direct_sub,
                cl_direct_sub_err,
                cl_true_sub,
                cl_true_sub_err,
            )
            ratio_sum, ratio_sum_err = _ratio_with_uncertainty(
                cl_sum_sub,
                cl_sum_sub_err,
                cl_true_sub,
                cl_true_sub_err,
            )

            med_direct, p16_direct, p84_direct, frac_unity_direct, n_ell_direct = _compute_summary_stats(
                ratio_direct[ell_mask], ratio_direct_err[ell_mask]
            )
            med_sum, p16_sum, p84_sum, frac_unity_sum, n_ell_sum = _compute_summary_stats(
                ratio_sum[ell_mask], ratio_sum_err[ell_mask]
            )

            # Weighted-median normalization from cross/galaxy ratio Cx/Cg.
            # Build a fixed-bias model C_I,pred = (b_med)^2 * C_g,clust with
            # b_med = weighted median of Cx/Cg over the selected ell range.
            shot_g_direct = _estimate_shot_noise_level(res.lb, res.clg_zlt1, res.clg_zlt1_err, target_ell=args.shot_ell)
            clg_direct_sub, clg_direct_sub_err = _subtract_shot_noise(res.clg_zlt1, res.clg_zlt1_err, shot_g_direct)

            shot_g_sum = _estimate_shot_noise_level(
                res.lb, res.clg_sum_slices, res.clg_sum_slices_err, target_ell=args.shot_ell
            )
            clg_sum_sub, clg_sum_sub_err = _subtract_shot_noise(res.clg_sum_slices, res.clg_sum_slices_err, shot_g_sum)

            ratio_xg_direct, ratio_xg_direct_err = _ratio_with_uncertainty(
                res.clx_zlt1,
                res.clx_zlt1_err,
                clg_direct_sub,
                clg_direct_sub_err,
            )
            ratio_xg_sum, ratio_xg_sum_err = _ratio_with_uncertainty(
                res.clx_sum_slices,
                res.clx_sum_slices_err,
                clg_sum_sub,
                clg_sum_sub_err,
            )

            w_direct = _safe_divide(np.ones_like(ratio_xg_direct[ell_mask]), ratio_xg_direct_err[ell_mask] ** 2)
            w_sum = _safe_divide(np.ones_like(ratio_xg_sum[ell_mask]), ratio_xg_sum_err[ell_mask] ** 2)

            wmed_direct = _weighted_median(ratio_xg_direct[ell_mask], w_direct)
            wmed_sum = _weighted_median(ratio_xg_sum[ell_mask], w_sum)

            clI_pred_direct_wmed = (wmed_direct ** 2) * clg_direct_sub
            clI_pred_direct_wmed_err = (wmed_direct ** 2) * clg_direct_sub_err
            clI_pred_sum_wmed = (wmed_sum ** 2) * clg_sum_sub
            clI_pred_sum_wmed_err = (wmed_sum ** 2) * clg_sum_sub_err

            ratio_direct_rescaled, ratio_direct_rescaled_err = _ratio_with_uncertainty(
                clI_pred_direct_wmed,
                clI_pred_direct_wmed_err,
                cl_true_sub,
                cl_true_sub_err,
            )
            ratio_sum_rescaled, ratio_sum_rescaled_err = _ratio_with_uncertainty(
                clI_pred_sum_wmed,
                clI_pred_sum_wmed_err,
                cl_true_sub,
                cl_true_sub_err,
            )

            (
                med_direct_rescaled,
                p16_direct_rescaled,
                p84_direct_rescaled,
                frac_unity_direct_rescaled,
                _,
            ) = _compute_summary_stats(ratio_direct_rescaled[ell_mask], ratio_direct_rescaled_err[ell_mask])
            (
                med_sum_rescaled,
                p16_sum_rescaled,
                p84_sum_rescaled,
                frac_unity_sum_rescaled,
                _,
            ) = _compute_summary_stats(ratio_sum_rescaled[ell_mask], ratio_sum_rescaled_err[ell_mask])

            n_ell = min(n_ell_direct, n_ell_sum)
            summary_rows.append(
                RatioSummaryRow(
                    tm=tm,
                    sample=sample_key,
                    n_ell=n_ell,
                    direct_median=med_direct,
                    direct_p16=p16_direct,
                    direct_p84=p84_direct,
                    direct_frac_consistent_unity=frac_unity_direct,
                    direct_xg_weighted_median=wmed_direct,
                    direct_rescaled_median=med_direct_rescaled,
                    direct_rescaled_p16=p16_direct_rescaled,
                    direct_rescaled_p84=p84_direct_rescaled,
                    direct_rescaled_frac_consistent_unity=frac_unity_direct_rescaled,
                    sum_median=med_sum,
                    sum_p16=p16_sum,
                    sum_p84=p84_sum,
                    sum_frac_consistent_unity=frac_unity_sum,
                    sum_xg_weighted_median=wmed_sum,
                    sum_rescaled_median=med_sum_rescaled,
                    sum_rescaled_p16=p16_sum_rescaled,
                    sum_rescaled_p84=p84_sum_rescaled,
                    sum_rescaled_frac_consistent_unity=frac_unity_sum_rescaled,
                )
            )
            print(
                f"TM{tm} {sample_key}: direct/true_clust median={med_direct:.4f} "
                f"(p16,p84)=({p16_direct:.4f}, {p84_direct:.4f}), "
                f"direct_wmed={wmed_direct:.4f}, direct_rescaled_med={med_direct_rescaled:.4f}; "
                f"sum/true_clust median={med_sum:.4f} "
                f"(p16,p84)=({p16_sum:.4f}, {p84_sum:.4f}), "
                f"sum_wmed={wmed_sum:.4f}, sum_rescaled_med={med_sum_rescaled:.4f}; "
                f"frac_ell_consistent_with_unity_1sigma=(direct={frac_unity_direct:.3f}, sum={frac_unity_sum:.3f}) "
                f"for ell in [{args.ell_min:.0f}, {args.ell_max:.0f}], "
                f"shot ell={args.shot_ell:.0f}, uncertainty={args.uncertainty_model}"
            )

    _write_summary_text(
        args.outdir,
        summary_rows,
        shot_rows,
        args.ell_min,
        args.ell_max,
        args.shot_ell,
        args.uncertainty_model,
    )


if __name__ == "__main__":
    main()
