#!/usr/bin/env python3
"""Generate CIBER galaxy-cross paper figures from a single CLI.

This script wraps plotting functions used across notebooks and provides:
- subcommands for individual figures
- batch generation with `all`
- optional prediction-source switching
- per-figure timing diagnostics (load/plot/save/total)
- diagnostics persistence to JSON and CSV

Notes
-----
- Default mode preserves existing plotting-function defaults.
- Standard IGL mode requires `--pred-basepath`.
- Some figure types require precomputed input files (see subcommand help).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# -----------------------------
# Dataclasses and timing helpers
# -----------------------------

@dataclass
class FigureTiming:
    figure_key: str
    output_path: str
    load_sec: float
    plot_sec: float
    save_sec: float
    total_sec: float
    status: str
    error: str = ""


@dataclass
class GeneratedFigure:
    figure_key: str
    fig: Any
    stem: str


def _now() -> float:
    return time.perf_counter()


def _ensure_trailing_sep(path: str) -> str:
    return path if path.endswith(os.sep) else path + os.sep


def _normalize_pred_basepath(basepath: str) -> str:
    """Normalize prediction base path to root that contains mock_ps_pred/.

    Accepts either:
    - .../data/jordan_mocks/v2/
    - .../data/jordan_mocks/v3_boxed_outputs/tiles_10p0deg/
    - .../mock_ps_pred/
    """
    p = Path(basepath).expanduser()
    if p.name == "mock_ps_pred":
        p = p.parent
    if not p.exists():
        raise FileNotFoundError(f"Prediction base path does not exist: {p}")
    return _ensure_trailing_sep(str(p))


def _prediction_path(basepath: str, inst: int, addstr: str) -> str:
    return os.path.join(
        basepath,
        "mock_ps_pred",
        f"TM{inst}",
        "field_average",
        f"pred_cls_TM{inst}_{addstr}.npz",
    )


def _safe_as_float_array(arr: Any) -> Optional[np.ndarray]:
    if arr is None:
        return None
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        if arr.shape == () and arr.item() is None:
            return None
    out = np.asarray(arr)
    if out.dtype == object and out.shape == () and out.item() is None:
        return None
    return np.asarray(out, dtype=float)


def _load_beam_ell(inst: int, ifield: int, target_lb: np.ndarray) -> Optional[np.ndarray]:
    """Load B_ell for TM{inst}, mapped to target ell bins.

    Beam files store per-field B_ell arrays without explicit lb. If lengths differ,
    perform a conservative index-based interpolation to target length.
    """
    bls_fpath = (
        REPO_ROOT
        / "data"
        / "fluctuation_data"
        / f"TM{inst}"
        / "beam_correction"
        / f"bl_est_postage_stamps_TM{inst}_081121.npz"
    )
    if not bls_fpath.exists():
        return None

    dat = np.load(str(bls_fpath), allow_pickle=True)
    if "B_ells_post" not in dat:
        return None

    B_ells_post = np.asarray(dat["B_ells_post"], dtype=float)
    if B_ells_post.ndim != 2 or B_ells_post.shape[0] == 0:
        return None

    if "ifield_list" in dat:
        ifields = np.asarray(dat["ifield_list"]).astype(int).tolist()
        if ifield in ifields:
            idx = ifields.index(ifield)
        else:
            idx = 0
    else:
        idx = max(0, min(B_ells_post.shape[0] - 1, ifield - 4))

    b = B_ells_post[idx]
    n_tgt = int(np.size(target_lb))
    if n_tgt <= 0:
        return None
    if b.size == n_tgt:
        out = b
    else:
        x_src = np.linspace(0.0, 1.0, b.size)
        x_tgt = np.linspace(0.0, 1.0, n_tgt)
        out = np.interp(x_tgt, x_src, b)

    # Avoid division spikes from pathological/zero values.
    return np.clip(out, 1.0e-6, None)


def _apply_beam_correction_record(rec: Dict[str, np.ndarray], inst: int, ifield: int) -> Dict[str, np.ndarray]:
    """Apply deconvolution by B_ell consistent with legacy field-average mocks."""
    b_ell = _load_beam_ell(inst=inst, ifield=ifield, target_lb=rec["lb"])
    if b_ell is None:
        return rec

    out = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in rec.items()}
    out["cross"] = out["cross"] / b_ell
    out["cross_err"] = out["cross_err"] / b_ell
    out["intensity_auto_tracer"] = out["intensity_auto_tracer"] / (b_ell**2)
    out["intensity_auto_tracer_err"] = out["intensity_auto_tracer_err"] / (b_ell**2)
    out["intensity_auto_full"] = out["intensity_auto_full"] / (b_ell**2)
    out["intensity_auto_full_err"] = out["intensity_auto_full_err"] / (b_ell**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        rlx = out["cross"] / np.sqrt(out["gal_auto"] * out["intensity_auto_full"])
        rel_cross = np.where(out["cross"] != 0, out["cross_err"] / np.abs(out["cross"]), np.nan)
        rel_gal = np.where(out["gal_auto"] > 0, out["gal_auto_err"] / np.abs(out["gal_auto"]), np.nan)
        rel_ifull = np.where(
            out["intensity_auto_full"] > 0,
            out["intensity_auto_full_err"] / np.abs(out["intensity_auto_full"]),
            np.nan,
        )
        out["rlx_tracer_full"] = rlx
        out["rlx_err_tracer_full"] = np.abs(rlx) * np.sqrt(rel_cross**2 + 0.25 * (rel_gal**2 + rel_ifull**2))

    return out


def _find_single_indiv_file(pattern: str) -> Optional[str]:
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[0]


def _load_indiv_pair(
    basepath: str,
    inst: int,
    tracer_addstr: str,
    zmin: float,
    zmax: float,
    rlz: int = 1,
    beam_correct: bool = False,
    beam_ifield: int = 8,
) -> Optional[Dict[str, np.ndarray]]:
    indiv_dir = os.path.join(basepath, "mock_ps_pred", f"TM{inst}", "indiv")
    pred_pat = os.path.join(
        indiv_dir,
        f"rlz{rlz}_TM{inst}_auto_cross_pred_{tracer_addstr}_zmin={zmin}_zmax={zmax}_tile*.npz",
    )
    full_pat = os.path.join(
        indiv_dir,
        f"rlz{rlz}_TM{inst}_auto_fullCIBER_zmin={zmin}_zmax={zmax}_tile*.npz",
    )

    pred_fp = _find_single_indiv_file(pred_pat)
    full_fp = _find_single_indiv_file(full_pat)
    if pred_fp is None or full_fp is None:
        return None

    pred = np.load(pred_fp, allow_pickle=True)
    full = np.load(full_fp, allow_pickle=True)

    lb = _safe_as_float_array(pred["lb"])
    cross = _safe_as_float_array(pred["clx_comb"])
    cross_err = _safe_as_float_array(pred["clx_err_comb"])
    gal_auto = _safe_as_float_array(pred["clg_comb"])
    gal_auto_err = _safe_as_float_array(pred["clg_err_comb"])
    intensity_auto_tracer = _safe_as_float_array(pred["clI_comb"])
    intensity_auto_tracer_err = _safe_as_float_array(pred["clI_err_comb"])
    intensity_auto_full = _safe_as_float_array(full["clI_comb"])
    intensity_auto_full_err = _safe_as_float_array(full["clI_err_comb"])

    if any(
        x is None
        for x in [
            lb,
            cross,
            cross_err,
            gal_auto,
            gal_auto_err,
            intensity_auto_tracer,
            intensity_auto_tracer_err,
            intensity_auto_full,
            intensity_auto_full_err,
        ]
    ):
        return None

    with np.errstate(divide="ignore", invalid="ignore"):
        rlx = cross / np.sqrt(gal_auto * intensity_auto_full)
        rel_cross = np.where(cross != 0, cross_err / np.abs(cross), np.nan)
        rel_gal = np.where(gal_auto > 0, gal_auto_err / np.abs(gal_auto), np.nan)
        rel_ifull = np.where(
            intensity_auto_full > 0,
            intensity_auto_full_err / np.abs(intensity_auto_full),
            np.nan,
        )
        rlx_err = np.abs(rlx) * np.sqrt(rel_cross**2 + 0.25 * (rel_gal**2 + rel_ifull**2))

    rec = {
        "lb": lb,
        "cross": cross,
        "cross_err": cross_err,
        "gal_auto": gal_auto,
        "gal_auto_err": gal_auto_err,
        "intensity_auto_tracer": intensity_auto_tracer,
        "intensity_auto_tracer_err": intensity_auto_tracer_err,
        "intensity_auto_full": intensity_auto_full,
        "intensity_auto_full_err": intensity_auto_full_err,
        "rlx_tracer_full": rlx,
        "rlx_err_tracer_full": rlx_err,
    }
    if beam_correct:
        rec = _apply_beam_correction_record(rec, inst=inst, ifield=beam_ifield)
    return rec


def _sum_pred_records(records: Sequence[Dict[str, np.ndarray]]) -> Optional[Dict[str, np.ndarray]]:
    if not records:
        return None
    keys_sum = [
        "cross",
        "gal_auto",
        "intensity_auto_tracer",
        "intensity_auto_full",
    ]
    keys_err = [
        "cross_err",
        "gal_auto_err",
        "intensity_auto_tracer_err",
        "intensity_auto_full_err",
    ]
    out: Dict[str, np.ndarray] = {"lb": records[0]["lb"].copy()}
    for k in keys_sum:
        out[k] = np.nansum(np.array([r[k] for r in records]), axis=0)
    for k in keys_err:
        out[k] = np.sqrt(np.nansum(np.array([r[k] for r in records]) ** 2, axis=0))

    with np.errstate(divide="ignore", invalid="ignore"):
        rlx = out["cross"] / np.sqrt(out["gal_auto"] * out["intensity_auto_full"])
        rel_cross = np.where(out["cross"] != 0, out["cross_err"] / np.abs(out["cross"]), np.nan)
        rel_gal = np.where(out["gal_auto"] > 0, out["gal_auto_err"] / np.abs(out["gal_auto"]), np.nan)
        rel_ifull = np.where(
            out["intensity_auto_full"] > 0,
            out["intensity_auto_full_err"] / np.abs(out["intensity_auto_full"]),
            np.nan,
        )
        rlx_err = np.abs(rlx) * np.sqrt(rel_cross**2 + 0.25 * (rel_gal**2 + rel_ifull**2))

    out["rlx_tracer_full"] = rlx
    out["rlx_err_tracer_full"] = rlx_err
    return out


def _load_zlt1_aggregate_record(
    basepath: str,
    inst: int,
    tracer_addstr: str,
    zbinedges: Sequence[float],
    rlz: int = 1,
    beam_correct: bool = False,
    beam_ifield: int = 8,
) -> Optional[Dict[str, np.ndarray]]:
    """Load explicit z<1 aggregate tracer spectra and pair with full-CIBER auto.

    The z<1 tracer file is expected at:
    rlzX_TM{inst}_auto_cross_pred_{tracer_addstr}_zmin=0.0_zmax=1.0_tile*.npz

    Full-CIBER z<1 is assembled by summing per-slice auto_fullCIBER files in
    quadrature for errors when an explicit z<1 full-CIBER file is unavailable.
    """
    indiv_dir = os.path.join(basepath, "mock_ps_pred", f"TM{inst}", "indiv")
    pred_pat = os.path.join(
        indiv_dir,
        f"rlz{rlz}_TM{inst}_auto_cross_pred_{tracer_addstr}_zmin=0.0_zmax=1.0_tile*.npz",
    )
    pred_fp = _find_single_indiv_file(pred_pat)
    if pred_fp is None:
        return None

    pred = np.load(pred_fp, allow_pickle=True)
    lb = _safe_as_float_array(pred["lb"])
    cross = _safe_as_float_array(pred["clx_comb"])
    cross_err = _safe_as_float_array(pred["clx_err_comb"])
    gal_auto = _safe_as_float_array(pred["clg_comb"])
    gal_auto_err = _safe_as_float_array(pred["clg_err_comb"])
    intensity_auto_tracer = _safe_as_float_array(pred["clI_comb"])
    intensity_auto_tracer_err = _safe_as_float_array(pred["clI_err_comb"])

    if any(
        x is None
        for x in [
            lb,
            cross,
            cross_err,
            gal_auto,
            gal_auto_err,
            intensity_auto_tracer,
            intensity_auto_tracer_err,
        ]
    ):
        return None

    full_specs: List[np.ndarray] = []
    full_errs: List[np.ndarray] = []
    for i in range(len(zbinedges) - 1):
        zmin, zmax = zbinedges[i], zbinedges[i + 1]
        full_pat = os.path.join(
            indiv_dir,
            f"rlz{rlz}_TM{inst}_auto_fullCIBER_zmin={zmin}_zmax={zmax}_tile*.npz",
        )
        full_fp = _find_single_indiv_file(full_pat)
        if full_fp is None:
            continue
        full = np.load(full_fp, allow_pickle=True)
        clI_full = _safe_as_float_array(full["clI_comb"])
        clI_full_err = _safe_as_float_array(full["clI_err_comb"])
        if clI_full is None or clI_full_err is None:
            continue
        full_specs.append(clI_full)
        full_errs.append(clI_full_err)

    if not full_specs:
        return None

    intensity_auto_full = np.nansum(np.array(full_specs), axis=0)
    intensity_auto_full_err = np.sqrt(np.nansum(np.array(full_errs) ** 2, axis=0))

    with np.errstate(divide="ignore", invalid="ignore"):
        rlx = cross / np.sqrt(gal_auto * intensity_auto_full)
        rel_cross = np.where(cross != 0, cross_err / np.abs(cross), np.nan)
        rel_gal = np.where(gal_auto > 0, gal_auto_err / np.abs(gal_auto), np.nan)
        rel_ifull = np.where(
            intensity_auto_full > 0,
            intensity_auto_full_err / np.abs(intensity_auto_full),
            np.nan,
        )
        rlx_err = np.abs(rlx) * np.sqrt(rel_cross**2 + 0.25 * (rel_gal**2 + rel_ifull**2))

    rec = {
        "lb": lb,
        "cross": cross,
        "cross_err": cross_err,
        "gal_auto": gal_auto,
        "gal_auto_err": gal_auto_err,
        "intensity_auto_tracer": intensity_auto_tracer,
        "intensity_auto_tracer_err": intensity_auto_tracer_err,
        "intensity_auto_full": intensity_auto_full,
        "intensity_auto_full_err": intensity_auto_full_err,
        "rlx_tracer_full": rlx,
        "rlx_err_tracer_full": rlx_err,
    }
    if beam_correct:
        rec = _apply_beam_correction_record(rec, inst=inst, ifield=beam_ifield)
    return rec


def _save_pred_cls(outpath: str, dat: Dict[str, np.ndarray]) -> None:
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        outpath,
        lb=dat["lb"],
        cross=dat["cross"],
        cross_err=dat["cross_err"],
        gal_auto=dat["gal_auto"],
        gal_auto_err=dat["gal_auto_err"],
        intensity_auto_tracer=dat["intensity_auto_tracer"],
        intensity_auto_tracer_err=dat["intensity_auto_tracer_err"],
        intensity_auto_full=dat["intensity_auto_full"],
        intensity_auto_full_err=dat["intensity_auto_full_err"],
        rlx_tracer_full=dat["rlx_tracer_full"],
        rlx_err_tracer_full=dat["rlx_err_tracer_full"],
    )


def _copy_v2_pred_if_missing(basepath: str, inst: int, addstr: str) -> None:
    dst = _prediction_path(basepath, inst, addstr)
    if Path(dst).exists():
        return
    src = _prediction_path("data/jordan_mocks/v2/", inst, addstr)
    if Path(src).exists():
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _ensure_single_realization_pred_cls(
    basepath: str,
    rlz: int = 1,
    beam_correct: bool = True,
    beam_ifield: int = 8,
) -> None:
    """Create field_average/pred_cls files from a single indiv realization.

    This bridges 10x10 single-tile mocks into the pred_cls format expected by
    plotting functions, while keeping model curves derived from saved mock data.
    """
    zbinedges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    samples = {
        "sdss_z_lt_22.0_CIBERfidmask": {
            "coarse_head": "sdss_z_lt_22.0_CIBERfidmask",
            "coarse_head_alt": "sdss_z_lt_22.0",
            "zmax_head": "sdss_z_lt_22.0_CIBERfidmask_zmax=1.0",
            "zmax_head_alt": "sdss_z_lt_22.0_zmax=1.0",
        },
        "hsc_i_lt_25.0_CIBERfidmask": {
            "coarse_head": "hsc_i_lt_25.0_CIBERfidmask",
            # plotting code also requests this headstr form for coarse-z paths
            "coarse_head_alt": "hsc_i_lt_25.0",
            "coarse_head_alt2": "hsc_ilt25.0",
            "zmax_head": "hsc_i_lt_25.0_CIBERfidmask_zmax=1.0",
            "zmax_head_alt": "hsc_i_lt_25.0_zmax=1.0",
            "zmax_head_alt2": "hsc_ilt25.0_zmax=1.0",
        },
    }

    for inst in [1, 2]:
        for tracer_addstr, meta in samples.items():
            z_records: List[Dict[str, np.ndarray]] = []
            for i in range(len(zbinedges) - 1):
                zmin, zmax = zbinedges[i], zbinedges[i + 1]
                rec = _load_indiv_pair(
                    basepath=basepath,
                    inst=inst,
                    tracer_addstr=tracer_addstr,
                    zmin=zmin,
                    zmax=zmax,
                    rlz=rlz,
                    beam_correct=beam_correct,
                    beam_ifield=beam_ifield,
                )
                if rec is None:
                    continue
                z_records.append(rec)

                out_add = f"{meta['coarse_head']}_zmin={zmin}_zmax={zmax}"
                _save_pred_cls(_prediction_path(basepath, inst, out_add), rec)

                if "coarse_head_alt" in meta:
                    out_add_alt = f"{meta['coarse_head_alt']}_zmin={zmin}_zmax={zmax}"
                    _save_pred_cls(_prediction_path(basepath, inst, out_add_alt), rec)
                if "coarse_head_alt2" in meta:
                    out_add_alt2 = f"{meta['coarse_head_alt2']}_zmin={zmin}_zmax={zmax}"
                    _save_pred_cls(_prediction_path(basepath, inst, out_add_alt2), rec)

            summed = _load_zlt1_aggregate_record(
                basepath=basepath,
                inst=inst,
                tracer_addstr=tracer_addstr,
                zbinedges=zbinedges,
                rlz=rlz,
                beam_correct=beam_correct,
                beam_ifield=beam_ifield,
            )
            if summed is None:
                summed = _sum_pred_records(z_records)
            if summed is not None:
                _save_pred_cls(_prediction_path(basepath, inst, meta["zmax_head"]), summed)
                if "zmax_head_alt" in meta:
                    _save_pred_cls(_prediction_path(basepath, inst, meta["zmax_head_alt"]), summed)
                if "zmax_head_alt2" in meta:
                    _save_pred_cls(_prediction_path(basepath, inst, meta["zmax_head_alt2"]), summed)

    # WISE 10x10 mocks are not available in this set; keep behavior by fallback.
    for inst in [1, 2]:
        _copy_v2_pred_if_missing(basepath, inst, "wise_W1_lt_20.2_CIBERfidmask")


@contextmanager
def instrument_load_timing() -> Any:
    """Instrument cumulative file-loading time during figure construction.

    We wrap two key loaders used by galaxy-cross plotting pathways:
    - numpy.load
    - ciber.plotting.gal_plotting_fns.load_ciber_gal_ps
    """
    import numpy as _np
    import ciber.plotting.gal_plotting_fns as _gpf

    stats = {"load_sec": 0.0}
    state = {"in_load_ciber": False}

    orig_np_load = _np.load
    orig_load_ciber = _gpf.load_ciber_gal_ps

    def timed_np_load(*args: Any, **kwargs: Any) -> Any:
        t0 = _now()
        out = orig_np_load(*args, **kwargs)
        # Avoid double-counting when np.load is called from load_ciber_gal_ps wrapper.
        if not state["in_load_ciber"]:
            stats["load_sec"] += _now() - t0
        return out

    def timed_load_ciber(*args: Any, **kwargs: Any) -> Any:
        t0 = _now()
        state["in_load_ciber"] = True
        out = orig_load_ciber(*args, **kwargs)
        state["in_load_ciber"] = False
        stats["load_sec"] += _now() - t0
        return out

    with patch("numpy.load", new=timed_np_load), patch(
        "ciber.plotting.gal_plotting_fns.load_ciber_gal_ps", new=timed_load_ciber
    ):
        yield stats


def _validate_files_exist(paths: Sequence[str], label: str) -> None:
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        missing_preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"Missing {label} files ({len(missing)} missing). First entries:\n{missing_preview}"
        )


def configure_matplotlib(show: bool) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt

    if not show:
        plt.ioff()


def _save_figure(
    fig: Any,
    outdir: Path,
    stem: str,
    ext: str,
    overwrite: bool,
    add_timestamp: bool,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    if add_timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{stem}_{ts}"

    outpath = outdir / f"{stem}.{ext}"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if outpath.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {outpath}. Use --overwrite to replace."
        )

    fig.savefig(outpath, bbox_inches="tight")
    return outpath


def _serialize_np_scalar(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def write_diagnostics(
    records: Sequence[FigureTiming],
    outdir: Path,
    basename: str,
) -> Tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / f"{basename}.json"
    csv_path = outdir / f"{basename}.csv"

    payload = [
        {k: _serialize_np_scalar(v) for k, v in asdict(rec).items()} for rec in records
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    fieldnames = [
        "figure_key",
        "output_path",
        "load_sec",
        "plot_sec",
        "save_sec",
        "total_sec",
        "status",
        "error",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload:
            writer.writerow(row)

    return json_path, csv_path


def print_timing_table(records: Sequence[FigureTiming]) -> None:
    if not records:
        return

    print("\nTiming diagnostics:")
    print(
        f"{'figure':28s} {'status':10s} {'load(s)':>10s} {'plot(s)':>10s} "
        f"{'save(s)':>10s} {'total(s)':>10s}"
    )
    print("-" * 88)
    for rec in records:
        print(
            f"{rec.figure_key:28s} {rec.status:10s} "
            f"{rec.load_sec:10.3f} {rec.plot_sec:10.3f} "
            f"{rec.save_sec:10.3f} {rec.total_sec:10.3f}"
        )


# -----------------------------
# Plot wrappers
# -----------------------------


def _import_plotting_functions() -> Dict[str, Any]:
    from matplotlib import pyplot as plt

    from ciber.plotting.gal_plotting_fns import (
        compare_r_ell_hsc_LS_zlt1,
        create_omnibus_plot,
        field_consistency_gal_cross,
        gen_cross_spectrum_plots_vs_z,
        plot_clIG_forecast,
        plot_cross_fit_components_from_file,
        plot_fieldav_ciber_gal_ps,
        plot_gal_ps_vs_redshift,
        plot_perfield_gal_auto,
        plot_rl_vs_z_vs_scale_DESILS,
    )

    return {
        "plt": plt,
        "compare_r_ell_hsc_LS_zlt1": compare_r_ell_hsc_LS_zlt1,
        "create_omnibus_plot": create_omnibus_plot,
        "field_consistency_gal_cross": field_consistency_gal_cross,
        "gen_cross_spectrum_plots_vs_z": gen_cross_spectrum_plots_vs_z,
        "plot_clIG_forecast": plot_clIG_forecast,
        "plot_cross_fit_components_from_file": plot_cross_fit_components_from_file,
        "plot_fieldav_ciber_gal_ps": plot_fieldav_ciber_gal_ps,
        "plot_gal_ps_vs_redshift": plot_gal_ps_vs_redshift,
        "plot_perfield_gal_auto": plot_perfield_gal_auto,
        "plot_rl_vs_z_vs_scale_DESILS": plot_rl_vs_z_vs_scale_DESILS,
    }


def run_omnibus(args: argparse.Namespace) -> List[GeneratedFigure]:
    fns = _import_plotting_functions()
    create_omnibus_plot = fns["create_omnibus_plot"]
    tl_pix_correct = not args.omnibus_no_tl_pix_correct

    common_kwargs = {
        "figsize": tuple(args.omnibus_figsize),
        "tl_pix_correct": tl_pix_correct,
        "ifield_use": args.omnibus_tl_ifield,
        "tl_pix_template": args.omnibus_tl_pix_template,
        "ls_gal_auto_large_fpath": args.ls_gal_auto_large,
    }

    if args.pred_source == "current":
        fig = create_omnibus_plot(include_ciber_auto=False, **common_kwargs) if args.omnibus_include_full else None
        fig_ls_hsc = create_omnibus_plot(include_wise=False, include_ciber_auto=False, **common_kwargs)
    elif args.pred_source == "standard-igl":
        base = _normalize_pred_basepath(args.pred_basepath)
        _ensure_single_realization_pred_cls(
            base,
            beam_correct=not args.pred_no_beam_correct,
            beam_ifield=args.pred_beam_ifield,
        )
        fig = create_omnibus_plot(jmock_basedir=base, include_ciber_auto=False, **common_kwargs) if args.omnibus_include_full else None
        fig_ls_hsc = create_omnibus_plot(
            jmock_basedir=base,
            include_wise=False,
            include_ciber_auto=False,
            **common_kwargs,
        )
    else:
        raise ValueError(
            "parametric mode is not supported for omnibus overlays in this wrapper; "
            "use compare-mock-vs-parametric subcommand for dedicated fit plots."
        )

    out = [
        GeneratedFigure("omnibus:ls-hsc-no-wise-no-ciber-auto", fig_ls_hsc, "ciber_gal_cross_omnibus_ls_hsc_only"),
    ]
    if fig is not None:
        out.insert(0, GeneratedFigure("omnibus", fig, "ciber_gal_auto_cross_omnibus"))
    return out


def _field_consistency_variant_settings(variant: str) -> Tuple[str, str, str]:
    mapping = {
        "wise-cross": ("WISE", "unWISE_W1lt17p5_JHlt16_wFFerr", "cross"),
        "wise-auto": ("WISE", "unWISE_W1lt17p5_JHlt16_wFFerr", "auto"),
        "ls-cross": ("LS", "0.0_z_1.0_wrandsub_JHlt16_wFFerr", "cross"),
        "ls-auto": ("LS", "0.0_z_1.0_wrandsub_JHlt16_wFFerr", "auto"),
    }
    if variant not in mapping:
        raise ValueError(f"Unknown field-consistency variant: {variant}")
    return mapping[variant]


def run_field_consistency_single(args: argparse.Namespace, variant: str) -> List[GeneratedFigure]:
    fns = _import_plotting_functions()
    field_consistency_gal_cross = fns["field_consistency_gal_cross"]

    catname, addstr, ps_type = _field_consistency_variant_settings(variant)
    # Match notebook convention: auto field-consistency shown in sigma units
    # (z-score), i.e. no yerr-style fractional-deviation error bars.
    use_zscore_default = ps_type == "auto"
    try:
        textxpos = 1500 if ps_type == "auto" else 850
        fig, _, _ = field_consistency_gal_cross(
            catname,
            addstr,
            ps_type=ps_type,
            ell_min=args.ell_min,
            ell_max=args.ell_max,
            use_zscore=use_zscore_default,
            textxpos=textxpos,
        )
    except ValueError as exc:
        # Some cross-spectrum fields can carry negative error estimates in the
        # legacy plotting function. Fallback to z-score mode, which avoids
        # yerr usage but preserves the same field-consistency diagnostic intent.
        if ps_type == "cross" and "'yerr' must not contain negative values" in str(exc):
            print(
                f"[field-consistency:{variant}] Falling back to use_zscore=True "
                "due to negative yerr values in cross-spectrum errors."
            )
            fig, _, _ = field_consistency_gal_cross(
                catname,
                addstr,
                ps_type=ps_type,
                ell_min=args.ell_min,
                ell_max=args.ell_max,
                use_zscore=True,
                textxpos=textxpos,
            )
        else:
            raise

    # Cross-case legends can overlap stacked panels; push further right.
    if ps_type == "cross":
        for ax in fig.axes:
            leg = ax.get_legend()
            if leg is not None:
                leg.set_bbox_to_anchor((1.68, 0.12))
                leg._loc = 4
    else:
        for ax in fig.axes:
            leg = ax.get_legend()
            if leg is not None:
                leg.set_bbox_to_anchor((1.50, 0.12))
                leg._loc = 4

    stem = f"field_consistency_{variant.replace('-', '_')}"
    return [GeneratedFigure(f"field-consistency:{variant}", fig, stem)]


def run_forecast(args: argparse.Namespace) -> List[GeneratedFigure]:
    fns = _import_plotting_functions()
    plot_clIG_forecast = fns["plot_clIG_forecast"]

    inpath = Path(args.input_npz).expanduser()
    dat = np.load(inpath, allow_pickle=True)
    required = ["lb", "lrange", "dcl_terms_bp", "dcl_vs_nbar", "xerr"]
    for key in required:
        if key not in dat:
            raise KeyError(f"Missing key '{key}' in {inpath}")

    fig = plot_clIG_forecast(
        lb=dat["lb"],
        lrange=dat["lrange"],
        dcl_terms_bp=dat["dcl_terms_bp"],
        dcl_vs_nbar=dat["dcl_vs_nbar"],
        xerr=dat["xerr"],
    )
    return [GeneratedFigure("forecast", fig, "plot_clIG_forecast")]


def _load_rl_vs_z_inputs(inpath: Path) -> Tuple[Dict[str, Any], Any]:
    dat = np.load(inpath, allow_pickle=True)

    if "res_meas" in dat:
        res_meas_obj = dat["res_meas"]
        if isinstance(res_meas_obj, np.ndarray) and res_meas_obj.dtype == object:
            res_meas = res_meas_obj.item()
        else:
            res_meas = res_meas_obj
    else:
        req = [
            "zcen",
            "zbinedges",
            "lb_mins",
            "lb_maxs",
            "mean_rl_diffscale",
            "std_rl_diffscale",
        ]
        for key in req:
            if key not in dat:
                raise KeyError(
                    f"Missing key '{key}' in rl-vs-z input. Provide either res_meas or all required keys."
                )
        res_meas = {k: dat[k] for k in req}

    if "mean_rl_diffscale_pred" not in dat:
        raise KeyError("Missing key 'mean_rl_diffscale_pred' in rl-vs-z input file")

    return res_meas, dat["mean_rl_diffscale_pred"]


def run_rl_vs_z_scale(args: argparse.Namespace) -> List[GeneratedFigure]:
    fns = _import_plotting_functions()
    plot_rl_vs_z_vs_scale_DESILS = fns["plot_rl_vs_z_vs_scale_DESILS"]

    inpath = Path(args.input_npz).expanduser()
    res_meas, mean_rl_diffscale_pred = _load_rl_vs_z_inputs(inpath)

    fig = plot_rl_vs_z_vs_scale_DESILS(
        res_meas=res_meas,
        mean_rl_diffscale_pred=mean_rl_diffscale_pred,
    )
    return [GeneratedFigure("rl-vs-z-scale", fig, "plot_rl_vs_z_vs_scale_DESILS")]


def fit_gaia_cross_poisson_damping(
    inst: int,
    addstr: str = "stars_glt20p5_JHlt14_wFFerr",
    fit_save_dir: str = "data/gaia_cross_fits",
    mask_frac: float = 0.7,
    ifield_list: List[int] = None,
    ifield_use: int = 8,
    startidx: int = 2,
    endidx: int = -1,
    nwalkers: int = 32,
    nsteps: int = 2000,
    nburn: int = 500,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Fit CIBER x Gaia cross spectrum with Poisson + astrometry damping model.

    Model: D_ell = A_shot * ell*(ell+1)/(2*pi) * exp(-0.5 * (sigma_damp_rad * ell)^2)
    
    Fitted parameters: [A_shot, sigma_damp_arcsec]

    Results are saved to ``fit_save_dir/gaia_cross_fit_TM{inst}.npz``.

    Parameters
    ----------
    inst : int
        CIBER instrument (1=J, 2=H).
    addstr : str
        Suffix identifying the Gaia catalog file.
    fit_save_dir : str
        Directory to write the NPZ fit result.
    mask_frac : float
        Assumed unmasked sky fraction per field.
    ifield_list : list of int, optional
        Fields to include. Defaults to [4, 5, 6, 7, 8].
    ifield_use : int
        Reference field for pixel transfer function correction.
    startidx, endidx : int
        Multipole index range slice applied before fitting.
    nwalkers, nsteps, nburn : int
        MCMC configuration.
    verbose : bool
        Print fit diagnostics.

    Returns
    -------
    dict
        Result dict with keys ``lb``, ``fieldav_cl_cross``, ``fieldav_clerr_cross``,
        ``samples``, ``params``, ``params_16``, ``params_84``, ``param_names_fitted``.
    """
    import emcee
    import ciber.plotting.gal_plotting_fns as gpf
    from ciber.core.powerspec_pipeline import CIBER_PS_pipeline
    from ciber.plotting.gal_plotting_fns import (
        load_ciber_gal_ps,
        compute_weighted_cl,
        estimate_cross_uncertainties,
    )

    if ifield_list is None:
        ifield_list = [4, 5, 6, 7, 8]

    cbps = CIBER_PS_pipeline()
    bandstr_list = ["J", "H"]

    cgps_file = load_ciber_gal_ps(inst, "gaia", addstr=addstr)
    lb = cgps_file["lb"]
    all_cl_gal = cgps_file["all_cl_gal"]
    all_cl_cross = cgps_file["all_cl_cross"]
    all_clerr_cross = cgps_file["all_clerr_cross"]
    ifield_list_use = cgps_file["ifield_list_use"]
    nfield = len(ifield_list_use)

    ciber_auto = gpf._load_ciber_auto_file(bandstr_list[inst - 1])
    lb_auto, cl_auto = ciber_auto["lb"], ciber_auto["fieldav_cl"]

    cl_weights = 1.0 / all_clerr_cross ** 2
    fieldav_cl_cross, fieldav_clerr_cross = compute_weighted_cl(
        all_cl_cross.copy(), cl_weights
    )
    fieldav_cl_gal = np.mean(all_cl_gal, axis=0)

    cross_knox = np.sqrt(1.0 / ((2 * lb + 1) * cbps.Mkk_obj.delta_ell))
    fsky = mask_frac * nfield * 2 * 2 / 41253.0
    cross_knox /= np.sqrt(fsky)
    cross_knox *= np.abs(fieldav_cl_cross)
    fieldav_clerr_cross = np.sqrt(cross_knox ** 2 + fieldav_clerr_cross ** 2)
    fieldav_clerr_cross = estimate_cross_uncertainties(
        lb, fieldav_cl_cross, fieldav_clerr_cross,
        cl_auto, fieldav_cl_gal, nfield, startidx=2, endidx=-1,
    )

    tl_pix = np.load(
        f"data/fluctuation_data/transfer_function/tl_clx_pix_TM{inst}_ifield{ifield_use}.npz"
    )["tl_clx_pix"]
    fieldav_cl_cross = fieldav_cl_cross / tl_pix
    fieldav_clerr_cross = fieldav_clerr_cross / tl_pix

    pf = lb * (lb + 1) / (2 * np.pi)
    dl_data = pf * fieldav_cl_cross
    dl_err = pf * fieldav_clerr_cross

    # Select fit range
    lbmask = np.ones(len(lb), dtype=bool)
    lbmask[:startidx] = False
    if endidx != -1:
        lbmask[endidx:] = False
    
    lb_fit = lb[lbmask]
    dl_fit = dl_data[lbmask]
    dl_err_fit = dl_err[lbmask]

    # Only fit positive-signal bins
    pos_mask = dl_fit > 0
    lb_fit = lb_fit[pos_mask]
    dl_fit = dl_fit[pos_mask]
    dl_err_fit = dl_err_fit[pos_mask]

    pf_fit = lb_fit * (lb_fit + 1) / (2 * np.pi)
    arcsec_to_rad = (1.0 / 3600.0) * (np.pi / 180.0)

    # MCMC setup: 2 parameters [A_shot, sigma_damp_arcsec]
    A_shot_lo, A_shot_hi = 0.0, 1e-3
    sig_lo, sig_hi = 0.1, 20.0  # arcsec

    def _log_prior(p):
        A, sig = p
        if A_shot_lo <= A <= A_shot_hi and sig_lo <= sig <= sig_hi:
            return 0.0
        return -np.inf

    def _log_likelihood(p):
        A, sig = p
        sig_r = sig * arcsec_to_rad
        model = A * pf_fit * np.exp(-0.5 * (sig_r * lb_fit) ** 2)
        return -0.5 * np.sum(((dl_fit - model) / dl_err_fit) ** 2)

    def _log_prob(p):
        lp = _log_prior(p)
        return lp + _log_likelihood(p) if np.isfinite(lp) else -np.inf

    # Initial guess: A_shot from high-ell mean, sigma_damp = 2 arcsec
    shot_mask = lb_fit >= 0.5 * lb_fit.max()
    A_shot_init = float(np.nanmean(dl_fit[shot_mask] / pf_fit[shot_mask])) if np.any(shot_mask) else 1e-5
    A_shot_init = np.clip(A_shot_init, A_shot_lo + 1e-8, A_shot_hi - 1e-8)
    p0_center = np.array([A_shot_init, 2.0])
    p0 = p0_center + np.array([1e-6, 0.5]) * np.random.randn(nwalkers, 2)
    p0[:, 0] = np.clip(p0[:, 0], A_shot_lo + 1e-10, A_shot_hi - 1e-10)
    p0[:, 1] = np.clip(p0[:, 1], sig_lo + 1e-4, sig_hi - 1e-4)

    if verbose:
        print(f"Running Gaia cross MCMC (TM{inst}): [A_shot, sigma_damp], "
              f"{nwalkers} walkers x {nsteps} steps...")

    sampler = emcee.EnsembleSampler(nwalkers, 2, _log_prob)
    sampler.run_mcmc(p0, nsteps, progress=verbose)
    samples = sampler.get_chain(discard=nburn, flat=True)  # (N, 2)

    params_med = np.median(samples, axis=0)
    params_16 = np.percentile(samples, 16, axis=0)
    params_84 = np.percentile(samples, 84, axis=0)

    if verbose:
        print(f"  A_shot   = {params_med[0]:.3e} [{params_16[0]:.3e}, {params_84[0]:.3e}]")
        print(f"  σ_damp   = {params_med[1]:.2f} [{params_16[1]:.2f}, {params_84[1]:.2f}] arcsec")

    result = {
        "lb": lb,
        "fieldav_cl_cross": fieldav_cl_cross,
        "fieldav_clerr_cross": fieldav_clerr_cross,
        "samples": samples,
        "params": params_med,
        "params_16": params_16,
        "params_84": params_84,
        "param_names_fitted": ["A_shot", "sigma_damp_arcsec"],
    }

    os.makedirs(fit_save_dir, exist_ok=True)
    save_path = os.path.join(fit_save_dir, f"gaia_cross_fit_TM{inst}.npz")
    np.savez(
        save_path,
        lb=lb,
        fieldav_cl_cross=fieldav_cl_cross,
        fieldav_clerr_cross=fieldav_clerr_cross,
        samples=result["samples"],
        params=result["params"],
        params_16=result["params_16"],
        params_84=result["params_84"],
        param_names_fitted=result["param_names_fitted"],
    )
    if verbose:
        print(f"Saved Gaia cross fit (TM{inst}) to {save_path}")

    return result


def _load_gaia_cross_fit(inst: int, fit_save_dir: str = "data/gaia_cross_fits") -> Optional[Dict[str, Any]]:
    """Load a previously saved Gaia cross fit NPZ, or return None if not found."""
    path = os.path.join(fit_save_dir, f"gaia_cross_fit_TM{inst}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def run_gaia_auto(args: argparse.Namespace) -> List[GeneratedFigure]:
    fns = _import_plotting_functions()
    plt = fns["plt"]
    plot_perfield_gal_auto = fns["plot_perfield_gal_auto"]
    import ciber.plotting.gal_plotting_fns as gpf

    fig, _, _, _ = plot_perfield_gal_auto(
        catname="gaia",
        inst=1,
        addstr="stars_glt20p5_JHlt14_wFFerr",
        ylabel="$D_{\\ell}$",
        textxpos=300,
        textypos=50,
        textstr="Gaia star auto\n($G<20.5$, $J>14.0$)",
        ylim=[1e-4, 1e3],
        xlim=[250, 1.1e5]
    )

    # Post-process to match notebook styling:
    # - make per-field points less dominant
    # - add an explicitly labeled, prominent field-average trace
    # - add best-fit high-ell shot-noise line from field-average C_ell
    cgps = gpf.load_ciber_gal_ps(1, "gaia", addstr="stars_glt20p5_JHlt14_wFFerr")
    lb = cgps["lb"]
    all_cl_gal = cgps["all_cl_gal"]
    fieldav_cl = np.mean(all_cl_gal, axis=0)
    fieldav_cl_std = np.std(all_cl_gal, axis=0)
    pf = lb * (lb + 1) / (2 * np.pi)

    shot_mask = (lb >= 5.0e4) & (lb <= 8.0e4)
    if np.any(shot_mask):
        cl_shot_best = float(np.nanmean(fieldav_cl[shot_mask]))
    else:
        cl_shot_best = float(np.nanmean(fieldav_cl[-3:]))

    ax = fig.axes[0]
    for line in ax.lines:
        if line.get_color() != "k":
            line.set_alpha(0.45)

    ax.errorbar(
        lb,
        pf * fieldav_cl,
        yerr=pf * fieldav_cl_std,
        fmt="o",
        color="k",
        markersize=3,
        capsize=3,
        linewidth=1.2,
        zorder=20,
        label="Field average",
    )
    ax.plot(
        lb,
        pf * cl_shot_best,
        linestyle="--",
        linewidth=1.0,
        color="k",
        alpha=0.9,
        label=f"Best-fit shot noise",
    )
    ax.set_ylim([1e-4, 1e3])
    ax.legend(loc=4, fontsize=10, ncol=2)

    return [GeneratedFigure("gaia-auto", fig, "gaia_star_glt20p5_auto")]


def run_gaia_cross(args: argparse.Namespace) -> List[GeneratedFigure]:
    import matplotlib.pyplot as plt

    fns = _import_plotting_functions()
    plot_fieldav_ciber_gal_ps = fns["plot_fieldav_ciber_gal_ps"]

    rerun_fit = getattr(args, "rerun_fit", False)
    fit_save_dir = "data/gaia_cross_fits"
    addstr = "stars_glt20p5_JHlt14_wFFerr"
    inst_list = [1, 2]
    colors = ["b", "r"]

    # Load or rerun fits for each instrument
    fits: Dict[int, Dict[str, Any]] = {}
    for inst in inst_list:
        cached = None if rerun_fit else _load_gaia_cross_fit(inst, fit_save_dir)
        if cached is None:
            print(f"Running Gaia cross fit for TM{inst}...")
            cached = fit_gaia_cross_poisson_damping(
                inst, addstr=addstr, fit_save_dir=fit_save_dir
            )
        fits[inst] = cached

    fig, lb, _, _ = plot_fieldav_ciber_gal_ps(
        inst_list=inst_list,
        catname="gaia",
        addstr=addstr,
        textstr="CIBER $\\times$ Gaia stars\n($G<20.5$, $J>14.0$)",
        textxpos=350,
        labels=["Data", None],  # Only show "Data" label for first instrument
    )

    ax = fig.axes[0]
    ell_plot = np.geomspace(lb[2], lb[-2], 300)
    pf_plot = ell_plot * (ell_plot + 1) / (2 * np.pi)

    damping_values = []  # Store damping values for text annotation
    
    for idx, inst in enumerate(inst_list):
        color = colors[idx]
        fit = fits[inst]
        samples = fit["samples"]  # shape (N_samples, 2): [A_shot, sigma_damp_arcsec]
        params = fit["params"]    # [A_shot_median, sigma_damp_median]
        A_shot_med = float(params[0])
        sigma_damp_med = float(params[1])

        sigma_rad_med = sigma_damp_med * (1.0 / 3600.0) * (np.pi / 180.0)

        # Best-fit damped curve
        dl_damped = A_shot_med * pf_plot * np.exp(-0.5 * (sigma_rad_med * ell_plot) ** 2)
        # Undamped Poisson
        dl_poisson = A_shot_med * pf_plot

        # 1-sigma band from posterior samples (vectorized: N_samples x N_ell)
        arcsec_to_rad = (1.0 / 3600.0) * (np.pi / 180.0)
        A_samp = samples[:, 0][:, np.newaxis]          # (N, 1)
        sig_r_samp = samples[:, 1][:, np.newaxis] * arcsec_to_rad  # (N, 1)
        dl_band = A_samp * pf_plot * np.exp(-0.5 * (sig_r_samp * ell_plot) ** 2)
        dl_lo = np.percentile(dl_band, 16, axis=0)
        dl_hi = np.percentile(dl_band, 84, axis=0)

        lam_str = {1: "1.1", 2: "1.8"}[inst]
    
        # Plot Poisson level (dashed line)
        label_poisson = "Best-fit shot noise" if idx == 0 else None
        ax.plot(ell_plot, dl_poisson, color=color, linewidth=1.0,
                linestyle="--", zorder=4, alpha=0.7, label=label_poisson)

        # Plot with damping (solid line)
        label_damp = "With damping" if idx == 0 else None
        ax.plot(ell_plot, dl_damped, color=color, linewidth=1.2, zorder=5,
                label=label_damp, alpha=0.7)
        ax.fill_between(ell_plot, dl_lo, dl_hi, color=color, alpha=0.2, zorder=4)
        

        # Store damping values with color info for bottom-right annotation
        damping_values.append((sigma_damp_med, lam_str, color))

    # Add data label - plot invisible line for legend
    # ax.plot([], [], 'k-', linewidth=1.5, label="Data (1.1 $\\mu$m)")

    # Create legend
    ax.legend(loc=2, bbox_to_anchor=[-0.2, 1.15], fontsize=11, ncol=3)
    
    # Add damping values in bottom right with colored text
    y_pos = 0.2
    for sigma_damp, lam_str, color in damping_values:
        text_str = f"$\\sigma_{{\\rm damp}}$ = {sigma_damp:.1f}'' ({lam_str} $\\mu$m)"
        ax.text(0.95, y_pos, text_str, transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', horizontalalignment='right',
                color=color)
        y_pos -= 0.1
    
    ax.set_ylim([1e-3, 1e3])
    ax.set_xlim([280, 1.1e5])

    return [GeneratedFigure("gaia-cross", fig, "ciber_gaia_star_glt20p5_cross")]


def run_compare_r_ell(args: argparse.Namespace) -> List[GeneratedFigure]:
    fns = _import_plotting_functions()
    compare_r_ell_hsc_LS_zlt1 = fns["compare_r_ell_hsc_LS_zlt1"]

    kwargs: Dict[str, Any] = {
        # Match notebook-like placement: up/left enough to clear top panel content.
        "bbox_to_anchor": [-0.05, 1.42],
    }

    if args.add_trilegal_isl:
        kwargs.update(
            {
                "plot_isl_adjusted": True,
                "isl_use_trilegal": True,
                "isl_trilegal_datestr": args.isl_trilegal_datestr,
                "isl_trilegal_maglim_vega": args.isl_maglim_vega,
                "isl_trilegal_stat": args.isl_trilegal_stat,
                "isl_label": "IGL + unresolved ISL (TRILEGAL)",
            }
        )
        if args.isl_trilegal_basepath is not None:
            kwargs["isl_trilegal_basepath"] = args.isl_trilegal_basepath

    if args.pred_source == "standard-igl":
        base = _normalize_pred_basepath(args.pred_basepath)
        _ensure_single_realization_pred_cls(
            base,
            beam_correct=not args.pred_no_beam_correct,
            beam_ifield=args.pred_beam_ifield,
        )

        ls_add = "sdss_z_lt_22.0_CIBERfidmask_zmax=1.0"
        hsc_add = "hsc_i_lt_25.0_CIBERfidmask_zmax=1.0"
        wise_add = "wise_W1_lt_20.2_CIBERfidmask"

        ls_pred = [_prediction_path(base, inst, ls_add) for inst in [1, 2]]
        hsc_pred = [_prediction_path(base, inst, hsc_add) for inst in [1, 2]]
        wise_pred = [_prediction_path(base, inst, wise_add) for inst in [1, 2]]
        _validate_files_exist(ls_pred + hsc_pred + wise_pred, "standard-igl compare-r-ell")

        kwargs.update(
            {
                "ls_pred_fpaths": ls_pred,
                "hsc_pred_fpaths": hsc_pred,
                "wise_pred_fpaths": wise_pred,
            }
        )
    elif args.pred_source == "parametric":
        raise ValueError(
            "parametric mode is not supported in compare-r-ell wrapper; "
            "use compare-mock-vs-parametric for dedicated fit/component figures."
        )

    fig, _, _ = compare_r_ell_hsc_LS_zlt1(**kwargs)

    kwargs_no_wise = dict(kwargs)
    kwargs_no_wise["include_wise"] = False
    kwargs_no_wise["wise_pred_fpaths"] = None
    fig_no_wise, _, _ = compare_r_ell_hsc_LS_zlt1(**kwargs_no_wise)

    return [
        GeneratedFigure("compare-r-ell", fig, "compare_r_ell_hsc_LS_zlt1"),
        GeneratedFigure("compare-r-ell:no-wise", fig_no_wise, "compare_r_ell_hsc_LS_zlt1_no_wise"),
    ]


def run_cross_redshift(args: argparse.Namespace) -> List[GeneratedFigure]:
    """Generate cross-spectrum redshift-bin figures used in the paper flow.

    Produces:
    - DESI-LS fine bins (dz=0.1)
    - LS+HSC coarse bins (dz=0.2)
    """
    fns = _import_plotting_functions()
    gen_cross_spectrum_plots_vs_z = fns["gen_cross_spectrum_plots_vs_z"]

    if args.pred_source == "standard-igl":
        base = _normalize_pred_basepath(args.pred_basepath)
        _ensure_single_realization_pred_cls(
            base,
            beam_correct=not args.pred_no_beam_correct,
            beam_ifield=args.pred_beam_ifield,
        )
    elif args.pred_source == "parametric":
        raise ValueError(
            "parametric mode is not supported for cross-redshift wrapper; use compare-mock-vs-parametric"
        )

    bias_cache = str(getattr(args, "bias_cache_fpath", "") or "").strip() or None

    fig_fine, fig_coarse = gen_cross_spectrum_plots_vs_z(
        plot_fine=True,
        plot_coarse=True,
        bias_cache_fpath=bias_cache,
    )

    out: List[GeneratedFigure] = []
    if fig_fine is not None:
        out.append(
            GeneratedFigure(
                "cross-redshift:desils-dz0.1",
                fig_fine,
                "cross_ps_desils_dz0p1",
            )
        )
    if fig_coarse is not None:
        out.append(
            GeneratedFigure(
                "cross-redshift:ls-hsc-dz0.2",
                fig_coarse,
                "cross_ps_ls_hsc_dz0p2",
            )
        )

    return out


def _zlt1_coarse_bins() -> List[float]:
    return [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _extract_measured_ciber_auto_dl(
    res_ps: Dict[str, Any],
    inst_idx: int,
    startidx: int,
    endidx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lb = np.asarray(res_ps["lb"])[startidx:endidx]
    pf = lb * (lb + 1.0) / (2.0 * np.pi)
    cl = np.asarray(res_ps["full_cl_ciber_auto"])[inst_idx][startidx:endidx]
    clerr = np.asarray(res_ps["full_clerr_ciber_auto"])[inst_idx][startidx:endidx]
    return lb, pf * cl, pf * clerr


def _load_measured_ciber_auto_dl_exact(inst_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from ciber.plotting.gal_plotting_fns import _load_ciber_auto_file

    band = "J" if inst_idx == 0 else "H"
    dat = _load_ciber_auto_file(band)
    lb = np.asarray(dat["lb"], dtype=float)
    pf = lb * (lb + 1.0) / (2.0 * np.pi)

    if "fieldav_dl" in dat:
        dl = np.asarray(dat["fieldav_dl"], dtype=float)
        dlerr = np.asarray(dat["fieldav_dlerr"], dtype=float)
        mode = "native_dl"
    else:
        cl = np.asarray(dat["fieldav_cl"], dtype=float)
        clerr = np.asarray(dat["fieldav_clerr"], dtype=float)
        dl = pf * cl
        dlerr = pf * clerr
        mode = "cl_to_dl"

    print(
        f"[Measured CIBER auto shown] TM{inst_idx+1} mode={mode} "
        f"source={dat.get('source_path', 'unknown')} lb=[{lb[0]:.1f}, {lb[-1]:.1f}] n={len(lb)} "
        f"D_ell=[{np.nanmin(dl):.3e}, {np.nanmax(dl):.3e}] "
        f"dD_ell=[{np.nanmin(dlerr):.3e}, {np.nanmax(dlerr):.3e}]"
    )

    return lb, dl, dlerr


def _expand_param_samples_to_full(
    samples: np.ndarray,
    full_params: np.ndarray,
) -> np.ndarray:
    """Expand fitted-parameter samples to the full model parameter vector shape."""
    s = np.asarray(samples, dtype=float)
    p = np.asarray(full_params, dtype=float)
    if s.ndim != 2:
        raise ValueError(f"Expected 2D samples array, got shape {s.shape}")
    if s.shape[1] == p.size:
        return s

    # Fixed-shape one-halo with damping: fitted [A_2h, A_1h, A_shot, sigma_damp]
    # full [A_2h, A_1h, mu_1h, sigma_1h, A_shot, sigma_damp]
    if p.size == 6 and s.shape[1] == 4:
        out = np.zeros((s.shape[0], 6), dtype=float)
        out[:, 0] = s[:, 0]
        out[:, 1] = s[:, 1]
        out[:, 2] = p[2]
        out[:, 3] = p[3]
        out[:, 4] = s[:, 2]
        out[:, 5] = s[:, 3]
        return out

    # Fixed-shape one-halo without damping: fitted [A_2h, A_1h, A_shot]
    # full [A_2h, A_1h, mu_1h, sigma_1h, A_shot]
    if p.size == 5 and s.shape[1] == 3:
        out = np.zeros((s.shape[0], 5), dtype=float)
        out[:, 0] = s[:, 0]
        out[:, 1] = s[:, 1]
        out[:, 2] = p[2]
        out[:, 3] = p[3]
        out[:, 4] = s[:, 2]
        return out

    raise ValueError(
        "Unsupported fitted-sample dimensionality for expansion: "
        f"samples_dim={s.shape[1]}, full_dim={p.size}"
    )


def _extract_fit_samples_cell(
    fit_res: Dict[str, Any],
    inst_idx: int,
    z_idx: int,
) -> Optional[np.ndarray]:
    if "samples" not in fit_res:
        return None
    samples_obj = fit_res["samples"]
    if not isinstance(samples_obj, np.ndarray) or samples_obj.dtype != object:
        return None
    if samples_obj.ndim != 2:
        return None
    if inst_idx >= samples_obj.shape[0] or z_idx >= samples_obj.shape[1]:
        return None
    cell = samples_obj[inst_idx, z_idx]
    if cell is None:
        return None
    cell_arr = np.asarray(cell, dtype=float)
    if cell_arr.ndim != 2 or cell_arr.shape[0] < 2:
        return None
    return cell_arr


def _column_percentile(values: np.ndarray, q: float) -> np.ndarray:
    out = np.full(values.shape[1], np.nan, dtype=float)
    for j in range(values.shape[1]):
        col = values[:, j]
        m = np.isfinite(col)
        if np.any(m):
            out[j] = float(np.percentile(col[m], q))
    return out


def _estimate_ciber_shot_noise_dl(
    lb: np.ndarray,
    dl: np.ndarray,
    ell_min: float,
    ell_max: float,
    lb_eval: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate a flat shot-noise C_ell from high ell and return D_ell curve."""
    lb = np.asarray(lb, dtype=float)
    dl = np.asarray(dl, dtype=float)
    lb_use = lb if lb_eval is None else np.asarray(lb_eval, dtype=float)
    pf = lb * (lb + 1.0) / (2.0 * np.pi)
    pf_use = lb_use * (lb_use + 1.0) / (2.0 * np.pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        cl = np.where(pf > 0.0, dl / pf, np.nan)

    m = (
        (lb >= float(ell_min))
        & (lb <= float(ell_max))
        & np.isfinite(cl)
    )
    if not np.any(m):
        m = np.isfinite(cl)
    cl_shot = float(np.nanmean(cl[m])) if np.any(m) else np.nan
    return pf_use * cl_shot


def _try_load_ciber_dgl_auto_constraints(
    lb: np.ndarray,
    dgl_mode: str = "sfd_clean",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load per-band DGL auto constraints regridded to requested ell bins."""
    from ciber.io.ciber_data_utils import load_dglpred_regrid

    lb = np.asarray(lb, dtype=float)
    dgl_dl = np.full((2, lb.size), np.nan, dtype=float)
    dgl_derr = np.full((2, lb.size), np.nan, dtype=float)

    try:
        import config as ciber_config
        external_base = Path(str(ciber_config.ciber_basepath)) / "data" / "fluctuation_data"
    except Exception:
        external_base = None

    for inst_idx, inst in enumerate([1, 2]):
        patterns = [
            str(
                REPO_ROOT
                / "data"
                / "fluctuation_data"
                / f"TM{inst}"
                / "dgl_tracer_maps"
                / dgl_mode
                / f"dgl_auto_constraints_TM{inst}_{dgl_mode}_*.npz"
            )
        ]
        if external_base is not None:
            patterns.append(
                str(
                    external_base
                    / f"TM{inst}"
                    / "dgl_tracer_maps"
                    / dgl_mode
                    / f"dgl_auto_constraints_TM{inst}_{dgl_mode}_*.npz"
                )
            )

        matches: List[str] = []
        for pat in patterns:
            matches.extend(glob.glob(pat))
        matches = sorted(set(matches))
        if not matches:
            return None, None

        dgl_path = matches[-1]
        dl_pred, dl_err = load_dglpred_regrid(dgl_path, lb)
        dgl_dl[inst_idx] = np.asarray(dl_pred, dtype=float)
        dgl_derr[inst_idx] = np.asarray(dl_err, dtype=float)

    return dgl_dl, dgl_derr


def _load_optional_mock_igl_overlay(
    fpath: str,
    lb_target: np.ndarray,
) -> np.ndarray:
    """Load optional mock IGL overlay and interpolate to lb_target.

    Supported payloads:
    - keys: lb + dl_tm1 + dl_tm2
    - keys: lb + dl_igl_tm1 + dl_igl_tm2
    - keys: lb + dl_igl_zlt1_tm1 + dl_igl_zlt1_tm2
    - key : dl_igl_zlt1 with shape (2, n_ell)
    """
    dat = np.load(str(fpath), allow_pickle=True)
    lb_src = np.asarray(dat["lb"], dtype=float)

    key_pairs = [
        ("dl_tm1", "dl_tm2"),
        ("dl_igl_tm1", "dl_igl_tm2"),
        ("dl_igl_zlt1_tm1", "dl_igl_zlt1_tm2"),
    ]

    tm1 = None
    tm2 = None
    for k1, k2 in key_pairs:
        if k1 in dat and k2 in dat:
            tm1 = np.asarray(dat[k1], dtype=float)
            tm2 = np.asarray(dat[k2], dtype=float)
            break

    if tm1 is None or tm2 is None:
        if "dl_igl_zlt1" in dat:
            arr = np.asarray(dat["dl_igl_zlt1"], dtype=float)
            if arr.ndim == 2 and arr.shape[0] == 2:
                tm1 = arr[0]
                tm2 = arr[1]

    if tm1 is None or tm2 is None:
        raise ValueError(
            "Unsupported mock IGL file format. Expected one of: "
            "(lb, dl_tm1, dl_tm2), (lb, dl_igl_tm1, dl_igl_tm2), "
            "(lb, dl_igl_zlt1_tm1, dl_igl_zlt1_tm2), or dl_igl_zlt1[2, n_ell]."
        )

    out = np.full((2, np.size(lb_target)), np.nan, dtype=float)
    out[0] = np.interp(lb_target, lb_src, tm1, left=np.nan, right=np.nan)
    out[1] = np.interp(lb_target, lb_src, tm2, left=np.nan, right=np.nan)
    return out


def _try_load_default_mock_igl_zlt1(lb_target: np.ndarray) -> Optional[np.ndarray]:
    """Load default z<1 mock IGL auto from Jordan mock field-average files.

    Uses TM1/TM2 files for HSC i<24 and z=[0,1]. These files store C_ell; convert
    to D_ell for overlay.
    """
    try:
        import config as ciber_config
        external_base = Path(str(ciber_config.ciber_basepath)) / "data"
    except Exception:
        external_base = None

    roots = [REPO_ROOT / "data"]
    if external_base is not None:
        roots.append(external_base)

    tm_paths: List[str] = []
    for inst in [1, 2]:
        found = None
        for root in roots:
            p = (
                root
                / "jordan_mocks"
                / "v2"
                / "mock_ps_pred"
                / f"TM{inst}"
                / "field_average"
                / f"pred_cls_TM{inst}_hsc_i_lt_24.0_zmin=0.0_zmax=1.0.npz"
            )
            if p.exists():
                found = str(p)
                break
        if found is None:
            return None
        tm_paths.append(found)

    lb_target = np.asarray(lb_target, dtype=float)
    pf_target = lb_target * (lb_target + 1.0) / (2.0 * np.pi)
    out = np.full((2, lb_target.size), np.nan, dtype=float)

    for inst_idx, p in enumerate(tm_paths):
        d = np.load(p)
        if "lb" not in d:
            return None
        lb_src = np.asarray(d["lb"], dtype=float)
        if "intensity_auto_tracer" in d:
            cl_src = np.asarray(d["intensity_auto_tracer"], dtype=float)
        elif "intensity_auto_full" in d:
            cl_src = np.asarray(d["intensity_auto_full"], dtype=float)
        else:
            return None
        pf_src = lb_src * (lb_src + 1.0) / (2.0 * np.pi)
        dl_src = pf_src * cl_src
        dl_interp = np.interp(lb_target, lb_src, dl_src, left=np.nan, right=np.nan)
        out[inst_idx] = np.where(np.isfinite(dl_interp), dl_interp, np.nan)

    _ = pf_target  # retained for clarity on D_ell convention
    return out


def _predict_auto_2h1h_from_cross_and_shot_subtracted_auto(
    lb_eval: np.ndarray,
    cross_params: np.ndarray,
    cross_params_err: np.ndarray,
    cross_samples: Optional[np.ndarray],
    gal_lb: np.ndarray,
    gal_cl: np.ndarray,
    gal_cl_err: np.ndarray,
    ell_range_for_scaling: Tuple[float, float] = (300.0, 3000.0),
    shot_ell_min: float = 5.0e4,
    shot_ell_max: float = 8.0e4,
    gal_auto_denominator_mode: str = "shot-subtracted-data",
    ratio_scaling_mode: str = "direct-ratio",
    gal_auto_params: Optional[np.ndarray] = None,
    gal_auto_params_err: Optional[np.ndarray] = None,
    gal_auto_samples: Optional[np.ndarray] = None,
    nsamp: int = 2000,
    seed: int = 123,
) -> Dict[str, np.ndarray]:
    from ciber.theory.cross_ps_parametric_model import CrossPowerSpectrumModel

    lb_eval = np.asarray(lb_eval, dtype=float)
    cross_params = np.asarray(cross_params, dtype=float)
    cross_params_err = np.asarray(cross_params_err, dtype=float)
    gal_lb = np.asarray(gal_lb, dtype=float)
    gal_cl = np.asarray(gal_cl, dtype=float)
    gal_cl_err = np.asarray(gal_cl_err, dtype=float)
    rng = np.random.default_rng(seed)

    if gal_auto_denominator_mode not in {"shot-subtracted-data", "smooth-model"}:
        raise ValueError(
            "gal_auto_denominator_mode must be one of ['shot-subtracted-data', 'smooth-model'], "
            f"got: {gal_auto_denominator_mode}"
        )
    if ratio_scaling_mode not in {"direct-ratio", "separate-2h-1h"}:
        raise ValueError(
            "ratio_scaling_mode must be one of ['direct-ratio', 'separate-2h-1h'], "
            f"got: {ratio_scaling_mode}"
        )

    # Build posterior draws for cross fit (fallback to Gaussian approx if samples unavailable).
    use_posterior_samples = False
    if cross_samples is not None and np.asarray(cross_samples).ndim == 2:
        s_raw = np.asarray(cross_samples, dtype=float)
        if s_raw.shape[0] >= 2:
            if s_raw.shape[1] != cross_params.size:
                s_full = _expand_param_samples_to_full(s_raw, cross_params)
            else:
                s_full = s_raw
            take = min(int(nsamp), s_full.shape[0])
            idx = rng.choice(s_full.shape[0], size=take, replace=(take > s_full.shape[0]))
            cross_draws = s_full[idx]
            use_posterior_samples = True
        else:
            cross_draws = None
    else:
        cross_draws = None

    if cross_draws is None:
        ns = int(max(200, nsamp))
        std = np.where(np.isfinite(cross_params_err), np.abs(cross_params_err), 0.0)
        cross_draws = cross_params[None, :] + rng.normal(0.0, std[None, :], size=(ns, cross_params.size))
        # Keep amplitudes non-negative and damping positive.
        for idx in [0, 1, 4]:
            if idx < cross_draws.shape[1]:
                cross_draws[:, idx] = np.clip(cross_draws[:, idx], 0.0, None)
        if cross_draws.shape[1] >= 6:
            cross_draws[:, 5] = np.clip(cross_draws[:, 5], 0.1, None)

    cross_model = CrossPowerSpectrumModel(
        lb_eval,
        use_powerlaw_2h=True,
        alpha_2h_fixed=0.0,
        use_lorentzian_1h=False,
        use_astrometry_damping=(cross_params.size >= 6),
        use_one_halo=True,
    )

    ns = cross_draws.shape[0]
    nlb = lb_eval.size
    dl_cross_2h_draws = np.full((ns, nlb), np.nan, dtype=float)
    dl_cross_1h_draws = np.full((ns, nlb), np.nan, dtype=float)
    dl_cross_2h1h_draws = np.full((ns, nlb), np.nan, dtype=float)
    for i in range(ns):
        comp = cross_model.model_components(lb_eval, *cross_draws[i])
        d2 = np.asarray(comp["two_halo"], dtype=float)
        d1 = np.asarray(comp["one_halo"], dtype=float)
        dl_cross_2h_draws[i] = d2
        dl_cross_1h_draws[i] = d1
        dl_cross_2h1h_draws[i] = d2 + d1

    denom_draws = np.full((ns, nlb), np.nan, dtype=float)
    denom_med = np.full(nlb, np.nan, dtype=float)
    denom_err = np.full(nlb, np.nan, dtype=float)
    dl_auto_2h_draws = np.full((ns, nlb), np.nan, dtype=float)
    dl_auto_1h_draws = np.full((ns, nlb), np.nan, dtype=float)

    if gal_auto_denominator_mode == "shot-subtracted-data":
        gal_sub = _estimate_auto_2h_from_shot_subtracted_cl(
            gal_lb,
            gal_cl,
            gal_cl_err,
            ell_2h_max=float(ell_range_for_scaling[1]),
            shot_ell_min=float(shot_ell_min),
            shot_ell_max=float(shot_ell_max),
        )
        dl_sub_native = np.asarray(gal_sub["dl_sub"], dtype=float)
        dl_sub_err_native = np.asarray(gal_sub["dl_sub_err"], dtype=float)
        denom_med = np.interp(lb_eval, gal_lb, dl_sub_native, left=np.nan, right=np.nan)
        denom_err = np.interp(lb_eval, gal_lb, dl_sub_err_native, left=np.nan, right=np.nan)
        denom_draws = denom_med[None, :] + rng.normal(0.0, denom_err[None, :], size=(ns, nlb))
    else:
        if gal_auto_params is None:
            raise ValueError("smooth-model denominator requires gal_auto_params")
        p_auto = np.asarray(gal_auto_params, dtype=float)
        p_auto_err = np.asarray(gal_auto_params_err, dtype=float) if gal_auto_params_err is not None else np.zeros_like(p_auto)

        if gal_auto_samples is not None and np.asarray(gal_auto_samples).ndim == 2 and np.asarray(gal_auto_samples).shape[0] >= 2:
            s_auto_raw = np.asarray(gal_auto_samples, dtype=float)
            if s_auto_raw.shape[1] != p_auto.size:
                s_auto_full = _expand_param_samples_to_full(s_auto_raw, p_auto)
            else:
                s_auto_full = s_auto_raw
            idx_auto = rng.choice(s_auto_full.shape[0], size=ns, replace=(ns > s_auto_full.shape[0]))
            auto_draws = s_auto_full[idx_auto]
        else:
            std_auto = np.where(np.isfinite(p_auto_err), np.abs(p_auto_err), 0.0)
            auto_draws = p_auto[None, :] + rng.normal(0.0, std_auto[None, :], size=(ns, p_auto.size))
            for idx in [0, 1, 4]:
                if idx < auto_draws.shape[1]:
                    auto_draws[:, idx] = np.clip(auto_draws[:, idx], 0.0, None)

        auto_model = CrossPowerSpectrumModel(
            lb_eval,
            use_powerlaw_2h=True,
            alpha_2h_fixed=0.0,
            use_lorentzian_1h=False,
            use_astrometry_damping=False,
            use_one_halo=True,
        )
        for i in range(ns):
            comp_auto = auto_model.model_components(lb_eval, *np.asarray(auto_draws[i][:5], dtype=float))
            d2a = np.asarray(comp_auto["two_halo"], dtype=float)
            d1a = np.asarray(comp_auto["one_halo"], dtype=float)
            dl_auto_2h_draws[i] = d2a
            dl_auto_1h_draws[i] = d1a
            denom_draws[i] = d2a + d1a

        denom_med = _column_percentile(denom_draws, 50.0)
        denom_p16 = _column_percentile(denom_draws, 16.0)
        denom_p84 = _column_percentile(denom_draws, 84.0)
        denom_err = 0.5 * (denom_p84 - denom_p16)

    with np.errstate(divide="ignore", invalid="ignore"):
        if ratio_scaling_mode == "separate-2h-1h":
            if gal_auto_denominator_mode != "smooth-model":
                raise ValueError(
                    "ratio_scaling_mode='separate-2h-1h' requires gal_auto_denominator_mode='smooth-model'"
                )
            valid2 = (
                np.isfinite(dl_cross_2h_draws)
                & np.isfinite(dl_auto_2h_draws)
                & (dl_auto_2h_draws > 0.0)
                & (dl_cross_2h_draws >= 0.0)
            )
            valid1 = (
                np.isfinite(dl_cross_1h_draws)
                & np.isfinite(dl_auto_1h_draws)
                & (dl_auto_1h_draws > 0.0)
                & (dl_cross_1h_draws >= 0.0)
            )
            pred2 = np.where(valid2, (dl_cross_2h_draws ** 2) / dl_auto_2h_draws, np.nan)
            pred1 = np.where(valid1, (dl_cross_1h_draws ** 2) / dl_auto_1h_draws, np.nan)
            dl_pred_draws = pred2 + pred1
            valid = np.isfinite(dl_pred_draws)
            ratio_draws = np.where(
                np.isfinite(dl_cross_2h1h_draws) & np.isfinite(denom_draws) & (denom_draws > 0.0),
                dl_cross_2h1h_draws / denom_draws,
                np.nan,
            )
        else:
            valid = (
                np.isfinite(dl_cross_2h1h_draws)
                & np.isfinite(denom_draws)
                & (denom_draws > 0.0)
                & (dl_cross_2h1h_draws >= 0.0)
            )
            dl_pred_draws = np.where(valid, (dl_cross_2h1h_draws ** 2) / denom_draws, np.nan)
            ratio_draws = np.where(valid, dl_cross_2h1h_draws / denom_draws, np.nan)

    dl_pred_med = _column_percentile(dl_pred_draws, 50.0)
    dl_pred_p16 = _column_percentile(dl_pred_draws, 16.0)
    dl_pred_p84 = _column_percentile(dl_pred_draws, 84.0)
    dl_pred_err = 0.5 * (dl_pred_p84 - dl_pred_p16)
    dl_pred_2h_only_med = np.full(nlb, np.nan, dtype=float)
    dl_pred_1h_only_med = np.full(nlb, np.nan, dtype=float)
    if ratio_scaling_mode == "separate-2h-1h" and gal_auto_denominator_mode == "smooth-model":
        pred2 = np.where(
            np.isfinite(dl_cross_2h_draws) & np.isfinite(dl_auto_2h_draws) & (dl_auto_2h_draws > 0.0),
            (dl_cross_2h_draws ** 2) / dl_auto_2h_draws,
            np.nan,
        )
        pred1 = np.where(
            np.isfinite(dl_cross_1h_draws) & np.isfinite(dl_auto_1h_draws) & (dl_auto_1h_draws > 0.0),
            (dl_cross_1h_draws ** 2) / dl_auto_1h_draws,
            np.nan,
        )
        dl_pred_2h_only_med = _column_percentile(pred2, 50.0)
        dl_pred_1h_only_med = _column_percentile(pred1, 50.0)

    dl_cross_2h1h_med = _column_percentile(dl_cross_2h1h_draws, 50.0)

    ratio_mask = (lb_eval >= float(ell_range_for_scaling[0])) & (lb_eval <= float(ell_range_for_scaling[1]))
    ratio_scalar_draw = np.full(ns, np.nan, dtype=float)
    for i in range(ns):
        row = ratio_draws[i]
        m = ratio_mask & np.isfinite(row)
        if np.any(m):
            ratio_scalar_draw[i] = float(np.nanmedian(row[m]))
    m_ratio = np.isfinite(ratio_scalar_draw)
    if np.any(m_ratio):
        s2h1h = float(np.percentile(ratio_scalar_draw[m_ratio], 50.0))
        s2h1h_p16 = float(np.percentile(ratio_scalar_draw[m_ratio], 16.0))
        s2h1h_p84 = float(np.percentile(ratio_scalar_draw[m_ratio], 84.0))
        s2h1h_err = 0.5 * (s2h1h_p84 - s2h1h_p16)
    else:
        s2h1h = np.nan
        s2h1h_err = np.nan

    n_eff = np.sum(np.isfinite(dl_pred_draws), axis=0).astype(int)
    valid_frac = n_eff.astype(float) / float(ns)

    return {
        "dl_pred_2h1h": dl_pred_med,
        "dl_pred_2h1h_err": dl_pred_err,
        "dl_pred_2h1h_p16": dl_pred_p16,
        "dl_pred_2h1h_p84": dl_pred_p84,
        "dl_pred_2h_only": dl_pred_2h_only_med,
        "dl_pred_1h_only": dl_pred_1h_only_med,
        "dl_cross_2h1h_med": dl_cross_2h1h_med,
        "dl_gal_sub": denom_med,
        "dl_gal_sub_err": denom_err,
        "s2h1h": np.array(s2h1h),
        "s2h1h_err": np.array(s2h1h_err),
        "n_eff": n_eff,
        "valid_frac": valid_frac,
        "n_draws": np.array(int(ns)),
        "used_posterior_samples": np.array(bool(use_posterior_samples)),
        "gal_auto_denominator_mode": np.array(gal_auto_denominator_mode),
        "ratio_scaling_mode": np.array(ratio_scaling_mode),
    }


def _make_zlt1_fit_diagnostics_figure(
    res_ls: Dict[str, Any],
    res_hsc: Dict[str, Any],
    params_ls_cross: np.ndarray,
    params_hsc_cross: np.ndarray,
    params_ls_auto: np.ndarray,
    params_hsc_auto: np.ndarray,
    ls_cross_fit: Dict[str, Any],
    hsc_cross_fit: Dict[str, Any],
    ls_auto_fit: Dict[str, Any],
    hsc_auto_fit: Dict[str, Any],
    startidx: int,
    endidx: int,
) -> Any:
    from ciber.theory.cross_ps_parametric_model import CrossPowerSpectrumModel
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8), sharex=False, sharey=False)
    lb = np.asarray(res_ls["lb"], dtype=float)[startidx:endidx]
    pf = lb * (lb + 1.0) / (2.0 * np.pi)

    panels = [
        ("DESI-LS cross TM1", "cross", 0, res_ls, params_ls_cross[0, 0], ls_cross_fit, 0),
        ("DESI-LS cross TM2", "cross", 1, res_ls, params_ls_cross[1, 0], ls_cross_fit, 1),
        ("HSC cross TM1", "cross", 0, res_hsc, params_hsc_cross[0, 0], hsc_cross_fit, 0),
        ("HSC cross TM2", "cross", 1, res_hsc, params_hsc_cross[1, 0], hsc_cross_fit, 1),
        ("DESI-LS auto TM1", "auto", 0, res_ls, params_ls_auto[0, 0], ls_auto_fit, 0),
        ("HSC auto TM1", "auto", 0, res_hsc, params_hsc_auto[0, 0], hsc_auto_fit, 0),
    ]

    for ax, (title, mode, inst_idx, res, params, fitres, fit_inst_idx) in zip(axes.ravel(), panels):
        if mode == "cross":
            data = pf * np.asarray(res["full_cl_cross"][inst_idx, 0], dtype=float)[startidx:endidx]
            derr = pf * np.asarray(res["full_clerr_cross"][inst_idx, 0], dtype=float)[startidx:endidx]
            model = CrossPowerSpectrumModel(
                lb,
                use_powerlaw_2h=True,
                alpha_2h_fixed=0.0,
                use_lorentzian_1h=False,
                use_astrometry_damping=True,
                use_one_halo=True,
            )
            comp = model.model_components(lb, *np.asarray(params, dtype=float))
        else:
            data = pf * np.asarray(res["full_cl_gal"][0, 0], dtype=float)[startidx:endidx]
            derr = pf * np.asarray(res["full_clerr_gal"][0, 0], dtype=float)[startidx:endidx]
            model = CrossPowerSpectrumModel(
                lb,
                use_powerlaw_2h=True,
                alpha_2h_fixed=0.0,
                use_lorentzian_1h=False,
                mu_1h_fixed=8.0,
                sigma_1h_fixed=0.7,
                use_astrometry_damping=False,
                use_one_halo=True,
            )
            comp = model.model_components(lb, *np.asarray(params, dtype=float)[:5])

        good = np.isfinite(lb) & np.isfinite(data) & np.isfinite(derr)
        good &= lb > 0
        good &= data > 0
        if not np.any(good):
            ax.text(0.5, 0.5, "No valid points", ha="center", va="center", transform=ax.transAxes)
            continue

        lbp = lb[good]
        datap = data[good]
        derrp = derr[good]
        ax.errorbar(lbp, datap, yerr=derrp, fmt="o", color="k", markersize=2.8, capsize=1.8, alpha=0.85)
        ax.plot(lbp, np.asarray(comp["total"])[good], color="C3", linewidth=1.7, label="total")
        ax.plot(lbp, np.asarray(comp["two_halo"])[good], color="C0", linewidth=1.2, label="2h")
        ax.plot(lbp, np.asarray(comp["one_halo"])[good], color="C2", linewidth=1.2, label="1h")
        ax.plot(lbp, np.asarray(comp["shot_noise"])[good], color="C4", linewidth=1.2, linestyle="--", label="shot")

        chisq = float(np.asarray(fitres["chisq"])[fit_inst_idx, 0])
        rchi2 = float(np.asarray(fitres["reduced_chisq"])[fit_inst_idx, 0])
        ndof = chisq / rchi2 if np.isfinite(rchi2) and rchi2 > 0 else np.nan
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        ax.text(0.05, 0.07, f"$\\chi^2/N_{{dof}}={chisq:.1f}/{ndof:.0f}={rchi2:.2f}$", transform=ax.transAxes, fontsize=8)
        ax.set_xlabel("$\\ell$")
        ax.set_ylabel("$D_\\ell$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=9, frameon=False)
        fig.subplots_adjust(top=0.90)

    fig.tight_layout()
    return fig


def _derive_fixed_one_halo_from_slice_fits(
    slice_fit_files: Sequence[str],
    mode: str = "slice-median",
    effective_z: float = 0.5,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
) -> Tuple[float, float, int]:
    """Derive one fixed (mu_1h, sigma_1h) from sliced fit-result files.

    Parameters
    ----------
    slice_fit_files
        Iterable of fit-result npz files that contain `params`, `param_names`, and z metadata.
    mode
        `slice-median` for global medians over all slice entries, or
        `slice-effective-z` for medians at slice center nearest `effective_z`, or
        `slice-z-range` for medians within [z_min, z_max].
    effective_z
        Target redshift used when mode is `slice-effective-z`.
    """

    from ciber.theory.cross_ps_parametric_model import load_fit_results_npz

    records: List[Tuple[float, float, float]] = []

    for fpath in slice_fit_files:
        fit = load_fit_results_npz(str(fpath))
        if "params" not in fit or "param_names" not in fit:
            continue

        params = np.asarray(fit["params"], dtype=float)
        if params.ndim != 3:
            continue

        param_names = [str(x) for x in np.asarray(fit["param_names"]).tolist()]
        if "mu_1h" not in param_names or "sigma_1h" not in param_names:
            continue

        mu_idx = param_names.index("mu_1h")
        sigma_idx = param_names.index("sigma_1h")

        if "z_centers" in fit:
            z_centers = np.asarray(fit["z_centers"], dtype=float)
        elif "zbinedges" in fit:
            zbedges = np.asarray(fit["zbinedges"], dtype=float)
            z_centers = 0.5 * (zbedges[:-1] + zbedges[1:])
        else:
            continue

        nz = min(params.shape[1], z_centers.shape[0])
        for zidx in range(nz):
            zc = float(z_centers[zidx])
            for iinst in range(params.shape[0]):
                mu_val = float(params[iinst, zidx, mu_idx])
                sigma_val = float(params[iinst, zidx, sigma_idx])
                if np.isfinite(mu_val) and np.isfinite(sigma_val):
                    records.append((zc, mu_val, sigma_val))

    if len(records) == 0:
        raise ValueError(
            "No usable (z, mu_1h, sigma_1h) records found in slice-fit files. "
            "Provide files that contain param_names with mu_1h/sigma_1h and 3D params arrays."
        )

    rec_arr = np.asarray(records, dtype=float)
    if mode == "slice-effective-z":
        dz = np.abs(rec_arr[:, 0] - float(effective_z))
        zmin = np.min(dz)
        use = rec_arr[np.isclose(dz, zmin)]
    elif mode == "slice-z-range":
        if z_min is None or z_max is None:
            raise ValueError("mode='slice-z-range' requires z_min and z_max")
        z0 = float(min(z_min, z_max))
        z1 = float(max(z_min, z_max))
        use = rec_arr[(rec_arr[:, 0] >= z0) & (rec_arr[:, 0] <= z1)]
        if use.size == 0:
            raise ValueError(
                f"No usable one-halo slice entries in z-range [{z0:.3f}, {z1:.3f}]"
            )
    else:
        use = rec_arr

    mu_fixed = float(np.median(use[:, 1]))
    sigma_fixed = float(np.median(use[:, 2]))
    return mu_fixed, sigma_fixed, int(use.shape[0])


def _plot_zlt1_auto_prediction_comparison(
    lb: np.ndarray,
    dl_meas: np.ndarray,
    dl_meas_err: np.ndarray,
    dl_pred_ls: np.ndarray,
    dl_pred_ls_err: np.ndarray,
    dl_pred_hsc: np.ndarray,
    dl_pred_hsc_err: np.ndarray,
    dl_pred_ls_2h: Optional[np.ndarray] = None,
    dl_pred_hsc_2h: Optional[np.ndarray] = None,
    dgl_dl: Optional[np.ndarray] = None,
    dgl_dl_err: Optional[np.ndarray] = None,
    shot_dl: Optional[np.ndarray] = None,
    mock_igl_dl: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (7.0, 3.5),
    xlim: Tuple[float, float] = (275.0, 1.0e5),
    ylim: Tuple[float, float] = (1e-2, 1e4),
) -> Any:
    from matplotlib import pyplot as plt

    def _extend_series_loglog(x_src: np.ndarray, y_src: np.ndarray, x_tgt: np.ndarray) -> np.ndarray:
        """Interpolate in log-log space and extrapolate with end slopes."""
        x_src = np.asarray(x_src, dtype=float)
        y_src = np.asarray(y_src, dtype=float)
        x_tgt = np.asarray(x_tgt, dtype=float)

        out = np.full(x_tgt.shape, np.nan, dtype=float)
        m = np.isfinite(x_src) & np.isfinite(y_src) & (x_src > 0.0) & (y_src > 0.0)
        if np.sum(m) < 2:
            return out

        xs = x_src[m]
        ys = y_src[m]
        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]
        lx = np.log10(xs)
        ly = np.log10(ys)
        lxt = np.log10(np.clip(x_tgt, 1.0e-12, None))

        out_interp = np.interp(lxt, lx, ly, left=np.nan, right=np.nan)
        out = np.power(10.0, out_interp)

        if xs.size >= 2:
            left_slope = (ly[1] - ly[0]) / (lx[1] - lx[0]) if lx[1] != lx[0] else 0.0
            right_slope = (ly[-1] - ly[-2]) / (lx[-1] - lx[-2]) if lx[-1] != lx[-2] else 0.0

            m_left = x_tgt < xs[0]
            if np.any(m_left):
                out[m_left] = np.power(10.0, ly[0] + left_slope * (lxt[m_left] - lx[0]))

            m_right = x_tgt > xs[-1]
            if np.any(m_right):
                out[m_right] = np.power(10.0, ly[-1] + right_slope * (lxt[m_right] - lx[-1]))

        return out

    fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    x_line = np.logspace(np.log10(float(xlim[0])), np.log10(float(xlim[1])), 256)
    titles = ["CIBER 1.1 $\\mu$m", "CIBER 1.8 $\\mu$m"]

    for idx in range(2):
        ax[idx].errorbar(
            lb,
            dl_meas[idx],
            yerr=dl_meas_err[idx],
            fmt="o",
            color="k",
            markersize=3,
            capsize=2.5,
            label="CIBER auto (Feder+25b)",
            zorder=20,
        )
        ax[idx].plot(
            lb,
            dl_pred_ls[idx],
            color="C0",
            linestyle="solid",
            linewidth=2.0,
            label="Reconstructed auto (DESI-LS, $z<1$)",
        )
        ax[idx].plot(
            lb,
            dl_pred_hsc[idx],
            color="C1",
            linestyle="solid",
            linewidth=2.0,
            label="Reconstructed auto (HSC, $z<1$)",
        )

        if dl_pred_ls_2h is not None:
            ax[idx].plot(
                lb,
                dl_pred_ls_2h[idx],
                color="C0",
                linestyle="--",
                linewidth=1.4,
                alpha=0.9,
                label="DESI-LS 2h model",
            )
        if dl_pred_hsc_2h is not None:
            ax[idx].plot(
                lb,
                dl_pred_hsc_2h[idx],
                color="C1",
                linestyle="--",
                linewidth=1.4,
                alpha=0.9,
                label="HSC 2h model",
            )

        if np.any(np.isfinite(dl_pred_ls_err[idx])):
            ax[idx].fill_between(
                lb,
                np.clip(dl_pred_ls[idx] - dl_pred_ls_err[idx], 1e-12, None),
                dl_pred_ls[idx] + dl_pred_ls_err[idx],
                color="C0",
                alpha=0.15,
            )
        if np.any(np.isfinite(dl_pred_hsc_err[idx])):
            ax[idx].fill_between(
                lb,
                np.clip(dl_pred_hsc[idx] - dl_pred_hsc_err[idx], 1e-12, None),
                dl_pred_hsc[idx] + dl_pred_hsc_err[idx],
                color="C1",
                alpha=0.15,
            )

        if dgl_dl is not None and np.any(np.isfinite(dgl_dl[idx])):
            y = np.asarray(dgl_dl[idx], dtype=float)
            y_ext = _extend_series_loglog(lb, y, x_line)
            ax[idx].plot(x_line, y_ext, color="k", linewidth=1.4, linestyle="-", label="DGL (Feder+25b)")
            if dgl_dl_err is not None and np.any(np.isfinite(dgl_dl_err[idx])):
                dy = np.asarray(dgl_dl_err[idx], dtype=float)
                y_lo_ext = _extend_series_loglog(lb, np.clip(y - dy, 1e-12, None), x_line)
                y_hi_ext = _extend_series_loglog(lb, np.clip(y + dy, 1e-12, None), x_line)
                ax[idx].fill_between(
                    x_line,
                    np.clip(y_lo_ext, 1e-12, None),
                    np.clip(y_hi_ext, 1e-12, None),
                    color="k",
                    alpha=0.12,
                )

        if shot_dl is not None and np.any(np.isfinite(shot_dl[idx])):
            shot_curve = _extend_series_loglog(lb, np.asarray(shot_dl[idx], dtype=float), x_line)
            ax[idx].plot(
                x_line,
                shot_curve,
                color="0.45",
                linewidth=1.4,
                linestyle="--",
                label="Best-fit Poisson level",
            )

        if mock_igl_dl is not None and np.any(np.isfinite(mock_igl_dl[idx])):
            mock_curve = _extend_series_loglog(lb, np.asarray(mock_igl_dl[idx], dtype=float), x_line)
            ax[idx].plot(
                x_line,
                mock_curve,
                color="C3",
                linewidth=1.5,
                linestyle=":",
                label="Mock IGL ($z<1$)",
            )

        ax[idx].text(
            0.03,
            0.97,
            titles[idx],
            transform=ax[idx].transAxes,
            ha="left",
            va="top",
            fontsize=12,
        )
        ax[idx].set_xscale("log")
        ax[idx].set_yscale("log")
        ax[idx].set_xlim(xlim)
        ax[idx].set_ylim(ylim)
        ax[idx].set_xlabel("$\\ell$", fontsize=12)
        ax[idx].grid(alpha=0.3)

    ax[0].set_ylabel("$D_\\ell^{II}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]", fontsize=12)
    handles, labels = ax[0].get_legend_handles_labels()
    if handles:
        handle_by_label = {lab: h for h, lab in zip(handles, labels)}
        target_order = [
            "CIBER auto (Feder+25b)",
            "DGL (Feder+25b)",
            "Best-fit Poisson level",
            "Mock IGL ($z<1$)",
            "Reconstructed auto (DESI-LS, $z<1$)",
            "Reconstructed auto (HSC, $z<1$)",
        ]
        ordered_labels = [lab for lab in target_order if lab in handle_by_label]
        ordered_handles = [handle_by_label[lab] for lab in ordered_labels]
        fig.legend(
            ordered_handles,
            ordered_labels,
            loc="upper left",
            bbox_to_anchor=(0.15, 1.1),
            ncol=2,
            fontsize=9,
            frameon=False,
        )
    plt.subplots_adjust(wspace=0.05, top=0.82)
    return fig


def run_zlt1_auto_prediction(args: argparse.Namespace) -> List[GeneratedFigure]:
    """Run z<1 parametric-fit workflow and generate CIBER auto prediction comparison.

    Steps:
    1) Fit cross spectra for DESI-LS and HSC with 2h+1h+shot parametric model.
    2) Fit galaxy auto spectra (saved for reproducibility).
    3) Predict CIBER auto from z<1 cross+galaxy spectra per tracer (LS/HSC).
    4) Save prediction products and return two-panel comparison figure.
    """

    from ciber.plotting.gal_plotting_fns import collect_ciber_gal_vs_redshift
    from ciber.theory.cross_ps_parametric_model import (
        load_fit_results_npz,
        run_gal_auto_fits,
        run_gal_cross_fits,
    )

    if args.fit_mode == "bulk":
        # Match omnibus-style bulk measurements: one z<1 bin.
        zbinedges = [0.0, 1.0]
    else:
        zbinedges = _zlt1_coarse_bins()
    inst_list = [1, 2]
    ifield_list = [4, 5, 6, 7, 8]
    ifield_list_hsc = [8]
    cross_lmax = float(args.cross_fit_lmax)
    auto_lmax = float(args.auto_fit_lmax)
    cross_use_damping = True
    auto_use_damping = False

    fit_tag = args.fit_tag
    nwalkers_fit = max(int(args.fit_nwalkers), 20)
    ihl_1h_params_path = Path(args.ihl_1h_params_path).expanduser()
    if not ihl_1h_params_path.is_absolute():
        ihl_1h_params_path = REPO_ROOT / ihl_1h_params_path
    has_ihl_1h_params = ihl_1h_params_path.exists()

    fixed_mu_1h: Optional[float] = None
    fixed_sigma_1h: Optional[float] = None
    if args.one_halo_template_mode != "legacy":
        if not args.slice_template_fit_files:
            raise ValueError(
                "one-halo-template-mode requires --slice-template-fit-files when mode is not legacy"
            )
        slice_files = [str(Path(p).expanduser()) for p in args.slice_template_fit_files]
        fixed_mu_1h, fixed_sigma_1h, n_used = _derive_fixed_one_halo_from_slice_fits(
            slice_files,
            mode=args.one_halo_template_mode,
            effective_z=float(args.one_halo_effective_z),
        )
        print(
            "Derived fixed one-halo template from slice fits: "
            f"mode={args.one_halo_template_mode}, mu_1h={fixed_mu_1h:.4f}, "
            f"sigma_1h={fixed_sigma_1h:.4f}, n_used={n_used}"
        )

    # For cross fits, fixed 1h shape requires IHL parameter file.
    # Fall back to free 1h shape if file is absent, with corresponding priors/starts.
    cross_fix_ihl_1h_shape = (fixed_mu_1h is not None and fixed_sigma_1h is not None) or has_ihl_1h_params
    if cross_fix_ihl_1h_shape:
        if cross_use_damping:
            cross_lower_bounds = np.array([0.0, 0.0, 0.0, 0.1], dtype=float)
            cross_upper_bounds = np.array([10.0, 100.0, 10.0, 7.0], dtype=float)
            cross_initial_guess = np.array([0.1, 1.0, 0.5, 2.0], dtype=float)
        else:
            cross_lower_bounds = np.array([0.0, 0.0, 0.0], dtype=float)
            cross_upper_bounds = np.array([10.0, 100.0, 10.0], dtype=float)
            cross_initial_guess = np.array([0.1, 1.0, 0.5], dtype=float)
    else:
        if cross_use_damping:
            cross_lower_bounds = np.array([0.0, 0.0, 6.5, 0.2, 0.0, 0.1], dtype=float)
            cross_upper_bounds = np.array([10.0, 100.0, 9.5, 1.2, 10.0, 7.0], dtype=float)
            cross_initial_guess = np.array([0.1, 1.0, 8.0, 0.7, 0.5, 2.0], dtype=float)
        else:
            cross_lower_bounds = np.array([0.0, 0.0, 6.5, 0.2, 0.0], dtype=float)
            cross_upper_bounds = np.array([10.0, 100.0, 9.5, 1.2, 10.0], dtype=float)
            cross_initial_guess = np.array([0.1, 1.0, 8.0, 0.7, 0.5], dtype=float)
    cross_prior_bounds = np.array([cross_lower_bounds, cross_upper_bounds], dtype=float)

    # Galaxy-auto fits in run_gal_auto_fits always fit amplitudes with fixed 1h shape
    # (IHL-derived when available; approximation otherwise), so keep 3/4-D bounds here.
    if auto_use_damping:
        auto_lower_bounds = np.array([0.0, 0.0, 0.0, 0.1], dtype=float)
        auto_upper_bounds = np.array([10.0, 100.0, 10.0, 7.0], dtype=float)
    else:
        auto_lower_bounds = np.array([0.0, 0.0, 0.0], dtype=float)
        auto_upper_bounds = np.array([10.0, 100.0, 10.0], dtype=float)
    auto_prior_bounds = np.array([auto_lower_bounds, auto_upper_bounds], dtype=float)

    if (fixed_mu_1h is None or fixed_sigma_1h is None) and not has_ihl_1h_params:
        print(
            "Warning: IHL 1h parameter file not found at "
            f"{ihl_1h_params_path}. Cross fits will free mu/sigma with tempered priors."
        )

    print(
        "Running with fit configuration: "
        f"mode={args.fit_mode}, zbins={zbinedges}, cross_lMax={cross_lmax}, auto_lMax={auto_lmax}, "
        f"cross_damping={cross_use_damping}, auto_damping={auto_use_damping}, ihl_1h_params={ihl_1h_params_path}, "
        f"cross_fix_1h_shape={cross_fix_ihl_1h_shape}"
    )

    print(
        "Enforced fit plan: 4 cross fits (DESILS/HSC x TM1/TM2) with damping, "
        "2 galaxy-auto fits (DESILS/HSC, TM1-only) without damping."
    )

    if nwalkers_fit != int(args.fit_nwalkers):
        print(
            f"Requested --fit-nwalkers={args.fit_nwalkers} is too small for 5-parameter MCMC; "
            f"using {nwalkers_fit}."
        )
    ls_cross_file = f"ciber_cl_fits_DESILS_coarsez_{fit_tag}.npz"
    hsc_cross_file = f"ciber_cl_fits_HSC_coarsez_{fit_tag}.npz"
    ls_auto_file = f"gal_auto_fits_DESILS_coarsez_{fit_tag}.npz"
    hsc_auto_file = f"gal_auto_fits_HSC_coarsez_{fit_tag}.npz"

    save_intermediate = not args.no_save_intermediate_fits
    base_intermediate_dir = REPO_ROOT / "figures" / "generated_gal_cross_10deg" / f"zlt1_intermediate_{fit_tag}"
    cross_fig_dir_ls = base_intermediate_dir / "cross_fit_desils"
    cross_fig_dir_hsc = base_intermediate_dir / "cross_fit_hsc"
    auto_fig_dir_ls = base_intermediate_dir / "auto_fit_desils"
    auto_fig_dir_hsc = base_intermediate_dir / "auto_fit_hsc"
    for d in [cross_fig_dir_ls, cross_fig_dir_hsc, auto_fig_dir_ls, auto_fig_dir_hsc]:
        d.mkdir(parents=True, exist_ok=True)

    print("Running z<1 cross-spectrum fits (DESI-LS, HSC)...")
    run_gal_cross_fits(
        inst_list=inst_list,
        ifield_list=ifield_list,
        cat="DESILS",
        zbinedges=zbinedges,
        maskstr="JHlt16_wFFerr",
        chi2_eval_max=cross_lmax,
        lMax_fit=cross_lmax,
        use_ihl_templates=False,
        fix_ihl_1h_shape=cross_fix_ihl_1h_shape,
        use_ihl_1h_params=True,
        ihl_1h_params_path=str(ihl_1h_params_path),
        save_figs=save_intermediate,
        figbasedir=str(cross_fig_dir_ls) + os.sep,
        save_results=True,
        file_fpath=ls_cross_file,
        fitstr=fit_tag,
        prior_bounds=cross_prior_bounds,
        initial_guess=cross_initial_guess,
        chi2_lim=[-6, 6],
        use_astrometry_damping=cross_use_damping,
        mu_1h_fixed_override=fixed_mu_1h,
        sigma_1h_fixed_override=fixed_sigma_1h,
        nwalkers=nwalkers_fit,
        nsteps=args.fit_nsteps,
        nburn=args.fit_nburn,
    )
    run_gal_cross_fits(
        inst_list=inst_list,
        ifield_list=ifield_list_hsc,
        cat="HSC",
        zbinedges=zbinedges,
        maskstr=None,
        headstr=args.hsc_headstr,
        chi2_eval_max=cross_lmax,
        lMax_fit=cross_lmax,
        use_ihl_templates=False,
        fix_ihl_1h_shape=cross_fix_ihl_1h_shape,
        use_ihl_1h_params=True,
        ihl_1h_params_path=str(ihl_1h_params_path),
        save_figs=save_intermediate,
        figbasedir=str(cross_fig_dir_hsc) + os.sep,
        save_results=True,
        file_fpath=hsc_cross_file,
        fitstr=fit_tag,
        prior_bounds=cross_prior_bounds,
        initial_guess=cross_initial_guess,
        chi2_lim=[-6, 6],
        use_astrometry_damping=cross_use_damping,
        mu_1h_fixed_override=fixed_mu_1h,
        sigma_1h_fixed_override=fixed_sigma_1h,
        nwalkers=nwalkers_fit,
        nsteps=args.fit_nsteps,
        nburn=args.fit_nburn,
    )

    print("Running z<1 galaxy-auto fits (DESI-LS, HSC)...")
    run_gal_auto_fits(
        inst_list=[1],
        cat="DESILS",
        zbinedges=zbinedges,
        headstr=None,
        ifield_list=ifield_list,
        chi2_eval_max=auto_lmax,
        lMax_fit=auto_lmax,
        ihl_1h_params_path=str(ihl_1h_params_path),
        save_figs=save_intermediate,
        figbasedir=str(auto_fig_dir_ls) + os.sep,
        save_results=True,
        file_fpath=ls_auto_file,
        fitstr=fit_tag,
        prior_bounds=auto_prior_bounds,
        chi2_lim=[-6, 6],
        use_astrometry_damping=auto_use_damping,
        mu_1h_fixed_override=fixed_mu_1h,
        sigma_1h_fixed_override=fixed_sigma_1h,
        nwalkers=nwalkers_fit,
        nsteps=args.fit_nsteps,
        nburn=args.fit_nburn,
    )
    run_gal_auto_fits(
        inst_list=[1],
        cat="HSC",
        zbinedges=zbinedges,
        headstr=args.hsc_headstr,
        ifield_list=ifield_list_hsc,
        chi2_eval_max=auto_lmax,
        lMax_fit=auto_lmax,
        ihl_1h_params_path=str(ihl_1h_params_path),
        save_figs=save_intermediate,
        figbasedir=str(auto_fig_dir_hsc) + os.sep,
        save_results=True,
        file_fpath=hsc_auto_file,
        fitstr=fit_tag,
        prior_bounds=auto_prior_bounds,
        chi2_lim=[-6, 6],
        use_astrometry_damping=auto_use_damping,
        mu_1h_fixed_override=fixed_mu_1h,
        sigma_1h_fixed_override=fixed_sigma_1h,
        nwalkers=nwalkers_fit,
        nsteps=args.fit_nsteps,
        nburn=args.fit_nburn,
    )

    print("Collecting spectra and computing CIBER auto predictions from z<1 tracers...")
    res_ls = collect_ciber_gal_vs_redshift(
        "LS",
        subtract_randoms=True,
        inst_list=inst_list,
        zbinedges=zbinedges,
        maskstr="JHlt16_wFFerr",
        subtract_sn=False,
        tl_pix_correct=True,
        ifield_list=ifield_list,
    )
    res_hsc = collect_ciber_gal_vs_redshift(
        "HSC",
        subtract_randoms=True,
        inst_list=inst_list,
        zbinedges=zbinedges,
        maskstr=None,
        subtract_sn=False,
        tl_pix_correct=True,
        ifield_list=ifield_list_hsc,
        with_ff_err=True,
        headstr=args.hsc_headstr,
    )

    ell_scaling = (args.scale_ell_min, args.scale_ell_max)
    fit_range = (args.auto_fit_ell_min, args.auto_fit_ell_max)

    dl_meas = []
    dl_meas_err = []
    dl_pred_ls = []
    dl_pred_ls_err = []
    dl_pred_ls_2h = []
    dl_pred_hsc = []
    dl_pred_hsc_err = []
    dl_pred_hsc_2h = []

    lb_common: Optional[np.ndarray] = None

    ls_cross_res = load_fit_results_npz(str(REPO_ROOT / "data" / "cross_cl_fits" / ls_cross_file))
    hsc_cross_res = load_fit_results_npz(str(REPO_ROOT / "data" / "cross_cl_fits" / hsc_cross_file))
    ls_auto_res = load_fit_results_npz(str(REPO_ROOT / "data" / "gal_auto_fits" / ls_auto_file))
    hsc_auto_res = load_fit_results_npz(str(REPO_ROOT / "data" / "gal_auto_fits" / hsc_auto_file))

    params_ls_cross = np.asarray(ls_cross_res["params"], dtype=float)
    params_ls_cross_err = np.asarray(ls_cross_res["params_err"], dtype=float)
    params_hsc_cross = np.asarray(hsc_cross_res["params"], dtype=float)
    params_hsc_cross_err = np.asarray(hsc_cross_res["params_err"], dtype=float)
    params_ls_auto = np.asarray(ls_auto_res["params"], dtype=float)
    params_ls_auto_err = np.asarray(ls_auto_res["params_err"], dtype=float)
    params_hsc_auto = np.asarray(hsc_auto_res["params"], dtype=float)
    params_hsc_auto_err = np.asarray(hsc_auto_res["params_err"], dtype=float)
    ls_auto_samples = _extract_fit_samples_cell(ls_auto_res, 0, 0)
    hsc_auto_samples = _extract_fit_samples_cell(hsc_auto_res, 0, 0)

    formula_tag = (
        (
            "separate_2h1h_sum_crosssq_over_gg_components_model"
            if args.ratio_scaling_mode == "separate-2h-1h"
            else "per_ell_ratio_2h1h_crosssq_over_gg_model"
        )
        if args.gal_auto_denominator_mode == "smooth-model"
        else "per_ell_ratio_2h1h_crosssq_over_gg_sub"
    )

    for inst_idx in range(2):
        lb_meas, dl_m, dl_merr = _load_measured_ciber_auto_dl_exact(inst_idx)
        if lb_common is None:
            lb_common = lb_meas

        ls_cross_samples = _extract_fit_samples_cell(ls_cross_res, inst_idx, 0)
        hsc_cross_samples = _extract_fit_samples_cell(hsc_cross_res, inst_idx, 0)

        ls_comp = _predict_auto_2h1h_from_cross_and_shot_subtracted_auto(
            lb_meas,
            params_ls_cross[inst_idx, 0],
            params_ls_cross_err[inst_idx, 0],
            ls_cross_samples,
            np.asarray(res_ls["lb"], dtype=float),
            np.asarray(res_ls["full_cl_gal"][0, 0], dtype=float),
            np.asarray(res_ls["full_clerr_gal"][0, 0], dtype=float),
            ell_range_for_scaling=ell_scaling,
            shot_ell_min=float(args.shot_ell_min),
            shot_ell_max=float(args.shot_ell_max),
            gal_auto_denominator_mode=str(args.gal_auto_denominator_mode),
            ratio_scaling_mode=str(args.ratio_scaling_mode),
            gal_auto_params=params_ls_auto[0, 0],
            gal_auto_params_err=params_ls_auto_err[0, 0],
            gal_auto_samples=ls_auto_samples,
            nsamp=int(args.pred_nsamp),
            seed=123 + inst_idx,
        )
        hsc_comp = _predict_auto_2h1h_from_cross_and_shot_subtracted_auto(
            lb_meas,
            params_hsc_cross[inst_idx, 0],
            params_hsc_cross_err[inst_idx, 0],
            hsc_cross_samples,
            np.asarray(res_hsc["lb"], dtype=float),
            np.asarray(res_hsc["full_cl_gal"][0, 0], dtype=float),
            np.asarray(res_hsc["full_clerr_gal"][0, 0], dtype=float),
            ell_range_for_scaling=ell_scaling,
            shot_ell_min=float(args.shot_ell_min),
            shot_ell_max=float(args.shot_ell_max),
            gal_auto_denominator_mode=str(args.gal_auto_denominator_mode),
            ratio_scaling_mode=str(args.ratio_scaling_mode),
            gal_auto_params=params_hsc_auto[0, 0],
            gal_auto_params_err=params_hsc_auto_err[0, 0],
            gal_auto_samples=hsc_auto_samples,
            nsamp=int(args.pred_nsamp),
            seed=223 + inst_idx,
        )

        print(
            f"[2h+1h scaling] TM{inst_idx+1} DESI-LS s={float(ls_comp['s2h1h']):.3e} +/- {float(ls_comp['s2h1h_err']):.3e}; "
            f"HSC s={float(hsc_comp['s2h1h']):.3e} +/- {float(hsc_comp['s2h1h_err']):.3e}"
        )

        dl_meas.append(dl_m)
        dl_meas_err.append(dl_merr)
        dl_pred_ls.append(ls_comp["dl_pred_2h1h"])
        dl_pred_ls_err.append(ls_comp["dl_pred_2h1h_err"])
        dl_pred_ls_2h.append(ls_comp["dl_cross_2h1h_med"])
        dl_pred_hsc.append(hsc_comp["dl_pred_2h1h"])
        dl_pred_hsc_err.append(hsc_comp["dl_pred_2h1h_err"])
        dl_pred_hsc_2h.append(hsc_comp["dl_cross_2h1h_med"])

    dl_meas_arr = np.asarray(dl_meas)
    dl_meas_err_arr = np.asarray(dl_meas_err)
    dl_pred_ls_arr = np.asarray(dl_pred_ls)
    dl_pred_ls_err_arr = np.asarray(dl_pred_ls_err)
    dl_pred_ls_2h_arr = np.asarray(dl_pred_ls_2h)
    dl_pred_hsc_arr = np.asarray(dl_pred_hsc)
    dl_pred_hsc_err_arr = np.asarray(dl_pred_hsc_err)
    dl_pred_hsc_2h_arr = np.asarray(dl_pred_hsc_2h)

    fig_fit_diag = _make_zlt1_fit_diagnostics_figure(
        res_ls,
        res_hsc,
        params_ls_cross,
        params_hsc_cross,
        params_ls_auto,
        params_hsc_auto,
        ls_cross_res,
        hsc_cross_res,
        ls_auto_res,
        hsc_auto_res,
        args.startidx,
        args.endidx,
    )

    pred_outdir = REPO_ROOT / "data" / "ciber_auto_predictions"
    pred_outdir.mkdir(parents=True, exist_ok=True)
    pred_save_path = pred_outdir / f"ciber_auto_pred_from_zlt1_{fit_tag}.npz"
    np.savez(
        pred_save_path,
        lb=lb_common,
        inst_list=np.array(inst_list),
        zbinedges=np.array(zbinedges),
        dl_ciber_auto_measured=dl_meas_arr,
        dl_ciber_auto_measured_err=dl_meas_err_arr,
        dl_ciber_auto_pred_ls=dl_pred_ls_arr,
        dl_ciber_auto_pred_ls_err=dl_pred_ls_err_arr,
        dl_ciber_auto_pred_ls_2h=dl_pred_ls_2h_arr,
        dl_ciber_auto_pred_hsc=dl_pred_hsc_arr,
        dl_ciber_auto_pred_hsc_err=dl_pred_hsc_err_arr,
        dl_ciber_auto_pred_hsc_2h=dl_pred_hsc_2h_arr,
        pred_formula=np.array(formula_tag),
        pred_uncertainty=np.array("posterior_sample_propagation"),
        pred_nsamp=int(args.pred_nsamp),
        shot_sub_mode=np.array("prediction_stage" if args.gal_auto_denominator_mode == "shot-subtracted-data" else "not_used"),
        gal_auto_denominator_mode=np.array(str(args.gal_auto_denominator_mode)),
        ratio_scaling_mode=np.array(str(args.ratio_scaling_mode)),
        shot_ell_min=float(args.shot_ell_min),
        shot_ell_max=float(args.shot_ell_max),
        scale_ell_min=float(args.scale_ell_min),
        scale_ell_max=float(args.scale_ell_max),
        auto_fit_ell_min=float(args.auto_fit_ell_min),
        auto_fit_ell_max=float(args.auto_fit_ell_max),
        fit_tag=str(fit_tag),
        cross_fit_file_ls=str(REPO_ROOT / "data" / "cross_cl_fits" / ls_cross_file),
        cross_fit_file_hsc=str(REPO_ROOT / "data" / "cross_cl_fits" / hsc_cross_file),
        gal_auto_fit_file_ls=str(REPO_ROOT / "data" / "gal_auto_fits" / ls_auto_file),
        gal_auto_fit_file_hsc=str(REPO_ROOT / "data" / "gal_auto_fits" / hsc_auto_file),
    )
    print(f"Saved z<1 auto-prediction product -> {pred_save_path}")

    fig = _plot_zlt1_auto_prediction_comparison(
        lb=lb_common,
        dl_meas=dl_meas_arr,
        dl_meas_err=dl_meas_err_arr,
        dl_pred_ls=dl_pred_ls_arr,
        dl_pred_ls_err=dl_pred_ls_err_arr,
        dl_pred_ls_2h=dl_pred_ls_2h_arr,
        dl_pred_hsc=dl_pred_hsc_arr,
        dl_pred_hsc_err=dl_pred_hsc_err_arr,
        dl_pred_hsc_2h=dl_pred_hsc_2h_arr,
    )

    return [
        GeneratedFigure(
            "zlt1-auto-prediction:measured-vs-predicted",
            fig,
            "ciber_auto_vs_predicted_from_zlt1",
        ),
        GeneratedFigure(
            "zlt1-auto-prediction:fit-diagnostics",
            fig_fit_diag,
            "zlt1_cross_auto_fit_diagnostics",
        )
    ]


def _make_zlt1_simple_fit_figure(
    res_ls: Dict[str, Any],
    res_hsc: Dict[str, Any],
    ls_cross_fit: Dict[str, Any],
    hsc_cross_fit: Dict[str, Any],
    ls_auto_fit: Dict[str, Any],
    hsc_auto_fit: Dict[str, Any],
    lmax: float,
    startidx: int,
    endidx: int,
) -> Any:
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8), sharex=False, sharey=False)

    panels = [
        ("DESI-LS cross TM1", "cross", res_ls, ls_cross_fit, 0),
        ("DESI-LS cross TM2", "cross", res_ls, ls_cross_fit, 1),
        ("HSC cross TM1", "cross", res_hsc, hsc_cross_fit, 0),
        ("HSC cross TM2", "cross", res_hsc, hsc_cross_fit, 1),
        ("DESI-LS auto TM1", "auto", res_ls, ls_auto_fit, 0),
        ("HSC auto TM1", "auto", res_hsc, hsc_auto_fit, 0),
    ]

    for ax, (title, mode, res, fitres, inst_idx) in zip(axes.ravel(), panels):
        lb = np.asarray(res["lb"], dtype=float)[startidx:endidx]
        pf = lb * (lb + 1) / (2.0 * np.pi)

        if mode == "cross":
            dl_data = pf * np.asarray(res["full_cl_cross"][inst_idx, 0], dtype=float)[startidx:endidx]
            dl_err = pf * np.asarray(res["full_clerr_cross"][inst_idx, 0], dtype=float)[startidx:endidx]
        else:
            dl_data = pf * np.asarray(res["full_cl_gal"][0, 0], dtype=float)[startidx:endidx]
            dl_err = pf * np.asarray(res["full_clerr_gal"][0, 0], dtype=float)[startidx:endidx]

        lb_fit = np.asarray(fitres["lb_fit"][inst_idx, 0], dtype=float)
        model_dl = np.asarray(fitres["model_dl"][inst_idx, 0], dtype=float)
        reduced_chisq = float(np.asarray(fitres["reduced_chisq"])[inst_idx, 0])
        chisq = float(np.asarray(fitres["chisq"])[inst_idx, 0])

        ndof = chisq / reduced_chisq if np.isfinite(reduced_chisq) and reduced_chisq > 0 else np.nan

        m = np.isfinite(lb) & np.isfinite(dl_data) & np.isfinite(dl_err) & (lb > 0) & (dl_data > 0)
        mf = np.isfinite(lb_fit) & np.isfinite(model_dl) & (lb_fit > 0) & (model_dl > 0)
        if np.any(m):
            ax.errorbar(lb[m], dl_data[m], yerr=dl_err[m], fmt="o", color="k", markersize=2.8, capsize=1.8, alpha=0.85)
        if np.any(mf):
            ax.plot(lb_fit[mf], model_dl[mf], color="C3", linewidth=1.8)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        ax.text(0.05, 0.07, f"$\\chi^2/N_{{dof}}={chisq:.1f}/{ndof:.0f}={reduced_chisq:.2f}$", transform=ax.transAxes, fontsize=8)
        ax.set_xlabel("$\\ell$")
        ax.set_ylabel("$D_\\ell$")

    fig.suptitle(
        f"z<1 simple fits: cross=2h+shot+damping, auto=2h+shot, $\\ell_{{max}}$={int(lmax)}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    return fig


def _estimate_auto_2h_from_shot_subtracted_cl(
    lb: np.ndarray,
    cl: np.ndarray,
    cl_err: np.ndarray,
    ell_2h_max: float = 2000.0,
    shot_ell_min: float = 5.0e4,
    shot_ell_max: float = 8.0e4,
) -> Dict[str, np.ndarray]:
    """Estimate 2h level from galaxy auto by subtracting shot noise in C_ell.

    Strategy:
    1) Estimate shot-noise level from high-ell C_ell average.
    2) Subtract shot-noise from C_ell.
    3) Convert to D_ell and average D_ell over ell < ell_2h_max.
    """
    lb = np.asarray(lb, dtype=float)
    cl = np.asarray(cl, dtype=float)
    cl_err = np.asarray(cl_err, dtype=float)
    pf = lb * (lb + 1.0) / (2.0 * np.pi)

    m_shot = (
        (lb >= float(shot_ell_min))
        & (lb <= float(shot_ell_max))
        & np.isfinite(cl)
        & np.isfinite(cl_err)
        & (cl_err > 0)
    )
    if np.any(m_shot):
        w_shot = 1.0 / (cl_err[m_shot] ** 2)
        cl_shot = float(np.sum(w_shot * cl[m_shot]) / np.sum(w_shot))
        cl_shot_err = float(1.0 / np.sqrt(np.sum(w_shot)))
    else:
        m_fallback = np.isfinite(cl)
        cl_shot = float(np.nanmean(cl[m_fallback])) if np.any(m_fallback) else np.nan
        cl_shot_err = np.nan

    cl_sub = cl - cl_shot
    cl_sub_err = np.sqrt(cl_err**2 + cl_shot_err**2) if np.isfinite(cl_shot_err) else cl_err.copy()

    dl_sub = pf * cl_sub
    dl_sub_err = pf * cl_sub_err

    m_2h = (
        (lb < float(ell_2h_max))
        & np.isfinite(dl_sub)
        & np.isfinite(dl_sub_err)
        & (dl_sub_err > 0)
    )
    if np.any(m_2h):
        w_2h = 1.0 / (dl_sub_err[m_2h] ** 2)
        dl_2h_level = float(np.sum(w_2h * dl_sub[m_2h]) / np.sum(w_2h))
        dl_2h_level_err = float(1.0 / np.sqrt(np.sum(w_2h)))
    else:
        m2_fallback = (lb < float(ell_2h_max)) & np.isfinite(dl_sub)
        dl_2h_level = float(np.nanmean(dl_sub[m2_fallback])) if np.any(m2_fallback) else np.nan
        dl_2h_level_err = np.nan

    return {
        "cl_shot": np.array(cl_shot),
        "cl_shot_err": np.array(cl_shot_err),
        "dl_2h_level": np.array(dl_2h_level),
        "dl_2h_level_err": np.array(dl_2h_level_err),
        "lb": lb,
        "dl_sub": dl_sub,
        "dl_sub_err": dl_sub_err,
    }


def _make_zlt1_alt_auto_2h_figure(
    res_ls: Dict[str, Any],
    res_hsc: Dict[str, Any],
    ls_alt: Dict[str, np.ndarray],
    hsc_alt: Dict[str, np.ndarray],
    ell_2h_max: float,
    shot_ell_min: float,
    shot_ell_max: float,
    startidx: int,
    endidx: int,
) -> Any:
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharex=True, sharey=True)
    datasets = [
        ("DESI-LS", res_ls, ls_alt, "C0"),
        ("HSC", res_hsc, hsc_alt, "C3"),
    ]

    for ax, (label, res, alt, color) in zip(axes, datasets):
        lb = np.asarray(res["lb"], dtype=float)[startidx:endidx]
        pf = lb * (lb + 1.0) / (2.0 * np.pi)
        cl = np.asarray(res["full_cl_gal"][0, 0], dtype=float)[startidx:endidx]
        cl_err = np.asarray(res["full_clerr_gal"][0, 0], dtype=float)[startidx:endidx]
        dl = pf * cl
        dl_err = pf * cl_err

        m = np.isfinite(lb) & np.isfinite(dl) & np.isfinite(dl_err) & (lb > 0)
        if np.any(m):
            ax.errorbar(lb[m], dl[m], yerr=dl_err[m], fmt="o", color="k", markersize=3.0, capsize=2.0, alpha=0.85, label="Galaxy auto data")

        lb_alt = np.asarray(alt["lb"], dtype=float)
        dl_sub = np.asarray(alt["dl_sub"], dtype=float)
        dl_sub_err = np.asarray(alt["dl_sub_err"], dtype=float)
        ms = np.isfinite(lb_alt) & np.isfinite(dl_sub) & np.isfinite(dl_sub_err) & (lb_alt > 0)
        if np.any(ms):
            ax.errorbar(lb_alt[ms], dl_sub[ms], yerr=dl_sub_err[ms], fmt="o", color=color, markersize=2.5, capsize=1.8, alpha=0.8, label="Shot-subtracted auto")

        d2h = float(np.asarray(alt["dl_2h_level"]))
        d2h_err = float(np.asarray(alt["dl_2h_level_err"]))
        if np.isfinite(d2h):
            ax.axhline(d2h, color=color, linewidth=2.0, linestyle="--", label=f"2h level = {d2h:.2e}")
            if np.isfinite(d2h_err):
                ax.axhspan(max(d2h - d2h_err, 1.0e-12), d2h + d2h_err, color=color, alpha=0.15)

        ax.axvspan(shot_ell_min, shot_ell_max, color="grey", alpha=0.15)
        ax.axvline(ell_2h_max, color="grey", linestyle=":", linewidth=1.2)

        ax.set_title(f"{label} galaxy auto (TM1)", fontsize=11)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim([250, 1.05e5])
        ax.set_ylim([1e-4, 1e3])
        ax.grid(alpha=0.3)
        ax.set_xlabel("$\\ell$")

        cshot = float(np.asarray(alt["cl_shot"]))
        cshot_err = float(np.asarray(alt["cl_shot_err"]))
        ax.text(
            0.04,
            0.05,
            f"$C_{{shot}}={cshot:.2e}\\pm{cshot_err:.1e}$\n"
            f"$D_{{2h}}(\\ell<{int(ell_2h_max)})={d2h:.2e}\\pm{d2h_err:.1e}$",
            transform=ax.transAxes,
            fontsize=8,
        )

    axes[0].set_ylabel("$D_\\ell$")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "Alternate 2h strategy: subtract high-ell shot noise, then average $D_\\ell$ at low ell",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def run_zlt1_simple_model_diagnostics(args: argparse.Namespace) -> List[GeneratedFigure]:
    """Run z<1 simplified fits and render diagnostics for a list of ell_max values."""

    from ciber.plotting.gal_plotting_fns import collect_ciber_gal_vs_redshift
    from ciber.theory.cross_ps_parametric_model import (
        load_fit_results_npz,
        run_gal_auto_fits,
        run_gal_cross_fits,
    )

    inst_list = [1, 2]
    ifield_list = [4, 5, 6, 7, 8]
    ifield_list_hsc = [8]
    zbinedges = [0.0, 1.0]

    # Data for plotting fits.
    res_ls = collect_ciber_gal_vs_redshift(
        "LS",
        subtract_randoms=True,
        inst_list=inst_list,
        zbinedges=zbinedges,
        maskstr="JHlt16_wFFerr",
        subtract_sn=False,
        tl_pix_correct=True,
        ifield_list=ifield_list,
    )
    res_hsc = collect_ciber_gal_vs_redshift(
        "HSC",
        subtract_randoms=True,
        inst_list=inst_list,
        zbinedges=zbinedges,
        maskstr=None,
        subtract_sn=False,
        tl_pix_correct=True,
        ifield_list=ifield_list_hsc,
        with_ff_err=True,
        headstr=args.hsc_headstr,
    )

    generated: List[GeneratedFigure] = []
    for lmax in args.ell_max_list:
        fit_tag = f"{args.fit_tag}_lmax{int(lmax)}"
        print(
            "Running simplified z<1 diagnostics with "
            f"ell_max={lmax}, fit_tag={fit_tag}"
        )

        ls_cross_file = f"ciber_cl_fits_DESILS_coarsez_{fit_tag}.npz"
        hsc_cross_file = f"ciber_cl_fits_HSC_coarsez_{fit_tag}.npz"
        ls_auto_file = f"gal_auto_fits_DESILS_coarsez_{fit_tag}.npz"
        hsc_auto_file = f"gal_auto_fits_HSC_coarsez_{fit_tag}.npz"

        run_gal_cross_fits(
            inst_list=inst_list,
            ifield_list=ifield_list,
            cat="DESILS",
            zbinedges=zbinedges,
            maskstr="JHlt16_wFFerr",
            chi2_eval_max=float(lmax),
            lMax_fit=float(lmax),
            use_ihl_templates=False,
            use_one_halo=False,
            use_astrometry_damping=True,
            save_figs=not args.no_save_intermediate_fits,
            figbasedir=str(REPO_ROOT / "figures" / "generated_gal_cross_10deg" / f"zlt1_simple_lmax{int(lmax)}" / "cross_desils") + os.sep,
            save_results=True,
            file_fpath=ls_cross_file,
            fitstr=fit_tag,
            nwalkers=int(args.fit_nwalkers),
            nsteps=int(args.fit_nsteps),
            nburn=int(args.fit_nburn),
        )
        run_gal_cross_fits(
            inst_list=inst_list,
            ifield_list=ifield_list_hsc,
            cat="HSC",
            zbinedges=zbinedges,
            maskstr=None,
            headstr=args.hsc_headstr,
            chi2_eval_max=float(lmax),
            lMax_fit=float(lmax),
            use_ihl_templates=False,
            use_one_halo=False,
            use_astrometry_damping=True,
            save_figs=not args.no_save_intermediate_fits,
            figbasedir=str(REPO_ROOT / "figures" / "generated_gal_cross_10deg" / f"zlt1_simple_lmax{int(lmax)}" / "cross_hsc") + os.sep,
            save_results=True,
            file_fpath=hsc_cross_file,
            fitstr=fit_tag,
            nwalkers=int(args.fit_nwalkers),
            nsteps=int(args.fit_nsteps),
            nburn=int(args.fit_nburn),
        )

        run_gal_auto_fits(
            inst_list=[1],
            cat="DESILS",
            zbinedges=zbinedges,
            headstr=None,
            ifield_list=ifield_list,
            chi2_eval_max=float(lmax),
            lMax_fit=float(lmax),
            save_figs=not args.no_save_intermediate_fits,
            figbasedir=str(REPO_ROOT / "figures" / "generated_gal_cross_10deg" / f"zlt1_simple_lmax{int(lmax)}" / "auto_desils") + os.sep,
            save_results=True,
            file_fpath=ls_auto_file,
            fitstr=fit_tag,
            use_astrometry_damping=False,
            use_one_halo=False,
            nwalkers=int(args.fit_nwalkers),
            nsteps=int(args.fit_nsteps),
            nburn=int(args.fit_nburn),
        )
        run_gal_auto_fits(
            inst_list=[1],
            cat="HSC",
            zbinedges=zbinedges,
            headstr=args.hsc_headstr,
            ifield_list=ifield_list_hsc,
            chi2_eval_max=float(lmax),
            lMax_fit=float(lmax),
            save_figs=not args.no_save_intermediate_fits,
            figbasedir=str(REPO_ROOT / "figures" / "generated_gal_cross_10deg" / f"zlt1_simple_lmax{int(lmax)}" / "auto_hsc") + os.sep,
            save_results=True,
            file_fpath=hsc_auto_file,
            fitstr=fit_tag,
            use_astrometry_damping=False,
            use_one_halo=False,
            nwalkers=int(args.fit_nwalkers),
            nsteps=int(args.fit_nsteps),
            nburn=int(args.fit_nburn),
        )

        ls_cross_res = load_fit_results_npz(str(REPO_ROOT / "data" / "cross_cl_fits" / ls_cross_file))
        hsc_cross_res = load_fit_results_npz(str(REPO_ROOT / "data" / "cross_cl_fits" / hsc_cross_file))
        ls_auto_res = load_fit_results_npz(str(REPO_ROOT / "data" / "gal_auto_fits" / ls_auto_file))
        hsc_auto_res = load_fit_results_npz(str(REPO_ROOT / "data" / "gal_auto_fits" / hsc_auto_file))

        fig = _make_zlt1_simple_fit_figure(
            res_ls=res_ls,
            res_hsc=res_hsc,
            ls_cross_fit=ls_cross_res,
            hsc_cross_fit=hsc_cross_res,
            ls_auto_fit=ls_auto_res,
            hsc_auto_fit=hsc_auto_res,
            lmax=float(lmax),
            startidx=int(args.startidx),
            endidx=int(args.endidx),
        )

        generated.append(
            GeneratedFigure(
                f"zlt1-simple-model-diagnostics:lmax{int(lmax)}",
                fig,
                f"zlt1_simple_fit_diagnostics_lmax{int(lmax)}",
            )
        )

        if args.alt_auto_2h_strategy:
            lb_ls = np.asarray(res_ls["lb"], dtype=float)[int(args.startidx):int(args.endidx)]
            cl_ls = np.asarray(res_ls["full_cl_gal"][0, 0], dtype=float)[int(args.startidx):int(args.endidx)]
            clerr_ls = np.asarray(res_ls["full_clerr_gal"][0, 0], dtype=float)[int(args.startidx):int(args.endidx)]

            lb_hsc = np.asarray(res_hsc["lb"], dtype=float)[int(args.startidx):int(args.endidx)]
            cl_hsc = np.asarray(res_hsc["full_cl_gal"][0, 0], dtype=float)[int(args.startidx):int(args.endidx)]
            clerr_hsc = np.asarray(res_hsc["full_clerr_gal"][0, 0], dtype=float)[int(args.startidx):int(args.endidx)]

            ls_alt = _estimate_auto_2h_from_shot_subtracted_cl(
                lb_ls,
                cl_ls,
                clerr_ls,
                ell_2h_max=float(args.alt_2h_ell_max),
                shot_ell_min=float(args.alt_shot_ell_min),
                shot_ell_max=float(args.alt_shot_ell_max),
            )
            hsc_alt = _estimate_auto_2h_from_shot_subtracted_cl(
                lb_hsc,
                cl_hsc,
                clerr_hsc,
                ell_2h_max=float(args.alt_2h_ell_max),
                shot_ell_min=float(args.alt_shot_ell_min),
                shot_ell_max=float(args.alt_shot_ell_max),
            )

            print(
                "[ALT auto 2h] "
                f"ell_max={int(lmax)} LS: Cshot={float(ls_alt['cl_shot']):.3e}, D2h={float(ls_alt['dl_2h_level']):.3e}; "
                f"HSC: Cshot={float(hsc_alt['cl_shot']):.3e}, D2h={float(hsc_alt['dl_2h_level']):.3e}"
            )

            alt_fig = _make_zlt1_alt_auto_2h_figure(
                res_ls=res_ls,
                res_hsc=res_hsc,
                ls_alt=ls_alt,
                hsc_alt=hsc_alt,
                ell_2h_max=float(args.alt_2h_ell_max),
                shot_ell_min=float(args.alt_shot_ell_min),
                shot_ell_max=float(args.alt_shot_ell_max),
                startidx=int(args.startidx),
                endidx=int(args.endidx),
            )

            generated.append(
                GeneratedFigure(
                    f"zlt1-simple-model-diagnostics:alt-auto2h:lmax{int(lmax)}",
                    alt_fig,
                    f"zlt1_alt_auto2h_shotsub_diagnostics_lmax{int(lmax)}",
                )
            )

            alt_outdir = REPO_ROOT / "data" / "ciber_auto_predictions"
            alt_outdir.mkdir(parents=True, exist_ok=True)
            np.savez(
                alt_outdir / f"zlt1_alt_auto2h_levels_{fit_tag}.npz",
                ell_max=float(lmax),
                alt_2h_ell_max=float(args.alt_2h_ell_max),
                alt_shot_ell_min=float(args.alt_shot_ell_min),
                alt_shot_ell_max=float(args.alt_shot_ell_max),
                ls_cl_shot=float(ls_alt["cl_shot"]),
                ls_cl_shot_err=float(ls_alt["cl_shot_err"]),
                ls_dl_2h=float(ls_alt["dl_2h_level"]),
                ls_dl_2h_err=float(ls_alt["dl_2h_level_err"]),
                hsc_cl_shot=float(hsc_alt["cl_shot"]),
                hsc_cl_shot_err=float(hsc_alt["cl_shot_err"]),
                hsc_dl_2h=float(hsc_alt["dl_2h_level"]),
                hsc_dl_2h_err=float(hsc_alt["dl_2h_level_err"]),
            )

    return generated


def run_zlt1_alt2h_prediction(args: argparse.Namespace) -> List[GeneratedFigure]:
    """Predict z<1 CIBER auto using cross 2h amplitudes and alternate 2h gg levels.

    For each ell_max and each instrument/catalog:
      D_II,pred = A_2h,cross^2 / D_2h,gg
    where D_2h,gg is from the shot-subtracted low-ell estimator.
    """

    from ciber.theory.cross_ps_parametric_model import load_fit_results_npz

    generated: List[GeneratedFigure] = []
    pred_outdir = REPO_ROOT / "data" / "ciber_auto_predictions"
    pred_outdir.mkdir(parents=True, exist_ok=True)

    for lmax in args.ell_max_list:
        fit_tag = f"{args.fit_tag}_lmax{int(lmax)}"

        ls_cross_path = REPO_ROOT / "data" / "cross_cl_fits" / f"ciber_cl_fits_DESILS_coarsez_{fit_tag}.npz"
        hsc_cross_path = REPO_ROOT / "data" / "cross_cl_fits" / f"ciber_cl_fits_HSC_coarsez_{fit_tag}.npz"
        alt_level_path = pred_outdir / f"zlt1_alt_auto2h_levels_{fit_tag}.npz"

        if not ls_cross_path.exists() or not hsc_cross_path.exists() or not alt_level_path.exists():
            raise FileNotFoundError(
                "Missing required inputs for alt2h prediction at "
                f"ell_max={int(lmax)}. Need:\n"
                f"  {ls_cross_path}\n"
                f"  {hsc_cross_path}\n"
                f"  {alt_level_path}"
            )

        ls_cross_res = load_fit_results_npz(str(ls_cross_path))
        hsc_cross_res = load_fit_results_npz(str(hsc_cross_path))
        alt = np.load(str(alt_level_path))

        g2h_ls = float(alt["ls_dl_2h"])
        g2h_ls_err = float(alt["ls_dl_2h_err"])
        g2h_hsc = float(alt["hsc_dl_2h"])
        g2h_hsc_err = float(alt["hsc_dl_2h_err"])

        params_ls = np.asarray(ls_cross_res["params"], dtype=float)
        perr_ls = np.asarray(ls_cross_res["params_err"], dtype=float)
        params_hsc = np.asarray(hsc_cross_res["params"], dtype=float)
        perr_hsc = np.asarray(hsc_cross_res["params_err"], dtype=float)

        lb_common: Optional[np.ndarray] = None
        dl_meas_list: List[np.ndarray] = []
        dl_meas_err_list: List[np.ndarray] = []
        dl_pred_ls_list: List[np.ndarray] = []
        dl_pred_ls_err_list: List[np.ndarray] = []
        dl_pred_hsc_list: List[np.ndarray] = []
        dl_pred_hsc_err_list: List[np.ndarray] = []

        for inst_idx in range(2):
            lb_meas, dl_m, dl_merr = _load_measured_ciber_auto_dl_exact(inst_idx)
            if lb_common is None:
                lb_common = lb_meas

            a_ls = float(params_ls[inst_idx, 0, 0])
            a_ls_err = float(perr_ls[inst_idx, 0, 0])
            a_hsc = float(params_hsc[inst_idx, 0, 0])
            a_hsc_err = float(perr_hsc[inst_idx, 0, 0])

            d_pred_ls = (a_ls ** 2) / g2h_ls if np.isfinite(g2h_ls) and g2h_ls != 0 else np.nan
            d_pred_hsc = (a_hsc ** 2) / g2h_hsc if np.isfinite(g2h_hsc) and g2h_hsc != 0 else np.nan

            if np.isfinite(d_pred_ls) and a_ls != 0 and np.isfinite(g2h_ls_err):
                rel_ls = np.sqrt((2.0 * a_ls_err / abs(a_ls)) ** 2 + (g2h_ls_err / abs(g2h_ls)) ** 2)
                d_pred_ls_err = abs(d_pred_ls) * rel_ls
            else:
                d_pred_ls_err = np.nan

            if np.isfinite(d_pred_hsc) and a_hsc != 0 and np.isfinite(g2h_hsc_err):
                rel_hsc = np.sqrt((2.0 * a_hsc_err / abs(a_hsc)) ** 2 + (g2h_hsc_err / abs(g2h_hsc)) ** 2)
                d_pred_hsc_err = abs(d_pred_hsc) * rel_hsc
            else:
                d_pred_hsc_err = np.nan

            print(
                f"[ALT2H pred] ell_max={int(lmax)} TM{inst_idx+1}: "
                f"LS A2h={a_ls:.3e} -> DII={d_pred_ls:.3e}; "
                f"HSC A2h={a_hsc:.3e} -> DII={d_pred_hsc:.3e}"
            )

            dl_meas_list.append(dl_m)
            dl_meas_err_list.append(dl_merr)
            dl_pred_ls_list.append(np.full_like(lb_meas, d_pred_ls, dtype=float))
            dl_pred_ls_err_list.append(np.full_like(lb_meas, d_pred_ls_err, dtype=float))
            dl_pred_hsc_list.append(np.full_like(lb_meas, d_pred_hsc, dtype=float))
            dl_pred_hsc_err_list.append(np.full_like(lb_meas, d_pred_hsc_err, dtype=float))

        dl_meas = np.asarray(dl_meas_list)
        dl_meas_err = np.asarray(dl_meas_err_list)
        dl_pred_ls = np.asarray(dl_pred_ls_list)
        dl_pred_ls_err = np.asarray(dl_pred_ls_err_list)
        dl_pred_hsc = np.asarray(dl_pred_hsc_list)
        dl_pred_hsc_err = np.asarray(dl_pred_hsc_err_list)

        out_npz = pred_outdir / f"ciber_auto_pred_from_zlt1_alt2h_{fit_tag}.npz"
        np.savez(
            out_npz,
            ell_max=float(lmax),
            lb=lb_common,
            dl_ciber_auto_measured=dl_meas,
            dl_ciber_auto_measured_err=dl_meas_err,
            dl_ciber_auto_pred_ls=dl_pred_ls,
            dl_ciber_auto_pred_ls_err=dl_pred_ls_err,
            dl_ciber_auto_pred_hsc=dl_pred_hsc,
            dl_ciber_auto_pred_hsc_err=dl_pred_hsc_err,
            ls_cross_fit_file=str(ls_cross_path),
            hsc_cross_fit_file=str(hsc_cross_path),
            alt_auto2h_level_file=str(alt_level_path),
        )
        print(f"Saved ALT2H prediction product -> {out_npz}")

        fig = _plot_zlt1_auto_prediction_comparison(
            lb=np.asarray(lb_common, dtype=float),
            dl_meas=dl_meas,
            dl_meas_err=dl_meas_err,
            dl_pred_ls=dl_pred_ls,
            dl_pred_ls_err=dl_pred_ls_err,
            dl_pred_hsc=dl_pred_hsc,
            dl_pred_hsc_err=dl_pred_hsc_err,
        )

        generated.append(
            GeneratedFigure(
                f"zlt1-alt2h-prediction:lmax{int(lmax)}",
                fig,
                f"ciber_auto_vs_predicted_from_zlt1_alt2h_lmax{int(lmax)}",
            )
        )

    return generated


def _cross_component_bands(
    lb: np.ndarray,
    params: np.ndarray,
    params_err: np.ndarray,
    nsamp: int = 300,
    seed: int = 123,
) -> Dict[str, np.ndarray]:
    """Compute component and total uncertainty bands from parameter draws."""
    from ciber.theory.cross_ps_parametric_model import CrossPowerSpectrumModel

    lb = np.asarray(lb, dtype=float)
    p = np.asarray(params, dtype=float)
    pe = np.asarray(params_err, dtype=float)

    model = CrossPowerSpectrumModel(
        lb,
        use_powerlaw_2h=True,
        alpha_2h_fixed=0.0,
        use_lorentzian_1h=False,
        use_astrometry_damping=(p.size >= 6),
        use_one_halo=True,
    )

    comp0 = model.model_components(lb, *p)
    total0 = model.model_dl(lb, *p) if p.size == 5 else model.model_dl(lb, *p[:5], sigma_damp=p[5])

    rng = np.random.default_rng(seed)
    sig = np.where(np.isfinite(pe), pe, 0.0)
    draws = rng.normal(loc=p, scale=sig, size=(nsamp, p.size))
    draws[:, 0] = np.clip(draws[:, 0], 0.0, None)   # A_2h
    draws[:, 1] = np.clip(draws[:, 1], 0.0, None)   # A_1h
    draws[:, 3] = np.clip(draws[:, 3], 1.0e-3, None)  # sigma_1h
    draws[:, 4] = np.clip(draws[:, 4], 0.0, None)   # A_shot
    if p.size >= 6:
        draws[:, 5] = np.clip(draws[:, 5], 0.1, None)  # sigma_damp

    c2h = np.zeros((nsamp, lb.size), dtype=float)
    c1h = np.zeros((nsamp, lb.size), dtype=float)
    cshot = np.zeros((nsamp, lb.size), dtype=float)
    ctot = np.zeros((nsamp, lb.size), dtype=float)
    for i in range(nsamp):
        pi = draws[i]
        comp = model.model_components(lb, *pi)
        c2h[i] = np.asarray(comp["two_halo"], dtype=float)
        c1h[i] = np.asarray(comp["one_halo"], dtype=float)
        cshot[i] = np.asarray(comp["shot_noise"], dtype=float)
        if p.size >= 6:
            ctot[i] = model.model_dl(lb, *pi[:5], sigma_damp=pi[5])
        else:
            ctot[i] = model.model_dl(lb, *pi)

    def _p16_84(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.percentile(arr, 16.0, axis=0), np.percentile(arr, 84.0, axis=0)

    c2h_lo, c2h_hi = _p16_84(c2h)
    c1h_lo, c1h_hi = _p16_84(c1h)
    cshot_lo, cshot_hi = _p16_84(cshot)
    ctot_lo, ctot_hi = _p16_84(ctot)

    return {
        "two_halo": np.asarray(comp0["two_halo"], dtype=float),
        "one_halo": np.asarray(comp0["one_halo"], dtype=float),
        "shot_noise": np.asarray(comp0["shot_noise"], dtype=float),
        "total": np.asarray(total0, dtype=float),
        "two_halo_lo": c2h_lo,
        "two_halo_hi": c2h_hi,
        "one_halo_lo": c1h_lo,
        "one_halo_hi": c1h_hi,
        "shot_noise_lo": cshot_lo,
        "shot_noise_hi": cshot_hi,
        "total_lo": ctot_lo,
        "total_hi": ctot_hi,
    }


def _make_zlt1_cross_1h_component_diagnostic(
    res_ls: Dict[str, Any],
    res_hsc: Dict[str, Any],
    ls_cross_fit: Dict[str, Any],
    hsc_cross_fit: Dict[str, Any],
    lmax: float,
    startidx: int,
    endidx: int,
    nsamp_unc: int = 300,
    show_lsq_init: bool = True,
    lsq_initial_guess: Optional[np.ndarray] = None,
    lsq_bounds: Optional[np.ndarray] = None,
    fixed_mu_1h: Optional[float] = None,
    fixed_sigma_1h: Optional[float] = None,
) -> Any:
    from matplotlib import pyplot as plt
    from scipy.optimize import curve_fit
    from ciber.theory.cross_ps_parametric_model import CrossPowerSpectrumModel

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), sharex=True, sharey=True)
    panels = [
        ("DESI-LS TM1", res_ls, ls_cross_fit, 0),
        ("DESI-LS TM2", res_ls, ls_cross_fit, 1),
        ("HSC TM1", res_hsc, hsc_cross_fit, 0),
        ("HSC TM2", res_hsc, hsc_cross_fit, 1),
    ]

    for ax, (title, res, fitres, inst_idx) in zip(axes.ravel(), panels):
        lb = np.asarray(res["lb"], dtype=float)[startidx:endidx]
        pf = lb * (lb + 1.0) / (2.0 * np.pi)
        dl_data = pf * np.asarray(res["full_cl_cross"][inst_idx, 0], dtype=float)[startidx:endidx]
        dl_err = pf * np.asarray(res["full_clerr_cross"][inst_idx, 0], dtype=float)[startidx:endidx]

        params = np.asarray(fitres["params"][inst_idx, 0], dtype=float)
        params_err = np.asarray(fitres["params_err"][inst_idx, 0], dtype=float)
        bands = _cross_component_bands(lb, params, params_err, nsamp=nsamp_unc, seed=100 + inst_idx)

        good = np.isfinite(lb) & np.isfinite(dl_data) & np.isfinite(dl_err) & (lb > 0) & (dl_data > 0)
        if np.any(good):
            ax.errorbar(lb[good], dl_data[good], yerr=dl_err[good], fmt="o", color="k", markersize=2.8, capsize=1.6, alpha=0.85, label="data")

        comp_styles = [
            ("two_halo", "C0", "2h"),
            ("one_halo", "C2", "1h"),
            ("shot_noise", "C4", "shot"),
            ("total", "C3", "total"),
        ]
        for key, color, label in comp_styles:
            y = np.asarray(bands[key], dtype=float)
            ylo = np.asarray(bands[f"{key}_lo"], dtype=float)
            yhi = np.asarray(bands[f"{key}_hi"], dtype=float)
            m = np.isfinite(lb) & np.isfinite(y) & np.isfinite(ylo) & np.isfinite(yhi) & (lb > 0) & (y > 0)
            if np.any(m):
                lw = 1.8 if key == "total" else 1.3
                ax.plot(lb[m], y[m], color=color, linewidth=lw, label=label)
                ax.fill_between(lb[m], np.clip(ylo[m], 1e-12, None), np.clip(yhi[m], 1e-12, None), color=color, alpha=0.14)

        if show_lsq_init and fixed_mu_1h is not None and fixed_sigma_1h is not None:
            try:
                model_lsq = CrossPowerSpectrumModel(
                    lb,
                    use_powerlaw_2h=True,
                    alpha_2h_fixed=0.0,
                    use_lorentzian_1h=False,
                    mu_1h_fixed=float(fixed_mu_1h),
                    sigma_1h_fixed=float(fixed_sigma_1h),
                    use_astrometry_damping=True,
                    use_one_halo=True,
                )

                fit_mask = (
                    np.isfinite(lb)
                    & np.isfinite(dl_data)
                    & np.isfinite(dl_err)
                    & (lb >= 300.0)
                    & (lb <= float(lmax))
                    & (dl_err > 0)
                )
                if np.any(fit_mask):
                    lb_fit = lb[fit_mask]
                    dl_fit = dl_data[fit_mask]
                    de_fit = dl_err[fit_mask]

                    def _model4(ell_vals: np.ndarray, a2h: float, a1h: float, ashot: float, sdamp: float) -> np.ndarray:
                        return model_lsq.model_dl(
                            ell_vals,
                            float(a2h),
                            float(a1h),
                            float(fixed_mu_1h),
                            float(fixed_sigma_1h),
                            float(ashot),
                            sigma_damp=float(sdamp),
                        )

                    p0 = np.array([0.1, 1.0, 0.5, 2.0], dtype=float)
                    if lsq_initial_guess is not None and np.asarray(lsq_initial_guess).size == 4:
                        p0 = np.asarray(lsq_initial_guess, dtype=float)

                    bounds = (
                        np.array([0.0, 0.0, 0.0, 0.1], dtype=float),
                        np.array([10.0, 100.0, 10.0, 7.0], dtype=float),
                    )
                    if lsq_bounds is not None and np.asarray(lsq_bounds).shape == (2, 4):
                        bounds = (np.asarray(lsq_bounds[0], dtype=float), np.asarray(lsq_bounds[1], dtype=float))

                    p_lsq, _ = curve_fit(
                        _model4,
                        lb_fit,
                        dl_fit,
                        p0=p0,
                        sigma=de_fit,
                        absolute_sigma=True,
                        bounds=bounds,
                        maxfev=20000,
                    )
                    dl_lsq = _model4(lb, *p_lsq)
                    m_lsq = np.isfinite(lb) & np.isfinite(dl_lsq) & (lb > 0) & (dl_lsq > 0)
                    if np.any(m_lsq):
                        ax.plot(
                            lb[m_lsq],
                            dl_lsq[m_lsq],
                            color="C1",
                            linewidth=1.4,
                            linestyle=":",
                            label="LSQ init",
                        )
            except Exception:
                pass

        rchi2 = float(np.asarray(fitres["reduced_chisq"])[inst_idx, 0])
        chisq = float(np.asarray(fitres["chisq"])[inst_idx, 0])
        ndof = chisq / rchi2 if np.isfinite(rchi2) and rchi2 > 0 else np.nan

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim([250, 1.05e5])
        ax.set_ylim([1e-3, 1e3])
        ax.grid(alpha=0.25)
        ax.set_title(title, fontsize=11)
        ax.text(0.04, 0.05, f"$\\chi^2/N_{{dof}}={chisq:.1f}/{ndof:.0f}={rchi2:.2f}$", transform=ax.transAxes, fontsize=8)
        ax.set_xlabel("$\\ell$")
        ax.set_ylabel("$D_\\ell$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=9, frameon=False)
    fig.suptitle(f"z<1 cross fits with 1h term + damping, component uncertainty bands ($\\ell_{{max}}$={int(lmax)})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def _make_zlt1_gal_auto_component_diagnostic(
    res_ls: Dict[str, Any],
    res_hsc: Dict[str, Any],
    ls_auto_fit: Dict[str, Any],
    hsc_auto_fit: Dict[str, Any],
    lmax: float,
    startidx: int,
    endidx: int,
    nsamp_unc: int = 300,
    fixed_mu_1h: Optional[float] = None,
    fixed_sigma_1h: Optional[float] = None,
) -> Any:
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9), sharex=True, sharey=True)
    panels = [
        ("DESI-LS auto TM1", res_ls, ls_auto_fit, 0),
        ("HSC auto TM1", res_hsc, hsc_auto_fit, 1),
    ]

    for ax, (title, res, fitres, seed_offset) in zip(axes.ravel(), panels):
        lb = np.asarray(res["lb"], dtype=float)[startidx:endidx]
        pf = lb * (lb + 1.0) / (2.0 * np.pi)
        dl_data = pf * np.asarray(res["full_cl_gal"][0, 0], dtype=float)[startidx:endidx]
        dl_err = pf * np.asarray(res["full_clerr_gal"][0, 0], dtype=float)[startidx:endidx]

        params = np.asarray(fitres["params"][0, 0], dtype=float)
        params_err = np.asarray(fitres["params_err"][0, 0], dtype=float)

        # Auto fits may be saved in compact [A_2h, A_1h, A_shot] form when 1h shape is fixed.
        if params.size == 3:
            mu_use = float(fixed_mu_1h) if fixed_mu_1h is not None else 8.0
            sigma_use = float(fixed_sigma_1h) if fixed_sigma_1h is not None else 0.7
            params = np.array([params[0], params[1], mu_use, sigma_use, params[2]], dtype=float)

            if params_err.size == 3:
                params_err = np.array([params_err[0], params_err[1], 0.0, 0.0, params_err[2]], dtype=float)

        bands = _cross_component_bands(lb, params, params_err, nsamp=nsamp_unc, seed=700 + seed_offset)

        good = np.isfinite(lb) & np.isfinite(dl_data) & np.isfinite(dl_err) & (lb > 0) & (dl_data > 0)
        if np.any(good):
            ax.errorbar(lb[good], dl_data[good], yerr=dl_err[good], fmt="o", color="k", markersize=2.8, capsize=1.6, alpha=0.85, label="data")

        comp_styles = [
            ("two_halo", "C0", "2h"),
            ("one_halo", "C2", "1h"),
            ("shot_noise", "C4", "shot"),
            ("total", "C3", "total"),
        ]
        for key, color, label in comp_styles:
            y = np.asarray(bands[key], dtype=float)
            ylo = np.asarray(bands[f"{key}_lo"], dtype=float)
            yhi = np.asarray(bands[f"{key}_hi"], dtype=float)
            m = np.isfinite(lb) & np.isfinite(y) & np.isfinite(ylo) & np.isfinite(yhi) & (lb > 0) & (y > 0)
            if np.any(m):
                lw = 1.8 if key == "total" else 1.3
                ax.plot(lb[m], y[m], color=color, linewidth=lw, label=label)
                ax.fill_between(lb[m], np.clip(ylo[m], 1e-12, None), np.clip(yhi[m], 1e-12, None), color=color, alpha=0.14)

        rchi2 = float(np.asarray(fitres["reduced_chisq"])[0, 0])
        chisq = float(np.asarray(fitres["chisq"])[0, 0])
        ndof = chisq / rchi2 if np.isfinite(rchi2) and rchi2 > 0 else np.nan

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim([250, 1.05e5])
        ax.set_ylim([1e-4, 1e3])
        ax.grid(alpha=0.25)
        ax.set_title(title, fontsize=11)
        ax.text(0.04, 0.05, f"$\\chi^2/N_{{dof}}={chisq:.1f}/{ndof:.0f}={rchi2:.2f}$", transform=ax.transAxes, fontsize=8)
        ax.set_xlabel("$\\ell$")
        ax.set_ylabel("$D_\\ell$")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=9, frameon=False)
    fig.suptitle(f"z<1 galaxy-auto fits (2h + 1h + shot), component uncertainty bands ($\\ell_{{max}}$={int(lmax)})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return fig


def run_zlt1_cross_with1h_update_prediction(args: argparse.Namespace) -> List[GeneratedFigure]:
    """Refit z<1 cross spectra with 1h+damping, then conditionally update auto predictions."""
    from ciber.plotting.gal_plotting_fns import collect_ciber_gal_vs_redshift
    from ciber.theory.cross_ps_parametric_model import load_fit_results_npz, run_gal_auto_fits, run_gal_cross_fits

    inst_list = [1, 2]
    ifield_list = [4, 5, 6, 7, 8]
    ifield_list_hsc = [8]
    zbinedges = [0.0, 1.0]

    res_ls = collect_ciber_gal_vs_redshift(
        "LS",
        subtract_randoms=True,
        inst_list=inst_list,
        zbinedges=zbinedges,
        maskstr="JHlt16_wFFerr",
        subtract_sn=False,
        tl_pix_correct=True,
        ifield_list=ifield_list,
    )
    res_hsc = collect_ciber_gal_vs_redshift(
        "HSC",
        subtract_randoms=True,
        inst_list=inst_list,
        zbinedges=zbinedges,
        maskstr=None,
        subtract_sn=False,
        tl_pix_correct=True,
        ifield_list=ifield_list_hsc,
        with_ff_err=True,
        headstr=args.hsc_headstr,
    )

    generated: List[GeneratedFigure] = []
    pred_outdir = REPO_ROOT / "data" / "ciber_auto_predictions"
    pred_outdir.mkdir(parents=True, exist_ok=True)

    fixed_mu_1h: Optional[float] = None
    fixed_sigma_1h: Optional[float] = None
    if args.one_halo_template_mode != "legacy":
        if not args.slice_template_fit_files:
            raise ValueError(
                "one-halo-template-mode requires --slice-template-fit-files when mode is not legacy"
            )
        fixed_mu_1h, fixed_sigma_1h, n_used = _derive_fixed_one_halo_from_slice_fits(
            [str(Path(p).expanduser()) for p in args.slice_template_fit_files],
            mode=args.one_halo_template_mode,
            effective_z=float(args.one_halo_effective_z),
            z_min=float(args.one_halo_z_min),
            z_max=float(args.one_halo_z_max),
        )
        print(
            "Derived fixed one-halo template for z<1 cross refit: "
            f"mode={args.one_halo_template_mode}, "
            f"mu_1h={fixed_mu_1h:.4f}, sigma_1h={fixed_sigma_1h:.4f}, n_used={n_used}"
        )

    cross_fix_1h_shape = fixed_mu_1h is not None and fixed_sigma_1h is not None
    if cross_fix_1h_shape:
        cross_prior_bounds = np.array([
            [0.0, 0.0, 0.0, 0.1],
            [10.0, 100.0, 10.0, 7.0],
        ], dtype=float)
        cross_initial_guess = np.array([0.1, 1.0, 0.5, 2.0], dtype=float)
        auto_prior_bounds = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 100.0, 10.0],
        ], dtype=float)
        print(
            "Cross model (fixed-shape 1h): D_ell = 2h + A*1h + shot, all x damping"
        )
    else:
        cross_prior_bounds = np.array([
            [0.0, 0.0, 6.5, 0.2, 0.0, 0.1],
            [10.0, 100.0, 9.5, 1.2, 10.0, 7.0],
        ], dtype=float)
        cross_initial_guess = np.array([0.1, 1.0, 8.0, 0.7, 0.5, 2.0], dtype=float)
        auto_prior_bounds = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 100.0, 10.0],
        ], dtype=float)
        print(
            "Cross model (free-shape 1h fallback): D_ell = 2h + A*1h + shot, all x damping"
        )

    print("Cross-fit prior bounds used:")
    if cross_fix_1h_shape:
        print("  A_2h       in [0, 10]")
        print("  A_1h       in [0, 100]")
        print("  A_shot     in [0, 10]")
        print("  sigma_damp in [0.1, 7] arcsec")
    else:
        print("  A_2h       in [0, 10]")
        print("  A_1h       in [0, 100]")
        print("  mu_1h      in [6.5, 9.5]")
        print("  sigma_1h   in [0.2, 1.2]")
        print("  A_shot     in [0, 10]")
        print("  sigma_damp in [0.1, 7] arcsec")

    lb_diag = np.asarray(res_ls["lb"], dtype=float)[int(args.startidx):int(args.endidx)]
    ell_mask = np.isfinite(lb_diag) & (lb_diag >= 300.0) & (lb_diag <= float(np.max(args.ell_max_list)))
    if np.any(ell_mask):
        print(
            "Bandpower range after slicing and fit cut: "
            f"ell=[{float(np.min(lb_diag[ell_mask])):.1f}, {float(np.max(lb_diag[ell_mask])):.1f}], "
            f"n={int(np.sum(ell_mask))}"
        )

    # Enforce robust MCMC settings for cross refits to improve convergence behavior.
    fit_dim = 4 if cross_fix_1h_shape else 6
    min_walkers = max(2 * fit_dim, 32)
    nwalkers_fit = int(args.fit_nwalkers)
    nsteps_fit = int(args.fit_nsteps)
    nburn_fit = int(args.fit_nburn)

    if nwalkers_fit < min_walkers:
        print(
            f"Requested --fit-nwalkers={nwalkers_fit} is too small for {fit_dim}-parameter MCMC; "
            f"using {min_walkers}."
        )
        nwalkers_fit = min_walkers

    if nsteps_fit < 3000:
        print(
            f"Requested --fit-nsteps={nsteps_fit} is short for stable posteriors; "
            "using 3000."
        )
        nsteps_fit = 3000

    if nburn_fit < 1000:
        print(
            f"Requested --fit-nburn={nburn_fit} is short for stable burn-in removal; "
            "using 1000."
        )
        nburn_fit = 1000

    if nburn_fit >= nsteps_fit:
        nburn_fit = max(500, int(0.25 * nsteps_fit))
        if nburn_fit >= nsteps_fit:
            nburn_fit = nsteps_fit - 1
        print(
            "Adjusted burn-in to be below total chain length: "
            f"--fit-nburn={nburn_fit} for --fit-nsteps={nsteps_fit}."
        )

    print(
        "Cross1h MCMC settings used: "
        f"nwalkers={nwalkers_fit}, nsteps={nsteps_fit}, nburn={nburn_fit}, ndim={fit_dim}"
    )

    for lmax in args.ell_max_list:
        fit_tag = f"{args.fit_tag}_lmax{int(lmax)}"
        ls_cross_file = f"ciber_cl_fits_DESILS_coarsez_{fit_tag}.npz"
        hsc_cross_file = f"ciber_cl_fits_HSC_coarsez_{fit_tag}.npz"
        ls_auto_file = f"gal_auto_fits_DESILS_coarsez_{fit_tag}.npz"
        hsc_auto_file = f"gal_auto_fits_HSC_coarsez_{fit_tag}.npz"
        ls_cross_path = REPO_ROOT / "data" / "cross_cl_fits" / ls_cross_file
        hsc_cross_path = REPO_ROOT / "data" / "cross_cl_fits" / hsc_cross_file
        ls_auto_path = REPO_ROOT / "data" / "gal_auto_fits" / ls_auto_file
        hsc_auto_path = REPO_ROOT / "data" / "gal_auto_fits" / hsc_auto_file

        use_cache = bool(getattr(args, "reuse_fit_cache", True))

        # Refit cross with one-halo included.
        if use_cache and ls_cross_path.exists() and hsc_cross_path.exists():
            print(
                f"[cross1h] Reusing cached cross-fit outputs for ell_max={int(lmax)}: "
                f"{ls_cross_path.name}, {hsc_cross_path.name}"
            )
        else:
            run_gal_cross_fits(
                inst_list=inst_list,
                ifield_list=ifield_list,
                cat="DESILS",
                zbinedges=zbinedges,
                maskstr="JHlt16_wFFerr",
                chi2_eval_max=float(lmax),
                lMax_fit=float(lmax),
                use_ihl_templates=False,
                use_one_halo=True,
                fix_ihl_1h_shape=cross_fix_1h_shape,
                use_ihl_1h_params=False,
                mu_1h_fixed_override=fixed_mu_1h,
                sigma_1h_fixed_override=fixed_sigma_1h,
                use_astrometry_damping=True,
                prior_bounds=cross_prior_bounds,
                initial_guess=cross_initial_guess,
                save_figs=not args.no_save_intermediate_fits,
                figbasedir=str(REPO_ROOT / "figures" / "generated_gal_cross_10deg" / f"zlt1_cross1h_lmax{int(lmax)}" / "cross_desils") + os.sep,
                save_results=True,
                file_fpath=ls_cross_file,
                fitstr=fit_tag,
                nwalkers=nwalkers_fit,
                nsteps=nsteps_fit,
                nburn=nburn_fit,
            )
            run_gal_cross_fits(
                inst_list=inst_list,
                ifield_list=ifield_list_hsc,
                cat="HSC",
                zbinedges=zbinedges,
                maskstr=None,
                headstr=args.hsc_headstr,
                chi2_eval_max=float(lmax),
                lMax_fit=float(lmax),
                use_ihl_templates=False,
                use_one_halo=True,
                fix_ihl_1h_shape=cross_fix_1h_shape,
                use_ihl_1h_params=False,
                mu_1h_fixed_override=fixed_mu_1h,
                sigma_1h_fixed_override=fixed_sigma_1h,
                use_astrometry_damping=True,
                prior_bounds=cross_prior_bounds,
                initial_guess=cross_initial_guess,
                save_figs=not args.no_save_intermediate_fits,
                figbasedir=str(REPO_ROOT / "figures" / "generated_gal_cross_10deg" / f"zlt1_cross1h_lmax{int(lmax)}" / "cross_hsc") + os.sep,
                save_results=True,
                file_fpath=hsc_cross_file,
                fitstr=fit_tag,
                nwalkers=nwalkers_fit,
                nsteps=nsteps_fit,
                nburn=nburn_fit,
            )

        ls_cross_res = load_fit_results_npz(str(ls_cross_path))
        hsc_cross_res = load_fit_results_npz(str(hsc_cross_path))

        fig_diag = _make_zlt1_cross_1h_component_diagnostic(
            res_ls=res_ls,
            res_hsc=res_hsc,
            ls_cross_fit=ls_cross_res,
            hsc_cross_fit=hsc_cross_res,
            lmax=float(lmax),
            startidx=int(args.startidx),
            endidx=int(args.endidx),
            nsamp_unc=int(args.component_unc_samples),
            show_lsq_init=bool(args.show_lsq_init),
            lsq_initial_guess=cross_initial_guess if cross_fix_1h_shape else None,
            lsq_bounds=cross_prior_bounds if cross_fix_1h_shape else None,
            fixed_mu_1h=fixed_mu_1h,
            fixed_sigma_1h=fixed_sigma_1h,
        )
        generated.append(
            GeneratedFigure(
                f"zlt1-cross1h-diagnostics:lmax{int(lmax)}",
                fig_diag,
                f"zlt1_cross1h_component_diagnostics_lmax{int(lmax)}",
            )
        )

        # Fit-quality gate.
        rchi2_vals = np.array([
            float(np.asarray(ls_cross_res["reduced_chisq"])[0, 0]),
            float(np.asarray(ls_cross_res["reduced_chisq"])[1, 0]),
            float(np.asarray(hsc_cross_res["reduced_chisq"])[0, 0]),
            float(np.asarray(hsc_cross_res["reduced_chisq"])[1, 0]),
        ], dtype=float)
        good = np.all(np.isfinite(rchi2_vals)) and np.all(rchi2_vals <= float(args.good_fit_rchi2_max))

        if not good:
            print(
                f"[cross1h] ell_max={int(lmax)} failed quality gate: reduced-chi2={rchi2_vals} "
                f"> max {float(args.good_fit_rchi2_max):.2f}. Skipping auto prediction update."
            )
            continue

        ls_auto_res = None
        hsc_auto_res = None
        if args.gal_auto_denominator_mode == "smooth-model":
            if use_cache and ls_auto_path.exists() and hsc_auto_path.exists():
                print(
                    f"[cross1h] Reusing cached galaxy-auto fits for ell_max={int(lmax)}: "
                    f"{ls_auto_path.name}, {hsc_auto_path.name}"
                )
            else:
                run_gal_auto_fits(
                    inst_list=[1],
                    cat="DESILS",
                    zbinedges=zbinedges,
                    headstr=None,
                    ifield_list=ifield_list,
                    chi2_eval_max=float(lmax),
                    lMax_fit=float(lmax),
                    save_figs=not args.no_save_intermediate_fits,
                    figbasedir=str(REPO_ROOT / "figures" / "generated_gal_cross_10deg" / f"zlt1_cross1h_lmax{int(lmax)}" / "auto_desils") + os.sep,
                    save_results=True,
                    file_fpath=ls_auto_file,
                    fitstr=fit_tag,
                    prior_bounds=auto_prior_bounds,
                    use_astrometry_damping=False,
                    mu_1h_fixed_override=fixed_mu_1h,
                    sigma_1h_fixed_override=fixed_sigma_1h,
                    nwalkers=nwalkers_fit,
                    nsteps=nsteps_fit,
                    nburn=nburn_fit,
                )
                run_gal_auto_fits(
                    inst_list=[1],
                    cat="HSC",
                    zbinedges=zbinedges,
                    headstr=args.hsc_headstr,
                    ifield_list=ifield_list_hsc,
                    chi2_eval_max=float(lmax),
                    lMax_fit=float(lmax),
                    save_figs=not args.no_save_intermediate_fits,
                    figbasedir=str(REPO_ROOT / "figures" / "generated_gal_cross_10deg" / f"zlt1_cross1h_lmax{int(lmax)}" / "auto_hsc") + os.sep,
                    save_results=True,
                    file_fpath=hsc_auto_file,
                    fitstr=fit_tag,
                    prior_bounds=auto_prior_bounds,
                    use_astrometry_damping=False,
                    mu_1h_fixed_override=fixed_mu_1h,
                    sigma_1h_fixed_override=fixed_sigma_1h,
                    nwalkers=nwalkers_fit,
                    nsteps=nsteps_fit,
                    nburn=nburn_fit,
                )
            ls_auto_res = load_fit_results_npz(str(ls_auto_path))
            hsc_auto_res = load_fit_results_npz(str(hsc_auto_path))

            fig_auto_diag = _make_zlt1_gal_auto_component_diagnostic(
                res_ls=res_ls,
                res_hsc=res_hsc,
                ls_auto_fit=ls_auto_res,
                hsc_auto_fit=hsc_auto_res,
                lmax=float(lmax),
                startidx=int(args.startidx),
                endidx=int(args.endidx),
                nsamp_unc=int(args.component_unc_samples),
                fixed_mu_1h=fixed_mu_1h,
                fixed_sigma_1h=fixed_sigma_1h,
            )
            generated.append(
                GeneratedFigure(
                    f"zlt1-cross1h-gal-auto-diagnostics:lmax{int(lmax)}",
                    fig_auto_diag,
                    f"zlt1_gal_auto_component_diagnostics_lmax{int(lmax)}",
                )
            )

        params_ls = np.asarray(ls_cross_res["params"], dtype=float)
        perr_ls = np.asarray(ls_cross_res["params_err"], dtype=float)
        params_hsc = np.asarray(hsc_cross_res["params"], dtype=float)
        perr_hsc = np.asarray(hsc_cross_res["params_err"], dtype=float)

        formula_tag = (
            (
                "separate_2h1h_sum_crosssq_over_gg_components_model"
                if args.ratio_scaling_mode == "separate-2h-1h"
                else "per_ell_ratio_2h1h_crosssq_over_gg_model"
            )
            if args.gal_auto_denominator_mode == "smooth-model"
            else "per_ell_ratio_2h1h_crosssq_over_gg_sub"
        )

        lb_common: Optional[np.ndarray] = None
        dl_meas_list: List[np.ndarray] = []
        dl_meas_err_list: List[np.ndarray] = []
        dl_pred_ls_list: List[np.ndarray] = []
        dl_pred_ls_err_list: List[np.ndarray] = []
        dl_pred_hsc_list: List[np.ndarray] = []
        dl_pred_hsc_err_list: List[np.ndarray] = []

        for inst_idx in range(2):
            lb_meas, dl_m, dl_merr = _load_measured_ciber_auto_dl_exact(inst_idx)
            if lb_common is None:
                lb_common = lb_meas

            ls_cross_samples = _extract_fit_samples_cell(ls_cross_res, inst_idx, 0)
            hsc_cross_samples = _extract_fit_samples_cell(hsc_cross_res, inst_idx, 0)
            ls_auto_samples = _extract_fit_samples_cell(ls_auto_res, 0, 0) if ls_auto_res is not None else None
            hsc_auto_samples = _extract_fit_samples_cell(hsc_auto_res, 0, 0) if hsc_auto_res is not None else None

            ls_comp = _predict_auto_2h1h_from_cross_and_shot_subtracted_auto(
                lb_meas,
                params_ls[inst_idx, 0],
                perr_ls[inst_idx, 0],
                ls_cross_samples,
                np.asarray(res_ls["lb"], dtype=float),
                np.asarray(res_ls["full_cl_gal"][0, 0], dtype=float),
                np.asarray(res_ls["full_clerr_gal"][0, 0], dtype=float),
                ell_range_for_scaling=(float(args.scale_ell_min), float(args.scale_ell_max)),
                shot_ell_min=float(args.shot_ell_min),
                shot_ell_max=float(args.shot_ell_max),
                gal_auto_denominator_mode=str(args.gal_auto_denominator_mode),
                ratio_scaling_mode=str(args.ratio_scaling_mode),
                gal_auto_params=(np.asarray(ls_auto_res["params"], dtype=float)[0, 0] if ls_auto_res is not None else None),
                gal_auto_params_err=(np.asarray(ls_auto_res["params_err"], dtype=float)[0, 0] if ls_auto_res is not None else None),
                gal_auto_samples=ls_auto_samples,
                nsamp=int(args.pred_nsamp),
                seed=401 + inst_idx,
            )
            hsc_comp = _predict_auto_2h1h_from_cross_and_shot_subtracted_auto(
                lb_meas,
                params_hsc[inst_idx, 0],
                perr_hsc[inst_idx, 0],
                hsc_cross_samples,
                np.asarray(res_hsc["lb"], dtype=float),
                np.asarray(res_hsc["full_cl_gal"][0, 0], dtype=float),
                np.asarray(res_hsc["full_clerr_gal"][0, 0], dtype=float),
                ell_range_for_scaling=(float(args.scale_ell_min), float(args.scale_ell_max)),
                shot_ell_min=float(args.shot_ell_min),
                shot_ell_max=float(args.shot_ell_max),
                gal_auto_denominator_mode=str(args.gal_auto_denominator_mode),
                ratio_scaling_mode=str(args.ratio_scaling_mode),
                gal_auto_params=(np.asarray(hsc_auto_res["params"], dtype=float)[0, 0] if hsc_auto_res is not None else None),
                gal_auto_params_err=(np.asarray(hsc_auto_res["params_err"], dtype=float)[0, 0] if hsc_auto_res is not None else None),
                gal_auto_samples=hsc_auto_samples,
                nsamp=int(args.pred_nsamp),
                seed=501 + inst_idx,
            )

            dl_meas_list.append(dl_m)
            dl_meas_err_list.append(dl_merr)
            dl_pred_ls_list.append(np.asarray(ls_comp["dl_pred_2h1h"], dtype=float))
            dl_pred_ls_err_list.append(np.asarray(ls_comp["dl_pred_2h1h_err"], dtype=float))
            dl_pred_hsc_list.append(np.asarray(hsc_comp["dl_pred_2h1h"], dtype=float))
            dl_pred_hsc_err_list.append(np.asarray(hsc_comp["dl_pred_2h1h_err"], dtype=float))

        dl_meas = np.asarray(dl_meas_list)
        dl_meas_err = np.asarray(dl_meas_err_list)
        dl_pred_ls = np.asarray(dl_pred_ls_list)
        dl_pred_ls_err = np.asarray(dl_pred_ls_err_list)
        dl_pred_hsc = np.asarray(dl_pred_hsc_list)
        dl_pred_hsc_err = np.asarray(dl_pred_hsc_err_list)

        overlay_dgl_dl: Optional[np.ndarray] = None
        overlay_dgl_err: Optional[np.ndarray] = None
        if bool(getattr(args, "include_dgl_auto_constraints", False)):
            try:
                overlay_dgl_dl, overlay_dgl_err = _try_load_ciber_dgl_auto_constraints(
                    np.asarray(lb_common, dtype=float),
                    dgl_mode=str(getattr(args, "dgl_mode", "sfd_clean")),
                )
            except Exception as exc:
                print(f"Warning: failed to load DGL auto constraints for overlay: {exc}")
                overlay_dgl_dl, overlay_dgl_err = None, None

        overlay_shot_dl: Optional[np.ndarray] = None
        if bool(getattr(args, "include_shot_noise_fit", True)):
            overlay_shot_dl = np.zeros_like(dl_meas, dtype=float)
            for inst_idx in range(dl_meas.shape[0]):
                overlay_shot_dl[inst_idx] = _estimate_ciber_shot_noise_dl(
                    np.asarray(lb_common, dtype=float),
                    np.asarray(dl_meas[inst_idx], dtype=float),
                    ell_min=float(args.shot_ell_min),
                    ell_max=float(args.shot_ell_max),
                    lb_eval=np.asarray(lb_common, dtype=float),
                )

        overlay_mock_igl_dl: Optional[np.ndarray] = None
        mock_igl_file = str(getattr(args, "mock_igl_zlt1_file", "")).strip()
        if mock_igl_file:
            try:
                overlay_mock_igl_dl = _load_optional_mock_igl_overlay(
                    mock_igl_file,
                    np.asarray(lb_common, dtype=float),
                )
            except Exception as exc:
                print(f"Warning: failed to load mock IGL overlay from '{mock_igl_file}': {exc}")
                overlay_mock_igl_dl = None
        elif bool(getattr(args, "auto_mock_igl_zlt1", True)):
            try:
                overlay_mock_igl_dl = _try_load_default_mock_igl_zlt1(np.asarray(lb_common, dtype=float))
                if overlay_mock_igl_dl is None:
                    print("Warning: no default z<1 mock IGL files found; skipping mock IGL overlay")
            except Exception as exc:
                print(f"Warning: failed to auto-load default z<1 mock IGL overlay: {exc}")
                overlay_mock_igl_dl = None

        out_npz = pred_outdir / f"ciber_auto_pred_from_zlt1_cross1h_alt2h_{fit_tag}.npz"
        np.savez(
            out_npz,
            ell_max=float(lmax),
            lb=lb_common,
            dl_ciber_auto_measured=dl_meas,
            dl_ciber_auto_measured_err=dl_meas_err,
            dl_ciber_auto_pred_ls=dl_pred_ls,
            dl_ciber_auto_pred_ls_err=dl_pred_ls_err,
            dl_ciber_auto_pred_hsc=dl_pred_hsc,
            dl_ciber_auto_pred_hsc_err=dl_pred_hsc_err,
            pred_formula=np.array(formula_tag),
            pred_uncertainty=np.array("posterior_sample_propagation"),
            pred_nsamp=int(args.pred_nsamp),
            shot_sub_mode=np.array("prediction_stage" if args.gal_auto_denominator_mode == "shot-subtracted-data" else "not_used"),
            gal_auto_denominator_mode=np.array(str(args.gal_auto_denominator_mode)),
            ratio_scaling_mode=np.array(str(args.ratio_scaling_mode)),
            shot_ell_min=float(args.shot_ell_min),
            shot_ell_max=float(args.shot_ell_max),
            scale_ell_min=float(args.scale_ell_min),
            scale_ell_max=float(args.scale_ell_max),
            ls_cross_fit_file=str(REPO_ROOT / "data" / "cross_cl_fits" / ls_cross_file),
            hsc_cross_fit_file=str(REPO_ROOT / "data" / "cross_cl_fits" / hsc_cross_file),
            gal_auto_fit_file_ls=(str(REPO_ROOT / "data" / "gal_auto_fits" / ls_auto_file) if ls_auto_res is not None else np.array("not_used")),
            gal_auto_fit_file_hsc=(str(REPO_ROOT / "data" / "gal_auto_fits" / hsc_auto_file) if hsc_auto_res is not None else np.array("not_used")),
            alt_auto2h_level_file=np.array("not_used_in_2h1h_shot_sub_mode"),
            reduced_chisq=rchi2_vals,
        )

        fig_pred = _plot_zlt1_auto_prediction_comparison(
            lb=np.asarray(lb_common, dtype=float),
            dl_meas=dl_meas,
            dl_meas_err=dl_meas_err,
            dl_pred_ls=dl_pred_ls,
            dl_pred_ls_err=dl_pred_ls_err,
            dl_pred_hsc=dl_pred_hsc,
            dl_pred_hsc_err=dl_pred_hsc_err,
            dgl_dl=overlay_dgl_dl,
            dgl_dl_err=overlay_dgl_err,
            shot_dl=overlay_shot_dl,
            mock_igl_dl=overlay_mock_igl_dl,
            figsize=(float(getattr(args, "auto_fig_width", 7.0)), float(getattr(args, "auto_fig_height", 3.5))),
        )
        generated.append(
            GeneratedFigure(
                f"zlt1-cross1h-alt2h-prediction:lmax{int(lmax)}",
                fig_pred,
                f"ciber_auto_vs_predicted_from_zlt1_cross1h_alt2h_lmax{int(lmax)}",
            )
        )

    return generated


def _compare_desils_with_without_cmgs_plot(
    clres: Dict[str, Any],
    figsize: Tuple[float, float] = (10, 5),
    inst_list: Sequence[int] = (1, 2),
    labfs: int = 14,
    show: bool = True,
    capsize: float = 2.5,
    bbox_to_anchor: Sequence[float] = (0.0, 1.3),
    ylim: Sequence[float] = (1e-3, 1e3),
    ylim_ratio: Sequence[float] = (-0.25, 0.6),
    xlim: Sequence[float] = (250, 1e5),
    markersize: float = 3,
    top_row_height: float = 1.0,
    bottom_row_height: float = 0.5,
    shot_ell_min: float = 5.0e4,
    shot_ell_max: float = 8.0e4,
):
    import matplotlib.pyplot as plt

    colors = ["k", "C3", "C0"]
    lb = clres["lb"]
    pf = lb * (lb + 1) / (2 * np.pi)

    labels = ["DESI-LS (full)", "DESI-LS CMGs", "with CMGs removed"]
    titles = [
        "Galaxy auto",
        "CIBER 1.1 $\\mu$m $\\times$ DESI-LS",
        "CIBER 1.8 $\\mu$m $\\times$ DESI-LS",
    ]

    fig, ax = plt.subplots(
        figsize=figsize,
        nrows=2,
        ncols=3,
        sharex=True,
        sharey=False,
        gridspec_kw={"height_ratios": [top_row_height, bottom_row_height]},
    )

    shot_mask = (lb >= shot_ell_min) & (lb <= shot_ell_max)
    if not np.any(shot_mask):
        shot_mask = np.ones_like(lb, dtype=bool)

    clg_shot_levels = [
        float(np.nanmean(clres["clg_full"][shot_mask])),
        float(np.nanmean(clres["clg_cmg"][shot_mask])),
        float(np.nanmean(clres["clg_nocmg"][shot_mask])),
    ]
    clig_shot_levels = [
        [
            float(np.nanmean(clres["clig_full"][idx][shot_mask])),
            float(np.nanmean(clres["clig_cmg"][idx][shot_mask])),
            float(np.nanmean(clres["clig_nocmg"][idx][shot_mask])),
        ]
        for idx in range(len(inst_list))
    ]

    ax[0, 0].errorbar(
        lb,
        pf * clres["clg_full"],
        yerr=pf * clres["clgerr_full"],
        fmt="o",
        color=colors[0],
        label=labels[0],
        capsize=capsize,
        markersize=markersize,
    )
    ax[0, 0].errorbar(
        lb,
        pf * clres["clg_cmg"],
        yerr=pf * clres["clgerr_cmg"],
        fmt="o",
        color=colors[1],
        label=labels[1],
        capsize=capsize,
        markersize=markersize,
    )
    ax[0, 0].errorbar(
        lb,
        pf * clres["clg_nocmg"],
        yerr=pf * clres["clgerr_nocmg"],
        fmt="o",
        color=colors[2],
        label=labels[2],
        capsize=capsize,
        markersize=markersize,
    )

    ax[1, 0].errorbar(
        lb,
        1.0 - clres["clg_nocmg"] / clres["clg_full"],
        yerr=clres["clgerr_nocmg"] / clres["clg_full"],
        fmt="o",
        color=colors[0],
        capsize=capsize,
        markersize=markersize,
    )

    ax[0, 0].legend(loc=2, ncol=3, bbox_to_anchor=bbox_to_anchor, fontsize=14)

    # Add thin dashed shot-noise guide curves in top row using high-ell averages.
    for cidx, c in enumerate(colors):
        ax[0, 0].plot(lb, pf * clg_shot_levels[cidx], linestyle="--", linewidth=0.9, color=c, alpha=0.9)

    for idx, inst in enumerate(inst_list):
        ax[0, inst].errorbar(
            lb,
            pf * clres["clig_full"][idx],
            yerr=pf * clres["cligerr_full"][idx],
            fmt="o",
            color=colors[0],
            capsize=capsize,
            markersize=markersize,
        )
        ax[0, inst].errorbar(
            lb,
            pf * clres["clig_cmg"][idx],
            yerr=pf * clres["cligerr_cmg"][idx],
            fmt="o",
            color=colors[1],
            capsize=capsize,
            markersize=markersize,
        )
        ax[0, inst].errorbar(
            lb,
            pf * clres["clig_nocmg"][idx],
            yerr=pf * clres["cligerr_nocmg"][idx],
            fmt="o",
            color=colors[2],
            capsize=capsize,
            markersize=markersize,
        )

        ax[1, inst].errorbar(
            lb,
            1.0 - (clres["clig_nocmg"][idx] / clres["clig_full"][idx]),
            yerr=np.abs(clres["cligerr_nocmg"][idx] / clres["clig_full"][idx]),
            fmt="o",
            color=colors[0],
            capsize=capsize,
            markersize=markersize,
        )

        for cidx, c in enumerate(colors):
            ax[0, inst].plot(
                lb,
                pf * clig_shot_levels[idx][cidx],
                linestyle="--",
                linewidth=0.9,
                color=c,
                alpha=0.9,
            )

    for x in range(3):
        ax[1, x].set_xlabel("$\\ell$", fontsize=labfs)
        ax[1, x].set_ylim(ylim_ratio)
        ax[0, x].set_ylim(ylim)
        ax[0, x].set_yscale("log")
        ax[0, x].set_title(titles[x], fontsize=14)

        if x > 0:
            ax[0, x].set_yticks([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], ["" for _ in range(7)])
            ax[1, x].set_yticks([-0.25, 0.0, 0.25, 0.5], ["" for _ in range(4)])

        for k in range(2):
            ax[k, x].set_xlim(xlim)
            ax[k, x].set_xscale("log")
            ax[k, x].grid(alpha=0.2)

    ax[0, 0].set_ylabel("$\\ell(\\ell+1)C_{\\ell}/2\\pi$", fontsize=labfs)
    ax[1, 0].set_ylabel("$1-\\frac{C_{\\ell}^{\\rm w/o CMGs}}{C_{\\ell}^{\\rm full}}$", fontsize=labfs)

    plt.subplots_adjust(hspace=0.1, wspace=0)
    if show:
        plt.show()

    return fig


def run_compare_desils_cmgs(args: argparse.Namespace) -> List[GeneratedFigure]:
    fns = _import_plotting_functions()
    plt = fns["plt"]
    plot_fieldav_ciber_gal_ps = fns["plot_fieldav_ciber_gal_ps"]
    plot_perfield_gal_auto = fns["plot_perfield_gal_auto"]

    clres: Dict[str, Any] = {}

    addstr_cmg = "all_JHlt14_wFFerr"
    addstr_nocmg = "0.0_z_1.0_nocmg_wrandsub_JHlt16_noelat30_wFFerr"
    addstr_full = "0.0_z_1.0_wrandsub_JHlt16_wFFerr"

    fig_tmp, _, clig_cmg, cligerr_cmg = plot_fieldav_ciber_gal_ps(
        [1, 2],
        "wen_cluster_gals",
        addstr=addstr_cmg,
        figsize=(5, 4),
        capsize=3,
        markersize=3.5,
        startidx=2,
        pred_fpaths=None,
        tl_pix_correct=True,
        ifield_use=8,
        plot_unc=False,
    )
    plt.close(fig_tmp)

    fig_tmp, lb, clg_cmg, clgerr_cmg = plot_perfield_gal_auto(
        "wen_cluster_gals",
        1,
        addstr=addstr_cmg,
        capsize=3,
        markersize=4,
        ylim=[5e-5, 5e1],
        lab_fs=16,
        alph=1.0,
        legend_fs=12,
        ifield_list=[8],
        startidx=0,
        xlim=[300, 1.05e5],
        plot_fieldav=False,
        colors=["k"],
        figsize=(5, 4),
        ylabel="$D_{\\ell}^{gg}$",
        include_legend=False,
        pred_fpaths=None,
    )
    plt.close(fig_tmp)

    fig_tmp, _, clig_nocmg, cligerr_nocmg = plot_fieldav_ciber_gal_ps(
        [1, 2],
        "LS",
        addstr=addstr_nocmg,
        figsize=(5, 4),
        capsize=3,
        markersize=3.5,
        startidx=2,
        pred_fpaths=None,
        tl_pix_correct=True,
        ifield_use=8,
        plot_unc=False,
    )
    plt.close(fig_tmp)

    fig_tmp, _, clg_nocmg, clgerr_nocmg = plot_perfield_gal_auto(
        "LS",
        1,
        addstr=addstr_nocmg,
        capsize=3,
        markersize=4,
        ylim=[5e-5, 5e1],
        lab_fs=16,
        alph=1.0,
        legend_fs=12,
        ifield_list=[8],
        startidx=0,
        xlim=[300, 1.05e5],
        plot_fieldav=False,
        colors=["k"],
        figsize=(5, 4),
        ylabel="$D_{\\ell}^{gg}$",
        include_legend=False,
        pred_fpaths=None,
    )
    plt.close(fig_tmp)

    fig_tmp, _, clig_full, cligerr_full = plot_fieldav_ciber_gal_ps(
        [1, 2],
        "LS",
        addstr=addstr_full,
        figsize=(5, 4),
        capsize=3,
        markersize=3.5,
        startidx=2,
        pred_fpaths=None,
        tl_pix_correct=True,
        ifield_use=8,
        plot_unc=False,
    )
    plt.close(fig_tmp)

    fig_tmp, _, clg_full, clgerr_full = plot_perfield_gal_auto(
        "LS",
        1,
        addstr=addstr_full,
        capsize=3,
        markersize=4,
        ylim=[5e-5, 5e1],
        lab_fs=16,
        alph=1.0,
        legend_fs=12,
        ifield_list=[8],
        startidx=0,
        xlim=[300, 1.05e5],
        plot_fieldav=False,
        colors=["k"],
        figsize=(5, 4),
        ylabel="$D_{\\ell}^{gg}$",
        include_legend=False,
        pred_fpaths=None,
    )
    plt.close(fig_tmp)

    clres["lb"] = lb
    clres["clig_cmg"] = np.array(clig_cmg)
    clres["cligerr_cmg"] = np.array(cligerr_cmg)
    clres["clig_nocmg"] = np.array(clig_nocmg)
    clres["cligerr_nocmg"] = np.array(cligerr_nocmg)
    clres["clig_full"] = np.array(clig_full)
    clres["cligerr_full"] = np.array(cligerr_full)
    clres["clg_cmg"] = np.array(clg_cmg)
    clres["clgerr_cmg"] = np.array(clgerr_cmg)
    clres["clg_nocmg"] = np.array(clg_nocmg)
    clres["clgerr_nocmg"] = np.array(clgerr_nocmg)
    clres["clg_full"] = np.array(clg_full)
    clres["clgerr_full"] = np.array(clgerr_full)

    fig = _compare_desils_with_without_cmgs_plot(clres, show=args.show)
    return [GeneratedFigure("compare-desils-cmgs", fig, "ciber_desils_cmgs")]


def run_amplitude_vs_z(args: argparse.Namespace) -> List[GeneratedFigure]:
    """Plot fitted A_2h (and optionally A_1h) vs redshift with bias-corrected model overlay.

    Loads parametric cross-fit results for DESI-LS and/or HSC, calls
    plot_amplitude_comparison, and overlays the b_g-rescaled IGL mock A_2h
    prediction on the two-halo panel.

    Usage example::

        python scripts/generate_gal_cross_paper_figures.py \\
            amplitude-vs-z \\
            --fit-results-ls data/cross_cl_fits/ciber_cl_fits_DESILS_coarsez_<tag>.npz \\
            --fit-results-hsc data/cross_cl_fits/ciber_cl_fits_HSC_coarsez_<tag>.npz \\
            --bias-cache-fpath scripts/effective_bias_ls_cache.npz
    """
    from ciber.plotting.gal_plotting_fns import plot_amplitude_comparison_by_instrument
    from ciber.theory.cross_ps_parametric_model import load_fit_results_npz
    from ciber.theory.cl_predictions import grab_ciber_cross_vs_z_predfpaths

    zbinedges_coarse = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    z_centers = 0.5 * (np.array(zbinedges_coarse[:-1]) + np.array(zbinedges_coarse[1:]))

    # --- load fit results (one per catalog) ---
    configs = []

    if args.fit_results_ls:
        ls_res = load_fit_results_npz(args.fit_results_ls)
        configs.append({"results": ls_res, "label": "CIBER × DESI-LS", "marker": "o"})

    if args.fit_results_hsc:
        hsc_res = load_fit_results_npz(args.fit_results_hsc)
        configs.append({"results": hsc_res, "label": "CIBER × HSC", "marker": "s"})

    if not configs:
        raise ValueError("At least one of --fit-results-ls or --fit-results-hsc must be provided.")

    # --- build overlays per instrument: one per catalog (LS + HSC if available) ---
    bias_model_overlays = None

    if args.bias_cache_fpath and (args.fit_results_ls or args.fit_results_hsc):
        # Default mock basepath to v2 if not supplied
        mock_basepath = args.pred_basepath if args.pred_basepath else "data/jordan_mocks/v2"
        base = _normalize_pred_basepath(mock_basepath)
        cache = np.load(args.bias_cache_fpath, allow_pickle=False)
        coeffs_ls = np.asarray(cache["coarse_poly_coeffs"])
        b_g_ls = np.poly1d(coeffs_ls)(z_centers)

        # HSC bias: b(z) = 1 + 0.84*z
        b_g_hsc = 1.0 + 0.84 * z_centers

        overlays_per_inst = []
        for inst in args.inst_list:
            inst_overlays = []

            # LS overlay
            if args.fit_results_ls:
                _ls_headstrs = ["sdss_z_lt_22.0_CIBERfidmask", "sdss_z_lt_22.0"]
                pred_fpaths_ls = None
                for headstr in _ls_headstrs:
                    candidate = grab_ciber_cross_vs_z_predfpaths(
                        inst_list=[inst],
                        zbinedges=zbinedges_coarse,
                        jmock_basedir=str(base) + "/",
                        headstr=headstr,
                    )[0]
                    if any(os.path.exists(p) for p in candidate):
                        pred_fpaths_ls = candidate
                        break

                if pred_fpaths_ls is not None:
                    inst_overlays.append({
                        "pred_fpaths": pred_fpaths_ls,
                        "z_centers":   z_centers,
                        "b_g_values":  b_g_ls,
                        "label":       "IGL prediction",
                        "color":       "k",
                        "marker":      "o",
                        "x_offset":    -0.3,
                    })
                else:
                    print(f"[amplitude-vs-z] No LS mock pred files found for TM{inst}")

            # HSC overlay
            if args.fit_results_hsc:
                _hsc_headstrs = ["hsc_i_lt_25.0_CIBERfidmask", "hsc_i_lt_25.0"]
                pred_fpaths_hsc = None
                for headstr in _hsc_headstrs:
                    candidate = grab_ciber_cross_vs_z_predfpaths(
                        inst_list=[inst],
                        zbinedges=zbinedges_coarse,
                        jmock_basedir=str(base) + "/",
                        headstr=headstr,
                    )[0]
                    if any(os.path.exists(p) for p in candidate):
                        pred_fpaths_hsc = candidate
                        break

                if pred_fpaths_hsc is not None:
                    inst_overlays.append({
                        "pred_fpaths": pred_fpaths_hsc,
                        "z_centers":   z_centers,
                        "b_g_values":  b_g_hsc,
                        "label":       None,
                        "color":       "k",
                        "marker":      "s",
                        "x_offset":    +0.3,
                    })
                else:
                    print(f"[amplitude-vs-z] No HSC mock pred files found for TM{inst}")

            overlays_per_inst.append(inst_overlays if inst_overlays else None)

        if any(ov is not None for ov in overlays_per_inst):
            bias_model_overlays = overlays_per_inst

    # --- plot using by-instrument layout (columns=inst, rows=A2h/A1h) ---
    ylim_2h = list(args.ylim_2h) if args.ylim_2h else None
    ylim_ihl = list(args.ylim_ihl) if args.ylim_ihl else None

    fig = plot_amplitude_comparison_by_instrument(
        configs,
        inst_list=tuple(args.inst_list),
        figsize=tuple(args.figsize),
        ylim_2h=ylim_2h,
        ylim_ihl=ylim_ihl,
        legend_ncol=args.legend_ncol,
        use_cmap=False,
        bias_model_overlay=bias_model_overlays,
    )

    return [GeneratedFigure("amplitude-vs-z", fig, "amplitude_vs_z")]


def run_compare_mock_vs_parametric(args: argparse.Namespace) -> List[GeneratedFigure]:
    """Dedicated parametric-fit component plotting for selected fit files.

    This subcommand intentionally keeps outputs separate from legacy figures
    to avoid clutter and to inspect model components by z-bin and instrument.
    """
    fns = _import_plotting_functions()
    plt = fns["plt"]
    plot_cross_fit_components_from_file = fns["plot_cross_fit_components_from_file"]

    def _mock_pred_path_for(
        mock_basepath: Optional[str],
        tracer_key: str,
        inst: int,
        zlo: float,
        zhi: float,
    ) -> Optional[Path]:
        if mock_basepath is None:
            return None

        if tracer_key == "ls":
            heads = ["sdss_z_lt_22.0_CIBERfidmask", "sdss_z_lt_22.0"]
        else:
            heads = ["hsc_i_lt_25.0_CIBERfidmask", "hsc_i_lt_25.0", "hsc_ilt25.0"]

        base = Path(mock_basepath) / "mock_ps_pred" / f"TM{inst}" / "field_average"
        for head in heads:
            fp = base / f"pred_cls_TM{inst}_{head}_zmin={zlo}_zmax={zhi}.npz"
            if fp.exists():
                return fp
        return None

    def _mock_knox_unc(
        lb: np.ndarray,
        clx: np.ndarray,
        clg: Optional[np.ndarray],
        cli: Optional[np.ndarray],
        area_deg2: float = 20.0,
    ) -> np.ndarray:
        fsky = area_deg2 / 41253.0
        if lb.size < 2:
            d_ell = np.full_like(lb, 1.0)
        else:
            d_ell = np.empty_like(lb)
            d_ell[1:-1] = 0.5 * (lb[2:] - lb[:-2])
            d_ell[0] = lb[1] - lb[0]
            d_ell[-1] = lb[-1] - lb[-2]
            d_ell = np.clip(d_ell, 1.0, None)

        nmode = (2.0 * lb + 1.0) * d_ell * fsky
        nmode = np.clip(nmode, 1.0e-12, None)
        if clg is not None and cli is not None:
            var = clx**2 + np.abs(clg * cli)
        else:
            var = clx**2
        return np.sqrt(np.abs(var) / nmode)

    def _evaluate_parametric_total(
        fit_dat: Any,
        inst_idx: int,
        zbin_idx: int,
        ell_eval: np.ndarray,
        rng_seed: int,
        nsamp: int = 200,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        from ciber.theory.cross_ps_parametric_model import CrossPowerSpectrumModel

        if "params" not in fit_dat:
            raise KeyError("Missing params in fit results")

        params = np.asarray(fit_dat["params"][inst_idx, zbin_idx], dtype=float)
        if params.size < 5:
            raise ValueError("Expected at least 5 fit parameters for parametric model evaluation")

        perr = None
        if "params_err" in fit_dat:
            perr = np.asarray(fit_dat["params_err"][inst_idx, zbin_idx], dtype=float)

        use_powerlaw_2h = bool(fit_dat["use_powerlaw_2h"]) if "use_powerlaw_2h" in fit_dat else True
        alpha_2h_fixed = float(fit_dat["alpha_2h_fixed"]) if "alpha_2h_fixed" in fit_dat else 0.0
        use_lorentzian_1h = bool(fit_dat["use_lorentzian_1h"]) if "use_lorentzian_1h" in fit_dat else False

        lb_ref_obj = fit_dat["lb_fit"][inst_idx, zbin_idx] if "lb_fit" in fit_dat else None
        if lb_ref_obj is None:
            lb_ref = ell_eval.copy()
        else:
            lb_ref = np.asarray(lb_ref_obj, dtype=float)
            if lb_ref.size == 0:
                lb_ref = ell_eval.copy()

        model = CrossPowerSpectrumModel(
            lb=lb_ref,
            use_powerlaw_2h=use_powerlaw_2h,
            alpha_2h_fixed=alpha_2h_fixed,
            use_lorentzian_1h=use_lorentzian_1h,
            use_astrometry_damping=(params.size >= 6),
        )

        if params.size >= 6:
            y = model.model_dl(
                ell_eval,
                params[0],
                params[1],
                params[2],
                params[3],
                params[4],
                sigma_damp=params[5],
            )
        else:
            y = model.model_dl(ell_eval, params[0], params[1], params[2], params[3], params[4])

        if perr is None or perr.size < 5 or not np.any(np.isfinite(perr)):
            return y, None, None

        rng = np.random.default_rng(rng_seed)
        sig = np.where(np.isfinite(perr), perr, 0.0)
        draws = rng.normal(loc=params, scale=sig, size=(nsamp, params.size))

        # Enforce positivity for amplitudes and width-like terms.
        draws[:, 0] = np.clip(draws[:, 0], 0.0, None)  # A_2h
        draws[:, 1] = np.clip(draws[:, 1], 0.0, None)  # A_1h
        draws[:, 3] = np.clip(draws[:, 3], 1.0e-3, None)  # sigma_1h
        draws[:, 4] = np.clip(draws[:, 4], 0.0, None)  # A_shot
        if params.size >= 6:
            draws[:, 5] = np.clip(draws[:, 5], 0.0, None)  # sigma_damp arcsec

        ys = np.zeros((nsamp, ell_eval.size), dtype=float)
        for i in range(nsamp):
            p = draws[i]
            if p.size >= 6:
                ys[i] = model.model_dl(ell_eval, p[0], p[1], p[2], p[3], p[4], sigma_damp=p[5])
            else:
                ys[i] = model.model_dl(ell_eval, p[0], p[1], p[2], p[3], p[4])

        ylo = np.percentile(ys, 16.0, axis=0)
        yhi = np.percentile(ys, 84.0, axis=0)
        return y, ylo, yhi

    def _build_total_data_vs_model_fig(
        tracer_label: str,
        source_stem: str,
        zbin_idx: int,
        zbins: np.ndarray,
        lb_fit: np.ndarray,
        model_dl: np.ndarray,
        measured_dir: Path,
        tracer_key: str,
        mock_basepath: Optional[str],
        fit_dat: Any,
        fit_lmax: float,
    ) -> Any:
        fig, axes = plt.subplots(1, 2, figsize=(6, 3.5), sharex=True, sharey=True)
        colors = ["C0", "C3"]
        lams = [1.1, 1.8]

        zlo = float(zbins[zbin_idx])
        zhi = float(zbins[zbin_idx + 1])

        def _measured_npz_for(inst_idx: int) -> Path:
            inst = inst_idx + 1
            if tracer_key == "ls":
                return measured_dir / "DESILS_coarsez" / f"cl_CIBER_TM{inst}_DESILS_zbin{zbin_idx}.npz"
            return measured_dir / "HSC_coarsez" / f"cl_CIBER_TM{inst}_HSC_ilt25.0_zbin{zbin_idx}.npz"

        for inst_idx in range(min(2, lb_fit.shape[0])):
            ax = axes[inst_idx]
            lb_vals = lb_fit[inst_idx, zbin_idx]
            model_vals = model_dl[inst_idx, zbin_idx]

            if lb_vals is None or model_vals is None:
                continue

            lb_arr = np.asarray(lb_vals, dtype=float)
            model_arr = np.asarray(model_vals, dtype=float)

            if lb_arr.size == 0 or model_arr.size == 0:
                continue

            color = colors[inst_idx]
            lam = lams[inst_idx]

            # Omit first two bandpowers as in the standard fit plotting flow.
            if lb_arr.size > 2 and model_arr.size > 2:
                lb_arr = lb_arr[2:]
                model_arr = model_arr[2:]

            mock_pred_path = _mock_pred_path_for(
                mock_basepath=mock_basepath,
                tracer_key=tracer_key,
                inst=inst_idx + 1,
                zlo=zlo,
                zhi=zhi,
            )
            if mock_pred_path is not None:
                pred = np.load(str(mock_pred_path), allow_pickle=True)
                if "lb" in pred and "cross" in pred:
                    lb_mock = np.asarray(pred["lb"], dtype=float)
                    clx_mock = np.asarray(pred["cross"], dtype=float)
                    clg_mock = np.asarray(pred["gal_auto"], dtype=float) if "gal_auto" in pred else None
                    cli_mock = (
                        np.asarray(pred["intensity_auto_full"], dtype=float)
                        if "intensity_auto_full" in pred
                        else None
                    )
                    if lb_mock.size > 2 and clx_mock.size > 2:
                        lb_mock = lb_mock[2:]
                        clx_mock = clx_mock[2:]
                        if clg_mock is not None and clg_mock.size > 2:
                            clg_mock = clg_mock[2:]
                        if cli_mock is not None and cli_mock.size > 2:
                            cli_mock = cli_mock[2:]

                    pf_mock = lb_mock * (lb_mock + 1.0) / (2.0 * np.pi)
                    area_deg2 = 20.0 if tracer_key == "ls" else 4.0
                    dclx_mock = _mock_knox_unc(lb_mock, clx_mock, clg_mock, cli_mock, area_deg2=area_deg2)
                    dl_mock = pf_mock * clx_mock
                    ddl_mock = pf_mock * dclx_mock

                    ax.fill_between(
                        lb_mock,
                        np.clip(np.abs(dl_mock) - ddl_mock, 1.0e-12, None),
                        np.abs(dl_mock) + ddl_mock,
                        color=color,
                        alpha=0.12,
                        zorder=0,
                    )
                    ax.plot(
                        lb_mock,
                        np.abs(dl_mock),
                        color=color,
                        lw=1.8,
                        linestyle="dashed",
                        alpha=0.9,
                        label=f"TM{inst_idx + 1} mock ({lam:.1f} um)",
                        zorder=1,
                    )

            ell_model = np.logspace(np.log10(300.0), np.log10(1.2e5), 450)
            y_param, ylo_param, yhi_param = _evaluate_parametric_total(
                fit_dat=fit_dat,
                inst_idx=inst_idx,
                zbin_idx=zbin_idx,
                ell_eval=ell_model,
                rng_seed=1000 + 100 * zbin_idx + inst_idx,
            )

            if ylo_param is not None and yhi_param is not None:
                ax.fill_between(
                    ell_model,
                    np.clip(ylo_param, 1.0e-12, None),
                    np.clip(yhi_param, 1.0e-12, None),
                    color=color,
                    alpha=0.18,
                    zorder=2,
                )

            ax.plot(
                ell_model,
                np.clip(y_param, 1.0e-12, None),
                color=color,
                lw=2.0,
                linestyle="solid",
                label=f"TM{inst_idx + 1} parametric ({lam:.1f} um)",
                zorder=2.5,
            )

            data_path = _measured_npz_for(inst_idx)
            if not data_path.exists():
                continue

            d = np.load(str(data_path), allow_pickle=True)
            if not all(k in d for k in ["lb", "cl", "clerr"]):
                continue
            lb_data = np.asarray(d["lb"], dtype=float)
            cl_data = np.asarray(d["cl"], dtype=float)
            clerr_data = np.asarray(d["clerr"], dtype=float)

            if lb_data.size > 2 and cl_data.size > 2 and clerr_data.size > 2:
                lb_data = lb_data[2:]
                cl_data = cl_data[2:]
                clerr_data = clerr_data[2:]

            pf_data = lb_data * (lb_data + 1.0) / (2.0 * np.pi)

            dl_data = pf_data * cl_data
            dlerr_data = pf_data * clerr_data
            posmask = dl_data > 0
            negmask = dl_data < 0

            if np.any(posmask):
                ax.errorbar(
                    lb_data[posmask],
                    dl_data[posmask],
                    yerr=dlerr_data[posmask],
                    fmt="o",
                    ms=3.5,
                    capsize=2,
                    color=color,
                    alpha=0.9,
                    label=f"TM{inst_idx + 1} data",
                    zorder=4,
                )
            if np.any(negmask):
                ax.errorbar(
                    lb_data[negmask],
                    np.abs(dl_data[negmask]),
                    yerr=dlerr_data[negmask],
                    fmt="o",
                    ms=3.5,
                    capsize=2,
                    color=color,
                    mfc="white",
                    alpha=0.9,
                    zorder=4,
                )

            ax.axvspan(fit_lmax, 1.0e5, color="lightgrey", alpha=0.25, zorder=0)
            ax.set_title(f"{lams[inst_idx]:.1f} $\\mu$m", fontsize=10)

        for ax in axes:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim([300, 1.0e5])
            ax.set_ylim([5e-3, 1e3])
            ax.grid(alpha=0.3)

        axes[0].set_ylabel(r"$D_\ell$ [nW m$^{-2}$ sr$^{-1}$]")
        for ax in axes:
            ax.set_xlabel(r"$\ell$")
        axes[0].legend(fontsize=8, ncol=1, loc="upper left")
        fig.suptitle(f"{tracer_label}: {zlo:.1f} < z < {zhi:.1f}", fontsize=11)
        fig.tight_layout()

        stem = (
            "compare_mock_vs_parametric/by_z/"
            f"parametric_total_data_{tracer_label.lower().replace('-', '')}_{source_stem}_"
            f"z{zbin_idx}_zmin{zlo:.1f}_zmax{zhi:.1f}"
        )
        return fig, stem

    tracer_choice = getattr(args, "tracers", "both")
    wanted_ls = tracer_choice in {"ls", "both"}
    wanted_hsc = tracer_choice in {"hsc", "both"}

    fit_pairs: List[Tuple[str, Path]] = []
    if args.fit_results_ls is not None and wanted_ls:
        fit_pairs.append(("ls", Path(args.fit_results_ls).expanduser()))
    if args.fit_results_hsc is not None and wanted_hsc:
        fit_pairs.append(("hsc", Path(args.fit_results_hsc).expanduser()))

    if not fit_pairs:
        fit_dir = Path(args.fit_search_dir).expanduser()
        if not fit_dir.exists():
            raise FileNotFoundError(f"Fit search directory does not exist: {fit_dir}")

        family = args.fit_family
        lmax = int(args.fit_lmax)

        if wanted_ls:
            ls_candidates = [
                fit_dir / f"DESILS_coarsez_cross_cl_fits_{family}_lMax={lmax}.npz",
            ]
            ls_path = next((p for p in ls_candidates if p.exists()), None)
            if ls_path is None:
                raise FileNotFoundError(
                    "Could not auto-find LS fit file. Tried: "
                    + ", ".join(str(p) for p in ls_candidates)
                )
            fit_pairs.append(("ls", ls_path))

        if wanted_hsc:
            hsc_candidates = [
                fit_dir / f"HSC_coarsez_ilt25.0_cross_cl_fits_{family}_lMax={lmax}.npz",
                fit_dir / f"HSC_coarsez_cross_cl_fits_{family}_lMax={lmax}.npz",
            ]
            hsc_path = next((p for p in hsc_candidates if p.exists()), None)
            if hsc_path is None:
                raise FileNotFoundError(
                    "Could not auto-find HSC fit file. Tried: "
                    + ", ".join(str(p) for p in hsc_candidates)
                )
            fit_pairs.append(("hsc", hsc_path))

    if not fit_pairs:
        raise ValueError("No fit files selected for compare-mock-vs-parametric")

    out: List[GeneratedFigure] = []
    measured_dir = Path(getattr(args, "measured_cross_dir", "data/ciber_gal_cross_cls")).expanduser()
    mock_basepath: Optional[str] = None
    if not getattr(args, "no_mock_overlay", False):
        mock_base_in = args.mock_pred_basepath if args.mock_pred_basepath else args.pred_basepath
        if mock_base_in is not None:
            try:
                mock_basepath = _normalize_pred_basepath(mock_base_in)
                _ensure_single_realization_pred_cls(
                    mock_basepath,
                    beam_correct=not args.pred_no_beam_correct,
                    beam_ifield=args.pred_beam_ifield,
                )
            except Exception:
                mock_basepath = None

    for tracer, fp in fit_pairs:
        if not fp.exists():
            raise FileNotFoundError(f"Fit results file not found: {fp}")

        dat = np.load(fp, allow_pickle=True)
        if "zbinedges" not in dat:
            raise KeyError(f"Missing 'zbinedges' in {fp}")
        zbinedges = dat["zbinedges"]

        cat_label = "DESI-LS" if tracer == "ls" else "HSC"
        fig, _ = plot_cross_fit_components_from_file(
            npz_path=str(fp),
            zbinedges=zbinedges,
            inst_list=[1, 2],
            cat=cat_label,
            organize_by="zbin",
            figsize=(16, 9),
            show_data=True,
        )
        stem = f"compare_mock_vs_parametric/parametric_components_{tracer}_{fp.stem}"
        out.append(GeneratedFigure(f"compare-mock-vs-parametric:{tracer}:{fp.stem}", fig, stem))

        if not all(k in dat for k in ["lb_fit", "model_dl"]):
            continue

        lb_fit = np.asarray(dat["lb_fit"], dtype=object)
        model_dl = np.asarray(dat["model_dl"], dtype=object)

        n_zbin = len(zbinedges) - 1
        for zidx in range(n_zbin):
            fig_z, stem_z = _build_total_data_vs_model_fig(
                tracer_label=cat_label,
                source_stem=fp.stem,
                zbin_idx=zidx,
                zbins=np.asarray(zbinedges, dtype=float),
                lb_fit=lb_fit,
                model_dl=model_dl,
                measured_dir=measured_dir,
                tracer_key=tracer,
                mock_basepath=mock_basepath,
                fit_dat=dat,
                fit_lmax=float(args.fit_lmax),
            )
            out.append(
                GeneratedFigure(
                    f"compare-mock-vs-parametric:{tracer}:{fp.stem}:z{zidx}",
                    fig_z,
                    stem_z,
                )
            )

    return out


# -----------------------------
# Runner and CLI plumbing
# -----------------------------


def _run_generated_figures(
    args: argparse.Namespace,
    run_callable: Callable[[], List[GeneratedFigure]],
) -> List[FigureTiming]:
    plt = _import_plotting_functions()["plt"]
    records: List[FigureTiming] = []

    t_total0 = _now()
    load_stats: Dict[str, float] = {"load_sec": 0.0}

    try:
        t_build0 = _now()
        with instrument_load_timing() as measured_stats:
            load_stats = measured_stats
            generated = run_callable()
        build_sec = _now() - t_build0
        load_sec = max(0.0, float(load_stats.get("load_sec", 0.0)))
        plot_sec = max(0.0, build_sec - load_sec)

        for gen in generated:
            t_save0 = _now()
            outpath = _save_figure(
                gen.fig,
                outdir=Path(args.outdir).expanduser(),
                stem=gen.stem,
                ext=args.format,
                overwrite=args.overwrite,
                add_timestamp=args.timestamp,
            )
            save_sec = _now() - t_save0

            total_sec = _now() - t_total0

            records.append(
                FigureTiming(
                    figure_key=gen.figure_key,
                    output_path=str(outpath),
                    load_sec=load_sec,
                    plot_sec=plot_sec,
                    save_sec=save_sec,
                    total_sec=total_sec,
                    status="ok",
                    error="",
                )
            )

            print(f"Saved {gen.figure_key} -> {outpath}")
            plt.close(gen.fig)

    except Exception as exc:
        import traceback

        traceback.print_exc()
        total_sec = _now() - t_total0
        records.append(
            FigureTiming(
                figure_key=getattr(args, "command", "unknown"),
                output_path="",
                load_sec=max(0.0, float(load_stats.get("load_sec", 0.0))),
                plot_sec=0.0,
                save_sec=0.0,
                total_sec=total_sec,
                status="failed",
                error=str(exc),
            )
        )

    return records


def _dispatch_single(args: argparse.Namespace, command: str) -> List[FigureTiming]:
    if command == "omnibus":
        return _run_generated_figures(args, lambda: run_omnibus(args))

    if command == "field-consistency":
        variants = [args.variant]
        if args.variant == "all":
            variants = ["wise-cross", "wise-auto", "ls-cross", "ls-auto"]

        all_records: List[FigureTiming] = []
        for variant in variants:
            all_records.extend(
                _run_generated_figures(
                    args,
                    lambda v=variant: run_field_consistency_single(args, v),
                )
            )
        return all_records

    if command == "forecast":
        return _run_generated_figures(args, lambda: run_forecast(args))

    if command == "rl-vs-z-scale":
        return _run_generated_figures(args, lambda: run_rl_vs_z_scale(args))

    if command == "gaia-auto":
        return _run_generated_figures(args, lambda: run_gaia_auto(args))

    if command == "gaia-cross":
        return _run_generated_figures(args, lambda: run_gaia_cross(args))

    if command == "compare-r-ell":
        return _run_generated_figures(args, lambda: run_compare_r_ell(args))

    if command == "cross-redshift":
        return _run_generated_figures(args, lambda: run_cross_redshift(args))

    if command == "zlt1-auto-prediction":
        return _run_generated_figures(args, lambda: run_zlt1_auto_prediction(args))

    if command == "zlt1-simple-model-diagnostics":
        return _run_generated_figures(args, lambda: run_zlt1_simple_model_diagnostics(args))

    if command == "zlt1-alt2h-prediction":
        return _run_generated_figures(args, lambda: run_zlt1_alt2h_prediction(args))

    if command == "zlt1-cross1h-refit-update":
        return _run_generated_figures(args, lambda: run_zlt1_cross_with1h_update_prediction(args))

    if command == "compare-desils-cmgs":
        return _run_generated_figures(args, lambda: run_compare_desils_cmgs(args))

    if command == "compare-mock-vs-parametric":
        return _run_generated_figures(args, lambda: run_compare_mock_vs_parametric(args))

    if command == "amplitude-vs-z":
        return _run_generated_figures(args, lambda: run_amplitude_vs_z(args))

    raise ValueError(f"Unknown command: {command}")


def _default_all_commands() -> List[str]:
    return [
        "omnibus",
        "field-consistency",
        "forecast",
        "rl-vs-z-scale",
        "gaia-auto",
        "gaia-cross",
        "cross-redshift",
        "zlt1-auto-prediction",
        "zlt1-alt2h-prediction",
        "zlt1-cross1h-refit-update",
        "compare-r-ell",
        "compare-desils-cmgs",
        "compare-mock-vs-parametric",
        "amplitude-vs-z",
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate CIBER galaxy-cross paper figures from one CLI.",
    )

    parser.add_argument(
        "--outdir",
        default="figures/generated_gal_cross",
        help="Directory to save generated figures and diagnostics",
    )
    parser.add_argument(
        "--format",
        default="pdf",
        choices=["pdf", "png"],
        help="Figure output format",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs",
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Append timestamp suffix to output filenames",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively",
    )
    parser.add_argument(
        "--pred-source",
        choices=["current", "standard-igl", "parametric"],
        default="current",
        help="Prediction source mode",
    )
    parser.add_argument(
        "--pred-basepath",
        default=None,
        help="Base path containing mock_ps_pred for standard-igl mode",
    )
    parser.add_argument(
        "--pred-beam-ifield",
        type=int,
        default=8,
        help="CIBER field index for B_ell beam correction when constructing standard-igl pred curves",
    )
    parser.add_argument(
        "--pred-no-beam-correct",
        action="store_true",
        help="Disable B_ell deconvolution when constructing standard-igl pred curves",
    )
    parser.add_argument(
        "--diagnostics-basename",
        default="timing_diagnostics",
        help="Basename for diagnostics JSON/CSV",
    )
    parser.add_argument(
        "--omnibus-tl-ifield",
        type=int,
        default=8,
        help="CIBER field index used to load tl_pix transfer-function for omnibus cross correction",
    )
    parser.add_argument(
        "--omnibus-tl-pix-template",
        default="data/fluctuation_data/transfer_function/tl_clx_pix_TM{inst}_ifield{ifield}.npz",
        help="Path template for omnibus tl_pix files; supports {inst} and {ifield}",
    )
    parser.add_argument(
        "--omnibus-no-tl-pix-correct",
        action="store_true",
        help="Disable tl_pix correction for omnibus measured and model cross spectra",
    )
    parser.add_argument(
        "--omnibus-include-full",
        action="store_true",
        help="Also generate the full omnibus (with unWISE/CIBER-auto panels); default is LS/HSC-only omnibus",
    )
    parser.add_argument(
        "--omnibus-figsize",
        type=float,
        nargs=2,
        default=[10, 6],
        help="Figure size (width, height) for omnibus plots (default: 7 6)",
    )
    parser.add_argument(
        "--ls-gal-auto-large",
        default=None,
        help="Path to .npz from compute_gal_auto_spectrum_large() to replace the LS galaxy auto with a larger-footprint version",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("omnibus", help="Generate create_omnibus_plot output")

    p_field = subparsers.add_parser(
        "field-consistency",
        help="Generate appendix field-consistency plot(s)",
    )
    p_field.add_argument(
        "--variant",
        default="all",
        choices=["all", "wise-cross", "wise-auto", "ls-cross", "ls-auto"],
        help="Field-consistency variant to run",
    )
    p_field.add_argument("--ell-min", type=float, default=300)
    p_field.add_argument("--ell-max", type=float, default=10000)

    p_fore = subparsers.add_parser(
        "forecast",
        help="Generate plot_clIG_forecast from a precomputed npz input",
    )
    p_fore.add_argument(
        "--input-npz",
        required=True,
        help="NPZ file with lb, lrange, dcl_terms_bp, dcl_vs_nbar, xerr",
    )

    p_rl = subparsers.add_parser(
        "rl-vs-z-scale",
        help="Generate plot_rl_vs_z_vs_scale_DESILS from precomputed input",
    )
    p_rl.add_argument(
        "--input-npz",
        required=True,
        help="NPZ with res_meas and mean_rl_diffscale_pred (or equivalent keys)",
    )

    subparsers.add_parser("gaia-auto", help="Generate Gaia star auto-spectrum figure")
    p_gaia_cross = subparsers.add_parser(
        "gaia-cross", help="Generate CIBER x Gaia star cross-spectrum figure"
    )
    p_gaia_cross.add_argument(
        "--rerun-fit",
        action="store_true",
        dest="rerun_fit",
        help="Rerun MCMC fit even if cached results exist in data/gaia_cross_fits/",
    )

    p_compare = subparsers.add_parser(
        "compare-r-ell",
        help="Generate compare_r_ell_hsc_LS_zlt1 figure",
    )
    p_compare.add_argument(
        "--add-trilegal-isl",
        action="store_true",
        help="Overlay IGL+ISL curves using unresolved ISL residual C_ell from TRILEGAL catalogs",
    )
    p_compare.add_argument(
        "--isl-trilegal-datestr",
        default="112022",
        help="Date-tag directory under data/ciber_mocks for TRILEGAL residual spectra",
    )
    p_compare.add_argument(
        "--isl-maglim-vega",
        type=float,
        default=16.0,
        help="Vega magnitude threshold for unresolved-star residual ISL column selection",
    )
    p_compare.add_argument(
        "--isl-trilegal-stat",
        choices=["mean", "median"],
        default="mean",
        help="Statistic over TRILEGAL simidx realizations for the unresolved ISL C_ell",
    )
    p_compare.add_argument(
        "--isl-trilegal-basepath",
        default=None,
        help="Optional explicit path to TRILEGAL ISL residual spectra directory",
    )

    p_cross_z = subparsers.add_parser(
        "cross-redshift",
        help="Generate DESI-LS dz=0.1 and LS+HSC dz=0.2 cross-spectrum redshift-bin figures",
    )
    p_cross_z.add_argument(
        "--bias-cache-fpath",
        default=None,
        metavar="PATH",
        help="Path to effective_bias_ls_cache.npz produced by compute_effective_bias_ls.py. "
             "When provided, replaces the noisy mock IGL curve with a smooth shot-noise + "
             "two-halo model rescaled by the measured b_g at each redshift.",
    )

    p_zlt1 = subparsers.add_parser(
        "zlt1-auto-prediction",
        help="Fit z<1 DESI-LS/HSC cross+galaxy-auto spectra and compare measured vs predicted CIBER auto in two panels",
    )
    p_zlt1.add_argument("--fit-tag", default="zlt1_parametric", help="Tag appended to saved fit and prediction filenames")
    p_zlt1.add_argument(
        "--fit-mode",
        choices=["bulk", "coarsez"],
        default="bulk",
        help="Fit either one bulk z<1 bin (bulk, omnibus-style) or coarse redshift slices (coarsez)",
    )
    p_zlt1.add_argument(
        "--cross-fit-lmax",
        type=float,
        default=80000.0,
        help="Maximum ell used for galaxy-cross parametric fits",
    )
    p_zlt1.add_argument(
        "--auto-fit-lmax",
        type=float,
        default=80000.0,
        help="Maximum ell used for galaxy-auto parametric fits",
    )
    p_zlt1.add_argument(
        "--ihl-1h-params-path",
        default="ihl_templates/ihl_1h_param_fit_v0.npz",
        help="Path to IHL-derived one-halo parameter file used to temper priors/fixed 1h shape",
    )
    p_zlt1.add_argument(
        "--one-halo-template-mode",
        choices=["legacy", "slice-median", "slice-effective-z", "slice-z-range"],
        default="legacy",
        help=(
            "One-halo template policy for bulk z<1 fits: legacy behavior or derive one fixed "
            "template from sliced fit files (median or effective-z)."
        ),
    )
    p_zlt1.add_argument(
        "--slice-template-fit-files",
        nargs="+",
        default=None,
        help=(
            "Sliced fit-result npz files used to derive fixed one-halo template when "
            "--one-halo-template-mode is not legacy."
        ),
    )
    p_zlt1.add_argument(
        "--one-halo-effective-z",
        type=float,
        default=0.5,
        help="Target redshift when --one-halo-template-mode=slice-effective-z",
    )
    p_zlt1.add_argument("--hsc-headstr", default="hsc_ilt25.0", help="HSC headstr used to load z-binned spectra")
    p_zlt1.add_argument("--fit-nwalkers", type=int, default=32, help="Number of MCMC walkers for fit stages")
    p_zlt1.add_argument("--fit-nsteps", type=int, default=2000, help="Number of MCMC steps for fit stages")
    p_zlt1.add_argument("--fit-nburn", type=int, default=500, help="Number of burn-in MCMC steps")
    p_zlt1.add_argument(
        "--no-save-intermediate-fits",
        action="store_true",
        help="Disable saving intermediate fit/decomposition figures for cross/auto model stages",
    )
    p_zlt1.add_argument(
        "--use-astrometry-damping",
        action="store_true",
        help="Deprecated for this command: cross fits always use damping and galaxy-auto fits always disable damping",
    )
    p_zlt1.add_argument("--startidx", type=int, default=2, help="Start index in ell bins for prediction products")
    p_zlt1.add_argument("--endidx", type=int, default=-1, help="End index in ell bins for prediction products")
    p_zlt1.add_argument("--scale-ell-min", type=float, default=300.0, help="Min ell for cross/auto scaling ratio")
    p_zlt1.add_argument("--scale-ell-max", type=float, default=3000.0, help="Max ell for cross/auto scaling ratio")
    p_zlt1.add_argument("--auto-fit-ell-min", type=float, default=300.0, help="Min ell for galaxy-auto fit inside prediction step")
    p_zlt1.add_argument("--auto-fit-ell-max", type=float, default=80000.0, help="Max ell for galaxy-auto fit inside prediction step")
    p_zlt1.add_argument("--pred-nsamp", type=int, default=2000, help="Posterior draws used for auto-prediction uncertainty propagation")
    p_zlt1.add_argument(
        "--gal-auto-denominator-mode",
        choices=["shot-subtracted-data", "smooth-model"],
        default="smooth-model",
        help="Denominator used in auto reconstruction: shot-subtracted data or fitted smooth 2h+1h galaxy-auto model",
    )
    p_zlt1.add_argument(
        "--ratio-scaling-mode",
        choices=["direct-ratio", "separate-2h-1h"],
        default="direct-ratio",
        help="Use direct (2h+1h)/(2h+1h) ratio or separate 2h/1h scaling in reconstruction",
    )
    p_zlt1.add_argument("--shot-ell-min", type=float, default=5.0e4, help="Min ell used to estimate galaxy shot noise in prediction stage")
    p_zlt1.add_argument("--shot-ell-max", type=float, default=8.0e4, help="Max ell used to estimate galaxy shot noise in prediction stage")

    p_zlt1_simple = subparsers.add_parser(
        "zlt1-simple-model-diagnostics",
        help=(
            "Run z<1 simplified fits and diagnostics: cross=2h+shot+damping, "
            "auto=2h+shot, for a list of ell_max values"
        ),
    )
    p_zlt1_simple.add_argument(
        "--fit-tag",
        default="zlt1_simple",
        help="Tag prefix for saved fit and figure outputs",
    )
    p_zlt1_simple.add_argument(
        "--ell-max-list",
        type=float,
        nargs="+",
        default=[30000.0, 50000.0, 80000.0],
        help="List of ell_max values for repeated diagnostics",
    )
    p_zlt1_simple.add_argument(
        "--hsc-headstr",
        default="hsc_ilt25.0",
        help="HSC headstr used to load z-binned spectra",
    )
    p_zlt1_simple.add_argument("--fit-nwalkers", type=int, default=24, help="Number of MCMC walkers")
    p_zlt1_simple.add_argument("--fit-nsteps", type=int, default=300, help="Number of MCMC steps")
    p_zlt1_simple.add_argument("--fit-nburn", type=int, default=80, help="Number of burn-in MCMC steps")
    p_zlt1_simple.add_argument("--startidx", type=int, default=2, help="Start index in ell bins for fit diagnostics")
    p_zlt1_simple.add_argument("--endidx", type=int, default=-1, help="End index in ell bins for fit diagnostics")
    p_zlt1_simple.add_argument(
        "--no-save-intermediate-fits",
        action="store_true",
        help="Disable saving intermediate fit/corner figures from model stages",
    )
    p_zlt1_simple.add_argument(
        "--alt-auto-2h-strategy",
        action="store_true",
        help=(
            "Also compute alternate galaxy-auto 2h estimate by subtracting high-ell shot noise "
            "and averaging shot-subtracted D_ell below --alt-2h-ell-max"
        ),
    )
    p_zlt1_simple.add_argument(
        "--alt-2h-ell-max",
        type=float,
        default=2000.0,
        help="Maximum ell used to average shot-subtracted D_ell for alternate 2h level",
    )
    p_zlt1_simple.add_argument(
        "--alt-shot-ell-min",
        type=float,
        default=50000.0,
        help="Minimum ell used to estimate shot-noise level in C_ell",
    )
    p_zlt1_simple.add_argument(
        "--alt-shot-ell-max",
        type=float,
        default=80000.0,
        help="Maximum ell used to estimate shot-noise level in C_ell",
    )

    p_zlt1_alt = subparsers.add_parser(
        "zlt1-alt2h-prediction",
        help=(
            "Predict z<1 CIBER auto from cross 2h amplitudes and alternate 2h gg levels "
            "(shot-subtracted low-ell strategy)"
        ),
    )
    p_zlt1_alt.add_argument(
        "--fit-tag",
        default="zlt1_simple_alt2h",
        help="Tag prefix used when reading *_lmax{ell} cross-fit and alt-level products",
    )
    p_zlt1_alt.add_argument(
        "--ell-max-list",
        type=float,
        nargs="+",
        default=[30000.0, 50000.0, 80000.0],
        help="List of ell_max values to process",
    )

    p_cross1h = subparsers.add_parser(
        "zlt1-cross1h-refit-update",
        help=(
            "Refit z<1 cross with 1h+damping, make component uncertainty diagnostics, "
            "and conditionally update auto-vs-predicted using alternate 2h gg levels"
        ),
    )
    p_cross1h.add_argument(
        "--fit-tag",
        default="zlt1_cross1h_alt2h",
        help="Tag prefix for cross-fit outputs (appends _lmax{ell})",
    )
    p_cross1h.add_argument(
        "--reuse-fit-cache",
        action="store_true",
        default=True,
        help="Reuse existing cross/auto fit NPZ files when present (default behavior)",
    )
    p_cross1h.add_argument(
        "--force-refit",
        action="store_false",
        dest="reuse_fit_cache",
        help="Force rerunning cross/auto fits even if cached NPZ files already exist",
    )
    p_cross1h.add_argument(
        "--one-halo-template-mode",
        choices=["legacy", "slice-median", "slice-effective-z", "slice-z-range"],
        default="slice-z-range",
        help=(
            "One-halo template policy for z<1 refits. Default uses slice-z-range over "
            "0.2<z<0.4 from provided slice fit files."
        ),
    )
    p_cross1h.add_argument(
        "--slice-template-fit-files",
        nargs="+",
        default=[
            "data/cross_cl_fits/DESILS_coarsez_cross_cl_fits_IHL1hfit_fixshape_wdamp_lMax=50000.npz",
            "data/cross_cl_fits/HSC_coarsez_cross_cl_fits_IHL1hfit_fixshape_wdamp_lMax=50000.npz",
        ],
        help="Sliced fit-result npz files used to derive fixed one-halo template",
    )
    p_cross1h.add_argument(
        "--one-halo-effective-z",
        type=float,
        default=0.3,
        help="Target redshift when --one-halo-template-mode=slice-effective-z",
    )
    p_cross1h.add_argument(
        "--one-halo-z-min",
        type=float,
        default=0.2,
        help="Minimum redshift for --one-halo-template-mode=slice-z-range",
    )
    p_cross1h.add_argument(
        "--one-halo-z-max",
        type=float,
        default=0.4,
        help="Maximum redshift for --one-halo-template-mode=slice-z-range",
    )
    p_cross1h.add_argument(
        "--alt-level-fit-tag",
        default="zlt1_simple_alt2h",
        help="Tag prefix used to locate zlt1_alt_auto2h_levels_* files",
    )
    p_cross1h.add_argument(
        "--ell-max-list",
        type=float,
        nargs="+",
        default=[30000.0, 50000.0, 80000.0],
        help="List of ell_max values to process",
    )
    p_cross1h.add_argument("--hsc-headstr", default="hsc_ilt25.0", help="HSC headstr used to load z-binned spectra")
    p_cross1h.add_argument("--fit-nwalkers", type=int, default=32, help="Number of MCMC walkers")
    p_cross1h.add_argument("--fit-nsteps", type=int, default=4000, help="Number of MCMC steps")
    p_cross1h.add_argument("--fit-nburn", type=int, default=1000, help="Number of burn-in MCMC steps")
    p_cross1h.add_argument("--startidx", type=int, default=2, help="Start index in ell bins for diagnostics")
    p_cross1h.add_argument("--endidx", type=int, default=-1, help="End index in ell bins for diagnostics")
    p_cross1h.add_argument(
        "--component-unc-samples",
        type=int,
        default=300,
        help="Number of parameter draws for component uncertainty bands",
    )
    p_cross1h.add_argument("--pred-nsamp", type=int, default=2000, help="Posterior draws used for auto-prediction uncertainty propagation")
    p_cross1h.add_argument(
        "--gal-auto-denominator-mode",
        choices=["shot-subtracted-data", "smooth-model"],
        default="smooth-model",
        help="Denominator used in auto reconstruction: shot-subtracted data or fitted smooth 2h+1h galaxy-auto model",
    )
    p_cross1h.add_argument(
        "--ratio-scaling-mode",
        choices=["direct-ratio", "separate-2h-1h"],
        default="direct-ratio",
        help="Use direct (2h+1h)/(2h+1h) ratio or separate 2h/1h scaling in reconstruction",
    )
    p_cross1h.add_argument("--scale-ell-min", type=float, default=300.0, help="Min ell for cross/auto ratio summary in prediction stage")
    p_cross1h.add_argument("--scale-ell-max", type=float, default=3000.0, help="Max ell for cross/auto ratio summary in prediction stage")
    p_cross1h.add_argument("--shot-ell-min", type=float, default=5.0e4, help="Min ell used to estimate galaxy shot noise in prediction stage")
    p_cross1h.add_argument("--shot-ell-max", type=float, default=8.0e4, help="Max ell used to estimate galaxy shot noise in prediction stage")
    p_cross1h.add_argument(
        "--auto-fig-width",
        type=float,
        default=7.0,
        help="Width of auto prediction figure in inches",
    )
    p_cross1h.add_argument(
        "--auto-fig-height",
        type=float,
        default=3.5,
        help="Height of auto prediction figure in inches",
    )
    p_cross1h.add_argument(
        "--include-dgl-auto-constraints",
        action="store_true",
        default=True,
        help="Overlay best-fit DGL auto constraints on auto prediction panels",
    )
    p_cross1h.add_argument(
        "--no-include-dgl-auto-constraints",
        action="store_false",
        dest="include_dgl_auto_constraints",
        help="Disable DGL auto-constraint overlay",
    )
    p_cross1h.add_argument(
        "--dgl-mode",
        default="sfd_clean",
        help="DGL map mode used to load DGL auto constraints",
    )
    p_cross1h.add_argument(
        "--include-shot-noise-fit",
        action="store_true",
        default=True,
        help="Overlay high-ell shot-noise fit as black dashed line",
    )
    p_cross1h.add_argument(
        "--no-include-shot-noise-fit",
        action="store_false",
        dest="include_shot_noise_fit",
        help="Disable shot-noise dashed-line overlay",
    )
    p_cross1h.add_argument(
        "--mock-igl-zlt1-file",
        default="",
        help="Optional npz file for mock z<1 IGL overlay (keys documented in script helper)",
    )
    p_cross1h.add_argument(
        "--auto-mock-igl-zlt1",
        action="store_true",
        default=True,
        help="Auto-load default z<1 mock IGL overlay from Jordan mock field-average files",
    )
    p_cross1h.add_argument(
        "--no-auto-mock-igl-zlt1",
        action="store_false",
        dest="auto_mock_igl_zlt1",
        help="Disable automatic default z<1 mock IGL overlay",
    )
    p_cross1h.add_argument(
        "--show-lsq-init",
        action="store_true",
        default=True,
        help="Overlay least-squares initialization curve in each component diagnostic panel",
    )
    p_cross1h.add_argument(
        "--no-show-lsq-init",
        action="store_false",
        dest="show_lsq_init",
        help="Disable least-squares initialization curve overlay",
    )
    p_cross1h.add_argument(
        "--good-fit-rchi2-max",
        type=float,
        default=10.0,
        help="Maximum reduced chi2 allowed (all four cross fits must pass) before auto prediction update",
    )
    p_cross1h.add_argument(
        "--no-save-intermediate-fits",
        action="store_true",
        help="Disable saving intermediate fit/corner figures from model stages",
    )

    subparsers.add_parser(
        "compare-desils-cmgs",
        help="Generate DESI-LS with/without CMGs comparison figure",
    )

    p_param = subparsers.add_parser(
        "compare-mock-vs-parametric",
        help="Generate dedicated parametric component figures from fit-result npz file(s)",
    )
    p_param.add_argument("--fit-results-ls", default=None, help="Fit npz for LS")
    p_param.add_argument("--fit-results-hsc", default=None, help="Fit npz for HSC")
    p_param.add_argument(
        "--tracers",
        choices=["both", "ls", "hsc"],
        default="both",
        help="Tracer set to render when fit files are auto-discovered",
    )
    p_param.add_argument(
        "--fit-family",
        default="IHL1hfit_fixshape_newcl_thetacut",
        help="Fit family stem used for auto-discovery in cross_cl_fits",
    )
    p_param.add_argument(
        "--fit-lmax",
        type=int,
        default=50000,
        help="lMax value used for fit-file auto-discovery",
    )
    p_param.add_argument(
        "--fit-search-dir",
        default="data/cross_cl_fits",
        help="Directory containing parametric cross fit .npz outputs",
    )
    p_param.add_argument(
        "--measured-cross-dir",
        default="data/ciber_gal_cross_cls",
        help="Directory containing measured coarse-z cross spectra used for data overlays",
    )
    p_param.add_argument(
        "--mock-pred-basepath",
        default="data/jordan_mocks/v3_boxed_outputs/tiles_10p0deg",
        help="Base path containing mock_ps_pred used for dashed mock overlays in per-z plots",
    )
    p_param.add_argument(
        "--no-mock-overlay",
        action="store_true",
        help="Disable dashed mock-prediction overlays in per-z total comparison figures",
    )

    p_avz = subparsers.add_parser(
        "amplitude-vs-z",
        help="Plot fitted A_2h (and A_1h) vs redshift by instrument with IGL model overlay",
    )
    p_avz.add_argument(
        "--fit-results-ls",
        default=None,
        metavar="PATH",
        help="Parametric cross-fit .npz for DESI-LS (from cross_cl_fits/)",
    )
    p_avz.add_argument(
        "--fit-results-hsc",
        default=None,
        metavar="PATH",
        help="Parametric cross-fit .npz for HSC (from cross_cl_fits/)",
    )
    p_avz.add_argument(
        "--bias-cache-fpath",
        default=None,
        metavar="PATH",
        help="Path to effective_bias_ls_cache.npz (from compute_effective_bias_ls.py). "
             "Enables the b_g-corrected IGL model overlay on the A_2h panel.",
    )
    p_avz.add_argument(
        "--inst-list",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Instruments to include (default: 1 2)",
    )
    p_avz.add_argument(
        "--ylim-2h",
        nargs=2,
        type=float,
        default=None,
        metavar=("LO", "HI"),
        help="Y-axis limits for A_2h panel (default: auto)",
    )
    p_avz.add_argument(
        "--ylim-ihl",
        nargs=2,
        type=float,
        default=None,
        metavar=("LO", "HI"),
        help="Y-axis limits for A_1h panel (default: auto)",
    )
    p_avz.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[6.0, 6.5],
        metavar=("W", "H"),
        help="Figure size in inches (default: 6 6.5)",
    )
    p_avz.add_argument(
        "--legend-ncol",
        type=int,
        default=3,
        help="Number of legend columns (default: 3)",
    )

    p_all = subparsers.add_parser(
        "all",
        help="Run a batch of figure subcommands",
    )
    p_all.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Optional subset of commands to include in all-run",
    )

    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.pred_source == "standard-igl" and not args.pred_basepath:
        raise ValueError("--pred-basepath is required when --pred-source standard-igl")

    if args.pred_source == "standard-igl" and args.pred_basepath:
        _normalize_pred_basepath(args.pred_basepath)



def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    configure_matplotlib(show=args.show)

    try:
        _validate_args(args)
    except Exception as exc:
        print(f"Argument validation error: {exc}")
        return 2

    all_records: List[FigureTiming] = []

    if args.command == "all":
        commands = _default_all_commands()
        if args.include:
            requested = set(args.include)
            commands = [cmd for cmd in commands if cmd in requested]

        # Forecast and rl-vs-z-scale require explicit inputs, so skip by default in all-run.
        commands = [cmd for cmd in commands if cmd not in {"forecast", "rl-vs-z-scale"}]

        for cmd in commands:
            child_ns = argparse.Namespace(**vars(args))
            child_ns.command = cmd
            if cmd == "field-consistency":
                child_ns.variant = "all"
                child_ns.ell_min = 300
                child_ns.ell_max = 10000
            all_records.extend(_dispatch_single(child_ns, cmd))
    else:
        all_records.extend(_dispatch_single(args, args.command))

    outdir = Path(args.outdir).expanduser()
    json_path, csv_path = write_diagnostics(
        all_records,
        outdir=outdir,
        basename=args.diagnostics_basename,
    )

    print_timing_table(all_records)
    print(f"\nSaved diagnostics: {json_path}")
    print(f"Saved diagnostics: {csv_path}")

    failed = [r for r in all_records if r.status != "ok"]
    if failed:
        print("\nFailures:")
        for rec in failed:
            print(f"- {rec.figure_key}: {rec.error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
