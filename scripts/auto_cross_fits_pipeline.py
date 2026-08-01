#!/usr/bin/env python3
"""Pipeline for galaxy auto and CIBER x galaxy cross spectrum fits.

Modes
-----
run_auto        Run galaxy auto fits (run_gal_auto_fits_two_stage) across catalogs/lMax values.
run_cross       Run CIBER x galaxy cross fits (run_gal_cross_fits) across catalogs/lMax values.
plot_auto       Load saved auto .npz files and regenerate amplitude + chi2 comparison plots.
plot_cross      Load saved cross .npz files and regenerate amplitude + chi2 comparison plots.
plot_components Load a single cross .npz and plot the 1h/2h/shot spectral decomposition.
plot_compare_cats
                Two-panel figure comparing A_1h and A_2h vs redshift across catalogs at
                a fixed lMax (--lmax-compare, default 50000).
plot_corr_a1h_a2h
                Plot r(A_2h, A_1h) correlation coefficient vs redshift for all
                catalog/instrument combinations at a fixed lMax (--lmax-compare, default 50000).
plot_sigma_damp Panel figure + heatmap showing sigma_damp (astrometric damping) consistency
                across ell_max and catalog/redshift combinations.
all             Run all modes in order.

Examples
--------
# Re-plot cross fits for HSC without re-running
python scripts/auto_cross_fits_pipeline.py --mode plot_cross --cat HSC --lmax 20000 30000 50000 70000 90000

# Run auto fits for DESILS at two ell_max values
python scripts/auto_cross_fits_pipeline.py --mode run_auto --cat DESILS --lmax 20000 50000

# Compare HSC and DESILS A_1h/A_2h at lMax=50000
python scripts/auto_cross_fits_pipeline.py --mode plot_compare_cats --cat HSC DESILS --lmax-compare 50000

# Plot sigma_damp consistency across lMax values
python scripts/auto_cross_fits_pipeline.py --mode plot_sigma_damp --cat HSC DESILS --lmax 30000 50000 70000 90000

# Full pipeline, forced overwrite
python scripts/auto_cross_fits_pipeline.py --mode all --cat HSC DESILS --lmax 20000 50000 90000 --overwrite
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402  (sets ciber_basepath)
from ciber.theory.cross_ps_parametric_model import (
    plot_fit_fixed_1h_templates,
    run_gal_auto_fits_two_stage,
    run_gal_cross_fits,
    CrossPowerSpectrumModel,
    attach_onehalo_template_to_model,
    resolve_full_param_value,
    expand_fit_samples_to_full_vector,
)
from ciber.theory.onehalo_predict import load_onehalo_results
from ciber.io.ciber_data_utils import load_fit_results_npz
from ciber.plotting.gal_plotting_fns import (
    plot_amplitude_comparison,
    plot_chi2_comparison,
    plot_cross_fit_components_from_file,
    plot_amplitude_chi2_by_instrument,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CAT_CMAP = {"HSC": "RdPu", "DESILS": "Greens"}


_LMAX_COLORS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]


def _savefig(fig, path: Path, fmt: str) -> None:
    """Save figure as PDF or PNG (dpi=300), creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "png":
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=300)
        print(f"saved {path.with_suffix('.png')}")
    else:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        print(f"saved {path.with_suffix('.pdf')}")


def _build_plot_path_with_model(path: Path, args=None, results=None) -> Path:
    """Append an fsat-model suffix to a figure path when metadata is available."""
    suffix_parts = []

    fsat_model = None
    concentration_scale = None
    if results is not None:
        fsat_model = results.get("onehalo_fsat_model", None)
        concentration_scale = results.get("onehalo_concentration_scale", None)
    if fsat_model is None and args is not None:
        fsat_model = getattr(args, "onehalo_fsat_model", None)
    if concentration_scale is None and args is not None:
        concentration_scale = getattr(args, "concentration_scale", None)

    if fsat_model:
        suffix_parts.append(f"fsat{fsat_model}")
    if concentration_scale is not None and concentration_scale != 1.0:
        suffix_parts.append(f"concscale{float(concentration_scale):.2f}".replace('.', 'p'))

    if not suffix_parts:
        return path

    if path.suffix:
        return path.with_name(f"{path.stem}_" + "_".join(suffix_parts) + path.suffix)
    return path.with_name(f"{path.name}_" + "_".join(suffix_parts))


def _normalize_model_components(comps, ell):
    """Return a dict-like component mapping expected by the plotting code."""
    if isinstance(comps, dict):
        return comps

    arr = np.asarray(comps, dtype=float)
    if arr.ndim == 0:
        arr = np.full_like(np.asarray(ell, dtype=float), float(arr), dtype=float)

    return {
        "two_halo": np.zeros_like(arr, dtype=float),
        "one_halo": np.zeros_like(arr, dtype=float),
        "shot_noise": np.zeros_like(arr, dtype=float),
        "total": arr,
    }


def _headstr_tag(headstr: Optional[str]) -> str:
    """Convert headstr to the file-name infix used by the saving functions.

    Cross fits: 'hsc_ilt25.0' -> '_ilt25.0'   (e.g. HSC_coarsez_ilt25.0_cross_...)
    Auto fits use the full headstr; handled separately in _auto_fpath.
    """
    if not headstr:
        return ""
    # Keep everything from 'ilt' onwards, prepend underscore
    if "ilt" in headstr:
        return "_" + headstr[headstr.index("ilt"):]
    return ""

def _auto_fpath(datadir: str, cat: str, fitstr: str, lMax: int, headstr: Optional[str] = None) -> Path:
    """Build path for a galaxy auto fit result file.

    Files saved by run_gal_auto_fits_two_stage embed the headstr between
    'coarsez_' and '_gal_auto_fits_' when headstr is set, e.g.:
      HSC_coarsez_hsc_ilt25.0_gal_auto_fits_two_stage_fixed_1h_lMax=20000.npz
    Without headstr:
      DESILS_coarsez_gal_auto_fits_two_stage_fixed_1h_lMax=20000.npz
    """
    if headstr:
        return Path(datadir) / f"{cat}_coarsez_{headstr}_gal_auto_fits_{fitstr}_lMax={lMax}.npz"
    return Path(datadir) / f"{cat}_coarsez_gal_auto_fits_{fitstr}_lMax={lMax}.npz"


def _cross_fpath(datadir: str, cat: str, headstr: Optional[str], fitstr: str, lMax: int,
                 maskstr: Optional[str] = None) -> Path:
    tag = _headstr_tag(headstr)
    mask_tag = f"_{maskstr}" if maskstr else ""
    return Path(datadir) / f"{cat}_coarsez{tag}{mask_tag}_cross_cl_fits_{fitstr}_lMax={lMax}.npz"


def _load_cross_results_merged_jh14(datadir: str, 
                                    cat: str, 
                                    headstr: Optional[str],
                                    fitstr: str, 
                                    lMax: int, 
                                    lMax_list: Optional[List[int]] = None,
                                    maskstr: Optional[str] = None) -> Optional[dict]:
    """Load cross-fit results, merging JHlt14 z<0.2 with fiducial z>0.2.
    
    For DESILS: always uses JHlt14 for z<0.2, and the specified maskstr for z>0.2.
    For other catalogs: returns fiducial results as-is.
    
    Args:
        maskstr: Mask string to include in the fiducial filename (e.g., 'JHlt16')
    
    Returns merged results dict, or None if files don't exist.
    """
    # For DESILS: ensure z<0.2 is always JHlt14
    if cat == "DESILS":
        # Load JHlt14 for z<0.2
        fpath_jh14 = _cross_fpath(datadir, cat, headstr, fitstr, lMax, maskstr='JHlt14')
        if not fpath_jh14.exists():
            # Fallback: try to load with specified maskstr if JHlt14 doesn't exist
            fpath_fid = _cross_fpath(datadir, cat, headstr, fitstr, lMax, maskstr=maskstr)
            if not fpath_fid.exists():
                return None
            return load_fit_results_npz(str(fpath_fid))
        
        res_jh14 = load_fit_results_npz(str(fpath_jh14))
        
        # If maskstr is already JHlt14, just return JHlt14 results
        if maskstr == 'JHlt14' or maskstr is None:
            return res_jh14
        
        # Load the specified maskstr (e.g., JHlt16) for z>0.2
        fpath_fid = _cross_fpath(datadir, cat, headstr, fitstr, lMax, maskstr=maskstr)
        if not fpath_fid.exists():
            # No fiducial, return JHlt14 as-is
            return res_jh14
        
        res_fid = load_fit_results_npz(str(fpath_fid))
        
        # Merge: z<0.2 from JHlt14, z>0.2 from fiducial (maskstr)
        merged = {}
        # Get the number of z-bins from the fiducial results (has all z-bins)
        # JHlt14 only has 1 z-bin, so we can't use its shape
        n_zbins = res_fid.get('zbinedges').shape[0] - 1 if 'zbinedges' in res_fid else (res_fid['params'].shape[1] if 'params' in res_fid and res_fid['params'].ndim > 1 else 1)
        
        for key in res_jh14.keys():
            val_jh14 = res_jh14[key]
            
            # Arrays to merge (z-dimension is axis 1)
            if key in ['params', 'params_err', 'params_16', 'params_84', 'params_95',
                       'chisq', 'reduced_chisq', 'ndof', 'samples', 'samples_fitted']:
                # Start with fiducial results (has all z-bins) and replace z<0.2 with JHlt14
                val_fid = res_fid[key]
                merged[key] = np.array(val_fid, copy=True, dtype=object) if isinstance(val_fid, np.ndarray) and val_fid.dtype == object else np.array(val_fid, copy=True)
                # Replace z<0.2 (zidx=0) with JHlt14
                merged[key][:, 0] = val_jh14[:, 0] if val_jh14.ndim > 1 else val_jh14
            else:
                # Metadata unchanged (take from fiducial which has correct zbinedges)
                merged[key] = res_fid.get(key, val_jh14)
        
        return merged
    else:
        # For non-DESILS catalogs: return fiducial as-is
        fpath_fid = _cross_fpath(datadir, cat, headstr, fitstr, lMax, maskstr=maskstr)
        if not fpath_fid.exists():
            return None
        
        return load_fit_results_npz(str(fpath_fid))


def _ifield_list(cat: str, args: argparse.Namespace) -> List[int]:
    return args.ifield_hsc if cat == "HSC" else args.ifield_ls


def _lmax_configs(all_res, inst: int, cat: str, lmax_list: List[int], fit_type: str = "auto") -> list:
    """Build configs list for plot_amplitude_comparison / plot_chi2_comparison."""
    lams = {1: 1.1, 2: 1.8}
    if fit_type == "auto":
        first_label = f"{cat} Auto ($\\ell<{lmax_list[0]}$)"
    else:
        first_label = f"{cat} × CIBER {lams[inst]} μm ($\\ell<{lmax_list[0]}$)"

    configs = []
    for idx, (res, lMax) in enumerate(zip(all_res, lmax_list)):
        label = first_label if idx == 0 else f"($\\ell<{lMax}$)"
        configs.append({"results": res, "inst": inst, "label": label, "color": _LMAX_COLORS[idx]})
    return configs


# ---------------------------------------------------------------------------
# Fit stages
# ---------------------------------------------------------------------------

def _run_auto_fits(args: argparse.Namespace) -> None:
    figbasedir = os.path.join(args.figdir, "gal_auto_fits_two_stage/")
    for cat in args.cat:
        ifield_list = _ifield_list(cat, args)
        for lMax in args.lmax:
            fpath = _auto_fpath(args.datadir_auto, cat, args.fitstr_auto, lMax,
                                headstr=args.headstr if cat == "HSC" else None)
            if not args.overwrite and fpath.exists():
                print(f"[run_auto] skipping {fpath.name} (already exists)")
                continue
            print(f"[run_auto] {cat} lMax={lMax}")
            run_gal_auto_fits_two_stage(
                inst_list=[1, 2],
                cat=cat,
                ifield_list=ifield_list,
                zbinedges=args.zbinedges,
                lMax_fit=lMax,
                chi2_eval_max=lMax,
                fitstr=args.fitstr_auto,
                figbasedir=figbasedir,
                save_figs=True,
                save_results=True,
                file_fpath=fpath.name,
                ihl_1h_params_path=args.ihl_params,
                nwalkers=args.nwalkers,
                nsteps_stage1=args.nsteps1,
                nsteps_stage2=args.nsteps2,
                nburn_stage1=args.nburn1,
                nburn_stage2=args.nburn2,
                headstr=args.headstr if cat == "HSC" else None,
                fmask=args.fmask,
            )


# ---------------------------------------------------------------------------
# Reference tables
# ---------------------------------------------------------------------------

def _make_parameter_priors_table(args: argparse.Namespace) -> None:
    """Generate a LaTeX table documenting full model parameters and their priors.
    
    Output: {args.figdir}/{args.fitstr_cross}/parameter_priors_table.tex
    """
    outdir = Path(args.figdir) / args.fitstr_cross
    outdir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"\centering")
    lines.append(r"\caption{Full model parameters and priors for phenomenological cross-spectrum fits.}")
    lines.append(r"\label{tab:cross_fit_parameters}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Parameter} & \textbf{Description} & \textbf{Prior Type} & \textbf{Bounds/Values} \\")
    lines.append(r"\midrule")
    
    lines.append(r"$A_{2h}$ & 2-halo amplitude & Uniform & $[0, 10]$ \\")
    lines.append(r"$A_{1h}$ & 1-halo amplitude & Uniform & $[0, 10]$ \\")
    
    lines.append(r"\multirow{2}{*}{$\mu_{1h}$} & \multirow{2}{*}{Peak scale (log-normal)} & \multirow{2}{*}{Gaussian\textsuperscript{a}} & \multirow{2}{*}{$[\ln(1000), \ln(10000)]$} \\")
    
    lines.append(r"\multirow{2}{*}{$\sigma_{1h}$} & \multirow{2}{*}{Width (log-normal $\sigma$)} & \multirow{2}{*}{Gaussian\textsuperscript{a}} & \multirow{2}{*}{$[0.2, 1.2]$} \\")
    
    lines.append(r"$A_{\mathrm{shot}}$ & Shot noise amplitude & Uniform & $[0, 100]$ \\")
    lines.append(r"$\sigma_{\mathrm{damp}}$ & Astrometric damping & Uniform & $[0.1, 4.0]$ arcsec \\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\\\[0.5em\]")
    lines.append(r"{\small \textsuperscript{a}Gaussian priors (mean, std) set from IHL-derived one-halo properties if available; otherwise flat priors used.}")
    lines.append(r"\end{table}")

    tex_content = "\n".join(lines) + "\n"
    outpath = outdir / "parameter_priors_table.tex"
    outpath.write_text(tex_content)
    print(f"[param_priors_table] written → {outpath}")


# ---------------------------------------------------------------------------
# IGL A_2h prediction helper
# ---------------------------------------------------------------------------

# Default IGL prediction headstrs per catalog (no CIBERfidmask suffix so z-bin
# files with 0.2-wide bins match the standard run zbinedges)
_IGL_DEFAULT_HEADSTR = {
    "DESILS": "sdss_z_lt_22.0",
    "HSC": "hsc_i_lt_25.0",
}


def _bi_model_suffix(bi_model: str) -> str:
    """Return the fitstr suffix for a given b_I(z) model name."""
    return {"constant": "", "linear": "_biLinear", "quadratic": "_biQuadratic"}.get(bi_model, "")


def _bi_function(bi_model: str, z: float) -> float:
    """Evaluate the IHL brightness bias b_I at redshift z.

    Parameters
    ----------
    bi_model : str
        One of 'constant' (b_I=1), 'linear' (1+0.6z), 'quadratic' ((1+z)^2).
    z : float
        Redshift.

    Returns
    -------
    float
        b_I(z) value.
    """
    if bi_model == "linear":
        return 1.0 + 0.6 * z
    elif bi_model == "quadratic":
        return (1.0 + z) ** 2
    else:  # constant / default
        return 1.0


def _compute_di_dz_upper_limits(A_2h_ul, bg_dndz_pred, z_centers, zbinedges):
    """Convert A_2h upper limits to dI/dz upper limits by dividing by bias terms and bin width.
    
    Parameters
    ----------
    A_2h_ul : ndarray, shape (n_inst, n_zbin)
        Upper limit values from MCMC (params_95).
    bg_dndz_pred : ndarray, shape (n_inst, n_zbin)
        Model predictions for bin-integrated b_g * dN/dz.
    z_centers : ndarray, shape (n_zbin,)
        Redshift bin centers.
    zbinedges : ndarray, shape (n_zbin+1,)
        Redshift bin edges.
    
    Returns
    -------
    di_dz_dict : dict
        Dictionary mapping b_I model names to dI/dz upper limit arrays, shape (n_inst, n_zbin).
    """
    di_dz_dict = {}
    dz_array = np.diff(zbinedges)  # bin widths
    
    for bi_model in ("constant", "linear", "quadratic"):
        # Compute b_I(z) for each bin center
        bi_vals = np.array([_bi_function(bi_model, z) for z in z_centers])
        
        # Convert from bin-integrated to differential form by dividing by Δz
        bg_dndz_eff = bg_dndz_pred / dz_array[np.newaxis, :]  # shape (n_inst, n_zbin)
        
        # Compute dI/dz upper limits
        denominator = bg_dndz_eff * bi_vals[np.newaxis, :]
        di_dz_dict[bi_model] = np.where(denominator > 0, A_2h_ul / denominator, np.nan)
    
    return di_dz_dict


def _compute_igl_a2h_predictions(cat, zbinedges, inst_list,
                                  jmock_basedir="data/jordan_mocks/v2/",
                                  bias_cache_fpath=None, headstr=None,
                                  bi_model="constant",
                                  a2h_cache_fpath=None):
    """Compute IGL-predicted A_2h amplitudes (bias-corrected) per inst and z-bin.

    Parameters
    ----------
    cat : str
        Catalog name, e.g. 'DESILS' or 'HSC'.
    zbinedges : array-like
        Redshift bin edges, e.g. [0.0, 0.2, 0.4, 0.6, 0.8, 1.0].
    inst_list : list of int
        CIBER instrument indices, e.g. [1, 2].
    jmock_basedir : str
        Base directory for Jordan mock predictions.
    bias_cache_fpath : str or None
        Path to ``effective_bias_ls_cache.npz`` for DESILS bias polynomial.
        If None for DESILS, galaxy bias is assumed = 1 (no correction).
    headstr : str or None
        Prediction headstring override.  Defaults to catalog-specific value.
    bi_model : str
        IHL brightness-bias model.  One of 'constant' (b_I=1, default),
        'linear' (b_I = 1+0.6z), or 'quadratic' (b_I = (1+z)^2).
        The predicted A_2h is scaled by b_I(z_center) before being used.

    Returns
    -------
    np.ndarray, shape (n_inst, n_zbin)
        Bias-corrected A_2h predictions.  NaN where file not found.
    """
    zbinedges = np.asarray(zbinedges, dtype=float)
    n_zbins = len(zbinedges) - 1
    n_inst = len(inst_list)
    a2h_arr = np.full((n_inst, n_zbins), np.nan)
    a2h_gal_auto = np.full((n_inst, n_zbins), np.nan)

    z_centers = 0.5 * (zbinedges[:-1] + zbinedges[1:])

    # ---------------------------------------------------------------
    # Fast path: load pre-computed A_2h values from JSON cache
    # ---------------------------------------------------------------
    if a2h_cache_fpath is not None and os.path.exists(a2h_cache_fpath):
        import json
        with open(a2h_cache_fpath) as _f:
            _cache = json.load(_f)
        _cache_results = _cache.get("results", [])
        # Build lookup: (inst, zmin_str, zmax_str) -> a2h
        _lkup = {}
        for _entry in _cache_results:
            _p = _entry.get("path", "")
            import re as _re
            _m = _re.search(r'TM(\d).*?zmin=([\d.]+)_zmax=([\d.]+)', _p)
            if _m:
                _lkup[(int(_m.group(1)), _m.group(2), _m.group(3))] = _entry["a2h"]

        # Galaxy bias per z-bin
        if cat == "DESILS":
            if bias_cache_fpath is not None and os.path.exists(bias_cache_fpath):
                _bc = np.load(bias_cache_fpath, allow_pickle=False)
                _coeffs = np.asarray(_bc["coarse_poly_coeffs"])
                b_g = np.poly1d(_coeffs)(z_centers)
            else:
                b_g = np.ones(n_zbins)
        else:
            b_g = 1.0 + 0.84 * z_centers
        
        # Print bias values for debugging
        bias_str = ", ".join(f"z={z:.2f}: b_g={b:.4f}" for z, b in zip(z_centers, b_g))
        print(f"[igl_a2h] Galaxy bias for {cat}: {bias_str}")


        for inst_idx, inst in enumerate(inst_list):
            for zidx in range(n_zbins):
                zmin_s = f"{zbinedges[zidx]:.1f}"
                zmax_s = f"{zbinedges[zidx + 1]:.1f}"
                A_2h_raw = _lkup.get((inst, zmin_s, zmax_s), np.nan)
                if np.isfinite(A_2h_raw):
                    a2h_arr[inst_idx, zidx] = (
                        b_g[zidx] * _bi_function(bi_model, z_centers[zidx]) * max(A_2h_raw, 0.0)
                    )
                    a2h_gal_auto[inst_idx, zidx] = np.sqrt(b_g[zidx]) * max(A_2h_raw, 0.0)

                else:
                    print(f"[igl_a2h] Warning: no cache entry for TM{inst} zmin={zmin_s} zmax={zmax_s}")

        bi_str = f" [b_I model: {bi_model}]" if bi_model != "constant" else ""
        if bi_model != "constant":
            bi_vals = ", ".join(f"z={z:.2f}: b_I={_bi_function(bi_model, z):.4f}" for z in z_centers)
            print(f"[igl_a2h] IHL brightness bias: {bi_vals}")
        print(f"[igl_a2h] A_2h predictions for {cat} (inst_list={inst_list}){bi_str} [from cache]:")
        for inst_idx, inst in enumerate(inst_list):
            vals = ", ".join(f"{v:.4f}" if np.isfinite(v) else "NaN"
                             for v in a2h_arr[inst_idx])
            print(f"  TM{inst}: [{vals}]")
        return a2h_arr, a2h_gal_auto

    # ---------------------------------------------------------------
    # Slow path: load from individual .npz prediction files
    # ---------------------------------------------------------------

    # Galaxy bias per z-bin
    if cat == "DESILS":
        if bias_cache_fpath is not None and os.path.exists(bias_cache_fpath):
            cache = np.load(bias_cache_fpath, allow_pickle=False)
            coeffs = np.asarray(cache["coarse_poly_coeffs"])
            b_g = np.poly1d(coeffs)(z_centers)
        else:
            if bias_cache_fpath is not None:
                print(f"[igl_a2h] Warning: bias cache not found at {bias_cache_fpath}, using b_g=1")
            b_g = np.ones(n_zbins)
    else:  # HSC
        b_g = 1.0 + 0.84 * z_centers
    
    # Print bias values for debugging
    bias_str = ", ".join(f"z={z:.2f}: b_g={b:.4f}" for z, b in zip(z_centers, b_g))
    print(f"[igl_a2h] Galaxy bias for {cat}: {bias_str}")

    # Try headstrs with/without CIBERfidmask (same order as run_amplitude_vs_z)
    _base_headstr = headstr if headstr is not None else _IGL_DEFAULT_HEADSTR.get(cat, _IGL_DEFAULT_HEADSTR["DESILS"])
    _headstr_candidates = [_base_headstr + "_CIBERfidmask", _base_headstr]

    from ciber.theory.cl_predictions import grab_ciber_cross_vs_z_predfpaths

    for inst_idx, inst in enumerate(inst_list):
        # Find which headstr has existing files, like run_amplitude_vs_z does
        pred_fpaths = None
        for hs in _headstr_candidates:
            candidate = grab_ciber_cross_vs_z_predfpaths(
                inst_list=[inst],
                zbinedges=list(zbinedges),
                jmock_basedir=jmock_basedir,
                headstr=hs,
            )[0]
            if any(os.path.exists(p) for p in candidate):
                pred_fpaths = candidate
                break

        if pred_fpaths is None:
            print(f"[igl_a2h] Warning: no prediction files found for {cat} TM{inst} "
                  f"in {jmock_basedir} (tried: {_headstr_candidates})")
            continue

        for zidx, fpath in enumerate(pred_fpaths):
            if not os.path.exists(fpath):
                print(f"[igl_a2h] Warning: prediction file not found: {fpath}")
                continue
            d = np.load(fpath)
            lb_m = np.asarray(d["lb"], dtype=float)
            cl_m = np.asarray(d["cross"], dtype=float)
            pf_m = lb_m * (lb_m + 1.0) / (2.0 * np.pi)
            dl_m = pf_m * cl_m

            # process galaxy auto to get divisor for A2h^IG
            cl_g_nobias = np.asarray(d["gal_auto"], dtype=float)
            dl_g_nobias = pf_m * cl_g_nobias
            shot_mask = (lb_m >= 30000.) & (lb_m <= 80000.) & np.isfinite(dl_m)
            pf_shot = lb_m[shot_mask] * (lb_m[shot_mask] + 1.0) / (2.0 * np.pi)
            A_shot = float(np.nanmean(dl_g_nobias[shot_mask] / pf_shot)) if shot_mask.any() else 0.0
            ell_norm = 300.
            which_lb_m = np.argmin(np.abs(lb_m - ell_norm))

            # amplitude of sn subtracted ps at ell=300
            # A_2h_matter = float(dl_g_nobias[which_lb_m] - A_shot * pf_m[which_lb_m]) if np.isfinite(dl_g_nobias[which_lb_m]) else 0.0
            A_2h_matter = float(dl_g_nobias[which_lb_m]) if np.isfinite(dl_g_nobias[which_lb_m]) else 0.0

            # A_2h_matter = max(float(np.nanmax(dl_g_nobias[twoh_mask] - A_shot * pf_m[twoh_mask])), 0.0) if twoh_mask.any() else 0.0

            # Estimate shot noise from high-ell tail
            shot_mask = (lb_m >= 30000.0) & (lb_m <= 80000.0) & np.isfinite(dl_m)
            if shot_mask.any():
                A_shot_est = float(np.nanmean(dl_m[shot_mask] / pf_m[shot_mask]))
            else:
                A_shot_est = 0.0

            # IGL 2-halo amplitude from low-ell (shot-subtracted)
            twoh_mask = (lb_m <= 3000.0) & np.isfinite(dl_m)
            if twoh_mask.any():
                A_2h_raw = float(np.nanmean(dl_m[twoh_mask] - A_shot_est * pf_m[twoh_mask]))
            else:
                A_2h_raw = 0.0

            a2h_arr[inst_idx, zidx] = b_g[zidx] * _bi_function(bi_model, z_centers[zidx]) * max(A_2h_raw, 0.0)

            a2h_gal_auto[inst_idx, zidx] = np.sqrt(b_g[zidx]) * A_2h_matter


    bi_str = f" [b_I model: {bi_model}]" if bi_model != "constant" else ""
    if bi_model != "constant":
        bi_vals = ", ".join(f"z={z:.2f}: b_I={_bi_function(bi_model, z):.4f}" for z in z_centers)
        print(f"[igl_a2h] IHL brightness bias: {bi_vals}")
    print(f"[igl_a2h] A_2h predictions for {cat} (inst_list={inst_list}){bi_str}:")
    for inst_idx, inst in enumerate(inst_list):
        vals = ", ".join(f"{v:.4f}" if np.isfinite(v) else "NaN"
                         for v in a2h_arr[inst_idx])
        print(f"  TM{inst}: [{vals}]")
    return a2h_arr, a2h_gal_auto


def _run_cross_fits(args: argparse.Namespace) -> None:
    # With fix_ihl_1h_shape=True and a valid ihl_params file, mu_1h and sigma_1h
    # are fixed per-zbin from the precomputed IHL template fits, giving a
    # 3-parameter MCMC: [A_2h, A_1h, A_shot]. prior_bounds=None uses the
    # correct 3-param defaults built inside fit_model_mcmc.
    # If use_one_halo=False, fits only 2h+shot (no 1h term).
    # If use_two_halo=False, fits only 1h+shot (no 2h term).
    fitstr_to_use = args.fitstr_cross
    if not args.use_two_halo:
        fitstr_to_use = args.fitstr_cross + "_no2h"
    elif not args.use_one_halo:
        fitstr_to_use = args.fitstr_cross + "_no1h"
    elif getattr(args, "fix_a2h_igl", False):
        bi_model = getattr(args, "bi_model", "constant")
        fitstr_to_use = args.fitstr_cross + "_fixA2h_IGL" + _bi_model_suffix(bi_model)

    # When --combined-zbin is set, treat the full 0<z<1 range as a single bin
    zbinedges_use = np.array([args.zbinedges[0], args.zbinedges[-1]]) \
        if args.combined_zbin else args.zbinedges

    # Load effective mu_1h/sigma_1h from cache when using combined z-bin
    mu_1h_override = None
    sigma_1h_override = None
    if args.combined_zbin:
        try:
            from ciber.theory.ihl_1h_template_cache import OneHaloTemplateCache
            cache = OneHaloTemplateCache()
            mu_1h_override, sigma_1h_override = cache.get_effective_lognormal_params(slope=1.0)
            print(f"[run_cross] Using effective 1h params from cache: "
                  f"mu_1h={mu_1h_override:.4f} (ell_peak≈{np.exp(mu_1h_override):.0f}), "
                  f"sigma_1h={sigma_1h_override:.4f}")
        except Exception as e:
            print(f"[run_cross] Warning: Could not load effective 1h params from cache: {e}")

    for cat in args.cat:
        ifield_list = _ifield_list(cat, args)

        # Pre-compute IGL A_2h predictions for all inst/zbins if requested
        a2h_fixed_arr = None
        if getattr(args, "fix_a2h_igl", False):
            inst_list_for_cat = [1, 2]  # default; run_gal_cross_fits uses same default
            jmock_basedir = getattr(args, "igl_pred_basedir", "data/jordan_mocks/v2/")
            igl_headstr = getattr(args, "igl_pred_headstr", None)
            bi_model = getattr(args, "bi_model", "constant")
            a2h_fixed_arr, _ = _compute_igl_a2h_predictions(
                cat=cat,
                zbinedges=zbinedges_use,
                inst_list=inst_list_for_cat,
                jmock_basedir=jmock_basedir,
                bias_cache_fpath=args.bias_cache_fpath,
                headstr=igl_headstr,
                bi_model=bi_model,
            )

            # a2h_gal_arr  # for computing dI/dz upper limits
        

        for lMax in args.lmax:
            fpath = _cross_fpath(args.datadir_cross, cat, args.headstr if cat == "HSC" else None,
                                 fitstr_to_use, lMax, maskstr=getattr(args, 'maskstr', None))
            if not args.overwrite and fpath.exists():
                print(f"[run_cross] skipping {fpath.name} (already exists)")
                continue
            print(f"[run_cross] {cat} lMax={lMax}")
            run_gal_cross_fits(
                cat=cat,
                ifield_list=ifield_list,
                save_results=True,
                file_fpath=fpath.name,
                zbinedges=zbinedges_use,
                lMax_fit=lMax,
                use_ihl_1h_params=True,
                fix_ihl_1h_shape=True,
                ihl_1h_params_path=args.ihl_params,
                mu_1h_fixed_override=mu_1h_override,
                sigma_1h_fixed_override=sigma_1h_override,
                fitstr=fitstr_to_use,
                save_figs=True,
                use_astrometry_damping=args.use_damping,
                chi2_lim=[-5, 5],
                headstr=args.headstr if cat == "HSC" else None,
                maskstr=getattr(args, 'maskstr', None),
                use_one_halo=args.use_one_halo,
                use_two_halo=args.use_two_halo,
                prior_bounds=None,
                uniform_weight_ell=args.uniform_weight_ell,
                A_2h_fixed_arr=a2h_fixed_arr,
                use_linear_2h=args.use_linear_2h,
                sigma_damp_fixed=args.sigma_damp_fixed,
                onehalo_output_dir=args.onehalo_dir,
                onehalo_generate_type=args.onehalo_generate_type,
                onehalo_fsat_model=args.onehalo_fsat_model,
                onehalo_concentration_scale=args.concentration_scale,
                onehalo_population=getattr(args, 'onehalo_population', 'combined'),
                onehalo_fit_popmix=getattr(args, 'onehalo_fit_popmix', False),
                nwalkers=args.nwalkers,
            )


# ---------------------------------------------------------------------------
# Plot stages
# ---------------------------------------------------------------------------

def _plot_auto(args: argparse.Namespace) -> None:
    figdir = Path(args.figdir) / args.fitstr_auto

    for cat in args.cat:
        all_res = []
        for lMax in args.lmax:
            fpath = _auto_fpath(args.datadir_auto, cat, args.fitstr_auto, lMax,
                                headstr=args.headstr if cat == "HSC" else None)
            if not fpath.exists():
                print(f"[plot_auto] missing {fpath}, skipping lMax={lMax} for {cat}")
                continue
            all_res.append((lMax, load_fit_results_npz(str(fpath))))

        if not all_res:
            print(f"[plot_auto] no results found for {cat}, skipping")
            continue

        lmax_list = [lm for lm, _ in all_res]
        res_list = [r for _, r in all_res]
        cmap_name = _CAT_CMAP.get(cat, "Greens")

        for inst in [1, 2]:
            configs = _lmax_configs(res_list, inst, cat, lmax_list, fit_type="auto")

            fig = plot_amplitude_comparison(
                configs, save_path=None, ylim_2h=[-0.005, 0.02], ylim_ihl=[-0.02, 1.5],
                legend_ncol=2, figsize=(6, 6), bbox_to_anchor=[0.02, 1.45],
                use_cmap=True, cmap_name=cmap_name,
            )
            _savefig(fig, figdir / f"gal_auto_fit_{args.fitstr_auto}_{cat}_CIBERTM{inst}_vs_lMax", args.fig_fmt)
            plt.close(fig)

            fig = plot_chi2_comparison(
                configs, save_path=None, figsize=(5.5, 4), legend_ncol=2,
                bbox_to_anchor=[0.0, 1.45], ylim_chi2=[0.0, 10],
                use_cmap=True, cmap_name=cmap_name,
            )
            _savefig(fig, figdir / f"chi2_reduced_gal_auto_{args.fitstr_auto}_{cat}_CIBERTM{inst}_vs_lMax", args.fig_fmt)
            plt.close(fig)


def _plot_cross(args: argparse.Namespace) -> None:
    figdir = Path(args.figdir) / args.fitstr_cross

    for cat in args.cat:
        headstr = args.headstr if cat == "HSC" else None
        all_res = []
        for lMax in args.lmax:
            results = _load_cross_results_merged_jh14(
                args.datadir_cross, cat, headstr, args.fitstr_cross, lMax, maskstr=args.maskstr
            )
            if results is None:
                fpath = _cross_fpath(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax, maskstr=args.maskstr)
                print(f"[plot_cross] missing {fpath}, skipping lMax={lMax} for {cat}")
                continue
            all_res.append((lMax, results))

        if not all_res:
            print(f"[plot_cross] no results found for {cat}, skipping")
            continue

        lmax_list = [lm for lm, _ in all_res]
        res_list = [r for _, r in all_res]
        cmap_name = _CAT_CMAP.get(cat, "Purples")

        for inst in [1, 2]:
            ylim_2h = [-0.02, 0.5]
            configs = _lmax_configs(res_list, inst, cat, lmax_list, fit_type="cross")

            fig = plot_amplitude_comparison(
                configs, save_path=None, ylim_2h=ylim_2h, ylim_ihl=[-0.02, 1.5],
                legend_ncol=2, figsize=(6, 6), bbox_to_anchor=[0.02, 1.45],
                use_cmap=True, cmap_name=cmap_name,
            )
            _savefig(fig, figdir / f"cl_fit_{args.fitstr_cross}_{cat}_CIBERTM{inst}_vs_lMax", args.fig_fmt)
            plt.close(fig)

            fig = plot_chi2_comparison(
                configs, save_path=None, figsize=(5.5, 4), legend_ncol=2,
                bbox_to_anchor=[0.0, 1.45], ylim_chi2=[0.0, 4],
                use_cmap=True, cmap_name=cmap_name,
            )
            _savefig(fig, figdir / f"chi2_reduced_{args.fitstr_cross}_{cat}_CIBERTM{inst}_vs_lMax", args.fig_fmt)
            plt.close(fig)


def _plot_components(args: argparse.Namespace) -> None:
    figdir = Path(args.figdir) / args.fitstr_cross

    for cat in args.cat:
        headstr = args.headstr if cat == "HSC" else None
        fpath = _cross_fpath(args.datadir_cross, cat, headstr, args.fitstr_cross, args.lmax_components, maskstr=args.maskstr)
        if not fpath.exists():
            print(f"[plot_components] missing {fpath}, skipping {cat}")
            continue

        stem = figdir / f"{cat}_cross_components_{args.fitstr_cross}_lMax={args.lmax_components}"
        stem = _build_plot_path_with_model(stem, args=args)
        fig, _ = plot_cross_fit_components_from_file(
            str(fpath),
            zbinedges=args.zbinedges,
            inst_list=[1, 2],
            cat=cat,
            save_path=None,
            figsize=(7, 8),
        )
        _savefig(fig, stem, args.fig_fmt)
        plt.close(fig)


def _plot_param(ax, z_centers, vals, errs, x_offset, color, marker, label, params_95=None):
    """Plot one parameter trace with 2-sigma upper-limit handling.

    Upper limits are drawn as downward arrows from the 95th-percentile value
    (or mean + 2*std) to exactly y=0, so they never extend below the axis.
    """
    is_ul = (vals - 2 * errs) <= 0

    det = ~is_ul
    if np.any(det):
        ax.errorbar(
            z_centers[det] + x_offset, vals[det], yerr=errs[det],
            fmt=marker, color=color, label=label,
            markersize=5, capsize=4, capthick=1.5, alpha=0.85, linestyle="none",
        )
        label = None  # don't repeat in legend for upper-limit points

    if np.any(is_ul):
        ul_vals = params_95[is_ul] if params_95 is not None else vals[is_ul] + 2 * errs[is_ul]
        xs = z_centers[is_ul] + x_offset
        # Plot marker at the upper-limit value
        ax.plot(xs, ul_vals, marker="v", color=color, label=label,
                markersize=5, alpha=0.85, linestyle="none")
        # Draw arrow from ul_val down to exactly y=0 for each point
        for x, y_top in zip(xs, ul_vals):
            ax.annotate(
                "", xy=(x, 0.0), xytext=(x, y_top),
                arrowprops=dict(arrowstyle="-|>", color=color, alpha=0.85, lw=1.2),
            )


def _igl_pred_path(mock_basepath: Optional[str], cat: str, inst: int,
                   zlo: float, zhi: float) -> Optional[Path]:
    """Return path to v3 boxed mock IGL prediction, or None if not found."""
    if mock_basepath is None:
        return None
    heads = (
        ["sdss_z_lt_22.0_CIBERfidmask", "sdss_z_lt_22.0"]
        if cat == "DESILS" else
        ["hsc_i_lt_25.0_CIBERfidmask", "hsc_i_lt_25.0", "hsc_ilt25.0"]
    )
    base = Path(mock_basepath) / "mock_ps_pred" / f"TM{inst}" / "field_average"
    for head in heads:
        fp = base / f"pred_cls_TM{inst}_{head}_zmin={zlo}_zmax={zhi}.npz"
        if fp.exists():
            return fp
    return None


def _plot_fit_spectra(args: argparse.Namespace) -> None:
    """Plot power spectrum fits (data + model + components) for each redshift bin and instrument.

    Uses plot_fit_fixed_1h_templates — the same function used during fitting.
    Reloads data the same way as during fitting to ensure consistency.
    """
    from ciber.theory.cross_ps_parametric_model import plot_fit_fixed_1h_templates
    from ciber.cross_correlation.galaxy_cross import collect_ciber_gal_vs_redshift

    figdir = Path(args.figdir) / args.fitstr_cross / "spectra"

    for cat in args.cat:
        headstr = args.headstr if cat == "HSC" else None
        for lMax in args.lmax:
            results = _load_cross_results_merged_jh14(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax, maskstr=args.maskstr)
            if results is None:
                fpath = _cross_fpath(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax, maskstr=args.maskstr)
                print(f"[plot_fit_spectra] missing {fpath}, skipping {cat} lMax={lMax}")
                continue

            zbinedges = results["zbinedges"]
            inst_list = list(results["inst_list"])
            lams = {1: 1.1, 2: 1.8}

            # Reload data exactly as run_gal_cross_fits does for each catalog
            catname = "LS" if cat == "DESILS" else "HSC"
            ifield_list = args.ifield_hsc if cat == "HSC" else args.ifield_ls

            if cat == "DESILS":
                res_ps = collect_ciber_gal_vs_redshift(
                    catname, subtract_randoms=True, inst_list=inst_list,
                    zbinedges=zbinedges, maskstr='JHlt16_wFFerr', subtract_sn=False,
                    tl_pix_correct=True, ifield_list=ifield_list,
                )
                # Replace z<0.2 bin (zidx=0) with JHlt14 data to match the fit
                res_jh14 = collect_ciber_gal_vs_redshift(
                    catname, subtract_randoms=True, inst_list=inst_list,
                    zbinedges=[zbinedges[0], zbinedges[1]], maskstr='JHlt14_wFFerr',
                    subtract_sn=False, tl_pix_correct=True,
                    ifield_list=ifield_list,
                )
                full_cl_cross = res_ps['full_cl_cross'].copy()
                full_clerr_cross = res_ps['full_clerr_cross'].copy()
                full_cl_cross[:, 0, :] = res_jh14['full_cl_cross'][:, 0, :]
                full_clerr_cross[:, 0, :] = res_jh14['full_clerr_cross'][:, 0, :]
            else:  # HSC
                res_ps = collect_ciber_gal_vs_redshift(
                    catname, subtract_randoms=True, inst_list=inst_list,
                    zbinedges=zbinedges, maskstr=None, subtract_sn=False,
                    tl_pix_correct=True, ifield_list=ifield_list,
                    headstr=headstr, with_ff_err=True,
                )
                full_cl_cross = res_ps['full_cl_cross']
                full_clerr_cross = res_ps['full_clerr_cross']

            lb = res_ps['lb']

            pf_data = lb * (lb + 1) / (2 * np.pi)

            # Same trim as in run_gal_cross_fits before passing to plot_fit_fixed_1h_templates
            startidx, endidx = 2, -1
            lb_fit = lb[startidx:endidx]

            for inst_idx, inst in enumerate(inst_list):
                for zidx in range(len(zbinedges) - 1):
                    zlo, zhi = zbinedges[zidx], zbinedges[zidx + 1]
                    zcen = 0.5 * (zlo + zhi)

                    dl_data    = pf_data * full_cl_cross[inst_idx, zidx]
                    dlerr_data = pf_data * full_clerr_cross[inst_idx, zidx]

                    data_dl    = dl_data[startidx:endidx]
                    data_dlerr = dlerr_data[startidx:endidx]

                    params     = results["params"][inst_idx, zidx, :]
                    params_err = results["params_err"][inst_idx, zidx, :]
                    n_params_stored = int(np.sum(~np.isnan(params)))
                    params     = params[:n_params_stored]
                    params_err = params_err[:n_params_stored]

                    # Get ndof from saved results (already correctly calculated in fit_model_mcmc)
                    ndof_correct = int(results["ndof"][inst_idx, zidx]) if results.get("ndof") is not None else len(lb_fit) - n_params_stored

                    # Detect damping from param count: parametric with damping has 6 params
                    # [A_2h, A_1h, mu_1h, sigma_1h, A_shot, sigma_damp]
                    # without damping has 5. Inspect fitted param names when available.
                    pnf = results.get("param_names_fitted", None)
                    pnf_bin = pnf[inst_idx, zidx] if pnf is not None else None
                    use_damping = (pnf_bin is not None and
                                   any("damp" in str(p).lower() for p in pnf_bin))

                    # Reconstruct fit_result. Always force ihl_templates=None so
                    # plot_fit_fixed_1h_templates takes the parametric branch, which
                    # handles both 5-param (no damping) and 6-param (with damping) cases.
                    samples_bin = results.get("samples", np.empty((0,)))[inst_idx, zidx]
                    fit_result = {
                        "params":                params,
                        "params_err":            params_err,
                        "chisq":                 float(results["chisq"][inst_idx, zidx]),
                        "reduced_chisq":         float(results["reduced_chisq"][inst_idx, zidx]),
                        "ndof":                  ndof_correct,
                        "z_value":               zcen,
                        "use_single_slope":      None,
                        "one_halo_params_dict":  None,
                        "sigma_fixed":           None,
                        "use_astrometry_damping": use_damping,
                        "ihl_templates":         None,
                        "template_names":        None,
                        "samples":               samples_bin if samples_bin is not None and len(np.asarray(samples_bin).shape) > 0 else None,
                        "onehalo_mode":          bool(results.get("onehalo_mode", False)),
                        "onehalo_output_dir":    results.get("onehalo_output_dir", ""),
                        "onehalo_generate_type": results.get("onehalo_generate_type", "bulk"),
                        "onehalo_fsat_model":    results.get("onehalo_fsat_model", "single"),
                        "onehalo_population":    results.get("onehalo_population", getattr(args, 'onehalo_population', 'combined')),
                        "onehalo_fit_popmix":    bool(results.get("onehalo_fit_popmix", getattr(args, 'onehalo_fit_popmix', False))),
                        "onehalo_concentration_scale": float(results.get("onehalo_concentration_scale", getattr(args, 'concentration_scale', 1.0))),
                        "inst":                  int(inst),
                        "cat":                   cat,
                    }

                    # Extract model configuration from results
                    use_powerlaw_2h = results.get("use_powerlaw_2h", True)
                    alpha_2h_fixed = results.get("alpha_2h_fixed", -1.5)
                    use_linear_2h = results.get("use_linear_2h", False)
                    
                    # Regenerate linear 2H templates if needed (with high ell_max for full plotting range)
                    dl_2h_lin_per_zbin = {}
                    if use_linear_2h:
                        from ciber.theory.cross_ps_parametric_model import _compute_linear_2h_templates_per_zbin
                        zbinedges = results.get("zbinedges", np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
                        dl_2h_lin_per_zbin = _compute_linear_2h_templates_per_zbin(zbinedges, 1e5, verbose=False)

                    model = CrossPowerSpectrumModel(
                        lb=lb_fit, use_powerlaw_2h=use_powerlaw_2h,
                        alpha_2h_fixed=alpha_2h_fixed,
                        use_astrometry_damping=use_damping,
                        use_linear_2h=use_linear_2h,
                        dl_2h_lin_per_zbin=dl_2h_lin_per_zbin,
                    )

                    title = (f"CIBER {lams[inst]} μm × {cat}, "
                             f"{zlo:.1f}<z<{zhi:.1f}, \ell_{max}={lMax}")

                    fig, axes = plot_fit_fixed_1h_templates(
                        model, lb_fit, data_dl, data_dlerr, fit_result,
                        figsize=(6, 6), title=title, title_fs=13,
                        ylim=[1e-3, 5e2], lMax_fit=lMax,
                        chi2_lim=[-5, 5], z_bin_index=zidx,
                    )
                    ax = axes[0] if hasattr(axes, '__len__') else axes

                    # Overlay IGL prediction from v3 boxed sims
                    mock_basepath = getattr(args, 'mock_basepath', None)
                    igl_path = _igl_pred_path(mock_basepath, cat, inst, zlo, zhi)
                    if igl_path is not None:
                        pred = np.load(str(igl_path), allow_pickle=True)
                        if "lb" in pred and "cross" in pred:
                            lb_m = np.asarray(pred["lb"])[2:]
                            cl_m = np.asarray(pred["cross"])[2:]
                            pf_m = lb_m * (lb_m + 1) / (2 * np.pi)
                            ax.plot(lb_m, pf_m * cl_m, 'k:', linewidth=1.5,
                                    label='IGL (v3 sim)', alpha=0.8)
                            ax.legend(fontsize=10, loc=4)

                    stem = figdir / f"{cat}_TM{inst}_z{zidx:02d}_lMax={lMax}"
                    _savefig(fig, stem, args.fig_fmt)
                    plt.close(fig)


def _parse_sigma_damp_fixed_mapping(source: object) -> dict:
    """Normalize fixed sigma_damp values from either args or results objects."""
    raw = None
    if source is None:
        return {}
    if hasattr(source, "sigma_damp_fixed"):
        raw = getattr(source, "sigma_damp_fixed", None)
    elif isinstance(source, dict):
        raw = source.get("sigma_damp_fixed", None)
    if raw is None:
        return {}
    if isinstance(raw, (list, tuple, np.ndarray)):
        return {i + 1: float(val) for i, val in enumerate(raw)}
    if isinstance(raw, dict):
        return {int(k): float(v) for k, v in raw.items()}
    return {}


def _extract_effective_fpop_from_weights(weight_pop1, z_idx=None):
    """Return the baseline effective f_pop implied by the combined one-halo weights."""
    if weight_pop1 is None:
        return None
    arr = np.asarray(weight_pop1, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if z_idx is None:
        return arr
    if z_idx < arr.size:
        return float(arr[z_idx])
    return float(arr[-1])


def _plot_fpop_vs_redshift(args: argparse.Namespace) -> None:
    """Plot fitted f_pop versus redshift alongside the baseline effective f_pop from combined one-halo predictions."""
    figdir = Path(args.figdir) / args.fitstr_cross / "popmix"
    lams = {1: 1.1, 2: 1.8}

    for cat in args.cat:
        headstr = args.headstr if cat == "HSC" else None
        for lMax in args.lmax:
            results = _load_cross_results_merged_jh14(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax, maskstr=args.maskstr)
            if results is None:
                fpath = _cross_fpath(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax, maskstr=args.maskstr)
                print(f"[plot_fpop_vs_redshift] missing {fpath}, skipping")
                continue
            if not bool(results.get("onehalo_fit_popmix", False)):
                print(f"[plot_fpop_vs_redshift] {cat} lMax={lMax} has no popmix fit, skipping")
                continue

            zbinedges = np.asarray(results.get("zbinedges", np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])), dtype=float)
            zcenters = 0.5 * (zbinedges[:-1] + zbinedges[1:])
            inst_list = list(results.get("inst_list", [1, 2]))

            onehalo_output_dir = getattr(args, "onehalo_dir", None) or results.get("onehalo_output_dir", None)
            fsat_model = results.get("onehalo_fsat_model", getattr(args, "onehalo_fsat_model", "single"))
            generate_type = results.get("onehalo_generate_type", getattr(args, "onehalo_generate_type", "bulk"))
            concentration_scale = results.get("onehalo_concentration_scale", getattr(args, "concentration_scale", 1.0))

            if cat == "HSC":
                bandstr_select = "hsc_i"
                mag_cut = 25.0
            else:
                bandstr_select = "sdss_z"
                mag_cut = 22.0

            fig, axes = plt.subplots(1, len(inst_list), figsize=(8, 4), sharey=True)
            if len(inst_list) == 1:
                axes = [axes]

            for inst_idx, inst in enumerate(inst_list):
                ax = axes[inst_idx]
                fitted = []
                fitted_lo = []
                fitted_hi = []
                baseline = []

                for z_idx in range(len(zcenters)):
                    params = np.asarray(results["params"], dtype=float)
                    params_err = np.asarray(results.get("params_err", np.full_like(params, np.nan)), dtype=float)
                    params_16 = np.asarray(results.get("params_16", params - params_err), dtype=float)
                    params_84 = np.asarray(results.get("params_84", params + params_err), dtype=float)
                    pnf = results.get("param_names_fitted", None)
                    pnf_bin = pnf[inst_idx, z_idx] if pnf is not None else None

                    n_params_stored = int(np.sum(~np.isnan(params[inst_idx, z_idx, :])))
                    params_i = params[inst_idx, z_idx, :n_params_stored]
                    params_err_i = params_err[inst_idx, z_idx, :n_params_stored]
                    params_16_i = params_16[inst_idx, z_idx, :n_params_stored]
                    params_84_i = params_84[inst_idx, z_idx, :n_params_stored]
                    pnf_bin_i = pnf_bin[:n_params_stored] if pnf_bin is not None else None

                    use_damping = bool(results.get("use_astrometry_damping", False))
                    if pnf_bin_i is not None:
                        use_damping = use_damping or any("damp" in str(p).lower() for p in pnf_bin_i)

                    med = resolve_full_param_value(
                        params_i,
                        pnf_bin_i,
                        "f_pop",
                        use_astrometry_damping=use_damping,
                        use_onehalo_popmix=True,
                    )
                    lo = resolve_full_param_value(
                        params_16_i,
                        pnf_bin_i,
                        "f_pop",
                        use_astrometry_damping=use_damping,
                        use_onehalo_popmix=True,
                    )
                    hi = resolve_full_param_value(
                        params_84_i,
                        pnf_bin_i,
                        "f_pop",
                        use_astrometry_damping=use_damping,
                        use_onehalo_popmix=True,
                    )

                    fitted.append(float(med))
                    fitted_lo.append(float(lo))
                    fitted_hi.append(float(hi))

                    if onehalo_output_dir is not None and os.path.exists(onehalo_output_dir):
                        try:
                            onehalo_result = load_onehalo_results(
                                onehalo_output_dir,
                                fsat_model,
                                bandstr_select,
                                inst=int(inst),
                                mag_min=18.0,
                                mag_cut=mag_cut,
                                z0=0.05,
                                generate_type=generate_type,
                                mode="Ig",
                                concentration_scale=float(concentration_scale),
                                prefer_merged=False,
                            )
                            baseline.append(_extract_effective_fpop_from_weights(onehalo_result.get("weight_pop1"), z_idx=z_idx))
                        except Exception as exc:
                            print(f"[plot_fpop_vs_redshift] failed to load baseline one-halo result for {cat} TM{inst}: {exc}")
                            baseline.append(np.nan)
                    else:
                        baseline.append(np.nan)

                fitted = np.asarray(fitted, dtype=float)
                fitted_lo = np.asarray(fitted_lo, dtype=float)
                fitted_hi = np.asarray(fitted_hi, dtype=float)
                baseline = np.asarray(baseline, dtype=float)
                valid_baseline = np.isfinite(baseline)

                ax.errorbar(
                    zcenters,
                    fitted,
                    yerr=[np.clip(fitted - fitted_lo, 0.0, None), np.clip(fitted_hi - fitted, 0.0, None)],
                    fmt='o',
                    color='C0',
                    ecolor='C0',
                    capsize=3,
                    markersize=5,
                    label='Fitted f_pop',
                    zorder=5,
                )
                if np.any(valid_baseline):
                    ax.plot(zcenters[valid_baseline], baseline[valid_baseline], 'k--', lw=1.5, label='Baseline combined', zorder=4)
                ax.set_xticks(zcenters)
                ax.set_xlim([zbinedges[0], zbinedges[-1]])
                ax.set_ylim([0.0, 1.0])
                ax.set_xlabel(r'$z$', fontsize=12)
                ax.set_ylabel(r'$f_{\rm pop}$', fontsize=12)
                ax.set_title(f"CIBER {lams[int(inst)]} μm × {cat}", fontsize=12)
                ax.grid(True, alpha=0.25)
                ax.axhline(0.5, color='gray', linestyle=':', lw=1.0)

            axes[0].legend(loc='best', fontsize=10)
            fig.suptitle(f"Popmix f_pop vs redshift, ℓ_max={lMax}", fontsize=13)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            stem = figdir / f"{cat}_popmix_fpop_vs_redshift_lMax={lMax}"
            stem = _build_plot_path_with_model(stem, args=args, results=results)
            _savefig(fig, stem, args.fig_fmt)
            plt.close(fig)


def _plot_spectra_summary(args: argparse.Namespace) -> None:
    """Two rows of 5 panels each (top: data + model, bottom: residuals).

    Each top panel shows data + model + components. Bottom panels show
    (data - model) / error. A single shared legend sits above top panels in five columns.
    """
    # from ciber.theory.cross_ps_parametric_model import plot_fit_fixed_1h_templates
    from ciber.cross_correlation.galaxy_cross import collect_ciber_gal_vs_redshift

    figdir = Path(args.figdir) / args.fitstr_cross / "spectra"


    colors = {
        "data": "#000000",
        "igl": "#595959",
        "total": "red",      # strong red-orange
        "two_halo": "blue",      # muted blue
        "one_halo": "#C28B1E",      # green (CB-safe shade)
        "shot_noise": "grey",       # warm orange
    }

    for cat in args.cat:
        headstr = args.headstr if cat == "HSC" else None
        for lMax in args.lmax:
            results = _load_cross_results_merged_jh14(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax, maskstr=args.maskstr)
            if results is None:
                fpath = _cross_fpath(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax, maskstr=args.maskstr)
                print(f"[plot_spectra_summary] missing {fpath}, skipping")
                continue
            zbinedges = results["zbinedges"]
            inst_list = list(results["inst_list"])
            n_zbins = len(zbinedges) - 1
            lams = {1: 1.1, 2: 1.8}

            if cat == "DESILS":
                res_ps = collect_ciber_gal_vs_redshift(
                    "LS", subtract_randoms=True, inst_list=inst_list,
                    zbinedges=zbinedges, maskstr='JHlt16_wFFerr', subtract_sn=False,
                    tl_pix_correct=True, ifield_list=args.ifield_ls,
                )
                # Replace z<0.2 bin (zidx=0) with JHlt14 data to match the fit
                res_jh14 = collect_ciber_gal_vs_redshift(
                    "LS", subtract_randoms=True, inst_list=inst_list,
                    zbinedges=[zbinedges[0], zbinedges[1]], maskstr='JHlt14_wFFerr',
                    subtract_sn=False, tl_pix_correct=True,
                    ifield_list=args.ifield_ls,
                )
                full_cl_cross = res_ps['full_cl_cross'].copy()
                full_clerr_cross = res_ps['full_clerr_cross'].copy()
                full_cl_cross[:, 0, :] = res_jh14['full_cl_cross'][:, 0, :]
                full_clerr_cross[:, 0, :] = res_jh14['full_clerr_cross'][:, 0, :]
            else:
                res_ps = collect_ciber_gal_vs_redshift(
                    "HSC", subtract_randoms=True, inst_list=inst_list,
                    zbinedges=zbinedges, maskstr=None, subtract_sn=False,
                    tl_pix_correct=True, ifield_list=args.ifield_hsc,
                    headstr=headstr, with_ff_err=True,
                )
                full_cl_cross = res_ps['full_cl_cross']
                full_clerr_cross = res_ps['full_clerr_cross']

            lb = res_ps['lb']
            pf_data = lb * (lb + 1) / (2 * np.pi)
            startidx, endidx = 2, -1
            lb_fit = lb[startidx:endidx]

            use_powerlaw_2h = results.get("use_powerlaw_2h", True)
            alpha_2h_fixed = results.get("alpha_2h_fixed", 0.0)
            use_linear_2h = results.get("use_linear_2h", False)
            
            # Regenerate linear 2H templates if needed (with high ell_max for full plotting range)
            dl_2h_lin_per_zbin = {}
            if use_linear_2h:
                from ciber.theory.cross_ps_parametric_model import _compute_linear_2h_templates_per_zbin
                zbinedges = results.get("zbinedges", np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
                dl_2h_lin_per_zbin = _compute_linear_2h_templates_per_zbin(zbinedges, 1.2e5, verbose=False)
            
            pnf_arr = results.get("param_names_fitted", None)
            sigma_damp_fixed_map = _parse_sigma_damp_fixed_mapping(args)

            for inst_idx, inst in enumerate(inst_list):
                fig = plt.figure(figsize=(14, 5))
                gs = fig.add_gridspec(2, n_zbins, height_ratios=[2.5, 1], hspace=0.1, wspace=0.05)
                spec_axes = []
                res_axes = []
                for i in range(n_zbins):
                    sharex_ax = spec_axes[0] if i > 0 else None
                    sharey_ax = spec_axes[0] if i > 0 else None
                    ax_spec = fig.add_subplot(gs[0, i], sharex=sharex_ax, sharey=sharey_ax)
                    spec_axes.append(ax_spec)

                    sharex_res = spec_axes[i]
                    sharey_res = res_axes[0] if i > 0 else None
                    ax_res = fig.add_subplot(gs[1, i], sharex=sharex_res, sharey=sharey_res)
                    res_axes.append(ax_res)

                ylim = [1e-3, 5e2]
                legend_handles = None
                legend_labels = None

                # Build IGL prediction paths from default jordan_mocks/v2 location.
                # No --mock-basepath needed; uses grab_ciber_cross_vs_z_predfpaths defaults.
                from ciber.theory.cl_predictions import grab_ciber_cross_vs_z_predfpaths
                bias_cache = None
                _default_bias_cache = Path(__file__).resolve().parent / 'effective_bias_ls_cache.npz'
                _bias_cache_fpath = getattr(args, 'bias_cache_fpath', None) or (
                    str(_default_bias_cache) if _default_bias_cache.exists() else None
                )
                if _bias_cache_fpath and os.path.exists(_bias_cache_fpath):
                    bias_cache = np.load(_bias_cache_fpath, allow_pickle=False)

                pred_fpaths_by_zbin = {}
                if cat == 'DESILS':
                    heads = ['sdss_z_lt_22.0_CIBERfidmask', 'sdss_z_lt_22.0']
                else:
                    heads = ['hsc_i_lt_25.0', 'hsc_i_lt_25.0_CIBERfidmask', 'hsc_ilt25.0']
                for hs in heads:
                    cands = grab_ciber_cross_vs_z_predfpaths(
                        inst_list=[inst], zbinedges=list(zbinedges),
                        jmock_basedir=None, headstr=hs)[0]
                    if any(os.path.exists(p) for p in cands):
                        pred_fpaths_by_zbin = {zi: p for zi, p in enumerate(cands)}
                        break

                zbin_plot_data = []
                for zidx in range(n_zbins):
                    ax_spec = spec_axes[zidx]
                    ax_res = res_axes[zidx]
                    zlo, zhi = zbinedges[zidx], zbinedges[zidx + 1]
                    zcen = 0.5 * (zlo + zhi)

                    data_dl    = (pf_data * full_cl_cross[inst_idx, zidx])[startidx:endidx]
                    data_dlerr = (pf_data * full_clerr_cross[inst_idx, zidx])[startidx:endidx]

                    params     = np.asarray(results["params"])[inst_idx, zidx, :]
                    params_err = np.asarray(results["params_err"])[inst_idx, zidx, :]
                    params_16  = np.asarray(results.get("params_16", results["params"] - results["params_err"]))[inst_idx, zidx, :]
                    params_84  = np.asarray(results.get("params_84", results["params"] + results["params_err"]))[inst_idx, zidx, :]
                    samples_bin = results.get("samples", np.empty((0,)))[inst_idx, zidx]
                    n_params_stored = int(np.sum(~np.isnan(params)))
                    params     = params[:n_params_stored]
                    params_err = params_err[:n_params_stored]
                    params_16  = params_16[:n_params_stored]
                    params_84  = params_84[:n_params_stored]

                    # Get ndof from saved results (already correctly calculated in fit_model_mcmc)
                    ndof_correct = int(results["ndof"][inst_idx, zidx]) if results.get("ndof") is not None else len(lb_fit) - n_params_stored

                    pnf_bin = pnf_arr[inst_idx, zidx] if pnf_arr is not None else None
                    use_damping = (pnf_bin is not None and
                                   any("damp" in str(p).lower() for p in pnf_bin))
                    sigma_damp_fixed_for_inst = sigma_damp_fixed_map.get(int(inst), None)
                    if sigma_damp_fixed_for_inst is not None:
                        use_damping = True

                    model = CrossPowerSpectrumModel(
                        lb=lb_fit, use_powerlaw_2h=use_powerlaw_2h,
                        alpha_2h_fixed=alpha_2h_fixed,
                        use_astrometry_damping=use_damping,
                        use_linear_2h=use_linear_2h,
                        dl_2h_lin_per_zbin=dl_2h_lin_per_zbin,
                    )

                    fit_result = {
                        "params": params,
                        "params_err": params_err,
                        "use_astrometry_damping": use_damping,
                        "samples": samples_bin if samples_bin is not None and len(np.asarray(samples_bin).shape) > 0 else None,
                        "param_names_fitted": pnf_bin,
                        "onehalo_mode": bool(results.get("onehalo_mode", False)),
                        "onehalo_output_dir": results.get("onehalo_output_dir", ""),
                        "onehalo_generate_type": results.get("onehalo_generate_type", "bulk"),
                        "onehalo_fsat_model": results.get("onehalo_fsat_model", "single"),
                        "onehalo_population": results.get("onehalo_population", getattr(args, 'onehalo_population', 'combined')),
                        "onehalo_fit_popmix": bool(results.get("onehalo_fit_popmix", getattr(args, 'onehalo_fit_popmix', False))),
                        "onehalo_concentration_scale": float(results.get("onehalo_concentration_scale", getattr(args, 'concentration_scale', 1.0))),
                        "inst": int(inst),
                        "cat": cat,
                    }
                    attach_onehalo_template_to_model(
                        model, fit_result, z_bin_index=zidx, use_default_if_missing=False, zbinedges=zbinedges
                    )

                    zbin_plot_data.append({
                        "zidx": zidx,
                        "model": model,
                        "params": params,
                        "use_damping": True,
                        "use_popmix": bool(fit_result.get("onehalo_fit_popmix", False)),
                        "sd_med": (
                            sigma_damp_fixed_for_inst
                            if sigma_damp_fixed_for_inst is not None
                            else resolve_full_param_value(
                                params,
                                pnf_bin,
                                "sigma_damp",
                                use_astrometry_damping=True,
                                use_onehalo_popmix=bool(fit_result.get("onehalo_fit_popmix", False)),
                            )
                        ) if use_damping else None,
                        "f_pop_med": resolve_full_param_value(
                            params,
                            pnf_bin,
                            "f_pop",
                            use_astrometry_damping=True,
                            use_onehalo_popmix=bool(fit_result.get("onehalo_fit_popmix", False)),
                        ) if bool(fit_result.get("onehalo_fit_popmix", False)) else None,
                        "data_dl": data_dl,
                        "data_dlerr": data_dlerr,
                        "zlo": zlo,
                        "zhi": zhi,
                    })

                    # Top panel: spectra
                    ax_spec.errorbar(lb_fit, data_dl, yerr=data_dlerr, fmt='o',
                                     color='k', markersize=3, capsize=2, alpha=0.6, label='Data', zorder=7)

                    ell_m = np.logspace(np.log10(100), np.log10(1.2e5), 500)
                    use_popmix_bin = bool(fit_result.get("onehalo_fit_popmix", False))
                    f_pop_med = resolve_full_param_value(
                        params,
                        pnf_bin,
                        "f_pop",
                        use_astrometry_damping=True,
                        use_onehalo_popmix=use_popmix_bin,
                    ) if use_popmix_bin else None
                    sd_med = (
                        sigma_damp_fixed_for_inst
                        if sigma_damp_fixed_for_inst is not None
                        else resolve_full_param_value(
                            params,
                            pnf_bin,
                            "sigma_damp",
                            use_astrometry_damping=True,
                            use_onehalo_popmix=use_popmix_bin,
                        )
                    ) if use_damping else None
                    comps_result = model.model_components(
                        ell_m,
                        *params[:5],
                        sigma_damp=sd_med,
                        z_bin_index=zidx,
                        f_pop=f_pop_med,
                    )
                    
                    # Ensure comps is a dict; if not, wrap it
                    if not isinstance(comps_result, dict):
                        if comps_result is None:
                            print(f"[plot_spectra_summary] Warning: model_components returned None for zidx={zidx}")
                            comps = {
                                'two_halo': np.zeros(ell_m.size),
                                'one_halo': np.zeros(ell_m.size),
                                'shot_noise': np.zeros(ell_m.size),
                                'total': np.zeros(ell_m.size),
                            }
                        else:
                            comps = _normalize_model_components(comps_result, ell_m)
                    else:
                        comps = comps_result


                    # Match plot_fit_spectra uncertainty behavior: use posterior samples first.
                    bands = None
                    if samples_bin is not None:
                        s = np.asarray(samples_bin, dtype=float)
                        if s.ndim == 2 and s.shape[0] > 1:
                            nfull = 5 + (1 if use_popmix_bin else 0) + (1 if use_damping else 0)
                            if s.shape[1] == nfull:
                                sfull = s
                            else:
                                sfull = expand_fit_samples_to_full_vector(
                                    s,
                                    np.asarray(params[:nfull], dtype=float),
                                    param_names_fitted=pnf_bin,
                                    use_astrometry_damping=True,
                                    use_onehalo_popmix=use_popmix_bin,
                                )

                            c2h = np.zeros((sfull.shape[0], ell_m.size))
                            c1h = np.zeros((sfull.shape[0], ell_m.size))
                            csh = np.zeros((sfull.shape[0], ell_m.size))
                            ctot = np.zeros((sfull.shape[0], ell_m.size))
                            for ii in range(sfull.shape[0]):
                                f_pop_i = sfull[ii, 5] if (use_popmix_bin and sfull.shape[1] > 5) else None
                                if use_damping:
                                    damp_idx = 6 if use_popmix_bin else 5
                                    sd_i = sfull[ii, damp_idx] if sfull.shape[1] > damp_idx else None
                                else:
                                    sd_i = None
                                cc = _normalize_model_components(
                                    model.model_components(
                                        ell_m,
                                        sfull[ii, 0], sfull[ii, 1], sfull[ii, 2], sfull[ii, 3], sfull[ii, 4],
                                        sigma_damp=sd_i,
                                        z_bin_index=zidx,
                                        f_pop=f_pop_i,
                                    ),
                                    ell_m,
                                )
                                c2h[ii] = cc['two_halo']
                                c1h[ii] = cc['one_halo']
                                csh[ii] = cc['shot_noise']
                                ctot[ii] = cc['total']

                            bands = {
                                'two_halo': (np.percentile(c2h, 16, axis=0), np.percentile(c2h, 84, axis=0)),
                                'one_halo': (np.percentile(c1h, 16, axis=0), np.percentile(c1h, 84, axis=0)),
                                'shot_noise': (np.percentile(csh, 16, axis=0), np.percentile(csh, 84, axis=0)),
                                'total': (np.percentile(ctot, 16, axis=0), np.percentile(ctot, 84, axis=0)),
                            }

                    # Fallback for older files without chain samples.
                    if bands is None:
                        f_lo = resolve_full_param_value(
                            params_16,
                            pnf_bin,
                            "f_pop",
                            use_astrometry_damping=use_damping,
                            use_onehalo_popmix=use_popmix_bin,
                        ) if use_popmix_bin else None
                        f_hi = resolve_full_param_value(
                            params_84,
                            pnf_bin,
                            "f_pop",
                            use_astrometry_damping=use_damping,
                            use_onehalo_popmix=use_popmix_bin,
                        ) if use_popmix_bin else None
                        sd_lo = (
                            sigma_damp_fixed_for_inst
                            if sigma_damp_fixed_for_inst is not None
                            else resolve_full_param_value(
                                params_16,
                                pnf_bin,
                                "sigma_damp",
                                use_astrometry_damping=use_damping,
                                use_onehalo_popmix=use_popmix_bin,
                            )
                        ) if use_damping else None
                        sd_hi = (
                            sigma_damp_fixed_for_inst
                            if sigma_damp_fixed_for_inst is not None
                            else resolve_full_param_value(
                                params_84,
                                pnf_bin,
                                "sigma_damp",
                                use_astrometry_damping=use_damping,
                                use_onehalo_popmix=use_popmix_bin,
                            )
                        ) if use_damping else None
                        comps_lo = _normalize_model_components(
                            model.model_components(ell_m, *params_16[:5], sigma_damp=sd_lo, z_bin_index=zidx, f_pop=f_lo),
                            ell_m,
                        )
                        comps_hi = _normalize_model_components(
                            model.model_components(ell_m, *params_84[:5], sigma_damp=sd_hi, z_bin_index=zidx, f_pop=f_hi),
                            ell_m,
                        )
                        bands = {
                            'total': (comps_lo['total'], comps_hi['total']),
                            'two_halo': (comps_lo['two_halo'], comps_hi['two_halo']),
                            'one_halo': (comps_lo['one_halo'], comps_hi['one_halo']),
                            'shot_noise': (comps_lo['shot_noise'], comps_hi['shot_noise']),
                        }

                    ax_spec.plot(ell_m, comps['total'], color=colors['total'], lw=2.5, label='Best-fit model', zorder=7)
                    ax_spec.fill_between(ell_m, bands['total'][0], bands['total'][1],
                                         color=colors['total'], alpha=0.15)
                    ax_spec.plot(ell_m, comps['two_halo'], color=colors['two_halo'], linestyle='dashdot', lw=1.2, alpha=0.7, label='Two-halo', zorder=6)
                    ax_spec.fill_between(ell_m, bands['two_halo'][0], bands['two_halo'][1],
                                         color=colors["two_halo"], alpha=0.12)
                    ax_spec.plot(ell_m, comps['one_halo'], color=colors['one_halo'], lw=1.2, alpha=0.7, label='One-halo', zorder=6)
                    ax_spec.fill_between(ell_m, bands['one_halo'][0], bands['one_halo'][1],
                                         color=colors['one_halo'], alpha=0.12)
                    
                    
                    ax_spec.plot(ell_m, comps['shot_noise'], color=colors['shot_noise'], linestyle='dashed', lw=1.2, alpha=0.7, label='Poisson level')
                    ax_spec.fill_between(ell_m, bands['shot_noise'][0], bands['shot_noise'][1],
                                         color=colors['shot_noise'], alpha=0.12)


                    # IGL overlay — same approach as _plot_redshift_panels_2x2
                    pred_fpath = pred_fpaths_by_zbin.get(zidx)
                    if pred_fpath is not None and os.path.exists(pred_fpath):
                        try:
                            from ciber.plotting.gal_plotting_fns import smooth_mock_cross_with_bias, load_onehalo_spectrum
                            if cat == 'DESILS' and bias_cache is not None:
                                b_g = float(np.poly1d(np.asarray(bias_cache['coarse_poly_coeffs']))(zcen))
                            else:
                                b_g = 1.0 + 0.84 * zcen
                            ell_igl = np.geomspace(lb_fit.min() * 0.8, lb_fit.max() * 1.2, 300)
                            _, dl_igl = smooth_mock_cross_with_bias(pred_fpath, zcen, b_g, ell_eval=ell_igl)

                            onehalo_output_dir ='data/jordan_mocks/v3/fov_10.0/onehalo_predict/'

                            if cat=='HSC':
                                bandstr_select = 'hsc_i'
                                mag_cut = 25.0
                            elif cat=='DESILS':
                                bandstr_select = 'sdss_z'
                                mag_cut = 22.0

                            oh_data_Ig = load_onehalo_spectrum(
                                        onehalo_output_dir, 'single', bandstr_select,
                                        inst=inst_idx+1, mag_min=18.0, mag_cut=mag_cut, z0=0.05, mode='Ig', generate_type='fine',
                                        population=getattr(args, 'onehalo_population', 'combined'))
                            ell_1h = oh_data_Ig['ell_arr']
                            dl_1h = oh_data_Ig['dl_spectrum'][zidx]

                            dl_1h_interp = np.interp(ell_igl, ell_1h, dl_1h)

                            if zidx==0:
                                dl_1h_interp *= 0.5

                            dl_igl += dl_1h_interp


                            ax_spec.plot(ell_igl, dl_igl, color='k', linestyle='solid', lw=2.5,
                                         label='IGL prediction', alpha=0.5, zorder=6)
                        except Exception as e:
                            print(f'[plot_spectra_summary] IGL overlay failed z=[{zlo},{zhi}]: {e}')

                    ax_spec.set_xscale('log')
                    ax_spec.set_yscale('log')
                    ax_spec.set_xlim([lb_fit.min() * 0.8, lb_fit.max() * 1.2])
                    ax_spec.set_ylim(ylim)
                    ax_spec.grid(True, alpha=0.2, which='major')
                    ax_spec.axvspan(lMax, lb_fit.max() * 1.2, color='lightgray', alpha=0.3, zorder=0)
                    ax_spec.set_xticklabels([])

                    # Panel label in upper-left
                    # Display chi2/dof with bandpower count and parameter count
                    # n_bandpowers = len(lb_fit)
                    # n_floated = int(n_params_fit) if n_params_fit is not None and not np.isnan(n_params_fit) else n_params_stored
                    
                    reduced_chi2_val = float(results['reduced_chisq'][inst_idx, zidx]) if results.get('reduced_chisq') is not None else np.nan
                    # chi2_str = f"χ²/dof={reduced_chi2_val:.2f})"
                    chi2_str = '$\\chi^2/{\\rm dof}=$'+f"{results['chisq'][inst_idx, zidx]:.1f}/{int(results['ndof'][inst_idx, zidx])}" + f" ({reduced_chi2_val:.2f})"
                    # ax_spec.text(0.04, 0.97, f"{zlo:.1f}<z<{zhi:.1f}\n{chi2_str}",
                                # transform=ax_spec.transAxes, fontsize=9, va='top', ha='left')
                    
                    if cat=='DESILS':
                        catstr = 'DESI-LS'
                    else:
                        catstr = cat
                    plotstr = 'CIBER '+str(lams[inst])+' $\\mu$m $\\times$ '+catstr+'\n'+f"{zlo:.1f}<z<{zhi:.1f}\n"+chi2_str
                    # f"CIBER z∈[{zlo:.1f},{zhi:.1f}]\n{chi2_str}"
                    ax_spec.text(0.04, 0.97, plotstr,
                                 transform=ax_spec.transAxes, fontsize=9, va='top', ha='left')


                    if zidx == 0:
                        ax_spec.set_ylabel(r'$D_\ell^{\rm Ig}$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=12)
                        ax_spec.tick_params(axis='y', labelleft=True)
                    else:
                        ax_spec.tick_params(axis='y', labelleft=False)
                    ax_spec.set_xticks([1e3, 1e4, 1e5])
                    # ax_spec.set_xticklabels(['', '', ''])
                    ax_spec.tick_params(axis='x', labelbottom=False)

                    # Bottom panel: residuals
                    model_at_data = _normalize_model_components(
                        model.model_components(lb_fit, *params[:5], sigma_damp=sd_med, z_bin_index=zidx, f_pop=f_pop_med),
                        lb_fit,
                    )['total']
                    residuals = (data_dl - model_at_data) / data_dlerr
                    ax_res.plot(lb_fit, residuals, 'o', color='k', markersize=3, zorder=5)
                    ax_res.axhline(0, color='r', linestyle='-', lw=1.5, alpha=0.7)

                    ax_res.axhspan(-1, 1, color='grey', alpha=0.3)
                    ax_res.axhspan(-2.5, 2.5, color='grey', alpha=0.1)
                    

                    ax_res.set_xscale('log')
                    ax_res.set_xlim([lb_fit.min() * 0.8, lb_fit.max() * 1.2])
                    ax_res.set_ylim([-5, 5])
                    ax_res.grid(True, alpha=0.3, which='major')
                    ax_res.axvspan(lMax, lb_fit.max() * 1.2, color='lightgray', alpha=0.3, zorder=0)
                    ax_res.set_xscale('log')

                    if zidx == 0:
                        # ax_res.set_ylabel(r'(Data - Model) / $\sigma$', fontsize=10)
                        ax_res.set_ylabel(r'$\chi$', fontsize=12)

                        ax_res.tick_params(axis='both', labelleft=True, labelbottom=True)
                    else:
                        ax_res.tick_params(axis='y', labelleft=False)
                    ax_res.set_xlabel(r'$\ell$', fontsize=12)

                # Capture legend from the last z-bin axis (most complete with IGL if available)
                legend_handles, legend_labels = spec_axes[-1].get_legend_handles_labels()

                # Single legend above top panels in five columns
                if legend_handles:
                    fig.legend(legend_handles, legend_labels,
                               loc='upper center', ncol=6,
                               fontsize=14, frameon=True,
                               bbox_to_anchor=(0.5, 1.02))

                stem = figdir / f"{cat}_TM{inst}_lMax={lMax}_summary"
                stem = _build_plot_path_with_model(stem, args=args, results=results)
                _savefig(fig, stem, args.fig_fmt)
                plt.close(fig)

                # Generate second figure: data/model ratio vs ell for all z-bins in one panel
                fig, ax = plt.subplots(figsize=(8, 5))
                
                if inst==1:
                    colors_ratio = plt.cm.Blues(np.linspace(0.4, 1, n_zbins))
                else:
                    colors_ratio = plt.cm.Reds(np.linspace(0.4, 1, n_zbins))

                ratios = []
                for zbin_info in zbin_plot_data:
                    zidx = zbin_info["zidx"]
                    zlo, zhi = zbin_info["zlo"], zbin_info["zhi"]

                    data_dl = zbin_info["data_dl"]
                    data_dlerr = zbin_info["data_dlerr"]
                    params = zbin_info["params"]
                    model = zbin_info["model"]
                    sd_med = zbin_info["sd_med"]
                    f_pop_med = zbin_info.get("f_pop_med", None)

                    model_dl = model.model_components(
                        lb_fit,
                        *params[:5],
                        sigma_damp=sd_med,
                        z_bin_index=zidx,
                        f_pop=f_pop_med,
                    )['total']

                    ratio = data_dl / model_dl
                    ratio_err = data_dlerr / model_dl

                    ratios.append(ratio)

                    ax.errorbar(lb_fit, ratio, yerr=ratio_err, fmt='o',
                                color=colors_ratio[zidx], markersize=5, capsize=3,
                                alpha=0.9, label=f'z∈[{zlo:.1f},{zhi:.1f}]')

                # ratios = np.array(ratios)

                # ax.plot(lb_fit, np.mean(ratios, axis=0), 'k--', lw=1.5, alpha=0.5)
                
                ax.axhline(1.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
                ax.set_xscale('log')
                ax.set_xlabel(r'$\ell$', fontsize=14)
                ax.set_ylabel('Data / Model', fontsize=14)
                ax.set_xlim([lb_fit.min() * 0.8, lb_fit.max() * 1.2])
                ax.set_ylim([0.0, 3])
                # ax.set_yscale('log')
                ax.grid(True, alpha=0.3, which='major')
                ax.axvspan(lMax*0.9, lb_fit.max() * 1.2, color='lightgray', alpha=0.3, zorder=0)
                ax.legend(fontsize=12, loc=2)
                ax.tick_params(labelsize=14)
                
                if cat == 'DESILS':
                    catstr = 'DESI-LS'
                else:
                    catstr = cat
                title = f'CIBER {lams[inst]} μm × {catstr}, ℓ_max={lMax}'
                ax.set_title(title, fontsize=14)
                
                plt.tight_layout()
                stem = figdir / f"{cat}_TM{inst}_lMax={lMax}_ratio"
                stem = _build_plot_path_with_model(stem, args=args, results=results)
                _savefig(fig, stem, args.fig_fmt)
                plt.close(fig)


def _plot_corner(args: argparse.Namespace) -> None:
    """Plot corner plots of MCMC posteriors for each redshift bin and instrument."""
    figdir = Path(args.figdir) / args.fitstr_cross / "corners"

    for cat in args.cat:
        headstr = args.headstr if cat == "HSC" else None
        for lMax in args.lmax:
            results = _load_cross_results_merged_jh14(
                args.datadir_cross, cat, headstr, args.fitstr_cross, lMax, maskstr=args.maskstr
            )
            if results is None:
                fpath = _cross_fpath(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax, maskstr=args.maskstr)
                print(f"[plot_corner] missing {fpath}, skipping {cat} lMax={lMax}")
                continue

            zbinedges = results["zbinedges"]
            inst_list = list(results["inst_list"])
            param_names = results["param_names"]
            samples_array = results.get("samples", None)

            if samples_array is None:
                print(f"[plot_corner] no MCMC samples in {fpath.name}, skipping {cat}")
                continue


            if cat == "HSC":
                color = "#e377c2"  # same pink as other plots
            else:
                color = "C2"  # default blue for other catalogs

            for inst_idx, inst in enumerate(inst_list):
                for zidx in range(len(zbinedges) - 1):
                    zlo, zhi = zbinedges[zidx], zbinedges[zidx + 1]

                    samples = samples_array[inst_idx, zidx]
                    if samples is None:
                        continue

                    sf_arr = results.get("samples_fitted", None)
                    pnf_arr = results.get("param_names_fitted", None)

                    # Build minimal fit_result for corner plot
                    fit_result = {
                        "samples": samples,
                        "samples_fitted": sf_arr[inst_idx, zidx] if sf_arr is not None else None,
                        "param_names": param_names,
                        "param_names_fitted": pnf_arr[inst_idx, zidx] if pnf_arr is not None else None,
                        "use_astrometry_damping": results.get("use_astrometry_damping", False),
                    }

                    lams = [1.1, 1.8]
                    title = "CIBER " + str(lams[inst-1]) + " $\\mu$m $\\times$ " + str(cat) + f"\n{zlo:.1f}<z<{zhi:.1f}"
                    fig = CrossPowerSpectrumModel.plot_mcmc_corner(
                        fit_result, figsize=(5, 5),
                        save_path=None,
                        color=color,
                        title=title,
                    )
                    stem = figdir / f"{cat}_TM{inst}_z{zidx:02d}_lMax={lMax}"
                    _savefig(fig, stem, args.fig_fmt)
                    plt.close(fig)


# ---------------------------------------------------------------------------
# Corner plot overlay helpers
# ---------------------------------------------------------------------------




def _plot_corner_overlay(args: argparse.Namespace) -> None:
    """Overlay MCMC posteriors from multiple catalogs using corner.corner.

    For each redshift bin and instrument, creates a corner plot with the first
    catalog as the base (full styling) and overlays 2D contours from the second
    catalog for comparison.
    """
    try:
        import corner
    except ImportError:
        raise ImportError("corner is required for corner plots. Install with: pip install corner")

    if len(args.cat) < 2:
        print("[plot_corner_overlay] requires at least 2 catalogs; skipping")
        return

    figdir = Path(args.figdir) / args.fitstr_cross / "corners"
    figdir.mkdir(parents=True, exist_ok=True)

    # Load results for all catalogs at all lMax values
    cat_results = {}
    for cat in args.cat:
        headstr = args.headstr if cat == "HSC" else None
        cat_results[cat] = {}
        for lMax in args.lmax:
            results = _load_cross_results_merged_jh14(
                args.datadir_cross, cat, headstr, args.fitstr_cross, lMax, maskstr=args.maskstr
            )
            if results is None:
                fpath = _cross_fpath(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax, maskstr=args.maskstr)
                print(f"[plot_corner_overlay] missing {fpath}, skipping {cat} lMax={lMax}")
                continue
            cat_results[cat][lMax] = results

    # Color scheme for catalogs
    cat_colors = {"HSC": "#c0392b", "DESILS": "#2980b9"}
    cat_display = {"HSC": "HSC", "DESILS": "DESI-LS"}

    # Iterate over lMax values
    for lMax in args.lmax:
        # Check which catalogs have results at this lMax
        available_cats = sorted([cat for cat in args.cat if lMax in cat_results[cat]])
        if len(available_cats) < 2:
            print(f"[plot_corner_overlay] fewer than 2 catalogs available at lMax={lMax}; skipping")
            continue

        # Use the first available catalog to determine redshift structure
        ref_cat = available_cats[0]
        ref_results = cat_results[ref_cat][lMax]
        zbinedges = ref_results["zbinedges"]
        inst_list = list(ref_results["inst_list"])

        # Iterate over each (inst, z-bin) pair
        for inst_idx, inst in enumerate(inst_list):
            for zidx in range(len(zbinedges) - 1):
                zlo, zhi = zbinedges[zidx], zbinedges[zidx + 1]

                # Collect samples from available catalogs for this (inst, z-bin)
                catalog_samples = {}
                for cat in available_cats:
                    results = cat_results[cat][lMax]
                    samples_array = results.get("samples", None)
                    sf_arr = results.get("samples_fitted", None)

                    if samples_array is None:
                        continue

                    samples = samples_array[inst_idx, zidx]
                    if samples is None:
                        continue

                    # Prefer fitted samples if available
                    if sf_arr is not None:
                        sf = sf_arr[inst_idx, zidx]
                        if sf is not None:
                            samples = sf

                    if samples is None or len(samples) < 10:
                        continue

                    catalog_samples[cat] = samples.copy()

                if len(catalog_samples) < 2:
                    continue

                # Use first catalog as base for corner.corner
                base_cat = available_cats[0]
                overlay_cat = available_cats[1]
                base_samples = catalog_samples[base_cat]
                overlay_samples = catalog_samples[overlay_cat]

                # Prepare labels (parametric model)
                param_names = ref_results.get("param_names", None)
                if param_names is None:
                    n_params = base_samples.shape[1]
                    if n_params == 5:
                        labels = [
                            r'$A_{\rm 2h}$',
                            r'$A_{\rm 1h}$',
                            r'$\mu_{\rm 1h}$',
                            r'$\sigma_{\rm 1h}$',
                            r'$A_{\rm shot} \times 10^7$'
                        ]
                    else:
                        labels = [f'$p_{i}$' for i in range(n_params)]
                else:
                    labels = list(param_names)

                # Scale shot noise for display (last parameter)
                base_samples_scaled = base_samples.copy()
                base_samples_scaled[:, -1] = base_samples_scaled[:, -1] * 1e7
                overlay_samples_scaled = overlay_samples.copy()
                overlay_samples_scaled[:, -1] = overlay_samples_scaled[:, -1] * 1e7

                # Update labels for scaled shot noise
                labels = labels.copy()
                if 'shot' in labels[-1].lower() or 'shot' in str(labels[-1]):
                    labels[-1] = labels[-1].replace('shot}$', 'shot} \\times 10^7$')

                # Create base corner plot with first catalog
                title = f"{cat_display.get(base_cat, base_cat)} × CIBER TM{inst}, z∈[{zlo:.1f},{zhi:.1f}], ℓ_max={lMax}"
                fig = corner.corner(
                    base_samples_scaled,
                    labels=labels,
                    quantiles=[0.16, 0.5, 0.84],
                    show_titles=True,
                    title_kwargs={"fontsize": 10},
                    label_kwargs={"fontsize": 11},
                )

                # Overlay second catalog's contours on off-diagonal panels
                ndim = base_samples_scaled.shape[1]
                axes = np.array(fig.axes).reshape((ndim, ndim))

                for i in range(ndim):
                    for j in range(ndim):
                        ax = axes[i, j]

                        # Only add overlay contours to off-diagonal panels
                        if i != j:
                            overlay_2d = overlay_samples_scaled[:, [j, i]]

                            # Compute axis ranges from both samples
                            x_min = min(np.percentile(base_samples_scaled[:, j], 0.5),
                                       np.percentile(overlay_samples_scaled[:, j], 0.5))
                            x_max = max(np.percentile(base_samples_scaled[:, j], 99.5),
                                       np.percentile(overlay_samples_scaled[:, j], 99.5))
                            y_min = min(np.percentile(base_samples_scaled[:, i], 0.5),
                                       np.percentile(overlay_samples_scaled[:, i], 0.5))
                            y_max = max(np.percentile(base_samples_scaled[:, i], 99.5),
                                       np.percentile(overlay_samples_scaled[:, i], 99.5))

                            # Subsample for KDE if needed
                            if len(overlay_2d) > 10000:
                                idx = np.random.choice(len(overlay_2d), 10000, replace=False)
                                overlay_2d = overlay_2d[idx]

                            try:
                                # Compute 2D KDE for overlay catalog
                                kde = stats.gaussian_kde(overlay_2d.T)
                                x = np.linspace(x_min, x_max, 50)
                                y = np.linspace(y_min, y_max, 50)
                                X, Y = np.meshgrid(x, y)
                                Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

                                # Compute credible levels (68% and 95%)
                                flat = np.sort(Z.ravel())[::-1]
                                cumsum = np.cumsum(flat) / np.sum(flat)
                                levels = []
                                for level in [0.68, 0.95]:
                                    idx_lev = np.searchsorted(cumsum, level, side="right")
                                    idx_lev = min(max(idx_lev, 0), len(flat) - 1)
                                    levels.append(float(flat[idx_lev]))
                                levels = sorted(set(levels))
                                if len(levels) == 1:
                                    levels = [levels[0], levels[0] * 1.001]

                                ax.contour(X, Y, Z, levels=levels,
                                         colors=[cat_colors.get(overlay_cat, "C0")],
                                         alpha=0.7, linewidths=1.5, linestyles='--')
                            except Exception as e:
                                print(f"[plot_corner_overlay] contour error for {overlay_cat} panel ({i},{j}): {e}")

                # Add title with both catalog names
                cat_str = f"{cat_display.get(base_cat, base_cat)} (solid) vs {cat_display.get(overlay_cat, overlay_cat)} (dashed)"
                fig.suptitle(f"{cat_str} — z∈[{zlo:.1f},{zhi:.1f}], TM{inst}, ℓ_max={lMax}",
                            fontsize=12, y=0.995)

                # Save
                cat_suffix = "_".join(available_cats)
                stem = figdir / f"{cat_suffix}_TM{inst}_z{zidx:02d}_lMax={lMax}_overlay"
                _savefig(fig, stem, args.fig_fmt)
                plt.close(fig)



def _plot_compare_cats(args: argparse.Namespace) -> None:
    """4-panel figure comparing catalogs across both CIBER bands.

    Layout: 2 rows x 2 columns, y-axes shared within each row.
      Row 0 (top):    A_1h  — left=TM1 (1.1 μm), right=TM2 (1.8 μm)
      Row 1 (bottom): A_2h  — left=TM1 (1.1 μm), right=TM2 (1.8 μm)
    """
    figdir = Path(args.figdir) / args.fitstr_cross

    # Use lmax from --lmax list if available, otherwise use --lmax-compare
    lMax = args.lmax[-1] if args.lmax else args.lmax_compare

    cat_results = {}
    for cat in args.cat:
        headstr = args.headstr if cat == "HSC" else None
        results = _load_cross_results_merged_jh14(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax, maskstr=args.maskstr)
        if results is None:
            fpath = _cross_fpath(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax, maskstr=args.maskstr)
            print(f"[plot_compare_cats] missing {fpath}, skipping {cat}")
            continue
        cat_results[cat] = results

    if not cat_results:
        print("[plot_compare_cats] no results found for any catalog, skipping")
        return

    cat_colors  = {"HSC": "tab:orange", "DESILS": "tab:blue"}
    cat_markers = {"HSC": "o",          "DESILS": "s"}
    cat_labels  = {"HSC": r"CIBER $\times$ HSC", "DESILS": r"CIBER $\times$ DESI-LS"}
    # DESILS first in legend
    legend_cat_order = [c for c in ["DESILS", "HSC"] if c in cat_results]
    lams = {1: 1.1, 2: 1.8}

    zbin_edges   = np.arange(0.0, 1.0 + 1e-9, 0.2)
    shade_colors = ("#f5f5f5", "#eeeeee")
    xticks       = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    n_cats = len(cat_results)
    x_offset_scale = 0.04

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5), sharex=True, sharey="row")

    inst_col = {1: 0, 2: 1}

    for inst in [1, 2]:
        col = inst_col[inst]
        ax_1h = axes[0, col]
        ax_2h = axes[1, col]

        for j in range(len(zbin_edges) - 1):
            z0, z1 = zbin_edges[j], zbin_edges[j + 1]
            shade = shade_colors[j % 2]
            ax_1h.axvspan(z0, z1, color=shade, alpha=0.22, zorder=0)
            ax_2h.axvspan(z0, z1, color=shade, alpha=0.22, zorder=0)

        for cat_idx, (cat, res) in enumerate(cat_results.items()):
            inst_list = list(res["inst_list"])
            if inst not in inst_list:
                continue
            inst_idx = inst_list.index(inst)

            z_centers = res["z_centers"]
            x_offset  = (cat_idx - (n_cats - 1) / 2.0) * x_offset_scale
            color     = cat_colors.get(cat, f"C{cat_idx}")
            marker    = cat_markers.get(cat, "o")
            # Attach label only on inst==1 so legend entries appear exactly once
            label     = cat_labels.get(cat, cat) if inst == 1 else None

            params_95 = res.get("params_95", None)
            n_params  = res["params"].shape[-1]

            if n_params >= 2:
                A_1h     = res["params"][inst_idx, :, 1]
                A_1h_err = res["params_err"][inst_idx, :, 1]
                p95_1h   = params_95[inst_idx, :, 1] if params_95 is not None else None
                _plot_param(ax_1h, z_centers, A_1h, A_1h_err, x_offset, color, marker, label, p95_1h)

            A_2h     = res["params"][inst_idx, :, 0]
            A_2h_err = res["params_err"][inst_idx, :, 0]
            p95_2h   = params_95[inst_idx, :, 0] if params_95 is not None else None
            _plot_param(ax_2h, z_centers, A_2h, A_2h_err, x_offset, color, marker, None, p95_2h)

        for ax in [ax_1h, ax_2h]:
            ax.set_xlim(0.0, 1.0)
            ax.set_xticks(xticks)
            ax.axhline(0, color="k", linewidth=0.7, linestyle="--", alpha=0.5)
            ax.grid(alpha=0.25, zorder=1)

        ax_1h.set_ylim(0.0, 1.5)
        ax_2h.set_ylim(0.0, 0.4)
        ax_2h.set_xlabel("Redshift", fontsize=11)

        for ax in [ax_1h, ax_2h]:
            ax.text(0.04, 0.94, f"CIBER {lams[inst]:.1f} $\\mu$m",
                    transform=ax.transAxes, fontsize=12,
                    va="top", ha="left")

    # y-axis labels on left column only (sharey handles the right column)
    axes[0, 0].set_ylabel(r"$A_{\rm 1h}$", fontsize=12)
    axes[1, 0].set_ylabel(r"$A_{\rm 2h}$", fontsize=12)

    # Legend above all panels — DESILS first, then HSC
    all_handles, all_labels = axes[0, 0].get_legend_handles_labels()
    label_to_handle = dict(zip(all_labels, all_handles))
    ordered_labels  = [cat_labels[c] for c in legend_cat_order if cat_labels[c] in label_to_handle]
    ordered_handles = [label_to_handle[l] for l in ordered_labels]
    fig.legend(ordered_handles, ordered_labels, loc="upper center", ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, 0.98), frameon=True)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.08, top=0.86)

    _savefig(fig, figdir / f"compare_cats_{args.fitstr_cross}_bothbands_lMax={lMax}", args.fig_fmt)
    plt.close(fig)


def _plot_a1h_vs_redshift_three_row(args: argparse.Namespace) -> None:
    """Plot A_1h vs redshift in a 3-row layout: DESI-LS, HSC i<22, HSC i<25.
    
    Each row shows both TM1 and TM2 bands. Includes all four model variants 
    (full, fixA2h_IGL constant/linear/quadratic). Panel titles indicate magnitude selection.
    """
    figdir = Path(args.figdir) / args.fitstr_cross
    figdir.mkdir(parents=True, exist_ok=True)

    lMax = args.lmax[-1] if args.lmax else args.lmax_compare

    fitstr_fixA2h   = args.fitstr_cross + "_fixA2h_IGL"
    fitstr_biLinear = args.fitstr_cross + "_fixA2h_IGL_biLinear"
    fitstr_biQuad   = args.fitstr_cross + "_fixA2h_IGL_biQuadratic"

    # variant_defs = [
    #     ("full",     args.fitstr_cross),
    #     ("fixA2h",   fitstr_fixA2h),
    #     ("biLinear", fitstr_biLinear),
    #     ("biQuad",   fitstr_biQuad),
    # ]

    variant_defs = [
        ("full",     args.fitstr_cross),
        ("fixA2h",   fitstr_fixA2h),
        ("biQuad",   fitstr_biQuad),
    ]

    # Define the three rows: (catalog, headstr, label)
    # If --headstr specified (non-empty), override HSC rows to use that headstr
    if hasattr(args, 'headstr') and args.headstr and args.headstr != "hsc_ilt25.0":
        # Extract magnitude limit from headstr (e.g., "hsc_ilt25.0" -> "i<25")
        mag_str = args.headstr.replace("hsc_ilt", "i<")
        row_defs = [
            ("DESILS", None, r"DESI-LS ($z_{\rm AB} < 22$)"),
            ("HSC", args.headstr, rf"HSC (${mag_str}$)"),
        ]
    else:
        # row_defs = [
        #     ("DESILS", None, r"DESI-LS ($z_{\rm AB} < 22$)"),
        #     ("HSC", "hsc_ilt22.0", r"HSC ($18<i_{\rm AB} < 22$)"),
        #     ("HSC", "hsc_ilt25.0", r"HSC ($18<i_{\rm AB} < 25$)"),
        # ]
        row_defs = [
            ("DESILS", None, r"DESI-LS ($z_{\rm AB} < 22$)"),
            # ("HSC", "hsc_zlt22.0", r"HSC ($z_{\rm AB} < 22$)"),
            ("HSC", "hsc_ilt25.0", r"HSC ($18<i_{\rm AB} < 25$)"),
        ]

    # Load results for each combination
    # Structure: (cat, headstr) -> {variant_key -> {source (fiducial/jh14) -> {inst -> data}}}
    results_all = {}
    for cat, headstr, label in row_defs:
        key = (cat, headstr)
        results_all[key] = {var_key: {} for var_key, _ in variant_defs}
        for var_key, fitstr_name in variant_defs:
            # Load fiducial (z>0.2 bins)
            fpath = _cross_fpath(args.datadir_cross, cat, headstr, fitstr_name, lMax, maskstr=args.maskstr)
            if fpath.exists():
                res = load_fit_results_npz(str(fpath))
                inst_list = list(res["inst_list"])
                if 'fiducial' not in results_all[key][var_key]:
                    results_all[key][var_key]['fiducial'] = {}
                for inst in inst_list:
                    results_all[key][var_key]['fiducial'][inst] = res
            else:
                if var_key in ("full", "fixA2h"):
                    print(f"[plot_a1h_vs_redshift_three_row] missing fiducial {fpath.name}")
            
            # Load JHlt14 (z<0.2 bin only)
            fpath_jh14 = _cross_fpath(args.datadir_cross, cat, headstr, fitstr_name, lMax, maskstr='JHlt14')
            if fpath_jh14.exists():
                res_jh14 = load_fit_results_npz(str(fpath_jh14))
                inst_list_jh14 = list(res_jh14["inst_list"])
                if 'jh14' not in results_all[key][var_key]:
                    results_all[key][var_key]['jh14'] = {}
                for inst in inst_list_jh14:
                    results_all[key][var_key]['jh14'][inst] = res_jh14
            elif var_key in ("full", "fixA2h"):
                print(f"[plot_a1h_vs_redshift_three_row] missing JHlt14 {fpath_jh14.name}")

    # Build panel info: (row_idx, inst, cat, headstr, label)
    panel_info = []
    for row_idx, (cat, headstr, label) in enumerate(row_defs):
        key = (cat, headstr)
        res_full = results_all[key]["full"].get('fiducial', {})
        if not res_full:
            continue
        # Get inst_list from any available variant
        for inst in [1, 2]:  # TM1, TM2
            if inst in res_full:
                panel_info.append((row_idx, inst, cat, headstr, label))

    if not panel_info:
        print("[plot_a1h_vs_redshift_three_row] no results found, skipping")
        return

    # Create grid: n_rows × 2 cols (TM1, TM2), where n_rows = unique row indices
    n_rows = len(row_defs)
    figsize_height = 1. + 2.0 * n_rows
    # fig, axes = plt.subplots(n_rows, 2, figsize=(7, figsize_height), sharex=True, sharey=True)

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True, sharey=True)

    # Ensure axes is always 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Variant-specific labels and markers
    var_labels = {
        "full":     "Full model (Float $A_{\\rm 2h}$, $A_{\\rm 1h}$)",
        "fixA2h":   r"Fix $A_{\rm 2h}$; $b_I=1$",
        "biLinear": r"Fix $A_{\rm 2h}$; $b_I=1+0.6z$",
        "biQuad":   r"Fix $A_{\rm 2h}$; $b_I=(1+z)^2$",
    }
    var_markers = {
        "full":     "o",
        "fixA2h":   "^",
        "biLinear": "D",
        "biQuad":   "v",
    }

    # Tracer-specific color palettes (darker to brighter for each variant)
    tracer_color_palettes = {
        ("DESILS", None): {  # Greens
            "full":     "#0c5a0c",  # dark green
            "fixA2h":   "#1f771f",  # medium green (C0)
            "biLinear": "#5fad5f",  # light green
            "biQuad":   "#aecdac",  # very light green
        },
        ("HSC", "hsc_zlt22.0"): {  # Bright gold palette
            "full":     "#d4a425",  # dark gold
            "fixA2h":   "#e6ba2c",  # medium gold
            "biLinear": "#f5d54a",  # bright gold
            "biQuad":   "#fde999",  # very bright gold
        },
        ("HSC", "hsc_ilt25.0"): {  # Pink/Purple
            "full":     "#7a0177",  # dark purple-magenta
            "fixA2h":   "#ae017e",  # strong magenta
            "biLinear": "#dd3497",  # pink-magenta
            "biQuad":   "#f768a1",  # light pink
        },
    }

    cat_display = {"DESILS": "DESI-LS", "HSC": "HSC"}
    lams = {1: 1.1, 2: 1.8}
    shade_colors = ("#f5f5f5", "#eeeeee")
    x_offset_scale = 0.04

    for panel_idx, (row_idx, inst, cat, headstr, row_label) in enumerate(panel_info):
        # col = (inst - 1)  # TM1 -> col 0, TM2 -> col 1
        col = 0
        

        # ax = axes[inst-1, 0]

        ax = axes[inst-1]
        # Get reference result for zbinedges (from fiducial, which has all z-bins)
        key = (cat, headstr)
        res_full_fiducial = results_all[key]["full"].get('fiducial', {}).get(inst)
        if not res_full_fiducial:
            continue

        zbinedges = res_full_fiducial["zbinedges"]
        z_centers = 0.5 * (zbinedges[:-1] + zbinedges[1:])

        # Add redshift bin shading
        for j in range(len(zbinedges) - 1):
            z0, z1 = zbinedges[j], zbinedges[j + 1]
            shade = shade_colors[j % 2]
            ax.axvspan(z0, z1, color=shade, alpha=0.22, zorder=0)

        # Add grey overlay shading for HSC z<0.2 bin to indicate omitted measurements
        # if cat == "HSC":
        #     z0_omit, z1_omit = zbinedges[0], zbinedges[1]
        #     ax.axvspan(z0_omit, z1_omit, color="#a9a9a9", alpha=0.25, zorder=1, linewidth=2, edgecolor="grey")

        # Plot each variant with horizontal offset
        for var_idx, (var_key, _) in enumerate(variant_defs):
            res_fid = results_all[key][var_key].get('fiducial', {}).get(inst)
            res_jh14 = results_all[key][var_key].get('jh14', {}).get(inst)
            if not res_fid:
                continue

            inst_list_var = list(res_fid["inst_list"])
            if inst not in inst_list_var:
                continue

            inst_idx = inst_list_var.index(inst)
            n_params = res_fid["params"].shape[-1]

            # Find A_1h index (use fiducial for lookup)
            pnames_var = res_fid.get("param_names_fitted")
            a1h_idx = 1
            if pnames_var is not None:
                try:
                    rep_names = pnames_var[inst_idx, 0]
                    if rep_names is not None:
                        for k, nm in enumerate(rep_names):
                            if "A_1h" in str(nm) or "A_{1h}" in str(nm):
                                a1h_idx = k
                                break
                except (IndexError, TypeError):
                    pass
            if n_params <= a1h_idx:
                continue

            # Build parameter arrays, using JHlt14 for z-bin 0 (z<0.2) if available, else fiducial
            n_zbins = len(zbinedges) - 1
            A_1h = np.zeros(n_zbins)
            A_1h_err_lo = np.zeros(n_zbins)
            A_1h_err_hi = np.zeros(n_zbins)
            A_1h_95_vals = np.zeros(n_zbins)
            
            for z_idx in range(n_zbins):
                # Use JHlt14 for z<0.2 bin (z_idx=0), fiducial otherwise
                if z_idx == 0 and res_jh14 is not None:
                    res_use = res_jh14
                    z_idx_use = 0  # JHlt14 only has one z-bin
                else:
                    res_use = res_fid
                    z_idx_use = z_idx
                
                A_1h[z_idx] = res_use["params"][inst_idx, z_idx_use, a1h_idx]
                
                if "params_16" in res_use and "params_84" in res_use:
                    A_1h_err_lo[z_idx] = res_use["params"][inst_idx, z_idx_use, a1h_idx] - res_use["params_16"][inst_idx, z_idx_use, a1h_idx]
                    A_1h_err_hi[z_idx] = res_use["params_84"][inst_idx, z_idx_use, a1h_idx] - res_use["params"][inst_idx, z_idx_use, a1h_idx]
                else:
                    A_1h_err_lo[z_idx] = res_use["params_err"][inst_idx, z_idx_use, a1h_idx]
                    A_1h_err_hi[z_idx] = res_use["params_err"][inst_idx, z_idx_use, a1h_idx]
                
                if "params_95" in res_use:
                    A_1h_95_vals[z_idx] = res_use["params_95"][inst_idx, z_idx_use, a1h_idx]
                else:
                    A_1h_95_vals[z_idx] = A_1h[z_idx] + 2 * A_1h_err_hi[z_idx]
            
            yerr = np.array([A_1h_err_lo, A_1h_err_hi])

            # Exclude HSC z<0.2 data (unreliable)
            if cat == "HSC":
                A_1h[0] = np.nan
                yerr[:, 0] = np.nan
                A_1h_95_vals[0] = np.nan

            # Identify upper limits
            is_ul = (A_1h - 2 * yerr[0]) <= 0
            is_det = ~is_ul

            # Apply horizontal offset to distinguish variants
            # x_offset = (var_idx - (len(variant_defs) - 1) / 2.0) * x_offset_scale
            # z_offset = z_centers + x_offset


            at_idx = None
            for idx, (row_cat, row_headstr, _) in enumerate(row_defs):
                if row_cat == cat and row_headstr == headstr:
                    cat_idx = idx
                    break
            
            # Catalog-level offset: spread catalogs left/right within bin
            # Variant-level offset: spread variants within catalog group
            n_cats = len(row_defs)
            cat_offset_scale = x_offset_scale * 2  # Slightly larger than variant scale
            cat_offset = (cat_idx - (n_cats - 1) / 2.0) * cat_offset_scale
            var_offset = (var_idx - (len(variant_defs) - 1) / 2.0) * (x_offset_scale * 0.6)  # Smaller variant spacing
            x_offset = cat_offset + var_offset
            z_offset = z_centers + x_offset

            # Look up color from tracer-specific palette
            color = tracer_color_palettes[key].get(var_key, "#666666")
            marker = var_markers[var_key]
            label = var_labels[var_key]

            markersize = 6


            # Plot detections
            if np.any(is_det):
                ax.errorbar(z_offset[is_det], A_1h[is_det],
                           yerr=np.array([yerr[0][is_det], yerr[1][is_det]]),
                           marker=marker, color=color,
                           label=label, linestyle='None',
                           markersize=markersize, capsize=3, linewidth=1.5, alpha=1.0)
                label = None

            # Plot upper limits
            if np.any(is_ul):
                ul_vals = A_1h_95_vals[is_ul]
                xs_ul = z_offset[is_ul]
                ax.plot(xs_ul, ul_vals, marker="v", color=color,
                       label=label, markersize=markersize, alpha=1.0, linestyle='none')
                for x, y_top in zip(xs_ul, ul_vals):
                    ax.annotate('', xy=(x, 0.0), xytext=(x, y_top),
                               arrowprops=dict(arrowstyle='-|>', color=color,
                                            alpha=1.0, lw=1.5))

        # Panel title: row label + band info
        title_text = f"CIBER {lams.get(inst, '?')} μm × {row_label}"

        if row_idx == 0:
            textypos = 0.97
            mode = 'fixA2h'
        else:
            textypos = 0.87
            mode = 'biQuad'
        color = tracer_color_palettes[key].get(mode, "#666666")
        ax.text(0.02, textypos, title_text, transform=ax.transAxes,
                fontsize=14, color=color, verticalalignment='top')
        ax.tick_params(labelsize=12)

        ax.grid(True, alpha=0.3)
        ax.set_xlim(zbinedges[0], zbinedges[-1])
        ax.set_ylim(0, 0.9)

    # Add shared legend above top row
    # handles, labels = axes[0, 0].get_legend_handles_labels()

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.54, 1.15),
               ncol=2, fontsize=13, frameon=True)

    # Set axis labels on outer edges
    for row in range(n_rows):
        axes[row].set_ylabel(r"$A_{\rm 1h}^{\rm Ig}$ [nW m$^{-2}$ sr$^{-1}$]", fontsize=15)

    # for col in range(2):
    axes[-1].set_xlabel("Redshift (z)", fontsize=15)

    fig.subplots_adjust(left=0.11, right=0.98, top=0.96, bottom=0.08, hspace=0.1, wspace=0.09)

    stem = figdir / f"a1h_vs_redshift_three_row_lMax={lMax}"
    _savefig(fig, stem, args.fig_fmt)
    plt.close(fig)
    print(f"[plot_a1h_vs_redshift_three_row] generated {stem.with_suffix('.pdf')}")


def _plot_a1h_vs_redshift_alternate_layout(args: argparse.Namespace) -> None:
    """Alternate layout for A_1h vs redshift: one panel per band, three tracer configs grouped.
    
    Panel arrangement:
    - Row 0: TM1 (1.1 μm)
    - Row 1: TM2 (1.8 μm)
    
    Each panel shows three tracer configurations (DESILS, HSC i<22, HSC i<25) with different colors:
    - DESILS: Blues
    - HSC i<22: Oranges
    - HSC i<25: Greens
    
    For each tracer, all four modeling variants (full, fixA2h, biLinear, biQuad) are shown with markers.
    """
    figdir = Path(args.figdir) / args.fitstr_cross
    figdir.mkdir(parents=True, exist_ok=True)

    lMax = args.lmax[-1] if args.lmax else args.lmax_compare

    fitstr_fixA2h   = args.fitstr_cross + "_fixA2h_IGL"
    fitstr_biLinear = args.fitstr_cross + "_fixA2h_IGL_biLinear"
    fitstr_biQuad   = args.fitstr_cross + "_fixA2h_IGL_biQuadratic"

    variant_defs = [
        ("full",     args.fitstr_cross),
        ("fixA2h",   fitstr_fixA2h),
        ("biLinear", fitstr_biLinear),
        ("biQuad",   fitstr_biQuad),
    ]

    # Tracer configs: (catalog, headstr, label)
    tracer_defs = [
        ("DESILS", None, r"DESI-LS ($z_{\rm AB} < 22$)"),
        # ("HSC", "hsc_zlt22.0", r"HSC ($z_{\rm AB}<22$)"),
        ("HSC", "hsc_ilt25.0", r"HSC ($18<i_{\rm AB}<25$)"),
    ]

    # Color palettes per tracer (from colormap)
    tracer_colors = {
        "DESILS": ["C2", "C2", "C2", "C2"],
        "HSC_22":  ["#E45DA8", "#E45DA8", "#E45DA8", "#E45DA8"],
        "HSC_25":  ["#E45DA8", "#E45DA8", "#E45DA8", "#E45DA8"],
    }

    variant_markers = {"full": "o", "fixA2h": "^", "biLinear": "D", "biQuad": "v"}

    lams = {1: 1.1, 2: 1.8}
    shade_colors = ("#f5f5f5", "#eeeeee")
    x_offset_scale = 0.08  # Larger offset for better visual separation

    # Load results for all combinations
    # Structure: (cat, headstr) -> {variant_key -> {source (fiducial/jh14) -> {inst -> data}}}
    results_all = {}
    for cat, headstr, label in tracer_defs:
        key = (cat, headstr)
        results_all[key] = {var_key: {} for var_key, _ in variant_defs}
        for var_key, fitstr_name in variant_defs:
            # Load fiducial (z>0.2 bins)
            fpath = _cross_fpath(args.datadir_cross, cat, headstr, fitstr_name, lMax)
            if fpath.exists():
                res = load_fit_results_npz(str(fpath))
                inst_list = list(res["inst_list"])
                if 'fiducial' not in results_all[key][var_key]:
                    results_all[key][var_key]['fiducial'] = {}
                for inst in inst_list:
                    results_all[key][var_key]['fiducial'][inst] = res
            else:
                if var_key in ("full", "fixA2h"):
                    print(f"[plot_a1h_vs_redshift_alternate] missing fiducial {fpath.name}")
            
            # Load JHlt14 (z<0.2 bin only)
            fpath_jh14 = _cross_fpath(args.datadir_cross, cat, headstr, fitstr_name, lMax, maskstr='JHlt14')
            if fpath_jh14.exists():
                res_jh14 = load_fit_results_npz(str(fpath_jh14))
                inst_list_jh14 = list(res_jh14["inst_list"])
                if 'jh14' not in results_all[key][var_key]:
                    results_all[key][var_key]['jh14'] = {}
                for inst in inst_list_jh14:
                    results_all[key][var_key]['jh14'][inst] = res_jh14
            elif var_key in ("full", "fixA2h"):
                print(f"[plot_a1h_vs_redshift_alternate] missing JHlt14 {fpath_jh14.name}")

    # Create 2-row layout (one per band)
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True, sharey=True)

    for band_idx, inst in enumerate([1, 2]):
        ax = axes[band_idx]
        
        # Get zbinedges from first available result
        zbinedges = None
        for cat, headstr, label in tracer_defs:
            key = (cat, headstr)
            res_full = results_all[key]["full"].get('fiducial', {}).get(inst)
            if res_full is not None:
                zbinedges = res_full["zbinedges"]
                break
        
        if zbinedges is None:
            print(f"[plot_a1h_vs_redshift_alternate] no zbinedges found for inst {inst}")
            continue
        
        z_centers = 0.5 * (zbinedges[:-1] + zbinedges[1:])
        n_zbins = len(zbinedges) - 1

        # Add redshift bin shading
        for j in range(n_zbins):
            z0, z1 = zbinedges[j], zbinedges[j + 1]
            shade = shade_colors[j % 2]
            ax.axvspan(z0, z1, color=shade, alpha=0.22, zorder=0)

        # Add grey overlay shading for HSC z<0.2 bin to indicate omitted measurements
        z0_omit, z1_omit = zbinedges[0], zbinedges[1]
        ax.axvspan(z0_omit, z1_omit, color="#a9a9a9", alpha=0.25, zorder=0.5, linewidth=1, edgecolor="lightgrey")

        # Plot each tracer with its own color scheme
        for tracer_idx, (cat, headstr, tracer_label) in enumerate(tracer_defs):
            key = (cat, headstr)
            tracer_key_short = "DESILS" if cat == "DESILS" else ("HSC_22" if headstr == "hsc_zlt22.0" else "HSC_25")
            colors_for_tracer = tracer_colors[tracer_key_short]
            
            # Plot each variant with different marker and color
            for var_idx, (var_key, _) in enumerate(variant_defs):
                res_fid = results_all[key][var_key].get('fiducial', {}).get(inst)
                res_jh14 = results_all[key][var_key].get('jh14', {}).get(inst)
                if not res_fid:
                    continue

                inst_list_var = list(res_fid["inst_list"])
                if inst not in inst_list_var:
                    continue

                inst_idx = inst_list_var.index(inst)
                n_params = res_fid["params"].shape[-1]

                # Find A_1h index
                pnames_var = res_fid.get("param_names_fitted")
                a1h_idx = 1
                if pnames_var is not None:
                    try:
                        rep_names = pnames_var[inst_idx, 0]
                        if rep_names is not None:
                            for k, nm in enumerate(rep_names):
                                if "A_1h" in str(nm) or "A_{1h}" in str(nm):
                                    a1h_idx = k
                                    break
                    except (IndexError, TypeError):
                        pass
                if n_params <= a1h_idx:
                    continue

                # Build parameter arrays using JHlt14 for z<0.2 if available
                A_1h = np.zeros(n_zbins)
                A_1h_err_lo = np.zeros(n_zbins)
                A_1h_err_hi = np.zeros(n_zbins)
                A_1h_95_vals = np.zeros(n_zbins)
                
                for z_idx in range(n_zbins):
                    if z_idx == 0 and res_jh14 is not None:
                        res_use = res_jh14
                        z_idx_use = 0
                    else:
                        res_use = res_fid
                        z_idx_use = z_idx
                    
                    A_1h[z_idx] = res_use["params"][inst_idx, z_idx_use, a1h_idx]
                    
                    if "params_16" in res_use and "params_84" in res_use:
                        A_1h_err_lo[z_idx] = res_use["params"][inst_idx, z_idx_use, a1h_idx] - res_use["params_16"][inst_idx, z_idx_use, a1h_idx]
                        A_1h_err_hi[z_idx] = res_use["params_84"][inst_idx, z_idx_use, a1h_idx] - res_use["params"][inst_idx, z_idx_use, a1h_idx]
                    else:
                        A_1h_err_lo[z_idx] = res_use["params_err"][inst_idx, z_idx_use, a1h_idx]
                        A_1h_err_hi[z_idx] = res_use["params_err"][inst_idx, z_idx_use, a1h_idx]
                    
                    if "params_95" in res_use:
                        A_1h_95_vals[z_idx] = res_use["params_95"][inst_idx, z_idx_use, a1h_idx]
                    else:
                        A_1h_95_vals[z_idx] = A_1h[z_idx] + 2 * A_1h_err_hi[z_idx]
                
                yerr = np.array([A_1h_err_lo, A_1h_err_hi])

                # Exclude HSC z<0.2 data
                if cat == "HSC":
                    A_1h[0] = np.nan
                    yerr[:, 0] = np.nan
                    A_1h_95_vals[0] = np.nan

                # Identify upper limits
                is_ul = (A_1h - 2 * yerr[0]) <= 0
                is_det = ~is_ul

                # Apply horizontal offset to distinguish variants within a tracer
                x_offset = (var_idx - (len(variant_defs) - 1) / 2.0) * x_offset_scale / len(tracer_defs)
                x_tracer_offset = (tracer_idx - (len(tracer_defs) - 1) / 2.0) * x_offset_scale
                z_offset = z_centers + x_tracer_offset + x_offset

                color = colors_for_tracer[var_idx]
                marker = variant_markers[var_key]
                label = f"{tracer_label} {var_key}" if tracer_idx == 0 and var_idx == 0 else None

                # Plot detections
                if np.any(is_det):
                    ax.errorbar(z_offset[is_det], A_1h[is_det],
                               yerr=np.array([yerr[0][is_det], yerr[1][is_det]]),
                               marker=marker, color=color,
                               linestyle='None',
                               markersize=5, capsize=3, linewidth=1.5, alpha=0.8)
                
                # Plot upper limits
                if np.any(is_ul):
                    ul_vals = A_1h_95_vals[is_ul]
                    xs_ul = z_offset[is_ul]
                    ax.plot(xs_ul, ul_vals, marker="v", color=color,
                           markersize=5, alpha=0.8, linestyle='none')
                    for x, y_top in zip(xs_ul, ul_vals):
                        ax.annotate('', xy=(x, 0.0), xytext=(x, y_top),
                                   arrowprops=dict(arrowstyle='-|>', color=color,
                                                alpha=0.8, lw=1.5))

        # Panel title
        title_text = f"CIBER {lams.get(inst, '?')} μm"
        ax.text(0.02, 0.95, title_text, transform=ax.transAxes,
                fontsize=14, verticalalignment='top', weight='bold')

        # Add tracer labels in the matching colors
        cat_display = {"DESILS": "DESI-LS", "HSC": "HSC"}
        for tracer_idx, (cat, headstr, tracer_label) in enumerate(tracer_defs):
            tracer_key_short = "DESILS" if cat == "DESILS" else ("HSC_22" if headstr == "hsc_ilt22.0" else "HSC_25")

            color = tracer_colors[tracer_key_short][0]
            ax.text(0.04 + 0.28 * tracer_idx, 0.90, cat_display[cat],
                    transform=ax.transAxes, fontsize=11, color=color,
                    weight='bold', va='top', ha='left')

        ax.grid(True, alpha=0.3)
        ax.set_xlim(zbinedges[0], zbinedges[-1])
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1e0)
        ax.tick_params(labelsize=12)

    # Set axis labels
    axes[0].set_ylabel(r"$A_{1h}^{\rm Ig}$ [arb. unit]", fontsize=15)
    axes[1].set_ylabel(r"$A_{1h}^{\rm Ig}$ [arb. unit]", fontsize=15)
    axes[1].set_xlabel("Redshift (z)", fontsize=15)

    fig.subplots_adjust(left=0.11, right=0.98, top=0.92, bottom=0.10, hspace=0.2, wspace=0.07)

    stem = figdir / f"a1h_vs_redshift_alternate_layout_lMax={lMax}"
    _savefig(fig, stem, args.fig_fmt)
    plt.close(fig)
    print(f"[plot_a1h_vs_redshift_alternate_layout] generated {stem.with_suffix('.pdf')}")



def _plot_a1h_model_pred_vs_redshift(args: argparse.Namespace, ell_eval: float = 10000.0, ylim: tuple = (4e-3, 5)) -> None:
    """Compare fitted one-halo power versus model-predicted one-halo power vs redshift.

    The figure mirrors the structure of the existing spectrum-panel plotting helpers:
    it rebuilds the model from the saved fit parameters, attaches the one-halo template
    using the same helper as the spectrum plots, and evaluates both the fit and the
    one-halo prediction at the requested multipole.
    """
    figdir = Path(args.figdir) / args.fitstr_cross
    figdir.mkdir(parents=True, exist_ok=True)

    lMax = args.lmax[-1] if args.lmax else args.lmax_compare
    fitstr = args.fitstr_cross
    maskstr = getattr(args, "maskstr", None)
    result_source_label = "merged" if maskstr in (None, "", "merged") else str(maskstr)

    zbinedges = args.zbinedges

    tracer_defs = [
        ("DESILS", None, r"DESI-LS ($z_{\rm AB} < 22$)"),
        ("HSC", "hsc_ilt25.0", r"HSC ($18<i_{\rm AB}<25$)"),
    ]
    lams = {1: 1.1, 2: 1.8}
    shade_colors = ("#f5f5f5", "#eeeeee")
    tracer_colors = {
        "DESILS": ["C2", "C2", "C2", "C2"],
        "HSC_22": ["#E45DA8", "#E45DA8", "#E45DA8", "#E45DA8"],
        "HSC_25": ["#E45DA8", "#E45DA8", "#E45DA8", "#E45DA8"],
    }
    colors = {"DESILS": "C2", "HSC": "#E45DA8"}

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, sharey=True)

    for band_idx, inst in enumerate([1, 2]):
        ax = axes[band_idx]
        ax.set_xscale("linear")
        ax.set_yscale("log")

        for tracer_idx, (cat, headstr, tracer_label) in enumerate(tracer_defs):
            # Load results using merged_jh14 helper with maskstr, matching _plot_spectra_summary
            res = _load_cross_results_merged_jh14(
                args.datadir_cross, cat, headstr, fitstr, lMax, maskstr=args.maskstr
            )
            if res is None:
                fpath = _cross_fpath(args.datadir_cross, cat, headstr, fitstr, lMax, maskstr=args.maskstr)
                print(f"[plot_a1h_model_pred_vs_redshift] missing {fpath.name}")
                continue

            inst_list = list(res["inst_list"])
            if inst not in inst_list:
                continue
            inst_idx = inst_list.index(inst)

            zbinedges = np.asarray(res["zbinedges"])
            z_centers = 0.5 * (zbinedges[:-1] + zbinedges[1:])
            n_zbins = len(zbinedges) - 1

            for zbin_idx in range(n_zbins):
                z0, z1 = zbinedges[zbin_idx], zbinedges[zbin_idx + 1]
                ax.axvspan(z0, z1, color=shade_colors[zbin_idx % 2], alpha=0.22, zorder=0)

            # Load model configuration from results, same as _plot_spectra_summary
            use_powerlaw_2h = res.get("use_powerlaw_2h", True)
            alpha_2h_fixed = res.get("alpha_2h_fixed", 0.0)
            use_linear_2h = res.get("use_linear_2h", False)
            
            # Regenerate linear 2H templates if needed (with high ell_max for full plotting range)
            dl_2h_lin_per_zbin = {}
            if use_linear_2h:
                from ciber.theory.cross_ps_parametric_model import _compute_linear_2h_templates_per_zbin
                zbinedges_res = res.get("zbinedges", np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
                dl_2h_lin_per_zbin = _compute_linear_2h_templates_per_zbin(zbinedges_res, 1.2e5, verbose=False)

            pnames = res.get("param_names_fitted")
            sigma_damp_fixed_map = _parse_sigma_damp_fixed_mapping(args)
            
            # Build model once with ell range that includes ell_eval for interpolation
            ell_m = np.logspace(np.log10(100), np.log10(1.2e5), 500)
            model = CrossPowerSpectrumModel(
                lb=ell_m, use_powerlaw_2h=use_powerlaw_2h,
                alpha_2h_fixed=alpha_2h_fixed,
                use_astrometry_damping=True,
                use_linear_2h=use_linear_2h,
                dl_2h_lin_per_zbin=dl_2h_lin_per_zbin,
            )
            
            fit_vals = np.full(n_zbins, np.nan)
            fit_lo = np.full(n_zbins, np.nan)
            fit_hi = np.full(n_zbins, np.nan)
            pred_vals = np.full(n_zbins, np.nan)

            for zbin_idx in range(n_zbins):
                if cat == "HSC" and zbin_idx == 0:
                    continue

                params = np.asarray(res["params"])[inst_idx, zbin_idx, :]
                params_err = np.asarray(res["params_err"])[inst_idx, zbin_idx, :]
                params_16 = np.asarray(res.get("params_16", res["params"] - res["params_err"]))[inst_idx, zbin_idx, :]
                params_84 = np.asarray(res.get("params_84", res["params"] + res["params_err"]))[inst_idx, zbin_idx, :]
                n_params_stored = int(np.sum(~np.isnan(params)))
                params = params[:n_params_stored]
                params_err = params_err[:n_params_stored]
                params_16 = params_16[:n_params_stored]
                params_84 = params_84[:n_params_stored]

                pnf_bin = pnames[inst_idx, zbin_idx] if pnames is not None else None
                use_popmix = bool(res.get("onehalo_fit_popmix", False))
                use_damping = (pnf_bin is not None and
                               any("damp" in str(p).lower() for p in pnf_bin))
                sigma_damp_fixed_for_inst = sigma_damp_fixed_map.get(int(inst), None)
                if sigma_damp_fixed_for_inst is not None:
                    use_damping = True

                # Attach one-halo template to model, matching _plot_spectra_summary
                fit_result = {
                    "params": params,
                    "params_err": params_err,
                    "use_astrometry_damping": use_damping,
                    "samples": res.get("samples", np.empty((0,)))[inst_idx, zbin_idx] if res.get("samples") is not None else None,
                    "param_names_fitted": pnf_bin,
                    "onehalo_mode": bool(res.get("onehalo_mode", False)),
                    "onehalo_output_dir": res.get("onehalo_output_dir", ""),
                    "onehalo_generate_type": res.get("onehalo_generate_type", "bulk"),
                    "onehalo_fsat_model": res.get("onehalo_fsat_model", "single"),
                    "onehalo_population": res.get("onehalo_population", getattr(args, 'onehalo_population', 'combined')),
                    "onehalo_fit_popmix": bool(res.get("onehalo_fit_popmix", getattr(args, 'onehalo_fit_popmix', False))),
                    "onehalo_concentration_scale": float(res.get("onehalo_concentration_scale", getattr(args, 'concentration_scale', 1.0))),
                    "inst": int(inst),
                    "cat": cat,
                }
                attach_onehalo_template_to_model(
                    model, fit_result, z_bin_index=zbin_idx, use_default_if_missing=False, zbinedges=zbinedges
                )

                # Evaluate model over ell_m range at median parameters, then interpolate to ell_eval
                sd_med = (
                    sigma_damp_fixed_for_inst
                    if sigma_damp_fixed_for_inst is not None
                    else resolve_full_param_value(
                        params,
                        pnf_bin,
                        "sigma_damp",
                        use_astrometry_damping=use_damping,
                        use_onehalo_popmix=use_popmix,
                    )
                ) if use_damping else None
                f_pop_med = (
                    resolve_full_param_value(
                        params,
                        pnf_bin,
                        "f_pop",
                        use_astrometry_damping=use_damping,
                        use_onehalo_popmix=use_popmix,
                    )
                    if use_popmix else None
                )

                comps = model.model_components(
                    ell_m,
                    *params[:5],
                    sigma_damp=sd_med,
                    z_bin_index=zbin_idx,
                    f_pop=f_pop_med,
                )
                if comps is not None and 'one_halo' in comps:
                    fit_vals[zbin_idx] = float(np.interp(ell_eval, ell_m, comps['one_halo']))

                # Extract uncertainty from samples if available, otherwise from parameter percentiles
                samples_bin = fit_result.get("samples", None)
                if samples_bin is not None:
                    s = np.asarray(samples_bin, dtype=float)
                    if s.ndim == 2 and s.shape[0] > 1:
                        nfull = 5 + (1 if use_popmix else 0) + (1 if use_damping else 0)
                        if s.shape[1] == nfull:
                            sfull = s
                        else:
                            sfull = expand_fit_samples_to_full_vector(
                                s,
                                np.asarray(params[:nfull], dtype=float),
                                param_names_fitted=pnf_bin,
                                use_astrometry_damping=use_damping,
                                use_onehalo_popmix=use_popmix,
                            )

                        c1h = np.zeros(sfull.shape[0])
                        for ii in range(sfull.shape[0]):
                            f_pop_i = sfull[ii, 5] if (use_popmix and sfull.shape[1] > 5) else None
                            if use_damping:
                                damp_idx = 6 if use_popmix else 5
                                sd_i = sfull[ii, damp_idx] if sfull.shape[1] > damp_idx else None
                            else:
                                sd_i = None
                            cc = model.model_components(
                                ell_m,
                                sfull[ii, 0], sfull[ii, 1], sfull[ii, 2], sfull[ii, 3], sfull[ii, 4],
                                sigma_damp=sd_i,
                                z_bin_index=zbin_idx,
                                f_pop=f_pop_i,
                            )
                            if cc is not None and 'one_halo' in cc:
                                c1h[ii] = float(np.interp(ell_eval, ell_m, cc['one_halo']))
                        fit_lo[zbin_idx] = float(np.percentile(c1h, 16))
                        fit_hi[zbin_idx] = float(np.percentile(c1h, 84))
                else:
                    # Fallback: evaluate at parameter percentiles
                    sd_lo = (
                        sigma_damp_fixed_for_inst
                        if sigma_damp_fixed_for_inst is not None
                        else resolve_full_param_value(
                            params_16,
                            pnf_bin,
                            "sigma_damp",
                            use_astrometry_damping=use_damping,
                            use_onehalo_popmix=use_popmix,
                        )
                    ) if use_damping else None
                    f_pop_lo = (
                        resolve_full_param_value(
                            params_16,
                            pnf_bin,
                            "f_pop",
                            use_astrometry_damping=use_damping,
                            use_onehalo_popmix=use_popmix,
                        )
                        if use_popmix else None
                    )

                    comps_lo = model.model_components(
                        ell_m,
                        *params_16[:5],
                        sigma_damp=sd_lo,
                        z_bin_index=zbin_idx,
                        f_pop=f_pop_lo,
                    )
                    if comps_lo is not None and 'one_halo' in comps_lo:
                        fit_lo[zbin_idx] = float(np.interp(ell_eval, ell_m, comps_lo['one_halo']))

                    sd_hi = (
                        sigma_damp_fixed_for_inst
                        if sigma_damp_fixed_for_inst is not None
                        else resolve_full_param_value(
                            params_84,
                            pnf_bin,
                            "sigma_damp",
                            use_astrometry_damping=use_damping,
                            use_onehalo_popmix=use_popmix,
                        )
                    ) if use_damping else None
                    f_pop_hi = (
                        resolve_full_param_value(
                            params_84,
                            pnf_bin,
                            "f_pop",
                            use_astrometry_damping=use_damping,
                            use_onehalo_popmix=use_popmix,
                        )
                        if use_popmix else None
                    )

                    comps_hi = model.model_components(
                        ell_m,
                        *params_84[:5],
                        sigma_damp=sd_hi,
                        z_bin_index=zbin_idx,
                        f_pop=f_pop_hi,
                    )
                    if comps_hi is not None and 'one_halo' in comps_hi:
                        fit_hi[zbin_idx] = float(np.interp(ell_eval, ell_m, comps_hi['one_halo']))

                # Evaluate model prediction (predicted one-halo power with A_1h not amplified)
                # Load one-halo spectrum prediction and interpolate to ell_eval
                try:
                    from ciber.plotting.gal_plotting_fns import load_onehalo_spectrum
                    
                    onehalo_output_dir = getattr(args, "onehalo_dir", None) or fit_result.get("onehalo_output_dir", None)
                    if onehalo_output_dir and os.path.exists(onehalo_output_dir):
                        bandstr_select = "sdss_z" if cat == "DESILS" else "hsc_i"
                        mag_cut = 22.0 if cat == "DESILS" else 25.0
                        generate_type = getattr(args, "onehalo_generate_type", fit_result.get("onehalo_generate_type", "bulk"))
                        fsat_model = getattr(args, "onehalo_fsat_model", fit_result.get("onehalo_fsat_model", "single"))
                        population = getattr(args, "onehalo_population", fit_result.get("onehalo_population", "combined"))
                        concentration_scale = float(fit_result.get("onehalo_concentration_scale", 1.0))

                        oh_data_Ig = load_onehalo_spectrum(
                            onehalo_output_dir,
                            fsat_model,
                            bandstr_select,
                            inst=int(inst),
                            mag_min=18.0,
                            mag_cut=mag_cut,
                            z0=0.05,
                            mode="Ig",
                            generate_type=generate_type,
                            concentration_scale=concentration_scale,
                            population=population,
                        )
                        if oh_data_Ig is not None:
                            ell_1h = np.asarray(oh_data_Ig["ell_arr"], dtype=float)
                            dl_1h = np.asarray(oh_data_Ig["dl_spectrum"], dtype=float)
                            if dl_1h.ndim == 1:
                                dl_1h_zbin = dl_1h
                            elif dl_1h.ndim >= 2:
                                dl_1h_zbin = dl_1h[zbin_idx]
                            else:
                                dl_1h_zbin = np.array([], dtype=float)

                            if zbin_idx == 0:
                                dl_1h_zbin *= 0.25

                            if ell_1h.size > 0 and dl_1h_zbin.size > 0:
                                pred_vals[zbin_idx] = float(np.interp(ell_eval, ell_1h, dl_1h_zbin))
                except Exception as exc:
                    pass

            valid_fit = np.isfinite(fit_vals)
            valid_pred = np.isfinite(pred_vals)

            fit_err_lo = np.where(valid_fit, fit_vals - fit_lo, np.nan)
            fit_err_hi = np.where(valid_fit, fit_hi - fit_vals, np.nan)
            is_ul_fit = valid_fit & np.isfinite(fit_err_lo) & ((fit_vals - 2.0 * fit_err_lo) <= 0.0)
            is_det_fit = valid_fit & ~is_ul_fit

            dz_shift = 0.04 * (tracer_idx - (len(tracer_defs) - 1) / 2.0)
            z_centers_shifted = z_centers + dz_shift

            if np.any(is_det_fit):
                ax.errorbar(
                    z_centers_shifted[is_det_fit], fit_vals[is_det_fit],
                    yerr=np.array([fit_vals[is_det_fit] - fit_lo[is_det_fit], fit_hi[is_det_fit] - fit_vals[is_det_fit]]),
                    fmt='o',
                    color=colors[cat],
                    markerfacecolor=colors[cat],
                    markeredgecolor=colors[cat],
                    linestyle='None',
                    markersize=7,
                    capsize=7,
                    linewidth=2.0,
                    capthick=2,
                    alpha=0.9,
                    label=f"This work" if tracer_idx == 0 and band_idx == 0 else None,
                )

            if np.any(is_ul_fit):
                # choose floor exactly like the A2h style
                ymin = max(ylim[0], 1e-3)   # or just ymin = 1e-3 if your ylim starts <= 1e-3

                ul_vals = fit_vals[is_ul_fit] + 2.0 * fit_err_hi[is_ul_fit]
                xs_ul = z_centers_shifted[is_ul_fit]

                # horizontal cap at UL value
                ax.plot(
                    xs_ul, ul_vals,
                    marker='_', linestyle='none',
                    color=colors[cat],
                    markersize=12, markeredgewidth=2,
                    alpha=0.85, zorder=6
                )

                # downward arrow
                for x, y_top in zip(xs_ul, ul_vals):
                    if np.isfinite(y_top) and y_top > ymin:
                        ax.annotate(
                            '',
                            xy=(x, ymin), xytext=(x, y_top),
                            arrowprops=dict(
                                arrowstyle='-|>',
                                color=colors[cat],
                                alpha=0.85,
                                lw=2.5,
                                mutation_scale=15,
                                shrinkA=0, shrinkB=0
                            ),
                            zorder=6
                        )


            if np.any(valid_pred):
                ax.plot(
                    z_centers_shifted[valid_pred], pred_vals[valid_pred],
                    color=colors[cat],
                    linestyle='--',
                    marker='^',
                    linewidth=1.8,
                    markersize=4,
                    alpha=0.95,
                    label=f"IGL one-halo prediction" if tracer_idx == 0 and band_idx == 0 else None,
                )

        if band_idx==0:
            ax.legend(fontsize=16, loc=2, bbox_to_anchor=(0.02, 1.25), ncols=2)

        cat_display = {"DESILS": "DESI-LS ($z_{\\rm AB} < 22$)", "HSC": "HSC ($18 < i_{\\rm AB} < 25$)"}
        for tracer_idx, (cat, headstr, tracer_label) in enumerate(tracer_defs):
            tracer_key_short = "DESILS" if cat == "DESILS" else ("HSC_22" if headstr == "hsc_ilt22.0" else "HSC_25")
            color = tracer_colors[tracer_key_short][0]

            ax.text(0.4, 0.95 - 0.12 * tracer_idx,
                    fr"CIBER {lams[inst]:.1f} $\mu$m $\times$ {cat_display[cat]}",
                    transform=ax.transAxes, fontsize=14, color=color,
                    va='top', ha='left')


        ax.text(0.05, 0.1, '$\\ell='+str(int(ell_eval))+'$', transform=ax.transAxes, fontsize=18, color='black', va='bottom', ha='left')

        ax.grid(True, alpha=0.3)
        ax.set_xlim(zbinedges[0], zbinedges[-1])
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=12)

    # ylabel = '$C_{\\ell='+str(int(ell_eval))+'}^{\\rm 1h, Ig}$ [nW m$^{-2}$ sr$^{-1}$]'
    ylabel = '$D_{\\ell, \\rm 1h}^{\\rm Ig}$ [nW m$^{-2}$ sr$^{-1}$]'

    axes[0].set_ylabel(ylabel, fontsize=14)
    axes[1].set_ylabel(ylabel, fontsize=14)
    axes[1].set_xlabel("Redshift (z)", fontsize=14)

    fig.subplots_adjust(wspace=0.03, hspace=0.02)

    # fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.10, hspace=0.18, wspace=0.07)
    stem = figdir / f"a1h_model_pred_vs_redshift_lMax={lMax}_ell={int(ell_eval)}"
    _savefig(fig, stem, args.fig_fmt)
    plt.close(fig)
    print(f"[plot_a1h_model_pred_vs_redshift] generated {stem.with_suffix('.pdf')}")


def _plot_a1h_band_ratio_vs_redshift(args: argparse.Namespace) -> None:
    """Plot A_1h(1.1 μm) / A_1h(1.8 μm) vs. redshift for DESI-LS and HSC (i<25).

    Two panels side-by-side, one per catalog. Each panel shows all four model
    variants. Ratios are computed directly from MCMC samples, with 68% credible
    intervals (16th to 84th percentile) shown as error bars.
    """
    figdir = Path(args.figdir) / args.fitstr_cross
    figdir.mkdir(parents=True, exist_ok=True)

    lMax = args.lmax[-1] if args.lmax else args.lmax_compare

    fitstr_fixA2h   = args.fitstr_cross + "_fixA2h_IGL"
    fitstr_biLinear = args.fitstr_cross + "_fixA2h_IGL_biLinear"
    fitstr_biQuad   = args.fitstr_cross + "_fixA2h_IGL_biQuadratic"

    variant_defs = [
        ("full",     args.fitstr_cross),
        ("fixA2h",   fitstr_fixA2h),
        ("biLinear", fitstr_biLinear),
        ("biQuad",   fitstr_biQuad),
    ]

    panel_defs = [
        ("DESILS", None,          r"DESI-LS ($z_{\rm AB} < 22$)"),
        ("HSC",    "hsc_ilt25.0", r"HSC ($18 < i_{\rm AB} < 25$)"),
    ]

    var_styles = {
        "full":     dict(color="#2d2d2d", marker="o", label="Full model (Float $A_{2h}$, $A_{1h}$)"),
        "fixA2h":   dict(color="#666666", marker="^", label=r"Fix $A_{2h}$; $b_I=1$"),
        "biLinear": dict(color="#999999", marker="D", label=r"Fix $A_{2h}$; $b_I=1+0.6z$"),
        "biQuad":   dict(color="#cccccc", marker="v", label=r"Fix $A_{2h}$; $b_I=(1+z)^2$"),
    }

    shade_colors  = ("#e8f4ff", "#fff3e6")
    x_offset_scale = 0.04

    def _extract_a1h_ratios_from_samples(res_fid, res_jh14, n_zbins):
        """Compute A_1h(TM1) / A_1h(TM2) ratios from samples for all z-bins.
        
        Returns (ratio_med, ratio_lo, ratio_hi) arrays, or None if data unavailable.
        Uses 16th/84th percentiles from sample distributions.
        """
        inst_list = list(res_fid.get("inst_list", []))
        if 1 not in inst_list or 2 not in inst_list:
            return None

        # Try to get chains; fall back to params+params_16/84 if unavailable
        chains_fid = res_fid.get("chains")  # shape: (n_walkers, n_steps, n_params)
        chains_jh14 = res_jh14.get("chains") if res_jh14 else None

        pnames = res_fid.get("param_names_fitted")
        a1h_idx = {}
        for inst in [1, 2]:
            ii = inst_list.index(inst)
            a1h_idx[inst] = 1
            if pnames is not None:
                try:
                    rep = pnames[ii, 0]
                    if rep is not None:
                        for k, nm in enumerate(rep):
                            if "A_1h" in str(nm) or "A_{1h}" in str(nm):
                                a1h_idx[inst] = k
                                break
                except (IndexError, TypeError):
                    pass

        ratio_med = np.full(n_zbins, np.nan)
        ratio_lo = np.full(n_zbins, np.nan)
        ratio_hi = np.full(n_zbins, np.nan)

        for zi in range(n_zbins):
            # Use JHlt14 for z<0.2 bin (zi=0), fiducial otherwise
            if zi == 0 and res_jh14 is not None:
                res_use = res_jh14
                chains_use = chains_jh14
                zi_use = 0
            else:
                res_use = res_fid
                chains_use = chains_fid
                zi_use = zi

            if chains_use is None:
                # Fallback: use parameter values and percentiles
                inst_list_u = list(res_use.get("inst_list", []))
                if 1 not in inst_list_u or 2 not in inst_list_u:
                    continue
                i1 = inst_list_u.index(1)
                i2 = inst_list_u.index(2)
                a1 = res_use["params"][i1, zi_use, a1h_idx[1]]
                a2 = res_use["params"][i2, zi_use, a1h_idx[2]]
                if a2 > 0:
                    ratio_med[zi] = a1 / a2
                    if "params_16" in res_use and "params_84" in res_use:
                        a1_lo = res_use["params_16"][i1, zi_use, a1h_idx[1]]
                        a1_hi = res_use["params_84"][i1, zi_use, a1h_idx[1]]
                        a2_lo = res_use["params_16"][i2, zi_use, a1h_idx[2]]
                        a2_hi = res_use["params_84"][i2, zi_use, a1h_idx[2]]
                        r_med = ratio_med[zi]
                        ratio_lo[zi] = r_med - (a1_lo / a2_hi)
                        ratio_hi[zi] = (a1_hi / a2_lo) - r_med
                    else:
                        ratio_lo[zi] = ratio_hi[zi] = 0.0
            else:
                # Extract samples for both instruments and compute ratio distribution
                i1 = inst_list.index(1)
                i2 = inst_list.index(2)
                idx1 = a1h_idx[1]
                idx2 = a1h_idx[2]
                
                # Reshape chains to (n_samples, n_params)
                n_walkers, n_steps = chains_use.shape[:2]
                samples_flat = chains_use.reshape(-1, chains_use.shape[-1])
                
                a1_samples = samples_flat[:, idx1]
                a2_samples = samples_flat[:, idx2]
                
                # Compute ratio for each sample, excluding invalid ratios
                valid = (a1_samples > 0) & (a2_samples > 0)
                if np.sum(valid) > 1:
                    ratio_samples = a1_samples[valid] / a2_samples[valid]
                    ratio_med[zi] = np.median(ratio_samples)
                    ratio_lo[zi] = ratio_med[zi] - np.percentile(ratio_samples, 16)
                    ratio_hi[zi] = np.percentile(ratio_samples, 84) - ratio_med[zi]

        return ratio_med, ratio_lo, ratio_hi

    fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True, sharey=True)

    for row, (cat, headstr, panel_label) in enumerate(panel_defs):
        ax = axes[row]

        fid_res  = {}
        jh14_res = {}
        zbinedges = None
        for var_key, fitstr_name in variant_defs:
            fpath = _cross_fpath(args.datadir_cross, cat, headstr, fitstr_name, lMax)
            if fpath.exists():
                r = load_fit_results_npz(str(fpath))
                fid_res[var_key] = r
                if zbinedges is None and "zbinedges" in r:
                    zbinedges = r["zbinedges"]
            elif var_key in ("full", "fixA2h"):
                print(f"[plot_a1h_band_ratio] missing fiducial {fpath.name}")
            fpath_jh14 = _cross_fpath(args.datadir_cross, cat, headstr, fitstr_name, lMax, maskstr='JHlt14')
            if fpath_jh14.exists():
                jh14_res[var_key] = load_fit_results_npz(str(fpath_jh14))
            elif var_key in ("full", "fixA2h"):
                print(f"[plot_a1h_band_ratio] missing JHlt14 {fpath_jh14.name}")

        if zbinedges is None or "full" not in fid_res:
            print(f"[plot_a1h_band_ratio] no results for {cat}/{headstr}, skipping panel")
            continue

        n_zbins   = len(zbinedges) - 1
        z_centers = 0.5 * (zbinedges[:-1] + zbinedges[1:])

        for j in range(n_zbins):
            ax.axvspan(zbinedges[j], zbinedges[j + 1], color=shade_colors[j % 2], alpha=0.22, zorder=0)

        for var_idx, (var_key, _) in enumerate(variant_defs):
            if var_key not in fid_res:
                continue
            
            out = _extract_a1h_ratios_from_samples(fid_res[var_key], jh14_res.get(var_key), n_zbins)
            if out is None:
                continue
            ratio_med, ratio_lo, ratio_hi = out

            # Only plot points with valid median values
            valid = ~np.isnan(ratio_med)
            if not np.any(valid):
                continue

            x_off = z_centers + (var_idx - (len(variant_defs) - 1) / 2.0) * x_offset_scale
            st = var_styles[var_key]

            ax.errorbar(x_off[valid], ratio_med[valid],
                        yerr=np.array([ratio_lo[valid], ratio_hi[valid]]),
                        marker=st["marker"], color=st["color"],
                        label=st["label"], linestyle='None',
                        markersize=5, capsize=3, linewidth=1.5, alpha=1.0)

        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
        ax.set_title(panel_label, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(zbinedges[0], zbinedges[-1])
        ax.set_ylim(0, 2)
        ax.set_xlabel("Redshift (z)", fontsize=13)
        ax.tick_params(labelsize=12)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.00),
               ncol=2, fontsize=13, frameon=True)

    axes[0].set_ylabel(r"$A_{1h}^{1.1\,\mu{\rm m}} \,/\, A_{1h}^{1.8\,\mu{\rm m}}$", fontsize=14)

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    stem = figdir / f"a1h_band_ratio_vs_redshift_lMax={lMax}"
    _savefig(fig, stem, args.fig_fmt)
    plt.close(fig)
    print(f"[plot_a1h_band_ratio_vs_redshift] generated {stem.with_suffix('.pdf')}")


def _plot_d_ell_1h_evolution(args: argparse.Namespace) -> None:
    """Plot best-fit one-halo D_ℓ templates vs ℓ for all redshift bins, in a 3×2 grid.

    Rows: DESI-LS, HSC i<22, HSC i<25. Columns: TM1 (1.1 μm, blues), TM2 (1.8 μm, reds).
    Each panel overlays 5 redshift bins as spectra with equal-spaced color shades.
    Uses the fixA2h_IGL configuration. For each z-bin, the 2h D_ℓ (dashed, same shade) is
    computed directly from the fixed IGL A_2h parameter. A discrete colorbar per panel
    indicates the five redshift bins.
    """
    import matplotlib.colors as mcolors
    from ciber.theory.cross_ps_parametric_model import CrossPowerSpectrumModel

    figdir = Path(args.figdir) / args.fitstr_cross
    figdir.mkdir(parents=True, exist_ok=True)

    lMax = args.lmax[-1] if args.lmax else args.lmax_compare
    fitstr_fixA2h = args.fitstr_cross + "_fixA2h_IGL"

    row_defs = [
        ("DESILS", None,          r"DESI-LS ($z_{\rm AB} < 22$)"),
        ("HSC",    "hsc_ilt22.0", r"HSC ($18<i_{\rm AB}<22$)"),
        ("HSC",    "hsc_ilt25.0", r"HSC ($18<i_{\rm AB}<25$)"),
    ]

    # Load fixA2h_IGL results for all three catalogs
    results_by_row = {}
    for cat, headstr, label in row_defs:
        fpath = _cross_fpath(args.datadir_cross, cat, headstr, fitstr_fixA2h, lMax)
        if fpath.exists():
            results_by_row[(cat, headstr)] = load_fit_results_npz(str(fpath))
        else:
            print(f"[plot_d_ell_1h_evolution] missing {fpath.name}")
            results_by_row[(cat, headstr)] = None

    # Evaluation grid: log-spaced ℓ covering the full fit range
    ell_grid = np.logspace(np.log10(200), np.log10(1.1e5), 300)

    # Discrete colors for the 5 redshift bins
    discrete_colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    lams  = {1: 1.1, 2: 1.8}
    n_zbins = 5

    fig, axes = plt.subplots(3, 2, figsize=(6.5, 7), sharex=True, sharey=True)


    fig.suptitle('One-halo $D_\\ell$ evolution with redshift', fontsize=16, y=1.01)
    for row_idx, (cat, headstr, row_label) in enumerate(row_defs):
        res = results_by_row[(cat, headstr)]

        for col_idx, inst in enumerate([1, 2]):
            ax = axes[row_idx, col_idx]
            ax.set_xscale("log")
            ax.set_yscale("log")

            if res is None:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", color="gray", fontsize=10)
                continue

            inst_list = list(res["inst_list"])
            if inst not in inst_list:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", color="gray", fontsize=10)
                continue
            inst_idx = inst_list.index(inst)

            zbinedges = res["zbinedges"]
            z_centers = 0.5 * (zbinedges[:-1] + zbinedges[1:])

            # Parameter indices in the stored params array:
            # [A_2h, A_1h, mu_1h, sigma_1h, A_shot, sigma_damp]
            idx_A2h, idx_A1h, idx_mu, idx_sig = 0, 1, 2, 3

            # Build model instance on dense ell_grid
            use_powerlaw_2h = res.get("use_powerlaw_2h", True)
            alpha_2h_fixed = res.get("alpha_2h_fixed", -0.0)
            use_linear_2h = res.get("use_linear_2h", False)
            
            # Regenerate linear 2H templates if needed (with high ell_max for full plotting range)
            dl_2h_lin_per_zbin = {}
            if use_linear_2h:
                from ciber.theory.cross_ps_parametric_model import _compute_linear_2h_templates_per_zbin
                zbinedges = res.get("zbinedges", np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
                dl_2h_lin_per_zbin = _compute_linear_2h_templates_per_zbin(zbinedges, 1e5, verbose=False)
            
            model = CrossPowerSpectrumModel(
                lb=ell_grid,
                use_powerlaw_2h=use_powerlaw_2h,
                alpha_2h_fixed=alpha_2h_fixed,
                use_astrometry_damping=False,
                use_linear_2h=use_linear_2h,
                dl_2h_lin_per_zbin=dl_2h_lin_per_zbin,
            )

            A_1h_lo_arr = res.get("params_16")
            A_1h_hi_arr = res.get("params_84")
            A_1h_95_arr = res.get("params_95")

            for zbin_idx in range(n_zbins):
                color = discrete_colors[zbin_idx]

                # 2h D_ℓ from fixed IGL A_2h — per z-bin, same color, dashed
                A_2h_z = float(res["params"][inst_idx, zbin_idx, idx_A2h])
                dl_2h  = model.powerlaw_2h_component(ell_grid, A_2h_z, -0.0)
                ax.plot(ell_grid, dl_2h, color=color, lw=1.5, linestyle="dashed")

                # 1h D_ℓ from fitted A_1h with fixed mu_1h, sigma_1h
                A_1h  = float(res["params"][inst_idx, zbin_idx, idx_A1h])
                mu_1h = float(res["params"][inst_idx, zbin_idx, idx_mu])
                sig1h = float(res["params"][inst_idx, zbin_idx, idx_sig])

                if A_1h_lo_arr is not None and A_1h_hi_arr is not None:
                    A_1h_lo = float(A_1h_lo_arr[inst_idx, zbin_idx, idx_A1h])
                    A_1h_hi = float(A_1h_hi_arr[inst_idx, zbin_idx, idx_A1h])
                else:
                    sigma_A1h = float(res["params_err"][inst_idx, zbin_idx, idx_A1h])
                    A_1h_lo   = max(0.0, A_1h - sigma_A1h)
                    A_1h_hi   = A_1h + sigma_A1h

                is_ul = A_1h_lo <= 0.0

                if not is_ul and A_1h > 0:
                    dl_best = model.lognormal_component(ell_grid, A_1h, mu_1h, sig1h)
                    dl_lo   = model.lognormal_component(ell_grid, max(A_1h_lo, 0.0), mu_1h, sig1h)
                    dl_hi   = model.lognormal_component(ell_grid, A_1h_hi, mu_1h, sig1h)
                    ax.plot(ell_grid, dl_best, color=color, lw=1.8, zorder=3)
                    ax.fill_between(ell_grid, dl_lo, dl_hi, color=color, alpha=0.25, zorder=2)
                else:
                    # Upper limit: dotted thin curve at 95th-pct A_1h
                    if A_1h_95_arr is not None:
                        A_1h_ul = float(A_1h_95_arr[inst_idx, zbin_idx, idx_A1h])
                    else:
                        A_1h_ul = A_1h_hi
                    if A_1h_ul > 0:
                        dl_ul = model.lognormal_component(ell_grid, A_1h_ul, mu_1h, sig1h)
                        ax.plot(ell_grid, dl_ul, color=color, lw=1.2,
                                linestyle=":", alpha=0.3, zorder=2)

            # Panel annotation
            ax.text(0.03, 0.97,
                    f"CIBER {lams[inst]} μm × {row_label}",
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment="top")
            ax.grid(alpha=0.3)

            ax.set_ylim(1e-3, 2)
            ax.set_xlim(280, 1.1e5)

            # Discrete colorbar only on right-hand panels
            if col_idx == 1:
                cmap_disc   = mcolors.ListedColormap(discrete_colors)
                norm_disc   = mcolors.BoundaryNorm(zbinedges, n_zbins)
                sm = plt.cm.ScalarMappable(cmap=cmap_disc, norm=norm_disc)
                sm.set_array([])
                cax = ax.inset_axes([1.04, 0, 0.05, 1.0])
                cbar = fig.colorbar(sm, cax=cax, ticks=z_centers)
                cbar.set_ticklabels(
                    [f"{zbinedges[i]:.1f}–{zbinedges[i+1]:.1f}" for i in range(n_zbins)],
                    fontsize=8,
                )
                cbar.ax.tick_params(labelsize=8, length=2)
                cbar.ax.yaxis.set_ticks_position("right")
                cbar.ax.yaxis.set_label_position("right")

    # Shared axis labels
    for row in range(3):
        axes[row, 0].set_ylabel(r"$D_\ell^{Ig}$", fontsize=14)

    for col in range(2):
        axes[2, col].set_xlabel(r"$\ell$", fontsize=14)


    fig.subplots_adjust(left=0.10, right=0.97, top=0.97, bottom=0.07,
                        hspace=0.05, wspace=0.1)

    stem = figdir / f"d_ell_1h_evolution_lMax={lMax}"
    _savefig(fig, stem, args.fig_fmt)
    plt.close(fig)
    print(f"[plot_d_ell_1h_evolution] generated {stem.with_suffix('.pdf')}")


def _plot_r1h_ratio(args: argparse.Namespace) -> None:
    """Plot R_1h(z) = A_1h^{Ig}(z) / A_1h^{II}(z) vs redshift, alongside IHL templates.

    Left panel: IHL auto-spectrum 1h D_ℓ^{II}(ℓ) templates for each redshift bin.
    Right panel: R_1h(z) for each catalog × band combination, using the fixA2h_IGL results.
    A_1h^{II}(z) is taken from the IHL 1h parameter file (args.ihl_params).
    """
    from ciber.theory.cross_ps_parametric_model import CrossPowerSpectrumModel

    figdir = Path(args.figdir) / args.fitstr_cross
    figdir.mkdir(parents=True, exist_ok=True)

    lMax = args.lmax[-1] if args.lmax else args.lmax_compare
    fitstr_fixA2h = args.fitstr_cross + "_fixA2h_IGL"

    # ---- Load IHL 1h parameters ----
    ihl_path = Path(args.ihl_params)
    if not ihl_path.exists():
        print(f"[plot_r1h_ratio] IHL params file not found: {ihl_path}")
        return
    ihl_data  = np.load(ihl_path, allow_pickle=True)
    params_dict_ihl   = ihl_data["params_dict"].item()   # (z_idx, slope) -> param dict
    a1h_by_slope      = ihl_data["a1h_by_slope"].item()  # slope_float -> list[A_1h^{II}]
    a1h_err_by_slope  = ihl_data["a1h_err_by_slope"].item()
    zbinedges_ihl     = ihl_data["zbinedges"]
    z_centers_ihl     = ihl_data["z_centers"]

    slope    = 1.0
    n_zbins  = len(z_centers_ihl)
    A_1h_II      = np.array(a1h_by_slope[slope])      # (n_zbins,)
    A_1h_II_err  = np.array(a1h_err_by_slope[slope])  # (n_zbins,)

    discrete_colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    lams = {1: 1.1, 2: 1.8}

    # ---- Catalog row definitions ----
    row_defs = [
        ("DESILS", None,           r"DESI-LS"),
        ("HSC",    "hsc_ilt22.0",  r"HSC $i{<}22$"),
        ("HSC",    "hsc_ilt25.0",  r"HSC $i{<}25$"),
    ]
    results_by_row = {}
    for cat, headstr, _ in row_defs:
        fpath = _cross_fpath(args.datadir_cross, cat, headstr, fitstr_fixA2h, lMax)
        results_by_row[(cat, headstr)] = load_fit_results_npz(str(fpath)) if fpath.exists() else None

    # ---- Figure: 1 row × 2 panels ----
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4.5))

    # === Left panel: IHL auto-spectrum 1h templates ===
    ax_left.set_xscale("log")
    ax_left.set_yscale("log")
    ell_grid = np.logspace(np.log10(200), np.log10(1.1e5), 300)
    model = CrossPowerSpectrumModel(
        lb=ell_grid,
        use_powerlaw_2h=True,
        alpha_2h_fixed=0.0,
        use_astrometry_damping=False,
    )
    for z_idx in range(n_zbins):
        p    = params_dict_ihl[(z_idx, slope)]
        A_II = A_1h_II[z_idx]
        dA   = A_1h_II_err[z_idx]
        mu   = p["mu_1h"]
        sig  = p["sigma_1h"]
        c    = discrete_colors[z_idx]
        zlbl = f"$z={zbinedges_ihl[z_idx]:.1f}$–${zbinedges_ihl[z_idx+1]:.1f}$"
        dl_best = model.lognormal_component(ell_grid, A_II, mu, sig)
        dl_lo   = model.lognormal_component(ell_grid, max(A_II - dA, 0.0), mu, sig)
        dl_hi   = model.lognormal_component(ell_grid, A_II + dA, mu, sig)
        ax_left.plot(ell_grid, dl_best, color=c, lw=2.0, label=zlbl, zorder=3)
        ax_left.fill_between(ell_grid, dl_lo, dl_hi, color=c, alpha=0.20, zorder=2)
    ax_left.set_xlim(280, 1.1e5)
    ax_left.set_xlabel(r"$\ell$", fontsize=13)
    ax_left.set_ylabel(r"$D_\ell^{II,{\rm 1h}}\ \left[{\rm (nW\,m^{-2}\,sr^{-1})^2}\right]$",
                       fontsize=10)
    ax_left.set_title(r"IHL auto-spectrum 1h templates", fontsize=14, fontweight="bold")

        # fig2.suptitle('One-halo fits to cross-power spectra', fontsize=18, y=1.0, fontweight='bold')

    ax_left.legend(fontsize=8, loc="lower left")
    ax_left.grid(alpha=0.3)

    # === Right panel: R_1h(z) = A_1h^{Ig} / A_1h^{II} ===
    ax_right.set_yscale("log")
    idx_A1h = 1
    # one x-offset per (cat_row, inst) pair to avoid overlap
    cat_markers   = ['o', 's', '^']
    cat_colors_r  = ['#222222', '#CC5522', '#2255CC']
    inst_ls       = {1: '-', 2: '--'}
    inst_dz       = {1: -0.012, 2: +0.012}  # small horizontal jitter per band
    cat_dz_scale  = [-1, 0, 1]               # per catalog row

    for r_idx, (cat, headstr, row_label) in enumerate(row_defs):
        res = results_by_row[(cat, headstr)]
        if res is None:
            continue
        inst_list  = list(res["inst_list"])
        zbinedges  = res["zbinedges"]
        z_c        = 0.5 * (zbinedges[:-1] + zbinedges[1:])

        A_lo_all = res.get("params_16")
        A_hi_all = res.get("params_84")
        A_95_all = res.get("params_95")

        for inst in [1, 2]:
            if inst not in inst_list:
                continue
            inst_idx = inst_list.index(inst)

            A_Ig = res["params"][inst_idx, :, idx_A1h].astype(float)
            if A_lo_all is not None and A_hi_all is not None:
                A_lo = A_lo_all[inst_idx, :, idx_A1h].astype(float)
                A_hi = A_hi_all[inst_idx, :, idx_A1h].astype(float)
            else:
                dsig = res["params_err"][inst_idx, :, idx_A1h].astype(float)
                A_lo = A_Ig - dsig
                A_hi = A_Ig + dsig
            A_95 = A_95_all[inst_idx, :, idx_A1h].astype(float) if A_95_all is not None else A_hi

            R      = A_Ig  / A_1h_II
            R_lo   = A_lo  / A_1h_II
            R_hi   = A_hi  / A_1h_II
            R_95   = A_95  / A_1h_II
            is_ul  = A_lo <= 0

            color  = cat_colors_r[r_idx]
            marker = cat_markers[r_idx]
            ls     = inst_ls[inst]
            zplot  = z_c + inst_dz[inst] * (1 + cat_dz_scale[r_idx])

            # Detections
            det = ~is_ul
            if det.any():
                ax_right.errorbar(
                    zplot[det], R[det],
                    yerr=[R[det] - R_lo[det], R_hi[det] - R[det]],
                    fmt=marker, color=color, ms=5.5, mec="k", mew=0.5,
                    capsize=3, elinewidth=1.2, zorder=4, ls="none",
                )
                if det.sum() > 1:
                    ax_right.plot(zplot[det], R[det], color=color, lw=1.0, ls=ls, zorder=3)

            # Upper limits (downward arrows at 95th-pct value)
            ul = is_ul
            if ul.any():
                for zi in np.where(ul)[0]:
                    rv = R_95[zi] if R_95[zi] > 0 else R_hi[zi]
                    if rv <= 0:
                        continue
                    ax_right.annotate(
                        "", xy=(zplot[zi], rv * 0.60), xytext=(zplot[zi], rv),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3),
                        zorder=4,
                    )
                    ax_right.plot(zplot[zi], rv, marker=marker, color=color,
                                  ms=5.5, mec="k", mew=0.5, alpha=0.7, ls="none", zorder=4)

            # Legend proxy
            ax_right.errorbar([], [], [], fmt=marker, color=color, ms=5.5, mec="k",
                               mew=0.5, capsize=3, ls=ls,
                               label=f"{row_label} {lams[inst]} μm")

    ax_right.set_xlabel(r"$z$", fontsize=13)
    ax_right.set_ylabel(
        r"$R_{\rm 1h}(z) = A_{\rm 1h}^{Ig} / A_{\rm 1h}^{II}$"
        r"$\ \left[{\rm (nW\,m^{-2}\,sr^{-1})^{-1}}\right]$",
        fontsize=9,
    )
    ax_right.set_title(r"One-halo amplitude ratio vs. redshift", fontsize=11)
    ax_right.legend(fontsize=7.5, ncol=2, loc="best")
    ax_right.set_xlim(-0.05, 1.05)
    ax_right.grid(alpha=0.3)

    fig.tight_layout()
    stem = figdir / f"r1h_ratio_lMax={lMax}"
    _savefig(fig, stem, args.fig_fmt)
    plt.close(fig)
    print(f"[plot_r1h_ratio] generated {stem.with_suffix('.pdf')}")


def _plot_ihl_and_dell_combined(args: argparse.Namespace) -> None:
    """Generates two separate figures:
    1. IHL auto-spectrum (square, equal width/height)
    2. D_ell evolution 2×3 grid (2 bands × 3 catalogs) with shared colorbar

    Figure 1: IHL auto-spectrum 1h D_ℓ^{II,1h}(ℓ) templates for each z-bin.
    Figure 2: D_ell evolution
      - Row 0: TM1 (1.1 μm) across 3 catalogs (DESILS, HSC i<22, HSC i<25)
      - Row 1: TM2 (1.8 μm) across same 3 catalogs
      - Shared colorbar spanning all redshift bins
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.colors as mcolors
    from ciber.theory.cross_ps_parametric_model import CrossPowerSpectrumModel

    figdir = Path(args.figdir) / args.fitstr_cross
    figdir.mkdir(parents=True, exist_ok=True)

    lMax = args.lmax[-1] if args.lmax else args.lmax_compare
    fitstr_fixA2h = args.fitstr_cross + "_fixA2h_IGL"

    # ---- Load IHL 1h parameters ----
    ihl_path = Path(args.ihl_params)
    if not ihl_path.exists():
        print(f"[plot_ihl_and_dell_combined] IHL params file not found: {ihl_path}")
        return
    ihl_data = np.load(ihl_path, allow_pickle=True)
    params_dict_ihl = ihl_data["params_dict"].item()
    a1h_by_slope = ihl_data["a1h_by_slope"].item()
    a1h_err_by_slope = ihl_data["a1h_err_by_slope"].item()
    zbinedges_ihl = ihl_data["zbinedges"]
    z_centers_ihl = ihl_data["z_centers"]

    slope = 1.0
    n_zbins = len(z_centers_ihl)
    A_1h_II = np.array(a1h_by_slope[slope])
    A_1h_II_err = np.array(a1h_err_by_slope[slope])

    discrete_colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    lams = {1: 1.1, 2: 1.8}

    # ---- Catalog row definitions ----
    row_defs = [
        ("DESILS", None, r"DESI-LS ($z_{\rm AB}{<}22$)"),
        ("HSC", "hsc_ilt22.0", r"HSC ($18<i_{\rm AB}<22$)"),
        ("HSC", "hsc_ilt25.0", r"HSC ($18<i_{\rm AB}<25$)"),
    ]
    results_by_row = {}
    for cat, headstr, _ in row_defs:
        fpath = _cross_fpath(args.datadir_cross, cat, headstr, fitstr_fixA2h, lMax)
        results_by_row[(cat, headstr)] = (
            load_fit_results_npz(str(fpath)) if fpath.exists() else None
        )

    # ---- Figure 1: IHL auto-spectrum (square) ----
    fig1 = plt.figure(figsize=(6, 6))
    ax_ihl = fig1.add_subplot(111)
    ax_ihl.set_xscale("log")
    ax_ihl.set_yscale("log")

    ell_grid = np.logspace(np.log10(200), np.log10(1.1e5), 300)
    model = CrossPowerSpectrumModel(
        lb=ell_grid,
        use_powerlaw_2h=True,
        alpha_2h_fixed=0.0,
        use_astrometry_damping=False,
    )
    for z_idx in range(n_zbins):
        p = params_dict_ihl[(z_idx, slope)]
        A_II = A_1h_II[z_idx]
        dA = A_1h_II_err[z_idx]
        mu = p["mu_1h"]
        sig = p["sigma_1h"]
        c = discrete_colors[z_idx]
        zlbl = f"$z={zbinedges_ihl[z_idx]:.1f}$–${zbinedges_ihl[z_idx+1]:.1f}$"
        dl_best = model.lognormal_component(ell_grid, A_II, mu, sig)
        dl_lo = model.lognormal_component(ell_grid, max(A_II - dA, 0.0), mu, sig)
        dl_hi = model.lognormal_component(ell_grid, A_II + dA, mu, sig)
        ax_ihl.plot(ell_grid, dl_best, color=c, lw=4.0, label=zlbl, zorder=3)

    ax_ihl.set_xlim(280, 1.1e5)
    ax_ihl.set_xlabel(r"$\ell$", fontsize=18)
    ax_ihl.set_ylabel(r"$D_\ell^{II,{\rm 1h}}$ [(nW m$^{-2}$ sr$^{-1}$)$^2$]", fontsize=18)
    ax_ihl.set_title(r"One-halo templates (IHL auto)", fontsize=18, fontweight="bold")

    ax_ihl.legend(fontsize=14, loc="lower left")
    ax_ihl.tick_params(labelsize=14)
    ax_ihl.grid(alpha=0.3)
    
    stem_ihl = figdir / f"ihl_auto_spectrum_lMax={lMax}"
    _savefig(fig1, stem_ihl, args.fig_fmt)

    # ---- Figure 2: D_ell evolution 2×3 grid ----
    fig2 = plt.figure(figsize=(10.5, 6.5))

    fig2.suptitle('One-halo fits to cross-power spectra', fontsize=18, y=1.0, fontweight='bold')
    gs = gridspec.GridSpec(2, 4, figure=fig2, width_ratios=[2.0, 2.0, 2.0, 0.08],
                           hspace=0.05, wspace=0.05)

    # D_ell evolution panels: 2 rows (inst) × 3 cols (catalogs)
    idx_A2h, idx_A1h, idx_mu, idx_sig = 0, 1, 2, 3
    ax_grid = {}
    ref_ax = None  # Reference axis for sharing x and y axes

    for row_idx, inst in enumerate([1, 2]):  # row 0 = TM1, row 1 = TM2
        for col_idx, (cat, headstr, cat_label) in enumerate(row_defs):
            gs_col = col_idx

            # Share axes with the first panel in the grid
            if ref_ax is None:
                ax = fig2.add_subplot(gs[row_idx, gs_col])
                ref_ax = ax
            else:
                ax = fig2.add_subplot(gs[row_idx, gs_col], sharex=ref_ax, sharey=ref_ax)
            
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax_grid[(inst, cat, headstr)] = ax

            res = results_by_row[(cat, headstr)]

            if res is None:
                ax.text(
                    0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="gray", fontsize=10,
                )
                continue

            inst_list = list(res["inst_list"])
            if inst not in inst_list:
                ax.text(
                    0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="gray", fontsize=10,
                )
                continue

            inst_idx = inst_list.index(inst)
            zbinedges = res["zbinedges"]
            z_centers = 0.5 * (zbinedges[:-1] + zbinedges[1:])

            A_1h_lo_arr = res.get("params_16")
            A_1h_hi_arr = res.get("params_84")
            A_1h_95_arr = res.get("params_95")

            for zbin_idx in range(n_zbins):
                # Skip lowest z-bin (z<0.2) for HSC (unreliable)
                if zbin_idx == 0 and cat == "HSC":
                    continue
                
                color = discrete_colors[zbin_idx]

                # 2h component
                A_2h_z = float(res["params"][inst_idx, zbin_idx, idx_A2h])
                dl_2h = model.powerlaw_2h_component(ell_grid, A_2h_z, -0.0)
                ax.plot(ell_grid, dl_2h, color=color, lw=1.5, linestyle="--",
                        alpha=0.8, zorder=2)

                # 1h component
                A_1h = float(res["params"][inst_idx, zbin_idx, idx_A1h])
                mu_1h = float(res["params"][inst_idx, zbin_idx, idx_mu])
                sig1h = float(res["params"][inst_idx, zbin_idx, idx_sig])

                if A_1h_lo_arr is not None and A_1h_hi_arr is not None:
                    A_1h_lo = float(A_1h_lo_arr[inst_idx, zbin_idx, idx_A1h])
                    A_1h_hi = float(A_1h_hi_arr[inst_idx, zbin_idx, idx_A1h])
                else:
                    sigma_A1h = float(res["params_err"][inst_idx, zbin_idx, idx_A1h])
                    A_1h_lo = max(0.0, A_1h - sigma_A1h)
                    A_1h_hi = A_1h + sigma_A1h

                is_ul = A_1h_lo <= 0.0

                if not is_ul and A_1h > 0:
                    dl_best = model.lognormal_component(ell_grid, A_1h, mu_1h, sig1h)
                    dl_lo = model.lognormal_component(ell_grid, max(A_1h_lo, 0.0), mu_1h, sig1h)
                    dl_hi = model.lognormal_component(ell_grid, A_1h_hi, mu_1h, sig1h)
                    ax.plot(ell_grid, dl_best, color=color, lw=4, zorder=3)
                    # ax.fill_between(ell_grid, dl_lo, dl_hi, color=color, alpha=0.2, zorder=2)
                else:
                    if A_1h_95_arr is not None:
                        A_1h_ul = float(A_1h_95_arr[inst_idx, zbin_idx, idx_A1h])
                    else:
                        A_1h_ul = A_1h_hi
                    if A_1h_ul > 0:
                        dl_ul = model.lognormal_component(ell_grid, A_1h_ul, mu_1h, sig1h)
                        ax.plot(ell_grid, dl_ul, color=color, lw=1.2,
                                linestyle=":", alpha=0.7, zorder=2)

            ax.set_ylim(1e-3, 5)
            ax.set_xlim(280, 1.1e5)
            ax.grid(alpha=0.3)

            # Title: catalog name on top row
            if row_idx == 0:
                ax.set_title(cat_label, fontsize=14)
                # Hide x tick labels on top row
                ax.tick_params(labelbottom=False)
            else:
                # Show x labels on bottom row
                ax.set_xlabel(r"$\ell$", fontsize=14)

            # y-label only on leftmost column (col_idx == 0)
            if col_idx == 0:
                ax.set_ylabel(r"$D_\ell^{Ig}$ [nW m$^{-2}$ sr$^{-1}$]", fontsize=14)
            else:
                # Hide y tick labels on right columns
                ax.tick_params(labelleft=False)

            # Band label in top-left of each panel
            ax.text(
                0.05, 0.95, f"CIBER {lams[inst]} μm",
                transform=ax.transAxes, fontsize=14,
                verticalalignment="top", horizontalalignment="left",
            )

    # ---- Shared colorbar on far right ----
    cmap_disc = mcolors.ListedColormap(discrete_colors)
    norm_disc = mcolors.BoundaryNorm(zbinedges_ihl, n_zbins)
    sm = plt.cm.ScalarMappable(cmap=cmap_disc, norm=norm_disc)
    sm.set_array([])

    ax_cbar = fig2.add_subplot(gs[:, 3])  # spans both rows, rightmost column
    cbar = fig2.colorbar(sm, cax=ax_cbar, ticks=z_centers_ihl)
    cbar.set_label(r"Redshift bin", fontsize=12)
    cbar.set_ticklabels(
        [f"{zbinedges_ihl[i]:.1f}–{zbinedges_ihl[i+1]:.1f}" for i in range(n_zbins)],
        fontsize=12,
    )
    cbar.ax.tick_params(labelsize=12, length=2)

    stem_dell = figdir / f"dell_evolution_2x3_lMax={lMax}"
    _savefig(fig2, stem_dell, args.fig_fmt)
    plt.close(fig2)
    print(f"[plot_ihl_and_dell_combined] generated {stem_dell.with_suffix('.pdf')}")


def _plot_a1h_vs_redshift(args: argparse.Namespace) -> None:
    """Plot A_1h (one-halo amplitude) vs redshift for base variants and any b_I variants.

    Core variants: full model, fixA2h_IGL (b_I=constant).
    If b_I variant result files also exist (biLinear, biQuadratic) they are
    overlaid as additional series on the fixA2h panel.
    Creates one panel per band/catalog combination with shared axes.
    Includes horizontal redshift bin shading and staggered x positions for clarity.
    """
    figdir = Path(args.figdir) / args.fitstr_cross
    figdir.mkdir(parents=True, exist_ok=True)

    lMax = args.lmax[-1] if args.lmax else args.lmax_compare

    fitstr_fixA2h   = args.fitstr_cross + "_fixA2h_IGL"
    fitstr_biLinear = args.fitstr_cross + "_fixA2h_IGL_biLinear"
    fitstr_biQuad   = args.fitstr_cross + "_fixA2h_IGL_biQuadratic"

    # All variant keys and their fitstr names (no2h removed)
    variant_defs = [
        ("full",     args.fitstr_cross),
        ("fixA2h",   fitstr_fixA2h),
        ("biLinear", fitstr_biLinear),
        ("biQuad",   fitstr_biQuad),
    ]

    results_variants = {k: {} for k, _ in variant_defs}

    for cat in args.cat:
        headstr = args.headstr if cat == "HSC" else None
        for var_key, fitstr_name in variant_defs:
            results_variants[var_key][cat] = {}
            fpath = _cross_fpath(args.datadir_cross, cat, headstr, fitstr_name, lMax)
            if fpath.exists():
                results_variants[var_key][cat] = load_fit_results_npz(str(fpath))
            else:
                # biLinear / biQuad are optional — suppress warning unless core variant
                if var_key in ("full", "fixA2h"):
                    print(f"[plot_a1h_vs_redshift] missing {fpath.name}")

    # Build panel list from full-model results
    panel_info = []
    for cat in args.cat:
        res = results_variants["full"].get(cat)
        if not res:
            continue
        for inst in list(res["inst_list"]):
            panel_info.append((cat, inst))

    if not panel_info:
        print("[plot_a1h_vs_redshift] no results found, skipping")
        return

    n_panels = len(panel_info)
    n_cols = 2
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6, 5), sharex=True, sharey=True)
    if n_panels == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)

    # Styling per variant (linestyle removed so no lines connect points)
    var_styles = {
        "full":     dict(color="C0", marker="o", label="Full model (Float $A_{2h}$, $A_{1h}$)"),
        "fixA2h":   dict(color="C2", marker="^", label=r"Fix $A_{2h}$ (Standard IGL, $b_I=1$)"),
        "biLinear": dict(color="C3", marker="D", label=r"Fix $A_{2h}$ ($b_I=1+0.6z$)"),
        "biQuad":   dict(color="C4", marker="v", label=r"Fix $A_{2h}$ ($b_I=(1+z)^2$)"),
    }

    cat_display = {"DESILS": "DESI-LS", "HSC": "HSC"}
    lams = {1: 1.1, 2: 1.8}
    shade_colors = ("#f5f5f5", "#eeeeee")
    x_offset_scale = 0.04  # Stagger variants horizontally

    for panel_idx, (cat, inst) in enumerate(panel_info):
        row = panel_idx // n_cols
        col = panel_idx % n_cols
        ax = axes[row, col]

        res_full = results_variants["full"].get(cat)
        if not res_full:
            continue
        zbinedges = res_full["zbinedges"]
        z_centers = 0.5 * (zbinedges[:-1] + zbinedges[1:])

        # Add redshift bin shading
        for j in range(len(zbinedges) - 1):
            z0, z1 = zbinedges[j], zbinedges[j + 1]
            shade = shade_colors[j % 2]
            ax.axvspan(z0, z1, color=shade, alpha=0.22, zorder=0)

        # Plot each variant with horizontal offset
        for var_idx, (var_key, _) in enumerate(variant_defs):
            res = results_variants[var_key].get(cat)
            if not res:
                continue
            inst_list_var = list(res["inst_list"])
            if inst not in inst_list_var:
                continue

            inst_idx_var = inst_list_var.index(inst)
            n_params = res["params"].shape[-1]

            # Find A_1h index using param_names_fitted (use zidx=0 as representative)
            pnames_var = res.get("param_names_fitted")
            a1h_idx = 1  # fallback: A_1h is typically at index 1
            if pnames_var is not None:
                try:
                    rep_names = pnames_var[inst_idx_var, 0]
                    if rep_names is not None:
                        for k, nm in enumerate(rep_names):
                            if "A_1h" in str(nm) or "A_{1h}" in str(nm):
                                a1h_idx = k
                                break
                except (IndexError, TypeError):
                    pass
            if n_params <= a1h_idx:
                continue

            A_1h     = res["params"][inst_idx_var, :, a1h_idx]
            A_1h_lo  = res.get("params_16")
            A_1h_hi  = res.get("params_84")
            A_1h_95  = res.get("params_95")
            
            if A_1h_lo is not None and A_1h_hi is not None:
                yerr_lo = A_1h - A_1h_lo[inst_idx_var, :, a1h_idx]
                yerr_hi = A_1h_hi[inst_idx_var, :, a1h_idx] - A_1h
                yerr = np.array([yerr_lo, yerr_hi])
            else:
                yerr_lo = res["params_err"][inst_idx_var, :, a1h_idx]
                yerr = np.array([yerr_lo, yerr_lo])

            # Identify upper limits: where A_1h - 2*sigma <= 0 (following _plot_param logic)
            is_ul = (A_1h - 2 * yerr[0]) <= 0
            is_det = ~is_ul

            # Apply horizontal offset to distinguish variants
            x_offset = (var_idx - (len(variant_defs) - 1) / 2.0) * x_offset_scale
            z_offset = z_centers + x_offset

            st = var_styles[var_key]
            label = st["label"]
            
            # Plot detections
            if np.any(is_det):
                ax.errorbar(z_offset[is_det], A_1h[is_det], 
                           yerr=np.array([yerr[0][is_det], yerr[1][is_det]]),
                           marker=st["marker"], color=st["color"],
                           label=label, linestyle='None',
                           markersize=5, capsize=3)
                label = None  # don't repeat for upper limits
            
            # Plot upper limits as downward arrows
            if np.any(is_ul):
                ul_vals = A_1h_95[inst_idx_var, :, a1h_idx][is_ul] if A_1h_95 is not None else A_1h[is_ul] + 2 * yerr[1][is_ul]
                xs_ul = z_offset[is_ul]
                # Plot marker at upper limit value
                ax.plot(xs_ul, ul_vals, marker="v", color=st["color"],
                       label=label, markersize=5, alpha=0.85, linestyle='none')
                # Draw arrows from ul_val down to y=0
                for x, y_top in zip(xs_ul, ul_vals):
                    ax.annotate('', xy=(x, 0.0), xytext=(x, y_top),
                               arrowprops=dict(arrowstyle='-|>', color=st["color"], 
                                            alpha=0.85, lw=1.2))

        # Panel title in top-left corner
        lam_str = f"TM{inst} ({lams.get(inst, '?'):.1f} μm)"
        title_text = 'CIBER '+str(lams.get(inst, '?'))+' $\\mu$m $\\times$ '+cat_display.get(cat, cat)
        # title_text = f"{cat_display.get(cat, cat)} × CIBER {lam_str}"
        ax.text(0.02, 0.95, title_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top')

        ax.grid(True, alpha=0.3)
        ax.set_xlim(zbinedges[0], zbinedges[-1])
        ax.set_ylim(0, 1.0)
        # ax.set_ylim(1e-2, 0.5)
        # ax.set_yscale('log')

    # Delete unused subplots
    for idx in range(n_panels, len(axes.flat)):
        fig.delaxes(axes.flat[idx])

    # Add shared legend above top row
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.53, 1.08),
               ncol=2, fontsize=12, frameon=True)

    # Set axis labels on outer edges
    for row in range(n_rows):
        axes[row, 0].set_ylabel(r"$A_{1h}$", fontsize=14)
    for col in range(n_cols):
        axes[n_rows - 1, col].set_xlabel("Redshift", fontsize=14)

    # Adjust layout manually (no tight_layout)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.12, hspace=0.1, wspace=0.1)

    # Include HSC headstr tag in filename so different magnitude selections don't overwrite each other
    hsc_tag = _headstr_tag(args.headstr) if "HSC" in args.cat else ""
    stem = figdir / f"a1h_vs_redshift{hsc_tag}_lMax={lMax}"
    _savefig(fig, stem, args.fig_fmt)
    plt.close(fig)


def _plot_a1h_vs_redshift_mag_comparison(args: argparse.Namespace) -> None:
    """Plot A_1h (one-halo amplitude) vs redshift comparing different magnitude selections.

    Compares multiple HSC magnitude limits (e.g., i<22, i<25) on the same axes.
    One panel per band with shared axes. Uses the full model variant.
    """
    figdir = Path(args.figdir) / args.fitstr_cross / "magnitude_comparisons"
    figdir.mkdir(parents=True, exist_ok=True)

    lMax = args.lmax[-1] if args.lmax else args.lmax_compare

    # Define magnitude selections to compare
    mag_selections = [
        ("hsc_ilt22.0", "i < 22", "C0"),
        ("hsc_ilt25.0", "i < 25 (main)", "C1"),
    ]

    # Load results for each magnitude selection
    results_mag = {}
    for headstr, label, color in mag_selections:
        results_mag[headstr] = {"label": label, "color": color, "data": {}}
        fpath = _cross_fpath(args.datadir_cross, "HSC", headstr, args.fitstr_cross, lMax)
        if fpath.exists():
            results_mag[headstr]["data"] = load_fit_results_npz(str(fpath))
        else:
            print(f"[plot_a1h_vs_redshift_mag_comparison] missing {fpath.name}")

    # Build panel list from first available result
    first_res = None
    for headstr, info in results_mag.items():
        if info["data"]:
            first_res = info["data"]
            break

    if not first_res:
        print("[plot_a1h_vs_redshift_mag_comparison] no results found, skipping")
        return

    inst_list = list(first_res["inst_list"])
    n_panels = len(inst_list)
    n_cols = 2
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7, 5), sharex=True, sharey=True)
    if n_panels == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)

    cat_display = {"HSC": "HSC"}
    lams = {1: 1.1, 2: 1.8}
    shade_colors = ("#f5f5f5", "#eeeeee")
    x_offset_scale = 0.05  # Stagger magnitude selections horizontally

    zbinedges = first_res["zbinedges"]
    z_centers = 0.5 * (zbinedges[:-1] + zbinedges[1:])

    for panel_idx, inst in enumerate(inst_list):
        row = panel_idx // n_cols
        col = panel_idx % n_cols
        ax = axes[row, col]

        # Add redshift bin shading
        for j in range(len(zbinedges) - 1):
            z0, z1 = zbinedges[j], zbinedges[j + 1]
            shade = shade_colors[j % 2]
            ax.axvspan(z0, z1, color=shade, alpha=0.22, zorder=0)

        # Plot each magnitude selection with horizontal offset
        for mag_idx, (headstr, info) in enumerate(results_mag.items()):
            res = info["data"]
            if not res:
                continue

            inst_list_res = list(res["inst_list"])
            if inst not in inst_list_res:
                continue

            inst_idx = inst_list_res.index(inst)
            n_params = res["params"].shape[-1]

            # Find A_1h index
            pnames = res.get("param_names_fitted")
            a1h_idx = 1  # fallback
            if pnames is not None:
                try:
                    rep_names = pnames[inst_idx, 0]
                    if rep_names is not None:
                        for k, nm in enumerate(rep_names):
                            if "A_1h" in str(nm) or "A_{1h}" in str(nm):
                                a1h_idx = k
                                break
                except (IndexError, TypeError):
                    pass
            if n_params <= a1h_idx:
                continue

            A_1h     = res["params"][inst_idx, :, a1h_idx]
            A_1h_lo  = res.get("params_16")
            A_1h_hi  = res.get("params_84")
            A_1h_95  = res.get("params_95")

            if A_1h_lo is not None and A_1h_hi is not None:
                yerr_lo = A_1h - A_1h_lo[inst_idx, :, a1h_idx]
                yerr_hi = A_1h_hi[inst_idx, :, a1h_idx] - A_1h
                yerr = np.array([yerr_lo, yerr_hi])
            else:
                yerr_lo = res["params_err"][inst_idx, :, a1h_idx]
                yerr = np.array([yerr_lo, yerr_lo])

            # Identify upper limits
            is_ul = (A_1h - 2 * yerr[0]) <= 0
            is_det = ~is_ul

            # Apply horizontal offset to distinguish magnitude selections
            x_offset = (mag_idx - (len(results_mag) - 1) / 2.0) * x_offset_scale
            z_offset = z_centers + x_offset

            label = info["label"]
            color = info["color"]

            # Plot detections
            if np.any(is_det):
                ax.errorbar(z_offset[is_det], A_1h[is_det],
                           yerr=np.array([yerr[0][is_det], yerr[1][is_det]]),
                           marker="o", color=color, label=label, linestyle='None',
                           markersize=6, capsize=3, linewidth=1.5, alpha=0.85)

            # Plot upper limits as downward arrows
            if np.any(is_ul):
                ul_vals = A_1h_95[inst_idx, :, a1h_idx][is_ul] if A_1h_95 is not None else A_1h[is_ul] + 2 * yerr[1][is_ul]
                xs_ul = z_offset[is_ul]
                ax.plot(xs_ul, ul_vals, marker="v", color=color,
                       markersize=5, alpha=0.85, linestyle='none')
                for x, y_top in zip(xs_ul, ul_vals):
                    ax.annotate('', xy=(x, 0.0), xytext=(x, y_top),
                               arrowprops=dict(arrowstyle='-|>', color=color,
                                            alpha=0.85, lw=1.2))

        # Panel title
        title_text = f'CIBER {lams.get(inst, "?")} μm × HSC'
        ax.text(0.02, 0.95, title_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top', fontweight='bold')

        ax.grid(True, alpha=0.3)
        ax.set_xlim(zbinedges[0], zbinedges[-1])
        ax.set_ylim(0, 1.0)

    # Delete unused subplots
    for idx in range(n_panels, len(axes.flat)):
        fig.delaxes(axes.flat[idx])

    # Add shared legend above top row
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08),
                   ncol=2, fontsize=12, frameon=True)

    # Set axis labels
    for row in range(n_rows):
        axes[row, 0].set_ylabel(r"$A_{1h}$", fontsize=13)
    for col in range(n_cols):
        axes[n_rows - 1, col].set_xlabel("Redshift (z)", fontsize=13)

    fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.12, hspace=0.1, wspace=0.1)

    stem = figdir / f"a1h_vs_redshift_mag_comparison_lMax={lMax}"
    _savefig(fig, stem, args.fig_fmt)
    plt.close(fig)
    print(f"[plot_a1h_vs_redshift_mag_comparison] generated {stem.with_suffix('.pdf')}")

def _plot_a2h_vs_redshift(args: argparse.Namespace) -> None:
    figdir = Path(args.figdir) / args.fitstr_cross
    figdir.mkdir(parents=True, exist_ok=True)

    lMax = args.lmax[-1] if args.lmax else args.lmax_compare

    # Load fit results
    results_full = {}
    for cat in args.cat:
        headstr = args.headstr if cat == "HSC" else None
        fpath = _cross_fpath(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax, maskstr=args.maskstr)
        if fpath.exists():
            results_full[cat] = load_fit_results_npz(str(fpath))
        else:
            print(f"[plot_a2h_vs_redshift] missing {fpath.name}")

    if not any(c in results_full for c in args.cat):
        print("[plot_a2h_vs_redshift] no results found, skipping")
        return

    # --- 2-panel layout (like a1h): one panel per band ---
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, sharey=True)
    inst_order = [1, 2]

    plot_alpha = 0.8

    multfac = 10

    dz = 0.2

    # ymin = 3e-3 * multfac

    ymin = 1.0
    shade_colors = ("#f5f5f5", "#eeeeee")
    lams = {1: 1.1, 2: 1.8}
    color_map = {"DESILS": "C2", "HSC": "#E45DA8"}
    cat_display = {"DESILS": r"DESI-LS ($z_{\rm AB}<22$)", "HSC": r"HSC ($18<i_{\rm AB}<25$)"}

    model_styles = {
        "constant": dict(color="k", linestyle="-",  label=r"IGL 2h prediction; $b_I=1$"),
        "linear":   dict(color="k", linestyle="--", label=r"$b_I=1+0.6z$"),
        "quadratic":dict(color="k", linestyle=":",  label=r"$b_I=(1+z)^2$"),
    }

    ls_didz = np.load('data/jordan_mocks/mock_dIdz_LS_zbins_unmasked_dz=0.2_091625.npz')
    zedges, all_mock_dI_dz = ls_didz['zedges'], ls_didz['all_mock_dI_dz']
    zcenter = 0.5 * (zedges[:-1] + zedges[1:])

    all_galautodivs = []

    # Precompute model predictions
    model_preds = {m: {} for m in ("constant", "linear", "quadratic")}
    for bi_model in model_preds:
        for cat in args.cat:
            res = results_full.get(cat)
            if res is None:
                continue
            preds, galautodiv = _compute_igl_a2h_predictions(
                cat,
                res["zbinedges"],
                list(res["inst_list"]),
                jmock_basedir=args.igl_pred_basedir,
                bias_cache_fpath=args.bias_cache_fpath,
                headstr=None,
                bi_model=bi_model,
                a2h_cache_fpath=getattr(args, "mock_a2h_cache", None),
            )

            print('gal auto div is ', galautodiv)
            all_galautodivs.append(galautodiv*dz)


            if bi_model == "constant":
                # For constant b_I=1, the predictions are just the mock dI/dz values
                model_preds[bi_model][cat] = np.median(all_mock_dI_dz, axis=2)
            elif bi_model == "linear":
                model_preds[bi_model][cat] = np.median(all_mock_dI_dz, axis=2) * (1 + zcenter[:,None] * 0.6)
            elif bi_model == "quadratic":
                model_preds[bi_model][cat] = np.median(all_mock_dI_dz, axis=2) * (1 + zcenter[:,None]) ** 2

            print('model preds for bias model', bi_model, 'is', model_preds[bi_model][cat])
            # print(preds.shape, galautodiv.shape)
            # model_preds[bi_model][cat] = preds

    # Mask HSC z<0.2 predictions
    for bi_model in model_preds:
        if model_preds[bi_model].get("HSC") is not None:
            model_preds[bi_model]["HSC"][:,0] = np.nan

    for band_idx, inst in enumerate(inst_order):
        ax = axes[band_idx]
        ax.set_yscale("log")

        drew_bin_shading = False
        xlim_set = (0., 1.0)

        for tracer_idx, cat in enumerate(args.cat):
            res = results_full.get(cat)
            if res is None:
                continue

            inst_list = list(res["inst_list"])
            if inst not in inst_list:
                continue
            inst_idx = inst_list.index(inst)

            zbinedges = res["zbinedges"]
            z_centers = 0.5 * (zbinedges[:-1] + zbinedges[1:])
            dz_shift = 0.04 * (tracer_idx - (len(args.cat) - 1) / 2.0)
            z_plot = z_centers + dz_shift
            color_plot = color_map.get(cat, "k")

            # Data vectors
            A_2h = np.array(res["params"][inst_idx, :, 0], dtype=float)
            A_2h_16 = res.get("params_16")
            A_2h_84 = res.get("params_84")
            A_2h_95 = res.get("params_95")

            if A_2h_16 is not None and A_2h_84 is not None:
                yerr_lo = A_2h - A_2h_16[inst_idx, :, 0]
                yerr_hi = A_2h_84[inst_idx, :, 0] - A_2h
            else:
                yerr_lo = np.array(res["params_err"][inst_idx, :, 0], dtype=float)
                yerr_hi = yerr_lo.copy()

            # Exclude HSC first bin
            if cat == "HSC":
                A_2h[0] = np.nan
                yerr_lo[0] = np.nan
                yerr_hi[0] = np.nan
                if A_2h_95 is not None:
                    A_2h_95[inst_idx, 0, 0] = np.nan

            is_valid = np.isfinite(A_2h) & np.isfinite(yerr_lo) & np.isfinite(yerr_hi)
            is_ul = is_valid & ((A_2h - 2.0 * yerr_lo) <= 0.0)
            is_det = is_valid & ~is_ul

            # Detections
            if np.any(is_det):
                ax.errorbar(
                    z_plot[is_det], A_2h[is_det] / all_galautodivs[tracer_idx][inst_idx, is_det],
                    yerr=np.array([yerr_lo[is_det] / all_galautodivs[tracer_idx][inst_idx, is_det],
                                    yerr_hi[is_det] / all_galautodivs[tracer_idx][inst_idx, is_det]]),
                    fmt="o", color=color_plot, markerfacecolor=color_plot, markeredgecolor=color_plot,
                    linestyle="None", markersize=7, capsize=6, capthick=2,
                    label=f"This work" if band_idx == 0 and tracer_idx == 0 else None,
                )

            # Upper limits
            if np.any(is_ul):
                ul_vals = A_2h_95[inst_idx, :, 0][is_ul] / all_galautodivs[tracer_idx][inst_idx, is_ul] if A_2h_95 is not None else (A_2h[is_ul] + 2.0 * yerr_hi[is_ul]) / all_galautodivs[tracer_idx][inst_idx, is_ul]
                xs_ul = z_plot[is_ul]

                ax.plot(xs_ul, ul_vals, marker="_", color=color_plot, markersize=12,
                        markeredgewidth=2, linestyle="none", alpha=0.85)

                for x, y_top in zip(xs_ul, ul_vals):
                    if np.isfinite(y_top) and y_top > ymin:
                        ax.annotate(
                            "", xy=(x, ymin), xytext=(x, y_top),
                            arrowprops=dict(arrowstyle="-|>", color=color_plot, alpha=0.85, lw=2.5, mutation_scale=15),
                        )

            # Model curves
            if tracer_idx == 0:
                for bi_model in ("constant", "linear", "quadratic"):
                    pred_arr = model_preds[bi_model].get(cat)

                    if pred_arr is None:
                        continue
                    # preds = pred_arr[inst_idx, :]
                    preds = pred_arr[:, inst_idx]

                    print(f"[plot_a2h_vs_redshift] plotting model {bi_model} for {cat} band {inst}: preds={preds}")
                    st = model_styles[bi_model]
                    ax.plot(
                        z_centers, preds,
                        color='grey', linestyle=st["linestyle"], linewidth=2.0, alpha=plot_alpha,
                        marker=None,
                        label=st["label"] if (band_idx == 0 and tracer_idx == 0) else None,
                    )

        # Panel text
        for i, cat_txt in enumerate(["DESILS", "HSC"]):
            ax.text(
                0.4, 0.95 - 0.12 * i,
                fr"CIBER {lams[inst]:.1f} $\mu$m $\times$ {cat_display[cat_txt]}",
                transform=ax.transAxes, color=color_map[cat_txt], fontsize=14, va="top", ha="left"
            )

        ax.grid(True, alpha=0.3)
        if xlim_set is not None:
            ax.set_xlim(*xlim_set)
        ax.set_ylim(ymin, 300.0)
        ax.tick_params(labelsize=13)

    # axes[0].set_ylabel(r"$A_{\rm 2h}^{\rm Ig}=b_g \frac{dN}{dz} b_I \frac{dI}{dz}$", fontsize=16)
    # axes[1].set_ylabel(r"$A_{\rm 2h}^{\rm Ig}=b_g \frac{dN}{dz} b_I \frac{dI}{dz}$", fontsize=16)

    axes[0].set_ylabel(r"$b_I \times dI/dz$", fontsize=18)
    axes[1].set_ylabel(r"$b_I \times dI/dz$", fontsize=18)

    ax_twin = axes[0].twinx()
    ax_twin.set_yticks([])
    ax_twin.set_ylabel('[nW m$^{-2}$ sr$^{-1}$]', fontsize=16)
    ax_twin = axes[1].twinx()
    ax_twin.set_yticks([])
    ax_twin.set_ylabel('[nW m$^{-2}$ sr$^{-1}$]', fontsize=16)

    axes[1].set_xlabel("Redshift (z)", fontsize=14)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 1.02), ncol=2, fontsize=14, frameon=True)

    fig.subplots_adjust(wspace=0.02, hspace=0.02)

    stem = figdir / f"a2h_vs_redshift_lMax={lMax}"
    _savefig(fig, stem, args.fig_fmt)
    plt.close(fig)

def _plot_di_dz_upper_limits(args: argparse.Namespace) -> None:
    """Plot dI/dz upper limits derived from A_2h vs redshift.
    
    Converts A_2h upper limits to dI/dz constraints by dividing by model-predicted
    b_g*dN/dz and b_I(z) bias terms, accounting for redshift bin width.
    Shows three b_I variants per z-bin with horizontal offsets.
    """
    figdir = Path(args.figdir) / args.fitstr_cross
    figdir.mkdir(parents=True, exist_ok=True)

    lMax = args.lmax[-1] if args.lmax else args.lmax_compare

    # Load full model results to get A_2h upper limits
    results_full = {}
    for cat in args.cat:
        headstr = args.headstr if cat == "HSC" else None
        fpath = _cross_fpath(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax)
        if fpath.exists():
            results_full[cat] = load_fit_results_npz(str(fpath))
        else:
            print(f"[plot_di_dz_upper_limits] missing {fpath.name}")

    # Build panel list from full-model results
    panel_info = []
    for cat in args.cat:
        res = results_full.get(cat)
        if not res:
            continue
        for inst in list(res["inst_list"]):
            panel_info.append((cat, inst))

    if not panel_info:
        print("[plot_di_dz_upper_limits] no results found, skipping")
        return

    n_panels = len(panel_info)
    n_cols = 2
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7, 6), sharex=True, sharey=True)
    if n_panels == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)

    # Styling for b_I variants
    variant_styles = {
        "constant": dict(marker="o", color="C2", label=r"$b_I=1$"),
        "linear":   dict(marker="^", color="C3", label=r"$b_I=1+0.6z$"),
        "quadratic": dict(marker="D", color="C4", label=r"$b_I=(1+z)^2$"),
    }

    cat_display = {"DESILS": "DESI-LS", "HSC": "HSC"}
    lams = {1: 1.1, 2: 1.8}
    shade_colors = ("#f5f5f5", "#eeeeee")
    x_offset_scale = 0.04
    ymin = 0.1
    ymax = 100.0

    # Compute model predictions for b_g*dN/dz (using constant b_I to get pure bias term)
    model_preds = {}
    for cat in args.cat:
        res = results_full.get(cat)
        if not res:
            continue
        zbinedges = res["zbinedges"]
        inst_list = list(res["inst_list"])
        preds = _compute_igl_a2h_predictions(
            cat, zbinedges, inst_list,
            jmock_basedir=args.igl_pred_basedir,
            bias_cache_fpath=args.bias_cache_fpath,
            headstr=None,
            bi_model="constant",
            a2h_cache_fpath=getattr(args, "mock_a2h_cache", None)
        )
        model_preds[cat] = preds

    for panel_idx, (cat, inst) in enumerate(panel_info):
        row = panel_idx // n_cols
        col = panel_idx % n_cols
        ax = axes[row, col]

        res = results_full.get(cat)
        if not res:
            continue

        zbinedges = res["zbinedges"]
        z_centers = 0.5 * (zbinedges[:-1] + zbinedges[1:])
        inst_list = list(res["inst_list"])
        inst_idx = inst_list.index(inst)

        # Add redshift bin shading
        for j in range(len(zbinedges) - 1):
            z0, z1 = zbinedges[j], zbinedges[j + 1]
            shade = shade_colors[j % 2]
            ax.axvspan(z0, z1, color=shade, alpha=0.22, zorder=0)

        # Add grey overlay shading for HSC z<0.2 bin to indicate omitted measurements
        if cat == "HSC":
            z0_omit, z1_omit = zbinedges[0], zbinedges[1]
            ax.axvspan(z0_omit, z1_omit, color="#a9a9a9", alpha=0.25, zorder=1, linewidth=2, edgecolor="grey")

        # Extract A_2h_95 (upper limits)
        A_2h_95 = res.get("params_95")
        if A_2h_95 is None:
            print(f"[plot_di_dz_upper_limits] No params_95 found for {cat}, skipping")
            continue

        A_2h_ul = A_2h_95[inst_idx, :, 0].copy()
        
        # Exclude HSC z<0.2 (unreliable)
        if cat == "HSC":
            A_2h_ul[0] = np.nan

        # Get model predictions for this instrument
        bg_dndz_pred = model_preds[cat]
        if bg_dndz_pred is None:
            print(f"[plot_di_dz_upper_limits] No model predictions for {cat}, skipping")
            continue
        bg_dndz_pred = bg_dndz_pred[inst_idx, :].copy()

        # Convert A_2h upper limits to dI/dz upper limits
        di_dz_dict = _compute_di_dz_upper_limits(
            A_2h_ul[np.newaxis, :], 
            bg_dndz_pred[np.newaxis, :],
            z_centers,
            zbinedges
        )

        # Plot each b_I variant with horizontal offset
        for var_idx, (bi_model, style) in enumerate(variant_styles.items()):
            di_dz_ul = di_dz_dict[bi_model][0, :]  # extract for this instrument
            
            # Apply horizontal offset to distinguish variants
            x_offset = (var_idx - 1.0) * x_offset_scale
            z_offset = z_centers + x_offset

            # Plot upper limit arrows (all points are upper limits)
            ul_mask = ~np.isnan(di_dz_ul)
            if np.any(ul_mask):
                # Plot horizontal bar at top of UL (wider capsize)
                ax.plot(z_offset[ul_mask], di_dz_ul[ul_mask], marker=style["marker"], 
                       color=style["color"], linestyle='none', markersize=5, 
                       label=style["label"] if panel_idx == 0 else None, alpha=1.0)
                
                # Draw downward arrow from UL position to ymin
                for x, y_top in zip(z_offset[ul_mask], di_dz_ul[ul_mask]):
                    ax.annotate('', xy=(x, ymin), xytext=(x, y_top),
                               arrowprops=dict(arrowstyle='-|>', color=style["color"],
                                            alpha=0.85, lw=1.5, mutation_scale=12))

        # Panel title in top-left corner
        title_text = 'CIBER '+str(lams.get(inst, '?'))+' $\\mu$m $\\times$ '+cat_display.get(cat, cat)
        ax.text(0.4, 0.95, title_text, transform=ax.transAxes,
                fontsize=15, verticalalignment='top')

        ax.grid(True, alpha=0.3)
        ax.set_xlim(zbinedges[0], zbinedges[-1])
        ax.set_ylim(ymin, ymax)
        ax.set_yscale('log')
        ax.tick_params(labelsize=13)

    # Delete unused subplots
    for idx in range(n_panels, len(axes.flat)):
        fig.delaxes(axes.flat[idx])

    # Add shared legend above top row
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08),
               ncol=3, fontsize=14, frameon=True)

    # Set axis labels on outer edges
    for row in range(n_rows):
        axes[row, 0].set_ylabel(r"$dI/dz$ [nW m$^{-2}$ sr$^{-1}$]", fontsize=16)
    for col in range(n_cols):
        axes[n_rows - 1, col].set_xlabel("Redshift", fontsize=16)

    # Adjust layout manually
    fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.12, hspace=0.1, wspace=0.1)

    stem = figdir / f"di_dz_upper_limits_lMax={lMax}"
    _savefig(fig, stem, args.fig_fmt)
    plt.close(fig)


def _plot_corr_a1h_a2h(args: argparse.Namespace) -> None:
    """Plot r(A_2h, A_1h) correlation coefficient vs redshift.

    Shows one curve per catalog/instrument combination at fixed lMax.
    """
    figdir = Path(args.figdir) / args.fitstr_cross

    # Use lmax from --lmax list if available, otherwise use --lmax-compare
    lMax = args.lmax[-1] if args.lmax else args.lmax_compare
    all_data = []

    for cat in args.cat:
        headstr = args.headstr if cat == "HSC" else None
        results = _load_cross_results_merged_jh14(
            args.datadir_cross, cat, headstr, args.fitstr_cross, lMax, maskstr=args.maskstr
        )
        if results is None:
            fpath = _cross_fpath(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax, maskstr=args.maskstr)
            print(f"[plot_corr_a1h_a2h] missing {fpath}, skipping {cat}")
            continue

        zbinedges = results["zbinedges"]
        n_zbins = len(zbinedges) - 1
        inst_list = list(results["inst_list"])
        samples_array = results.get("samples", None)

        if samples_array is None:
            print(f"[plot_corr_a1h_a2h] no MCMC samples in {fpath.name}, skipping {cat}")
            continue

        for inst_idx, inst in enumerate(inst_list):
            for zidx in range(n_zbins):
                zlo, zhi = zbinedges[zidx], zbinedges[zidx + 1]
                samples = samples_array[inst_idx, zidx]

                if samples is None or len(samples) == 0:
                    corr = np.nan
                else:
                    a2h_samples = samples[:, 0]
                    a1h_samples = samples[:, 1]
                    shot_samples = samples[:, 2]
                    corr_matrix = np.corrcoef(a2h_samples, a1h_samples)
                    corr = corr_matrix[0, 1]

                    corr_matrix_shot_1h = np.corrcoef(a1h_samples, shot_samples)
                    corr_shot_1h = corr_matrix_shot_1h[0, 1]

                all_data.append({
                    'cat': cat,
                    'inst': inst,
                    'z_lo': zlo,
                    'z_hi': zhi,
                    'corr': corr, 
                    'corr_shot_1h': corr_shot_1h,
                })

    if not all_data:
        print("[plot_corr_a1h_a2h] no data found for any catalog, skipping")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(4, 5), nrows=2, sharex=True)

    colors = {'HSC': {'TM1': 'C0', 'TM2': 'C1'}, 'DESILS': {'TM1': 'C2', 'TM2': 'C3'}}
    linestyles = {'HSC': '-', 'DESILS': '--'}

    hsc_color = "#E45DA8"
    colors = ['C2', hsc_color]

    cat_labels = ['DESI-LS', 'HSC']

    for c, cat in enumerate(args.cat):
        if cat not in [d['cat'] for d in all_data]:
            continue
        for inst in [1, 2]:
            # Extract z-midpoints and correlations for this cat/inst combination
            z_mids = []
            corrs = []
            corrs_shot_1h = []
            for row in all_data:
                if row['cat'] == cat and row['inst'] == inst:
                    z_mids.append(0.5 * (row['z_lo'] + row['z_hi']))
                    corrs.append(row['corr'])
                    corrs_shot_1h.append(row['corr_shot_1h'])

            if corrs:
                label = f"{cat} TM{inst}"
                label =  f"$\\rho(A_{{2h}}, A_{{1h}})$; " +cat_labels[c]
                ax[inst-1].plot(z_mids, corrs, 'o', color=colors[c],
                        linestyle='solid', linewidth=2, markersize=6, label=label)

                if c==1:
                    shot1hlab = '$\\rho(A_{1h}, A_{shot})$'
                else:
                    shot1hlab = None
                
                ax[inst-1].plot(z_mids, corrs_shot_1h, color=colors[c],
                        linestyle='dashed', linewidth=2, label=shot1hlab)

    # for inst in [1, 2]:
        # ax[inst-1].set_title(f'TM{inst}', fontsize=12)

    lams_ciber = [1.1, 1.8]
    for inst in [1, 2]:
        ax[inst-1].axhline(0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

        if inst==2:
            ax[inst-1].set_xlabel('redshift', fontsize=12)
        ax[inst-1].set_ylabel(r'Correlation coeff. $\rho$', fontsize=11)
        ax[inst-1].grid(True, alpha=0.3, which='major')

        if inst==1:
            ax[inst-1].legend(loc='upper center', ncol=2, fontsize=11, bbox_to_anchor=[0.45, 1.4])
        ax[inst-1].set_ylim([-1.0, 0.5])
        ax[inst-1].set_xlim([0.0, 1.0])
        ax[inst-1].text(0.05, 0.25, f'CIBER {lams_ciber[inst-1]} μm', fontsize=14)

    # fig.tight_layout()
    _savefig(fig, figdir / f"corr_a1h_a2h_{args.fitstr_cross}_lMax={lMax}", args.fig_fmt)
    plt.close(fig)


def _plot_parameter_consistency_vs_lmax(args: argparse.Namespace) -> None:
    """3×2 panel figure showing A_2h, A_1h, and reduced χ² consistency across lMax values.

    Rows: A_2h (top), A_1h (middle), reduced χ² (bottom)
    Columns: TM1 (1.1 μm), TM2 (1.8 μm)
    
    Each lMax appears as a separate trace (color) with small x-offsets to avoid overlap.
    Shows parameter consistency and goodness-of-fit across different multipole cuts.
    """
    figdir = Path(args.figdir) / args.fitstr_cross
    figdir.mkdir(parents=True, exist_ok=True)

    # Collect results for all lMax values across catalogs
    all_configs = []


    for cat in args.cat:
        all_configs = []

        for lMax in args.lmax:
            headstr = args.headstr if cat == "HSC" else None
            results = _load_cross_results_merged_jh14(
                args.datadir_cross, cat, headstr, args.fitstr_cross, lMax, maskstr=args.maskstr
            )
            if results is None:
                fpath = _cross_fpath(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax, maskstr=args.maskstr)
                print(f"[plot_parameter_consistency_vs_lmax] missing {fpath}")
                continue
            all_configs.append({
                'results': results,
                'inst': None,  # Both instruments
                'label': f"{cat} ℓ_max={lMax}",
                'cat_name': cat,
                'lMax': lMax,
            })

        if not all_configs:
            print("[plot_parameter_consistency_vs_lmax] no results found, skipping")
            return


        if cat=='DESILS':
            cmap_name = 'Greens'
        
        elif cat=='HSC':
            cmap_name = 'RdPu'

        # Generate figure
        stem = figdir / f"parameter_consistency_vs_lmax_{args.fitstr_cross}_{cat}"
        plot_amplitude_chi2_by_instrument(
            all_configs,
            inst_list=(1, 2),
            figsize=(8.5, 7),
            save_path=str(stem.with_suffix(f'.{args.fig_fmt}')),
            legend_ncol=2,
            bbox_to_anchor=(0.5, 1.12),
            use_cmap=True,
            cmap_name=cmap_name,
            x_offset_scale=0.03,
        )
        print(f"[plot_parameter_consistency_vs_lmax] generated {stem.with_suffix(f'.{args.fig_fmt}')}")


def _plot_sigma_damp(args: argparse.Namespace) -> None:
    """Plot sigma_damp (astrometric damping) consistency across lMax and catalog choices.

    Creates two figures:
    1. Panel figure: Each panel shows sigma_damp vs redshift for different lMax values
       (one row per catalog, one column per instrument).
    2. Summary figure: Heatmap-style view showing median sigma_damp for each (catalog, inst, z_bin, lMax).

    sigma_damp represents fine-scale astrometric error and should be consistent across
    different maximum multipole choices and catalog selections.
    """
    figdir = Path(args.figdir) / args.fitstr_cross

    # Collect all sigma_damp data across lMax values
    cat_results = {}
    for cat in args.cat:
        headstr = args.headstr if cat == "HSC" else None
        cat_results[cat] = {}
        for lMax in args.lmax:
            results = _load_cross_results_merged_jh14(
                args.datadir_cross, cat, headstr, args.fitstr_cross, lMax, maskstr=args.maskstr
            )
            if results is None:
                fpath = _cross_fpath(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax, maskstr=args.maskstr)
                print(f"[plot_sigma_damp] missing {fpath}, skipping {cat} lMax={lMax}")
                continue
            cat_results[cat][lMax] = results

    if not cat_results or not any(cat_results.values()):
        print("[plot_sigma_damp] no results found for any catalog, skipping")
        return

    # Get common properties from first available result
    first_result = next((r for cat_dict in cat_results.values() for r in cat_dict.values()), None)
    if first_result is None:
        return
    zbinedges = first_result["zbinedges"]
    n_zbins = len(zbinedges) - 1
    z_centers = 0.5 * (zbinedges[:-1] + zbinedges[1:])
    lams = {1: 1.1, 2: 1.8}

    # ========== Figure 1: Panel figure (sigma_damp vs z for each lMax) ==========
    fig, axes = plt.subplots(
        len(args.cat), 2,
        figsize=(10, 4 * len(args.cat)),
        sharex=True,
        sharey=True
    )
    if len(args.cat) == 1:
        axes = axes.reshape(1, -1)

    lmax_colors = {lm: _LMAX_COLORS[i] for i, lm in enumerate(args.lmax)}

    for cat_idx, cat in enumerate(args.cat):
        if cat not in cat_results or not cat_results[cat]:
            continue

        for inst_idx, inst in enumerate([1, 2]):
            ax = axes[cat_idx, inst_idx]

            # Plot each lMax as a separate series
            for lMax in args.lmax:
                if lMax not in cat_results[cat]:
                    continue
                results = cat_results[cat][lMax]
                inst_list = list(results["inst_list"])
                if inst not in inst_list:
                    continue
                i_inst = inst_list.index(inst)

                # Extract sigma_damp (6th parameter, index 5) and errors
                sigma_damp = results["params"][i_inst, :, 5]
                sigma_damp_err = results["params_err"][i_inst, :, 5]

                ax.errorbar(
                    z_centers, sigma_damp, yerr=sigma_damp_err,
                    fmt='o', color=lmax_colors[lMax], label=f"ℓ_max={lMax}",
                    markersize=6, capsize=4, capthick=1.5, alpha=0.75, linestyle="-", linewidth=1.5
                )

            ax.set_xlabel("Redshift", fontsize=11)
            ax.set_ylabel(r"$\sigma_{\rm damp}$ [arcsec]", fontsize=11)
            ax.text(
                0.98, 0.97,
                f"{cat} × CIBER {lams[inst]:.1f} μm",
                transform=ax.transAxes,
                fontsize=11,
                ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
            ax.grid(True, alpha=0.3, which='major')
            ax.set_xlim([zbinedges[0] - 0.05, zbinedges[-1] + 0.05])

    # Add legend in upper-left of first subplot
    if len(args.cat) > 0:
        axes[0, 0].legend(loc='upper left', fontsize=10, framealpha=0.95)

    fig.suptitle(
        f"{args.fitstr_cross}: σ_damp Consistency Across ℓ_max",
        fontsize=13,
        y=0.995
    )
    fig.tight_layout()
    _savefig(fig, figdir / f"sigma_damp_vs_lmax_{args.fitstr_cross}", args.fig_fmt)
    plt.close(fig)

    # ========== Figure 2: Consistency heatmap ==========
    # Create a figure showing median sigma_damp for each (cat, inst, z_bin) across lMax
    fig, axes = plt.subplots(len(args.cat), 2, figsize=(10, 3.5 * len(args.cat)))
    if len(args.cat) == 1:
        axes = axes.reshape(1, -1)

    for cat_idx, cat in enumerate(args.cat):
        if cat not in cat_results or not cat_results[cat]:
            continue

        for inst_idx, inst in enumerate([1, 2]):
            ax = axes[cat_idx, inst_idx]

            # Build matrix: rows = z_bins, columns = lMax values
            sigma_damp_matrix = []
            sigma_damp_err_matrix = []
            valid_lmaxes = []

            for lMax in args.lmax:
                if lMax not in cat_results[cat]:
                    continue
                valid_lmaxes.append(lMax)
                results = cat_results[cat][lMax]
                inst_list = list(results["inst_list"])
                if inst not in inst_list:
                    sigma_damp_matrix.append([np.nan] * n_zbins)
                    sigma_damp_err_matrix.append([np.nan] * n_zbins)
                    continue
                i_inst = inst_list.index(inst)

                sigma_damp = results["params"][i_inst, :, 5]
                sigma_damp_err = results["params_err"][i_inst, :, 5]
                sigma_damp_matrix.append(sigma_damp)
                sigma_damp_err_matrix.append(sigma_damp_err)

            if not sigma_damp_matrix:
                continue

            sigma_damp_matrix = np.array(sigma_damp_matrix).T  # Shape: (n_zbins, n_valid_lmaxes)
            sigma_damp_err_matrix = np.array(sigma_damp_err_matrix).T

            # Plot as colored boxes with text annotations
            im = ax.imshow(
                sigma_damp_matrix, cmap='viridis', aspect='auto',
                vmin=np.nanmin(sigma_damp_matrix) * 0.8,
                vmax=np.nanmax(sigma_damp_matrix) * 1.2,
            )

            # Add text annotations
            for i in range(n_zbins):
                for j in range(len(valid_lmaxes)):
                    val = sigma_damp_matrix[i, j]
                    err = sigma_damp_err_matrix[i, j]
                    if not np.isnan(val):
                        text = ax.text(
                            j, i, f"{val:.2f}\n±{err:.2f}",
                            ha="center", va="center",
                            color="white" if val > np.nanmedian(sigma_damp_matrix) else "black",
                            fontsize=9,
                        )

            ax.set_xticks(np.arange(len(valid_lmaxes)))
            ax.set_yticks(np.arange(n_zbins))
            ax.set_xticklabels([f"{lm}" for lm in valid_lmaxes], rotation=45)
            ax.set_yticklabels([f"z∈[{zbinedges[i]:.1f},{zbinedges[i+1]:.1f}]" for i in range(n_zbins)])
            ax.set_xlabel("ℓ_max", fontsize=11)
            ax.set_ylabel("Redshift bin", fontsize=11)
            ax.text(
                0.98, 0.97,
                f"{cat} × CIBER {lams[inst]:.1f} μm",
                transform=ax.transAxes,
                fontsize=11,
                ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            # Add colorbar for this subplot
            cbar = plt.colorbar(im, ax=ax, label=r"$\sigma_{\rm damp}$ [arcsec]")

    fig.suptitle(
        f"{args.fitstr_cross}: σ_damp Heatmap (Median ± Std)",
        fontsize=13,
        y=0.995
    )
    fig.tight_layout()
    _savefig(fig, figdir / f"sigma_damp_heatmap_{args.fitstr_cross}", args.fig_fmt)
    plt.close(fig)


def _plot_redshift_panels_2x2(args: argparse.Namespace) -> None:
    """Generate 2x2 redshift panel figures for each redshift bin.

    Top row: CIBER × DESILS (TM1 and TM2)
    Bottom row: CIBER × HSC (TM1 and TM2)

    Each panel shows data + fitted model with uncertainty bands.
    Shared x/y axes and a single legend positioned above all panels.
    Saves to spectra/ subdirectory to match existing plot layout.
    """
    from ciber.theory.cross_ps_parametric_model import load_fit_results_npz
    from ciber.theory.cl_predictions import grab_ciber_cross_vs_z_predfpaths

    figdir = Path(args.figdir) / args.fitstr_cross / "spectra"
    lMax = args.lmax_components

    # Load results for both catalogs
    results = {}
    for cat in ["DESILS", "HSC"]:
        headstr = args.headstr if cat == "HSC" else None
        cat_results = _load_cross_results_merged_jh14(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax, maskstr=args.maskstr)
        if cat_results is None:
            fpath = _cross_fpath(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax, maskstr=args.maskstr)
            print(f"[plot_redshift_panels_2x2] missing {fpath}, skipping {cat}")
            continue
        results[cat] = cat_results

    if len(results) < 2:
        print("[plot_redshift_panels_2x2] need both DESILS and HSC, skipping")
        return

    desils_results = results["DESILS"]
    hsc_results = results["HSC"]

    n_zbin = desils_results['params'].shape[1]
    zbinedges = desils_results['zbinedges']
    lams = {1: 1.1, 2: 1.8}

    # colors = {
    #     'data': 'k',
    #     'total': 'r',
    #     'two_halo': 'b',
    #     'one_halo': 'g',
    #     'shot_noise': 'grey',
    #     'igl': 'magenta',
    # }

    colors = {
        "data": "#000000",
        "igl": "#595959",
        "total": "red",      # strong red-orange
        "two_halo": "blue",      # muted blue
        # "one_halo": "#E6AC00",      # green (CB-safe shade)
        "one_halo": "#C28B1E",
        "shot_noise": "grey",       # warm orange
    }


    # Load bias cache and build mock pred fpaths (auto-detect from defaults if not provided)
    bias_cache = None
    ls_pred_fpaths_by_inst = {}   # inst -> list of paths, one per zbin
    hsc_pred_fpaths_by_inst = {}
    
    # Auto-detect default bias cache if not provided
    _default_bias_cache = Path(__file__).resolve().parent / 'effective_bias_ls_cache.npz'
    _bias_cache_fpath = getattr(args, 'bias_cache_fpath', None) or (
        str(_default_bias_cache) if _default_bias_cache.exists() else None
    )
    
    # Auto-detect default mock base path if not provided
    _default_mock_base = 'data/jordan_mocks/v2/'
    _mock_basepath = getattr(args, 'mock_basepath', None) or _default_mock_base
    _mock_base = _mock_basepath.rstrip('/') + '/'
    
    # Load bias cache and pred fpaths
    if _bias_cache_fpath and os.path.exists(_bias_cache_fpath):
        try:
            bias_cache = np.load(_bias_cache_fpath, allow_pickle=False)
        except Exception as e:
            print(f"[plot_redshift_panels_2x2] Failed to load bias cache {_bias_cache_fpath}: {e}")
    
    # Try to load pred fpaths from the mock base
    for inst in [1, 2]:
        # LS: try CIBERfidmask headstr first, fall back to plain
        for hs in ['sdss_z_lt_22.0_CIBERfidmask', 'sdss_z_lt_22.0']:
            cands = grab_ciber_cross_vs_z_predfpaths(
                inst_list=[inst], zbinedges=list(zbinedges),
                jmock_basedir=_mock_base, headstr=hs)[0]
            if any(os.path.exists(p) for p in cands):
                ls_pred_fpaths_by_inst[inst] = cands
                break
        # HSC
        for hs in ['hsc_i_lt_25.0', 'hsc_i_lt_25.0_CIBERfidmask']:
            cands = grab_ciber_cross_vs_z_predfpaths(
                inst_list=[inst], zbinedges=list(zbinedges),
                jmock_basedir=_mock_base, headstr=hs)[0]
            if any(os.path.exists(p) for p in cands):
                hsc_pred_fpaths_by_inst[inst] = cands
                break

    # For each redshift bin, create 2×2 panel figure
    for z_idx in range(n_zbin):
        z_low, z_high = zbinedges[z_idx], zbinedges[z_idx + 1]
        z_center = 0.5 * (z_low + z_high)

        # Bias values for this redshift
        b_g_ls = float(np.poly1d(np.asarray(bias_cache['coarse_poly_coeffs']))(z_center)) \
            if bias_cache is not None else None
        b_g_hsc = 1.0 + 0.84 * z_center if bias_cache is not None else None

        fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)

        for row, (cat_results, cat_pred_by_inst, b_g) in enumerate([
            (desils_results, ls_pred_fpaths_by_inst,  b_g_ls),
            (hsc_results,    hsc_pred_fpaths_by_inst, b_g_hsc),
        ]):
            cat_label = "DESI-LS" if row == 0 else "HSC"
            for col, inst in enumerate([1, 2]):
                pred_fpath = cat_pred_by_inst.get(inst, [None] * n_zbin)[z_idx] \
                    if cat_pred_by_inst else None
                _plot_2x2_spectrum_panel(
                    axes[row, col], cat_results, col, z_idx, lMax, colors, lams,
                    title=f"CIBER {lams[inst]} μm × {cat_label}",
                    chi2_reduced=cat_results['reduced_chisq'][col, z_idx],
                    igl_pred_fpath=pred_fpath,
                    b_g=b_g,
                    cat="DESILS" if row == 0 else "HSC",
                    zbinedges_plot=zbinedges,
                    sigma_damp_fixed_map=_parse_sigma_damp_fixed_mapping(args),
                )

        # Add shared legend above all panels
        handles = [
            plt.errorbar([0], [0], yerr=[0.], color=colors['data'], marker='o', capsize=2.5, markersize=3, linestyle='none', label='Data ('+str(z_low)+'$ < z_{\\rm phot} < $'+str(z_high)+')'),
            plt.Line2D([0], [0], linewidth=3.0, linestyle='solid', color='k', alpha=0.6, label='IGL prediction (2h+1h+P)'),
            plt.Line2D([0], [0], color=colors['total'], linewidth=2.5, label='Best-fit model'),
            plt.Line2D([0], [0], color=colors['two_halo'], linestyle='dashdot', linewidth=1.5, alpha=0.7, label='Two-halo'),
            plt.Line2D([0], [0], color=colors['one_halo'], linewidth=1.2, alpha=0.7, linestyle='solid', label='One-halo'),
            plt.Line2D([0], [0], color=colors['shot_noise'], linewidth=1.2, linestyle='dashed', alpha=0.7, label='Poisson level'),
        ]
        # if bias_cache is not None:
        #     handles.append(
        #         plt.Line2D([0], [0], linewidth=2.0, linestyle='solid', color='k', alpha=0.6, label='IGL prediction')
        #     )
        fig.legend(
            handles=handles,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            fontsize=12,
        )
        
        # Common axis labels
        axes[1, 0].set_xlabel(r"$\ell$", fontsize=14)
        axes[1, 1].set_xlabel(r"$\ell$", fontsize=14)
        axes[0, 0].set_ylabel(r"$D_\ell^{\rm Ig}$ [nW m$^{-2}$ sr$^{-1}$]", fontsize=14)
        axes[1, 0].set_ylabel(r"$D_\ell^{\rm Ig}$ [nW m$^{-2}$ sr$^{-1}$]", fontsize=14)
        
        # fig.suptitle(f"Tomographic bin: {z_low} < z < {z_high}", fontsize=14, y=1.05)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        # Save to spectra/ subdirectory
        stem = figdir / f"cross_spectrum_2x2_z{z_low:.01f}_{z_high:.01f}_lMax{lMax}"
        meta_results = desils_results if desils_results.get("onehalo_fsat_model") is not None else hsc_results
        stem = _build_plot_path_with_model(stem, args=args, results=meta_results)
        _savefig(fig, stem, args.fig_fmt)
        plt.close(fig)

def _plot_2x2_spectrum_panel(ax, results, inst_idx, z_idx, lMax, colors, lams,
                              title="", chi2_reduced=None,
                              igl_pred_fpath=None, b_g=None, cat=None, zbinedges_plot=None,
                              sigma_damp_fixed_map=None, show_1h_iglpred=False):
    """Plot a single spectrum panel into a pre-existing axis for the 2x2 figure.

    Mirrors the uncertainty band logic of plot_fit_fixed_1h_templates as called
    by _plot_fit_spectra — which never passes cov_matrix, so always uses params_err
    for per-component bounds and simple linear addition for the total.
    """
    from ciber.theory.cross_ps_parametric_model import CrossPowerSpectrumModel

    lb_fit     = results['lb_fit'][inst_idx, z_idx]
    data_dl    = results['data_dl'][inst_idx, z_idx]
    data_dlerr = results['data_dlerr'][inst_idx, z_idx]

    # Strip NaN-padded params (same as _plot_fit_spectra)
    params     = results['params'][inst_idx, z_idx, :]
    params_err = results['params_err'][inst_idx, z_idx, :]
    samples_bin = results.get('samples', np.empty((0,)))[inst_idx, z_idx]
    n_params_stored = int(np.sum(~np.isnan(params)))
    params     = params[:n_params_stored]
    params_err = params_err[:n_params_stored]

    # Get ndof from saved results (already correctly calculated in fit_model_mcmc)
    ndof_correct = int(results["ndof"][inst_idx, z_idx]) if results.get("ndof") is not None else len(lb_fit) - n_params_stored

    # Detect damping from fitted param names if available
    pnf         = results.get('param_names_fitted', None)
    pnf_bin     = pnf[inst_idx, z_idx] if pnf is not None else None
    use_damping = (pnf_bin is not None and
                   any('damp' in str(p).lower() for p in pnf_bin))
    sigma_damp_fixed_map = _parse_sigma_damp_fixed_mapping(results) if sigma_damp_fixed_map is None else sigma_damp_fixed_map
    sigma_damp_fixed_for_inst = sigma_damp_fixed_map.get(int(results.get('inst_list', [1])[inst_idx]), None)
    if sigma_damp_fixed_for_inst is not None:
        use_damping = True

    use_powerlaw_2h = bool(results.get('use_powerlaw_2h', True))
    alpha_2h_fixed  = float(results.get('alpha_2h_fixed', -1.5))
    use_linear_2h = bool(results.get('use_linear_2h', False))
    
    # Regenerate linear 2H templates if needed (with high ell_max for full plotting range)
    dl_2h_lin_per_zbin = {}
    if use_linear_2h:
        from ciber.theory.cross_ps_parametric_model import _compute_linear_2h_templates_per_zbin
        zbinedges = results.get("zbinedges", np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
        dl_2h_lin_per_zbin = _compute_linear_2h_templates_per_zbin(zbinedges, 1.2e5, verbose=False)

    model = CrossPowerSpectrumModel(
        lb=lb_fit,
        use_powerlaw_2h=use_powerlaw_2h,
        alpha_2h_fixed=alpha_2h_fixed,
        use_astrometry_damping=use_damping,
        use_linear_2h=use_linear_2h,
        dl_2h_lin_per_zbin=dl_2h_lin_per_zbin,
    )

    fit_result = {
        "params": params,
        "params_err": params_err,
        "use_astrometry_damping": use_damping,
        "samples": samples_bin if samples_bin is not None and len(np.asarray(samples_bin).shape) > 0 else None,
        "param_names_fitted": pnf_bin,
        "onehalo_mode": bool(results.get("onehalo_mode", False)),
        "onehalo_output_dir": results.get("onehalo_output_dir", ""),
        "onehalo_generate_type": results.get("onehalo_generate_type", "bulk"),
        "onehalo_fsat_model": results.get("onehalo_fsat_model", "single"),
        "onehalo_population": results.get("onehalo_population", "combined"),
        "onehalo_fit_popmix": bool(results.get("onehalo_fit_popmix", False)),
        "inst": int(results.get("inst_list", [1])[inst_idx]),
        "cat": cat,
    }
    
    # Use provided plotting zbinedges if available, otherwise extract from results
    if zbinedges_plot is None:
        zbinedges_plot = results.get("zbinedges", np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    
    attach_onehalo_template_to_model(
        model, fit_result, z_bin_index=z_idx, use_default_if_missing=False, zbinedges=zbinedges_plot
    )

    # Smooth ell grid matching plot_fit_fixed_1h_templates
    ell_m = np.logspace(2, 5.5, 200)

    # ------------------------------------------------------------------ #
    # Build components
    # ------------------------------------------------------------------ #
    use_popmix = bool(fit_result.get("onehalo_fit_popmix", False))
    f_pop_med = resolve_full_param_value(
        params,
        pnf_bin,
        "f_pop",
        use_astrometry_damping=use_damping,
        use_onehalo_popmix=use_popmix,
    ) if use_popmix else None
    sigma_damp_med = (
        sigma_damp_fixed_for_inst
        if sigma_damp_fixed_for_inst is not None
        else resolve_full_param_value(
            params,
            pnf_bin,
            "sigma_damp",
            use_astrometry_damping=use_damping,
            use_onehalo_popmix=use_popmix,
        )
    ) if use_damping else None
    components = model.model_components(
        ell_m,
        *params[:5],
        sigma_damp=sigma_damp_med,
        z_bin_index=z_idx,
        f_pop=f_pop_med,
    )

    # ------------------------------------------------------------------ #
    # Uncertainty bands: sample-driven percentiles first, params_err fallback.
    # ------------------------------------------------------------------ #
    uncertainty_bands = None

    if samples_bin is not None:
        s = np.asarray(samples_bin, dtype=float)
        if s.ndim == 2 and s.shape[0] > 1:
            nfull = 5 + (1 if use_popmix else 0) + (1 if use_damping else 0)
            if s.shape[1] == nfull:
                sfull = s
            else:
                sfull = expand_fit_samples_to_full_vector(
                    s,
                    np.asarray(params[:nfull], dtype=float),
                    param_names_fitted=pnf_bin,
                    use_astrometry_damping=use_damping,
                    use_onehalo_popmix=use_popmix,
                )

            c2h = np.zeros((sfull.shape[0], ell_m.size))
            c1h = np.zeros((sfull.shape[0], ell_m.size))
            csh = np.zeros((sfull.shape[0], ell_m.size))
            ctot = np.zeros((sfull.shape[0], ell_m.size))

            for ii in range(sfull.shape[0]):
                f_pop_i = sfull[ii, 5] if (use_popmix and sfull.shape[1] > 5) else None
                if use_damping:
                    damp_idx = 6 if use_popmix else 5
                    sd_i = sfull[ii, damp_idx] if sfull.shape[1] > damp_idx else None
                else:
                    sd_i = None
                cc = model.model_components(
                    ell_m,
                    sfull[ii, 0], sfull[ii, 1], sfull[ii, 2], sfull[ii, 3], sfull[ii, 4],
                    sigma_damp=sd_i,
                    z_bin_index=z_idx,
                    f_pop=f_pop_i,
                )
                c2h[ii] = cc['two_halo']
                c1h[ii] = cc['one_halo']
                csh[ii] = cc['shot_noise']
                ctot[ii] = cc['total']

            uncertainty_bands = {
                'two_halo':   (np.percentile(c2h, 16, axis=0), np.percentile(c2h, 84, axis=0)),
                'one_halo':   (np.percentile(c1h, 16, axis=0), np.percentile(c1h, 84, axis=0)),
                'shot_noise': (np.percentile(csh, 16, axis=0), np.percentile(csh, 84, axis=0)),
                'total':      (np.percentile(ctot, 16, axis=0), np.percentile(ctot, 84, axis=0)),
            }

    if uncertainty_bands is None and params_err is not None and not np.any(np.isnan(params_err)):

        # 2-halo bounds
        if model.use_powerlaw_2h:
            dl_2h_upper = model.powerlaw_2h_component(ell_m, params[0] + params_err[0], model.alpha_2h_fixed)
            dl_2h_lower = model.powerlaw_2h_component(ell_m, max(0, params[0] - params_err[0]), model.alpha_2h_fixed)
        elif model.use_linear_2h and z_idx in model.dl_2h_lin_per_zbin:
            # Use linear 2H template for uncertainty bounds
            ell_lin, dl_lin = model.dl_2h_lin_per_zbin[z_idx]
            dl_lin_upper = np.interp(ell_m, ell_lin, dl_lin)
            dl_lin_lower = np.interp(ell_m, ell_lin, dl_lin)
            dl_2h_upper = (params[0] + params_err[0]) * dl_lin_upper
            dl_2h_lower = max(0, params[0] - params_err[0]) * dl_lin_lower
        else:
            pf = ell_m * (ell_m + 1) / (2 * np.pi)
            dl_2h_upper = (params[0] + params_err[0]) * pf * np.interp(ell_m, model.lb, model.cl_2h_pred)
            dl_2h_lower = max(0, params[0] - params_err[0]) * pf * np.interp(ell_m, model.lb, model.cl_2h_pred)

        # 1-halo bounds (amplitude only, shape params fixed at best-fit)
        dl_1h_upper = model.lognormal_component(ell_m, params[1] + params_err[1], params[2], params[3])
        dl_1h_lower = model.lognormal_component(ell_m, max(0, params[1] - params_err[1]), params[2], params[3])

        # Shot noise bounds
        dl_shot_upper = model.shot_noise_component(ell_m, params[4] + params_err[4])
        dl_shot_lower = model.shot_noise_component(ell_m, max(0, params[4] - params_err[4]))

        # Total bounds: simple linear addition (matches plot_fit_fixed_1h_templates fallback)
        if use_damping:
            dl_total_undamped = components.get('total_undamped',
                                               components['two_halo'] +
                                               components['one_halo'] +
                                               components['shot_noise'])
            damping_factor = model.astrometry_damping_component(
                ell_m,
                (
                    sigma_damp_fixed_for_inst
                    if sigma_damp_fixed_for_inst is not None
                    else resolve_full_param_value(
                        params,
                        pnf_bin,
                        "sigma_damp",
                        use_astrometry_damping=use_damping,
                        use_onehalo_popmix=use_popmix,
                    )
                ) if use_damping else None,
            )
            dl_total_upper = (dl_2h_upper + dl_1h_upper + dl_shot_upper) * damping_factor
            dl_total_lower = np.maximum(0, (dl_2h_lower + dl_1h_lower + dl_shot_lower) * damping_factor)
        else:
            dl_total_upper = dl_2h_upper + dl_1h_upper + dl_shot_upper
            dl_total_lower = np.maximum(0, dl_2h_lower + dl_1h_lower + dl_shot_lower)

        uncertainty_bands = {
            'two_halo':   (dl_2h_lower,   dl_2h_upper),
            'one_halo':   (dl_1h_lower,   dl_1h_upper),
            'shot_noise': (dl_shot_lower, dl_shot_upper),
            'total':      (dl_total_lower, dl_total_upper),
        }

    # ------------------------------------------------------------------ #
    # Plot data
    # ------------------------------------------------------------------ #
    ax.errorbar(lb_fit, data_dl, yerr=data_dlerr, fmt='o',
                color=colors['data'], markersize=3, capsize=1.5,
                elinewidth=1.0, alpha=0.8, zorder=7)

    # ------------------------------------------------------------------ #
    # Plot model components + uncertainty bands
    # ------------------------------------------------------------------ #
    ax.loglog(ell_m, components['total'],      color=colors['total'],      lw=2.5, zorder=6)
    ax.loglog(ell_m, components['two_halo'],   color=colors['two_halo'],   lw=1.2, alpha=0.7, linestyle='dashdot', zorder=6)
    ax.loglog(ell_m, components['one_halo'],   color=colors['one_halo'],   lw=1.2, alpha=0.7, zorder=6, linestyle='solid')
    ax.loglog(ell_m, components['shot_noise'], color=colors['shot_noise'], lw=1.2, alpha=0.7, linestyle='dashed', zorder=6)

    if uncertainty_bands is not None:
        ax.fill_between(ell_m,
                        uncertainty_bands['total'][0],      uncertainty_bands['total'][1],
                        color=colors['total'],      alpha=0.2)
        ax.fill_between(ell_m,
                        uncertainty_bands['two_halo'][0],   uncertainty_bands['two_halo'][1],
                        color=colors['two_halo'],   alpha=0.1, zorder=1)
        ax.fill_between(ell_m,
                        uncertainty_bands['one_halo'][0],   uncertainty_bands['one_halo'][1],
                        color=colors['one_halo'],   alpha=0.15, zorder=1)
        ax.fill_between(ell_m,
                        uncertainty_bands['shot_noise'][0], uncertainty_bands['shot_noise'][1],
                        color=colors['shot_noise'], alpha=0.15, zorder=1)

    # ------------------------------------------------------------------ #
    # IGL prediction overlay (bias-scaled smooth model)
    # ------------------------------------------------------------------ #
    if igl_pred_fpath is not None and b_g is not None and os.path.exists(igl_pred_fpath):
        try:
            from ciber.plotting.gal_plotting_fns import smooth_mock_cross_with_bias
            from ciber.theory.onehalo_predict import load_onehalo_spectrum
            ell_igl = np.geomspace(lb_fit.min() * 0.8, lb_fit.max() * 1.2, 300)
            _, dl_igl = smooth_mock_cross_with_bias(igl_pred_fpath, 0.0, b_g, ell_eval=ell_igl)

            onehalo_output_dir ='data/jordan_mocks/v3/fov_10.0/onehalo_predict/'

            if 'hsc' in igl_pred_fpath.lower():
                bandstr_select = 'hsc_i'
                mag_cut = 25.0
            else:
                bandstr_select = 'sdss_z'
                mag_cut = 22.0

            oh_data_Ig = load_onehalo_spectrum(
                        onehalo_output_dir, 'single', bandstr_select,
                        inst=inst_idx+1, mag_min=18.0, mag_cut=mag_cut, z0=0.05, mode='Ig', generate_type='fine')
            ell_1h = oh_data_Ig['ell_arr']
            dl_1h = oh_data_Ig['dl_spectrum'][z_idx]

            dl_1h_interp = np.interp(ell_igl, ell_1h, dl_1h)
            dl_igl += dl_1h_interp


            ax.plot(ell_igl, dl_igl, color='k',
                    linewidth=3.0, linestyle='solid', alpha=0.5, zorder=10)

            if show_1h_iglpred:
                ax.plot(ell_igl, dl_1h_interp, color='k',
                        linewidth=1.5, linestyle='dashed', alpha=0.5, zorder=5)
        except Exception as e:
            print(f"[_plot_2x2_spectrum_panel] IGL overlay failed: {e}")

    # ------------------------------------------------------------------ #
    # Axes formatting
    # ------------------------------------------------------------------ #
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([lb_fit.min() * 0.8, lb_fit.max() * 1.2])
    ax.set_ylim([1e-3, 5e2])
    ax.grid(True, alpha=0.3, which='major')
    ax.set_xticks([1e3, 1e4, 1e5])
    ax.tick_params(axis='both', which='major', labelsize=9)

    # Shade region excluded from fit
    ax.axvspan(lMax-2000, lb_fit.max() * 1.2, color='lightgray', alpha=0.3, zorder=0)

    # Panel label with chi2, bandpower count, and parameter count
    n_bandpowers = len(lb_fit)
    # n_floated = int(n_params_fit) if n_params_fit is not None and not np.isnan(n_params_fit) else n_params_stored
    chi2_str = f"$\chi^2$/dof = {chi2_reduced*ndof_correct:.1f}/{ndof_correct} ({chi2_reduced:.2f})"
    ax.text(0.04, 0.97, f"{title}\n{chi2_str}",
            transform=ax.transAxes, fontsize=12, va='top', ha='left')
    


def _chi2_comparison_with_without_1h(args: argparse.Namespace) -> None:
    """Compare chi2 (both total and reduced) from fits with 1h vs without 1h component.

    Creates figures for each lmax value showing:
    - Total chi2 and reduced chi2 comparisons
    - Degrees of freedom for each fit
    - Chi2 improvement (delta chi2) from including 1h
    
    The improvement shows both how much total chi2 decreases and how the reduced chi2 changes,
    accounting for the different degrees of freedom (with 1h has 3 params, without has 2 params).
    """
    figdir = Path(args.figdir) / args.fitstr_cross

    # Determine the fitstr for no-1h fits
    fitstr_no1h = args.fitstr_cross + "_no1h"

    # Load results for both with and without 1h
    cat_results_with1h = {}
    cat_results_no1h = {}

    for cat in args.cat:
        headstr = args.headstr if cat == "HSC" else None
        cat_results_with1h[cat] = {}
        cat_results_no1h[cat] = {}

        for lMax in args.lmax:
            # With 1h
            fpath_with1h = _cross_fpath(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax)
            if fpath_with1h.exists():
                cat_results_with1h[cat][lMax] = load_fit_results_npz(str(fpath_with1h))

            # Without 1h
            fpath_no1h = _cross_fpath(args.datadir_cross, cat, headstr, fitstr_no1h, lMax)
            if fpath_no1h.exists():
                cat_results_no1h[cat][lMax] = load_fit_results_npz(str(fpath_no1h))

    if not cat_results_with1h or not cat_results_no1h:
        print("[plot_chi2_comparison_with_without_1h] missing results for comparison, skipping")
        return

    # Get common properties
    first_result = next((r for cat_dict in cat_results_with1h.values() for r in cat_dict.values()), None)
    if first_result is None:
        return
    zbinedges = first_result["zbinedges"]
    n_zbins = len(zbinedges) - 1
    z_centers = 0.5 * (zbinedges[:-1] + zbinedges[1:])
    lams = {1: 1.1, 2: 1.8}

    # Generate separate figures for each lmax
    for lMax in args.lmax:
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

        for inst_idx, inst in enumerate([1, 2]):
            col = inst_idx

            # Extract data for this lmax and instrument
            chi2_with_data = {}
            chi2_no_data = {}
            dof_with = {}
            dof_no = {}

            for cat in args.cat:
                if (cat not in cat_results_with1h or lMax not in cat_results_with1h[cat] or
                    cat not in cat_results_no1h or lMax not in cat_results_no1h[cat]):
                    continue

                res_with = cat_results_with1h[cat][lMax]
                res_no = cat_results_no1h[cat][lMax]

                inst_list_with = list(res_with["inst_list"])
                inst_list_no = list(res_no["inst_list"])

                if inst not in inst_list_with or inst not in inst_list_no:
                    continue

                i_inst_with = inst_list_with.index(inst)
                i_inst_no = inst_list_no.index(inst)

                chi2_with_data[cat] = res_with["chisq"][i_inst_with, :]
                chi2_no_data[cat] = res_no["chisq"][i_inst_no, :]
                chi2red_with = res_with["reduced_chisq"][i_inst_with, :]
                chi2red_no = res_no["reduced_chisq"][i_inst_no, :]

                # Compute dof from chi2 and reduced chi2
                dof_with[cat] = chi2_with_data[cat] / chi2red_with
                dof_no[cat] = chi2_no_data[cat] / chi2red_no

            # ROW 0: Total Chi2 comparison
            ax0 = fig.add_subplot(gs[0, col])
            for cat in args.cat:
                if cat in chi2_with_data:
                    ax0.plot(z_centers, chi2_with_data[cat], 'o-', label=f"{cat} (with 1h)",
                            markersize=7, linewidth=2, alpha=0.8)
                if cat in chi2_no_data:
                    ax0.plot(z_centers, chi2_no_data[cat], 's--', label=f"{cat} (no 1h)",
                            markersize=6, linewidth=1.8, alpha=0.6)
            ax0.set_ylabel(r"$\chi^2$ (total)", fontsize=11, fontweight='bold')
            ax0.grid(True, alpha=0.3)
            ax0.legend(loc='best', fontsize=9)
            ax0.set_title(f"CIBER {lams[inst]:.1f} μm", fontsize=12, fontweight='bold')

            # ROW 1: Reduced Chi2 comparison
            ax1 = fig.add_subplot(gs[1, col])
            for cat in args.cat:
                if cat in chi2_with_data:
                    chi2red_with = chi2_with_data[cat] / dof_with[cat]
                    ax1.plot(z_centers, chi2red_with, 'o-', label=f"{cat} (with 1h)",
                            markersize=7, linewidth=2, alpha=0.8)
                if cat in chi2_no_data:
                    chi2red_no = chi2_no_data[cat] / dof_no[cat]
                    ax1.plot(z_centers, chi2red_no, 's--', label=f"{cat} (no 1h)",
                            markersize=6, linewidth=1.8, alpha=0.6)
            ax1.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax1.set_ylabel(r"$\chi^2_{\rm red}$", fontsize=11, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='best', fontsize=9)

            # ROW 2: Delta Chi2 (improvement with 1h)
            ax2 = fig.add_subplot(gs[2, col])
            for cat in args.cat:
                if cat in chi2_with_data and cat in chi2_no_data:
                    delta_chi2 = chi2_no_data[cat] - chi2_with_data[cat]
                    ax2.plot(z_centers, delta_chi2, 'D-', label=cat, markersize=8, linewidth=2.5)
            ax2.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            ax2.set_xlabel("Redshift", fontsize=11)
            ax2.set_ylabel(r"$\Delta \chi^2$ (no1h − with1h)", fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='best', fontsize=9)
            ax2.fill_between(ax2.get_xlim(), 0, ax2.get_ylim()[1], alpha=0.1, color='green',
                             label='1h improves fit' if col == 0 else '')

        # Add text box with summary info
        summary_text = f"""
            ℓ_max = {lMax}
            With 1h: 3 params (A₂ₕ, A₁ₕ, Ashot) + damping
            No 1h: 2 params (A₂ₕ, Ashot) + damping
            Positive Δχ² → 1h helps fit
        """
        fig.text(0.5, 0.02, summary_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        fig.suptitle(
            f"χ² Analysis: With vs Without 1h Component (ℓ_max={lMax})\n"
            f"Top: Total χ², Middle: Reduced χ², Bottom: Improvement from 1h",
            fontsize=13, fontweight='bold', y=0.995
        )
        _savefig(fig, figdir / f"chi2_analysis_with_vs_without_1h_{args.fitstr_cross}_lMax={lMax}", args.fig_fmt)
        plt.close(fig)

    # Create a summary heatmap showing delta_chi2 across all lmax and redshift bins
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for inst_idx, inst in enumerate([1, 2]):
        ax = axes[inst_idx]

        # Build matrix: rows = z_bins, columns = lmax values
        delta_chi2_matrix = []
        dof_diff_matrix = []
        valid_lmaxes = []

        for lMax in args.lmax:
            valid_lmaxes.append(lMax)
            row_delta = []
            row_dof = []

            for cat in args.cat:
                if (cat not in cat_results_with1h or lMax not in cat_results_with1h[cat] or
                    cat not in cat_results_no1h or lMax not in cat_results_no1h[cat]):
                    row_delta = [np.nan] * n_zbins
                    row_dof = [np.nan] * n_zbins
                    break

                res_with = cat_results_with1h[cat][lMax]
                res_no = cat_results_no1h[cat][lMax]

                inst_list_with = list(res_with["inst_list"])
                inst_list_no = list(res_no["inst_list"])

                if inst not in inst_list_with or inst not in inst_list_no:
                    row_delta = [np.nan] * n_zbins
                    row_dof = [np.nan] * n_zbins
                    break

                i_inst_with = inst_list_with.index(inst)
                i_inst_no = inst_list_no.index(inst)

                chi2_with = res_with["chisq"][i_inst_with, :]
                chi2_no = res_no["chisq"][i_inst_no, :]
                chi2red_with = res_with["reduced_chisq"][i_inst_with, :]
                chi2red_no = res_no["reduced_chisq"][i_inst_no, :]

                dof_with_arr = chi2_with / chi2red_with
                dof_no_arr = chi2_no / chi2red_no

                row_delta = chi2_no - chi2_with
                row_dof = dof_with_arr - dof_no_arr  # Difference in dof (should be ~1)

            delta_chi2_matrix.append(row_delta)
            dof_diff_matrix.append(row_dof)

        if not delta_chi2_matrix:
            continue

        delta_chi2_array = np.array(delta_chi2_matrix).T  # Shape: (n_zbins, n_lmax)

        # Plot heatmap
        vmax = np.nanpercentile(delta_chi2_array, 95)
        vmin = -vmax * 0.2  # Allow some negative values
        im = ax.imshow(delta_chi2_array, cmap='RdYlGn', aspect='auto', vmin=vmin, vmax=vmax)

        # Add text annotations
        for i in range(n_zbins):
            for j in range(len(valid_lmaxes)):
                val = delta_chi2_array[i, j]
                if not np.isnan(val):
                    text_color = 'white' if abs(val) > vmax * 0.5 else 'black'
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                           color=text_color, fontsize=10, fontweight='bold')

        ax.set_xticks(np.arange(len(valid_lmaxes)))
        ax.set_yticks(np.arange(n_zbins))
        ax.set_xticklabels([f"{lm//1000}k" for lm in valid_lmaxes], rotation=45)
        ax.set_yticklabels([f"z∈[{zbinedges[i]:.1f},{zbinedges[i+1]:.1f}]" for i in range(n_zbins)])
        ax.set_xlabel("ℓ_max", fontsize=11, fontweight='bold')
        ax.set_title(f"CIBER {lams[inst]:.1f} μm", fontsize=12, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax, label=r"$\Delta \chi^2$ (positive = 1h helps)")

    fig.suptitle(
        f"Total χ² Improvement from Including 1h Component\n"
        f"Heatmap of Δχ² = χ²(no1h) − χ²(with1h) across ℓ_max and redshift",
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout()
    _savefig(fig, figdir / f"chi2_improvement_heatmap_all_lmax_{args.fitstr_cross}", args.fig_fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _chi2_comparison_with_without_2h(args: argparse.Namespace) -> None:
    """Compare chi2 (both total and reduced) from fits with 2h vs without 2h component.

    Creates figures for each lmax value showing:
    - Total chi2 and reduced chi2 comparisons
    - Degrees of freedom for each fit
    - Chi2 improvement (delta chi2) from including 2h
    
    The improvement shows both how much total chi2 decreases and how the reduced chi2 changes,
    accounting for the different degrees of freedom (with 2h has 3 params, without has 2 params).
    """
    figdir = Path(args.figdir) / args.fitstr_cross

    # Determine the fitstr for no-2h fits
    fitstr_no2h = args.fitstr_cross + "_no2h"

    # Load results for both with and without 2h
    cat_results_with2h = {}
    cat_results_no2h = {}

    for cat in args.cat:
        headstr = args.headstr if cat == "HSC" else None
        cat_results_with2h[cat] = {}
        cat_results_no2h[cat] = {}

        for lMax in args.lmax:
            # With 2h
            fpath_with2h = _cross_fpath(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax)
            if fpath_with2h.exists():
                cat_results_with2h[cat][lMax] = load_fit_results_npz(str(fpath_with2h))

            # Without 2h
            fpath_no2h = _cross_fpath(args.datadir_cross, cat, headstr, fitstr_no2h, lMax)
            if fpath_no2h.exists():
                cat_results_no2h[cat][lMax] = load_fit_results_npz(str(fpath_no2h))

    if not cat_results_with2h or not cat_results_no2h:
        print("[plot_chi2_comparison_with_without_2h] missing results for comparison, skipping")
        return

    # Get common properties
    first_result = next((r for cat_dict in cat_results_with2h.values() for r in cat_dict.values()), None)
    if first_result is None:
        return
    zbinedges = first_result["zbinedges"]
    n_zbins = len(zbinedges) - 1
    z_centers = 0.5 * (zbinedges[:-1] + zbinedges[1:])
    lams = {1: 1.1, 2: 1.8}

    # Generate separate figures for each lmax
    for lMax in args.lmax:
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

        for inst_idx, inst in enumerate([1, 2]):
            col = inst_idx

            # Extract data for this lmax and instrument
            chi2_with_data = {}
            chi2_no_data = {}
            dof_with = {}
            dof_no = {}

            for cat in args.cat:
                if (cat not in cat_results_with2h or lMax not in cat_results_with2h[cat] or
                    cat not in cat_results_no2h or lMax not in cat_results_no2h[cat]):
                    continue

                res_with = cat_results_with2h[cat][lMax]
                res_no = cat_results_no2h[cat][lMax]

                inst_list_with = list(res_with["inst_list"])
                inst_list_no = list(res_no["inst_list"])

                if inst not in inst_list_with or inst not in inst_list_no:
                    continue

                i_inst_with = inst_list_with.index(inst)
                i_inst_no = inst_list_no.index(inst)

                chi2_with_data[cat] = res_with["chisq"][i_inst_with, :]
                chi2_no_data[cat] = res_no["chisq"][i_inst_no, :]
                chi2red_with = res_with["reduced_chisq"][i_inst_with, :]
                chi2red_no = res_no["reduced_chisq"][i_inst_no, :]

                # Compute dof from chi2 and reduced chi2
                dof_with[cat] = chi2_with_data[cat] / chi2red_with
                dof_no[cat] = chi2_no_data[cat] / chi2red_no

            # ROW 0: Total Chi2 comparison
            ax0 = fig.add_subplot(gs[0, col])
            for cat in args.cat:
                if cat in chi2_with_data:
                    ax0.plot(z_centers, chi2_with_data[cat], 'o-', label=f"{cat} (with 2h)",
                            markersize=7, linewidth=2, alpha=0.8)
                if cat in chi2_no_data:
                    ax0.plot(z_centers, chi2_no_data[cat], 's--', label=f"{cat} (no 2h)",
                            markersize=6, linewidth=1.8, alpha=0.6)
            ax0.set_ylabel(r"$\chi^2$ (total)", fontsize=11, fontweight='bold')
            ax0.grid(True, alpha=0.3)
            ax0.legend(loc='best', fontsize=9)
            ax0.set_title(f"CIBER {lams[inst]:.1f} μm", fontsize=12, fontweight='bold')

            # ROW 1: Reduced Chi2 comparison
            ax1 = fig.add_subplot(gs[1, col])
            for cat in args.cat:
                if cat in chi2_with_data:
                    chi2red_with = chi2_with_data[cat] / dof_with[cat]
                    ax1.plot(z_centers, chi2red_with, 'o-', label=f"{cat} (with 2h)",
                            markersize=7, linewidth=2, alpha=0.8)
                if cat in chi2_no_data:
                    chi2red_no = chi2_no_data[cat] / dof_no[cat]
                    ax1.plot(z_centers, chi2red_no, 's--', label=f"{cat} (no 2h)",
                            markersize=6, linewidth=1.8, alpha=0.6)
            ax1.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax1.set_ylabel(r"$\chi^2_{\rm red}$", fontsize=11, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='best', fontsize=9)

            # ROW 2: Delta Chi2 (improvement with 2h)
            ax2 = fig.add_subplot(gs[2, col])
            for cat in args.cat:
                if cat in chi2_with_data and cat in chi2_no_data:
                    delta_chi2 = chi2_no_data[cat] - chi2_with_data[cat]
                    ax2.plot(z_centers, delta_chi2, 'D-', label=cat, markersize=8, linewidth=2.5)
            ax2.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            ax2.set_xlabel("Redshift", fontsize=11)
            ax2.set_ylabel(r"$\Delta \chi^2$ (no2h − with2h)", fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='best', fontsize=9)
            ax2.fill_between(ax2.get_xlim(), 0, ax2.get_ylim()[1], alpha=0.1, color='green',
                             label='2h improves fit' if col == 0 else '')

        # Add text box with summary info
        summary_text = f"""
ℓ_max = {lMax}
With 2h: 3 params (A₂ₕ, A₁ₕ, Ashot) + damping
No 2h: 2 params (A₁ₕ, Ashot) + damping
Positive Δχ² → 2h helps fit
        """
        fig.text(0.5, 0.02, summary_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        fig.suptitle(
            f"χ² Analysis: With vs Without 2h Component (ℓ_max={lMax})\n"
            f"Top: Total χ², Middle: Reduced χ², Bottom: Improvement from 2h",
            fontsize=13, fontweight='bold', y=0.995
        )
        _savefig(fig, figdir / f"chi2_analysis_with_vs_without_2h_{args.fitstr_cross}_lMax={lMax}", args.fig_fmt)
        plt.close(fig)

    # Create a summary heatmap showing delta_chi2 across all lmax and redshift bins
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for inst_idx, inst in enumerate([1, 2]):
        ax = axes[inst_idx]

        # Build matrix: rows = z_bins, columns = lmax values
        delta_chi2_matrix = []
        dof_diff_matrix = []
        valid_lmaxes = []

        for lMax in args.lmax:
            valid_lmaxes.append(lMax)
            row_delta = []
            row_dof = []

            for cat in args.cat:
                if (cat not in cat_results_with2h or lMax not in cat_results_with2h[cat] or
                    cat not in cat_results_no2h or lMax not in cat_results_no2h[cat]):
                    row_delta = [np.nan] * n_zbins
                    row_dof = [np.nan] * n_zbins
                    break

                res_with = cat_results_with2h[cat][lMax]
                res_no = cat_results_no2h[cat][lMax]

                inst_list_with = list(res_with["inst_list"])
                inst_list_no = list(res_no["inst_list"])

                if inst not in inst_list_with or inst not in inst_list_no:
                    row_delta = [np.nan] * n_zbins
                    row_dof = [np.nan] * n_zbins
                    break

                i_inst_with = inst_list_with.index(inst)
                i_inst_no = inst_list_no.index(inst)

                chi2_with = res_with["chisq"][i_inst_with, :]
                chi2_no = res_no["chisq"][i_inst_no, :]
                chi2red_with = res_with["reduced_chisq"][i_inst_with, :]
                chi2red_no = res_no["reduced_chisq"][i_inst_no, :]

                dof_with_arr = chi2_with / chi2red_with
                dof_no_arr = chi2_no / chi2red_no

                row_delta = chi2_no - chi2_with
                row_dof = dof_with_arr - dof_no_arr  # Difference in dof (should be ~1)

            delta_chi2_matrix.append(row_delta)
            dof_diff_matrix.append(row_dof)

        if not delta_chi2_matrix:
            continue

        delta_chi2_array = np.array(delta_chi2_matrix).T  # Shape: (n_zbins, n_lmax)

        # Plot heatmap
        vmax = np.nanpercentile(delta_chi2_array, 95)
        vmin = -vmax * 0.2  # Allow some negative values
        im = ax.imshow(delta_chi2_array, cmap='RdYlGn', aspect='auto', vmin=vmin, vmax=vmax)

        # Add text annotations
        for i in range(n_zbins):
            for j in range(len(valid_lmaxes)):
                val = delta_chi2_array[i, j]
                if not np.isnan(val):
                    text_color = 'white' if abs(val) > vmax * 0.5 else 'black'
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                           color=text_color, fontsize=10, fontweight='bold')

        ax.set_xticks(np.arange(len(valid_lmaxes)))
        ax.set_yticks(np.arange(n_zbins))
        ax.set_xticklabels([f"{lm//1000}k" for lm in valid_lmaxes], rotation=45)
        ax.set_yticklabels([f"z∈[{zbinedges[i]:.1f},{zbinedges[i+1]:.1f}]" for i in range(n_zbins)])
        ax.set_xlabel("ℓ_max", fontsize=11, fontweight='bold')
        ax.set_title(f"CIBER {lams[inst]:.1f} μm", fontsize=12, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax, label=r"$\Delta \chi^2$ (positive = 2h helps)")

    fig.suptitle(
        f"Total χ² Improvement from Including 2h Component\n"
        f"Heatmap of Δχ² = χ²(no2h) − χ²(with2h) across ℓ_max and redshift",
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout()
    _savefig(fig, figdir / f"chi2_improvement_heatmap_2h_all_lmax_{args.fitstr_cross}", args.fig_fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chi2 LaTeX table
# ---------------------------------------------------------------------------

def _make_chi2_latex_table(args: argparse.Namespace) -> None:
    """Generate a LaTeX table of chi2 values for full, no-1h, no-2h, and fixed-A2h variants.

    For each lMax in args.lmax, writes a .tex file with a table* environment
    showing chi2 and delta-chi2 for: full model, no-1h ablation, no-2h ablation,
    and fixed-A2h ablations using b_I=1, b_I=1+0.6z, and b_I=(1+z)^2.
    Rows are (z-bin, lambda) pairs. Delta chi2 entries with values > 4 are bolded.

    Output path: {args.figdir}/{args.fitstr_cross}/chi2_table_{fitstr_cross}_lMax={lMax}.tex
    """
    outdir = Path(args.figdir) / args.fitstr_cross
    outdir.mkdir(parents=True, exist_ok=True)

    fitstr_no1h = args.fitstr_cross + "_no1h"
    fitstr_no2h = args.fitstr_cross + "_no2h"
    fitstr_fixA2h = args.fitstr_cross + "_fixA2h_IGL"
    fitstr_fixA2h_lin = args.fitstr_cross + "_fixA2h_IGL_biLinear"
    fitstr_fixA2h_quad = args.fitstr_cross + "_fixA2h_IGL_biQuadratic"

    # Load results for all variants and all catalogs/lMaxes
    results_full: dict = {}
    results_no1h: dict = {}
    results_no2h: dict = {}
    results_fixA2h: dict = {}
    results_fixA2h_lin: dict = {}
    results_fixA2h_quad: dict = {}

    for cat in args.cat:
        headstr = args.headstr if cat == "HSC" else None
        results_full[cat] = {}
        results_no1h[cat] = {}
        results_no2h[cat] = {}
        results_fixA2h[cat] = {}
        results_fixA2h_lin[cat] = {}
        results_fixA2h_quad[cat] = {}
        for lMax in args.lmax:
            for fitstr_variant, store in [
                (args.fitstr_cross, results_full),
                (fitstr_no1h,       results_no1h),
                (fitstr_no2h,       results_no2h),
                (fitstr_fixA2h,     results_fixA2h),
                (fitstr_fixA2h_lin, results_fixA2h_lin),
                (fitstr_fixA2h_quad, results_fixA2h_quad),
            ]:
                # Use merged JHlt14 z<0.2 results for DESILS; falls back to fiducial if no JHlt14 file
                res = _load_cross_results_merged_jh14(args.datadir_cross, cat, headstr, fitstr_variant, lMax, maskstr=args.maskstr if fitstr_variant == args.fitstr_cross else None)
                if res is not None:
                    store[cat][lMax] = res
                else:
                    fpath = _cross_fpath(args.datadir_cross, cat, headstr, fitstr_variant, lMax, maskstr=args.maskstr if fitstr_variant == args.fitstr_cross else None)
                    print(f"[make_chi2_table] not found: {fpath.name}")

    BOLD_THRESH = 4.0

    cat_display = {
        "DESILS": r"CIBER $\times$ DESI-LS",
        "HSC":    r"CIBER $\times$ HSC",
    }

    def _fmt_chi2(val: float) -> str:
        return f"{val:.1f}"

    def _fmt_delta(val: float) -> str:
        sign = "+" if val >= 0 else ""
        s = f"{sign}{val:.1f}"
        if val > BOLD_THRESH:
            return r"$\mathbf{" + s + r"}$"
        return f"${s}$"

    def _fmt_delta_txt(val: float) -> str:
        sign = "+" if val >= 0 else ""
        return f"{sign}{val:.1f}"

    def _get_chi2(res, inst: int, zidx: int):
        """Return total chi2 for (inst, zidx), or None if unavailable."""
        if res is None:
            return None
        inst_list = list(res["inst_list"])
        if inst not in inst_list:
            return None
        return float(res["chisq"][inst_list.index(inst), zidx])

    for lMax in args.lmax:
        cats_present = [c for c in args.cat if lMax in results_full.get(c, {})]
        if not cats_present:
            print(f"[make_chi2_table] no full-model results for lMax={lMax}, skipping")
            continue

        first_res = results_full[cats_present[0]][lMax]
        zbinedges = first_res["zbinedges"]
        n_zbins = len(zbinedges) - 1

        n_cats = len(args.cat)
        # Per catalog: full/no1h/delta/no2h/delta/fix_const/delta/fix_lin/delta/fix_quad/delta
        col_spec = "ll" + "ccccccccccc" * n_cats

        lines = []
        lines.append(r"\begin{table*}")

        lmax_fmt = f"{lMax:,}".replace(",", "{,}")
        lines.append(
            r"\caption{Chi-squared values at fiducial $\ell_{\mathrm{max}}=" + lmax_fmt + r"$ "
            r"for model fits to cross-spectra. "
            r"$\Delta\chi^2$ represents the improvement from including each component or fixing $A_{2h}$, "
            r"with positive values indicating an improvement (or change) in the fit. "
            r"Fixed-$A_{2h}$ cases include $b_I=1$, $b_I=1+0.6z$, and $b_I=(1+z)^2$. "
            r"All configurations with $\Delta \chi^2 > 4$ are indicated in bold.}"
        )
        lines.append(r"\label{tab:chi2_comparison}")
        lines.append(r"\centering")
        lines.append(r"\begin{tabular}{" + col_spec + r"}")
        lines.append(r"\toprule")

        # Top header: catalog names, each spanning 11 columns
        cat_headers = [
            r"\multicolumn{11}{c}{\textbf{" + cat_display.get(c, c) + r"}}"
            for c in args.cat
        ]
        lines.append("& & " + " & ".join(cat_headers) + r" \\")

        # Cmidrules below catalog names
        cmidrules = []
        for ci in range(n_cats):
            lo = 3 + ci * 11
            hi = lo + 10
            cmidrules.append(r"\cmidrule(lr){" + f"{lo}-{hi}" + r"}")
        lines.append(" ".join(cmidrules))

        # Sub-header row 1 (column labels)
        sub1 = [r"Redshift bin", r"$\lambda_{\rm CIBER}$"]
        sub2 = ["", ""]
        for _ in args.cat:
            sub1 += [
                r"$\chi^2$", r"$\chi^2$", r"$\Delta\chi^2_{1h}$",
                r"$\chi^2$", r"$\Delta\chi^2_{2h}$",
                r"$\chi^2$", r"$\Delta\chi^2$",
                r"$\chi^2$", r"$\Delta\chi^2$",
                r"$\chi^2$", r"$\Delta\chi^2$"
            ]
            sub2 += [
                r"(full)", r"(no 1h)", "", r"(no 2h)", "",
                r"(fix $A_{2h}$, $b_I{=}1$)", r"(vs full)",
                r"(fix $A_{2h}$, $b_I{=}1+0.6z$)", r"(vs full)",
                r"(fix $A_{2h}$, $b_I{=}(1+z)^2$)", r"(vs full)"
            ]
        lines.append(" & ".join(sub1) + r" \\")
        lines.append(" & ".join(sub2) + r" \\")
        lines.append(r"\midrule")

        txt_lines = []
        txt_lines.append(f"# Chi2 summary for lMax={lMax}")
        txt_lines.append("# Columns: zbin, lambda_um, then per catalog:")
        txt_lines.append("# full_chi2, no1h_chi2, delta_chi2_1h, no2h_chi2, delta_chi2_2h, fixA2h_const_chi2, delta_chi2_fixA2h_const, fixA2h_linear_chi2, delta_chi2_fixA2h_linear, fixA2h_quadratic_chi2, delta_chi2_fixA2h_quadratic")
        header = ["zbin", "lambda_um"]
        for cat in args.cat:
            cat_tag = cat.lower()
            header += [
                f"{cat_tag}_full_chi2",
                f"{cat_tag}_no1h_chi2",
                f"{cat_tag}_delta_chi2_1h",
                f"{cat_tag}_no2h_chi2",
                f"{cat_tag}_delta_chi2_2h",
                f"{cat_tag}_fixA2h_const_chi2",
                f"{cat_tag}_delta_chi2_fixA2h_const",
                f"{cat_tag}_fixA2h_linear_chi2",
                f"{cat_tag}_delta_chi2_fixA2h_linear",
                f"{cat_tag}_fixA2h_quadratic_chi2",
                f"{cat_tag}_delta_chi2_fixA2h_quadratic",
            ]
        txt_lines.append("\t".join(header))

        # Data rows
        for zidx in range(n_zbins):
            zlo = zbinedges[zidx]
            zhi = zbinedges[zidx + 1]
            z_label = r"\multirow{2}{*}{$" + f"{zlo:.1f}" + r"$--$" + f"{zhi:.1f}" + r"$}"

            for ii, (inst, lam_str) in enumerate([(1, r"$1.1\,\mu$m"), (2, r"$1.8\,\mu$m")]):
                first_col = z_label if ii == 0 else ""
                cells = [first_col, lam_str]
                txt_cells = [f"{zlo:.1f}-{zhi:.1f}", f"{1.1 if inst == 1 else 1.8:.1f}"]

                for cat in args.cat:
                    v_full = _get_chi2(results_full.get(cat, {}).get(lMax), inst, zidx)
                    v_no1h = _get_chi2(results_no1h.get(cat, {}).get(lMax), inst, zidx)
                    v_no2h = _get_chi2(results_no2h.get(cat, {}).get(lMax), inst, zidx)
                    v_fixA2h = _get_chi2(results_fixA2h.get(cat, {}).get(lMax), inst, zidx)
                    v_fixA2h_lin = _get_chi2(results_fixA2h_lin.get(cat, {}).get(lMax), inst, zidx)
                    v_fixA2h_quad = _get_chi2(results_fixA2h_quad.get(cat, {}).get(lMax), inst, zidx)

                    cells.append(_fmt_chi2(v_full) if v_full is not None else "--")
                    cells.append(_fmt_chi2(v_no1h) if v_no1h is not None else "--")
                    cells.append(
                        _fmt_delta(v_no1h - v_full)
                        if (v_full is not None and v_no1h is not None) else "--"
                    )
                    cells.append(_fmt_chi2(v_no2h) if v_no2h is not None else "--")
                    cells.append(
                        _fmt_delta(v_no2h - v_full)
                        if (v_full is not None and v_no2h is not None) else "--"
                    )
                    cells.append(_fmt_chi2(v_fixA2h) if v_fixA2h is not None else "--")
                    cells.append(
                        _fmt_delta(v_fixA2h - v_full)
                        if (v_full is not None and v_fixA2h is not None) else "--"
                    )
                    cells.append(_fmt_chi2(v_fixA2h_lin) if v_fixA2h_lin is not None else "--")
                    cells.append(
                        _fmt_delta(v_fixA2h_lin - v_full)
                        if (v_full is not None and v_fixA2h_lin is not None) else "--"
                    )
                    cells.append(_fmt_chi2(v_fixA2h_quad) if v_fixA2h_quad is not None else "--")
                    cells.append(
                        _fmt_delta(v_fixA2h_quad - v_full)
                        if (v_full is not None and v_fixA2h_quad is not None) else "--"
                    )

                    txt_cells.append(_fmt_chi2(v_full) if v_full is not None else "--")
                    txt_cells.append(_fmt_chi2(v_no1h) if v_no1h is not None else "--")
                    txt_cells.append(
                        _fmt_delta_txt(v_no1h - v_full)
                        if (v_full is not None and v_no1h is not None) else "--"
                    )
                    txt_cells.append(_fmt_chi2(v_no2h) if v_no2h is not None else "--")
                    txt_cells.append(
                        _fmt_delta_txt(v_no2h - v_full)
                        if (v_full is not None and v_no2h is not None) else "--"
                    )
                    txt_cells.append(_fmt_chi2(v_fixA2h) if v_fixA2h is not None else "--")
                    txt_cells.append(
                        _fmt_delta_txt(v_fixA2h - v_full)
                        if (v_full is not None and v_fixA2h is not None) else "--"
                    )
                    txt_cells.append(_fmt_chi2(v_fixA2h_lin) if v_fixA2h_lin is not None else "--")
                    txt_cells.append(
                        _fmt_delta_txt(v_fixA2h_lin - v_full)
                        if (v_full is not None and v_fixA2h_lin is not None) else "--"
                    )
                    txt_cells.append(_fmt_chi2(v_fixA2h_quad) if v_fixA2h_quad is not None else "--")
                    txt_cells.append(
                        _fmt_delta_txt(v_fixA2h_quad - v_full)
                        if (v_full is not None and v_fixA2h_quad is not None) else "--"
                    )

                lines.append(" & ".join(cells) + r" \\")
                txt_lines.append("\t".join(txt_cells))

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table*}")

        tex_content = "\n".join(lines) + "\n"
        outpath = outdir / f"chi2_table_{args.fitstr_cross}_lMax={lMax}.tex"
        outpath.write_text(tex_content)
        print(f"[make_chi2_table] written → {outpath}")

        txt_content = "\n".join(txt_lines) + "\n"
        outpath_txt = outdir / f"chi2_table_{args.fitstr_cross}_lMax={lMax}.txt"
        outpath_txt.write_text(txt_content)
        print(f"[make_chi2_table] written → {outpath_txt}")


# ---------------------------------------------------------------------------
# Amplitude LaTeX table helpers
# ---------------------------------------------------------------------------

def _fmt_asym(val, lo, hi):
    """Format a parameter with asymmetric 68% CI bounds for LaTeX."""
    if val is None:
        return "--"
    try:
        if np.isnan(val):
            return "--"
    except TypeError:
        return "--"
    if lo is None or hi is None:
        return f"${val:.2e}$"
    try:
        if np.isnan(lo) or np.isnan(hi):
            return f"${val:.2e}$"
    except TypeError:
        return f"${val:.2e}$"
    up_err = hi - val
    lo_err = val - lo
    return f"${val:.2e}^{{+{up_err:.2e}}}_{{-{lo_err:.2e}}}$"


def _fmt_sig(sig):
    """Format A_1h detection significance for LaTeX."""
    if sig is None:
        return "--"
    try:
        if np.isnan(sig):
            return "--"
    except TypeError:
        return "--"
    return f"${sig:.1f}\\sigma$"


def _a1h_significance(res, inst_idx, zidx, a1h_param_idx):
    """Compute A_1h detection significance from posterior samples.

    Uses the fraction of posterior samples with A_1h > 0, converted to
    equivalent Gaussian sigma via the normal CDF inverse.  When that fraction
    implies significance >= 5σ (i.e. too few samples in the tail to be
    reliable), falls back to a Gaussian estimate: median / half-CI width.

    Parameters
    ----------
    res : dict
        Loaded fit-results dict (from load_fit_results_npz).
    inst_idx : int
        Instrument index into the (n_inst, n_zbins, …) arrays.
    zidx : int
        Redshift-bin index.
    a1h_param_idx : int
        Index of A_1h in the params / params_16 / params_84 arrays.

    Returns
    -------
    float
        Detection significance in units of σ, or NaN if unavailable.
    """
    from scipy.stats import norm as _spnorm
    SIG_THRESHOLD = 5.0
    cdf_5sig = _spnorm.cdf(SIG_THRESHOLD)
    
    # Early check: if the median value is exactly 0 or NaN, return 0 significance
    par_arr = res.get('params')
    if par_arr is not None:
        try:
            med = float(par_arr[inst_idx, zidx, a1h_param_idx])
            if med <= 0 or np.isnan(med):
                return 0.0
        except (IndexError, TypeError, ValueError):
            pass

    # Prefer fitted-only samples; fall back to full samples
    chain = None
    for key in ('samples_fitted', 'samples'):
        s = res.get(key)
        if s is not None:
            try:
                candidate = s[inst_idx, zidx]
                if candidate is not None and hasattr(candidate, 'ndim') and candidate.ndim == 2 and candidate.shape[0] >= 10:
                    chain = candidate
                    break
            except (IndexError, TypeError):
                pass
    if chain is None:
        return np.nan

    # Find the A_1h column in the chain via param_names_fitted
    chain_a1h_idx = None
    pnames = res.get('param_names_fitted')
    if pnames is not None:
        try:
            bin_names = pnames[inst_idx, zidx]
            if bin_names is not None:
                for k, name in enumerate(bin_names):
                    if 'A_{1h}' in str(name) or 'A_1h' in str(name):
                        chain_a1h_idx = k
                        break
        except (IndexError, TypeError):
            pass
    if chain_a1h_idx is None:
        chain_a1h_idx = a1h_param_idx if a1h_param_idx < chain.shape[1] else 0
    if chain_a1h_idx >= chain.shape[1]:
        return np.nan

    frac = float(np.mean(chain[:, chain_a1h_idx] > 0))

    if frac >= cdf_5sig:
        # Fallback: Gaussian estimate from 16th/84th percentile
        p16_arr = res.get('params_16')
        p84_arr = res.get('params_84')
        if par_arr is not None and p16_arr is not None and p84_arr is not None:
            try:
                med = float(par_arr[inst_idx, zidx, a1h_param_idx])
                lo  = float(p16_arr[inst_idx, zidx, a1h_param_idx])
                hi  = float(p84_arr[inst_idx, zidx, a1h_param_idx])
                half_ci = (hi - lo) / 2.0
                return med / half_ci if half_ci > 0 else SIG_THRESHOLD
            except (IndexError, TypeError, ValueError):
                pass
        return SIG_THRESHOLD
    elif frac <= 0:
        return 0.0
    else:
        return float(_spnorm.ppf(frac))


# ---------------------------------------------------------------------------
# Amplitude LaTeX table
# ---------------------------------------------------------------------------

def _make_amplitude_table(args: argparse.Namespace) -> None:
    """Generate a LaTeX table of amplitudes for full, no-1h, no-2h, and fixed-A2h variants.

    For each lMax in args.lmax, writes a .tex file with a table showing the best-fit amplitudes
    extracted from the 'params' array in the fit results. Shows full model, no-1h ablation, no-2h ablation,
    and fixed-A_2h-to-IGL ablations (b_I=1, b_I=1+0.6z, b_I=(1+z)^2).

    Output path: {args.figdir}/{args.fitstr_cross}/amplitude_table_{fitstr_cross}_lMax={lMax}.tex
    """
    outdir = Path(args.figdir) / args.fitstr_cross
    outdir.mkdir(parents=True, exist_ok=True)

    fitstr_no1h = args.fitstr_cross + "_no1h"
    fitstr_no2h = args.fitstr_cross + "_no2h"
    fitstr_fixA2h = args.fitstr_cross + "_fixA2h_IGL"
    fitstr_fixA2h_lin = args.fitstr_cross + "_fixA2h_IGL_biLinear"
    fitstr_fixA2h_quad = args.fitstr_cross + "_fixA2h_IGL_biQuadratic"

    # Load results for full, no-1h, no-2h, and fixed-A2h variants
    results_full: dict = {}
    results_no1h: dict = {}
    results_no2h: dict = {}
    results_fixA2h: dict = {}
    results_fixA2h_lin: dict = {}
    results_fixA2h_quad: dict = {}

    for cat in args.cat:
        headstr = args.headstr if cat == "HSC" else None
        results_full[cat] = {}
        results_no1h[cat] = {}
        results_no2h[cat] = {}
        results_fixA2h[cat] = {}
        results_fixA2h_lin[cat] = {}
        results_fixA2h_quad[cat] = {}
        for lMax in args.lmax:
            fpath_full = _cross_fpath(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax)
            if fpath_full.exists():
                results_full[cat][lMax] = load_fit_results_npz(str(fpath_full))
            else:
                print(f"[amplitude_table] not found: {fpath_full.name}")

            fpath_no1h = _cross_fpath(args.datadir_cross, cat, headstr, fitstr_no1h, lMax)
            if fpath_no1h.exists():
                results_no1h[cat][lMax] = load_fit_results_npz(str(fpath_no1h))
            else:
                print(f"[amplitude_table] not found: {fpath_no1h.name}")

            fpath_no2h = _cross_fpath(args.datadir_cross, cat, headstr, fitstr_no2h, lMax)
            if fpath_no2h.exists():
                results_no2h[cat][lMax] = load_fit_results_npz(str(fpath_no2h))
            else:
                print(f"[amplitude_table] not found: {fpath_no2h.name}")

            fpath_fixA2h = _cross_fpath(args.datadir_cross, cat, headstr, fitstr_fixA2h, lMax)
            if fpath_fixA2h.exists():
                results_fixA2h[cat][lMax] = load_fit_results_npz(str(fpath_fixA2h))
            else:
                print(f"[amplitude_table] not found: {fpath_fixA2h.name}")

            fpath_fixA2h_lin = _cross_fpath(args.datadir_cross, cat, headstr, fitstr_fixA2h_lin, lMax)
            if fpath_fixA2h_lin.exists():
                results_fixA2h_lin[cat][lMax] = load_fit_results_npz(str(fpath_fixA2h_lin))
            else:
                print(f"[amplitude_table] not found: {fpath_fixA2h_lin.name}")

            fpath_fixA2h_quad = _cross_fpath(args.datadir_cross, cat, headstr, fitstr_fixA2h_quad, lMax)
            if fpath_fixA2h_quad.exists():
                results_fixA2h_quad[cat][lMax] = load_fit_results_npz(str(fpath_fixA2h_quad))
            else:
                print(f"[amplitude_table] not found: {fpath_fixA2h_quad.name}")

    cat_display = {
        "DESILS": r"CIBER $\times$ DESI-LS",
        "HSC":    r"CIBER $\times$ HSC",
    }

    def _fmt_asym_txt(val, lo, hi):
        if val is None:
            return "--"
        try:
            if np.isnan(val):
                return "--"
        except TypeError:
            return "--"
        if lo is None or hi is None:
            return f"{val:.2e}"
        try:
            if np.isnan(lo) or np.isnan(hi):
                return f"{val:.2e}"
        except TypeError:
            return f"{val:.2e}"
        up_err = hi - val
        lo_err = val - lo
        return f"{val:.2e} (+{up_err:.2e}/-{lo_err:.2e})"

    def _fmt_sig_txt(sig):
        if sig is None:
            return "--"
        try:
            if np.isnan(sig):
                return "--"
        except TypeError:
            return "--"
        return f"{sig:.1f} sigma"

    for lMax in args.lmax:
        cats_present = [c for c in args.cat if lMax in results_full.get(c, {})]
        if not cats_present:
            print(f"[amplitude_table] no full-model results for lMax={lMax}, skipping")
            continue

        first_res = results_full[cats_present[0]][lMax]
        zbinedges = first_res["zbinedges"]
        n_zbins = len(zbinedges) - 1
        lams = {1: 1.1, 2: 1.8}

        n_cats = len(args.cat)
        # Column spec per catalog:
        # Full (4) | No-1h (2) | No-2h (3) | Fix const (3) | Fix linear (3) | Fix quadratic (3)
        col_spec = "ll" + "cccc|cc|ccc|ccc|ccc|ccc" * n_cats

        lines = []
        lines.append(r"\begin{table*}")

        lmax_fmt = f"{lMax:,}".replace(",", "{,}")
        lines.append(
            r"\caption{Best-fit amplitude parameters with asymmetric 68\% credible intervals at "
            r"$\ell_{\mathrm{max}}=" + lmax_fmt + r"$. "
            r"Significance ($\sigma$) is derived from the posterior fraction with $A_{1h}>0$, "
            r"falling back to a Gaussian estimate (median/half-CI) above $5\sigma$. "
            r"Per catalog: Full model; No-1h ablation; No-2h ablation; "
            r"Fixed $A_{2h}$ to IGL with $b_I=1$, $b_I=1+0.6z$, and $b_I=(1+z)^2$.}"
        )
        lines.append(r"\label{tab:amplitude_comparison}")
        lines.append(r"\centering")
        lines.append(r"\begin{tabular}{" + col_spec + r"}")
        lines.append(r"\toprule")

        # Top header: catalog names (18 cols per cat)
        cat_headers = []
        for c in args.cat:
            cat_headers.append(
                r"\multicolumn{18}{c}{\textbf{" + cat_display.get(c, c) + r"}}"
            )
        lines.append("& & " + " & ".join(cat_headers) + r" \\")

        # Cmidrules (18 cols per cat, starting at col 3)
        cmidrules = []
        for ci in range(n_cats):
            lo = 3 + ci * 18
            hi = lo + 17
            cmidrules.append(r"\cmidrule(lr){" + f"{lo}-{hi}" + r"}")
        lines.append(" ".join(cmidrules))

        # Sub-header: parameter names
        sub1 = [r"Redshift bin", r"$\lambda_{\rm CIBER}$"]
        sub2 = ["", ""]
        for _ in args.cat:
            sub1 += [r"$A_{2h}$", r"$A_{1h}$", r"$\sigma_{A_{1h}}$", r"$A_{\mathrm{shot}}$",
                     r"$A_{2h}$", r"$A_{\mathrm{shot}}$",
                     r"$A_{1h}$", r"$\sigma_{A_{1h}}$", r"$A_{\mathrm{shot}}$",
                     r"$A_{1h}$", r"$\sigma_{A_{1h}}$", r"$A_{\mathrm{shot}}$",
                     r"$A_{1h}$", r"$\sigma_{A_{1h}}$", r"$A_{\mathrm{shot}}$",
                     r"$A_{1h}$", r"$\sigma_{A_{1h}}$", r"$A_{\mathrm{shot}}$"]
            sub2 += [r"(full)", r"(full)", r"(full)", r"(full)",
                     r"(no 1h)", r"(no 1h)",
                     r"(no 2h)", r"(no 2h)", r"(no 2h)",
                     r"(fix $A_{2h}$, $b_I{=}1$)", r"(fix $A_{2h}$, $b_I{=}1$)", r"(fix $A_{2h}$, $b_I{=}1$)",
                     r"(fix $A_{2h}$, $b_I{=}1+0.6z$)", r"(fix $A_{2h}$, $b_I{=}1+0.6z$)", r"(fix $A_{2h}$, $b_I{=}1+0.6z$)",
                     r"(fix $A_{2h}$, $b_I{=}(1+z)^2$)", r"(fix $A_{2h}$, $b_I{=}(1+z)^2$)", r"(fix $A_{2h}$, $b_I{=}(1+z)^2$)"]
        lines.append(" & ".join(sub1) + r" \\")
        lines.append(" & ".join(sub2) + r" \\")
        lines.append(r"\midrule")

        txt_lines = []
        txt_lines.append(f"# Amplitude and significance summary for lMax={lMax}")
        txt_lines.append("# Columns: zbin, lambda_um, then per catalog:")
        txt_lines.append("# full_A2h, full_A1h, full_sigA1h, full_Ashot, no1h_A2h, no1h_Ashot, no2h_A1h, no2h_sigA1h, no2h_Ashot, fixA2h_const_A1h, fixA2h_const_sigA1h, fixA2h_const_Ashot, fixA2h_linear_A1h, fixA2h_linear_sigA1h, fixA2h_linear_Ashot, fixA2h_quadratic_A1h, fixA2h_quadratic_sigA1h, fixA2h_quadratic_Ashot")
        header = ["zbin", "lambda_um"]
        for cat in args.cat:
            cat_tag = cat.lower()
            header += [
                f"{cat_tag}_full_A2h",
                f"{cat_tag}_full_A1h",
                f"{cat_tag}_full_sigA1h",
                f"{cat_tag}_full_Ashot",
                f"{cat_tag}_no1h_A2h",
                f"{cat_tag}_no1h_Ashot",
                f"{cat_tag}_no2h_A1h",
                f"{cat_tag}_no2h_sigA1h",
                f"{cat_tag}_no2h_Ashot",
                f"{cat_tag}_fixA2h_const_A1h",
                f"{cat_tag}_fixA2h_const_sigA1h",
                f"{cat_tag}_fixA2h_const_Ashot",
                f"{cat_tag}_fixA2h_linear_A1h",
                f"{cat_tag}_fixA2h_linear_sigA1h",
                f"{cat_tag}_fixA2h_linear_Ashot",
                f"{cat_tag}_fixA2h_quadratic_A1h",
                f"{cat_tag}_fixA2h_quadratic_sigA1h",
                f"{cat_tag}_fixA2h_quadratic_Ashot",
            ]
        txt_lines.append("\t".join(header))

        # Data rows
        for zidx in range(n_zbins):
            zlo = zbinedges[zidx]
            zhi = zbinedges[zidx + 1]
            z_label = r"\multirow{2}{*}{$" + f"{zlo:.1f}" + r"$--$" + f"{zhi:.1f}" + r"$}"

            for ii, (inst, lam_str) in enumerate([(1, r"$1.1\,\mu$m"), (2, r"$1.8\,\mu$m")]):
                first_col = z_label if ii == 0 else ""
                cells = [first_col, lam_str]
                txt_cells = [f"{zlo:.1f}-{zhi:.1f}", f"{lams[inst]:.1f}"]

                for cat in args.cat:
                    res_full   = results_full.get(cat, {}).get(lMax)
                    res_no1h   = results_no1h.get(cat, {}).get(lMax)
                    res_no2h   = results_no2h.get(cat, {}).get(lMax)
                    res_fixA2h = results_fixA2h.get(cat, {}).get(lMax)
                    res_fixA2h_lin = results_fixA2h_lin.get(cat, {}).get(lMax)
                    res_fixA2h_quad = results_fixA2h_quad.get(cat, {}).get(lMax)

                    def _extract(res, param_idx):
                        """Extract (median, p16, p84) for param at param_idx from a result dict."""
                        if res is None:
                            return (None, None, None)
                        il = list(res["inst_list"])
                        if inst not in il:
                            return (None, None, None)
                        i_i = il.index(inst)
                        par = res["params"][i_i, zidx, :]
                        n_p = int(np.sum(~np.isnan(par)))
                        if n_p <= param_idx:
                            return (None, None, None)
                        val = float(par[param_idx])
                        p16 = res.get("params_16")
                        p84 = res.get("params_84")
                        lo  = float(p16[i_i, zidx, param_idx]) if p16 is not None else None
                        hi  = float(p84[i_i, zidx, param_idx]) if p84 is not None else None
                        return (val, lo, hi)

                    def _shot_idx(res):
                        """Return A_shot parameter index for this result dict/instrument/zbin."""
                        if res is None:
                            return None
                        il = list(res["inst_list"])
                        if inst not in il:
                            return None
                        i_i = il.index(inst)
                        par = res["params"][i_i, zidx, :]
                        n_p = int(np.sum(~np.isnan(par)))
                        if n_p <= 1:    return None
                        if n_p == 2:    return 1   # [A_1h, A_shot] or [A_2h, A_shot]
                        if n_p == 3:    return 2   # [A_2h, A_1h, A_shot]
                        if n_p >= 5:    return 4   # [A_2h, A_1h, mu, sigma, A_shot, ...]
                        return n_p - 1  # fallback

                    def _sig(res, a1h_pidx):
                        """Compute A_1h significance for inst/zidx using posterior samples."""
                        if res is None:
                            return np.nan
                        il = list(res["inst_list"])
                        if inst not in il:
                            return np.nan
                        return _a1h_significance(res, il.index(inst), zidx, a1h_pidx)

                    def _find_a1h_idx(res, fallback: int = 1) -> int:
                        """Return the A_1h column index from param_names_fitted, or fallback."""
                        if res is None:
                            return fallback
                        pnames = res.get("param_names_fitted")
                        if pnames is not None:
                            il = list(res["inst_list"])
                            if inst not in il:
                                return fallback
                            i_i = il.index(inst)
                            try:
                                bin_names = pnames[i_i, zidx]
                                if bin_names is not None:
                                    for k, nm in enumerate(bin_names):
                                        if "A_1h" in str(nm) or "A_{1h}" in str(nm):
                                            return k
                            except (IndexError, TypeError):
                                pass
                        return fallback

                    # --- Full model: A_2h[0], A_1h[1], A_shot[shot_idx] ---
                    a2h_full  = _extract(res_full, 0)
                    a1h_full  = _extract(res_full, 1)
                    shot_full = _extract(res_full, _shot_idx(res_full)) if _shot_idx(res_full) is not None else (None, None, None)
                    sig_full  = _sig(res_full, 1)

                    # --- No-1h model: A_2h[0], A_shot[1] ---
                    a2h_no1h  = _extract(res_no1h, 0)
                    shot_no1h = _extract(res_no1h, 1)

                    # --- No-2h model: A_1h[name-based], A_shot[shot_idx] ---
                    _no2h_a1h_idx = _find_a1h_idx(res_no2h, fallback=0)
                    a1h_no2h  = _extract(res_no2h, _no2h_a1h_idx)
                    shot_no2h = _extract(res_no2h, _shot_idx(res_no2h)) if _shot_idx(res_no2h) is not None else (None, None, None)
                    sig_no2h  = _sig(res_no2h, _no2h_a1h_idx)

                    # --- FixA2h model: A_1h[name-based], A_shot[shot_idx] ---
                    _fixA2h_a1h_idx = _find_a1h_idx(res_fixA2h, fallback=1)
                    a1h_fixA2h  = _extract(res_fixA2h, _fixA2h_a1h_idx)
                    shot_fixA2h = _extract(res_fixA2h, _shot_idx(res_fixA2h)) if _shot_idx(res_fixA2h) is not None else (None, None, None)
                    sig_fixA2h  = _sig(res_fixA2h, _fixA2h_a1h_idx)

                    # --- FixA2h linear model: A_1h[name-based], A_shot[shot_idx] ---
                    _fixA2h_lin_a1h_idx = _find_a1h_idx(res_fixA2h_lin, fallback=1)
                    a1h_fixA2h_lin  = _extract(res_fixA2h_lin, _fixA2h_lin_a1h_idx)
                    shot_fixA2h_lin = _extract(res_fixA2h_lin, _shot_idx(res_fixA2h_lin)) if _shot_idx(res_fixA2h_lin) is not None else (None, None, None)
                    sig_fixA2h_lin  = _sig(res_fixA2h_lin, _fixA2h_lin_a1h_idx)

                    # --- FixA2h quadratic model: A_1h[name-based], A_shot[shot_idx] ---
                    _fixA2h_quad_a1h_idx = _find_a1h_idx(res_fixA2h_quad, fallback=1)
                    a1h_fixA2h_quad  = _extract(res_fixA2h_quad, _fixA2h_quad_a1h_idx)
                    shot_fixA2h_quad = _extract(res_fixA2h_quad, _shot_idx(res_fixA2h_quad)) if _shot_idx(res_fixA2h_quad) is not None else (None, None, None)
                    sig_fixA2h_quad  = _sig(res_fixA2h_quad, _fixA2h_quad_a1h_idx)

                    # Append columns: full (4) | no1h (2) | no2h (3) | fixA2h const (3) | fixA2h linear (3) | fixA2h quadratic (3)
                    cells.append(_fmt_asym(*a2h_full))
                    cells.append(_fmt_asym(*a1h_full))
                    cells.append(_fmt_sig(sig_full))
                    cells.append(_fmt_asym(*shot_full))

                    cells.append(_fmt_asym(*a2h_no1h))
                    cells.append(_fmt_asym(*shot_no1h))

                    cells.append(_fmt_asym(*a1h_no2h))
                    cells.append(_fmt_sig(sig_no2h))
                    cells.append(_fmt_asym(*shot_no2h))

                    cells.append(_fmt_asym(*a1h_fixA2h))
                    cells.append(_fmt_sig(sig_fixA2h))
                    cells.append(_fmt_asym(*shot_fixA2h))

                    cells.append(_fmt_asym(*a1h_fixA2h_lin))
                    cells.append(_fmt_sig(sig_fixA2h_lin))
                    cells.append(_fmt_asym(*shot_fixA2h_lin))

                    cells.append(_fmt_asym(*a1h_fixA2h_quad))
                    cells.append(_fmt_sig(sig_fixA2h_quad))
                    cells.append(_fmt_asym(*shot_fixA2h_quad))

                    txt_cells.append(_fmt_asym_txt(*a2h_full))
                    txt_cells.append(_fmt_asym_txt(*a1h_full))
                    txt_cells.append(_fmt_sig_txt(sig_full))
                    txt_cells.append(_fmt_asym_txt(*shot_full))

                    txt_cells.append(_fmt_asym_txt(*a2h_no1h))
                    txt_cells.append(_fmt_asym_txt(*shot_no1h))

                    txt_cells.append(_fmt_asym_txt(*a1h_no2h))
                    txt_cells.append(_fmt_sig_txt(sig_no2h))
                    txt_cells.append(_fmt_asym_txt(*shot_no2h))

                    txt_cells.append(_fmt_asym_txt(*a1h_fixA2h))
                    txt_cells.append(_fmt_sig_txt(sig_fixA2h))
                    txt_cells.append(_fmt_asym_txt(*shot_fixA2h))

                    txt_cells.append(_fmt_asym_txt(*a1h_fixA2h_lin))
                    txt_cells.append(_fmt_sig_txt(sig_fixA2h_lin))
                    txt_cells.append(_fmt_asym_txt(*shot_fixA2h_lin))

                    txt_cells.append(_fmt_asym_txt(*a1h_fixA2h_quad))
                    txt_cells.append(_fmt_sig_txt(sig_fixA2h_quad))
                    txt_cells.append(_fmt_asym_txt(*shot_fixA2h_quad))

                lines.append(" & ".join(cells) + r" \\")
                txt_lines.append("\t".join(txt_cells))

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table*}")

        tex_content = "\n".join(lines) + "\n"
        outpath = outdir / f"amplitude_table_{args.fitstr_cross}_lMax={lMax}.tex"
        outpath.write_text(tex_content)
        print(f"[amplitude_table] written → {outpath}")

        txt_content = "\n".join(txt_lines) + "\n"
        outpath_txt = outdir / f"amplitude_table_{args.fitstr_cross}_lMax={lMax}.txt"
        outpath_txt.write_text(txt_content)
        print(f"[amplitude_table] written → {outpath_txt}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Galaxy auto and CIBER x galaxy cross fit pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        nargs="+",
        choices=["run_auto", "run_cross", "plot_auto", "plot_cross", "plot_components",
                 "plot_compare_cats", "plot_fit_spectra", "plot_spectra_summary",
                 "plot_fpop_vs_redshift", "plot_corner", "plot_corner_overlay", "plot_corr_a1h_a2h", "plot_sigma_damp", "plot_chi2_1h",
                 "plot_chi2_2h", "plot_redshift_panels_2x2", "plot_a1h_vs_redshift",
                 "plot_a1h_vs_redshift_three_row", "plot_a1h_vs_redshift_alternate_layout", "plot_a1h_vs_redshift_mag_comparison",
                 "plot_a1h_model_pred_vs_redshift", "plot_a1h_band_ratio_vs_redshift", "plot_parameter_consistency_vs_lmax",
                 "plot_a2h_vs_redshift", "plot_di_dz_upper_limits", "plot_d_ell_1h_evolution",
                 "plot_r1h_ratio", "plot_ihl_and_dell_combined",
                 "make_chi2_table", "make_amplitude_table", 
                 "param_priors_table", "all"],
        default=["plot_auto", "plot_cross"],
        help="Pipeline mode(s) to execute",
    )
    parser.add_argument(
        "--cat",
        nargs="+",
        choices=["HSC", "DESILS"],
        default=["HSC"],
        help="Catalog(s) to process",
    )
    parser.add_argument(
        "--lmax",
        type=int,
        nargs="+",
        default=[20000, 30000, 50000],
        help="Multipole maximum values to sweep over",
    )

    # Fit labels
    parser.add_argument("--fitstr-auto", default="two_stage_fixed_1h", help="Fit label for auto fits")
    parser.add_argument(
        "--fitstr-cross",
        default="IHL1hfit_fixshape_v8_unifhighell",
        help="Fit label for cross fits (default: IHL1hfit_fixshape_v8_unifhighell)",
    )

    # Field / catalog settings
    parser.add_argument("--ifield-hsc", type=int, nargs="+", default=[8], help="ifield list for HSC")
    parser.add_argument("--ifield-ls", type=int, nargs="+", default=[4, 5, 6, 7, 8], help="ifield list for DESI-LS")
    parser.add_argument("--headstr", default="hsc_ilt25.0", help="Header string (magnitude limit tag) for HSC; set to None to override 3-row layout in plot_a1h_vs_redshift_three_row")
    parser.add_argument(
        "--zbinedges",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        help="Redshift bin edges",
    )
    parser.add_argument("--combined-zbin", action="store_true", default=False,
                        help="Treat full redshift range as a single bin (0<z<1)")
    parser.add_argument("--fmask", type=float, default=0.7, help="Mask fraction")

    # MCMC settings
    parser.add_argument("--nwalkers", type=int, default=32, help="MCMC walkers")
    parser.add_argument("--nsteps1", type=int, default=1000, help="MCMC steps stage 1")
    parser.add_argument("--nsteps2", type=int, default=4000, help="MCMC steps stage 2")
    parser.add_argument("--nburn1", type=int, default=500, help="Burn-in stage 1")
    parser.add_argument("--nburn2", type=int, default=1000, help="Burn-in stage 2")
    parser.add_argument("--no-damping", action="store_false", dest="use_damping", default=True,
                        help="Disable astrometric damping term in cross fits (default: enabled)")
    parser.add_argument("--no-one-halo", action="store_false", dest="use_one_halo", default=True,
                        help="Disable one-halo component in cross fits (default: enabled)")
    parser.add_argument("--no-two-halo", action="store_false", dest="use_two_halo", default=True,
                        help="Disable two-halo component in cross fits (default: enabled)")
    parser.add_argument("--use-linear-2h", action="store_true", dest="use_linear_2h", default=False,
                        help="Use linear matter power spectrum C_ell^lin for 2h template instead of power-law. "
                             "Pre-computes templates per z-bin via Limber projection. "
                             "Output fitstr gains '_lin2h' suffix.")
    parser.add_argument("--fix-sigma-damp", type=float, nargs=2, default=None,
                        dest="sigma_damp_fixed",
                        metavar=("TM1_ARCSEC", "TM2_ARCSEC"),
                        help="Fix astrometric damping sigma_damp to specified values (in arcsec) for each instrument. "
                             "Two values required: one for TM1 (1.1um), one for TM2 (1.8um). "
                             "Example: --fix-sigma-damp 2.5 1.8 fixes TM1 to 2.5 arcsec, TM2 to 1.8 arcsec. "
                             "Output fitstr gains '_fixsigma' suffix.")
    parser.add_argument("--fix-a2h-igl", action="store_true", dest="fix_a2h_igl", default=False,
                        help="Fix A_2h to IGL-predicted (bias-corrected) values per z-bin, then "
                             "fit only A_1h + A_shot.  Output fitstr gains '_fixA2h_IGL' suffix.")
    parser.add_argument("--bi-model", dest="bi_model", default="constant",
                        choices=["constant", "linear", "quadratic"],
                        help="IHL brightness-bias model used when --fix-a2h-igl is set. "
                             "'constant': b_I=1 (default, backward compatible); "
                             "'linear': b_I=1+0.6z; 'quadratic': b_I=(1+z)^2. "
                             "Non-constant models gain a suffix on the fitstr "
                             "(_biLinear or _biQuadratic).")
    parser.add_argument("--igl-pred-basedir", default="data/jordan_mocks/v2/",
                        help="Base directory for Jordan mock IGL cross predictions "
                             "(default: data/jordan_mocks/v2/).")
    parser.add_argument("--igl-pred-headstr", default=None,
                        help="Headstring for IGL prediction files, e.g. 'sdss_z_lt_22.0'. "
                             "Defaults to catalog-specific value if not set.")
    parser.add_argument("--uniform-weight-ell", type=float, default=None,
                        help="Uniform weighting threshold (ell_min). Above this multipole, apply uniform field weighting instead of error-weighted. Default: None (use error-weighted for all)")
    parser.add_argument("--maskstr", default=None,
                        help="Mask string tag appended to cross spectra filenames, e.g. 'JHlt14' or 'JHlt15'. "
                             "Selects files named *_wrandsub_<maskstr>_wFFerr.npz. Default: None (fiducial JHlt16 masks).")

    # Paths
    parser.add_argument("--figdir", default="figures/", help="Output figure directory")
    parser.add_argument("--fig-fmt", choices=["pdf", "png"], default="pdf",
                        help="Figure format: pdf (default) or png (dpi=300)")
    parser.add_argument("--datadir-auto", default="data/gal_auto_fits/", help="Directory for auto fit .npz files")
    parser.add_argument("--datadir-cross", default="data/cross_cl_fits/", help="Directory for cross fit .npz files")
    parser.add_argument(
        "--ihl-params",
        default="data/ihl_1h_params_corrected.npz",
        help="Path to IHL 1h parameter file",
    )
    parser.add_argument(
        "--onehalo-dir",
        default=None,
        help="Directory containing onehalo_predict outputs to use as fixed one-halo templates",
    )
    parser.add_argument(
        "--onehalo-generate-type",
        default="bulk",
        choices=["bulk", "fine"],
        help="onehalo_predict output type to load",
    )
    parser.add_argument(
        "--onehalo-fsat-model",
        default="single",
        help="Satellite fraction model used when loading onehalo_predict outputs",
    )
    parser.add_argument(
        "--concentration-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the NFW concentration when loading one-halo templates; use a value != 1.0 to select concentration-specific saved files",
    )
    parser.add_argument(
        "--onehalo-population",
        default="combined",
        choices=["combined", "pop0", "pop1"],
        help="Population template to use when loading precomputed one-halo spectra. 'combined' uses the existing weighted combination, while 'pop0'/'pop1' select a single population and fall back to the previous z-bin when that population is missing.",
    )
    parser.add_argument(
        "--onehalo-fit-popmix",
        action="store_true",
        default=False,
        help="Fit an additional one-halo population-mix parameter f_pop using pop0/pop1 templates when available.",
    )

    # plot_components / plot_compare_cats specific
    parser.add_argument(
        "--lmax-components",
        type=int,
        default=30000,
        help="Fixed lMax used for plot_components",
    )
    parser.add_argument(
        "--lmax-compare",
        type=int,
        default=50000,
        help="Fixed lMax used for plot_compare_cats",
    )
    parser.add_argument(
        "--ell-eval-1h",
        type=float,
        default=10000.0,
        help="Multipole at which to evaluate the one-halo power in the 1h comparison figure",
    )

    parser.add_argument("--overwrite", action="store_true", help="Recompute fits even if output .npz exists")
    parser.add_argument("--mock-basepath", default=None,
                        help="Base directory for v3 boxed sim IGL predictions "
                             "(e.g. data/v3_boxed_outputs/tiles_10p0deg). "
                             "If set, IGL curves are overlaid on spectrum plots.")
    parser.add_argument("--bias-cache-fpath", default=None,
                        help="Path to effective_bias_ls_cache.npz from compute_effective_bias_ls.py. "
                             "When provided with --mock-basepath, overlays bias-scaled smooth IGL "
                             "prediction on the 2x2 spectrum panels.")
    parser.add_argument("--mock-a2h-cache", default=None,
                        help="Path to a2h_cache.json produced by the Jordan mock pipeline "
                             "(e.g. data/jordan_mocks/v2/a2h_cache.json or "
                             "data/jordan_mocks/v3_boxed_outputs/tiles_10p0deg/a2h_cache.json). "
                             "When provided, A_2h predictions are read directly from the cache "
                             "instead of loading individual .npz files.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_ALL_MODES = ["run_auto", "run_cross", "plot_auto", "plot_cross", "plot_components",
              "plot_compare_cats", "plot_fit_spectra", "plot_spectra_summary", "plot_fpop_vs_redshift", "plot_corner", "plot_corner_overlay",
              "plot_corr_a1h_a2h", "plot_sigma_damp", "plot_chi2_1h", "plot_chi2_2h",
              "plot_redshift_panels_2x2", "plot_a1h_vs_redshift", "plot_a1h_vs_redshift_three_row",
              "plot_a1h_vs_redshift_alternate_layout", "plot_a1h_vs_redshift_mag_comparison", "plot_a1h_model_pred_vs_redshift", "plot_a1h_band_ratio_vs_redshift",
              "plot_d_ell_1h_evolution", "plot_parameter_consistency_vs_lmax",
              "plot_r1h_ratio", "plot_ihl_and_dell_combined",
              "make_chi2_table", "make_amplitude_table", "param_priors_table"]


def main() -> None:
    args = parse_args()

    modes = set(args.mode)
    if "all" in modes:
        modes = set(_ALL_MODES)

    print(f"Modes: {sorted(modes)}")
    print(f"Catalogs: {args.cat}")
    print(f"lMax values: {args.lmax}")

    if "run_auto" in modes:
        _run_auto_fits(args)
    if "run_cross" in modes:
        _run_cross_fits(args)
    if "plot_auto" in modes:
        _plot_auto(args)
    if "plot_cross" in modes:
        _plot_cross(args)
    if "plot_components" in modes:
        _plot_components(args)
    if "plot_fit_spectra" in modes:
        _plot_fit_spectra(args)
    if "plot_spectra_summary" in modes:
        _plot_spectra_summary(args)
    if "plot_fpop_vs_redshift" in modes:
        _plot_fpop_vs_redshift(args)
    if "plot_corner" in modes:
        _plot_corner(args)
    if "plot_corner_overlay" in modes:
        _plot_corner_overlay(args)
    if "plot_compare_cats" in modes:
        _plot_compare_cats(args)
    if "plot_corr_a1h_a2h" in modes:
        _plot_corr_a1h_a2h(args)
    if "plot_sigma_damp" in modes:
        _plot_sigma_damp(args)
    if "plot_parameter_consistency_vs_lmax" in modes:
        _plot_parameter_consistency_vs_lmax(args)
    if "plot_chi2_1h" in modes:
        _chi2_comparison_with_without_1h(args)
    if "plot_chi2_2h" in modes:
        _chi2_comparison_with_without_2h(args)
    if "plot_redshift_panels_2x2" in modes:
        _plot_redshift_panels_2x2(args)
    if "plot_a1h_vs_redshift" in modes:
        _plot_a1h_vs_redshift(args)
    if "plot_a1h_vs_redshift_three_row" in modes:
        _plot_a1h_vs_redshift_three_row(args)
    if "plot_a1h_vs_redshift_alternate_layout" in modes:
        _plot_a1h_vs_redshift_alternate_layout(args)
    if "plot_a1h_model_pred_vs_redshift" in modes:
        _plot_a1h_model_pred_vs_redshift(args, ell_eval=args.ell_eval_1h)
    if "plot_a1h_band_ratio_vs_redshift" in modes:
        _plot_a1h_band_ratio_vs_redshift(args)
    if "plot_d_ell_1h_evolution" in modes:
        _plot_d_ell_1h_evolution(args)
    if "plot_r1h_ratio" in modes:
        _plot_r1h_ratio(args)
    if "plot_ihl_and_dell_combined" in modes:
        _plot_ihl_and_dell_combined(args)
    if "plot_a1h_vs_redshift_mag_comparison" in modes:
        _plot_a1h_vs_redshift_mag_comparison(args)
    if "plot_a2h_vs_redshift" in modes:
        _plot_a2h_vs_redshift(args)
    if "plot_di_dz_upper_limits" in modes:
        _plot_di_dz_upper_limits(args)
    if "make_chi2_table" in modes:
        _make_chi2_latex_table(args)
    if "make_amplitude_table" in modes:
        _make_amplitude_table(args)
    if "param_priors_table" in modes:
        _make_parameter_priors_table(args)


if __name__ == "__main__":
    main()
