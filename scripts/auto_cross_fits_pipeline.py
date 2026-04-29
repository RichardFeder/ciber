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

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402  (sets ciber_basepath)
from ciber.theory.cross_ps_parametric_model import (
    run_gal_auto_fits_two_stage,
    run_gal_cross_fits,
    CrossPowerSpectrumModel,
)
from ciber.io.ciber_data_utils import load_fit_results_npz
from ciber.plotting.gal_plotting_fns import (
    plot_amplitude_comparison,
    plot_chi2_comparison,
    plot_cross_fit_components_from_file,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CAT_CMAP = {"HSC": "Oranges", "DESILS": "Blues"}
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


def _cross_fpath(datadir: str, cat: str, headstr: Optional[str], fitstr: str, lMax: int) -> Path:
    tag = _headstr_tag(headstr)
    return Path(datadir) / f"{cat}_coarsez{tag}_cross_cl_fits_{fitstr}_lMax={lMax}.npz"


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


def _run_cross_fits(args: argparse.Namespace) -> None:
    # With fix_ihl_1h_shape=True and a valid ihl_params file, mu_1h and sigma_1h
    # are fixed per-zbin from the precomputed IHL template fits, giving a
    # 3-parameter MCMC: [A_2h, A_1h, A_shot]. prior_bounds=None uses the
    # correct 3-param defaults built inside fit_model_mcmc.
    # If use_one_halo=False, fits only 2h+shot (no 1h term).
    for cat in args.cat:
        ifield_list = _ifield_list(cat, args)
        for lMax in args.lmax:
            fpath = _cross_fpath(args.datadir_cross, cat, args.headstr if cat == "HSC" else None, args.fitstr_cross, lMax)
            if not args.overwrite and fpath.exists():
                print(f"[run_cross] skipping {fpath.name} (already exists)")
                continue
            print(f"[run_cross] {cat} lMax={lMax}")
            run_gal_cross_fits(
                cat=cat,
                ifield_list=ifield_list,
                save_results=True,
                file_fpath=fpath.name,
                lMax_fit=lMax,
                use_ihl_templates=False,
                use_ihl_1h_params=True,
                fix_ihl_1h_shape=True,
                ihl_1h_params_path=args.ihl_params,
                fitstr=args.fitstr_cross,
                save_figs=True,
                use_astrometry_damping=args.use_damping,
                chi2_lim=[-5, 5],
                headstr=args.headstr if cat == "HSC" else None,
                use_one_halo=args.use_one_halo,
                prior_bounds=None,
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
            fpath = _cross_fpath(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax)
            if not fpath.exists():
                print(f"[plot_cross] missing {fpath}, skipping lMax={lMax} for {cat}")
                continue
            all_res.append((lMax, load_fit_results_npz(str(fpath))))

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
        fpath = _cross_fpath(args.datadir_cross, cat, headstr, args.fitstr_cross, args.lmax_components)
        if not fpath.exists():
            print(f"[plot_components] missing {fpath}, skipping {cat}")
            continue

        stem = figdir / f"{cat}_cross_components_{args.fitstr_cross}_lMax={args.lmax_components}"
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
            fpath = _cross_fpath(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax)
            if not fpath.exists():
                print(f"[plot_fit_spectra] missing {fpath}, skipping {cat} lMax={lMax}")
                continue

            results = load_fit_results_npz(str(fpath))
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
            else:  # HSC
                res_ps = collect_ciber_gal_vs_redshift(
                    catname, subtract_randoms=True, inst_list=inst_list,
                    zbinedges=zbinedges, maskstr=None, subtract_sn=False,
                    tl_pix_correct=True, ifield_list=ifield_list,
                    headstr=headstr, with_ff_err=True,
                )

            lb = res_ps['lb']
            full_cl_cross = res_ps['full_cl_cross']
            full_clerr_cross = res_ps['full_clerr_cross']

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
                    n_params   = int(np.sum(~np.isnan(params)))
                    params     = params[:n_params]
                    params_err = params_err[:n_params]

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
                    fit_result = {
                        "params":                params,
                        "params_err":            params_err,
                        "chisq":                 float(results["chisq"][inst_idx, zidx]),
                        "reduced_chisq":         float(results["reduced_chisq"][inst_idx, zidx]),
                        "ndof":                  len(lb_fit) - n_params,
                        "z_value":               zcen,
                        "use_single_slope":      None,
                        "one_halo_params_dict":  None,
                        "sigma_fixed":           None,
                        "use_astrometry_damping": use_damping,
                        "ihl_templates":         None,
                        "template_names":        None,
                    }

                    # Extract model configuration from results
                    use_powerlaw_2h = results.get("use_powerlaw_2h", True)
                    alpha_2h_fixed = results.get("alpha_2h_fixed", -1.5)

                    model = CrossPowerSpectrumModel(
                        lb=lb_fit, use_powerlaw_2h=use_powerlaw_2h,
                        alpha_2h_fixed=alpha_2h_fixed,
                        use_astrometry_damping=use_damping,
                    )

                    title = (f"CIBER {lams[inst]} μm × {cat}, "
                             f"z∈[{zlo:.1f},{zhi:.1f}], ℓ_max={lMax}")

                    fig, axes = plot_fit_fixed_1h_templates(
                        model, lb_fit, data_dl, data_dlerr, fit_result,
                        figsize=(6, 6), title=title, title_fs=13,
                        ylim=[1e-3, 5e2], lMax_fit=lMax,
                        chi2_lim=[-5, 5],
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


def _plot_spectra_summary(args: argparse.Namespace) -> None:
    """Two rows of 5 panels each (top: data + model, bottom: residuals).

    Each top panel shows data + model + components. Bottom panels show
    (data - model) / error. A single shared legend sits above top panels in five columns.
    """
    from ciber.theory.cross_ps_parametric_model import plot_fit_fixed_1h_templates
    from ciber.cross_correlation.galaxy_cross import collect_ciber_gal_vs_redshift

    figdir = Path(args.figdir) / args.fitstr_cross / "spectra"

    for cat in args.cat:
        headstr = args.headstr if cat == "HSC" else None
        for lMax in args.lmax:
            fpath = _cross_fpath(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax)
            if not fpath.exists():
                print(f"[plot_spectra_summary] missing {fpath}, skipping")
                continue

            results = load_fit_results_npz(str(fpath))
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
            else:
                res_ps = collect_ciber_gal_vs_redshift(
                    "HSC", subtract_randoms=True, inst_list=inst_list,
                    zbinedges=zbinedges, maskstr=None, subtract_sn=False,
                    tl_pix_correct=True, ifield_list=args.ifield_hsc,
                    headstr=headstr, with_ff_err=True,
                )

            lb = res_ps['lb']
            full_cl_cross = res_ps['full_cl_cross']
            full_clerr_cross = res_ps['full_clerr_cross']
            pf_data = lb * (lb + 1) / (2 * np.pi)
            startidx, endidx = 2, -1
            lb_fit = lb[startidx:endidx]

            use_powerlaw_2h = results.get("use_powerlaw_2h", True)
            alpha_2h_fixed = results.get("alpha_2h_fixed", 0.0)
            pnf_arr = results.get("param_names_fitted", None)

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

                legend_handles = None
                ylim = [1e-3, 5e2]
                mock_basepath = getattr(args, 'mock_basepath', None)

                for zidx in range(n_zbins):
                    ax_spec = spec_axes[zidx]
                    ax_res = res_axes[zidx]
                    zlo, zhi = zbinedges[zidx], zbinedges[zidx + 1]
                    zcen = 0.5 * (zlo + zhi)

                    data_dl    = (pf_data * full_cl_cross[inst_idx, zidx])[startidx:endidx]
                    data_dlerr = (pf_data * full_clerr_cross[inst_idx, zidx])[startidx:endidx]

                    params     = results["params"][inst_idx, zidx, :]
                    params_16  = results.get("params_16",  results["params"] - results["params_err"])[inst_idx, zidx, :]
                    params_84  = results.get("params_84",  results["params"] + results["params_err"])[inst_idx, zidx, :]
                    n_params   = int(np.sum(~np.isnan(params)))
                    params     = params[:n_params]
                    params_16  = params_16[:n_params]
                    params_84  = params_84[:n_params]

                    pnf_bin = pnf_arr[inst_idx, zidx] if pnf_arr is not None else None
                    use_damping = (pnf_bin is not None and
                                   any("damp" in str(p).lower() for p in pnf_bin))

                    model = CrossPowerSpectrumModel(
                        lb=lb_fit, use_powerlaw_2h=use_powerlaw_2h,
                        alpha_2h_fixed=alpha_2h_fixed,
                        use_astrometry_damping=use_damping,
                    )

                    # Top panel: spectra
                    ax_spec.errorbar(lb_fit, data_dl, yerr=data_dlerr, fmt='o',
                                     color='k', markersize=3, capsize=2, label='Data', zorder=5)

                    ell_m = np.logspace(np.log10(lb_fit.min()), np.log10(lb_fit.max()), 200)
                    sd_med = params[5] if use_damping else None
                    sd_lo  = params_16[5] if use_damping else None
                    sd_hi  = params_84[5] if use_damping else None

                    comps     = model.model_components(ell_m, *params[:5],    sigma_damp=sd_med)
                    comps_lo  = model.model_components(ell_m, *params_16[:5], sigma_damp=sd_lo)
                    comps_hi  = model.model_components(ell_m, *params_84[:5], sigma_damp=sd_hi)

                    ax_spec.plot(ell_m, comps['total'], 'r-', lw=2, label='Total')
                    ax_spec.fill_between(ell_m, comps_lo['total'], comps_hi['total'],
                                         color='red', alpha=0.15)
                    ax_spec.plot(ell_m, comps['two_halo'], 'b-', lw=1.5, alpha=0.7, label='2-halo')
                    ax_spec.fill_between(ell_m, comps_lo['two_halo'], comps_hi['two_halo'],
                                         color='blue', alpha=0.12)
                    ax_spec.plot(ell_m, comps['one_halo'], 'g-', lw=1.5, alpha=0.7, label='1-halo')
                    ax_spec.fill_between(ell_m, comps_lo['one_halo'], comps_hi['one_halo'],
                                         color='green', alpha=0.12)
                    ax_spec.plot(ell_m, comps['shot_noise'], 'm--', lw=1.5, alpha=0.7, label='Shot noise')
                    ax_spec.fill_between(ell_m, comps_lo['shot_noise'], comps_hi['shot_noise'],
                                         color='magenta', alpha=0.12)

                    # IGL overlay
                    igl_path = _igl_pred_path(mock_basepath, cat, inst, zlo, zhi)
                    if igl_path is not None:
                        pred = np.load(str(igl_path), allow_pickle=True)
                        if "lb" in pred and "cross" in pred:
                            lb_m2 = np.asarray(pred["lb"])[2:]
                            cl_m2 = np.asarray(pred["cross"])[2:]
                            pf_m2 = lb_m2 * (lb_m2 + 1) / (2 * np.pi)
                            ax_spec.plot(lb_m2, pf_m2 * cl_m2, 'k:', lw=1.5,
                                         label='IGL (v3 sim)', alpha=0.8)

                    ax_spec.set_xscale('log')
                    ax_spec.set_yscale('log')
                    ax_spec.set_xlim([lb_fit.min() * 0.8, lb_fit.max() * 1.2])
                    ax_spec.set_ylim(ylim)
                    ax_spec.grid(True, alpha=0.3, which='major')
                    ax_spec.axvspan(lMax, lb_fit.max() * 1.2, color='lightgray', alpha=0.3, zorder=0)
                    ax_spec.set_xticklabels([])

                    # Panel label in upper-left
                    chi2_str = f"χ²/dof={results['reduced_chisq'][inst_idx, zidx]:.2f}"
                    # ax_spec.text(0.04, 0.97, f"z∈[{zlo:.1f},{zhi:.1f}]\n{chi2_str}",
                                #  transform=ax_spec.transAxes, fontsize=9, va='top', ha='left')
                    
                    plotstr = 'CIBER '+str(lams[inst])+' $\\mu$m $\\times$ '+cat+'\nz∈['+str(zlo)+','+str(zhi)+']\n'+chi2_str
                    # f"CIBER z∈[{zlo:.1f},{zhi:.1f}]\n{chi2_str}"
                    ax_spec.text(0.04, 0.97, plotstr,
                                 transform=ax_spec.transAxes, fontsize=9, va='top', ha='left')


                    if zidx == 0:
                        ax_spec.set_ylabel(r'$D_\ell$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=12)
                        ax_spec.tick_params(axis='y', labelleft=True)
                    else:
                        ax_spec.tick_params(axis='y', labelleft=False)
                    ax_spec.set_xticks([1e3, 1e4, 1e5])
                    # ax_spec.set_xticklabels(['', '', ''])
                    ax_spec.tick_params(axis='x', labelbottom=False)

                    if legend_handles is None:
                        legend_handles, legend_labels = ax_spec.get_legend_handles_labels()

                    # Bottom panel: residuals
                    model_at_data = model.model_components(lb_fit, *params[:5], sigma_damp=sd_med)['total']
                    residuals = (data_dl - model_at_data) / data_dlerr
                    ax_res.plot(lb_fit, residuals, 'o', color='k', markersize=3, zorder=5)
                    ax_res.axhline(0, color='r', linestyle='-', lw=1.5, alpha=0.7)
                    # ax_res.axhline(1, color='gray', linestyle='--', lw=0.8, alpha=0.5)
                    # ax_res.axhline(-1, color='gray', linestyle='--', lw=0.8, alpha=0.5)
                    # ax_res.fill_between([lb_fit.min() * 0.8, lb_fit.max() * 1.2], -1, 1,
                    #                     color='green', alpha=0.1, zorder=0)
                    
                    # ax_res.fill_between([lb_fit.min() * 0.8, lb_fit.max() * 1.2], -3, 3,
                                        # color='lightgreen', alpha=0.1, zorder=0)

                    ax_res.axhspan(-1, 1, color='green', alpha=0.1)
                    ax_res.axhspan(-3, 3, color='yellow', alpha=0.1)
                    

                    ax_res.set_xscale('log')
                    ax_res.set_xlim([lb_fit.min() * 0.8, lb_fit.max() * 1.2])
                    ax_res.set_ylim([-5, 5])
                    ax_res.grid(True, alpha=0.3, which='major')
                    ax_res.axvspan(lMax, lb_fit.max() * 1.2, color='lightgray', alpha=0.3, zorder=0)

                    if zidx == 0:
                        # ax_res.set_ylabel(r'(Data - Model) / $\sigma$', fontsize=10)
                        ax_res.set_ylabel(r'$\chi$', fontsize=12)

                        ax_res.tick_params(axis='both', labelleft=True, labelbottom=True)
                    else:
                        ax_res.tick_params(axis='y', labelleft=False)
                    ax_res.set_xlabel(r'$\ell$', fontsize=12)

                # Single legend above top panels in five columns
                if legend_handles is not None:
                    fig.legend(legend_handles, legend_labels,
                               loc='upper center', ncol=5,
                               fontsize=14, frameon=True,
                               bbox_to_anchor=(0.5, 1.02))

                stem = figdir / f"{cat}_TM{inst}_lMax={lMax}_summary"
                _savefig(fig, stem, args.fig_fmt)
                plt.close(fig)


def _plot_corner(args: argparse.Namespace) -> None:
    """Plot corner plots of MCMC posteriors for each redshift bin and instrument."""
    figdir = Path(args.figdir) / args.fitstr_cross / "corners"

    for cat in args.cat:
        headstr = args.headstr if cat == "HSC" else None
        for lMax in args.lmax:
            fpath = _cross_fpath(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax)
            if not fpath.exists():
                print(f"[plot_corner] missing {fpath}, skipping {cat} lMax={lMax}")
                continue

            results = load_fit_results_npz(str(fpath))
            zbinedges = results["zbinedges"]
            inst_list = list(results["inst_list"])
            param_names = results["param_names"]
            samples_array = results.get("samples", None)

            if samples_array is None:
                print(f"[plot_corner] no MCMC samples in {fpath.name}, skipping {cat}")
                continue

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

                    title = f"{cat} × CIBER TM{inst}, z∈[{zlo:.1f},{zhi:.1f}], ℓ_max={lMax}"
                    fig = CrossPowerSpectrumModel.plot_mcmc_corner(
                        fit_result, title=title, figsize=(5, 5),
                        save_path=None,
                    )
                    stem = figdir / f"{cat}_TM{inst}_z{zidx:02d}_lMax={lMax}"
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
        fpath = _cross_fpath(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax)
        if not fpath.exists():
            print(f"[plot_compare_cats] missing {fpath}, skipping {cat}")
            continue
        cat_results[cat] = load_fit_results_npz(str(fpath))

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
    shade_colors = ("#e8f4ff", "#fff3e6")
    xticks       = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    n_cats = len(cat_results)
    x_offset_scale = 0.025

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
        fpath = _cross_fpath(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax)
        if not fpath.exists():
            print(f"[plot_corr_a1h_a2h] missing {fpath}, skipping {cat}")
            continue

        results = load_fit_results_npz(str(fpath))
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
                    corr_matrix = np.corrcoef(a2h_samples, a1h_samples)
                    corr = corr_matrix[0, 1]

                all_data.append({
                    'cat': cat,
                    'inst': inst,
                    'z_lo': zlo,
                    'z_hi': zhi,
                    'corr': corr
                })

    if not all_data:
        print("[plot_corr_a1h_a2h] no data found for any catalog, skipping")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(4, 3))

    colors = {'HSC': {'TM1': 'C0', 'TM2': 'C1'}, 'DESILS': {'TM1': 'C2', 'TM2': 'C3'}}
    linestyles = {'HSC': '-', 'DESILS': '--'}

    for cat in args.cat:
        if cat not in [d['cat'] for d in all_data]:
            continue
        for inst in [1, 2]:
            # Extract z-midpoints and correlations for this cat/inst combination
            z_mids = []
            corrs = []
            for row in all_data:
                if row['cat'] == cat and row['inst'] == inst:
                    z_mids.append(0.5 * (row['z_lo'] + row['z_hi']))
                    corrs.append(row['corr'])

            if corrs:
                label = f"{cat} TM{inst}"
                ax.plot(z_mids, corrs, 'o', color=colors[cat][f'TM{inst}'],
                        linestyle=linestyles[cat], linewidth=2, markersize=6, label=label)
                ax.plot(z_mids, corrs, color=colors[cat][f'TM{inst}'],
                        linestyle=linestyles[cat], linewidth=2)

    ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Redshift (bin center)', fontsize=10)
    ax.set_ylabel(r'$r(A_{2h}, A_{1h})$', fontsize=10)
    ax.set_title(f'{args.fitstr_cross}: Correlation Coefficient', fontsize=11)
    ax.grid(True, alpha=0.3, which='major')
    ax.legend(loc='best', fontsize=9)
    ax.set_ylim([-1.0, 1.0])
    ax.set_xlim([0.0, 1.0])

    fig.tight_layout()
    _savefig(fig, figdir / f"corr_a1h_a2h_{args.fitstr_cross}_lMax={lMax}", args.fig_fmt)
    plt.close(fig)


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
            fpath = _cross_fpath(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax)
            if not fpath.exists():
                print(f"[plot_sigma_damp] missing {fpath}, skipping {cat} lMax={lMax}")
                continue
            cat_results[cat][lMax] = load_fit_results_npz(str(fpath))

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
    
    Each panel shows data + fitted components with uncertainty bands.
    Shared x/y axes and a single legend positioned above all panels.
    """
    from ciber.theory.cross_ps_parametric_model import (
        CrossPowerSpectrumModel, load_fit_results_npz, load_ihl_template_for_zbin
    )
    
    figdir = Path(args.figdir) / args.fitstr_cross
    
    # Load results for both catalogs at fiducial lMax
    lMax = args.lmax_components
    
    results = {}
    for cat in ["DESILS", "HSC"]:
        headstr = args.headstr if cat == "HSC" else None
        fpath = _cross_fpath(args.datadir_cross, cat, headstr, args.fitstr_cross, lMax)
        if not fpath.exists():
            print(f"[plot_redshift_panels_2x2] missing {fpath}, skipping {cat}")
            continue
        results[cat] = load_fit_results_npz(str(fpath))
    
    if len(results) < 2:
        print("[plot_redshift_panels_2x2] need both DESILS and HSC, skipping")
        return
    
    # Extract info
    desils_results = results["DESILS"]
    hsc_results = results["HSC"]
    
    n_zbin = desils_results['params'].shape[1]
    zbinedges = desils_results['zbinedges']
    ell_model = np.logspace(np.log10(100), np.log10(100000), 500)
    
    # Common plotting settings
    colors_components = {
        'two_halo': 'C0',
        'one_halo': 'C1',
        'shot_noise': 'C2',
        'total': 'k'
    }
    
    # For each redshift bin, create 2×2 panel figure
    for z_idx in range(n_zbin):
        z_low, z_high = zbinedges[z_idx], zbinedges[z_idx + 1]
        z_label = f"z = {z_low:.1f}–{z_high:.1f}"
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
        
        # Title with redshift bin
        fig.suptitle(f"Power spectrum fits: {z_label}", fontsize=14, y=0.98)
        
        # Top-left: DESILS TM1
        ax = axes[0, 0]
        _plot_cross_spectrum_panel(
            ax, desils_results, 0, z_idx, ell_model, None,
            colors_components, title="DESILS × CIBER 1.1 μm"
        )
        
        # Top-right: DESILS TM2
        ax = axes[0, 1]
        _plot_cross_spectrum_panel(
            ax, desils_results, 1, z_idx, ell_model, None,
            colors_components, title="DESILS × CIBER 1.8 μm"
        )
        
        # Bottom-left: HSC TM1
        ax = axes[1, 0]
        _plot_cross_spectrum_panel(
            ax, hsc_results, 0, z_idx, ell_model, None,
            colors_components, title="HSC × CIBER 1.1 μm"
        )
        
        # Bottom-right: HSC TM2
        ax = axes[1, 1]
        _plot_cross_spectrum_panel(
            ax, hsc_results, 1, z_idx, ell_model, None,
            colors_components, title="HSC × CIBER 1.8 μm"
        )
        
        # Add shared legend above all panels
        handles = [
            plt.Line2D([0], [0], color=colors_components['total'], linewidth=2, label='Total model'),
            plt.Line2D([0], [0], color=colors_components['two_halo'], linewidth=2, label='2-halo'),
            plt.Line2D([0], [0], color=colors_components['one_halo'], linewidth=2, label='1-halo'),
            plt.Line2D([0], [0], color=colors_components['shot_noise'], linewidth=2, label='Shot noise'),
            plt.Line2D([0], [0], color='gray', linewidth=2, label='Data'),
        ]
        fig.legend(
            handles=handles,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.00),
            ncol=5,
            fontsize=11,
            frameon=True,
        )
        
        # Common labels
        axes[1, 0].set_xlabel(r"$\ell$", fontsize=12)
        axes[1, 1].set_xlabel(r"$\ell$", fontsize=12)
        axes[0, 0].set_ylabel(r"$D_\ell$ [μK$^2$]", fontsize=12)
        axes[1, 0].set_ylabel(r"$D_\ell$ [μK$^2$]", fontsize=12)
        
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        stem = figdir / f"cross_spectrum_fit_2x2_zbin_{z_idx}_z{z_low:.1f}-{z_high:.1f}_{args.fitstr_cross}"
        _savefig(fig, stem, args.fig_fmt)
        plt.close(fig)


def _plot_cross_spectrum_panel(ax, results, inst_idx, z_idx, ell_model, args, colors_components, title=""):
    """Plot a single cross-spectrum panel with data and fitted components."""
    
    # Extract data
    data_dl = results['data_dl'][inst_idx, z_idx]
    data_dlerr = results['data_dlerr'][inst_idx, z_idx]
    lb_fit = results['lb_fit'][inst_idx, z_idx]
    model_dl = results['model_dl'][inst_idx, z_idx]
    
    # model_dl might be on a different ell grid than lb_fit; interpolate if needed
    if len(model_dl) != len(lb_fit):
        # Assume model_dl is on a finer grid - interpolate back to lb_fit
        model_dl_interp = np.interp(lb_fit, np.logspace(np.log10(lb_fit[0]), np.log10(lb_fit[-1]), len(model_dl)), model_dl)
    else:
        model_dl_interp = model_dl
    
    # For IHL templates, just plot the pre-computed model
    # The model_dl is already computed with all components
    ax.loglog(lb_fit, model_dl_interp, color=colors_components['total'], linewidth=2.5, zorder=10, label='Total model')
    
    # Plot data with error bars
    ax.errorbar(lb_fit, data_dl, yerr=data_dlerr, fmt='o', color='gray', markersize=5, 
                elinewidth=1.5, capsize=2, alpha=0.8, zorder=5, label='Data')
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3, linestyle=':')
    ax.set_xscale('log')
    ax.set_yscale('log')


def _plot_chi2_comparison_with_without_1h(args: argparse.Namespace) -> None:
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
                 "plot_corner", "plot_corr_a1h_a2h", "plot_sigma_damp", "plot_chi2_1h",
                 "plot_redshift_panels_2x2", "all"],
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
        default=[20000, 30000, 50000, 70000, 90000],
        help="Multipole maximum values to sweep over",
    )

    # Fit labels
    parser.add_argument("--fitstr-auto", default="two_stage_fixed_1h", help="Fit label for auto fits")
    parser.add_argument("--fitstr-cross", default="no1h_thetacut", help="Fit label for cross fits")

    # Field / catalog settings
    parser.add_argument("--ifield-hsc", type=int, nargs="+", default=[8], help="ifield list for HSC")
    parser.add_argument("--ifield-ls", type=int, nargs="+", default=[4, 5, 6, 7, 8], help="ifield list for DESI-LS")
    parser.add_argument("--headstr", default="hsc_ilt25.0", help="Header string (magnitude limit tag) for HSC")
    parser.add_argument(
        "--zbinedges",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        help="Redshift bin edges",
    )
    parser.add_argument("--fmask", type=float, default=0.7, help="Mask fraction")

    # MCMC settings
    parser.add_argument("--nwalkers", type=int, default=32, help="MCMC walkers")
    parser.add_argument("--nsteps1", type=int, default=2000, help="MCMC steps stage 1")
    parser.add_argument("--nsteps2", type=int, default=4000, help="MCMC steps stage 2")
    parser.add_argument("--nburn1", type=int, default=500, help="Burn-in stage 1")
    parser.add_argument("--nburn2", type=int, default=1000, help="Burn-in stage 2")
    parser.add_argument("--no-damping", action="store_false", dest="use_damping", default=True,
                        help="Disable astrometric damping term in cross fits (default: enabled)")
    parser.add_argument("--no-one-halo", action="store_false", dest="use_one_halo", default=True,
                        help="Disable one-halo component in cross fits (default: enabled)")

    # Paths
    parser.add_argument("--figdir", default="figures/", help="Output figure directory")
    parser.add_argument("--fig-fmt", choices=["pdf", "png"], default="pdf",
                        help="Figure format: pdf (default) or png (dpi=300)")
    parser.add_argument("--datadir-auto", default="data/gal_auto_fits/", help="Directory for auto fit .npz files")
    parser.add_argument("--datadir-cross", default="data/cross_cl_fits/", help="Directory for cross fit .npz files")
    parser.add_argument(
        "--ihl-params",
        default="data/ihl_templates/ihl_1h_param_fit_v0.npz",
        help="Path to IHL 1h parameter file",
    )

    # plot_components / plot_compare_cats specific
    parser.add_argument(
        "--lmax-components",
        type=int,
        default=50000,
        help="Fixed lMax used for plot_components",
    )
    parser.add_argument(
        "--lmax-compare",
        type=int,
        default=50000,
        help="Fixed lMax used for plot_compare_cats",
    )

    parser.add_argument("--overwrite", action="store_true", help="Recompute fits even if output .npz exists")
    parser.add_argument("--mock-basepath", default=None,
                        help="Base directory for v3 boxed sim IGL predictions "
                             "(e.g. data/v3_boxed_outputs/tiles_10p0deg). "
                             "If set, IGL curves are overlaid on spectrum plots.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_ALL_MODES = ["run_auto", "run_cross", "plot_auto", "plot_cross", "plot_components",
              "plot_compare_cats", "plot_fit_spectra", "plot_spectra_summary", "plot_corner",
              "plot_corr_a1h_a2h", "plot_sigma_damp", "plot_chi2_1h", "plot_redshift_panels_2x2"]


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
    if "plot_corner" in modes:
        _plot_corner(args)
    if "plot_compare_cats" in modes:
        _plot_compare_cats(args)
    if "plot_corr_a1h_a2h" in modes:
        _plot_corr_a1h_a2h(args)
    if "plot_sigma_damp" in modes:
        _plot_sigma_damp(args)
    if "plot_chi2_1h" in modes:
        _plot_chi2_comparison_with_without_1h(args)
    if "plot_redshift_panels_2x2" in modes:
        _plot_redshift_panels_2x2(args)


if __name__ == "__main__":
    main()
