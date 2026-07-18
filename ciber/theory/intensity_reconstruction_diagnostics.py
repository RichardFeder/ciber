"""
Intensity reconstruction diagnostics with F25b CIBER auto reference.

This module loads precomputed intensity and galaxy cross products, computes
coherence against the processed F25b CIBER auto spectrum, and generates
comparison figures for auto-power prediction and coherence.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import config
from ciber.plotting.gal_plotting_fns import _load_ciber_auto_file
from ciber.io.ciber_data_utils import load_dglpred_regrid


def _candidate_npz_paths(catname, inst, gal_addstr):
    filename = f"ciber_gal_ps_TM{inst}_{catname}_{gal_addstr}.npz"
    base = Path(getattr(config, "ciber_basepath", "."))
    return [
        base / "data" / "input_recovered_ps" / "ciber_gal_cross" / catname / f"TM{inst}" / filename,
        Path("data") / "input_recovered_ps" / "ciber_gal_cross" / catname / f"TM{inst}" / filename,
    ]

def _load_cross_product_npz(catname, inst, gal_addstr):
    tried = []
    for path in _candidate_npz_paths(catname, inst, gal_addstr):
        tried.append(str(path))
        if path.exists():
            return np.load(path, allow_pickle=True), str(path)
    raise FileNotFoundError("Could not find cross-product NPZ. Tried: " + ", ".join(tried))


def _interp_with_clamp(x_src, y_src, x_tgt):
    x_src = np.asarray(x_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    x_tgt = np.asarray(x_tgt, dtype=float)

    out = np.full_like(x_tgt, np.nan, dtype=float)
    valid = np.isfinite(x_src) & np.isfinite(y_src)
    if np.sum(valid) < 2:
        return out

    xs = x_src[valid]
    ys = y_src[valid]
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    return np.interp(x_tgt, xs, ys, left=ys[0], right=ys[-1])


def _load_f25b_auto_on_grid(inst, lb_target):
    band = "J" if int(inst) == 1 else "H"
    f25b = _load_ciber_auto_file(band)

    lb_src = np.asarray(f25b["lb"], dtype=float)
    cl_src = np.asarray(f25b["fieldav_cl"], dtype=float)
    clerr_src = np.asarray(f25b["fieldav_clerr"], dtype=float)

    cl_interp = _interp_with_clamp(lb_src, cl_src, lb_target)
    clerr_interp = _interp_with_clamp(lb_src, clerr_src, lb_target)

    return {
        "lb": np.asarray(lb_target, dtype=float),
        "cl": cl_interp,
        "clerr": clerr_interp,
        "source_path": f25b.get("source_path", "unknown"),
    }


def _safe_divide(num, den):
    out = np.full_like(np.asarray(num, dtype=float), np.nan, dtype=float)
    n = np.asarray(num, dtype=float)
    d = np.asarray(den, dtype=float)
    mask = np.isfinite(n) & np.isfinite(d) & (d > 0.0)
    out[mask] = n[mask] / d[mask]
    return out


def _compute_r_ell_from_components(cl_cross, cl_auto_tracer, cl_auto_ciber):
    denom = np.sqrt(np.abs(cl_auto_tracer * cl_auto_ciber))
    return _safe_divide(cl_cross, denom)


def compute_intensity_recon_ciber_prediction(
    inst_list,
    ifield_list,
    catname,
    gal_addstr,
    intensity_addstr,
    hsc_mag_column="i_cmodel_mag",
    zmin=0.0,
    zmax=1.0,
    mask_tail_list=None,
    gal_addstr_compare=None,
    verbose=True,
    per_field_weights=None,
):
    """Load spectra products and compute F25b-referenced predictions/coherences."""

    del intensity_addstr, mask_tail_list, gal_addstr_compare

    all_results = {}
    ell_array = None

    for inst in inst_list:
        try:
            ps_data, ps_path = _load_cross_product_npz(catname, inst, gal_addstr)
        except FileNotFoundError as exc:
            if verbose:
                print(f"[IntensityReconDiag] TM{inst}: {exc}")
            continue

        print('PS PATH is ', ps_path)

        lb = np.asarray(ps_data["lb"], dtype=float)
        if ell_array is None:
            ell_array = lb

        cl_intensity_auto = np.asarray(ps_data.get("all_cl_intensity_auto"), dtype=float)
        cl_intensity_cross = np.asarray(ps_data.get("all_cl_intensity_cross"), dtype=float)
        cl_gal_auto = np.asarray(ps_data.get("all_cl_gal"), dtype=float) if "all_cl_gal" in ps_data.files else None
        cl_gal_cross = np.asarray(ps_data.get("all_cl_cross"), dtype=float) if "all_cl_cross" in ps_data.files else None
        clerr_intensity_auto = np.asarray(ps_data.get("all_clerr_intensity_auto"), dtype=float) if "all_clerr_intensity_auto" in ps_data.files else None
        clerr_intensity_cross = np.asarray(ps_data.get("all_clerr_intensity_cross"), dtype=float) if "all_clerr_intensity_cross" in ps_data.files else None


        def average_field_data(arr, name, weights=None, is_error=False):
            """Average per-field data (2D → 1D) with optional weighting."""
            if arr is None or arr.ndim != 2:
                return arr
            
            if verbose:
                print(f"[IntensityReconDiag] TM{inst}: field-averaging {name} (shape {arr.shape})")
            
            if is_error:
                # Error propagation
                result = np.sqrt(np.nansum(arr ** 2, axis=0)) / arr.shape[0]
            elif weights is not None:
                # Weighted average
                result = np.average(arr, axis=0, weights=weights)
            else:
                # Simple average
                result = np.nanmean(arr, axis=0)
            
            print(f'{name} has shape {result.shape}')
            return result
        
        # full_perf_weights = np.zeros((len(inst_list), len(zbinedges)-1, len(ifield_list), len(lb)))

        # Extract per-field weights (shape: n_zbin × n_fields × n_ell)
        weights_inst = None
        if per_field_weights is not None and per_field_weights.ndim >= 2:
            weights_inst = per_field_weights[inst-1][0]  # Take first z-bin weights

            print('weights inst has shape', weights_inst.shape)
            print('weights inst:', weights_inst)

        # Apply field-averaging to all arrays
        cl_intensity_auto = average_field_data(cl_intensity_auto, 'cl_intensity_auto', weights=weights_inst)
        cl_intensity_cross = average_field_data(cl_intensity_cross, 'cl_intensity_cross')
        clerr_intensity_auto = average_field_data(clerr_intensity_auto, 'clerr_intensity_auto', is_error=True)
        clerr_intensity_cross = average_field_data(clerr_intensity_cross, 'clerr_intensity_cross', is_error=True)

        if cl_intensity_auto.size == 0 or cl_intensity_cross.size == 0:
            if verbose:
                print(f"[IntensityReconDiag] TM{inst}: missing intensity spectra arrays")
            continue


        # if per_field_weights is not None and per_field_weights.shape[0] >= 1:
        #     # Use provided per-field weights
        #     weights = per_field_weights[0]  # Take first z-bin weights
        #     weighted_mean = np.average(cl_intensity_auto, axis=0, weights=weights)
        #     cl_intensity_auto = weighted_mean
        # else:
        #     # Default: simple field average
        #     cl_intensity_auto = np.nanmean(cl_intensity_auto, axis=0)


        # # Field-average intensity spectra if per-field (shape: n_fields × n_ell)
        # if cl_intensity_auto.ndim == 2:
        #     if verbose:
        #         print(f"[IntensityReconDiag] TM{inst}: field-averaging intensity auto (shape {cl_intensity_auto.shape})")
        #     cl_intensity_auto = np.nanmean(cl_intensity_auto, axis=0)

        #     print('cl_intensity_auto has shape', cl_intensity_auto.shape)
        # if cl_intensity_cross.ndim == 2:
        #     if verbose:
        #         print(f"[IntensityReconDiag] TM{inst}: field-averaging intensity cross (shape {cl_intensity_cross.shape})")
        #     cl_intensity_cross = np.nanmean(cl_intensity_cross, axis=0)
        #     print('cl_intensity_cross has shape', cl_intensity_cross.shape)
        # if clerr_intensity_auto is not None and clerr_intensity_auto.ndim == 2:
        #     if verbose:
        #         print(f"[IntensityReconDiag] TM{inst}: field-averaging intensity auto err (shape {clerr_intensity_auto.shape})")
        #     # For errors: combine via error propagation
        #     clerr_intensity_auto = np.sqrt(np.nansum(clerr_intensity_auto ** 2, axis=0)) / clerr_intensity_auto.shape[0]
        #     print('clerr_intensity_auto has shape', clerr_intensity_auto.shape)
        # if clerr_intensity_cross is not None and clerr_intensity_cross.ndim == 2:
        #     if verbose:
        #         print(f"[IntensityReconDiag] TM{inst}: field-averaging intensity cross err (shape {clerr_intensity_cross.shape})")
        #     clerr_intensity_cross = np.sqrt(np.nansum(clerr_intensity_cross ** 2, axis=0)) / clerr_intensity_cross.shape[0]
        #     print('clerr_intensity_cross has shape', clerr_intensity_cross.shape)

        # if cl_intensity_auto.size == 0 or cl_intensity_cross.size == 0:
        #     if verbose:
        #         print(f"[IntensityReconDiag] TM{inst}: missing intensity spectra arrays")
        #     continue

        f25b_auto = _load_f25b_auto_on_grid(inst, lb)
        cl_f25b = f25b_auto["cl"]
        clerr_f25b = f25b_auto["clerr"]

        cl_ciber_pred = _safe_divide(cl_intensity_cross ** 2, cl_intensity_auto)

        clerr_ciber_pred = _safe_divide(
            (cl_intensity_cross ** 2) * np.sqrt(
                (clerr_intensity_cross / cl_intensity_cross) ** 2
                + (clerr_intensity_auto / cl_intensity_auto) ** 2
            ),
            cl_intensity_auto,
        )

        print('cl_intensity_auto has shape', cl_intensity_auto.shape)
        print('cl_intensity_cross has shape', cl_intensity_cross.shape)
        print('cl_f25b has shape', cl_f25b.shape)
        rl_intensity_f25b = _compute_r_ell_from_components(
            cl_intensity_cross,
            cl_intensity_auto,
            cl_f25b,
        )

        if cl_gal_cross is not None and cl_gal_auto is not None:
            rl_gal_f25b = _compute_r_ell_from_components(cl_gal_cross, cl_gal_auto, cl_f25b)
        else:
            rl_gal_f25b = None

        all_results[inst] = {
            "lb": lb,
            "npz_source_path": ps_path,
            "cl_intensity_auto": cl_intensity_auto,
            "cl_intensity_cross": cl_intensity_cross,
            "clerr_intensity_auto": clerr_intensity_auto,
            "clerr_intensity_cross": clerr_intensity_cross,
            "cl_gal_auto": cl_gal_auto,
            "cl_gal_cross": cl_gal_cross,
            "cl_ciber_auto_f25b": cl_f25b,
            "clerr_ciber_auto_f25b": clerr_f25b,
            "cl_ciber_auto_pred": cl_ciber_pred,
            "clerr_ciber_auto_pred": clerr_ciber_pred,
            "rl_intensity_f25b": rl_intensity_f25b,
            "rl_gal_f25b": rl_gal_f25b,
            "f25b_source_path": f25b_auto["source_path"],
            "ifield_list": list(ifield_list),
        }

        if verbose:
            print(
                f"[IntensityReconDiag] TM{inst}: loaded {ps_path} | "
                f"F25b={f25b_auto['source_path']}"
            )

    metadata = {
        "catname": catname,
        "gal_addstr": gal_addstr,
        "hsc_mag_column": hsc_mag_column,
        "zmin": zmin,
        "zmax": zmax,
        "inst_list": list(inst_list),
        "ifield_list": list(ifield_list),
    }

    return {
        "all_results": all_results,
        "ell_array": ell_array,
        "metadata": metadata,
    }


# def _field_mean_and_err(arr):
#     arr = np.asarray(arr, dtype=float)
#     mean = np.nanmean(arr, axis=0)
#     err = np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])
#     return mean, err

def _field_mean_and_err(arr):
    """Compute field mean and error from per-field data.
    
    If arr is 1D (already field-averaged), return as-is with zero errors.
    If arr is 2D (per-field), compute mean along field axis and error as std/sqrt(n_fields).
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        # Already field-averaged
        mean = arr
        err = np.zeros_like(arr)
    else:
        # Per-field data: average over fields
        mean = np.nanmean(arr, axis=0)
        err = np.nanstd(arr, axis=0) / np.sqrt(max(arr.shape[0], 1))
    return mean, err


def _ell_mask(lb, ell_min, ell_max):
    lb = np.asarray(lb, dtype=float)
    return np.isfinite(lb) & (lb >= float(ell_min)) & (lb <= float(ell_max))


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

def plot_reconstruction_comparison(
    results,
    ell_min=280.0,
    ell_max=1.1e5,
    figsize=(7.8, 3.7),
    outdir=None,
    figname="intensity_reconstruction_auto_prediction_comparison.png",
    plot=True,
    verbose=True,
):
    """Main zlt1-style figure: measured F25b auto vs predicted auto from intensity."""

    if verbose:
        print("[PlotReconComparison] Creating zlt1-style auto prediction comparison...")

    all_results = results["all_results"]
    metadata = results["metadata"]
    inst_list = metadata["inst_list"]

    fig, axes = plt.subplots(1, len(inst_list), figsize=figsize, sharex=True, sharey=True)
    if len(inst_list) == 1:
        axes = [axes]

    titles = {1: "CIBER 1.1 $\\mu$m", 2: "CIBER 1.8 $\\mu$m"}

    for ax, inst in zip(axes, inst_list):
        if inst not in all_results:
            ax.text(0.5, 0.5, f"No data for TM{inst}", ha="center", va="center", transform=ax.transAxes)
            continue

        res = all_results[inst]
        lb = res["lb"]
        pf = lb * (lb + 1.0) / (2.0 * np.pi)
        m = _ell_mask(lb, ell_min, ell_max)


        dgl_path = f'/Volumes/richext/workmac/ciber/ciber1/data/fluctuation_data/TM{inst}/dgl_tracer_maps/sfd_clean/dgl_auto_constraints_TM{inst}_sfd_clean_010924.npz'
        dl_pred, dl_err = load_dglpred_regrid(dgl_path, lb)
        dgl_dl = np.asarray(dl_pred, dtype=float)
        dgl_dl_err = np.asarray(dl_err, dtype=float)

        dl_meas = pf * res["cl_ciber_auto_f25b"]
        dl_meas_err = pf * res["clerr_ciber_auto_f25b"]

        pred_mean_cl, pred_err_cl = _field_mean_and_err(res["cl_ciber_auto_pred"])
        dl_pred = pf * pred_mean_cl
        dl_pred_err = pf * pred_err_cl

        vm = m & np.isfinite(dl_meas) & np.isfinite(dl_meas_err)
        vp = m & np.isfinite(dl_pred)

        ax.errorbar(
            lb[vm],
            dl_meas[vm],
            yerr=dl_meas_err[vm],
            fmt="o",
            color="k",
            markersize=3,
            capsize=2.5,
            label="CIBER auto (F25b)",
            zorder=20,
        )

        ax.plot(
            lb[vp],
            dl_pred[vp],
            color="C1",
            linestyle=":",
            linewidth=2.2,
            label=f"Reconstructed IGL (HSC, z<{metadata['zmax']:.1f})",
        )

        x_line = np.logspace(np.log10(ell_min), np.log10(float(ell_max)), 256)

        if dgl_dl is not None and np.any(np.isfinite(dgl_dl)):
            y = np.asarray(dgl_dl, dtype=float)
            y_ext = _extend_series_loglog(lb, y, x_line)
            ax.plot(x_line, y_ext, color="k", linewidth=1.4, linestyle="-", label="DGL (F25b)")
            if dgl_dl_err is not None and np.any(np.isfinite(dgl_dl_err)):
                dy = np.asarray(dgl_dl_err, dtype=float)
                y_lo_ext = _extend_series_loglog(lb, np.clip(y - dy, 1e-12, None), x_line)
                y_hi_ext = _extend_series_loglog(lb, np.clip(y + dy, 1e-12, None), x_line)
                ax.fill_between(
                    x_line,
                    np.clip(y_lo_ext, 1e-12, None),
                    np.clip(y_hi_ext, 1e-12, None),
                    color="k",
                    alpha=0.12,
                )


        vband = vp & np.isfinite(dl_pred_err)
        if np.any(vband):
            lo = np.clip(dl_pred[vband] - dl_pred_err[vband], 1e-12, None)
            hi = np.clip(dl_pred[vband] + dl_pred_err[vband], 1e-12, None)
            ax.fill_between(lb[vband], lo, hi, color="C1", alpha=0.18, linewidth=0.0)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(ell_min, ell_max)
        ax.set_ylim(1e-2, 1e4)
        ax.grid(alpha=0.25)
        ax.set_title(titles.get(inst, f"TM{inst}"), fontsize=11)
        ax.set_xlabel(r"$\ell$")

    axes[0].set_ylabel(r"$D_\ell = \ell(\ell+1)C_\ell/(2\pi)$")
    axes[0].legend(fontsize=8, loc="best")

    fig.suptitle(
        f"Intensity Reconstruction Auto Prediction vs F25b ({metadata['hsc_mag_column']}, "
        f"{metadata['zmin']:.1f}<z<{metadata['zmax']:.1f})",
        fontsize=11,
    )
    plt.tight_layout()

    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        fig_path = outdir / figname
        fig.savefig(fig_path, dpi=160, bbox_inches="tight")
        if verbose:
            print(f"[PlotReconComparison] Saved to {fig_path}")

    if plot:
        plt.show()

    return fig


def plot_r_ell_comparison(
    results,
    ell_min=280.0,
    ell_max=1.1e5,
    figsize=(8, 4),
    outdir=None,
    figname="intensity_reconstruction_r_ell_comparison.png",
    plot=True,
    verbose=True,
):
    """Plot r_ell comparison (galaxy vs intensity), both computed with F25b auto."""

    if verbose:
        print("[PlotREll] Creating r_ell comparison figure...")

    all_results = results["all_results"]
    metadata = results["metadata"]
    inst_list = metadata["inst_list"]

    fig, axes = plt.subplots(1, len(inst_list), figsize=figsize, sharex=True, sharey=True)
    if len(inst_list) == 1:
        axes = [axes]

    titles = {1: f"CIBER 1.1 $\\mu$m $\\times$ HSC $i<25$, $z_{{\\rm phot}}<{metadata['zmax']:.1f}$", 2: f"CIBER 1.8 $\\mu$m $\\times$ HSC $i<25$, $z_{{\\rm phot}}<{metadata['zmax']:.1f}$"}

    for ax, inst in zip(axes, inst_list):
        if inst not in all_results:
            ax.text(0.5, 0.5, f"No data for TM{inst}", ha="center", va="center", transform=ax.transAxes)
            continue

        res = all_results[inst]
        lb = res["lb"]
        m = _ell_mask(lb, ell_min, ell_max)

        r_int_mean, r_int_err = _field_mean_and_err(res["rl_intensity_f25b"])
        vint = m & np.isfinite(r_int_mean)
        ax.errorbar(
            lb[vint],
            r_int_mean[vint],
            yerr=r_int_err[vint],
            fmt="o-",
            color="C1",
            linewidth=1.9,
            markersize=3.8,
            capsize=2,
            label='$r_{\\ell}^{\\hat{I} \\times I_{CIBER}}$',
        )

        if res["rl_gal_f25b"] is not None:
            r_gal_mean, r_gal_err = _field_mean_and_err(res["rl_gal_f25b"])
            vgal = m & np.isfinite(r_gal_mean)
            ax.errorbar(
                lb[vgal],
                r_gal_mean[vgal],
                yerr=r_gal_err[vgal],
                fmt="s-",
                color="C0",
                linewidth=1.8,
                markersize=3.5,
                capsize=2,
                label='$r_{\\ell}^{I \\times g}$',
            )

        ax.set_xscale("log")
        ax.set_xlim(ell_min, ell_max)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
        ax.set_title(titles.get(inst, f"TM{inst}"), fontsize=11)
        ax.set_xlabel(r"$\ell$", fontsize=14)

    axes[0].set_ylabel(r"$r_\ell$", fontsize=14)
    axes[0].legend(fontsize=10, loc="best")

    # fig.suptitle(
    #     f"Coherence Comparison Using F25b CIBER Auto ({metadata['hsc_mag_column']}, "
    #     f"{metadata['zmin']:.1f}<z<{metadata['zmax']:.1f})",
    #     fontsize=11,
    # )
    # plt.tight_layout()

    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        fig_path = outdir / figname
        fig.savefig(fig_path, dpi=160, bbox_inches="tight")
        if verbose:
            print(f"[PlotREll] Saved to {fig_path}")

    if plot:
        plt.show()

    return fig
