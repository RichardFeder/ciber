#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path

# load ciber module from parent directory 

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import config
from ciber.core.ps_pipeline_go import ciber_gal_cross
from ciber.cross_correlation.intensity_recon_cross import (
    inspect_hsc_catalog_fields,
    preprocess_intensity_maps,
)


def _default_intensity_addstr(hsc_mag_column, zmin, zmax, tag="intensity_recon"):
    parts = [tag, hsc_mag_column]
    if zmin is not None:
        parts.append(f"zmin{zmin:.1f}")
    if zmax is not None:
        parts.append(f"zmax{zmax:.1f}")
    return "_".join(parts)


def _stack_if_finite(arr_list):
    valid = []
    for arr in arr_list:
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=float)
        if np.any(np.isfinite(arr)):
            valid.append(arr)
    if len(valid) == 0:
        return None
    return np.array(valid)


def _field_average(arr):
    arr = _stack_if_finite(arr)
    if arr is None:
        return None
    return np.nanmean(arr, axis=0)


def _compute_r_ell(cross_arr, auto_arr, ciber_auto_arr):
    cross = _field_average(cross_arr)
    auto = _field_average(auto_arr)
    ciber_auto = _field_average(ciber_auto_arr)
    if cross is None or auto is None or ciber_auto is None:
        return None

    denom = np.sqrt(np.abs(auto * ciber_auto))
    r_ell = np.full_like(cross, np.nan)
    mask = denom > 0
    r_ell[mask] = cross[mask] / denom[mask]
    return r_ell


def _plot_overlays(result_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for npz_path in result_files:

        #find TM pattern in npz_path:
        inst = npz_path.split("TM")[-1][0]
        print('INST = ', inst)
        dat = np.load(npz_path, allow_pickle=True)
        lb = np.asarray(dat["lb"], dtype=float)
        pf = lb / (2.0 * np.pi)

        cl_gal = _field_average(dat["all_cl_gal"])
        cl_cross_gal = _field_average(dat["all_cl_cross"])
        cl_cross_intensity = _field_average(dat["all_cl_intensity_cross"]) if "all_cl_intensity_cross" in dat else None
        cl_auto_intensity = _field_average(dat["all_cl_intensity_auto"]) if "all_cl_intensity_auto" in dat else None
        cl_auto_ciber = _field_average(dat["all_cl_ciber_auto_inplace"]) if "all_cl_ciber_auto_inplace" in dat else None

        dcl_gal = np.std(_stack_if_finite(dat["all_cl_gal"]), axis=0) if "all_cl_gal" in dat else None
        dcl_cross_gal = np.std(_stack_if_finite(dat["all_cl_cross"]), axis=0) if "all_cl_cross" in dat else None
        dcl_cross_intensity = np.std(_stack_if_finite(dat["all_cl_intensity_cross"]), axis=0) if "all_cl_intensity_cross" in dat else None
        dcl_auto_intensity = np.std(_stack_if_finite(dat["all_cl_intensity_auto"]), axis=0) if "all_cl_intensity_auto" in dat else None
       
        from ciber.theory.intensity_reconstruction_diagnostics import _load_f25b_auto_on_grid
        f25b_auto = _load_f25b_auto_on_grid(inst, lb)
        cl_f25b = f25b_auto["cl"]
        clerr_f25b = f25b_auto["clerr"]

        r_gal = _compute_r_ell(dat["all_cl_cross"], dat["all_cl_gal"], cl_f25b[None,:]) if cl_f25b is not None else None

        # if "all_rl_intensity_cross" in dat:
        #     r_int = _field_average(dat["all_rl_intensity_cross"])
        # else:
        r_int = _compute_r_ell(
            dat["all_cl_intensity_cross"] if "all_cl_intensity_cross" in dat else None,
            dat["all_cl_intensity_auto"] if "all_cl_intensity_auto" in dat else None,
            cl_f25b[None,:] if cl_f25b is not None else None,
        )

        tag = os.path.basename(npz_path).replace(".npz", "")

        fig, ax = plt.subplots(1, 2, figsize=(9, 4))

        s = 5
        if cl_gal is not None:
            ax[0].errorbar(lb, pf * cl_gal, yerr=pf * dcl_gal, marker="o", label="$C_{\\ell}^{\\rm g \\times g}$", color='darkgreen', markersize=3, capsize=2, alpha=0.7)
        if cl_auto_intensity is not None:
            ax[0].errorbar(lb, pf * cl_auto_intensity, yerr=pf * dcl_auto_intensity, marker="o", label="$C_{\\ell}^{\\rm \\hat{I}_{\\rm recon} \\times \\hat{I}_{\\rm recon}}$", color='saddlebrown', markersize=3, capsize=2, alpha=0.7)
        if cl_f25b is not None:
            ax[0].errorbar(lb, pf * cl_f25b, yerr=pf * clerr_f25b, marker="o", label="CIBER auto (F25B)", color='k', markersize=3, capsize=2, alpha=0.7)
        if cl_cross_gal is not None:
            ax[0].errorbar(lb, pf * cl_cross_gal, yerr=pf * dcl_cross_gal, marker="s", label="$C_{\\ell}^{\\rm I \\times g}$", color='C2', markersize=3, capsize=2, alpha=0.7)
        if cl_cross_intensity is not None:
            ax[0].errorbar(lb, pf * cl_cross_intensity, yerr=pf * dcl_cross_intensity, marker="s", label="$C_{\\ell}^{\\rm I \\times \\hat{I}_{\\rm recon}}$", color='C1', markersize=3, capsize=2, alpha=0.7)

        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_ylim(5e-6, 2e-0)
        ax[0].set_xlabel(r"$\ell$", fontsize=14)
        ax[0].set_ylabel(r"$\ell C_{\ell}/(2\pi)$", fontsize=14)
        ax[0].grid(alpha=0.3)
        ax[0].legend(fontsize=10, ncol=2, loc=2)

        if r_gal is not None:
            ax[1].plot(lb, r_gal, marker="s", linestyle="-", label=r"$r_\ell^{\rm I \times g}$", color='C2', markersize=5)
        if r_int is not None:
            ax[1].plot(lb, r_int, marker="s", linestyle="-", label=r"$r_\ell^{\rm I \times \hat{I}_{\rm recon}}$", color='C1', markersize=5)
        ax[1].set_xscale("log")
        ax[1].set_xlabel(r"$\ell$", fontsize=14)
        ax[1].set_ylabel(r"$r_\ell$", fontsize=14)
        ax[1].set_ylim(-0.2, 1.1)
        ax[1].grid(alpha=0.3)
        ax[1].legend(fontsize=12)

        fig.suptitle(tag, fontsize=11)
        out_png = os.path.join(output_dir, f"{tag}_comparison.png")
        # fig.tight_layout()
        fig.savefig(out_png, dpi=300)
        plt.close(fig)

        if r_gal is not None and r_int is not None:
            delta = r_int - r_gal
            finite = np.isfinite(delta)
            if np.any(finite):
                print(f"[{tag}] <Delta r_ell> = {np.nanmean(delta[finite]):.4f} over {np.sum(finite)} bins")


def main():
    parser = argparse.ArgumentParser(description="Run intensity-reconstruction x CIBER cross-correlation")
    parser.add_argument("--inst-list", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--ifield-list", type=int, nargs="+", default=[8])
    parser.add_argument("--catname", default="HSC", choices=["HSC", "LS"])
    parser.add_argument("--gal-addstr", default='hsc_ilt25.0', help="Existing galaxy-density addstr used by ciber_gal_cross")
    parser.add_argument("--intensity-addstr", default=None, help="Output addstr for intensity recon maps")
    parser.add_argument("--mag-column", default="z_cmodel_mag")
    parser.add_argument("--mag-select-column", default="i_cmodel_mag")
    parser.add_argument("--mag-min", type=float, default=18.0)
    parser.add_argument("--mag-max", type=float, default=25.0)
    parser.add_argument("--maskstr", default="JHlt16", help="Mask string for ciber_gal_cross")
    parser.add_argument("--zmin", type=float, default=0.0)
    parser.add_argument("--zmax", type=float, default=1.0)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output-dir", default=os.path.join(config.ciber_basepath, "figures/intensity_recon_cross"))
    args = parser.parse_args()

    if args.catname == "HSC":
        summary = inspect_hsc_catalog_fields(args.ifield_list, catname=args.catname)
        for ifield, info in summary.items():
            print(f"ifield={ifield}, n_rows={info['n_rows']}, mag_columns={info['mag_columns']}")

    if args.verify_only:
        return

    intensity_addstr = args.intensity_addstr
    if intensity_addstr is None:
        intensity_addstr = _default_intensity_addstr(args.mag_column, args.zmin, args.zmax)

    print(f"Using intensity addstr: {intensity_addstr}")

    for inst in args.inst_list:
        preprocess_intensity_maps(
            inst=inst,
            ifield_list=args.ifield_list,
            catname=args.catname,
            save=True,
            addstr=intensity_addstr,
            mag_column_recon=args.mag_column,
            mag_select_column=args.mag_select_column,
            mag_min=args.mag_min,
            mag_max=args.mag_max,
            zmin=args.zmin,
            zmax=args.zmax,
            show=False,
        )

    all_npz = []



    # all_ps_save_fpath = ciber_gal_cross(inst_list, ifield_list_use, catname, addstr=addstr, plot=False, \
    #                                    estimate_ciber_noise_gal=True, fc_sub=False, fc_sub_quad_offset=False, nsims=250, n_split=5, \
    #                                     compute_cl_theta=True, cl_theta_cut=True, n_rad_bins=8, per_quadrant=False, ell_min_wedge=500, \
    #                                    quadoff_grad=True, grad_sub=False, subtract_randoms=True, rad_offset=None, \
    #                                    gal_downgrade_fac=gal_downgrade_fac, apply_pixel_corr=False, maskstr=maskstr, \
    #                                    rand_downgrade_fac=4, save=True, masking_maglim_list=masking_maglim_list,
    #                                    include_ff_errors=True, randstr=randstr, mkk_maglim_list=[16.0, 16.0])



    for inst in args.inst_list:
        out = ciber_gal_cross(
            inst_list=[inst],
            ifield_list_use=args.ifield_list,
            catname=args.catname,
            addstr=args.gal_addstr,
            intensity_map_addstr=intensity_addstr,
            save=True,
            plot=args.plot,
            apply_pixel_corr=False,
            subtract_randoms=True,
            masking_maglim_list = [16.0],
            mkk_maglim_list=[16.0],
            quadoff_grad=True, grad_sub=False, maskstr=args.maskstr,
            compute_cl_theta=True, cl_theta_cut=True, n_rad_bins=8, per_quadrant=False,
            ell_min_wedge=500.,
        )

        if out is None:
            continue
        paths, _ = out
        for p in paths:
            npz_path = p + ".npz"
            if os.path.exists(npz_path):
                all_npz.append(npz_path)

    if len(all_npz) == 0:
        print("No output npz files were produced.")
        return

    _plot_overlays(all_npz, args.output_dir)
    print("Saved comparison plots to", args.output_dir)


if __name__ == "__main__":
    main()
