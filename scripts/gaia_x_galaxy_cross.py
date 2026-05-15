#!/usr/bin/env python3

import argparse
import os
from datetime import datetime
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from ciber.core.powerspec_pipeline import CIBER_PS_pipeline, make_theta_masks
from ciber.core.powerspec_utils import get_power_spec
from ciber.core.ps_pipeline_go import (
    return_default_cbps_dicts,
    set_up_filepaths_cbps,
    process_ciber_maps_perquad,
)
# from ciber.masking.mask_utils import iter_sigma_clip_mask
from ciber.core.powerspec_pipeline import precomp_filter_general, apply_filter_to_map_precomp
from ciber.io.ciber_data_utils import load_delta_g_maps


DEFAULT_INST = 1
DEFAULT_IFIELDS = [4, 5, 6, 7, 8]
DEFAULT_MASKING_MAGLIM = 17.5
DEFAULT_CLIP_SIGMA = 5
DEFAULT_NITERMAX = 5
DEFAULT_N_RAD_BINS = 8
DEFAULT_ELL_MIN_WEDGE = 2000
DEFAULT_COMPUTE_CL_THETA = True
DEFAULT_CL_THETA_CUT = True
DEFAULT_THETA0 = np.pi

DEFAULT_GAIA_ADDSTR = "stars_glt20p5"
DEFAULT_TRACERS = ["DESILS", "HSC", "WISE"]
DEFAULT_TRACER_ADDSTR = {
    "DESILS": None,
    "HSC": "hsc_glt24.0",
    "WISE": "unWISE_neo8",
}


def _resolve_catname(catname):
    if catname.lower() == "unwise":
        return "WISE"
    if catname.lower() == "gaia":
        return "gaia"
    return catname


def _default_mask_tail(inst, masking_maglim):
    bandstr_dict = {1: "J", 2: "H"}
    bandstr = bandstr_dict[inst]
    return f"maglim_{bandstr}_Vega_{masking_maglim}_111323_ukdebias"


def _load_masks(cbps, inst, ifield_list_full, mask_tail, catname, clip_sigma, nitermax):
    _, _, _, fpath_dict = return_default_cbps_dicts()
    fpath_dict, _, _, _ = set_up_filepaths_cbps(
        fpath_dict,
        inst,
        "test",
        "112022",
        datestr_trilegal="112022",
        data_type="observed",
        save_fpaths=True,
    )

    ciber_maps = np.zeros((len(ifield_list_full), cbps.dimx, cbps.dimy))
    masks = np.zeros_like(ciber_maps)
    dc_template = cbps.load_dark_current_template(inst, verbose=True, inplace=False)

    for fieldidx, ifield in enumerate(ifield_list_full):
        flight_im = cbps.load_flight_image(ifield, inst, inplace=False)
        flight_im *= cbps.cal_facs[inst]
        flight_im -= dc_template * cbps.cal_facs[inst]

        mask_fpath = (
            fpath_dict["mask_base_path"]
            + "/"
            + mask_tail
            + "/joint_mask_ifield"
            + str(ifield)
            + "_inst"
            + str(inst)
            + "_observed_"
            + mask_tail
            + ".fits"
        )
        mask = fits.open(mask_fpath)[1].data

        if catname == "HSC":
            if inst == 1:
                mask[-120:, -250:] = 0.0
                mask[:120, -250:] = 0.0
            elif inst == 2:
                mask[:250, :120] = 0.0
                mask[:250, -120:] = 0.0

        # sigclip = iter_sigma_clip_mask(
        #     flight_im, sig=clip_sigma, nitermax=nitermax, mask=mask
        # )
        # mask *= sigclip

        ciber_maps[fieldidx] = flight_im
        masks[fieldidx] = mask

    _, _, masks = process_ciber_maps_perquad(
        cbps,
        ifield_list_full,
        inst,
        ciber_maps,
        masks,
        clip_sigma=clip_sigma,
        nitermax=nitermax,
    )

    return masks


def _build_overdensity_map(count_map, mask, cbps, fc_sub_n_terms=2):
    masked = count_map.astype(float) * mask
    mean = np.mean(masked[mask != 0])
    masked[mask != 0] -= mean
    masked[mask != 0] /= mean
    masked *= mask

    dot1, X, mask_rav = precomp_filter_general(
        cbps.dimx,
        cbps.dimy,
        mask=mask,
        gradient_filter=True,
        quadoff_grad=False,
        fc_sub=False,
        fc_sub_quad_offset=False,
        fc_sub_n_terms=fc_sub_n_terms,
        fc_sub_with_gradient=False,
    )
    _, filter_comp = apply_filter_to_map_precomp(masked, dot1, X, mask_rav=mask_rav)
    masked -= filter_comp
    masked *= mask
    return masked


def _compute_cross_for_field(
    cbps,
    inst,
    ifield,
    mask,
    gaia_counts,
    tracer_counts,
    inv_mkk,
    weights,
    apply_pixel_corr=True,
):
    gaia_map = gaia_counts[f"ifield{ifield}"].data.transpose()
    tracer_map = tracer_counts[f"ifield{ifield}"].data.transpose()

    gaia_delta = _build_overdensity_map(gaia_map, mask, cbps)
    tracer_delta = _build_overdensity_map(tracer_map, mask, cbps)

    lb, cl_cross, clerr_cross = get_power_spec(
        gaia_delta,
        map_b=tracer_delta,
        lbinedges=cbps.Mkk_obj.binl,
        lbins=cbps.Mkk_obj.midbin_ell,
        weights=weights,
    )
    lb, cl_gaia, clerr_gaia = get_power_spec(
        gaia_delta,
        lbinedges=cbps.Mkk_obj.binl,
        lbins=cbps.Mkk_obj.midbin_ell,
        weights=weights,
    )
    lb, cl_tracer, clerr_tracer = get_power_spec(
        tracer_delta,
        lbinedges=cbps.Mkk_obj.binl,
        lbins=cbps.Mkk_obj.midbin_ell,
        weights=weights,
    )

    cl_cross = np.dot(inv_mkk.transpose(), cl_cross)
    clerr_cross = np.dot(inv_mkk.transpose(), clerr_cross)
    cl_gaia = np.dot(inv_mkk.transpose(), cl_gaia)
    clerr_gaia = np.dot(inv_mkk.transpose(), clerr_gaia)
    cl_tracer = np.dot(inv_mkk.transpose(), cl_tracer)
    clerr_tracer = np.dot(inv_mkk.transpose(), clerr_tracer)

    if apply_pixel_corr:
        pix_res_arcsec = 7.0
        wp_ell = get_pixel_window_function(lb, pix_res_arcsec)
        cl_cross /= np.sqrt(wp_ell)
        clerr_cross /= np.sqrt(wp_ell)
        cl_gaia /= wp_ell
        clerr_gaia /= wp_ell
        cl_tracer /= wp_ell
        clerr_tracer /= wp_ell

    return lb, cl_cross, clerr_cross, cl_gaia, clerr_gaia, cl_tracer, clerr_tracer


def _plot_cross(lb, cl_cross, clerr_cross, ifield_list, title, outbase):
    pf = lb * (lb + 1) / (2 * np.pi)

    fig = plt.figure(figsize=(6, 4))
    for idx, ifield in enumerate(ifield_list):
        pos = cl_cross[idx] > 0
        neg = cl_cross[idx] < 0
        if np.any(pos):
            plt.errorbar(
                lb[pos],
                (pf * cl_cross[idx])[pos],
                yerr=(pf * clerr_cross[idx])[pos],
                fmt="o",
                capsize=2.5,
                markersize=3,
                alpha=0.6,
                label=f"ifield{ifield}",
            )
        if np.any(neg):
            plt.errorbar(
                lb[neg],
                np.abs((pf * cl_cross[idx])[neg]),
                yerr=(pf * clerr_cross[idx])[neg],
                fmt="o",
                mfc="white",
                capsize=2.5,
                markersize=3,
                alpha=0.6,
            )

    fieldav_cl = np.mean(cl_cross, axis=0)
    fieldav_err = np.std(cl_cross, axis=0)
    pos = fieldav_cl > 0
    neg = fieldav_cl < 0
    if np.any(pos):
        plt.errorbar(
            lb[pos],
            (pf * fieldav_cl)[pos],
            yerr=(pf * fieldav_err)[pos],
            fmt="o",
            color="k",
            markersize=3.2,
            capsize=3,
            label="field average",
        )
    if np.any(neg):
        plt.errorbar(
            lb[neg],
            np.abs((pf * fieldav_cl)[neg]),
            yerr=(pf * fieldav_err)[neg],
            fmt="o",
            color="k",
            mfc="white",
            markersize=3.2,
            capsize=3,
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("$\\ell$")
    plt.ylabel("$D_{\\ell}$")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    fig.savefig(outbase + ".png", dpi=160)
    fig.savefig(outbase + ".pdf")
    plt.close(fig)


def run_gaia_x_galaxy_cross(
    inst,
    ifield_list,
    gaia_addstr,
    tracers,
    tracer_addstr,
    output_dir,
    tag,
    apply_pixel_corr=False,
    compute_cl_theta=DEFAULT_COMPUTE_CL_THETA,
    cl_theta_cut=DEFAULT_CL_THETA_CUT,
    n_rad_bins=DEFAULT_N_RAD_BINS,
    ell_min_wedge=DEFAULT_ELL_MIN_WEDGE,
    theta0=DEFAULT_THETA0,
):
    os.makedirs(output_dir, exist_ok=True)

    cbps = CIBER_PS_pipeline()
    mask_tail = _default_mask_tail(inst, DEFAULT_MASKING_MAGLIM)

    theta_masks = None
    if compute_cl_theta:
        rad_offset = -np.pi / n_rad_bins
        theta_masks = make_theta_masks(
            cbps.dimx,
            theta0=theta0,
            n_rad_bins=n_rad_bins,
            rad_offset=rad_offset,
            plot=False,
            ell_min_wedge=ell_min_wedge,
        )

    for tracer in tracers:
        catname = _resolve_catname(tracer)
        tracer_add = tracer_addstr.get(tracer, None)

        masks = _load_masks(
            cbps,
            inst,
            DEFAULT_IFIELDS,
            mask_tail,
            catname,
            DEFAULT_CLIP_SIGMA,
            DEFAULT_NITERMAX,
        )

        gaia_counts, _ = load_delta_g_maps("gaia", inst, gaia_addstr)
        tracer_counts, _ = load_delta_g_maps(catname, inst, tracer_add)

        all_cl_cross = []
        all_clerr_cross = []
        all_cl_gaia = []
        all_clerr_gaia = []
        all_cl_tracer = []
        all_clerr_tracer = []

        for ifield in ifield_list:
            fieldidx = ifield - 4
            mask = masks[fieldidx]

            mkk_type = "maskonly"
            mkkonly_savepath = (
                config.ciber_basepath
                + "data/fluctuation_data/TM"
                + str(inst)
                + "/mkk/"
                + mask_tail
                + "/mkk_"
                + mkk_type
                + "_estimate_ifield"
                + str(ifield)
                + "_observed_"
                + mask_tail
                + ".fits"
            )
            inv_mkk = fits.open(mkkonly_savepath)["inv_Mkk_" + str(ifield)].data

            if compute_cl_theta and cl_theta_cut:
                weights = np.ones_like(mask)
                for which_exclude in [0, n_rad_bins // 2]:
                    weights[theta_masks[which_exclude] == 1] = 0.0
            else:
                weights = np.ones_like(mask)

            lb, cl_cross, clerr_cross, cl_gaia, clerr_gaia, cl_tracer, clerr_tracer = (
                _compute_cross_for_field(
                    cbps,
                    inst,
                    ifield,
                    mask,
                    gaia_counts,
                    tracer_counts,
                    inv_mkk,
                    weights,
                    apply_pixel_corr=apply_pixel_corr,
                )
            )

            all_cl_cross.append(cl_cross)
            all_clerr_cross.append(clerr_cross)
            all_cl_gaia.append(cl_gaia)
            all_clerr_gaia.append(clerr_gaia)
            all_cl_tracer.append(cl_tracer)
            all_clerr_tracer.append(clerr_tracer)

        all_cl_cross = np.array(all_cl_cross)
        all_clerr_cross = np.array(all_clerr_cross)
        all_cl_gaia = np.array(all_cl_gaia)
        all_clerr_gaia = np.array(all_clerr_gaia)
        all_cl_tracer = np.array(all_cl_tracer)
        all_clerr_tracer = np.array(all_clerr_tracer)

        addstr_bits = ["gaia", gaia_addstr, catname]
        if tracer_add is not None:
            addstr_bits.append(tracer_add)
        addstr_bits.append(tag)
        addstr = "_".join(addstr_bits)

        save_fpath = os.path.join(
            output_dir,
            f"gaia_x_{catname}_TM{inst}_{tag}.npz",
        )
        np.savez(
            save_fpath,
            inst=inst,
            ifield_list_use=np.array(ifield_list),
            lb=lb,
            all_cl_cross=all_cl_cross,
            all_clerr_cross=all_clerr_cross,
            all_cl_gaia=all_cl_gaia,
            all_clerr_gaia=all_clerr_gaia,
            all_cl_tracer=all_cl_tracer,
            all_clerr_tracer=all_clerr_tracer,
            gaia_addstr=gaia_addstr,
            tracer_addstr=tracer_add,
            catname=catname,
            mask_tail=mask_tail,
            masking_maglim=DEFAULT_MASKING_MAGLIM,
            clip_sigma=DEFAULT_CLIP_SIGMA,
            nitermax=DEFAULT_NITERMAX,
            apply_pixel_corr=apply_pixel_corr,
            compute_cl_theta=compute_cl_theta,
            cl_theta_cut=cl_theta_cut,
            n_rad_bins=n_rad_bins,
            ell_min_wedge=ell_min_wedge,
            addstr=addstr,
        )
        print(f"Saved spectra to {save_fpath}")

        plot_base = os.path.join(
            output_dir,
            f"gaia_x_{catname}_TM{inst}_{tag}",
        )
        title = f"Gaia x {catname} (TM{inst})"
        _plot_cross(lb, all_cl_cross, all_clerr_cross, ifield_list, title, plot_base)
        print(f"Saved plots to {plot_base}.png/.pdf")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compute Gaia x galaxy cross spectra using CIBER masks (TM1 footprints)."
    )
    parser.add_argument(
        "--inst",
        type=int,
        default=DEFAULT_INST,
        choices=[1],
        help="CIBER instrument (TM1 only for this workflow).",
    )
    parser.add_argument(
        "--ifield",
        type=int,
        nargs="+",
        default=DEFAULT_IFIELDS,
        help="CIBER field list (default: 4 5 6 7 8).",
    )
    parser.add_argument(
        "--gaia-addstr",
        default=DEFAULT_GAIA_ADDSTR,
        help="Addstr for Gaia density maps (default matches paper plots).",
    )
    parser.add_argument(
        "--tracers",
        nargs="+",
        default=DEFAULT_TRACERS,
        help="Galaxy tracers (DESILS, HSC, WISE, unWISE).",
    )
    parser.add_argument(
        "--tracer-addstr",
        nargs="+",
        default=[],
        help="Override tracer addstr as key=value (e.g., DESILS=sdss_z_lt_22.0).",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("data", "cross_cl_fits", "gaia_x_galaxy"),
        help="Output directory for spectra and plots.",
    )
    parser.add_argument(
        "--tag",
        default=datetime.now().strftime("%Y%m%d"),
        help="Tag appended to output filenames (default: YYYYMMDD).",
    )
    parser.add_argument(
        "--no-pixel-corr",
        action="store_true",
        help="Disable pixel window correction.",
    )
    return parser


def _parse_tracer_addstr(kv_list):
    overrides = {}
    for item in kv_list:
        if "=" not in item:
            raise ValueError(f"Expected key=value for tracer addstr override: {item}")
        key, value = item.split("=", 1)
        overrides[key] = value
    return overrides


def main():
    parser = build_parser()
    args = parser.parse_args()

    tracer_addstr = dict(DEFAULT_TRACER_ADDSTR)
    tracer_addstr.update(_parse_tracer_addstr(args.tracer_addstr))

    run_gaia_x_galaxy_cross(
        inst=args.inst,
        ifield_list=args.ifield,
        gaia_addstr=args.gaia_addstr,
        tracers=args.tracers,
        tracer_addstr=tracer_addstr,
        output_dir=args.output_dir,
        tag=args.tag,
        apply_pixel_corr=not args.no_pixel_corr,
    )


if __name__ == "__main__":
    main()
