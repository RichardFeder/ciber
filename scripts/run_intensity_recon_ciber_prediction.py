#!/usr/bin/env python3
"""
Orchestration script: Intensity reconstruction CIBER auto-power prediction.

Loads precomputed intensity cross-correlation spectra, computes CIBER auto-power
predictions using the coherence-based estimator, and generates diagnostic figures.

Example usage:
    python scripts/run_intensity_recon_ciber_prediction.py \
        --gal-addstr baseline_hsc_zlt1 \
        --intensity-addstr i_cmodel_mag_z0.0_z1.0 \
        --hsc-mag-column i_cmodel_mag \
        --zmin 0.0 --zmax 1.0 \
        --outdir figures/intensity_recon_predictions/

Output:
    - Diagnostic figure (PNG) showing measured vs. predicted CIBER spectra
    - NPZ file with all intermediate products (spectra, predictions, residuals)
    - Summary statistics printed to stdout
"""

import argparse
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import numpy as np
import os

import config
from ciber.theory.intensity_reconstruction_diagnostics import (
    compute_intensity_recon_ciber_prediction,
    plot_reconstruction_comparison,
    plot_r_ell_comparison,
)


def list_available_gal_addstr(catname="HSC", inst=1):
    """List available gal_addstr values from NPZ filenames in data directories."""
    candidates = []
    search_dirs = [
        f"data/lens_prods/ciber_gal_cross/{catname}/TM{inst}",
        f"data/input_recovered_ps/ciber_gal_cross/{catname}/TM{inst}",
    ]
    
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        for filename in os.listdir(search_dir):
            if filename.endswith(".npz") and "ciber_gal_ps" in filename:
                # Extract gal_addstr from filename: ciber_gal_ps_TM{inst}_{catname}_{gal_addstr}.npz
                parts = filename.replace(".npz", "").split(f"_{catname}_")
                if len(parts) == 2:
                    candidates.append(parts[1])
    
    return sorted(set(candidates))


def main():
    parser = argparse.ArgumentParser(
        description="Intensity reconstruction CIBER auto-power prediction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--inst-list",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Instrument IDs [default: 1 2]",
    )
    parser.add_argument(
        "--ifield-list",
        type=int,
        nargs="+",
        default=[8],
        help="Field IDs [default: 4 5 6 7 8]",
    )
    parser.add_argument(
        "--catname",
        type=str,
        default="HSC",
        choices=["HSC", "LS"],
        help="Catalog name: HSC or LS [default: HSC]",
    )
    parser.add_argument(
        "--gal-addstr",
        type=str,
        default=None,
        help="Addstring from prior galaxy-cross run (appears in NPZ filename). "
             "If not provided, will attempt auto-detect from saved runs.",
    )
    parser.add_argument(
        "--intensity-addstr",
        type=str,
        default=None,
        help="Addstring from intensity-cross run [optional, for reference only]",
    )
    parser.add_argument(
        "--mag-column",
        type=str,
        default=None,
        help="Magnitude column for metadata (HSC: z_cmodel_mag, LS: inferred). "
             "If not provided, will use catalog default.",
    )
    parser.add_argument(
        "--zmin",
        type=float,
        default=0.0,
        help="Redshift minimum [default: 0.0]",
    )
    parser.add_argument(
        "--zmax",
        type=float,
        default=1.0,
        help="Redshift maximum [default: 1.0]",
    )
    parser.add_argument(
        "--gal-addstr-compare",
        type=str,
        default=None,
        help="Addstring from parallel galaxy-cross run for comparison [optional]",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default='figures/intensity_recon_predictions/',
        help="Output directory for figures and products [default: figures/intensity_recon_predictions/]",
    )
    parser.add_argument(
        "--mask-tail-list",
        type=str,
        nargs="+",
        default=None,
        help="Mask tail overrides per instrument [optional]",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Show figure interactively",
    )
    parser.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Do not show figure interactively",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        default=False,
        help="Check data integrity without plotting",
    )
    parser.add_argument(
        "--ell-lims",
        type=float,
        nargs=2,
        default=[280.0, 1.1e5],
        help="Multipole limits for plotting [ell_min ell_max]",
    )
    parser.add_argument(
        "--list-available-runs",
        action="store_true",
        help="List available gal_addstr values from saved NPZ files and exit",
    )

    args = parser.parse_args()

    # Handle --list-available-runs early
    if args.list_available_runs:
        print("\nAvailable gal_addstr values (NPZ runs):")
        available = list_available_gal_addstr(args.catname, args.inst_list[0])
        if available:
            for addstr in available:
                print(f"  {addstr}")
        else:
            print("  [None found in data/lens_prods/ or data/input_recovered_ps/]")
        return 0

    # Set defaults based on catalog
    if args.mag_column is None:
        args.mag_column = "z_cmodel_mag" if args.catname == "HSC" else "DESI_LS"
    
    if args.gal_addstr is None:
        # Try to auto-detect from available runs
        available = list_available_gal_addstr(args.catname, args.inst_list[0])
        if available:
            args.gal_addstr = available[0]
            print(f"[Auto-detected gal_addstr from available runs: {args.gal_addstr}]\n")
        else:
            print(f"[ERROR] No gal_addstr provided and none found in data directories.")
            print(f"        Provide --gal-addstr explicitly or run intensity cross first.")
            return 1

    # Set output directory
    if args.outdir is None:
        args.outdir = "figures/intensity_recon_predictions/"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Intensity Reconstruction → CIBER Auto-Power Prediction")
    print("=" * 80)
    print(f"\nInput Parameters:")
    print(f"  Catalog: {args.catname}")
    print(f"  Instruments: {args.inst_list}")
    print(f"  Fields: {args.ifield_list}")
    print(f"  Mag column: {args.mag_column}")
    print(f"  Redshift: {args.zmin:.1f} < z < {args.zmax:.1f}")
    print(f"  Gal addstr: {args.gal_addstr}")
    print(f"  Intensity addstr: {args.intensity_addstr}")
    print(f"  Output dir: {outdir}")
    print()

    # Compute predictions
    print("[Step 1] Loading precomputed intensity spectra...")
    results = compute_intensity_recon_ciber_prediction(
        inst_list=args.inst_list,
        ifield_list=args.ifield_list,
        catname=args.catname,
        gal_addstr=args.gal_addstr,
        intensity_addstr=args.intensity_addstr,
        hsc_mag_column=args.mag_column,
        zmin=args.zmin,
        zmax=args.zmax,
        mask_tail_list=args.mask_tail_list,
        gal_addstr_compare=args.gal_addstr_compare,
        verbose=True,
    )

    if results is None:
        print("[ERROR] Failed to load intensity spectra. Exiting.")
        return 1

    # Print summary statistics
    print("\n[Step 2] Computing statistics...")
    all_results = results["all_results"]
    metadata = results["metadata"]

    for inst in args.inst_list:
        if inst not in all_results:
            print(f"  TM{inst}: No data available")
            continue

        res = all_results[inst]
        rl = res["rl_intensity_f25b"]

        print(f"\n  TM{inst} Summary:")
        print(f"    Mean coherence r_ell: {np.nanmean(rl):.4f} ± {np.nanstd(rl):.4f}")
        print(f"    Median coherence: {np.nanmedian(rl):.4f}")
        print(f"    Coherence range: [{np.nanmin(rl):.4f}, {np.nanmax(rl):.4f}]")
        print(f"    Valid r_ell fraction: {np.sum(np.isfinite(rl)) / rl.size:.1%}")
        print()

    # Plot diagnostics
    if not args.verify_only:
        print("\n[Step 3] Generating zlt1-style auto prediction comparison...")
        fig = plot_reconstruction_comparison(
            results,
            ell_min=args.ell_lims[0],
            ell_max=args.ell_lims[1],
            outdir=outdir,
            figname="intensity_reconstruction_auto_prediction_comparison.png",
            plot=args.plot,
            verbose=True,
        )

        print("\n[Step 4] Generating r_ell comparison figure...")
        fig_rl = plot_r_ell_comparison(
            results,
            ell_min=args.ell_lims[0],
            ell_max=args.ell_lims[1],
            outdir=outdir,
            figname="intensity_reconstruction_r_ell_comparison.png",
            plot=args.plot,
            verbose=True,
        )

        # Save NPZ product
        print("\n[Step 5] Saving output products...")
        npz_fname = f"intensity_recon_prediction_{args.mag_column}_z{args.zmin:.1f}_{args.zmax:.1f}.npz"
        npz_path = outdir / npz_fname

        # Collect all data for NPZ
        npz_dict = {"metadata": metadata}
        for inst in args.inst_list:
            if inst in all_results:

                res = all_results[inst]
                print('cl intensity auto has shape', res["cl_intensity_auto"].shape)
                npz_dict[f"TM{inst}_ell"] = results["ell_array"]
                npz_dict[f"TM{inst}_cl_intensity_auto"] = res["cl_intensity_auto"]
                npz_dict[f"TM{inst}_cl_intensity_cross"] = res["cl_intensity_cross"]
                npz_dict[f"TM{inst}_cl_ciber_auto_f25b"] = res["cl_ciber_auto_f25b"]
                npz_dict[f"TM{inst}_clerr_ciber_auto_f25b"] = res["clerr_ciber_auto_f25b"]
                npz_dict[f"TM{inst}_cl_ciber_auto_pred"] = res["cl_ciber_auto_pred"]
                npz_dict[f"TM{inst}_rl_intensity_f25b"] = res["rl_intensity_f25b"]
                if res["rl_gal_f25b"] is not None:
                    npz_dict[f"TM{inst}_rl_gal_f25b"] = res["rl_gal_f25b"]

        np.savez(npz_path, **npz_dict)
        print(f"  Saved to {npz_path}")

    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
