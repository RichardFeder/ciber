#!/usr/bin/env python3
"""Compare in-situ CIBER auto-spectrum vs F25B file.

Loads in-situ auto from cross-product file and F25B auto file,
plots D_ell for both, and shows the ratio to quantify discrepancy vs ell.
"""

import numpy as np
from pathlib import Path
import sys
import argparse

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from ciber.io.ciber_data_utils import load_ciber_gal_ps
from ciber.plotting.gal_plotting_fns import _load_ciber_auto_file


def convert_cl_to_dell(lb, cl):
    """Convert C_ell to D_ell = ell(ell+1)/(2*pi) * C_ell."""
    return lb * (lb + 1) / (2 * np.pi) * cl


def main():
    parser = argparse.ArgumentParser(
        description="Compare in-situ CIBER auto-spectrum vs F25B file"
    )
    parser.add_argument(
        "--cat", type=str, default="HSC",
        help="Catalog name (HSC, DESILS, etc.)"
    )
    parser.add_argument(
        "--inst", type=int, nargs='+', default=[1, 2],
        help="Instrument indices to plot (default: 1 2)"
    )
    parser.add_argument(
        "--addstr", type=str, default=None,
        help="Additional string for file naming"
    )
    parser.add_argument(
        "--headstr", type=str, default="ilt25.0",
        help="Header string for HSC files (default: ilt25.0)"
    )
    args = parser.parse_args()

    catname = args.cat
    inst_list = args.inst
    addstr = args.addstr
    headstr = args.headstr if catname == "HSC" else None

    bandstr_list = {1: 'J', 2: 'H'}

    # Create output directory if needed
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    for inst in inst_list:
        print(f"\n{'='*80}")
        print(f"Processing {catname} TM{inst}...")
        print(f"{'='*80}")

        bandstr = bandstr_list[inst]

        # Build addstr_use similar to collect_ciber_gal_vs_redshift
        addstr_use = catname + '_coarsez'
        if headstr is not None:
            addstr_use += f'_{headstr}'
        if addstr is not None:
            addstr_use += f'_{addstr}'

        # Load in-situ auto from cross-product file
        try:
            cgps_file = load_ciber_gal_ps(inst, catname, addstr=addstr_use)
        except Exception as e:
            print(f"Failed to load cross-product file: {e}")
            continue

        if 'all_cl_ciber_auto_inplace' not in cgps_file.files:
            print(f"No in-situ auto in {addstr_use}; skipping")
            continue

        all_cl_ciber_auto_inplace = cgps_file['all_cl_ciber_auto_inplace']
        lb = cgps_file['lb']

        # Extract per-field auto spectra (already noise-subtracted)
        n_fields, n_ell = all_cl_ciber_auto_inplace.shape
        print(f"Loaded in-situ auto: shape {all_cl_ciber_auto_inplace.shape}")

        # Field-averaged in-situ auto
        cl_inplace_fieldav = np.nanmean(all_cl_ciber_auto_inplace, axis=0)

        # Load F25B auto file
        try:
            ciber_auto_f25b = _load_ciber_auto_file(bandstr)
            lb_f25b = ciber_auto_f25b['lb']
            cl_f25b = ciber_auto_f25b['fieldav_cl']
            print(f"Loaded F25B auto: {ciber_auto_f25b.get('source_path', 'unknown')}")
        except Exception as e:
            print(f"Failed to load F25B file: {e}")
            continue

        # Interpolate F25B onto in-situ grid
        cl_f25b_interp = np.interp(lb, lb_f25b, cl_f25b, left=cl_f25b[0], right=cl_f25b[-1])

        # Convert to D_ell for plotting (ell(ell+1)/(2*pi) * C_ell)
        dell_inplace_fieldav = convert_cl_to_dell(lb, cl_inplace_fieldav)
        dell_f25b = convert_cl_to_dell(lb, cl_f25b_interp)

        # ====================================================================
        # Figure 1: D_ell comparison (in-situ per field + field avg vs F25B)
        # ====================================================================
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot in-situ per field as thin lines
        for field_idx in range(n_fields):
            dell_field = convert_cl_to_dell(lb, all_cl_ciber_auto_inplace[field_idx])
            ax.loglog(lb, dell_field, color='C0', alpha=0.3, linewidth=1.0)

        # Field-averaged in-situ as thick solid line
        ax.loglog(lb, dell_inplace_fieldav, color='C0', linewidth=2.5, label='In-situ (field avg, signal-only)')

        # F25B as thick dashed line
        ax.loglog(lb, dell_f25b, color='C1', linewidth=2.5, linestyle='--', label='F25B (pre-computed)')

        ax.set_xlabel('Multipole $\ell$', fontsize=12)
        ax.set_ylabel('$D_\ell^{II} = \\ell(\\ell+1)/(2\pi) C_\ell^{II}$', fontsize=12)
        ax.set_title(f'{catname} TM{inst}: CIBER Auto-Spectrum Comparison', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=11, loc='best')
        fig.tight_layout()
        fig.savefig(fig_dir / f'compare_ciber_auto_dell_TM{inst}_{catname}.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {fig_dir / f'compare_ciber_auto_dell_TM{inst}_{catname}.png'}")
        plt.close(fig)

        # ====================================================================
        # Figure 2: Ratio (in-situ / F25B) vs ell
        # ====================================================================
        fig, ax = plt.subplots(figsize=(10, 6))

        # Avoid division by zero
        ratio = np.divide(dell_inplace_fieldav, dell_f25b, where=dell_f25b > 0, out=np.ones_like(dell_inplace_fieldav))

        ax.semilogx(lb, ratio, 'o-', color='C0', linewidth=2, markersize=6, label='In-situ / F25B')
        ax.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Reference (ratio = 1)')

        ax.set_xlabel('Multipole $\ell$', fontsize=12)
        ax.set_ylabel('Ratio: In-situ / F25B', fontsize=12)
        ax.set_title(f'{catname} TM{inst}: Auto-Spectrum Discrepancy', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=11, loc='best')
        ax.set_ylim([0.5, 2.0])
        fig.tight_layout()
        fig.savefig(fig_dir / f'compare_ciber_auto_ratio_TM{inst}_{catname}.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {fig_dir / f'compare_ciber_auto_ratio_TM{inst}_{catname}.png'}")
        plt.close(fig)

        # Print statistics
        print(f"\nStatistics for TM{inst}:")
        print(f"  In-situ D_ell range: [{np.nanmin(dell_inplace_fieldav):.3e}, {np.nanmax(dell_inplace_fieldav):.3e}]")
        print(f"  F25B D_ell range:    [{np.nanmin(dell_f25b):.3e}, {np.nanmax(dell_f25b):.3e}]")
        print(f"  Ratio range:         [{np.nanmin(ratio):.3f}, {np.nanmax(ratio):.3f}]")
        print(f"  Ratio mean:          {np.nanmean(ratio):.3f}")
        print(f"  Ratio std:           {np.nanstd(ratio):.3f}")

        # Find ell ranges with largest discrepancy
        sorted_idx = np.argsort(np.abs(ratio - 1.0))[::-1]
        print(f"\n  Top 5 ell with largest discrepancy:")
        for i in range(min(5, len(sorted_idx))):
            idx = sorted_idx[i]
            print(f"    ell={lb[idx]:.0f}: ratio={ratio[idx]:.3f}")

    print(f"\n{'='*80}")
    print("All comparisons complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
