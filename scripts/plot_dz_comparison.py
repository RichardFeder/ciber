#!/usr/bin/env python3
"""
Create visual comparison plots for dz=0.1 vs dz=0.2 r_ell measurements.
"""

import sys
import os
from pathlib import Path

# Change to project root and add scripts dir to path
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root / 'scripts'))
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import config  # noqa: E402 (sets ciber_basepath)

from ciber.plotting.gal_plotting_fns import (
    load_rlmeas_vs_z_DESILS,
    load_rlmeas_vs_z_DESILS_dz02,
)


def create_comparison_plots(res_meas_dz01, res_meas_dz02):
    """Create side-by-side comparison plots for dz=0.1 and dz=0.2."""

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('CIBER × DESI-LS: r_ell Comparison (dz=0.1 vs dz=0.2 binning)',
                 fontsize=16, fontweight='bold')

    z_edges_dz01 = res_meas_dz01['zbinedges']
    z_edges_dz02 = res_meas_dz02['zbinedges']

    lams = [1.1, 1.8]

    for inst_idx, inst in enumerate([1, 2]):
        for scale_idx, (lmin, lmax) in enumerate(
            zip(res_meas_dz01['lb_mins'], res_meas_dz01['lb_maxs'])
        ):
            ax = axes[inst_idx, scale_idx]

            # dz=0.1 data
            z_cen_dz01 = res_meas_dz01['zcen']
            r_ell_dz01 = res_meas_dz01['mean_rl_diffscale'][scale_idx, inst-1]
            r_ell_err_dz01 = res_meas_dz01['std_rl_diffscale'][scale_idx, inst-1]

            ax.errorbar(z_cen_dz01, r_ell_dz01, yerr=r_ell_err_dz01,
                       fmt='o-', label='dz=0.1', color='C0', markersize=5,
                       linewidth=1.5, capsize=3, alpha=0.7)

            # dz=0.2 data
            z_cen_dz02 = res_meas_dz02['zcen']
            r_ell_dz02 = res_meas_dz02['mean_rl_diffscale'][scale_idx, inst-1]
            r_ell_err_dz02 = res_meas_dz02['std_rl_diffscale'][scale_idx, inst-1]

            ax.errorbar(z_cen_dz02, r_ell_dz02, yerr=r_ell_err_dz02,
                       fmt='s--', label='dz=0.2', color='C1', markersize=6,
                       linewidth=1.5, capsize=3, alpha=0.7)

            ax.axhline(0, color='gray', linestyle=':', alpha=0.4, linewidth=1)
            ax.grid(alpha=0.3, linestyle='--')
            ax.set_xlabel('redshift', fontsize=11)
            ax.set_ylabel('$r_{\\ell}$', fontsize=12)

            scale_label = f'{lmin:.0f} < ℓ < {lmax:.0f}'
            inst_label = f'TM{inst} ({lams[inst-1]} μm)'
            ax.set_title(f'{inst_label} | {scale_label}', fontsize=12, fontweight='bold')

            ax.set_ylim([-0.1, 0.4])

            if scale_idx == 0 and inst_idx == 0:
                ax.legend(loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig('figures/rl_dz01_vs_dz02_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: figures/rl_dz01_vs_dz02_comparison.png")
    plt.show()


def create_difference_plots(res_meas_dz01, res_meas_dz02):
    """Create plots showing the differences between dz=0.1 and dz=0.2."""

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('CIBER × DESI-LS: Fractional Difference (dz=0.2 - dz=0.1) / dz=0.1',
                 fontsize=16, fontweight='bold')

    z_edges_dz01 = res_meas_dz01['zbinedges']
    z_edges_dz02 = res_meas_dz02['zbinedges']

    lams = [1.1, 1.8]

    for inst_idx, inst in enumerate([1, 2]):
        for scale_idx, (lmin, lmax) in enumerate(
            zip(res_meas_dz01['lb_mins'], res_meas_dz01['lb_maxs'])
        ):
            ax = axes[inst_idx, scale_idx]

            # For each dz=0.2 bin, find overlapping dz=0.1 bins
            diffs = []
            diff_errs = []
            z_plot = []

            for z02_idx in range(len(z_edges_dz02)-1):
                z02_lo = z_edges_dz02[z02_idx]
                z02_hi = z_edges_dz02[z02_idx+1]
                z_mid = 0.5 * (z02_lo + z02_hi)

                # Average dz=0.1 bins that overlap with this dz=0.2 bin
                overlapping_dz01_idx = [
                    i for i in range(len(z_edges_dz01)-1)
                    if z_edges_dz01[i] >= z02_lo and z_edges_dz01[i+1] <= z02_hi
                ]

                if not overlapping_dz01_idx:
                    continue

                dz02_val = res_meas_dz02['mean_rl_diffscale'][scale_idx, inst-1, z02_idx]
                dz02_err = res_meas_dz02['std_rl_diffscale'][scale_idx, inst-1, z02_idx]

                # Weighted average of overlapping dz=0.1 values
                dz01_vals_overlap = [
                    res_meas_dz01['mean_rl_diffscale'][scale_idx, inst-1, idx]
                    for idx in overlapping_dz01_idx
                ]
                dz01_errs_overlap = [
                    res_meas_dz01['std_rl_diffscale'][scale_idx, inst-1, idx]
                    for idx in overlapping_dz01_idx
                ]

                weights = 1.0 / np.array(dz01_errs_overlap)**2
                dz01_avg = np.sum(weights * dz01_vals_overlap) / np.sum(weights)
                dz01_avg_err = np.sqrt(1.0 / np.sum(weights))

                # Fractional difference
                if dz01_avg != 0:
                    frac_diff = (dz02_val - dz01_avg) / np.abs(dz01_avg)
                    # Error propagation for fractional difference
                    frac_err = np.sqrt((dz02_err/np.abs(dz01_avg))**2 +
                                      ((dz02_val - dz01_avg)*dz01_avg_err/dz01_avg**2)**2)
                else:
                    frac_diff = 0
                    frac_err = 0

                diffs.append(frac_diff)
                diff_errs.append(frac_err)
                z_plot.append(z_mid)

            if diffs:
                ax.errorbar(z_plot, diffs, yerr=diff_errs, fmt='o-', color='C2',
                           markersize=8, linewidth=2, capsize=5, alpha=0.7)
                ax.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
                ax.fill_between(z_plot, -0.2, 0.2, alpha=0.1, color='gray')

            ax.grid(alpha=0.3, linestyle='--')
            ax.set_xlabel('redshift', fontsize=11)
            ax.set_ylabel('Fractional difference', fontsize=12)

            scale_label = f'{lmin:.0f} < ℓ < {lmax:.0f}'
            inst_label = f'TM{inst} ({lams[inst-1]} μm)'
            ax.set_title(f'{inst_label} | {scale_label}', fontsize=12, fontweight='bold')
            ax.set_ylim([-0.6, 0.6])

    plt.tight_layout()
    plt.savefig('figures/rl_dz01_vs_dz02_fractional_difference.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: figures/rl_dz01_vs_dz02_fractional_difference.png")
    plt.show()


if __name__ == '__main__':
    print("Loading dz=0.1 binned data...")
    res_meas_dz01 = load_rlmeas_vs_z_DESILS()

    print("Loading dz=0.2 binned data...")
    res_meas_dz02 = load_rlmeas_vs_z_DESILS_dz02()

    print("\nCreating comparison plots...")
    try:
        create_comparison_plots(res_meas_dz01, res_meas_dz02)
    except Exception as e:
        print(f"Warning: could not display comparison plot: {e}")

    print("\nCreating fractional difference plots...")
    try:
        create_difference_plots(res_meas_dz01, res_meas_dz02)
    except Exception as e:
        print(f"Warning: could not display difference plot: {e}")

    print("\nDone!")
