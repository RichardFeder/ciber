#!/usr/bin/env python3
"""
Compare r_ell measurements for dz=0.1 and dz=0.2 redshift binning.
This script loads and plots r_ell vs redshift for both binning schemes
to verify consistency between the different sampling strategies.
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


def main():
    print("Loading dz=0.1 binned data...")
    res_meas_dz01 = load_rlmeas_vs_z_DESILS()

    print("Loading dz=0.2 binned data...")
    res_meas_dz02 = load_rlmeas_vs_z_DESILS_dz02()

    print("\n=== dz=0.1 Results ===")
    print(f"Redshift bin edges: {res_meas_dz01['zbinedges']}")
    print(f"Scale ranges (ell): {list(zip(res_meas_dz01['lb_mins'], res_meas_dz01['lb_maxs']))}")
    print(f"Number of z-bins: {len(res_meas_dz01['zcen'])}")

    print("\n=== dz=0.2 Results ===")
    print(f"Redshift bin edges: {res_meas_dz02['zbinedges']}")
    print(f"Scale ranges (ell): {list(zip(res_meas_dz02['lb_mins'], res_meas_dz02['lb_maxs']))}")
    print(f"Number of z-bins: {len(res_meas_dz02['zcen'])}")

    print("\n=== dz=0.1 r_ell values (TM1) ===")
    for lidx, (lmin, lmax) in enumerate(zip(res_meas_dz01['lb_mins'], res_meas_dz01['lb_maxs'])):
        print(f"\nScale {lmin:.0f} < ell < {lmax:.0f}:")
        print(f"  z-centers: {res_meas_dz01['zcen']}")
        print(f"  r_ell: {res_meas_dz01['mean_rl_diffscale'][lidx, 0]}")
        print(f"  std:   {res_meas_dz01['std_rl_diffscale'][lidx, 0]}")

    print("\n=== dz=0.2 r_ell values (TM1) ===")
    for lidx, (lmin, lmax) in enumerate(zip(res_meas_dz02['lb_mins'], res_meas_dz02['lb_maxs'])):
        print(f"\nScale {lmin:.0f} < ell < {lmax:.0f}:")
        print(f"  z-centers: {res_meas_dz02['zcen']}")
        print(f"  r_ell: {res_meas_dz02['mean_rl_diffscale'][lidx, 0]}")
        print(f"  std:   {res_meas_dz02['std_rl_diffscale'][lidx, 0]}")

    print("\nSkipping mock predictions - comparing measurements only")

    print("\n=== Consistency Check ===")
    print("Comparing measurements for overlapping redshift regions...\n")

    for scale_idx, (lmin, lmax) in enumerate(zip(res_meas_dz01['lb_mins'], res_meas_dz01['lb_maxs'])):
        print(f"Scale {lmin:.0f} < ell < {lmax:.0f}:")
        print("-" * 70)

        z_edges_dz01 = res_meas_dz01['zbinedges']
        z_edges_dz02 = res_meas_dz02['zbinedges']

        # For each dz=0.2 bin, find overlapping dz=0.1 bins
        for z02_idx in range(len(z_edges_dz02)-1):
            z02_lo = z_edges_dz02[z02_idx]
            z02_hi = z_edges_dz02[z02_idx+1]

            # Average dz=0.1 bins that overlap with this dz=0.2 bin
            overlapping_dz01_idx = [
                i for i in range(len(z_edges_dz01)-1)
                if z_edges_dz01[i] >= z02_lo and z_edges_dz01[i+1] <= z02_hi
            ]

            if not overlapping_dz01_idx:
                continue

            for inst in [1, 2]:
                inst_str = f"TM{inst}"
                dz02_val = res_meas_dz02['mean_rl_diffscale'][scale_idx, inst-1, z02_idx]
                dz02_err = res_meas_dz02['std_rl_diffscale'][scale_idx, inst-1, z02_idx]

                # Average the overlapping dz=0.1 values
                dz01_vals_overlap = [
                    res_meas_dz01['mean_rl_diffscale'][scale_idx, inst-1, idx]
                    for idx in overlapping_dz01_idx
                ]
                dz01_errs_overlap = [
                    res_meas_dz01['std_rl_diffscale'][scale_idx, inst-1, idx]
                    for idx in overlapping_dz01_idx
                ]

                # Weighted average of overlapping dz=0.1 values
                weights = 1.0 / np.array(dz01_errs_overlap)**2
                dz01_avg = np.sum(weights * dz01_vals_overlap) / np.sum(weights)
                dz01_avg_err = np.sqrt(1.0 / np.sum(weights))

                # Compare
                diff = np.abs(dz02_val - dz01_avg)
                combined_err = np.sqrt(dz02_err**2 + dz01_avg_err**2)
                sigma = diff / combined_err if combined_err > 0 else 0

                z_label = f"  z∈[{z02_lo:.1f}, {z02_hi:.1f}]"
                consistency = "✓" if sigma < 2 else "✗"
                print(f"{z_label} {inst_str}: dz=0.2={dz02_val:.5f}±{dz02_err:.5f}, "
                      f"dz=0.1(avg)={dz01_avg:.5f}±{dz01_avg_err:.5f}, Δ={sigma:.2f}σ {consistency}")

        print()


def create_comparison_plots(res_meas_dz01, res_meas_dz02):
    """Create plots comparing dz=0.1 and dz=0.2 measurements."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Comparison of r_ell vs redshift: dz=0.1 vs dz=0.2', fontsize=16)

    z_edges_dz01 = res_meas_dz01['zbinedges']
    z_edges_dz02 = res_meas_dz02['zbinedges']

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
                       fmt='o-', label='dz=0.1', color='C0', markersize=6, alpha=0.7)

            # dz=0.2 data
            z_cen_dz02 = res_meas_dz02['zcen']
            r_ell_dz02 = res_meas_dz02['mean_rl_diffscale'][scale_idx, inst-1]
            r_ell_err_dz02 = res_meas_dz02['std_rl_diffscale'][scale_idx, inst-1]

            ax.errorbar(z_cen_dz02, r_ell_dz02, yerr=r_ell_err_dz02,
                       fmt='s--', label='dz=0.2', color='C1', markersize=7, alpha=0.7)

            ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
            ax.grid(alpha=0.3)
            ax.set_xlabel('redshift (z)', fontsize=11)
            ax.set_ylabel('$r_{\\ell}$', fontsize=11)
            ax.set_title(f'TM{inst} | {lmin:.0f}<ℓ<{lmax:.0f}', fontsize=12)
            if scale_idx == 0:
                ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig('figures/rl_dz01_vs_dz02_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved comparison plot: figures/rl_dz01_vs_dz02_comparison.png")
    plt.show()


if __name__ == '__main__':
    main()
