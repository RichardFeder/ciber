#!/usr/bin/env python3
"""
Example usage of compute_effective_1h_template.py

This script demonstrates how to:
1. Load the effective 1h template computation
2. Access individual redshift bin components
3. Compare results and save for downstream analysis
"""

import sys
import os
from pathlib import Path
import numpy as np

# Change to project root and add scripts dir to path
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root / 'scripts'))
sys.path.insert(0, str(project_root))

from compute_effective_1h_template import compute_effective_1h_template


def analyze_effective_template(effective_1h, individual_1h, zbinedges):
    """
    Analyze and display results from effective template computation.

    Parameters
    ----------
    effective_1h : dict
        Effective 1h template from compute_effective_1h_template()
    individual_1h : dict
        Individual redshift bin components
    zbinedges : array_like
        Redshift bin edges
    """

    print("\n" + "="*70)
    print("EFFECTIVE 1H TEMPLATE ANALYSIS")
    print("="*70)

    for slope in effective_1h.keys():
        print(f"\n--- Slope {slope} ---")

        eff = effective_1h[slope]
        ell = eff['ell']
        one_halo_sum = eff['one_halo_sum']
        one_halo_norm = eff['one_halo_norm']

        print(f"\nEffective template properties:")
        print(f"  Peak location: ℓ_peak ~ {ell[np.argmax(one_halo_norm)]:.0f}")
        print(f"  Peak amplitude (sum): {np.max(one_halo_sum):.3e}")
        print(f"  FWHM (approximate): {ell[one_halo_norm > 0.5].max() - ell[one_halo_norm > 0.5].min():.0f}")

        # Compare individual bin contributions
        print(f"\nIndividual redshift bin contributions:")
        nzbins = len(zbinedges) - 1

        for zidx in range(nzbins):
            if zidx not in individual_1h[slope]:
                continue

            z_info = individual_1h[slope][zidx]
            z_low, z_high = z_info['z_range']
            z_mid = z_info['z_mid']

            peak_idx = np.argmax(z_info['one_halo'])
            peak_ell = z_info['ell'][peak_idx]
            peak_amp = z_info['one_halo'][peak_idx]

            print(f"  z∈[{z_low:.1f}, {z_high:.1f}] (z_mid={z_mid:.2f}):")
            print(f"    Peak ℓ: {peak_ell:.0f}")
            print(f"    Peak amplitude: {peak_amp:.3e}")
            print(f"    A_1h (fitted): {z_info['A_1h']:.3e}")
            print(f"    μ_1h (fitted): {z_info['mu_1h']:.3f} (ℓ_peak ~ {np.exp(z_info['mu_1h']):.0f})")
            print(f"    σ_1h (fitted): {z_info['sigma_1h']:.3f}")

        # Save effective template to file for downstream use
        print(f"\nSaving effective template to file...")
        save_path = f'data/effective_1h_template_slope{slope:.1f}.npz'
        np.savez(
            save_path,
            ell=eff['ell'],
            one_halo_sum=eff['one_halo_sum'],
            one_halo_avg=eff['one_halo_avg'],
            one_halo_norm=eff['one_halo_norm'],
            zbinedges=zbinedges,
            slope=slope
        )
        print(f"  ✓ Saved: {save_path}")


def main():
    """Main execution."""

    template_dir = 'data/ihl_templates'
    zbinedges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    slopes = [1.0]

    print("="*70)
    print("EXAMPLE: EFFECTIVE ONE-HALO TEMPLATE COMPUTATION")
    print("="*70)

    # Check if template directory exists
    if not os.path.exists(template_dir):
        print(f"\nError: Template directory '{template_dir}' not found")
        print(f"Please ensure IHL template files are in: {template_dir}")
        print(f"Expected files like: ihl_ps_z_0.0_0.2_slope_1.0.txt")
        print(f"\nCurrently in: {os.getcwd()}")
        return

    # Compute effective template
    try:
        effective_1h, individual_1h, fit_results = compute_effective_1h_template(
            template_dir=template_dir,
            zbinedges=zbinedges,
            slopes=slopes,
            plot=True,
            figsize=(14, 10)
        )

        # Analyze results
        analyze_effective_template(effective_1h, individual_1h, zbinedges)

        print("\n" + "="*70)
        print("SUCCESS: Effective template computed and analyzed")
        print("="*70)
        print("\nNext steps:")
        print("  1. Check the plots in figures/effective_1h_template_slope*.png")
        print("  2. Review saved templates in data/effective_1h_template_slope*.npz")
        print("  3. Use these in your cross-correlation fitting pipeline")

    except Exception as e:
        print(f"\nError during computation: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()
