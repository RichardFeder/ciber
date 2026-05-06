#!/usr/bin/env python3
"""
Compute IHL one-halo parameters for use in galaxy cross-spectrum fitting.

This script:
1. Loads and decomposes IHL templates for dz=0.2 redshift bins
2. Extracts one-halo parameters (mu_1h, sigma_1h) from each bin
3. Computes linear relationships between these parameters and redshift
4. Saves parameters for use by run_gal_cross_fits

Usage:
    python3 scripts/compute_ihl_1h_params.py
    python3 scripts/compute_ihl_1h_params.py --template-dir data/ihl_templates --output-file ihl_1h_params_corrected.npz --ell-scale 0.31831
"""

import sys
import os
from pathlib import Path
import numpy as np

# Setup paths
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root / 'scripts'))
sys.path.insert(0, str(project_root))

import config  # noqa: E402

from ciber.theory.cl_template import (
    fit_and_decompose_ihl_templates,
    save_ihl_1h_params,
)


def main(template_dir='data/ihl_templates', output_file='ihl_1h_params.npz', ell_scale=1.0):
    """Main execution."""

    print("\n" + "="*70)
    print("COMPUTE IHL ONE-HALO PARAMETERS FOR GALAXY CROSS FITS")
    print("="*70)
    if ell_scale != 1.0:
        print(f"Note: ell values will be scaled by {ell_scale}")
        print(f"      (To correct for ell labeled factor of π too high, use: {1.0/np.pi:.5f})")

    # Configuration
    zbinedges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    slopes = [1.0]

    # Check template directory
    if not os.path.exists(template_dir):
        print(f"\nError: Template directory not found: {template_dir}")
        print(f"Current directory: {os.getcwd()}")
        return False

    # Decompose IHL templates for all redshift bins
    print("\n" + "-"*70)
    print("DECOMPOSING IHL TEMPLATES FOR EACH REDSHIFT BIN")
    print("-"*70)

    try:
        fit_results = fit_and_decompose_ihl_templates(
            template_dir=template_dir,
            zbinedges=zbinedges,
            slopes=slopes,
            use_powerlaw_2h=True,
            alpha_2h_fixed=0.0,
            fit_ell_range=None,
            plot=False,
            verbose=True,
            ell_scale=ell_scale
        )

        # Save one-halo parameters and compute linear relationships
        print("\n" + "-"*70)
        print("COMPUTING LINEAR RELATIONSHIPS")
        print("-"*70)

        one_halo_params = save_ihl_1h_params(
            fit_results,
            output_file,
            zbinedges=zbinedges,
            slopes=slopes
        )

        print("\n" + "="*70)
        print("ONE-HALO PARAMETERS COMPUTED SUCCESSFULLY")
        print("="*70)
        print(f"\nParameters saved to: {output_file}")
        print(f"\nLinear relationships computed:")

        for slope in slopes:
            ln_rel = one_halo_params['ln_ell_peak_vs_z'].get(slope)
            sigma_rel = one_halo_params['sigma_vs_z'].get(slope)

            if ln_rel is not None:
                print(f"\n  Slope {slope}:")
                print(f"    ln(ell_peak) = {ln_rel['intercept']:.3f} + {ln_rel['slope']:.4f} * z")
                print(f"      R² = {ln_rel['r_value']**2:.4f}")
                print(f"    sigma = {sigma_rel['intercept']:.4f} + {sigma_rel['slope']:.4f} * z")
                print(f"      R² = {sigma_rel['r_value']**2:.4f}")

        print(f"\nTo use these parameters in fitting, run:")
        print(f"  python3 scripts/auto_cross_fits_pipeline.py ... --ihl-1h-params-path {output_file} ...")

        return True

    except Exception as e:
        print(f"\nError during parameter computation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute IHL one-halo parameters for galaxy cross-fits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--template-dir',
        default='data/ihl_templates',
        help='Directory containing IHL template files'
    )
    parser.add_argument(
        '--output-file',
        default='ihl_1h_params.npz',
        help='Output file to save parameters'
    )
    parser.add_argument(
        '--ell-scale',
        type=float,
        default=1.0,
        help='Scaling factor for ell values (e.g., 0.31831 for 1/π correction)'
    )
    args = parser.parse_args()

    success = main(
        template_dir=args.template_dir,
        output_file=args.output_file,
        ell_scale=args.ell_scale
    )
    sys.exit(0 if success else 1)
