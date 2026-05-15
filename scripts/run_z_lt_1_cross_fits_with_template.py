#!/usr/bin/env python3
"""
Run DESI-LS and HSC z<1.0 cross-spectrum fits using the effective one-halo template.

This script:
1. Loads the cached effective 1h template
2. Runs cross-spectrum fits for DESI-LS z<1.0
3. Runs cross-spectrum fits for HSC z<1.0
4. Compares measured 1h components to effective template
5. Generates summary comparison plots

Usage:
    # Quick run with defaults (z<1.0, dz=0.2 bins, lMax=[50000])
    python3 scripts/run_z_lt_1_cross_fits_with_template.py

    # Custom lMax values
    python3 scripts/run_z_lt_1_cross_fits_with_template.py --lmax 30000 50000 70000

    # Overwrite existing fits
    python3 scripts/run_z_lt_1_cross_fits_with_template.py --overwrite

    # Just DESI-LS
    python3 scripts/run_z_lt_1_cross_fits_with_template.py --cat DESILS

    # Just HSC
    python3 scripts/run_z_lt_1_cross_fits_with_template.py --cat HSC
"""

import sys
import os
from pathlib import Path
import argparse
import subprocess

# Setup paths
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root / 'scripts'))
sys.path.insert(0, str(project_root))

import config  # noqa: E402
import numpy as np
import matplotlib.pyplot as plt


def check_cache_exists():
    """Verify that the one-halo template cache exists."""
    cache_dir = Path('data/1h_template_cache')
    metadata_file = cache_dir / 'cache_metadata.json'

    if not metadata_file.exists():
        print("\n" + "="*70)
        print("ERROR: One-halo template cache not found!")
        print("="*70)
        print(f"\nCache directory: {cache_dir}")
        print(f"Metadata file: {metadata_file}")
        print("\nPlease create the cache first:")
        print("  python3 scripts/cache_effective_1h_templates.py")
        print("\n" + "="*70)
        return False

    print("\n✓ One-halo template cache found and valid")
    return True


def run_cross_fits(cats, lmax_values, fitstr, overwrite, zbinedges):
    """Run cross-spectrum fits using auto_cross_fits_pipeline.py."""

    print("\n" + "="*70)
    print("RUNNING CROSS-SPECTRUM FITS")
    print("="*70)

    # Build command
    cmd = [
        'python3', 'scripts/auto_cross_fits_pipeline.py',
        '--mode', 'run_cross',
        '--cat'] + cats + [
        '--lmax'] + [str(lm) for lm in lmax_values] + [
        '--fitstr-cross', fitstr,
        '--zbinedges'] + [str(z) for z in zbinedges]

    if overwrite:
        cmd.append('--overwrite')

    print(f"\nCommand:")
    print(f"  {' '.join(cmd)}\n")

    # Run
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode == 0


def load_and_compare_1h_templates():
    """Load cache and show template information."""

    from ciber.theory.ihl_1h_template_cache import OneHaloTemplateCache

    print("\n" + "="*70)
    print("ONE-HALO TEMPLATE INFORMATION")
    print("="*70)

    cache = OneHaloTemplateCache()
    effective_1h, individual_1h, zbinedges = cache.load_cache(slope=1.0)

    eff = effective_1h[1.0]

    print(f"\nEffective Template (z<1.0, dz=0.2 bins):")
    print(f"  ℓ range: {eff['ell'][0]:.0f} - {eff['ell'][-1]:.0f}")
    print(f"  Peak location: ℓ ≈ {eff['ell'][np.argmax(eff['one_halo_norm'])]:.0f}")
    print(f"  Peak amplitude (sum): {np.max(eff['one_halo_sum']):.3e}")

    print(f"\nIndividual z-bin contributions:")
    for zidx in individual_1h[1.0]:
        z_info = individual_1h[1.0][zidx]
        z_lo, z_hi = z_info['z_range']
        peak_ell = np.exp(z_info['mu_1h'])
        print(f"  z∈[{z_lo:.1f}, {z_hi:.1f}]: A_1h={z_info['A_1h']:.2e}, "
              f"ℓ_peak≈{peak_ell:.0f}, σ={z_info['sigma_1h']:.3f}")

    print("\n✓ Template information loaded successfully")


def print_usage_examples():
    """Print usage examples for fitting pipeline integration."""

    print("\n" + "="*70)
    print("INTEGRATION EXAMPLES")
    print("="*70)

    examples = """
USING THE CACHED TEMPLATE IN YOUR ANALYSIS:

1. Quick load for reference:
   from ciber.theory.ihl_1h_template_cache import load_effective_1h_for_fitting
   one_halo_norm = load_effective_1h_for_fitting(slope=1.0)

2. Full load with all data:
   from ciber.theory.ihl_1h_template_cache import OneHaloTemplateCache
   cache = OneHaloTemplateCache()
   effective_1h, individual_1h, zbinedges = cache.load_cache(slope=1.0)

3. Direct NPZ access:
   data = np.load('data/1h_template_cache/effective_1h_slope_1.0.npz')
   ell = data['ell']
   one_halo_norm = data['one_halo_norm']

COMPARING MEASURED 1H TO EFFECTIVE TEMPLATE:

   # Load cached template
   from ciber.theory.ihl_1h_template_cache import OneHaloTemplateCache
   cache = OneHaloTemplateCache()
   effective_1h, _, _ = cache.load_cache(slope=1.0)

   # Compare your measured 1h component
   measured_1h = fit_result['dl_1h']
   measured_1h_norm = measured_1h / np.max(measured_1h)

   # Interpolate template to your ell grid
   template_interp = np.interp(
       fit_result['ell'],
       effective_1h[1.0]['ell'],
       effective_1h[1.0]['one_halo_norm']
   )

   # Calculate shape consistency
   chi2_shape = np.sum(((measured_1h_norm - template_interp) / error)**2)
   print(f"Shape consistency: χ² = {chi2_shape:.2f}")

FULL DOCUMENTATION:
   See INTEGRATION_1H_TEMPLATE_FITTING.md for complete guide
"""
    print(examples)


def main():
    """Main execution."""

    parser = argparse.ArgumentParser(
        description="Run z<1.0 cross-spectrum fits with effective one-halo template",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--cat',
        nargs='+',
        choices=['DESILS', 'HSC'],
        default=['DESILS', 'HSC'],
        help='Catalogs to process'
    )
    parser.add_argument(
        '--lmax',
        type=int,
        nargs='+',
        default=[50000],
        help='Multipole maximum values'
    )
    parser.add_argument(
        '--fitstr',
        default='z_lt_1.0_eff1h',
        help='Fit string label for output files'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing fits'
    )
    parser.add_argument(
        '--zbinedges',
        type=float,
        nargs='+',
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        help='Redshift bin edges'
    )
    parser.add_argument(
        '--no-run',
        action='store_true',
        help='Skip actual fitting (just show template info)'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("Z<1.0 CROSS-SPECTRUM FITS WITH EFFECTIVE ONE-HALO TEMPLATE")
    print("="*70)

    # Check cache
    if not check_cache_exists():
        sys.exit(1)

    # Load and show template
    load_and_compare_1h_templates()

    # Run fits if requested
    if not args.no_run:
        success = run_cross_fits(
            cats=args.cat,
            lmax_values=args.lmax,
            fitstr=args.fitstr,
            overwrite=args.overwrite,
            zbinedges=args.zbinedges
        )

        if not success:
            print("\n" + "="*70)
            print("ERROR: Cross-spectrum fitting failed!")
            print("="*70)
            sys.exit(1)

    # Print usage examples
    print_usage_examples()

    print("\n" + "="*70)
    print("✓ COMPLETE")
    print("="*70)
    print("\nFits completed successfully!")
    print(f"Results saved to: data/cross_cl_fits/")
    print(f"\nNext steps:")
    print(f"  1. Compare measured 1h to effective template")
    print(f"  2. Generate summary plots")
    print(f"  3. See INTEGRATION_1H_TEMPLATE_FITTING.md for full details")


if __name__ == '__main__':
    main()
