#!/usr/bin/env python3
"""
Cache the effective one-halo template for use in power spectrum fitting.

This script:
1. Computes the effective 1h template from IHL decomposition
2. Saves it to a cache directory for quick access during fitting
3. Stores individual z-bin templates alongside
4. Creates metadata file for tracking cache contents

Usage:
    python3 scripts/cache_effective_1h_templates.py
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

import json
from scipy.optimize import curve_fit

import config  # noqa: E402

from ciber.theory.ihl_1h_template_cache import create_and_cache_effective_1h_template


def fit_lognormal_to_template(ell, dl):
    """Fit a log-normal to a 1h template and return (mu_1h, sigma_1h)."""
    def lognorm(ell, A, mu, sigma):
        return A * np.exp(-0.5 * ((np.log(ell) - mu) / sigma)**2)
    peak_ell = ell[np.argmax(dl)]
    popt, _ = curve_fit(lognorm, ell, dl,
                        p0=[np.max(dl), np.log(peak_ell), 1.8],
                        maxfev=20000)
    return float(popt[1]), float(popt[2])  # mu_1h, sigma_1h


def main(template_dir='data/ihl_templates', cache_dir='data/1h_template_cache', ell_scale=1.0):
    """Main execution."""

    print("\n" + "="*70)
    print("CACHE EFFECTIVE ONE-HALO TEMPLATES")
    print("="*70)
    if ell_scale != 1.0:
        print(f"Note: ell values will be scaled by {ell_scale}")

    # Configuration
    zbinedges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    slopes = [1.0]
    description = "Effective one-halo template from IHL decomposition (z<1.0, dz=0.2 bins)"

    # Check template directory
    if not os.path.exists(template_dir):
        print(f"\nError: Template directory not found: {template_dir}")
        print(f"Current directory: {os.getcwd()}")
        return False

    # Compute and cache
    try:
        # Compute effective template (which internally uses fit_and_decompose_ihl_templates)
        effective_1h, individual_1h, zbinedges = create_and_cache_effective_1h_template(
            template_dir=template_dir,
            zbinedges=zbinedges,
            slopes=slopes,
            cache_dir=cache_dir,
            description=description,
            plot=True,
            ell_scale=ell_scale
        )

        # Fit log-normal to effective template and store effective parameters
        metadata_path = Path(cache_dir) / 'cache_metadata.json'
        with open(metadata_path) as f:
            metadata = json.load(f)

        print("\nFitting log-normal to effective templates...")
        for slope in slopes:
            eff = effective_1h[slope]
            mu_eff, sigma_eff = fit_lognormal_to_template(eff['ell'], eff['one_halo_sum'])
            print(f"  slope={slope}: mu_1h={mu_eff:.4f} (ell_peak={np.exp(mu_eff):.0f}), sigma_1h={sigma_eff:.4f}")
            slope_str = str(float(slope))
            metadata['cached_slopes'][slope_str]['effective_mu_1h'] = mu_eff
            metadata['cached_slopes'][slope_str]['effective_sigma_1h'] = sigma_eff

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✓ Effective mu_1h/sigma_1h stored in cache_metadata.json")

        print("\n" + "="*70)
        print("CACHE CREATION SUCCESSFUL")
        print("="*70)
        print(f"\nCache location: {cache_dir}")

        return True

    except Exception as e:
        print(f"\nError during cache creation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Cache effective one-halo template",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--template-dir',
        default='data/ihl_templates',
        help='Directory containing IHL template files'
    )
    parser.add_argument(
        '--cache-dir',
        default='data/1h_template_cache',
        help='Directory to save cached templates'
    )
    parser.add_argument(
        '--ell-scale',
        type=float,
        default=1.0,
        help='Scaling factor for ell values when reading IHL templates'
    )
    args = parser.parse_args()

    success = main(
        template_dir=args.template_dir,
        cache_dir=args.cache_dir,
        ell_scale=args.ell_scale
    )
    sys.exit(0 if success else 1)
