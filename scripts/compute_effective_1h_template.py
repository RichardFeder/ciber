#!/usr/bin/env python3
"""
Compute effective one-halo template by summing unnormalized 1h components
from dz=0.2 redshift bins and compare to redshift-sliced measurements.

This script:
1. Loads IHL power spectrum templates for each dz=0.2 redshift bin
2. Decomposes each template into 2h, 1h, and shot noise components
3. Extracts the unnormalized 1h components (amplitude and shape)
4. Sums/averages the 1h components across all redshift bins
5. Compares the effective template to individual redshift bin measurements
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
import config  # noqa: E402

from ciber.theory.cl_template import (
    load_ihl_templates,
    fit_and_decompose_ihl_templates,
    get_ihl_components_at_ell,
)


def compute_effective_1h_template(template_dir, zbinedges, slopes=[1.0],
                                   ell_fit_range=None, plot=True,
                                   figsize=(14, 10), ell_scale=1.0):
    """
    Compute effective one-halo template by summing components from dz=0.2 bins.

    Parameters
    ----------
    template_dir : str
        Directory containing IHL template files
    zbinedges : array_like
        Redshift bin edges (e.g., [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    slopes : list of float
        Slope values to process (default [1.0])
    ell_fit_range : tuple, optional
        (ell_min, ell_max) for fitting range
    plot : bool
        Whether to create comparison plots
    figsize : tuple
        Figure size for plots

    Returns
    -------
    effective_1h : dict
        Dictionary with effective 1h template:
        - 'ell': multipole values
        - 'one_halo_sum': sum of unnormalized 1h components
        - 'one_halo_avg': average of unnormalized 1h components
        - 'one_halo_norm': normalized version (sum/max)
    individual_1h : dict
        Dictionary with 1h components for each redshift bin
    fit_results : dict
        Fitting results from decomposition
    """

    zbinedges = np.array(zbinedges)
    nzbins = len(zbinedges) - 1

    print("="*70)
    print("EFFECTIVE ONE-HALO TEMPLATE COMPUTATION")
    print("="*70)
    print(f"\nRedshift bins: {zbinedges}")
    print(f"Number of bins: {nzbins}")
    print(f"Slopes: {slopes}")
    print(f"Template directory: {template_dir}")

    # Fit and decompose IHL templates for all redshift bins
    print("\n" + "-"*70)
    print("FITTING IHL TEMPLATES FOR EACH REDSHIFT BIN")
    print("-"*70)

    fit_results = fit_and_decompose_ihl_templates(
        template_dir=template_dir,
        zbinedges=zbinedges,
        slopes=slopes,
        use_powerlaw_2h=True,
        alpha_2h_fixed=0.0,
        fit_ell_range=ell_fit_range,
        plot=False,  # We'll plot separately for clarity
        verbose=True,
        ell_scale=ell_scale
    )

    fits = fit_results['fits']

    # Extract 1h components for each redshift bin
    print("\n" + "-"*70)
    print("EXTRACTING ONE-HALO COMPONENTS FROM FITS")
    print("-"*70)

    # Use a common ell grid for comparison
    # Find the first successful fit (skip ones with errors)
    first_fit = None
    for fit in fits.values():
        if 'error' not in fit:
            first_fit = fit
            break

    if first_fit is None:
        raise ValueError("No successful fits found")

    ell_grid = first_fit['ell_eval']

    individual_1h = {}
    all_1h_components = []

    for slope in slopes:
        individual_1h[slope] = {}

        for zidx in range(nzbins):
            zlow = zbinedges[zidx]
            zhigh = zbinedges[zidx + 1]
            z_mid = 0.5 * (zlow + zhigh)

            fit_key = f"z{zlow}_{zhigh}_slope{slope}"

            if fit_key not in fits:
                print(f"  Warning: Fit result for {fit_key} not found, skipping")
                continue

            fit_result = fits[fit_key]

            # Check if fit was successful
            if 'error' in fit_result:
                print(f"  Warning: Fit for {fit_key} has error: {fit_result['error']}, skipping")
                continue

            # Get components already evaluated at ell_eval
            components = fit_result['components']
            one_halo = components['one_halo']
            ell_eval = fit_result['ell_eval']

            # If ell grids don't match, interpolate to common grid
            if not np.allclose(ell_eval, ell_grid):
                one_halo = np.interp(ell_grid, ell_eval, one_halo)

            individual_1h[slope][zidx] = {
                'z_range': (zlow, zhigh),
                'z_mid': z_mid,
                'ell': ell_grid,
                'one_halo': one_halo,
                'A_1h': fit_result['params'][1],
                'mu_1h': fit_result['params'][2],
                'sigma_1h': fit_result['params'][3],
            }

            all_1h_components.append(one_halo)

            print(f"  z∈[{zlow}, {zhigh}]: A_1h={fit_result['params'][1]:.2e}, "
                  f"μ_1h={np.exp(fit_result['params'][2]):.0f}, "
                  f"σ_1h={fit_result['params'][3]:.3f}")

    # Compute effective template by summing unnormalized 1h components
    print("\n" + "-"*70)
    print("COMPUTING EFFECTIVE ONE-HALO TEMPLATE")
    print("-"*70)

    effective_1h = {}

    for slope in slopes:
        all_1h_components_slope = np.array([
            individual_1h[slope][zidx]['one_halo']
            for zidx in range(nzbins) if zidx in individual_1h[slope]
        ])

        # Sum unnormalized components
        one_halo_sum = np.sum(all_1h_components_slope, axis=0)

        # Average unnormalized components
        one_halo_avg = np.mean(all_1h_components_slope, axis=0)

        # Normalize to max
        one_halo_norm = one_halo_sum / np.max(one_halo_sum)

        effective_1h[slope] = {
            'ell': ell_grid,
            'one_halo_sum': one_halo_sum,
            'one_halo_avg': one_halo_avg,
            'one_halo_norm': one_halo_norm,
            'n_bins_summed': len(all_1h_components_slope),
        }

        print(f"\n  Slope {slope}:")
        print(f"    Summed {len(all_1h_components_slope)} redshift bins")
        print(f"    Peak of sum: {np.max(one_halo_sum):.3e} at ell={ell_grid[np.argmax(one_halo_sum)]:.0f}")
        print(f"    Peak of avg: {np.max(one_halo_avg):.3e} at ell={ell_grid[np.argmax(one_halo_avg)]:.0f}")

    # Create comparison plots
    if plot:
        print("\n" + "-"*70)
        print("CREATING COMPARISON PLOTS")
        print("-"*70)

        create_comparison_plots(
            effective_1h, individual_1h, zbinedges,
            figsize=figsize
        )

    return effective_1h, individual_1h, fit_results


def create_comparison_plots(effective_1h, individual_1h, zbinedges,
                            figsize=(14, 10)):
    """Create plots comparing effective vs individual 1h templates."""

    slopes = list(effective_1h.keys())

    for slope in slopes:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'One-Halo Template Comparison (Slope={slope})',
                     fontsize=16, fontweight='bold')

        eff = effective_1h[slope]
        ell = eff['ell']
        one_halo_sum = eff['one_halo_sum']
        one_halo_avg = eff['one_halo_avg']
        one_halo_norm = eff['one_halo_norm']

        # Plot 1: Unnormalized components (linear scale)
        ax = axes[0, 0]
        nzbins = len(zbinedges) - 1
        colors = plt.cm.viridis(np.linspace(0, 1, nzbins))

        for zidx in range(nzbins):
            if zidx in individual_1h[slope]:
                z_info = individual_1h[slope][zidx]
                label = f"z∈[{z_info['z_range'][0]:.1f}, {z_info['z_range'][1]:.1f}]"
                ax.plot(z_info['ell'], z_info['one_halo'], color=colors[zidx],
                       alpha=0.6, label=label, linewidth=1.5)

        ax.plot(ell, one_halo_sum, color='red', linewidth=2.5,
               label='Sum (Effective)', linestyle='-', zorder=10)
        ax.set_xlabel('Multipole (ℓ)', fontsize=12)
        ax.set_ylabel('D_ℓ (1h component)', fontsize=12)
        ax.set_title('Unnormalized 1h Components (Linear Scale)', fontsize=13)
        ax.set_yscale('linear')
        ax.legend(fontsize=9, loc='upper right', ncol=2)
        ax.grid(alpha=0.3)

        # Plot 2: Unnormalized components (log scale)
        ax = axes[0, 1]
        for zidx in range(nzbins):
            if zidx in individual_1h[slope]:
                z_info = individual_1h[slope][zidx]
                ax.loglog(z_info['ell'], np.abs(z_info['one_halo']) + 1e-10,
                         color=colors[zidx], alpha=0.6, linewidth=1.5)

        ax.loglog(ell, one_halo_sum, color='red', linewidth=2.5,
                 label='Sum (Effective)', linestyle='-', zorder=10)
        ax.set_xlabel('Multipole (ℓ)', fontsize=12)
        ax.set_ylabel('|D_ℓ (1h component)|', fontsize=12)
        ax.set_title('Unnormalized 1h Components (Log Scale)', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, which='both')

        # Plot 3: Sum vs Average
        ax = axes[1, 0]
        ax.semilogy(ell, one_halo_sum, color='red', linewidth=2.5,
                   label='Sum (Effective)', linestyle='-')
        ax.semilogy(ell, one_halo_avg, color='blue', linewidth=2.5,
                   label='Average', linestyle='--')
        ax.set_xlabel('Multipole (ℓ)', fontsize=12)
        ax.set_ylabel('D_ℓ (1h component)', fontsize=12)
        ax.set_title('Sum vs Average of 1h Components', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        # Plot 4: Normalized effective template
        ax = axes[1, 1]
        ax.semilogx(ell, one_halo_norm, color='darkred', linewidth=2.5,
                   label='Normalized Sum', linestyle='-')

        # Overlay normalized individual bins
        for zidx in range(nzbins):
            if zidx in individual_1h[slope]:
                z_info = individual_1h[slope][zidx]
                one_halo_norm_bin = z_info['one_halo'] / np.max(z_info['one_halo'])
                ax.semilogx(z_info['ell'], one_halo_norm_bin, color=colors[zidx],
                           alpha=0.4, linewidth=1, linestyle=':')

        ax.set_xlabel('Multipole (ℓ)', fontsize=12)
        ax.set_ylabel('Normalized 1h Shape', fontsize=12)
        ax.set_title('Normalized Effective Template (Colored: Individual Bins)', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_ylim([-0.05, 1.1])

        plt.tight_layout()
        plt.savefig(f'figures/effective_1h_template_slope{slope:.1f}.png',
                   dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: figures/effective_1h_template_slope{slope:.1f}.png")
        plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute effective one-halo template from IHL decomposition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--template-dir',
        default='data/ihl_templates',
        help='Directory containing IHL template files'
    )
    parser.add_argument(
        '--ell-scale',
        type=float,
        default=1.0,
        help='Scaling factor for ell values (e.g., 0.31831 for 1/π correction)'
    )
    args = parser.parse_args()

    # Configuration
    template_dir = args.template_dir
    zbinedges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    slopes = [1.0]

    # Check if template directory exists
    if not os.path.exists(template_dir):
        print(f"Error: Template directory '{template_dir}' not found")
        print(f"Current directory: {os.getcwd()}")
        sys.exit(1)

    # Compute effective template
    effective_1h, individual_1h, fit_results = compute_effective_1h_template(
        template_dir=template_dir,
        zbinedges=zbinedges,
        slopes=slopes,
        plot=True,
        figsize=(14, 10),
        ell_scale=args.ell_scale
    )

    print("\n" + "="*70)
    print("COMPUTATION COMPLETE")
    print("="*70)
    print(f"\nEffective 1h template computed and saved to figures/")
    print(f"Individual component analysis available for all {len(zbinedges)-1} redshift bins")
    if args.ell_scale != 1.0:
        print(f"Note: ell values were scaled by {args.ell_scale}")
