"""Compute and cache effective galaxy bias b_g for DESI-LS across redshift bins.

Runs three binning schemes used in the paper:
  - fine:   dz=0.1 bins from 0.0 to 1.0
  - coarse: dz=0.2 bins from 0.0 to 1.0
  - full:   single bin 0.0 to 1.0

Fits a smooth polynomial to the fine-bin b_eff(z) and saves all results
to a single .npz cache file.
"""

import numpy as np
import os
import sys
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ciber.cross_correlation.galaxy_cross import compute_effective_bias_ls


def fit_polynomial(z_centers, b_eff, b_eff_err, deg=3):
    """Fit a weighted polynomial to b_eff(z).

    Returns fit coefficients (poly1d convention, highest power first)
    and a callable that evaluates the fit at arbitrary z.
    """
    weights = 1.0 / b_eff_err
    # numpy polyfit: coefficients in descending power order
    coeffs = np.polyfit(z_centers, b_eff, deg=deg, w=weights)
    poly_fn = np.poly1d(coeffs)
    return coeffs, poly_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default=None,
                        help='Directory to write cache .npz (default: scripts dir)')
    parser.add_argument('--outname', default='effective_bias_ls_cache.npz',
                        help='Output filename (default: effective_bias_ls_cache.npz)')
    parser.add_argument('--poly-deg', type=int, default=2,
                        help='Polynomial degree for b_eff(z) smooth fit (default: 2)')
    parser.add_argument('--survey-area', type=float, default=19400.0,
                        help='LS DR8 survey area in deg^2 (default: 19400)')
    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, args.outname)

    # ------------------------------------------------------------------ #
    # Three binning schemes
    # ------------------------------------------------------------------ #
    zbins = {
        'fine':   np.arange(0.0, 1.1, 0.1),
        'coarse': np.arange(0.0, 1.2, 0.2),
        'full':   np.array([0.0, 1.0]),
    }

    results = {}
    for label, edges in zbins.items():
        print(f'\n--- {label} binning ({edges[0]:.1f} < z < {edges[-1]:.1f}, '
              f'{len(edges)-1} bins) ---')
        b_eff, b_eff_err, z_centers = compute_effective_bias_ls(
            edges, survey_area_deg2=args.survey_area)
        for z, b, berr in zip(z_centers, b_eff, b_eff_err):
            print(f'  z={z:.2f}: b_eff = {b:.3f} +/- {berr:.3f}')
        results[label] = dict(zbinedges=edges, z_centers=z_centers,
                              b_eff=b_eff, b_eff_err=b_eff_err)

    # ------------------------------------------------------------------ #
    # Polynomial fit to fine-bin results
    # ------------------------------------------------------------------ #
    fine = results['fine']
    good = np.isfinite(fine['b_eff']) & np.isfinite(fine['b_eff_err'])
    coeffs, poly_fn = fit_polynomial(
        fine['z_centers'][good], fine['b_eff'][good],
        fine['b_eff_err'][good], deg=args.poly_deg)

    z_eval = np.linspace(fine['zbinedges'][0], fine['zbinedges'][-1], 200)
    b_eff_smooth = poly_fn(z_eval)

    print(f'\n--- Polynomial fit (degree {args.poly_deg}) to fine-bin b_eff(z) ---')
    print(f'  coefficients (highest power first): {coeffs}')

    # ------------------------------------------------------------------ #
    # Save cache
    # ------------------------------------------------------------------ #
    save_dict = dict(
        survey_area_deg2=args.survey_area,
        poly_deg=args.poly_deg,
        poly_coeffs=coeffs,
        poly_z_eval=z_eval,
        poly_b_eff_smooth=b_eff_smooth,
    )
    for label, res in results.items():
        for key, val in res.items():
            save_dict[f'{label}_{key}'] = val

    # ------------------------------------------------------------------ #
    # Plots — also populates per-binning poly coeffs and model values
    # ------------------------------------------------------------------ #
    for label, dz_label in [('fine', r'$\delta z = 0.1$'), ('coarse', r'$\delta z = 0.2$')]:
        res = results[label]
        good = np.isfinite(res['b_eff']) & np.isfinite(res['b_eff_err'])

        # Fit polynomial independently for each binning scheme
        coeffs_plt, poly_plt = fit_polynomial(
            res['z_centers'][good], res['b_eff'][good],
            res['b_eff_err'][good], deg=args.poly_deg)
        z_curve = np.linspace(res['zbinedges'][0], res['zbinedges'][-1], 300)
        b_curve = poly_plt(z_curve)

        # Store best-fit model evaluated at bin centers and save to cache
        b_eff_model = poly_plt(res['z_centers'])
        save_dict[f'{label}_poly_coeffs'] = coeffs_plt
        save_dict[f'{label}_b_eff_model'] = b_eff_model

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.errorbar(res['z_centers'], res['b_eff'], yerr=res['b_eff_err'],
                    fmt='o', color='steelblue', capsize=3, ms=5,
                    label='Measured $b_\\mathrm{eff}$')
        ax.plot(z_curve, b_curve, color='tomato', lw=1.5,
                label=f'Poly fit (deg {args.poly_deg})')
        ax.set_xlabel('Redshift $z$', fontsize=12)
        ax.set_ylabel('Effective bias $b_g$', fontsize=12)
        ax.set_title(f'DESI-LS DR8 effective bias — {dz_label}', fontsize=11)
        ax.legend(fontsize=10)
        ax.set_xlim(res['zbinedges'][0] - 0.05, res['zbinedges'][-1] + 0.05)
        ax.set_ylim(bottom=0)
        fig.tight_layout()

        plotpath = os.path.join(args.outdir,
                                f'effective_bias_ls_{label}.png')
        fig.savefig(plotpath, dpi=150)
        plt.close(fig)
        print(f'Plot saved: {plotpath}')

    np.savez(outpath, **save_dict)
    print(f'\nCached results written to: {outpath}')


if __name__ == '__main__':
    main()
