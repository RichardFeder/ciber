#!/usr/bin/env python3
"""Interpret chi-values in context of chi2/dof and degrees of freedom."""

import numpy as np
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from ciber.io.ciber_data_utils import load_fit_results_npz

fitstr = "IHL1hfit_fixshape_v6"
datadir = Path("data/cross_cl_fits")

print("\nCHI-VALUE INTERPRETATION GUIDE")
print("="*100)
print("\nFor N_dof degrees of freedom:")
print("  - Expected chi2 from standard normal: N_dof ± sqrt(2*N_dof)")
print("  - Expected chi2/dof: 1.0 ± sqrt(2/N_dof)")
print("  - Expected mean |chi| per point: sqrt(π/2) ≈ 1.25 for good fit")
print("  - But with N_dof data points, you expect some scatter\n")

for cat in ["HSC", "DESILS"]:
    print(f"\n{'='*100}")
    print(f"{cat} - {fitstr}")
    print(f"{'='*100}")

    headstr = "ilt25.0" if cat == "HSC" else None
    tag = f"_{headstr}" if headstr else ""
    fpath = datadir / f"{cat}_coarsez{tag}_cross_cl_fits_{fitstr}_lMax=50000.npz"

    if not fpath.exists():
        print(f"File not found: {fpath}")
        continue

    results = load_fit_results_npz(str(fpath))
    zbinedges = results["zbinedges"]
    inst_list = list(results["inst_list"])
    n_zbins = len(zbinedges) - 1
    reduced_chisq = results.get("reduced_chisq", None)
    residuals_array = results.get("residuals", None)

    if reduced_chisq is None or residuals_array is None:
        print("Missing chi2 or residuals data")
        continue

    print(f"\n{'z range':<15} {'inst':<6} {'χ²/dof':<10} {'N_dof':<8} {'Expected χ²/dof':<20} {'Per-point |χ|':<15}")
    print("-"*100)

    for inst_idx, inst in enumerate(inst_list):
        for zidx in range(n_zbins):
            zlo, zhi = zbinedges[zidx], zbinedges[zidx + 1]

            chi2_dof = reduced_chisq[inst_idx, zidx]
            residuals = residuals_array[inst_idx, zidx]
            n_dof = len(residuals)

            # Expected chi2/dof for a good fit
            expected_chi2_dof = 1.0
            expected_std = np.sqrt(2.0 / n_dof)

            # Mean absolute chi per point
            mean_abs_chi = np.mean(np.abs(residuals))
            expected_mean_abs = np.sqrt(np.pi / 2.0)  # ≈ 1.25

            # Interpretation
            if chi2_dof < 1.0 - 2*expected_std:
                interp = "UNDER-fit (underfitting)"
            elif chi2_dof > 1.0 + 2*expected_std:
                interp = "OVER-fit (data scatter > model)"
            else:
                interp = "GOOD fit"

            print(f"{zlo:.1f}-{zhi:.1f}       TM{inst}    {chi2_dof:6.2f}     {n_dof:3d}      "
                  f"1.0 ± {expected_std:.3f}          {mean_abs_chi:6.3f} (exp: {expected_mean_abs:.3f})  {interp}")

    # Summary statistics
    print("\n" + "="*100)
    print("SUMMARY STATISTICS:")
    print("="*100)

    all_chi2_dof = reduced_chisq[np.isfinite(reduced_chisq)]
    all_mean_abs_chi = []
    for inst_idx in range(len(inst_list)):
        for zidx in range(n_zbins):
            residuals = residuals_array[inst_idx, zidx]
            all_mean_abs_chi.append(np.mean(np.abs(residuals)))

    all_mean_abs_chi = np.array(all_mean_abs_chi)

    print(f"\nχ²/dof across all bins:")
    print(f"  Mean: {np.mean(all_chi2_dof):.3f}")
    print(f"  Std:  {np.std(all_chi2_dof):.3f}")
    print(f"  Min:  {np.min(all_chi2_dof):.3f}")
    print(f"  Max:  {np.max(all_chi2_dof):.3f}")

    print(f"\nMean |chi| per point across all bins:")
    print(f"  Mean: {np.mean(all_mean_abs_chi):.3f} (expected: 1.25 for good fit)")
    print(f"  Std:  {np.std(all_mean_abs_chi):.3f}")
    print(f"  Min:  {np.min(all_mean_abs_chi):.3f}")
    print(f"  Max:  {np.max(all_mean_abs_chi):.3f}")

    print(f"\nInterpretation:")
    mean_chi2_dof = np.mean(all_chi2_dof)
    if mean_chi2_dof > 2.0:
        print(f"  χ²/dof > 2: Data variance is {mean_chi2_dof:.1f}× statistical expectation")
        print(f"  → Either: (1) errors are underestimated")
        print(f"           (2) model is inadequate in some ell ranges")
        print(f"           (3) systematic errors not accounted for")
    elif mean_chi2_dof < 0.5:
        print(f"  χ²/dof < 0.5: Fit too good, likely underfitting or overestimated errors")
    else:
        print(f"  χ²/dof ≈ 1: Reasonably good fit")

    print(f"\n  Mean |chi| = {np.mean(all_mean_abs_chi):.3f}: Individual points are ~1σ scattered")
    print(f"  This is expected for Gaussian residuals; χ-values of 1-3 are normal.")
    print(f"  But χ²/dof integrates these: Σ(chi²)/dof = χ²/dof")
