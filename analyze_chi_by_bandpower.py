#!/usr/bin/env python3
"""Analyze chi-values by bandpower across redshift bins.

The model_dl array contains trimmed data (lMax cut applied), while data_dl
is the full untrimmed spectrum. We compute chi from the residuals array
which is already the (data - model) / error at the fitted ell points.
"""

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

for cat in ["HSC", "DESILS"]:
    print(f"\n{'='*90}")
    print(f"{cat} - {fitstr}")
    print(f"{'='*90}")

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

    lb_fit_array = results.get("lb_fit", None)
    residuals_array = results.get("residuals", None)

    if residuals_array is None:
        print("residuals not in results, skipping")
        continue

    for inst_idx, inst in enumerate(inst_list):
        print(f"\nTM{inst} (λ = {1.1 if inst == 1 else 1.8} μm):")
        print("-" * 90)
        print(f"{'ell':<8} {'ell_range':<25} {'mean |chi|':<15} {'std chi':<15} {'max |chi|':<15}")
        print("-" * 90)

        # Collect chi values for each bandpower across all redshift bins
        all_chi_by_ell = {}  # (ell_idx, ell_value) -> list of chi values

        for zidx in range(n_zbins):
            zlo, zhi = zbinedges[zidx], zbinedges[zidx + 1]

            residuals = residuals_array[inst_idx, zidx]
            if lb_fit_array is not None:
                lb_fit = lb_fit_array[inst_idx, zidx]
            else:
                lb_fit = np.arange(len(residuals))

            # Store by ell index
            for ell_idx, ell in enumerate(lb_fit):
                key = (ell_idx, ell)
                if key not in all_chi_by_ell:
                    all_chi_by_ell[key] = []
                all_chi_by_ell[key].append(residuals[ell_idx])

        # Analyze and print by ell
        for (ell_idx, ell) in sorted(all_chi_by_ell.keys(), key=lambda x: x[1]):
            chi_vals = np.array(all_chi_by_ell[(ell_idx, ell)])

            mean_abs_chi = np.mean(np.abs(chi_vals))
            std_chi = np.std(chi_vals)
            max_abs_chi = np.max(np.abs(chi_vals))

            # Categorize ell range
            if ell < 500:
                ell_range = "Low (ℓ < 500)"
            elif ell < 1000:
                ell_range = "Low-Mid (500-1k)"
            elif ell < 10000:
                ell_range = "Intermediate (1k-10k)"
            else:
                ell_range = "High (ℓ ≥ 10k)"

            print(f"{ell:7.0f}  {ell_range:<25} {mean_abs_chi:+.4f}          {std_chi:+.4f}          {max_abs_chi:+.4f}")

        # Summary statistics by ell range
        print("\n" + "="*90)
        print("SUMMARY BY ELL RANGE:")
        print("="*90)

        ell_ranges = [
            ("Low (ℓ < 500)", 0, 500),
            ("Low-Mid (500 ≤ ℓ < 1000)", 500, 1000),
            ("Intermediate (1000 ≤ ℓ < 10000)", 1000, 10000),
            ("High (ℓ ≥ 10000)", 10000, np.inf),
        ]

        print(f"{'Range':<35} {'mean |chi|':<15} {'std chi':<15} {'max |chi|':<15} {'# bandpowers':<15}")
        print("-" * 90)

        for range_name, ell_lo, ell_hi in ell_ranges:
            chi_in_range = []
            count_in_range = 0
            for (ell_idx, ell) in all_chi_by_ell:
                if ell_lo <= ell < ell_hi:
                    chi_in_range.extend(all_chi_by_ell[(ell_idx, ell)])
                    count_in_range += 1

            if chi_in_range:
                chi_arr = np.array(chi_in_range)
                mean_abs = np.mean(np.abs(chi_arr))
                std_all = np.std(chi_arr)
                max_abs = np.max(np.abs(chi_arr))
                print(f"{range_name:<35} {mean_abs:+.4f}          {std_all:+.4f}          {max_abs:+.4f}          {count_in_range:2d}")
