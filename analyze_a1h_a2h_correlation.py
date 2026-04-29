#!/usr/bin/env python3
"""Analyze A_1h vs A_2h anticorrelation from MCMC posteriors."""

import numpy as np
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from ciber.io.ciber_data_utils import load_fit_results_npz

fitstr = "IHL1hfit_fixshape_v6"
datadir = Path("data/cross_cl_fits")

# Collect all results
all_data = []

for cat in ["HSC", "DESILS"]:
    headstr = "ilt25.0" if cat == "HSC" else None
    tag = f"_{headstr}" if headstr else ""
    fpath = datadir / f"{cat}_coarsez{tag}_cross_cl_fits_{fitstr}_lMax=50000.npz"

    if not fpath.exists():
        print(f"File not found: {fpath}")
        continue

    results = load_fit_results_npz(str(fpath))
    zbinedges = results["zbinedges"]
    n_zbins = len(zbinedges) - 1
    inst_list = list(results["inst_list"])

    samples_array = results.get("samples", None)

    if samples_array is None:
        print("No MCMC samples found")
        continue

    for inst_idx, inst in enumerate(inst_list):
        for zidx in range(n_zbins):
            zlo, zhi = zbinedges[zidx], zbinedges[zidx + 1]
            samples = samples_array[inst_idx, zidx]

            if samples is None or len(samples) == 0:
                corr = np.nan
            else:
                a2h_samples = samples[:, 0]
                a1h_samples = samples[:, 1]
                corr_matrix = np.corrcoef(a2h_samples, a1h_samples)
                corr = corr_matrix[0, 1]

            all_data.append({
                'cat': cat,
                'inst': inst,
                'z_lo': zlo,
                'z_hi': zhi,
                'corr': corr
            })

# Print table
print(f"\n{'Catalog':<10} {'Inst':<6} {'z range':<12} {'r(A_2h, A_1h)':<15}")
print("-" * 50)

for row in all_data:
    z_str = f"{row['z_lo']:.1f}-{row['z_hi']:.1f}"
    corr_str = f"{row['corr']:+.4f}" if not np.isnan(row['corr']) else "N/A"
    print(f"{row['cat']:<10} TM{row['inst']:<4} {z_str:<12} {corr_str:<15}")

# Plot r(A_2h, A_1h) vs redshift
print("\n\nGenerating plot...")

fig, ax = plt.subplots(figsize=(8, 5))

colors = {'HSC': {'TM1': 'C0', 'TM2': 'C1'}, 'DESILS': {'TM1': 'C2', 'TM2': 'C3'}}
linestyles = {'HSC': '-', 'DESILS': '--'}

for cat in ["HSC", "DESILS"]:
    for inst in [1, 2]:
        # Extract z-midpoints and correlations for this cat/inst combination
        z_mids = []
        corrs = []
        for row in all_data:
            if row['cat'] == cat and row['inst'] == inst:
                z_mids.append(0.5 * (row['z_lo'] + row['z_hi']))
                corrs.append(row['corr'])

        if corrs:
            label = f"{cat} TM{inst}"
            ax.plot(z_mids, corrs, 'o-', color=colors[cat][f'TM{inst}'],
                    linestyle=linestyles[cat], linewidth=2, markersize=6, label=label)

ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.set_xlabel('Redshift (bin center)', fontsize=11)
ax.set_ylabel(r'$r(A_{2h}, A_{1h})$', fontsize=11)
ax.set_title(f'{fitstr}: A_2h vs A_1h Correlation Coefficient', fontsize=12)
ax.grid(True, alpha=0.3, which='major')
ax.legend(loc='best', fontsize=10)
ax.set_ylim([-1.0, 0.2])
ax.set_xlim([0.0, 1.0])

fig.tight_layout()
fig.savefig('figures/a1h_a2h_correlation_vs_redshift.png', dpi=150, bbox_inches='tight')
print(f"Saved: figures/a1h_a2h_correlation_vs_redshift.png")
