#!/usr/bin/env python3
"""Plot chi-values by redshift bin and ell range (merged bins).
Also plot chi^2 vs ell like a power spectrum."""

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

# Merged ell ranges
ell_ranges = [
    ("300–1k", 300, 1000),
    ("1k–10k", 1000, 10000),
    ("10k–50k", 10000, 50000),
]

# ============================================================================
# Figure 1: Chi by redshift and merged ell ranges (grouped bars)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for cat_idx, cat in enumerate(["HSC", "DESILS"]):
    print(f"\nProcessing {cat}...")

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
    residuals_array = results.get("residuals", None)
    lb_fit_array = results.get("lb_fit", None)

    # Process each instrument
    for inst_idx, inst in enumerate(inst_list):
        ax = axes[cat_idx * 2 + inst_idx]
        lam = 1.1 if inst == 1 else 1.8

        # For each ell range, collect chi-values across all z-bins
        data_to_plot = {range_name: [] for range_name, _, _ in ell_ranges}
        z_labels = [f"{zbinedges[i]:.1f}–{zbinedges[i+1]:.1f}" for i in range(n_zbins)]

        for zidx in range(n_zbins):
            residuals = residuals_array[inst_idx, zidx]
            lb_fit = lb_fit_array[inst_idx, zidx]

            # Bin residuals by ell range
            for range_name, ell_lo, ell_hi in ell_ranges:
                chi_in_range = []
                for ell_idx, ell in enumerate(lb_fit):
                    if ell_lo <= ell < ell_hi:
                        chi_in_range.append(residuals[ell_idx])
                if chi_in_range:
                    data_to_plot[range_name].append(np.mean(np.abs(chi_in_range)))
                else:
                    data_to_plot[range_name].append(np.nan)

        # Plot as grouped bars
        x = np.arange(n_zbins)
        width = 0.25
        colors = ['C0', 'C1', 'C2']

        for range_idx, (range_name, _, _) in enumerate(ell_ranges):
            offset = (range_idx - 1) * width
            values = data_to_plot[range_name]
            ax.bar(x + offset, values, width, label=range_name, color=colors[range_idx], alpha=0.8)

        # Add horizontal line for expected mean |chi|
        ax.axhline(np.sqrt(np.pi/2), color='red', linestyle='--', linewidth=2,
                   label=f'Expected (1.25)', alpha=0.7)

        ax.set_ylabel('Mean |χ|', fontsize=11)
        ax.set_xlabel('Redshift bin', fontsize=11)
        ax.set_title(f'{cat} TM{inst} (λ={lam} μm)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(z_labels, fontsize=9)
        ax.set_ylim([0, 1.8])
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=9, loc='upper left')

fig.tight_layout()
fig.savefig('figures/chi_by_redshift_merged_ellrange.png', dpi=150, bbox_inches='tight')
print(f"Saved: figures/chi_by_redshift_merged_ellrange.png")
plt.close(fig)

# ============================================================================
# Figure 2: Chi^2 vs ell (like a power spectrum)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for cat_idx, cat in enumerate(["HSC", "DESILS"]):
    headstr = "ilt25.0" if cat == "HSC" else None
    tag = f"_{headstr}" if headstr else ""
    fpath = datadir / f"{cat}_coarsez{tag}_cross_cl_fits_{fitstr}_lMax=50000.npz"

    if not fpath.exists():
        continue

    results = load_fit_results_npz(str(fpath))
    zbinedges = results["zbinedges"]
    inst_list = list(results["inst_list"])
    n_zbins = len(zbinedges) - 1
    residuals_array = results.get("residuals", None)
    lb_fit_array = results.get("lb_fit", None)

    # Process each instrument
    for inst_idx, inst in enumerate(inst_list):
        ax = axes[cat_idx * 2 + inst_idx]
        lam = 1.1 if inst == 1 else 1.8

        # Collect chi^2 values for each ell, averaged across z-bins
        chi2_by_ell = {}  # ell -> list of chi^2 values

        for zidx in range(n_zbins):
            zlo, zhi = zbinedges[zidx], zbinedges[zidx + 1]
            residuals = residuals_array[inst_idx, zidx]
            lb_fit = lb_fit_array[inst_idx, zidx]

            for ell_idx, ell in enumerate(lb_fit):
                ell_key = float(ell)
                if ell_key not in chi2_by_ell:
                    chi2_by_ell[ell_key] = []
                chi2_by_ell[ell_key].append(residuals[ell_idx]**2)

        # Plot chi^2 vs ell
        ells = sorted(chi2_by_ell.keys())
        mean_chi2 = [np.mean(chi2_by_ell[ell]) for ell in ells]
        std_chi2 = [np.std(chi2_by_ell[ell]) for ell in ells]

        ax.errorbar(ells, mean_chi2, yerr=std_chi2, fmt='o-', color='C0',
                    markersize=6, linewidth=2, capsize=3, label='Mean χ²')

        # Add reference lines
        ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Expected (1.0)', alpha=0.7)
        ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

        # Shade ell ranges
        ax.axvspan(300, 1000, alpha=0.1, color='C0', label='300–1k')
        ax.axvspan(1000, 10000, alpha=0.1, color='C1', label='1k–10k')
        ax.axvspan(10000, 50000, alpha=0.1, color='C2', label='10k–50k')

        ax.set_xscale('log')
        ax.set_ylabel('χ²', fontsize=11)
        ax.set_xlabel('ℓ', fontsize=11)
        ax.set_title(f'{cat} TM{inst} (λ={lam} μm)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=9, loc='best')

fig.tight_layout()
fig.savefig('figures/chi2_vs_ell.png', dpi=150, bbox_inches='tight')
print(f"Saved: figures/chi2_vs_ell.png")
plt.close(fig)

# ============================================================================
# Figure 3: Chi^2/dof heatmap with merged ranges (summary table)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for cat_idx, cat in enumerate(["HSC", "DESILS"]):
    headstr = "ilt25.0" if cat == "HSC" else None
    tag = f"_{headstr}" if headstr else ""
    fpath = datadir / f"{cat}_coarsez{tag}_cross_cl_fits_{fitstr}_lMax=50000.npz"

    if not fpath.exists():
        continue

    results = load_fit_results_npz(str(fpath))
    zbinedges = results["zbinedges"]
    inst_list = list(results["inst_list"])
    n_zbins = len(zbinedges) - 1
    reduced_chisq = results.get("reduced_chisq", None)

    # Create heatmap data
    heatmap_data = reduced_chisq.T  # Shape: (n_zbins, n_inst)

    ax = axes[cat_idx]
    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn_r', vmin=0.5, vmax=3.5)

    # Set ticks and labels
    z_labels = [f"{zbinedges[i]:.1f}–{zbinedges[i+1]:.1f}" for i in range(n_zbins)]
    inst_labels = [f'TM{inst}' for inst in inst_list]

    ax.set_xticks(np.arange(len(inst_list)))
    ax.set_yticks(np.arange(n_zbins))
    ax.set_xticklabels(inst_labels)
    ax.set_yticklabels(z_labels)

    ax.set_ylabel('Redshift bin', fontsize=11)
    ax.set_xlabel('Instrument', fontsize=11)
    ax.set_title(f'{cat}: χ²/dof', fontsize=12, fontweight='bold')

    # Add text annotations
    for i in range(n_zbins):
        for j in range(len(inst_list)):
            val = heatmap_data[i, j]
            color = 'white' if val > 1.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('χ²/dof', fontsize=10)

fig.tight_layout()
fig.savefig('figures/chi2dof_heatmap.png', dpi=150, bbox_inches='tight')
print(f"Saved: figures/chi2dof_heatmap.png")
plt.close(fig)

print("\nAll plots saved!")
