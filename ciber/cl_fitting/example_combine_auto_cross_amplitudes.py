"""
Example: Combine galaxy auto and galaxy×CIBER cross A_2h amplitudes.

This script demonstrates how to extract intensity bias information by combining
the 2-halo amplitudes from:
1. Galaxy auto-spectrum fits (A_2h^{gal})
2. Galaxy×CIBER cross-spectrum fits (A_2h^{cross})

The ratio A_2h^{cross} / A_2h^{gal} provides information about the CIBER intensity:
    A_2h^{cross} / A_2h^{gal} = Δz × b_I × dI/dz

where:
- Δz is the redshift bin width
- b_I is the CIBER intensity bias
- dI/dz is the mean intensity per unit redshift

This approach leverages the fact that galaxy bias (b_g) cancels in the ratio,
leaving only intensity-related information.
"""

import numpy as np
import matplotlib.pyplot as plt
from ciber.theory.cross_ps_parametric_model import (
    run_gal_auto_fits_two_stage,
    run_gal_cross_fits,
    combine_auto_cross_A2h_samples
)

# =============================================================================
# Configuration
# =============================================================================
inst_list = [1, 2]
cat = 'HSC'
zbinedges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
lams = [1.1, 1.8]
headstr = 'hsc_ilt24.0'

# =============================================================================
# Step 1: Run galaxy auto-spectrum fits (two-stage)
# =============================================================================
print("="*80)
print("STEP 1: Running galaxy auto-spectrum fits")
print("="*80)

auto_results = run_gal_auto_fits_two_stage(
    inst_list=inst_list,
    cat=cat,
    zbinedges=zbinedges,
    lMax_fit=80000,
    chi2_eval_max=10000,
    fitstr='two_stage_for_ratio',
    figbasedir='figures/auto_cross_ratio/',
    save_figs=True,
    save_results=False,
    ihl_1h_params_path='ihl_1h_params.npz',
    nwalkers=32,
    nsteps_stage1=2000,
    nsteps_stage2=4000,
    nburn_stage1=500,
    nburn_stage2=1000,
    headstr=headstr
)

# =============================================================================
# Step 2: Run galaxy×CIBER cross-spectrum fits
# =============================================================================
print("\n" + "="*80)
print("STEP 2: Running galaxy×CIBER cross-spectrum fits")
print("="*80)

cross_results = run_gal_cross_fits(
    inst_list=inst_list,
    cat=cat,
    zbinedges=zbinedges,
    lMax_fit=80000,
    chi2_eval_max=10000,
    fitstr='for_ratio',
    figbasedir='figures/auto_cross_ratio/',
    save_figs=True,
    save_results=False,
    ihl_1h_params_path='ihl_1h_params.npz',
    nwalkers=32,
    nsteps=4000,
    nburn=1000,
    headstr=headstr
)

# =============================================================================
# Step 3: Combine A_2h samples to extract intensity information
# =============================================================================
print("\n" + "="*80)
print("STEP 3: Combining A_2h samples from auto and cross fits")
print("="*80)

ratio_results = combine_auto_cross_A2h_samples(
    auto_results,
    cross_results,
    inst_list=inst_list,
    zbinedges=zbinedges,
    use_stage2_auto=True  # Use Stage 2 from two-stage auto fits
)

# =============================================================================
# Step 4: Plot results
# =============================================================================
print("\n" + "="*80)
print("STEP 4: Plotting ratio results")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(f'Galaxy Auto vs Cross 2-Halo Amplitudes - {cat}', fontsize=14, y=0.995)

nzbin = len(zbinedges) - 1
zcens = 0.5 * (np.array(zbinedges[:-1]) + np.array(zbinedges[1:]))

for idx, inst in enumerate(inst_list):
    
    # Collect data for this instrument
    ratios, ratio_errs = [], []
    A2h_gal, A2h_gal_err = [], []
    A2h_cross, A2h_cross_err = [], []
    
    for zidx in range(nzbin):
        key = f'inst{inst}_zbin{zidx}'
        if key in ratio_results:
            res = ratio_results[key]
            ratios.append(res['ratio_median'])
            ratio_errs.append(res['ratio_std'])
            A2h_gal.append(res['A2h_gal_median'])
            A2h_gal_err.append(res['A2h_gal_std'])
            A2h_cross.append(res['A2h_cross_median'])
            A2h_cross_err.append(res['A2h_cross_std'])
    
    ratios = np.array(ratios)
    ratio_errs = np.array(ratio_errs)
    A2h_gal = np.array(A2h_gal)
    A2h_gal_err = np.array(A2h_gal_err)
    A2h_cross = np.array(A2h_cross)
    A2h_cross_err = np.array(A2h_cross_err)
    
    # Plot 1: Galaxy auto A_2h
    ax = axes[0, idx]
    ax.errorbar(zcens[:len(A2h_gal)], A2h_gal, yerr=A2h_gal_err,
                fmt='o', capsize=5, label=f'TM{inst} ({lams[idx]:.1f} μm)')
    ax.set_xlabel('Redshift', fontsize=11)
    ax.set_ylabel('$A_{2h}^{gal}$', fontsize=11)
    ax.set_title(f'Galaxy Auto 2h Amplitude - TM{inst}', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Plot 2: Galaxy×CIBER cross A_2h
    ax = axes[1, idx]
    ax.errorbar(zcens[:len(A2h_cross)], A2h_cross, yerr=A2h_cross_err,
                fmt='s', capsize=5, color='C1', label=f'TM{inst} ({lams[idx]:.1f} μm)')
    ax.set_xlabel('Redshift', fontsize=11)
    ax.set_ylabel('$A_{2h}^{cross}$', fontsize=11)
    ax.set_title(f'Galaxy×CIBER Cross 2h Amplitude - TM{inst}', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig('figures/auto_cross_ratio/A2h_amplitudes_vs_z.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot ratio (intensity information)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

for idx, inst in enumerate(inst_list):
    ratios, ratio_errs = [], []
    
    for zidx in range(nzbin):
        key = f'inst{inst}_zbin{zidx}'
        if key in ratio_results:
            res = ratio_results[key]
            ratios.append(res['ratio_median'])
            ratio_errs.append(res['ratio_std'])
    
    ratios = np.array(ratios)
    ratio_errs = np.array(ratio_errs)
    
    ax.errorbar(zcens[:len(ratios)], ratios, yerr=ratio_errs,
                fmt='o' if inst==1 else 's', capsize=5, 
                label=f'TM{inst} ({lams[idx]:.1f} μm)', alpha=0.8)

ax.set_xlabel('Redshift', fontsize=13)
ax.set_ylabel('$A_{2h}^{cross} / A_{2h}^{gal}$', fontsize=13)
ax.set_title(f'2-Halo Amplitude Ratio vs Redshift - {cat}\n' + 
             r'$\propto \Delta z \times b_I \times dI/dz$', fontsize=13)
ax.grid(alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('figures/auto_cross_ratio/A2h_ratio_vs_z.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# Print summary table
# =============================================================================
print("\n" + "="*80)
print("SUMMARY: A_2h Ratio Results")
print("="*80)
print(f"{'Inst':<6} {'z_cen':<8} {'A_2h^gal':<15} {'A_2h^cross':<15} {'Ratio':<15}")
print("-"*80)

for inst in inst_list:
    for zidx in range(nzbin):
        key = f'inst{inst}_zbin{zidx}'
        if key in ratio_results:
            res = ratio_results[key]
            print(f"TM{inst:<4} {res['zcen']:<8.2f} "
                  f"{res['A2h_gal_median']:<7.2e}±{res['A2h_gal_std']:<6.2e} "
                  f"{res['A2h_cross_median']:<7.2e}±{res['A2h_cross_std']:<6.2e} "
                  f"{res['ratio_median']:<7.3e}±{res['ratio_std']:<6.3e}")

print("="*80)

# =============================================================================
# Plot corner plots for one example bin
# =============================================================================
print("\n" + "="*80)
print("Example: Corner plot for TM1, z-bin 0")
print("="*80)

key = 'inst1_zbin0'
if key in ratio_results:
    res = ratio_results[key]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # A_2h^gal distribution
    ax = axes[0, 0]
    ax.hist(res['A2h_gal_samples'], bins=50, alpha=0.7, color='C0', edgecolor='k')
    ax.axvline(res['A2h_gal_median'], color='r', linestyle='--', linewidth=2, label='Median')
    ax.set_xlabel('$A_{2h}^{gal}$', fontsize=11)
    ax.set_ylabel('Counts', fontsize=11)
    ax.set_title('Galaxy Auto 2h Amplitude', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # A_2h^cross distribution
    ax = axes[0, 1]
    ax.hist(res['A2h_cross_samples'], bins=50, alpha=0.7, color='C1', edgecolor='k')
    ax.axvline(res['A2h_cross_median'], color='r', linestyle='--', linewidth=2, label='Median')
    ax.set_xlabel('$A_{2h}^{cross}$', fontsize=11)
    ax.set_ylabel('Counts', fontsize=11)
    ax.set_title('Galaxy×CIBER Cross 2h Amplitude', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Ratio distribution
    ax = axes[1, 0]
    ax.hist(res['ratio_samples'], bins=50, alpha=0.7, color='C2', edgecolor='k')
    ax.axvline(res['ratio_median'], color='r', linestyle='--', linewidth=2, label='Median')
    ax.axvline(res['ratio_percentiles'][0], color='orange', linestyle=':', linewidth=1.5, label='16/84%')
    ax.axvline(res['ratio_percentiles'][2], color='orange', linestyle=':', linewidth=1.5)
    ax.set_xlabel('$A_{2h}^{cross} / A_{2h}^{gal}$', fontsize=11)
    ax.set_ylabel('Counts', fontsize=11)
    ax.set_title('Ratio Distribution', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2D scatter: A_2h^cross vs A_2h^gal
    ax = axes[1, 1]
    ax.scatter(res['A2h_gal_samples'], res['A2h_cross_samples'], 
               alpha=0.3, s=1, c='C3')
    ax.set_xlabel('$A_{2h}^{gal}$', fontsize=11)
    ax.set_ylabel('$A_{2h}^{cross}$', fontsize=11)
    ax.set_title('Sample Correlation', fontsize=12)
    ax.grid(alpha=0.3)
    
    plt.suptitle(f'TM1, z={res["zcen"]:.2f}', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(f'figures/auto_cross_ratio/ratio_distributions_TM1_zbin0.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
