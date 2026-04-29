"""
Example script demonstrating how to load and decompose IHL templates 
into two-halo, one-halo, and shot noise contributions.

Author: Richard Feder
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from ciber.theory.cross_ps_parametric_model import fit_and_decompose_ihl_templates

# Set up paths
template_dir = '/Users/richardfeder/Documents/ciber/ihl_templates'

# Define redshift bins matching your template files
zbinedges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# Run the decomposition
print("="*70)
print("IHL Template Decomposition Example")
print("="*70)

results = fit_and_decompose_ihl_templates(
    template_dir=template_dir,
    zbinedges=zbinedges,
    slopes=[1.0],  # Your templates use slope=1.0
    use_powerlaw_2h=True,  # Model 2-halo as power law
    alpha_2h_fixed=-1.5,  # Linear clustering index
    fit_ell_range=(500, 10000),  # Fit range (optional)
    plot=True,  # Create diagnostic plots
    figsize=(15, 10),
    save_path='ihl_decomposition_example.png',  # Save figure
    verbose=True
)

# ============================================================================
# Access and use the results
# ============================================================================

print("\n" + "="*70)
print("Using the Results")
print("="*70)

# 1. Print summary table
print("\n1. Summary of all fits:")
print(results['summary'])

# 2. Access individual template fits
print("\n2. Accessing individual template components:")
for template_name, fit_result in list(results['fits'].items())[:2]:  # First 2 templates
    print(f"\n   Template: {template_name}")
    if 'error' in fit_result:
        print(f"   Error: {fit_result['error']}")
        continue
    
    params = fit_result['params']
    print(f"   Two-halo amplitude: {params[0]:.3e}")
    print(f"   One-halo amplitude: {params[1]:.3e}")
    print(f"   One-halo peak location: ℓ ~ {np.exp(params[2]):.0f}")
    print(f"   Shot noise amplitude: {params[4]:.3e}")
    
    # Components are available at fine ell resolution
    components = fit_result['components']
    print(f"   Components available: {list(components.keys())}")

# 3. Create custom plot for a specific template
print("\n3. Creating custom plot for z=0.0-0.2 template...")
template_name = 'z0.0_0.2_slope1.0'
if template_name in results['fits']:
    fit = results['fits'][template_name]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left panel: Full decomposition
    ax = axes[0]
    ax.loglog(fit['ell_template'], fit['dl_template'], 
             'ko', markersize=8, label='IHL Template', alpha=0.7)
    ax.loglog(fit['ell_eval'], fit['components']['two_halo'], 
             '--', linewidth=2.5, label='Two-halo', color='blue')
    ax.loglog(fit['ell_eval'], fit['components']['one_halo'], 
             '--', linewidth=2.5, label='One-halo', color='green')
    ax.loglog(fit['ell_eval'], fit['components']['shot_noise'], 
             '--', linewidth=2.5, label='Shot noise', color='red')
    ax.loglog(fit['ell_eval'], fit['components']['total'], 
             '-', linewidth=3, label='Total Fit', color='orange')
    ax.set_xlabel(r'Multipole $\ell$', fontsize=13)
    ax.set_ylabel(r'$D_\ell$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=13)
    ax.set_title(f'IHL Template Decomposition: z=[0.0, 0.2]', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(alpha=0.3, which='both')
    
    # Right panel: Fractional contributions
    ax = axes[1]
    total = fit['components']['total']
    ell_eval = fit['ell_eval']
    ax.semilogx(ell_eval, fit['components']['two_halo']/total, 
               linewidth=2.5, label='Two-halo', color='blue')
    ax.semilogx(ell_eval, fit['components']['one_halo']/total, 
               linewidth=2.5, label='One-halo', color='green')
    ax.semilogx(ell_eval, fit['components']['shot_noise']/total, 
               linewidth=2.5, label='Shot noise', color='red')
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    ax.set_xlabel(r'Multipole $\ell$', fontsize=13)
    ax.set_ylabel(r'Fractional Contribution to $D_\ell$', fontsize=13)
    ax.set_title('Component Fractions', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig('ihl_custom_plot_example.png', dpi=200, bbox_inches='tight')
    print(f"   ✓ Saved custom plot to: ihl_custom_plot_example.png")
    plt.close()

# 4. Analyze redshift evolution of components
print("\n4. Analyzing redshift evolution...")
if len(results['summary']) > 0:
    summary = results['summary']
    if 'z_center' in summary.columns:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Two-halo amplitude vs redshift
        ax = axes[0, 0]
        ax.errorbar(summary['z_center'], summary['A_2h'], 
                   yerr=summary['A_2h_err'],
                   fmt='o-', markersize=8, linewidth=2, capsize=5)
        ax.set_xlabel('Redshift', fontsize=12)
        ax.set_ylabel(r'$A_{\rm 2h}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=12)
        ax.set_title('Two-Halo Amplitude Evolution', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # One-halo amplitude vs redshift
        ax = axes[0, 1]
        ax.errorbar(summary['z_center'], summary['A_1h'], 
                   yerr=summary['A_1h_err'],
                   fmt='o-', markersize=8, linewidth=2, capsize=5, color='green')
        ax.set_xlabel('Redshift', fontsize=12)
        ax.set_ylabel(r'$A_{\rm 1h}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=12)
        ax.set_title('One-Halo Amplitude Evolution', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Peak location vs redshift
        ax = axes[1, 0]
        ax.plot(summary['z_center'], summary['ell_peak'], 
               'o-', markersize=8, linewidth=2, color='purple')
        ax.set_xlabel('Redshift', fontsize=12)
        ax.set_ylabel(r'$\ell_{\rm peak}$ (One-Halo)', fontsize=12)
        ax.set_title('One-Halo Peak Location Evolution', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Shot noise amplitude vs redshift
        ax = axes[1, 1]
        ax.errorbar(summary['z_center'], summary['A_shot'], 
                   yerr=summary['A_shot_err'],
                   fmt='o-', markersize=8, linewidth=2, capsize=5, color='red')
        ax.set_xlabel('Redshift', fontsize=12)
        ax.set_ylabel(r'$A_{\rm shot}$', fontsize=12)
        ax.set_title('Shot Noise Amplitude Evolution', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ihl_redshift_evolution.png', dpi=200, bbox_inches='tight')
        print(f"   ✓ Saved redshift evolution plot to: ihl_redshift_evolution.png")
        plt.close()

# 5. Save results to file
print("\n5. Saving results to numpy file...")
np.savez('ihl_decomposition_results.npz',
         summary=results['summary'].to_dict('list'),
         templates=list(results['templates'].keys()),
         allow_pickle=True)
print("   ✓ Saved results to: ihl_decomposition_results.npz")

print("\n" + "="*70)
print("Example Complete!")
print("="*70)
print("\nGenerated files:")
print("  - ihl_decomposition_example.png     : Main decomposition plots")
print("  - ihl_custom_plot_example.png       : Custom analysis for z=0.0-0.2")
print("  - ihl_redshift_evolution.png        : Evolution of components with redshift")
print("  - ihl_decomposition_results.npz     : Numerical results")
