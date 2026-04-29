"""
Example: Using IHL one-halo parameters automatically in run_gal_cross_fits

This demonstrates how run_gal_cross_fits now automatically loads and uses
IHL-derived one-halo parameters by default.

Author: Richard Feder
Date: January 2026
"""

import numpy as np
from ciber.theory.cross_ps_parametric_model import (
    run_gal_cross_fits,
    fit_and_decompose_ihl_templates,
    save_ihl_1h_params
)

print("="*70)
print("Automatic IHL Parameter Integration in run_gal_cross_fits")
print("="*70)

# ============================================================================
# Step 1: Create IHL 1h parameters file (if not already done)
# ============================================================================

print("\nStep 1: Creating IHL 1h parameters file")
print("-"*70)

# Check if the file already exists
import os
ihl_params_file = 'ihl_1h_params.npz'

if not os.path.exists(ihl_params_file):
    print(f"File '{ihl_params_file}' not found. Creating it now...")
    
    # Fit IHL templates
    zbinedges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    results = fit_and_decompose_ihl_templates(
        template_dir='ihl_templates/',
        zbinedges=zbinedges,
        slopes=[1.0],
        plot=False,
        verbose=True
    )
    
    # Save parameters
    save_ihl_1h_params(
        results,
        save_path=ihl_params_file,
        zbinedges=zbinedges,
        slopes=[1.0]
    )
    print(f"\n✓ Created {ihl_params_file}")
else:
    print(f"✓ File '{ihl_params_file}' already exists")

# ============================================================================
# Step 2: Run galaxy cross-fits with automatic IHL parameter loading
# ============================================================================

print("\n" + "="*70)
print("Step 2: Running Galaxy Cross-Fits")
print("="*70)

print("""
Now when you call run_gal_cross_fits(), it will automatically:

1. Look for 'ihl_1h_params.npz' in the current directory
2. Load the IHL-derived one-halo parameters
3. Use them to set priors for phenomenological fits
4. Print information about what parameters are being used

Example call (commented out - adjust parameters for your actual run):
""")

print("""
# For IHL template fitting (uses templates directly)
results = run_gal_cross_fits(
    inst_list=[1, 2],
    ifield_list=[4, 5, 6, 7, 8],
    maskstr='JHlt16_wFFerr',
    cat='DESILS',
    zbinedges=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    use_ihl_templates=True,          # Use IHL templates
    ihl_1h_params_path='ihl_1h_params.npz',  # Path to IHL params (default)
    use_ihl_1h_params=True,          # Enable IHL param loading (default)
    nwalkers=32,
    nsteps=4000,
    nburn=1000
)

# For phenomenological fitting (uses log-normal/Lorentzian model)
# IHL params will automatically set priors!
results = run_gal_cross_fits(
    inst_list=[1, 2],
    ifield_list=[4, 5, 6, 7, 8],
    maskstr='JHlt16_wFFerr',
    cat='DESILS',
    zbinedges=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    use_ihl_templates=False,         # Use phenomenological model
    use_lorentzian_1h=False,         # Use log-normal (not Lorentzian)
    ihl_1h_params_path='ihl_1h_params.npz',  # Automatically loaded
    use_ihl_1h_params=True,          # Priors set from IHL params
    nwalkers=32,
    nsteps=4000,
    nburn=1000
)
""")

# ============================================================================
# Step 3: Understanding the behavior
# ============================================================================

print("\n" + "="*70)
print("Step 3: Understanding the Automatic Behavior")
print("="*70)

print("""
Default Behavior (use_ihl_1h_params=True):
------------------------------------------
1. run_gal_cross_fits looks for 'ihl_1h_params.npz'
2. If found, loads IHL-derived one-halo parameters
3. Prints confirmation message and loaded parameters

For IHL Template Fits (use_ihl_templates=True):
-----------------------------------------------
- Templates are used directly
- IHL 1h params loaded for reference but not directly used in fitting
- Useful for consistency checking

For Phenomenological Fits (use_ihl_templates=False):
----------------------------------------------------
- If mu_1h_prior and sigma_1h_prior are NOT manually specified:
  * Automatically sets priors based on IHL parameters
  * Uses linear relations at mid-redshift
  * Provides sensible prior widths (±30% for mu, ±20% for sigma)
- If priors ARE manually specified:
  * Your manual priors take precedence
  * IHL params still loaded but not used for prior setting

To Disable IHL Parameter Loading:
---------------------------------
Set use_ihl_1h_params=False in your call to run_gal_cross_fits():

results = run_gal_cross_fits(
    ...,
    use_ihl_1h_params=False  # Don't load IHL params
)

To Use Custom IHL Parameter File:
---------------------------------
Specify a different path:

results = run_gal_cross_fits(
    ...,
    ihl_1h_params_path='my_custom_ihl_params.npz'
)
""")

# ============================================================================
# Step 4: Benefits of automatic IHL parameter integration
# ============================================================================

print("\n" + "="*70)
print("Step 4: Benefits")
print("="*70)

print("""
✓ Consistency: All fits use the same IHL-derived constraints
✓ Automation: No need to manually set priors for each fit
✓ Accuracy: Priors based on actual IHL template decomposition
✓ Flexibility: Can still override with manual priors if needed
✓ Backward Compatible: Old code works without IHL params file
✓ Informative: Prints what parameters are being used

The IHL parameters act as informed priors that guide the phenomenological
fits toward physically reasonable one-halo shapes, improving convergence
and preventing unphysical solutions.
""")

# ============================================================================
# Step 5: What gets printed during run_gal_cross_fits
# ============================================================================

print("\n" + "="*70)
print("Step 5: Expected Console Output")
print("="*70)

print("""
When you run run_gal_cross_fits with IHL params, you'll see:

======================================================================
Using IHL-derived one-halo parameters
======================================================================
✓ Loaded one-halo parameters from: ihl_1h_params.npz
  - 5 redshift bins
  - 1 slope value(s): [1.]

Linear relations from IHL template fits:
  Slope 1.0:
    ln(ell_peak) = 8.440 + 7.400 * z
    sigma = 1.560 + 2.430 * z

Automatically set priors from IHL parameters (at z=0.50):
  mu_1h_prior = (12.140, 0.300)
  sigma_1h_prior = (2.775, 0.200)

This confirms:
- IHL params were loaded successfully
- Linear relations extracted from template fits
- Priors automatically calculated for the mid-redshift
""")

print("\n" + "="*70)
print("Setup Complete!")
print("="*70)
print(f"\nYou now have '{ihl_params_file}' ready to use.")
print("\nrun_gal_cross_fits will automatically use these parameters unless you:")
print("  1. Set use_ihl_1h_params=False, or")
print("  2. Manually specify mu_1h_prior and sigma_1h_prior")
print("\nThe integration is seamless and backward-compatible!")
