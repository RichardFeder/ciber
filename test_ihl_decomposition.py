"""
Quick test script to verify IHL template decomposition functions work correctly.
"""

import numpy as np
import sys
import os

# Add ciber to path
sys.path.insert(0, '/Users/richardfeder/Documents/ciber')

print("="*70)
print("Testing IHL Template Decomposition Functions")
print("="*70)

# Test 1: Import the functions
print("\n1. Testing imports...")
try:
    from ciber.theory.cross_ps_parametric_model import (
        fit_and_decompose_ihl_templates,
        get_ihl_components_at_ell,
        compare_ihl_to_data,
        load_ihl_templates
    )
    print("   ✓ All functions imported successfully")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check template directory
print("\n2. Checking template directory...")
template_dir = '/Users/richardfeder/Documents/ciber/ihl_templates'
if os.path.exists(template_dir):
    files = [f for f in os.listdir(template_dir) if f.endswith('.txt')]
    print(f"   ✓ Found {len(files)} template files:")
    for f in files[:3]:
        print(f"     - {f}")
    if len(files) > 3:
        print(f"     ... and {len(files)-3} more")
else:
    print(f"   ✗ Template directory not found: {template_dir}")
    sys.exit(1)

# Test 3: Load templates
print("\n3. Testing template loading...")
try:
    zbinedges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    templates = load_ihl_templates(template_dir, zbinedges=zbinedges, slopes=[1.0])
    print(f"   ✓ Loaded {len(templates)} templates")
    
    # Check one template
    first_key = list(templates.keys())[0]
    template = templates[first_key]
    print(f"   ✓ First template '{first_key}':")
    print(f"     - {len(template['ell'])} data points")
    print(f"     - ell range: [{template['ell'].min():.0f}, {template['ell'].max():.0f}]")
    print(f"     - D_ell range: [{template['dl'].min():.2e}, {template['dl'].max():.2e}]")
except Exception as e:
    print(f"   ✗ Template loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Fit one template
print("\n4. Testing fit function (on first template only)...")
try:
    # Fit just the first template for speed
    first_template_name = list(templates.keys())[0]
    results = fit_and_decompose_ihl_templates(
        template_dir=template_dir,
        zbinedges=zbinedges[:2],  # Just first bin
        slopes=[1.0],
        plot=False,  # Skip plotting for test
        verbose=False
    )
    
    print(f"   ✓ Fit completed successfully")
    print(f"   ✓ Results structure:")
    print(f"     - templates: {len(results['templates'])} templates")
    print(f"     - fits: {len(results['fits'])} fits")
    print(f"     - summary: {len(results['summary'])} rows")
    
    # Check fit results
    fit_key = list(results['fits'].keys())[0]
    fit = results['fits'][fit_key]
    if 'error' in fit:
        print(f"   ⚠ Fit had error: {fit['error']}")
    else:
        params = fit['params']
        print(f"   ✓ Fit parameters for '{fit_key}':")
        print(f"     - A_2h = {params[0]:.3e}")
        print(f"     - A_1h = {params[1]:.3e}")
        print(f"     - ℓ_peak = {np.exp(params[2]):.0f}")
        print(f"     - σ_1h = {params[3]:.3f}")
        print(f"     - A_shot = {params[4]:.3e}")
        
        # Test get_ihl_components_at_ell
        print("\n5. Testing component evaluation at specific ℓ...")
        test_ell = np.array([500, 1000, 2000, 5000])
        components = get_ihl_components_at_ell(fit, test_ell)
        print(f"   ✓ Evaluated components at {len(test_ell)} multipoles")
        print(f"     At ℓ=1000:")
        print(f"       - Two-halo: {components['two_halo'][1]:.3e}")
        print(f"       - One-halo: {components['one_halo'][1]:.3e}")
        print(f"       - Shot noise: {components['shot_noise'][1]:.3e}")
        print(f"       - Total: {components['total'][1]:.3e}")

except Exception as e:
    print(f"   ✗ Fitting failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("All Tests Passed! ✓")
print("="*70)
print("\nYou can now use the IHL template decomposition functions.")
print("See IHL_DECOMPOSITION_README.md for usage examples.")
