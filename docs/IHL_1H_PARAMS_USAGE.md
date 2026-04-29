# Using IHL Template One-Halo Parameters in Galaxy Cross-Fits

## Overview

You can now extract one-halo component parameters (μ_1h and σ_1h) from your IHL template fits and use them in `run_gal_cross_fits` instead of the default analytic formulae.

## Quick Start

### 1. Extract and Save Parameters from IHL Templates

```python
from ciber.theory.cross_ps_parametric_model import (
    fit_and_decompose_ihl_templates,
    save_ihl_1h_params
)
import numpy as np

# Fit IHL templates
zbinedges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
results = fit_and_decompose_ihl_templates(
    template_dir='ihl_templates/',
    zbinedges=zbinedges,
    slopes=[1.0],
    plot=True
)

# Save one-halo parameters
one_halo_params = save_ihl_1h_params(
    results,
    save_path='ihl_1h_params.npz',
    zbinedges=zbinedges,
    slopes=[1.0]
)
```

### 2. Load and Use in run_gal_cross_fits

```python
from ciber.theory.cross_ps_parametric_model import load_ihl_1h_params

# Load the saved parameters
one_halo_params = load_ihl_1h_params('ihl_1h_params.npz')

# Use in fitting functions
fit_result = model.fit_model_fixed_1h_templates(
    lb_data, dl_data, z_value,
    one_halo_params_dict=one_halo_params,  # Pass here
    ...
)
```

### 3. Use with interpolate_1h_params

```python
from ciber.theory.cross_ps_parametric_model import interpolate_1h_params

# Without IHL params (uses default formulae)
ln_ell_peak, sigma = interpolate_1h_params(z_value=0.3)

# With IHL params (uses template-derived values)
ln_ell_peak, sigma = interpolate_1h_params(
    z_value=0.3, 
    slope=1.0,
    one_halo_params_dict=one_halo_params
)
```

## What Gets Saved

The `save_ihl_1h_params()` function saves:

1. **Individual parameters** for each (redshift bin, slope) combination:
   - `mu_1h`: ln(ℓ_peak) for the one-halo log-normal
   - `sigma_1h`: Log-width parameter
   - `ell_peak`: The actual peak location (= exp(mu_1h))

2. **Linear relations** fitted to the data:
   - `ln(ell_peak) = intercept + slope * z`
   - `sigma = intercept + slope * z`

These linear relations allow `interpolate_1h_params` to calculate parameters at any redshift, not just the bin centers.

## File Structure

The saved `.npz` file contains:

```python
{
    'zbinedges': array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    'slopes': array([1.0]),
    'params_dict': {
        (0, 1.0): {'mu_1h': 8.56, 'sigma_1h': 1.78, 'ell_peak': 5234, ...},
        (1, 1.0): {'mu_1h': 9.10, 'sigma_1h': 2.01, 'ell_peak': 8955, ...},
        ...
    },
    'ln_ell_peak_vs_z': {
        1.0: {'intercept': 8.44, 'slope': 7.4, 'r_value': 0.98, ...}
    },
    'sigma_vs_z': {
        1.0: {'intercept': 1.56, 'slope': 2.43, 'r_value': 0.95, ...}
    },
    'z_centers': array([0.1, 0.3, 0.5, 0.7, 0.9])
}
```

## Key Functions

### `save_ihl_1h_params(results, save_path, zbinedges, slopes)`

Extracts one-halo parameters from IHL template fits and saves to file.

**Parameters:**
- `results`: Output from `fit_and_decompose_ihl_templates()`
- `save_path`: Where to save (e.g., `'ihl_1h_params.npz'`)
- `zbinedges`: Array of redshift bin edges
- `slopes`: List of slope values

**Returns:** Dictionary with parameters and linear fits

### `load_ihl_1h_params(load_path)`

Loads saved one-halo parameters.

**Parameters:**
- `load_path`: Path to saved `.npz` file

**Returns:** Dictionary with structure compatible with `interpolate_1h_params`

### `interpolate_1h_params(z_value, slope, one_halo_params_dict, sigma_fixed)`

Calculate one-halo parameters at any redshift (updated to support IHL-derived params).

**Parameters:**
- `z_value`: Redshift
- `slope`: Slope value (optional if dict provided)
- `one_halo_params_dict`: IHL-derived parameters (optional, uses defaults if None)
- `sigma_fixed`: Override sigma with fixed value (optional)

**Returns:** `(ln_ell_peak, sigma)`

## Workflow Integration

### In run_gal_cross_fits

You can now pass IHL-derived parameters to any fitting function that accepts `one_halo_params_dict`:

```python
# At the start of run_gal_cross_fits or in your analysis script
one_halo_params = load_ihl_1h_params('ihl_1h_params.npz')

# Later in the fitting loop
for zidx in range(len(zbinedges) - 1):
    zcen = 0.5 * (zbinedges[zidx] + zbinedges[zidx + 1])
    
    # Get IHL-derived parameters for this redshift
    ln_ell_peak, sigma = interpolate_1h_params(
        zcen, slope=1.0, one_halo_params_dict=one_halo_params
    )
    
    # Use in fitting...
```

### Comparison: Default vs IHL-Derived

**Default formulae** (hard-coded):
```
ln(ell_peak) = 7.4 * z + 8.44
sigma = 2.43 * z + 1.56
```

**IHL-derived** (from template fits):
```
ln(ell_peak) = intercept + slope * z    # Fitted from templates
sigma = intercept + slope * z           # Fitted from templates
```

The IHL-derived parameters should be more accurate because they come from actual fits to your collaborator's templates.

## Complete Example

See `example_save_ihl_1h_params.py` for a complete working example that:
1. Fits IHL templates
2. Extracts and saves one-halo parameters
3. Loads and uses them
4. Compares IHL-derived vs default parameters
5. Visualizes the linear relations

Run it with:
```bash
cd /Users/richardfeder/Documents/ciber
python example_save_ihl_1h_params.py
```

## Benefits

1. **More accurate**: Parameters derived from actual IHL template fits
2. **Consistent**: Uses the same one-halo shapes across analyses
3. **Flexible**: Can still use default formulae if IHL params not available
4. **Backward compatible**: Existing code works without changes

## Files Created

Running the workflow creates:
- `ihl_1h_params_for_gal_cross_fits.npz` - Parameter file
- `ihl_1h_params_linear_relations.png` - Visualization of fits

These can be version-controlled and shared with collaborators to ensure everyone uses the same one-halo parameters.
