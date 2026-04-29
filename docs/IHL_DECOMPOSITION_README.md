# IHL Template Decomposition Guide

## Overview

I've created a comprehensive framework for loading and decomposing your IHL (Intra-Halo Light) templates into three physical components:

1. **Two-halo term**: Large-scale clustering (modeled as power law)
2. **One-halo term**: Non-linear clustering within halos (modeled as log-normal)
3. **Shot noise**: Poisson fluctuations (modeled as ℓ²)

The mathematical model is:
```
D_ℓ = A_2h * (ℓ/1000)^α + A_1h * exp(-(ln(ℓ) - μ)²/(2σ²)) + A_shot * ℓ(ℓ+1)/(2π)
```

## What's Been Added

### Main Function: `fit_and_decompose_ihl_templates()`

Located in: `/Users/richardfeder/Documents/ciber/ciber/theory/cross_ps_parametric_model.py`

This function:
- Loads IHL templates from your `ihl_templates/` directory
- Fits each template with the parametric model
- Returns best-fit parameters and components
- Creates diagnostic plots showing the decomposition

### Helper Functions

1. **`get_ihl_components_at_ell()`**: Evaluate fitted components at specific multipole values
2. **`compare_ihl_to_data()`**: Compare IHL template fit to your actual data measurements

### Example Script

I've created a complete working example: `example_ihl_template_decomposition.py`

## Quick Start

### Basic Usage

```python
from ciber.theory.cross_ps_parametric_model import fit_and_decompose_ihl_templates
import numpy as np

# Define your redshift bins
zbinedges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# Run the decomposition
results = fit_and_decompose_ihl_templates(
    template_dir='ihl_templates/',
    zbinedges=zbinedges,
    slopes=[1.0],
    plot=True,
    save_path='ihl_decomposition.png'
)

# Access results
print(results['summary'])  # Pandas DataFrame with all fit parameters

# Get components for a specific template
fit = results['fits']['z0.0_0.2_slope1.0']
params = fit['params']  # [A_2h, A_1h, mu_1h, sigma_1h, A_shot]
components = fit['components']  # Dict with 'two_halo', 'one_halo', 'shot_noise', 'total'
```

### Accessing Fit Results

```python
# Loop through all templates
for template_name, fit in results['fits'].items():
    if 'error' in fit:
        print(f"Fit failed for {template_name}")
        continue
    
    params = fit['params']
    print(f"{template_name}:")
    print(f"  Two-halo amplitude: {params[0]:.3e}")
    print(f"  One-halo amplitude: {params[1]:.3e}")
    print(f"  One-halo peak (ℓ): {np.exp(params[2]):.0f}")
    print(f"  Shot noise amplitude: {params[4]:.3e}")
```

### Evaluating Components at Specific Multipoles

```python
from ciber.theory.cross_ps_parametric_model import get_ihl_components_at_ell

# Get fit for a specific template
fit = results['fits']['z0.0_0.2_slope1.0']

# Evaluate at your data multipoles
data_ell = np.array([500, 1000, 2000, 5000, 10000])
components_at_data = get_ihl_components_at_ell(fit, data_ell)

print(f"Two-halo at ell=1000: {components_at_data['two_halo'][1]:.3e}")
print(f"One-halo at ell=1000: {components_at_data['one_halo'][1]:.3e}")
print(f"Total at ell=1000: {components_at_data['total'][1]:.3e}")
```

### Comparing to Your Data

```python
from ciber.theory.cross_ps_parametric_model import compare_ihl_to_data

# Your measured cross-spectrum
data_ell = np.array([500, 1000, 2000, 5000])
data_dl = np.array([1.5, 3.2, 2.8, 1.1])  # Your measurements
data_dl_err = np.array([0.2, 0.3, 0.3, 0.2])  # Uncertainties

# Compare to IHL template
comparison = compare_ihl_to_data(
    template_dir='ihl_templates/',
    zbinedges=np.array([0.0, 0.2, 0.4]),
    slopes=[1.0],
    data_ell=data_ell,
    data_dl=data_dl,
    data_dl_err=data_dl_err,
    z_idx=0,  # First redshift bin
    plot=True,
    save_path='ihl_vs_data.png'
)

print(f"Reduced chi-squared: {comparison['data_comparison']['reduced_chisq']:.2f}")
```

## Output Structure

### `results` Dictionary

```python
results = {
    'templates': {
        'z0.0_0.2_slope1.0': {
            'ell': array,      # Template multipoles
            'dl': array,       # Template D_ell values
            'zbinedges': (0.0, 0.2),
            'slope': 1.0,
            ...
        },
        ...
    },
    
    'fits': {
        'z0.0_0.2_slope1.0': {
            'params': [A_2h, A_1h, mu_1h, sigma_1h, A_shot],
            'params_err': [errors...],
            'param_names': ['A_2h', 'A_1h', 'mu_1h', 'sigma_1h', 'A_shot'],
            'chisq': float,
            'reduced_chisq': float,
            'components': {
                'two_halo': array,
                'one_halo': array,
                'shot_noise': array,
                'total': array
            },
            'ell_eval': array,  # Multipoles where components are evaluated
            'ell_template': array,  # Original template multipoles
            'dl_template': array,   # Original template D_ell
            'zbinedges': (0.0, 0.2),
            'slope': 1.0
        },
        ...
    },
    
    'summary': pandas.DataFrame with columns:
        - template: template name
        - A_2h, A_2h_err: two-halo amplitude
        - A_1h, A_1h_err: one-halo amplitude
        - mu_1h, mu_1h_err: one-halo peak location (ln ℓ)
        - ell_peak: one-halo peak (ℓ = exp(mu_1h))
        - sigma_1h, sigma_1h_err: one-halo width
        - A_shot, A_shot_err: shot noise amplitude
        - chisq, reduced_chisq: fit quality
        - z_low, z_high, z_center: redshift info (if applicable)
}
```

## Parameters

### Fit Parameters (5 parameters)

1. **A_2h**: Two-halo amplitude at ℓ=1000
2. **A_1h**: One-halo log-normal amplitude (peak height)
3. **mu_1h**: One-halo peak location in log-space (ln(ℓ_peak))
   - To get ℓ_peak: `ell_peak = np.exp(mu_1h)`
4. **sigma_1h**: One-halo width in log-space
5. **A_shot**: Shot noise amplitude (in C_ℓ units)

### Function Arguments

Key arguments for `fit_and_decompose_ihl_templates()`:

- `template_dir`: Path to your IHL templates directory
- `zbinedges`: Array of redshift bin edges (e.g., `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`)
- `slopes`: List of slope values (e.g., `[1.0]`)
- `use_powerlaw_2h=True`: Model 2-halo as power law (recommended)
- `alpha_2h_fixed=-1.5`: Power-law index (linear clustering)
- `fit_ell_range`: Tuple `(ℓ_min, ℓ_max)` to restrict fit range
- `plot=True`: Create diagnostic plots
- `save_path`: Where to save the plot (None = display interactively)
- `verbose=True`: Print detailed fitting information

## Your Template Files

Located in: `/Users/richardfeder/Documents/ciber/ihl_templates/`

Files found:
- `ihl_ps_z_0.0_0.2_slope_1.0.txt`
- `ihl_ps_z_0.2_0.4_slope_1.0.txt`
- `ihl_ps_z_0.4_0.6_slope_1.0.txt`
- `ihl_ps_z_0.6_0.8_slope_1.0.txt`
- `ihl_ps_z_0.8_1.0_slope_1.0.txt`

Format: Two columns (ell, D_ell) with one header row

## Running the Example

```bash
cd /Users/richardfeder/Documents/ciber
python example_ihl_template_decomposition.py
```

This will generate:
- `ihl_decomposition_example.png` - Main decomposition plots for all templates
- `ihl_custom_plot_example.png` - Custom analysis for z=0.0-0.2
- `ihl_redshift_evolution.png` - Evolution of components with redshift
- `ihl_decomposition_results.npz` - Numerical results

## Tips and Best Practices

1. **Fit Range**: If your templates have noise at high-ℓ, use `fit_ell_range=(500, 10000)` to exclude problematic regions

2. **Initialization**: The function automatically guesses good initial parameters, but you can override with `p0=[A_2h, A_1h, mu_1h, sigma_1h, A_shot]`

3. **Bounds**: Physical bounds are set automatically, but can be customized with the `bounds` parameter

4. **Multiple Slopes**: If you have templates at different slopes, pass them as `slopes=[0.8, 1.0, 1.2]`

5. **Redshift Evolution**: The summary DataFrame makes it easy to analyze how components evolve:
   ```python
   import matplotlib.pyplot as plt
   summary = results['summary']
   plt.plot(summary['z_center'], summary['A_1h'], 'o-')
   plt.xlabel('Redshift')
   plt.ylabel('One-halo Amplitude')
   ```

## Interpretation

- **Two-halo dominates at low ℓ** (~300-1000): Large-scale structure correlation
- **One-halo peaks at intermediate ℓ** (~1000-5000): Non-linear clustering within halos
- **Shot noise dominates at high ℓ** (>5000): Discrete source Poisson fluctuations

The peak location (ℓ_peak) relates to the characteristic halo scale and depends on redshift and mass distribution.

## Questions or Issues?

If you need to:
- Modify the model (e.g., different functional forms)
- Add priors to the fit
- Change the fitting method
- Add more components

The code is modular and well-documented. Check the docstrings in `cross_ps_parametric_model.py`.
