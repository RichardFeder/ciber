# IHL Template Fitting: Two Distinct Methods

## Overview

There are **two completely separate approaches** for using IHL (Intra-Halo Light) templates:

### Method 1: Direct IHL Template Fitting
**Use when:** You want to fit the actual IHL template files provided by your collaborator

**How it works:**
- Loads raw IHL template files from `ihl_templates/` directory
- Uses `fit_model_with_ihl_templates()` method
- Fits amplitudes to the interpolated template values
- May show "kinks" at ℓ < 300 due to extrapolation

**In `run_gal_cross_fits`:**
```python
run_gal_cross_fits(
    ...
    use_ihl_templates=True,  # KEY: enables template method
    ihl_template_dir='ihl_templates/',
    ...
)
```

**What gets fitted:**
- `A_2h`: Two-halo amplitude
- `A_ihl_z0.8`, `A_ihl_z1.0`, `A_ihl_z1.2`: IHL template amplitudes per redshift bin
- `A_shot`: Shot noise amplitude

**Model equation:**
```
D_ℓ = A_2h * powerlaw(ℓ) + Σ A_ihl * IHL_template(ℓ, z) + A_shot * ℓ(ℓ+1)/(2π)
```

---

### Method 2: Smooth Parametric Model with IHL-Derived Parameters
**Use when:** You want a smooth, continuous log-normal model using parameters extracted from IHL templates

**How it works:**
1. First, run `fit_and_decompose_ihl_templates()` to extract parametric shape from IHL templates
2. Save one-halo parameters with `save_ihl_1h_params()`
3. Load these parameters in subsequent fits
4. Fits use smooth log-normal function (no interpolation, no kinks)

**In `run_gal_cross_fits`:**
```python
run_gal_cross_fits(
    ...
    use_ihl_templates=False,  # KEY: uses parametric model
    use_ihl_1h_params=True,   # KEY: uses IHL-derived priors
    ihl_1h_params_path='ihl_1h_params.npz',
    ...
)
```

**What gets fitted:**
- `A_2h`: Two-halo amplitude (with power law index α)
- `A_1h`: One-halo log-normal amplitude
- `mu_1h`: Log-normal center (ln(ℓ_peak))
- `sigma_1h`: Log-normal width
- `A_shot`: Shot noise amplitude

**Model equation:**
```
D_ℓ = A_2h * (ℓ/1000)^α + A_1h * exp(-(ln(ℓ) - μ_1h)²/(2σ_1h²)) + A_shot * ℓ(ℓ+1)/(2π)
```

**Priors:** When `use_ihl_1h_params=True`, the fitting will use IHL-derived linear relations:
- `ln(ℓ_peak) = intercept + slope * z`
- `σ = intercept + slope * z`

These priors guide the fit but the actual parameters are still fitted.

---

## Key Differences

| Aspect | Method 1 (Template) | Method 2 (Parametric) |
|--------|-------------------|---------------------|
| **One-halo term** | Interpolated from raw template | Smooth log-normal function |
| **Kinks at low ℓ** | Yes (extrapolation artifacts) | No (fully smooth) |
| **Parameters** | Template amplitudes only | Full log-normal parameters |
| **Flexibility** | Fixed to template shape | Flexible parametric form |
| **Use case** | Direct template comparison | General modeling, extrapolation |
| **Enable with** | `use_ihl_templates=True` | `use_ihl_templates=False` |

---

## Complete Workflow Example

### Step 1: Extract Parameters from IHL Templates
```python
from ciber.theory.cross_ps_parametric_model import (
    fit_and_decompose_ihl_templates, save_ihl_1h_params
)

# Fit IHL templates to extract parametric shape
results = fit_and_decompose_ihl_templates(
    template_dir='ihl_templates/',
    zbinedges=np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    slopes=[1.0],
    alpha_2h_fixed=0.0,  # Flat two-halo in D_ell
    plot=True
)

# Save one-halo parameters for future use
save_ihl_1h_params(
    results, 
    save_path='ihl_1h_params.npz',
    zbinedges=np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    slopes=[1.0]
)
```

### Step 2a: Use Direct Template Method
```python
from ciber.theory.cross_ps_parametric_model import run_gal_cross_fits

# Fit using actual IHL template files
fit_results = run_gal_cross_fits(
    cat='gaia',
    cat_path='/path/to/catalog',
    cl_fpath='/path/to/powerspectra',
    use_ihl_templates=True,  # Use templates directly
    ihl_template_dir='ihl_templates/',
    zbinedges=np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    slopes=[1.0],
    # ... other parameters
)
```

### Step 2b: Use Smooth Parametric Method with IHL-Derived Priors
```python
# Fit using smooth parametric model with IHL-derived priors
fit_results = run_gal_cross_fits(
    cat='gaia',
    cat_path='/path/to/catalog',
    cl_fpath='/path/to/powerspectra',
    use_ihl_templates=False,  # Use parametric model
    use_ihl_1h_params=True,   # Use IHL-derived priors
    ihl_1h_params_path='ihl_1h_params.npz',
    # ... other parameters
)
```

---

## Helper Functions

### `get_ihl_components_at_ell(fit_result, ell_values)`
Evaluates the **smooth parametric model** at specific multipoles using parameters from `fit_and_decompose_ihl_templates()`.

**Important:** Returns smooth log-normal components, NOT raw template values.

```python
# Get smooth parametric components
results = fit_and_decompose_ihl_templates('ihl_templates/', ...)
fit = results['fits']['z0.0_0.2_slope1.0']
ell_data = np.array([500, 1000, 2000, 5000])
components = get_ihl_components_at_ell(fit, ell_data)

print(f"Smooth 1h at ell=1000: {components['one_halo'][1]:.3e}")
```

### `compare_ihl_to_data(template_dir, zbinedges, slopes, data_ell, data_dl, ...)`
Loads IHL templates, fits them, and compares to data.

---

## Important Notes

1. **No hybrid approach**: These two methods are completely independent. Use one or the other, not both simultaneously.

2. **Priors vs Fixed Templates**: When `use_ihl_1h_params=True` with `use_ihl_templates=False`, the IHL-derived parameters serve as **priors** (guiding the fit), not fixed values. The fit still optimizes all parameters.

3. **Two-halo behavior**: 
   - Template method: Uses power law with fitted or fixed index
   - Parametric method: Typically use `alpha_2h_fixed=0.0` for flat D_ell (per user preference)

4. **Extrapolation**: 
   - Template method: May have kinks at ℓ < 300 due to log-log extrapolation
   - Parametric method: Smooth everywhere (log-normal function)

---

## Files Created

- `ihl_1h_params.npz`: Saved one-halo parameters with linear relations
- `example_ihl_template_decomposition.py`: Full decomposition workflow
- `example_save_ihl_1h_params.py`: Parameter saving example
- `example_automatic_ihl_params_integration.py`: Integration with `run_gal_cross_fits`
- `test_ihl_decomposition.py`: Validation script

---

## Questions?

If you're unsure which method to use:
- **Use Method 1** if you want to directly fit your collaborator's IHL templates
- **Use Method 2** if you want smooth parametric modeling informed by IHL template shapes
