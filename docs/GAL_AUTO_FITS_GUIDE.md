# Galaxy Auto-Spectrum Fitting Guide

## Overview

The `run_gal_auto_fits()` function fits galaxy auto-spectra (C_ℓ^gg) using a parametric 2-halo + 1-halo + shot noise model where the 1-halo shape parameters are **fixed** from IHL (Intrahalo Light) decomposition.

## Key Features

- **Fixed 1-halo shape**: Uses IHL-derived parameters (μ_1h, σ_1h) from `ihl_1h_params.npz`
- **Free amplitudes only**: Fits only A_2h, A_1h, and A_shot
- **Compatible with cross-spectra**: Uses same redshift binning and format as `run_gal_cross_fits()`
- **MCMC fitting**: Robust parameter estimation with uncertainties
- **Automatic saving**: Saves fit results in same format as cross-spectrum fits

## Model

The galaxy auto-spectrum is modeled as:

```
D_ℓ^gg = A_2h · D_ℓ^2h(α) + A_1h · exp[-(ln ℓ - μ)²/(2σ²)] + A_shot
```

Where:
- **A_2h, A_1h, A_shot**: Free amplitude parameters (fit by MCMC)
- **α**: Fixed power-law index for 2-halo term (default 0.0)
- **μ_1h, σ_1h**: Fixed 1-halo shape parameters from IHL decomposition

## Usage

```python
from ciber.theory.cross_ps_parametric_model import run_gal_auto_fits

# Basic usage
fit_results = run_gal_auto_fits(
    inst_list=[1, 2],                    # CIBER instruments (1.1μm, 1.8μm)
    cat='HSC',                           # Catalog: 'HSC' or 'LS'
    zbinedges=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # Redshift bins
    lMax_fit=80000,                      # Maximum ℓ for fitting
    save_results=True,                   # Save fit results
    file_fpath='gal_auto_fits_HSC.npz',  # Output filename
    ihl_1h_params_path='ihl_1h_params.npz',  # IHL parameters file
    headstr='hsc_ilt24.0'                # Data file prefix
)
```

## Parameters

### Required
- `inst_list`: List of instruments (e.g., [1, 2])
- `cat`: Catalog name ('HSC' or 'LS')
- `ihl_1h_params_path`: Path to IHL one-halo parameters file

### Data Selection
- `zbinedges`: Redshift bin edges (default: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
- `headstr`: Header string for data files (e.g., 'hsc_ilt24.0', 'sdss_z_lt_22.0')
- `startidx`, `endidx`: Index range for fitting (default: 2, -1)

### Fitting Control
- `lMax_fit`: Maximum multipole for fitting (default: 80000)
- `chi2_eval_max`: Maximum ℓ for χ² evaluation (default: 10000)
- `alpha_from_mock`: Fixed 2-halo power-law index (default: 0.0)
- `prior_bounds`: Optional custom prior bounds

### MCMC Settings
- `nwalkers`: Number of MCMC walkers (default: 32)
- `nsteps`: Number of MCMC steps (default: 4000)
- `nburn`: Burn-in steps to discard (default: 1000)

### Uncertainty Quantification
- `use_iterative_knox`: Use model-based iterative Knox covariance (default: False)
  - When `True`: Knox errors computed from model at each MCMC step (recommended)
  - When `False`: Knox errors computed from measured spectrum (standard)
  - See `ITERATIVE_KNOX_GUIDE.md` for details on when to use this
- `fmask`: Mask fraction per field (default: 0.67), only used if `use_iterative_knox=True`

### Output Control
- `save_figs`: Save corner and power spectrum plots (default: True)
- `save_results`: Save fit results to .npz file (default: False)
- `file_fpath`: Output filename for fit results
- `figbasedir`: Base directory for figures (default: 'figures/gal_auto_fits/')
- `fitstr`: String identifier for filenames (default: 'gal_auto')

## Output Format

### Fit Results Dictionary

```python
# Structure: all_fit_results_mcmc[f'inst{inst}_zbin{zidx}']
{
    'fit_result': {
        'params': [A_2h, A_1h, A_shot],         # Median values
        'params_err': [σ_A2h, σ_A1h, σ_Ashot],  # Standard deviations
        'percentiles': {...},                    # 16th, 84th, 95th, 99.7th
        'samples': array,                        # Full MCMC chain
        'chisq': float,                          # χ² value
        'ndof': int,                             # Degrees of freedom
        'reduced_chisq': float,                  # χ²/dof
        'lb_fit': array,                         # Multipoles used in fit
        'model_dl': array,                       # Best-fit model
        'residuals': array                       # Data - model
    },
    'inst': int,        # Instrument index (1 or 2)
    'zidx': int,        # Redshift bin index
    'zcen': float       # Redshift bin center
}
```

### Saved NPZ File

When `save_results=True`, creates `.npz` file with:
- Individual fit results for each (instrument, redshift) combination
- Metadata: dataset name, redshift bins, instruments
- Same format as cross-spectrum fits for easy comparison

## Accessing Results

```python
# Load saved results
data = np.load('data/gal_auto_fits/gal_auto_fits_HSC_coarsez_fixed1h.npz', allow_pickle=True)
fit_results = data['fit_results'].item()

# Access specific fit
result = fit_results['inst1_zbin2']['fit_result']

print(f"2-halo amplitude: {result['params'][0]:.3e} ± {result['params_err'][0]:.3e}")
print(f"1-halo amplitude: {result['params'][1]:.3e} ± {result['params_err'][1]:.3e}")
print(f"Shot noise: {result['params'][2]:.3e} ± {result['params_err'][2]:.3e}")
print(f"χ²/dof = {result['chisq']:.1f}/{result['ndof']} = {result['reduced_chisq']:.2f}")

# Plot model
import matplotlib.pyplot as plt
lb = result['lb_fit']
model = result['model_dl']
plt.plot(lb, model, label='Best-fit model')
plt.show()
```

## Combining with Cross-Spectra

The galaxy auto fits can be combined with cross-spectrum fits to predict CIBER auto-spectra:

```python
# Load both galaxy auto and CIBER×galaxy cross fits
gal_auto_fits = np.load('data/gal_auto_fits/gal_auto_fits_HSC.npz', allow_pickle=True)
cross_fits = np.load('data/cross_cl_fits/ciber_cl_fits_HSC.npz', allow_pickle=True)

# Extract clustering components (2h+1h without shot)
for key in gal_auto_fits['fit_results'].item():
    gal_result = gal_auto_fits['fit_results'].item()[key]['fit_result']
    cross_result = cross_fits['fit_results'].item()[key]['fit_result']
    
    # Galaxy clustering (2h+1h, no shot)
    A_2h_gal, A_1h_gal = gal_result['params'][:2]
    
    # Cross-spectrum amplitudes
    A_2h_cross, A_1h_cross = cross_result['params'][:2]
    
    # Predict CIBER auto: (cross)²/gal_clustering
    # ... (see predict_ciber_auto_vs_redshift for full implementation)
```

## Differences from run_gal_cross_fits()

| Feature | run_gal_auto_fits | run_gal_cross_fits |
|---------|-------------------|-------------------|
| Data type | Galaxy auto (C_ℓ^gg) | CIBER×galaxy cross (C_ℓ^Ig) |
| IHL templates | Not used (parametric only) | Optional |
| 1h shape | Always fixed from IHL | Optional (can be fixed or free) |
| Free parameters | 3 (A_2h, A_1h, A_shot) | 3-6 depending on config |
| Damping | Not modeled | Optional |
| Field averaging | Uses collect_ciber_gal_vs_redshift | Uses per-field data |

## Requirements

1. **IHL parameters file** (`ihl_1h_params.npz`):
   - Generated by `fit_and_decompose_ihl_templates()`
   - Contains fixed μ_1h and σ_1h per redshift bin
   
2. **Galaxy auto-spectra data**:
   - Loaded via `collect_ciber_gal_vs_redshift()`
   - Format: field-averaged C_ℓ per (inst, zbin)

3. **Dependencies**:
   - numpy, matplotlib
   - emcee (for MCMC)
   - ciber package modules

## Example Workflow

```python
# 1. Generate or load IHL parameters
from ciber.theory.cross_ps_parametric_model import fit_and_decompose_ihl_templates
# ... (generate ihl_1h_params.npz)

# 2. Fit galaxy auto-spectra
fit_results_gal = run_gal_auto_fits(
    cat='HSC',
    save_results=True,
    file_fpath='gal_auto_HSC.npz'
)

# 3. Fit CIBER×galaxy cross-spectra (if needed)
from ciber.theory.cross_ps_parametric_model import run_gal_cross_fits
fit_results_cross = run_gal_cross_fits(
    cat='HSC',
    use_ihl_templates=True,
    save_results=True
)

# 4. Combine to predict CIBER auto
# ... (use both fit results)
```

## Notes

- The function only handles the fixed-1h-shape case (no free shape parameters)
- No astrometry damping is modeled (use_astrometry_damping=False)
- Shot noise is included as a constant offset in the model
- Fits are performed in D_ℓ = ℓ(ℓ+1)C_ℓ/(2π) space
- MCMC chains are saved for post-processing and corner plots

## See Also

- `run_gal_cross_fits()`: Fit CIBER×galaxy cross-spectra
- `fit_and_decompose_ihl_templates()`: Generate IHL 1h parameters
- `predict_ciber_auto_vs_redshift()`: Predict CIBER auto from cross and galaxy auto
- `save_fit_results_npz()`: Save fit results to file
- `load_fit_results_npz()`: Load saved fit results
