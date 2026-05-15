# Integration: One-Halo Templates with Power Spectrum Fitting

## Overview

This guide explains how to use the cached effective one-halo templates in your DESI-LS and HSC auto/cross power spectrum fitting workflow.

## Quick Start: Create Cache

First, compute and cache the effective 1h template:

```bash
cd /Users/richardfeder/Documents/ciber
python3 scripts/cache_effective_1h_templates.py
```

This creates:
```
data/1h_template_cache/
├── effective_1h_slope_1.0.npz      # Effective template (sum of all z-bins)
├── individual_1h_slope_1.0.npz     # Individual z-bin templates
└── cache_metadata.json              # Cache metadata
```

## Using Cached Templates in Fitting

### Option 1: Quick Load for Reference

```python
from ciber.theory.ihl_1h_template_cache import load_effective_1h_for_fitting

# Load the effective 1h normalized shape
one_halo_norm = load_effective_1h_for_fitting(slope=1.0)

# Use in your fitting as reference or comparison
```

### Option 2: Full Cache Access

```python
from ciber.theory.ihl_1h_template_cache import OneHaloTemplateCache

# Load cache
cache = OneHaloTemplateCache(cache_dir='data/1h_template_cache')

# Get effective template
effective_1h, individual_1h, zbinedges = cache.load_cache(slope=1.0)

# Access effective template
ell = effective_1h[1.0]['ell']
one_halo_norm = effective_1h[1.0]['one_halo_norm']
one_halo_sum = effective_1h[1.0]['one_halo_sum']

# Access individual z-bin templates
for zidx in individual_1h[1.0]:
    z_info = individual_1h[1.0][zidx]
    z_mid = z_info['z_mid']
    A_1h = z_info['A_1h']
    mu_1h = z_info['mu_1h']
    sigma_1h = z_info['sigma_1h']
    # Use in fitting...
```

### Option 3: Direct NPZ Access

```python
import numpy as np

# Load effective template directly
eff_data = np.load('data/1h_template_cache/effective_1h_slope_1.0.npz')
ell = eff_data['ell']
one_halo_norm = eff_data['one_halo_norm']

# Load individual templates
ind_data = np.load('data/1h_template_cache/individual_1h_slope_1.0.npz')
```

## Integration with auto_cross_fits_pipeline.py

### Using as Template Constraint

The cached effective 1h template can be used in fitting as:

1. **Prior shape constraint**: Regularize fitting to stay close to effective shape
2. **Comparison reference**: Check if individual measurements deviate significantly
3. **Redshift evolution**: Compare z-bin measurements to effective template

### Example: Fit with Template Reference

```python
# In your fitting code (e.g., in run_gal_cross_fits or similar)

from ciber.theory.ihl_1h_template_cache import OneHaloTemplateCache

# Load cache
cache = OneHaloTemplateCache()
effective_1h, individual_1h, zbinedges = cache.load_cache(slope=1.0)

# Get effective template
ell_template = effective_1h[1.0]['ell']
one_halo_template = effective_1h[1.0]['one_halo_norm']

# In fitting loop, for each catalog/z-bin:
for cat in ['DESI-LS', 'HSC']:
    for zidx in range(len(zbinedges) - 1):
        # Fit power spectrum
        fit_result = fit_power_spectrum(...)
        
        # Compare 1h component to effective template
        measured_1h = fit_result['dl_1h']
        measured_1h_norm = measured_1h / np.max(measured_1h)
        
        # Interpolate template to fit ell grid
        template_interp = np.interp(
            fit_result['ell'],
            ell_template,
            one_halo_template
        )
        
        # Calculate shape consistency
        chi2_shape = np.sum(
            ((measured_1h_norm - template_interp) / error)**2
        )
        
        # Use as quality metric or prior constraint
```

## Data Structure Reference

### Cache Contents

```
data/1h_template_cache/
├── cache_metadata.json
│   ├── description: str
│   ├── zbinedges: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
│   ├── slopes: [1.0]
│   ├── nzbins: 5
│   └── cached_slopes:
│       └── 1.0:
│           ├── effective_file: "effective_1h_slope_1.0.npz"
│           └── individual_file: "individual_1h_slope_1.0.npz"
│
├── effective_1h_slope_1.0.npz
│   ├── ell: array (multipole grid)
│   ├── one_halo_sum: array (summed unnormalized components)
│   ├── one_halo_avg: array (averaged unnormalized components)
│   └── one_halo_norm: array (normalized shape)
│
└── individual_1h_slope_1.0.npz
    ├── zbin_0_ell: array
    ├── zbin_0_one_halo: array
    ├── zbin_0_z_range: [0.0, 0.2]
    ├── zbin_0_z_mid: 0.1
    ├── zbin_0_A_1h: float
    ├── zbin_0_mu_1h: float
    ├── zbin_0_sigma_1h: float
    ├── zbin_1_ell: array
    ... (repeat for zbins 1-4)
```

### Effective Template Structure

```python
effective_1h = {
    1.0: {  # slope value
        'ell': array,              # Multipole grid
        'one_halo_sum': array,     # Sum of unnormalized components
        'one_halo_avg': array,     # Average of unnormalized components
        'one_halo_norm': array,    # Normalized shape (range [0, 1])
        'n_bins_summed': 5         # Number of z-bins combined
    }
}
```

### Individual Z-Bin Structure

```python
individual_1h = {
    1.0: {  # slope value
        0: {  # z-bin index
            'ell': array,              # Multipole grid
            'one_halo': array,         # Unnormalized 1h component
            'z_range': (0.0, 0.2),     # Redshift bin edges
            'z_mid': 0.1,              # Bin center
            'A_1h': 10.20,             # Fitted amplitude
            'mu_1h': 9.356,            # Peak location (ln scale)
            'sigma_1h': 1.751          # Log-width
        },
        1: {...},  # z-bin 1 (0.2-0.4)
        # ... etc
    }
}
```

## Auto/Cross Power Spectrum Fitting Workflow

### Step 1: Cache Templates (One-time)

```bash
python3 scripts/cache_effective_1h_templates.py
```

### Step 2: Load Cache in Fitting Pipeline

```python
# In auto_cross_fits_pipeline.py or your fitting code

from ciber.theory.ihl_1h_template_cache import OneHaloTemplateCache

# Initialize cache loader
cache = OneHaloTemplateCache(cache_dir='data/1h_template_cache')

# Load templates once at start
effective_1h, individual_1h, zbinedges = cache.load_cache(slope=1.0)
```

### Step 3: Use in Fitting

```python
# For DESI-LS z<1 auto spectrum
fit_result_desi_auto = run_gal_auto_fits(
    lb=lb,
    cl_desi_auto=cl_desi_auto,
    cat='DESI-LS',
    fitstr='z_lt_1.0',
    # ... other parameters
)

# For DESI-LS z<1 cross spectrum  
fit_result_desi_cross = run_gal_cross_fits(
    lb=lb,
    cl_cross=cl_desi_cross,
    cat='DESI-LS',
    fitstr='z_lt_1.0',
    # ... other parameters
)

# For HSC z<1 auto spectrum
fit_result_hsc_auto = run_gal_auto_fits(
    lb=lb,
    cl_hsc_auto=cl_hsc_auto,
    cat='HSC',
    fitstr='z_lt_1.0',
    # ... other parameters
)

# For HSC z<1 cross spectrum
fit_result_hsc_cross = run_gal_cross_fits(
    lb=lb,
    cl_cross=cl_hsc_cross,
    cat='HSC',
    fitstr='z_lt_1.0',
    # ... other parameters
)

# Compare all to effective template
for cat_name, fit_result in [
    ('DESI-LS auto', fit_result_desi_auto),
    ('DESI-LS cross', fit_result_desi_cross),
    ('HSC auto', fit_result_hsc_auto),
    ('HSC cross', fit_result_hsc_cross)
]:
    measured_1h = fit_result['dl_1h']
    measured_1h_norm = measured_1h / np.max(measured_1h)
    
    # Compare to effective template
    template_interp = np.interp(
        fit_result['ell'],
        effective_1h[1.0]['ell'],
        effective_1h[1.0]['one_halo_norm']
    )
    
    chi2 = np.sum(((measured_1h_norm - template_interp) / error)**2)
    print(f"{cat_name}: χ²={chi2:.2f}")
```

## API Reference

### OneHaloTemplateCache

```python
class OneHaloTemplateCache:
    """Cache manager for one-halo templates."""
    
    def __init__(self, cache_dir: str = 'data/1h_template_cache'):
        """Initialize cache."""
        
    def save_cache(self, effective_1h, individual_1h, zbinedges, slopes, 
                   description=""):
        """Save templates to cache."""
        
    def load_cache(self, slope=1.0) -> (dict, dict, array):
        """Load templates from cache."""
        
    def get_effective_1h_shape(self, slope=1.0) -> (array, array):
        """Quick load of effective 1h normalized shape."""
        
    def get_effective_1h_sum(self, slope=1.0) -> (array, array):
        """Load unnormalized effective 1h sum."""
        
    def list_cached_slopes(self) -> list:
        """List available slope values in cache."""
        
    def cache_exists(self) -> bool:
        """Check if cache is valid."""
```

### Convenience Functions

```python
def create_and_cache_effective_1h_template(
    template_dir='data/ihl_templates',
    zbinedges=None,
    slopes=[1.0],
    cache_dir=None,
    description="",
    plot=True
) -> (dict, dict, array):
    """Compute and save effective 1h template to cache."""

def load_effective_1h_for_fitting(
    slope=1.0,
    cache_dir=None
) -> array:
    """Quick load of effective 1h normalized shape."""
```

## Example: Full Workflow

```bash
# Step 1: Create cache (one-time)
python3 scripts/cache_effective_1h_templates.py

# Step 2: Run fitting with cached templates
python3 scripts/auto_cross_fits_pipeline.py \
    --mode run_auto \
    --cat DESI-LS \
    --fitstr z_lt_1.0

python3 scripts/auto_cross_fits_pipeline.py \
    --mode run_cross \
    --cat DESI-LS \
    --fitstr z_lt_1.0

# (Repeat for HSC)

# Step 3: Analyze results with template comparison
python3 scripts/analyze_fits_with_1h_templates.py
```

## Benefits

✅ **Fast access**: Templates loaded once, reused in all fits
✅ **Consistency**: All fits use same reference 1h template
✅ **Comparison**: Easy to check individual fits against effective template
✅ **Redshift evolution**: Can compare z-bin template to overall effective template
✅ **Quality control**: Template deviations flag measurement issues

## Troubleshooting

### Cache not found
```python
from pathlib import Path
cache_dir = Path('data/1h_template_cache')
if cache_dir.exists():
    cache = OneHaloTemplateCache()
else:
    # Run cache creation script first
    print("Run: python3 scripts/cache_effective_1h_templates.py")
```

### Loading wrong slope
```python
cache = OneHaloTemplateCache()
print("Available slopes:", cache.list_cached_slopes())
```

### Interpolation mismatch
```python
# Always interpolate template to your ell grid
template_at_ell = np.interp(ell_fit, ell_template, template_data)
```

## References

- [Effective 1H Template Documentation](EFFECTIVE_1H_TEMPLATE_README.md)
- [Auto/Cross Fitting Pipeline](scripts/auto_cross_fits_pipeline.py)
- [PowerSpectrum Model](ciber/theory/cross_ps_parametric_model.py)
