# Effective One-Halo Template: Implementation Guide

## Architecture Overview

### Main Components

```
compute_effective_1h_template.py
├── compute_effective_1h_template()
│   ├── Load and fit IHL templates for all dz=0.2 bins
│   ├── Extract unnormalized 1h components
│   ├── Sum components to create effective template
│   └── Return results + create plots
│
└── create_comparison_plots()
    ├── Panel 1: Linear scale comparison
    ├── Panel 2: Log-log scale comparison
    ├── Panel 3: Sum vs. Average
    └── Panel 4: Normalized shapes

example_effective_1h_usage.py
├── analyze_effective_template()
│   ├── Print peak locations
│   ├── Print amplitudes for each z-bin
│   ├── Print fitted parameters (μ_1h, σ_1h)
│   └── Save to NPZ files
│
└── main()
    ├── Call compute_effective_1h_template()
    └── Call analyze_effective_template()
```

## Function Signatures

### Main Function

```python
def compute_effective_1h_template(
    template_dir: str,              # Path to IHL templates
    zbinedges: array_like,          # Redshift bin edges
    slopes: list = [1.0],           # Slope values to process
    ell_fit_range: tuple = None,    # Optional: (ell_min, ell_max)
    plot: bool = True,              # Create plots
    figsize: tuple = (14, 10)       # Figure size
) -> tuple:
    """
    Returns:
        effective_1h: dict
            {slope: {
                'ell': array,
                'one_halo_sum': array,
                'one_halo_avg': array,
                'one_halo_norm': array,
                'n_bins_summed': int
            }}
        
        individual_1h: dict
            {slope: {
                zidx: {
                    'z_range': (z_low, z_high),
                    'z_mid': float,
                    'ell': array,
                    'one_halo': array,
                    'A_1h': float,
                    'mu_1h': float,
                    'sigma_1h': float
                }
            }}
        
        fit_results: dict
            Results from fit_and_decompose_ihl_templates()
    """
```

### Plotting Function

```python
def create_comparison_plots(
    effective_1h: dict,         # Effective template data
    individual_1h: dict,        # Individual z-bin data
    zbinedges: array_like,      # Redshift bin edges
    figsize: tuple = (14, 10)   # Figure size
) -> None:
    """
    Creates 4-panel plots for each slope and saves to figures/
    """
```

## Data Flow

```
IHL Template Files (data/ihl_templates/)
    ↓
fit_and_decompose_ihl_templates()
    ↓
[A_2h, A_1h, μ_1h, σ_1h, A_shot] for each z-bin
    ↓
get_ihl_components_at_ell()
    ├─ Evaluates smooth log-normal: A_1h × exp(-0.5*((ln(ℓ)-μ)²/σ²))
    └─ Returns: {two_halo, one_halo, shot_noise, total}
    ↓
Extract 'one_halo' component for each z-bin
    ↓
Sum all unnormalized 1h components
    ├─ one_halo_sum = Σ D^1h(ℓ, z)
    ├─ one_halo_avg = mean(D^1h(ℓ, z))
    └─ one_halo_norm = one_halo_sum / max(one_halo_sum)
    ↓
create_comparison_plots()
    ↓
Output files:
    ├─ figures/effective_1h_template_slope*.png
    └─ data/effective_1h_template_slope*.npz (optional)
```

## Key Algorithm Steps

### 1. Template Loading and Fitting

```python
# Uses existing function
fit_results = fit_and_decompose_ihl_templates(
    template_dir=template_dir,
    zbinedges=zbinedges,
    slopes=slopes,
    use_powerlaw_2h=True,
    alpha_2h_fixed=0.0,
    fit_ell_range=ell_fit_range,
    plot=False,
    verbose=True
)

# Results structure
fits = {
    'z0.0_0.2_slope1.0': {
        'params': [A_2h, A_1h, μ_1h, σ_1h, A_shot],
        'template_data': {'ell': array, 'dl': array},
        ...
    },
    'z0.2_0.4_slope1.0': {...},
    ...
}
```

### 2. Component Extraction

```python
for each_fit_result:
    components = get_ihl_components_at_ell(fit_result, ell_grid)
    # Returns smooth parametric model evaluated at ell_grid
    one_halo_component = components['one_halo']
    # Shape: [n_ell]
```

### 3. Summation

```python
all_1h_components = []
for zidx in range(nzbins):
    one_halo = get_ihl_components_at_ell(...)['one_halo']
    all_1h_components.append(one_halo)  # Shape: [n_ell]

# Stack and sum
all_1h_array = np.array(all_1h_components)  # Shape: [nzbins, n_ell]
one_halo_sum = np.sum(all_1h_array, axis=0)  # Shape: [n_ell]
```

### 4. Normalization

```python
one_halo_norm = one_halo_sum / np.max(one_halo_sum)
# Range: [min_value/max, 1.0]
```

## Extracting Information

### Get Peak Location

```python
peak_idx = np.argmax(one_halo_norm)
peak_ell = ell[peak_idx]

# For individual z-bins
for zidx in individual_1h[slope]:
    z_info = individual_1h[slope][zidx]
    peak_ell_z = np.exp(z_info['mu_1h'])
```

### Get Amplitudes

```python
# Effective template
max_amplitude = np.max(one_halo_sum)

# Individual contributions
for zidx in individual_1h[slope]:
    z_info = individual_1h[slope][zidx]
    A_1h = z_info['A_1h']
    contribution_fraction = np.max(z_info['one_halo']) / max_amplitude
```

### Get Shape Parameters

```python
for zidx in individual_1h[slope]:
    z_info = individual_1h[slope][zidx]
    z_mid = z_info['z_mid']
    mu_1h = z_info['mu_1h']           # ln(ℓ_peak)
    sigma_1h = z_info['sigma_1h']     # Log-width
    ell_peak = np.exp(mu_1h)          # Physical peak location
    
    # FWHM ≈ 2.355 × σ_1h in log-space
    fwhm_log = 2.355 * sigma_1h
```

## Comparison with Individual Measurements

### Aligned Comparison

```python
# Get effective template
eff_norm = effective_1h[slope]['one_halo_norm']
eff_ell = effective_1h[slope]['ell']

# For each z-bin
for zidx in individual_1h[slope]:
    z_info = individual_1h[slope][zidx]
    z_norm = z_info['one_halo'] / np.max(z_info['one_halo'])
    z_ell = z_info['ell']
    
    # Interpolate effective to z_ell grid
    interp_eff = np.interp(z_ell, eff_ell, eff_norm)
    
    # Compare
    residual = z_norm - interp_eff
    chi2 = np.sum((residual / error)**2)
```

## Customization Examples

### Example 1: Different Redshift Binning

```python
# Use coarser binning
zbinedges = np.array([0.0, 0.5, 1.0])

effective_1h, individual_1h, _ = compute_effective_1h_template(
    template_dir='data/ihl_templates',
    zbinedges=zbinedges,
    slopes=[1.0],
    plot=True
)
```

### Example 2: Restricted Scale Range

```python
# Only fit small scales where 1h dominates
effective_1h, individual_1h, _ = compute_effective_1h_template(
    template_dir='data/ihl_templates',
    zbinedges=np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    slopes=[1.0],
    ell_fit_range=(100, 5000),  # ← Only fit in this range
    plot=True
)
```

### Example 3: Multiple Slopes Comparison

```python
# Compare different 2h power-law assumptions
for slope_2h in [0.0, 0.5, 1.0]:
    # Note: Current code uses alpha_2h_fixed=0.0
    # You would need to modify fit_and_decompose_ihl_templates
    # to vary this parameter
    pass
```

## Saving and Loading Results

### Save Effective Template

```python
# Manual saving
np.savez(
    'data/effective_1h_template_slope1.0.npz',
    ell=effective_1h[1.0]['ell'],
    one_halo_sum=effective_1h[1.0]['one_halo_sum'],
    one_halo_avg=effective_1h[1.0]['one_halo_avg'],
    one_halo_norm=effective_1h[1.0]['one_halo_norm'],
    zbinedges=zbinedges,
    slope=1.0
)
```

### Load and Use

```python
# Load effective template
data = np.load('data/effective_1h_template_slope1.0.npz')
ell = data['ell']
one_halo_norm = data['one_halo_norm']

# Use in your analysis
# ...
```

## Performance Considerations

### Computation Time

- **Template loading**: ~1-2 seconds per bin
- **Fitting**: ~5-10 seconds per bin (5 z-bins ≈ 25-50 seconds)
- **Component extraction**: <1 second total
- **Plotting**: ~5-10 seconds

Total: ~30-60 seconds for full pipeline

### Memory Usage

- Templates: ~50-100 MB (depends on ℓ resolution)
- Fits: ~1 MB
- Plots: Built and immediately saved/closed

### Optimization

If fitting is slow, consider:
```python
# Use coarser ℓ grid (e.g., bin templates to Δℓ = 10)
# Use restricted fit range
ell_fit_range=(100, 10000)  # Avoid high-ℓ noise
```

## Extending the Code

### Add Custom Analysis

```python
def analyze_effective_template_custom(effective_1h, individual_1h, zbinedges):
    """Your custom analysis function"""
    for slope in effective_1h:
        eff = effective_1h[slope]
        # Your analysis here
```

### Modify Fitting Model

```python
# In fit_and_decompose_ihl_templates, you can change:
# - use_powerlaw_2h: True/False (power-law vs log-normal 2h)
# - alpha_2h_fixed: fixed exponent for 2h power-law
# - Fitting bounds and initial guesses
```

### Add More Decomposition Products

```python
# In get_ihl_components_at_ell, components also include:
components['two_halo']     # Power-law component
components['shot_noise']   # White noise component
components['total']        # Full model
```

## Validation Checklist

- [ ] Template files exist in `data/ihl_templates/`
- [ ] All z-bins have corresponding templates
- [ ] Fits converged (check console output)
- [ ] Peaks are physically reasonable (ℓ_peak ~1000-5000)
- [ ] Amplitudes decrease or stay consistent with z (or show expected trend)
- [ ] Plots show reasonable individual bins and effective template
- [ ] Output files saved successfully

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'ciber'"
```python
# Make sure config is imported first
import config  # This sets ciber_basepath
from ciber.theory.cl_template import ...
```

### Issue: Very slow fitting
```python
# Try smaller ℓ range
ell_fit_range=(100, 5000)

# Or coarser ℓ binning in template files
```

### Issue: Fits not converging
```python
# Check template file format (should have header row)
# Try adjusting initial guesses in fit_and_decompose_ihl_templates()
# Try smaller ℓ fitting range to focus on dominant 1h region
```
