# Effective One-Halo Template Computation

## Overview

This toolkit computes an **effective one-halo template** by summing the unnormalized one-halo (1h) components extracted from the IHL power spectra across all dz=0.2 redshift bins (z=0.0–1.0). The resulting effective template can be compared against measurements from individual redshift slices to assess consistency and redshift dependence.

## Key Concepts

### One-Halo Components
The IHL power spectrum for each redshift bin is decomposed into:
- **Two-Halo (2h)**: Large-scale power-law component
- **One-Halo (1h)**: Peaked component, typically log-normal in shape
- **Shot Noise**: Scale-independent contribution

### Effective Template
The effective 1h template is computed by:
1. Extracting the unnormalized 1h component from each dz=0.2 bin
2. **Summing** all 1h components (not averaging) to get the total integrated signal
3. Creating a normalized version (divided by its maximum)

This approach preserves the amplitude information from the individual bins while creating a single reference template.

## Usage

### Basic Usage

```python
from compute_effective_1h_template import compute_effective_1h_template
import numpy as np

# Define redshift bins
zbinedges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# Compute effective template
effective_1h, individual_1h, fit_results = compute_effective_1h_template(
    template_dir='data/ihl_templates',
    zbinedges=zbinedges,
    slopes=[1.0],
    plot=True,
    figsize=(14, 10)
)
```

### Understanding the Output

**`effective_1h[slope]` dictionary contains:**
- `'ell'`: Multipole grid (from first template)
- `'one_halo_sum'`: Sum of unnormalized 1h components from all bins
- `'one_halo_avg'`: Average of unnormalized 1h components
- `'one_halo_norm'`: Normalized version (sum/max) for shape comparison
- `'n_bins_summed'`: Number of redshift bins included

**`individual_1h[slope][zidx]` contains per-redshift-bin data:**
- `'z_range'`: (z_low, z_high) tuple
- `'z_mid'`: Redshift bin center
- `'ell'`: Multipole grid
- `'one_halo'`: Unnormalized 1h component
- `'A_1h'`: Fitted amplitude (from log-normal decomposition)
- `'mu_1h'`: Peak location in log-space (ln(ℓ_peak))
- `'sigma_1h'`: Log-width of the log-normal

### Example Script

```bash
python3 scripts/example_effective_1h_usage.py
```

This script:
1. Loads all dz=0.2 IHL templates
2. Fits each one to extract 1h components
3. Computes the effective template
4. Creates comparison plots
5. Saves results to NPZ files

## Output

### Plots
Generated in `figures/`:
- `effective_1h_template_slope1.0.png`: 4-panel comparison showing:
  - **Top Left**: Individual bin 1h components (linear scale) with effective sum overlay
  - **Top Right**: Same but log-log scale for better visibility at all scales
  - **Bottom Left**: Sum vs. average comparison
  - **Bottom Right**: Normalized shapes of all bins and the effective template

### Data Files
Saved in `data/`:
- `effective_1h_template_slope{slope}.npz`: Contains the effective template arrays for downstream use

## Interpreting Results

### Key Questions Answered

1. **Is the 1h peak location consistent across z?**
   - Check if μ_1h varies significantly with redshift
   - The effective template peak location indicates the "average" peak

2. **Does the 1h amplitude scale with redshift?**
   - Compare A_1h values across the individual_1h dictionary
   - Larger A_1h at higher z typically indicates more 1h contribution

3. **Is the 1h shape (width, σ_1h) redshift-dependent?**
   - Examine σ_1h values in individual_1h
   - Compare normalized shapes in the bottom-right plot

4. **How much does each z-bin contribute to the total?**
   - Compare individual 1h component amplitudes in top plots
   - Larger curves indicate more contribution to the effective template

## Advanced Usage

### Custom Fitting Range

```python
# Only fit to small scales (e.g., ℓ < 10,000)
effective_1h, individual_1h, fit_results = compute_effective_1h_template(
    template_dir='data/ihl_templates',
    zbinedges=zbinedges,
    slopes=[1.0],
    ell_fit_range=(100, 10000),  # Restrict fitting range
    plot=True
)
```

### Multiple Slopes

```python
# Compare different slope assumptions (1.0, 0.5, 1.5)
effective_1h, individual_1h, fit_results = compute_effective_1h_template(
    template_dir='data/ihl_templates',
    zbinedges=zbinedges,
    slopes=[0.5, 1.0, 1.5],  # Multiple slopes
    plot=True
)

# Access results for each slope
for slope in effective_1h.keys():
    print(f"Peak location for slope {slope}: {ell[np.argmax(effective_1h[slope]['one_halo_norm'])]} ")
```

### Custom Analysis

```python
# Access and analyze individual results
for slope in individual_1h.keys():
    for zidx in range(len(zbinedges) - 1):
        z_info = individual_1h[slope][zidx]
        z_mid = z_info['z_mid']
        A_1h = z_info['A_1h']
        mu_1h = z_info['mu_1h']
        
        # Your custom analysis here
        print(f"z={z_mid:.2f}: A_1h={A_1h:.2e}, peak_ell={np.exp(mu_1h):.0f}")
```

## File Structure

```
.
├── data/
│   ├── ihl_templates/
│   │   ├── ihl_ps_z_0.0_0.2_slope_1.0.txt
│   │   ├── ihl_ps_z_0.2_0.4_slope_1.0.txt
│   │   ├── ihl_ps_z_0.4_0.6_slope_1.0.txt
│   │   ├── ihl_ps_z_0.6_0.8_slope_1.0.txt
│   │   └── ihl_ps_z_0.8_1.0_slope_1.0.txt
│   └── effective_1h_template_slope*.npz      # Output files
├── figures/
│   └── effective_1h_template_slope*.png      # Comparison plots
└── scripts/
    ├── compute_effective_1h_template.py      # Main computation function
    └── example_effective_1h_usage.py         # Example usage script
```

## Integration with Cross-Spectrum Fitting

The effective 1h template can be used as a reference in your cross-spectrum fitting pipeline:

```python
# Load the effective template
eff_data = np.load('data/effective_1h_template_slope1.0.npz')
eff_ell = eff_data['ell']
eff_1h = eff_data['one_halo_norm']  # Use normalized version

# Use as reference shape in fitting, or as a prior on 1h component shape
# in your galaxy cross-spectrum analysis
```

## Technical Details

### Decomposition Model
Each IHL template is fitted with:
```
D_ℓ = A_2h * (ℓ/ℓ₀)^α + A_1h * LogNormal(ℓ; μ, σ) + A_shot
```

Where:
- **LogNormal**: Represents the 1h component shape
- **μ = ln(ℓ_peak)**: Peak location
- **σ**: Log-width of the distribution

### Effective Template Computation
```
D_eff(ℓ) = Σ_{z_bins} A_1h(z) * LogNormal(ℓ; μ(z), σ(z))
```

The sum naturally weights contributions by their amplitudes, so high-amplitude bins contribute more to the effective shape.

## Troubleshooting

### Error: "Template directory not found"
Ensure IHL template files are in the specified directory with the correct naming convention:
```
ihl_ps_z_{z_low}_{z_high}_slope_{slope}.txt
```

### Warning: "Fit result for z_X_Y not found"
Check that:
1. The template file exists for that redshift bin
2. The fitting procedure completed successfully for that bin
3. The file format is correct (with header row)

### Peak location seems unrealistic
Check the `fit_ell_range` parameter - restricting the fitting range might improve results if you have noisy data at certain scales.

## Related Functions

- `load_ihl_templates()`: Load raw IHL template data
- `fit_and_decompose_ihl_templates()`: Decompose templates into 2h, 1h, shot components
- `get_ihl_components_at_ell()`: Extract smooth model components at specific ℓ values
- `interpolate_1h_params()`: Use fitted 1h parameters for arbitrary redshifts

See `ciber/theory/cl_template.py` for full API documentation.

## Citation

If you use the effective 1h template computation in your work, please cite the IHL template data source and the CIBER cross-correlation analysis paper.
