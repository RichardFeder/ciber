# Effective One-Halo Template: Implementation Summary

## What Was Created

I've created a complete framework for computing an **effective one-halo template** by summing unnormalized 1h components from your dz=0.2 redshift bins. Here's what's included:

### Files Created

1. **`scripts/compute_effective_1h_template.py`** (Main Implementation)
   - `compute_effective_1h_template()`: Core function that:
     - Loads IHL templates for all dz=0.2 bins
     - Fits each template to extract 1h component
     - Sums the unnormalized components
     - Creates normalized version
     - Generates comparison plots
   - `create_comparison_plots()`: 4-panel visualization showing individual bins vs. effective template

2. **`scripts/example_effective_1h_usage.py`** (Example Usage)
   - Demonstrates how to use the main function
   - Shows how to access results for each redshift bin
   - Saves results to NPZ files for downstream use
   - Prints detailed analysis of peaks, amplitudes, and shapes

3. **`EFFECTIVE_1H_TEMPLATE_README.md`** (Complete Documentation)
   - Detailed explanation of the methodology
   - Usage examples (basic and advanced)
   - Interpretation guide
   - Troubleshooting section
   - Integration with cross-spectrum fitting

## How It Works

### Step-by-Step Process

```
For each dz=0.2 redshift bin (z=[0.0-0.2], [0.2-0.4], ..., [0.8-1.0]):
  ├─ Load IHL power spectrum template
  ├─ Fit template: D_ℓ = A_2h × (ℓ/ℓ₀)^α + A_1h × LogNormal(ℓ) + A_shot
  ├─ Extract 1h component: A_1h × LogNormal(ℓ; μ_1h, σ_1h)
  └─ Store unnormalized component

Compute effective template:
  ├─ one_halo_sum = Σ all_unnormalized_1h_components
  ├─ one_halo_avg = mean(all_unnormalized_1h_components)
  └─ one_halo_norm = one_halo_sum / max(one_halo_sum)

Create visualizations:
  ├─ Panel 1: All individual bins + sum (linear scale)
  ├─ Panel 2: Same components (log scale)
  ├─ Panel 3: Sum vs. Average comparison
  └─ Panel 4: Normalized shapes (all bins + effective)
```

### Output Structure

```python
effective_1h = {
    slope: {
        'ell': array,                  # Multipole grid
        'one_halo_sum': array,         # Sum of unnormalized 1h
        'one_halo_avg': array,         # Average of unnormalized 1h
        'one_halo_norm': array,        # Normalized version
        'n_bins_summed': int
    }
}

individual_1h = {
    slope: {
        zidx: {
            'z_range': (z_low, z_high),
            'z_mid': float,
            'ell': array,
            'one_halo': array,         # This z-bin's unnormalized 1h
            'A_1h': float,             # Fitted amplitude
            'mu_1h': float,            # Peak location (ln scale)
            'sigma_1h': float          # Log-width
        }
    }
}
```

## Key Features

### 1. Comprehensive Decomposition
- Uses the existing `fit_and_decompose_ihl_templates()` function
- Extracts smooth parametric 1h components (log-normal shapes)
- Preserves amplitude information from individual fits

### 2. Multiple Comparison Views
- **Linear scale**: See absolute amplitudes and contributions
- **Log scale**: Identify features across the full ℓ range
- **Sum vs. Average**: Understand how combining matters
- **Normalized shapes**: Compare profile shapes independent of amplitude

### 3. Redshift-Dependent Analysis
- Access individual bin properties: A_1h, μ_1h, σ_1h
- Identify redshift dependence of peak location and width
- Quantify relative contributions to the effective template

### 4. Downstream Integration
- Results saved as NPZ files for easy loading
- Compatible with your cross-spectrum fitting pipeline
- Can be used as reference or prior in further analysis

## Quick Start

```bash
# Run the example
cd /Users/richardfeder/Documents/ciber
python3 scripts/example_effective_1h_usage.py
```

This will:
1. Load all dz=0.2 IHL templates
2. Fit each one to extract 1h components
3. Compute the effective template
4. Create 4-panel comparison plots
5. Save results to `data/effective_1h_template_slope*.npz`
6. Print detailed analysis

## Customization Options

### Change Redshift Binning
```python
zbinedges = np.array([0.0, 0.3, 0.6, 0.9])  # Different binning
```

### Restrict Fitting Range
```python
effective_1h, ... = compute_effective_1h_template(
    ...,
    ell_fit_range=(100, 5000),  # Only fit small scales
)
```

### Multiple Slopes
```python
effective_1h, ... = compute_effective_1h_template(
    ...,
    slopes=[0.5, 1.0, 1.5],  # Compare different assumptions
)
```

### Access Individual Results
```python
for slope in individual_1h.keys():
    for zidx, z_info in individual_1h[slope].items():
        z_mid = z_info['z_mid']
        A_1h = z_info['A_1h']
        peak_ell = np.exp(z_info['mu_1h'])
        width = z_info['sigma_1h']
        # Your custom analysis...
```

## Questions This Answers

1. **What's the "typical" 1h shape across all redshifts?**
   → Use `effective_1h[slope]['one_halo_norm']` as reference

2. **How much does each z-bin contribute?**
   → Compare amplitudes in `individual_1h[slope][zidx]['A_1h']`

3. **Does the 1h peak location change with z?**
   → Track `mu_1h` values: higher μ → higher ℓ_peak

4. **Is the 1h width (σ_1h) consistent across z?**
   → Compare `sigma_1h` values in individual_1h

5. **How well do individual z-bins match the effective template?**
   → Check bottom-right plot: normalized shapes should align well if no strong evolution

## Integration with Cross-Spectrum Analysis

The effective template can be used as:

1. **Reference Shape**: Compare measured 1h components to this reference
2. **Template Prior**: Use as a prior or constraint in fitting
3. **Quality Check**: Verify individual z-bin measurements are consistent
4. **Uncertainty Estimation**: Use spread across bins to estimate systematic uncertainty

Example:
```python
# Load effective template
eff = np.load('data/effective_1h_template_slope1.0.npz')
eff_ell = eff['ell']
eff_1h_shape = eff['one_halo_norm']

# Compare to measured 1h in your cross-spectrum fit
measured_1h_shape = measured_1h_component / np.max(measured_1h_component)

# Calculate chi² for shape consistency
chi2_shape = np.sum(((measured_1h_shape - eff_1h_shape) / error)**2)
```

## Next Steps

1. **Review the plots**: Check `figures/effective_1h_template_slope*.png`
2. **Examine the redshift evolution**: Look at how A_1h, μ_1h, σ_1h vary with z
3. **Compare to your measurements**: Use as reference in your cross-spectrum analysis
4. **Adjust as needed**: Modify fitting range, binning, or slopes if desired
5. **Save results**: Effective templates are automatically saved to `data/`

## Technical Notes

- The function uses the existing `CrossPowerSpectrumModel` class for smooth parametric decomposition
- 1h components are extracted as smooth log-normal models (not raw template values)
- Sum (not average) is used to preserve total integrated amplitude information
- All plots are saved in `figures/` with resolution of 150 DPI
- Results can be loaded later with `np.load()` for reproducibility

## Questions or Issues?

See `EFFECTIVE_1H_TEMPLATE_README.md` for:
- Complete API documentation
- Troubleshooting guide
- Advanced usage examples
- Technical details on the decomposition model
