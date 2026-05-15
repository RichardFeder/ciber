# Effective One-Halo Template: Quick Start Guide

## What Was Fixed

The effective 1h template computation scripts now work correctly! Two issues were resolved:

1. **Missing import** in `cl_template.py` - Added CrossPowerSpectrumModel import
2. **Incorrect data structure access** - Updated to match actual fit results structure

## Running the Code

### Option 1: Run the Example (Recommended)

```bash
cd /Users/richardfeder/Documents/ciber
python3 scripts/example_effective_1h_usage.py
```

This will:
- Load 5 dz=0.2 IHL templates (z=0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0)
- Fit each template to extract 1h components
- Compute the effective template by summing components
- Create 4-panel comparison plot
- Save results to NPZ file
- Print detailed analysis

**Expected runtime:** ~1-2 minutes

### Option 2: Use in Your Own Code

```python
from compute_effective_1h_template import compute_effective_1h_template
import numpy as np

# Compute effective template
zbinedges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
effective_1h, individual_1h, fit_results = compute_effective_1h_template(
    template_dir='data/ihl_templates',
    zbinedges=zbinedges,
    slopes=[1.0],
    plot=True
)

# Access results
for slope in effective_1h.keys():
    eff_template = effective_1h[slope]
    ell = eff_template['ell']
    one_halo_norm = eff_template['one_halo_norm']
    
    # Your analysis here...
```

## What You Get

### Plots
**File:** `figures/effective_1h_template_slope1.0.png`

Four panels showing:
1. **Top-Left (Linear):** Individual 1h components + effective sum
2. **Top-Right (Log-Log):** Same data on log scales
3. **Bottom-Left:** Sum vs. Average comparison
4. **Bottom-Right:** Normalized shapes (all bins + effective)

### Data
**File:** `data/effective_1h_template_slope1.0.npz`

Contains:
- `ell`: Multipole grid
- `one_halo_sum`: Summed unnormalized components
- `one_halo_avg`: Averaged unnormalized components
- `one_halo_norm`: Normalized version for shape comparison
- `zbinedges`: Redshift bin edges
- `slope`: Slope value used

Load with:
```python
data = np.load('data/effective_1h_template_slope1.0.npz')
ell = data['ell']
one_halo_norm = data['one_halo_norm']
```

## Key Results from Example Run

### Individual Redshift Bins

| z-bin | A_1h | ℓ_peak | σ |
|-------|------|--------|-----|
| 0.0-0.2 | 10.20 | 11,568 | 1.751 |
| 0.2-0.4 | 9.47 | 27,971 | 1.825 |
| 0.4-0.6 | 6.55 | 47,698 | 1.906 |
| 0.6-0.8 | 5.89 | 76,650 | 1.906 |
| 0.8-1.0 | 4.88 | 80,000 | 1.726 |

### Key Observations

✅ **Peak location shifts with z:** Low-z peaks at ~12k, high-z peaks at ~80k
✅ **Amplitude decreases with z:** From ~10 to ~5
✅ **Width relatively constant:** σ ≈ 1.7-1.9 across all bins
✅ **Effective template peak:** ℓ ≈ 32,563 (weighted average)

## Use Cases

### 1. Reference Template
Use the normalized effective template as a reference shape:
```python
eff_norm = data['one_halo_norm']
measured_shape = measured_1h / np.max(measured_1h)

# Compare shapes
chi2 = np.sum(((measured_shape - eff_norm) / error)**2)
```

### 2. Redshift Evolution Study
Track how individual z-bins deviate from effective template:
```python
for zidx in individual_1h[slope]:
    z_info = individual_1h[slope][zidx]
    # Study redshift dependence
```

### 3. Cross-Spectrum Fitting
Use as a prior or constraint:
```python
# In your fitting code
prior_1h_shape = eff_norm
# Apply as regularization or comparison
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'ciber'"
Make sure you're in the project root directory:
```bash
cd /Users/richardfeder/Documents/ciber
python3 scripts/example_effective_1h_usage.py
```

### "FileNotFoundError: Template directory not found"
Verify template files exist:
```bash
ls -la data/ihl_templates/
# Should show ihl_ps_z_*.txt files
```

### Plots not saving
Verify figures directory exists:
```bash
mkdir -p figures
```

## Next Steps

1. ✅ Review the generated plot: `figures/effective_1h_template_slope1.0.png`
2. ✅ Load the data: `np.load('data/effective_1h_template_slope1.0.npz')`
3. ✅ Compare to your measurements
4. ✅ Use in downstream analysis (cross-spectrum fitting, etc.)

## Related Files

- **Implementation:** `scripts/compute_effective_1h_template.py`
- **Example:** `scripts/example_effective_1h_usage.py`
- **Documentation:** `EFFECTIVE_1H_TEMPLATE_README.md`
- **Technical Details:** `EFFECTIVE_1H_IMPLEMENTATION_GUIDE.md`
- **What Was Fixed:** `EFFECTIVE_1H_TEMPLATE_FIX_LOG.md`
- **Architecture:** `EFFECTIVE_1H_TEMPLATE_SUMMARY.md`

## Questions?

Refer to the comprehensive documentation files listed above for:
- Complete API documentation
- Advanced usage examples
- Technical implementation details
- Troubleshooting guide
