# Final Integration: Effective One-Halo Template with Auto/Cross Fitting

## Complete Solution Overview

You now have a fully integrated system for fitting DESI-LS and HSC z<1.0 auto/cross power spectra using the effective one-halo template computed from IHL decomposition.

### Three Components

1. **Cache System** - Stores computed templates for fast reuse
2. **Wrapper Script** - Simplified interface to run cross-spectrum fits
3. **Integration Guide** - Full documentation for advanced usage

---

## THE COMMAND YOU ASKED FOR

### Run z<1.0 Cross-Spectrum Fits with Effective 1H Template

```bash
python3 scripts/run_z_lt_1_cross_fits_with_template.py
```

That's it! This single command will:
1. Verify the effective 1h template is cached (or create it if missing)
2. Display template information
3. Run cross-spectrum fits for both DESI-LS and HSC with z<1.0 data
4. Save results to `data/cross_cl_fits/`

---

## Quick Start (3 Steps)

### Step 1: Create Cache (One-time)

```bash
python3 scripts/cache_effective_1h_templates.py
```

**Output:**
- Computes 5 dz=0.2 IHL templates (z=0.0-1.0)
- Fits each to extract 1h component
- Sums into effective template (peak at ℓ≈32,563)
- Saves to `data/1h_template_cache/`

**Time:** ~1-2 minutes

### Step 2: Run Cross-Spectrum Fits

```bash
python3 scripts/run_z_lt_1_cross_fits_with_template.py
```

**Default settings:**
- Catalogs: DESI-LS + HSC
- Multipole: lMax = 50,000
- Redshift bins: dz=0.2 (0.0-1.0)
- Fit label: z_lt_1.0_eff1h

**Output:** Fit results saved to `data/cross_cl_fits/`

### Step 3: Use Results

```python
import numpy as np
from ciber.io.ciber_data_utils import load_fit_results_npz
from ciber.theory.ihl_1h_template_cache import OneHaloTemplateCache

# Load cache
cache = OneHaloTemplateCache()
effective_1h, _, _ = cache.load_cache(slope=1.0)

# Load fit results
fit = load_fit_results_npz('data/cross_cl_fits/DESILS_coarsez/z_lt_1.0_eff1h_lMax50000.npz')

# Compare measured 1h to effective template
measured_1h = fit['dl_1h']
measured_norm = measured_1h / np.max(measured_1h)
template_norm = effective_1h[1.0]['one_halo_norm']

# Interpolate template to fit ell grid
template_at_ell = np.interp(fit['lb'], 
                             effective_1h[1.0]['ell'], 
                             template_norm)

# Calculate shape consistency
chi2 = np.sum(((measured_norm - template_at_ell) / error)**2)
print(f"Shape consistency: χ² = {chi2:.2f}")
```

---

## Customization

### Different Catalogs

```bash
# Just DESI-LS
python3 scripts/run_z_lt_1_cross_fits_with_template.py --cat DESILS

# Just HSC  
python3 scripts/run_z_lt_1_cross_fits_with_template.py --cat HSC
```

### Different lMax Values

```bash
# Single value
python3 scripts/run_z_lt_1_cross_fits_with_template.py --lmax 30000

# Multiple values
python3 scripts/run_z_lt_1_cross_fits_with_template.py --lmax 30000 50000 70000
```

### Custom Redshift Binning

```bash
# Wider bins
python3 scripts/run_z_lt_1_cross_fits_with_template.py --zbinedges 0.0 0.5 1.0

# Finer bins
python3 scripts/run_z_lt_1_cross_fits_with_template.py --zbinedges 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
```

### Combine Options

```bash
# DESI-LS only, multiple lMax, custom label
python3 scripts/run_z_lt_1_cross_fits_with_template.py \
  --cat DESILS \
  --lmax 40000 60000 \
  --fitstr custom_desils_label
```

### Overwrite Existing Fits

```bash
python3 scripts/run_z_lt_1_cross_fits_with_template.py --overwrite
```

### Just View Template Info (No Fitting)

```bash
python3 scripts/run_z_lt_1_cross_fits_with_template.py --no-run
```

---

## What's in the Effective Template

### Summary

```
Effective z<1.0 One-Halo Template (dz=0.2 bins)
├── Peak location: ℓ ≈ 32,563
├── Peak amplitude (sum): 34.01
├── Number of z-bins: 5
└── ℓ range: 320 - 728,555

Individual Contributions:
├── z=0.0-0.2: A_1h=10.20, ℓ_peak≈11,568
├── z=0.2-0.4: A_1h=9.47, ℓ_peak≈27,971
├── z=0.4-0.6: A_1h=6.55, ℓ_peak≈47,698
├── z=0.6-0.8: A_1h=5.89, ℓ_peak≈76,650
└── z=0.8-1.0: A_1h=4.88, ℓ_peak≈80,000
```

### Access Template Data

```python
from ciber.theory.ihl_1h_template_cache import OneHaloTemplateCache
import numpy as np

cache = OneHaloTemplateCache()
effective_1h, individual_1h, zbinedges = cache.load_cache(slope=1.0)

# Effective template properties
eff = effective_1h[1.0]
ell = eff['ell']
shape_norm = eff['one_halo_norm']        # Normalized shape [0, 1]
shape_sum = eff['one_halo_sum']          # Unnormalized sum
shape_avg = eff['one_halo_avg']          # Unnormalized average

# Individual z-bin properties
for zidx in individual_1h[1.0]:
    z_info = individual_1h[1.0][zidx]
    z_low, z_high = z_info['z_range']
    z_mid = z_info['z_mid']
    A_1h = z_info['A_1h']                # Amplitude
    mu_1h = z_info['mu_1h']              # Peak location (ln scale)
    sigma_1h = z_info['sigma_1h']        # Log-width
```

---

## Output Files

After running fits:

```
data/
├── 1h_template_cache/
│   ├── effective_1h_slope_1.0.npz
│   ├── individual_1h_slope_1.0.npz
│   └── cache_metadata.json
│
└── cross_cl_fits/
    ├── DESILS_coarsez/
    │   └── z_lt_1.0_eff1h_lMax*.npz (fit results)
    └── HSC_coarsez/
        └── z_lt_1.0_eff1h_lMax*.npz (fit results)

figures/
└── (Cross-spectrum plots if you run plot_cross mode)
```

---

## File Reference

### New Scripts
- `scripts/cache_effective_1h_templates.py` - Create cache
- `scripts/run_z_lt_1_cross_fits_with_template.py` - Run fits (✓ Recommended)

### New Modules
- `ciber/theory/ihl_1h_template_cache.py` - Cache system (300 lines)

### Documentation
- `INTEGRATION_1H_TEMPLATE_FITTING.md` - Complete integration guide
- `ONE_HALO_TEMPLATE_CACHING_SUMMARY.md` - Caching system details
- `RUN_COMMANDS_WITH_EFFECTIVE_1H.md` - Command examples
- `EFFECTIVE_1H_TEMPLATE_README.md` - Template computation

---

## Architecture

```
┌─────────────────────────────────────────────┐
│  IHL Templates (data/ihl_templates/*.txt)   │
└──────────────────┬──────────────────────────┘
                   │
     ┌─────────────▼──────────────────┐
     │ compute_effective_1h_template() │
     │   - Fit 5 dz=0.2 bins          │
     │   - Extract 1h components      │
     │   - Sum to get effective       │
     └──────────────┬───────────────────┘
                   │
    ┌──────────────▼────────────────────┐
    │ OneHaloTemplateCache.save_cache() │
    │  - effective_1h_slope_1.0.npz    │
    │  - individual_1h_slope_1.0.npz   │
    │  - cache_metadata.json            │
    └──────────────┬────────────────────┘
                   │
  ┌────────────────▼─────────────────────────┐
  │ run_z_lt_1_cross_fits_with_template.py   │
  │  - Loads cache                           │
  │  - Runs auto_cross_fits_pipeline.py      │
  │  - Shows template info                   │
  └────────────────┬─────────────────────────┘
                   │
  ┌────────────────▼──────────────────────┐
  │  DESI-LS & HSC Cross Fits (z<1.0)    │
  │  - Results in data/cross_cl_fits/     │
  └──────────────────────────────────────┘
         │
         └──> Compare measured 1h to effective template
              Calculate shape consistency (χ²)
              Generate summary plots
```

---

## Next Steps After Running Fits

1. **Compare Results**
   ```python
   # See examples above
   ```

2. **Generate Plots**
   ```bash
   python3 scripts/auto_cross_fits_pipeline.py \
     --mode plot_cross \
     --cat DESILS HSC \
     --lmax 50000
   ```

3. **Analyze Evolution**
   - Check how individual z-bins compare to effective template
   - Look for redshift dependence
   - Compare DESI-LS vs HSC

4. **Advanced Analysis**
   - See `INTEGRATION_1H_TEMPLATE_FITTING.md` for more options

---

## Support & Documentation

| Question | Resource |
|----------|----------|
| How do I use the cache? | `INTEGRATION_1H_TEMPLATE_FITTING.md` |
| What are the commands? | `RUN_COMMANDS_WITH_EFFECTIVE_1H.md` |
| How was the template made? | `EFFECTIVE_1H_TEMPLATE_README.md` |
| API reference | `ciber/theory/ihl_1h_template_cache.py` (docstrings) |

---

## Summary

✅ **Complete Solution Ready**

```bash
# One-time setup
python3 scripts/cache_effective_1h_templates.py

# Run z<1.0 cross fits
python3 scripts/run_z_lt_1_cross_fits_with_template.py

# That's it!
```

Results saved to `data/cross_cl_fits/` with effective 1h template cached and ready for comparison!
