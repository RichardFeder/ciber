# Commands to Run Z<1.0 Cross-Spectrum Fits with Effective 1H Template

## Quick Start

### Step 1: Create Cache (One-time)

```bash
cd /Users/richardfeder/Documents/ciber
python3 scripts/cache_effective_1h_templates.py
```

**Expected output:**
- ✓ Loading and fitting 5 dz=0.2 redshift bins
- ✓ Computing effective template (peak at ℓ≈32,563)
- ✓ Saving to `data/1h_template_cache/`

**Time:** ~1-2 minutes

---

### Step 2: Run Cross-Spectrum Fits

#### Default (DESI-LS + HSC, lMax=50000, dz=0.2 bins)

```bash
python3 scripts/run_z_lt_1_cross_fits_with_template.py
```

#### Custom Options

**Just DESI-LS:**
```bash
python3 scripts/run_z_lt_1_cross_fits_with_template.py --cat DESILS
```

**Just HSC:**
```bash
python3 scripts/run_z_lt_1_cross_fits_with_template.py --cat HSC
```

**Multiple lMax values:**
```bash
python3 scripts/run_z_lt_1_cross_fits_with_template.py --lmax 30000 50000 70000
```

**Custom redshift bins:**
```bash
python3 scripts/run_z_lt_1_cross_fits_with_template.py --zbinedges 0.0 0.3 0.6 0.9
```

**Overwrite existing fits:**
```bash
python3 scripts/run_z_lt_1_cross_fits_with_template.py --overwrite
```

**Custom fit string label:**
```bash
python3 scripts/run_z_lt_1_cross_fits_with_template.py --fitstr z_lt_1_custom_label
```

**Just show template info (no fitting):**
```bash
python3 scripts/run_z_lt_1_cross_fits_with_template.py --no-run
```

---

## Direct Command (Advanced)

If you prefer to use the main pipeline directly:

```bash
python3 scripts/auto_cross_fits_pipeline.py \
  --mode run_cross \
  --cat DESILS HSC \
  --lmax 50000 \
  --fitstr-cross z_lt_1.0_eff1h \
  --zbinedges 0.0 0.2 0.4 0.6 0.8 1.0
```

---

## Full Workflow Example

### Run everything end-to-end:

```bash
# Step 1: Create cache (first time only)
python3 scripts/cache_effective_1h_templates.py

# Step 2: Run DESI-LS cross fits
python3 scripts/run_z_lt_1_cross_fits_with_template.py --cat DESILS --lmax 50000

# Step 3: Run HSC cross fits
python3 scripts/run_z_lt_1_cross_fits_with_template.py --cat HSC --lmax 50000

# Step 4: Generate comparison plots (use existing pipeline)
python3 scripts/auto_cross_fits_pipeline.py \
  --mode plot_cross \
  --cat DESILS HSC \
  --lmax 50000
```

---

## Output Files

After running, you'll find:

```
data/cross_cl_fits/
├── DESILS_coarsez/
│   └── *.npz (fit results)
└── HSC_coarsez/
    └── *.npz (fit results)

figures/
├── ciber_LS_crosscorr_*.pdf (plots)
└── ... (additional comparison plots)
```

---

## Understanding the Output

The fit results contain:

```python
fit_result = {
    'lb': multipole grid,
    'params': [A_2h, A_1h, A_shot, ...],
    'dl_2h': two-halo component,
    'dl_1h': one-halo component,
    'dl_shot': shot noise,
    'dl_clustering': 2h + 1h (no shot),
    'dl_total': full model,
    'chisq': chi-squared value,
    ...
}
```

---

## Comparing to Effective Template

Once fits are complete, compare measured 1h to effective template:

```python
import numpy as np
from ciber.theory.ihl_1h_template_cache import OneHaloTemplateCache
from ciber.io.ciber_data_utils import load_fit_results_npz

# Load cache
cache = OneHaloTemplateCache()
effective_1h, _, _ = cache.load_cache(slope=1.0)

# Load a fit result
fit_result = load_fit_results_npz('data/cross_cl_fits/DESILS_coarsez/some_fit.npz')

# Get measured 1h component
measured_1h = fit_result['dl_1h']
measured_1h_norm = measured_1h / np.max(measured_1h)

# Get effective template
eff_template = effective_1h[1.0]['one_halo_norm']
eff_ell = effective_1h[1.0]['ell']

# Interpolate to fit ell grid
template_at_ell = np.interp(fit_result['lb'], eff_ell, eff_template)

# Compare
chi2 = np.sum(((measured_1h_norm - template_at_ell) / error)**2)
print(f"Shape consistency: χ² = {chi2:.2f}")

# Plot comparison
plt.figure()
plt.loglog(fit_result['lb'], measured_1h_norm, 'o-', label='Measured')
plt.loglog(eff_ell, eff_template, 's--', label='Effective template')
plt.xlabel('Multipole (ℓ)')
plt.ylabel('Normalized 1h shape')
plt.legend()
plt.show()
```

---

## Troubleshooting

### Error: "Cache not found"

Run the cache creation script first:
```bash
python3 scripts/cache_effective_1h_templates.py
```

### Error: "Module not found"

Make sure you're in the correct directory:
```bash
cd /Users/richardfeder/Documents/ciber
```

### Fits are very slow

Try using smaller lMax values:
```bash
python3 scripts/run_z_lt_1_cross_fits_with_template.py --lmax 30000
```

Or restrict the multipole fitting range (edit the pipeline code).

### Want to use different 1h template

Create a new cache with custom settings:
```python
from ciber.theory.ihl_1h_template_cache import create_and_cache_effective_1h_template
import numpy as np

zbinedges = np.array([0.0, 0.5, 1.0])  # Custom bins
create_and_cache_effective_1h_template(
    zbinedges=zbinedges,
    cache_dir='data/1h_template_cache_custom'
)
```

---

## Key Parameters

| Parameter | Default | Options | Notes |
|-----------|---------|---------|-------|
| `--cat` | DESILS HSC | DESILS, HSC | Catalogs to fit |
| `--lmax` | 50000 | integer | Multipole max |
| `--fitstr` | z_lt_1.0_eff1h | string | Output label |
| `--zbinedges` | 0.0 0.2 ... 1.0 | floats | Redshift bins |
| `--overwrite` | False | flag | Recompute |
| `--no-run` | False | flag | Show info only |

---

## Documentation

For complete details, see:
- `INTEGRATION_1H_TEMPLATE_FITTING.md` - Integration guide
- `ONE_HALO_TEMPLATE_CACHING_SUMMARY.md` - Cache system summary
- `EFFECTIVE_1H_TEMPLATE_README.md` - Template computation details

---

## Support

Questions? Check the integration guide:
```bash
cat /Users/richardfeder/Documents/ciber/INTEGRATION_1H_TEMPLATE_FITTING.md
```

Or look at the inline documentation:
```bash
python3 scripts/run_z_lt_1_cross_fits_with_template.py --help
```
