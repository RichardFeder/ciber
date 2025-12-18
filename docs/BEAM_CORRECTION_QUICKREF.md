# Quick Reference: Beam Correction

## TL;DR

Add beam correction to your quadratic estimator in 3 lines:

```python
# 1. Define beam function
beam_fn = lambda ell: np.exp(-0.5 * ell**2 * sigma**2)

# 2. Add to cl_fns dictionary (after build_cl_fns_and_corr_facs)
cl_fns['B_ell'] = beam_fn

# 3. Run as normal - beam automatically applied!
results = compute_ciber_kappa_products(baseMap, map_dict, cl_fns, ...)
```

## What Changed?

| Function | New Parameter | Description |
|----------|---------------|-------------|
| `computeQuadEstPhiNormalizationFFT()` | `fB_ell=None` | Core: applies beam in normalization |
| `computeQuadEstKappaNorm()` | `fB_ell=None` | Passes beam to normalization |
| `run_kappa_est()` | `fB_ell=None` | Passes beam through pipeline |
| `compute_ciber_kappa_products()` | Uses `cl_fns['B_ell']` | Extracts beam from cl_fns dict |

All parameters are **optional** - existing code works unchanged!

## Where Beam is Applied

```
Normalization: N_L = 1 / ∫_ℓ F_{ℓ,L-ℓ} f^κ_{ℓ,L-ℓ} B_ℓ B_{L-ℓ}
                                                      ↑    ↑
                                                      Added here!
```

**Term 1 (C map):**
- `C₀(ℓ)²/C_tot(ℓ)` → `C₀(ℓ)² B(ℓ)² / C_tot(ℓ)`

**Term 2 (Wiener filter):**
- `C₀(ℓ)/C_tot(ℓ)` → `C₀(ℓ) B(ℓ) / C_tot(ℓ)`

## Common Beam Functions

### CIBER Gaussian (7" FWHM)
```python
def ciber_beam(ell):
    sigma_rad = (7.0/3600) * (np.pi/180) / np.sqrt(8*np.log(2))
    return np.exp(-0.5 * ell**2 * sigma_rad**2)
```

### General Gaussian
```python
def gaussian_beam(ell, fwhm_arcmin):
    sigma_rad = (fwhm_arcmin/60) * (np.pi/180) / np.sqrt(8*np.log(2))
    return np.exp(-0.5 * ell**2 * sigma_rad**2)
```

### From Measurements
```python
from scipy.interpolate import interp1d
beam_interp = interp1d(ell_measured, beam_measured, 
                       kind='cubic', bounds_error=False, fill_value=0)
```

## Quick Checks

### Test Your Beam
```python
# Should be close to 1
print(f"B(ell→0) = {beam_fn(1e-10):.4f}")

# Plot it
import matplotlib.pyplot as plt
ell = np.logspace(2, 5, 1000)
plt.loglog(ell, [beam_fn(l) for l in ell])
plt.xlabel('ℓ'); plt.ylabel('B_ℓ'); plt.show()
```

### Compare With/Without Beam
```python
# Run twice
cl_fns['B_ell'] = None  # Without
results_no_beam = compute_ciber_kappa_products(...)

cl_fns['B_ell'] = beam_fn  # With
results_with_beam = compute_ciber_kappa_products(...)

# Check difference
ratio = results_with_beam['clxs'][0] / results_no_beam['clxs'][0]
```

## File Structure

```
FlatSkyQE/
├── flat_map.py                      [MODIFIED] Core QE functions
├── kappa_auto_cross_fns.py          [MODIFIED] Pipeline functions  
├── BEAM_CORRECTION_GUIDE.md         [NEW] Comprehensive guide
├── BEAM_CORRECTION_SUMMARY.md       [NEW] Implementation details
├── BEAM_CORRECTION_QUICKREF.md      [NEW] This file
└── example_beam_correction.py       [NEW] Example code
```

## FAQ

**Q: Will this break my existing code?**  
A: No! All changes are backwards compatible. If you don't provide a beam, behavior is identical to before.

**Q: Do I need to modify my data maps?**  
A: No! The beam correction is only in the **normalization**. Your data stays the same.

**Q: Which beam function should I use?**  
A: For CIBER: 7 arcsec FWHM Gaussian is a good start. Refine based on your beam measurements.

**Q: Can I turn it off after enabling?**  
A: Yes! Just set `cl_fns['B_ell'] = None` or remove the key.

**Q: Does this affect the numerator of the QE?**  
A: No! The beam correction is only in the **denominator** (normalization). The numerator (quadratic combination of maps) is unchanged.

**Q: What about cross-spectra between different bands?**  
A: Currently uses same beam for both. For different beams, you'd need to extend the implementation.

## Example Output

With beam correction enabled, you should see:
```
Doing full calculation: computeQuadEstPhiNormalizationFFT
Including beam corrections in normalization
lMin, lMax inputs to computeQuadEstKappaNorm: 1000 50000
```

## Performance

- ✅ No overhead when beam=None
- ✅ Minimal overhead when beam provided (~same runtime)
- ✅ Caching still works
- ✅ Parallel-safe

## References

📖 Full guide: [BEAM_CORRECTION_GUIDE.md](BEAM_CORRECTION_GUIDE.md)  
📋 Details: [BEAM_CORRECTION_SUMMARY.md](BEAM_CORRECTION_SUMMARY.md)  
💻 Examples: [example_beam_correction.py](example_beam_correction.py)

---
Last updated: December 17, 2025
