# Using Beam Correction in cib_lensing_mocks.ipynb

## Summary

Good news! Your `delta_fn_sources_test_v2` function **already supports beam correction** - it just needed one small fix in `compute_lensing_ps_quantities_v2` to pass the beam to the normalization.

## What Was Changed

**Modified file:** `mock_lens_test.py`
- Updated `compute_lensing_ps_quantities_v2()` to pass `fB_ell` to `run_kappa_est()`

**No changes needed to:**
- `delta_fn_sources_test_v2()` - already passes `B_ell` in `cl_fns` dict
- Your notebook - the beam correction is automatically applied when `grab_cib_sim=True`!

## How It Works Now

### Automatic Beam Correction (CIBER PSF)

When you call `delta_fn_sources_test_v2` with `grab_cib_sim=True`:

```python
res = delta_fn_sources_test_v2(
    nsim=5, 
    lMax=80000, 
    lMin=1e4, 
    add_noise=False,
    sigma_noise_pix=40, 
    psf_pix_fwhm=None,  # None means use CIBER beam
    grab_cib_sim=True,   # This loads the CIBER PSF
    datestr='111125'
)
```

**What happens internally:**
1. Lines 469-472 of `mock_lens_test.py`:
   ```python
   elif grab_cib_sim:
       clf.load_bl(ciber_inst, ifield, inplace=True, plot=False)
       B_ell_fn = clf.bl
   ```

2. Line 604: Beam is added to `cl_fns` dictionary:
   ```python
   cl_fns = dict({'cib_unlensed_auto':cib_unlensed_auto, 'obs_auto':obs_auto, 
                  'W_ell':W_ell, 'B_ell':b_ell_use})
   ```

3. The beam is automatically used in:
   - **QE normalization** (now fixed!)
   - **Bispectrum correction** (already working)
   - **CIB power spectrum correction** (already working)

### Gaussian Beam Correction

When you use a Gaussian PSF:

```python
res = delta_fn_sources_test_v2(
    nsim=5, 
    lMax=80000, 
    lMin=1e4,
    psf_pix_fwhm=2.0,  # Gaussian beam with 2 pixel FWHM
    grab_cib_sim=False
)
```

**What happens:**
1. Lines 467-469:
   ```python
   if param_dict['psf_pix_fwhm'] is not None:
       param_dict['sigma_b'] = psf_pix_fwhm_to_sigma_rad(...)
       B_ell_fn = lambda ell: np.exp(-(ell*param_dict['sigma_b'])**2/2)
   ```

2. Gaussian beam is used everywhere

### No Beam Correction

If you want to disable beam correction:

```python
res = delta_fn_sources_test_v2(
    nsim=5,
    psf_pix_fwhm=None,   # No Gaussian beam
    grab_cib_sim=False    # Don't load CIBER beam
)
```

This sets `b_ell_use = None` (line 600), so no beam correction is applied.

## Your Existing Notebook Cell

Your current cell (lines 380-471) will now automatically use beam correction in the QE normalization:

```python
for apply_mask in [False, True]:
    for lMax in lMax_range:
        for add_noise in [False, True]:
            for psf_pix_fwhm in fwhms:
                
                # Your existing code...
                res = delta_fn_sources_test_v2(
                    nsim=nsim, lMax=lMax, lMin=1e4, 
                    add_noise=add_noise,
                    sigma_noise_pix=sigma_noise_pix, 
                    psf_pix_fwhm=psf_pix_fwhm,
                    N_CIB_PER_PIXEL=0.2, 
                    N_G_PER_PIXEL=0.2, 
                    alpha=0., 
                    plot=False, 
                    grab_cib_sim=grab_cib_sim,  # ← Beam correction enabled when True
                    datestr='111125', 
                    apply_mask=apply_mask, 
                    pixel_fn_correct=pixel_fn_correct, 
                    lensmode=lensmode, 
                    mockstr=mockstr
                )
```

**No changes needed to your notebook!** The beam correction is now properly applied to the QE normalization.

## Verification

To verify the beam correction is working, you can check that:

1. When `grab_cib_sim=True`, you should see in the output:
   ```
   Doing full calculation: computeQuadEstPhiNormalizationFFT
   Including beam corrections in normalization
   ```

2. Compare results with and without beam:
   ```python
   # With CIBER beam
   res_with_beam = delta_fn_sources_test_v2(..., grab_cib_sim=True, psf_pix_fwhm=None)
   
   # Without beam
   res_no_beam = delta_fn_sources_test_v2(..., grab_cib_sim=False, psf_pix_fwhm=None)
   
   # Check difference
   ratio = res_with_beam['clkg'] / res_no_beam['clkg']
   ```

## What Gets Beam-Corrected?

When beam correction is enabled:

| Quantity | Beam Correction Applied? | Where |
|----------|-------------------------|-------|
| **QE Normalization** | ✅ YES (NEW!) | `computeQuadEstPhiNormalizationFFT()` |
| **Bispectrum** | ✅ YES | `proc_skewspec()` in `compute_lensing_ps_quantities_v2()` |
| **CIB Auto-Power** | ✅ YES | Lines 972-974 of `mock_lens_test.py` |
| **κ-g Cross-Power** | ❌ NO | QE estimator is beam-independent |

## Optional: Add Diagnostic Output

If you want to see when beam correction is active, add this to your notebook:

```python
# After calling delta_fn_sources_test_v2
if psf_pix_fwhm is None and grab_cib_sim:
    print("✓ Beam correction: CIBER PSF")
elif psf_pix_fwhm is not None:
    print(f"✓ Beam correction: Gaussian FWHM={psf_pix_fwhm} pix")
else:
    print("○ Beam correction: DISABLED")
```

## Summary

✅ **No changes needed to your notebook**
✅ **Beam correction automatically applied when `grab_cib_sim=True`**
✅ **Backwards compatible - old behavior when beam is None**
✅ **Optional - controlled by `grab_cib_sim` and `psf_pix_fwhm` parameters**

Your existing mock tests will now use beam-corrected QE normalization!
