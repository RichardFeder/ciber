# Beam Correction in Quadratic Estimator Normalization

## Overview

The quadratic estimator (QE) for convergence $\kappa$ has been modified to account for instrumental beam effects in the normalization. This guide explains how to use the new beam correction functionality.

## Mathematical Background

### Standard QE Formula
The quadratic estimator for convergence is:
$$\hat{\kappa}_L = \frac{\int_\ell F_{\ell, L-\ell} I^{obs}_{\ell} I^{obs}_{L-\ell}}{\int_\ell F_{\ell, L-\ell} f^{\kappa}_{\ell, L-\ell}}$$

### Beam-Corrected QE Formula
When accounting for the beam $B_\ell$, the estimator becomes:
$$\hat{\kappa}_L = \frac{\int_\ell F_{\ell, L-\ell} I^{obs}_{\ell} I^{obs}_{L-\ell}}{\int_\ell F_{\ell, L-\ell} f^{\kappa}_{\ell, L-\ell} B_{\ell} B_{L-\ell}}$$

The beam function $B_\ell$ represents the Fourier transform of the point spread function (PSF).

## Implementation Details

### Modified Functions

1. **`computeQuadEstPhiNormalizationFFT()`** - Core normalization calculation
   - New parameter: `fB_ell` (optional beam function)
   - Applies beam corrections in Fourier space

2. **`computeQuadEstKappaNorm()`** - Normalized kappa estimator
   - New parameter: `fB_ell` (optional beam function)
   - Passes beam to normalization calculation

3. **`run_kappa_est()`** - Kappa estimation wrapper
   - New parameter: `fB_ell` (optional beam function)
   - Passes beam through pipeline

4. **`compute_ciber_kappa_products()`** - Full pipeline computation
   - Extracts `B_ell` from `cl_fns` dictionary
   - Automatically applies if present

### Where Beam Corrections are Applied

The beam corrections are applied in three places within `computeQuadEstPhiNormalizationFFT`:

1. **C map (Term 1)**: 
   - The factor $\frac{C_0(\ell)^2}{C_{tot}(\ell)}$ is multiplied by $B_\ell^2$
   - This accounts for the convolution: $(C \ast B) \ast B$

2. **Wiener Filter (Term 2)**:
   - The factor $\frac{C_0(\ell)}{C_{tot}(\ell)}$ is multiplied by $B_\ell$
   - This accounts for filtering through the beam once

## Usage Examples

### Example 1: Basic Usage with run_flatskyqe_ciber

```python
import numpy as np
from kappa_auto_cross_fns import run_flatskyqe_ciber

# Define your beam function
def ciber_beam(ell, sigma_arcmin=10.0):
    """
    Gaussian beam function.
    
    Parameters:
    -----------
    ell : float or array
        Multipole
    sigma_arcmin : float
        Beam width in arcminutes
    
    Returns:
    --------
    B_ell : float or array
        Beam transfer function
    """
    sigma_rad = sigma_arcmin * (np.pi / 180.0 / 60.0)  # Convert to radians
    return np.exp(-0.5 * ell**2 * sigma_rad**2)

# Add beam to cl_fns dictionary in build_cl_fns_and_corr_facs()
# The pipeline will automatically use it if present in cl_fns['B_ell']
```

### Example 2: Direct Use with FlatMap

```python
from FlatSkyQE.flat_map import FlatMap

# Initialize flat map
baseMap = FlatMap(nX=512, nY=512, sizeX=1.0, sizeY=1.0)

# Define power spectra
def C_unlensed(ell):
    # Your unlensed power spectrum
    return ...

def C_observed(ell):
    # Your observed power spectrum (includes noise)
    return ...

# Define beam function
def beam_function(ell):
    sigma_arcmin = 10.0
    sigma_rad = sigma_arcmin * (np.pi / 180.0 / 60.0)
    return np.exp(-0.5 * ell**2 * sigma_rad**2)

# Compute kappa with beam correction
dataFourier = baseMap.fourier(your_data_map)

kappa_Fourier, norm_Fourier = baseMap.computeQuadEstKappaNorm(
    fC0=C_unlensed,
    fCtot=C_observed,
    lMin=1000,
    lMax=50000,
    dataFourier=dataFourier,
    fB_ell=beam_function  # Pass beam function here
)

# Without beam correction (backwards compatible):
kappa_Fourier_no_beam, norm_Fourier_no_beam = baseMap.computeQuadEstKappaNorm(
    fC0=C_unlensed,
    fCtot=C_observed,
    lMin=1000,
    lMax=50000,
    dataFourier=dataFourier
    # fB_ell=None is default - no beam correction
)
```

### Example 3: CIBER-Specific Beam

```python
import numpy as np

def ciber_beam_from_fwhm(ell, fwhm_arcsec, inst=1):
    """
    CIBER beam function from FWHM.
    
    Parameters:
    -----------
    ell : float or array
        Multipole
    fwhm_arcsec : float
        Full width at half maximum in arcseconds
    inst : int
        CIBER instrument (1 or 2)
    
    Returns:
    --------
    B_ell : float or array
        Beam transfer function (Gaussian approximation)
    """
    # Convert FWHM to sigma
    sigma_rad = (fwhm_arcsec / 3600.0) * (np.pi / 180.0) / np.sqrt(8 * np.log(2))
    return np.exp(-0.5 * ell**2 * sigma_rad**2)

# CIBER instrument 1: ~7" FWHM at 1.1 micron
# CIBER instrument 2: ~7" FWHM at 1.8 micron
fwhm_ciber = 7.0  # arcseconds

beam_fn = lambda ell: ciber_beam_from_fwhm(ell, fwhm_ciber, inst=1)
```

### Example 4: Non-Gaussian Beam

```python
from scipy.interpolate import interp1d

def create_beam_from_profile(ell_samples, beam_samples):
    """
    Create beam function from measured profile.
    
    Parameters:
    -----------
    ell_samples : array
        Multipoles where beam is measured
    beam_samples : array
        Measured beam values
    
    Returns:
    --------
    beam_fn : function
        Interpolated beam function
    """
    # Create interpolator with extrapolation
    beam_interp = interp1d(
        ell_samples, 
        beam_samples,
        kind='cubic',
        bounds_error=False,
        fill_value=(beam_samples[0], 0.0)  # Constant at low ell, zero at high ell
    )
    
    def beam_fn(ell):
        result = beam_interp(ell)
        # Ensure non-negative and handle arrays
        if isinstance(result, np.ndarray):
            result[result < 0] = 0.0
        elif result < 0:
            result = 0.0
        return result
    
    return beam_fn

# Example usage with measured beam:
ell_meas = np.logspace(2, 5, 100)
beam_meas = np.exp(-0.5 * (ell_meas / 10000)**2)  # Example measurement
beam_fn = create_beam_from_profile(ell_meas, beam_meas)
```

## Integration with Existing Pipeline

### In `run_flatskyqe_ciber`

The beam is passed through the `cl_fns` dictionary returned by `build_cl_fns_and_corr_facs()`. 

To enable beam correction, modify your call to include the beam:

```python
# In your analysis script:
cl_fns, corr_facs = build_cl_fns_and_corr_facs(
    clf, param_dict, config_dict,
    ld.ciber_unlensed_auto, ld.ciber_obs_auto, ld.mask
)

# Add beam function to cl_fns
cl_fns['B_ell'] = your_beam_function

# The rest of the pipeline will automatically use it
results = compute_ciber_kappa_products(
    baseMap, map_dict, cl_fns, param_dict, config_dict, corr_facs, paths
)
```

## Backwards Compatibility

**All code remains backwards compatible!** 

- If `fB_ell` is not provided (or is `None`), the functions behave exactly as before
- No beam correction is applied when `fB_ell=None`
- Existing scripts will continue to work without modification

## Testing Your Beam Function

```python
import matplotlib.pyplot as plt

# Test your beam function
ell_test = np.logspace(2, 5, 1000)
beam_test = np.array([your_beam_fn(l) for l in ell_test])

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.loglog(ell_test, beam_test)
plt.xlabel('$\\ell$')
plt.ylabel('$B_\\ell$')
plt.title('Beam Transfer Function')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.semilogx(ell_test, beam_test)
plt.xlabel('$\\ell$')
plt.ylabel('$B_\\ell$')
plt.title('Beam Transfer Function (Linear)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Test that beam is properly normalized
print(f"B(ell=0) ≈ {your_beam_fn(1e-10):.6f} (should be close to 1.0)")
print(f"B(ell=10000) = {your_beam_fn(10000):.6e}")
```

## Common Beam Functions

### 1. Gaussian Beam
```python
def gaussian_beam(ell, fwhm_arcmin):
    sigma_rad = (fwhm_arcmin / 60.0) * (np.pi / 180.0) / np.sqrt(8 * np.log(2))
    return np.exp(-0.5 * ell**2 * sigma_rad**2)
```

### 2. Airy Disk (Circular Aperture)
```python
from scipy.special import j1

def airy_beam(ell, D_meters, wavelength_meters):
    """Airy disk beam for circular aperture."""
    theta = ell * wavelength_meters / D_meters
    x = np.pi * theta
    # Prevent division by zero
    result = np.where(x > 1e-10, (2 * j1(x) / x)**2, 1.0)
    return result
```

### 3. Pixelized Beam
```python
def pixel_window(ell, pixel_size_arcmin):
    """Square pixel window function."""
    theta = pixel_size_arcmin * (np.pi / 180.0 / 60.0)
    x = 0.5 * ell * theta
    return np.where(x > 1e-10, np.sin(x) / x, 1.0)**2
```

## Validation

To validate the beam correction is working:

1. **Compare with and without beam**: Run the same data with and without beam correction
2. **Check normalization scaling**: The normalization should be modified by factors of $B_\ell^2$ 
3. **Inspect term contributions**: Use `test=True` in computeQuadEstPhiNormalizationFFT to plot individual terms

```python
# Validation example
norm_with_beam = baseMap.computeQuadEstPhiNormalizationFFT(
    fC0, fCtot, lMin=1000, lMax=50000, 
    test=True,  # Plot intermediate results
    fB_ell=beam_fn
)

norm_without_beam = baseMap.computeQuadEstPhiNormalizationFFT(
    fC0, fCtot, lMin=1000, lMax=50000,
    test=True
)

# Compare
L = baseMap.l.flatten()
ratio = (norm_with_beam / norm_without_beam).flatten()
plt.loglog(L, ratio, 'b.', alpha=0.5)
plt.xlabel('$L$')
plt.ylabel('$N^{\\rm with\\ beam}_L / N^{\\rm no\\ beam}_L$')
plt.title('Effect of Beam Correction on Normalization')
plt.show()
```

## References

- Hu & Okamoto (2002) - Quadratic estimator formalism
- Planck 2018 Lensing paper - Beam treatment in lensing reconstruction
- Your CIBER beam characterization papers

## Questions?

If you encounter issues or have questions about the beam correction implementation, check:
1. Is your beam function normalized correctly? ($B_{\ell=0} \approx 1$)
2. Is the beam smooth and well-behaved?
3. Are you passing the beam through the correct parameter in the pipeline?

---
Last updated: December 17, 2025
