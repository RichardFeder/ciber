"""
Halo model templates for cross-power spectra using pyccl

This module provides functions to compute 1-halo and 2-halo terms
for galaxy × intensity cross-correlations using the Limber approximation.

Author: Generated for CIBER analysis
Date: January 2026
"""

import numpy as np
import pyccl as ccl


def get_1h_2h_templates(ells, z_grid, dndz, cosmo_params=None, 
                        z_CIB=None, bias=1.0, compute_1h=True,
                        verbose=False):
    """
    Compute 1-halo and 2-halo C_ell templates for galaxy × CIB cross-correlation.
    
    Uses pyccl to compute matter power spectra and projects using Limber approximation.
    
    Parameters
    ----------
    ells : array_like
        Multipole values to compute
    z_grid : array_like
        Redshift grid for galaxy distribution (should be fine enough, e.g., dz~0.01)
    dndz : array_like
        Galaxy redshift distribution dN/dz (normalized or unnormalized, same length as z_grid)
    cosmo_params : dict, optional
        Dictionary of cosmological parameters for pyccl.Cosmology. If None, uses Planck18.
        Keys can include: Omega_c, Omega_b, h, sigma8, n_s, etc.
    z_CIB : array_like, optional
        CIB intensity redshift kernel (same length as z_grid). If None, assumes
        CIB traces galaxies (i.e., z_CIB = dndz).
    bias : float or array_like, optional
        Linear galaxy bias. If scalar, assumed constant with z. If array, must match z_grid.
    compute_1h : bool, optional
        Whether to compute 1-halo term (approximate model). Default True.
    verbose : bool, optional
        Print diagnostic information
    
    Returns
    -------
    cl_2h : array_like
        Two-halo C_ell (linear clustering term)
    cl_1h : array_like or None
        One-halo C_ell (shot noise + small-scale clustering). None if compute_1h=False.
    
    Notes
    -----
    - The 2-halo term uses the linear matter power spectrum.
    - The 1-halo term is a simple model: constant shot noise + power-law damping at high-ell.
      For precise 1-halo predictions, use halo model codes like HMCode or CCL halo model.
    - Uses Limber approximation: k = (ell + 0.5) / chi(z)
    
    Example
    -------
    >>> import numpy as np
    >>> ells = np.logspace(2, 4, 30)
    >>> z_grid = np.linspace(0.1, 3.0, 100)
    >>> dndz = np.exp(-((z_grid - 1.5)/0.3)**2)  # Gaussian at z=1.5
    >>> cl_2h, cl_1h = get_1h_2h_templates(ells, z_grid, dndz)
    """
    
    # Setup cosmology
    if cosmo_params is None:
        # Default to Planck18
        cosmo = ccl.Cosmology(
            Omega_c=0.26,
            Omega_b=0.049,
            h=0.67,
            sigma8=0.81,
            n_s=0.96,
            transfer_function='boltzmann_camb'
        )
        if verbose:
            print("Using default Planck18-like cosmology")
    else:
        cosmo = ccl.Cosmology(**cosmo_params)
        if verbose:
            print(f"Using custom cosmology: {cosmo_params}")
    
    # Normalize dndz
    z_grid = np.asarray(z_grid)
    dndz = np.asarray(dndz)
    dndz_norm = dndz / np.trapz(dndz, z_grid)
    
    # CIB kernel (if not provided, assume it traces galaxies)
    if z_CIB is None:
        z_CIB = dndz_norm
    else:
        z_CIB = np.asarray(z_CIB)
        z_CIB = z_CIB / np.trapz(z_CIB, z_grid)  # Normalize
    
    # Handle bias
    if np.isscalar(bias):
        bias_z = np.full_like(z_grid, bias)
    else:
        bias_z = np.asarray(bias)
        if len(bias_z) != len(z_grid):
            raise ValueError("bias array must match z_grid length")
    
    # Precompute comoving distances
    a_grid = 1.0 / (1.0 + z_grid)
    chi_grid = ccl.comoving_radial_distance(cosmo, a_grid)
    H_over_c = ccl.h_over_h0(cosmo, a_grid) * cosmo['h'] / 2997.92458  # H(z)/c in h/Mpc
    
    # Limber kernel: W(z) = dN/dz * bias(z) / chi(z)^2 * c/H(z)
    # For cross-correlation: W_gal * W_CIB
    W_gal = dndz_norm * bias_z / chi_grid**2 / H_over_c
    W_CIB = z_CIB / chi_grid**2 / H_over_c
    
    # 2-halo term: integrate P_lin(k, z) with Limber
    ells = np.asarray(ells)
    cl_2h = np.zeros_like(ells, dtype=float)
    
    if verbose:
        print(f"Computing 2-halo term for {len(ells)} multipoles...")
    
    for i, ell in enumerate(ells):
        # Limber approximation: k(z) = (ell + 0.5) / chi(z)
        k_grid = (ell + 0.5) / chi_grid
        
        # Get linear matter power at these k and z
        # Call element-wise to avoid broadcasting issues in some pyccl versions
        P_lin = np.array([ccl.linear_matter_power(cosmo, k_grid[j], a_grid[j]) 
                         for j in range(len(z_grid))])
        
        # Limber integrand: W_gal(z) * W_CIB(z) * P_lin(k(z), z)
        integrand = W_gal * W_CIB * P_lin
        
        # Integrate over redshift (returns scalar)
        cl_2h[i] = np.trapz(integrand, z_grid)
    
    # 1-halo term (simplified model)
    if compute_1h:
        if verbose:
            print("Computing 1-halo term (simplified model)...")
        
        # Simple 1-halo model: shot noise + damped power law
        # C_ell^1h ≈ A_shot + A_1h * (ell / ell_0)^alpha * exp(-(ell/ell_cut)^2)
        # This is a placeholder; for real analysis, use halo model or fit to sims
        
        # Estimate shot noise level from high-ell limit
        # For now, use a fixed fraction of 2-halo at low ell
        A_shot = 0.1 * np.mean(cl_2h[:3]) if len(cl_2h) > 3 else 0.01 * cl_2h[0]
        
        # 1-halo amplitude (empirical: ~10-30% of 2-halo at ell~1000-3000)
        A_1h = 0.2 * np.interp(2000, ells, cl_2h)
        ell_0 = 2000.0  # Peak around ell~2000
        alpha = -0.5    # Mild power law
        ell_cut = 5000.0  # Exponential cutoff
        
        cl_1h = A_shot + A_1h * (ells / ell_0)**alpha * np.exp(-(ells / ell_cut)**2)
        
        if verbose:
            print(f"  1-halo shot noise: {A_shot:.3e}")
            print(f"  1-halo amplitude: {A_1h:.3e}")
    else:
        cl_1h = None
    
    if verbose:
        print("Done.")
    
    return cl_2h, cl_1h


def get_1h_2h_templates_with_tracers(ells, z_grid, dndz_gal, dndz_CIB=None,
                                      cosmo_params=None, bias_gal=1.0, bias_CIB=1.0,
                                      use_nonlin_2h=False, verbose=False):
    """
    Compute 1h/2h templates using pyccl's tracer framework (simpler, more robust).
    
    This uses ccl.NumberCountsTracer and ccl.angular_cl to compute projected power.
    
    Parameters
    ----------
    ells : array_like
        Multipoles to compute
    z_grid : array_like
        Redshift grid
    dndz_gal : array_like
        Galaxy dN/dz
    dndz_CIB : array_like, optional
        CIB intensity redshift kernel. If None, uses dndz_gal.
    cosmo_params : dict, optional
        Cosmology parameters for ccl.Cosmology
    bias_gal : float or (z, b(z)) tuple, optional
        Galaxy bias (constant or z-dependent)
    bias_CIB : float or (z, b(z)) tuple, optional
        CIB bias
    use_nonlin_2h : bool, optional
        If True, uses nonlinear matter power for 2-halo. Default False (linear).
    verbose : bool, optional
        Print info
    
    Returns
    -------
    cl_total : array_like
        Total C_ell from ccl.angular_cl (includes 1h + 2h internally)
    
    Notes
    -----
    This method is simpler and uses CCL's internal Limber implementation.
    The 1-halo/2-halo split is handled by CCL's halo model (if configured).
    By default, angular_cl uses linear matter power (2-halo dominated).
    """
    
    # Setup cosmology
    if cosmo_params is None:
        cosmo = ccl.Cosmology(
            Omega_c=0.26, Omega_b=0.049, h=0.67, sigma8=0.81, n_s=0.96,
            transfer_function='boltzmann_camb',
            matter_power_spectrum='linear' if not use_nonlin_2h else 'halofit'
        )
    else:
        cosmo = ccl.Cosmology(**cosmo_params)
    
    # Create galaxy tracer
    z_grid = np.asarray(z_grid)
    dndz_gal = np.asarray(dndz_gal)
    
    # Handle bias
    if np.isscalar(bias_gal):
        bias_gal_z = (z_grid, np.full_like(z_grid, bias_gal))
    else:
        bias_gal_z = bias_gal  # assume (z, b(z)) tuple
    
    tracer_gal = ccl.NumberCountsTracer(
        cosmo, has_rsd=False, dndz=(z_grid, dndz_gal), bias=bias_gal_z
    )
    
    # Create CIB tracer (or reuse galaxy if not provided)
    if dndz_CIB is None:
        tracer_CIB = tracer_gal
    else:
        dndz_CIB = np.asarray(dndz_CIB)
        if np.isscalar(bias_CIB):
            bias_CIB_z = (z_grid, np.full_like(z_grid, bias_CIB))
        else:
            bias_CIB_z = bias_CIB
        tracer_CIB = ccl.NumberCountsTracer(
            cosmo, has_rsd=False, dndz=(z_grid, dndz_CIB), bias=bias_CIB_z
        )
    
    # Compute angular power spectrum
    ells = np.asarray(ells)
    cl_total = ccl.angular_cl(cosmo, tracer_gal, tracer_CIB, ells)
    
    if verbose:
        print(f"Computed C_ell using ccl.angular_cl (2-halo dominated)")
        print(f"  Matter power: {'nonlinear' if use_nonlin_2h else 'linear'}")
    
    return cl_total


# Example usage
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Example: compute templates for HSC galaxy × CIBER CIB cross
    ells = np.logspace(2.5, 4, 30)
    z_grid = np.linspace(0.1, 3.5, 150)
    
    # Example galaxy dN/dz (peaked at z~1.5)
    dndz_gal = np.exp(-((z_grid - 1.5) / 0.4)**2)
    
    # Example CIB kernel (broader, peaks at z~2)
    dndz_CIB = np.exp(-((z_grid - 2.0) / 0.6)**2)
    
    print("Computing 1h/2h templates with manual Limber projection...")
    cl_2h, cl_1h = get_1h_2h_templates(
        ells, z_grid, dndz_gal, z_CIB=dndz_CIB, 
        bias=1.5, verbose=True
    )
    
    print("\nComputing with ccl.angular_cl (tracer method)...")
    cl_total = get_1h_2h_templates_with_tracers(
        ells, z_grid, dndz_gal, dndz_CIB=dndz_CIB,
        bias_gal=1.5, bias_CIB=1.0, verbose=True
    )
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(ells, cl_2h, 'b-', label='2-halo (manual Limber)', linewidth=2)
    if cl_1h is not None:
        ax.loglog(ells, cl_1h, 'g--', label='1-halo (simplified)', linewidth=2)
        ax.loglog(ells, cl_2h + cl_1h, 'r-', label='Total (2h+1h)', linewidth=2, alpha=0.7)
    ax.loglog(ells, cl_total, 'k:', label='CCL angular_cl', linewidth=2)
    
    ax.set_xlabel(r'$\ell$', fontsize=14)
    ax.set_ylabel(r'$C_\ell$', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_title('Galaxy × CIB Cross-Power Templates', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("\nExample completed!")
