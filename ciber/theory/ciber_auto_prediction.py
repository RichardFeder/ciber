"""
Predict CIBER intensity auto-spectrum from galaxy cross and auto spectra

Author: Richard Feder
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def fit_galaxy_auto_clustering(lb, cl_gal_auto, cl_gal_auto_err=None,
                                fit_range=(300, 80000), chi2_eval_max=80000,
                               initial_guess=None, verbose=True,
                               nwalkers=32, nsteps=2000, nburn=500,
                               z_value=None, ln_ell_peak_relation=None):
    """
    Fit galaxy auto-spectrum with 2-halo + 1-halo + shot noise model using MCMC.
    
    Uses CrossPowerSpectrumModel.fit_model_mcmc() to perform robust MCMC fitting.
    Returns the smooth clustering component (2h+1h) without shot noise.
    
    Parameters
    ----------
    lb : array-like
        Multipole bin centers
    cl_gal_auto : array-like
        Galaxy auto-spectrum C_ℓ
    cl_gal_auto_err : array-like, optional
        Uncertainties on galaxy auto-spectrum
    fit_range : tuple, optional
        (ℓ_min, ℓ_max) range for fitting. Default (300, 80000)
    chi2_eval_max : float, optional
        Maximum ℓ for chi-squared evaluation
    initial_guess : dict, optional
        Initial parameter guesses (not used with MCMC)
    verbose : bool, optional
        Print fit results
    nwalkers : int, optional
        Number of MCMC walkers (default 32)
    nsteps : int, optional
        Number of MCMC steps (default 2000)
    nburn : int, optional
        Number of burn-in steps (default 500)
    z_value : float, optional
        Redshift value for fixing ln_ell_peak via linear relation
    ln_ell_peak_relation : dict, optional
        Linear relation for ln_ell_peak: {'slope': 7.4, 'intercept': 8.44}
        If provided with z_value, fixes ln_ell_peak = intercept + slope * z
    
    Returns
    -------
    dict
        Dictionary with:
        - 'params': Best-fit parameters [A_2h, A_1h, mu, sigma, A_shot] (or reduced if mu fixed)
        - 'params_err': parameter uncertainties
        - 'dl_2h': 2-halo component D_ℓ
        - 'dl_1h': 1-halo component D_ℓ
        - 'dl_shot': shot noise component D_ℓ
        - 'dl_clustering': 2-halo + 1-halo (no shot)
        - 'dl_total': full model
        - 'cl_clustering': clustering C_ℓ (no shot)
        - 'chisq': chi-squared
        - 'ndof': degrees of freedom
        - 'fit_result_mcmc': full MCMC fit result
    """
    from ciber.theory.cross_ps_parametric_model import CrossPowerSpectrumModel
    
    lb = np.asarray(lb)
    cl_gal_auto = np.asarray(cl_gal_auto)
    
    # Convert to D_ℓ
    pf = lb * (lb + 1) / (2 * np.pi)
    dl_gal_auto = pf * cl_gal_auto
    if cl_gal_auto_err is not None:
        dl_gal_auto_err = pf * np.asarray(cl_gal_auto_err)
        if verbose:
            print(f"Input C_ℓ errors: min={np.min(cl_gal_auto_err):.3e}, max={np.max(cl_gal_auto_err):.3e}")
            print(f"Converted D_ℓ errors: min={np.min(dl_gal_auto_err):.3e}, max={np.max(dl_gal_auto_err):.3e}")
            print(f"Typical SNR in C_ℓ: {np.median(cl_gal_auto/cl_gal_auto_err):.1f}")
            print(f"Typical SNR in D_ℓ: {np.median(dl_gal_auto/dl_gal_auto_err):.1f}")
    else:
        dl_gal_auto_err = None
    
    # Set up ln_ell_peak relation if provided
    # Note: CrossPowerSpectrumModel expects (intercept, slope) tuple
    if ln_ell_peak_relation is None and z_value is not None:
        # Use default relation: ln_ell_peak = 8.44 + 7.4*z
        ln_ell_peak_relation = (8.44, 7.4)  # (intercept, slope)
    
    if verbose and ln_ell_peak_relation is not None and z_value is not None:
        ln_ell_peak_fixed = ln_ell_peak_relation[0] + ln_ell_peak_relation[1] * z_value
        print(f"Fixing ln(ℓ_peak) = {ln_ell_peak_fixed:.2f} (ℓ_peak ~ {np.exp(ln_ell_peak_fixed):.0f}) at z={z_value:.2f}")
    
    # Create model instance
    model = CrossPowerSpectrumModel(
        lb,
        use_powerlaw_2h=True,
        alpha_2h_fixed=0.0,  # Constant 2-halo in D_ℓ
        chi2_eval_max=chi2_eval_max,
        use_lorentzian_1h=False,  # Use log-normal for 1-halo
        ln_ell_peak_relation=ln_ell_peak_relation,  # Pass relation as (intercept, slope)
    )
    
    # Apply fit range
    mask = (lb >= fit_range[0]) & (lb <= fit_range[1])
    lb_fit = lb[mask]
    dl_fit = dl_gal_auto[mask]
    if dl_gal_auto_err is not None:
        dlerr_fit = dl_gal_auto_err[mask]
    else:
        dlerr_fit = None
    
    if verbose:
        print(f"Fitting galaxy auto with MCMC ({nwalkers} walkers, {nsteps} steps, {nburn} burn-in)...")
        print(f"Fit range: ℓ ∈ [{fit_range[0]}, {fit_range[1]}]")
        print(f"Chi² evaluation: ℓ < {chi2_eval_max}")
    
    # Fit using MCMC
    try:
        fit_result_mcmc = model.fit_model_mcmc(
            lb_fit,
            dl_fit,
            dl_err=dlerr_fit,
            fit_range=fit_range,
            chi2_eval_max=chi2_eval_max,
            nwalkers=nwalkers,
            nsteps=nsteps,
            nburn=nburn,
            progress=True,
            verbose=verbose,
            z_value=z_value  # Pass z_value for fixing ln_ell_peak
        )
        
        # Extract best-fit parameters
        params = fit_result_mcmc['params']
        params_err = fit_result_mcmc['params_err']
        
        # Generate model components at all ℓ using best-fit parameters
        # params = [A_2h, A_1h, mu_1h, sigma_1h, A_shot]
        components = model.model_components(lb, *params)
        
        dl_2h = components['two_halo']
        dl_1h = components['one_halo']
        dl_shot = components['shot_noise']
        dl_total = components['total']
        dl_clustering = dl_2h
        
        # Convert clustering back to C_ℓ
        cl_clustering = dl_clustering / pf
        
        if verbose:
            print(f"\nGalaxy auto MCMC fit results:")
            if ln_ell_peak_relation is not None:
                # 4-parameter case: [A_2h, A_1h, sigma, A_shot]
                print(f"  A_2h = {params[0]:.3e} ± {params_err[0]:.3e}")
                print(f"  A_1h = {params[1]:.3e} ± {params_err[1]:.3e}")
                ln_ell_peak_fixed = ln_ell_peak_relation[0] + ln_ell_peak_relation[1] * z_value
                print(f"  mu = {ln_ell_peak_fixed:.3f} (fixed from relation, ℓ_peak ~ {np.exp(ln_ell_peak_fixed):.0f})")
                print(f"  sigma = {params[2]:.3f} ± {params_err[2]:.3f}")
                print(f"  A_shot = {params[3]:.3e} ± {params_err[3]:.3e}")
            else:
                # 5-parameter case: [A_2h, A_1h, mu, sigma, A_shot]
                print(f"  A_2h = {params[0]:.3e} ± {params_err[0]:.3e}")
                print(f"  A_1h = {params[1]:.3e} ± {params_err[1]:.3e}")
                print(f"  mu = {params[2]:.3f} ± {params_err[2]:.3f} (ℓ_peak ~ {np.exp(params[2]):.0f})")
                print(f"  sigma = {params[3]:.3f} ± {params_err[3]:.3f}")
                print(f"  A_shot = {params[4]:.3e} ± {params_err[4]:.3e}")
            print(f"  χ²/dof = {fit_result_mcmc['chisq']:.1f}/{fit_result_mcmc['ndof']} = {fit_result_mcmc['reduced_chisq']:.2f}")
        
        # Determine parameter names based on whether peak is fixed
        if ln_ell_peak_relation is not None:
            param_names = ['A_2h', 'A_1h', 'sigma', 'A_shot']
        else:
            param_names = ['A_2h', 'A_1h', 'mu', 'sigma', 'A_shot']
        
        return {
            'params': params,
            'params_err': params_err,
            'param_names': param_names,
            'dl_2h': dl_2h,
            'dl_1h': dl_1h,
            'dl_shot': dl_shot,
            'dl_clustering': dl_clustering,  # 2h + 1h (no shot)
            'dl_total': dl_total,
            'cl_clustering': cl_clustering,  # Convert back to C_ℓ
            'chisq': fit_result_mcmc['chisq'],
            'ndof': fit_result_mcmc['ndof'],
            'reduced_chisq': fit_result_mcmc['reduced_chisq'],
            'fit_result_mcmc': fit_result_mcmc  # Keep full MCMC result
        }
        
    except Exception as e:
        print(f"MCMC fit failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_galaxy_auto_fit(lb, dl_gal_auto, fit_result, dl_gal_auto_err=None, 
                         fit_range=(300, 80000), title=None, ax=None):
    """
    Plot galaxy auto D_ℓ with fitted model components.
    
    Parameters
    ----------
    lb : array-like
        Multipole values
    dl_gal_auto : array-like
        Observed galaxy auto D_ℓ spectrum
    fit_result : dict
        Output from fit_galaxy_auto_clustering()
    dl_gal_auto_err : array-like, optional
        Errors on observed spectrum
    fit_range : tuple, optional
        (ℓ_min, ℓ_max) range used for fitting
    title : str, optional
        Plot title
    ax : matplotlib axis, optional
        Axis to plot on. If None, creates new figure.
        
    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.get_figure()
    
    # Extract components
    dl_2h = fit_result['dl_2h']
    dl_1h = fit_result['dl_1h']
    dl_shot = fit_result['dl_shot']
    dl_clustering = fit_result['dl_clustering']
    dl_total = fit_result['dl_total']
    params = fit_result['params']
    params_err = fit_result['params_err']
    
    # Plot observed data
    if dl_gal_auto_err is not None:
        ax.errorbar(lb, dl_gal_auto, yerr=dl_gal_auto_err, fmt='o', 
                   label='Observed', alpha=0.5, markersize=3)
    else:
        ax.plot(lb, dl_gal_auto, 'o', label='Observed', alpha=0.5, markersize=3)
    
    # Plot fitted components
    ax.plot(lb, dl_total, 'k-', linewidth=2, label='Total fit')
    ax.plot(lb, dl_clustering, 'r--', linewidth=2, label='Clustering (2h+1h)')
    ax.plot(lb, dl_2h, 'b:', linewidth=1.5, label='2-halo')
    ax.plot(lb, dl_1h, 'g:', linewidth=1.5, label='1-halo')
    ax.plot(lb, dl_shot, 'm:', linewidth=1.5, label='Shot noise')
    
    # Mark fit range
    ax.axvline(fit_range[0], color='gray', linestyle='--', alpha=0.3)
    ax.axvline(fit_range[1], color='gray', linestyle='--', alpha=0.3)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\ell$', fontsize=14)
    ax.set_ylabel(r'$D_\ell$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-4, 1e3)
    
    # Add fit parameters to title or text box
    if title is None:
        title = 'Galaxy Auto Spectrum Fit'
    
    textstr = (f"$A_{{2h}}$ = {params[0]:.2e} ± {params_err[0]:.2e}\\n"
              f"$A_{{1h}}$ = {params[1]:.2e} ± {params_err[1]:.2e}\\n"
              f"$\\mu$ = {params[2]:.2f} ± {params_err[2]:.2f} ($\\ell_p$ ~ {np.exp(params[2]):.0f})\\n"
              f"$\\sigma$ = {params[3]:.2f} ± {params_err[3]:.2f}\\n"
              f"$A_{{shot}}$ = {params[4]:.2e} ± {params_err[4]:.2e}\\n"
              f"$\\chi^2$/dof = {fit_result['chisq']:.1f}/{fit_result['ndof']} = {fit_result['reduced_chisq']:.2f}")
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    return fig, ax


def predict_ciber_auto_from_cross(lb, cl_ciber_gal_cross, cl_gal_auto, 
                                   ell_range_for_scaling=None,
                                   cl_ciber_gal_cross_err=None, cl_gal_auto_err=None,
                                   cl_gal_shot=None, estimate_gal_shot=True,
                                   ell_shot_min=50000, use_fitted_clustering=False,
                                   fit_galaxy_auto=False, fit_range=(300, 80000),
                                   z_value=None, ln_ell_peak_relation=None):
    """
    Predict CIBER intensity auto-spectrum from cross-spectrum and galaxy auto-spectrum.
    
    Uses the relation:
    C_ℓ^{CIBER×CIBER} = (C_ℓ^{CIBER×gal} / C_ℓ^{gal×gal,clust})² × C_ℓ^{gal×gal,clust}
    
    where C_ℓ^{gal×gal,clust} = C_ℓ^{gal×gal} - shot_noise is the clustering component.
    Shot noise doesn't participate in the bias scaling relation.
    
    Parameters
    ----------
    lb : array_like
        Multipole bin centers
    cl_ciber_gal_cross : array_like
        CIBER × galaxy cross-spectrum C_ℓ
    cl_gal_auto : array_like
        Galaxy auto-spectrum C_ℓ (total including shot noise)
    ell_range_for_scaling : tuple, optional
        (ℓ_min, ℓ_max) range for computing median scaling (for diagnostics).
        If None, uses all ℓ.
    cl_ciber_gal_cross_err : array_like, optional
        Uncertainties on cross-spectrum for error propagation
    cl_gal_auto_err : array_like, optional
        Uncertainties on galaxy auto-spectrum for error propagation
    cl_gal_shot : float or array_like, optional
        Galaxy shot noise level to subtract from auto before scaling.
        Can be constant or ℓ-dependent. If None and estimate_gal_shot=True,
        will be estimated from high-ℓ tail. Ignored if fit_galaxy_auto=True.
    estimate_gal_shot : bool, optional
        If True and cl_gal_shot is None, estimate shot noise from high-ℓ
        galaxy auto (ℓ > ell_shot_min). Default True. Ignored if fit_galaxy_auto=True.
    ell_shot_min : float, optional
        Minimum multipole for shot noise estimation. Default 50000.
    fit_galaxy_auto : bool, optional
        If True, fit galaxy auto with smooth 2h+1h+shot model and use the
        2h+1h clustering component (without shot) for scaling. Default False.
    fit_range : tuple, optional
        (ℓ_min, ℓ_max) range for galaxy auto fitting. Default (300, 80000).
    z_value : float, optional
        Redshift value for fixing ln_ell_peak in galaxy auto fit
    ln_ell_peak_relation : dict, optional
        Linear relation for ln_ell_peak: {'slope': 7.4, 'intercept': 8.44}
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'cl_ciber_auto_pred': Predicted CIBER auto-spectrum
        - 'cl_ciber_auto_pred_err': Uncertainty (if errors provided)
        - 'scaling_factor': Median (cross/auto_clust) ratio in ell_range_for_scaling
        - 'scaling_factor_err': Uncertainty on median scaling factor
        - 'cl_gal_auto_clust': Galaxy auto clustering component used
    """
    lb = np.asarray(lb)
    cl_ciber_gal_cross = np.asarray(cl_ciber_gal_cross)
    cl_gal_auto = np.asarray(cl_gal_auto)

    pf = lb*(lb+1)/(2*np.pi)
    
    # Option 1: Fit galaxy auto and use smooth 2h+1h clustering component
    if fit_galaxy_auto:
        print("Fitting galaxy auto-spectrum with 2h+1h+shot model...")
        fit_result = fit_galaxy_auto_clustering(
            lb, cl_gal_auto, cl_gal_auto_err,
            fit_range=fit_range, verbose=True,
            z_value=z_value, ln_ell_peak_relation=ln_ell_peak_relation
        )
        if fit_result is None:
            print("Warning: Galaxy auto fit failed, falling back to simple subtraction")
            fit_galaxy_auto = False
        else:
            cl_gal_auto_clust = fit_result['cl_clustering']
            print(f"Using fitted 2h+1h clustering component (shot removed)")
            
            # Plot the fit
            plot_galaxy_auto_fit(
                lb, pf * cl_gal_auto, fit_result,
                dl_gal_auto_err=pf * cl_gal_auto_err if cl_gal_auto_err is not None else None,
                fit_range=fit_range,
                title='Galaxy Auto Spectrum Fit'
            )
            plt.show()
    # Option 2: Simple shot noise subtraction
    if not fit_galaxy_auto:
        # Estimate or use provided shot noise
        if cl_gal_shot is None and estimate_gal_shot:
            # Estimate from high-ℓ tail where shot noise dominates
            shot_mask = lb > ell_shot_min
            if np.sum(shot_mask) > 0:
                cl_gal_shot = np.mean(cl_gal_auto[shot_mask])
                print(f"Estimated galaxy shot noise from ℓ > {ell_shot_min}: {cl_gal_shot:.3e}")
            else:
                print(f"Warning: No multipoles > {ell_shot_min} for shot noise estimation")
                cl_gal_shot = None
        
        # Subtract shot noise from galaxy auto to get clustering component
        if cl_gal_shot is not None:
            cl_gal_shot = np.asarray(cl_gal_shot)
            cl_gal_auto_clust = cl_gal_auto - cl_gal_shot
            if not estimate_gal_shot:  # Only print if manually provided
                print(f"Subtracting galaxy shot noise: {np.mean(cl_gal_shot):.3e} (mean)")
        else:
            cl_gal_auto_clust = cl_gal_auto
            print("No galaxy shot noise subtraction")
    
    # Apply ell range mask for computing scaling factor
    if ell_range_for_scaling is not None:
        mask = (lb >= ell_range_for_scaling[0]) & (lb <= ell_range_for_scaling[1])
        cross_scaling = cl_ciber_gal_cross[mask]
        auto_scaling = cl_gal_auto_clust[mask]
        if cl_ciber_gal_cross_err is not None:
            cross_err_scaling = np.asarray(cl_ciber_gal_cross_err)[mask]
        else:
            cross_err_scaling = None
        if cl_gal_auto_err is not None:
            auto_err_scaling = np.asarray(cl_gal_auto_err)[mask]
        else:
            auto_err_scaling = None
    else:
        cross_scaling = cl_ciber_gal_cross
        auto_scaling = cl_gal_auto_clust
        cross_err_scaling = cl_ciber_gal_cross_err
        auto_err_scaling = cl_gal_auto_err
    
    # Compute scaling factor: weighted mean of (cross/auto) in the specified range
    # This represents b_CIBER/b_gal and should be ~O(1)
    scaling_per_ell = cross_scaling / auto_scaling
    
    # Uncertainty on scaling factor
    if cross_err_scaling is not None and auto_err_scaling is not None:
        # Error propagation for each ℓ: δ(cross/auto)
        rel_err_squared = (cross_err_scaling / cross_scaling)**2 + \
                         (auto_err_scaling / auto_scaling)**2
        scaling_err_per_ell = scaling_per_ell * np.sqrt(rel_err_squared)
        
        # Weighted mean using inverse variance weights
        weights = 1.0 / scaling_err_per_ell**2
        scaling_factor = np.average(scaling_per_ell, weights=weights)
        scaling_factor_err = 1.0 / np.sqrt(np.sum(weights))
    else:
        # Simple unweighted mean if no errors available
        scaling_factor = np.mean(scaling_per_ell)
        scaling_factor_err = np.std(scaling_per_ell) / np.sqrt(len(scaling_per_ell))
    
    # Apply scaling factor squared to galaxy auto clustering component at all ℓ
    # C_ℓ^{CIBER×CIBER} = (b_CIBER/b_gal)² × C_ℓ^{gal×gal,clust}
    cl_ciber_auto_pred = scaling_factor**2 * cl_gal_auto_clust
    
    # Propagate uncertainties
    if cl_gal_auto_err is not None:
        cl_gal_auto_err = np.asarray(cl_gal_auto_err)
        # Error: sqrt((2 × scaling × auto × d_scaling)² + (scaling² × d_auto)²)
        cl_ciber_auto_pred_err = np.sqrt(
            (2 * scaling_factor * cl_gal_auto * scaling_factor_err)**2 + 
            (scaling_factor**2 * cl_gal_auto_err)**2
        )
    else:
        cl_ciber_auto_pred_err = None
    
    return {
        'cl_ciber_auto_pred': cl_ciber_auto_pred,
        'cl_ciber_auto_pred_err': cl_ciber_auto_pred_err,
        'scaling_factor': scaling_factor,
        'scaling_factor_err': scaling_factor_err,
        'cl_gal_auto_clust': cl_gal_auto_clust
    }


def predict_ciber_auto_vs_redshift(res_ps, inst_idx=0, zbinedges=None,
                                   ell_range_for_scaling=(300, 3000),
                                   startidx=2, endidx=-1, gal_shot_per_zbin=None,
                                   estimate_gal_shot=True, ell_shot_min=50000,
                                   fit_galaxy_auto=False, fit_range=(300, 80000)):
    """
    Predict CIBER auto-spectrum from cross and galaxy auto for multiple redshift bins.
    
    Parameters
    ----------
    res_ps : dict
        Output from collect_ciber_gal_vs_redshift containing:
        - 'lb': multipole bins
        - 'full_cl_cross': CIBER × galaxy cross-spectra [n_inst, n_zbin, n_ell]
        - 'full_clerr_cross': cross-spectrum errors
        - 'full_cl_gal': galaxy auto-spectra [n_inst, n_zbin, n_ell]
        - 'full_clerr_gal': galaxy auto errors
    inst_idx : int, optional
        Instrument index (0 for TM1/1.1μm, 1 for TM2/1.8μm)
    zbinedges : array_like, optional
        Redshift bin edges. If None, uses equal spacing based on n_zbin
    ell_range_for_scaling : tuple, optional
        (ℓ_min, ℓ_max) range for computing scaling factor
    startidx : int, optional
        Starting index for multipole range
    endidx : int, optional  
        Ending index for multipole range
    gal_shot_per_zbin : array_like, optional
        Shot noise levels for galaxy auto per z bin [n_zbin] or [n_zbin, n_ell].
        If None and estimate_gal_shot=True, will be estimated from high-ℓ.
        Ignored if fit_galaxy_auto=True.
    estimate_gal_shot : bool, optional
        If True and gal_shot_per_zbin is None, estimate shot noise from high-ℓ
        galaxy auto for each z bin. Default True. Ignored if fit_galaxy_auto=True.
    ell_shot_min : float, optional
        Minimum multipole for shot noise estimation. Default 50000.
    fit_galaxy_auto : bool, optional
        If True, fit galaxy auto with smooth 2h+1h+shot model for each z bin
        and use the 2h+1h clustering component (without shot) for scaling. Default False.
    fit_range : tuple, optional
        (ℓ_min, ℓ_max) range for galaxy auto fitting. Default (300, 80000).
    ell_range_for_scaling : tuple, optional
        (ℓ_min, ℓ_max) range for computing scaling factor
    startidx : int, optional
        Starting index for multipole range
    endidx : int, optional  
        Ending index for multipole range
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'lb': multipole bins
        - 'dl_ciber_auto_pred': predicted auto D_ℓ per redshift bin [n_zbin, n_ell]
        - 'dl_ciber_auto_pred_err': uncertainties
        - 'dl_ciber_auto_pred_total': sum over redshift bins [n_ell]
        - 'dl_ciber_auto_pred_total_err': total uncertainty
        - 'z_centers': redshift bin centers
        - 'scaling_factors': scaling factor per redshift bin
    """
    # Extract data
    lb = res_ps['lb'][startidx:endidx]
    
    # Note: collect_ciber_gal_vs_redshift returns arrays with shape [2, n_zbin, n_ell]
    # where the first dimension is for the 2 instruments
    full_cl_cross = res_ps['full_cl_cross'][inst_idx, :, startidx:endidx]  # [n_zbin, n_ell]
    full_clerr_cross = res_ps['full_clerr_cross'][inst_idx, :, startidx:endidx]
    full_cl_auto = res_ps['full_cl_gal'][inst_idx, :, startidx:endidx]  # Galaxy auto [n_zbin, n_ell]
    full_clerr_auto = res_ps['full_clerr_gal'][inst_idx, :, startidx:endidx]
    
    n_zbin = full_cl_cross.shape[0]
    n_ell = len(lb)
    
    # Redshift centers
    if zbinedges is None:
        zbinedges = np.linspace(0, 1, n_zbin + 1)
    z_centers = 0.5 * (zbinedges[:-1] + zbinedges[1:])
    
    # Convert to D_ℓ
    pf = lb * (lb + 1) / (2 * np.pi)
    
    # Storage for results
    dl_ciber_auto_pred = np.zeros((n_zbin, n_ell))
    dl_ciber_auto_pred_err = np.zeros((n_zbin, n_ell))
    scaling_factors = np.zeros(n_zbin)
    scaling_factors_err = np.zeros(n_zbin)
    
    # Process each redshift bin
    for zidx in range(n_zbin):
        # Get cross and auto for this redshift
        cl_cross = full_cl_cross[zidx]
        cl_cross_err = full_clerr_cross[zidx]
        cl_auto = full_cl_auto[zidx]
        if full_clerr_auto is not None:
            cl_auto_err = full_clerr_auto[zidx]
        else:
            cl_auto_err = None
        
        # Get shot noise for this z bin if provided
        if gal_shot_per_zbin is not None:
            gal_shot_per_zbin = np.asarray(gal_shot_per_zbin)
            if gal_shot_per_zbin.ndim == 1:
                # Single value per z bin
                cl_gal_shot = gal_shot_per_zbin[zidx]
            else:
                # Array per z bin
                cl_gal_shot = gal_shot_per_zbin[zidx]
        else:
            cl_gal_shot = None
        
        # Print diagnostics
        print(f"\nz = {z_centers[zidx]:.2f}:")
        print(f"  Cross spectrum range: {np.min(cl_cross):.3e} to {np.max(cl_cross):.3e}")
        print(f"  Galaxy auto range: {np.min(cl_auto):.3e} to {np.max(cl_auto):.3e}")
        print(f"  Cross²/Auto range: {np.min(cl_cross**2/cl_auto):.3e} to {np.max(cl_cross**2/cl_auto):.3e}")
        
        # Predict auto
        result = predict_ciber_auto_from_cross(
            lb, cl_cross, cl_auto,
            ell_range_for_scaling=ell_range_for_scaling,
            cl_ciber_gal_cross_err=cl_cross_err,
            cl_gal_auto_err=cl_auto_err,
            cl_gal_shot=cl_gal_shot,
            estimate_gal_shot=estimate_gal_shot,
            ell_shot_min=ell_shot_min,
            fit_galaxy_auto=fit_galaxy_auto,
            fit_range=fit_range,
            z_value=z_centers[zidx]  # Pass redshift for fixing ln_ell_peak
        )
        
        # Convert to D_ℓ and store
        dl_ciber_auto_pred[zidx] = pf * result['cl_ciber_auto_pred']
        if result['cl_ciber_auto_pred_err'] is not None:
            dl_ciber_auto_pred_err[zidx] = pf * result['cl_ciber_auto_pred_err']
        
        scaling_factors[zidx] = result['scaling_factor']
        scaling_factors_err[zidx] = result['scaling_factor_err']
        
        print(f"  Scaling factor: {result['scaling_factor']:.3e} ± {result['scaling_factor_err']:.3e}")
        print(f"  Predicted D_ℓ range: {np.min(dl_ciber_auto_pred[zidx]):.3e} to {np.max(dl_ciber_auto_pred[zidx]):.3e}")
    
    # Sum over redshift bins for total
    dl_ciber_auto_pred_total = np.sum(dl_ciber_auto_pred, axis=0)
    # Error propagation: add in quadrature assuming independent bins
    dl_ciber_auto_pred_total_err = np.sqrt(np.sum(dl_ciber_auto_pred_err**2, axis=0))
    
    return {
        'lb': lb,
        'dl_ciber_auto_pred': dl_ciber_auto_pred,
        'dl_ciber_auto_pred_err': dl_ciber_auto_pred_err,
        'dl_ciber_auto_pred_total': dl_ciber_auto_pred_total,
        'dl_ciber_auto_pred_total_err': dl_ciber_auto_pred_total_err,
        'z_centers': z_centers,
        'zbinedges': zbinedges,
        'scaling_factors': scaling_factors,
        'scaling_factors_err': scaling_factors_err,
        'inst_idx': inst_idx,
        'ell_range_used': ell_range_for_scaling
    }


def plot_ciber_auto_prediction(pred_results, figsize=(10, 6), xlim=[200, 1e5],
                               ylim=[1e-1, 1e3], title=None, save_path=None,
                               colors=None, alpha=0.6, show_total_err=True):
    """
    Plot predicted CIBER auto-spectrum vs redshift.
    
    Parameters
    ----------
    pred_results : dict
        Output from predict_ciber_auto_vs_redshift
    figsize : tuple, optional
        Figure size
    xlim : list, optional
        x-axis limits
    ylim : list, optional
        y-axis limits
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure
    colors : list, optional
        Colors for each redshift bin
    alpha : float, optional
        Alpha for individual redshift lines
    show_total_err : bool, optional
        Whether to show uncertainty band on total
    
    Returns
    -------
    fig, ax
    """
    lb = pred_results['lb']
    dl_pred = pred_results['dl_ciber_auto_pred']
    dl_pred_err = pred_results['dl_ciber_auto_pred_err']
    dl_total = pred_results['dl_ciber_auto_pred_total']
    dl_total_err = pred_results['dl_ciber_auto_pred_total_err']
    z_centers = pred_results['z_centers']
    zbinedges = pred_results['zbinedges']

    lb_auto, cl_auto, clerr_auto = [pred_results[key] for key in ['lb_auto', 'full_cl_ciber_auto', 'full_clerr_ciber_auto']]
    
    n_zbin = len(z_centers)
    
    # Default colors
    if colors is None:
        cmap = plt.cm.jet
        colors = cmap(np.linspace(0.3, 1.0, n_zbin))
        # colors = [cmap(i / n_zbin) for i in range(n_zbin)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot individual redshift contributions
    for zidx in range(n_zbin):
        label = f'{zbinedges[zidx]:.1f} < z < {zbinedges[zidx+1]:.1f}'
        ax.plot(lb, dl_pred[zidx], color=colors[zidx], 
                linewidth=1.5, alpha=alpha, label=label)
        
        # Optional: show uncertainty bands
        if show_total_err and np.any(dl_pred_err[zidx] > 0):
            ax.fill_between(lb, 
                           dl_pred[zidx] - dl_pred_err[zidx],
                           dl_pred[zidx] + dl_pred_err[zidx],
                           color=colors[zidx], alpha=0.15)
    
    # Plot total (sum over redshift)
    ax.plot(lb, dl_total, 'k-', linewidth=3, label='Total (all z)', zorder=10)
    
    if show_total_err and np.any(dl_total_err > 0):
        ax.fill_between(lb, 
                       dl_total - dl_total_err,
                       dl_total + dl_total_err,
                       color='black', alpha=0.2, zorder=9)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\ell$', fontsize=14)
    ax.set_ylabel(r'$D_\ell$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=14)
    
    if title is None:
        inst_labels = {0: '1.1 μm', 1: '1.8 μm'}
        inst_idx = pred_results.get('inst_idx', 0)
        title = f'Predicted CIBER {inst_labels.get(inst_idx, "")} Auto-Spectrum'

    pf_auto = lb_auto*(lb_auto+1)/(2*np.pi)
    ax.errorbar(lb_auto, (pf_auto*cl_auto), yerr=(pf_auto*clerr_auto), fmt='o', color='k', 
                capsize=3, markersize=3)


    ax.set_title(title, fontsize=15)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='upper left', ncol=2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved plot to: {save_path}")
    
    return fig, ax


def plot_cross_and_gal_auto_vs_redshift(res_ps, inst_idx=0, zbinedges=None,
                                        startidx=2, endidx=-1, 
                                        figsize=(14, 12), 
                                        ylim=[1e-4, 1e3]):
    """
    Plot galaxy auto and CIBER×galaxy cross spectra for each redshift bin.
    
    Parameters
    ----------
    res_ps : dict
        Output from collect_ciber_gal_vs_redshift
    inst_idx : int
        Instrument index (0 for TM1, 1 for TM2)
    zbinedges : array_like, optional
        Redshift bin edges
    startidx : int
        Starting multipole index
    endidx : int
        Ending multipole index  
    figsize : tuple
        Figure size
    
    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    # Extract data
    lb = res_ps['lb'][startidx:endidx]
    full_cl_cross = res_ps['full_cl_cross'][inst_idx, :, startidx:endidx]
    full_cl_auto = res_ps['full_cl_gal'][inst_idx, :, startidx:endidx]
    
    n_zbin = full_cl_cross.shape[0]
    
    if zbinedges is None:
        zbinedges = np.linspace(0, 1, n_zbin + 1)
    z_centers = 0.5 * (zbinedges[:-1] + zbinedges[1:])
    
    # Convert to D_ℓ
    pf = lb * (lb + 1) / (2 * np.pi)
    
    # Create subplots
    nrows = (n_zbin + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_zbin))
    
    for zidx in range(n_zbin):
        ax = axes[zidx]
        
        dl_cross = pf * full_cl_cross[zidx]
        dl_auto = pf * full_cl_auto[zidx]
        
        # Plot on twin axes with different scales
        ax.loglog(lb, np.abs(dl_cross), 'o-', color=colors[zidx], 
                 label=f'CIBER×gal cross', alpha=0.7, markersize=4)
        ax.set_xlabel(r'$\ell$', fontsize=12)
        ax.set_ylabel(r'$D_{\ell}^{\rm CIBER \times gal}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', 
                     fontsize=11, color=colors[zidx])
        ax.tick_params(axis='y', labelcolor=colors[zidx])
        
        # Second y-axis for galaxy auto
        ax2 = ax.twinx()
        ax2.loglog(lb, dl_auto, 's--', color='gray', 
                  label=f'Galaxy auto', alpha=0.5, markersize=4)
        ax2.set_ylabel(r'$D_{\ell}^{\rm gal \times gal}$ [sr$^{-2}$]', 
                      fontsize=11, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        
        # Title with redshift
        ax.set_title(f'z = [{zbinedges[zidx]:.1f}, {zbinedges[zidx+1]:.1f}]', 
                    fontsize=12)
        
        # Add ratio on the plot
        ratio = np.median(dl_cross**2 / dl_auto)
        ax.text(0.05, 0.95, f'Median $(cross)^2/auto$ = {ratio:.2e}',
               transform=ax.transAxes, fontsize=9, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        ax2.legend(loc='lower left', fontsize=9)
        ax.set_ylim(ylim)
        ax2.set_ylim(ylim)

    
    # Hide unused subplots
    for idx in range(n_zbin, len(axes)):
        axes[idx].axis('off')
    
    # plt.tight_layout()
    
    return fig, axes
