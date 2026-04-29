import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, Callable
import os

def load_ihl_templates(template_dir, template_names=None, zbinedges=None, slopes=None):
    """
    Load IHL (Intra-Halo Light) templates from files in the specified directory.
    
    Parameters
    ----------
    template_dir : str
        Path to directory containing template files
    template_names : list of str, optional
        List of template filenames to load. If None and zbinedges is None, 
        loads all .txt and .dat files in the directory.
    zbinedges : array_like, optional
        Redshift bin edges for automatic filename generation. If provided,
        generates filenames like "ihl_ps_z_{zlow}_{zhigh}_slope_{slope}.txt"
    slopes : list of float, optional
        List of slope values to use with zbinedges. If None, defaults to [1.0]
    
    Returns
    -------
    templates : dict
        Dictionary with template names as keys and each value is a dict with:
        - 'ell': array of multipole values
        - 'dl': array of D_ell values
        - 'filename': original filename
        - 'zbinedges': redshift bin edges (if applicable)
        - 'slope': slope value (if applicable)
    
    Notes
    -----
    Files are expected to have:
    - First column: ell values
    - Second column: D_ell values  
    - One header row (which is skipped)
    
    Filename format (when using zbinedges):
    "ihl_ps_z_{zlow}_{zhigh}_slope_{slope}.txt"
    """
    import os
    import glob
    
    if not os.path.exists(template_dir):
        raise ValueError(f"Template directory {template_dir} does not exist")
    
    templates = {}
    
    if zbinedges is not None:
        # Auto-generate filenames based on redshift bin edges
        zbinedges = np.array(zbinedges)
        if slopes is None:
            slopes = [1.0]
        
        template_files = []
        for i in range(len(zbinedges) - 1):
            zlow = zbinedges[i]
            zhigh = zbinedges[i + 1]
            for slope in slopes:
                filename = f"ihl_ps_z_{zlow}_{zhigh}_slope_{slope}.txt"
                template_files.append((filename, zlow, zhigh, slope, i))
                
    elif template_names is None:
        # Find all .txt and .dat files in directory
        pattern1 = os.path.join(template_dir, "*.txt")
        pattern2 = os.path.join(template_dir, "*.dat")
        found_files = glob.glob(pattern1) + glob.glob(pattern2)
        template_files = [(os.path.basename(f), None, None, None, None) for f in found_files]
    else:
        # Use provided template names
        template_files = [(name, None, None, None, None) for name in template_names]
    
    for file_info in template_files:
        template_name, zlow, zhigh, slope, zidx = file_info
        filepath = os.path.join(template_dir, template_name)
        
        if not os.path.exists(filepath):
            print(f"Warning: Template file {filepath} not found, skipping")
            continue
            
        try:
            # Load data, skipping header row
            data = np.loadtxt(filepath, skiprows=1)
            
            if data.shape[1] < 2:
                print(f"Warning: Template file {filepath} doesn't have at least 2 columns, skipping")
                continue
                
            ell = data[:, 0]
            dl = data[:, 1]
            
            # Create template key
            if zbinedges is not None:
                # Use descriptive key for redshift bin templates
                template_key = f"z{zlow}_{zhigh}_slope{slope}"
            else:
                # Use filename without extension as template key
                template_key = os.path.splitext(template_name)[0]
            
            templates[template_key] = {
                'ell': ell,
                'dl': dl, 
                'filename': template_name,
                'zbinedges': (zlow, zhigh) if zbinedges is not None else None,
                'slope': slope,
                'zbin_index': zidx
            }
            
            if zbinedges is not None:
                print(f"Loaded template '{template_key}': {len(ell)} data points, "
                      f"ell range [{ell.min():.0f}, {ell.max():.0f}], z=[{zlow}, {zhigh}], slope={slope}")
            else:
                print(f"Loaded template '{template_key}': {len(ell)} data points, "
                      f"ell range [{ell.min():.0f}, {ell.max():.0f}]")
                  
        except Exception as e:
            print(f"Error loading template {filepath}: {e}")
            continue
    
    print(f"\nSuccessfully loaded {len(templates)} IHL templates")
    return templates


def load_ihl_template_for_zbin(template_dir, zbinedges, zidx, slopes=None):
    """
    Load IHL templates for a specific redshift bin.
    
    Parameters
    ----------
    template_dir : str
        Path to directory containing template files
    zbinedges : array_like
        Redshift bin edges
    zidx : int
        Redshift bin index (0 = first bin, etc.)
    slopes : list of float, optional
        List of slope values to load. If None, defaults to [1.0]
    
    Returns
    -------
    templates : dict
        Dictionary of templates for this redshift bin
    zlow, zhigh : float
        Lower and upper redshift bin edges
    """
    zbinedges = np.array(zbinedges)
    
    if zidx < 0 or zidx >= len(zbinedges) - 1:
        raise ValueError(f"zidx {zidx} out of range for zbinedges with {len(zbinedges)} edges")
    
    zlow = zbinedges[zidx]
    zhigh = zbinedges[zidx + 1]
    
    if slopes is None:
        slopes = [1.0]
    
    # Load only the templates for this redshift bin
    templates = {}
    for slope in slopes:
        filename = f"ihl_ps_z_{zlow}_{zhigh}_slope_{slope}.txt"
        filepath = os.path.join(template_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: Template file {filepath} not found, skipping slope {slope}")
            continue
        
        try:
            data = np.loadtxt(filepath, skiprows=1)
            if data.shape[1] < 2:
                print(f"Warning: Template file {filepath} doesn't have at least 2 columns, skipping")
                continue
                
            ell = data[:, 0]
            dl = data[:, 1]
            
            template_key = f"slope_{slope}"
            templates[template_key] = {
                'ell': ell,
                'dl': dl,
                'filename': filename,
                'zbinedges': (zlow, zhigh),
                'slope': slope,
                'zbin_index': zidx
            }
            
            print(f"Loaded template '{template_key}': {len(ell)} data points, "
                  f"ell range [{ell.min():.0f}, {ell.max():.0f}], z=[{zlow}, {zhigh}]")
                  
        except Exception as e:
            print(f"Error loading template {filepath}: {e}")
            continue
    
    print(f"Loaded {len(templates)} templates for z-bin [{zlow}, {zhigh}]")
    return templates, zlow, zhigh


def interpolate_1h_params(z_value, slope=None, one_halo_params_dict=None, sigma_fixed=None):
    """
    Calculate 1-halo log-normal parameters for a given redshift.
    
    Can use either:
    1. IHL-derived parameters from one_halo_params_dict (if provided)
    2. Default analytic formulae:
       ln(ell_peak) = 7.4 * z + 8.44
       sigma = 2.43 * z + 1.56
    
    Parameters
    ----------
    z_value : float
        Redshift value
    slope : float, optional
        Slope value for selecting parameters from one_halo_params_dict.
        If None and dict is provided, uses first available slope.
    one_halo_params_dict : dict, optional
        Dictionary with IHL-derived parameters from load_ihl_1h_params().
        Should have keys 'ln_ell_peak_vs_z' and 'sigma_vs_z' with linear fit parameters.
        If None, uses default analytic formulae.
    sigma_fixed : float, optional
        If provided, overrides sigma calculation and uses this fixed value.
    
    Returns
    -------
    ln_ell_peak : float
        ln(ell_peak) 
    sigma : float
        sigma (log-width parameter)
    
    Examples
    --------
    # Use default analytic formulae
    ln_ell_peak, sigma = interpolate_1h_params(z_value=0.3)
    
    # Use IHL-derived parameters
    ihl_params = load_ihl_1h_params('ihl_1h_params.npz')
    ln_ell_peak, sigma = interpolate_1h_params(z_value=0.3, slope=1.0, 
                                                one_halo_params_dict=ihl_params)
    """
    # If IHL-derived parameters are provided, use them
    if one_halo_params_dict is not None:
        # Get slope to use
        if slope is None:
            if 'slopes' in one_halo_params_dict:
                slope = one_halo_params_dict['slopes'][0]
            else:
                slope = 1.0
        
        # Try to get linear relations
        if 'ln_ell_peak_vs_z' in one_halo_params_dict and 'sigma_vs_z' in one_halo_params_dict:
            ln_rel = one_halo_params_dict['ln_ell_peak_vs_z'].get(slope)
            sigma_rel = one_halo_params_dict['sigma_vs_z'].get(slope)
            
            if ln_rel is not None:
                ln_ell_peak = ln_rel['intercept'] + ln_rel['slope'] * z_value
            else:
                # Fallback to default formula
                ln_ell_peak = 7.4 * z_value + 8.44
            
            if sigma_rel is not None and sigma_fixed is None:
                sigma = sigma_rel['intercept'] + sigma_rel['slope'] * z_value
            elif sigma_fixed is not None:
                sigma = sigma_fixed
            else:
                # Fallback to default formula
                sigma = 2.43 * z_value + 1.56
        else:
            # Old-style params dict, use default formulae
            ln_ell_peak = 7.4 * z_value + 8.44
            if sigma_fixed is not None:
                sigma = sigma_fixed
            else:
                sigma = 2.43 * z_value + 1.56
    else:
        # Use default analytic formulae (original behavior)
        ln_ell_peak = 7.4 * z_value + 8.44
        
        if sigma_fixed is not None:
            sigma = sigma_fixed
        else:
            sigma = 2.43 * z_value + 1.56
    
    return ln_ell_peak, sigma


def fit_and_decompose_ihl_templates(template_dir, zbinedges=None, slopes=None,
                                    template_names=None, 
                                    use_powerlaw_2h=True, alpha_2h_fixed=0.0,
                                    fit_ell_range=None, plot=True, 
                                    figsize=(14, 10), save_path=None,
                                    p0=None, bounds=None, method='leastsq',
                                    verbose=True, ylim=[1e-4, 1e3]):
    """
    Load IHL templates and fit them to decompose into two-halo, one-halo, and shot noise contributions.
    
    This function provides a complete workflow to:
    1. Load IHL template power spectra from files
    2. Fit each template with a parametric model: D_ℓ = D_2h + D_1h + D_shot
       - Two-halo: power law (ℓ/1000)^α or fixed theory prediction
       - One-halo: log-normal exp(-(ln(ℓ) - μ)²/(2σ²)) to capture non-linear clustering
       - Shot noise: ℓ(ℓ+1)/(2π) to model Poisson fluctuations
    3. Return best-fit parameters and components for each template
    4. Optionally visualize the decomposition
    
    Parameters
    ----------
    template_dir : str
        Path to directory containing IHL template files
    zbinedges : array_like, optional
        Redshift bin edges. If provided, loads templates with naming convention:
        "ihl_ps_z_{zlow}_{zhigh}_slope_{slope}.txt"
    slopes : list of float, optional
        Slope values to use with zbinedges. Default [1.0]
    template_names : list of str, optional
        Explicit list of template filenames to load. Used if zbinedges is None.
    use_powerlaw_2h : bool, optional
        If True, model 2-halo as power law. If False, requires theory prediction. Default True.
    alpha_2h_fixed : float, optional
        Fixed power-law index for 2-halo term. Default -1.5 (linear clustering).
    fit_ell_range : tuple, optional
        (ℓ_min, ℓ_max) range to fit over. If None, uses full template range.
    plot : bool, optional
        Whether to create diagnostic plots. Default True.
    figsize : tuple, optional
        Figure size for plots. Default (14, 10).
    save_path : str, optional
        Path to save figure. If None, displays interactively.
    p0 : array_like, optional
        Initial parameter guess [A_2h, A_1h, mu_1h, sigma_1h, A_shot].
        If None, uses smart defaults based on template data.
    bounds : tuple of array_like, optional
        Parameter bounds as ([lower bounds], [upper bounds]).
        If None, uses physically reasonable defaults.
    method : str, optional
        Fitting method: 'leastsq' (curve_fit) or 'minimize' (scipy minimize). Default 'leastsq'.
    verbose : bool, optional
        Print detailed information. Default True.
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'templates': dict of loaded templates (keys: template names)
        - 'fits': dict of fit results for each template with:
            * 'params': [A_2h, A_1h, mu_1h, sigma_1h, A_shot]
            * 'params_err': parameter uncertainties
            * 'chisq': chi-squared value
            * 'reduced_chisq': reduced chi-squared
            * 'components': dict with 'two_halo', 'one_halo', 'shot_noise', 'total'
            * 'ell_eval': multipoles where components are evaluated
        - 'summary': DataFrame with fit parameters for all templates
    
    """
    import pandas as pd
    
    # Load templates
    if verbose:
        print("="*70)
        print("Loading IHL Templates")
        print("="*70)
    
    templates = load_ihl_templates(template_dir, template_names=template_names,
                                   zbinedges=zbinedges, slopes=slopes)
    
    if len(templates) == 0:
        raise ValueError("No templates loaded. Check template_dir and file naming.")
    
    # Initialize results storage
    fits = {}
    
    if verbose:
        print("\n" + "="*70)
        print("Fitting Templates")
        print("="*70)
    
    # Fit each template
    for template_name, template in templates.items():
        if verbose:
            print(f"\n--- Fitting template: {template_name} ---")
        
        ell_template = template['ell']
        dl_template = template['dl']
        
        # Create model instance
        model = CrossPowerSpectrumModel(
            lb=ell_template,
            use_powerlaw_2h=use_powerlaw_2h,
            alpha_2h_fixed=alpha_2h_fixed
        )
        
        # Set initial parameters if not provided
        if p0 is None:
            # Smart initialization based on template shape
            A_2h_init = np.median(dl_template[:5]) if len(dl_template) > 5 else dl_template[0]
            A_1h_init = np.max(dl_template) * 0.7  # One-halo typically dominates peak
            mu_1h_init = np.log(ell_template[np.argmax(dl_template)])  # Peak location
            sigma_1h_init = 0.5  # Typical log-width
            # Estimate shot noise from high-ell behavior
            if len(dl_template) > 3:
                # Fit ℓ² to last few points to get shot noise level
                ell_high = ell_template[-1]
                dl_high = dl_template[-1]
                pf_high = ell_high * (ell_high + 1) / (2 * np.pi)
                A_shot_init = np.mean(dl_high / pf_high)
            else:
                A_shot_init = dl_template[-1] / (ell_template[-1] * (ell_template[-1] + 1) / (2 * np.pi))
            # p0_fit = [A_2h_init, A_1h_init, mu_1h_init, sigma_1h_init, A_shot_init]
            p0_fit = [0.01, 1.0, 10, 3.0, A_shot_init]
            print('p0 fit:', p0_fit)


        else:
            p0_fit = p0
        
        # Set bounds if not provided
        if bounds is None:
            # Physically reasonable bounds
            bounds_fit = (
                [0., 0., np.log(1000), 1.0, 0.],  # Lower bounds
                [np.inf, np.inf, np.log(80000), 4.0, np.inf]  # Upper bounds
            )
        else:
            bounds_fit = bounds
        
        # Perform fit
        try:
            fit_result = model.fit_model(
                lb_data=ell_template,
                dl_data=dl_template,
                dl_err=None,  # No errors for template fitting
                p0=p0_fit,
                bounds=bounds_fit,
                method=method,
                fit_range=fit_ell_range,
                verbose=verbose
            )
            
            # Extract best-fit parameters
            params = fit_result['params']
            params_err = fit_result.get('params_err', np.full_like(params, np.nan))
            
            # Evaluate model components on fine ell grid for plotting
            ell_eval = np.logspace(np.log10(ell_template.min()), 
                                   np.log10(ell_template.max()), 200)
            components = model.model_components(ell_eval, *params)
            
            # Store results
            fits[template_name] = {
                'params': params,
                'params_err': params_err,
                'param_names': ['A_2h', 'A_1h', 'mu_1h', 'sigma_1h', 'A_shot'],
                'chisq': fit_result.get('chisq', np.nan),
                'reduced_chisq': fit_result.get('reduced_chisq', np.nan),
                'components': components,
                'ell_eval': ell_eval,
                'ell_template': ell_template,
                'dl_template': dl_template,
                'zbinedges': template['zbinedges'],
                'slope': template['slope']
            }
            
            if verbose:
                print(f"✓ Fit successful:")
                print(f"  A_2h = {params[0]:.3e} ± {params_err[0]:.3e}")
                print(f"  A_1h = {params[1]:.3e} ± {params_err[1]:.3e}")
                print(f"  μ_1h (ln ℓ_peak) = {params[2]:.3f} ± {params_err[2]:.3f}  [ℓ_peak ~ {np.exp(params[2]):.0f}]")
                print(f"  σ_1h = {params[3]:.3f} ± {params_err[3]:.3f}")
                print(f"  A_shot = {params[4]:.3e} ± {params_err[4]:.3e}")
                print(f"  χ²/dof = {fit_result.get('reduced_chisq', np.nan):.2f}")
                
        except Exception as e:
            if verbose:
                print(f"✗ Fit failed: {e}")
            fits[template_name] = {
                'params': np.full(5, np.nan),
                'params_err': np.full(5, np.nan),
                'param_names': ['A_2h', 'A_1h', 'mu_1h', 'sigma_1h', 'A_shot'],
                'error': str(e)
            }
    
    # Create summary DataFrame
    summary_data = []
    for template_name, fit in fits.items():
        if 'error' not in fit:
            params = fit['params']
            params_err = fit['params_err']
            row = {
                'template': template_name,
                'A_2h': params[0],
                'A_2h_err': params_err[0],
                'A_1h': params[1],
                'A_1h_err': params_err[1],
                'mu_1h': params[2],
                'mu_1h_err': params_err[2],
                'ell_peak': np.exp(params[2]),
                'sigma_1h': params[3],
                'sigma_1h_err': params_err[3],
                'A_shot': params[4],
                'A_shot_err': params_err[4],
                'chisq': fit.get('chisq', np.nan),
                'reduced_chisq': fit.get('reduced_chisq', np.nan)
            }
            if fit['zbinedges'] is not None:
                row['z_low'] = fit['zbinedges'][0]
                row['z_high'] = fit['zbinedges'][1]
                row['z_center'] = (fit['zbinedges'][0] + fit['zbinedges'][1]) / 2
            summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create diagnostic plots
    if plot and len(fits) > 0:
        n_templates = len([f for f in fits.values() if 'error' not in f])
        ncols = min(3, n_templates)
        nrows = (n_templates + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_templates == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        plot_idx = 0
        for template_name, fit in fits.items():
            if 'error' in fit:
                continue
            
            ax = axes[plot_idx]
            
            # Plot template data
            ax.loglog(fit['ell_template'], fit['dl_template'], 
                     'o', color='black', markersize=6, alpha=0.7, label='IHL Template')
            
            # Plot model components
            components = fit['components']
            ell_eval = fit['ell_eval']
            ax.loglog(ell_eval, components['two_halo'], '--', 
                     color='blue', linewidth=2, label='Two-halo')
            ax.loglog(ell_eval, components['one_halo'], '--', 
                     color='green', linewidth=2, label='One-halo')
            ax.loglog(ell_eval, components['shot_noise'], '--', 
                     color='red', linewidth=2, label='Shot noise')
            ax.loglog(ell_eval, components['total'], '-', 
                     color='orange', linewidth=2.5, label='Total Model')
            
            # Labels and formatting
            ax.set_xlabel(r'$\ell$', fontsize=12)
            ax.set_ylabel(r'$D_\ell$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=12)
            
            # Title with template info
            title = template_name
            if fit['zbinedges'] is not None:
                z_low, z_high = fit['zbinedges']
                title = f"z=[{z_low:.1f}, {z_high:.1f}]"
                if fit['slope'] is not None:
                    title += f", slope={fit['slope']}"
            ax.set_title(title, fontsize=11, fontweight='bold')
            
            ax.legend(fontsize=9, loc=1, framealpha=0.9)
            ax.grid(alpha=0.3, which='both')
            
            # Add text box with fit parameters
            params = fit['params']
            textstr = f"$A_{{2h}}$ = {params[0]:.2e}\n"
            textstr += f"$A_{{1h}}$ = {params[1]:.2e}\n"
            textstr += f"$\ell_{{peak}}$ = {np.exp(params[2]):.0f}\n"
            textstr += f"$A_{{shot}}$ = {params[4]:.2e}"
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plot_idx += 1

            ax.set_ylim(ylim)


        
        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].axis('off')

        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            if verbose:
                print(f"\n✓ Saved decomposition plots to: {save_path}")
        else:
            plt.show()
    
    # Final summary
    if verbose:
        print("\n" + "="*70)
        print("Fitting Complete")
        print("="*70)
        print(f"Successfully fit {len([f for f in fits.values() if 'error' not in f])}/{len(templates)} templates")
        if len(summary_df) > 0:
            print("\nSummary of fit parameters:")
            print(summary_df.to_string(index=False))
    
    return {
        'templates': templates,
        'fits': fits,
        'summary': summary_df
    }



def get_ihl_components_at_ell(fit_result, ell_values):
    """
    Evaluate smooth PARAMETRIC model at specific multipoles using IHL-derived parameters.
    
    IMPORTANT: This returns the SMOOTH LOG-NORMAL model derived from the fitted 
    parameters (A_2h, A_1h, mu_1h, sigma_1h, A_shot), NOT the raw IHL template values.
    
    Use this when you want:
    - Smooth, continuous model without interpolation kinks
    - Parametric representation of IHL template fits
    - Model evaluation at arbitrary multipoles
    
    DO NOT use this for the actual IHL template fitting method - that uses the
    raw template files loaded per redshift bin with fit_model_with_ihl_templates().
    
    Parameters
    ----------
    fit_result : dict
        Fit result dictionary from fit_and_decompose_ihl_templates (element of 'fits')
    ell_values : array_like
        Multipole values where components should be evaluated
    
    Returns
    -------
    dict
        Dictionary with SMOOTH parametric components:
        - 'two_halo': smooth two-halo power law
        - 'one_halo': smooth one-halo log-normal (NOT raw template)
        - 'shot_noise': smooth shot noise
        - 'total': smooth total model
    
    """
    if 'error' in fit_result:
        raise ValueError(f"Fit has error: {fit_result['error']}")
    
    params = fit_result['params']
    A_2h, A_1h, mu_1h, sigma_1h, A_shot = params
    
    ell_values = np.asarray(ell_values)
    
    # Create SMOOTH parametric model (log-normal, not template)
    model = CrossPowerSpectrumModel(lb=ell_values, use_powerlaw_2h=True, alpha_2h_fixed=0.0)
    
    # Get smooth parametric components
    components = model.model_components(ell_values, A_2h, A_1h, mu_1h, sigma_1h, A_shot)
    
    return components


def save_ihl_1h_params(results, save_path, zbinedges=None, slopes=None):
    """
    Save one-halo parameters from IHL template decomposition for use in galaxy cross-fits.
    
    This function extracts the one-halo (mu_1h, sigma_1h) parameters from IHL template
    fits and saves them in a format that can be loaded by run_gal_cross_fits and related
    functions. The parameters are organized by redshift and slope.
    
    Parameters
    ----------
    results : dict
        Results dictionary from fit_and_decompose_ihl_templates() containing 'fits' and 'summary'
    save_path : str
        Path where to save the parameters file (e.g., 'ihl_1h_params.npz')
    zbinedges : array_like, optional
        Redshift bin edges. If None, extracted from results.
    slopes : list of float, optional
        Slope values. If None, extracted from results.
    
    Returns
    -------
    one_halo_params_dict : dict
        Dictionary with structure:
        {
            'zbinedges': array of redshift bin edges,
            'slopes': list of slope values,
            'params': dict with keys like (zidx, slope) -> {'mu_1h': float, 'sigma_1h': float},
            'ln_ell_peak_vs_z': dict with slope -> (intercept, slope) for ln(ell_peak) = intercept + slope*z,
            'sigma_vs_z': dict with slope -> (intercept, slope) for sigma = intercept + slope*z
        }
    
    """
    import numpy as np
    from scipy.stats import linregress
    
    fits = results['fits']
    summary = results['summary']
    
    # Extract zbinedges and slopes from results if not provided
    if zbinedges is None:
        if len(summary) > 0 and 'z_low' in summary.columns:
            z_lows = summary['z_low'].unique()
            z_highs = summary['z_high'].unique()
            zbinedges = np.sort(np.unique(np.concatenate([z_lows, z_highs])))
        else:
            raise ValueError("Cannot determine zbinedges from results. Please provide zbinedges parameter.")
    
    if slopes is None:
        # Extract slopes from fit results
        slopes_set = set()
        for fit in fits.values():
            if fit.get('slope') is not None:
                slopes_set.add(fit['slope'])
        slopes = sorted(list(slopes_set)) if slopes_set else [1.0]
    
    zbinedges = np.asarray(zbinedges)
    
    # Initialize storage
    params_dict = {}
    
    # Organize parameters by (z_center, slope)
    z_centers = []
    mu_1h_by_slope = {slope: [] for slope in slopes}
    sigma_1h_by_slope = {slope: [] for slope in slopes}
    
    for zidx in range(len(zbinedges) - 1):
        z_low = zbinedges[zidx]
        z_high = zbinedges[zidx + 1]
        z_center = (z_low + z_high) / 2
        
        if zidx == 0:
            z_centers.append(z_center)
        
        for slope in slopes:
            # Find matching template
            template_key = None
            for key in fits.keys():
                if f"z{z_low}_{z_high}" in key and f"slope{slope}" in key:
                    template_key = key
                    break
            
            if template_key is None:
                print(f"Warning: No fit found for z=[{z_low}, {z_high}], slope={slope}")
                continue
            
            fit = fits[template_key]
            
            if 'error' in fit:
                print(f"Warning: Fit failed for {template_key}, skipping")
                continue
            
            params = fit['params']
            mu_1h = params[2]  # ln(ell_peak)
            sigma_1h = params[3]  # log-width
            
            # Store parameters
            params_dict[(zidx, slope)] = {
                'mu_1h': mu_1h,
                'sigma_1h': sigma_1h,
                'ell_peak': np.exp(mu_1h),
                'z_low': z_low,
                'z_high': z_high,
                'z_center': z_center
            }
            
            mu_1h_by_slope[slope].append(mu_1h)
            sigma_1h_by_slope[slope].append(sigma_1h)
    
    # Fit linear relations: ln(ell_peak) vs z and sigma vs z for each slope
    ln_ell_peak_relations = {}
    sigma_relations = {}
    
    z_centers_array = 0.5 * (zbinedges[:-1] + zbinedges[1:])
    
    for slope in slopes:
        if len(mu_1h_by_slope[slope]) > 1:
            # Fit ln(ell_peak) = intercept + slope_param * z
            result_mu = linregress(z_centers_array, mu_1h_by_slope[slope])
            ln_ell_peak_relations[slope] = {
                'intercept': result_mu.intercept,
                'slope': result_mu.slope,
                'r_value': result_mu.rvalue,
                'stderr': result_mu.stderr
            }
            
            # Fit sigma = intercept + slope_param * z
            result_sigma = linregress(z_centers_array, sigma_1h_by_slope[slope])
            sigma_relations[slope] = {
                'intercept': result_sigma.intercept,
                'slope': result_sigma.slope,
                'r_value': result_sigma.rvalue,
                'stderr': result_sigma.stderr
            }
            
            print(f"\nSlope {slope}:")
            print(f"  ln(ell_peak) = {result_mu.intercept:.3f} + {result_mu.slope:.3f} * z  (R² = {result_mu.rvalue**2:.3f})")
            print(f"  sigma = {result_sigma.intercept:.3f} + {result_sigma.slope:.3f} * z  (R² = {result_sigma.rvalue**2:.3f})")
        else:
            print(f"Warning: Only {len(mu_1h_by_slope[slope])} data point(s) for slope {slope}, cannot fit linear relation")
            ln_ell_peak_relations[slope] = None
            sigma_relations[slope] = None
    
    # Create output dictionary
    one_halo_params_dict = {
        'zbinedges': zbinedges,
        'slopes': np.array(slopes),
        'params': params_dict,
        'ln_ell_peak_vs_z': ln_ell_peak_relations,
        'sigma_vs_z': sigma_relations,
        'z_centers': z_centers_array,
        'mu_1h_by_slope': mu_1h_by_slope,
        'sigma_1h_by_slope': sigma_1h_by_slope
    }
    
    # Save to file
    np.savez(save_path, 
             zbinedges=zbinedges,
             slopes=np.array(slopes),
             params_dict=params_dict,
             ln_ell_peak_vs_z=ln_ell_peak_relations,
             sigma_vs_z=sigma_relations,
             z_centers=z_centers_array,
             allow_pickle=True)
    
    print(f"\n✓ Saved one-halo parameters to: {save_path}")
    print(f"  - {len(zbinedges)-1} redshift bins")
    print(f"  - {len(slopes)} slope value(s)")
    print(f"  - {len(params_dict)} parameter sets")
    
    return one_halo_params_dict



def compare_ihl_to_data(template_dir, zbinedges, slopes, 
                       data_ell, data_dl, data_dl_err=None,
                       z_idx=0, plot=True, save_path=None):
    """
    Quick function to load, fit, and compare an IHL template to actual data.
    
    Parameters
    ----------
    template_dir : str
        Path to directory containing IHL template files
    zbinedges : array_like
        Redshift bin edges
    slopes : list of float
        Slope values to load
    data_ell : array_like
        Multipole bins for data
    data_dl : array_like
        Data D_ℓ values
    data_dl_err : array_like, optional
        Uncertainties on data D_ℓ
    z_idx : int, optional
        Which redshift bin to compare (default 0)
    plot : bool, optional
        Create comparison plot (default True)
    save_path : str, optional
        Path to save plot
    
    Returns
    -------
    dict
        Dictionary with:
        - 'template': template info
        - 'fit': fit results
        - 'data_comparison': dict with data and model at data multipoles
    
    Example
    -------
    # Load your measured cross-spectrum
    data_ell = np.array([500, 1000, 2000, 5000])
    data_dl = np.array([1.5, 3.2, 2.8, 1.1])  # Your measurements
    
    comparison = compare_ihl_to_data(
        template_dir='ihl_templates/',
        zbinedges=np.array([0.0, 0.2, 0.4]),
        slopes=[1.0],
        data_ell=data_ell,
        data_dl=data_dl,
        z_idx=0,  # First redshift bin
        plot=True
    )
    """
    # Load and fit the specific template
    results = fit_and_decompose_ihl_templates(
        template_dir=template_dir,
        zbinedges=zbinedges,
        slopes=slopes,
        plot=False,
        verbose=False
    )
    
    # Find the template for this redshift bin
    z_low = zbinedges[z_idx]
    z_high = zbinedges[z_idx + 1]
    template_key = None
    for key in results['fits'].keys():
        if f"z{z_low}_{z_high}" in key:
            template_key = key
            break
    
    if template_key is None:
        raise ValueError(f"No template found for z=[{z_low}, {z_high}]")
    
    fit_result = results['fits'][template_key]
    
    # Evaluate model at data multipoles
    components_at_data = get_ihl_components_at_ell(fit_result, data_ell)
    
    # Calculate chi-squared
    model_at_data = components_at_data['total']
    if data_dl_err is not None:
        chisq = np.sum(((data_dl - model_at_data) / data_dl_err)**2)
        reduced_chisq = chisq / (len(data_dl) - 5)  # 5 parameters
    else:
        chisq = np.sum((data_dl - model_at_data)**2)
        reduced_chisq = np.nan
    
    # Create comparison plot
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left panel: Full comparison
        ax = axes[0]
        
        # Plot data
        if data_dl_err is not None:
            ax.errorbar(data_ell, data_dl, yerr=data_dl_err,
                       fmt='o', markersize=10, capsize=5, capthick=2,
                       color='black', label='Data', zorder=10)
        else:
            ax.loglog(data_ell, data_dl, 'o', markersize=10,
                     color='black', label='Data', zorder=10)
        
        # Plot template and fit
        ax.loglog(fit_result['ell_template'], fit_result['dl_template'],
                 's', markersize=6, alpha=0.4, color='gray', label='IHL Template')
        
        ell_eval = fit_result['ell_eval']
        components = fit_result['components']
        ax.loglog(ell_eval, components['two_halo'], '--', 
                 linewidth=2, label='Two-halo', color='blue')
        ax.loglog(ell_eval, components['one_halo'], '--',
                 linewidth=2, label='One-halo', color='green')
        ax.loglog(ell_eval, components['shot_noise'], '--',
                 linewidth=2, label='Shot noise', color='red')
        ax.loglog(ell_eval, components['total'], '-',
                 linewidth=3, label='Total Model', color='orange')
        
        ax.set_xlabel(r'Multipole $\ell$', fontsize=13)
        ax.set_ylabel(r'$D_\ell$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=13)
        ax.set_title(f'IHL Template vs Data: z=[{z_low:.1f}, {z_high:.1f}]', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(alpha=0.3, which='both')
        
        # Add chi-squared info
        textstr = f"χ²/dof = {reduced_chisq:.2f}"
        ax.text(0.05, 0.05, textstr, transform=ax.transAxes,
               fontsize=11, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Right panel: Residuals
        ax = axes[1]
        residuals = data_dl - model_at_data
        if data_dl_err is not None:
            normalized_residuals = residuals / data_dl_err
            ax.errorbar(data_ell, normalized_residuals, yerr=1.0,
                       fmt='o', markersize=8, capsize=5, color='black')
            ax.set_ylabel('Normalized Residuals (Data - Model) / σ', fontsize=12)
        else:
            ax.semilogx(data_ell, residuals, 'o', markersize=8, color='black')
            ax.set_ylabel('Residuals: Data - Model', fontsize=12)
        
        ax.axhline(0, color='orange', linestyle='-', linewidth=2, alpha=0.7)
        ax.axhline(1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(-1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel(r'Multipole $\ell$', fontsize=13)
        ax.set_title('Fit Residuals', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"✓ Saved comparison plot to: {save_path}")
        else:
            plt.show()
    
    return {
        'template': results['templates'][template_key],
        'fit': fit_result,
        'data_comparison': {
            'data_ell': data_ell,
            'data_dl': data_dl,
            'data_dl_err': data_dl_err,
            'model_dl': model_at_data,
            'components': components_at_data,
            'chisq': chisq,
            'reduced_chisq': reduced_chisq
        }
    }



def load_ihl_1h_params(load_path):
    """
    Load one-halo parameters from IHL template fits for use in galaxy cross-fits.
    
    This function loads the one-halo parameters saved by save_ihl_1h_params() and
    returns them in a format compatible with run_gal_cross_fits and interpolate_1h_params.
    
    Parameters
    ----------
    load_path : str
        Path to the saved parameters file (e.g., 'ihl_1h_params.npz')
    
    Returns
    -------
    one_halo_params_dict : dict
        Dictionary with structure compatible with interpolate_1h_params:
        {
            'zbinedges': array of redshift bin edges,
            'slopes': array of slope values,
            'params': dict with keys (zidx, slope) -> {'mu_1h': float, 'sigma_1h': float},
            'ln_ell_peak_vs_z': dict with slope -> linear fit parameters,
            'sigma_vs_z': dict with slope -> linear fit parameters,
            'z_centers': array of redshift bin centers
        }
    """
    import numpy as np
    
    data = np.load(load_path, allow_pickle=True)
    
    one_halo_params_dict = {
        'zbinedges': data['zbinedges'],
        'slopes': data['slopes'],
        'params': data['params_dict'].item() if isinstance(data['params_dict'], np.ndarray) else data['params_dict'],
        'ln_ell_peak_vs_z': data['ln_ell_peak_vs_z'].item() if isinstance(data['ln_ell_peak_vs_z'], np.ndarray) else data['ln_ell_peak_vs_z'],
        'sigma_vs_z': data['sigma_vs_z'].item() if isinstance(data['sigma_vs_z'], np.ndarray) else data['sigma_vs_z'],
        'z_centers': data['z_centers']
    }
    
    print(f"✓ Loaded one-halo parameters from: {load_path}")
    print(f"  - {len(one_halo_params_dict['zbinedges'])-1} redshift bins")
    print(f"  - {len(one_halo_params_dict['slopes'])} slope value(s): {one_halo_params_dict['slopes']}")
    
    # Print linear relations if available
    print("\nLinear relations from IHL template fits:")
    for slope in one_halo_params_dict['slopes']:
        ln_rel = one_halo_params_dict['ln_ell_peak_vs_z'].get(slope)
        sigma_rel = one_halo_params_dict['sigma_vs_z'].get(slope)
        
        if ln_rel is not None:
            print(f"  Slope {slope}:")
            print(f"    ln(ell_peak) = {ln_rel['intercept']:.3f} + {ln_rel['slope']:.3f} * z")
            if sigma_rel is not None:
                print(f"    sigma = {sigma_rel['intercept']:.3f} + {sigma_rel['slope']:.3f} * z")
    
    return one_halo_params_dict


def update_interpolate_1h_params_from_ihl(one_halo_params_dict, slope=None):
    """
    Update the interpolate_1h_params function to use IHL-derived parameters.
    
    This function modifies the global behavior of interpolate_1h_params to use
    the parameters derived from IHL template fits instead of hard-coded values.
    
    Parameters
    ----------
    one_halo_params_dict : dict
        Dictionary from load_ihl_1h_params() with IHL-derived parameters
    slope : float, optional
        Which slope to use. If None, uses the first available slope.
    
    Returns
    -------
    dict
        Dictionary with the linear relations that will be used:
        - 'ln_ell_peak': (intercept, slope_param)
        - 'sigma': (intercept, slope_param)
    
    Example
    -------
    # Load IHL parameters
    one_halo_params = load_ihl_1h_params('ihl_1h_params.npz')
    
    # Get the relations for a specific slope
    relations = update_interpolate_1h_params_from_ihl(one_halo_params, slope=1.0)
    
    # Now interpolate_1h_params will use these relations
    ln_ell_peak, sigma = interpolate_1h_params(z_value=0.3)
    """
    if slope is None:
        slope = one_halo_params_dict['slopes'][0]
        print(f"No slope specified, using slope={slope}")
    
    ln_rel = one_halo_params_dict['ln_ell_peak_vs_z'].get(slope)
    sigma_rel = one_halo_params_dict['sigma_vs_z'].get(slope)
    
    if ln_rel is None or sigma_rel is None:
        raise ValueError(f"No linear relations available for slope {slope}")
    
    relations = {
        'ln_ell_peak': (ln_rel['intercept'], ln_rel['slope']),
        'sigma': (sigma_rel['intercept'], sigma_rel['slope']),
        'slope': slope
    }
    
    print(f"\nUsing IHL-derived relations for slope {slope}:")
    print(f"  ln(ell_peak) = {ln_rel['intercept']:.3f} + {ln_rel['slope']:.3f} * z")
    print(f"  sigma = {sigma_rel['intercept']:.3f} + {sigma_rel['slope']:.3f} * z")
    
    return relations