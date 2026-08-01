"""
Parametric modeling for CIBER cross-power spectra

This module implements a parametric decomposition of cross-power spectra into:
1. Two-halo term (linear matter power spectrum shape with free amplitude)
2. One-halo term (log-normal in D_ℓ to capture non-linear fluctuations)
3. Shot noise term (ℓ² with free amplitude)

The model is: D_ℓ(ℓ) = A_2h * D_ℓ^{2h}(ℓ) + A_1h * exp(-(log(ℓ) - μ)²/(2σ²)) + A_shot * ℓ²

Author: Richard Feder
Date: January 2026
"""

from tabnanny import verbose

import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, Callable
import os
from functools import partial

from ciber.plotting.gal_plotting_fns import *
from ciber.theory.cl_template import *
from ciber.theory.onehalo_predict import load_onehalo_spectrum


from dataclasses import dataclass

DEFAULT_BOUNDS = {
    'A_2h': (0., 10.),
    'A_1h': (0., 10.),
    'mu_1h': (np.log(500.), np.log(30000.)),  # log(ell_peak) bounds
    'sigma_1h': (0.1, 1.5),  # log-space width bounds
    'A_shot': (0., 100.),
    'f_pop': (0., 1.),
    'sigma_damp': (0.1, 4.0),
}


def _full_param_name_sequence(include_sigma_damp=False, use_onehalo_popmix=False):
    names = ['A_2h', 'A_1h', 'mu_1h', 'sigma_1h', 'A_shot']
    if use_onehalo_popmix:
        names.append('f_pop')
    if include_sigma_damp:
        names.append('sigma_damp')
    return names


def _normalize_param_name(name):
    if name is None:
        return ''
    text = str(name).strip().lower()
    text = text.replace('$', '').replace('\\', '').replace('{', '').replace('}', '').replace(' ', '')
    aliases = {
        'a2h': 'A_2h',
        'a_2h': 'A_2h',
        'a1h': 'A_1h',
        'a_1h': 'A_1h',
        'mu1h': 'mu_1h',
        'mu_1h': 'mu_1h',
        'sigma1h': 'sigma_1h',
        'sigma_1h': 'sigma_1h',
        'ashot': 'A_shot',
        'a_shot': 'A_shot',
        'fpop': 'f_pop',
        'f_pop': 'f_pop',
        'sigmadamp': 'sigma_damp',
        'sigma_damp': 'sigma_damp',
        'damp': 'sigma_damp',
    }
    # Try exact match first
    if text in aliases:
        return aliases[text]
    # Check if key substrings are present (for LaTeX-formatted names)
    if 'damp' in text:
        return 'sigma_damp'
    if 'fpop' in text or 'f_pop' in text:
        return 'f_pop'
    return text


def resolve_full_param_value(params, param_names, param_name, use_astrometry_damping=False, use_onehalo_popmix=False, model=None):
    """Resolve a parameter value from either a full parameter vector or a fitted-parameter subset."""
    if params is None:
        return None

    if isinstance(param_names, (list, tuple, np.ndarray)):
        names = list(param_names)
    else:
        names = _full_param_name_sequence(
            include_sigma_damp=bool(use_astrometry_damping),
            use_onehalo_popmix=bool(use_onehalo_popmix),
        )

    if isinstance(params, dict):
        return params.get(param_name, None)

    params_arr = np.asarray(params, dtype=float)
    if params_arr.ndim == 0:
        return None

    normalized_target = _normalize_param_name(param_name)
    if len(names) == len(params_arr):
        for idx, name in enumerate(names):
            if _normalize_param_name(name) == normalized_target:
                return float(params_arr[idx])
        return None

    # Allow a fitted-parameter subset to be mapped using a standard full-parameter ordering.
    if param_name == 'A_2h':
        return float(params_arr[0]) if len(params_arr) > 0 else None
    if param_name == 'A_1h':
        return float(params_arr[1]) if len(params_arr) > 1 else None
    if param_name == 'mu_1h':
        return float(params_arr[2]) if len(params_arr) > 2 else None
    if param_name == 'sigma_1h':
        return float(params_arr[3]) if len(params_arr) > 3 else None
    if param_name == 'A_shot':
        return float(params_arr[4]) if len(params_arr) > 4 else None
    if param_name == 'f_pop':
        if use_onehalo_popmix:
            return float(params_arr[5]) if len(params_arr) > 5 else None
        return None
    if param_name == 'sigma_damp':
        if use_astrometry_damping:
            # sigma_damp is at index 6 if popmix is enabled, else at index 5
            damp_idx = 6 if use_onehalo_popmix else 5
            print('SIGMA DAMP RESOLVED IS ', params_arr[damp_idx])
            return float(params_arr[damp_idx]) if len(params_arr) > damp_idx else None
        return None
    return None


def expand_fit_samples_to_full_vector(samples, params_full, param_names_fitted=None, use_astrometry_damping=False, use_onehalo_popmix=False):
    """Expand MCMC samples from fitted-parameter space into the full model parameter vector."""
    if samples is None:
        return None

    samples_arr = np.asarray(samples, dtype=float)
    if samples_arr.ndim == 1:
        samples_arr = samples_arr[np.newaxis, :]

    n_samples = samples_arr.shape[0]
    full_names = _full_param_name_sequence(
        include_sigma_damp=bool(use_astrometry_damping),
        use_onehalo_popmix=bool(use_onehalo_popmix),
    )
    params_full_arr = np.asarray(params_full, dtype=float)[:len(full_names)]

    if samples_arr.shape[1] == len(full_names):
        return samples_arr

    full_samples = np.tile(params_full_arr, (n_samples, 1))
    if samples_arr.shape[1] == 0:
        return full_samples

    if param_names_fitted is None:
        param_names_fitted = []

    name_labels = [_normalize_param_name(name) for name in param_names_fitted]
    if len(name_labels) == samples_arr.shape[1]:
        for fit_idx, label in enumerate(name_labels):
            if label in {'A_2h', 'a2h'}:
                target = 'A_2h'
            elif label in {'A_1h', 'a1h'}:
                target = 'A_1h'
            elif label in {'mu_1h', 'mu1h'}:
                target = 'mu_1h'
            elif label in {'sigma_1h', 'sigma1h'}:
                target = 'sigma_1h'
            elif label in {'A_shot', 'ashot'}:
                target = 'A_shot'
            elif label in {'f_pop', 'fpop'}:
                target = 'f_pop'
            elif label in {'sigma_damp'}:
                target = 'sigma_damp'
            else:
                target = None

            if target is not None:
                target_idx = full_names.index(target)
                full_samples[:, target_idx] = samples_arr[:, fit_idx]
        return full_samples

    # Fallback for common fixed-shape cases.
    if use_astrometry_damping and use_onehalo_popmix:
        # Fitted order: [A_2h, A_1h, A_shot, f_pop, sigma_damp]
        if samples_arr.shape[1] >= 5:
            full_samples[:, 0] = samples_arr[:, 0]
            full_samples[:, 1] = samples_arr[:, 1]
            full_samples[:, 4] = samples_arr[:, 2]
            full_samples[:, 5] = samples_arr[:, 3]
            full_samples[:, 6] = samples_arr[:, 4]
            return full_samples
    if use_astrometry_damping and not use_onehalo_popmix:
        if samples_arr.shape[1] >= 4:
            full_samples[:, 0] = samples_arr[:, 0]
            full_samples[:, 1] = samples_arr[:, 1]
            full_samples[:, 4] = samples_arr[:, 2]
            full_samples[:, 5] = samples_arr[:, 3]
            return full_samples
    if use_onehalo_popmix and not use_astrometry_damping:
        if samples_arr.shape[1] >= 4:
            full_samples[:, 0] = samples_arr[:, 0]
            full_samples[:, 1] = samples_arr[:, 1]
            full_samples[:, 4] = samples_arr[:, 2]
            full_samples[:, 5] = samples_arr[:, 3]
            return full_samples
    if samples_arr.shape[1] >= 3:
        full_samples[:, 0] = samples_arr[:, 0]
        full_samples[:, 1] = samples_arr[:, 1]
        full_samples[:, 4] = samples_arr[:, 2]
    return full_samples


def _bounds_from_names(names):
    lo = np.array([DEFAULT_BOUNDS[n][0] for n in names], float)
    hi = np.array([DEFAULT_BOUNDS[n][1] for n in names], float)
    return lo, hi


def _autocorr_time(x):
    """Estimate the integrated autocorrelation time for a 1D chain."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return np.nan

    try:
        from emcee.autocorr import integrated_time as emcee_integrated_time
        return float(emcee_integrated_time(x, quiet=True))
    except Exception:
        x = x - np.mean(x)
        n = x.size
        corr = np.correlate(x, x, mode='full')[n - 1:] / np.arange(n, 0, -1)
        corr = corr / corr[0]
        tau = 1.0
        for lag in range(1, len(corr)):
            if corr[lag] <= 0:
                break
            tau += 2.0 * corr[lag]
        return float(tau)


def _gelman_rubin(chains):
    """Compute the Gelman-Rubin R-hat statistic for a set of chains."""
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 3:
        raise ValueError("chains must have shape (nchains, nsteps, nparams)")

    n_chains, n_steps, n_params = chains.shape
    if n_chains < 2:
        return np.ones(n_params, dtype=float)

    chain_means = np.mean(chains, axis=1)
    chain_vars = np.var(chains, axis=1, ddof=1)

    mean_of_means = np.mean(chain_means, axis=0)
    between = n_steps * np.var(chain_means, axis=0, ddof=1)
    within = np.mean(chain_vars, axis=0)
    var_est = ((n_steps - 1.0) / n_steps) * within + between / (n_steps * n_chains)
    rhat = np.sqrt(np.divide(var_est, within, out=np.ones_like(var_est), where=np.abs(within) > 1e-30))
    return rhat


def _compute_mcmc_diagnostics(samples, sampler=None, discard=0, param_names=None):
    """Compute autocorrelation time, R-hat, and effective sample size."""
    samples_arr = np.asarray(samples, dtype=float)
    if samples_arr.size == 0:
        tau = np.array([], dtype=float)
        rhat = np.array([], dtype=float)
        ess = np.array([], dtype=float)
        return {'tau': tau, 'rhat': rhat, 'ess': ess, 'param_names': list(param_names or [])}

    if samples_arr.ndim == 1:
        samples_arr = samples_arr[:, None]
    if samples_arr.ndim != 2:
        raise ValueError("samples must be 1D or 2D")

    n_samples, n_params = samples_arr.shape
    tau = np.array([_autocorr_time(samples_arr[:, j]) for j in range(n_params)], dtype=float)
    ess = np.array([float(n_samples / max(tau_j, 1e-12)) for tau_j in tau], dtype=float)
    ess = np.clip(ess, 0.0, float(n_samples))

    if sampler is not None:
        try:
            chain_samples = sampler.get_chain(discard=discard, flat=False)
            if chain_samples.ndim == 3:
                chains = np.moveaxis(chain_samples, 0, 1)
                rhat = _gelman_rubin(chains)
            else:
                rhat = np.ones(n_params, dtype=float)
        except Exception:
            rhat = np.ones(n_params, dtype=float)
    else:
        rhat = np.ones(n_params, dtype=float)

    return {'tau': tau, 'rhat': rhat, 'ess': ess, 'param_names': list(param_names or [])}


@dataclass
class FitConfig:
    # model switches
    use_two_halo: bool
    use_one_halo: bool
    use_astrometry_damping: bool

    # fixed flags
    fixed_A2h: bool
    fixed_mu_sigma: bool
    fixed_sigma_damp: bool

    # fixed values
    A2h_val: float | None = None
    mu_val: float | None = None
    sigma_val: float | None = None
    ln_ell_peak_val: float | None = None
    sigma_damp_val: float | None = None

    # fitted parameter names (subset space)
    fit_names: list[str] = None
    full_param_names: list[str] = None
    fixed_values: dict | None = None
    use_onehalo_popmix: bool = False


@dataclass
class PlotConfig:
    """Configuration extracted from fit_result for plotting."""
    params: np.ndarray
    params_err: np.ndarray | None = None
    use_damping: bool = False
    cov_matrix: np.ndarray | None = None
    samples: np.ndarray | None = None
    chi2_eval_max: float | None = None
    z_value: float | None = None
    one_halo_params_dict: dict | None = None
    sigma_fixed: float | None = None


def load_mock_prediction_component(npz_path, component='cross'):
    """Load a saved mock prediction spectrum and return (ell, cl, cl_err).

    The saved mock products are not perfectly uniform across generators, so this
    helper accepts the common key variants used in the repository.
    """
    pred = np.load(npz_path, allow_pickle=True)
    lb = np.asarray(pred['lb'], dtype=float)

    key_candidates = {
        'cross': ('cross', 'clx', 'clx_comb', 'cross_pred'),
        'gal_auto': ('gal_auto', 'clg', 'clg_comb', 'auto'),
        'intensity_auto_tracer': ('intensity_auto_tracer',),
        'intensity_auto_full': ('intensity_auto_full',),
    }
    err_candidates = {
        'cross': ('cross_err', 'clx_err', 'clx_comb_err', 'cross_pred_err'),
        'gal_auto': ('gal_auto_err', 'clg_err', 'clg_comb_err', 'auto_err'),
        'intensity_auto_tracer': ('intensity_auto_tracer_err',),
        'intensity_auto_full': ('intensity_auto_full_err',),
    }

    if component not in key_candidates:
        raise KeyError(f"Unsupported mock prediction component: {component}")

    cl = None
    for key in key_candidates[component]:
        if key in pred:
            cl = np.asarray(pred[key], dtype=float)
            break
    if cl is None:
        raise KeyError(f"Could not find a spectrum key for component '{component}' in {npz_path}")

    cl_err = None
    for key in err_candidates[component]:
        if key in pred:
            cl_err = np.asarray(pred[key], dtype=float)
            break

    return lb, cl, cl_err


def estimate_mock_two_halo_amplitude(npz_path, component='cross',
                                     shot_ell_range=(50000.0, 80000.0),
                                     signal_ell_max=2000.0):
    """Estimate an effective 2h amplitude from a saved mock prediction spectrum.

    The estimator works in D_ell units: it converts the loaded spectrum from C_ell,
    estimates the shot-noise plateau from a high-ell window, subtracts it, and
    averages the low-ell residual to produce an effective A_2h summary.
    """
    lb, cl, cl_err = load_mock_prediction_component(npz_path, component=component)
    pf = lb * (lb + 1.0) / (2.0 * np.pi)
    dl = pf * cl
    dl_err = pf * cl_err if cl_err is not None else None

    shot_mask = np.isfinite(lb) & np.isfinite(dl) & (lb >= shot_ell_range[0]) & (lb <= shot_ell_range[1])
    if not np.any(shot_mask):
        shot_mask = np.isfinite(lb) & np.isfinite(dl) & (lb >= 0)
    signal_mask = np.isfinite(lb) & np.isfinite(dl) & (lb <= signal_ell_max)
    if not np.any(signal_mask):
        signal_mask = np.isfinite(lb) & np.isfinite(dl)

    shot_level = float(np.nanmean(dl[shot_mask]))
    shot_level_err = float(np.nanstd(dl[shot_mask], ddof=1) / np.sqrt(np.sum(shot_mask))) if np.sum(shot_mask) > 1 else np.nan

    dl_sub = dl - shot_level
    a2h = float(np.nanmean(dl_sub[signal_mask]))

    if dl_err is not None:
        signal_err = float(np.sqrt(np.nansum(dl_err[signal_mask] ** 2)) / np.sum(signal_mask)) if np.sum(signal_mask) > 0 else np.nan
    else:
        signal_err = float(np.nanstd(dl_sub[signal_mask], ddof=1) / np.sqrt(np.sum(signal_mask))) if np.sum(signal_mask) > 1 else np.nan

    a2h_err = float(np.sqrt(signal_err ** 2 + shot_level_err ** 2)) if np.isfinite(signal_err) and np.isfinite(shot_level_err) else signal_err

    return {
        'path': str(npz_path),
        'component': component,
        'ell': lb,
        'dl': dl,
        'dl_err': dl_err,
        'shot_ell_range': tuple(float(v) for v in shot_ell_range),
        'signal_ell_max': float(signal_ell_max),
        'shot_level': shot_level,
        'shot_level_err': shot_level_err,
        'a2h': a2h,
        'a2h_err': a2h_err,
        'dl_subtracted': dl_sub,
    }


class CrossPowerSpectrumModel:
    """
    Parametric model for cross-power spectra decomposition.
    
    Parameters
    ----------
    lb : array_like
        Multipole bin centers
    cl_2h_pred : array_like, optional
        Two-halo prediction from theory/mocks (in C_ℓ). If provided, used for reference only.
    cl_shot_pred : array_like, optional
        Shot noise prediction (in C_ℓ). If None, will be estimated from model.
    use_powerlaw_2h : bool, optional
        If True, model 2-halo term as power law. If False, use cl_2h_pred. Default True.
    alpha_2h_fixed : float, optional
        Fixed power-law index for 2-halo term (default -1.5 for linear clustering)
    chi2_eval_max : float, optional
        Maximum multipole for chi-square evaluation. Default 5000.

    """
    
    def __init__(self, lb, cl_2h_pred=None, cl_shot_pred=None, use_powerlaw_2h=True, alpha_2h_fixed=-1.5,
                 chi2_eval_max=5000., mu_1h_fixed=None, sigma_1h_fixed=None, use_astrometry_damping=False,
                 use_one_halo=True, use_two_halo=True, A_2h_fixed=None, use_linear_2h=False, 
                 dl_2h_lin_per_zbin=None, sigma_damp_fixed=None, onehalo_template_1h_dl=None):
        self.lb = np.asarray(lb)
        self.cl_2h_pred = np.asarray(cl_2h_pred) if cl_2h_pred is not None else None
        self.cl_shot_pred = cl_shot_pred
        self.use_powerlaw_2h = use_powerlaw_2h
        self.alpha_2h_fixed = alpha_2h_fixed
        self.chi2_eval_max = chi2_eval_max
        self.mu_1h_fixed = mu_1h_fixed
        self.sigma_1h_fixed = sigma_1h_fixed
        self.use_astrometry_damping = use_astrometry_damping
        self.use_one_halo = use_one_halo
        self.use_two_halo = use_two_halo
        self.A_2h_fixed = A_2h_fixed
        self.use_linear_2h = use_linear_2h
        self.dl_2h_lin_per_zbin = dl_2h_lin_per_zbin if dl_2h_lin_per_zbin is not None else {}
        self.sigma_damp_fixed = sigma_damp_fixed if sigma_damp_fixed is not None else {}
        self.onehalo_template_1h_dl = onehalo_template_1h_dl
        
        # Convert to D_ℓ
        self.pf = self.lb * (self.lb + 1) / (2 * np.pi)
        
        if self.cl_2h_pred is not None:
            self.dl_2h_pred = self.pf * self.cl_2h_pred
        else:
            self.dl_2h_pred = None
        
        if self.cl_shot_pred is not None:
            self.dl_shot_pred = self.pf * np.asarray(self.cl_shot_pred)
        else:
            self.dl_shot_pred = None
    
    @staticmethod
    def lognormal_component(ell, amplitude, mu, sigma):
        """
        Log-normal component in D_ℓ to model one-halo term.
        
        Parameters
        ----------
        ell : array_like
            Multipole values
        amplitude : float
            Amplitude of the log-normal
        mu : float
            Center of the log-normal in log-space (typically log(ℓ_peak))
        sigma : float
            Width of the log-normal
        
        Returns
        -------
        array_like
            D_ℓ contribution from one-halo term
        """
        log_ell = np.log(ell)
        return amplitude * np.exp(-(log_ell - mu)**2 / (2 * sigma**2))
        
    @staticmethod
    def shot_noise_component(ell, amplitude):
        """
        Shot noise component proportional to ℓ².
        
        Parameters
        ----------
        ell : array_like
            Multipole values
        amplitude : float
            Shot noise amplitude
        
        Returns
        -------
        array_like
            D_ℓ contribution from shot noise
        """
        pf = ell * (ell + 1) / (2 * np.pi)
        return amplitude * pf
    
    @staticmethod
    def astrometry_damping_component(ell, sigma_arcsec):
        """
        Exponential damping factor to model high-ell suppression from astrometry errors.
        
        Applies exp(-0.5 * (σ * ℓ)²) where σ is in arcseconds.
        This models Gaussian smoothing in real space which becomes exponential damping in Fourier space.
        
        Parameters
        ----------
        ell : array_like
            Multipole values
        sigma_arcsec : float
            Astrometry error in arcseconds (typically 1-10 arcsec)
            
        Returns
        -------
        array_like
            Damping factors (multiplicative, between 0 and 1)
        """
        ell = np.asarray(ell)
        # Convert arcsec to radians for dimensionless ℓσ
        sigma_rad = sigma_arcsec * (1.0 / 3600.0) * (np.pi / 180.0)
        return np.exp(-0.5 * (sigma_rad * ell)**2)
    
    @staticmethod
    def powerlaw_2h_component(ell, amplitude, index, ell_pivot=1000.):
        """
        Power-law component for two-halo term.
        
        Parameters
        ----------
        ell : array_like
            Multipole values
        amplitude : float
            Amplitude at ell_pivot
        index : float
            Power-law index (typically negative)
        ell_pivot : float, optional
            Pivot scale for power law
        
        Returns
        -------
        array_like
            D_ℓ contribution from two-halo power law
        """
        return amplitude * (ell / ell_pivot)**index
    
    @staticmethod
    def ihl_template_component(ell, amplitude, template_ell, template_dl):
        """
        IHL template component with interpolation to data multipoles.
        For ell < 300 and below the minimum template ell, uses linear extrapolation
        with the same slope as the lowest template bins.
        
        Parameters
        ----------
        ell : array_like
            Multipole values where component is evaluated
        amplitude : float
            Amplitude scaling factor for template
        template_ell : array_like
            Template multipole values
        template_dl : array_like
            Template D_ℓ values
        
        Returns
        -------
        array_like
            D_ℓ contribution from scaled IHL template, interpolated to ell
        """
        ell = np.asarray(ell)
        template_ell = np.asarray(template_ell)
        template_dl = np.asarray(template_dl)
        
        # Find points that need extrapolation (ell < 300 and below minimum template ell)
        min_template_ell = template_ell[0]
        extrapolate_mask = (ell < 300) & (ell < min_template_ell)
        
        if np.any(extrapolate_mask):
            # Calculate slope from first two template points (in log-log space for power-law behavior)
            if len(template_ell) >= 2:
                log_ell1, log_ell2 = np.log(template_ell[0]), np.log(template_ell[1])
                log_dl1, log_dl2 = np.log(template_dl[0]), np.log(template_dl[1])
                slope = (log_dl2 - log_dl1) / (log_ell2 - log_ell1)
                
                # Extrapolate in log-log space
                log_ell_extrap = np.log(ell[extrapolate_mask])
                log_dl_extrap = log_dl1 + slope * (log_ell_extrap - log_ell1)
                dl_extrap = np.exp(log_dl_extrap)
            else:
                # Fallback: constant extrapolation if only one template point
                dl_extrap = np.full(np.sum(extrapolate_mask), template_dl[0])
            
            # Interpolate normally for all other points
            dl_interp = np.interp(ell, template_ell, template_dl, left=0.0, right=template_dl[-1])
            
            # Replace extrapolated values
            dl_interp[extrapolate_mask] = dl_extrap
        else:
            # No extrapolation needed, use normal interpolation
            dl_interp = np.interp(ell, template_ell, template_dl, left=0.0, right=template_dl[-1])
        
        return amplitude * dl_interp
    
    def _get_onehalo_template(self, z_bin_index=None):
        """Return a fixed one-halo template for the requested redshift bin, if available."""
        if self.onehalo_template_1h_dl is None:
            return None

        if isinstance(self.onehalo_template_1h_dl, dict):
            if z_bin_index is not None and z_bin_index in self.onehalo_template_1h_dl:
                template = self.onehalo_template_1h_dl[z_bin_index]
            else:
                template = self.onehalo_template_1h_dl.get(None)
                if template is None and len(self.onehalo_template_1h_dl) == 1:
                    template = next(iter(self.onehalo_template_1h_dl.values()))
                elif template is None and z_bin_index is None:
                    template = next(iter(self.onehalo_template_1h_dl.values()))
        else:
            template = self.onehalo_template_1h_dl

        if template is None:
            return None

        if isinstance(template, dict):
            if 'ell_arr' in template:
                ell_arr = template['ell_arr']
                dl_spectrum = template['dl_spectrum']
            elif 'ell' in template and 'dl' in template:
                ell_arr = template['ell']
                dl_spectrum = template['dl']
            else:
                raise ValueError("One-halo template dict must contain 'ell_arr'/'dl_spectrum' or 'ell'/'dl'.")
        elif isinstance(template, (list, tuple)) and len(template) == 2:
            ell_arr, dl_spectrum = template
        else:
            raise TypeError("Unsupported one-halo template format.")

        return {
            'ell_arr': np.asarray(ell_arr),
            'dl_spectrum': np.asarray(dl_spectrum),
            'dl_spectrum_pop0': np.asarray(template['dl_spectrum_pop0']) if isinstance(template, dict) and template.get('dl_spectrum_pop0') is not None else None,
            'dl_spectrum_pop1': np.asarray(template['dl_spectrum_pop1']) if isinstance(template, dict) and template.get('dl_spectrum_pop1') is not None else None,
        }

    def _supports_onehalo_population_mixture(self, z_bin_index=None):
        template_data = self._get_onehalo_template(z_bin_index=z_bin_index)
        if template_data is None:
            return False
        return template_data.get('dl_spectrum_pop0') is not None and template_data.get('dl_spectrum_pop1') is not None

    def _full_param_name_sequence(self, include_sigma_damp=False, use_onehalo_popmix=False):
        return _full_param_name_sequence(
            include_sigma_damp=include_sigma_damp,
            use_onehalo_popmix=use_onehalo_popmix,
        )

    def _expand_fit_params(self, params_subset, cfg: FitConfig, zero_fixed=False):
        full = dict(cfg.fixed_values or {})
        for i, name in enumerate(cfg.fit_names):
            full[name] = float(params_subset[i])

        if zero_fixed:
            for name, value in list(full.items()):
                if name not in cfg.fit_names and value is not None:
                    full[name] = 0.0

        ordered = [full[name] for name in cfg.full_param_names]
        return np.asarray(ordered, dtype=float)

    def _onehalo_template_component(self, ell, amplitude, mu_1h=None, sigma_1h=None, z_bin_index=None, f_pop=None):
        """Evaluate the one-halo term from either a fixed template or a lognormal fallback."""
        if not self.use_one_halo:
            return np.zeros_like(ell, dtype=float)

        template_data = self._get_onehalo_template(z_bin_index=z_bin_index)
        if template_data is not None:
            dl_template = template_data['dl_spectrum']
            dl_pop0 = template_data.get('dl_spectrum_pop0')
            dl_pop1 = template_data.get('dl_spectrum_pop1')
            if dl_pop0 is not None and dl_pop1 is not None and f_pop is not None:
                f_use = float(np.clip(f_pop, 0.0, 1.0))
                dl_template = (1.0 - f_use) * dl_pop0 + f_use * dl_pop1
            return self.ihl_template_component(
                ell,
                amplitude,
                template_data['ell_arr'],
                dl_template,
            )

        if mu_1h is None or sigma_1h is None:
            raise ValueError("mu_1h and sigma_1h are required when no one-halo template is provided")
        return self.lognormal_component(ell, amplitude, mu_1h, sigma_1h)

    def _build_fit_config(self, z_value=None, inst=None, verbose=True, z_bin_index=None):
        fixed_mu_sigma = (self.mu_1h_fixed is not None and self.sigma_1h_fixed is not None)
        fixed_A2h = (self.A_2h_fixed is not None) and self.use_one_halo and self.use_two_halo
        use_onehalo_popmix = self.use_one_halo and self._supports_onehalo_population_mixture(z_bin_index=z_bin_index)

        ln_val = None

        fixed_sigma_damp = False
        sigma_damp_val = None
        if self.use_astrometry_damping and inst is not None and self.sigma_damp_fixed:
            sigma_damp_val = self.sigma_damp_fixed.get(inst, None)
            fixed_sigma_damp = sigma_damp_val is not None

        full_param_names = self._full_param_name_sequence(
            include_sigma_damp=self.use_astrometry_damping,
            use_onehalo_popmix=use_onehalo_popmix,
        )

        fixed_values = {
            'A_2h': None,
            'A_1h': None,
            'mu_1h': None,
            'sigma_1h': None,
            'A_shot': None,
        }
        if use_onehalo_popmix:
            fixed_values['f_pop'] = None
        if self.use_astrometry_damping:
            fixed_values['sigma_damp'] = None

        if not self.use_two_halo:
            fixed_values['A_2h'] = 0.0
        elif fixed_A2h:
            fixed_values['A_2h'] = self.A_2h_fixed

        if not self.use_one_halo:
            fixed_values['A_1h'] = 0.0
            fixed_values['mu_1h'] = 0.0
            fixed_values['sigma_1h'] = 0.0
            if use_onehalo_popmix:
                fixed_values['f_pop'] = 0.5
        elif fixed_mu_sigma:
            fixed_values['mu_1h'] = self.mu_1h_fixed
            fixed_values['sigma_1h'] = self.sigma_1h_fixed

        if self.use_astrometry_damping and fixed_sigma_damp:
            fixed_values['sigma_damp'] = sigma_damp_val

        fit_names = [name for name in full_param_names if fixed_values.get(name) is None]

        return FitConfig(
            use_two_halo=self.use_two_halo,
            use_one_halo=self.use_one_halo,
            use_astrometry_damping=self.use_astrometry_damping,
            fixed_A2h=fixed_A2h,
            fixed_mu_sigma=fixed_mu_sigma,
            fixed_sigma_damp=fixed_sigma_damp,
            A2h_val=self.A_2h_fixed if fixed_A2h else None,
            mu_val=self.mu_1h_fixed if fixed_mu_sigma else None,
            sigma_val=self.sigma_1h_fixed if fixed_mu_sigma else None,
            ln_ell_peak_val=ln_val,
            sigma_damp_val=sigma_damp_val,
            fit_names=fit_names,
            full_param_names=full_param_names,
            fixed_values=fixed_values,
            use_onehalo_popmix=use_onehalo_popmix,
        )


    def model_dl(self, ell, A_2h, A_1h, mu_1h, sigma_1h, A_shot, sigma_damp=None, z_bin_index=None, f_pop=None):
        """
        Full parametric model in D_ℓ space.
        
        Parameters
        ----------
        ell : array_like
            Multipole values
        A_2h : float
            Amplitude of two-halo term (at ℓ=1000 for power law, or scaling for prediction)
        A_1h : float
            Amplitude of one-halo log-normal
        mu_1h : float
            Center of one-halo log-normal
        sigma_1h : float
            Width of one-halo log-normal
        A_shot : float
            Shot noise amplitude
        sigma_damp : float, optional
            Astrometry error in arcseconds for high-ell damping (if use_astrometry_damping=True)
            Typically 1-10 arcsec range
        z_bin_index : int or None, optional
            Index of redshift bin for selecting linear 2H template. Required if use_linear_2h=True.
        
        Returns
        -------
        array_like
            Total D_ℓ model
        """
        # Two-halo contribution
        if self.use_linear_2h:
            # Use linear 2H template for this z-bin
            if z_bin_index is None:
                raise ValueError("z_bin_index required when use_linear_2h=True")
            if z_bin_index not in self.dl_2h_lin_per_zbin:
                raise ValueError(f"Linear 2H template not found for z_bin_index={z_bin_index}")
            ell_lin, dl_lin = self.dl_2h_lin_per_zbin[z_bin_index]
            dl_2h = A_2h * np.interp(ell, ell_lin, dl_lin, left=0.0, right=0.0)
        elif self.use_powerlaw_2h:
            dl_2h = self.powerlaw_2h_component(ell, A_2h, self.alpha_2h_fixed)
        else:
            if self.dl_2h_pred is None:
                raise ValueError("No 2-halo prediction provided and use_powerlaw_2h=False")
            dl_2h = A_2h * np.interp(ell, self.lb, self.dl_2h_pred)
        

        dl_1h = self._onehalo_template_component(
            ell,
            A_1h,
            mu_1h=mu_1h,
            sigma_1h=sigma_1h,
            z_bin_index=z_bin_index,
            f_pop=f_pop,
        )
        
        # Shot noise contribution
        dl_shot = self.shot_noise_component(ell, A_shot)
        
        # Sum components
        dl_total = dl_2h + dl_1h + dl_shot
        
        # Apply astrometry damping if enabled
        if self.use_astrometry_damping and sigma_damp is not None:
            damping_factor = self.astrometry_damping_component(ell, sigma_damp)
            dl_total = dl_total * damping_factor
        
        return dl_total
        
    def model_components(self, ell, A_2h, A_1h, mu_1h, sigma_1h, A_shot, sigma_damp=None, z_bin_index=None, f_pop=None):
        """
        Get individual model components.
        
        Parameters
        ----------
        sigma_damp : float, optional
            Astrometry error in arcminutes for high-ell damping (if use_astrometry_damping=True)
        z_bin_index : int or None, optional
            Index of redshift bin for selecting linear 2H template. Required if use_linear_2h=True.
        
        Returns
        -------
        dict
            Dictionary with keys 'two_halo', 'one_halo', 'shot_noise', 'total'
            If damping enabled, also includes 'damping' and 'total_undamped'
            If use_one_halo=False, 'one_halo' will be zero array
            If use_two_halo=False, 'two_halo' will be zero array
        """
        if self.use_two_halo:
            if self.use_linear_2h:
                # Use linear 2H template for this z-bin
                if z_bin_index is None:
                    raise ValueError("z_bin_index required when use_linear_2h=True")
                if z_bin_index not in self.dl_2h_lin_per_zbin:
                    raise ValueError(f"Linear 2H template not found for z_bin_index={z_bin_index}")
                ell_lin, dl_lin = self.dl_2h_lin_per_zbin[z_bin_index]
                dl_2h = A_2h * np.interp(ell, ell_lin, dl_lin, left=0.0, right=0.0)
            elif self.use_powerlaw_2h:
                dl_2h = self.powerlaw_2h_component(ell, A_2h, self.alpha_2h_fixed)
            else:
                dl_2h = A_2h * np.interp(ell, self.lb, self.dl_2h_pred)
        else:
            dl_2h = np.zeros_like(ell)
        
        if self.use_one_halo:
            dl_1h = self._onehalo_template_component(
                ell,
                A_1h,
                mu_1h=mu_1h,
                sigma_1h=sigma_1h,
                z_bin_index=z_bin_index,
                f_pop=f_pop,
            )
        else:
            dl_1h = np.zeros_like(ell)
            
        dl_shot = self.shot_noise_component(ell, A_shot)
        
        dl_total = dl_2h + dl_1h + dl_shot
        
        components = {
            'two_halo': dl_2h,
            'one_halo': dl_1h,
            'shot_noise': dl_shot,
        }
        
        # Apply astrometry damping if enabled
        if self.use_astrometry_damping and sigma_damp is not None:
            damping_factor = self.astrometry_damping_component(ell, sigma_damp)
            components['damping'] = damping_factor
            components['total_undamped'] = dl_total
            components['total'] = dl_total * damping_factor
        else:
            components['total'] = dl_total
        
        return components
    
    def model_dl_fixed_1h_templates(self, ell, A_2h, A_1h_08, A_1h_10, A_1h_12, A_shot, 
                                     z_value, one_halo_params_dict):
        """
        Model with linear combination of fixed 1-halo templates at different slopes.
        
        Parameters
        ----------
        ell : array_like
            Multipole values
        A_2h : float
            Amplitude of two-halo term
        A_1h_08, A_1h_10, A_1h_12 : float
            Amplitudes for 1-halo templates at slopes 0.8, 1.0, 1.2
        A_shot : float
            Shot noise amplitude
        z_value : float
            Redshift for interpolating template parameters
        one_halo_params_dict : dict
            Dictionary of 1-halo parameters organized by slope
        
        Returns
        -------
        array_like
            Total D_ℓ model
        """
        # Two-halo contribution
        if self.use_powerlaw_2h:
            dl_2h = self.powerlaw_2h_component(ell, A_2h, self.alpha_2h_fixed)
        else:
            if self.dl_2h_pred is None:
                raise ValueError("No 2-halo prediction provided and use_powerlaw_2h=False")
            dl_2h = A_2h * np.interp(ell, self.lb, self.dl_2h_pred)
        
        # One-halo contribution from three fixed templates
        dl_1h = np.zeros_like(ell, dtype=float)
        for slope, amplitude in zip([0.8, 1.0, 1.2], [A_1h_08, A_1h_10, A_1h_12]):
            ln_ell_peak, sigma = interpolate_1h_params(z_value, slope, one_halo_params_dict)
            dl_1h += self.lognormal_component(ell, amplitude, ln_ell_peak, sigma)
        
        # Shot noise contribution
        dl_shot = self.shot_noise_component(ell, A_shot)
        
        return dl_2h + dl_1h + dl_shot

    
    def log_prior(self, A_2h, A_1h, mu_1h, sigma_1h, A_shot):
        """
        Calculate log-prior probability for parameters.
        
        Parameters
        ---------
        A_2h, A_1h, mu_1h, sigma_1h, A_shot : float
            Model parameters
            For log-normal: mu_1h = log(ell_peak), sigma_1h = log-width
        
        Returns
        -------
        log_prior : float
            Log of prior probability (unnormalized). Returns 0 if no priors set.
        """
        log_p = 0.0
        
        return log_p
    
    def fit_model(self, lb_data, dl_data, dl_err=None, 
                  p0=None, bounds=None, method='leastsq',
                  fit_range=None, chi2_eval_max=None, verbose=True, z_bin_index=None):
        """
        Fit the parametric model to data.
        
        Parameters
        ----------
        lb_data : array_like
            Multipole bin centers from data
        dl_data : array_like
            Measured D_ℓ values
        dl_err : array_like, optional
            Uncertainties on D_ℓ
        p0 : array_like, optional
            Initial parameter guess [A_2h, A_1h, mu_1h, sigma_1h, A_shot]
        bounds : tuple of array_like, optional
            Lower and upper bounds for parameters
        method : str, optional
            Fitting method: 'leastsq' (curve_fit) or 'minimize' (scipy.optimize.minimize)
        fit_range : tuple, optional
            (ℓ_min, ℓ_max) range to fit over
        chi2_eval_max : float, optional
            Maximum multipole for chi² evaluation (default 5000)
        verbose : bool, optional
            Print fit results
        
        Returns
        -------
        dict
            Fit results with keys 'params', 'params_err', 'chisq', 'reduced_chisq'
        """
        # Apply fit range mask
        if fit_range is not None:
            mask = (lb_data >= fit_range[0]) & (lb_data <= fit_range[1])
            lb_fit = lb_data[mask]
            dl_fit = dl_data[mask]
            dl_err_fit = dl_err[mask] if dl_err is not None else None
        else:
            lb_fit = lb_data
            dl_fit = dl_data
            dl_err_fit = dl_err
        
        # Set default initial parameters
        if p0 is None:
            # Estimate initial parameters
            A_2h_init = np.mean(dl_fit[:3]) if len(dl_fit) > 3 else 1.0
            A_1h_init = np.max(dl_fit) * 0.5
            # Use prior mean if available, otherwise default
            mu_1h_init = np.log(2500.)  # Peak around ℓ~2500 (log space)
            sigma_1h_init = 0.5  # Log-width
            A_shot_init = dl_fit[-1] / (lb_fit[-1] * (lb_fit[-1] + 1) / (2 * np.pi))
            p0 = [A_2h_init, A_1h_init, mu_1h_init, sigma_1h_init, A_shot_init]
        
        # Set default bounds (constrain to physically reasonable ranges)
        if bounds is None:
            bounds = (
                [0., 0., np.log(500), 0.1, 0.],  # Lower: peak > 500, log-width > 0.1
                [np.inf, np.inf, np.log(30000), 1.5, np.inf]  # Upper: peak < 10000, log-width < 1.5
            )
        
        # Perform fit
        if method == 'leastsq':
            try:
                # Create wrapper function to include z_bin_index parameter
                if z_bin_index is not None:
                    def model_func(ell, A_2h, A_1h, mu_1h, sigma_1h, A_shot):
                        return self.model_dl(ell, A_2h, A_1h, mu_1h, sigma_1h, A_shot, z_bin_index=z_bin_index)
                else:
                    model_func = self.model_dl
                
                popt, pcov = curve_fit(
                    model_func, lb_fit, dl_fit,
                    p0=p0, sigma=dl_err_fit, absolute_sigma=True,
                    bounds=bounds, maxfev=10000
                )
                perr = np.sqrt(np.diag(pcov))
            except Exception as e:
                if verbose:
                    print(f"Fit failed: {e}")
                    print("Returning initial parameters")
                popt = np.array(p0)
                perr = np.full_like(popt, np.nan)
        
        elif method == 'minimize':
            # Chi-squared objective with priors
            def objective(params):
                if z_bin_index is not None:
                    model = self.model_dl(lb_fit, *params, z_bin_index=z_bin_index)
                else:
                    model = self.model_dl(lb_fit, *params)
                if dl_err_fit is not None:
                    chi2 = np.sum(((dl_fit - model) / dl_err_fit)**2)
                else:
                    chi2 = np.sum((dl_fit - model)**2)
                # Add prior term (negative log-prior adds to chi-square)
                log_prior = self.log_prior(*params)
                return chi2 - 2.0 * log_prior  # -2*ln(prior) to match chi2 metric

            # Support bounds in either curve_fit-style (lb_array, ub_array)
            # or minimize-style sequence-of-(lb,ub). Convert to sequence-of-pairs if needed.
            bounds_for_minimize = bounds
            try:
                # If bounds is a tuple/list of two arrays (lower, upper), convert
                if (isinstance(bounds, (list, tuple)) and len(bounds) == 2 and
                        (len(bounds[0]) == len(p0) or isinstance(bounds[0], (list, np.ndarray)))):
                    lb_arr = np.asarray(bounds[0], dtype=float)
                    ub_arr = np.asarray(bounds[1], dtype=float)
                    bounds_for_minimize = list(zip(lb_arr.tolist(), ub_arr.tolist()))
            except Exception:
                bounds_for_minimize = bounds

            # Ensure initial guess p0 is feasible for the minimizer
            p0 = np.asarray(p0, dtype=float)
            if isinstance(bounds_for_minimize, (list, tuple)):
                # extract lower/upper for clipping
                lower_bounds = np.array([np.nan if b[0] is None else b[0] for b in bounds_for_minimize], dtype=float)
                upper_bounds = np.array([np.nan if b[1] is None else b[1] for b in bounds_for_minimize], dtype=float)
                for i in range(len(p0)):
                    if np.isfinite(lower_bounds[i]) and p0[i] < lower_bounds[i]:
                        p0[i] = lower_bounds[i] + 1e-12 * max(1.0, abs(lower_bounds[i]))
                    if np.isfinite(upper_bounds[i]) and p0[i] > upper_bounds[i]:
                        p0[i] = upper_bounds[i] - 1e-12 * max(1.0, abs(upper_bounds[i]))

            # Run minimizer with robust handling
            try:
                result = minimize(objective, p0, bounds=bounds_for_minimize, method='L-BFGS-B')
            except Exception as e:
                if verbose:
                    print(f"Minimizer failed to start: {e}")
                    print("Returning initial parameters")
                popt = np.array(p0)
                perr = np.full_like(popt, np.nan)
            else:
                popt = result.x
                if not result.success and verbose:
                    print(f"Minimizer warning: {result.message}")
                # Estimate errors from inverse Hessian where available
                try:
                    hess_inv = result.hess_inv.todense() if hasattr(result.hess_inv, 'todense') else result.hess_inv
                    perr = np.sqrt(np.abs(np.diag(hess_inv)))
                    perr = np.where(np.isfinite(perr), perr, np.nan)
                except Exception:
                    perr = np.full_like(popt, np.nan)
        
        # Compute chi-squared on restricted range if specified
        if chi2_eval_max is None:
            chi2_eval_max = self.chi2_eval_max
        # Apply both lower (300) and upper (chi2_eval_max) bounds for chi2 calculation
        chi2_mask = (lb_fit >= 300) & (lb_fit <= chi2_eval_max)
        lb_chi2 = lb_fit[chi2_mask]
        dl_chi2 = dl_fit[chi2_mask]
        if z_bin_index is not None:
            model_chi2 = self.model_dl(lb_chi2, *popt, z_bin_index=z_bin_index)
        else:
            model_chi2 = self.model_dl(lb_chi2, *popt)
        
        if dl_err_fit is not None:
            dl_err_chi2 = dl_err_fit[chi2_mask]
            chisq = np.sum(((dl_chi2 - model_chi2) / dl_err_chi2)**2)
        else:
            chisq = np.sum((dl_chi2 - model_chi2)**2)
        
        # ndof counts only the FREE parameters being fit: A_2h, A_1h, mu_1h, sigma_1h, A_shot
        # For the fixed 1h case, this should be 3 (A_2h, A_1h, A_shot)
        ndof = len(lb_chi2) - len(popt)
        reduced_chisq = chisq / ndof if ndof > 0 else np.nan
        
        if verbose:
            print("Fit Results:")
            print(f"  Model: {'Log-normal'} 1-halo term")
            print(f"  A_2h     = {popt[0]:.4f} ± {perr[0]:.4f}")
            print(f"  A_1h     = {popt[1]:.4f} ± {perr[1]:.4f}")
            mu_str = f"  mu_1h    = {popt[2]:.4f} ± {perr[2]:.4f} (ℓ_peak ~ {np.exp(popt[2]):.1f})"
            print(mu_str)
            sigma_str = f"  sigma_1h = {popt[3]:.4f} ± {perr[3]:.4f}"
            print(sigma_str)
            print(f"  A_shot   = {popt[4]:.4f} ± {perr[4]:.4f}")
            if self.use_powerlaw_2h:
                print(f"  alpha_2h = {self.alpha_2h_fixed:.2f} (fixed)")
            print(f"  χ²/dof   = {chisq:.2f}/{ndof} = {reduced_chisq:.2f} (ℓ < {chi2_eval_max})")
        
        return {
            'params': popt,
            'params_err': perr,
            'param_names': ['A_2h', 'A_1h', 'mu_1h', 'sigma_1h', 'A_shot'],
            'chisq': chisq,
            'reduced_chisq': reduced_chisq,
            'ndof': ndof,
            'chi2_eval_max': chi2_eval_max
        }
    
    def fit_model_mcmc(self, lb_data, dl_data, dl_err=None, 
                       p0=None, prior_bounds=None,
                       fit_range=None, chi2_eval_max=None, 
                       nwalkers=32, nsteps=2000, nburn=500,
                       verbose=True, progress=True, initial_guess=None,
                       z_value=None, mock_samples=None, z_bin_index=None, inst=None):
        """
        Fit the parametric log-normal model using MCMC (emcee).
        Better handles parameter degeneracies compared to least squares.
        
        Parameters
        ----------
        lb_data : array_like
            Multipole bin centers from data
        dl_data : array_like
            Measured D_ell values
        dl_err : array_like, optional
            Uncertainties on D_ell (required for MCMC)
        p0 : array_like, optional
            Initial parameter guess [A_2h, A_1h, mu_1h, sigma_1h, A_shot]
            (deprecated, use initial_guess instead)
        prior_bounds : tuple of array_like, optional
            (lower, upper) bounds for uniform priors on each parameter.
            For log-normal: [A_2h, A_1h, ln(ell_peak), sigma, A_shot]
        fit_range : tuple, optional
            (ell_min, ell_max) range to fit over
        chi2_eval_max : float, optional
            Maximum multipole for chi² evaluation
        nwalkers : int, optional
            Number of MCMC walkers (default 32)
        nsteps : int, optional
            Number of MCMC steps per walker (default 2000)
        nburn : int, optional
            Number of burn-in steps to discard (default 500)
        verbose : bool, optional
            Print fit results
        progress : bool, optional
            Show progress bar during MCMC
        initial_guess : array_like, optional
            Initial parameter values. If None, uses least squares fit.
        cov_matrix : array_like, optional
            Full covariance matrix for the data (N_data x N_data).
            If provided, uses proper correlated likelihood instead of diagonal errors.
            Takes precedence over mock_samples if both are provided.
        mock_samples : array_like, optional
            Mock samples (N_mocks x N_data) from which to compute covariance matrix.
            Used if cov_matrix is not directly provided.
            
        Returns
        -------
        dict
            Fit results with keys 'params' (medians), 'params_err' (std), 
            'samples', 'percentiles', etc.
        """
        try:
            import emcee
        except ImportError:
            raise ImportError("emcee is required for MCMC fitting. Install with: pip install emcee")
        
        if dl_err is None:
            raise ValueError("dl_err is required for MCMC fitting when no covariance provided")
    
        cfg = self._build_fit_config(z_value=z_value, inst=inst, verbose=verbose, z_bin_index=z_bin_index)

        param_names = list(cfg.fit_names)
        n_params = len(param_names)
        use_fixed_mu_sigma = cfg.fixed_mu_sigma
        use_fixed_A_2h = cfg.fixed_A2h
        use_fixed_sigma_damp = cfg.fixed_sigma_damp

        # Apply fit range mask
        if fit_range is not None:
            mask = (lb_data >= fit_range[0]) & (lb_data <= fit_range[1])
            lb_fit, dl_fit, dl_err_fit = lb_data[mask], dl_data[mask], dl_err[mask]
            dl_fit = dl_data[mask]
            dl_err_fit = dl_err[mask]
        else:
            lb_fit, dl_fit, dl_err_fit = lb_data, dl_data, dl_err
        
        lower_bounds, upper_bounds = _bounds_from_names(cfg.fit_names)
        
        print('n_params:', n_params)
        print('param_names:', param_names)
        print('lower bounds:', lower_bounds)
        print('upper bounds:', upper_bounds)
        
        if verbose:
            print("\n" + "="*60)
            print("MCMC FIT CONFIGURATION")
            print("="*60)
            print(f"Number of parameters: {n_params}")
            print(f"Parameter names: {param_names}")
            if use_fixed_mu_sigma:
                print(f"Fixed parameters: mu_1h={self.mu_1h_fixed:.3f}, sigma_1h={self.sigma_1h_fixed:.3f}")
            if use_fixed_A_2h:
                print(f"Fixed A_2h = {self.A_2h_fixed:.4f} (IGL prediction)")
            if cfg.use_onehalo_popmix:
                print("Fitting one-halo population mix fraction: f_pop")
            print(f"Prior bounds:")
            for i, name in enumerate(param_names):
                print(f"  {name}: [{lower_bounds[i]:.4g}, {upper_bounds[i]:.4g}]")
            print("="*60 + "\n")

        # Define log prior
        def log_prior(params):
            if np.all((params >= lower_bounds) & (params <= upper_bounds)):
                # Add Gaussian priors if specified (only when not fixing parameters)
                log_p = 0.0
                # 3-parameter case (use_fixed_mu_sigma): no shape priors needed
                return log_p
            return -np.inf
        
        # Define log likelihood
        def log_likelihood(params):
            params_full = self._expand_fit_params(params, cfg)
            A_2h, A_1h, mu_1h, sigma_1h, A_shot = params_full[:5]
            offset = 5
            f_pop = None
            if cfg.use_onehalo_popmix:
                f_pop = params_full[offset]
                offset += 1
            sigma_damp = params_full[offset] if cfg.use_astrometry_damping else None

            model = self.model_dl(
                lb_fit,
                A_2h,
                A_1h,
                mu_1h,
                sigma_1h,
                A_shot,
                sigma_damp=sigma_damp,
                z_bin_index=z_bin_index,
                f_pop=f_pop,
            )
            
            # Compute likelihood
            residual = dl_fit - model
            return -0.5 * np.sum((residual / dl_err_fit)**2)
        
        # Define log probability
        def log_probability(params):
            lp = log_prior(params)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(params)
        
        # Calculate prior widths for use in initialization
        prior_widths = upper_bounds - lower_bounds
        
        # Initial guess: draw from prior bounds
        if initial_guess is None:
            if p0 is not None:
                initial_guess = p0  # Backward compatibility
            else:
                # Skip LSQ fitting - initial values from prior bounds are sufficient for MCMC
                if verbose:
                    print("Drawing initial guess from prior bounds...")
                initial_guess = np.zeros(n_params)
                for i in range(n_params):
                    # Sample from middle of prior range
                    initial_guess[i] = lower_bounds[i] + 0.5 * prior_widths[i]
                
                # Validate initial guess
                if not (np.all(np.isfinite(initial_guess)) and 
                        np.all(initial_guess >= lower_bounds) and 
                        np.all(initial_guess <= upper_bounds)):
                    raise ValueError("Initial guess outside prior bounds or non-finite")
                
                if verbose:
                    print(f"Initial guess from prior: {initial_guess}")
        else:
            # User provided initial_guess - ensure it has correct dimension
            initial_guess = np.array(initial_guess)

            if len(initial_guess) != n_params:
                raise ValueError(f"Initial guess has wrong dimension: {len(initial_guess)} vs expected {n_params}")
            
            # Validate initial guess
            if len(initial_guess) != len(lower_bounds):
                raise ValueError(f"Initial guess dimension mismatch after adjustment: {len(initial_guess)} vs {len(lower_bounds)}")
            
            if not (np.all(np.isfinite(initial_guess)) and 
                    np.all(initial_guess >= lower_bounds) and 
                    np.all(initial_guess <= upper_bounds)):
                raise ValueError(f"Provided initial guess outside prior bounds or non-finite. "
                               f"initial_guess={initial_guess}, bounds=[{lower_bounds}, {upper_bounds}]")
        
        # Initialize walkers with proper spread to ensure linear independence
        prior_widths = upper_bounds - lower_bounds
        perturbation_scale = prior_widths * 0.01  # Default 1% of prior range

        if 'sigma_damp' in param_names:
            damp_idx = param_names.index('sigma_damp')
            perturbation_scale[damp_idx] = 1.0
        
        # Generate initial positions
        pos = initial_guess + perturbation_scale * np.random.randn(nwalkers, n_params)
        pos = np.clip(pos, lower_bounds + 1e-8, upper_bounds - 1e-8)
        
        # Check condition number - if too large, sample from prior instead
        try:
            pos_normalized = (pos - lower_bounds) / prior_widths
            cond = np.linalg.cond(pos_normalized)
            if cond > 1e10 or not np.isfinite(cond):
                raise ValueError(f"Large condition number: {cond}")
        except (ValueError, np.linalg.LinAlgError) as e:
            if verbose:
                print(f"Walker positions have poor conditioning ({e}), sampling from prior instead...")
            for i in range(n_params):
                pos[:, i] = np.random.uniform(lower_bounds[i], upper_bounds[i], nwalkers)
        
        # Verify walkers are not degenerate
        for i in range(n_params):
            if np.std(pos[:, i]) < 1e-10:
                if verbose:
                    print(f"Warning: Parameter {i} has very small spread, adding more noise")
                pos[:, i] = np.random.uniform(lower_bounds[i], upper_bounds[i], nwalkers)
        
        # Run MCMC
        if verbose:
            print(f"\nRunning MCMC with {nwalkers} walkers for {nsteps} steps...")
        
        sampler = emcee.EnsembleSampler(nwalkers, n_params, log_probability)
        sampler.run_mcmc(pos, nsteps, progress=progress)
        
        # Extract samples (discard burn-in)
        samples = sampler.get_chain(discard=nburn, flat=True)
        
        # Compute statistics
        params_median = np.median(samples, axis=0)
        params_std = np.std(samples, axis=0)
        params_16, params_84 = np.percentile(samples, [16, 84], axis=0)
        params_95 = np.percentile(samples, 95, axis=0)  # 95th percentile (2σ upper limit)
        params_997 = np.percentile(samples, 99.7, axis=0)  # 99.7th percentile (3σ upper limit)
        
        # Compute covariance matrix
        cov_matrix = np.cov(samples.T)
        
        def reconstruct_full_params(params_subset, zero_fixed=False):
            return self._expand_fit_params(params_subset, cfg, zero_fixed=zero_fixed)
        
        params_median_full = reconstruct_full_params(params_median)
        params_std_full = reconstruct_full_params(params_std, zero_fixed=True)
        params_16_full = reconstruct_full_params(params_16)
        params_84_full = reconstruct_full_params(params_84)
        params_95_full = reconstruct_full_params(params_95)
        params_997_full = reconstruct_full_params(params_997)
        
        # Compute chi-squared with median parameters
        if chi2_eval_max is None:
            chi2_eval_max = self.chi2_eval_max
        
        # Apply both lower (300) and upper (chi2_eval_max) bounds for chi2 calculation
        chi2_mask = (lb_fit >= 300) & (lb_fit <= chi2_eval_max)
        lb_chi2 = lb_fit[chi2_mask]
        dl_chi2 = dl_fit[chi2_mask]
        offset = 5
        f_pop_med = None
        if cfg.use_onehalo_popmix:
            f_pop_med = params_median_full[offset]
            offset += 1
        sigma_damp_med = params_median_full[offset] if cfg.use_astrometry_damping else None

        model_chi2 = self.model_dl(
            lb_chi2,
            *params_median_full[:5],
            sigma_damp=sigma_damp_med,
            z_bin_index=z_bin_index,
            f_pop=f_pop_med,
        )
        dl_err_chi2 = dl_err_fit[chi2_mask]
        
        chisq = np.sum(((dl_chi2 - model_chi2) / dl_err_chi2)**2)
        
        # ndof = number of data points - number of floated parameters
        # n_params is the count of parameters actually floated during fitting
        ndof = len(lb_chi2) - n_params
        reduced_chisq = chisq / ndof if ndof > 0 else np.nan
        
        # Compute acceptance fraction
        acceptance_fraction = np.mean(sampler.acceptance_fraction)
        
        if verbose:
            print("\nMCMC Fit Results (Parametric Model):")
            if not self.use_one_halo:
                print(f"  Model: 2-halo + shot noise only (no 1-halo term)")
            else:
                print(f"  Model: {'Log-normal'} 1-halo term")
                if use_fixed_mu_sigma:
                    print(f"  Using fixed mu_1h = {self.mu_1h_fixed:.3f}, sigma_1h = {self.sigma_1h_fixed:.3f}")
                print(f"  Acceptance fraction: {acceptance_fraction:.3f}")
            
            # Determine if 2-halo should be reported as upper limit
            # Rule: report upper limit if 16th percentile is consistent with zero (< 10% of median)
            report_A2h_upper_limit = params_16_full[0] < 0.1 * params_median_full[0]
            
            if report_A2h_upper_limit:
                print(f"  A_2h     < {params_95_full[0]:.4f} (95% upper limit)")
                print(f"           < {params_997_full[0]:.4f} (99.7% upper limit)")
                print(f"           [median = {params_median_full[0]:.4f}, 16-84%: {params_16_full[0]:.4f}-{params_84_full[0]:.4f}]")
            else:
                print(f"  A_2h     = {params_median_full[0]:.4f} ± {params_std_full[0]:.4f} [{params_16_full[0]:.4f}, {params_84_full[0]:.4f}]")
            
            if self.use_one_halo:
                print(f"  A_1h     = {params_median_full[1]:.4f} ± {params_std_full[1]:.4f} [{params_16_full[1]:.4f}, {params_84_full[1]:.4f}]")
                if use_fixed_mu_sigma:
                    print(f"  mu_1h    = {params_median_full[2]:.4f} (fixed, ℓ_peak ~ {np.exp(params_median_full[2]):.1f})")
                else:
                    print(f"  mu_1h    = {params_median_full[2]:.4f} ± {params_std_full[2]:.4f} [{params_16_full[2]:.4f}, {params_84_full[2]:.4f}] (ℓ_peak ~ {np.exp(params_median_full[2]):.1f})")
                if use_fixed_mu_sigma:
                    print(f"  sigma_1h = {params_median_full[3]:.4f} (fixed)")
                else:
                    print(f"  sigma_1h = {params_median_full[3]:.4f} ± {params_std_full[3]:.4f} [{params_16_full[3]:.4f}, {params_84_full[3]:.4f}]")
                if cfg.use_onehalo_popmix:
                    f_idx = 5
                    print(f"  f_pop    = {params_median_full[f_idx]:.4f} ± {params_std_full[f_idx]:.4f} [{params_16_full[f_idx]:.4f}, {params_84_full[f_idx]:.4f}]")
            print(f"  A_shot   = {1e7*params_median_full[4]:.4f} ± {1e7*params_std_full[4]:.4f} [{1e7*params_16_full[4]:.4f}, {1e7*params_84_full[4]:.4f}]")
            if self.use_astrometry_damping:
                damp_idx = 6 if cfg.use_onehalo_popmix else 5
                if use_fixed_sigma_damp:
                    print(f"  σ_damp   = {params_median_full[damp_idx]:.2f} arcsec (fixed)")
                else:
                    print(f"  σ_damp   = {params_median_full[damp_idx]:.2f} ± {params_std_full[damp_idx]:.2f} [{params_16_full[damp_idx]:.2f}, {params_84_full[damp_idx]:.2f}] arcsec")
            if self.use_powerlaw_2h:
                print(f"  alpha_2h = {self.alpha_2h_fixed:.2f} (fixed)")
            print(f"  χ²/dof   = {chisq:.2f}/{ndof} = {reduced_chisq:.2f} (ℓ < {chi2_eval_max})")
            
            # Guidance on reporting upper limits
            if report_A2h_upper_limit:
                print("\n  ⚠️  Consider reporting A_2h as upper limit (posterior piled up near zero)")
        
        pretty_param_names = {
            'A_2h': '$A_{2h}$',
            'A_1h': '$A_{1h}$',
            'mu_1h': r'$\mu_{1h}$',
            'sigma_1h': r'$\sigma_{1h}$',
            'A_shot': '$A_{shot}$',
            'f_pop': '$f_{pop}$',
            'sigma_damp': r'$\sigma_{\rm damp}$',
        }
        param_names_fitted = [pretty_param_names.get(name, name) for name in cfg.fit_names]
        
        # Compute best-fit model and residuals for full data range
        model_full = self.model_dl(
            lb_fit,
            *params_median_full[:5],
            sigma_damp=sigma_damp_med,
            z_bin_index=z_bin_index,
            f_pop=f_pop_med,
        )
        
        residuals = (dl_fit - model_full) / dl_err_fit  # Normalized residuals (z-scores)
        
        return {
            'params': params_median_full,  # Full parameter array (5 or 6 elements depending on damping)
            'params_err': params_std_full,
            'params_16': params_16_full,
            'params_84': params_84_full,
            'params_95': params_95_full,  # 95th percentile for 2σ upper limits
            'params_997': params_997_full,  # 99.7th percentile for 3σ upper limits
            'cov_matrix': cov_matrix,  # This is still for the fitted (3-6) parameters
            'samples': samples,  # These are the actual MCMC samples (3-6 params depending on config)
            'param_names': cfg.full_param_names,
            'samples_fitted': samples,  # Samples for only fitted parameters
            'param_names_fitted': param_names_fitted,  # Labels for only fitted parameters
            'chisq': chisq,
            'reduced_chisq': reduced_chisq,
            'ndof': ndof,
            'chi2_eval_max': chi2_eval_max,
            'acceptance_fraction': acceptance_fraction,
            'sampler': sampler,
            'use_fixed_mu_sigma': use_fixed_mu_sigma,
            'use_astrometry_damping': self.use_astrometry_damping,  # NEW: flag for damping
            'use_powerlaw_2h': self.use_powerlaw_2h,  # Model configuration for 2-halo
            'alpha_2h_fixed': self.alpha_2h_fixed,  # 2-halo power-law index
            'n_params_fit': n_params,  # Number of parameters actually fitted (4 or 5)
            # Add compatibility fields for plotting
            'z_value': z_value,
            'one_halo_params_dict': None,
            'sigma_fixed': None,
            # Best-fit model and residuals for diagnostics
            'lb_fit': lb_fit,
            'model_dl': model_full,
            'residuals': residuals,  # Normalized (data - model) / error
            'template_names': None,
            'ihl_templates': None,
            'model_wrapper': None
        }
    
    @staticmethod
    def plot_mcmc_corner(fit_result, labels=None, title=None, save_path=None, figsize=(5,5), fix_bias=False):
        """
        Plot corner plot of MCMC posterior distributions.
        Handles both IHL template fits and parametric log-normal fits.
        
        Parameters
        ----------
        fit_result : dict
            Result from fit_model_mcmc or fit_model_with_ihl_templates_mcmc containing 'samples'
        labels : list of str, optional
            Parameter labels. If None, auto-generates labels based on model type
        title : str, optional
            Plot title
        save_path : str, optional
            Path to save figure
        figsize : tuple, optional
            Figure size (width, height) in inches
            
        Returns
        -------
        figure
            Corner plot figure
        """
        try:
            import corner
        except ImportError:
            raise ImportError("corner is required for corner plots. Install with: pip install corner")
        
        # Use fitted-only samples if available (for cases with fixed parameters)
        if ('samples_fitted' in fit_result and fit_result['samples_fitted'] is not None and
            'param_names_fitted' in fit_result and fit_result['param_names_fitted'] is not None):
            samples_original = fit_result['samples_fitted'].copy()
            param_names_fitted = fit_result['param_names_fitted']
            labels = list(param_names_fitted)
            n_params = samples_original.shape[1]
            samples = samples_original.copy()

            # Find A_shot by name — it's always labelled with 'shot' in param_names_fitted.
            # This is robust regardless of whether damping is present.
            shot_idx = next(
                (i for i, p in enumerate(param_names_fitted) if 'shot' in str(p).lower()),
                None
            )
            print('shot idx here is ', shot_idx)

            if fix_bias and shot_idx is not None:
                shot_idx -= 1

                
            if shot_idx is not None:
                samples[:, shot_idx] = samples[:, shot_idx] * 1e7
                labels[shot_idx] = labels[shot_idx].replace('shot}$', 'shot} \\times 10^7$')
        else:
            # Fallback to original logic for backward compatibility
            if 'samples' not in fit_result or fit_result['samples'] is None:
                raise ValueError("No MCMC samples found in fit_result. Cannot generate corner plot.")
            samples_original = fit_result['samples'].copy()
            n_params = samples_original.shape[1]
            
            # Detect model type
            is_parametric = True  # Always assume parametric log-normal model
                        
            # Scale shot noise amplitude for better display
            samples = samples_original.copy()
            samples[:, -1] = samples[:, -1] * 1e7  # Scale shot noise by 10^7
            
            # Build labels
            if labels is None:
                param_names = fit_result.get('param_names', [f'p{i}' for i in range(n_params)])
                labels = []
                
                if is_parametric:
                    # Parametric model
                    # 5-parameter case: [A_2h, A_1h, mu_1h, sigma_1h, A_shot]
                    labels = [
                        r'$A_{\rm 2h}$',
                        r'$A_{\rm 1h}$',
                        r'$\mu_{\rm 1h}$',
                        r'$\sigma_{\rm 1h}$',
                        r'$A_{\rm shot} \times 10^7$'
                    ]
            else:
                # Generic fallback
                for i, name in enumerate(param_names):
                    if 'shot' in name.lower():
                        labels.append(r'$A_{\rm shot} \times 10^7$')
                    else:
                        labels.append(name)
        
        # Check for parameters with no dynamic range and provide manual ranges
        # Use the SCALED samples for this check to handle shot noise properly
        ranges = []
        has_fixed_param = False
        for i in range(n_params):
            param_min = np.min(samples[:, i])
            param_max = np.max(samples[:, i])
            param_range = param_max - param_min
            
            # If parameter has no variation, provide a small range around its value
            if param_range == 0 or not np.isfinite(param_range):
                has_fixed_param = True
                param_val = np.median(samples[:, i])
                if param_val == 0:
                    ranges.append((-0.01, 0.01))
                else:
                    # Use 1% of the value as range
                    delta = max(abs(param_val) * 0.01, 1e-10)
                    ranges.append((param_val - delta, param_val + delta))
            else:
                # Use 99.7% range (3-sigma equivalent)
                q = np.percentile(samples[:, i], [0.15, 99.85])
                ranges.append((q[0], q[1]))
        
        # If no parameters are fixed, let corner auto-determine all ranges
        if not has_fixed_param:
            ranges = None
        
        # Create corner plot with 3 decimal places for all parameters
        fig = corner.corner(samples, labels=labels, 
                           range=ranges,
                           quantiles=[0.16, 0.5, 0.84],
                           show_titles=True,
                           title_fmt='.2g',  # 2 significant figures
                           title_kwargs={"fontsize": 14},
                           label_kwargs={"fontsize": 20},
                           figsize=figsize)
        
        if title:
            fig.suptitle(title, y=1.05, fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Saved corner plot to {save_path}")
        
        return fig
    
    def plot_fit(self, lb_data, dl_data, dl_err, fit_result, 
                 figsize=(5, 4), xlim=[250, 1e5], ylim=None,
                 show_components=True, title=None, save_path=None, title_fs=14, 
                 ncol=1, legend_fs=11):
        """
        Plot the fitted model with data and components.
        
        Parameters
        ----------
        lb_data : array_like
            Multipole values from data
        dl_data : array_like
            Measured D_ℓ
        dl_err : array_like
            Uncertainties
        fit_result : dict
            Output from fit_model()
        figsize : tuple, optional
            Figure size
        xlim : tuple, optional
            x-axis limits
        ylim : tuple, optional
            y-axis limits
        show_components : bool, optional
            Show individual components
        title : str, optional
            Title for the plot
        save_path : str, optional
            Path to save figure
        """
        # Two-panel figure: top = data + model, bottom = per-bandpower chi
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]}, sharex=True
        )

        # Top: data and model
        ax = ax1

        if title is not None:
            ax.set_title(title, fontsize=title_fs)
        ax.errorbar(lb_data, dl_data, yerr=dl_err, fmt='o',
                    color='k', markersize=4, capsize=3,
                    label='Data', zorder=5)

        # Generate smooth model curve
        ell_model = np.logspace(0.2*np.log10(lb_data.min()),
                                2.0*np.log10(lb_data.max()), 200)

        params = fit_result['params']
        components = self.model_components(ell_model, *params)

        # Plot total model
        ax.plot(ell_model, components['total'], 'r-',
                linewidth=2.5, label='Total', zorder=4)

        # Plot components
        if show_components:
            ax.plot(ell_model, components['two_halo'], 'b-',
                    linewidth=1.5, label=r'$C_\ell^\mathrm{2h} = A_\mathrm{2h}\ell^{-2}$', alpha=0.7)
            # Label depending on 1-halo model
            one_halo_label = r'$C_\ell^\mathrm{1h}=A_\mathrm{1h}\ell^{-2}\exp\left[-\frac{(\ln \ell - \mu_\mathrm{1h})^2}{2\sigma_\mathrm{1h}^2}\right]$'
            ax.plot(ell_model, components['one_halo'], 'g-',
                    linewidth=1.5, label=one_halo_label, alpha=0.7)
            ax.plot(ell_model, components['shot_noise'], 'm--',
                    linewidth=1.5, label='Shot noise', alpha=0.7)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(r'$D_\ell$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=14)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=legend_fs, loc=4, ncol=ncol)

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        # Add fit statistics as text in top panel (include chi2 evaluation scale and key amplitudes)
        chi2_eval_max = fit_result.get('chi2_eval_max', getattr(self, 'chi2_eval_max', None))
        chisq_text = f"χ²/dof = {fit_result['chisq']:.1f}/{fit_result['ndof']} = {fit_result['reduced_chisq']:.2f}"
        if chi2_eval_max is not None:
            try:
                chi2_scale = int(chi2_eval_max)
            except Exception:
                chi2_scale = chi2_eval_max
            chisq_text = chisq_text + f" $(300<\\ell<{chi2_scale})$"

        # Key amplitudes with 3 significant digits and uncertainties if available
        try:
            A_2h_val = params[0]
            A_1h_val = params[1]
            perr = fit_result.get('params_err', None)
            if perr is not None and len(perr) >= 2 and not np.all(np.isnan(perr[:2])):
                A_2h_err = perr[0]
                A_1h_err = perr[1]
                amps_text = (
                    f"$A_{{2h}} = {A_2h_val:.3g} \\pm {A_2h_err:.3g}$\n"
                    f"$A_{{1h}} = {A_1h_val:.3g} \\pm {A_1h_err:.3g}$"
                )
            else:
                amps_text = (
                    f"$A_{{2h}} = {A_2h_val:.3g}$\n"
                    f"$A_{{1h}} = {A_1h_val:.3g}$"
                )
        except Exception:
            amps_text = ""

        full_text = chisq_text
        if amps_text:
            full_text = chisq_text + "\n" + amps_text

        ax.text(0.05, 0.95, full_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.tick_params(labelsize=12)


        # Bottom: per-bandpower chi distribution (in units of sigma)
        # Compute model at data ell and chi values
        dl_model_at_data = self.model_dl(lb_data, *params)
        dl_err_safe = np.where(np.array(dl_err) <= 0, np.nan, dl_err)
        chi_vals = (np.array(dl_data) - dl_model_at_data) / dl_err_safe

        # Plot residuals in units of sigma
        ax2.axhline(0.0, color='gray', linestyle='--', linewidth=1)
        ax2.axhline(5.0, color='gray', linestyle=':', linewidth=0.8)
        ax2.axhline(-5.0, color='gray', linestyle=':', linewidth=0.8)
        ax2.plot(lb_data, chi_vals, 'o', color='C3', markersize=4, label='(data - model)/σ')

        ax2.set_xscale('log')
        # Symmetric log scale on y with linear region for |y| <= 5
        # ax2.set_yscale('symlog', linthresh=5.0)
        ax2.set_ylabel(r'$\frac{\mathrm{data} - \mathrm{model}}{\sigma}$', fontsize=16)
        ax2.set_xlabel(r'$\ell$', fontsize=16)

        ax2.tick_params(labelsize=12)

        ax2.set_ylim(-6, 6)

        ax2.set_yticks([-5, -2.5, 0, 2.5, 5])

        ax2.grid(alpha=0.3)


        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Saved figure to {save_path}")
        
        plt.show()
        
        return fig, ax

def separate_2h_shot_from_prediction(lb, cl_pred_total, ell_fit_max=3000, ell_shot_min=5000, 
                                      alpha_2h=None, verbose=False):
    """
    Separate two-halo and shot noise components from a total prediction by fitting.
    
    Fits a power law to the low-ℓ part of the prediction (where clustering dominates)
    and estimates shot noise from the high-ℓ plateau.
    
    Parameters
    ----------
    lb : array_like
        Multipole bins
    cl_pred_total : array_like
        Total C_ℓ prediction (2h + shot)
    ell_fit_max : float, optional
        Maximum multipole for fitting 2-halo power law (default 3000)
    ell_shot_min : float, optional
        Minimum multipole for estimating shot noise level (default 5000)
    alpha_2h : float or None, optional
        Power-law index to use for 2-halo fit. If None, will fit the index (default None)
    verbose : bool, optional
        Print fit diagnostics
    
    Returns
    -------
    cl_2h : array_like
        Power-law model for two-halo C_ℓ
    cl_shot : array_like
        Constant shot noise C_ℓ
    fit_params : dict
        Dictionary with 'A_2h', 'A_shot', 'alpha_2h', and 'alpha_2h_err' (if fitted)
    """
    lb = np.asarray(lb)
    cl_pred_total = np.asarray(cl_pred_total)
    pf = lb * (lb + 1) / (2 * np.pi)
    dl_pred_total = pf * cl_pred_total
    
    # Estimate shot noise from high-ℓ plateau
    high_ell_mask = lb >= ell_shot_min
    if np.sum(high_ell_mask) > 0:
        # Assume D_ℓ ~ constant at high ℓ for shot noise
        dl_high = dl_pred_total[high_ell_mask]
        pf_high = pf[high_ell_mask]
        A_shot_est = np.median(dl_high / pf_high)
    else:
        # Fallback: use last few points
        A_shot_est = np.mean(dl_pred_total[-3:] / pf[-3:])
    
    # Fit power law to low-ℓ data (subtract shot noise first)
    low_ell_mask = lb <= ell_fit_max
    lb_fit = lb[low_ell_mask]
    dl_fit = dl_pred_total[low_ell_mask] - A_shot_est * pf[low_ell_mask]
    
    # Fit in log space: log(D_ℓ) = log(A) + alpha * log(ℓ/1000)
    log_ell_fit = np.log(lb_fit / 1000.)
    log_dl_fit = np.log(np.maximum(dl_fit, 1e-10))
    
    if alpha_2h is None:
        # Fit both amplitude and slope
        # Linear regression: y = a + b*x where y=log(D_ℓ), x=log(ℓ/1000)
        coeffs = np.polyfit(log_ell_fit, log_dl_fit, 1)
        alpha_2h_fit = coeffs[0]  # slope
        log_A_fit = coeffs[1]     # intercept
        A_2h_est = np.exp(log_A_fit)
        
        # Estimate uncertainty on slope
        residuals = log_dl_fit - (coeffs[1] + coeffs[0] * log_ell_fit)
        residual_std = np.std(residuals)
        # Simple error estimate (not rigorous but gives sense of uncertainty)
        alpha_2h_err = residual_std / np.std(log_ell_fit) / np.sqrt(len(log_ell_fit))
        
        if verbose:
            print(f"Power-law fit to mock prediction (fitted index):")
            print(f"  A_2h = {A_2h_est:.4e} (at ℓ=1000)")
            print(f"  alpha_2h = {alpha_2h_fit:.3f} ± {alpha_2h_err:.3f} (fitted)")
            print(f"  A_shot = {A_shot_est:.4e}")
            print(f"  Fit range: ℓ ∈ [{lb[0]:.0f}, {ell_fit_max:.0f}]")
            print(f"  Shot noise from: ℓ ≥ {ell_shot_min:.0f}")
    else:
        # Fixed slope, fit only amplitude
        log_A_est = np.median(log_dl_fit - alpha_2h * log_ell_fit)
        A_2h_est = np.exp(log_A_est)
        alpha_2h_fit = alpha_2h
        alpha_2h_err = 0.0
        
        if verbose:
            print(f"Power-law fit to mock prediction (fixed index):")
            print(f"  A_2h = {A_2h_est:.4e} (at ℓ=1000)")
            print(f"  alpha_2h = {alpha_2h:.2f} (fixed)")
            print(f"  A_shot = {A_shot_est:.4e}")
            print(f"  Fit range: ℓ ∈ [{lb[0]:.0f}, {ell_fit_max:.0f}]")
            print(f"  Shot noise from: ℓ ≥ {ell_shot_min:.0f}")
    
    # Generate power-law 2-halo model
    dl_2h_model = A_2h_est * (lb / 1000.)**alpha_2h_fit
    cl_2h = dl_2h_model / pf
    
    # Constant shot noise
    cl_shot = A_shot_est * np.ones_like(lb)
    
    fit_params = {
        'A_2h': A_2h_est,
        'A_shot': A_shot_est,
        'alpha_2h': alpha_2h_fit,
        'alpha_2h_err': alpha_2h_err
    }
    
    return cl_2h, cl_shot, fit_params


def _select_onehalo_template_for_zbin(dl_spectrum, z_bin_index=None, zbinedges=None, z0=0.0):
    """Select the one-halo template that matches a requested redshift bin.

    One-halo fine files can contain a spectrum for each default dz=0.2 redshift
    bin spanning 0.0 to 1.0. When the fit is for a narrower bin such as
    [0.2, 0.4], select the template whose redshift interval overlaps the fit
    interval most closely instead of rejecting the file.
    """
    if np.ndim(dl_spectrum) == 1:
        return dl_spectrum

    dl_spectrum = np.asarray(dl_spectrum)
    if dl_spectrum.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D one-halo spectrum, got shape {dl_spectrum.shape}")

    if z_bin_index is not None and z_bin_index < dl_spectrum.shape[0]:
        if zbinedges is None or len(zbinedges) < 2:
            return dl_spectrum[z_bin_index]
        requested_nbins = len(zbinedges) - 1
        if requested_nbins == dl_spectrum.shape[0]:
            return dl_spectrum[z_bin_index]

    if zbinedges is None:
        return dl_spectrum[0]

    if len(zbinedges) < 2:
        return dl_spectrum[0]

    if z_bin_index is None:
        z_bin_index = 0

    if z_bin_index >= len(zbinedges) - 1:
        return dl_spectrum[0]

    requested_zlo = float(zbinedges[z_bin_index])
    requested_zhi = float(zbinedges[z_bin_index + 1])

    template_bins = []
    z_left = float(z0)
    while z_left < 1.0:
        z_right = min(z_left + 0.2, 1.0)
        template_bins.append((z_left, z_right))
        z_left = z_right

    for idx, (tlo, thi) in enumerate(template_bins[:dl_spectrum.shape[0]]):
        if np.isclose(requested_zlo, tlo) and np.isclose(requested_zhi, thi):
            return dl_spectrum[idx]

    return dl_spectrum[0]


def resolve_onehalo_template_from_fit_result(
    fit_result,
    onehalo_output_dir=None,
    z_bin_index=None,
    inst=None,
    cat='HSC',
    use_default_if_missing=True,
    zbinedges=None,
):
    """Resolve a fixed one-halo template from fit-result metadata or a supplied directory.
    
    Parameters
    ----------
    fit_result : dict
        Saved fit result metadata
    onehalo_output_dir : str, optional
        Directory containing one-halo predictions
    z_bin_index : int, optional
        Index of the redshift bin (in the plotting context)
    inst : int, optional
        Instrument number (1 or 2)
    cat : str, optional
        Catalog name ('HSC' or 'DESILS')
    use_default_if_missing : bool, optional
        If True, fall back to defaults; if False, return None
    zbinedges : array_like, optional
        Redshift bin edges for the current plotting context.
        If provided, used to select the correct fine-mode spectrum.
    """
    if not fit_result.get('onehalo_mode', False):
        return None

    if onehalo_output_dir is None:
        onehalo_output_dir = fit_result.get('onehalo_output_dir')

    if onehalo_output_dir is None:
        if use_default_if_missing:
            onehalo_output_dir = 'data/jordan_mocks/v3/fov_10.0/onehalo_predict/'
        else:
            return None

    if not os.path.exists(onehalo_output_dir):
        if use_default_if_missing:
            return None
        return None

    if cat == 'HSC':
        bandstr_select = 'hsc_i'
        mag_cut = 25.0
    elif cat == 'DESILS':
        bandstr_select = 'sdss_z'
        mag_cut = 22.0
    else:
        bandstr_select = 'sdss_z'
        mag_cut = 22.0

    generate_type = fit_result.get('onehalo_generate_type', 'bulk')
    fsat_model = fit_result.get('onehalo_fsat_model', 'single')
    population = fit_result.get('onehalo_population', 'combined')

    if inst is None:
        inst = fit_result.get('inst', 1)

    try:
        result = load_onehalo_spectrum(
            onehalo_output_dir,
            fsat_model,
            bandstr_select,
            inst=inst,
            mag_min=18.0,
            mag_cut=mag_cut,
            z0=0.05,
            mode='Ig',
            generate_type=generate_type,
            concentration_scale=fit_result.get('onehalo_concentration_scale', 1.0),
            population=population,
        )
    except Exception as exc:
        if use_default_if_missing:
            return None
        raise RuntimeError(f"Could not load one-halo template from {onehalo_output_dir}: {exc}") from exc

    if result is None:
        return None

    use_popmix = bool(fit_result.get('onehalo_fit_popmix', False))

    if np.ndim(result['dl_spectrum']) == 1:
        return {
            'ell_arr': result['ell_arr'],
            'dl_spectrum': result['dl_spectrum'],
            'dl_spectrum_pop0': result.get('dl_spectrum_pop0', None) if use_popmix else None,
            'dl_spectrum_pop1': result.get('dl_spectrum_pop1', None) if use_popmix else None,
        }

    dl_spectrum = np.asarray(result['dl_spectrum'])
    dl_pop0 = result.get('dl_spectrum_pop0', None) if use_popmix else None
    dl_pop1 = result.get('dl_spectrum_pop1', None) if use_popmix else None

    if dl_pop0 is not None and np.ndim(dl_pop0) > 1:
        dl_pop0 = _select_onehalo_template_for_zbin(
            np.asarray(dl_pop0), z_bin_index=z_bin_index, zbinedges=zbinedges
        )
    if dl_pop1 is not None and np.ndim(dl_pop1) > 1:
        dl_pop1 = _select_onehalo_template_for_zbin(
            np.asarray(dl_pop1), z_bin_index=z_bin_index, zbinedges=zbinedges
        )

    if dl_spectrum.ndim == 1:
        return {
            'ell_arr': result['ell_arr'],
            'dl_spectrum': dl_spectrum,
            'dl_spectrum_pop0': dl_pop0,
            'dl_spectrum_pop1': dl_pop1,
        }

    if z_bin_index is None:
        return {
            'ell_arr': result['ell_arr'],
            'dl_spectrum': _select_onehalo_template_for_zbin(
                dl_spectrum, z_bin_index=None, zbinedges=zbinedges
            ),
            'dl_spectrum_pop0': dl_pop0,
            'dl_spectrum_pop1': dl_pop1,
        }
    if z_bin_index < dl_spectrum.shape[0] and zbinedges is None:
        return {
            'ell_arr': result['ell_arr'],
            'dl_spectrum': dl_spectrum[z_bin_index],
            'dl_spectrum_pop0': dl_pop0,
            'dl_spectrum_pop1': dl_pop1,
        }
    return {
        'ell_arr': result['ell_arr'],
        'dl_spectrum': _select_onehalo_template_for_zbin(
            dl_spectrum, z_bin_index=z_bin_index, zbinedges=zbinedges
        ),
        'dl_spectrum_pop0': dl_pop0,
        'dl_spectrum_pop1': dl_pop1,
    }


def _extract_plot_config(fit_result, model):
    """
    Extract PlotConfig from fit_result dictionary.
    
    Consolidates all the scattered fit_result.get() calls into one place,
    applying sensible defaults and validation.
    
    Parameters
    ----------
    fit_result : dict
        Output from fit_model_mcmc() or similar
    model : CrossPowerSpectrumModel
        Model instance for accessing defaults
    
    Returns
    -------
    PlotConfig
        Structured configuration object
    """
    return PlotConfig(
        params=fit_result['params'],
        params_err=fit_result.get('params_err', None),
        use_damping=fit_result.get('use_astrometry_damping', False),
        cov_matrix=fit_result.get('cov_matrix', None),
        samples=fit_result.get('samples', None),
        chi2_eval_max=fit_result.get('chi2_eval_max', model.chi2_eval_max),
        z_value=fit_result.get('z_value', None),
        one_halo_params_dict=fit_result.get('one_halo_params_dict', None),
        sigma_fixed=fit_result.get('sigma_fixed', None),
    )


def attach_onehalo_template_to_model(model, fit_result, z_bin_index=None, use_default_if_missing=False, zbinedges=None):
    """Attach a fixed one-halo template to the model from saved fit metadata when available.
    
    Parameters
    ----------
    model : CrossPowerSpectrumModel
        The model to attach the template to
    fit_result : dict
        Saved fit result metadata
    z_bin_index : int, optional
        Index of the redshift bin in the current context
    use_default_if_missing : bool, optional
        If True, use default paths; if False, return None on missing data
    zbinedges : array_like, optional
        Redshift bin edges for the current context.
        If provided, used to select the correct fine-mode spectrum.
    """
    if not getattr(model, 'use_one_halo', False):
        return None

    resolved_template = None
    try:
        resolved_template = resolve_onehalo_template_from_fit_result(
            fit_result,
            onehalo_output_dir=fit_result.get('onehalo_output_dir'),
            z_bin_index=z_bin_index,
            inst=fit_result.get('inst'),
            cat=fit_result.get('cat', 'HSC'),
            use_default_if_missing=use_default_if_missing,
            zbinedges=zbinedges,
        )
    except Exception:
        resolved_template = None

    if resolved_template is None:
        return None

    existing_template = getattr(model, 'onehalo_template_1h_dl', None)
    if existing_template is None:
        model.onehalo_template_1h_dl = {z_bin_index: resolved_template} if z_bin_index is not None else {None: resolved_template}
    elif isinstance(existing_template, dict):
        if z_bin_index is not None and z_bin_index not in existing_template:
            existing_template[z_bin_index] = resolved_template
        elif z_bin_index is None and None not in existing_template:
            existing_template[None] = resolved_template

    return resolved_template


def plot_fit_fixed_1h_templates(model, lb_data, dl_data, dl_err, fit_result,
                                figsize=(5, 4), xlim=[250, 1e5], ylim=None,
                                show_components=True, title=None, save_path=None, 
                                title_fs=14, ncol=1, legend_fs=11, 
                                lMax_fit=None, chi2_lim=[-5, 5], 
                                textxpos=300, textypos=50, text_fs=12, z_bin_index=None):
    """
    Plot the fitted model with fixed 1-halo templates.
    
    Parameters
    ----------
    model : CrossPowerSpectrumModel
        Model instance
    lb_data : array_like
        Multipole values from data
    dl_data : array_like
        Measured D_ℓ
    dl_err : array_like
        Uncertainties
    fit_result : dict
        Output from fit_model_fixed_1h_templates()
    figsize : tuple, optional
        Figure size
    xlim : tuple, optional
        x-axis limits
    ylim : tuple, optional
        y-axis limits
    show_components : bool, optional
        Show individual components
    title : str, optional
        Title for the plot
    save_path : str, optional
    save_path : str, optional
        Path to save figure
    floated_params : int, optional
        Number of floated parameters in the fit (used for chi2 calculation)
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]}, sharex=True
    )
    
    ax = ax1
    
    if title is not None:
        ax.set_title(title, fontsize=title_fs)
    ax.errorbar(lb_data, dl_data, yerr=dl_err, fmt='o',
                color='k', markersize=4, capsize=3,
                label='Data', zorder=5)
    
    # Generate finely sampled smooth model curve in log ell, extending slightly beyond plot limits
    # Use a wider range than xlim to ensure smooth curves outside the visible range
    if xlim is not None:
        ell_min = xlim[0] / 2.0  # Extend below lower xlim
        ell_max = xlim[1] * 2.0  # Extend above upper xlim
    else:
        ell_min = lb_data.min() / 2.0
        ell_max = lb_data.max() * 2.0
    
    # Finely sample in log-ell space (500 points for smooth curves)
    ell_model = np.logspace(np.log10(ell_min), np.log10(ell_max), 500)
    
    # Extract plot configuration from fit_result
    plot_cfg = _extract_plot_config(fit_result, model)

    # Reconstruct a fixed one-halo template from saved fit metadata when needed.
    if getattr(model, 'use_one_halo', False):
        attach_onehalo_template_to_model(
            model,
            fit_result,
            z_bin_index=z_bin_index,
            use_default_if_missing=False,
        )
    
    # Initialize uncertainty_bands to None (will be computed if errors available)
    uncertainty_bands = None
    
    mix_active = model._supports_onehalo_population_mixture(z_bin_index=z_bin_index)
    f_pop_plot = resolve_full_param_value(
        plot_cfg.params,
        plot_cfg.params,
        'f_pop',
        use_astrometry_damping=plot_cfg.use_damping,
        use_onehalo_popmix=mix_active,
        model=model,
    ) if mix_active else None
    sigma_damp_plot = resolve_full_param_value(
        plot_cfg.params,
        plot_cfg.params,
        'sigma_damp',
        use_astrometry_damping=plot_cfg.use_damping,
        use_onehalo_popmix=mix_active,
        model=model,
    ) if plot_cfg.use_damping else None

    components = model.model_components(
        ell_model,
        *plot_cfg.params[:5],
        sigma_damp=sigma_damp_plot,
        z_bin_index=z_bin_index,
        f_pop=f_pop_plot,
    )

    # Compute uncertainty bands from MCMC samples if available, otherwise from param errors
    if plot_cfg.samples is not None and len(plot_cfg.samples) > 0:
        # Use MCMC samples to compute posterior percentiles for each component
        # samples shape: (n_samples, n_fitted_params) - may not include fixed params like mu_1h, sigma_1h
        # Need to expand to full parameter space
        
        samples_arr = np.asarray(plot_cfg.samples, dtype=float)
        if samples_arr.ndim == 1:
            # Only one sample - wrap it
            samples_arr = samples_arr[np.newaxis, :]
        
        # Expand fitted-parameter samples to the full model parameter vector.
        # Saved chains often exclude fixed parameters (e.g. mu_1h, sigma_1h, sigma_damp).
        n_samples = len(samples_arr)
        n_params_full = 5 + (1 if mix_active else 0) + (1 if plot_cfg.use_damping else 0)
        n_params_fit = samples_arr.shape[1]

        if n_params_fit == n_params_full:
            samples_expanded = samples_arr
        else:
            samples_expanded = expand_fit_samples_to_full_vector(
                samples_arr,
                plot_cfg.params[:n_params_full],
                param_names_fitted=fit_result.get('param_names_fitted', None),
                use_astrometry_damping=plot_cfg.use_damping,
                use_onehalo_popmix=mix_active,
            )
        
        # Initialize arrays to store component values for each sample
        dl_2h_samples = np.zeros((n_samples, len(ell_model)))
        dl_1h_samples = np.zeros((n_samples, len(ell_model))) if model.use_one_halo else None
        dl_shot_samples = np.zeros((n_samples, len(ell_model)))
        dl_total_samples = np.zeros((n_samples, len(ell_model)))
        
        # Evaluate model components for each MCMC sample
        for i in range(n_samples):
            A_2h = samples_expanded[i, 0]
            A_1h = samples_expanded[i, 1]
            mu_1h = samples_expanded[i, 2]
            sigma_1h = samples_expanded[i, 3]
            A_shot = samples_expanded[i, 4]
            
            f_pop_i = samples_expanded[i, 5] if mix_active and samples_expanded.shape[1] > 5 else None
            if plot_cfg.use_damping:
                damp_idx = 6 if mix_active else 5
                sigma_damp = samples_expanded[i, damp_idx] if samples_expanded.shape[1] > damp_idx else None
            else:
                sigma_damp = None
            
            # Compute components for this sample
            sample_components = model.model_components(ell_model, A_2h, A_1h, mu_1h, sigma_1h, A_shot,
                                                       sigma_damp=sigma_damp, 
                                                       z_bin_index=z_bin_index,
                                                       f_pop=f_pop_i)
            
            dl_2h_samples[i] = sample_components['two_halo']
            if model.use_one_halo:
                dl_1h_samples[i] = sample_components['one_halo']
            dl_shot_samples[i] = sample_components['shot_noise']
            dl_total_samples[i] = sample_components['total']
        
        # Compute 16th and 84th percentiles for each component
        dl_2h_lower = np.percentile(dl_2h_samples, 16, axis=0)
        dl_2h_upper = np.percentile(dl_2h_samples, 84, axis=0)
        
        if model.use_one_halo:
            dl_1h_lower = np.percentile(dl_1h_samples, 16, axis=0)
            dl_1h_upper = np.percentile(dl_1h_samples, 84, axis=0)
        
        dl_shot_lower = np.percentile(dl_shot_samples, 16, axis=0)
        dl_shot_upper = np.percentile(dl_shot_samples, 84, axis=0)
        
        dl_total_lower = np.percentile(dl_total_samples, 16, axis=0)
        dl_total_upper = np.percentile(dl_total_samples, 84, axis=0)
        
        # Store uncertainty bands
        uncertainty_bands = {
            'two_halo': (dl_2h_lower, dl_2h_upper),
            'shot_noise': (dl_shot_lower, dl_shot_upper),
            'total': (dl_total_lower, dl_total_upper)
        }
        if model.use_one_halo:
            uncertainty_bands['one_halo'] = (dl_1h_lower, dl_1h_upper)
    
    elif plot_cfg.params_err is not None and not np.any(np.isnan(plot_cfg.params_err)):
        # Fallback to parameter error method if samples not available
        # Check model type for proper uncertainty calculation

        # Pure parametric MCMC case (with or without damping)
        # params[:5] are always [A_2h, A_1h, mu_1h, sigma_1h, A_shot]
        # params[5] is sigma_damp (if use_damping=True)
        
        # 2-halo bounds
        if model.use_powerlaw_2h:
            dl_2h_upper = model.powerlaw_2h_component(ell_model, plot_cfg.params[0] + plot_cfg.params_err[0], model.alpha_2h_fixed)
            dl_2h_lower = model.powerlaw_2h_component(ell_model, max(0, plot_cfg.params[0] - plot_cfg.params_err[0]), model.alpha_2h_fixed)
        else:
            # Check if using linear 2h templates
            if model.use_linear_2h and z_bin_index is not None and z_bin_index in model.dl_2h_lin_per_zbin:
                # Get linear template for this z-bin
                ell_lin, dl_2h_lin = model.dl_2h_lin_per_zbin[z_bin_index]
                # Interpolate linear template to ell_model grid (already in D_ell units)
                dl_2h_template = np.interp(ell_model, ell_lin, dl_2h_lin)
                dl_2h_upper = (plot_cfg.params[0] + plot_cfg.params_err[0]) * dl_2h_template
                dl_2h_lower = max(0, plot_cfg.params[0] - plot_cfg.params_err[0]) * dl_2h_template
            else:
                pf = ell_model * (ell_model + 1) / (2 * np.pi)
                dl_2h_upper = (plot_cfg.params[0] + plot_cfg.params_err[0]) * pf * np.interp(ell_model, model.lb, model.cl_2h_pred)
                dl_2h_lower = max(0, plot_cfg.params[0] - plot_cfg.params_err[0]) * pf * np.interp(ell_model, model.lb, model.cl_2h_pred)
        
        # 1-halo bounds (only if one-halo term is enabled)
        if model.use_one_halo:
            # Vary amplitude while keeping shape parameters at best-fit   
            dl_1h_upper = model._onehalo_template_component(
                ell_model,
                plot_cfg.params[1] + plot_cfg.params_err[1],
                mu_1h=plot_cfg.params[2],
                sigma_1h=plot_cfg.params[3],
                z_bin_index=z_bin_index,
                f_pop=f_pop_plot,
            )
            dl_1h_lower = model._onehalo_template_component(
                ell_model,
                max(0, plot_cfg.params[1] - plot_cfg.params_err[1]),
                mu_1h=plot_cfg.params[2],
                sigma_1h=plot_cfg.params[3],
                z_bin_index=z_bin_index,
                f_pop=f_pop_plot,
            )
    
        # Shot noise bounds (parameter index depends on whether one-halo is enabled)
        shot_idx = 4 if model.use_one_halo else 1
        dl_shot_upper = model.shot_noise_component(ell_model, plot_cfg.params[shot_idx] + plot_cfg.params_err[shot_idx])
        dl_shot_lower = model.shot_noise_component(ell_model, max(0, plot_cfg.params[shot_idx] - plot_cfg.params_err[shot_idx]))
        
        # For total: use proper uncertainty propagation with covariance matrix
        if plot_cfg.cov_matrix is not None:
            # Check which parameters were fixed
            use_fixed_mu_sigma = (model.mu_1h_fixed is not None and model.sigma_1h_fixed is not None)
            
            # Handle case where one-halo term is disabled
            if not model.use_one_halo:
                # 2-parameter case (or 3 with damping): [A_2h, A_shot] or [A_2h, A_shot, sigma_damp]
                n_params_no_damp = 2
                
                templates_matrix = np.zeros((len(ell_model), n_params_no_damp))
                
                # Column 0: 2-halo
                if model.use_powerlaw_2h:
                    templates_matrix[:, 0] = model.powerlaw_2h_component(ell_model, amplitude=1.0, index=model.alpha_2h_fixed)
                else:
                    # Check if using linear 2h templates
                    if model.use_linear_2h and z_bin_index is not None and z_bin_index in model.dl_2h_lin_per_zbin:
                        # Get linear template for this z-bin (already in D_ell units)
                        ell_lin, dl_2h_lin = model.dl_2h_lin_per_zbin[z_bin_index]
                        templates_matrix[:, 0] = np.interp(ell_model, ell_lin, dl_2h_lin)
                    else:
                        pf = ell_model * (ell_model + 1) / (2 * np.pi)
                        templates_matrix[:, 0] = pf * np.interp(ell_model, model.lb, model.cl_2h_pred)
                
                # Column 1: shot noise
                templates_matrix[:, 1] = model.shot_noise_component(ell_model, amplitude=1.0)
                
                # Uncertainty at each ℓ: σ²(ℓ) = T(ℓ)^T * Cov * T(ℓ)
                # Note: cov_matrix is 2x2 for [A_2h, A_shot] (damping handled separately if needed)
                total_var = np.sum((templates_matrix @ plot_cfg.cov_matrix[:2, :2]) * templates_matrix, axis=1)
                total_std = np.sqrt(np.maximum(0, total_var))
                
                # Apply damping if enabled
                if plot_cfg.use_damping:
                    # Get undamped total
                    dl_total_undamped = components.get('total_undamped')
                    if dl_total_undamped is None:
                        dl_total_undamped = components['two_halo'] + components['shot_noise']
                    
                    sigma_damp = resolve_full_param_value(
                        plot_cfg.params,
                        plot_cfg.params,
                        'sigma_damp',
                        use_astrometry_damping=plot_cfg.use_damping,
                        use_onehalo_popmix=mix_active,
                        model=model,
                    )
                    damping_factor = model.astrometry_damping_component(ell_model, sigma_damp)
                    
                    dl_total_upper = (dl_total_undamped + total_std) * damping_factor
                    dl_total_lower = np.maximum(0, (dl_total_undamped - total_std) * damping_factor)
                else:
                    dl_total_upper = components['total'] + total_std
                    dl_total_lower = np.maximum(0, components['total'] - total_std)
                
                # Store uncertainty bands for no-one-halo case (already computed dl_2h and dl_shot)
                uncertainty_bands = {
                    'two_halo': (dl_2h_lower, dl_2h_upper),
                    'shot_noise': (dl_shot_lower, dl_shot_upper),
                    'total': (dl_total_lower, dl_total_upper)
                }
            
            else:
                # One-halo term is enabled - build 5-column template matrix
                templates_matrix = np.zeros((len(ell_model), 5))
                
                # Column 0: 2-halo
                if model.use_powerlaw_2h:
                    templates_matrix[:, 0] = model.powerlaw_2h_component(ell_model, amplitude=1.0, index=model.alpha_2h_fixed)
                else:
                    # Check if using linear 2h templates
                    if model.use_linear_2h and z_bin_index is not None and z_bin_index in model.dl_2h_lin_per_zbin:
                        # Get linear template for this z-bin (already in D_ell units)
                        ell_lin, dl_2h_lin = model.dl_2h_lin_per_zbin[z_bin_index]
                        templates_matrix[:, 0] = np.interp(ell_model, ell_lin, dl_2h_lin)
                    else:
                        pf = ell_model * (ell_model + 1) / (2 * np.pi)
                        templates_matrix[:, 0] = pf * np.interp(ell_model, model.lb, model.cl_2h_pred)
                
                # Columns 1-3: 1-halo partial derivatives
                # For A_1h (column 1): just the shape function
                templates_matrix[:, 1] = model.lognormal_component(ell_model, amplitude=1.0, mu=plot_cfg.params[2], sigma=plot_cfg.params[3])
                
                if use_fixed_mu_sigma:
                    # 3-parameter case: cov_matrix is 3x3 for [A_2h, A_1h, A_shot]
                    # Columns 2 and 3 (mu_1h, sigma_1h): zero since they're fixed
                    templates_matrix[:, 2] = 0.0
                    templates_matrix[:, 3] = 0.0
                    
                    # Column 4: shot noise
                    templates_matrix[:, 4] = model.shot_noise_component(ell_model, amplitude=1.0)
                    
                    # Build expanded 5x5 covariance with zeros for fixed parameters
                    cov_matrix_expanded = np.zeros((5, 5))
                    # Map 3-parameter cov to 5-parameter: [A_2h, A_1h, A_shot] -> [A_2h, A_1h, 0, 0, A_shot]
                    cov_matrix_expanded[0, 0] = plot_cfg.cov_matrix[0, 0]  # A_2h variance
                    cov_matrix_expanded[0, 1] = cov_matrix_expanded[1, 0] = plot_cfg.cov_matrix[0, 1]  # A_2h-A_1h
                    cov_matrix_expanded[1, 1] = plot_cfg.cov_matrix[1, 1]  # A_1h variance
                    # Rows/cols 2, 3 (mu_1h, sigma_1h) stay zero
                    cov_matrix_expanded[0, 4] = cov_matrix_expanded[4, 0] = plot_cfg.cov_matrix[0, 2]  # A_2h-A_shot
                    cov_matrix_expanded[1, 4] = cov_matrix_expanded[4, 1] = plot_cfg.cov_matrix[1, 2]  # A_1h-A_shot
                    cov_matrix_expanded[4, 4] = plot_cfg.cov_matrix[2, 2]  # A_shot variance
                    
                    total_var = np.sum((templates_matrix @ cov_matrix_expanded) * templates_matrix, axis=1)
                else:
                    # 5-parameter case: normal calculation
                    # For mu_1h and sigma_1h: numerical derivatives (small perturbation)
                    delta_mu = 0.01 * plot_cfg.params[2] if plot_cfg.params[2] != 0 else 0.01
                    delta_sigma = 0.01 * plot_cfg.params[3] if plot_cfg.params[3] != 0 else 0.01
                    
                    templates_matrix[:, 2] = (model.lognormal_component(ell_model, plot_cfg.params[1], plot_cfg.params[2] + delta_mu, plot_cfg.params[3]) - 
                                                model.lognormal_component(ell_model, plot_cfg.params[1], plot_cfg.params[2], plot_cfg.params[3])) / delta_mu
                    templates_matrix[:, 3] = (model.lognormal_component(ell_model, plot_cfg.params[1], plot_cfg.params[2], plot_cfg.params[3] + delta_sigma) - 
                                                model.lognormal_component(ell_model, plot_cfg.params[1], plot_cfg.params[2], plot_cfg.params[3])) / delta_sigma
                
                    # Column 4: shot noise
                    templates_matrix[:, 4] = model.shot_noise_component(ell_model, amplitude=1.0)

                    # Uncertainty at each ℓ: σ²(ℓ) = T(ℓ)^T * Cov * T(ℓ)
                    # With astrometric damping enabled, covariance may include an extra
                    # sigma_damp parameter (6x6), while templates_matrix remains 5 columns.
                    # Use the core covariance block for [A_2h, A_1h, mu_1h, sigma_1h, A_shot].
                    if plot_cfg.use_damping and plot_cfg.cov_matrix.shape[0] >= 6:
                        cov_core = plot_cfg.cov_matrix[:5, :5]
                    else:
                        cov_core = plot_cfg.cov_matrix
                    total_var = np.sum((templates_matrix @ cov_core) * templates_matrix, axis=1)
                
                total_std = np.sqrt(np.maximum(0, total_var))
                
                # If damping is enabled, apply it to the total uncertainty bounds
                # Note: components['total'] is already damped, so we need to apply damping to the undamped bounds
                if plot_cfg.use_damping:
                    # Get undamped total from components
                    dl_total_undamped = components.get('total_undamped')
                    if dl_total_undamped is None:
                        # If not available, reconstruct it
                        dl_total_undamped = components['two_halo'] + components['one_halo'] + components['shot_noise']
                    
                    # Apply damping to the uncertainty bounds
                    sigma_damp = sigma_damp_plot
                    damping_factor = model.astrometry_damping_component(ell_model, sigma_damp)
                    
                    dl_total_upper = (dl_total_undamped + total_std) * damping_factor
                    dl_total_lower = np.maximum(0, (dl_total_undamped - total_std) * damping_factor)
                else:
                    dl_total_upper = components['total'] + total_std
                    dl_total_lower = np.maximum(0, components['total'] - total_std)
                
                # Store uncertainty bands for one-halo enabled case
                uncertainty_bands = {
                    'two_halo': (dl_2h_lower, dl_2h_upper),
                    'one_halo': (dl_1h_lower, dl_1h_upper),
                    'shot_noise': (dl_shot_lower, dl_shot_upper),
                    'total': (dl_total_lower, dl_total_upper)
                }
        else:
            uncertainty_bands = None
            
    else:
        uncertainty_bands = None
    
    # Plot total model with uncertainty band
    ax.plot(ell_model, components['total'], 'r-',
            linewidth=2.5, label='Total', zorder=4)
    
    if uncertainty_bands is not None:
        ax.fill_between(ell_model, uncertainty_bands['total'][0], uncertainty_bands['total'][1],
                        color='red', alpha=0.2, zorder=3)
    
    # Plot components with uncertainty bands
    if show_components:
        # 2-halo component
        ax.plot(ell_model, components['two_halo'], 'b-',
                linewidth=1.5, label='2-halo', alpha=0.7)
        if uncertainty_bands is not None:
            ax.fill_between(ell_model, uncertainty_bands['two_halo'][0], uncertainty_bands['two_halo'][1],
                            color='blue', alpha=0.15, zorder=1)
        
        if 'one_halo' in components:
            # Pure parametric MCMC case (log-normal 1-halo) - works with or without damping
            ax.plot(ell_model, components['one_halo'], 'g-',
                    linewidth=1.5, label='1-halo', alpha=0.7)
            if uncertainty_bands is not None and 'one_halo' in uncertainty_bands:
                ax.fill_between(ell_model, uncertainty_bands['one_halo'][0], uncertainty_bands['one_halo'][1],
                                color='green', alpha=0.15, zorder=1)
        
        # Shot noise component
        ax.plot(ell_model, components['shot_noise'], 'm--',
                linewidth=1.5, label='Shot noise', alpha=0.7)
        if uncertainty_bands is not None:
            ax.fill_between(ell_model, uncertainty_bands['shot_noise'][0], uncertainty_bands['shot_noise'][1],
                            color='magenta', alpha=0.15, zorder=1)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$D_\ell$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=legend_fs, loc=4, ncol=ncol)
    
    # Shade region beyond lMax_fit (not included in fit) if lMax_fit is provided
    if lMax_fit is not None and xlim is not None:
        ax.axvspan(lMax_fit, xlim[1], color='lightgray', alpha=0.3, zorder=0, 
                   label='Not fitted' if lMax_fit < xlim[1] else None)
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Add fit statistics

    chisq_text = f"χ²/dof = {fit_result['chisq']:.1f}/{fit_result['ndof']} = {fit_result['reduced_chisq']:.2f}"
    if plot_cfg.chi2_eval_max is not None:
        chisq_text += f" $(304<\\ell<{int(plot_cfg.chi2_eval_max)})$"
            
    # Add chi2/dof and ell range in top left corner
    chi2_reduced = fit_result['reduced_chisq']
    chi2_value = fit_result['chisq']
    ndof = fit_result['ndof']
    print('ndof:', ndof)
    top_left_text = f"$\\chi^2 / N_{{dof}} = {chi2_value:.1f}/{ndof} = {chi2_reduced:.2f}$\n$304 < \\ell < {lMax_fit}$"
    ax.text(textxpos, textypos, top_left_text, fontsize=text_fs,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Bottom panel: per-bandpower chi residuals
    # Generate model at data points for chi residuals

    if len(plot_cfg.params) >= 5:
        model_data = model.model_dl(
            lb_data,
            *plot_cfg.params[:5],
            sigma_damp=sigma_damp_plot,
            z_bin_index=z_bin_index,
            f_pop=f_pop_plot,
        )

    chi_vals = (dl_data - model_data) / dl_err if dl_err is not None else (dl_data - model_data)
    
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.plot(lb_data, chi_vals, 'ko', markersize=4)
    ax2.set_xlabel(r'$\ell$', fontsize=14)
    ax2.set_ylabel(r'$\chi$', fontsize=12)
    ax2.set_xscale('log')
    ax2.set_ylim(chi2_lim)
    ax2.grid(alpha=0.3)
    
    # Shade region beyond lMax_fit (not included in fit) if lMax_fit is provided
    if lMax_fit is not None and xlim is not None:
        ax2.axvspan(lMax_fit, xlim[1], color='lightgray', alpha=0.3, zorder=0)

    ax2.axhspan(-1, 1, color='green', alpha=0.1)
    ax2.axhspan(-3, 3, color='yellow', alpha=0.1)
    
    if xlim is not None:
        ax2.set_xlim(xlim)
    
    # plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig, (ax1, ax2)


def run_gal_auto_fits(inst_list=[1, 2], cat='DESILS', 
                      startidx=2, endidx=-1, zbinedges=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                      lams=[1.1, 1.8], alpha_from_mock=0.0,
                      chi2_eval_max=10000., lMax_fit=80000, fitstr='gal_auto',
                      figbasedir='figures/gal_auto_fits/', save_figs=True, save_results=False,
                      file_fpath=None, ihl_1h_params_path='ihl_1h_params.npz',
                      nwalkers=32, nsteps=4000, nburn=1000, prior_bounds=None,
                      chi2_lim=[-20, 5], headstr='hsc_ilt24.0', ifield_list=[4, 5, 6, 7, 8],
                      use_iterative_knox=False, fmask=0.67,
                      use_astrometry_damping=False,
                      use_one_halo=True,
                      mu_1h_fixed_override=None,
                      sigma_1h_fixed_override=None,
                      mu_1h_fixed_default=8.0,
                      sigma_1h_fixed_default=0.7,
                      gal_ps_dict=None):
    """
    Run galaxy auto-spectrum fits using parametric model with fixed IHL-derived 1h shape.
    
    This function fits galaxy auto-spectra (C_ℓ^gg) with a 2h+1h+shot model where the
    1h shape parameters (μ, σ) are fixed from IHL decomposition. Only the amplitudes
    (A_2h, A_1h, A_shot) are fit.
    
    Parameters
    ----------
    inst_list : list, optional
        CIBER instrument list (1=1.1um, 2=1.8um). Used for matching with cross-spectra.
    cat : str, optional
        Catalog name ('DESILS' or 'HSC')
    startidx : int, optional
        Starting index for fit range
    endidx : int, optional
        Ending index for fit range
    zbinedges : array_like, optional
        Redshift bin edges
    lams : list, optional
        Wavelengths in microns for each instrument (for labeling)
    alpha_from_mock : float, optional
        Fixed power-law index for 2-halo term (default 0.0)
    chi2_eval_max : float, optional
        Maximum multipole for chi-squared evaluation
    lMax_fit : float, optional
        Maximum multipole for fitting
    fitstr : str, optional
        Fit string identifier for output filenames
    figbasedir : str, optional
        Base directory for saving figures
    save_figs : bool, optional
        Whether to save figures
    save_results : bool, optional
        Whether to save fit results
    file_fpath : str, optional
        File path for saved results
    ihl_1h_params_path : str, optional
        Path to IHL one-halo parameters file
    nwalkers : int, optional
        Number of MCMC walkers
    nsteps : int, optional
        Number of MCMC steps
    nburn : int, optional
        Number of burn-in steps
    prior_bounds : tuple, optional
        (lower, upper) bounds for uniform priors
    chi2_lim : list, optional
        y-axis limits for chi residuals plot
    headstr : str, optional
        Header string for catalog files (e.g., 'hsc_ilt24.0')
    ifield_list : list, optional
        List of field indices to use
    use_iterative_knox : bool, optional
        If True, use model-based iterative Knox covariance (recommended for auto-spectra).
        If False, use standard Knox errors computed from measured spectrum (default False).
        See ITERATIVE_KNOX_GUIDE.md for details.
    fmask : float, optional
        Mask fraction per field (default 0.67). Only used if use_iterative_knox=True.
    
    Returns
    -------
    dict
        Dictionary of fit results for each instrument and redshift bin
        
    Notes
    -----
    This function assumes:
    - IHL-derived 1h shape parameters are preferred and used as fixed values when available
    - If IHL 1h params are unavailable, fixed approximate defaults are used
    - Only amplitude parameters are free: A_2h, A_1h, A_shot
    - Astrometry damping is optional via use_astrometry_damping
    
    When use_iterative_knox=True:
    - Knox cosmic variance is computed from the model at each MCMC step
    - Avoids bias from cosmic variance in the measured spectrum
    - Requires fsky and delta_ell to be computed from field geometry
    - Recommended for galaxy auto-spectra where Knox errors are significant
    """
    
    if save_results and file_fpath is None:
        file_fpath = 'gal_auto_fits_'+cat+'_coarsez_'+fitstr+'.npz'
    
    # Load IHL one-halo parameters when available; otherwise use fixed approximations.
    # Explicit overrides take precedence over both IHL-derived and default values.
    import os
    if not use_one_halo:
        ihl_1h_params = None
        using_approx_fixed_1h = False
    elif os.path.exists(ihl_1h_params_path):
        ihl_1h_params = load_ihl_1h_params(ihl_1h_params_path)
        using_approx_fixed_1h = False
    else:
        ihl_1h_params = None
        using_approx_fixed_1h = True
        print(f"\nWarning: IHL 1h parameters file not found: {ihl_1h_params_path}")
        print(
            f"Using approximate fixed 1h shape defaults: "
            f"mu_1h={mu_1h_fixed_default:.3f}, sigma_1h={sigma_1h_fixed_default:.3f}"
        )

    print("\n" + "="*70)
    if not use_one_halo:
        print("Using no one-halo term for galaxy auto fits (2h + shot only)")
    elif using_approx_fixed_1h:
        print("Using approximate fixed one-halo parameters for galaxy auto fits")
    else:
        print("Using IHL-derived one-halo parameters for galaxy auto fits")
    if use_iterative_knox:
        print("Using iterative Knox covariance (model-based)")
    else:
        print("Using standard Knox covariance (data-based)")
    if use_astrometry_damping:
        print("Using astrometry damping term")
    if use_one_halo and mu_1h_fixed_override is not None and sigma_1h_fixed_override is not None:
        print(
            f"Using explicit fixed one-halo override: "
            f"mu_1h={mu_1h_fixed_override:.3f}, sigma_1h={sigma_1h_fixed_override:.3f}"
        )
    print("="*70)

    # Compute Knox parameters if using iterative Knox
    if use_iterative_knox:
        nfield = len(ifield_list)
        field_area = 2.0 * 2.0  # deg^2 per field (2° × 2°)
        full_sky = 41253.0  # deg^2
        fsky = fmask * (field_area / full_sky) * nfield
        print(f"\nKnox parameters:")
        print(f"  Number of fields: {nfield}")
        print(f"  Mask fraction per field: {fmask:.3f}")
        print(f"  Effective sky fraction: {fsky:.6f}")

    # Load galaxy auto-spectra
    if gal_ps_dict is not None:
        lb = np.asarray(gal_ps_dict['lb'], dtype=float)
        full_cl_gal = np.asarray(gal_ps_dict['full_cl_gal'], dtype=float)
        full_clerr_gal = np.asarray(gal_ps_dict['full_clerr_gal'], dtype=float)
        print("Using provided galaxy auto power spectra (gal_ps_dict override)")
    else:
        from ciber.plotting.gal_plotting_fns import collect_ciber_gal_vs_redshift

        subtract_randoms = True
        maskstr = 'wFFerr' if cat == 'HSC' else 'JHlt16_wFFerr'
        catname = 'LS' if cat == 'DESILS' else cat

        print('headstr before collect_ciber_gal_vs_redshift is ', headstr)
        res_ps = collect_ciber_gal_vs_redshift(
            catname,
            subtract_randoms=subtract_randoms,
            ifield_list=ifield_list,
            inst_list=inst_list,
            zbinedges=zbinedges,
            maskstr=maskstr,
            subtract_sn=False,
            tl_pix_correct=False,
            headstr=headstr,
            with_ff_err=False,
        )

        lb = res_ps['lb']
        full_cl_gal = res_ps['full_cl_gal']
        full_clerr_gal = res_ps['full_clerr_gal']
    pf_data = lb * (lb + 1) / (2 * np.pi)

    all_fit_results_mcmc = {}
    nzbin = len(zbinedges) - 1

    for zidx in range(nzbin):
        zcen = 0.5 * (zbinedges[zidx] + zbinedges[zidx + 1])
        print("\n" + "=" * 70)
        print(f"Fitting redshift bin {zidx}: {zbinedges[zidx]} < z < {zbinedges[zidx+1]}")
        print("=" * 70)

        for idx, inst in enumerate(inst_list):
            if not use_one_halo:
                mu_1h_fixed = None
                sigma_1h_fixed = None
                print("Disabling one-halo term for this fit")
            elif mu_1h_fixed_override is not None and sigma_1h_fixed_override is not None:
                mu_1h_fixed = float(mu_1h_fixed_override)
                sigma_1h_fixed = float(sigma_1h_fixed_override)
                print(
                    f"Fixing 1h shape from override: mu_1h={mu_1h_fixed:.3f}, "
                    f"sigma_1h={sigma_1h_fixed:.3f}"
                )
            elif using_approx_fixed_1h:
                mu_1h_fixed = mu_1h_fixed_default
                sigma_1h_fixed = sigma_1h_fixed_default
                print(f"Fixing 1h shape: mu_1h={mu_1h_fixed:.3f}, sigma_1h={sigma_1h_fixed:.3f}")
            else:
                slope = ihl_1h_params['slopes'][0]
                if (zidx, slope) not in ihl_1h_params['params']:
                    print(f"Warning: No IHL parameters for zidx={zidx}, slope={slope}. Using defaults.")
                    mu_1h_fixed = mu_1h_fixed_default
                    sigma_1h_fixed = sigma_1h_fixed_default
                else:
                    mu_1h_fixed = ihl_1h_params['params'][(zidx, slope)]['mu_1h']
                    sigma_1h_fixed = ihl_1h_params['params'][(zidx, slope)]['sigma_1h']
                print(f"Fixing 1h shape: mu_1h={mu_1h_fixed:.3f}, sigma_1h={sigma_1h_fixed:.3f}")

            model = CrossPowerSpectrumModel(
                lb,
                use_powerlaw_2h=True,
                alpha_2h_fixed=alpha_from_mock,
                chi2_eval_max=chi2_eval_max,
                mu_1h_fixed=mu_1h_fixed,
                sigma_1h_fixed=sigma_1h_fixed,
                use_astrometry_damping=use_astrometry_damping,
                use_one_halo=use_one_halo,
            )

            dl_data = pf_data * full_cl_gal[idx, zidx]
            dlerr_data = pf_data * full_clerr_gal[idx, zidx]

            title = f'Galaxy Auto {cat} {lams[idx]} $\\mu$m'
            title += f', {zbinedges[zidx]}<z<{zbinedges[zidx+1]}'

            if use_iterative_knox:
                from ciber.core.powerspec_pipeline import CIBER_PS_pipeline

                cbps = CIBER_PS_pipeline()
                delta_ell = cbps.Mkk_obj.delta_ell
                if np.ndim(delta_ell) > 0:
                    delta_ell_sliced = delta_ell[startidx:endidx]
                else:
                    delta_ell_sliced = delta_ell

                print(f"  Bandpower width: {delta_ell_sliced if np.ndim(delta_ell_sliced) == 0 else 'array'}")
                fit_result_mcmc = model.fit_model_mcmc_iterative_knox(
                    lb[startidx:endidx],
                    dl_data[startidx:endidx],
                    fit_range=[300, lMax_fit],
                    chi2_eval_max=lMax_fit,
                    nwalkers=nwalkers,
                    nsteps=nsteps,
                    nburn=nburn,
                    progress=True,
                    verbose=True,
                    prior_bounds=prior_bounds,
                    z_value=zcen,
                    initial_guess=None,
                    fsky=fsky,
                    delta_ell=delta_ell_sliced,
                    measurement_cov=None,
                )
            else:
                fit_result_mcmc = model.fit_model_mcmc(
                    lb[startidx:endidx],
                    dl_data[startidx:endidx],
                    dl_err=dlerr_data[startidx:endidx],
                    fit_range=[300, lMax_fit],
                    chi2_eval_max=lMax_fit,
                    nwalkers=nwalkers,
                    nsteps=nsteps,
                    nburn=nburn,
                    progress=True,
                    verbose=True,
                    prior_bounds=prior_bounds,
                    z_value=zcen,
                    initial_guess=None,
                )

            # Corner plot
            fig = CrossPowerSpectrumModel.plot_mcmc_corner(
                fit_result_mcmc,
                title=title, 
                figsize=(5, 5)
            )
            plt.show()

            # Power spectrum plot
            fig_mcmc, ax = plot_fit_fixed_1h_templates(
                model, 
                lb[startidx:endidx], 
                dl_data[startidx:endidx], 
                dlerr_data[startidx:endidx],
                fit_result_mcmc, 
                save_path=None, 
                ylim=[1e-3, 5e2], 
                figsize=(6, 6), 
                title=title, 
                title_fs=16, 
                textxpos=350,
                lMax_fit=lMax_fit, 
                chi2_lim=chi2_lim
            )

            if save_figs:
                os.makedirs(figbasedir+cat+'_coarsez/corner/', exist_ok=True)
                os.makedirs(figbasedir+cat+'_coarsez/ps_fits/', exist_ok=True)
                
                fig.savefig(figbasedir+cat+'_coarsez/corner/gal_auto_fit_corner_'+str(zbinedges[zidx])+'_z_'+str(zbinedges[zidx+1])+'_TM'+str(inst)+'_'+cat+'_'+fitstr+'_lMaxfit='+str(lMax_fit)+'.png', bbox_inches='tight', dpi=300)
                fig_mcmc.savefig(figbasedir+cat+'_coarsez/ps_fits/gal_auto_fit_'+str(zbinedges[zidx])+'_z_'+str(zbinedges[zidx+1])+'_TM'+str(inst)+'_'+cat+'_'+fitstr+'_lMaxfit='+str(lMax_fit)+'.png', bbox_inches='tight', dpi=300) 

            key = f'inst{inst}_zbin{zidx}'

            all_fit_results_mcmc[key] = {
                'fit_result': fit_result_mcmc,
                'inst': inst,
                'zidx': zidx,
                'zcen': zcen
            }

    if save_results:
        save_fpath = 'data/gal_auto_fits/'+file_fpath
        os.makedirs(os.path.dirname(save_fpath), exist_ok=True)

        print('saving results to ', save_fpath, '..')
        save_fit_results_npz(all_fit_results_mcmc, zbinedges, inst_list, save_fpath, dataset_name=cat)

    return all_fit_results_mcmc


def run_gal_auto_fits_two_stage(inst_list=[1, 2], cat='DESILS',
                                 startidx=2, endidx=-1, zbinedges=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                 lams=[1.1, 1.8], alpha_from_mock=0.0,
                                 chi2_eval_max=10000., lMax_fit=80000, fitstr='gal_auto_2stage',
                                 figbasedir='figures/gal_auto_fits/', save_figs=True, save_results=False,
                                 file_fpath=None, ihl_1h_params_path='ihl_1h_params.npz',
                                 nwalkers=32, nsteps_stage1=2000, nsteps_stage2=4000,
                                 nburn_stage1=500, nburn_stage2=1000, prior_bounds=None,
                                 chi2_lim=[-20, 5], headstr='hsc_ilt25.0', ifield_list=[4, 5, 6, 7, 8],
                                 fmask=0.67, gal_ps_dict=None):
    """
    Run galaxy auto-spectrum fits using a two-stage approach with model-based Knox uncertainties.
    
    This implements a simplified two-stage procedure:
    
    Stage 1: Fit with data-based Knox errors
    -----------------------------------------
    Use the measured spectrum to compute Knox sample variance. This gives
    an initial estimate of the model parameters and signal level.
    
    Stage 2: Refit with FIXED model-based Knox errors from Stage 1
    ---------------------------------------------------------------
    Use the Stage 1 best-fit model to compute a FIXED Knox covariance for
    Stage 2. This avoids bias from cosmic variance in the measured spectrum.
    The Knox errors remain constant throughout Stage 2 MCMC (unlike the
    iterative Knox method which updates at every step).
    
    This approach is simpler and more stable than iterative Knox covariance.
    
    Parameters
    ----------
    inst_list : list, optional
        CIBER instrument list (1=1.1um, 2=1.8um)
    cat : str, optional
        Catalog name ('DESILS' or 'HSC')
    startidx : int, optional
        Starting index for fit range
    endidx : int, optional
        Ending index for fit range
    zbinedges : array_like, optional
        Redshift bin edges
    lams : list, optional
        Wavelengths in microns for each instrument
    alpha_from_mock : float, optional
        Fixed power-law index for 2-halo term
    chi2_eval_max : float, optional
        Maximum multipole for chi-squared evaluation
    lMax_fit : float, optional
        Maximum multipole for fitting
    fitstr : str, optional
        Fit string identifier for output filenames
    figbasedir : str, optional
        Base directory for saving figures
    save_figs : bool, optional
        Whether to save figures
    save_results : bool, optional
        Whether to save fit results
    file_fpath : str, optional
        File path for saved results
    ihl_1h_params_path : str, optional
        Path to IHL one-halo parameters file
    nwalkers : int, optional
        Number of MCMC walkers for both stages
    nsteps_stage1 : int, optional
        Number of MCMC steps for Stage 1 (default 2000)
    nsteps_stage2 : int, optional
        Number of MCMC steps for Stage 2 (default 4000)
    nburn_stage1 : int, optional
        Number of burn-in steps for Stage 1 (default 500)
    nburn_stage2 : int, optional
        Number of burn-in steps for Stage 2 (default 1000)
    prior_bounds : tuple, optional
        (lower, upper) bounds for uniform priors
    chi2_lim : list, optional
        y-axis limits for chi residuals plot
    headstr : str, optional
        Header string for catalog files
    ifield_list : list, optional
        List of field indices to use
    fmask : float, optional
        Mask fraction per field (default 0.67)
    
    Returns
    -------
    dict
        Dictionary with Stage 1 and Stage 2 fit results for each instrument and redshift bin.
        Keys: 'inst{inst}_zbin{zidx}_stage1', 'inst{inst}_zbin{zidx}_stage2'
        
    Notes
    -----
    The two-stage approach ensures that:
    1. Stage 1 provides good initial parameter estimates using data-based Knox
    2. Stage 2 uses FIXED Knox uncertainties from Stage 1 best-fit model
    3. Measurement uncertainties (from noise, mode scatter) remain consistent between stages
    4. Only the Knox cosmic variance component is updated (once, from Stage 1 to Stage 2)
    
    Comparison with iterative Knox (fit_model_mcmc_iterative_knox):
    - Two-stage: Knox computed ONCE from Stage 1 model, then held fixed in Stage 2
    - Iterative: Knox recomputed at EVERY MCMC step from current model
    - Two-stage is simpler and more stable; iterative is more self-consistent
    - Both avoid bias from data cosmic variance; results should be similar
    
    See Also
    --------
    run_gal_auto_fits : Single-stage fitting (with option for iterative Knox)
    fit_model_mcmc_iterative_knox : MCMC fitting with model-based Knox covariance
    """
    

    inst_list = [1] # temporary
    catname = 'LS' if cat == 'DESILS' else cat

    if save_results and file_fpath is None:
        file_fpath = 'gal_auto_fits_2stage_'+catname+'_coarsez_'+fitstr+'.npz'

    # Load IHL one-halo parameters
    import os
    if not os.path.exists(ihl_1h_params_path):
        raise FileNotFoundError(f"IHL 1h parameters file not found: {ihl_1h_params_path}")
    
    ihl_1h_params = load_ihl_1h_params(ihl_1h_params_path)
    
    print("\n" + "="*80)
    print("TWO-STAGE GALAXY AUTO-SPECTRUM FITTING")
    print("="*80)
    print("Using IHL-derived one-halo parameters")
    print(f"Stage 1: {nsteps_stage1} steps with data-based Knox errors")
    print(f"Stage 2: {nsteps_stage2} steps with model-based Knox errors")
    print("="*80 + "\n")
    
    # Compute Knox parameters for Stage 2
    nfield = len(ifield_list)
    field_area = 2.0 * 2.0  # deg^2 per field
    full_sky = 41253.0  # deg^2
    fsky = fmask * (field_area / full_sky) * nfield
    
    # Load galaxy auto-spectra
    if gal_ps_dict is not None:
        # Use pre-loaded spectra (e.g. from large-footprint maps via collect_gal_auto_large_vs_redshift)
        lb = gal_ps_dict['lb']
        full_cl_gal = gal_ps_dict['full_cl_gal']       # [n_inst, n_zbin, n_ell]
        full_clerr_gal = gal_ps_dict['full_clerr_gal']
    else:
        from ciber.plotting.gal_plotting_fns import collect_ciber_gal_vs_redshift

        subtract_randoms = True
        maskstr = 'wFFerr' if cat=='HSC' else 'JHlt16_wFFerr'

        res_ps = collect_ciber_gal_vs_redshift(
            catname, subtract_randoms=subtract_randoms,
            ifield_list=ifield_list,
            inst_list=inst_list, zbinedges=zbinedges,
            maskstr=maskstr, subtract_sn=False,
            tl_pix_correct=False, headstr=headstr,
            with_ff_err=False, fmask=fmask
        )

        lb = res_ps['lb']
        full_cl_gal = res_ps['full_cl_gal']       # [n_inst, n_zbin, n_ell]
        full_clerr_gal = res_ps['full_clerr_gal']  # Includes data-based Knox errors

    pf_data = lb * (lb + 1) / (2 * np.pi)
    
    # Get delta_ell for Knox calculation
    from ciber.core.powerspec_pipeline import CIBER_PS_pipeline
    cbps = CIBER_PS_pipeline()
    delta_ell = cbps.Mkk_obj.delta_ell
    if np.ndim(delta_ell) > 0:
        delta_ell_sliced = delta_ell[startidx:endidx]
    else:
        delta_ell_sliced = delta_ell
    
    all_fit_results = {}
    nzbin = len(zbinedges) - 1
    
    for zidx in range(nzbin):
        zcen = 0.5 * (zbinedges[zidx] + zbinedges[zidx+1])
        
        print("\n" + "="*80)
        print(f"Redshift bin {zidx}: {zbinedges[zidx]:.2f} < z < {zbinedges[zidx+1]:.2f}")
        print("="*80)
        
        for idx, inst in enumerate(inst_list):
            
            # Get IHL-derived fixed parameters
            slope = ihl_1h_params['slopes'][0]
            if (zidx, slope) not in ihl_1h_params['params']:
                print(f"Warning: No IHL parameters for zidx={zidx}, slope={slope}. Skipping.")
                continue
            
            mu_1h_fixed = ihl_1h_params['params'][(zidx, slope)]['mu_1h']
            sigma_1h_fixed = ihl_1h_params['params'][(zidx, slope)]['sigma_1h']
            
            print(f"\nInstrument TM{inst} ({lams[idx]:.1f} μm)")
            print(f"Fixed 1h shape: mu_1h={mu_1h_fixed:.3f}, sigma_1h={sigma_1h_fixed:.3f}")
            
            # Create model instance
            model = CrossPowerSpectrumModel(
                lb,
                use_powerlaw_2h=True,
                alpha_2h_fixed=alpha_from_mock,
                chi2_eval_max=chi2_eval_max,
                mu_1h_fixed=mu_1h_fixed,
                sigma_1h_fixed=sigma_1h_fixed,
                use_astrometry_damping=False
            )
            
            # Prepare data
            dl_data = pf_data * full_cl_gal[idx, zidx]
            dlerr_data_stage1 = pf_data * full_clerr_gal[idx, zidx]  # Includes data-based Knox
            
            # ========================================================================
            # STAGE 1: Fit with data-based Knox errors
            # ========================================================================
            print("\n" + "-"*80)
            print("STAGE 1: Fitting with data-based Knox errors")
            print("-"*80)
            
            fit_result_stage1 = model.fit_model_mcmc(
                lb[startidx:endidx], 
                dl_data[startidx:endidx], 
                dl_err=dlerr_data_stage1[startidx:endidx],
                fit_range=[lb[startidx], lMax_fit],
                chi2_eval_max=lMax_fit,
                nwalkers=nwalkers,
                nsteps=nsteps_stage1,
                nburn=nburn_stage1,
                progress=True,
                verbose=True,
                prior_bounds=prior_bounds,
                z_value=zcen,
                initial_guess=None
            )
            
            print(f"\nStage 1 complete: χ²/dof = {fit_result_stage1['chisq']:.1f}/{fit_result_stage1['ndof']}")
            
            # ========================================================================
            # STAGE 2: Refit with fixed model-based Knox errors from Stage 1
            # ========================================================================
            print("\n" + "-"*80)
            print("STAGE 2: Refitting with fixed Knox errors from Stage 1 best-fit model")
            print("-"*80)
            
            # Compute model D_ell from Stage 1 best-fit parameters
            model_dl_stage1 = model.model_dl(lb[startidx:endidx], *fit_result_stage1['params'])
            # Convert from D_ell to C_ell
            model_cl_stage1_Cl = model_dl_stage1 / pf_data[startidx:endidx]
            
            print(f"Stage 1 best-fit model: mean C_ell = {np.mean(model_cl_stage1_Cl):.2e}")
            
            # Compute Knox variance from Stage 1 model (FIXED for all of Stage 2)
            from ciber.core.powerspec_utils import compute_knox_errors_from_model
            
            # Handle delta_ell_sliced properly
            if np.ndim(delta_ell_sliced) > 0:
                delta_ell_for_knox = delta_ell_sliced
            else:
                delta_ell_for_knox = delta_ell_sliced * np.ones_like(model_cl_stage1_Cl)
            
            knox_err_model = compute_knox_errors_from_model(
                lb[startidx:endidx], 
                model_cl_stage1_Cl,
                delta_ell_for_knox,
                fsky,
                mode='auto'
            )
            
            # Compute measurement-only uncertainties (without Knox component)
            # The full_clerr_gal includes: sqrt(measurement_var + knox_var_data)
            # We need to back out the measurement-only component
            
            # Data-based Knox variance (from original data)
            knox_frac = np.sqrt(2.0 / ((2*lb + 1) * cbps.Mkk_obj.delta_ell * fsky))
            knox_var_data = (knox_frac * np.abs(full_cl_gal[idx, zidx]))**2
            
            # Total variance from Stage 1
            total_var_stage1 = (full_clerr_gal[idx, zidx])**2
            
            # Measurement variance (subtract Knox component)
            measurement_var_cl = np.maximum(total_var_stage1 - knox_var_data, 0)  # Ensure non-negative
            
            # Now combine measurement variance with model-based Knox variance
            # Total variance for Stage 2 (in C_ell space)
            total_var_stage2_cl = measurement_var_cl[startidx:endidx] + knox_err_model**2
            
            # Convert to D_ell space for the fit
            dlerr_data_stage2 = pf_data[startidx:endidx] * np.sqrt(total_var_stage2_cl)
            
            print(f"Measurement variance (without Knox): mean = {np.mean(measurement_var_cl[startidx:endidx]):.2e}")
            print(f"Model-based Knox variance: mean = {np.mean(knox_err_model**2):.2e}")
            print(f"Total Stage 2 variance: mean = {np.mean(total_var_stage2_cl):.2e}")
            print(f"Knox parameters: fsky={fsky:.6f}, delta_ell={delta_ell_sliced if np.ndim(delta_ell_sliced)==0 else 'array'}")
            
            # Use Stage 1 results as initial guess for Stage 2
            # Get the median of the FITTED parameters (not the full reconstructed params)
            # For fixed 1h shape, this should be [A_2h, A_1h, A_shot]
            initial_guess_stage2 = np.median(fit_result_stage1['samples_fitted'], axis=0)
            
            print(f"Stage 1 fitted params for initial guess: {initial_guess_stage2} (length={len(initial_guess_stage2)})")
            
            # Fit with fixed model-based Knox errors (standard MCMC)
            fit_result_stage2 = model.fit_model_mcmc(
                lb[startidx:endidx], 
                dl_data[startidx:endidx],
                dl_err=dlerr_data_stage2,  # Fixed errors from Stage 1 model
                fit_range=[lb[startidx], lMax_fit],
                chi2_eval_max=lMax_fit,
                nwalkers=nwalkers,
                nsteps=nsteps_stage2,
                nburn=nburn_stage2,
                progress=True,
                verbose=True,
                prior_bounds=prior_bounds,
                z_value=zcen,
                initial_guess=initial_guess_stage2
            )
            
            print(f"\nStage 2 complete: χ²/dof = {fit_result_stage2['chisq']:.1f}/{fit_result_stage2['ndof']}")
            
            # ========================================================================
            # Plotting
            # ========================================================================
            title = f'Galaxy Auto {cat} {lams[idx]} $\\mu$m'
            title += f', {zbinedges[zidx]:.1f}<z<{zbinedges[zidx+1]:.1f}'
            


            # Stage 1 plots
            fig1 = CrossPowerSpectrumModel.plot_mcmc_corner(
                fit_result_stage1,
                title=title + ' (Stage 1)', 
                figsize=(5, 5),
                
            )
            
            fig1_ps, ax1 = plot_fit_fixed_1h_templates(
                model, 
                lb[startidx:endidx], 
                dl_data[startidx:endidx], 
                dlerr_data_stage1[startidx:endidx],
                fit_result_stage1, 
                save_path=None, 
                ylim=[1e-4, 5e2], 
                xlim=[50, 1e5],
                figsize=(6, 6), 
                title=title + ' (Stage 1)', 
                title_fs=16, 
                textxpos=350,
                lMax_fit=lMax_fit, 
                chi2_lim=chi2_lim
            )
            
            # Stage 2 plots
            fig2 = CrossPowerSpectrumModel.plot_mcmc_corner(
                fit_result_stage2,
                title=title + ' (Stage 2)', 
                figsize=(5, 5)
            )
            
            # Use the same Stage 2 uncertainties (already computed above)
            fig2_ps, ax2 = plot_fit_fixed_1h_templates(
                model, 
                lb[startidx:endidx], 
                dl_data[startidx:endidx], 
                dlerr_data_stage2,  # Already includes measurement + model-based Knox
                fit_result_stage2, 
                save_path=None, 
                ylim=[1e-4, 5e2], 
                xlim=[50, 1e5],
                figsize=(6, 6), 
                title=title + ' (Stage 2)', 
                title_fs=16, 
                textxpos=350,
                lMax_fit=lMax_fit, 
                chi2_lim=chi2_lim
            )
            
            plt.show()
            
            # Save figures
            if save_figs:
                os.makedirs(figbasedir+catname+'_coarsez/corner/', exist_ok=True)
                os.makedirs(figbasedir+catname+'_coarsez/ps_fits/', exist_ok=True)

                base_fname = f'gal_auto_fit_{zbinedges[zidx]:.1f}_z_{zbinedges[zidx+1]:.1f}_TM{inst}_{catname}_{fitstr}_lMaxfit={lMax_fit}'

                fig1.savefig(f'{figbasedir}{catname}_coarsez/corner/{base_fname}_stage1.png',
                            bbox_inches='tight', dpi=300)
                fig1_ps.savefig(f'{figbasedir}{catname}_coarsez/ps_fits/{base_fname}_stage1.png',
                               bbox_inches='tight', dpi=300)
                fig2.savefig(f'{figbasedir}{catname}_coarsez/corner/{base_fname}_stage2.png',
                            bbox_inches='tight', dpi=300)

                print(f'ps path: {figbasedir}{catname}_coarsez/ps_fits/{base_fname}_stage2.png')
                fig2_ps.savefig(f'{figbasedir}{catname}_coarsez/ps_fits/{base_fname}_stage2.png',
                               bbox_inches='tight', dpi=300)
            
            # Store only Stage 2 results with standard keys (matching cross-spectrum format)
            key = f'inst{inst}_zbin{zidx}'
            all_fit_results[key] = {
                'fit_result': fit_result_stage2,
                'inst': inst,
                'zidx': zidx,
                'zcen': zcen
            }
            
            print("\n" + "="*80)
            print("Comparison:")
            print(f"  Stage 1 χ²/dof: {fit_result_stage1['reduced_chisq']:.2f}")
            print(f"  Stage 2 χ²/dof: {fit_result_stage2['reduced_chisq']:.2f}")
            print("="*80)
    
    if save_results:
        save_fpath = 'data/gal_auto_fits/'+file_fpath
        os.makedirs(os.path.dirname(save_fpath), exist_ok=True)
        print(f'\nSaving results to {save_fpath}')
        save_fit_results_npz(all_fit_results, zbinedges, inst_list, save_fpath, dataset_name=cat)
    
    return all_fit_results


def combine_auto_cross_A2h_samples(gal_auto_results, gal_cross_results, 
                                    inst_list=[1, 2], zbinedges=None,
                                    use_stage2_auto=True):
    """
    Combine A_2h samples from galaxy auto and galaxy×CIBER cross fits to extract intensity bias.
    
    The ratio A_2h^{cross} / A_2h^{gal} relates to the intensity bias and mean intensity:
        A_2h^{cross} / A_2h^{gal} = Δz × b_I × dI/dz
    
    where:
    - Δz is the redshift bin width
    - b_I is the CIBER intensity bias
    - dI/dz is the mean intensity per unit redshift
    
    This function takes MCMC samples from both fits and computes the distribution
    of the ratio, propagating uncertainties through the division.
    
    Parameters
    ----------
    gal_auto_results : dict
        Results from run_gal_auto_fits() or run_gal_auto_fits_two_stage().
        Keys should be 'inst{inst}_zbin{zidx}' or 'inst{inst}_zbin{zidx}_stage{1,2}'
    gal_cross_results : dict
        Results from run_gal_cross_fits().
        Keys should be 'inst{inst}_zbin{zidx}'
    inst_list : list, optional
        Instrument indices to process [1, 2]
    zbinedges : array_like, optional
        Redshift bin edges. If None, extracts from results.
    use_stage2_auto : bool, optional
        If True and two-stage results available, use Stage 2 for auto.
        Default True.
    
    Returns
    -------
    dict
        Dictionary with keys 'inst{inst}_zbin{zidx}' containing:
        - 'ratio_samples': MCMC samples of A_2h^{cross} / A_2h^{gal}
        - 'ratio_median': Median of ratio
        - 'ratio_std': Standard deviation of ratio
        - 'ratio_percentiles': [16th, 50th, 84th] percentiles
        - 'A2h_cross_samples': Cross-spectrum A_2h samples
        - 'A2h_gal_samples': Galaxy auto A_2h samples
        - 'delta_z': Redshift bin width
        - 'zcen': Central redshift
        - 'inst': Instrument index
        - 'zidx': Redshift bin index
        
    Notes
    -----
    The 2-halo amplitude ratio provides information about the intensity field:
    
    For galaxy auto-spectrum:
        C_ℓ^{gg} = A_2h^{gal} × P_ℓ(z)
    
    For galaxy×CIBER cross-spectrum:
        C_ℓ^{gI} = A_2h^{cross} × P_ℓ(z)
    
    The ratio A_2h^{cross} / A_2h^{gal} is related to the intensity contribution:
        A_2h^{cross} / A_2h^{gal} = (b_I / b_g) × (mean I in bin)
    
    where b_g cancels in the ratio, leaving the intensity bias and mean intensity.

    """
    
    combined_results = {}
    
    # Debug: print what keys are available
    print(f"\nDebug: Available auto result keys: {list(gal_auto_results.keys())[:10]}...")  # Show first 10
    print(f"Debug: Available cross result keys: {list(gal_cross_results.keys())[:10]}...")
    
    # Determine number of redshift bins
    if zbinedges is None:
        # Try to infer from results
        nzbin = 0
        for key in gal_auto_results.keys():
            if 'zbin' in key:
                zidx = int(key.split('zbin')[1].split('_')[0])
                nzbin = max(nzbin, zidx + 1)
    else:
        nzbin = len(zbinedges) - 1
    
    for inst in inst_list:
        for zidx in range(nzbin):
            
            # Construct keys for auto results - try multiple formats
            key_auto = None
            if use_stage2_auto:
                # Try stage2 first, then stage1, then no suffix
                for suffix in ['_stage2', '_stage1', '']:
                    test_key = f'inst{inst}_zbin{zidx}{suffix}'
                    if test_key in gal_auto_results:
                        key_auto = test_key
                        break
            else:
                # Try stage1 first, then no suffix, then stage2
                for suffix in ['_stage1', '', '_stage2']:
                    test_key = f'inst{inst}_zbin{zidx}{suffix}'
                    if test_key in gal_auto_results:
                        key_auto = test_key
                        break
            
            # Construct key for cross results
            key_cross = f'inst{inst}_zbin{zidx}'
            
            # Check if both exist
            if key_auto is None:
                print(f"Warning: No auto results found for inst{inst}_zbin{zidx} (tried with/without stage suffix), skipping")
                continue
            if key_cross not in gal_cross_results:
                print(f"Warning: {key_cross} not found in cross results, skipping")
                continue
            
            # Extract samples
            auto_fit = gal_auto_results[key_auto]['fit_result']
            cross_fit = gal_cross_results[key_cross]['fit_result']
            
            # A_2h is the first parameter in both fits
            A2h_gal_samples = auto_fit['samples'][:, 0]  # Shape: (nsamples,)
            A2h_cross_samples = cross_fit['samples'][:, 0]
            
            # Handle different sample sizes by resampling
            n_gal = len(A2h_gal_samples)
            n_cross = len(A2h_cross_samples)
            n_samples = min(n_gal, n_cross)
            
            if n_gal > n_samples:
                idx_gal = np.random.choice(n_gal, n_samples, replace=False)
                A2h_gal_samples = A2h_gal_samples[idx_gal]
            
            if n_cross > n_samples:
                idx_cross = np.random.choice(n_cross, n_samples, replace=False)
                A2h_cross_samples = A2h_cross_samples[idx_cross]
            
            # Compute ratio samples
            ratio_samples = A2h_cross_samples / A2h_gal_samples
            
            # Compute statistics using percentiles (more robust for skewed distributions)
            ratio_median = np.median(ratio_samples)
            ratio_percentiles = np.percentile(ratio_samples, [16, 50, 84])
            # Use 68% confidence interval (16th-84th percentile) for error estimate
            ratio_err_lower = ratio_median - ratio_percentiles[0]
            ratio_err_upper = ratio_percentiles[2] - ratio_median
            ratio_err = 0.5 * (ratio_err_lower + ratio_err_upper)  # Symmetric approximation
            
            # Also compute std for reference (but may be unreliable if distribution is skewed)
            ratio_std = np.std(ratio_samples)
            
            # Check if A_2h^gal is likely an upper limit (16th percentile near zero)
            A2h_gal_percentiles = np.percentile(A2h_gal_samples, [16, 50, 84])
            is_gal_upper_limit = A2h_gal_percentiles[0] < 0.1 * A2h_gal_percentiles[1]
            
            if is_gal_upper_limit:
                print(f"  WARNING: A_2h^gal appears to be an upper limit (16th percentile << median)")
                print(f"           Ratio uncertainties may be unreliable. Consider excluding this bin.")
            
            # Get redshift info
            zcen = gal_auto_results[key_auto]['zcen']
            
            # Compute delta_z if zbinedges provided
            if zbinedges is not None:
                delta_z = zbinedges[zidx+1] - zbinedges[zidx]
            else:
                delta_z = None
            
            # Store results
            output_key = f'inst{inst}_zbin{zidx}'
            combined_results[output_key] = {
                'ratio_samples': ratio_samples,
                'ratio_median': ratio_median,
                'ratio_err': ratio_err,  # 68% confidence interval (more robust)
                'ratio_err_lower': ratio_err_lower,
                'ratio_err_upper': ratio_err_upper,
                'ratio_std': ratio_std,  # Standard deviation (for reference)
                'ratio_percentiles': ratio_percentiles,
                'A2h_cross_samples': A2h_cross_samples,
                'A2h_gal_samples': A2h_gal_samples,
                'A2h_cross_median': np.median(A2h_cross_samples),
                'A2h_cross_std': np.std(A2h_cross_samples),
                'A2h_cross_percentiles': np.percentile(A2h_cross_samples, [16, 50, 84]),
                'A2h_gal_median': np.median(A2h_gal_samples),
                'A2h_gal_std': np.std(A2h_gal_samples),
                'A2h_gal_percentiles': A2h_gal_percentiles,
                'is_gal_upper_limit': is_gal_upper_limit,
                'delta_z': delta_z,
                'zcen': zcen,
                'inst': inst,
                'zidx': zidx,
                'n_samples': n_samples
            }
            
            print(f"\nInstrument {inst}, z-bin {zidx} (z={zcen:.2f}):")
            print(f"  A_2h^gal   = {combined_results[output_key]['A2h_gal_median']:.3e} "
                  f"+{A2h_gal_percentiles[2] - A2h_gal_percentiles[1]:.2e}/-{A2h_gal_percentiles[1] - A2h_gal_percentiles[0]:.2e}")
            print(f"  A_2h^cross = {combined_results[output_key]['A2h_cross_median']:.3e} ± {combined_results[output_key]['A2h_cross_std']:.3e}")
            print(f"  Ratio      = {ratio_median:.3e} +{ratio_err_upper:.2e}/-{ratio_err_lower:.2e} (68% CI)")
            if is_gal_upper_limit:
                print(f"               [WARNING: A_2h^gal is likely an upper limit]")
            if delta_z is not None:
                print(f"  Δz × b_I × dI/dz ≈ {ratio_median:.3e}")
    
    return combined_results


def _compute_linear_2h_templates_per_zbin(zbinedges, lmax_fit, verbose=True, cache_dir='data'):
    """
    Pre-compute linear matter power spectrum angular power (C_ell^lin) for each redshift bin.
    
    Caches individual z-bin templates over a fixed wide range (100 < ell < 1.2e5) to avoid
    redundant Limber projection computation. Templates are then interpolated to the requested
    lmax_fit if needed. Uses fiducial LCDM cosmology with Limber projection from 
    compute_matter_cell_predictions.py
    
    Parameters
    ----------
    zbinedges : array_like
        Redshift bin edges
    lmax_fit : float
        Maximum multipole for fitting. If lmax_fit < 1.2e5, templates are interpolated.
        If lmax_fit > 1.2e5, a warning is issued and clamping is used.
    verbose : bool, optional
        Print progress information
    cache_dir : str, optional
        Directory to cache per-zbin templates. Default 'data'.
        Cache files are named: linear_2h_zbin_{zidx}.npz (independent of lmax_fit)
    
    Returns
    -------
    dict
        Dictionary with keys 0, 1, 2, ... for each z-bin index
        Each value is a tuple (ell_array, dl_array) covering 100 < ell <= lmax_fit
    
    Notes
    -----
    All templates are computed once over the range 100 < ell < 1.2e5 and cached.
    Different lmax_fit values use interpolation from the cached template, which is
    nearly instant. This avoids redundant expensive Limber projections.
    """
    from pathlib import Path
    from scipy.interpolate import interp1d
    
    try:
        from compute_matter_cell_predictions import CAMBPowerSpectra, limber_project_with_power
    except ImportError:
        # Try importing from the scripts directory
        import sys
        sys.path.insert(0, str(Path('.').resolve()))
        from compute_matter_cell_predictions import CAMBPowerSpectra, limber_project_with_power
    
    # Ensure cache directory exists
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Fiducial LCDM cosmology
    cosmo_params = {
        'H0': 67.5,
        'ombh2': 0.022,
        'omch2': 0.122,
        'ns': 0.965,
        'As': 2.1e-9,
    }
    
    # Fixed cache range (once per z-bin, regardless of lmax_fit)
    CACHE_LMAX = 1.2e5
    
    # Initialize CAMB power spectrum object
    if verbose:
        print("\n" + "="*70)
        print("PRE-COMPUTING LINEAR 2H TEMPLATES")
        print("="*70)
        print(f"Cosmology: H0={cosmo_params['H0']}, ombh2={cosmo_params['ombh2']}, omch2={cosmo_params['omch2']}")
        print(f"Redshift bins: {zbinedges}")
        print(f"Requested lMax_fit: {lmax_fit}")
        print(f"Cache range: 100 < ell <= {CACHE_LMAX:.0e}")
        print(f"Cache directory: {cache_path.resolve()}")
    
    # Compute linear 2H template for each z-bin
    dl_2h_lin_per_zbin = {}
    camb_ps = None  # Lazy-initialize only if we need to compute
    
    for zidx in range(len(zbinedges) - 1):
        z_min = zbinedges[zidx]
        z_max = zbinedges[zidx + 1]
        z_cen = 0.5 * (z_min + z_max)
        
        # Check for cached template (independent of lmax_fit)
        cache_file = cache_path / f"linear_2h_zbin_{zidx}.npz"
        
        if cache_file.exists():
            if verbose:
                print(f"\nZ-bin {zidx}: z=[{z_min:.2f}, {z_max:.2f}] (z_cen={z_cen:.2f})")
                print(f"  ✓ Loaded from cache: {cache_file.name}")
            
            # Load cached template
            cached = np.load(cache_file, allow_pickle=True)
            ell_values_full = cached['ell_values']
            dl_lin_full = cached['dl_lin']

        else:
            # Need to compute - initialize CAMB only on first compute
            if camb_ps is None:
                camb_ps = CAMBPowerSpectra(cosmo_params, k_min=1e-4, k_max=1e3, nk=512)
            
            if verbose:
                print(f"\nZ-bin {zidx}: z=[{z_min:.2f}, {z_max:.2f}] (z_cen={z_cen:.2f})")
            
            # Limber projection for this z-bin (full range)
            # Note: limber_project_with_power returns linearly-spaced ell values
            ell_values_linear, cl_lin = limber_project_with_power(
                camb_ps,
                z_min, z_max,
                power_type='linear',
                ell_min=100,
                ell_max=CACHE_LMAX,
                n_ell_bin=100,
                nbin=50  # Redshift integration bins
            )
            
            # Convert linear output to D_ell
            pf = ell_values_linear * (ell_values_linear + 1) / (2 * np.pi)
            dl_lin_linear = pf * cl_lin
            # dl_lin_linear /= np.max(dl_lin_linear) # normalize to peak D_ell units

            # for double checking normalization, let's normalize to a specific ell value (e.g., ell=300) instead of peak
            ell_norm = 300
            dl_lin_linear /= dl_lin_linear[np.argmin(np.abs(ell_values_linear - ell_norm))] # normalize to peak D_ell units

            print('dl lin linear:', dl_lin_linear)
            # Rebin to logarithmic spacing for cache (finely sampled)
            # Use ~100 bins per log decade for smooth interpolation
            n_logbins = int(100 * np.log10(CACHE_LMAX / 100.0))
            ell_values_full = np.logspace(np.log10(100), np.log10(CACHE_LMAX), n_logbins)
            
            # Interpolate linear output onto log-spaced grid
            interp_func_linear = interp1d(ell_values_linear, dl_lin_linear, kind='cubic',
                                          bounds_error=False, fill_value='extrapolate')
            dl_lin_full = interp_func_linear(ell_values_full)
            
            # Save to cache with logarithmically-spaced ell values
            np.savez(cache_file, ell_values=ell_values_full, dl_lin=dl_lin_full)
            
            if verbose:
                print(f"  Computed C_ell^lin for {len(ell_values_full)} multipoles")
                print(f"  ✓ Cached to: {cache_file.name}")
        
        # Now interpolate to requested lmax_fit if needed
        # Use logarithmic spacing for smooth coverage across ell range
        if lmax_fit > CACHE_LMAX:
            if verbose:
                print(f"  ⚠ Requested lMax_fit ({lmax_fit:.0e}) > cache range ({CACHE_LMAX:.0e})")
                print(f"    Using clamping (no extrapolation)")
            # Use edge value for ell > CACHE_LMAX; log-space up to CACHE_LMAX
            n_logbins = int(100 * np.log10(CACHE_LMAX / 100.0))
            ell_fit = np.logspace(np.log10(100), np.log10(min(lmax_fit, CACHE_LMAX)), n_logbins)
        else:
            n_logbins = int(100 * np.log10(lmax_fit / 100.0))
            ell_fit = np.logspace(np.log10(100), np.log10(lmax_fit), n_logbins)
        
        # Interpolate from cached full-range template
        interp_func = interp1d(ell_values_full, dl_lin_full, kind='cubic', 
                               bounds_error=False, fill_value='extrapolate')
        dl_lin_interp = interp_func(ell_fit)
        
        # Clamp at ell boundaries (no extrapolation)
        dl_lin_interp = np.clip(dl_lin_interp, 
                               np.min(dl_lin_full), np.max(dl_lin_full))
        
        dl_2h_lin_per_zbin[zidx] = (ell_fit, dl_lin_interp)
        
        if verbose:
            print(f"  ✓ Interpolated to fit range: {ell_fit[0]:.1f} - {ell_fit[-1]:.1f}")
            print(f"  D_ell^lin range: {np.min(dl_lin_interp):.2e} - {np.max(dl_lin_interp):.2e}")
    
    if verbose:
        print("\n" + "="*70 + "\n")
    
    return dl_2h_lin_per_zbin


def run_gal_cross_fits(inst_list=[1, 2], ifield_list=[4,5,6,7,8], maskstr='JHlt16', cat='DESILS',
                       startidx=2, endidx=-1, zbinedges=[0.0, 0.2, 0.4, 0.6,  0.8,  1.0], lams=[1.1, 1.8], alpha_from_mock=0.0,
                       chi2_eval_max=10000., lMax_fit=80000, fitstr='IHLtemp',
                       figbasedir='figures/ciber_cl_fits_011526/', save_figs=True, save_results=False, file_fpath=None,
                       use_one_halo=True, use_two_halo=True, ihl_1h_params_path='ihl_1h_params_corrected.npz', use_ihl_1h_params=True, fix_ihl_1h_shape=False,
                       mu_1h_fixed_override=None, sigma_1h_fixed_override=None,
                       nwalkers=32, nsteps=4000, nburn=1000, prior_bounds=None, chi2_lim=[-20, 5],
                       use_astrometry_damping=False, initial_guess=None, headstr = 'hsc_ilt24.0', uniform_weight_ell=None,
                       A_2h_fixed_arr=None, use_linear_2h=False, sigma_damp_fixed=None,
                       onehalo_output_dir=None, onehalo_generate_type='bulk', onehalo_fsat_model='single',
                       onehalo_concentration_scale=1.0, onehalo_population='combined',
                       onehalo_fit_popmix=False):
    """
    Run galaxy cross-spectrum fits for CIBER data.
    
    Parameters
    ----------
    inst_list : list, optional
        CIBER instrument list (1=1.1um, 2=1.8um)
    ifield_list : list, optional
        Field indices to include
    maskstr : str, optional
        Mask string identifier
    cat : str, optional
        Catalog name ('DESILS' or 'HSC')
    startidx : int, optional
        Starting index for fit range
    endidx : int, optional
        Ending index for fit range
    zbinedges : array_like, optional
        Redshift bin edges
    lams : list, optional
        Wavelengths in microns for each instrument
    alpha_from_mock : float, optional
        Fixed power-law index for 2-halo term
    chi2_eval_max : float, optional
        Maximum multipole for chi-squared evaluation
    lMax_fit : float, optional
        Maximum multipole for fitting
    fitstr : str, optional
        Fit string identifier for output filenames
    figbasedir : str, optional
        Base directory for saving figures
    save_figs : bool, optional
        Whether to save figures
    save_results : bool, optional
        Whether to save fit results
    file_fpath : str, optional
        File path for saved results
    use_one_halo : bool, optional
        If True (default), include one-halo term in model. If False, fit only 2h + shot + damping.
    ihl_1h_params_path : str, optional
        Path to IHL one-halo parameters file (default: 'ihl_1h_params.npz').
        Used to load IHL-derived 1h parameters for better priors and initialization.
    use_ihl_1h_params : bool, optional
        If True, attempt to load and use IHL-derived one-halo parameters (default True).
        If the file doesn't exist, falls back to default behavior.
    nwalkers : int, optional
        Number of MCMC walkers
    nsteps : int, optional
        Number of MCMC steps
    nburn : int, optional
        Number of burn-in steps
    use_linear_2h : bool, optional
        If True, use linear matter power spectrum angular power C_ell^lin for 2h template
        instead of power-law. Pre-computes templates per z-bin. Default False.
    sigma_damp_fixed : dict or list/tuple, optional
        Fixed sigma_damp values instead of floating with prior. Can be:
        - dict: {1: sigma_damp_tm1, 2: sigma_damp_tm2} for per-instrument values (arcsec)
        - list/tuple: [sigma_damp_tm1, sigma_damp_tm2] (converted to dict internally)
        If None (default), sigma_damp floats normally when use_astrometry_damping=True.
    onehalo_output_dir : str, optional
        Directory containing precomputed onehalo_predict outputs. If provided,
        the fit uses those templates as the fixed 1h shape instead of the
        old lognormal IHL path.
    onehalo_generate_type : str, optional
        onehalo_predict output type: 'bulk' or 'fine'. Default 'bulk'.
    onehalo_fsat_model : str, optional
        Satellite fraction model used when loading onehalo_predict outputs.
        Default 'single'.
    
    Returns
    -------
    dict
        Dictionary of fit results for each instrument and redshift bin
    """
    
    if save_results and file_fpath is None:
        file_fpath = 'ciber_cl_fits_'+cat+'_coarsez_'+fitstr+'.npz'
    
    # Try to load IHL one-halo parameters if enabled
    ihl_1h_params = None
    if use_ihl_1h_params:
        if os.path.exists(ihl_1h_params_path):
            try:
                ihl_1h_params = load_ihl_1h_params(ihl_1h_params_path)
                print("\n" + "="*70)
                print("Using IHL-derived one-halo parameters")
                print("="*70)
                        
            except Exception as e:
                print(f"\nWarning: Could not load IHL 1h parameters from {ihl_1h_params_path}: {e}")
                print("Continuing with default behavior...")
                ihl_1h_params = None
        else:
            print(f"\nNote: IHL 1h params file not found at {ihl_1h_params_path}")
            print("To create this file, run:")
            print("  from ciber.theory.cross_ps_parametric_model import fit_and_decompose_ihl_templates, save_ihl_1h_params")
            print("  results = fit_and_decompose_ihl_templates('ihl_templates/', zbinedges=[...], slopes=[1.0])")
            print(f"  save_ihl_1h_params(results, '{ihl_1h_params_path}', zbinedges=[...], slopes=[1.0])")
            return None
        
    # Update fitstr with new flags
    fitstr_updated = fitstr
    if use_linear_2h:
        fitstr_updated += '_lin2h'
    if sigma_damp_fixed is not None and len(sigma_damp_fixed) > 0:
        fitstr_updated += '_fixsigma'

    # Pre-compute linear 2H templates if requested
    dl_2h_lin_per_zbin = {}
    if use_linear_2h:
        dl_2h_lin_per_zbin = _compute_linear_2h_templates_per_zbin(zbinedges, lMax_fit, verbose=True)
    
    for zidx in range(len(zbinedges) - 1):
        result = dl_2h_lin_per_zbin.get(zidx, None)
        
        if result is not None:
            ell_values, dl_lin = result

            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(6, 5))
            # plt.plot(ell_values, dl_lin, label=f'zbin {zidx}' if dl_lin is not None else None)
            # plt.xlabel('Multipole moment l')
            # plt.ylabel('$D_{\\ell}$ (2h linear)', fontsize=14)
            # plt.yscale('log')
            # plt.xscale('log')
            # plt.xlabel('Multipole $\\ell$', fontsize=14)
            # plt.title(f'Linear 2H Template for zbin {zidx}')
            # plt.legend()
            # plt.savefig(f'linear_2h_template_zbin_{zidx}.png', bbox_inches='tight')
            # plt.close()

    # Ensure sigma_damp_fixed is a dict if provided
    if sigma_damp_fixed is None:
        sigma_damp_fixed = {}
    elif isinstance(sigma_damp_fixed, (list, tuple, np.ndarray)):
        # Convert to dict: {1: sigma_damp_fixed[0], 2: sigma_damp_fixed[1], ...}
        sigma_damp_fixed = {i+1: float(val) for i, val in enumerate(sigma_damp_fixed)}

    onehalo_templates_by_inst = {}
    if onehalo_output_dir is not None:
        if not os.path.exists(onehalo_output_dir):
            raise FileNotFoundError(f"One-halo template directory not found: {onehalo_output_dir}")

        if cat == 'HSC':
            bandstr_select = 'hsc_i'
            mag_cut = 25.0
        elif cat == 'DESILS':
            bandstr_select = 'sdss_z'
            mag_cut = 22.0
        else:
            bandstr_select = 'sdss_z'
            mag_cut = 22.0

        print(f"\nLoading fixed one-halo templates from {onehalo_output_dir} ({onehalo_generate_type})")
        for inst in inst_list:
            templates_by_zidx = {}
            for zidx in range(len(zbinedges) - 1):
                result = load_onehalo_spectrum(
                    onehalo_output_dir,
                    onehalo_fsat_model,
                    bandstr_select,
                    inst=inst,
                    mag_min=18.0,
                    mag_cut=mag_cut,
                    z0=0.05,
                    mode='Ig',
                    generate_type=onehalo_generate_type,
                    concentration_scale=onehalo_concentration_scale,
                    population=onehalo_population,
                )
                if result is None:
                    raise FileNotFoundError(
                        f"Could not locate one-halo template for inst={inst}, zidx={zidx} in {onehalo_output_dir}"
                    )

                if np.ndim(result['dl_spectrum']) == 1:
                    template_data = {
                        'ell_arr': result['ell_arr'],
                        'dl_spectrum': result['dl_spectrum'],
                    }
                    if onehalo_fit_popmix:
                        template_data['dl_spectrum_pop0'] = result.get('dl_spectrum_pop0', None)
                        template_data['dl_spectrum_pop1'] = result.get('dl_spectrum_pop1', None)
                    templates_by_zidx[zidx] = template_data
                else:
                    dl_spectrum = np.asarray(result['dl_spectrum'])
                    if dl_spectrum.shape[0] == len(zbinedges) - 1:
                        selected_template = dl_spectrum[zidx]
                    else:
                        selected_template = _select_onehalo_template_for_zbin(
                            dl_spectrum,
                            z_bin_index=zidx,
                            zbinedges=zbinedges,
                            z0=0.0,
                        )
                    templates_by_zidx[zidx] = {
                        'ell_arr': result['ell_arr'],
                        'dl_spectrum': selected_template,
                    }
                    if onehalo_fit_popmix:
                        dl_pop0 = result.get('dl_spectrum_pop0', None)
                        dl_pop1 = result.get('dl_spectrum_pop1', None)
                        if dl_pop0 is not None:
                            dl_pop0 = np.asarray(dl_pop0)
                            if dl_pop0.ndim == 1:
                                templates_by_zidx[zidx]['dl_spectrum_pop0'] = dl_pop0
                            else:
                                templates_by_zidx[zidx]['dl_spectrum_pop0'] = _select_onehalo_template_for_zbin(
                                    dl_pop0,
                                    z_bin_index=zidx,
                                    zbinedges=zbinedges,
                                    z0=0.0,
                                )
                        if dl_pop1 is not None:
                            dl_pop1 = np.asarray(dl_pop1)
                            if dl_pop1.ndim == 1:
                                templates_by_zidx[zidx]['dl_spectrum_pop1'] = dl_pop1
                            else:
                                templates_by_zidx[zidx]['dl_spectrum_pop1'] = _select_onehalo_template_for_zbin(
                                    dl_pop1,
                                    z_bin_index=zidx,
                                    zbinedges=zbinedges,
                                    z0=0.0,
                                )
            onehalo_templates_by_inst[inst] = templates_by_zidx

        print(f"Loaded one-halo templates for {len(onehalo_templates_by_inst)} instruments")

    if cat=='DESILS':
        catname = 'LS'

        maskstr += '_wFFerr'
        res_ps = collect_ciber_gal_vs_redshift(catname, subtract_randoms=True, \
                              inst_list=inst_list, zbinedges=zbinedges, \
                              maskstr=maskstr, subtract_sn=False, 
                              tl_pix_correct=True, ifield_list=ifield_list,
                              with_ff_err=False,
                              uniform_weight_ell=uniform_weight_ell)

    elif cat=='HSC':
        catname = 'HSC'
        # headstr = 'hsc_ilt24.0'
        
        maskstr = None
        res_ps = collect_ciber_gal_vs_redshift(catname, subtract_randoms=True, \
                                    inst_list=inst_list, zbinedges=zbinedges, \
                                    maskstr=maskstr, subtract_sn=False,
                                    tl_pix_correct=True, ifield_list=ifield_list,
                                      headstr=headstr, with_ff_err=True,
                                      uniform_weight_ell=uniform_weight_ell)

    # Import plotting libraries needed for comparison figures
    import matplotlib.pyplot as plt
        
    lb, full_cl_cross, full_clerr_cross = [res_ps[key] for key in ['lb', 'full_cl_cross', 'full_clerr_cross']]

    pf_data = lb * (lb + 1) / (2 * np.pi)

    all_fit_results_mcmc = {}


    for zidx in range(len(zbinedges) - 1):
    
        zcen = 0.5 * (zbinedges[zidx] + zbinedges[zidx + 1])

        for idx, inst in enumerate(inst_list):
            
            # Get fixed one-halo parameters from explicit override, or IHL-derived values.
            mu_1h_fixed = None
            sigma_1h_fixed = None
            if mu_1h_fixed_override is not None and sigma_1h_fixed_override is not None:
                mu_1h_fixed = float(mu_1h_fixed_override)
                sigma_1h_fixed = float(sigma_1h_fixed_override)
                print(
                    f"Fixing 1h shape from override: mu_1h={mu_1h_fixed:.3f}, "
                    f"sigma_1h={sigma_1h_fixed:.3f}"
                )
            elif fix_ihl_1h_shape and use_ihl_1h_params and ihl_1h_params is not None:
                slope = ihl_1h_params['slopes'][0]
                # Get parameters for this redshift bin
                if (zidx, slope) in ihl_1h_params['params']:
                    mu_1h_fixed = ihl_1h_params['params'][(zidx, slope)]['mu_1h']
                    sigma_1h_fixed = ihl_1h_params['params'][(zidx, slope)]['sigma_1h']
                    print(f"Fixing 1h shape: mu_1h={mu_1h_fixed:.3f}, sigma_1h={sigma_1h_fixed:.3f}")
            
            # Create model instance
            a2h_fixed_val = None
            fix_bias=False
            if A_2h_fixed_arr is not None:
                a2h_fixed_val = float(A_2h_fixed_arr[idx, zidx])
                print(f"[run_gal_cross_fits] Fixing A_2h = {a2h_fixed_val:.4f} (IGL prediction, inst={inst}, zidx={zidx})")
                fix_bias=True
            # Get fixed sigma_damp for this instrument if provided
            sigma_damp_for_inst = sigma_damp_fixed.get(inst, None)
            onehalo_template_1h_dl = onehalo_templates_by_inst.get(inst, None) if onehalo_output_dir is not None else None
            
            model = CrossPowerSpectrumModel(
                lb,
                use_powerlaw_2h=not use_linear_2h,
                alpha_2h_fixed=alpha_from_mock,
                chi2_eval_max=chi2_eval_max,
                mu_1h_fixed=mu_1h_fixed,
                sigma_1h_fixed=sigma_1h_fixed,
                use_astrometry_damping=use_astrometry_damping,
                use_one_halo=use_one_halo,
                use_two_halo=use_two_halo,
                A_2h_fixed=a2h_fixed_val,
                use_linear_2h=use_linear_2h,
                dl_2h_lin_per_zbin=dl_2h_lin_per_zbin,
                sigma_damp_fixed={inst: sigma_damp_for_inst} if sigma_damp_for_inst is not None else {},
                onehalo_template_1h_dl=onehalo_template_1h_dl,
            )
            
            # Prepare data
            dl_data = pf_data * full_cl_cross[idx, zidx]
            dlerr_data = pf_data * full_clerr_cross[idx, zidx]

            # Title for plot
            title = f'CIBER {lams[idx]} $\\mu$m $\\times$ {cat}'
            title += f', {zbinedges[zidx]}<z<{zbinedges[zidx+1]}'

            # Phenomenological log-normal fitting
            fit_result_mcmc = model.fit_model_mcmc(
                lb[startidx:endidx], 
                dl_data[startidx:endidx], 
                dl_err=dlerr_data[startidx:endidx],
                fit_range=[300, lMax_fit],
                chi2_eval_max=lMax_fit,
                nwalkers=nwalkers,
                nsteps=nsteps,
                nburn=nburn,
                progress=True,
                verbose=True,
                prior_bounds=prior_bounds,
                z_value=zcen,
                initial_guess=initial_guess,
                z_bin_index=zidx,
                inst=inst
            )

            diagnostics = _compute_mcmc_diagnostics(
                fit_result_mcmc.get('samples'),
                sampler=fit_result_mcmc.get('sampler'),
                discard=nburn,
                param_names=fit_result_mcmc.get('param_names_fitted'),
            )
            fit_result_mcmc['mcmc_diagnostics'] = diagnostics
            fit_result_mcmc['mcmc_tau'] = diagnostics['tau']
            fit_result_mcmc['mcmc_rhat'] = diagnostics['rhat']
            fit_result_mcmc['mcmc_ess'] = diagnostics['ess']

            diag_text = []
            for name, tau_val, rhat_val, ess_val in zip(
                fit_result_mcmc.get('param_names_fitted', []),
                diagnostics['tau'],
                diagnostics['rhat'],
                diagnostics['ess'],
            ):
                diag_text.append(f"{name}: tau={tau_val:.2f}, Rhat={rhat_val:.3f}, ESS={ess_val:.1f}")
            if diag_text:
                print("  MCMC diagnostics: " + "; ".join(diag_text))
            else:
                print("  MCMC diagnostics unavailable")

            # Corner plot
            fig = CrossPowerSpectrumModel.plot_mcmc_corner(
                fit_result_mcmc,
                title=title, 
                figsize=(5, 5),
                fix_bias=fix_bias,

            )
            plt.show()

            # Power spectrum plot - plot_fit_fixed_1h_templates handles both IHL and parametric models
            fig_mcmc, ax = plot_fit_fixed_1h_templates(
                model, 
                lb[startidx:endidx], 
                dl_data[startidx:endidx], 
                dlerr_data[startidx:endidx], 
                fit_result_mcmc, 
                save_path=None, 
                ylim=[1e-3, 5e2], 
                figsize=(6, 6), 
                title=title, 
                title_fs=16, 
                textxpos=350,
                lMax_fit=lMax_fit,
                z_bin_index=zidx, 
                chi2_lim=chi2_lim 
            )

            if save_figs:
                os.makedirs(figbasedir+cat+'_coarsez/corner/', exist_ok=True)
                os.makedirs(figbasedir+cat+'_coarsez/ps_fits/', exist_ok=True)
                fig.savefig(figbasedir+cat+'_coarsez/corner/ciber_cl_fit_corner_'+str(zbinedges[zidx])+'_z_'+str(zbinedges[zidx+1])+'_TM'+str(inst)+'_'+cat+'_'+fitstr_updated+'_lMaxfit='+str(lMax_fit)+'.png', bbox_inches='tight', dpi=300)
                # fig.savefig(figbasedir+cat+'_coarsez/corner/ciber_cl_fit_corner_'+str(zbinedges[zidx])+'_z_'+str(zbinedges[zidx+1])+'_TM'+str(inst)+'_'+cat+'_'+fitstr_updated+'.png', bbox_inches='tight', dpi=300)
                fig_mcmc.savefig(figbasedir+cat+'_coarsez/ps_fits/ciber_cl_fit_'+str(zbinedges[zidx])+'_z_'+str(zbinedges[zidx+1])+'_TM'+str(inst)+'_'+cat+'_'+fitstr_updated+'_lMaxfit='+str(lMax_fit)+'.png', bbox_inches='tight', dpi=300)

            key = f'inst{inst}_zbin{zidx}'

            # Inject data into fit_result so save_fit_results_npz can persist them
            fit_result_mcmc['lb_fit'] = lb[startidx:endidx]
            fit_result_mcmc['data_dl'] = dl_data[startidx:endidx]
            fit_result_mcmc['data_dlerr'] = dlerr_data[startidx:endidx]
            fit_result_mcmc['use_linear_2h'] = use_linear_2h
            fit_result_mcmc['onehalo_mode'] = onehalo_output_dir is not None
            fit_result_mcmc['onehalo_output_dir'] = onehalo_output_dir if onehalo_output_dir is not None else ""
            fit_result_mcmc['onehalo_generate_type'] = onehalo_generate_type
            fit_result_mcmc['onehalo_fsat_model'] = onehalo_fsat_model
            fit_result_mcmc['onehalo_population'] = onehalo_population
            fit_result_mcmc['onehalo_fit_popmix'] = bool(onehalo_fit_popmix)

            all_fit_results_mcmc[key] = {
                'fit_result': fit_result_mcmc,
                'inst': inst,
                'zidx': zidx,
                'zcen': zcen
            }

    if save_results:
        save_fpath = 'data/cross_cl_fits/'+file_fpath

        print('saving results to ', save_fpath, '..')
        save_fit_results_npz(all_fit_results_mcmc, zbinedges, inst_list, save_fpath, dataset_name=cat)

        # Generate comparison figures: in-situ vs F25B CIBER auto-spectrum
        try:
            from ciber.plotting.gal_plotting_fns import _load_ciber_auto_file
            import matplotlib.pyplot as plt

            bandstr_list = {1: 'J', 2: 'H'}
            os.makedirs(figbasedir + cat + '_coarsez/auto_spectrum_comparison/', exist_ok=True)

            for inst in inst_list:
                bandstr = bandstr_list[inst]
                headstr = headstr if cat == 'HSC' else None
                addstr_use = cat + '_coarsez'
                if headstr is not None:
                    addstr_use += f'_{headstr}'

                # Try to load in-situ auto from cross-product file
                try:
                    from ciber.io.ciber_data_utils import load_ciber_gal_ps
                    cgps_file = load_ciber_gal_ps(inst, cat, addstr=addstr_use)
                    if 'all_cl_ciber_auto_inplace' in cgps_file.files:
                        lb_cross = cgps_file['lb']
                        all_cl_ciber_auto_inplace = cgps_file['all_cl_ciber_auto_inplace']
                        cl_inplace_fieldav = np.nanmean(all_cl_ciber_auto_inplace, axis=0)

                        # Load F25B auto file
                        ciber_auto_f25b = _load_ciber_auto_file(bandstr)
                        lb_f25b = ciber_auto_f25b['lb']
                        cl_f25b = ciber_auto_f25b['fieldav_cl']

                        # Interpolate F25B onto in-situ grid
                        cl_f25b_interp = np.interp(lb_cross, lb_f25b, cl_f25b, left=cl_f25b[0], right=cl_f25b[-1])

                        # Convert to D_ell for plotting
                        dell_inplace = lb_cross * (lb_cross + 1) / (2 * np.pi) * cl_inplace_fieldav
                        dell_f25b = lb_cross * (lb_cross + 1) / (2 * np.pi) * cl_f25b_interp

                        # Figure 1: D_ell comparison
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.loglog(lb_cross, dell_inplace, 'o-', color='C0', linewidth=2.5, markersize=6,
                                 label='In-situ (from cross-spectrum)')
                        ax.loglog(lb_f25b, np.nan_to_num(lb_f25b * (lb_f25b + 1) / (2 * np.pi) * cl_f25b),
                                 's--', color='C1', linewidth=2.5, markersize=5, label='F25B (pre-computed)')
                        ax.set_xlabel('Multipole $\ell$', fontsize=12)
                        ax.set_ylabel('$D_\ell^{II}$', fontsize=12)
                        ax.set_title(f'{cat} TM{inst}: CIBER Auto-Spectrum Comparison', fontsize=13, fontweight='bold')
                        ax.grid(True, alpha=0.3, which='both')
                        ax.legend(fontsize=11, loc='best')
                        fig.tight_layout()
                        fig.savefig(figbasedir + cat + '_coarsez/auto_spectrum_comparison/' +
                                   f'ciber_auto_dell_comparison_TM{inst}_{cat}_{fitstr}.png',
                                   dpi=150, bbox_inches='tight')
                        plt.close(fig)

                        # Figure 2: Ratio
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ratio = np.divide(dell_inplace, dell_f25b, where=dell_f25b > 0, out=np.ones_like(dell_inplace))
                        ax.semilogx(lb_cross, ratio, 'o-', color='C0', linewidth=2, markersize=6,
                                   label='In-situ / F25B')
                        ax.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Reference')
                        ax.set_xlabel('Multipole $\ell$', fontsize=12)
                        ax.set_ylabel('Ratio: In-situ / F25B', fontsize=12)
                        ax.set_title(f'{cat} TM{inst}: Auto-Spectrum Discrepancy', fontsize=13, fontweight='bold')
                        ax.set_ylim([0.5, 2.0])
                        ax.grid(True, alpha=0.3, which='both')
                        ax.legend(fontsize=11, loc='best')
                        fig.tight_layout()
                        fig.savefig(figbasedir + cat + '_coarsez/auto_spectrum_comparison/' +
                                   f'ciber_auto_ratio_comparison_TM{inst}_{cat}_{fitstr}.png',
                                   dpi=150, bbox_inches='tight')
                        plt.close(fig)

                        print(f'[auto-spectrum comparison] Generated figures for TM{inst}')
                    else:
                        print(f'[auto-spectrum comparison] TM{inst}: in-situ auto not found in {addstr_use}')
                except Exception as e:
                    print(f'[auto-spectrum comparison] TM{inst}: could not generate comparison: {e}')
        except Exception as e:
            print(f'[auto-spectrum comparison] Could not generate comparison figures: {e}')

    return all_fit_results_mcmc




def should_report_upper_limit(samples, confidence=0.95):
    """
    Determine whether to report a parameter as an upper limit vs Gaussian uncertainty.
    
    Use upper limits when:
    1. Posterior is truncated by non-negativity boundary
    2. Lower confidence bound is consistent with zero
    3. Posterior is highly skewed (non-Gaussian)
    
    Parameters
    ----------
    samples : array_like
        MCMC samples for a single parameter (1D array)
    confidence : float, optional
        Confidence level (default 0.95 for 2σ)
    
    Returns
    -------
    dict
        - 'use_upper_limit': bool, whether to report upper limit
        - 'reason': str, reason for recommendation
        - 'median': float, median value
        - 'percentile_16': float, 16th percentile
        - 'percentile_84': float, 84th percentile  
        - 'percentile_95': float, 95th percentile (2σ)
        - 'percentile_997': float, 99.7th percentile (3σ)
        - 'mean': float, mean value
        - 'std': float, standard deviation
        - 'skewness': float, distribution skewness

    """
    from scipy.stats import skew
    
    samples = np.asarray(samples)
    
    median = np.median(samples)
    mean = np.mean(samples)
    std = np.std(samples)
    p16, p84 = np.percentile(samples, [16, 84])
    p95 = np.percentile(samples, 95)
    p997 = np.percentile(samples, 99.7)
    skewness = skew(samples)
    
    # Criteria for upper limit reporting
    use_upper_limit = False
    reason = "Gaussian uncertainty appropriate"
    
    # 1. Check if lower bound consistent with zero (within 10% of median)
    if p16 < 0.1 * median:
        use_upper_limit = True
        reason = "Lower bound (16th percentile) consistent with zero"
    
    # 2. Check if distribution piled up at boundary (>20% of samples near zero)
    elif np.sum(samples < 0.05 * median) / len(samples) > 0.2:
        use_upper_limit = True
        reason = "Posterior piled up near zero boundary"
    
    # 3. Check for strong positive skewness (skew > 1 indicates non-Gaussian tail)
    elif skewness > 1.0:
        use_upper_limit = True
        reason = f"Highly skewed distribution (skewness = {skewness:.2f})"
    
    return {
        'use_upper_limit': use_upper_limit,
        'reason': reason,
        'median': median,
        'mean': mean,
        'std': std,
        'percentile_16': p16,
        'percentile_84': p84,
        'percentile_95': p95,
        'percentile_997': p997,
        'skewness': skewness
    }




def compute_mean_chi2_per_bandpower(all_fit_results, inst_list=[1, 2], zbinedges=None):
    """
    Compute mean chi² per bandpower, averaged over redshift bins.
    
    This diagnostic helps identify which multipole ranges systematically contribute
    to poor fits across multiple redshift bins. High values at specific ℓ indicate
    potential model inadequacies, data issues, or underestimated uncertainties.
    
    Parameters
    ----------
    all_fit_results : dict
        Can be either:
        1. Results from run_gal_cross_fits() with keys like 'inst1_zbin0', 'inst2_zbin1'
           Each must contain 'fit_result' with 'lb_fit', 'model_dl', 'residuals' fields.
        2. Loaded results from load_fit_results_npz() with 'lb_fit', 'model_dl', 'residuals'
           as (n_inst, n_zbins) object arrays
    inst_list : list, optional
        List of instruments to analyze (default [1, 2])
    zbinedges : array_like, optional
        Redshift bin edges for reference
        
    Returns
    -------
    dict
        Dictionary with keys for each instrument containing:
        - 'lb': multipole bin centers (common across all z-bins)
        - 'mean_chi2_per_bp': mean chi² contribution per bandpower
        - 'std_chi2_per_bp': standard deviation across z-bins
        - 'mean_residual': mean normalized residual (should be ~0)
        - 'std_residual': std of residuals (should be ~1)
        - 'n_zbins': number of redshift bins averaged
    """
    result_dict = {}
    
    # Determine if this is from run_gal_cross_fits or load_fit_results_npz
    has_fit_result_keys = any('inst' in k and 'zbin' in k for k in all_fit_results.keys())
    
    for inst_idx, inst in enumerate(inst_list):
        # Collect chi2 per bandpower for all z-bins of this instrument
        all_chi2_per_bp = []
        all_residuals = []
        lb_ref = None
        
        if zbinedges is not None:
            n_zbins = len(zbinedges) - 1
        elif has_fit_result_keys:
            # Infer number of z-bins from keys
            n_zbins = len([k for k in all_fit_results.keys() if f'inst{inst}' in k])
        else:
            # From loaded npz: get from array shape
            if 'residuals' in all_fit_results:
                n_zbins = all_fit_results['residuals'].shape[1]
            else:
                print(f"Error: Cannot determine number of z-bins for inst{inst}")
                continue
        
        for zidx in range(n_zbins):
            # Extract data based on format
            if has_fit_result_keys:
                # Format from run_gal_cross_fits
                key = f'inst{inst}_zbin{zidx}'
                if key not in all_fit_results:
                    continue
                    
                fit_result = all_fit_results[key]['fit_result']
                
                # Check if residuals are saved
                if 'residuals' not in fit_result or 'lb_fit' not in fit_result:
                    print(f"Warning: {key} missing 'residuals' or 'lb_fit'. Skipping.")
                    print(f"  Available keys: {list(fit_result.keys())}")
                    continue
                
                lb = fit_result['lb_fit']
                residuals = fit_result['residuals']
            else:
                # Format from load_fit_results_npz
                if 'residuals' not in all_fit_results or 'lb_fit' not in all_fit_results:
                    print(f"Error: Loaded results missing 'residuals' or 'lb_fit' fields.")
                    print(f"  This file may have been created before these fields were saved.")
                    print(f"  Please re-run fits to generate updated result files.")
                    return {}
                
                lb = all_fit_results['lb_fit'][inst_idx, zidx]
                residuals = all_fit_results['residuals'][inst_idx, zidx]
                
                # Skip if None (empty entry)
                if lb is None or residuals is None:
                    continue
            
            chi2_per_bp = residuals**2
            
            if lb_ref is None:
                lb_ref = lb
            elif len(lb) != len(lb_ref) or not np.allclose(lb, lb_ref):
                print(f"Warning: inst{inst} zbin{zidx} has different ℓ bins. Skipping.")
                continue
            
            all_chi2_per_bp.append(chi2_per_bp)
            all_residuals.append(residuals)
        
        if len(all_chi2_per_bp) == 0:
            print(f"Warning: No valid fit results found for inst{inst}")
            continue
        
        # Compute statistics
        all_chi2_per_bp = np.array(all_chi2_per_bp)  # shape: (n_zbins, n_bandpowers)
        all_residuals = np.array(all_residuals)
        
        mean_chi2 = np.mean(all_chi2_per_bp, axis=0)
        std_chi2 = np.std(all_chi2_per_bp, axis=0)
        min_chi2 = np.min(all_chi2_per_bp, axis=0)
        max_chi2 = np.max(all_chi2_per_bp, axis=0)
        
        mean_residual = np.mean(all_residuals, axis=0)
        std_residual = np.std(all_residuals, axis=0)
        min_residual = np.min(all_residuals, axis=0)
        max_residual = np.max(all_residuals, axis=0)
        
        result_dict[inst] = {
            'lb': lb_ref,
            'mean_chi2_per_bp': mean_chi2,
            'std_chi2_per_bp': std_chi2,
            'min_chi2_per_bp': min_chi2,
            'max_chi2_per_bp': max_chi2,
            'mean_residual': mean_residual,
            'std_residual': std_residual,
            'min_residual': min_residual,
            'max_residual': max_residual,
            'n_zbins': len(all_chi2_per_bp)
        }
    
    return result_dict



def run_gal_cross_fits_per_field(inst_list=[1, 2], ifield_list=[4,5,6,7,8], maskstr='JHlt16_wFFerr', 
                                 cat='DESILS', startidx=2, endidx=-1, 
                                 zbinedges=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], lams=[1.1, 1.8], 
                                 alpha_from_mock=0.0, chi2_eval_max=10000., lMax_fit=80000, 
                                 fitstr='IHLtemp', figbasedir='figures/ciber_cl_fits_011526/', 
                                 save_figs=False, save_results=False, file_fpath=None, 
                                 ihl_1h_params_path='ihl_1h_params.npz', use_ihl_1h_params=True, 
                                 fix_ihl_1h_shape=True, nwalkers=32, nsteps=4000, nburn=1000, 
                                 prior_bounds=None, chi2_lim=[-20, 5],
                                 use_astrometry_damping=False, verbose=False, initial_guess=None,
                                 tl_pix_correct=True, fmask=0.7):
    """
    Run galaxy cross-spectrum fits for CIBER data on INDIVIDUAL FIELDS (not field-averaged).
    
    This function is useful for diagnostic purposes to check if certain fields drive
    inconsistencies in the fits. Returns fit results for each field separately so you
    can compare chi², inferred parameters, and identify outlier fields.
    
    **IMPORTANT**: Per-field uncertainties are computed using the same approach as the
    field-averaged fits:
    - Knox errors based on field-averaged cross spectrum
    - Rescaled to individual field sky fraction (fsky ∝ 1 field vs 5 fields)
    - Includes noise × galaxy shot noise from MC realizations
    - Transfer function corrections applied
    
    This ensures that per-field fits have proper uncertainty quantification rather than
    using raw per-field noise estimates.
    
    Parameters
    ----------
    tl_pix_correct : bool, optional
        If True, apply transfer function correction for pixelization effects (default True)
    fmask : float, optional
        Masked fraction parameter for sky coverage (default 0.7)
    
    All other parameters match run_gal_cross_fits() exactly - see that function for full documentation.
    
    Returns
    -------
    dict
        Dictionary with keys like 'inst1_zbin0_ifield4', 'inst1_zbin0_ifield5', etc.
        Each entry contains:
        - 'fit_result': fit result dictionary
        - 'inst': instrument index
        - 'zidx': redshift bin index
        - 'ifield': field index
        - 'zcen': central redshift of bin
    """
    from ciber.plotting.gal_plotting_fns import load_ciber_gal_ps, estimate_cross_uncertainties
    
    if save_results and file_fpath is None:
        file_fpath = 'ciber_cl_fits_'+cat+'_coarsez_'+fitstr+'_perfield.npz'
    
    # Try to load IHL one-halo parameters if enabled
    ihl_1h_params = None
    if use_ihl_1h_params:
        import os
        if os.path.exists(ihl_1h_params_path):
            try:
                ihl_1h_params = load_ihl_1h_params(ihl_1h_params_path)
                print("\n" + "="*70)
                print("Using IHL-derived one-halo parameters")
                print("="*70)
                
                # If priors not explicitly set and we're not using IHL templates,
                # set them based on IHL-derived parameters
                        
            except Exception as e:
                print(f"\nWarning: Could not load IHL 1h parameters from {ihl_1h_params_path}: {e}")
                print("Continuing with default behavior...")
                ihl_1h_params = None
        else:
            print(f"\nNote: IHL 1h params file not found at {ihl_1h_params_path}")
            print("To create this file, run:")
            print("  from ciber.theory.cross_ps_parametric_model import fit_and_decompose_ihl_templates, save_ihl_1h_params")
            print("  results = fit_and_decompose_ihl_templates('ihl_templates/', zbinedges=[...], slopes=[1.0])")
            print(f"  save_ihl_1h_params(results, '{ihl_1h_params_path}', zbinedges=[...], slopes=[1.0])")
            return None
        
    # Set catalog name
    if cat == 'DESILS':
        catname = 'LS'
    elif cat == 'HSC':
        catname = 'HSC'
        headstr = 'hsc_ilt24.0'
    else:
        catname = cat
    
    # Get multipole bins
    from ciber.core.powerspec_pipeline import CIBER_PS_pipeline
    cbps = CIBER_PS_pipeline()
    lb = cbps.Mkk_obj.midbin_ell
    pf_data = lb * (lb + 1) / (2 * np.pi)
    
    all_fit_results_per_field = {}
    
    for zidx in range(len(zbinedges) - 1):
        zcen = 0.5 * (zbinedges[zidx] + zbinedges[zidx + 1])
        
        # Load IHL templates if using template-based fitting
        templates = None
        
        # Load per-field data for this redshift bin
        z0, z1 = zbinedges[zidx], zbinedges[zidx+1]
        addstr = str(np.round(z0, 1))+'_z_'+str(np.round(z1, 1))
        addstr_use = addstr + '_wrandsub'
        
        if cat == 'HSC':
            addstr_use = headstr + '_' + addstr_use + '_wFFerr'
        
        if maskstr is not None and cat == 'DESILS':
            addstr_use += '_' + maskstr
        
        for idx, inst in enumerate(inst_list):
            # Load data file containing per-field power spectra
            cgps_file = load_ciber_gal_ps(inst, catname, addstr=addstr_use)
            lb_loaded, all_cl_cross, all_clerr_cross, all_cl_gal, all_clerr_gal, ifield_list_loaded = [
                cgps_file[key] for key in ['lb', 'all_cl_cross', 'all_clerr_cross', 'all_cl_gal', 'all_clerr_gal', 'ifield_list_use']
            ]
            
            all_clerr_cross /= fmask

            # Compute field-averaged spectra for uncertainty estimation (matches collect_ciber_gal_vs_redshift)
            from ciber.plotting.gal_plotting_fns import mini_proc_clav
            pf_data_temp = lb_loaded * (lb_loaded + 1) / (2 * np.pi)
            pf_gal, _, _, fieldav_cl_gal, _ = mini_proc_clav(
                all_cl_gal, all_clerr_gal, lb_loaded, startidx, endidx, mode='auto'
            )
            pf_cross, _, _, fieldav_cl_cross, _ = mini_proc_clav(
                all_cl_cross, all_clerr_cross, lb_loaded, startidx, endidx, mode='cross'
            )
            
            # Load CIBER auto-spectrum for Knox error calculation
            bandstr_list = ['J', 'H']
            ciber_auto = np.load(f'data/ciber_auto_{bandstr_list[idx]}lt16.0_F25B.npz')
            lb_auto, cl_auto, clerr_auto = [ciber_auto[key] for key in ['lb', 'fieldav_cl', 'fieldav_clerr']]
            
            # Compute properly weighted uncertainties for each field
            # This matches the approach in collect_ciber_gal_vs_redshift
            per_field_clerr_cross = np.zeros_like(all_cl_cross)

            mean_norms = [cbps.zl_levels_ciber_fields[inst][cbps.ciber_field_dict[ifield]] 
                        for ifield in ifield_list]

            # Compute flat field weights
            weights_ff = cbps.compute_ff_weights(inst, mean_norms, ifield_list, photon_noise=True)

            # Compute flat field bias correction for each field
            # This returns the multiplicative correction factor (1 + bias)
            ff_bias_factors = compute_ff_bias(mean_norms, weights=weights_ff)

            for fieldidx, ifield in enumerate(ifield_list_loaded):
                # Use field-averaged cross spectrum as basis, scaled to single field
                per_field_clerr_cross[fieldidx] = estimate_cross_uncertainties(
                    lb_loaded, fieldav_cl_cross, all_clerr_cross[fieldidx],
                    cl_auto*ff_bias_factors[fieldidx], fieldav_cl_gal, nfield=1,  # nfield=1 for individual field
                    startidx=startidx, endidx=endidx, fmask=fmask
                )
            
            # Apply transfer function correction if enabled
            if tl_pix_correct:
                ifield_use = 6  # Use field 6 as reference
                try:
                    tl_pix_file = np.load(f'data/fluctuation_data/transfer_function/tl_clx_pix_TM{inst}_ifield{ifield_use}.npz')
                    tl_pix = tl_pix_file['tl_clx_pix']
                    if verbose:
                        print(f"  Applying transfer function correction from field {ifield_use}")
                except:
                    if verbose:
                        print(f"  Warning: Could not load transfer function for TM{inst}, field {ifield_use}. Skipping correction.")
                    tl_pix = np.ones_like(lb_loaded)
            else:
                tl_pix = np.ones_like(lb_loaded)
            
            # Get IHL-derived fixed parameters if fix_ihl_1h_shape is enabled
            mu_1h_fixed = None
            sigma_1h_fixed = None
            if fix_ihl_1h_shape and use_ihl_1h_params and ihl_1h_params is not None:
                slope = ihl_1h_params['slopes'][0]
                # Get parameters for this redshift bin
                if (zidx, slope) in ihl_1h_params['params']:
                    mu_1h_fixed = ihl_1h_params['params'][(zidx, slope)]['mu_1h']
                    sigma_1h_fixed = ihl_1h_params['params'][(zidx, slope)]['sigma_1h']
                    if verbose:
                        print(f"  Setting fixed 1h shape: mu_1h={mu_1h_fixed:.3f}, sigma_1h={sigma_1h_fixed:.3f}")
                else:
                    print(f"  WARNING: Could not find IHL params for zidx={zidx}, slope={slope}")
                    print(f"  Available keys: {list(ihl_1h_params['params'].keys())}")
                    print(f"  Will fit all 5 parameters instead of fixing mu/sigma")
            elif fix_ihl_1h_shape:
                print(f"  WARNING: fix_ihl_1h_shape=True but IHL params not available")
                print(f"    use_ihl_1h_params={use_ihl_1h_params}, ihl_1h_params={ihl_1h_params is not None}")
                print(f"  Will fit all 5 parameters instead of fixing mu/sigma")
            
            # Create model instance (EXACT COPY FROM run_gal_cross_fits)
            model = CrossPowerSpectrumModel(
                lb,
                use_powerlaw_2h=True,
                alpha_2h_fixed=alpha_from_mock,
                chi2_eval_max=chi2_eval_max,
                mu_1h_fixed=mu_1h_fixed,
                sigma_1h_fixed=sigma_1h_fixed,
                use_astrometry_damping=use_astrometry_damping
            )
            
            # Fit each field separately
            for fieldidx, ifield in enumerate(ifield_list_loaded):
                if ifield not in ifield_list:
                    continue
                
                # Get data for this specific field
                cl_cross_field = all_cl_cross[fieldidx]
                clerr_cross_field = per_field_clerr_cross[fieldidx]  # Use properly computed uncertainties
                
                # Apply transfer function correction
                cl_cross_field = cl_cross_field / tl_pix
                clerr_cross_field = clerr_cross_field / tl_pix
                
                # Prepare data (EXACT COPY FROM run_gal_cross_fits)
                dl_data = pf_data * cl_cross_field
                dlerr_data = pf_data * clerr_cross_field
                
                # Title for plot
                title = f'CIBER {lams[idx]} $\\mu$m $\\times$ {cat}, Field {ifield}'
                title += f', {zbinedges[zidx]}<z<{zbinedges[zidx+1]}'
                
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"Fitting: Inst {inst}, z-bin {zidx}, Field {ifield}")
                    print(f"{'='*60}")
                
                # Phenomenological log-normal fitting
                fit_result_mcmc = model.fit_model_mcmc(
                    lb[startidx:endidx], 
                    dl_data[startidx:endidx], 
                    dl_err=dlerr_data[startidx:endidx],
                    fit_range=[300, lMax_fit],
                    chi2_eval_max=lMax_fit,
                    nwalkers=nwalkers,
                    nsteps=nsteps,
                    nburn=nburn,
                    progress=False if not verbose else True,
                    verbose=verbose,
                    prior_bounds=prior_bounds,
                    z_value=zcen,
                    initial_guess=initial_guess
                )
                
                # Optional: Create corner plot
                if save_figs:
                    # Create field-specific subdirectories
                    field_corner_dir = figbasedir+cat+'_coarsez/corner/field'+str(ifield)+'/'
                    field_ps_dir = figbasedir+cat+'_coarsez/ps_fits/field'+str(ifield)+'/'
                    os.makedirs(field_corner_dir, exist_ok=True)
                    os.makedirs(field_ps_dir, exist_ok=True)
                    
                    fig = CrossPowerSpectrumModel.plot_mcmc_corner(
                        fit_result_mcmc,
                        title=title, 
                        figsize=(5, 5)
                    )
                    fig.savefig(field_corner_dir+'ciber_cl_fit_corner_'+str(zbinedges[zidx])+'_z_'+str(zbinedges[zidx+1])+'_TM'+str(inst)+'_'+cat+'_'+fitstr+'_lMaxfit='+str(lMax_fit)+'.png', bbox_inches='tight', dpi=300)
                    plt.close(fig)
                    
                    # Power spectrum plot
                    fig_mcmc, ax = plot_fit_fixed_1h_templates(
                        model, 
                        lb[startidx:endidx], 
                        dl_data[startidx:endidx], 
                        dlerr_data[startidx:endidx],
                        fit_result_mcmc, 
                        save_path=None, 
                        ylim=[1e-3, 5e2], 
                        figsize=(6, 6), 
                        title=title, 
                        title_fs=16, 
                        textxpos=350,
                        lMax_fit=lMax_fit, 
                        chi2_lim=chi2_lim
                    )
                    fig_mcmc.savefig(field_ps_dir+'ciber_cl_fit_'+str(zbinedges[zidx])+'_z_'+str(zbinedges[zidx+1])+'_TM'+str(inst)+'_'+cat+'_'+fitstr+'_lMaxfit='+str(lMax_fit)+'.png', bbox_inches='tight', dpi=300)
                    plt.close(fig_mcmc)
                
                if verbose:
                    print(f"→ chi²_red = {fit_result_mcmc['reduced_chisq']:.2f}")
                
                # Store results
                key = f'inst{inst}_zbin{zidx}_ifield{ifield}'
                all_fit_results_per_field[key] = {
                    'fit_result': fit_result_mcmc,
                    'inst': inst,
                    'zidx': zidx,
                    'ifield': ifield,
                    'zcen': zcen
                }
    
    if save_results and file_fpath is not None:
        # Save each field's results to a separate file
        # This allows using load_fit_results_npz without modification
        
        # Get unique field indices from results
        ifield_set = set()
        for key in all_fit_results_per_field.keys():
            ifield_set.add(all_fit_results_per_field[key]['ifield'])
        
        for ifield in sorted(ifield_set):
            # Create dict with results for this field only (across all inst/zbin)
            field_results = {}
            for key, val in all_fit_results_per_field.items():
                if val['ifield'] == ifield:
                    # Create key without field suffix for save function
                    new_key = f"inst{val['inst']}_zbin{val['zidx']}"
                    field_results[new_key] = val
            
            # Save using existing save function
            save_fpath = 'data/cross_cl_fits/' + file_fpath.replace('.npz', f'_ifield{ifield}.npz')
            save_fit_results_npz(
                field_results, 
                zbinedges=zbinedges,
                inst_list=inst_list,
                save_path=save_fpath,
                dataset_name=f'{cat}_Field{ifield}'
            )
    
    print(f"\n✓ Completed fits for {len(all_fit_results_per_field)} field/inst/zbin combinations")
    
    return all_fit_results_per_field

