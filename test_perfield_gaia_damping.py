"""Test script: Fit per-field CIBER x Gaia cross spectrum and compare sigma_damp constraints."""

import numpy as np
import emcee
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add repo to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from ciber.plotting.gal_plotting_fns import load_ciber_gal_ps, compute_weighted_cl, estimate_cross_uncertainties, _load_ciber_auto_file
from ciber.core.powerspec_pipeline import CIBER_PS_pipeline


def fit_single_field_gaia_damping(
    inst: int,
    ifield: int,
    all_cl_cross: np.ndarray,
    all_clerr_cross: np.ndarray,
    all_cl_gal: np.ndarray,
    lb: np.ndarray,
    cl_auto: np.ndarray,
    addstr: str = "stars_glt20p5_JHlt14_wFFerr",
    ifield_use: int = 8,
    startidx: int = 2,
    endidx: int = -1,
    mask_frac: float = 0.7,
    nwalkers: int = 32,
    nsteps: int = 2000,
    nburn: int = 500,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Fit single field cross spectrum with Poisson + damping model.
    
    Parameters
    ----------
    inst : int
        Instrument (1 or 2)
    ifield : int
        Field index
    all_cl_cross : ndarray (nfield, nell)
        Cross-spectrum for all fields
    all_clerr_cross : ndarray (nfield, nell)
        Cross-spectrum errors for all fields
    all_cl_gal : ndarray (nfield, nell)
        Galaxy auto-spectrum
    lb : ndarray
        Multipole bin centers
    cl_auto : ndarray
        CIBER auto spectrum
    ... other parameters as in main fit
    
    Returns
    -------
    dict with fit results including sigma_damp constraints
    """
    cbps = CIBER_PS_pipeline()
    
    # Single field data
    cl_cross_field = all_cl_cross[ifield]
    clerr_cross_field = all_clerr_cross[ifield]
    cl_gal_field = all_cl_gal[ifield]
    
    # Knox error for single field
    cross_knox = np.sqrt(1.0 / ((2 * lb + 1) * cbps.Mkk_obj.delta_ell))
    fsky_single = mask_frac * 2 * 2 / 41253.0  # single field
    cross_knox /= np.sqrt(fsky_single)
    cross_knox *= np.abs(cl_cross_field)
    clerr_cross_field = np.sqrt(cross_knox ** 2 + clerr_cross_field ** 2)
    
    # Estimate cross uncertainties
    clerr_cross_field = estimate_cross_uncertainties(
        lb, cl_cross_field, clerr_cross_field,
        cl_auto, cl_gal_field, 1, startidx=2, endidx=-1,
    )
    
    # Transfer function correction
    tl_pix = np.load(
        f"data/fluctuation_data/transfer_function/tl_clx_pix_TM{inst}_ifield{ifield_use}.npz"
    )["tl_clx_pix"]
    cl_cross_field = cl_cross_field / tl_pix
    clerr_cross_field = clerr_cross_field / tl_pix
    
    pf = lb * (lb + 1) / (2 * np.pi)
    dl_data = pf * cl_cross_field
    dl_err = pf * clerr_cross_field
    
    # Select fit range
    lbmask = np.ones(len(lb), dtype=bool)
    lbmask[:startidx] = False
    if endidx != -1:
        lbmask[endidx:] = False
    
    lb_fit = lb[lbmask]
    dl_fit = dl_data[lbmask]
    dl_err_fit = dl_err[lbmask]
    
    # Only fit positive-signal bins
    pos_mask = dl_fit > 0
    lb_fit = lb_fit[pos_mask]
    dl_fit = dl_fit[pos_mask]
    dl_err_fit = dl_err_fit[pos_mask]
    
    pf_fit = lb_fit * (lb_fit + 1) / (2 * np.pi)
    arcsec_to_rad = (1.0 / 3600.0) * (np.pi / 180.0)
    
    # MCMC setup
    A_shot_lo, A_shot_hi = 0.0, 1e-3
    sig_lo, sig_hi = 0.1, 20.0
    
    def _log_prior(p):
        A, sig = p
        if A_shot_lo <= A <= A_shot_hi and sig_lo <= sig <= sig_hi:
            return 0.0
        return -np.inf
    
    def _log_likelihood(p):
        A, sig = p
        sig_r = sig * arcsec_to_rad
        model = A * pf_fit * np.exp(-0.5 * (sig_r * lb_fit) ** 2)
        return -0.5 * np.sum(((dl_fit - model) / dl_err_fit) ** 2)
    
    def _log_prob(p):
        lp = _log_prior(p)
        return lp + _log_likelihood(p) if np.isfinite(lp) else -np.inf
    
    # Initial guess
    shot_mask = lb_fit >= 0.5 * lb_fit.max()
    A_shot_init = float(np.nanmean(dl_fit[shot_mask] / pf_fit[shot_mask])) if np.any(shot_mask) else 1e-5
    A_shot_init = np.clip(A_shot_init, A_shot_lo + 1e-8, A_shot_hi - 1e-8)
    p0_center = np.array([A_shot_init, 2.0])
    p0 = p0_center + np.array([1e-6, 0.5]) * np.random.randn(nwalkers, 2)
    p0[:, 0] = np.clip(p0[:, 0], A_shot_lo + 1e-10, A_shot_hi - 1e-10)
    p0[:, 1] = np.clip(p0[:, 1], sig_lo + 1e-4, sig_hi - 1e-4)
    
    if verbose:
        print(f"    Field {ifield}: Running MCMC...")
    
    sampler = emcee.EnsembleSampler(nwalkers, 2, _log_prob)
    sampler.run_mcmc(p0, nsteps, progress=False)
    samples = sampler.get_chain(discard=nburn, flat=True)
    
    params_med = np.median(samples, axis=0)
    params_16 = np.percentile(samples, 16, axis=0)
    params_84 = np.percentile(samples, 84, axis=0)
    
    return {
        "ifield": ifield,
        "A_shot": params_med[0],
        "A_shot_err_lo": params_med[0] - params_16[0],
        "A_shot_err_hi": params_84[0] - params_med[0],
        "sigma_damp": params_med[1],
        "sigma_damp_err_lo": params_med[1] - params_16[1],
        "sigma_damp_err_hi": params_84[1] - params_med[1],
        "samples": samples,
    }


def main():
    """Run per-field fits and compare to field average."""
    
    print("=" * 70)
    print("Per-field CIBER x Gaia cross-spectrum damping parameter fit")
    print("=" * 70)
    
    addstr = "stars_glt20p5_JHlt14_wFFerr"
    inst_list = [1, 2]
    lam_dict = {1: 1.1, 2: 1.8}
    
    # Field-average reference values (from previous fits)
    fieldav_ref = {
        1: {"sigma_damp": 2.3, "label": "TM1 (1.1 μm)"},
        2: {"sigma_damp": 2.1, "label": "TM2 (1.8 μm)"},
    }
    
    for inst in inst_list:
        print(f"\n{'=' * 70}")
        print(f"Instrument {inst} ({lam_dict[inst]} μm)")
        print(f"{'=' * 70}")
        
        # Load data
        cgps_file = load_ciber_gal_ps(inst, "gaia", addstr=addstr)
        lb = cgps_file["lb"]
        all_cl_gal = cgps_file["all_cl_gal"]
        all_cl_cross = cgps_file["all_cl_cross"]
        all_clerr_cross = cgps_file["all_clerr_cross"]
        ifield_list_use = cgps_file["ifield_list_use"]
        
        ciber_auto = _load_ciber_auto_file(["J", "H"][inst - 1])
        cl_auto = ciber_auto["fieldav_cl"]
        
        print(f"\nAvailable fields: {ifield_list_use}")
        print(f"Field-average reference: σ_damp = {fieldav_ref[inst]['sigma_damp']}''")
        print()
        
        # Fit each field
        results = []
        for field_idx, ifield in enumerate(ifield_list_use):
            result = fit_single_field_gaia_damping(
                inst, field_idx, all_cl_cross, all_clerr_cross, all_cl_gal, lb, cl_auto
            )
            results.append(result)
            
            sigma_damp = result["sigma_damp"]
            sigma_damp_err = (result["sigma_damp_err_lo"] + result["sigma_damp_err_hi"]) / 2
            
            print(f"Field {ifield}:  σ_damp = {sigma_damp:.2f} ± {sigma_damp_err:.2f} arcsec")
        
        # Summary statistics
        sigma_damps = np.array([r["sigma_damp"] for r in results])
        sigma_damp_mean = np.mean(sigma_damps)
        sigma_damp_std = np.std(sigma_damps)
        sigma_damp_sem = sigma_damp_std / np.sqrt(len(sigma_damps))
        
        print()
        print("Summary statistics (per-field):")
        print(f"  Mean:   {sigma_damp_mean:.2f} arcsec")
        print(f"  Std:    {sigma_damp_std:.2f} arcsec")
        print(f"  SEM:    {sigma_damp_sem:.2f} arcsec")
        
        # Comparison to field average
        ref_sigma_damp = fieldav_ref[inst]["sigma_damp"]
        delta = sigma_damp_mean - ref_sigma_damp
        n_sigma = delta / sigma_damp_sem if sigma_damp_sem > 0 else 0
        
        print()
        print("Comparison to field-average reference:")
        print(f"  Reference (field avg):  {ref_sigma_damp:.2f}''")
        print(f"  Per-field mean:         {sigma_damp_mean:.2f}''")
        print(f"  Difference:             {delta:+.2f}'' ({n_sigma:+.1f}σ)")
        
        if abs(n_sigma) <= 1:
            print("  ✓ Consistent (within 1σ)")
        elif abs(n_sigma) <= 2:
            print("  ⚠ Marginally consistent (1-2σ)")
        else:
            print("  ✗ Inconsistent (>2σ)")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
