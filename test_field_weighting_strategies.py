"""Test: Compare field averaging strategies for damping parameter fit."""

import numpy as np
import emcee
from pathlib import Path
from typing import Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent))

from ciber.plotting.gal_plotting_fns import load_ciber_gal_ps, compute_weighted_cl, estimate_cross_uncertainties, _load_ciber_auto_file
from ciber.core.powerspec_pipeline import CIBER_PS_pipeline


def fit_damping_generic(
    inst: int,
    cl_cross_avg: np.ndarray,
    clerr_cross_avg: np.ndarray,
    cl_gal_avg: np.ndarray,
    lb: np.ndarray,
    cl_auto: np.ndarray,
    nfield: int,
    ifield_use: int = 8,
    startidx: int = 2,
    endidx: int = -1,
    mask_frac: float = 0.7,
    nwalkers: int = 32,
    nsteps: int = 2000,
    nburn: int = 500,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Generic damping fit for any averaged spectrum."""
    cbps = CIBER_PS_pipeline()
    
    # Knox error
    cross_knox = np.sqrt(1.0 / ((2 * lb + 1) * cbps.Mkk_obj.delta_ell))
    fsky = mask_frac * nfield * 2 * 2 / 41253.0
    cross_knox /= np.sqrt(fsky)
    cross_knox *= np.abs(cl_cross_avg)
    clerr_cross_avg = np.sqrt(cross_knox ** 2 + clerr_cross_avg ** 2)
    
    # Estimate cross uncertainties
    clerr_cross_avg = estimate_cross_uncertainties(
        lb, cl_cross_avg, clerr_cross_avg,
        cl_auto, cl_gal_avg, nfield, startidx=2, endidx=-1,
    )
    
    # Transfer function correction
    tl_pix = np.load(
        f"data/fluctuation_data/transfer_function/tl_clx_pix_TM{inst}_ifield{ifield_use}.npz"
    )["tl_clx_pix"]
    cl_cross_avg = cl_cross_avg / tl_pix
    clerr_cross_avg = clerr_cross_avg / tl_pix
    
    pf = lb * (lb + 1) / (2 * np.pi)
    dl_data = pf * cl_cross_avg
    dl_err = pf * clerr_cross_avg
    
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
        print("    Running MCMC...")
    
    sampler = emcee.EnsembleSampler(nwalkers, 2, _log_prob)
    sampler.run_mcmc(p0, nsteps, progress=False)
    samples = sampler.get_chain(discard=nburn, flat=True)
    
    params_med = np.median(samples, axis=0)
    params_16 = np.percentile(samples, 16, axis=0)
    params_84 = np.percentile(samples, 84, axis=0)
    
    return {
        "A_shot": params_med[0],
        "A_shot_err_lo": params_med[0] - params_16[0],
        "A_shot_err_hi": params_84[0] - params_med[0],
        "sigma_damp": params_med[1],
        "sigma_damp_err_lo": params_med[1] - params_16[1],
        "sigma_damp_err_hi": params_84[1] - params_med[1],
        "samples": samples,
    }


def main():
    """Compare error-weighted vs uniform field weighting."""
    
    print("=" * 80)
    print("Field Averaging Strategy Test: Error-weighted vs Uniform weighting")
    print("=" * 80)
    
    addstr = "stars_glt20p5_JHlt14_wFFerr"
    inst_list = [1, 2]
    lam_dict = {1: 1.1, 2: 1.8}
    
    for inst in inst_list:
        print(f"\n{'=' * 80}")
        print(f"Instrument {inst} ({lam_dict[inst]} μm)")
        print(f"{'=' * 80}")
        
        # Load data
        cgps_file = load_ciber_gal_ps(inst, "gaia", addstr=addstr)
        lb = cgps_file["lb"]
        all_cl_gal = cgps_file["all_cl_gal"]
        all_cl_cross = cgps_file["all_cl_cross"]
        all_clerr_cross = cgps_file["all_clerr_cross"]
        ifield_list_use = cgps_file["ifield_list_use"]
        nfield = len(ifield_list_use)
        
        ciber_auto = _load_ciber_auto_file(["J", "H"][inst - 1])
        cl_auto = ciber_auto["fieldav_cl"]
        
        print(f"Fields: {ifield_list_use} (n={nfield})")
        print()
        
        # --- Strategy 1: Error-weighted field average (current) ---
        print("Strategy 1: Error-weighted field average (current approach)")
        cl_weights = 1.0 / all_clerr_cross ** 2
        fieldav_cl_cross_err_weighted, fieldav_clerr_cross_err_weighted = compute_weighted_cl(
            all_cl_cross.copy(), cl_weights
        )
        fieldav_cl_gal_err_weighted = np.mean(all_cl_gal, axis=0)
        
        result_err_weighted = fit_damping_generic(
            inst, fieldav_cl_cross_err_weighted, fieldav_clerr_cross_err_weighted,
            fieldav_cl_gal_err_weighted, lb, cl_auto, nfield, verbose=True
        )
        
        sigma_damp_err_weighted = result_err_weighted["sigma_damp"]
        sigma_damp_err_weighted_err = (result_err_weighted["sigma_damp_err_lo"] + result_err_weighted["sigma_damp_err_hi"]) / 2
        
        print(f"  σ_damp = {sigma_damp_err_weighted:.2f} ± {sigma_damp_err_weighted_err:.2f} arcsec")
        print()
        
        # --- Strategy 2: Uniform field average ---
        print("Strategy 2: Uniform field average (equal weight to all fields)")
        fieldav_cl_cross_uniform = np.mean(all_cl_cross, axis=0)
        fieldav_clerr_cross_uniform = np.mean(all_clerr_cross, axis=0)
        fieldav_cl_gal_uniform = np.mean(all_cl_gal, axis=0)
        
        result_uniform = fit_damping_generic(
            inst, fieldav_cl_cross_uniform, fieldav_clerr_cross_uniform,
            fieldav_cl_gal_uniform, lb, cl_auto, nfield, verbose=True
        )
        
        sigma_damp_uniform = result_uniform["sigma_damp"]
        sigma_damp_uniform_err = (result_uniform["sigma_damp_err_lo"] + result_uniform["sigma_damp_err_hi"]) / 2
        
        print(f"  σ_damp = {sigma_damp_uniform:.2f} ± {sigma_damp_uniform_err:.2f} arcsec")
        print()
        
        # --- Per-field mean (reference) ---
        print("Reference: Per-field mean")
        # Need to recompute per-field means for comparison
        per_field_results = []
        for field_idx, ifield in enumerate(ifield_list_use):
            cl_cross_field = all_cl_cross[field_idx]
            clerr_cross_field = all_clerr_cross[field_idx]
            cl_gal_field = all_cl_gal[field_idx]
            
            result = fit_damping_generic(
                inst, cl_cross_field, clerr_cross_field, cl_gal_field,
                lb, cl_auto, 1, verbose=False
            )
            per_field_results.append(result["sigma_damp"])
        
        sigma_damp_perfield_mean = np.mean(per_field_results)
        sigma_damp_perfield_std = np.std(per_field_results)
        sigma_damp_perfield_sem = sigma_damp_perfield_std / np.sqrt(len(per_field_results))
        
        print(f"  σ_damp = {sigma_damp_perfield_mean:.2f} ± {sigma_damp_perfield_sem:.2f} arcsec (std={sigma_damp_perfield_std:.2f})")
        print()
        
        # --- Comparison ---
        print("=" * 80)
        print("Comparison to per-field mean:")
        print()
        
        delta_err_weighted = sigma_damp_err_weighted - sigma_damp_perfield_mean
        n_sigma_err_weighted = delta_err_weighted / np.sqrt(sigma_damp_err_weighted_err**2 + sigma_damp_perfield_sem**2)
        
        print(f"Error-weighted approach:")
        print(f"  Δσ_damp vs per-field mean: {delta_err_weighted:+.3f}\" ({n_sigma_err_weighted:+.2f}σ)")
        
        delta_uniform = sigma_damp_uniform - sigma_damp_perfield_mean
        n_sigma_uniform = delta_uniform / np.sqrt(sigma_damp_uniform_err**2 + sigma_damp_perfield_sem**2)
        
        print(f"Uniform weighting approach:")
        print(f"  Δσ_damp vs per-field mean: {delta_uniform:+.3f}\" ({n_sigma_uniform:+.2f}σ)")
        print()
        
        if abs(n_sigma_uniform) < abs(n_sigma_err_weighted):
            print("✓ Uniform weighting brings estimate CLOSER to per-field mean")
            print(f"  Improvement: {abs(n_sigma_err_weighted) - abs(n_sigma_uniform):.1f}σ")
        else:
            print("✗ Uniform weighting does NOT improve agreement")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
