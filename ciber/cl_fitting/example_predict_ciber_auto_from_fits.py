"""
Predict CIBER intensity auto-spectrum from fitted galaxy cross and auto 2-halo components.

Uses the relation:
    D_ell^{I auto, 2h} = (D_ell^{cross, 2h})^2 / D_ell^{gal auto, 2h}

Where:
- D_ell^{cross, 2h} comes from galaxy × CIBER cross fits (with uncertainties)
- D_ell^{gal auto, 2h} comes from galaxy auto fits (mean only, no uncertainty propagation)
"""

import numpy as np
import matplotlib.pyplot as plt
from ciber.theory.cross_ps_parametric_model import (
    load_fit_results_npz, CrossPowerSpectrumModel, collect_ciber_gal_vs_redshift
)

# =============================================================================
# Configuration
# =============================================================================

cat = 'HSC'
zbinedges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
inst_list = [1, 2]
lams = [1.1, 1.8]
lMax_cross = 50000
lMax_auto = 20000

# Load fitted results

if cat=='HSC':
    cross_fit_path = 'data/cross_cl_fits/HSC_coarsez_ilt25.0_cross_cl_fits_IHL1hfit_fixshape_newcl_thetacut_lMax='+str(lMax_cross)+'.npz'
    auto_fit_path = 'data/gal_auto_fits/HSC_coarsez_gal_auto_fits_two_stage_fixed_1h_lMax='+str(lMax_auto)+'.npz'

else:
    cross_fit_path = 'data/cross_cl_fits/'+cat+'_coarsez_cross_cl_fits_IHL1hfit_fixshape_newcl_thetacut_lMax='+str(lMax_cross)+'.npz'
    auto_fit_path = 'data/gal_auto_fits/'+cat+'_coarsez_gal_auto_fits_two_stage_fixed_1h_lMax='+str(lMax_auto)+'.npz'

cross_results = load_fit_results_npz(cross_fit_path)
auto_results = load_fit_results_npz(auto_fit_path)

# Get measured CIBER auto-spectrum
res_ps = collect_ciber_gal_vs_redshift(
    'LS', subtract_randoms=True,
    inst_list=inst_list,
    zbinedges=zbinedges,
    maskstr='JHlt16_wFFerr',
    subtract_sn=False,
    tl_pix_correct=True,
    ifield_list=[4, 5, 6, 7, 8]
)

# =============================================================================
# Predict intensity auto-spectrum from fits
# =============================================================================

def predict_intensity_auto_from_fits(cross_results, auto_results, inst_idx, 
                                     zbinedges, ell_plot=None, alpha_2h_fixed=0.0):
    """
    Predict CIBER intensity auto-spectrum from fitted 2-halo components.
    
    Uses: D_ell^{I auto, 2h} = (D_ell^{cross, 2h})^2 / D_ell^{gal auto, 2h}
    
    Parameters
    ----------
    cross_results : dict
        Loaded cross-spectrum fit results
    auto_results : dict
        Loaded galaxy auto fit results
    inst_idx : int
        Instrument index (0 for TM1, 1 for TM2)
    zbinedges : array_like
        Redshift bin edges
    ell_plot : array_like, optional
        Multipoles for prediction (default: 100 to 100000)
    alpha_2h_fixed : float, optional
        Power-law index for 2-halo term
    
    Returns
    -------
    dict
        Prediction results with keys:
        - 'ell': multipole array
        - 'dl_pred_by_zbin': dict of predictions per z-bin
        - 'dl_pred_upper_by_zbin': dict of upper bounds
        - 'dl_pred_lower_by_zbin': dict of lower bounds
    """
    if ell_plot is None:
        ell_plot = np.logspace(np.log10(100), np.log10(100000), 500)
    
    n_zbin = len(zbinedges) - 1
    inst = inst_list[inst_idx]
    
    # Create model for computing 2-halo components
    lb_dummy = np.logspace(2, 5, 100)
    model = CrossPowerSpectrumModel(
        lb_dummy,
        use_powerlaw_2h=True,
        alpha_2h_fixed=alpha_2h_fixed
    )
    
    predictions = {
        'ell': ell_plot,
        'dl_pred_by_zbin': {},
        'dl_pred_upper_by_zbin': {},
        'dl_pred_lower_by_zbin': {},
        'inst': inst,
        'zbinedges': zbinedges
    }
    
    for zidx in range(n_zbin):
        # Get cross-spectrum 2-halo amplitude and uncertainty
        params_cross = cross_results['params'][inst_idx, zidx, :]
        params_err_cross = cross_results['params_err'][inst_idx, zidx, :]
        A_2h_cross = params_cross[0]
        A_2h_cross_err = params_err_cross[0]
        
        # Get galaxy auto 2-halo amplitude (no uncertainty)
        params_auto = auto_results['params'][inst_idx, zidx, :]
        A_2h_auto = params_auto[0]
        
        # Compute 2-halo components
        dl_2h_cross = model.powerlaw_2h_component(ell_plot, A_2h_cross, alpha_2h_fixed)
        dl_2h_auto = model.powerlaw_2h_component(ell_plot, A_2h_auto, alpha_2h_fixed)

        ratio = A_2h_cross / A_2h_auto
        print(f"z-bin {zidx}: A_2h_cross = {A_2h_cross:.3e} ± {A_2h_cross_err:.3e}, "+
              f"A_2h_auto = {A_2h_auto:.3e}, Ratio = {ratio:.3e}")


        # Predict intensity auto: (cross)^2 / (gal auto)
        # dl_intensity_auto_pred = (dl_2h_cross**2) / dl_2h_auto
        dl_intensity_auto_pred = dl_2h_auto*(ratio**2)
        # Uncertainty from cross amplitude only: d(I_auto)/d(A_cross) * sigma(A_cross)
        # d/dA_cross [(A_cross * template)^2 / (A_auto * template)] 
        #   = 2 * A_cross * template^2 / (A_auto * template)
        #   = 2 * (A_cross / A_auto) * template
        dl_2h_cross_upper = model.powerlaw_2h_component(ell_plot, A_2h_cross + A_2h_cross_err, alpha_2h_fixed)
        dl_2h_cross_lower = model.powerlaw_2h_component(ell_plot, max(0, A_2h_cross - A_2h_cross_err), alpha_2h_fixed)
        

        dl_intensity_auto_upper = dl_2h_auto * ((A_2h_cross + A_2h_cross_err)/ A_2h_auto)**2
        dl_intensity_auto_lower = dl_2h_auto * (max(0, A_2h_cross - A_2h_cross_err)/ A_2h_auto)**2
        # dl_intensity_auto_upper = (dl_2h_cross_upper**2) / dl_2h_auto
        # dl_intensity_auto_lower = (dl_2h_cross_lower**2) / dl_2h_auto
        
        # Store predictions
        predictions['dl_pred_by_zbin'][zidx] = dl_intensity_auto_pred
        predictions['dl_pred_upper_by_zbin'][zidx] = dl_intensity_auto_upper
        predictions['dl_pred_lower_by_zbin'][zidx] = dl_intensity_auto_lower
    
    return predictions


# =============================================================================
# Plot predictions for each instrument
# =============================================================================

for inst_idx, inst in enumerate(inst_list):
    print(f"\n{'='*80}")
    print(f"Predicting CIBER TM{inst} ({lams[inst_idx]} μm) intensity auto-spectrum")
    print(f"{'='*80}")
    
    # Get predictions
    pred_results = predict_intensity_auto_from_fits(
        cross_results, auto_results, inst_idx,
        zbinedges=zbinedges,
        alpha_2h_fixed=0.0  # Update if needed
    )
    
    # Create figure
    n_zbin = len(zbinedges) - 1
    cmap = plt.cm.jet(np.linspace(0.1, 0.9, n_zbin))
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    # Plot measured CIBER auto-spectrum (total, not per z-bin)
    lb_auto = res_ps['lb_auto']
    cl_ciber_auto = res_ps['full_cl_ciber_auto'][inst_idx]
    clerr_ciber_auto = res_ps['full_clerr_ciber_auto'][inst_idx]
    
    # Convert to D_ell
    pf = lb_auto * (lb_auto + 1) / (2 * np.pi)
    dl_ciber_auto = pf * cl_ciber_auto
    dlerr_ciber_auto = pf * clerr_ciber_auto
    
    # Plot total measured CIBER auto (no redshift information)
    ax.errorbar(lb_auto, dl_ciber_auto, yerr=dlerr_ciber_auto,
               fmt='s', color='black', markersize=5, capsize=3,
               alpha=0.6, label='CIBER Data (total)', zorder=15)
    
    # Plot predictions per z-bin
    for zidx in range(n_zbin):
        zcen = 0.5 * (zbinedges[zidx] + zbinedges[zidx+1])
        color = cmap[zidx]
        
        # Plot prediction
        ell_plot = pred_results['ell']
        dl_pred = pred_results['dl_pred_by_zbin'][zidx]
        dl_upper = pred_results['dl_pred_upper_by_zbin'][zidx]
        dl_lower = pred_results['dl_pred_lower_by_zbin'][zidx]
        
        label = 'Predicted ('+str(zbinedges[zidx])+'$<z<$'+str(zbinedges[zidx+1])+')'
        ax.plot(ell_plot, dl_pred, '-', color=color, linewidth=2.5,
               label=label, alpha=0.8, zorder=5)
        ax.fill_between(ell_plot, dl_lower, dl_upper,
                       color=color, alpha=0.2, zorder=3)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Multipole $\ell$', fontsize=14)
    ax.set_ylabel(r'$D_\ell^{\rm I,auto}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=14)
    ax.set_title(f'{cat} → CIBER {lams[inst_idx]} μm Intensity Auto',
                 fontsize=14)
    ax.set_xlim([200, 1e5])
    ax.set_ylim([1e-1, 1e3])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc=2, ncol=2)
    
    plt.tight_layout()
    plt.savefig(f'figures/ciber_auto_prediction_from_fits_TM{inst}_{cat}.png',
                dpi=200, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Saved figure to figures/ciber_auto_prediction_from_fits_TM{inst}_{cat}.png")

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
