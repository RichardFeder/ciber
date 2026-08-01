import numpy as np
import config
from pathlib import Path
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt

from ciber.plotting.plotting_fns import plot_map, generate_colors
from scipy.stats import chi2

from ciber.io.ciber_data_utils import load_ciber_gal_ps
from ciber.core.powerspec_utils import *
from ciber.core.powerspec_pipeline import *

from ciber.theory.cl_predictions import grab_ciber_cross_vs_z_predfpaths
from ciber.processing.numerical import weighted_mean_and_uncertainty

from dataclasses import dataclass
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D


@dataclass
class auto_cross_cl:
	lb: np.ndarray
	pf: np.ndarray
	posmask_auto: np.ndarray
	negmask_auto: np.ndarray
	fieldav_cl_gal: np.ndarray
	fieldav_clerr_gal: np.ndarray
	posmask: np.ndarray
	negmask: np.ndarray
	fieldav_cl_cross: np.ndarray
	fieldav_clerr_cross: np.ndarray
	ciber_auto_cl: np.ndarray
	ciber_auto_clerr: np.ndarray
	r_ell: np.ndarray
	r_ell_unc: np.ndarray


def _load_ciber_auto_file(bandstr):
	"""Load CIBER auto-spectrum file with backward-compatible path fallback."""
	base = Path(getattr(config, 'ciber_basepath', '.')) / 'data'
	if bandstr == 'J':
		spitzer_name = base / 'input_recovered_ps' / '111323' / 'TM1' / 'observed_J_maglim_Jlt16_Hlt15p5_CH1lt15_122524' / 'ciber_Jband_autos_with_spitzer_mask_F25B.npz'
	else:
		spitzer_name = base / 'input_recovered_ps' / '111323' / 'TM2' / 'observed_H_maglim_Jlt16_Hlt15p5_CH1lt15_122524' / 'ciber_Hband_autos_with_spitzer_mask_F25B.npz'

	candidates = [
		Path('data') / 'feder25_ciber_spitzer' / f'ciber_auto_{bandstr}lt16.0_F25B.npz',
		base / 'feder25_ciber_spitzer' / f'ciber_auto_{bandstr}lt16.0_F25B.npz',
		Path('data') / 'feder_ciber_spitzer' / f'ciber_auto_{bandstr}lt16.0_F25B.npz',
		Path('data') / f'ciber_auto_{bandstr}lt16.0_F25B.npz',
		base / 'feder_ciber_spitzer' / f'ciber_auto_{bandstr}lt16.0_F25B.npz',
		base / f'ciber_auto_{bandstr}lt16.0_F25B.npz',
		spitzer_name,
	]
	for path in candidates:
		if path.exists():
			dat = np.load(str(path), allow_pickle=True)
			if all(k in dat for k in ['lb', 'fieldav_cl', 'fieldav_clerr']):
				out = {
					'lb': np.asarray(dat['lb']),
					'fieldav_cl': np.asarray(dat['fieldav_cl']),
					'fieldav_clerr': np.asarray(dat['fieldav_clerr']),
					'source_path': str(path),
					'source_mode': 'fieldav',
				}
				if 'fieldav_dl' in dat and 'fieldav_dlerr' in dat:
					out['fieldav_dl'] = np.asarray(dat['fieldav_dl'])
					out['fieldav_dlerr'] = np.asarray(dat['fieldav_dlerr'])
				return out
			if all(k in dat for k in ['lb', 'recovered_ps_est_nofluc', 'recovered_dcl']):
				return {
					'lb': np.asarray(dat['lb']),
					'fieldav_cl': np.asarray(dat['recovered_ps_est_nofluc'])[-1],
					'fieldav_clerr': np.asarray(dat['recovered_dcl'])[-1],
					'source_path': str(path),
					'source_mode': 'recovered_last',
				}

	raise FileNotFoundError(
		"Could not find CIBER auto file. Tried: " + ", ".join(str(p) for p in candidates)
	)


def load_onehalo_spectrum(onehalo_output_dir, fsat_model, bandstr_select, inst,
                               mag_min, mag_cut, z0, mode='Ig', generate_type='bulk',
                               logM_min=None, concentration_scale=None, population='combined'):
	"""
	Load one-halo predictions for a given configuration and mode.

	Parameters
	----------
	onehalo_output_dir : str
		Directory where onehalo_predict results are saved
	fsat_model : str
		Satellite fraction model ('single', 'double', etc.)
	bandstr_select : str
		Band selection ('hsc_i', 'sdss_z', etc.)
	inst : int
		Instrument (1 or 2)
	mag_min : float
		Minimum magnitude
	mag_cut : float
		Maximum magnitude
	z0 : float
		Minimum redshift
	mode : str, optional
		One-halo spectrum mode: 'Ig' (cross, default), 'gg' (galaxy auto), 'II' (intensity auto)
	generate_type : str, optional
		Result type: 'bulk' (default) or 'fine'
	logM_min : float, optional
		Minimum halo mass in log10(M/Msun) used when generating the result.
		Must match the run configuration whenever it differs from the
		default of 10.0, since it's encoded in the filename.

	Returns
	-------
	dict or None
		Dictionary with keys 'ell_arr' and 'dl_spectrum' (total spectrum),
		or None if file not found
	"""
	import os
	from ciber.theory.onehalo_predict import generate_config_suffix, generate_onehalo_filename

	# Generate suffix matching onehalo_predict convention
	config_suffix = generate_config_suffix(
		fsat_model, bandstr_select, inst,
		mag_min=mag_min, mag_cut=mag_cut, z0=z0, logM_min=logM_min,
		concentration_scale=concentration_scale,
	)

	filename = generate_onehalo_filename(generate_type, config_suffix, mode=mode)
	filepath = os.path.join(onehalo_output_dir, filename)

	if not os.path.exists(filepath):
		return None

	print('Loading from ', filepath)
	# Load the .npz file
	npz_data = np.load(filepath, allow_pickle=True)

	# Extract ell and predictions
	ell_arr = npz_data['ell_arr']
	all_cross_terms = npz_data['all_cross_terms_plot']  # shape (1, 4, n_ell) for bulk

	# Sum over appropriate pair terms based on mode to get total spectrum
	# Pair term indices: 0=cen×sat, 1=sat×cen, 2=sat×sat, 3=cen×cen
	# Ig mode: term2 + term3 (indices 1, 2)
	# gg mode: term1 + term2 + term3 (indices 0, 1, 2)
	# II mode: term1 + term2 + term3 + term4 (indices 0, 1, 2, 3)
	if mode == 'Ig':
		term_indices = [0, 1, 2]
	elif mode == 'gg':
		term_indices = [0, 1, 2]
	elif mode == 'II':
		term_indices = [0, 1, 2, 3]
	else:
		raise ValueError(f"Unknown mode '{mode}'")


	print("Loading from population selection:", population)

	def _extract_population_spectrum(dl_bypop_arr, pop_idx):
		if dl_bypop_arr is None:
			return None
		arr = np.asarray(dl_bypop_arr)
		if arr.ndim == 2:
			arr = arr[None, :, :]
		if arr.ndim != 3 or arr.shape[0] <= pop_idx:
			return None
		pop_spectra = np.asarray(arr[pop_idx])
		if pop_spectra.ndim == 1:
			return pop_spectra
		selected = np.empty_like(pop_spectra, dtype=float)
		for zidx in range(pop_spectra.shape[0]):
			spec = pop_spectra[zidx]
			if np.all(np.isfinite(spec)) and np.any(np.abs(spec) > 0.0):
				selected[zidx] = spec
			elif zidx > 0:
				selected[zidx] = selected[zidx - 1]
			else:
				selected[zidx] = spec
		return np.asarray(selected)

	dl_bypop = npz_data.get('dl_bypop', None)
	if isinstance(dl_bypop, np.ndarray):
		dl_bypop_arr = np.asarray(dl_bypop)
	elif isinstance(dl_bypop, dict):
		dl_bypop_arr = np.asarray(dl_bypop.get(mode, next(iter(dl_bypop.values()))))
	else:
		dl_bypop_arr = None

	dl_spectrum_pop0 = _extract_population_spectrum(dl_bypop_arr, 0)
	dl_spectrum_pop1 = _extract_population_spectrum(dl_bypop_arr, 1)

	population_key = str(population).lower()
	if population_key in {'combined', 'all', 'default'}:
		if all_cross_terms.ndim == 3 and all_cross_terms.shape[0] == 1:
			# Bulk: shape (1, 4, n_ell)
			dl_spectrum = np.sum(all_cross_terms[0, term_indices, :], axis=0)
		else:
			# Fine: shape (n_zbins, 4, n_ell)
			print('all_cross terms has shape', all_cross_terms.shape)
			dl_spectrum = np.sum(all_cross_terms[term_indices, :, :], axis=0)  # shape (n_zbins, n_ell)
			# return None
			print('dl spectrum has shape', dl_spectrum.shape)
	else:
		pop_idx = 0 if population_key == 'pop0' else 1 if population_key == 'pop1' else None
		if pop_idx is None:
			raise ValueError(f"Unknown population '{population}'. Expected 'combined', 'pop0', or 'pop1'.")

		if pop_idx == 0:
			dl_spectrum = dl_spectrum_pop0
		else:
			dl_spectrum = dl_spectrum_pop1
		if dl_spectrum is None:
			raise ValueError(f"Could not interpret dl_bypop for requested population '{population}'")

	return {
		'ell_arr': ell_arr,
		'dl_spectrum': dl_spectrum,
		'dl_spectrum_pop0': dl_spectrum_pop0,
		'dl_spectrum_pop1': dl_spectrum_pop1,
		'all_cross_terms': all_cross_terms,
	}


def load_onehalo_cross(onehalo_output_dir, fsat_model, bandstr_select, inst,
                            mag_min, mag_cut, z0, generate_type='bulk', logM_min=None):
	"""
	Load one-halo cross predictions (Ig mode) for a given configuration.
	Wrapper for backward compatibility.
	"""
	result = load_onehalo_spectrum(
		onehalo_output_dir, fsat_model, bandstr_select, inst,
		mag_min, mag_cut, z0, mode='Ig', generate_type=generate_type, logM_min=logM_min
	)
	if result is None:
		return None
	return {
		'ell_arr': result['ell_arr'],
		'dl_cross_total': result['dl_spectrum'],
		'all_cross_terms': result['all_cross_terms'],
	}



def plot_mean_chi2_per_bandpower(chi2_diag, inst_list=[1, 2], 
                                 figsize=(6, 5), save_path=None,
                                 ylim_chi2=None, ylim_resid=None, catname='DESILS'):
    """
    Plot mean chi² per bandpower from compute_mean_chi2_per_bandpower().
    
    Creates two separate figures:
    1. Mean chi² per bandpower with min/max bounds as dashed lines
    2. Mean normalized residuals with min/max bounds as dashed lines
    
    Parameters
    ----------
    chi2_diag : dict
        Output from compute_mean_chi2_per_bandpower()
    inst_list : list, optional
        Instruments to plot (default [1, 2])
    figsize : tuple, optional
        Figure size for each plot (default (6, 5))
    save_path : str, optional
        Base path to save figures. Will append '_chi2.png' and '_residuals.png'
    ylim_chi2 : tuple, optional
        Y-axis limits for chi² plot
    ylim_resid : tuple, optional
        Y-axis limits for residuals plot
    catname : str, optional
        Catalog name for labels (default 'DESILS')
        
    Returns
    -------
    fig_chi2, fig_resid : tuple of matplotlib.figure.Figure
        Two separate figures
    """
    colors = ['b', 'r']
    labels = [catname+' $\\times$ CIBER 1.1 $\\mu$m', catname+' $\\times$ CIBER 1.8 $\\mu$m']
    
    # Figure 1: Chi² per bandpower
    fig_chi2, ax_chi2 = plt.subplots(1, 1, figsize=figsize)
    
    # Figure 2: Residuals
    fig_resid, ax_resid = plt.subplots(1, 1, figsize=figsize)
    
    for idx, inst in enumerate(inst_list):
        if inst not in chi2_diag:
            continue
        
        data = chi2_diag[inst]
        lb = data['lb']
        mean_chi2 = data['mean_chi2_per_bp']
        min_chi2 = data['min_chi2_per_bp']
        max_chi2 = data['max_chi2_per_bp']
        mean_resid = data['mean_residual']
        min_resid = data['min_residual']
        max_resid = data['max_residual']
        n_zbins = data['n_zbins']
        
        # Chi² plot
        ax_chi2.plot(lb, mean_chi2, color=colors[idx], label=labels[idx], marker='o', ms=3, lw=1.5)
        ax_chi2.plot(lb, min_chi2, color=colors[idx], linestyle='--', lw=1, alpha=0.6)
        ax_chi2.plot(lb, max_chi2, color=colors[idx], linestyle='--', lw=1, alpha=0.6)
        
        # Residuals plot
        ax_resid.plot(lb, mean_resid, color=colors[idx], label=labels[idx], marker='o', ms=3, lw=1.5)
        ax_resid.plot(lb, min_resid, color=colors[idx], linestyle='--', lw=1, alpha=0.6)
        ax_resid.plot(lb, max_resid, color=colors[idx], linestyle='--', lw=1, alpha=0.6)
    
    # Format chi² plot
    ax_chi2.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Expected')
    ax_chi2.set_xlabel('Multipole ℓ', fontsize=13)
    ax_chi2.set_ylabel(r'Mean $\chi^2$ per bandpower', fontsize=13)
    ax_chi2.legend(fontsize=10)
    ax_chi2.grid(alpha=0.3)
    ax_chi2.set_xscale('log')
    if ylim_chi2 is not None:
        ax_chi2.set_ylim(ylim_chi2)
    ax_chi2.set_title(f'Averaged over {n_zbins} redshift bins', fontsize=11)
    
    # Format residuals plot
    ax_resid.axhline(0.0, color='gray', linestyle='--', alpha=0.5)
    ax_resid.axhline(1.0, color='gray', linestyle=':', alpha=0.3)
    ax_resid.axhline(-1.0, color='gray', linestyle=':', alpha=0.3)
    ax_resid.set_xlabel('Multipole ℓ', fontsize=13)
    ax_resid.set_ylabel(r'Mean normalized residual $\sigma$', fontsize=13)
    ax_resid.legend(fontsize=10)
    ax_resid.grid(alpha=0.3)
    ax_resid.set_xscale('log')
    if ylim_resid is not None:
        ax_resid.set_ylim(ylim_resid)
    ax_resid.set_title('Residual distribution check', fontsize=11)
    
    fig_chi2.tight_layout()
    fig_resid.tight_layout()
    
    if save_path:
        # Save with appropriate suffixes
        if save_path.endswith('.png'):
            base_path = save_path[:-4]
        else:
            base_path = save_path
        
        chi2_path = base_path + '_chi2.png'
        resid_path = base_path + '_residuals.png'
        
        fig_chi2.savefig(chi2_path, dpi=200, bbox_inches='tight')
        fig_resid.savefig(resid_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved mean chi² plot to: {chi2_path}")
        print(f"✓ Saved residuals plot to: {resid_path}")
    
    return fig_chi2, fig_resid

def plot_amplitude_comparison(configs, colors=None, markers=None, linestyles=None,
                              figsize=(6, 6), save_path=None, legend_ncol=1, ylim_2h=[-0.05, 0.3], ylim_ihl=[0.0, 2.0],
                              bbox_to_anchor=[0.02, 1.45], use_cmap=False, cmap_name='Blues',
                              bias_model_overlay=None):
    """
    Plot and compare 2-halo and IHL amplitudes from multiple fit configurations.
    Flexible input to support different comparison scenarios.
    
    Parameters
    ----------
    configs : list of dict or list of results dicts
        Either:
        1. List of config dicts with keys:
           - 'results': results dict from load_fit_results_npz()
           - 'inst': instrument index (1 or 2), or None for all instruments
           - 'label': custom label for this trace
           - 'color': (optional) color for this trace
           - 'marker': (optional) marker for this trace
           - 'linestyle': (optional) linestyle for this trace
        2. List of results dicts (old format, auto-plots all instruments)
        
        Examples:
        # One catalog, both wavelengths:
        configs = [
            {'results': hsc_results, 'inst': 1, 'label': 'HSC 1.1 μm'},
            {'results': hsc_results, 'inst': 2, 'label': 'HSC 1.8 μm'},
        ]
        
        # Two catalogs, single wavelength:
        configs = [
            {'results': hsc_results, 'inst': 1, 'label': 'HSC'},
            {'results': ls_results, 'inst': 1, 'label': 'DESI-LS'},
        ]
        
        # Same data, different models:
        configs = [
            {'results': hsc_model1, 'inst': 1, 'label': 'Model A'},
            {'results': hsc_model2, 'inst': 1, 'label': 'Model B'},
        ]
        
    colors : list of str, optional
        Default colors for traces (overridden by config-specific colors)
    markers : list of str, optional
        Default markers for traces
    linestyles : list of str, optional
        Default linestyles for traces
    figsize : tuple, optional
        Figure size (width, height)
    save_path : str, optional
        Path to save figure
    legend_ncol : int, optional
        Number of columns in legend
    bbox_to_anchor : list or tuple, optional
        Bounding box anchor for legend placement
    use_cmap : bool, optional
        Whether to use a colormap for colors
    cmap_name : str, optional
        Name of the colormap to use if use_cmap is True
        
    Returns
    -------
    figure
        Matplotlib figure object
    """
    # Check if old format (list of results dicts) or new format (list of config dicts)
    if len(configs) > 0 and not isinstance(configs[0], dict):
        raise ValueError("configs must be a list of dicts")
    
    # Check if old format (results dict without 'results' key) or new format
    if 'params' in configs[0]:  # Old format: list of results dicts
        # Convert to new format
        new_configs = []
        for results in configs:
            inst_list = results['inst_list']
            dataset_name = results['dataset_name']
            for inst in inst_list:
                lam = 1.1 if inst == 1 else 1.8
                new_configs.append({
                    'results': results,
                    'inst': inst,
                    'label': f"{dataset_name} {lam} μm"
                })
        configs = new_configs
    
    n_traces = len(configs)
    
    # Set up default colors, markers, linestyles

    if use_cmap:
        print('USING CMAP')
        cmap = plt.get_cmap(cmap_name)
        colors = cmap(np.linspace(0.3, 1.0, n_traces))
    else:
        colors = ['C'+str(i) for i in range(n_traces)]

    # if colors is None:
    #     colors = plt.cm.tab10(np.linspace(0, 0.9, n_traces))

    if markers is None:
        markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p', '*', 'X']
    if linestyles is None:
        linestyles = ['-', '--', '-.', ':']
    
    # Ensure lists are long enough
    markers = (markers * (n_traces // len(markers) + 1))[:n_traces]
    linestyles = (linestyles * (n_traces // len(linestyles) + 1))[:n_traces]
    
    # Check if any configuration has one-halo term (check params shape)
    # If params has only 2 or 3 elements per redshift bin, there's no one-halo term
    has_one_halo = False
    for config in configs:
        results = config['results']
        # Check the number of parameters - if > 2, likely has one-halo
        n_params = results['params'].shape[-1]
        if n_params >= 3:  # [A_2h, A_1h, A_shot] or more
            has_one_halo = True
            break
    
    # Create figure with appropriate number of panels
    if has_one_halo:
        # Assume param order: [A_2h, A_IHL, A_shot]
        fig, axes = plt.subplots(2, 1, figsize=figsize)
    else:
        # Only A_2h plot (no one-halo term)
        fig, axes = plt.subplots(1, 1, figsize=(figsize[0], figsize[1]/2))
        axes = [axes]  # Make it a list for consistent indexing

    # Shade redshift bins of width dz=0.2 between z=0 and z=1
    # Use two contrasting pastel shades (alternating) so bins are easily distinguished
    zbin_edges = np.arange(0.0, 1.0 + 1e-9, 0.2)
    # Soft blue and soft peach
    shade_colors = ('#e8f4ff', '#fff3e6')
    for j in range(len(zbin_edges) - 1):
        z0 = zbin_edges[j]
        z1 = zbin_edges[j + 1]
        color_shade = shade_colors[j % 2]
        for ax_shade in axes:
            ax_shade.axvspan(z0, z1, color=color_shade, alpha=0.22, zorder=0)
    
    # Calculate x-offsets for multiple traces at same z values
    n_traces = len(configs)
    x_offset_scale = 0.02  # offset scale relative to z bin width
    
    # Plot 1: 2-halo amplitude (param index 0)
    ax = axes[0]
    
    # Track which configs have been labeled to ensure one label per config
    labeled_configs = set()
    
    for i, config in enumerate(configs):
        results = config['results']
        inst = config.get('inst', None)
        label = config['label']
        color = colors[i]
        # color = config.get('color', colors[i])
        marker = config.get('marker', markers[i])
        linestyle = config.get('linestyle', linestyles[i])
        
        z_centers = results['z_centers']
        inst_list = results['inst_list']
        
        # Calculate x-offset for this trace
        # Center offsets around 0
        x_offset = (i - (n_traces - 1) / 2) * x_offset_scale
        
        # Determine which instrument index to use
        if inst is not None:
            # Find index of this instrument in inst_list
            try:
                inst_idx = list(inst_list).index(inst)
            except ValueError:
                print(f"Warning: inst={inst} not found in results, skipping")
                continue
            insts_to_plot = [(inst_idx, inst)]
        else:
            # Plot all instruments
            insts_to_plot = list(enumerate(inst_list))
        
        for inst_idx, inst_val in insts_to_plot:
            A_2h = results['params'][inst_idx, :, 0]
            A_2h_err = results['params_err'][inst_idx, :, 0]
            
            # Get 95th percentile for upper limits if available
            A_2h_95 = results.get('params_95', None)
            if A_2h_95 is not None:
                A_2h_95 = A_2h_95[inst_idx, :, 0]
            
            # Determine which points should be upper limits
            # Criterion: posterior mean within 2σ of zero (mean - 2*std <= 0)
            is_upper_limit = (A_2h - 2*A_2h_err) <= 0
            
            # Determine if this config should get a label
            config_label = label if i not in labeled_configs else None
            if config_label is not None:
                labeled_configs.add(i)
            
            # Plot detections (not upper limits) with error bars
            detection_mask = ~is_upper_limit
            if np.any(detection_mask):
                ax.errorbar(z_centers[detection_mask] + x_offset, 
                           A_2h[detection_mask], 
                           yerr=A_2h_err[detection_mask],
                           fmt=marker, color=color, linestyle='none',
                           label=config_label,
                           markersize=5, capsize=5, capthick=2, alpha=0.8)
                # Clear label so upper limits don't get it too
                config_label = None
            
            # Plot upper limits with downward arrows
            if np.any(is_upper_limit):
                if A_2h_95 is not None:
                    upper_limit_values = A_2h_95[is_upper_limit]
                else:
                    # Fallback: use mean + 2*std as upper limit
                    upper_limit_values = A_2h[is_upper_limit] + 2*A_2h_err[is_upper_limit]
                
                # Plot downward arrows from upper limit to zero
                ax.errorbar(z_centers[is_upper_limit] + x_offset,
                           upper_limit_values,
                           yerr=upper_limit_values,  # arrow goes all the way to zero
                           fmt='v',  # downward triangle
                           color=color,
                           linestyle='none',
                           uplims=True,  # makes it a proper upper limit arrow
                           label=config_label,
                           markersize=5,
                           capsize=0,
                           alpha=0.8)
    
    # Bias-model overlay on the A_2h panel.
    # bias_model_overlay is a list of dicts, each with:
    #   'pred_fpaths': list of mock .npz paths (one per z bin, for one instrument)
    #   'z_centers':   array of redshift bin centres matching pred_fpaths
    #   'b_g_values':  array of b_g at each z_center
    #   'label':       legend label (optional)
    #   'color':       line color (optional, default 'k')
    #   'linestyle':   line style (optional, default '--')
    if bias_model_overlay is not None:
        overlays = bias_model_overlay if isinstance(bias_model_overlay, list) else [bias_model_overlay]
        for ov in overlays:
            pred_fpaths = ov['pred_fpaths']
            z_ov        = np.asarray(ov['z_centers'])
            b_g_ov      = np.asarray(ov['b_g_values'])
            ov_color    = ov.get('color', 'k')
            ov_ls       = ov.get('linestyle', '--')
            ov_label    = ov.get('label', 'Bias-corrected model')

            a2h_model = np.full(len(pred_fpaths), np.nan)
            for zidx, fp in enumerate(pred_fpaths):
                try:
                    d    = np.load(fp)
                    lb_m = np.asarray(d['lb'], dtype=float)
                    cl_m = np.asarray(d['cross'], dtype=float)
                    pf_m = lb_m * (lb_m + 1.0) / (2.0 * np.pi)
                    dl_m = pf_m * cl_m
                    shot_mask = (lb_m >= 30000.) & (lb_m <= 80000.) & np.isfinite(dl_m)
                    if shot_mask.any():
                        pf_s  = lb_m[shot_mask] * (lb_m[shot_mask] + 1.0) / (2.0 * np.pi)
                        A_sht = float(np.nanmean(dl_m[shot_mask] / pf_s))
                    else:
                        A_sht = 0.0
                    twoh_mask = (lb_m <= 3000.) & np.isfinite(dl_m)
                    if twoh_mask.any():
                        A_2h_raw = float(np.nanmean(dl_m[twoh_mask] - A_sht * pf_m[twoh_mask]))
                        a2h_model[zidx] = b_g_ov[zidx] * max(A_2h_raw, 0.0)
                except Exception:
                    pass

            ax.plot(z_ov, a2h_model, color=ov_color, linestyle=ov_ls,
                    linewidth=1.5, label=ov_label, zorder=3)

    # ax.set_xlabel('Redshift', fontsize=14)
    # ax.set_ylabel(r'$A_{\rm 2h}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=14)
    ax.set_ylabel(r'$A_{\rm 2h}$', fontsize=14)

    # ax.set_title('2-Halo Amplitude vs Redshift', fontsize=15)
    # ax.legend(fontsize=10, loc='best', ncol=legend_ncol)
    ax.legend(fontsize=10, loc=2, ncol=legend_ncol, bbox_to_anchor=bbox_to_anchor)

    ax.grid(alpha=0.3)
    if ylim_2h is not None:
        ax.set_ylim(ylim_2h)
    ax.set_xlim(0, 1.0)
    ax.tick_params(labelsize=12)
    
    # Only add x-label if this is the only panel
    if not has_one_halo:
        ax.set_xlabel('Redshift', fontsize=14)
    
    # Plot 2: IHL amplitude (param index 1) - only if one-halo term exists
    if has_one_halo:
        ax = axes[1]
        for i, config in enumerate(configs):
            results = config['results']
            inst = config.get('inst', None)
            label = config['label']
            color = colors[i]
            # color = config.get('color', colors[i])
            marker = config.get('marker', markers[i])
            linestyle = config.get('linestyle', linestyles[i])
            
            z_centers = results['z_centers']
            inst_list = results['inst_list']
            
            # Calculate x-offset for this trace (same as panel 1)
            x_offset = (i - (n_traces - 1) / 2) * x_offset_scale
            
            # Determine which instrument index to use
            if inst is not None:
                try:
                    inst_idx = list(inst_list).index(inst)
                except ValueError:
                    continue
                insts_to_plot = [(inst_idx, inst)]
            else:
                insts_to_plot = list(enumerate(inst_list))
            
            for inst_idx, inst_val in insts_to_plot:
                A_IHL = results['params'][inst_idx, :, 1]
                A_IHL_err = results['params_err'][inst_idx, :, 1]
                
                # Apply x-offset and remove linestyle (no connecting lines)
                ax.errorbar(z_centers + x_offset, A_IHL, yerr=A_IHL_err,
                           fmt=marker, color=color, linestyle='none',
                           label=label,
                           markersize=8, capsize=5, capthick=2, alpha=0.8)
        
        ax.set_xlabel('Redshift', fontsize=14)
        ax.set_ylabel(r'$A_{1h}$', fontsize=14)
        # ax.set_title('IHL Amplitude vs Redshift', fontsize=15)

        # ax.legend(fontsize=10, loc=2, ncol=2)
        ax.grid(alpha=0.3)
        if ylim_ihl is not None:
            ax.set_ylim(ylim_ihl)
        ax.set_xlim(0, 1.0)
        ax.tick_params(labelsize=12)
        plt.subplots_adjust(wspace=0.3)
        # plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved comparison plot to: {save_path}")
    
    return fig


def plot_amplitude_comparison_by_instrument(configs, inst_list=(1, 2), colors=None, markers=None,
														linestyles=None, figsize=(10, 6.5),
														save_path=None, legend_ncol=2,
														ylim_2h=None, ylim_ihl=None,
														bbox_to_anchor=(0.5, 1.0),
														use_cmap=True, cmap_name='Blues',
														x_offset_scale=0.05,
														bias_model_overlay=None, 
														markersize=5, capsize=5, capthick=2, alpha=0.8):
	"""Panel figure: rows are A_2h/A_1h, columns are instruments.

	configs: list of dicts with keys 'results', optional 'inst', 'label', 'color'.
	"""
	if len(configs) > 0 and not isinstance(configs[0], dict):
		raise ValueError("configs must be a list of dicts")

	if 'params' in configs[0]:
		new_configs = []
		for results in configs:
			dataset_name = results.get('dataset_name', 'dataset')
			new_configs.append({
				'results': results,
				'inst': None,
				'label': f"{dataset_name}",
			})
		configs = new_configs

	n_traces = len(configs)
	if use_cmap:
		cmap = plt.get_cmap(cmap_name)
		colors = cmap(np.linspace(0.3, 1.0, n_traces))
	else:
		colors = ['C' + str(i) for i in range(n_traces)]
	if markers is None:
		markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p', '*', 'X']
	if linestyles is None:
		linestyles = ['-', '--', '-.', ':']
	markers = (markers * (n_traces // len(markers) + 1))[:n_traces]
	linestyles = (linestyles * (n_traces // len(linestyles) + 1))[:n_traces]

	has_one_halo = False
	for config in configs:
		results = config['results']
		n_params = results['params'].shape[-1]
		if n_params >= 2:
			has_one_halo = True
			break

	n_rows = 2 if has_one_halo else 1
	n_cols = len(inst_list)
	fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey='row')
	axes = np.atleast_2d(axes)

	zbin_edges = np.arange(0.0, 1.0 + 1e-9, 0.2)
	shade_colors = ('#e8f4ff', '#fff3e6')
	for row in range(n_rows):
		for col in range(n_cols):
			ax = axes[row, col]
			for j in range(len(zbin_edges) - 1):
				z0 = zbin_edges[j]
				z1 = zbin_edges[j + 1]
				ax.axvspan(z0, z1, color=shade_colors[j % 2], alpha=0.22, zorder=0)

	labeled_configs = set()
	for col, inst in enumerate(inst_list):
		for i, config in enumerate(configs):
			results = config['results']
			inst_sel = config.get('inst', None)
			if inst_sel is not None and inst_sel != inst:
				continue

			insts_available = list(results['inst_list'])
			if inst not in insts_available:
				continue
			inst_idx = insts_available.index(inst)

			z_centers = results['z_centers']
			x_offset = (i - (n_traces - 1) / 2) * x_offset_scale
			color = config.get('color', colors[i])
			marker = config.get('marker', markers[i])
			label = config.get('label', None)
			label = label if (col == 0 and i not in labeled_configs) else None
			if label is not None:
				labeled_configs.add(i)

			# A_1h row (plot to row 0, which is top)
			if has_one_halo:
				ax = axes[0, col]  # A_1h on top row
				A_1h = results['params'][inst_idx, :, 1]
				A_1h_err = results['params_err'][inst_idx, :, 1]
				A_1h_95 = results.get('params_95', None)
				if A_1h_95 is not None:
					A_1h_95 = A_1h_95[inst_idx, :, 1]
				is_ul_1h = (A_1h - 2 * A_1h_err) <= 0
				detection_mask_1h = ~is_ul_1h
				if np.any(detection_mask_1h):
					ax.errorbar(
						z_centers[detection_mask_1h] + x_offset,
						A_1h[detection_mask_1h], yerr=A_1h_err[detection_mask_1h],
						fmt=marker, color=color, linestyle='none',
						label=None, markersize=markersize, capsize=capsize, capthick=capthick, alpha=alpha,
					)
				if np.any(is_ul_1h):
					upper_limit_values_1h = A_1h_95[is_ul_1h] if A_1h_95 is not None else A_1h[is_ul_1h] + 2 * A_1h_err[is_ul_1h]
					ax.errorbar(
						z_centers[is_ul_1h] + x_offset,
						upper_limit_values_1h,
						yerr=upper_limit_values_1h - 0.02,
						fmt=marker, color=color, linestyle='none',
						uplims=True, label=None, markersize=markersize, capsize=3, capthick=capthick, alpha=alpha,
					)

			# A_2h row (plot to row 1, which is bottom)
			ax = axes[1, col]  # A_2h on bottom row
			A_2h = results['params'][inst_idx, :, 0]
			A_2h_err = results['params_err'][inst_idx, :, 0]
			A_2h_95 = results.get('params_95', None)
			if A_2h_95 is not None:
				A_2h_95 = A_2h_95[inst_idx, :, 0]
			is_ul = (A_2h - 2 * A_2h_err) <= 0
			detection_mask = ~is_ul
			if np.any(detection_mask):
				ax.errorbar(
					z_centers[detection_mask] + x_offset,
					A_2h[detection_mask],
					yerr=A_2h_err[detection_mask],
					fmt=marker, color=color, linestyle='none',
					label=label, markersize=markersize, capsize=capsize, capthick=capthick, alpha=alpha,
				)
				label = None
			if np.any(is_ul):
				upper_limit_values = A_2h_95[is_ul] if A_2h_95 is not None else A_2h[is_ul] + 2 * A_2h_err[is_ul]
				# For log-scale A_2h panel, arrow extends from upper_limit to 1e-3
				yerr_ul = upper_limit_values - 1.3e-3
				ax.errorbar(
					z_centers[is_ul] + x_offset,
					upper_limit_values,
					yerr=yerr_ul,
					fmt=marker, color=color, linestyle='none',
					uplims=True, label=label, markersize=markersize, capsize=3, capthick=capthick, alpha=alpha,
				)

	# Bias-model overlay on A_2h panels — one per instrument column (now row 1).
	# bias_model_overlay: list (one per instrument) of either:
	#   - None (no overlay for this instrument)
	#   - a single overlay dict with keys: 'pred_fpaths', 'z_centers', 'b_g_values', 'label', 'color', 'marker'
	#   - a list of overlay dicts (multiple catalogs per instrument)
	if bias_model_overlay is not None:
		overlays = bias_model_overlay if isinstance(bias_model_overlay, list) else [bias_model_overlay]
		for col, inst in enumerate(inst_list):
			ov_list = overlays[col] if col < len(overlays) else None
			if ov_list is None:
				continue
			# Handle both single overlay dict and list of dicts
			if isinstance(ov_list, dict):
				ov_list = [ov_list]

			for ov_idx, ov in enumerate(ov_list):
				pred_fpaths = ov['pred_fpaths']
				z_ov        = np.asarray(ov['z_centers'])
				b_g_ov      = np.asarray(ov['b_g_values'])
				ov_color    = ov.get('color', 'k')
				ov_marker   = ov.get('marker', 'o')
				ov_label    = ov.get('label', None)
				ov_x_offset = ov.get('x_offset', 0.0)

				a2h_model = np.full(len(pred_fpaths), np.nan)
				for zidx, fp in enumerate(pred_fpaths):
					try:
						d    = np.load(fp)
						lb_m = np.asarray(d['lb'], dtype=float)
						cl_m = np.asarray(d['cross'], dtype=float)
						pf_m = lb_m * (lb_m + 1.0) / (2.0 * np.pi)
						dl_m = pf_m * cl_m
						shot_mask = (lb_m >= 30000.) & (lb_m <= 80000.) & np.isfinite(dl_m)
						pf_s  = lb_m[shot_mask] * (lb_m[shot_mask] + 1.0) / (2.0 * np.pi)
						A_sht = float(np.nanmean(dl_m[shot_mask] / pf_s)) if shot_mask.any() else 0.0
						twoh_mask = (lb_m <= 3000.) & np.isfinite(dl_m)
						if twoh_mask.any():
							A_2h_raw = float(np.nanmean(dl_m[twoh_mask] - A_sht * pf_m[twoh_mask]))
							a2h_model[zidx] = b_g_ov[zidx] * max(A_2h_raw, 0.0)
					except Exception:
						pass

				ov_x_offset =  (ov_idx - (n_traces - 1) / 2) * x_offset_scale
	
				good_ov = np.isfinite(a2h_model)
				# Only label on first overlay and first column
				label_str = ov_label if (ov_label is not None and ov_idx == 0 and col == 0) else None
				axes[1, col].scatter(z_ov[good_ov] + ov_x_offset, a2h_model[good_ov], color=ov_color,
				                     marker=ov_marker, s=markersize*5, zorder=3, label=label_str)

	for col, inst in enumerate(inst_list):
		wavelength = "1.1" if inst == 1 else "1.8"
		# Add wavelength label to top-left of top panel (A_1h)
		# ax_top = axes[0, col]

		for idx in [0, 1]:
			axes[idx, col].text(0.04, 0.95, f"CIBER {wavelength} $\\mu$m",
						fontsize=14, transform=axes[idx, col].transAxes,
						va='top', ha='left')

	for ax in axes.ravel():
		ax.grid(alpha=0.3)
		ax.set_xlim(0, 1.0)
		ax.tick_params(labelsize=12)

	# A_2h is now on row 1 (bottom), A_1h on row 0 (top)
	if has_one_halo:
		axes[0, 0].set_ylabel(r'$A_{\rm 1h}$', fontsize=14)
		if ylim_ihl is not None:
			for col in range(n_cols):
				axes[0, col].set_ylim(ylim_ihl)
		else:
			# Default A_1h limits
			for col in range(n_cols):
				axes[0, col].set_ylim(-0.03, 1.0)

	axes[1, 0].set_ylabel(r'$A_{\rm 2h}$', fontsize=14)
	axes[1, 0].set_yscale('log')
	for col in range(n_cols):
		axes[1, col].set_yscale('log')
		axes[1, col].set_xlabel('Redshift', fontsize=14)

	if ylim_2h is not None:
		for col in range(n_cols):
			axes[1, col].set_ylim(ylim_2h)
	else:
		# Default log scale limits: 1e-3 to 1e0
		for col in range(n_cols):
			axes[1, col].set_ylim(1e-3, 1e0)

	# fig.supxlabel('Redshift', fontsize=14)

	# Collect legend handles from data and overlays (from A_2h panel, row 1)
	handles, labels = axes[1, 0].get_legend_handles_labels()
	if handles:
		fig.legend(handles, labels, fontsize=14, loc='upper center',
					ncol=legend_ncol, bbox_to_anchor=bbox_to_anchor)

	# fig.tight_layout()
	fig.subplots_adjust(wspace=0.1, hspace=0.1)

	if save_path:
		plt.savefig(save_path, dpi=200, bbox_inches='tight')
		print(f"✓ Saved comparison plot to: {save_path}")

	return fig


def plot_amplitude_chi2_by_instrument(configs, inst_list=(1, 2), colors=None, markers=None,
														linestyles=None, figsize=(11, 9),
														save_path=None, legend_ncol=2,
														ylim_2h=[-0.05, 0.6], ylim_ihl=[-0.05, 5.0], ylim_chi2=[0, 4.0],
														bbox_to_anchor=(0.2, 1.2),
														use_cmap=True, cmap_name='Blues',
														x_offset_scale=0.05):
	"""Panel figure: rows are A_2h/A_1h/chi2, columns are instruments."""
	if len(configs) > 0 and not isinstance(configs[0], dict):
		raise ValueError("configs must be a list of dicts")

	if 'params' in configs[0]:
		new_configs = []
		for results in configs:
			dataset_name = results.get('dataset_name', 'dataset')
			new_configs.append({
				'results': results,
				'inst': None,
				'label': f"{dataset_name}",
			})
		configs = new_configs

	n_traces = len(configs)
	# if use_cmap:
	print('cmap name:', cmap_name)
	cmap = plt.get_cmap(cmap_name)
	colors = cmap(np.linspace(0.3, 1.0, n_traces))
	# else:
		# colors = ['C' + str(i) for i in range(n_traces)]
	if markers is None:
		markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p', '*', 'X']
	if linestyles is None:
		linestyles = ['-', '--', '-.', ':']
	markers = (markers * (n_traces // len(markers) + 1))[:n_traces]
	linestyles = (linestyles * (n_traces // len(linestyles) + 1))[:n_traces]

	has_one_halo = False
	for config in configs:
		results = config['results']
		n_params = results['params'].shape[-1]
		if n_params >= 2:
			has_one_halo = True
			break

	n_rows = 3 if has_one_halo else 2
	n_cols = len(inst_list)
	fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey='row')
	axes = np.atleast_2d(axes)

	zbin_edges = np.arange(0.0, 1.0 + 1e-9, 0.2)
	shade_colors = ('#e8f4ff', '#fff3e6')
	for row in range(n_rows):
		for col in range(n_cols):
			ax = axes[row, col]
			for j in range(len(zbin_edges) - 1):
				z0 = zbin_edges[j]
				z1 = zbin_edges[j + 1]
				ax.axvspan(z0, z1, color=shade_colors[j % 2], alpha=0.22, zorder=0)

	labeled_configs = set()
	for col, inst in enumerate(inst_list):
		for i, config in enumerate(configs):
			results = config['results']
			inst_sel = config.get('inst', None)

			lMax = config['lMax']
			cat_name = config['cat_name']
			if inst_sel is not None and inst_sel != inst:
				continue

			insts_available = list(results['inst_list'])
			if inst not in insts_available:
				continue
			inst_idx = insts_available.index(inst)

			z_centers = results['z_centers']
			x_offset = (i - (n_traces - 1) / 2) * x_offset_scale
			# color = config.get('color', colors[i])
			marker = config.get('marker', markers[i])
			label = config.get('label', None)
			label = label if (col == 0 and i not in labeled_configs) else None
			if label is not None:
				labeled_configs.add(i)

			# A_2h row
			ax = axes[0, col]
			A_2h = results['params'][inst_idx, :, 0]
			A_2h_err = results['params_err'][inst_idx, :, 0]
			A_2h_95 = results.get('params_95', None)
			if A_2h_95 is not None:
				A_2h_95 = A_2h_95[inst_idx, :, 0]
			is_ul = (A_2h - 2 * A_2h_err) <= 0
			detection_mask = ~is_ul

			A_1h_95 = results.get('params_95', None)
			print('A_1h_95:', A_1h_95)
			A_1h_95 = A_1h_95[inst_idx, :, 1]
			A_1h = results['params'][inst_idx, :, 1]
			A_1h_err = results['params_err'][inst_idx, :, 1]
			is_ul_1h = (A_1h - 2 * A_1h_err) <= 0
			detection_mask_1h = ~is_ul_1h

			# ax.set_yscale('log')

			if cat_name=='DESILS':
				cat_name_use = 'DESI-LS'
			elif cat_name=='HSC':
				cat_name_use = 'HSC'

			if i==0:

				label = 'CIBER $\\times$ '+str(cat_name_use)+'; $\\ell<'+str(lMax)+'$'
			else:
				label = '$\\ell<'+str(lMax)+'$'


			if np.any(detection_mask):
				ax.errorbar(
					z_centers[detection_mask] + x_offset,
					A_2h[detection_mask],
					yerr=A_2h_err[detection_mask],
					fmt=marker, color=colors[i], linestyle='none',
					label=label, markersize=5, capsize=5, capthick=2, alpha=0.8,
				)
				label = None
			if np.any(is_ul):
				upper_limit_values = A_2h_95[is_ul] if A_2h_95 is not None else A_2h[is_ul] + 2 * A_2h_err[is_ul]
				ax.errorbar(
					z_centers[is_ul] + x_offset,
					upper_limit_values,
					yerr=upper_limit_values,
					fmt='v', color=colors[i], linestyle='none',
					uplims=True, label=label, markersize=5, capsize=0, alpha=0.8,
				)

			# A_1h row
			if has_one_halo:
				ax = axes[1, col]
				# ax.set_yscale('log')

				A_1h = results['params'][inst_idx, :, 1]
				A_1h_err = results['params_err'][inst_idx, :, 1]

				if np.any(detection_mask_1h):
					ax.errorbar(
						z_centers[detection_mask_1h] + x_offset,
						A_1h[detection_mask_1h],
						yerr=A_1h_err[detection_mask_1h],
						fmt=marker, color=colors[i], linestyle='none',
						label=label, markersize=5, capsize=5, capthick=2, alpha=0.8,
					)
					label = None

				if np.any(is_ul_1h):
					upper_limit_values = A_1h_95[is_ul_1h] if A_1h_95 is not None else A_1h[is_ul_1h] + 2 * A_1h_err[is_ul_1h]
					ax.errorbar(
						z_centers[is_ul_1h] + x_offset,
						upper_limit_values,
						yerr=upper_limit_values,
						fmt='v', color=colors[i], linestyle='none',
						uplims=True, label=label, markersize=5, capsize=0, alpha=0.8,
					)


				# ax.errorbar(
				# 	z_centers + x_offset,
				# 	A_1h, yerr=A_1h_err,
				# 	fmt=marker, color=colors[i], linestyle='none',
				# 	label=None, markersize=8, capsize=5, capthick=2, alpha=0.8,
				# )

			# Chi2 row
			ax = axes[-1, col]
			chi2_values = results['reduced_chisq'][inst_idx, :]
			ax.plot(
				z_centers + x_offset, chi2_values,
				marker=marker, color=colors[i], linestyle='none',
				label=None, markersize=8, linewidth=2, alpha=0.8,
			)

	for col, inst in enumerate(inst_list):
		ax_top = axes[0, col]
		ax_top.set_title(f"CIBER {1.1 if inst == 1 else 1.8} $\mu$m", fontsize=16)

	for ax in axes.ravel():
		ax.grid(alpha=0.3)
		ax.set_xlim(0, 1.0)
		ax.tick_params(labelsize=12)

	for col in range(n_cols):
		axes[-1, col].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, zorder=1)
		axes[-1, col].set_xlabel('Redshift', fontsize=14)

	if ylim_2h is not None:
		for col in range(n_cols):
			axes[0, col].set_ylim(ylim_2h)
	if has_one_halo and ylim_ihl is not None:
		for col in range(n_cols):
			axes[1, col].set_ylim(ylim_ihl)
	if ylim_chi2 is not None:
		for col in range(n_cols):
			axes[-1, col].set_ylim(ylim_chi2)

	axes[0, 0].set_ylabel(r'$A_{\rm 2h}$', fontsize=14)
	if has_one_halo:
		axes[1, 0].set_ylabel(r'$A_{\rm 1h}$', fontsize=14)
	axes[-1, 0].set_ylabel(r'Reduced $\chi^2$', fontsize=14)

	handles, labels = axes[0, 0].get_legend_handles_labels()
	if handles:
		fig.legend(handles, labels, fontsize=18, loc='upper center',
					ncol=legend_ncol, bbox_to_anchor=bbox_to_anchor)

	# fig.tight_layout()
	fig.subplots_adjust(wspace=0.1, hspace=0.1)

	if save_path:
		plt.savefig(save_path, dpi=200, bbox_inches='tight')
		print(f"✓ Saved comparison plot to: {save_path}")

	return fig



def plot_chi2_comparison(configs, colors=None, markers=None, linestyles=None,
                        figsize=(6, 4), save_path=None, legend_ncol=1, ylim_chi2=None, 
                        bbox_to_anchor=[0.02, 1.15], plot_reduced=True, 
                        use_cmap=False, cmap_name='Blues'):
    """
    Plot and compare chi-squared values from multiple fit configurations vs redshift.
    Shows goodness-of-fit trends across redshift bins for different ell_max cuts or models.
    
    Parameters
    ----------
    configs : list of dict
        List of config dicts with keys:
        - 'results': results dict from load_fit_results_npz() containing 'chisq' and 'reduced_chisq'
        - 'inst': instrument index (1 or 2), or None for all instruments
        - 'label': custom label for this trace
        - 'color': (optional) color for this trace
        - 'marker': (optional) marker for this trace
        - 'linestyle': (optional) linestyle for this trace
        
    colors : list of str, optional
        Default colors for traces
    markers : list of str, optional  
        Default markers for traces
    linestyles : list of str, optional
        Default linestyles for traces
    figsize : tuple, optional
        Figure size (width, height). Default (8, 5)
    save_path : str, optional
        Path to save figure
    legend_ncol : int, optional
        Number of columns in legend
    ylim_chi2 : tuple, optional
        Y-axis limits for chi2 plot
    bbox_to_anchor : list, optional
        Legend position anchor
    plot_reduced : bool, optional
        If True, plot reduced chi-squared. If False, plot raw chi-squared. Default True.
        
    Returns
    -------
    figure
        Matplotlib figure object
        
    """
    
    # Check format and convert if needed (same as amplitude comparison)
    if len(configs) > 0 and not isinstance(configs[0], dict):
        raise ValueError("configs must be a list of dicts")
    
    # Check if old format (results dict without 'results' key) or new format
    if 'params' in configs[0]:  # Old format: list of results dicts
        # Convert to new format
        new_configs = []
        for results in configs:
            inst_list = results['inst_list']
            dataset_name = results['dataset_name']
            for inst in inst_list:
                lam = 1.1 if inst == 1 else 1.8
                new_configs.append({
                    'results': results,
                    'inst': inst,
                    'label': f"{dataset_name} {lam} μm"
                })
        configs = new_configs
    
    n_traces = len(configs)
    
    # Set up default colors, markers, linestyles
    if use_cmap:
        cmap = plt.get_cmap(cmap_name)
        colors = cmap(np.linspace(0.3, 1.0, n_traces))
    else:
        colors = ['C'+str(i) for i in range(n_traces)]
    # if colors is None:
    #     colors = plt.cm.tab10(np.linspace(0, 0.9, n_traces))
    if markers is None:
        markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p', '*', 'X']
    if linestyles is None:
        linestyles = ['-', '--', '-.', ':']
    
    # Ensure lists are long enough
    markers = (markers * (n_traces // len(markers) + 1))[:n_traces]
    linestyles = (linestyles * (n_traces // len(linestyles) + 1))[:n_traces]
    
    # Create single panel plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Shade redshift bins of width dz=0.2 between z=0 and z=1
    # Use two contrasting pastel shades (alternating) so bins are easily distinguished
    zbin_edges = np.arange(0.0, 1.0 + 1e-9, 0.2)
    # Soft blue and soft peach
    shade_colors = ('#e8f4ff', '#fff3e6')
    for j in range(len(zbin_edges) - 1):
        z0 = zbin_edges[j]
        z1 = zbin_edges[j + 1]
        color_shade = shade_colors[j % 2]
        ax.axvspan(z0, z1, color=color_shade, alpha=0.22, zorder=0)
    
    # Calculate x-offsets for multiple traces at same z values
    x_offset_scale = 0.02  # offset scale relative to z bin width
    
    # Plot chi-squared values
    for i, config in enumerate(configs):
        results = config['results']
        inst = config.get('inst', None)
        label = config['label']
        color = colors[i]
        # color = config.get('color', colors[i])
        marker = config.get('marker', markers[i])
        linestyle = config.get('linestyle', linestyles[i])
        
        z_centers = results['z_centers']
        inst_list = results['inst_list']
        
        # Calculate x-offset for this trace
        # Center offsets around 0
        x_offset = (i - (n_traces - 1) / 2) * x_offset_scale
        
        # Determine which instrument index to use
        if inst is not None:
            # Find index of this instrument in inst_list
            try:
                inst_idx = list(inst_list).index(inst)
            except ValueError:
                print(f"Warning: inst={inst} not found in results, skipping")
                continue
            insts_to_plot = [(inst_idx, inst)]
        else:
            # Plot all instruments
            insts_to_plot = list(enumerate(inst_list))
        
        for inst_idx, inst_val in insts_to_plot:
            # Get chi-squared values
            if plot_reduced:
                chi2_values = results['reduced_chisq'][inst_idx, :]
                ylabel = r'Reduced $\chi^2$'
            else:
                chi2_values = results['chisq'][inst_idx, :]
                ylabel = r'$\chi^2$'
                
            # Plot chi-squared vs redshift
            ax.plot(z_centers + x_offset, chi2_values,
                   marker=marker, color=color, linestyle='none',
                   label=label, markersize=8, linewidth=2, alpha=0.8)
    
    # Add horizontal line at chi2 = 1 for reduced chi-squared
    if plot_reduced:
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, zorder=1)
        # ax.text(0.98, 1.02, r'$\chi^2_{\rm red} = 1$', transform=ax.transAxes, 
        #         ha='right', va='bottom', fontsize=10, alpha=0.7)
    
    ax.set_xlabel('Redshift', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(fontsize=10, loc='upper left', ncol=legend_ncol, bbox_to_anchor=bbox_to_anchor)
    ax.grid(alpha=0.3)
    
    if ylim_chi2 is not None:
        ax.set_ylim(ylim_chi2)
    ax.set_xlim(0, 1.0)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved chi2 comparison plot to: {save_path}")
    
    return fig


def plot_chi2_comparison_by_instrument(configs, inst_list=(1, 2), colors=None, markers=None,
														linestyles=None, figsize=(9, 3.6),
														save_path=None, legend_ncol=2, ylim_chi2=None,
														bbox_to_anchor=(0.5, 1.03), plot_reduced=True,
														use_cmap=False, cmap_name='Blues',
														x_offset_scale=0.02):
	"""Two-panel chi2 comparison with one panel per instrument."""
	if len(configs) > 0 and not isinstance(configs[0], dict):
		raise ValueError("configs must be a list of dicts")

	if 'params' in configs[0]:
		new_configs = []
		for results in configs:
			dataset_name = results.get('dataset_name', 'dataset')
			new_configs.append({
				'results': results,
				'inst': None,
				'label': f"{dataset_name}",
			})
		configs = new_configs

	n_traces = len(configs)
	if use_cmap:
		cmap = plt.get_cmap(cmap_name)
		colors = cmap(np.linspace(0.3, 1.0, n_traces))
	else:
		colors = ['C' + str(i) for i in range(n_traces)]
	if markers is None:
		markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p', '*', 'X']
	if linestyles is None:
		linestyles = ['-', '--', '-.', ':']
	markers = (markers * (n_traces // len(markers) + 1))[:n_traces]
	linestyles = (linestyles * (n_traces // len(linestyles) + 1))[:n_traces]
	ylabel = r'Reduced $\chi^2$' if plot_reduced else r'$\chi^2$'

	n_cols = len(inst_list)
	fig, axes = plt.subplots(1, n_cols, figsize=figsize, sharex=True, sharey=True)
	if n_cols == 1:
		axes = [axes]

	zbin_edges = np.arange(0.0, 1.0 + 1e-9, 0.2)
	shade_colors = ('#e8f4ff', '#fff3e6')

	labeled_configs = set()
	for col, inst in enumerate(inst_list):
		ax = axes[col]
		for j in range(len(zbin_edges) - 1):
			z0 = zbin_edges[j]
			z1 = zbin_edges[j + 1]
			ax.axvspan(z0, z1, color=shade_colors[j % 2], alpha=0.22, zorder=0)

		for i, config in enumerate(configs):
			results = config['results']
			inst_sel = config.get('inst', None)
			if inst_sel is not None and inst_sel != inst:
				continue

			insts_available = list(results['inst_list'])
			if inst not in insts_available:
				continue
			inst_idx = insts_available.index(inst)

			z_centers = results['z_centers']
			x_offset = (i - (n_traces - 1) / 2) * x_offset_scale
			color = config.get('color', colors[i])
			marker = config.get('marker', markers[i])
			label = config.get('label', None)
			label = label if (col == 0 and i not in labeled_configs) else None
			if label is not None:
				labeled_configs.add(i)

			if plot_reduced:
				chi2_values = results['reduced_chisq'][inst_idx, :]
				ylabel = r'Reduced $\chi^2$'
			else:
				chi2_values = results['chisq'][inst_idx, :]
				ylabel = r'$\chi^2$'

			ax.plot(
				z_centers + x_offset, chi2_values,
				marker=marker, color=color, linestyle='none',
				label=label, markersize=8, linewidth=2, alpha=0.8,
			)

		if plot_reduced:
			ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, zorder=1)
		ax.set_xlim(0, 1.0)
		ax.grid(alpha=0.3)
		ax.tick_params(labelsize=12)
		ax.text(0.04, 0.94, f"CIBER {1.1 if inst == 1 else 1.8} um",
					transform=ax.transAxes, fontsize=12,
					va='top', ha='left')

	if ylim_chi2 is not None:
		for ax in axes:
			ax.set_ylim(ylim_chi2)

	axes[0].set_ylabel(ylabel, fontsize=14)
	fig.supxlabel('Redshift', fontsize=14)

	handles, labels = axes[0].get_legend_handles_labels()
	if handles:
		fig.legend(handles, labels, fontsize=10, loc='upper center',
					ncol=legend_ncol, bbox_to_anchor=bbox_to_anchor)

	fig.tight_layout()
	fig.subplots_adjust(top=0.86)

	if save_path:
		plt.savefig(save_path, dpi=200, bbox_inches='tight')
		print(f"✓ Saved chi2 comparison plot to: {save_path}")

	return fig


def plot_cross_fit_components_from_file(npz_path, zbinedges, inst_list=[1, 2],
                                         ihl_template_path='ihl_templates/',
                                         lams=[1.1, 1.8], cat='DESILS',
                                         figsize=(8, 12), ell_range=[100, 100000],
                                         ylim=[1e-3, 5e2], save_path=None,
                                         show_data=True, organize_by='inst', 
                                         cmap_name='jet'):
    """
    Load cross-spectrum fit results from .npz file and plot model components with 1σ uncertainties.
    
    Creates a 3-row figure showing:
    - Top row: Total model (2h + 1h + shot) with 1σ uncertainty band
    - Middle row: 2-halo component only with 1σ uncertainty band
    - Bottom row: 1-halo component only with 1σ uncertainty band
    
    Each column corresponds to an instrument (if organize_by='inst') or redshift bin (if organize_by='zbin').
    
    Parameters same as before...
    """
    from ciber.theory.cross_ps_parametric_model import (
        CrossPowerSpectrumModel, load_fit_results_npz, load_ihl_template_for_zbin
    )
    
    # Load results
    print(f"Loading fit results from {npz_path}...")
    results = load_fit_results_npz(npz_path)
    
    # Extract info
    params_array = results['params']  # (n_inst, n_zbin, n_params)
    params_err_array = results.get('params_err', None)  # (n_inst, n_zbin, n_params) or None
    n_inst, n_zbin, n_params = params_array.shape
    use_ihl_templates = results.get('use_ihl_templates', False)
    use_powerlaw_2h = results.get('use_powerlaw_2h', True)
    alpha_2h_fixed = results.get('alpha_2h_fixed', 0.0)
    use_lorentzian_1h = results.get('use_lorentzian_1h', False)
    template_names = results.get('template_names', [])
    
    print(f"Model type: {'IHL templates' if use_ihl_templates else 'Phenomenological'}")
    print(f"Instruments: {n_inst}, Redshift bins: {n_zbin}, Parameters: {n_params}")
    print(f"Uncertainty bands: {'Available' if params_err_array is not None else 'Not available'}")
    
    # Set up 3-row figure
    if organize_by == 'inst':
        fig, axes = plt.subplots(3, n_inst, figsize=figsize, sharex=True, sharey='row')
        if n_inst == 1:
            axes = axes.reshape(3, 1)
    else:
        fig, axes = plt.subplots(3, n_zbin, figsize=figsize, sharex=True, sharey='row')
        if n_zbin == 1:
            axes = axes.reshape(3, 1)
    
    # Generate smooth ell array
    ell_model = np.logspace(np.log10(ell_range[0]), np.log10(ell_range[1]), 500)
    
    # Define colors
    cmap = plt.get_cmap(cmap_name)
    z_colors = cmap(np.linspace(0.1, 0.9, n_zbin))
    
    # Dummy lb array
    lb_dummy = np.logspace(2, 5, 100)
    
    # Iterate through combinations
    for inst_idx, inst in enumerate(inst_list[:n_inst]):
        for zidx in range(n_zbin):
            zcen = 0.5 * (zbinedges[zidx] + zbinedges[zidx+1])
            
            # Get parameters and errors
            params = params_array[inst_idx, zidx, :]
            params_err = params_err_array[inst_idx, zidx, :] if params_err_array is not None else None
            
            # Determine column
            col_idx = inst_idx if organize_by == 'inst' else zidx
            ax_total = axes[0, col_idx]
            ax_2h = axes[1, col_idx]
            ax_1h = axes[2, col_idx]
            
            # Reconstruct model
            if use_ihl_templates:
                templates, _, _ = load_ihl_template_for_zbin(
                    ihl_template_path, zbinedges, zidx, slopes=[1.0]
                )
                
                model = CrossPowerSpectrumModel(
                    lb_dummy,
                    use_powerlaw_2h=use_powerlaw_2h,
                    alpha_2h_fixed=alpha_2h_fixed
                )
                
                components = model.model_components_with_ihl_templates(
                    ell_model, params, templates, template_names
                )
                
                components_plot = {
                    'two_halo': components['two_halo'],
                    'shot_noise': components['shot_noise'],
                    'total': components['total']
                }
                if len(template_names) == 1:
                    components_plot['one_halo'] = components[f'ihl_{template_names[0]}']
                else:
                    components_plot['one_halo'] = components['one_halo_total']
                
                # Compute uncertainty bands for IHL case
                uncertainty_bands = None
                if params_err is not None and not np.any(np.isnan(params_err)):
                    # 2-halo bounds
                    if use_powerlaw_2h:
                        dl_2h_upper = model.powerlaw_2h_component(ell_model, params[0] + params_err[0], alpha_2h_fixed)
                        dl_2h_lower = model.powerlaw_2h_component(ell_model, max(0, params[0] - params_err[0]), alpha_2h_fixed)
                    else:
                        pf = ell_model * (ell_model + 1) / (2 * np.pi)
                        dl_2h_upper = (params[0] + params_err[0]) * pf * np.interp(ell_model, model.lb, model.cl_2h_pred)
                        dl_2h_lower = max(0, params[0] - params_err[0]) * pf * np.interp(ell_model, model.lb, model.cl_2h_pred)
                    
                    # IHL template bounds
                    dl_1h_upper = np.zeros_like(ell_model)
                    dl_1h_lower = np.zeros_like(ell_model)
                    for i, template_name in enumerate(template_names):
                        template = templates[template_name]
                        dl_1h_upper += model.ihl_template_component(ell_model, params[i+1] + params_err[i+1],
                                                                    template['ell'], template['dl'])
                        dl_1h_lower += model.ihl_template_component(ell_model, max(0, params[i+1] - params_err[i+1]),
                                                                    template['ell'], template['dl'])
                    
                    # Shot noise bounds
                    dl_shot_upper = model.shot_noise_component(ell_model, params[-1] + params_err[-1])
                    dl_shot_lower = model.shot_noise_component(ell_model, max(0, params[-1] - params_err[-1]))
                    
                    # Total (simple addition for now)
                    dl_total_upper = dl_2h_upper + dl_1h_upper + dl_shot_upper
                    dl_total_lower = dl_2h_lower + dl_1h_lower + dl_shot_lower
                    
                    uncertainty_bands = {
                        'two_halo': (dl_2h_lower, dl_2h_upper),
                        'one_halo': (dl_1h_lower, dl_1h_upper),
                        'shot_noise': (dl_shot_lower, dl_shot_upper),
                        'total': (dl_total_lower, dl_total_upper)
                    }
                    
            else:
                # Phenomenological model
                model = CrossPowerSpectrumModel(
                    lb_dummy,
                    use_powerlaw_2h=use_powerlaw_2h,
                    alpha_2h_fixed=alpha_2h_fixed,
                    use_lorentzian_1h=use_lorentzian_1h
                )
                
                components_plot = model.model_components(ell_model, *params[:5])
                
                # Compute uncertainty bands for parametric case
                uncertainty_bands = None
                if params_err is not None and not np.any(np.isnan(params_err)):
                    # Similar to IHL case but for parametric components
                    if use_powerlaw_2h:
                        dl_2h_upper = model.powerlaw_2h_component(ell_model, params[0] + params_err[0], alpha_2h_fixed)
                        dl_2h_lower = model.powerlaw_2h_component(ell_model, max(0, params[0] - params_err[0]), alpha_2h_fixed)
                    else:
                        pf = ell_model * (ell_model + 1) / (2 * np.pi)
                        dl_2h_upper = (params[0] + params_err[0]) * pf * np.interp(ell_model, model.lb, model.cl_2h_pred)
                        dl_2h_lower = max(0, params[0] - params_err[0]) * pf * np.interp(ell_model, model.lb, model.cl_2h_pred)
                    
                    if use_lorentzian_1h:
                        dl_1h_upper = model.lorentzian_component(ell_model, params[1] + params_err[1], params[2], params[3])
                        dl_1h_lower = model.lorentzian_component(ell_model, max(0, params[1] - params_err[1]), params[2], params[3])
                    else:
                        dl_1h_upper = model.lognormal_component(ell_model, params[1] + params_err[1], params[2], params[3])
                        dl_1h_lower = model.lognormal_component(ell_model, max(0, params[1] - params_err[1]), params[2], params[3])
                    
                    dl_shot_upper = model.shot_noise_component(ell_model, params[4] + params_err[4])
                    dl_shot_lower = model.shot_noise_component(ell_model, max(0, params[4] - params_err[4]))
                    
                    dl_total_upper = dl_2h_upper + dl_1h_upper + dl_shot_upper
                    dl_total_lower = dl_2h_lower + dl_1h_lower + dl_shot_lower
                    
                    uncertainty_bands = {
                        'two_halo': (dl_2h_lower, dl_2h_upper),
                        'one_halo': (dl_1h_lower, dl_1h_upper),
                        'shot_noise': (dl_shot_lower, dl_shot_upper),
                        'total': (dl_total_lower, dl_total_upper)
                    }
            
            # Set color
            color = z_colors[zidx] if organize_by == 'inst' else f'C{inst_idx}'
#             label_suffix = f'z={zcen:.2f}' if organize_by == 'inst' else f'TM{inst}'
            
            label = str(zbinedges[zidx])+'$<z<$'+str(zbinedges[zidx+1])
            
            # Row 1: Total with all components and uncertainty band
            ax_total.plot(ell_model, components_plot['total'], '-',
                         color=color, linewidth=2.5, label=label, alpha=0.8, zorder=4)
            if uncertainty_bands is not None:
                ax_total.fill_between(ell_model, uncertainty_bands['total'][0], uncertainty_bands['total'][1],
                                     color=color, alpha=0.2, zorder=3)
            
            # Row 2: 2-halo with uncertainty band
            ax_2h.plot(ell_model, components_plot['two_halo'], '-',
                      color=color, linewidth=2.5, alpha=0.8, zorder=4)
            if uncertainty_bands is not None:
                ax_2h.fill_between(ell_model, uncertainty_bands['two_halo'][0], uncertainty_bands['two_halo'][1],
                                  color=color, alpha=0.2, zorder=3)
            
            # Row 3: 1-halo with uncertainty band
            ax_1h.plot(ell_model, components_plot['one_halo'], '-',
                      color=color, linewidth=2.5, alpha=0.8, zorder=4)
            if uncertainty_bands is not None:
                ax_1h.fill_between(ell_model, uncertainty_bands['one_halo'][0], uncertainty_bands['one_halo'][1],
                                  color=color, alpha=0.2, zorder=3)
            
            # Plot data on all panels (faded for reference)
            if show_data and 'data_dl' in results and 'lb_fit' in results:
                lb_data = results['lb_fit'][inst_idx, zidx]
                dl_data = results['data_dl'][inst_idx, zidx]
                dlerr_data = results.get('data_dlerr', [[None]*n_zbin]*n_inst)[inst_idx, zidx]
                
                if lb_data is not None and dl_data is not None:
                    plot_kwargs = {'fmt': 'o', 'color': color, 'markersize': 3,
                                  'capsize': 2, 'alpha': 0.3, 'zorder': 10}
                    
                    if dlerr_data is not None:
                        ax_total.errorbar(lb_data, dl_data, yerr=dlerr_data, **plot_kwargs)
                        ax_2h.errorbar(lb_data, dl_data, yerr=dlerr_data, **plot_kwargs)
                        ax_1h.errorbar(lb_data, dl_data, yerr=dlerr_data, **plot_kwargs)
                    else:
                        ax_total.plot(lb_data, dl_data, 'o', color=color, markersize=3, alpha=0.3, zorder=10)
                        ax_2h.plot(lb_data, dl_data, 'o', color=color, markersize=3, alpha=0.3, zorder=10)
                        ax_1h.plot(lb_data, dl_data, 'o', color=color, markersize=3, alpha=0.3, zorder=10)
    
    # Format axes
    n_cols = n_inst if organize_by == 'inst' else n_zbin
    for col_idx in range(n_cols):
        ax_total = axes[0, col_idx]
        ax_2h = axes[1, col_idx]
        ax_1h = axes[2, col_idx]
        
        textxpos = 130
        textypos = 50
        text_fs = 14
        
              
        ax_total.text(textxpos, textypos, 'Total (2h+1h+shot)', fontsize=text_fs)
        ax_2h.text(textxpos, textypos, 'Two-halo', fontsize=text_fs)
        ax_1h.text(textxpos, textypos, 'One-halo', fontsize=text_fs)

        if col_idx==0:
  
            ax_total.legend(fontsize=12, ncol=3, loc=2, bbox_to_anchor=[0., 1.5])
        
        # Titles on top row
        if organize_by == 'inst':
            inst_val = inst_list[col_idx]
            lam = lams[col_idx]
            ax_total.set_title(f'{cat} × CIBER {lam} $\\mu$m', fontsize=14)
        else:
            ax_total.set_title(f'{zbinedges[col_idx]:.1f} < z < {zbinedges[col_idx+1]:.1f}', fontsize=13)
        
        # Y-labels on leftmost column
        if col_idx == 0:
            ax_total.set_ylabel(r'$D_\ell$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=13)
            ax_2h.set_ylabel(r'$D_\ell^{2h}$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=13)
            ax_1h.set_ylabel(r'$D_\ell^{1h}$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=13)
        
        # X-label on bottom row
        ax_1h.set_xlabel(r'Multipole $\ell$', fontsize=13)
        
        # Common formatting
        for ax in [ax_total, ax_2h, ax_1h]:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(ell_range)
            ax.set_ylim(ylim)
            ax.grid(alpha=0.3)
            
            ax.tick_params(labelsize=11)
    
    plt.subplots_adjust(wspace=0, hspace=0.05)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    return fig, axes



def mini_proc_clav(all_cl, all_clerr, lb, startidx, endidx, mode='auto', fmask=0.7, model_cl_for_knox=None, uniform_weight_ell=None):
	"""
	Process field-averaged power spectra and add Knox errors.
	
	Parameters
	----------
	all_cl : array_like
		Per-field power spectra
	all_clerr : array_like
		Per-field uncertainties (measurement only, without Knox)
	lb : array_like
		Multipole bin centers
	startidx : int
		Starting index for analysis
	endidx : int
		Ending index for analysis
	mode : str, optional
		'auto' or 'cross' spectrum mode
	fmask : float, optional
		Mask fraction per field
	model_cl_for_knox : array_like, optional
		Model C_ell to use for Knox calculation instead of data.
		If provided, Knox errors are computed from this model rather than
		from the measured field-averaged spectrum. This avoids bias from
		cosmic variance in the data. Default is None (use data-based Knox).
	uniform_weight_ell : float, optional
		If provided, use uniform field weighting (instead of inverse-variance) 
		above this multipole threshold. Default: None (use inverse-variance for all).
	
	Returns
	-------
	pf : array_like
		Prefactor for D_ell conversion
	posmask : array_like
		Boolean mask for positive values
	negmask : array_like  
		Boolean mask for negative values
	fieldav_cl : array_like
		Field-averaged C_ell
	fieldav_clerr : array_like
		Field-averaged uncertainties including Knox cosmic variance
	"""
	
	cbps = CIBER_PS_pipeline()
	pf = lb*(lb+1)/(2*np.pi)
	lbmask = (lb >= lb[startidx])*(lb < lb[endidx])

	nfield = len(all_cl)
	
	if len(all_cl) > 1:
		# Standard error-weighted field averaging
		fieldav_cl, fieldav_clerr,\
			_, _ = compute_field_averaged_power_spectrum(all_cl.copy(), per_field_dcls=all_clerr.copy())
		
		# Apply uniform weighting above the threshold if requested
		if uniform_weight_ell is not None:
			uniform_mask = lb >= uniform_weight_ell
			if np.any(uniform_mask):
				# Re-average the bins above the threshold with uniform weighting
				fieldav_cl_uniform, fieldav_clerr_uniform, _, _ = compute_field_averaged_power_spectrum(
					all_cl.copy(), per_field_dcls=all_clerr.copy(), weight_mode='uniform'
				)
				fieldav_cl[uniform_mask] = fieldav_cl_uniform[uniform_mask]
				fieldav_clerr[uniform_mask] = fieldav_clerr_uniform[uniform_mask]
	else:
		fieldav_cl, fieldav_clerr = all_cl[0], all_clerr[0]
	
	if mode=='auto':
		num = 2.
	elif mode=='cross':
		num = 1.
		
	if mode=='auto':
		gal_knox_errors = np.sqrt(num/((2*lb+1)*cbps.Mkk_obj.delta_ell))
		fsky = nfield*fmask*2*2/(41253.)    
		gal_knox_errors /= np.sqrt(fsky)
		
		# Use model for Knox if provided, otherwise use data
		if model_cl_for_knox is not None:
			gal_knox_errors *= np.abs(model_cl_for_knox)
		else:
			gal_knox_errors *= np.abs(fieldav_cl)
			
		fieldav_clerr = np.sqrt(gal_knox_errors**2 + fieldav_clerr**2)

	posmask = lbmask*(fieldav_cl > 0)
	negmask = lbmask*(fieldav_cl < 0)
	
	return pf, posmask, negmask, fieldav_cl, fieldav_clerr

def estimate_cross_uncertainties(lb, clx, clx_err, clI_auto, clg_auto, nfield, startidx=2, endidx=-1, fmask=0.7):
	
	# clx_err includes N_ell^I x (C_ell^g + 1/n)

	cbps = CIBER_PS_pipeline()
	dclx_sq = np.ones_like(lb)
	lbmask = (lb >= lb[startidx])*(lb < lb[endidx])
	
	nmode_inv = 1./((2*lb+1)*cbps.Mkk_obj.delta_ell)
	fsky = nfield*fmask*2*2/(41253.) 
	nmode_inv /= fsky
	
	nbar_inv = np.mean(clg_auto[-4:endidx])

	if np.ndim(clI_auto) > 0:
		nmask = np.count_nonzero(lbmask)
		if len(clI_auto) == len(lb):
			clI_auto_use = clI_auto[lbmask]
		elif len(clI_auto) == nmask:
			clI_auto_use = clI_auto
		else:
			clI_auto_use = np.zeros(nmask)
			ncopy = min(nmask, len(clI_auto))
			clI_auto_use[:ncopy] = clI_auto[:ncopy]
	else:
		clI_auto_use = clI_auto
	
	# cl_terms = clx[lbmask]**2 + np.abs(clI_auto*clg_auto[lbmask]) + clx_err[lbmask]**2 + clI_auto*nbar_inv
	cl_terms = clx[lbmask]**2 + np.abs(clI_auto_use*clg_auto[lbmask]) + clI_auto_use*nbar_inv
	

	dclx_sq_A = nmode_inv[lbmask]*cl_terms 

	dclx_sq[lbmask] = dclx_sq_A + clx_err[lbmask]**2 # since noise x clg computed from MC realizations don't normalize by Nmodes, already done.
	
	return np.sqrt(dclx_sq)

def compute_rl_ciber_gal(addstr, inst_list=[1, 2], catname='LS', gal_label='LS ($z<1$)', startidx=2, endidx=-1, tl_pix_correct=True, 
						ifield_use=8):
	
	bandstr_list = ['J', 'H']
	lams = [1.1, 1.8]
	
	all_r_ell, all_r_ell_unc = [], []
	
	all_auto_cross_data = []
	
	keys = ['lb', 'all_cl_gal', 'all_clerr_gal', 'all_cl_cross', 'all_clerr_cross', 'ifield_list_use']
	
	for idx, inst in enumerate(inst_list):
				
		print(catname, addstr)
		cgps_file = load_ciber_gal_ps(inst, catname, addstr=addstr)
		
		lb, all_cl_gal, all_clerr_gal, \
			all_cl_cross, all_clerr_cross, ifield_list_use = [cgps_file[key] for key in keys]
		
		pf, posmask_auto, negmask_auto, fieldav_cl_gal_unw, fieldav_clerr_gal = mini_proc_clav(all_cl_gal, all_clerr_gal, lb, startidx, endidx, mode='auto')
		pf, posmask, negmask, fieldav_cl_cross_unw, fieldav_clerr_cross = mini_proc_clav(all_cl_cross, all_clerr_cross, lb, startidx, endidx, mode='cross')

		# # Prefer in-situ auto-spectrum from cross-product file if available
		# if 'all_cl_ciber_auto_inplace' in cgps_file.files:
		# 	all_cl_ciber_auto_inplace = cgps_file['all_cl_ciber_auto_inplace']

		# 	# Field-average in-situ auto (already noise-subtracted, already on full lb grid)
		# 	cl_auto_full = np.nanmean(all_cl_ciber_auto_inplace, axis=0)
		# 	cl_auto = cl_auto_full[startidx:endidx]
		# 	clerr_auto = np.zeros(len(cl_auto))  # Placeholder; not used in ratio calculation
		# elif catname=='HSC':
		# 	bandstr = bandstr_list[idx]
		# 	mag_lim = 16.0

		# 	observed_run_name = 'observed_'+bandstr+'lt'+str(mag_lim)+'_072424_quadoff_grad_fcsub_order2'

		# 	fpath = config.ciber_basepath+'data/input_recovered_ps/111323/TM'+str(inst)+'/'+observed_run_name+'/input_recovered_ps_estFF_simidx0.npz'

		# 	clfile = np.load(fpath)
		# 	# print([k for k in clfile.keys()])

		# 	lb_auto, cl_auto_all, clerr_auto_all = [clfile[key] for key in ['lb', 'recovered_ps_est_nofluc', 'recovered_dcl']]

		# 	# print('cl auto', cl_auto_all[-1])
		# 	cl_auto = cl_auto_all[-1,startidx:endidx]
		# 	clerr_auto = clerr_auto_all[-1, startidx:endidx]

		# else:
		ciber_auto = _load_ciber_auto_file(bandstr_list[idx])

		lb_auto, cl_auto, clerr_auto = [ciber_auto[key] for key in ['lb', 'fieldav_cl', 'fieldav_clerr']]



		# ifield_list_use loaded from file above
		ifield_list_full = [4, 5, 6, 7, 8]
		
		# Compute flat field bias correction for each field
		cbps = CIBER_PS_pipeline()
		mean_norms = [cbps.zl_levels_ciber_fields[inst][cbps.ciber_field_dict[ifield]] 
					for ifield in ifield_list_full]
		weights_ff = cbps.compute_ff_weights(inst, mean_norms, ifield_list_full, photon_noise=True)
		ff_bias_factors = compute_ff_bias(mean_norms, weights=weights_ff)
		
		# Compute per-field uncertainties with FF bias correction
		perf_clerr_cross = np.zeros_like(all_clerr_cross)
		for fieldidx, ifield in enumerate(ifield_list_use):
			idx_full = ifield_list_full.index(ifield)
			perf_clerr_cross[fieldidx] = estimate_cross_uncertainties(
				lb, fieldav_cl_cross_unw, all_clerr_cross[fieldidx],
				cl_auto*ff_bias_factors[idx_full], fieldav_cl_gal_unw, 1, startidx=2, endidx=-1
			)
		
		# Compute properly weighted field average
		if len(ifield_list_use) == 1:
			fieldav_cl_cross = fieldav_cl_cross_unw
			fieldav_clerr_cross_ana = perf_clerr_cross[0]
		else:
			fieldav_cl_cross, fieldav_clerr_cross_ana, _, _ = compute_field_averaged_power_spectrum(
				all_cl_cross.copy(), per_field_dcls=perf_clerr_cross.copy()
			)
		
		# Apply Knox errors to galaxy auto
		fieldav_cl_gal = fieldav_cl_gal_unw
		nfield = len(ifield_list_use)
		gal_knox_errors = np.sqrt(2./((2*lb+1)*cbps.Mkk_obj.delta_ell))
		fsky = 2*2/(41253.) * nfield
		gal_knox_errors /= np.sqrt(fsky)
		gal_knox_errors *= np.abs(fieldav_cl_gal)
		fieldav_clerr_gal = np.sqrt(gal_knox_errors**2 + fieldav_clerr_gal**2)
		
		if tl_pix_correct:

			tl_pix = np.load('data/fluctuation_data/transfer_function/tl_clx_pix_TM'+str(inst)+'_ifield'+str(ifield_use)+'.npz')['tl_clx_pix']
			fieldav_cl_cross /= tl_pix 
			fieldav_clerr_cross_ana /= tl_pix


		r_ell = fieldav_cl_cross[startidx:endidx]/np.sqrt((cl_auto)*fieldav_cl_gal[startidx:endidx])

		r_ell_unc = compute_rlx_unc_comps(cl_auto, fieldav_cl_gal[startidx:endidx], fieldav_cl_cross[startidx:endidx], \
							clerr_auto, fieldav_clerr_gal[startidx:endidx], fieldav_clerr_cross_ana[startidx:endidx])


		all_r_ell.append(r_ell)
		all_r_ell_unc.append(r_ell_unc)

				
		auto_cross_data = auto_cross_cl(lb=lb, pf=pf, posmask_auto=posmask_auto, negmask_auto=negmask_auto,\
						fieldav_cl_gal=fieldav_cl_gal, fieldav_clerr_gal=fieldav_clerr_gal,\
							posmask=posmask, negmask=negmask, fieldav_cl_cross=fieldav_cl_cross, fieldav_clerr_cross=fieldav_clerr_cross_ana, r_ell=r_ell, r_ell_unc=r_ell_unc, \
							ciber_auto_cl=cl_auto, ciber_auto_clerr=clerr_auto)
		
		all_auto_cross_data.append(auto_cross_data)
		
	return all_auto_cross_data

def plot_gal_and_ciber_auto(all_acdat, pred_fpaths=None,
							colors=['b', 'r'],
							xlim=[250, 1.1e5],
							ylims_gal=[[8e-4, 5e1], [8e-3, 2e2]],
							gal_labels=None,
							gal_labels_filenames=None,
							band_labels=None,
							startidx=2, endidx=-1,
							capsize=3, markersize=3,
							figsize=(10, 4.5), lab_fs=12, title_fs=12, legend_fs=10,
							pred_alpha=0.7, spacer_and_ciber_auto=[0.35, 1.2],
							tl_pix_correct=True, ifield_use=6, include_ciber_auto=True,
							tl_pix_template='data/fluctuation_data/transfer_function/tl_clx_pix_TM{inst}_ifield{ifield}.npz',
							bias_values=None,
							apply_satellite_correction=True,
							nl_corrections=False,
							show_linear_pred=True, z_eff_values=None, 
							include_1h_pred=True,
							onehalo_output_dir=None,
							onehalo_fsat_model='double'):

	n_gal = len(all_acdat)  # number of galaxy catalogs
	if gal_labels is None:
		gal_labels = [f"Catalog {i+1}" for i in range(n_gal)]
	if gal_labels_filenames is None:
		gal_labels_filenames = gal_labels
	if band_labels is None:
		band_labels = ['J', 'H']

	fig = plt.figure(figsize=figsize)

	if include_ciber_auto:
		widths = [1]*n_gal + spacer_and_ciber_auto  # small spacer column + CIBER panel
		gs = GridSpec(2, n_gal + 2, width_ratios=widths, wspace=0.0, hspace=0)
	else:
		gs = GridSpec(2, n_gal, wspace=0.0, hspace=0)

	if show_linear_pred:
		from ciber.theory.cross_ps_parametric_model import _compute_linear_2h_templates_per_zbin
		zbinedges = np.array([0.0, 1.0])
		dl_2h_lin_per_zbin_res = _compute_linear_2h_templates_per_zbin(zbinedges, 1e5, verbose=False)[0]
		lb_lin, dl_2h_lin = dl_2h_lin_per_zbin_res[0], dl_2h_lin_per_zbin_res[1]
		dl_2h_lin /= np.max(dl_2h_lin[lb_lin > 304])
		print('dl 2h lin per zbin has shape', dl_2h_lin.shape)
	

	# Left block: galaxy autos and crosses
	ax_gal = np.empty((2, n_gal), dtype=object)
	for row in range(2):
		for col in range(n_gal):
			ax_gal[row, col] = fig.add_subplot(gs[row, col])
			ax_gal[row, col].set_xscale('log')
			ax_gal[row, col].set_yscale('log')
			ax_gal[row, col].set_xlim(xlim)
			ax_gal[row, col].grid(alpha=0.3)

	# Loop over galaxy catalogs
	
	intensity_auto_preds = []
	all_snr_lt2000 = np.zeros((2, len(all_acdat[0])))  # Initialize the array before the loop over galaxy catalogs	
	
	# loop over catalogs
	for col, cat_acdats in enumerate(all_acdat):
			
		# loop over bands
		for idx_band, acdat in enumerate(cat_acdats):

			if pred_fpaths is not None:
			
				jmock_pred = np.load(pred_fpaths[col][idx_band])

				lb_pred, gal_auto, intensity_auto, cross = [jmock_pred[key] for key in ['lb', 'gal_auto', 'intensity_auto_full', 'cross']]
				pf_pred = lb_pred*(lb_pred+1)/(2*np.pi)
				
				if col==0:
					intensity_auto_preds.append(intensity_auto)

			if tl_pix_correct:
				tl_pix_path = tl_pix_template.format(inst=idx_band+1, ifield=ifield_use)
				tl_pix = np.load(tl_pix_path)['tl_clx_pix']
			else:
				tl_pix = np.ones_like(lb_pred)

			cross /= tl_pix
			acdat.fieldav_cl_cross /= tl_pix
			acdat.fieldav_clerr_cross /= tl_pix
			
			pf, lb = acdat.pf, acdat.lb
			# Use separate lb/pf for galaxy auto if a larger-footprint override was applied
			lb_auto = getattr(acdat, 'lb_gal_auto', lb)
			pf_auto = getattr(acdat, 'pf_gal_auto', pf)
			color = colors[idx_band]

			fpath_cl = 'data/ciber_gal_cross_cls/cl_CIBER_TM'+str(idx_band+1)+'_'+gal_labels_filenames[col]+'.npz'
			print('saving to ', fpath_cl)
			np.savez(fpath_cl, lb=lb, cl=acdat.fieldav_cl_cross, clerr=acdat.fieldav_clerr_cross,
				inst=idx_band+1, gal_label=gal_labels_filenames[col])

			fpath_cl_gal = 'data/ciber_gal_cross_cls/cl_TM'+str(idx_band+1)+'_'+gal_labels_filenames[col]+'.npz'
			print('saving to ', fpath_cl_gal)
			np.savez(fpath_cl_gal, lb=lb_auto, cl=acdat.fieldav_cl_gal, clerr=acdat.fieldav_clerr_gal,
				inst=idx_band+1, gal_label=gal_labels_filenames[col])

			lmax = 4000
			snr_lt_2000 = np.sum((acdat.fieldav_cl_cross[acdat.lb < lmax] / acdat.fieldav_clerr_cross[acdat.lb < lmax])**2)**0.5
			print(f"Catalog {gal_labels[col]}, band {band_labels[idx_band]}: SNR for ell < {lmax} = {snr_lt_2000:.2f}")
			
			
			all_snr_lt2000[col, idx_band] = snr_lt_2000
			# Load one-halo predictions once for both rows (before galaxy auto section)
			oh_data_Ig = None
			oh_data_gg = None
			oh_data_II = None
			ell_1h = None

			if include_1h_pred and onehalo_output_dir is not None:
				# Determine bandstr from gal_labels_filenames
				if col == 0:  # LS/DESI
					bandstr_select = 'sdss_z'
				elif col == 1:  # HSC
					bandstr_select = 'hsc_i'
				else:
					bandstr_select = None

				if bandstr_select is not None:
					# Determine mag_cut from bandstr
					mag_cut_map = {'sdss_z': 22.0, 'hsc_i': 25.0}
					mag_cut = mag_cut_map.get(bandstr_select)

					if mag_cut is not None:
						# Load Ig cross, gg auto, and II intensity auto one-halo predictions
						oh_data_Ig = load_onehalo_spectrum(
							onehalo_output_dir, onehalo_fsat_model, bandstr_select,
							inst=idx_band+1, mag_min=18.0, mag_cut=mag_cut, z0=0.05, mode='Ig',
							generate_type='bulk'
						)
						oh_data_gg = load_onehalo_spectrum(
							onehalo_output_dir, onehalo_fsat_model, bandstr_select,
							inst=idx_band+1, mag_min=18.0, mag_cut=mag_cut, z0=0.05, mode='gg',
							generate_type='bulk'
						)

						if oh_data_Ig is not None:
							ell_1h = oh_data_Ig['ell_arr']

			if idx_band==0:
				# Row 0: galaxy auto
				ax_gal[0, col].errorbar(
					lb_auto[acdat.posmask_auto],
					(pf_auto * acdat.fieldav_cl_gal)[acdat.posmask_auto],
					yerr=(pf_auto * acdat.fieldav_clerr_gal)[acdat.posmask_auto],
					color='k', fmt='o', capsize=capsize, markersize=markersize,
					zorder=15, label=band_labels[idx_band]
				)
				ax_gal[0, col].errorbar(
					lb_auto[acdat.negmask_auto],
					np.abs(pf_auto * acdat.fieldav_cl_gal)[acdat.negmask_auto],
					yerr=(pf_auto * acdat.fieldav_clerr_gal)[acdat.negmask_auto],
					color='k', fmt='o', capsize=capsize, markersize=markersize,
					mfc='white', zorder=15
				)

			if pred_fpaths is not None:

				b_g = bias_values[col] if (bias_values is not None and col < len(bias_values)) else None

				if b_g is not None:
					ell_eval = np.geomspace(xlim[0], xlim[1], 300)
					# cross: rescale by b_g
					ell_s, dl_cross_lin = smooth_mock_cross_with_bias(
						pred_fpaths[col][idx_band], z_center=0.5, b_g=b_g, ell_eval=ell_eval)

					pf_s = ell_s * (ell_s + 1.0) / (2.0 * np.pi)
					# Apply NL correction if enabled
					dl_cross_s = dl_cross_lin.copy()

					dl_cross_s /= np.interp(ell_s, lb_pred, tl_pix)
					if not include_1h_pred:
						ax_gal[1, col].plot(ell_s, dl_cross_s, color=colors[idx_band],
											linestyle='solid', alpha=0.4, linewidth=2.5)

					#normalize dl_2h_lin_per_zbin to match the amplitude of D_ell from intensity auto preds at low ell (e.g., ell ~ 1000)
					low_ell_mask = (ell_s >= 300) & (ell_s <= 1000)
					c_ell_sn = dl_cross_s[-2] / pf_s[-2]  # shot noise amplitude from high ell
					dl_shot = c_ell_sn * pf_s

					pf_lin = lb_lin * (lb_lin + 1.0) / (2.0 * np.pi)
					dl_shot_lin = c_ell_sn * pf_lin

					dl_cross_s_shot_subtracted = ((dl_cross_s / pf_s) - c_ell_sn)*pf_s

					if low_ell_mask.any():
						mean_dl_auto_pred = np.mean(dl_cross_s[low_ell_mask])
						if mean_dl_auto_pred > 0:
							ax_gal[1, col].plot(lb_lin, mean_dl_auto_pred * dl_2h_lin, color=colors[idx_band], alpha=0.7,
								label=f'Two-halo clustering', linewidth=2.0, linestyle='dashdot')

					# Plot one-halo predictions if available
					if include_1h_pred and oh_data_Ig is not None and ell_1h is not None:
						dl_1h_Ig = oh_data_Ig['dl_spectrum']

						# Interpolate Ig 1h to ell_s grid for combination
						dl_1h_Ig_interp = np.interp(ell_s, ell_1h, dl_1h_Ig)

						# Plot isolated Ig 1h component
						ax_gal[1, col].plot(ell_1h, dl_1h_Ig, color=colors[idx_band],
							linestyle='dotted', alpha=0.6, linewidth=2.5,
							label=f'One-halo clustering')

						# Plot combined Ig 1h + cross
						dl_cross_1h_combined = dl_cross_s + dl_1h_Ig_interp

						sigma_arcsec = 2.1


						# Convert arcsec to radians for dimensionless ℓσ
						sigma_rad = sigma_arcsec * (1.0 / 3600.0) * (np.pi / 180.0)
						damp_fac = np.exp(-0.5 * (sigma_rad * ell_s)**2)

						ax_gal[1, col].plot(ell_s, dl_cross_1h_combined*damp_fac,
							color=colors[idx_band], linestyle='solid',
							alpha=0.5, linewidth=3.0, zorder=4,
							label=f'IGL prediction (2h+1h+Poiss.)')


						model_match_lb_data = np.interp(acdat.lb, ell_s, dl_cross_1h_combined)
						dlerr_cross = acdat.fieldav_clerr_cross * acdat.pf

						print('')
						# Compute tension between data and model for ell < 2000
						sigma_tension_data_model = np.sqrt(np.sum(((pf * acdat.fieldav_cl_cross)[acdat.lb < lmax] - model_match_lb_data[acdat.lb < lmax])**2 / dlerr_cross[acdat.lb < lmax]**2))
						print(f"Catalog {gal_labels[col]}, band {band_labels[idx_band]}: Tension between data and model for ell < {lmax} = {sigma_tension_data_model:.2f} sigma")


					# plot galaxy shot noise level as ell^2

					ax_gal[1, col].plot(ell_s, dl_shot, color=colors[idx_band], alpha=0.7, linestyle='solid', linewidth=1.0)
						# Also plot gg auto 1h if available (for reference/debugging)
					if oh_data_gg is not None:
						dl_1h_gg = oh_data_gg['dl_spectrum']
			

					hsc_color = "#E45DA8"
					colors_iglpred = ['C2', hsc_color]
					if idx_band == 0:
						# auto: two-halo rescaled by b_g^2, shot noise unchanged
						pred = np.load(pred_fpaths[col][idx_band])
						lb_p = np.asarray(pred['lb'], dtype=float)
						cl_p = np.asarray(pred['gal_auto'], dtype=float)
						pf_p = lb_p * (lb_p + 1.0) / (2.0 * np.pi)
						dl_p = pf_p * cl_p
						shot_mask = (lb_p >= 30000.) & (lb_p <= 80000.) & np.isfinite(dl_p)
						pf_shot = lb_p[shot_mask] * (lb_p[shot_mask] + 1.0) / (2.0 * np.pi)
						A_shot = float(np.nanmean(dl_p[shot_mask] / pf_shot)) if shot_mask.any() else 0.0
						twoh_mask = (lb_p <= 1000.) & np.isfinite(dl_p)
						A_2h = max(float(np.nanmean(dl_p[twoh_mask] - A_shot * pf_p[twoh_mask])), 0.0) if twoh_mask.any() else 0.0
						pf_e = ell_eval * (ell_eval + 1.0) / (2.0 * np.pi)
						dl_auto_lin = b_g**2 * A_2h + A_shot * pf_e
						dl_auto_s = dl_auto_lin.copy()

						lb_auto = getattr(acdat, 'lb_gal_auto', lb_p)

						# Add one-halo gg auto prediction if available
						if include_1h_pred and oh_data_gg is not None and ell_1h is not None:
							dl_1h_gg = oh_data_gg['dl_spectrum']
							dl_1h_gg_interp = np.interp(ell_eval, ell_1h, dl_1h_gg)
							# Combine 2h + 1h for galaxy auto
							dl_auto_s = dl_auto_lin + dl_1h_gg_interp

						ax_gal[0, col].plot(ell_eval, dl_auto_s,
						                    alpha=0.7, linewidth=3, color=colors_iglpred[col], zorder=5)

						#normalize dl_2h_lin_per_zbin to match the amplitude of D_ell from intensity auto preds at low ell (e.g., ell ~ 1000)
						low_ell_mask = (ell_eval >= 300) & (ell_eval <= 1000)
						if low_ell_mask.any():
							mean_dl_auto_pred = np.mean(dl_auto_s[low_ell_mask])
							if mean_dl_auto_pred > 0:
								ax_gal[0, col].plot(lb_lin, mean_dl_auto_pred * dl_2h_lin, color=colors_iglpred[col], alpha=0.7,
									label=f'Two-halo clustering', linewidth=2.0, linestyle='dashdot', zorder=2)

						# plot galaxy shot noise level as ell^2
						# Extract shot noise from 2h-only prediction (not combined with 1h)
						c_ell_sn = dl_auto_lin[-1] / pf_e[-1]  # shot noise amplitude from high ell
						dl_shot = c_ell_sn * pf_e
						ax_gal[0, col].plot(ell_eval, dl_shot, color=colors_iglpred[col], alpha=0.7, linestyle='solid', linewidth=1.0, zorder=1)
						
						ax_gal[0, col].plot(ell_1h, dl_1h_gg, color=colors_iglpred[col],
						linestyle='dotted', alpha=0.7, linewidth=2.0,
							label=f'One-halo clustering')		

				else:
					if idx_band == 0:
						ax_gal[0, col].plot(lb_pred, pf_pred*gal_auto, color='grey',
						                    linestyle='dotted', alpha=pred_alpha)
					ax_gal[1, col].plot(lb_pred, pf_pred*cross, color=colors[idx_band],
					                    linestyle='dotted', alpha=pred_alpha)


			# Row 1: cross
			ax_gal[1, col].errorbar(
				lb[acdat.posmask],
				(pf * acdat.fieldav_cl_cross)[acdat.posmask],
				yerr=(pf * acdat.fieldav_clerr_cross)[acdat.posmask],
				color=color, fmt='o', capsize=capsize, markersize=markersize,
				zorder=15
			)
			ax_gal[1, col].errorbar(
				lb[acdat.negmask],
				np.abs(pf * acdat.fieldav_cl_cross)[acdat.negmask],
				yerr=(pf * acdat.fieldav_clerr_cross)[acdat.negmask],
				color=color, fmt='o', capsize=capsize, markersize=markersize,
				mfc='white', zorder=15
			)

		
		
	# Hide y-ticks for galaxy columns > 0
	for row in range(2):
		for col in range(1, n_gal):
			ax_gal[row, col].tick_params(labelleft=False)

	# Hide x-ticks for top row
	for col in range(n_gal):
		ax_gal[0, col].tick_params(labelbottom=False)

	# Shared y-axis labels for the leftmost column, hide others
	for row in range(2):
		ax_gal[row, 0].set_ylabel([r'$D_\ell^{\rm gg}$', r'$D_\ell^{\rm Ig}$ [nW m$^{-2}$ sr$^{-1}]$'][row], fontsize=lab_fs)
		for col in range(1, n_gal):
			ax_gal[row, col].set_yticklabels([])  # hide ticks
			ax_gal[row, col].set_ylim(ylims_gal[row])

	# Apply y-limits to first col and match others
	for row in range(2):
		ax_gal[row, 0].set_ylim(ylims_gal[row])

	# Bottom x-labels for galaxy panels
	for col in range(n_gal):
		ax_gal[1, col].set_xlabel(r'$\ell$', fontsize=lab_fs)

	second_lines = ['$z_{\\rm AB}<22; z_{\\rm phot}<1$', '$18<i_{\\rm AB}<25; z_{\\rm phot}<1$']
	third_lines = ['$b_g(z_{\\rm eff}='+str(z_eff_values[0])+')='+str(bias_values[0])+'$', '$b_g(z_{\\rm eff}='+str(z_eff_values[1])+')='+str(bias_values[1])+'$', 
				'$b_g='+str(bias_values[0])+'$; $b_I=1$', '$b_g='+str(bias_values[1])+'$; $b_I=1$']

	snr_lines = ['SNR$_{\\ell<2000}=$'+str(np.round(np.sum((all_acdat[0][0].fieldav_cl_cross[all_acdat[0][0].lb < 2000] / all_acdat[0][0].fieldav_clerr_cross[all_acdat[0][0].lb < 2000])**2)**0.5, 1))+'$',
				'SNR$_{\\ell<2000}=$'+str(np.round(np.sum((all_acdat[1][0].fieldav_cl_cross[all_acdat[1][0].lb < 2000] / all_acdat[1][0].fieldav_clerr_cross[all_acdat[1][0].lb < 2000])**2)**0.5, 1))+'$']




	if n_gal >= 2:
		panel_labels = {
			(0, 0): 'DESI-LS auto\n'+second_lines[0]+'\n'+third_lines[0],
			(0, 1): 'HSC auto\n'+second_lines[1]+'\n'+third_lines[1],
			(1, 0): 'CIBER x DESI-LS'+'\n'+third_lines[2],
			(1, 1): 'CIBER x HSC'+'\n'+third_lines[3],
		}
		if n_gal >= 3:
			panel_labels[(0, 2)] = 'WISE auto'
			panel_labels[(1, 2)] = 'CIBER x WISE'

		# first draw the base labels
		for (row, col), txt in panel_labels.items():
			ax_gal[row, col].text(
				0.04, 0.93, txt,
				transform=ax_gal[row, col].transAxes,
				ha='left', va='top', fontsize=title_fs
			)

		# add per-band SNR lines below existing text in cross panels (row=1), colored by band
		for col in range(min(n_gal, len(all_snr_lt2000))):
			y0 = 0.75      # start below existing text block; tweak if needed
			dy = 0.08      # vertical spacing between SNR lines
			for idx_band, snr_val in enumerate(all_snr_lt2000[col]):
				ax_gal[1, col].text(
					0.04, y0 - idx_band*dy,
					rf'SNR$_{{\ell<2000}}$ = {snr_val:.1f}',
					transform=ax_gal[1, col].transAxes,
					ha='left', va='top',
					fontsize=title_fs,
					color=colors[idx_band],   # blue/red etc by band index
				)

	legend_handles = [
		ax_gal[0, 0].errorbar(
			[1.0],
			[1.0],
			yerr=[0.2],
			fmt='o',
			color='k',
			capsize=capsize,
			markersize=markersize + 1,
			linestyle='None',
			label='Galaxy auto',
		),
		ax_gal[0, 0].errorbar(
			[1.0],
			[1.0],
			yerr=[0.2],
			fmt='o',
			color=colors[0],
			capsize=capsize,
			markersize=markersize + 1,
			linestyle='None',
			label=f'CIBER {band_labels[0]}',
		),
		ax_gal[0, 0].errorbar(
			[1.0],
			[1.0],
			yerr=[0.2],
			fmt='o',
			color=colors[1],
			capsize=capsize,
			markersize=markersize + 1,
			linestyle='None',
			label=f'CIBER {band_labels[1]}',
		),
	]
	if pred_fpaths is not None:


		legend_handles.append(
			Line2D([0], [0], color='k', linestyle='solid', linewidth=1.0, label='Poisson level')
		)

		legend_handles.append(
			Line2D([0], [0], color='k', linestyle='dashdot', linewidth=2.0, label='Two-halo')
		)

		legend_handles.append(
			Line2D([0], [0], color='k', linestyle='dotted', linewidth=2.5, label='One-halo ($L\\propto M$)')
		)
		legend_handles.append(
			Line2D([0], [0], color='C2', linestyle='solid', linewidth=3, alpha=0.5, label='IGL prediction (2h+1h+P)')
		)

		# legend_handles.append(
		# 	Line2D([0], [0], color='k', linestyle='dashed', linewidth=1.2, label='Poisson fluctuations')
		# )


	# fig.subplots_adjust(top=0.82)
	leg = fig.legend(
		handles=legend_handles,
		loc='upper center',
		bbox_to_anchor=(0.5, 1.02),
		ncol=3,
		fontsize=legend_fs,
	)
	# leg.get_frame().set_linewidth(0.8)

	plt.show()
	return fig


def create_omnibus_plot(all_addstr=None, jmock_basedir = None,
						hsc_str='hsc_ilt25.0_0.0_z_1.0_wrandsub_wFFerr',
					ls_str='0.0_z_1.0_wrandsub_JHlt16_wFFerr',
					wise_str='unWISE_W1lt17p5_JHlt16_wFFerr',
					ylims_gal=[[5e-4, 1e2], [2e-3, 2e2]],
					figsize=(8, 6),
					tl_pix_correct=True,
					ifield_use=8,
					tl_pix_template='data/fluctuation_data/transfer_function/tl_clx_pix_TM{inst}_ifield{ifield}.npz',
					include_wise=True,
					include_ciber_auto=True,
				ls_gal_auto_large_fpath=None,
				apply_satellite_correction=True,
				satellite_surveys=None,
				show_linear_pred=True, 
				include_1h_pred=True,
				onehalo_output_dir=None,
				onehalo_fsat_model='single'):
	"""
	Parameters
	----------
	ls_gal_auto_large_fpath : str, optional
		Path to .npz file from compute_gal_auto_spectrum_large() for the LS
		galaxy auto spectrum (larger footprint). If provided, replaces the
		standard 2x2deg LS galaxy auto with the larger-footprint version.
		Expected keys: 'lb', 'all_cl_gal', 'all_clerr_gal', 'ifield_list_use'.
	"""

	if all_addstr is None:
		all_addstr = ['sdss_z_lt_22.0_CIBERfidmask_zmax=1.0',
					'hsc_i_lt_25.0_CIBERfidmask_zmax=1.0',
					'wise_W1_lt_20.2_CIBERfidmask']

	if jmock_basedir is None:
		jmock_basedir = 'data/jordan_mocks/v2/'

	# Apply transfer-function correction once in the plotting stage for both
	# measurements and model curves, keeping a consistent correction choice.
	hsc_auto_cross = compute_rl_ciber_gal(hsc_str, catname='HSC', tl_pix_correct=False)
	ls_auto_cross = compute_rl_ciber_gal(ls_str, catname='LS', tl_pix_correct=False)

	# Optionally replace LS galaxy auto with larger-footprint version
	if ls_gal_auto_large_fpath is not None:
		print(f'[omnibus] Replacing LS galaxy auto with larger footprint from {ls_gal_auto_large_fpath}')
		large_dat = np.load(ls_gal_auto_large_fpath, allow_pickle=True)
		large_all_cl_gal = large_dat['all_cl_gal']    # [n_field, n_ell]
		large_all_clerr_gal = large_dat['all_clerr_gal']
		lb_large = large_dat['lb']

		# fmask for Knox: field_size^2 encodes the larger physical area per field
		field_size = float(large_dat.get('field_size', 4.0))
		fmask_large = field_size**2 / (2.0**2)  # ratio of large to standard 2x2 field area

		startidx, endidx = 2, -1

		for acdat in ls_auto_cross:
			# mini_proc_clav uses lb only for pf and Knox — pass lb_large so
			# the ell grid matches the large-file spectra
			pf, posmask_auto, negmask_auto, fieldav_cl_gal, fieldav_clerr_gal = mini_proc_clav(
				large_all_cl_gal, large_all_clerr_gal, lb_large, startidx, endidx,
				mode='auto', fmask=fmask_large
			)

			acdat.fieldav_cl_gal = fieldav_cl_gal
			acdat.fieldav_clerr_gal = fieldav_clerr_gal
			acdat.posmask_auto = posmask_auto
			acdat.negmask_auto = negmask_auto
			acdat.lb_gal_auto = lb_large
			acdat.pf_gal_auto = pf

	all_acdat_multi = [ls_auto_cross, hsc_auto_cross]
	gal_labels = ['DESI-LS ($z_{\\rm AB}<22, z_{\\rm phot}<1$)', 'HSC ($18<i_{\\rm AB}<25, z_{\\rm phot}<1$)']
	gal_labels_filenames = ['DESI-LS_z_AB_22_z_phot_1', 'HSC_i_AB_25_z_phot_1']
	addstr_use = [all_addstr[0], all_addstr[1]]

	# second_lines = ['$z_{\\rm AB}<22, z_{\\rm phot}<1$', '$18<i_{\\rm AB}<25, z_{\\rm phot}<1$']

	if include_wise:
		wise_auto_cross = compute_rl_ciber_gal(wise_str, catname='WISE', tl_pix_correct=False)
		all_acdat_multi.append(wise_auto_cross)
		gal_labels.append('unWISE ($W1_{\\rm Vega}<17.5$)')
		gal_labels_filenames.append('unWISE_W1_Vega_17.5')
		addstr_use.append(all_addstr[2])

	# Effective galaxy bias for each catalog (used to rescale smooth model curves).
	# LS z<1 value from compute_effective_bias_ls([0,1]); HSC and WISE are fixed priors.
	b_eff_ls_zlt1 = 1.29   # measured: 1.2899 +/- 0.0146
	bias_values = [b_eff_ls_zlt1, 1.53]  # LS, HSC z<1
	z_eff_values = [0.59, 0.63]
	if not include_wise:
		bias_values = bias_values[:2]

	pred_fpaths_multi = []
	for addstr in addstr_use:
		pred_fpaths = [jmock_basedir+'mock_ps_pred/TM'+str(inst)+'/field_average/pred_cls_TM'+str(inst)+'_'+addstr+'.npz' for inst in [1, 2]]
		pred_fpaths_multi.append(pred_fpaths)



	fig = plot_gal_and_ciber_auto(all_acdat_multi, pred_fpaths=pred_fpaths_multi, colors=['b', 'r'],
						xlim=[280, 1.1e5],
							ylims_gal=ylims_gal,
						gal_labels=gal_labels, gal_labels_filenames=gal_labels_filenames, band_labels=['1.1 $\\mu$m', '1.8 $\\mu$m'],
						startidx=2, endidx=-1,
						capsize=3, markersize=3, figsize=figsize,
						lab_fs=14, legend_fs=12, title_fs=12, pred_alpha=0.9, spacer_and_ciber_auto=[0.35, 1.1],
							tl_pix_correct=tl_pix_correct,
							ifield_use=ifield_use,
							tl_pix_template=tl_pix_template,
							include_ciber_auto=include_ciber_auto,
						bias_values=bias_values,
						apply_satellite_correction=apply_satellite_correction,
						show_linear_pred=show_linear_pred, 
						z_eff_values=z_eff_values,
						include_1h_pred=include_1h_pred,
						onehalo_output_dir=onehalo_output_dir,
						onehalo_fsat_model=onehalo_fsat_model
						)

	return fig


def rescale_predictions_bias(all_pred_fpaths, zbinedges, bias_model='1+z', 
														 ell_2h_max=3000, verbose=False):
	"""
	Load prediction files, rescale galaxy auto 2-halo component by bias model.

	This function:
	1. Loads each prediction file
	2. Fits and separates 2h/1h components of gal_auto
	3. Rescales gal_auto 2h by b_g^2
	4. Rescales cross-spectrum by b_g (since C_ℓ^Ig ∝ b_g)
	5. Returns list of modified data dictionaries (not file paths)

	Parameters
	----------
	all_pred_fpaths : list of list of str
		Prediction file paths [inst][zbin]
	zbinedges : array
		Redshift bin edges
	bias_model : str or float, optional
		Bias model (default '1+z')
	ell_2h_max : float, optional
		Max ℓ for 2h fitting (default 3000)
	verbose : bool, optional
		Print rescaling info

	Returns
	-------
	all_pred_data : list of list of dict
		Modified prediction data [inst][zbin], each containing rescaled arrays
	"""
	z_centers = 0.5 * (np.array(zbinedges[:-1]) + np.array(zbinedges[1:]))
	all_pred_data = []
	
	for inst_idx, inst_fpaths in enumerate(all_pred_fpaths):
		inst_pred_data = []
		
		for zidx, fpath in enumerate(inst_fpaths):
			# Load original prediction
			pred = np.load(fpath)
			
			# Create mutable copy
			pred_data = {key: pred[key].copy() for key in pred.keys()}
			
			# Rescale gal_auto 2-halo component
			gal_auto_rescaled, bias_factor = rescale_gal_auto_2halo_bias(
				pred['lb'], pred['gal_auto'], z_centers[zidx],
				bias_model=bias_model, ell_2h_max=ell_2h_max, verbose=verbose
			)
			
			# Update with rescaled galaxy auto (factor of b_g^2)
			pred_data['gal_auto'] = gal_auto_rescaled
			pred_data['gal_auto_original'] = pred['gal_auto']  # Keep original for reference
			
			# Rescale cross-spectrum by b_g (since C_ℓ^Ig ∝ b_g)
			pred_data['cross'] = pred['cross'] * bias_factor
			pred_data['cross_original'] = pred['cross']  # Keep original for reference
			
			pred_data['bias_factor'] = bias_factor
			
			if verbose:
				print(f"  Rescaled cross by b_g = {bias_factor:.2f}")
			
			inst_pred_data.append(pred_data)
		
		all_pred_data.append(inst_pred_data)
	
	return all_pred_data


def rescale_spectrum_2halo_bias(lb, spectrum, z_center, bias_model='1+z', 
								 ell_2h_max=3000, bias_power=2, verbose=False):
	"""
	Rescale the 2-halo component of a power spectrum using a bias model.
	
	Fits spectrum as constant 2h + power-law 1h at low-ell to separate components,
	then rescales the 2h component by bias factor b_g^bias_power.
	
	Parameters
	----------
	lb : array
		Multipole bin centers
	spectrum : array
		Power spectrum (C_ℓ or D_ℓ, function handles both)
		Can be galaxy auto (bias_power=2) or cross (bias_power=1)
	z_center : float
		Redshift bin center for evaluating bias model
	bias_model : str or float, optional
		Bias model. Options:
		- '1+z': b_g = 1 + z (default)
		- float: fixed bias value
	ell_2h_max : float, optional
		Maximum ℓ to consider as 2-halo dominated (default 3000)
	bias_power : int, optional
		Power of bias scaling: 2 for auto (C_ℓ ∝ b²), 1 for cross (C_ℓ ∝ b)
	verbose : bool, optional
		Print rescaling info
	
	Returns
	-------
	spectrum_rescaled : array
		Spectrum with rescaled 2-halo component
	bias_factor : float
		Bias factor applied (b_g)
	"""
	from scipy.optimize import curve_fit
	
	# Evaluate bias model
	if bias_model == '1+z':
		bias_factor = 1 + z_center
	elif isinstance(bias_model, (int, float)):
		bias_factor = float(bias_model)
	else:
		raise ValueError(f"Unknown bias_model: {bias_model}")
	
	# Fit 2h + 1h model at low ell
	mask_2h = lb <= ell_2h_max
	lb_fit = lb[mask_2h]
	spectrum_fit = spectrum[mask_2h]
	
	# Model: constant 2h + power-law 1h
	def model_2h_1h(ell, A_2h, A_1h, alpha_1h):
		return A_2h + A_1h * (ell / 1000.)**alpha_1h
	
	try:
		# Initial guess
		p0 = [np.median(spectrum_fit[:3]), np.max(spectrum_fit), -1.5]
		popt, _ = curve_fit(model_2h_1h, lb_fit, spectrum_fit, p0=p0, 
					   bounds=([0, 0, -5], [np.inf, np.inf, 5]))
		
		A_2h, A_1h, alpha_1h = popt
		
		# Separate 2h and 1h components over full ell range
		component_2h = A_2h * np.ones_like(lb)
		component_1h = A_1h * (lb / 1000.)**alpha_1h
		
		# Rescale 2h by bias^power (C_ℓ^gg ∝ b², C_ℓ^Ig ∝ b)
		component_2h_rescaled = component_2h * bias_factor**bias_power
		
		# Reconstruct: rescaled 2h + original 1h
		spectrum_rescaled = component_2h_rescaled + component_1h
		
		if verbose:
			print(f"  z={z_center:.2f}: b_g={bias_factor:.2f}, "
				  f"A_2h={A_2h:.3e} → {component_2h_rescaled[0]:.3e} "
				  f"(×{bias_factor**bias_power:.2f})")
	
	except Exception as e:
		if verbose:
			print(f"  Warning: 2h fit failed at z={z_center:.2f}, returning original spectrum")
		return spectrum, bias_factor
	
	return spectrum_rescaled, bias_factor


def rescale_gal_auto_2halo_bias(lb, gal_auto, z_center, bias_model='1+z',
								 ell_2h_max=3000, verbose=False):
	"""Convenience wrapper for rescaling galaxy auto (bias_power=2)"""
	return rescale_spectrum_2halo_bias(lb, gal_auto, z_center, bias_model=bias_model,
									   ell_2h_max=ell_2h_max, bias_power=2, verbose=verbose)


def smooth_mock_cross_with_bias(pred_fpath, z_center, b_g,
                                shot_ell_min=30000., shot_ell_max=80000.,
                                twoh_ell_max=1000., ell_eval=None, mode='cross'):
	"""Fit shot-noise + flat two-halo to a noisy mock cross-spectrum and rescale by b_g.

	The mock prediction files assume b_g = 1.  This function:
	  1. Converts mock C_ell -> D_ell.
	  2. Estimates the shot-noise plateau from ell in [shot_ell_min, shot_ell_max].
	  3. Estimates the two-halo amplitude as the mean D_ell residual at ell < twoh_ell_max.
	  4. Constructs a smooth model:  D_ell_smooth = A_2h * b_g  +  A_shot * pf(ell)
	     Only the two-halo term scales with b_g; shot noise is held fixed at the mock value.
	  5. Returns the smooth D_ell curve evaluated on *ell_eval*.

	Parameters
	----------
	pred_fpath : str
	    Path to the mock prediction .npz file.
	z_center : float
	    Redshift bin centre (used only for logging).
	b_g : float
	    Effective galaxy bias to apply (rescales only the two-halo amplitude).
	shot_ell_min, shot_ell_max : float
	    Ell range used to estimate the shot-noise plateau in D_ell.
	twoh_ell_max : float
	    Ell below which the two-halo average is computed.
	ell_eval : array_like or None
	    Ell values at which to return the smooth curve.  Defaults to the mock lb grid.

	Returns
	-------
	ell_eval : ndarray
	dl_smooth : ndarray
	    Smooth bias-rescaled D_ell cross-spectrum.
	"""
	from scipy.optimize import curve_fit

	pred = np.load(pred_fpath)
	lb   = np.asarray(pred['lb'], dtype=float)
	if mode=='cross':
		cl = np.asarray(pred['cross'], dtype=float)
	else:
		cl = np.asarray(pred['gal_auto'], dtype=float)
	pf   = lb * (lb + 1.0) / (2.0 * np.pi)
	dl   = pf * cl

	if ell_eval is None:
		ell_eval = lb
	ell_eval = np.asarray(ell_eval, dtype=float)
	pf_eval  = ell_eval * (ell_eval + 1.0) / (2.0 * np.pi)

	# --- shot-noise estimate from high-ell plateau ---
	shot_mask = np.isfinite(dl) & (lb >= shot_ell_min) & (lb <= shot_ell_max)
	if not np.any(shot_mask):
		shot_mask = np.isfinite(dl) & (lb >= lb.max() * 0.5)
	# shot noise in D_ell = A_shot * pf(ell)  =>  C_ell_shot = A_shot (constant)
	# Estimate A_shot = mean(dl / pf) over shot-noise window
	pf_shot   = lb[shot_mask] * (lb[shot_mask] + 1.0) / (2.0 * np.pi)
	A_shot    = float(np.nanmean(dl[shot_mask] / pf_shot))

	# --- two-halo estimate from low-ell residual ---
	twoh_mask = np.isfinite(dl) & (lb <= twoh_ell_max)
	dl_sub    = dl - A_shot * pf
	A_2h      = float(np.nanmean(dl_sub[twoh_mask])) if np.any(twoh_mask) else 0.0
	A_2h      = max(A_2h, 0.0)

	print('A_2h is ', A_2h)

	# --- smooth model: only two-halo rescaled by b_g (cross ∝ b_g^1); shot noise unchanged ---
	dl_smooth = b_g * A_2h + A_shot * pf_eval

	return ell_eval, dl_smooth


def _load_bias_for_z(bias_cache, z_center, scheme='fine'):
	"""Retrieve interpolated b_g from the cached bias npz at a given redshift.

	Parameters
	----------
	bias_cache : dict-like (np.NpzFile or dict)
	    Loaded .npz cache from compute_effective_bias_ls.py.
	z_center : float
	    Redshift to query.
	scheme : str
	    'fine' or 'coarse' — which binning's polynomial coeffs to use.

	Returns
	-------
	b_g : float
	"""
	key = f'{scheme}_poly_coeffs'
	if key not in bias_cache:
		# Fall back to model-evaluated centers if poly not available
		zc_key = f'{scheme}_z_centers'
		beff_key = f'{scheme}_b_eff'
		if zc_key in bias_cache:
			return float(np.interp(z_center,
			                       np.asarray(bias_cache[zc_key]),
			                       np.asarray(bias_cache[beff_key])))
		return 1.0
	coeffs = np.asarray(bias_cache[key])
	return float(np.poly1d(coeffs)(z_center))


def gen_cross_spectrum_plots_vs_z(inst_list = [1, 2],
								ifield_list = [4, 5, 6, 7, 8],
								maskstr = 'JHlt16_wFFerr',
								xlim=[250, 1.1e5],
								ylim=[5e-3, 1e3],
								markersize=3,
								capsize=3,
								rescale_gal_auto_bias=False,
								bias_model='1+z',
								headstr_hsc='hsc_ilt25.0',
								headstr_ls='sdss_z_lt_22.0',
								lab_ls='DESI-LS ($z_{\\rm AB}<22$)',
								lab_hsc='HSC ($18<i_{\\rm AB}<25$)',
								plot_fine=True,
								plot_coarse=True,
								bias_cache_fpath=None, 
								include_1h_pred=True, 
								onehalo_output_dir=None,
								onehalo_fsat_model='single',
								mode='cross'):


	if plot_fine:
		zbinedges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

		# fine bins for LS
		res_ls = collect_ciber_gal_vs_redshift('LS', subtract_randoms=True, \
											inst_list=inst_list, zbinedges=zbinedges, \
											maskstr=maskstr, subtract_sn=False, 
											tl_pix_correct=True, ifield_list=ifield_list)
		
		lb, full_cl_cross_ls, full_clerr_cross_ls = [res_ls[key] for key in ['lb', 'full_cl_cross', 'full_clerr_cross']]

		all_pred_fpaths_ls = grab_ciber_cross_vs_z_predfpaths(inst_list=inst_list, zbinedges=zbinedges, 
																	headstr='sdss_z_lt_22.0_CIBERfidmask')

		print('all pred fpaths ls:', len(all_pred_fpaths_ls[0]))

		fig_ls_ciber = plot_cross_ps_vs_redshift(inst_list, zbinedges, lb, full_cl_cross_ls, full_clerr_cross_ls, figsize=(11, 9.0), \
					xlim=xlim, ylim=ylim, markersize=markersize, capsize=capsize, alph=0.7, textxpos=300, \
					color='k', ncols=4, text_fs=11, textypos=300, \
					all_pred_fpaths=all_pred_fpaths_ls, bbox_to_anchor=[-0.02, 1.38], legend_fs=15,
					tl_pix_correct=True, nrows=3, catname='DESI-LS',
					rescale_gal_auto_bias=rescale_gal_auto_bias, bias_model=bias_model,
					bias_cache_fpath=bias_cache_fpath, bias_cache_scheme='coarse', include_1h_pred=include_1h_pred,
					onehalo_output_dir=onehalo_output_dir, onehalo_fsat_model=onehalo_fsat_model)
	else:
		fig_ls_ciber = None

	if plot_coarse:
		# coarse bins for LS and HSC
		zbinedges_coarse = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

		res_ls_coarse = collect_ciber_gal_vs_redshift('LS', subtract_randoms=True, \
											inst_list=inst_list, zbinedges=zbinedges_coarse, \
											maskstr=maskstr, subtract_sn=False, 
											tl_pix_correct=True)
		
		# lb, full_cl_cross_ls_coarse, full_clerr_cross_ls_coarse = [res_ls_coarse[key] for key in ['lb', 'full_cl_cross', 'full_clerr_cross']]

		lb, full_cl_cross_ls_coarse, full_clerr_cross_ls_coarse = [res_ls_coarse[key] for key in ['lb', 'full_cl_gal', 'full_clerr_gal']]

		# for zidx in range(len(zbinedges_coarse)-1):
			# fpath_ls = 'data/ciber_gal_cross_cls/DESILS_coarsez/cl_CIBER_DESILS_zbin'+str(zidx)+'.npz'
			# np.savez(fpath_ls, lb=lb, cl=full_cl_cross_ls_coarse[zidx], clerr=full_clerr_cross_ls_coarse[zidx], 
			#    inst_list=inst_list, zmin=zbinedges[zidx], zmax=zbinedges[zidx+1])


		all_pred_fpaths_ls_coarse = grab_ciber_cross_vs_z_predfpaths(inst_list=inst_list, zbinedges=zbinedges_coarse, 
																headstr=headstr_ls)

		res_hsc = collect_ciber_gal_vs_redshift('HSC', subtract_randoms=True, \
											inst_list=inst_list, zbinedges=zbinedges_coarse, \
											maskstr=None, subtract_sn=False, 
											tl_pix_correct=True, 
											ifield_list=[8], 
											with_ff_err=True, 
											headstr=headstr_hsc)
		
		# lb, cl_cross_hsc, clerr_cross_hsc = [res_hsc[key] for key in ['lb', 'full_cl_cross', 'full_clerr_cross']]
		lb, cl_cross_hsc, clerr_cross_hsc = [res_hsc[key] for key in ['lb', 'full_cl_gal', 'full_clerr_gal']]

		# for inst in inst_list:
		# 	for zidx in range(len(zbinedges_coarse)-1):
		# 		fpath_hsc = 'data/ciber_gal_cross_cls/HSC_coarsez/cl_CIBER_TM'+str(inst)+'_HSC_ilt25.0_zbin'+str(zidx)+'.npz'
		# 		print('saving to ', fpath_hsc)
		# 		print('cl_Cross_hsc has shape', cl_cross_hsc.shape)
		# 		np.savez(fpath_hsc, lb=lb, cl=cl_cross_hsc[inst-1][zidx], clerr=clerr_cross_hsc[inst-1][zidx], 
		# 				inst=inst, zmin=zbinedges_coarse[zidx], zmax=zbinedges_coarse[zidx+1])

		# 		fpath_ls = 'data/ciber_gal_cross_cls/DESILS_coarsez/cl_CIBER_TM'+str(inst)+'_DESILS_zbin'+str(zidx)+'.npz'
		# 		np.savez(fpath_ls, lb=lb, cl=full_cl_cross_ls_coarse[inst-1][zidx], clerr=full_clerr_cross_ls_coarse[inst-1][zidx], 
		# 				inst=inst, zmin=zbinedges_coarse[zidx], zmax=zbinedges_coarse[zidx+1])


		all_pred_fpaths_hsc = grab_ciber_cross_vs_z_predfpaths(inst_list=inst_list, zbinedges=zbinedges_coarse, 
																headstr='hsc_i_lt_25.0')

		catalog_names = [lab_ls, lab_hsc]

		all_catalogs_data = [full_cl_cross_ls_coarse, cl_cross_hsc]
		all_catalogs_error = [full_clerr_cross_ls_coarse, clerr_cross_hsc]
		all_catalogs_pred_fpaths = [all_pred_fpaths_ls_coarse, all_pred_fpaths_hsc]

		hsc_color = "#E45DA8"
		
		ls_color = 'C2'
		
		fig_coarse_ls_hsc = plot_cross_ps_by_wavelength_and_redshift(
			all_catalogs_cl_cross=all_catalogs_data,
			all_catalogs_clerr_cross=all_catalogs_error,
			catnames=catalog_names,
			zbinedges=zbinedges_coarse,
			lb=lb,
			figsize=(13.0, 5.2),
			ylim=[5e-3, 1e3],
			xlim=[250, 1.05e5],
			all_catalogs_pred_fpaths=all_catalogs_pred_fpaths,
			textxpos=350,
			colors_cat=[ls_color, hsc_color, 'C2'],
			bbox_to_anchor=[0.3, 1.3],
			linestyles_pred=['solid', 'solid'],
			text_fs=10,
			legend_fs=16,
			rescale_gal_auto_bias=rescale_gal_auto_bias,
			bias_model=bias_model,
			bias_cache_fpath=bias_cache_fpath, bias_cache_scheme='coarse',
			include_1h_pred=include_1h_pred,
			onehalo_fsat_model=onehalo_fsat_model,
			onehalo_output_dir=onehalo_output_dir,
			mode=mode
		)
	else:
		fig_coarse_ls_hsc = None

	return fig_ls_ciber, fig_coarse_ls_hsc


def plot_rl_vs_z_vs_scale_DESILS(res_meas, mean_rl_diffscale_pred, 
                                 figsize=(8,5.5), inst_list=[1, 2], lams=[1.1, 1.8], 
                                 colors=['b', 'r'], 
                                 markersize=6, cmap='inferno_r', 
                                 ylim=(-0.05, 0.45), textypos=0.41, textxpos=0.03):

    fig = plt.figure(figsize=figsize)

    cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(0.3, 1, 3))

    for inst in inst_list:
        plt.subplot(2,1,inst)
        plt.text(textxpos, textypos, 'CIBER '+str(lams[inst-1])+' $\\mu$m $\\times$ DESI-LS ($z_{\\rm AB}<22$)', fontsize=16)
        plt.axhline(0, color='grey', alpha=0.2)

        for lidx in range(len(res_meas['lb_mins'])):
            ellstr = str(int(res_meas['lb_mins'][lidx]))+'$<\\ell<$'+str(int(res_meas['lb_maxs'][lidx]))

            plt.errorbar(res_meas['zcen'], np.array(res_meas['mean_rl_diffscale'])[lidx, inst-1],
                          yerr=np.array(res_meas['std_rl_diffscale'])[lidx, inst-1], fmt='o', color=colors[lidx],
                        label=ellstr, marker='s', markersize=markersize, alpha=0.7)

            if lidx==len(res_meas['lb_mins'])-1:
                predlab = 'IGL model prediction'
            else:
                predlab = None

            plt.plot(res_meas['zcen'], np.array(mean_rl_diffscale_pred)[lidx, inst-1], color=colors[lidx], linestyle='dotted', 
                    label=predlab, linewidth=2)

        plt.grid(alpha=0.3)
        plt.ylabel('$\\langle r_{\\ell}^{\\rm I \\times g} \\rangle$', fontsize=14)
        plt.ylim(ylim)

        if inst==2:
#             plt.tick_params(labelsize=12)

            plt.xlabel('redshift', fontsize=14)
            plt.xticks(res_meas['zbinedges'])
            plt.tick_params(labelsize=12)


        else:
            plt.legend(fontsize=14, bbox_to_anchor=[0.03, 1.4], loc=2, ncol=2)
            plt.xticks(res_meas['zbinedges'], ['' for _ in range(len(res_meas['zbinedges']))])
            plt.tick_params(labelsize=12)

    plt.subplots_adjust(hspace=0.1)
    plt.show()

    return fig


def plot_cross_ps_by_wavelength_and_redshift(
	all_catalogs_cl_cross, all_catalogs_clerr_cross,
	catnames, zbinedges, lb, 
	all_catalogs_cl_galauto=None, all_catalogs_clerr_galauto=None,
	inst=[1, 2],figsize=(16, 8), 
	startidx=2, endidx=-1,
	xlim=[150, 1.1e5], ylim=[1e-4, 2e2], 
	legend_fs=16, capsize=3, markersize=3, alph=0.9,
	textxpos=280, textypos=1e2, text_fs=12, 
	colors_cat=['k', 'C1'], bbox_to_anchor=[1.0, 1.35],
	ncol_legend=2,
	all_catalogs_pred_fpaths=None,
	pred_alpha=0.5,
	tl_pix_correct=False,
	linestyles_pred = ['dotted', 'dotted'],
	label_fs=13,
	rescale_gal_auto_bias=False,
	bias_model='1+z',
	bias_cache_fpath=None, bias_cache_scheme='coarse', 
	include_1h_pred=True, onehalo_fsat_model='single', onehalo_output_dir='data/one_halo_model_output', 
	mode='cross'):
	"""
	Plots cross-power spectra for multiple wavelengths and redshift bins.

	Each row corresponds to a wavelength (1.1um, 1.8um).
	Each column corresponds to a redshift bin.
	Each subplot shows measurements for different galaxy catalogs and an optional model prediction.
	"""

	lam_dict = {1: 1.1, 2: 1.8}
	pf = lb * (lb + 1) / (2 * np.pi)
	lbmask = (lb >= lb[startidx]) & (lb < lb[endidx])
	bias_cache = np.load(bias_cache_fpath, allow_pickle=False) if bias_cache_fpath is not None else None

	nrows = len(inst)
	if all_catalogs_cl_galauto is not None:
		nrows += 1
	ncols = len(zbinedges) - 1
	
	fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)
	
	if nrows == 1 and ncols == 1:
		ax = np.array([[ax]])
	elif nrows == 1:
		ax = np.array([ax])
	elif ncols == 1:
		ax = np.array([[a] for a in ax])


	# Outermost loop: iterate over instruments/wavelengths (rows)
	for inst_idx, inst_indiv in enumerate(inst):
		
		# Middle loop: iterate over redshift bins (columns)
		for zidx, z0 in enumerate(zbinedges[:-1]):
			
			current_ax = ax[inst_idx, zidx]
			z1 = zbinedges[zidx+1]

			
			# Innermost loop: iterate over galaxy catalogs
			for cat_idx, catname in enumerate(catnames):

				if 'HSC' in catname:
					mag_cut = 25.0
					bandstr_select = 'hsc_i'
				else:
					mag_cut = 22.0
					bandstr_select = 'sdss_z'

				if include_1h_pred:
					oh_data_Ig = load_onehalo_spectrum(
							onehalo_output_dir, onehalo_fsat_model, bandstr_select,
							inst=inst_indiv, mag_min=18.0, mag_cut=mag_cut, z0=0.05, mode='Ig', generate_type='fine')
					ell_1h = oh_data_Ig['ell_arr']
					dl_1h = oh_data_Ig['dl_spectrum'][zidx]


				if 'HSC' in catname and zidx==0:
					# Skip HSC for the first redshift bin (z<0.2) due to low galaxy counts
					rescale_fac = 1e10
				else:
					rescale_fac = 1.0

				fieldav_cl_cross = all_catalogs_cl_cross[cat_idx][inst_idx][zidx]
				fieldav_clerr_cross = all_catalogs_clerr_cross[cat_idx][inst_idx][zidx]

				posmask = lbmask & (fieldav_cl_cross > 0)
				negmask = lbmask & (fieldav_cl_cross < 0)

				current_ax.errorbar(lb[posmask], rescale_fac*(pf * fieldav_cl_cross)[posmask], yerr=(pf * fieldav_clerr_cross)[posmask],
									color=colors_cat[cat_idx], fmt='o', capsize=capsize, markersize=markersize, 
									zorder=15, label=catname, alpha=alph)
				
				current_ax.errorbar(lb[negmask], rescale_fac*np.abs(pf * fieldav_cl_cross)[negmask], yerr=(pf * fieldav_clerr_cross)[negmask],
									color=colors_cat[cat_idx], fmt='o', capsize=capsize, markersize=markersize, 
									zorder=15, mfc='white', alpha=alph)
				
				
				# --- Handle model predictions (if provided) ---
				if all_catalogs_pred_fpaths is not None:
					pred_path = all_catalogs_pred_fpaths[cat_idx][inst_idx][zidx]
					z_center = 0.5 * (zbinedges[zidx] + zbinedges[zidx + 1])

					if bias_cache is not None:
						# Bias per catalog: LS uses the measured cache; HSC uses b(z)=1+0.84*z
						if cat_idx == 0:
							b_g = _load_bias_for_z(bias_cache, z_center, scheme=bias_cache_scheme)
						else:
							b_g = 1.0 + 0.84 * z_center
						ell_eval = np.geomspace(xlim[0], xlim[1], 300)
						ell_smooth, dl_smooth = smooth_mock_cross_with_bias(
							pred_path, z_center, b_g, ell_eval=ell_eval, mode=mode)
						if tl_pix_correct:
							ifield_use = 6
							tl_pix_path = f'data/fluctuation_data/transfer_function/tl_clx_pix_TM{inst_indiv}_ifield{ifield_use}.npz'
							tl_pix = np.load(tl_pix_path)['tl_clx_pix']
							tl_interp = np.interp(ell_smooth, np.arange(len(tl_pix)), tl_pix)
							dl_smooth = dl_smooth / tl_interp
						if inst_indiv == 1 and cat_idx == 0:
							lab_pred = 'IGL prediction (2h+1h+P)'
						else:
							lab_pred = None
						
						if include_1h_pred:
							dl_1h_interp = np.interp(ell_smooth, ell_1h, dl_1h)

							if zidx == 0:
								dl_1h_interp *= 0.5  # Reduce 1-halo amplitude for the first redshift bin to avoid overprediction
							dl_smooth += dl_1h_interp
						
						current_ax.plot(ell_smooth, rescale_fac*dl_smooth, color=colors_cat[cat_idx],
						                linestyle=linestyles_pred[cat_idx], alpha=pred_alpha, label=lab_pred, linewidth=2)
					else:
						# Fall back to raw (noisy) mock curve
						jmock_pred = np.load(pred_path)
						if rescale_gal_auto_bias:
							cross, _ = rescale_spectrum_2halo_bias(
								jmock_pred['lb'], jmock_pred['cross'], z_center,
								bias_model=bias_model, bias_power=1, verbose=False)
						else:
							cross = jmock_pred['cross']
						lb_pred = jmock_pred['lb']
						pf_pred = lb_pred * (lb_pred + 1) / (2 * np.pi)
						if tl_pix_correct:
							ifield_use = 6
							tl_pix_path = f'data/fluctuation_data/transfer_function/tl_clx_pix_TM{inst_indiv}_ifield{ifield_use}.npz'
							tl_pix = np.load(tl_pix_path)['tl_clx_pix']
							cross /= tl_pix
						if inst_indiv == 1 and cat_idx == 0:
							lab_pred = 'IGL prediction'
						else:
							lab_pred = None
						current_ax.plot(lb_pred, rescale_fac*pf_pred * cross, color=colors_cat[cat_idx],
						                linestyle=linestyles_pred[cat_idx], alpha=pred_alpha, label=lab_pred, linewidth=2)


				
				

			# --- SUBPLOT FORMATTING ---
			
			zlab = str(np.round(z0, 1))+'$<z_{\\rm phot}<$'+str(np.round(z1, 1))
#             gal_label = f'{z0:.1f} < $z_{\\rm phot}$ < {z1:.1f}'
			
			gal_label = f'CIBER {lam_dict[inst_indiv]} $\\mu$m\n'+zlab
			current_ax.text(textxpos, textypos, gal_label, fontsize=text_fs)
			
			current_ax.set_ylim(ylim)
			current_ax.set_xlim(xlim)
			current_ax.grid(alpha=0.3)
			current_ax.set_xscale('log')
			current_ax.set_yscale('log')

			if inst_idx == 0 and zidx == 0:
				# Increase ncol_legend to accommodate the prediction label if it exists
				legend_cols = ncol_legend + 1 if all_catalogs_pred_fpaths is not None and inst[0] == 1 else ncol_legend
				current_ax.legend(bbox_to_anchor=bbox_to_anchor, ncol=legend_cols, fontsize=legend_fs, loc=2)

			if inst_idx == nrows - 1:
				current_ax.set_xlabel('$\\ell$', fontsize=label_fs)
			
			if zidx == 0:
				ylabel_text = f'$D_\\ell^{{\\rm Ig}}$ [nW m$^{{-2}}$ sr$^{{-1}}$]'
				current_ax.set_ylabel(ylabel_text, fontsize=label_fs)

	# Remove any extra axes if grid has more slots than redshift bins
	n_panels = nrows * ncols
	n_needed = len(zbinedges) - 1
	if n_needed < n_panels:
		for idx in range(n_needed, n_panels):
			try:
				fig.delaxes(ax[idx])
			except Exception:
				pass

	plt.subplots_adjust(wspace=0, hspace=0)
	plt.show()
	
	return fig


def plot_bgdNdz_bIdIdz(figsize=(6, 6), inst_list=[1, 2], zbinedges=None, \
					  colors=['b', 'C3'], z0_color=0.0, z1_color=1.0,\
					   textxpos_bg=0.48, textypos_bg=22, textxpos_bI=0.55, textypos_bI=14.5, \
					  ylim=[-5, 18], grid_alpha=0.3):

	''' Not currently using '''
	
	lams=[1.1, 1.8]
	xerr=0.05

	if zbinedges is None:
		zbinedges_fine = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

	
	colors = plt.cm.jet(np.linspace(z0_color, z1_color, len(zbinedges_fine)-1))
	
	fig, ax = plt.subplots(ncols=1, nrows=3, sharex=True, figsize=figsize)

	# bg x dN/dz
	for zidx, z0 in enumerate(zbinedges_fine[:-1]):

		z1 = zbinedges_fine[zidx+1]

		ax[0].errorbar(z_fine, all_unnorm_dNdz_b[zidx], yerr=np.abs(all_unnorm_dNdz_b_err[zidx]), alpha=0.6, markersize=8, color=colors[zidx])
	ax[0].set_ylabel('$b_g dN/dz$', fontsize=14)
	ax[0].text(textxpos_bg, textypos_bg, 'Legacy Survey ($z<22$)', fontsize=15)
	
	
	# bI x dI/dz
	for x, inst in enumerate(inst_list):

		ax[inst].errorbar(z_coarse_fine, all_bI_I[x], yerr=all_bI_I_err[x], color='k', fmt='o', capsize=3, markersize=5, xerr=xerr)
		ax[inst].set_ylabel('$b_I dI/dz_{\\rm phot}$', fontsize=14)
		ax[inst].set_ylim(ylim)

		all_bI_dIdz = []
		
		for zidx in range(len(z_coarse_fine)):
			dndzuse = all_norm_dNdz_b[zidx]
			supernorm_dndzb = dndzuse/np.mean(dndzuse[(z_fine > zbinedges_fine[zidx])*(z_fine < zbinedges_fine[zidx+1])])
			ax[inst].plot(z_fine, supernorm_dndzb*all_bI_I[x][zidx], color=colors_ciber[x], alpha=0.2)
			all_bI_dIdz.append(supernorm_dndzb*all_bI_I[x][zidx])

		all_bI_dIdz = np.array(all_bI_dIdz)

		ax[inst].text(textxpos_bI, textypos_bI, 'CIBER '+str(lams[x])+' $\\mu$m $\\times$ LS', fontsize=15)


	for i in range(len(ax)):
		ax[i].grid(alpha=grid_alpha)
		ax[i].set_xlim(0, 1)
		ax[i].axhline(0, color='k', linestyle='dashed', alpha=0.5)


	plt.subplots_adjust(wspace=0, hspace=0.02)

#     plt.savefig('figures/ciber_LS_bI_dI_dzphot_0_zph_1_072625.pdf', bbox_inches='tight')

	plt.show()
	
	return fig


def plot_hsc_cross_spectrum(
	mag_lims=[24.0, 25.0, 26.0],inst_list=[1, 2],
	tailstr='linfitrand', subtract_randoms=True,
	lab_fs=12, xlim=[250, 1e5],
	startidx=2, endidx=-1,
	markersize=2.5, capsize=2.,
	lams=[1.1, 1.8], linewidth=1, alpha_line=0.8,
	legend_fs=9, hspace=0.07, wspace=0.35):

	colors_tm1 = generate_colors(len(mag_lims), cmap='Blues')
	colors_tm2 = generate_colors(len(mag_lims), cmap='Reds')
	colors_auto = generate_colors(len(mag_lims), cmap='Greys')
	ciber_colors = [colors_tm1, colors_tm2]
	
	fig, ax = plt.subplots(figsize=(11, 6.5), nrows=2, ncols=3, sharex=False)
	linestyle = 'dashdot'
	print(ax.shape)
	for m, maglim in enumerate(mag_lims):
		labels_hsc = ['CIBER 1.1 $\\mu$m', 'CIBER 1.8 $\\mu$m']
		textstr = f'CIBER $\\times$ HSC ($i<{maglim}$)\\nSWIRE (ELAIS-N1)'
		addstr = f'hsc_ilt{maglim}'
		addstr_pred = f'hsc_i_lt_{maglim}_CIBERfidmask'
		if subtract_randoms:
			addstr += '_wrandsub'
		addstr += '_wFFerr'
		addstr += f'_{tailstr}'
		acdat = compute_rl_ciber_gal(addstr, catname='HSC', tl_pix_correct=True)
		lb = acdat[0].lb
		pf = lb * (lb + 1) / (2 * np.pi)
		jmock_basedir = 'data/jordan_mocks/v2/'
		pred_fpaths = [
			jmock_basedir + f'mock_ps_pred/TM{inst}/field_average/pred_cls_TM{inst}_{addstr_pred}.npz'
			for inst in [1, 2]
		]
		galstr = f'HSC $i<{maglim}$'
		# galaxy auto
		ax[0, 0].errorbar(
			lb[startidx:endidx],
			(pf * acdat[0].fieldav_cl_gal)[startidx:endidx],
			yerr=(pf * acdat[0].fieldav_clerr_gal)[startidx:endidx],
			fmt='o', color=colors_auto[m], label=galstr, markersize=markersize, capsize=capsize
		)
		ax[0, 0].set_xscale('log')
		ax[0, 0].set_xticks([1e3, 1e4, 1e5])
		ax[0, 0].set_yscale('log')
		ax[0, 0].set_xlim(xlim)
		ax[0, 0].set_ylim(1e-4, 2e1)
		ax[0, 0].grid(alpha=0.3)
		ax[0, 0].legend(loc=2, fontsize=legend_fs)
		ax[0, 0].set_ylabel('$D_{\\ell}^{gg}$', fontsize=lab_fs)
		ax[0, 0].set_xlabel('$\\ell$', fontsize=lab_fs)
		ax[0, 0].tick_params(labelsize=10)
		jmock_pred = np.load(pred_fpaths[0])
		lb_pred, clg_pred = [jmock_pred[key] for key in ['lb', 'gal_auto']]
		modllab = 'IGL prediction' if m == 0 else None
		ax[0, 0].plot(
			lb_pred, pf * clg_pred, color=colors_auto[m], linestyle=linestyle,
			alpha=alpha_line, linewidth=linewidth, label=modllab
		)
		for idx, inst in enumerate(inst_list):
			magstr = f'{lams[idx]} $\\mu$m $\\times$ $i<{int(maglim)}$'
			tl_pix = np.load(f'data/fluctuation_data/transfer_function/tl_clx_pix_TM{inst}_ifield8.npz')['tl_clx_pix']
			ploty = acdat[idx].r_ell
			plotyerr = acdat[idx].r_ell_unc
			plotx = lb[startidx:endidx]
			whichpos = (ploty > 0)
			ax[idx, 2].errorbar(
				plotx[whichpos], ploty[whichpos], yerr=plotyerr[whichpos],
				fmt='o', color=ciber_colors[idx][m], markersize=markersize, capsize=capsize, label=magstr
			)
			ax[idx, 2].errorbar(
				plotx[~whichpos], np.abs(ploty[~whichpos]), yerr=plotyerr[~whichpos],
				fmt='o', mfc='white', color=ciber_colors[idx][m], markersize=markersize, capsize=capsize
			)
			jmock_pred = np.load(pred_fpaths[idx])
			print([k for k in jmock_pred.keys()])
			lb_pred, r_ell_ls_pred = [jmock_pred[key] for key in ['lb', 'rlx_tracer_full']]
			r_ell_ls_pred /= tl_pix
			modllab = 'IGL prediction' if m == 0 else None
			if m == 0:
				ax[idx, 2].set_xscale('log')
				ax[idx, 2].set_xlim(xlim)
				ax[idx, 2].set_ylim(-0.1, 1.1)
				ax[idx, 2].set_ylabel('$r_{\\ell}^{\\rm I\\times g}$', fontsize=lab_fs)
				if idx == 1:
					ax[idx, 2].set_xlabel('$\\ell$', fontsize=14)
				ax[idx, 2].grid(alpha=0.3)
			ax[idx, 2].plot(
				lb_pred, r_ell_ls_pred, color=ciber_colors[idx][m], linestyle=linestyle,
				alpha=alpha_line, linewidth=linewidth, label=modllab
			)
			ploty = (pf * acdat[idx].fieldav_cl_cross)[startidx:endidx]
			plotyerr = (pf * acdat[idx].fieldav_clerr_cross)[startidx:endidx]
			plotx = lb[startidx:endidx]
			whichpos = (ploty > 0)
			ax[idx, 1].errorbar(
				plotx[whichpos], ploty[whichpos], yerr=plotyerr[whichpos],
				fmt='o', color=ciber_colors[idx][m], markersize=markersize, capsize=capsize, label=magstr
			)
			ax[idx, 1].errorbar(
				plotx[~whichpos], np.abs(ploty[~whichpos]), yerr=plotyerr[~whichpos],
				fmt='o', mfc='white', color=ciber_colors[idx][m], markersize=markersize, capsize=capsize
			)
			jmock_pred = np.load(pred_fpaths[idx])
			lb_pred, clx_pred = [jmock_pred[key] for key in ['lb', 'cross']]
			clx_pred /= tl_pix
			if m == 0:
				ax[idx, 1].set_xscale('log')
				ax[idx, 1].set_yscale('log')
				ax[idx, 1].set_xlim(xlim)
				ax[idx, 1].set_ylim(1e-3, 1e2)
				ax[idx, 1].set_ylabel('$D_{\\ell}^{\\rm I\\times g}$ [nW m$^{-2}$ sr$^{-2}$]', fontsize=lab_fs)
				if idx == 1:
					ax[idx, 1].set_xlabel('$\\ell$', fontsize=14)
				ax[idx, 1].grid(alpha=0.3)
			ax[idx, 1].plot(
				lb_pred, pf * clx_pred, color=ciber_colors[idx][m], linestyle=linestyle,
				alpha=alpha_line, linewidth=linewidth, label=modllab
			)
			ax[idx, 1].tick_params(labelsize=10)
			ax[idx, 2].tick_params(labelsize=10)
	ax[0, 1].set_xticks([1e3, 1e4, 1e5], ['' for _ in range(3)])
	ax[0, 2].set_xticks([1e3, 1e4, 1e5], ['' for _ in range(3)])
	ax[1, 0].set_visible(False)
	ax[0, 1].legend(loc=2, fontsize=legend_fs)
	ax[1, 1].legend(loc=2, fontsize=legend_fs)

	plt.subplots_adjust(hspace=hspace, wspace=wspace)
	plt.show()

	return fig

def plot_bias_ratios(ell, cl_ig_list, err_cl_ig_list, cl_gg_data, labels=None, 
					  shot_noise_ell_min=20000, shot_noise_ell_max=80000, save_path=None, 
					figsize=(5, 4), ylim=[1e-1, 1e2], xlim=[300, 1e5], label_fs=14, legend_fs=12, 
					colors=['b', 'C3']):
	"""
	Calculates and plots one or more bias ratios on a single figure, with shot noise subtraction.

	This function supports two modes for the galaxy auto-power spectrum (cl_gg_data):
	1. Single Array: A single cl_gg is provided and used for all cross-spectra.
	2. List of Arrays: A list of cl_gg arrays is provided, one for each cross-spectrum.

	Shot noise is estimated by averaging cl_gg at ell >= shot_noise_ell_min and is
	subtracted before calculating the ratio.

	Args:
		ell (np.ndarray): Array of multipole moments, assumed common for all spectra.
		cl_ig_list (list[np.ndarray] or np.ndarray): A single cross-power spectrum or a list of them.
		err_cl_ig_list (list[np.ndarray] or np.ndarray): A single error array or a list of them.
		cl_gg_data (np.ndarray or list[np.ndarray]): A single galaxy auto-power spectrum or a list.
		labels (list[str], optional): A list of labels for the legend.
		colors (list[str], optional): A list of colors for each plot.
		shot_noise_ell_min (float, optional): The minimum ell to use for estimating shot noise.
		save_path (str, optional): File path to save the figure (e.g., 'bias_ratios.png').
	"""
	# --- 1. Input Validation and Standardization ---
	# Ensure list format for looping, even if only one dataset is provided
	if not isinstance(cl_ig_list, list):
		cl_ig_list = [cl_ig_list]
	if not isinstance(err_cl_ig_list, list):
		err_cl_ig_list = [err_cl_ig_list]

	num_datasets = len(cl_ig_list)

	plt.style.use('default')

	# --- 2. Plotting Setup ---
	fig, ax = plt.subplots(figsize=figsize) # Default figsize as requested

	# --- 3. Main Loop for Processing and Plotting Each Dataset ---
	for i in range(num_datasets):
		cl_ig = np.asarray(cl_ig_list[i])
		err_cl_ig = np.asarray(err_cl_ig_list[i])
		
		# Determine which cl_gg to use for this dataset
		if isinstance(cl_gg_data, list):
			# Case 1: A list of cl_gg is provided, one for each cl_ig
			current_cl_gg = np.asarray(cl_gg_data[i])
		else:
			# Case 2: A single cl_gg is provided for all datasets
			current_cl_gg = np.asarray(cl_gg_data)

		# --- Shot Noise Subtraction ---
		noise_indices = np.where((ell >= shot_noise_ell_min)*(ell <= shot_noise_ell_max))[0]
		if noise_indices.size > 0:
			shot_noise = np.mean(current_cl_gg[noise_indices])
			print(f"For dataset '{labels[i]}', estimated shot noise = {shot_noise:.2e}")
			shot_noise_clig = np.mean(cl_ig[noise_indices])
			
		else:
			shot_noise = 0.0
			print(f"Warning for dataset '{labels[i]}': No data found at ell >= {shot_noise_ell_min}. "
				  "Assuming zero shot noise.")
		
		# --- Ratio and Uncertainty Calculation ---
		# Use np.divide to handle potential division by zero

		bias = np.divide(cl_ig, current_cl_gg, out=np.full_like(cl_ig, np.nan), where=current_cl_gg!=0)
		bias_err = np.divide(err_cl_ig, current_cl_gg, out=np.full_like(err_cl_ig, np.nan), where=current_cl_gg!=0)

		# --- Plotting Current Dataset ---
		ax.errorbar(ell, bias, yerr=np.abs(bias_err), fmt='o', color=colors[i], 
					linewidth=1.5, capsize=3, markersize=4, label=labels[i])

	# --- 4. Final Plot Customization ---
#     ax.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.7, label='b = 1')
	ax.set_xlabel(r'Multipole, $\ell$', fontsize=label_fs)
	ax.set_ylabel(r'$b(\ell) = C_\ell^{I \times g} / C_\ell^{g \times g}$', fontsize=label_fs) # Updated y-label
	ax.set_xscale('log')
	ax.legend(fontsize=legend_fs)
	ax.tick_params(axis='both', which='major', labelsize=10)
	ax.set_ylim(ylim)
#     ax.set_yscale('log')
	ax.set_xlim(xlim)
	ax.grid(alpha=0.3)
	plt.tight_layout()

	# --- 5. Output ---
	if save_path:
		plt.savefig(save_path, dpi=300)
		print(f"Plot saved to {save_path}")
	plt.show()


def load_rlmeas_vs_z_DESILS(catname='LS', inst_list=[1, 2], 
						lb_mins = [304, 2000., 10000.], 
						lb_maxs = [2000., 10000., 80000.], 
						startidx=2, endidx=-1):


	zbinedges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	zcen = [0.5*(zbinedges[x]+zbinedges[x+1]) for x in range(len(zbinedges[:-1]))]

	std_rl_largescale = np.zeros((2, len(zbinedges)-1))

	mean_rl_diffscale = np.zeros((len(lb_mins), 2, len(zbinedges)-1))
	std_rl_diffscale = np.zeros((len(lb_mins), 2, len(zbinedges)-1))

	for zidx, zbin in enumerate(zbinedges[:-1]):

		addstr = str(zbinedges[zidx])+'_z_'+str(zbinedges[zidx+1])
		addstr += '_wrandsub_JHlt16_wFFerr'

		acdat = compute_rl_ciber_gal(addstr, inst_list=inst_list, catname=catname)
		
		for inst in inst_list:
			
			lb = acdat[inst-1].lb
			lbrestrict = lb[startidx:endidx]
		
			r_ell = acdat[inst-1].r_ell
			r_ell_unc = acdat[inst-1].r_ell_unc
			
			for lidx in range(len(lb_mins)):
				lbmask = (lbrestrict > lb_mins[lidx])*(lbrestrict < lb_maxs[lidx])
				w = 1 / r_ell_unc[lbmask]**2
				mean_rl_diffscale[lidx, inst-1, zidx] = np.sum(w * r_ell[lbmask]) / np.sum(w)
				std_rl_diffscale[lidx, inst-1, zidx] = np.sqrt(1 / np.sum(w))

	res = dict({'mean_rl_diffscale': mean_rl_diffscale,
				'std_rl_diffscale': std_rl_diffscale,
				'zcen': zcen, 
				'lb_mins': lb_mins,
				'lb_maxs': lb_maxs, 
				'zbinedges': zbinedges})
	
	return res

def load_rlpred_vs_z_DESILS(catname='LS', inst_list=[1, 2], 
						lb_mins = [304, 2000., 10000.], 
						lb_maxs = [2000., 10000., 80000.], 
						jmock_basedir = 'data/jordan_mocks/v2/', 
						ifield_use=8):


	zbinedges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

	all_pred_fpaths = []

	mean_rl_diffscale_pred = np.zeros((len(lb_mins), 2, len(zbinedges)-1))

	for idx, inst in enumerate(inst_list):
	
		tl_pix = np.load('data/fluctuation_data/transfer_function/tl_clx_pix_TM'+str(inst)+'_ifield'+str(ifield_use)+'.npz')['tl_clx_pix']
		
		basepath = jmock_basedir+'mock_ps_pred/TM'+str(inst)+'/field_average/'
		
		pred_fpaths = [basepath+'pred_cls_TM'+str(inst)+'_sdss_z_lt_22.0_CIBERfidmask_zmin='+str(zbinedges[zidx])+'_zmax='+str(zbinedges[zidx+1])+'.npz' for zidx in range(len(zbinedges[:-1]))]

		all_pred_fpaths.append(pred_fpaths)

		for zidx in range(len(zbinedges)-1):

			jmock_pred = np.load(pred_fpaths[zidx])
			lb_pred, r_ell_pred = [jmock_pred[key] for key in ['lb', 'rlx_tracer_full']]
			
			for lidx in range(len(lb_mins)):
				
				lbmask = (lb_pred > lb_mins[lidx])*(lb_pred < lb_maxs[lidx])
				
				mean_rl_diffscale_pred[lidx, inst-1, zidx] = np.mean(r_ell_pred[lbmask])

	return mean_rl_diffscale_pred


def load_rlmeas_vs_z_DESILS_dz02(catname='LS', inst_list=[1, 2],
						lb_mins = [304, 2000., 10000.],
						lb_maxs = [2000., 10000., 80000.],
						startidx=2, endidx=-1):

	zbinedges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
	zcen = [0.5*(zbinedges[x]+zbinedges[x+1]) for x in range(len(zbinedges[:-1]))]

	std_rl_largescale = np.zeros((2, len(zbinedges)-1))

	mean_rl_diffscale = np.zeros((len(lb_mins), 2, len(zbinedges)-1))
	std_rl_diffscale = np.zeros((len(lb_mins), 2, len(zbinedges)-1))

	for zidx, zbin in enumerate(zbinedges[:-1]):

		addstr = str(zbinedges[zidx])+'_z_'+str(zbinedges[zidx+1])
		addstr += '_wrandsub_JHlt16_wFFerr'

		acdat = compute_rl_ciber_gal(addstr, inst_list=inst_list, catname=catname)

		for inst in inst_list:

			lb = acdat[inst-1].lb
			lbrestrict = lb[startidx:endidx]

			r_ell = acdat[inst-1].r_ell
			r_ell_unc = acdat[inst-1].r_ell_unc

			for lidx in range(len(lb_mins)):
				lbmask = (lbrestrict > lb_mins[lidx])*(lbrestrict < lb_maxs[lidx])
				w = 1 / r_ell_unc[lbmask]**2
				mean_rl_diffscale[lidx, inst-1, zidx] = np.sum(w * r_ell[lbmask]) / np.sum(w)
				std_rl_diffscale[lidx, inst-1, zidx] = np.sqrt(1 / np.sum(w))

	res = dict({'mean_rl_diffscale': mean_rl_diffscale,
				'std_rl_diffscale': std_rl_diffscale,
				'zcen': zcen,
				'lb_mins': lb_mins,
				'lb_maxs': lb_maxs,
				'zbinedges': zbinedges})

	return res


def load_rlpred_vs_z_DESILS_dz02(catname='LS', inst_list=[1, 2],
						lb_mins = [304, 2000., 10000.],
						lb_maxs = [2000., 10000., 80000.],
						jmock_basedir = 'data/jordan_mocks/v2/',
						ifield_use=8):

	zbinedges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

	all_pred_fpaths = []

	mean_rl_diffscale_pred = np.zeros((len(lb_mins), 2, len(zbinedges)-1))

	for idx, inst in enumerate(inst_list):

		tl_pix = np.load('data/fluctuation_data/transfer_function/tl_clx_pix_TM'+str(inst)+'_ifield'+str(ifield_use)+'.npz')['tl_clx_pix']

		basepath = jmock_basedir+'mock_ps_pred/TM'+str(inst)+'/field_average/'

		pred_fpaths = [basepath+'pred_cls_TM'+str(inst)+'_sdss_z_lt_22.0_CIBERfidmask_zmin='+str(zbinedges[zidx])+'_zmax='+str(zbinedges[zidx+1])+'.npz' for zidx in range(len(zbinedges[:-1]))]

		all_pred_fpaths.append(pred_fpaths)

		for zidx in range(len(zbinedges)-1):

			jmock_pred = np.load(pred_fpaths[zidx])
			lb_pred, r_ell_pred = [jmock_pred[key] for key in ['lb', 'rlx_tracer_full']]

			for lidx in range(len(lb_mins)):

				lbmask = (lb_pred > lb_mins[lidx])*(lb_pred < lb_maxs[lidx])

				mean_rl_diffscale_pred[lidx, inst-1, zidx] = np.mean(r_ell_pred[lbmask])

	return mean_rl_diffscale_pred


def plot_rl_vs_z_vs_scale_DESILS_dz02(res_meas, mean_rl_diffscale_pred,
								 figsize=(6,6), inst_list=[1, 2], lams=[1.1, 1.8],
								 colors=['b', 'r'],
								 colors_ss=['C0', 'C3'],
								 markersize=8, cmap='inferno_r',
								 ylim=(-0.05, 0.45)):

	fig = plt.figure(figsize=figsize)

	cmap = plt.get_cmap(cmap)
	colors = cmap(np.linspace(0.3, 1, 3))

	for inst in inst_list:
		plt.subplot(2,1,inst)
		plt.text(0.05, 0.37, 'CIBER '+str(lams[inst-1])+' $\\mu$m $\\times$ DESI-LS ($z<22$)', fontsize=16)
		plt.axhline(0, color='grey', alpha=0.2)

		for lidx in range(len(res_meas['lb_mins'])):
			ellstr = str(int(res_meas['lb_mins'][lidx]))+'$<\\ell<$'+str(int(res_meas['lb_maxs'][lidx]))

			plt.errorbar(res_meas['zcen'], np.array(res_meas['mean_rl_diffscale'])[lidx, inst-1],
						  yerr=np.array(res_meas['std_rl_diffscale'])[lidx, inst-1], fmt='o', color=colors[lidx],
						label=ellstr, marker='x', markersize=markersize)

			if lidx==len(res_meas['lb_mins'])-1:
				predlab = 'IGL model prediction\n(Mirocha+25)'
			else:
				predlab = None

			plt.plot(res_meas['zcen'], np.array(mean_rl_diffscale_pred)[lidx, inst-1], color=colors[lidx], linestyle='dashed',
					label=predlab)

		plt.grid(alpha=0.3)
		plt.ylabel('$\\langle r_{\\ell} \\rangle$', fontsize=14)
		plt.ylim(ylim)

		if inst==2:
			plt.tick_params(labelsize=12)
			plt.xlabel('redshift', fontsize=14)
			plt.xticks(res_meas['zbinedges'])

		else:
			plt.legend(fontsize=12, bbox_to_anchor=[-0.02, 1.4], loc=2, ncol=2)
			plt.xticks(res_meas['zbinedges'], ['' for _ in range(len(res_meas['zbinedges']))])

	plt.show()

	return fig


def plot_spectrum_ratios(lb, mag_lims, fieldav_clg_vs_mag, fieldav_clg_vs_mag_norand, 
						  fieldav_clx_vs_mag, fieldav_clx_vs_mag_norand, 
						  fieldav_clxerr_vs_mag, fieldav_clxerr_vs_mag_norand, 
						inst_list=[1, 2], figsize=(8, 5), nrows=1, ncols = 3,
						ylim=[1e-1, 1e1], catname='HSC', magstr_gal='i', 
						lab_fs=12, bbox_to_anchor=[0.0, 1.5], ncol=4, legend_fs=12, title_fs=14, 
						ylim_gal=[0.2, 5.0], ylim_clx=[1e-1, 1e1], ylim_dclx=[0.3, 3.0], 
						xlim=[250, 1e5], colors=None, sharey=False, plot_unc=False):
	
	fig, ax = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, sharex=True, sharey=sharey)

	bandstrs = ['1.1 $\\mu$m', '1.8 $\\mu$m']
	
	
	if colors is None:
		colors = ['C'+str(x) for x in range(len(mag_lims))]

	for m, maglim in enumerate(mag_lims):
		
		for inst in inst_list:

			
			if inst==1:
				clxlab = catname+' $'+magstr_gal+'<'+str(int(maglim))+'$'
			else:
				clxlab = None
				
			
			ax[inst].plot(lb, 
						  fieldav_clx_vs_mag_norand[m][inst-1] / 
						  fieldav_clx_vs_mag[m][inst-1], 
						  label=clxlab, linestyle='solid',
							zorder=30.-mag_lims[m], color=colors[m])
			
			
			if m==0 and plot_unc:
				ax[inst].plot(lb, 1.-(fieldav_clxerr_vs_mag[m][inst-1]/fieldav_clx_vs_mag[m][inst-1]), color='k', linestyle='dotted')
				ax[inst].plot(lb, 1.+(fieldav_clxerr_vs_mag[m][inst-1]/fieldav_clx_vs_mag[m][inst-1]), color='k', linestyle='dotted')

			
			ax[inst].set_xscale('log')
			ax[inst].set_ylim(ylim_clx)
			ax[inst].set_xlim(xlim)
			ax[inst].grid(alpha=0.3, which='both')
			ax[inst].set_title('CIBER '+bandstrs[inst-1]+' $\\times$ '+catname, fontsize=title_fs)
			ax[inst].set_xlabel('$\\ell$', fontsize=lab_fs)


		ax[0].plot(lb, 
					 fieldav_clg_vs_mag_norand[m] / fieldav_clg_vs_mag[m], zorder=30.-mag_lims[m], 
				  color=colors[m])
		
		if m==0 and plot_unc:
			ax[0].plot(lb, 1.-(fieldav_clgerr_vs_mag[m]/fieldav_clg_vs_mag[m]), color='k', linestyle='dotted')
			ax[0].plot(lb, 1.+(fieldav_clgerr_vs_mag[m]/fieldav_clg_vs_mag[m]), color='k', linestyle='dotted')

		# Configure Auto Spectrum Plot
		ax[0].set_xscale('log')
		ax[0].set_ylabel('$\\frac{C_{\\ell} (uncorr.)}{C_{\\ell} (corr.)}$', fontsize=lab_fs)

		ax[0].set_title(catname+' auto', fontsize=title_fs)
		ax[0].set_ylim(ylim_gal)
		ax[0].set_xlim(xlim)
		
		ax[0].set_xlabel('$\\ell$', fontsize=lab_fs)

		ax[0].grid(alpha=0.3, which='both')
		
	ax[1].legend(bbox_to_anchor=bbox_to_anchor, fontsize=legend_fs, ncol=ncol)


	plt.subplots_adjust(hspace=0.01, wspace=0.2)
	plt.show()
	
	return fig


def plot_spectrum_with_fraction(lb, est, true, title, figsize=(5, 5), height_ratios=[3, 1], \
							   color_est = 'C0', color_true='k', markersize=3, capsize=3, 
							   ylim=[0., 2.0], lab_fs=14, tick_fs=12, legend_fs=12, title_fs=14, 
							   ylim_ps=[1e-3, 3e1]):
	
	pf = lb * (lb + 1) / (2 * np.pi)

	fig, (ax1, ax2) = plt.subplots(
		2, 1, figsize=figsize, gridspec_kw={'height_ratios': height_ratios}, sharex=True
	)
	
	# --- Top panel: Power spectrum ---
	ax1.errorbar(lb, pf * np.median(est, axis=0),
				 yerr=pf * np.std(est, axis=0),
				 capsize=capsize, markersize=markersize, fmt='o', color=color_est, label='Recovered', zorder=5)
	ax1.errorbar(lb, pf * np.median(true, axis=0),
				 yerr=pf * np.std(true, axis=0),
				 capsize=capsize, markersize=markersize, fmt='o', color=color_true, label='Input')
	
	ax1.set_xscale('log')
	ax1.set_yscale('log')
	ax1.set_xlim(150, 1e5)
	ax1.set_ylim(ylim_ps)
#     ax1.set_ylim(1e-3, 5.0)  # Adjust depending on your data
	ax1.grid(alpha=0.3)
	ax1.set_ylabel(r"$\ell(\ell+1)C_\ell / 2\pi$", fontsize=lab_fs)
	ax1.set_title(title, fontsize=title_fs)
	ax1.legend(fontsize=legend_fs)
	ax1.tick_params(labelsize=tick_fs)
	
	# --- Bottom panel: Fractional recovery ---
	frac = est / true
	ax2.errorbar(lb, np.median(frac, axis=0),
				 yerr=np.std(frac, axis=0),
				 capsize=capsize, markersize=markersize, fmt='o', color=color_est)
	
	ax2.axhline(1.0, color='k', linestyle='--', alpha=0.7)
	ax2.set_xscale('log')
	ax2.set_xlim(150, 1e5)
	ax2.set_ylim(ylim)
	ax2.grid(alpha=0.3)
	ax2.set_xlabel(r"$\ell$", fontsize=lab_fs)
	ax2.set_ylabel('$\\hat{C}_{\\ell}/C_{\\ell}^{input} - 1$', fontsize=lab_fs)
	ax2.tick_params(labelsize=tick_fs)

	plt.tight_layout()
	plt.show()
	
	return fig



def field_consistency_gal_cross(catname, addstr, ps_type='cross', inst_list=[1, 2], ifield_list=[4, 5, 6, 7, 8],
								 startidx=0, endidx=None,
								 ell_min=300, ell_max=10000, figsize=(6, 4), use_zscore=False, 
							   ylim=[-6, 6], textxpos=850, textypos=4.0, yticks=[-4, -2, 0, 2, 4]):
	"""
	Compute PTE values for per-field spectra relative to the field average,
	restricted to ell_min < ell < ell_max.
	Also makes a fractional deviation plot for all bands in one figure.

	Parameters
	----------
	catname : str
		Name of the galaxy catalog.
	addstr : str
		Additional string for file naming.
	ps_type : str, optional
		Type of power spectrum to analyze. Can be 'auto' for galaxy auto-power
		or 'cross' for CIBER x galaxy cross-power. Defaults to 'cross'.
	inst_list : list, optional
		List of CIBER instruments to process.
	ifield_list : list, optional
		List of CIBER fields to process.
	startidx : int, optional
		Starting index for ell binning.
	endidx : int, optional
		Ending index for ell binning.
	ell_min : int, optional
		Minimum ell value for analysis.
	ell_max : int, optional
		Maximum ell value for analysis.
	figsize : tuple, optional
		Size of the output figure.
	use_zscore : bool, optional
		If True, plot z-score (chi) instead of fractional deviation.
	"""

	cbps = CIBER_PS_pipeline()

	bandstr_list = ['J', 'H']
	lams = [1.1, 1.8]

	pte_results = np.zeros((len(inst_list), len(ifield_list)))

	# Create one figure with two panels: top = inst=1, bottom = inst=2
	fig, axes = plt.subplots(len(inst_list), 1, figsize=figsize, sharex=True)
	if len(inst_list) == 1:
		axes = [axes]  # make iterable if only one panel

	for idx, inst in enumerate(inst_list):

		# Load your data
		cgps_file = load_ciber_gal_ps(inst, catname, addstr=addstr)
		lb, all_cl_gal, all_clerr_gal, all_cl_cross, all_clerr_cross = [
			cgps_file[k] for k in ['lb', 'all_cl_gal', 'all_clerr_gal', 'all_cl_cross', 'all_clerr_cross']
		]

		if idx==0:
			# Define ell mask once
			ell_mask = (lb >= ell_min) & (lb <= ell_max)
			if endidx is not None:
				ell_mask &= (np.arange(len(lb)) < endidx)
			if startidx is not None:
				ell_mask &= (np.arange(len(lb)) >= startidx)

			all_chi_results = np.zeros((len(inst_list), len(ifield_list), np.sum(ell_mask)))


		# === SELECT DATA AND COMPUTE WEIGHTED FIELD AVERAGE ===
		if ps_type == 'cross':
			# For cross-spectra, we need to estimate uncertainties using CIBER auto and galaxy auto.
			ciber_auto = _load_ciber_auto_file(bandstr_list[idx])
			lb_auto, cl_auto, clerr_auto = [ciber_auto[key] for key in ['lb', 'fieldav_cl', 'fieldav_clerr']]

			# Get unweighted field averages needed for the uncertainty estimation
			_, _, _, fieldav_cl_gal_unw, _ = mini_proc_clav(
				all_cl_gal, all_clerr_gal, lb, startidx, endidx, mode='auto'
			)
			_, _, _, fieldav_cl_cross_unw, _ = mini_proc_clav(
				all_cl_cross, all_clerr_cross, lb, startidx, endidx, mode='cross'
			)

			# Compute flat field bias correction for each field
			ifield_list_full = [4, 5, 6, 7, 8]
			mean_norms = [cbps.zl_levels_ciber_fields[inst][cbps.ciber_field_dict[ifield]] 
						for ifield in ifield_list_full]
			weights_ff = cbps.compute_ff_weights(inst, mean_norms, ifield_list_full, photon_noise=True)
			ff_bias_factors = compute_ff_bias(mean_norms, weights=weights_ff)

			# Estimate per-field uncertainties for the cross-spectra with FF bias correction
			perf_clerr_est = np.zeros_like(all_clerr_cross)
			for fieldidx, ifield in enumerate(ifield_list):
				idx_full = ifield_list_full.index(ifield)
				perf_clerr_est[fieldidx] = estimate_cross_uncertainties(
					lb, fieldav_cl_cross_unw, all_clerr_cross[fieldidx],
					cl_auto*ff_bias_factors[idx_full], fieldav_cl_gal_unw, 1, startidx=2, endidx=-1
				)

			# Compute the properly weighted field average using the new uncertainties
			fieldav_cl, fieldav_clerr, _, _ = compute_field_averaged_power_spectrum(
				all_cl_cross.copy(), per_field_dcls=perf_clerr_est.copy())

			# Set generic variables for the main loop
			all_cl_to_use = all_cl_cross
			perf_clerr_to_use = perf_clerr_est
			if catname=='LS':
				catname_use='DESI-LS'
			else:
				catname_use = catname

			plot_text = f'CIBER {lams[idx]} $\\mu$m $\\times$ {catname_use}'
		elif ps_type == 'auto':
			# For auto-spectra, we use the provided errors to compute the weighted average.
#             fieldav_cl, fieldav_clerr, _, _ = compute_field_averaged_power_spectrum(
#                 all_cl_gal.copy(), per_field_dcls=all_clerr_gal.copy())
			
			
			# Get unweighted field averages needed for the uncertainty estimation
			_, _, _, fieldav_cl, fieldav_clerr_gal = mini_proc_clav(
				all_cl_gal, all_clerr_gal, lb, startidx, endidx, mode='auto'
			)
			
			fieldav_cl = np.mean(all_cl_gal, axis=0)

			n_fields = len(ifield_list)
			scaled_fieldav_clerr = fieldav_clerr_gal * np.sqrt(n_fields)

			# Copy the scaled uncertainty to all fields for perf_clerr_to_use
			perf_clerr_to_use = np.tile(scaled_fieldav_clerr, (n_fields, 1))

			
			# Set generic variables for the main loop
			all_cl_to_use = all_cl_gal
			
			if catname=='LS':
				plot_text = f'DESI-{catname} auto-power'

			else:
#             perf_clerr_to_use = all_clerr_gal
				plot_text = f'{catname} auto-power'

		else:
			raise ValueError("ps_type must be either 'auto' or 'cross'")

		ax = axes[idx]  # select subplot for this band
		offsets = np.linspace(-0.05, 0.05, len(ifield_list))

		# Compute PTE and plot fractional deviations for the selected ps_type
		for fieldidx, ifield in enumerate(ifield_list):
			cl_field = all_cl_to_use[fieldidx, ell_mask]
			cl_mean = fieldav_cl[ell_mask]
			cl_err = perf_clerr_to_use[fieldidx, ell_mask]

			# PTE calculation
			chi_perbp = (cl_field - cl_mean) / cl_err
			chi2_perbp = chi_perbp ** 2
			chi2_val = np.sum(chi2_perbp)
			dof = len(cl_field)
			pte = 1 - chi2.cdf(chi2_val, dof)
			pte_results[idx, fieldidx] = pte
			all_chi_results[idx, fieldidx] = chi_perbp

			print(f"Inst {inst}, Field {ifield}, Type '{ps_type}': chi2={chi2_val:.2f}, dof={dof}, PTE={pte:.3f}")

			label = cbps.ciber_field_dict[ifield]
			label += f' (PTE={np.round(pte, 3)})' if pte > 1e-3 else ' (PTE<0.001)'

			# Plotting values
			frac_dev = (cl_field - cl_mean) / cl_mean
			frac_err = cl_err / cl_mean
			zscore = (cl_field - cl_mean) / cl_err
			lb_shifted = lb[ell_mask] * (1 + offsets[fieldidx])

			if use_zscore:
				ax.scatter(lb_shifted, zscore, label=label, alpha=0.8)
			else:
				ax.errorbar(lb_shifted, frac_dev, yerr=frac_err,
							fmt='o', label=label, alpha=0.8)

		# Formatting for the plot
		ell_centers = lb[ell_mask]
		ell_edges = np.zeros(len(ell_centers) + 1)
		ell_edges[1:-1] = 0.5 * (ell_centers[1:] + ell_centers[:-1])
		ell_edges[0] = ell_centers[0] - (ell_centers[1] - ell_centers[0]) / 2
		ell_edges[-1] = ell_centers[-1] + (ell_centers[-1] - ell_centers[-2]) / 2

		for i in range(len(ell_centers)):
			if i % 2 == 0:
				ax.axvspan(ell_edges[i], ell_edges[i+1], color='gray', alpha=0.1, zorder=0)

		ax.axhline(0, color='k', lw=1)
		ax.set_xscale('log')
		ax.set_ylabel('$\\Delta C_{\\ell}^{i}/\\sigma(C_{\\ell}^i)$' if use_zscore else '$\\Delta C_{\\ell}^{i}/\\overline{C}_{\\ell}  - 1$', fontsize=14)
		ax.grid(alpha=0.3)
		ax.set_xlim(250, ell_max * 1.2)
		ax.legend(fontsize=12, loc=4, bbox_to_anchor=[1.5, 0.1])
		ax.text(textxpos, ylim[1]-0.2*(ylim[1]-ylim[0]), plot_text, fontsize=16)
		ax.set_yticks(yticks)
		ax.set_ylim(ylim)
		ax.tick_params(labelsize=12)

	axes[-1].set_xlabel(r'$\ell$', fontsize=14)
	plt.subplots_adjust(hspace=0.05)
	plt.show()

	return fig, pte_results, all_chi_results

def plot_deconvolution_comparison(results, figsize=(7, 4)):
	"""
	Plot comparison between direct inversion and constrained optimization.
	"""
	
	z_true = results['z_true']
		
	fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
	
	# Top panel: Solutions
	axes[0].errorbar(z_true, results['original_measurements'], yerr=results['original_errors'],
					fmt='o', color='black', label='Photo-z measurements')
	
	axes[0].errorbar(z_true, results['direct']['solution'], yerr=results['direct']['errors'],
					fmt='-', color='blue', linewidth=2, label='Direct inversion')
	
	axes[0].errorbar(z_true, results['constrained']['solution'], yerr=results['constrained']['errors'],
					fmt='-', color='red', linewidth=2, label='Non-negative constrained')
	
	axes[0].set_ylabel('$b_I(z)\\times dI/dz$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=14)
	axes[0].set_title('Deconvolution Methods Comparison', fontsize=16)
	axes[0].legend(fontsize=12)
	axes[0].grid(alpha=0.3)
	
	# Bottom panel: Reconstructed measurements
	axes[1].errorbar(z_true, results['original_measurements'], yerr=results['original_errors'],
					fmt='o', color='black', label='Original measurements')
	
	axes[1].plot(z_true, results['direct']['reconstructed'], 's-', color='blue',
				markersize=8, alpha=0.7, label='Direct reconstruction')
	
	axes[1].plot(z_true, results['constrained']['reconstructed'], 'd-', color='red',
				markersize=8, alpha=0.7, label='Constrained reconstruction')
	
	axes[1].set_xlabel('Redshift $z$', fontsize=14)
	axes[1].set_ylabel('Photo-z bin values', fontsize=14)
	axes[1].set_title('Reconstruction Quality Check', fontsize=16)
	axes[1].legend(fontsize=12)
	axes[1].grid(alpha=0.3)
	
	plt.tight_layout()
	plt.show()


def load_trilegal_unresolved_isl_cl(
	inst,
	ifield=8,
	maglim_vega=16.0,
	datestr='112022',
	stat='mean',
	basepath=None,
	max_mag_delta=0.26,
):
	"""Load unresolved ISL C_ell from TRILEGAL residual spectra at a Vega mag cut.

	Returns
	-------
	lb : np.ndarray
		Multipole bin centers.
	cl_isl : np.ndarray
		Residual ISL auto power for the requested mask depth.
	used_col : str
		Column name used from FITS table (nearest available mag cut).
	"""
	if stat not in {'mean', 'median'}:
		raise ValueError("stat must be 'mean' or 'median'")

	if basepath is None:
		basepath = (
			Path(getattr(config, 'ciber_basepath', '.'))
			/ 'data'
			/ 'ciber_mocks'
			/ str(datestr)
			/ f'TM{inst}'
			/ 'isl_resid_ps'
		)
	else:
		basepath = Path(basepath)

	pat = f'cls_isl_vs_maglim_ifield{ifield}_inst{inst}_simidx*_Vega_magcut.fits'
	fpaths = sorted(basepath.glob(pat))
	if not fpaths:
		raise FileNotFoundError(
			f'No TRILEGAL residual spectra found in {basepath} for pattern {pat}'
		)

	lb = None
	cl_stack = []
	used_cols = []
	for fp in fpaths:
		with fits.open(str(fp)) as hdul:
			tab = hdul[1].data
			if lb is None:
				lb = np.asarray(tab['lb'], dtype=float)

			mag_cols = [c for c in tab.columns.names if c.startswith('cl_maglim_')]
			if not mag_cols:
				continue

			all_mags = np.array([float(c.replace('cl_maglim_', '')) for c in mag_cols], dtype=float)
			col_idx = int(np.argmin(np.abs(all_mags - float(maglim_vega))))
			if np.abs(all_mags[col_idx] - float(maglim_vega)) > float(max_mag_delta):
				continue

			used_col = mag_cols[col_idx]
			cl_stack.append(np.asarray(tab[used_col], dtype=float))
			used_cols.append(used_col)

	if not cl_stack:
		raise ValueError(
			f'No TRILEGAL ISL files contained a mag-cut column near {maglim_vega} (delta<={max_mag_delta}).'
		)

	cl_stack = np.asarray(cl_stack, dtype=float)
	if stat == 'mean':
		cl_isl = np.nanmean(cl_stack, axis=0)
	else:
		cl_isl = np.nanmedian(cl_stack, axis=0)

	used_col = max(set(used_cols), key=used_cols.count)

	return lb, cl_isl, used_col


def estimate_r_ell_with_added_isl_from_ratio(r_ell_igl, cii_isl_over_igl, first_order=False):
	"""Return ISL-adjusted r_ell given ratio C_ell^ISL / C_ell^II,IGL.

	If ``first_order`` is True, use small-ratio approximation:
	r_new ~= r_old * (1 - 0.5 * ratio).
	Otherwise use exact scaling:
	r_new = r_old / sqrt(1 + ratio).
	"""
	r_ell_igl = np.asarray(r_ell_igl, dtype=float)
	ratio = np.asarray(cii_isl_over_igl, dtype=float)
	if ratio.ndim == 0:
		ratio = np.full_like(r_ell_igl, float(ratio))
	elif ratio.shape != r_ell_igl.shape:
		raise ValueError("ratio must be scalar or same shape as r_ell_igl")

	ratio = np.maximum(ratio, 0.0)
	if first_order:
		scale = 1.0 - 0.5 * ratio
	else:
		scale = 1.0 / np.sqrt(1.0 + ratio)
	return r_ell_igl * scale


def estimate_r_ell_with_added_isl_from_cl(r_ell_igl, cii_igl, cii_isl, first_order=False):
	"""Return ISL-adjusted r_ell given C_ell^II,IGL and C_ell^II,ISL.

	This computes ratio = C_ell^ISL / C_ell^IGL and forwards to
	``estimate_r_ell_with_added_isl_from_ratio``.
	"""
	r_ell_igl = np.asarray(r_ell_igl, dtype=float)
	cii_igl = np.asarray(cii_igl, dtype=float)
	cii_isl = np.asarray(cii_isl, dtype=float)
	if cii_igl.shape != r_ell_igl.shape or cii_isl.shape != r_ell_igl.shape:
		raise ValueError("r_ell_igl, cii_igl, and cii_isl must have matching shapes")

	ratio = np.divide(
		cii_isl,
		cii_igl,
		out=np.zeros_like(cii_isl, dtype=float),
		where=cii_igl > 0,
	)
	return estimate_r_ell_with_added_isl_from_ratio(
		r_ell_igl,
		ratio,
		first_order=first_order,
	)


def compare_r_ell_hsc_LS_zlt1(
	figsize=(4, 6),
	ls_addstr='0.0_z_1.0_wrandsub_JHlt16',
	hsc_addstr='hsc_ilt24.0_zlt1_wrandsub',
	wise_addstr='unWISE_W1lt17p5_JHlt16_wFFerr',
	startidx=2,
	endidx=-1,
	title_fs=13,
	lab_fs=14,
	legend_fs=12,
	textxpos=300,
	textypos=0.8,
	text_fs=14,
	ylim=[-0.1, 1.1],
	xlim=[250, 1.1e5],
	grid_alpha=0.3,
	capsize=3,
	capthick=1.5,
	markersize=3,
	ls_plotstr='Legacy Survey ($z_{\\rm AB}<22$, $z_{\\rm phot}<1$)',
	hsc_plotstr='HSC ($i_{\\rm AB}<25$, $z_{\\rm phot}<1$)',
	wise_plotstr='unWISE ($W1<17.5$)',
	hsc_pred_fpaths=None,
	ls_pred_fpaths=None,
	wise_pred_fpaths=None,
	alpha=0.8,
	tl_pix_correct=True,
	ifield_use=8,
	bbox_to_anchor=[0.0, 1.3],
	ell_max_mean=2000.0,
	include_wise=True,
	plot_isl_adjusted=False,
	isl_cii_over_igl=None,
	isl_cii_over_igl_by_tracer=None,
	isl_first_order=False,
	isl_linestyle='dashed',
	isl_alpha=0.7,
	isl_label='IGL + ISL (approx)',
	isl_use_trilegal=False,
	isl_trilegal_datestr='112022',
	isl_trilegal_maglim_vega=16.0,
	isl_trilegal_stat='mean',
	isl_trilegal_basepath=None,
):
	def _resolve_isl_ratio(tracer_key):
		if isl_cii_over_igl_by_tracer is not None and tracer_key in isl_cii_over_igl_by_tracer:
			return isl_cii_over_igl_by_tracer[tracer_key]
		return isl_cii_over_igl

	def _apply_isl_ratio(r_ell_pred, ratio):
		if ratio is None:
			return None

		ratio_arr = np.asarray(ratio, dtype=float)
		if ratio_arr.ndim == 0:
			ratio_arr = np.full_like(r_ell_pred, float(ratio_arr))
		elif ratio_arr.shape != r_ell_pred.shape:
			raise ValueError(
				"ISL ratio shape mismatch: expected scalar or array matching prediction length"
			)

		ratio_arr = np.maximum(ratio_arr, 0.0)
		if isl_first_order:
			scale = 1.0 - 0.5 * ratio_arr
		else:
			scale = 1.0 / np.sqrt(1.0 + ratio_arr)
		return r_ell_pred * scale

	def _infer_cii_from_prediction(pred):
		cross = np.asarray(pred['cross'], dtype=float)
		gal_auto = np.asarray(pred['gal_auto'], dtype=float)
		r_ell = np.asarray(pred['rlx_tracer_full'], dtype=float)
		r_denom = np.square(r_ell)
		return np.divide(
			np.square(cross),
			r_denom * gal_auto,
			out=np.zeros_like(cross, dtype=float),
			where=(r_denom > 0) & (gal_auto > 0),
		)

	lams = [1.1, 1.8]
	inst_list = [1, 2]

	ls_auto_cross = compute_rl_ciber_gal(
		ls_addstr,
		catname='LS',
		tl_pix_correct=tl_pix_correct,
		ifield_use=ifield_use,
	)
	hsc_auto_cross = compute_rl_ciber_gal(
		hsc_addstr,
		catname='HSC',
		tl_pix_correct=tl_pix_correct,
		ifield_use=ifield_use,
	)

	wise_auto_cross = None
	if include_wise:
		wise_auto_cross = compute_rl_ciber_gal(
			wise_addstr,
			catname='WISE',
			tl_pix_correct=tl_pix_correct,
			ifield_use=ifield_use,
		)

	fig, ax = plt.subplots(figsize=figsize, ncols=1, nrows=2, sharex=True)

	n_tracers = 3 if include_wise else 2
	all_rlmean, all_rlmeanunc = [np.zeros((len(inst_list), n_tracers)) for _ in range(2)]
	for idx, inst in enumerate(inst_list):
		title = 'CIBER ' + str(lams[idx]) + ' $\\mu$m $\\times$ $\\delta_g$'
		ax[idx].text(textxpos, textypos, title, fontsize=text_fs)

		lb = ls_auto_cross[idx].lb
		r_ell_ls = ls_auto_cross[idx].r_ell
		r_ell_unc_ls = ls_auto_cross[idx].r_ell_unc
		r_ell_hsc = hsc_auto_cross[idx].r_ell
		r_ell_unc_hsc = hsc_auto_cross[idx].r_ell_unc

		ax[idx].errorbar(
			lb[startidx:endidx],
			r_ell_ls,
			yerr=r_ell_unc_ls,
			fmt='o',
			capsize=capsize,
			markersize=markersize,
			capthick=capthick,
			color='C0',
			label=ls_plotstr,
		)
		ax[idx].errorbar(
			lb[startidx:endidx],
			r_ell_hsc,
			yerr=r_ell_unc_hsc,
			fmt='o',
			capsize=capsize,
			markersize=markersize,
			capthick=capthick,
			color='C1',
			label=hsc_plotstr,
		)

		all_rl = [r_ell_ls, r_ell_hsc]
		all_rl_unc = [r_ell_unc_ls, r_ell_unc_hsc]

		if include_wise and wise_auto_cross is not None:
			r_ell_wise = wise_auto_cross[idx].r_ell
			r_ell_unc_wise = wise_auto_cross[idx].r_ell_unc
			ax[idx].errorbar(
				lb[startidx:endidx],
				r_ell_wise,
				yerr=r_ell_unc_wise,
				fmt='o',
				capsize=capsize,
				markersize=markersize,
				capthick=capthick,
				color='C2',
				label=wise_plotstr,
			)
			all_rl.append(r_ell_wise)
			all_rl_unc.append(r_ell_unc_wise)

		lbmean_mask = lb[startidx:endidx] < ell_max_mean
		for x in range(len(all_rl)):
			rlmean, rlunc = weighted_mean_and_uncertainty(
				all_rl[x][lbmean_mask],
				all_rl_unc[x][lbmean_mask],
			)
			all_rlmean[idx, x] = rlmean
			all_rlmeanunc[idx, x] = rlunc

		ax[idx].set_xscale('log')
		ax[idx].set_ylim(ylim)
		ax[idx].set_ylabel('$r_{\\ell}=C_{\\ell}^{Ig}/\\sqrt{C_{\\ell}^{gg}C_{\\ell}^{II}}$', fontsize=14)
		if idx == 1:
			ax[idx].set_xlabel('$\\ell$', fontsize=lab_fs)
		ax[idx].grid(alpha=grid_alpha)
		ax[idx].set_xlim(xlim)

		ax[idx].axhspan(ylim[0], 0, facecolor='grey', alpha=0.2)
		ax[idx].axhspan(1, ylim[1], facecolor='grey', alpha=0.2)

		if tl_pix_correct:
			tl_pix = np.load(
				'data/fluctuation_data/transfer_function/tl_clx_pix_TM'
				+ str(inst)
				+ '_ifield'
				+ str(ifield_use)
				+ '.npz'
			)['tl_clx_pix']
		else:
			tl_pix = np.ones_like(lb)

		trilegal_isl = None
		trilegal_lb = None
		if plot_isl_adjusted and isl_use_trilegal:
			try:
				trilegal_lb, trilegal_isl, _ = load_trilegal_unresolved_isl_cl(
					inst=inst,
					ifield=ifield_use,
					maglim_vega=isl_trilegal_maglim_vega,
					datestr=isl_trilegal_datestr,
					stat=isl_trilegal_stat,
					basepath=isl_trilegal_basepath,
				)
			except Exception as e:
				print(f'WARNING: could not load TRILEGAL unresolved ISL for TM{inst}: {e}')
				trilegal_isl = None
				trilegal_lb = None

		if ls_pred_fpaths is not None:
			jmock_pred = np.load(ls_pred_fpaths[idx])
			lb_pred, r_ell_ls_pred = [jmock_pred[key] for key in ['lb', 'rlx_tracer_full']]
			r_ell_ls_pred /= tl_pix
			ax[idx].plot(
				lb_pred,
				r_ell_ls_pred,
				color='C0',
				linestyle='dotted',
				label='IGL prediction',
				alpha=alpha,
			)
			if plot_isl_adjusted:
				ratio = _resolve_isl_ratio('ls')
				if ratio is None and trilegal_isl is not None and trilegal_lb is not None:
					cii_igl_pred = _infer_cii_from_prediction(jmock_pred)
					cl_isl_interp = np.interp(lb_pred, trilegal_lb, trilegal_isl)
					ratio = np.divide(
						cl_isl_interp,
						cii_igl_pred,
						out=np.zeros_like(cl_isl_interp, dtype=float),
						where=cii_igl_pred > 0,
					)
				r_ell_ls_isl = _apply_isl_ratio(r_ell_ls_pred, ratio)
				if r_ell_ls_isl is not None:
					ax[idx].plot(
						lb_pred,
						r_ell_ls_isl,
						color='C0',
						linestyle=isl_linestyle,
						label=isl_label,
						alpha=isl_alpha,
					)

		if hsc_pred_fpaths is not None:
			jmock_pred = np.load(hsc_pred_fpaths[idx])
			lb_pred, r_ell_hsc_pred = [jmock_pred[key] for key in ['lb', 'rlx_tracer_full']]
			r_ell_hsc_pred /= tl_pix
			ax[idx].plot(lb_pred, r_ell_hsc_pred, color='C1', linestyle='dotted', alpha=alpha)
			if plot_isl_adjusted:
				ratio = _resolve_isl_ratio('hsc')
				if ratio is None and trilegal_isl is not None and trilegal_lb is not None:
					cii_igl_pred = _infer_cii_from_prediction(jmock_pred)
					cl_isl_interp = np.interp(lb_pred, trilegal_lb, trilegal_isl)
					ratio = np.divide(
						cl_isl_interp,
						cii_igl_pred,
						out=np.zeros_like(cl_isl_interp, dtype=float),
						where=cii_igl_pred > 0,
					)
				r_ell_hsc_isl = _apply_isl_ratio(r_ell_hsc_pred, ratio)
				if r_ell_hsc_isl is not None:
					ax[idx].plot(
						lb_pred,
						r_ell_hsc_isl,
						color='C1',
						linestyle=isl_linestyle,
						alpha=isl_alpha,
					)

		if include_wise and wise_pred_fpaths is not None:
			jmock_pred = np.load(wise_pred_fpaths[idx])
			lb_pred, r_ell_wise_pred = [jmock_pred[key] for key in ['lb', 'rlx_tracer_full']]
			r_ell_wise_pred /= tl_pix
			ax[idx].plot(lb_pred, r_ell_wise_pred, color='C2', linestyle='dotted', alpha=alpha)
			if plot_isl_adjusted:
				ratio = _resolve_isl_ratio('wise')
				if ratio is None and trilegal_isl is not None and trilegal_lb is not None:
					cii_igl_pred = _infer_cii_from_prediction(jmock_pred)
					cl_isl_interp = np.interp(lb_pred, trilegal_lb, trilegal_isl)
					ratio = np.divide(
						cl_isl_interp,
						cii_igl_pred,
						out=np.zeros_like(cl_isl_interp, dtype=float),
						where=cii_igl_pred > 0,
					)
				r_ell_wise_isl = _apply_isl_ratio(r_ell_wise_pred, ratio)
				if r_ell_wise_isl is not None:
					ax[idx].plot(
						lb_pred,
						r_ell_wise_isl,
						color='C2',
						linestyle=isl_linestyle,
						alpha=isl_alpha,
					)

		if idx == 0:
			ax[idx].legend(loc=2, bbox_to_anchor=bbox_to_anchor, ncol=1, fontsize=legend_fs)

	plt.subplots_adjust(wspace=0.05, hspace=0.05)
	plt.show()

	return fig, all_rlmean, all_rlmeanunc


def compare_r_ell_hsc_ls_2x2(
	figsize=(7, 5.2),
	ls_addstr='0.0_z_1.0_wrandsub_JHlt16',
	hsc_addstr='hsc_ilt24.0_zlt1_wrandsub',
	startidx=2,
	endidx=-1,
	title_fs=13,
	lab_fs=14,
	legend_fs=12,
	textxpos=1100,
	textypos=1.0,
	text_fs=11,
	ylim=[-0.15, 1.15],
	xlim=[250, 1.1e5],
	grid_alpha=0.3,
	capsize=3,
	capthick=1.4,
	markersize=3,
	ls_plotstr='DESI-LS ($z_{\\rm AB}<22$, $z_{\\rm phot}<1$)',
	hsc_plotstr='HSC ($i_{\\rm AB}<25$, $z_{\\rm phot}<1$)',
	ls_pred_fpaths=None,
	hsc_pred_fpaths=None,
	alpha=0.8,
	tl_pix_correct=True,
	ifield_use=8,
	plot_isl_adjusted=True,
	isl_first_order=False,
	isl_linestyle='dashed',
	isl_alpha=0.8,
	isl_label='IGL + unmasked ISL',
	isl_use_trilegal=True,
	isl_trilegal_datestr='112022',
	isl_trilegal_maglim_vega=16.0,
	isl_trilegal_stat='mean',
	isl_trilegal_basepath=None,
	save_path=None,
	show=False,
):
	"""Plot LS/HSC r_ell in a 2x2 layout (one panel per data combination).

	Rows correspond to CIBER wavelengths (1.1, 1.8 um), columns correspond to
	tracers (DESI-LS, HSC).
	"""
	lams = [1.1, 1.8]
	inst_list = [1, 2]

	ls_auto_cross = compute_rl_ciber_gal(
		ls_addstr,
		catname='LS',
		tl_pix_correct=tl_pix_correct,
		ifield_use=ifield_use,
	)
	hsc_auto_cross = compute_rl_ciber_gal(
		hsc_addstr,
		catname='HSC',
		tl_pix_correct=tl_pix_correct,
		ifield_use=ifield_use,
	)

	fig, ax = plt.subplots(figsize=figsize, ncols=2, nrows=2, sharex=True, sharey=True)

	legend_handles = None

	def _infer_cii_from_prediction_local(pred):
		cross = np.asarray(pred['cross'], dtype=float)
		gal_auto = np.asarray(pred['gal_auto'], dtype=float)
		r_ell = np.asarray(pred['rlx_tracer_full'], dtype=float)
		r_denom = np.square(r_ell)
		return np.divide(
			np.square(cross),
			r_denom * gal_auto,
			out=np.zeros_like(cross, dtype=float),
			where=(r_denom > 0) & (gal_auto > 0),
		)

	for ridx, inst in enumerate(inst_list):
		lb = ls_auto_cross[ridx].lb
		# Keep DESI-LS at default b/r while shifting HSC to nearby blue/red shades.
		if inst == 1:
			desi_color = 'b'
			hsc_color = 'dodgerblue'
		else:
			desi_color = 'r'
			hsc_color = 'tomato'

		if tl_pix_correct:
			tl_pix = np.load(
				f'data/fluctuation_data/transfer_function/tl_clx_pix_TM{inst}_ifield{ifield_use}.npz'
			)['tl_clx_pix']
		else:
			tl_pix = np.ones_like(lb)

		trilegal_isl = None
		trilegal_lb = None
		if plot_isl_adjusted and isl_use_trilegal:
			try:
				trilegal_lb, trilegal_isl, _ = load_trilegal_unresolved_isl_cl(
					inst=inst,
					ifield=ifield_use,
					maglim_vega=isl_trilegal_maglim_vega,
					datestr=isl_trilegal_datestr,
					stat=isl_trilegal_stat,
					basepath=isl_trilegal_basepath,
				)
			except Exception:
				trilegal_lb, trilegal_isl = None, None

		panel_specs = [
			(0, ls_auto_cross[ridx], ls_plotstr, desi_color, ls_pred_fpaths),
			(1, hsc_auto_cross[ridx], hsc_plotstr, hsc_color, hsc_pred_fpaths),
		]

		for cidx, acdat, tracer_label, color, pred_fpaths in panel_specs:
			ax_use = ax[ridx, cidx]

			h_obs = ax_use.errorbar(
				lb[startidx:endidx],
				acdat.r_ell,
				yerr=acdat.r_ell_unc,
				fmt='o',
				capsize=capsize,
				markersize=markersize,
				capthick=capthick,
				color=color,
				label='Observed',
			)

			h_igl = None
			h_isl = None

			if pred_fpaths is not None:
				jmock_pred = np.load(pred_fpaths[ridx])
				lb_pred, r_ell_pred = [jmock_pred[key] for key in ['lb', 'rlx_tracer_full']]
				r_ell_pred /= tl_pix
				h_igl, = ax_use.plot(
					lb_pred,
					r_ell_pred,
					color='k',
					linestyle='dotted',
					alpha=alpha,
					label='IGL prediction',
				)

				if plot_isl_adjusted:
					r_ell_pred_isl = None
					if trilegal_isl is not None and trilegal_lb is not None:
						cii_igl = _infer_cii_from_prediction_local(jmock_pred)
						cii_isl_interp = np.interp(lb_pred, trilegal_lb, trilegal_isl)
						r_ell_pred_isl = estimate_r_ell_with_added_isl_from_cl(
							r_ell_pred,
							cii_igl,
							cii_isl_interp,
							first_order=isl_first_order,
						)

					if r_ell_pred_isl is not None:
						h_isl, = ax_use.plot(
							lb_pred,
							r_ell_pred_isl,
							color='k',
							linestyle=isl_linestyle,
							alpha=isl_alpha,
							label=isl_label,
						)

			title = f'CIBER {lams[ridx]} $\\mu$m $\\times$ {"DESI-LS" if cidx == 0 else "HSC"}'
			ax_use.text(textxpos, textypos, title, fontsize=text_fs)
			ax_use.set_xscale('log')
			ax_use.set_xlim(xlim)
			ax_use.set_ylim(ylim)
			ax_use.grid(alpha=grid_alpha)

			if legend_handles is None:
				legend_handles = [h_obs.lines[0] if hasattr(h_obs, 'lines') else h_obs]
				if h_igl is not None:
					legend_handles.append(h_igl)
				if h_isl is not None:
					legend_handles.append(h_isl)

	for cidx in range(2):
		ax[1, cidx].set_xlabel('$\\ell$', fontsize=lab_fs)

	ax[0, 0].set_title('DESI-LS ($z_{\\rm AB}<22$, $z_{\\rm phot}<1$)', fontsize=title_fs)
	ax[0, 1].set_title('HSC ($i_{\\rm AB}<25$, $z_{\\rm phot}<1$)', fontsize=title_fs)

	for ridx in range(2):
		# ax[ridx, 0].set_ylabel('$r_{\\ell}=C_{\\ell}^{Ig}/\\sqrt{C_{\\ell}^{gg}C_{\\ell}^{II}}$', fontsize=lab_fs)
		ax[ridx, 0].set_ylabel('Coherence $r_{\\ell}^{I \\times g}$', fontsize=lab_fs)

	if legend_handles is not None:
		legend_labels = ['Observed', 'IGL prediction']
		if len(legend_handles) >= 3:
			legend_labels.append(isl_label)
		fig.legend(
			handles=legend_handles,
			labels=legend_labels,
			loc='upper center',
			ncol=len(legend_labels),
			fontsize=legend_fs,
			bbox_to_anchor=(0.5, 1.02),
		)

	plt.subplots_adjust(wspace=0.06, hspace=0.10, top=0.87)

	if save_path is not None:
		fig.savefig(save_path, bbox_inches='tight')

	if show:
		plt.show()

	return fig


def compare_r_ell_hsc_ls_wise_2x3(
	figsize=(10, 5.2),
	ls_addstr='0.0_z_1.0_wrandsub_JHlt16',
	hsc_addstr='hsc_ilt24.0_zlt1_wrandsub',
	wise_addstr='unWISE_W1lt17p5_JHlt16_wFFerr',
	startidx=2,
	endidx=-1,
	title_fs=13,
	lab_fs=14,
	legend_fs=12,
	textxpos=1100,
	textypos=1.0,
	text_fs=11,
	ylim=[-0.15, 1.15],
	xlim=[250, 1.1e5],
	grid_alpha=0.3,
	capsize=3,
	capthick=1.4,
	markersize=3,
	ls_plotstr='DESI-LS ($z_{\\rm AB}<22$, $z_{\\rm phot}<1$)',
	hsc_plotstr='HSC ($i_{\\rm AB}<25$, $z_{\\rm phot}<1$)',
	wise_plotstr='unWISE ($W1<17.5$)',
	ls_pred_fpaths=None,
	hsc_pred_fpaths=None,
	wise_pred_fpaths=None,
	alpha=0.8,
	tl_pix_correct=True,
	ifield_use=8,
	plot_isl_adjusted=True,
	isl_first_order=False,
	isl_linestyle='dashed',
	isl_alpha=0.8,
	isl_label='IGL + unmasked ISL',
	isl_use_trilegal=True,
	isl_trilegal_datestr='112022',
	isl_trilegal_maglim_vega=16.0,
	isl_trilegal_stat='mean',
	isl_trilegal_basepath=None,
	save_path=None,
	show=False,
):
	"""Plot LS/HSC/WISE r_ell in a 2x3 layout.

	Rows correspond to CIBER wavelengths (1.1, 1.8 um), columns correspond to
	tracers (DESI-LS, HSC, WISE).
	"""
	lams = [1.1, 1.8]
	inst_list = [1, 2]

	ls_auto_cross = compute_rl_ciber_gal(
		ls_addstr,
		catname='LS',
		tl_pix_correct=tl_pix_correct,
		ifield_use=ifield_use,
	)
	hsc_auto_cross = compute_rl_ciber_gal(
		hsc_addstr,
		catname='HSC',
		tl_pix_correct=tl_pix_correct,
		ifield_use=ifield_use,
	)
	wise_auto_cross = compute_rl_ciber_gal(
		wise_addstr,
		catname='WISE',
		tl_pix_correct=tl_pix_correct,
		ifield_use=ifield_use,
	)

	fig, ax = plt.subplots(figsize=figsize, ncols=3, nrows=2, sharex=True, sharey=True)

	legend_handles = None

	def _infer_cii_from_prediction_local(pred):
		cross = np.asarray(pred['cross'], dtype=float)
		gal_auto = np.asarray(pred['gal_auto'], dtype=float)
		r_ell = np.asarray(pred['rlx_tracer_full'], dtype=float)
		r_denom = np.square(r_ell)
		return np.divide(
			np.square(cross),
			r_denom * gal_auto,
			out=np.zeros_like(cross, dtype=float),
			where=(r_denom > 0) & (gal_auto > 0),
		)

	for ridx, inst in enumerate(inst_list):
		lb = ls_auto_cross[ridx].lb
		if inst == 1:
			desi_color = 'b'
			hsc_color = 'dodgerblue'
			wise_color = 'royalblue'
		else:
			desi_color = 'r'
			hsc_color = 'tomato'
			wise_color = 'firebrick'

		if tl_pix_correct:
			tl_pix = np.load(
				f'data/fluctuation_data/transfer_function/tl_clx_pix_TM{inst}_ifield{ifield_use}.npz'
			)['tl_clx_pix']
		else:
			tl_pix = np.ones_like(lb)

		trilegal_isl = None
		trilegal_lb = None
		if plot_isl_adjusted and isl_use_trilegal:
			try:
				trilegal_lb, trilegal_isl, _ = load_trilegal_unresolved_isl_cl(
					inst=inst,
					ifield=ifield_use,
					maglim_vega=isl_trilegal_maglim_vega,
					datestr=isl_trilegal_datestr,
					stat=isl_trilegal_stat,
					basepath=isl_trilegal_basepath,
				)
			except Exception:
				trilegal_lb, trilegal_isl = None, None

		panel_specs = [
			(0, ls_auto_cross[ridx], 'DESI-LS', desi_color, ls_pred_fpaths),
			(1, hsc_auto_cross[ridx], 'HSC', hsc_color, hsc_pred_fpaths),
			(2, wise_auto_cross[ridx], 'WISE', wise_color, wise_pred_fpaths),
		]

		for cidx, acdat, panel_label, color, pred_fpaths in panel_specs:
			ax_use = ax[ridx, cidx]

			h_obs = ax_use.errorbar(
				lb[startidx:endidx],
				acdat.r_ell,
				yerr=acdat.r_ell_unc,
				fmt='o',
				capsize=capsize,
				markersize=markersize,
				capthick=capthick,
				color=color,
				label='Observed',
			)

			h_igl = None
			h_isl = None

			if pred_fpaths is not None:
				jmock_pred = np.load(pred_fpaths[ridx])
				lb_pred, r_ell_pred = [jmock_pred[key] for key in ['lb', 'rlx_tracer_full']]
				r_ell_pred /= tl_pix
				h_igl, = ax_use.plot(
					lb_pred,
					r_ell_pred,
					color='k',
					linestyle='dotted',
					alpha=alpha,
					label='IGL prediction',
				)

				if plot_isl_adjusted:
					r_ell_pred_isl = None
					if trilegal_isl is not None and trilegal_lb is not None:
						cii_igl = _infer_cii_from_prediction_local(jmock_pred)
						cii_isl_interp = np.interp(lb_pred, trilegal_lb, trilegal_isl)
						r_ell_pred_isl = estimate_r_ell_with_added_isl_from_cl(
							r_ell_pred,
							cii_igl,
							cii_isl_interp,
							first_order=isl_first_order,
						)

					if r_ell_pred_isl is not None:
						h_isl, = ax_use.plot(
							lb_pred,
							r_ell_pred_isl,
							color='k',
							linestyle=isl_linestyle,
							alpha=isl_alpha,
							label=isl_label,
						)

			title = f'CIBER {lams[ridx]} $\\mu$m $\\times$ {panel_label}'
			ax_use.text(textxpos, textypos, title, fontsize=text_fs)
			ax_use.set_xscale('log')
			ax_use.set_xlim(xlim)
			ax_use.set_ylim(ylim)
			ax_use.grid(alpha=grid_alpha)

			if legend_handles is None:
				legend_handles = [h_obs.lines[0] if hasattr(h_obs, 'lines') else h_obs]
				if h_igl is not None:
					legend_handles.append(h_igl)
				if h_isl is not None:
					legend_handles.append(h_isl)

	for cidx in range(3):
		ax[1, cidx].set_xlabel('$\\ell$', fontsize=lab_fs)

	ax[0, 0].set_title(ls_plotstr, fontsize=title_fs)
	ax[0, 1].set_title(hsc_plotstr, fontsize=title_fs)
	ax[0, 2].set_title(wise_plotstr, fontsize=title_fs)

	for ridx in range(2):
		ax[ridx, 0].set_ylabel('Coherence $r_{\\ell}^{I \\times g}$', fontsize=lab_fs)

	if legend_handles is not None:
		legend_labels = ['Observed', 'IGL prediction']
		if len(legend_handles) >= 3:
			legend_labels.append(isl_label)
		fig.legend(
			handles=legend_handles,
			labels=legend_labels,
			loc='upper center',
			ncol=len(legend_labels),
			fontsize=legend_fs,
			bbox_to_anchor=(0.5, 1.02),
		)

	plt.subplots_adjust(wspace=0.06, hspace=0.10, top=0.87)

	if save_path is not None:
		fig.savefig(save_path, bbox_inches='tight')

	if show:
		plt.show()

	return fig


def plot_bandpowers_vs_magcut(catname, inst, mag_lims, n_bandpowers=6, startidx=0,
							  ifield_list=[8], figsize=(7, 4),
							  capsize=3, markersize=4, alph=1.0,
							  ylabel=r"$D_{\ell}^{gg}$", xlabel=r"$m_{\rm  max} [AB]$",
							  colors=('C0', 'C1'), lab_fs=16, legend_fs=10,
							  remove_shotnoise=False, ell_min=None, xlim=[21, 27.5], markers = ['o', 'o'], \
							 text_fs=12, bbox_to_anchor=(0.7, 1.2)):
	"""
	Plot bandpowers vs magnitude cut for two rows of ell bins.

	Parameters
	----------
	remove_shotnoise : bool
		Whether to subtract shot noise estimate from low-ell bandpowers.
	ell_min : float
		Minimum ell value to use for shot noise estimate (excluding highest ell bin).
	"""

	cbps = CIBER_PS_pipeline()
	lam_dict = {1: 1.1, 2: 1.8}
	lam = lam_dict[inst]
	
	
	zorder=[10, 2]

	# Create axes without shared y so we can set per-row limits
	fig, axes = plt.subplots(2, n_bandpowers//2, figsize=figsize, sharex=True)
	plt.subplots_adjust(wspace=0.0, hspace=0.02)

	axes = axes.ravel()

	# Store y-values per row for later scaling
	row_yvals = {0: [], 1: []}

	for widx, wrandsub in enumerate([True, False]):
		color = colors[widx]
		label_prefix = "HSC auto (w/ rand. corr.)" if wrandsub else "HSC (uncorrected)"

		for m, maglim in enumerate(mag_lims):
			addstr = f"hsc_ilt{maglim}"
			if wrandsub:
				addstr += "_wrandsub"

			cgps_file = load_ciber_gal_ps(inst, catname, addstr=addstr)
			lb, all_cl_gal, all_clerr_gal, ifield_list_use = [
				cgps_file[k] for k in ['lb', 'all_cl_gal', 'all_clerr_gal', 'ifield_list_use']
			]
			pf = lb * (lb + 1) / (2 * np.pi)

			# Field average for Knox uncertainty
			fieldav_cl = np.mean(all_cl_gal, axis=0)

			# Shot noise estimate (mean at high ell above ell_min, excluding last bin)
			shotnoise = 0
			if remove_shotnoise and ell_min is not None:
				mask = (lb >= ell_min)
				mask[-1] = False  # exclude highest ell bin
				shotnoise = np.mean(fieldav_cl[mask])

			for b in range(n_bandpowers):
				yvals, yerrs = [], []

				for fieldidx, ifield in enumerate(ifield_list):
					# Knox errors
					knox_errors = np.sqrt(2. / ((2*lb+1) * cbps.Mkk_obj.delta_ell))
					fsky = 0.7 * 2 * 2 / (41253.)
					knox_errors /= np.sqrt(fsky)
					knox_errors *= np.abs(fieldav_cl)
					clerr = np.sqrt(knox_errors**2 + all_clerr_gal[fieldidx]**2)

					cl_val = all_cl_gal[fieldidx][startidx + b] - shotnoise
					yvals.append(pf[startidx + b] * cl_val)
					yerrs.append(pf[startidx + b] * clerr[startidx + b])

				# Average over fields
				ymean = np.mean(yvals)
				yerr = np.sqrt(np.sum(np.array(yerrs)**2)) / len(yerrs)

				axes[b].errorbar(maglim, ymean, yerr=yerr, fmt='o', marker=markers[widx], zorder=zorder[widx],
								 color=color, alpha=alph, markersize=markersize,
								 capsize=capsize, label=label_prefix if m == 0 else None)

				# Store y-values for per-row limits
				row_index = 0 if b < n_bandpowers//2 else 1
				row_yvals[row_index].append(ymean)
				row_yvals[row_index].append(ymean + yerr)
				row_yvals[row_index].append(ymean - yerr)

	# Formatting
	for b, ax in enumerate(axes):
		ell = int(lb[startidx + b])
		
		bpstr = str(int(cbps.Mkk_obj.binl[b]))+'$<\\ell<$'+str(int(cbps.Mkk_obj.binl[b+1]))
		ax.text(0.05, 0.05, bpstr,
				transform=ax.transAxes, zorder=20, fontsize=text_fs, va='bottom', ha='left', bbox=dict({'facecolor':'white', 'alpha':0.8}))
		ax.set_yscale('log')
		ax.grid(alpha=0.3)

		# Remove x-axis labels for top row
		if b < n_bandpowers//2:
			ax.set_xlabel("")
		else:
			ax.set_xlabel(xlabel, fontsize=lab_fs)
			
		ax.set_xlim(xlim)
		
		if b >= n_bandpowers//2:
			ax.set_xticks(mag_lims)
			ax.set_xticklabels(mag_lims, rotation=90, ha='right')


		# Hide y-tick labels except for first column in each row
		col_index = b % (n_bandpowers//2)
		if col_index != 0:
			ax.set_yticklabels([])

	# Apply y-axis labels to first subplot of each row
	axes[0].set_ylabel(ylabel, fontsize=lab_fs)
	axes[n_bandpowers//2].set_ylabel(ylabel, fontsize=lab_fs)

	# Apply per-row y-limits
	for row in [0, 1]:
		row_axes = axes[row*(n_bandpowers//2):(row+1)*(n_bandpowers//2)]
		ymin = min(v for v in row_yvals[row] if v > 0) * 0.8
		ymax = max(row_yvals[row]) * 1.2
		for ax in row_axes:
			ax.set_ylim(ymin*0.1, ymax*1.5)

	fig.tight_layout(rect=[0, 0, 1, 0.92])

	# Legend on top
	handles, labels = axes[0].get_legend_handles_labels()
	fig.legend(handles, labels, loc=2, ncol=3, fontsize=legend_fs,
			   bbox_to_anchor=bbox_to_anchor)
	
	plt.subplots_adjust(wspace=0, hspace=0)

	plt.show()
	return fig


def plot_rl_gal(all_acdat, colors=['b', 'r'], inst_list=[1, 2], figsize=(5, 3), ylim=[-0.1, 1.1], \
			   lab_fs=14, markersize=3, capsize=3, startidx=2, endidx=-1, gal_label='LS ($z<1$)', \
			   pred_fpaths=None):
	
	lams = [1.1, 1.8]
	linestyles_pred = ['dotted', 'dotted']

	n_needed = len(zbinedges) - 1
	bottom_indices = set()
	for col in range(ncols):
		col_indices = list(range(col, n_needed, ncols))
		if len(col_indices) > 0:
			bottom_indices.add(col_indices[-1])


	fig = plt.figure(figsize=figsize)
	
	for idx, inst in enumerate(inst_list):
		
		cross_lab = 'CIBER '+str(lams[idx])+' $\\mu$m $\\times$ '+gal_label
		acdat = all_acdat[idx]
		
		plt.errorbar(acdat.lb[startidx:endidx], acdat.r_ell, yerr=acdat.r_ell_unc, fmt='o', markersize=3, capsize=3, color=colors[idx], label=cross_lab)
	
		if pred_fpaths is not None:
			
			jmock_pred = np.load(pred_fpaths[idx])
			
			linestyles_pred = ['dashed', 'dashdot']
			
			lb_pred, rlx = [jmock_pred[key] for key in ['lb', 'rlx_tracer_full']]
			
			if inst==1:
				lab_pred = 'IGL prediction'
			else:
				lab_pred = None
				
			plt.plot(lb_pred, rlx, color=colors[idx], label=lab_pred, linestyle=linestyles_pred[idx])
	
	
	plt.grid(alpha=0.3)
	plt.xscale('log')
	plt.xlabel('$\\ell$', fontsize=lab_fs)
	plt.ylabel('$r_{\\ell}=C_{\\ell}^{Ig}/\\sqrt{C_{\\ell}^{gg}C_{\\ell}^{II}}$', fontsize=lab_fs)
	plt.ylim(ylim)

	plt.show()
	
	return fig


def plot_auto_cross_gal(all_acdat, inst_list=[1, 2], colors=['b', 'r'], \
						xlim=[250, 1.1e5], ylim=[1e-3, 1e4], text_fs=16, alph=0.6, \
							 bbox_to_anchor=[-0.05, 1.25], legend_fs=10, capsize=3, markersize=3, \
							   gal_label = 'LS ($z<1$)', startidx=2, endidx=-1, \
					   ylims=[[5e-4, 5e1], [1e-2, 1e2], [1, 1e4]], lab_fs=12, figsize=(4, 9), \
					   textxpos=300, textyfac=0.3, pred_fpaths=None, pred_alpha=0.6, 
					   gal_auto_lab='LS galaxy auto ($z<1$)',
					   cross_lab='CIBER $\\times$ LS ($z<1$)', 
					   ciber_auto_lab='CIBER auto'):
	
	
	bandstr_list = ['J', 'H']
	lams = [1.1, 1.8]
	
	fig, ax = plt.subplots(figsize=figsize, ncols=1, nrows=3, sharex=True)
	
	for idx, inst in enumerate(inst_list):
		
		
		
		acdat = all_acdat[idx]
		
		pf = acdat.pf
		lb = acdat.lb
				
		ax[0].errorbar(lb[acdat.posmask_auto], (pf*acdat.fieldav_cl_gal)[acdat.posmask_auto], yerr=(pf*acdat.fieldav_clerr_gal)[acdat.posmask_auto], color=colors[idx], fmt='o', \
			capsize=capsize, markersize=markersize, zorder=15, label=gal_label)
		ax[0].errorbar(lb[acdat.negmask_auto], np.abs(pf*acdat.fieldav_cl_gal)[acdat.negmask_auto], yerr=(pf*acdat.fieldav_clerr_gal)[acdat.negmask_auto], color=colors[idx], fmt='o', \
			capsize=capsize, markersize=markersize, zorder=15, mfc='white')


		ax[1].errorbar(lb[acdat.posmask], (pf*acdat.fieldav_cl_cross)[acdat.posmask], yerr=(pf*acdat.fieldav_clerr_cross)[acdat.posmask], color=colors[idx], fmt='o', \
			capsize=capsize, markersize=markersize, zorder=15, label=gal_label)
		ax[1].errorbar(lb[acdat.negmask], np.abs(pf*acdat.fieldav_cl_cross)[acdat.negmask], yerr=(pf*acdat.fieldav_clerr_cross)[acdat.negmask], color=colors[idx], fmt='o', \
			capsize=capsize, markersize=markersize, zorder=15, mfc='white')

		
		ax[2].errorbar(lb[startidx:endidx], pf[startidx:endidx]*acdat.ciber_auto_cl, yerr=pf[startidx:endidx]*acdat.ciber_auto_clerr, fmt='o', color=colors[idx], capsize=3., markersize=3, \
					  label=str(lams[idx])+' $\\mu$m')
		
		
		if pred_fpaths is not None:
			
			jmock_pred = np.load(pred_fpaths[idx])
			
			linestyles_pred = ['dashed', 'dashdot']
			
			if inst==1:
				lab_pred = 'IGL prediction'
			else:
				lab_pred = None
			
			lb_pred, gal_auto, intensity_auto, cross = [jmock_pred[key] for key in ['lb', 'gal_auto', 'intensity_auto_full', 'cross']]
			pf_pred = lb_pred*(lb_pred+1)/(2*np.pi)
#             print('lb pred:', lb_pred)
			ax[0].plot(lb_pred, pf_pred*gal_auto, color=colors[idx], linestyle=linestyles_pred[idx], alpha=pred_alpha)
			
			ax[1].plot(lb_pred, pf_pred*cross, color=colors[idx], linestyle=linestyles_pred[idx], alpha=pred_alpha)
			ax[2].plot(lb_pred, pf_pred*intensity_auto, color=colors[idx], label=lab_pred, linestyle=linestyles_pred[idx], alpha=pred_alpha)
			
		
		if idx==1:
			ax[0].set_ylabel('$D_{\\ell}^{gg}$', fontsize=lab_fs)
			ax[1].set_ylabel('$D_{\\ell}^{Ig}$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=lab_fs)
			ax[2].set_ylabel('$D_{\\ell}^{II}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=lab_fs)
			
			ax[0].text(textxpos, ylims[0][1]*textyfac, gal_auto_lab, fontsize=14)
			ax[1].text(textxpos, ylims[1][1]*textyfac, cross_lab, fontsize=14)
			ax[2].text(textxpos, ylims[2][1]*textyfac, ciber_auto_lab, fontsize=14)

			ax[2].legend(fontsize=legend_fs, loc=4)
			
	ax[2].set_xlabel('$\\ell$', fontsize=lab_fs)

	for i in range(len(ax)):
		ax[i].grid(alpha=0.3)
		ax[i].set_xscale('log')
		ax[i].set_yscale('log')
		ax[i].set_xlim(xlim)
		if ylims is not None:
			ax[i].set_ylim(ylims[i])

	plt.subplots_adjust(hspace=0.05)
	plt.show()
	
	return fig

def plot_hsc_gal_auto_vs_magcut(catname, inst, mag_lims, figsize=(5, 4), capsize=3, markersize=3, startidx=2, endidx=-1, \
						  xlim=[300, 1.05e5], legend_fs=10, ifield_list=[4, 5, 6, 7, 8], alph=0.7, \
						  ylim=[1e-4, 2e2], textstr=None, textxpos=1e4, textypos=1e1, text_fs=14, \
						  ylabel=None, include_legend=True, colors=None, plot_fieldav=True, lab_fs=16, \
							   bbox_to_anchor=[0.0, 1.2], wrandsub=True, dl_ell=False, c_ell=False, \
							   remove_shotnoise=False, ell_min=5e4):
	
	''' inst only for regridding choice '''

	cbps = CIBER_PS_pipeline()

	ciber_field_dict = dict({4:'elat10', 5:'elat30', 6:'Bootes B', 7:'Bootes A', 8:'SWIRE'})
	lam_dict = dict({1:1.1, 2:1.8})
	lam = lam_dict[inst]

	all_fieldav_cl_cross, all_fieldav_clerr_cross = [], []
	
	
	fig = plt.figure(figsize=figsize)

	for m, maglim in enumerate(mag_lims):
		addstr = 'hsc_ilt'+str(maglim)
		if wrandsub:
			addstr += '_wrandsub'

		cgps_file = load_ciber_gal_ps(inst, catname, addstr=addstr)

		lb, all_cl_gal, all_clerr_gal, ifield_list_use = [cgps_file[key] for key in ['lb', 'all_cl_gal', 'all_clerr_gal', 'ifield_list_use']]  
		print('len of all cl gal:', len(all_cl_gal))
		pf = lb*(lb+1)/(2*np.pi)
		
		if dl_ell:
			pf /= lb
		
		elif c_ell:
			pf = np.ones_like(pf)
		lbmask = (lb >= lb[startidx])*(lb < lb[endidx])
		nfield = len(ifield_list_use)

		fieldav_cl = np.mean(all_cl_gal, axis=0) # used for sample variance estimate
		
		lab = '$i < '+str(maglim)+'$'
		

		for fieldidx, ifield in enumerate(ifield_list):

			posmask = lbmask*(all_cl_gal[fieldidx] > 0)
			negmask = lbmask*(all_cl_gal[fieldidx] < 0)


			knox_errors = np.sqrt(2./((2*lb+1)*cbps.Mkk_obj.delta_ell))
			fsky = 0.7*2*2/(41253.)   
			knox_errors /= np.sqrt(fsky)
			knox_errors *= np.abs(fieldav_cl)
			clerr = np.sqrt(knox_errors**2 + all_clerr_gal[fieldidx]**2)

			if colors is not None:
				color = colors[m]
			else:
				color = 'C'+str(fieldidx)
				
			# Shot noise estimate (mean at high ell above ell_min, excluding last bin)
			shotnoise = 0
			if remove_shotnoise and ell_min is not None:
				mask = (lb >= ell_min)
				mask[-1] = False  # exclude highest ell bin
				shotnoise = np.mean(fieldav_cl[mask])
			
			all_cl_gal[fieldidx] -= shotnoise

			plt.errorbar(lb[posmask], (pf*all_cl_gal[fieldidx])[posmask], yerr=(pf*clerr)[posmask],\
				capsize=capsize, label=lab, alpha=alph, fmt='o', color=color, markersize=markersize)

			plt.errorbar(lb[negmask], np.abs(pf*all_cl_gal[fieldidx])[negmask], yerr=(pf*clerr)[negmask],\
				mfc='white', capsize=capsize, fmt='o', alpha=alph, color=color, markersize=markersize)

		if plot_fieldav:
			
			# Shot noise estimate (mean at high ell above ell_min, excluding last bin)
			shotnoise = 0
			if remove_shotnoise and ell_min is not None:
				mask = (lb >= ell_min)
				mask[-1] = False  # exclude highest ell bin
				shotnoise = np.mean(fieldav_cl[mask])
				
			fieldav_cl -= shotnoise
			
			plt.errorbar(lb[lbmask], (pf*fieldav_cl)[lbmask], yerr=(pf*np.std(all_cl_gal, axis=0))[lbmask], color='C'+str(m), markersize=3, \
						capsize=3, fmt='o')
	
	if textstr is not None:
		plt.text(textxpos, textypos, textstr, fontsize=text_fs, color='k')
	plt.xscale('log')
	plt.yscale('log')
	plt.ylim(ylim)
	plt.xlim(xlim)
	plt.xlabel('$\\ell$', fontsize=lab_fs)
	plt.tick_params(labelsize=12)

	if ylabel is None:
		ylabel = '$\\ell(\\ell+1)C_{\\ell}/2\\pi$'
	plt.ylabel(ylabel, fontsize=lab_fs)
	plt.grid(alpha=0.3)
	if include_legend:
		plt.legend(ncol=3, loc=2, bbox_to_anchor=bbox_to_anchor)
	plt.show()
	
	return fig

def plot_clIG_forecast(lb, lrange, dcl_terms_bp, dcl_vs_nbar, xerr, \
					   nbar_fid=20000, nbar_list=[1000, 5000, 20000, 100000],\
					   xlim=[250, 1e5], ylim=[1e-2, 1e0], alpha=0.3, lab_fs=14, legend_fs=9, figsize=(7, 4), \
					  colors_nbar=['b', 'g', 'r'],\
					   nbar_labs=['$1\\times 10^2$', '$5\\times 10^2$', '$1\\times 10^3$', '$5\\times 10^3$', '$2\\times 10^4$', '$1\\times 10^5$'], \
					  Adeg=20, mask_frac=0.7, suptitle=None, title_fs=12, title=None):
	
	
	if title is None:
		title = '$\\overline{n}=$'+str(nbar_fid)+' deg$^{-2}$, $A_{eff}=14$ deg$^2$'

	term_labels = ['$\\propto (C_{\\ell}^{I\\times g})^2$', '$\\propto C_{\\ell}^{I}C_{\\ell}^g$', '$\\propto N_{\\ell}^{I}C_{\\ell}^g$', \
		 '$\\propto C_{\\ell}^{I} \\overline{n}^{-1}$', '$\\propto N_{\\ell}^{I}\\overline{n}^{-1}$']

	fig, ax = plt.subplots(ncols=2, nrows=1, figsize=figsize, sharey=True)
	
	if suptitle is not None:
		plt.suptitle(suptitle, y=0.95, fontsize=14)
		
	pf_lb = lb*(lb+1)/(2*np.pi)
	pf_lrange = lrange*(lrange+1)/(2*np.pi)
	
	for x in range(2):
		
		plt.subplot(1,2,x+1)

		if x==0:
			plt.title(title, fontsize=title_fs)
			# plot individual noise components
			
#             plt.text(5e3, 2e-3, '$\\overline{n}=$'+str(nbar_fid)+' deg$^{-2}$\n$A_{eff}=$'+str(np.round(Adeg*mask_frac))+' deg$^2$', fontsize=12, \
#                     bbox=dict({'facecolor':'white', 'alpha':0.8, 'edgecolor':'k'}))
			for t in range(len(term_labels)):
				ax[x].errorbar(lb, pf_lb*dcl_terms_bp[t], xerr=xerr, capsize=3, label=term_labels[t], alpha=0.8, fmt='none', color='C'+str(t))
			
			ax[x].set_ylabel('$D_{\\ell}^{Ig}$', fontsize=lab_fs)
			ax[x].errorbar(lb, pf_lb*dcl_vs_nbar[nbar_list.index(nbar_fid)], xerr=xerr, color='k', capsize=3, label='Total', fmt='none')
			
		else:
			# plot total with varying nbar
			plt.title('Varying $\\overline{n}$', fontsize=title_fs)

			cmap = plt.get_cmap("Reds")

			# 3. Map scalar values to colors
			colors_nbar = cmap(np.linspace(0.3, 1, len(nbar_list)))
			
			for n in range(len(nbar_list)):
				nblabel = '$\\overline{n}=$'+nbar_labs[n]
				ax[x].errorbar(lb, pf_lb*dcl_vs_nbar[n], xerr=xerr, capsize=3, label=nblabel, color=colors_nbar[n], fmt='none')
			
		ax[x].set_yscale('log')
		ax[x].set_xscale('log')
		ax[x].grid(alpha=alpha)
		ax[x].set_ylim(ylim)
		ax[x].set_xlim(xlim)
		ax[x].set_xlabel('$\\ell$', fontsize=lab_fs)
		ax[x].legend(ncol=2, fontsize=legend_fs, loc=2+x)

	plt.tight_layout() 
	plt.subplots_adjust(wspace=0)
	plt.show()
	
	return fig

def collect_ciber_gal_vs_redshift(catname, subtract_randoms=False, \
								  inst_list = [1, 2], \
								zbinedges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0], \
							 maskstr=None, subtract_sn=False, ell_min_sn=5e4, \
								  ifield_list=[4, 5, 6, 7, 8], \
								  startidx=0, endidx=-1,
								  tl_pix_correct=False, with_ff_err=False,
								  headstr=None,
								  ifield_list_full=[4, 5, 6, 7, 8], fmask=0.7,
								  model_cl_gal_for_knox=None, use_inplace_auto=True, uniform_weight_ell=None):
	"""
	Collect CIBER-galaxy cross and galaxy auto power spectra vs redshift.
	
	Parameters
	----------
	catname : str
		Catalog name ('HSC', 'LS', etc.)
	subtract_randoms : bool, optional
		Whether random subtraction was applied
	inst_list : list, optional
		CIBER instrument indices [1, 2]
	zbinedges : array_like, optional
		Redshift bin edges
	maskstr : str, optional
		Mask string identifier
	subtract_sn : bool, optional
		Deprecated no-op retained for backward compatibility.
	ell_min_sn : float, optional
		Deprecated no-op retained for backward compatibility.
	ifield_list : list, optional
		List of field indices to include
	startidx : int, optional
		Starting index for processing
	endidx : int, optional
		Ending index for processing
	tl_pix_correct : bool, optional
		Whether to apply pixel transfer function correction
	with_ff_err : bool, optional
		Whether flat field errors are included
	headstr : str, optional
		Header string for file naming
	ifield_list_full : list, optional
		Full list of field indices
	fmask : float, optional
		Mask fraction per field
	model_cl_gal_for_knox : array_like, optional
		Model galaxy auto C_ell to use for Knox calculation.
		Shape: [n_inst, n_zbin, n_ell]
		If provided, Knox cosmic variance for galaxy auto-spectra is computed
		from these model values rather than from the measured spectra.
		This avoids bias from cosmic variance in the data and is recommended
		for two-stage fitting. Default is None (use data-based Knox).
	use_inplace_auto : bool, optional
		If True (default) and in-situ CIBER auto-spectrum is available in the
		cross-product file, use it for error estimation instead of the F25B file.
		If False, always use the F25B file. Default is True.
	uniform_weight_ell : float, optional
		If provided, use uniform field weighting (instead of inverse-variance) 
		above this multipole threshold. Default: None (use inverse-variance for all).

	Returns
	-------
	dict
		Dictionary containing:
		- 'lb': multipole bin centers
		- 'full_cl_gal': galaxy auto C_ell [n_inst, n_zbin, n_ell]
		- 'full_clerr_gal': galaxy auto uncertainties [n_inst, n_zbin, n_ell]
		- 'full_cl_cross': CIBER-galaxy cross C_ell [n_inst, n_zbin, n_ell]
		- 'full_clerr_cross': cross uncertainties [n_inst, n_zbin, n_ell]
		- Additional per-field and CIBER auto information
	"""

	bandstr_list = ['J', 'H']

	nbin = len(zbinedges)-1

	cbps = CIBER_PS_pipeline()

	lb = cbps.Mkk_obj.midbin_ell

	full_cl_cross, full_clerr_cross = [[np.zeros((len(zbinedges)-1, len(lb))) for x in range(2)] for y in range(2)]
	full_cl_gal, full_clerr_gal = [[np.zeros((len(zbinedges)-1, len(lb))) for x in range(2)] for y in range(2)]

	full_cl_cross_perf, full_clerr_cross_perf = [np.zeros((len(inst_list), len(zbinedges)-1, len(ifield_list), len(lb))) for y in range(2)]

	full_cl_ciber_auto, full_clerr_ciber_auto = [np.zeros((2, len(lb))) for x in range(2)]

	full_perf_weights = np.zeros((len(inst_list), len(zbinedges)-1, len(ifield_list), len(lb)))

	
	for n in range(nbin):

		z0, z1 = zbinedges[n], zbinedges[n+1]

		# addstr = ''

		# if addstr_use is not None:
			# addstr += addstr_use+'_'

		addstr = str(np.round(z0, 1))+'_z_'+str(np.round(z1, 1))

		if subtract_randoms:
			addstr_use = addstr+'_wrandsub'
		else:
			addstr_use = addstr

		if with_ff_err:
			addstr_use += '_wFFerr'

		if maskstr is not None:
			addstr_use += '_'+maskstr


		if headstr is not None:
			addstr_use = headstr + '_' + addstr_use

#         print('addstr:', addstr_use)
		for idx, inst in enumerate(inst_list):
			cgps_file = load_ciber_gal_ps(inst, catname, addstr=addstr_use)
			lb, all_cl_gal, all_clerr_gal, all_cl_cross,\
				all_clerr_cross, ifield_list_use = [cgps_file[key] for key in ['lb', 'all_cl_gal', 'all_clerr_gal', 'all_cl_cross', 'all_clerr_cross', 'ifield_list_use']]

			all_clerr_cross /= fmask # mask NOT accounted for in MC variance

			# Check for in-situ CIBER auto-spectrum
			has_inplace_auto = ('all_cl_ciber_auto_inplace' in cgps_file.files
							   and use_inplace_auto)

			if len(ifield_list_use)==1:

				fieldav_cl_gal, fieldav_clerr_gal = all_cl_gal[0], all_clerr_gal[0]
				fieldav_cl_cross, fieldav_clerr_cross = all_cl_cross[0], all_clerr_cross[0]

			else:

				pf, posmask_auto, negmask_auto, fieldav_cl_gal, fieldav_clerr_gal = mini_proc_clav(
					all_cl_gal, all_clerr_gal, lb, startidx, endidx, mode='auto', uniform_weight_ell=uniform_weight_ell
				)
				pf, posmask, negmask, fieldav_cl_cross, fieldav_clerr_cross = mini_proc_clav(
					all_cl_cross, all_clerr_cross, lb, startidx, endidx, mode='cross', uniform_weight_ell=uniform_weight_ell
				)

			# Load CIBER auto for uncertainty estimation
			if has_inplace_auto:
				all_cl_ciber_auto_inplace = cgps_file['all_cl_ciber_auto_inplace']
				cl_auto_inplace_fieldav = np.nanmean(all_cl_ciber_auto_inplace, axis=0)
				cl_auto_use = cl_auto_inplace_fieldav   # already on analysis lb grid
				lb_auto = lb  # in-situ auto is on same grid as lb
				if n == 0:
					print(
						f"[CIBER auto load] TM{inst} using in-situ auto-spectrum from cross-product file "
						f"(signal-only, noise-subtracted, per-field shapes: {all_cl_ciber_auto_inplace.shape}) "
						f"lb=[{lb[0]:.1f},{lb[-1]:.1f}] n={len(lb)}"
					)
					print(
						f"[CIBER auto shown] TM{inst} cl range=[{np.nanmin(cl_auto_use):.3e}, {np.nanmax(cl_auto_use):.3e}]"
					)
			else:
				ciber_auto = _load_ciber_auto_file(bandstr_list[idx])
				lb_auto, cl_auto, clerr_auto = [ciber_auto[key] for key in ['lb', 'fieldav_cl', 'fieldav_clerr']]

				# Align CIBER auto to analysis ell grid using interpolation.
				# This avoids zero-padding artifacts when file/binning lengths differ.
				cl_auto_use = np.interp(lb, lb_auto, cl_auto, left=cl_auto[0], right=cl_auto[-1])
				clerr_auto_use = np.interp(lb, lb_auto, clerr_auto, left=clerr_auto[0], right=clerr_auto[-1])

				if n == 0:
					print(
						f"[CIBER auto load] TM{inst} source={ciber_auto.get('source_path', 'unknown')} "
						f"mode={ciber_auto.get('source_mode', 'unknown')} "
						f"lb_file=[{lb_auto[0]:.1f},{lb_auto[-1]:.1f}] n={len(lb_auto)} "
						f"lb_use=[{lb[0]:.1f},{lb[-1]:.1f}] n={len(lb)}"
					)
					print(
						f"[CIBER auto shown] TM{inst} cl range=[{np.nanmin(cl_auto_use):.3e}, {np.nanmax(cl_auto_use):.3e}] "
						f"clerr range=[{np.nanmin(clerr_auto_use):.3e}, {np.nanmax(clerr_auto_use):.3e}]"
					)



			full_cl_ciber_auto[idx] = cl_auto_use
			full_clerr_ciber_auto[idx] = np.zeros_like(cl_auto_use) if has_inplace_auto else clerr_auto_use

			# Per-field uncertainties
			perf_clerr_cross = np.zeros((len(ifield_list_use), fieldav_cl_cross.shape[0]))

			mean_norms = [cbps.zl_levels_ciber_fields[inst][cbps.ciber_field_dict[ifield]] 
						for ifield in ifield_list_full]
			# Compute flat field weights
			weights_ff = cbps.compute_ff_weights(inst, mean_norms, ifield_list_full, photon_noise=True)
			# Compute flat field bias correction for each field
			# This returns the multiplicative correction factor (1 + bias)
			ff_bias_factors = compute_ff_bias(mean_norms, weights=weights_ff)

			# print('perf clerr cross has shape', perf_clerr_cross)
			for fieldidx, ifield in enumerate(ifield_list):

				idx_full = ifield_list_full.index(ifield)

				if has_inplace_auto:
					cl_auto_field = all_cl_ciber_auto_inplace[fieldidx]
				else:
					cl_auto_field = cl_auto_use*ff_bias_factors[idx_full]

				perf_clerr_cross[fieldidx] = estimate_cross_uncertainties(
					lb, fieldav_cl_cross, all_clerr_cross[fieldidx],
					cl_auto_field, fieldav_cl_gal, 1, startidx=2, endidx=-1
				)

			full_clerr_cross_perf[idx, n] = perf_clerr_cross
			full_cl_cross_perf[idx, n] = all_cl_cross


			# === RECOMPUTE FIELD AVERAGE USING PROPER WEIGHTS ===

			if len(ifield_list_use)==1:
				# fieldav_cl, fieldav_clerr = fieldav_cl_cross, perf_clerr_cross[0]
				fieldav_clerr = perf_clerr_cross[0]

				perf_weights = np.ones_like(fieldav_cl_cross)

			else:
				print('Recomputing field-averaged cross-spectrum with proper weights...')
				fieldav_cl, fieldav_clerr,\
					_, perf_weights = compute_field_averaged_power_spectrum(all_cl_cross.copy(), per_field_dcls=perf_clerr_cross.copy())
				# print('perf weights in collect_ciber_gal_vs_redshift:', perf_weights)

			if tl_pix_correct:

				ifield_use = 6
				tl_pix = np.load('data/fluctuation_data/transfer_function/tl_clx_pix_TM'+str(inst)+'_ifield'+str(ifield_use)+'.npz')['tl_clx_pix']
			

			else:
				tl_pix = np.ones_like(fieldav_cl_cross)

			fieldav_cl_cross /= tl_pix 
			fieldav_clerr /= tl_pix

			# Add Knox errors to galaxy auto-spectrum
			nfield = len(ifield_list_use)
			gal_knox_errors = np.sqrt(2./((2*lb+1)*cbps.Mkk_obj.delta_ell))
			fsky = fmask*2*2/(41253.) * nfield  # Scale by number of fields
			gal_knox_errors /= np.sqrt(fsky)
			
			# Use model for Knox if provided, otherwise use measured data
			if model_cl_gal_for_knox is not None:
				gal_knox_errors *= np.abs(model_cl_gal_for_knox[idx, n])
			else:
				gal_knox_errors *= np.abs(fieldav_cl_gal)
				
			fieldav_clerr_gal = np.sqrt(gal_knox_errors**2 + fieldav_clerr_gal**2)

			# print('snr:', fieldav_cl_gal/fieldav_clerr_gal)

			# full_cl_cross[idx][n] = fieldav_cl
			# full_clerr_cross[idx][n] = fieldav_clerr # heeee

			full_cl_cross[idx][n] = fieldav_cl_cross
			full_clerr_cross[idx][n] = fieldav_clerr # heeee

			full_cl_gal[idx][n] = fieldav_cl_gal
			full_clerr_gal[idx][n] = fieldav_clerr_gal

			full_perf_weights[idx][n] = perf_weights



	res = dict({'lb':lb, 
				'full_cl_cross':np.array(full_cl_cross),
				'full_clerr_cross':np.array(full_clerr_cross),
				'full_cl_gal':np.array(full_cl_gal),
				'full_clerr_gal':np.array(full_clerr_gal),
				'full_cl_cross_perf':np.array(full_cl_cross_perf),
				'full_clerr_cross_perf':np.array(full_clerr_cross_perf),
				'full_cl_ciber_auto':np.array(full_cl_ciber_auto),
				'full_clerr_ciber_auto':np.array(full_clerr_ciber_auto),
				'lb_auto':lb_auto,
				'full_perf_weights':np.array(full_perf_weights),
				})


	return res


			
	# return lb, np.array(full_cl_cross), np.array(full_clerr_cross), np.array(full_cl_gal), np.array(full_clerr_gal)


def compute_galdens(catname, ifield_list=[4, 5, 6, 7, 8], Adeg=4., \
				   zbinedges=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0], \
				   masks=None, save=True, hsc_glim=24):

	nzbin = len(zbinedges)-1
	
	ngal_perz_perfield = np.zeros((nzbin, len(ifield_list)))
	
	galdens_basedir = config.ciber_basepath+'data/fluctuation_data/TM1/gal_density/'+catname+'/'

	# load galaxy counts and (optionally) apply mask before computing density in unmasked region
	
	for zidx in range(nzbin):
	
		for fieldidx, ifield in enumerate(ifield_list):
		 
			if catname=='HSC':
				galdens_fpath = galdens_basedir+'gal_density_'+catname+'_TM1_hsc_glt'+str(hsc_glim)+'_'+str(zbinedges[zidx])+'_z_'+str(zbinedges[zidx+1])+'.fits'
				
			else:
				galdens_fpath = galdens_basedir+'gal_density_'+catname+'_TM1_'+str(zbinedges[zidx])+'_z_'+str(zbinedges[zidx+1])+'.fits'
			
			galcounts = fits.open(galdens_fpath)['ifield'+str(ifield)].data
			
			ngal_perz_perfield[zidx, fieldidx] = np.sum(galcounts)

		print('total gal counts for zidx', zidx, ngal_perz_perfield[zidx])
	
	galdens_perz = ngal_perz_perfield / Adeg
	
	print('Galaxy densities [deg-2]:', galdens_perz)
	
	save_fpath = galdens_basedir+'gal_density_vs_redshift_'+catname+'.npz'
	
	if save:
		
		np.savez(save_fpath, zbinedges=zbinedges, galdens_perz=galdens_perz, ifield_list=ifield_list)

	return galdens_perz, save_fpath

def plot_photoz_dist(catnames, colors=['k', 'C4'], include_fieldav=True, \
					zbinedges=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0], \
					figsize=(5, 4), ylim=[1e2, 1e4], perfield_alpha=0.3, labels=None, \
					legend_fs=12):
	
	''' Plot galaxy density in redshift bins'''
	
	zbinedges = np.array(zbinedges)
	zcen = [0.5*(zbinedges[i]+zbinedges[i+1]) for i in range(len(zbinedges)-1)]
	xerr = [zcen-zbinedges[:-1], zbinedges[1:]-zcen]
	
	print('zcen:', zcen)
	print('xerr:', xerr)
	
	fig = plt.figure(figsize=figsize)
	
	for c, catname in enumerate(catnames):
		
		galdens_basedir = config.ciber_basepath+'data/fluctuation_data/TM1/gal_density/'+catname+'/'
		save_fpath = galdens_basedir+'gal_density_vs_redshift_'+catname+'.npz'

		galdens_perz = np.load(save_fpath)['galdens_perz']
		ifield_list = np.load(save_fpath)['ifield_list']
		print('for catname', catname, 'ifield list is ', ifield_list)
		
		if len(ifield_list)==1:
			print('galdens perz has shape', galdens_perz.shape)
			galdens_perz_use = galdens_perz[:,0]
			yerr = np.zeros_like(galdens_perz_use)
			print('one field:', galdens_perz_use.shape, yerr.shape, len(zcen))
		
		else:
			print('galdens perz has shape', galdens_perz.shape)
			galdens_perz_use = np.mean(galdens_perz, axis=1)
			yerr = np.array([galdens_perz_use-np.min(galdens_perz, axis=1), np.max(galdens_perz, axis=1)-galdens_perz_use])
			
#             print('multiple field:', galdens_perz_use.shape, yerr.shape, zcen.shape)
		if labels is not None:
			label = labels[c] 
		else:
			label = catname

		plt.errorbar(zcen, galdens_perz_use, xerr=xerr, yerr=yerr, capsize=3, fmt='o', color=colors[c], label=label)
		
	plt.yscale('log')
	plt.xticks(zbinedges)
	plt.grid(alpha=0.3)
	plt.ylim(ylim)
	plt.legend(loc=4, fontsize=legend_fs)
	plt.xlabel('Redshift $z$', fontsize=14)
	plt.ylabel('$\\overline{n}_g$ [deg$^{-2}$]', fontsize=14)
	plt.show()
	
	return fig


def plot_gal_ps_vs_redshift(inst, zbinedges, catname='LS', figsize=(5, 4), startidx=0, endidx=-1, \
						   xlim=[150, 1.1e5], ylim=[1e-4, 2e2], colors=['b', 'r'], \
							 textstr=None, textxpos=200, textypos=5e1, text_fs=16, alph=0.6, \
							 bbox_to_anchor=[-0.05, 1.25], legend_fs=10, capsize=3, markersize=3, \
							addstrs=None, headstr=None, subtract_randoms=True, maskstr=None):
	
	cbps = CIBER_PS_pipeline()
	
	fig = plt.figure(figsize=figsize)
	
	colors = plt.cm.jet(np.linspace(0, 1,len(zbinedges)-1))
	plt.title(catname+' overdensity power spectrum')
				
	for zidx, z0 in enumerate(zbinedges[:-1]):
				
		z1 = zbinedges[zidx+1]
		
		if addstrs is not None:
			addstr = addstrs[zidx]
		else:
			addstr = str(np.round(z0, 1))+'_z_'+str(np.round(z1, 1))

		if subtract_randoms:
			addstr +='_wrandsub'

		if maskstr is not None:
			addstr += '_'+maskstr


		if headstr is not None:
			addstr = headstr +'_'+addstr
			# addstr = 'hsc_zlt22_'+str(np.round(z0, 1))+'_z_'+str(np.round(z1, 1))
		
		cgps_file = load_ciber_gal_ps(inst, catname, addstr=addstr)
		all_cl_gal = cgps_file['all_cl_gal']
		all_clerr_gal = cgps_file['all_clerr_gal']
		
		if zidx==0:
			lb = cgps_file['lb']
			pf = lb*(lb+1)/(2*np.pi)
			lbmask = (lb >= lb[startidx])*(lb < lb[endidx])

		if len(all_cl_gal) > 1:
			fieldav_cl_gal, fieldav_clerr_gal,\
				_, _ = compute_field_averaged_power_spectrum(all_cl_gal.copy(), per_field_dcls=all_clerr_gal.copy())

		else:
			fieldav_cl_gal, fieldav_clerr_gal = all_cl_gal[0], all_clerr_gal[0]

		gal_knox_errors = np.sqrt(2./((2*lb+1)*cbps.Mkk_obj.delta_ell))
		fsky = 2*2/(41253.)    
		gal_knox_errors /= np.sqrt(fsky)
		gal_knox_errors *= np.abs(fieldav_cl_gal)
		fieldav_clerr_gal = np.sqrt(gal_knox_errors**2 + fieldav_clerr_gal**2)

		posmask = lbmask*(fieldav_cl_gal > 0)
		negmask = lbmask*(fieldav_cl_gal < 0)
		
		gal_label = str(np.round(z0, 1))+'$<z_{\\rm phot}<$'+str(np.round(z1, 1))


		plt.errorbar(lb[posmask], (pf*fieldav_cl_gal)[posmask], yerr=(pf*fieldav_clerr_gal)[posmask], color=colors[zidx], fmt='o', \
			capsize=capsize, markersize=markersize, zorder=15, label=gal_label)
		plt.errorbar(lb[negmask], np.abs(pf*fieldav_cl_gal)[negmask], yerr=(pf*fieldav_clerr_gal)[negmask], color=colors[zidx], fmt='o', \
			capsize=capsize, markersize=markersize, zorder=15, mfc='white')
	
		plt.legend(loc=4, ncol=2, fontsize=8)
		plt.xlabel('$\\ell$', fontsize=12)
		plt.ylabel('$D_{\\ell}^{gg}$', fontsize=12)
		plt.grid(alpha=0.3)
		plt.xscale('log')
		plt.yscale('log')
		plt.xlim(xlim)
		plt.ylim(ylim)
	plt.show()
	
	return fig



def plot_cross_ps_vs_redshift(inst, zbinedges, lb, all_fieldav_cl_cross, all_fieldav_clerr_cross, catname='LS', figsize=(5, 4), startidx=2, endidx=-1, \
							 xlim=[150, 1.1e5], ylim=[5e-3, 1e3], legend_fs=16, capsize=3, markersize=3, alph=0.8, \
							 textxpos=280, textypos=1e2, text_fs=12, color=None, color_inst=['b', 'r'], bbox_to_anchor=[2.0, 1.4], \
							 ncols=4, nrows=2, all_pred_fpaths=None, pred_alpha=0.5, \
							 ncol_legend=3, tl_pix_correct=False, rescale_gal_auto_bias=False, bias_model='1+z',
							 bias_cache_fpath=None, bias_cache_scheme='fine', include_1h_pred=True, onehalo_output_dir=None, onehalo_fsat_model='single'):
	
	lam_dict = dict({1:1.1, 2:1.8})
	
	pf = lb*(lb+1)/(2*np.pi)
	
	lbmask = (lb >= lb[startidx])*(lb < lb[endidx])

	fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)
	ax = ax.ravel()

	if color is not None:
		colors = [color for x in range(len(zbinedges[:-1]))]
	else:
		colors = plt.cm.jet(np.linspace(0, 1,len(zbinedges)-1))
		
		
	if type(inst)!= list:
		inst = [inst]
		
	linestyles_pred = ['solid', 'solid']
	n_needed = len(zbinedges) - 1
	bottom_indices = set()
	for col in range(ncols):
		col_indices = list(range(col, n_needed, ncols))
		if len(col_indices) > 0:
			bottom_indices.add(col_indices[-1])

	bias_cache = np.load(bias_cache_fpath, allow_pickle=False) if bias_cache_fpath is not None else None

	for zidx, z0 in enumerate(zbinedges[:-1]):

		for i, inst_indiv in enumerate(inst):

			if include_1h_pred:
				oh_data_Ig = load_onehalo_spectrum(
						onehalo_output_dir, onehalo_fsat_model, 'sdss_z',
						inst=inst_indiv, mag_min=18.0, mag_cut=22.0, z0=0.05, mode='Ig', generate_type='superfine')
				ell_1h = oh_data_Ig['ell_arr']
				dl_1h = oh_data_Ig['dl_spectrum'][zidx]

				

			if tl_pix_correct:
				ifield_use = 6
				tl_pix = np.load('data/fluctuation_data/transfer_function/tl_clx_pix_TM'+str(inst_indiv)+'_ifield'+str(ifield_use)+'.npz')['tl_clx_pix']

			if all_pred_fpaths is not None:

				z_center = 0.5 * (zbinedges[zidx] + zbinedges[zidx + 1])

				if bias_cache is not None:
					# Smooth shot-noise + two-halo fit, rescaled by measured b_g
					b_g = _load_bias_for_z(bias_cache, z_center, scheme=bias_cache_scheme)
					ell_eval = np.geomspace(xlim[0], xlim[1], 300)
					ell_smooth, dl_smooth = smooth_mock_cross_with_bias(
						all_pred_fpaths[i][zidx], z_center, b_g, ell_eval=ell_eval)
					# print('dl smooth for zidx', zidx, 'is ', dl_smooth)
					if tl_pix_correct:
						tl_interp = np.interp(ell_smooth, np.arange(len(tl_pix)), tl_pix)
						dl_smooth = dl_smooth / tl_interp
					if inst_indiv == 1:
						lab_pred = 'IGL prediction'
					else:
						lab_pred = None

					if include_1h_pred and dl_1h is not None:

						if np.isnan(dl_1h).any():
							print('Warning: NaN values found in 1-halo prediction for zidx', zidx)
							dl_1h = np.nan_to_num(dl_1h, nan=0.0)
						dl_1h_interp = np.interp(ell_smooth, ell_1h, dl_1h)

						if zidx == 0:
							dl_1h_interp *= 0.1

						dl_smooth += dl_1h_interp

					print('here dl smooth for zidx', zidx, 'is ', dl_smooth)
					ax[zidx].plot(ell_smooth, dl_smooth, color=color_inst[i],
					              linestyle=linestyles_pred[i], alpha=pred_alpha, label=lab_pred, linewidth=2)
				else:
					# Fall back to raw (noisy) mock curve, with optional 2h rescaling
					jmock_pred = np.load(all_pred_fpaths[i][zidx])
					if rescale_gal_auto_bias:
						cross, _ = rescale_spectrum_2halo_bias(
							jmock_pred['lb'], jmock_pred['cross'], z_center,
							bias_model=bias_model, bias_power=1, verbose=False)
					else:
						cross = jmock_pred['cross']
					lb_pred = jmock_pred['lb']
					pf_pred = lb_pred * (lb_pred + 1) / (2 * np.pi)
					if tl_pix_correct:
						cross /= tl_pix
					if inst_indiv == 1:
						lab_pred = 'IGL prediction'
					else:
						lab_pred = None
					ax[zidx].plot(lb_pred, pf_pred * cross, color=color_inst[i],
					              linestyle=linestyles_pred[i], alpha=pred_alpha, label=lab_pred, linewidth=2)

		
			label = 'CIBER '+str(lam_dict[inst_indiv])+' $\\mu$m $\\times$ '+catname
		
			fieldav_cl_cross = all_fieldav_cl_cross[i][zidx]
			fieldav_clerr_cross = all_fieldav_clerr_cross[i][zidx]

			z1 = zbinedges[zidx+1]
			addstr = str(np.round(z0, 1))+'_z_'+str(np.round(z1, 1))

			posmask = lbmask*(fieldav_cl_cross > 0)
			negmask = lbmask*(fieldav_cl_cross < 0)

			gal_label = str(np.round(z0, 1))+'$<z_{\\rm phot}<$'+str(np.round(z1, 1))

			ax[zidx].errorbar(lb[posmask], (pf*fieldav_cl_cross)[posmask], yerr=(pf*fieldav_clerr_cross)[posmask], color=color_inst[i], fmt='o', \
				capsize=capsize, markersize=markersize, zorder=15, label=label, alpha=alph)
			ax[zidx].errorbar(lb[negmask], np.abs(pf*fieldav_cl_cross)[negmask], yerr=(pf*fieldav_clerr_cross)[negmask], color=color_inst[i], fmt='o', \
				capsize=capsize, markersize=markersize, zorder=15, mfc='white', alpha=alph)

		if zidx==0:
			# skip per-axis legend; we'll add a centered figure legend below
			pass
		ax[zidx].text(textxpos, textypos, gal_label, fontsize=text_fs)
		
		if zidx in bottom_indices:
			ax[zidx].set_xlabel('$\\ell$', fontsize=12)
		ax[zidx].set_ylim(ylim)
		ax[zidx].set_xlim(xlim)
		# Keep each panel square regardless of figure dimensions.
		ax[zidx].set_box_aspect(1)
		
		if (zidx % ncols) == 0:
			ax[zidx].set_ylabel('$D_{\\ell}^{Ig}$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=12)
		ax[zidx].grid(alpha=0.3)
		ax[zidx].set_xscale('log')
		ax[zidx].set_yscale('log')

	# Remove extra axes when grid has more slots than z bins.
	n_panels = nrows * ncols
	if n_needed < n_panels:
		for idx in range(n_needed, n_panels):
			fig.delaxes(ax[idx])
	# Place a single centered legend at the top of the figure (centered horizontally)
	handles, labels = ax[0].get_legend_handles_labels()
	if len(handles) > 0:
		# Put legend inside the canvas near the top row to avoid extra whitespace.
		fig.legend(handles, labels, loc='upper center', ncol=ncol_legend, fontsize=legend_fs, bbox_to_anchor=(0.5, 0.965))

	# add a bit more vertical space between rows to avoid ticklabel overlap
	plt.subplots_adjust(wspace=0, hspace=0.10, top=0.89)
	plt.show()
	
	return fig



def plot_fieldav_ciber_gal_ps(inst_list, catname, addstr=None, labels=None, \
							 figsize=(5, 4), capsize=3, markersize=3, plot_perfield=False, plot_perfield_unc=True, \
							 startidx=0, endidx=-1, xlim=[150, 1.1e5], ylim=[1e-4, 2e2], colors=['b', 'r'], \
							 textstr=None, textxpos=200, textypos=5e1, text_fs=16, alph=0.6, \
							 bbox_to_anchor=[-0.05, 1.25], legend_fs=10, mask_frac=0.7, \
							  ifield_list=[4, 5, 6, 7, 8], lab_fs=14, pred_fpaths=None, pred_alpha = 0.9, 
							  with_cross_shot=False, tl_pix_correct=True, ifield_use=8, 
							  plot_unc=False, ylabel='$D_{\\ell}^{Ig}$ [nW m$^{-2}$ sr$^{-1}$]'):
	
	ciber_field_dict = dict({4:'elat10', 5:'elat30', 6:'Bootes B', 7:'Bootes A', 8:'SWIRE'})
	lam_dict = dict({1:1.1, 2:1.8})
	bandstr_list = ['J', 'H']
	all_fieldav_cl_cross, all_fieldav_clerr_cross = [], []

	cbps = CIBER_PS_pipeline()
	fig = plt.figure(figsize=figsize)

	
	# plot CIBER x galaxy PS
	for idx, inst in enumerate(inst_list):
		
		lam = lam_dict[inst]
		cgps_file = load_ciber_gal_ps(inst, catname, addstr=addstr)

		lb, all_cl_gal, all_clerr_gal, all_cl_cross,\
			all_clerr_cross, ifield_list_use = [cgps_file[key] for key in ['lb', 'all_cl_gal', 'all_clerr_gal', 'all_cl_cross', 'all_clerr_cross', 'ifield_list_use']]
		
		# load CIBER auto 
		ciber_auto = _load_ciber_auto_file(bandstr_list[idx])

		lb_auto, cl_auto, clerr_auto = [ciber_auto[key] for key in ['lb', 'fieldav_cl', 'fieldav_clerr']]


		print('ifield list use:', ifield_list_use)
				
		pf = lb*(lb+1)/(2*np.pi)
		lbmask = (lb >= lb[startidx])*(lb < lb[endidx])
		nfield = len(ifield_list_use)
		
		if nfield > 1:
			
			cl_weights = 1./all_clerr_cross**2
			fieldav_cl_cross, fieldav_clerr_cross = compute_weighted_cl(all_cl_cross.copy(), cl_weights)
			fieldav_cl_gal = np.mean(all_cl_gal, axis=0)

		else:
			fieldav_cl_cross = all_cl_cross[0]
			fieldav_clerr_cross = all_clerr_cross[0]
			fieldav_cl_gal = all_cl_gal[0]

		cross_knox_errors = np.sqrt(1./((2*lb+1)*cbps.Mkk_obj.delta_ell)) # factor of 1 in numerator since auto is a cross-epoch cross
		fsky = mask_frac*nfield*2*2/41253.   
		cross_knox_errors /= np.sqrt(fsky)
		cross_knox_errors *= np.abs(fieldav_cl_cross)
		fieldav_clerr_cross = np.sqrt(cross_knox_errors**2 + fieldav_clerr_cross**2)    

		fieldav_clerr_cross = estimate_cross_uncertainties(lb,
													   fieldav_cl_cross,
													   fieldav_clerr_cross,
													   cl_auto, fieldav_cl_gal, nfield, \
													  startidx=2, endidx=-1)
		


		if tl_pix_correct:
			tl_pix = np.load('data/fluctuation_data/transfer_function/tl_clx_pix_TM'+str(inst)+'_ifield'+str(ifield_use)+'.npz')['tl_clx_pix']
			fieldav_cl_cross /= tl_pix 
			fieldav_clerr_cross /= tl_pix


		all_fieldav_cl_cross.append(fieldav_cl_cross)
		all_fieldav_clerr_cross.append(fieldav_clerr_cross)
		
		
		for fieldidx, ifield in enumerate(ifield_list_use):
			# cross_knox_indiv = cross_knox_errors*np.sqrt(nfield)
			# all_clerr_cross[fieldidx] = np.sqrt(all_clerr_cross[fieldidx]**2 + cross_knox_indiv**2)
			all_clerr_cross[fieldidx] = fieldav_clerr_cross*np.sqrt(nfield)

		
		if plot_perfield:

			for fieldidx, ifield in enumerate(ifield_list_use):
#                 fieldidx = ifield-4
				
				posmask = lbmask*(all_cl_cross[fieldidx] > 0)
				negmask = lbmask*(all_cl_cross[fieldidx] < 0)
								
				plt.errorbar(lb[posmask], (pf*all_cl_cross[fieldidx])[posmask], yerr=(pf*all_clerr_cross[fieldidx])[posmask],\
					capsize=capsize, label=ciber_field_dict[ifield], alpha=alph, fmt='o', color='C'+str(ifield-4), markersize=markersize)
				
				plt.errorbar(lb[negmask], np.abs(pf*all_cl_cross[fieldidx])[negmask], yerr=(pf*all_clerr_cross[fieldidx])[negmask],\
					mfc='white', capsize=capsize, fmt='o', alpha=alph, color='C'+str(ifield-4), markersize=markersize)
		

				if plot_perfield_unc:
					plt.plot(lb, pf*all_clerr_cross[fieldidx], linestyle='dashdot', color='C'+str(ifield-4))

		posmask = lbmask*(fieldav_cl_cross > 0)
		negmask = lbmask*(fieldav_cl_cross < 0)
		
		if labels is None:
			label = 'Field average'
		else:
			label = labels[idx]
		
		if plot_perfield:
			plot_color = 'k'
		else:
			plot_color = colors[idx]
			
		plt.errorbar(lb[posmask], (pf*fieldav_cl_cross)[posmask], yerr=(pf*fieldav_clerr_cross)[posmask], fmt='o', \
					capsize=capsize, markersize=markersize, zorder=10, label=label, color=plot_color)
		plt.errorbar(lb[negmask], np.abs(pf*fieldav_cl_cross)[negmask], yerr=(pf*fieldav_clerr_cross)[negmask], fmt='o', \
			capsize=capsize, markersize=markersize, mfc='white', zorder=10, color=plot_color)


		if plot_unc:
			unc_colors = ['grey', 'k']
			plt.plot(lb[lbmask], (pf*fieldav_clerr_cross)[lbmask], color=unc_colors[idx], linestyle='dashed')

	   
		if pred_fpaths is not None:
			
			jmock_pred = np.load(pred_fpaths[idx])
			
			linestyles_pred = ['dashed', 'dashdot']
			
			if inst==1:
				lab_pred = 'IGL prediction'
				cross_sn_pred = 'Cross-shot noise level'
			else:
				lab_pred, cross_sn_pred = None, None

			lb_pred, gal_auto, intensity_auto, cross = [jmock_pred[key] for key in ['lb', 'gal_auto', 'intensity_auto_full', 'cross']]
			pf_pred = lb_pred*(lb_pred+1)/(2*np.pi)

			if tl_pix_correct:

				cross /= tl_pix


			plt.plot(lb_pred, pf_pred*cross, color=plot_color, linestyle=linestyles_pred[idx], alpha=pred_alpha, label=lab_pred)

			if with_cross_shot:
				plt.plot(lb_pred, pf_pred*jmock_pred['cross_poisson'], color='k', linestyle='dashed', label=cross_sn_pred)


	if textstr is not None:
		plt.text(textxpos, textypos, textstr, fontsize=text_fs)
		
	plt.xscale('log')
	plt.yscale('log')

	plt.xlabel('$\\ell$', fontsize=lab_fs)
	plt.ylabel(ylabel, fontsize=lab_fs)
	plt.grid(alpha=0.5)
	plt.tick_params(labelsize=12)
	plt.legend(ncol=2, loc=2, bbox_to_anchor=bbox_to_anchor, fontsize=legend_fs)
		
	plt.xlim(xlim)
	plt.ylim(ylim)
	plt.show()
	
	return fig, lb, all_fieldav_cl_cross, all_fieldav_clerr_cross


def plot_perfield_gal_auto(catname, inst, addstr=None, figsize=(5, 4), capsize=3, markersize=3, startidx=2, endidx=-1, \
						  xlim=[300, 1.05e5], legend_fs=10, ifield_list=[4, 5, 6, 7, 8], alph=0.7, \
						  ylim=[1e-4, 2e2], textstr=None, textxpos=1e4, textypos=1e1, text_fs=14, \
						  ylabel=None, include_legend=True, colors=None, plot_fieldav=True, lab_fs=16, 
						  pred_fpaths=None, legend_loc=4):
	
	''' inst only for regridding choice '''

	cbps = CIBER_PS_pipeline()

	ciber_field_dict = dict({4:'elat10', 5:'elat30', 6:'Bootes B', 7:'Bootes A', 8:'SWIRE'})
	lam_dict = dict({1:1.1, 2:1.8})
	lam = lam_dict[inst]

	all_fieldav_cl_cross, all_fieldav_clerr_cross = [], []
	
	cgps_file = load_ciber_gal_ps(inst, catname, addstr=addstr)
	
	lb, all_cl_gal, all_clerr_gal, ifield_list_use = [cgps_file[key] for key in ['lb', 'all_cl_gal', 'all_clerr_gal', 'ifield_list_use']]  
	print('len of all cl gal:', len(all_cl_gal))
	pf = lb*(lb+1)/(2*np.pi)
	lbmask = (lb >= lb[startidx])*(lb < lb[endidx])
	nfield = len(ifield_list_use)
	
	fieldav_cl = np.mean(all_cl_gal, axis=0) # used for sample variance estimate
	
	fig = plt.figure(figsize=figsize)

	for fieldidx, ifield in enumerate(ifield_list):
		
		posmask = lbmask*(all_cl_gal[fieldidx] > 0)
		negmask = lbmask*(all_cl_gal[fieldidx] < 0)
		
		
		knox_errors = np.sqrt(2./((2*lb+1)*cbps.Mkk_obj.delta_ell))
		fsky = 0.7*2*2/(41253.)   
		knox_errors /= np.sqrt(fsky)
		knox_errors *= np.abs(fieldav_cl)
		clerr = np.sqrt(knox_errors**2 + all_clerr_gal[fieldidx]**2)

		if colors is not None:
			color = colors[fieldidx]
		else:
			color = 'C'+str(fieldidx)

		plt.errorbar(lb[posmask], (pf*all_cl_gal[fieldidx])[posmask], yerr=(pf*clerr)[posmask],\
			capsize=capsize, label=ciber_field_dict[ifield], alpha=alph, fmt='o', color=color, markersize=markersize)

		plt.errorbar(lb[negmask], np.abs(pf*all_cl_gal[fieldidx])[negmask], yerr=(pf*clerr)[negmask],\
			mfc='white', capsize=capsize, fmt='o', alpha=alph, color=color, markersize=markersize)

	if plot_fieldav:
		plt.errorbar(lb[lbmask], (pf*fieldav_cl)[lbmask], yerr=(pf*np.std(all_cl_gal, axis=0))[lbmask], color='k', markersize=3, \
					capsize=3, fmt='o')


	if pred_fpaths is not None:
			
		jmock_pred = np.load(pred_fpaths[0])
		linestyles_pred = ['dashed', 'dashdot']
		lab_pred = 'IGL prediction'
		lb_pred, gal_auto = [jmock_pred[key] for key in ['lb', 'gal_auto']]
		pf_pred = lb_pred*(lb_pred+1)/(2*np.pi)

		plt.plot(lb_pred, pf_pred*gal_auto, color='C1', linestyle='dashdot', label=lab_pred)
	
	if textstr is not None:
		plt.text(textxpos, textypos, textstr, fontsize=text_fs, color='k')
	plt.xscale('log')
	plt.yscale('log')
	plt.ylim(ylim)
	plt.xlim(xlim)
	plt.xlabel('$\\ell$', fontsize=lab_fs)
	plt.tick_params(labelsize=12)

	if ylabel is None:
		ylabel = '$\\ell(\\ell+1)C_{\\ell}/2\\pi$'
	plt.ylabel(ylabel, fontsize=lab_fs)
	plt.grid(alpha=0.3)
	if include_legend:
		plt.legend(loc=legend_loc, fontsize=legend_fs)
	# plt.title(catname+', '+addstr, fontsize=14)
	plt.show()
	
	return fig, lb, all_cl_gal[0], all_clerr_gal[0]


def plot_twoband_fieldav_ciber_gal_ps(inst_list, catname, addstr=None, labels=None, \
							 figsize=(8, 4), capsize=3, markersize=3, plot_perfield=False, \
							 startidx=0, endidx=-1, xlim=[150, 1.1e5], ylim=[1e-4, 2e2], colors=['b', 'r'], \
							 textstrs=None, textxpos=200, textypos=5e1, text_fs=16, alph=0.6, \
							 bbox_to_anchor=[-0.05, 1.25], legend_fs=10, mask_frac=0.7, \
							  ifield_list=[4,5, 6, 7, 8], lab_fs=12):
	
	cbps = CIBER_PS_pipeline()
	ciber_field_dict = dict({4:'elat10', 5:'elat30', 6:'Bootes B', 7:'Bootes A', 8:'SWIRE'})
	lam_dict = dict({1:1.1, 2:1.8})
	all_fieldav_cl_cross, all_fieldav_clerr_cross = [], []

	fig, ax = plt.subplots(ncols=2, figsize=figsize, sharey=True)
	
	# plot CIBER x galaxy PS
	for idx, inst in enumerate(inst_list):
		
		lam = lam_dict[inst]
		cgps_file = load_ciber_gal_ps(inst, catname, addstr=addstr)

		lb, all_cl_gal, all_clerr_gal, all_cl_cross,\
			all_clerr_cross, ifield_list_use = [cgps_file[key] for key in ['lb', 'all_cl_gal', 'all_clerr_gal', 'all_cl_cross', 'all_clerr_cross', 'ifield_list_use']]
		
		pf = lb*(lb+1)/(2*np.pi)
		lbmask = (lb >= lb[startidx])*(lb < lb[endidx])
		nfield = len(ifield_list_use)
		
		if nfield > 1:
			
			cl_weights = 1./all_clerr_cross**2
#             cl_weights /= np.sum(cl_weights, axis=0)
			# fieldav_cl_cross, fieldav_clerr_cross = compute_weighted_cl(all_cl_cross.copy(), cl_weights)

			fieldav_cl_cross, fieldav_clerr_cross,\
				_, _ = compute_field_averaged_power_spectrum(all_cl_cross.copy(), per_field_dcls=all_clerr_cross.copy())


	
		else:
			fieldav_cl_cross = all_cl_cross[0]
			fieldav_clerr_cross = all_clerr_cross[0]
			
		cross_knox_errors = np.sqrt(1./((2*lb+1)*cbps.Mkk_obj.delta_ell)) # factor of 1 in numerator since auto is a cross-epoch cross
		fsky = mask_frac*nfield*2*2/(41253.) 
		cross_knox_errors /= np.sqrt(fsky)
		cross_knox_errors *= np.abs(fieldav_cl_cross)
		fieldav_clerr_cross = np.sqrt(cross_knox_errors**2 + fieldav_clerr_cross**2)
		
		all_fieldav_cl_cross.append(fieldav_cl_cross)
		all_fieldav_clerr_cross.append(fieldav_clerr_cross)
		
		
		for fieldidx, ifield in enumerate(ifield_list):
			cross_knox_indiv = cross_knox_errors*np.sqrt(nfield)
			all_clerr_cross[fieldidx] = np.sqrt(all_clerr_cross[fieldidx]**2 + cross_knox_indiv**2)
		
		if plot_perfield:
			for fieldidx, ifield in enumerate(ifield_list_use):
				
				posmask = lbmask*(all_cl_cross[fieldidx] > 0)
				negmask = lbmask*(all_cl_cross[fieldidx] < 0)
					
				
				ax[idx].errorbar(lb[posmask], (pf*all_cl_cross[fieldidx])[posmask], yerr=(pf*all_clerr_cross[fieldidx])[posmask],\
					capsize=capsize, label=ciber_field_dict[ifield], alpha=alph, fmt='o', color='C'+str(ifield-4), markersize=markersize)
				
				ax[idx].errorbar(lb[negmask], np.abs(pf*all_cl_cross[fieldidx])[negmask], yerr=(pf*all_clerr_cross[fieldidx])[negmask],\
					mfc='white', capsize=capsize, fmt='o', alpha=alph, color='C'+str(ifield-4), markersize=markersize)
		
		posmask = lbmask*(fieldav_cl_cross > 0)
		negmask = lbmask*(fieldav_cl_cross < 0)
		
		if labels is None:
			label = 'Field average'
		else:
			label = labels[idx]
		
		if plot_perfield:
			plot_color = 'k'
		else:
			plot_color = colors[idx]
			
			
		
		ax[idx].errorbar(lb[posmask], (pf*fieldav_cl_cross)[posmask], yerr=(pf*fieldav_clerr_cross)[posmask], fmt='o', \
					capsize=capsize, markersize=markersize, zorder=10, label='Field average', color=plot_color)
		ax[idx].errorbar(lb[negmask], np.abs(pf*fieldav_cl_cross)[negmask], yerr=(pf*fieldav_clerr_cross)[negmask], fmt='o', \
			capsize=capsize, markersize=markersize, mfc='white', zorder=10, color=plot_color)

		# if inst==1:
		#     textstr = 'CIBER 1.1 $\\mu$m $\\times$ unWISE\n"Red" sample, $W1<18$\nMask $J<17.5$'
		# elif inst==2:
		#     textstr = 'CIBER 1.8 $\\mu$m $\\times$ unWISE\n"Red" sample, $W1<18$\nMask $H<17.0$'
			
		if textstrs is not None:
			ax[idx].text(textxpos, textypos, textstrs[idx], fontsize=text_fs)

		ax[idx].set_xscale('log')
		ax[idx].set_yscale('log')
		ax[idx].set_xlim(xlim)
		ax[idx].set_ylim(ylim)
		ax[idx].set_xlabel('$\\ell$', fontsize=lab_fs)
		if idx==0:
			ax[idx].set_ylabel('$D_{\\ell}^{Ig}$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=lab_fs)
		ax[idx].grid(alpha=0.5)
#         ax[idx].tick_params(labelsize=12)

		if idx==0:
			ax[idx].legend(ncol=3, loc=2, bbox_to_anchor=bbox_to_anchor, fontsize=legend_fs)
		
	plt.subplots_adjust(wspace=0)
	plt.show()
	
	return fig, lb, all_fieldav_cl_cross[0], all_fieldav_clerr_cross[0]


def plot_correlation_matrices(corr_results, vmin=-0.5, vmax=1.0, 
                              figsize=(15, 3), cmap='RdBu_r'):
    """
    Plot correlation matrices for all redshift bins.
    
    Parameters
    ----------
    corr_results : dict
        Output from compute_correlation_matrices_from_mocks
    vmin, vmax : float
        Color scale limits
    figsize : tuple
        Figure size
    cmap : str
        Colormap name
    """
    
    corr_matrices = corr_results['corr_matrices']
    zbinedges = corr_results['zbinedges']
    lb = corr_results['lb']
    n_zbins = len(corr_matrices)
    
    fig, axes = plt.subplots(1, n_zbins, figsize=figsize)
    if n_zbins == 1:
        axes = [axes]
    
    for i, (ax, corr) in enumerate(zip(axes, corr_matrices)):
        im = ax.imshow(corr, cmap=cmap, vmin=vmin, vmax=vmax, 
                      aspect='auto', origin='lower')
        
        z0, z1 = zbinedges[i], zbinedges[i+1]
        ax.set_title(f'${z0:.1f} < z < {z1:.1f}$', fontsize=12)
        ax.set_xlabel(r'$\ell$ bin', fontsize=10)
        
        if i == 0:
            ax.set_ylabel(r'$\ell$ bin', fontsize=10)
        
        # Add tick labels at key multipoles
        n_ticks = 5
        tick_indices = np.linspace(0, len(lb)-1, n_ticks, dtype=int)
        ax.set_xticks(tick_indices)
        ax.set_yticks(tick_indices)
        ax.set_xticklabels([f'{int(lb[i])}' for i in tick_indices], fontsize=8)
        ax.set_yticklabels([f'{int(lb[i])}' for i in tick_indices], fontsize=8)
    
    plt.colorbar(im, ax=axes, label='Correlation', fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    return fig

def gen_paper_plots(inst_list=[1,2], save=True, dirname=None, basepath='figures/ciber_gal_cross/', \
				   maskstr='JHlt16'):
	
	# number counts of LS and HSC with redshift
	# with whatever cut we choose for HSC..
	zbinedges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

	labels = ['LS ($z<22$)', 'HSC ($g<25$)']
	fig_photoz = plot_photoz_dist(['LS', 'HSC'], colors=['k', 'C4'], zbinedges=zbinedges, ylim=[1e2, 1e5], labels=labels)
	
	# cross-correlation forecast (two panel, one for indiv. components and other for varying nbar)
	fig_forecast = plot_clIG_forecast()
	#  ------------------------------------- Legacy Survey ----------------------------------------
	
	
	# load LS data
	zbinedges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]



	lb, full_cl_cross, full_clerr_cross, full_cl_gal, full_clerr_gal = collect_ciber_gal_vs_redshift('LS', subtract_randoms=True, \
																									inst_list=inst_list, zbinedges=zbinedges)
	
	# plot with LS galaxy auto vs redshift
	fig_ls_auto = plot_gal_ps_vs_redshift(1, zbinedges, catname='LS', ylim=[3e-5, 2e4], headstr=None)
	# LS x CIBER cross vs redshift
	fig_ls_ciber = plot_cross_ps_vs_redshift(inst_list, zbinedges, lb, full_cl_cross, full_clerr_cross, figsize=(10, 5.5), \
							   xlim=[250, 1.1e5], ylim=[1e-3, 4e2], markersize=3, capsize=3, alph=0.8, textxpos=300, \
							   color='k', bbox_to_anchor=[3.5, 1.4])
	
	#  ---------------------- CIBER x unWISE auto, crosses (three panel) -------------------------
	
	labels = ['1.1 $\\mu$m $\\times$ unWISE', '1.8 $\\mu$m $\\times$ unWISE']

	fig_wise_auto = plot_perfield_gal_auto('WISE', 1, addstr='unWISE_neo8', capsize=3, markersize=3, \
							ylim=[5e-4, 1e2], \
							alph=0.6, legend_fs=12, startidx=0, xlim=[150, 1.05e5])


	fig_wise_ciber, _, _, _ = plot_twoband_fieldav_ciber_gal_ps(inst_list, 'WISE', addstr='unWISE_neo8', capsize=2.5, markersize=3., \
								   textstr=None, xlim=[300, 1.05e5], ylim=[5e-4, 1e2], textxpos=350, textypos=7, text_fs=12, \
								   plot_perfield=True, alph=0.5, bbox_to_anchor=[0.2, 1.3], legend_fs=12, labels=labels, \
									startidx=2, figsize=(8, 3.5))

	
	# ------------------------ CIBER x HSC auto, crosses (two panel) -------------------------------
	zbinedges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
	fig_hsc_auto = plot_gal_ps_vs_redshift(1, zbinedges, catname='HSC', ylim=[3e-5, 2e4], headstr='hsc_glt24')
	labels_hsc = ['1.1 $\\mu$m $\\times$ HSC ($g<24$)', '1.8 $\\mu$m $\\times$ HSC ($g<24$)']
	
	hsc_mag_max = 24.0
	textstr = 'HSC $g<'+str(hsc_mag_max)+'$\nSWIRE'

	fig_hsc_ciber = plot_fieldav_ciber_gal_ps([1, 2], 'HSC', addstr='hsc_glt24.0', figsize=(5, 4), capsize=3, markersize=3.5, \
								   textstr=textstr, xlim=[250, 1.05e5], ylim=[5e-4, 1e2], textxpos=350, textypos=10, text_fs=14, \
								   plot_perfield=False, alph=0.6, bbox_to_anchor=[0.0, 1.3], legend_fs=11, labels=labels, \
									startidx=0)
	
#     fig_list = [fig_photoz,\
#                 fig_ls_auto, fig_wise_auto, fig_hsc_auto,\
#                     fig_ls_ciber, fig_wise_ciber, fig_hsc_ciber]
		
	fig_list = [fig_photoz,\
				fig_ls_auto, fig_wise_auto,\
					fig_ls_ciber]
	
	
	# -------------------- save figures to result directory -----------------------
	if save:
		if dirname is None:
			print('Need directory name to continue')
		else:
			if not os.path.isdir(basepath+dirname):
				print('Making result directory..', basepath+dirname)
				os.makedirs(basepath+dirname)
				
			for f, fig_indiv in enumerate(fig_list):
				fig_indiv.savefig(basepath+dirname+'/fig'+str(f)+'.pdf', bbox_inches='tight')
	
	return fig_list


