import numpy as np
import config
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt

from plotting_fns import plot_map
from scipy.stats import chi2


from ciber_data_file_utils import load_ciber_gal_ps
from powerspec_utils import *
from ciber_powerspec_pipeline import *


from dataclasses import dataclass
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


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


def plot_cross_ps_by_wavelength_and_redshift(
	all_catalogs_cl_cross, 
	all_catalogs_clerr_cross,
	catnames, 
	zbinedges, 
	lb, 
	inst=[1, 2],
	figsize=(16, 8), 
	startidx=2, 
	endidx=-1,
	xlim=[150, 1.1e5], 
	ylim=[1e-4, 2e2], 
	legend_fs=12, 
	capsize=3, 
	markersize=3, 
	alph=0.8,
	textxpos=280, 
	textypos=1e2, 
	text_fs=12, 
	colors_cat=['C0', 'C1'], 
	bbox_to_anchor=[1.0, 1.35],
	ncol_legend=2,
	all_catalogs_pred_fpaths=None, 
	pred_alpha=0.8,
	tl_pix_correct=False,
	linestyles_pred = ['dashed', 'dashdot'],
	label_fs=13):
	"""
	Plots cross-power spectra for multiple wavelengths and redshift bins.

	Each row corresponds to a wavelength (1.1um, 1.8um).
	Each column corresponds to a redshift bin.
	Each subplot shows measurements for different galaxy catalogs and an optional model prediction.
	"""
	
	lam_dict = {1: 1.1, 2: 1.8}
	pf = lb * (lb + 1) / (2 * np.pi)
	lbmask = (lb >= lb[startidx]) & (lb < lb[endidx])

	nrows = len(inst)
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
				fieldav_cl_cross = all_catalogs_cl_cross[cat_idx][inst_idx][zidx]
				fieldav_clerr_cross = all_catalogs_clerr_cross[cat_idx][inst_idx][zidx]

				posmask = lbmask & (fieldav_cl_cross > 0)
				negmask = lbmask & (fieldav_cl_cross < 0)

				current_ax.errorbar(lb[posmask], (pf * fieldav_cl_cross)[posmask], yerr=(pf * fieldav_clerr_cross)[posmask],
									color=colors_cat[cat_idx], fmt='o', capsize=capsize, markersize=markersize, 
									zorder=15, label=catname, alpha=alph)
				
				current_ax.errorbar(lb[negmask], np.abs(pf * fieldav_cl_cross)[negmask], yerr=(pf * fieldav_clerr_cross)[negmask],
									color=colors_cat[cat_idx], fmt='o', capsize=capsize, markersize=markersize, 
									zorder=15, mfc='white', alpha=alph)
				
				
				# --- Handle model predictions (if provided) ---
				if all_catalogs_pred_fpaths is not None:
					pred_path = all_catalogs_pred_fpaths[cat_idx][inst_idx][zidx]
					jmock_pred = np.load(pred_path)

					if inst_indiv == 1 and cat_idx==0:
						lab_pred = 'IGL (Mirocha)'
					else:
						lab_pred = None

					lb_pred, cross = jmock_pred['lb'], jmock_pred['cross']
					pf_pred = lb_pred * (lb_pred + 1) / (2 * np.pi)

					if tl_pix_correct:
						ifield_use = 6
						tl_pix_path = f'data/fluctuation_data/transfer_function/tl_clx_pix_TM{inst_indiv}_ifield{ifield_use}.npz'
						tl_pix = np.load(tl_pix_path)['tl_clx_pix']
						cross /= tl_pix

					current_ax.plot(lb_pred, pf_pred * cross, color=colors_cat[cat_idx], 
									linestyle=linestyles_pred[cat_idx], alpha=pred_alpha, label=lab_pred)


				
				

			# --- SUBPLOT FORMATTING ---
			
			zlab = str(np.round(z0, 1))+'$<z_{phot}<$'+str(np.round(z1, 1))
#             gal_label = f'{z0:.1f} < $z_{phot}$ < {z1:.1f}'
			
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
				ylabel_text = f'$D_\\ell^{{Ig}}$ [nW m$^{{-2}}$ sr$^{{-1}}$]'
				current_ax.set_ylabel(ylabel_text, fontsize=label_fs)

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
		ax[inst].set_ylabel('$b_I dI/dz_{phot}$', fontsize=14)
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

#     if colors is None:
#         # Use matplotlib's default color cycle
# #         prop_cycle = plt.rcParams['axes.prop_cycle']
#         colors = [prop_cycle.by_key()['color'][i % len(prop_cycle)] for i in range(num_datasets)]

	plt.style.use('default')

	# --- 2. Plotting Setup ---
#     plt.style.use('seaborn-v0_8-whitegrid')
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

def mini_proc_clav(all_cl, all_clerr, lb, startidx, endidx, mode='auto', fmask=0.7):
	
	cbps = CIBER_PS_pipeline()
	pf = lb*(lb+1)/(2*np.pi)
	lbmask = (lb >= lb[startidx])*(lb < lb[endidx])

	nfield = len(all_cl)
	
	if len(all_cl) > 1:
		fieldav_cl, fieldav_clerr,\
			_, _ = compute_field_averaged_power_spectrum(all_cl.copy(), per_field_dcls=all_clerr.copy())
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
		gal_knox_errors *= np.abs(fieldav_cl)
		fieldav_clerr = np.sqrt(gal_knox_errors**2 + fieldav_clerr**2)

	posmask = lbmask*(fieldav_cl > 0)
	negmask = lbmask*(fieldav_cl < 0)
	
	return pf, posmask, negmask, fieldav_cl, fieldav_clerr


def compute_rl_ciber_gal(addstr, inst_list=[1, 2], catname='LS', gal_label='LS ($z<1$)', startidx=2, endidx=-1, tl_pix_correct=True, 
						ifield_use=8):
	
	bandstr_list = ['J', 'H']
	lams = [1.1, 1.8]
	
	all_r_ell, all_r_ell_unc = [], []
	
	all_auto_cross_data = []
	
	keys = ['lb', 'all_cl_gal', 'all_clerr_gal', 'all_cl_cross', 'all_clerr_cross']
	
	for idx, inst in enumerate(inst_list):
				
		print(catname, addstr)
		cgps_file = load_ciber_gal_ps(inst, catname, addstr=addstr)
		
		lb, all_cl_gal, all_clerr_gal, \
			all_cl_cross, all_clerr_cross = [cgps_file[key] for key in keys]
		
		pf, posmask_auto, negmask_auto, fieldav_cl_gal, fieldav_clerr_gal = mini_proc_clav(all_cl_gal, all_clerr_gal, lb, startidx, endidx, mode='auto')
		pf, posmask, negmask, fieldav_cl_cross, fieldav_clerr_cross = mini_proc_clav(all_cl_cross, all_clerr_cross, lb, startidx, endidx, mode='cross')

		
		if catname=='HSC':
			bandstr = bandstr_list[idx]
			mag_lim = 16.0

			observed_run_name = 'observed_'+bandstr+'lt'+str(mag_lim)+'_072424_quadoff_grad_fcsub_order2'

			fpath = config.ciber_basepath+'data/input_recovered_ps/111323/TM'+str(inst)+'/'+observed_run_name+'/input_recovered_ps_estFF_simidx0.npz'

			clfile = np.load(fpath)
			# print([k for k in clfile.keys()])

			lb_auto, cl_auto_all, clerr_auto_all = [clfile[key] for key in ['lb', 'recovered_ps_est_nofluc', 'recovered_dcl']]

			# print('cl auto', cl_auto_all[-1])
			lb_auto = lb_auto[startidx:endidx]
			cl_auto = cl_auto_all[-1,startidx:endidx]
			clerr_auto = clerr_auto_all[-1, startidx:endidx]
			
			nfield_use = 1
		else:
			ciber_auto = np.load('data/ciber_auto_'+bandstr_list[idx]+'lt16.0_F25B.npz')

			lb_auto, cl_auto, clerr_auto = [ciber_auto[key] for key in ['lb', 'fieldav_cl', 'fieldav_clerr']]

			nfield_use = 5
				
		fieldav_clerr_cross_ana = estimate_cross_uncertainties(lb,
															   fieldav_cl_cross,
															   fieldav_clerr_cross,
															   cl_auto, fieldav_cl_gal, nfield_use, \
															  startidx=2, endidx=-1)
		
		if tl_pix_correct:

			tl_pix = np.load('data/fluctuation_data/transfer_function/tl_clx_pix_TM'+str(inst)+'_ifield'+str(ifield_use)+'.npz')['tl_clx_pix']
			fieldav_cl_cross /= tl_pix 
			fieldav_clerr_cross_ana /= tl_pix


		r_ell = fieldav_cl_cross[startidx:endidx]/np.sqrt(cl_auto*fieldav_cl_gal[startidx:endidx])

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


def estimate_cross_uncertainties(lb, clx, clx_err, clI_auto, clg_auto, nfield, startidx=2, endidx=-1, fmask=0.7):
	
	# clx_err includes N_ell^I x (C_ell^g + 1/n)

	cbps = CIBER_PS_pipeline()
	dclx_sq = np.ones_like(lb)
	lbmask = (lb >= lb[startidx])*(lb < lb[endidx])
	
	nmode_inv = 1./((2*lb+1)*cbps.Mkk_obj.delta_ell)
	fsky = nfield*fmask*2*2/(41253.) 
	nmode_inv /= fsky
	
	nbar_inv = np.mean(clg_auto[-4:endidx])
	
	# cl_terms = clx[lbmask]**2 + np.abs(clI_auto*clg_auto[lbmask]) + clx_err[lbmask]**2 + clI_auto*nbar_inv
	cl_terms = clx[lbmask]**2 + np.abs(clI_auto*clg_auto[lbmask]) + clI_auto*nbar_inv
	

	dclx_sq_A = nmode_inv[lbmask]*cl_terms 

	dclx_sq[lbmask] = dclx_sq_A + clx_err[lbmask]**2 # since noise x clg computed from MC realizations don't normalize by Nmodes, already done.


	# dclx_sq[lbmask] = nmode_inv[lbmask]*cl_terms
	
	return np.sqrt(dclx_sq)



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



def field_consistency_gal_cross(catname, addstr, inst_list=[1, 2], ifield_list=[4, 5, 6, 7, 8],
								 startidx=0, endidx=None,
								 ell_min=300, ell_max=10000, figsize=(6, 4), use_zscore=False):
	"""
	Compute PTE values for per-field cross-spectra relative to the field average,
	restricted to ell_min < ell < ell_max.
	Also makes a fractional deviation plot for all bands in one figure.
	"""
	
	
	cbps = CIBER_PS_pipeline()
	
	bandstr_list = ['J', 'H']
	lams = [1.1, 1.8]

	pte_results = np.zeros((len(inst_list), len(ifield_list)))


	# Apply ell mask


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
			ell_mask = (lb >= ell_min) & (lb <= ell_max)
			if endidx is not None:
				ell_mask &= (np.arange(len(lb)) < endidx)
			if startidx is not None:
				ell_mask &= (np.arange(len(lb)) >= startidx)

			all_chi_results = np.zeros((len(inst_list), len(ifield_list), np.sum(ell_mask)))


		# Field averages
		pf, posmask_auto, negmask_auto, fieldav_cl_gal, fieldav_clerr_gal = mini_proc_clav(
			all_cl_gal, all_clerr_gal, lb, startidx, endidx, mode='auto'
		)
		pf, posmask, negmask, fieldav_cl_cross, fieldav_clerr_cross = mini_proc_clav(
			all_cl_cross, all_clerr_cross, lb, startidx, endidx, mode='cross'
		)

		# Load CIBER auto for uncertainty estimation
		ciber_auto = np.load(f'data/ciber_auto_{bandstr_list[idx]}lt16.0_F25B.npz')
		lb_auto, cl_auto, clerr_auto = [ciber_auto[key] for key in ['lb', 'fieldav_cl', 'fieldav_clerr']]

		# Per-field uncertainties
		perf_clerr_cross = np.zeros((len(ifield_list), fieldav_cl_cross.shape[0]))
		
		for fieldidx, ifield in enumerate(ifield_list):

			if ifield==8:
				fmask = 0.6
			else:
				fmask = 0.7

			perf_clerr_cross[fieldidx] = estimate_cross_uncertainties(
				lb, fieldav_cl_cross, all_clerr_cross[fieldidx],
				cl_auto, fieldav_cl_gal, 1, startidx=2, endidx=-1
			)
			
			 
		# === RECOMPUTE FIELD AVERAGE USING PROPER WEIGHTS ===
	
		fieldav_cl, fieldav_clerr,\
			_, perf_weights = compute_field_averaged_power_spectrum(all_cl_cross.copy(), per_field_dcls=perf_clerr_cross.copy())
			
		
		ax = axes[idx]  # select subplot for this band

		# Offset settings for better viewing
		offsets = np.linspace(-0.05, 0.05, len(ifield_list))  # small shifts in log-space

		# Compute PTE + plot fractional deviations
		for fieldidx, ifield in enumerate(ifield_list):
			cl_field = all_cl_cross[fieldidx, ell_mask]
			cl_mean = fieldav_cl_cross[ell_mask]
			cl_err = perf_clerr_cross[fieldidx, ell_mask]

			# PTE calculation
			chi_perbp = ((cl_field - cl_mean) / cl_err)

			chi2_perbp = ((cl_field - cl_mean) / cl_err) ** 2
			chi2_val = np.sum(chi2_perbp)
			dof = len(cl_field)
			pte = 1 - chi2.cdf(chi2_val, dof)
			pte_results[idx, fieldidx] = pte
			all_chi_results[idx, fieldidx] = chi_perbp

			print(f"Inst {inst}, Field {ifield}: chi2={chi2_val:.2f}, dof={dof}, PTE={pte:.3f}")
			
			label = cbps.ciber_field_dict[ifield]
			if pte > 1e-3:

				label += ' (PTE='+str(np.round(pte, 3))+')'
			else:
				label += ' (PTE$<$0.001)'

			# Fractional deviation
			frac_dev = (cl_field - cl_mean) / cl_mean
			frac_err = cl_err / cl_mean

			zscore = (cl_field - cl_mean)/ cl_err

			# Apply offset in log-space (multiplicative shift)
			lb_shifted = lb[ell_mask] * (1 + offsets[fieldidx])

			if use_zscore:
				ax.scatter(lb_shifted, zscore, label=label, alpha=0.8)
			else:
				ax.errorbar(lb_shifted, frac_dev, yerr=frac_err,
							fmt='o', label=label, alpha=0.8)

				
		ell_centers = lb[ell_mask]
		ell_edges = np.zeros(len(ell_centers) + 1)
		ell_edges[1:-1] = 0.5 * (ell_centers[1:] + ell_centers[:-1])
		ell_edges[0] = ell_centers[0] - (ell_centers[1] - ell_centers[0]) / 2
		ell_edges[-1] = ell_centers[-1] + (ell_centers[-1] - ell_centers[-2]) / 2

		# Shade alternating vertical bands
		for i in range(len(ell_centers)):
			if i % 2 == 0:  # alternate shading
				ax.axvspan(ell_edges[i], ell_edges[i+1], color='gray', alpha=0.1, zorder=0)


		ax.axhline(0, color='k', lw=1)
		ax.set_xscale('log')

		if use_zscore:

			ax.set_ylabel('$\\Delta C_{\\ell}^{i}/\\sigma(C_{\\ell}^i)$', fontsize=14)
		else:
			ax.set_ylabel('$\\Delta C_{\\ell}^{i}/\\overline{C}_{\\ell}  - 1$', fontsize=14)

		ax.grid(alpha=0.3)
		ax.set_xlim(250, ell_max*1.2)
		ax.legend(fontsize=12, loc=4, bbox_to_anchor=[1.5, 0.1])
		
		text = 'CIBER '+str(lams[idx])+' $\\mu$m $\\times$ '+catname
		ax.text(850, 4.0, text, fontsize=16)
		ax.set_ylim(-6, 6)
		ax.tick_params(labelsize=12)

	axes[-1].set_xlabel(r'$\ell$', fontsize=14)
	
	plt.subplots_adjust(hspace=0.)

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


def compare_r_ell_hsc_LS_zlt1(figsize=(7, 3),
							  ls_addstr='0.0_z_1.0_wrandsub_JHlt16',
							  hsc_addstr='hsc_ilt24.0_zlt1_wrandsub',
							  startidx=2, endidx=-1,
							  title_fs=14, lab_fs=14,
							  ylim=[-0.1, 1.1],
								xlim=[250, 1.1e5],
							  grid_alpha=0.3,
							  capsize=3, capthick=1.5, markersize=3, \
							  ls_plotstr='Legacy Survey ($z<1$)', 
							  hsc_plotstr='HSC ($z<1$, $i<24$)', 
							  hsc_pred_fpaths=None, ls_pred_fpaths=None, 
								alpha=0.8, tl_pix_correct=True, ifield_use=8):
	lams = [1.1, 1.8]
	inst_list = [1, 2]
	
	ls_auto_cross = compute_rl_ciber_gal(ls_addstr, catname='LS',
										 tl_pix_correct=tl_pix_correct, ifield_use=ifield_use)
	hsc_auto_cross = compute_rl_ciber_gal(hsc_addstr, catname='HSC',
										 tl_pix_correct=tl_pix_correct, ifield_use=ifield_use)


	fig, ax = plt.subplots(figsize=figsize, ncols=2, nrows=1, sharey=True)

	for idx, inst in enumerate(inst_list):
	
		title = 'CIBER '+str(lams[idx])+' $\\mu$m $\\times$ $\\delta_g$ ($z<1$)'
		ax[idx].set_title(title, fontsize=title_fs)
		lb, r_ell_ls, r_ell_unc_ls = ls_auto_cross[idx].lb, ls_auto_cross[idx].r_ell, ls_auto_cross[idx].r_ell_unc
		lb, r_ell_hsc, r_ell_unc_hsc = hsc_auto_cross[idx].lb, hsc_auto_cross[idx].r_ell, hsc_auto_cross[idx].r_ell_unc

		ax[idx].errorbar(lb[startidx:endidx], r_ell_ls, yerr=r_ell_unc_ls, fmt='o', capsize=capsize, markersize=markersize, capthick=capthick, color='C0', label=ls_plotstr)
		ax[idx].errorbar(lb[startidx:endidx], r_ell_hsc, yerr=r_ell_unc_hsc, fmt='o', capsize=capsize, markersize=markersize, capthick=capthick, color='C1', label=hsc_plotstr)

		ax[idx].set_xscale('log')
		ax[idx].set_ylim(ylim)
		if idx==0:
			ax[idx].set_ylabel('$r_{\\ell}=C_{\\ell}^{Ig}/\\sqrt{C_{\\ell}^{gg}C_{\\ell}^{II}}$', fontsize=14)
		ax[idx].set_xlabel('$\\ell$', fontsize=lab_fs)
		ax[idx].grid(alpha=grid_alpha)
		ax[idx].set_xlim(xlim)

		if tl_pix_correct:
			tl_pix = np.load('data/fluctuation_data/transfer_function/tl_clx_pix_TM'+str(inst)+'_ifield'+str(ifield_use)+'.npz')['tl_clx_pix']
		else:
			tl_pix = np.ones_like(lb_pred)

		if ls_pred_fpaths is not None:

			jmock_pred = np.load(ls_pred_fpaths[idx])
			lb_pred, r_ell_ls_pred = [jmock_pred[key] for key in ['lb', 'rlx_tracer_full']]

			r_ell_ls_pred /= tl_pix 

			ax[idx].plot(lb_pred, r_ell_ls_pred, color='C0', linestyle='dashdot', label='IGL (Mirocha) + ISL', alpha=alpha)


		if hsc_pred_fpaths is not None:

			jmock_pred = np.load(hsc_pred_fpaths[idx])
			lb_pred, r_ell_hsc_pred = [jmock_pred[key] for key in ['lb', 'rlx_tracer_full']]

			r_ell_hsc_pred /= tl_pix 


			ax[idx].plot(lb_pred, r_ell_hsc_pred, color='C1', linestyle='dashdot', alpha=alpha)

		if idx==1:
			ax[idx].legend()

	plt.subplots_adjust(wspace=0.05)

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


def plot_gal_and_ciber_auto(all_acdat, pred_fpaths=None,
							colors=['b', 'r'],
							xlim=[250, 1.1e5],
							ylims_gal=[[5e-4, 5e1], [2e-3, 2e2]],
							ylim_ciber=[1, 1e4],
							gal_labels=None,
							band_labels=None,
							startidx=2, endidx=-1,
							capsize=3, markersize=3,
							figsize=(10, 4.5), lab_fs=12, title_fs=12, legend_fs=10, \
						   pred_alpha=0.5, spacer_and_ciber_auto=[0.35, 1.2], 
						   tl_pix_correct=True, ifield_use=6):

	n_gal = len(all_acdat)  # number of galaxy catalogs
	if gal_labels is None:
		gal_labels = [f"Catalog {i+1}" for i in range(n_gal)]
	if band_labels is None:
		band_labels = ['J', 'H']

	fig = plt.figure(figsize=figsize)
	
	widths = [1]*n_gal + spacer_and_ciber_auto  # small spacer column + CIBER panel
	gs = GridSpec(2, n_gal + 2, width_ratios=widths, wspace=0.0, hspace=0)


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
	
	for col, cat_acdats in enumerate(all_acdat):
			
		
		for idx_band, acdat in enumerate(cat_acdats):

			
			if pred_fpaths is not None:
			
				jmock_pred = np.load(pred_fpaths[col][idx_band])

				lb_pred, gal_auto, intensity_auto, cross = [jmock_pred[key] for key in ['lb', 'gal_auto', 'intensity_auto_full', 'cross']]
				pf_pred = lb_pred*(lb_pred+1)/(2*np.pi)
				
				if col==0:
					intensity_auto_preds.append(intensity_auto)

			if tl_pix_correct:
				tl_pix = np.load('data/fluctuation_data/transfer_function/tl_clx_pix_TM'+str(idx_band+1)+'_ifield'+str(ifield_use)+'.npz')['tl_clx_pix']
			
			else:
				tl_pix = np.ones_like(lb_pred)

			cross /= tl_pix

			acdat.fieldav_cl_cross /= tl_pix
			acdat.fieldav_clerr_cross /= tl_pix
			
			pf, lb = acdat.pf, acdat.lb
			color = colors[idx_band]

			if idx_band==0:
				# Row 0: galaxy auto
				ax_gal[0, col].errorbar(
					lb[acdat.posmask_auto],
					(pf * acdat.fieldav_cl_gal)[acdat.posmask_auto],
					yerr=(pf * acdat.fieldav_clerr_gal)[acdat.posmask_auto],
					color='k', fmt='o', capsize=capsize, markersize=markersize,
					zorder=15, label=band_labels[idx_band]
				)
				ax_gal[0, col].errorbar(
					lb[acdat.negmask_auto],
					np.abs(pf * acdat.fieldav_cl_gal)[acdat.negmask_auto],
					yerr=(pf * acdat.fieldav_clerr_gal)[acdat.negmask_auto],
					color='k', fmt='o', capsize=capsize, markersize=markersize,
					mfc='white', zorder=15
				)
				
			if pred_fpaths is not None:
				
				if idx_band==0:

					ax_gal[0, col].plot(lb_pred, pf_pred*gal_auto, color='grey', linestyle='dotted', alpha=pred_alpha)


				ax_gal[1, col].plot(lb_pred, pf_pred*cross, color=colors[idx_band], linestyle='dotted', alpha=pred_alpha)


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

		ax_gal[0, col].set_title(gal_labels[col], fontsize=title_fs)
		
		
	# Hide y-ticks for galaxy columns > 0
	for row in range(2):
		for col in range(1, n_gal):
			ax_gal[row, col].tick_params(labelleft=False)

	# Hide x-ticks for top row
	for col in range(n_gal):
		ax_gal[0, col].tick_params(labelbottom=False)

	# Shared y-axis labels for the leftmost column, hide others
	for row in range(2):
		ax_gal[row, 0].set_ylabel([r'$D_\ell^{gg}$', r'$D_\ell^{Ig}$ [nW m$^{-2}$ sr$^{-1}]$'][row], fontsize=lab_fs)
		for col in range(1, n_gal):
			ax_gal[row, col].set_yticklabels([])  # hide ticks
			ax_gal[row, col].set_ylim(ylims_gal[row])

	# Apply y-limits to first col and match others
	for row in range(2):
		ax_gal[row, 0].set_ylim(ylims_gal[row])

	# Bottom x-labels for galaxy panels
	for col in range(n_gal):
		ax_gal[1, col].set_xlabel(r'$\ell$', fontsize=lab_fs)

	# Right block: CIBER auto, spans both rows
	# CIBER auto panel: loop over all bands for plotting
	
	gs_right = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:, -1], hspace=0, wspace=0.0, height_ratios=[0.5, 0.5])
	ax_ciber = fig.add_subplot(gs_right[1, 0])  # middle row => vertically centered
	
#     ax_ciber = fig.add_subplot(gs[:, -1])
	for idx_band, acdat in enumerate(all_acdat[0]):  # assuming all catalogs have same bands
		pf, lb = acdat.pf, acdat.lb
		ax_ciber.errorbar(
			lb[startidx:endidx],
			pf[startidx:endidx] * acdat.ciber_auto_cl,
			yerr=pf[startidx:endidx] * acdat.ciber_auto_clerr,
			fmt='o', color=colors[idx_band], capsize=capsize, markersize=markersize,
			label=f"CIBER {band_labels[idx_band]}"
		)
		
	if pred_fpaths is not None:
				
		for idx_band in range(len(cat_acdats)):
			if idx_band==0:
				pred_lab = 'IGL + ISL'
				
				ax_ciber.plot(lb_pred, np.zeros_like(intensity_auto_preds[idx_band]), color='k', linestyle='dotted', alpha=pred_alpha, \
						 label='IGL (Mirocha)')
			
			else:
				pred_lab = None

			ax_ciber.plot(lb_pred, pf_pred*intensity_auto_preds[idx_band], color=colors[idx_band], linestyle='dashdot', alpha=pred_alpha, \
						 label=pred_lab)

	ax_ciber.set_xscale('log')
	ax_ciber.set_yscale('log')
	ax_ciber.set_xlim(xlim)
	ax_ciber.set_ylim(ylim_ciber)
	ax_ciber.set_ylabel(r'$D_\ell^{II}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=lab_fs)
	ax_ciber.set_xlabel(r'$\ell$', fontsize=lab_fs)
	ax_ciber.grid(alpha=0.3)
	ax_ciber.legend(fontsize=legend_fs, bbox_to_anchor=[0.05, 1.8], loc=2)

	plt.show()
	return fig


def plot_rl_gal(all_acdat, colors=['b', 'r'], inst_list=[1, 2], figsize=(5, 3), ylim=[-0.1, 1.1], \
			   lab_fs=14, markersize=3, capsize=3, startidx=2, endidx=-1, gal_label='LS ($z<1$)', \
			   pred_fpaths=None):
	
	lams = [1.1, 1.8]
	linestyles_pred = ['dashed', 'dashdot']


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
				lab_pred = 'IGL (Mirocha)'
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
				lab_pred = 'IGL (Mirocha)'
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
								  headstr=None):

	bandstr_list = ['J', 'H']

	nbin = len(zbinedges)-1

	cbps = CIBER_PS_pipeline()

	lb = cbps.Mkk_obj.midbin_ell

	full_cl_cross, full_clerr_cross = [[np.zeros((len(zbinedges)-1, len(lb))) for x in range(2)] for y in range(2)]
	full_cl_gal, full_clerr_gal = [[np.zeros((len(zbinedges)-1, len(lb))) for x in range(2)] for y in range(2)]


	full_cl_cross_perf, full_clerr_cross_perf = [np.zeros((len(inst_list), len(zbinedges)-1, len(ifield_list), len(lb))) for y in range(2)]


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

			print('ifield use:', ifield_list_use)

			print('all cl gal has shape', all_cl_gal)

			print('all cl cross:', all_cl_cross)

#             gal_counts, noise_base_path = load_delta_g_maps(catname, inst, addstr)

			if len(ifield_list_use)==1:

				fieldav_cl_gal, fieldav_clerr_gal = all_cl_gal[0], all_clerr_gal[0]
				fieldav_cl_cross, fieldav_clerr_cross = all_cl_cross[0], all_clerr_cross[0]

			else:

				pf, posmask_auto, negmask_auto, fieldav_cl_gal, fieldav_clerr_gal = mini_proc_clav(
					all_cl_gal, all_clerr_gal, lb, startidx, endidx, mode='auto'
				)
				pf, posmask, negmask, fieldav_cl_cross, fieldav_clerr_cross = mini_proc_clav(
					all_cl_cross, all_clerr_cross, lb, startidx, endidx, mode='cross'
				)

			# Load CIBER auto for uncertainty estimation
			ciber_auto = np.load(f'data/ciber_auto_{bandstr_list[idx]}lt16.0_F25B.npz')
			lb_auto, cl_auto, clerr_auto = [ciber_auto[key] for key in ['lb', 'fieldav_cl', 'fieldav_clerr']]

			# Per-field uncertainties
			perf_clerr_cross = np.zeros((len(ifield_list_use), fieldav_cl_cross.shape[0]))

			# print('perf clerr cross has shape', perf_clerr_cross)
			for fieldidx, ifield in enumerate(ifield_list):
				perf_clerr_cross[fieldidx] = estimate_cross_uncertainties(
					lb, fieldav_cl_cross, all_clerr_cross[fieldidx],
					cl_auto, fieldav_cl_gal, 1, startidx=2, endidx=-1
				)


			full_clerr_cross_perf[idx, n] = perf_clerr_cross
			full_cl_cross_perf[idx, n] = all_cl_cross


			# === RECOMPUTE FIELD AVERAGE USING PROPER WEIGHTS ===

			if len(ifield_list_use)==1:
				fieldav_cl, fieldav_clerr = fieldav_cl_cross, perf_clerr_cross[0]

			else:

				fieldav_cl, fieldav_clerr,\
					_, perf_weights = compute_field_averaged_power_spectrum(all_cl_cross.copy(), per_field_dcls=perf_clerr_cross.copy())


			if tl_pix_correct:

				ifield_use = 6
				tl_pix = np.load('data/fluctuation_data/transfer_function/tl_clx_pix_TM'+str(inst)+'_ifield'+str(ifield_use)+'.npz')['tl_clx_pix']
			

			else:
				tl_pix = np.ones_like(fieldav_cl_cross)

			fieldav_cl_cross /= tl_pix 
			fieldav_clerr /= tl_pix


			# full_cl_cross[idx][n] = fieldav_cl
			# full_clerr_cross[idx][n] = fieldav_clerr # heeee

			full_cl_cross[idx][n] = fieldav_cl_cross
			full_clerr_cross[idx][n] = fieldav_clerr # heeee

			full_cl_gal[idx][n] = fieldav_cl_gal
			full_clerr_gal[idx][n] = fieldav_clerr_gal



	res = dict({'lb':lb, 
				'full_cl_cross':np.array(full_cl_cross),
				'full_clerr_cross':np.array(full_clerr_cross),
				'full_cl_gal':np.array(full_cl_gal),
				'full_clerr_gal':np.array(full_clerr_gal),
				'full_cl_cross_perf':np.array(full_cl_cross_perf),
				'full_clerr_cross_perf':np.array(full_clerr_cross_perf)
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
			
#             if zidx==0 and fieldidx==0:
#                 plot_map(galcounts, title='gal counts')
			
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
		
		gal_label = str(np.round(z0, 1))+'$<z_{phot}<$'+str(np.round(z1, 1))


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
							 xlim=[150, 1.1e5], ylim=[1e-4, 2e2], legend_fs=16, capsize=3, markersize=3, alph=0.8, \
							 textxpos=280, textypos=1e2, text_fs=12, color=None, color_inst=['b', 'C3'], bbox_to_anchor=[2.0, 1.4], \
							 ncols=4, nrows=2, all_pred_fpaths=None, pred_alpha=0.8, \
							 ncol_legend=3, tl_pix_correct=False):
	
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
		
	linestyles_pred = ['dashed', 'dashdot']

			
	for zidx, z0 in enumerate(zbinedges[:-1]):
		
		for i, inst_indiv in enumerate(inst):

			if tl_pix_correct:
				ifield_use = 6
				tl_pix = np.load('data/fluctuation_data/transfer_function/tl_clx_pix_TM'+str(inst_indiv)+'_ifield'+str(ifield_use)+'.npz')['tl_clx_pix']

			if all_pred_fpaths is not None:

				jmock_pred = np.load(all_pred_fpaths[i][zidx])
			
				if inst_indiv==1:
					lab_pred = 'IGL (Mirocha)'
				else:
					lab_pred = None

				lb_pred, cross = [jmock_pred[key] for key in ['lb', 'cross']]
				pf_pred = lb_pred*(lb_pred+1)/(2*np.pi)

				if tl_pix_correct:
					cross /= tl_pix


				ax[zidx].plot(lb_pred, pf_pred*cross, color=color_inst[i], linestyle=linestyles_pred[i], alpha=pred_alpha, label=lab_pred)

		
			label = 'CIBER '+str(lam_dict[inst_indiv])+' $\\mu$m $\\times$ '+catname
		
			fieldav_cl_cross = all_fieldav_cl_cross[i][zidx]
			fieldav_clerr_cross = all_fieldav_clerr_cross[i][zidx]

			z1 = zbinedges[zidx+1]
			addstr = str(np.round(z0, 1))+'_z_'+str(np.round(z1, 1))

			posmask = lbmask*(fieldav_cl_cross > 0)
			negmask = lbmask*(fieldav_cl_cross < 0)

			gal_label = str(np.round(z0, 1))+'$<z_{phot}<$'+str(np.round(z1, 1))

			ax[zidx].errorbar(lb[posmask], (pf*fieldav_cl_cross)[posmask], yerr=(pf*fieldav_clerr_cross)[posmask], color=color_inst[i], fmt='o', \
				capsize=capsize, markersize=markersize, zorder=15, label=label, alpha=alph)
			ax[zidx].errorbar(lb[negmask], np.abs(pf*fieldav_cl_cross)[negmask], yerr=(pf*fieldav_clerr_cross)[negmask], color=color_inst[i], fmt='o', \
				capsize=capsize, markersize=markersize, zorder=15, mfc='white', alpha=alph)

		if zidx==0:
			ax[zidx].legend(bbox_to_anchor=bbox_to_anchor, ncol=ncol_legend, fontsize=legend_fs)
		ax[zidx].text(textxpos, textypos, gal_label, fontsize=text_fs)
		
		# if zidx > 3:
		if zidx > ncols-1: # second row

			ax[zidx].set_xlabel('$\\ell$', fontsize=12)
		ax[zidx].set_ylim(ylim)
		ax[zidx].set_xlim(xlim)
		
		if zidx in [0, ncols]:
			ax[zidx].set_ylabel('$D_{\\ell}^{Ig}$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=12)
		ax[zidx].grid(alpha=0.3)
		ax[zidx].set_xscale('log')
		ax[zidx].set_yscale('log')
	plt.subplots_adjust(wspace=0, hspace=0)
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
		ciber_auto = np.load('data/ciber_auto_'+bandstr_list[idx]+'lt16.0_F25B.npz')

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
				lab_pred = 'IGL (Mirocha)'
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
		lab_pred = 'IGL (Mirocha)'
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