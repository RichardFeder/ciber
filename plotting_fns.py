import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits
# from mpl_toolkits.axes_grid.inset_locator import inset_axes
import numpy as np
from matplotlib import cm
# from PIL import Image
import glob
import config
import matplotlib.patches as mpatches
from ciber_data_file_utils import *
from ps_tests import *
from numerical_routines import *
# from spitzer_auto_cross import *
from powerspec_utils import *
# from mkk_parallel import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_ebl_fluc_spectrum(lb, all_compsub_cl, all_compsub_clerr, lams=[1.1, 1.8], \
							 all_compsub_cl_cross=None, all_compsub_clerr_cross=None, lams_cross=None, \
							 plot=True, figsize=(6, 5), lbins=None, lidx_ciber=None, colors=None, \
							colors_cross=None):
	
	if lbins is None:
		lbins = [[1000, 2000]]
		
	if colors is None:
		colors = ['b', 'r', 'k', 'g']
		
	if colors_cross is None:
		colors_cross = ['magenta' for x in range(len(lams_cross))]
		
		
	lam_mins = [0.9, 1.4, 3.3, 4.0]
	lam_maxs = [1.2, 2.1, 3.9, 5.0]
	
#     plt.axvline(wav, linestyle='solid', color='k')
	
	fig = plt.figure(figsize=figsize)
	
	pf = lb*(lb+1)/(2*np.pi)
	for lbin in lbins:
		
		plt.text(0.7, 6, str(lbin[0])+'$<\\ell<$'+str(lbin[1]), fontsize=16, color='k')
		
		lbmask = (lb > lbin[0])*(lb < lbin[1])
		for x in range(len(lams)):
			
			avdl_sqrt = np.sqrt(np.mean((pf*all_compsub_cl[x])[lbmask]))
			avdlerr = np.sqrt(np.sum((pf*all_compsub_clerr[x])[lbmask]**2)/np.sum(lbmask))/np.sqrt(np.sum(lbmask))
			
			avdlerr_sqrt = avdlerr/(2*avdl_sqrt)
			
			lamcen = lams[x]
			
			if lamcen in [1.1, 1.8]:
				label = 'CIBER '+str(lamcen)+' $\\mu$m'
			else:
				label = 'Spitzer '+str(lamcen)+' $\\mu$m'
			
			if lamcen==1.1:
				lamcen = 1.05
			
			# binned average power
			plt.errorbar([lamcen], [avdl_sqrt],\
					 yerr=[avdlerr], xerr=[[lams[x]-lam_mins[x]], [lam_maxs[x]-lams[x]]], color=colors[x],\
							 marker='o', capsize=5, capthick=1.5, markersize=5, label=label)

			
		if all_compsub_cl_cross is not None:
			
			for x in range(len(lams_cross)):
				
				label = str(lams_cross[x][0])+' $\\mu$m $\\times$ '+str(lams_cross[x][1])+' $\\mu$m'
			
				avdl_sqrt = np.sqrt(np.mean((pf*all_compsub_cl_cross[x])[lbmask]))
				avdlerr = np.sqrt(np.sum((pf*all_compsub_clerr_cross[x])[lbmask]**2)/np.sum(lbmask))/np.sqrt(np.sum(lbmask))
				avdlerr_sqrt = avdlerr/(2*avdl_sqrt)
				lam_cen = 0.5*(lams_cross[x][0]+lams_cross[x][1])
				plt.errorbar([lam_cen], [avdl_sqrt], yerr=[avdlerr_sqrt], label=label, marker='x', capsize=5, markersize=10, capthick=1.5, color=colors_cross[x])

	plt.legend(loc=1, fontsize=10)
	plt.xlabel('$\\lambda$ [$\\mu$m]', fontsize=14)
	plt.ylabel('$\\sqrt{D_{\\ell}}$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=14)
	plt.ylim(1e-1, 1e1)
	plt.yscale('log')
	plt.grid(alpha=0.5)
	plt.tick_params(labelsize=12)
	plt.xlim(0.5, 5.1)
	
	if plot:
		plt.show()
		
	return fig

def plot_ciber_corr_with_monopole(include_auto=True, include_cross=True, zl_levels = None, dgl_levels = None, plot=True, \
								 figsize=(6, 5), cross_run_name=None, datestr='111323', lmin=500, lmax=1000, lbidx=None, \
								 nsim_mock=2000, flatidx=0, observed_run_name=None, Mkk_obj=None, cbps=None):
	
	
	# cbps = CIBER_PS_pipeline()
	if Mkk_obj is None:
		Mkk_obj = Mkk_bare(dimx=1024, dimy=1024, ell_min=180.*(1024./1024.), nbins=25)
	
	ifield_list=[4, 5, 6, 7, 8]
	lamdict = dict({1:1.1, 2:1.8})
	
	auto_colors = ['b', 'r']
	fig = plt.figure(figsize=figsize)
	
	if dgl_levels is not None:
		fglab = 'DGL'
		sblevels = dgl_levels
	elif zl_levels is not None:
		fglab = 'ZL'
		sblevels = zl_levels

	else:
		print('need to choose either ZL or DGL, exiting')
		return None
	
	for plotidx, lbidx in enumerate(np.arange(2, 8)):
		
		plt.subplot(2, 3, plotidx+1)
	
		if include_auto:

			for inst in [1, 2]:

				obs_dict, mock_dict, cov_dict, cl_fpath = gather_fiducial_auto_ps_results_new(inst, nsim_mock=nsim_mock, \
																				 observed_run_name=observed_run_name, \
																				 flatidx=flatidx, cbps=cbps)

				lb = obs_dict['lb']

				if lbidx is not None:
					lbmask = (lb==lb[lbidx])
					pf = lb[lbidx]*(lb[lbidx]+1)/(2*np.pi)

				else:
					lbmask = (lb > lmin)*(lb < lmax)
					pf = 1

				auto_cl_levels, auto_clerr_levels = [np.zeros(len(ifield_list)) for x in range(2)]
				for fieldidx, ifield in enumerate(ifield_list):
					auto_cl = obs_dict['observed_recov_ps'][fieldidx]
					std_recov_mock_ps = 0.5*(np.percentile(mock_dict['all_mock_recov_ps'][:,fieldidx,:], 84, axis=0)-np.percentile(mock_dict['all_mock_recov_ps'][:,fieldidx,:], 16, axis=0))

					auto_cl_lbmask = auto_cl[lbmask]
					std_cl_lbmask = std_recov_mock_ps[lbmask]

					auto_cl_levels[fieldidx] = np.mean(auto_cl_lbmask)
					auto_clerr_levels[fieldidx] = (np.sqrt(np.sum(std_cl_lbmask**2))/np.sum(lbmask))
				
				label = str(lamdict[inst])+' $\\mu$m auto'
				plt.errorbar(sblevels, pf*auto_cl_levels, yerr=pf*auto_clerr_levels, label=label, fmt='o', capsize=3.5, markersize=4, color=auto_colors[inst-1], capthick=1.5)

		if include_cross:


			cross_cl_file = np.load(config.ciber_basepath+'data/input_recovered_ps/'+datestr+'/TM1/'+cross_run_name+'/input_recovered_ps_estFF_simidx0.npz')

			lb = cross_cl_file['lb']
			if lbidx is not None:
				lbmask = (lb==lb[lbidx])
				
				pf = lb[lbidx]*(lb[lbidx]+1)/(2*np.pi)
			else:
				lbmask = (lb > lmin)*(lb < lmax)
				pf = 1.


			cross_cl_levels, cross_clerr_levels = [np.zeros(len(ifield_list)) for x in range(2)]

			for fieldidx, ifield in enumerate(ifield_list):

				cl1d_obs = cross_cl_file['recovered_ps_est_nofluc'][fieldidx]
				dcl1d_obs = cross_cl_file['recovered_dcl'][fieldidx]

				frac_knox_errors = return_frac_knox_error(lb, Mkk_obj.delta_ell)
				knox_err = frac_knox_errors*cl1d_obs

				dcl1d_obs = np.sqrt(dcl1d_obs**2+knox_err**2)

				cross_cl_lbmask = cl1d_obs[lbmask]
				cross_std_cl_lbmask = dcl1d_obs[lbmask]

				cross_cl_levels[fieldidx] = np.mean(cross_cl_lbmask)
				cross_clerr_levels[fieldidx] = np.sqrt(np.sum(cross_std_cl_lbmask**2))/np.sum(lbmask)


			label = '1.1 $\\mu$m $\\times$ 1.8 $\\mu$m'
			plt.errorbar(sblevels, pf*cross_cl_levels, yerr=pf*cross_clerr_levels, label=label, color='purple', fmt='o', capsize=3.5, markersize=4, capthick=1.5)

		if plotidx==0 or plotidx==3:
			plt.ylabel('$D_{\\ell}^{CIBER}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=12)
			
		if plotidx > 2:
			if zl_levels is not None:
				plt.xlabel('$I_{ZL}(\\lambda=1.1$ $\\mu$m)\n[nW m$^{-2}$ sr$^{-1}$]', fontsize=12)
			
			if dgl_levels is not None:
				plt.xticks([0., 0.5, 1.0, 1.5])
				plt.xlabel('$I_{100 \\mu m}$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=12)

		else:
			if zl_levels is not None:
				plt.xticks([300, 400, 500], ['', '', ''])
			else:
				plt.xticks([0., 0.5, 1.0, 1.5], ['', '', '', ''])
		plt.title(str(int(Mkk_obj.binl[lbidx]))+'$<\\ell<$'+str(int(Mkk_obj.binl[lbidx+1])), fontsize=10)
		if zl_levels is not None:
			plt.xlim(250, 550)
		else:
			plt.xlim(0, 1.5)
		plt.grid(alpha=0.5)

	plt.legend(fontsize=12, bbox_to_anchor=[0.7, 2.6], ncol=3)

	if plot:
		plt.show()
	
	return fig

def plot_dgl_measurements_ciber_lit(dgl_meas_dicts, alph=0.7, markersize=7, capsize=4, figsize=(7, 5.5), \
								   xlim=[0.4, 4.5], ylim=[0.5, 100]):
	
	markers = ['d', 'd', 'v', 's', '^', 'o', 'v', 'h']
	colors = ['C3', 'k', 'k', 'k', 'k', 'grey', 'k', 'k']
	alphas = [1.0, 1.0, alph, alph, alph, alph, alph, alph]
	mfc = [None, None, 'white', 'white', None, None, None, None]

	fig, ax = plt.subplots(1, 1, figsize=figsize)
	
	for didx, dmd in enumerate(dgl_meas_dicts):
	
		if len(dmd.lam_width)==0:
			xerr = None
		else:
			xerr = dmd.lam_width

		plt.errorbar(dmd.lam_meas, dmd.dgl_color, yerr=dmd.dgl_color_unc, xerr=xerr, capsize=capsize,\
					 label=dmd.name, fmt='o', marker=markers[didx], alpha=alphas[didx], markersize=markersize, color=colors[didx], mfc=mfc[didx])

	plt.xlabel('$\\lambda$ [$\\mu$m]', fontsize=14)
	plt.ylabel('$\\lambda I_{\\lambda}$ / $I_{\\nu}^{100\\mu m}$ [nW m$^{-2}$ sr$^{-1}$ / MJy sr$^{-1}$]', fontsize=14)
	plt.xlim(xlim)
	plt.xscale('log')

	ax.set_xticks([0.4, 0.8, 1.0, 2.0, 3.0, 4.0], ['0.4', '0.8', '1.0', '2.0', '3.0', '4.0'])
	ax.set_xticks([], minor=True)

	plt.ylim(ylim)
	plt.yscale('log')
	plt.legend(ncol=1, loc=1, fontsize=9)
	plt.tick_params(labelsize=12)
	
	plt.show()
	
	return fig

def plot_ciber_dgl_constraints_largescale(all_best_ps_fit_av, all_AC_A1, all_dAC_sq, cl_basepath=None, figsize=(11, 4), colors=['C3', 'k'], \
                                         inst_list=[1.1, 1.8], text_fs=13, tick_fs=11, textxpos=200, textypos=300, \
                                         xlim=[1.8e2, 2.5e3], ylim=[1e-3, 9e2], inst_x=1, cross_inst_x=2, \
                                         maglim_J=17.5, bbox_to_anchor=[0.5, 1.05]):
    
    all_r_TM, all_sigma_r_TM = [], []
    all_cl_z14, all_lb_z14 = [], []

    
    if cl_basepath is None:
        cl_basepath = config.ciber_basepath+'data/input_recovered_ps/cl_files/'
          
    bandstr_dict = dict({1:'J', 2:'H'})
    maglim_dict = dict({1:17.5, 2:17.0})
    lb_modl, best_ps_fit_av_cross, AC_A1_cross, dAC_sq_TM1, dAC_sq_TM2  = load_clx_dgl()

    fig = plt.figure(figsize=figsize)
    
    for inst in [1, 2]:
    
        bandstr, mag_lim = bandstr_dict[inst], maglim_dict[inst]
        plt.subplot(1,3,inst)

        obs_name = 'observed_'+bandstr+'lt'+str(mag_lim)+'_012524_ukdebias'
        print('obs name:', obs_name)
        cl_fpath_obs = cl_basepath+'TM'+str(inst)+'/cl_'+obs_name+'.npz'

        lb, observed_recov_ps, observed_recov_dcl_perfield,\
        observed_field_average_cl, observed_field_average_dcl,\
            mock_all_field_cl_weights, _ = load_weighted_cl_file(cl_fpath_obs, cltype='cross')

        z14_df = np.array(pd.read_csv(config.ciber_basepath+'data/cl_predictions/lowz_galaxies_z14_TM'+str(inst)+'.csv'))
        lb_z14 = z14_df[:,0].astype(float)
        cl_z14 = z14_df[:,1].astype(float)
        plt.plot(lb_z14, cl_z14**2, color='b', label='Low-z galaxies (Z14)', linestyle='dashed')

        all_lb_z14.append(lb_z14)
        all_cl_z14.append(np.array(cl_z14))

        prefac = lb*(lb+1)/(2*np.pi)
        plt.errorbar(lb, prefac*observed_field_average_cl, yerr=prefac*observed_field_average_dcl, color='k', capsize=3, fmt='o', markersize=4, label='CIBER 4th flight measurements\n(this work)')


        for dgl_idx, dgl_mode in enumerate(['sfd_clean_plus_LSS', 'sfd_clean']):
        
            best_ps_fit_av = all_best_ps_fit_av[inst-1][dgl_idx]
            AC_A1 = all_AC_A1[inst-1][dgl_idx]
            dAC_sq = all_dAC_sq[inst-1][dgl_idx]
            observed_run_name, cross_text_lab = grab_dgl_config(dgl_mode, addstr='_I100')


            plt.plot(lb_modl, best_ps_fit_av*AC_A1**2, color=colors[dgl_idx], label='DGL prediction, '+cross_text_lab, linestyle='solid')
            plt.fill_between(lb_modl, best_ps_fit_av*(AC_A1**2-dAC_sq), best_ps_fit_av*(AC_A1**2+dAC_sq), color=colors[dgl_idx], alpha=0.3)
            plt.fill_between(lb_modl, best_ps_fit_av*(AC_A1**2-2*dAC_sq), best_ps_fit_av*(AC_A1**2+2*dAC_sq), color=colors[dgl_idx], alpha=0.1)

        if inst==1:
            plt.legend(loc=3, ncol=2, bbox_to_anchor=bbox_to_anchor, fontsize=12)

        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(ylim)
        plt.grid(alpha=0.5)
        plt.tick_params(labelsize=tick_fs)
        plt.xlim(xlim)
        plt.xlabel('$\\ell$', fontsize=16)

        plt.text(textxpos, textypos, 'CIBER '+str(inst_list[inst-1])+' $\\mu$m $\\times$ '+str(inst_list[inst-1])+' $\\mu$m', fontsize=text_fs)
        if inst==1:
            plt.ylabel('$D_{\\ell}$ [(nW m$^{-2}$ sr$^{-1}$)$^2$]', fontsize=16)

            
    plt.subplot(1,3,3)
    
    obs_name = 'ciber_cross_ciber_Jlt'+str(maglim_J)+'_Hlt'+str(maglim_J-0.5)+'_020724_interporder2'
    cl_fpath_obs = cl_basepath+'TM'+str(inst_x)+'_TM'+str(cross_inst_x)+'_cross/cl_'+obs_name+'.npz'

    lb, observed_recov_ps, observed_recov_dcl_perfield,\
    observed_field_average_cl, observed_field_average_dcl,\
        mock_all_field_cl_weights, _ = load_weighted_cl_file(cl_fpath_obs, cltype='cross')

    plt.plot(lb_z14, cl_z14**2, color='b', label='Low-z galaxies (Z14)', linestyle='dashed')
    plt.errorbar(lb, prefac*observed_field_average_cl, yerr=prefac*observed_field_average_dcl, color='k', capsize=3, fmt='o', markersize=4, label='CIBER cross spectrum')
    plt.text(textxpos, textypos, 'CIBER '+str(inst_list[0])+' $\\mu$m $\\times$ '+str(inst_list[1])+' $\\mu$m', fontsize=text_fs)

    for dgl_idx, dgl_mode in enumerate(['sfd_clean_plus_LSS', 'sfd_clean']):
        lb_modl, best_ps_fit_av_cross, AC_A1_cross, dAC_sq_TM1, dAC_sq_TM2  = load_clx_dgl(dgl_mode=dgl_mode)

        plt.plot(lb_modl, best_ps_fit_av_cross, color=colors[dgl_idx], label='DGL (CSFD)', linestyle='solid')
        plt.fill_between(lb_modl, best_ps_fit_av_cross*(1-0.5*(dAC_sq_TM1+dAC_sq_TM2)/AC_A1_cross), best_ps_fit_av_cross*(1+0.5*(dAC_sq_TM1+dAC_sq_TM2)/AC_A1_cross), color=colors[dgl_idx], alpha=0.2)
        plt.fill_between(lb_modl, best_ps_fit_av_cross*(1-(dAC_sq_TM1+dAC_sq_TM2)/AC_A1_cross), best_ps_fit_av_cross*(1+(dAC_sq_TM1+dAC_sq_TM2)/AC_A1_cross), color=colors[dgl_idx], alpha=0.1)

    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(ylim)
    plt.grid(alpha=0.5)
    plt.tick_params(labelsize=tick_fs)
    plt.xlim(xlim)
    plt.xlabel('$\\ell$', fontsize=16)
    
    plt.show()
    
    return fig

def plot_mixing_matrices_paper(inst, band, masking_maglim=17.5, ifield=4, \
							   vmin = 1e-5, vmax = 3, cmap='jet', linthresh=1e-4, figsize=(9, 6)):

	mask_tail = 'maglim_'+band+'_Vega_'+str(masking_maglim)+'_111323_ukdebias'

	mc_bdir_maskonly = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/mkk/'
	mc_bdir_hybrid = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/mkk_ffest/'

	mkk_fpath_maskonly = mc_bdir_maskonly+'/'+mask_tail+'/mkk_maskonly_estimate_ifield'+str(ifield)+'_observed_'+mask_tail+'.fits'
	av_mkk_maskonly = fits.open(mkk_fpath_maskonly)['Mkk_'+str(ifield)].data

	mkk_fpath_ff = mc_bdir_hybrid+'/'+mask_tail+'/mkk_ffest_nograd_ifield'+str(ifield)+'_observed_'+mask_tail+'.fits'
	av_mkk_ff = fits.open(mkk_fpath_ff)['Mkk_'+str(ifield)].data

	mkk_fpath_ff_filt = mc_bdir_hybrid+'/'+mask_tail+'/mkk_ffest_quadoff_grad_ifield'+str(ifield)+'_observed_'+mask_tail+'.fits'
	av_mkk_ff_filt = fits.open(mkk_fpath_ff_filt)['Mkk_'+str(ifield)].data

	fig = plt.figure(figsize=figsize)
	
	plt.subplot(2,3,1)
	plt.title('$M_{\\ell\\ell^{\\prime}}^{mask}$', fontsize=14)
	plt.imshow(av_mkk_maskonly, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=cmap, origin='lower')
	plt.colorbar(fraction=0.046, pad=0.04)
	plt.ylabel('Bandpower $b^{\\prime}$', fontsize=12)
	plt.xlabel('Bandpower $b$', fontsize=12)
	plt.yticks([], [])
	plt.xticks([], [])

	plt.subplot(2,3,2)
	plt.imshow(av_mkk_ff, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=cmap, origin='lower')
	plt.colorbar(fraction=0.046, pad=0.04)
	plt.yticks([], [])
	plt.xticks([], [])
	plt.ylabel('Bandpower $b^{\\prime}$', fontsize=12)
	plt.xlabel('Bandpower $b$', fontsize=12)
	plt.title('$M_{\\ell\\ell^{\\prime}}^{mask+FF}$', fontsize=14)

	plt.subplot(2,3,3)
	plt.imshow(av_mkk_ff_filt, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=cmap, origin='lower')
	plt.colorbar(fraction=0.046, pad=0.04)
	plt.yticks([], [])
	plt.xticks([], [])
	plt.ylabel('Bandpower $b^{\\prime}$', fontsize=12)
	plt.xlabel('Bandpower $b$', fontsize=12)
	plt.title('$M_{\\ell\\ell^{\\prime}}^{mask+FF+filt}$', fontsize=14)

	plt.subplot(2,3,5)
	plt.title('$M_{\\ell\\ell^{\\prime}}^{mask+FF}/M_{\\ell\\ell^{\\prime}}^{mask}$', fontsize=14)
	plt.imshow(av_mkk_ff/av_mkk_maskonly, cmap='viridis', origin='lower', vmin=0.5, vmax=5)
	plt.colorbar(fraction=0.046, pad=0.04)
	plt.ylabel('Bandpower $b^{\\prime}$', fontsize=12)
	plt.xlabel('Bandpower $b$', fontsize=12)
	plt.yticks([], [])
	plt.xticks([], [])

	plt.subplot(2,3,6)
	plt.title('$M_{\\ell\\ell^{\\prime}}^{mask+FF+filt}/M_{\\ell\\ell^{\\prime}}^{mask}$', fontsize=14)
	plt.imshow(av_mkk_ff_filt/av_mkk_maskonly, cmap='viridis', origin='lower', vmin=0.5, vmax=5)
	plt.colorbar(fraction=0.046, pad=0.04)
	plt.ylabel('Bandpower $b^{\\prime}$', fontsize=12)
	plt.xlabel('Bandpower $b$', fontsize=12)
	plt.yticks([], [])
	plt.xticks([], [])


	plt.tight_layout()
	plt.subplots_adjust(hspace=0)
	plt.show()
	
	return fig


def plot_ciber_b_ell(cbps, ifield_list = [4, 5, 6, 7, 8], startidx=0, endidx=-1, labs = ['CIBER 1.1 $\\mu$m', 'CIBER 1.8 $\\mu$m'], \
					 figsize=(8, 3.5), text_fs=14):
	

	lb = cbps.Mkk_obj.midbin_ell

	fig = plt.figure(figsize=figsize)

	for inst in [1, 2]:
		plt.subplot(1,2,inst)

		B_ells_post = np.load(config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/beam_correction/bl_est_postage_stamps_TM'+str(inst)+'_081121.npz')['B_ells_post']

		for fieldidx, ifield in enumerate(ifield_list):
			plt.plot(lb[startidx:endidx], B_ells_post[fieldidx][startidx:endidx]**2, marker='.', color='C'+str(fieldidx), label=cbps.ciber_field_dict[ifield])

		plt.yscale('log')
		plt.xscale('log')
		if inst==1:
			plt.legend(fontsize=11, bbox_to_anchor=[1.7, 1.3], ncol=3)
		plt.tick_params(labelsize=12)
		plt.xlabel('$\\ell$', fontsize=16)
		if inst==1:
			plt.ylabel('$B_{\\ell}^2$', fontsize=16)
		plt.grid(alpha=0.3, color='grey')
		plt.text(200, 1.25, labs[inst-1], fontsize=text_fs)
		plt.ylim(0.05, 2.05)
		plt.xlim(150, 1e5)
		
		if inst==2:
			plt.yticks([1e-1, 1e0], ['', ''])
			
	plt.tight_layout()
	plt.subplots_adjust(wspace=0.05)
	plt.show()
	
	return fig

def plot_ciber_corrcoeff_vs_mag(corrfile_fpath=None, lbins=[100, 1000, 10000, 100000], \
							   colors = ['k', 'r', 'b', 'g'], figsize=(5, 4)):
	
	if corrfile_fpath is None:
		corrfile_fpath = config.ciber_basepath+'data/input_recovered_ps/111323/TM1_TM2_cross/cross_corrcoeff_vs_magnitude.npz'
	
	corrfile = np.load(corrfile_fpath)

	lb = corrfile['lb']
	all_r_TM = corrfile['all_r_TM']
	all_sigma_r_TM = corrfile['all_sigma_r_TM']
	maglim_J_list = corrfile['maglim_J_list']
	
	all_r_TM = np.array(all_r_TM)
	all_sigma_r_TM = np.array(all_sigma_r_TM)
	
	fig = plt.figure(figsize=figsize)

	for lidx in range(len(lbins)-1):
		lmin, lmax = lbins[lidx], lbins[lidx+1]
		lab = str(lmin)+'$<\\ell<$'+str(lmax)
		lbmask = (lb > lmin)*(lb < lmax)

		r_TM_sel = all_r_TM[:, lbmask]
		sigma_r_TM_sel = all_sigma_r_TM[:, lbmask]

		weights = 1./sigma_r_TM_sel**2
		sumweights = np.sum(weights, axis=1)

		weighted_av_r = np.average(r_TM_sel, axis=1, weights=weights)
		weighted_std_r = 1./np.sqrt(sumweights)
		av_r_vs_mag = np.mean(r_TM_sel, axis=1)
		sumvar = np.sum(sigma_r_TM_sel**2, axis=1)
		av_sigma_r_vs_mag = np.sqrt(sumvar)/len(lb[lbmask])

		print('av sigma r vs mag', av_sigma_r_vs_mag)
		print('weighted_std_r ', weighted_std_r)

		plt.errorbar(maglim_J_list, weighted_av_r, yerr=weighted_std_r, alpha=0.8, fmt='o', label=lab, capsize=5, capthick=2., color=colors[lidx])

	plt.legend(fontsize=12)
	plt.ylabel('$\\langle r_{\\ell}^{1.1 \\times 1.8} \\rangle$', fontsize=14)
	plt.xlabel('Masking depth (J$_{Vega}$)', fontsize=14)
	plt.grid(alpha=0.5)
	plt.ylim(0.0, 1.)
	plt.xlim(11.5, 19.0)


	plt.show()
	
	return fig

def plot_cross_halfexpdiff(cbps, lb, all_clproc, all_clprocerr, ifield_list=[4, 5, 6, 7, 8], ymax=13, \
						  figsize=(6,5), startidx=0, endidx=-1):
	
	fig = plt.figure(figsize=figsize)
	
	prefac = lb*(lb+1)/(2*np.pi)
	for fieldidx, ifield in enumerate(ifield_list):
		
		plt.subplot(3, 2, fieldidx+1)
		
		plt.errorbar(lb[startidx:endidx], (prefac*all_clproc[fieldidx])[startidx:endidx], yerr=(prefac*all_clprocerr[fieldidx])[startidx:endidx], color='C'+str(fieldidx),\
					 fmt='o', capsize=3, markersize=4, label=str(cbps.ciber_field_dict[ifield]), zorder=10)
		plt.axhline(0, linestyle='dashed', color='k', alpha=0.5)
		plt.xscale('log')

		if type(ymax)==list:
			plt.ylim(-1.2*ymax[fieldidx], 1.2*ymax[fieldidx])
		else:
			plt.ylim(-1.2*ymax, 1.2*ymax)

		if ifield==8 or ifield==7:
			plt.xlabel('Multipole $\\ell$', fontsize=14)
		
		
		if ifield in [4, 5, 6]:
			plt.xticks([1e3, 1e4, 1e5], ['', '', ''])
		
		if ifield==6:
			plt.ylabel('$\\frac{\\ell(\\ell+1)N_{\\ell}^{1.1\\times 1.8}}{2\\pi}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=14)
		plt.tick_params(labelsize=11)
		plt.text(200, 0.7*ymax[fieldidx], cbps.ciber_field_dict[ifield], fontsize=14)
		plt.grid(alpha=0.3)
	plt.tight_layout()
	plt.subplots_adjust(hspace=0.1, wspace=0.12)
	plt.show()
	
	return fig


def plot_1d_transfer_functions(figsize=(5,4), xlim=[150, 1.5e5], ylim=[0., 1.5]):
	
	# plot gradient filter, per quadrant offsets + gradient filter, per quadrant offsets + fourier component model
	
	
	# grad filter only
	t_ell_whole_file = np.load(config.ciber_basepath+'data/transfer_function/t_ell_est_nsims=100.npz')
	lb, t_ell_av_g, t_ell_stderr_g  = t_ell_whole_file['lb'], t_ell_whole_file['t_ell_av'], t_ell_whole_file['t_ell_stderr']

	# grad filter + quadrant offsets
	t_ell_fpath_qog = config.ciber_basepath+'data/transfer_function/t_ell_quadoff_grad_nsims=500_n=3p0.npz'
	tell_qog = np.load(t_ell_fpath_qog)
	lb, t_ell_av_qog, t_ell_stderr_qog  = tell_qog['lb'], tell_qog['t_ell_av'], tell_qog['t_ell_stderr']
	
	# Fourier component model + quadrant offsets
	
	t_ell_fpath_qofc = config.ciber_basepath+'data/transfer_function/t_ell_fcsub_nterms=2_nsims=500_n=3p0.npz'
	tell_qofc = np.load(t_ell_fpath_qofc)
	lb, t_ell_av_qofc, t_ell_stderr_qofc  = tell_qog['lb'], tell_qofc['t_ell_av'], tell_qofc['t_ell_stderr']

	
	fig = plt.figure(figsize=figsize)
	plt.errorbar(lb, t_ell_av_g, yerr=t_ell_stderr_g, label='Gradient filter over full array', color='k', fmt='o-', markersize=3, capsize=2)
	plt.errorbar(lb, t_ell_av_qog, yerr=t_ell_stderr_qog, label='Per-quadrant offsets\n+Gradient filter over full array', color='r', fmt='o-', markersize=3, capsize=2)
	plt.errorbar(lb, t_ell_av_qofc, yerr=t_ell_stderr_qofc, capsize=3, marker='.', color='C2', label='Per-quadrant offsets\n+2nd order Fourier component filter')

	plt.legend(loc=4)
	plt.xscale('log')
	plt.xlabel('$\\ell$', fontsize=14)
	plt.ylabel('$T_{\\ell}$', fontsize=14)
	plt.grid(alpha=0.5)
	plt.xlim(xlim)
	plt.ylim(ylim)
	plt.tick_params(labelsize=12)
	plt.show()
	
	return fig

def plot_filter_curves(all_filter_lam, all_filter_T, figsize=(9, 3), xlim=[0.3, 6.0], \
					  ylim=[0, 1.1]):
	
	filter_colors = ['C'+str(x) for x in range(len(all_filter_lam))]

	filter_colors = ['b', 'r', 'r', 'cyan', 'C5', 'C6', 'C7', 'C8', 'C4', 'C3', 'C1', 'lightcoral', 'firebrick']
	filter_labs = ['CIBER 1.1 $\\mu$m', 'CIBER 1.8 $\\mu$m (Flights 1/2)', 'CIBER 1.8 $\\mu$m (Flights 3/4)', 'g (PS)', 'r (PS)', 'i (PS)', 'z (PS)', 'Y (PS)', 'J (2MASS)', 'H (2MASS)', 'K$_s$ (2MASS)', 'IRAC CH1', 'IRAC CH2']

	fig = plt.figure(figsize=figsize)

	for x in range(len(all_filter_T)):

		if x > 2:
			alpha = 0.5
			lw = 1.5
			linestyle='solid'
		elif x==1:
			linestyle='dashed'
			alpha = 1.0
		else:
			alpha = 1.0
			lw = 2
			linestyle='solid'

		plt.plot(all_filter_lam[x], all_filter_T[x], linewidth=lw, color=filter_colors[x], label=filter_labs[x], alpha=alpha, linestyle=linestyle)

	plt.xlabel('$\\lambda$ [$\\mu$m]', fontsize=14)
	plt.ylabel('$\\tilde{T}(\\lambda)$', fontsize=14)
	plt.xlim(xlim)

	plt.ylim(ylim)
	plt.xscale('log')

	plt.tick_params(labelsize=12)

	plt.legend(bbox_to_anchor=[-0.01, 1.5], ncol=4, loc=2, fontsize=11)
	plt.grid(alpha=0.2, which='both')
#     plt.savefig('figures/ciber_2MASS_IRAC_PS_filters_051524.pdf', bbox_inches='tight')
	plt.show()
	
	return fig


def plot_noise_model_and_sample(ifield=4, noise_model_tail='full', axlabsize=14, ticklabsize=13, textfs=14, \
                               textcolor='k', textx=50, texty=900, figsize=(9,8), ):

    all_noisemodl, all_expdiff = [], []
    
    lams = [1.1, 1.8]
    
    for inst in [1, 2]:
        noise_model_dir = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/noise_model/'
        noisemodl = fits.open(noise_model_dir+'noise_model_TM'+str(inst)+'_ifield'+str(ifield)+'_'+noise_model_tail+'.fits')['noise_model_'+str(ifield)].data
        expdiff = np.load(noise_model_dir+'dark_exp_diffs/dark_exp_diff_TM'+str(inst)+'_ifield'+str(ifield)+'_'+noise_model_tail+'_051324.npz')['expdiff']
        
        all_noisemodl.append(noisemodl)
        all_expdiff.append(expdiff)
        
    textbbox = dict(facecolor='white', alpha=0.8, edgecolor='None')
    
    
    fig = plt.figure(figsize=figsize)
    
    for inst in [1, 2]:

        plt.subplot(2,2,1+2*(inst-1))
        plt.imshow(all_expdiff[inst-1], origin='lower', vmax=0.3, vmin=-0.3)
        cbar = plt.colorbar(pad=0.04, fraction=0.046)
        cbar.ax.set_title('[e-/s]', fontsize=14)
        cbar.ax.tick_params(labelsize=ticklabsize)
        plt.tick_params(labelsize=ticklabsize)
        plt.xlabel('x [pix]', fontsize=axlabsize)
        plt.ylabel('y [pix]', fontsize=axlabsize)
        plt.text(textx, texty, 'TM'+str(inst)+' ('+str(lams[inst-1])+' $\\mu$m)', fontsize=textfs, color=textcolor, bbox=textbbox)

        plt.subplot(2,2,2+2*(inst-1))
        plt.imshow(all_noisemodl[inst-1], norm=matplotlib.colors.LogNorm(vmin=2e-12, vmax=1e-9), origin='lower')
        cbar = plt.colorbar(pad=0.04, fraction=0.046)
        cbar.ax.set_title('[(e-/s)$^2$]', fontsize=14)
        cbar.ax.tick_params(labelsize=ticklabsize)
        plt.xlabel('$\\ell_x$ [$\\times 10^{-5}$]', fontsize=axlabsize)
        plt.ylabel('$\\ell_y$ [$\\times 10^{-5}$]', fontsize=axlabsize)
        plt.text(textx, texty, 'TM'+str(inst)+' ('+str(lams[inst-1])+' $\\mu$m)', fontsize=textfs, color=textcolor, bbox=textbbox)
        plt.xticks([0, 256, 512, 512+256, 1024], [-2, -1, 0, 1, 2])
        plt.yticks([0, 256, 512, 512+256, 1024], [-2, -1, 0, 1, 2])
        plt.tick_params(labelsize=ticklabsize)
        
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_difference_spectra_against_auto(inst, maglim, lb, processed_diff, cl_diff_err, mean_indiv, std_indiv, startidx=0, endidx=-1):

	pf = lb*(lb+1)/(2*np.pi)
	fig = plt.figure(figsize=(6,5))
	
	lbmask = (lb >= lb[startidx])*(lb < lb[endidx])
	
	negmask = lbmask*(processed_diff < 0)
	posmask = lbmask*(processed_diff > 0)
	plt.errorbar(lb[posmask], (pf*processed_diff/2.)[posmask], yerr=(pf*cl_diff_err/2.)[posmask], fmt='o', capsize=3, markersize=4, capthick=1.5, linewidth=2., zorder=10, color='k', label='Field difference\n(Bootes A - Bootes B)')
	plt.errorbar(lb[negmask], np.abs((pf*processed_diff/2.)[negmask]), yerr=(pf*cl_diff_err/2.)[negmask], fmt='o', mfc='white', capsize=3, markersize=4, capthick=1.5, linewidth=2., zorder=10, color='k')

	if mean_indiv is not None:
		plt.errorbar(lb[startidx:endidx], (pf*mean_indiv)[startidx:endidx], yerr=(pf*std_indiv)[startidx:endidx], fmt='o', color='C3', capsize=3, markersize=4, capthick=1.5, linewidth=2., label='Bootes field average')

	lamdict = dict({1:1.1, 2:1.8})
	bandstr_dict = dict({1:'J', 2:'H'})
	bandstr = bandstr_dict[inst]

	plt.text(200, 7e2, 'CIBER '+str(lamdict[inst])+' $\\mu$m\nObserved data\nMask $'+bandstr+'<'+str(maglim)+'$', fontsize=16)

	plt.xscale('log')
	plt.yscale('log')
	plt.ylabel('$D_{\\ell}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=14)
	plt.xlabel('$\\ell$', fontsize=14)
	plt.legend(loc=4, fontsize=12)
	plt.ylim(1e-1, 1e4)
	plt.grid(alpha=0.5)
	plt.tick_params(labelsize=12)
	# plt.savefig('/Users/richardfeder/Downloads/ciber_diff_auto_compare_BootesAB_TM'+str(inst)+'_maglim'+str(maglim)+'_perquadsub.png', bbox_inches='tight', dpi=300)
	plt.show()

	return fig

def plot_ciber_field_consistency_test(inst, lb, observed_recov_ps, observed_field_average_cl, all_mock_recov_ps, all_mock_signal_ps, mock_all_field_cl_weights, \
									  lmax=10000, startidx=1, endidx=None, mode='chi2', \
									ybound=5, mock_run_name=None, datestr='111323', ifield_list = [4, 5, 6, 7, 8], \
										 mod_ratio=4, rescale_mode='diag', mod_ell_max=2000, xlim=[150, 1.1e5], \
												 textxpos = 200, textypos = None):
	
	ciber_field_dict = dict({4:'elat10', 5:'elat30', 6:'Bootes B', 7:'Bootes A', 8:'SWIRE'})

	all_chistat = []
	
	lamdict = dict({1:1.1, 2:1.8})
	lbmask_chistat = (lb < lmax)*(lb >= lb[startidx])

	if mode=='chi2':
		observed_chi2_red = []
	
	prefac = lb*(lb+1)/(2*np.pi)
	
	if mock_run_name is not None:
		
		cov_fpath = config.ciber_basepath+'data/ciber_mocks/'+datestr+'/TM'+str(inst)+'/covariance/'+mock_run_name+'/ciber_field_consistency_cov_matrices_TM'+str(inst)+'_'+mock_run_name+'.npz'

		cov_dat = np.load(cov_fpath)
		all_resid_data_matrices = cov_dat['all_resid_data_matrices']
		resid_joint_data_matrix = cov_dat['resid_joint_data_matrix']
		cov_joint = np.cov(resid_joint_data_matrix.transpose())
		
		corr_joint = np.corrcoef(resid_joint_data_matrix.transpose())
		
		all_cov_indiv_full = [np.cov(all_resid_data_matrices[fieldidx,:,lbmask_chistat]) for fieldidx in range(len(ifield_list))]
		all_cov_indiv_full = np.array(all_cov_indiv_full)
		
		all_corr_indiv_full = [np.corrcoef(all_resid_data_matrices[fieldidx,:,lbmask_chistat]) for fieldidx in range(len(ifield_list))]
		all_corr_indiv_full = np.array(all_corr_indiv_full)
		
	else:
		all_cov_indiv_full = None
		cov_joint = None
		
#     plot_map(corr_joint)
#     for x in range(len(all_corr_indiv_full)):
#         plot_map(all_corr_indiv_full[x], title=x)
	
				
	all_inv_cov_indiv_lbmask, all_inv_cov_indiv_mod = [np.zeros_like(all_cov_indiv_full) for x in range(2)]
	ratio_obs_mock_full = None
	if all_cov_indiv_full is not None:
		
		all_cov_indiv_mod = all_cov_indiv_full.copy()

		mean_input_mock_ps = np.mean(all_mock_signal_ps[:,0], axis=0)
		ratio_obs_mock_full = observed_field_average_cl/mean_input_mock_ps
		gtr_unity_mask_full = (np.abs(ratio_obs_mock_full) > mod_ratio)*lbmask_chistat
		gtr_unity_mask_full *= (lb < mod_ell_max)
		ratio_obs_mock_full *= gtr_unity_mask_full

		for fieldidx, ifield in enumerate(ifield_list):
			
			mean_cl_obs = observed_recov_ps[fieldidx]
			std_recov_mock_ps = np.std(all_mock_recov_ps[:,fieldidx], axis=0)
			mean_input_mock_ps = np.mean(all_mock_signal_ps[:,fieldidx], axis=0)
			ratio_obs_mock = (observed_field_average_cl/mean_input_mock_ps)[lbmask_chistat]
		
			gtr_unity_mask = (np.abs(ratio_obs_mock) > mod_ratio)
			
			gtr_unity_mask *= (lb[lbmask_chistat] < mod_ell_max)

			which_gtr_unity = np.where(gtr_unity_mask)[0]
			
			for which in which_gtr_unity:
				if rescale_mode=='diag':

					all_cov_indiv_mod[fieldidx, which, which] *= ratio_obs_mock[which]
				else:
					all_cov_indiv_mod[fieldidx, which, :] *= ratio_obs_mock[which]
					all_cov_indiv_mod[fieldidx, :, which] *= ratio_obs_mock[which]
			
			print('condition number on indiv is ', np.linalg.cond(all_cov_indiv_full[fieldidx]))
			all_inv_cov_indiv_lbmask[fieldidx] = np.linalg.inv(all_cov_indiv_full[fieldidx])
			all_inv_cov_indiv_mod[fieldidx] = np.linalg.inv(all_cov_indiv_mod[fieldidx])
			
	chistat_perfield_mock, pte_perfield_mock, all_chistat_largescale = mock_consistency_chistat(lb, all_mock_recov_ps, mock_all_field_cl_weights, lmax=lmax,\
																								mode=mode, all_cov_indiv_full=all_cov_indiv_full, cov_joint=None, startidx=startidx)
	f = plt.figure(figsize=(10, 7))

	for fieldidx, ifield in enumerate(ifield_list):
		ax = f.add_subplot(2,3,fieldidx+1)
			
		mean_cl_obs = observed_recov_ps[fieldidx]
		std_recov_mock_ps = np.std(all_mock_recov_ps[:,fieldidx], axis=0)
		mean_input_mock_ps = np.mean(all_mock_signal_ps[:,fieldidx], axis=0)
		ratio_obs_mock = (observed_field_average_cl/mean_input_mock_ps)
		
		which_gtr_unity = (np.abs(ratio_obs_mock) > mod_ratio)
		# which_gtr_unity *= (lb < 2000)
		
		std_recov_mock_ps[which_gtr_unity] *= np.sqrt(ratio_obs_mock[which_gtr_unity])
		
		print('ratio obs mock', ratio_obs_mock)
		
		tot_std_ps = np.sqrt(std_recov_mock_ps**2)
		plt.axhline(0, linestyle='dashed', color='k', linewidth=2)
		# plt.axvline(lmax, color='k', linestyle='dashed', linewidth=1.5)

		frac_cl_field = ((mean_cl_obs-observed_field_average_cl)/observed_field_average_cl)[startidx:endidx]
		plt.errorbar(lb[startidx:endidx], frac_cl_field, yerr=(tot_std_ps/observed_field_average_cl)[startidx:endidx],\
					 label=ciber_field_dict[ifield], fmt='.', color='C'+str(fieldidx), zorder=2, alpha=1.0,\
					 capthick=2, capsize=3, linewidth=2, markersize=5)

		resid = observed_field_average_cl-mean_cl_obs

		if mode=='chi2':
			if all_cov_indiv_full is not None:
				chistat_mean_cl_mockstd = np.multiply(resid[lbmask_chistat].transpose(), np.dot(all_inv_cov_indiv_mod[fieldidx], resid[lbmask_chistat]))
			else:
				chistat_mean_cl_mockstd = resid**2/(tot_std_ps**2)
		elif mode=='chi':
			chistat_mean_cl_mockstd = resid/tot_std_ps
		
		all_chistat.append(chistat_mean_cl_mockstd)

		if all_cov_indiv_full is not None:
			chistat_largescale = np.array(all_chistat[fieldidx])
		else:
			chistat_largescale = np.array(all_chistat[fieldidx])[lbmask_chistat]
			
		ndof = len(lb[lbmask_chistat])
		if mode=='chi2':
			chi2_red = np.sum(chistat_largescale)/ndof
			observed_chi2_red.append(chi2_red)
			chistat_info_string = '$\\chi^2/N_{dof}$: '+str(np.round(np.sum(chistat_largescale), 1))+'/'+str(ndof)+' ('+str(np.round(chi2_red, 2))+')'
		elif mode=='chi':
			chistat_info_string = '$\\chi/N_{dof}$: '+str(np.round(np.sum(chistat_largescale), 1))+'/'+str(ndof)
			
			
		# if startidx > 0:
		#     plt.axvline(0.5*(lb[startidx]+lb[startidx-1]), color='k', linestyle='dashed', linewidth=1.5)
		# else:
		#     plt.axvline(0.7*lb[0], color='k', linestyle='dashed', linewidth=1.5)

		plt.axvspan(lmax, 1.1e5, color='grey', alpha=0.2)

		bbox_dict = dict({'facecolor':'white', 'alpha':0.8, 'edgecolor':'None', 'pad':0.})

		if textypos is None:
			textypos = ybound*0.5

		plt.text(textxpos, textypos, ciber_field_dict[ifield]+', '+str(lamdict[inst])+' $\\mu$m\n'+chistat_info_string, color='C'+str(fieldidx), fontsize=11, \
			bbox=bbox_dict)
			
		plt.tick_params(labelsize=12)
		if fieldidx==0 or fieldidx==3:
			plt.ylabel('Fractional deviation $\\Delta C_{\\ell}/\\langle \\hat{C}_{\\ell}\\rangle$', fontsize=12)
		plt.xlabel('$\\ell$', fontsize=16)
		plt.grid(alpha=0.2, color='grey')

		plt.xscale('log')
		plt.ylim(-ybound, ybound)
		plt.xlim(xlim)
		
		
		axin = inset_axes(ax, 
				width="45%", # width = 30% of parent_bbox
				height=0.75, # height : 1 inch
				loc=4, borderpad=1.6)
			
		chistat_mock = all_chistat_largescale[fieldidx]
		chistat_mock /= ndof
		chistat_order_idx = np.argsort(chistat_mock)

		if mode=='chi2':
			bins = np.linspace(0, 4, 30)
		elif mode=='chi':
			bins = np.linspace(-2, 2, 30)
			
		plt.hist(chistat_mock, bins=bins, linewidth=1, histtype='stepfilled', color='k', alpha=0.2, label=ciber_field_dict[ifield])
		plt.axvline(np.median(chistat_mock), color='k', alpha=0.5)

		if mode=='chi2':
			pte = 1.-(np.digitize(observed_chi2_red[fieldidx], chistat_mock[chistat_order_idx])/len(chistat_mock))
			plt.axvline(chi2_red, color='C'+str(fieldidx), linestyle='solid', linewidth=2, label='Observed data')
			axin.set_xlabel('$\\chi^2_{red}$', fontsize=10)

		elif mode=='chi':
			pte = 1.-(np.digitize(np.sum(chistat_largescale)/ndof, chistat_mock[chistat_order_idx])/len(chistat_mock))
			plt.axvline(np.sum(chistat_largescale)/ndof, color='C'+str(fieldidx), linestyle='solid', linewidth=2, label='Observed data')
			axin.set_xlabel('$\\chi_{red}$', fontsize=10)

		axin.xaxis.set_label_position('top')
		
		if mode=='chi2':
			plt.xticks([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
			plt.xlim(0, 4)
			xpos_inset_text = 1.5
		else:
			plt.xticks([-1, 0, 1], [-1, 0, 1])
			plt.xlim(-1.5, 1.5)
			xpos_inset_text = -1.4

		axin.tick_params(labelsize=8,bottom=True, top=True, labelbottom=True, labeltop=False)
		plt.yticks([], [])
		hist = np.histogram(chistat_mock, bins=bins)
		
		plt.text(xpos_inset_text, 0.8*np.max(hist[0]), 'Mocks', color='grey', fontsize=9, bbox=bbox_dict)
		plt.text(xpos_inset_text, 0.6*np.max(hist[0]), 'PTE='+str(np.round(pte, 2)), color='C'+str(fieldidx), fontsize=9, bbox=bbox_dict)
			
	plt.tight_layout()
	plt.show()
	
	return f, all_cov_indiv_full, all_corr_indiv_full, ratio_obs_mock_full


def plot_cov_corr_matrices_mock_perfield(inst, lb, all_mock_recov_ps, ifield_list=[4, 5, 6, 7, 8], mode='corr'):
	
	ciber_field_dict = dict({4:'elat10', 5:'elat30', 6:'Bootes B', 7:'Bootes A', 8:'SWIRE'})
	lamdict = dict({1:1.1, 2:1.8})
	lam = lamdict[inst]
	
	all_cov_indiv, all_corr_indiv = [], []
	
	prefac = lb*(lb+1)/(2*np.pi)
	
	fig = plt.figure(figsize=(10,6))
	
	for fieldidx, ifield in enumerate(ifield_list):
		all_mock_recov_ps_indiv = all_mock_recov_ps[:,fieldidx,:]

		plt.subplot(2,3,fieldidx+1)

		cov_indiv = np.cov(all_mock_recov_ps_indiv.transpose())
		corr_indiv = np.corrcoef(all_mock_recov_ps_indiv.transpose())
		all_cov_indiv.append(cov_indiv)
		all_corr_indiv.append(corr_indiv)
		
		print('condition number of indiv covariance matrix is ', np.linalg.cond(cov_indiv))
		
		if mode=='cov':
			plt.title('Covariance: '+ciber_field_dict[ifield]+', '+str(lam)+' $\\mu$m', fontsize=12)
			plt.imshow(cov_indiv, norm=matplotlib.colors.SymLogNorm(linthresh=1, vmax=1e3), origin='lower')

		elif mode=='corr':
			plt.title('Correlation: '+ciber_field_dict[ifield]+', '+str(lam)+' $\\mu$m', fontsize=12)
			plt.imshow(corr_indiv, vmin=-1, vmax=1, origin='lower', cmap='bwr')
			
		plt.colorbar(fraction=0.046, pad=0.04)
		
		if fieldidx==0 or fieldidx==3:
			plt.ylabel('Bandpower index', fontsize=12)
		if fieldidx > 2:
			plt.xlabel('Bandpower index', fontsize=12)
	plt.tight_layout()
	plt.show()
	
	return fig, all_corr_indiv, all_cov_indiv


def plot_corr_matrix_full(inst, lb, all_mock_recov_ps, ifield_list=[4, 5, 6, 7, 8], title=None, title_fs=16):

	ciber_field_dict = dict({4:'elat10', 5:'elat30', 6:'Bootes B', 7:'Bootes A', 8:'SWIRE'})
	lamdict = dict({1:1.1, 2:1.8})
	lam = lamdict[inst]
	
	prefac = lb*(lb+1)/(2*np.pi)
	
	nmock = all_mock_recov_ps.shape[0]
	
	all_mock_recov_ps_reshape = (prefac*all_mock_recov_ps).reshape((nmock, len(ifield_list)*all_mock_recov_ps.shape[2]))

	cov_full = np.cov(all_mock_recov_ps_reshape.transpose())
	corr_full = np.corrcoef(all_mock_recov_ps_reshape.transpose())
	
	fig = plt.figure(figsize=(7,7))

	if title is not None:
		plt.title(title, fontsize=title_fs)

	plt.imshow(corr_full, vmin=-1, vmax=1, origin='lower', cmap='bwr')
	plt.colorbar(fraction=0.046, pad=0.04)
	xticks = np.array([0., 25., 50., 75., 100., 125.])
	xticks -= 0.5

	plt.xticks(xticks, ['', '', '', '', '', ''])
	plt.yticks(xticks, ['', '', '', '', '', ''])
	
	for tickidx, xtick in enumerate(xticks[:-1]):
		plt.text(0.5*(xticks[tickidx]+xticks[tickidx+1])-8, -5, ciber_field_dict[ifield_list[tickidx]], fontsize=14, color='k')
		plt.text(-5, 0.5*(xticks[tickidx]+xticks[tickidx+1])-6, ciber_field_dict[ifield_list[tickidx]], fontsize=14, color='k', rotation='vertical')

		if tickidx > 0:
			plt.axhline(xtick, linewidth=0.5, color='k')
			plt.axvline(xtick, linewidth=0.5, color='k')

	plt.tight_layout()
	plt.show()
	
	return fig, corr_full, cov_full

def compare_field_average_to_mocks(lb, observed_field_average_cl, observed_field_average_dcl,\
								   mock_all_field_averaged_cls, mock_mean_input_ps, \
								  startidx=0, endidx=-1):

	
	field_average_error = mock_all_field_averaged_cls - mock_mean_input_ps
	field_average_frac_error = (mock_all_field_averaged_cls - mock_mean_input_ps)/mock_mean_input_ps

	print('field average frac error has shape ', field_average_frac_error.shape)
	which_borked = (np.mean(field_average_frac_error, axis=1) > 5)

	print('which is borked :', np.where(which_borked)[0])

	field_average_error = field_average_error[~which_borked,:]
	field_average_frac_error = field_average_frac_error[~which_borked,:]
	mock_mean_input_ps = mock_mean_input_ps[~which_borked,:]
	mock_all_field_averaged_cls = mock_all_field_averaged_cls[~which_borked,:]

	print('new shape ', field_average_error.shape, field_average_frac_error.shape, mock_mean_input_ps.shape)

	median_frac_err = np.median(field_average_frac_error, axis=0)
	std_frac_err = 0.5*(np.percentile(np.abs(field_average_error), 84, axis=0)-np.percentile(np.abs(field_average_error), 16, axis=0))

	frac_err = field_average_error/mock_mean_input_ps

	median_fieldav_err = np.mean(np.abs(field_average_error), axis=0)

	print('mean fieldav err is ', median_fieldav_err)
	prefac = lb*(lb+1)/(2*np.pi)
	
	fig = plt.figure(figsize=(6, 5))
	
	plt.errorbar(lb[startidx:endidx], prefac[startidx:endidx]*observed_field_average_cl[startidx:endidx], yerr=prefac[startidx:endidx]*np.median(np.abs(field_average_error), axis=0)[startidx:endidx], label='Field average (observed)', fmt='o',\
						 capthick=1.5, zorder=10, color='r', capsize=3, markersize=5, linewidth=2.)
	plt.plot(lb, prefac*np.mean(mock_mean_input_ps, axis=0), color='k', linestyle='dashed', zorder=10, label='Mean input mock sky')
	
	plt.plot(lb[startidx:endidx], prefac[startidx:endidx]*observed_field_average_dcl[startidx:endidx], linestyle='solid', color='r', label='Field average uncertainty\n(from field scatter)')
	plt.plot(lb[startidx:endidx], prefac[startidx:endidx]*np.mean(np.abs(field_average_error), axis=0)[startidx:endidx], color='k', linewidth=1.5, label='Mean absolute mock error')

	
	lb_mask_science = (lb > 400)*(lb < 2000)
	snrs_mock = observed_field_average_cl[lb_mask_science]/np.std(mock_all_field_averaged_cls, axis=0)[lb_mask_science]
	snrs_science_sv = observed_field_average_cl[lb_mask_science]/np.mean(np.abs(field_average_error), axis=0)[lb_mask_science]

	snr_tot_mock = np.sqrt(np.sum(snrs_mock**2))
	snr_tot_sv = np.sqrt(np.sum(snrs_science_sv**2))

	print('snr tot with uncertainty from mocks is ', snr_tot_mock)
	print('snr tot with uncertainty from field field scatter is ', snr_tot_sv)

	plt.xscale('log')
	plt.yscale("log")

	plt.ylim(1e-2, 1e4)
	plt.tick_params(labelsize=12)
	plt.xlim(150, 1.2e5)

	plt.xlabel('$\\ell$', fontsize=16)
	plt.ylabel('$D_{\\ell}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=16)

	plt.grid()
	plt.legend(fontsize=10, loc=2, framealpha=0.95)
	plt.tight_layout()

	plt.show()
	
	return fig, median_fieldav_err, median_frac_err, std_frac_err

def plot_mock_recovery_multipanel(inst, lb, all_mock_recov_ps, all_mock_signal_ps, \
								mock_all_field_averaged_cls, masking_maglim=None, ifield_list=[4, 5, 6, 7, 8], \
								plot_field_av=True, mockstr='Mock data', av_mode='median'):
	
	ciber_field_dict = dict({4:'elat10', 5:'elat30', 6:'Bootes B', 7:'Bootes A', 8:'SWIRE'})

	lamdict = dict({1:1.1, 2:1.8})
	bandstr_dict = dict({1:'J', 2:'H'})
	lam = lamdict[inst]
	bandstr = bandstr_dict[inst]
	
	if masking_maglim is None:
		if inst==1:
			masking_maglim = 17.5
		elif inst==2:
			masking_maglim = 17.0
	
	prefac = lb*(lb+1)/(2*np.pi)
	
	fig, ax = plt.subplots(figsize=(6, 7), sharex=True, nrows=2, gridspec_kw={'height_ratios': [1, 0.4]})
	plt.subplots_adjust(hspace=0.06)

	for fieldidx, ifield in enumerate(ifield_list):
		
		std_mock_err = 0.5*(np.percentile(all_mock_recov_ps[:,fieldidx,:], 84, axis=0)-np.percentile(all_mock_recov_ps[:,fieldidx,:], 16, axis=0))
		
		if av_mode=='median':
			ax[0].errorbar(lb, (prefac*np.median(all_mock_recov_ps[:,fieldidx,:], axis=0)), yerr=prefac*std_mock_err, label=ciber_field_dict[ifield], fmt='.', color='C'+str(fieldidx), alpha=0.5, capsize=3, markersize=6)
			ax[1].errorbar(lb, np.median(all_mock_recov_ps[:,fieldidx,:], axis=0)/np.mean(all_mock_signal_ps[:,fieldidx,:], axis=0), yerr=std_mock_err/np.mean(all_mock_signal_ps[:,fieldidx,:], axis=0), label=ciber_field_dict[ifield], fmt='.', color='C'+str(fieldidx), alpha=0.5, capsize=3, markersize=6)
		elif av_mode=='mean':
			ax[0].errorbar(lb, (prefac*np.mean(all_mock_recov_ps[:,fieldidx,:], axis=0)), yerr=prefac*std_mock_err, label=ciber_field_dict[ifield], fmt='.', color='C'+str(fieldidx), alpha=0.5, capsize=3, markersize=6)
			ax[1].errorbar(lb, np.mean(all_mock_recov_ps[:,fieldidx,:], axis=0)/np.mean(all_mock_signal_ps[:,fieldidx,:], axis=0), yerr=std_mock_err/np.mean(all_mock_signal_ps[:,fieldidx,:], axis=0), label=ciber_field_dict[ifield], fmt='.', color='C'+str(fieldidx), alpha=0.5, capsize=3, markersize=6)

	ax[0].plot(lb, prefac*np.mean(all_mock_signal_ps, axis=(0, 1)), color='grey', linewidth=2, linestyle='dashed', label='Input sky')

	
	# print('ratio ', np.mean(mock_all_field_averaged_cls, axis=0)/np.mean(all_mock_signal_ps, axis=(0, 1)))
	
	std_mock_err_av = 0.5*(np.percentile(mock_all_field_averaged_cls, 84, axis=0)-np.percentile(mock_all_field_averaged_cls, 16, axis=0))
	if plot_field_av:

		if av_mode=='median':

			median_frac = np.median(mock_all_field_averaged_cls/np.mean(all_mock_signal_ps, axis=1), axis=0)
			ax[1].errorbar(lb, median_frac, yerr=std_mock_err_av/np.mean(all_mock_signal_ps, axis=(0, 1)), zorder=5, color='b', fmt='o', capsize=3, capthick=2, markersize=4, label='Field average', alpha=0.8)

			ax[1].errorbar(lb, np.median(mock_all_field_averaged_cls, axis=0)/np.mean(all_mock_signal_ps, axis=(0, 1)), yerr=std_mock_err_av/np.mean(all_mock_signal_ps, axis=(0, 1)), zorder=5, color='k', fmt='o', capsize=3, capthick=2, markersize=4, label='Field average', alpha=0.8)
			ax[0].errorbar(lb, prefac*np.median(mock_all_field_averaged_cls, axis=0), yerr=prefac*std_mock_err_av, zorder=6, color='k', fmt='o', capsize=3, capthick=2, markersize=4, label='Field average')
		else:

			mean_frac = np.mean(mock_all_field_averaged_cls/np.mean(all_mock_signal_ps, axis=1), axis=0)
			ax[1].errorbar(lb, mean_frac, yerr=std_mock_err_av/np.mean(all_mock_signal_ps, axis=(0, 1)), zorder=5, color='b', fmt='o', capsize=3, capthick=2, markersize=4, label='Field average', alpha=0.8)

			ax[1].errorbar(lb, np.mean(mock_all_field_averaged_cls, axis=0)/np.mean(all_mock_signal_ps, axis=(0, 1)), yerr=std_mock_err_av/np.mean(all_mock_signal_ps, axis=(0, 1)), zorder=5, color='k', fmt='o', capsize=3, capthick=2, markersize=4, label='Field average', alpha=0.8)
			ax[0].errorbar(lb, prefac*np.mean(mock_all_field_averaged_cls, axis=0), yerr=prefac*std_mock_err_av, zorder=6, color='k', fmt='o', capsize=3, capthick=2, markersize=4, label='Field average')

	ax[0].set_xlim(1.5e2, 1e5)
	ax[0].set_yscale('log')
	ax[0].set_xscale('log')
	ax[0].tick_params(labelsize=14)
	ax[1].set_xlabel('$\\ell$', fontsize=16)
	ax[1].tick_params(labelsize=12)
	ax[1].set_ylabel('$C_{\\ell}^{out}/C_{\\ell}^{in}-1$', fontsize=14)
	ax[1].set_ylim(-1.0, 3.0)

	ax[0].set_ylabel('$D_{\\ell}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=16)
	ax[0].grid(alpha=0.3, color='grey')
	ax[1].grid(color='grey', alpha=0.3)
	ax[1].axhline(1.0, color='grey', alpha=0.3, linestyle='dashed')
	ax[0].set_ylim(1e-2, 5e3)
	yticks = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
	yticks_plot = np.array(yticks)-1.
	ax[1].set_yticks(yticks, yticks_plot)
	
	ax[0].text(2e2, 2e2, 'CIBER '+str(lam)+' $\\mu$m\n'+mockstr+'\nMask $'+bandstr+'<'+str(masking_maglim)+'$', fontsize=16)
	ax[0].legend(fontsize=10, loc=4, ncol=1, facecolor='white').set_zorder(10)

	plt.show()

	return fig

def plot_mock_fieldav_cl_recovery_vs_magnitude(inst, mag_lims, cl_files, include_frac_err=False, startidx=1, endidx=-1, \
								 return_fig=True, colors=None, ylim=None, textstr=None, textxpos=None, textypos=None, addstr=None, dl_l=False):
	inst_to_bandstr = dict({1:'J', 2:'H'})
	lamdict = dict({1:1.1, 2:1.8})
	band = inst_to_bandstr[inst]
	
	if textstr is None:
		
		textstr = 'CIBER '+str(lamdict[inst])+' $\\mu$m\nMock data'
		
	if addstr is not None:
		textstr += '\n'+addstr
	
	power_maglim_obs = []
	
	if colors is None:
		colors = ['C'+str(x) for x in range(len(cl_files))]
		
	if include_frac_err:
		fig, ax = plt.subplots(figsize=(6, 8), sharex=True, nrows=2, gridspec_kw={'height_ratios': [1, 0.3]})
		plt.subplots_adjust(hspace=0.06)

	else:
		fig = plt.figure(figsize=(6, 5))
		
	for m, mag_lim in enumerate(mag_lims):
	
		lb, mock_mean_input_ps, mock_all_field_averaged_cls,\
				mock_all_field_cl_weights, all_mock_recov_ps, all_mock_signal_ps = load_weighted_cl_file(cl_files[m], mode='mock')

		prefac = lb*(lb+1)/(2*np.pi)
		
		if dl_l:
			prefac = lb/(2*np.pi)

		mean_recov_ps_fieldav = np.mean(mock_all_field_averaged_cls, axis=0)
		
		std_recov_ps_fieldav = 0.5*(np.percentile(mock_all_field_averaged_cls, 84, axis=0)-np.percentile(mock_all_field_averaged_cls, 16, axis=0))

		plt.errorbar(lb[:endidx], (prefac*mean_recov_ps_fieldav)[:endidx], yerr=(prefac*std_recov_ps_fieldav)[:endidx], zorder=4-m, label='$'+str(band)+'<$'+str(mag_lim), color='C'+str(m), fmt='o', capsize=3, markersize=4)
	
		plt.plot(lb, (prefac*np.mean(mock_mean_input_ps, axis=0)), color=colors[m], linestyle='dashed', alpha=0.5)


	plt.xscale('log')
	plt.yscale('log')
	
	plt.text(textxpos, textypos, textstr, fontsize=18)
	plt.legend(loc=4, ncol=3, fontsize=12, bbox_to_anchor=[0.98, 1.0])
	plt.grid(alpha=0.3, color='grey')
	plt.xlabel('$\\ell$', fontsize=16)
	if dl_l:
		plt.ylabel('$D_{\\ell}/\\ell$', fontsize=16)
	else:
		plt.ylabel('$D_{\\ell}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=16)
		
	plt.xlim(1.5e2, 1e5)
	plt.tick_params(labelsize=14)
	
	if ylim is not None:
		plt.ylim(ylim[0], ylim[1])
		
			
	plt.show()
	
	if return_fig:
		return fig


def plot_mean_posts_magbins(sopred, bandstr, return_fig=True):
	f = plt.figure(figsize=(10, 5))
	for m in range(len(sopred.mmin_range)):

		plt.subplot(1, len(sopred.mmin_range), m+1)
		plt.imshow(sopred.all_mean_post[m])
		plt.title(bandstr+'$\\in$['+str(sopred.mmin_range[m])+','+str(sopred.mmin_range[m]+sopred.dms[m])+']\n$N_{src}$='+str(int(sopred.all_nsrc[m])), fontsize=12)
		plt.colorbar(pad=0.04, fraction=0.046)

	plt.tight_layout()
	plt.show()
	
	if return_fig:
		return f

def plot_ciber_stack_flux_pred(cbps, inst, mmin_range_Vega, all_stack_mag_AB_eff, allfield_meanpredmag_AB, ifield_list=[4, 5, 6, 7, 8], figsize=(5, 4)):
	
	
	weights_perfield = dict({1:np.array([0.18, 0.08, 0.25, 0.25, 0.25]), \
						2: np.array([0.18, 0.10, 0.23, 0.21, 0.25])})
	
	std_color_corr_dict = dict({1:np.array([0.07766909, 0.0359452,  0.04860506, 0.03304105, 0.03791485]), \
						2:np.array([0.03891492, 0.04843235, 0.06048847, 0.06128753, 0.07497544])})
	
	bandstr = cbps.bandstr_dict[inst]

	allfield_meanpredmag_AB = np.array(allfield_meanpredmag_AB)
	mmin_range_AB = np.array(mmin_range_Vega)+cbps.Vega_to_AB[inst]

	weights = weights_perfield[inst]
	weights /= np.sum(weights)
	
	all_stack_flux_eff = 10**(-0.4*(np.array(all_stack_mag_AB_eff)-23.9))
	for fieldidx, ifield in enumerate(ifield_list):
		all_stack_flux_eff[fieldidx] *= weights[fieldidx]

	mean_stack_flux_eff = np.sum(all_stack_flux_eff, axis=0)
	mean_stack_mag_AB_eff = -2.5*np.log10(mean_stack_flux_eff)+23.9
	
	all_pred_flux_eff = 10**(-0.4*(np.array(allfield_meanpredmag_AB)-23.9))
	mean_pred_flux_eff = np.mean(all_pred_flux_eff, axis=0)
	mean_pred_mag_AB_eff = -2.5*np.log10(mean_pred_flux_eff)+23.9

	fig = plt.figure(figsize=figsize)
	
	for fieldidx, ifield in enumerate(ifield_list):
		plt.scatter(mmin_range_AB, all_stack_mag_AB_eff[fieldidx], color='C'+str(fieldidx), marker='x', label=cbps.ciber_field_dict[ifield], s=80, alpha=0.8)
	
	plt.plot(mmin_range_AB, mean_stack_mag_AB_eff, color='r', marker='.', markersize=10, alpha=0.8, linewidth=2, label='Field average')
	plt.errorbar(mmin_range_AB, mean_pred_mag_AB_eff, yerr=std_color_corr_dict[inst], color='k', fmt='o', marker='*', markersize=10, capsize=5, capthick=1.5, label='Mean prediction')

	xticklabs = ['['+str(np.round(mmin_range_AB[m], 1))+','+str(np.round(mmin_range_AB[m]+0.5, 1))+']' for m in range(len(mmin_range_AB))]    
	plt.xticks(mmin_range_AB, xticklabs)

	plt.xlabel('UVISTA '+bandstr+'-band [AB]', fontsize=12)
	plt.ylabel('Stacked magnitude [AB]', fontsize=12)

	plt.gca().invert_yaxis()
	plt.ylim(mmin_range_AB[-1]+0.8, mmin_range_AB[0]-0.5)
	plt.legend(ncol=2, bbox_to_anchor=[0.9, 1.35])
	plt.grid(alpha=0.5)
	
	plt.show()
	
	return fig


def plot_ciber_spitzer_corrcoeff_vs_ell(lb, all_rlx, all_rlx_unc, irac_list_save, inst_list_save, figsize=(5,4), colors=['b', 'dodgerblue', 'r', 'darkred'], \
									   irac_lams = [3.6, 3.6, 4.5, 4.5], ciber_lams= [1.1, 1.8, 1.1, 1.8], \
									   startidx=1, endidx=-1):
	
	fig = plt.figure(figsize=figsize)

	for x in range(len(irac_list_save)):
		label = str(ciber_lams[x])+' $\\mu$m $\\times$ '+str(irac_lams[x])+' $\\mu$m'
		plt.errorbar(lb[startidx:endidx], all_rlx[x][startidx:endidx], yerr=all_rlx_unc[x][startidx:endidx], capsize=3, fmt='o', color=colors[x], label=label, marker='.')

		if x==0:
			label = 'Shot noise predictions'
		else:
			label = None

		plt.axhline(all_rlx_snpred[x], color=colors[x], linestyle='dashed')
		
	label = '- - - - Shot noise predictions'
	plt.text(1000, 0.9, label, color='k', fontsize=13)
	
	plt.xscale('log')
	plt.ylim(-0.2, 1.0)
	plt.xlim(900, 1.1e5)
	plt.legend(loc=2, ncol=2, fontsize=11, bbox_to_anchor=[-0.03, 1.25])
	plt.xlabel('$\\ell$', fontsize=16)
	plt.ylabel('$r_{\\ell} = C_{\\ell}^{\\times}/\\sqrt{C_{\\ell}^{CIBER}C_{\\ell}^{IRAC}}$', fontsize=14)
	plt.grid()
	plt.tick_params(labelsize=12)

	plt.show()
	
	return fig


def plot_spitzer_auto_cross_compare_mosaics(irac_lams = [3.6, 3.6, 4.5, 4.5], ciber_lams= [1.1, 1.8, 1.1, 1.8],\
										colors = ['b', 'dodgerblue', 'r', 'darkred'], figsize=(8, 5), \
										   startidx=0, endidx=-1):
	
	auto_cross_save_dir = config.ciber_basepath+'data/input_recovered_ps/ciber_spitzer/final_auto_cross/'
	auto_cross_mopex_save_fpath =auto_cross_save_dir + 'ciber_spitzer_auto_cross_cl_Bootes_newcal_CH1CH2mask_mopex_conserve_flux.npz'
	auto_cross_save_fpath =auto_cross_save_dir + 'ciber_spitzer_auto_cross_cl_Bootes_newcal_CH1CH2mask_conserve_flux.npz'

	ac = np.load(auto_cross_save_fpath)
	ac_mopex = np.load(auto_cross_mopex_save_fpath)

	# print(all_rlx_snpred)
	irac_list_save = ac['irac_list_save']
	inst_list_save = ac['inst_list_save']

	all_spitzer_auto_cl = ac['all_spitzer_auto_cl']
	all_spitzer_auto_cl_mopex = ac_mopex['all_spitzer_auto_cl']

	all_ciber_spitzer_cross_cl = ac['all_ciber_spitzer_cross_cl']
	all_ciber_spitzer_cross_cl_mopex = ac_mopex['all_ciber_spitzer_cross_cl']

	lb = ac['lb']

	fig = plt.figure(figsize=figsize)
	for x in range(len(irac_list_save)):

		plt.subplot(2,2,x+1)

		plt.plot(lb[startidx:endidx], np.array(all_ciber_spitzer_cross_cl[x]/all_ciber_spitzer_cross_cl_mopex[x])[startidx:endidx], label='CIBER $\\times$ Spitzer', color=colors[x])

		plt.plot(lb[startidx:endidx], np.array(all_spitzer_auto_cl[x]/all_spitzer_auto_cl_mopex[x])[startidx:endidx], color=colors[x], label='Spitzer auto', linestyle='dashed')

		plt.xscale('log')
		plt.ylim(0.5, 1.5)
		plt.grid(alpha=0.5)

		if x==0 or x==2:
			plt.ylabel('$C_{\\ell}^{\\text{self-cal}}/C_{\\ell}^{\\text{MOPEX}}$', fontsize=14)

		if x > 1:
			plt.xlabel('$\\ell$', fontsize=14)


		label = '$\\lambda_{CIBER}=$'+str(ciber_lams[x])+' $\\mu$m\n $\\lambda_{IRAC}=$'+str(irac_lams[x])+' $\\mu$m'
		plt.text(2e3, 1.2, label, fontsize=13)
		plt.axhspan(0.95, 1.05, color='grey', alpha=0.3)
		if x==0:
			plt.legend(loc=2, ncol=2, bbox_to_anchor=[0.3, 1.3], fontsize=13)

	# plt.tight_layout()
	plt.show()
	
	return fig

def plot_perfield_ciber_spitzer_cross(inst_list=[1, 2], irac_ch_list=[1, 2], bootes_ifield_list=[6, 7], \
									 include_snpred=True, bootes_names = ['Bootes B', 'Bootes A'], \
									 textxpos=200, textypos=200, textypos_cross=500, xlim=[150., 1.1e5], xlim_cross=[150., 1.1e5], ylim=[1e-3, 500], ylim_cross=[1e-3, 500], \
									 ell_min_cut=950, startidx=0, endidx=-1, tailstr='newcal_conserve_flux', \
									 capsize=3, markersize=4, alph=0.5, label_fs=12, linestyle='dashdot', \
									 text_fs=13, linewidth=0.75):
	
	
	bootes_colors = ['b', 'r']
	
	ps_basepath = config.ciber_basepath+'data/input_recovered_ps/ciber_spitzer/'
	
	fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 5.5))
	
	lams_irac = [3.6, 4.5]
	
	lams_ciber= [1.1, 1.8]
	
	
	if include_snpred:
		snpred_file = np.load(config.ciber_basepath+'data/cl_predictions/irac_vs_ciber_pv_cosmos15_nuInu_CH1CH2mask.npz')
		irac_pv = snpred_file['irac_pv']
		ciber_irac_pv = snpred_file['ciber_irac_pv']
		
	
	# spitzer autos
	for idx, irac_ch in enumerate(irac_ch_list):
		
		auto_file = np.load(ps_basepath+'ciber_spitzer_cross_auto_cl_TM1_IRACCH'+str(irac_ch)+'_'+tailstr+'.npz')
		lb = auto_file['lb']
		pf = lb*(lb+1)/(2*np.pi)
		lbmask = (lb >= lb[startidx])*(lb < lb[endidx])
		
		all_cl_spitzer = np.array(auto_file['all_cl_crossep'])
		all_dcl_spitzer = np.array(auto_file['all_clerr_crossep'])
		
		fieldav_cl_spitzer = np.array(auto_file['fieldav_cl_crossep'])
		fieldav_dcl_spitzer = np.array(auto_file['fieldav_clerr_crossep'])
				
		for fieldidx, ifield in enumerate(bootes_ifield_list):
			
			cl_spitzer = all_cl_spitzer[fieldidx]
			dcl_spitzer = all_dcl_spitzer[fieldidx]

			negmask = (cl_spitzer < 0)*lbmask
			posmask = (cl_spitzer > 0)*lbmask
			
			ax[idx, 0].errorbar(lb[posmask], (pf*cl_spitzer)[posmask], yerr=(pf*dcl_spitzer)[posmask],\
								fmt='o', capsize=capsize, markersize=markersize, color=bootes_colors[fieldidx], alpha=alph)
			
			ax[idx, 0].errorbar(lb[negmask], np.abs(pf*cl_spitzer)[negmask], yerr=(pf*dcl_spitzer)[negmask],\
								fmt='o', capsize=capsize, mfc='white', markersize=markersize, color=bootes_colors[fieldidx], alpha=alph)
			
			ax[idx, 0].plot(lb[lbmask], (pf*dcl_spitzer)[lbmask], linestyle=linestyle, linewidth=linewidth, color=bootes_colors[fieldidx], alpha=alph)
				
		
		negmask = (fieldav_cl_spitzer < 0)*lbmask
		posmask = (fieldav_cl_spitzer > 0)*lbmask
			
		ax[idx, 0].errorbar(lb[posmask], (pf*fieldav_cl_spitzer)[posmask], yerr=(pf*fieldav_dcl_spitzer)[posmask], fmt='o', capsize=capsize, markersize=markersize, \
						   color='k', label='Field average')
		
		ax[idx, 0].errorbar(lb[negmask], np.abs(pf*fieldav_cl_spitzer)[negmask], yerr=(pf*fieldav_dcl_spitzer)[negmask], fmt='o', capsize=capsize, markersize=markersize, \
						   color='k', mfc='white')
		
		if ell_min_cut is not None:
			ax[idx, 0].axvspan(xlim[0], ell_min_cut, color='grey', alpha=0.2)
		
		av_dcl_spitzer = np.sqrt(all_dcl_spitzer[0]**2 + all_dcl_spitzer[1]**2)/2.
		
		# plot uncertainty
		ax[idx, 0].plot(lb[lbmask], (pf*av_dcl_spitzer)[lbmask], linestyle=linestyle, linewidth=linewidth, color='k', label='Field average uncertainty')
		
		if include_snpred:
			lb_pred = np.array([50, 100]+list(lb))
			pf_pred = lb_pred*(lb_pred+1)/(2*np.pi)
			
			ax[idx, 0].plot(lb_pred, pf_pred*irac_pv[idx, 0], linestyle='dashed', color='grey')
		
		
		ax[idx, 0].set_xscale('log')
		ax[idx, 0].set_yscale('log')
		if idx==1:
			ax[idx, 0].set_xlabel('$\\ell$', fontsize=label_fs)
		ax[idx,0].set_ylabel('$D_{\\ell}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=label_fs)
		
		ax[idx,0].text(textxpos, textypos, 'IRAC CH'+str(irac_ch)+' ('+str(lams_irac[idx])+' $\\mu$m)', fontsize=text_fs)
		
		if xlim is not None:
			ax[idx,0].set_xlim(xlim)
		if ylim is not None:
			ax[idx,0].set_ylim(ylim)
			
		if idx==0:
			ax[idx, 0].set_xticks([1e3, 1e4, 1e5], ['' for x in range(3)])
			
			
		ax[idx,0].grid(alpha=0.5)
		
	# ciber spitzer cross
	
	for idx, irac_ch in enumerate(irac_ch_list):
		
		for idxc, inst in enumerate(inst_list):
			
			cross_file = np.load(ps_basepath+'ciber_spitzer_cross_auto_cl_TM'+str(inst)+'_IRACCH'+str(irac_ch)+'_'+tailstr+'.npz')
			lb = cross_file['lb']
			pf = lb*(lb+1)/(2*np.pi)
			
			lam_ciber = lams_ciber[idxc]
			lam_irac = lams_irac[idx]
			
			textstr = str(lam_ciber)+' $\\mu$m $\\times$ '+str(lam_irac)+' $\\mu$m'
			ax[idx,idxc+1].text(textxpos, textypos, textstr, fontsize=text_fs)

			lbmask = (lb >= lb[startidx])*(lb < lb[endidx])
			
			all_cl_cross = cross_file['all_cl_cross']
			all_clerr_cross_tot = cross_file['all_clerr_cross_tot']
			all_clerr_spitzer_noise_ciber = cross_file['all_clerr_spitzer_noise_ciber']
			all_clerr_ciber_noise_spitzer = cross_file['all_clerr_ciber_noise_spitzer']
			all_clerr_ciber_noise_spitzer_noise = cross_file['all_clerr_ciber_noise_spitzer_noise']
			
			av_dcl_spitzer_noise_ciber = np.sqrt(all_clerr_spitzer_noise_ciber[0]**2+all_clerr_spitzer_noise_ciber[1]**2)/2.
			av_dcl_ciber_noise_spitzer = np.sqrt(all_clerr_ciber_noise_spitzer[0]**2+all_clerr_ciber_noise_spitzer[1]**2)/2.
			av_dcl_ciber_noise_spitzer_noise = np.sqrt(all_clerr_ciber_noise_spitzer_noise[0]**2+all_clerr_ciber_noise_spitzer_noise[1]**2)/2.
			
			fieldav_cl_cross = cross_file['fieldav_cl_cross']
			fieldav_clerr_cross = cross_file['fieldav_clerr_cross']
			
#             print(len(all_cl_cross))
			
#             print(bootes_names, bootes_colors)
			for fieldidx, ifield in enumerate(bootes_ifield_list):

				cl_cross_indiv = all_cl_cross[fieldidx]
				negmask = (cl_cross_indiv < 0)*lbmask
				posmask = (cl_cross_indiv > 0)*lbmask
		
				ax[idx, idxc+1].errorbar(lb[posmask], (pf*cl_cross_indiv)[posmask], yerr=(pf*all_clerr_cross_tot[fieldidx])[posmask],\
										 fmt='o', label=bootes_names[fieldidx], color=bootes_colors[fieldidx], \
										alpha=alph, capsize=capsize, markersize=markersize)
				
				ax[idx, idxc+1].errorbar(lb[negmask], np.abs(pf*cl_cross_indiv)[negmask], yerr=(pf*all_clerr_cross_tot[fieldidx])[negmask],\
										 fmt='o', mfc='white', color=bootes_colors[fieldidx], \
										alpha=alph, capsize=capsize, markersize=markersize)
				
			
			
#             # field average

			negmask = (fieldav_cl_cross < 0)*lbmask
			posmask = (fieldav_cl_cross > 0)*lbmask
		
			ax[idx, idxc+1].errorbar(lb[posmask], (pf*fieldav_cl_cross)[posmask], yerr=(pf*fieldav_clerr_cross)[posmask], \
									fmt='o', label='Field average', color='k', \
									capsize=capsize, markersize=markersize)
			ax[idx, idxc+1].errorbar(lb[negmask], np.abs(pf*fieldav_cl_cross)[negmask], yerr=(pf*fieldav_clerr_cross)[negmask], \
									fmt='o', mfc='white', color='k', \
									capsize=capsize, markersize=markersize)

			ax[idx, idxc+1].plot(lb[lbmask], (pf*av_dcl_ciber_noise_spitzer)[lbmask], color='C0', linestyle=linestyle, linewidth=linewidth, label='$N_{\\ell}^{CIBER}$ $\\times$ $C_{\\ell}^{IRAC}$', alpha=0.8)
			ax[idx, idxc+1].plot(lb[lbmask], (pf*av_dcl_spitzer_noise_ciber)[lbmask], color='C1', linestyle=linestyle, linewidth=linewidth, label='$N_{\\ell}^{IRAC}$ $\\times$ $C_{\\ell}^{CIBER}$', alpha=0.8)
			ax[idx, idxc+1].plot(lb[lbmask], (pf*av_dcl_ciber_noise_spitzer_noise)[lbmask], color='C2', linestyle=linestyle, linewidth=linewidth, label='$N_{\\ell}^{IRAC}$ $\\times$ $N_{\\ell}^{CIBER}$', alpha=0.8)

			ax[idx, idxc+1].plot(lb[lbmask], (pf*fieldav_clerr_cross)[lbmask], color='k', linestyle=linestyle, linewidth=linewidth, label='Field average uncertainty')
			
			
			if include_snpred:
				lb_pred = np.array([50, 100]+list(lb))
				pf_pred = lb_pred*(lb_pred+1)/(2*np.pi)

				ax[idx, idxc+1].plot(lb_pred, pf_pred*ciber_irac_pv[idx, idxc], linestyle='dashed', color='grey', label='$C_{\\ell}^{SN, pred}$ (IGL+ISL)')

			
			
			ax[idx, idxc+1].set_xscale('log')
			ax[idx, idxc+1].set_yscale('log')
			
#             ax[idx, idxc+1].text(textxpos, textypos, )
			
			if xlim_cross is not None:
				ax[idx,idxc+1].set_xlim(xlim_cross)
			if ylim_cross is not None:
				ax[idx,idxc+1].set_ylim(ylim_cross)
			
			if idx==1:
				ax[idx, idxc+1].set_xlabel('$\\ell$', fontsize=label_fs)
				
#             if idxc==1:
			ax[idx, idxc+1].set_yticks([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2], ['' for x in range(6)])
			
			ax[idx, idxc+1].grid(alpha=0.5)
			if ell_min_cut is not None:
				ax[idx, idxc+1].axvspan(xlim[0], ell_min_cut, color='grey', alpha=0.2)
		
			if idx==0:
				ax[idx, idxc+1].set_xticks([1e3, 1e4, 1e5], ['' for x in range(3)])
			
	handles, labels = plt.gca().get_legend_handles_labels()

	order = [6, 5, 7, 4, 0, 1, 2, 3]
		
	plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], \
			ncol=4, bbox_to_anchor=[1.05, 2.35], fontsize=9)
	
#     plt.legend(ncol=4, loc=2, bbox_to_anchor=[-2.2, 2.4])
	plt.subplots_adjust(wspace=0, hspace=0)
	
	plt.show()
	
	return fig

# def plot_ciber_spitzer_auto_cross(spitzer_mask_string=None, ciber_tailstrs=None, startidx=0, endidx=-1, \
#                                  mag_lim_sdwfs = None, mag_lims_ciber=None, keywd = None, \
#                                  xlim=[180, 1.5e5], alph=0.6, include_z14=False, include_snpred=False, add_str=None):
	
#     irac_lamdict = dict({1:3.6, 2:4.5})
#     bandstr_dict = dict({1:'J', 2:'H'})
#     ciber_lamdict = dict({1:1.1, 2:1.8})
	
#     if mag_lims_ciber is None:
#         mag_lims_ciber = [17.5, 17.0] # J, H
		
#     spitzer_auto_basepath = config.ciber_basepath+'data/'

	
#     colors = ['b', 'C3']

#     fig, ax = plt.subplots(figsize=(7, 7), nrows=2, ncols=2)
		
#     if spitzer_mask_string is None:
#         spitzer_mask_string = '052623_gradsub_byepoch_CH'+str(irac_ch)
		
#     for irac_ch in [1]: 
#         lamirac = irac_lamdict[irac_ch]

		
#         for inst in [1]:
#             bandstr = bandstr_dict[inst]
#             lam_ciber = ciber_lamdict[inst]
			
# #             if ciber_tailstr is None:
# #                 ciber_mask_tail = 'maglim_'+bandstr+'_Vega_'+str(mag_lims_ciber[inst-1])+'_062623'

#             # load CIBER x Spitzer cross corr ---------------------------
			
#             save_cl_fpath = spitzer_auto_basepath+'input_recovered_ps/ciber_spitzer/ciber_spitzer_cross_auto_cl_TM'+str(inst)+'_IRACCH'+str(irac_ch)
#             if add_str is not None:
#                 save_cl_fpath += '_'+add_str
#             save_cl_fpath +='.npz'
#             ciber_spitzer_clfile = np.load(save_cl_fpath)
			
#             lb = ciber_spitzer_clfile['lb']
#             prefac = lb*(lb+1)/(2*np.pi)
			
#             all_cl_cross = ciber_spitzer_clfile['all_cl_cross']
	
#             all_clerr_cross = ciber_spitzer_clfile['all_clerr_cross_tot']
#             all_clerr_spitzer_noise_ciber = ciber_spitzer_clfile['all_clerr_spitzer_noise_ciber']
#             all_clerr_ciber_noise_spitzer = ciber_spitzer_clfile['all_clerr_ciber_noise_spitzer']
#             all_clerr_cross_tot = ciber_spitzer_clfile['all_clerr_cross_tot']
			
#             fieldav_cl_cross = ciber_spitzer_clfile['fieldav_cl_cross']
#             fieldav_clerr_cross = ciber_spitzer_clfile['fieldav_clerr_cross']
			
#             snr_cross_largescale, tot_snr_cross = compute_cl_snr(lb, fieldav_cl_cross, fieldav_clerr_cross, lb_max=2000)

#             print('snr cross large scale:', snr_cross_largescale)
#             print('total snr large scale cross:', tot_snr_cross)
			
	
#             # load Spitzer auto corr -------------------------
		
			
#             all_cl_spitzer = ciber_spitzer_clfile['all_cl_spitzer']
#             all_clerr_spitzer= ciber_spitzer_clfile['all_clerr_spitzer']
			
#             fieldav_cl_spitzer = ciber_spitzer_clfile['fieldav_cl_spitzer']
#             fieldav_clerr_spitzer = ciber_spitzer_clfile['fieldav_clerr_spitzer']
			
# #             mean_cl_spitzer = np.mean(np.array(all_cl_spitzer), axis=0)
# #             sum_err_spitz = np.sqrt(all_clerr_spitzer[0]**2 + all_clerr_spitzer[1]**2)/np.sqrt(2)
# #             spitz_auto_knox_errors = np.sqrt(1./((2*lb+1)*cbps.Mkk_obj.delta_ell)) # factor of 1 in numerator since auto is a cross-epoch cross
# #             A_eff = 2*4*0.6 # two fields 2x2 degree, with 60% unmasked
# #             fsky = A_eff/(41253.)    
# #             spitz_auto_knox_errors /= np.sqrt(fsky)
# #             print('spitz_auto_knox_errors fractional', spitz_auto_knox_errors)
# #             spitz_auto_knox_errors *= np.abs(mean_cl_spitzer)
# #             sum_err_spitz = np.sqrt(sum_err_spitz**2 + spitz_auto_knox_errors**2)
			
			
#             # ----------------- load CIBER auto spectra ---------------------------
#             # grab bootes fields and compute mean/uncertainty from two-field average

# #             obs_name = ''
# #             cl_fpath_obs = 'data/input_recovered_ps/cl_files/TM'+str(inst)+'/cl_'+obs_name+'.npz'
# #             lb, observed_recov_ps, observed_recov_dcl_perfield,\
# #                 observed_field_average_cl, observed_field_average_dcl,\
# #                     mock_all_field_cl_weights = load_weighted_cl_file(cl_fpath_obs)
			
# #             mean_ciber_bootes_ps = 0.5*(observed_recov_ps[2]+observed_recov_ps[3])
# #             ciber_mean_clerr = None
			
#             # plot something!
			
#             lbmask = (lb >=lb[startidx])*(lb < lb[endidx])

#             cross_negmask = (fieldav_cl_cross < 0)*lbmask
#             cross_posmask = (fieldav_cl_cross > 0)*lbmask

#             ax[irac_ch-1, inst-1].errorbar(lb[cross_posmask], (prefac*fieldav_cl_cross)[cross_posmask], yerr=(prefac*fieldav_clerr_cross)[cross_posmask], fmt='o', color='k', alpha=alph, markersize=4, capsize=4, label='CIBER x Spitzer')
#             ax[irac_ch-1, inst-1].errorbar(lb[cross_negmask], np.abs(prefac*fieldav_cl_cross)[cross_negmask], yerr=(prefac*fieldav_clerr_cross)[cross_negmask], fmt='o', color='k', markersize=4, capsize=4, mfc='white')

#             negmask_spitzerauto = (fieldav_cl_spitzer < 0)*lbmask
#             posmask_spitzerauto = (fieldav_cl_spitzer > 0)*lbmask

# #             print(fieldav_cl_spitz)
#             ax[irac_ch-1, inst-1].errorbar(lb[posmask_spitzerauto], (prefac*fieldav_cl_spitzer)[posmask_spitzerauto], yerr=(prefac*fieldav_clerr_spitzer)[posmask_spitzerauto], markersize=4, fmt='o', capsize=4, color='grey', label='Spitzer auto')
#             ax[irac_ch-1, inst-1].errorbar(lb[negmask_spitzerauto], np.abs((prefac*fieldav_cl_spitzer)[negmask_spitzerauto]), yerr=(prefac*fieldav_clerr_spitzer)[negmask_spitzerauto], markersize=4, mfc='white', fmt='o', capsize=4, color='grey')

# #             prefac_ciber = lb*(lb+1)/(2*np.pi)
# #             ax[inst-1].errorbar(lb[startidx:endidx], (prefac*all_mean_bootes_ps)[lbmask], yerr=(prefac_ciber*ciber_mean_clerr)[startidx:endidx], markersize=4, fmt='o', alpha=alph, capsize=4, color=colors[inst-1], label='CIBER auto')


#             ax[irac_ch-1, inst-1].set_xscale("log")
#             ax[irac_ch-1, inst-1].set_yscale("log")
#             ax[irac_ch-1, inst-1].set_ylim(1e-6, 2e4)
#             ax[irac_ch-1, inst-1].tick_params(labelsize=12)
			
#             if irac_ch==2:
#                 ax[irac_ch-1, inst-1].set_xlabel('$\\ell$', fontsize=16)
#             if inst==1:
#                 ax[irac_ch-1, inst-1].set_ylabel('$D_{\\ell}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=16)
#             ax[irac_ch-1, inst-1].set_xlim(xlim)
	
#             if inst==1:
#                 ax[irac_ch-1, inst-1].set_yticks([1e-3, 1e-1, 1e1, 1e3])
#             elif inst==2:
#                 ax[irac_ch-1, inst-1].set_yticks([1e-3, 1e-1, 1e1, 1e3], ['', '', '', ''])
				
#             ax[irac_ch-1, inst-1].grid(alpha=0.3)
#             ax[irac_ch-1,inst-1].text(250, 60, str(lam_ciber)+' $\\mu$m $\\times$ '+str(lamirac)+' $\\mu$m\nMask $'+bandstr+'<'+str(mag_lims_ciber[inst-1])+'$', fontsize=14, color='k', bbox=dict({'facecolor':'white', 'alpha':0.4, 'edgecolor':'None'}))

#     plt.legend(fontsize=14, bbox_to_anchor=[1.0, 1.4], ncol=2)
#     plt.subplots_adjust(wspace=0.0)

#     plt.show()

#     return fig


def plot_spitzer_auto(inst, irac_ch, lb, spitzer_auto_cl, spitzer_auto_clerr, all_cl_spitzer=None, all_clerr_spitzer=None, return_fig=True, \
						capsize=3, markersize=5, capthick=1.5, alph=0.5, startidx=0, endidx=-1, ylim=[1e-4, 1e3], include_z14=False, include_snpred=False):
	
	''' Plot cross spectra with individual cross noise terms '''
	
	pf = lb*(lb+1)/(2*np.pi)

	lbmask = (lb >= lb[startidx])*(lb < lb[endidx])
	
	f = plt.figure(figsize=(5,4))
	
	plt.title('Cross-epoch, IRAC CH'+str(irac_ch), fontsize=14)

	if all_cl_spitzer is not None and all_clerr_spitzer is not None:

		posmask_A = lbmask*(all_cl_spitzer[1] > 0)
		negmask_A = lbmask*(all_cl_spitzer[1] < 0)

		posmask_B = lbmask*(all_cl_spitzer[0] > 0)
		negmask_B = lbmask*(all_cl_spitzer[0] < 0)

		plt.errorbar(lb[posmask_A], (pf*all_cl_spitzer[1])[posmask_A], yerr=(pf*all_clerr_spitzer[1])[posmask_A], capsize=capsize, markersize=markersize, capthick=capthick, alpha=alph, fmt='o', color='r', label='Bootes A')
		plt.errorbar(lb[posmask_B], (pf*all_cl_spitzer[0])[posmask_B], yerr=(pf*all_clerr_spitzer[0])[posmask_B], capsize=capsize, markersize=markersize, capthick=capthick, alpha=alph, fmt='o', color='b', label='Bootes B')
	
		plt.errorbar(lb[negmask_A], np.abs(pf*all_cl_spitzer[1])[negmask_A], yerr=(pf*all_clerr_spitzer[1])[negmask_A], mfc='white', capsize=capsize, markersize=markersize, capthick=capthick, alpha=alph, fmt='o', color='r')
		plt.errorbar(lb[negmask_B], np.abs(pf*all_cl_spitzer[0])[negmask_B], yerr=(pf*all_clerr_spitzer[0])[negmask_B], mfc='white', capsize=capsize, markersize=markersize, capthick=capthick, alpha=alph, fmt='o', color='b')
	
		# plt.plot(lb, pf*all_clerr_spitzer[1], color='r', linestyle='dashed', alpha=alph)
		# plt.plot(lb, pf*all_clerr_spitzer[0], color='b', linestyle='dashed', alpha=alph)


	posmask = lbmask*(spitzer_auto_cl > 0)
	negmask = lbmask*(spitzer_auto_cl < 0)

	plt.errorbar(lb[posmask], (pf*spitzer_auto_cl)[posmask], yerr=(pf*spitzer_auto_clerr)[posmask], fmt='o', capsize=capsize, markersize=markersize, capthick=capthick, color='k', label='Spitzer auto')
	plt.errorbar(lb[negmask], np.abs(pf*spitzer_auto_cl)[negmask], yerr=(pf*spitzer_auto_clerr)[negmask], fmt='o', mfc='white', capsize=capsize, markersize=markersize, capthick=capthick, color='k')

	plt.plot(lb, pf*spitzer_auto_clerr, color='k', linestyle='dashed', alpha=0.8)

	if include_z14 and irac_ch==1:
		zemcov_auto = np.loadtxt(config.ciber_basepath+'/data/zemcov14_ps/ciber_3.6x3.6_dCl.txt', skiprows=8)
		zemcov_lb = zemcov_auto[:,0]
							
		plt.errorbar(zemcov_lb, zemcov_auto[:,1], yerr=zemcov_auto[:,2], label='Zemcov+14', fmt='o', markersize=markersize, capsize=capsize, capthick=capthick, color='forestgreen', alpha=0.7)


	plt.xscale('log')
	plt.yscale('log')
	if ylim is not None:
		plt.ylim(ylim[0], ylim[1])
	plt.xlim(150, 1.2e5)
	plt.xlabel('$\\ell$', fontsize=14)
	plt.ylabel('$D_{\\ell}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=14)
	plt.legend(loc=2)
	plt.grid(alpha=0.5)
	plt.show()
	
	if return_fig:
		return f


def plot_fieldav_ciber_spitzer_cross(inst, irac_ch, lb, fieldav_cl_cross, fieldav_clerr_cross, all_cl_cross, all_clerr_cross_tot, \
									return_fig=True, startidx=0, endidx=-1, capsize=3, markersize=5, capthick=1.5, alph=0.5, include_z14=False, \
									include_snpred=False):

	pf = lb*(lb+1)/(2*np.pi)

	f = plt.figure(figsize=(5, 4))

	plt.title('CIBER TM'+str(inst)+' x IRAC CH'+str(irac_ch), fontsize=14)

	lbmask = (lb >= lb[startidx])*(lb < lb[endidx])

	posmask_A = (all_cl_cross[1] > 0)*lbmask
	posmask_B = (all_cl_cross[0] > 0)*lbmask
	negmask_A = (all_cl_cross[1] < 0)*lbmask
	negmask_B = (all_cl_cross[0] < 0)*lbmask

	posmask_av = (fieldav_cl_cross > 0)*lbmask
	negmask_av = (fieldav_cl_cross < 0)*lbmask

	plt.errorbar(lb[posmask_A], (pf*all_cl_cross[1])[posmask_A], yerr=(pf*all_clerr_cross_tot[1])[posmask_A], capsize=capsize, markersize=markersize, capthick=capthick, alpha=0.7, fmt='o', color='r', label='Bootes A')
	plt.errorbar(lb[negmask_A], np.abs(pf*all_cl_cross[1])[negmask_A], yerr=(pf*all_clerr_cross_tot[1])[negmask_A], capsize=capsize, markersize=markersize, capthick=capthick, alpha=0.7, fmt='o', mfc='white', color='r')

	# plt.plot(lb[lbmask], (pf*all_clerr_cross_tot[1])[lbmask], linestyle='dashed', color='r', alpha=alph)

	plt.errorbar(lb[posmask_B], (pf*all_cl_cross[0])[posmask_B], yerr=(pf*all_clerr_cross_tot[0])[posmask_B], capsize=capsize, markersize=markersize, capthick=capthick, alpha=0.7, fmt='o', color='b', label='Bootes B')
	plt.errorbar(lb[negmask_B], np.abs(pf*all_cl_cross[0])[negmask_B], yerr=(pf*all_clerr_cross_tot[0])[negmask_B], capsize=capsize, markersize=markersize, capthick=capthick, alpha=0.7, fmt='o', mfc='white', color='b')
	# plt.plot(lb[lbmask], (pf*all_clerr_cross_tot[0])[lbmask], linestyle='dashed', color='b', alpha=alph)

	plt.errorbar(lb[posmask_av], (pf*fieldav_cl_cross)[posmask_av], yerr=(pf*fieldav_clerr_cross)[posmask_av], fmt='o', capsize=capsize, markersize=markersize, capthick=capthick, color='k', label='Field average')
	plt.errorbar(lb[negmask_av], np.abs(pf*fieldav_cl_cross)[negmask_av], yerr=(pf*fieldav_clerr_cross)[negmask_av], mfc='white', capsize=capsize, markersize=markersize, capthick=capthick, fmt='o', color='k')

	plt.plot(lb[lbmask], (pf*fieldav_clerr_cross)[lbmask], linestyle='dashed', color='k', alpha=0.8)

	if include_z14 and irac_ch==1:
		lam_dict_z14 = dict({1:1.1, 2:1.6})

		zemcov_auto = np.loadtxt(config.ciber_basepath+'/data/zemcov14_ps/ciber_'+str(lam_dict_z14[inst])+'x3.6_dCl.txt', skiprows=8)
		zemcov_lb = zemcov_auto[:,0]
							
		plt.errorbar(zemcov_lb, zemcov_auto[:,1], yerr=zemcov_auto[:,2], label='Zemcov+14', fmt='o', markersize=markersize, capsize=capsize, capthick=capthick, color='forestgreen', alpha=0.7)


	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel('$\\ell$', fontsize=14)
	plt.ylabel('$D_{\\ell}$ [nW$^2$ m$^{-4}$ sr$^{-1}$]', fontsize=14)
	plt.grid(alpha=0.5)
	plt.legend(loc=2)
	plt.xlim(150, 1.2e5)
	plt.ylim(1e-4, 1e3)
	plt.show()

	if return_fig:
		return f

 
def plot_ciber_crossep_auto_with_terms(inst, irac_ch, fieldname, lb, crossep_cl, crossep_clerr, \
										std_nl1ds_nAsB, std_nl1ds_nBsA, std_nl1ds_nAnB, return_fig=True, \
										ylim=[5e-5, 1e3], startidx=0, endidx=-1, markersize=5, capsize=3, capthick=1.5, \
										alph=0.7):

	pf = lb*(lb+1)/(2*np.pi)
	
	f = plt.figure(figsize=(5,4))
	
	plt.title('Cross-epoch IRAC CH'+str(irac_ch)+', '+fieldname, fontsize=14)

	lbmask = (lb >= lb[startidx])*(lb < lb[endidx])
	posmask = (crossep_cl > 0)*lbmask
	negmask = (crossep_cl < 0)*lbmask

	plt.errorbar(lb[posmask], (pf*crossep_cl)[posmask], yerr=(pf*crossep_clerr)[posmask], fmt='o', label='0.5(A+B) x 0.5(C+D)', color='k', markersize=markersize, capsize=capsize, capthick=capthick)
	plt.errorbar(lb[negmask], np.abs(pf*crossep_cl)[negmask], yerr=(pf*crossep_clerr)[negmask], mfc='white', fmt='o', color='k', markersize=markersize, capsize=capsize, capthick=capthick)
	
	plt.plot(lb[lbmask], (pf*std_nl1ds_nAsB)[lbmask], color='C0', linestyle='dashed', label='Noise AB x CD', alpha=alph)
	plt.plot(lb[lbmask], (pf*std_nl1ds_nBsA)[lbmask], color='C1', linestyle='dashed', label='Noise CD x AB', alpha=alph)
	plt.plot(lb[lbmask], (pf*std_nl1ds_nAnB)[lbmask], color='C2', linestyle='dashed', label='Noise AB x Noise CD', alpha=alph)
	plt.plot(lb[lbmask], (pf*crossep_clerr)[lbmask], color='k', linestyle='dashed', label='Total uncertainty', alpha=alph)
	
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel('$\\ell$', fontsize=14)
	plt.ylabel('$D_{\\ell}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=14)
	plt.legend()
	plt.ylim(ylim[0], ylim[1])
	plt.tick_params(labelsize=12)
	plt.grid(alpha=0.5)
	plt.show()
	
	if return_fig:
		return f


def plot_ciber_spitzer_cross_with_terms(inst, irac_ch, fieldname, lb, cl_cross, clerr_cross_tot, clerr_ciber_noise_spitzer, \
									   clerr_spitzer_noise_ciber, clerr_ciber_noise_spitzer_noise, return_fig=True, \
									   startidx=0, endidx=-1, capsize=3, markersize=5, capthick=1.5):
	
	''' Plot cross spectra with individual cross noise terms '''
	
	pf = lb*(lb+1)/(2*np.pi)
	
	f = plt.figure(figsize=(5,4))
	
	plt.title('CIBER TM'+str(inst)+' x IRAC CH'+str(irac_ch)+', '+fieldname, fontsize=14)

	lbmask = (lb >= lb[startidx])*(lb < lb[endidx])
	
	posmask = (cl_cross > 0)*lbmask
	negmask = (cl_cross < 0)*lbmask

	plt.errorbar(lb[posmask], (pf*cl_cross)[posmask], yerr=(pf*clerr_cross_tot)[posmask], fmt='o', color='k', capsize=capsize, markersize=markersize, capthick=capthick)
	plt.errorbar(lb[negmask], np.abs(pf*cl_cross)[negmask], yerr=(pf*clerr_cross_tot)[negmask], fmt='o', mfc='white', color='k', capsize=capsize, markersize=markersize, capthick=capthick)

	# plt.errorbar(lb, pf*cl_cross, yerr=pf*clerr_cross_tot, fmt='o', label='CIBER x Spitzer')
	
	plt.plot(lb[lbmask], (pf*clerr_ciber_noise_spitzer)[lbmask], color='C0', linestyle='dashed', label='CIBER noise x Spitzer')
	plt.plot(lb[lbmask], (pf*clerr_spitzer_noise_ciber)[lbmask], color='C1', linestyle='dashed', label='Spitzer noise x CIBER')
	plt.plot(lb[lbmask], (pf*clerr_ciber_noise_spitzer_noise)[lbmask], color='C2', linestyle='dashed', label='CIBER noise x Spitzer noise')
	plt.plot(lb[lbmask], (pf*clerr_cross_tot)[lbmask], color='k', linestyle='dashed', label='Total uncertainty')
	
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel('$\\ell$', fontsize=14)
	plt.ylabel('$D_{\\ell}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=14)
	plt.legend()
	plt.grid(alpha=0.5)
	plt.tick_params(labelsize=12)
	plt.ylim(1e-4, 1e3)
	plt.show()
	
	if return_fig:
		return f


def plot_irac_beam(irac_ch, irac_lb, irac_bl, interp_bl=None):
	fig = plt.figure()
	plt.scatter(irac_lb, irac_bl)
	if interp_bl is None:
		plt.plot(irac_lb, interp_bl, color='r')
	plt.yscale('log')
	plt.xscale('log')
	plt.title('IRAC CH'+str(irac_ch)+' beam')
	plt.xlabel('$\\ell$')
	plt.ylabel('$B_{\\ell}$')
	plt.xlim(100, 1e5)
	plt.ylim(1e-2, 1e0)
	plt.show()
	
	return fig

def plot_field_weights_ciber_bands(mock_field_cl_basepath, return_fig=True, show=True):
	

	fig = plt.figure(figsize=(10,4))

	linestyles = ['solid', 'dashed']
	textstrs = ['CIBER 1.1$\\mu$m\nMask $J<17.5$', 'CIBER 1.8$\\mu$m\nMask $H<17.0$']
	lams = [1.1, 1.8]
	for i, inst in enumerate([1, 2]):
		plt.subplot(1,3,inst)
		mock_all_field_cl_weights = np.load(mock_field_cl_basepath+'mock_all_field_cl_weights_TM'+str(inst)+'.npz')['mock_all_field_cl_weights']
		cl_sumweights = np.sum(mock_all_field_cl_weights, axis=0)
		lam = lams[i]
		for k in range(5):
			if k==0:
				lab = 'CIBER '+str(lam)+' $\\mu$m'
			else:
				lab = None
			plt.plot(lb, mock_all_field_cl_weights[k]/cl_sumweights, color='C'+str(k), linestyle=linestyles[i], label=lab)
		plt.text(200, 0.35, textstrs[i], fontsize=14)
		plt.axhline(0.2, alpha=0.5, linestyle='dashdot', color='k')
		plt.xscale('log')
		plt.xlabel('$\\ell$', fontsize=14)
		plt.ylabel('Field weights', fontsize=14)
		plt.tick_params(labelsize=11)
		plt.ylim(0, 0.45)
		plt.grid(alpha=0.3)

	plt.subplot(1,3,3)
	textstrs_plot = ['CIBER 1.1$\\mu$m', 'CIBER 1.8$\\mu$m']
	marker_list = ['*', 'o']
	for i, inst in enumerate([1, 2]):
		neff_frac = np.load(mock_field_cl_basepath+'mode_fraction_TM'+str(inst)+'.npz')['mode_fraction']
		plt.scatter(lb, neff_frac, marker=marker_list[i], color='k', label=textstrs_plot[i])
		
	plt.legend()
	plt.xscale('log')
	plt.xlabel('$\\ell$', fontsize=14)
	plt.ylim(0, 1)
	plt.tick_params(labelsize=11)
	plt.grid(alpha=0.5)
	plt.ylabel('$N_{eff}^{weighted}/N_{eff}^{unweighted}$', fontsize=14)
	plt.tight_layout()
	plt.tight_layout()
	if show:
		plt.show()
	
	if return_fig:
		return fig

def plot_relative_ps_vs_maskradius(inst, mag_lim, observed_run_names_vary_mask=None, \
								  obs_colors = ['k', 'C5', 'C6'], startidx=1, endidx =-1, \
								  ylim = [0, 2], figsize=(6,5)):
	
	lam_dict = dict({1:1.1, 2:1.8})
	bandstr_dict = dict({1:'J', 2:'H'})
	
	if inst==1:
		obs_labels = ['Fiducial', '$r(J>14) \\times 1.25$', '$r(J>14) \\times 0.75$']

		if observed_run_names_vary_mask is None:
			observed_run_names_vary_mask = ['observed_Jlt'+str(mag_lim)+'_062623', \
								'observed_Jlt'+str(mag_lim)+'_062623_times_1p25_radii', \
							   'observed_Jlt'+str(mag_lim)+'_062623_times_0p75_radii']
			
	elif inst==2:
		obs_labels = ['Fiducial', 'r$(H>14) \\times 1.25$', 'r$(H>14) \\times 0.75$']
		if observed_run_names_vary_mask is None:
			observed_run_names_vary_mask = ['observed_Hlt'+str(mag_lim)+'_062623', \
								'observed_Hlt'+str(mag_lim)+'_062623_times_1p25_radii', \
							   'observed_Hlt'+str(mag_lim)+'_062623_times_0p75_radii']

	obs_fieldav_cl_vary_mask = []
	
	for obs_name in observed_run_names_vary_mask:
		cl_fpath_obs = config.ciber_basepath+'data/input_recovered_ps/cl_files/TM'+str(inst)+'/cl_'+obs_name+'.npz'
		lb, observed_recov_ps, observed_recov_dcl_perfield,\
		observed_field_average_cl, observed_field_average_dcl,\
			mock_all_field_cl_weights = load_weighted_cl_file(cl_fpath_obs)

		obs_fieldav_cl_vary_mask.append(observed_field_average_cl)
		
	fig = plt.figure(figsize=figsize)
	for obs_idx, obs_cl in enumerate(obs_fieldav_cl_vary_mask):
		if obs_idx==0:
			alpha = 0.5
		else:
			alpha = 0.8
		plt.errorbar(lb[startidx:endidx], (obs_cl/obs_fieldav_cl_vary_mask[0])[startidx:endidx], yerr=(np.median(np.abs(field_average_error), axis=0)/obs_fieldav_cl_vary_mask[0])[startidx:endidx], \
					fmt='o', capthick=1.5, color=obs_colors[obs_idx], capsize=3, markersize=4, linewidth=2, \
					label=obs_labels[obs_idx], alpha=alpha)

	plt.xscale('log')
	plt.tick_params(labelsize=14)
	plt.xlabel('$\\ell$', fontsize=16)
	plt.ylabel('$C_{\\ell}/C_{\\ell}^{fiducial}$', fontsize=16)
	plt.grid(alpha=0.5, color='grey')
	plt.text(250, 1.51, 'CIBER '+str(lam_dict[inst])+'$\\mu$m\nObserved data\nMask '+bandstr_dict[inst]+'$<'+str(mag_lim)+'$', fontsize=18)
	plt.legend(fontsize=10, loc=4, ncol=1)
	plt.ylim(ylim)
	plt.tight_layout()
	plt.show()
	
	return fig

def plot_relative_ps_vs_g1(inst, mag_lim, observed_run_names_vary_gain=None, \
						  obs_colors = ['k', 'C5', 'C6', 'C7', 'C8'], zorders = [2, 1, 1, 1, 1], \
						  startidx=1, endidx=-1, figsize=(6,3), ylim=[0.5, 1.5]):
	
	lam_dict = dict({1:1.1, 2:1.8})
	bandstr_dict = dict({1:'J', 2:'H'})
	
	if inst==1:
		obs_labels = ['$g_1=-2.7$ (fiducial)', '$g_1=-2.5$', '$g_1=-2.6$', '$g_1=-2.8$', '$g_1=-2.9$']

		if observed_run_names_vary_gain is None:
		
			observed_run_names_vary_gain = ['observed_Jlt'+str(mag_lim)+'_062623', \
											'observed_Jlt'+str(mag_lim)+'_062623_g12p5', \
											'observed_Jlt'+str(mag_lim)+'_062623_g12p6', \
											'observed_Jlt'+str(mag_lim)+'_062623_g12p8', \
											'observed_Jlt'+str(mag_lim)+'_062623_g12p9']    
	elif inst==2:
		obs_labels = ['$g_1=-3.0$ (fiducial)', '$g_1=-2.8$', '$g_1=-2.9$', '$g_1=-3.1$', '$g_1=-3.2$']
		if observed_run_names_vary_gain is None:

			observed_run_names_vary_gain = ['observed_Hlt'+str(mag_lim)+'_062623', \
								'observed_Hlt'+str(mag_lim)+'_062623_g12p8', \
								'observed_Hlt'+str(mag_lim)+'_062623_g12p9', \
								'observed_Hlt'+str(mag_lim)+'_062623_g13p1', \
								'observed_Hlt'+str(mag_lim)+'_062623_g13p2']
			
	obs_fieldav_cl_vary_gain = []
	for obs_name in observed_run_names_vary_gain:
		cl_fpath_obs = config.ciber_basepath+'data/input_recovered_ps/cl_files/TM'+str(inst)+'/cl_'+obs_name+'.npz'

		lb, observed_recov_ps, observed_recov_dcl_perfield,\
		observed_field_average_cl, observed_field_average_dcl,\
			mock_all_field_cl_weights = load_weighted_cl_file(cl_fpath_obs)

		obs_fieldav_cl_vary_gain.append(observed_field_average_cl)
		
	# plot it!
	fig = plt.figure(figsize=figsize)
	for obs_idx, obs_cl in enumerate(obs_fieldav_cl_vary_gain):
		if obs_idx==0:
			alpha = 0.5
		else:
			alpha = 0.8
		plt.errorbar(lb[startidx:endidx], (obs_cl/obs_fieldav_cl_vary_gain[0])[startidx:endidx], yerr=(np.median(np.abs(field_average_error), axis=0)/obs_fieldav_cl_vary_gain[0])[startidx:endidx], \
					fmt='o', capthick=1.5, color=obs_colors[obs_idx], capsize=3, markersize=4, linewidth=2, zorder=zorders[obs_idx], \
					label=obs_labels[obs_idx], alpha=alpha)

	plt.xscale('log')
	plt.tick_params(labelsize=14)
	plt.xlabel('$\\ell$', fontsize=16)
	plt.ylabel('$C_{\\ell}/C_{\\ell}^{fiducial}$', fontsize=16)
	plt.grid(alpha=0.5, color='grey')
	# plt.text(250, 1.51, 'CIBER '+str(lam_dict[inst])+'$\\mu$m\nObserved data\nMask '+bandstr_dict[inst]+'$<'+str(mag_lim)+'$', fontsize=18)
	plt.legend(fontsize=10, loc=4, ncol=2)
	plt.ylim(ylim)
	plt.tight_layout()
	plt.show()
	
	return fig

def plot_indiv_ps_results_fftest(lb, list_of_recovered_cls, cls_truth=None, n_skip_last = 3, mean_labels=None, return_fig=True, ciblab = 'CIB + DGL ground truth', truthlab='truth field average', ylim=[1e-3, 1e2]):
	prefac = lb*(lb+1)/(2*np.pi)
	
	if mean_labels is None:
		mean_labels = [None for x in range(len(list_of_recovered_cls))]
		
	f = plt.figure(figsize=(8,6))
	
	for i, recovered_cls in enumerate(list_of_recovered_cls):
		
		for j in range(recovered_cls.shape[0]):
			
			plt.plot(lb[:-n_skip_last], np.sqrt(prefac*np.abs(recovered_cls[j]))[:-n_skip_last], linewidth=1, marker='.', color='C'+str(i+2), alpha=0.3)
			
		plt.plot(lb[:-n_skip_last], np.sqrt(prefac*np.abs(np.mean(np.abs(recovered_cls), axis=0)))[:-n_skip_last], marker='*', label=mean_labels[i], color='C'+str(i+2), linewidth=3)

	if cls_truth is not None:
		for j in range(cls_truth.shape[0]):
			label = None
			if j==0:
				label = ciblab
			plt.plot(lb[:-n_skip_last], np.sqrt(prefac*cls_truth[j])[:-n_skip_last], color='k', alpha=0.3, linewidth=1, linestyle='dashed', marker='.', label=label)

		plt.plot(lb[:-n_skip_last], np.sqrt(prefac*np.mean(cls_truth, axis=0))[:-n_skip_last], color='k', linewidth=3, label=truthlab)

				
	plt.legend(fontsize=14)
	plt.xscale('log')
	plt.yscale('log')
	plt.ylim(ylim)
	plt.xlabel('Multipole $\\ell$', fontsize=20)
	plt.ylabel('$\\left[\\frac{\\ell(\\ell+1)}{2\\pi}C_{\\ell}\\right]^{1/2}$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=20)
	plt.tick_params(labelsize=16)
	# plt.savefig('/Users/luminatech/Downloads/input_recover_powerspec_fivefields_estimated_ff_bkg250_bl_cut_simidx'+str(simidx)+'_min_stack_ff='+str(min_stack_ff)+'.png', bbox_inches='tight')
	plt.show()
	
	if return_fig:
		return f

def make_smoothed_ciber_spitzer_maps(masked_irac_maps, tm1_meansub, tm2_regrid, mask, \
                                    xlabel = '$\\theta_x$ [deg.]', ylabel = '$\\theta_y$ [deg.]', \
                                    xticks = [0, 256, 512, 512+256, 1023], yticks = [0, 256, 512, 512+256, 1023], \
                                    xticklabs = [-1.0, -0.5, 0.0, 0.5, 1.0], yticklabs = [-1.0, -0.5, 0.0, 0.5, 1.0], \
                                    smooth_maps=True, sig_arcmin=None, sig_pix = 42, figsize=(5, 5), pixsize_arcsec=7.):
    
    bbox = dict({'facecolor':'white', 'edgecolor':'None', 'alpha':0.9})

    if smooth_maps:
        
        if sig_arcmin is not None:
            sig = sig_arcmin*60/7.
        else:
            sig = sig_pix
        
        tm2_regrid = gaussian_filter(tm2_regrid, sigma=sig)*mask
        tm1_meansub = gaussian_filter(tm1_meansub, sigma=sig)*mask
        for irac_idx in range(2):
            masked_irac_maps[irac_idx] = gaussian_filter(masked_irac_maps[irac_idx], sigma=sig)*mask

        
    f_TM1 = plot_map(tm1_meansub, cbar_label=None, cbar_fontsize=16, vmin=-5, vmax=5, \
                    xticks=xticks, yticks=yticks, xticklabs=xticklabs, yticklabs=yticklabs, \
                    xlabel=None, ylabel=ylabel, figsize=figsize, bbox_dict=bbox, text_fontsize=14, text_xpos=50, text_ypos=880,\
                        textstr='CIBER 1.1 $\\mu$m\nBootes A', cmap='bwr', return_fig=True)

    f_TM2 = plot_map(tm2_regrid, cbar_label=None, cbar_fontsize=16, vmin=-5, vmax=5, \
                        xticks=xticks, yticks=yticks, xticklabs=xticklabs, yticklabs=yticklabs, \
                        xlabel=xlabel, ylabel=ylabel, figsize=figsize, bbox_dict=bbox, text_fontsize=14, text_xpos=50, text_ypos=880,\
                            textstr='CIBER 1.8 $\\mu$m\nBootes A', cmap='bwr', return_fig=True)

    f_IRACCH1 = plot_map(masked_irac_maps[0], cbar_label='[nW m$^{-2}$ sr$^{-1}$]', vmin=-0.6, vmax=0.6, cbar_fontsize=16, \
                        xticks=xticks, yticks=yticks, xticklabs=xticklabs, yticklabs=yticklabs, \
                                xlabel=None, bbox_dict=bbox, ylabel=None, figsize=figsize, text_xpos=50, text_ypos=880, text_fontsize=14, textstr='IRAC 3.6 $\\mu$m\nBootes A', cmap='bwr', return_fig=True)

    f_IRACCH2 = plot_map(masked_irac_maps[1], cbar_label='[nW m$^{-2}$ sr$^{-1}$]', vmin=-0.6, vmax=0.6, cbar_fontsize=16, \
                        xticks=xticks, yticks=yticks, xticklabs=xticklabs, yticklabs=yticklabs, \
                         xlabel=xlabel, ylabel=None, figsize=figsize, text_xpos=50, text_ypos=880, text_fontsize=14,\
                               bbox_dict=bbox, textstr='IRAC 4.5 $\\mu$m\nBootes A', cmap='bwr', return_fig=True)


    return f_TM1, f_TM2, f_IRACCH1, f_IRACCH2

def single_panel_observed_ps_results(inst, masking_maglim, lb, observed_field_average_cl, observed_recov_ps, median_fieldav_err, all_mock_recov_ps, \
									include_dgl_ul=True, include_igl_helgason=True, include_c15_snpred=False, \
									 include_z14=False, rescale_Z14=False, fac=None, include_perfield=True, startidx=0, endidx=-1, show=True, \
									return_fig=True, zcolor='grey', xlim=[1.5e2, 1e5], ylim=[1e-2, 4e3], textpos=[2e2, 2e2], \
									obs_labels=['4th flight field average'], obs_colors=['k'], figsize=(6,5), zorders=None, bbox_to_anchor=None, legend_fs=14, \
										tick_fs=14, lab_fs=16, ncol=2, text_fs=16):
		
	lam_dict = dict({1:1.1, 2:1.8})
	lam_dict_z14 = dict({1:1.1, 2:1.6})
	bandstr_dict = dict({1:'J', 2:'H'})
	ifield_list = [4, 5, 6, 7, 8]

	ciber_field_dict = dict({4:'elat10', 5:'elat30', 6:'Bootes B', 7:'Bootes A', 8:'SWIRE'})
	prefac = lb[startidx:endidx]*(lb[startidx:endidx]+1)/(2*np.pi)


	if type(inst) != list:
		print('not identified as a list')
		inst = [inst]
		masking_maglim = [masking_maglim]
		observed_field_average_cl = [observed_field_average_cl]
		observed_recov_ps = [observed_recov_ps]
		median_fieldav_err = [median_fieldav_err]
		all_mock_recov_ps = [all_mock_recov_ps]


	if len(inst)==1:
		fig = plt.figure(figsize=figsize)
	else:
		fig, ax = plt.subplots(ncols=len(inst), nrows=1, figsize=figsize)
	

	for x in range(len(inst)):

		bandstr = bandstr_dict[inst[x]]

		plt.subplot(1, len(inst), x+1)
		if include_perfield:
			for fieldidx, ifield in enumerate(ifield_list):
#                 print('all mock recov ps shape ', np.array(all_mock_recov_ps[x]).shape)
				std_recov_mock_ps = 0.5*(np.percentile(np.array(all_mock_recov_ps[x])[:,fieldidx,:], 84, axis=0)-np.percentile(np.array(all_mock_recov_ps[x])[:,fieldidx,:], 16, axis=0))
				
				plt.errorbar(lb[startidx:endidx], prefac*observed_recov_ps[x][fieldidx,startidx:endidx], yerr=prefac*std_recov_mock_ps[startidx:endidx], label=ciber_field_dict[ifield], fmt='.', color='C'+str(fieldidx), alpha=0.3, capsize=4, markersize=10)
				
		if len(np.array(observed_field_average_cl[x]).shape)==2:
			for obs_idx, obs_cl in enumerate(observed_field_average_cl[x]):
				print((observed_field_average_cl[x][obs_idx][startidx:endidx]))
#                 print((np.median(np.abs(field_average_error[x]), axis=0)[startidx:endidx]))
				print(obs_labels[obs_idx], obs_colors[obs_idx])
				plt.errorbar(lb[startidx:endidx], prefac*(observed_field_average_cl[x][obs_idx][startidx:endidx]), yerr=(prefac)*median_fieldav_err[x][startidx:endidx],\
							 label=obs_labels[obs_idx], fmt='o', capthick=1.5, color=obs_colors[obs_idx], capsize=3, markersize=3, linewidth=1.5)
		else:     
			plt.errorbar(lb[startidx:endidx], prefac*observed_field_average_cl[x][startidx:endidx], yerr=(prefac)*median_fieldav_err[x][startidx:endidx], label=obs_labels[0], fmt='o', capthick=1.5, color='k', capsize=3, markersize=4, linewidth=2.)


		if include_dgl_ul:
		
			dgl_auto_pred = np.load(config.ciber_basepath+'data/fluctuation_data/TM'+str(inst[x])+'/dgl_tracer_maps/sfd_clean/dgl_auto_constraints_TM'+str(inst[x])+'_sfd_clean_053023.npz')
			lb_modl = dgl_auto_pred['lb_modl']
			best_ps_fit_av = dgl_auto_pred['best_ps_fit_av']
			AC_A1 = dgl_auto_pred['AC_A1']
			dAC_sq = dgl_auto_pred['dAC_sq']
			
			
			lb_extend = [4000, 6000, 10000, 20000]
			ps_extend = best_ps_fit_av[-1]*(np.array(lb_extend)/lb_modl[-1])**(-1.2)
			best_ps_fit_av = np.array(list(best_ps_fit_av) + list(ps_extend))
			lb_modl = np.array(list(lb_modl) + list(lb_extend))
			
			plt.plot(lb_modl, best_ps_fit_av*AC_A1**2, color='k', label='DGL (CSFD)', linestyle='solid')
			plt.fill_between(lb_modl, best_ps_fit_av*(AC_A1**2-dAC_sq), best_ps_fit_av*(AC_A1**2+dAC_sq), color='k', alpha=0.2)
			plt.fill_between(lb_modl, best_ps_fit_av*(AC_A1**2-2*dAC_sq), best_ps_fit_av*(AC_A1**2+2*dAC_sq), color='k', alpha=0.1)
		   
		if include_igl_helgason:
			print('inst = ', inst)
			config_dict, pscb_dict, float_param_dict, fpath_dict = return_default_cbps_dicts()
			ciber_mock_fpath = config.ciber_basepath+'data/ciber_mocks/'
			fpath_dict, list_of_dirpaths, base_path, trilegal_base_path = set_up_filepaths_cbps(fpath_dict, inst, 'test', '112022',\
																							datestr_trilegal='112022', data_type='mock', \
																						   save_fpaths=True)

			all_cl = []
			prefac_full = lb*(lb+1)/(2*np.pi)
			
			isl_igl = np.load(config.ciber_basepath+'data/cl_predictions/TM'+str(inst)+'/igl_isl_pred_mlim='+str(masking_maglim)+'_meas.npz')['isl_igl']
			plt.plot(lb, prefac_full*isl_igl, linestyle='dashdot', color='grey', label='IGL+ISL')

			
		if include_c15_snpred:
			
			c15_ccorr = np.load(config.ciber_basepath+'data/cl_predictions/color_corr_vs_'+bandstr+'_min_cosmos15.npz')
			mmin_range_cc = c15_ccorr['mmin_list']
			all_pv_c15 = c15_ccorr['all_pv_c15']
			
			lb_pred = np.array([50, 100]+list(lb))
			pf_pred = lb_pred*(lb_pred+1)/(2*np.pi)
			which_cc_match = np.where((mmin_range_cc==masking_maglim[x]))[0][0]

			plt.plot(lb_pred, (pf_pred*all_pv_c15[which_cc_match]), linestyle='dashed', color='k', label='Predicted $C_{\\ell}^{SN}$\n(IGL+ISL)')

			
		if include_z14:
			zemcov_auto = np.loadtxt(config.ciber_basepath+'/data/zemcov14_ps/ciber_'+str(lam_dict_z14[inst[x]])+'x'+str(lam_dict_z14[inst[x]])+'_dCl.txt', skiprows=8)
			zemcov_lb = zemcov_auto[:,0]
					
			if rescale_Z14:
				if inst==1:
					fac = 1.6
				else:
					fac = 2.09
					
			if type(zcolor)==list:
				zcol = zcolor[x]
			else:
				zcol = zcolor
			
			plt.plot(zemcov_lb, zemcov_auto[:,1]*fac**2, label='Zemcov+14', marker='.', color=zcol, alpha=0.3)
			plt.fill_between(zemcov_lb, (zemcov_auto[:,1]-zemcov_auto[:,2])*fac**2, (zemcov_auto[:,1]+zemcov_auto[:,3])*fac**2, color=zcol, alpha=0.15)

			plt.plot(zemcov_lb, np.abs(0.5*(zemcov_auto[:,2]+zemcov_auto[:,3])), linestyle='dashdot', color=zcol)
			plt.plot(lb[startidx:endidx], (prefac)*median_fieldav_err[0][startidx:endidx], color='k', linestyle='dashdot', label='1$\\sigma$ PS uncertainty')
#             plt.plot(zemcov_lb, 0.5*(zemcov_auto[:,3]-zemcov_auto[:,2]), linestyle='dashed', color=zcol)

		plt.xlim(xlim)
		plt.ylim(ylim)

		plt.yscale('log')
		plt.xscale('log')
		plt.tick_params(labelsize=tick_fs)
		plt.xlabel('$\\ell$', fontsize=lab_fs)
		if x==0:
			plt.ylabel('$D_{\\ell}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=lab_fs)
		plt.grid(alpha=0.5, color='grey')

		plt.text(textpos[0], textpos[1], 'CIBER '+str(lam_dict[inst[x]])+' $\\mu$m\nObserved data\nMask '+bandstr_dict[inst[x]]+'$<'+str(masking_maglim[x])+'$', fontsize=text_fs)
		
		
	handles, labels = plt.gca().get_legend_handles_labels()
	
	if include_perfield:
		
		order = list(2+np.arange(6))+[0, 1]
		
		plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], \
			  ncol=3, fontsize=legend_fs, bbox_to_anchor=bbox_to_anchor)
		
	else:
		order = [1, 3, 4, 5, 0, 2]
		plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], \
			  ncol=3, fontsize=legend_fs, bbox_to_anchor=bbox_to_anchor)
		
	if show:
		plt.show()
		
	if return_fig:
		return fig

def make_figure_cross_spec_vs_masking_magnitude_new(inst=1, cross_inst=2, maglim_J=[17.5, 18.0, 18.5, 19.0], \
											   maglim_H=[17.0, 17.5, 18.0, 18.5], observed_run_names_cross=None,\
											 return_fig=True, show=True, startidx=1, endidx=-1, colors=None, \
											 load_igl_isl_pred=False, load_snpred=False, textxpos = 250, textypos = 4e4, \
												   dl_l=True, bbox_anchor=[0.92, 1.0], include_dgl_ul=True):

	if observed_run_names_cross is None:
		observed_run_names_cross = ['ciber_cross_ciber_perquad_regrid_Jlt'+str(maglim_J[j])+'_Hlt'+str(maglim_H[j])+'_111923' for j in range(len(maglim_J))]
	
	obs_labels = ['$J<'+str(maglim_J[m])+'\\times H<'+str(maglim_H[m])+'$' for m in range(len(maglim_J))]
	obs_fieldav_cross_cl, obs_fieldav_cross_dcl = [], []    
	obs_colors = ['indigo', 'darkviolet', 'mediumorchid', 'plum']

	if load_snpred:
		snpred = np.load(config.ciber_basepath+'data/cl_predictions/snpred_color_corr_vs_mag_JH_cosmos15_nuInu_union_magcut.npz')
		all_pv_JH = snpred['all_pv_JH']
		mmin_J_pv = snpred['mmin_J_list']
		
		
		snpred_twom = np.load(config.ciber_basepath+'data/cl_predictions/snpred_vs_mag_JH_2MASS_nuInu.npz')
		all_pv_JH_twom = snpred_twom['av_pv_JH']
		mmin_J_twom = snpred_twom['mmin_J_list']

	if load_igl_isl_pred:

		all_cl_cross_pred_vs_mag = []
		idxmatch = []
		
		igl_isl_pred_fpath = config.ciber_basepath+'data/cl_predictions/COSMOS15/cross/cl_predictions_cross_J_HCH1CH2.npz'
		igl_isl_pred = np.load(igl_isl_pred_fpath)
		m_min_J_pred = igl_isl_pred['m_min_J_list']

		print('igl isl pred has shape ', igl_isl_pred['all_clx_J'].shape)
		
		all_clx_J = igl_isl_pred['all_clx_J']

		lb_pred = igl_isl_pred['lb']
		pf_pred = igl_isl_pred['pf']
		
	
	cl_base_path = config.ciber_basepath+'data/input_recovered_ps/cl_files/'
	for obs_name in observed_run_names_cross:
		cl_fpath_obs = cl_base_path+'TM'+str(inst)+'_TM'+str(cross_inst)+'_cross/cl_'+obs_name+'.npz'
		lb, observed_recov_ps, observed_recov_dcl_perfield,\
		observed_field_average_cl, observed_field_average_dcl,\
			_, _ = load_weighted_cl_file(cl_fpath_obs, cltype='cross')

		obs_fieldav_cross_cl.append(observed_field_average_cl)
		obs_fieldav_cross_dcl.append(observed_field_average_dcl)
		
	fig = plt.figure(figsize=(6,5))
	
	if dl_l:
		prefac = lb/(2*np.pi)
	else:
		prefac = lb*(lb+1)/(2*np.pi)
		
	if include_dgl_ul:
	
		dgl_auto_TM1 = np.load(config.ciber_basepath+'data/fluctuation_data/TM1/dgl_tracer_maps/sfd_clean/dgl_auto_constraints_TM1_sfd_clean_053023.npz')
		lb_modl = dgl_auto_TM1['lb_modl']
		best_ps_fit_av_TM1 = dgl_auto_TM1['best_ps_fit_av']
		
		dgl_auto_TM2 = np.load(config.ciber_basepath+'data/fluctuation_data/TM2/dgl_tracer_maps/sfd_clean/dgl_auto_constraints_TM2_sfd_clean_053023.npz')
		lb_modl = dgl_auto_TM2['lb_modl']
		best_ps_fit_av_TM2 = dgl_auto_TM2['best_ps_fit_av']
		
		AC_A1_TM1 = dgl_auto_TM1['AC_A1']
		dAC_sq_TM1 = dgl_auto_TM1['dAC_sq']
		
		AC_A1_TM2 = dgl_auto_TM2['AC_A1']
		dAC_sq_TM2 = dgl_auto_TM2['dAC_sq']
		
		print('AC A1 values:', AC_A1_TM1, AC_A1_TM2)
		print('dAC A1 values:', np.sqrt(dAC_sq_TM1), np.sqrt(dAC_sq_TM2))
		AC_A1_cross = (AC_A1_TM1*AC_A1_TM2)
		
		best_ps_fit_av = np.sqrt(best_ps_fit_av_TM1*best_ps_fit_av_TM2)*AC_A1_cross
		
		lb_extend = [4000, 6000, 10000, 20000]
		
		ps_extend = best_ps_fit_av[-1]*(np.array(lb_extend)/lb_modl[-1])**(-1.2)
		
		best_ps_fit_av = np.array(list(best_ps_fit_av) + list(ps_extend))
		
		lb_modl = np.array(list(lb_modl) + list(lb_extend))
				
		if dl_l:
			best_ps_fit_av /= (lb_modl+1)
		
		plt.plot(lb_modl, best_ps_fit_av, color='k', label='DGL (CSFD)', linestyle='solid')
		plt.fill_between(lb_modl, best_ps_fit_av*(1-0.5*(dAC_sq_TM1+dAC_sq_TM2)/AC_A1_cross), best_ps_fit_av*(1+0.5*(dAC_sq_TM1+dAC_sq_TM2)/AC_A1_cross), color='k', alpha=0.2)
		plt.fill_between(lb_modl, best_ps_fit_av*(1-(dAC_sq_TM1+dAC_sq_TM2)/AC_A1_cross), best_ps_fit_av*(1+(dAC_sq_TM1+dAC_sq_TM2)/AC_A1_cross), color='k', alpha=0.1)
		
		
	for m, maglim in enumerate(maglim_J):
		
		if colors is not None:
			color = colors[m]
		else:
			color = 'C'+str(m+1)


		if load_snpred:

			if m==len(maglim_J)-1:
				snlab = 'Predicted $C_{\ell}^{SN}$ (IGL+ISL)'
			else:
				snlab = None
			if maglim < 15.0:                
				which_pv = np.where((mmin_J_twom==maglim))[0][0]
				pv_cross = all_pv_JH_twom[which_pv]                
			else:
				which_pv = np.where((mmin_J_pv==maglim))[0][0]
				pv_cross = all_pv_JH[which_pv]

			if pv_cross is not None:
				lb_sn = np.array([50., 100.]+list(lb))
				pf_sn = lb_sn*(lb_sn+1)/(2*np.pi)
				
				if dl_l:
					plt.plot(lb_sn, pf_sn*pv_cross/(lb_sn+1), color=color, linestyle='dashed', alpha=0.4, label=snlab)
				else:
					plt.plot(lb_sn, pf_sn*pv_cross, color=color, linestyle='dashed', alpha=0.4)

		
		if load_igl_isl_pred:
			if maglim in m_min_J_pred and maglim > 16.0:
				idx = np.where((m_min_J_pred==maglim))[0][0]
				print(idx, m_min_J_pred[idx], maglim)
				plt.plot(lb_pred, pf_pred*all_clx_J[idx, 0], linestyle='dashed', color=color, alpha=0.4)

		whichpos = np.where((lb >= lb[startidx])*(lb <= lb[endidx])*(obs_fieldav_cross_cl[m]>0))[0]
		whichneg = np.where((lb >= lb[startidx])*(lb <= lb[endidx])*(obs_fieldav_cross_cl[m]<0))[0]

		plt.errorbar(lb[whichpos], (prefac*obs_fieldav_cross_cl[m])[whichpos], yerr=(prefac*obs_fieldav_cross_dcl[m])[whichpos], label=obs_labels[m], fmt='o', capthick=1.5, color=color, capsize=3, markersize=4, linewidth=2.)
		plt.errorbar(lb[whichneg], np.abs((prefac*obs_fieldav_cross_cl[m])[whichneg]), yerr=(prefac*obs_fieldav_cross_dcl[m])[whichneg], mfc='white', fmt='o', capthick=1.5, color=color, capsize=3, markersize=4, linewidth=2.)

	plt.yscale('log')
	plt.xscale('log')
	plt.tick_params(labelsize=14)
	plt.xlabel('$\\ell$', fontsize=16)
	
	
	if dl_l:
		textypos = 5
		plt.ylabel('$D_{\\ell}/\\ell$', fontsize=18)
		plt.ylim(3e-5, 5e1)
		plt.text(1.05e4, 6e-5, '- - COSMOS 2015\n  $C_{\\ell}^{SN}$ predictions', color=colors[m], fontsize=13)

	else:
		plt.ylabel('$D_{\\ell}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=18)
		plt.ylim(1e-1, 1e6)
		plt.text(9e3, 4e-1, '- - COSMOS 2015\n  $C_{\\ell}^{SN}$ predictions', color=colors[m], fontsize=14)


	plt.xlim(1.5e2, 1e5)
	plt.grid(alpha=0.5, color='grey')

	plt.text(textxpos, textypos, 'CIBER 1.1 $\\mu$m $\\times$ 1.8 $\\mu$m\nObserved data', fontsize=16)

	bbox_dict = dict({'facecolor':'white', 'alpha':0.8, 'edgecolor':'None', 'pad':0.})

	handles, labels = plt.gca().get_legend_handles_labels()
	
	print(len(handles), len(colors))
	
	order = [2+x for x in np.arange(len(colors))]+[0, 1]

	plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], \
		  fontsize=9, loc=4, ncol=3, framealpha=1., bbox_to_anchor=bbox_anchor)

	if show:
		plt.show()
	if return_fig:
		return fig

def make_figure_cross_corrcoeff_ciber_ciber_vs_mag(maglim_J = [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 17.5, 18.0], show=True, return_fig=True, \
												  verbose=False, observed_run_names_cross=None, observed_run_names_auto=None, alpha=0.5, ylim=None, \
												  startidx=1, endidx=-1, bbox_to_anchor =[1.0, 1.36], yticks=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25]):
	
	if observed_run_names_cross is None:
		observed_run_names_cross = ['ciber_cross_ciber_perquad_regrid_Jlt'+str(maglim_J[j])+'_Hlt'+str(maglim_J[j]-0.5)+'_111923' for j in range(len(maglim_J))]

	all_r_TM, all_sigma_r_TM = [], []
	
	colors = plt.cm.PuRd(np.linspace(0.2, 1,len(maglim_J)))

	for o, obs_name_AB in enumerate(observed_run_names_cross):
	
		maglim_H = maglim_J[o] - 0.5
		
		if observed_run_names_auto is not None:
			obs_name_A = observed_run_names_auto[o]
		else:
			obs_name_A = 'observed_Jlt'+str(maglim_J[o])+'_Hlt'+str(maglim_H)+'_111323_ukdebias' # union mask        
		
		obs_name_B = obs_name_A
	
		print(obs_name_A, obs_name_B, obs_name_AB)
	
		lb, r_TM, sigma_r_TM, obs_fieldav_cls, obs_fieldav_dcls = ciber_ciber_rl_coefficient(obs_name_A, obs_name_B, obs_name_AB)
		all_r_TM.append(r_TM)
		all_sigma_r_TM.append(sigma_r_TM)
		
		if verbose:
			print('r_TM:', r_TM)
			print('sigma_r_TM:', sigma_r_TM)
		
	fig = plt.figure(figsize=(5, 4))

	for obs_idx in range(len(all_r_TM)):

		lbmask_science = (lb < 2000)
		weights = 1./(all_sigma_r_TM[obs_idx][lbmask_science])**2
		weighted_average = np.sum(weights*all_r_TM[obs_idx][lbmask_science])/np.sum(weights)
		weighted_variance = 1./np.sum(weights)
		
		if verbose:
			print('weighted variance for lb < 2000:', weighted_variance)

		plt.scatter(lb[startidx:endidx], all_r_TM[obs_idx][startidx:endidx], color=colors[obs_idx], \
					label='$J<'+str(maglim_J[obs_idx])+'\\times H<$'+str(maglim_J[obs_idx]-0.5), s=8)

		plt.errorbar(lb[startidx:endidx], all_r_TM[obs_idx][startidx:endidx], yerr=all_sigma_r_TM[obs_idx][startidx:endidx], fmt='o', capsize=2.5, color=colors[obs_idx], \
					markersize=4, capthick=1.5, alpha=alpha)
	plt.xscale('log')
	if ylim is not None:
		plt.ylim(ylim[0], ylim[1])
	else:
		plt.ylim(-0.1, 1.5)
	bbox_dict = dict({'facecolor':'white', 'alpha':0.9, 'edgecolor':'k', 'linewidth':0.5})
	plt.xlabel('$\\ell$', fontsize=14)
	plt.ylabel('$r_{\\ell} = C_{\\ell}^{1.1\\times1.8}/\\sqrt{C_{\\ell}^{1.1}C_{\\ell}^{1.8}}$', fontsize=12)
	plt.tick_params(labelsize=11)

	ticks = ['' for x in range(len(yticks))]
	if obs_idx==1 or obs_idx==3:
		plt.yticks(yticks, ticks)
	plt.grid(alpha=0.3)
	
	plt.legend(ncol=2, bbox_to_anchor=bbox_to_anchor)

	if show:
		plt.show()
	if return_fig:
		return fig, all_r_TM, all_sigma_r_TM, lb

def plot_ciber_x_ciber_ps(cbps, ifield_list, lb, all_cl1d_obs, all_nl1d_unc, field_weights, maglim_J,\
						  startidx=1, endidx=-1, return_fig=True, flatidx=7, cl_prediction=None, \
						 plot_cl_errs=False, snpred_cross=None, ylim=[1e-2, 1e4], \
								 textxpos = 200, textypos = 4e2, include_legend=True, include_dgl_ul=False, \
							 markersize_alpha=0.5, xlim=[1.5e2, 1e5]):
	
	
	maglim_H = maglim_J - 0.5
	
	f = plt.figure(figsize=(6,5))

	prefac = lb*(lb+1)/(2*np.pi)
	
	for fieldidx, ifield in enumerate(ifield_list):
		
		whichpos = (lb >= lb[startidx])*(lb <= lb[endidx])*(all_cl1d_obs[fieldidx] > 0)
		whichneg = (lb >= lb[startidx])*(lb <= lb[endidx])*(all_cl1d_obs[fieldidx] < 0)
		
#         plt.errorbar(lb[whichpos], (prefac*all_cl1d_obs[fieldidx])[whichpos], yerr=(prefac*all_nl1d_unc[fieldidx])[whichpos], label=cbps.ciber_field_dict[ifield], fmt='o', color='C'+str(fieldidx), alpha=0.3, capsize=4, markersize=10)
#         plt.errorbar(lb[whichneg], (prefac*all_cl1d_obs[fieldidx])[whichneg], yerr=(prefac*all_nl1d_unc[fieldidx])[whichneg], label=cbps.ciber_field_dict[ifield], fmt='o', mfc='white', color='C'+str(fieldidx), alpha=0.3, capsize=4, markersize=10)

		plt.errorbar(lb[startidx:endidx], (prefac*all_cl1d_obs[fieldidx])[startidx:endidx], yerr=(prefac*all_nl1d_unc[fieldidx])[startidx:endidx], label=cbps.ciber_field_dict[ifield], fmt='.', color='C'+str(fieldidx), alpha=markersize_alpha, capsize=3, markersize=8)
		
		if plot_cl_errs:
			plt.plot(lb[startidx:endidx], (prefac*all_nl1d_unc[fieldidx])[startidx:endidx], linestyle='dashed', color='C'+str(fieldidx))

	mean_cl_cross = np.mean(all_cl1d_obs, axis=0)
	std_cl_cross = np.std(all_cl1d_obs, axis=0)/np.sqrt(5)

	weighted_cross_average_cl, weighted_cross_average_dcl = compute_weighted_cl(all_cl1d_obs, field_weights)
	weighted_cross_average_cl[:flatidx] = mean_cl_cross[:flatidx]
	weighted_cross_average_dcl[:flatidx] = std_cl_cross[:flatidx]
	
	whichpos = np.where((lb >= lb[startidx])*(lb <= lb[endidx])*(weighted_cross_average_cl > 0))[0]
	whichneg = np.where((lb >= lb[startidx])*(lb <= lb[endidx])*(weighted_cross_average_cl < 0))[0]
	
	plt.errorbar(lb[whichpos], (prefac*weighted_cross_average_cl)[whichpos], yerr=(prefac*weighted_cross_average_dcl)[whichpos], label='Field average', fmt='o', capthick=1.5, color='k', capsize=3, markersize=4, linewidth=1.5, zorder=10)
	plt.errorbar(lb[whichneg], np.abs(prefac*weighted_cross_average_cl)[whichneg], yerr=2*(prefac*weighted_cross_average_dcl-np.abs(weighted_cross_average_cl))[whichneg], lolims=False, uplims=whichneg, color='k', capsize=3, capthick=1.5, linewidth=1.5, marker='v')

	if snpred_cross is not None:
		lb_sn = np.array([50., 100.]+list(lb))
		pf_sn = lb_sn*(lb_sn+1)/(2*np.pi)
		plt.plot(lb_sn, pf_sn*snpred_cross, color='k', linestyle='dashed', label='IGL+ISL prediction')
		
	
	if cl_prediction is not None:
		plt.plot(lb[startidx:endidx], (prefac*cl_prediction)[startidx:endidx], color='Grey', linestyle='dashed', label='IGL+ISL prediction')
	
	if plot_cl_errs:
		plt.plot(lb[startidx:endidx], (prefac*weighted_cross_average_dcl)[startidx:endidx], linestyle='dashed', color='k')
	
	
	plt.yscale('log')
	plt.xscale('log')
	plt.tick_params(labelsize=14)
	plt.xlabel('$\\ell$', fontsize=16)
	plt.ylabel('$D_{\\ell}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=18)
	plt.grid(alpha=0.5, color='grey')
	
	plt.xlim(xlim)
	plt.ylim(ylim)
	plt.text(textxpos, textypos, 'CIBER 1.1 $\\mu$m $\\times$ 1.8 $\\mu$m\nObserved data\nMask $J<'+str(maglim_J)+'$ and $H<'+str(maglim_H)+'$', fontsize=16)
			
	if include_dgl_ul:
	
		dgl_auto_TM1 = np.load(config.ciber_basepath+'data/fluctuation_data/TM1/dgl_tracer_maps/sfd_clean/dgl_auto_constraints_TM1_sfd_clean_053023.npz')
		lb_modl = dgl_auto_TM1['lb_modl']
		best_ps_fit_av_TM1 = dgl_auto_TM1['best_ps_fit_av']
		
		dgl_auto_TM2 = np.load(config.ciber_basepath+'data/fluctuation_data/TM2/dgl_tracer_maps/sfd_clean/dgl_auto_constraints_TM2_sfd_clean_053023.npz')
		lb_modl = dgl_auto_TM2['lb_modl']
		best_ps_fit_av_TM2 = dgl_auto_TM2['best_ps_fit_av']
		
		AC_A1_TM1 = dgl_auto_TM1['AC_A1']
		dAC_sq_TM1 = dgl_auto_TM1['dAC_sq']
		
		AC_A1_TM2 = dgl_auto_TM2['AC_A1']
		dAC_sq_TM2 = dgl_auto_TM2['dAC_sq']
		
		AC_A1_cross = (AC_A1_TM1*AC_A1_TM2)
		best_ps_fit_av = np.sqrt(best_ps_fit_av_TM1*best_ps_fit_av_TM2)*AC_A1_cross
		lb_extend = [4000, 6000, 10000, 20000]
		ps_extend = best_ps_fit_av[-1]*(np.array(lb_extend)/lb_modl[-1])**(-1.2)
		best_ps_fit_av = np.array(list(best_ps_fit_av) + list(ps_extend))
		lb_modl = np.array(list(lb_modl) + list(lb_extend))
		
		
		plt.plot(lb_modl, best_ps_fit_av, color='k', label='DGL (CSFD)', linestyle='solid')
		plt.fill_between(lb_modl, best_ps_fit_av*(1-0.5*(dAC_sq_TM1+dAC_sq_TM2)/AC_A1_cross), best_ps_fit_av*(1+0.5*(dAC_sq_TM1+dAC_sq_TM2)/AC_A1_cross), color='k', alpha=0.2)
		plt.fill_between(lb_modl, best_ps_fit_av*(1-(dAC_sq_TM1+dAC_sq_TM2)/AC_A1_cross), best_ps_fit_av*(1+(dAC_sq_TM1+dAC_sq_TM2)/AC_A1_cross), color='k', alpha=0.1)
			
	if include_legend:
		handles, labels = plt.gca().get_legend_handles_labels()
		print(labels)
		order = [2,3,4,5,6,7,0, 1]


		plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=10,\
						   ncol=2, loc=4, framealpha=1., bbox_to_anchor=[0.85, 1.0])

	plt.show()
	
	if return_fig:
		return f, weighted_cross_average_cl, weighted_cross_average_dcl
	else:
		return weighted_cross_average_cl, weighted_cross_average_dcl


def plot_bandpowers_vs_magnitude(cbps, inst, mag_lims, prefac_binned, binned_obs_fieldav, binned_obs_fieldav_dcl, igl_isl_vs_maglim, igl_vs_maglim=None,\
								 nbp=12, nrow=3, ncol=4, mode='diff', idx0=0, return_fig=True, show=True, \
								xticks=[11, 13, 15, 17, 19], binned_obs_fieldav_list=None, binned_obs_fieldav_dcl_list=None,\
								 labels=None, colors=None, obs_color='b', igl_isl_color='grey', obs_label=None):
			
	if inst==1:
		if mode=='diff':
			textypos = 20
		else:
			textypos = 100
		bandstr = '$J_{lim}$'
	else:
		if mode=='diff':
			textypos = 20
		else:
			
			textypos = 100
			
		bandstr = '$H_{lim}$'
		
	mag_labels = []
	
	if mode=='diff':
		for x in range(igl_isl_vs_maglim.shape[0]-1):        
			maglabel = '['+str(mag_lims[x])+'] - ['+str(mag_lims[x+1])+']'
			mag_labels.append(maglabel)
		
		xticks = mag_lims[1:]
		textxpos = mag_lims[1]
	else:
		textxpos = mag_lims[0]
		
		if binned_obs_fieldav_list is not None:
			textxpos -= 0.5

	fig = plt.figure(figsize=(9, 6))
	
	for idx in range(nbp):
		
		plt.subplot(nrow,ncol,idx+1)
		plt.text(textxpos, textypos, str(int(cbps.Mkk_obj.binl[idx0+2*idx]))+'$<\\ell \\leq$'+str(int(cbps.Mkk_obj.binl[idx0+2*(idx+1)])), fontsize=10)

		if mode=='diff':
			delta_obs_fieldav = binned_obs_fieldav[:-1,idx]-binned_obs_fieldav[1:,idx]
			delta_igl_isl = igl_isl_vs_maglim[:-1,idx]-igl_isl_vs_maglim[1:,idx]
			
			plt.errorbar(mag_lims[1:], prefac_binned[idx]*delta_obs_fieldav, color=obs_color, label=obs_label, fmt='o-', capsize=2, markersize=3, zorder=10)
			plt.plot(mag_lims[1:], prefac_binned[idx]*delta_igl_isl, color=igl_isl_color, marker='.')
			
			if igl_vs_maglim is not None:
				delta_igl = igl_vs_maglim[:-1,idx]-igl_vs_maglim[1:,idx]
				plt.plot(mag_lims[1:], prefac_binned[idx]*delta_igl, color='r', marker='.')

		else:
			
			if binned_obs_fieldav_list is not None:
				
				if colors is None:
					colors = ['C'+str(x) for x in range(len(binned_obs_fieldav_list))]
					
				for x in range(len(binned_obs_fieldav_list)):
					plt.errorbar(mag_lims, prefac_binned[idx]*binned_obs_fieldav_list[x][:,idx], yerr=prefac_binned[idx]*binned_obs_fieldav_dcl_list[x][:,idx], color=colors[x], label=labels[x], fmt='o-', capsize=2, markersize=3, zorder=10)
					
			else:
				plt.errorbar(mag_lims, prefac_binned[idx]*binned_obs_fieldav[:,idx], yerr=prefac_binned[idx]*binned_obs_fieldav_dcl[:,idx], label=obs_label, color=obs_color, fmt='o-', capsize=2, markersize=3, zorder=10)
				plt.plot(mag_lims, prefac_binned[idx]*igl_isl_vs_maglim[:,idx], color=igl_isl_color, marker='.', label='IGL + ISL prediction')
				
				if igl_vs_maglim is not None:
					plt.plot(mag_lims, prefac_binned[idx]*igl_vs_maglim[:,idx], color='r', marker='.')
			
		plt.yscale('log')
		
		if idx==0:
			plt.legend(bbox_to_anchor=[3.8, 1.5], fontsize=12, ncol=3)
		
		
		if idx > 7:
			
			if mode!='diff':
				plt.xlabel(bandstr+' [Vega]', fontsize=12)
				plt.xticks(xticks, xticks)
			else:
				plt.xticks(xticks, mag_labels, rotation='vertical')
			plt.tick_params(labelsize=9)
		else:
			plt.xticks(xticks, ['' for x in range(len(xticks))])

		if idx in [0, 4, 8]:
			if mode=='diff':
				plt.ylabel('$\Delta(D_{\\ell}/\\ell)$', fontsize=12)
			else:
				plt.ylabel('$D_{\\ell}/\\ell$', fontsize=12)
			
		else:
			plt.yticks([1e-3, 1e-1, 1e1, 1e3], ['', '', '', ''])
			
		plt.grid(alpha=0.5)

		if mode=='diff':
			plt.xlim(xticks[0]-0.25, xticks[-1]+0.25)
			if inst==1:
				plt.ylim(5e-5, 2e2)
			else:
				plt.ylim(5e-6, 1e2)
		else:
			if inst==1:
				plt.ylim(5e-5, 1e3)
			else:
				plt.ylim(5e-5, 1e3)
			
	if show:
		plt.show()
			
	if return_fig:
		return fig


def plot_nframe_difference(cl_diffs_flight, cl_diffs_shotnoise, ifield_list, cl_elat30=None, ifield_elat30=5, nframe=10, show=True, return_fig=True, \
						  titlefontsize=20):
	
	labfs = 18
	legendfs = 12
	tickfs = 14
	
	cl_diffs_flight = np.array(cl_diffs_flight)
	cl_diffs_shotnoise = np.array(cl_diffs_shotnoise)
	clav = np.mean(cl_diffs_flight-cl_diffs_shotnoise, axis=0)
	
	f = plt.figure(figsize=(12,5))
	
	plt.subplot(1,2,1)
	
	# show Dl power spectra
	plt.title(str(nframe)+'-frame differences', fontsize=titlefontsize)
	plt.errorbar(lb, prefac*np.mean(cl_diffs_flight-cl_diffs_shotnoise, axis=0), yerr=prefac*np.std(cl_diffs_flight-cl_diffs_shotnoise, axis=0),\
			linewidth=1.5, color='k', linestyle='solid', capsize=3, capthick=2, label='Field average\n(no elat30)')
	
	if cl_elat30 is not None:
		plt.errorbar(lb, prefac*cl_elat30, label='elat30 noise model', linestyle='dashed', color='C1', linewidth=2, zorder=2)
	
	plot_colors = ['C0', 'C2', 'C3', 'C4']

	for i, ifield in enumerate(ifield_list):
		snlab, fsnlab = None, ''
		if i==0:
			snlab = str(nframe)+'-frame photon noise'
			fsnlab = 'Flight diff. - $N_{\\ell}^{phot}$\n'

		fsnlab += cbps.ciber_field_dict[ifield]
		zorder=0
	
		if ifield==5:
			zorder=10
			
		plt.plot(lb, prefac*(cl_diffs_flight[i]-cl_diffs_shotnoise[i]), zorder=zorder, linewidth=2, color=plot_colors[i], linestyle='solid', label=fsnlab)

	plt.yscale('log')
	plt.xscale('log')
	plt.legend(fontsize=legendfs, loc=4)
	plt.xlabel('Multipole $\\ell$', fontsize=labfs)
	plt.ylabel('$\\ell(\\ell+1)C_{\\ell}/2\\pi$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=labfs)
	plt.tick_params(labelsize=tickfs)
	
	# show ratio with respect to non-elat30 fields (nframe=10) or full five-field average (nframe=5)

	plt.subplot(1,2,2)
	plt.title(str(nframe)+'-frame noise spectra', fontsize=titlefontsize)
	for i, ifield in enumerate(ifield_list):

		plt.plot(lb, (cl_diffs_flight[i]-cl_diffs_shotnoise[i])/clav, label=cbps.ciber_field_dict[ifield], color=plot_colors[i], linewidth=2)
	
	if cl_elat30 is not None:
		plt.plot(lb, cl_elat30/clav, linewidth=2, label='elat30', linestyle='dashed', color='C1', zorder=2)
	plt.axhline(1.0, linestyle='dashed', color='k', label='Field average\n(no elat30)')
		
	plt.ylabel('$N_{\\ell}^{read}/\\langle N_{\\ell}^{read}\\rangle$', fontsize=labfs)
	plt.xscale('log')
	plt.xlabel('Multipole $\\ell$', fontsize=labfs)
	plt.tick_params(labelsize=tickfs)
	
	plt.tight_layout()

	
	if show:
		plt.show()
	if return_fig:
		return f
		


def plot_g1_hist(g1facs, mask=None, return_fig=True, show=True, title=None):
	if mask is not None:
		g1facs = g1facs[mask]
		
	std_g1 = np.std(g1facs)/np.sqrt(len(g1facs))
	f = plt.figure(figsize=(6,5))
	if title is not None:
		plt.title(title, fontsize=18)
	plt.hist(g1facs)
	plt.axvline(np.median(g1facs), label='Median $G_1$ = '+str(np.round(np.median(g1facs), 3))+'$\\pm$'+str(np.round(std_g1, 3)), linestyle='dashed', color='k', linewidth=2)
	plt.axvline(np.mean(g1facs), label='Mean $G_1$ = '+str(np.round(np.mean(g1facs), 3))+'$\\pm$'+str(np.round(std_g1, 3)), linestyle='dashed', color='r', linewidth=2)
	plt.xlabel('$G_1$ [(e-/s)/(ADU/fr)]', fontsize=16)
	plt.ylabel('N', fontsize=16)
	plt.legend(fontsize=14)
	plt.tick_params(labelsize=14)
	
	if show:
		plt.show()
	
	if return_fig:
		return f
		

def plot_noise_modl_val_flightdiff(lb, N_ells_modl, N_ell_flight, shot_noise=None, title=None, ifield=None, inst=None, show=True, return_fig=True):
	
	''' 
	
	Parameters
	----------
	
	lb : 
	N_ells_modl : 'np.array' of type 'float' and shape (nsims, n_ps_bins). Noise model power spectra
	N_ell_flight : 'np.array' of type 'float' and shape (n_ps_bins). Flight half difference power spectrum
	ifield (optional) 'str'.
		Default is None.
	inst (optional) : 'str'.
		Default is None.
		
	show (optional) : 'bool'. 
		Default is True.
	return_fig (optional) : 'bool'.
		Default is True.
	
	
	Returns
	-------
	
	f (optional): Matplotlib figure
	
	
	'''
	ciber_field_dict = dict({4:'elat10', 5:'elat30',6:'BootesB', 7:'BootesA', 8:'SWIRE'})

	# compute the mean and scatter of noise model power spectra realizations
	
	mean_N_ell_modl = np.mean(N_ells_modl, axis=0)
	std_N_ell_modl = np.std(N_ells_modl, axis=0)
	
	print('mean Nell shape', mean_N_ell_modl.shape)
	print('std Nell shape', std_N_ell_modl.shape)
	print(shot_noise)
	print(N_ell_flight.shape)

	
	prefac = lb*(lb+1)/(2*np.pi)
	
	f = plt.figure(figsize=(8,6))
	if title is None:
		if inst is not None and ifield is not None:
			title = 'TM'+str(inst)+' , '+ciber_field_dict[ifield]
			
	plt.title(title, fontsize=18)
	
	plt.plot(lb, prefac*N_ell_flight, label='Flight difference', color='r', linewidth=2, marker='.') 
	if shot_noise is not None:
		plt.plot(lb, prefac*shot_noise, label='Photon noise', color='C1', linestyle='dashed')
	plt.errorbar(lb, prefac*(mean_N_ell_modl), yerr=prefac*std_N_ell_modl, label='Read noise model', color='k', capsize=5, linewidth=2)
	
	plt.legend(fontsize=14)
	plt.xscale('log')
	plt.yscale('log')
	plt.tick_params(labelsize=14)
	
	if show:
		plt.show()
		
	if return_fig:
		return f

def plot_cumulative_numbercounts(cat, mag_idx, field=None, df=True, m_min=18, m_max=30, vlines=None, hlines=None, \
								nbin=100, label=None, pdf_or_png='pdf', band_label='W1', show=True, \
								  return_fig=True, magsys='AB'):
	f = plt.figure()
	title = 'Cumulative number count distribution'
	if field is not None:
		title += ' -- '+str(field)
	plt.title(title, fontsize=14)
	magspace = np.linspace(m_min, m_max, nbin)
	if df:
		cdf_nm = np.array([len(cat.loc[cat[mag_idx]<x]) for x in magspace])
	else:
		print(len(cat))
		cdf_nm = np.array([len(cat[(cat[:,mag_idx] < x)]) for x in magspace])
	plt.plot(magspace, cdf_nm, label=label)
	if vlines is not None:
		for vline in vlines:
			plt.axvline(vline, linestyle='dashed')
	if hlines is not None:
		for hline in hlines:
			plt.axhline(hline, linestyle='dashed')
	plt.yscale('log')
	plt.ylabel('N['+band_label+' < magnitude]', fontsize=14)
	plt.xlabel(band_label+' magnitude ('+magsys+')', fontsize=14)
	plt.legend()
	if show:
		plt.show()
	if return_fig:
		return f

def plot_number_counts_uk(df, binedges=None, bands=['yAB', 'jAB', 'hAB', 'kAB'], magstr='', add_title_str='', return_fig=False):
	if binedges is None:
		binedges = np.arange(7,23,0.5)

	f = plt.figure()
	plt.title(magstr+add_title_str, fontsize=16)
	bins = (binedges[:-1] + binedges[1:])/2
	
	for band in bands:
		d = df[band+magstr]
		shist,_ = np.histogram(d, bins = binedges)
		plt.plot(bins,shist / (binedges[1]-binedges[0]) / 4,'o-',label=band)

	plt.yscale('log')
	plt.xlabel('$m_{AB}$', fontsize=14)
	plt.ylabel('counts / mag / deg$^2$', fontsize=14)
	plt.legend()
	plt.grid()
	plt.show()
	if return_fig:
		return f

def plot_number_counts(df, binedges=None, bands=['g', 'r', 'i', 'z', 'y'], magstr='MeanPSFMag', add_title_str='', return_fig=False):
	if binedges is None:
		binedges = np.arange(7,23,0.5)

	f = plt.figure()
	plt.title(magstr+add_title_str, fontsize=16)
	bins = (binedges[:-1] + binedges[1:])/2
	
	for band in bands:
		d = df[band+magstr]
		shist,_ = np.histogram(d, bins = binedges)
		plt.plot(bins,shist / (binedges[1]-binedges[0]) / 4,'o-',label=band)

	plt.yscale('log')
	plt.xlabel('$m_{AB}$', fontsize=14)
	plt.ylabel('counts / mag / deg$^2$', fontsize=14)
	plt.legend()
	plt.grid()
	plt.show()
	if return_fig:
		return f

def plot_dm_powerspec(show=True):
	
	mass_function = MassFunction(z=0., dlog10m=0.02)
	
	f = plt.figure()
	plt.title('Halo Model Dark Matter Power Spectrum, z=0')
	# plt.plot(mass_function.k, mass_function.nonlinear_delta_k, label='halofit')
	plt.plot(mass_function.k, mass_function.nonlinear_power, label='halofit')
	plt.yscale('log')
	plt.xscale('log')
	plt.legend()
	plt.ylabel('$P(k)$ [$h^{-3}$ $Mpc^3$]', fontsize=14)
	plt.xlabel('$k$ [h $Mpc^{-1}$]', fontsize=14)
	plt.ylim(1e-4, 1e5)
	plt.xlim(1e-3, 1e3)
	if show:
		plt.show()
	
	return f

def plot_map(image, figsize=(8,8), title=None, titlefontsize=16, xlabel='x [pix]', ylabel='y [pix]',\
			 x0=None, x1=None, y0=None, y1=None, lopct=5, hipct=99, xticks=None, yticks=None, xticklabs=None, yticklabs=None, \
			 return_fig=False, show=True, nanpct=True, cl2d=False, cmap='viridis', noxticks=False, noyticks=False, \
			 cbar_label=None, norm=None, vmin=None, vmax=None, scatter_xs=None, scatter_ys=None, scatter_marker='x', scatter_color='r', \
			 interpolation='none', cbar_fontsize=14, xylabel_fontsize=16, tick_fontsize=14, \
			 textstr=None, text_xpos=None, text_ypos=None, bbox_dict=None, text_fontsize=16, origin='lower'):

	f = plt.figure(figsize=figsize)



	if vmin is None:
		vmin = np.nanpercentile(image, lopct)
	if vmax is None:
		vmax = np.nanpercentile(image, hipct)

	if title is not None:
	
		plt.title(title, fontsize=titlefontsize)

	print('min max of image in plot map are ', np.min(image), np.max(image))
	plt.imshow(image, vmin=vmin, vmax=vmax, cmap=cmap, interpolation=interpolation, origin=origin, norm=norm)

	# if nanpct:
	#     plt.imshow(image, vmin=np.nanpercentile(image, lopct), vmax=np.nanpercentile(image, hipct), cmap=cmap, interpolation='None', origin='lower', norm=norm)
	# else:
	#     plt.imshow(image, cmap=cmap, origin='lower', interpolation='none', norm=norm)
	cbar = plt.colorbar(fraction=0.046, pad=0.04)
	cbar.ax.tick_params(labelsize=14)
	if cbar_label is not None:
		cbar.set_label(cbar_label, fontsize=cbar_fontsize)

	if scatter_xs is not None and scatter_ys is not None:
		plt.scatter(scatter_xs, scatter_ys, marker=scatter_marker, color=scatter_color)
	if x0 is not None and x1 is not None:
		plt.xlim(x0, x1)
		plt.ylim(y0, y1)
		
	if cl2d:
		plt.xlabel('$\\ell_x$', fontsize=xylabel_fontsize)
		plt.ylabel('$\\ell_y$', fontsize=xylabel_fontsize)
	else:
		plt.xlabel(xlabel, fontsize=xylabel_fontsize)
		plt.ylabel(ylabel, fontsize=xylabel_fontsize)

	if xticks is not None:
		plt.xticks(xticks, xticklabs)
	if yticks is not None:
		plt.yticks(yticks, yticklabs)

	if noxticks:
		plt.xticks([], [])
	if noyticks:
		plt.yticks([], [])

	if textstr is not None:
		if bbox_dict is None:
			bbox_dict = dict({'facecolor':'white', 'edgecolor':'None', 'alpha':0.7})
		plt.text(text_xpos, text_ypos, textstr, fontsize=text_fontsize, bbox=bbox_dict)
		
	plt.tick_params(labelsize=tick_fontsize)

	if show:
		plt.show()
	if return_fig:
		return f

def compute_all_intermediate_power_spectra(lb, cls_inter, inter_labels, signal_ps=None, fieldidx=0, show=True, return_fig=True):
	
	f = plt.figure(figsize=(8, 6))

	for interidx, inter_label in enumerate(inter_labels):
		prefac = lb*(lb+1)/(2*np.pi)
		plt.plot(lb, prefac*cls_inter[interidx][fieldidx], marker='.',  label=inter_labels[interidx], color='C'+str(interidx%10))

	if signal_ps is not None:
		plt.plot(lb, prefac*signal_ps[fieldidx], marker='*', label='Signal PS', color='k', zorder=2)
	plt.legend(fontsize=14, bbox_to_anchor=[1.01, 1.3], ncol=3, loc=1)
	plt.ylabel('$D_{\\ell}$', fontsize=18)
	plt.xlabel('Multipole $\\ell$', fontsize=18)
	plt.tick_params(labelsize=16)
	plt.text(200, 3e3, 'ifield='+str(fieldidx+4)+' (TM2)', color='C2', fontsize=24)
	plt.xscale('log')
	plt.yscale('log')
	plt.grid(alpha=0.5, color='grey')
	plt.ylim(1e-1, 1e4)
	# plt.savefig('/Users/luminatech/Downloads/ps_step_by_step_bootesB_tm1_072122.png', bbox_inches='tight')
	if show:
		plt.show()
	if return_fig:
		return f


def plot_means_vs_vars(m_o_m, v_o_d, timestr_cut=None, var_frac_thresh=None, xlim=None, ylim=None, all_set_numbers_list=None, all_timestr_list=None,\
					  nfr=5, inst=1, fit_line=True, itersigma=4.0, niter=5):
	
	mask = [True for x in range(len(m_o_m))]
	
	from astropy.stats import sigma_clip
	from astropy.modeling import models, fitting
	
	if timestr_cut is not None:
		mask *= np.array([t==timestr_cut for t in all_timestr_list])
	
	photocurrent = m_o_m[mask]
	varcurrent = v_o_d[mask]
	
	if fit_line:
		# initialize a linear fitter
		fit = fitting.LinearLSQFitter()
		# initialize the outlier removal fitter
		or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=niter, sigma=itersigma)
		# initialize a linear model
		line_init = models.Linear1D()
		# fit the data with the fitter
		sigmask, fitted_line = or_fit(line_init, photocurrent, varcurrent)
		
		g1_iter = get_g1_from_slope_T_N(0.5*fitted_line.slope.value, N=nfr)

	
	if var_frac_thresh is not None:
		median_var = np.median(varcurrent)
		print('median variance is ', median_var)
		mask *= (np.abs(v_o_d-median_var) < var_frac_thresh*median_var) 
		
		print(np.sum(mask), len(v_o_d))
		
	min_x_val, max_x_val = np.min(photocurrent), np.max(photocurrent)
	
		
	if all_set_numbers_list is not None:
		colors = np.array(all_set_numbers_list)[mask]
	else:
		colors = np.arange(len(m_o_m))[mask]
		
	f = plt.figure(figsize=(12, 6))
	title = 'TM'+str(inst)
	if timestr_cut is not None:
		title += ', '+timestr_cut
		
	plt.title(title, fontsize=18)
		
	timestrs = ['03-13-2013', '04-25-2013', '05-13-2013']
	markers = ['o', 'x', '*']
	set_color_idxs = []
	for t, target_timestr in enumerate(timestrs):
		tstrmask = np.array([tstr==target_timestr for tstr in all_timestr_list])
		
		tstr_colors = None
		if all_set_numbers_list is not None:
			tstr_colors = np.array(all_set_numbers_list)[mask*tstrmask]
		
		if fit_line:
			if t==0:
				if inst==1:
					xvals = np.linspace(-5, -18, 100)
				else:
					xvals = np.linspace(np.min(photocurrent), np.max(photocurrent), 100)
					print('xvals:', xvals)
				plt.plot(xvals, fitted_line(xvals), label='Iterative sigma clip, $G_1$='+str(np.round(g1_iter, 3)))
			plt.scatter(photocurrent[tstrmask], sigmask[tstrmask], s=100, marker=markers[t], c=tstr_colors, label=target_timestr)
			plt.xlim(xlim)

		else:
			plt.scatter(photocurrent[tstrmask], varcurrent[tstrmask], s=100, marker=markers[t], c=tstr_colors, label=target_timestr)
			plt.xlim(xlim)
			plt.ylim(ylim)            

	plt.legend(fontsize=16)
	plt.xlabel('mean [ADU/fr]', fontsize=18)
	plt.ylabel('$\\sigma^2$ [(ADU/fr)$^2$]', fontsize=18)
	plt.tick_params(labelsize=14)
	plt.show()
	
	return m, f


def plot_hankel_integrand(bins_rad, fine_bins, wtheta, spline_wtheta, ell, integration_max):
	
	f = plt.figure(figsize=(10, 8))
	plt.subplot(2,2,3)
	plt.title('Discrete Integrand', fontsize=16)
	plt.plot(bins_rad, bins_rad*wtheta*scipy.special.j0(ell*bins_rad), label='$\\ell = $'+str(ell))
	plt.ylabel('$\\theta w(\\theta)J_0(\\ell\\theta)$', fontsize=18)
	plt.xlabel('$\\theta$ [rad]', fontsize=18)
	plt.legend(fontsize=14)
	plt.xscale('log')
	plt.subplot(2,2,4)
	plt.title('Spline Integrand', fontsize=16)
	plt.plot(fine_bins, fine_bins*spline_wtheta(fine_bins)*scipy.special.j0(ell*fine_bins), label='$\\ell = $'+str(ell))
	plt.ylabel('$\\theta w(\\theta)J_0(\\ell\\theta)$', fontsize=18)
	plt.xlabel('$\\theta$ [rad]', fontsize=18)
	plt.axvline(integration_max, linestyle='dashed', color='k', label='$\\theta_{max}$='+str(np.round(integration_max*(180./np.pi), 2))+' deg.')
	plt.legend(fontsize=14)
	plt.xscale('log')
	plt.subplot(2,2,1)
	plt.title('Angular 2PCF', fontsize=16)
	plt.plot(bins_rad, wtheta, marker='.', color='k')
	plt.plot(np.linspace(np.min(bins_rad), np.max(bins_rad), 1000), spline_wtheta(np.linspace(np.min(bins_rad), np.max(bins_rad), 1000)), color='b', linestyle='dashed')
	plt.xscale('log')
	plt.yscale('symlog')
	plt.xlabel('$\\theta$ [rad]', fontsize=18)
	plt.ylabel('$w(\\theta)$', fontsize=18)
	plt.subplot(2,2,2)
	plt.title('Bessel function, $\\ell = $'+str(ell), fontsize=16)
	plt.plot(fine_bins, scipy.special.j0(ell*fine_bins), label='$\\ell = $'+str(ell))
	plt.ylabel('$J_0(\\ell\\theta)$', fontsize=18)
	plt.xlabel('$\\theta$ [rad]', fontsize=18)
	plt.xscale('log')
	plt.tight_layout()
	plt.show()
	
	return f

def create_multipanel_figure(images, names, colormap, xlabel='x [pixels]', ylabel='y [pixels]'):
	num_images = len(images)
	num_cols = min(num_images, 3)  # Maximum 3 columns for better visualization
	
	num_rows = num_images // num_cols
	if num_images % num_cols != 0:
		num_rows += 1

	fig, axes = plt.subplots(3, 2, figsize=(8,10))
	axes = axes.flatten()  # Flatten axes into a 1D array for easier indexing
	
	if type(colormap)==str:
		colormap = [colormap for x in range(len(axes))]
	
	for i in range(num_images):
		ax = axes[i]
		image = images[i]
		name = names[i]
		
		# Calculate vmin and vmax based on the image data
		vmin = np.percentile(image, 5)  # Adjust percentile values as needed
		
		if i==1:
			vmax= np.percentile(image, 99)
		elif i==4:
			vmax = np.percentile(image, 84)
			vmin = np.percentile(image, 16)
		else:
			vmax = np.percentile(image, 95)   # Adjust percentile values as needed
		
		im = ax.imshow(image, cmap=colormap[i], vmin=vmin, vmax=vmax)
#         ax.set_title(name)
		# if i==0 or i==2:
		if i==4 or i==5:
			ax.set_xlabel(xlabel)
		if i in [0, 2, 4]:
			ax.set_ylabel(ylabel)

#         ax.axis('off')
		
		ax.text(0.95, 0.95, name, transform=ax.transAxes,
				fontsize=12, ha='right', va='top', bbox=dict({'facecolor':'white', 'alpha':0.8, 'edgecolor':None}))
		
		# Add colorbar with units
		cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.04, fraction=0.046)
		
		if i%2==1:
			cbar.set_label("nW m$^{-2}$ sr$^{-1}$", fontsize=12)
	
	# Hide unused subplots
	for j in range(num_images, num_rows * num_cols):
		fig.delaxes(axes[j])
	
	plt.subplots_adjust(hspace=0, wspace=0)  # Remove blank space between subplots

	plt.tight_layout()
	plt.show()
	
	return fig


def plot_means_vs_vars(m_o_m, v_o_d, timestrs, timestr_cut=None, var_frac_thresh=None, xlim=None, ylim=None, all_set_numbers_list=None, all_timestr_list=None,\
					  nfr=5, inst=1, fit_line=True, itersigma=4.0, niter=5, imdim=1024, figure_size=(12,6), markersize=100, titlestr=None, mode='linear', jackknife_g1=False, split_boot=10, boot_g1=True, n_boot=100):
	
	mask = [True for x in range(len(m_o_m))]
	
	if timestr_cut is not None:
		mask *= np.array([t==timestr_cut for t in all_timestr_list])
	
	photocurrent = np.array(m_o_m[mask])
	varcurrent = np.array(v_o_d[mask])

	if fit_line:
		if boot_g1:

			boot_g1s = np.zeros((n_boot,))
			for i in range(n_boot):
				randidxs = np.random.choice(np.arange(len(photocurrent)), len(photocurrent)//split_boot)
				phot = photocurrent[randidxs]
				varc = varcurrent[randidxs]

				_, _, g1_boot = fit_meanphot_vs_varphot(phot, 0.5*varc, nfr=nfr, itersigma=itersigma, niter=niter)
				boot_g1s[i] = g1_boot

			print('bootstrap sigma(G1) = '+str(np.std(boot_g1s)))
			print('while bootstrap mean is '+str(np.mean(boot_g1s)))
				
		fitted_line, sigmask, g1_iter = fit_meanphot_vs_varphot(photocurrent, 0.5*varcurrent, nfr=nfr, itersigma=itersigma, niter=niter)

	else:
		g1_iter = None
		
	
	if var_frac_thresh is not None:
		median_var = np.median(varcurrent)
		mask *= (np.abs(v_o_d-median_var) < var_frac_thresh*median_var) 
				
	min_x_val, max_x_val = np.min(photocurrent), np.max(photocurrent)
			
	if all_set_numbers_list is not None:
		colors = np.array(all_set_numbers_list)[mask]
	else:
		colors = np.arange(len(m_o_m))[mask]
		
	f = plt.figure(figsize=figure_size)
	title = 'TM'+str(inst)
	if timestr_cut is not None:
		title += ', '+timestr_cut
	if titlestr is not None:
		title += ' '+titlestr
				
	plt.title(title, fontsize=18)
		
	markers = ['o', 'x', '*', '^', '+']
	set_color_idxs = []
	
	
	for t, target_timestr in enumerate(timestrs):
		tstrmask = np.array([tstr==target_timestr for tstr in all_timestr_list])
		
		tstr_colors = None
		if all_set_numbers_list is not None:
			tstr_colors = np.array(all_set_numbers_list)[mask*tstrmask]
		
		if fit_line:
			if t==0:
				if inst==1:
					xvals = np.linspace(-1, -18, 100)
				else:
					xvals = np.linspace(np.min(photocurrent), np.max(photocurrent), 100)

				plt.plot(xvals, fitted_line(xvals), label='$G_1$='+str(np.round(np.mean(boot_g1s), 3))+'$\\pm$'+str(np.round(np.std(boot_g1s), 3)))
			plt.scatter(photocurrent[tstrmask], sigmask[tstrmask], s=markersize, marker=markers[t], c=tstr_colors, label=target_timestr)
			if ylim is not None:
				plt.ylim(ylim)
			if xlim is not None:
				plt.xlim(xlim)
		else:
			plt.scatter(photocurrent[tstrmask], 0.5*varcurrent[tstrmask], s=markersize, marker=markers[t], c=tstr_colors, label=target_timestr)
			plt.xlim(xlim)
			plt.ylim(ylim)            

	plt.legend(fontsize=16)
	plt.xlabel('mean [ADU/fr]', fontsize=18)
	plt.ylabel('$\\sigma^2$ [(ADU/fr)$^2$]', fontsize=18)
	plt.tick_params(labelsize=14)
	plt.show()
	
	return f, g1_iter

def plot_2d_fourier_modes(ps2d, fw_image=None, ps2d_dos=None, fw_image_dos=None, imshow=True, title=None, title_fs=20, show=True, return_fig=True,\
						 label=None, label_dos=None):

	f = plt.figure(figsize=(7,6))
	if title is not None:
		plt.suptitle(title, fontsize=title_fs)

	all_ps2d_rav = []
	l2d = get_l2d(cbps.dimx, cbps.dimy, pixsize=7.)

	for binidx in np.arange(9):
		
		lmin, lmax = cbps.Mkk_obj.binl[binidx], cbps.Mkk_obj.binl[binidx+1]
		l2d_mask = (lmin < l2d)*(l2d < lmax)
		l2d_mask[l2d==0] = 0
		
		if fw_image is not None:
			ps2d_copy, fw_image_copy = weighted_powerspec_2d(ps2d, l2d_mask.astype(int), fw_image=fw_image)
		else:
			ps2d_copy = weighted_powerspec_2d(ps2d, l2d_mask.astype(int))

		nonz_ps2d = ps2d_copy.ravel()[ps2d_copy.ravel()!=0]
		
		if ps2d_dos is not None:
			if fw_image_dos is not None:
				ps2d_copy_dos, fw_image_copy_dos = weighted_powerspec_2d(ps2d_dos, l2d_mask.astype(int), fw_image=fw_image_dos)
			else:
				ps2d_copy_dos = weighted_powerspec_2d(ps2d_dos, l2d_mask.astype(int))

			nonz_ps2d_dos = ps2d_copy_dos.ravel()[ps2d_copy_dos.ravel()!=0]

		
		all_ps2d_rav.append(nonz_ps2d)

		plt.subplot(3,3,binidx+1)
		plt.title(str(int(lmin))+'$<\\ell<$'+str(int(lmax)), fontsize=12)
		
		if imshow:
			posidxs = np.where(ps2d_copy !=0)

			xmin, xmax = np.min(posidxs[0])-1, np.max(posidxs[0])+2
			ymin, ymax = np.min(posidxs[1])-1, np.max(posidxs[1])+2
			
			vmin, vmax = np.nanmin(ps2d_copy[ps2d_copy!=0]), np.nanmax(ps2d_copy[ps2d_copy!=0])
			
			ps2d_copy[(ps2d_copy==0)] = np.nan
			plt.imshow(ps2d_copy[xmin:xmax,ymin:ymax],origin='lower', vmin=vmin, vmax=vmax)

			plt.xticks([], [])
			plt.yticks([], [])
			plt.colorbar(fraction=0.046, pad=0.04)
			
		else:

			if ps2d_copy_dos is not None:
				min_ps2d_rav = np.nanmin([np.nanmin(nonz_ps2d), np.nanmin(nonz_ps2d_dos)])
				max_ps2d_rav = np.nanmax([np.nanmax(nonz_ps2d), np.nanmax(nonz_ps2d_dos)])
			else:
				min_ps2d_rav = np.nanmin(nonz_ps2d)
				max_ps2d_rav = np.nanmax(nonz_ps2d)
				
			bins = np.linspace(min_ps2d_rav, max_ps2d_rav, 10)
			plt.subplot(3,3,binidx+1)

			plt.title(str(int(lmin))+'$<\\ell<$'+str(int(lmax)), fontsize=14)

			plt.hist(nonz_ps2d, bins=bins, color='k', histtype='step', label=label)
			if ps2d_copy_dos is not None:
				plt.hist(nonz_ps2d_dos, bins=bins, color='C3', histtype='step', label=label_dos)


	plt.tight_layout()
	if show:
		plt.show()
	if return_fig:
		return f

def plot_mean_var_modes(all_cl2ds, return_fig=True, plot=True):

	var_modes = np.var(all_cl2ds, axis=0)
	mean_modes = np.mean(all_cl2ds, axis=0)
	mean_rav = np.log10(mean_modes.ravel())
	var_rav = np.log10(var_modes.ravel())

	logvarmin, logvarmax = -14, -0
	logmeanmin, logmeanmax = -7, -0

	within_bounds = (var_rav > logvarmin)*(var_rav < logvarmax)*(mean_rav > logmeanmin)*(mean_rav < logmeanmax)

	f = plt.figure(figsize=(6,5))
	plt.hexbin(mean_rav[within_bounds], var_rav[within_bounds], bins=100, norm=matplotlib.colors.LogNorm(vmax=1e2))
	plt.colorbar()
	plt.plot(np.linspace(-7, -0.5, 1000), np.linspace(-14, -1, 1000), linestyle='dashed', color='r')
	plt.xlim(logmeanmin, logmeanmax)
	plt.ylim(logvarmin, logvarmax)
	if plot:
		plt.show()
	if return_fig:
		return f

def plot_var_ratios(var_ratios, labels, plot=True):
	
	f = plt.figure(figsize=(6,5))
	for v, var_ratio in enumerate(var_ratios):
		plt.scatter(lb, var_ratio, color='C'+str(v), label=labels[v])

	plt.plot(lb, np.mean(np.array(var_ratios), axis=0), color='k', label='Field average')
	plt.ylabel('$\\sigma(N_{\\ell}^{model})/\\sigma(N_{\\ell}^{data})$', fontsize=14)
	plt.xlabel('$\\ell$', fontsize=14)
	plt.yscale('log')
	plt.xscale("log")
	plt.ylim(1e-1, 1e1)
	plt.grid()
	plt.legend(fontsize=12, ncol=2, loc=3)
	plt.tick_params(labelsize=12)
	plt.tight_layout()
	if plot:
		plt.show()

	return f

def plot_compare_rdnoise_darkdiff_cl(inst, fieldname, lb, prefac, cl_modl, cl_modl_std, meancl_dd, stdcl_dd, \
							   return_fig=True, plot=True, ymin=1e-4, ymax=1e3):

	f = plt.figure(figsize=(8,4))
	plt.subplot(1,2,1)
	plt.errorbar(lb, prefac*cl_modl, yerr=prefac*cl_modl_std, label='Simulated read noise', color='C3', capsize=4, fmt='o', marker='x')
	plt.errorbar(lb, prefac*meancl_dd, yerr=prefac*stdcl_dd, color='k', marker='+', label='Dark differences (real data)', capsize=4, fmt='o')
	plt.text(200, 150, 'TM'+str(inst)+' ('+fieldname+')', fontsize=16)
	plt.xscale('log')
	plt.yscale('log')
	plt.grid()
	plt.xlabel('$\\ell$', fontsize=14)
	plt.ylabel('$D_{\\ell}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=14)
	plt.legend(loc=4)
	if fieldname=='elat30':
		plt.ylim(ymin, 10*ymax)
	else:
		plt.ylim(ymin, ymax)
	plt.tick_params(labelsize=12)

	plt.subplot(1,2,2)
	plt.errorbar(lb, cl_modl/cl_modl, yerr=cl_modl_std/cl_modl, color='C3', label='Simulated read noise', capsize=4, fmt='o', marker='x', markersize=5)
	plt.errorbar(lb, meancl_dd/cl_modl, yerr=stdcl_dd/cl_modl, color='k', label='Dark differences (real data)', markersize=5, capsize=4, fmt='o', marker='+')
	plt.ylabel('$N_{\\ell}/\\langle N_{\\ell}^{NM}\\rangle$', fontsize=14)
	plt.xscale('log')
	plt.axhline(1.0, linestyle='dashed', color='k', linewidth=2)
	plt.ylim(-1., 3.0)
	plt.xlabel('$\\ell$', fontsize=16)
	plt.legend(loc=4)

	plt.grid()
	plt.tick_params(labelsize=12)

	plt.tight_layout()

	if plot:
		plt.show()
	if return_fig:
		return f

