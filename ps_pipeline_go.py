import numpy as np
import astropy.io.fits as fits
import astropy.wcs as wcs
import pandas as pd
import scipy.io
from scipy.ndimage import gaussian_filter
import matplotlib
import matplotlib.pyplot as plt
import pickle
# from astropy.convolution import convolve
# from astropy.convolution import Gaussian2DKernel
from astropy.wcs import WCS
from scipy.io import loadmat
import os

from powerspec_utils import *
from ciber_powerspec_pipeline import *
# from ciber_powerspec_pipeline import CIBER_PS_pipeline, iterative_gradient_ff_solve, lin_interp_powerspec,\
									 # grab_all_simidx_dat, grab_recovered_cl_dat, generate_synthetic_mock_test_set,\
									 # instantiate_dat_arrays_fftest, instantiate_cl_arrays_fftest, calculate_powerspec_quantities, compute_powerspectra_realdat
# from cross_spectrum_analysis import *
# from ciber_data_helpers import load_psf_params_dict
from plotting_fns import plot_map
from ciber_mocks import *
from flat_field_est import *
from mkk_parallel import compute_inverse_mkk, plot_mkk_matrix
from masking_utils import *
import config
from ciber_data_file_utils import *


def grab_fpath_vals(fpath_dict, name_list):
	path_list = []
	for name in name_list:
		path_list.append(fpath_dict[name])
		
	return path_list

def Merge(dict1, dict2):
	res = dict1.copy()
	res.update(dict2)
	return res

def grab_ps_set(maps, ifield_list, ps_set_shape, cbps, masks=None):

	cls_array = np.zeros(ps_set_shape)

	for i, ifield in enumerate(ifield_list):

		obs_masksub = maps[i].copy()
		if masks is not None:
			obs_masksub *= masks[i]
			obs_masksub[masks[i]==1] -= np.mean(obs_masksub[masks[i]==1])
		else:
			obs_masksub -= np.mean(obs_masksub)
		
		lb, cl, clerr = get_power_spec(obs_masksub, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)

		cls_array[i] = cl 

	return np.array(cls_array)


def update_dicts(list_of_dicts, kwargs):

	for key, value in kwargs.items():
		for indiv_dict in list_of_dicts:
			if key in indiv_dict:
				print('Setting '+key+' to '+str(value))
				indiv_dict[key] = value 
	return list_of_dicts 


def ciber_difference_spectrum(cbps, fpath_dict, config_dict, inst, ifieldA, ifieldB, mask_tail, masking_maglim, mask=None, sigma_clip=False, nsig=5, \
                             per_quadrant=False):
    
    if inst==1:
        masking_maglim_ff = max(17.5, masking_maglim)
    else:
        masking_maglim_ff = max(17.0, masking_maglim)
    
    fieldidxA, fieldidxB = ifieldA-4, ifieldB-4
    
    read_noise_models = cbps.grab_noise_model_set([ifieldA, ifieldB], inst, noise_model_base_path=fpath_dict['read_noise_modl_base_path'], noise_modl_type=config_dict['noise_modl_type'])
    
    dc_template = cbps.load_dark_current_template(inst, verbose=True, inplace=False)
    
    mask_save_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/masks/'+mask_tail+'/joint_mask_ifield'+str(ifieldA)+'_ifield'+str(ifieldB)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'
    mask = fits.open(mask_save_fpath)['joint_mask_'+str(ifieldA)].data
    
    plot_map(mask, title='mask')

    mkkonly_savepath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/mkk/'+mask_tail+'/mkk_maskonly_estimate_ifield'+str(ifieldA)+'_ifield'+str(ifieldB)+'_observed_'+mask_tail+'.fits'
    inv_Mkk = fits.open(mkkonly_savepath)['inv_Mkk_'+str(ifieldA)].data
    
    # load beams and average Bootes fields
    bls_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/beam_correction/bl_est_postage_stamps_TM'+str(inst)+'_081121.npz'

    B_ells = np.load(bls_fpath)['B_ells_post']
    B_ell_eff = np.sqrt(B_ells[fieldidxA]*B_ells[fieldidxB])
    
    
    ptsrcfile = fits.open(config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/point_src_maps_for_ffnoise/point_src_maps'+'_TM'+str(inst)+'_mmin='+str(masking_maglim)+'_mmax='+str(masking_maglim_ff)+'_merge.fits', overwrite=True)

    max_vals_ptsrc_A = ptsrcfile['ifield'+str(ifieldA)].header['maxval']
    max_vals_ptsrc_B = ptsrcfile['ifield'+str(ifieldB)].header['maxval']
        
    max_vals_ptsrc = max(max_vals_ptsrc_A, max_vals_ptsrc_B)
    
    print('max val ptsrc:', max_vals_ptsrc)
    
    if per_quadrant:
        t_ell_fpath = config.ciber_basepath+'data/transfer_function/t_ell_masked_wdgl_2500_n=2p5_wgrad_quadrant_nsims=1000.npz'
    else:
        t_ell_fpath = config.ciber_basepath+'data/transfer_function/t_ell_est_nsims=100.npz'
        
    t_ell_av = np.load(t_ell_fpath)['t_ell_av']
    print('t_ell = ', t_ell_av)

    
    for fieldidx, ifield in enumerate([ifieldA, ifieldB]):

        cbps.load_flight_image(ifield, inst, verbose=True, ytmap=False) # default loads aducorr maps
        observed_im = cbps.image*cbps.cal_facs[inst]
        observed_im -= dc_template*cbps.cal_facs[inst]

        
        if ifield==ifieldA:
            observed_im_A = observed_im
        if ifield==ifieldB:
            observed_im_B = observed_im   
    
    
    mean_sb_A = np.mean((observed_im_A*mask)[mask != 0])
    mean_sb_B = np.mean((observed_im_B*mask)[mask != 0])
    
    shot_sigma_sb_zl_perfA = cbps.compute_shot_sigma_map(inst, mean_sb_A*np.ones_like(observed_im_A), nfr=cbps.field_nfrs[ifieldA])
    shot_sigma_sb_zl_perfB = cbps.compute_shot_sigma_map(inst, mean_sb_B*np.ones_like(observed_im_B), nfr=cbps.field_nfrs[ifieldB])

    mean_shot_sigma_sb = 0.5*(shot_sigma_sb_zl_perfA+shot_sigma_sb_zl_perfB)
    
    plot_map(mean_shot_sigma_sb, title='shot sigma sb')
    
    difference_im = observed_im_A - observed_im_B
    
    if sigma_clip:
        mask = iter_sigma_clip_mask(difference_im, sig=nsig, nitermax=5, mask=mask)
    
    if inst==2:
        corner_mask = np.ones_like(mask)
        corner_mask[900:, 0:200] = 0.
        mask *= corner_mask
        
    plot_map(difference_im*mask)
    
    meansub = np.mean(difference_im[mask!=0])
    
    masked_difference = difference_im*mask
    masked_difference[mask != 0] -= np.mean(masked_difference[mask != 0]) 
    
    lb, cl, clerr = get_power_spec(masked_difference, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
    
    pf = lb*(lb+1)/(2*np.pi)
    plt.figure()
    plt.errorbar(lb, pf*cl, yerr=pf*clerr, fmt='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    
    
    fourier_weights_diff, \
            mean_nl2d_diff, nl1ds_diff = cbps.estimate_noise_power_spectrum(nsims=500, n_split=10, inst=inst, ifield=ifieldA, field_nfr=cbps.field_nfrs[ifieldA], mask=mask, \
                                      noise_model=read_noise_models[1], difference=True, shot_sigma_sb=mean_shot_sigma_sb, compute_1d_cl=True)
    
    plot_map(fourier_weights_diff, title='fourier weights')
    plot_map(mean_nl2d_diff, title='nl2d')
    
    lb, N_ell_est, N_ell_err = cbps.compute_noise_power_spectrum(inst, noise_Cl2D=mean_nl2d_diff.copy(), inplace=False, apply_FW=True, weights=fourier_weights_diff)

    lb, processed_ps_nf, cl_proc_err, _ = cbps.compute_processed_power_spectrum(inst, apply_mask=True, \
                                 mask=mask, image=masked_difference, convert_adufr_sb=False, \
                                mkk_correct=True, inv_Mkk=inv_Mkk, beam_correct=True, B_ell=B_ell_eff, \
                                apply_FW=True, verbose=True, noise_debias=True, \
                             FF_correct=False, FW_image=fourier_weights_diff, FF_image=None,\
                                 gradient_filter=True, save_intermediate_cls=False, N_ell=N_ell_est, \
                                 per_quadrant=per_quadrant, max_val_after_sub=max_vals_ptsrc)
    
    

    processed_ps_nf /= t_ell_av
    cl_proc_err /= t_ell_av
    
    plt.figure()
    plt.errorbar(lb, pf*processed_ps_nf/2., yerr=pf*cl_proc_err, fmt='o')
    plt.plot(lb, pf*N_ell_est/2., color='r', linestyle='dashed')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('$D_{\\ell}$', fontsize=14)
    plt.ylabel('$\\ell$', fontsize=14)
    plt.ylim(1e-2, 1e4)
    plt.grid(alpha=0.5)
    plt.show()
    
    return lb, processed_ps_nf, cl_proc_err, N_ell_est


def set_up_filepaths_cbps(fpath_dict, inst, run_name, datestr, datestr_trilegal=None, data_type='mock', cross_inst=None, save_fpaths=True, \
	verbose=False):

	mock_base_path = fpath_dict['ciber_mock_fpath']+datestr+'/'
	trilegal_base_path = fpath_dict['ciber_mock_fpath']+datestr_trilegal+'/'
	fpath_dict['trilegal_base_path'] = trilegal_base_path+'trilegal/'

	if data_type=='mock':
		base_path = fpath_dict['ciber_mock_fpath']+datestr+'/'

	elif data_type=='observed':
		base_path = fpath_dict['observed_base_path']
		trilegal_base_path = None


	list_of_dirpaths = []

	# base_path = fpath_dict['ciber_mock_fpath']+datestr+'/'

	mock_info_path_names = ['cib_realiz_path', 'cib_resid_ps_path', 'isl_resid_ps_path']
	mock_info_path_dirnames = ['cib_realiz', 'cib_resid_ps', 'isl_resid_ps']

	base_path_names = ['mask_base_path', 'mkk_base_path', 'mkk_ffest_base_path']
	base_path_dirnames = ['masks', 'mkk', 'mkk_ffest']

	# base_path_names = ['mask_base_path', 'mkk_base_path', 'mkk_ffest_base_path', 'cib_realiz_path', 'cib_resid_ps_path', 'isl_resid_ps_path']
	# base_path_dirnames = ['masks', 'mkk', 'mkk_ffest', 'cib_realiz', 'cib_resid_ps', 'isl_resid_ps']

	base_fluc_path = config.ciber_basepath+'data/fluctuation_data/'
	# base_fluc_path = config.exthdpath+'ciber_fluctuation_data/'
	base_fluc_names = ['read_noise_modl_base_path', 'bls_base_path', 'subpixel_psfs_path', 'iris_base_path']
	base_fluc_dirnames = ['noise_model', 'beam_correction', 'subpixel_psfs', 'iris_regrid']

	tm_string = 'TM'+str(inst)

	if cross_inst is not None:
		tm_string += '_TM'+str(cross_inst)+'_cross'
		base_path_names.append('proc_regrid_base_path')
		base_path_dirnames.append('proc_regrid')

	for b, base_name in enumerate(mock_info_path_names):
		if fpath_dict[base_name] is None:
			fpath_dict[base_name] = mock_base_path+tm_string+'/'+mock_info_path_dirnames[b]
			if verbose:
				print('setting '+base_name+' to '+fpath_dict[base_name])
		list_of_dirpaths.append(fpath_dict[base_name])

	for b, base_name in enumerate(base_path_names):
		if fpath_dict[base_name] is None:
			fpath_dict[base_name] = base_path+tm_string+'/'+base_path_dirnames[b]
			if verbose:
				print('setting '+base_name+' to '+fpath_dict[base_name])

		list_of_dirpaths.append(fpath_dict[base_name])

	for b, base_fluc in enumerate(base_fluc_names):
		if fpath_dict[base_fluc] is None:
			fpath_dict[base_fluc] = base_fluc_path+'TM'+str(inst)+'/'+base_fluc_dirnames[b]
			if verbose:
				print('setting '+base_fluc+' to '+fpath_dict[base_fluc])

		if fpath_dict['cross_'+base_fluc] is None and cross_inst is not None:
			fpath_dict['cross_'+base_fluc] = base_fluc_path+'TM'+str(cross_inst)+'/'+base_fluc_dirnames[b]
			if verbose:
				print('Setting cross_'+base_fluc+' to '+fpath_dict['cross_'+base_fluc])
			list_of_dirpaths.append(fpath_dict['cross_'+base_fluc])

		list_of_dirpaths.append(fpath_dict[base_fluc])

	if fpath_dict['tls_base_path'] is None:
		fpath_dict['tls_base_path'] = base_path+'transfer_function/'
		if verbose:
			print('tls base path is ', fpath_dict['tls_base_path'])
		list_of_dirpaths.append(fpath_dict['tls_base_path'])

	if fpath_dict['ff_est_dirpath'] is None:
		fpath_dict['ff_est_dirpath'] = fpath_dict['ciber_mock_fpath']+'030122/ff_realizations/'
		if fpath_dict['ff_run_name'] is None:
			fpath_dict['ff_run_name'] = run_name
		fpath_dict['ff_est_dirpath'] += fpath_dict['ff_run_name']
		if verbose:
			print('ff_est_dirpath is ', fpath_dict['ff_est_dirpath'])

	if verbose:
		print(list_of_dirpaths)
	if save_fpaths:
		make_fpaths(list_of_dirpaths)


	return fpath_dict, list_of_dirpaths, base_path, trilegal_base_path



def return_default_cbps_dicts():
	config_dict = dict({'ps_type':'auto', 'cross_type':'ciber', 'cross_inst':None, 'cross_gal':None, 'simidx0':0, \
						'full_mask_tail':'maglim_17_Vega_test', 'bright_mask_tail':'maglim_11_Vega_test', \
						'noise_modl_type':'full', 'dgl_mode':'sfd_clean', 'cmblens_mode':None})

	pscb_dict = dict({'ff_estimate_correct':True, 'apply_mask':True, 'with_inst_noise':True, 'with_photon_noise':True, 'apply_FW':True, 'generate_diffuse_realization':True, \
					'apply_smooth_FF':True, 'compute_beam_correction':False, 'same_clus_levels':True, 'same_zl_levels':False, 'apply_zl_gradient':True, 'gradient_filter':False, \
					'iterate_grad_ff':True ,'mkk_ffest_hybrid':True, 'apply_sum_fluc_image_mask':False, 'same_int':False, 'same_dgl':True, 'use_ff_weights':True, 'plot_ff_error':False, 'load_ptsrc_cib':True, \
					'load_trilegal':True , 'subtract_subthresh_stars':False, 'ff_bias_correct':True, 'save_ff_ests':True, 'plot_maps':True, \
					 'draw_cib_setidxs':False, 'aug_rotate':False, 'noise_debias':True, 'load_noise_bias':False, 'transfer_function_correct':True, 'compute_transfer_function':False,\
					  'save_intermediate_cls':True, 'verbose':False, 'show_plots':False, 'show_ps':True, 'save':True, 'bl_post':False, 'ff_sigma_clip':False, \
					  'per_quadrant':False, 'use_dc_template':True, 'ff_estimate_cross':False, 'map_photon_noise':False, 'zl_photon_noise':True, \
					  'compute_ps_per_quadrant':False, 'apply_wen_cluster_mask':True, 'low_responsivity_blob_mask':True, \
					  'pt_src_ffnoise':False, 'shut_off_plots':False, 'max_val_clip':False, 'point_src_ffnoise':False})

	float_param_dict = dict({'ff_min':0.5, 'ff_max':2.0, 'clip_sigma':5,'clip_sigma_ff':5, 'ff_stack_min':1, 'nmc_ff':10, \
					  'theta_mag':0.01, 'niter':5, 'dgl_scale_fac':5, 'smooth_sigma':5, 'indiv_ifield':6,\
					  'nfr_same':25, 'J_bright_Bl':11, 'J_faint_Bl':17.5, 'n_FW_sims':500, 'n_FW_split':10, \
					  'ell_norm_blest':5000, 'n_realiz_t_ell':100, 'nitermax':5, 'n_cib_isl_sims':100})

	fpath_dict = dict({'ciber_mock_fpath':config.ciber_basepath+'data/ciber_mocks/', \
						'sim_test_fpath':config.ciber_basepath+'data/input_recovered_ps/sim_tests_030122/', \
						'base_fluc_path':config.ciber_basepath+'data/fluctuation_data/', \
						'ff_smooth_fpath':'data/flatfield/TM1_FF/TM1_field4_mag=17.5_FF.fits', \
						'observed_base_path':config.ciber_basepath+'data/fluctuation_data/', 'ff_est_dirpath':None, \
						'cib_resid_ps_path':None, 'isl_resid_ps_path':None, 'proc_regrid_base_path':None, \
						'mask_base_path':None, 'mkk_base_path':None, 'mkk_ffest_base_path':None, 'mkk_ffest_mask_tail':None, 'bls_base_path':None, 'isl_rms_fpath':None, 'bls_fpath':None, 'bls_fpath_quad':None, \
						't_ell_fpath':None, 'cib_realiz_path':None, 'tls_base_path':None, 'ff_run_name':None, \
						'noisemodl_basepath':None, 'subpixel_psfs_path':None, 'read_noise_modl_base_path':None, 'noisemodl_run_name':None, 'add_savestr':None, \
						'iris_base_path':None, 'cross_read_noise_modl_base_path':None, 'cross_bls_base_path':None, 'cross_subpixel_psfs_path':None, \
					  'cross_iris_base_path':None, 'bls_fpath_cross':None, 'ciber_regrid_fpaths':None, 'ffest_run_name':None, 'ffest_run_name_cross':None})

	return config_dict, pscb_dict, float_param_dict, fpath_dict


def generate_ff_error_realizations(cbps, run_name, inst, ifield_list, joint_masks, simmaps_dc_all, shot_sigma_sb_zl, read_noise_models, weights_ff, \
									fpath_dict, pscb_dict, float_param_dict, obs_levels):

	ff_realization_estimates = np.zeros_like(joint_masks)
	observed_ims_nofluc = np.zeros_like(joint_masks)

	for ffidx in range(float_param_dict['nmc_ff']):
		for fieldidx, ifield in enumerate(ifield_list):
			zl_realiz = generate_zl_realization(obs_levels[fieldidx], False, dimx=cbps.dimx, dimy=cbps.dimy)

			if pscb_dict['with_photon_noise']:
				# this line below was a bug for a while, zl_perfield for observed is zero so it was adding zero photon noise to the realizations
				zl_realiz += shot_sigma_sb_zl[fieldidx]*np.random.normal(0, 1, size=cbps.map_shape) # temp
				observed_ims_nofluc[fieldidx] = zl_realiz

			if pscb_dict['with_inst_noise']: # true for observed
				read_noise_indiv, _ = cbps.noise_model_realization(inst, cbps.map_shape, read_noise_models[fieldidx], \
										read_noise=True, photon_noise=False, chisq=False)

				# if pscb_dict['show_plots'] and ffidx==0:
					# plot_map(read_noise_indiv, title='readnoise indiv in MC FF draws')

				observed_ims_nofluc[fieldidx] += read_noise_indiv


		print('ffiter = ', ffidx, ' at beginning of iterate grad ff in ff error realiz, sum mask 0 is ', np.sum(joint_masks[0]))


		if pscb_dict['iterate_grad_ff']:

			if pscb_dict['per_quadrant']:
				print('RUNNING ITERATE GRAD FF per quadrant on realizations')
				# processed_ims = np.zeros_like(observed_ims_nofluc)
				ff_realization_estimates = np.zeros_like(observed_ims_nofluc)

				observed_ims_nofluc_byquad = [observed_ims_nofluc[:, cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]] for q in range(4)]

				for q in range(4):

					masks_quad = joint_masks[:,cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]]

					_, ff_realization_estimates_byquad,\
								 _, _, _ = iterative_gradient_ff_solve(observed_ims_nofluc_byquad[q], niter=float_param_dict['niter'], masks=masks_quad, weights_ff=weights_ff, \
																														ff_stack_min=float_param_dict['ff_stack_min']) # masks already accounting for ff_stack_min previously

					# processed_ims[:,cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]] = processed_ims_byquad
					ff_realization_estimates[:,cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]] = ff_realization_estimates_byquad

			else:
				if pscb_dict['verbose']:
					print('RUNNING ITERATE GRAD FF full on realizations')

				_, ff_realization_estimates, _, _, _ = iterative_gradient_ff_solve(observed_ims_nofluc, niter=float_param_dict['niter'], masks=joint_masks, weights_ff=weights_ff, \
																														ff_stack_min=float_param_dict['ff_stack_min']) # masks already accounting for ff_stack_min previously
		print('ffiter = ', ffidx, ' at end of iterate grad ff in ff error realiz, sum mask 0 is ', np.sum(joint_masks[0]))

		np.savez(fpath_dict['ff_est_dirpath']+'/'+run_name+'/ff_realization_estimate_ffidx'+str(ffidx)+'.npz', \
				ff_realization_estimates = ff_realization_estimates)


def run_cbps_pipeline(cbps, inst, nsims, run_name, ifield_list = None, \
							datestr='100921', datestr_trilegal=None, data_type='mock',\
							masking_maglim=17.5, masking_maglim_ff=None, mask_tail='abc110821', regrid_mask_tail=None, mask_tail_ffest=None, mask_tail_cross=None, \
							zl_levels=None, ff_biases=None, **kwargs):


	config_dict, pscb_dict, float_param_dict, fpath_dict = return_default_cbps_dicts()	

	if masking_maglim_ff is None:
		masking_maglim_ff = masking_maglim


	if not pscb_dict['load_trilegal']:
		print('load trilegal = False, setting trilegal_fpath to None..')
		trilegal_fpath=None
	elif datestr_trilegal is None:
		datestr_trilegal = datestr
		
	config_dict, pscb_dict, float_param_dict, fpath_dict = update_dicts([config_dict, pscb_dict, float_param_dict, fpath_dict], kwargs)
	fpath_dict, list_of_dirpaths, base_path, trilegal_base_path = set_up_filepaths_cbps(fpath_dict, inst, run_name, datestr, cross_inst=config_dict['cross_inst'], datestr_trilegal=datestr_trilegal, data_type=data_type)

	if pscb_dict['compute_ps_per_quadrant']:
		cbps_quad = CIBER_PS_pipeline(dimx=512, dimy=512)

	if ifield_list is None:
		ifield_list = [4, 5, 6, 7, 8]
		print('Setting ifield list to ', ifield_list)
	nfield = len(ifield_list)

	# if pscb_dict['save_intermediate_cls']:
	# 	cls_inter, inter_labels = [], [] # each cl added will have a corresponding key

	mean_isl_rms = None
	if fpath_dict['isl_rms_fpath'] is not None:
		print('Loading ISL RMS from ', fpath_dict['isl_rms_fpath']+'..')
		mean_isl_rms = load_isl_rms(fpath_dict['isl_rms_fpath'], masking_maglim, nfield)
	
	field_set_shape = (nfield, cbps.dimx, cbps.dimy)
	ps_set_shape = (nsims, nfield, cbps.n_ps_bin)
	single_ps_set_shape = (nfield, cbps.n_ps_bin)

	max_vals_ptsrc = [None for x in range(len(ifield_list))]
	point_src_comps_for_ff = [None for x in range(len(ifield_list))]

	print('we have float_param_dict[ffmin] = ', float_param_dict['ff_min'], float_param_dict['ff_max'])

	
	# if masking_maglim_ff is not None:
	# 	if masking_maglim_ff >= masking_maglim:
	# 		print('loading point source maps for ff noise')
	if pscb_dict['pt_src_ffnoise']:
		print('Loading point source maps for FF noise terms')
		ptsrcfile = fits.open(config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/point_src_maps_for_ffnoise/point_src_maps'+'_TM'+str(inst)+'_mmin='+str(masking_maglim)+'_mmax='+str(masking_maglim_ff)+'_merge.fits', overwrite=True)
		point_src_comps_for_ff = np.zeros(field_set_shape)
		# 		max_vals_ptsrc = np.zeros((len(ifield_list)))

		for fieldidx, ifield in enumerate(ifield_list):
			point_src_comps_for_ff[fieldidx] = ptsrcfile['ifield'+str(ifield)].data
		# 			if not pscb_dict['shut_off_plots']:
		# 				plot_map(point_src_comps_for_ff[fieldidx], title='ifield '+str(ifield)+' point src map')
		# 			max_vals_ptsrc[fieldidx] = ptsrcfile['ifield'+str(ifield)].header['maxval']


	# print('max ptsrc values:', max_vals_ptsrc)

	if data_type=='observed':
		print('DATA TYPE IS OBSERVED')
		config_dict['simidx0'], nsims = 0, 1
		pscb_dict['same_int'], pscb_dict['apply_zl_gradient'], pscb_dict['load_trilegal'],\
				 pscb_dict['load_ptsrc_cib'], pscb_dict['apply_smooth_FF'] = [False for x in range(5)]
		
		if config_dict['ps_type']=='auto' or config_dict['cross_type']=='ciber':
			pscb_dict['with_inst_noise'] = True
			pscb_dict['with_photon_noise'] = True
		else:
			pscb_dict['with_inst_noise'] = False
			pscb_dict['with_photon_noise'] = False
		print('Observed data, setting same_int, apply_zl_gradient, load_ptsrc_cib to False, with_inst_noise and with_photon_noise to True..')

	if not pscb_dict['ff_estimate_correct']:
		print('Not estimating/correcting for flat field in PS:')
		# pscb_dict['mkk_ffest_hybrid'] = False 
		pscb_dict['iterate_grad_ff'] = False 
		pscb_dict['use_ff_weights'] = False
		pscb_dict['ff_bias_correct'] = False 
		pscb_dict['apply_smooth_FF'] = False 
		pscb_dict['save_ff_ests'] = False

	if pscb_dict['mkk_ffest_hybrid']:
		mode_couple_base_dir = fpath_dict['mkk_ffest_base_path']
		# print('Using hybrid mask-FF mode coupling matrices, setting ff_bias_correct, transfer_function_correct to False')
		print('Using hybrid mask-FF mode coupling matrices, setting ff_bias_correct to False')
		pscb_dict['ff_bias_correct'] = False 

	else:
		mode_couple_base_dir = fpath_dict['mkk_base_path']

	if pscb_dict['same_int']:
		nfr_fields = [float_param_dict['nfr_same'] for i in range(nfield)]
		ifield_noise_list = [float_param_dict['indiv_ifield'] for i in range(nfield)]
		ifield_list_ff = [float_param_dict['indiv_ifield'] for i in range(nfield)]
	else:
		nfr_fields = [cbps.field_nfrs[ifield] for ifield in ifield_list]
		ifield_noise_list = ifield_list.copy()
		ifield_list_ff = ifield_list.copy()

	if config_dict['ps_type']=='auto' and not pscb_dict['with_inst_noise'] and not pscb_dict['with_photon_noise']:
		print('pscb_dict[with_inst_noise] and pscb_dict[with_photon_noise] set to False, so setting apply_FW and noise_debias to False..')
		pscb_dict['apply_FW'] = False
		pscb_dict['noise_debias'] = False

	ciber_cross_ciber, ciber_cross_gal, ciber_cross_dgl, ciber_cross_lens = [False for x in range(4)]
	if config_dict['ps_type']=='cross':
		if config_dict['cross_type']=='ciber':
			ciber_cross_ciber = True
			cross_inst=config_dict['cross_inst']
			pscb_dict['save_ff_ests'] = False
		elif config_dict['cross_type']=='galaxy':
			ciber_cross_gal = True
			pscb_dict['iterate_grad_ff'] = False
		elif config_dict['cross_type']=='dgl':
		# elif config_dict['cross_type'] in ['IRIS', 'sfd_clean', 'sfd']:
			print('cross type is DGL we are here, setting ciber_cross_dgl to True..')
			print('dgl mode is ', config_dict['dgl_mode'])
			ciber_cross_dgl = True
		elif config_dict['cross_type']=='cmblens':
			ciber_cross_lens = True


	if pscb_dict['same_zl_levels']:
		zl_fieldnames = [cbps.ciber_field_dict[float_param_dict['indiv_ifield']] for i in range(nfield)]
	else:
		zl_fieldnames = [cbps.ciber_field_dict[ifield] for ifield in ifield_list]
	if zl_levels is None:
		zl_levels = [cbps.zl_levels_ciber_fields[inst][zl_fieldname] for zl_fieldname in zl_fieldnames]
	if ciber_cross_ciber:
		zl_levels_cross = [cbps.zl_levels_ciber_fields[cross_inst][zl_fieldname] for zl_fieldname in zl_fieldnames]
		print('zl_levels_cross is ', zl_levels_cross)

	if pscb_dict['verbose']:
		print('NFR fields is ', nfr_fields)
		print('ZL levels are ', zl_levels)

	if pscb_dict['use_ff_weights']: # cross
		weights_photonly = cbps.compute_ff_weights(inst, zl_levels, read_noise_models=None, ifield_list=ifield_list_ff, additional_rms=mean_isl_rms)
		print('FF weights are :', weights_photonly)
		if ciber_cross_ciber:
			weights_photonly_cross = cbps.compute_ff_weights(cross_inst, zl_levels_cross, ifield_list=ifield_list_ff)

	#  ------------------- instantiate data arrays  --------------------
	clus_realizations, zl_perfield, ff_estimates, observed_ims = [np.zeros(field_set_shape) for i in range(4)]
	if config_dict['ps_type']=='cross': # cross ciber
		observed_ims_cross = np.zeros(field_set_shape)
		if config_dict['cross_type']=='ciber':
			ff_estimates_cross = np.zeros(field_set_shape)

	inv_Mkks, joint_maskos, joint_maskos_ffest = None, None, None

	# if cross power spectrum then load union mask, but might use separate masks for flat field estimation?

	final_masked_images = np.zeros(field_set_shape)

	if pscb_dict['apply_mask']:
		joint_maskos = np.zeros(field_set_shape) 

		if mask_tail_ffest is not None:
			joint_maskos_ffest = np.zeros_like(joint_maskos)

		if pscb_dict['compute_ps_per_quadrant']:
			joint_maskos_per_quadrant = np.zeros((4, field_set_shape[0], field_set_shape[1], field_set_shape[2]))

	signal_power_spectra, recovered_power_spectra, recovered_dcl = [np.zeros(ps_set_shape) for i in range(3)]
	if pscb_dict['compute_ps_per_quadrant']:
		recovered_power_spectra_per_quadrant = np.zeros((4, ps_set_shape[0], ps_set_shape[1], ps_set_shape[2]))
		recovered_dcl_per_quadrant = np.zeros((4, ps_set_shape[0], ps_set_shape[1], ps_set_shape[2]))

	# ------------------------------------------------------------------

	smooth_ff = None
	if pscb_dict['apply_smooth_FF'] and data_type != 'observed':
		print('loading smooth FF from ', fpath_dict['ff_smooth_fpath'])
		ff = fits.open(fpath_dict['ff_smooth_fpath'])
		smooth_ff = gaussian_filter(ff[1].data, sigma=float_param_dict['smooth_sigma'])
		if pscb_dict['show_plots'] and not pscb_dict['shut_off_plots']:
			plot_map(smooth_ff, title='flat field, smoothed with $\\sigma=$'+str(float_param_dict['smooth_sigma']))

	read_noise_models = [None for x in ifield_noise_list]
	read_noise_models_per_quad = None
	if pscb_dict['with_inst_noise']:
		read_noise_models = cbps.grab_noise_model_set(ifield_noise_list, inst, noise_model_base_path=fpath_dict['read_noise_modl_base_path'], noise_modl_type=config_dict['noise_modl_type'])
		
		if pscb_dict['compute_ps_per_quadrant']:
			read_noise_models_per_quad = []
			for q in range(4):

				read_noise_models_indiv_quad = cbps.grab_noise_model_set(ifield_noise_list, inst, \
													noise_model_base_path=fpath_dict['read_noise_modl_base_path'], noise_modl_type=config_dict['noise_modl_type']+'_quad'+str(q))
				read_noise_models_per_quad.append(read_noise_models_indiv_quad)


		if ciber_cross_ciber: # cross
			print('Loading read noise models for cross instrument TM', cross_inst)
			read_noise_models_cross = cbps.grab_noise_model_set(ifield_noise_list, cross_inst, noise_model_base_path=fpath_dict['cross_read_noise_modl_base_path'])

	if pscb_dict['transfer_function_correct']:

		if config_dict['ps_type']=='cross' and config_dict['cross_type']=='ciber':
			# fpath_dict['t_ell_fpath'] = fpath_dict['tls_base_path']+'tl_regrid_tm2_to_tm1_ifield4_041323.npz'
			fpath_dict['t_ell_fpath'] = fpath_dict['tls_base_path']+'t_ell_masked_wdgl_2500_n=2p5_wgrad_quadrant_nsims=1000.npz'

		else:
			if pscb_dict['per_quadrant']:
				fpath_dict['t_ell_fpath'] = fpath_dict['tls_base_path']+'t_ell_masked_wdgl_2500_n=2p5_wgrad_quadrant_nsims=1000.npz'
			else:
				fpath_dict['t_ell_fpath'] = fpath_dict['tls_base_path']+'t_ell_est_nsims='+str(float_param_dict['n_realiz_t_ell'])+'.npz'
			
			if pscb_dict['compute_ps_per_quadrant']:
				fpath_dict['t_ell_fpath_quad'] = fpath_dict['tls_base_path']+'t_ell_est_nsims='+str(float_param_dict['n_realiz_t_ell'])+'_quad.npz'
		

		if pscb_dict['compute_transfer_function']:
			print('Computing transfer function from '+str(float_param_dict['n_realiz_t_ell']))
			lb, t_ell_av, t_ell_stderr, t_ells, cls_orig, cls_filt = cbps.calculate_transfer_function(nsims=float_param_dict['n_realiz_t_ell'], plot=False)
			if fpath_dict['t_ell_fpath'] is not None:
				print('Saving transfer function correction to ', fpath_dict['t_ell_fpath'])
				np.savez(fpath_dict['t_ell_fpath'], lb=lb, t_ell_av=t_ell_av, t_ell_stderr=t_ell_stderr, t_ells=t_ells, cls_orig=cls_orig, cls_filt=cls_filt)
		elif fpath_dict['t_ell_fpath'] is not None:
			# if config_dict['ps_type']=='cross' and config_dict['cross_type']=='ciber':
			# 	t_ell_key = 'tl_regrid'
			# else:
			# 	t_ell_key = 't_ell_av'

			t_ell_key = 't_ell_av'
			print('Loading transfer function from ', fpath_dict['t_ell_fpath'])
			t_ell_av = np.load(fpath_dict['t_ell_fpath'])[t_ell_key]

			if pscb_dict['compute_ps_per_quadrant']:
				print('loading transfer function for single quadrant from ', fpath_dict['t_ell_fpath_quad'])
				t_ell_av_quad = np.load(fpath_dict['t_ell_fpath_quad'])['t_ell_av_quad']
				if pscb_dict['verbose']:
					print('t_ell_av_quad:', t_ell_av_quad)

		else:
			print('No transfer function path provided, and compute_transfer_function set to False, exiting..')
			return 

	B_ells = [None for x in range(nfield)]

	if not pscb_dict['compute_beam_correction']:

		if fpath_dict['bls_fpath'] is not None: # cross
			print('loading B_ells from ', fpath_dict['bls_fpath'])
			B_ells = np.load(fpath_dict['bls_fpath'])['B_ells_post']

			if pscb_dict['compute_ps_per_quadrant']:
				print('loading B_ells quad from ', fpath_dict['bls_fpath_quad'])
				B_ells_quad = np.load(fpath_dict['bls_fpath_quad'])['B_ells_post']

			if pscb_dict['verbose']:
				print('B_ells = ', B_ells)
		
		if config_dict['ps_type']=='cross' and fpath_dict['bls_fpath_cross'] is not None:
			print('loading B_ells for cross from ', fpath_dict['bls_fpath_cross'])
			B_ells_cross = np.load(fpath_dict['bls_fpath_cross'])['B_ells_post']


	# loop through simulations
	for i in np.arange(config_dict['simidx0'], nsims):
		# pscb_dict['verbose'] = True
		# if i>10:
		# 	pscb_dict['verbose'] = True

		if pscb_dict['save_intermediate_cls']:
			cls_inter, inter_labels = [], [] # each cl added will have a corresponding key


		if data_type=='mock':
			print('Simulation set '+str(i)+' of '+str(nsims)+'..')
		elif data_type=='observed':
			print('Running observed data..')

		cib_setidxs, inv_Mkks, orig_mask_fractions = [], [], []
		if pscb_dict['compute_ps_per_quadrant']:
			inv_Mkks_per_quadrant = []

		cib_setidx = i%float_param_dict['n_cib_isl_sims']
		print('cib setidx = ', cib_setidx)

		if data_type=='mock':
			if pscb_dict['load_trilegal']:
				trilegal_fpath = fpath_dict['ciber_mock_fpath']+datestr_trilegal+'/trilegal/mock_trilegal_simidx'+str(cib_setidx)+'_'+datestr_trilegal+'.fits'
			
			# test_set_fpath = fpath_dict['ciber_mock_fpath']+datestr+'/TM'+str(inst)+'/cib_realiz/cib_with_tracer_5field_set'+str(cib_setidx)+'_'+datestr+'_TM'+str(inst)+'.fits'
			test_set_fpath = fpath_dict['ciber_mock_fpath']+datestr+'/TM'+str(inst)+'/cib_realiz/cib_with_tracer_with_dpoint_5field_set'+str(cib_setidx)+'_'+datestr+'_TM'+str(inst)+'.fits'
			# assume same PS for ell^-3 sky clustering signal (indiv_ifield corresponds to one DGL field that is scaled by dgl_scale_fac)
			
			merge_dict = Merge(pscb_dict, float_param_dict) # merge dictionaries
			joint_masks, observed_ims, total_signals,\
					snmaps, rnmaps, shot_sigma_sb_maps, noise_models,\
					ff_truth, diff_realizations, zl_perfield, mock_cib_ims = cbps.generate_synthetic_mock_test_set(inst, ifield_list,\
														test_set_fpath=test_set_fpath, mock_trilegal_path=trilegal_fpath,\
														 noise_models=read_noise_models, ff_truth=smooth_ff, zl_levels=zl_levels, **merge_dict)

			if config_dict['ps_type']=='cross' and pscb_dict['load_ptsrc_cib'] is False:
				print('mode = cross and no point sources, so making additional set of mocks in different band assuming fixed colors..')

				if config_dict['cross_type']=='ciber':
					color_ratio = cbps.iras_color_facs[cross_inst] / cbps.iras_color_fac[inst]
					print('color ratio is ', color_ratio)

					diff_realizations_cross = diff_realizations*color_ratio 
					zl_ratio = zl_levels_cross / zl_levels
					zl_perfield_cross = np.array([zl_indiv*zl_ratio for zl_indiv in zl_perfield])

				joint_masks_cross, observed_ims_cross, total_signals_cross,\
				snmaps_cross, rnmaps_cross, shot_sigma_sb_maps_cross, noise_models_cross,\
				ff_truth_cross, _, _, mock_cib_ims = cbps.generate_synthetic_mock_test_set(cross_inst, ifield_list,\
													test_set_fpath=test_set_fpath, mock_trilegal_path=trilegal_fpath,\
													 noise_models=read_noise_models_cross, diff_realizations=diff_realizations_cross, \
													 zl_perfield=zl_perfield_cross, ff_truth=smooth_ff, **merge_dict)
		
			
		else:
			if pscb_dict['use_dc_template']:
				dc_template = cbps.load_dark_current_template(inst, verbose=True, inplace=False)
				
				if pscb_dict['show_plots'] and not pscb_dict['shut_off_plots']:
					plot_map(dc_template, title='DC template TM'+str(inst))


			for fieldidx, ifield in enumerate(ifield_list):

				cbps.load_flight_image(ifield, inst, verbose=True, ytmap=False) # default loads aducorr maps
				observed_ims[fieldidx] = cbps.image*cbps.cal_facs[inst]

				if pscb_dict['use_dc_template']:
					print('subtracting dark current template from image')
					observed_ims[fieldidx] -= dc_template*cbps.cal_facs[inst]

				if ciber_cross_ciber:
					print('Loading data products for cross CIBER TM'+str(cross_inst))

					obs_cross_fpath = fpath_dict['proc_regrid_base_path']+'/'+regrid_mask_tail+'/proc_regrid_TM'+str(cross_inst)+'_to_TM'+str(inst)+'_ifield'+str(ifield)+'_'+regrid_mask_tail+'.fits'
					obs_cross_file = fits.open(obs_cross_fpath)
					obs_cross = obs_cross_file['proc_regrid_'+str(ifield)].data 

					if fieldidx==0:
						obs_levels_cross = np.zeros(len(ifield_list))

					obs_levels_cross[fieldidx] = obs_cross_file[0].header['obs_level']
					observed_ims_cross[fieldidx] = obs_cross

				elif ciber_cross_gal:
					observed_ims_cross[fieldidx] = cbps.load_gal_density(ifield, verbose=True, inplace=False)
				
				elif ciber_cross_dgl:

					print('config dict [dgl mode] is ', config_dict['dgl_mode'])
					regrid_map = load_regrid_dgl_map(inst, ifield, config_dict['dgl_mode'])
					observed_ims_cross[fieldidx] = regrid_map
					fourier_weights_cross = None

				elif ciber_cross_lens:
					regrid_map = load_regrid_lens_map(inst, ifield, config_dict['cmblens_mode'])
					observed_ims_cross[fieldidx] = regrid_map
					fourier_weights_cross = None


				if pscb_dict['plot_maps'] and not pscb_dict['shut_off_plots']:
					plot_map(observed_ims[fieldidx], title='observed ims ifield = '+str(ifield))
					
					if config_dict['ps_type']=='cross':
						plot_map(observed_ims_cross[fieldidx], title='observed ims cross ifield = '+str(ifield))

		if i==0 and pscb_dict['compute_beam_correction'] and config_dict['ps_type']=='auto': # cross
			# if doing cross spectrum, lets just expect that both B_ells are precomputed
			if fpath_dict['bls_fpath'] is None:
				if data_type=='mock':
					tailstr = 'simidx'+str(i)
				elif data_type=='observed':
					tailstr = 'observed'
				bl_est_path = fpath_dict['bls_base_path']+'ptsrc_blest_TM'+str(inst)+'_'+tailstr+'.npz'
				print('Computing beam correction to be saved to bls_fpath='+bl_est_path+'..')
			else:
				bl_est_path = fpath_dict['bls_fpath']

			lb, diff_norm_array,\
					cls_masked_tot, cls_masked_tot_bright = cbps.estimate_b_ell_from_maps(inst, ifield_list, observed_ims, simidx=i, plot=True, save=False, \
										niter = float_param_dict['niter'], ff_stack_min=float_param_dict['ff_stack_min'], data_type=data_type, mask_base_path=fpath_dict['mask_base_path'], ell_norm=float_param_dict['ell_norm_blest'], \
										full_mask_tail=config_dict['full_mask_tail'], bright_mask_tail=config_dict['bright_mask_tail'], \
										iterate_grad_ff=pscb_dict['iterate_grad_ff'], mkk_correct=True, mkk_base_path=fpath_dict['mkk_base_path'])
				
			B_ells = np.sqrt(diff_norm_array).copy()

			print('B_ells is ', B_ells)

			np.savez(bl_est_path, \
				lb=lb, J_bright_mag=float_param_dict['J_bright_Bl'], J_tot_mag=float_param_dict['J_faint_Bl'],\
					 cls_masked_tot_bright=cls_masked_tot_bright, cls_masked_tot=cls_masked_tot, B_ells=B_ells)


		# ----------- load masks and mkk matrices ------------

		if pscb_dict['apply_mask']: # if applying mask load masks and inverse Mkk matrices

			if pscb_dict['mkk_ffest_hybrid']:
				mkk_type = 'ffest_nograd'
			else:
				mkk_type = 'maskonly_estimate'

			for fieldidx, ifield in enumerate(ifield_list):
				
				if data_type=='observed': # cross
					if ciber_cross_ciber:
						mask_fpath = fpath_dict['mask_base_path']+'/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_TM'+str(inst)+'_TM'+str(cross_inst)+'_observed_'+mask_tail+'.fits'
					else:
						mask_fpath = fpath_dict['mask_base_path']+'/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'

					joint_maskos[fieldidx] = fits.open(mask_fpath)[1].data

					if ifield==5:
						elat30_mask=True 
					else:
						elat30_mask = False 

					joint_maskos[fieldidx] = additional_masks(cbps, joint_maskos[fieldidx], inst, ifield,\
																 low_responsivity_blob_mask=pscb_dict['low_responsivity_blob_mask'],\
																 apply_wen_cluster_mask=pscb_dict['apply_wen_cluster_mask'], \
																 corner_mask=True, elat30_mask=elat30_mask)

					if ifield==4:
						print('line 679 sum of mask 0 is ', np.sum(joint_maskos[fieldidx]))

					if float_param_dict['clip_sigma'] is not None:

						if pscb_dict['compute_ps_per_quadrant']:
							print('applying sigma clip for individual quadrants..')
							for q in range(4):
								sigclip_quad = iter_sigma_clip_mask(observed_ims[fieldidx, cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]], sig=float_param_dict['clip_sigma'], nitermax=float_param_dict['nitermax'], mask=joint_maskos[fieldidx, cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]].astype(int))
								joint_maskos[fieldidx, cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]] *= sigclip_quad
						else:
							print('applying sigma clip to uncorrected flight image..')
							sigclip = iter_sigma_clip_mask(observed_ims[fieldidx], sig=float_param_dict['clip_sigma'], nitermax=float_param_dict['nitermax'], mask=joint_maskos[fieldidx].astype(int))
							print('np.sum sigclip:', np.sum(sigclip))
							plot_map(sigclip, title='sigma clip 699 fieldidx '+str(fieldidx))
							joint_maskos[fieldidx] *= sigclip

							plot_map(observed_ims[fieldidx]*joint_maskos[fieldidx], cmap='Greys_r', title='masked image line 699 , ifield '+str(ifield))
					inv_Mkk_fpath = mode_couple_base_dir+'/'+mask_tail+'/mkk_'+mkk_type+'_ifield'+str(ifield)+'_observed_'+mask_tail+'.fits'
					inv_Mkks.append(fits.open(inv_Mkk_fpath)['inv_Mkk_'+str(ifield)].data)

					if pscb_dict['compute_ps_per_quadrant']:
						inv_Mkk_fpaths_per_quadrant, inv_Mkks_indiv = [], []
						for q in range(4):
							inv_Mkk_fpaths_per_quadrant.append(mode_couple_base_dir+'/'+mask_tail+'/mkk_'+mkk_type+'_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'_quad'+str(q)+'.fits')
							inv_Mkks_indiv.append(fits.open(inv_Mkk_fpaths_per_quadrant[q])['inv_Mkk_'+str(ifield)].data)
						inv_Mkks_per_quadrant.append(inv_Mkks_indiv)

					if mask_tail_ffest is not None:
						print('Loading full mask from ', mask_tail_ffest) 

						if ciber_cross_ciber:
							mask_fpath_ffest = fpath_dict['mask_base_path']+'/'+mask_tail_ffest+'/joint_mask_ifield'+str(ifield)+'_TM'+str(inst)+'_TM'+str(cross_inst)+'_observed_'+mask_tail_ffest+'.fits'

						else:
							mask_fpath_ffest = fpath_dict['mask_base_path']+'/'+mask_tail_ffest+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail_ffest+'.fits'
						joint_maskos_ffest[fieldidx] = fits.open(mask_fpath_ffest)['joint_mask_'+str(ifield)].data


						joint_maskos_ffest[fieldidx] = additional_masks(cbps, joint_maskos_ffest[fieldidx], inst, ifield,\
																	 low_responsivity_blob_mask=pscb_dict['low_responsivity_blob_mask'],\
																	 apply_wen_cluster_mask=pscb_dict['apply_wen_cluster_mask'], \
																	 corner_mask=True, elat30_mask=elat30_mask)

						if ifield==4:
							print('line 715 sum of ff mask 0 is ', np.sum(joint_maskos_ffest[fieldidx]))

						# plot_map(observed_ims[fieldidx]*joint_maskos_ffest[fieldidx], title='im * ff mask ')
				else:
					if pscb_dict['verbose']:
						print('mask path is ', fpath_dict['mask_base_path']+'/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_'+mask_tail+'.fits')
					joint_maskos[fieldidx] = fits.open(fpath_dict['mask_base_path']+'/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_'+mask_tail+'.fits')['joint_mask_'+str(ifield)].data
					
					if mask_tail_ffest is not None:
						print('Loading FFest full mask from ', mask_tail_ffest) 
						mask_fpath_ffest = fpath_dict['mask_base_path']+'/'+mask_tail_ffest+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_'+mask_tail_ffest+'.fits'
						joint_maskos_ffest[fieldidx] = fits.open(mask_fpath_ffest)['joint_mask_'+str(ifield)].data
				
					if pscb_dict['mkk_ffest_hybrid']:
						# updated 8/14/23, for mocks should use mode coupling matrix of main mask
						mkk_ffest_mask_tail = mask_tail
						inv_Mkks.append(fits.open(mode_couple_base_dir+'/'+mkk_ffest_mask_tail+'/mkk_ffest_grad_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_'+mkk_ffest_mask_tail+'.fits')['inv_Mkk_'+str(ifield)].data)
						# inv_Mkks.append(fits.open(mode_couple_base_dir+'/'+mkk_ffest_mask_tail+'/mkk_ffest_grad_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_'+mkk_ffest_mask_tail+'.fits')['inv_Mkk_'+str(ifield)].data)
					else:
						inv_Mkks.append(fits.open(mode_couple_base_dir+'/'+mask_tail+'/mkk_maskonly_estimate_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_'+mask_tail+'.fits')['inv_Mkk_'+str(ifield)].data)

			orig_mask_fractions = np.array(orig_mask_fractions)

			if pscb_dict['iterate_grad_ff']:
				print('mask fractions before FF estimation are ', orig_mask_fractions)


		# if config_dict['ps_type']=='cross' and config_dict['cross_gal'] is not None:
		# 	cross_maps = 
		# multiply signal by flat field and add read noise on top

		# 11/13/23 removed weights_ff_cross since pre-processed and then regrid for cross
		if pscb_dict['iterate_grad_ff']:
			weights_ff = None
			if pscb_dict['use_ff_weights']:
				weights_ff = weights_photonly

			if pscb_dict['verbose']:
				print('RUNNING ITERATE GRAD FF for observed images')
				print('and weights_ff is ', weights_ff)

			if pscb_dict['save_intermediate_cls']:
				cls_inter.append(grab_ps_set(observed_ims, ifield_list, single_ps_set_shape, cbps, masks=joint_maskos))
				inter_labels.append('preffgrad_masked')

			if pscb_dict['per_quadrant']:
				if pscb_dict['verbose']:
					print('Processing quadrants of input maps separately..')
					print('NOTE: need to update this so that wrapper works')

				processed_ims = np.zeros_like(observed_ims)
				ff_estimates = np.zeros_like(observed_ims)

				print('line 768 sum of mask 0 ', np.sum(joint_maskos[0]))
				# print('line 769 sum of ff mask 0 ', np.sum(joint_maskos_ffest[0]))
				if joint_maskos_ffest is not None:
					joint_maskos_ffest, stack_masks = stack_masks_ffest(joint_maskos_ffest, float_param_dict['ff_stack_min'])
					joint_maskos *= stack_masks

				quad_means_masked_by_ifield, quad_stds_masked_by_ifield = [np.zeros((len(ifield_list), 4)) for x in range(2)]
				ciber_maps_byquad = [observed_ims[:, cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]] for q in range(4)]
				# ciber_masks_byquad_temp = [joint_maskos[:, cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]] for q in range(4)]

				print('line 777 sum of mask 0 ', np.sum(joint_maskos[0]))
				# print('line 778 sum of ff mask 0 ', np.sum(joint_maskos_ffest[0]))

				for q, quad in enumerate(ciber_maps_byquad):
					if pscb_dict['verbose']:
						print('q = ', q)

					if joint_maskos_ffest is not None:
						masks_quad = joint_maskos_ffest[:,cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]]
						clip_sigma_ff=float_param_dict['clip_sigma_ff']
					else:
						masks_quad = joint_maskos[:,cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]]
						clip_sigma_ff=None

					# sigma clip happening further up in code already, set to None here
					processed_ciber_maps_quad, ff_estimates_quad,\
						final_planes, stack_masks, ff_weights = process_ciber_maps(cbps, ifield_list, inst, ciber_maps_byquad[q], masks_quad,\
																				 clip_sigma=clip_sigma_ff, nitermax=float_param_dict['nitermax'], \
																					niter=float_param_dict['niter'], ff_stack_min=float_param_dict['ff_stack_min'])

					# processed_ciber_maps_quad, ff_estimates_quad,\
					# 	final_planes, stack_masks, ff_weights = process_ciber_maps(cbps, ifield_list, inst, ciber_maps_byquad[q], masks_quad,\
					# 															 clip_sigma=None, nitermax=float_param_dict['nitermax'], \
					# 																niter=float_param_dict['niter'], ff_stack_min=float_param_dict['ff_stack_min'])


					processed_ims[:,cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]] = processed_ciber_maps_quad
					if pscb_dict['apply_mask']:
						print('Multiplying total masks by stack masks..')
						joint_maskos[:,cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]] *= stack_masks

						if joint_maskos_ffest is not None:
							joint_maskos_ffest[:,cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]] *= stack_masks

					ff_estimates[:,cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]] = ff_estimates_quad


				print('line 802 sum of mask 0 ', np.sum(joint_maskos[0]))
				# print('line 803 sum of ff mask 0 ', np.sum(joint_maskos_ffest[0]))

				ciber_masks_byquad_temp = [joint_maskos[:, cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]] for q in range(4)]

				processed_ims_byquad = [processed_ims[:,cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]] for q in range(4)]
				for fieldidx, ifield in enumerate(ifield_list):

					plot_map(processed_ims[fieldidx]*joint_maskos[fieldidx], cmap='Greys', title='processed * mask after process_ciber_maps')

					quad_medians_masked = [np.median((cbmap[fieldidx]*ciber_masks_byquad_temp[q][fieldidx])[(ciber_masks_byquad_temp[q][fieldidx] != 0)]) for q, cbmap in enumerate(processed_ims_byquad)]

					if pscb_dict['verbose']:
						print('quad medians masked for ifield '+str(ifield)+' is ', quad_medians_masked)

					quad_means_masked_by_ifield[fieldidx] = quad_medians_masked

			else:

				print('Performing FF gradient filtering over full image')
				if joint_maskos_ffest is not None:
					print('using joint maskos ffest for flat field grad estimation')
					processed_ims, ff_estimates, final_planes, stack_masks, all_coeffs = iterative_gradient_ff_solve(observed_ims, niter=float_param_dict['niter'], masks=joint_maskos_ffest, \
																							weights_ff=weights_ff, ff_stack_min=float_param_dict['ff_stack_min'])
				else:
					processed_ims, ff_estimates, final_planes, stack_masks, all_coeffs = iterative_gradient_ff_solve(observed_ims, niter=float_param_dict['niter'], masks=joint_maskos, \
																							weights_ff=weights_ff, ff_stack_min=float_param_dict['ff_stack_min'])

				if pscb_dict['apply_mask']:
					if pscb_dict['verbose']:
						print('Multiplying total masks by stack masks..')
					joint_maskos *= stack_masks

			if pscb_dict['save_intermediate_cls']:
				# check if memory leak
				cls_inter.append(grab_ps_set(processed_ims, ifield_list, single_ps_set_shape, cbps, masks=joint_maskos))
				inter_labels.append('postffgrad_masked')

			# if pscb_dict['ff_sigma_clip'] and pscb_dict['apply_mask']:
			# 	print('Unsharp masking on flat field estimate')
			# 	for newidx, newifield in enumerate(ifield_list):
			# 		print('loooping im on ', newidx, newifield)
			# 		ff_smoothed = gaussian_filter(ff_estimates[newidx], sigma=5)
			# 		ff_smoothsubtract = ff_estimates[newidx]-ff_smoothed
			# 		sigclip = sigma_clip_maskonly(ff_smoothsubtract, previous_mask=joint_maskos[newidx], sig=float_param_dict['clip_sigma'])
			# 		joint_maskos[newidx] *= sigclip
			# 		print('sum of sigclip is ', np.sum(~sigclip))

			if float_param_dict['ff_min'] is not None and float_param_dict['ff_max'] is not None:
				print('we have float_param_dict[ffmin] = ', float_param_dict['ff_min'], float_param_dict['ff_max'])
				if pscb_dict['apply_mask']:

					if pscb_dict['verbose']:
						print('Clipping on ff_min=', float_param_dict['ff_min'], 'ff_max=', float_param_dict['ff_max'])
					ff_masks = (ff_estimates > float_param_dict['ff_min'])*(ff_estimates < float_param_dict['ff_max'])
					joint_maskos *= ff_masks

			mask_fractions = np.ones((len(ifield_list)))
			if pscb_dict['apply_mask']:
				mask_fractions = np.array([float(np.sum(joint_masko))/float(cbps.dimx*cbps.dimy) for joint_masko in joint_maskos])
				if pscb_dict['verbose']:
					print('masking fraction is nooww ', mask_fractions)

			observed_ims = processed_ims.copy()
			if ciber_cross_ciber and pscb_dict['ff_estimate_cross']:
				observed_ims_cross = processed_ims_cross.copy()

			for k in range(len(ff_estimates)):

				if pscb_dict['apply_mask']:
					ff_estimates[k][joint_maskos[k]==0] = 1.

					if ciber_cross_ciber:
						ff_estimates_cross[k][joint_maskos[k]==0] = 1.

					if pscb_dict['show_plots'] and not pscb_dict['shut_off_plots']:
						plot_map(ff_estimates[k], title='ff estimates k 894')
						plot_map(processed_ims[k]*joint_maskos[k], title='masked im')

						masked_sub = observed_ims[k]*joint_maskos[k]

						masked_sub[masked_sub != 0] -= np.mean(masked_sub[masked_sub != 0])

						plot_map(masked_sub, title='masked sub (observed_ims), 1045')

						if ciber_cross_ciber:
							plot_map(observed_ims_cross[k]*joint_maskos[k], title='masked im cross')


				if k==0 and pscb_dict['show_plots'] and not pscb_dict['shut_off_plots']:
					plot_map(ff_estimates[k], title='ff estimates k')

		
		if pscb_dict['apply_mask']: # cross
			obs_levels = np.array([np.median(obs[joint_maskos[o]==1]) for o, obs in enumerate(observed_ims)])

			print('obs levels:', obs_levels)
			# if config_dict['ps_type']=='cross' and config_dict['cross_type'] == 'ciber':

			if config_dict['ps_type']=='cross' and config_dict['cross_type'] != 'dgl' and config_dict['cross_type'] != 'cmblens':
				# obs_levels_cross = np.array([np.mean(obs[joint_maskos[o]==1]) for o, obs in enumerate(observed_ims_cross)])
				print('obs levels cross:', obs_levels_cross)

		else:
			obs_levels = np.array([np.mean(obs) for obs in observed_ims])
			if config_dict['ps_type']=='cross' and config_dict['cross_type'] != 'dgl':
				print('obs levels cross:', obs_levels_cross)

				# obs_levels_cross = np.array([np.mean(obs) for o, obs in enumerate(observed_ims_cross)])

		if pscb_dict['verbose']:
			print('obs levels are ', obs_levels)
		if pscb_dict['ff_bias_correct']:
			if config_dict['ps_type']=='auto' and ff_biases is None:
				ff_biases = compute_ff_bias(obs_levels, weights=weights_photonly, mask_fractions=mask_fractions)
				if pscb_dict['verbose']:
					print('FF biases are ', ff_biases)
			elif config_dict['ps_type']=='cross':
				if ciber_cross_ciber:
					print("calculating diagonal FF bias for ciber x ciber")
					ff_biases = compute_ff_bias(obs_levels, weights=weights_photonly, mask_fractions=mask_fractions, \
												mean_normalizations_cross=obs_levels_cross, weights_cross=weights_photonly_cross)
					print('FF biases for cross spectrum are ', ff_biases)
				elif ciber_cross_gal or ciber_cross_dgl:
					ff_biases = np.ones((nfield))

		# nls_estFF_nofluc = []
		N_ells_est = np.zeros((nfield, cbps.n_ps_bin))

		if pscb_dict['compute_ps_per_quadrant']:
			N_ells_est_per_quadrant = np.zeros((4, nfield, cbps_quad.n_ps_bin))

		# N_ells_est = []
		# for the tests with bright masking thresholds, we can calculate the ff realizations with the bright mask and this should be fine for the noise bias
		# for the actual flat field estimates of the observations, apply the deeper masks when stacking the off fields. The ff error will be that of the deep masks, 
		# and so there may be some bias unaccounted for by the analytic multiplicative bias.
		# I think the same holds for the gradients, just use the deeper masks

		# make monte carlo FF realizations that are used for the noise realizations (how many do we ultimately need?)
		if pscb_dict['verbose']:
			print('i = ', i, 'pscb_dict[save_ff_ests] = ', pscb_dict['save_ff_ests'], 'noise_debias = ', pscb_dict['noise_debias'])
		

		simmaps_dc_all, shot_sigma_sb_zl = [np.zeros(field_set_shape) for x in range(2)]

		for fieldidx, ifield in enumerate(ifield_list):
			
			if pscb_dict['map_photon_noise']:			
				shot_sigma_sb_zl_perf = cbps.compute_shot_sigma_map(inst, observed_ims[fieldidx], nfr=nfr_fields[fieldidx])
				simmaps_dc_all[fieldidx] = obs_levels[fieldidx]

			elif pscb_dict['zl_photon_noise']:
				if pscb_dict['per_quadrant']:
					mean_levels_perquad = np.ones_like(observed_ims[fieldidx])

					for q in range(4):
						mean_levels_perquad[cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]] = quad_means_masked_by_ifield[fieldidx, q]
					shot_sigma_sb_zl_perf = cbps.compute_shot_sigma_map(inst, mean_levels_perquad, nfr=nfr_fields[fieldidx])
					simmaps_dc_all[fieldidx] = mean_levels_perquad

				else:
					if pscb_dict['verbose']:
						print('obs levels [fieldidx] = ', obs_levels[fieldidx])
					shot_sigma_sb_zl_perf = cbps.compute_shot_sigma_map(inst, obs_levels[fieldidx]*np.ones_like(observed_ims[fieldidx]), nfr=nfr_fields[fieldidx])
					simmaps_dc_all[fieldidx] = obs_levels[fieldidx]

			shot_sigma_sb_zl[fieldidx] = shot_sigma_sb_zl_perf

			# if i==0 and pscb_dict['show_plots']:
			# 	plot_map(shot_sigma_sb_zl_perf, title='shot sigma map ifield'+str(ifield))
			# 	plot_map(simmaps_dc_all[fieldidx], title='simmaps dc all ifield'+str(ifield))
			

		if i==0 and pscb_dict['save_ff_ests'] and pscb_dict['noise_debias']:
			print('pscb_dict[save_ff_ests] and pscb_dict[noise_debias] are both True..')

			weights_ff_nofluc = None
			if pscb_dict['use_ff_weights']:
				if pscb_dict['verbose']:
					print('weights for ff realization are ', weights_photonly)
				weights_ff_nofluc = weights_photonly

			print('before ff error realiz, sum mask 0 is ', np.sum(joint_maskos[0]))
			if joint_maskos_ffest is not None:
				print('using ff masks for ff error realizations..')
				generate_ff_error_realizations(cbps, run_name, inst, ifield_list, joint_maskos_ffest, simmaps_dc_all, shot_sigma_sb_zl, read_noise_models, weights_ff_nofluc, \
											 fpath_dict, pscb_dict, float_param_dict, obs_levels)
			else:
				generate_ff_error_realizations(cbps, run_name, inst, ifield_list, joint_maskos, simmaps_dc_all, shot_sigma_sb_zl, read_noise_models, weights_ff_nofluc, \
											 fpath_dict, pscb_dict, float_param_dict, obs_levels)

			print('after ff error realiz, sum mask 0 is ', np.sum(joint_maskos[0]))

		if pscb_dict['save_intermediate_cls']:
			cls_masked_prenl, cls_postffb_corr, cls_masked_postnl,\
				 cls_postmkk_prebl, cls_postbl, cls_postffb_corr, cls_post_isl_sub, cls_post_tcorr = [np.zeros(single_ps_set_shape) for x in range(8)]

		for fieldidx, obs in enumerate(observed_ims):
			print('fieldidx = ', fieldidx)

			if pscb_dict['apply_mask']:
				target_mask = joint_maskos[fieldidx]

				# plot_map(obs*target_mask, figsize=(5,5), title='line 953 ifield '+str(ifield))

				target_invMkk = inv_Mkks[fieldidx]

				if pscb_dict['compute_ps_per_quadrant']:
					target_invMkks_per_quadrant = inv_Mkks_per_quadrant[fieldidx]
			else:
				stack_mask, target_mask, target_invMkk = None, None, None

			if not pscb_dict['ff_estimate_correct']:
				simmap_dc, simmap_dc_cross = None, None
			else:
				if pscb_dict['apply_mask']:
					simmap_dc = np.median(obs[target_mask==1])
					if config_dict['ps_type']=='cross':
						simmap_dc_cross = np.mean(observed_ims_cross[fieldidx][target_mask==1])
				else:
					simmap_dc = np.median(obs)
					if config_dict['ps_type']=='cross':
						simmap_dc_cross = np.median(observed_ims_cross[fieldidx])

			# noise bias from sims with no fluctuations    
			if pscb_dict['verbose']:
				print('and nowwww here on i = ', i)

			nl_estFF_nofluc, fourier_weights_nofluc = None, None

			if pscb_dict['with_inst_noise'] or pscb_dict['with_photon_noise']:

				# if noise model has already been saved just load it
				if i > 0 or pscb_dict['load_noise_bias']:

					# temporary comment out
					if fpath_dict['noisemodl_basepath'] is None:
						fpath_dict['noisemodl_basepath'] = fpath_dict['ciber_mock_fpath']+'030122/noise_models_sim/'
					if fpath_dict['noisemodl_run_name'] is None:
						fpath_dict['noisemodl_run_name'] = run_name 

					noisemodl_tailpath = '/noise_bias_fieldidx'+str(fieldidx)+'.npz'
					noisemodl_fpath = fpath_dict['noisemodl_basepath'] + fpath_dict['noisemodl_run_name'] + noisemodl_tailpath
					noisemodl_file = np.load(noisemodl_fpath)
					fourier_weights_nofluc = noisemodl_file['fourier_weights_nofluc']

					if pscb_dict['compute_ps_per_quadrant']:
						fourier_weights_nofluc_per_quadrant, mean_cl2d_cross_per_quadrant = [[] for x in range(2)]
						for q in range(4):
							noisemodl_tailpath_per_quadrant = '/noise_bias_fieldidx'+str(fieldidx)+'_quad'+str(q)+'.npz'
							noisemodl_fpath_per_quadrant = fpath_dict['noisemodl_basepath'] + fpath_dict['noisemodl_run_name'] + noisemodl_tailpath_per_quadrant
							noisemodl_file_indiv_quadrant = np.load(noisemodl_fpath_per_quadrant)
							
							fourier_weights_nofluc_per_quadrant.append(noisemodl_file_indiv_quadrant['fourier_weights_nofluc'])
							mean_cl2d_cross_per_quadrant.append(noisemodl_file_indiv_quadrant['mean_cl2d_nofluc'])
							N_ells_est_per_quadrant[q, fieldidx] = noisemodl_file_indiv_quadrant['nl_estFF_nofluc']

					if pscb_dict['verbose']:
						print('LOADING NOISE MODEL FROM ', noisemodl_fpath)

					if ciber_cross_ciber:
						fourier_weights_cross = noisemodl_file['fourier_weights_nofluc']
						mean_cl2d_cross = noisemodl_file['mean_cl2d_nofluc']
						N_ell_est = noisemodl_file['nl_estFF_nofluc']
						lb, N_ell_est, N_ell_err = cbps.compute_noise_power_spectrum(inst, noise_Cl2D=mean_cl2d_cross.copy(), inplace=False, apply_FW=pscb_dict['apply_FW'], weights=fourier_weights_cross)

						plt.figure()
						prefac = lb*(lb+1)/(2*np.pi)
						plt.errorbar(lb, prefac*N_ell_est, yerr=prefac*N_ell_err, fmt='o')
						plt.xscale('log')
						plt.show()

						print('N_ell_est for cross ciber is ', N_ell_est)
					else:
						mean_cl2d_nofluc = noisemodl_file['mean_cl2d_nofluc']
						N_ell_est = noisemodl_file['nl_estFF_nofluc']
						# fourier_weights_cross = None

				else:
					all_ff_ests_nofluc = None

					print('before if iterate statement ')
					# load the MC FF estimates obtained by several draws of noise
					if pscb_dict['mkk_ffest_hybrid']:

						all_ff_ests_nofluc, all_ff_ests_nofluc_cross = None, None 

						if not ciber_cross_ciber:
							all_ff_ests_nofluc, all_ff_ests_nofluc_cross = cbps.collect_ff_realiz_estimates(fieldidx, run_name, fpath_dict, pscb_dict, config_dict, float_param_dict, datestr=datestr, ff_min=float_param_dict['ff_min'], ff_max=float_param_dict['ff_max'])

					if pscb_dict['verbose']:
						print('median obs is ', np.median(obs))
						print('simmap dc is ', simmap_dc)

					cross_shot_sigma_sb_zl_noisemodl = None
					shot_sigma_sb_zl_noisemodl = cbps.compute_shot_sigma_map(inst, image=obs_levels[fieldidx]*np.ones_like(obs), nfr=nfr_fields[fieldidx])

					if ciber_cross_ciber:
						print('obs levels cross  is ', obs_levels_cross[fieldidx])
						cross_shot_sigma_sb_zl_noisemodl = cbps.compute_shot_sigma_map(cross_inst, image=obs_levels_cross[fieldidx]*np.ones_like(obs), nfr=nfr_fields[fieldidx])

					# todo cross
					if ciber_cross_ciber:
						plot_map(shot_sigma_sb_zl_noisemodl, title='shot_sigma_sb_zl_noisemodl')
						plot_map(cross_shot_sigma_sb_zl_noisemodl, title='cross_shot_sigma_sb_zl_noisemodl')

						if pscb_dict['mkk_ffest_hybrid']:
							simmap_dc_use = simmaps_dc_all[fieldidx] 
							simmap_dc_cross_use = obs_levels_cross[fieldidx]*np.ones_like(simmap_dc_use)
						else:
							simmap_dc_use, simmap_dc_cross_use = None, None

						fourier_weights_cross, mean_cl2d_cross, \
								mean_nl2d_nAsB, var_nl2d_nAsB,\
								 mean_nl2d_nBsA, var_nl2d_nBsA, \
								 nl1ds_nAnB, nl1ds_nAsB, nl1ds_nBsA = cbps.estimate_cross_noise_ps(inst, cross_inst, ifield_noise_list[fieldidx], nsims=float_param_dict['n_FW_sims'], n_split=float_param_dict['n_FW_split'],\
														mask=target_mask, read_noise=pscb_dict['with_inst_noise'], noise_model=read_noise_models[fieldidx], cross_noise_model=read_noise_models_cross[fieldidx], \
														 photon_noise=pscb_dict['with_photon_noise'], shot_sigma_sb=shot_sigma_sb_zl_noisemodl, cross_shot_sigma_sb=cross_shot_sigma_sb_zl_noisemodl, \
														 simmap_dc=simmap_dc_use, simmap_dc_cross=simmap_dc_cross_use, image=obs, image_cross=observed_ims_cross[fieldidx], \
														  mc_ff_estimates = None, mc_ff_estimates_cross=None, gradient_filter=pscb_dict['gradient_filter'], \
														  inplace=False, show=False)

						plot_map(np.log10(fourier_weights_cross), title='log10 fourier_weights_cross')
						plot_map(mean_cl2d_cross, title='log10 mean_nl2d_cross')

						mean_nl2d_cross_total = mean_cl2d_cross + mean_nl2d_nAsB + mean_nl2d_nBsA
						fourier_weights_cross = 1./((1./fourier_weights_cross) + var_nl2d_nAsB + var_nl2d_nBsA)

						plot_map(mean_nl2d_cross_total, title='mean_nl2d_cross_total')
						plot_map(fourier_weights_cross, title='fourier_weights_cross_total')

						lb, N_ell_est, N_ell_err = cbps.compute_noise_power_spectrum(inst, noise_Cl2D=mean_nl2d_cross_total.copy(), inplace=False, apply_FW=pscb_dict['apply_FW'], weights=fourier_weights_cross)

						plt.figure()
						prefac = lb*(lb+1)/(2*np.pi)
						plt.errorbar(lb, prefac*N_ell_est, yerr=prefac*N_ell_err, fmt='o')
						plt.xscale('log')
						
						plt.show()

					else:
						simmap_dc_use = None
						if pscb_dict['mkk_ffest_hybrid']:
							simmap_dc_use = simmaps_dc_all[fieldidx] 

						if pscb_dict['compute_ps_per_quadrant']:

							fourier_weights_nofluc_per_quadrant, mean_cl2d_nofluc_per_quadrant, N_ell_est_per_quadrant = [[] for x in range(3)]

							for q in range(4):
								target_mask_indiv = target_mask[cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]]
								all_ff_ests_nofluc_indiv = all_ff_ests_nofluc[:, cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]]
								shot_sigma_sb_zl_indiv = shot_sigma_sb_zl[fieldidx, cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]]

								fourier_weights_nofluc_indiv, mean_cl2d_nofluc_indiv = cbps_quad.estimate_noise_power_spectrum(nsims=float_param_dict['n_FW_sims'], n_split=float_param_dict['n_FW_split'], apply_mask=pscb_dict['apply_mask'], \
											   noise_model=read_noise_models_per_quad[q][fieldidx], read_noise=pscb_dict['with_inst_noise'], inst=inst, ifield=ifield_noise_list[fieldidx], show=False, mask=target_mask_indiv, \
												photon_noise=pscb_dict['with_photon_noise'], ff_estimate=None, shot_sigma_sb=shot_sigma_sb_zl_indiv, inplace=False, \
												field_nfr=nfr_fields[fieldidx], simmap_dc=simmap_dc_use[cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]], ff_truth=smooth_ff, mc_ff_estimates=all_ff_ests_nofluc_indiv, gradient_filter=pscb_dict['gradient_filter'],\
												chisq=False)
								lb, N_ell_est_indiv, N_ell_err_indiv = cbps_quad.compute_noise_power_spectrum(inst, noise_Cl2D=mean_cl2d_nofluc_indiv.copy(), inplace=False, apply_FW=pscb_dict['apply_FW'], weights=fourier_weights_nofluc_indiv)
								N_ell_est_per_quadrant.append(N_ell_est_indiv)

								N_ells_est_per_quadrant[q, fieldidx] = N_ell_est_indiv

								fourier_weights_nofluc_per_quadrant.append(fourier_weights_nofluc_indiv)
								mean_cl2d_nofluc_per_quadrant.append(mean_cl2d_nofluc_indiv)

								if not pscb_dict['shut_off_plots']:
									plot_map(np.log10(fourier_weights_nofluc_indiv), title='log10 fourier_weights_cross quad '+str(q))
									plot_map(mean_cl2d_nofluc_indiv, title='log10 mean_cl2d_cross quad '+str(q))

						else:
							if not pscb_dict['shut_off_plots']:
								plot_map(target_mask*shot_sigma_sb_zl_noisemodl, title='shot noise sigma ifield '+str(ifield_noise_list[fieldidx]))
							
							fourier_weights_nofluc, mean_cl2d_nofluc = cbps.estimate_noise_power_spectrum(nsims=float_param_dict['n_FW_sims'], n_split=float_param_dict['n_FW_split'], apply_mask=pscb_dict['apply_mask'], \
										   noise_model=read_noise_models[fieldidx], read_noise=pscb_dict['with_inst_noise'], inst=inst, ifield=ifield_noise_list[fieldidx], show=False, mask=target_mask, \
											photon_noise=pscb_dict['with_photon_noise'], ff_estimate=None, shot_sigma_sb=shot_sigma_sb_zl_noisemodl, inplace=False, \
											field_nfr=nfr_fields[fieldidx], simmap_dc=simmap_dc_use, ff_truth=smooth_ff, mc_ff_estimates=all_ff_ests_nofluc, gradient_filter=pscb_dict['gradient_filter'],\
											chisq=False, per_quadrant=pscb_dict['per_quadrant'], point_src_comp=point_src_comps_for_ff[fieldidx])


							# fourier_weights_nofluc, mean_cl2d_nofluc = cbps.estimate_noise_power_spectrum(nsims=float_param_dict['n_FW_sims'], n_split=float_param_dict['n_FW_split'], apply_mask=pscb_dict['apply_mask'], \
							# 			   noise_model=read_noise_models[fieldidx], read_noise=pscb_dict['with_inst_noise'], inst=inst, ifield=ifield_noise_list[fieldidx], show=False, mask=target_mask, \
							# 				photon_noise=pscb_dict['with_photon_noise'], ff_estimate=None, shot_sigma_sb=shot_sigma_sb_zl_noisemodl, inplace=False, \
							# 				field_nfr=nfr_fields[fieldidx], simmap_dc=simmap_dc_use, ff_truth=smooth_ff, mc_ff_estimates=all_ff_ests_nofluc, gradient_filter=pscb_dict['gradient_filter'],\
							# 				chisq=False, per_quadrant=pscb_dict['per_quadrant'], point_src_comp=point_src_comps_for_ff[fieldidx])

							lb, N_ell_est, N_ell_err = cbps.compute_noise_power_spectrum(inst, noise_Cl2D=mean_cl2d_nofluc.copy(), inplace=False, apply_FW=pscb_dict['apply_FW'], weights=fourier_weights_nofluc)
							
							if not pscb_dict['shut_off_plots']:
								plot_map(np.log10(fourier_weights_nofluc), title='log10 fourier_weights')
								plot_map(mean_cl2d_nofluc, title='log10 mean_cl2d_cross')

								plt.figure()
								prefac = lb*(lb+1)/(2*np.pi)
								plt.errorbar(lb, prefac*N_ell_est, yerr=prefac*N_ell_err, fmt='o')
								plt.xscale('log')
								plt.yscale('log')
								plt.show()

							if pscb_dict['verbose']:
								print('N_ell_est = ', N_ell_est)

					if fpath_dict['noisemodl_basepath'] is None:
						fpath_dict['noisemodl_basepath'] = fpath_dict['ciber_mock_fpath']+'030122/noise_models_sim/'

					if fpath_dict['noisemodl_run_name'] is None:
						fpath_dict['noisemodl_run_name'] = run_name 

					noisemodl_tailpath = '/noise_bias_fieldidx'+str(fieldidx)+'.npz'
					noisemodl_fpath = fpath_dict['noisemodl_basepath'] + fpath_dict['noisemodl_run_name'] + noisemodl_tailpath

					if pscb_dict['save'] or not os.path.exists(noisemodl_fpath):
						print('SAVING NOISE MODEL BIAS FILE..')
						if os.path.exists(noisemodl_fpath):
							print('Overwriting existing noise model bias file..')

						if ciber_cross_ciber:
							np.savez(noisemodl_fpath, fourier_weights_nofluc=fourier_weights_cross, mean_cl2d_nofluc=mean_nl2d_cross_total, nl_estFF_nofluc=N_ell_est)
						else:
							np.savez(noisemodl_fpath, fourier_weights_nofluc=fourier_weights_nofluc, mean_cl2d_nofluc=mean_cl2d_nofluc, nl_estFF_nofluc=N_ell_est)
							
							if pscb_dict['compute_ps_per_quadrant']:
								for q in range(4):
									noisemodl_tailpath_indiv = '/noise_bias_fieldidx'+str(fieldidx)+'_quad'+str(q)+'.npz'
									noisemodl_fpath_indiv = fpath_dict['noisemodl_basepath'] + fpath_dict['noisemodl_run_name'] + noisemodl_tailpath_indiv
									np.savez(noisemodl_fpath_indiv, fourier_weights_nofluc=fourier_weights_nofluc_per_quadrant[q], mean_cl2d_nofluc=mean_cl2d_nofluc_per_quadrant[q], \
										nl_estFF_nofluc=N_ells_est_per_quadrant[q, fieldidx])

				if ciber_cross_ciber:
					cbps.FW_image=fourier_weights_cross.copy()
				else:
					cbps.FW_image=fourier_weights_nofluc.copy()
				
				N_ells_est[fieldidx] = N_ell_est

			if pscb_dict['load_ptsrc_cib'] or pscb_dict['load_trilegal'] or data_type=='observed':
				if config_dict['ps_type']=='auto' or config_dict['cross_type'] == 'ciber':
				# if config_dict['ps_type']=='auto' or config_dict['cross_type'] != 'dgl' or config_dict['cross_type'] != 'cmblens':
					print('setting beam correct to True..')
					beam_correct = True
				else:
					if pscb_dict['verbose']:
						print('setting beam_correct to False..')
					beam_correct = False

				B_ell_field = B_ells[fieldidx]

				if pscb_dict['compute_ps_per_quadrant']:
					B_ell_field_quad = B_ells_quad[fieldidx]

				if ciber_cross_ciber:
					print('Loading B_ell for cross field TM ', cross_inst)
					B_ell_field_cross = B_ells_cross[fieldidx]
					B_ell_field = np.sqrt(B_ell_field_cross*B_ell_field)

			else:
				beam_correct = False
				B_ell_field = None

			if pscb_dict['iterate_grad_ff']: # set gradient_filter, FF_correct to False here because we've already done gradient filtering/FF estimation
				if pscb_dict['verbose']:
					print('iterate_grad_ff is True, setting gradient filter and FF correct to False..')
				gradient_filter = False
				FF_correct = False
			else:
				gradient_filter=pscb_dict['gradient_filter']
				FF_correct = pscb_dict['ff_estimate_correct']

			# if not pscb_dict['shut_off_plots']:
			# 	plot_map(obs*target_mask, figsize=(5,5), title='line 1206 ifield '+str(ifield))
			print('sum of target mask is ', np.sum(target_mask))

			if pscb_dict['verbose']:
				print('obs mean is ', np.median(obs[target_mask==1]))

			if pscb_dict['mkk_ffest_hybrid'] and pscb_dict['verbose']:
				print('Correcting power spectrum with hybrid mask-FF mode coupling matrix..')

			if pscb_dict['show_plots'] and not pscb_dict['shut_off_plots']:
				maskobs = obs*target_mask

				if pscb_dict['per_quadrant']:
					for q in range(4):
						mquad = target_mask[cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]]
						maskobs[cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]][mquad==1] -= np.mean(maskobs[cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]][mquad==1])

				else:
					maskobs[maskobs!=0]-=np.mean(maskobs[maskobs!=0])
				# if not pscb_dict['shut_off_plots']:
				plot_map(maskobs, title='FF corrected, mean subtracted CIBER map ('+cbps.ciber_field_dict[ifield_list[fieldidx]]+')', hipct=99.999, lopct=0., cmap='bwr')

				maskobs_smooth = target_mask*gaussian_filter(maskobs, sigma=10)
				maskobs_smooth[target_mask==1]-=np.mean(maskobs_smooth[target_mask==1])
				# if not pscb_dict['shut_off_plots']:
				plot_map(maskobs_smooth, title='FF corrected, smoothed, mean-subtracted CIBER map ('+cbps.ciber_field_dict[ifield_list[fieldidx]]+')', cmap='bwr', vmax=20, vmin=-20)


			if config_dict['ps_type']=='auto':

				if pscb_dict['max_val_clip']:
					max_val_after_sub = max_vals_ptsrc[fieldidx]
				else:
					max_val_after_sub = None

				if i-config_dict['simidx0'] > 5:
					verb = False 
				else:
					verb = pscb_dict['verbose']

				lb, processed_ps_nf, cl_proc_err, masked_image = cbps.compute_processed_power_spectrum(inst, apply_mask=pscb_dict['apply_mask'], \
												 mask=target_mask, image=obs, convert_adufr_sb=False, \
												mkk_correct=pscb_dict['apply_mask'], inv_Mkk=target_invMkk, beam_correct=beam_correct, B_ell=B_ell_field, \
												apply_FW=pscb_dict['apply_FW'], verbose=verb, noise_debias=pscb_dict['noise_debias'], \
											 FF_correct=FF_correct, FW_image=fourier_weights_nofluc, FF_image=ff_estimates[fieldidx],\
												 gradient_filter=gradient_filter, save_intermediate_cls=pscb_dict['save_intermediate_cls'], N_ell=N_ells_est[fieldidx], \
												 per_quadrant=pscb_dict['per_quadrant'], max_val_after_sub=max_val_after_sub)

				final_masked_images[fieldidx] = masked_image


				if pscb_dict['compute_ps_per_quadrant']:
					processed_ps_per_quad, processed_ps_err_per_quad = [np.zeros((4, cbps_quad.n_ps_bin)) for x in range(2)]
					for q in range(4):
						target_mask_indiv = target_mask[cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]]
						obs_indiv = obs[cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]]
						lb_quad, processed_ps_indiv, cl_proc_err_indiv, _ = cbps_quad.compute_processed_power_spectrum(inst, apply_mask=pscb_dict['apply_mask'], \
									 mask=target_mask_indiv, image=obs_indiv, convert_adufr_sb=False, \
									mkk_correct=pscb_dict['apply_mask'], inv_Mkk=target_invMkks_per_quadrant[q], beam_correct=beam_correct, B_ell=B_ell_field_quad, \
									apply_FW=pscb_dict['apply_FW'], verbose=pscb_dict['verbose'], noise_debias=pscb_dict['noise_debias'], \
								 FF_correct=FF_correct, FW_image=fourier_weights_nofluc_per_quadrant[q], FF_image=ff_estimates[fieldidx,cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]],\
									 gradient_filter=gradient_filter, save_intermediate_cls=pscb_dict['save_intermediate_cls'], N_ell=N_ells_est_per_quadrant[q, fieldidx], \
									 per_quadrant=False)
						processed_ps_per_quad[q] = processed_ps_indiv
						processed_ps_err_per_quad[q] = cl_proc_err_indiv


			elif config_dict['ps_type']=='cross':

				if pscb_dict['show_plots']:
					maskobs_cross = observed_ims_cross[fieldidx]*target_mask
					maskobs_cross[target_mask==1]-=np.mean(maskobs_cross[target_mask==1])

					if config_dict['cross_type']=='IRIS':
						plot_map(maskobs_cross, title='Mean-subtracted IRIS map ('+cbps.ciber_field_dict[ifield_list[fieldidx]]+')', hipct=99.999, lopct=0., cmap='bwr')
					elif config_dict['cross_type']=='ciber':

						maskobs_cross_smooth = target_mask*gaussian_filter(maskobs_cross, sigma=10)
						maskobs_cross_smooth[target_mask==1] -= np.mean(maskobs_cross_smooth[target_mask==1])
						plot_map(maskobs_cross_smooth, title='Mean-subtracted TM2 map ('+cbps.ciber_field_dict[ifield_list[fieldidx]]+')', vmin=-20, vmax=20, cmap='bwr')
						plot_map(maskobs_cross, title='Mean-subtracted TM2 map ('+cbps.ciber_field_dict[ifield_list[fieldidx]]+')', hipct=99.999, lopct=0., cmap='bwr')


				if ciber_cross_ciber:
					tl_regrid_fpath = fpath_dict['tls_base_path']+'tl_regrid_tm2_to_tm1_ifield4_041323.npz'
					tl_regrid = np.load(tl_regrid_fpath)['tl_regrid']
				else:
					tl_regrid = None


				lb, processed_ps_nf, cl_proc_err, _ = cbps.compute_processed_power_spectrum(inst, apply_mask=pscb_dict['apply_mask'], \
												 mask=target_mask, image=obs, cross_image=observed_ims_cross[fieldidx], convert_adufr_sb=False, \
												mkk_correct=pscb_dict['apply_mask'], inv_Mkk=target_invMkk, beam_correct=beam_correct, B_ell=B_ell_field, \
												apply_FW=pscb_dict['apply_FW'], verbose=pscb_dict['verbose'], noise_debias=False, \
											 FF_correct=FF_correct, FF_image=ff_estimates[fieldidx], FW_image=fourier_weights_cross, \
												 gradient_filter=False, tl_regrid=tl_regrid, save_intermediate_cls=pscb_dict['save_intermediate_cls'], N_ell=N_ells_est[fieldidx])


			if pscb_dict['save_intermediate_cls']:
				cls_masked_prenl[fieldidx,:] = cbps.masked_Cl_pre_Nl_correct
				cls_masked_postnl[fieldidx,:] = cbps.masked_Cl_post_Nl_correct
				cls_postmkk_prebl[fieldidx,:] = cbps.cl_post_mkk_pre_Bl
				cls_postbl[fieldidx,:] = processed_ps_nf.copy()

			if pscb_dict['ff_bias_correct']:
				if pscb_dict['verbose']:
					print('before ff bias correct ps is ', processed_ps_nf)
					print('correcting', fieldidx, ' with ff bias of ', ff_biases[fieldidx])
				processed_ps_nf /= ff_biases[fieldidx]
				cl_proc_err /= ff_biases[fieldidx]

				if pscb_dict['save_intermediate_cls']:
					cls_postffb_corr[fieldidx,:] = processed_ps_nf.copy()

			if pscb_dict['iterate_grad_ff']: # .. but we want gradient filtering still for the ground truth signal.
				gradient_filter = True

			cl_prenlcorr = cbps.masked_Cl_pre_Nl_correct.copy()

			# if using CIB model, ground truth for CIB is loaded from Helgason J > Jlim and added to ground truth of DGL-like component
			if data_type=='mock':
				if pscb_dict['load_ptsrc_cib']: # false for observed

					lb, diff_cl, cl_proc_err_mock, _ = cbps.compute_processed_power_spectrum(inst, apply_mask=False, \
														 image=diff_realizations[fieldidx], convert_adufr_sb=False, \
														mkk_correct=False, beam_correct=True, B_ell=B_ell_field, \
														apply_FW=False, verbose=False, noise_debias=False, \
													 FF_correct=False, save_intermediate_cls=pscb_dict['save_intermediate_cls'], gradient_filter=False) # gradient is False for now 082222
					
					cib_cl_file = fits.open(fpath_dict['cib_resid_ps_path']+'/cls_cib_vs_maglim_ifield'+str(ifield_list[fieldidx])+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_Vega_magcut.fits')
					unmasked_cib_cl = cib_cl_file['cls_cib'].data['cl_maglim_'+str(masking_maglim)]
					true_ps = unmasked_cib_cl + diff_cl
					if pscb_dict['verbose']:
						print('unmasked cib cl', unmasked_cib_cl)

					if pscb_dict['load_trilegal'] and masking_maglim < 16:

						print('Adding power spectrum from all stars fainter than masking limit..')
						if pscb_dict['subtract_subthresh_stars']:
							print('Setting subtract_subthresh_stars to False..')
							pscb_dict['subtract_subthresh_stars'] = False

					if pscb_dict['load_trilegal'] and not pscb_dict['subtract_subthresh_stars']:
						trilegal_cl_fpath = fpath_dict['isl_resid_ps_path']+'/cls_isl_vs_maglim_ifield'+str(ifield_list[fieldidx])+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_Vega_magcut.fits'
						trilegal_cl = fits.open(trilegal_cl_fpath)['cls_isl'].data['cl_maglim_'+str(masking_maglim)]
						true_ps += trilegal_cl 
						if pscb_dict['verbose']:
							print('trilegal_cl is ', trilegal_cl)
							print('and true cl is now ', true_ps)

				elif pscb_dict['load_trilegal']:
					if pscb_dict['verbose']:
						print('Ground truth in this case is the ISL fainter than our masking depth..')

					trilegal_cl_fpath = fpath_dict['isl_resid_ps_path']+'/cls_isl_vs_maglim_ifield'+str(ifield_list[fieldidx])+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_Vega_magcut_test.fits'
					# trilegal_cl_fpath = trilegal_base_path+'TM'+str(inst)+'/powerspec/cls_trilegal_vs_maglim_ifield'+str(ifield_list[fieldidx])+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_Vega_magcut_test.fits'
					true_ps = fits.open(trilegal_cl_fpath)['cls_isl'].data['cl_maglim_'+str(masking_maglim)]


				else:
					full_realiz = mock_cib_ims[fieldidx]+diff_realizations[fieldidx]
					full_realiz -= np.mean(full_realiz)
					if pscb_dict['show_plots']:
						plot_map(full_realiz, title='full realiz fieldidx '+str(fieldidx))
					lb, true_ps, cl_proc_err_mock, _ = cbps.compute_processed_power_spectrum(inst, apply_mask=False, \
														 image=full_realiz, convert_adufr_sb=False, \
														mkk_correct=False, beam_correct=beam_correct, B_ell=None, \
														apply_FW=False, verbose=False, noise_debias=False, \
													 FF_correct=False, save_intermediate_cls=pscb_dict['save_intermediate_cls'], gradient_filter=pscb_dict['gradient_filter'])


			if pscb_dict['transfer_function_correct']:
				if pscb_dict['verbose']:
					print('Correcting for transfer function..')
					print('t_ell_av here is ', t_ell_av)

				processed_ps_nf /= t_ell_av

				if pscb_dict['compute_ps_per_quadrant']:
					if pscb_dict['verbose']:
						print('correcting per quadrant ps for transfer function..')
					for q in range(4):
						processed_ps_per_quad[q] /= t_ell_av_quad
						processed_ps_err_per_quad[q] /= t_ell_av_quad

				if pscb_dict['save_intermediate_cls']:
					cls_post_tcorr[fieldidx,:] = processed_ps_nf.copy()

			
			if pscb_dict['subtract_subthresh_stars']:
				# maybe load precomputed power spectra and assume we have it nailed down
				if pscb_dict['verbose']:
					print('before trilegal star correct correct ps is ', processed_ps_nf)
					print('subtracting sub-threshold stars!!!!')
					print('isl resid ps path:', fpath_dict['isl_resid_ps_path'])

				trilegal_cl_fpath = fpath_dict['isl_resid_ps_path']+'/cls_isl_vs_maglim_ifield'+str(ifield_list[fieldidx])+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_Vega_magcut.fits'
				
				trilegal_cl = fits.open(trilegal_cl_fpath)['cls_isl'].data['cl_maglim_'+str(masking_maglim)]
				if pscb_dict['verbose']:
					print(trilegal_cl)
				processed_ps_nf -= trilegal_cl

				if pscb_dict['save_intermediate_cls']:
					cls_post_isl_sub[fieldidx,:] = processed_ps_nf.copy()

			if pscb_dict['verbose']:
				print('proc:', processed_ps_nf)


			recovered_power_spectra[i, fieldidx, :] = processed_ps_nf
			recovered_dcl[i, fieldidx, :] = cl_proc_err

			if pscb_dict['compute_ps_per_quadrant']:

				recovered_power_spectra_per_quadrant[:, i, fieldidx] = processed_ps_per_quad
				recovered_dcl_per_quadrant[:, i, fieldidx] = processed_ps_err_per_quad

			if pscb_dict['show_ps'] and i<10:
				plt.figure(figsize=(6, 5))
				prefac = lb*(lb+1)/(2*np.pi)
				if data_type=='mock':
					plt.errorbar(lb, prefac*true_ps, yerr=prefac*cl_proc_err, fmt='o-', capsize=3, color='k', label='ground truth')
				plt.errorbar(lb, prefac*processed_ps_nf, yerr=prefac*cl_proc_err, fmt='o-', capsize=3, color='r', label='recovered')
				plt.plot(lb, prefac*N_ells_est[fieldidx], linestyle='dashed', color='b', label='Noise bias')
				plt.legend(fontsize=14)
				plt.yscale('log')
				plt.xscale('log')
				plt.ylim(1e-2, max(1e4, 1.2*np.max(prefac*processed_ps_nf)))
				plt.grid()
				plt.title('ifield '+str(ifield_list[fieldidx]), fontsize=16)
				plt.xlabel('$\\ell$', fontsize=18)
				plt.ylabel('$D_{\\ell}$', fontsize=18)
				plt.show()

				if pscb_dict['compute_ps_per_quadrant']:
					plt.figure(figsize=(10, 8))
					prefac_quad = lb_quad*(lb_quad+1)/(2*np.pi)
					for q in range(4):
						plt.subplot(2,2,q+1)
						plt.errorbar(lb_quad, prefac*processed_ps_per_quad[q], yerr=prefac_quad*processed_ps_per_quad[q], fmt='o-', capsize=3, color='C'+str(q), label='Recovered PS (indiv quadrant)')
						plt.plot(lb_quad, prefac_quad*N_ells_est_per_quadrant[q, fieldidx], linestyle='dashed', color='C'+str(q), label='Noise bias (indiv quadrant)')
						plt.errorbar(lb, prefac*processed_ps_nf, yerr=prefac*cl_proc_err, fmt='o-', capsize=3, color='k', label='Recovered PS (full map)')
						plt.plot(lb, prefac*N_ells_est[fieldidx], linestyle='dashed', color='b', label='Noise bias (full map)')

						plt.legend(fontsize=12)
						plt.yscale('log')
						plt.xscale('log')
						plt.ylim(1e-2, 2e4)
						plt.grid()
						plt.title('ifield '+str(ifield_list[fieldidx])+', quad '+str(q), fontsize=14)
						plt.xlabel('$\\ell$', fontsize=18)
						plt.ylabel('$D_{\\ell}$', fontsize=18)
					plt.tight_layout()
					plt.show()

			if data_type=='mock':
				signal_power_spectra[i, fieldidx, :] = true_ps

				if pscb_dict['verbose']:

					print('true ps', true_ps)
					print('proc (nofluc) / ebl = ', processed_ps_nf/true_ps)
					print('mean ps bias is ', np.mean(recovered_power_spectra[i,:,:]/signal_power_spectra[i,:,:], axis=0))



		if not pscb_dict['save_intermediate_cls']:
			cls_inter, inter_labels = None, None 
		else:
			cls_inter.extend([cls_masked_prenl, cls_masked_postnl, cls_postmkk_prebl, cls_postbl, cls_postffb_corr, cls_post_tcorr, cls_post_isl_sub])
			inter_labels.extend(['masked_Cl_pre_Nl_correct', 'masked_Cl_post_Nl_correct', 'post_mkk_pre_Bl', 'post_Bl_pre_ffbcorr', 'post_ffb_corr','post_tcorr', 'post_isl_sub'])
			
			if pscb_dict['verbose']:
				print('cls inter has shape', np.array(cls_inter).shape)
				print('inter labels has length', len(inter_labels))
				
		if pscb_dict['save']:
			savestr = fpath_dict['sim_test_fpath']+run_name+'/input_recovered_ps_estFF_simidx'+str(i)

			if fpath_dict['add_savestr'] is not None:
				savestr += '_'+fpath_dict['add_savestr']

			if pscb_dict['compute_ps_per_quadrant']:
				savestr += '_per_quadrant'
				np.savez(savestr+'.npz', lb=lb_quad, data_type=data_type, \
					signal_ps=signal_power_spectra[i], recovered_ps_est_per_quadrant=recovered_power_spectra_per_quadrant[:,i], recovered_dcl_per_quadrant=recovered_dcl_per_quadrant[:,i], \
						 dgl_scale_fac=float_param_dict['dgl_scale_fac'], niter=float_param_dict['niter'], cls_inter=None, inter_labels=inter_labels, **pscb_dict)
			else:
				np.savez(savestr+'.npz', lb=lb, data_type=data_type, \
					signal_ps=signal_power_spectra[i], recovered_ps_est_nofluc=recovered_power_spectra[i], recovered_dcl=recovered_dcl[i], \
						 dgl_scale_fac=float_param_dict['dgl_scale_fac'], niter=float_param_dict['niter'], cls_inter=cls_inter, inter_labels=inter_labels, **pscb_dict)


			if i==0:
				with open(fpath_dict['sim_test_fpath']+run_name+'/params_read.txt', 'w') as file:
					for dicto in [fpath_dict, pscb_dict, float_param_dict, config_dict]:
						for key in dicto:
							file.write(key+': '+str(dicto[key])+'\n')

	if pscb_dict['compute_ps_per_quadrant']:
		return lb_quad, signal_power_spectra, recovered_power_spectra_per_quadrant, recovered_dcl_per_quadrant, N_ells_est_per_quadrant, None, None


	return lb, signal_power_spectra, recovered_power_spectra, recovered_dcl, N_ells_est, cls_inter, inter_labels, ff_estimates, final_masked_images


