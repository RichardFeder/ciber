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
from ciber_noise_data_utils import *
from cross_spectrum_analysis import *
from ciber_data_helpers import load_psf_params_dict
from plotting_fns import plot_map
from ciber_mocks import *
from flat_field_est import *
from mkk_parallel import compute_inverse_mkk, plot_mkk_matrix
from masking_utils import *


def init_mocktest_fpaths(ciber_mock_fpath, run_name):
	ff_fpath = ciber_mock_fpath+'030122/ff_realizations/'+run_name+'/'
	noisemod_fpath = ciber_mock_fpath +'030122/noise_models_sim/'+run_name+'/'
	input_recov_ps_fpath = 'data/input_recovered_ps/sim_tests_030122/'+run_name+'/'

	fpaths = [input_recov_ps_fpath, ff_fpath, noisemod_fpath]
	for fpath in fpaths:
		if not os.path.isdir(fpath):
			print('making directory path for ', fpath)
			os.makedirs(fpath)
		else:
			print(fpath, 'already exists')

	return ff_fpath, noisemod_fpath, input_recov_ps_fpath

def Merge(dict1, dict2):
	res = dict1.copy()
	res.update(dict2)
	# res = dict1.update(dict2)
	# res = {**dict1, **dict2}
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


def run_cbps_pipeline(cbps, inst, nsims, run_name, ifield_list = None, \
							datestr='100921', datestr_trilegal=None, data_type='mock',\
							masking_maglim=17.5, mask_tail='abc110821', mask_tail_ffest=None, \
							zl_levels=None, ff_biases=None, **kwargs):

	config_dict = dict({'ps_type':'auto', 'cross_type':'ciber', 'cross_inst':2, 'cross_gal':None, 'simidx0':0, \
						'full_mask_tail':'maglim_17_Vega_test', 'bright_mask_tail':'maglim_11_Vega_test'})

	pscb_dict = dict({'apply_mask':True, 'with_inst_noise':True, 'with_photon_noise':True, 'apply_FW':True, \
					'apply_smooth_FF':True, 'compute_beam_correction':True, 'same_zl_levels':False, 'apply_zl_gradient':True, 'gradient_filter':True, \
					'iterate_grad_ff':True , 'same_int':False, 'same_dgl':True, 'use_ff_weights':True, 'plot_ff_error':False, 'load_ptsrc_cib':True, \
					'load_trilegal':True , 'subtract_subthresh_stars':True, 'ff_bias_correct':True, 'save_ff_ests':True, 'plot_maps':True, \
					 'draw_cib_setidxs':False, 'aug_rotate':False, 'load_noise_bias':False, 'transfer_function_correct':False, 'compute_transfer_function':False,\
					  'save_intermediate_cls':True, 'verbose':True, 'save':True})

	float_param_dict = dict({'ff_min':0.5, 'ff_max':2.0, 'clip_sigma':5, 'ff_stack_min':2, 'nmc_ff':10, \
					  'theta_mag':0.01, 'niter':10, 'dgl_scale_fac':5, 'smooth_sigma':5, 'indiv_ifield':6,\
					  'nfr_same':25, 'J_bright_Bl':11, 'J_faint_Bl':17.5, 'n_FW_sims':500, 'n_FW_split':10, \
					  'ell_norm_blest':5000, 'n_realiz_t_ell':100})

	fpath_dict = dict({'ciber_mock_fpath':'/Volumes/Seagate Backup Plus Drive/Toolkit/Mirror/Richard/ciber_mocks/', \
						'sim_test_fpath':'data/input_recovered_ps/sim_tests_030122/', \
						'base_fluc_path':'/Volumes/Seagate Backup Plus Drive/Toolkit/Mirror/Richard/ciber_fluctuation_data/', \
						'ff_smooth_fpath':'data/flatfield/TM1_FF/TM1_field4_mag=17.5_FF.fits', \
						'observed_base_path':'data/fluctuation_data/', \
						'mask_base_path':None, 'mkk_base_path':None, 'bls_base_path':None, 'isl_rms_fpath':None, 'bls_fpath':None, \
						't_ell_fpath':None, 'tls_base_path':None})
	
	config_dict, pscb_dict, float_param_dict, fpath_dict = update_dicts([config_dict, pscb_dict, float_param_dict, fpath_dict], kwargs)

	base_path = fpath_dict['ciber_mock_fpath']+datestr+'/'

	
	trilegal_base_path = fpath_dict['ciber_mock_fpath']+datestr_trilegal+'/'

	if fpath_dict['mask_base_path'] is None:
		if data_type=='mock':
			fpath_dict['mask_base_path'] = base_path+'TM'+str(inst)+'/masks/'
		elif data_type=='observed':
			fpath_dict['mask_base_path'] = fpath_dict['observed_base_path']+'TM'+str(inst)+'/masks/'

		print('setting mask_base_path to ', fpath_dict['mask_base_path'])
	if fpath_dict['mkk_base_path'] is None:
		if data_type=='mock':
			fpath_dict['mkk_base_path'] = base_path+'TM'+str(inst)+'/mkk/'
		elif data_type=='observed':
			fpath_dict['mkk_base_path'] = fpath_dict['observed_base_path']+'TM'+str(inst)+'/mkk/'
		print('setting mkk_base_path to ', fpath_dict['mkk_base_path'])


	if fpath_dict['bls_base_path'] is None:
		if data_type=='mock':
			fpath_dict['bls_base_path'] = base_path+'TM'+str(inst)+'/beam_correction/'
		elif data_type=='observed':
			fpath_dict['bls_base_path'] = fpath_dict['observed_base_path']+'TM'+str(inst)+'/beam_correction/'

		print('setting bls_base_path to ', fpath_dict['bls_base_path'])
	if fpath_dict['tls_base_path'] is None:
		fpath_dict['tls_base_path'] = base_path+'transfer_function/'
		print('tls base path is ', fpath_dict['tls_base_path'])

	list_of_dirpaths = [fpath_dict['mask_base_path'], fpath_dict['mkk_base_path']]
	if data_type=='mock':
		list_of_dirpaths.append(fpath_dict['tls_base_path'])
		list_of_dirpaths.append(fpath_dict['bls_base_path'])
	print(list_of_dirpaths)
	make_fpaths(list_of_dirpaths)

	if ifield_list is None:
		ifield_list = [4, 5, 6, 7, 8]
		print('Setting ifield list to ', ifield_list)
	nfield = len(ifield_list)

	if pscb_dict['save_intermediate_cls']:
		cls_inter, inter_labels = [], [] # each cl added will have a corresponding key

	mean_isl_rms = None
	if fpath_dict['isl_rms_fpath'] is not None:
		print('Loading ISL RMS from ', fpath_dict['isl_rms_fpath']+'..')
		mean_isl_rms = load_isl_rms(fpath_dict['isl_rms_fpath'], masking_maglim, nfield)

	if not pscb_dict['load_trilegal']:
		print('load trilegal = False, setting trilegal_fpath to None..')
		trilegal_fpath=None
	else:
		if datestr_trilegal is None:
			datestr_trilegal = datestr
		
	single_ps_set_shape = (nfield, cbps.n_ps_bin)

	if data_type=='observed':
		print('DATA TYPE IS OBSERVED')
		config_dict['simidx0'], nsims = 0, 1
		pscb_dict['same_int'] = False
		pscb_dict['apply_zl_gradient'] = False
		pscb_dict['with_inst_noise'] = True
		pscb_dict['load_ptsrc_cib'] = False
		pscb_dict['apply_smooth_FF'] = False
		print('Observed data, setting same_int, apply_zl_gradient, load_ptsrc_cib to False, with_inst_noise to True..')
		
	if pscb_dict['same_int']:
		nfr_fields = [float_param_dict['nfr_same'] for i in range(nfield)]
		ifield_noise_list = [float_param_dict['indiv_ifield'] for i in range(nfield)]
		ifield_list_ff = [float_param_dict['indiv_ifield'] for i in range(nfield)]
	else:
		nfr_fields = [cbps.field_nfrs[ifield] for ifield in ifield_list]
		ifield_noise_list = ifield_list.copy()
		ifield_list_ff = ifield_list.copy()

	if zl_levels is None:
		if pscb_dict['same_zl_levels']:
			zl_levels = [cbps.zl_levels_ciber_fields[inst][cbps.ciber_field_dict[float_param_dict['indiv_ifield']]] for i in range(nfield)]
		else:
			zl_levels = [cbps.zl_levels_ciber_fields[inst][cbps.ciber_field_dict[ifield]] for ifield in ifield_list]

	if pscb_dict['verbose']:
		print('NFR fields is ', nfr_fields)
		print('ZL levels are ', zl_levels)

	if pscb_dict['use_ff_weights']:
		weights_photonly = cbps.compute_ff_weights(inst, zl_levels, read_noise_models=None, ifield_list=ifield_list_ff, additional_rms=mean_isl_rms)
		print('FF weights are :', weights_photonly)

	#  ------------------- instantiate data arrays  --------------------
	field_set_shape = (nfield, cbps.dimx, cbps.dimy)
	ps_set_shape = (nsims, nfield, cbps.n_ps_bin)
	single_ps_set_shape = (nfield, cbps.n_ps_bin)
	clus_realizations, zl_perfield, ff_estimates, observed_ims = [np.zeros(field_set_shape) for i in range(4)]
	inv_Mkks, joint_maskos, joint_maskos_ffest = None, None, None

	if pscb_dict['apply_mask']:
		joint_maskos = np.zeros(field_set_shape)

		if mask_tail_ffest is not None:
			joint_maskos_ffest = np.zeros(field_set_shape)


	signal_power_spectra, recovered_power_spectra = [np.zeros(ps_set_shape) for i in range(2)]
	# ------------------------------------------------------------------

	smooth_ff = None
	if pscb_dict['apply_smooth_FF']:
		print('loading smooth FF from ', fpath_dict['ff_smooth_fpath'])
		ff = fits.open(fpath_dict['ff_smooth_fpath'])
		smooth_ff = gaussian_filter(ff[0].data, sigma=float_param_dict['smooth_sigma'])

	ff_fpath, noisemod_fpath, input_recov_ps_fpath = init_mocktest_fpaths(fpath_dict['ciber_mock_fpath'], run_name)

	if pscb_dict['with_inst_noise']:
		read_noise, read_noise_models = [np.zeros(field_set_shape) for i in range(2)]
		for fieldidx, ifield in enumerate(ifield_noise_list):
			noise_cl2d = cbps.load_noise_Cl2D(ifield, inst, noise_fpath='data/fluctuation_data/TM'+str(inst)+'/noiseCl2D/field'+str(ifield)+'_noiseCl2D_110421.fits',\
											  inplace=False, transpose=False)
			read_noise_models[fieldidx] = noise_cl2d

	B_ells = [None for x in range(nfield)]

	if pscb_dict['transfer_function_correct']:
		fpath_dict['t_ell_fpath'] = fpath_dict['tls_base_path']+'t_ell_est_nsims='+str(float_param_dict['n_realiz_t_ell'])+'.npz'
		if pscb_dict['compute_transfer_function']:
			

			print('Computing transfer function from '+str(float_param_dict['n_realiz_t_ell']))
			lb, t_ell_av, t_ell_stderr, t_ells, cls_orig, cls_filt = cbps.calculate_transfer_function(1, float_param_dict['n_realiz_t_ell'], plot=False)
			
			print('t_ell_av is ', t_ell_av)

			if fpath_dict['t_ell_fpath'] is not None:
				print('Saving transfer function correction to ', fpath_dict['t_ell_fpath'])
				np.savez(fpath_dict['t_ell_fpath'], lb=lb, t_ell_av=t_ell_av, t_ell_stderr=t_ell_stderr, t_ells=t_ells, cls_orig=cls_orig, cls_filt=cls_filt)


		elif fpath_dict['t_ell_fpath'] is not None:
			print('Loading transfer function from ', fpath_dict['t_ell_fpath'])
			t_ell_av = np.load(fpath_dict['t_ell_fpath'])['t_ell_av']
		else:
			print('No transfer function path provided, and compute_transfer_function set to False, exiting..')
			return 


	# if not os.path.exists('data/b_ell_estimates_beam_correction_tm'+str(inst)+'.npz'):
	# 	print('Path does not exist..')
	# 	for fieldidx, ifield in enumerate(ifield_list):
	# 		lb, B_ell, B_ell_list = cbps.compute_beam_correction_posts(ifield, inst, nbins=cbps.n_ps_bin, n_fine_bin=10, tail_path='data/psf_model_dict_updated_081121_ciber.npz')
	# 		B_ells[fieldidx] = B_ell

	# 	np.savez('data/b_ell_estimates_beam_correction_tm'+str(inst)+'.npz', lb=lb, B_ells=B_ells, ifield_list=ifield_list, inst=inst, n_fine_bin=10)
	B_ells = np.load('data/fluctuation_data/TM'+str(inst)+'/beam_correction/ptsrc_blest_TM'+str(inst)+'_observed.npz')
	# B_ells = np.load('data/b_ell_estimates_beam_correction_tm'+str(inst)+'.npz')['B_ells']
	print('B_ells = ', B_ells)

	if fpath_dict['bls_fpath'] is not None:
#         if pscb_dict['compute_beam_correction']:
#             print('Computing beam correction to be saved to bls_fpath='+bls_fpath+'..')
# #             lb, diff_norm_array = estimate_b_ell_from_maps(cbps, inst, ifield_list, ciber_maps=None, pixsize=7., J_bright_mag=11., J_tot_mag=17.5, nsim=1, ell_norm=10000, plot=False, save=False, \
# #                                     niter = niter, ff_stack_min=ff_stack_min, data_type='mock')
# #             print('diff norm array:', diff_norm_array)            
            # for fieldidx, ifield in enumerate(ifield_list):
            #     lb, B_ell, B_ell_list = cbps.compute_beam_correction_posts(ifield, inst, nbins=cbps.n_ps_bin, n_fine_bin=10, tail_path='data/psf_model_dict_updated_081121_ciber.npz')
            #     B_ells[fieldidx] = B_ell
#             np.savez(bls_fpath, B_ells=B_ells, ifield_list=ifield_list)
		
        # else:
		if not pscb_dict['compute_beam_correction']:
			print('loading B_ells from ', fpath_dict['bls_fpath'])
			B_ells = np.load(fpath_dict['bls_fpath'])['B_ells']

	# loop through simulations
	for i in np.arange(config_dict['simidx0'], nsims):
		pscb_dict['verbose'] = False
		if i==0:
			pscb_dict['verbose'] = True

		if data_type=='mock':
			print('Simulation set '+str(i)+' of '+str(nsims)+'..')
		elif data_type=='observed':
			print('Running observed data..')

		cib_setidxs, inv_Mkks, orig_mask_fractions = [], [], []
#         diff_realizations = np.zeros(field_set_shape)
		cib_setidx = i

		if data_type=='mock':
			# cib_setidx = i
			
			if pscb_dict['load_trilegal']:
				trilegal_fpath = fpath_dict['ciber_mock_fpath']+datestr_trilegal+'/trilegal/mock_trilegal_simidx'+str(cib_setidx)+'_'+datestr_trilegal+'.fits'
			
			test_set_fpath = fpath_dict['ciber_mock_fpath']+datestr+'/TM'+str(inst)+'/cib_with_tracer_5field_set'+str(cib_setidx)+'_'+datestr+'_TM'+str(inst)+'.fits'

			# assume same PS for ell^-3 sky clustering signal (indiv_ifield corresponds to one DGL field that is scaled by dgl_scale_fac)
			


			merge_dict = Merge(pscb_dict, float_param_dict) # merge dictionaries
			joint_masks, observed_ims, total_signals,\
					rnmaps, shot_sigma_sb_maps, noise_models,\
					ff_truth, diff_realizations, zl_perfield = cbps.generate_synthetic_mock_test_set(inst, ifield_list,\
														test_set_fpath=test_set_fpath, mock_trilegal_path=trilegal_fpath,\
														 noise_models=read_noise_models, ff_truth=smooth_ff, **merge_dict)
			
			
			
			# if i==0 and pscb_dict['compute_beam_correction']:
				
			# 	if fpath_dict['bls_fpath'] is None:
			# 		bl_est_path = fpath_dict['bls_base_path']+'ptsrc_blest_TM'+str(inst)+'_simidx'+str(cib_setidx)+'.npz'
			# 		print('Computing beam correction to be saved to bls_fpath='+bl_est_path+'..')
			# 	else:
			# 		bl_est_path = fpath_dict['bls_fpath']
			# 	# J_bright_Bl=pscb_dict['J_bright_Bl'], J_faint_Bl=float_param_dict['J_faint_Bl'],

			# 	lb, diff_norm_array,\
			# 			cls_masked_tot, cls_masked_tot_bright = cbps.estimate_b_ell_from_maps(inst, ifield_list, observed_ims, simidx=i, plot=False, save=False, \
			# 								niter = float_param_dict['niter'], ff_stack_min=float_param_dict['ff_stack_min'], data_type='mock', mask_base_path=fpath_dict['mask_base_path'], ell_norm=float_param_dict['ell_norm_blest'], \
			# 								full_mask_tail=config_dict['full_mask_tail'], bright_mask_tail=config_dict['bright_mask_tail'])
					
			# 	B_ells = np.sqrt(diff_norm_array).copy()
			# 	print('B_ells is ', B_ells)

			# 	np.savez(bl_est_path, \
			# 		lb=lb, J_bright_mag=float_param_dict['J_bright_Bl'], J_tot_mag=float_param_dict['J_faint_Bl'],\
			# 			 cls_masked_tot_bright=cls_masked_tot_bright, cls_masked_tot=cls_masked_tot, B_ells=B_ells)
			
		else:
			for fieldidx, ifield in enumerate(ifield_list):

				cbps.load_data_products(ifield, inst, verbose=True)
				observed_ims[fieldidx] = cbps.image*cbps.cal_facs[inst]
				
				if pscb_dict['plot_maps']:
					plot_map(observed_ims[fieldidx], title='observed ims ifield = '+str(ifield))
				
		if i==0 and pscb_dict['compute_beam_correction']:

			if fpath_dict['bls_fpath'] is None:
				if data_type=='mock':
					tailstr = 'simidx'+str(i)
				elif data_type=='observed':
					tailstr = 'observed'
				bl_est_path = fpath_dict['bls_base_path']+'ptsrc_blest_TM'+str(inst)+'_'+tailstr+'.npz'
				print('Computing beam correction to be saved to bls_fpath='+bl_est_path+'..')
			else:
				bl_est_path = fpath_dict['bls_fpath']
			# J_bright_Bl=pscb_dict['J_bright_Bl'], J_faint_Bl=float_param_dict['J_faint_Bl'],

			lb, diff_norm_array,\
					cls_masked_tot, cls_masked_tot_bright = cbps.estimate_b_ell_from_maps(inst, ifield_list, observed_ims, simidx=i, plot=True, save=False, \
										niter = float_param_dict['niter'], ff_stack_min=float_param_dict['ff_stack_min'], data_type=data_type, mask_base_path=fpath_dict['mask_base_path'], ell_norm=float_param_dict['ell_norm_blest'], \
										full_mask_tail=config_dict['full_mask_tail'], bright_mask_tail=config_dict['bright_mask_tail'])
				
			B_ells = np.sqrt(diff_norm_array).copy()
			print('B_ells is ', B_ells)

			np.savez(bl_est_path, \
				lb=lb, J_bright_mag=float_param_dict['J_bright_Bl'], J_tot_mag=float_param_dict['J_faint_Bl'],\
					 cls_masked_tot_bright=cls_masked_tot_bright, cls_masked_tot=cls_masked_tot, B_ells=B_ells)



		# ----------- load masks and mkk matrices ------------

		if pscb_dict['apply_mask']: # if applying mask load masks and inverse Mkk matrices

			for fieldidx, ifield in enumerate(ifield_list):
				
				if data_type=='observed':

					mask_fpath = fpath_dict['mask_base_path']+'joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'
					joint_maskos[fieldidx] = fits.open(mask_fpath)[1].data

					# mask_fpath = 'data/fluctuation_data/TM'+str(inst)+'/mask/field'+str(ifield)+'_TM'+str(inst)+'_fullmask_Jlim='+str(masking_maglim)+'_071922_Jbright14_maskInst.fits'
					# joint_maskos[fieldidx] = fits.open(mask_fpath)[0].data

					if float_param_dict['clip_sigma'] is not None:
						print('applying sigma clip to uncorrected flight image..')
						print('mask goes from ', float(np.sum(joint_maskos[fieldidx]))/float(cbps.dimx**2))
						sigclip = iter_sigma_clip_mask(observed_ims[fieldidx], sig=float_param_dict['clip_sigma'], nitermax=10, initial_mask=joint_maskos[fieldidx].astype(np.int))
						joint_maskos[fieldidx] *= sigclip

					inv_Mkk_fpath = fpath_dict['mkk_base_path']+'mkk_estimate_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'
					inv_Mkks.append(fits.open(inv_Mkk_fpath)['inv_Mkk_'+str(ifield)].data)

					if mask_tail_ffest is not None:
						print('Loading full mask from ', mask_tail_ffest)
						mask_fpath_ffest = fpath_dict['mask_base_path']+'joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail_ffest+'.fits'
						joint_maskos_ffest[fieldidx] = fits.open(mask_fpath_ffest)['joint_mask_'+str(ifield)].data
				
					# inv_Mkks.append(fits.open('data/fluctuation_data/TM'+str(inst)+'/mkk/mkk_estimate_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_JVlim='+str(masking_maglim)+'_Jbright14_maskInst.fits')['inv_Mkk_'+str(ifield)].data)
				else:
					if pscb_dict['load_ptsrc_cib']:
						print('mask path is ', fpath_dict['mask_base_path']+'joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(i)+'_'+mask_tail+'.fits')
						joint_maskos[fieldidx] = fits.open(fpath_dict['mask_base_path']+'joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(i)+'_'+mask_tail+'.fits')['joint_mask_'+str(ifield)].data
						inv_Mkks.append(fits.open(fpath_dict['mkk_base_path']+'mkk_estimate_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(i)+'_'+mask_tail+'.fits')['inv_Mkk_'+str(ifield)].data)
						
						if mask_tail_ffest is not None:
							print('Loading full mask from ', mask_tail_ffest)
							joint_maskos_ffest[fieldidx] = fits.open(fpath_dict['mask_base_path']+'joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(i)+'_'+mask_tail_ffest+'.fits')['joint_mask_'+str(ifield)].data
					
					else:
						maskidx = i%10
						joint_maskos[fieldidx] = fits.open(base_path+'TM'+str(inst)+'/masks/ff_joint_masks/joint_mask_with_ffmask_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(maskidx)+'_abc110821.fits')['joint_mask_'+str(ifield)].data
						inv_Mkks.append(fits.open(base_path+'TM'+str(inst)+'/mkk/ff_joint_masks/mkk_estimate_with_ffmask_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(maskidx)+'_abc110821.fits')['inv_Mkk_'+str(ifield)].data)

				orig_mask_fractions.append(float(np.sum(joint_maskos[fieldidx]))/float(cbps.dimx**2))

			orig_mask_fractions = np.array(orig_mask_fractions)
			print('mask fractions before FF estimation are ', orig_mask_fractions)


		# if config_dict['ps_type']=='cross' and config_dict['cross_gal'] is not None:
		# 	cross_maps = 
		# multiply signal by flat field and add read noise on top

		if pscb_dict['iterate_grad_ff']:
			weights_ff = None
			if pscb_dict['use_ff_weights']:
				weights_ff = weights_photonly

			print('RUNNING ITERATE GRAD FF for observed images')
			print('and weights_ff is ', weights_ff)

			if pscb_dict['save_intermediate_cls']:

				cls_inter.append(grab_ps_set(observed_ims, ifield_list, single_ps_set_shape, cbps, masks=joint_maskos))
				inter_labels.append('preffgrad_masked')


			processed_ims, ff_estimates, final_planes, stack_masks, all_coeffs = iterative_gradient_ff_solve(observed_ims, niter=float_param_dict['niter'], masks=joint_maskos, \
																					weights_ff=weights_ff, ff_stack_min=float_param_dict['ff_stack_min'], masks_ffest=joint_maskos_ffest)


			print('multiplying joint masks by stack masks..')
			joint_maskos *= stack_masks
			if joint_maskos_ffest is not None:
				joint_maskos_ffest *= stack_masks

			if pscb_dict['save_intermediate_cls']:
				# check if memory leak
				cls_inter.append(grab_ps_set(processed_ims, ifield_list, single_ps_set_shape, cbps, masks=joint_maskos))
				inter_labels.append('postffgrad_masked')

			for newidx, newifield in enumerate(ifield_list):
				print('loooping im on ', newidx, newifield)
				ff_smoothed = gaussian_filter(ff_estimates[newidx], sigma=5)
				ff_smoothsubtract = ff_estimates[newidx]-ff_smoothed
				# sigclip = sigma_clip_maskonly(ff_smoothsubtract, previous_mask=joint_maskos[newidx]*stack_masks[newidx], sig=float_param_dict['clip_sigma'])
				sigclip = sigma_clip_maskonly(ff_smoothsubtract, previous_mask=joint_maskos[newidx], sig=float_param_dict['clip_sigma'])
				joint_maskos[newidx] *= sigclip
				print('sum of sigclip is ', np.sum(~sigclip))

			if float_param_dict['ff_min'] is not None and float_param_dict['ff_max'] is not None:
				ff_masks = (ff_estimates > float_param_dict['ff_min'])*(ff_estimates < float_param_dict['ff_max'])
				joint_maskos *= ff_masks


			mask_fractions = np.array([float(np.sum(joint_masko))/float(cbps.dimx*cbps.dimy) for joint_masko in joint_maskos])
			print('masking fraction is nooww ', mask_fractions)

			observed_ims = processed_ims.copy()

			for k in range(len(ff_estimates)):
				ff_estimates[k][joint_maskos[k]==0] = 1.

				if k==0:
					plot_map(processed_ims[k]*joint_maskos[k], title='masked im')
					plot_map(ff_estimates[k], title='ff estimates k')


		if pscb_dict['apply_mask']:
			obs_levels = np.array([np.mean(obs[joint_maskos[o]==1]) for o, obs in enumerate(observed_ims)])
			print('masked OBS LEVELS HERE ARE ', obs_levels)
		else:
			obs_levels = np.array([np.mean(obs) for obs in observed_ims])
			print('unmasked OBS LEVELS HERE ARE ', obs_levels)

		if pscb_dict['ff_bias_correct']:
			if ff_biases is None:
				ff_biases = compute_ff_bias(obs_levels, weights=weights_photonly, mask_fractions=mask_fractions)
				print('FF biases are ', ff_biases)

		nls_estFF_nofluc = []

		# for the tests with bright masking thresholds, we can calculate the ff realizations with the bright mask and this should be fine for the noise bias
		
		# for the actual flat field estimates of the observations, apply the deeper masks when stacking the off fields. The ff error will be that of the deep masks, 
		# and so there may be some bias unaccounted for by the analytic multiplicative bias.

		# I think the same holds for the gradients, just use the deeper masks

		# make monte carlo FF realizations that are used for the noise realizations (how many do we ultimately need?)
		if i==0 and pscb_dict['save_ff_ests']:

			ff_realization_estimates = np.zeros(field_set_shape)
			observed_ims_nofluc = np.zeros(field_set_shape)

			for ffidx in range(float_param_dict['nmc_ff']):

				# ---------------------------- add ZL -------------------------

				for fieldidx, ifield in enumerate(ifield_list):

					zl_realiz = generate_zl_realization(zl_levels[fieldidx], False, dimx=cbps.dimx, dimy=cbps.dimy)
					if pscb_dict['with_photon_noise']:
						# print('adding photon noise to FF realization for noise model..')
						shot_sigma_sb_zl_perf = cbps.compute_shot_sigma_map(inst, zl_perfield[fieldidx], nfr=nfr_fields[fieldidx])
						zl_realiz += shot_sigma_sb_zl_perf*np.random.normal(0, 1, size=cbps.map_shape)
					observed_ims_nofluc[fieldidx] = zl_realiz

					if pscb_dict['apply_smooth_FF']:
						# print('applying smooth FF to FF realization for noise model')
						observed_ims_nofluc[fieldidx] *= smooth_ff

				if pscb_dict['with_inst_noise']: # true for observed
					# print('applying read noise to FF realization for noise model..')
					for fieldidx, ifield in enumerate(ifield_list):

						read_noise_indiv, _ = cbps.noise_model_realization(inst, cbps.map_shape, read_noise_models[fieldidx], \
												read_noise=True, photon_noise=False)
						observed_ims_nofluc[fieldidx] += read_noise_indiv

				if pscb_dict['iterate_grad_ff']:
					weights_ff_nofluc = None
					if pscb_dict['use_ff_weights']:
						print('weights for ff realization are ', weights_photonly)
						weights_ff_nofluc = weights_photonly

					print('RUNNING ITERATE GRAD FF on realizations')
					processed_ims, ff_realization_estimates, final_planes, _, coeffs_vs_niter = iterative_gradient_ff_solve(observed_ims_nofluc, niter=float_param_dict['niter'], masks=joint_maskos, weights_ff=weights_ff_nofluc, \
																															ff_stack_min=float_param_dict['ff_stack_min'], masks_ffest=joint_maskos_ffest) # masks already accounting for ff_stack_min previously
				else:
					print('only doing single iteration (do iterative pls)')

				np.savez(fpath_dict['ciber_mock_fpath']+'030122/ff_realizations/'+run_name+'/ff_realization_estimate_ffidx'+str(ffidx)+'.npz', \
						ff_realization_estimates = ff_realization_estimates)

		if pscb_dict['save_intermediate_cls']:
			cls_masked_prenl, cls_postffb_corr, cls_masked_postnl,\
				 cls_postmkk_prebl, cls_postbl, cls_postffb_corr, cls_post_isl_sub, cls_post_tcorr = [np.zeros(single_ps_set_shape) for x in range(8)]


		for fieldidx, obs in enumerate(observed_ims):
			print('fieldidx = ', fieldidx)

			if pscb_dict['apply_mask']:
				target_mask = joint_maskos[fieldidx]
				target_invMkk = inv_Mkks[fieldidx]
			else:
				stack_mask, target_mask = None, None
				target_invMkk = None

			if pscb_dict['plot_ff_error'] and data_type=='mock':
				plot_map(ff_estimate, title='ff estimate i='+str(i))
				if pscb_dict['apply_smooth_FF'] and pscb_dict['plot_map']:
					plot_map(ff_estimate_nofluc, title='ff estimate no fluc')
					if pscb_dict['apply_mask']:
						plot_map(target_mask*(ff_estimate-smooth_ff), title='flat field error (masked)')
					else:
						plot_map(ff_estimate-smooth_ff, title='flat field error (unmasked)')

			if pscb_dict['apply_mask']:
				simmap_dc = np.mean(obs[target_mask==1])
			else:
				simmap_dc = np.mean(obs)

			# noise bias from sims with no fluctuations    
			print('and nowwww here on i = ', i)

			# if noise model has already been saved just load it
			if i > 0 or pscb_dict['load_noise_bias']:

				# temporary comment out
				if data_type=='mock':
					noisemodl_fpath = fpath_dict['ciber_mock_fpath']+'030122/noise_models_sim/'+run_name+'/noise_model_fieldidx'+str(fieldidx)+'.npz'
				else:
					noisemodl_fpath = fpath_dict['ciber_mock_fpath']+'030122/noise_models_sim/'+run_name+'/noise_model_observed_fieldidx'+str(fieldidx)+'.npz'

				print('LOADING NOISE MODEL FROM ', noisemod_fpath)

				noisemodl_file = np.load(noisemodl_fpath)
				fourier_weights_nofluc = noisemodl_file['fourier_weights_nofluc']
				mean_cl2d_nofluc = noisemodl_file['mean_cl2d_nofluc']
				nl_estFF_nofluc = noisemodl_file['nl_estFF_nofluc']

			else:
				all_ff_ests_nofluc = []
				for ffidx in range(float_param_dict['nmc_ff']):

					ff_file = np.load(fpath_dict['ciber_mock_fpath']+'030122/ff_realizations/'+run_name+'/ff_realization_estimate_ffidx'+str(ffidx)+'.npz')
					all_ff_ests_nofluc.append(ff_file['ff_realization_estimates'][fieldidx])

					if ffidx==0:
						plot_map(all_ff_ests_nofluc[0], title='loaded MC ff estimate 0')

				print('median obs is ', np.median(obs))
				print('simmap dc is ', simmap_dc)

				shot_sigma_sb_zl_noisemodl = cbps.compute_shot_sigma_map(inst, image=obs, nfr=nfr_fields[fieldidx])

				fourier_weights_nofluc, mean_cl2d_nofluc = cbps.estimate_noise_power_spectrum(nsims=float_param_dict['n_FW_sims'], n_split=float_param_dict['n_FW_split'], apply_mask=pscb_dict['apply_mask'], \
				   noise_model=read_noise_models[fieldidx], read_noise=pscb_dict['with_inst_noise'], inst=inst, ifield=ifield_noise_list[fieldidx], show=False, mask=target_mask, \
					photon_noise=pscb_dict['with_photon_noise'], ff_estimate=None, shot_sigma_sb=shot_sigma_sb_zl_noisemodl, inplace=False, \
					field_nfr=nfr_fields[fieldidx], simmap_dc=simmap_dc, ff_truth=smooth_ff, mc_ff_estimates=all_ff_ests_nofluc, gradient_filter=pscb_dict['gradient_filter'])

				nl_estFF_nofluc = cbps.compute_noise_power_spectrum(inst, noise_Cl2D=mean_cl2d_nofluc.copy(), inplace=False, apply_FW=pscb_dict['apply_FW'], weights=fourier_weights_nofluc)

				print('nl_estFF_nofluc = ', nl_estFF_nofluc)

				if pscb_dict['save']:
					print('SAVING NOISE MODEL FILE..')
					if data_type=='mock':
						noisemodl_fpath = '/Volumes/Seagate Backup Plus Drive/Toolkit/Mirror/Richard/ciber_mocks/030122/noise_models_sim/'+run_name+'/noise_model_fieldidx'+str(fieldidx)+'.npz'
					else:
						noisemodl_fpath = '/Volumes/Seagate Backup Plus Drive/Toolkit/Mirror/Richard/ciber_mocks/030122/noise_models_sim/'+run_name+'/noise_model_observed_fieldidx'+str(fieldidx)+'.npz'

					np.savez(noisemodl_fpath, \
							fourier_weights_nofluc=fourier_weights_nofluc, mean_cl2d_nofluc=mean_cl2d_nofluc, nl_estFF_nofluc=nl_estFF_nofluc)


			cbps.FW_image=fourier_weights_nofluc.copy()
			nls_estFF_nofluc.append(nl_estFF_nofluc)

			if pscb_dict['load_ptsrc_cib'] or data_type=='observed':
				beam_correct = True
			else:
				beam_correct = False

			FF_correct = True # divides image by estimated flat field, not to be confused with flat field bias correction
			if pscb_dict['iterate_grad_ff']: # set gradient_filter, FF_correct to False here because we've already done gradient filtering/FF estimation
				gradient_filter = False
				FF_correct = False

			# if data_type=='observed' and clip_sigma is not None:
			# 	# print('mask goes from ', float(np.sum(target_mask))/float(cbps.dimx**2))
			# 	sigclip = sigma_clip_maskonly(obs, previous_mask=target_mask, sig=clip_sigma)
	  #       	target_mask *= sigclip
	  #       	# print('to ', float(np.sum(target_mask))/float(cbps.dimx**2))

			print('obs mean is ', np.mean(obs[target_mask==1]))
			print('nl estFF no fluc:', nl_estFF_nofluc)
			lb, processed_ps_nf, cl_proc_err, _ = cbps.compute_processed_power_spectrum(inst, apply_mask=pscb_dict['apply_mask'], \
											 mask=target_mask, image=obs, convert_adufr_sb=False, \
											mkk_correct=pscb_dict['apply_mask'], inv_Mkk=target_invMkk, beam_correct=beam_correct, B_ell=B_ells[fieldidx], \
											apply_FW=pscb_dict['apply_FW'], verbose=pscb_dict['verbose'], noise_debias=True, \
										 FF_correct=FF_correct, FF_image=ff_estimates[fieldidx],\
											 gradient_filter=gradient_filter, save_intermediate_cls=pscb_dict['save_intermediate_cls'], N_ell=nl_estFF_nofluc) # ff bias


			if pscb_dict['save_intermediate_cls']:
				cls_masked_prenl[fieldidx,:] = cbps.masked_Cl_pre_Nl_correct
				cls_masked_postnl[fieldidx,:] = cbps.masked_Cl_post_Nl_correct
				cls_postmkk_prebl[fieldidx,:] = cbps.cl_post_mkk_pre_Bl
				cls_postbl[fieldidx,:] = processed_ps_nf.copy()

			if pscb_dict['ff_bias_correct']:
				print('before ff bias correct ps is ', processed_ps_nf)
				print('correcting', fieldidx, ' with ff bias of ', ff_biases[fieldidx])
				processed_ps_nf /= ff_biases[fieldidx]

				if pscb_dict['save_intermediate_cls']:
					cls_postffb_corr[fieldidx,:] = processed_ps_nf.copy()

			if pscb_dict['iterate_grad_ff']: # .. but we want gradient filtering still for the ground truth signal.
				gradient_filter = True

			cl_prenlcorr = cbps.masked_Cl_pre_Nl_correct.copy()

			# if using CIB model, ground truth for CIB is loaded from Helgason J > Jlim and added to ground truth of DGL-like component
			if data_type=='mock':
				if pscb_dict['load_ptsrc_cib']: # false for observed

					lb, diff_cl, cl_proc_err, _ = cbps.compute_processed_power_spectrum(inst, apply_mask=False, \
														 image=diff_realizations[fieldidx], convert_adufr_sb=False, \
														mkk_correct=False, beam_correct=True, B_ell=B_ells[fieldidx], \
														apply_FW=False, verbose=False, noise_debias=False, \
													 FF_correct=False, save_intermediate_cls=pscb_dict['save_intermediate_cls'], gradient_filter=False) # gradient is False for now 082222


					cib_cl_file = fits.open(base_path+'TM'+str(inst)+'/powerspec/cls_cib_vs_maglim_ifield'+str(ifield_list[fieldidx])+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_Vega_magcut_test.fits')
					unmasked_cib_cl = cib_cl_file['cls_cib'].data['cl_maglim_'+str(masking_maglim)]
					ebl_ps = unmasked_cib_cl + diff_cl
					# print('diff_cl:', diff_cl)
					print('unmasked cib cl', unmasked_cib_cl)

					if pscb_dict['load_trilegal'] and masking_maglim < 16:

						print('Adding power spectrum from all stars fainter than masking limit..')
						if pscb_dict['subtract_subthresh_stars']:
							print('Setting subtract_subthresh_stars to False..')
							pscb_dict['subtract_subthresh_stars'] = False
						trilegal_cl_fpath = trilegal_base_path+'TM'+str(inst)+'/powerspec/cls_trilegal_vs_maglim_ifield'+str(ifield_list[fieldidx])+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_Vega_magcut_test.fits'
						trilegal_cl = fits.open(trilegal_cl_fpath)['cls_trilegal'].data['cl_maglim_'+str(masking_maglim)]
						ebl_ps += trilegal_cl 
						print('trilegal_cl is ', trilegal_cl)
						print('and ebl is now ', ebl_ps)

				else:

					lb, ebl_ps, cl_proc_err, _ = cbps.compute_processed_power_spectrum(inst, apply_mask=False, \
														 image=clus_realizations[fieldidx], convert_adufr_sb=False, \
														mkk_correct=False, beam_correct=True, B_ell=B_ells[fieldidx], \
														apply_FW=False, verbose=False, noise_debias=False, \
													 FF_correct=False, save_intermediate_cls=pscb_dict['save_intermediate_cls'], gradient_filter=pscb_dict['gradient_filter'])

			if pscb_dict['transfer_function_correct']:
				processed_ps_nf /= t_ell_av

				if pscb_dict['save_intermediate_cls']:
					cls_post_tcorr[fieldidx,:] = processed_ps_nf.copy()


			if pscb_dict['subtract_subthresh_stars']:
				# maybe load precomputed power spectra and assume we have it nailed down
				print('before trilegal star correct correct ps is ', processed_ps_nf)
				print('subtracting sub-threshold stars!!!!')
				trilegal_cl_fpath = trilegal_base_path+'TM'+str(inst)+'/powerspec/cls_trilegal_vs_maglim_ifield'+str(ifield_list[fieldidx])+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_Vega_magcut_test.fits'
				
				trilegal_cl = fits.open(trilegal_cl_fpath)['cls_trilegal'].data['cl_maglim_'+str(masking_maglim)]
				print(trilegal_cl)
				processed_ps_nf -= trilegal_cl

				if pscb_dict['save_intermediate_cls']:
					cls_post_isl_sub[fieldidx,:] = processed_ps_nf.copy()

			print('proc:', processed_ps_nf)

			recovered_power_spectra[i, fieldidx, :] = processed_ps_nf

			if data_type=='mock':
				print('ebl ps', ebl_ps)
				signal_power_spectra[i, fieldidx, :] = ebl_ps
				print('proc (nofluc) / ebl = ', processed_ps_nf/ebl_ps)
				print('mean ps bias is ', np.mean(recovered_power_spectra[i,:,:]/signal_power_spectra[i,:,:], axis=0))

		nls_estFF_nofluc = np.array(nls_estFF_nofluc)
		print('mean nl:', np.mean(nls_estFF_nofluc, axis=0))

		if not pscb_dict['save_intermediate_cls']:
			cls_inter, inter_labels = None, None 
		else:
			cls_inter.extend([cls_masked_prenl, cls_masked_postnl, cls_postmkk_prebl, cls_postbl, cls_postffb_corr, cls_post_tcorr, cls_post_isl_sub])
			inter_labels.extend(['masked_Cl_pre_Nl_correct', 'masked_Cl_post_Nl_correct', 'post_mkk_pre_Bl', 'post_Bl_pre_ffbcorr', 'post_ffb_corr','post_tcorr', 'post_isl_sub'])
			print('cls inter has shape', np.array(cls_inter).shape)
			print('inter labels has length', len(inter_labels))
			
		if pscb_dict['save']:
			
			
			np.savez(fpath_dict['sim_test_fpath']+run_name+'/input_recovered_ps_estFF_simidx'+str(i)+'.npz', lb=lb, data_type=data_type, \
				signal_ps=signal_power_spectra[i], recovered_ps_est_nofluc=recovered_power_spectra[i],\
					 dgl_scale_fac=float_param_dict['dgl_scale_fac'], niter=float_param_dict['niter'], cls_inter=cls_inter, inter_labels=inter_labels, **pscb_dict)

	return lb, signal_power_spectra, recovered_power_spectra, nls_estFF_nofluc, cls_inter, inter_labels






