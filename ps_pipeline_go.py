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
# from ciber_data_helpers import load_psf_params_dict
from plotting_fns import plot_map
from ciber_mocks import *
from flat_field_est import *
from mkk_parallel import compute_inverse_mkk, plot_mkk_matrix
from masking_utils import *
import config


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


def set_up_filepaths_cbps(fpath_dict, inst, run_name, datestr, datestr_trilegal=None, data_type='mock', save_fpaths=True):

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

	base_fluc_path = config.exthdpath+'ciber_fluctuation_data/'
	base_fluc_names = ['read_noise_modl_base_path', 'bls_base_path', 'subpixel_psfs_path', 'iris_base_path']
	base_fluc_dirnames = ['noise_model', 'beam_correction', 'subpixel_psfs', 'iris_regrid']


	for b, base_name in enumerate(mock_info_path_names):
		if fpath_dict[base_name] is None:
			fpath_dict[base_name] = mock_base_path+'TM'+str(inst)+'/'+mock_info_path_dirnames[b]
			print('setting '+base_name+' to '+fpath_dict[base_name])
		list_of_dirpaths.append(fpath_dict[base_name])

	for b, base_name in enumerate(base_path_names):
		if fpath_dict[base_name] is None:
			fpath_dict[base_name] = base_path+'TM'+str(inst)+'/'+base_path_dirnames[b]
			print('setting '+base_name+' to '+fpath_dict[base_name])
		list_of_dirpaths.append(fpath_dict[base_name])

	for b, base_fluc in enumerate(base_fluc_names):
		if fpath_dict[base_fluc] is None:
			fpath_dict[base_fluc] = base_fluc_path+'TM'+str(inst)+'/'+base_fluc_dirnames[b]
			print('setting '+base_fluc+' to '+fpath_dict[base_fluc])
		list_of_dirpaths.append(fpath_dict[base_fluc])

	if fpath_dict['tls_base_path'] is None:
		fpath_dict['tls_base_path'] = base_path+'transfer_function/'
		print('tls base path is ', fpath_dict['tls_base_path'])
		list_of_dirpaths.append(fpath_dict['tls_base_path'])

	if fpath_dict['ff_est_dirpath'] is None:
		fpath_dict['ff_est_dirpath'] = fpath_dict['ciber_mock_fpath']+'030122/ff_realizations/'
		if fpath_dict['ff_run_name'] is None:
			fpath_dict['ff_run_name'] = run_name
		fpath_dict['ff_est_dirpath'] += fpath_dict['ff_run_name']
		print('ff_est_dirpath is ', fpath_dict['ff_est_dirpath'])

	print(list_of_dirpaths)
	if save_fpaths:
		make_fpaths(list_of_dirpaths)


	return fpath_dict, list_of_dirpaths, base_path, trilegal_base_path


def run_cbps_pipeline(cbps, inst, nsims, run_name, ifield_list = None, \
							datestr='100921', datestr_trilegal=None, data_type='mock',\
							masking_maglim=17.5, mask_tail='abc110821', mask_tail_ffest=None, mask_tail_cross=None, \
							zl_levels=None, ff_biases=None, **kwargs):

	config_dict = dict({'ps_type':'auto', 'cross_type':'ciber', 'cross_inst':2, 'cross_gal':None, 'simidx0':0, \
						'full_mask_tail':'maglim_17_Vega_test', 'bright_mask_tail':'maglim_11_Vega_test'})

	pscb_dict = dict({'ff_estimate_correct':True, 'apply_mask':True, 'with_inst_noise':True, 'with_photon_noise':True, 'apply_FW':True, 'generate_diffuse_realization':True, \
					'apply_smooth_FF':True, 'compute_beam_correction':True, 'same_zl_levels':False, 'apply_zl_gradient':True, 'gradient_filter':True, \
					'iterate_grad_ff':True ,'mkk_ffest_hybrid':False, 'apply_sum_fluc_image_mask':False, 'same_int':False, 'same_dgl':True, 'use_ff_weights':True, 'plot_ff_error':False, 'load_ptsrc_cib':True, \
					'load_trilegal':True , 'subtract_subthresh_stars':True, 'ff_bias_correct':True, 'save_ff_ests':True, 'plot_maps':True, \
					 'draw_cib_setidxs':False, 'aug_rotate':False, 'noise_debias':True, 'load_noise_bias':False, 'transfer_function_correct':False, 'compute_transfer_function':False,\
					  'save_intermediate_cls':True, 'verbose':True, 'show_plots':False, 'save':True, 'bl_post':False, 'ff_sigma_clip':False})

	float_param_dict = dict({'ff_min':0.5, 'ff_max':2.0, 'clip_sigma':5, 'ff_stack_min':2, 'nmc_ff':10, \
					  'theta_mag':0.01, 'niter':5, 'dgl_scale_fac':5, 'smooth_sigma':5, 'indiv_ifield':6,\
					  'nfr_same':25, 'J_bright_Bl':11, 'J_faint_Bl':17.5, 'n_FW_sims':500, 'n_FW_split':10, \
					  'ell_norm_blest':5000, 'n_realiz_t_ell':100})

	fpath_dict = dict({'ciber_mock_fpath':config.exthdpath+'ciber_mocks/', \
						'sim_test_fpath':'data/input_recovered_ps/sim_tests_030122/', \
						'base_fluc_path':config.exthdpath+'ciber_fluctuation_data/', \
						'ff_smooth_fpath':'data/flatfield/TM1_FF/TM1_field4_mag=17.5_FF.fits', \
						'observed_base_path':'data/fluctuation_data/', 'ff_est_dirpath':None, \
						'cib_resid_ps_path':None, 'isl_resid_ps_path':None, \
						'mask_base_path':None, 'mkk_base_path':None, 'mkk_ffest_base_path':None, 'bls_base_path':None, 'isl_rms_fpath':None, 'bls_fpath':None, \
						't_ell_fpath':None, 'cib_realiz_path':None, 'tls_base_path':None, 'ff_run_name':None, \
						'noisemodl_basepath':None, 'subpixel_psfs_path':None, 'read_noise_modl_base_path':None, 'noisemodl_run_name':None, 'add_savestr':None, \
						'iris_base_path':None, 'bls_fpath_cross':None})
	

	if not pscb_dict['load_trilegal']:
		print('load trilegal = False, setting trilegal_fpath to None..')
		trilegal_fpath=None
	elif datestr_trilegal is None:
		datestr_trilegal = datestr
		
	config_dict, pscb_dict, float_param_dict, fpath_dict = update_dicts([config_dict, pscb_dict, float_param_dict, fpath_dict], kwargs)
	fpath_dict, list_of_dirpaths, base_path, trilegal_base_path = set_up_filepaths_cbps(fpath_dict, inst, run_name, datestr, datestr_trilegal=datestr_trilegal, data_type=data_type)

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


	field_set_shape = (nfield, cbps.dimx, cbps.dimy)
	ps_set_shape = (nsims, nfield, cbps.n_ps_bin)
	single_ps_set_shape = (nfield, cbps.n_ps_bin)

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
		# pscb_dict['save_ff_ests'] = False
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
		pscb_dict['ff_bias_correct'] = False
		mode_couple_base_dir = fpath_dict['mkk_ffest_base_path']
		# print('Using hybrid mask-FF mode coupling matrices, setting ff_bias_correct, transfer_function_correct to False')
		print('Using hybrid mask-FF mode coupling matrices, setting ff_bias_correct')
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

	ciber_cross_ciber, ciber_cross_gal, ciber_cross_iris = False, False, False
	if config_dict['ps_type']=='cross':
		if config_dict['cross_type']=='ciber':
			ciber_cross_ciber = True
			pscb_dict['save_ff_ests'] = False
		elif config_dict['cross_type']=='galaxy':
			ciber_cross_gal = True
			pscb_dict['iterate_grad_ff'] = False
		elif config_dict['cross_type']=='iris':
			print('cross type is IRIS we are here, setting ciber_cross_iris to True..')
			ciber_cross_iris = True
			# pscb_dict['iterate_grad_ff'] = False


	if pscb_dict['same_zl_levels']:
		zl_fieldnames = [cbps.ciber_field_dict[float_param_dict['indiv_ifield']] for i in range(nfield)]
	else:
		zl_fieldnames = [cbps.ciber_field_dict[ifield] for ifield in ifield_list]
	if zl_levels is None:
		zl_levels = [cbps.zl_levels_ciber_fields[inst][zl_fieldname] for zl_fieldname in zl_fieldnames]
	if ciber_cross_ciber:
		zl_levels_cross = [cbps.zl_levels_ciber_fields[config_dict['cross_inst']][zl_fieldname] for zl_fieldname in zl_fieldnames]
		print('zl_levels_cross is ', zl_levels_cross)

	if pscb_dict['verbose']:
		print('NFR fields is ', nfr_fields)
		print('ZL levels are ', zl_levels)
		if config_dict['ps_type']=='cross':
			print('ZL cross levels are ', zl_levels_cross)

	if pscb_dict['use_ff_weights']: # cross
		weights_photonly = cbps.compute_ff_weights(inst, zl_levels, read_noise_models=None, ifield_list=ifield_list_ff, additional_rms=mean_isl_rms)
		print('FF weights are :', weights_photonly)
		if ciber_cross_ciber:
			weights_photonly_cross = cbps.compute_ff_weights(config_dict['cross_inst'], zl_levels_cross, ifield_list=ifield_list_ff)

	#  ------------------- instantiate data arrays  --------------------
	clus_realizations, zl_perfield, ff_estimates, observed_ims = [np.zeros(field_set_shape) for i in range(4)]
	if config_dict['ps_type']=='cross' and config_dict['cross_type']=='ciber': # cross ciber
		observed_ims_cross, ff_estimates_cross = [np.zeros(field_set_shape) for x in range(2)]

	inv_Mkks, joint_maskos, joint_maskos_ffest = None, None, None

	# if cross power spectrum then load union mask, but might use separate masks for flat field estimation?
	if pscb_dict['apply_mask']:
		joint_maskos = np.zeros(field_set_shape) 

	signal_power_spectra, recovered_power_spectra, recovered_dcl = [np.zeros(ps_set_shape) for i in range(3)]
	# ------------------------------------------------------------------

	smooth_ff = None
	if pscb_dict['apply_smooth_FF'] and pscb_dict['data_type'] != 'observed':
		print('loading smooth FF from ', fpath_dict['ff_smooth_fpath'])
		ff = fits.open(fpath_dict['ff_smooth_fpath'])
		smooth_ff = gaussian_filter(ff[0].data, sigma=float_param_dict['smooth_sigma'])

	read_noise_models = None
	if pscb_dict['with_inst_noise']:
		read_noise_models = cbps.grab_noise_model_set(ifield_noise_list, inst, field_set_shape=field_set_shape, noise_model_base_path=fpath_dict['read_noise_modl_base_path'])
		if ciber_cross_ciber: # cross
			print('Loading noise power spectra for cross instrument TM', config_dict['cross_inst'])
			read_noise_models_cross = cbps.grab_noise_model_set(ifield_noise_list, config_dict['cross_inst'], field_set_shape=field_set_shape, noise_model_base_path=fpath_dict['cross_read_noise_modl_base_path'])

	if pscb_dict['transfer_function_correct']:
		fpath_dict['t_ell_fpath'] = fpath_dict['tls_base_path']+'t_ell_est_nsims='+str(float_param_dict['n_realiz_t_ell'])+'.npz'
		if pscb_dict['compute_transfer_function']:
			print('Computing transfer function from '+str(float_param_dict['n_realiz_t_ell']))
			lb, t_ell_av, t_ell_stderr, t_ells, cls_orig, cls_filt = cbps.calculate_transfer_function(nsims=float_param_dict['n_realiz_t_ell'], plot=False)
			if fpath_dict['t_ell_fpath'] is not None:
				print('Saving transfer function correction to ', fpath_dict['t_ell_fpath'])
				np.savez(fpath_dict['t_ell_fpath'], lb=lb, t_ell_av=t_ell_av, t_ell_stderr=t_ell_stderr, t_ells=t_ells, cls_orig=cls_orig, cls_filt=cls_filt)
		elif fpath_dict['t_ell_fpath'] is not None:
			print('Loading transfer function from ', fpath_dict['t_ell_fpath'])
			t_ell_av = np.load(fpath_dict['t_ell_fpath'])['t_ell_av']
			# print('t_ell_av is ', t_ell_av)

		else:
			print('No transfer function path provided, and compute_transfer_function set to False, exiting..')
			return 

	B_ells = [None for x in range(nfield)]

	if not pscb_dict['compute_beam_correction']:

		if fpath_dict['bls_fpath'] is not None: # cross
			print('loading B_ells from ', fpath_dict['bls_fpath'])
			B_ells = np.load(fpath_dict['bls_fpath'])['B_ells_post']

			# try:
			# 	B_ells = np.load(fpath_dict['bls_fpath'])['B_ells']
			# except:
			# 	B_ells = np.load(fpath_dict['bls_fpath'])['B_ells_post']

			print('B_ells = ', B_ells)
		
		if config_dict['ps_type']=='cross' and fpath_dict['bls_fpath_cross'] is not None:
			print('loading B_ells for cross from ', fpath_dict['bls_fpath_cross'])
			B_ells_cross = np.load(fpath_dict['bls_fpath_cross'])['B_ells']


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

		cib_setidx = i

		if data_type=='mock':
			
			if pscb_dict['load_trilegal']:
				trilegal_fpath = fpath_dict['ciber_mock_fpath']+datestr_trilegal+'/trilegal/mock_trilegal_simidx'+str(cib_setidx)+'_'+datestr_trilegal+'.fits'
			
			# test_set_fpath = fpath_dict['ciber_mock_fpath']+datestr+'/TM'+str(inst)+'/cib_realiz/cib_with_tracer_5field_set'+str(cib_setidx)+'_'+datestr+'_TM'+str(inst)+'.fits'
			test_set_fpath = fpath_dict['ciber_mock_fpath']+datestr+'/TM'+str(inst)+'/cib_realiz/cib_with_tracer_with_dpoint_5field_set'+str(cib_setidx)+'_'+datestr+'_TM'+str(inst)+'.fits'
			# assume same PS for ell^-3 sky clustering signal (indiv_ifield corresponds to one DGL field that is scaled by dgl_scale_fac)
			
			merge_dict = Merge(pscb_dict, float_param_dict) # merge dictionaries
			joint_masks, observed_ims, total_signals,\
					rnmaps, shot_sigma_sb_maps, noise_models,\
					ff_truth, diff_realizations, zl_perfield, mock_cib_ims = cbps.generate_synthetic_mock_test_set(inst, ifield_list,\
														test_set_fpath=test_set_fpath, mock_trilegal_path=trilegal_fpath,\
														 noise_models=read_noise_models, ff_truth=smooth_ff, **merge_dict)

			if config_dict['ps_type']=='cross' and pscb_dict['load_ptsrc_cib'] is False:
				print('mode = cross and no point sources, so making additional set of mocks in different band assuming fixed colors..')

				if config_dict['cross_type']=='ciber':
					color_ratio = cbps.iras_color_facs[config_dict['cross_inst']] / cbps.iras_color_fac[inst]
					print('color ratio is ', color_ratio)

					diff_realizations_cross = diff_realizations*color_ratio 
					zl_ratio = zl_levels_cross / zl_levels
					zl_perfield_cross = np.array([zl_indiv*zl_ratio for zl_indiv in zl_perfield])

				joint_masks_cross, observed_ims_cross, total_signals_cross,\
				rnmaps_cross, shot_sigma_sb_maps_cross, noise_models_cross,\
				ff_truth_cross, _, _, mock_cib_ims = cbps.generate_synthetic_mock_test_set(config_dict['cross_inst'], ifield_list,\
													test_set_fpath=test_set_fpath, mock_trilegal_path=trilegal_fpath,\
													 noise_models=read_noise_models_cross, diff_realizations=diff_realizations_cross, \
													 zl_perfield=zl_perfield_cross, ff_truth=smooth_ff, **merge_dict)
		
			
		else:
			# cross
			for fieldidx, ifield in enumerate(ifield_list):
				cbps.load_flight_image(ifield, inst, verbose=True)
				# cbps.load_data_products(ifield, inst, verbose=True)
				observed_ims[fieldidx] = cbps.image*cbps.cal_facs[inst]

				if ciber_cross_ciber:
					print('Loading data products for cross CIBER TM'+str(config_dict['cross_inst']))
					obs_cross = cbps.load_flight_image(ifield, config_dict['cross_inst'], verbose=True, inplace=False)
					observed_ims_cross[fieldidx] = obs_cross*cbps.cal_facs[config_dict['cross_inst']]
				elif ciber_cross_gal:
					observed_ims_cross[fieldidx] = cbps.load_gal_density(ifield, verbose=True, inplace=False)
				elif ciber_cross_iris:
					print('loading regridded IRIS image..')
					regrid_fpath_iris = config.exthdpath+'ciber_fluctuation_data/TM'+str(inst)+'/iris_regrid/iris_regrid_ifield'+str(ifield)+'_TM'+str(inst)+'_fromhp_nside=2048.fits'
					iris_map = cbps.iras_color_facs[inst]*fits.open(regrid_fpath_iris)[1].data
					iris_map[np.abs(iris_map) > 1000] = 0.
					observed_ims_cross[fieldidx] = iris_map

					# observed_ims_cross[fieldidx] = cbps.iras_color_facs[inst]*fits.open(regrid_fpath_iris)[1].data
					# obserevd_ims_cross[fieldidx]
					# if fieldidx < 4:
					# 	fieldidx_load = fieldidx+1
					# else:
					# 	fieldidx_load = 0
					# observed_ims_cross[fieldidx_load] = cbps.iras_color_facs[inst]*fits.open(regrid_fpath_iris)['TM2_regrid'].data

					# observed_ims_cross[fieldidx] = cbps.iras_color_facs[inst]*fits.open(regrid_fpath_iris)['TM2_regrid'].data
					# observed_ims_cross[fieldidx] = np.rot90(observed_ims_cross[fieldidx])

					# observed_ims_cross[fieldidx] = observed_ims_cross[fieldidx].transpose()
					# observed_ims_cross[fieldidx] = np.rot90(fits.open(regrid_fpath_iris)['TM2_regrid'].data)
					# observed_ims_cross[fieldidx] = cbps.load_regrid_image(ifield, inst, config_dict['cross_type'], regrid_fpath=regrid_fpath_iris,\
					# 													 verbose=True, inplace=False)

				if pscb_dict['plot_maps']:
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

			print('B_ells is nowww ', B_ells)

			np.savez(bl_est_path, \
				lb=lb, J_bright_mag=float_param_dict['J_bright_Bl'], J_tot_mag=float_param_dict['J_faint_Bl'],\
					 cls_masked_tot_bright=cls_masked_tot_bright, cls_masked_tot=cls_masked_tot, B_ells=B_ells)


		# ----------- load masks and mkk matrices ------------

		if pscb_dict['apply_mask']: # if applying mask load masks and inverse Mkk matrices

			for fieldidx, ifield in enumerate(ifield_list):
				
				if data_type=='observed': # cross
					# temporary
					# mask_fpath = fpath_dict['mask_base_path']+'/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'_harsherbright_120222.fits'

					mask_fpath = fpath_dict['mask_base_path']+'/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'
					joint_maskos[fieldidx] = fits.open(mask_fpath)[1].data

					# mask_fpath = 'data/fluctuation_data/TM'+str(inst)+'/mask/field'+str(ifield)+'_TM'+str(inst)+'_fullmask_Jlim='+str(masking_maglim)+'_071922_Jbright14_maskInst.fits'
					# joint_maskos[fieldidx] = fits.open(mask_fpath)[0].data

					if float_param_dict['clip_sigma'] is not None:
						print('applying sigma clip to uncorrected flight image..')
						print('mask goes from ', float(np.sum(joint_maskos[fieldidx]))/float(cbps.dimx**2))
						sigclip = iter_sigma_clip_mask(observed_ims[fieldidx], sig=float_param_dict['clip_sigma'], nitermax=10, mask=joint_maskos[fieldidx].astype(np.int))
						joint_maskos[fieldidx] *= sigclip

						if config_dict['ps_type']=='cross':
							sigclip_cross = iter_sigma_clip_mask(observed_ims_cross[fieldidx], sig=float_param_dict['clip_sigma'], nitermax=10, mask=joint_maskos[fieldidx].astype(np.int))
							joint_maskos[fieldidx] *= sigclip_cross

					if pscb_dict['mkk_ffest_hybrid']:
						# mkk_type = 'ffest_grad'
						mkk_type = 'ffest_nograd'
					else:
						mkk_type = 'maskonly_estimate'

					# inv_Mkk_fpath = mode_couple_base_dir+'/'+mask_tail+'/mkk_'+mkk_type+'_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'_newcross.fits' # what is newcross??
					inv_Mkk_fpath = mode_couple_base_dir+'/'+mask_tail+'/mkk_'+mkk_type+'_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'
					# inv_Mkk_fpath = fpath_dict['mkk_base_path']+'mkk_estimate_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'
					inv_Mkks.append(fits.open(inv_Mkk_fpath)['inv_Mkk_'+str(ifield)].data)

					# if mask_tail_ffest is not None:
						# print('Loading full mask from ', mask_tail_ffest) 
						# mask_fpath_ffest = fpath_dict['mask_base_path']+'/'+mask_tail+'joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail_ffest+'.fits'
						# joint_maskos_ffest[fieldidx] = fits.open(mask_fpath_ffest)['joint_mask_'+str(ifield)].data
				
				else:
					if pscb_dict['verbose']:
						print('mask path is ', fpath_dict['mask_base_path']+'/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(i)+'_'+mask_tail+'.fits')
					joint_maskos[fieldidx] = fits.open(fpath_dict['mask_base_path']+'/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(i)+'_'+mask_tail+'.fits')['joint_mask_'+str(ifield)].data
					
					if pscb_dict['mkk_ffest_hybrid']:
						mkk_type = 'ffest_nograd'
						# mkk_type = 'ffest_grad'
					else:
						mkk_type = 'maskonly_estimate'
					inv_Mkks.append(fits.open(mode_couple_base_dir+'/'+mask_tail+'/mkk_'+mkk_type+'_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(i)+'_'+mask_tail+'.fits')['inv_Mkk_'+str(ifield)].data)

					# if pscb_dict['mkk_ffest_hybrid']:
					# 	inv_Mkks.append(fits.open(mode_couple_base_dir+'/'+mask_tail+'/mkk_ffest_grad_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(i)+'_'+mask_tail+'.fits')['inv_Mkk_'+str(ifield)].data)
					# else:
					# 	inv_Mkks.append(fits.open(mode_couple_base_dir+'/'+mask_tail+'/mkk_maskonly_estimate_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(i)+'_'+mask_tail+'.fits')['inv_Mkk_'+str(ifield)].data)
					# # inv_Mkks.append(fits.open(fpath_dict['mkk_base_path']+'mkk_estimate_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(i)+'_'+mask_tail+'.fits')['inv_Mkk_'+str(ifield)].data)

					# if mask_tail_ffest is not None:
						# if pscb_dict['verbose']:
							# print('Loading full mask from ', mask_tail_ffest)
						# joint_maskos_ffest[fieldidx] = fits.open(fpath_dict['mask_base_path']+'/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(i)+'_'+mask_tail_ffest+'.fits')['joint_mask_'+str(ifield)].data
						# orig_mask_fractions.append(float(np.sum(joint_maskos[fieldidx]))/float(cbps.dimx**2))

			orig_mask_fractions = np.array(orig_mask_fractions)

			if pscb_dict['iterate_grad_ff']:
				print('mask fractions before FF estimation are ', orig_mask_fractions)


		# if config_dict['ps_type']=='cross' and config_dict['cross_gal'] is not None:
		# 	cross_maps = 
		# multiply signal by flat field and add read noise on top

		if pscb_dict['iterate_grad_ff']: # cross
			weights_ff, weights_ff_cross = None, None
			if pscb_dict['use_ff_weights']:
				weights_ff = weights_photonly
				if ciber_cross_ciber:
					weights_ff_cross = weights_photonly_cross
					print('weights_ff_cross is ', weights_ff_cross)

			print('RUNNING ITERATE GRAD FF for observed images')
			print('and weights_ff is ', weights_ff)

			if pscb_dict['save_intermediate_cls']:
				cls_inter.append(grab_ps_set(observed_ims, ifield_list, single_ps_set_shape, cbps, masks=joint_maskos))
				inter_labels.append('preffgrad_masked')

			# processed_ims, ff_estimates, final_planes, stack_masks, all_coeffs = iterative_gradient_ff_solve(observed_ims, niter=float_param_dict['niter'], masks=joint_maskos, \
			# 																		weights_ff=weights_ff, ff_stack_min=float_param_dict['ff_stack_min'], masks_ffest=joint_maskos_ffest)
			processed_ims, ff_estimates, final_planes, stack_masks, all_coeffs = iterative_gradient_ff_solve(observed_ims, niter=float_param_dict['niter'], masks=joint_maskos, \
																					weights_ff=weights_ff, ff_stack_min=float_param_dict['ff_stack_min'])

			print('multiplying joint masks by stack masks..')
			if pscb_dict['apply_mask']:
				joint_maskos *= stack_masks


			if pscb_dict['apply_sum_fluc_image_mask']:
				print('applying mask on sum of fluctuation images')

			if ciber_cross_ciber:
				processed_ims_cross, ff_estimates_cross, final_planes, stack_masks, all_coeffs = iterative_gradient_ff_solve(observed_ims_cross, niter=float_param_dict['niter'], masks=joint_maskos, \
																										weights_ff=weights_ff_cross, ff_stack_min=float_param_dict['ff_stack_min'])


			if pscb_dict['save_intermediate_cls']:
				# check if memory leak
				cls_inter.append(grab_ps_set(processed_ims, ifield_list, single_ps_set_shape, cbps, masks=joint_maskos))
				inter_labels.append('postffgrad_masked')

			if pscb_dict['ff_sigma_clip'] and pscb_dict['apply_mask']:
				print('Unsharp masking on flat field estimate')
				for newidx, newifield in enumerate(ifield_list):
					print('loooping im on ', newidx, newifield)
					ff_smoothed = gaussian_filter(ff_estimates[newidx], sigma=5)
					ff_smoothsubtract = ff_estimates[newidx]-ff_smoothed
					sigclip = sigma_clip_maskonly(ff_smoothsubtract, previous_mask=joint_maskos[newidx], sig=float_param_dict['clip_sigma'])
					joint_maskos[newidx] *= sigclip
					print('sum of sigclip is ', np.sum(~sigclip))

			elif float_param_dict['ff_min'] is not None and float_param_dict['ff_max'] is not None:
				if pscb_dict['apply_mask']:
					print('Clipping on ff_min=', float_param_dict['ff_min'], 'ff_max=', float_param_dict['ff_max'])
					ff_masks = (ff_estimates > float_param_dict['ff_min'])*(ff_estimates < float_param_dict['ff_max'])
					joint_maskos *= ff_masks

			mask_fractions = np.ones((len(ifield_list)))
			if pscb_dict['apply_mask']:
				mask_fractions = np.array([float(np.sum(joint_masko))/float(cbps.dimx*cbps.dimy) for joint_masko in joint_maskos])
				print('masking fraction is nooww ', mask_fractions)

			observed_ims = processed_ims.copy()
			if ciber_cross_ciber:
				observed_ims_cross = processed_ims_cross.copy()

			for k in range(len(ff_estimates)):

				if pscb_dict['apply_mask']:
					ff_estimates[k][joint_maskos[k]==0] = 1.

					if ciber_cross_ciber:
						ff_estimates_cross[k][joint_maskos[k]==0] = 1.
					if k==0 and pscb_dict['show_plots']:
						plot_map(processed_ims[k]*joint_maskos[k], title='masked im')

				if k==0 and pscb_dict['show_plots']:
					plot_map(ff_estimates[k], title='ff estimates k')

		
		if pscb_dict['apply_mask']: # cross
			# obs_levels = np.array([np.mean(obs[joint_maskos[o]==1]) for o, obs in enumerate(observed_ims)])
			obs_levels = np.array([np.mean(obs[joint_maskos[o]==1]) for o, obs in enumerate(observed_ims)])

			if config_dict['ps_type']=='cross':
				obs_levels_cross = np.array([np.mean(obs[joint_maskos[o]==1]) for o, obs in enumerate(observed_ims_cross)])
		else:
			obs_levels = np.array([np.mean(obs) for obs in observed_ims])
			if config_dict['ps_type']=='cross':
				obs_levels_cross = np.array([np.mean(obs) for o, obs in enumerate(observed_ims_cross)])

		print('obs levels are ', obs_levels)
		if pscb_dict['ff_bias_correct']: # cross
			if config_dict['ps_type']=='auto' and ff_biases is None:
				ff_biases = compute_ff_bias(obs_levels, weights=weights_photonly, mask_fractions=mask_fractions)
				print('FF biases are ', ff_biases)
			elif config_dict['ps_type']=='cross':
				if ciber_cross_ciber:
					print("calculating diagonal FF bias for ciber x ciber")
					ff_biases = compute_ff_bias(obs_levels, weights=weights_photonly, mask_fractions=mask_fractions, \
												mean_normalizations_cross=obs_levels_cross, weights_cross=weights_photonly_cross)
					print('FF biases for cross spectrum are ', ff_biases)
				elif ciber_cross_gal or ciber_cross_iris:
					ff_biases = np.ones((nfield))

		# nls_estFF_nofluc = []
		N_ells_est = []
		# for the tests with bright masking thresholds, we can calculate the ff realizations with the bright mask and this should be fine for the noise bias
		# for the actual flat field estimates of the observations, apply the deeper masks when stacking the off fields. The ff error will be that of the deep masks, 
		# and so there may be some bias unaccounted for by the analytic multiplicative bias.
		# I think the same holds for the gradients, just use the deeper masks

		# make monte carlo FF realizations that are used for the noise realizations (how many do we ultimately need?)
		if i==0 and pscb_dict['save_ff_ests'] and pscb_dict['noise_debias']:

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

				# fpath_dict['ff_est_dirpath'] = fpath_dict['ciber_mock_fpath']+'030122/ff_realizations/'+run_name

				np.savez(fpath_dict['ff_est_dirpath']+'/'+run_name+'/ff_realization_estimate_ffidx'+str(ffidx)+'.npz', \
						ff_realization_estimates = ff_realization_estimates)

				# np.savez(fpath_dict['ciber_mock_fpath']+'030122/ff_realizations/'+run_name+'/ff_realization_estimate_ffidx'+str(ffidx)+'.npz', \
				# 		ff_realization_estimates = ff_realization_estimates)

		if pscb_dict['save_intermediate_cls']:
			cls_masked_prenl, cls_postffb_corr, cls_masked_postnl,\
				 cls_postmkk_prebl, cls_postbl, cls_postffb_corr, cls_post_isl_sub, cls_post_tcorr = [np.zeros(single_ps_set_shape) for x in range(8)]


		for fieldidx, obs in enumerate(observed_ims):
			print('fieldidx = ', fieldidx)

			if pscb_dict['apply_mask']:
				target_mask = joint_maskos[fieldidx]

				# plot_map(obs*target_mask, title='obs*target mask')

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


			if not pscb_dict['ff_estimate_correct']:
				simmap_dc, simmap_dc_cross = None, None
			else:
				if pscb_dict['apply_mask']:
					simmap_dc = np.mean(obs[target_mask==1])
					if config_dict['ps_type']=='cross':
						simmap_dc_cross = np.mean(observed_ims_cross[fieldidx][target_mask==1])
				else:
					simmap_dc = np.mean(obs)
					if config_dict['ps_type']=='cross':
						simmap_dc_cross = np.mean(observed_ims_cross[fieldidx])

			# noise bias from sims with no fluctuations    
			print('and nowwww here on i = ', i)

			nl_estFF_nofluc = None
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

					print('LOADING NOISE MODEL FROM ', noisemodl_fpath)

					noisemodl_file = np.load(noisemodl_fpath)
					fourier_weights_nofluc = noisemodl_file['fourier_weights_nofluc']

					mean_cl2d_nofluc = noisemodl_file['mean_cl2d_nofluc']
					N_ell_est = noisemodl_file['nl_estFF_nofluc']


				else:
					all_ff_ests_nofluc = None
					if pscb_dict['iterate_grad_ff']:
						all_ff_ests_nofluc = []
						for ffidx in range(float_param_dict['nmc_ff']):
							ff_file = np.load(fpath_dict['ff_est_dirpath']+'/'+run_name+'/ff_realization_estimate_ffidx'+str(ffidx)+'.npz')
							# ff_file = np.load(fpath_dict['ff_est_dirpath']+'/ff_realization_estimate_ffidx'+str(ffidx)+'.npz')
							# ff_file = np.load(fpath_dict['ciber_mock_fpath']+'030122/ff_realizations/'+run_name+'/ff_realization_estimate_ffidx'+str(ffidx)+'.npz')
							all_ff_ests_nofluc.append(ff_file['ff_realization_estimates'][fieldidx])

							if ffidx==0:
								plot_map(all_ff_ests_nofluc[0], title='loaded MC ff estimate 0')

					print('median obs is ', np.median(obs))
					print('simmap dc is ', simmap_dc)

					shot_sigma_sb_zl_noisemodl = cbps.compute_shot_sigma_map(inst, image=obs, nfr=nfr_fields[fieldidx])


					# todo cross
					if ciber_cross_ciber:
						fourier_weights_cross, mean_cl2d_cross= cbps.estimate_noise_power_spectrum(nsims=float_param_dict['n_FW_sims'], n_split=float_param_dict['n_FW_split'], apply_mask=pscb_dict['apply_mask'], \
						   noise_model=read_noise_models[fieldidx], read_noise=pscb_dict['with_inst_noise'], inst=inst, ifield=ifield_noise_list[fieldidx], show=False, mask=target_mask, \
							photon_noise=pscb_dict['with_photon_noise'], ff_estimate=None, shot_sigma_sb=shot_sigma_sb_zl_noisemodl, inplace=False, \
							field_nfr=nfr_fields[fieldidx], simmap_dc=simmap_dc, ff_truth=smooth_ff, mc_ff_estimates=all_ff_ests_nofluc, gradient_filter=pscb_dict['gradient_filter'])
						
						N_ell_est = cbps.compute_noise_power_spectrum(inst, noise_Cl2D=mean_cl2d_nofluc.copy(), inplace=False, apply_FW=pscb_dict['apply_FW'], weights=fourier_weights_nofluc)

					else:
						fourier_weights_nofluc, mean_cl2d_nofluc = cbps.estimate_noise_power_spectrum(nsims=float_param_dict['n_FW_sims'], n_split=float_param_dict['n_FW_split'], apply_mask=pscb_dict['apply_mask'], \
									   noise_model=read_noise_models[fieldidx], read_noise=pscb_dict['with_inst_noise'], inst=inst, ifield=ifield_noise_list[fieldidx], show=False, mask=target_mask, \
										photon_noise=pscb_dict['with_photon_noise'], ff_estimate=None, shot_sigma_sb=shot_sigma_sb_zl_noisemodl, inplace=False, \
										field_nfr=nfr_fields[fieldidx], simmap_dc=simmap_dc, ff_truth=smooth_ff, mc_ff_estimates=all_ff_ests_nofluc, gradient_filter=pscb_dict['gradient_filter'])
						N_ell_est = cbps.compute_noise_power_spectrum(inst, noise_Cl2D=mean_cl2d_nofluc.copy(), inplace=False, apply_FW=pscb_dict['apply_FW'], weights=fourier_weights_nofluc)

					print('N_ell_est = ', N_ell_est)

					if fpath_dict['noisemodl_run_name'] is None:
						fpath_dict['noisemodl_run_name'] = run_name 

					noisemodl_tailpath = '/noise_bias_fieldidx'+str(fieldidx)+'.npz'
					noisemodl_fpath = fpath_dict['noisemodl_basepath'] + fpath_dict['noisemodl_run_name'] + noisemodl_tailpath


					if pscb_dict['save'] or not os.path.exists(noisemodl_fpath):
						print('SAVING NOISE MODEL BIAS FILE..')
						if os.path.exists(noisemodl_fpath):
							print('Overwriting existing noise model bias file..')

						# if data_type=='mock':
						# 	# noisemodl_fpath = '/Volumes/Seagate Backup Plus Drive/Toolkit/Mirror/Richard/ciber_mocks/030122/noise_models_sim/'+run_name+'/noise_model_fieldidx'+str(fieldidx)+'.npz'
						# else:
						# 	noisemodl_fpath = '/Volumes/Seagate Backup Plus Drive/Toolkit/Mirror/Richard/ciber_mocks/030122/noise_models_sim/'+run_name+'/noise_model_observed_fieldidx'+str(fieldidx)+'.npz'

						np.savez(noisemodl_fpath, \
								fourier_weights_nofluc=fourier_weights_nofluc, mean_cl2d_nofluc=mean_cl2d_nofluc, nl_estFF_nofluc=N_ell_est)


				cbps.FW_image=fourier_weights_nofluc.copy() #cross?
				N_ells_est.append(N_ell_est)

			if pscb_dict['load_ptsrc_cib'] or pscb_dict['load_trilegal'] or data_type=='observed':

				if config_dict['ps_type']=='auto' or config_dict['cross_type'] != 'iris':
					beam_correct = True
				else:
					print('setting beam_correct to False..')
					beam_correct = False
				B_ell_field = B_ells[fieldidx]
				if ciber_cross_ciber:
					print('Loading B_ell for cross field TM ', cross_inst)
					B_ell_field_cross = B_ells_cross[fieldidx]

			else:
				beam_correct = False
				B_ell_field = None


			# FF_correct = True # divides image by estimated flat field, not to be confused with flat field bias correction
			if pscb_dict['iterate_grad_ff']: # set gradient_filter, FF_correct to False here because we've already done gradient filtering/FF estimation
				print('iterate_grad_ff is True, setting gradient filter and FF correct to False..')
				gradient_filter = False
				FF_correct = False
			else:
				gradient_filter=pscb_dict['gradient_filter']
				FF_correct = pscb_dict['ff_estimate_correct']

			print('obs mean is ', np.mean(obs[target_mask==1]))

			if pscb_dict['mkk_ffest_hybrid']:
				print('Correcting power spectrum with hybrid mask-FF mode coupling matrix..')

			if pscb_dict['show_plots']:
				maskobs = obs*target_mask
				maskobs[target_mask==1]-=np.mean(maskobs[target_mask==1])
				plot_map(maskobs, title='FF corrected, mean subtracted CIBER map ('+cbps.ciber_field_dict[ifield_list[fieldidx]]+')', hipct=99.999, lopct=0., cmap='bwr')

				maskobs_smooth = target_mask*gaussian_filter(maskobs, sigma=10)
				maskobs_smooth[target_mask==1]-=np.mean(maskobs_smooth[target_mask==1])

				plot_map(maskobs_smooth, title='FF corrected, smoothed, mean-subtracted CIBER map ('+cbps.ciber_field_dict[ifield_list[fieldidx]]+')', cmap='bwr', hipct=99.999, lopct=0.)

				# plot_map(obs*target_mask, title='obs*target mask')


			if config_dict['ps_type']=='auto':
				lb, processed_ps_nf, cl_proc_err, _ = cbps.compute_processed_power_spectrum(inst, apply_mask=pscb_dict['apply_mask'], \
												 mask=target_mask, image=obs, convert_adufr_sb=False, \
												mkk_correct=pscb_dict['apply_mask'], inv_Mkk=target_invMkk, beam_correct=beam_correct, B_ell=B_ell_field, \
												apply_FW=pscb_dict['apply_FW'], verbose=pscb_dict['verbose'], noise_debias=pscb_dict['noise_debias'], \
											 FF_correct=FF_correct, FW_image=fourier_weights_nofluc, FF_image=ff_estimates[fieldidx],\
												 gradient_filter=gradient_filter, save_intermediate_cls=pscb_dict['save_intermediate_cls'], N_ell=N_ells_est[fieldidx])

			elif config_dict['ps_type']=='cross':

				if pscb_dict['show_plots']:
					maskobs_cross = observed_ims_cross[fieldidx]*target_mask
					maskobs_cross[target_mask==1]-=np.mean(maskobs_cross[target_mask==1])

					if config_dict['cross_type']=='IRIS':
						plot_map(maskobs_cross, title='Mean-subtracted IRIS map ('+cbps.ciber_field_dict[ifield_list[fieldidx]]+')', hipct=99.999, lopct=0., cmap='bwr')
					elif config_dict['cross_type']=='ciber':
						plot_map(maskobs_cross, title='Mean-subtracted TM2 map ('+cbps.ciber_field_dict[ifield_list[fieldidx]]+')', hipct=99.999, lopct=0., cmap='bwr')


				lb, processed_ps_nf, cl_proc_err, _ = cbps.compute_processed_power_spectrum(inst, apply_mask=pscb_dict['apply_mask'], \
												 mask=target_mask, image=obs, cross_image=observed_ims_cross[fieldidx], convert_adufr_sb=False, \
												mkk_correct=pscb_dict['apply_mask'], inv_Mkk=target_invMkk, beam_correct=beam_correct, B_ell=B_ell_field, \
												apply_FW=pscb_dict['apply_FW'], verbose=pscb_dict['verbose'], noise_debias=pscb_dict['noise_debias'], \
											 FF_correct=FF_correct, FF_image=ff_estimates[fieldidx],\
												 gradient_filter=gradient_filter, save_intermediate_cls=pscb_dict['save_intermediate_cls'], N_ell=N_ells_est[fieldidx])



			if pscb_dict['save_intermediate_cls']:
				cls_masked_prenl[fieldidx,:] = cbps.masked_Cl_pre_Nl_correct
				cls_masked_postnl[fieldidx,:] = cbps.masked_Cl_post_Nl_correct
				cls_postmkk_prebl[fieldidx,:] = cbps.cl_post_mkk_pre_Bl
				cls_postbl[fieldidx,:] = processed_ps_nf.copy()

			if pscb_dict['ff_bias_correct']:
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
					# cib_cl_file = fits.open(base_path+'TM'+str(inst)+'/powerspec/cls_cib_vs_maglim_ifield'+str(ifield_list[fieldidx])+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_Vega_magcut_test.fits')
					unmasked_cib_cl = cib_cl_file['cls_cib'].data['cl_maglim_'+str(masking_maglim)]
					true_ps = unmasked_cib_cl + diff_cl
					# print('diff_cl:', diff_cl)
					print('unmasked cib cl', unmasked_cib_cl)

					if pscb_dict['load_trilegal'] and masking_maglim < 16:

						print('Adding power spectrum from all stars fainter than masking limit..')
						if pscb_dict['subtract_subthresh_stars']:
							print('Setting subtract_subthresh_stars to False..')
							pscb_dict['subtract_subthresh_stars'] = False

						trilegal_cl_fpath = fpath_dict['isl_resid_ps_path']+'/cls_trilegal_vs_maglim_ifield'+str(ifield_list[fieldidx])+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_Vega_magcut.fits'
						# trilegal_cl_fpath = trilegal_base_path+'TM'+str(inst)+'/powerspec/cls_trilegal_vs_maglim_ifield'+str(ifield_list[fieldidx])+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_Vega_magcut_test.fits'
						trilegal_cl = fits.open(trilegal_cl_fpath)['cls_trilegal'].data['cl_maglim_'+str(masking_maglim)]
						true_ps += trilegal_cl 
						print('trilegal_cl is ', trilegal_cl)
						print('and true cl is now ', true_ps)

				elif pscb_dict['load_trilegal']:
					print('Ground truth in this case is the ISL fainter than our masking depth..')

					trilegal_cl_fpath = fpath_dict['isl_resid_ps_path']+'/cls_trilegal_vs_maglim_ifield'+str(ifield_list[fieldidx])+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_Vega_magcut_test.fits'
					# trilegal_cl_fpath = trilegal_base_path+'TM'+str(inst)+'/powerspec/cls_trilegal_vs_maglim_ifield'+str(ifield_list[fieldidx])+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_Vega_magcut_test.fits'
					true_ps = fits.open(trilegal_cl_fpath)['cls_trilegal'].data['cl_maglim_'+str(masking_maglim)]


				else:
					# lb, cl_diffreal, clerr_diffreal = get_power_spec(diff_realizations[fieldidx]-np.mean(diff_realizations[fieldidx]))
					# print('cl_diffreal is ', cl_diffreal)
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
				print('Correcting for transfer function..')
				print('t_ell_av here is ', t_ell_av)
				processed_ps_nf /= t_ell_av

				if pscb_dict['save_intermediate_cls']:
					cls_post_tcorr[fieldidx,:] = processed_ps_nf.copy()

			
			if pscb_dict['subtract_subthresh_stars']:
				# maybe load precomputed power spectra and assume we have it nailed down
				print('before trilegal star correct correct ps is ', processed_ps_nf)
				print('subtracting sub-threshold stars!!!!')

				print('isl resid ps path:', fpath_dict['isl_resid_ps_path'])

				trilegal_cl_fpath = fpath_dict['isl_resid_ps_path']+'/cls_isl_vs_maglim_ifield'+str(ifield_list[fieldidx])+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_Vega_magcut.fits'
				# trilegal_cl_fpath = trilegal_base_path+'TM'+str(inst)+'/powerspec/cls_trilegal_vs_maglim_ifield'+str(ifield_list[fieldidx])+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_Vega_magcut_test.fits'
				
				trilegal_cl = fits.open(trilegal_cl_fpath)['cls_isl'].data['cl_maglim_'+str(masking_maglim)]
				print(trilegal_cl)
				processed_ps_nf -= trilegal_cl

				if pscb_dict['save_intermediate_cls']:
					cls_post_isl_sub[fieldidx,:] = processed_ps_nf.copy()

			print('proc:', processed_ps_nf)

			recovered_power_spectra[i, fieldidx, :] = processed_ps_nf

			recovered_dcl[i, fieldidx, :] = cl_proc_err

			# if pscb_dict['show_plots'] and pscb_dict['data_type']=='mock':
			# 	plt.figure(figsize=(6, 5))
			# 	prefac = lb*(lb+1)/(2*np.pi)
			# 	plt.errorbar(lb, prefac*true_ps, yerr=prefac*cl_proc_err, fmt='o-', capsize=3, color='k', label='ground truth')
			# 	plt.errorbar(lb, prefac*processed_ps_nf, fmt='o-', capsize=3, color='r', label='recovered')
			# 	plt.legend(fontsize=14)
			# 	plt.yscale('log')
			# 	plt.xscale('log')
			# 	plt.title('cl_orig:')
			# 	plt.xlabel('$\\ell$', fontsize=18)
			# 	plt.ylabel('$D_{\\ell}$', fontsize=18)
			# 	plt.show()

			if pscb_dict['show_plots']:
				plt.figure(figsize=(6, 5))
				prefac = lb*(lb+1)/(2*np.pi)
				if data_type=='mock':
					plt.errorbar(lb, prefac*true_ps, yerr=prefac*cl_proc_err, fmt='o-', capsize=3, color='k', label='ground truth')
				plt.errorbar(lb, prefac*processed_ps_nf, yerr=prefac*cl_proc_err, fmt='o-', capsize=3, color='r', label='recovered')
				plt.legend(fontsize=14)
				plt.yscale('log')
				plt.xscale('log')
				plt.ylim(1e-2, 1e4)
				plt.grid()
				plt.title('ifield '+str(ifield_list[fieldidx]), fontsize=16)
				plt.xlabel('$\\ell$', fontsize=18)
				plt.ylabel('$D_{\\ell}$', fontsize=18)
				plt.show()


			if data_type=='mock':
				print('true ps', true_ps)
				signal_power_spectra[i, fieldidx, :] = true_ps
				print('proc (nofluc) / ebl = ', processed_ps_nf/true_ps)
				print('mean ps bias is ', np.mean(recovered_power_spectra[i,:,:]/signal_power_spectra[i,:,:], axis=0))


		# nls_estFF_nofluc = np.array(nls_estFF_nofluc)
		N_ells_est = np.array(N_ells_est)
		print('mean nl:', np.mean(N_ells_est, axis=0))

		if not pscb_dict['save_intermediate_cls']:
			cls_inter, inter_labels = None, None 
		else:
			cls_inter.extend([cls_masked_prenl, cls_masked_postnl, cls_postmkk_prebl, cls_postbl, cls_postffb_corr, cls_post_tcorr, cls_post_isl_sub])
			inter_labels.extend(['masked_Cl_pre_Nl_correct', 'masked_Cl_post_Nl_correct', 'post_mkk_pre_Bl', 'post_Bl_pre_ffbcorr', 'post_ffb_corr','post_tcorr', 'post_isl_sub'])
			print('cls inter has shape', np.array(cls_inter).shape)
			print('inter labels has length', len(inter_labels))
			
		if pscb_dict['save']:
			savestr = fpath_dict['sim_test_fpath']+run_name+'/input_recovered_ps_estFF_simidx'+str(i)

			if fpath_dict['add_savestr'] is not None:
				savestr += '_'+fpath_dict['add_savestr']
			np.savez(savestr+'.npz', lb=lb, data_type=data_type, \
				signal_ps=signal_power_spectra[i], recovered_ps_est_nofluc=recovered_power_spectra[i], recovered_dcl=recovered_dcl[i], \
					 dgl_scale_fac=float_param_dict['dgl_scale_fac'], niter=float_param_dict['niter'], cls_inter=cls_inter, inter_labels=inter_labels, **pscb_dict)

			if i==0:
				with open(fpath_dict['sim_test_fpath']+run_name+'/params_read.txt', 'w') as file:
					for dicto in [fpath_dict, pscb_dict, float_param_dict, config_dict]:
						for key in dicto:
							file.write(key+': '+str(dicto[key])+'\n')



	return lb, signal_power_spectra, recovered_power_spectra, recovered_dcl, N_ells_est, cls_inter, inter_labels



# if not os.path.exists('data/b_ell_estimates_beam_correction_tm'+str(inst)+'.npz'):
# 	print('Path does not exist..')
# 	for fieldidx, ifield in enumerate(ifield_list):
# 		lb, B_ell, B_ell_list = cbps.compute_beam_correction_posts(ifield, inst, nbins=cbps.n_ps_bin, n_fine_bin=10, tail_path='data/psf_model_dict_updated_081121_ciber.npz')
# 		B_ells[fieldidx] = B_ell

# 	np.savez('data/b_ell_estimates_beam_correction_tm'+str(inst)+'.npz', lb=lb, B_ells=B_ells, ifield_list=ifield_list, inst=inst, n_fine_bin=10)
# B_ells = np.load('data/fluctuation_data/TM'+str(inst)+'/beam_correction/ptsrc_blest_TM'+str(inst)+'_observed.npz')
# B_ells = np.load('data/b_ell_estimates_beam_correction_tm'+str(inst)+'.npz')['B_ells']

#         if pscb_dict['compute_beam_correction']:
#             print('Computing beam correction to be saved to bls_fpath='+bls_fpath+'..')
# #             lb, diff_norm_array = estimate_b_ell_from_maps(cbps, inst, ifield_list, ciber_maps=None, pixsize=7., J_bright_mag=11., J_tot_mag=17.5, nsim=1, ell_norm=10000, plot=False, save=False, \
# #                                     niter = niter, ff_stack_min=ff_stack_min, data_type='mock')
# #             print('diff norm array:', diff_norm_array)            
            # for fieldidx, ifield in enumerate(ifield_list):
            #     lb, B_ell, B_ell_list = cbps.compute_beam_correction_posts(ifield, inst, nbins=cbps.n_ps_bin, n_fine_bin=10, tail_path='data/psf_model_dict_updated_081121_ciber.npz')
            #     B_ells[fieldidx] = B_ell
#             np.savez(bls_fpath, B_ells=B_ells, ifield_list=ifield_list)
		


