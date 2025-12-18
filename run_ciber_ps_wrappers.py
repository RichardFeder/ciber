import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from scipy import interpolate
from ciber.instrument.noise_model import *
import config
from ciber.core.powerspec_pipeline import *
from ps_pipeline_go import *
from ciber.mocks.cib_mocks import *
from ciber.io.ciber_data_utils import *


def run_ciber_cross_observed(inst, cross_inst, run_name_list, mag_lim_J_list, mask_tail_list=None, mask_tail_ffest_list=None, ifield_list=[4, 5, 6, 7, 8], datestr='111323', \
							load_noise_bias=False, save_ff_ests=True, sigma_clip_dict=None, ff_bounds_dict=None, ff_transition_mag=15.0, \
							show_plots=True, save=True, verbose=True, \
							ffest_run_name_list=None, ffest_run_name_cross_list=None, apply_FW=False, \
							ff_estimate_correct = True, mkk_mask_type='quadoff_grad_fcsub_order2', mkk_ffest_type='ffest_quadoff_grad_fcsub_order2', \
							fc_sub=True, fc_sub_quad_offsets=True, fc_sub_n_terms=2, interp_order=1, estimate_cross_noise=False):
	data_type='observed'
	cbps = CIBER_PS_pipeline()
	config_dict, pscb_dict, float_param_dict, fpath_dict = return_default_cbps_dicts()
	
	if sigma_clip_dict is None:
		sigma_clip_dict = dict({11.0:None, 12.0:None, 13.0:None, 14.0:30, 15.0:10, 16.0:10.0, 16.5:5, 17.0:4, \
					   17.5:4, 18.0:4, 18.5:4})
	
	if ff_bounds_dict is None:
		ff_bounds_dict = dict({1:[0.7, 1.4], 2:[0.6, 1.45]})
		
	ff_min, ff_max = ff_bounds_dict[inst][0], ff_bounds_dict[inst][1]

	
	all_recovered_power_spectra = []

	for m, maglim_J in enumerate(mag_lim_J_list):
		
		
		
		run_name = run_name_list[m]
		
		maglim_H = maglim_J - 0.5 
		maglim_J_ffest = max(17.5, maglim_J)
		maglim_H_ffest = maglim_J_ffest - 0.5
		clip_sigma = sigma_clip_dict[maglim_J]
		
		print('maglims:', maglim_J, maglim_J_ffest, maglim_H, maglim_H_ffest)

		if mask_tail_list is None:
			cross_union_mask_tail = 'maglim_Vega_J='+str(maglim_J)+'_H='+str(maglim_H)+'_ukdebias_111323'
		else:
			cross_union_mask_tail = mask_tail_list[m]
		regrid_mask_tail = 'Jlim_Vega_'+str(maglim_J)+'_Hlim_Vega_'+str(maglim_H)+'_ukdebias_111323'
			
		if mask_tail_ffest_list is None:
			mask_tail_ffest = 'maglim_Vega_J='+str(maglim_J_ffest)+'_H='+str(maglim_H_ffest)+'_ukdebias_111323'
		else:
			mask_tail_ffest = mask_tail_ffest_list[m]
		
		print('cross union tail, mask tail ffest: ', cross_union_mask_tail, mask_tail_ffest)

		if maglim_J <= ff_transition_mag:
			mkk_ffest_hybrid = False
			mkk_type = mkk_mask_type
		else:
			mkk_ffest_hybrid = True
			mkk_type = mkk_ffest_type
			
		bls_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/beam_correction/bl_est_postage_stamps_TM'+str(inst)+'_081121.npz'
		bls_fpath_cross = config.ciber_basepath+'data/fluctuation_data/TM'+str(cross_inst)+'/beam_correction/bl_est_postage_stamps_TM'+str(cross_inst)+'_081121.npz'
		tls_base_path = config.ciber_basepath+'data/transfer_function/'
		
		sim_test_fpath = config.ciber_basepath+'data/input_recovered_ps/'+datestr+'/TM'+str(inst)+'/'
		noise_models_sim_dirpath = config.ciber_basepath+'data/noise_models_sim/'+datestr+'/TM'+str(inst)+'_TM'+str(cross_inst)+'_cross/'
		make_fpaths([sim_test_fpath+run_name, noise_models_sim_dirpath+run_name])
		
		iterate_grad_ff = True
		
		if ffest_run_name_list is None:
			ffest_run_name = 'observed_Jlt'+str(maglim_J)+'_Hlt'+str(maglim_H)+'_072424_quadoff_grad_fcsub_order2'
#             ffest_run_name = 'observed_Jlt'+str(maglim_J)+'_072424_quadoff_grad_fcsub_order2'
#             ffest_run_name = 'observed_Jlt'+str(maglim_J)+'_Hlt'+str(maglim_H)+'_020624_ukdebias'
		else:
			ffest_run_name = ffest_run_name_list[m]
			
		if ffest_run_name_cross_list is None:
			ffest_run_name_cross = 'observed_Jlt'+str(maglim_J)+'_Hlt'+str(maglim_H)+'_072424_quadoff_grad_fcsub_order2'
#             ffest_run_name_cross = 'observed_Hlt'+str(maglim_J)+'_072424_quadoff_grad_fcsub_order2'
#             ffest_run_name_cross = 'observed_Jlt'+str(maglim_J)+'_Hlt'+str(maglim_H)+'_020624_ukdebias'
		else:
			ffest_run_name_cross = ffest_run_name_cross_list[m]
			
		noisemodl_run_name = None
#         noisemodl_run_name = 'ciber_cross_ciber_Jlt'+str(maglim_J)+'_Hlt'+str(maglim_H)+'_020724'
		
		
		lb, signal_power_spectra, \
			recovered_power_spectra, recovered_dcl, nls_estFF_nofluc,\
			cls_inter, inter_labels, ff_estimates, final_masked_images = run_cbps_pipeline(cbps, inst, 1, run_name, tls_base_path=tls_base_path, \
														ps_type='cross', cross_type='ciber', cross_inst=cross_inst,\
														sim_test_fpath=sim_test_fpath, n_FW_sims=500, n_FW_split=20, \
														apply_FW=False, datestr=datestr, datestr_trilegal=datestr,\
														show_plots=show_plots, mask_tail=cross_union_mask_tail, mask_tail_ffest=mask_tail_ffest, regrid_mask_tail=regrid_mask_tail,\
														data_type=data_type, noisemodl_basepath=noise_models_sim_dirpath, \
														per_quadrant=True, bls_fpath=bls_fpath, bls_fpath_cross=bls_fpath_cross, \
														noise_debias=False, iterate_grad_ff=iterate_grad_ff, load_noise_bias=load_noise_bias, \
														gradient_filter=False, ff_estimate_correct=ff_estimate_correct, \
														save=save, clip_sigma=clip_sigma, ffest_run_name=ffest_run_name, ffest_run_name_cross=ffest_run_name_cross, \
														 mkk_ffest_hybrid=mkk_ffest_hybrid, shut_off_plots=False, \
														 ff_min=ff_min, ff_max=ff_max, apply_tl_regrid=True, interp_order=interp_order, conserve_flux=False, \
														noisemodl_run_name=noisemodl_run_name, fc_sub=fc_sub, fc_sub_quad_offsets=fc_sub_quad_offsets, \
															fc_sub_n_terms=fc_sub_n_terms, mkk_type=mkk_type, clip_outlier_norm_modes=False, clip_norm_thresh=10, \
															estimate_cross_noise=estimate_cross_noise, quadoff_grad=False)


		all_recovered_power_spectra.append(recovered_power_spectra)
		
	return lb, all_recovered_power_spectra
		

def run_ciber_auto_observed(inst, run_name_list, mag_lim_list, mask_tail_list=None, mask_tail_ffest_list=None, ifield_list=[4, 5, 6, 7, 8], datestr='111323', \
							load_noise_bias=False, noisemodl_tailstr_list=None, save_ff_ests=True, sigma_clip_dict=None, ff_bounds_dict=None, ff_transition_mag=15.0, \
						   show_plots=True, save=True, verbose=True, mkk_type = None, \
						   mkk_mask_type='quadoff_grad_fcsub_order2', mkk_ffest_type='ffest_quadoff_grad_fcsub_order2', \
						   invert_spliced_matrix=False, compute_mkk_pinv=False, g1_use=None, \
						   compute_cl_theta=False, n_rad_bins=8, rad_offset=-np.pi/8., ff_max_nr=1.8, ell_min_wedge=1000, cut_cl_theta=False):
	
	masking_maglim_ff_dict = dict({1:17.5, 2:17.0})
	
	if sigma_clip_dict is None:
		sigma_clip_dict = dict({11.0:None, 12.0:None, 13.0:None, 14.0:20, 15.0:7.0, 16.0:7.0, 16.5:5, 17.0:4, \
						   17.5:4, 18.0:4, 18.5:4})
		
	if ff_bounds_dict is None:
		ff_bounds_dict = dict({1:[0.7, 1.4], 2:[0.6, 1.45]})
		
	ff_min, ff_max = ff_bounds_dict[inst][0], ff_bounds_dict[inst][1]
	
	cbps = CIBER_PS_pipeline()
	config_dict, pscb_dict, float_param_dict, fpath_dict = return_default_cbps_dicts()
	fpath_dict, list_of_dirpaths, base_path, trilegal_base_path = set_up_filepaths_cbps(fpath_dict, inst, 'test', datestr,\
																					datestr_trilegal='112022', data_type='observed', \
																				   save_fpaths=True)

	if g1_use is not None:

		g1_fac_orig = cbps.g1_facs[inst]

		cbps.g1_facs[inst] = g1_use 

		cbps.g2_facs[inst] *= g1_fac_orig/g1_use

		print('g1 fac is ', cbps.g1_facs[inst], 'g2 is ', cbps.g2_facs[inst], 'total cal is ', cbps.cal_facs[inst], 'g1xg2 = ', cbps.g1_facs[inst]*cbps.g2_facs[inst])


	masking_maglim_ff_ref = masking_maglim_ff_dict[inst]
	band = cbps.inst_to_band[inst]
	
	all_recovered_power_spectra, all_masked_images = [], []
	
	for m, maglim_iter in enumerate(mag_lim_list):
		
		run_name = run_name_list[m]
		print('RUN NAME:', run_name)
		masking_maglim = maglim_iter
		masking_maglim_ff = max(masking_maglim_ff_ref, masking_maglim)
					
		if mask_tail_list is None:
			mask_tail = 'maglim_'+band+'_Vega_'+str(masking_maglim)+'_111323_ukdebias'            
		else:
			mask_tail = mask_tail_list[m]
			
		if mask_tail_ffest_list is None:
			mask_tail_ffest = 'maglim_'+band+'_Vega_'+str(masking_maglim_ff)+'_111323_ukdebias'
		else:
			mask_tail_ffest = mask_tail_ffest_list[m]

		if masking_maglim <= ff_transition_mag:
			mkk_ffest_hybrid = False
			mkk_type = mkk_mask_type
		else:
			mkk_ffest_hybrid = True
			mkk_type = mkk_ffest_type

		clip_sigma = sigma_clip_dict[masking_maglim]
		print('clip sigma is ', clip_sigma)
		
		if masking_maglim < masking_maglim_ff:
			pt_src_ffnoise = True
			clip_outlier_norm_modes=False
			diff_cl2d_clipthresh=None
			diff_cl2d_fwclip=False
			
		else:
			pt_src_ffnoise = False
			clip_outlier_norm_modes=True
			diff_cl2d_clipthresh=5
			diff_cl2d_fwclip=True
		   
		clip_outlier_norm_modes=True

			
		ff_est_dirpath = config.ciber_basepath+'data/ff_mc_ests/'+datestr+'/TM'+str(inst)+'/'
		noise_models_sim_dirpath = config.ciber_basepath+'data/noise_models_sim/'+datestr+'/TM'+str(inst)+'/'
		sim_test_fpath = config.ciber_basepath+'data/input_recovered_ps/'+datestr+'/TM'+str(inst)+'/'
		make_fpaths([noise_models_sim_dirpath+run_name, sim_test_fpath+run_name, ff_est_dirpath+run_name])
		bls_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/beam_correction/bl_est_postage_stamps_TM'+str(inst)+'_081121.npz'

		mkk_base_path = fpath_dict['mkk_base_path']
		mkk_ffest_base_path = fpath_dict['mkk_ffest_base_path']
		
		noisemodl_run_name = None
		if load_noise_bias:

			if noisemodl_tailstr_list is not None:
				noisemodl_run_name = noisemodl_tailstr_list[m]
			else:
				noisemodl_run_name = run_name
			
		lb, signal_power_spectra, \
			recovered_power_spectra, recovered_dcl, nls_estFF_nofluc,\
			cls_inter, inter_labels, ff_estimates, final_masked_images = run_cbps_pipeline(cbps, inst, 1, run_name, ifield_list=ifield_list,\
													datestr=datestr, show_plots=show_plots, \
													mask_tail=mask_tail, mask_tail_ffest=mask_tail_ffest, \
													mkk_base_path=mkk_base_path, mkk_ffest_base_path=mkk_ffest_base_path, \
													noisemodl_basepath=noise_models_sim_dirpath,\
													sim_test_fpath=sim_test_fpath, load_noise_bias=load_noise_bias,\
													bls_fpath=bls_fpath, masking_maglim=masking_maglim, \
													clip_sigma=clip_sigma, save_ff_ests=save_ff_ests, ff_est_dirpath=ff_est_dirpath, \
													mkk_ffest_hybrid=mkk_ffest_hybrid, noisemodl_run_name=noisemodl_run_name,\
													niter=1, save=save, ff_max=ff_max, ff_min=ff_min, \
													masking_maglim_ff=masking_maglim_ff, \
													verbose=verbose, pt_src_ffnoise=pt_src_ffnoise,\
													per_quadrant=True, quadoff_grad=False, save_fourier_planes=False, \
													mkk_type=mkk_type, noise_modl_type='full_051324', \
													diff_cl2d_fwclip=diff_cl2d_fwclip, diff_cl2d_clipthresh=diff_cl2d_clipthresh, clip_outlier_norm_modes=clip_outlier_norm_modes, clip_norm_thresh=10, \
													fc_sub=True, fc_sub_quad_offset=True, fc_sub_n_terms=2, make_preprocess_file=False, nitermax=10, \
													invert_spliced_matrix=invert_spliced_matrix, compute_mkk_pinv=compute_mkk_pinv, \
													compute_cl_theta=compute_cl_theta, n_rad_bins=n_rad_bins, rad_offset=rad_offset, \
													ff_max_nr=ff_max_nr, ell_min_wedge=ell_min_wedge, cut_cl_theta=cut_cl_theta)


		all_recovered_power_spectra.append(recovered_power_spectra)

		all_masked_images.append(final_masked_images)
		
	return lb, all_recovered_power_spectra, all_masked_images
