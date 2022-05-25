import numpy as np
import astropy.io.fits as fits
import astropy.wcs as wcs
import pandas as pd
import scipy.io
from scipy.ndimage import gaussian_filter
import matplotlib
import matplotlib.pyplot as plt
import pickle
from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel
from astropy.wcs import WCS
from scipy.io import loadmat
import os

from powerspec_utils import *
from ciber_powerspec_pipeline import CIBER_PS_pipeline, iterative_gradient_ff_solve, lin_interp_powerspec,\
									 grab_all_simidx_dat, grab_recovered_cl_dat, generate_synthetic_mock_test_set,\
									 instantiate_dat_arrays_fftest, instantiate_cl_arrays_fftest, calculate_powerspec_quantities, compute_powerspectra_realdat
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



def run_cbps_pipeline(cbps, inst, nsims, run_name, simidx0=0, datestr='100921', data_type='mock', ciber_mock_fpath='/Volumes/Seagate Backup Plus Drive/Toolkit/Mirror/Richard/ciber_mocks/',\
	sim_test_fpath = 'data/input_recovered_ps/sim_tests_030122/', load_ptsrc_cib=False, load_trilegal=False, masking_maglim=17.5, compute_beam_correction=True, bls_fpath=None,\
	n_cib_set=20, draw_cib_setidxs=False, aug_rotate=False, ifield_list = None, apply_mask = False,\
	with_read_noise = True, with_photon_noise=True, apply_FW = True, n_FW_sims=500, n_FW_split=10, \
	apply_smooth_FF = True, ff_fpath='data/flatfield/TM1_FF/TM1_field4_mag=17.5_FF.fits', smooth_sigma=5,\
	same_zl_levels = False, zl_levels=None, apply_zl_gradient = True, theta_mag=0.01, gradient_filter = True,\
	iterate_grad_ff = False, niter=3, transfer_function_correct=False, ff_bias_correct=False, ff_biases=None, \
	same_int = False, same_dgl = True, dgl_scale_fac = 5, use_ff_weights = True, \
	plot_ff_error=False, indiv_ifield=6, nfr_same = 25, \
	save = True, save_ff_ests = True, nmc_ff = 10, verbose=True):

	''' 
	Wrapper function for CIBER power spectrum pipeline. This is the latest version used for mock data, though I would like to 
	extend this to handle observed data as well. 

	'''

	# base_path = '/Volumes/Seagate Backup Plus Drive/Toolkit/Mirror/Richard/ciber_mocks/100921/'
	# datestr = '100921'

	base_path = ciber_mock_fpath+datestr+'/'

	if ifield_list is None:
		ifield_list = [4, 5, 6, 7, 8]

	nfield = len(ifield_list)

	if same_int:
		nfr_fields = [nfr_same for i in range(nfield)]
		ifield_noise_list = [indiv_ifield for i in range(nfield)]
		ifield_list_ff = [indiv_ifield for i in range(nfield)]
	else:
		nfr_fields = [cbps.field_nfrs[ifield] for ifield in ifield_list]
		ifield_noise_list = ifield_list.copy()
		ifield_list_ff = ifield_list.copy()

	if zl_levels is None:
		if same_zl_levels:
			zl_levels = [cbps.zl_levels_ciber_fields[inst][cbps.ciber_field_dict[indiv_ifield]] for i in range(nfield)]
		else:
			zl_levels = [cbps.zl_levels_ciber_fields[inst][cbps.ciber_field_dict[ifield]] for ifield in ifield_list]

	if verbose:
		print('NFR fields is ', nfr_fields)
		print('ZL levels are ', zl_levels)

	if use_ff_weights:
		weights_photonly = cbps.compute_ff_weights(inst, zl_levels, read_noise_models=None, ifield_list=ifield_list_ff)
		print('FF weights are :', weights_photonly)


	# instantiate data arrays 
	field_set_shape = (len(ifield_list), cbps.dimx, cbps.dimy)
	ps_set_shape = (nsims, len(ifield_list), cbps.n_ps_bin)

	clus_realizations, zl_perfield, ff_estimates = [np.zeros(field_set_shape) for i in range(3)]
	
	inv_Mkks, joint_maskos = None, None


	if apply_zl_gradient:
		zl_gradients = np.zeros(field_set_shape)
	if apply_mask:
		joint_maskos = np.zeros(field_set_shape)

	signal_power_spectra, recovered_power_spectra = [np.zeros(ps_set_shape) for i in range(2)]

	smooth_ff = None
	if apply_smooth_FF:
		ff = fits.open(ff_fpath)
		smooth_ff = gaussian_filter(ff[0].data, sigma=smooth_sigma)

	ff_fpath, noisemod_fpath, input_recov_ps_fpath = init_mocktest_fpaths(ciber_mock_fpath, run_name)


	if with_read_noise:
		read_noise, read_noise_models = [np.zeros(field_set_shape) for i in range(2)]
		for fieldidx, ifield in enumerate(ifield_noise_list):
			noise_cl2d = cbps.load_noise_Cl2D(ifield, inst, noise_fpath='data/fluctuation_data/TM'+str(inst)+'/noiseCl2D/field'+str(ifield)+'_noiseCl2D_110421.fits',\
											  inplace=False, transpose=False)
			read_noise_models[fieldidx] = noise_cl2d

	B_ells = [None for x in range(len(ifield_list))]

	if load_ptsrc_cib:

		if compute_beam_correction:


			for fieldidx, ifield in enumerate(ifield_list):

				lb, B_ell, B_ell_list = cbps.compute_beam_correction_posts(ifield, inst, nbins=cbps.n_ps_bin, n_fine_bin=10, tail_path='data/psf_model_dict_updated_081121_ciber.npz')
				B_ells[fieldidx] = B_ell

		else:
			B_ells = np.load(bls_fpath)['B_ells']

		if bls_fpath is not None and compute_beam_correction:
			np.savez(bls_fpath, B_ells=B_ells, ifield_list=ifield_list)

	# loop through simulations
	for i in np.arange(simidx0, nsims):
		verbose = False
		if i==0:
			verbose = True

		print('Simulation set '+str(i)+' of '+str(nsims)+'..')

		cib_setidxs = []
		inv_Mkks = []

		diff_realizations = np.zeros(field_set_shape)

		for fieldidx, ifield in enumerate(ifield_list):

			# ----------------- make DGL ----------------

			if same_dgl:
				diff_realization = generate_custom_dgl_clustering(cbps, dgl_scale_fac=dgl_scale_fac, gen_ifield=indiv_ifield)
			else:
				diff_realization = generate_custom_dgl_clustering(cbps, dgl_scale_fac=dgl_scale_fac, ifield=ifield, gen_ifield=indiv_ifield)
				
			if load_ptsrc_cib:  

				diff_realizations[fieldidx] = diff_realization

				if draw_cib_setidxs:
					cib_setidx = np.random.choice(np.arange(n_cib_set))
					print('cib set idxs is ', cib_setidxs)
				else:
					cib_setidx = i

				cib_setidxs.append(cib_setidx)

				test_set_fpath = base_path+'TM'+str(inst)+'/cib_with_tracer_5field_set'+str(cib_setidx)+'_'+datestr+'_TM'+str(inst)+'.fits'
				cib_realiz = fits.open(test_set_fpath)['map_'+str(ifield)].data.transpose()
				# cib_realiz = fits.open(test_set_fpath)['map'+str(ifield)].data # earlier versions didnt have underscore

				if apply_mask:
					joint_maskos[fieldidx] = fits.open(base_path+'TM'+str(inst)+'/masks/ff_joint_masks/joint_mask_with_ffmask_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_abc110821.fits')['joint_mask_'+str(ifield)].data
					inv_Mkks.append(fits.open(base_path+'TM'+str(inst)+'/mkk/ff_joint_masks/mkk_estimate_with_ffmask_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_abc110821.fits')['inv_Mkk_'+str(ifield)].data)

				if aug_rotate: # augment sims with random rotations (0, 90, 180, 270)
					nrot = np.random.choice([0, 1, 2, 3])
					for nr in range(nrot):
						cib_realiz = np.rot90(cib_realiz)

						if apply_mask:
							joint_maskos[fieldidx] = np.rot90(joint_maskos[fieldidx])

	
			else:
				_, _, shotcomp = generate_diffuse_realization(cbps.dimx, cbps.dimy, power_law_idx=0.0, scale_fac=3e8)

			clus_realizations[fieldidx] = diff_realization

			if load_ptsrc_cib:
				clus_realizations[fieldidx] += cib_realiz
			else:
				clus_realizations[fieldidx] += shotcomp

			# ------------------- make zl ------------------

			zl_perfield[fieldidx] = generate_zl_realization(zl_levels[fieldidx], apply_zl_gradient, theta_mag=theta_mag, dimx=cbps.dimx, dimy=cbps.dimy)

			if load_trilegal:
				mock_trilegal_path = base_path+'trilegal/mock_trilegal_simidx'+str(cib_setidx)+'_'+datestr+'.fits'
				mock_trilegal = fits.open(mock_trilegal_path)
				mock_trilegal_map = mock_trilegal['trilegal_'+str(cbps.inst_to_band[inst])+'_'+str(ifield)].data
				clus_realizations[fieldidx] += mock_trilegal_map.transpose() # transpose is because astronomical mask has x, y flipped, same for Helgason galaxies


		print('cib set idx:', cib_setidxs)

		# ----------- load masks and mkk matrices ------------
		
		if not load_ptsrc_cib and apply_mask: # if applying mask load masks and inverse Mkk matrices
			inv_Mkks = []
			for fieldidx, ifield in enumerate(ifield_list):

				maskidx = i%10
				
				joint_maskos[fieldidx] = fits.open(base_path+'TM'+str(inst)+'/masks/ff_joint_masks/joint_mask_with_ffmask_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(maskidx)+'_abc110821.fits')['joint_mask_'+str(ifield)].data
				inv_Mkks.append(fits.open(base_path+'TM'+str(inst)+'/mkk/ff_joint_masks/mkk_estimate_with_ffmask_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(maskidx)+'_abc110821.fits')['inv_Mkk_'+str(ifield)].data)
		
		# B_ells = [None for x in range(len(ifield_list))]

		# if load_ptsrc_cib:
		# 	for fieldidx, ifield in enumerate(ifield_list):

		# 		lb, B_ell, B_ell_list = cbps.compute_beam_correction_posts(ifield, inst, nbins=cbps.n_ps_bin, n_fine_bin=10, tail_path='data/psf_model_dict_updated_081121_ciber.npz')
		# 		B_ells[fieldidx] = B_ell

		# ----------------------- generate noise for ZL/EBL realizations --------------------------------------
		
		zl_noise = np.zeros(field_set_shape)

		if with_photon_noise:
			# zl_noise = np.zeros(field_set_shape)

			if i==0 and save_ff_ests:
				zl_noise2 = np.zeros(field_set_shape)


			for fieldidx, ifield in enumerate(ifield_list):

				# shot_sigma_sb_zl_perf = cbps.compute_shot_sigma_map(inst, zl_perfield[fieldidx], nfr=nfr_fields[fieldidx])
				# zl_noise[fieldidx] = shot_sigma_sb_zl_perf*np.random.normal(0, 1, size=(cbps.dimx, cbps.dimy))

				shot_sigma_sb_sky_perf = cbps.compute_shot_sigma_map(inst, zl_perfield[fieldidx]+clus_realizations[fieldidx], nfr=nfr_fields[fieldidx])
				zl_noise[fieldidx] = shot_sigma_sb_sky_perf*np.random.normal(0, 1, size=(cbps.dimx, cbps.dimy))

		# ------------------------------------ add read noise ------------------------------------------------
		if with_read_noise:
			for fieldidx, ifield in enumerate(ifield_list):

				read_noise_indiv, _ = cbps.noise_model_realization(inst, (cbps.dimx, cbps.dimy), read_noise_models[fieldidx], \
												read_noise=True, photon_noise=False)
			
				read_noise[fieldidx] = read_noise_indiv


		# --------------------- add components together to get observed images -------------------------------

		observed_ims = zl_perfield + zl_noise + clus_realizations

		obs_levels = np.array([np.mean(obs) for obs in observed_ims])
		print('OBS LEVELS HERE ARE ', obs_levels)


		# multiply signal by flat field and add read noise on top
			
		if apply_smooth_FF:
			observed_ims *= smooth_ff
			print('applied smooth flat field to observed ims..')
		if with_read_noise:
			print('adding read noise')
			observed_ims += read_noise


		nls_estFF_nofluc = []

		# make monte carlo FF realizations that are used for the noise realizations (how many do we ultimately need?)
		if i==0 and save_ff_ests:

			ff_realization_estimates = np.zeros(field_set_shape)
			observed_ims_nofluc = np.zeros(field_set_shape)

			for ffidx in range(nmc_ff):

				# ---------------------------- add ZL -------------------------

				for fieldidx, ifield in enumerate(ifield_list):

					zl_realiz = generate_zl_realization(zl_levels[fieldidx], False, dimx=cbps.dimx, dimy=cbps.dimy)

					if with_photon_noise:

						shot_sigma_sb_zl_perf = cbps.compute_shot_sigma_map(inst, zl_perfield[fieldidx], nfr=nfr_fields[fieldidx])
						zl_realiz += shot_sigma_sb_zl_perf*np.random.normal(0, 1, size=(cbps.dimx, cbps.dimy))
			

					observed_ims_nofluc[fieldidx] = zl_realiz

				# apply FF and read noise
				if apply_smooth_FF:
					observed_ims_nofluc *= smooth_ff

				if with_read_noise:

					for fieldidx, ifield in enumerate(ifield_list):

						read_noise_indiv, _ = cbps.noise_model_realization(inst, (cbps.dimx, cbps.dimy), read_noise_models[fieldidx], \
												read_noise=True, photon_noise=False)
			
						observed_ims_nofluc[fieldidx] += read_noise_indiv


				if iterate_grad_ff:

					weights_ff_nofluc = None
					if use_ff_weights:
						weights_ff_nofluc = weights_photonly

					print('RUNNING ITERATE GRAD FF')
					processed_ims, ff_realization_estimates, final_planes, _, coeffs_vs_niter, _ = iterative_gradient_ff_solve(observed_ims_nofluc, niter=niter, masks=joint_maskos, \
																							inv_Mkks=inv_Mkks, compute_ps=False, weights_ff=weights_ff_nofluc)
				
				else:
					for imidx, obs_nf in enumerate(observed_ims_nofluc):
							
						if apply_mask:
							stack_mask = list(joint_maskos.copy().astype(np.bool))
							target_mask = joint_maskos[imidx]
							del(stack_mask[imidx])
							target_invMkk = inv_Mkks[imidx]
						else:
							stack_mask, target_mask = None, None
							target_invMkk = None

						stack_obs_nofluc = list(observed_ims_nofluc.copy())
						del(stack_obs_nofluc[imidx])
						
						if use_ff_weights:
							weights_ff = list(weights_photonly.copy())
							del(weights_ff[imidx])
							print('weights here are ', weights_ff)
						else:
							weights_ff = None
						

						ff_estimate_nofluc, _, ff_weights_nofluc = compute_stack_ff_estimate(stack_obs_nofluc, target_mask=target_mask, masks=stack_mask, \
																   inv_var_weight=False, ff_stack_min=1, \
																		field_nfrs=nfr_fields, weights=weights_ff)
						

						ff_realization_estimates[imidx] = ff_estimate_nofluc

				np.savez(ciber_mock_fpath+'030122/ff_realizations/'+run_name+'/ff_realization_estimate_ffidx'+str(ffidx)+'.npz', \
						ff_realization_estimates = ff_realization_estimates)


		if iterate_grad_ff:
			weights_ff = None
			if use_ff_weights:
				weights_ff = weights_photonly

			print('RUNNING ITERATE GRAD FF for observed images')
			processed_ims, ff_estimates, final_planes, _, coeffs_vs_niter, _ = iterative_gradient_ff_solve(observed_ims, niter=niter, masks=joint_maskos, \
																					inv_Mkks=inv_Mkks, compute_ps=False, weights_ff=weights_ff)
		

			observed_ims = processed_ims.copy()

		for fieldidx, obs in enumerate(observed_ims):
		
			print('fieldidx = ', fieldidx)

			if apply_mask:
				target_mask = joint_maskos[fieldidx]
				target_invMkk = inv_Mkks[fieldidx]
			else:
				stack_mask, target_mask = None, None
				target_invMkk = None
				

				
			if not iterate_grad_ff: # then look at stacks

				if apply_mask:
					stack_mask = list(joint_maskos.copy().astype(np.bool))
					del(stack_mask[fieldidx])

				stack_obs = list(observed_ims.copy()) 
				del(stack_obs[fieldidx])

				print('not using iterative FF here')
				if use_ff_weights:
					weights_ff = list(weights_photonly.copy())
					del(weights_ff[fieldidx])
					print('weights here are ', weights_ff)
				else:
					weights_ff = None

				ff_estimate, _, ff_weights = compute_stack_ff_estimate(stack_obs, target_mask=target_mask, masks=stack_mask, \
																	   inv_var_weight=False, ff_stack_min=1, \
																			field_nfrs=nfr_fields, weights=weights_ff)

				ff_estimates[fieldidx] = ff_estimate.copy()



			if plot_ff_error:
				plot_map(ff_estimate, title='ff estimate i='+str(i))
				if apply_smooth_FF:
					plot_map(ff_estimate_nofluc, title='ff estimate no fluc')
					if apply_mask:
						plot_map(target_mask*(ff_estimate-smooth_ff), title='flat field error (masked)')
					else:
						plot_map(ff_estimate-smooth_ff, title='flat field error (unmasked)')
						
			if apply_mask:
				simmap_dc = np.mean(obs[target_mask==1])
			else:
				simmap_dc = np.mean(obs)

			# noise bias from sims with no fluctuations    
			print('and nowwww here on i = ', i)


			# if noise model has already been saved just load it
			if i > 0:
				print('LOADING NOISE MODEL FROM FILE..')
				noisemodl_file = np.load(ciber_mock_fpath+'030122/noise_models_sim/'+run_name+'/noise_model_fieldidx'+str(fieldidx)+'.npz')
				fourier_weights_nofluc = noisemodl_file['fourier_weights_nofluc']
				mean_cl2d_nofluc = noisemodl_file['mean_cl2d_nofluc']
				nl_estFF_nofluc = noisemodl_file['nl_estFF_nofluc']

			else:
				all_ff_ests_nofluc = []
				for ffidx in range(nmc_ff):
					
					ff_file = np.load(ciber_mock_fpath+'030122/ff_realizations/'+run_name+'/ff_realization_estimate_ffidx'+str(ffidx)+'.npz')
					all_ff_ests_nofluc.append(ff_file['ff_realization_estimates'][fieldidx])
					
					if ffidx==0:
						plot_map(all_ff_ests_nofluc[0], title='loaded MC ff estimate')
				
				print('median obs is ', np.median(obs))
		
				shot_sigma_sb_zl_noisemodl = cbps.compute_shot_sigma_map(inst, zl_perfield[fieldidx], nfr=nfr_fields[fieldidx])
		
				fourier_weights_nofluc, mean_cl2d_nofluc = cbps.estimate_noise_power_spectrum(nsims=n_FW_sims, n_split=n_FW_split, apply_mask=apply_mask, \
				   noise_model=read_noise_models[fieldidx], read_noise=with_read_noise, inst=inst, ifield=ifield_noise_list[fieldidx], show=False, mask=target_mask, \
					photon_noise=with_photon_noise, ff_estimate=None, shot_sigma_sb=shot_sigma_sb_zl_noisemodl, inplace=False, \
					field_nfr=nfr_fields[fieldidx], simmap_dc=simmap_dc, ff_truth=smooth_ff, mc_ff_estimates=all_ff_ests_nofluc, gradient_filter=gradient_filter)

				nl_estFF_nofluc = cbps.compute_noise_power_spectrum(inst, noise_Cl2D=mean_cl2d_nofluc.copy(), inplace=False, apply_FW=apply_FW, weights=fourier_weights_nofluc)
			
				print('nl_estFF_nofluc = ', nl_estFF_nofluc)

				if save:
					print('SAVING NOISE MODEL FILE..')
					np.savez('/Volumes/Seagate Backup Plus Drive/Toolkit/Mirror/Richard/ciber_mocks/030122/noise_models_sim/'+run_name+'/noise_model_fieldidx'+str(fieldidx)+'.npz', \
							fourier_weights_nofluc=fourier_weights_nofluc, mean_cl2d_nofluc=mean_cl2d_nofluc, nl_estFF_nofluc=nl_estFF_nofluc)
			

			cbps.FW_image=fourier_weights_nofluc.copy()
			nls_estFF_nofluc.append(nl_estFF_nofluc)


			if load_ptsrc_cib:
				beam_correct = True 
			else:
				beam_correct = False			

			FF_correct = True
			if iterate_grad_ff: # set gradient_filter to False here because we've already done gradient filtering
				gradient_filter = False
				FF_correct = False

			lb, processed_ps_nf, cl_proc_err, _ = cbps.compute_processed_power_spectrum(inst, apply_mask=apply_mask, \
											 mask=target_mask, image=obs, convert_adufr_sb=False, \
											mkk_correct=apply_mask, inv_Mkk=target_invMkk, beam_correct=beam_correct, B_ell=B_ells[fieldidx], \
											apply_FW=apply_FW, verbose=False, noise_debias=True, \
										 FF_correct=FF_correct, FF_image=ff_estimates[fieldidx],\
											 gradient_filter=gradient_filter, save_intermediate_cls=True, N_ell=nl_estFF_nofluc) # ff bias


			if iterate_grad_ff: # .. but we want gradient filtering still for the ground truth signal.
				gradient_filter = True

			cl_prenlcorr = cbps.masked_Cl_pre_Nl_correct.copy()

			# if using CIB model, ground truth for CIB is loaded from Helgason J > Jlim and added to ground truth of DGL-like component
			if load_ptsrc_cib:

				lb, diff_cl, cl_proc_err, _ = cbps.compute_processed_power_spectrum(inst, apply_mask=False, \
													 image=diff_realizations[fieldidx], convert_adufr_sb=False, \
													mkk_correct=False, beam_correct=True, B_ell=B_ells[fieldidx], \
													apply_FW=False, verbose=False, noise_debias=False, \
												 FF_correct=False, save_intermediate_cls=True, gradient_filter=gradient_filter)


				cib_cl_file = fits.open(base_path+'TM'+str(inst)+'/powerspec/cls_cib_vs_maglim_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(cib_setidxs[fieldidx])+'.fits')

				unmasked_cib_cl = cib_cl_file['cls_cib'].data['cl_maglim_'+str(masking_maglim)]

				# print('unmasked cib cl', unmasked_cib_cl)

				ebl_ps = unmasked_cib_cl + diff_cl

			else:


				lb, ebl_ps, cl_proc_err, _ = cbps.compute_processed_power_spectrum(inst, apply_mask=False, \
													 image=clus_realizations[fieldidx], convert_adufr_sb=False, \
													mkk_correct=False, beam_correct=True, B_ell=B_ells[fieldidx], \
													apply_FW=False, verbose=False, noise_debias=False, \
												 FF_correct=False, save_intermediate_cls=True, gradient_filter=gradient_filter)



			if ff_bias_correct:
				if ff_biases is None:
					print('no ff biases provided...')
				else:
					print('corrected by factor ', ff_biases[fieldidx])
					processed_ps_nf /= ff_biases[fieldidx]

			if transfer_function_correct:
				processed_ps_nf /= t_ell


			signal_power_spectra[i, fieldidx, :] = ebl_ps
			recovered_power_spectra[i, fieldidx, :] = processed_ps_nf

			print('proc:', processed_ps_nf)
			print('ebl ps', ebl_ps)
			print('proc (nofluc) / ebl = ', processed_ps_nf/ebl_ps)


		print('mean ps bias is ', np.mean(recovered_power_spectra[i,:,:]/signal_power_spectra[i,:,:], axis=0))
	   
		nls_estFF_nofluc = np.array(nls_estFF_nofluc)
		
		print('mean nl:', np.mean(nls_estFF_nofluc, axis=0))


		if save:
			np.savez(sim_test_fpath+run_name+'/input_recovered_ps_estFF_simidx'+str(i)+'.npz', lb=lb, \
				signal_ps=signal_power_spectra[i], recovered_ps_est_nofluc=recovered_power_spectra[i],\
					 dgl_scale_fac=dgl_scale_fac, apply_mask = apply_mask,\
					 with_read_noise = with_read_noise, apply_FW = apply_FW, apply_smooth_FF = apply_smooth_FF,\
					 same_zl_levels = same_zl_levels,apply_zl_gradient = apply_zl_gradient,\
					 gradient_filter = gradient_filter, same_int = same_int,\
					 same_dgl = same_dgl, use_ff_weights = use_ff_weights, niter=niter, iterate_grad_ff=iterate_grad_ff)

		
	return lb, signal_power_spectra, recovered_power_spectra






