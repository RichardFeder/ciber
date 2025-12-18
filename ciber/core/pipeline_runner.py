import numpy as np
import astropy.io.fits as fits
import astropy.wcs as wcs
import pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import os

# import scipy.io
# from scipy.signal import fftconvolve
# import matplotlib
# import pickle
# from astropy.convolution import convolve
# from astropy.convolution import Gaussian2DKernel
# from astropy.wcs import WCS
# from scipy.io import loadmat

from ciber.core.powerspec_utils import *
from ciber.core.powerspec_pipeline import *
# from ciber.core.powerspec_pipeline import CIBER_PS_pipeline, iterative_gradient_ff_solve, lin_interp_powerspec,\
									 # grab_all_simidx_dat, grab_recovered_cl_dat, generate_synthetic_mock_test_set,\
									 # instantiate_dat_arrays_fftest, instantiate_cl_arrays_fftest, calculate_powerspec_quantities, compute_powerspectra_realdat
# from cross_spectrum_analysis import *
# from ciber_data_helpers import load_psf_params_dict
from ciber.plotting.plot_utils import plot_map
from ciber.plotting.galaxy_plots import *
from ciber.mocks.cib_mocks import *
from ciber.instrument.flat_field import *
from ciber.pseudo_cl.mkk_compute import compute_inverse_mkk

from ciber.masking.mask_utils import *
import config
from ciber.io.ciber_data_utils import *


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

def add_bootes_corner_mask(mask):
	corner_mask = np.ones_like(mask)
	corner_mask[900:, 0:200] = 0.
	mask *= corner_mask

	return mask


def downgrade_map_resolution(original_map, factor):
	"""
	Downgrades the resolution of a 2D map by an integer factor using averaging.

	Args:
		original_map (np.ndarray): The input 2D map (NumPy array).
		factor (int): The integer factor by which to downgrade the resolution.
					  Must be a positive integer and a divisor of both
					  the map's height and width.

	Returns:
		np.ndarray: The downgraded resolution map.

	Raises:
		ValueError: If factor is not a positive integer, or if the map
					dimensions are not divisible by the factor.
	"""
	if not isinstance(factor, int) or factor <= 0:
		raise ValueError("Factor must be a positive integer.")

	height, width = original_map.shape

	if height % factor != 0 or width % factor != 0:
		raise ValueError(
			f"Map dimensions ({height}x{width}) must be divisible by the factor ({factor})."
		)

	new_height = height // factor
	new_width = width // factor

	# Reshape the original map to group blocks of 'factor' x 'factor' pixels
	# The trick here is to add new axes to allow for efficient averaging.
	# For example, if factor=2:
	# (H, W) -> (H/2, 2, W/2, 2)
	# Then average over the 2nd and 4th axes.
	reshaped_map = original_map.reshape(new_height, factor, new_width, factor)

	# Calculate the mean over the blocks
	# downgraded_map = reshaped_map.mean(axis=(1, 3))
	downgraded_map = np.mean(reshaped_map, axis=(1, 3))


	return downgraded_map

def upscale_map_from_downgraded(downgraded_map, original_shape, factor):
	"""
	Upscales a downgraded map back to the original resolution, filling blocks
	with the corresponding averaged value.

	Args:
		downgraded_map (np.ndarray): The 2D map at lower resolution.
		original_shape (tuple): A tuple (height, width) representing the
								original resolution of the map.
		factor (int): The integer factor by which the map was downgraded.
					  Must be a positive integer and a divisor of both
					  the original_shape dimensions.

	Returns:
		np.ndarray: The upscaled map at the original resolution.

	Raises:
		ValueError: If factor is not a positive integer, or if the original
					dimensions are not divisible by the factor, or if the
					downgraded_map shape doesn't match expected after division.
	"""
	if not isinstance(factor, int) or factor <= 0:
		raise ValueError("Factor must be a positive integer.")

	original_height, original_width = original_shape
	downgraded_height, downgraded_width = downgraded_map.shape

	if original_height % factor != 0 or original_width % factor != 0:
		raise ValueError(
			f"Original map dimensions ({original_height}x{original_width}) "
			f"must be divisible by the factor ({factor})."
		)

	expected_downgraded_height = original_height // factor
	expected_downgraded_width = original_width // factor

	if downgraded_height != expected_downgraded_height or \
	   downgraded_width != expected_downgraded_width:
		raise ValueError(
			f"Downgraded map shape ({downgraded_height}x{downgraded_width}) "
			f"does not match expected shape ({expected_downgraded_height}x{expected_downgraded_width}) "
			f"for the given original_shape and factor."
		)

	# Use NumPy's Kronecker product (np.kron) for efficient upscaling
	# This function is perfect for repeating blocks of values.
	# It essentially multiplies each element of downgraded_map by a 'factor x factor'
	# block of ones.
	upscaled_map = np.kron(downgraded_map, np.ones((factor, factor)))

	return upscaled_map



def compute_overdensity_with_smoothing(
    gal_map, rand_map, mask, lb, pix_size_arcmin, sigma_pix=None
):
    """
    Compute a variance-friendly overdensity map with optional Gaussian smoothing.
    
    Parameters
    ----------
    gal_map : 2D array
        Galaxy counts map.
    rand_map : 2D array
        Random counts map (unscaled).
    mask : 2D boolean array
        True for unmasked pixels, False for masked.
    lb : 1D array
        ℓ-bin centers for bandpowers.
    pix_size_arcmin : float
        Pixel size in arcminutes.
    sigma_pix : float or None
        Gaussian smoothing sigma in pixels. If None, no smoothing is applied.
    
    Returns
    -------
    delta_g : 2D array
        Overdensity map (mean zero in unmasked region).
    B_ell : 1D array
        Beam transfer function for the given lb.
    """
    
    # Apply mask to counts
    gal_masked = np.where(mask, gal_map, 0.0)
    rand_masked = np.where(mask, rand_map, 0.0)
    
    # Normalization factor alpha
    alpha = gal_masked.sum() / rand_masked.sum()
    
    # Optional smoothing
    if sigma_pix is not None and sigma_pix > 0:
        gal_masked = gaussian_filter(gal_masked.astype(float), sigma=sigma_pix)
        rand_masked = gaussian_filter(rand_masked.astype(float), sigma=sigma_pix)
        
        # Compute beam transfer function
        pix_size_rad = (pix_size_arcmin / 60.0) * np.pi / 180.0
        sigma_rad = sigma_pix * pix_size_rad
        B_ell = np.exp(-0.5 * lb * (lb + 1) * sigma_rad**2)
    else:
        B_ell = np.ones_like(lb)
    
    # Overdensity calculation
    # Avoid division by zero in masked or empty pixels
    denom = alpha * rand_masked
    denom[denom == 0] = np.nan  # will become NaN in masked/empty pixels
    
    delta_g = (gal_masked - alpha * rand_masked) / denom
    
    # Set masked pixels to zero (or NaN if you prefer)
    delta_g[~mask] = 0.0
    
    return delta_g, B_ell
	
def ciber_gal_cross(inst_list, ifield_list_use, catname, addstr=None, randstr=None, mask_tail_list=None, masking_maglim_list=None, \
				   estimate_ciber_noise_gal=False, estimate_gal_noise_ciber=False, \
				   estimate_ciber_noise_gal_noise=False, ifield_list_full=[4, 5, 6, 7, 8], \
				   clip_sigma=5, niter=5, nitermax=5, per_quadrant=True, save=True, plot=False, \
				   nsims=500, n_split=10, fc_sub=False, fc_sub_quad_offset=True, fc_sub_n_terms=2, \
				   compute_cl_theta=True, cl_theta_cut=True, n_rad_bins=8, ell_min_wedge=2000, \
				   quadoff_grad=False, grad_sub=False, subtract_randoms=False, maskstr=None, \
				   rad_offset=None, theta0=np.pi, gal_downgrade_fac=None, apply_pixel_corr=True, \
				   rand_downgrade_fac=4,
				   include_ff_errors=True,
				   observed_run_name = 'observed_Jlt16.0_Hlt15.5_072424_quadoff_grad_fcsub_order2', 
				   tailstr_save=None):
	
	
	cbps = CIBER_PS_pipeline()
	
	bandstr_dict = dict({1:'J', 2:'H'})
	masking_maglim_dict = dict({1:17.5, 2:17.0})
	fieldidx_list_use = np.array(ifield_list_use)-4
	ciber_maps, masks = [np.zeros((len(ifield_list_full), cbps.dimx, cbps.dimy)) for x in range(2)]
	all_ps_save_fpath, all_addstr_use = [], []


	if randstr is None:

		randstr = addstr+'_random'
	
	if compute_cl_theta:
		if rad_offset is None:
			rad_offset = -np.pi/n_rad_bins
		theta_masks =  make_theta_masks(cbps.dimx, theta0=theta0, n_rad_bins=n_rad_bins, rad_offset=rad_offset, \
										plot=True, ell_min_wedge=ell_min_wedge)

	# if per_quadrant:
	# 	t_ell_file = np.load(config.ciber_basepath+'data/transfer_function/t_ell_masked_wdgl_2500_n=2p5_wgrad_quadrant_nsims=1000.npz')
	# else:
	t_ell_file = np.load(config.ciber_basepath+'data/transfer_function/t_ell_est_nsims=100.npz')
	t_ell = t_ell_file['t_ell_av']
	

	for idx, inst in enumerate(inst_list):
		
		all_cl_cross, all_cl_gal, all_clerr_cross, all_clerr_gal = [[] for x in range(4)]

		
		config_dict, pscb_dict, float_param_dict, fpath_dict = return_default_cbps_dicts()
		fpath_dict, list_of_dirpaths, base_path, trilegal_base_path = set_up_filepaths_cbps(fpath_dict, inst, 'test', '112022',\
																						datestr_trilegal='112022', data_type='observed', \
																					   save_fpaths=True)
		
		# CIBER beam
		bls_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/beam_correction/bl_est_postage_stamps_TM'+str(inst)+'_081121.npz'
		B_ells = np.load(bls_fpath)['B_ells_post']
		
		bandstr = bandstr_dict[inst]
		
		if masking_maglim_list is None:
			masking_maglim = masking_maglim_dict[inst]
		else:
			masking_maglim = masking_maglim_list[idx]
		


		# gal_densities, noise_base_path = load_delta_g_maps(catname, inst, addstr)
		gal_counts, _, noise_base_path = load_delta_g_maps(catname, inst, addstr)

		if subtract_randoms:
			rand_counts, _, _ = load_delta_g_maps(catname, inst, randstr)

		if mask_tail_list is not None:
			mask_tail = mask_tail_list[idx]
		else:
			mask_tail = 'maglim_'+bandstr+'_Vega_'+str(masking_maglim)+'_111323_ukdebias'


		print('Loading from mask tail = ', mask_tail)

		dc_template = cbps.load_dark_current_template(inst, verbose=True, inplace=False)

		# if catname=='WISE':

		# 	wise_masks = np.load('data/unWISE_coadds/unwise_masks/ciber_regrid_wise_bitmasks_all_TM'+str(inst)+'.npz')['wise_mask_regrid']
		
		for fieldidx, ifield in enumerate(ifield_list_full):
			flight_im = cbps.load_flight_image(ifield, inst, inplace=False)
			flight_im *= cbps.cal_facs[inst]
			flight_im -= dc_template*cbps.cal_facs[inst]
			mask_fpath = fpath_dict['mask_base_path']+'/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'
			mask = fits.open(mask_fpath)[1].data   

			if catname=='HSC':
				
				# hscmask = np.load('data/20250731_manualmasks_hsc_swire/TM'+str(inst)+'/hsc_mask_TM'+str(inst)+'_073125.npz')['hscmask']
				# mask *= hscmask

				if inst==1:
					mask[-120:, -250:] = 0.
					mask[:120, -250:] = 0.
				elif inst==2:
					mask[:250, :120] = 0.
					mask[:250, -120:] = 0. 

			# if catname=='WISE':
			# 	print('Multiplying by WISE bit mask')
			# 	mask *= wise_masks[fieldidx]

			sigclip = iter_sigma_clip_mask(flight_im, sig=clip_sigma, nitermax=nitermax, mask=mask)

			mask *= sigclip
			
			ciber_maps[fieldidx] = flight_im
			masks[fieldidx] = mask
			
			
		if per_quadrant:
			processed_ciber_maps, ff_estimates, masks = process_ciber_maps_perquad(cbps, ifield_list_full, inst, ciber_maps, masks, clip_sigma=clip_sigma,\
																		 nitermax=nitermax)
		else:
			
			_, stack_masks = stack_masks_ffest(masks, 1)
			masks *= stack_masks
			
			mean_norms = [cbps.zl_levels_ciber_fields[inst][cbps.ciber_field_dict[ifield]] for ifield in ifield_list_full]
			ff_weights = cbps.compute_ff_weights(inst, mean_norms, ifield_list_full, photon_noise=True)
			
			print('ff weights:', ff_weights)
			print('fcsub:', fc_sub, fc_sub_quad_offset, quadoff_grad)
			
			processed_ciber_maps, ff_estimates, final_planes, stack_masks, all_coeffs, all_sub_comp = iterative_gradient_ff_solve(ciber_maps, niter=1, masks=masks, weights_ff=ff_weights, \
																	ff_stack_min=1, plot=False, \
																	fc_sub=fc_sub, fc_sub_quad_offset=fc_sub_quad_offset, fc_sub_n_terms=fc_sub_n_terms, \
																	return_subcomp=True, quadoff_grad=quadoff_grad) # masks already accounting for ff_stack_min previously

			

		for fidx, ifield in enumerate(ifield_list_use):
			
			fieldidx = fieldidx_list_use[fidx]
			
			if fc_sub:
				mkk_type = 'quadoff_grad_fcsub_order'+str(fc_sub_n_terms)+'_estimate'
				Mkk = fits.open(config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/mkk/'+mask_tail+'/mkk_'+mkk_type+'_ifield'+str(ifield)+'_observed_'+mask_tail+'.fits')['Mkk_'+str(ifield)].data 
				truncated_mkk_matrix, inv_Mkk_truncated = truncate_invert_mkkmat(Mkk)

			else:
				if quadoff_grad:
					mkk_type = 'quadoff_grad'
				else:
					mkk_type = 'maskonly'

				mkkonly_savepath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/mkk/'+mask_tail+'/mkk_'+mkk_type+'_estimate_ifield'+str(ifield)+'_observed_'+mask_tail+'.fits'
				
				# mkk for CIBER + HSC mask
				# if catname=='HSC':
					# mkkonly_savepath.replace('.fits', '_hsc.fits')

				inv_Mkk = fits.open(mkkonly_savepath)['inv_Mkk_'+str(ifield)].data

			gal_map = gal_counts['ifield'+str(ifield)].data.transpose()

			if gal_downgrade_fac is not None:
				gal_map_shape = gal_map.shape
				gal_map = downgrade_map_resolution(gal_map, gal_downgrade_fac)
				gal_map = upscale_map_from_downgraded(gal_map, gal_map_shape, gal_downgrade_fac)

			if plot:
				plot_map(processed_ciber_maps[fieldidx]*masks[fieldidx], figsize=(6, 6))
				plot_map(gal_map*masks[fieldidx], figsize=(6, 6), title='galaxy map')



			if subtract_randoms:
				print('Subtracting randoms...')
				rand_map = rand_counts['ifield'+str(ifield)].data.transpose()

				rand_map = downgrade_map_resolution(rand_map, rand_downgrade_fac)
				rand_map = upscale_map_from_downgraded(rand_map, (1024, 1024), rand_downgrade_fac)

				# rand_map = gaussian_filter(rand_map.astype(float), sigma=10)

				gal_sum = gal_map[masks[fieldidx]!=0].sum()
				rand_sum = rand_map[masks[fieldidx]!=0].sum()
				scale = gal_sum / rand_sum

				print('scale is ', scale)

				plot_map(scale*rand_map, figsize=(6, 6), title='alpha*rand_map')

				# Subtract scaled randoms from galaxy counts

				masked_gal_map = np.zeros_like(gal_map, dtype=float)
				masked_gal_map[masks[fieldidx]!=0] = gal_map[masks[fieldidx]!=0] - scale * rand_map[masks[fieldidx]!=0]
				masked_gal_map /= np.mean(scale * rand_map[masks[fieldidx]!=0])


				# masked_gal_map, B_ell_smooth = compute

			else:
				print('not subtracting randoms...')
				masked_gal_map = gal_map*masks[fieldidx] # gal_densities fits file is actually galaxy counts
				meandens = np.mean(masked_gal_map[masks[fieldidx]!= 0])

				masked_gal_map[masks[fieldidx]!= 0] -= meandens
				masked_gal_map[masks[fieldidx]!= 0] /= meandens   

			masked_gal_map *= masks[fieldidx]

			if plot:
				plt.figure()
				plt.hist((masked_gal_map*masks[fieldidx]).ravel(), bins=50)
				plt.yscale('log')
				plt.xlabel('$\\delta_g$', fontsize=14)
				plt.show()
				plot_map(masked_gal_map, figsize=(6, 6), title='gal before filtering, CIBER resolution')
				plot_map(gaussian_filter(masked_gal_map, 2), figsize=(6,6), title='gal before filtering')

			# gradient filter for galaxies
			dot1, X, mask_rav = precomp_filter_general(cbps.dimx, cbps.dimy, mask=masks[fieldidx], gradient_filter=True, quadoff_grad=False, fc_sub=False, fc_sub_quad_offset=False, fc_sub_n_terms=fc_sub_n_terms, fc_sub_with_gradient=False)
			theta, filter_comp = apply_filter_to_map_precomp(masked_gal_map, dot1, X, mask_rav=mask_rav)
			masked_gal_map -= filter_comp
			masked_gal_map *= masks[fieldidx]

			if plot:
				plot_map(gaussian_filter(masked_gal_map*masks[fieldidx], 1), figsize=(6,6), title='gal after filtering')

			masked_ciber_map = processed_ciber_maps[fieldidx]*masks[fieldidx]
			masked_ciber_map[masks[fieldidx]!= 0] -= np.mean(masked_ciber_map[masks[fieldidx]!= 0])

			if plot:
				plot_map(masked_ciber_map, figsize=(6,6), cmap='bwr', title='masked ciber mask after mean subtraction')
				plot_map(masked_gal_map, figsize=(6,6), cmap='bwr')
				
			if estimate_ciber_noise_gal:
				nl_save_fpath, lb, all_nl1ds = estimate_ciber_noise_cross_gal(cbps, inst, ifield, catname, masked_gal_map, mask, \
																			 include_ff_errors=include_ff_errors, add_str=addstr, plot=False, \
																			 nsims=nsims, n_split=n_split, observed_run_name=observed_run_name)
			else:
				nl_save_fpath = noise_base_path+'nl1ds_TM'+str(inst)+'_ifield'+str(ifield)+'_ciber_noise'
				if addstr is not None:
					nl_save_fpath += '_'+addstr

				nl_save_fpath += '.npz'
				all_nl1ds = np.load(nl_save_fpath)['all_nl1ds_cross_gal']
				
			if pscb_dict['compute_cl_theta'] and pscb_dict['cut_cl_theta']:
				weights = np.ones_like(masked_ciber_map)
				for which_exclude in [0, n_rad_bins//2]:
					weights[theta_masks[which_exclude]==1] = 0.
			else:
				weights = np.ones_like(masked_ciber_map)




			lb, clproc, clerr_raw = get_power_spec(masked_ciber_map, map_b=masked_gal_map, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell, weights=weights)
			lb, clproc_gal, clerr_raw_gal = get_power_spec(masked_gal_map, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell, weights=weights)

			if fc_sub and fc_sub_n_terms==2:
				clproc[2:] = np.dot(inv_Mkk_truncated.transpose(), clproc[2:])
				clproc_gal[2:] = np.dot(inv_Mkk_truncated.transpose(), clproc_gal[2:])
			else:
				clproc = np.dot(inv_Mkk.transpose(), clproc)
				clproc_gal = np.dot(inv_Mkk.transpose(), clproc_gal)

			
			if apply_pixel_corr:
				pix_res_arcsec = 7. 
				if gal_downgrade_fac is not None:
					pix_res_arcsec *= gal_downgrade_fac
				wp_ell = get_pixel_window_function(lb, pix_res_arcsec)

				plt.figure()
				plt.plot(lb, wp_ell)
				plt.yscale('log')
				plt.xscale('log')
				plt.xlabel('$\\ell$', fontsize=14)
				plt.ylabel('$W_p(\\ell)', fontsize=14)
				plt.grid(alpha=0.3)
				plt.show()

				clproc /= np.sqrt(wp_ell)
				clproc_gal /= wp_ell


			clproc /= B_ells[fieldidx]			
			clerr_raw /= B_ells[fieldidx]

			std_nl1ds = np.std(all_nl1ds, axis=0)
			std_nl1ds /= B_ells[fieldidx]
	
			all_cl_gal.append(clproc_gal)
			all_clerr_gal.append(clerr_raw_gal)
			
			all_cl_cross.append(clproc)
			all_clerr_cross.append(std_nl1ds)
	
		all_cl_cross, all_clerr_cross = np.array(all_cl_cross), np.array(all_clerr_cross)
		all_cl_gal, all_clerr_gal = np.array(all_cl_gal), np.array(all_clerr_gal)
		
		if save:

			if subtract_randoms:
				addstr_save = addstr+'_wrandsub'
			else:
				addstr_save = addstr

			if maskstr is not None:
				addstr_save += '_'+maskstr

			if include_ff_errors:
				addstr_save += '_wFFerr'

			if tailstr_save is not None:
				addstr_save += '_'+tailstr_save

			ps_save_fpath = save_ciber_gal_ps(inst, ifield_list_use, catname, lb, all_cl_gal, all_clerr_gal, all_cl_cross, all_clerr_cross, \
							  masking_maglim=masking_maglim, addstr=addstr_save)
		
			all_ps_save_fpath.append(ps_save_fpath)
			all_addstr_use.append(addstr_save)
			
	if save:  
		return all_ps_save_fpath, all_addstr_use


def ciber_difference_spectrum(cbps, fpath_dict, config_dict, inst, ifieldA, ifieldB, mask_tail, masking_maglim, mask=None, sigma_clip=False, nsig=5, \
							 per_quadrant=False, fc_sub=True, fc_sub_quad_offset=True, fc_sub_n_terms=2, fc_sub_with_gradient=True, \
							 n_sims_noise=500, n_split_noise=10):

	maglim_ff_dict = dict({1:17.5, 2:17.0})
	masking_maglim_ff = max(maglim_ff_dict[inst], masking_maglim)

	fieldidxA, fieldidxB = ifieldA-4, ifieldB-4
	read_noise_models = cbps.grab_noise_model_set([ifieldA, ifieldB], inst, noise_model_base_path=fpath_dict['read_noise_modl_base_path'], noise_modl_type=config_dict['noise_modl_type'])

	dc_template = cbps.load_dark_current_template(inst, verbose=True, inplace=False)

	mask_save_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/masks/'+mask_tail+'/joint_mask_ifield'+str(ifieldA)+'_ifield'+str(ifieldB)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'
	mask = fits.open(mask_save_fpath)['joint_mask_'+str(ifieldA)].data

	plot_map(mask, title='mask')
	
	if fc_sub:
		mkk_type = 'quadoff_grad_fcsub_order'+str(fc_sub_n_terms)
		mkk_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/mkk/'+mask_tail+'/mkk_'+mkk_type+'_ifield'+str(ifieldA)+'_ifield'+str(ifieldB)+'_observed_'+mask_tail+'.fits'
		
		Mkk = fits.open(mkk_fpath)['Mkk_'+str(ifieldA)].data
		
		inv_Mkk = np.zeros_like(Mkk)
		inv_Mkk = np.linalg.inv(Mkk[2:,2:])
		
		t_ell_av = None
		
	else:
		mkkonly_savepath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/mkk/'+mask_tail+'/mkk_maskonly_estimate_ifield'+str(ifieldA)+'_ifield'+str(ifieldB)+'_observed_'+mask_tail+'.fits'
		inv_Mkk = fits.open(mkkonly_savepath)['inv_Mkk_'+str(ifieldA)].data
		
		if per_quadrant:
			t_ell_fpath = config.ciber_basepath+'data/transfer_function/t_ell_masked_wdgl_2500_n=2p5_wgrad_quadrant_nsims=1000.npz'
		else:
			t_ell_fpath = config.ciber_basepath+'data/transfer_function/t_ell_est_nsims=100.npz'

		t_ell_av = np.load(t_ell_fpath)['t_ell_av']
		print('t_ell = ', t_ell_av)


	# load beams and average Bootes fields
	bls_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/beam_correction/bl_est_postage_stamps_TM'+str(inst)+'_081121.npz'

	B_ells = np.load(bls_fpath)['B_ells_post']
	B_ell_eff = np.sqrt(B_ells[fieldidxA]*B_ells[fieldidxB])


	ptsrcfile = fits.open(config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/point_src_maps_for_ffnoise/point_src_maps'+'_TM'+str(inst)+'_mmin='+str(masking_maglim)+'_mmax='+str(masking_maglim_ff)+'_merge.fits', overwrite=True)

	max_vals_ptsrc_A = ptsrcfile['ifield'+str(ifieldA)].header['maxval']
	max_vals_ptsrc_B = ptsrcfile['ifield'+str(ifieldB)].header['maxval']
	max_vals_ptsrc = max(max_vals_ptsrc_A, max_vals_ptsrc_B)

	print('max val ptsrc:', max_vals_ptsrc)


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
		mask = add_bootes_corner_mask(mask)

	plot_map(difference_im*mask)
	
	
	dot1, X, mask_rav = precomp_filter_general(cbps.dimx, cbps.dimy, mask=mask, fc_sub=fc_sub, fc_sub_quad_offset=fc_sub_quad_offset, \
						  fc_sub_n_terms=fc_sub_n_terms, fc_sub_with_gradient=fc_sub_with_gradient)
	
	
	theta, filter_comp = apply_filter_to_map_precomp(difference_im, dot1, X, mask_rav=mask_rav)
	
	difference_im -= mask*filter_comp

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


	# fourier_weights_diff, \
	# 		mean_nl2d_diff, nl1ds_diff = cbps.estimate_noise_power_spectrum(nsims=500, n_split=10, inst=inst, ifield=ifieldA, field_nfr=cbps.field_nfrs[ifieldA], mask=mask, \
	# 								  noise_model=read_noise_models[1], difference=True, shot_sigma_sb=mean_shot_sigma_sb, compute_1d_cl=True)

	noise_bias_dict = cbps.estimate_noise_power_spectrum(nsims=n_sims_noise, n_split=n_split_noise, inst=inst, ifield=ifieldA, field_nfr=cbps.field_nfrs[ifieldA], mask=mask, \
									  noise_model=read_noise_models[1], apply_filtering=True, difference=True, shot_sigma_sb=mean_shot_sigma_sb, compute_1d_cl=True, \
														fc_sub=fc_sub, fc_sub_quad_offset=fc_sub_quad_offset, fc_sub_n_terms=fc_sub_n_terms, with_gradient=fc_sub_with_gradient)

	fourier_weights_diff = noise_bias_dict['fourier_weights_diff']
	mean_nl2d_diff = noise_bias_dict['mean_nl2d_diff']
	nl1ds_diff = noise_bias_dict['nl1ds_diff']

	plot_map(fourier_weights_diff, title='fourier weights')
	plot_map(mean_nl2d_diff, title='nl2d')

	nl_dict = cbps.compute_noise_power_spectrum(inst, noise_Cl2D=mean_nl2d_diff.copy(), inplace=False, apply_FW=True, weights=fourier_weights_diff)
	# lb, N_ell_est, N_ell_err = cbps.compute_noise_power_spectrum(inst, noise_Cl2D=mean_nl2d_diff.copy(), inplace=False, apply_FW=True, weights=fourier_weights_diff)

	lb = nl_dict['lbins']
	N_ell_est = nl_dict['Cl_noise']
	N_ell_err = nl_dict['Clerr']


	lb, processed_ps_nf, cl_proc_err, _, _ = cbps.compute_processed_power_spectrum(inst, apply_mask=True, \
								 mask=mask, image=masked_difference, convert_adufr_sb=False, \
								mkk_correct=True, inv_Mkk=inv_Mkk, beam_correct=True, B_ell=B_ell_eff, \
								apply_FW=True, verbose=True, noise_debias=True, \
							 FF_correct=False, FW_image=fourier_weights_diff, \
								 gradient_filter=False, save_intermediate_cls=False, N_ell=N_ell_est, \
								 per_quadrant=per_quadrant, max_val_after_sub=max_vals_ptsrc, \
																			   fc_sub=fc_sub)

	if t_ell_av is not None:
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

def ciber_difference_cross_spectrum(cbps, inst, cross_inst, ifieldA, ifieldB, mask_tail, masking_maglim, mask=None, sigma_clip=False, nsig=5, \
								   per_quadrant=True):
	
	masking_maglim_ff = max(17.5, masking_maglim)
	fieldidxA, fieldidxB = ifieldA-4, ifieldB-4
	
	astr_dir = config.ciber_basepath+'data/'

	mask_save_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/masks/'+mask_tail+'/joint_mask_ifield'+str(ifieldA)+'_ifield'+str(ifieldB)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'
	mask = fits.open(mask_save_fpath)['joint_mask_'+str(ifieldA)].data
	
	plot_map(mask, title='mask')
	
	dc_template = cbps.load_dark_current_template(inst, verbose=True, inplace=False)
	dc_template_cross = cbps.load_dark_current_template(cross_inst, verbose=True, inplace=False)

	
	if per_quadrant:
		t_ell_fpath = config.ciber_basepath+'data/transfer_function/t_ell_masked_wdgl_2500_n=2p5_wgrad_quadrant_nsims=1000.npz'
	else:
		t_ell_fpath = config.ciber_basepath+'data/transfer_function/t_ell_est_nsims=100.npz'
		
	t_ell_av = np.load(t_ell_fpath)['t_ell_av']
	print('t_ell = ', t_ell_av)
	
	
	tl_regrid_fpath = config.ciber_basepath+'data/transfer_function/regrid/tl_regrid_tm2_to_tm1_order=2.npz'
	tl_regrid = np.load(tl_regrid_fpath)['tl_regrid']

	tl_regrid = np.sqrt(tl_regrid)
	t_ell_av *= tl_regrid
	
	
	mkkonly_savepath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/mkk/'+mask_tail+'/mkk_maskonly_estimate_ifield'+str(ifieldA)+'_ifield'+str(ifieldB)+'_observed_'+mask_tail+'.fits'
	inv_Mkk = fits.open(mkkonly_savepath)['inv_Mkk_'+str(ifieldA)].data
	
	
	# beams
	# load beams and average Bootes fields
	bls_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/beam_correction/bl_est_postage_stamps_TM'+str(inst)+'_081121.npz'
	B_ells = np.load(bls_fpath)['B_ells_post']
	B_ell_eff = np.sqrt(B_ells[fieldidxA]*B_ells[fieldidxB])
	
	bls_fpath_cross = config.ciber_basepath+'data/fluctuation_data/TM'+str(cross_inst)+'/beam_correction/bl_est_postage_stamps_TM'+str(cross_inst)+'_081121.npz'
	B_ells_cross = np.load(bls_fpath_cross)['B_ells_post']
	B_ell_eff_cross = np.sqrt(B_ells_cross[fieldidxA]*B_ells_cross[fieldidxB])
	
	
	B_ell_eff_tot = B_ell_eff*B_ell_eff_cross
	
	if per_quadrant:
		t_ell_fpath = config.ciber_basepath+'data/transfer_function/t_ell_masked_wdgl_2500_n=2p5_wgrad_quadrant_nsims=1000.npz'
	else:
		t_ell_fpath = config.ciber_basepath+'data/transfer_function/t_ell_est_nsims=100.npz'
		
	all_obs_im, all_obs_im_cross = [], []
	
	for fieldidx, ifield in enumerate([ifieldA, ifieldB]):

		cbps.load_flight_image(ifield, inst, verbose=True, ytmap=False) # default loads aducorr maps
		observed_im = cbps.image*cbps.cal_facs[inst]
		observed_im -= dc_template*cbps.cal_facs[inst]
		
		all_obs_im.append(observed_im)
		
		cbps.load_flight_image(ifield, cross_inst, verbose=True, ytmap=False) # default loads aducorr maps
		observed_im_cross = cbps.image*cbps.cal_facs[cross_inst]
		observed_im_cross -= dc_template_cross*cbps.cal_facs[cross_inst]
		
		corner_mask = np.ones_like(mask)
		corner_mask[900:, 0:200] = 0.
		
		observed_im_cross *= corner_mask
		
		cross_regrid, ciber_fp_regrid = regrid_arrays_by_quadrant(observed_im_cross, ifield, inst0=inst, inst1=cross_inst, \
													 plot=False, astr_dir=astr_dir, order=0, conserve_flux=False)


		
		all_obs_im_cross.append(cross_regrid)
		
	
	difference_im = all_obs_im[0]-all_obs_im[1]
	difference_im_cross = all_obs_im_cross[0]-all_obs_im_cross[1]
	
	plot_map(difference_im, title='difference im')
	plot_map(difference_im_cross, title='difference im cross reproj')
	plot_map(difference_im-difference_im_cross, title='diff')
	
	if sigma_clip:
		mask *= iter_sigma_clip_mask(difference_im, sig=nsig, nitermax=5, mask=mask)
		mask *= iter_sigma_clip_mask(difference_im_cross, sig=nsig, nitermax=5, mask=mask)
	
	mask *= (difference_im != 0)*(difference_im_cross != 0)
	
	difference_im *= mask
	difference_im_cross *= mask
	
	theta_quad, plane = fit_gradient_to_map(difference_im, mask=mask)
	difference_im -= plane

	theta_quad, plane = fit_gradient_to_map(difference_im_cross, mask=mask)
	difference_im_cross -= plane

	
#     for q in range(4):
#         theta_quad, plane_quad = fit_gradient_to_map(difference_im[cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]], mask=mask[cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]])
#         difference_im[cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]] -= plane_quad

	posmask = (difference_im != 0)*(difference_im_cross != 0)
	
	meansub = np.mean(difference_im[posmask])
	difference_im[posmask] -= meansub
	
	meansub_cross = np.mean(difference_im_cross[posmask])
	difference_im_cross[posmask] -= meansub_cross
	
	plot_map(difference_im, title='diff im masked')
	plot_map(difference_im_cross, title='diff im masked')
	
	lb, cl, clerr = get_power_spec(difference_im, map_b=difference_im_cross, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
	
	cl = np.dot(inv_Mkk.transpose(), cl)

	
	cl /= B_ell_eff_tot
	clerr /= B_ell_eff_tot
	cl /= t_ell_av
	clerr /= t_ell_av
	
	cl /= 2
	clerr /= 2
	
	pf = lb*(lb+1)/(2*np.pi)
	plt.figure()
	plt.errorbar(lb[:-1], (pf*cl)[:-1], yerr=(pf*clerr)[:-1], fmt='o', color='k')
	plt.xscale('log')
	plt.yscale('log')
	plt.ylim(1e-1, 1e4)
	plt.grid(alpha=0.5)
	plt.xlim(150, 1e5)
	plt.show()
		
	return lb, cl, clerr


def grab_sim_fpaths(inst, load_trilegal, load_ptsrc_cib, datestr_trilegal, datestr, cib_setidx, ciber_mock_fpath, fpath_dict):
	if load_trilegal:
		trilegal_fpath = ciber_mock_fpath+datestr_trilegal+'/trilegal/mock_trilegal_simidx'+str(cib_setidx)+'_'+datestr_trilegal+'.fits'
	else:
		trilegal_fpath = None 

	if load_ptsrc_cib:
		test_set_fpath = fpath_dict['ciber_mock_fpath']+datestr+'/TM'+str(inst)+'/cib_realiz/cib_with_tracer_with_dpoint_5field_set'+str(cib_setidx)+'_'+datestr+'_TM'+str(inst)+'.fits'
	else:
		test_set_fpath = None

	return trilegal_fpath, test_set_fpath

def grab_B_ell(fpath_dict, config_dict, pscb_dict, nfield):
	B_ells = [None for x in range(nfield)]
	B_ells_cross = [None for x in range(nfield)]

	if fpath_dict['bls_fpath'] is not None: # cross
		print('Loading B_ells from ', fpath_dict['bls_fpath'])
		B_ells = np.load(fpath_dict['bls_fpath'])['B_ells_post']

		if pscb_dict['compute_ps_per_quadrant']:
			print('loading B_ells quad from ', fpath_dict['bls_fpath_quad'])
			B_ells_quad = np.load(fpath_dict['bls_fpath_quad'])['B_ells_post']
	
	if config_dict['ps_type']=='cross' and fpath_dict['bls_fpath_cross'] is not None:
		print('Loading B_ells for cross from ', fpath_dict['bls_fpath_cross'])
		B_ells_cross = np.load(fpath_dict['bls_fpath_cross'])['B_ells_post']

	return B_ells, B_ells_cross

def grab_t_ell(config_dict, fpath_dict, float_param_dict, pscb_dict):
	if config_dict['ps_type']=='cross' and config_dict['cross_type']=='ciber':
		t_ell_fpath = fpath_dict['tls_base_path']+'t_ell_masked_wdgl_2500_n=2p5_wgrad_quadrant_nsims=1000.npz'
	else:
		if pscb_dict['quadoff_grad']:
			print("Loading 1D quadoff + grad transfer function for testing")
			t_ell_fpath = fpath_dict['tls_base_path']+'t_ell_quadoff_grad_nsims=500_n=3p0.npz'
		elif pscb_dict['fc_sub']:

			if pscb_dict['fc_sub_quad_offset']:
				t_ell_fpath = fpath_dict['tls_base_path']+'t_ell_quadoff_fcsub_nterms='+str(float_param_dict['fc_sub_n_terms'])+'_nsims=500_n=3p0.npz'
			else:
				t_ell_fpath = fpath_dict['tls_base_path']+'t_ell_fcsub_nterms='+str(float_param_dict['fc_sub_n_terms'])+'_nsims=500_n=3p0.npz'

		elif pscb_dict['per_quadrant']:
			t_ell_fpath = fpath_dict['tls_base_path']+'t_ell_masked_wdgl_2500_n=2p5_wgrad_quadrant_nsims=1000.npz'
		else:
			t_ell_fpath = fpath_dict['tls_base_path']+'t_ell_est_nsims='+str(float_param_dict['n_realiz_t_ell'])+'.npz'


	if t_ell_fpath is not None:
		t_ell_key = 't_ell_av'
		print('Loading transfer function from ', fpath_dict['t_ell_fpath'])
		t_ell_av = np.load(t_ell_fpath)[t_ell_key]

		if pscb_dict['compute_ps_per_quadrant']:
			print('loading transfer function for single quadrant from ', fpath_dict['t_ell_fpath_quad'])
			t_ell_av_quad = np.load(fpath_dict['t_ell_fpath_quad'])['t_ell_av_quad']
			if pscb_dict['verbose']:
				print('t_ell_av_quad:', t_ell_av_quad)

	else:
		print('No transfer function path provided, and compute_transfer_function set to False, exiting..')
		return 
		
	return t_ell_av, t_ell_fpath 

def load_proc_regrid_map(fpath_dict, float_param_dict, pscb_dict, regrid_mask_tail, inst, cross_inst, ifield):

	obs_cross_fpath = fpath_dict['proc_regrid_base_path']+'/'+regrid_mask_tail+'/proc_regrid_TM'+str(cross_inst)+'_to_TM'+str(inst)+'_ifield'+str(ifield)+'_'+regrid_mask_tail
	if float_param_dict['interp_order'] is not None:
		obs_cross_fpath += '_order='+str(float_param_dict['interp_order'])
	if pscb_dict['conserve_flux']:
		obs_cross_fpath += '_conserve_flux'

	if pscb_dict['fc_sub']:
		if pscb_dict['fc_sub_quad_offset']:
			obs_cross_fpath += '_quadoff_grad_fcsub_order'+str(float_param_dict['fc_sub_n_terms'])
		else:
			obs_cross_fpath += '_fcsub_order'+str(float_param_dict['fc_sub_n_terms'])
		# obs_cross_fpath += '_quadoff_grad_fcsub_order'+str(float_param_dict['fc_sub_n_terms'])

	obs_cross_fpath += '.fits'
	obs_cross_file = fits.open(obs_cross_fpath)
	obs_cross = obs_cross_file['proc_regrid_'+str(ifield)].data 

	obs_level_cross = obs_cross_file[0].header['obs_level']

	return obs_cross, obs_level_cross

def load_mkk_fcsub(float_param_dict, mode_couple_base_dir, mkk_mask_tail, mkk_type, inst, ifield, cib_setidx=None, invert_spliced_matrix=False, compute_pinv=False):
	print('opening fc sub with mask tail', mkk_mask_tail)
	if cib_setidx is not None:
		Mkk = fits.open(mode_couple_base_dir+'/'+mkk_mask_tail+'/mkk_'+mkk_type+'_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_'+mkk_mask_tail+'.fits')['Mkk_'+str(ifield)].data 
	else:
		Mkk = fits.open(mode_couple_base_dir+'/'+mkk_mask_tail+'/mkk_'+mkk_type+'_ifield'+str(ifield)+'_observed_'+mkk_mask_tail+'.fits')['Mkk_'+str(ifield)].data 

	if float_param_dict['fc_sub_n_terms']==2:
		if invert_spliced_matrix:
			print('Splicing and inverting mkk matrix..')

			truncated_mkk_matrix, inv_Mkk_truncated = truncate_invert_mkkmat(Mkk)
		elif compute_pinv:
			print('Computing pseudo-inverse, retaining original dimension..')

			print('second row of Mkk is ', Mkk[:,1])
			Mkk[:,1] = 0.
			inv_Mkk_pinv = np.linalg.pinv(Mkk)
			# inv_Mkk_pinv[0,0] = 0.

			return inv_Mkk_pinv

		else:
			inv_Mkk_truncated = np.linalg.inv(Mkk[2:,2:])
	elif float_param_dict['fc_sub_n_terms']==3:
		inv_Mkk_truncated = np.linalg.inv(Mkk[4:,4:])
	else:
		inv_Mkk_truncated = np.linalg.inv(Mkk)

	return inv_Mkk_truncated 

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
						'noise_modl_type':'quadsub_021523', 'dgl_mode':'sfd_clean', 'cmblens_mode':None, 'mkk_type':None, \
						'diff_noise_modl_type':'halfexp_full_051324'})

	pscb_dict = dict({'ff_estimate_correct':True, 'apply_mask':True, 'with_inst_noise':True, 'with_photon_noise':True, 'apply_FW':True, 'generate_diffuse_realization':True, \
					'add_ellm2_clus':False, 'apply_smooth_FF':True, 'compute_beam_correction':False, 'same_clus_levels':True, 'same_zl_levels':False, 'apply_zl_gradient':True, 'gradient_filter':False, \
					'iterate_grad_ff':True ,'mkk_ffest_hybrid':True, 'apply_sum_fluc_image_mask':False, 'same_int':False, 'same_dgl':True, 'use_ff_weights':True, 'plot_ff_error':False, 'load_ptsrc_cib':True, \
					'load_trilegal':True , 'subtract_subthresh_stars':False, 'ff_bias_correct':False, 'save_ff_ests':True, \
					 'draw_cib_setidxs':False, 'aug_rotate':False, 'noise_debias':True, 'load_noise_bias':False, 'transfer_function_correct':False, 'compute_transfer_function':False,\
					  'save_intermediate_cls':True, 'verbose':False, 'show_plots':False, 'show_ps':True, 'save':True, 'bl_post':False, 'ff_sigma_clip':False, \
					  'per_quadrant':False, 'use_dc_template':True, 'ff_estimate_cross':False, 'map_photon_noise':False, 'zl_photon_noise':True, \
					  'compute_ps_per_quadrant':False, 'apply_wen_cluster_mask':False, 'low_responsivity_blob_mask':False, \
					  'pt_src_ffnoise':False, 'shut_off_plots':False, 'max_val_clip':False, 'point_src_ffnoise':False, 'unsharp_mask':False, \
					  'apply_tl_regrid':True, 'conserve_flux':False, 'quadoff_grad':False, 'load_cib_fferr':False, 'load_isl_fferr':False, \
					  'save_fourier_planes':False, 'clip_clproc_premkk':False, 'read_noise_fw':False, 'diff_cl2d_fwclip':True, 'clip_outlier_norm_modes':False, \
					  'fc_sub':False, 'fc_sub_quad_offset':True, 'make_preprocess_file':False, 'estimate_cross_noise':False, 'invert_spliced_matrix':False, \
					  'compute_mkk_pinv':False, 'compute_cl_theta':False, 'cut_cl_theta':False})

	float_param_dict = dict({'ff_min':0.5, 'ff_max':2.0, 'clip_sigma':5,'clip_sigma_ff':5, 'ff_stack_min':1, 'nmc_ff':10, \
					  'theta_mag':0.01, 'niter':1, 'dgl_scale_fac':5, 'smooth_sigma':5, 'indiv_ifield':6,\
					  'nfr_same':25, 'J_bright_Bl':11, 'J_faint_Bl':17.5, 'n_FW_sims':500, 'n_FW_split':10, \
					  'ell_norm_blest':5000, 'n_realiz_t_ell':100, 'nitermax':10, 'n_cib_isl_sims':100, \
					  'unsharp_pct':95, 'unsharp_sigma':1.0, 'noise_modl_rescale_fac':None, 'interp_order':None, 'poly_order':1, 'fc_sub_n_terms':2, \
					  'weight_power':1.0, 'remove_outlier_fac':None, 'diff_cl2d_clipthresh':5, 'clip_norm_thresh':5, 'ff_min_nr':0.5, 'ff_max_nr':1.8, \
					  'n_rad_bins':8., 'rad_offset':-np.pi/8., 'ell_min_wedge':500})

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


def generate_ff_error_realizations_simp(cbps, nmc_ff, inst, ifield_list, joint_masks, obs_levels, weights_ff, shot_sigma_sb_maps=None, read_noise_models=None):

	maplist_shape = (len(ifield_list), cbps.dimx, cbps.dimy)
	ff_realization_estimates = np.zeros((nmc_ff, len(ifield_list), cbps.dimx, cbps.dimy))
	observed_ims_nofluc = np.zeros(maplist_shape)
	bandstr_dict = dict({1:'J', 2:'H'})
	bandstr = bandstr_dict[inst]

	for ffidx in range(nmc_ff):

		print('On set ', ffidx, 'of ', nmc_ff)
		for fieldidx, ifield in enumerate(ifield_list):
			zl_realiz = generate_zl_realization(obs_levels[fieldidx], False, dimx=cbps.dimx, dimy=cbps.dimy, theta_mag=None)


			if shot_sigma_sb_maps is not None:
				# this line below was a bug for a while, zl_perfield for observed is zero so it was adding zero photon noise to the realizations
				zl_realiz += shot_sigma_sb_maps[fieldidx]*np.random.normal(0, 1, size=cbps.map_shape) # temp
				observed_ims_nofluc[fieldidx] = zl_realiz

			if read_noise_models is not None: # true for observed
				read_noise_indiv, _ = cbps.noise_model_realization(inst, cbps.map_shape, read_noise_models[fieldidx], \
										read_noise=True, photon_noise=False, chisq=False)

				observed_ims_nofluc[fieldidx] += read_noise_indiv

		for fieldidx, image in enumerate(observed_ims_nofluc):

			stack_obs = list(observed_ims_nofluc.copy())                
			del(stack_obs[fieldidx])

			weights_ff_iter = None
		
			if weights_ff is not None:
				weights_ff_iter = weights_ff[(np.arange(len(observed_ims_nofluc)) != fieldidx)]
			
			# if masks_ffest is not None:
				# ff_estimate, _, ff_weights = compute_stack_ff_estimate(stack_obs, masks=masks_ffest[(np.arange(len(masks_ffest))!=imidx),:], weights=weights_ff_iter)

			ff_estimate, _, ff_weights = compute_stack_ff_estimate(stack_obs, masks=masks[(np.arange(len(masks))!=fieldidx),:], weights=weights_ff_iter)
			ff_realization_estimates[ffidx, fieldidx] = ff_estimate

	return ff_realization_estimates



def generate_ff_error_realizations(cbps, run_name, inst, ifield_list, joint_masks, simmaps_dc_all, shot_sigma_sb_zl, read_noise_models, weights_ff, \
									fpath_dict, pscb_dict, float_param_dict, obs_levels, datestr='111323', datestr_trilegal='112022', read_noise_fw=False, \
									apply_zl_gradient=False, theta_mag=0.01, save=True, return_realiz=False):

	maplist_shape = (len(ifield_list), cbps.dimx, cbps.dimy)
	ff_realization_estimates = np.zeros(maplist_shape)
	observed_ims_nofluc = np.zeros(maplist_shape)
	
	ptsrc_ff = False

	if pscb_dict['load_cib_fferr'] or pscb_dict['load_isl_fferr']:
		ptsrc_ff = True
		observed_ims_ptsrc = np.zeros(maplist_shape)
		ff_realization_ptsrc_estimates = np.zeros(maplist_shape)

	bandstr_dict = dict({1:'J', 2:'H'})
	bandstr = bandstr_dict[inst]

	for ffidx in range(float_param_dict['nmc_ff']):

		if pscb_dict['load_cib_fferr']:
			cib_dffs_basepath = config.ciber_basepath+'data/ciber_mocks/'+datestr+'/TM'+str(inst)+'/cib_dffs/'
			cib_dffs_fpath = cib_dffs_basepath+'/cib_dffs_default_5field_set'+str(ffidx)+'_'+datestr+'_TM'+str(inst)+'.fits'
			print('Loading CIB subthreshold from ', cib_dffs_fpath)
			cib_subthresh = fits.open(cib_dffs_fpath)

		if pscb_dict['load_isl_fferr']:
			isl_dffs_basepath = config.ciber_basepath+'data/ciber_mocks/'+datestr_trilegal+'/trilegal_dffs/'
			isl_dffs_fpath = isl_dffs_basepath+'/mock_trilegal_simidx'+str(ffidx)+'_'+datestr_trilegal+'.fits'
			print('Loading ISL subthreshold from ', isl_dffs_fpath)
			isl_subthresh = fits.open(isl_dffs_fpath)

		for fieldidx, ifield in enumerate(ifield_list):
			zl_realiz = generate_zl_realization(obs_levels[fieldidx], apply_zl_gradient, dimx=cbps.dimx, dimy=cbps.dimy, theta_mag=theta_mag)

			if ptsrc_ff:
				observed_ims_ptsrc[fieldidx] = zl_realiz

				if pscb_dict['load_cib_fferr']:
					observed_ims_ptsrc[fieldidx] += cib_subthresh['CIB_'+bandstr+'_'+str(ifield)].data

				if pscb_dict['load_isl_fferr']:
					observed_ims_ptsrc[fieldidx] += isl_subthresh['TRILEGAL_'+bandstr+'_'+str(ifield)].data

				if ffidx==0:
					plot_map(observed_ims_ptsrc[fieldidx], title='observed im with point sources')


			if pscb_dict['with_photon_noise'] and not read_noise_fw:
				# this line below was a bug for a while, zl_perfield for observed is zero so it was adding zero photon noise to the realizations
				zl_realiz += shot_sigma_sb_zl[fieldidx]*np.random.normal(0, 1, size=cbps.map_shape) # temp
				observed_ims_nofluc[fieldidx] = zl_realiz

			if pscb_dict['with_inst_noise']: # true for observed
				read_noise_indiv, _ = cbps.noise_model_realization(inst, cbps.map_shape, read_noise_models[fieldidx], \
										read_noise=True, photon_noise=False, chisq=False)


				if read_noise_fw and ffidx==0 and fieldidx==0:
					plot_map(read_noise_indiv, title='read noise indiv ff error')
				observed_ims_nofluc[fieldidx] += read_noise_indiv

				if read_noise_fw:
					# then add ZL since not added at photon noise step
					observed_ims_nofluc[fieldidx] += zl_realiz


		if pscb_dict['iterate_grad_ff']:

			if pscb_dict['quadoff_grad'] or pscb_dict['fc_sub']:

				if pscb_dict['read_noise_fw']:
					niter_use = 1
				else:
					niter_use = float_param_dict['niter']

				print('niter use is ', niter_use)
				print('fitting per quadrant offsets and gradient over full array..')
				# print('poly order ', float_param_dict['poly_order'])
				print('fc sub is ', pscb_dict['fc_sub'])
				print('quadoff grad is ', pscb_dict['quadoff_grad'])
				_, ff_realization_estimates, _, _, _, all_sub_comp = iterative_gradient_ff_solve(observed_ims_nofluc, niter=niter_use, masks=joint_masks, weights_ff=weights_ff, \
																					ff_stack_min=float_param_dict['ff_stack_min'], quadoff_grad=pscb_dict['quadoff_grad'], order=float_param_dict['poly_order'], \
																					fc_sub=pscb_dict['fc_sub'], fc_sub_quad_offset=pscb_dict['fc_sub_quad_offset'], fc_sub_n_terms=float_param_dict['fc_sub_n_terms'], \
																					return_subcomp=True) # masks already accounting for ff_stack_min previously

				# print('Theta for quad offsets:', np.array(all_theta)[:,-4:])


				if read_noise_fw and ffidx==0:

					plot_map(ff_realization_estimates[0], title='read noise ff realization')

				if ptsrc_ff:
					_, ff_realization_ptsrc_estimates, _, _, _ = iterative_gradient_ff_solve(observed_ims_ptsrc, niter=float_param_dict['niter'], masks=joint_masks, weights_ff=weights_ff, \
																						ff_stack_min=float_param_dict['ff_stack_min'], quadoff_grad=pscb_dict['quadoff_grad'], \
																						fc_sub=pscb_dict['fc_sub'], fc_sub_quad_offset=pscb_dict['fc_sub_quad_offset'], fc_sub_n_terms=float_param_dict['fc_sub_n_terms']) # masks already accounting for ff_stack_min previously




			elif pscb_dict['per_quadrant']:
				print('RUNNING ITERATE GRAD FF per quadrant on realizations')
				# processed_ims = np.zeros_like(observed_ims_nofluc)
				ff_realization_estimates = np.zeros_like(observed_ims_nofluc)

				observed_ims_nofluc_byquad = [observed_ims_nofluc[:, cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]] for q in range(4)]

				for q in range(4):

					if pscb_dict['apply_mask']:
						masks_quad = joint_masks[:,cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]]
					else:
						masks_quad = None
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

		if save:
			if read_noise_fw:
				np.savez(fpath_dict['ff_est_dirpath']+'/'+run_name+'/ff_realization_estimate_ffidx'+str(ffidx)+'_readnoiseonly.npz', \
						ff_realization_estimates = ff_realization_estimates)
			else:
				np.savez(fpath_dict['ff_est_dirpath']+'/'+run_name+'/ff_realization_estimate_ffidx'+str(ffidx)+'.npz', \
						ff_realization_estimates = ff_realization_estimates)

			if ptsrc_ff:
				print('Saving point source FF error realizations..')
				np.savez(fpath_dict['ff_est_dirpath']+'/'+run_name+'/ff_realization_ptsrc_estimate_ffidx'+str(ffidx)+'.npz', \
						ff_realization_estimates = ff_realization_ptsrc_estimates)




def run_cbps_pipeline(cbps, inst, nsims, run_name, ifield_list = None, \
							datestr='111323', datestr_trilegal=None, data_type='observed',\
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

	# bandstr_dict = dict({1:'J', 2:'H'})
	bandstr = cbps.bandstr_dict[inst]

	mean_isl_rms = None
	if fpath_dict['isl_rms_fpath'] is not None:
		print('Loading ISL RMS from ', fpath_dict['isl_rms_fpath']+'..')
		mean_isl_rms = load_isl_rms(fpath_dict['isl_rms_fpath'], masking_maglim, nfield)
	
	field_set_shape = (nfield, cbps.dimx, cbps.dimy)
	ps_set_shape = (nsims, nfield, cbps.n_ps_bin)
	single_ps_set_shape = (nfield, cbps.n_ps_bin)

	max_vals_ptsrc = [None for x in range(len(ifield_list))]
	point_src_comps_for_ff = [None for x in range(len(ifield_list))]

	
	if pscb_dict['pt_src_ffnoise']:
		print('Loading point source maps for FF noise terms')
		cib_dffs_basepath = config.ciber_basepath+'data/ciber_mocks/111323/TM'+str(inst)+'/cib_dffs/'
		cib_dffs_fpath = cib_dffs_basepath+'/cib_dffs_default_5field_set0_111323_TM'+str(inst)+'.fits'
		print('Loading CIB subthreshold from ', cib_dffs_fpath)
		cib_subthresh = fits.open(cib_dffs_fpath)

		isl_dffs_basepath = config.ciber_basepath+'data/ciber_mocks/112022/trilegal_dffs/'
		isl_dffs_fpath = isl_dffs_basepath+'/mock_trilegal_simidx0_112022.fits'
		print('Loading ISL subthreshold from ', isl_dffs_fpath)
		isl_subthresh = fits.open(isl_dffs_fpath)
		point_src_comps_for_ff = np.zeros(field_set_shape)

		for fieldidx, ifield in enumerate(ifield_list):

			point_src_comps_for_ff[fieldidx] = cib_subthresh['CIB_'+bandstr+'_'+str(ifield)].data + isl_subthresh['TRILEGAL_'+bandstr+'_'+str(ifield)].data
			if pscb_dict['show_plots']:
				plot_map(point_src_comps_for_ff[fieldidx], title='ifield '+str(ifield)+' point src map')


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
		pscb_dict['iterate_grad_ff'] = False 
		pscb_dict['use_ff_weights'] = False
		pscb_dict['apply_smooth_FF'] = False 
		pscb_dict['save_ff_ests'] = False


	if pscb_dict['mkk_ffest_hybrid']:
		mode_couple_base_dir = fpath_dict['mkk_ffest_base_path']

		if config_dict['mkk_type'] is not None:
			mkk_type = config_dict['mkk_type']
		else:
			if pscb_dict['quadoff_grad']:

				# mkk_type = 'ffest_nograd' # temporary to test mode coupling vs. 1d filtering transfer function
				# mkk_type = 'ffest_quadoff_grad'
				mkk_type = 'ffest_quadoff_grad_noelat30'
				# mkk_type = 'ffest_nograd_noelat30'

				if float_param_dict['poly_order']>=2:
					# mkk_type += '_order2'
					mkk_type += '_order'+str(float_param_dict['poly_order'])
					print('mkk type ', mkk_type)
			else:
				if data_type=='observed':
					mkk_type='ffest_nograd'
				else:
					mkk_type = 'ffest_grad'

	else:
		mode_couple_base_dir = fpath_dict['mkk_base_path']

		if config_dict['mkk_type'] is not None:
			mkk_type = config_dict['mkk_type']
		else:
			mkk_type = 'maskonly_estimate'


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


	#  ------------------- instantiate data arrays  --------------------
	clus_realizations, zl_perfield, ff_estimates, observed_ims = [np.zeros(field_set_shape) for i in range(4)]
	if config_dict['ps_type']=='cross': # cross ciber
		observed_ims_cross = np.zeros(field_set_shape)
		if config_dict['cross_type']=='ciber':
			ff_estimates_cross = np.zeros(field_set_shape)

	inv_Mkks, joint_maskos, joint_maskos_ffest = None, None, None

	l2d = get_l2d(cbps.dimx, cbps.dimy, cbps.pixsize)


	final_masked_images = np.zeros(field_set_shape)
	ps2d_unweighted_perfield = np.zeros(field_set_shape)

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
		print('Loading smooth FF from ', fpath_dict['ff_smooth_fpath'])
		smooth_ff = fits.open(fpath_dict['ff_smooth_fpath'])[1].data
		plot_map(smooth_ff, title='flat field, smoothed with $\\sigma=$'+str(float_param_dict['smooth_sigma']))


	if pscb_dict['compute_cl_theta']:

		theta_masks =  make_theta_masks(cbps.dimx, n_rad_bins=float_param_dict['n_rad_bins'], rad_offset=float_param_dict['rad_offset'], \
										plot=False, ell_min_wedge=float_param_dict['ell_min_wedge'])
	else:
		theta_masks = None
		N_ell_theta = None

	read_noise_models = [None for x in ifield_noise_list]
	read_noise_models_per_quad = None
	if pscb_dict['with_inst_noise']:
		read_noise_models = cbps.grab_noise_model_set(ifield_noise_list, inst, noise_model_base_path=fpath_dict['read_noise_modl_base_path'], noise_modl_type=config_dict['noise_modl_type'])
		
		# plot_map(read_noise_models[0], title='read noise model')
		if float_param_dict['noise_modl_rescale_fac'] is not None:
			print('Artificially scaling read noise models by ', float_param_dict['noise_modl_rescale_fac'])
			read_noise_models *= float_param_dict['noise_modl_rescale_fac']

		if pscb_dict['compute_ps_per_quadrant']:
			read_noise_models_per_quad = []
			for q in range(4):

				read_noise_models_indiv_quad = cbps.grab_noise_model_set(ifield_noise_list, inst, \
													noise_model_base_path=fpath_dict['read_noise_modl_base_path'], noise_modl_type=config_dict['noise_modl_type']+'_quad'+str(q))
				read_noise_models_per_quad.append(read_noise_models_indiv_quad)


		if ciber_cross_ciber:
			print('Loading read noise models for cross instrument TM', cross_inst)
			read_noise_models_cross = cbps.grab_noise_model_set(ifield_noise_list, cross_inst, noise_model_base_path=fpath_dict['cross_read_noise_modl_base_path'])

	if pscb_dict['use_ff_weights']:
		weights_photonly = cbps.compute_ff_weights(inst, zl_levels, read_noise_models=read_noise_models, ifield_list=ifield_list_ff, additional_rms=mean_isl_rms)
		print('FF weights are :', weights_photonly)
		if ciber_cross_ciber:
			weights_photonly_cross = cbps.compute_ff_weights(cross_inst, zl_levels_cross, ifield_list=ifield_list_ff)


	if pscb_dict['transfer_function_correct']:
		t_ell_av, t_ell_fpath = grab_t_ell(config_dict, fpath_dict, float_param_dict, pscb_dict)
		fpath_dict['t_ell_fpath'] = t_ell_fpath


	B_ells, B_ells_cross = grab_B_ell(fpath_dict, config_dict, pscb_dict, nfield)

	# loop through simulations
	for i in np.arange(config_dict['simidx0'], nsims):

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

		if pscb_dict['make_preprocess_file']:
			unmasked_maps_ex, masked_maps_ex, ff_est_ex, filt_comp_ex, proc_map_ex = [[] for x in range(5)]

		if data_type=='mock':

			trilegal_fpath, test_set_fpath = grab_sim_fpaths(inst, pscb_dict['load_trilegal'], pscb_dict['load_ptsrc_cib'], datestr_trilegal, datestr, cib_setidx, fpath_dict['ciber_mock_fpath'], fpath_dict)

			# assume same PS for ell^-3 sky clustering signal (indiv_ifield corresponds to one DGL field that is scaled by dgl_scale_fac)
			merge_dict = Merge(pscb_dict, float_param_dict) # merge dictionaries
			joint_masks, observed_ims, total_signals,\
					snmaps, rnmaps, shot_sigma_sb_maps, noise_models,\
					ff_truth, diff_realizations, zl_perfield, mock_cib_ims = cbps.generate_synthetic_mock_test_set(inst, ifield_list,\
														test_set_fpath=test_set_fpath, mock_trilegal_path=trilegal_fpath,\
														 noise_models=read_noise_models, ff_truth=smooth_ff, zl_levels=zl_levels, **merge_dict)

			# 4/28/24 got rid of proto code for cross mock generation
		
		else:

			if pscb_dict['use_dc_template']:
				print('Loading DC template..')
				dc_template = cbps.load_dark_current_template(inst, verbose=True, inplace=False)
				# if pscb_dict['show_plots'] and not pscb_dict['shut_off_plots']:
				# 	plot_map(dc_template, title='DC template TM'+str(inst))

			for fieldidx, ifield in enumerate(ifield_list):

				cbps.load_flight_image(ifield, inst, verbose=True, ytmap=False) # default loads aducorr maps
				observed_ims[fieldidx] = cbps.image*cbps.cal_facs[inst]

				if pscb_dict['make_preprocess_file']:
					unmasked_maps_ex.append(observed_ims[fieldidx])

				if pscb_dict['use_dc_template']:
					print('subtracting dark current template from image')
					observed_ims[fieldidx] -= dc_template*cbps.cal_facs[inst]

				if ciber_cross_ciber:
					print('Loading data products for cross CIBER TM'+str(cross_inst))

					obs_cross, obs_level_cross = load_proc_regrid_map(fpath_dict, float_param_dict, pscb_dict, regrid_mask_tail, inst, cross_inst, ifield)

					if fieldidx==0:
						obs_levels_cross = np.zeros(len(ifield_list))

					obs_levels_cross[fieldidx] = obs_level_cross
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

				if pscb_dict['show_plots'] and not pscb_dict['shut_off_plots']:
					plot_map(observed_ims[fieldidx], title='observed ims ifield = '+str(ifield))
					
					if config_dict['ps_type']=='cross':
						plot_map(observed_ims_cross[fieldidx], title='observed ims cross ifield = '+str(ifield))

		# 4/28/24 got rid of code for cbps.estimate_b_ell_from_maps, not used but refer to previous versions if interested

		# ----------- load masks and mkk matrices ------------

		if pscb_dict['apply_mask']: # if applying mask load masks and inverse Mkk matrices

			for fieldidx, ifield in enumerate(ifield_list):
				
				if data_type=='observed': # cross
					if ciber_cross_ciber:
						mask_fpath = fpath_dict['mask_base_path']+'/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_TM'+str(inst)+'_TM'+str(cross_inst)+'_observed_'+mask_tail+'.fits'
					else:
						mask_fpath = fpath_dict['mask_base_path']+'/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'

					joint_maskos[fieldidx] = fits.open(mask_fpath)[1].data

					joint_maskos[fieldidx] = additional_masks(cbps, joint_maskos[fieldidx], inst, ifield,\
																 low_responsivity_blob_mask=pscb_dict['low_responsivity_blob_mask'],\
																 apply_wen_cluster_mask=pscb_dict['apply_wen_cluster_mask'], \
																 corner_mask=True)

					if pscb_dict['make_preprocess_file']:
						masked_maps_ex.append(observed_ims[fieldidx]*joint_maskos[fieldidx])

					if ifield==4:
						print('line 968 sum of mask 0 is ', np.sum(joint_maskos[fieldidx]))

					if float_param_dict['clip_sigma'] is not None:

						if pscb_dict['compute_ps_per_quadrant']:
							print('applying sigma clip for individual quadrants..')
							for q in range(4):
								sigclip_quad = iter_sigma_clip_mask(observed_ims[fieldidx, cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]], sig=float_param_dict['clip_sigma'], nitermax=float_param_dict['nitermax'], mask=joint_maskos[fieldidx, cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]].astype(int))
								joint_maskos[fieldidx, cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]] *= sigclip_quad
						else:
							print('Applying sigma clip to uncorrected flight image..')
							sigclip = iter_sigma_clip_mask(observed_ims[fieldidx], sig=float_param_dict['clip_sigma'], nitermax=float_param_dict['nitermax'], mask=joint_maskos[fieldidx].astype(int))
							joint_maskos[fieldidx] *= sigclip
							if pscb_dict['show_plots']:

								# maskobs_smooth = gaussian_filter(joint_maskos[fieldidx]*observed_ims[fieldidx], sigma=2)
								# maskobs_smooth[joint_maskos[fieldidx]==1]-=np.mean(maskobs_smooth[joint_maskos[fieldidx]==1])

								# # maskobs_smooth[maskobs_smooth==0] = np.nan

								# plot_map(maskobs_smooth, title='Smoothed map before FF correction ('+cbps.ciber_field_dict[ifield_list[fieldidx]]+')', hipct=99.999, lopct=0., cmap='jet')


								plot_map(observed_ims[fieldidx]*joint_maskos[fieldidx], cmap='Greys_r', title='masked image line 699 , ifield '+str(ifield))

						if ifield==4:
							print('line 986 sum of joint masko  0 is ', np.sum(joint_maskos[fieldidx]))

					if fpath_dict['mkk_ffest_mask_tail'] is None:
						mkk_ffest_mask_tail = mask_tail
					else:
						mkk_ffest_mask_tail = fpath_dict['mkk_ffest_mask_tail']


					print('mkk ffest mask tail before load mkk fcsub is ', mkk_ffest_mask_tail)
					if pscb_dict['fc_sub']:

						inv_Mkk_truncated = load_mkk_fcsub(float_param_dict, mode_couple_base_dir, mkk_ffest_mask_tail, mkk_type, inst, ifield, invert_spliced_matrix=pscb_dict['invert_spliced_matrix'], compute_pinv=pscb_dict['compute_mkk_pinv'])

						inv_Mkks.append(inv_Mkk_truncated)

					else:
						inv_Mkk_fpath = mode_couple_base_dir+'/'+mkk_ffest_mask_tail+'/mkk_'+mkk_type+'_ifield'+str(ifield)+'_observed_'+mkk_ffest_mask_tail+'.fits'
						print('Loading inv Mkk from ', inv_Mkk_fpath)
						inv_Mkks.append(fits.open(inv_Mkk_fpath)['inv_Mkk_'+str(ifield)].data)

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
																	 corner_mask=True)

						if float_param_dict['clip_sigma_ff'] is not None:

							sigclip = iter_sigma_clip_mask(observed_ims[fieldidx], sig=float_param_dict['clip_sigma_ff'], nitermax=float_param_dict['nitermax'], mask=joint_maskos_ffest[fieldidx].astype(int))
							joint_maskos_ffest[fieldidx] *= sigclip

						if ifield==4:
							print('line 715 sum of ff mask 0 is ', np.sum(joint_maskos_ffest[fieldidx]))

				else:
					if pscb_dict['verbose']:
						print('mask path is ', fpath_dict['mask_base_path']+'/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_'+mask_tail+'.fits')
					joint_maskos[fieldidx] = fits.open(fpath_dict['mask_base_path']+'/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_'+mask_tail+'.fits')['joint_mask_'+str(ifield)].data
					
					if mask_tail_ffest is not None:
						print('Loading FFest full mask from ', mask_tail_ffest) 
						mask_fpath_ffest = fpath_dict['mask_base_path']+'/'+mask_tail_ffest+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_'+mask_tail_ffest+'.fits'
						joint_maskos_ffest[fieldidx] = fits.open(mask_fpath_ffest)['joint_mask_'+str(ifield)].data
				
					if pscb_dict['mkk_ffest_hybrid']:
						mkk_ffest_mask_tail = mask_tail
						if pscb_dict['quadoff_grad']:
							print('loading quad off grad mkk matrix')
							inv_Mkks.append(fits.open(mode_couple_base_dir+'/'+mkk_ffest_mask_tail+'/mkk_'+mkk_type+'_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_'+mkk_ffest_mask_tail+'.fits')['inv_Mkk_'+str(ifield)].data)
						
						elif pscb_dict['fc_sub']:

							inv_Mkk_truncated = load_mkk_fcsub(float_param_dict, mode_couple_base_dir, mkk_ffest_mask_tail, mkk_type, inst, ifield, cib_setidx=cib_setidx, invert_spliced_matrix=pscb_dict['invert_spliced_matrix'], compute_pinv=pscb_dict['compute_mkk_pinv'])
							inv_Mkks.append(inv_Mkk_truncated)

						else:
							inv_Mkks.append(fits.open(mode_couple_base_dir+'/'+mkk_ffest_mask_tail+'/mkk_'+mkk_type+'_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_'+mkk_ffest_mask_tail+'.fits')['inv_Mkk_'+str(ifield)].data)
					else:
						if pscb_dict['quadoff_grad']:
							print('Opening mask + filtering mkk matrices..')
							inv_Mkks.append(fits.open(mode_couple_base_dir+'/'+mask_tail+'/mkk_'+mkk_type+'_mask_estimate_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_'+mask_tail+'.fits')['inv_Mkk_'+str(ifield)].data)
						else:
							inv_Mkks.append(fits.open(mode_couple_base_dir+'/'+mask_tail+'/mkk_maskonly_estimate_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_'+mask_tail+'.fits')['inv_Mkk_'+str(ifield)].data)

			orig_mask_fractions = np.array(orig_mask_fractions)

			if pscb_dict['iterate_grad_ff']:
				print('mask fractions before FF estimation are ', orig_mask_fractions)

		# 11/13/23 removed weights_ff_cross since pre-processed and then regrid for cross
		if pscb_dict['iterate_grad_ff']:
			weights_ff = None
			if pscb_dict['use_ff_weights']:
				weights_ff = weights_photonly

			print('weights ff is ', weights_ff)

			if pscb_dict['verbose']:
				print('RUNNING ITERATE GRAD FF for observed images')
				print('and weights_ff is ', weights_ff)

			if pscb_dict['save_intermediate_cls']:
				cls_inter.append(grab_ps_set(observed_ims, ifield_list, single_ps_set_shape, cbps, masks=joint_maskos))
				inter_labels.append('preffgrad_masked')


			if pscb_dict['quadoff_grad']:
				quad_means_masked_by_ifield, quad_stds_masked_by_ifield = [np.zeros((len(ifield_list), 4)) for x in range(2)]

				if joint_maskos_ffest is not None:
					joint_maskos_ffest, stack_masks = stack_masks_ffest(joint_maskos_ffest, float_param_dict['ff_stack_min'])
					joint_maskos *= stack_masks

					print('FFest not None, Fitting per quadrant offsets and gradient over full array..')
					# processed_ims, ff_estimates, final_planes, stack_masks, all_coeffs = iterative_gradient_ff_solve(observed_ims, niter=float_param_dict['niter'], masks=joint_maskos, weights_ff=weights_ff, \
					# 																		ff_stack_min=float_param_dict['ff_stack_min'], quadoff_grad=pscb_dict['quadoff_grad'], plot=False) # masks already accounting for ff_stack_min previously

					processed_ims, ff_estimates, final_planes, stack_masks, all_coeffs = iterative_gradient_ff_solve(observed_ims, niter=float_param_dict['niter'], masks=joint_maskos_ffest, weights_ff=weights_ff, \
																						ff_stack_min=float_param_dict['ff_stack_min'], quadoff_grad=pscb_dict['quadoff_grad'], plot=False, \
																						order=float_param_dict['poly_order']) # masks already accounting for ff_stack_min previously

				else:
					print('FFest is None, Fitting per quadrant offsets and gradient over full array..')
					processed_ims, ff_estimates, final_planes, stack_masks, all_coeffs = iterative_gradient_ff_solve(observed_ims, niter=float_param_dict['niter'], masks=joint_maskos, weights_ff=weights_ff, \
																						ff_stack_min=float_param_dict['ff_stack_min'], quadoff_grad=pscb_dict['quadoff_grad']) # masks already accounting for ff_stack_min previously


			elif pscb_dict['fc_sub']:
				quad_means_masked_by_ifield, quad_stds_masked_by_ifield = [np.zeros((len(ifield_list), 4)) for x in range(2)]

				if joint_maskos_ffest is not None:
					joint_maskos_ffest, stack_masks = stack_masks_ffest(joint_maskos_ffest, float_param_dict['ff_stack_min'])
					joint_maskos *= stack_masks

					processed_ims, ff_estimates, final_planes, stack_masks, all_coeffs, all_sub_comp = iterative_gradient_ff_solve(observed_ims, niter=1, masks=joint_maskos_ffest, weights_ff=weights_ff, \
																						ff_stack_min=float_param_dict['ff_stack_min'], plot=False, \
																						fc_sub=pscb_dict['fc_sub'], fc_sub_quad_offset=pscb_dict['fc_sub_quad_offset'], fc_sub_n_terms=float_param_dict['fc_sub_n_terms'], \
																						return_subcomp=True) # masks already accounting for ff_stack_min previously

					print('Quad offsets are:', all_coeffs[0,:,-4:])

					if pscb_dict['make_preprocess_file']:
						proc_map_ex = processed_ims.copy()
						ff_est_ex = ff_estimates.copy()

				else:
					processed_ims, ff_estimates, final_planes, stack_masks, all_coeffs, all_sub_comp = iterative_gradient_ff_solve(observed_ims, niter=1, weights_ff=weights_ff, \
																						ff_stack_min=float_param_dict['ff_stack_min'], plot=False, \
																						fc_sub=pscb_dict['fc_sub'], fc_sub_quad_offset=pscb_dict['fc_sub_quad_offset'], fc_sub_n_terms=float_param_dict['fc_sub_n_terms'], \
																						return_subcomp=True) # masks already accounting for ff_stack_min previously

				if i==0 and pscb_dict['show_plots']:

					for fieldidx, ifield in enumerate(ifield_list):

						plot_map(all_sub_comp[fieldidx]-np.mean(all_sub_comp[fieldidx]), title='FC+quad offset comp, ifield '+str(ifield), figsize=(6, 6), cmap='bwr')


			elif pscb_dict['per_quadrant']:
				if pscb_dict['verbose']:
					print('Processing quadrants of input maps separately..')
					print('NOTE: need to update this so that wrapper works')

				processed_ims = np.zeros_like(observed_ims)
				ff_estimates = np.zeros_like(observed_ims)

				if joint_maskos_ffest is not None:
					joint_maskos_ffest, stack_masks = stack_masks_ffest(joint_maskos_ffest, float_param_dict['ff_stack_min'])
					joint_maskos *= stack_masks

				quad_means_masked_by_ifield, quad_stds_masked_by_ifield = [np.zeros((len(ifield_list), 4)) for x in range(2)]
				ciber_maps_byquad = [observed_ims[:, cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]] for q in range(4)]

				for q, quad in enumerate(ciber_maps_byquad):
					if pscb_dict['verbose']:
						print('q = ', q)

					if pscb_dict['apply_mask']:
						if joint_maskos_ffest is not None:
							masks_quad = joint_maskos_ffest[:,cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]]
							clip_sigma_ff=float_param_dict['clip_sigma_ff']
						else:
							masks_quad = joint_maskos[:,cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]]
							clip_sigma_ff=None
					else:
						masks_quad = np.ones((len(ifield_list), 512, 512))
						clip_sigma_ff = None

					# sigma clip happening further up in code already, set to None here
					processed_ciber_maps_quad, ff_estimates_quad,\
						final_planes, stack_masks, ff_weights = process_ciber_maps(cbps, ifield_list, inst, ciber_maps_byquad[q], masks_quad,\
																				 clip_sigma=clip_sigma_ff, nitermax=float_param_dict['nitermax'], \
																					niter=float_param_dict['niter'], ff_stack_min=float_param_dict['ff_stack_min'], \
																					ff_weights=weights_photonly)

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

			if pscb_dict['quadoff_grad'] or pscb_dict['per_quadrant'] or pscb_dict['fc_sub_quad_offset']:

				if pscb_dict['apply_mask']:
					ciber_masks_byquad_temp = [joint_maskos[:, cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]] for q in range(4)]

				processed_ims_byquad = [processed_ims[:,cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]] for q in range(4)]
				for fieldidx, ifield in enumerate(ifield_list):

					# plot_map(processed_ims[fieldidx]*joint_maskos[fieldidx], cmap='Greys', title='processed * mask after process_ciber_maps')
					if pscb_dict['apply_mask']:
						quad_medians_masked = [np.median((cbmap[fieldidx]*ciber_masks_byquad_temp[q][fieldidx])[(ciber_masks_byquad_temp[q][fieldidx] != 0)]) for q, cbmap in enumerate(processed_ims_byquad)]
					else:
						quad_medians_masked = [np.median((cbmap[fieldidx])) for q, cbmap in enumerate(processed_ims_byquad)]

					if pscb_dict['verbose']:
						print('quad medians masked for ifield '+str(ifield)+' is ', quad_medians_masked)

					quad_means_masked_by_ifield[fieldidx] = quad_medians_masked

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

			if pscb_dict['make_preprocess_file']:

				print('for preprocess file:', len(unmasked_maps_ex), len(masked_maps_ex), len(ff_est_ex), len(all_sub_comp), len(proc_map_ex))
				for fieldidx, ifield in enumerate(ifield_list):

					proc_map_save = proc_map_ex[fieldidx]*joint_maskos[fieldidx]

					proc_map_save[proc_map_save != 0] -= np.mean(proc_map_save[proc_map_save!=0])

					hdul = make_proc_fits_file(inst, ifield, dc_template, unmasked_maps_ex[fieldidx], masked_maps_ex[fieldidx], ff_est_ex[fieldidx], all_sub_comp[fieldidx], proc_map_save)
					hdul.writeto('data/preprocess_example/preprocess_example_TM'+str(inst)+'_ifield'+str(ifield)+'.fits', overwrite=True)

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
			# if ciber_cross_ciber and pscb_dict['ff_estimate_cross']:
			# 	observed_ims_cross = processed_ims_cross.copy()

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

		
		else:

			if pscb_dict['quadoff_grad']:
				quad_means_masked_by_ifield, quad_stds_masked_by_ifield = [np.zeros((len(ifield_list), 4)) for x in range(2)]
				print('fitting per quadrant offsets and gradient over full array..')

				processed_ims = np.zeros_like(observed_ims)
				for fieldidx, ifield in enumerate(ifield_list):

					dot1, X, mask_rav = precomp_offset_gradient(cbps.dimx, cbps.dimy, mask=joint_maskos[fieldidx])

					theta, plane_offsets = offset_gradient_fit_precomp(observed_ims[fieldidx], dot1, X, mask_rav=mask_rav)

					print('theta = ', theta)

					processed_ims[fieldidx] = observed_ims[fieldidx]-plane_offsets

					# plot_map(processed_ims[fieldidx], title='after quadoff grad filter')

				# processed_ims, ff_estimates, final_planes, stack_masks, all_coeffs = iterative_gradient_ff_solve(observed_ims, niter=float_param_dict['niter'], masks=joint_maskos, weights_ff=weights_ff, \
				# 																	ff_stack_min=float_param_dict['ff_stack_min'], quadoff_grad=pscb_dict['quadoff_grad']) # masks already accounting for ff_stack_min previously


			elif pscb_dict['per_quadrant']:
				quad_means_masked_by_ifield = np.zeros((len(ifield_list), 4))
				ciber_masks_byquad_temp = [joint_maskos[:, cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]] for q in range(4)]

				observed_ims_byquad = [observed_ims[:,cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]] for q in range(4)]

				# for perfect FF case and per quadrant
				for fieldidx, ifield in enumerate(ifield_list):
					quad_medians_masked = [np.median((cbmap[fieldidx]*ciber_masks_byquad_temp[q][fieldidx])[(ciber_masks_byquad_temp[q][fieldidx] != 0)]) for q, cbmap in enumerate(observed_ims_byquad)]
					# if pscb_dict['verbose']:
					print('quad medians masked for ifield '+str(ifield)+' is ', quad_medians_masked)
					quad_means_masked_by_ifield[fieldidx] = quad_medians_masked

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


			# print('before ff error realiz, sum mask 0 is ', np.sum(joint_maskos[0]))
			if joint_maskos_ffest is not None:
				print('using ff masks for ff error realizations..')

				if pscb_dict['read_noise_fw']:
					generate_ff_error_realizations(cbps, run_name, inst, ifield_list, joint_maskos_ffest, simmaps_dc_all, shot_sigma_sb_zl, read_noise_models, weights_ff_nofluc, \
												 fpath_dict, pscb_dict, float_param_dict, obs_levels, read_noise_fw=True)

				generate_ff_error_realizations(cbps, run_name, inst, ifield_list, joint_maskos_ffest, simmaps_dc_all, shot_sigma_sb_zl, read_noise_models, weights_ff_nofluc, \
											 fpath_dict, pscb_dict, float_param_dict, obs_levels, apply_zl_gradient=True, theta_mag=float_param_dict['theta_mag'])
			else:
				generate_ff_error_realizations(cbps, run_name, inst, ifield_list, joint_maskos, simmaps_dc_all, shot_sigma_sb_zl, read_noise_models, weights_ff_nofluc, \
											 fpath_dict, pscb_dict, float_param_dict, obs_levels, apply_zl_gradient=True, theta_mag=float_param_dict['theta_mag'])

			# print('after ff error realiz, sum mask 0 is ', np.sum(joint_maskos[0]))

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

				if config_dict['ps_type']=='cross':
					print('1581 Simmap dc cross :', simmap_dc_cross)

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
					mean_cl2d_nofluc = noisemodl_file['mean_cl2d_nofluc']

					if pscb_dict['compute_cl_theta'] and pscb_dict['cut_cl_theta']:
						for which_exclude in [0, float_param_dict['n_rad_bins']//2]:
							fourier_weights_nofluc[theta_masks[which_exclude]==1] = 0.

					if pscb_dict['read_noise_fw'] and ifield_list[fieldidx] != 5:
						print('loading read noise fourier weights')
						noisemodl_tailpath_readnoiseonly = '/noise_bias_fieldidx'+str(fieldidx)+'_readnoiseonly.npz'
						noisemodl_fpath_readnoiseonly = fpath_dict['noisemodl_basepath'] + fpath_dict['noisemodl_run_name'] + noisemodl_tailpath_readnoiseonly
						noisemodl_file_readnoiseonly = np.load(noisemodl_fpath_readnoiseonly)
						fourier_weights_readnoiseonly = noisemodl_file_readnoiseonly['fourier_weights']

						plot_map(fourier_weights_readnoiseonly, title='read noise fourier weights')

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

						nl1ds_nAnB = np.array(noisemodl_file['nl1ds_nAnB'])
						nl1ds_nAsB = np.array(noisemodl_file['nl1ds_nAsB'])
						nl1ds_nBsA = np.array(noisemodl_file['nl1ds_nBsA'])

						var_nAnB = np.var(nl1ds_nAnB, axis=0)
						var_nAsB = np.var(nl1ds_nAsB, axis=0)
						var_nBsA = np.var(nl1ds_nBsA, axis=0)

						dcl_cross = np.sqrt(var_nAnB+var_nAsB+var_nBsA)

						# lb, N_ell_est, N_ell_err = cbps.compute_noise_power_spectrum(inst, noise_Cl2D=mean_cl2d_cross.copy(), inplace=False, apply_FW=pscb_dict['apply_FW'], weights=fourier_weights_cross)
						nl_dict = cbps.compute_noise_power_spectrum(inst, noise_Cl2D=mean_cl2d_cross.copy(), inplace=False, apply_FW=pscb_dict['apply_FW'], weights=fourier_weights_cross)
						lb = nl_dict['lbins']
						N_ell_est = nl_dict['Cl_noise']
						N_ell_err = nl_dict['Clerr']

						plt.figure()
						prefac = lb*(lb+1)/(2*np.pi)
						plt.errorbar(lb, prefac*N_ell_est, yerr=prefac*N_ell_err, fmt='o')
						plt.plot(lb, prefac*dcl_cross, color='r', linestyle='dashed', label='dcl in quadrature')
						plt.xlabel('$\\ell$', fontsize=14)
						plt.ylabel('$D_{\\ell}$', fontsize=14)
						plt.legend()
						plt.xscale('log')
						plt.show()

						print('N_ell_est for cross ciber is ', N_ell_est)
					else:
						# mean_cl2d_nofluc = noisemodl_file['mean_cl2d_nofluc']

						if pscb_dict['read_noise_fw'] and ifield_list[fieldidx] != 5:

							nl_dict = cbps.compute_noise_power_spectrum(inst, noise_Cl2D=mean_cl2d_nofluc.copy(), inplace=False, apply_FW=pscb_dict['apply_FW'], weights=fourier_weights_readnoiseonly, weight_power=float_param_dict['weight_power'], theta_masks=theta_masks)
							# lb, N_ell_est, N_ell_err = cbps.compute_noise_power_spectrum(inst, noise_Cl2D=mean_cl2d_nofluc.copy(), inplace=False, apply_FW=pscb_dict['apply_FW'], weights=fourier_weights_readnoiseonly, weight_power=float_param_dict['weight_power'])
							
							lb = nl_dict['lbins']
							N_ell_est = nl_dict['Cl_noise']
							N_ell_err = nl_dict['Clerr']

							if pscb_dict['compute_cl_theta']:
								N_ell_theta = nl_dict['N_ell_theta']
								# print('N_ell_theta is ', N_ell_theta)
								N_ell_theta_err = nl_dict['N_ell_theta_err']


						else:
							N_ell_est = noisemodl_file['nl_estFF_nofluc']

							if pscb_dict['compute_cl_theta']:

								nl_dict = cbps.compute_noise_power_spectrum(inst, noise_Cl2D=mean_cl2d_nofluc.copy(), inplace=False, apply_FW=pscb_dict['apply_FW'], weights=fourier_weights_nofluc, weight_power=float_param_dict['weight_power'], theta_masks=theta_masks)

								N_ell_theta = nl_dict['N_ell_theta']
								# print('N_ell_theta is ', N_ell_theta)
								N_ell_theta_err = nl_dict['N_ell_theta_err']

							# nl_dict = cbps.compute_noise_power_spectrum(inst, noise_Cl2D=mean_cl2d_nofluc.copy(), inplace=False, apply_FW=pscb_dict['apply_FW'], weights=fourier_weights_readnoiseonly, weight_power=float_param_dict['weight_power'], theta_masks=theta_masks)



						# fourier_weights_cross = None

				else:
					all_ff_ests_nofluc = None
					all_ff_ests_ptsrc = None
					dcl_cross=None

					if pscb_dict['verbose']:
						print('median obs is ', np.median(obs))
						print('simmap dc is ', simmap_dc)

					cross_shot_sigma_sb_zl_noisemodl = None
					shot_sigma_sb_zl_noisemodl = cbps.compute_shot_sigma_map(inst, image=obs_levels[fieldidx]*np.ones_like(obs), nfr=nfr_fields[fieldidx])

					if ciber_cross_ciber:
						if pscb_dict['estimate_cross_noise']:
							if pscb_dict['iterate_grad_ff']:
								all_ff_ests_nofluc, all_ff_ests_nofluc_cross = cbps.collect_ff_realiz_estimates(fieldidx, run_name, fpath_dict, pscb_dict, config_dict, float_param_dict, datestr=datestr, ff_min=float_param_dict['ff_min_nr'], ff_max=float_param_dict['ff_max_nr'])
						


							print('obs levels cross  is ', obs_levels_cross[fieldidx])
							cross_shot_sigma_sb_zl_noisemodl = cbps.compute_shot_sigma_map(cross_inst, image=obs_levels_cross[fieldidx]*np.ones_like(obs), nfr=nfr_fields[fieldidx])

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
															  inplace=False, show=True, per_quadrant=pscb_dict['per_quadrant'], regrid_cross=False, fc_sub=pscb_dict['fc_sub'], fc_sub_quad_offset=pscb_dict['fc_sub_quad_offset'], \
															  fc_sub_n_terms=float_param_dict['fc_sub_n_terms'])
							
							var_nAnB = np.var(nl1ds_nAnB, axis=0)
							var_nAsB = np.var(nl1ds_nAsB, axis=0)
							var_nBsA = np.var(nl1ds_nBsA, axis=0)

							dcl_cross = np.sqrt(var_nAnB+var_nAsB+var_nBsA)

							print('dl dcl cross:', (cbps.Mkk_obj.midbin_ell**2)*dcl_cross/(2*np.pi))

							plot_map(np.log10(fourier_weights_cross), title='log10 fourier_weights_cross')
							plot_map(mean_cl2d_cross, title='log10 mean_nl2d_cross')

							var_nl2d_total = var_nl2d_nAsB+var_nl2d_nBsA

							mean_nl2d_cross_total = mean_cl2d_cross + mean_nl2d_nAsB + mean_nl2d_nBsA
							fourier_weights_cross = 1./((1./fourier_weights_cross) + var_nl2d_nAsB + var_nl2d_nBsA)

							plot_map(mean_nl2d_cross_total, title='mean_nl2d_cross_total')
							plot_map(fourier_weights_cross, title='fourier_weights_cross_total')

							nl_dict = cbps.compute_noise_power_spectrum(inst, noise_Cl2D=mean_nl2d_cross_total.copy(), inplace=False, apply_FW=pscb_dict['apply_FW'], weights=fourier_weights_cross, weight_power=float_param_dict['weight_power'])
							# lb, N_ell_est, N_ell_err = cbps.compute_noise_power_spectrum(inst, noise_Cl2D=mean_nl2d_cross_total.copy(), inplace=False, apply_FW=pscb_dict['apply_FW'], weights=fourier_weights_cross, weight_power=float_param_dict['weight_power'])

							lb = nl_dict['lbins']
							N_ell_est = nl_dict['Cl_noise']
							N_ell_err = nl_dict['Clerr']



							plt.figure()
							prefac = lb*(lb+1)/(2*np.pi)
							plt.errorbar(lb, prefac*N_ell_est, yerr=prefac*N_ell_err, fmt='o')
							plt.xscale('log')
							plt.show()

					else:
						if pscb_dict['iterate_grad_ff']:

							all_ff_ests_nofluc, all_ff_ests_nofluc_cross = cbps.collect_ff_realiz_estimates(fieldidx, run_name, fpath_dict, pscb_dict, config_dict, float_param_dict, datestr=datestr, ff_min=float_param_dict['ff_min_nr'], ff_max=float_param_dict['ff_max_nr'])

							if pscb_dict['load_cib_fferr'] or pscb_dict['load_isl_fferr']:
								all_ff_ests_ptsrc, _ = cbps.collect_ff_realiz_estimates(fieldidx, run_name, fpath_dict, pscb_dict, config_dict, float_param_dict, datestr=datestr, ff_min=float_param_dict['ff_min'], ff_max=float_param_dict['ff_max'], \
																						ff_type='ptsrc_estimate')

						# simmap_dc_use = None
						# if pscb_dict['mkk_ffest_hybrid']:
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
												field_nfr=nfr_fields[fieldidx], simmap_dc=simmap_dc_use[cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]], ff_truth=None, mc_ff_estimates=all_ff_ests_nofluc_indiv, gradient_filter=pscb_dict['gradient_filter'],\
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
							if not pscb_dict['shut_off_plots'] and pscb_dict['apply_mask']:
								plot_map(target_mask*shot_sigma_sb_zl_noisemodl, title='shot noise sigma ifield '+str(ifield_noise_list[fieldidx]))
							
							if all_ff_ests_nofluc is None:
								print('all_ff_ests_nofluc is None!!! 1396')


							if pscb_dict['read_noise_fw'] and ifield_list[fieldidx] != 5:

								# print('collecting read noise FF error realizations')
								# all_ff_ests_readnoiseonly, _ = cbps.collect_ff_realiz_estimates(fieldidx, run_name, fpath_dict, pscb_dict, config_dict, float_param_dict, datestr=datestr, ff_min=0.6, ff_max=1.7, read_noise_fw=True)
								all_ff_ests_readnoiseonly, _ = cbps.collect_ff_realiz_estimates(fieldidx, run_name, fpath_dict, pscb_dict, config_dict, float_param_dict, datestr=datestr, read_noise_fw=True)

								print('Generating read noise Fourier weights..')


								noise_bias_dict = cbps.estimate_noise_power_spectrum(nsims=float_param_dict['n_FW_sims'], n_split=float_param_dict['n_FW_split'], apply_mask=pscb_dict['apply_mask'], \
											   noise_model=read_noise_models[fieldidx], read_noise=pscb_dict['with_inst_noise'], inst=inst, ifield=ifield_noise_list[fieldidx], show=False, mask=target_mask, \
												photon_noise=False, ff_estimate=None, inplace=False, \
												field_nfr=nfr_fields[fieldidx], simmap_dc=simmap_dc_use, mc_ff_estimates=all_ff_ests_readnoiseonly, gradient_filter=pscb_dict['gradient_filter'],\
												per_quadrant=pscb_dict['per_quadrant'], quadoff_grad=pscb_dict['quadoff_grad'], order=float_param_dict['poly_order'], \
												fc_sub=pscb_dict['fc_sub'], fc_sub_quad_offset=pscb_dict['fc_sub_quad_offset'], fc_sub_n_terms=float_param_dict['fc_sub_n_terms'], theta_masks=theta_masks)

								fourier_weights_readnoiseonly = noise_bias_dict['fourier_weights']
								mean_cl2d_readnoiseonly = noise_bias_dict['mean_cl2d']

								# fourier_weights_readnoiseonly, mean_cl2d_readnoiseonly = cbps.estimate_noise_power_spectrum(nsims=float_param_dict['n_FW_sims'], n_split=float_param_dict['n_FW_split'], apply_mask=pscb_dict['apply_mask'], \
								# 			   noise_model=read_noise_models[fieldidx], read_noise=pscb_dict['with_inst_noise'], inst=inst, ifield=ifield_noise_list[fieldidx], show=False, mask=target_mask, \
								# 				photon_noise=False, ff_estimate=None, inplace=False, \
								# 				field_nfr=nfr_fields[fieldidx], simmap_dc=simmap_dc_use, mc_ff_estimates=all_ff_ests_readnoiseonly, gradient_filter=pscb_dict['gradient_filter'],\
								# 				per_quadrant=pscb_dict['per_quadrant'], quadoff_grad=pscb_dict['quadoff_grad'], order=float_param_dict['poly_order'], \
								# 				fc_sub=pscb_dict['fc_sub'], fc_sub_quad_offset=pscb_dict['fc_sub_quad_offset'], fc_sub_n_terms=float_param_dict['fc_sub_n_terms'])



							
							# fourier_weights_nofluc, mean_cl2d_nofluc = cbps.estimate_noise_power_spectrum(nsims=float_param_dict['n_FW_sims'], n_split=float_param_dict['n_FW_split'], apply_mask=pscb_dict['apply_mask'], \
							# 			   noise_model=read_noise_models[fieldidx], read_noise=pscb_dict['with_inst_noise'], inst=inst, ifield=ifield_noise_list[fieldidx], show=False, mask=target_mask, \
							# 				photon_noise=pscb_dict['with_photon_noise'], ff_estimate=None, shot_sigma_sb=shot_sigma_sb_zl_noisemodl, inplace=False, \
							# 				field_nfr=nfr_fields[fieldidx], simmap_dc=simmap_dc_use, mc_ff_estimates=all_ff_ests_nofluc, gradient_filter=pscb_dict['gradient_filter'],\
							# 				per_quadrant=pscb_dict['per_quadrant'], point_src_comp=point_src_comps_for_ff[fieldidx], \
							# 				quadoff_grad=pscb_dict['quadoff_grad'], order=float_param_dict['poly_order'], mc_ff_ptsrc_estimates=all_ff_ests_ptsrc, \
							# 				fc_sub=pscb_dict['fc_sub'], fc_sub_quad_offset=pscb_dict['fc_sub_quad_offset'], fc_sub_n_terms=float_param_dict['fc_sub_n_terms'])


							noise_bias_dict = cbps.estimate_noise_power_spectrum(nsims=float_param_dict['n_FW_sims'], n_split=float_param_dict['n_FW_split'], apply_mask=pscb_dict['apply_mask'], \
										   noise_model=read_noise_models[fieldidx], read_noise=pscb_dict['with_inst_noise'], inst=inst, ifield=ifield_noise_list[fieldidx], show=False, mask=target_mask, \
											photon_noise=pscb_dict['with_photon_noise'], ff_estimate=None, shot_sigma_sb=shot_sigma_sb_zl_noisemodl, inplace=False, \
											field_nfr=nfr_fields[fieldidx], simmap_dc=simmap_dc_use, mc_ff_estimates=all_ff_ests_nofluc, gradient_filter=pscb_dict['gradient_filter'],\
											per_quadrant=pscb_dict['per_quadrant'], point_src_comp=point_src_comps_for_ff[fieldidx], \
											quadoff_grad=pscb_dict['quadoff_grad'], order=float_param_dict['poly_order'], mc_ff_ptsrc_estimates=all_ff_ests_ptsrc, \
											fc_sub=pscb_dict['fc_sub'], fc_sub_quad_offset=pscb_dict['fc_sub_quad_offset'], fc_sub_n_terms=float_param_dict['fc_sub_n_terms'])

							fourier_weights_nofluc = noise_bias_dict['fourier_weights']
							mean_cl2d_nofluc = noise_bias_dict['mean_nl2d']

							# fourier_weights_nofluc, mean_cl2d_nofluc = cbps.estimate_noise_power_spectrum(nsims=float_param_dict['n_FW_sims'], n_split=float_param_dict['n_FW_split'], apply_mask=pscb_dict['apply_mask'], \
							# 			   noise_model=read_noise_models[fieldidx], read_noise=pscb_dict['with_inst_noise'], inst=inst, ifield=ifield_noise_list[fieldidx], show=False, mask=target_mask, \
							# 				photon_noise=pscb_dict['with_photon_noise'], ff_estimate=None, shot_sigma_sb=shot_sigma_sb_zl_noisemodl, inplace=False, \
							# 				field_nfr=nfr_fields[fieldidx], simmap_dc=simmap_dc_use, ff_truth=smooth_ff, mc_ff_estimates=all_ff_ests_nofluc, gradient_filter=pscb_dict['gradient_filter'],\
							# 				chisq=False, per_quadrant=pscb_dict['per_quadrant'], point_src_comp=point_src_comps_for_ff[fieldidx])

							if pscb_dict['read_noise_fw'] and ifield_list[fieldidx] != 5:
								print('noise bias with read noise fourier weights')
								nl_dict = cbps.compute_noise_power_spectrum(inst, noise_Cl2D=mean_cl2d_nofluc.copy(), inplace=False, apply_FW=pscb_dict['apply_FW'], weights=fourier_weights_readnoiseonly, weight_power=float_param_dict['weight_power'], theta_masks=theta_masks)
								# lb, N_ell_est, N_ell_err = cbps.compute_noise_power_spectrum(inst, noise_Cl2D=mean_cl2d_nofluc.copy(), inplace=False, apply_FW=pscb_dict['apply_FW'], weights=fourier_weights_readnoiseonly, weight_power=float_param_dict['weight_power'], theta_masks=theta_masks)
							else:
								nl_dict = cbps.compute_noise_power_spectrum(inst, noise_Cl2D=mean_cl2d_nofluc.copy(), inplace=False, apply_FW=pscb_dict['apply_FW'], weights=fourier_weights_nofluc, weight_power=float_param_dict['weight_power'], theta_masks=theta_masks)
								# lb, N_ell_est, N_ell_err = cbps.compute_noise_power_spectrum(inst, noise_Cl2D=mean_cl2d_nofluc.copy(), inplace=False, apply_FW=pscb_dict['apply_FW'], weights=fourier_weights_nofluc, weight_power=float_param_dict['weight_power'], theta_masks)

							lb = nl_dict['lbins']
							N_ell_est = nl_dict['Cl_noise']
							N_ell_err = nl_dict['Clerr']

							if pscb_dict['compute_cl_theta']:

								N_ell_theta = nl_dict['N_ell_theta']
								N_ell_theta_err = nl_dict['N_ell_theta_err']


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

					if pscb_dict['read_noise_fw'] and ifield_list[fieldidx] != 5:
						noisemodl_tailpath_readnoiseonly = '/noise_bias_fieldidx'+str(fieldidx)+'_readnoiseonly.npz'
						noisemodl_fpath_readnoiseonly = fpath_dict['noisemodl_basepath'] + fpath_dict['noisemodl_run_name'] + noisemodl_tailpath_readnoiseonly


					if pscb_dict['save'] or not os.path.exists(noisemodl_fpath):

						if ciber_cross_ciber:
							if pscb_dict['estimate_cross_noise']:
								print('SAVING CROSS NOISE UNCERTAINTY FILE..')
								if os.path.exists(noisemodl_fpath):
									print('Overwriting existing noise model bias file..')
								# np.savez(noisemodl_fpath, fourier_weights_nofluc=fourier_weights_cross, mean_cl2d_nofluc=mean_nl2d_cross_total, nl_estFF_nofluc=N_ell_est)
								np.savez(noisemodl_fpath, fourier_weights_nofluc=fourier_weights_cross, mean_cl2d_nofluc=mean_nl2d_cross_total, nl_estFF_nofluc=N_ell_est, \
									nl1ds_nAnB=nl1ds_nAnB, nl1ds_nAsB=nl1ds_nAsB, nl1ds_nBsA=nl1ds_nBsA)

						else:
							print('SAVING NOISE MODEL BIAS FILE..')
							if os.path.exists(noisemodl_fpath):
								print('Overwriting existing noise model bias file..')
							if pscb_dict['read_noise_fw'] and ifield_list[fieldidx] != 5:
								print('Saving read noise fourier weights')
								np.savez(noisemodl_fpath_readnoiseonly, fourier_weights=fourier_weights_readnoiseonly)


							np.savez(noisemodl_fpath, fourier_weights_nofluc=fourier_weights_nofluc, mean_cl2d_nofluc=mean_cl2d_nofluc, nl_estFF_nofluc=N_ell_est, \
									N_ell_theta=N_ell_theta)
							
							if pscb_dict['compute_ps_per_quadrant']:
								for q in range(4):
									noisemodl_tailpath_indiv = '/noise_bias_fieldidx'+str(fieldidx)+'_quad'+str(q)+'.npz'
									noisemodl_fpath_indiv = fpath_dict['noisemodl_basepath'] + fpath_dict['noisemodl_run_name'] + noisemodl_tailpath_indiv
									np.savez(noisemodl_fpath_indiv, fourier_weights_nofluc=fourier_weights_nofluc_per_quadrant[q], mean_cl2d_nofluc=mean_cl2d_nofluc_per_quadrant[q], \
										nl_estFF_nofluc=N_ells_est_per_quadrant[q, fieldidx])

				if ciber_cross_ciber:
					if pscb_dict['apply_FW']:
						cbps.FW_image=fourier_weights_cross.copy()
				else:
					if pscb_dict['read_noise_fw'] and ifield_list[fieldidx] != 5:
						cbps.FW_image=fourier_weights_readnoiseonly.copy()
					else:
						cbps.FW_image=fourier_weights_nofluc.copy()
				
					N_ells_est[fieldidx] = N_ell_est

					# if N_ell_theta is not None:
					# 	N_ells_theta[fieldidx] = N_ell_theta

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

			if pscb_dict['iterate_grad_ff'] or pscb_dict['quadoff_grad']: # set gradient_filter, FF_correct to False here because we've already done gradient filtering/FF estimation
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
				if pscb_dict['apply_mask']:
					maskobs = obs*target_mask
				else:
					maskobs = obs

				if pscb_dict['per_quadrant']:
					for q in range(4):

						# maskobs[cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]][mquad==1] -= np.mean(maskobs[cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]][mquad==1])

						mquad = target_mask[cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]]

						obsquad = maskobs[cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]]*mquad
						maskobs[cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]][obsquad!=0] -= np.mean(maskobs[cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]][obsquad!=0])

						# maskobs[cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]][mquad==1] -= np.mean(maskobs[cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]][mquad==1])

				else:
					maskobs[maskobs!=0]-=np.mean(maskobs[maskobs!=0])
				# if not pscb_dict['shut_off_plots']:

				plot_map(maskobs, title='FF corrected, mean subtracted CIBER map ('+cbps.ciber_field_dict[ifield_list[fieldidx]]+')', hipct=99.999, lopct=0., cmap='jet')

				if pscb_dict['apply_mask']:

					maskobs_smooth_2 = gaussian_filter(target_mask*maskobs, sigma=2)
					maskobs_smooth_2[target_mask==1]-=np.mean(maskobs_smooth_2[target_mask==1])

					# maskobs_smooth[maskobs_smooth==0] = np.nan

					plot_map(maskobs_smooth_2, title='Final map smoothed with Gaussian, sigma = 2 pix, ('+cbps.ciber_field_dict[ifield_list[fieldidx]]+')', hipct=99.99, lopct=0., cmap='jet')

					maskobs_smooth = target_mask*gaussian_filter(maskobs, sigma=10)
					maskobs_smooth[target_mask==1]-=np.mean(maskobs_smooth[target_mask==1])
					# if not pscb_dict['shut_off_plots']:
					plot_map(maskobs_smooth, title='FF corrected, smoothed, mean-subtracted CIBER map ('+cbps.ciber_field_dict[ifield_list[fieldidx]]+')', cmap='bwr', vmax=20, vmin=-20)

					hp_image = maskobs-maskobs_smooth
					hp_image[target_mask==1] -= np.mean(hp_image[target_mask==1])
					plot_map(hp_image, title='Data - Low-pass filtered map ('+cbps.ciber_field_dict[ifield_list[fieldidx]]+')', hipct=99.99, lopct=0.01, cmap='jet')

					hp_image2 = maskobs_smooth_2-maskobs_smooth
					hp_image2[target_mask==1] -= np.mean(hp_image2[target_mask==1])
					fig = plot_map(hp_image2, title='2 pix sigma smoothed - Low-pass filtered map ('+cbps.ciber_field_dict[ifield_list[fieldidx]]+')', hipct=99.99, lopct=0.01, cmap='jet', return_fig=True)

					fig.savefig('figures/ciber_unsharp_image_2pix_10pix_sigma_TM'+str(inst)+'_ifield'+str(ifield_list[fieldidx])+'.png', bbox_inches='tight', dpi=300)



			if config_dict['ps_type']=='auto':

				if pscb_dict['max_val_clip']:
					max_val_after_sub = max_vals_ptsrc[fieldidx]
				else:
					max_val_after_sub = None

				if i-config_dict['simidx0'] > 5:
					verb = False 
				else:
					verb = pscb_dict['verbose']

				if pscb_dict['read_noise_fw'] and ifield_list[fieldidx] != 5:
					fourier_weights_use = fourier_weights_readnoiseonly
				else:
					fourier_weights_use = fourier_weights_nofluc


				if pscb_dict['diff_cl2d_fwclip']:

					print('loading cl2d fourier weights for clipping')

					base_nv_path = config.ciber_basepath+'data/noise_model_validation_data/'

					nl_savedir = base_nv_path + 'TM'+str(inst)+'/validationHalfExp/field'+str(ifield_list[fieldidx])+'/nl_flight_diff/'
					nl_save_fpath = nl_savedir+'nls_flight_diff_TM'+str(inst)+'_ifield'+str(ifield_list[fieldidx])+'_'+config_dict['diff_noise_modl_type']+'_quadoffgrad'
					weighted_cl2d_norm = np.load(nl_save_fpath+'.npz')['weighted_cl2d_norm']

					clip_mask = (weighted_cl2d_norm > float_param_dict['diff_cl2d_clipthresh'])

					# plot_map(clip_mask.astype(int), title='fw clip mask')
					fourier_weights_use[clip_mask] = 0.

				if pscb_dict['compute_cl_theta'] and pscb_dict['cut_cl_theta']:
					# which_include=[1, 2, 3, 5, 6, 7] # temporary

					for which_exclude in [0, float_param_dict['n_rad_bins']//2]:
						fourier_weights_use[(theta_masks[which_exclude]==1)] = 0.

					plot_map(fourier_weights_use, title='fourier weights with theta cut')



				lb, processed_ps_nf, cl_proc_err, masked_image, ps2d_unweighted = cbps.compute_processed_power_spectrum(inst, apply_mask=pscb_dict['apply_mask'], \
												 mask=target_mask, image=obs, convert_adufr_sb=False, \
												mkk_correct=pscb_dict['apply_mask'], inv_Mkk=target_invMkk, beam_correct=beam_correct, B_ell=B_ell_field, \
												apply_FW=pscb_dict['apply_FW'], verbose=verb, noise_debias=pscb_dict['noise_debias'], \
												FF_correct=FF_correct, FW_image=fourier_weights_use, \
												 gradient_filter=gradient_filter, save_intermediate_cls=pscb_dict['save_intermediate_cls'], N_ell=N_ells_est[fieldidx], \
												 per_quadrant=pscb_dict['per_quadrant'], max_val_after_sub=max_val_after_sub, \
												 unsharp_mask=pscb_dict['unsharp_mask'], unsharp_pct=float_param_dict['unsharp_pct'], unsharp_sigma=float_param_dict['unsharp_sigma'], \
												 weight_power=float_param_dict['weight_power'], remove_outlier_fac=float_param_dict['remove_outlier_fac'], \
												 clip_clproc_premkk=pscb_dict['clip_clproc_premkk'], clip_outlier_norm_modes=pscb_dict['clip_outlier_norm_modes'], clip_norm_thresh=float_param_dict['clip_norm_thresh'], \
												 fc_sub=pscb_dict['fc_sub'], fc_sub_n_terms=float_param_dict['fc_sub_n_terms'], invert_spliced_matrix=pscb_dict['invert_spliced_matrix'], \
												 compute_mkk_pinv=pscb_dict['compute_mkk_pinv'], theta_masks=theta_masks, N_ell_theta=N_ell_theta, \
												 ifield=ifield_list[fieldidx])

				final_masked_images[fieldidx] = masked_image


				ps2d_unweighted_perfield[fieldidx] = ps2d_unweighted


				if pscb_dict['compute_ps_per_quadrant']:
					processed_ps_per_quad, processed_ps_err_per_quad = [np.zeros((4, cbps_quad.n_ps_bin)) for x in range(2)]
					for q in range(4):
						target_mask_indiv = target_mask[cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]]
						obs_indiv = obs[cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]]
						lb_quad, processed_ps_indiv, cl_proc_err_indiv, _, _ = cbps_quad.compute_processed_power_spectrum(inst, apply_mask=pscb_dict['apply_mask'], \
									 mask=target_mask_indiv, image=obs_indiv, convert_adufr_sb=False, \
									mkk_correct=pscb_dict['apply_mask'], inv_Mkk=target_invMkks_per_quadrant[q], beam_correct=beam_correct, B_ell=B_ell_field_quad, \
									apply_FW=pscb_dict['apply_FW'], verbose=pscb_dict['verbose'], noise_debias=pscb_dict['noise_debias'], \
								 FF_correct=FF_correct, FW_image=fourier_weights_nofluc_per_quadrant[q], \
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


				if ciber_cross_ciber and pscb_dict['apply_tl_regrid']:

					tl_regrid_fpath = fpath_dict['tls_base_path']+'regrid/tl_ptsrc_regrid_TM2_to_TM1'

					if float_param_dict['interp_order'] is not None:
						# tl_regrid_fpath += '_order='+str(float_param_dict['interp_order'])

						tl_regrid_fpath += '_order=1'
						# tl_regrid_fpath += '_conserve_flux'

						if pscb_dict['conserve_flux']:
							tl_regrid_fpath += '_conserve_flux'

						tl_regrid_fpath +='.npz'

						print('OPENING Tl regrid:', tl_regrid_fpath)
							
						# if float_param_dict['']
						# 	tl_regrid_fpath = fpath_dict['tls_base_path']+'regrid/tl_regrid_tm2_to_tm1_order='+str(float_param_dict['interp_order'])+'.npz'

					else:
						tl_regrid_fpath = fpath_dict['tls_base_path']+'regrid/tl_regrid_tm2_to_tm1_ifield4_041323.npz'
					tl_regrid = np.load(tl_regrid_fpath)['tl_regrid']
					
					tl_regrid = np.sqrt(tl_regrid)

					# if pscb_dict['conserve_flux']:
					# 	tl_regrid = np.sqrt(tl_regrid)
				else:
					tl_regrid = None

				print('B_ell_field is ', B_ell_field)

				mask_orig = target_mask*(obs!=0)
				mask_cross = (observed_ims_cross[fieldidx]!=0).astype(int)

				plot_map(mask_orig, title='1.1 band mask before ps')
				plot_map(mask_cross, title='1.8 band mask before ps')

				print(np.sum(mask_orig)/(1024.**2), np.sum(mask_cross)/(1024.**2))
				print(np.sum(mask_orig)/(1024.**2), np.sum(mask_cross)/(1024.**2))

				target_mask_update = mask_orig*mask_cross
				print('target mask orig:', np.sum(joint_maskos[fieldidx])/(1024.**2))

				print('target mask update:', np.sum(target_mask_update)/(1024.**2))

				observed_ims_cross[fieldidx] *= target_mask_update

				lb, processed_ps_nf, cl_proc_err, _, _ = cbps.compute_processed_power_spectrum(inst, apply_mask=pscb_dict['apply_mask'], \
												 mask=target_mask_update, image=obs, cross_image=observed_ims_cross[fieldidx], convert_adufr_sb=False, \
												mkk_correct=pscb_dict['apply_mask'], inv_Mkk=target_invMkk, beam_correct=beam_correct, B_ell=B_ell_field, \
												apply_FW=pscb_dict['apply_FW'], verbose=pscb_dict['verbose'], noise_debias=False, \
												FF_correct=FF_correct, FW_image=None, per_quadrant=pscb_dict['per_quadrant'], \
												gradient_filter=False, tl_regrid=tl_regrid, save_intermediate_cls=pscb_dict['save_intermediate_cls'], N_ell=None, \
												fc_sub=pscb_dict['fc_sub'], fc_sub_n_terms=float_param_dict['fc_sub_n_terms'], clip_outlier_norm_modes=pscb_dict['clip_outlier_norm_modes'], clip_norm_thresh=float_param_dict['clip_norm_thresh'], \
												invert_spliced_matrix=pscb_dict['invert_spliced_matrix'], compute_mkk_pinv=pscb_dict['compute_mkk_pinv'])

				# lb, processed_ps_nf, cl_proc_err, _ = cbps.compute_processed_power_spectrum(inst, apply_mask=pscb_dict['apply_mask'], \
				# 								 mask=target_mask, image=obs, cross_image=observed_ims_cross[fieldidx], convert_adufr_sb=False, \
				# 								mkk_correct=pscb_dict['apply_mask'], inv_Mkk=target_invMkk, beam_correct=beam_correct, B_ell=B_ell_field, \
				# 								apply_FW=False, verbose=pscb_dict['verbose'], noise_debias=False, \
				# 							 FF_correct=FF_correct, FF_image=ff_estimates[fieldidx], FW_image=fourier_weights_cross, \
				# 								 gradient_filter=False, tl_regrid=tl_regrid, save_intermediate_cls=pscb_dict['save_intermediate_cls'], N_ell=N_ells_est[fieldidx])


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

					lb, diff_cl, cl_proc_err_mock, _, _ = cbps.compute_processed_power_spectrum(inst, apply_mask=False, \
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

				if pscb_dict['fc_sub']:

					if float_param_dict['fc_sub_n_terms']==2: # we lose all information on two lowest bandpowers
						processed_ps_nf[:2] = 0.
					if float_param_dict['fc_sub_n_terms']==3: # we lose all information on three lowest bandpowers
						processed_ps_nf[:3] = 0.

				if config_dict['ps_type']=='cross' and ciber_cross_ciber and dcl_cross is not None:
					print('dividing dcl cross by transfer function..')
					dcl_cross /= t_ell_av

				if pscb_dict['compute_ps_per_quadrant']:
					if pscb_dict['verbose']:
						print('correcting per quadrant ps for transfer function..')
					for q in range(4):
						processed_ps_per_quad[q] /= t_ell_av_quad
						processed_ps_err_per_quad[q] /= t_ell_av_quad

				if pscb_dict['save_intermediate_cls']:
					cls_post_tcorr[fieldidx,:] = processed_ps_nf.copy()


			if config_dict['ps_type']=='cross' and ciber_cross_ciber and dcl_cross is not None:

				dcl_cross /= B_ell_field**2
			
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

			if config_dict['ps_type']=='cross' and ciber_cross_ciber and dcl_cross is not None:
				recovered_dcl[i, fieldidx, :] = dcl_cross
			else:
				recovered_dcl[i, fieldidx, :] = cl_proc_err

			if pscb_dict['compute_ps_per_quadrant']:

				recovered_power_spectra_per_quadrant[:, i, fieldidx] = processed_ps_per_quad
				recovered_dcl_per_quadrant[:, i, fieldidx] = processed_ps_err_per_quad

			if pscb_dict['show_ps'] and i-config_dict['simidx0']<10:
				plt.figure(figsize=(6, 5))
				prefac = lb*(lb+1)/(2*np.pi)
				if data_type=='mock':
					plt.errorbar(lb, prefac*true_ps, yerr=prefac*cl_proc_err, fmt='o-', capsize=3, color='k', label='ground truth')
				plt.errorbar(lb, prefac*processed_ps_nf, yerr=prefac*recovered_dcl[i, fieldidx, :], fmt='o-', capsize=3, color='r', label='recovered')

				if config_dict['ps_type']=='cross' and ciber_cross_ciber:
					plt.plot(lb, prefac*recovered_dcl[i, fieldidx, :], linestyle='dashed', color='b', label='Noise uncertainty')

				else:
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

			if pscb_dict['save_fourier_planes']:
				savestr_fp = fpath_dict['sim_test_fpath']+run_name+'/ps2d_unweighted.npz'
				print('Saving fourier planes of maps to ', savestr_fp)
				np.savez(savestr_fp, ps2d_unweighted_perfield=ps2d_unweighted_perfield)

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


