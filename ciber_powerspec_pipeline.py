import numpy as np
import matplotlib 
from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd
import astropy.wcs as wcs
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky
from scipy.ndimage import gaussian_filter
from sklearn import tree
import os

import config
from cross_spectrum_analysis import *
from mkk_parallel import *
from flat_field_est import *
from masking_utils import *
from powerspec_utils import *
from ciber_mocks import *
from filtering_utils import calculate_plane, fit_gradient_to_map
# from ps_pipeline_go import update_dicts
# from ciber_noise_data_utils import iter_sigma_clip_mask, sigma_clip_maskonly

''' 

CIBER_PS_pipeline() is the core module for CIBER power spectrum analysis in Python. 
It contains data processing routines and the basic steps going from flight images in e-/s to
estimates of CIBER auto- and cross-power spectra. 

'''



def update_dicts(list_of_dicts, kwargs, verbose=False):

	for key, value in kwargs.items():
		for indiv_dict in list_of_dicts:
			if key in indiv_dict:
				if verbose:
					print('Setting '+key+' to '+str(value))
				indiv_dict[key] = value 
	return list_of_dicts 

def mean_sub_masked_image(image, mask):
	masked_image = image*mask # dividing by the flat field introduces infinities (and maybe NaNs too?)
	masked_image[np.isinf(masked_image)] = 0
	masked_image[np.isnan(masked_image)] = 0
	unmasked_mean = np.mean(masked_image[mask==1])
	masked_image[mask==1] -= unmasked_mean
	
	return masked_image

def verbprint(verbose, text):
	if verbose:
		print(text)

def compute_fourier_weights(cl2d_all, stdpower=2, mode='mean'):
	
	fw_std = np.std(cl2d_all, axis=0)
	fw_std[fw_std==0] = np.inf
	fourier_weights = 1./(fw_std**stdpower)
	print('max of fourier weights is ', np.max(fourier_weights))
	fourier_weights /= np.max(fourier_weights)
	
	if mode=='mean':
		mean_cl2d = np.mean(cl2d_all, axis=0)
	elif mode=='median':
		mean_cl2d = np.median(cl2d_all, axis=0)
	
	return mean_cl2d, fourier_weights


def load_isl_rms(isl_rms_fpath, masking_maglim, nfield):
	
	isl_rms = np.load(isl_rms_fpath)['isl_sb_rms']
	isl_maglims = np.load(isl_rms_fpath)['mag_lim_list']

	matchidx = np.where(isl_maglims == masking_maglim)[0]
	if len(matchidx)==0:
		print('Cannot find masking magnitude limit of ', masking_maglim, ' in ', isl_rms_fpath)
		mean_isl_rms = np.zeros((nfield))
	else:
		mean_isl_rms = np.mean(isl_rms[:, :, matchidx[0]], axis=0)
		print("mean isl rms of each field is ", mean_isl_rms)
		
	return mean_isl_rms


def stack_masks_ffest(all_masks, ff_stack_min):

	stack_masks = np.zeros_like(all_masks)

	for maskidx in range(len(all_masks)):
		sum_mask = np.sum(all_masks[(np.arange(len(all_masks)) != maskidx), :], axis=0)
		stack_masks[maskidx] = (sum_mask >= ff_stack_min)
		all_masks[maskidx] *= stack_masks[maskidx]

	return all_masks, stack_masks


# def iterative_gradient_ff_solve_bulk(orig_images, niter=3, masks=None, weights_ff=None, plot=False, ff_stack_min=2, masks_ffest=None):



def iterative_gradient_ff_solve(orig_images, niter=3, masks=None, weights_ff=None, plot=False, ff_stack_min=2, masks_ffest=None):
	
	'''
	This function takes a set of images and masks and estimates both the gradients of the maps and a set of stacked flat field estimates. 

	Parameters
	----------
	orig_images : 
	niter : `int'.
	masks : 
	weights_ff : 
	plot :
	ff_stack_min : 
	masks_ffest : 


	Returns
	------- 


	'''
	# maps at end of each iteration
	images = np.array(list(orig_images.copy()))
	nfields = len(images)
	all_coeffs = np.zeros((niter, nfields, 3))    

	final_planes = np.zeros_like(images)

	add_maskfrac_stack = []
	stack_masks = None
	#  --------------- masked images ------------------
	if masks is not None:
		print('Using masks to make masked images, FF stack masks with ff_stack_min = '+str(ff_stack_min)+'..')
		
		stack_masks = np.zeros_like(images)
		
		if masks_ffest is not None:
			masks_ffest, stack_masks = stack_masks_ffest(masks_ffest, ff_stack_min)
			masked_images = np.array([np.ma.array(images[i], mask=(stack_masks[i]==0)) for i in range(len(images))])
		else:
			masks, stack_masks = stack_masks_ffest(masks, ff_stack_min)
			masked_images = np.array([np.ma.array(images[i], mask=(masks[i]==0)) for i in range(len(images))])


		# for imidx, image in enumerate(images):
			# sum_mask = np.sum(masks[(np.arange(len(masks))!=imidx),:], axis=0)
			# stack_masks[imidx] = (sum_mask >= ff_stack_min)
			# mfrac = float(np.sum(masks[imidx]))/float(masks[imidx].shape[0]**2)

			# add_maskfrac_stack.append(mfrac)
			# masks[imidx] *= stack_masks[imidx]

			# mfrac = float(np.sum(masks[imidx]))/float(masks[imidx].shape[0]**2)
			# add_maskfrac_stack[imidx] -= mfrac
		# masked_images = np.array([np.ma.array(images[i], mask=(masks[i]==0)) for i in range(len(images))])
		
	# print('add maskfrac stack is ', add_maskfrac_stack)
	# if means is None:
	#     if masks is not None:
	#         means = [np.ma.mean(im) for im in masked_images]
	#     else:
	#         means = [np.mean(im) for im in images]

	#     print('Means are ', means)
				
	ff_estimates = np.zeros_like(images)

	for n in range(niter):
		print('n = ', n)
		
		# make copy to compute corrected versions of
		running_images = images.copy()
		
		for imidx, image in enumerate(images):

			stack_obs = list(images.copy())                
			del(stack_obs[imidx])

			weights_ff_iter = None
		
			if weights_ff is not None:
				weights_ff_iter = weights_ff[(np.arange(len(images)) != imidx)]
			
			if masks_ffest is not None:
				ff_estimate, _, ff_weights = compute_stack_ff_estimate(stack_obs, masks=masks_ffest[(np.arange(len(masks_ffest))!=imidx),:], weights=weights_ff_iter)
			elif masks is not None:
				ff_estimate, _, ff_weights = compute_stack_ff_estimate(stack_obs, masks=masks[(np.arange(len(masks))!=imidx),:], weights=weights_ff_iter)
			else:
				ff_estimate, _, ff_weights = compute_stack_ff_estimate(stack_obs, masks=None, weights=weights_ff_iter)

			ff_estimates[imidx] = ff_estimate
			running_images[imidx] /= ff_estimate

			if masks_ffest is not None:
				theta, plane = fit_gradient_to_map(running_images[imidx], mask=masks_ffest[imidx])
			elif masks is not None:
				theta, plane = fit_gradient_to_map(running_images[imidx], mask=masks[imidx])
			else:
				theta, plane = fit_gradient_to_map(running_images[imidx])


			running_images[imidx] -= (plane-np.mean(plane))
			
			all_coeffs[n, imidx] = theta[:,0]

			if plot:
				plot_map(plane, title='best fit plane imidx = '+str(imidx))
				plot_map(ff_estimate, title='ff_estimate imidx = '+str(imidx))
	
			if n<niter-1:
				running_images[imidx] *= ff_estimate
			else:
				final_planes[imidx] = plane
		
		images = running_images.copy()

	images[np.isnan(images)] = 0.
	ff_estimates[np.isnan(ff_estimates)] = 1.
	ff_estimates[ff_estimates==0] = 1.
	
	return images, ff_estimates, final_planes, stack_masks, all_coeffs


def compute_residual_source_shot_noise(mag_lim_list, inst, ifield_list, datestr, fpath_dict=None, mode='cib', cmock=None, cbps=None, convert_Vega_to_AB=True, \
									simidx0=0, nsims=100, ifield_plot=4, save=True, \
									  ciber_mock_dirpath='/Volumes/Seagate Backup Plus Drive/Toolkit/Mirror/Richard/ciber_mocks/'):
	
	
	if fpath_dict is not None:
		powerspec_truth_fpath = fpath_dict[mode+'_resid_ps_path']
		trilegal_base_path = fpath_dict['trilegal_base_path']
		cib_realiz_path = fpath_dict['cib_realiz_path']
		bls_fpath = fpath_dict['bls_base_path']+'/bl_est_postage_stamps_TM'+str(inst)+'_081121.npz'
		tempbank_dirpath = fpath_dict['subpixel_psfs_path']+'/'
	else:
		bls_fpath = ciber_mock_dirpath+datestr+'/'
		powerspec_truth_fpath = base_path+'TM'+str(inst)+'/powerspec/'
		print('base path is ', base_path)
		print('powerspec_truth_fpath is ', powerspec_truth_fpath)
		make_fpaths([base_path, powerspec_truth_fpath])
		trilegal_base_path = base_path+'trilegal'
		cib_realiz_path = base_path+'TM'+str(inst)
		bls_base_path = None
		tempbank_dirpath = 'data/subpixel_psfs/'

	print('powerspec truth fpath is ', powerspec_truth_fpath)
	if mode=='isl':
		print('trilegal base path is ', trilegal_base_path)
	elif mode=='cib':
		print('cib realiz path is ', cib_realiz_path)
	print('bls fpath is ', bls_fpath)
	print('temp bank dirpath is ', tempbank_dirpath)
	
	if cbps is None:
		cbps = CIBER_PS_pipeline()
	if cmock is None:
		cmock = ciber_mock(ciberdir='/Users/luminatech/Documents/ciber2/ciber/')

	if mode=='isl':
		isl_sb_rms = np.zeros((nsims, len(ifield_list), len(mag_lim_list)))
		trilegal_band_dict = dict({1:'j_m', 2:'h_m'})
		trilegal_fnu_dict = dict({1:'j_nuInu', 2:'h_nuInu'})
		bandstr, Iarr_str = trilegal_band_dict[inst], trilegal_fnu_dict[inst]
		print('bandstr, Iarr_str = ', bandstr, Iarr_str)
	elif mode=='cib':
		bandstr = 'm_app'

	if bls_fpath is not None:
		B_ells = np.load(bls_fpath)['B_ells_post']
	else:
		B_ells = np.zeros((len(ifield_list), cbps.n_ps_bin))
		for fieldidx, ifield in enumerate(ifield_list):
			B_ells[fieldidx, :] = cbps.load_bl(ifield, inst, inplace=False)


	for setidx in np.arange(simidx0, nsims):
		
		for fieldidx, ifield in enumerate(ifield_list):

			if setidx==0 and fieldidx==0:
				plt.figure()
				plt.loglog(cbps.Mkk_obj.midbin_ell, B_ells[fieldidx]**2)
				plt.show()
			   
			if mode=='isl':
				mock_trilegal = fits.open(trilegal_base_path+'/mock_trilegal_simidx'+str(setidx)+'_'+datestr+'.fits')
				full_src_map = mock_trilegal['trilegal_'+str(cbps.inst_to_band[inst])+'_'+str(ifield)].data
				mock_trilegal_cat = mock_trilegal['tracer_cat_'+str(ifield)].data
				full_tracer_cat = np.array([mock_trilegal_cat['x'], mock_trilegal_cat['y'], mock_trilegal_cat[bandstr], mock_trilegal_cat[Iarr_str]])

			elif mode=='cib':
				# load in mock cib image/tracer catalog

				# mock_gal = fits.open(cib_realiz_path+'/cib_with_tracer_5field_set'+str(setidx)+'_'+datestr+'_TM'+str(inst)+'.fits')
				mock_gal = fits.open(cib_realiz_path+'/cib_with_tracer_with_dpoint_5field_set'+str(setidx)+'_'+datestr+'_TM'+str(inst)+'.fits')
				mock_gal_cat = mock_gal['tracer_cat_'+str(ifield)].data
				full_src_map = mock_gal['cib_'+cbps.inst_to_band[inst]+'_'+str(ifield)].data
				
				# full_src_map = mock_gal['map_'+str(ifield)].data
				# Tracer catalog magnitudes are in AB system
				I_arr_full = cmock.mag_2_nu_Inu(mock_gal_cat[bandstr], inst-1)
				full_tracer_cat = np.array([mock_gal_cat['x'], mock_gal_cat['y'], mock_gal_cat[bandstr], I_arr_full])
			else:
				print('mode not recognized, try again!')
				return None
			
			print('tracer cat has shape ', full_tracer_cat.shape)

			for m, mag_lim in enumerate(mag_lim_list):                
				if convert_Vega_to_AB:
					if m==0:
						print('converting mag limit threshold from Vega to AB to match mock catalog..')
					if setidx==0:
						print('AB mag lim for '+bandstr+' is ', mag_lim+cmock.Vega_to_AB[inst])
					mag_limit_match = mag_lim + cmock.Vega_to_AB[inst]
				else:
					if m==0:
						print('Vega mag lim for '+bandstr+' is ', mag_lim)
					mag_limit_match = mag_lim
					
				bright_cut = np.where(full_tracer_cat[2,:] < mag_limit_match)[0]
				cut_cat = full_tracer_cat[:, bright_cut].transpose()
				if m==0:
					load_precomp_tempbank = True 
				else:
					load_precomp_tempbank = False
				bright_src_map = cmock.make_srcmap_temp_bank(ifield, inst, cut_cat, flux_idx=-1, load_precomp_tempbank=load_precomp_tempbank, \
															tempbank_dirpath=tempbank_dirpath)
				
				diff_full_bright = full_src_map - bright_src_map
				diff_full_bright -= np.mean(diff_full_bright)
				
				lb, cl_diff_maglim, clerr_maglim = get_power_spec(diff_full_bright, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
				
				if mode=='isl':
					isl_sb_rms[setidx, fieldidx, m] = np.std(diff_full_bright)

				# cl_diff_maglim /= B_ell**2
				cl_diff_maglim /= B_ells[fieldidx]**2

				if m==0:
					lb, cl_full, clerr_full = get_power_spec(full_src_map-np.mean(full_src_map), lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
					# cl_full /= B_ell**2
					cl_full /= B_ells[fieldidx]**2

					cl_table = [lb, cl_full]
					names = ['lb', 'cl_full']
				cl_table.append(cl_diff_maglim)
				names.append('cl_maglim_'+str(mag_lim))


			cl_save_fpath = powerspec_truth_fpath+'/cls_'+mode+'_vs_maglim_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(setidx)+'_Vega_magcut.fits'
			
			if save:
				tab = Table(cl_table, names=tuple(names))
				hdu_cl = fits.BinTableHDU(tab, name='cls_'+mode)
				hdr = fits.Header()
				hdr['inst'] = inst
				hdr['ifield'] = ifield
				hdr['sim_idx'] = setidx
				prim = fits.PrimaryHDU(header=hdr)
				hdul = fits.HDUList([prim, hdu_cl])
				hdul.writeto(cl_save_fpath, overwrite=True)
			
			cmock.psf_temp_bank = None
				
			if ifield==ifield_plot:
				cl_table_test = fits.open(cl_save_fpath)['cls_'+mode].data

				plt.figure()
				plt.loglog(lb, lb**2*cl_table_test['cl_full']/(2*np.pi), label='full')
				for m in range(len(mag_lim_list)):
					plt.loglog(lb, lb**2*cl_table_test['cl_maglim_'+str(mag_lim_list[m])]/(2*np.pi), label=cbps.inst_to_band[inst]+' > '+str(mag_lim_list[m]))
				plt.legend(fontsize=14)
				plt.ylim(1e-3, 1e4)
				plt.show()

	if mode=='isl':
		return isl_sb_rms
	elif mode=='cib':
		return None


def estimate_mkk_ffest(cbps, nsims, masks, ifield_list=None, n_split=1, mean_normalizations=None, ff_weights=None, \
					  verbose=False, grad_sub=False, niter=1):
	
	if ifield_list is None:
		ifield_list = [4, 5, 6, 7, 8]
		
	masks = masks.astype(np.float)
		
	nfield = len(ifield_list)
	nsims_perf = nsims//n_split
	nstack = nsims_perf*nfield
	
	if mean_normalizations is None:
		mean_normalizations = np.zeros_like(ifield_list)
			
	maplist_split_shape = (nstack, cbps.dimx, cbps.dimy)
	maplist_split_shapenew = (nfield,nsims_perf, cbps.dimx, cbps.dimy)
	
	# empty aligned object will have nfield, nsims perfield shape. so fftsq iterates over fields and extends by nsims per field.

	print('maplist split shape is ', maplist_split_shape)
	
	cbps.Mkk_obj.precompute_mkk_quantities(precompute_all=True)

	cbps.Mkk_obj.empty_aligned_objs, cbps.Mkk_obj.fft_objs = construct_pyfftw_objs_gen(4, maplist_split_shapenew, axes=(2,3), \
																				  n_fft_obj=2, fft_idx0_list=[0, 2], fft_idx1_list=[1,3])

	n_total_bins = len(cbps.Mkk_obj.unshifted_ringmasks)
	Mkks_per_field = np.zeros((nfield, nsims, n_total_bins, n_total_bins))
	
	# only need to compute ff estimates for one field at a time, so nstack instead of nstack*nfield
	perf_mapshape = (nsims_perf, cbps.dimx, cbps.dimy)
	
	ff_estimates = np.zeros(perf_mapshape)
	
	if ff_weights is None:
		ff_weights = np.ones((nfield))
		
	ff_weight_ims = np.array([ff_weights[m]*mask for m, mask in enumerate(masks)])

	for i in range(n_split):
		print('Split '+str(i+1)+' of '+str(n_split))
		
		# compute noise realizations beforehand so time isn't wasted generating a whole new set each time. We can do this because the ring masks
		# that we multiply by the noise realizations are spatially disjoint, i.e. the union of all of the Fourier masks is the null set. 
		cbps.Mkk_obj.noise = np.random.normal(size=maplist_split_shapenew)+ 1j*np.random.normal(size=maplist_split_shapenew)
		
		for j in range(n_total_bins):
			print('band ', j)
			cbps.Mkk_obj.map_from_phase_realization(j, nsims=nstack)
			
			# now we have 5 x nsims/nsplit maps. add mean normalization to each and apply mask.
			# I think I just need to get the relative mean normalizations correct.
			# does it matter what the relative fluctuation amplitude is? 
			
			for k in range(nfield):

				cbps.Mkk_obj.empty_aligned_objs[1][k].real += mean_normalizations[k]
				cbps.Mkk_obj.empty_aligned_objs[1][k].real *= masks[k]
				
			# for each source estimate the flat field from the other four fields. 
			for k in range(nfield):
				if verbose:
					print('now computing for k = ', k)
				notkrange = [kp for kp in range(nfield) if kp != k]

				ff_weight_ims = np.array([ff_weights[kp]*masks[kp] for kp in notkrange])

				sum_ff_weight_ims = np.sum(ff_weight_ims, axis=0)

				for kp in notkrange:
					cbps.Mkk_obj.empty_aligned_objs[1][kp] *= ff_weights[kp]/mean_normalizations[kp]

				if verbose:
					print('shaaape', np.sum(cbps.Mkk_obj.empty_aligned_objs[1][notkrange,:,:,:].real, axis=0).shape)

				ff_estimates = np.sum(cbps.Mkk_obj.empty_aligned_objs[1][notkrange].real, axis=0)/sum_ff_weight_ims
				ff_estimates[np.isnan(ff_estimates)] = 1.   

				cbps.Mkk_obj.empty_aligned_objs[2][k].real = cbps.Mkk_obj.empty_aligned_objs[1][k].real/ff_estimates

				if grad_sub:
					planes = np.array([fit_gradient_to_map(cbps.Mkk_obj.empty_aligned_objs[2][k,s].real, masks[k])[1] for s in range(nsims_perf)])
#                     plot_map(planes[0], title='plane kp = '+str(kp))
					cbps.Mkk_obj.empty_aligned_objs[2][k].real -= np.array([masks[k]*planes[s] for s in range(nsims_perf)])
#                     plot_map(cbps.Mkk_obj.empty_aligned_objs[2][k,0].real, title='post plane sub kp = '+str(kp))

				unmasked_means = np.array([np.mean(cbps.Mkk_obj.empty_aligned_objs[2][k,s].real[masks[k]==1.]) for s in range(nsims_perf)])                
				cbps.Mkk_obj.empty_aligned_objs[2][k] -= np.array([masks[k]*unmasked_means[s] for s in range(nsims_perf)])
				unmasked_means_sub = np.array([np.mean(cbps.Mkk_obj.empty_aligned_objs[2][k,s].real[masks[k]==1.]) for s in range(nsims_perf)])
				if k==0 and i==0:
					print('unmasked mean pre sub:', unmasked_means)
					print('unmasked mean post subs', unmasked_means_sub)
					plot_map(cbps.Mkk_obj.empty_aligned_objs[2][k,0].real, title='cbps.Mkk_obj.empty_aligned_objs[2][k,0].real post mean sub')

				# undo scaling of off fields to ff_estimate
				for kp in notkrange:
					cbps.Mkk_obj.empty_aligned_objs[1][kp] /= ff_weights[kp]/mean_normalizations[kp]

				if verbose:
					print('unmasked means sub:', unmasked_means_sub)
					print('ff scatter for field'+str(k)+':'+str(np.std([ff_estimate[masks[k]==1.] for ff_estimate in ff_estimates])))
	
			masked_Cl = cbps.Mkk_obj.get_ensemble_angular_autospec_ndim(nsims=nstack, apply_mask=False, obj_idx0=2, obj_idx1=3)
			for k in range(nfield):
				if verbose and i==0:
					print('masked CL field '+str(k)+':')
					print(np.mean(masked_Cl[k*nsims_perf:(k+1)*nsims_perf], axis=0))

				Mkks_per_field[k, i*nsims_perf:(i+1)*nsims_perf, j, :] = np.array(masked_Cl)[k*nsims_perf:(k+1)*nsims_perf]
			
		average_Mkks_perfield = np.mean(Mkks_per_field, axis=1)
		
		for k in range(nfield):
			plot_mkk_matrix(average_Mkks_perfield[k])            


	return average_Mkks_perfield, Mkks_per_field

	
class CIBER_PS_pipeline():

	# fixed quantities for CIBER
	photFactor = np.sqrt(1.2)
	pixsize = 7. # arcseconds
	Npix = 2.03
	inst_to_band = dict({1:'J', 2:'H'})
	inst_to_trilegal_magstr = dict({1:'j_m', 2:'h_m'})
	Vega_to_AB = dict({1:0.91 , 2: 1.39}) # add these numbers to go from Vega to AB magnitude
	iras_color_facs = dict({1:6.4, 2:2.6}) # from MIRIS observations, Onishi++2018
	ciber_field_dict = dict({4:'elat10', 5:'elat30',6:'BootesB', 7:'BootesA', 8:'SWIRE', 'train':'UDS'})
	field_nfrs = dict({4:24, 5:10, 6:29, 7:28, 8:25}) # unique to fourth flight CIBER dataset, elat30 previously noted as 9 frames but Chi says it is 10 (9/17/21)
	frame_period = 1.78 # seconds

	# zl_levels_ciber_fields = dict({2:dict({'elat10': 199.16884143344222, 'BootesA': 106.53451615117534, 'elat30': 147.02015318942148, 'BootesB': 108.62357310134063, 'SWIRE': 90.86593718752026}), \
	#                               1:dict({'NEP':249., 'Lockman':435., 'elat10':558., 'elat30':402., 'BootesB':301., 'BootesA':295., 'SWIRE':245.})})
	# zl_levels_ciber_fields = dict({2:dict({'elat10': 199.16884143344222, 'BootesA': 106.53451615117534, 'elat30': 147.02015318942148, 'BootesB': 108.62357310134063, 'SWIRE': 90.86593718752026}), \
	# 							  1:dict({'NEP':249., 'Lockman':435., 'elat10':515.7, 'elat30':381.4, 'BootesB':328.2, 'BootesA':318.15, 'SWIRE':281.4})})

	# from phil's ZL values
	# zl_levels_ciber_fields = dict({2:dict({'elat10': 552., 'BootesA': 291.0, 'elat30': 398.0, 'BootesB': 297., 'SWIRE': 241.7}), \
	# 							  1:dict({'NEP':249., 'Lockman':435., 'elat10':515.7, 'elat30':381.4, 'BootesB':328.2, 'BootesA':318.15, 'SWIRE':281.4})})

	# for TM2 from mean level in map
	zl_levels_ciber_fields = dict({2:dict({'elat10': 373., 'BootesA': 247.0, 'elat30': 272.7, 'BootesB': 257.1, 'SWIRE': 216.8}), \
								  1:dict({'NEP':249., 'Lockman':435., 'elat10':515.7, 'elat30':381.4, 'BootesB':328.2, 'BootesA':318.15, 'SWIRE':281.4})})

	# self.g1_facs = dict({1:-1.5459, 2:-1.3181}) # multiplying by g1 converts from ADU/fr to e-/s NOTE: these are factory values but not correct
	# these are the latest and greatest calibration factors as of 9/20/2021
	# self.cal_facs = dict({1:-311.35, 2:-121.09}) # ytc values
	# self.cal_facs = dict({1:-170.3608, 2:-57.2057}) # multiplying by cal_facs converts from ADU/fr to nW m^-2 sr^-1

	def __init__(self, \
				base_path='/Users/luminatech/Documents/ciber2/ciber/', \
				data_path=None, \
				dimx=1024, \
				dimy=1024, \
				n_ps_bin=25):
		
		for attr, valu in locals().items():
			setattr(self, attr, valu)


		if data_path is None:
			self.data_path = self.base_path+'data/fluctuation_data/'
			
		self.Mkk_obj = Mkk_bare(dimx=dimx, dimy=dimy, ell_min=180.*(1024./dimx), nbins=n_ps_bin)
		self.Mkk_obj.precompute_mkk_quantities(precompute_all=True)
		self.B_ell = None
		self.map_shape = (self.dimx, self.dimy)
		
		self.g2_facs = dict({1:110./self.Npix, 2:44.2/self.Npix}) # divide by Npix now for correct convention when comparing mean surface brightness (1/31/22)
		
		# self.g1_facs = dict({1:-1.5459, 2:-1.3181}) # multiplying by g1 converts from ADU/fr to e-/s NOTE: these are factory values but not correct

		self.g1_facs = dict({1:-2.67, 2:-3.04}) # these are updated values derived from flight data, similar to those esimated with focus data

		self.cal_facs = dict({1:self.g1_facs[1]*self.g2_facs[1], 2:self.g1_facs[2]*self.g2_facs[2]})
		self.pixlength_in_deg = self.pixsize/3600. # pixel angle measured in degrees
		self.arcsec_pp_to_radian = self.pixlength_in_deg*(np.pi/180.) # conversion from arcsecond per pixel to radian per pixel
		self.Mkk_obj.delta_ell = self.Mkk_obj.binl[1:]-self.Mkk_obj.binl[:-1]
		self.fsky = self.dimx*self.dimy*self.Mkk_obj.arcsec_pp_to_radian**2 / (4*np.pi)

		self.frame_rate = 1./self.frame_period
		self.powerspec_dat_path = self.base_path+'data/powerspec_dat/'

	  
	'''
	Repeated arguments
	------------------
	verbose : boolean, optional
		If True, functions output with many print statements describing what steps are occurring.
		Default is 'False'.
	
	inplace : boolean, optional
		If True, output variable stored in CIBER_PS_pipeline class object. 
		Default is 'True'.
			   
	ifield : integer
		Index of science field. Currently required as a parameter but may want to make optional.
	
	inst : integer
		Indicates which instrument data to use (1 == J band, 2 == H band)
	
	'''
	
	def compute_mkk_matrix(self, mask, nsims=100, n_split=2, inplace=True):
		''' 
		Computes Mkk matrix and its inverse for a given mask.

		Parameters
		----------

		mask : `~np.array~` of type `float` or `int`, shape (cbps.n_ps_bins, cbps.n_ps_bins).
		nsims : `int`.
			Default is 100.
		n_split : `int`.
			Default is 2. 
		inplace : `bool`. 
			Default is True.

		Returns
		-------

		Mkk_matrix : `~np.array~` of type `float`, shape (cbps.n_ps_bins, cbps.n_ps_bins).
		inv_Mkk : `~np.array~` of type `float`, shape (cbps.n_ps_bins, cbps.n_ps_bins).

		'''
		Mkk_matrix = self.Mkk_obj.get_mkk_sim(mask, nsims, n_split=n_split)
		inv_Mkk = compute_inverse_mkk(Mkk_matrix)
		
		if inplace:
			self.Mkk_matrix = Mkk_matrix
			self.inv_Mkk = inv_Mkk
		else:
			return Mkk_matrix, inv_Mkk

	def knox_errors(self, C_ell, N_ell, use_beam_fac=True, B_ell=None, snr=False):
		
		if use_beam_fac and B_ell is None:
			B_ell = self.B_ell
		
		output = compute_knox_errors(self.Mkk_obj.midbin_ell, C_ell, N_ell, self.Mkk_obj.delta_ell, fsky=self.fsky, \
									 B_ell=B_ell, snr=snr)
		
		return output
	
	def noise_model_realization(self, inst, maplist_split_shape, noise_model=None, fft_obj=None, adu_to_sb=True, chisq=True, div_fac=2.83, \
							   read_noise=True, photon_noise=True, shot_sigma_sb=None, image=None, nfr=None, frame_rate=None):
		
		'''
		Generate noise model realization given some noise model.

		Parameters
		----------

		inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band.
		

		Returns
		-------


		'''
		self.noise = np.random.normal(0, 1, size=maplist_split_shape)+ 1j*np.random.normal(0, 1, size=maplist_split_shape)

		rnmaps = None
		snmaps = None
		
		if read_noise:

			# print('min/max of readnoise model in noise_model_realization ', np.min(noise_model), np.max(noise_model))
			if chisq:
				chi2_realiz = (np.random.chisquare(2., size=maplist_split_shape)/div_fac)**2

			if fft_obj is not None:
				if chisq:
					rnmaps = np.sqrt(self.dimx*self.dimy)*fft_obj(self.noise*ifftshift(np.sqrt(chi2_realiz*noise_model)))
				else:
					rnmaps = np.sqrt(self.dimx*self.dimy)*fft_obj(self.noise*ifftshift(np.sqrt(noise_model)))
			else:
				rnmaps = []
				if len(maplist_split_shape)==3:
					for r in range(maplist_split_shape[0]):
						if chisq:
							rnmap = np.sqrt(self.dimx*self.dimy)*ifft2(self.noise[r]*ifftshift(np.sqrt(chi2_realiz[r]*noise_model)))
						else:
							rnmap = np.sqrt(self.dimx*self.dimy)*ifft2(self.noise[r]*ifftshift(np.sqrt(noise_model)))

						rnmaps.append(rnmap)
				else:
					assert len(self.noise.shape) == 2
					assert len(chi2_realiz.shape) == 2
					# single map

					if chisq:
						rnmaps = np.sqrt(self.dimx*self.dimy)*ifft2(self.noise*ifftshift(np.sqrt(chi2_realiz*noise_model)))
					else:
						rnmaps = np.sqrt(self.dimx*self.dimy)*ifft2(self.noise*ifftshift(np.sqrt(noise_model)))

			rnmaps = np.array(rnmaps).real

		
		if photon_noise:
			nsims = 1
			if len(maplist_split_shape)==3:
				nsims = maplist_split_shape[0]
			snmaps = self.compute_shot_noise_maps(inst, image, nsims, shot_sigma_map=shot_sigma_sb, nfr=nfr) # if not providing shot_sigma_sb, need  to provide nfr/frame_rate

		if adu_to_sb and read_noise:
			rnmaps *= self.cal_facs[inst]/self.arcsec_pp_to_radian

		if len(maplist_split_shape)==2:
			if photon_noise:
				snmaps = snmaps[0]

		return rnmaps, snmaps


	def estimate_noise_power_spectrum(self, inst=None, ifield=None, field_nfr=None,  mask=None, apply_mask=True, noise_model=None, noise_model_fpath=None, verbose=False, inplace=True, \
								 nsims = 50, n_split=5, simmap_dc = None, show=False, read_noise=True, photon_noise=True, shot_sigma_sb=None, image=None,\
								  ff_estimate=None, transpose_noise=False, ff_truth=None, mc_ff_estimates = None, gradient_filter=False):

		''' 
		This function generates realizations of the CIBER read + photon noise model and applies the relevant observational effects that are needed to 
		obtain a minimum variance estimator of the noise power spectrum. 

		This is an updated version of the function self.compute_FW_from_noise_sim(), and makes default the pyfftw package for efficient power spectrum computation. 

			
		Parameters
		----------

		ifield : `int`. Index of CIBER field.
			Default is None.
		inst (optional): `int`. 1 for 1.1 um band, 2 for 1.8 um band.
		field_nfr (optional, default=None) : 'int'.
		mask (optional, default=None) : `np.array' of type 'float' or 'int' and shape (self.dimx, self.dimy).
		apply_mask (default=True) : 'bool'.
		noise_model (optional, default=None) : `np.array' of type 'float'.
		noise_model_fpath (optional, default=None) : 'str'. 
		verbose (default=False) : 'bool'.



		Returns
		-------

		mean_nl2d : 
		fourier_weights : 

		'''

		verbprint(verbose, 'Computing fourier weight image from noise sims for TM'+str(inst)+', field '+str(ifield)+'..')
		# verbprint(verbose, 'WARNING : This operation can take a lot of memory if caution is thrown to the wind')
	

		if image is None and photon_noise and shot_sigma_sb is None:
			print('Photon noise is set to True, but no shot noise map provided. Setting image to self.image for use in estimating shot_sigma_sb..')
			image = self.image

		if field_nfr is None:
			if self.field_nfrs is not None and ifield is not None:
				print('Using field nfr, ifield='+str(ifield))
				field_nfr = self.field_nfrs[ifield]

			else:
				print('field nfr = None')
			# else:
				# print('Setting field_nfr = self.nfr = '+str(self.nfr))
				# field_nfr = self.nfr

		verbprint(verbose, 'Field nfr used here for ifield '+str(ifield)+', inst '+str(inst)+' is '+str(field_nfr))

		if noise_model is None and read_noise:
			if noise_model_fpath is None:
				if ifield is not None and inst is not None:
					noise_model_fpath = self.data_path + 'TM' + str(inst) + '/noise_model/noise_model_Cl2D'+str(ifield)+'.fits'
				else:
					print('Noise model not provided, noise_model_fpath not provided, and ifield/inst not provided, need more information..')
					return
			# WEIRDDDDDDDDD fix this?
			noise_fpath=noise_model_base_path+'/noise_model_TM'+str(inst)+'_ifield'+str(ifield)+'_full.fits'
			noise_model = self.load_noise_Cl2D(noise_fpath=noise_fpath, inplace=False, transpose=transpose_noise)

			# noise_model = self.load_noise_Cl2D(noise_fpath='data/fluctuation_data/TM'+str(inst)+'/noiseCl2D/field'+str(ifield)+'_noiseCl2D_110421.fits', inplace=False, transpose=transpose_noise)

		if mask is None and apply_mask:
			verbprint(verbose, 'No mask provided but apply_mask=True, using union of self.strmask and self.maskInst_clean..')
			mask = self.strmask*self.maskInst

		# compute the pixel-wise shot noise estimate given the image (which gets converted to e-/s) and the number of frames used in the integration
		if photon_noise and shot_sigma_sb is None:
			print('photon noise set to True but no shot noise sigma map provided')
			if image is not None:
				print('getting sigma map from input image')
				shot_sigma_sb = self.compute_shot_sigma_map(inst, image=image, nfr=field_nfr)

		# allocate memory for Welfords online algorithm computing mean and variance
		# now we want to allocate the FFT objects in memory so that we can quickly compute the Fourier transforms for many sims at once.
		# first direction (fft_objs[0]->fft_objs[1]) is inverse DFT, second (fft_objs[1]->fft_objs[2]) is regular DFT

		maplist_split_shape = (nsims//n_split, self.dimx, self.dimy)
		empty_aligned_objs, fft_objs = construct_pyfftw_objs(3, maplist_split_shape)
		simmaps = np.zeros(maplist_split_shape)

		sterad_per_pix = (self.pixsize/3600/180*np.pi)**2
		V = self.dimx*self.dimy*sterad_per_pix
		count = 0

		mean_nl2d = np.zeros(self.map_shape)
		M2_nl2d = np.zeros(self.map_shape)

		if gradient_filter:
			if apply_mask:
				dotgrad, Xgrad, mask_rav = precomp_gradient_dat(self.dimx, self.dimy, mask=mask) # precomputed quantities for faster gradient subtraction
			else:
				dotgrad, Xgrad = precomp_gradient_dat(self.dimx, self.dimy, mask=mask) # precomputed quantities for faster gradient subtraction

		for i in range(n_split):
			print('Split '+str(i+1)+' of '+str(n_split)+'..')

			if photon_noise or read_noise:
				rnmaps, snmaps = self.noise_model_realization(inst, maplist_split_shape, noise_model, fft_obj=fft_objs[0],\
													  read_noise=read_noise, photon_noise=photon_noise, shot_sigma_sb=shot_sigma_sb)


			simmaps = np.zeros(maplist_split_shape)
			print('simmaps has shape', simmaps.shape)


			if simmap_dc is not None:
				print('adding constant sky background..')
				simmaps += simmap_dc

			if photon_noise:
				print('adding photon noise')
				simmaps += snmaps 

			if ff_truth is not None:
				print('multiplying simmaps by ff_truth..')
				simmaps *= ff_truth

			if read_noise:
				print('adding read noise..')
				simmaps += rnmaps

			if mc_ff_estimates is not None:
				print('using MC flat field estimate')
				nmc = len(mc_ff_estimates)
				simmaps /= mc_ff_estimates[i%nmc]

			elif ff_estimate is not None:

				if i==0:
					print('std on simmaps before ff estimate is ', np.std(simmaps))
					print('mean, std on ff_estimate are ', np.mean(ff_estimate), np.std(ff_estimate))
				simmaps /= ff_estimate

				if i==0:
					print('std on simmaps after ff estimate is ', np.std(simmaps))


			if gradient_filter:
				verbprint(True, 'Gradient filtering image in the noiiiise bias..')
				# print('mask has shape ', mask.shape)

				for s in range(len(simmaps)):
					theta, plane = fit_gradient_to_map(simmaps[s], mask=mask)
					simmaps[s] -= plane

				# print([np.mean(simmap) for simmap in simmaps])

			if apply_mask and mask is not None:
				simmaps *= mask
				unmasked_means = [np.mean(simmap[mask==1]) for simmap in simmaps]
				simmaps -= np.array([mask*unmasked_mean for unmasked_mean in unmasked_means])
				# print('simmaps have means : ', [np.mean(simmap) for simmap in simmaps])
			else:
				simmaps -= np.array([np.full(self.map_shape, np.mean(simmap)) for simmap in simmaps])
				# print('simmaps have means : ', [np.mean(simmap) for simmap in simmaps])

			fft_objs[1](simmaps*sterad_per_pix)

			cl2ds = np.array([fftshift(dentry*np.conj(dentry)).real for dentry in empty_aligned_objs[2]])

			count, mean_nl2d, M2_nl2d = update_meanvar(count, mean_nl2d, M2_nl2d, cl2ds/V/self.cal_facs[inst]**2)

		mean_nl2d, var_nl2d, svar_nl2d = finalize_meanvar(count, mean_nl2d, M2_nl2d)

		if show:
			plot_map(mean_nl2d, title='mean Nl2d')
			plot_map(1./var_nl2d, title='inverse variance of Nl2d')
			plot_map(mean_nl2d/var_nl2d, title='Fourier-weighted Nl2d')


		fourier_weights = 1./var_nl2d
		
		if show:
			plt.figure(figsize=(8,8))
			plt.title('Fourier weights')
			plt.imshow(fourier_weights, origin='lower', cmap='Greys', norm=matplotlib.colors.LogNorm())
			plt.xticks([], [])
			plt.yticks([], [])
			plt.colorbar()
			plt.show()
		
		if inplace:
			self.FW_image = fourier_weights
			return mean_nl2d
		else:
			return fourier_weights, mean_nl2d

	def generate_custom_sky_clustering(self, inst, dgl_scale_fac=1, gen_ifield=6, ifield=None, cl_pivot_fac_gen=None, power_law_idx=-3.0):


		# field_name_gen = self.ciber_field_dict[gen_ifield]

		if cl_pivot_fac_gen is None:
			cl_pivot_fac_gen = grab_cl_pivot_fac(gen_ifield, inst, dimx=self.dimx, dimy=self.dimy)

		diff_realization = np.zeros(self.map_shape)

		if ifield is not None:
			
			# cl_pivot_fac_gen *= dgl_scale_fac
			# field_name = self.ciber_field_dict[ifield]
			
			cl_pivot_fac = grab_cl_pivot_fac(ifield, inst, dimx=self.dimx, dimy=self.dimy)
			cl_pivot_fac *= dgl_scale_fac

			_, _, diff_realization_varydgl = generate_diffuse_realization(self.dimx, self.dimy, power_law_idx=power_law_idx, scale_fac=cl_pivot_fac)
			diff_realization += diff_realization_varydgl

		else:
			cl_pivot_fac_gen *= dgl_scale_fac

		if cl_pivot_fac_gen > 0:
			_, _, diff_realization_gen = generate_diffuse_realization(self.dimx, self.dimy, power_law_idx=power_law_idx, scale_fac=cl_pivot_fac_gen)
			diff_realization += diff_realization_gen
			
		return diff_realization


	def load_flight_image(self, ifield, inst, flight_fpath=None, verbose=False, inplace=True):
		''' 
		Loads flight image from data release.

		Parameters
		----------
		ifield : `int`. Index of CIBER field.
		inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band.

		'''
		verbprint(verbose, 'Loading flight image from TM'+str(inst)+', field '+str(ifield)+'..')

		if flight_fpath is None:
			flight_fpath = config.exthdpath+'noise_model_validation_data/TM'+str(inst)+'/validationHalfExp/field'+str(ifield)+'/flightMap.FITS'
			# flight_fpath = self.data_path + 'TM'+str(inst)+'/flight/field'+str(ifield)+'_flight.fits'
		
		if inplace:
			self.image = fits.open(flight_fpath)[0].data
		else:
			return fits.open(flight_fpath)[0].data

	def load_regrid_image(self, ifield, inst, cross_type, regrid_fpath=None, verbose=False, inplace=True):
		''' 
		Loads flight image from data release.

		Parameters
		----------
		ifield : `int`. Index of CIBER field.
		inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band.

		'''
		verbprint(verbose, 'Loading flight image from TM'+str(inst)+', field '+str(ifield)+'..')

		if regrid_fpath is None:
			regrid_fpath = self.data_path + 'TM'+str(inst)+'/'+cross_type+'_regrid/field'+str(ifield)+'_flight.fits'
		
		print('Loading regrid image from ', regrid_fpath)
		if inplace:
			self.regrid_image = fits.open(regrid_fpath)[0].data
		else:
			return fits.open(regrid_fpath)[0].data

	def load_gal_density(self, ifield, verbose=True, inplace=False):

		return None

	def grab_noise_model_set(self, ifield_noise_list, inst, noise_model_base_path='data/fluctuation_data/', field_set_shape=None, verbose=False):

		if noise_model_base_path is None:
			noise_model_base_path = 'data/fluctuation_data/TM'+str(inst)+'/noiseCl2D/'
		if field_set_shape is None: 
			field_set_shape = (len(ifield_noise_list), self.dimx, self.dimy)

		read_noise_models = np.zeros(field_set_shape)
		for fieldidx, ifield in enumerate(ifield_noise_list):
			noise_fpath=noise_model_base_path+'/noise_model_TM'+str(inst)+'_ifield'+str(ifield)+'_full.fits'
			noise_cl2d = self.load_noise_Cl2D(ifield, inst, noise_fpath=noise_fpath, inplace=False)
			read_noise_models[fieldidx] = noise_cl2d

		return read_noise_models

		
	def load_noise_Cl2D(self, ifield=None, inst=None, noise_model=None, noise_fpath=None, verbose=False, inplace=True, transpose=False, mode=None, use_abs=False):
		
		''' Loads 2D noise power spectrum from data release
		
		Parameters
		----------

		ifield : `int`. Index of CIBER field.
		inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band.
		noise_fpath : str, optional
			Default is 'None'.
		
		transpose : boolean, optional
			Default is 'True'.
		
		mode : str, optional 
			Default is 'None'.
		
		Returns
		-------
		
		noise_Cl2D : `~numpy.ndarray' of shape (self.dimx, self.dimy)
			2D noise power spectrum. If inplace==True, stored as class object.
		
		'''
		
		if noise_model is None:
			if noise_fpath is None:
				if ifield is not None and inst is not None:
					verbprint(verbose, 'Loading 2D noise power spectrum from TM'+str(inst)+', field '+str(ifield)+'..')            
					noise_fpath = self.data_path+'/TM'+str(inst)+'/noiseCl2D/field'+str(ifield)+'_noiseCl2D_110421.fits'                    
				else:
					print('Out of luck, need more information. Returning None')
					return None

			noise_Cl2D = fits.open(noise_fpath)['noise_model_'+str(ifield)].data

			if transpose:
				print('using the transpose here (why)')
				noise_Cl2D = noise_Cl2D.transpose()


		# remove any NaNs/infs from power spectrum
		noise_Cl2D[np.isnan(noise_Cl2D)] = 0.
		noise_Cl2D[np.isinf(noise_Cl2D)] = 0.

		if inplace:
			if mode=='unmasked':
				self.unmasked_noise_Cl2D = noise_Cl2D
			elif mode=='masked':
				self.masked_noise_Cl2D = noise_Cl2D
			else:
				self.noise_Cl2D = noise_Cl2D
		else:
			return noise_Cl2D
		
		
	def load_FF_image(self, ifield, inst, ff_fpath=None, verbose=False, inplace=True):
		''' 
		Loads flat field estimate from data release 
		
		Parameters
		----------

		ifield : `int`. Index of CIBER field.
		inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band.
		ff_fpath : str, optional
			Default is 'None'.
		
		'''
		verbprint(verbose, 'Loading flat field image from TM'+str(inst)+', field '+str(ifield)+'..')

		if ff_fpath is None:
			ff_fpath = self.data_path + 'TM'+str(inst)+'/FF/field'+str(ifield)+'_FF.fits'
		
		if inplace:
			self.FF_image = fits.open(ff_fpath)[0].data  
		else:
			return fits.open(ff_fpath)[0].data

		
	def load_FW_image(self, ifield, inst, fw_fpath=None, verbose=False, inplace=True):
		''' 
		Loads Fourier weight image estimate from data release.

		Parameters
		----------
		ifield : `int`. Index of CIBER field.
		inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band. 


		'''
		verbprint(verbose, 'Loading fourier weight image estimate from TM'+str(inst)+', field '+str(ifield)+'..')

		if fw_fpath is None:
			fw_fpath = self.data_path + 'TM'+str(inst)+'/FW/field'+str(ifield)+'_FW.fits'
		
		if inplace:
			self.FW_image = fits.open(fw_fpath)[0].data
		else:
			return fits.open(fw_fpath)[0].data
		   
			
	def load_mkk_mats(self, ifield, inst, mkk_fpath=None, verbose=False, inplace=True):
		''' 
		Loads Mkk and inverse Mkk matrices.

		Parameters
		----------
		ifield : `int`. Index of CIBER field.
		inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band.

		'''
		verbprint(verbose, 'Loading Mkk and inverse Mkk matrix estimates from TM'+str(inst)+', field '+str(ifield)+'..')
		 
		if mkk_fpath is None:
			mkk_fpath = self.data_path + 'TM'+str(inst)+'/mkk/field'+str(ifield)+'_mkk_mats.npz'
		
		mkkfile = np.load(mkk_fpath)
		if inplace:
			self.Mkk_matrix = mkkfile['Mkk']
			self.inv_Mkk = mkkfile['inv_Mkk']
		else:
			return mkkfile['Mkk'], mkkfile['inv_Mkk']
		
	def load_mask(self, ifield, inst, masktype=None, mask_fpath=None, verbose=False, inplace=True, instkey=None):
		''' 
		Loads mask.

		Parameters
		----------
		ifield : `int`. 
			Index of CIBER field.
		inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band.
		masktype : 'str'. 
			Type of mask to load. 

		'''
		
		verbprint(verbose, 'Loading mask image from TM'+str(inst)+', field '+str(ifield))

		if mask_fpath is None:
			mask_fpath = self.data_path + 'TM'+str(inst)+'/mask/field'+str(ifield)+'_'+masktype+'.fits'
		

		if instkey is None:
			instkey = 0

		if inplace:
			if masktype=='maskInst_clean':
				self.maskInst_clean = fits.open(mask_fpath)[instkey].data 
			elif masktype=='maskInst':
				self.maskInst = fits.open(mask_fpath)[instkey].data
			elif masktype=='strmask':
				self.strmask = fits.open(mask_fpath)[instkey].data
			elif masktype=='bigmask':
				self.bigmask = fits.open(mask_fpath)[instkey].data
		else:
			return fits.open(mask_fpath)[instkey].data
	
		
	def load_dark_current_template(self, ifield, inst, dc_fpath=None, verbose=False, inplace=True):
		''' 
		Loads dark current template from data release.
		
		Parameters
		----------
		ifield : `int`. Index of CIBER field.
		inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band.
		
		'''
		
		verbprint(verbose, 'Loading dark current template from TM'+str(inst)+'..')

		if dc_fpath is None:
			dc_fpath = self.data_path + 'TM'+str(inst)+'/DCtemplate.fits'
		if inplace:
			self.dc_template = fits.open(dc_fpath)[0].data
		else:
			return fits.open(dc_fpath)[0].data
		
	def load_psf(self, ifield, inst, psf_fpath=None, verbose=False, inplace=True):
		''' 
		Loads PSF estimate from data release.

		Parameters
		----------
		ifield : `int`. Index of CIBER field.
		inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band.

		'''
		
		verbprint(verbose, 'Loading PSF template from TM'+str(inst)+', field '+str(ifield)+'..')

		if psf_fpath is None:
			psf_fpath = self.data_path + 'TM'+str(inst)+'/beam_effect/field'+str(ifield)+'_psf.fits'
		
		# place PSF in CIBER sized image array, some measured PSFs are evaluated over smaller region 
		psf = fits.open(psf_fpath)[0].data
		psf_template = np.zeros(shape=self.map_shape)
		psf_template[:psf.shape[0], :psf.shape[1]] = psf
		
		if inplace:
			self.psf_template = psf_template
		else:
			return psf_template


	def load_bl(self, ifield, inst, inplace=True, verbose=False):

		''' 
		Loads beam transfer function. 

		Parameters
		----------
		ifield : `int`. Index of CIBER field.
		inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band.

		'''

		verbprint(verbose, 'Loading Bl estimates for TM'+str(inst)+', field '+str(ifield)+'..')

		B_ell_path = self.powerspec_dat_path+'B_ells/beam_correction_ifield'+str(ifield)+'_inst'+str(inst)+'.npz'
		
		beamdat = np.load(B_ell_path)

		# make sure multipole bins in saved file match those desired
		if not all(self.Mkk_obj.midbin_ell==beamdat['lb']):
			print('loaded multipole bin centers do not match bins required')
			print(beamdat['lb'])
			print(self.Mkk_obj.midbin_ell)
			if not inplace:
				return None
		else:
			B_ell = beamdat['mean_bl']
		
			if inplace:
				self.B_ell = B_ell
				# print('self B_ell is ', self.B_ell)
			else:
				return B_ell
		
	def load_data_products(self, ifield, inst, load_all=True, flight_image=False, dark_current=False, psf=False, \
						   maskInst=False, strmask=False, mkk_mats=False, mkk_fpath=None, \
						   FW_image=False, FF_image=False, noise_Cl2D=False, beam_correction=False, verbose=True, ifield_psf=None, transpose=False):
		
		'''
		Parameters
		----------
		ifield : `int`. Index of CIBER field.
		inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band.

		'''

		if flight_image or load_all:
			self.load_flight_image(ifield, inst, verbose=verbose)
		if dark_current or load_all:
			self.load_dark_current_template(ifield, inst, verbose=verbose)
		if noise_Cl2D or load_all:
			self.load_noise_Cl2D(ifield, inst, verbose=verbose, transpose=transpose)
		if beam_correction or load_all:
			self.load_bl(ifield, inst, verbose=verbose)
		if mkk_mats or load_all:
			self.load_mkk_mats(ifield, inst, mkk_fpath=mkk_fpath)
		if psf or load_all:
			if ifield_psf is None:
				ifield_psf = ifield
			self.beta, self.rc, self.norm = load_psf_params_dict(inst, ifield=ifield, verbose=verbose)
		if maskInst or load_all:
			self.load_mask(ifield, inst, 'maskInst', verbose=verbose)
		if strmask or load_all:
			self.load_mask(ifield, inst, 'strmask', verbose=verbose)
		if FW_image or load_all:
			self.load_FW_image(ifield, inst, verbose=verbose)
		if FF_image or load_all:
			self.load_FF_image(ifield, inst, verbose=verbose)
			
	def compute_shot_noise_maps(self, inst, image, nsims, shot_sigma_map=None, nfr=None, frame_rate=None, ifield=None, g2_correct=True):

		''' 
		If no shot_sigma_map provided, uses image to generate estimate of shot noise and subsequent photon noise realizations.

		Parameters
		----------
		inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band.

		image : 
		nsims : (int)
		shot_sigma_map (optional) : 
		nfr :
		frame_rate : 
		ifield : 
		g2_correct :


		Returns
		-------

		shot_noise
		

		'''
		if nsims*self.dimx*self.dimy > 1e8:
			print('WARNING -- using a lot of memory at once')
		if nsims*self.dimx*self.dimy > 1e9:
			print('probably too much memory, exiting')
			return None

		if shot_sigma_map is None:
			if nfr is None and ifield is not None:
				nfr = self.field_nfrs[ifield]
			else:
				print('No nfr provided, no shot_sigma_map, and no ifield to get nfr, exiting..')
				return None

			print('computing shot sigma map from within compute_shot_noise_maps')
			shot_sigma_map = self.compute_shot_sigma_map(inst, image, nfr=nfr, frame_rate=frame_rate, g2_correct=g2_correct)


		shot_noise = np.random.normal(0, 1, size=(nsims, self.dimx, self.dimy))*shot_sigma_map

		return shot_noise

	def compute_shot_sigma_map(self, inst, image, nfr=None, frame_rate=None, verbose=False, g2_correct=True):

		''' 
		Assume image is in surface brightness units, returns shot noise map in surface brightness units. 

		Parameters
		----------
		
		inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band.
		image : `~np.array~` of type `float` and shape (dimx, dimy).
		nfr : `int`. Number of frames in exposure.
			Default is None.
		frame_rate : Number of frames per second.
			Default is None.
		verbose : `bool`. 
			Default is False.
		g2_correct : `bool`. If True, convert map from units of surface brightness to units of photocurrent ini e-/s.
			Default is True. 
		
		Returns
		-------
		shot_sigma_map : `~np.array~` of type `float` and shape (dimx, dimy).

		'''

		if nfr is None:
			nfr = self.nfr
		if frame_rate is None:
			frame_rate = self.frame_rate
		if verbose:
			print('nfr, frame_rate:', nfr, frame_rate)
		
		flight_signal = image.copy()
		if g2_correct:
			flight_signal /= self.g2_facs[inst]

		shot_sigma = np.sqrt((np.abs(flight_signal)/(nfr/frame_rate))*((nfr**2+1.)/(nfr**2 - 1.)))

		shot_sigma_map = shot_sigma.copy()
		if g2_correct:
			shot_sigma_map *= self.g2_facs[inst]

		return shot_sigma_map
		
	def compute_noise_power_spectrum(self, inst, noise_Cl2D=None, apply_FW=True, weights=None, verbose=False, inplace=True):
		
		''' 
		For Fourier weighting, need apply_FW to be True, and then you need to specify the weights otherwise it grabs
		self.FW_image.
		
		Parameters
		----------

		inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band.
		noise_Cl2D : 
			Default is None.
		apply_FW : `bool`. 
			Default is True.
		weights : 
			Default is None.
		verbose : `bool`. 
			Default is False.
		inplace : `bool`. If True, saves noise power spectrum to class object
			Default is True.

		Returns
		-------

		Cl_noise : `np.array` of type 'float'. If not inplace, function returns 1-D noise power spectrum.

		'''

		verbprint(verbose, 'Computing 1D noise power spectrum from 2D power spectrum..')
		
		if noise_Cl2D is None:
			verbprint(verbose, 'No Cl2D explicitly provided, using self.noise_Cl2D..')
				
			noise_Cl2D = self.noise_Cl2D.copy()
			
		if apply_FW and weights is None:
			weights = self.FW_image
		elif not apply_FW:
			print('weights provided but apply_FW = False, setting weights to None')
			weights = None
		
		l2d = get_l2d(self.dimx, self.dimy, self.pixsize)
		lbins, Cl_noise, Clerr = azim_average_cl2d(noise_Cl2D*self.cal_facs[inst]**2, l2d, weights=weights, lbinedges=self.Mkk_obj.binl, lbins=self.Mkk_obj.midbin_ell, verbose=verbose)
		
		if verbose:
			print('ell^2 N_ell / 2pi = ', lbins**2*Cl_noise/(2*np.pi))

		if inplace:
			self.N_ell = Cl_noise
		else:
			return Cl_noise
		
		
	def compute_beam_correction_posts(self, ifield, inst, nbins=25, n_fine_bin=10, \
										 psf_postage_stamps=None, beta=None, rc=None, norm=None,\
										tail_path='data/psf_model_dict_updated_081121_ciber.npz', \
										ndim=1):
	
		''' 

		Computes an average beam correction <B_ell> as an average over PSF templates uniformly sampled across the pixel function.

		Inputs
		------
		
		ifield : type 'int'. CIBER field index (4, 5, 6, 7, 8 for science fields)
		band : type 'int'. CIBER band index (inst variable = band + 1)
		nbins (optional): type 'int'. Number of bins in power spectrum. Default = 25. 
		psf_postage_stamps (optional) : 'np.array' of floats with size (fac_upsample, fac_upsample, postage_stamp_dimx, postage_stamp_dimy).
										Grid of precomputed PSF templates.

		beta, rc, norm (optional): type 'float'. Parameters of PSF beta model. Default is None for all three.

		tail_path (optional): file path pointing to PSF beta model parameter dictionary file. 
							Default is 'data/psf_model_dict_updated_081121_ciber.npz'.

		ndim (optional, default=1): type 'int'. Specifies dimension of power spectrum returned (1- or 2-d). Sometimes the 2D Bl is useful

		Returns
		-------

		lb : 'np.array' of type 'float'. Multipole bins of power spectrum
		mean_bl : 'np.array' of size 'nbins'. Beam correction averaged over all PSF postage stamps.
		bls : 'np.array' of size (fac_upsample**2, nbins). List of beam corrections corresponding to individual PSF postage stamps.

		'''
		if psf_postage_stamps is None:
			if beta is None:
				beta, rc, norm = load_psf_params_dict(inst, ifield=ifield, tail_path=tail_path)
			psf_postage_stamps, subpixel_dists = generate_psf_template_bank(beta, rc, norm, n_fine_bin=n_fine_bin)
		
		bls = []
		if ndim==2:
			sum_bl2d = np.zeros(self.map_shape)

		psf_stamp_dim = psf_postage_stamps[0,0].shape[0]
		
		psf_large = np.zeros(self.map_shape)

		for p in range(psf_postage_stamps.shape[0]):
			for q in range(psf_postage_stamps.shape[1]):
	 
				# place postage stamp in middle of image
				psf_large[self.dimx//2-psf_stamp_dim:self.dimx//2, self.dimy//2-psf_stamp_dim:self.dimy//2] = psf_postage_stamps[p,q]

				if ndim==2:
					
					l2d, bl2d = get_power_spectrum_2d(psf_large, pixsize=self.Mkk_obj.pixsize)
					sum_bl2d += bl2d

				lb, clb, clberr = get_power_spec(psf_large - np.mean(psf_large), lbinedges=self.Mkk_obj.binl, lbins=self.Mkk_obj.midbin_ell)    
				bls.append(np.sqrt(clb/np.max(clb)))

		mean_bl = np.mean(bls, axis=0)
		bls = np.array(bls)

		if ndim==2:
			mean_bl2d = sum_bl2d/n_fine_bin**2
			return lb, mean_bl, bls, mean_bl2d
			
		return lb, mean_bl, bls


	def compute_ff_weights(self, inst, mean_norms, ifield_list, photon_noise=True, read_noise_models=None, nread=5, additional_rms=None):
		''' 
		Compute flat field weights based on relative photon noise, read noise and mean normalization across the set of off-fields.
		This weighting is fairly optimal, ignoring the presence of signal fluctuations contributing to flat field error. 

		Inputs
		------

		inst : `int`. 1 == 1.1 um, 2 == 1.8 um.
		mean_norms : `list` of `floats`. Mean surface brightness levels across fields.
		ifield_list : `list` of `ints`. Field indices.
		photon_noise : `bool`. If True, include photon noise in noise estimate.
			Default is True.
		read_noise_models : `list` of `~np.arrays~` of type `float`, shape (self.dimx,self.dimy). Read noise models
			Default is None.
		additional_rms : array_like
			Default is None.
	
		Returns
		-------

		weights : `list` of `floats`. Final weights, normalized to sum to unity.

		'''

		if photon_noise is False and read_noise_models is None:
			print("Neither photon noise nor read noise models provided, weighting fields by mean brightness only..")
			weights = mean_norms
			return weights/np.sum(weights)

		rms_read = np.zeros_like(mean_norms)
		rms_phot = np.zeros_like(mean_norms)
		if additional_rms is None:
			additional_rms = np.zeros_like(mean_norms)

		if photon_noise:
			for i, ifield in enumerate(ifield_list):
				shot_sigma_sb = self.compute_shot_sigma_map(inst, image=mean_norms[i]*np.ones((10, 10)), nfr=self.field_nfrs[ifield])         
				rms_phot[i] = shot_sigma_sb[0,0]

		if read_noise_models is not None:
			# compute some read noise realizations and measure mean rms. probably don't need that many realizations
			for i, ifield in enumerate(ifield_list):

				rnmaps, _ = self.noise_model_realization(inst, (nread, self.dimx, self.dimy), read_noise_models[i],\
												  read_noise=True, photon_noise=False)

				rms_read[i] = np.std(rnmaps)

		weights = (mean_norms/(np.sqrt(rms_phot**2 + rms_read**2+additional_rms**2)))**2
		print('ff weights are computed to be ', weights)

		weights /= np.sum(weights)

		return weights



	def calculate_transfer_function(self, nsims, \
						 niter_grad_ff=1, dgl_scale_fac=5., indiv_ifield=6, apply_FF=False,\
						 grad_sub=True, FF_image=None, FF_fpath=None, plot=False, \
						 mean_normalizations=None, ifield_list=None, \
						 masks=None, inv_Mkks=None, niter=5, inst=1, maskidx=0, ff_stack_min=2, \
						 apply_mask_ffest=True, noise_scalefac=1., off_fac=None, \
						 include_dgl=True, cl_pivot_fac_gen=None, power_law_idx=-3.0, \
						 ff_bias_correct=True, smooth_sig=None):

		''' 
		Estimate the transfer function of gradient filtering by passing through a number of sky realizations and measuring the ratio of input/output power spectra.

		I would like to estimate the transfer function that comes from the flat field + gradient corrections. For this I think
		we can assume the flat field errors from instrument noise fluctuations and that from the underlying signals are independent.
		So, generate mean normalization + fluctuation signal, add gradient and apply flat field, then estimate flat fields/gradients. 
		What is the resulting power spectrum of the maps?

		Parameters
		----------
		inst :
		nsims : 
		load_ptsrc_cib : 
		niter_grad_ff : 

		
		Returns
		-------
		lb : 
		t_ell_av : Mean transfer function from nsims sky realizations.
		t_ell_stderr : Error on mean transfer functino from nsims sky realizations.
		t_ells : Transfer functions of all realizations
		cls_orig : Power spectra of original maps.
		cls_filt : Power spectra of filtered maps.
		'''

		if apply_FF:
			if ifield_list is None:
				ifield_list = [4, 5, 6, 7, 8]

			ps_set_shape = (nsims, len(ifield_list), self.n_ps_bin)
		else:
			ps_set_shape = (nsims, self.n_ps_bin)
		
		cls_orig = np.zeros(ps_set_shape)
		cls_filt = np.zeros(ps_set_shape)
		t_ells = np.zeros(ps_set_shape)


		if apply_FF:

			if FF_image is None:
				if FF_fpath is not None:
					FF_image = gaussian_filter(fits.open(FF_fpath)[0].data, sigma=5.)

				else:
					print('need to either provide file path to FF or FF image..')

			if plot and FF_image is not None:
				plot_map(FF_image, title='FF image')

			# here doing sets of CIBER mocks
			for setidx in range(nsims):
				print('set ', setidx, 'of ', nsims)

				mock_sky_ims = []
				for fieldidx, ifield in enumerate(ifield_list):
					_, ps, mock_cib_im = generate_diffuse_realization(self.dimx, self.dimy, power_law_idx=0.0, scale_fac=3e8)

					if include_dgl:
						# diff_realization = self.generate_custom_sky_clustering(dgl_scale_fac=5., ifield=ifield, gen_ifield=6, cl_pivot_fac_gen=cl_pivot_fac_gen)
						diff_realization = self.generate_custom_sky_clustering(inst, dgl_scale_fac=5., gen_ifield=6, cl_pivot_fac_gen=cl_pivot_fac_gen, power_law_idx=power_law_idx)
					
						sky_realiz = mock_cib_im+diff_realization
					else:
						sky_realiz = mock_cib_im

					if smooth_sig is not None:
						sky_realiz = gaussian_filter(sky_realiz, sigma=smooth_sig)

					lb, cl_orig, clerr_orig = get_power_spec(sky_realiz-np.mean(sky_realiz), lbinedges=self.Mkk_obj.binl, lbins=self.Mkk_obj.midbin_ell)
					cls_orig[setidx, fieldidx, :] = cl_orig



					if mean_normalizations is not None:
						# if setidx==0:
							# print('adding mean normalization')
						sky_realiz += mean_normalizations[fieldidx]

					if plot and fieldidx==0:
						plot_map(sky_realiz, title='Mock sky realization, set '+str(setidx))
						plt.figure(figsize=(6, 5))
						plt.plot(lb, lb*(lb+1)*cl_orig/(2*np.pi))
						plt.yscale('log')
						plt.xscale('log')
						plt.title('cl_orig:')
						plt.xlabel('$\\ell$', fontsize=18)
						plt.ylabel('$D_{\\ell}$', fontsize=18)
						plt.show()
						# plot_map(sky_realiz, title='sky realization')

					if FF_image is not None:
						sky_realiz *= FF_image 

					if plot and fieldidx==0:
						if masks is not None:
							plot_map(sky_realiz*masks[fieldidx], title='Mock sky realization x flat field x mask, ifield='+str(ifield))

					mock_sky_ims.append(sky_realiz)

				mock_sky_ims = np.array(mock_sky_ims)

				if masks is not None:
					mask_fractions_init = np.array([float(np.sum(masks[fieldidx]))/float(self.dimx**2) for fieldidx in range(len(ifield_list))])
					print('mask fractions are initially ', mask_fractions_init)

					masks, stack_masks = stack_masks_ffest(masks, ff_stack_min)
					mask_fractions = np.array([float(np.sum(masks[fieldidx]))/float(self.dimx**2) for fieldidx in range(len(ifield_list))])
					obs_levels = [np.mean(mock_sky_ims[fieldidx][masks[fieldidx] != 0]) for fieldidx in range(len(ifield_list))]
					
					print('obs levels are ', obs_levels)
					print('mask fractions are ', mask_fractions)
				else:
					obs_levels = [np.mean(mock_sky_ims[fieldidx]) for fieldidx in range(len(ifield_list))]
					mask_fractions=None
				
				weights_photonly = self.compute_ff_weights(inst, obs_levels, read_noise_models=None, ifield_list=ifield_list, photon_noise=False)
				
				if not grad_sub and not apply_mask_ffest:
					print('setting mask fractions to None..')
					mask_fractions = None

				if ff_bias_correct:
					ff_biases = compute_ff_bias(obs_levels, weights=weights_photonly, mask_fractions=mask_fractions)
					# print('weights are ', weights_photonly)
					# print('ff_biases are ', ff_biases)

				if apply_FF and grad_sub:

					processed_ims, ff_estimates, final_planes, stack_masks_bright, coeffs_vs_niter = iterative_gradient_ff_solve(mock_sky_ims, niter=niter, masks=masks, ff_stack_min=ff_stack_min)
					if masks is not None:
						masks *= stack_masks_bright

					mask_fractions_new = np.array([float(np.sum(masks[fieldidx]))/float(self.dimx**2) for fieldidx in range(len(ifield_list))])
					obs_levels_new = [np.mean(processed_ims[fieldidx][masks[fieldidx] != 0]) for fieldidx in range(len(ifield_list))]
					print('obs levels are ', obs_levels_new)
					print('mask fractions are ', mask_fractions_new)
					weights_photonly = self.compute_ff_weights(inst, obs_levels_new, read_noise_models=None, ifield_list=ifield_list, photon_noise=False)
					if ff_bias_correct:
						ff_biases_new = compute_ff_bias(obs_levels_new, weights=weights_photonly, mask_fractions=mask_fractions_new)
						print('updated ff biases are ', ff_biases_new)

					if setidx==0:
						plot_map(final_planes[0]-np.mean(final_planes[0]), title='Final gradient estimate, ifield=4')


				elif not grad_sub:

					processed_ims = mock_sky_ims.copy()
					for fieldidx, ifield in enumerate(ifield_list):
						stack_obs = list(mock_sky_ims.copy())      

						del(stack_obs[fieldidx])
						
						if off_fac is not None:
							for s in range(len(stack_obs)):
								stack_obs[s] += off_fac*mean_normalizations[fieldidx]*FF_image

						weights_ff_test = weights_photonly[(np.arange(len(processed_ims)) != fieldidx)]

						if masks is not None:

							if apply_mask_ffest:
								ff_estimate, _, ff_weights = compute_stack_ff_estimate(stack_obs, masks=masks[(np.arange(len(masks))!=fieldidx),:], weights=weights_ff_test)
							else:
								ff_estimate, _, ff_weights = compute_stack_ff_estimate(stack_obs, masks=None, weights=weights_ff_test)

						else:
							ff_estimate, _, ff_weights = compute_stack_ff_estimate(stack_obs, masks=None, weights=weights_ff_test)
						
						ff_estimate[np.isnan(ff_estimate)] = 1.
						ff_estimate[ff_estimate==0] = 1.
						
						if fieldidx==0 and setidx==0:
							plot_map(ff_estimate*masks[fieldidx], title='ffestimate stack ff')
							if FF_image is not None:
								plot_map((FF_image-ff_estimate)*masks[fieldidx], title='true - estimated FF')

						if fieldidx==0 and FF_image is not None:
							plt.figure()
							fferr_nonzrav = (masks[fieldidx]*(FF_image-ff_estimate)).ravel()
							fferr_nonzrav = fferr_nonzrav[(fferr_nonzrav!=1)*(~np.isnan(fferr_nonzrav))*(fferr_nonzrav!=0)]
							plt.hist(fferr_nonzrav, bins=20)
							plt.title(np.round(np.nanstd(fferr_nonzrav), 3), fontsize=18)
							plt.yscale('log')
							plt.show()
						processed_ims[fieldidx] /= ff_estimate
				
				# if masks is not None:
				# 	masks *= stack_masks_bright



				for fieldidx, ifield in enumerate(ifield_list):	

					if masks is not None:
						processed_ims[fieldidx][masks[fieldidx] != 0] -= np.mean( processed_ims[fieldidx][masks[fieldidx] != 0])

					# print('mean of processed im is ', np.mean(processed_ims[fieldidx]*masks[fieldidx]))

					if plot:
						if masks is not None:
							plot_map(processed_ims[fieldidx]*masks[fieldidx], title='Processed map x mask, ifield = '+str(ifield))
						else:
							plot_map(processed_ims[fieldidx], title='Processed map, ifield = '+str(ifield))

					if masks is not None:
						lb, cl_filt, clerr_filt = get_power_spec(processed_ims[fieldidx]*masks[fieldidx], lbinedges=self.Mkk_obj.binl, lbins=self.Mkk_obj.midbin_ell)
						
						# noise bias subtraction 
						if inv_Mkks is not None:
							cl_filt = np.dot(inv_Mkks[fieldidx].transpose(), cl_filt)
					else:
						lb, cl_filt, clerr_filt = get_power_spec(processed_ims[fieldidx], lbinedges=self.Mkk_obj.binl, lbins=self.Mkk_obj.midbin_ell)


					if off_fac is not None:
						obs_levels_off_fac = np.array(obs_levels).copy()
						# obs_levels_off_fac = np.array(obs_levels_off_fac)

						for s in range(len(obs_levels_off_fac)):
							if s != fieldidx:
								obs_levels_off_fac[s] = off_fac*mean_normalizations[s]

						if ff_bias_correct:
							ff_biases_off = compute_ff_bias(obs_levels_off_fac, weights=weights_photonly, mask_fractions=mask_fractions)[fieldidx]
							print('fieldidx = ', fieldidx, 'ff bias off ', ff_biases_off)
							cl_filt /= ff_biases_off
					else:
						if ff_bias_correct:
							cl_filt /= ff_biases[fieldidx]


					cls_filt[setidx, fieldidx, :] = cl_filt
					t_ell_indiv = cl_filt/cls_orig[setidx, fieldidx, :]
					t_ells[setidx, fieldidx, :] = t_ell_indiv

				if setidx%10==0:
					print('for setidx = ', setidx, 't_ells = ', t_ells[setidx])
					print('for setidx = ', setidx, 'mean t_ell = ', np.mean(t_ells[setidx], axis=0))

		else:

			for i in range(nsims):

				_, _, diff_realization = generate_diffuse_realization(self.dimx, self.dimy, power_law_idx=0.0, scale_fac=3e8)

				if include_dgl:
					dgl_realization = self.generate_custom_sky_clustering(inst, dgl_scale_fac=dgl_scale_fac, gen_ifield=indiv_ifield, cl_pivot_fac_gen=cl_pivot_fac_gen)
					diff_realization += dgl_realization 
   
				# else:
					# print('need to load CIB here')

				if smooth_sig is not None:
					diff_realization = gaussian_filter(diff_realization, sigma=smooth_sig)

				if plot:
					plot_map(diff_realization, title='Mock sky realization, set '+str(i))

				lb, cl_orig, clerr_orig = get_power_spec(diff_realization - np.mean(diff_realization), lbinedges=self.Mkk_obj.binl, lbins=self.Mkk_obj.midbin_ell)

				if grad_sub:
					if masks is not None:
						theta, plane = fit_gradient_to_map(diff_realization, masks[maskidx])
					else:
						theta, plane = fit_gradient_to_map(diff_realization)

					diff_realization -= plane

				if masks is not None:
					diff_realization[masks[maskidx] != 0] -= np.mean(diff_realization[masks[maskidx] != 0])
					diff_realization *= masks[maskidx]
					# print('mean of processed im is ', np.mean(diff_realization*masks[maskidx]))
					if i==0:
						plot_map(diff_realization, title='Masked, gradient subtracted signal')
				else:
					diff_realization -= np.mean(diff_realization)

				# lb, cl_filt, clerr_filt = get_power_spec(diff_realization - np.mean(diff_realization), lbinedges=self.Mkk_obj.binl, lbins=self.Mkk_obj.midbin_ell)
				
				lb, cl_filt, clerr_filt = get_power_spec(diff_realization, lbinedges=self.Mkk_obj.binl, lbins=self.Mkk_obj.midbin_ell)
				
				if inv_Mkks is not None:
					cl_filt = np.dot(inv_Mkks[maskidx].transpose(), cl_filt)

				cls_orig[i] = cl_orig
				cls_filt[i] = cl_filt
				t_ell_indiv = cl_filt/cl_orig 

				t_ells[i] = t_ell_indiv

		t_ell_av = np.median(t_ells, axis=0)
		t_ell_stderr = np.std(t_ells, axis=0)/np.sqrt(t_ells.shape[0])


		return lb, t_ell_av, t_ell_stderr, t_ells, cls_orig, cls_filt

  
	def compute_processed_power_spectrum(self, inst, bare_bones=False, mask=None, N_ell=None, B_ell=None, inv_Mkk=None,\
										 image=None, cross_image=None, ff_bias_correct=None, FF_image=None, verbose=False, \
										 FW_image=None, **kwargs):
		
		''' 
		Computes processed power spectrum for input map/mask/etc.

		Parameters
		----------
		inst : 
		bare_bones : If set to True, computes simple power spectrum (assume no mask, weighting, noise bias subtraction, etc.)
			Default is False.
		mask :
			Default is None.
		N_ell : Noise bias power spectrum
			Default is None.
		B_ell : Beam correction
			Default is None.
		inv_Mkk : Inverse mode coupling matrix.
			Default is None.
		image : If None, grabs image from cbps.image
			Default is None.
		ff_bias_correct : 
		FF_image :
		verbose : 
			Default is False.

		Returns
		-------
		lbins : 
		cl_proc :
		cl_proc_err : 
		masked_image :

		'''

		self.masked_Cl_pre_Nl_correct = None
		self.masked_Cl_post_Nl_correct = None
		self.cl_post_mkk_pre_Bl = None

		pps_dict = dict({'apply_mask':True, 'mkk_correct':True, 'beam_correct':True, 'FF_correct':True, 'gradient_filter':False, 'apply_FW':True, \
			'noise_debias':True, 'convert_adufr_sb':True, 'save_intermediate_cls':True})

		pps_dict = update_dicts([pps_dict], kwargs)[0]

		if bare_bones:
			pps_dict['FF_correct'] = False 
			pps_dict['apply_mask'] = False
			pps_dict['noise_debias'] = False
			pps_dict['mkk_correct'] = False
			pps_dict['beam_correct'] = False 

		if image is None:
			image = self.image.copy()
		else:
			xim = image.copy()
			image = xim.copy()

		# image = image.copy()
		
		if pps_dict['convert_adufr_sb']:
			image *= self.cal_facs[inst] # convert from ADU/fr to nW m^-2 sr^-1
		
		if pps_dict['FF_correct']:
			verbprint(True, 'Applying flat field correction..')

			if FF_image is not None:
				verbprint(verbose, 'Mean of FF_image is '+str(np.mean(FF_image))+' with standard deviation '+str(np.std(FF_image)))

				image = image / FF_image

		if mask is None and pps_dict['apply_mask']:
			verbprint(verbose, 'Getting mask from maskinst and strmask')
			mask = self.maskInst*self.strmask # load mask


		if pps_dict['gradient_filter']: 
			verbprint(True, 'Gradient filtering image..')
			theta, plane = fit_gradient_to_map(image, mask=mask)
			image -= plane

		cross_masked_image = None # only set to something if cross spectrum being calculated and cross map provided

		if pps_dict['apply_mask']:

			verbprint(True, 'Applying mask..')
			masked_image = mean_sub_masked_image(image, mask) # apply mask and mean subtract unmasked pixels
			if cross_image is not None:
				cross_masked_image = mean_sub_masked_image(cross_image, mask)

		else:
			verbprint(True, 'No masking, subtracting image by its mean..')
			masked_image = image - np.mean(image)
			mask = None
			if cross_image is not None:
				cross_masked_image = cross_image - np.mean(cross_image)

		verbprint(verbose, 'Mean of masked image is '+str(np.mean(masked_image)))
		 
		weights=None
		if pps_dict['apply_FW']:
			verbprint(True, 'Using Fourier weights..')
			if FW_image is not None:
				weights = FW_image
			else:
				weights = self.FW_image # this feeds Fourier weights into get_power_spec, which uses them with Cl2D
			
		lbins, cl_proc, cl_proc_err = get_power_spec(masked_image, map_b=cross_masked_image, mask=None, weights=weights, lbinedges=self.Mkk_obj.binl, lbins=self.Mkk_obj.midbin_ell)
		
		# lbins, cl_proc, cl_proc_err = get_power_spec(masked_image, mask=None, weights=weights, lbinedges=self.Mkk_obj.binl, lbins=self.Mkk_obj.midbin_ell)

		
		if pps_dict['save_intermediate_cls']:
			self.masked_Cl_pre_Nl_correct = cl_proc.copy()

		verbprint(verbose, 'cl_proc after get_power spec is ')
		verbprint(verbose, cl_proc)
			
		if pps_dict['noise_debias']: # subtract noise bias
			verbprint(True, 'Applying noise bias..')

			if N_ell is None:
				cl_proc -= self.N_ell
			else:
				cl_proc -= N_ell     
				
			verbprint(verbose, 'after noise subtraction, ')
			verbprint(verbose, cl_proc)

			if pps_dict['save_intermediate_cls']:
				self.masked_Cl_post_Nl_correct = cl_proc.copy()

  
		if pps_dict['mkk_correct']: # apply inverse mode coupling matrix to masked power spectrum
			verbprint(True, 'Applying Mkk correction..')

			if inv_Mkk is None:
				print('inv_Mkk is None, using self.inv_Mkk!!!')
				cl_proc = np.dot(self.inv_Mkk.transpose(), cl_proc)
				# cl_proc_err= np.dot(np.abs(self.inv_Mkk.transpose()), cl_proc_err)
			else:
				cl_proc = np.dot(inv_Mkk.transpose(), cl_proc)
				cl_err_inv_Mkk_dot = np.dot(np.abs(inv_Mkk.transpose())**2, cl_proc_err**2)
				cl_proc_err = np.sqrt(cl_err_inv_Mkk_dot)

				# cl_proc_err = np.sqrt(np.sum(cl_err_inv_Mkk_dot**2))
				# cl_proc_err = np.sqrt(np.sum((np.dot(np.abs(inv_Mkk.transpose()))))
				# cl_proc_err= np.sqrt(np.dot(np.abs(inv_Mkk.transpose()), cl_proc_err**2))
				# cl_proc_err= np.sqrt(np.dot(np.abs(inv_Mkk.transpose()), cl_proc_err**2))

			verbprint(verbose, 'After Mkk correction, cl_proc is ')
			verbprint(verbose, cl_proc)
			verbprint(verbose, 'After Mkk correction, cl_proc_err is ')
			verbprint(verbose, cl_proc_err)

			if pps_dict['save_intermediate_cls']:
				self.cl_post_mkk_pre_Bl = cl_proc.copy()
							
		if pps_dict['beam_correct']: # undo the effects of the PSF by dividing power spectrum by B_ell
			verbprint(True, 'Applying beam correction..')

			if B_ell is None:
				B_ell = self.B_ell
			assert len(B_ell)==len(cl_proc)
			cl_proc /= B_ell**2
			cl_proc_err /= B_ell**2

		verbprint(verbose, 'Processed angular power spectrum is ')
		verbprint(verbose, cl_proc)
		verbprint(verbose, 'Processed angular power spectrum error is ')
		verbprint(verbose, cl_proc_err)


		return lbins, cl_proc, cl_proc_err, masked_image

	def estimate_b_ell_from_maps(self, inst, ifield_list, ciber_maps, simidx=0, ell_norm=5000, plot=False, save=False, \
										niter = 5, ff_stack_min=2, data_type='mock', datestr_mock='062322', datestr_trilegal=None, \
										ciber_mock_fpath='/Volumes/Seagate Backup Plus Drive/Toolkit/Mirror/Richard/ciber_mocks/', \
										mask_base_path=None, full_mask_tail='maglim_17_Vega_test', bright_mask_tail='maglim_11_Vega_test', \
										iterate_grad_ff=True, mkk_correct=False, mkk_base_path=None):

		
		''' 
		This function estimates the beam correction function from the maps. This is done by computing the power spectrum 
		of the point source dominated maps and then subtracting the power spectrum of masked maps to remove noise bias. The final power
		spectrum is then normalized to unity at a pivot ell value. 
		
		Note that there is some numerical weirdness with doing this on individual realizations when normalizing the power spectrum
		at lower ell. This is fixed by either assuming the beam transfer function is >0.999 below some multipole, or by averaging over several realizations.
		
		Parameters
		----------
		
		inst : 'int'. Instrument index, 1==J, 2==H
		ifield_list : 'list' of 'ints'. 
		pixsize : 'float'.
			Default is 7.
		J_bright_mag : 'float'.
		J_tot_mag : 'float'.
		nsim : 'int'. Number of realizations.
			Default is 1.
		ell_norm : 'float'. Multipole used to normalize power spectra for B_ell estimate. 
			Default is 10000. 
		niter : 'int'. Number of iterations on FF/grad estimation
		ff_stack_min : 'int'. Minimum number of off-field measurements for each pixel.
		data_type : 'string'. Distinguishes between observed and mock data.
			Default is 'mock'.
		
		Returns
		-------
		lb : 'np.array' of 'floats'. Power spectrum multipole bin centers.
		diff_norm : normalized profiles
		
		'''
		
		print('full mask tail:', full_mask_tail)
		print('bright mask tail:', bright_mask_tail)
		ps_set_shape = (len(ifield_list), self.n_ps_bin)
		imarray_shape = (len(ifield_list),self.dimx, self.dimy)
		diff_norm_array, cls_masked_tot, cls_masked_tot_bright = [np.zeros(ps_set_shape) for x in range(3)]
		
		if datestr_trilegal is None:
			datestr_trilegal = datestr_mock
			
		if mask_base_path is None:
			mask_base_path = base_path+'TM'+str(inst)+'/masks/'
			
		print('simidx is ', simidx)

		tot_masks, bright_masks, tot_obs = [np.zeros(imarray_shape) for x in range(3)]
		inv_Mkks_bright, inv_Mkks_tot = [np.zeros((len(ifield_list), self.n_ps_bin, self.n_ps_bin)) for x in range(2)]
		for fieldidx, ifield in enumerate(ifield_list):
			
			print('ifield = ', ifield)
			if data_type=='mock':
				joint_mask = fits.open(mask_base_path+'joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(simidx)+'_'+full_mask_tail+'.fits')['joint_mask_'+str(ifield)].data
				bright_mask = fits.open(mask_base_path+'joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(simidx)+'_'+bright_mask_tail+'.fits')['joint_mask_'+str(ifield)].data
				tot_masks[fieldidx] = joint_mask
				bright_masks[fieldidx] = bright_mask

				if mkk_correct:
					bright_inv_Mkk_fpath = mkk_base_path+'mkk_estimate_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(simidx)+'_'+bright_mask_tail+'.fits'
					inv_Mkks_bright[fieldidx] = fits.open(bright_inv_Mkk_fpath)['inv_Mkk_'+str(ifield)].data

					tot_inv_Mkk_fpath = mkk_base_path+'mkk_estimate_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(simidx)+'_'+full_mask_tail+'.fits'
					inv_Mkks_tot[fieldidx] = fits.open(tot_inv_Mkk_fpath)['inv_Mkk_'+str(ifield)].data

				
				if fieldidx==0:
					plot_map(tot_masks[fieldidx]*ciber_maps[fieldidx], title='tot mask x ciber maps')
					plot_map(bright_masks[fieldidx]*ciber_maps[fieldidx], title='bright mask x ciber maps')
					
			else:
				self.load_data_products(ifield, inst, verbose=False)

				# joint_mask = fits.open(mask_base_path+'joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+full_mask_tail+'.fits')['joint_mask_'+str(ifield)].data
				# bright_mask = fits.open(mask_base_path+'joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+bright_mask_tail+'.fits')['joint_mask_'+str(ifield)].data
				joint_mask = fits.open(mask_base_path+'joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+full_mask_tail+'.fits')[1].data
				bright_mask = fits.open(mask_base_path+'joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+bright_mask_tail+'.fits')[1].data

				if mkk_correct:
					bright_inv_Mkk_fpath = mkk_base_path+'mkk_estimate_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+bright_mask_tail+'.fits'
					inv_Mkks_bright[fieldidx] = fits.open(bright_inv_Mkk_fpath)['inv_Mkk_'+str(ifield)].data

					tot_inv_Mkk_fpath = mkk_base_path+'mkk_estimate_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+full_mask_tail+'.fits'
					inv_Mkks_tot[fieldidx] = fits.open(tot_inv_Mkk_fpath)['inv_Mkk_'+str(ifield)].data


				tot_masks[fieldidx] = joint_mask
				bright_masks[fieldidx] = bright_mask

				plot_map(tot_masks[fieldidx]*ciber_maps[fieldidx], title='tot mask x ciber maps')
				plot_map(bright_masks[fieldidx]*ciber_maps[fieldidx], title='bright mask x ciber maps')

		if iterate_grad_ff:
			print('Computing gradients/flat fields using bright masks')

			bright_images, ff_estimates_bright, final_planes,\
					stack_masks_bright, all_coeffs = iterative_gradient_ff_solve(ciber_maps, \
																		  niter=niter, masks=bright_masks, \
																		 ff_stack_min=ff_stack_min)

			print('computing for total mask')
			tot_images, ff_estimates_tot, final_planes,\
					stack_masks_tot, all_coeffs = iterative_gradient_ff_solve(ciber_maps, \
																		  niter=niter, masks=tot_masks, \
																		 ff_stack_min=ff_stack_min)
		else:
			bright_images = ciber_maps*bright_masks
			tot_images = ciber_maps*tot_masks



		print('now computing power spectra of maps..')
		for fieldidx, ifield in enumerate(ifield_list):

			print('computing point source dominated power spectrum..')
			# compute point source dominated power spectrum
			# if iterate_grad_ff:
			fullmaskbright = bright_masks[fieldidx].copy() 
			fullmasktot = tot_masks[fieldidx].copy()
			if iterate_grad_ff:
				fullmaskbright *= stack_masks_bright[fieldidx]
				fullmasktot *= stack_masks_tot[fieldidx]
			
			# fullmaskbright = stack_masks_bright[fieldidx]*bright_masks[fieldidx]
			masked_bright_meansub = ciber_maps[fieldidx]*fullmaskbright
			masked_bright_meansub[fullmaskbright != 0] -= np.mean(masked_bright_meansub[fullmaskbright != 0])
			lb, cl_masked_bright, clerr_masked = get_power_spec(masked_bright_meansub, lbinedges=self.Mkk_obj.binl, lbins=self.Mkk_obj.midbin_ell)
			
			if mkk_correct:
				cl_masked_bright = np.dot(inv_Mkks_bright[fieldidx].transpose(), cl_masked_bright)
			cls_masked_tot_bright[fieldidx] = cl_masked_bright
			
			print('computing masked noise power spectrum..')
			# compute masked noise power spectrum
			masked_tot_meansub = ciber_maps[fieldidx]*fullmasktot
			masked_tot_meansub[fullmasktot != 0] -= np.mean(masked_tot_meansub[fullmasktot != 0])
			lb, cl_masked_tot, clerr_masked = get_power_spec(masked_tot_meansub, lbinedges=self.Mkk_obj.binl, lbins=self.Mkk_obj.midbin_ell)
			if mkk_correct:
				cl_masked_tot = np.dot(inv_Mkks_tot[fieldidx].transpose(), cl_masked_tot)
			cls_masked_tot[fieldidx] = cl_masked_tot

			# compute power spectrum difference
			bright_faint_diff_cl = cl_masked_bright - cl_masked_tot

			if plot:
				plt.figure(figsize=(8,6))
				prefac = lb**2/(2*np.pi)
				plt.plot(lb, prefac*cl_masked_bright, label='J < 10 masked')
				plt.plot(lb, prefac*cl_masked_tot, label='masked J < 18')
				plt.plot(lb, prefac*bright_faint_diff_cl, label='difference')
				plt.xscale('log')
				plt.legend(fontsize=14)
				plt.yscale('log')
				plt.tick_params(labelsize=14)
				plt.show()

			print('Normalizing power spectra..')

			# normalize power spectrum by maximum value for ell > ell_norm
			diff_norm = bright_faint_diff_cl / np.max(bright_faint_diff_cl[lb > ell_norm])
			# anything for ell < ell_norm that is greater than one set to one
			diff_norm[diff_norm > 1] = 1. 
			diff_norm[lb < ell_norm] = 1.

			diff_norm_array[fieldidx, :] = diff_norm 

			if plot:
				plt.figure()
				plt.plot(lb, diff_norm, label='flight map differenced')
				plt.ylabel('$B_{\\ell}^2$', fontsize=18)
				plt.xlabel('Multipole $\\ell$', fontsize=18)
				plt.yscale('log')
				plt.xscale('log')
				plt.legend()
				plt.show()

		return lb, diff_norm_array, cls_masked_tot, cls_masked_tot_bright

	

	def generate_synthetic_mock_test_set(self, inst, ifield_list, test_set_fpath=None, mock_trilegal_path=None, cmock=None,\
									 zl_levels=None, noise_models=None, ciberdir='/Users/luminatech/Documents/ciber2/ciber/', \
									 ff_truth=None, ff_fpath=None, diffuse_realizations=None, zl_realizations=None, **kwargs):

		''' 
		Wrapper function for generating synthetic mock test set. 
		This assumes some level of precomputed products, in particular:
			- CIB and TRILEGAL source maps
			- masks 
			- smooth flat field (if available)
		'''
		
		gsmt_dict = dict({'same_zl_levels':False, 'apply_zl_gradient':True,\
						  'apply_smooth_FF':False, 'with_inst_noise':True,\
						  'with_photon_noise':True, 'load_ptsrc_cib':True, 'load_trilegal':True, \
						 'load_noise_model':True, 'show_plots':False, 'verbose':False, 'generate_diffuse_realization':True})
		
		float_param_dict = dict({'ff_min':0.5, 'ff_max':1.5, 'clip_sigma':5, 'ff_stack_min':2, 'nmc_ff':10, \
					  'theta_mag':0.01, 'niter':3, 'dgl_scale_fac':5, 'smooth_sigma':5, 'indiv_ifield':6, 'nfr_same':25})

		gsmt_dict, float_param_dict = update_dicts([gsmt_dict, float_param_dict], kwargs)
		
		nfields = len(ifield_list)

		if gsmt_dict['apply_smooth_FF'] and ff_truth is None:
			if ff_fpath is None:
				print('need to provide file path to flat field if not providing ff_truth')
				print('setting ff_truth to all ones..')
				# ff_truth = np.ones_like(total_signal)
				# field_set_shape = (self.dimx, self.dimy)
				ff_truth = np.ones((self.dimx, self.dimy))
			else:
				print('loading smooth FF from ', ff_fpath)
				ff_truth = gaussian_filter(fits.open(ff_fpath)[0].data, sigma=5)
			
		else:
			ff_truth = np.ones((self.dimx, self.dimy))

		if noise_models is not None:
			print('already provided noise models to generate_synthetic_mock_test_set(), setting gsmt_dict[load_noise_model] = False')
			gsmt_dict['load_noise_model'] = False

		if cmock is None:
			cmock = ciber_mock(ciberdir=ciberdir)
			

		if gsmt_dict['load_ptsrc_cib']:
			print('Loading point source CIB component..')
			if 'fits' in test_set_fpath:
				mock_cib_file = fits.open(test_set_fpath)
				# mock_cib_ims = np.array([mock_cib_file['map_'+str(ifield)].data.transpose() for ifield in ifield_list])
				# mock_cib_ims = np.array([mock_cib_file['cib_'+str(self.inst_to_band[inst])+'_'+str(ifield)].data.transpose() for ifield in ifield_list])
				mock_cib_ims = np.array([mock_cib_file['cib_'+str(self.inst_to_band[inst])+'_'+str(ifield)].data for ifield in ifield_list])

			else:
				print('unrecognized file type for '+str(test_set_fpath))
				return None
		elif not gsmt_dict['load_trilegal']:
			print('gsmt_dict[load_trilegal] is False, so make input sky signal from diffuse realization with shot noise..')
			# print('Generating from diffuse realization')
			# mock_cib_ims = np.zeros(imarray_shape)
			mock_cib_ims = []
			for fieldidx, ifield in enumerate(ifield_list):
				_, ps, mock_cib_im = generate_diffuse_realization(self.dimx, self.dimy, power_law_idx=0.0, scale_fac=3e8)
				lb, cl_mock, clerr_mock = get_power_spec(mock_cib_im - np.mean(mock_cib_im), lbinedges=self.Mkk_obj.binl, lbins=self.Mkk_obj.midbin_ell)
				print('cl mock is ', cl_mock)
				mock_cib_ims.append(mock_cib_im)


			mock_cib_ims = np.array(mock_cib_ims)
			# mock_cib_ims = np.array([generate_diffuse_realization(self.dimx, self.dimy, power_law_idx=0.0, scale_fac=3e8)[2] for ifield in ifield_list])
			# plot_map(mock_cib_ims[0], title='shot noise realization')
		else:
			mock_cib_ims = np.zeros((len(ifield_list), self.dimx, self.dimy))

		# if no mean levels provided, use ZL mean levels from file in kelsall folder
		if zl_levels is None:
			if gsmt_dict['same_zl_levels']:
				zl_levels = [self.zl_levels_ciber_fields[inst][self.ciber_field_dict[float_param_dict['indiv_ifield']]] for ifield in ifield_list]
			else:
				zl_levels = [self.zl_levels_ciber_fields[inst][self.ciber_field_dict[ifield]] for ifield in ifield_list]
		
		observed_ims, total_signals, zl_perfield, \
			shot_sigma_sb_maps, rnmaps, joint_masks, diff_realizations = instantiate_dat_arrays(self.dimx, self.dimy, nfields, 7)

		if gsmt_dict['with_inst_noise'] and gsmt_dict['load_noise_model']:
			noise_models = np.zeros_like(observed_ims)
		
		for fieldidx, ifield in enumerate(ifield_list):

			if self.field_nfrs is not None:
				field_nfr = self.field_nfrs[ifield]
			field_name_trilegal = cmock.ciber_field_dict[ifield]
		
			print('image ', fieldidx+1, ' of '+str(nfields))
			
			# self.load_data_products(ifield, inst, verbose=gsmt_dict['verbose'])
			# cmock.get_psf(ifield=ifield)
			
			if gsmt_dict['with_photon_noise'] or gsmt_dict['with_inst_noise']:
				
				if gsmt_dict['with_inst_noise'] and gsmt_dict['load_noise_model']:
					noise_fpath=noise_model_base_path+'/noise_model_TM'+str(inst)+'_ifield'+str(ifield)+'_full.fits'
					# noise_model = self.load_noise_Cl2D(ifield, inst, noise_fpath='data/fluctuation_data/TM'+str(inst)+'/noiseCl2D/field'+str(ifield)+'_noiseCl2D_110421.fits', inplace=False)
					noise_model = self.load_noise_Cl2D(ifield, inst, noise_fpath=noise_fpath, inplace=False)
					noise_models[fieldidx] = noise_model
			
			if mock_trilegal_path is not None:
				if '.npz' in mock_trilegal_path:
					mock_trilegal = np.load(mock_trilegal_path)
					mock_trilegal_im = mock_trilegal['srcmaps'][inst-1,:,:]
					mock_trilegal_cat = mock_trilegal['cat']
				elif '.fits' in mock_trilegal_path:
					print('loading trilegal fits file..')
					mock_trilegal = fits.open(mock_trilegal_path)
					# mock_trilegal_im = mock_trilegal['trilegal_'+str(self.inst_to_band[inst])+'_'+str(ifield)].data.transpose()
					mock_trilegal_im = mock_trilegal['trilegal_'+str(self.inst_to_band[inst])+'_'+str(ifield)].data

				else:
					print('unrecognized file type for '+str(test_set_fpath))
					return None

				if gsmt_dict['show_plots'] and fieldidx==0:
					plot_map(mock_trilegal_im, title='TRILEGAL, ifield'+str(ifield))
				
			# ------------- diffuse dgl realization -----------------------
			diff_realization = None 
			if gsmt_dict['generate_diffuse_realization']:
				diff_realization = self.generate_custom_sky_clustering(inst, dgl_scale_fac=float_param_dict['dgl_scale_fac'], ifield=ifield, gen_ifield=float_param_dict['indiv_ifield'])
				diff_realizations[fieldidx] = diff_realization

				if gsmt_dict['show_plots'] and fieldidx==0:
					plot_map(diff_realization, title='diff_realization, ifield'+str(ifield))

				# if not gsmt_dict['load_ptsrc_cib']:
				# 	diff_realizations[fieldidx] += mock_cib_ims
				
			# ------------- combine individual components into total_signal --------------------

			total_signal = mock_cib_ims[fieldidx].copy()

			if mock_trilegal_path is not None:
				print('adding mock trilegal image..')
				total_signal += mock_trilegal_im

			if diff_realization is not None:
				print('adding mock diffuse realization..')
				total_signal += diff_realization
				
			zl_perfield[fieldidx] = generate_zl_realization(zl_levels[fieldidx], gsmt_dict['apply_zl_gradient'], theta_mag=float_param_dict['theta_mag'], dimx=self.dimx, dimy=self.dimy)
			if gsmt_dict['show_plots'] and fieldidx==0:
				plot_map(zl_perfield[fieldidx], title='zl + gradient')
				
			print('adding ZL realization..')
			total_signal += zl_perfield[fieldidx] 
			
			#  ----------------- generate read and shot noise realizations -------------------
			
			shot_sigma_sb = self.compute_shot_sigma_map(inst, image=total_signal, nfr=field_nfr)
			if gsmt_dict['with_photon_noise']:
				shot_sigma_sb_maps[fieldidx] = shot_sigma_sb
				
			if gsmt_dict['with_photon_noise'] or gsmt_dict['with_inst_noise']:
				rnmap, snmap = self.noise_model_realization(inst, self.map_shape, noise_models[fieldidx], read_noise=gsmt_dict['with_inst_noise'], shot_sigma_sb=shot_sigma_sb, image=total_signal)
				if gsmt_dict['show_plots'] and fieldidx==0:
					plot_map(noise_models[fieldidx], title='noise model')
					plot_map(rnmap, title='rnmap, ifield'+str(ifield))
					plot_map(snmap, title='snmap, ifield'+str(ifield))

			# ------------------- add noise to signal and multiply sky signal by the flat field to get "observed" images -----------------------
			
			sum_mock_noise = total_signal.copy()

			if gsmt_dict['apply_smooth_FF']:
				print('Applying smooth FF..')
				sum_mock_noise *= ff_truth
			# sum_mock_noise = ff_truth*total_signal

			if gsmt_dict['with_photon_noise']:
				if gsmt_dict['apply_smooth_FF']:
					snmap *= ff_truth 
				if gsmt_dict['verbose']:
					print('Adding photon noise..')
				sum_mock_noise += snmap
					

			if gsmt_dict['with_inst_noise']:
				if gsmt_dict['verbose']:
					print('Adding read noise..')
				sum_mock_noise += rnmap

			total_signals[fieldidx] = total_signal
			
			if gsmt_dict['with_inst_noise']:
				rnmaps[fieldidx] = rnmap

			observed_ims[fieldidx] = sum_mock_noise
			
			if gsmt_dict['show_plots'] and fieldidx==0:
				
				# for plotting image limits
				x0, x1 = 0, self.dimx
				y0, y1 = 0, self.dimy
				
				fm = plot_map(gaussian_filter(mock_cib_ims[fieldidx], sigma=20), title='CIB (smoothed)', x0=x0, x1=x1, y0=y0, y1=y1)
				fm = plot_map(mock_cib_ims[fieldidx], title='CIB', x0=x0, x1=x1, y0=y0, y1=y1)

				if mock_trilegal_path is not None:
					f = plot_map(mock_trilegal_im, title='trilegal', x0=x0, x1=x1, y0=y0, y1=y1)
				if diff_realization is not None:
					f = plot_map(diff_realization, title='diff realization', x0=x0, x1=x1, y0=y0, y1=y1)
				if gsmt_dict['with_photon_noise']:
					f = plot_map(snmap, title='shot noise', x0=x0, x1=x1, y0=y0, y1=y1)
				if gsmt_dict['with_inst_noise']:
					f = plot_map(rnmap, title='read noise', x0=x0, x1=x1, y0=y0, y1=y1)
				f = plot_map(sum_mock_noise, title='Sum mock', x0=x0, x1=x1, y0=y0, y1=y1)
				f = plot_map(ff_truth, title='flat field', x0=x0, x1=x1, y0=y0, y1=y1)
				f = plot_map(observed_ims[fieldidx], title='post ff image', x0=x0, x1=x1, y0=y0, y1=y1)


		return joint_masks, observed_ims, total_signals, rnmaps, shot_sigma_sb_maps, noise_models, ff_truth, diff_realizations, zl_perfield, mock_cib_ims
	   




def small_Nl2D_from_larger(dimx_small, dimy_small, n_ps_bin,ifield, noise_model=None, \
						   inst=1, dimx_large=1024, dimy_large=1024, nsims=100, \
						  div_fac = 2.83):
	
	cbps_large = CIBER_PS_pipeline(n_ps_bin=n_ps_bin, dimx=dimx_large, dimy=dimy_large)
	cbps_small = CIBER_PS_pipeline(n_ps_bin=n_ps_bin, dimx=dimx_small, dimy=dimy_small)
	
	cl2d_all_small = np.zeros((nsims, dimx_small, dimy_small))
	
	if noise_model is None:
		noise_model = cbps_large.load_noise_Cl2D(ifield, inst, inplace=False, transpose=False)
		
	
	clprocs = []
	clprocs_large_chi2 = []
	
	for i in range(nsims):
		
		if i%100==0:
			print('i = '+str(i))
		noise = np.random.normal(0, 1, (dimx_large, dimy_large)) + 1j*np.random.normal(0, 1, (dimx_large, dimy_large))
		
		chi2_realiz = (np.random.chisquare(2., size=(dimx_large, dimy_large))/div_fac)**2
		rnmap_large_chi2 = np.sqrt(dimx_large*dimy_large)*ifft2(noise*ifftshift(np.sqrt(chi2_realiz*noise_model))).real
		rnmap_large_chi2 *= cbps_large.cal_facs[inst]/cbps_large.arcsec_pp_to_radian

		lb, cl_proc_large_chi2, _ = get_power_spec(rnmap_large_chi2-np.mean(rnmap_large_chi2),\
												  mask=None, weights=None, lbinedges=cbps_large.Mkk_obj.binl,\
												  lbins=cbps_large.Mkk_obj.midbin_ell)

		clprocs_large_chi2.append(cl_proc_large_chi2)

		rnmap_small = rnmap_large_chi2[:cbps_small.dimx, :cbps_small.dimy]

		lbins, cl_proc, cl_proc_err = get_power_spec(rnmap_small-np.mean(rnmap_small), mask=None, weights=None,\
													 lbinedges=cbps_small.Mkk_obj.binl, lbins=cbps_small.Mkk_obj.midbin_ell)

		clprocs.append(cl_proc)
		l2d, cl2d = get_power_spectrum_2d(rnmap_small)
		cl2d_all_small[i,:,:] = cl2d/cbps_small.cal_facs[inst]**2

	av_cl2d = np.mean(cl2d_all_small, axis=0)

	return av_cl2d, clprocs, clprocs_large_chi2, cbps_small


def iter_sigma_clip_mask(image, sig=5, nitermax=10, initial_mask=None):
	# image assumed to be 2d
	iteridx = 0
	
	summask = image.shape[0]*image.shape[1]
	running_mask = (image != 0).astype(np.int)
	if initial_mask is not None:
		running_mask *= initial_mask
		
	while iteridx < nitermax:
		
		new_mask = sigma_clip_maskonly(image, previous_mask=running_mask, sig=sig)
		
		if np.sum(running_mask*new_mask) < summask:
			running_mask *= new_mask
			summask = np.sum(running_mask)
		else:
			return running_mask

		iteridx += 1
		
	return running_mask

def sigma_clip_maskonly(vals, previous_mask=None, sig=5):
	
	valcopy = vals.copy()
	if previous_mask is not None:
		valcopy[previous_mask==0] = np.nan
		sigma_val = np.nanstd(valcopy)
	else:
		sigma_val = np.nanstd(valcopy)
	
	abs_dev = np.abs(vals-np.nanmedian(valcopy))
	mask = (abs_dev < sig*sigma_val).astype(np.int)

	return mask



