import numpy as np
from astropy.io import fits
import glob
import os
import config
import astropy
from numpy.fft import fftshift as fftshift

import astropy.wcs as wcs
from astropy import units as u

from ciber_powerspec_pipeline import *
from ciber_mocks import *
from ciber_noise_data_utils import *
from mock_galaxy_catalogs import *
from lognormal_counts import *
# from ciber_data_helpers import *
from helgason import *
from ps_pipeline_go import *
# from cross_spectrum_analysis import *
from mkk_parallel import *
from mkk_diagnostics import *
from flat_field_est import *
from plotting_fns import *
from powerspec_utils import *
from numerical_routines import *
from ciber_data_file_utils import *


# ciber_file_data_utils.py
# def load_read_noise_modl_filedat(fw_basepath, inst, ifield, tailstr=None, apply_FW=False, load_cl1d=False, \
# 								mean_nl2d_key='mean_nl2d', fw_key='fourier_weights'):


class CIBER_NoiseModel():
	
	
	
	def __init__(self, n_ps_bin=25, ifield_list=[4, 5, 6, 7, 8], cbps=None, save_fpath=None, base_path=None):
		
		self.ifield_list = ifield_list
		if cbps is None:
			cbps = CIBER_PS_pipeline(n_ps_bin=n_ps_bin)
		self.cbps = cbps
		
		if save_fpath is None:
			save_fpath = config.ciber_basepath+'data/fluctuation_data/'
			# save_fpath = config.exthdpath +'ciber_fluctuation_data/'
			print('save directory is ', save_fpath)

		if base_path is None:
			base_path = config.ciber_basepath+'data/noise_model_validation_data/'
			# base_path = config.exthdpath+'noise_model_validation_data/'
			print('base path is ', base_path)
		
		self.save_fpath = save_fpath
		self.base_path = base_path
		
		
	def grab_all_exposures(self, inst, ifield, labexp_dir=None, key='full'):
		
		if labexp_dir is None:
			labexp_dir = self.base_path+'/TM'+str(inst)+'/validationHalfExp/'
			if inst==2:
				labexp_dir += 'alldarkEXP_20211001/'
				
		labexp_fpaths = glob.glob(labexp_dir+'field'+str(ifield)+'/'+key+'/*.FITS')

		return labexp_fpaths
		
	def estimate_read_noise_modl(self, inst, per_quadrant=False, ifield_plot=[4], save=False, tailstr='quadsub_021323', compute_ps2d_per_quadrant=False, \
									halves=False):
		
		# noise_model_dir = self.save_fpath+'TM'+str(inst)+'/noise_model/'
		noise_model_dir = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/noise_model/'

		print('Estimating read noise model for exposures corresponding to ifield_list=', self.ifield_list)

		noise_models, dark_exp_diff_fpaths, noise_model_fpaths = [[] for x in range(3)]
		labexp_fpaths1, labexp_fpaths2, labexp_fpaths = [None for x in range(3)]

		for fieldidx, ifield in enumerate(self.ifield_list):
			if halves:
				labexp_fpaths1 = self.grab_all_exposures(inst, ifield, key='first')
				labexp_fpaths2 = self.grab_all_exposures(inst, ifield, key='second')
				nexp = len(labexp_fpaths1)
			else:
				labexp_fpaths = self.grab_all_exposures(inst, ifield, key='full')
				nexp = len(labexp_fpaths)
			
			maskInst_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/masks/maskInst_102422/field'+str(ifield)+'_TM'+str(inst)+'_maskInst_102422.fits'
			mask_inst = fits.open(maskInst_fpath)['maskInst'].data # 10/24/22

			plot_map(mask_inst, title='mask inst')

			if ifield in ifield_plot:
				plot=True 
			else:
				plot=False 

			if compute_ps2d_per_quadrant:
				print('were in compute_ps2d_per_quadrant')
				av_cl2d_per_quadrant, av_means, expdiff, cls_2d_noise_per_quadrant = labexp_to_noisemodl_quad(self.cbps, inst, ifield, labexp_fpaths=labexp_fpaths, clip_sigma=3, \
												 plot=plot, cal_facs=None, maskInst_fpath=maskInst_fpath, \
													 per_quadrant=per_quadrant, compute_ps2d_per_quadrant=compute_ps2d_per_quadrant, \
													 labexp_fpaths1=labexp_fpaths1, labexp_fpaths2=labexp_fpaths2, halves=halves)

				noise_models.append(av_cl2d_per_quadrant)

			else:

			

				av_cl2d, av_means, expdiff, cls_2d_noise_full = labexp_to_noisemodl_quad(self.cbps, inst, ifield, labexp_fpaths=labexp_fpaths, clip_sigma=3, \
																 plot=plot, cal_facs=None, mask_inst=mask_inst, maskInst_fpath=maskInst_fpath, \
																	 per_quadrant=per_quadrant, compute_ps2d_per_quadrant=compute_ps2d_per_quadrant, \
																	 labexp_fpaths1=labexp_fpaths1, labexp_fpaths2=labexp_fpaths2, halves=halves)

				
				noise_models.append(av_cl2d)

			if save:
				if compute_ps2d_per_quadrant:
					for q in range(4):

						dark_exp_diff_fpath_indiv = noise_model_dir+'dark_exp_diffs/dark_exp_diff_TM'+str(inst)+'_ifield'+str(ifield)+'_'+tailstr+'_quad'+str(q)+'.npz'
						dark_exp_diff_fpaths.append(dark_exp_diff_fpath_indiv)
						np.savez(dark_exp_diff_fpath_indiv, ifield=ifield, inst=inst, cls_2d_noise_full=cls_2d_noise_per_quadrant)

						hdul = write_noise_model_fits(av_cl2d_per_quadrant[q], ifield, inst)
						noise_model_fpath_indiv = noise_model_dir+'noise_model_TM'+str(inst)+'_ifield'+str(ifield)+'_'+tailstr+'_quad'+str(q)+'.fits'
						print('saving to ', noise_model_fpath_indiv)
						noise_model_fpaths.append(noise_model_fpath_indiv)
						hdul.writeto(noise_model_fpath_indiv, overwrite=True)

				else:
					dark_exp_diff_fpath = noise_model_dir+'dark_exp_diffs/dark_exp_diff_TM'+str(inst)+'_ifield'+str(ifield)+'_'+tailstr+'.npz'
					dark_exp_diff_fpaths.append(dark_exp_diff_fpath)
					np.savez(dark_exp_diff_fpath, ifield=ifield, inst=inst, expdiff=expdiff, cls_2d_noise_full=cls_2d_noise_full)

					hdul = write_noise_model_fits(av_cl2d, ifield, inst)

					if halves:
						noise_model_fpath = noise_model_dir+'halfexp/noise_model_TM'+str(inst)+'_ifield'+str(ifield)+'_'+tailstr+'.fits'
					else:
						noise_model_fpath = noise_model_dir+'noise_model_TM'+str(inst)+'_ifield'+str(ifield)+'_'+tailstr+'.fits'


					print('Saving noise model to ', noise_model_fpath)
					noise_model_fpaths.append(noise_model_fpath)
					hdul.writeto(noise_model_fpath, overwrite=True)

		return noise_models, dark_exp_diff_fpaths, noise_model_fpaths
	
	def compare_read_noise_modl_dark_dat(self, inst, ifield, darkdiff_cl2ds, inv_Mkk=None, av_mask_frac=None,\
		 mean_nl2d_key='mean_nl2d', fw_key='fourier_weights', fw_basepath='/Users/richardfeder/Downloads',\
		 tailstr=None, apply_FW=False, load_cl1d=False, clip_sigma=5):



		mean_nl2d, fourier_weights, cl1d_unweighted  = load_read_noise_modl_filedat(fw_basepath, inst, ifield, tailstr=tailstr, apply_FW=apply_FW, load_cl1d=load_cl1d, \
								mean_nl2d_key='mean_nl2d', fw_key='fourier_weights')

				
		l2d = get_l2d(self.cbps.dimx, self.cbps.dimy, self.cbps.pixsize)

		cls_dd = np.zeros((len(darkdiff_cl2ds), self.cbps.n_ps_bin))
		clerrs_dd = np.zeros((len(darkdiff_cl2ds), self.cbps.n_ps_bin))

		for i, dd_cl2d in enumerate(darkdiff_cl2ds):
			
			if clip_sigma is not None:
				fourier_mode_mask = iter_sigma_clip_mask(dd_cl2d, sig=clip_sigma, nitermax=2)
				dd_cl2d *= fourier_mode_mask
				if fourier_weights is not None:
					fourier_weights *= fourier_mode_mask

			lb, cl_dd, clerr_dd = azim_average_cl2d(dd_cl2d, l2d, weights=fourier_weights, lbinedges=self.cbps.Mkk_obj.binl, lbins=self.cbps.Mkk_obj.midbin_ell)

			if inv_Mkk is not None:
				cls_dd[i] = np.dot(inv_Mkk.transpose(), cl_dd)
			elif av_mask_frac is not None:
				cls_dd[i] = cl_dd/av_mask_frac
			else:
				cls_dd[i] = cl_dd
			
			clerrs_dd[i] = clerr_dd
				
		lb, cl_modl, clerr_modl = azim_average_cl2d(mean_nl2d, l2d, weights=fourier_weights, lbinedges=self.cbps.Mkk_obj.binl, lbins=self.cbps.Mkk_obj.midbin_ell)

		if inv_Mkk is not None:
			cl_modl = np.dot(inv_Mkk.transpose(), cl_modl)
		elif av_mask_frac is not None:
			cl_modl /= av_mask_frac
			
		if load_cl1d:
			return lb, cl_modl, clerr_modl, cls_dd, clerrs_dd, mean_nl2d, cl1d_unweighted

		return lb, cl_modl, clerr_modl, cls_dd, clerrs_dd, mean_nl2d

	
	# def compute_flight_exp_differences(self, ifield, inst, mask, base_nv_path=None):
	# 	if base_nv_path is None:
	# 		base_nv_path = config.ciber_basepath+'data/noise_model_validation_data/'
	
	def compute_lab_flight_exp_differences(self, ifield, inst, mask, base_nv_path=None, mask_fpath=None, \
										  mask_inst=None, stdpower=2, flight_exp_diff=None, verbose=False, plot=False, clip_sigma = 5, nitermax=10, \
										   compute_noise_cov=False, return_read_nl2d=False, \
										   nsims=500, n_split=10, chisq=False, per_quadrant=False, gradient_filter=False, mean_nl2d_diff=None, \
										   compute_weighted_1d_nls=False, fourier_weights=None, load_fw_mean_nl2d=False):

		''' 
		mask_inst used if one needs a 2D power spectrum for the read noise alone, as is used in compute_noise_covariance_matrix()
		'''

		if base_nv_path is None:
			base_nv_path = config.ciber_basepath+'data/noise_model_validation_data/'

		print('base nv path:', base_nv_path)

		if mask_fpath is None and mask is None:
			mask_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/masks/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'
			print('both mask_fpath and mask are None, loading mask from default ', mask_fpath)        
			mask = fits.open(mask_fpath)['joint_mask_'+str(ifield)].data
		if plot:
			plot_map(mask, title='mask')

		if mask_inst is not None:
			print('we have instrument mask')
			plot_map(mask_inst, title='mask inst')

		data_path = base_nv_path+'TM'+str(inst)+'/validationHalfExp/field'+str(ifield)+'/'

		if inst==2:
			lab_path = base_nv_path+'TM'+str(inst)+'/validationHalfExp/alldarkEXP_20211001/field'+str(ifield)
		else:
			lab_path = base_nv_path+'TM'+str(inst)+'/validationHalfExp/field'+str(ifield)

		labexp_fpaths1 = self.grab_all_exposures(inst, ifield, key='first')
		labexp_fpaths2 = self.grab_all_exposures(inst, ifield, key='second')

		if verbose:
			print('all labexp fpaths 1:', labexp_fpaths1)
			print('all labexp fpaths 2:', labexp_fpaths2)

		n_image = len(labexp_fpaths1)
		imarray_shape = (n_image, self.cbps.dimx, self.cbps.dimy)
		psarray_shape = (n_image, self.cbps.n_ps_bin)
		dark_diffs, cl2ds_dark = [np.zeros(imarray_shape) for x in range(2)]

		cl2ds_read, mean_nl2d_read = None, None
		if return_read_nl2d:
			cl2ds_read = np.zeros(imarray_shape)
		cl_diffs_darkexp, cl_diffs_phot = [np.zeros(psarray_shape) for x in range(2)]

		if verbose:
			print('base_nv_path is ', base_nv_path)
			print('mask_fpath is ', mask_fpath)
			print('data_path is ', data_path)
			print('cal fac is :', self.cbps.cal_facs[inst])
			print('nimage = ', n_image)

		if flight_exp_diff is None:
			pathA = data_path+'/flightMap_2.FITS'
			pathB = data_path+'/flightMap_1.FITS'
			flight_exp_diff, meanA, meanB, mean_flight_test, expA, expB = compute_exp_difference(inst, cal_facs=self.cbps.cal_facs, pathA=pathA, pathB=pathB, mask=mask, mode='flight', return_maps=True)

			print('meanA = ', meanA, 'mean B = ', meanB)

		flight_sc_mask = iter_sigma_clip_mask(flight_exp_diff, sig=clip_sigma, nitermax=nitermax, mask=mask.astype(int))

		if plot:
			plot_map(expA-expB, title='expA - expB')
			plot_map(flight_sc_mask, title='new mask')
			plot_map(flight_sc_mask-mask, title='new mask - mask')        
			plot_map(flight_exp_diff*flight_sc_mask, title='flight exp diff sigma clipped')
			plot_map(gaussian_filter(flight_exp_diff*flight_sc_mask, sigma=20)*flight_sc_mask, title='smoothed flight exp diff sigma clipped')

		if verbose:
			print('meanA, meanB ', meanA, meanB)
			print(self.cbps.field_nfrs[ifield], self.cbps.field_nfrs[ifield]//2)

		
		mean_image = 0.5*(meanA+meanB)*np.ones_like(flight_exp_diff)
		
		shot_sigma_sb_map = self.cbps.compute_shot_sigma_map(inst, mean_image, nfr=self.cbps.field_nfrs[ifield]//2)

		dsnmaps = np.sqrt(2)*shot_sigma_sb_map*np.random.normal(0, 1, size=imarray_shape)

		for i in range(n_image):

			pathA, pathB = labexp_fpaths1[i], labexp_fpaths2[i]
			if verbose:
				print('i = ', i, ' pathA:', pathA)
				print('i = ', i, ' pathB:', pathB)

			dark_exp_diff, darkmeanA, darkmeanB, mean_dark = compute_exp_difference(inst, mask=mask, cal_facs=self.cbps.cal_facs, pathA=pathA, pathB=pathB)
			dark_sc_mask = iter_sigma_clip_mask(dark_exp_diff, sig=clip_sigma, nitermax=nitermax, mask=mask.astype(int))
			dark_exp_diff += dsnmaps[i]

			if per_quadrant:
				for q in range(4):
					mquad_sc = dark_sc_mask[self.cbps.x0s[q]:self.cbps.x1s[q],self.cbps.y0s[q]:self.cbps.y1s[q]]

					dark_exp_diff[self.cbps.x0s[q]:self.cbps.x1s[q],self.cbps.y0s[q]:self.cbps.y1s[q]][mquad_sc==1] -= np.mean(dark_exp_diff[self.cbps.x0s[q]:self.cbps.x1s[q],self.cbps.y0s[q]:self.cbps.y1s[q]][mquad_sc==1])
			else:
				dark_exp_diff[dark_sc_mask==1] -= np.mean(dark_exp_diff[dark_sc_mask==1])

			dark_diffs[i] = dark_exp_diff*dark_sc_mask
			# dark_diffs[i][dark_sc_mask==1] -= np.mean(dark_diffs[i][dark_sc_mask==1])
			l2d, cl2d_dark = get_power_spectrum_2d(dark_diffs[i])
			cl2ds_dark[i] = cl2d_dark

			if plot:
				plot_map(dark_sc_mask, title='dark sc mask')
				plot_map(dark_exp_diff, title='dark_exp_diff i = '+str(i))
				plot_map(dark_diffs[i], title='dark diffs i = '+str(i))

			if return_read_nl2d:
				rn_exp_diff, darkmeanA, darkmeanB, mean_dark = compute_exp_difference(inst, mask=mask_inst, cal_facs=self.cbps.cal_facs, pathA=pathA, pathB=pathB)

				if i==0:
					plot_map(rn_exp_diff, title='rn diff')
				rn_sc_mask = iter_sigma_clip_mask(rn_exp_diff, sig=clip_sigma, nitermax=nitermax, mask=mask_inst.astype(int))

				rn_exp_diff *= rn_sc_mask

				rn_exp_diff[rn_sc_mask==1] -= np.mean(rn_exp_diff[rn_sc_mask==1])
				l2d, cl2d_read = get_power_spectrum_2d(rn_exp_diff)
				cl2ds_read[i] = cl2d_read

		# compute mean 2D power spectrum and compute inverse variance weights
		mean_nl2d_dark = np.mean(cl2ds_dark, axis=0)

		if return_read_nl2d:
			mean_nl2d_read = np.mean(cl2ds_read, axis=0)/2.


		plot_map(shot_sigma_sb_map, title='shot_sigma_sb_map')

		if mean_nl2d_diff is None:
			mean_nl2d_diff = mean_nl2d_dark

		if not load_fw_mean_nl2d:
			print('Estimating noise power spectrum mean and Fourier weights..')
			fourier_weights, mean_nl2d_modl, nl1ds_diff = self.cbps.estimate_noise_power_spectrum(noise_model=mean_nl2d_diff, mask=mask, shot_sigma_sb=shot_sigma_sb_map, field_nfr=self.cbps.field_nfrs[ifield]//2,\
																				 nsims=nsims, n_split=n_split, inst=inst, ifield=ifield, photon_noise=True, read_noise=True, inplace=False, chisq=chisq, \
																				 difference=True, compute_1d_cl=True)
			fourier_weights /= np.nanmax(fourier_weights)
			plot_map(np.log10(fourier_weights), title='log10(w($\\ell_x, \\ell_y$))')


		nl1ds_diff_weighted, nl1ds_diff = None, None
		if compute_weighted_1d_nls and fourier_weights is not None:
			print('now computing weighted 1d nls..')
			_, mean_nl2d_modl, nl1ds_diff_weighted = self.cbps.estimate_noise_power_spectrum(noise_model=mean_nl2d_diff, mask=mask, shot_sigma_sb=shot_sigma_sb_map, field_nfr=self.cbps.field_nfrs[ifield]//2,\
																	 nsims=nsims, n_split=n_split, inst=inst, ifield=ifield, photon_noise=True, read_noise=True, inplace=False, chisq=chisq, \
																	 difference=True, compute_1d_cl=True, fw_diff=fourier_weights)
			nl1ds_diff = nl1ds_diff_weighted

		proc_nls, noise_cov = None, None
		if compute_noise_cov:

			if mean_nl2d_read is None:
				print('need the read noise 2d power spectra with instrument mask applied to get this right')
				return
			else:
				proc_nls, noise_cov = compute_noise_covariance_matrix(fourier_weights, mean_nl2d_read, mask=mask, shot_sigma_sb=shot_sigma_sb_map, \
														   field_nfr=self.cbps.field_nfrs[ifield]//2)


		# now that we have the fourier weights, let's compute the FW'd lab and noise model power spectra
		# lb, cl_diff_noisemodl, _ = azim_average_cl2d(mean_nl2d_modl, l2d, weights=fourier_weights, lbinedges=self.cbps.Mkk_obj.binl, lbins=self.cbps.Mkk_obj.midbin_ell)
		lb, cl_diff_noisemodl, clerr_diff_noisemodl = self.cbps.compute_noise_power_spectrum(inst, noise_Cl2D=mean_nl2d_modl, inplace=False, weights=fourier_weights, apply_FW=True)

		cl_diff_noisemodl /= 2.

		if nl1ds_diff is not None:
			nl1ds_diff = np.array(nl1ds_diff)/2.

			clerr_diff_noisemodl = 0.5*(np.percentile(np.array(nl1ds_diff), 84, axis=0)-np.percentile(np.array(nl1ds_diff), 16, axis=0))
		else:
			clerr_diff_noisemodl = None

		for i in range(n_image):

			lb, cl_diff_phot, _ = get_power_spec(mask*dsnmaps[i], weights=fourier_weights, lbinedges=self.cbps.Mkk_obj.binl, lbins=self.cbps.Mkk_obj.midbin_ell)
			
			lb, cl_diff_darkexp, _ = azim_average_cl2d(cl2ds_dark[i], l2d, weights=fourier_weights, lbinedges=self.cbps.Mkk_obj.binl, lbins=self.cbps.Mkk_obj.midbin_ell)

			cl_diffs_phot[i] = cl_diff_phot/2.
			cl_diffs_darkexp[i] = cl_diff_darkexp/2.

		if per_quadrant:
			print('subtracting flight data per quadrant')

			flight_masked_diff = (flight_exp_diff*flight_sc_mask).copy()
			
			for q in range(4):
				mquad = flight_sc_mask[self.cbps.x0s[q]:self.cbps.x1s[q], self.cbps.y0s[q]:self.cbps.y1s[q]]

				if gradient_filter:
					theta_quad, plane_quad = fit_gradient_to_map(flight_exp_diff[self.cbps.x0s[q]:self.cbps.x1s[q], self.cbps.y0s[q]:self.cbps.y1s[q]], mask=mquad)
					flight_exp_diff[self.cbps.x0s[q]:self.cbps.x1s[q], self.cbps.y0s[q]:self.cbps.y1s[q]] -= plane_quad

				flight_exp_diff[self.cbps.x0s[q]:self.cbps.x1s[q], self.cbps.y0s[q]:self.cbps.y1s[q]][mquad==1] -= np.mean(flight_masked_diff[self.cbps.x0s[q]:self.cbps.x1s[q], self.cbps.y0s[q]:self.cbps.y1s[q]][mquad==1])

			if plot:
				plot_map(flight_exp_diff*flight_sc_mask, title='gradient subtracted maps')
				plot_map(gaussian_filter(flight_exp_diff*flight_sc_mask, sigma=20)*flight_sc_mask, title='smoothed, gradient subtracted maps')
			
		else:
			flight_exp_diff[flight_sc_mask==1] -= np.mean(flight_exp_diff[flight_sc_mask==1])
		
		lb, cl_diff_flight, clerr_diff_flight = get_power_spec(flight_exp_diff*flight_sc_mask, weights=fourier_weights, lbinedges=self.cbps.Mkk_obj.binl, lbins=self.cbps.Mkk_obj.midbin_ell)

		cl_diff_flight /= 2.
		clerr_diff_flight /= 2.

		if verbose:
			print('mean cl_diff_lab : ', np.mean(np.array(cl_diffs_lab), axis=0))
			print('mean cl_diff_phot : ', np.mean(np.array(cl_diffs_phot), axis=0))
			print('cl_diff_flight : ', cl_diff_flight)

		return lb, cl_diff_flight, clerr_diff_flight, nl1ds_diff, cl_diffs_darkexp, cl_diff_noisemodl, clerr_diff_noisemodl, cl_diffs_phot,\
						expA, expB, fourier_weights, cl2ds_dark, mean_nl2d_read, proc_nls, noise_cov, shot_sigma_sb_map, nl1ds_diff_weighted


	def exposure_half_difference_test_validation(self, inst, ifield_list=[4,5,6,7,8], ifield_plot=[], save_nvdat=False, tailstr='112222_newmask', masking_maglim=None, \
												nsims=500, n_split=10, plot=False, per_quadrant=False, load_inst_mask=False, compute_ps2d_per_quadrant=False, mask_tail=None, \
												mask_base_path=None, clip_sigma=5., verbose=False, gradient_filter=True, noisemodl_tailstr=None, compute_weighted_1d_nls=True, \
												load_fw_mean_nl2d=False):

		exps_A, exps_B, all_fourier_weights = [np.zeros((len(ifield_list), self.cbps.dimx, self.cbps.dimy)) for x in range(3)]

		all_cl2ds_dark, fpaths_save = [], []

		if masking_maglim is None:
			if inst==1:
				masking_maglim = 17.5
			elif inst==2:
				masking_maglim = 17.0

		if mask_tail is None:
			if inst==1:
				mask_tail = 'maglim_J_Vega_'+str(masking_maglim)+'_111323_ukdebias'
			elif inst==2:
				mask_tail = 'maglim_H_Vega_'+str(masking_maglim)+'_111323_ukdebias'

		for fieldidx, ifield in enumerate(ifield_list):
			
			plot = False
			if ifield in ifield_plot:
				plot=True
				
			# load mask
			if mask_base_path is None:
				mask_base_path = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/masks/'
			mask_fpath = mask_base_path+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'
			full_mask = fits.open(mask_fpath)[1].data.astype(int)    

			if load_inst_mask:
				maskInst_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/masks/maskInst_102422/field'+str(ifield)+'_TM'+str(inst)+'_maskInst_102422.fits'
				mask_inst = fits.open(maskInst_fpath)['maskInst'].data # 10/24/22
				full_mask *= mask_inst.astype(int)

			mean_nl2d_diff=None
			if noisemodl_tailstr is not None:

				# noise_model_dir = config.exthdpath+'ciber_fluctuation_data/TM'+str(inst)+'/noise_model/'
				noise_model_dir = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/noise_model/'
				mean_nl2d_diff, _, _ = load_read_noise_modl_filedat(noise_model_dir, inst, ifield, tailstr=noisemodl_tailstr, load_cl1d=False, \
																						mean_nl2d_key='mean_nl2d', apply_FW=False)

		
				plot_map(full_mask, title='full mask')
				plot_map(mean_nl2d_diff, title='nl2d diff')

			if load_fw_mean_nl2d:
				tailstr='062423'
				print('Loading Fourier weights from noise model..')
				clfile = np.load(config.exthdpath+'noise_model_validation_data/TM'+str(inst)+'/validationHalfExp/field'+str(ifield)+'/powerspec/cls_expdiff_lab_flight_TM'+str(inst)+'_ifield'+str(ifield)+'_'+tailstr+'.npz')
				fourier_weights = clfile['fourier_weights']
			else:
				fourier_weights = None	

			lb, cl_diff_flight, clerr_diff_flight, nl1ds_diff, \
				cl_diffs_darkexp, cl_diff_noisemodl, clerr_diff_noisemodl, cl_diffs_phot,\
				expA, expB, fourier_weights, cl2ds_dark,\
				mean_nl2d_read, proc_nls, noise_cov,\
					shot_sigma_sb_map, nl1ds_diff_weighted = self.compute_lab_flight_exp_differences(ifield, inst, full_mask, base_nv_path=None, clip_sigma=clip_sigma, nitermax=10,\
																			   stdpower=2, compute_noise_cov=False, plot=plot,\
																			   mask_inst=mask_inst, nsims=nsims, n_split=n_split, \
																			   per_quadrant=per_quadrant, mask_fpath=mask_fpath, verbose=verbose, gradient_filter=gradient_filter, \
																			   mean_nl2d_diff=mean_nl2d_diff, compute_weighted_1d_nls=compute_weighted_1d_nls, load_fw_mean_nl2d=load_fw_mean_nl2d, \
																			   fourier_weights=fourier_weights)
			
			prefac = lb*(lb+1)/(2*np.pi)
			plt.figure()
			plt.errorbar(lb, prefac*cl_diff_flight, yerr=prefac*clerr_diff_flight, color='r', label='Flight difference')
			plt.errorbar(lb, prefac*cl_diff_noisemodl, yerr=prefac*clerr_diff_noisemodl, color='k', label='Noise model')
			if nl1ds_diff_weighted is not None:
				for nidx in range(len(nl1ds_diff_weighted)):
					plt.plot(lb, prefac*nl1ds_diff_weighted[nidx], color='grey', alpha=0.1)

			plt.xscale('log')
			plt.yscale('log')
			plt.legend()
			plt.xlabel('$\\ell$', fontsize=14)
			plt.ylabel('$D_{\\ell}$', fontsize=14)
			plt.tick_params(labelsize=12)
			plt.show()


			mean_cl2d = np.mean(cl2ds_dark, axis=0)

			plot_map(mean_cl2d, title='mean cl2d')

			if mean_nl2d_read is not None:
				plot_map(mean_nl2d_read, title='mean_nl2d read')
				plot_map(mean_nl2d_read/mean_cl2d, title='read/ total')

			exps_A[fieldidx] = expA
			exps_B[fieldidx] = expB
			all_fourier_weights[fieldidx] = fourier_weights
			all_cl2ds_dark.append(cl2ds_dark)

			# make sure 2d power spectrum is being passed through analysis correctly
			if save_nvdat:
				fpath_save = config.ciber_basepath+'data/noise_model_validation_data/TM'+str(inst)+'/validationHalfExp/field'+str(ifield)+'/powerspec/cls_expdiff_lab_flight_TM'+str(inst)+'_ifield'+str(ifield)+'_'+tailstr+'.npz'
				# fpath_save = config.exthdpath+'noise_model_validation_data/TM'+str(inst)+'/validationHalfExp/field'+str(ifield)+'/powerspec/cls_expdiff_lab_flight_TM'+str(inst)+'_ifield'+str(ifield)+'_'+tailstr+'.npz'
				print('save fpath is ', fpath_save)
				# np.savez(fpath_save, \
				# 		lb=lb, mean_cl2d=mean_cl2d, fourier_weights=fourier_weights, cl_diff_flight=cl_diff_flight, clerr_diff_flight=clerr_diff_flight, cl_diffs_lab=cl_diffs_noisemodl, cl_diffs_phot=cl_diffs_phot, \
				# 		mean_nl2d_read=mean_nl2d_read, shot_sigma_sb_map=shot_sigma_sb_map)
				np.savez(fpath_save, \
						lb=lb, mean_cl2d=mean_cl2d, nl1ds_diff=nl1ds_diff, fourier_weights=fourier_weights, cl_diff_flight=cl_diff_flight, clerr_diff_flight=clerr_diff_flight, \
						cl_diffs_darkexp=cl_diffs_darkexp, cl_diff_noisemodl=cl_diff_noisemodl, clerr_diff_noisemodl=clerr_diff_noisemodl, cl_diffs_phot=cl_diffs_phot, \
						mean_nl2d_read=mean_nl2d_read, shot_sigma_sb_map=shot_sigma_sb_map, nl1ds_diff_weighted=nl1ds_diff_weighted)


				fpaths_save.append(fpath_save)
				
		
		return exps_A, exps_B, all_fourier_weights


	def compute_noise_covariance_matrix(self, fourier_weights, mean_nl2d_read, inst, mask=None, shot_sigma_sb=None, \
									nsims=500, n_split=10, photon_noise=True, read_noise=True, pixsize=7., inv_Mkk=None):
	
		''' inst used for querying cal_facs '''
		dimx, dimy = mean_nl2d_read.shape[0], mean_nl2d_read.shape[1]
		n_per_split = nsims // n_split

		if shot_sigma_sb is not None:
			photon_noise=True
		else:
			photon_noise=False

		maplist_split_shape = (nsims//n_split, dimx, dimy)
		empty_aligned_objs, fft_objs = construct_pyfftw_objs(3, maplist_split_shape)
		sterad_per_pix = (pixsize/3600/180*np.pi)**2
		V = dimx*dimy*sterad_per_pix

		proc_nls = np.zeros((nsims, self.cbps.n_ps_bin))

		for i in range(n_split):
			simmaps = np.zeros(maplist_split_shape)

			print('Split '+str(i+1)+' of '+str(n_split)+'..')

			rnmaps, snmaps = self.cbps.noise_model_realization(inst, maplist_split_shape, mean_nl2d_read, fft_obj=fft_objs[0],\
												  read_noise=read_noise, photon_noise=False)

			if shot_sigma_sb is not None: # differences
				dsnmaps = shot_sigma_sb*np.random.normal(0, 1, size=maplist_split_shape)


			if i==0:
				if read_noise:
					plot_map(rnmaps[0], title='rn map')
				if photon_noise:
					plot_map(dsnmaps[0], title='sn map')

			if read_noise:
				simmaps += rnmaps
			if photon_noise:
				simmaps += dsnmaps

			if mask is not None:
				simmaps *= mask
				unmasked_means = [np.mean(simmap[mask==1]) for simmap in simmaps]
				simmaps -= np.array([mask*unmasked_mean for unmasked_mean in unmasked_means])
			else:
				simmaps -= np.array([np.mean(simmap) for simmap in simmaps])

			if i==0:
				plot_map(simmaps[0])

			fft_objs[1](simmaps*sterad_per_pix)

			nl2ds = np.array([fftshift(dentry*np.conj(dentry)).real for dentry in empty_aligned_objs[2]])

			for nidx, nl2d in enumerate(nl2ds):
				lb, nlweighted, Clerr = self.cbps.compute_noise_power_spectrum(inst, noise_Cl2D=nl2d, inplace=False,\
															   weights=fourier_weights, apply_FW=True)


				proc_nls[i*n_per_split + nidx, :] = nlweighted

				if i==0 and nidx==0:
					lb, nlunweighted, Clerr = self.cbps.compute_noise_power_spectrum(inst, noise_Cl2D=nl2d, inplace=False,\
													   weights=None, apply_FW=False)
					print('proc_nls[0]', proc_nls[0])
					print('nl unweighted:', nlunweighted)


		noise_cov = np.cov(proc_nls.transpose())

		print('noise cov has shape', noise_cov.shape)
		print('proc_nls:', proc_nls)

		return lb, proc_nls, noise_cov




def labexp_to_noisemodl_quad(cbps, inst, ifield, labexp_fpaths=None, \
						clip_sigma = 4, sigma_clip_cl2d=False, \
						cl2d_clip_sigma=4, cal_facs=None, nitermax=10,\
						plot=True, base_fluc_path=None, \
						mask_inst=None, maskInst_fpath=None, per_quadrant=False, compute_ps2d_per_quadrant=False, \
						halves=False, labexp_fpaths1=None, labexp_fpaths2=None):
	
	
	''' 
	Estimates a read noise model from a collection of dark exposure pair differences. 
	
	If per_quadrant set to True, half-exposure differences are mean subtracted by quadrant before 
	
	'''

	if base_fluc_path is None:
		base_fluc_path = config.exthdpath+'ciber_fluctuation_data/'

	if halves:
		nexp = len(labexp_fpaths1)
		assert nexp == len(labexp_fpaths2)
	else:
		nexp = len(labexp_fpaths)

	if halves:
		npair = nexp 
	else:
		npair = nexp // 2

	if compute_ps2d_per_quadrant:
		cbps_quad = CIBER_PS_pipeline(dimx=512, dimy=512)
		cls_2d_noise_per_quadrant = np.zeros((4, npair, cbps_quad.dimx, cbps_quad.dimy))


	print('number of pairs : ', npair)
	
	mameans_full = np.zeros((npair,))
	cls_2d_noise_full = np.zeros((npair, cbps.dimx, cbps.dimy))
	

	meanAs, meanBs, mask_fracs = [], [], []

	if mask_inst is None:
		if maskInst_fpath is None:
			mask_inst = cbps.load_mask(ifield=ifield, inst=inst, masktype='maskInst_clean', inplace=False)
		else:
			mask_inst = fits.open(maskInst_fpath)['maskInst'].data # 10/24/22
		
	for i in range(npair):

		if halves:
			lab_exp_diff, meanA, meanB, mean_exp = compute_exp_difference(inst, cal_facs=cal_facs, pathA=labexp_fpaths1[i], pathB=labexp_fpaths2[i], mask=mask_inst, plot_diff_hist=False)
		else:
			lab_exp_diff, meanA, meanB, mean_exp = compute_exp_difference(inst, cal_facs=cal_facs, pathA=labexp_fpaths[2*i], pathB=labexp_fpaths[2*i+1], mask=mask_inst, plot_diff_hist=False)
		
		if plot:
			plot_map(lab_exp_diff*mask_inst, title='lab exp diff * mask_inst')
		
		lab_exp_diff /= np.sqrt(2)
		new_mask = iter_sigma_clip_mask(lab_exp_diff, sig=clip_sigma, nitermax=nitermax, mask=mask_inst.astype(int))
		mask_inst *= new_mask
		
		if plot:
			plot_map(lab_exp_diff*mask_inst, title='lab exp diff sigma clipped')
		
		meanAs.append(meanA)
		meanBs.append(meanB)
		masked_full = lab_exp_diff*mask_inst
				
		fdet = float(np.sum(mask_inst))/float(cbps.dimx*cbps.dimy)
		mask_fracs.append(fdet)
		
		if per_quadrant:

			masked_meansub_diff = masked_full.copy()
			
			for q in range(4):
				mquad = mask_inst[cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]]
				masked_meansub_diff[cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]][mquad==1] -= np.mean(lab_exp_diff[cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]][mquad==1])
			
				if compute_ps2d_per_quadrant:
					l2d_indiv, cl2d_indiv = get_power_spectrum_2d(masked_meansub_diff[cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]])
					cls_2d_noise_per_quadrant[q,i] = cl2d_indiv

		else:
			mamean = np.mean(masked_full[masked_full!=0])
			mameans_full[i] = mamean
			# get 2d power spectrum
			masked_meansub_diff = mask_inst*(lab_exp_diff-mamean)
			
		if plot:
			plot_map(masked_meansub_diff, title='masked meansub diff')
			
		l2d, cl2d = get_power_spectrum_2d(masked_meansub_diff, pixsize=cbps.Mkk_obj.pixsize)
		
		cls_2d_noise_full[i] = cl2d
		
	print('cls_2d_noise_Full has shape', cls_2d_noise_full.shape)
	av_cl2d = np.mean(cls_2d_noise_full, axis=0)

	if compute_ps2d_per_quadrant:
		av_cl2d_per_quadrant = np.mean(cls_2d_noise_per_quadrant, axis=1)

		for q in range(4):
			plot_map(np.log10(av_cl2d_per_quadrant[q]), title='log10 av cl2d quad '+str(q))
	
	plot_map(np.log10(av_cl2d), title='log10 av_cl2d')

	# correct 2d power spectrum for mask with 1/fsky
	print('mask fractions:', mask_fracs)
	av_mask_frac = np.mean(mask_fracs)
	
	print('average mask fraction:', av_mask_frac)

	av_cl2d /= av_mask_frac

	if compute_ps2d_per_quadrant:
		av_cl2d_per_quadrant /= av_mask_frac

		for q in range(4):
			plot_map(av_cl2d_per_quadrant[q], title='mask corrected av cl2d per quad '+str(q))
	
	av_means = 0.5*(np.mean(meanAs)+np.mean(meanBs))
	
	plot_map(av_cl2d, title='av cl2d')
		
	if compute_ps2d_per_quadrant:

		return av_cl2d_per_quadrant, av_means, lab_exp_diff*mask_inst, cls_2d_noise_per_quadrant

	return av_cl2d, av_means, lab_exp_diff*mask_inst, cls_2d_noise_full


def plot_noise_model_consistency_flight(inst, ifield_list=[4, 5, 6, 7, 8], startidx=1, endidx=-1, startidx_plot=0, \
									   figsize=(10, 7), tailstr=None, lmax=10000, load_cl_dpoint=False, \
										xtext = 300, text_fs=12, xlim=[150, 1.1e5], data_type='observed', bl_correct=False):
	
	cbps = CIBER_PS_pipeline()
	ciber_mock_fpath = config.ciber_basepath+'data/ciber_mocks/'

	config_dict, pscb_dict, float_param_dict, fpath_dict = return_default_cbps_dicts()
	
	fpath_dict, list_of_dirpaths, base_path, trilegal_base_path = set_up_filepaths_cbps(fpath_dict, inst, 'test', '111323',\
																					datestr_trilegal='112022', data_type=data_type, \
																				   save_fpaths=True)

	bandstr_dict = dict({1:'J', 2:'H'})
	band = bandstr_dict[inst]
	lam_dict = dict({1:1.1, 2:1.8})
	maglim_dict = dict({1:17.5, 2:17.0})
	
	base_nv_path = config.ciber_basepath+'data/noise_model_validation_data/'

	
	masking_maglim = maglim_dict[inst]
	lam = lam_dict[inst]
	
	bbox_dict = dict({'facecolor':'white', 'alpha':0.8, 'edgecolor':'None', 'pad':0.})
	if inst==1:
		ylim = [1e-1, 2e5]
		textypos = 4e4
		
		fac = 3
	else:
		ylim = [1e-2, 5e4]
		textypos = 8e3
		fac = 3
		
	masktail = 'maglim_'+band+'_Vega_'+str(masking_maglim)+'_111323_ukdebias'
	mode_couple_base_dir = fpath_dict['mkk_base_path']

	if bl_correct:
		bls_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/beam_correction/bl_est_postage_stamps_TM'+str(inst)+'_081121.npz'
		B_ells = np.load(bls_fpath)['B_ells_post']
	
	all_cls_dpoint = None
	if load_cl_dpoint:
		all_cls_dpoint = np.load(config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/cl_masked_pointing_offset/cls_masked_pointing_offset_TM'+str(inst)+'_with_wcshdrs_sigclip5_niter1.npz')['cls']
		
		fac = cbps.cal_facs[inst]/(cbps.g1_facs_factory[inst]*cbps.g2_facs_Z14[inst])
		
		all_cls_dpoint *= fac**2

	chi2_bins = np.linspace(0, 4, 30)


	all_nl1d_covs, all_nl1d_corrs, inv_Mkks = [[] for x in range(3)]

	f = plt.figure(figsize=(12, 8))

	for fieldidx, ifield in enumerate(ifield_list):
		
#         inv_Mkk_fpath = mode_couple_base_dir+'/'+masktail+'/mkk_ffest_nograd_ifield'+str(ifield)+'_observed_'+masktail+'.fits'
		inv_Mkk_fpath = mode_couple_base_dir+'/'+masktail+'/mkk_maskonly_estimate_ifield'+str(ifield)+'_observed_'+masktail+'.fits'
		inv_Mkks.append(fits.open(inv_Mkk_fpath)['inv_Mkk_'+str(ifield)].data)
	
		# flight half exposure differences
		nl_flight_diff_dir = base_nv_path + 'TM'+str(inst)+'/validationHalfExp/field'+str(ifield)+'/nl_flight_diff/'
		nl_flight_diff = np.load(nl_flight_diff_dir+'nls_flight_diff_TM'+str(inst)+'_ifield'+str(ifield)+'_'+tailstr+'_quadoffgrad.npz')
		cl_diff_flight = nl_flight_diff['cl_diff_flight']
		clerr_diff_flight = nl_flight_diff['clerr_diff_flight']
	
		# noise model realizations
		nl_noisemodl_dir = base_nv_path + 'TM'+str(inst)+'/validationHalfExp/field'+str(ifield)+'/nl_noisemodl/'
		nls_noisemodl = np.load(nl_noisemodl_dir + 'nls_noisemodl_TM'+str(inst)+'_ifield'+str(ifield)+'_'+tailstr+'_quadoffgrad.npz')
		nl1ds_diff_weighted = nls_noisemodl['nl1ds_diff_weighted']
		
		# photon noise
		nls_noisemodl = np.load(nl_noisemodl_dir + 'nls_noisemodl_TM'+str(inst)+'_ifield'+str(ifield)+'_'+tailstr+'_quadoffgrad_withphot.npz')
		nl1ds_diff_phot_weighted = np.array(nls_noisemodl['cl_diff_noisemodl_phot'])
		
		lb = nls_noisemodl['lb']
		lb_mask = (lb < lmax)*(lb >= lb[startidx])

		which_large = np.where((cl_diff_flight > np.mean(nl1ds_diff_weighted, axis=0)))[0]
		print('which large:', which_large)

		for n in range(nl1ds_diff_weighted.shape[1]):
			if n in which_large:
				nl1ds_diff_weighted[:,n] += np.random.normal(0, clerr_diff_flight[n], nl1ds_diff_weighted.shape[0])        
		
	
		cl_diff_flight = np.dot(inv_Mkks[fieldidx].transpose(), cl_diff_flight)

		for m in range(nl1ds_diff_weighted.shape[0]):
			nl1ds_diff_weighted[m] = np.dot(inv_Mkks[fieldidx].transpose(), nl1ds_diff_weighted[m])

		for m in range(nl1ds_diff_phot_weighted.shape[0]):
			
			nl1ds_diff_phot_weighted[m] = np.dot(inv_Mkks[fieldidx].transpose(), nl1ds_diff_phot_weighted[m])
			
		cl_diff_noisemodl = np.median(nl1ds_diff_weighted, axis=0)
		cl_diff_noisemodl_phot = np.median(nl1ds_diff_phot_weighted, axis=0)
			
		# noise covariance of mkk corrected spectra
		subnl1d_weighted = nl1ds_diff_weighted[:,lb_mask]
		nl1d_cov = np.cov(subnl1d_weighted.transpose())
		nl1d_corr = np.corrcoef(subnl1d_weighted.transpose())
		print('condition number is ', np.linalg.cond(nl1d_cov))
		inv_cov = np.linalg.inv(nl1d_cov)
		
		all_nl1d_covs.append(nl1d_cov)
		all_nl1d_corrs.append(nl1d_corr)
		
		cl_resid = cl_diff_flight - cl_diff_noisemodl

		if all_cls_dpoint is not None:
			mean_ptsrc_bias = np.mean(all_cls_dpoint[:,fieldidx,:], axis=0)/2.
			mean_ptsrc_bias = np.dot(inv_Mkks[fieldidx].transpose(), mean_ptsrc_bias)
			cl_resid -= mean_ptsrc_bias
		
		cl_resid = cl_resid[lb_mask]

		chi_squared = np.dot(cl_resid.transpose(), np.dot(inv_cov, cl_resid))
		pte = 1 - stats.chi2.cdf(chi_squared, len(cl_resid))
		
		# chi2 for noise realizations
		all_chi_squared_nr = []
		for nidx in range(len(nl1ds_diff_weighted)):

			cl_resid_nr = nl1ds_diff_weighted[nidx]-cl_diff_noisemodl
			cl_resid_nr = cl_resid_nr[lb_mask]
			chi_squared_nr = np.dot(cl_resid_nr.transpose(), np.dot(inv_cov, cl_resid_nr))
			all_chi_squared_nr.append(chi_squared_nr/len(cl_resid_nr))
			
		all_chi_squared_nr = np.array(all_chi_squared_nr)

		# plot the darn thing
		
		ax = f.add_subplot(2,3,fieldidx+1)
		
		prefac = lb*(lb+1)/(2*np.pi)
		
		if bl_correct:
			cl_diff_flight /= B_ells[fieldidx]**2
			cl_diff_noisemodl /= B_ells[fieldidx]**2
			cl_diff_noisemodl_phot /= B_ells[fieldidx]**2
			mean_ptsrc_bias /= B_ells[fieldidx]**2

		plt.errorbar(lb[startidx_plot:endidx], (prefac*cl_diff_flight)[startidx_plot:endidx], yerr=(prefac*clerr_diff_flight)[startidx_plot:endidx], label='Flight difference', color='C'+str(fieldidx), markersize=3, fmt='o', capsize=3)

	#     for nidx in range(len(nl1ds_diff_weighted)):
	#         plt.plot(lb[startidx:endidx], (prefac*nl1ds_diff_weighted[nidx])[startidx:endidx], color='grey', alpha=0.05)
	#     nl1d_av_cl = np.mean(nl1ds_diff_weighted, axis=0)
	
#         yerrs_sims = np.array([prefac*np.abs(cl_diff_noisemodl-np.percentile(nl1ds_diff_weighted, 16, axis=0)), prefac*np.abs(np.percentile(nl1ds_diff_weighted, 84, axis=0)-cl_diff_noisemodl)])
#         plt.errorbar(lb[startidx:endidx], (prefac*cl_diff_noisemodl)[startidx:endidx], yerr=yerrs_sims[:,startidx:endidx], fmt='o', color='k', markersize=3, alpha=0.5, capsize=3, label='Noise model')
		plt.errorbar(lb[startidx_plot:endidx], (prefac*cl_diff_noisemodl)[startidx_plot:endidx], yerr=(prefac*np.std(nl1ds_diff_weighted, axis=0))[startidx_plot:endidx], fmt='o', color='k', markersize=3, alpha=0.5, capsize=3, label='Noise model')
		
		
		plt.plot(lb[startidx_plot:endidx], (prefac*mean_ptsrc_bias)[startidx_plot:endidx], color='k', linestyle='dashed', alpha=0.4, label='Point source leakage')
#         plt.errorbar(lb[startidx:endidx], (prefac*cl_diff_noisemodl_B)[startidx:endidx], yerr=yerrs_sims[:,startidx:endidx], fmt='o', color='r', markersize=3, alpha=0.5, capsize=3, label='Noise model')
	   
		
		plt.plot(lb[startidx:endidx], (prefac*cl_diff_noisemodl_phot)[startidx:endidx], linestyle='solid', color='k', alpha=0.5, label='Photon noise')

		if fieldidx==4:
			plt.legend(fontsize=14, bbox_to_anchor=[1.25, 0.8])
		plt.xscale('log')
		plt.yscale('log')
		
#         plt.ylim(ylim)
		
		if ifield != 5:
			plt.ylim(ylim[0], ylim[1])
			plt.text(xtext, textypos, str(lam)+' $\\mu$m, '+str(cbps.ciber_field_dict[ifield]), fontsize=text_fs, bbox=bbox_dict)
			plt.text(xtext, textypos*0.3, '$\\chi^2/N_{dof}$ = '+str(np.round(chi_squared, 1))+'/'+str(len(cl_resid))+' ('+str(np.round(chi_squared/len(cl_resid), 2))+')', fontsize=text_fs, bbox=bbox_dict)
		else:
			plt.ylim(ylim[0]*3, ylim[1]*3)

			plt.text(xtext, textypos*3, str(lam)+' $\\mu$m, '+str(cbps.ciber_field_dict[ifield]), fontsize=text_fs, bbox=bbox_dict)
			plt.text(xtext, textypos*3*0.3, '$\\chi^2/N_{dof}$ = '+str(np.round(chi_squared, 1))+'/'+str(len(cl_resid))+' ('+str(np.round(chi_squared/len(cl_resid), 2))+')', fontsize=text_fs, bbox=bbox_dict)
			
#         plt.axvline(lmax, linestyle='dashed', color='k')
#         plt.axvline(0.5*(lb[0]+lb[1]), linestyle='dashed', color='k')
		
		plt.axvspan(lmax, 1.1e5, color='grey', alpha=0.2)
		# plt.axvspan(xlim[0], 0.5*(lb[0]+lb[1]), color='grey', alpha=0.2)
		plt.xlim(xlim)
		
		plt.tick_params(labelsize=12)

		# if fieldidx > 1:
		plt.xlabel('$\\ell$', fontsize=16)
		if fieldidx==0 or fieldidx==3:
			plt.ylabel('$D_{\\ell}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=16)
		plt.grid(alpha=0.5)
		
		plt.tick_params(labelsize=10)
		
		# inset with rank distribution
		axin = inset_axes(ax, width="45%", # width = 30% of parent_bbox
						height=0.75, # height : 1 inch
						loc=4, borderpad=1.6)
	
		ndof = nl1ds_diff_weighted.shape[1]

		chistat_order_idx = np.argsort(all_chi_squared_nr)
		
		plt.hist(all_chi_squared_nr, bins=chi2_bins, linewidth=1, histtype='stepfilled', color='k', alpha=0.2, label=cbps.ciber_field_dict[ifield])
		plt.axvline(np.mean(all_chi_squared_nr), color='k', alpha=0.5)

		pte = 1.-(np.digitize(chi_squared/len(cl_resid), all_chi_squared_nr[chistat_order_idx]))/len(all_chi_squared_nr)
		plt.axvline(chi_squared/len(cl_resid), color='C'+str(fieldidx), linestyle='solid', linewidth=2, label='Observed data')
		print('pte = ', pte)
		axin.set_xlabel('$\\chi^2_{red}$', fontsize=10)
		axin.xaxis.set_label_position('top')
		plt.xticks([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
		plt.xlim(0, 4)
		xpos_inset_text = 1.3
		axin.tick_params(labelsize=8,bottom=True, top=True, labelbottom=True, labeltop=False)
		plt.yticks([], [])

		hist = np.histogram(all_chi_squared_nr, bins=chi2_bins)
		plt.text(xpos_inset_text, 0.8*np.max(hist[0]), 'Noise model', color='grey', fontsize=9, bbox=bbox_dict)

		plt.text(xpos_inset_text, 0.6*np.max(hist[0]), 'PTE='+str(np.round(pte, 3)), color='C'+str(fieldidx), fontsize=9, bbox=bbox_dict)


	plt.subplots_adjust(wspace=0.2, hspace=0.2)
	plt.show()
	
	return f


def compute_cross_halfexpdiff(flight_diff_basepath, ifield_list = [4, 5, 6, 7, 8], tailstr = 'halfexp_full_051324', plot=False, sign_flip=False):
	all_clproc, all_clprocerr = [], []

	cbps = CIBER_PS_pipeline()

	for fieldidx, ifield in enumerate(ifield_list):
		

		nl_savedir_TM1 = flight_diff_basepath + 'TM1/validationHalfExp/field'+str(ifield)+'/nl_flight_diff/'
		nl_save_fpath_TM1 = nl_savedir_TM1+'nls_flight_diff_TM1_ifield'+str(ifield)+'_'+tailstr
		flight_diff_TM1 = np.load(nl_save_fpath_TM1+'.npz')['flight_diff']
		
		
		nl_savedir_TM2 = flight_diff_basepath + 'TM2/validationHalfExp/field'+str(ifield)+'/nl_flight_diff/'
		nl_save_fpath_TM2 = nl_savedir_TM2+'nls_flight_diff_TM2_ifield'+str(ifield)+'_'+tailstr
		flight_diff_TM2 = np.load(nl_save_fpath_TM2+'.npz')['flight_diff']
		
		flight_masked_TM2_to_TM1, footprint = regrid_arrays_by_quadrant(flight_diff_TM2, ifield, inst0=1, inst1=2, quad_list=['A', 'B', 'C', 'D'], \
						 xoff=[0,0,512,512], yoff=[0,512,0,512], plot=False, astr_dir=config.ciber_basepath+'data/')

		
		flight_diff_TM1[flight_masked_TM2_to_TM1==0] = 0.
		flight_masked_TM2_to_TM1[flight_diff_TM1==0] = 0.
		
		if plot:
			plot_map(flight_diff_TM1, title='TM1 flight diff', cmap='bwr', figsize=(6,6))
			plot_map(flight_masked_TM2_to_TM1, title='TM2 flight diff regrid', cmap='bwr', figsize=(6,6))
			
		flight_diff_TM1[flight_diff_TM1 != 0] -= np.mean(flight_diff_TM1[flight_diff_TM1 != 0])
		
		flight_masked_TM2_to_TM1[flight_masked_TM2_to_TM1 != 0] -= np.mean(flight_masked_TM2_to_TM1[flight_masked_TM2_to_TM1 != 0])


		if sign_flip:
			print('flipping TM2 regrid..')
			flight_masked_TM2_to_TM1 *= -1.
			
		lb, cl_proc, cl_proc_err = get_power_spec(flight_diff_TM1, map_b=flight_masked_TM2_to_TM1, mask=None, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)

		all_clproc.append(cl_proc)
		all_clprocerr.append(cl_proc_err)
		
		
	return lb, all_clproc, all_clprocerr

def read_noise_modl_consistency_test(cbps_nm, inst, tailstr_noisemodl, nsims=500, n_split=10, estimate_noise_power=False, \
									noise_model_dir=None, ifield_list=[4, 5, 6, 7, 8], per_quadrant=False, plot=False, \
									ymin=1e-2, ymax=1e4, apply_filtering=True, dd_tailstr=None, halves=False, \
									xlim=[150, 1e5], textxpos=200, textypos=300, ylim_ratio=[-1, 3.0]):
	
	sterad_per_pix = (cbps_nm.cbps.pixsize/3600/180*np.pi)**2
	V = cbps_nm.cbps.dimx*cbps_nm.cbps.dimy*sterad_per_pix
	conv_fac = cbps_nm.cbps.cal_facs[inst]**2*V
	
	if dd_tailstr is None:
		dd_tailstr = tailstr_noisemodl
	
	if per_quadrant:
		addstr = '_per_quadrant'
	else:
		addstr = '_nofilter'
	
	if noise_model_dir is None:
		noise_model_dir = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/noise_model/'
		
	
	all_cl_modl_mean, all_cl_modl_std, all_cls_dd_mean,\
		all_cls_dd_std, all_mean_nl2d, all_cls_1d_unweighted, figs, var_ratios = [[] for x in range(8)]
	
	print('Dark difference tail string is ', dd_tailstr)
	print('Noise model tail string is ', tailstr_noisemodl)
	
	for fieldidx, ifield in enumerate(ifield_list):
		
		if halves:
			noise_modl_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/noise_model/halfexp/noise_model_TM'+str(inst)+'_ifield'+str(ifield)+'_'+tailstr_noisemodl+'.fits'
		else:
			noise_modl_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/noise_model/noise_model_TM'+str(inst)+'_ifield'+str(ifield)+'_'+tailstr_noisemodl+'.fits'
		
		# open cl2d of dark differences
		darkdiff_fpath = noise_model_dir+'dark_exp_diffs/dark_exp_diff_TM'+str(inst)+'_ifield'+str(ifield)+'_'+dd_tailstr+'.npz'
		darkdiff_cl2ds = np.load(darkdiff_fpath)['cls_2d_noise_full']

		noisemodl = fits.open(noise_modl_fpath)[1].data

		maskInst_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/masks/maskInst_102422/field'+str(ifield)+'_TM'+str(inst)+'_maskInst_102422.fits'
		mask_inst = fits.open(maskInst_fpath)['maskInst'].data # 10/24/22

		mkkpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/mkk/mkk_maskonly_estimate_ifield'+str(ifield)+'_inst'+str(inst)+'_maskInst_nmc500_102422.fits'
		inv_Mkk = fits.open(mkkpath)['inv_Mkk_'+str(ifield)].data
		inv_mkk_diag = np.array([1./inv_Mkk[i,i] for i in range(inv_Mkk.shape[0])])

		if estimate_noise_power:

			if halves:
				field_nfr = cbps_nm.cbps.field_nfrs[ifield]//2
				print('field nfr = ', field_nfr)
			else:
				field_nfr = cbps_nm.cbps.field_nfrs[ifield]
			fourier_weights, mean_nl2d, cls_1d_unweighted = cbps_nm.cbps.estimate_noise_power_spectrum(inst=inst, ifield=ifield, mask=mask_inst, apply_mask=True, \
																						   noise_model=noisemodl, photon_noise=False, read_noise=True, inplace=False, \
																							field_nfr=field_nfr, nsims=nsims, n_split=n_split, \
																							 compute_1d_cl=True, per_quadrant=per_quadrant, gradient_filter=True, \
																									apply_filtering=apply_filtering)

			
			noisemodl_save_fpath = noise_model_dir+'fourier_weights_TM'+str(inst)+'_ifield'+str(ifield)+'_'+tailstr_noisemodl+addstr+'.npz'
			print('Saving noise model realizations to ', noisemodl_save_fpath)
			np.savez(noisemodl_save_fpath, nsim=nsims, \
					 fourier_weights=fourier_weights, mean_nl2d=mean_nl2d, cls_1d_unweighted=cls_1d_unweighted)

		lb, cl_modl, clerr_modl, cls_dd,\
			clerrs_dd, mean_nl2d, cls_1d_unweighted = cbps_nm.compare_read_noise_modl_dark_dat(inst, ifield, darkdiff_cl2ds, inv_Mkk=inv_Mkk, \
																	fw_basepath=noise_model_dir, apply_FW=False, tailstr=tailstr_noisemodl+addstr, load_cl1d=True, clip_sigma=None)


		clmodlstd = 0.5*(np.percentile(np.array(cls_1d_unweighted), 84, axis=0)-np.percentile(np.array(cls_1d_unweighted), 16, axis=0))

		cls_dd_mean = np.mean(cls_dd, axis=0)
		cls_dd_std = np.std(cls_dd, axis=0)
		
		all_cl_modl_mean.append(cl_modl)
		all_cl_modl_std.append(clmodlstd)
		all_cls_dd_mean.append(cls_dd_mean)
		all_cls_dd_std.append(cls_dd_std)
		all_cls_1d_unweighted.append(cls_1d_unweighted)
		all_mean_nl2d.append(mean_nl2d)
		
		if plot:
			
			prefac = cbps_nm.cbps.cal_facs[inst]**2*lb*(lb+1)/(2*np.pi)
			fig = plot_compare_rdnoise_darkdiff_cl(inst, cbps_nm.cbps.ciber_field_dict[ifield], lb, prefac,\
									   cl_modl, clmodlstd, cls_dd_mean, cls_dd_std, \
										   ymin=ymin, ymax=ymax, textypos=textypos, textxpos=textxpos, xlim=xlim, \
							   ylim_ratio=ylim_ratio)
			
			figs.append(fig)
			
			
	rd_valid_dict = dict({'lb':lb, 'all_cl_modl_mean':all_cl_modl_mean, 'all_cl_modl_std':all_cl_modl_std, 'all_cls_dd_mean':all_cls_dd_mean, \
						 'all_cls_dd_std':all_cls_dd_std, 'all_cls_1d_unweighted':all_cls_1d_unweighted, 'all_mean_nl2d':all_mean_nl2d})
	
	return rd_valid_dict, figs

