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
from ciber_data_helpers import *
from helgason import *
from ps_pipeline_go import *
from cross_spectrum_analysis import *
from mkk_parallel import *
from mkk_diagnostics import *
from flat_field_est import *
from plotting_fns import *
from powerspec_utils import *


# experimented with skew functions
def randn_skew_fast(N=None, shape=None, alpha=0.0, loc=0.0, scale=1.0):
	if shape is None:
		shape = (N,N)
	sigma = alpha / np.sqrt(1.0 + alpha**2) 
	u0 = np.random.normal(0, 1, shape)
	v = np.random.normal(0, 1, shape)
	u1 = (sigma*u0 + np.sqrt(1.0 - sigma**2)*v) * scale
	u1[u0 < 0] *= -1
	u1 = u1 + loc
	return u1

def generate_rand_skew_layered(skew_upsamp, realiz_shape, skew_max=8):
	
	skew_realiz = np.zeros(realiz_shape)
	
	skew_upsamp_round = np.round(skew_upsamp)
	skew_upsamp_round[skew_upsamp_round > skew_max] = skew_max
	
	plot_map(skew_upsamp_round, title='skew upsamp round')
	
	for skewval in range(skew_max+1):
		
		if skewval==0:
			p = np.ones_like(skew_realiz)
		else:
			p = randn_skew_fast(shape=realiz_shape, alpha=skewval, loc=1.0)
		
		skew_realiz_pmask = np.broadcast_to(skew_upsamp_round==skewval, realiz_shape)
		p[~skew_realiz_pmask] = 0.
		skew_realiz += p
		
		
	print('skew realiz hasshape ', skew_realiz.shape)
	plot_map(skew_realiz[0], title='skew realiz')
	
	return skew_realiz

def weighted_powerspec_2d(ps2d, l2d_mask, fw_image=None):
	ps2d_copy = ps2d.copy()
	ps2d_copy[l2d_mask==0] = 0.

	if fw_image is not None:
		fw_image_copy = fw_image.copy()
		
		fw_image_copy[ps2d_copy==0.] = 0.
		fw_image_copy[fw_image_copy > 0.] /= np.sum(fw_image_copy) 

		ps2d_copy *= fw_image_copy
				
		return ps2d_copy, fw_image_copy
	
	return ps2d_copy

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



def load_read_noise_modl_filedat(fw_basepath, inst, ifield, tailstr=None, apply_FW=False, load_cl1d=False, \
								mean_nl2d_key='mean_nl2d', fw_key='fourier_weights'):
	fw_fpath = fw_basepath+'/fourier_weights_TM'+str(inst)+'_ifield'+str(ifield)
	if tailstr is not None:
		fw_fpath += '_'+tailstr
	print('loading fourier weights and mean nl2d from ', fw_fpath+'.npz')
	
	fourier_weights=None 
	nm_file = np.load(fw_fpath+'.npz')

	if apply_FW:
		fourier_weights = nm_file[fw_key]
	
	mean_nl2d = nm_file[mean_nl2d_key]  
	cl1d_unweighted = None 
	if load_cl1d:
		cl1d_unweighted = nm_file['cls_1d_unweighted']

	return mean_nl2d, fourier_weights, cl1d_unweighted

class CIBER_NoiseModel():
	
	
	
	def __init__(self, n_ps_bin=25, ifield_list=[4, 5, 6, 7, 8], cbps=None, save_fpath=None, base_path=None):
		
		self.ifield_list = ifield_list
		if cbps is None:
			cbps = CIBER_PS_pipeline(n_ps_bin=n_ps_bin)
		self.cbps = cbps
		
		if save_fpath is None:
			save_fpath = config.exthdpath +'ciber_fluctuation_data/'
			print('save directory is ', save_fpath)

		if base_path is None:
			base_path = config.exthdpath+'noise_model_validation_data/'
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
		
		noise_model_dir = self.save_fpath+'TM'+str(inst)+'/noise_model/'

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
			
			maskInst_fpath = self.save_fpath+'TM'+str(inst)+'/masks/maskInst_102422/field'+str(ifield)+'_TM'+str(inst)+'_maskInst_102422.fits'
			mask_inst = fits.open(maskInst_fpath)['maskInst'].data # 10/24/22

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
																 plot=plot, cal_facs=None, maskInst_fpath=maskInst_fpath, \
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
					noise_model_fpath = noise_model_dir+'noise_model_TM'+str(inst)+'_ifield'+str(ifield)+'_'+tailstr+'.fits'
					noise_model_fpaths.append(noise_model_fpath)
					hdul.writeto(noise_model_fpath, overwrite=True)

		return noise_models, dark_exp_diff_fpaths, noise_model_fpaths
	
	def compare_read_noise_modl_dark_dat(self, inst, ifield, darkdiff_cl2ds, inv_Mkk=None, av_mask_frac=None,\
		 mean_nl2d_key='mean_nl2d', fw_key='fourier_weights', fw_basepath='/Users/richardfeder/Downloads',\
		 tailstr=None, apply_FW=False, load_cl1d=False, clip_sigma=5):



		mean_nl2d, xx, yy  = load_read_noise_modl_filedat(fw_basepath, inst, ifield, tailstr=tailstr, apply_FW=apply_FW, load_cl1d=load_cl1d, \
								mean_nl2d_key='mean_nl2d', fw_key='fourier_weights')

		# fw_fpath = fw_basepath+'/fourier_weights_TM'+str(inst)+'_ifield'+str(ifield)
		# if tailstr is not None:
		# 	fw_fpath += '_'+tailstr
		# print('loading fourier weights and mean nl2d from ', fw_fpath+'.npz')
		
		# fourier_weights=None 
		# nm_file = np.load(fw_fpath+'.npz')

		# if apply_FW:
		# 	fourier_weights = nm_file[fw_key]
		
		# mean_nl2d = nm_file[mean_nl2d_key]   
		# if load_cl1d:
		# 	cl1d_unweighted = nm_file['cls_1d_unweighted']

				
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
				
		lb, cl_modl, clerr_modl = azim_average_cl2d(mean_nl2d, l2d, weights=fourier_weights, lbinedges=self.cbps.Mkk_obj.binl, lbins=self.cbps.Mkk_obj.midbin_ell, return_weighted_clerr=True)

		if inv_Mkk is not None:
			cl_modl = np.dot(inv_Mkk.transpose(), cl_modl)
		elif av_mask_frac is not None:
			cl_modl /= av_mask_frac
			
		if load_cl1d:
			return lb, cl_modl, clerr_modl, cls_dd, clerrs_dd, mean_nl2d, cl1d_unweighted

		return lb, cl_modl, clerr_modl, cls_dd, clerrs_dd, mean_nl2d

	
	
	def compute_lab_flight_exp_differences(self, ifield, inst, base_nv_path=None, mask=None, mask_fpath=None, \
										  mask_inst=None, stdpower=2, flight_exp_diff=None, verbose=False, plot=False, clip_sigma = 5, nitermax=10, \
										   compute_noise_cov=False, return_read_nl2d=False, \
										   nsims=500, n_split=10, chisq=False, per_quadrant=False, gradient_filter=False, mean_nl2d_diff=None, \
										   compute_weighted_1d_nls=False, fourier_weights=None, load_fw_mean_nl2d=False):

		''' 
		mask_inst used if one needs a 2D power spectrum for the read noise alone, as is used in compute_noise_covariance_matrix()
		'''

		if base_nv_path is None:
			base_nv_path = config.exthdpath+'noise_model_validation_data/'
		if mask_fpath is None and mask is None:
			mask_fpath = config.exthdpath+'ciber_fluctuation_data/TM'+str(inst)+'/masks/joint_mask_with_ffmask_ifield'+str(ifield)+'_inst'+str(inst)+'.fits'
			print('both mask_fpath and mask are None, loading mask from default ', mask_fpath)        
			mask = fits.open(mask_fpath)['joint_mask_'+str(ifield)].data
			if plot:
				plot_map(mask, title='mask')
		if mask_inst is not None:
			print('we have instrument mask')
			plot_map(mask_inst, title='mask inst')

		data_path = base_nv_path+'TM'+str(inst)+'/validationHalfExp/field'+str(ifield)

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
			flight_exp_diff, meanA, meanB, mean_flight_test, expA, expB = compute_exp_difference(inst, cal_facs=self.cbps.cal_facs, pathA=pathA, pathB=pathB, mask=mask, mode='flight', return_maps=True, \
																								per_quadrant=per_quadrant)

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

		x0s = [0, 0, 512, 512]
		x1s = [512, 512, 1024, 1024]
		y0s = [0, 512, 0, 512]
		y1s = [512, 1024, 512, 1024]

		if per_quadrant:
			mean_image = np.ones_like(flight_exp_diff)
			for q in range(4):
				mean_image[x0s[q]:x1s[q],y0s[q]:y1s[q]] = 0.5*(meanA[q]+meanB[q])

		else:
			mean_image = 0.5*(meanA+meanB)*np.ones_like(flight_exp_diff)

		shot_sigma_sb_map = self.cbps.compute_shot_sigma_map(inst, mean_image, nfr=self.cbps.field_nfrs[ifield]//2)

		# shot_sigma_sb_map = self.cbps.compute_shot_sigma_map(inst, 0.5*(meanA+meanB)*np.ones_like(flight_exp_diff), nfr=self.cbps.field_nfrs[ifield]//2)
		dsnmaps = np.sqrt(2)*shot_sigma_sb_map*np.random.normal(0, 1, size=imarray_shape)

		for i in range(n_image):

			pathA, pathB = labexp_fpaths1[i], labexp_fpaths2[i]
			if verbose:
				print('i = ', i, ' pathA:', pathA)
				print('i = ', i, ' pathB:', pathB)

			dark_exp_diff, darkmeanA, darkmeanB, mean_dark = compute_exp_difference(inst, mask=mask, cal_facs=self.cbps.cal_facs, pathA=pathA, pathB=pathB, per_quadrant=per_quadrant)
			dark_sc_mask = iter_sigma_clip_mask(dark_exp_diff, sig=clip_sigma, nitermax=nitermax, mask=mask.astype(int))
			dark_exp_diff += dsnmaps[i]

			if per_quadrant:
				for q in range(4):
					mquad_sc = dark_sc_mask[x0s[q]:x1s[q],y0s[q]:y1s[q]]

					dark_exp_diff[x0s[q]:x1s[q],y0s[q]:y1s[q]][mquad_sc==1] -= np.mean(dark_exp_diff[x0s[q]:x1s[q],y0s[q]:y1s[q]][mquad_sc==1])
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

		print('Estimating noise power spectrum mean and Fourier weights..')
		plot_map(shot_sigma_sb_map, title='shot_sigma_sb_map')

		if mean_nl2d_diff is None:
			mean_nl2d_diff = mean_nl2d_dark

		if not load_fw_mean_nl2d:
			# if mean_nl2d_diff is None:
			# 	mean_nl2d_diff = mean_nl2d_dark

			fourier_weights, mean_nl2d_modl, nl1ds_diff = self.cbps.estimate_noise_power_spectrum(noise_model=mean_nl2d_diff, mask=mask, shot_sigma_sb=shot_sigma_sb_map, field_nfr=self.cbps.field_nfrs[ifield]//2,\
																				 nsims=nsims, n_split=n_split, inst=inst, ifield=ifield, photon_noise=True, read_noise=True, inplace=False, chisq=chisq, \
																				 difference=True, compute_1d_cl=True)
			fourier_weights /= np.nanmax(fourier_weights)
			plot_map(np.log10(fourier_weights), title='log10(w($\\ell_x, \\ell_y$))')




		if compute_weighted_1d_nls and fourier_weights is not None:
			print('now computing weighted 1d nls..')
			_, mean_nl2d_modl, nl1ds_diff_weighted = self.cbps.estimate_noise_power_spectrum(noise_model=mean_nl2d_diff, mask=mask, shot_sigma_sb=shot_sigma_sb_map, field_nfr=self.cbps.field_nfrs[ifield]//2,\
																	 nsims=nsims, n_split=n_split, inst=inst, ifield=ifield, photon_noise=True, read_noise=True, inplace=False, chisq=chisq, \
																	 difference=True, compute_1d_cl=True, fw_diff=fourier_weights)
			nl1ds_diff = nl1ds_diff_weighted
		else:
			nl1ds_diff_weighted, nl1ds_diff = None, None

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

			# cl_diffs_noisemodl[i] = cl_diff_noisemodl/2.
			cl_diffs_phot[i] = cl_diff_phot/2.
			cl_diffs_darkexp[i] = cl_diff_darkexp/2.

		if per_quadrant:
			print('subtracting flight data per quadrant')
			x0s = [0, 0, 512, 512]
			x1s = [512, 512, 1024, 1024]
			y0s = [0, 512, 0, 512]
			y1s = [512, 1024, 512, 1024]

			flight_masked_diff = (flight_exp_diff*flight_sc_mask).copy()
			
			for q in range(4):
				mquad = flight_sc_mask[x0s[q]:x1s[q], y0s[q]:y1s[q]]

				if gradient_filter:
					theta_quad, plane_quad = fit_gradient_to_map(flight_exp_diff[x0s[q]:x1s[q], y0s[q]:y1s[q]], mask=mquad)
					flight_exp_diff[x0s[q]:x1s[q], y0s[q]:y1s[q]] -= plane_quad

				flight_exp_diff[x0s[q]:x1s[q], y0s[q]:y1s[q]][mquad==1] -= np.mean(flight_masked_diff[x0s[q]:x1s[q], y0s[q]:y1s[q]][mquad==1])

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
				mask_tail = 'maglim_J_Vega_'+str(masking_maglim)+'_112922'
			elif inst==2:
				mask_tail = 'maglim_H_Vega_'+str(masking_maglim)+'_112922'

		for fieldidx, ifield in enumerate(ifield_list):
			if ifield in ifield_plot:
				plot=True
			else:
				plot=False
				
			if mask_base_path is None:
				mask_base_path = config.exthdpath+'ciber_fluctuation_data/TM'+str(inst)+'/masks/'

			mask_fpath = mask_base_path+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'
			full_mask = fits.open(mask_fpath)[1].data.astype(np.int)    

			if load_inst_mask:
				maskInst_fpath = self.save_fpath+'TM'+str(inst)+'/masks/maskInst_102422/field'+str(ifield)+'_TM'+str(inst)+'_maskInst_102422.fits'
				mask_inst = fits.open(maskInst_fpath)['maskInst'].data # 10/24/22
				full_mask *= mask_inst.astype(np.int)

			mean_nl2d_diff=None
			if noisemodl_tailstr is not None:

				noise_model_dir = config.exthdpath+'ciber_fluctuation_data/TM'+str(inst)+'/noise_model/'

				mean_nl2d_diff, _, _ = load_read_noise_modl_filedat(noise_model_dir, inst, ifield, tailstr=noisemodl_tailstr, apply_FW=False, load_cl1d=False, \
																						mean_nl2d_key='mean_nl2d')

		
				plot_map(full_mask, title='full mask')

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
					shot_sigma_sb_map, nl1ds_diff_weighted = self.compute_lab_flight_exp_differences(ifield, inst, mask=full_mask, clip_sigma=clip_sigma, nitermax=10,\
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
			plt.xlabel('$\\ell$', fontsize=14)
			plt.ylabel('$D_{\\ell}$', fontsize=14)
			plt.tick_params(labelsize=12)
			plt.show()
			# lb, cl_diff_flight, clerr_diff_flight,\
			# 	cl_diffs_noisemodl, cl_diffs_phot,\
			# 	expA, expB, fourier_weights, cl2ds_dark,\
			# 	mean_nl2d_read, proc_nls, noise_cov,\
			# 		shot_sigma_sb_map = self.compute_lab_flight_exp_differences(ifield, inst, mask=full_mask, clip_sigma=clip_sigma, nitermax=10,\
			# 																   stdpower=2, compute_noise_cov=False, plot=plot,\
			# 																   mask_inst=mask_inst, nsims=nsims, n_split=n_split, \
			# 																   per_quadrant=per_quadrant, mask_fpath=mask_fpath, verbose=verbose, gradient_filter=gradient_filter)


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
				fpath_save = config.exthdpath+'noise_model_validation_data/TM'+str(inst)+'/validationHalfExp/field'+str(ifield)+'/powerspec/cls_expdiff_lab_flight_TM'+str(inst)+'_ifield'+str(ifield)+'_'+tailstr+'.npz'
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

		if save_nvdat:
			return fpaths_save
				

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
												  read_noise=read_noise, photon_noise=False, chisq=False)

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



