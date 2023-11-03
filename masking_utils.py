import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from ciber_data_helpers import make_radius_map, compute_radmap_full
from plotting_fns import plot_map
import config
import astropy.wcs as wcs
from astropy import units as u
from astropy.io import fits
import scipy
import pandas as pd
from powerspec_utils import write_mask_file, write_Mkk_fits

# from ciber_source_mask_construction_pipeline import find_alpha_beta

''' TO DO : CAMB does not compile with Python 3 at the moment -- need to update Fortran compiler '''
import sys

if sys.version_info[0]==2:
	from ciber_mocks import *

# Yun-Ting's code for masking is here https://github.com/yuntingcheng/python_ciber/blob/master/stack_modelfit/mask.py


def make_fpaths(fpaths):
	for fpath in fpaths:
		if not os.path.isdir(fpath):
			print('making directory path for ', fpath)
			os.makedirs(fpath)
		else:
			print(fpath, 'already exists')

def find_alpha_beta(intercept, minrad=10, dm=3, pivot=16.):
	
	alpha_m = -(intercept - minrad)/dm
	beta_m = intercept - pivot*alpha_m
	
	return alpha_m, beta_m


def filter_trilegal_cat(trilegal_cat, m_min=4, m_max=17, filter_band_idx=16):
	
	filtered_trilegal_cat = np.array([x for x in trilegal_cat if x[filter_band_idx]<m_max and x[filter_band_idx]>m_min])

	return filtered_trilegal_cat

def radius_vs_mag_gaussian(mags, a1, b1, c1):
	return a1*np.exp(-((mags-b1)/c1)**2)

def magnitude_to_radius_linear(magnitudes, alpha_m=-6.25, beta_m=110.):
	''' Masking radius function as given by Zemcov+2014. alpha_m has units of arcsec mag^{-1}, while beta_m
	has units of arcseconds.

	Parameters
	----------
	
	magnitudes : float, or list of floats
		Source magnitudes used to find masking radii. 

	alpha_m : float, optional
		Slope of radius/magnitude equation. Default is -6.25. 

	beta_m : float, optional
		Zero point of radius/magnitude equation. Default from Zemcov+14 is 110.

	Returns
	-------

	r : float, or list of floats
		masking radii for input sources 

	'''
	
	r = alpha_m*magnitudes + beta_m

	return r

def make_synthetic_trilegal_cat(trilegal_path, J_band_idx=16, H_band_idx=17, imdim=1024.):
	trilegal = np.loadtxt(trilegal_path)
	nsrc = trilegal.shape[0]
	synthetic_cat = np.random.uniform(0, imdim, size=(nsrc, 2))
	synthetic_cat = np.array([synthetic_cat[:,0], synthetic_cat[:,1], trilegal[:,J_band_idx], trilegal[:,H_band_idx]]).transpose()
	
	print('synthetic cat has shape ', synthetic_cat.shape)
	
	return synthetic_cat


def make_blob_mask(cbps, inst):
    blobcat = pd.read_csv('data/bad_detector_positions_TM'+str(inst)+'.csv', header=None, names=['x', 'y'])
    additional_maskx = np.array(blobcat['x'])
    additional_masky = np.array(blobcat['y'])
    cluster_radius = 21
    blob_mask = np.ones((cbps.dimx, cbps.dimy))
    for addidx, cenx in enumerate(additional_maskx):
        radmap = make_radius_map(dimx=cbps.dimx, dimy=cbps.dimy, cenx=cenx, ceny=additional_masky[addidx], sqrt=False)
        blob_mask[radmap<cluster_radius**2/cbps.pixsize**2] = 0.

    return blob_mask

def make_cluster_mask(cbps, inst, ifield):
    wencat = pd.read_csv('data/cluster_dat/cluster_catalog_wen12_sdss_'+cbps.ciber_field_dict[ifield]+'.csv')
    additional_maskx = np.array(wencat['x'+str(inst)])
    additional_masky = np.array(wencat['y'+str(inst)])
    cluster_radius = 21
    cluster_mask = np.ones((cbps.dimx, cbps.dimy))
    for addidx, cenx in enumerate(additional_maskx):
        radmap = make_radius_map(dimx=cbps.dimx, dimy=cbps.dimy, cenx=cenx, ceny=additional_masky[addidx], sqrt=False)
        cluster_mask[radmap<cluster_radius**2/cbps.pixsize**2] = 0.
    return cluster_mask

def make_bright_elat30_mask(cbps, inst):
    
    elat30_star_radius = 500
    elat30_mask = np.ones((cbps.dimx, cbps.dimy))

    if inst==1:
        cenx, ceny = 270, 420
    elif inst==2:
        cenx, ceny = 430, 720

    radmap = make_radius_map(dimx=cbps.dimx, dimy=cbps.dimy, cenx=cenx, ceny=ceny, sqrt=False)
    elat30_mask[radmap<elat30_star_radius**2/cbps.pixsize**2] = 0.

    return elat30_mask



def mask_from_cat(xs=None, ys=None, mags=None, cat_df=None, dimx=1024, dimy=1024, pixsize=7.,\
					interp_maskfn=None, mode=None, magstr='zMeanPSFMag', alpha_m=-6.25, beta_m=110, a1=252.8, b1=3.632, c1=8.52,\
					 Vega_to_AB = 0., mag_lim_min=0, mag_lim=None, fixed_radius=None, radii=None, compute_radii=True, inst=1, \
					radmap_full=None, rc=1., plot=True, interp_max_mag=None, interp_min_mag=None, m_min_thresh=None, radcap=200., \
					mag_fudge=14, fudge_fac=None, mask_transition_mag=14):
	
	if fixed_radius is not None or radii is not None:
		compute_radii = False
		
	mask = np.ones([dimx,dimy], dtype=int)
	
	if compute_radii:
		if mags is None and cat_df is None:
			print('Need magnitudes one way or another to compute radii, please specify with mags parameter or cat_df..')
			return
		if cat_df is None:
			if xs is None or ys is None or mags is None:
				print('cat_df=None, but no input information for xs, ys, mags..')
				return

	if mag_lim is not None:
		if cat_df is not None:
			mag_lim_mask = np.where((cat_df[magstr] < mag_lim)&(cat_df[magstr] > mag_lim_min))[0]
			cat_df = cat_df.iloc[mag_lim_mask]
			xs = np.array(cat_df['x'+str(inst)])
			ys = np.array(cat_df['y'+str(inst)])

			if radii is not None:
				radii = radii[mag_lim_mask]
			print('length after cutting on '+magstr+'< '+str(mag_lim)+' is '+str(len(xs)))

			
		elif mags is not None:
			mag_lim_mask = (mags < mag_lim)*(mags > mag_lim_min)
			mags = mags[mag_lim_mask]
			xs = xs[mag_lim_mask]
			ys = ys[mag_lim_mask]

	if interp_maskfn is not None:
		# magnitudes need to be in Vega system as this is how interp_maskfn is defined!! 
		print("Using interpolated function to get masking radii..")
		if cat_df is not None and len(cat_df) > 0:
			mags = cat_df[magstr]

			print('max mag is ', np.max(mags), np.nanmax(mags))
			print('interp max mag is ', interp_max_mag)

			if interp_max_mag is not None:
				mags[mags > interp_max_mag] = interp_max_mag
			if interp_min_mag is not None:
				mags[mags < interp_min_mag] = interp_min_mag

			radii = interp_maskfn(np.array(mags))

			if m_min_thresh is not None:
				radii[np.array(mags) < m_min_thresh] = radcap

			# if fudge_fac is not None:
			# 	radii[np.array(mags) > mag_fudge] *= 2

			if plot:
				plt.figure()
				plt.scatter(mags, radii, s=3, color='k')
				plt.xlabel('Vega mags')
				plt.ylabel('radii [arcsec]')
				plt.show()
		else:
			return None, []

	if compute_radii or radii is None:
		print('Computing radii based on magnitudes..')
		if cat_df is not None:
			mags = cat_df[magstr]
		if mode=='Zemcov+14':

			if interp_maskfn is not None:
				print('using zemcov 14 radii for sources fainter than ', mask_transition_mag)
				radii_z14 = magnitude_to_radius_linear(mags[mags > mask_transition_mag], alpha_m=alpha_m, beta_m=beta_m)
				print('radii z14:', radii_z14)
				radii[mags > mask_transition_mag] = radii_z14
			else:
				radii = magnitude_to_radius_linear(mags, alpha_m=alpha_m, beta_m=beta_m)
			print('alpha, beta = ', alpha_m, beta_m)
		elif mode=='Simon':
			AB_mags = np.array(cat_df[magstr]) + Vega_to_AB
			radii = radius_vs_mag_gaussian(mags, a1=a1, b1=b1, c1=c1)

	if radii is not None:
		for i, r in enumerate(radii):
			radmap = make_radius_map(dimx=dimx, dimy=dimy, cenx=xs[i], ceny=ys[i], sqrt=False)
			mask[radmap<r**2/pixsize**2] = 0.
			
		return mask, radii

	else:
		if radmap_full is None:
			xx, yy = compute_meshgrids(dimx, dimy)
			radmap_full = compute_radmap_full(xs, ys, xx, yy)
		
		if fixed_radius is not None:
			thresh = fixed_radius**2/(pixsize*pixsize)
			mask[(radmap_full < thresh*rc**2)] = 0.
 
		return mask, radmap_full, thresh
	

def get_masks(star_cat_df, mask_fn_param_combo, intercept_mag_AB, mag_lim_AB, inst=1, instrument_mask=None, minrad=14., dm=3, dimx=1024, dimy=1024, verbose=True, Vega_to_AB=0., magstr='j_m'):
	
	'''
	Computes astronomical source mask for catalog

	'''


	intercept = radius_vs_mag_gaussian(intercept_mag_AB, a1=mask_fn_param_combo[0], b1=mask_fn_param_combo[1],\
									   c1=mask_fn_param_combo[2])
	
	alpha_m, beta_m = find_alpha_beta(intercept, minrad=minrad, dm=dm, pivot=intercept_mag_AB)
	
	if verbose:
		print('param_combo is ', mask_fn_param_combo)
		print('making bright star mask..')
		
	mask_stars_simon, radii_stars_simon = mask_from_cat(cat_df = star_cat_df, mag_lim_min=0, inst=inst,\
																	mag_lim=intercept_mag_AB, mode='Simon', a1=mask_fn_param_combo[0], b1=mask_fn_param_combo[1], c1=mask_fn_param_combo[2], magstr=magstr, Vega_to_AB=Vega_to_AB, dimx=dimx, dimy=dimy)


	if verbose:
		print('alpha, beta are ', alpha_m, beta_m)
		print('making faint source mask..')
	mask_stars_Z14, radii_stars_Z14 = mask_from_cat(cat_df = star_cat_df, inst=inst, mag_lim_min=intercept_mag_AB, mag_lim=mag_lim_AB, mode='Zemcov+14', alpha_m=alpha_m, beta_m=beta_m, magstr=magstr, Vega_to_AB=Vega_to_AB, dimx=dimx, dimy=dimy)

	joint_mask = mask_stars_simon*mask_stars_Z14
	
	print('joint mask is type ', type(joint_mask))
	joint_mask = joint_mask.astype(np.int)
	
	if instrument_mask is not None:
		if verbose:
			print('instrument mask being applied as well')
		joint_mask *= instrument_mask.astype(np.int)
		
	return joint_mask, radii_stars_simon, radii_stars_Z14, alpha_m, beta_m

def get_mask_radius_th_rf(m_arr, beta, rc, norm, band=0, Ith=1., fac=0.7, plot=False):

	m_arr = np.array(m_arr)
	Nlarge = 100
	radmap = make_radius_map_yt(np.zeros([2*Nlarge+1, 2*Nlarge+1]), Nlarge, Nlarge)
	radmap *= fac
	Imap_large = norm * (1. + (radmap/rc)**2)**(-3.*beta/2.)
	
	Imap_large /= np.sum(Imap_large)
	
	# get Imap of PSF, multiply by flux, see where 
	
	lam_effs = np.array([1.05, 1.79]) # effective wavelength for bands in micron

	lambdaeff = lam_effs[band]

	sr = ((7./3600.0)*(np.pi/180.0))**2

	I_arr=3631*10**(-m_arr/2.5)*(3./lambdaeff)*1e6/(sr*1e9) # 1e6 is to convert from microns
	r_arr = np.zeros_like(m_arr, dtype=float)
	for i, I in enumerate(I_arr):
		sp = np.where(Imap_large*I > Ith)
		if len(sp[0])>0:
			r_arr[i] = np.max(radmap[sp])
			
	
	if plot:
		f = plt.figure(figsize=(10,10))

		plt.suptitle('I_th = '+str(Ith)+', multiplicative fac = '+str(fac), fontsize=20)
		plt.subplot(2,2,1)
		plt.title('radmap')
		plt.imshow(radmap)
		plt.colorbar()

		plt.subplot(2,2,2)
		plt.title('Imap_large')
		plt.imshow(Imap_large)
		plt.colorbar()

		plt.subplot(2,2,3)
		plt.scatter(m_arr, I_arr)
		plt.xlabel('$m_I$', fontsize=16)
		plt.ylabel('I_arr', fontsize=16)

		plt.subplot(2,2,4)
		plt.scatter(m_arr, r_arr)
		plt.xlabel('$m_I$', fontsize=16)
		plt.ylabel('$r$ [arcsec]', fontsize=16)
		plt.show()
		
		return r_arr, f

	return r_arr


def I_threshold_mask_simple(xs, ys, ms,psf=None, beta=None, rc=None, norm=None, ciber_mock=None, ifield=None, band=0, dimx=1024, dimy=1024, m_min=-np.inf, m_max=20, \
					I_thresh=1., method='radmap', nwide=12, fac=0.7, plot=False):
	
	# get the CIBER PSF for a given ifield if desired

	'''
	Parameters
	----------


	Returns
	-------


	'''
	beta, rc, norm = ciber_mock.get_psf(ifield=ifield, band=band, nx=dimx, ny=dimy, poly_fit=False, nwide=nwide)

	# beta, rc, norm = 1.593e+00, 4.781e+00, 9.477e-03
	
	print('beta, rc, norm:', beta, rc, norm)
	mask = np.ones([dimx, dimy], dtype=int)

	mag_mask = np.where((ms > m_min) & (ms < m_max))[0]
	
	xs, ys, ms = xs[mag_mask], ys[mag_mask], ms[mag_mask]
	
	if plot:
		rs, f = get_mask_radius_th_rf(ms, beta, rc, norm, band=band, Ith=I_thresh, fac=fac, plot=True)
	else:
		rs = get_mask_radius_th_rf(ms, beta, rc, norm, band=band, Ith=I_thresh, fac=fac, plot=False)

		
	for i,(x,y,r) in enumerate(zip(xs, ys, rs)):
		radmap = make_radius_map_yt(np.zeros(shape=(dimx, dimy)), x, y)

		mask[radmap < r/7.] = 0

		if i%1000==0:
			print('i='+str(i)+' of '+str(len(xs)))        
		
	return mask, rs, beta, rc, norm


def I_threshold_image_th(xs, ys, ms, psf=None, beta=None, rc=None, norm=None, ciber_mock=None, ifield=None, band=0, dimx=1024, dimy=1024, m_min=-np.inf, m_max=20, \
					I_thresh=1., nwide=17, nc=35):

	# get the CIBER PSF for a given ifield if desired
	if ifield is not None:
		if ciber_mock is None:
			print('Need ciber_mock class to obtain PSF for ifield '+str(ifield))
			return None
		else:
			ciber_mock.get_psf(poly_fit=True, nwide=17)
			psf = ciber_mock.psf_template

	mask = np.ones([dimx, dimy], dtype=int)    

	mag_mask = np.where((ms > m_min) & (ms < m_max))[0]
	
	xs, ys, ms = xs[mag_mask], ys[mag_mask], ms[mag_mask]
	
	print('len xs is ', len(xs))

	if ciber_mock is not None:
		Is = ciber_mock.mag_2_nu_Inu(ms, band)
	else:
		lam_effs = np.array([1.05, 1.79])*1e-6 # effective wavelength for bands
		sr = ((7./3600.0)*(np.pi/180.0))**2
		Is=3631*10**(-ms/2.5)*(3/lam_effs[band])*1e6/(sr*1e9)

	print("Using bright source map threshold..")
	srcmap = image_model_eval(np.array(xs).astype(np.float32)+2.5, np.array(ys).astype(np.float32)+1.0 ,np.array(Is.value).astype(np.float32),0., (dimx, dimy), nc, ciber_mock.cf, lib=ciber_mock.libmmult.pcat_model_eval)
	
	mask[srcmap > I_thresh] = 0

	return mask, srcmap


def I_threshold_mask(xs, ys, ms,psf=None, beta=None, rc=None, norm=None, ciber_mock=None, ifield=None, band=0, dimx=1024, dimy=1024, m_min=-np.inf, m_max=20, \
					I_thresh=1., method='radmap'):
	
	# get the CIBER PSF for a given ifield if desired
	if ifield is not None:
		if ciber_mock is None:
			print('Need ciber_mock class to obtain PSF for ifield '+str(ifield))
			return None
		else:
			beta, rc, norm = ciber_mock.get_psf(ifield=ifield, band=band, nx=dimx, ny=dimy, poly_fit=True)
			psf = ciber_mock.psf_template

	mask = np.ones([dimx, dimy], dtype=int)

	num = np.zeros([dimx, dimy], dtype=int)

	mag_mask = np.where((ms > m_min) & (ms < m_max))[0]
	
	xs, ys, ms = xs[mag_mask], ys[mag_mask], ms[mag_mask]
	
	if ciber_mock is not None:
		Is = ciber_mock.mag_2_nu_Inu(ms, band)
		
		print(Is)
	else:
		lam_effs = np.array([1.05, 1.79])*1e-6 # effective wavelength for bands
		sr = ((7./3600.0)*(np.pi/180.0))**2
		Is=3631*10**(-ms/2.5)*(3/lam_effs[band])*1e6/(sr*1e9)
	 
	
	rs = rc*((I_thresh/(Is.value*norm))**(-2/(3.*beta)) - 1.)

	if method=='radmap':
		print('Using radmap method..')
		for i,(x,y,r) in enumerate(zip(xs, ys, rs)):

			radmap = make_radius_map(dimx, dimy, x, y, 1.)

			mask[radmap < r] = 0
			num[radmap < r] += 1

			if i%1000==0:
				print('i='+str(i)+' of '+str(len(xs)))
					
	else:
		print("Using bright source map threshold..")
		nc = 25
		srcmap = image_model_eval(np.array(xs).astype(np.float32), np.array(ys).astype(np.float32) ,np.array(Is.value).astype(np.float32),0., (dimx, dimy), nc, ciber_mock.cf, lib=ciber_mock.libmmult.pcat_model_eval)
		mask[srcmap > I_thresh] = 0
		
		
	return mask, num, rs, beta, rc, norm



def simon_r_m(mags, a1=252.8, b1=3.632, c1=8.52, Vega_to_AB=0.):
	'''
	Masking radius formula based on best fit from Simon's analysis.

	Parameters
	----------

	mags : `numpy.ndarray' of shape (Nsrc,)
		Source magnitudes

	a1 : `float', optional
		Normalization coefficient. Default is 252.8.
	b1 : `float', optional
		mean of fit Gaussian. Default is 3.632.
	c1 : `float', optional
		scale radius of Gaussian. Default is 8.52.

	Returns
	-------
	radii : `numpy.ndarray' of shape (Nsrc,)
		Masking radii, given in arcseconds. 


	'''
	radii = a1*np.exp(-((mags-b1)/c1)**2)

	return radii

def perturb_mags_with_types(all_interp_fns_maskerrs, type_weights_bysel, true_mags, mag_bin_edges):

	type_weights_bysel[np.isnan(type_weights_bysel)] = 0.
	mags_with_errors = np.zeros_like(true_mags)

	for m, mag in enumerate(mag_bin_edges[:-1]):
		magmask = (true_mags > mag)*(true_mags <= mag_bin_edges[m+1])

		which_magmask = np.where(magmask)[0]

		true_mags_mask = true_mags[magmask]

		mask_errors = np.zeros_like(true_mags_mask)
		# print('probabilities are ', type_weights_bysel[m,:])

		if np.sum(type_weights_bysel[m,:])==0: # if we don't have training data, assume masking error is zero (i.e., we assume perfect bright sources)
			continue

		which_type = np.random.choice(np.arange(3), len(true_mags_mask), p=type_weights_bysel[m,:])
		# print('which type:', which_type)

		for mask_type in range(3):
			mags_which_type = true_mags_mask[which_type==mask_type]
			mask_rms_type = all_interp_fns_maskerrs[mask_type](mags_which_type)
			mask_errors_type = np.random.normal(0, 1, len(mask_rms_type))*mask_rms_type
			mask_errors[which_type==mask_type] = mask_errors_type

		mags_with_errors[which_magmask] = true_mags_mask + mask_errors

	return mags_with_errors


def generate_full_masks_050823(cbps, ifield_list, sim_idxs, masktail, inst = 1, dat_type='mock', generate_starmask = True, generate_galmask = True,\
							  use_inst_mask = True, mag_lim_Vega=17.5, save_mask = True, \
							  datestr = '062322', datestr_trilegal='062422', convert_AB_to_Vega = True, \
							 dm = 3., a1=160., b1=3.632, c1=8.0, intercept_mag=16.0, \
							 include_ff_mask = False, interp_mask_fn_fpaths=None, max_depth=8, \
							 mag_depth_obs=17.0, cib_file_mode='cib_with_tracer', dx=0., dy=0., interp_maskfn=None, plot=True, \
							m_min_thresh=None, radcap=200., inst_mag_mask=None, \
							twomass_only=False, generate_shotnoise_cat=False, add_spitzer_Lmask=False, mag_key_sdwfs='CH1_mag_auto', mag_lim_sdwfs=18.0, \
							add_wise_Lmask=False, mag_lim_wise=16.0, \
							fudge_fac=None, min_mag_fudge=14, max_mag_fudge=16.0, wcs_headers=None, mask_cat_fpaths=None, mode='Zemcov+14', \
							apply_mask_errs=False, mask_err_vs_mag_fpath=None, mask_transition_mag=14):

	if twomass_only:
		generate_starmask = True
		generate_galmask = False 
		print('twomass only is True, setting starmask to True since this is where I load the 2MASS catalog..')

	if inst_mag_mask is not None:
		mag_lim_AB = mag_lim_Vega + cbps.Vega_to_AB[inst_mag_mask]
	else:
		mag_lim_AB = mag_lim_Vega + cbps.Vega_to_AB[inst]
	print('mag_lim_AB is ', mag_lim_AB)

	catalog_basepath = config.exthdpath+'/ciber_fluctuation_data/catalogs/'

	if interp_mask_fn_fpaths is None:
		param_combo = [a1, b1, c1]

	all_interp_fns_maskerrs = None
	type_weights_bysel = None

	if apply_mask_errs and mask_err_vs_mag_fpath is not None:
		mask_err_vs_mag_file = np.load(mask_err_vs_mag_fpath)

		mags = mask_err_vs_mag_file['mags']
		mid_mags = mask_err_vs_mag_file['mid_mags']
		labels = mask_err_vs_mag_file['labels']
		mag_errs_vs_rms_list = mask_err_vs_mag_file['mag_errs_vs_rms_list']

		type_weights_bysel = mask_err_vs_mag_file['type_weights_bysel']

		all_interp_fns_maskerrs = []
		for lidx, lab in enumerate(labels):
			interp_mask_errs_fn = scipy.interpolate.interp1d(mid_mags, mag_errs_vs_rms_list[lidx])
			all_interp_fns_maskerrs.append(interp_mask_errs_fn)


		plt.figure()
		for lidx, lab in enumerate(labels):
			plt.scatter(mid_mags, mag_errs_vs_rms_list[lidx], label=lab, color='C'+str(lidx))
			plt.plot(mid_mags, all_interp_fns_maskerrs[lidx](mid_mags), color='C'+str(lidx))
		plt.legend()
		plt.show()

	magkey_dict = dict({1:'j_m', 2:'h_m'})
	inst_to_band = dict({1:'J', 2:'H'})

	if inst_mag_mask is not None:
		magkey = magkey_dict[inst_mag_mask]
	else:
		magkey = magkey_dict[inst]

	print('magkey is ', magkey)

	imarray_shape = (len(ifield_list), cbps.dimx, cbps.dimy)

	if dat_type=='mock':
		base_path = config.exthdpath+'ciber_mocks/'
		mask_fpath = base_path+datestr+'/TM'+str(inst)+'/masks/'
		mkk_fpath = base_path+datestr+'/TM'+str(inst)+'/mkk/'
		mkk_ffest_fpath = base_path+datestr+'/TM'+str(inst)+'/mkk_ffest/'

	elif dat_type=='real':
		base_path = 'data/fluctuation_data/'
		mask_fpath = base_path+'TM'+str(inst)+'/masks/'
		mkk_fpath = base_path+'TM'+str(inst)+'/mkk/'
		mkk_ffest_fpath = base_path+'/TM'+str(inst)+'/mkk_ffest/'

	fpaths = [base_path, mask_fpath, mkk_fpath]
	make_fpaths(fpaths)

	mask_fpath += masktail+'/'
	if not os.path.isdir(mask_fpath):
		os.makedirs(mask_fpath)

	mkk_fpath += masktail+'/'
	if not os.path.isdir(mkk_fpath):
		os.makedirs(mkk_fpath)

	mkk_ffest_fpath += masktail+'/'
	if not os.path.isdir(mkk_ffest_fpath):
		os.makedirs(mkk_ffest_fpath)

	for s, sim_idx in enumerate(sim_idxs):
		if s>0:
			plot = False

		joint_masks = np.zeros(imarray_shape)
		for fieldidx, ifield in enumerate(ifield_list):

			field_name = cbps.ciber_field_dict[ifield]
			print(fieldidx, ifield, field_name)

			if interp_mask_fn_fpaths is not None:
				interp_mask_file = np.load(interp_mask_fn_fpaths[fieldidx])
				cent_mags, rad_vs_cent_mags = interp_mask_file['cent_mags'], interp_mask_file['radii']
				min_mag, max_mag = np.min(cent_mags), np.max(cent_mags)


				if fudge_fac is not None:
					rad_vs_cent_mags[(cent_mags > min_mag_fudge)*(cent_mags < max_mag_fudge)] *= fudge_fac

				interp_maskfn = scipy.interpolate.interp1d(cent_mags[rad_vs_cent_mags!= 0], rad_vs_cent_mags[rad_vs_cent_mags != 0])
				a1, b1, c1, dm, alpha_m, beta_m = [None for x in range(6)]

				if s==0:
					plt.figure()
					plt.scatter(cent_mags, rad_vs_cent_mags, marker='.', color='k')
					cent_mags_fine = np.linspace(np.min(cent_mags), np.max(cent_mags), 1000)
					plt.plot(cent_mags_fine, interp_maskfn(cent_mags_fine), color='r')
					plt.yscale('log')
					plt.ylabel('masking radius [arcsec]')
					plt.xlabel('magnitude (Vega)')
					plt.show()
			else:
				min_mag, max_mag = None, None


			# instrument mask
			joint_mask = np.ones(cbps.map_shape)
			if use_inst_mask:

				inst_mask_fpath = config.exthdpath+'/ciber_fluctuation_data/TM'+str(inst)+'/masks/maskInst_080423/field'+str(ifield)+'_TM'+str(inst)+'_maskInst_080423.fits'
				# inst_mask_fpath = config.exthdpath+'/ciber_fluctuation_data/TM'+str(inst)+'/masks/maskInst_102422/field'+str(ifield)+'_TM'+str(inst)+'_maskInst_102422.fits'
				mask_inst = cbps.load_mask(ifield, inst, mask_fpath=inst_mask_fpath, instkey='maskinst', inplace=False)
				joint_mask *= mask_inst


			if generate_starmask: 
				if dat_type=='mock':
					mock_trilegal_path = base_path+datestr_trilegal+'/trilegal/mock_trilegal_simidx'+str(sim_idx)+'_'+datestr_trilegal+'.fits'
					mock_trilegal = fits.open(mock_trilegal_path)
					mock_trilegal_cat = mock_trilegal['tracer_cat_'+str(ifield)].data
				elif dat_type=='real':
					# these are already in Vega magnitudes
					print('using inst = ', inst, 'for positions')

					if twomass_only:
						print('loading catalog down to J = 17.5')
						twomass_cat_df = pd.read_csv(catalog_basepath+'2MASS/filt/2MASS_filt_rdflag_wxy_'+field_name+'_Jlt17.5.csv')
					else:
						twomass_cat_df = pd.read_csv(catalog_basepath+'2MASS/filt/2MASS_filt_rdflag_wxy_'+field_name+'_Jlt16.0.csv')
					
					twomass_x = twomass_cat_df['x'+str(inst)]
					twomass_y = twomass_cat_df['y'+str(inst)]
					print('using positions x/y'+str(inst))
					twomass_mag = twomass_cat_df[magkey] # magkey
					# twomass_mag = twomass_lt_16[magkey_dict[inst]]
					twomass_cat = np.array([twomass_x, twomass_y, twomass_mag]).transpose()

				if convert_AB_to_Vega and dat_type=='mock':
					print('Converting AB magnitudes to Vega, subtracting by ', cbps.Vega_to_AB[inst])
					mock_trilegal_cat[magkey_dict[inst]] -= cbps.Vega_to_AB[inst]

				starcatcols = [magkey, 'x'+str(inst), 'y'+str(inst)]
				if dat_type=='mock':
					print('mock trilegal cat has shape', mock_trilegal_cat.shape)

					star_cat_df = pd.DataFrame({magkey:mock_trilegal_cat[magkey].byteswap().newbyteorder(), 'y'+str(inst):mock_trilegal_cat['x'].byteswap().newbyteorder()+dx, 'x'+str(inst):mock_trilegal_cat['y'].byteswap().newbyteorder()+dy}, \
						columns=starcatcols)

					if apply_mask_errs and all_interp_fns_maskerrs is not None:

						orig_mags = np.array(star_cat_df[magkey])

						# def perturb_mags_with_types(all_interp_fns_maskerrs, type_weights_bysel, true_mags, mag_bin_edges):

						mags_with_errors = perturb_mags_with_types(all_interp_fns_maskerrs, type_weights_bysel, orig_mags, mags)
						
						mag_errs = mags_with_errors-orig_mags

						mags_with_errors[np.abs(mag_errs) > 2] = orig_mags[np.abs(mag_errs) > 2]

						if s < 5:
							plt.figure()
							plt.hist(mag_errs, bins=np.linspace(-2, 2, 30))
							plt.xlabel('mag errors stars')
							plt.yscale('log')
							plt.show()
						star_cat_df[magkey] = mags_with_errors

						# mask_errs_star = interp_mask_errs_fn(np.array(star_cat_df[magkey]))
						# print('mask errors in mag are ', mask_errs_star)
						# star_cat_df[magkey] += np.random.normal(0, 1, len(mask_errs_star))*mask_errs_star



				elif dat_type=='real':

					if wcs_headers is not None:

						# src_coord = SkyCoord(ra=twomass_cat['ra']*u.degree, dec=twomass_dec['dec']*u.degree, frame='icrs')

						twomass_x_wcs, twomass_y_wcs = wcs_headers[fieldidx].all_world2pix(twomass_cat_df['ra']*u.degree,twomass_cat_df['dec']*u.degree,0)
						# print('twomass x wcs:', twomass_x_wcs)
						print(twomass_x_wcs.shape)
						star_cat_df = pd.DataFrame({magkey:twomass_cat[:,2], 'x'+str(inst):twomass_x_wcs,'y'+str(inst):twomass_y_wcs}, columns=starcatcols)


					else:
						star_cat_df = pd.DataFrame({magkey:twomass_cat[:,2], 'x'+str(inst):twomass_cat[:,0],'y'+str(inst):twomass_cat[:,1]}, columns=starcatcols)


				starmask, radii_stars = mask_from_cat(cat_df = star_cat_df, mag_lim_min=0, inst=inst,\
															mag_lim=mag_lim_Vega, interp_maskfn=interp_maskfn,\
													  magstr=magkey, Vega_to_AB=0., dimx=cbps.dimx, dimy=cbps.dimy, plot=False, \
													 interp_max_mag = max_mag, interp_min_mag=min_mag, m_min_thresh=m_min_thresh, radcap=radcap, \
													 mode=mode, mask_transition_mag=mask_transition_mag)

				# if interp_mask_fn_fpaths is not None:
				# 	# simulated sources changed to Vega magnitudes with convert_AB_to_Vega, masking function magnitudes in Vega units
				# 	# magnitude limit should be in Vega since we already converted magnitudes to Vega
				# 	starmask, radii_stars = mask_from_cat(cat_df = star_cat_df, mag_lim_min=0, inst=inst,\
				# 												mag_lim=mag_lim_Vega, interp_maskfn=interp_maskfn,\
				# 										  magstr=magkey, Vega_to_AB=0., dimx=cbps.dimx, dimy=cbps.dimy, plot=False, \
				# 										 interp_max_mag = max_mag, interp_min_mag=min_mag, m_min_thresh=m_min_thresh, radcap=radcap, \
				# 										 mode=mode)

				# else:
				# 	starmask, radii_stars_simon, radii_stars_Z14, alpha_m, beta_m = get_masks(star_cat_df, param_combo, intercept_mag, mag_lim_Vega, dm=dm, magstr=magkey, inst=inst, dimx=cbps.dimx, dimy=cbps.dimy, verbose=True)



				joint_mask *= starmask


			if generate_galmask:

				if dat_type=='mock':
					midxdict = dict({'x':0, 'y':1, 'redshift':2, 'm_app':3, 'M_abs':4, 'Mh':5, 'Rvir':6})
					mock_gal = fits.open(base_path+datestr+'/TM'+str(inst)+'/cib_realiz/'+cib_file_mode+'_5field_set'+str(sim_idx)+'_'+datestr+'_TM'+str(inst)+'.fits')
					mock_gal_cat = mock_gal['tracer_cat_'+str(ifield)].data
				else:

					if mask_cat_fpaths is not None:
						mask_cat_unWISE_PS = pd.read_csv(mask_cat_fpaths[fieldidx])
					else:
						mask_cat_unWISE_PS = pd.read_csv(catalog_basepath+'mask_predict/mask_predict_unWISE_PS_fullmerge_'+field_name+'_ukdebias.csv')
					
						# mask_cat_unWISE_PS = pd.read_csv(catalog_basepath+'mask_predict/mask_predict_unWISE_PS_fullmerge_'+field_name+'.csv')
					if wcs_headers is not None:

						faint_cat_x, faint_cat_y = wcs_headers[fieldidx].all_world2pix(mask_cat_unWISE_PS['ra'],mask_cat_unWISE_PS['dec'],0)

					else:
						faint_cat_x = mask_cat_unWISE_PS['x'+str(inst)]
						faint_cat_y = mask_cat_unWISE_PS['y'+str(inst)]
					# faint_cat_mag = mask_cat_unWISE_PS[magkey+'_Vega_predict'] # magkey
					if inst_mag_mask is not None:
						print('faint cat mags from ', inst_to_band[inst_mag_mask]+'_Vega_predict')
						faint_cat_mag = mask_cat_unWISE_PS[inst_to_band[inst_mag_mask]+'_Vega_predict']
					else:
						faint_cat_mag = mask_cat_unWISE_PS[inst_to_band[inst]+'_Vega_predict']

				if convert_AB_to_Vega and dat_type=='mock':
					print('converting galaxy magnitudes to Vega mag with ', cbps.Vega_to_AB[inst])
					mock_gal_cat['m_app'] -= cbps.Vega_to_AB[inst]

				galcatcols = ['m_app', 'x'+str(inst), 'y'+str(inst)]
				if dat_type=='mock':
					gal_cat = {'m_app':mock_gal_cat['m_app'].byteswap().newbyteorder(), 'y'+str(inst):mock_gal_cat['x'].byteswap().newbyteorder()+dx, 'x'+str(inst):mock_gal_cat['y'].byteswap().newbyteorder()+dy}
					gal_cat_df = pd.DataFrame(gal_cat, columns = galcatcols) # check magnitude system of Helgason model
					# print('gal_cat_df is')
					# print(gal_cat_df)

					if apply_mask_errs and interp_mask_errs_fn is not None:

						gal_orig_mags = np.array(gal_cat_df['m_app'])

						gal_mags_with_errors = perturb_mags_with_types(all_interp_fns_maskerrs, type_weights_bysel, gal_orig_mags, mags)
						gal_mag_errs = gal_mags_with_errors-gal_orig_mags


						gal_mags_with_errors[np.abs(gal_mag_errs) > 2] = gal_orig_mags[np.abs(gal_mag_errs) > 2]
						
						gal_mag_errs = gal_mags_with_errors-gal_orig_mags

						if s < 5:
							plt.figure()
							plt.hist(gal_mag_errs, bins=np.linspace(-2, 2, 30))
							plt.xlabel('mag errors galaxies')
							plt.yscale('log')

							plt.show()
						gal_cat_df[magkey] = gal_mags_with_errors

						# mask_errs_gals = interp_mask_errs_fn(np.array(gal_cat_df['m_app']))
						# print('mask errors in mag for gals are ', mask_errs_gals)
						# gal_cat_df[magkey] += np.random.normal(0, 1, len(mask_errs_gals))*mask_errs_gals



				elif dat_type=='real':
					gal_cat_df = pd.DataFrame({'m_app':faint_cat_mag, 'x'+str(inst):faint_cat_x, 'y'+str(inst):faint_cat_y}, columns = galcatcols)


				galmask, radii_gals = mask_from_cat(cat_df = gal_cat_df, mag_lim_min=0, inst=inst,\
										mag_lim=mag_lim_Vega, interp_maskfn=interp_maskfn, magstr='m_app', Vega_to_AB=0., dimx=cbps.dimx, dimy=cbps.dimy, plot=False, \
												   interp_max_mag = max_mag, interp_min_mag=min_mag, mode=mode, mask_transition_mag=mask_transition_mag)
				
				# if interp_mask_fn_fpaths is not None:

				# 	print('mag lim Vega for galaxies is ', mag_lim_Vega)
				# 	galmask, radii_gals = mask_from_cat(cat_df = gal_cat_df, mag_lim_min=0, inst=inst,\
				# 							mag_lim=mag_lim_Vega, interp_maskfn=interp_maskfn, magstr='m_app', Vega_to_AB=0., dimx=cbps.dimx, dimy=cbps.dimy, plot=False, \
				# 									   interp_max_mag = max_mag, interp_min_mag=min_mag)

				# else:
				# 	galmask, radii_stars_simon, radii_stars_Z14, alpha_m, beta_m = get_masks(gal_cat_df, param_combo, intercept_mag, mag_lim_Vega, dm=dm, magstr='m_app', inst=inst, dimx=cbps.dimx, dimy=cbps.dimy, verbose=True)

				if len(radii_gals) > 0:
					print('len radii gals is ', len(radii_gals))
					joint_mask *= galmask
				else:
					print('radii gals is ', radii_gals, 'and galmask is ', galmask)


			if add_spitzer_Lmask and ifield in [6, 7]:
				print('Combining with spitzer mask down to W1 = ', str(mag_lim_sdwfs))
				sdwfs_cat = pd.read_csv('data/Spitzer/sdwfs_catalogs/sdwfs_wxy_CIBER_ifield'+str(ifield)+'.csv')
				sdwfs_mag = np.array(sdwfs_cat[mag_key_sdwfs])
				sdwfs_x, sdwfs_y = np.array(sdwfs_cat['x'+str(inst)]), np.array(sdwfs_cat['y'+str(inst)])

				sdwfs_cat = np.array([sdwfs_x, sdwfs_y, sdwfs_mag]).transpose()

				sdwfs_xymask = (sdwfs_x > 0)*(sdwfs_x < cbps.dimx)*(sdwfs_y > 0)*(sdwfs_y < cbps.dimy)
				sdwfs_cat = sdwfs_cat[np.where(sdwfs_xymask)[0], :]


				sdwfs_catcols = ['x'+str(inst), 'y'+str(inst), 'm_app']
				sdwfs_cat_df = pd.DataFrame(sdwfs_cat, columns = sdwfs_catcols) # check magnitude system of Helgason model

				print('sdwfs cat df:', sdwfs_cat_df)

				sdwfs_radii = 7.*np.ones_like(sdwfs_mag)

				sdwfs_mask, radii_gals_sdwfs = mask_from_cat(cat_df = sdwfs_cat_df, mag_lim_min=15, inst=inst,\
										mag_lim=mag_lim_sdwfs, magstr='m_app', Vega_to_AB=0., dimx=cbps.dimx, dimy=cbps.dimy, plot=False, \
												   radii=sdwfs_radii) 
				if plot:
					plot_map(sdwfs_mask, title='SDWFS mask L < '+str(mag_lim_wise))

				if len(radii_gals_sdwfs) > 0:
					print('len radii gals wise is ', len(radii_gals_sdwfs))
					joint_mask *= sdwfs_mask


			if add_wise_Lmask:
				print('Combining with WISE mask down to W1 = ', str(mag_lim_wise))
				wise_cat = pd.read_csv(catalog_basepath+'unWISE/filt/unWISE_filt_wxy_magcut_Vega_'+field_name+'.csv')
				wise_mag = np.array(wise_cat['mag_W1'])
				wise_x, wise_y = np.array(wise_cat['x'+str(inst)]), np.array(wise_cat['y'+str(inst)])
				
				wise_cat = np.array([wise_x, wise_y, wise_mag]).transpose()
				wisecatcols = ['x'+str(inst), 'y'+str(inst), 'm_app']
				wise_cat_df = pd.DataFrame(wise_cat, columns = wisecatcols)

				wisemask, radii_gals_wise = mask_from_cat(cat_df = wise_cat_df, mag_lim_min=0, inst=inst,\
										mag_lim=mag_lim_wise, interp_maskfn=interp_maskfn, magstr='m_app', Vega_to_AB=0., dimx=cbps.dimx, dimy=cbps.dimy, plot=False, \
												   interp_max_mag = max_mag) 
				if plot:
					plot_map(wisemask, title='WISE mask L < '+str(mag_lim_wise))

				if len(radii_gals_wise) > 0:
					print('len radii gals wise is ', len(radii_gals_wise))
					joint_mask *= wisemask
					
			print(float(np.sum(joint_mask))/float(1024**2))

			joint_masks[fieldidx] = joint_mask.copy()

			if plot:
				plot_map(joint_masks[fieldidx], title='joint_masks '+str(fieldidx))

			   

		mask_save_fpaths = []
		if save_mask:

			for fieldidx, ifield in enumerate(ifield_list):
				
				if plot and fieldidx==0:
					plot_map(joint_masks[fieldidx], title='joint mask, simidx = '+str(sim_idx))
				hdul = write_mask_file(np.array(joint_masks[fieldidx]), ifield=ifield, inst=inst, sim_idx=sim_idx, generate_galmask=generate_galmask, \
									  generate_starmask=generate_starmask, use_inst_mask=use_inst_mask, dat_type=dat_type, mag_lim_AB=mag_lim_AB, \
									  a1=a1, b1=b1, c1=c1, dm=dm, alpha_m=alpha_m, beta_m=beta_m)

				if dat_type=='mock':
					mask_save_fpath = mask_fpath+'joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(sim_idx)+'_'+masktail+'.fits'
				elif dat_type=='real':
					mask_save_fpath = mask_fpath+'joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+masktail+'.fits'

				mask_save_fpaths.append(mask_save_fpath)

				print('Saving mask to ', mask_save_fpath)
				hdul.writeto(mask_save_fpath, overwrite=True)

	return joint_masks, mask_save_fpaths


