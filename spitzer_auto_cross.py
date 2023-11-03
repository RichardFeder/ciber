import numpy as np
from astropy.io import fits
from plotting_fns import plot_map
import config
from ciber_powerspec_pipeline import *
from noise_model import *
from cross_spectrum_analysis import *
from cross_spectrum import *
from mkk_parallel import *
from mkk_diagnostics import *
from flat_field_est import *
from plotting_fns import *
from powerspec_utils import *
from astropy.coordinates import SkyCoord
import healpy as hp
import scipy.io




def load_irac_bl(irac_ch, base_path='data/Spitzer/irac_bl/'):

	irac_bl_fpath = base_path+'irac_ch'+str(irac_ch)+'_071323.csv'
	print('loading from ', irac_bl_fpath)
	irac_bl = pd.read_csv(irac_bl_fpath, header=None)
	irac_lb, irac_blval = np.array(irac_bl)[:,0], np.array(irac_bl)[:,1]
	
	irac_lb = np.array([50., 100.]+list(irac_lb))
	irac_blval = np.array([1., 1.]+list(irac_blval))

	irac_lb *= 2
	return irac_lb, irac_blval


def load_spitzer_bootes_maps(inst, irac_ch, bootes_ifields, spitzer_string):
	
	spitzer_regrid_maps, spitzer_regrid_masks = [], []
	irac_lam_dict = dict({1:3.6, 2:4.5})
	
	all_diff1, all_diff2, all_spitz_epochs = [], [], []
	for idx, ifield in enumerate(bootes_ifields):
		spitzer_fpath = 'data/Spitzer/spitzer_regrid/sdwfs_regrid_TM'+str(inst)+'_ifield'+str(ifield)+'_'+spitzer_string+'.fits'
		print('spitzer_fpath', spitzer_fpath)
		spitz = fits.open(spitzer_fpath)
		spitzer_regrid_to_ciber = spitz['sdwfs_map_regrid_ifield'+str(ifield)].data
		spitzer_mask_regrid_to_ciber = spitz['sdwfs_mask_regrid_ifield'+str(ifield)].data

		diff1 = spitz['sdwfs_diff1_ifield'+str(ifield)].data
		diff2 = spitz['sdwfs_diff2_ifield'+str(ifield)].data
		
		spitz_epochs = spitz['sdwfs_epochs_ifield'+str(ifield)].data

		lam = irac_lam_dict[irac_ch]
		fac = convert_MJysr_to_nWm2sr(lam)
		print("fac for lambda = "+str(lam)+" is "+str(fac))

		spitzer_regrid_to_ciber /= fac
		diff1 /= fac
		diff2 /= fac
		spitz_epochs /= fac
		
		all_diff1.append(diff1)
		all_diff2.append(diff2)
		all_spitz_epochs.append(spitz_epochs)
		spitzer_mask_regrid_to_ciber = spitz['sdwfs_mask_regrid_ifield'+str(ifield)].data
		
		spitzer_regrid_maps.append(spitzer_regrid_to_ciber)
		spitzer_regrid_masks.append(spitzer_mask_regrid_to_ciber)
		
	return spitzer_regrid_maps, spitzer_regrid_masks, all_diff1, all_diff2, all_spitz_epochs




def compute_nl2d_spitzer_crossepoch_diffs(diff1, diff2, combined_mask, plot=False):
	''' 
	Compute 2D noise power spectrum of Spitzer cross epoch differences. This involves taking the average of 
	the two differences and then dividing by two to get the per epoch noise level.
	'''
	
	# testing new FFT combination
	l2d, nl2d_diff = get_power_spectrum_2d_epochav(diff1, diff2, pixsize=7., verbose=False)

	# previous
	# l2d, nl2d_diff_cross = get_power_spectrum_2d(diff1, map_b=diff2, pixsize=7.)
	# nl2d_diff = 0.5*nl2d_diff_cross

	# l2d, nl2d_diff1 = get_power_spectrum_2d(diff1, pixsize=7.)
	# l2d, nl2d_diff2 = get_power_spectrum_2d(diff2, pixsize=7.)
	
	# nl2d_diff = 0.25*(nl2d_diff1 + nl2d_diff2)
	
	if plot:
		plot_map(nl2d_diff, title='nl2d diff')
		plot_map(np.log10(nl2d_diff), title='log10 nl2d diff')

	
	return l2d, nl2d_diff



def regrid_spitzer_to_ciber(ifield, inst, spitzer_map, spitzer_RA, spitzer_DEC, spitzer_mask=None, ciber_ras=None, ciber_decs=None, 
							 use_ciber_wcs=True, plot=True, order=0, npix_persamp = 64, ciber_pixsize=7.):
	
	''' 
	
	Regrid Spitzer maps to CIBER
	
	For each CIBER pixel, query the RA/DEC, grab Spitzer pixels and sample them
	
	Parameters
	----------
	
	Returns
	-------
	
	'''

	ciber_field_dict = dict({4:'elat10', 5:'elat30',6:'BootesB', 7:'BootesA', 8:'SWIRE', 'train':'UDS'})

	fieldname = ciber_field_dict[ifield]
	hdrdir = 'data/astroutputs/inst'+str(inst)+'/'

	if use_ciber_wcs:
		wcs_hdrs = []
		for iquad,quad in enumerate(['A','B','C','D']):
			print('quad '+quad)
			hdulist = fits.open(hdrdir + fieldname + '_' + quad + '_astr.fits')
			wcs_hdr=wcs.WCS(hdulist[('primary',1)].header, hdulist)
			wcs_hdrs.append(wcs_hdr)
	
	dimx, dimy = 1024, 1024
	ncoarse_samp = dimx//npix_persamp    
	spitzer_regrid = np.zeros((dimx, dimy))

	if spitzer_mask is not None:
		spitzer_mask_regrid = np.zeros((dimx, dimy))
		
	xs_full, ys_full = np.meshgrid(np.arange(dimx), np.arange(dimy))
	ciber_ra_av_full, ciber_dec_av_full = wcs_hdrs[0].all_pix2world(xs_full, ys_full, 0)
	
	plot_map(ciber_ra_av_full, title='ra')
	plot_map(ciber_dec_av_full, title='dec')
	print('ciber ra av full shape ', ciber_ra_av_full.shape)
	
	for ix in range(ncoarse_samp):
		print('ix = ', ix)
			
		for iy in range(ncoarse_samp):
				
			x0, x1 = ix*npix_persamp, (ix+1)*npix_persamp - 1
			y0, y1 = iy*npix_persamp, (iy+1)*npix_persamp - 1
			
			xs, ys = np.meshgrid(np.arange(x0, x1+1), np.arange(y0, y1+1))
			ciber_ra_av_all, ciber_dec_av_all = wcs_hdrs[0].all_pix2world(ys, xs, 0)

			ciber_mean_ra, ciber_mean_dec = np.mean(ciber_ra_av_all), np.mean(ciber_dec_av_all) 

			dra_mean = spitzer_RA-ciber_mean_ra
			ddec_mean = spitzer_DEC-ciber_mean_dec
			dpos = np.sqrt(dra_mean**2+ddec_mean**2)
			grabinbound = (3600*dpos < 2*ciber_pixsize*npix_persamp)
			grabmask = np.where(grabinbound)
			
			grab_ra = spitzer_RA[grabmask[0], grabmask[1]]
			grab_dec = spitzer_DEC[grabmask[0], grabmask[1]]
			grab_map = spitzer_map[grabmask[0], grabmask[1]]
			
			if spitzer_mask is not None:
				grab_mask = spitzer_mask[grabmask[0], grabmask[1]]
			
			for ix2 in range(npix_persamp):
				for iy2 in range(npix_persamp):
										
					ciber_indiv_ra, ciber_indiv_dec = ciber_ra_av_full[x0+ix2, y0+iy2], ciber_dec_av_full[x0+ix2, y0+iy2]
					
					dra_indiv = grab_ra-ciber_indiv_ra
					ddec_indiv = grab_dec-ciber_indiv_dec
					dpos_indiv = np.sqrt(dra_indiv**2+ddec_indiv**2)
					if 3600*np.min(dpos_indiv) < 10.:
						which_spitz_indiv = np.where((dpos_indiv==np.min(dpos_indiv)))[0][0]
						spitzer_regrid[x0+ix2, y0+iy2] = grab_map[which_spitz_indiv]
						spitzer_mask_regrid[x0+ix2, y0+iy2] = grab_mask[which_spitz_indiv]

	plot_map(spitzer_regrid, title='spitzer regrid')
	plot_map(spitzer_regrid*spitzer_mask_regrid, title='spitzer regrid * mask')

	if spitzer_mask is not None:
		return spitzer_regrid, spitzer_mask_regrid
	
	return spitzer_regrid


def ciber_spitzer_crosscorr_all_combinations(compute_mkk=False, compute_nl_ciber_unc=False, compute_nl_spitzer_unc=False):
	bandstr_dict = dict({1:'J', 2:'H'})

	all_save_cl_fpath = []
	
	for irac_ch in [1, 2]:
		for inst in [1, 2]:
			if inst==1:
				masking_maglim = 17.5
			elif inst==2:
				masking_maglim = 17.0
				
			mask_tail = 'maglim_'+bandstr_dict[inst]+'_Vega_'+str(masking_maglim)+'_062623'

			spitzer_string = '052623_gradsub_byepoch_CH'+str(irac_ch)

			
			save_cl_fpath = ciber_spitzer_crosscorr_full(irac_ch, inst, spitzer_string, mask_tail, compute_mkk=compute_mkk,\
														 compute_nl_spitzer_unc=compute_nl_spitzer_unc, compute_nl_ciber_unc=compute_nl_ciber_unc, plot=True, \
														mask_string_mkk=mask_tail+'_IRAC_CH'+str(irac_ch), photon_noise=True, include_ff_errors=True, add_str_ciber=None, add_str_spitzer=None)

			all_save_cl_fpath.append(save_cl_fpath)
			
	return all_save_cl_fpath 

	
def ciber_spitzer_crosscorr_full(irac_ch, inst, spitzer_string, mask_tail, compute_mkk=True, photon_noise=True, compute_nl_spitzer_unc=True, compute_nl_ciber_unc=True, bootes_idxs = [2,3],\
								 plot=False, base_path='/Users/richardfeder/Downloads/ciber_downloads/', \
								n_mkk_sims=200, n_split=10, n_sims_noise=500, n_split_noise=20, mask_string_mkk='bmask17',\
								 observed_run_name = 'observed_032823_e30mask_clusmask', include_ff_errors=True, \
								 save=True, ifield_list_full = [4, 5, 6, 7, 8], add_str=None, add_str_ciber=None, add_str_spitzer=None, \
								 apply_FW=False):


	cbps_nm = CIBER_NoiseModel()
	base_nv_path = config.exthdpath+'noise_model_validation_data/'

	irac_lb, irac_blval = load_irac_bl(irac_ch=irac_ch)
	masks, ciber_maps = [np.zeros((len(ifield_list_full), cbps_nm.cbps.dimx, cbps_nm.cbps.dimy)) for x in range(2)]
	lb = cbps_nm.cbps.Mkk_obj.midbin_ell
	interp_maskfn = scipy.interpolate.interp1d(np.log10(irac_lb), np.log10(irac_blval))

	full_bls = []
	cibermatch_bl = 10**interp_maskfn(np.log10(lb))
	bls_fpath = config.exthdpath+'ciber_fluctuation_data/TM'+str(inst)+'/beam_correction/bl_est_postage_stamps_TM'+str(inst)+'_081121.npz'
	B_ells = np.load(bls_fpath)['B_ells_post']

	for fieldidx, ifield in enumerate(ifield_list_full):
		
		full_bls.append(np.sqrt(cibermatch_bl*B_ells[fieldidx,:]))

#         mask_fpath = config.exthdpath+'ciber_fluctuation_data/TM'+str(inst)+'/masks/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'
		mask_fpath = 'data/fluctuation_data/TM'+str(inst)+'/masks/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'
		masks[fieldidx] = fits.open(mask_fpath)[1].data.astype(np.int)

		flight_map = cbps_nm.cbps.load_flight_image(ifield, inst, inplace=False, verbose=True)
		ciber_maps[fieldidx,:,:] = flight_map*cbps_nm.cbps.cal_facs[inst]
	
	
	processed_ciber_maps, masks = process_ciber_maps_by_quadrant(cbps_nm.cbps, ifield_list_full, inst, ciber_maps, masks, clip_sigma=5,\
																				 nitermax=10)

	if plot:
		for c, cb in enumerate(processed_ciber_maps):
			plot_map(cb*masks[c])
		
	bootes_obs = [processed_ciber_maps[idx] for idx in bootes_idxs]
	bootes_masks = [masks[idx] for idx in bootes_idxs]
	bootes_ifields = [ifield_list_full[idx] for idx in bootes_idxs]
	bootes_bls = [full_bls[idx] for idx in bootes_idxs]
	
	all_cl_spitzer, all_clerr_spitzer, all_cl_cross, all_clerr_cross,\
		all_clerr_cross_tot, all_nl1d_diff, all_nl1d_err_diff, all_fft2_spitzer, \
			all_clerr_cibernoise_spitzer_cross, all_clerr_spitzernoise_ciber_cross= [[] for x in range(10)]
	

	spitzer_regrid_maps, spitzer_regrid_masks, all_diff1, all_diff2, all_spitz_by_epoch = load_spitzer_bootes_maps(inst, irac_ch, bootes_ifields, spitzer_string)
	combined_masks = [bootes_masks[idx]*spitzer_regrid_mask for idx, spitzer_regrid_mask in enumerate(spitzer_regrid_masks)]
	spitzer_regrid_maps_meansub = np.zeros_like(spitzer_regrid_maps)
	
	
	for idx, ifield in enumerate(bootes_ifields):
		
		if compute_mkk:
			Mkk = cbps_nm.cbps.Mkk_obj.get_mkk_sim(combined_masks[idx], n_mkk_sims, n_split=n_split, store_Mkks=False)
			inv_Mkk = compute_inverse_mkk(Mkk)
			plot_mkk_matrix(inv_Mkk, inverse=True, symlogscale=True)
			hdul = write_Mkk_fits(Mkk, inv_Mkk, ifield, inst, \
								 use_inst_mask=True, dat_type='ciber_spitzer_cross')

			mkkpath = 'data/Spitzer/spitzer_ciber_mkk/mkk_maskonly_estimate_ifield'+str(ifield)+'_inst'+str(inst)+'_spitzunion_'+mask_string_mkk+'.fits'
			hdul.writeto(mkkpath, overwrite=True)
		else:
			mkkpath = 'data/Spitzer/spitzer_ciber_mkk/mkk_maskonly_estimate_ifield'+str(ifield)+'_inst'+str(inst)+'_spitzunion_'+mask_string_mkk+'.fits'
			inv_Mkk = fits.open(mkkpath)['inv_Mkk_'+str(ifield)].data
				
		meansubspitz = spitzer_regrid_maps[idx].copy()
		meansubspitz *= combined_masks[idx]

		diff1_indiv = combined_masks[idx]*all_diff1[idx]
		diff2_indiv = combined_masks[idx]*all_diff2[idx]        

		meansubspitz[meansubspitz != 0] -= np.mean(meansubspitz[meansubspitz != 0])
		diff1_indiv[diff1_indiv != 0] -= np.mean(diff1_indiv[diff1_indiv != 0])
		diff2_indiv[diff2_indiv != 0] -= np.mean(diff2_indiv[diff2_indiv != 0])

		plot_map(diff1_indiv, title='Epoch 1 - Epoch 2, gradient filtered')
		plot_map(diff2_indiv, title='Epoch 3 - Epoch 4, gradient filtered')
		
		spitzer_regrid_maps_meansub[idx] = meansubspitz
		l2d, nl2d_diff = compute_nl2d_spitzer_crossepoch_diffs(diff1_indiv, diff2_indiv, combined_masks[idx], plot=plot)
		nl2d_diff /= 8. # noise model is for difference of individual epochs, so divide by 8 to get 4-epoch averaged
		# all_fft2_spitzer.append(nl2d_diff/4.)
		
		combined_mask_fraction = float(np.sum(combined_masks[idx]))/float(cbps_nm.cbps.dimx**2)
		
		# correct per azimuthally averaged bin
		nl2d_diff_orig = nl2d_diff.copy()

		print('inv mkk diagonal is ', [inv_Mkk.transpose()[x, x] for x in range(inv_Mkk.shape[0])])

		for bandidx in range(inv_Mkk.shape[0]):
			lmin, lmax = cbps_nm.cbps.Mkk_obj.binl[bandidx], cbps_nm.cbps.Mkk_obj.binl[bandidx+1]
			sp = np.where((l2d>=lmin) & (l2d<lmax))
			nl2d_diff[sp] *= inv_Mkk.transpose()[bandidx, bandidx]

		plot_map(nl2d_diff/nl2d_diff_orig, title='mkk corrected / original')

		# nl2d_diff /= combined_mask_fraction
		all_fft2_spitzer.append(nl2d_diff)

		
		lb, nl1d_diff, nl1d_err_diff = azim_average_cl2d(nl2d_diff, l2d, nbins=25, lbinedges=cbps_nm.cbps.Mkk_obj.binl, lbins=cbps_nm.cbps.Mkk_obj.midbin_ell, logbin=True, verbose=False)
		all_nl1d_diff.append(nl1d_diff)
		all_nl1d_err_diff.append(nl1d_err_diff)

	
	if add_str is not None:
		add_str = '_'+add_str
	else:
		add_str = ''
		
	if add_str_spitzer is None:
		add_str_spitzer = add_str
	else:
		add_str_spitzer = '_'+add_str_spitzer
		
	if add_str_ciber is None:
		add_str_ciber = add_str
	else:
		add_str_ciber = '_'+add_str_ciber
	
	if compute_nl_spitzer_unc:
		
		print('Estimating spitzer noise x ciber..')
		all_nl_spitzer_fpath, lb, \
				all_nl1ds_spitzer_both, all_nl1ds_cross_ciber_both = estimate_spitzer_noise_cross_ciber(cbps_nm, irac_ch, inst, bootes_obs, combined_masks, spitzer_string, nsims=n_sims_noise, n_split=n_split_noise, \
													  base_path=base_path, fft2_spitzer=np.array(all_fft2_spitzer), compute_spitzer_FW=True, save=True, add_str=add_str_spitzer)
		
	if compute_nl_ciber_unc:
		print('Estimating ciber noise x spitzer..')
		all_nl_ciber_fpath, lb, all_nl1ds_cross_spitzer_both = estimate_ciber_noise_cross_spitzer(cbps_nm, irac_ch, inst, spitzer_regrid_maps_meansub, combined_masks, spitzer_string,\
																										nsims=n_sims_noise, n_split=n_split_noise, include_ff_errors=include_ff_errors, observed_run_name=observed_run_name, \
																									  base_path=base_path, save=True, photon_noise=photon_noise, add_str=add_str_ciber)
		print('all nl ciber fpath:', all_nl_ciber_fpath)

		
				
	for idx, ifield in enumerate(bootes_ifields):
		
		if compute_nl_spitzer_unc:
			nmfile_spitzernoise_ciber = np.load(all_nl_spitzer_fpath[idx])
		else:
			nl_spitzer_fpath = base_path+'ciber_meeting_030223/nl1ds_TM'+str(inst)+'_ifield'+str(ifield)+'_irac_ch'+str(irac_ch)+'_spitzernoise_ciber_cross_'+add_str_spitzer+'.npz'
			print('Loading Spitzer noise cross model from ', nl_spitzer_fpath)
			nmfile_spitzernoise_ciber = np.load(nl_spitzer_fpath)

		if compute_nl_ciber_unc:
			nmfile_cibernoise_spitzer = np.load(all_nl_ciber_fpath[idx])
		else:
			nl_ciber_fpath = base_path+'ciber_meeting_030223/nl1ds_TM'+str(inst)+'_ifield'+str(ifield)+'_irac_ch'+str(irac_ch)+'_cibernoise_spitzer_cross_'+add_str_ciber+'.npz'
			print('Loading CIBER noise cross model from ', nl_ciber_fpath)
			nmfile_cibernoise_spitzer = np.load(nl_ciber_fpath)

		lb = nmfile_cibernoise_spitzer['lb']

		all_nl1ds_cross_ciber = nmfile_spitzernoise_ciber['all_nl1ds_cross_ciber']
		
				
		av_nl1ds_spitzernoise_ciber_cross = np.mean(all_nl1ds_cross_ciber, axis=0)
		std_nl1ds_spitzernoise_ciber_cross = np.std(all_nl1ds_cross_ciber, axis=0)
		
		all_nl1ds_cross_spitzer = nmfile_cibernoise_spitzer['all_nl1ds_cross_spitzer']
		std_nl1ds_cibernoise_spitzer_cross = np.std(all_nl1ds_cross_spitzer, axis=0)

		if apply_FW:
			var_nl2d_spitzernoise_cross_ciber = nmfile_spitzernoise_ciber['var_nl2d']
			var_nl2d_cibernoise_cross_spitzer = nmfile_cibernoise_spitzer['var_nl2d']
			total_var_nl2d = var_nl2d_cibernoise_cross_spitzer + var_nl2d_spitzernoise_cross_ciber
			fourier_weights_cross = 1./total_var_nl2d

			plot_map(var_nl2d_spitzernoise_cross_ciber, title='var_nl2d_spitzernoise_cross_ciber', lopct=5, hipct=95)
			plot_map(var_nl2d_cibernoise_cross_spitzer, title='var_nl2d_cibernoise_cross_spitzer', lopct=5, hipct=95)
			plot_map(total_var_nl2d, title='total_var_nl2d', lopct=5, hipct=95)
			plot_map(fourier_weights_cross, title='fourier_weights_cross', lopct=5, hipct=95)

		else:
			fourier_weights_cross = None
			  
		if plot:
			plt.figure()
			pf = lb*(lb+1)/(2*np.pi)
			plt.title('CIBER noise x Spitzer')
			for nl in all_nl1ds_cross_spitzer:
				plt.plot(lb,pf*nl, color='grey', alpha=0.1)
			plt.errorbar(lb, pf*np.mean(all_nl1ds_cross_spitzer, axis=0), yerr=pf*std_nl1ds_cibernoise_spitzer_cross, color='r', capsize=4)
			plt.xscale('log')
			plt.xlabel('$\\ell$', fontsize=14)
			plt.ylabel('$D_{\\ell}$', fontsize=14)
			plt.show()
			
			plt.figure()
			plt.title('Spitzer noise x CIBER')
			pf = lb*(lb+1)/(2*np.pi)
			for nl in all_nl1ds_cross_ciber:
				plt.plot(lb,pf*nl, color='grey', alpha=0.1)
			plt.errorbar(lb, pf*np.mean(all_nl1ds_cross_ciber, axis=0), yerr=pf*std_nl1ds_spitzernoise_ciber_cross, color='r', capsize=4)
			plt.xscale('log')
			plt.xlabel('$\\ell$', fontsize=14)
			plt.ylabel('$D_{\\ell}$', fontsize=14)
			plt.show()
			
			plt.figure()
			plt.title('Spitzer noise x CIBER Dl, inst '+str(inst)+', ifield '+str(ifield)+', irac ch '+str(irac_ch))
			plt.plot(lb, 1e6*pf*std_nl1ds_spitzernoise_ciber_cross, color='r', linestyle='dashed')
			plt.xscale('log')
			plt.yscale('log')
			plt.xlabel('$\\ell$', fontsize=14)
			plt.ylabel('$D_{\\ell}$', fontsize=14)
			plt.show()
			
			plt.figure()
			plt.title('Spitzer noise, CIBER TM'+str(inst)+', ifield '+str(ifield)+', irac ch '+str(irac_ch))
			plt.plot(lb, pf*av_nl1ds_spitzernoise_ciber_cross, color='r', linestyle='dashed')
			plt.xscale('log')
			plt.yscale('log')
			plt.xlabel('$\\ell$', fontsize=14)
			plt.ylabel('$D_{\\ell}$', fontsize=14)
			plt.show()

		spitzmask_bootes_obs = bootes_obs[idx]*combined_masks[idx]
		
		for q in range(4):
			quad_spitz = spitzmask_bootes_obs[cbps_nm.cbps.x0s[q]:cbps_nm.cbps.x1s[q], cbps_nm.cbps.y0s[q]:cbps_nm.cbps.y1s[q]]
			spitzmask_bootes_obs[cbps_nm.cbps.x0s[q]:cbps_nm.cbps.x1s[q], cbps_nm.cbps.y0s[q]:cbps_nm.cbps.y1s[q]][quad_spitz!=0] -= np.mean(quad_spitz[quad_spitz!=0])

		noisemodl_run_name = 'observed_quadsub_nochisq_rdnoise_022223'
		noisemodl_base_path = 'data/noise_models_sim/112022/TM'+str(inst)+'/'
		noisemodl_tailpath = '/noise_bias_fieldidx'+str(bootes_idxs[idx])+'.npz'
		noisemodl_fpath = noisemodl_base_path + noisemodl_run_name + noisemodl_tailpath
		print('LOADING NOISE MODEL FROM ', noisemodl_fpath)

		noisemodl_file = np.load(noisemodl_fpath)
		fourier_weights_nofluc = noisemodl_file['fourier_weights_nofluc']

		if plot:
			plot_map(spitzmask_bootes_obs, title='Filtered CIBER mask, ifield'+str(ifield)+', TM'+str(inst))
			plot_map(gaussian_filter(spitzmask_bootes_obs, sigma=20), title='Smoothed CIBER mask, ifield'+str(ifield)+', TM'+str(inst))

		# lb, processed_ps_nf, cl_proc_err = get_power_spec(spitzmask_bootes_obs, map_b=spitzer_regrid_maps_meansub[idx], mask=None, weights=fourier_weights_nofluc, lbinedges=cbps_nm.cbps.Mkk_obj.binl, lbins=cbps_nm.cbps.Mkk_obj.midbin_ell)
		lb, processed_ps_nf, cl_proc_err = get_power_spec(spitzmask_bootes_obs, map_b=spitzer_regrid_maps_meansub[idx], mask=None, weights=fourier_weights_cross, lbinedges=cbps_nm.cbps.Mkk_obj.binl, lbins=cbps_nm.cbps.Mkk_obj.midbin_ell)

		lb, processed_spitz_auto, cl_proc_err_spitz_auto = get_power_spec(spitzer_regrid_maps_meansub[idx], mask=None, weights=None, lbinedges=cbps_nm.cbps.Mkk_obj.binl, lbins=cbps_nm.cbps.Mkk_obj.midbin_ell)


		prefac_cross = lb*(lb+1)/(2*np.pi)

		print('Correcting for mode coupling')
		processed_ps_nf = np.dot(inv_Mkk.transpose(), processed_ps_nf)
		print('Correcting for beam transfer function..')
		processed_ps_nf /= bootes_bls[idx]**2
		cl_proc_err /= bootes_bls[idx]**2
		print('Correcting for filtering transfer function..')
		t_ell_av_grad = np.load('data/transfer_function/t_ell_est_nsims=100.npz')['t_ell_av']
		t_ell_av_quad = np.load('data/transfer_function/t_ell_masked_wdgl_2500_n=2p5_wgrad_quadrant_nsims=1000.npz')['t_ell_av']
		t_ell_av = np.sqrt(t_ell_av_grad*t_ell_av_quad)
		
		plt.figure()
		plt.plot(lb, t_ell_av_grad, label='Spitzer transfer function')
		plt.plot(lb, t_ell_av_quad, label='CIBER transfer function')

		plt.plot(lb, t_ell_av, label='Geometric mean')
		plt.xscale('log')
		plt.yscale('log')
		plt.xlabel('$\\ell$', fontsize=14)
		plt.ylabel('$T_{\\ell}$', fontsize=14)
		plt.show()

		print('t_ell av:', t_ell_av)
		processed_ps_nf /= t_ell_av
		processed_spitz_auto /= t_ell_av_grad
		cl_proc_err_spitz_auto /= t_ell_av_grad

		combined_mask_fraction = float(np.sum(combined_masks[idx]))/float(cbps_nm.cbps.dimx**2)
		print('dividing total error by combined mask fraction = ', combined_mask_fraction)

		cl_proc_err /= combined_mask_fraction
		std_nl1ds_cibernoise_spitzer_cross /= combined_mask_fraction
		std_nl1ds_spitzernoise_ciber_cross /= combined_mask_fraction

		clerr_cross_tot = np.sqrt(std_nl1ds_cibernoise_spitzer_cross**2+std_nl1ds_spitzernoise_ciber_cross**2)
		# clerr_cross_tot = np.sqrt(cl_proc_err**2+std_nl1ds_cibernoise_spitzer_cross**2+std_nl1ds_spitzernoise_ciber_cross**2)
		clerr_cross_tot /= t_ell_av
		clerr_cross_tot /= bootes_bls[idx]**2

		processed_spitz_auto /= cibermatch_bl**2
		cl_proc_err_spitz_auto /= cibermatch_bl**2



		all_cl_cross.append(processed_ps_nf)
		all_clerr_cross.append(cl_proc_err)
		all_clerr_cross_tot.append(clerr_cross_tot)
		
		all_clerr_cibernoise_spitzer_cross.append(std_nl1ds_cibernoise_spitzer_cross/bootes_bls[idx]**2/t_ell_av)
		all_clerr_spitzernoise_ciber_cross.append(std_nl1ds_spitzernoise_ciber_cross/bootes_bls[idx]**2/t_ell_av)
		
		if plot:
			

			plt.figure()
			plt.title('Spitzer auto (epoch average), inst '+str(inst)+', ifield '+str(ifield)+', irac ch '+str(irac_ch))
			plt.plot(lb, pf*processed_spitz_auto, color='k')
			plt.xscale('log')
			plt.yscale('log')
			plt.xlabel('$\\ell$', fontsize=14)
			plt.ylabel('$D_{\\ell}$', fontsize=14)
			plt.legend()
			plt.show()
			
			plt.figure()
			plt.title('Spitzer noise x CIBER ell*Cl/2pi, inst '+str(inst)+', ifield '+str(ifield)+', irac ch '+str(irac_ch))
			plt.plot(lb, pf*clerr_cross_tot, color='k', linestyle='dashed', label='total')
			plt.plot(lb, pf*std_nl1ds_cibernoise_spitzer_cross, color='C0', linestyle='dashed', label='CIBER noise x Spitzer')
			plt.plot(lb, pf*std_nl1ds_spitzernoise_ciber_cross, color='C1', linestyle='dashed', label='Spitzer noise x CIBER')

			plt.plot(lb, pf*cl_proc_err, color='C3', linestyle='dashed', label='Dispersion of modes in bandpower')
			plt.scatter(lb, pf*processed_ps_nf, color='k')
			plt.xscale('log')
			plt.yscale('log')
			plt.xlabel('$\\ell$', fontsize=14)
			plt.ylabel('$D_{\\ell}$', fontsize=14)
			plt.legend()
			plt.show()
			
			plt.figure()
			plt.errorbar(lb, prefac_cross*processed_ps_nf, yerr=prefac_cross*clerr_cross_tot, color='k')
			plt.xscale('log')
			plt.yscale('log')
			plt.plot(lb, prefac_cross*clerr_cross_tot, color='k', linestyle='dashed')
			plt.show()
		
	
	save_cl_fpath = base_path+'/cl_spitzer_ciber_xcorr_TM'+str(inst)+'_'+spitzer_string+'_'+add_str_spitzer+'_'+add_str_ciber+'.npz'
		
	if save:
		print('saving to ', save_cl_fpath)
		np.savez(save_cl_fpath, \
			lb_cross=lb, prefac_cross=prefac_cross, all_cl_cross=all_cl_cross, \
				all_clerr_cross=all_clerr_cross, \
				all_clerr_cross_tot = all_clerr_cross_tot,\
				 all_clerr_spitzernoise_ciber_cross=all_clerr_spitzernoise_ciber_cross, \
				all_clerr_cibernoise_spitzer_cross=all_clerr_cibernoise_spitzer_cross, \
				all_nl1d_diff=all_nl1d_diff, all_nl1d_err_diff=all_nl1d_err_diff)
	
	return save_cl_fpath


# load fourier weights
# noisemodl_run_name = 'observed_quadsub_nochisq_rdnoise_022223'
# noisemodl_base_path = 'data/noise_models_sim/112022/TM'+str(inst)+'/'
# noisemodl_tailpath = '/noise_bias_fieldidx'+str(bootes_fieldidxs[bootes_fieldidx])+'.npz'
# noisemodl_fpath = noisemodl_base_path + noisemodl_run_name + noisemodl_tailpath
# print('LOADING NOISE MODEL FROM ', noisemodl_fpath)
# noisemodl_file = np.load(noisemodl_fpath)
# fourier_weights_nofluc = noisemodl_file['fourier_weights_nofluc']
# plot_map(fourier_weights_nofluc)
		

def estimate_spitzer_noise_cross_ciber(cbps_nm, irac_ch, inst, bootes_ciber_maps, combined_masks, spitzer_string, bootes_ifield_list=[6, 7], bootes_fieldidxs=[2,3],\
									   nsims=200, n_split=4, plot=True, fft2_spitzer=None, compute_spitzer_FW=False, \
									  base_path='/Users/richardfeder/Downloads/ciber_downloads/', save=True, add_str=None):
	
	''' feed in bootes maps for CIBER and their maps'''
	maplist_split_shape = (nsims//n_split, cbps_nm.cbps.dimx, cbps_nm.cbps.dimy)
	empty_aligned_objs, fft_objs = construct_pyfftw_objs(3, maplist_split_shape)
	empty_aligned_objs_cross, fft_objs_cross = construct_pyfftw_objs(3, (1, cbps_nm.cbps.dimx, cbps_nm.cbps.dimy))
	empty_aligned_objs_maps, fft_objs_maps = construct_pyfftw_objs(3, (1, maplist_split_shape[1], maplist_split_shape[2]))

	lb = cbps_nm.cbps.Mkk_obj.midbin_ell

	sterad_per_pix = (cbps_nm.cbps.pixsize/3600/180*np.pi)**2
	V = cbps_nm.cbps.dimx*cbps_nm.cbps.dimy*sterad_per_pix
	l2d = get_l2d(cbps_nm.cbps.dimx, cbps_nm.cbps.dimy, cbps_nm.cbps.pixsize)
	
	if len(fft2_spitzer.shape)==2:
		print('single fft2, recasting into list of nfield='+str(len(bootes_ifield_list))+' copies..')
		fft2_spitzer = [fft2_spitzer for x in range(len(bootes_ifield_list))]
		plot_map(fft2_spitzer[0])

	all_nl1ds_spitzer_both, all_nl1ds_cross_ciber_both, all_nl_save_fpaths, all_fourier_weights = [[] for x in range(4)]

	for bootes_fieldidx, bootes_ifield in enumerate(bootes_ifield_list):
		
		if not compute_spitzer_FW:
			fourier_weights = []


		all_nl1ds_spitzer, all_nl1ds_cross_ciber = [], []

		ciber_bootes_obs_indiv = bootes_ciber_maps[bootes_fieldidx]
		combined_mask_indiv = combined_masks[bootes_fieldidx]
		
		masked_ciber_bootes_obs = ciber_bootes_obs_indiv*combined_mask_indiv
		
		count = 0

		mean_nl2d, M2_nl2d = [np.zeros(ciber_bootes_obs_indiv.shape) for x in range(2)]
		
		for q in range(4):
			quad_ciber = masked_ciber_bootes_obs[cbps_nm.cbps.x0s[q]:cbps_nm.cbps.x1s[q], cbps_nm.cbps.y0s[q]:cbps_nm.cbps.y1s[q]]
			masked_ciber_bootes_obs[cbps_nm.cbps.x0s[q]:cbps_nm.cbps.x1s[q], cbps_nm.cbps.y0s[q]:cbps_nm.cbps.y1s[q]][quad_ciber!=0] -= np.mean(quad_ciber[quad_ciber!=0])

		fft_objs_maps[1](np.array([masked_ciber_bootes_obs*sterad_per_pix]))

		if plot:
			plot_map(combined_mask_indiv, title='combined mask')
			plot_map(masked_ciber_bootes_obs, title='CIBER obs * combined_mask')

		for i in range(n_split):
			print('Split '+str(i+1)+' of '+str(n_split)+'..')
			if i==0:
				plot_map(fft2_spitzer[bootes_fieldidx], title='fft2 bootes fieldidx')
				plot_map(np.log10(np.abs(fft2_spitzer[bootes_fieldidx])), title='log abs fft2 bootes fieldidx')
			
			# read/photon noise is from the CIBER convention, read_noise=True just means Gaussian realizations from fft2_spitzer here
			spitzer_noise_realiz, _ = cbps_nm.cbps.noise_model_realization(inst, maplist_split_shape, fft2_spitzer[bootes_fieldidx], fft_obj=fft_objs[0],\
												  read_noise=True, photon_noise=False, adu_to_sb=False)

			spitzer_noise_realiz /= cbps_nm.cbps.arcsec_pp_to_radian
			
			if i==0:
				plot_map(spitzer_noise_realiz[0]*combined_mask_indiv, title='spitzer noise')
				plot_map(gaussian_filter(masked_ciber_bootes_obs, sigma=10), title='smoothed ciber map')
				# lb = get_power_spec(combined_mask_indiv*spitzer_noise_realiz[0], lbinedges=cbps_nm.cbps.Mkk_obj.binl, lbins=cbps_nm.cbps.Mkk_obj.midbin_ell)[0]
				print('lb is ', lb)
		
			fft_objs[1](spitzer_noise_realiz*sterad_per_pix)

			# if compute_spitzer_FW:
			# 	# use fft objects to get running estimate of 2D PS and Fourier weights
			# 	cl2ds = np.array([fftshift(dentry*np.conj(dentry)).real for dentry in empty_aligned_objs[2]])
			# 	count, mean_nl2d, M2_nl2d = update_meanvar(count, mean_nl2d, M2_nl2d, cl2ds/V/cbps_nm.cbps.cal_facs[inst]**2)


			# nl1ds_spitzer = [get_power_spec(combined_mask_indiv*indiv_map, weights=None, lbinedges=cbps_nm.cbps.Mkk_obj.binl, lbins=cbps_nm.cbps.Mkk_obj.midbin_ell)[1] for indiv_map in spitzer_noise_realiz]

			
			nl2ds_spitzer_noise_ciber_map = np.array([fftshift(dentry*np.conj(empty_aligned_objs_maps[2][0])).real for dentry in empty_aligned_objs[2]])
			# nl1ds_cross_ciber = [azim_average_cl2d(nl2d/V/(cbps_nm.cbps.cal_facs[inst]), l2d, lbinedges=cbps_nm.cbps.Mkk_obj.binl, lbins=cbps_nm.cbps.Mkk_obj.midbin_ell)[1] for nl2d in nl2ds_spitzer_noise_ciber_map]
			nl1ds_cross_ciber = [azim_average_cl2d(nl2d/V, l2d, lbinedges=cbps_nm.cbps.Mkk_obj.binl, lbins=cbps_nm.cbps.Mkk_obj.midbin_ell)[1] for nl2d in nl2ds_spitzer_noise_ciber_map]

			if compute_spitzer_FW:
				# use fft objects to get running estimate of 2D PS and Fourier weights
				count, mean_nl2d, M2_nl2d = update_meanvar(count, mean_nl2d, M2_nl2d, nl2ds_spitzer_noise_ciber_map/V/cbps_nm.cbps.cal_facs[inst])


			# nl1ds_cross_ciber = [get_power_spec(combined_mask_indiv*indiv_map, map_b=masked_ciber_bootes_obs, weights=None, lbinedges=cbps_nm.cbps.Mkk_obj.binl, lbins=cbps_nm.cbps.Mkk_obj.midbin_ell)[1] for indiv_map in spitzer_noise_realiz]

			# all_nl1ds_spitzer.extend(nl1ds_spitzer)
			all_nl1ds_cross_ciber.extend(nl1ds_cross_ciber)
			
		# all_nl1ds_spitzer = np.array(all_nl1ds_spitzer)
		all_nl1ds_cross_ciber = np.array(all_nl1ds_cross_ciber)
		
		# all_nl1ds_spitzer_both.append(all_nl1ds_spitzer)
		all_nl1ds_cross_ciber_both.append(all_nl1ds_cross_ciber)
		
		if compute_spitzer_FW:
			mean_nl2d, var_nl2d, svar_nl2d = finalize_meanvar(count, mean_nl2d, M2_nl2d)
			fourier_weights = 1./var_nl2d
			plot_map(fourier_weights, title='Fourier weights')
			all_fourier_weights.append(fourier_weights)
			# all_var_nl2d.append(var_nl2d)
		
		if save:
			if add_str is not None:
				if bootes_fieldidx == 0:
					add_str = '_'+add_str
			else:
				add_str = ''

			print('add string:', add_str)
			nl_save_fpath = base_path+'ciber_meeting_030223/nl1ds_TM'+str(inst)+'_ifield'+str(bootes_ifield)+'_irac_ch'+str(irac_ch)+'_spitzernoise_ciber_cross'+add_str+'.npz'

			all_nl_save_fpaths.append(nl_save_fpath)
#             np.savez(nl_save_fpath, \
#                  all_nl1ds_spitzer=all_nl1ds_spitzer, all_nl1ds_cross_ciber=all_nl1ds_cross_ciber, lb=cbps_nm.cbps.Mkk_obj.midbin_ell)
						
			np.savez(nl_save_fpath, fourier_weights_spitzer=fourier_weights, \
				 all_nl1ds_spitzer=all_nl1ds_spitzer, all_nl1ds_cross_ciber=all_nl1ds_cross_ciber, lb=lb, \
				 var_nl2d=var_nl2d, mean_nl2d=mean_nl2d)
			
	if save:
		return all_nl_save_fpaths, lb, all_nl1ds_spitzer, all_nl1ds_cross_ciber

	else:
		return lb, all_nl1ds_spitzer_both, all_nl1ds_cross_ciber_both


def estimate_ciber_noise_cross_spitzer(cbps_nm, irac_ch, inst, bootes_spitzer_maps, combined_masks, spitzer_string, bootes_ifield_list=[6, 7], bootes_fieldidxs=[2,3],\
									   nsims=200, n_split=10, plot=True, read_noise=True, photon_noise=False, \
									   include_ff_errors=True, observed_run_name=None, nmc_ff=10, \
									  base_path='/Users/richardfeder/Downloads/ciber_downloads/',\
									   ff_est_dirpath=None, save=True, add_str=None, n_FF_realiz=10):
	
	if ff_est_dirpath is None:
		ff_est_dirpath = 'data/ff_mc_ests/112022/TM'+str(inst)+'/'

	''' feed in bootes maps for CIBER and their maps'''
	maplist_split_shape = (nsims//n_split, cbps_nm.cbps.dimx, cbps_nm.cbps.dimy)
	empty_aligned_objs, fft_objs = construct_pyfftw_objs(3, maplist_split_shape)
	empty_aligned_objs_maps, fft_objs_maps = construct_pyfftw_objs(3, (1, maplist_split_shape[1], maplist_split_shape[2]))

	sterad_per_pix = (cbps_nm.cbps.pixsize/3600/180*np.pi)**2
	V = cbps_nm.cbps.dimx*cbps_nm.cbps.dimy*sterad_per_pix
	l2d = get_l2d(cbps_nm.cbps.dimx, cbps_nm.cbps.dimy, cbps_nm.cbps.pixsize)

	lb = cbps_nm.cbps.Mkk_obj.midbin_ell
	all_nl1ds_cross_spitzer_both, nl_save_fpath_both = [], []
	
	nmfile = np.load(base_path+'noise_model_fpaths_TM'+str(inst)+'_021523.npz')
	noise_model_fpaths_quad = nmfile['noise_model_fpaths_quad']
	ciber_bootes_noise_models = [fits.open(noise_model_fpaths_quad[f])[1].data for f in bootes_fieldidxs]
	
	all_ff_ests_nofluc_both, simmap_dc = None, None
	# if include_ff_errors and observed_run_name is not None:
	if include_ff_errors or photon_noise:
		if observed_run_name is not None:
			all_ff_ests_nofluc_both, simmap_dc = [], []
			for idx, bootes_ifield in enumerate(bootes_ifield_list):
				# load the MC FF estimates obtained by several draws of noise
				if include_ff_errors:
					all_ff_ests_nofluc = []
					for ffidx in range(n_FF_realiz):
						ff_file = np.load(ff_est_dirpath+'/'+observed_run_name+'/ff_realization_estimate_ffidx'+str(ffidx)+'.npz')
						all_ff_ests_nofluc.append(ff_file['ff_realization_estimates'][bootes_fieldidxs[idx]])
						if ffidx==0:
							plot_map(all_ff_ests_nofluc[0], title='loaded MC ff estimate 0')
					
					all_ff_ests_nofluc_both.append(all_ff_ests_nofluc)
				simmap_dc.append(cbps_nm.cbps.zl_levels_ciber_fields[inst][cbps_nm.cbps.ciber_field_dict[bootes_ifield]])
				
			print("simmap dc:", simmap_dc)
			if include_ff_errors:
				all_ff_ests_nofluc_both = np.array(all_ff_ests_nofluc_both)
				print('all_ff_ests_nofluc_both has shape', all_ff_ests_nofluc_both.shape)
				
	
	for bootes_fieldidx, bootes_ifield in enumerate(bootes_ifield_list):
		
		mean_nl2d, M2_nl2d = [np.zeros_like(ciber_bootes_noise_models[0]) for x in range(2)]
		count = 0

		if photon_noise:
			field_nfr = cbps_nm.cbps.field_nfrs[bootes_ifield]
			print('field nfr for '+str(bootes_ifield)+' is '+str(field_nfr))

			shot_sigma_sb = cbps_nm.cbps.compute_shot_sigma_map(inst, image=simmap_dc[bootes_fieldidx]*np.ones_like(ciber_bootes_noise_models[0]), nfr=field_nfr)
			plot_map(shot_sigma_sb, title='shot sigma sb for CIBER noise x Spitzer')
		else:
			shot_sigma_sb = None

		all_nl1ds_cross_spitzer = []
		
		spitzer_bootes_obs_indiv = bootes_spitzer_maps[bootes_fieldidx]
		combined_mask_indiv = combined_masks[bootes_fieldidx]

		fft_objs_maps[1](np.array([combined_mask_indiv*spitzer_bootes_obs_indiv*sterad_per_pix]))

		
		if plot:
			plot_map(combined_mask_indiv, title='combined mask')
			plot_map(spitzer_bootes_obs_indiv*combined_mask_indiv, title='CIBER obs * combined_mask')
			plot_map(ciber_bootes_noise_models[bootes_fieldidx], title='CIBER noise model')
			
		for i in range(n_split):
			print('Split '+str(i+1)+' of '+str(n_split)+'..')

			ciber_noise_realiz, snmaps = cbps_nm.cbps.noise_model_realization(inst, maplist_split_shape, ciber_bootes_noise_models[bootes_fieldidx], fft_obj=fft_objs[0],\
												  read_noise=read_noise, photon_noise=photon_noise, shot_sigma_sb=shot_sigma_sb, adu_to_sb=True)
			
			if photon_noise:
				print('adding photon noise to read noise realizations')
				# plot_map(snmaps[0], title='snmaps[0]')
				ciber_noise_realiz += snmaps
			
			if i==0 and plot:
				plot_map(ciber_noise_realiz[0])
				plot_map(gaussian_filter(ciber_noise_realiz[0], sigma=10))
			
			
			if all_ff_ests_nofluc_both is not None:
				# print('adding ')
				ciber_noise_realiz += simmap_dc[bootes_fieldidx]
				# print('all_ff_ests_nofluc_both[bootes_fieldidx][i] has shape:', all_ff_ests_nofluc_both[bootes_fieldidx][i%n_FF_realiz].shape)
				ciber_noise_realiz /= all_ff_ests_nofluc_both[bootes_fieldidx][i%n_FF_realiz]
				
				if i==0:
					plot_map(ciber_noise_realiz[0], title='added normalization and flat field error')
				
			unmasked_means = [np.mean(simmap[combined_mask_indiv==1]) for simmap in ciber_noise_realiz]
			ciber_noise_realiz -= np.array([combined_mask_indiv*unmasked_mean for unmasked_mean in unmasked_means])

			fft_objs[1](combined_mask_indiv*ciber_noise_realiz*sterad_per_pix)
			# cl2ds = np.array([fftshift(dentry*np.conj(dentry)).real for dentry in empty_aligned_objs[2]])
			# count, mean_nl2d, M2_nl2d = update_meanvar(count, mean_nl2d, M2_nl2d, cl2ds/V/cbps_nm.cbps.cal_facs[inst]**2)


			# all_masked_realiz = np.array([combined_mask_indiv*indiv_map for indiv_map in ciber_noise_realiz])
			# for m, masked_realiz in enumerate(all_masked_realiz):
			# 	meansub = np.mean(masked_realiz[masked_realiz!=0])
			# 	if m==0:
			# 		print('meansub[0] = ', meansub)
			# 	all_masked_realiz[m][masked_realiz!=0] -= np.mean(masked_realiz[masked_realiz!=0])

			# if i==0:
			# 	plot_map(all_masked_realiz[0], title='mean subtracted full noise realization')
			

			nl2ds_ciber_noise_spitzer_map = np.array([fftshift(dentry*np.conj(empty_aligned_objs_maps[2][0])).real for dentry in empty_aligned_objs[2]])
			nl1ds_cross_spitzer = [azim_average_cl2d(nl2d/V, l2d, lbinedges=cbps_nm.cbps.Mkk_obj.binl, lbins=cbps_nm.cbps.Mkk_obj.midbin_ell)[1] for nl2d in nl2ds_ciber_noise_spitzer_map]
			count, mean_nl2d, M2_nl2d = update_meanvar(count, mean_nl2d, M2_nl2d, nl2ds_ciber_noise_spitzer_map/V)


			# nl1ds_cross_spitzer = [get_power_spec(indiv_map, map_b=spitzer_bootes_obs_indiv, weights=None, lbinedges=cbps_nm.cbps.Mkk_obj.binl, lbins=cbps_nm.cbps.Mkk_obj.midbin_ell)[1] for indiv_map in all_masked_realiz]

			all_nl1ds_cross_spitzer.extend(nl1ds_cross_spitzer)
			
		all_nl1ds_cross_spitzer = np.array(all_nl1ds_cross_spitzer)
		all_nl1ds_cross_spitzer_both.append(all_nl1ds_cross_spitzer)

		mean_nl2d, var_nl2d, svar_nl2d = finalize_meanvar(count, mean_nl2d, M2_nl2d)

		plot_map(var_nl2d, title='var nl2d')
		plot_map(mean_nl2d, title='mean nl2d')
		
		if save:
			if add_str is not None:
				if bootes_fieldidx==0:
					add_str = '_'+add_str
			else:
				add_str = ''
			nl_save_fpath = base_path+'ciber_meeting_030223/nl1ds_TM'+str(inst)+'_ifield'+str(bootes_ifield)+'_irac_ch'+str(irac_ch)+'_cibernoise_spitzer_cross'+add_str+'.npz'
			nl_save_fpath_both.append(nl_save_fpath)
			print('saving Nl realizations to ', nl_save_fpath)
			np.savez(nl_save_fpath, all_nl1ds_cross_spitzer=all_nl1ds_cross_spitzer, lb=lb, var_nl2d=var_nl2d, mean_nl2d=mean_nl2d)
			
	if save:
		return nl_save_fpath_both, lb, all_nl1ds_cross_spitzer_both
	else:
		return lb, all_nl1ds_cross_spitzer_both


