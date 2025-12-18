import numpy as np
import astropy.io.fits as fits
import astropy.wcs as wcs
import pandas as pd
from PIL import Image       
import config
import scipy
import scipy.io
from scipy.ndimage import gaussian_filter
import matplotlib
import matplotlib.pyplot as plt
from ciber.pseudo_cl.mkk_compute import *
from ciber.core.powerspec_pipeline import *
from ciber.mocks.cib_mocks import *


def save_cl_predictions(inst, ifield_list, mag_lims, power_maglim_isl_igl, power_maglim_igl, keywd='meas'):
	save_fpaths = []
	for x in range(len(mag_lims)):
		save_fpath = config.ciber_basepath+'data/cl_predictions/TM'+str(inst)+'/igl_isl_pred_mlim='+str(mag_lims[x])+'_'+keywd+'.npz'
		np.savez(save_fpath, isl_igl=power_maglim_isl_igl[x], igl=power_maglim_igl[x])
		save_fpaths.append(save_fpath)
		
	return save_fpaths


# def generate_auto_dl_trilegal_simp(lb, cat_sel, idx, lam, aeff=4.0):
# 	cmock = ciber_mock()
# 	nbands = len(lams)

# 	Vega_to_AB_lam = dict({1.1:0.91, 1.8:1.39, 3.6:2.699, 4.5:3.339})
	
# 	volfac = (1./aeff)/3.046e-4
# 	pf = lb*(lb+1)/(2*np.pi)

# 	nu_Inu = cmock.mag_2_nu_Inu(cat_sel[:,idx], lam_eff=lam*1e-6*u.m)*cmock.pix_sr
	
# 	auto_poisson_var = np.sum(volfac*(nu_Inu**2))
# 	auto_dl_pred = pf*(auto_poisson_var.value)  

# 	return auto_dl_pred  

def field_av_trilegal_predictions(cbps, lb, ifield_list=[4, 5, 6, 7, 8], L_cut=17.7):

    cl_matrix_perf = np.zeros((len(ifield_list), 4, 4, len(lb)))

    for fieldidx, ifield in enumerate(ifield_list):

        fieldname = cbps.ciber_field_dict[ifield]

        trilegal_fpath = 'data/TRILEGAL/trilegal_2MASS_Spitzer_WISE_'+fieldname+'.dat'
        
        trileg = make_synthetic_trilegal_cat_2MSW(trilegal_fpath)

        whichsel3 = (trileg[:,2] > 16.9)*(trileg[:,3] > 16.9)
        whichsel3 *= (trileg[:,4] > L_cut)

        trileg_sel = trileg[whichsel3,2:]

        cl_matrix_perf[fieldidx] = generate_trilegal_cl_matrix_from_cat(lb, trileg_sel)
        
    cl_matrix_field_av = np.mean(cl_matrix_perf, axis=0)
        
    return cl_matrix_field_av, cl_matrix_perf
    

# def calc_cross_cl_shot_noise(cat_sel, lam, idx, aeff=4.0):
def calc_cross_cl_shot_noise(mags, lam, aeff=4.0):

    """
    Calculate cross-shot noise power between CIB intensity and a galaxy catalog.
    
    lb      : array of ell values
    cat_sel : mock catalog selection (rows = sources, columns = properties)
    lam     : wavelength in microns
    idx     : column index for magnitudes in cat_sel
    aeff    : effective survey area in deg^2
    """
    cmock = ciber_mock()

    # steradian per mock area
    volfac = (1. / aeff) / 3.046e-4  # 1 deg^2 = 3.046e-4 sr

    # Convert magnitudes to nu*I_nu per pixel (flux density per sr)
    # nu_Inu = cmock.mag_2_nu_Inu(cat_sel[:, idx], lam_eff=lam*1e-6*u.m) * cmock.pix_sr
    nu_Inu = cmock.mag_2_nu_Inu(mags, lam_eff=lam*1e-6*u.m) * cmock.pix_sr

    # Number of galaxies in catalog
    N_gal = len(mags)

    # Surface density of galaxies (per sr)
    n_gal_sr = N_gal * volfac  # galaxies / sr

    # Sum of fluxes for galaxies in bin (W/m^2/sr or equivalent units)
    sum_flux = np.sum(volfac * nu_Inu)  # integrated over area

    # Cross-shot noise amplitude (flat in ell)
    cross_poisson_var = sum_flux / n_gal_sr  # matches your formula

    cross_cl_pred = cross_poisson_var.value

    return cross_cl_pred

def calc_auto_dl_trilegal_simp(lb, cat_sel, lam, idx, aeff=4.0):
    cmock = ciber_mock()

    volfac = (1./aeff)/3.046e-4
    pf = lb*(lb+1)/(2*np.pi)

    nu_Inu = cmock.mag_2_nu_Inu(cat_sel[:,idx], lam_eff=lam*1e-6*u.m)*cmock.pix_sr

    auto_poisson_var = np.sum(volfac*(nu_Inu**2))
    auto_dl_pred = pf*(auto_poisson_var.value)

    return auto_dl_pred   


def generate_auto_dl_pred_trilegal(lb_use, inst, maglim, ifield_list=[4, 5, 6, 7, 8], m_max=28):
    
    cbps = CIBER_PS_pipeline()
    m_min_perfield = dict({4: 5.69, 5: 4.52, 6: 7.82, 7:7.20, 8:6.63}) # J band
    isl_dl_pred = np.zeros((len(ifield_list), len(lb_use)))
    trilegal_base_path = config.ciber_basepath+'data/mock_trilegal_realizations_111722/'
    
    Vega_to_AB = dict({1:0.91, 2:1.39})
    lam_dict = dict({1:1.05, 2:1.79})
    idx_dict = dict({1:2, 2:3})
    
    for fieldidx, ifield in enumerate(ifield_list):
        trilegal_path = trilegal_base_path+'trilegal_'+cbps.ciber_field_dict[ifield]+'.dat'
        trilegal_cat = make_synthetic_trilegal_cat(trilegal_path, J_band_idx=26, H_band_idx=27)

        filt_cat = filter_trilegal_cat(trilegal_cat, filter_band_idx=2, m_min=m_min_perfield[ifield], m_max=m_max)
        m_AB = maglim + Vega_to_AB[inst]
        mask_sel = np.where(((filt_cat[:,idx_dict[inst]] > m_AB)))[0]
        trilegal_cat_sel = filt_cat[mask_sel, :]
        
        isl_dl_pred[fieldidx] = calc_auto_dl_trilegal_simp(lb_use, trilegal_cat_sel, lam_dict[inst], idx_dict[inst])

    return isl_dl_pred


def generate_cross_dl_pred_trilegal(lb_use, maglim_J, maglim_H, ifield_list=[4, 5, 6, 7, 8], idx=2, idxc=3, m_max=28):
    
    cbps = CIBER_PS_pipeline()
    m_min_perfield = dict({4: 5.69, 5: 4.52, 6: 7.82, 7:7.20, 8:6.63}) # J band

    isl_dl_pred = np.zeros((len(ifield_list), len(lb_use)))

    trilegal_base_path = config.ciber_basepath+'data/mock_trilegal_realizations_111722/'

    for fieldidx, ifield in enumerate(ifield_list):

        trilegal_path = trilegal_base_path+'trilegal_'+cbps.ciber_field_dict[ifield]+'.dat'
        trilegal_cat = make_synthetic_trilegal_cat(trilegal_path, J_band_idx=26, H_band_idx=27)

        filt_cat = filter_trilegal_cat(trilegal_cat, filter_band_idx=2, m_min=m_min_perfield[ifield], m_max=m_max)

        J_AB, H_AB = maglim_J+0.91, maglim_H+1.39
        mask_sel = np.where(((filt_cat[:,idx] > J_AB)*(filt_cat[:,idxc] > H_AB)))[0]

        trilegal_cat_sel = filt_cat[mask_sel, :]

        print('trilegal cat sel has shape:', trilegal_cat_sel.shape)
        print('first row:', trilegal_cat_sel[0,:])

        isl_dl_pred[fieldidx] = calc_cross_dl_trilegal_simp(lb_use, trilegal_cat_sel, 1.05, 1.79, idx, idxc)
        
    return isl_dl_pred

def calc_cross_dl_trilegal_simp(lb, cat_sel, lam, lam_cross, idx, idxc, aeff=4.0):
    cmock = ciber_mock()

    volfac = (1./aeff)/3.046e-4
    pf = lb*(lb+1)/(2*np.pi)

    nu_Inu = cmock.mag_2_nu_Inu(cat_sel[:,idx], lam_eff=lam*1e-6*u.m)*cmock.pix_sr
    nu_Inu_cross = cmock.mag_2_nu_Inu(cat_sel[:,idxc], lam_eff=lam_cross*1e-6*u.m)*cmock.pix_sr

    cross_poisson_var = np.sum(volfac*(nu_Inu*nu_Inu_cross))
    cross_dl_pred = pf*(cross_poisson_var.value)

    return cross_dl_pred   


def generate_trilegal_cl_matrix_from_cat(lb, cosmos_cat_sel, lams=[1.1, 1.8, 3.6, 4.5], aeff=4.0):

    cmock = ciber_mock()
    nbands = len(lams)

    Vega_to_AB_lam = dict({1.1:0.91, 1.8:1.39, 3.6:2.699, 4.5:3.339})
    
    volfac = (1./aeff)/3.046e-4
    
    cl_matrix = np.zeros((nbands, nbands, len(lb)))
    
    pf = lb*(lb+1)/(2*np.pi)
    
    for idx, lam in enumerate(lams):
                
        nu_Inu = cmock.mag_2_nu_Inu(cosmos_cat_sel[:,idx], lam_eff=lam*1e-6*u.m)*cmock.pix_sr
        
        auto_poisson_var = np.sum(volfac*(nu_Inu**2))
        auto_cl_pred = pf*(auto_poisson_var.value)    
        cl_matrix[idx, idx] = auto_cl_pred
        
        
    for idx in range(nbands-1):
        for idxc in range(idx+1, nbands):
            
            nu_Inu = cmock.mag_2_nu_Inu(cosmos_cat_sel[:,idx], lam_eff=lams[idx]*1e-6*u.m)*cmock.pix_sr
            nu_Inu_cross = cmock.mag_2_nu_Inu(cosmos_cat_sel[:,idxc], lam_eff=lams[idxc]*1e-6*u.m)*cmock.pix_sr
            
            cross_poisson_var = np.sum(volfac*(nu_Inu*nu_Inu_cross))
            cross_cl_pred = pf*(cross_poisson_var.value)
            cl_matrix[idx, idxc] = cross_cl_pred

    
    return cl_matrix

def make_binned_coverage_mask_and_mkk(posx, posy, catalog_mode, nx=None, ny=None, n_mkk_sim=100, n_split=2, \
									 cl_pred_basepath=None, save=False, addstr=None, hitmin=0, nreg=None):
		
	cbps = CIBER_PS_pipeline(dimx=nx, dimy=ny, n_ps_bin=23)
	cmock = ciber_mock(nx=nx, ny=ny)
	
	if cl_pred_basepath is None:
		cl_pred_basepath = config.ciber_basepath+'data/cl_predictions/'
			
	if nreg is None:
		nreg = nx+1
			
	H, xedges, yedges = np.histogram2d(posx, posy, bins=[np.linspace(0, nx, nreg), np.linspace(0, ny, nreg)])
	
	
	coverage_mask = (H > hitmin)
	print('coverage mask ', np.sum(coverage_mask)/float(coverage_mask.shape[0]*coverage_mask.shape[1]))

	coverage_mask_fullres = np.array(Image.fromarray(coverage_mask).resize((nx, ny)))
	coverage_mask_fullres[coverage_mask_fullres < 0.5] = 0
	coverage_mask_fullres[coverage_mask_fullres >= 0.5] = 1
	
	plt.figure()
	plt.title('Coverage mask (coarse)')
	plt.imshow(coverage_mask, origin='lower')
	plt.colorbar()
	plt.show()
	
	plt.figure()
	plt.title('Coverage mask (upsampled)')
	plt.imshow(coverage_mask_fullres, origin='lower')
	plt.colorbar()
	plt.show()
	
	plt.figure()
	plt.hist(H.ravel(), bins=np.linspace(-0.5, 9.5, 10))
	plt.yscale('log')
	plt.xlabel('Nsrc per pixel')
	plt.show()


	av_Mkk = cbps.Mkk_obj.get_mkk_sim(coverage_mask_fullres, n_mkk_sim, n_split=n_split)
	inv_Mkk = compute_inverse_mkk(av_Mkk)
	
	if save:
		save_fpath = cl_pred_basepath+catalog_mode+'/Mkk_file_'+catalog_mode+'_coverage'
		if addstr is not None:
			save_fpath += '_'+addstr

		save_fpath += '.npz'
		print('saving to ', save_fpath)
		inv_Mkk = save_mkks(save_fpath, av_Mkk=av_Mkk, return_inv_Mkk=True, mask=coverage_mask_fullres, \
						   midbin_ell=cbps.Mkk_obj.midbin_ell, bins=cbps.Mkk_obj.binl)

		return av_Mkk, inv_Mkk, coverage_mask, save_fpath

	return av_Mkk, inv_Mkk, coverage_mask
	
def preprocess_c15_catalog(catalog_fpath=None, radec_cut=True, ramin=149.4, ramax=150.8, decmin=1.5, decmax=2.9, save=False, \
						  binned_coverage_mask_mkk=False, nreg=None, capak_mask=False):

	if catalog_fpath is None:
		catalog_fpath = '../spherex/inhouse_code/data/COSMOS2015_Laigle_v1.1.fits'
		
	c15_cat = fits.open(catalog_fpath)
	
	c15_dat = c15_cat[1].data

	c15_ra = np.array(c15_dat['ALPHA_J2000'])
	c15_dec = np.array(c15_dat['DELTA_J2000'])
	c15_flag_hjmcc = np.array(c15_dat['FLAG_HJMCC']) # in UltraVISTA region (i.e., with NIR photometry)
	c15_flag_capak = np.array(c15_dat['FLAG_PETER'])
	c15_type = np.array(c15_dat['TYPE'])
	
	c15_r_AB = -2.5*np.log10(c15_dat['r_FLUX_APER3'])+23.9
	c15_i_AB = -2.5*np.log10(c15_dat['ip_FLUX_APER3'])+23.9
	c15_z_AB = -2.5*np.log10(c15_dat['zp_FLUX_APER3'])+23.9
	c15_y_AB = -2.5*np.log10(c15_dat['Y_FLUX_APER3'])+23.9
	c15_J_AB = -2.5*np.log10(c15_dat['J_FLUX_APER3'])+23.9
	c15_H_AB = -2.5*np.log10(c15_dat['H_FLUX_APER3'])+23.9
	
	c15_Ks_AB = -2.5*np.log10(c15_dat['Ks_FLUX_APER3'])+23.9
	c15_CH1_AB = -2.5*np.log10(c15_dat['SPLASH_1_FLUX'])+23.9
	c15_CH2_AB = -2.5*np.log10(c15_dat['SPLASH_2_FLUX'])+23.9
	
	c15_redshift = np.array(c15_dat['PHOTOZ'])
	
	deg_2_pix = 3600/7.
	c15_x = (c15_ra-ramin)*deg_2_pix # convert to arcseconds and then CIBER pixels
	c15_y = (c15_dec-decmin)*deg_2_pix # convert to arcseconds and then CIBER pixels
	
	c15_catalog = np.array([c15_ra, c15_dec, c15_x, c15_y, c15_J_AB, \
						   c15_H_AB, c15_CH1_AB, c15_CH2_AB, c15_type, c15_r_AB, c15_i_AB, c15_z_AB, c15_y_AB, c15_redshift, c15_Ks_AB]).transpose()
	
	c15mask = (c15_flag_hjmcc==0) # in UVISTA region
	
	if radec_cut:
		c15mask *= (c15_ra > ramin)*(c15_ra < ramax)*(c15_dec > decmin)*(c15_dec < decmax)
	
	if capak_mask:
		c15mask *= (c15_flag_capak==0)

	c15 = c15_catalog[c15mask,:]
	
	if radec_cut:
		x_min, x_max = np.min(c15[:,2]), np.max(c15[:,2])
		y_min, y_max = np.min(c15[:,3]), np.max(c15[:,3])

		nx = int(x_max - x_min)
		ny = int(y_max - y_min)

		sqdim = min(nx, ny)

		c15[:,2] -= x_min
		c15[:,3] -= y_min

		sqmask = (c15[:,2] < sqdim)*(c15[:,3] < sqdim)
		c15 = c15[sqmask,:]
		
		
	c15_x_sel = c15[:,2]
	c15_y_sel = c15[:,3]

	plt.figure()
	plt.hist(c15[:,4], bins=np.linspace(10, 25, 30))
	plt.yscale('log')
	plt.xlabel('J band magnitude')
	plt.show()

	plt.figure(figsize=(8, 8))
	plt.title('COSMOS 2015', fontsize=14)
	plt.scatter(c15[:,2], c15[:,3], s=2, color='k', alpha=0.02)
	plt.xlabel('x [CIBER pixels]', fontsize=14)
	plt.xlabel('y [CIBER pixels]', fontsize=14)
	plt.grid()
	plt.show()
	
	
	full_df_allbands = pd.DataFrame({'x':c15_x_sel, 'y':c15_y_sel, 'r_AB':c15[:,9], \
									 'i_AB':c15[:,10], 'z_AB':c15[:,11], 'y_AB':c15[:,12],\
									 'J': c15[:,4], 'H':c15[:,5], 'mag_CH1':c15[:,6], 'mag_CH2':c15[:,7], 'type':c15[:,8], 'redshift':c15[:,13], \
									'Ks':c15[:,14]})
	
	if save:
		tailstr = 'COSMOS15_rizy_JHKs_CH1CH2_AB_hjmcc'
		if capak_mask:
			tailstr += '_capak'
		
		save_fpath = config.ciber_basepath+'data/catalogs/COSMOS15/'+tailstr+'.csv'

		print('Saving catalog to ', save_fpath)
		full_df_allbands.to_csv(save_fpath)

		
	if binned_coverage_mask_mkk:
		av_Mkk, inv_Mkk, coverage_mask, save_fpath = make_binned_coverage_mask_and_mkk(c15_x_sel, c15_y_sel, 'COSMOS15',\
																			   nx=sqdim, ny=sqdim, addstr='peter_hjmcc', save=save, nreg=nreg)
	
		
	return full_df_allbands


def predict_auto_cross_cl_C15(m_min_J_list, m_min_H_list, inst=1, include_IRAC_mask=False, maglim_IRAC=18., m_max=28, \
							 inv_Mkk_fpath=None, mkk_correct=True, catalog_fpath=None, cl_pred_basepath=None, \
							 ifield_choose=4, addstr=None, save=True, filter_type=None):
	
	if cl_pred_basepath is None:
		cl_pred_basepath = config.ciber_basepath+'data/cl_predictions/'
	
	if catalog_fpath is None:
		catalog_fpath = config.ciber_basepath+'data/catalogs/COSMOS15/COSMOS15_hjmcc_JH_CH1CH2_AB.csv'
		
	cosmos_catalog = pd.read_csv(catalog_fpath)
	cosmos_xpos = np.array(cosmos_catalog['x'])
	cosmos_ypos = np.array(cosmos_catalog['y'])
	cosmos_J_mag = np.array(cosmos_catalog['J'])
	cosmos_H_mag = np.array(cosmos_catalog['H'])
	cosmos_CH1_mag = np.array(cosmos_catalog['mag_CH1'])
	cosmos_CH2_mag = np.array(cosmos_catalog['mag_CH2'])
	
	if filter_type is not None:
		cosmos_type = np.array(cosmos_catalog['type'])
	
	mag_list = [cosmos_J_mag, cosmos_H_mag, cosmos_CH1_mag, cosmos_CH2_mag]
	
	magstr_list = ['J', 'H', 'CH1', 'CH2']

	lameffs = [1.05, 1.79, 3.6, 4.5]
	
	Vega_to_AB = dict({'g':-0.08, 'r':0.16, 'i':0.37, 'z':0.54, 'y':0.634, 'J':0.91, 'H':1.39, 'K':1.85, \
					  'CH1':2.699, 'CH2':3.339})
	

	mkkobj = np.load(inv_Mkk_fpath)

	inv_Mkk = mkkobj['inverse_Mkk']
	mask = np.array(mkkobj['mask']).astype(int)
	print('mask :', mask)
	plot_map(mask, figsize=(5,5), title='coverage mask')

	npsbin = inv_Mkk.shape[0]
	nx = mask.shape[0]    
	
	print('npsbin, nx', npsbin, nx)

	cmock = ciber_mock(nx=nx, ny=nx)
	cbps = CIBER_PS_pipeline(dimx=nx, dimy=nx, n_ps_bin=npsbin)
	
	n_ps_save = len(m_min_J_list)
	
	subpixel_psf_dirpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/subpixel_psfs/'
	psfbank = np.load(subpixel_psf_dirpath+'psf_temp_bank_inst1_ifield4.npz')
	psf_postage_stamps = psfbank['psf_temp_bank']
	
	lb, mean_bl, bls = cbps.compute_beam_correction_posts(ifield_choose, inst, psf_postage_stamps=psf_postage_stamps)    

	print('mean bl:', mean_bl)
	all_clautos = np.zeros((n_ps_save, len(lameffs), npsbin))
	all_clx_J, all_clx_H, all_rlx_J, all_rlx_H = [np.zeros((n_ps_save, len(lameffs)-1, npsbin)) for x in range(4)]

	all_clautos_unc = np.zeros((n_ps_save, len(lameffs), npsbin))
	all_clx_J_unc, all_clx_H_unc, all_rlx_J_unc, all_rlx_H_unc = [np.zeros((n_ps_save, len(lameffs)-1, npsbin)) for x in range(4)]
	
	
#     all_clx_H = np.zeros((n_ps_save, len(lameffs)-1, npsbin))

	which_crossidx_J = [1, 2, 3]
	which_crossidx_H = [2, 3]
	
#     all_clauto_J, all_clauto_H, all_clauto_CH1, all_clauto_CH2, \
#         all_clx_JH, all_clx_J_CH1, all_clx_J_CH2, all_clx_H_CH1,\
#             all_clx_H_CH2, all_clx_CH1_CH2 = [np.zeros((n_ps_save, npsbin)) for x in range(10)]
	
	
	for magidx in range(len(m_min_J_list)):
		
		src_maps = []

		magmask_J = (cosmos_J_mag-Vega_to_AB['J'] > m_min_J_list[magidx])*(cosmos_J_mag-Vega_to_AB['J'] < m_max)
		magmask_H = (cosmos_H_mag-Vega_to_AB['H'] > m_min_H_list[magidx])*(cosmos_H_mag-Vega_to_AB['H'] < m_max)
		
		magmask = magmask_J*magmask_H

		if include_IRAC_mask:
			print('adding IRAC mask L < '+str(maglim_IRAC))
			magmask *= (cosmos_CH1_mag-Vega_to_AB['CH1'] > maglim_IRAC)
			
		if filter_type is not None:
			
			filtmask = (cosmos_type==filter_type) # 0 for galaxies, 1 for stars
			
			magmask *= filtmask
		# loop through bands
		
		for lidx, lam in enumerate(lameffs):
			I_arr = cmock.mag_2_nu_Inu(mag_list[lidx], band=None, lam_eff=lam*1e-6*u.m)
			I_arr[np.isnan(I_arr)] = 0.
			mock_cat = np.array([cosmos_xpos[magmask], cosmos_ypos[magmask], mag_list[lidx][magmask], I_arr[magmask]]).transpose()
			
			sourcemap = cmock.make_srcmap_temp_bank(ifield_choose, 1, mock_cat, flux_idx=-1, n_fine_bin=10, nwide=17,tempbank_dirpath=subpixel_psf_dirpath, load_precomp_tempbank=True)
			sourcemap *= mask
			
			if magidx==0:
				plot_map(sourcemap, figsize=(6,6), title='source map lam = '+str(lam))
			sourcemap[mask != 0] -= np.mean(sourcemap[mask != 0])
			src_maps.append(sourcemap)

			# auto power spectra
			lb, clauto, clerr_auto = get_power_spec(sourcemap, \
										 lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
			
			cl_proc = np.dot(inv_Mkk.transpose(), clauto)
			clerr_proc = clerr_auto * np.diagonal(inv_Mkk)
			
			cl_proc /= mean_bl**2

			all_clautos[magidx, lidx] = cl_proc
			all_clautos_unc[magidx, lidx] = clerr_proc
			
		
		for lidx, crossidx_J in enumerate(which_crossidx_J):
			# J x {H, CH1, CH2}
			lb, clcross, clerr_cross = get_power_spec(src_maps[0], map_b=src_maps[crossidx_J], \
							 lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)

			clx_proc = np.dot(inv_Mkk.transpose(), clcross)
			clx_proc /= mean_bl**2
			
			clx_err_proc = clerr_cross * np.diagonal(inv_Mkk)
			clx_err_proc /= mean_bl**2
			all_clx_J[magidx, lidx] = clx_proc
			all_clx_J_unc[magidx, lidx] = clx_err_proc
			
			
			
			# cross correlation coefficients
			
			rlx_J = clx_proc / np.sqrt(all_clautos[magidx,0]*all_clautos[magidx, crossidx_J])
			all_rlx_J[magidx, lidx] = rlx_J
				
			rlx_J_unc = compute_rlx_unc_comps(all_clautos[magidx, 0], all_clautos[magidx, crossidx_J], clx_proc, all_clautos_unc[magidx,0], all_clautos_unc[magidx, crossidx_J], clx_err_proc)
			all_rlx_J_unc[magidx, lidx] = rlx_J_unc
			
		for lidx, crossidx_H in enumerate(which_crossidx_H):
			# H x {CH1, CH2}
			lb, clcross, clerr_cross = get_power_spec(src_maps[1], map_b=src_maps[crossidx_H], \
							 lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)

			clx_proc = np.dot(inv_Mkk.transpose(), clcross)
			clx_proc /= mean_bl**2
			
			clx_err_proc = clerr_cross * np.diagonal(inv_Mkk)
			clx_err_proc /= mean_bl**2
			
			all_clx_H[magidx, lidx] = clx_proc
			all_clx_H_unc[magidx, lidx] = clx_err_proc
			
			# cross correlation coefficients
			rlx_H = clx_proc / np.sqrt(all_clautos[magidx,1]*all_clautos[magidx, crossidx_H])
			all_rlx_H[magidx, lidx] = rlx_H
			
			rlx_H_unc = compute_rlx_unc_comps(all_clautos[magidx, 1], all_clautos[magidx, crossidx_H], clx_proc, all_clautos_unc[magidx,1], all_clautos_unc[magidx, crossidx_H], clx_err_proc)
			all_rlx_H_unc[magidx, lidx] = rlx_H_unc
			
			
	pf = lb*(lb+1)/(2*np.pi)
	
	plt.figure(figsize=(8, 6))
	for lidx in range(4):
		plt.subplot(2,2,lidx+1)
		
		for magidx in range(len(m_min_J_list)):
			
			plt.errorbar(lb, pf*all_clautos[magidx, lidx], yerr=pf*all_clautos_unc[magidx, lidx], fmt='o')
			
		plt.xscale('log')
		plt.yscale('log')
		plt.xlabel('$\\ell$')
		plt.ylabel('$D_{\\ell}$')
		plt.grid(alpha=0.5)
			
	plt.tight_layout()
	plt.show()
	
	plt.figure(figsize=(9, 6))
	
	
	for lidx, crossidx_J in enumerate(which_crossidx_J):
		plt.subplot(2, 3, lidx+1)
		plt.title('J $\\times$ '+magstr_list[crossidx_J], fontsize=12)

		for magidx in range(len(m_min_J_list)):
			plt.errorbar(lb, all_rlx_J[magidx, lidx], yerr=all_rlx_J_unc[magidx, lidx], fmt='o', color='C'+str(magidx))
		
		plt.xscale('log')
		plt.ylim(-0.2, 1.1)
	
	
	for lidx, crossidx_H in enumerate(which_crossidx_H):
		
		plt.subplot(2,3, 4+lidx)
		
		plt.title('H $\\times$ '+magstr_list[crossidx_H], fontsize=12)
		for magidx in range(len(m_min_J_list)):
#             plt.plot(lb, all_rlx_H[magidx, lidx], color='C'+str(magidx))
			plt.errorbar(lb, all_rlx_H[magidx, lidx], yerr=all_rlx_H_unc[magidx, lidx], fmt='o', color='C'+str(magidx))

		plt.xscale('log')
		plt.ylim(-0.2, 1.1)
	
	plt.tight_layout()
	plt.show()
	
	if save:
		
		save_fpath_auto = cl_pred_basepath+'COSMOS15/cl_predictions_auto_JH_CH1CH2'
		save_fpath_cross_J = cl_pred_basepath+'COSMOS15/cl_predictions_cross_J_HCH1CH2'
		save_fpath_cross_H = cl_pred_basepath+'COSMOS15/cl_predictions_cross_H_CH1CH2'
		
		if addstr is not None:
			save_fpath_auto += '_'+addstr
			save_fpath_cross_J += '_'+addstr
			save_fpath_cross_H += '_'+addstr

		save_fpath_auto += '.npz'
		save_fpath_cross_J += '.npz'
		save_fpath_cross_H += '.npz'

		print('saving auto spectra to ', save_fpath_auto)
		
		np.savez(save_fpath_auto, m_min_J_list=m_min_J_list, m_min_H_list=m_min_H_list, include_IRAC_mask=include_IRAC_mask, \
				maglim_IRAC=maglim_IRAC, all_clautos=all_clautos, all_clautos_unc=all_clautos_unc, lameffs=lameffs, lb=lb, pf=pf)
		
		print('saving J x spectra to ', save_fpath_cross_J)

		np.savez(save_fpath_cross_J, m_min_J_list=m_min_J_list, m_min_H_list=m_min_H_list, include_IRAC_mask=include_IRAC_mask, \
			maglim_IRAC=maglim_IRAC, all_clx_J=all_clx_J, all_rlx_J=all_rlx_J, all_clx_J_unc=all_clx_J_unc, all_rlx_J_unc=all_rlx_J_unc, cross_idxs=crossidx_J, lameffs=lameffs, lb=lb, pf=pf)

		print('saving H x spectra to ', save_fpath_cross_H)

		np.savez(save_fpath_cross_H, m_min_J_list=m_min_J_list, m_min_H_list=m_min_H_list, include_IRAC_mask=include_IRAC_mask, \
			maglim_IRAC=maglim_IRAC, all_clx_H=all_clx_H, all_rlx_H=all_rlx_H, all_clx_H_unc=all_clx_H_unc, all_rlx_H_unc=all_rlx_H_unc, cross_idxs=crossidx_H, lameffs=lameffs, lb=lb, pf=pf)

	
	return all_clautos, all_clx_J, all_clx_H, lameffs, lb, pf 


def ukidss_uds_cl_predict_JH(m_min_J_list, m_min_H_list, dimx=450, m_max = 28, psf_inst=1, ifield_choose=4, \
							skipidx = 9, plot=True, cl_pred_basepath=config.ciber_basepath+'data/cl_predictions/', \
							add_str=None):
	

	subpixel_psf_dirpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(psf_inst)+'/subpixel_psfs/'
	catalog_basepath = config.ciber_basepath+'data/catalogs/'
	ukidss_uds = pd.read_csv(catalog_basepath+'UKIDSS/ukidss_dr11_plus_UDS_12_7_20.csv')
	Vega_to_AB = dict({'g':-0.08, 'r':0.16, 'i':0.37, 'z':0.54, 'y':0.634, 'J':0.91, 'H':1.39, 'K':1.85})
	
	ukidss_ra_pd = np.array(ukidss_uds['# ra'])[skipidx:].astype(np.float)
	ukidss_dec_pd = np.array(ukidss_uds['dec'])[skipidx:].astype(np.float)
	
	ukidss_J = np.array(ukidss_uds['jAB'])[skipidx:].astype(np.float)
	ukidss_H = np.array(ukidss_uds['hAB'])[skipidx:].astype(np.float)
	ukidss_K = np.array(ukidss_uds['kAB'])[skipidx:].astype(np.float)
	
	cmock = ciber_mock(nx=dimx, ny=dimx)
	
	ramin, ramax = np.min(ukidss_ra_pd), np.max(ukidss_ra_pd)
	decmin, decmax = np.min(ukidss_dec_pd), np.max(ukidss_dec_pd)
	
	npixsize_x = (ramax-ramin)*3600//cbps.pixsize
	npixsize_y = (decmax-decmin)*3600//cbps.pixsize

	cbps_uds = CIBER_PS_pipeline(dimx=dimx, dimy=dimx, n_ps_bin=23)
	lb, mean_bl, bls = cbps_uds.compute_beam_correction_posts(ifield_choose, psf_inst)    

	ukidss_xpos = (ukidss_ra_pd-ramin)*3600/cbps.pixsize
	ukidss_ypos = (ukidss_dec_pd-decmin)*3600/cbps.pixsize

	ukidss_bordermask = (ukidss_xpos < dimx)*(ukidss_ypos < dimx)
	
	r_ell_vs_mag, cl_auto_J_vs_mag, cl_auto_H_vs_mag, clcross_vs_mag = [], [], [], []

	for magidx in range(len(m_min_J_list)):

		ukidss_magmask_J = (ukidss_J-Vega_to_AB['J'] > m_min_J_list[magidx])*(ukidss_J-Vega_to_AB['J'] < m_max)
		ukidss_magmask_H = (ukidss_H-Vega_to_AB['H'] > m_min_H_list[magidx])*(ukidss_H-Vega_to_AB['H'] < m_max)
		ukidss_magmask = ukidss_magmask_J*ukidss_magmask_H
		ukidss_mask = ukidss_bordermask*ukidss_magmask
		
		I_arr_J = cmock.mag_2_nu_Inu(ukidss_J, 0)
		I_arr_H = cmock.mag_2_nu_Inu(ukidss_H, 1)

		mock_uds_cat_J = np.array([ukidss_xpos[ukidss_mask], ukidss_ypos[ukidss_mask], ukidss_J[ukidss_mask], I_arr_J[ukidss_mask]]).transpose()
		mock_uds_cat_H = np.array([ukidss_xpos[ukidss_mask], ukidss_ypos[ukidss_mask], ukidss_H[ukidss_mask], I_arr_H[ukidss_mask]]).transpose()

		sourcemap_uds_J = cmock.make_srcmap_temp_bank(ifield_choose, psf_inst, mock_uds_cat_J, flux_idx=-1, n_fine_bin=10, nwide=17,tempbank_dirpath=subpixel_psf_dirpath, load_precomp_tempbank=True)
		sourcemap_uds_H = cmock.make_srcmap_temp_bank(ifield_choose, psf_inst, mock_uds_cat_H, flux_idx=-1, n_fine_bin=10, nwide=17,tempbank_dirpath=subpixel_psf_dirpath, load_precomp_tempbank=True)
		
		if plot:
			plot_map(sourcemap_uds_J, title='J')
			plot_map(sourcemap_uds_H, title='H')

		lb, clcross, clerr_cross = get_power_spec(sourcemap_uds_J-np.mean(sourcemap_uds_J), map_b=sourcemap_uds_H-np.mean(sourcemap_uds_H), \
												 lbinedges=cbps_uds.Mkk_obj.binl, lbins=cbps_uds.Mkk_obj.midbin_ell)

		lb, clauto_J, clerr_auto_J = get_power_spec(sourcemap_uds_J-np.mean(sourcemap_uds_J), \
												 lbinedges=cbps_uds.Mkk_obj.binl, lbins=cbps_uds.Mkk_obj.midbin_ell)

		lb, clauto_H, clerr_auto_H = get_power_spec(sourcemap_uds_H-np.mean(sourcemap_uds_H), \
												 lbinedges=cbps_uds.Mkk_obj.binl, lbins=cbps_uds.Mkk_obj.midbin_ell)

		clcross /= mean_bl**2
		clauto_J /= mean_bl**2
		clauto_H /= mean_bl**2
		
		if plot:
			prefac = lb*(lb+1)/(2*np.pi)

			plt.figure()
			plt.plot(lb, prefac*clcross, label='clcross')
			plt.plot(lb, prefac*clauto_J, label='cl auto J')
			plt.plot(lb, prefac*clauto_H, label='cl auto H')
			plt.legend()

			plt.xscale('log')
			plt.yscale('log')
			plt.xlabel('$\\ell$', fontsize=14)
			plt.ylabel('$D_{\\ell}$', fontsize=14)
			plt.show()
		
		clcross_vs_mag.append(clcross)
		cl_auto_J_vs_mag.append(clauto_J)
		cl_auto_H_vs_mag.append(clauto_H)

	if save:
		cl_save_fpath = cl_pred_basepath+'ukidss_uds_cl_prediction_vs_mag_auto_cross_JH'

		if add_str is not None:
			cl_save_fpath += '_'+add_str

		print("Saving cls to ", cl_save_fpath)

		np.savez(cl_save_fpath+'.npz', lb=lb, \
			m_min_J_list = m_min_J_list, m_min_H_list = m_min_H_list, cl_auto_J_vs_mag=cl_auto_J_vs_mag, \
			cl_auto_H_vs_mag=cl_auto_H_vs_mag, clcross_vs_mag=clcross_vs_mag)

	return lb, cl_auto_J_vs_mag, cl_auto_H_vs_mag, clcross_vs_mag


def simulate_deep_cats_correlation_coeff_cosmos(m_min_J_list, m_min_H_list, inst=1, ifield_choose = 4, include_IRAC_mask=False, maglim_IRAC=18., m_max=28, \
											   inv_Mkk=None, mkk_correct=True, coverage_mask=None, m_min_CH1_list=[16.0, 17.0, 18.0, 19.0], \
											   m_min_CH2_list=None, catalog_fpath=None, startidx=1, endidx=-1, nx=650, ny=650, \
											   cl_pred_basepath=config.ciber_basepath+'data/cl_predictions/'):

	Vega_to_AB = dict({'g':-0.08, 'r':0.16, 'i':0.37, 'z':0.54, 'y':0.634, 'J':0.91, 'H':1.39, 'K':1.85, \
					  'CH1':2.699, 'CH2':3.339})
	
	if catalog_fpath is None:
		catalog_fpath = 'data/cosmos/cosmos20_farmer_catalog_wxy_073123.npz'
	   
	cbps = CIBER_PS_pipeline(dimx=nx, dimy=ny)
	
	bls_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/beam_correction/bl_est_postage_stamps_TM'+str(inst)+'_081121.npz'
	B_ells = np.load(bls_fpath)['B_ells_post']
	
	mean_bl = B_ells[0]
	
	print('mean bl:', mean_bl)

	subpixel_psf_dirpath = config.exthdpath+'ciber_fluctuation_data/TM'+str(inst)+'/subpixel_psfs/'
	lb, mean_bl, bls = cbps.compute_beam_correction_posts(ifield_choose, inst)    
	
	cmock = ciber_mock(nx=nx, ny=ny)
	
	if m_min_CH2_list is None:
		m_min_CH2_list = m_min_CH1_list

	cosmos_catalog = np.load(catalog_fpath)

	# these are all AB mags from Farmer catalog
	cosmos_xpos = cosmos_catalog['cosmos_xpos']
	cosmos_ypos = cosmos_catalog['cosmos_ypos']
	cosmos_J_mag = cosmos_catalog['cosmos_J_mag']
	cosmos_H_mag = cosmos_catalog['cosmos_H_mag']
	cosmos_CH1_mag = cosmos_catalog['cosmos_CH1_mag']
	cosmos_CH2_mag = cosmos_catalog['cosmos_CH2_mag']
	
	magbins = np.linspace(13, 26, 30)
	plt.figure()
	plt.hist(cosmos_J_mag, bins=magbins, histtype='step', label='J')
	plt.hist(cosmos_H_mag, bins=magbins, histtype='step', label='H')
	plt.hist(cosmos_CH1_mag, bins=magbins, histtype='step', label='CH1')
	plt.hist(cosmos_CH2_mag, bins=magbins, histtype='step', label='CH2')
	plt.yscale('log')
	plt.legend()
	plt.show()

	cosmos_bordermask = (cosmos_xpos < nx)*(cosmos_ypos < ny)*(cosmos_xpos > 0)*(cosmos_ypos > 0)
	
	if inv_Mkk is None and mkk_correct:
		H, xedges, yedges = np.histogram2d(cosmos_xpos, cosmos_ypos, bins=[np.linspace(0, nx, nx+1), np.linspace(0, ny, ny+1)])

		coverage_mask = (H != 0)

		H_mask = H*coverage_mask
		plt.figure()
		plt.imshow(H_mask, origin='lower')
		plt.colorbar()
		plt.show()

		print('coverage_mask has shape ', coverage_mask.shape)
		plt.figure()
		plt.imshow(coverage_mask, origin='lower')
		plt.colorbar()
		plt.show()
		
		av_Mkk = cbps.Mkk_obj.get_mkk_sim(coverage_mask, 100, n_split=1)
		
		inv_Mkk = save_mkks(cl_pred_basepath+'cosmos/Mkk_file_C20_coverage.npz', av_Mkk=av_Mkk, return_inv_Mkk=True, mask=coverage_mask)
	
	
	all_clauto_J, all_clauto_H, all_clauto_CH1, all_clauto_CH2 = [[] for x in range(4)]
	all_clx_JH, all_clx_J_CH1, all_clx_J_CH2, all_clx_H_CH1, all_clx_H_CH2, all_clx_CH1_CH2 = [[] for x in range(6)]
	

	for magidx in range(len(m_min_J_list)):

		magmask_J = (cosmos_J_mag-Vega_to_AB['J'] > m_min_J_list[magidx])*(cosmos_J_mag-Vega_to_AB['J'] < m_max)
		magmask_H = (cosmos_H_mag-Vega_to_AB['H'] > m_min_H_list[magidx])*(cosmos_H_mag-Vega_to_AB['H'] < m_max)
		magmask = magmask_J*magmask_H
		
		if include_IRAC_mask:
			print('adding IRAC mask L < '+str(maglim_IRAC))
			magmask *= (cosmos_CH1_mag-Vega_to_AB['CH1'] > maglim_IRAC)
		
		mask_J = cosmos_bordermask*magmask
		mask_H = cosmos_bordermask*magmask
		mask = cosmos_bordermask*magmask
		
		I_arr_J = cmock.mag_2_nu_Inu(cosmos_J_mag, band=None, lam_eff=1.05*1e-6*u.m)
		I_arr_H = cmock.mag_2_nu_Inu(cosmos_H_mag, band=None, lam_eff=1.79*1e-6*u.m)
		I_arr_CH1 = cmock.mag_2_nu_Inu(cosmos_CH1_mag, band=None, lam_eff=3.6*1e-6*u.m)
		I_arr_CH2 = cmock.mag_2_nu_Inu(cosmos_CH2_mag, band=None, lam_eff=4.5*1e-6*u.m)
		

		I_arr_J[np.isnan(I_arr_J)] = 0.
		I_arr_H[np.isnan(I_arr_H)] = 0.
		I_arr_CH1[np.isnan(I_arr_CH1)] = 0.
		I_arr_CH2[np.isnan(I_arr_CH2)] = 0.
		
		mock_cat_J = np.array([cosmos_xpos[mask_J], cosmos_ypos[mask_J], cosmos_J_mag[mask_J], I_arr_J[mask_J]]).transpose()
		mock_cat_H = np.array([cosmos_xpos[mask_H], cosmos_ypos[mask_H], cosmos_H_mag[mask_H], I_arr_H[mask_H]]).transpose()
		mock_cat_CH1 = np.array([cosmos_xpos[mask], cosmos_ypos[mask], cosmos_CH1_mag[mask], I_arr_CH1[mask]]).transpose()
		mock_cat_CH2 = np.array([cosmos_xpos[mask], cosmos_ypos[mask], cosmos_CH2_mag[mask], I_arr_CH2[mask]]).transpose()
		
		sourcemap_J = cmock.make_srcmap_temp_bank(ifield_choose, inst, mock_cat_J, flux_idx=-1, n_fine_bin=10, nwide=17,tempbank_dirpath=subpixel_psf_dirpath, load_precomp_tempbank=True)
		sourcemap_H = cmock.make_srcmap_temp_bank(ifield_choose, inst, mock_cat_H, flux_idx=-1, n_fine_bin=10, nwide=17,tempbank_dirpath=subpixel_psf_dirpath, load_precomp_tempbank=True)
		sourcemap_CH1 = cmock.make_srcmap_temp_bank(ifield_choose, inst, mock_cat_CH1, flux_idx=-1, n_fine_bin=10, nwide=17,tempbank_dirpath=subpixel_psf_dirpath, load_precomp_tempbank=True)
		sourcemap_CH2 = cmock.make_srcmap_temp_bank(ifield_choose, inst, mock_cat_CH2, flux_idx=-1, n_fine_bin=10, nwide=17,tempbank_dirpath=subpixel_psf_dirpath, load_precomp_tempbank=True)        
		
		sourcemap_J *= coverage_mask
		sourcemap_H *= coverage_mask
		sourcemap_CH1 *= coverage_mask
		sourcemap_CH2 *= coverage_mask
		
		plot_map(sourcemap_J, title='COSMOS $J$ band map (VISTA filter)')
		
		sourcemap_J[coverage_mask != 0] -= np.mean(sourcemap_J[coverage_mask != 0])
		sourcemap_H[coverage_mask != 0] -= np.mean(sourcemap_H[coverage_mask != 0])
		sourcemap_CH1[coverage_mask != 0] -= np.mean(sourcemap_CH1[coverage_mask != 0])
		sourcemap_CH2[coverage_mask != 0] -= np.mean(sourcemap_CH2[coverage_mask != 0])
		
		# autos
		lb, clauto_J, clerr_auto_J = get_power_spec(sourcemap_J, \
												 lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
		lb, clauto_H, clerr_auto_H = get_power_spec(sourcemap_H, \
												 lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
		lb, clauto_CH1, clerr_auto_CH1 = get_power_spec(sourcemap_CH1, \
												 lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
		lb, clauto_CH2, clerr_auto_CH2 = get_power_spec(sourcemap_CH2, \
												 lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
		
		cl_autos = [clauto_J, clauto_H, clauto_CH1, clauto_CH2]
		
		# now compute crosses
		lb, clx_J_H, clerr_x_J_H = get_power_spec(sourcemap_J, map_b=sourcemap_H, \
								 lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
		lb, clx_J_CH1, clerr_x_J_CH1 = get_power_spec(sourcemap_J, map_b=sourcemap_CH1, \
								 lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
		lb, clx_J_CH2, clerr_x_J_CH2 = get_power_spec(sourcemap_J, map_b=sourcemap_CH2, \
								 lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
		lb, clx_H_CH1, clerr_x_H_CH1 = get_power_spec(sourcemap_H, map_b=sourcemap_CH1, \
						 lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
		lb, clx_H_CH2, clerr_x_H_CH2 = get_power_spec(sourcemap_H, map_b=sourcemap_CH2, \
				 lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
		lb, clx_CH1_CH2, clerr_x_CH1_CH2 = get_power_spec(sourcemap_CH1, map_b=sourcemap_CH2, \
				 lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
		
		cl_crosses = [clx_J_H, clx_J_CH1, clx_J_CH2, clx_H_CH1, clx_H_CH2, clx_CH1_CH2]

		for clidx, cl in enumerate(cl_autos):
			if mkk_correct:
				cl_autos[clidx] = np.dot(inv_Mkk.transpose(), cl)
			cl_autos[clidx] = cl_autos[clidx] / mean_bl**2
			
			
		for clidx, clx in enumerate(cl_crosses):
			if mkk_correct:
				cl_crosses[clidx] = np.dot(inv_Mkk.transpose(), clx)
			cl_crosses[clidx] = cl_crosses[clidx] / mean_bl**2
	
	
		r_ell_J_H = cl_crosses[0]/np.sqrt(cl_autos[0]*cl_autos[1])
		r_ell_J_CH1 = cl_crosses[1]/np.sqrt(cl_autos[0]*cl_autos[2])
		r_ell_J_CH2 = cl_crosses[2]/np.sqrt(cl_autos[0]*cl_autos[3])
		
		r_ell_H_CH1 = cl_crosses[3]/np.sqrt(cl_autos[1]*cl_autos[2])
		r_ell_H_CH2 = cl_crosses[4]/np.sqrt(cl_autos[1]*cl_autos[3])
		r_ell_CH1_CH2 = cl_crosses[5]/np.sqrt(cl_autos[2]*cl_autos[3])
		
		plt.figure()
		plt.plot(lb[startidx:endidx], r_ell_J_H[startidx:endidx], label='J x H', color='C5')
		plt.plot(lb[startidx:endidx], r_ell_J_CH1[startidx:endidx], label='J x CH1', color='C6')
		plt.plot(lb[startidx:endidx], r_ell_J_CH2[startidx:endidx], label='J x CH2', color='C7')
		plt.plot(lb[startidx:endidx], r_ell_H_CH1[startidx:endidx], label='H x CH1', color='C8')
		plt.plot(lb[startidx:endidx], r_ell_H_CH2[startidx:endidx], label='H x CH2', color='C9')
		plt.plot(lb[startidx:endidx], r_ell_CH1_CH2[startidx:endidx], label='CH1 x CH2', color='C10')
		plt.xscale('log')
		plt.ylim(0, 1.3)
		plt.xlabel('$\\ell$', fontsize=16)
		plt.ylabel('$r_{\\ell}$', fontsize=16)
		plt.legend(fontsize=12, ncol=2, loc=3)
		textstr = 'COSMOS field\nMask $J<$'+str(m_min_J_list[magidx])+' and $H<$'+str(m_min_H_list[magidx])
		if include_IRAC_mask:
			textstr += ' $\\cup$ $L<$'+str(maglim_IRAC)
		plt.text(400, 1.05, textstr, color='k', fontsize=16)
		plt.grid()
#         if include_IRAC_mask:
#             plt.savefig('/Users/richardfeder/Downloads/r_ell_cosmos_JHCH1CH2_mask_Jlt'+str(m_min_J_list[magidx])+'_Hlt'+str(m_min_H_list[magidx])+'_CH1lt'+str(maglim_IRAC)+'.png', bbox_inches='tight', dpi=200)
#         else:
#             plt.savefig('/Users/richardfeder/Downloads/r_ell_cosmos_JHCH1CH2_mask_Jlt'+str(m_min_J_list[magidx])+'_Hlt'+str(m_min_H_list[magidx])+'.png', bbox_inches='tight', dpi=200)
		plt.show()
		
		prefac = lb*(lb+1)/(2*np.pi)
		
		plt.figure()
		plt.text(300, 600, 'COSMOS2020 catalog source map \nMask $J<$'+str(m_min_J_list[magidx])+' and $H<$'+str(m_min_H_list[magidx]), fontsize=16)
		plt.plot(lb, prefac*cl_autos[0], label='$J$ auto')
		plt.plot(lb, prefac*cl_autos[1], label='$H$ auto')
		plt.plot(lb, prefac*cl_crosses[0], label='$J\\times H$ cross')
		plt.plot(lb, prefac*cl_autos[2], label='CH1 auto')
		plt.plot(lb, prefac*cl_autos[3], label='CH2 auto')
		plt.legend(loc=3, ncol=2, fontsize=12)
		plt.xscale('log')
		plt.yscale('log')
		plt.tick_params(labelsize=14)
		plt.xlabel('$\\ell$', fontsize=14)
		plt.ylabel('$D_{\\ell}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=14)
		plt.ylim(1e-2, 4e3)
		plt.grid()
#         plt.savefig('/Users/richardfeder/Downloads/cosmos2020_Jlt'+str(m_min_J_list[magidx])+'_Hlt'+str(m_min_H_list[magidx])+'_081423.png', bbox_inches='tight')
		plt.show()
		
		all_clauto_J.append(cl_autos[0])
		all_clauto_H.append(cl_autos[1])
		all_clauto_CH1.append(cl_autos[2])
		all_clauto_CH2.append(cl_autos[3])
		
		all_clx_JH.append(cl_crosses[0])
		all_clx_J_CH1.append(cl_crosses[1])
		all_clx_J_CH2.append(cl_crosses[2])
		all_clx_H_CH1.append(cl_crosses[3])
		all_clx_H_CH2.append(cl_crosses[4])
		all_clx_CH1_CH2.append(cl_crosses[5])
		
	return lb, cl_autos, cl_crosses, all_clauto_J, all_clauto_H, all_clx_JH, all_clauto_CH1, all_clauto_CH2, \
			all_clx_J_CH1, all_clx_J_CH2, all_clx_H_CH1, all_clx_H_CH2
		

def cl_predictions_vs_magcut(inst, ifield_list=[4, 5, 6, 7, 8], mag_lims=None, mag_cut_cat = 15.0, nsim=10, ifield_choose=4, \
							load_igl_pred=False):
	
	if mag_lims is None:
		if inst==1:
			mag_lims = [13.0, 14.0, 15.0, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5]
		elif inst==2:
			mag_lims = [14.0, 15.0, 16.0, 16.5, 17.0, 17.5, 18.0]
			
	magkey_dict = dict({1:'j_m', 2:'h_m'})

	cmock = ciber_mock()
	cbps_nm = CIBER_NoiseModel()
	
	config_dict, pscb_dict, float_param_dict, fpath_dict = return_default_cbps_dicts()
	
	ciber_mock_fpath = config.ciber_basepath+'data/ciber_mocks/'
	fpath_dict, list_of_dirpaths, base_path, trilegal_base_path = set_up_filepaths_cbps(fpath_dict, inst, 'test', '112022',\
																					datestr_trilegal='112022', data_type='observed', \
																				   save_fpaths=True)
	

	all_cls_add = np.zeros((len(mag_lims), len(cbps_nm.cbps.Mkk_obj.midbin_ell)))
	base_fluc_path = config.ciber_basepath+'data/'
	tempbank_dirpath = base_fluc_path+'fluctuation_data/TM'+str(inst)+'/subpixel_psfs/'
	catalog_basepath = base_fluc_path+'catalogs/'
	bls_fpath = base_fluc_path+'fluctuation_data/TM'+str(inst)+'/beam_correction/bl_est_postage_stamps_TM'+str(inst)+'_081121.npz'
	B_ells = np.load(bls_fpath)['B_ells_post']
	all_twomass_cats = []
	
	for fieldidx, ifield in enumerate(ifield_list):
		field_name = cbps_nm.cbps.ciber_field_dict[ifield]
		twomass_cat = pd.read_csv(catalog_basepath+'2MASS/filt/2MASS_filt_rdflag_wxy_'+field_name+'_Jlt17.5.csv')
		all_twomass_cats.append(twomass_cat)

	
	power_maglim_isl_igl, power_maglim_igl = [], []
	
	for magidx, mag_lim in enumerate(mag_lims):
		
		if mag_lim <= mag_cut_cat:

			all_maps, all_masks = [], []

			for fieldidx, ifield in enumerate(ifield_list):
				
				maskInst_fpath = base_fluc_path+'fluctuation_data/TM'+str(inst)+'/masks/maskInst_102422/field'+str(ifield)+'_TM'+str(inst)+'_maskInst_102422.fits'
				mask_inst = fits.open(maskInst_fpath)['maskInst'].data # 10/24/22

				twomass_x = np.array(all_twomass_cats[fieldidx]['x'+str(inst)])
				twomass_y = np.array(all_twomass_cats[fieldidx]['y'+str(inst)])
				twomass_mag = np.array(all_twomass_cats[fieldidx][magkey_dict[inst]]) # magkey                
				twomass_magcut = (twomass_mag < mag_cut_cat)*(twomass_mag > mag_lim)

				twomass_x_sel = twomass_x[twomass_magcut]
				twomass_y_sel = twomass_y[twomass_magcut]
				twomass_mag_sel = twomass_mag[twomass_magcut]

				# convert back to AB mag for generating map 
				twomass_mag_sel_AB = twomass_mag_sel+cmock.Vega_to_AB[inst]
				I_arr_full = cmock.mag_2_nu_Inu(twomass_mag_sel_AB, inst-1)
				full_tracer_cat = np.array([twomass_x_sel, twomass_y_sel, twomass_mag_sel_AB, I_arr_full])

				bright_src_map = cmock.make_srcmap_temp_bank(ifield, inst, full_tracer_cat.transpose(), flux_idx=-1, load_precomp_tempbank=True, \
															tempbank_dirpath=tempbank_dirpath)

				all_maps.append(bright_src_map)
				all_masks.append(mask_inst)
				
			cls_indiv = []
			for idx, indiv in enumerate(all_maps):
				lb, clb, clerr = get_power_spec(all_maps[idx]-np.mean(all_maps[idx]), lbinedges=cbps_nm.cbps.Mkk_obj.binl, lbins=cbps_nm.cbps.Mkk_obj.midbin_ell)
				cls_indiv.append(clb/B_ells[idx]**2)
				
			av_clb = np.mean(np.array(cls_indiv), axis=0)
			all_cls_add[magidx] = av_clb
			
		all_cl, all_cl_igl = [], []
		
		if load_igl_pred:
			meas_pred = np.load('data/poisson_gal_TM'+str(inst)+'_meas_vs_helgason.npz')
			meas_pred_mags = meas_pred['mag_mins']
			poissonvars_meas = meas_pred['poissonvars_meas']
			print('maglim is ', mag_lim)
			whichmatch = np.where((np.array(meas_pred_mags)==mag_lim))[0]
			print('which match :', whichmatch, meas_pred_mags)
			poissonvar_choose = poissonvars_meas[whichmatch]
			
		for cib_setidx in range(nsim):
			perfield_cl = []
			for fieldidx, ifield in enumerate(ifield_list):
				cib_cl_file = fits.open(fpath_dict['cib_resid_ps_path']+'/cls_cib_vs_maglim_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_Vega_magcut.fits')
				isl_cl_file = fits.open(fpath_dict['isl_resid_ps_path']+'/cls_isl_vs_maglim_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_Vega_magcut.fits')

				if mag_lim <= mag_cut_cat:
					perfield_cl.append(cib_cl_file['cls_cib'].data['cl_maglim_'+str(mag_cut_cat)] + isl_cl_file['cls_isl'].data['cl_maglim_'+str(mag_cut_cat)] + cls_indiv[fieldidx])
				else:
					
					if load_igl_pred:
						igl_isl = isl_cl_file['cls_isl'].data['cl_maglim_'+str(mag_lim)]
						igl_isl += poissonvar_choose
						
						perfield_cl.append(igl_isl)
					else:
						perfield_cl.append(cib_cl_file['cls_cib'].data['cl_maglim_'+str(mag_lim)] + isl_cl_file['cls_isl'].data['cl_maglim_'+str(mag_lim)])

			perfield_cl = np.array(perfield_cl)
			all_cl.append(np.mean(perfield_cl, axis=0))
				
#             cib_cl_file_magcutcat = fits.open(fpath_dict['cib_resid_ps_path']+'/cls_cib_vs_maglim_ifield'+str(ifield_choose)+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_Vega_magcut.fits')
#             isl_cl_file_magcutcat = fits.open(fpath_dict['isl_resid_ps_path']+'/cls_isl_vs_maglim_ifield'+str(ifield_choose)+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_Vega_magcut.fits')
#             if mag_lim <= mag_cut_cat:
#                 all_cl.append(cib_cl_file['cls_cib'].data['cl_maglim_'+str(mag_cut_cat)] + isl_cl_file['cls_isl'].data['cl_maglim_'+str(mag_cut_cat)] + av_clb)
#             else:
#                 all_cl.append(cib_cl_file['cls_cib'].data['cl_maglim_'+str(mag_lim)] + isl_cl_file['cls_isl'].data['cl_maglim_'+str(mag_lim)])
			
			all_cl_igl.append(cib_cl_file['cls_cib'].data['cl_maglim_'+str(mag_lim)])
			
		power_maglim_isl_igl.append(np.mean(all_cl, axis=0))
		power_maglim_igl.append(np.mean(all_cl_igl, axis=0))
			
			
	return power_maglim_isl_igl, power_maglim_igl

def compute_cross_shot_noise_trilegal(mag_lim_list, inst, cross_inst, ifield_list, datestr, fpath_dict, mode='isl', mag_lim_list_cross=None, \
									 simidx0=0, nsims=100, ifield_plot=4, save=True, ciberdir='.', ciber_mock_fpath=None, convert_Vega_to_AB=True, simstr=None):
	
	trilegal_band_dict = dict({1:'j_m', 2:'h_m'})
	trilegal_fnu_dict = dict({1:'j_nuInu', 2:'h_nuInu'})
	
	trilegal_base_path = fpath_dict['trilegal_base_path']    
	base_fluc_path = config.exthdpath+'ciber_fluctuation_data/'
	tempbank_dirpath = base_fluc_path+'/TM'+str(inst)+'/subpixel_psfs/'
	tempbank_cross_dirpath = base_fluc_path+'/TM'+str(cross_inst)+'/subpixel_psfs/'
	bls_fpath = fpath_dict['bls_base_path']+'/bl_est_postage_stamps_TM'+str(inst)+'_081121.npz'
	bls_cross_fpath = base_fluc_path+'TM'+str(cross_inst)+'/beam_correction/bl_est_postage_stamps_TM'+str(cross_inst)+'_081121.npz'

	bandstr, Iarr_str = trilegal_band_dict[inst], trilegal_fnu_dict[inst]
	bandstr_cross, Iarr_str_cross = trilegal_band_dict[cross_inst], trilegal_fnu_dict[cross_inst]

	print(bandstr, Iarr_str, bandstr_cross, Iarr_str_cross)
	
	if ciber_mock_fpath is None:
		ciber_mock_fpath = config.exthdpath+'ciber_mocks/'
		
	powerspec_truth_fpath = ciber_mock_fpath+datestr+'/TM1_TM2_cross/'+mode+'_resid_ps/'

	if simstr is not None:
		powerspec_truth_fpath += simstr+'/'
	
	cbps = CIBER_PS_pipeline()
	cmock = ciber_mock(ciberdir=ciberdir)

	isl_sb_rms = np.zeros((nsims, len(ifield_list), len(mag_lim_list)))

	B_ells = np.load(bls_fpath)['B_ells_post']
	B_ells_cross = np.load(bls_cross_fpath)['B_ells_post']

	for setidx in np.arange(simidx0, nsims):

		for fieldidx, ifield in enumerate(ifield_list):

			if setidx==0 and fieldidx==0:
				plt.figure()
				plt.loglog(cbps.Mkk_obj.midbin_ell, B_ells[fieldidx]**2, label='TM'+str(inst))
				plt.loglog(cbps.Mkk_obj.midbin_ell, B_ells_cross[fieldidx]**2, label='TM'+str(inst))
				plt.legend()
				plt.show()
				
			mock_trilegal = fits.open(trilegal_base_path+'/mock_trilegal_simidx'+str(setidx)+'_'+datestr+'.fits')
			full_src_map = mock_trilegal['trilegal_'+str(cbps.inst_to_band[inst])+'_'+str(ifield)].data
			full_src_map_cross = mock_trilegal['trilegal_'+str(cbps.inst_to_band[cross_inst])+'_'+str(ifield)].data
			
			mock_trilegal_cat = mock_trilegal['tracer_cat_'+str(ifield)].data
			
			full_tracer_cat = np.array([mock_trilegal_cat[key] for key in ['x', 'y', bandstr, Iarr_str, bandstr_cross, Iarr_str_cross]])

			for m, mag_lim in enumerate(mag_lim_list):   
								
				mag_lim_AB = mag_lim.copy()
				name = 'cl_maglim_'+str(mag_lim)

				if mag_lim_list_cross is not None:
					mag_lim_AB_cross = mag_lim_list_cross[m].copy()
					name += '_'+str(mag_lim_list_cross[m])
				
				if convert_Vega_to_AB:
					mag_lim_AB += cmock.Vega_to_AB[inst]
					if mag_lim_list_cross is not None:
						mag_lim_AB_cross += cmock.Vega_to_AB[cross_inst]
					if m==0:
						print('converting mag limit threshold from Vega to AB to match mock catalog..')
				if setidx==0:
					print('AB mag lim for '+bandstr+' is ', mag_lim_AB)

				bright_mask = (full_tracer_cat[2,:] < mag_lim_AB)
				if mag_lim_list_cross is not None:
					bright_mask *= (full_tracer_cat[4,:] < mag_lim_AB_cross)
					
				bright_cut = np.where(bright_mask)[0]
				cut_cat = full_tracer_cat[:, bright_cut].transpose()

				load_precomp_tempbank = True 

				bright_src_map = cmock.make_srcmap_temp_bank(ifield, inst, cut_cat, flux_idx=3, load_precomp_tempbank=True, \
															tempbank_dirpath=tempbank_dirpath)
				
				bright_src_map_cross = cmock.make_srcmap_temp_bank(ifield, cross_inst, cut_cat, flux_idx=-1, load_precomp_tempbank=True, \
															tempbank_dirpath=tempbank_cross_dirpath)

				diff_full_bright = full_src_map - bright_src_map
				diff_full_bright -= np.mean(diff_full_bright)
				
				diff_full_bright_cross = full_src_map_cross - bright_src_map_cross
				diff_full_bright_cross -= np.mean(diff_full_bright_cross)

				lb, cl_diff_maglim, clerr_maglim = get_power_spec(diff_full_bright, map_b=diff_full_bright_cross,\
																  lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)

				cl_diff_maglim /= (B_ells[fieldidx]*B_ells_cross[fieldidx])
				
				if m==0:
					lb, cl_full, clerr_full = get_power_spec(full_src_map-np.mean(full_src_map), map_b=full_src_map_cross-np.mean(full_src_map_cross),\
															 lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
					cl_full /= (B_ells[fieldidx]*B_ells_cross[fieldidx])

					cl_table = [lb, cl_full]
					names = ['lb', 'cl_full']
				cl_table.append(cl_diff_maglim)
				names.append(name)

			cl_save_fpath = powerspec_truth_fpath+'/cls_cross_'+mode+'_vs_maglim_ifield'+str(ifield)+'_inst'+str(inst)+'_crossinst'+str(cross_inst)+'_simidx'+str(setidx)+'.fits'

			if save:
				save_resid_cl_file(cl_table, names, mode=mode, save=True, cl_save_fpath=cl_save_fpath, inst=inst, cross_inst=cross_inst, ifield=ifield, setidx=setidx)

			cmock.psf_temp_bank = None

			if ifield==ifield_plot:
				cl_table_test = fits.open(cl_save_fpath)['cls_'+mode].data
				plt.figure()
				plt.loglog(lb, lb**2*cl_table_test['cl_full']/(2*np.pi), label='full')
				for m in range(len(mag_lim_list)):
					maglim_key = 'cl_maglim_'+str(mag_lim_list[m])
					if mag_lim_list_cross is not None:
						maglim_key += '_'+str(mag_lim_list_cross[m])
					
					plt.loglog(lb, lb**2*cl_table_test[maglim_key]/(2*np.pi), label=cbps.inst_to_band[inst]+' > '+str(mag_lim_list[m]))
				plt.legend(fontsize=14)
				plt.ylim(1e-3, 1e4)
				plt.show()


def compute_residual_source_shot_noise(mag_lim_list, inst, ifield_list, datestr, fpath_dict=None, mode='cib', cmock=None, cbps=None, convert_Vega_to_AB=True, \
									simidx0=0, nsims=100, ifield_plot=4, save=True, return_cls=True, \
									  ciber_mock_dirpath='/Volumes/Seagate Backup Plus Drive/Toolkit/Mirror/Richard/ciber_mocks/', trilegal_fpaths=None, \
									  tailstr='Vega_magcut', return_src_maps=False, simstr='with_dpoint', \
									  collect_cls=False, ):
	
	
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

	if return_src_maps:
		src_maps = np.zeros((len(ifield_list), len(mag_lim_list), cbps.dimx, cbps.dimy))

	if collect_cls:
		src_map_cls = np.zeros((len(ifield_list), len(mag_lim_list), nsims, cbps.n_ps_bin))


	for setidx in np.arange(simidx0, nsims):
		print('On set ', setidx, 'of ', nsims)
		for fieldidx, ifield in enumerate(ifield_list):
			if setidx==0 and fieldidx==0:
				plt.figure()
				plt.loglog(cbps.Mkk_obj.midbin_ell, B_ells[fieldidx]**2)
				plt.show()
			   
			if mode=='isl':
				if trilegal_fpaths is not None:
					trilegal_path = trilegal_fpaths[setidx][fieldidx]
				else:
					trilegal_path = trilegal_base_path+'/mock_trilegal_simidx'+str(setidx)+'_'+datestr+'.fits'

				mock_trilegal = fits.open(trilegal_path)
				full_src_map = mock_trilegal['trilegal_'+str(cbps.inst_to_band[inst])+'_'+str(ifield)].data
				mock_trilegal_cat = mock_trilegal['tracer_cat_'+str(ifield)].data
				full_tracer_cat = np.array([mock_trilegal_cat['x'], mock_trilegal_cat['y'], mock_trilegal_cat[bandstr], mock_trilegal_cat[Iarr_str]])

			elif mode=='cib':
				# load in mock cib image/tracer catalog

				# mock_gal = fits.open(cib_realiz_path+'/cib_with_tracer_5field_set'+str(setidx)+'_'+datestr+'_TM'+str(inst)+'.fits')
				# mock_gal = fits.open(cib_realiz_path+'/cib_with_tracer_with_dpoint_5field_set'+str(setidx)+'_'+datestr+'_TM'+str(inst)+'.fits')
				
				mock_gal = fits.open(cib_realiz_path+'/cib_with_tracer_'+simstr+'_5field_set'+str(setidx)+'_'+datestr+'_TM'+str(inst)+'.fits')

				mock_gal_cat = mock_gal['tracer_cat_'+str(ifield)].data
				full_src_map = mock_gal['cib_'+cbps.inst_to_band[inst]+'_'+str(ifield)].data
				
				# full_src_map = mock_gal['map_'+str(ifield)].data
				# Tracer catalog magnitudes are in AB system
				I_arr_full = cmock.mag_2_nu_Inu(mock_gal_cat[bandstr], inst-1)
				full_tracer_cat = np.array([mock_gal_cat['x'], mock_gal_cat['y'], mock_gal_cat[bandstr], I_arr_full])
			else:
				print('mode not recognized, try again!')
				return None
			
			# print('tracer cat for ifield'+str(ifield)+' has shape ', full_tracer_cat.shape)

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
				print('Number of sources > mag lim is ', len(np.where(full_tracer_cat[2,:] < mag_limit_match)[0]))
				cut_cat = full_tracer_cat[:, bright_cut].transpose()
				if m==0:
					load_precomp_tempbank = True 
				else:
					load_precomp_tempbank = False
				bright_src_map = cmock.make_srcmap_temp_bank(ifield, inst, cut_cat, flux_idx=-1, load_precomp_tempbank=load_precomp_tempbank, \
															tempbank_dirpath=tempbank_dirpath)
				
				diff_full_bright = full_src_map - bright_src_map

				if return_src_maps and setidx==simidx0:
					src_maps[fieldidx, m] = diff_full_bright

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

				if collect_cls:
					# print(cl_diff_maglim)
					src_map_cls[fieldidx, m, setidx] = cl_diff_maglim
				cl_table.append(cl_diff_maglim)
				names.append('cl_maglim_'+str(mag_lim))


			cl_save_fpath = powerspec_truth_fpath+'/cls_'+mode+'_vs_maglim_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(setidx)
			if tailstr is not None:
				cl_save_fpath+= '_'+tailstr

			cl_save_fpath += '.fits'
			
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
				
			if ifield==ifield_plot and save:
				cl_table_test = fits.open(cl_save_fpath)['cls_'+mode].data

				plt.figure()
				plt.loglog(lb, lb**2*cl_table_test['cl_full']/(2*np.pi), label='full')
				for m in range(len(mag_lim_list)):
					plt.loglog(lb, lb**2*cl_table_test['cl_maglim_'+str(mag_lim_list[m])]/(2*np.pi), label=cbps.inst_to_band[inst]+' > '+str(mag_lim_list[m]))
				plt.legend(fontsize=14)
				plt.ylim(1e-3, 1e4)
				plt.grid()
				plt.show()

	if return_src_maps:
		return src_maps

	if collect_cls:
		return src_map_cls
	if mode=='isl':
		return isl_sb_rms
	elif mode=='cib':
		return None
	 

def poisson_pred_ciber_spitzer_wrapper(mag_min_irac=16.0, modes = ['full', 'igl', 'isl'], src_types = [None, 0, 1], \
												apply_color_correction=False, mag_max=30., ifield_list=[4, 5, 6, 7, 8], \
													twomass_bandstrs = ['j_m', 'h_m'], inst_list = [1, 2], irac_ch_list=[1, 2], \
												bandstrs=['J', 'H'], datestr=None):
	''' One script to do it all, incorporate modifications to Jordan model '''
	
	cosmos_aeff = 1.58 # deg^2
	save_fpath  = config.ciber_basepath+'data/catalogs/COSMOS15/COSMOS15_rizy_JHKs_CH1CH2_AB_hjmcc.csv'
	c15_df = pd.read_csv(save_fpath)
	
	twomass_aeff = 4.0

	if apply_color_correction:
		c15_ccorr = np.load(config.ciber_basepath+'data/cl_predictions/c15_shotnoise_ratio_ciber_uvista_nuInu_indiv_magcut.npz')
		mmin_range_ccorr = c15_ccorr['mmin_range']
#         ratio_c15_ccorr = c15_ccorr['ratio_cib_uv']

		
	# CIBER autos vs masking depth, single band masking selection
	# 2MASS, COSMOS, TRILEGAL
	J_mag_range = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 17.5, 18.0, 18.5]
	H_mag_range = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 17.5, 18.0]
	ciber_band_mags = [J_mag_range, H_mag_range]

	for idx, inst in enumerate(inst_list):
		
		auto_pv_c15 = np.zeros((len(modes), len(ciber_band_mags[idx])))
		
		# loop over COSMOS
		for m, mode in enumerate(modes):
			for midx, mag in enumerate(ciber_band_mags[idx]):

				auto_pv = calc_catalog_poisson_fluc(inst, c15_df, mag_min=mag, mag_max=mag_max, aeff=cosmos_aeff,\
													convert_to_AB=True, lam_eff=lam_eff)

				if apply_color_correction:
					ccorr = grab_color_corr(mag, mmin_range_ccorr, ratio_cib_uv)
					
				auto_pv_c15[m, midx] = auto_pv
		  
		auto_pv_COSMOS_fpath = ''
		
		if datestr is not None:
			auto_pv_COSMOS_fpath += '_'+datestr
		
		print('Saving TM'+str(inst)+' COSMOS 2015 Poisson predictions to '+auto_pv_COSMOS_fpath+'.npz..')
		np.savez(auto_pv_COSMOS_fpath+'.npz', inst=inst, ifield_list=ifield_list, mag_lims=ciber_band_mags[idx], \
				auto_pv_c15=auto_pv_c15)

		# 2MASS 
			
		auto_pv_2MASS = np.zeros((len(ifield_list), len(ciber_band_mags[idx])))
		for fieldidx, ifield in enumerate(ifield_list):
			fieldname = cbps.ciber_field_dict[ifield]
			twom_basepath = config.ciber_basepath+'data/catalogs/2MASS/filt/'
			twom_fpath = twom_basepath+'2MASS_filt_rdflag_wxy_'+fieldname+'_Jlt17.5.csv'
			twomass_df = pd.read_csv(twom_fpath)

			for midx, mag in enumerate(ciber_band_mags[idx]):

				auto_pv = calc_catalog_poisson_fluc(inst, twomass_df, mag_min=mag, mag_max=mag_max,\
													bandstr=twomass_bandstrs[idx], aeff=twomass_aeff, convert_to_AB=True)

				auto_pv_2MASS[fieldidx, midx] = auto_pv
				
		auto_pv_2MASS_fpath = ''
		
		if datestr is not None:
			auto_pv_2MASS_fpath += '_'+datestr
		
		print('Saving TM'+str(inst)+' 2MASS Poisson predictions to '+auto_pv_2MASS_fpath+'.npz..')
		np.savez(auto_pv_2MASS_fpath+'.npz', inst=inst, ifield_list=ifield_list, mag_lims=ciber_band_mags[idx], \
				auto_pv_2MASS=auto_pv_2MASS)

	
	# CIBER autos vs masking depth, J+H band masking selection
	# CIBER crosses vs masking depth, J+H band masking selection
	# 2MASS, COSMOS, TRILEGAL
	# IGL, ISL
	J_mag_range = [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 17.5, 18.0, 18.5]
	H_mag_range = np.array(J_mag_range)-0.5
			
	all_auto_J_pv, all_auto_H_pv, all_cross_JH_pv = [np.zeros((len(modes), len(J_mag_range))) for x in range(3)]
	
	# COSMOS
	for m, mode in enumerate(modes):
		for midx, mag in enumerate(J_mag_range):
			auto_pv, auto_pv_crossinst, cross_pv = calc_catalog_poisson_fluc(1, c15_df, mag_min=mag, mag_max=mag_max, \
											  bandstr_cut=bandstr, bandstr=bandstr, aeff=cosmos_aeff, \
												 lam_eff=1.65, src_type=src_types[m], cross_inst=2, mag_min_cross=H_mag_range[midx], \
												convert_to_AB=True)

			if apply_color_correction:
				ccorr = grab_color_corr(mag, mmin_range_ccorr, ratio_cib_uv)

			all_auto_J_pv[m, midx] = auto_pv
			all_auto_H_pv[m, midx] = auto_pv_crossinst
			all_cross_JH_pv[m, midx] = cross_pv
					
	c15_auto_cross_fpath = ''
	if datestr is not None:
		c15_auto_cross_fpath += '_'+datestr
	np.savez(c15_auto_cross_fpath+'.npz', J_mag_range=J_mag_range, H_mag_range=H_mag_range, apply_color_correction=apply_color_correction, \
			modes=modes, all_auto_J_pv=all_auto_J_pv, all_auto_H_pv=all_auto_H_pv, all_cross_JH_pv=all_cross_JH_pv)
		
	# 2MASS
	auto_pv_J_2MASS, auto_pv_H_2MASS, cross_pv_JH_2MASS = [np.zeros((len(ifield_list), len(ciber_band_mags[idx]))) for x in range(3)]
	lam_eff, lam_eff_cross = 1.25, 1.65
	for fieldidx, ifield in enumerate(ifield_list):
		
		for midx, mag in enumerate(J_mag_range):

			auto_pv_J, auto_pv_H, cross_pv_JH = calc_catalog_poisson_fluc(inst, twomass_df, cross_inst=cross_inst, mag_min=mag, mag_min_cross=H_mag_range[midx],\
															mag_max=mag_max, bandstr='j_m', bandstr_cross='h_m', aeff=4.0, convert_to_AB=True, \
															   lam_eff=lam_eff, lam_eff_cross=lam_eff_cross)

			auto_pv_J_2MASS[fieldidx, midx] = auto_pv_J
			auto_pv_H_2MASS[fieldidx, midx] = auto_pv_H
			cross_pv_JH_2MASS[fieldidx, midx] = cross_pv_JH
		
		
	pv_filename_2MASS = ''
	if datestr is not None:
		pv_filename_2MASS += '_'+datestr
	np.savez(pv_filename_2MASS+'.npz', auto_pv_J_2MASS=auto_pv_J_2MASS, auto_pv_H_2MASS=auto_pv_H_2MASS, cross_pv_JH_2MASS=cross_pv_JH_2MASS, \
			inst=inst, cross_inst=cross_inst, J_mag_range=J_mag_range, H_mag_range=H_mag_range)

	
	# Spitzer autos vs masking depth, J + CH1 + CH2 selection
	# CIBER x Spitzer crosses vs masking depth, J/H + CH1 + CH2 selection
	
	mag_mins_ciber = [17.5, 17.0] # J/H bands    
	
	all_irac_pv_c15, all_ciber_irac_pv_c15, all_ciber_pv_c15 = [np.zeros((len(modes), len(irac_ch_list), len(inst_list))) for x in range(3)]

	for m, mode in enumerate(modes):

		irac_pv, ciber_irac_pv, ciber_pv = calc_cosmos15_shotnoise_ciber_irac_new(c15_df, mag_mins_ciber=mag_mins_ciber,\
																				  mag_min_irac=mag_min_irac, src_type=src_type, \
																						inst_list=inst_list, irac_ch_list=irac_ch_list)

		all_irac_pv_c15[m,:,:] = irac_pv
		all_ciber_irac_pv_c15[m,:,:] = ciber_irac_pv
		all_ciber_pv_c15[m] = ciber_pv
		
	ciber_irac_pv_c15_fpath = config.ciber_basepath+'data/cl_predictions/irac_vs_ciber_pv_cosmos15_nuInu_CH1CH2mask'
	if datestr is not None:
		ciber_irac_pv_c15_fpath += '_'+datestr
	
	print('Saving CIBER x Spitzer auto/cross Poisson predictions to ', ciber_irac_pv_c15_fpath+'.npz')
	np.savez(ciber_irac_pv_c15_fpath+'.npz', mag_min_irac=mag_min_irac, modes=modes, irac_ch_list=irac_ch_list, inst_list=inst_list, \
			all_irac_pv_c15=all_irac_pv_c15, all_ciber_irac_pv_c15=all_ciber_irac_pv_c15, all_ciber_pv_c15=all_ciber_pv)
		
	


def calc_catalog_poisson_fluc(inst, cat_df, irac_ch=None, irac_ch_cross=None, cross_inst=None, mag_min=17.5, mag_min_cross=None, mag_max=30.0, aeff=1.37,\
							bandstr=None, bandstr_cut=None, bandstr_cross=None, convert_to_AB=False, lam_eff=None, lam_eff_cross=None):

	''' All shot noise prediction inputs should be in AB magnitudes '''
	
	cmock = ciber_mock()
	
		
	bandstr_dict = dict({1:'J', 2:'H'})
	lam_dict = dict({1:1.05, 2:1.79})
	Vega_to_AB = dict({1:0.91, 2:1.39})
	
	lam_irac_dict = dict({1:3.6, 2:4.5})
	Vega_to_AB_irac = dict({1:2.699, 2:3.339})
	
	if lam_eff is None:
		lam_eff = lam_dict[inst]
	if lam_eff_cross is None and cross_inst is not None:
		lam_eff_cross = lam_dict[cross_inst]
		
	if bandstr is None:
		bandstr = bandstr_dict[inst]
	if bandstr_cross is None and cross_inst is not None:
		bandstr_cross = bandstr_dict[cross_inst]
		
	cat_mag = np.array(cat_df[bandstr])
	
	if src_type is not None:
		cat_type = np.array(cat_df['type'])
	
	if convert_to_AB:
#         print('converting to AB magntiudes')
		cat_mag += Vega_to_AB[inst]
	
	if cross_inst is not None:
		cat_mag_cross = np.array(cat_df[bandstr_cross])
			
		if convert_to_AB:
			cat_mag_cross += Vega_to_AB[cross_inst]

	if bandstr_cut is None:
		catmask = (cat_mag > mag_min)*(cat_mag < mag_max)
	else:
		cat_mag_sel = np.array(cat_df[bandstr_cut])
		catmask = (cat_mag_sel > mag_min)*(cat_mag_sel < mag_max)
		
	if mag_min_cross is not None:
		catmask *= (cat_mag_cross > mag_min_cross)*(cat_mag_cross < mag_max)

	if src_type is not None:
		catmask *= (cat_type==src_type)
	
		
	volfac = (1./aeff)/3.046e-4
		
	nu_Inu = cmock.mag_2_nu_Inu(cat_mag[catmask], lam_eff=lam_eff*1e-6*u.m)*cmock.pix_sr
	auto_poisson_var = np.sum(volfac*(nu_Inu**2))

	if cross_inst is not None:
		cat_mag_sel_cross = cat_mag_cross[catmask]
		
		nu_Inu_crossinst = cmock.mag_2_nu_Inu(cat_mag_cross[catmask], lam_eff=lam_eff_cross*1e-6*u.m)*cmock.pix_sr
		auto_poisson_var_crossinst = np.sum(volfac*(nu_Inu_crossinst**2))
		cross_poisson_var = np.sum(volfac*(nu_Inu*nu_Inu_crossinst))
		
		return auto_poisson_var, auto_poisson_var_crossinst, cross_poisson_var
	
	return auto_poisson_var


def calc_shotnoise_from_cat(inst, cat_df, xlims, ylims, mag_min=17.0, mag_max=30.0, cat_type='predict', convert_to_AB=True):
	
	if inst==1:
		lam = 1.05
		Vega_to_AB = 0.91
		bandstr = 'J'
	else:
		lam = 1.79
		Vega_to_AB = 1.31
		bandstr = 'H'
	# load catalog
	
	cat_x = np.array(cat_df['x'+str(inst)])
	cat_y = np.array(cat_df['y'+str(inst)])
	
	if cat_type=='predict':
		cat_mag = np.array(cat_df[bandstr+'_Vega_predict'])
	else:
		cat_mag = np.array(cat_df[bandstr])
		
	
	catmask = (cat_mag > mag_min)*(cat_mag < mag_max)
			
	xmask = (cat_x > xlims[0])*(cat_x < xlims[1])
	catmask *= xmask
		
	ymask = (cat_y > ylims[0])*(cat_y < ylims[1])
	catmask *= ymask
		
	cat_x_sel = cat_x[catmask]
	cat_y_sel = cat_y[catmask]
	cat_mag_sel = cat_mag[catmask]
		
		
	if convert_to_AB:
		# convert Vega magnitudes to AB
		cat_mag_sel += Vega_to_AB
		
	# calculate nuInu for each source
	nu_Inu = cmock.mag_2_nu_Inu(cat_mag_sel, lam_eff=lam*1e-6*u.m)*cmock.pix_sr

	nm_perdeg = len(cat_mag_sel)
	
	aeff = (xlims[1]-xlims[0])*(ylims[1]-ylims[0])*(7**2)/(3600**2)
	
	volfac = 1./aeff # 1/deg^2
	volfac /= 3.046e-4 # 1/sr    
	
	poisson_var = np.sum(volfac*(nu_Inu**2))
	
	return poisson_var

def calc_cosmos15_shotnoise_ciber_irac(cat_df, inst_list=[1, 2], irac_ch_list = [1, 2],\
									   mag_mins_ciber=[17.5, 17.0], mag_min_irac=16.0, mag_max=30.0, aeff=1.37):
	
	cmock = ciber_mock()
	volfac = (1./aeff)/3.046e-4
	
	mag_mins_ciber = np.array(mag_mins_ciber)

	bandstr_dict = dict({1:'J', 2:'H'})
	lam_ciber_dict = dict({1:1.05, 2:1.79})
	Vega_to_AB_ciber = dict({1:0.91, 2:1.39})
	
	lam_irac_dict = dict({1:3.6, 2:4.5})
	Vega_to_AB_irac = dict({1:2.699, 2:3.339})
	
	mag_mins_ciber[0] += Vega_to_AB_ciber[1]
	mag_mins_ciber[1] += Vega_to_AB_ciber[2]
	mag_min_irac += Vega_to_AB_irac[1]
	
	print('mag mins ciber (AB):', mag_mins_ciber)
	print('mag min irac (AB):', mag_min_irac)
	
	cat_mag_TM1 = np.array(cat_df['J'])
	cat_mag_TM2 = np.array(cat_df['H'])
	
	cat_mags_ciber = [cat_mag_TM1, cat_mag_TM2]
	
	cat_mag_CH1 = np.array(cat_df['mag_CH1'])
	cat_mag_CH2 = np.array(cat_df['mag_CH2'])
	
	cat_mags_irac = [cat_mag_CH1, cat_mag_CH2]
	
	irac_pv = np.zeros((len(irac_ch_list), len(inst_list)))
	ciber_irac_pv = np.zeros((len(irac_ch_list), len(inst_list)))
	
	
	for idx, irac_ch in enumerate(irac_ch_list):
		
		for ciberidx, inst in enumerate(inst_list):
			
			catmask = (cat_mags_ciber[ciberidx] > mag_mins_ciber[ciberidx])*(cat_mags_ciber[ciberidx] < mag_max)
			catmask *= (cat_mag_CH1 > mag_min_irac)
			catmask *= (~np.isinf(cat_mags_ciber[ciberidx]))*(~np.isinf(cat_mag_CH1))*(~np.isinf(cat_mags_irac[idx]))
			
			plt.figure()
			plt.hist(cat_mags_irac[idx][catmask], bins=20, histtype='step', label='IRAC select')
			plt.hist(cat_mags_ciber[idx][catmask], bins=20, histtype='step', label='CIBER select')
			plt.xlabel('mAB')
			plt.yscale('log')
			plt.legend()
			plt.show()
			
			nu_Inu_ciber = cmock.mag_2_nu_Inu(cat_mags_ciber[ciberidx][catmask], lam_eff=lam_ciber_dict[inst]*1e-6*u.m)*cmock.pix_sr
			nu_Inu_irac = cmock.mag_2_nu_Inu(cat_mags_irac[idx][catmask], lam_eff=lam_irac_dict[irac_ch]*1e-6*u.m)*cmock.pix_sr
	
			irac_auto_poisson_var = np.nansum(volfac*(nu_Inu_irac**2))
			ciber_irac_cross_poisson_var = np.nansum(volfac*(nu_Inu_irac*nu_Inu_ciber))
		
			irac_pv[idx, ciberidx] = irac_auto_poisson_var.value
			ciber_irac_pv[idx, ciberidx] = ciber_irac_cross_poisson_var.value
			
	return irac_pv, ciber_irac_pv

