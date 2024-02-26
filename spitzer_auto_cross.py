import numpy as np
from astropy.io import fits
from plotting_fns import plot_map
import config
from ciber_powerspec_pipeline import *
from noise_model import *
# from cross_spectrum_analysis import *
from cross_spectrum import *
from mkk_parallel import *
from mkk_diagnostics import *
from flat_field_est import *
from plotting_fns import *
from powerspec_utils import *
from astropy.coordinates import SkyCoord
import healpy as hp
import scipy.io



class spitzer_cl():
    
    def __init__(self):
        self.all_cl_spitzer = []
        self.all_clerr_spitzer = []
        self.all_cl_cross = []
        self.all_clerr_cross = []
        self.all_clerr_cross_tot = []
        self.all_nl1d_diff = []
        self.all_nl1d_err_diff = []
        self.all_fft2_spitzer = []
        self.all_clerr_spitzer_noise_ciber = []
        self.all_clerr_ciber_noise_spitzer = []
        
    def append_nls(self, nl1d_diff=None, nl1d_err_diff=None, nl2d_diff=None):
        
        if nl1d_diff is not None:
            self.all_nl1d_diff.append(nl1d_diff)
        if nl1d_err_diff is not None:
            self.all_nl1d_err_diff.append(nl1d_err_diff)
        if nl2d_diff is not None:
            self.all_fft2_spitzer.append(nl2d_diff)
        
    def append_cls(self, cl_cross=None, clerr_cross=None, clerr_cross_tot=None, \
                  clerr_ciber_noise_spitzer=None, clerr_spitzer_noise_ciber=None, \
                  cl_spitzer=None, clerr_spitzer=None):
        
        if cl_cross is not None:
            self.all_cl_cross.append(cl_cross)
        if clerr_cross is not None:
            self.all_clerr_cross.append(clerr_cross)
        if clerr_cross_tot is not None:
            self.all_clerr_cross_tot.append(clerr_cross_tot)
        if clerr_ciber_noise_spitzer is not None:
            self.all_clerr_ciber_noise_spitzer.append(clerr_ciber_noise_spitzer)
        if clerr_spitzer_noise_ciber is not None:
            self.all_clerr_spitzer_noise_ciber.append(clerr_spitzer_noise_ciber)
        if cl_spitzer is not None:
            self.all_cl_spitzer.append(cl_spitzer)
        if clerr_spitzer is not None:
            self.all_clerr_spitzer.append(clerr_spitzer)

def load_irac_bl(irac_ch, base_path=None):

	if base_path is None:
		base_path = config.ciber_basepath+'data/Spitzer/'
	irac_bl_fpath = base_path+'irac_ch'+str(irac_ch)+'_bl.csv'
	print('loading from ', irac_bl_fpath)
	irac_bl = pd.read_csv(irac_bl_fpath, header=None)
	irac_lb, irac_blval = np.array(irac_bl)[:,0], np.array(irac_bl)[:,1]
	
	irac_lb = np.array([50., 100.]+list(irac_lb))
	irac_blval = np.array([1., 1.]+list(irac_blval))

	irac_lb *= 2
	return irac_lb, irac_blval


def calc_cl_simp(cl, cl_err, bl=None, t_ell=None, inv_Mkk=None):

    if inv_Mkk is not None:
        cl = np.dot(inv_Mkk.transpose(), cl)
        invmkkdiag = np.diagonal(inv_Mkk.transpose())
        print('invmkkdiag = ', invmkkdiag)
        cl_err /= invmkkdiag

    if bl is not None:
        cl /= bl**2
        cl_err /= bl**2

    if t_ell is not None:
        cl /= t_ell
        cl_err /= t_ell

    return cl, cl_err

def load_ciber_spitzer_data_products(cbps, irac_ch, inst, ifield_list, spitzer_string, mask_tail, plot=False, \
                                    base_path=None):
    
    if base_path is None:
        base_path = config.ciber_basepath+'data/'
        
    bootes_ifields = [6, 7]
        
    field_set_shape = (len(ifield_list), cbps.dimx, cbps.dimy)
    masks, ciber_maps = [np.zeros(field_set_shape) for x in range(2)]
    
    lb = cbps.Mkk_obj.midbin_ell
        
    # IRAC beam interpolate to CIBER bandpass centers
    irac_lb, irac_blval = load_irac_bl(irac_ch=irac_ch)
    interp_maskfn = scipy.interpolate.interp1d(np.log10(irac_lb), np.log10(irac_blval))
    cibermatch_bl = 10**interp_maskfn(np.log10(lb))
    
    # CIBER beams
    bls_fpath = base_path+'fluctuation_data/TM'+str(inst)+'/beam_correction/bl_est_postage_stamps_TM'+str(inst)+'_081121.npz'
    B_ells = np.load(bls_fpath)['B_ells_post']
    

    # transfer function 
    t_ell_av_grad = np.load(base_path+'transfer_function/t_ell_est_nsims=100.npz')['t_ell_av']
    t_ell_av_quad = np.load(base_path+'transfer_function/t_ell_masked_wdgl_2500_n=2p5_wgrad_quadrant_nsims=1000.npz')['t_ell_av']
    t_ell_av = np.sqrt(t_ell_av_grad*t_ell_av_quad)
    
    if plot:
        plt.figure(figsize=(6,5))
        plt.scatter(lb, cibermatch_bl, label='IRAC')
        plt.plot(lb, B_ells[0], label='CIBER')
        plt.plot(lb, np.sqrt(cibermatch_bl*B_ells[0]), label='IRAC x CIBER')
        plt.plot(lb, t_ell_av, color='k', label='Filtering transfer function')
        plt.legend()
        plt.xlabel('$\\ell$', fontsize=14)
        plt.ylabel('$T_{\\ell}$', fontsize=14)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
    
    # compute effective beam functions and loads maps
    full_bls = np.zeros_like(B_ells)

    for fieldidx, ifield in enumerate(ifield_list):

        full_bls[fieldidx] = np.sqrt(cibermatch_bl*B_ells[fieldidx,:])

        mask_fpath = base_path+'fluctuation_data/TM'+str(inst)+'/masks/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'
        masks[fieldidx] = fits.open(mask_fpath)[1].data.astype(int)

        flight_map = cbps.load_flight_image(ifield, inst, inplace=False, verbose=True)
        print('Subtracting by dark current')
        dark_current = cbps.load_dark_current_template(inst, inplace=False)
        flight_map -= dark_current
        ciber_maps[fieldidx,:,:] = flight_map*cbps.cal_facs[inst]
            
    spitzer_regrid_maps, spitzer_regrid_masks, \
        all_diff1, all_diff2, all_spitz_by_epoch = load_spitzer_bootes_maps(inst, irac_ch, bootes_ifields, spitzer_string)

            
    return ciber_maps, masks, full_bls, cibermatch_bl, t_ell_av, lb, \
            spitzer_regrid_maps, spitzer_regrid_masks, all_diff1, all_diff2, all_spitz_by_epoch

def load_spitzer_bootes_maps(inst, irac_ch, bootes_ifields, spitzer_string, base_path=None):
	
	if base_path is None:
		base_path = config.ciber_basepath+'data/'

	spitzer_regrid_maps, spitzer_regrid_masks, all_diff1, all_diff2, all_spitz_epochs = [[] for x in range(5)]
	irac_lam_dict = dict({1:3.6, 2:4.5})
	
	for idx, ifield in enumerate(bootes_ifields):
		spitzer_fpath = base_path+'Spitzer/spitzer_regrid/sdwfs_regrid_TM'+str(inst)+'_ifield'+str(ifield)+'_'+spitzer_string+'.fits'
		
		print('Loading Spitzer map from', spitzer_fpath)
		spitz = fits.open(spitzer_fpath)
		spitzer_regrid_to_ciber = spitz['sdwfs_map_regrid_ifield'+str(ifield)].data
		spitzer_mask_regrid_to_ciber = spitz['sdwfs_mask_regrid_ifield'+str(ifield)].data

		diff1 = spitz['sdwfs_diff1_ifield'+str(ifield)].data
		diff2 = spitz['sdwfs_diff2_ifield'+str(ifield)].data
		spitz_epochs = spitz['sdwfs_epochs_ifield'+str(ifield)].data

		lam = irac_lam_dict[irac_ch]
		fac = convert_MJysr_to_nWm2sr(lam)
		print("fac for lambda = "+str(lam)+" is "+str(fac))
		
		all_diff1.append(diff1/fac)
		all_diff2.append(diff2/fac)
		all_spitz_epochs.append(spitz_epochs/fac)

		spitzer_mask_regrid_to_ciber = spitz['sdwfs_mask_regrid_ifield'+str(ifield)].data
		spitzer_regrid_maps.append(spitzer_regrid_to_ciber/fac)
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

	
def ciber_spitzer_crosscorr_full_new(irac_ch, inst, spitzer_string, mask_tail, \
                                 compute_mkk=True, photon_noise=True, compute_nl_spitzer_unc=True,\
                                 compute_nl_ciber_unc=True, bootes_idxs = [2,3],\
                                 plot=False, base_path=None, n_mkk_sims=200, n_split=2,\
                                 n_sims_noise=500, n_split_noise=20, mask_string_mkk=None,\
                                 observed_run_name = None, include_ff_errors=True, \
                                 save=True, ifield_list_full = [4, 5, 6, 7, 8],\
                                 add_str=None, add_str_ciber=None, add_str_spitzer=None, apply_FW=False, \
                                    datestr='111323'):
    
    cbps = CIBER_PS_pipeline()
    spitz_obj = spitzer_cl() # for storing power spectrum quantities

    if mask_string_mkk is None:
        mask_string_mkk = mask_tail + '_IRAC_CH'+str(irac_ch)

    if base_path is None:
        base_path = config.ciber_basepath+'data/'
       
    spitz_crossnoise_basepath = base_path+'Spitzer/cross_noise/'
    cibern_spitzm_noise_basepath = spitz_crossnoise_basepath+'ciber_noise_spitzer_map/'
    spitzn_ciberm_noise_basepath = spitz_crossnoise_basepath+'spitzer_noise_ciber_map/'
    
    bootes_ifields = [ifield_list_full[idx] for idx in bootes_idxs]

        
    print('ifield list full is ', ifield_list_full)
    ciber_maps, masks, full_bls, cibermatch_bl, t_ell_av, lb, \
            spitzer_regrid_maps, spitzer_regrid_masks,\
                all_diff1, all_diff2, all_spitz_by_epoch = load_ciber_spitzer_data_products(cbps, irac_ch, inst,\
                                                                                 ifield_list_full, spitzer_string,\
                                                                                 mask_tail, plot=plot, \
                                                                                base_path=base_path)
    
    processed_ciber_maps, masks = process_ciber_maps_by_quadrant(cbps, ifield_list_full, inst, ciber_maps, masks, clip_sigma=5,\
                                                                             nitermax=10)
    
    bootes_obs = [processed_ciber_maps[idx] for idx in bootes_idxs]
    bootes_masks = [masks[idx] for idx in bootes_idxs]
    bootes_bls = [full_bls[idx] for idx in bootes_idxs]
    combined_masks = [bootes_masks[idx]*spitzer_regrid_mask for idx, spitzer_regrid_mask in enumerate(spitzer_regrid_masks)]
    bootes_inv_Mkks = []
    
    if plot:
        for c, cb in enumerate(processed_ciber_maps):
            plot_map(cb*masks[c], title='FF corrected map, ifield '+str(ifield_list_full[c]))
        
        for x in range(len(bootes_ifields)):

            plot_map(spitzer_regrid_maps[x]*combined_masks[x], title='Spitzer x mask ifield '+str(bootes_ifields[x]), figsize=(6,6))
    
    
    spitzer_regrid_maps_meansub = np.zeros_like(spitzer_regrid_maps)
    
    for idx, ifield in enumerate(bootes_ifields):
        
        mkk_basepath = base_path + 'fluctuation_data/TM'+str(inst)+'/mkk/'+mask_string_mkk

        if compute_mkk:
            make_fpaths([mkk_basepath])
            
            Mkk, inv_Mkk = cbps.compute_mkk_matrix(combined_masks[idx], nsims=n_mkk_sims, n_split=n_split, inplace=False)

            plot_mkk_matrix(inv_Mkk, inverse=True, symlogscale=True)
            hdul = write_Mkk_fits(Mkk, inv_Mkk, ifield, inst, \
                                 use_inst_mask=True, dat_type='ciber_spitzer_cross')

            mkkpath = mkk_basepath+'/mkk_maskonly_estimate_ifield'+str(ifield)+'_inst'+str(inst)+'_'+mask_string_mkk+'.fits'
            hdul.writeto(mkkpath, overwrite=True)
        else:
            mkkpath = mkk_basepath+'/mkk_maskonly_estimate_ifield'+str(ifield)+'_inst'+str(inst)+'_'+mask_string_mkk+'.fits'
            inv_Mkk = fits.open(mkkpath)['inv_Mkk_'+str(ifield)].data
            
            plot_mkk_matrix(inv_Mkk, inverse=True)
            
        bootes_inv_Mkks.append(inv_Mkk)

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
        
        # we want to compute an effective Spitzer noise model, which we do by computing 2D power spectra of the epoch differences. 
        # since we are drawing from 2D Fourier space, want to apply mode coupling correction to azimuthally spaced bins
        l2d, nl2d_diff = compute_nl2d_spitzer_crossepoch_diffs(diff1_indiv, diff2_indiv, combined_masks[idx], plot=plot)
        # noise model is for difference of individual epochs, so divide by 8 to get 4-epoch averaged
        nl2d_diff /= 8. 
        
        nl2d_corr = cbps.correct_mkk_azim_cl2d(nl2d_diff, inv_Mkk)
        plot_map(nl2d_corr/nl2d_diff, title='mkk corrected / original')
        
        lb, nl1d_diff, nl1d_err_diff = azim_average_cl2d(nl2d_corr, l2d, nbins=25, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell, logbin=True, verbose=False)

        spitz_obj.append_nls(nl1d_diff=nl1d_diff, nl1d_err_diff=nl1d_err_diff, nl2d_diff=nl2d_corr)
        
        hdul = write_noise_model_fits(nl2d_corr, ifield, inst, dat_type='IRAC_CH'+str(irac_ch))

        noise_fpath = base_path+'fluctuation_data/TM'+str(inst)+'/noise_model/spitzer_noise/spitzer_CH'+str(irac_ch)+'_noisemodl_TM'+str(inst)+'_ifield'+str(ifield)+'.fits'
        print('Saving Spitzer noise model to ', noise_fpath)
        hdul.writeto(noise_fpath, overwrite=True)

    # computes noise terms for both bootes fields ("both")
    if compute_nl_spitzer_unc:
        print('Estimating Spitzer noise x CIBER..')
        all_nl_spitzer_fpath, lb, all_nl1ds_cross_ciber_both = estimate_spitzer_noise_cross_ciber(cbps, irac_ch, inst, bootes_obs, combined_masks, spitzer_string, nsims=n_sims_noise, n_split=n_split_noise, \
                                                      base_path=spitzn_ciberm_noise_basepath, fft2_spitzer=np.array(spitz_obj.all_fft2_spitzer), compute_spitzer_FW=True, save=True, add_str=add_str_spitzer)
        
        
    if compute_nl_ciber_unc:
        
        print('Estimating CIBER noise x Spitzer..')
        all_nl_ciber_fpath, lb, all_nl1ds_cross_spitzer_both = estimate_ciber_noise_cross_spitzer(cbps, irac_ch, inst, spitzer_regrid_maps_meansub, combined_masks, spitzer_string,\
                                                                                                        nsims=n_sims_noise, n_split=n_split_noise, include_ff_errors=include_ff_errors, observed_run_name=observed_run_name, \
                                                                                                      base_path=cibern_spitzm_noise_basepath, save=True, photon_noise=photon_noise, add_str=add_str_ciber, \
                                                                                                 datestr=datestr)
        print('all nl ciber fpath:', all_nl_ciber_fpath)

        
        
    for idx, ifield in enumerate(bootes_ifields):
        
        spitzmask_bootes_obs = bootes_obs[idx]*combined_masks[idx] # ciber maps

        if compute_nl_spitzer_unc:
            nmfile_spitzernoise_ciber = np.load(all_nl_spitzer_fpath[idx])
        else:
            add_str_spitzer = ''
            nl_spitzer_fpath = spitzn_ciberm_noise_basepath+'nl1ds_TM'+str(inst)+'_ifield'+str(ifield)+'_irac_ch'+str(irac_ch)+'_spitzer_noise_ciber_map'+add_str_spitzer+'.npz'
            print('Loading Spitzer noise cross model from ', nl_spitzer_fpath)
            nmfile_spitzernoise_ciber = np.load(nl_spitzer_fpath)

        if compute_nl_ciber_unc:
            nmfile_cibernoise_spitzer = np.load(all_nl_ciber_fpath[idx])
        else:
            add_str_ciber = ''

            nl_ciber_fpath = cibern_spitzm_noise_basepath+'nl1ds_TM'+str(inst)+'_ifield'+str(ifield)+'_irac_ch'+str(irac_ch)+'_ciber_noise_spitzer_map'+add_str_ciber+'.npz'
            print('Loading CIBER noise cross model from ', nl_ciber_fpath)
            nmfile_cibernoise_spitzer = np.load(nl_ciber_fpath)
        
        lb = nmfile_cibernoise_spitzer['lb']
        all_nl1ds_cross_ciber = nmfile_spitzernoise_ciber['all_nl1ds_cross_ciber']
        av_nl1ds_spitzernoise_ciber_cross = np.mean(all_nl1ds_cross_ciber, axis=0)
        std_nl1ds_spitzernoise_ciber_cross = np.std(all_nl1ds_cross_ciber, axis=0)

        all_nl1ds_cross_spitzer = nmfile_cibernoise_spitzer['all_nl1ds_cross_spitzer']
        std_nl1ds_cibernoise_spitzer_cross = np.std(all_nl1ds_cross_spitzer, axis=0)
        
        spitzmask_bootes_obs = cbps.mean_sub_masked_image_per_quadrant(spitzmask_bootes_obs, (spitzmask_bootes_obs != 0))
        
        lb, ciber_spitzer_cross_cl, ciber_spitzer_cross_clerr = get_power_spec(spitzmask_bootes_obs, map_b=spitzer_regrid_maps_meansub[idx], mask=None, weights=None, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
        lb, spitzer_auto_cl, spitzer_auto_clerr = get_power_spec(spitzer_regrid_maps_meansub[idx], mask=None, weights=None, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)


        ciber_spitzer_cross_cl, ciber_spitzer_cross_clerr = calc_cl_simp(ciber_spitzer_cross_cl, ciber_spitzer_cross_clerr, \
                                                                                       bl=bootes_bls[idx], t_ell=t_ell_av, inv_Mkk=bootes_inv_Mkks[idx])

        spitzer_auto_cl, spitzer_auto_clerr = calc_cl_simp(spitzer_auto_cl, spitzer_auto_clerr, \
                                                                                       bl=cibermatch_bl, t_ell=t_ell_av, inv_Mkk=bootes_inv_Mkks[idx])

        
        pf = lb*(lb+1)/(2*np.pi)
        
        clerr_cross_tot = np.sqrt(std_nl1ds_cibernoise_spitzer_cross**2+std_nl1ds_spitzernoise_ciber_cross**2)
        clerr_cross_tot /= t_ell_av
        clerr_cross_tot /= bootes_bls[idx]**2

        clerr_ciber_noise_spitzer = std_nl1ds_cibernoise_spitzer_cross/bootes_bls[idx]**2/t_ell_av
        clerr_spitzer_noise_ciber = std_nl1ds_spitzernoise_ciber_cross/bootes_bls[idx]**2/t_ell_av
        
                
        spitz_obj.append_cls(cl_cross=ciber_spitzer_cross_cl, clerr_cross=ciber_spitzer_cross_clerr, clerr_cross_tot=clerr_cross_tot, \
                            clerr_ciber_noise_spitzer=clerr_ciber_noise_spitzer, clerr_spitzer_noise_ciber=clerr_spitzer_noise_ciber, \
                            cl_spitzer=spitzer_auto_cl, clerr_spitzer=spitzer_auto_clerr)

        fieldname = cbps.ciber_field_dict[ifield]
        fig = plot_ciber_spitzer_cross_with_terms(inst, irac_ch, fieldname, lb, ciber_spitzer_cross_cl, clerr_cross_tot, clerr_ciber_noise_spitzer, \
                                       clerr_spitzer_noise_ciber, return_fig=True)
        
        fig.savefig('figures/ciber_spitzer/ciber_spitzer_crosscl_TM'+str(inst)+'_IRACCH'+str(irac_ch)+'_'+fieldname+'_withcrossterms.png', bbox_inches='tight', dpi=200)
        

    all_clerr_cross_tot = np.array(spitz_obj.all_clerr_cross_tot)
    all_cl_cross = np.array(spitz_obj.all_cl_cross)
    
    # bootes A and B have roughly same noise properties, so just average them for final result
    
    fieldav_cl_cross = np.mean(all_cl_cross, axis=0)
    fieldav_clerr_cross = np.sqrt(all_clerr_cross_tot[0]**2 + all_clerr_cross_tot[1]**2)/np.sqrt(2)
    
    all_cl_spitzer = np.array(spitz_obj.all_cl_spitzer)
    all_clerr_spitzer = np.array(spitz_obj.all_clerr_spitzer)
    
    fieldav_cl_spitzer = np.mean(all_cl_spitzer, axis=0)
    fieldav_clerr_spitzer = np.sqrt(all_clerr_spitzer[0]**2 + all_clerr_spitzer[1]**2)/np.sqrt(2)
    
    
    fig = plot_spitzer_auto(inst, irac_ch,lb, fieldav_cl_spitzer, fieldav_clerr_spitzer, all_cl_spitzer, all_clerr_spitzer, return_fig=True)
    fig.savefig('figures/ciber_spitzer/spitzer_auto_cl_TM'+str(inst)+'_IRACCH'+str(irac_ch)+'.png', bbox_inches='tight', dpi=200)
    
    
    fig = plot_fieldav_ciber_spitzer_cross(inst, irac_ch, lb, fieldav_cl_cross, fieldav_clerr_cross, all_cl_cross, all_clerr_cross_tot)
    fig.savefig('figures/ciber_spitzer/ciber_spitzer_fieldav_crosscl_TM'+str(inst)+'_IRACCH'+str(irac_ch)+'.png', bbox_inches='tight', dpi=200)
    
    
    if save:
        save_cl_fpath = base_path+'input_recovered_ps/ciber_spitzer/ciber_spitzer_cross_auto_cl_TM'+str(inst)+'_IRACCH'+str(irac_ch)+'.npz'
        print('saving to ', save_cl_fpath)
        np.savez(save_cl_fpath, lb=lb, \
                all_cl_cross=spitz_obj.all_cl_cross, \
                all_clerr_cross=spitz_obj.all_clerr_cross, \
                all_clerr_cross_tot = spitz_obj.all_clerr_cross_tot,\
                all_clerr_spitzer_noise_ciber=spitz_obj.all_clerr_spitzer_noise_ciber, \
                all_clerr_ciber_noise_spitzer=spitz_obj.all_clerr_ciber_noise_spitzer, \
                all_nl1d_diff=spitz_obj.all_nl1d_diff, \
                all_nl1d_err_diff=spitz_obj.all_nl1d_err_diff, \
                all_cl_spitzer=spitz_obj.all_cl_spitzer, \
                all_clerr_spitzer=spitz_obj.all_clerr_spitzer, \
                fieldav_cl_spitzer = fieldav_cl_spitzer, \
                fieldav_clerr_spitzer = fieldav_clerr_spitzer, \
                fieldav_cl_cross = fieldav_cl_cross, \
                fieldav_clerr_cross = fieldav_clerr_cross)

        return save_cl_fpath
    
    return None

def estimate_spitzer_noise_cross_ciber(cbps, irac_ch, inst, bootes_ciber_maps, combined_masks, spitzer_string, bootes_ifield_list=[6, 7], bootes_fieldidxs=[2,3],\
                                       nsims=200, n_split=4, plot=True, fft2_spitzer=None, compute_spitzer_FW=False, \
                                      base_path=None, save=True, add_str=None):

    ''' feed in bootes maps for CIBER and their maps'''
    
    if base_path is None:
        base_path = config.ciber_basepath+'data/Spitzer/cross_noise/spitzer_noise_cross_ciber/'
        
    maplist_split_shape = (nsims//n_split, cbps.dimx, cbps.dimy)
    empty_aligned_objs, fft_objs = construct_pyfftw_objs(3, maplist_split_shape)
    empty_aligned_objs_cross, fft_objs_cross = construct_pyfftw_objs(3, (1, cbps.dimx, cbps.dimy))
    empty_aligned_objs_maps, fft_objs_maps = construct_pyfftw_objs(3, (1, maplist_split_shape[1], maplist_split_shape[2]))

    lb = cbps.Mkk_obj.midbin_ell

    sterad_per_pix = (cbps.pixsize/3600/180*np.pi)**2
    V = cbps.dimx*cbps.dimy*sterad_per_pix
    l2d = get_l2d(cbps.dimx, cbps.dimy, cbps.pixsize)

    if len(fft2_spitzer.shape)==2:
        print('single fft2, recasting into list of nfield='+str(len(bootes_ifield_list))+' copies..')
        fft2_spitzer = [fft2_spitzer for x in range(len(bootes_ifield_list))]
        plot_map(fft2_spitzer[0])

    all_nl1ds_spitzer_both, all_nl1ds_cross_ciber_both,\
        all_nl_save_fpaths, all_fourier_weights = [[] for x in range(4)]

    for bootes_fieldidx, bootes_ifield in enumerate(bootes_ifield_list):

        if not compute_spitzer_FW:
            fourier_weights = []
        all_nl1ds_spitzer, all_nl1ds_cross_ciber = [], []

        ciber_bootes_obs_indiv = bootes_ciber_maps[bootes_fieldidx]
        combined_mask_indiv = combined_masks[bootes_fieldidx]
        mean_nl2d, M2_nl2d = [np.zeros(ciber_bootes_obs_indiv.shape) for x in range(2)]

        count = 0

        masked_ciber_bootes_obs = cbps.mean_sub_masked_image_per_quadrant(ciber_bootes_obs_indiv, combined_mask_indiv)

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
            spitzer_noise_realiz, _ = cbps.noise_model_realization(inst, maplist_split_shape, fft2_spitzer[bootes_fieldidx], fft_obj=fft_objs[0],\
                                                  read_noise=True, photon_noise=False, adu_to_sb=False)

            spitzer_noise_realiz /= cbps.arcsec_pp_to_radian

            if i==0:
                plot_map(spitzer_noise_realiz[0]*combined_mask_indiv, title='spitzer noise')
                plot_map(gaussian_filter(masked_ciber_bootes_obs, sigma=10), title='smoothed ciber map')

            fft_objs[1](spitzer_noise_realiz*sterad_per_pix)

            nl2ds_spitzer_noise_ciber_map = np.array([fftshift(dentry*np.conj(empty_aligned_objs_maps[2][0])).real for dentry in empty_aligned_objs[2]])
            nl1ds_cross_ciber = [azim_average_cl2d(nl2d/V, l2d, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)[1] for nl2d in nl2ds_spitzer_noise_ciber_map]

            if compute_spitzer_FW:
                # use fft objects to get running estimate of 2D PS and Fourier weights
                count, mean_nl2d, M2_nl2d = update_meanvar(count, mean_nl2d, M2_nl2d, nl2ds_spitzer_noise_ciber_map/V/cbps.cal_facs[inst])

            all_nl1ds_cross_ciber.extend(nl1ds_cross_ciber)

        all_nl1ds_cross_ciber = np.array(all_nl1ds_cross_ciber)
        
        all_nl1ds_cross_ciber_both.append(all_nl1ds_cross_ciber)

        if compute_spitzer_FW:
            mean_nl2d, var_nl2d, svar_nl2d = finalize_meanvar(count, mean_nl2d, M2_nl2d)
            fourier_weights = 1./var_nl2d
            plot_map(fourier_weights, title='Fourier weights')
            all_fourier_weights.append(fourier_weights)

        if save:
            if add_str is not None:
                if bootes_fieldidx == 0:
                    add_str = '_'+add_str
            else:
                add_str = ''

            print('add string:', add_str)
            
            nl_save_fpath = base_path+'nl1ds_TM'+str(inst)+'_ifield'+str(bootes_ifield)+'_irac_ch'+str(irac_ch)+'_spitzer_noise_ciber_map'+add_str+'.npz'

            all_nl_save_fpaths.append(nl_save_fpath)

            np.savez(nl_save_fpath, fourier_weights_spitzer=fourier_weights, \
                 all_nl1ds_spitzer=all_nl1ds_spitzer, all_nl1ds_cross_ciber=all_nl1ds_cross_ciber, lb=lb, \
                 var_nl2d=var_nl2d, mean_nl2d=mean_nl2d)

    if save:
        return all_nl_save_fpaths, lb, all_nl1ds_cross_ciber
    else:
        return lb, all_nl1ds_cross_ciber_both

def estimate_ciber_noise_cross_spitzer(cbps, irac_ch, inst, bootes_spitzer_maps, combined_masks, spitzer_string, bootes_ifield_list=[6, 7], bootes_fieldidxs=[2,3],\
                                       nsims=200, n_split=10, plot=True, read_noise=True, photon_noise=False, \
                                       include_ff_errors=True, observed_run_name=None, nmc_ff=10, \
                                      base_path=None, datestr='111323', \
                                       ff_est_dirpath=None, save=True, add_str=None, n_FF_realiz=10):

    if base_path is None:
        base_path = config.ciber_basepath+'data/Spitzer/cross_noise/ciber_noise_spitzer_map/'
    
    
    if ff_est_dirpath is None:
#         ff_est_dirpath = 'data/ff_mc_ests/112022/TM'+str(inst)+'/'
        ff_est_dirpath = config.ciber_basepath+'data/ff_mc_ests/'+datestr+'/TM'+str(inst)+'/'

    ''' feed in bootes maps for CIBER and their maps'''
    maplist_split_shape = (nsims//n_split, cbps.dimx, cbps.dimy)
    empty_aligned_objs, fft_objs = construct_pyfftw_objs(3, maplist_split_shape)
    empty_aligned_objs_maps, fft_objs_maps = construct_pyfftw_objs(3, (1, maplist_split_shape[1], maplist_split_shape[2]))

    sterad_per_pix = (cbps.pixsize/3600/180*np.pi)**2
    V = cbps.dimx*cbps.dimy*sterad_per_pix
    l2d = get_l2d(cbps.dimx, cbps.dimy, cbps.pixsize)

    lb = cbps.Mkk_obj.midbin_ell
    all_nl1ds_cross_spitzer_both, nl_save_fpath_both = [], []
    
    ciber_bootes_noise_models = cbps.grab_noise_model_set(bootes_ifield_list, inst, noise_modl_type='quadsub_021523')
    
    all_ff_ests_nofluc_both, simmap_dc = None, None
    # if include_ff_errors and observed_run_name is not None:
    if include_ff_errors or photon_noise:
        if observed_run_name is not None:
            all_ff_ests_nofluc_both = np.zeros((2, n_FF_realiz, cbps.dimx, cbps.dimy))
            simmap_dc = np.zeros((2))
            for idx, bootes_ifield in enumerate(bootes_ifield_list):
                # load the MC FF estimates obtained by several draws of noise
                if include_ff_errors:
                    all_ff_ests_nofluc_both[idx] = cbps.collect_ff_realiz_simp(bootes_fieldidxs[idx], inst, observed_run_name, n_FF_realiz, datestr=datestr)
                
                simmap_dc[idx] = cbps.zl_levels_ciber_fields[inst][cbps.ciber_field_dict[bootes_ifield]]

            print("simmap dc:", simmap_dc)
            
        else:
            print('need observed run name')
            return None

    for bootes_fieldidx, bootes_ifield in enumerate(bootes_ifield_list):

        mean_nl2d, M2_nl2d = [np.zeros_like(ciber_bootes_noise_models[0]) for x in range(2)]
        count = 0

        if photon_noise:
            field_nfr = cbps.field_nfrs[bootes_ifield]
            print('field nfr for '+str(bootes_ifield)+' is '+str(field_nfr))

            shot_sigma_sb = cbps.compute_shot_sigma_map(inst, image=simmap_dc[bootes_fieldidx]*np.ones_like(ciber_bootes_noise_models[0]), nfr=field_nfr)
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

            ciber_noise_realiz, snmaps = cbps.noise_model_realization(inst, maplist_split_shape, ciber_bootes_noise_models[bootes_fieldidx], fft_obj=fft_objs[0],\
                                                  read_noise=read_noise, photon_noise=photon_noise, shot_sigma_sb=shot_sigma_sb, adu_to_sb=True)

            if photon_noise:
                print('adding photon noise to read noise realizations')
                ciber_noise_realiz += snmaps

            if i==0 and plot:
                plot_map(ciber_noise_realiz[0])
                plot_map(gaussian_filter(ciber_noise_realiz[0], sigma=10))

            if all_ff_ests_nofluc_both is not None:
                ciber_noise_realiz += simmap_dc[bootes_fieldidx]
                ciber_noise_realiz /= all_ff_ests_nofluc_both[bootes_fieldidx][i%n_FF_realiz]

                if i==0:
                    plot_map(ciber_noise_realiz[0], title='added normalization and flat field error')

            unmasked_means = [np.mean(simmap[combined_mask_indiv==1]) for simmap in ciber_noise_realiz]
            ciber_noise_realiz -= np.array([combined_mask_indiv*unmasked_mean for unmasked_mean in unmasked_means])

            fft_objs[1](combined_mask_indiv*ciber_noise_realiz*sterad_per_pix)

            nl2ds_ciber_noise_spitzer_map = np.array([fftshift(dentry*np.conj(empty_aligned_objs_maps[2][0])).real for dentry in empty_aligned_objs[2]])
            nl1ds_cross_spitzer = [azim_average_cl2d(nl2d/V, l2d, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)[1] for nl2d in nl2ds_ciber_noise_spitzer_map]
            count, mean_nl2d, M2_nl2d = update_meanvar(count, mean_nl2d, M2_nl2d, nl2ds_ciber_noise_spitzer_map/V)

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
                
            nl_save_fpath = base_path+'nl1ds_TM'+str(inst)+'_ifield'+str(bootes_ifield)+'_irac_ch'+str(irac_ch)+'_ciber_noise_spitzer_map'+add_str+'.npz'
            
            nl_save_fpath_both.append(nl_save_fpath)
            print('saving Nl realizations to ', nl_save_fpath)
            np.savez(nl_save_fpath, all_nl1ds_cross_spitzer=all_nl1ds_cross_spitzer, lb=lb, var_nl2d=var_nl2d, mean_nl2d=mean_nl2d)

    if save:
        return nl_save_fpath_both, lb, all_nl1ds_cross_spitzer_both
    else:
        return lb, all_nl1ds_cross_spitzer_both


def estimate_spitzer_cross_auto_uncertainty(inst, nl2d_diff, combined_mask, \
                                    nsims=500, n_split=10, cbps=None, cross_auto=False, map_A=None, map_B=None):
    
    if cbps is None:
        cbps = CIBER_PS_pipeline()
    
    maplist_split_shape = (nsims//n_split, cbps.dimx, cbps.dimy)
    empty_aligned_objs, fft_objs = construct_pyfftw_objs(3, maplist_split_shape)
    
    sterad_per_pix = (cbps.pixsize/3600/180*np.pi)**2
    V = cbps.dimx*cbps.dimy*sterad_per_pix
    count = 0

    mean_nl2d = np.zeros(nl2d_diff.shape)
    M2_nl2d = np.zeros(nl2d_diff.shape)
    
    
    for i in range(n_split):
        print('Split '+str(i+1)+' of '+str(n_split)+'..')
        if i==0:
            plot_map(nl2d_diff, title='fft2 bootes')
            plot_map(np.log10(np.abs(nl2d_diff)), title='log abs fft2 bootes')

        spitzer_noise_realiz, _ = cbps.noise_model_realization(inst, maplist_split_shape, nl2d_diff, fft_obj=fft_objs[0],\
                                              read_noise=True, photon_noise=False, adu_to_sb=False, chisq=False)

        spitzer_noise_realiz /= cbps.arcsec_pp_to_radian
        
        fft_objs[1](spitzer_noise_realiz*sterad_per_pix)
        nl2ds = np.array([fftshift(dentry*np.conj(dentry)).real for dentry in empty_aligned_objs[2]])
        count, mean_nl2d, M2_nl2d = update_meanvar(count, mean_nl2d, M2_nl2d, nl2ds/V/cbps.cal_facs[inst]**2)
    
    mean_nl2d, var_nl2d, svar_nl2d = finalize_meanvar(count, mean_nl2d, M2_nl2d)
    spitz_fourier_weights = 1./var_nl2d
    plot_map(spitz_fourier_weights, title='Fourier weights')
    
    return spitz_fourier_weights, var_nl2d


def estimate_spitzer_fourier_weights(inst, nl2d_diff, combined_mask, \
                                    nsims=500, n_split=10, cbps=None, cross_auto=False, maps_A=None, maps_B=None):
    
    if cbps is None:
        cbps = CIBER_PS_pipeline()
    
    maplist_split_shape = (nsims//n_split, cbps.dimx, cbps.dimy)
    empty_aligned_objs, fft_objs = construct_pyfftw_objs(3, maplist_split_shape)
    
    sterad_per_pix = (cbps.pixsize/3600/180*np.pi)**2
    V = cbps.dimx*cbps.dimy*sterad_per_pix
    count = 0

    mean_nl2d = np.zeros(nl2d_diff.shape)
    M2_nl2d = np.zeros(nl2d_diff.shape)
    
    
    for i in range(n_split):
        print('Split '+str(i+1)+' of '+str(n_split)+'..')
        if i==0:
            plot_map(nl2d_diff, title='fft2 bootes')
            plot_map(np.log10(np.abs(nl2d_diff)), title='log abs fft2 bootes')

        spitzer_noise_realiz, _ = cbps.noise_model_realization(inst, maplist_split_shape, nl2d_diff, fft_obj=fft_objs[0],\
                                              read_noise=True, photon_noise=False, adu_to_sb=False, chisq=False)

        spitzer_noise_realiz /= cbps.arcsec_pp_to_radian
        
        fft_objs[1](spitzer_noise_realiz*sterad_per_pix)
        nl2ds = np.array([fftshift(dentry*np.conj(dentry)).real for dentry in empty_aligned_objs[2]])
        count, mean_nl2d, M2_nl2d = update_meanvar(count, mean_nl2d, M2_nl2d, nl2ds/V)
    
    mean_nl2d, var_nl2d, svar_nl2d = finalize_meanvar(count, mean_nl2d, M2_nl2d)
    spitz_fourier_weights = 1./var_nl2d
    plot_map(spitz_fourier_weights, title='Fourier weights')
    
    return spitz_fourier_weights, var_nl2d


def compute_spitzer_auto(inst, irac_ch, bootes_ifields, ciber_mask_tail, spitzer_string, \
                        mask_string_mkk, compute_mkk=False, n_mkk_sims=200, n_mkk_split=4, pair_idxs=[[0, 1], [2, 3]], \
                        apply_FW=False, compute_FW=False, n_FW_sims=500, n_FW_split=10):
    
    cbps = CIBER_PS_pipeline()

    all_cl_spitzer, all_dcl_spitzer, all_std_nl1ds_nAnB, all_std_nl1ds_nAsB, all_std_nl1ds_nBsA = [[] for x in range(5)]
    bootes_ciber_masks = []
    # load ciber masks
        
    for idx, ifield in enumerate(bootes_ifields):
        mask_fpath = 'data/fluctuation_data/TM'+str(inst)+'/masks/'+ciber_mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+ciber_mask_tail+'.fits'

        full_mask = fits.open(mask_fpath)[1].data.astype(np.int)    
        bootes_ciber_masks.append(full_mask)
    
    spitzer_regrid_maps, spitzer_regrid_masks, all_diff1, all_diff2, all_spitz_by_epoch = load_spitzer_bootes_maps(inst, irac_ch, bootes_ifields, spitzer_string)
    combined_masks = [bootes_ciber_masks[idx]*spitzer_regrid_mask for idx, spitzer_regrid_mask in enumerate(spitzer_regrid_masks)]
    
    irac_lb, irac_bl = load_irac_bl(irac_ch)
    
    lb = cbps.Mkk_obj.midbin_ell
    
    interp_maskfn = scipy.interpolate.interp1d(np.log10(irac_lb), np.log10(irac_bl))
    
    cibermatch_bl = 10**interp_maskfn(np.log10(lb))

    plt.figure()
    plt.scatter(irac_lb, irac_bl)
    plt.plot(irac_lb, 10**interp_maskfn(np.log10(irac_lb)), color='r')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('IRAC CH'+str(irac_ch)+' beam')
    plt.xlabel('$\\ell$')
    plt.ylabel('$B_{\\ell}$')
    plt.xlim(100, 1e5)
    plt.ylim(1e-2, 1e0)
    plt.show()
    
    
    for idx, ifield in enumerate(bootes_ifields):
        
        print('unmasked fraction for ifield', ifield,' is ', np.sum(combined_masks[idx])/float(1024**2))
        plot_map(combined_masks[idx])

        if compute_mkk:
            Mkk = cbps.Mkk_obj.get_mkk_sim(combined_masks[idx], n_mkk_sims, n_split=n_mkk_split, store_Mkks=False)
            inv_Mkk = compute_inverse_mkk(Mkk)
            plot_mkk_matrix(inv_Mkk, inverse=True, symlogscale=True)
            hdul = write_Mkk_fits(Mkk, inv_Mkk, ifield, inst, \
                                 use_inst_mask=True, dat_type='ciber_spitzer_cross', mag_lim_AB=masking_maglim)

            mkkpath = 'data/Spitzer/spitzer_ciber_mkk/mkk_maskonly_estimate_ifield'+str(ifield)+'_inst'+str(inst)+'_spitzunion_IRAC_CH'+str(irac_ch)+'_'+mask_string_mkk+'.fits'
            hdul.writeto(mkkpath, overwrite=True)
        else:
#             mkkpath = 'data/Spitzer/spitzer_ciber_mkk/mkk_maskonly_estimate_ifield'+str(ifield)+'_inst'+str(inst)+'_spitzunion_IRAC_CH'+str(irac_ch)+'_'+mask_string_mkk+'.fits'
            mkkpath = 'data/Spitzer/spitzer_ciber_mkk/mkk_maskonly_estimate_ifield'+str(ifield)+'_inst'+str(inst)+'_spitzunion_'+mask_string_mkk+'_IRAC_CH'+str(irac_ch)+'.fits'

            inv_Mkk = fits.open(mkkpath)['inv_Mkk_'+str(ifield)].data
            plot_mkk_matrix(inv_Mkk, inverse=True, symlogscale=True)


        spitz_A = all_spitz_by_epoch[idx][0]*combined_masks[idx]
        spitz_B = all_spitz_by_epoch[idx][1]*combined_masks[idx]
        spitz_C = all_spitz_by_epoch[idx][2]*combined_masks[idx]
        spitz_D = all_spitz_by_epoch[idx][3]*combined_masks[idx]
        
        all_spitz_maps = [spitz_A, spitz_B, spitz_C, spitz_D]

        for epidx, spitz_map in enumerate(all_spitz_maps):
            spitz_map[spitz_map != 0] -= np.mean(spitz_map[spitz_map != 0])

#         lbm, cl_spitzer_masked_AB, clerr_spitz_AB = get_power_spec(spitz_A, map_b=spitz_B, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
#         lbm, cl_spitzer_masked_CD, clerr_spitz_CD = get_power_spec(spitz_C, map_b=spitz_C, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
        
    
        spitz_diff_1 = all_spitz_maps[pair_idxs[0][0]] - all_spitz_maps[pair_idxs[0][1]]
        spitz_diff_2 = all_spitz_maps[pair_idxs[1][0]] - all_spitz_maps[pair_idxs[1][1]]
        
        l2d, nl2d_diff = compute_nl2d_spitzer_crossepoch_diffs(spitz_diff_1, spitz_diff_2, combined_masks[idx])
        
        # scale nl2d_diff to pair average
        nl2d_diff /= 4.
        
        # mkk correct
        nl2d_diff_orig = nl2d_diff.copy()

        print('inv mkk diagonal is ', [inv_Mkk.transpose()[x, x] for x in range(inv_Mkk.shape[0])])

        for bandidx in range(inv_Mkk.shape[0]):
            lmin, lmax = cbps.Mkk_obj.binl[bandidx], cbps.Mkk_obj.binl[bandidx+1]
            sp = np.where((l2d>=lmin) & (l2d<lmax))
            nl2d_diff[sp] *= inv_Mkk.transpose()[bandidx, bandidx]

        plot_map(nl2d_diff/nl2d_diff_orig, title='mkk corrected / original')
        
        plot_map(nl2d_diff, title='nl2d diff 1-2 x 3-4')
        
        nl2d_diff = np.abs(nl2d_diff)
        
        if compute_FW and apply_FW:
#             spitz_FW = estimate_spitzer_fourier_weights(inst, nl2d_diff, combined_masks[idx], nsims=n_FW_sims, n_split=n_FW_split)
            
            fourier_weights, mean_nl2d_nAnB, mean_nl2d_nAsB,\
                var_nl2d_nAsB, mean_nl2d_nBsA, var_nl2d_nBsA, \
                    nl1ds_nAnB, nl1ds_nAsB, nl1ds_nBsA= cbps.estimate_cross_noise_ps(inst, inst, ifield, \
                                                                                           nsims=n_FW_sims, n_split=n_FW_split, \
                                                                                           mask=combined_masks[idx], apply_mask=True, \
                                                                                           read_noise=True, noise_model=nl2d_diff, cross_noise_model=nl2d_diff, \
                                                                                           photon_noise=False, image=all_spitz_maps[pair_idxs[0][0]], image_cross=all_spitz_maps[pair_idxs[0][1]],\
                                                                                           inplace=False, show=True, spitzer=True)
            
            
            std_nl1ds_nAnB = np.std(nl1ds_nAnB, axis=0)
            std_nl1ds_nAsB = np.std(nl1ds_nAsB, axis=0)
            std_nl1ds_nBsA = np.std(nl1ds_nBsA, axis=0)
            
            
#             mean_nl2d_cross_total = mean_cl2d_cross + mean_nl2d_nAsB + mean_nl2d_nBsA
            mean_nl2d_cross_total = mean_nl2d_nAsB + mean_nl2d_nBsA
    
            plot_map(mean_nl2d_cross_total, title='mean nl2d cross total')
            lb, N_ell_est, N_ell_err = cbps.compute_noise_power_spectrum(inst, noise_Cl2D=mean_nl2d_cross_total.copy(), apply_FW=False, inplace=False, stderr=False)

            spitz_FW=None
            
#             spitz_FW = 1./((1./fourier_weights_cross) + var_nl2d_nAsB + var_nl2d_nBsA)

#             plot_map(mean_nl2d_cross_total, title='mean_nl2d_cross_total')
#             plot_map(spitz_FW, title='fourier_weights_cross_total')

#             lb, N_ell_est, N_ell_err = cbps.compute_noise_power_spectrum(inst, noise_Cl2D=mean_nl2d_cross_total.copy(), inplace=False, apply_FW=pscb_dict['apply_FW'], weights=spitz_FW)

#             spitz_FW_AB, var_nl2d_AB = estimate_spitzer_cross_auto_uncertainty(inst, nl2d_diff, combined_masks[idx], nsims=n_FW_sims, n_split=n_FW_split, \
#                                                                  map_A=all_spitz_maps[pair_idxs[0][0]], map_B=all_spitz_maps[pair_idxs[0][1]])
            
            np.savez('data/Spitzer/spitzer_auto_FW/spitzer_auto_FW_ifield'+str(ifield)+'_inst'+str(inst)+'_'+mask_string_mkk+'.npz', \
                    spitz_FW=spitz_FW, N_ell_est=N_ell_est, N_ell_err=N_ell_err, std_nl1ds_nAnB=std_nl1ds_nAnB, \
                    std_nl1ds_nAsB=std_nl1ds_nAsB, std_nl1ds_nBsA=std_nl1ds_nBsA)
            
        else:
            spitz_FW_file = np.load('data/Spitzer/spitzer_auto_FW/spitzer_auto_FW_ifield'+str(ifield)+'_inst'+str(inst)+'_'+mask_string_mkk+'.npz')
            N_ell_est=spitz_FW_file['N_ell_est']
            N_ell_err=spitz_FW_file['N_ell_err']
            std_nl1ds_nAnB = spitz_FW_file['std_nl1ds_nAnB']
            std_nl1ds_nAsB = spitz_FW_file['std_nl1ds_nAsB']
            std_nl1ds_nBsA = spitz_FW_file['std_nl1ds_nBsA']

            spitz_FW = None

            
                
        all_std_nl1ds_nAnB.append(std_nl1ds_nAnB)
        all_std_nl1ds_nAsB.append(std_nl1ds_nAsB)
        all_std_nl1ds_nBsA.append(std_nl1ds_nBsA)
        
        plot_map(nl2d_diff, title='spitzer difference 2d power spectrum')

        avmap_AB = 0.5*(all_spitz_maps[pair_idxs[0][0]]+all_spitz_maps[pair_idxs[0][1]])
        avmap_CD = 0.5*(all_spitz_maps[pair_idxs[1][0]]+all_spitz_maps[pair_idxs[1][1]])
        
        plot_map(avmap_AB, title='avmap_AB')
        plot_map(avmap_CD, title='avmap_CD')
        
        print('avmap AB has mean of ', np.mean(avmap_AB))
        print('avmap CD has mean of ', np.mean(avmap_CD))
        
        lbm, cl_spitzer_masked_AB_CD, clerr_spitz_AB_CD = get_power_spec(avmap_AB, map_b=avmap_CD,\
                                                                   lbinedges=cbps.Mkk_obj.binl, lbins=lb, \
                                                                  weights=None)        
        
        
        print('lbm:', lbm)
        prefac = lb*(lb+1)/(2*np.pi)
        
        average_cl_spitzer_auto = cl_spitzer_masked_AB_CD

        average_cl_spitzer_auto_orig = average_cl_spitzer_auto.copy()
        
        average_cl_spitzer_auto = np.dot(inv_Mkk.transpose(), average_cl_spitzer_auto)
        
        average_dcl_spitzer_auto = np.sqrt(std_nl1ds_nAnB**2+std_nl1ds_nAsB**2 + std_nl1ds_nBsA**2)


        print('Correcting for filtering transfer function..')
        t_ell_av = np.load('data/transfer_function/t_ell_est_nsims=100.npz')['t_ell_av']
        print('t_ell av:', t_ell_av)
        average_cl_spitzer_auto /= t_ell_av
        average_dcl_spitzer_auto /= t_ell_av

        average_cl_spitzer_auto /= cibermatch_bl**2 # correcting by IRAC beam
        average_dcl_spitzer_auto /= cibermatch_bl**2 # correcting by IRAC beam
        
        auto_knox_errors = np.sqrt(1./((2*lb+1)*cbps.Mkk_obj.delta_ell)) # factor of 1 in numerator since auto is a cross-epoch cross
        fsky = 2*2/(41253.)    
        auto_knox_errors /= np.sqrt(fsky)
        auto_knox_errors *= np.abs(average_cl_spitzer_auto)
        
#         average_dcl_spitzer_auto = np.sqrt(average_dcl_spitzer_auto**2 + auto_knox_errors**2)
        
        plt.figure(figsize=(6,5))
#             plt.errorbar(lbm, prefac*average_cl_spitzer_auto_orig, yerr=prefac*average_dcl_spitzer_auto, fmt='o', capsize=4, color='g', label='Spitzer cross epoch auto (pre noise sub)')
        plt.errorbar(lb, prefac*average_cl_spitzer_auto, yerr=prefac*average_dcl_spitzer_auto, fmt='o', capsize=4, color='r', label='Spitzer cross epoch auto')
#         plt.plot(lbm, prefac*av_nl1ds_spitzer, linestyle='dashed', color='r', label='Spitzer auto noise bias')
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('$\\ell$', fontsize=14)
        plt.ylabel('$D_{\\ell}$', fontsize=14)
        plt.tick_params(labelsize=14)
        plt.ylim(1e-5, 1e1)
        plt.grid()
        plt.show()
        
        all_cl_spitzer.append(average_cl_spitzer_auto)
        all_dcl_spitzer.append(average_dcl_spitzer_auto)
        
    return lbm, all_cl_spitzer, all_dcl_spitzer, all_std_nl1ds_nAnB, all_std_nl1ds_nAsB, all_std_nl1ds_nBsA



