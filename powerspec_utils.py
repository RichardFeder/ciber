import numpy as np
import astropy.io.fits as fits
import astropy.wcs as wcs
import pandas as pd
import scipy.io
from scipy.ndimage import gaussian_filter
import matplotlib
import matplotlib.pyplot as plt
from astropy.table import Table

from mkk_parallel import *
# from ciber_powerspec_pipeline import *
# from ciber_powerspec_pipeline import CIBER_PS_pipeline



def compute_sqrt_cl_errors(cl, dcl):
    sqrt_cl_errors = 0.5*(dcl*cl**(-0.5))
    return sqrt_cl_errors


class powerspec():

	def __init__(self, n_ps_bin=25, dimx=1024, \
				dimy=1024):

		self.Mkk_obj = Mkk_bare(dimx=dimx, dimy=dimy, ell_min=180.*(1024./dimx), nbins=n_ps_bin)


	def load_in_powerspec(self, powerspec_fpath, mode=None, inplace=False):

		# if inplace:
		# 	if mode==''
		# 	self.ps = powerspec

		# else:
		# 	return powerspec

		pass


	def compute_est_true_ratio(self, ps_measured=None, ps_ground_truth=None, inplace=True):

		''' Computes ratio between power spectra 

		Inputs
		------
		
		ps_measured (optional): np.array of shape (nfields, n_ps_bin), (nsims, n_ps_bin) or (nsims, nfields, n_ps_bin). 
								Default is None.

		ps_ground_truth (optional): np.array of shape (nfields, n_ps_bin), (nsims, n_ps_bin) or (nsims, nfields, n_ps_bin). 
								Default is None.	

		inplace : type 'boolean'. If True, stores power spectrum ratio in object. Default is True. 	

		Returns
		-------

		est_true_ratio : np.array of shape (nfields, n_ps_bin), (nsims, n_ps_bin) or (nsims, nfields, n_ps_bin). 
						Ratio of (sets of) power spectra.

		'''
		if ps_measured is None:
			ps_measured = self.ps_measured
		if ps_ground_truth is None:
			ps_ground_truth = self.ps_ground_truth

		est_true_ratio = ps_measured/ps_ground_truth

		if inplace:
			self.est_true_ratio = est_true_ratio
		else:
			return est_true_ratio


	def grab_all_simidx_dat(self, fpaths, mean_or_median='mean', inter_idx=None, per_field=True, bls=None, inplace=True):

		nsims = len(fpaths)

		# here, "mean" power spectrum refers to average over fields
		est_true_ratios = np.zeros((nsims, nfield, self.n_ps_bin))
		all_mean_true_cls = np.zeros((nsims, self.n_ps_bin))
		all_true_cls = np.zeros((nsims, nfield, self.n_ps_bin))
		all_recovered_cls = np.zeros((nsims, nfield, self.n_ps_bin))
		all_mean_recovered_cls = np.zeros((nsims, self.n_ps_bin))


		for f, fpath in enumerate(fpaths):
			lb, recovered_cls, true_cls, mean_recovered_cls, mean_true_cl, est_true_ratio = grab_recovered_cl_dat(fpath, mean_or_median=mean_or_median, inter_idx=inter_idx, \
																												  per_field=per_field, bls=bls)

			est_true_ratios[f] = est_true_ratio
			all_recovered_cls[f] = recovered_cls
			all_true_cls[f] = true_cls
			all_mean_recovered_cls[f] = mean_recovered_cls

		if inplace:
			self.est_true_ratios = est_true_ratios
			self.all_mean_true_cls = all_mean_true_cls
			self.all_recovered_cls = all_recovered_cls
			self.all_true_cls = all_true_cls
			self.lb = lb 

		else:
			return lb, est_true_ratios, all_mean_true_cls, all_recovered_cls, all_recovered_mean_cls, all_true_cls



def compute_field_averaged_power_spectrum(per_field_cls, per_field_dcls=None, per_field_cl_weights=None, weight_mode='invvar', \
                                         which_idxs=None, verbose=False):
    
    '''
    Inputs
    ------
    
    lb : np.array or list of multipole bin centers
    per_field_cls : np.array of shape (nfields, nbins)
    per_field_dcls (optional) : np.array of shape (nfields, nbins)
    per_field_cl_weights (optional) : np.array of shape (nfields, nbins)
    weight_mode (optional) : 'str'. If set to 'invvar', computes inverse variance weights using per_field_dcls.
            Default is 'invvar'.
    which_idxs (optional) : 
    verbose (optional) : 'bool'. 
    
    Returns
    -------
    
    field_averaged_cl :
    field_averaged_std :
    cl_sumweights : 
    per_field_cl_weights : 
    
    '''
    
    if which_idxs is not None:
        which_idxs = np.array(which_idxs)
        per_field_cls = per_field_cls[which_idxs,:]
        if per_field_dcls is not None:
            per_field_dcls = per_field_dcls[which_idxs,:]
    
    if per_field_cl_weights is None:
        per_field_cl_weights = np.zeros_like(per_field_cls)
    
        if weight_mode=='invvar':
            if verbose:
                print(per_field_cl_weights.shape, per_field_cls.shape)
            per_field_cl_weights = per_field_dcls**(-2)
            per_field_cl_weights[per_field_cls < 0.] = 0.
                
    else:
        if which_idxs is not None:
            per_field_cl_weights = per_field_cl_weights[which_idxs,:]
        
    cl_sumweights = np.sum(per_field_cl_weights, axis=0)
    if verbose:
        print("per field cl weights ", per_field_cl_weights)
        print('cl_sumweights has shape', cl_sumweights.shape)
        print('cl_sumweights:', cl_sumweights)
    
    weighted_variance = 1./cl_sumweights
    
    field_averaged_std = np.sqrt(weighted_variance)
    
    per_field_cls[per_field_cls < 0] = 0
    
    nfields_per_bin = np.count_nonzero(per_field_cls, axis=0)
    
    per_field_cls[per_field_cls == 0] = np.nan
    field_averaged_cl = np.nansum(per_field_cl_weights*per_field_cls, axis=0)/cl_sumweights

    if verbose:
        print('at this step')
        print(per_field_cl_weights*per_field_cls)
        print('field_averaged_cl is ', field_averaged_cl)
        print('field averaged std is', field_averaged_std)
    
    return field_averaged_cl, field_averaged_std, cl_sumweights, per_field_cl_weights


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


def get_observed_power_spectra(est_fpath=None, recov_obs_fpath=None, recov_mock_fpath=None,\
                               recovered_cls_obs=None, est_true_ratios=None, \
                               all_recovered_cls=None, mean_recovered_cls=None, std_recovered_cls=None, \
                               apply_correction=True, skip_elat30=False, verbose=False, recover_errs=True):
    if est_fpath is not None:
        est_true_ratios = np.load(est_fpath)['est_true_ratios']
    if recov_obs_fpath is not None:
        recov_observed = np.load(recov_obs_fpath)
        recovered_cls_obs = recov_observed['recovered_cls']
        if recover_errs:
            recovered_dcls_obs = recov_observed['recovered_dcls']
        
    if recov_mock_fpath is not None:
        recov_full = np.load(recov_mock_fpath)
        all_recovered_cls = recov_full['all_recovered_cls']
        mean_recovered_cls = recov_full['mean_recovered_cls']
        std_recovered_cls = recov_full['std_recovered_cls']
    
    recovered_cls_obs_corrected = np.zeros_like(recovered_cls_obs)
    recovered_dcls_obs_corrected = np.zeros_like(recovered_cls_obs)

    if skip_elat30:
        which_idxs = [0, 2, 3, 4]
    else:
        which_idxs = np.arange(5)

    print('which idxs is ', which_idxs)
        
    for i in np.arange(5):
                
        recovered_cls_obs_corrected[i] = recovered_cls_obs[i]
        recovered_dcls_obs_corrected[i] = std_recovered_cls[i]
        if apply_correction:
            print(np.mean(est_true_ratios[:,i,:], axis=0))
            recovered_cls_obs_corrected[i] /= np.mean(est_true_ratios[:,i,:], axis=0)
            recovered_dcls_obs_corrected[i] /= np.mean(est_true_ratios[:,i,:], axis=0)
            

    field_averaged_cl, field_averaged_std, cl_sumweights, per_field_cl_weights = compute_field_averaged_power_spectrum(recovered_cls_obs_corrected, per_field_dcls=recovered_dcls_obs_corrected, \
                                                                                                                      which_idxs=which_idxs, verbose=verbose)
    recovered_cls_obs_corrected[recovered_cls_obs_corrected < 0] = np.nan
    std_cls_basic = np.nanstd(recovered_cls_obs_corrected, axis=0)
    
    return field_averaged_cl, field_averaged_std, cl_sumweights, per_field_cl_weights, recovered_cls_obs_corrected, recovered_dcls_obs_corrected, std_cls_basic, recovered_dcls_obs
        
 

def compute_flatfield_prods(ifield_list, inst, observed_ims, joint_masks, cbps, show_plots=False, ff_stack_min=1, \
                           inv_var_weight=True, field_nfrs=None, ff_weights=None):

    
    ff_estimates = np.zeros((len(ifield_list), cbps.dimx, cbps.dimy))
    ff_joint_masks = np.zeros((len(ifield_list), cbps.dimx, cbps.dimy))
    
    if weights is not None:
        inv_var_weight = False

    for i, ifield in enumerate(ifield_list):
        stack_obs = list(observed_ims.copy())
        stack_mask = list(joint_masks.copy().astype(np.bool))

        if weights is not None:
            ff_weights = list(np.array(weights).copy())
            del(ff_weights[i])
        else:
            ff_weights = None

        if field_nfrs is not None:
            ff_field_nfrs = list(np.array(field_nfrs).copy())
        elif cbps.field_nfrs is not None:
            ff_field_nfrs = list(cbps.field_nfrs.copy())
            del(ff_field_nfrs[imidx])
        else:
            ff_field_nfrs = None

        del(stack_obs[i])
        del(stack_mask[i])

        ff_estimate, ff_mask, ff_weights = compute_stack_ff_estimate(stack_obs, target_mask=joint_masks[i], masks=stack_mask, means=None, inv_var_weight=inv_var_weight, ff_stack_min=ff_stack_min, \
                                                                    field_nfrs=ff_field_nfrs, weights=ff_weights)
        ff_estimates[i] = ff_estimate

        sum_stack_mask = np.sum(stack_mask, axis=0)
        
        
        ff_joint_masks[i] = joint_masks[i]*ff_mask

        if show_plots:
            plot_map(ff_estimate)
            sumstackfig = plot_map(sum_stack_mask, title='sum stack mask')
            
    return ff_estimates, ff_joint_masks

def get_mock_result_powerspec(fpaths, true_cl_fpaths=None, J_mag_lim=17.5, ifield_list=[4, 5, 6, 7, 8]):
    
    all_est_true_ratios, all_mean_true_cls, all_true_cls, all_recovered_cls, all_recovered_mean_cls, all_std_recovered_cls = [[] for x in range(6)]
    
    for i, fpath in enumerate(fpaths):
        est_true_ratios = []
        cls_true = []
        lb, recovered_cls, true_cls, _, _, est_true_ratio = grab_recovered_cl_dat(fpath)
        
        if true_cl_fpaths is not None:
            
            for j, ifield in enumerate(ifield_list):
                cl_true = fits.open(true_cl_fpaths[i][j])['cls_cib'].data['cl_maglim_'+str(J_mag_lim)]
#                 print('cl_true has shape', cl_true.shape)
                cls_true.append(cl_true)
                est_true_ratios.append(recovered_cls[j]/cl_true)
            all_mean_true_cls.append(cls_true)
            all_est_true_ratios.append(est_true_ratios)
        else:
            all_est_true_ratios.append(est_true_ratio)
            # all_mean_true_cls.append(true_cls)
            
        all_recovered_cls.append(recovered_cls)
        all_true_cls.append(cls_true)

    all_recovered_cls = np.array(all_recovered_cls)
    all_recovered_mean_cls = np.mean(all_recovered_cls, axis=0)
    all_recovered_std_cls = np.std(all_recovered_cls, axis=0)
    all_true_cls = np.array(all_true_cls)

    all_est_true_ratios = np.array(all_est_true_ratios)
    print('all_recovered_mean cls has shape ', all_recovered_mean_cls.shape)
    print('all_recovered cls has shape ', all_recovered_cls.shape)
    print('all est true ratios ', all_est_true_ratios.shape)
    
    return lb, all_recovered_cls, all_recovered_mean_cls, all_recovered_std_cls, all_est_true_ratios, all_true_cls



def grab_recovered_cl_dat(fpath, mean_or_median='mean', inter_idx=None, per_field=True, bls=None):
    
    ff_test_dat = np.load(fpath, allow_pickle=True)
    
    lb = ff_test_dat['lb']
    
    if inter_idx is not None:
        recovered_cls = [ff_test_dat['cls_intermediate'][j][inter_idx] for j in range(len(ff_test_dat['cls_intermediate']))]
        if recovered_cls[0] is None:
            recovered_cls = [ff_test_dat['cls_intermediate'][j][inter_idx-1] for j in range(len(ff_test_dat['cls_intermediate']))]

    else:
        recovered_cls = ff_test_dat['recovered_cls']
            
    true_cls = ff_test_dat['true_cls']
    
    if recovered_cls is None:
        mean_recovered_cls, mean_true_cls, est_true_ratio = None, None, None                
    else:

        if per_field:   
            if bls is not None:
                true_cls = [true_cls[field_idx]/bls[field_idx]**2 for field_idx in range(recovered_cls.shape[0])]
            est_true_ratio = np.array([recovered_cls[field_idx]/true_cls[field_idx] for field_idx in range(recovered_cls.shape[0])])
            mean_recovered_cls, mean_recovered_cls, mean_true_cls = None, None, None
        else:
        
            if mean_or_median=='median':
                mean_recovered_cls = np.median(recovered_cls, axis=0)
                mean_true_cls = np.median(true_cls, axis=0)
            else:
                mean_recovered_cls = np.mean(recovered_cls, axis=0)
                mean_true_cls = np.mean(true_cls, axis=0)

            est_true_ratio = mean_recovered_cls/mean_true_cls
        

    return lb, recovered_cls, true_cls, mean_recovered_cls, mean_true_cls, est_true_ratio



def grab_all_simidx_dat(fpaths, mean_or_median='mean', inter_idx=None, per_field=True, bls=None):
    est_true_ratios, all_mean_true_cls, all_true_cls, all_recovered_cls, all_recovered_mean_cls = [], [], [], [], []
    
    for fpath in fpaths:
        lb, recovered_cls, true_cls, mean_recovered_cls, mean_true_cl, est_true_ratio = grab_recovered_cl_dat(fpath, mean_or_median=mean_or_median, inter_idx=inter_idx, \
                                                                                                              per_field=per_field, bls=bls)
        est_true_ratios.append(est_true_ratio)
        all_recovered_cls.append(recovered_cls)
        all_true_cls.append(true_cls)
        all_recovered_mean_cls.append(mean_recovered_cls)
        all_mean_true_cls.append(mean_true_cl)

    est_true_ratios = np.array(est_true_ratios)
    all_mean_true_cls = np.array(all_mean_true_cls)
    all_recovered_cls = np.array(all_recovered_cls)
    all_true_cls = np.array(all_true_cls)
    
    return lb, est_true_ratios, all_mean_true_cls, all_recovered_cls, all_recovered_mean_cls, all_true_cls


def instantiate_dat_arrays_fftest(dimx, dimy, nfields):
	''' this function initializes the arrays for the mock pipeline test '''
	observed_ims = np.zeros((nfields, dimx, dimy))
	total_signals = np.zeros((nfields, dimx, dimy))
	noise_models = np.zeros((nfields, dimx, dimy))
	diff_realizations = np.zeros((nfields, dimx, dimy))
	shot_sigma_sb_maps = np.zeros((nfields, dimx, dimy))
	rn_maps = np.zeros((nfields, dimx, dimy))
	joint_masks = np.zeros((nfields, dimx, dimy))
	
	return observed_ims, total_signals, noise_models, shot_sigma_sb_maps, rn_maps, joint_masks, diff_realizations
	
	
def instantiate_cl_arrays_fftest(nfields, n_ps_bin):
	''' this function initializes the arrays for the power spcetra computed at various points of the mock pipeline test'''
	cls_gt = np.zeros((nfields, n_ps_bin))
	cls_diffuse = np.zeros((nfields, n_ps_bin))
	cls_cib = np.zeros((nfields, n_ps_bin))
	cls_postff = np.zeros((nfields, n_ps_bin))
	B_ells = np.zeros((nfields, n_ps_bin))
	
	return cls_gt, cls_diffuse, cls_cib, cls_postff, B_ells

def lin_interp_powerspec(lb_orig, lb_interp, powerspec_orig):
	''' this function uses linear interpolation between one set of multipoles and another tot regrid an input tpower spectrum '''
	powerspec_interp = np.zeros_like(lb_interp)
	for l, lb_int in enumerate(lb_interp):
		# find nearest bins
		lb_diff = np.abs(lb_orig - lb_int)
		arglb_nearest = np.argmin(lb_diff)
		lb_nearest = lb_orig[arglb_nearest]
		
		if lb_nearest > lb_int:
			lb_upper = lb_nearest
			lb_lower = lb_orig[arglb_nearest-1]
			
			cl_lower = powerspec_orig[arglb_nearest-1]
			cl_upper = powerspec_orig[arglb_nearest]

		else:
			lb_lower = lb_nearest
			if arglb_nearest == len(lb_orig)-1:
				continue
			else:
				lb_upper = lb_orig[arglb_nearest+1]
			cl_lower = powerspec_orig[arglb_nearest]
			cl_upper = powerspec_orig[arglb_nearest+1]
			
		slope = (cl_upper-cl_lower)/(lb_upper-lb_lower)

		powerspec_interp[l] = cl_lower + slope*(lb_int-lb_lower)

	return powerspec_interp




def compute_knox_errors(lbins, C_ell, N_ell, delta_ell, fsky=None, B_ell=None, snr=False):
	
	knox_errors = np.sqrt(2./((2*lbins+1)*delta_ell))
	if fsky is not None:
		knox_errors /= np.sqrt(fsky)

	beam_fac = 1.
	if B_ell is not None:
		beam_fac = 1./B_ell**2
		
	knox_errors *= (C_ell + N_ell*beam_fac)

	if snr:
		snr = C_ell / knox_errors
		return snr

	return knox_errors



def compute_sqrt_cl_errors(cl, dcl):
	sqrt_cl_errors = 0.5*(dcl*cl**(-0.5))
	return sqrt_cl_errors



def read_ciber_powerspectra(filename):
	''' Given some file path name, this function loads/parses previously measured CIBER power spectra'''
	array = np.loadtxt(filename, skiprows=8)
	ells = array[:,0]
	norm_cl = array[:,1]
	norm_dcl_lower = array[:,2]
	norm_dcl_upper = array[:,3]
	return np.array([ells, norm_cl, norm_dcl_lower, norm_dcl_upper])


def write_Mkk_fits(Mkk, inv_Mkk, ifield, inst, sim_idx=None, generate_starmask=True, generate_galmask=True, \
                  use_inst_mask=True, dat_type=None, mag_lim_AB=None):
    hduim = fits.ImageHDU(inv_Mkk, name='inv_Mkk_'+str(ifield))        
    hdum = fits.ImageHDU(Mkk, name='Mkk_'+str(ifield))

    hdup = fits.PrimaryHDU()
    hdup.header['ifield'] = ifield
    hdup.header['inst'] = inst
    if sim_idx is not None:
        hdup.header['sim_idx'] = sim_idx
    hdup.header['generate_galmask'] = generate_galmask
    hdup.header['generate_starmask'] = generate_starmask
    hdup.header['use_inst_mask'] = use_inst_mask
    if dat_type is not None:
        hdup.header['dat_type'] = dat_type
    if mag_lim_AB is not None:
        hdup.header['mag_lim_AB'] = mag_lim_AB
        
    hdup.header['n_ps_bin'] = Mkk.shape[0]

    hdul = fits.HDUList([hdup, hdum, hduim])
    return hdul


def write_ff_file(ff_estimate, ifield, inst, sim_idx=None, dat_type=None, mag_lim_AB=None, ff_stack_min=None):
    hdum = fits.ImageHDU(ff_estimate, name='ff_'+str(ifield))
    hdup = fits.PrimaryHDU()
    hdup.header['ifield'] = ifield
    hdup.header['inst'] = inst
    
    if sim_idx is not None:
        hdup.header['sim_idx'] = sim_idx
    if dat_type is not None:
        hdup.header['dat_type'] = dat_type
    if mag_lim_AB is not None:
        hdup.header['mag_lim_AB'] = mag_lim_AB
    if ff_stack_min is not None:
        hdup.header['ff_stack_min'] = ff_stack_min

    hdul = fits.HDUList([hdup, hdum])
    
    return hdul


def write_mask_file(joint_mask, ifield, inst, sim_idx=None, generate_galmask=None, generate_starmask=None, use_inst_mask=None, \
                   dat_type=None, mag_lim_AB=None, with_ff_mask=None, name=None):

    if name is None:
        name = 'joint_mask_'+str(ifield)
    hdum = fits.ImageHDU(joint_mask, name=name)
    hdup = fits.PrimaryHDU()
    hdup.header['ifield'] = ifield
    hdup.header['inst'] = inst
    if sim_idx is not None:
        hdup.header['sim_idx'] = sim_idx
    if generate_galmask is not None:
        hdup.header['generate_galmask'] = generate_galmask
    if generate_starmask is not None:
        hdup.header['generate_starmask'] = generate_starmask
    if use_inst_mask is not None:
        hdup.header['use_inst_mask'] = use_inst_mask
    if dat_type is not None:
        hdup.header['dat_type'] = dat_type
    if mag_lim_AB is not None:
        hdup.header['mag_lim_AB'] = mag_lim_AB
    if with_ff_mask is not None:
        hdup.header['with_ff_mask'] = with_ff_mask


    hdul = fits.HDUList([hdup, hdum])
    
    return hdul




def compute_neff(weights):
    weights = np.array(weights)
    return np.sum(weights)**2/np.sum(weights**2)

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return average, np.sqrt(variance)

def compute_sim_corrected_fieldaverage(recovered_ps_by_field, input_ps_by_field, obs_idx=0, compute_field_weights=True, \
                                      apply_field_weights=True, name=None, plot=False, save_plot=False):
    
    ''' 
    Compute estimated pipeline bias from mocks, or provide, and use one realization
    to represent "observed" data. Correct observed data by mean FF bias of each field and then
    compute field average. Scatter from standard error on mean, with some weighting.
        
    '''
    
    nfield = recovered_ps_by_field.shape[0]
    nsims = recovered_ps_by_field.shape[1]
    nbins = recovered_ps_by_field.shape[2]
    
    print('nsims = ', nsims, ', nbins = ', nbins)
    obs_ps_by_field = recovered_ps_by_field[:, obs_idx]
    
    ratio_ps_by_field = recovered_ps_by_field / input_ps_by_field
    av_psratio_perfield = np.median(ratio_ps_by_field, axis=1)
    
    field_weights = None
    if compute_field_weights:
        
        all_corr_by_field = np.zeros_like(recovered_ps_by_field)
        print('al_corr by field has shape', all_corr_by_field.shape)
        for i in range(nsims):
            all_corr_by_field[:,i] = recovered_ps_by_field[:, i] / av_psratio_perfield

        field_weights = 1./np.var(all_corr_by_field, axis=1) # inverse variance weights of each corrected field
        
        for n in range(nbins):
            field_weights[:,n] /= np.sum(field_weights[:,n])
        
        if plot:
            plt.figure()
            for k in range(5):
                plt.plot(lb, field_weights[k], color='C'+str(k))
            plt.xscale('log')
            plt.xlabel('$\\ell$', fontsize=16)
            plt.ylabel('Field weights', fontsize=16)
            plt.tick_params(labelsize=14)
            if save_plot:
                plt.savefig('/Users/luminatech/Downloads/ffbias_mocks/041922/field_weights/field_weights_'+str(name)+'.png', bbox_inches='tight')
            plt.show()
            
    
    obs_ps_corr_by_field = obs_ps_by_field / av_psratio_perfield
    
    all_pscorr_fieldstd = np.zeros((nsims, nbins))
    
    if apply_field_weights:
        
        pscorr_fieldstd = np.zeros((nbins))
        pscorr_fieldaverage = np.zeros((nbins))

        
        for n in range(nbins):
            
            field_weights[:,n] /= np.sum(field_weights[:,n])
            
            neff_indiv = compute_neff(field_weights[:,n])
            psav_indivbin = np.average(obs_ps_corr_by_field[:,n], weights = field_weights[:,n])
                    
            psvar_indivbin = np.sum(field_weights[:,n]*(obs_ps_corr_by_field[:,n] - psav_indivbin)**2)*neff_indiv/(neff_indiv-1.)
            pscorr_fieldstd[n] = np.sqrt(psvar_indivbin/neff_indiv)

            pscorr_fieldaverage[n] = psav_indivbin

            for i in range(nsims):
                psav_indivbin = np.average(all_corr_by_field[:,i,n], weights = field_weights[:,n])

                psvar_indivbin = np.sum(field_weights[:,n]*(all_corr_by_field[:,i,n] - psav_indivbin)**2)*neff_indiv/(neff_indiv-1.)
                all_pscorr_fieldstd[i,n] = np.sqrt(psvar_indivbin/neff_indiv)


        simav_pscorr_fieldstd = np.median(all_pscorr_fieldstd, axis=0)

    else:
    
        pscorr_fieldaverage = np.mean(obs_ps_corr_by_field, axis=0)
        pscorr_fieldstd = np.sqrt(np.var(obs_ps_corr_by_field, axis=0)*nfield/(nfield-1))
    
    
    return pscorr_fieldaverage, pscorr_fieldstd, field_weights, obs_ps_corr_by_field, simav_pscorr_fieldstd

    









