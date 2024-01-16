import numpy as np
import astropy.io.fits as fits
import astropy.wcs as wcs
import pandas as pd
import config
import scipy
import scipy.io
from scipy.ndimage import gaussian_filter
import matplotlib
import matplotlib.pyplot as plt
from astropy.table import Table
from reproject import reproject_interp
from astropy.coordinates import SkyCoord
from astropy import units as u
import healpy as hp
from scipy import interpolate
import os
from plotting_fns import *
# from mkk_parallel import *
# from ciber_noise_data_utils import iter_sigma_clip_mask
# from ciber_powerspec_pipeline import *
# from ciber_powerspec_pipeline import CIBER_PS_pipeline

def make_fpaths(fpaths):
    for fpath in fpaths:
        if not os.path.isdir(fpath):
            print('making directory path for ', fpath)
            os.makedirs(fpath)
        else:
            print(fpath, 'already exists')


def compute_sqrt_cl_errors(cl, dcl):
    sqrt_cl_errors = 0.5*(dcl*cl**(-0.5))
    return sqrt_cl_errors


def mock_consistency_chistat(lb, all_mock_recov_ps, mock_all_field_cl_weights, ifield_list = [4, 5, 6, 7, 8], lmax=10000, mode='chi2', all_cov_indiv_full=None, \
                            cov_joint=None):
    ''' 
    mode either 'chi2' or 'chi'
    
    '''
    
    lbmask_chistat = (lb < lmax)*(lb > lb[0])
            
    if all_cov_indiv_full is not None:
        print('all cov indiv shape ', np.array(all_cov_indiv_full).shape)

        all_inv_cov_indiv_lbmask = np.zeros_like(all_cov_indiv_full)
        


    ndof = len(lb[lbmask_chistat])
    
    if cov_joint is not None:
        inv_cov_joint = np.linalg.inv(cov_joint)
        print('condition number of joint cov matrix is ', np.linalg.cond(cov_joint))
    
    all_std_recov_mock = []
    for fieldidx, ifield in enumerate(ifield_list):
        std_recov_mock_ps = np.std(all_mock_recov_ps[:,fieldidx], axis=0)
        all_std_recov_mock.append(std_recov_mock_ps)
        
        if all_cov_indiv_full is not None:
            all_inv_cov_indiv_lbmask[fieldidx] = np.linalg.inv(all_cov_indiv_full[fieldidx])
    
    all_std_recov_mock = np.array(all_std_recov_mock)
    all_chistat_largescale = np.zeros((len(ifield_list), all_mock_recov_ps.shape[0]))
    chistat_perfield = np.zeros_like(all_mock_recov_ps)
    pte_perfield = np.zeros((all_mock_recov_ps.shape[0], all_mock_recov_ps.shape[1]))

    for x in range(all_mock_recov_ps.shape[0]):

        mock_field_average_cl_indiv = np.zeros_like(mock_all_field_cl_weights[0,:])
        for n in range(len(mock_field_average_cl_indiv)):
            mock_field_average_cl_indiv[n] = np.average(all_mock_recov_ps[x,:,n], weights = mock_all_field_cl_weights[:,n])

        if cov_joint is not None:
            resid_joint = []
            for fieldidx, ifield in enumerate(ifield_list):
                resid_joint.extend(all_mock_recov_ps[x,fieldidx,lbmask_chistat]-mock_field_average_cl_indiv[lbmask_chistat])
            resid_joint = np.array(resid_joint)
            
            chistat_joint_mockstd = np.multiply(resid_joint, np.dot(inv_cov_joint, resid_joint.transpose()))      

            

            
        for fieldidx, ifield in enumerate(ifield_list):
            
            resid = all_mock_recov_ps[x,fieldidx,:]-mock_field_average_cl_indiv
            # deviation from field average of individual realization
            
            if mode=='chi2':
                if cov_joint is not None:
                    chistat_mean_cl_mockstd = chistat_joint_mockstd[fieldidx*ndof:(fieldidx+1)*ndof]
                elif all_cov_indiv_full is not None:
                    chistat_mean_cl_mockstd = np.multiply(resid[lbmask_chistat], np.dot(all_inv_cov_indiv_lbmask[fieldidx], resid[lbmask_chistat].transpose()))      
                else:
                    chistat_mean_cl_mockstd = resid**2/(all_std_recov_mock[fieldidx]**2)        
            elif mode=='chi':
                chistat_mean_cl_mockstd = resid/all_std_recov_mock[fieldidx]       

            
            if all_cov_indiv_full is not None or cov_joint is not None:
                chistat_largescale = chistat_mean_cl_mockstd
            else:
                chistat_perfield[x, fieldidx] = chistat_mean_cl_mockstd
                chistat_largescale = chistat_perfield[x, fieldidx, lbmask_chistat]
            
            all_chistat_largescale[fieldidx, x] = np.sum(chistat_largescale)
            pte_indiv = 1. - scipy.stats.chi2.cdf(np.sum(chistat_largescale), ndof)
            pte_perfield[x, fieldidx] = pte_indiv
            
            
    return chistat_perfield, pte_perfield, all_chistat_largescale

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


def get_power_spectrum_2d_epochav(map_a, map_b, pixsize=7., verbose=False):
    dimx, dimy = map_a.shape 
    sterad_per_pix = (pixsize/3600/180*np.pi)**2
    V = dimx * dimy * sterad_per_pix

    ffta = np.fft.fftn(map_a*sterad_per_pix)
    fftb = np.fft.fftn(map_b*sterad_per_pix)

    av_fft = 0.5*(ffta+fftb)

    ps2d_av = np.real(av_fft * np.conj(av_fft)) / V 
    ps2d_av = np.fft.ifftshift(ps2d_av)
    
    l2d = get_l2d(dimx, dimy, pixsize)

    return l2d, ps2d_av

def get_power_spectrum_2d(map_a, map_b=None, pixsize=7., verbose=False):
    '''
    calculate 2d cross power spectrum Cl
    
    Inputs:
    =======
    map_a: 
    map_b:map_b=map_a if no input, i.e. map_a auto
    pixsize:[arcsec]
    
    Outputs:
    ========
    l2d: corresponding ell modes
    ps2d: 2D Cl
    '''
    
    if map_b is None:
        map_b = map_a.copy()
        
    dimx, dimy = map_a.shape
    sterad_per_pix = (pixsize/3600/180*np.pi)**2
    V = dimx * dimy * sterad_per_pix
    
    ffta = np.fft.fftn(map_a*sterad_per_pix)
    fftb = np.fft.fftn(map_b*sterad_per_pix)
    ps2d = np.real(ffta * np.conj(fftb)) / V 
    ps2d = np.fft.ifftshift(ps2d)
    
    l2d = get_l2d(dimx, dimy, pixsize)

    return l2d, ps2d

def get_l2d(dimx, dimy, pixsize):
    lx = np.fft.fftfreq(dimx)*2
    ly = np.fft.fftfreq(dimy)*2
    lx = np.fft.ifftshift(lx)*(180*3600./pixsize)
    ly = np.fft.ifftshift(ly)*(180*3600./pixsize)
    ly, lx = np.meshgrid(ly, lx)
    l2d = np.sqrt(lx**2 + ly**2)
    
    return l2d

def get_power_spec(map_a, map_b=None, mask=None, pixsize=7., 
                   lbinedges=None, lbins=None, nbins=29, 
                   logbin=True, weights=None, return_full=False, return_Dl=False, verbose=False):
    '''
    calculate 1d cross power spectrum Cl
    
    Inputs:
    =======
    map_a: 
    map_b:map_b=map_a if no input, i.e. map_a auto
    mask: common mask for both map
    pixsize:[arcsec]
    lbinedges: predefined lbinedges
    lbins: predefined lbinedges
    nbins: number of ell bins
    logbin: use log or linear ell bin
    weights: Fourier weight
    return_full: return full output or not
    return_Dl: return Dl=Cl*l*(l+1)/2pi or Cl
    
    Outputs:
    ========
    lbins: 1d ell bins
    ps2d: 2D Cl
    Clerr: Cl error, calculate from std(Cl2d(bins))/sqrt(Nmode)
    Nmodes: # of ell modes per ell bin
    lbinedges: 1d ell binedges
    l2d: 2D ell modes
    ps2d: 2D Cl before radial binning
    '''

    if map_b is None:
        map_b = map_a.copy()

    if mask is not None:
        map_a = map_a*mask - np.mean(map_a[mask==1])
        map_b = map_b*mask - np.mean(map_b[mask==1])
    else:
        map_a = map_a - np.mean(map_a)
        map_b = map_b - np.mean(map_b)
        
    l2d, ps2d = get_power_spectrum_2d(map_a, map_b=map_b, pixsize=pixsize)
            
    lbins, Cl, Clerr = azim_average_cl2d(ps2d, l2d, nbins=nbins, lbinedges=lbinedges, lbins=lbins, weights=weights, logbin=logbin, verbose=verbose)
    
    if return_Dl:
        Cl = Cl * lbins * (lbins+1) / 2 / np.pi
        
    if return_full:
        return lbins, Cl, Clerr, Nmodes, lbinedges, l2d, ps2d
    else:
        return lbins, Cl, Clerr


def update_meanvar(count, mean, M2, newValues, plot=False):
    ''' 
    Uses Welfords online algorithm to update ensemble mean and variance. 
    This is written to handle batches of new samples at a time. 
    
    Slightly modified from:
    https://stackoverflow.com/questions/56402955/whats-the-formula-for-welfords-algorithm-for-variance-std-with-batch-updates
    
    Parameters
    ----------
    
    count : 'int'. Running number of samples, which gets increased by size of input batch.
    mean : 'np.array'. Running mean
    M2 : 'np.array'. Running sum of squares of deviations from sample mean.
    newValues : 'np.array'. New data samples.
    plot (optional, default=False) : 'bool'.
    
    Returns
    -------
    
    count, mean, M2. Same definitions as above but updated to include contribution from newValues.
    
    '''
    
    count += len(newValues) # (nsim/nsplit, dimx, dimy)
    delta = np.subtract(newValues, [mean for x in range(len(newValues))])
    mean += np.sum(delta / count, axis=0)
    
    delta2 = np.subtract(newValues, [mean for x in range(len(newValues))])
    M2 += np.sum(delta*delta2, axis=0)
        
    if plot:
        plot_map(M2, title='M2')
        plot_map(delta[0], title='delta')
        plot_map(mean, title='mean')
        
    return count, mean, M2

    
def finalize_meanvar(count, mean, M2):
    ''' 
    Returns final mean, variance, and sample variance. 
    
    Parameters
    ----------
      
    count : 'int'. Running number of samples, which gets increased by size of input batch.
    mean : 'np.array'. Ensemble mean
    M2 : 'np.array'. Running sum of squares of deviations from sample mean.
    
    Returns
    -------
    
    mean : 'np.array'. Final ensemble mean.
    variance : 'np.array'. Estimated variance.
    sampleVariance : 'np.array'. Same as variance but with population correction (count-1).
    
    '''
    mean, variance, sampleVariance = mean, M2/count, M2/(count - 1)
    if count < 2:
        return float('nan')
    else:
        return mean, variance, sampleVariance



# powerspec_utils.py
def return_frac_knox_error(lb, delta_ell, nsidedeg=2):
    frac_knox_errors = np.sqrt(2./((2*lb+1)*delta_ell))
    fsky = float(nsidedeg**2)/float(41253.)    
    frac_knox_errors /= np.sqrt(fsky)
    return frac_knox_errors
    

def azimuthalAverage(image, ell_min=90, center=None, logbins=True, nbins=60, sterad_term=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    ell_min - the minimum multipole used to set range of multipoles
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    logbins - boolean True if log bins else uniform bins
    nbins - number of bins to use

             
    code adapted from https://www.astrobetter.com/blog/2010/03/03/fourier-transforms-of-images-in-python/
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if center is None:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]
    if sterad_term is not None:
        i_sorted /= sterad_term


    ell_max = ell_min*image.shape[0]/np.sqrt(0.5)
    
    if logbins:
        radbins = 10**(np.linspace(np.log10(ell_min), np.log10(ell_max), nbins+1))
    else:
        radbins = np.linspace(ell_min, ell_max, nbins+1)
    
    # convert multipole bins into pixel values
    radbins /= np.min(radbins)
    rbin_idxs = get_bin_idxs(r_sorted, radbins)

    rad_avg = np.zeros(nbins)
    rad_std = np.zeros(nbins)
    
    for i in range(len(rbin_idxs)-1):
        nmodes= len(i_sorted[rbin_idxs[i]:rbin_idxs[i+1]])
        rad_avg[i] = np.mean(i_sorted[rbin_idxs[i]:rbin_idxs[i+1]])
        rad_std[i] = np.std(i_sorted[rbin_idxs[i]:rbin_idxs[i+1]])/np.sqrt(nmodes)
        
    av_rbins = (radbins[:-1]+radbins[1:])/2

    return av_rbins, np.array(rad_avg), np.array(rad_std)

def azim_average_cl2d(ps2d, l2d, nbins=29, lbinedges=None, lbins=None, weights=None, logbin=False, verbose=False, stderr=True):
    
    if lbinedges is None:
        lmin = np.min(l2d[l2d!=0])
        lmax = np.max(l2d[l2d!=0])
        if logbin:
            lbinedges = np.logspace(np.log10(lmin), np.log10(lmax), nbins)
            lbins = np.sqrt(lbinedges[:-1] * lbinedges[1:])
        else:
            lbinedges = np.linspace(lmin, lmax, nbins)
            lbins = (lbinedges[:-1] + lbinedges[1:]) / 2

        lbinedges[-1] = lbinedges[-1]*(1.01)
        
    if weights is None:
        weights = np.ones(ps2d.shape)
        
    Cl = np.zeros(len(lbins))
    Clerr = np.zeros(len(lbins))
    
    Nmodes = np.zeros(len(lbins),dtype=int)
    Neffs = np.zeros(len(lbins))

    for i,(lmin, lmax) in enumerate(zip(lbinedges[:-1], lbinedges[1:])):
        sp = np.where((l2d>=lmin) & (l2d<lmax))
        p = ps2d[sp]
        w = weights[sp]

        Neff = compute_Neff(w)

        Cl[i] = np.sum(p*w) / np.sum(w)

        if verbose:
            print('sum weights:', np.sum(w))
            
        Clerr[i] = np.std(p)
        if stderr:
            Clerr[i] /= np.sqrt(len(p))
        Nmodes[i] = len(p)
        Neffs[i] = Neff

    if verbose:
        print('Nmodes:', Nmodes)
        print('Neffs:', Neffs)
    
    return lbins, Cl, Clerr

def compute_Neff(weights):
    N_eff = np.sum(weights)**2/np.sum(weights**2)

    return N_eff

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

def instantiate_dat_arrays(dimx, dimy, nfields, ntype):
    imarray_shape = (nfields, dimx, dimy)
    imarrays = [np.zeros(imarray_shape) for x in range(ntype)]
    return imarrays

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

def generate_map_meshgrid(ra_cen, dec_cen, nside_deg, dimx, dimy):
    
    ra_range = np.linspace(ra_cen - 0.5*nside_deg, ra_cen + 0.5*nside_deg, dimx)
    dec_range = np.linspace(dec_cen - 0.5*nside_deg, dec_cen + 0.5*nside_deg, dimy)
    map_ra, map_dec = np.meshgrid(ra_range, dec_range)
    
    return map_ra, map_dec

def load_all_ciber_quad_wcs_hdrs(inst, field, hdrdir=None):
    
    if hdrdir is None:
        hdrdir = 'data/astroutputs/inst'+str(inst)+'/'

    xoff = [0,0,512,512]
    yoff = [0,512,0,512]

    wcs_hdrs = []
    for iquad,quad in enumerate(['A','B','C','D']):
        print('quad '+quad)
        hdulist = fits.open(hdrdir + field + '_' + quad + '_astr.fits')
        wcs_hdr=wcs.WCS(hdulist[('primary',1)].header, hdulist)
        wcs_hdrs.append(wcs_hdr)

    return wcs_hdrs

def interpolate_iris_maps_hp(iris_hp, cbps, inst, ncoarse_samp, nside_upsample=None,\
                             ifield_list=[4, 5, 6, 7, 8], use_ciber_wcs=True, nside_deg=4, \
                            plot=True):
    ''' 
    Interpolate IRIS maps from Healpix format to cartesian grid 
    If nside_upsample provided, maps upsampled to desired resolution, 
        otherwise they are kept at resolution of ncoarse_samp.
    '''
    
    all_maps = []
    
    field_center_ras = dict({'elat10':191.5, 'elat30':193.943, 'BootesB':218.109, 'BootesA':219.249, 'SWIRE':241.53})
    field_center_decs = dict({'elat10':8.25, 'elat30':27.998, 'BootesB':33.175, 'BootesA':34.832, 'SWIRE':54.767})

    
    for fieldidx, ifield in enumerate(ifield_list):
        
        iris_coarse_sample = np.zeros((ncoarse_samp, ncoarse_samp))
        
        
        if use_ciber_wcs:
            field = cbps.ciber_field_dict[ifield]
            wcs_hdrs = load_all_ciber_quad_wcs_hdrs(inst, field)
            print('TM'+str(inst), 'ifield '+str(ifield))
            
            for ix in range(ncoarse_samp):
                if ix %10 == 0:
                    print('ix = ', ix)
                for iy in range(ncoarse_samp):

                    x0, x1 = ix*npix_persamp, (ix+1)*npix_persamp - 1
                    y0, y1 = iy*npix_persamp, (iy+1)*npix_persamp - 1


                    xs, ys = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
                    ciber_ra_av_all, ciber_dec_av_all = wcs_hdrs[0].all_pix2world(xs, ys, 0)
                    map_ra_av_new = np.mean(ciber_ra_av_all)
                    map_dec_av_new = np.mean(ciber_dec_av_all)
                    
                    c = SkyCoord(ra=map_ra_av_new*u.degree, dec=map_decs[ix,iy]*u.degree, frame='icrs')
                    ipix = hp.pixelfunc.ang2pix(nside=2048, theta=c.galactic.l.degree, phi=c.galactic.b.degree, lonlat=True)
                    iris_coarse_sample[ix, iy] = iris_hp[ipix]

        else:
            ra_cen = field_center_ras[cbps.ciber_field_dict[ifield]]
            dec_cen = field_center_decs[cbps.ciber_field_dict[ifield]]

            print('ra/dec:', ra_cen, dec_cen)
            map_ras, map_decs = generate_map_meshgrid(ra_cen, dec_cen, nside_deg, dimx=ncoarse_samp, dimy=ncoarse_samp)
            if plot:
                plot_map(map_ras, title='RA')
                plot_map(map_decs, title='DEC')

            for ix in range(ncoarse_samp):
                for iy in range(ncoarse_samp):
                    c = SkyCoord(ra=map_ras[ix,iy]*u.degree, dec=map_decs[ix,iy]*u.degree, frame='icrs')
                    ipix = hp.pixelfunc.ang2pix(nside=2048, theta=c.galactic.l.degree, phi=c.galactic.b.degree, lonlat=True)
                    iris_coarse_sample[ix, iy] = iris_hp[ipix]

        if nside_upsample is not None:
            iris_resize = np.array(Image.fromarray(iris_coarse_sample).resize((nside_upsample, nside_upsample))).transpose()
            
            plot_map(iris_resize)

            all_maps.append(iris_resize)
        else:
            plot_map(np.array(iris_coarse_sample).transpose())
                    
            all_maps.append(np.array(iris_coarse_sample).transpose())
        
    return all_maps
        

def compute_rms_subpatches(image, mask, npix_perside=16, pct=False):
    ''' Compute RMS in small subpatches for comparison with photon noise level expected.'''
    all_std, all_mean = [], []
    dimx, dimy = image.shape[0], image.shape[1]
    
    binned_std = np.zeros_like(image)
    
    ndiv = dimx//npix_perside
    print('ndiv = ', ndiv)
    all_x, all_y = [], []
    
    for nx in range(ndiv-1):
        if nx%50==0:
            print('nx = ', nx)
        for ny in range(ndiv-1):
            
            impatch = image[nx*npix_perside:(nx+1)*npix_perside, ny*npix_perside:(ny+1)*npix_perside]
            maskpatch = mask[nx*npix_perside:(nx+1)*npix_perside, ny*npix_perside:(ny+1)*npix_perside]
            
            if np.sum((maskpatch==1)) > 0.9*npix_perside**2:
                all_mean.append(np.mean(impatch[maskpatch==1]))
                if pct:
                    all_std.append(0.5*(np.nanpercentile(impatch[maskpatch==1], 84)-np.nanpercentile(impatch[maskpatch==1], 16)))
                else:
                    all_std.append(np.std(impatch[maskpatch==1]))
                all_x.append((nx+0.5)*npix_perside)
                all_y.append((ny+0.5)*npix_perside)
                
                binned_std[nx*npix_perside:(nx+1)*npix_perside, ny*npix_perside:(ny+1)*npix_perside] = np.std(impatch[maskpatch==1])
                
    return all_mean, all_std, all_x, all_y, binned_std  



def compute_fieldav_powerspectra(ifield_list, all_signal_ps, all_recov_ps, flatidx=8, apply_field_weights=True):
    # being used as of 12/6/22

    print(all_signal_ps.shape, all_recov_ps.shape)
    
    npsbin = all_signal_ps.shape[2]
    nfield = len(ifield_list)
    nsim = all_signal_ps.shape[0]
    
    all_field_cl_weights = np.zeros((nfield, npsbin))
    all_field_averaged_cls = np.zeros((nsim, npsbin))
    
    for fieldidx, ifield in enumerate(ifield_list):
        # recov_ps_dcl = np.std(all_recov_ps[:,fieldidx,:], axis=0)
        recov_ps_dcl = 0.5*(np.percentile(all_recov_ps[:,fieldidx,:], 84, axis=0)-np.percentile(all_recov_ps[:,fieldidx,:], 16, axis=0))
        all_field_cl_weights[fieldidx] = 1./recov_ps_dcl**2
    
    all_field_cl_weights[:,:flatidx] = 1.0  
    
    if not apply_field_weights:
        all_field_cl_weights = 1.0

    cl_sumweights = np.sum(all_field_cl_weights, axis=0)
    

    for i in range(nsim):
        for n in range(npsbin):
            all_field_averaged_cls[i,n] = np.sum(all_field_cl_weights[:,n]*all_recov_ps[i,:,n])/cl_sumweights[n]

    mean_input_powerspectra = np.mean(all_signal_ps, axis=1)

    return mean_input_powerspectra, all_field_averaged_cls, all_field_cl_weights


def process_mock_powerspectra(cbps, datestr, ifield_list, inst, run_name, nsim, flatidx=8, apply_field_weights=True, sim_test_fpath=None):
    # being used as of 12/6/22
    all_signal_ps = np.zeros((nsim, len(ifield_list), cbps.n_ps_bin))
    all_recov_ps = np.zeros((nsim, len(ifield_list), cbps.n_ps_bin))
    
    if sim_test_fpath is None:
        sim_test_fpath = config.ciber_basepath+'data/input_recovered_ps/'+datestr+'/TM'+str(inst)+'/'
    
    for simidx in np.arange(nsim):
        clpath = np.load(sim_test_fpath+run_name+'/input_recovered_ps_estFF_simidx'+str(simidx)+'.npz')
        signal_ps = clpath['signal_ps']
        lb = clpath['lb']
        recovered_ps = clpath['recovered_ps_est_nofluc']

        all_signal_ps[simidx] = signal_ps
        all_recov_ps[simidx] = recovered_ps
        
    mean_input_powerspectra, \
        all_field_averaged_cls,\
            all_field_cl_weights = compute_fieldav_powerspectra(ifield_list, all_signal_ps, all_recov_ps, flatidx=flatidx, apply_field_weights=apply_field_weights)
    
    return mean_input_powerspectra, all_signal_ps, all_recov_ps, all_field_averaged_cls, all_field_cl_weights, lb

def process_observed_powerspectra(cbps, datestr, ifield_list, inst, observed_run_name, mock_run_name, nsim_mock, flatidx=8, apply_field_weights=True, per_quadrant=False, \
                                    datestr_mock=None):
    
    if datestr_mock is None:
        datestr_mock = datestr

    sim_test_fpath_mock = config.ciber_basepath+'data/input_recovered_ps/'+datestr_mock+'/TM'+str(inst)+'/'
    sim_test_fpath_obs = config.ciber_basepath+'data/input_recovered_ps/'+datestr+'/TM'+str(inst)+'/'

    mock_mean_input_ps, all_mock_signal_ps, all_mock_recov_ps,\
            mock_all_field_averaged_cls, mock_all_field_cl_weights, lb = process_mock_powerspectra(cbps, datestr_mock, ifield_list, \
                                                                                                inst, mock_run_name, nsim_mock, flatidx=flatidx, \
                                                                                                sim_test_fpath=sim_test_fpath_mock)
    cl_sumweights = np.sum(mock_all_field_cl_weights, axis=0)
    
    if per_quadrant:

        obs_clfile = np.load(sim_test_fpath_obs+observed_run_name+'/input_recovered_ps_estFF_simidx0_per_quadrant.npz')
        observed_recov_ps_per_quadant = obs_clfile['recovered_ps_est_per_quadrant']
        observed_recov_dcl_per_quadrant = obs_clfile['recovered_dcl_per_quadrant']
        lb = obs_clfile['lb']



    else:
        obs_clfile = np.load(sim_test_fpath_obs+observed_run_name+'/input_recovered_ps_estFF_simidx0.npz')
        observed_recov_ps = obs_clfile['recovered_ps_est_nofluc']
        observed_recov_dcl_perfield = obs_clfile['recovered_dcl']
        lb = obs_clfile['lb']

    observed_field_average_cl = np.zeros((cbps.n_ps_bin))
    observed_field_average_dcl = np.zeros((cbps.n_ps_bin))
    
    if apply_field_weights:
        for n in range(cbps.n_ps_bin):
            mock_all_field_cl_weights[:,n] /= np.sum(mock_all_field_cl_weights[:,n])
            observed_field_average_cl[n] = np.average(observed_recov_ps[:,n], weights = mock_all_field_cl_weights[:,n])
            neff_indiv = compute_neff(mock_all_field_cl_weights[:,n])
            psvar_indivbin = np.sum(mock_all_field_cl_weights[:,n]*(observed_recov_ps[:,n] - observed_field_average_cl[n])**2)*neff_indiv/(neff_indiv-1.)
            observed_field_average_dcl[n] = np.sqrt(psvar_indivbin/neff_indiv)

    else:
        observed_field_average_cl = np.mean(observed_recov_ps, axis=0)
        observed_field_average_dcl = np.std(observed_recov_ps, axis=0)
        
        
    return lb, observed_recov_ps, observed_recov_dcl_perfield, observed_field_average_cl, observed_field_average_dcl, \
                mock_mean_input_ps, mock_all_field_averaged_cls,\
                    mock_all_field_cl_weights, all_mock_recov_ps, all_mock_signal_ps
    

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

def regrid_iris_by_quadrant(fieldname, inst=1, quad_list=['A', 'B', 'C', 'D'], \
                             xoff=[0,0,512,512], yoff=[0,512,0,512], astr_dir='../data/astroutputs/', \
                             plot=True, dimx=1024, dimy=1024):
    
    ''' 
    Used for regridding maps from IRIS to CIBER. For the CIBER1 imagers the 
    astrometric solution is computed for each quadrant separately, so this function iterates
    through the quadrants when constructing the full regridded images. 
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    
    iris_regrid = np.zeros((dimx, dimy))    
    astr_hdrs = [fits.open(astr_dir+'inst'+str(inst)+'/'+fieldname+'_'+quad+'_astr.fits')[0].header for quad in quad_list]

    # loop over quadrants of first imager
    
    for iquad, quad in enumerate(quad_list):
        
        arrays, footprints = [], []
        
        iris_interp = mosaic(astr_hdrs[iquad])
        
        iris_regrid[yoff[iquad]:yoff[iquad]+512, xoff[iquad]:xoff[iquad]+512] = iris_interp
        if plot:
            plot_map(iris_interp, title='IRIS map interpolated to CIBER, quadrant '+quad)
    
    if plot:
        plot_map(iris_regrid, title='IRIS map interpolated to CIBER')

    return iris_regrid


def load_quad_hdrs(ifield, inst, base_path='/Users/richardfeder/Downloads/ciber_flight/', quad_list=['A', 'B', 'C', 'D'], halves=True):
    
    if halves:
        fpaths_first = [base_path+'/TM'+str(inst)+'/firsthalf/ifield'+str(ifield)+'/ciber_wcs_ifield'+str(ifield)+'_TM'+str(inst)+'_quad'+quad_str+'_firsthalf.fits' for quad_str in quad_list]
        fpaths_second = [base_path+'/TM'+str(inst)+'/secondhalf/ifield'+str(ifield)+'/ciber_wcs_ifield'+str(ifield)+'_TM'+str(inst)+'_quad'+quad_str+'_secondhalf.fits' for quad_str in quad_list]
        
        all_wcs_first = [wcs.WCS(fits.open(fpath_first)[0].header) for fpath_first in fpaths_first]
        all_wcs_second = [wcs.WCS(fits.open(fpath_second)[0].header) for fpath_second in fpaths_second]
        
        return all_wcs_first, all_wcs_second
    else:
        ciber_field_dict = dict({4:'elat10', 5:'elat30',6:'BootesB', 7:'BootesA', 8:'SWIRE', 'train':'UDS'})

        fpaths = [base_path+'/astroutputs/inst'+str(inst)+'/'+ciber_field_dict[ifield]+'_'+quad_str+'_astr.fits' for quad_str in quad_list]
        all_wcs = [wcs.WCS(fits.open(fpath)[0].header) for fpath in fpaths]
        return all_wcs



def regrid_arrays_by_quadrant(map1, ifield, inst0=1, inst1=2, quad_list=['A', 'B', 'C', 'D'], \
                             xoff=[0,0,512,512], yoff=[0,512,0,512], astr_map0_hdrs=None, astr_map1_hdrs=None, indiv_map0_hdr=None, indiv_map1_hdr=None, astr_dir=None, \
                             plot=True, order=0):
    
    ''' 
    Used for regridding maps from one imager to another. For the CIBER1 imagers the 
    astrometric solution is computed for each quadrant separately, so this function iterates
    through the quadrants when constructing the full regridded images. 
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''

    if astr_dir is None:
        astr_dir = '../../ciber/data/'

    ciber_field_dict = dict({4:'elat10', 5:'elat30',6:'BootesB', 7:'BootesA', 8:'SWIRE', 'train':'UDS'})

    map1_regrid, map1_fp_regrid = [np.zeros_like(map1) for x in range(2)]

    fieldname = ciber_field_dict[ifield]
    
    map1_quads = [map1[yoff[iquad]:yoff[iquad]+512, xoff[iquad]:xoff[iquad]+512] for iquad in range(len(quad_list))]
    
    if astr_map0_hdrs is None and indiv_map0_hdr is None:
        print('loading WCS for inst = ', inst0)
        astr_map0_hdrs = load_quad_hdrs(ifield, inst0, base_path=astr_dir, halves=False)
    if astr_map1_hdrs is None and indiv_map1_hdr is None:
        print('loading WCS for inst = ', inst1)
        astr_map1_hdrs = load_quad_hdrs(ifield, inst1, base_path=astr_dir, halves=False)

    # if astr_map0_hdrs is None and indiv_map0_hdr is None:
    #     astr_map0_hdrs = [fits.open(astr_dir+'inst'+str(inst0)+'/'+fieldname+'_'+quad+'_astr.fits')[0].header for quad in quad_list]
    # if astr_map1_hdrs is None and indiv_map1_hdr is None:
    #     astr_map1_hdrs = [fits.open(astr_dir+'inst'+str(inst1)+'/'+fieldname+'_'+quad+'_astr.fits')[0].header for quad in quad_list]
    # loop over quadrants of first imager
    
    for iquad, quad in enumerate(quad_list):
        
        # arrays, footprints = [], []
        run_sum_footprint, sum_array = [np.zeros_like(map1_quads[0]) for x in range(2)]

        # reproject each quadrant of second imager onto first imager
        if indiv_map0_hdr is None:
            for iquad2, quad2 in enumerate(quad_list):
                input_data = (map1_quads[iquad2], astr_map1_hdrs[iquad2])
                array, footprint = reproject_interp(input_data, astr_map0_hdrs[iquad], (512, 512), order=order)

                array[np.isnan(array)] = 0.
                footprint[np.isnan(footprint)] = 0.

                run_sum_footprint += footprint 
                sum_array[run_sum_footprint < 2] += array[run_sum_footprint < 2]
                run_sum_footprint[run_sum_footprint > 1] = 1

                # arrays.append(array)
                # footprints.append(footprint)
       
        # sumarray = np.nansum(arrays, axis=0)
        # sumfootprints = np.nansum(footprints, axis=0)

        if plot:
            plot_map(sum_array, title='sum array')
            plot_map(run_sum_footprint, title='sum footprints')
        
        # print('number of pixels with > 1 footprint', np.sum((sumfootprints==2)))
        # map1_regrid[yoff[iquad]:yoff[iquad]+512, xoff[iquad]:xoff[iquad]+512] = sumarray
        # map1_fp_regrid[yoff[iquad]:yoff[iquad]+512, xoff[iquad]:xoff[iquad]+512] = sumfootprints

        map1_regrid[yoff[iquad]:yoff[iquad]+512, xoff[iquad]:xoff[iquad]+512] = sum_array
        map1_fp_regrid[yoff[iquad]:yoff[iquad]+512, xoff[iquad]:xoff[iquad]+512] = run_sum_footprint

    return map1_regrid, map1_fp_regrid

def regrid_iris_ciber_science_fields(ifield_list=[4,5,6,7,8], inst=1, tail_name=None, plot=False, \
            save_fpath=config.exthdpath+'ciber_fluctuation_data/'):
    
    ciber_field_dict = dict({4:'elat10', 5:'elat30',6:'BootesB', 7:'BootesA', 8:'SWIRE'})

    save_paths = []
    for fieldidx, ifield in enumerate(ifield_list):
        print('Loading ifield', ifield)
        fieldname = ciber_field_dict[ifield]
        print(fieldname)
        irismap = regrid_iris_by_quadrant(fieldname, inst=inst)

        # make fits file saving data
    
        hduim = fits.ImageHDU(irismap, name='TM2_regrid')
        
        hdup = fits.PrimaryHDU()
        hdup.header['ifield'] = ifield
        hdup.header['dat_type'] = 'iris_interp'
        hdul = fits.HDUList([hdup, hduim])
        save_fpath_full = save_fpath+'TM'+str(inst)+'/iris_regrid/iris_regrid_ifield'+str(ifield)+'_TM'+str(inst)
        if tail_name is not None:
            save_fpath_full += '_'+tail_name
        hdul.writeto(save_fpath_full+'.fits', overwrite=True)
        
        save_paths.append(save_fpath_full+'.fits')
        
    return save_paths


def regrid_tm2_to_tm1_science_fields(ifield_list=[4,5,6,7,8], inst0=1, inst1=2, \
                                    inst0_maps=None, inst1_maps=None, flight_dat_base_path=config.exthdpath+'noise_model_validation_data/', \
                                    save_fpath=config.exthdpath+'/ciber_fluctuation_data/', \
                                    tail_name=None, plot=False, cal_facs=None, astr_dir=None):
    
    ciber_field_dict = dict({4:'elat10', 5:'elat30',6:'BootesB', 7:'BootesA', 8:'SWIRE'})

    save_paths, regrid_ims = [], []
    for fieldidx, ifield in enumerate(ifield_list):
        print('Loading ifield', ifield)
        fieldname = ciber_field_dict[ifield]
        print(fieldname)
        
        if inst0_maps is not None and inst1_maps is not None:
            print('taking directly from inst0_maps and inst1_maps..')
            tm1 = inst0_maps[fieldidx]
            tm2 = inst1_maps[fieldidx]
        else:
            tm1 = fits.open(flight_dat_base_path+'TM1/validationHalfExp/field'+str(ifield)+'/flightMap.FITS')[0].data
            tm2 = fits.open(flight_dat_base_path+'TM2/validationHalfExp/field'+str(ifield)+'/flightMap.FITS')[0].data       


        tm2_regrid, tm2_fp_regrid = regrid_arrays_by_quadrant(tm2, ifield, inst0=inst0, inst1=inst1, \
                                                             plot=plot, astr_dir=astr_dir)

        if cal_facs is not None:
            plot_map(cal_facs[1]*tm2, title='TM2 data')
            plot_map(cal_facs[1]*tm1, title='TM1 data')
            plot_map(cal_facs[2]*tm2_regrid, title='TM2 regrid')
            plot_map(cal_facs[1]*(tm1+tm2_regrid), title='TM1+TM2 data')
        regrid_ims.append(tm2_regrid)
    
        # make fits file saving data
    
        hduim = fits.ImageHDU(tm2_regrid, name='TM2_regrid')
        hdufp = fits.ImageHDU(tm2_fp_regrid, name='TM2_footprint')
        
        hdup = fits.PrimaryHDU()
        hdup.header['ifield'] = ifield
        hdup.header['dat_type'] = 'observed'
        hdul = fits.HDUList([hdup, hduim, hdufp])
        save_fpath_full = save_fpath+'TM'+str(inst1)+'/ciber_regrid/flightMap_ifield'+str(ifield)+'_TM'+str(inst1)+'_regrid_to_TM'+str(inst0)
        if tail_name is not None:
            save_fpath_full += '_'+tail_name
        hdul.writeto(save_fpath_full+'.fits', overwrite=True)
        
        save_paths.append(save_fpath_full+'.fits')
        
    return regrid_ims, save_paths

def ciber_x_ciber_regrid_all_masks(cbps, inst, cross_inst, cross_union_mask_tail, mask_tail, mask_tail_cross, ifield_list=[4,5,6,7,8], astr_base_path='data/', \
                                  base_fluc_path = 'data/fluctuation_data/', save=True, cross_mask=None, cross_inst_only=False, plot=False, plot_quad=False):
    
    ''' Function to regrid masks from one imager to another and return/save union of masks. 12/20/22'''
    mask_base_path = base_fluc_path+'TM'+str(inst)+'/masks/'
    mask_base_path_cross_inst = base_fluc_path+'TM'+str(cross_inst)+'/masks/'
    mask_base_path_cross_union = base_fluc_path+'TM'+str(inst)+'_TM'+str(cross_inst)+'_cross/masks/'

    masks, masks_cross, masks_cross_regrid, masks_union = [np.zeros((len(ifield_list), cbps.dimx, cbps.dimy)) for x in range(4)]
    union_savefpaths = []

    for fieldidx, ifield in enumerate(ifield_list):
        print('ifield ', ifield)
        
        astr_map0_hdrs = load_quad_hdrs(ifield, inst, base_path=astr_base_path, halves=False)
        astr_map1_hdrs = load_quad_hdrs(ifield, cross_inst, base_path=astr_base_path, halves=False)
        make_fpaths([mask_base_path_cross_union+cross_union_mask_tail])
        
        masks[fieldidx] = fits.open(mask_base_path+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits')['joint_mask_'+str(ifield)].data

        if cross_mask is None:
            if cross_inst_only:
                print('Only using instrument mask of cross CIBER map..')
                inst_mask_fpath = config.exthdpath+'/ciber_fluctuation_data/TM'+str(cross_inst)+'/masks/maskInst_102422/field'+str(ifield)+'_TM'+str(cross_inst)+'_maskInst_102422.fits'
                masks_cross[fieldidx] = cbps.load_mask(ifield, inst, mask_fpath=inst_mask_fpath, instkey='maskinst', inplace=False)
            else:
                print('Using full cross map mask..')
                masks_cross[fieldidx] = fits.open(mask_base_path_cross_inst+mask_tail_cross+'/joint_mask_ifield'+str(ifield)+'_inst'+str(cross_inst)+'_observed_'+mask_tail_cross+'.fits')['joint_mask_'+str(ifield)].data

            mask_cross_regrid, mask_cross_fp_regrid = regrid_arrays_by_quadrant(masks_cross[fieldidx], ifield, inst0=inst, inst1=cross_inst, \
                                                     astr_map0_hdrs=astr_map0_hdrs, astr_map1_hdrs=astr_map1_hdrs, plot=plot_quad)

            masks_cross_regrid[fieldidx] = mask_cross_regrid
        masks_union[fieldidx] = masks[fieldidx]*mask_cross_regrid

        if plot:
            plot_map(masks[fieldidx], title='mask fieldidx = '+str(fieldidx))
            plot_map(masks_cross[fieldidx], title='mask fieldidx = '+str(fieldidx))

        masks_union[fieldidx][masks_union[fieldidx] > 1] = 1.0
        
        print('Mask fraction for mask TM'+str(inst)+' is ', np.sum(masks[fieldidx])/float(1024**2))
        print('Mask fraction for mask TM'+str(cross_inst)+' is ', np.sum(masks_cross[fieldidx])/float(1024**2))
        print('Mask fraction for union mask is ', np.sum(masks_union[fieldidx])/float(1024**2))
        if plot:
            plot_map(masks_union[fieldidx], title='mask x mask cross')

        if save:
            hdul = write_mask_file(masks_union[fieldidx], ifield, inst, dat_type='cross_observed', cross_inst=cross_inst)
            union_savefpath = mask_base_path_cross_union+cross_union_mask_tail+'/joint_mask_ifield'+str(ifield)+'_TM'+str(inst)+'_TM'+str(cross_inst)+'_observed_'+cross_union_mask_tail+'.fits'
            print('Saving cross mask to ', union_savefpath)

            hdul.writeto(union_savefpath, overwrite=True)
            union_savefpaths.append(union_savefpath)
    if save:
        return masks, masks_cross, masks_cross_regrid, masks_union, union_savefpaths
    
    return masks, masks_cross, masks_cross_regrid, masks_union


def write_Mkk_fits(Mkk, inv_Mkk, ifield, inst, cross_inst=None, sim_idx=None, generate_starmask=True, generate_galmask=True, \
                  use_inst_mask=True, dat_type=None, mag_lim_AB=None):
    hduim = fits.ImageHDU(inv_Mkk, name='inv_Mkk_'+str(ifield))        
    hdum = fits.ImageHDU(Mkk, name='Mkk_'+str(ifield))

    hdup = fits.PrimaryHDU()
    hdup.header['ifield'] = ifield
    hdup.header['inst'] = inst

    if cross_inst is not None:
        hdup.header['cross_inst'] = cross_inst
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


def write_regrid_proc_file(masked_proc, ifield, inst, regrid_to_inst, mask_tail=None,\
                           dat_type='observed', mag_lim=None, mag_lim_cross=None, obs_level=None):
    hdum = fits.ImageHDU(masked_proc, name='proc_regrid_'+str(ifield))
    hdup = fits.PrimaryHDU()
    hdup.header['ifield'] = ifield
    hdup.header['inst'] = inst
    hdup.header['regrid_to_inst'] = regrid_to_inst
    
    if mask_tail is not None:
        hdup.header['mask_tail'] = mask_tail
    if dat_type is not None:
        hdup.header['dat_type'] = dat_type
    if mag_lim is not None:
        hdup.header['mag_lim'] = mag_lim
    if mag_lim_cross is not None:
        hdup.header['mag_lim_cross'] = mag_lim_cross
    if obs_level is not None:
        hdup.header['obs_level'] = obs_level

    hdul = fits.HDUList([hdup, hdum])
    
    return hdul


def write_mask_file(mask, ifield, inst, cross_inst=None, sim_idx=None, generate_galmask=None, generate_starmask=None, use_inst_mask=None, \
                   dat_type=None, mag_lim_AB=None, with_ff_mask=None, name=None, a1=None, b1=None, c1=None, dm=None, alpha_m=None, beta_m=None):

    if name is None:
        name = 'joint_mask_'+str(ifield)
    hdum = fits.ImageHDU(mask, name=name)
    hdup = fits.PrimaryHDU()
    hdup.header['ifield'] = ifield
    hdup.header['inst'] = inst

    if cross_inst is not None:
        hdup.header['cross_inst'] = cross_inst
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

    if a1 is not None:
        hdup.header['a1'] = a1
    if b1 is not None:
        hdup.header['b1'] = b1
    if c1 is not None:
        hdup.header['c1'] = c1
    if dm is not None:
        hdup.header['dm'] = dm

    if alpha_m is not None:
        hdup.header['alpha_m'] = alpha_m
    if beta_m is not None:
        hdup.header['beta_m'] = beta_m
        
    hdul = fits.HDUList([hdup, hdum])
    
    return hdul

def save_resid_cl_file(cl_table, names, mode='isl', return_hdul=False, save=True, cl_save_fpath=None, **kwargs):
    tab = Table(cl_table, names=tuple(names))
    hdu_cl = fits.BinTableHDU(tab, name='cls_'+mode)
    hdr = fits.Header()

    for key, value in kwargs.items():
        hdr[key] = value
    prim = fits.PrimaryHDU(header=hdr)
    hdul = fits.HDUList([prim, hdu_cl])
    if save:
        if cl_save_fpath is None:
            print("No cl_save_fpath provided..")
        else:
            hdul.writeto(cl_save_fpath, overwrite=True)

    if return_hdul:
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


def compute_weighted_cl(all_cl, all_clerr):
    
    all_cl = np.array(all_cl)
    all_clerr = np.array(all_clerr)
    
    variance = all_clerr**2
    
    weights = 1./variance
    
    cl_sumweights = np.sum(weights, axis=0)
    
    weighted_variance = 1./cl_sumweights
    
    field_averaged_std = np.sqrt(weighted_variance)
    
    field_averaged_cl = np.nansum(weights*all_cl, axis=0)/cl_sumweights
    
    return field_averaged_cl, field_averaged_std


def load_weighted_cl_file_cross(cl_fpath, mode='observed'):

    clfile = np.load(cl_fpath)
    
    if mode=='observed':
        observed_recov_ps = clfile['observed_recov_ps']
        observed_recov_dcl_perfield = clfile['observed_recov_dcl_perfield']
        observed_field_average_cl = clfile['observed_field_average_cl']
        observed_field_average_dcl = clfile['observed_field_average_dcl']
        lb = clfile['lb']
    
        return lb, observed_recov_ps, observed_recov_dcl_perfield, observed_field_average_cl, observed_field_average_dcl, None
    
    elif mode=='mock':
        mock_mean_input_ps = clfile['mock_mean_input_ps']
        mock_all_field_averaged_cls = clfile['mock_all_field_averaged_cls']
        mock_all_field_cl_weights = clfile['mock_all_field_cl_weights']
        all_mock_recov_ps = clfile['all_mock_recov_ps']
        all_mock_signal_ps = clfile['all_mock_signal_ps']
        lb = clfile['lb']
    
        return lb, mock_mean_input_ps, mock_all_field_averaged_cls, mock_all_field_cl_weights, all_mock_recov_ps, all_mock_signal_ps
    

def compute_sim_corrected_fieldaverage(recovered_ps_by_field, input_ps_by_field, lb, obs_idx=0, compute_field_weights=True, \
                                      apply_field_weights=True, name=None, plot=True, save_plot=False):
    
    ''' 
    Compute estimated pipeline bias from mocks, or provide, and use one realization
    to represent "observed" data. Correct observed data by mean FF bias of each field and then
    compute field average. Scatter from standard error on mean, with some weighting.
        
    '''
    
    nfield = recovered_ps_by_field.shape[0]
    nsims = recovered_ps_by_field.shape[1]
    nbins = recovered_ps_by_field.shape[2]
    
    print('nsims = ', nsims, ', nbins = ', nbins)
    print('obs idx is ', obs_idx)
    obs_ps_by_field = recovered_ps_by_field[:, obs_idx]
    
    ratio_ps_by_field = recovered_ps_by_field / input_ps_by_field
    av_psratio_perfield = np.median(ratio_ps_by_field, axis=1)
    
    field_weights = None
    fig = None
    if compute_field_weights:
        
        all_corr_by_field = np.zeros_like(recovered_ps_by_field)
        print('al_corr by field has shape', all_corr_by_field.shape)
        for i in range(nsims):
            all_corr_by_field[:,i] = recovered_ps_by_field[:, i] / av_psratio_perfield

        field_weights = 1./np.var(all_corr_by_field, axis=1) # inverse variance weights of each corrected field
        
        for n in range(nbins):
            field_weights[:,n] /= np.sum(field_weights[:,n])
        
        if plot:
            fig = plt.figure()
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
    
    
    return pscorr_fieldaverage, pscorr_fieldstd, field_weights, obs_ps_corr_by_field, simav_pscorr_fieldstd, fig
    

class powerspec():

    def __init__(self, n_ps_bin=25, dimx=1024, \
                dimy=1024):

        self.Mkk_obj = Mkk_bare(dimx=dimx, dimy=dimy, ell_min=180.*(1024./dimx), nbins=n_ps_bin)


    def load_in_powerspec(self, powerspec_fpath, mode=None, inplace=False):

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









