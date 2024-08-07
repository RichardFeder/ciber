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
    


def gather_fiducial_auto_ps_results_new(inst, nsim_mock=None, observed_run_name=None, mock_run_name=None,\
                                    mag_lim=None, ifield_list=[4, 5, 6, 7, 8], flatidx=0, \
                                   datestr_obs='111323', datestr_mock='112022', lmax_cov=10000, \
                                   save_cov=True, startidx=1, compute_cov=False, cbps=None):
    
    if cbps is None:
        cbps = CIBER_PS_pipeline()
        
    bandstr_dict = dict({1:'J', 2:'H'})
    maglim_default = dict({1:17.5, 2:17.0})
    
    bandstr = bandstr_dict[inst]
    
    if mag_lim is None:
        mag_lim = maglim_default[inst]
        
    if observed_run_name is None:
        observed_run_name = 'observed_'+bandstr+'lt'+str(mag_lim)+'_012524_ukdebias'
        
    if mock_run_name is None:
        if mag_lim <= 15:
            mock_mode = 'maskmkk'
        else:
            mock_mode = 'mkkffest'
        mock_run_name = 'mock_'+str(bandstr)+'lt'+str(mag_lim)+'_121823_'+mock_mode+'_perquad'
#         mock_run_name += '_ellm2clus'

    
    lb, observed_recov_ps, observed_recov_dcl_perfield, observed_field_average_cl, observed_field_average_dcl,\
        mock_mean_input_ps, mock_all_field_averaged_cls, mock_all_field_cl_weights, \
            all_mock_recov_ps, all_mock_signal_ps = process_observed_powerspectra(cbps, datestr_obs, ifield_list, inst, \
                                                                                    observed_run_name, mock_run_name, nsim_mock, \
                                                                                 flatidx=flatidx, apply_field_weights=True, \
                                                                                 datestr_mock=datestr_mock) 
    
    
    obs_dict = dict({'lb':lb, 'observed_run_name':observed_run_name, 'mag_lim':mag_lim, 'observed_recov_ps':observed_recov_ps, 'observed_recov_dcl_perfield':observed_recov_dcl_perfield, \
                    'observed_field_average_cl':observed_field_average_cl, 'observed_field_average_dcl':observed_field_average_dcl})
    
    mock_dict = dict({'mock_run_name':mock_run_name, 'mock_mean_input_ps':mock_mean_input_ps, 'mock_all_field_averaged_cls':mock_all_field_averaged_cls, \
                     'mock_all_field_cl_weights':mock_all_field_cl_weights, 'all_mock_recov_ps':all_mock_recov_ps, 'all_mock_signal_ps':all_mock_signal_ps})
    
    cov_dict = None
    cl_fpath = None
    if compute_cov:
        lb_mask, all_cov_indiv_full,\
            all_resid_data_matrices,\
                resid_joint_data_matrix = compute_mock_covariance_matrix(lb, inst, all_mock_recov_ps, mock_all_field_averaged_cls, \
                                                                        lmax=lmax_cov, save=save_cov, mock_run_name=mock_run_name, plot=False, startidx=startidx)

        cov_dict = dict({'lb_mask':lb_mask, 'all_cov_indiv_full':all_cov_indiv_full, 'all_resid_data_matrices':all_resid_data_matrices, \
                        'resid_joint_data_matrix':resid_joint_data_matrix})

        if save_cov:
            cl_fpath = save_weighted_mock_cl_file(lb, inst, mock_run_name, mock_mean_input_ps, mock_all_field_averaged_cls, mock_all_field_cl_weights, all_mock_recov_ps, \
                            all_mock_signal_ps)
        
    
    return obs_dict, mock_dict, cov_dict, cl_fpath

def calc_binned_ps_vs_mag(mag_lims, ifield_list, obs_fieldav_cl, obs_fieldav_dcl, pf, observed_recov_dcl_perfield=None,\
                          power_maglim_isl_igl=None, nbp=12):

    nmag = len(mag_lims)
    binned_obs_fieldav, binned_obs_fieldav_dcl, igl_isl_vs_maglim = [np.zeros((nmag, nbp)) for x in range(3)]
    if observed_recov_dcl_perfield is not None:
        binned_obs_perfield, binned_obs_perfield_dcl = [np.zeros((nmag, len(ifield_list), nbp)) for x in range(2)]

    prefac_binned = np.zeros((nbp))
    
    for magidx, maglim in enumerate(mag_lims):
        for idx in range(nbp):

            if idx==0:
                prefac_binned[idx] = pf[2*idx+1]
                binned_obs_fieldav[magidx,idx] = obs_fieldav_cl[2*idx+1]
                binned_obs_fieldav_dcl[magidx,idx] = obs_fieldav_dcl[2*idx+1]

                if observed_recov_dcl_perfield is not None:
                    for fieldidx, ifield in enumerate(ifield_list):
                        binned_obs_perfield[magidx,fieldidx,idx] = observed_recov_ps[fieldidx,2*idx+1]
                        binned_obs_perfield_dcl[magidx,fieldidx,idx] = observed_recov_dcl_perfield[fieldidx,2*idx+1]


                if power_maglim_isl_igl is not None:
                    igl_isl_vs_maglim[magidx,idx] = power_maglim_isl_igl[2*idx+1]

            else:

                prefac_binned[idx] = np.sqrt(pf[2*idx]*pf[2*idx+1])

                binned_obs_fieldav[magidx,idx] = 0.5*(obs_fieldav_cl[2*idx]+obs_fieldav_cl[2*idx+1])
                binned_obs_fieldav_dcl[magidx,idx] = np.sqrt(obs_fieldav_dcl[2*idx]**2 + obs_fieldav_dcl[2*idx+1]**2)/np.sqrt(2)

                if observed_recov_dcl_perfield is not None:
                    for fieldidx, ifield in enumerate(ifield_list):
                        binned_obs_perfield[magidx,fieldidx,idx] = 0.5*(observed_recov_ps[fieldidx,2*idx]+observed_recov_ps[fieldidx,2*idx+1])
                        binned_obs_perfield_dcl[magidx,fieldidx,idx] = np.sqrt(observed_recov_dcl_perfield[fieldidx,2*idx]**2 + observed_recov_dcl_perfield[fieldidx,2*idx+1]**2)/np.sqrt(2)

                if power_maglim_isl_igl is not None:
                    igl_isl_vs_maglim[magidx,idx] = 0.5*(power_maglim_isl_igl[2*idx] + power_maglim_isl_igl[2*idx+1])



    return prefac_binned, binned_obs_fieldav, binned_obs_fieldav_dcl,\
                binned_obs_perfield, binned_obs_perfield_dcl, igl_isl_vs_maglim 


def compute_rlx_unc_comps(cl_auto_a, cl_auto_b, cl_cross_ab, dcl_auto_a, dcl_auto_b, dcl_cross_ab):

    term1 = (1./(cl_auto_a*cl_auto_b))
    term2 = (dcl_cross_ab**2 + (cl_cross_ab*dcl_auto_a/(2*cl_auto_a))**2 + (cl_cross_ab*dcl_auto_b/(2*cl_auto_b))**2)

    rlx_unc = np.sqrt(term1*term2)
    return rlx_unc

def compute_snr(average, errors):
    ''' Given some array of values and errors on those values, this function computes the signal to noise ratio (SNR) in each bin as well as
        the summed SNR over elements''' 

    snrs_per_bin = average / errors
    total_sq = np.sum(snrs_per_bin**2)
    return np.sqrt(total_sq), snrs_per_bin

def compute_cl_snr(lb, cl, clerr, lb_min=None, lb_max=None):

    snr_vec = cl/clerr
    snr_mask = (cl > 0)
    if lb_min is not None:
        snr_mask *= (lb >= lb_min)
    if lb_max is not None:
        snr_mask *= (lb <= lb_max)

    snr_sel = snr_vec[snr_mask]
    tot_snr = np.sqrt(np.sum(snr_sel**2))

    return snr_sel, tot_snr

def compute_sqrt_cl_errors(cl, dcl):
    sqrt_cl_errors = 0.5*(dcl*cl**(-0.5))
    return sqrt_cl_errors

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

def compute_mock_covariance_matrix(lb, inst, all_mock_recov_ps, mock_all_field_averaged_cls, lmax=None, ifield_list = [4, 5, 6, 7, 8], save=False, \
                                  mock_run_name=None, datestr='111323', plot=False, per_field=True, startidx=None):
    
    
    prefac = lb*(lb+1)/(2*np.pi)

    if lmax is not None:
        lb_mask = (lb < lmax)
    else:
        lb_mask = np.ones_like(lb)
        
    if startidx is not None:
        lb_mask *= (lb >= lb[startidx])
        
    ndof = np.sum(lb_mask).astype(int)

    print('ndof:', ndof)
    nsim = all_mock_recov_ps.shape[0]
    
    print('nsim = ', nsim)
    
    resid_data_matrix = np.zeros_like(all_mock_recov_ps)

    print('resid_data_matrix has shape ', resid_data_matrix.shape)
    print('resid_joint_data_matrix has shape ', resid_joint_data_matrix.shape)


    resid_joint_data_matrix = np.zeros((nsim, ndof*resid_data_matrix.shape[1]))
    all_cov_indiv_full, all_resid_data_matrices, all_corr_indiv_full = [[] for x in range(3)]
    


    
    if per_field:
    
        for fieldidx, ifield in enumerate(ifield_list):    
            for simidx in range(nsim):
                resid_data_matrix[simidx, fieldidx] = all_mock_recov_ps[simidx, fieldidx] - mock_all_field_averaged_cls[simidx]

                resid_joint_data_matrix[simidx, fieldidx*ndof:(fieldidx+1)*ndof] = resid_data_matrix[simidx, fieldidx, lb_mask]

            resid_data_matrix_indiv = resid_data_matrix[:,fieldidx,:].copy()
            resid_data_matrix_indiv -= np.mean(resid_data_matrix_indiv)
            all_resid_data_matrices.append(resid_data_matrix_indiv)

            cov_indiv_full = np.cov(resid_data_matrix_indiv.transpose())
            all_cov_indiv_full.append(cov_indiv_full)

            corr_indiv_full = np.corrcoef(resid_data_matrix_indiv.transpose())
            all_corr_indiv_full.append(cov_indiv_full)

            if plot:
                plot_map(np.log10(cov_indiv_full), title='covariance', figsize=(4,4))
                plot_map(corr_indiv_full, title='correlation', figsize=(4,4))

                
    else:
        all_mock_recov_ps_reshape = (prefac*all_mock_recov_ps).reshape((1000, len(ifield_list)*all_mock_recov_ps.shape[2]))
        print('all_mock_recov_ps_reshape has shape ', all_mock_recov_ps_reshape.shape)
        cov_allfields = np.cov(all_mock_recov_ps_reshape.transpose())
        corr_allfields = np.corrcoef(all_mock_recov_ps_reshape.transpose())
        inv_cov_allfields = np.linalg.inv(cov_allfields)
        
        if plot:
            plot_map(cov_allfields, title='cov allfields')
            plot_map(corr_allfields, title='corr allfields')

    if save and mock_run_name is not None:
        save_dir = config.ciber_basepath+'data/ciber_mocks/'+datestr+'/TM'+str(inst)+'/covariance/'+mock_run_name+'/'
        make_fpaths([save_dir])
        
        if per_field:
        
            save_fpath = save_dir+'/ciber_field_consistency_cov_matrices_TM'+str(inst)+'_'+mock_run_name+'.npz'

            np.savez(save_fpath, lb=lb, all_cov_indiv_full=all_cov_indiv_full,\
                     all_resid_data_matrices=all_resid_data_matrices,\
                     resid_joint_data_matrix=resid_joint_data_matrix)
        else:
            save_fpath = save_dir+'/ciber_fullvec_cov_matrices_TM'+str(inst)+'_'+mock_run_name+'.npz'
            
            np.savez(save_fpath, lb=lb, cov_allfields=cov_allfields, corr_allfields=corr_allfields)

    return lb_mask, all_cov_indiv_full, all_resid_data_matrices, resid_joint_data_matrix

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


def get_power_spec(map_a, map_b=None, mask=None, pixsize=7., 
                   lbinedges=None, lbins=None, nbins=29, 
                   logbin=True, weights=None, return_full=False, return_Dl=False, verbose=False, \
                   remove_outlier_fac=None):
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
            
    lbins, Cl, Clerr = azim_average_cl2d(ps2d, l2d, nbins=nbins, lbinedges=lbinedges, lbins=lbins, weights=weights, logbin=logbin, verbose=verbose, \
                                        remove_outlier_fac=remove_outlier_fac)
        
    if return_Dl:
        Cl = Cl * lbins * (lbins+1) / 2 / np.pi
        
    if return_full:
        return lbins, Cl, Clerr, lbinedges, l2d, ps2d
    else:
        return lbins, Cl, Clerr


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

def azim_average_cl2d(ps2d, l2d, nbins=29, lbinedges=None, lbins=None, weights=None, logbin=False, verbose=False, stderr=True, \
                    remove_outlier_fac=None):
    
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
        lbmask = (l2d>=lmin)*(l2d<lmax)
        sp = np.where(lbmask)
        
        # sp = np.where((l2d>=lmin) & (l2d<lmax))
        p = ps2d[sp]
        w = weights[sp]

        Neff = compute_Neff(w)


        if remove_outlier_fac is not None and i > 5:

            prav = p.ravel()
            wrav = w.ravel()
            mean_prav = np.mean(prav)
            sp_remove_outlier = (prav < mean_prav*remove_outlier_fac)
            Cl[i] = np.sum(prav[sp_remove_outlier]*wrav[sp_remove_outlier])/np.sum(wrav[sp_remove_outlier])
            print('for bin ', i, 'sum weights with/without outlier removal: ', np.sum(wrav[sp_remove_outlier]), np.sum(w))
        else:

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
    # used for ensemble of mock power spectra

    print(all_signal_ps.shape, all_recov_ps.shape)
    
    npsbin = all_signal_ps.shape[2]
    nfield = len(ifield_list)
    nsim = all_signal_ps.shape[0]
    
    all_field_cl_weights = np.zeros((nfield, npsbin))
    all_field_averaged_cls = np.zeros((nsim, npsbin))
    
    for fieldidx, ifield in enumerate(ifield_list):
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


def process_mock_powerspectra(cbps, datestr, ifield_list, inst, run_name, nsim, flatidx=8, apply_field_weights=True, sim_test_fpath=None, pct68=True):
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
                                    datestr_mock=None, pct68=True):
    
    if datestr_mock is None:
        datestr_mock = datestr

    sim_test_fpath_mock = config.ciber_basepath+'data/input_recovered_ps/'+datestr_mock+'/TM'+str(inst)+'/'
    sim_test_fpath_obs = config.ciber_basepath+'data/input_recovered_ps/'+datestr+'/TM'+str(inst)+'/'

    mock_mean_input_ps, all_mock_signal_ps, all_mock_recov_ps,\
            mock_all_field_averaged_cls, mock_all_field_cl_weights, lb = process_mock_powerspectra(cbps, datestr_mock, ifield_list, \
                                                                                                inst, mock_run_name, nsim_mock, flatidx=flatidx, \
                                                                                                sim_test_fpath=sim_test_fpath_mock, pct68=pct68)
    cl_sumweights = np.sum(mock_all_field_cl_weights, axis=0)
    
    if per_quadrant:

        obs_clfile = np.load(sim_test_fpath_obs+observed_run_name+'/input_recovered_ps_estFF_simidx0_per_quadrant.npz')
        observed_recov_ps_per_quadant = obs_clfile['recovered_ps_est_per_quadrant']
        observed_recov_dcl_per_quadrant = obs_clfile['recovered_dcl_per_quadrant']
    else:
        obs_clfile = np.load(sim_test_fpath_obs+observed_run_name+'/input_recovered_ps_estFF_simidx0.npz')
        observed_recov_ps = obs_clfile['recovered_ps_est_nofluc']
        observed_recov_dcl_perfield = obs_clfile['recovered_dcl']
    
    lb = obs_clfile['lb']

    observed_field_average_cl = np.zeros((cbps.n_ps_bin))
    observed_field_average_dcl = np.zeros((cbps.n_ps_bin))
    
    if apply_field_weights:

        observed_field_average_cl, observed_field_average_dcl = compute_weighted_cl(observed_recov_ps, mock_all_field_cl_weights)
        # for n in range(cbps.n_ps_bin):
        #     mock_all_field_cl_weights[:,n] /= np.sum(mock_all_field_cl_weights[:,n])
        #     observed_field_average_cl[n] = np.average(observed_recov_ps[:,n], weights = mock_all_field_cl_weights[:,n])
        #     neff_indiv = compute_Neff(mock_all_field_cl_weights[:,n])
        #     psvar_indivbin = np.sum(mock_all_field_cl_weights[:,n]*(observed_recov_ps[:,n] - observed_field_average_cl[n])**2)*neff_indiv/(neff_indiv-1.)
        #     observed_field_average_dcl[n] = np.sqrt(psvar_indivbin/neff_indiv)

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


    
def compute_weighted_cl(indiv_cl, field_weights):
    
    n_ps_bin = len(indiv_cl[0])
    field_average_cl, field_average_dcl = [np.zeros_like(indiv_cl[0]) for x in range(2)]
    
    for n in range(n_ps_bin):
        field_weights[:,n] /= np.sum(field_weights[:,n])
        field_average_cl[n] = np.average(indiv_cl[:,n], weights=field_weights[:,n])
        neff_indiv = compute_Neff(field_weights[:,n])

        
        psvar_indivbin = np.sum(field_weights[:,n]*(indiv_cl[:,n] - field_average_cl[n])**2)*neff_indiv/(neff_indiv-1.)
        field_average_dcl[n] = np.sqrt(psvar_indivbin/neff_indiv)
        
    return field_average_cl, field_average_dcl

def ciber_ciber_rl_coefficient(obs_name_A, obs_name_B, obs_name_AB, startidx=1, endidx=-1):

    
    obs_fieldav_cls, obs_fieldav_dcls = [], []
    inst_list = [1, 2]
    for clidx, obs_name in enumerate([obs_name_A, obs_name_B, obs_name_AB]):
        if clidx<2:
            
            cl_fpath_obs = config.ciber_basepath+'data/input_recovered_ps/cl_files/TM'+str(inst_list[clidx])+'/cl_'+obs_name+'.npz'
            lb, observed_recov_ps, observed_recov_dcl_perfield,\
                observed_field_average_cl, observed_field_average_dcl,\
                    mock_all_field_cl_weights = load_weighted_cl_file(cl_fpath_obs)     
        else:
            cl_fpath_obs = config.ciber_basepath+'data/input_recovered_ps/cl_files/TM1_TM2_cross/cl_'+obs_name+'.npz'
            lb, observed_recov_ps, observed_recov_dcl_perfield,\
                observed_field_average_cl, observed_field_average_dcl,\
                    mock_all_field_cl_weights = load_weighted_cl_file(cl_fpath_obs)

        obs_fieldav_cls.append(observed_field_average_cl)
        obs_fieldav_dcls.append(observed_field_average_dcl)

    r_TM = obs_fieldav_cls[2]/np.sqrt(obs_fieldav_cls[0]*obs_fieldav_cls[1])

    sigma_r_TM = np.sqrt((1./(obs_fieldav_cls[0]*obs_fieldav_cls[1]))*(obs_fieldav_dcls[2]**2 + (obs_fieldav_cls[2]*obs_fieldav_dcls[0]/(2*obs_fieldav_cls[0]))**2+(obs_fieldav_cls[2]*obs_fieldav_dcls[1]/(2*obs_fieldav_cls[1]))**2))

    return lb, r_TM, sigma_r_TM, obs_fieldav_cls, obs_fieldav_dcls


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
            
            neff_indiv = compute_Neff(field_weights[:,n])
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
    


def knox_spectra(radprofs_auto, radprofs_cross=None, radprofs_gal=None, \
                 ngs=None, fsky=0.0000969, lmin=90., Nside=1024, mode='auto'):
    
    ''' 
    Compute dC_ell and corresponding cross terms/total signal to noise for a given auto or cross power spectrum. This uses
    the Knox formula from Knox (1995). I think in general this will underestimate the total error, because it assumes a gaussian 
    beam and that errors from different terms are uncorrelated and have uniform variance.
    '''

    sb_intensity_unit = u.nW/u.m**2/u.steradian
    npix = Nside**2
    pixel_solidangle = 49*u.arcsecond**2
    print(len(rbin))
    ells = rbin[:-1]*lmin
    d_ells = ells[1:]-ells[:-1]
    
    mode_counting_term = 2./(fsky*(2*ells+1))
    sb_sens_perpix = [33.1, 17.5]*sb_intensity_unit
    
    beam_term = np.exp((pixel_solidangle.to(u.steradian).value)*ells**2)
    noise = (4*np.pi*fsky*u.steradian)*sb_sens_perpix[band]**2/npix
    
    cl_noise = noise*beam_term
    
    if mode=='auto':
        dCl_sq = mode_counting_term*((radprofs_auto[0]) + cl_noise.value)**2
        snr_sq = (radprofs_auto[0])**2 / dCl_sq
        
        return snr_sq, dCl_sq, ells
    
    elif mode=='cross':
        
        snr_sq_cross_list, list_of_crossterms = [], []

         
        for i in range(len(radprofs_cross)):
            print(len(radprofs_auto[0]), len(cl_noise.value), len(radprofs_gal[i]), len(mode_counting_term))
            dCl_sq = mode_counting_term*((radprofs_cross[i])**2 +(radprofs_auto[0] + cl_noise.value)*(radprofs_gal[i] +ngs[i]**(-1)))
            snr_sq_cross = (radprofs_cross[i])**2 / dCl_sq
            snr_sq_cross_list.append(snr_sq_cross)

            cross_terms = [radprofs_cross[i]**2, radprofs_auto[0]*radprofs_gal[i], radprofs_auto[0]*ngs[i]**(-1), cl_noise.value*radprofs_gal[i], cl_noise.value*ngs[i]**(-1)]
            list_of_crossterms.append(cross_terms)

        return snr_sq_cross_list, dCl_sq, ells, list_of_crossterms

