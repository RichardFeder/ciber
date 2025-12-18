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
from ciber.core.powerspec_pipeline import *
from ciber.core.powerspec_utils import *
from ciber.io.ciber_data_utils import *
from ciber.plotting.plotting_fns import *


def generate_field_consistency_plots(cbps, all_inst=[1, 2], bandstrs=['J', 'H'], mag_lims=[17.5, 17.0], ifield_list=[4, 5, 6, 7, 8], \
                                     lmax_cov=10000, rescale_mode='full', startidx=2, endidx=-1, ybounds=[5, 4], textypos=[3.0, 2.5]):
    
    field_consistency_figs = []
    for inst in all_inst:

        obs_dict, mock_dict, cov_dict, cl_fpath = gather_fiducial_auto_ps_results(cbps, inst, nsim_mock=1000, save_cov=True, \
                                                                                 compute_cov=True, lmax_cov=2e5, startidx=startidx)


        fc_stats_dict, fig = ciber_field_consistency_test(inst, obs_dict['lb'], obs_dict['observed_recov_ps'], obs_dict['observed_field_average_cl'],\
                                                          mock_dict['all_mock_recov_ps'], mock_dict['all_mock_signal_ps'], mock_dict['mock_all_field_cl_weights'], \
                                                         mock_run_name=mock_dict['mock_run_name'], startidx=startidx, endidx=endidx, \
                                                         lmax_cov=lmax_cov, textypos=textypos[inst-1], ybound=ybounds[inst-1],\
                                                          mod_ratio=1.0, mod_ratio_min=0.3, mod_ell_max=3000, \
                                                         rescale_mode=rescale_mode, plot=False)

        field_consistency_figs.append(fig)
        
    return field_consistency_figs

def gather_fiducial_auto_ps_results(cbps, inst, nsim_mock=None, observed_run_name=None, mock_run_name=None,\
                                    mag_lim=None, ifield_list=[4, 5, 6, 7, 8], flatidx=0, \
                                   datestr_obs='111323', datestr_mock='112022', lmax_cov=2e5, \
                                   save_cov=True, startidx=1, compute_cov=False, ff_transition_mag=15.0, nsim_startidx=0):
    
    bandstr_dict = dict({1:'J', 2:'H'})
    maglim_default = dict({1:17.5, 2:17.0})
    
    bandstr = bandstr_dict[inst]
    
    if mag_lim is None:
        mag_lim = maglim_default[inst]
        
    if observed_run_name is None:
        # observed_run_name = 'observed_'+bandstr+'lt'+str(mag_lim)+'_072424_quadoff_grad_fcsub_order2'
        observed_run_name = 'observed_'+bandstr+'lt'+str(mag_lim)+'_080924_quadoff_grad_fcsub_order2'

    if mock_run_name is None:
        if mag_lim <= ff_transition_mag:
            mock_mode = 'maskmkk'
        else:
            mock_mode = 'mkkffest'        
        mock_run_name = 'mock_'+bandstr+'lt'+str(mag_lim)+'_072324_'+mock_mode+'_quadoff_grad_fcsub_order=2_theta0p01'

    
    lb, observed_recov_ps, observed_recov_dcl_perfield, observed_field_average_cl, observed_field_average_dcl,\
        mock_mean_input_ps, mock_all_field_averaged_cls, mock_all_field_cl_weights, \
            all_mock_recov_ps, all_mock_signal_ps = process_observed_powerspectra(cbps, datestr_obs, ifield_list, inst, \
                                                                                    observed_run_name, mock_run_name, nsim_mock, \
                                                                                 flatidx=flatidx, apply_field_weights=True, \
                                                                                 datestr_mock=datestr_mock, pct68=True) 
    
    
    obs_dict = dict({'lb':lb, 'observed_run_name':observed_run_name, 'mag_lim':mag_lim, 'observed_recov_ps':observed_recov_ps, 'observed_recov_dcl_perfield':observed_recov_dcl_perfield, \
                    'observed_field_average_cl':observed_field_average_cl, 'observed_field_average_dcl':observed_field_average_dcl})
    
    mock_dict = dict({'mock_run_name':mock_run_name, 'mock_mean_input_ps':mock_mean_input_ps, 'mock_all_field_averaged_cls':mock_all_field_averaged_cls, \
                     'mock_all_field_cl_weights':mock_all_field_cl_weights, 'all_mock_recov_ps':all_mock_recov_ps, 'all_mock_signal_ps':all_mock_signal_ps})
    
    cov_dict, cl_fpath = None, None
    
    if compute_cov:
        lb_mask, all_cov_indiv_full,\
            all_resid_data_matrices,\
                resid_joint_data_matrix = compute_mock_covariance_matrix(lb, inst, all_mock_recov_ps, mock_all_field_averaged_cls, \
                                                                        ifield_list=ifield_list, lmax=lmax_cov, save=save_cov,\
                                                                         mock_run_name=mock_run_name, plot=False, startidx=startidx, nsim_startidx=nsim_startidx)

        cov_dict = dict({'lb_mask':lb_mask, 'all_cov_indiv_full':all_cov_indiv_full, 'all_resid_data_matrices':all_resid_data_matrices, \
                        'resid_joint_data_matrix':resid_joint_data_matrix})

        if save_cov:
            print('Saving covariance')
            cl_fpath = save_weighted_mock_cl_file(lb, inst, mock_run_name, mock_mean_input_ps, mock_all_field_averaged_cls, mock_all_field_cl_weights, all_mock_recov_ps, \
                            all_mock_signal_ps)
            print('Saved to ', cl_fpath)
        
    
    return obs_dict, mock_dict, cov_dict, cl_fpath


def ciber_field_consistency_test(inst, lb, observed_recov_ps, observed_field_average_cl, all_mock_recov_ps, all_mock_signal_ps, mock_all_field_cl_weights, \
                                lmax_cov=10000, startidx=1, endidx=-1, mode='chi2', \
                                ybound=5, mock_run_name=None, datestr='111323', ifield_list = [4, 5, 6, 7, 8], \
                                mod_ratio=4, mod_ratio_min=0.3, rescale_mode='diag', mod_ell_max=2000, xlim=[150, 1.1e5], \
                                textxpos = 200, textypos = None, xlim_zoom=[-1.5, 1.5], plot=True, verbose=True, \
                                figsize=(10, 7)):
    
    ciber_field_dict = dict({4:'elat10', 5:'elat30', 6:'Bootes B', 7:'Bootes A', 8:'SWIRE'})

    all_chistat = []
    
    lamdict = dict({1:1.1, 2:1.8})
    lbmask_chistat = (lb < lmax_cov)*(lb >= lb[startidx])
    
    
    fc_stats_dict = compute_field_consistency_stats(inst, lb, observed_recov_ps, observed_field_average_cl, all_mock_recov_ps, all_mock_signal_ps, mock_all_field_cl_weights, \
                                                   lmax_cov=lmax_cov, startidx=startidx, endidx=endidx, mode=mode, \
                                                    mock_run_name=mock_run_name, datestr=datestr, ifield_list = ifield_list, \
                                                     mod_ratio=mod_ratio, mod_ratio_min=mod_ratio_min, rescale_mode=rescale_mode, mod_ell_max=mod_ell_max, plot=plot, verbose=verbose)
    
    
    print('observed recov ps shape:', observed_recov_ps.shape)
    print('observed field average cl shape:', observed_field_average_cl.shape)
    
    fig = plot_ciber_field_consistency(inst, fc_stats_dict, lb, observed_recov_ps, observed_field_average_cl, ifield_list=ifield_list, figsize=figsize, \
                                textxpos=200, textypos=textypos, startidx=startidx, endidx=endidx, lmax_cov=lmax_cov, mode=mode, ybound=ybound, lmin_cov=250, xlim=[150, 1e5])
    
    
    return fc_stats_dict, fig

def rescale_covariance_sims(lb, ifield_list, all_cov_indiv, observed_recov_cl, observed_field_average_cl, all_mock_recov_ps, all_mock_signal_ps, \
                           mean_input_mock_ps, lbmask_chistat, mod_ratio=2, mod_ratio_min=0.5, startidx=0, mod_ell_max=3000, rescale_mode='diag'):

    ''' These should truncate the covariances of anything less startidx, but not sub-matrix for consistency test. '''
    lb_trunc = lb[startidx:]
    
    all_cov_indiv_rescale = all_cov_indiv.copy()

    all_ratio_obs_mock_pos, all_std_recov_mock_ps = [], []

    mean_input_mock_ps = np.mean(all_mock_signal_ps, axis=(0, 1))

    ratio_obs_mock_pos = (observed_field_average_cl/mean_input_mock_ps)[startidx:]
    print('ratio obs mock:', ratio_obs_mock_pos)
    gtr_modl_mask = np.logical_or((ratio_obs_mock_pos > mod_ratio), (ratio_obs_mock_pos > mod_ratio_min))*(lb_trunc < mod_ell_max)
    print('which gtr modl:', gtr_modl_mask)
    which_gtr_modl = np.where(gtr_modl_mask)[0]
    ratio_obs_mock_pos[~gtr_modl_mask] = 1.0
    print('ratio obs mock pos:', ratio_obs_mock_pos)

    for fieldidx, ifield in enumerate(ifield_list):
        
#         ratio_obs_mock_pos = (observed_recov_cl[fieldidx]/mean_input_mock_ps)[startidx:]
#         print('ratio obs mock:', ratio_obs_mock_pos)
#         gtr_modl_mask = (ratio_obs_mock_pos > mod_ratio)*(lb_trunc < mod_ell_max)
#         print('which gtr modl:', gtr_modl_mask)
#         which_gtr_modl = np.where(gtr_modl_mask)[0]
#         ratio_obs_mock_pos[~gtr_modl_mask] = 1.0
#         print('ratio obs mock pos:', ratio_obs_mock_pos)

        std_recov_mock_ps = 0.5*(np.nanpercentile(all_mock_recov_ps[:,fieldidx], 84, axis=0)-np.nanpercentile(all_mock_recov_ps[:,fieldidx], 16, axis=0))[startidx:]
        # std_recov_mock_ps = np.std(all_mock_recov_ps[:,fieldidx], axis=0)[startidx:]
        std_recov_mock_ps *= np.sqrt(ratio_obs_mock_pos)
    
        for which in which_gtr_modl:

            if rescale_mode=='diag':
                all_cov_indiv_rescale[fieldidx,which, which] *= ratio_obs_mock_pos[which]
            else:
                all_cov_indiv_rescale[fieldidx, which, :] *= ratio_obs_mock_pos[which]
                all_cov_indiv_rescale[fieldidx, :, which] *= ratio_obs_mock_pos[which]

        all_std_recov_mock_ps.append(std_recov_mock_ps)
        all_ratio_obs_mock_pos.append(ratio_obs_mock_pos)
        
    return all_cov_indiv_rescale, all_ratio_obs_mock_pos, all_std_recov_mock_ps, mean_input_mock_ps


def compute_field_consistency_stats(inst, lb, observed_recov_ps, observed_field_average_cl, all_mock_recov_ps, all_mock_signal_ps, mock_all_field_cl_weights, \
                                   lmax_cov=10000, startidx=1, endidx=None, mode='chi2', \
                                    ybound=5, mock_run_name=None, datestr='111323', ifield_list = [4, 5, 6, 7, 8], \
                                         mod_ratio=4, rescale_mode='diag', mod_ell_max=2000, plot=True, verbose=True, mod_ratio_min=0.3):
    
    ciber_field_dict = dict({4:'elat10', 5:'elat30', 6:'Bootes B', 7:'Bootes A', 8:'SWIRE'})

    all_chistat, observed_chi2_red = [],[]
    
    lb_trunc = lb[startidx:]
    
    lbmask_chistat = (lb < lmax_cov)*(lb >= lb[startidx])
    
    gtr_idxs = next(x[0] for x in enumerate(lb_trunc) if x[1] > lmax_cov)
    endidx_cov = np.min(gtr_idxs)

    mean_input_mock_ps = np.mean(all_mock_signal_ps, axis=(0, 1))
        
    if mock_run_name is not None:
        
        cov_joint, corr_joint,\
            all_cov_indiv_full,\
                all_corr_indiv_full = load_covariance_field_consistency(inst, datestr,\
                                                                        mock_run_name,\
                                                                        ifield_list=ifield_list)
        
        all_inv_cov_indiv_lbmask, all_inv_cov_indiv_mod = [np.zeros_like(all_cov_indiv_full) for x in range(2)]

    else:
        all_cov_indiv_full, cov_joint = None, None
        

    if plot:
        plot_map(corr_joint, title='Joint correlation matrix')
        plot_map(cov_joint, title='Joint covariance matrix')

        for x in range(len(all_corr_indiv_full)):
            if x==0:
                print('all corr indiv full has shape ', all_corr_indiv_full[x].shape)
                plot_map(all_corr_indiv_full[x], title='Correlation, ifield '+ciber_field_dict[x+4], cmap='bwr', figsize=(4,4))
                plot_map(all_cov_indiv_full[x], title='Covariance, ifield '+str(ciber_field_dict[x+4]), cmap='jet', figsize=(4,4))

    
    ratio_obs_mock_full = None
    
    if all_cov_indiv_full is not None:
        
        all_cov_indiv_rescale, all_ratio_obs_mock_pos,\
                all_std_recov_mock_ps, mean_input_mock_ps = rescale_covariance_sims(lb, ifield_list, all_cov_indiv_full, observed_recov_ps, observed_field_average_cl, all_mock_recov_ps, all_mock_signal_ps, \
                                                                mean_input_mock_ps, lbmask_chistat, startidx=startidx, mod_ratio=mod_ratio, mod_ell_max=mod_ell_max, \
                                                                                   rescale_mode=rescale_mode, mod_ratio_min=mod_ratio_min)
        
        if verbose:
            print('After rescale covariance sims, all_cov_indiv_full has shape ', np.array(all_cov_indiv_full).shape)
            print('After rescale covariance sims, all_cov_indiv_rescale has shape ', np.array(all_cov_indiv_rescale).shape)

        # at this stage, take truncated covariance matrices and isolate consistency test lb range
        
        lbmask_chistat_trunc = (lb_trunc < lmax_cov)
        
                
        all_cov_indiv_rescale_trunc = np.array([all_cov_indiv_rescale[fieldidx,:endidx_cov,:endidx_cov] for fieldidx in range(len(ifield_list))])
        # not rescaled for mocks
        all_cov_indiv_trunc = np.array([all_cov_indiv_full[fieldidx,:endidx_cov,:endidx_cov] for fieldidx in range(len(ifield_list))])
        
        all_inv_cov_indiv_mod = np.ones_like(all_cov_indiv_rescale_trunc)
        
        if verbose:
            print('all_inv_cov_indiv_lbmask.shape', all_inv_cov_indiv_lbmask.shape)
        
        for fieldidx, ifield in enumerate(ifield_list):
            
            if plot:
                plot_map(all_cov_indiv_rescale_trunc[fieldidx], figsize=(4,4), title='cov rescaled ifield '+str(ifield))
            
            if verbose:
                print('condition number on indiv is ', np.linalg.cond(all_cov_indiv_rescale_trunc[fieldidx]))
            all_inv_cov_indiv_mod[fieldidx] = np.linalg.inv(all_cov_indiv_rescale_trunc[fieldidx])
    
        
        
        
    chistat_perfield_mock, pte_perfield_mock, all_chistat_largescale_mock = mock_consistency_chistat(lb, all_mock_recov_ps, mock_all_field_cl_weights, lmax=lmax_cov,\
                                                                                                mode=mode, all_cov_indiv_full=all_cov_indiv_trunc, cov_joint=None, startidx=startidx, \
                                                                                               ifield_list=ifield_list)
    
    
    all_chistat, all_chistat_largescale = [], []
    
    for fieldidx, ifield in enumerate(ifield_list):

        resid = observed_recov_ps[fieldidx]-observed_field_average_cl

        if mode=='chi2':
            chistat_mean_cl_mockstd = np.multiply(resid[lbmask_chistat].transpose(), np.dot(all_inv_cov_indiv_mod[fieldidx], resid[lbmask_chistat]))
        elif mode=='chi':
            chistat_mean_cl_mockstd = resid/tot_std_ps
        
        all_chistat.append(chistat_mean_cl_mockstd)

        if all_cov_indiv_full is not None:
            chistat_largescale = np.array(all_chistat[fieldidx])
        else:
            chistat_largescale = np.array(all_chistat[fieldidx])[lbmask_chistat]
                    
        all_chistat_largescale.append(chistat_largescale)
        
    fc_stats_dict = dict({'chistat_perfield_mock':chistat_perfield_mock, 'pte_perfield_mock':pte_perfield_mock, 'all_chistat_largescale':all_chistat_largescale, \
                         'all_chistat_largescale_mock':all_chistat_largescale_mock, 'all_cov_indiv_full':all_cov_indiv_full, 'all_corr_indiv_full':all_corr_indiv_full, 'all_std_recov_mock_ps':all_std_recov_mock_ps, 'all_inv_cov_indiv_mod':all_inv_cov_indiv_mod,'all_inv_cov_indiv_lbmask':all_inv_cov_indiv_lbmask, \
                         'all_cov_indiv_rescale':all_cov_indiv_rescale, 'all_ratio_obs_mock_pos':all_ratio_obs_mock_pos, 'mean_input_mock_ps':mean_input_mock_ps})
    
    return fc_stats_dict


def process_mock_results_with_without_fferr(cbps, inst_list=[1, 2], nsim_mock_perfect=1000, nsim_mock_est=1000, flatidx=0, \
                                           mag_lim=[17.5, 17.0], datestr_mock='112022', \
                                           ifield_list = [4, 5, 6, 7, 8], mock_run_name_list=None, mock_run_name_dff_list=None):
    # cbps = CIBER_PS_pipeline()
    bandstr_dict = dict({1:'J', 2:'H'})
    
    all_av_mock_signal_ps, all_av_mock_signal_ps_dff,\
        all_field_averaged_recov_ps, all_field_averaged_recov_ps_dff = [[] for x in range(4)]

    for idx, inst in enumerate(inst_list):
        
        band = bandstr_dict[inst]
    
        if mock_run_name_list is None:
            mock_run_name = 'mock_'+str(band)+'lt'+str(mag_lim[inst-1])+'_042223_perfectFF_quadoff_grad'
        else:
            mock_run_name = mock_run_name_list[idx]
            
        print('mock run name;', mock_run_name)

        mock_mean_input_ps, all_mock_signal_ps, all_mock_recov_ps,\
                    mock_all_field_averaged_cls, mock_all_field_cl_weights, lb = process_mock_powerspectra(cbps, datestr_mock, ifield_list, \
                                                                                                        inst, mock_run_name, nsim_mock_perfect, flatidx=flatidx, \
                                                                                                        sim_test_fpath=None, pct68=True)

        if mock_run_name_dff_list is None:
            mock_run_name_dff = 'mock_'+str(band)+'lt'+str(mag_lim[inst-1])+'_042523_mkkffest_quadoff_grad_labFFreally_ptsrcffnoise'
        else:
            mock_run_name_dff_list = mock_run_name_dff_list[idx]

        mock_mean_input_ps_dff, all_mock_signal_ps_dff, all_mock_recov_ps_dff,\
                    mock_all_field_averaged_cls_dff, mock_all_field_cl_weights_dff, lb = process_mock_powerspectra(cbps, datestr_mock, ifield_list, \
                                                                                                        inst, mock_run_name_dff, nsim_mock_est, flatidx=flatidx, \
                                                                                                        sim_test_fpath=None, pct68=True)


        all_av_mock_signal_ps.append(mock_mean_input_ps)
        all_av_mock_signal_ps_dff.append(mock_mean_input_ps_dff)

        all_field_averaged_recov_ps.append(mock_all_field_averaged_cls)
        all_field_averaged_recov_ps_dff.append(mock_all_field_averaged_cls_dff)
        
    return lb, all_av_mock_signal_ps, all_av_mock_signal_ps_dff, all_field_averaged_recov_ps, all_field_averaged_recov_ps_dff

    


def mock_consistency_chistat(lb, all_mock_recov_ps, mock_all_field_cl_weights, ifield_list = [4, 5, 6, 7, 8], lmax=10000, mode='chi2', all_cov_indiv_full=None, \
                            cov_joint=None, startidx=1):
    ''' 
    mode either 'chi2' or 'chi'
    
    '''
    
    # lbmask_chistat = (lb < lmax)*(lb > lb[0])
            
    lbmask_chistat = (lb < lmax)*(lb >= lb[startidx])
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

def plot_ciber_corr_with_monopole(include_auto=True, include_cross=True, zl_levels = None, dgl_levels = None, plot=True, \
                                 figsize=(6, 5), cross_run_name=None, datestr='111323', lmin=500, lmax=1000, lbidx=None, \
                                 nsim_mock=2000, flatidx=0, observed_run_name=None, Mkk_obj=None, cbps=None):
    
    
    cbps = CIBER_PS_pipeline()
    if Mkk_obj is None:
        Mkk_obj = Mkk_bare(dimx=1024, dimy=1024, ell_min=180.*(1024./1024.), nbins=25)
    
    ifield_list=[4, 5, 6, 7, 8]
    lamdict = dict({1:1.1, 2:1.8})
    
    auto_colors = ['b', 'r']
    fig = plt.figure(figsize=figsize)
    
    if dgl_levels is not None:
        fglab = 'DGL'
        sblevels = dgl_levels
    elif zl_levels is not None:
        fglab = 'ZL'
        sblevels = zl_levels

    else:
        print('need to choose either ZL or DGL, exiting')
        return None
    
    for plotidx, lbidx in enumerate(np.arange(2, 8)):
        
        plt.subplot(2, 3, plotidx+1)
    
        if include_auto:

            for inst in [1, 2]:

                obs_dict, mock_dict, cov_dict, cl_fpath = gather_fiducial_auto_ps_results(cbps, inst, nsim_mock=nsim_mock, \
                                                                                 observed_run_name=observed_run_name, \
                                                                                 flatidx=flatidx)

                lb = obs_dict['lb']

                if lbidx is not None:
                    lbmask = (lb==lb[lbidx])
                    pf = lb[lbidx]*(lb[lbidx]+1)/(2*np.pi)

                else:
                    lbmask = (lb > lmin)*(lb < lmax)
                    pf = 1

                auto_cl_levels, auto_clerr_levels = [np.zeros(len(ifield_list)) for x in range(2)]
                for fieldidx, ifield in enumerate(ifield_list):
                    auto_cl = obs_dict['observed_recov_ps'][fieldidx]
                    std_recov_mock_ps = 0.5*(np.percentile(mock_dict['all_mock_recov_ps'][:,fieldidx,:], 84, axis=0)-np.percentile(mock_dict['all_mock_recov_ps'][:,fieldidx,:], 16, axis=0))

                    auto_cl_lbmask = auto_cl[lbmask]
                    std_cl_lbmask = std_recov_mock_ps[lbmask]

                    auto_cl_levels[fieldidx] = np.mean(auto_cl_lbmask)
                    auto_clerr_levels[fieldidx] = (np.sqrt(np.sum(std_cl_lbmask**2))/np.sum(lbmask))
                
                label = str(lamdict[inst])+' $\\mu$m auto'
                plt.errorbar(sblevels, pf*auto_cl_levels, yerr=pf*auto_clerr_levels, label=label, fmt='o', capsize=3.5, markersize=4, color=auto_colors[inst-1], capthick=1.5)

        if include_cross:


            cross_cl_file = np.load(config.ciber_basepath+'data/input_recovered_ps/'+datestr+'/TM1/'+cross_run_name+'/input_recovered_ps_estFF_simidx0.npz')

            lb = cross_cl_file['lb']
            if lbidx is not None:
                lbmask = (lb==lb[lbidx])
                
                pf = lb[lbidx]*(lb[lbidx]+1)/(2*np.pi)
            else:
                lbmask = (lb > lmin)*(lb < lmax)
                pf = 1.


            cross_cl_levels, cross_clerr_levels = [np.zeros(len(ifield_list)) for x in range(2)]

            for fieldidx, ifield in enumerate(ifield_list):

                cl1d_obs = cross_cl_file['recovered_ps_est_nofluc'][fieldidx]
                dcl1d_obs = cross_cl_file['recovered_dcl'][fieldidx]

                frac_knox_errors = return_frac_knox_error(lb, Mkk_obj.delta_ell)
                knox_err = frac_knox_errors*cl1d_obs

                dcl1d_obs = np.sqrt(dcl1d_obs**2+knox_err**2)

                cross_cl_lbmask = cl1d_obs[lbmask]
                cross_std_cl_lbmask = dcl1d_obs[lbmask]

                cross_cl_levels[fieldidx] = np.mean(cross_cl_lbmask)
                cross_clerr_levels[fieldidx] = np.sqrt(np.sum(cross_std_cl_lbmask**2))/np.sum(lbmask)


            label = '1.1 $\\mu$m $\\times$ 1.8 $\\mu$m'
            plt.errorbar(sblevels, pf*cross_cl_levels, yerr=pf*cross_clerr_levels, label=label, color='purple', fmt='o', capsize=3.5, markersize=4, capthick=1.5)

        if plotidx==0 or plotidx==3:
            plt.ylabel('$D_{\\ell}^{CIBER}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=12)
            
        if plotidx > 2:
            if zl_levels is not None:
                plt.xlabel('$I_{ZL}(\\lambda=1.1$ $\\mu$m)\n[nW m$^{-2}$ sr$^{-1}$]', fontsize=12)
            
            if dgl_levels is not None:
                plt.xticks([0., 0.5, 1.0, 1.5])
                plt.xlabel('$I_{100 \\mu m}$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=12)

        else:
            if zl_levels is not None:
                plt.xticks([300, 400, 500], ['', '', ''])
            else:
                plt.xticks([0., 0.5, 1.0, 1.5], ['', '', '', ''])
        plt.title(str(int(Mkk_obj.binl[lbidx]))+'$<\\ell<$'+str(int(Mkk_obj.binl[lbidx+1])), fontsize=10)
        if zl_levels is not None:
            plt.xlim(250, 550)
        else:
            plt.xlim(0, 1.5)
        plt.grid(alpha=0.5)

    plt.legend(fontsize=12, bbox_to_anchor=[0.7, 2.6], ncol=3)

    if plot:
        plt.show()
    
    return fig


