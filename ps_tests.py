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

