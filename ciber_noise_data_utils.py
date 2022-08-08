import numpy as np
import astropy.io.fits as fits
import astropy.wcs as wcs
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting
import pandas as pd
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
from astropy.table import Table
import glob
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from ciber_powerspec_pipeline import CIBER_PS_pipeline
from datetime import datetime

from mkk_parallel import *
from plotting_fns import plot_map
from cross_spectrum_analysis import *

def chop_up_masks(sigmap, nside=5, ravel=False, show=False, verbose=True):
    
    nx, ny = sigmap.shape[0], sigmap.shape[1]
    
    masks = np.zeros((nside,nside, sigmap.shape[0], sigmap.shape[1]))
    for i in range(nside):
        for j in range(nside):
            masks[i,j,i*nx//nside:(i+1)*nx//nside,j*ny//nside:(j+1)*ny//nside] = 1
            
    masks = np.reshape(masks, (nside**2, nx, ny))        
    
    if verbose:
        print('masks has shape ', masks.shape)
    
    if show:
        plt.figure()
        plt.imshow(masks[0])
        plt.colorbar()
        plt.show()
    
    return masks

def compute_exp_difference(inst, cal_facs=None, pathA=None, pathB=None, expA=None, expB=None, data_path=None,idxA=None, idxB=None, mask=None, mode='flight', \
                          plot_diff_hist=False):
    
    ''' Compute exposure differences and return mean values of (masked) images '''

    if expA is None:
        if pathA is None:
            pathA = data_path+'/'+mode+'Map_'+str(idxA)+'.FITS'
        expA = fits.open(pathA)[0].data

    if expB is None:
        if pathB is None:
            pathB = data_path+'/'+mode+'Map_'+str(idxB)+'.FITS'
        expB = fits.open(pathB)[0].data
    
    
    if cal_facs is not None:
        expA *= cal_facs[inst]
        expB *= cal_facs[inst]
        
    meanA = np.median(expA[mask==1])
    meanB = np.median(expB[mask==1])
        
    if plot_diff_hist:
        plt.figure()
        _, bins, _ = plt.hist(expA.ravel()[mask.ravel()==1], bins=30, histtype='step')
        plt.hist(expB.ravel()[mask.ravel()==1], bins=bins, histtype='step')
        plt.axvline(meanA, label='meanA')
        plt.axvline(meanB, label='meanB')

        plt.yscale('log')
        plt.show()
            
    if mask is None:
        mask = np.ones_like(expA)
    
    exp_diff = mask*(expA-expB)

    mean_flight = 0.5*(expA+expB)
    
    return exp_diff, meanA, meanB, mean_flight

def compute_fourier_weights(cl2d_all, stdpower=2):
    
    fw_std = np.std(cl2d_all, axis=0)
    fourier_weights = 1./fw_std**stdpower
    fourier_weights /= np.max(fourier_weights)
    
    mean_cl2d = np.mean(cl2d_all, axis=0)
    
    return mean_cl2d, fourier_weights


def compute_masked_quantities_by_quadrant(ifield, inst, labidx=1, dimx=1024, dimy=1024, nregion=2, n_ps_bin=25, \
                                    noise_validation_path = '/Volumes/Seagate Backup Plus Drive/Toolkit/Mirror/Richard/noise_model_validation_data/', \
                                    n_mkk_sim=10, compute_inverse_mkks=True, plot_labexp=False, \
                                    mode='lab', nfr=10, ifield_list_flight=[4, 6, 7, 8], sig=3, nitermax=10):
    
    ''' 
    
    Notes
    -----
        In mode=='lab', cls_noise will correspond to a single field, per lab exposure per quadrant
        In mode=='flight', cls_noise is per field per quadrant (i.e. all fields computed in for loop)

    '''
    cbps_quad = CIBER_PS_pipeline(dimx=dimx//nregion, dimy=dimy//nregion, n_ps_bin=n_ps_bin)
    
    xmins = [0, dimx//2, 0, dimy//2]
    xmaxs = [dimx//2, dimx, dimx//2, dimx]
    ymins = [0, 0, dimy//2, dimy//2]
    ymaxs = [dimy//2, dimy//2, dimy, dimy]
    
    labexp_basepath = noise_validation_path+'TM'+str(inst)+'/validationHalfExp/'
    if inst==2:
        labexp_basepath += 'alldarkEXP_20211001/'


    if mode=='lab':
        data_fpaths = glob.glob(labexp_basepath+'field'+str(ifield)+'/full/*.FITS')

        print('labexp fpaths is ', data_fpaths)

    elif mode=='flight':
        data_fpaths = ['data/fluctuation_data/TM'+str(inst)+'/short_'+str(nfr)+'frames/field'+str(fieldidx) for fieldidx in ifield_list_flight]
    
    inverse_quad_mkks = None
    if compute_inverse_mkks:
        inverse_quad_mkks = np.zeros((len(data_fpaths), 4, cbps_quad.n_ps_bin, cbps_quad.n_ps_bin))
        
    mameans = np.zeros((len(data_fpaths), 4))
    cls_2d_noise = np.zeros((len(data_fpaths), 4, cbps_quad.dimx, cbps_quad.dimy))
    cls_noise = np.zeros((len(data_fpaths), 4, cbps_quad.n_ps_bin))

    for lidx, data_fpath in enumerate(data_fpaths):

        mask_inst = cbps.load_mask(ifield=ifield, inst=inst, masktype='maskInst_clean', inplace=False)
        if mode=='flight':
            pathA, pathB = data_fpath+'/flight1.FITS', data_fpath+'/flight2.FITS'

            exp_image, meanA, meanB, mean_flight = compute_exp_difference(inst, cal_facs=cbps.cal_facs, pathA=pathA, pathB=pathB, mask=mask_inst, plot_diff_hist=False)

        elif mode=='lab':
            exp_image = cbps.cal_facs[inst]*fits.open(data_fpath)[0].data
        
        new_mask = iter_sigma_clip_mask(exp_image, sig=sig, nitermax=nitermax, initial_mask=mask_inst.astype(np.int))
        mask_inst *= new_mask

        if plot_labexp:
            plot_map(exp_image*mask_inst*new_mask, title='lab exposure (masked) '+str(lidx))

        
        quads = np.array([exp_image[xmins[i]:xmaxs[i],ymins[i]:ymaxs[i]] for i in range(len(xmins))])
        quad_masks = np.array([mask_inst[xmins[i]:xmaxs[i],ymins[i]:ymaxs[i]] for i in range(len(xmins))])

        for qidx, quad_mask in enumerate(quad_masks):
            masked_quad = np.ma.array(quads[qidx], mask=~quad_mask.astype(np.bool))
            mamean = np.ma.mean(masked_quad)
            mameans[lidx,qidx] = mamean

            l2d, cl2d = get_power_spectrum_2d(quad_mask*(quads[qidx]-mamean), pixsize=cbps_quad.Mkk_obj.pixsize)

            if lidx==0 and plot_labexp:
                plot_map(cl2d, title='lidx = '+str(lidx)+', qidx = '+str(qidx))
                plot_map(quads[qidx]*quad_mask, title='signal*mask, lidx = '+str(lidx)+', qidx = '+str(qidx))

            if compute_inverse_mkks:
                quad_mkk = cbps_quad.Mkk_obj.get_mkk_sim(quad_mask, nsims=n_mkk_sim, store_Mkks=False)
                inverse_quad_mkk = compute_inverse_mkk(quad_mkk)
                inverse_quad_mkks[lidx, qidx] = inverse_quad_mkk
                
            
            cls_2d_noise[lidx, qidx] = cl2d
            lb, cl_noise, clerr_noise = get_power_spec(quad_mask*(quads[qidx]-mamean), lbinedges=cbps_quad.Mkk_obj.binl, lbins=cbps_quad.Mkk_obj.midbin_ell)

            cls_noise[lidx, qidx] = cl_noise
            
    return mameans, cls_2d_noise, cls_noise, inverse_quad_mkks, lb


def compute_lab_flight_exp_differences(ifield, inst, cbps, base_path=None, mask=None, use_mask=True, mask_fpath=None, J_mag_lim=17.5, \
                                      stdpower=1, n_image=10, verbose=True, plot=True):
    
    if base_path is None:
        base_path = '/Volumes/Seagate Backup Plus Drive/Toolkit/Mirror/Richard/noise_model_validation_data/'
    if mask_fpath is None:
        mask_fpath = cbps.data_path+'/TM'+str(inst)+'/mask/field'+str(ifield)+'_TM'+str(inst)+'_strmask_Jlim='+str(J_mag_lim)+'_040521.fits'
    data_path = base_path+'TM'+str(inst)+'/validationHalfExp/field'+str(ifield)

    print('base_path is ', base_path)
    print('mask_fpath is ', mask_fpath)
    print('data_path is ', data_path)
    
    imarray_shape = (n_image, cbps.dimx, cbps.dimy)
    psarray_shape = (n_image, cbps.n_ps_bin)
    lab_diffs, cl2ds_lab = [np.zeros(imarray_shape) for x in range(2)]

    pathA = data_path+'/flightMap_2.FITS'
    pathB = data_path+'/flightMap_1.FITS'
    
    # load in astronomical + instrument mask
    if mask is None and use_mask:
        mask = cbps.load_mask(ifield=ifield, inst=inst, masktype='maskInst_clean', inplace=False)
    strmask = cbps.load_mask(ifield, inst, mask_fpath=mask_fpath, masktype='strmask', inplace=False)
    mask *= strmask
    
    cl_diffs_lab, cl_diffs_phot = [np.zeros(psarray_shape) for x in range(2)]
    
    flight_exp_diff, meanA, meanB = compute_exp_difference(ifield, inst, cal_facs=cbps.cal_facs, pathA=pathA, pathB=pathB, mask=mask, mode='flight')
    if verbose:
        print('meanA, meanB ', meanA, meanB)
        print(cbps.field_nfrs[ifield], cbps.field_nfrs[ifield]//2)
    
    shot_sigma_map = cbps.compute_shot_sigma_map(inst, meanA*np.ones_like(flight_exp_diff), nfr=cbps.field_nfrs[ifield]//2)
    
    dsnmaps = shot_sigma_sb_map*np.sqrt(2)*np.random.normal(0, 1, size=imarray_shape)



    for i in range(n_image):
        str_num = '{0:03}'.format(i+1)
        
        pathA = data_path+'/first/labMap_'+str_num+'_1.FITS'
        pathB = data_path+'/second/labMap_'+str_num+'_2.FITS'
        
        lab_exp_diff, labmeanA, labmeanB = compute_exp_difference(ifield, inst, mask=mask, cal_facs=cbps.cal_facs, pathA=pathA, pathB=pathB)
        lab_exp_diff += mask*dsnmaps[i]
        lab_diffs[i] = lab_exp_diff

        l2d, cl2dlab = get_power_spectrum_2d(lab_diffs[i]-np.mean(lab_diffs[i]))
        cl2ds_lab[i] = cl2dlab

    # compute mean 2D power spectrum and compute inverse variance weights
    mean_cl2d = np.mean(cl2ds_lab, axis=0)
    fw_std = np.std(cl2ds_lab, axis=0)
    fourier_weights = 1./fw_std**stdpower
    fourier_weights /= np.nanmax(fourier_weights)
    
    if plot:
        fcl2d = plot_map(np.log10(mean_cl2d), title='$\\log_{10}(C(\\ell_x,\\ell_y))$, '+str(cbps.ciber_field_dict[ifield]), noxticks=True, noyticks=True, xlabel='$\\ell_x$', ylabel='$\\ell_y$', return_fig=True)
        fwf = plot_map(np.log10(fourier_weights), title='$\\log_{10}(w(\\ell_x,\\ell_y))$, '+str(cbps.ciber_field_dict[ifield]),noxticks=True, noyticks=True,  xlabel='$\\ell_x$', ylabel='$\\ell_y$',return_fig=True)
        fwcl2d = plot_map(np.log10(fourier_weights*mean_cl2d), title='$\\log_{10}(w(\\ell_x,\\ell_y)C(\\ell_x,\\ell_y))$, '+str(cbps.ciber_field_dict[ifield]), noxticks=True, noyticks=True,  xlabel='$\\ell_x$', ylabel='$\\ell_y$', return_fig=True)

    # now that we have the fourier weights, let's compute the FW'd lab power spectra
    for i in range(n_image):

        lb, cl_diff_phot, _ = get_power_spec(mask*dsnmaps[i], weights=fourier_weights, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
        lb, cl_diff_lab, _ = azim_average_cl2d(cl2ds_lab[i], l2d, weights=fourier_weights, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)

        cl_diffs_lab[i] = cl_diff_lab/2.
        cl_diffs_phot[i] = cl_diff_phot/2.
        
    # reduce by factor of sqrt(2) to account for difference increasing noise by sqrt(2)
    lb, cl_diff_flight, _ = get_power_spec(flight_exp_diff-np.mean(flight_exp_diff), weights=fourier_weights, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
    cl_diff_flight /= 2.
    
    if verbose:
        print('mean cl_diff_lab : ', np.mean(np.array(cl_diffs_lab), axis=0))
        print('mean cl_diff_phot : ', np.mean(np.array(cl_diffs_phot), axis=0))
        print('cl_diff_flight : ', cl_diff_flight)
        
    return lb, cl_diff_flight, cl_diffs_lab, cl_diffs_phot


def compute_nframe_differences(cbps, ifield_list, nframe=10, inst=1, base_path=None, verbose=True):
    
    if base_path is None:
        base_path = '/Volumes/Seagate Backup Plus Drive/Toolkit/Mirror/Richard/noise_model_validation_data/'
    elat30_lab_path = base_path+'TM'+str(inst)+'/validationHalfExp/field5'
    base_fluc_path = '/Volumes/Seagate Backup Plus Drive/Toolkit/Mirror/Richard/ciber_fluctuation_data/'
    
    print(cbps.cal_facs[inst])
    cl_diffs_flight, cl_diffs_shotnoise, cl_diffs_shotnoise_elat30 = [], [], []
    if inst==2:
        n_lab_exp = 6
    elif inst==1:
        n_lab_exp = 10
        
    for i, ifield in enumerate(ifield_list):
        
        mask_fpath = base_fluc_path+'/TM'+str(inst)+'/masks/joint_mask_with_ffmask_ifield'+str(ifield)+'_inst'+str(inst)+'.fits'
        mask = fits.open(mask_fpath)['joint_mask_'+str(ifield)].data
        
        if ifield==5:
            noise_model = cbps.load_noise_Cl2D(noise_fpath='data/fluctuation_data/TM'+str(inst)+'/noiseCl2D/field'+str(ifield)+'_noiseCl2D_110421.fits', ifield=ifield, inst=inst, inplace=False)
            print('min/max of noise model are ', np.min(noise_model), np.max(noise_model))
            cbps.load_flight_image(ifield=ifield, inst=inst, inplace=True)
            observed_im = cbps.image*cbps.cal_facs[inst]
            plot_map(observed_im*mask, title='observed im elat 30 * mask')
            
            shot_sigma_sb_map = cbps.compute_shot_sigma_map(inst, image=observed_im, nfr=nframe)
            dsnmaps = shot_sigma_sb_map*np.sqrt(2)*np.random.normal(0, 1, size=imarray_shape)

            plot_map(shot_sigma_sb_map, title='shot sigma map, ifield'+str(ifield))
            plot_map(dsnmaps[0]*mask, title='dsnmaps * mask, ifield '+str(ifield))

            fourier_weights_nofluc, mean_cl2d = cbps.estimate_noise_power_spectrum(nsims=500, n_split=10, apply_mask=True, \
                   noise_model=noise_model, read_noise=True, inst=inst, ifield=ifield, show=False, mask=mask, \
                    photon_noise=True,shot_sigma_sb=shot_sigma_sb_map, inplace=False, ff_estimate=None, \
                    field_nfr=nframe)

            cbps.compute_noise_power_spectrum(inst, noise_Cl2D=mean_cl2d, inplace=True, apply_FW=True, weights=fourier_weights)

            all_cl_sn_elat30 = []
            for i in range(n_lab_exp):
                lb, cl_diff_shotnoise, _ = get_power_spec(mask*(dsnmaps[i]-np.mean(dsnmaps[i])), weights=fourier_weights, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
                cl_diff_shotnoise /= 2.
                all_cl_sn_elat30.append(cl_diff_shotnoise)

            all_cl_sn_elat30 = np.array(all_cl_sn_elat30)
            cl_diffs_shotnoise_elat30 = np.mean(all_cl_sn_elat30, axis=0)

            if verbose:
                print(' ifield 5 noise power spectrum is ', cbps.N_ell)
                print(' ifield 5 shot noise power is ', cl_diffs_shotnoise_elat30)
                
            cl_noise_elat30 = (cbps.N_ell).copy()

        else:
            
            imarray_shape = (n_lab_exp, cbps.dimx, cbps.dimy)
            flight_data_path = 'data/fluctuation_data/TM'+str(inst)+'/short_'+str(nframe)+'frames/field'+str(ifield)

            pathA = flight_data_path+'/flight1.FITS'
            pathB = flight_data_path+'/flight2.FITS'
            flight_exp_diff, meanA, meanB, flightmean = compute_exp_difference(inst, cal_facs=cbps.cal_facs, pathA=pathA, pathB=pathB, mask=mask, plot_diff_hist=True)
            l2d, cl2dflight = get_power_spectrum_2d(flight_exp_diff-np.mean(flight_exp_diff))
            plot_map(cl2dflight, title='flight cl2d')

            shot_sigma_map = cbps.compute_shot_sigma_map(inst, 0.5*(meanA+meanB)*np.ones_like(flight_exp_diff), nfr=nframe)
            snmaps1 = np.random.normal(0, 1, size=imarray_shape)
            snmaps2 = np.random.normal(0, 1, size=imarray_shape)
            dsnmaps = shot_sigma_map*(snmaps1-snmaps2)

            plot_map(dsnmaps[0]*mask, title='dsnmaps * mask, ifield '+str(ifield))

            # estimate Fourier weights using elat30-length lab exposure differences and mean photon noise levels from fields
            lab_diffs = np.zeros(imarray_shape)
            cl2ds_lab = np.zeros(imarray_shape)
            for labidx in range(n_lab_exp):
                str_num = '{0:03}'.format(labidx+1)

                pathA = elat30_lab_path+'/first/labMap_'+str_num+'_1.FITS'
                pathB = elat30_lab_path+'/second/labMap_'+str_num+'_2.FITS'

                lab_exp_diff, labmeanA, labmeanB, fullmean = compute_exp_difference(inst, mask=mask, cal_facs=cbps.cal_facs, pathA=pathA, pathB=pathB)
                lab_exp_diff += mask*dsnmaps[labidx]
                lab_diffs[labidx] = lab_exp_diff

                l2d, cl2dlab = get_power_spectrum_2d((lab_diffs[labidx]-np.mean(lab_diffs[labidx])))
                cl2ds_lab[labidx] = cl2dlab

            # use 2d power spectra from lab + photon noise for 4 frame exposure to get fourier weights
            stdpower = 1
            fw_std = np.std(cl2ds_lab, axis=0) # returns std within each 2d fourier mode
            fourier_weights = 1./fw_std**stdpower
            fourier_weights /= np.nanmax(fourier_weights)
            mean_cl2d = np.mean(cl2ds_lab, axis=0)

            plot_map(shot_sigma_map, title='shot sigma map, ifield'+str(ifield))

            all_cl_sn = []
            for labidx in range(n_lab_exp):
                lb, cl_diff_shotnoise, _ = get_power_spec(mask*(dsnmaps[labidx]-np.mean(dsnmaps[labidx])), weights=fourier_weights, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
                cl_diff_shotnoise /= 2.
                all_cl_sn.append(cl_diff_shotnoise)

            # reduce by factor of sqrt(2) to account for difference increasing noise by sqrt(2)
            lb, cl_diff_flight, _ = get_power_spec(flight_exp_diff-np.mean(flight_exp_diff), weights=fourier_weights, mask=mask, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
            cl_diff_flight /= 2.
            cl_diffs_flight.append(cl_diff_flight)

            all_cl_sn = np.array(all_cl_sn)
            cl_diffs_shotnoise.append(np.mean(all_cl_sn, axis=0))

    cl_noise_elat30 = np.array(cl_noise_elat30)
    cl_diffs_shotnoise_elat30 = np.array(cl_diffs_shotnoise_elat30)
    cl_diffs_flight = np.array(cl_diffs_flight)
    cl_diffs_shotnoise = np.array(cl_diffs_shotnoise)
                
    return lb, cl_noise_elat30, cl_diffs_shotnoise_elat30, cl_diffs_flight, cl_diffs_shotnoise



def extract_datetime_strings_mat(basepath, timestr, inst=1, verbose=True):
    fpath = basepath+timestr+'/TM'+str(inst)+'/'
        
    setfilelist = glob.glob(fpath+'set*')
        
    print(setfilelist)
    all_datetime_strings = []
    for s, setfpath in enumerate(setfilelist):
            
        set_number = int(setfpath[-1])

        within_set_filelist = glob.glob(setfpath+'/*.mat')
        
        for set_file in within_set_filelist:
            pt = datetime.strptime(timestr+' '+set_file[-21:-13],'%m-%d-%Y %H-%M-%S')
            
            all_datetime_strings.append(pt)
            
    if verbose:
        print('All datetime strings:')
        print(all_datetime_strings)
            
    return all_datetime_strings

def fit_meanphot_vs_varphot(meanphot, varphot, nfr=5, itersigma=4.0, niter=5):
    
    fit = fitting.LinearLSQFitter()
    # initialize the outlier removal fitter
    or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=niter, sigma=itersigma)
    # initialize a linear model
    line_init = models.Linear1D()
    # fit the data with the fitter
    sigmask, fitted_line = or_fit(line_init, meanphot, varphot)
    slope = fitted_line.slope.value
    g1_iter = get_g1_from_slope_T_N(slope, N=nfr)
    
    return fitted_line, sigmask, g1_iter

def fit_meanphot_vs_varphot_levmar(meanphot, varphot, nfr=5, itersigma=4.0, niter=5, mode='linear'):
    # does not currently work, using fit_meanphot_vs_varphot()

    if mode=='linear':
        fit = fitting.LinearLSQFitter()
    elif mode=='LevMar':
        fit = fitting.LevMarLSQFitter()

    # initialize the outlier removal fitter
    or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=niter, sigma=itersigma)
    # initialize a linear model
    line_init = models.Linear1D()
    # fit the data with the fitter
    sigmask, fitted_line = or_fit(line_init, meanphot, varphot)
    slope = fitted_line.slope.value
    g1_iter = get_g1_from_slope_T_N(slope, N=nfr)
    
    if mode=='LevMar':
        cov_diag = np.diag(or_fit.fit_info['param_cov'])
        print('cov diag:')
        print(cov_diag)
    
    return fitted_line, sigmask, g1_iter


def focus_data_means_and_vars(timestr_list, nfr=5, inst=1, ravel=True, basepath='data/Focus/slopedata/', get_inst_mask=True,\
                              sub_divide_images=False, bins=10, plot=False, chop_up_images=False, nside=5, maxframe=8):
    
    inst_mask = None
    if get_inst_mask:
        cbps = CIBER_PS_pipeline()
        cbps.load_data_products(4, inst, verbose=True) # don't care about individual field here, just need instrument mask

        inst_mask = cbps.maskInst_clean
        
        if inst==2:
            inst_mask[512:, 512:] = 0
        
        if plot:
            plt.figure()
            plt.title('Instrument mask', fontsize=14)
            plt.imshow(inst_mask, origin='lower')
            plt.colorbar()
            plt.show()
            
    all_timestr_list, all_set_numbers_list, all_vars_of_diffs, all_means_of_means = [[] for x in range(4)] # for each pair difference
    
    all_mask_idxs = []
    for t, timestr in enumerate(timestr_list):
        
        fpath = basepath+timestr+'/TM'+str(inst)+'/'
        
        setfilelist = glob.glob(fpath+'set*')
        
        print(setfilelist)
        
        for s, setfpath in enumerate(setfilelist):
            
            set_number = int(setfpath[-1])
            
            within_set_filelist = glob.glob(setfpath+'/*.mat')
            
            im_list = []
            
            for f in within_set_filelist:
                
                mat_im = load_focus_mat(f, nfr)
                if mat_im is not None:
                    im_list.append(mat_im)


            print('im list has length ', len(im_list))
            
            lightsrcmask, masked_sum_image = get_lightsrc_mask_unsharp_masking(im_list, inst_mask=inst_mask, small_scale=2, large_scale=10, nsig=3)
            
            im_list = np.array(im_list)
            im_means = [np.mean(im) for im in im_list]
            
            print('image means are ', im_means)
            print('var of means: ', np.var(im_means))
            
            if np.var(im_means) > 0.1:
                print('too large, going to next set')
                continue
            
            if ravel:
                im_list = [im.ravel() for im in im_list]
                lightsrcmask = lightsrcmask.ravel()
            
            binmasks = None
            if sub_divide_images:
                if chop_up_images:
                    submasks = chop_up_masks(masked_sum_image, nside=nside, show=True)
                else:
                    
                    
                    submasks = masks_from_sigmap(gaussian_filter(masked_sum_image, 5), bins=bins, show=True)
                if ravel:
                    binmasks = [lightsrcmask*(mask.ravel()) for mask in submasks]
                else:
                    binmasks = [lightsrcmask*mask for mask in submasks]
                    
            else:
                binmasks = [lightsrcmask]

            for im_idx, image_mask in enumerate(binmasks):
                
                pair_means_cut, pair_diffs_cut, vars_of_diffs, means_of_means = pairwise_means_variances(im_list, initial_mask=image_mask, plot=plot)

                if np.sum(image_mask) > 1000:
                
                    all_set_numbers_list.extend([set_number for x in range(len(vars_of_diffs))])            
                    all_timestr_list.extend([timestr for x in range(len(vars_of_diffs))])
                    all_vars_of_diffs.extend(vars_of_diffs)
                    all_means_of_means.extend(means_of_means)
                    
                    all_mask_idxs.extend([im_idx for i in range(len(means_of_means))])

    return all_set_numbers_list, all_timestr_list, all_vars_of_diffs, all_means_of_means, all_mask_idxs, binmasks


def get_g1_from_slope_T_N(slope, N=5, T_frame=1.78):
    # follows Garnett and Forest (1993?) equation for photon noise associated with signal
    
    return (slope*T_frame*N)**(-1.)*(6./5.)*(N**2+1.)/(N**2-1.)

def get_lightsrc_mask_unsharp_masking(im_list, inst_mask=None, small_scale=5, large_scale=10, nsig=4, plot=True):
    
    sum_im = np.mean(np.array(im_list), axis=0)
    
    print('sum im has shape ', sum_im.shape, 'inst mask has shape ', inst_mask.shape)
    if inst_mask is not None:
        sum_im *= inst_mask
        
    brightpixmask = sigma_clip_maskonly(sum_im, sig=4)
    plt.figure(figsize=(8,8))
    plt.title('orig image')
    plt.imshow(sum_im, vmin=np.nanpercentile(sum_im, 5), vmax=np.nanpercentile(sum_im, 95))
    plt.colorbar()
    plt.show()
    sum_im *= brightpixmask
    
    small_smooth_sum_im = gaussian_filter(sum_im, small_scale)
    large_smooth_sum_im = gaussian_filter(sum_im, large_scale)

    small_over_large = small_smooth_sum_im/large_smooth_sum_im

    
    if plot:
        
        plt.figure()
        hist_small_over_large = small_over_large.ravel()
        nanmask = (~np.isnan(hist_small_over_large))*(~np.isinf(hist_small_over_large))
        plt.hist(hist_small_over_large[nanmask], bins=20, histtype='step')
        plt.axvline(np.median(hist_small_over_large[nanmask]), linestyle='dashed', color='r')
        plt.axvline(np.percentile(hist_small_over_large[nanmask], 84), linestyle='dashed', color='b')
        plt.axvline(np.percentile(hist_small_over_large[nanmask], 16), linestyle='dashed', color='b')
        plt.axvline(np.percentile(hist_small_over_large[nanmask], 5), linestyle='dashed', color='g')
        plt.axvline(np.percentile(hist_small_over_large[nanmask], 95), linestyle='dashed', color='g')
        plt.legend()
        plt.yscale('log')
        plt.show()
        
    small_over_large[np.isnan(small_over_large)] = 0.

    mask = (np.abs(small_over_large-np.nanmedian(small_over_large)) < nsig*np.nanstd(small_over_large[small_over_large != 0])).astype(np.float)

    if inst_mask is not None:
        mask *= inst_mask
        
    mask *= brightpixmask
        
    if plot:
        
        plt.figure(figsize=(15, 8))
        plt.subplot(1,2,1)
        plt.title('Image ($\\sigma_{smooth}=2$) / Image ($\\sigma_{smooth}=10$)', fontsize=18)

        plt.imshow(small_over_large, origin='lower', vmin=0.5, vmax=2.)

        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        plt.xlabel('x [pix]', fontsize=16)
        plt.ylabel('y [pix]', fontsize=16)
        
        plt.subplot(1,2,2)
        plt.title('Masked sum image', fontsize=18)
        plt.imshow(mask*sum_im, origin='lower', vmin=np.nanpercentile(mask*sum_im, 5), vmax=np.nanpercentile(mask*sum_im, 70))

        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        plt.xlabel('x [pix]', fontsize=16)
        plt.ylabel('y [pix]', fontsize=16)
        
        plt.tight_layout()
        plt.show()
         
    return mask, sum_im*mask


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

def labexp_to_noisemodl(cbps, inst, ifield, labexp_fpaths, labexp_fpaths_2=None,\
                        clip_sigma = 4, sigma_clip_cl2d=False, cl2d_clip_sigma=4, cal_facs=None, nitermax=10,\
                        plot=True, base_fluc_path='/Volumes/Seagate Backup Plus Drive/Toolkit/Mirror/Richard/ciber_fluctuation_data/'):
    
    
    ''' estimates a read noise model from a collection of dark exposure pair differences. '''


    nexp = len(labexp_fpaths)
    
    if labexp_fpaths_2 is not None:
        nexp += len(labexp_fpaths_2)
    
    print('number of pairs : ', nexp//2)
    
    inverse_full_mkks = np.zeros((nexp//2, cbps.n_ps_bin, cbps.n_ps_bin))
    mameans_full = np.zeros((nexp//2,))
    cls_2d_noise_full = np.zeros((nexp//2, cbps.dimx, cbps.dimy))
    
    meanAs, meanBs, mask_fracs = [], [], []
        
    for i in range(nexp//2):
        mask_inst = cbps.load_mask(ifield=ifield, inst=inst, masktype='maskInst_clean', inplace=False)
        
        
        if labexp_fpaths_2 is not None:
            lab_exp_diff, meanA, meanB, mean_exp = compute_exp_difference(inst, cal_facs=cal_facs, pathA=labexp_fpaths[i], pathB=labexp_fpaths_2[i], mask=mask_inst, plot_diff_hist=False)
        else:
            lab_exp_diff, meanA, meanB, mean_exp = compute_exp_difference(inst, cal_facs=cal_facs, pathA=labexp_fpaths[2*i], pathB=labexp_fpaths[2*i+1], mask=mask_inst, plot_diff_hist=False)
        
        
        if plot:
            plot_map(lab_exp_diff*mask_inst, title='lab exp diff')
        
        lab_exp_diff /= np.sqrt(2)
        new_mask = iter_sigma_clip_mask(lab_exp_diff, sig=clip_sigma, nitermax=nitermax, initial_mask=mask_inst.astype(np.int))
        mask_inst *= new_mask
        
        if plot:
            plot_map(lab_exp_diff*mask_inst, title='lab exp diff sigma clipped')
        
        meanAs.append(meanA)
        meanBs.append(meanB)

        masked_full = np.ma.array(lab_exp_diff, mask=~mask_inst.astype(np.bool))
        mamean = np.ma.mean(masked_full)
        mameans_full[i] = mamean
        
        fdet = float(np.sum(mask_inst))/float(cbps.dimx*cbps.dimy)
        mask_fracs.append(fdet)

        # get 2d power spectrum
        l2d, cl2d = get_power_spectrum_2d(mask_inst*(lab_exp_diff-mamean), pixsize=cbps.Mkk_obj.pixsize)
        
        if sigma_clip_cl2d:
            
            # we are only going to try to mask the spurious modes at higher ell_x and ell_y, so we ignore the central band
            ybandtop = 512+100
            ybandbottom = 512-100

            # make average 2D Cl excluding horizontal band
            noband_noise = np.zeros_like(cl2d)
            noband_noise[ybandtop:,:] = cl2d[ybandtop:,:]
            noband_noise[:ybandbottom,:] = cl2d[:ybandbottom,:]
            
            itermask = iter_sigma_clip_mask(noband_noise, sig=cl2d_clip_sigma, nitermax=10, initial_mask=np.ones_like(noband_noise.astype(np.int)))

            fourier_mask_full = itermask.copy()
            fourier_mask_full[ybandbottom:ybandtop,:] = 1.
            
            if plot:
                plot_map(fourier_mask_full, title='fourier mask')
                plot_map(fourier_mask_full*cl2d, title='cl2d * fourier mask')

            cl2d *= fourier_mask_full
            
        cls_2d_noise_full[i] = cl2d
        
        
    av_cl2d = np.mean(cls_2d_noise_full, axis=0)
    
    # correct 2d power spectrum for mask with 1/fsky
    av_mask_frac = np.mean(mask_fracs)
    av_cl2d /= av_mask_frac
    
    av_means = 0.5*(np.mean(meanAs)+np.mean(meanBs))
    
    if plot:
        plot_map(av_cl2d, title='av cl2d')
        
    return av_cl2d, av_means


def load_focus_mat(filename, nfr):
    x = loadmat(filename, struct_as_record=True)
    nfr_arr = x['framedat'][0,0]['nfr_arr'][0]
    if nfr in nfr_arr:
        nfr_idx = list(nfr_arr).index(nfr)

        if np.max(nfr_arr) < maxframe:
            return x['framedat'][0,0]['map_arr'][nfr_idx]
        else:
            return None
        

def masks_from_sigmap(sigmap, bins=10, ravel=False, show=False):
    
    sigmap[sigmap==0] = np.nan
    
    if type(bins)==int:
        bins = np.linspace(np.nanpercentile(sigmap, 1), np.nanpercentile(sigmap, 99), bins+1)
    
    print('bins are ', bins)
    
    if ravel:
        masks = np.zeros((len(bins), sigmap.shape[0]*sigmap.shape[1]))
    else:
        masks = np.zeros((len(bins), sigmap.shape[0], sigmap.shape[1]))
            
    for b in range(len(bins)-1):
        binmask = (sigmap > bins[b])*(sigmap < bins[b+1])
        
        if show:
            plt.figure(figsize=(8,8))
            plt.title(str(np.round(bins[b], 2))+' < val < '+str(np.round(bins[b+1], 2)), fontsize=18)
            plt.imshow(sigmap*np.array(binmask).astype(np.int), origin='lower', vmin=bins[b], vmax=bins[b+1])
            plt.colorbar()
            plt.show()
        
        if ravel:
            masks[b,:] = binmask.ravel()
        else:
            masks[b,:,:] = binmask
            
    return masks

def pairwise_means_variances(im_list, initial_mask=None, plot=False, imdim=1024, savedir=None, verbose=True, inst=1, show_diffs=False):
    
    
    ''' I am going to assume that the im_list is a list of 1d arrays, this makes some of the code much easier'''
    
    im_len = len(im_list)
    print('image list length is ', im_len)
    idxstart = 0
    if im_len%2 != 0:
        idxstart = 1
        
    diffidx=0
    pair_means_cut, pair_diffs_cut, sumdiffs = [], [], []

    if initial_mask is not None:
        pair_mask = initial_mask.copy()
    else:
        pair_mask = np.ones_like(im_list[0]).astype(np.int)
    
    sigclipmasks = []
    
    for idx in np.arange(idxstart, im_len, 2):

        pair_mean = 0.5*(im_list[idx]+im_list[idx+1])
        pair_diff = im_list[idx]-im_list[idx+1]
        
        pair_diff[pair_mask==0] = np.nan
        pair_mean[pair_mask==0] = np.nan

        if plot:            
            f = plot_exposure_pair_diffs_means(pair_diff, pair_mean, pair_mask, diffidx, show=True)
            if savedir is not None:
                f.savefig(savedir+'/pair_diff_'+str(idx)+'.png', bbox_inches='tight', dpi=150)
        
        sigclipmask = sigma_clip_maskonly(pair_diff, previous_mask=pair_mask, sig=4)
        sigclipmasks.append(sigclipmask.astype(np.int))
        
        pair_means_cut.append(pair_mean)
        pair_diffs_cut.append(pair_diff)
        sumdiffs.append(np.nansum(np.abs(pair_diff)))
    
        diffidx += 1

    vars_of_diffs = np.array([np.nanvar(pair_diff_cut[sigclipmasks[i]==1]) for i, pair_diff_cut in enumerate(pair_diffs_cut)])
    means_of_means = np.array([np.nanmean(pair_mean_cut[sigclipmasks[i]==1]) for i, pair_mean_cut in enumerate(pair_means_cut)])
    
    print('len of pairdiffscut', len(pair_diffs_cut))
    
    if inst==2:
        sumdiffs = np.array(sumdiffs)

        median_sumdiff = np.median(sumdiffs)
        min_sumdiff = np.min(sumdiffs)
        sumdiff_mask = (sumdiffs < 1.5*min_sumdiff)

        plt.figure()
        plt.title('sum diffs')
        plt.hist(sumdiffs, bins=10)
        plt.show()

        print('sumdiff mask is ', sumdiff_mask)

        vars_of_diffs = vars_of_diffs[sumdiff_mask]
        means_of_means = means_of_means[sumdiff_mask]
        pair_diffs_cut = [pair_diffs_cut[p] for p in range(len(pair_diffs_cut)) if sumdiff_mask[p]]
        pair_means_cut = [pair_diffs_cut[p] for p in range(len(pair_diffs_cut)) if sumdiff_mask[p]]

        sumdiffs = sumdiffs[sumdiff_mask]

        plt.figure()
        plt.title('sum diffs after masking')
        plt.hist(sumdiffs, bins=10)
        plt.show()
    
    if show_diffs:
        for p, pair_diff in enumerate(pair_diffs_cut):

            pair_d = np.reshape(pair_diff, (1024, 1024))
            pair_diff = np.array(pair_diff)[np.nonzero(pair_diff)]
            print('variance is '+str(np.round(vars_of_diffs[p], 2))+'sum diff is '+str(int(np.nansum(np.abs(pair_diff)))))

            plt.figure(figsize=(10, 4))
            plt.subplot(1,2,1)
            plt.title('$\\langle \\sigma^2 \\rangle$ = '+str(np.round(vars_of_diffs[p], 2)), fontsize=18)
            plt.imshow(pair_d, vmin=np.percentile(pair_d, 5), vmax=np.percentile(pair_d, 95))
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.xlabel('x [pix]', fontsize=16)
            plt.ylabel('y [pix]', fontsize=16)
            plt.tick_params(labelsize=14)
            
            plt.subplot(1,2,2)
            plt.hist(pair_diff, bins=np.linspace(-100, 100, 30))
            plt.yscale('log')
            plt.tight_layout()
            plt.xlabel('Difference value [ADU/fr]', fontsize=16)
            plt.ylabel('$N_{pix}$', fontsize=16)
            plt.tick_params(labelsize=14)
            plt.show()
    
    if verbose:
        print('vars of diffs:', vars_of_diffs)
        print('means of means:', means_of_means)

    return pair_means_cut, pair_diffs_cut, vars_of_diffs, means_of_means



def plot_means_vs_vars(m_o_m, v_o_d, timestrs, timestr_cut=None, var_frac_thresh=None, xlim=None, ylim=None, all_set_numbers_list=None, all_timestr_list=None,\
                      nfr=5, inst=1, fit_line=True, itersigma=4.0, niter=5, imdim=1024, figure_size=(12,6), markersize=100, titlestr=None, mode='linear', jackknife_g1=False, split_boot=10, boot_g1=True, n_boot=100):
    
    mask = [True for x in range(len(m_o_m))]
    
    if timestr_cut is not None:
        mask *= np.array([t==timestr_cut for t in all_timestr_list])
    
    photocurrent = np.array(m_o_m[mask])
    varcurrent = np.array(v_o_d[mask])

    if fit_line:
        if boot_g1:

            boot_g1s = np.zeros((n_boot,))
            for i in range(n_boot):
                randidxs = np.random.choice(np.arange(len(photocurrent)), len(photocurrent)//split_boot)
                phot = photocurrent[randidxs]
                varc = varcurrent[randidxs]

                _, _, g1_boot = fit_meanphot_vs_varphot(phot, 0.5*varc, nfr=nfr, itersigma=itersigma, niter=niter)
                boot_g1s[i] = g1_boot

            print('bootstrap sigma(G1) = '+str(np.std(boot_g1s)))
            print('while bootstrap mean is '+str(np.mean(boot_g1s)))
                
        fitted_line, sigmask, g1_iter = fit_meanphot_vs_varphot(photocurrent, 0.5*varcurrent, nfr=nfr, itersigma=itersigma, niter=niter)

    else:
        g1_iter = None
        
    
    if var_frac_thresh is not None:
        median_var = np.median(varcurrent)
        mask *= (np.abs(v_o_d-median_var) < var_frac_thresh*median_var) 
                
    min_x_val, max_x_val = np.min(photocurrent), np.max(photocurrent)
            
    if all_set_numbers_list is not None:
        colors = np.array(all_set_numbers_list)[mask]
    else:
        colors = np.arange(len(m_o_m))[mask]
        
    f = plt.figure(figsize=figure_size)
    title = 'TM'+str(inst)
    if timestr_cut is not None:
        title += ', '+timestr_cut
    if titlestr is not None:
        title += ' '+titlestr
                
    plt.title(title, fontsize=18)
        
    markers = ['o', 'x', '*', '^', '+']
    set_color_idxs = []
    
    
    for t, target_timestr in enumerate(timestrs):
        tstrmask = np.array([tstr==target_timestr for tstr in all_timestr_list])
        
        tstr_colors = None
        if all_set_numbers_list is not None:
            tstr_colors = np.array(all_set_numbers_list)[mask*tstrmask]
        
        if fit_line:
            if t==0:
                if inst==1:
                    xvals = np.linspace(-1, -18, 100)
                else:
                    xvals = np.linspace(np.min(photocurrent), np.max(photocurrent), 100)

                plt.plot(xvals, fitted_line(xvals), label='$G_1$='+str(np.round(np.mean(boot_g1s), 3))+'$\\pm$'+str(np.round(np.std(boot_g1s), 3)))
            plt.scatter(photocurrent[tstrmask], sigmask[tstrmask], s=markersize, marker=markers[t], c=tstr_colors, label=target_timestr)
            if ylim is not None:
                plt.ylim(ylim)
            if xlim is not None:
                plt.xlim(xlim)
        else:
            plt.scatter(photocurrent[tstrmask], 0.5*varcurrent[tstrmask], s=markersize, marker=markers[t], c=tstr_colors, label=target_timestr)
            plt.xlim(xlim)
            plt.ylim(ylim)            

    plt.legend(fontsize=16)
    plt.xlabel('mean [ADU/fr]', fontsize=18)
    plt.ylabel('$\\sigma^2$ [(ADU/fr)$^2$]', fontsize=18)
    plt.tick_params(labelsize=14)
    plt.show()
    
    return f, g1_iter

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

def slice_by_timestr(all_means_of_means, all_vars_of_diffs, all_timestr_list, timestrs, all_set_nos=None, mask_nos=None, all_mask_nos=None, set_bins=None, photocurrent_bins=None, \
                    itersigma=4.0, niter=5, inst=1, minpts=10, nfr=5, jackknife_g1=False, boot_g1=True, n_boot=100):

    listo_param_masks = []
    listo_g1s = []
    listo_tstrs, listo_sets = [], []
    listo_figs = []
    listo_masknos = []
    
    print('timestrs are ', timestrs)
    if set_bins is not None:
        print('set bins are ', set_bins)
    if mask_nos is not None:
        print('mask_nos are ', mask_nos)
    if all_mask_nos is not None:
        print('while all_mask_nos is ', all_mask_nos)

        
    for t, tstr in enumerate(timestrs):
        tstr_mask = (all_timestr_list==tstr)
        
        if set_bins is not None and all_set_nos is not None:
            for s, set_no in enumerate(set_bins):
                set_no_mask = (all_set_nos==set_no)
                
                if all_mask_nos is not None:
                    for m, maskno in enumerate(mask_nos):
                        maskno_mask = (all_mask_nos==maskno)
                        listo_param_masks.append(set_no_mask*tstr_mask*maskno_mask)
                        listo_tstrs, listo_sets, _, listo_masknos = update_lists(listo_tstrs=listo_tstrs, listo_sets=listo_sets, listo_masknos=listo_masknos, tstr=tstr, set_no=set_no, maskno=maskno)
                
                else:
                    listo_param_masks.append(set_no_mask*tstr_mask)
                    listo_tstrs, listo_sets, _, _ = update_lists(listo_tstrs=listo_tstrs, listo_sets=listo_sets, tstr=tstr, set_no=set_no)

        else:   
            if all_mask_nos is not None:
                for m, maskno in enumerate(mask_nos):
                    maskno_mask = (all_mask_nos==maskno)
                    listo_param_masks.append(tstr_mask*maskno_mask)
                    listo_tstrs, _, _, listo_masknos = update_lists(listo_tstrs=listo_tstrs, listo_masknos=listo_masknos, tstr=tstr, maskno=maskno)

            else:
                listo_param_masks.append(tstr_mask)
                listo_tstrs, _, _, _ = update_lists(listo_tstrs=listo_tstrs, tstr=tstr)

                
    final_param_masks = []
    for p, param_mask in enumerate(listo_param_masks):
        if np.sum(param_mask) > minpts:


            f_tm, g1_iter = plot_means_vs_vars(all_means_of_means[param_mask], all_vars_of_diffs[param_mask],\
                                                 [listo_tstrs[p]], itersigma=itersigma, niter=niter, inst=inst,\
                                                 fit_line=True, all_set_numbers_list=all_mask_nos[param_mask],\
                                                 all_timestr_list=all_timestr_list[param_mask], figure_size=(6, 6), markersize=20, \
                                              nfr=nfr, jackknife_g1=jackknife_g1, boot_g1=boot_g1, n_boot=n_boot)
            listo_g1s.append(g1_iter)
            final_param_masks.append(param_mask)
            listo_figs.append(f_tm)

        else:
            print('nope :(')
    
    return np.array(listo_g1s), final_param_masks, listo_figs

def update_lists(listo_tstrs=None, listo_sets=None, listo_photocurrents=None, listo_masknos=None, tstr=None, set_no=None, maskno=None, phot=None):
    if listo_tstrs is not None:
        listo_tstrs.append(tstr)
    if listo_sets is not None:
        listo_sets.append(set_no)
    if listo_photocurrents is not None:
        listo_photocurrents.append(phot)
    if listo_masknos is not None:
        listo_masknos.append(maskno)
    
    return listo_tstrs, listo_sets, listo_photocurrents, listo_masknos



def validate_noise_model_flight_diff(cbps, inst, flight_halves=None, ifield_list=[4, 5, 6, 7, 8], noise_models=None, flight_spatial_shot=False, flight_mask=None, flight_data_base_path=None, noise_model_fpaths=None,\
                                    nsims=20, n_split=1, nshot=20, mkk_correct=False, n_mkk_sim=20, use_lab_phot=False, verbose=True, plot_comparison=True):
    
    ''' 
    noise model validation. This involves:
    - generating a number of read noise realizations from the model and computing their difference power spectra
    - computing the means of the half exposures and computing their difference power spectra
    - adding photon noise from flight to read noise realizations
    - plot them, do they agree?


    Parameters
    ----------

    cbps : "CIBER_PS_PIPELINE" class object.
    inst :
    flight_halves (optional): 
    ifield_list (optional): 
    noise_models (optional): 
    flight_mask (optional): 
    flight_data_base_path (optional): 
    noise_model_fpaths (optional): 
    nsims (optional): 
    nshot (optional): 
    mkk_correct (optional): 
    use_lab_phot (optional): 

    Returns
    -------
    
    '''
    
    if flight_data_base_path is None:
        flight_data_base_path = '/Volumes/Seagate Backup Plus Drive/Toolkit/Mirror/Richard/noise_model_validation_data/TM'+str(inst)+'/validationHalfExp/'
    
    ps_set_shape = (len(ifield_list), nsims, cbps.n_ps_bin)
    maplist_split_shape = (nsims//n_split, cbps.dimx, cbps.dimy)

    flight_diffs = []

    print('flight data base path is ', flight_data_base_path)
    N_ells, N_ells_phot, N_ells_flight, N_ell_errs_flight = [np.zeros(ps_set_shape) for x in range(4)]
    N_ells_flight = np.zeros((len(ifield_list), cbps.n_ps_bin))
    empty_aligned_objs, fft_objs = construct_pyfftw_objs(2, maplist_split_shape)
    
    for i, ifield in enumerate(ifield_list):
        
        nfr = cbps.field_nfrs[ifield]//2
        base_fluc_path = '/Volumes/Seagate Backup Plus Drive/Toolkit/Mirror/Richard/ciber_fluctuation_data/'

        mask_fpath = base_fluc_path+'/TM'+str(inst)+'/masks/joint_mask_with_ffmask_ifield'+str(ifield)+'_inst'+str(inst)+'.fits'
        mask = fits.open(mask_fpath)['joint_mask_'+str(ifield)].data

        if noise_models is None:
            if noise_model_fpaths is not None:
                print("loading noise model from ", noise_model_fpaths[i])
                noise_model = fits.open(noise_model_fpaths[i])['noise_model_'+str(ifield)].data
                plot_map(noise_model, title='noise model at top')
            else:
                print('Function requires either noise model or path to noise model. Exiting')
                return
        else:
            noise_model=noise_models[i]

        flightA_path = flight_data_base_path+'field'+str(ifield)+'/flightMap_1.FITS'
        flightB_path = flight_data_base_path+'field'+str(ifield)+'/flightMap_2.FITS'
        flight_exp_diff, meanA, meanB, mean_flight = compute_exp_difference(inst, cal_facs=cbps.cal_facs, pathA=flightA_path, pathB=flightB_path, mask=mask, plot_diff_hist=True)
        plot_map(flight_exp_diff, title='flight exp diff')
        
        if flight_spatial_shot:
            shot_sigma_sb = cbps.compute_shot_sigma_map(inst, mean_flight, nfr=nfr)
        else:
            shot_sigma_sb = cbps.compute_shot_sigma_map(inst, 0.5*(meanA+meanB)*np.ones_like(flight_exp_diff), nfr=nfr)
        # compute noise model fourier weights
        noise_model = fits.open(noise_model_fpaths[i])['noise_model_'+str(ifield)].data

        if use_lab_phot:
            photon_eps = fits.open(noise_model_fpaths[i])[0].header['photon_eps']
            photon_sb = photon_eps*cbps.g2_facs[inst]
            plot_map(noise_model, title='noise model')
                                                                
        meanAB = 0.5*(meanA+meanB)
        
        if use_lab_phot:
            meanAB -= photon_sb
            
        shot_sigma_sb = cbps.compute_shot_sigma_map(inst, meanAB*np.ones_like(flight_exp_diff), nfr=nfr)
        
        fourier_weights, mean_cl2d = cbps.estimate_noise_power_spectrum(ifield=ifield, nsims=nsims, n_split=n_split, apply_mask=True, \
                   mask=mask, noise_model=noise_model, inst=inst, show=False,\
                    read_noise=True, photon_noise=False, shot_sigma_sb=shot_sigma_sb, inplace=False, \
                    ff_estimate=None, field_nfr=nfr, verbose=verbose)
        
        for j in range(n_split):

            rnmaps, _ = cbps.noise_model_realization(inst, maplist_split_shape, noise_model, fft_obj=fft_objs[0],\
                                          read_noise=True, photon_noise=False)
        
            snmaps1 = np.random.normal(0, 1, size=(nsims//n_split, cbps.dimx, cbps.dimy))
            dsnmaps = shot_sigma_sb*snmaps1
        
            for n in np.arange(nsims//n_split):
                rnmaps[n] += dsnmaps[n]
                
                lb, N_ell_sim, _ = get_power_spec(mask*(rnmaps[n]-np.mean(rnmaps[n])), weights=fourier_weights, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
                lb, N_ell_sim_phot, _ = get_power_spec(mask*(dsnmaps[n]-np.mean(dsnmaps[n])), weights=fourier_weights, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)

                N_ells[i, j*nsims//n_split + n,:] = N_ell_sim                
                N_ells_phot[i, j*nsims//n_split + n,:] = N_ell_sim_phot
        
        print('np.mean of flight exp diff is ', np.mean(flight_exp_diff))
    
        lb, N_ell_flight, N_ell_flight_err = get_power_spec(mask*(flight_exp_diff-np.mean(flight_exp_diff)), mask=mask, weights=fourier_weights, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
        N_ell_flight /= 2. 
        N_ells_flight[i] = N_ell_flight
        N_ell_errs_flight[i] = N_ell_flight_err
        
        flight_diffs.append(flight_exp_diff)
        
        if plot_comparison:
            print(N_ells.shape, N_ells[i].shape)
            print(np.mean(N_ells[i], axis=0).shape)
            plot_noise_modl_val_flightdiff(lb, N_ells[i], N_ells_flight[i], ifield=ifield, inst=inst, show=True, return_fig=False)
            
    return lb, N_ells, N_ells_flight, N_ells_phot, N_ell_errs_flight, flight_diffs


def write_noise_model_fits(noise_model, ifield, inst, pixsize=7., photon_eps=None, \
                  dat_type=None):

    ''' writes noise model to FITS file with some metadata. '''

    hduim = fits.ImageHDU(noise_model, name='noise_model_'+str(ifield))        

    hdup = fits.PrimaryHDU()
    hdup.header['ifield'] = ifield
    hdup.header['inst'] = inst
    hdup.header['pixsize'] = pixsize
    if photon_eps is not None:
        hdup.header['photon_eps'] = photon_eps
    if dat_type is not None:
        hdup.header['dat_type'] = dat_type        

    hdul = fits.HDUList([hdup, hduim])
    return hdul




