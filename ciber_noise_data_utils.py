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
                x = loadmat(f, struct_as_record=True)
                
                nfr_arr = x['framedat'][0,0]['nfr_arr'][0]
                if nfr in nfr_arr:
                    nfr_idx = list(nfr_arr).index(nfr)
                    
                    if np.max(nfr_arr) < maxframe: # simple check for dark exposures among shorter exposures
                    
                        im_list.append(x['framedat'][0,0]['map_arr'][nfr_idx])

            print('im list has length ', len(im_list))
            
            lightsrcmask, masked_sum_image = get_lightsrc_mask_unsharp_masking(im_list, inst_mask=inst_mask, small_scale=2, large_scale=10, nsig=3)
            
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
                      nfr=5, inst=1, fit_line=True, itersigma=4.0, niter=5, imdim=1024, figure_size=(12,6), markersize=100, titlestr=None, mode='linear', jackknife_g1=False):
    
    mask = [True for x in range(len(m_o_m))]
    
    if timestr_cut is not None:
        mask *= np.array([t==timestr_cut for t in all_timestr_list])
    
    photocurrent = np.array(m_o_m[mask])
    varcurrent = np.array(v_o_d[mask])

    if fit_line:
        
        if jackknife_g1:
            jack_g1s = np.zeros_like(photocurrent)

            for i in range(len(photocurrent)):
                phot = np.concatenate((photocurrent[:i],photocurrent[(i+1):]))
                varc = np.concatenate((varcurrent[:i],varcurrent[(i+1):]))

                _, _, g1_jack = fit_meanphot_vs_varphot(phot, 0.5*varc, nfr=nfr, itersigma=itersigma, niter=niter)
                jack_g1s[i] = g1_jack
                
            print('sigma(G1) = '+str(np.std(jack_g1s)))
        fitted_line, sigmask, g1_iter = fit_meanphot_vs_varphot(photocurrent, 0.5*varcurrent, nfr=nfr, itersigma=itersigma, niter=niter)

    else:
        g1_iter = None
        
    
    if var_frac_thresh is not None:
        median_var = np.median(varcurrent)
        print('median variance is ', median_var)
        mask *= (np.abs(v_o_d-median_var) < var_frac_thresh*median_var) 
        
        print(np.sum(mask), len(v_o_d))
        
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

                plt.plot(xvals, fitted_line(xvals), label='$G_1$='+str(np.round(g1_iter, 3)))
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
                    itersigma=4.0, niter=5, inst=1, minpts=10, nfr=5, jackknife_g1=True):

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


            f_tm, g1_iter = plot_means_vs_vars2(all_means_of_means[param_mask], all_vars_of_diffs[param_mask],\
                                                 [listo_tstrs[p]], itersigma=itersigma, niter=niter, inst=inst,\
                                                 fit_line=True, all_set_numbers_list=all_mask_nos[param_mask],\
                                                 all_timestr_list=all_timestr_list[param_mask], figure_size=(6, 6), markersize=20, \
                                              nfr=nfr, jackknife_g1=jackknife_g1)
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



