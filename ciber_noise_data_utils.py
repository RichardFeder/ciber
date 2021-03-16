import numpy as np
import astropy.io.fits as fits
import astropy.wcs as wcs
import pandas as pd
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
from astropy.table import Table
import glob
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

def focus_data_means_and_vars(timestr_list, nfr=5, inst=1, ravel=True, basepath='data/Focus/slopedata/', get_inst_mask=True,\
                              sub_divide_images=False, bins=10, plot=False):
    
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
                
                if x['framedat'][0,0]['nfr_arr'][0][-1]==nfr:
                    im_list.append(x['framedat'][0,0]['map_arr'][-1])
                    
            print('im list has length ', len(im_list))
            
            lightsrcmask, masked_sum_image = get_lightsrc_mask_unsharp_masking(im_list, inst_mask=inst_mask, small_scale=2, large_scale=10, nsig=3)
            
            if ravel:
                im_list = [im.ravel() for im in im_list]
                lightsrcmask = lightsrcmask.ravel()
            
            binmasks = None
            if sub_divide_images:
                submasks = masks_from_sigmap(gaussian_filter(masked_sum_image, 5), bins=bins, show=True)
                if ravel:
                    binmasks = [lightsrcmask*(mask.ravel()) for mask in submasks]
                else:
                    binmasks = [lightsrcmask*mask for mask in submasks]
                    
            else:
                binmasks = [lightsrcmask]

            for image_mask in binmasks:
                
                print('im list and image mask have shapes', im_list[0].shape, image_mask.shape)
                pair_means_cut, pair_diffs_cut, vars_of_diffs, means_of_means = pairwise_means_variances(im_list, initial_mask=image_mask, plot=plot)

                if np.sum(image_mask) > 1000:
                
                    all_set_numbers_list.extend([set_number for x in range(len(vars_of_diffs))])            
                    all_timestr_list.extend([timestr for x in range(len(vars_of_diffs))])
                    all_vars_of_diffs.extend(vars_of_diffs)
                    all_means_of_means.extend(means_of_means)

            
    return all_set_numbers_list, all_timestr_list, all_vars_of_diffs, all_means_of_means


def get_g1_from_slope_T_N(slope, N=5, T_frame=1.78):
    
    return (slope*T_frame*N)**(-1.)*(6./5.)*(N**2+1.)/(N**2-1.)

def get_lightsrc_mask_unsharp_masking(im_list, inst_mask=None, small_scale=5, large_scale=10, nsig=4, plot=True):
    
    sum_im = np.mean(np.array(im_list), axis=0)
    
    print('sum im has shape ', sum_im.shape, 'inst mask has shape ', inst_mask.shape)
    if inst_mask is not None:
        sum_im *= inst_mask
        
    brightpixmask = sigma_clip_maskonly(sum_im, sig=4)
    
    sum_im *= brightpixmask
    plt.figure(figsize=(8,8))
    plt.imshow(sum_im*brightpixmask)
    plt.colorbar()
    plt.show()
    
    small_smooth_sum_im = gaussian_filter(sum_im, small_scale)
    large_smooth_sum_im = gaussian_filter(sum_im, large_scale)

    small_over_large = small_smooth_sum_im/large_smooth_sum_im
    large_over_small = large_smooth_sum_im/small_smooth_sum_im

    large_over_small[np.abs(large_over_small) > 50] = 0.
    
    if plot:
        
        plt.figure(figsize=(10,10))
        plt.title('small over large')
        plt.imshow(small_over_large, origin='lower')
        plt.colorbar()
        plt.show()
        
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
        
    if plot:
        plt.figure(figsize=(8,8))
        plt.title("light source mask")
        plt.imshow(sum_im*mask, origin='lower')
        plt.colorbar()
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

def pairwise_means_variances(im_list, initial_mask=None, plot=False, imdim=1024, savedir=None):
    
    
    ''' I am going to assume that the im_list is a list of 1d arrays, this makes some of the code much easier'''
    
    im_len = len(im_list)
    print('image list length is ', im_len)
    idxstart = 0
    if im_len%2 != 0:
        idxstart = 1
        
    diffidx=0
    pair_means_cut, pair_diffs_cut = [], []

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
        
        sigclipmask = sigma_clip_maskonly(pair_diff, previous_mask=pair_mask, sig=4,show=False)
        sigclipmasks.append(sigclipmask.astype(np.int))
        
        pair_means_cut.append(pair_mean)
        pair_diffs_cut.append(pair_diff)
        
#         print('pair mean, pair diff, sigclipmask have shapes', pair_mean.shape, pair_diff.shape, sigclipmask.shape)
#         print('sum of sigclipmask is ', np.sum(sigclipmask))
#         print(len(pair_diff[~np.isnan(pair_diff)]), len(pair_diff[(~np.isnan(pair_diff))*sigclipmasks[diffidx]]))
#         print(pair_diff[(~np.isnan(pair_diff))*sigclipmasks[diffidx]])
#         plt.figure()
#         plt.subplot(1,2,1)
#         plt.hist(pair_diff[~np.isnan(pair_diff)], bins=30)
#         plt.yscale('log')
#         plt.subplot(1,2,2)
#         plt.hist(pair_diff[sigclipmasks[diffidx]==1], bins=30)
#         plt.yscale('log')
#         plt.show()
        
#         print('number of pixels in estimate is ', np.sum(pair_mask*sigclipmask))
    

        diffidx += 1

    vars_of_diffs = np.array([np.nanvar(pair_diff_cut[sigclipmasks[i]==1]) for i, pair_diff_cut in enumerate(pair_diffs_cut)])
    means_of_means = np.array([np.nanmean(pair_mean_cut[sigclipmasks[i]==1]) for i, pair_mean_cut in enumerate(pair_means_cut)])
    
    print('vars of diffs:', vars_of_diffs)
    print('means of means:', means_of_means)

    return pair_means_cut, pair_diffs_cut, vars_of_diffs, means_of_means


def sigma_clip_maskonly(vals, previous_mask=None, sig=5, show=False):

    if previous_mask is not None:
        vals[previous_mask==0] = np.nan
        sigma_val = np.nanstd(vals)
    else:
        sigma_val = np.nanstd(vals)
            
    abs_dev = np.abs(vals-np.nanmedian(vals))
    
    if show:
        plt.figure()
        plt.title('absolute deviations')
        plt.hist(abs_dev[~np.isnan(abs_dev)], bins=30)
        plt.yscale('log')
        plt.show()
    
    mask = (abs_dev < sig*sigma_val).astype(np.int)
    mask *= (~np.isnan(abs_dev)).astype(np.int)
    
    if show:
        
        plt.figure()
        plt.title('hist in sigma_clip_maskonly')
        plt.hist(vals[mask==1], bins=30)
        plt.yscale('log')
        plt.show()

    return mask



