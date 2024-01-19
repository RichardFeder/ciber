import numpy as np
import astropy.io.fits as fits
import astropy.wcs as wcs
import pandas as pd
import config
import scipy
import scipy.io

''' Various numerical routines and general data operations '''


def compute_fourier_weights(cl2d_all, stdpower=2, mode='mean'):
	
	fw_std = np.std(cl2d_all, axis=0)
	fw_std[fw_std==0] = np.inf
	fourier_weights = 1./(fw_std**stdpower)
	print('max of fourier weights is ', np.max(fourier_weights))
	fourier_weights /= np.max(fourier_weights)
	
	if mode=='mean':
		mean_cl2d = np.mean(cl2d_all, axis=0)
	elif mode=='median':
		mean_cl2d = np.median(cl2d_all, axis=0)
	
	return mean_cl2d, fourier_weights

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


def iter_sigma_clip_mask(image, sig=5, nitermax=10, mask=None):
	# this version makes copy of the mask to be modified, rather than modifying the original
	# image assumed to be 2d
	iteridx = 0
	
	summask = image.shape[0]*image.shape[1]

	if mask is not None:
		running_mask = mask.copy()
	else:
		running_mask = np.ones_like(image)
		
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
    mask = (abs_dev < sig*sigma_val).astype(int)

    return mask

def compute_Neff(weights):
    N_eff = np.sum(weights)**2/np.sum(weights**2)

    return N_eff

def get_l2d(dimx, dimy, pixsize):
    lx = np.fft.fftfreq(dimx)*2
    ly = np.fft.fftfreq(dimy)*2
    lx = np.fft.ifftshift(lx)*(180*3600./pixsize)
    ly = np.fft.ifftshift(ly)*(180*3600./pixsize)
    ly, lx = np.meshgrid(ly, lx)
    l2d = np.sqrt(lx**2 + ly**2)
    
    return l2d

def generate_map_meshgrid(ra_cen, dec_cen, nside_deg, dimx, dimy):
    
    ra_range = np.linspace(ra_cen - 0.5*nside_deg, ra_cen + 0.5*nside_deg, dimx)
    dec_range = np.linspace(dec_cen - 0.5*nside_deg, dec_cen + 0.5*nside_deg, dimy)
    map_ra, map_dec = np.meshgrid(ra_range, dec_range)
    
    return map_ra, map_dec


def get_q0_post(q, nwide):
	q0 = int(np.floor(q)-nwide)
	if q - np.floor(q) >= 0.5:
		q0 += 1
	return q0

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return average, np.sqrt(variance)

def mean_sub_masked_image(image, mask):
	masked_image = image*mask
	masked_image[np.isinf(masked_image)] = 0
	masked_image[np.isnan(masked_image)] = 0
	unmasked_mean = np.mean(masked_image[mask==1])
	masked_image[mask==1] -= unmasked_mean
	
	return masked_image

def precomp_gradsub(masks, dimx, dimy):
    dot1s, mask_ravs = [], []
    for m, mask in enumerate(masks):
        dot1, Xgrad, mask_rav = precomp_gradient_dat(dimx, dimy, mask=mask)
        dot1s.append(dot1)
        mask_ravs.append(mask_rav)
    mask_ravs = np.array(mask_ravs)

    return dot1s, mask_ravs

def perform_grad_sub(k, empty_aligned_objs, mask, obj_idx, dot1s, Xgrad, mask_rav, nsims_perf):
    planes = np.array([fit_gradient_to_map_precomp(empty_aligned_objs[obj_idx][k,s].real, dot1s[k], Xgrad, mask_rav=mask_rav) for s in range(nsims_perf)])
    empty_aligned_objs[obj_idx][k].real -= np.array([mask*planes[s] for s in range(nsims_perf)])
    return planes


def randn_skew_fast(N=None, shape=None, alpha=0.0, loc=0.0, scale=1.0):
	if shape is None:
		shape = (N,N)
	sigma = alpha / np.sqrt(1.0 + alpha**2) 
	u0 = np.random.normal(0, 1, shape)
	v = np.random.normal(0, 1, shape)
	u1 = (sigma*u0 + np.sqrt(1.0 - sigma**2)*v) * scale
	u1[u0 < 0] *= -1
	u1 = u1 + loc
	return u1

def generate_rand_skew_layered(skew_upsamp, realiz_shape, skew_max=8, plot=False):
	
	skew_realiz = np.zeros(realiz_shape)
	
	skew_upsamp_round = np.round(skew_upsamp)
	skew_upsamp_round[skew_upsamp_round > skew_max] = skew_max
	
	plot_map(skew_upsamp_round, title='skew upsamp round')
	
	for skewval in range(skew_max+1):
		
		if skewval==0:
			p = np.ones_like(skew_realiz)
		else:
			p = randn_skew_fast(shape=realiz_shape, alpha=skewval, loc=1.0)
		
		skew_realiz_pmask = np.broadcast_to(skew_upsamp_round==skewval, realiz_shape)
		p[~skew_realiz_pmask] = 0.
		skew_realiz += p
		
	if plot:
		plot_map(skew_realiz[0], title='skew realiz')
	
	return skew_realiz

def weighted_powerspec_2d(ps2d, l2d_mask, fw_image=None):
	ps2d_copy = ps2d.copy()
	ps2d_copy[l2d_mask==0] = 0.

	if fw_image is not None:
		fw_image_copy = fw_image.copy()
		
		fw_image_copy[ps2d_copy==0.] = 0.
		fw_image_copy[fw_image_copy > 0.] /= np.sum(fw_image_copy) 

		ps2d_copy *= fw_image_copy
				
		return ps2d_copy, fw_image_copy
	
	return ps2d_copy

def dist(NAXIS):

    axis = np.linspace(-NAXIS/2+1, NAXIS/2, NAXIS)
    result = np.sqrt(axis**2 + axis[:,np.newaxis]**2)
    return np.roll(result, NAXIS/2+1, axis=(0,1))