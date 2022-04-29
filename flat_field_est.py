import numpy as np
import astropy.io.fits as fits
import astropy.wcs as wcs
import pandas as pd
import scipy.io
from scipy.ndimage import gaussian_filter
import matplotlib
import matplotlib.pyplot as plt
import pickle
from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel


def compute_ff_bias(mean_normalizations, noise_rms=None, weights=None, mask_fractions=None):
    
    ''' Calculates multiplicative bias in power spectrum based on '''
    ff_biases = []
    
    mean_normalizations = np.array(mean_normalizations)
    weights = np.array(weights)
    if weights is None:
        weights = np.ones_like(mean_normalizations)
        
        
    for i, mean_norm in enumerate(mean_normalizations):
        
        ff_weights = list(weights.copy())
        ff_meannorms = list(mean_normalizations.copy())
        
        del(ff_weights[i])
        del(ff_meannorms[i])
        
        ff_weights /= np.sum(ff_weights)
        
        ff_bias = mean_norm**2*(np.sum(np.array(ff_weights)**2/np.array(ff_meannorms)**2))
        
        if mask_fractions is not None:
            ff_mask_fractions = list(mask_fractions.copy())
            
            del(ff_mask_fractions[i])
            
            mask_fac = np.sum(ff_mask_fractions)
            
            ff_bias *= len(ff_mask_fractions)/mask_fac
        
        ff_biases.append(1.+ff_bias)
    
    
    return ff_biases

def load_flat_from_mat(mat=None, matpath=None, flatidx=6):
	
	if mat is None:
		if matpath is not None:
			mat = loadmat(matpath)
		else:
			return None
		
	flat = mat[0][0][flatidx][0][0][0]
	
	return flat

def compute_ff_mask(joint_masks, ff_stack_min=1):
	''' 
	Computes the additional masking for each on-field by computing the mask intersection of off-fields used to estimate the flat field.
	Each original on-field mask is then multiplied by the additional FF mask.


	Inputs
	------
	joint_masks : `~np.array~` or `list` of `ints` or `bools`.
	ff_stack_min : `int`. Minimum number of unmasked fields required to include in flat field estimate. 
		This is evaluated for the stacked masks on a per pixel basis. 
		Default is 1.

	Returns
	-------

	ff_joint_masks : `~np.array~` of type `bool`. Product of original masks and FF-derived masks

	'''
	
	ff_joint_masks = []
	for j, joint_mask in enumerate(joint_masks):
		
		stack_mask = list(np.array(joint_masks).copy().astype(np.bool))
		
		del(stack_mask[j])
		
		sum_stack_mask = np.sum(stack_mask, axis=0)
		
		ff_joint_mask = (sum_stack_mask > ff_stack_min)
		
		ff_joint_masks.append(ff_joint_mask*joint_mask)
		
	return np.array(ff_joint_masks)


def compute_stack_ff_estimate(images, masks=None, target_image=None, target_mask=None, weights=None, means=None, inv_var_weight=True,  \
							 show_plots=False, infill=True, infill_smooth_scale=3., ff_stack_min=1, ff_min=0.2, field_nfrs=None, verbose=False):
	

	''' 
	Computes the stacked ff estimate given a collection of images. 

	Inputs
	------

	images : 
	masks (optional) : 
		Default is None.
	target_image (optional) : 
		Default is None.
	target_mask (optional) : 
		Default is None.
	weights (optional) : Can be unnormalized
		Default is None.
	means (optional) : 
		Default is None.
	inv_var_weight : 
		Default is True.
	show_plots : 
	infill :
		Default is True.

	infill_smooth_scale : 
		Default is 3 [pixels].
	ff_stack_min : 
	ff_min : 
	field_nfrs : 
	verbose : 
		Default is False.

	Returns
	-------
	
	ff_estimate : 
	ff_mask :
	weights :

	'''
	if field_nfrs is None:
		field_nfrs = np.array([1. for x in range(len(images))])

	stack_mask = None
	if masks is not None and masks[0] is not None:
		masked_images = [np.ma.array(images[i], mask=~masks[i]) for i in range(len(images))]

		if show_plots:
			for i in range(len(masked_images)):
				plt.figure(figsize=(10, 5))
				plt.subplot(1,2,1)
				plt.title('masked image i='+str(i))
				plt.imshow(images[i]*masks[i], origin='lower')

				plt.subplot(1,2,2)
				plt.title('image i='+str(i))
				plt.imshow(images[i], origin='lower')

				plt.show()
		
		sum_mask = np.sum(np.array(masks), axis=0)

		stack_mask = (sum_mask >= ff_stack_min)

		if show_plots:
			plt.figure()
			plt.imshow(sum_mask)
			plt.colorbar()
			plt.show()
	else:
		masks = [np.full(images[i].shape, True) for i in range(len(images))]
		masked_images = [np.ma.array(images[i], mask=np.full(images[i].shape, False)) for i in range(len(images))]
		
		
	if means is None:
		means = [np.ma.mean(im) for im in masked_images]
	
	if verbose:
		print('but means here are ', means)

	if weights is None:
		if inv_var_weight:
			weights = np.array([1./(image_mean*field_nfrs[x]) for x, image_mean in enumerate(means)])
		else:
			weights = np.ones((len(images),))

	# just to make sure
	weights /= np.sum(weights)        


	if verbose:
		print('weights are ', weights)
	# obs_maps = np.array([weights[b]*(images[b]/np.ma.mean(images[b])) for b in range(len(weights))])   
	# obs_maps = np.array([weights[b]*(images[b]/np.sqrt(np.ma.mean(images[b]))) for b in range(len(weights))])
	# print('min max of obs_maps ', np.min(obs_maps), np.max(obs_maps))

	sumim = np.zeros_like(images[0])

	# sqrtim = np.zeros_like(images[0])
	weightim = np.zeros_like(images[0])

	weight_ims = np.array([mask.astype(np.float) for mask in masks])
	ff_indiv = np.zeros_like(images)

	for i in range(len(images)):
		obsmean = np.ma.mean(masked_images[i])

		ff_indiv[i] = masks[i]*images[i]/obsmean
		weight_ims[i] *= weights[i]

		# sumim += weights[i]*masks[i]*images[i]/obsmean # up until 1/18
		# obssqrt = np.sqrt(obsmean)
		# sumim += images[i]*masks[i]
		# weightim += obsmean*np.ones_like(images[i])*masks[i]
		# sumim += images[i]*masks[i]/obssqrt # previous weighting
		# weightim += obssqrt*np.ones_like(images[i])*masks[i]
		

	sum_weight_ims = np.sum(weight_ims, axis=0)

	sumim = np.sum(ff_indiv*weight_ims, axis=0)/sum_weight_ims
	
	ff_estimate = sumim.copy()

	if infill:
		if verbose:
			print('infilling masked pixels with smoothed version of FF..')
		kernel = Gaussian2DKernel(infill_smooth_scale)
		astropy_conv = convolve(sumim, kernel)

		# ff_estimate = sumim.copy()
		ff_estimate[np.isnan(ff_estimate)] = astropy_conv[np.isnan(ff_estimate)]

		if target_mask is not None:
			ff_estimate[target_mask==0] = 1.0

		ff_mask = (ff_estimate != 0)*(ff_estimate > ff_min)

		if stack_mask is not None:
			ff_mask *= stack_mask

		ff_estimate[ff_mask==0] = 1.0

		return ff_estimate, ff_mask, weights

	else:
		return ff_estimate, None, weights


def plot_indiv_ps_results_fftest(lb, list_of_recovered_cls, cls_truth=None, n_skip_last = 3, mean_labels=None, return_fig=True, ciblab = 'CIB + DGL ground truth', truthlab='truth field average', ylim=[1e-3, 1e2]):
	prefac = lb*(lb+1)/(2*np.pi)
	
	if mean_labels is None:
		mean_labels = [None for x in range(len(list_of_recovered_cls))]
		
	f = plt.figure(figsize=(8,6))
	
	for i, recovered_cls in enumerate(list_of_recovered_cls):
		
		for j in range(recovered_cls.shape[0]):
			
			plt.plot(lb[:-n_skip_last], np.sqrt(prefac*np.abs(recovered_cls[j]))[:-n_skip_last], linewidth=1, marker='.', color='C'+str(i+2), alpha=0.3)
			
		plt.plot(lb[:-n_skip_last], np.sqrt(prefac*np.abs(np.mean(np.abs(recovered_cls), axis=0)))[:-n_skip_last], marker='*', label=mean_labels[i], color='C'+str(i+2), linewidth=3)

	if cls_truth is not None:
		for j in range(cls_truth.shape[0]):
			label = None
			if j==0:
				label = ciblab
			plt.plot(lb[:-n_skip_last], np.sqrt(prefac*cls_truth[j])[:-n_skip_last], color='k', alpha=0.3, linewidth=1, linestyle='dashed', marker='.', label=label)

		plt.plot(lb[:-n_skip_last], np.sqrt(prefac*np.mean(cls_truth, axis=0))[:-n_skip_last], color='k', linewidth=3, label=truthlab)

				
	plt.legend(fontsize=14)
	plt.xscale('log')
	plt.yscale('log')
	plt.ylim(ylim)
	plt.xlabel('Multipole $\\ell$', fontsize=20)
	plt.ylabel('$\\left[\\frac{\\ell(\\ell+1)}{2\\pi}C_{\\ell}\\right]^{1/2}$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=20)
	plt.tick_params(labelsize=16)
	# plt.savefig('/Users/luminatech/Downloads/input_recover_powerspec_fivefields_estimated_ff_bkg250_bl_cut_simidx'+str(simidx)+'_min_stack_ff='+str(min_stack_ff)+'.png', bbox_inches='tight')
	plt.show()
	
	if return_fig:
		return f


