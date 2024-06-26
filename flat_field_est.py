import numpy as np
import astropy.io.fits as fits
import astropy.wcs as wcs
import pandas as pd
import scipy.io
from scipy.ndimage import gaussian_filter
import matplotlib
import matplotlib.pyplot as plt
import pickle
# from astropy.convolution import convolve
# from astropy.convolution import Gaussian2DKernel


def compute_ff_bias(mean_normalizations, noise_rms=None, weights=None, mask_fractions=None, \
					mean_normalizations_cross=None, weights_cross=None):
	
	''' Calculates multiplicative bias in power spectrum due to stacked flat field error


	Parameters
	----------

	mean_normalizations : 
	noise_rms : 
	weights : 
	mask_fractions : 

	'''
	ff_biases = []
	
	mean_normalizations = np.array(mean_normalizations)
	weights = np.array(weights)
	if weights is None:
		weights = np.ones_like(mean_normalizations)
		
		
	for j, mean_norm in enumerate(mean_normalizations):
		
		ff_weights = list(weights.copy())
		ff_meannorms = list(mean_normalizations.copy())
		del(ff_weights[j])
		del(ff_meannorms[j])
		ff_weights /= np.sum(ff_weights)

		if weights_cross is not None and mean_normalizations_cross is not None:
			ff_weights_b = list(weights_cross.copy())
			ff_meannorms_b = list(mean_normalizations_cross)

			del(ff_weights_b[j])
			del(ff_meannorms_b[j])
			ff_weights_b /= np.sum(ff_weights_b)

			onf_meanprod = mean_normalizations[j]*mean_normalizations_cross[j]
			normratio = onf_meanprod*(ff_weights*ff_weights_b)/(ff_meannorms*ff_meannorms_b)
		
			ff_bias = np.sum(normratio**2)

		else:
			ff_bias = mean_norm**2*(np.sum(np.array(ff_weights)**2/np.array(ff_meannorms)**2))
		
		if mask_fractions is not None:
			ff_mask_fractions = list(mask_fractions.copy())
			
			del(ff_mask_fractions[j])
			
			mask_fac = np.sum(ff_mask_fractions)
			
			ff_bias *= len(ff_mask_fractions)/mask_fac
		
		ff_biases.append(1.+ff_bias)
	
	
	return ff_biases


def infill_ff(sumim, ff_estimate, infill_smooth_scale=3., target_mask=None, stack_mask=None):
	
	kernel = Gaussian2DKernel(infill_smooth_scale)
	astropy_conv = convolve(sumim, kernel)
	ff_estimate[np.isnan(ff_estimate)] = astropy_conv[np.isnan(ff_estimate)]
	if target_mask is not None:
		ff_estimate[target_mask==0] = 1.0
	ff_mask = (ff_estimate != 0)*(ff_estimate > ff_min)
	if stack_mask is not None:
		ff_mask *= stack_mask
	ff_estimate[ff_mask==0] = 1.0
	
	return ff_estimate, ff_mask

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


def compute_stack_ff_estimate(images, masked_images=None, masks=None,  target_image=None, target_mask=None, \
							  weights=None, means=None, inv_var_weight=False,  \
							 show_plots=False, infill=False, infill_smooth_scale=3., stack_mask=None, \
							  ff_min=0.2, field_nfrs=None, verbose=False):
	"""
	Computes the stacked ff estimate given a collection of images. 

	Parameters
	----------

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
		Default is False.


	"""

	if masks is None:
		if verbose:
			print('Masks is None, setting all to ones..')
		masks = [np.full(images[i].shape, 1) for i in range(len(images))]
			
	if masked_images is None:
		masked_images = [np.ma.array(images[i], mask=(masks[i]==0)) for i in range(len(images))]
	if means is None:
		means = [np.ma.mean(im) for im in masked_images]
		
	if weights is None:
		if inv_var_weight:
			if field_nfrs is None:
				field_nfrs = np.array([1. for x in range(len(images))])
			weights = np.array([1./(image_mean*field_nfrs[x]) for x, image_mean in enumerate(means)])
		else:
			weights = np.ones((len(masked_images),))
	weights /= np.sum(weights)        
	
	if verbose:
		print('weights are ', weights)
		print('means are ', means)
	
	weight_ims = np.array([mask.astype(float) for mask in masks])
	ff_indiv = np.zeros_like(masked_images)

	for i in range(len(images)):
		
		obsmean = np.ma.mean(masked_images[i])
		ff_indiv[i] = masks[i]*images[i]/obsmean
		weight_ims[i] *= weights[i]

	sum_weight_ims = np.sum(weight_ims, axis=0)
	sumim = np.sum(ff_indiv*weight_ims, axis=0)/sum_weight_ims
	ff_estimate = sumim.copy()

	if infill:
		if verbose:
			print('infilling masked pixels with smoothed version of FF..')
		ff_estimate, ff_mask = infill_ff(sumim, ff_estimate, infill_smooth_scale=infill_smooth_scale, target_mask=target_mask, stack_mask=stack_mask)
		return ff_estimate, ff_mask, weights

	else:
		return ff_estimate, None, weights



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

