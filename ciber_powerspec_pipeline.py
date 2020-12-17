import numpy as np
import matplotlib 
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd

import astropy.wcs as wcs
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky
from scipy.ndimage import gaussian_filter
from sklearn import tree

import os

from cross_spectrum_analysis import *
from mkk_parallel import *

''' 

CIBER_PS_pipeline() is the core module for CIBER power spectrum analysis in Python. 
It contains data processing routines and the basic steps going from flight images in e-/s to
estimates of CIBER auto- and cross-power spectra. 


'''

def mean_sub_masked_image(image, mask):
    masked_image = image*mask # dividing by the flat field introduces infinities (and maybe NaNs too?)
    masked_image[np.isinf(masked_image)] = 0
    masked_image[np.isnan(masked_image)] = 0
    unmasked_mean = np.mean(masked_image[mask==1])
    masked_image[mask==1] -= unmasked_mean
    
    return masked_image

class CIBER_PS_pipeline():
    
    def __init__(self, \
                base_path='/Users/luminatech/Documents/ciber2/ciber/', \
                data_path=None, \
                dimx=1024, \
                dimy=1024, \
                n_ps_bin=19):
        
        self.base_path = base_path
        self.data_path = data_path
        if data_path is None:
            self.data_path = self.base_path+'data/fluctuation_data/'
            
        self.Mkk_obj = Mkk_bare(dimx=dimx, dimy=dimy, ell_min=180.*(1024./dimx), nbins=n_ps_bin)
        self.Mkk_obj.precompute_mkk_quantities(precompute_all=True)
        self.B_ell = None
        self.cal_facs = dict({1:-170.3608, 2:-57.2057})
        self.dimx, self.dimy = dimx, dimy
        self.n_ps_bin = n_ps_bin
    
    def load_flight_image(self, ifield, inst, flight_fpath=None, verbose=False, inplace=True):
        ''' Loads flight image from data release '''
        if verbose:
            print('Loading flight image from TM'+str(inst)+', field '+str(ifield)+'..')
        if flight_fpath is None:
            flight_fpath = self.data_path + 'TM'+str(inst)+'/flight/field'+str(ifield)+'_flight.fits'
        
        if inplace:
            self.flight_image = fits.open(flight_fpath)[0].data
        else:
            return fits.open(flight_fpath)[0].data
        
    def load_FF_image(self, ifield, inst, ff_fpath=None, verbose=False, inplace=True):
        ''' Loads flat field estimate from data release '''
        if verbose:
            print('Loading flat field image from TM'+str(inst)+', field '+str(ifield)+'..')
        if ff_fpath is None:
            ff_fpath = self.data_path + 'TM'+str(inst)+'/FF/field'+str(ifield)+'_FF.fits'
        
        if inplace:
            self.FF_image = fits.open(ff_fpath)[0].data  
        else:
            return fits.open(ff_fpath)[0].data

    def load_FW_image(self, ifield, inst, fw_fpath=None, verbose=False, inplace=True):
        ''' Loads fourier weight image estimate from data release '''
        if verbose:
            print('Loading fourier weight image estimate from TM'+str(inst)+', field '+str(ifield)+'..')
        if fw_fpath is None:
            fw_fpath = self.data_path + 'TM'+str(inst)+'/FW/field'+str(ifield)+'_FW.fits'
        
        if inplace:
            self.FW_image = fits.open(fw_fpath)[0].data  
        else:
            return fits.open(fw_fpath)[0].data
        
    def load_mkk_mats(self, ifield, inst, mkk_fpath=None, verbose=False, inplace=True):
        ''' Loads Mkk and inverse Mkk matrices '''
        if verbose:
            print('Loading Mkk and inverse Mkk matrix estimates from TM'+str(inst)+', field '+str(ifield)+'..')
        
        if mkk_fpath is None:
            mkk_fpath = self.data_path + 'TM'+str(inst)+'/mkk/field'+str(ifield)+'_mkk_mats.npz'
        
        mkkfile = np.load(mkk_fpath)
        if inplace:
            self.Mkk_matrix = mkkfile['Mkk']
            self.inv_Mkk = mkkfile['inv_Mkk']
        else:
            return mkkfile['Mkk'], mkkfile['inv_Mkk']
        
    def load_mask(self, ifield, inst, masktype, mask_fpath=None, verbose=False, inplace=True):
        ''' Loads mask from data release '''
        if verbose:
            print('Loading mask image from TM'+str(inst)+', field '+str(ifield))
        if mask_fpath is None:
            mask_fpath = self.data_path + 'TM'+str(inst)+'/mask/field'+str(ifield)+'_'+masktype+'.fits'
        
        if inplace:
            if masktype=='maskInst_clean':
                self.maskInst_clean = fits.open(mask_fpath)[0].data 
            elif masktype=='strmask':
                self.strmask = fits.open(mask_fpath)[0].data
            elif masktype=='bigmask':
                self.bigmask = fits.open(mask_fpath)[0].data
        else:
            return fits.open(mask_fpath)[0].data
    
    def compute_mkk_matrix(self, mask, nsims=100, n_split=2, inplace=True):
        Mkk_matrix = self.Mkk_obj.get_mkk_sim(mask, nsims, n_split=n_split)
        inv_Mkk = compute_inverse_mkk(Mkk_matrix)
        
        if inplace:
            self.Mkk_matrix = Mkk_matrix
            self.inv_Mkk = inv_Mkk
        else:
            return Mkk_matrix, inv_Mkk
        
    def load_dark_current_template(self, ifield, inst, dc_fpath=None, verbose=False, inplace=True):
        ''' Loads dark current template from data release '''
        if verbose:
            print('Loading dark current template from TM'+str(inst)+'..')
        if dc_fpath is None:
            dc_fpath = self.data_path + 'TM'+str(inst)+'/DCtemplate.fits'
        
        if inplace:
            self.dc_template = fits.open(dc_fpath)[0].data  
        else:
            return fits.open(dc_fpath)[0].data
        
    def load_psf(self, ifield, inst, psf_fpath=None, verbose=False, inplace=True):
        ''' Loads PSF estimate from data release '''
        
        if verbose:
            print('Loading PSF template from TM'+str(inst)+', field '+str(ifield)+'..')
        if psf_fpath is None:
            psf_fpath = self.data_path + 'TM'+str(inst)+'/beam_effect/field'+str(ifield)+'_psf.fits'
        
        psf = fits.open(psf_fpath)[0].data
        psf_template = np.zeros(shape=(self.dimx, self.dimy))
        psf_template[:psf.shape[0], :psf.shape[1]] = psf
        
        if inplace:
            self.psf_template = psf_template
        else:
            return psf_template
        
    def load_data_products(self, ifield, inst, load_all=True, flight_image=False, dark_current=False, psf=False, \
                           maskInst_clean=False, strmask=False, \
                           FW_image=False, FF_image=False, verbose=True):
        
        if flight_image or load_all:
            self.load_flight_image(ifield, inst, verbose=verbose)
        if dark_current or load_all:
            self.load_dark_current_template(ifield, inst, verbose=verbose)
        if psf or load_all:
            self.load_psf(ifield, inst, verbose=verbose)
        if maskInst_clean or load_all:
            self.load_mask(ifield, inst, 'maskInst_clean', verbose=verbose)
        if strmask or load_all:
            self.load_mask(ifield, inst, 'strmask', verbose=verbose)
        if FW_image or load_all:
            self.load_FW_image(ifield, inst, verbose=verbose)
        if FF_image or load_all:
            self.load_FF_image(ifield, inst, verbose=verbose)
            
    def compute_beam_correction(self, psf=None, verbose=False, inplace=True):
        
        if verbose:
            print('Computing beam correction..')
        if psf is None:
            if verbose:
                print('No template explicitly provided, using self.psf_template to compute Bl..')
            psf = self.psf_template
        
        rb, Bl, Bl_std = get_power_spec(psf-np.mean(psf), lbinedges=self.Mkk_obj.binl, lbins=self.Mkk_obj.midbin_ell, return_Dl=False)
        
        B_ell = np.sqrt(Bl)/np.max(np.sqrt(Bl))
        B_ell_std = (Bl_std/(2*np.sqrt(Bl)))/np.max(np.sqrt(Bl))
        
        if inplace:
            self.B_ell = B_ell
            self.B_ell_std = B_ell_std
        else:
            return B_ell, B_ell_std
        
    def compute_process_cross_power_spectrum(self, insts,  ifields,\
                                               flight_fpaths=[None, None], psf_fpaths=[None, None], \
                                               mask_fpaths=[None, None]):
        ''' This will compute the cross power spectrum between two maps and make the appropriate corrections. 
        
        Status : Work in progress
        
        Parameters
        ----------
        
        insts : `list' of ints, length 2
        
        ifields : `list' of ints, length 2

        flight_fpaths : `list' of strings, length 2, optional
        
            Default is [None, None].
            
        Returns
        -------
        
        TODO
        
        '''
        
        flight_images, psfs, srcmasks, instmasks = [[] for x in range(5)]
        for i in range(2):
            flight_images.append(self.load_flight_image(ifields[i], insts[i], flight_fpath=flight_fpaths[i], inplace=False))
            psfs.append(self.load_psf(ifields[i], insts[i], inplace=False))
            srcmasks.append(self.load)
        
        
        flight_image_1 = self.load_flight_image(ifield1, inst1, flight_fpath=flight_fpaths[0], inplace=False)
        flight_image_2 = self.load_flight_image(ifield2, inst2, flight_fpath=flight_fpaths[1], inplace=False)

        
        
        
                
    def compute_processed_power_spectrum(self, inst, mask=None, B_ell=None, inv_Mkk=None,\
                                         flight_image=None, mkk_correct=True, beam_correct=True,\
                                         FF_correct=True, apply_FW=True, noise_debias=False):
        
        if flight_image is None:
            flight_image = self.flight_image.copy()
            
        flight_image *= self.cal_facs[inst] # convert from ADU/fr to nW m^-2 sr^-1
        
        if FF_correct:
            flight_image /= self.FF_image # divide image by the flat field estimate
            
        if mask is None:
            mask = self.maskInst_clean*self.strmask # load mask
        
        masked_image = mean_sub_masked_image(flight_image, mask) # apply mask and mean subtract unmasked pixels

        weights=None
        if apply_FW:
            weights = self.FW_image # this feeds Fourier weights into get_power_spec, which uses them with Cl2D
            
        lbins, cl_proc, cl_proc_err = get_power_spec(masked_image, mask=mask, weights=weights, lbinedges=self.Mkk_obj.binl, lbins=self.Mkk_obj.midbin_ell)
            
        if noise_debias: # subtract noise bias
            if N_ell is None:
                cl_proc -= self.N_ell
            else:
                cl_proc -= N_ell
                
        if mkk_correct: # apply inverse mode coupling matrix to masked power spectrum
            if inv_Mkk is None:
                cl_proc = np.dot(self.inv_Mkk.transpose(), cl_proc)
            else:
                cl_proc = np.dot(inv_Mkk.transpose(), cl_proc)
            
        if beam_correct: # undo the effects of the PSF by dividing power spectrum by B_ell
            if B_ell is None:
                B_ell = self.B_ell
            assert len(B_ell)==len(cl_proc)
            cl_proc /= B_ell
                
        return lbins, cl_proc, masked_image
            





