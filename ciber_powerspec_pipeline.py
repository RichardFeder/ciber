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

def verbprint(verbose, text):
	if verbose:
		print(text)

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
        self.cal_facs = dict({1:-170.3608, 2:-57.2057}) # multiplying by cal_facs converts from ADU/fr to nW m^-2 sr^-1
        self.g1_facs = dict({1:-1.5459, 2:-1.3181}) # multiplying by g1 converts from ADU/fr to e-/s
        self.dimx, self.dimy = dimx, dimy
        self.n_ps_bin = n_ps_bin
        self.pixsize = 7. # arcseconds
        self.pixlength_in_deg = self.pixsize/3600. # pixel angle measured in degrees
        self.arcsec_pp_to_radian = self.pixlength_in_deg*(np.pi/180.) # conversion from arcsecond per pixel to radian per pixel
        self.photFactor = np.sqrt(1.2)
        self.Mkk_obj.delta_ell = self.Mkk_obj.binl[1:]-self.Mkk_obj.binl[:-1]
        self.fsky = self.dimx*self.dimy*self.Mkk_obj.arcsec_pp_to_radian**2 / (4*np.pi)
        
        self.field_exposure = 50. # seconds
        self.frame_period = 1.78 # seconds
        self.nfr = int(self.field_exposure/self.frame_period)
        self.frame_rate = 1./self.frame_period
      
    '''
        Repeated arguments
        ------------------
        verbose : boolean, optional
            If True, functions output with many print statements describing what steps are occurring.
            Default is 'False'.
        
        inplace : boolean, optional
            If True, output variable stored in CIBER_PS_pipeline class object. 
            Default is 'True'.
                   
        ifield : integer
            Index of science field. Currently required as a parameter but may want to make optional.
        
        inst : integer
            Indicates which instrument data to use (1 == J band, 2 == H band)
    
    '''
    
    def compute_mkk_matrix(self, mask, nsims=100, n_split=2, inplace=True):
        ''' Computes Mkk matrix and its inverse for a given mask '''
        Mkk_matrix = self.Mkk_obj.get_mkk_sim(mask, nsims, n_split=n_split)
        inv_Mkk = compute_inverse_mkk(Mkk_matrix)
        
        if inplace:
            self.Mkk_matrix = Mkk_matrix
            self.inv_Mkk = inv_Mkk
        else:
            return Mkk_matrix, inv_Mkk

    def knox_errors(self, C_ell, N_ell, use_beam_fac=True, B_ell=None, snr=False):
        
        if use_beam_fac and B_ell is None:
            B_ell = self.B_ell
        
        output = compute_knox_errors(self.Mkk_obj.midbin_ell, C_ell, N_ell, self.Mkk_obj.delta_ell, fsky=self.fsky, \
                                     B_ell=B_ell, snr=snr)
        
        return output
    
    def noise_model_realization(self, inst, maplist_split_shape, noise_model, fft_obj=None, adu_to_sb=True, chisq=True, div_fac=2.83, \
                               read_noise=True, photon_noise=True, shot_sigma_sb=None, image=None, nfr=None, frame_rate=None):
        
        self.noise = np.random.normal(0, 1, size=maplist_split_shape)+ 1j*np.random.normal(0, 1, size=maplist_split_shape)

        rnmaps = None
        snmaps = None
        
        if read_noise:
            if chisq:
                chi2_realiz = (np.random.chisquare(2., size=maplist_split_shape)/div_fac)**2

            if fft_obj is not None:
                if chisq:
                    rnmaps = np.sqrt(self.dimx*self.dimy)*fft_obj(self.noise*ifftshift(np.sqrt(chi2_realiz*noise_model)))
                else:
                    rnmaps = np.sqrt(self.dimx*self.dimy)*fft_obj(self.noise*ifftshift(np.sqrt(noise_model)))
            else:
                rnmaps = []
                if len(maplist_split_shape)==3:
                    for r in range(maplist_split_shape[0]):
                        if chisq:
                            rnmap = np.sqrt(self.dimx*self.dimy)*ifft2(self.noise[r]*ifftshift(np.sqrt(chi2_realiz[r]*noise_model)))
                        else:
                            rnmap = np.sqrt(self.dimx*self.dimy)*ifft2(self.noise[r]*ifftshift(np.sqrt(noise_model)))

                    rnmaps.append(rnmap)
                else:
                    assert len(self.noise.shape()) == 2
                    assert len(chi2_realiz.shape()) == 2
                    # single map

                    if chisq:
                        rnmaps = np.sqrt(self.dimx*self.dimy)*ifft2(self.noise*ifftshift(np.sqrt(chi2_realiz*noise_model)))
                    else:
                        rnmaps = np.sqrt(self.dimx*self.dimy)*ifft2(self.noise*ifftshift(np.sqrt(noise_model)))

            rnmaps = np.array(rnmaps).real
        
        if photon_noise:
            nsims = 1
            if len(maplist_split_shape)==3:
                nsims = maplist_split_shape[0]
            
            snmaps = self.compute_shot_noise_maps(inst, image, nsims, shot_sigma_sb=shot_sigma_sb, nfr=nfr, frame_rate=frame_rate)

        if adu_to_sb:
            rnmaps *= self.cal_facs[inst]/self.arcsec_pp_to_radian

        return rnmaps, snmaps
    
    
    def compute_FW_from_noise_sim(self, ifield=None, inst=None, mask=None, apply_mask=True, noise_model=None, noise_model_fpath=None, verbose=False, inplace=True, \
                                 use_pyfftw = True, nsims = 50, n_split=5, show=False, read_noise=True, photon_noise=True, shot_sigma_sb=None, image=None, stdpower=1):
        
        verbprint(verbose, 'Computing fourier weight image from noise sims for TM'+str(inst)+', field '+str(ifield)+'..')
        verbprint(verbose, 'WARNING : This operation can take a lot of memory if caution is thrown to the wind')
        
        if noise_model is None:
            if noise_model_fpath is None:
                if ifield is not None and inst is not None:
                    noise_model_fpath = self.data_path + 'TM' + str(inst) + '/noise_model/noise_model_Cl2D'+str(ifield)+'.fits'
                else:
                    print('Noise model not provided, noise_model_fpath not provided, and ifield/inst not provided, need more information..')
                    return
                
            noise_model = self.load_noise_Cl2D(ifield=ifield, inst=inst, noise_fpath=noise_model_fpath, inplace=False, transpose=False)

            
        if mask is None and apply_mask:
            verbprint(verbose, 'No mask provided but apply_mask=True, using union of self.strmask and self.maskInst_clean..')

            mask = self.strmask*self.maskInst_clean
            
        if show:
            plt.figure(figsize=(10, 5))
            plt.subplot(1,2,1)
            plt.imshow(noise_model, norm=matplotlib.colors.LogNorm())
            plt.colorbar()
            plt.subplot(1,2,2)
            plt.title('Mask')
            plt.imshow(mask)
            plt.colorbar()
            plt.tight_layout()
            plt.show()
        
        cl2d_all = np.zeros((nsims, self.dimx, self.dimy))
        
        if use_pyfftw:
            
            maplist_split_shape = (nsims//n_split, self.dimx, self.dimy)
            empty_aligned_objs, fft_objs = construct_pyfftw_objs(2, maplist_split_shape)
            
            for i in range(n_split):
                print('Split '+str(i+1)+' of '+str(n_split)+'..')

                rnmaps, snmaps = self.noise_model_realization(inst, maplist_split_shape, noise_model, fft_obj=fft_objs[0],\
                                                      read_noise=read_noise, photon_noise=photon_noise, shot_sigma_sb=shot_sigma_sb)
        
                for j in range(nsims // n_split):
            
                    if j==0:
                        if show:
                            plt.figure()
                            plt.subplot(1,2,1)
                            plt.title('read noise')
                            if read_noise:
                                plt.imshow(rnmaps[j], vmax=np.percentile(rnmaps[j], 95), vmin=np.percentile(rnmaps[j], 5))
                                plt.colorbar()
                            plt.subplot(1,2,2)
                            plt.title('photon noise')
                            if photon_noise:
                                plt.imshow(self.photFactor*snmaps[j], vmax=np.percentile(self.photFactor*snmaps[j], 95), vmin=np.percentile(self.photFactor*snmaps[j], 5))
                                plt.colorbar()
                            plt.show()
                            
                    if read_noise:
                        simmap = rnmaps[j].copy()
                        if photon_noise:
                            simmap += snmaps[j]

                    else:
                        if photon_noise:
                            simmap = snmaps[j].copy()

                    # multiply simulated noise map by mask
                    if apply_mask and mask is not None:
                        simmap *= mask
                                                
                    l2d, cl2drn = get_power_spectrum_2d(simmap-np.mean(simmap))
                    cl2d_all[i*nsims//n_split + j, :,:] = cl2drn/self.cal_facs[inst]**2
                
        
        fw_std = np.std(cl2d_all, axis=0) # returns std within each 2d fourier mode
        
        fourier_weights = 1./fw_std**stdpower

            
        fourier_weights /= np.max(fourier_weights)
        
        mean_cl2d = np.mean(cl2d_all, axis=0)
        
        if show:
            plt.figure(figsize=(8,8))
            plt.title('Fourier weights')
            plt.imshow(fourier_weights, origin='lower', cmap='Greys', norm=matplotlib.colors.LogNorm())
            plt.xticks([], [])
            plt.yticks([], [])
            plt.colorbar()
            plt.show()
        
        if inplace:
            self.FW_image = fourier_weights
            return mean_cl2d
        else:
            return fourier_weights, mean_cl2d
        
    def load_flight_image(self, ifield, inst, flight_fpath=None, verbose=False, inplace=True):
        ''' Loads flight image from data release '''
        verbprint(verbose, 'Loading flight image from TM'+str(inst)+', field '+str(ifield)+'..')

        if flight_fpath is None:
            flight_fpath = self.data_path + 'TM'+str(inst)+'/flight/field'+str(ifield)+'_flight.fits'
        
        if inplace:
            self.image = fits.open(flight_fpath)[0].data
        else:
            return fits.open(flight_fpath)[0].data
        
    def load_noise_Cl2D(self, ifield=None, inst=None, noise_model=None, noise_fpath=None, verbose=False, inplace=True, transpose=True, mode=None):
        
        ''' Loads 2D noise power spectrum from data release
        
        Parameters (unique)
        ----------

        noise_fpath : str, optional
        
            Default is 'None'.
        
        transpose : boolean, optional
            
            Default is 'True'.
        
        mode : str, optional 
        
            Default is 'None'.
        
        Returns
        -------
        
        noise_Cl2D : `~numpy.ndarray' of shape (self.dimx, self.dimy)
            2D noise power spectrum. If inplace==True, stored as class object.
        
        '''
        
        if noise_model is None:
            if noise_fpath is None:
                if ifield is not None and inst is not None:
                    verbprint(verbose, 'Loading 2D noise power spectrum from TM'+str(inst)+', field '+str(ifield)+'..')            
                    noise_fpath = self.data_path + 'TM'+str(inst)+'/noiseCl2D/field'+str(ifield)+'_noiseCl2D.fits'
                else:
                    print('Out of luck, need more information. Returning None')
                    return None
                
            noise_Cl2D = fits.open(noise_fpath)[0].data
            
        if transpose:
            noise_Cl2D = noise_Cl2D.transpose()
        
        if inplace:
            if mode=='unmasked':
                self.unmasked_noise_Cl2D = noise_Cl2D
            elif mode=='masked':
                self.masked_noise_Cl2D = noise_Cl2D
            else:
                self.noise_Cl2D = noise_Cl2D
        else:
            return noise_Cl2D
        
        
    def load_FF_image(self, ifield, inst, ff_fpath=None, verbose=False, inplace=True):
        ''' Loads flat field estimate from data release 
        
        Parameters (unique):
        
        ff_fpath : str, optional
        
            Default is 'None'.
        
        '''
        verbprint(verbose, 'Loading flat field image from TM'+str(inst)+', field '+str(ifield)+'..')

        if ff_fpath is None:
            ff_fpath = self.data_path + 'TM'+str(inst)+'/FF/field'+str(ifield)+'_FF.fits'
        
        if inplace:
            self.FF_image = fits.open(ff_fpath)[0].data  
        else:
            return fits.open(ff_fpath)[0].data

        
    def load_FW_image(self, ifield, inst, fw_fpath=None, verbose=False, inplace=True):
        ''' Loads fourier weight image estimate from data release '''
        verbprint(verbose, 'Loading fourier weight image estimate from TM'+str(inst)+', field '+str(ifield)+'..')

        if fw_fpath is None:
            fw_fpath = self.data_path + 'TM'+str(inst)+'/FW/field'+str(ifield)+'_FW.fits'
        
        if inplace:
            self.FW_image = fits.open(fw_fpath)[0].data
        else:
            return fits.open(fw_fpath)[0].data
        
        
    def monte_carlo_power_spectrum_sim(self, ifield, inst, image=None, mask=None, apply_mask=True, noise_model=None, masked_noise_model=None, noise_model_fpath=None,\
                                       FW_image=None, apply_FW=True, n_FW_sims=100, n_FW_split=5, verbose=False, use_pyfftw = True, \
                                       nsims = 100, n_split=5, show=False, mkk_fpath=None, mkk_correct=True, inv_Mkk=None, beam_correct=True, noise_debias=True, \
                                      FF_correct=False, add_noise=True, chisq=True, read_noise=True, photon_noise=True, nfr=None, frame_rate=None):
        
        if image is not None:
            self.image = image
            
        # load noise model
        if noise_model is None:
            
            if noise_model_fpath is None:
                noise_model_fpath = self.data_path + 'TM' + str(inst) + '/noise_model/noise_model_Cl2D'+str(ifield)+'.fits'
            verbprint(verbose, 'No noise model provided, loading from '+str(noise_model_fpath)+'..')
  
            noise_model = self.load_noise_Cl2D(ifield, inst, noise_fpath=noise_model_fpath, inplace=False, transpose=False)

        if mask is None and apply_mask:
            
            verbprint(verbose, 'No mask provided but apply_mask=True, using union of self.strmask and self.maskInst_clean..')
            mask = self.strmask*self.maskInst_clean
        
        if show:
            plt.figure()
            plt.title('Noise model [nW$^2$ m$^{-4}$ sr$^{-2}$]')
            plt.imshow(noise_model*self.cal_facs[inst]**2, norm=matplotlib.colors.LogNorm(), vmax=np.percentile(noise_model*self.cal_facs[inst]**2, 95), vmin=np.percentile(noise_model*self.cal_facs[inst]**2, 5))
            plt.colorbar()
            plt.show()
               
        if mkk_correct:
            if inv_Mkk is None and mkk_fpath is not None:   
                verbprint(verbose, 'Loading Mkk matrices from '+mkk_fpath+'..')

                mkk, inv_Mkk = self.load_mkk_mats(ifield, inst, mkk_fpath=mkk_fpath, verbose=verbose, inplace=False)

        # compute noise bias and fourier weight estimates (if not included already)
        
        shot_sigma_sb = None
        if add_noise and photon_noise:
            # we will need the "sigma" map for the image several times
            shot_sigma_sb = self.compute_shot_sigma_map(inst, self.image, nfr=nfr, frame_rate=frame_rate)
        
        if apply_FW:
            if FW_image is None:
                verbprint(verbose, 'No Fourier weight image provided but apply_FW=True. Estimating Fourier weights from noise model..')

                FW_image, masked_noisecl2d = self.compute_FW_from_noise_sim(ifield, inst, mask=mask, apply_mask=apply_mask, noise_model=noise_model,\
                                               verbose=verbose, inplace=False, use_pyfftw = True,\
                                               nsims = n_FW_sims, n_split=n_FW_split, show=show, shot_sigma_sb=shot_sigma_sb)

                self.FW_image = FW_image
                if masked_noise_model is None:
                    masked_noise_model = masked_noisecl2d
            else:
                self.FW_image = FW_image
        verbprint(verbose, 'Computing noise power spectrum..')

        
        if noise_debias:
            
            self.compute_noise_power_spectrum(noise_Cl2D=masked_noise_model, apply_FW=apply_FW, weights=self.FW_image, verbose=False, inplace=True)
            print('N_l from masked noise model: ', self.N_ell)
        
        # for a number of splits, we want to:
        # 1) generate noise realizations
        # 2) add mock image to noise realizations
        # 3) compute Bl/Mkk/Nl-corrected power spectra 
        
        maplist_split_shape = (nsims//n_split, self.dimx, self.dimy)

        empty_aligned_objs, fft_objs = construct_pyfftw_objs(3, maplist_split_shape)
        
        all_cl_proc = np.zeros((nsims, self.n_ps_bin))
            
        verbprint(verbose, 'Computing processed power spectra..')
             
        for i in range(n_split):
            print('Split '+str(i+1)+' of '+str(n_split)+'..')
                        
            if add_noise:
                rnmaps, snmaps = self.noise_model_realization(inst, maplist_split_shape, noise_model, fft_obj=fft_objs[0],\
                                                      chisq=chisq, read_noise=read_noise, photon_noise=photon_noise, shot_sigma_sb=shot_sigma_sb)

            for j in range(nsims // n_split):
                
                simmap = self.image.copy()
                
                if add_noise:
                    if read_noise:
                        verbprint(verbose, 'Adding read noise..')

                        simmap += rnmaps[j]
                    if photon_noise:
                        verbprint(verbose, 'Adding photon noise..')

                        simmap += snmaps[j]

                #multiply simulated noise map by mask
                if apply_mask and mask is not None:
    
                    verbprint(verbose, 'Applying mask..')
                        
                    simmap *= mask

                if j == 0 and show:
                    plt.figure()
                    plt.title('sim map')
                    plt.imshow(simmap, cmap='Greys')
                    plt.colorbar()
                    plt.show()
                    
                    if apply_mask:
                        plt.figure()
                        plt.title('mask')
                        plt.imshow(mask, cmap='Greys')
                        plt.colorbar()
                        plt.show()
                
                lbins, cl_proc, _ = self.compute_processed_power_spectrum(inst, mask=mask, apply_mask=apply_mask, \
                                                                             image=simmap, convert_adufr_sb=False, \
                                                                            mkk_correct=mkk_correct, beam_correct=beam_correct, \
                                                                            apply_FW=apply_FW, verbose=verbose, noise_debias=noise_debias, \
                                                                         FF_correct=FF_correct, inv_Mkk=inv_Mkk)
                
                all_cl_proc[int(i*(nsims // n_split) + j), :] = cl_proc
                
                
        return_N_ell = None
       
        return all_cl_proc, lbins, return_N_ell

        
            
    def load_mkk_mats(self, ifield, inst, mkk_fpath=None, verbose=False, inplace=True):
        ''' Loads Mkk and inverse Mkk matrices '''
        verbprint(verbose, 'Loading Mkk and inverse Mkk matrix estimates from TM'+str(inst)+', field '+str(ifield)+'..')
         
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
        
        verbprint(verbose, 'Loading mask image from TM'+str(inst)+', field '+str(ifield))

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
    
        
    def load_dark_current_template(self, ifield, inst, dc_fpath=None, verbose=False, inplace=True):
        ''' Loads dark current template from data release '''
        
        verbprint(verbose, 'Loading dark current template from TM'+str(inst)+'..')

        if dc_fpath is None:
            dc_fpath = self.data_path + 'TM'+str(inst)+'/DCtemplate.fits'
        if inplace:
            self.dc_template = fits.open(dc_fpath)[0].data
        else:
            return fits.open(dc_fpath)[0].data
        
    def load_psf(self, ifield, inst, psf_fpath=None, verbose=False, inplace=True):
        ''' Loads PSF estimate from data release '''
        
        verbprint(verbose, 'Loading PSF template from TM'+str(inst)+', field '+str(ifield)+'..')

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
                           FW_image=False, FF_image=False, noise_Cl2D=False, verbose=True):
        
        if flight_image or load_all:
            self.load_flight_image(ifield, inst, verbose=verbose)
        if dark_current or load_all:
            self.load_dark_current_template(ifield, inst, verbose=verbose)
        if noise_Cl2D or load_all:
            self.load_noise_Cl2D(ifield, inst, verbose=verbose)
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
            
    def compute_shot_noise_maps(self, inst, image, nsims, shot_sigma_sb=None, nfr=None, frame_rate=None):

        if nsims*self.dimx*self.dimy > 1e8:
            print('WARNING -- using a lot of memory at once')
        if nsims*self.dimx*self.dimy > 1e9:
            print('probably too much memory, exiting')
            return None

        if shot_sigma_sb is None:

            shot_sigma_sb = self.compute_shot_sigma_map(inst, image, nfr=nfr, frame_rate=frame_rate)

        unit_noise = np.random.normal(0, 1, size=(nsims, self.dimx, self.dimy))

        return unit_noise*shot_sigma_sb

    def compute_shot_sigma_map(self, inst, image, nfr=None, frame_rate=None):
        # assume image is in surface brightness units, returns shot noise map in surface brightness units
        if nfr is None:
            nfr = self.nfr
        if frame_rate is None:
            frame_rate = self.frame_rate

        print('nfr, frame_rate:', nfr, frame_rate)
        flight_signal = image*self.g1_facs[inst]/self.cal_facs[inst]
        shot_sigma = np.sqrt((np.abs(flight_signal)/(nfr/frame_rate))*((nfr**2+1.)/(nfr**2 - 1.)))

        shot_sigma_sb = shot_sigma*self.cal_facs[inst]/self.g1_facs[inst]

        return shot_sigma_sb
        
    def compute_noise_power_spectrum(self, noise_Cl2D=None, apply_FW=True, weights=None, verbose=False, inplace=True):
        
        verbprint(verbose, 'Computing 1D noise power spectrum from 2D power spectrum..')
        
        if noise_Cl2D is None:
            verbprint(verbose, 'No Cl2D explicitly provided, using self.noise_Cl2D..')
                
            noise_Cl2D = self.noise_Cl2D.copy()
            
        if apply_FW and weights is None:
            weights = self.FW_image
        
        l2d = get_l2d(self.dimx, self.dimy, self.pixsize)
        
        lbins, Cl_noise, Clerr = azim_average_cl2d(noise_Cl2D*self.cal_facs[inst]**2, l2d, weights=weights, lbinedges=self.Mkk_obj.binl, lbins=self.Mkk_obj.midbin_ell)
        # Cl_noise has units of sb squared
        if inplace:
            self.N_ell = Cl_noise
            
            print("self.N_ell is ", self.N_ell)
            print('ell^2 N_ell = ', lbins**2*self.N_ell)
        else:
            return Cl_noise
        
        
    def compute_beam_correction(self, psf=None, verbose=False, inplace=True):
        
        verbprint(verbose, 'Computing beam correction..')

        if psf is None:
            verbprint(verbose, 'No template explicitly provided, using self.psf_template to compute Bl..')

            psf = self.psf_template.copy()
        
        rb, Bl, Bl_std = get_power_spec(psf-np.mean(psf), lbinedges=self.Mkk_obj.binl, lbins=self.Mkk_obj.midbin_ell, return_Dl=False)
        
        B_ell = np.sqrt(Bl)/np.max(np.sqrt(Bl))
        B_ell_std = (Bl_std/(2*np.sqrt(Bl)))/np.max(np.sqrt(Bl))
        
        if inplace:
            self.B_ell = B_ell
            self.B_ell_std = B_ell_std
        else:
            return B_ell, B_ell_std
        
  
    def compute_processed_power_spectrum(self, inst, mask=None, apply_mask=True, N_ell=None, B_ell=None, inv_Mkk=None,\
                                         image=None, mkk_correct=True, beam_correct=True,\
                                         FF_correct=True, apply_FW=True, noise_debias=True, verbose=True, \
                                        convert_adufr_sb=True):
        
        if image is None:
            image = self.image.copy()
        
        if convert_adufr_sb:
            image *= self.cal_facs[inst] # convert from ADU/fr to nW m^-2 sr^-1
        
        if FF_correct:
            verbprint(verbose, 'Applying flat field correction..')
            
            image /= self.FF_image # divide image by the flat field estimate
            image[np.isinf(image)] = 0.
            image[np.isnan(image)] = 0.
        
        if mask is None and apply_mask:
            verbprint(verbose, 'Getting mask from maskinst clean and strmask')
            mask = self.maskInst_clean*self.strmask # load mask
        
        if apply_mask:
            verbprint(verbose, 'Applying mask..')
            masked_image = mean_sub_masked_image(image, mask) # apply mask and mean subtract unmasked pixels
        else:
            verbprint(verbose, 'No masking, subtracting image by its mean..')

            masked_image = image - np.mean(image)
            mask = None
        
        verbprint(verbose, 'Mean of masked image is '+str(np.mean(masked_image)))
         
        weights=None
        if apply_FW:
            verbprint(verbose, 'Using Fourier weights..')
            
            weights = self.FW_image # this feeds Fourier weights into get_power_spec, which uses them with Cl2D
            
        lbins, cl_proc, cl_proc_err = get_power_spec(masked_image, mask=None, weights=weights, lbinedges=self.Mkk_obj.binl, lbins=self.Mkk_obj.midbin_ell)
        
        verbprint(verbose, 'cl_proc after get_power spec is ')
        verbprint(verbose, cl_proc)
            
        if noise_debias: # subtract noise bias
            verbprint(verbose, 'Applying noise bias..')

            if N_ell is None:
                cl_proc -= self.N_ell
            else:
                cl_proc -= N_ell     
                
            verbprint(verbose, 'after noise subtraction, ')
            verbprint(verbose, cl_proc)

  
        if mkk_correct: # apply inverse mode coupling matrix to masked power spectrum
            verbprint(verbose, 'Applying Mkk correction..')

            if inv_Mkk is None:
                cl_proc = np.dot(self.inv_Mkk.transpose(), cl_proc)
            else:
                cl_proc = np.dot(inv_Mkk.transpose(), cl_proc)
            verbprint(verbose, 'After Mkk correction, cl_proc is ')
            verbprint(verbose, cl_proc)
                            
        if beam_correct: # undo the effects of the PSF by dividing power spectrum by B_ell
            verbprint(verbose, 'Applying beam correction..')

            if B_ell is None:
                B_ell = self.B_ell
            assert len(B_ell)==len(cl_proc)
            cl_proc /= B_ell**2

        verbprint(verbose, 'Processed angular power spectrum is ')
        verbprint(verbose, cl_proc)
            
        return lbins, cl_proc, masked_image
            


def small_Nl2D_from_larger(dimx_small, dimy_small, n_ps_bin,ifield, noise_model=None, \
						   inst=1, dimx_large=1024, dimy_large=1024, nsims=100, \
						  div_fac = 2.83):
	
	cbps_large = CIBER_PS_pipeline(n_ps_bin=n_ps_bin, dimx=dimx_large, dimy=dimy_large)
	cbps_small = CIBER_PS_pipeline(n_ps_bin=n_ps_bin, dimx=dimx_small, dimy=dimy_small)
	
	cl2d_all_small = np.zeros((nsims, dimx_small, dimy_small))
	
	if noise_model is None:
		noise_model = cbps_large.load_noise_Cl2D(ifield, inst, inplace=False, transpose=False)
		
	
	clprocs = []
	clprocs_large_chi2 = []
	
	for i in range(nsims):
		
		if i%100==0:
			print('i = '+str(i))
		noise = np.random.normal(0, 1, (dimx_large, dimy_large)) + 1j*np.random.normal(0, 1, (dimx_large, dimy_large))
		
		chi2_realiz = (np.random.chisquare(2., size=(dimx_large, dimy_large))/div_fac)**2
		rnmap_large_chi2 = np.sqrt(dimx_large*dimy_large)*ifft2(noise*ifftshift(np.sqrt(chi2_realiz*noise_model))).real
		rnmap_large_chi2 *= cbps_large.cal_facs[inst]/cbps_large.arcsec_pp_to_radian

		lb, cl_proc_large_chi2, _ = get_power_spec(rnmap_large_chi2-np.mean(rnmap_large_chi2),\
												  mask=None, weights=None, lbinedges=cbps_large.Mkk_obj.binl,\
												  lbins=cbps_large.Mkk_obj.midbin_ell)

		clprocs_large_chi2.append(cl_proc_large_chi2)

		rnmap_small = rnmap_large_chi2[:cbps_small.dimx, :cbps_small.dimy]

		lbins, cl_proc, cl_proc_err = get_power_spec(rnmap_small-np.mean(rnmap_small), mask=None, weights=None,\
													 lbinedges=cbps_small.Mkk_obj.binl, lbins=cbps_small.Mkk_obj.midbin_ell)

		clprocs.append(cl_proc)
		l2d, cl2d = get_power_spectrum_2d(rnmap_small)
		cl2d_all_small[i,:,:] = cl2d/cbps_small.cal_facs[inst]**2

	av_cl2d = np.mean(cl2d_all_small, axis=0)


	return av_cl2d, clprocs, clprocs_large_chi2, cbps_small


def monte_carlo_test_vs_Jlim(ifield, inst, dimx, dimy, Jmags, model_image, mkk_fpaths, noise_fpath,\
                             apply_FW=True, psf_full=None, noise_debias=True, add_noise=True, apply_mask=True, mkk_correct=True, \
                             chisq=True, n_ps_bin=17, n_FW_sims=500, n_FW_split=5, nsims=500, n_split=5, stdpower=1, \
                            show=False):
    cbps = CIBER_PS_pipeline(n_ps_bin=n_ps_bin, dimx=dimx, dimy=dimy)
    cbps.load_data_products(ifield, inst, verbose=False, load_all=True)
    
    beam_correct = False
    if psf_full is not None:
        cbps.compute_beam_correction(verbose=True, psf=psf_full)
        beam_correct=True
        

    all_cl_proc_vs_Jlim, fourier_weight_list, mean_cl2ds, masked_N_ells = [], [], [], []
    
    noise_model = np.load(noise_fpath)['noise_model']
    clnoise = cbps.compute_noise_power_spectrum(noise_model, inplace=False, apply_FW=False)

    for jidx, Jmax in enumerate(Jmags):
        
        joint_mask = None
        if apply_mask:
            joint_mask = np.load(mkk_fpaths[jidx])['mask']
        
        
        shot_sigma_sb = cbps.compute_shot_sigma_map(inst, image=model_image)
        
        

        fourier_weights, mean_cl2d = cbps.compute_FW_from_noise_sim(nsims=n_FW_sims, n_split=n_FW_split, apply_mask=apply_mask, inst=inst,\
                                                       inplace=False, noise_model=noise_model, show=False,\
                                                       mask=joint_mask, use_pyfftw=True, shot_sigma_sb=shot_sigma_sb, \
                                                               stdpower=stdpower)
        
        
        fourier_weight_list.append(fourier_weights)
        mean_cl2ds.append(mean_cl2d)

        cbps.compute_noise_power_spectrum(mean_cl2d, inplace=True, apply_FW=apply_FW, weights=fourier_weights)

        masked_N_ells.append(cbps.N_ell)
        
        if show:
	        plt.figure(figsize=(8,6))
	        plt.plot(cbps.Mkk_obj.midbin_ell, clnoise, linestyle='dashed', color='g', label='Unmasked noise power spectrum')
	        plt.plot(cbps.Mkk_obj.midbin_ell, cbps.N_ell, linestyle='dashed', color='r', label='Masked, weighted noise power spectrum')
	        plt.xscale('log')
	        plt.yscale('log')
	        plt.legend()
	        plt.show()
        
        mkkmat, inv_Mkk = cbps.load_mkk_mats(ifield=None, inplace=False, inst=inst, verbose=True, mkk_fpath=mkk_fpaths[jidx])

        if show:
            plot_mkk_matrix(mkkmat, inverse=False, logscale=True)
            plot_mkk_matrix(inv_Mkk, inverse=True, symlogscale=True)

        all_cl_proc, lbins, N_ell = cbps.monte_carlo_power_spectrum_sim(ifield, inst, apply_mask=apply_mask, mask=joint_mask, mkk_correct=mkk_correct, noise_debias=noise_debias, \
                                                             beam_correct=beam_correct, apply_FW=apply_FW, FW_image=fourier_weights, noise_model=noise_model, \
                                                                        masked_noise_model=mean_cl2d, image=model_image, inv_Mkk=inv_Mkk, \
                                                                    verbose=False, nsims=nsims, n_split=n_split, add_noise=add_noise, chisq=chisq)
        all_cl_proc_vs_Jlim.append(all_cl_proc)

        
        
    return lbins, all_cl_proc_vs_Jlim, clnoise, fourier_weight_list, mean_cl2ds, masked_N_ells, cbps


def monte_carlo_test_vs_mask_params(ifield, inst, dimx, dimy, param_combo_list, model_image, mkk_fpaths, noise_fpath,\
                             mkk_single_fpath=None, apply_FW=True, psf_full=None, noise_debias=True, add_noise=True, apply_mask=True, mkk_correct=True, \
                             chisq=True, n_ps_bin=15, n_FW_sims=500, n_FW_split=5, nsims=500, n_split=5, stdpower=1, \
                            show=False):
    cbps = CIBER_PS_pipeline(n_ps_bin=n_ps_bin, dimx=dimx, dimy=dimy)
    cbps.load_data_products(ifield, inst, verbose=False, load_all=True)
    
    beam_correct = False
    if psf_full is not None:
        cbps.compute_beam_correction(verbose=True, psf=psf_full)
        beam_correct=True
        
    if mkk_single_fpath is not None:
        mkk_mats = np.load(mkk_single_fpath)['Mkk_list']
        inv_mkk_mats = np.load(mkk_single_fpath)['inv_Mkk_list']
        mask_list = np.load(mkk_single_fpath)['mask_list']
        
    all_cl_proc_vs_a1, fourier_weight_list, mean_cl2ds, masked_N_ells = [], [], [], []
    
    noise_model = np.load(noise_fpath)['noise_model']
    clnoise = cbps.compute_noise_power_spectrum(noise_model, inplace=False, apply_FW=False)

    for p, param_combo in enumerate(param_combo_list):
        
        joint_mask = None
        
        if mkk_single_fpath is not None:
            joint_mask = mask_list[p]
        else:
            if apply_mask:
                joint_mask = np.load(mkk_fpaths[p])['mask']
        
        plot_srcmap_mask(joint_mask, 'joint mask ', 3)
        
        if show:
            plt.figure(figsize=(10, 10))
            plt.imshow(image*joint_mask, vmin=-50, vmax=50, origin='lower')
            plt.colorbar()
            plt.show()

            plt.figure(figsize=(10, 10))
            plt.imshow(image, vmin=-50, vmax=50, origin='lower')
            plt.colorbar()
            plt.show()
        
        
        shot_sigma_sb = cbps.compute_shot_sigma_map(inst, image=model_image)
        
        

        fourier_weights, mean_cl2d = cbps.compute_FW_from_noise_sim(nsims=n_FW_sims, n_split=n_FW_split, apply_mask=apply_mask, inst=inst,\
                                                       inplace=False, noise_model=noise_model, show=False,\
                                                       mask=joint_mask, use_pyfftw=True, shot_sigma_sb=shot_sigma_sb, \
                                                               stdpower=stdpower)
        
        
        fourier_weight_list.append(fourier_weights)
        mean_cl2ds.append(mean_cl2d)

        cbps.compute_noise_power_spectrum(mean_cl2d, inplace=True, apply_FW=apply_FW, weights=fourier_weights)

        masked_N_ells.append(cbps.N_ell)
        
        if show:
            plt.figure(figsize=(8,6))
            plt.plot(cbps.Mkk_obj.midbin_ell, clnoise, linestyle='dashed', color='g', label='Unmasked noise power spectrum')
            plt.plot(cbps.Mkk_obj.midbin_ell, cbps.N_ell, linestyle='dashed', color='r', label='Masked, weighted noise power spectrum')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()
            plt.show()
        
        if mkk_single_fpath is not None:
            mkkmat = mkk_mats[p]
            inv_Mkk = inv_mkk_mats[p]
        else:
            mkkmat, inv_Mkk = cbps.load_mkk_mats(ifield=None, inplace=False, inst=inst, verbose=True, mkk_fpath=mkk_fpaths[jidx])

        if show:
            plot_mkk_matrix(mkkmat, inverse=False, logscale=True)
            plot_mkk_matrix(inv_Mkk, inverse=True, symlogscale=True)

        all_cl_proc, lbins, N_ell = cbps.monte_carlo_power_spectrum_sim(ifield, inst, apply_mask=apply_mask, mask=joint_mask, mkk_correct=mkk_correct, noise_debias=noise_debias, \
                                                             beam_correct=beam_correct, apply_FW=apply_FW, FW_image=fourier_weights, noise_model=noise_model, \
                                                                        masked_noise_model=mean_cl2d, image=model_image, inv_Mkk=inv_Mkk, \
                                                                    verbose=False, nsims=nsims, n_split=n_split, add_noise=add_noise, chisq=chisq)
        all_cl_proc_vs_a1.append(all_cl_proc)

        
        
    return lbins, all_cl_proc_vs_a1, clnoise, fourier_weight_list, mean_cl2ds, masked_N_ells, cbps






