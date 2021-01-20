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
        self.cal_facs = dict({1:-170.3608, 2:-57.2057}) # multiplying by cal_facs converts from ADU/fr to nW m^-2 sr^-1
        self.dimx, self.dimy = dimx, dimy
        self.n_ps_bin = n_ps_bin
        self.pixsize = 7. # arcseconds
        self.pixlength_in_deg = self.pixsize/3600. # pixel angle measured in degrees
        self.arcsec_pp_to_radian = self.pixlength_in_deg*(np.pi/180.) # conversion from arcsecond per pixel to radian per pixel

        self.photFactor = np.sqrt(1.2)
      
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

    
    
    def compute_FW_from_noise_sim(self, ifield, inst, mask=None, apply_mask=True, noise_model=None, noise_model_fpath=None, verbose=False, inplace=True, \
                                 use_pyfftw = True, nsims = 50, n_split=5, show=False):
        if verbose:
            print('Computing fourier weight image from noise sims for TM'+str(inst)+', field '+str(ifield)+'..')
            print('WARNING : This operation takes a lot of memory')
        
        if noise_model is None:
            if noise_model_fpath is None:
                noise_model_fpath = self.data_path + 'TM' + str(inst) + '/noise_model/noise_model_Cl2D'+str(ifield)+'.fits'
            noise_model = self.load_noise_Cl2D(ifield, inst, noise_fpath=noise_model_fpath, inplace=False, transpose=False)

            plt.figure()
            plt.imshow(noise_model, norm=matplotlib.colors.LogNorm())
            plt.colorbar()
            plt.show()
            
        if mask is None and apply_mask:
            if verbose:
                print('No mask provided but apply_mask=True, using union of self.strmask and self.maskInst_clean..')
            mask = self.strmask*self.maskInst_clean
            
        if show:
            plt.figure(figsize=(8,8))
            plt.title('Mask')
            plt.imshow(mask)
            plt.colorbar()
            plt.show()
        
        cl2d_all = np.zeros((nsims, self.dimx, self.dimy))
        
        if use_pyfftw:
            
            maplist_split_shape = (nsims//n_split, self.dimx, self.dimy)

            empty_aligned_objs, fft_objs = construct_pyfftw_objs(3, maplist_split_shape)
            
            for i in range(n_split):
                
                print('Split '+str(i+1)+' of '+str(n_split)+'..')
                simmaps = np.zeros(maplist_split_shape)
                
                self.noise = np.random.normal(0, 1, size=maplist_split_shape)+ 1j*np.random.normal(0, 1, size=maplist_split_shape)
                rnmaps = np.sqrt(self.dimx*self.dimy)*fft_objs[0](self.noise*ifftshift(np.sqrt(noise_model)))
                
                for j in range(nsims // n_split):
                    
#                     simmap = ifft2(noise_model*self.cal_facs[inst]**2*fft2(np.random.normal(0, 1, (self.dimx, self.dimy)))).real
                    simmap = self.cal_facs[inst]*rnmaps[j].real/self.arcsec_pp_to_radian                   
#                     simmap = rnmaps[j].real
#                     simmap += self.photFactor*phmap
                    # multiply simulated noise map by mask
                    if apply_mask and mask is not None:
                        simmap *= mask
                                                
                    simmaps[j, :, :] = simmap

                cl2d_all[i*nsims//n_split:(i+1)*nsims//n_split,:,:] = fft_objs[1](simmaps).real
        
        fw_std = np.std(cl2d_all, axis=0) # returns std within each 2d fourier mode
        
        fw_std = 1./fw_std    
            
        fw_std /= np.max(fw_std)
        
        mean_cl2d = np.mean(cl2d_all, axis=0)
        
        if show:
            plt.figure(figsize=(8,8))
            plt.title('Fourier weights')
            plt.imshow(fw_std, origin='lower', cmap='Greys', norm=matplotlib.colors.LogNorm())
            plt.xticks([], [])
            plt.yticks([], [])
            plt.colorbar()
            plt.show()
        
        if inplace:
            self.FW_image = fw_std
            return mean_cl2d
        else:
            return fw_std, mean_cl2d
        
    def load_flight_image(self, ifield, inst, flight_fpath=None, verbose=False, inplace=True):
        ''' Loads flight image from data release '''
        if verbose:
            print('Loading flight image from TM'+str(inst)+', field '+str(ifield)+'..')
        if flight_fpath is None:
            flight_fpath = self.data_path + 'TM'+str(inst)+'/flight/field'+str(ifield)+'_flight.fits'
        
        if inplace:
            self.image = fits.open(flight_fpath)[0].data
        else:
            return fits.open(flight_fpath)[0].data
        
    def load_noise_Cl2D(self, ifield, inst, noise_fpath=None, verbose=False, inplace=True, transpose=True, mode=None):
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
        
        if verbose:
            print('Loading 2D noise power spectrum from TM'+str(inst)+', field '+str(ifield)+'..')
            
        if noise_fpath is None:
            noise_fpath = self.data_path + 'TM'+str(inst)+'/noiseCl2D/field'+str(ifield)+'_noiseCl2D.fits'
        
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
        
        
    def monte_carlo_power_spectrum_sim(self, ifield, inst, image=None, mask=None, apply_mask=True, noise_model=None, noise_model_fpath=None,\
                                       FW_image=None, apply_FW=True, n_FW_sims=100, n_FW_split=5, verbose=False, use_pyfftw = True, \
                                       nsims = 100, n_split=5, show=False, mkk_fpath=None, mkk_correct=True, beam_correct=True, noise_debias=True, \
                                      FF_correct=False):
        
        if image is not None:
            self.image = image
            
        # load noise model
        if noise_model is None:
            
            if noise_model_fpath is None:
                noise_model_fpath = self.data_path + 'TM' + str(inst) + '/noise_model/noise_model_Cl2D'+str(ifield)+'.fits'
            
            if verbose:
                print('No noise model provided, loading from '+str(noise_model_fpath)+'..')
            noise_model = self.load_noise_Cl2D(ifield, inst, noise_fpath=noise_model_fpath, inplace=False, transpose=False)

        if mask is None and apply_mask:
            if verbose:
                print('No mask provided but apply_mask=True, using union of self.strmask and self.maskInst_clean..')
            mask = self.strmask*self.maskInst_clean
        
        if show:
            plt.figure()
            plt.title('Noise model [nW$^2$ m$^{-4}$ sr$^{-2}$]')
            plt.imshow(noise_model*self.cal_facs[inst]**2, norm=matplotlib.colors.LogNorm(), vmax=np.percentile(noise_model*self.cal_facs[inst]**2, 95), vmin=np.percentile(noise_model*self.cal_facs[inst]**2, 5))
            plt.colorbar()
            plt.show()
               
        if mkk_correct and mkk_fpath is not None:   
            if verbose:
                print('Loading Mkk matrices from '+mkk_fpath+'..')
            self.load_mkk_mats(ifield, inst, mkk_fpath=mkk_fpath, verbose=verbose, inplace=True)

        # compute noise bias and fourier weight estimates (if not included already)
        
        if FW_image is None and apply_FW:
            if verbose:
                print('No Fourier weight image provided but apply_FW=True. Estimating Fourier weights from noise model..')
            masked_noisecl2d = self.compute_FW_from_noise_sim(ifield, inst, mask=mask, apply_mask=apply_mask, noise_model=noise_model,\
                                           verbose=verbose, inplace=True, use_pyfftw = True,\
                                           nsims = n_FW_sims, n_split=n_FW_split, show=show)

        if verbose:
            print('Computing noise power spectrum..')
        
        self.compute_noise_power_spectrum(noise_Cl2D=noise_model, apply_FW=apply_FW, weights=None, verbose=False, inplace=True)

        
        # for a number of splits, we want to:
        # 1) generate noise realizations
        # 2) add mock image to noise realizations
        # 3) compute Bl/Mkk/Nl-corrected power spectra 
        
        maplist_split_shape = (nsims//n_split, self.dimx, self.dimy)

        empty_aligned_objs, fft_objs = construct_pyfftw_objs(3, maplist_split_shape)
        
        all_cl_proc = np.zeros((nsims, self.n_ps_bin))
            
        if verbose:
            print('Computing processed power spectra..')
        for i in range(n_split):

            print('Split '+str(i+1)+' of '+str(n_split)+'..')
            
            simmaps = np.zeros(maplist_split_shape)
            self.noise = np.random.normal(0, 1, size=maplist_split_shape)+ 1j*np.random.normal(0, 1, size=maplist_split_shape)

            rnmaps = np.sqrt(self.dimx*self.dimy)*fft_objs[0](self.noise*ifftshift(np.sqrt(noise_model)))

            for j in range(nsims // n_split):
                
#                 fftgrand = ifftshift(np.sqrt(noise_model))*fft2(np.random.normal(0., 1., (self.dimx, self.dimy)))
#                 simmap = ifft2(fftgrand)
                
                simmap = self.cal_facs[inst]*rnmaps[j].real/self.arcsec_pp_to_radian                   
                simmap += self.image
                
                #multiply simulated noise map by mask
                if apply_mask and mask is not None:
                    simmap *= mask

                if j == 0:
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
                
                simmaps[j, :, :] = simmap

                lbins, cl_proc, _ = self.compute_processed_power_spectrum(inst, mask=mask, apply_mask=apply_mask, \
                                                                             image=simmap, convert_adufr_sb=False, \
                                                                            mkk_correct=mkk_correct, beam_correct=beam_correct, \
                                                                            apply_FW=apply_FW, verbose=False, noise_debias=noise_debias, \
                                                                         FF_correct=FF_correct)
                
                all_cl_proc[int(i*(nsims // n_split) + j), :] = cl_proc
                
                
        return all_cl_proc, lbins, self.N_ell

        
            
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
        ''' Computes Mkk matrix and its inverse for a given mask '''
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
            
            
    def compute_noise_power_spectrum(self, noise_Cl2D=None, apply_FW=True, weights=None, verbose=False, inplace=True):
        
        if verbose:
            print('Computing 1D noise power spectrum from 2D power spectrum..')
        
        if noise_Cl2D is None:
            if verbose:
                print('No Cl2D explicitly provided, using self.noise_Cl2D..')
                
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
        
        if verbose:
            print('Computing beam correction..')
        if psf is None:
            if verbose:
                print('No template explicitly provided, using self.psf_template to compute Bl..')
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
            if verbose:
                print('Applying flat field correction..')
            
            image /= self.FF_image # divide image by the flat field estimate
            image[np.isinf(image)] = 0.
            image[np.isnan(image)] = 0.
            
        if mask is None and apply_mask:
            print('getting mask from maskinst clean and strmask')
            mask = self.maskInst_clean*self.strmask # load mask
        
        if apply_mask:
            masked_image = mean_sub_masked_image(image, mask) # apply mask and mean subtract unmasked pixels
        else:
            masked_image = image - np.mean(image)
        
        if verbose:
            print('Mean of masked image is ', np.mean(masked_image))
        
        weights=None
        if apply_FW:
            if verbose:
                print('Using Fourier weights')
            weights = self.FW_image # this feeds Fourier weights into get_power_spec, which uses them with Cl2D
            
        lbins, cl_proc, cl_proc_err = get_power_spec(masked_image, mask=mask, weights=weights, lbinedges=self.Mkk_obj.binl, lbins=self.Mkk_obj.midbin_ell)
        
        if verbose:
            print('cl_proc after get_power spec is ', cl_proc)
            
        if noise_debias: # subtract noise bias
            if verbose:
                print('Applying noise bias..')
            if N_ell is None:
                cl_proc -= self.N_ell
            else:
                cl_proc -= N_ell                                
                
        if mkk_correct: # apply inverse mode coupling matrix to masked power spectrum
            if verbose:
                print('Applying Mkk correction..')
            if inv_Mkk is None:
                cl_proc = np.dot(self.inv_Mkk.transpose(), cl_proc)
            else:
                cl_proc = np.dot(inv_Mkk.transpose(), cl_proc)
            
        if beam_correct: # undo the effects of the PSF by dividing power spectrum by B_ell
            if verbose:
                print('Applying beam correction..')
            if B_ell is None:
                B_ell = self.B_ell
            assert len(B_ell)==len(cl_proc)
            cl_proc /= B_ell

        if verbose:
            print('Processed angular power spectrum is ', cl_proc)
            
        return lbins, cl_proc, masked_image
            





