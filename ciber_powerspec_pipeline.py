import numpy as np
import matplotlib 
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
from flat_field_est import *
from masking_utils import *
from powerspec_utils import *
from filtering_utils import calculate_plane, fit_gradient_to_map
# from ciber_noise_data_utils import iter_sigma_clip_mask, sigma_clip_maskonly


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

def compute_fourier_weights(cl2d_all, stdpower=2):
    
    fw_std = np.std(cl2d_all, axis=0)
    fw_std[fw_std==0] = np.inf
    fourier_weights = 1./(fw_std**stdpower)
    print('max of fourier weights is ', np.max(fourier_weights))
    fourier_weights /= np.max(fourier_weights)
    
    mean_cl2d = np.mean(cl2d_all, axis=0)

    # median_cl2d = np.median(cl2d_all, axis=0)
    
    return mean_cl2d, fourier_weights


def iterative_gradient_ff_solve(orig_images, niter=3, masks=None, means=None, weights_ff=None, plot=False, ff_stack_min=1):
    
    # maps at end of each iteration
    images = np.array(list(orig_images.copy()))
    nfields = len(images)
    all_coeffs = np.zeros((niter, nfields, 3))    

    final_planes = np.zeros_like(images)

    add_maskfrac_stack = []

    #  --------------- masked images ------------------
    if masks is not None:
        print('Using masks to make masked images, FF stack masks with ff_stack_min = '+str(ff_stack_min)+'..')
        
        stack_masks = np.zeros_like(images)
        
        for imidx, image in enumerate(images):
            sum_mask = np.sum(masks[(np.arange(len(masks))!=imidx),:], axis=0)
            stack_masks[imidx] = (sum_mask >= ff_stack_min)
            mfrac = float(np.sum(masks[imidx]))/float(masks[imidx].shape[0]**2)

            add_maskfrac_stack.append(mfrac)
            masks[imidx] *= stack_masks[imidx]

            mfrac = float(np.sum(masks[imidx]))/float(masks[imidx].shape[0]**2)
            add_maskfrac_stack[imidx] -= mfrac

#             if plot:
#                 plot_map(sum_mask, title='sum map imidx = '+str(imidx))
#                 plot_map(stack_masks[imidx], title='stack masks imidx = '+str(imidx))
#                 plot_map(stack_masks[imidx]*masks[imidx], title='stack masks x mask imidx = '+str(imidx))

        masked_images = np.array([np.ma.array(images[i], mask=(masks[i]==0)) for i in range(len(images))])
        
    print('add maskfrac stack is ', add_maskfrac_stack)
    if means is None:
        if masks is not None:
            means = [np.ma.mean(im) for im in masked_images]
        else:
            means = [np.mean(im) for im in images]
                
    ff_estimates = np.zeros_like(images)

    for n in range(niter):
        print('n = ', n)
        
        # make copy to compute corrected versions of
        running_images = images.copy()
        
        for imidx, image in enumerate(images):

            stack_obs = list(images.copy())                
            del(stack_obs[imidx])

            weights_ff_iter = None
        
            if weights_ff is not None:
                weights_ff_iter = weights_ff[(np.arange(len(masks)) != imidx)]
            
            ff_estimate, _, ff_weights = compute_stack_ff_estimate(stack_obs, masks=masks[(np.arange(len(masks))!=imidx),:], weights=weights_ff_iter)

            ff_estimates[imidx] = ff_estimate
            running_images[imidx] /= ff_estimate
            theta, plane = fit_gradient_to_map(running_images[imidx], mask=masks[imidx])
            running_images[imidx] -= (plane-np.mean(plane))
            
            all_coeffs[n, imidx] = theta[:,0]

            if plot:
                plot_map(plane, title='best fit plane imidx = '+str(imidx))
                plot_map(ff_estimate, title='ff_estimate imidx = '+str(imidx))
    
            if n<niter-1:
                running_images[imidx] *= ff_estimate
            else:
                final_planes[imidx] = plane
        
        # print('dt:', time.time()-t0)
        images = running_images.copy()

    images[np.isnan(images)] = 0.
    ff_estimates[np.isnan(ff_estimates)] = 1.
    ff_estimates[ff_estimates==0] = 1.
    
    return images, ff_estimates, final_planes, stack_masks, all_coeffs


    
class CIBER_PS_pipeline():

    # fixed quantities for CIBER
    photFactor = np.sqrt(1.2)
    pixsize = 7.
    Npix = 2.03
    inst_to_band = dict({1:'J', 2:'H'})
    ciber_field_dict = dict({4:'elat10', 5:'elat30',6:'BootesB', 7:'BootesA', 8:'SWIRE'})
    field_nfrs = dict({4:24, 5:10, 6:29, 7:28, 8:25}) # unique to fourth flight CIBER dataset, elat30 previously noted as 9 frames but Chi says it is 10 (9/17/21)
    frame_period = 1.78 # seconds

    # zl_levels_ciber_fields = dict({2:dict({'elat10': 199.16884143344222, 'BootesA': 106.53451615117534, 'elat30': 147.02015318942148, 'BootesB': 108.62357310134063, 'SWIRE': 90.86593718752026}), \
    #                               1:dict({'NEP':249., 'Lockman':435., 'elat10':558., 'elat30':402., 'BootesB':301., 'BootesA':295., 'SWIRE':245.})})
    zl_levels_ciber_fields = dict({2:dict({'elat10': 199.16884143344222, 'BootesA': 106.53451615117534, 'elat30': 147.02015318942148, 'BootesB': 108.62357310134063, 'SWIRE': 90.86593718752026}), \
                                  1:dict({'NEP':249., 'Lockman':435., 'elat10':515.7, 'elat30':381.4, 'BootesB':328.2, 'BootesA':318.15, 'SWIRE':281.4})})


    # self.g1_facs = dict({1:-1.5459, 2:-1.3181}) # multiplying by g1 converts from ADU/fr to e-/s NOTE: these are factory values but not correct
    # these are the latest and greatest calibration factors as of 9/20/2021
    # self.cal_facs = dict({1:-311.35, 2:-121.09}) # ytc values
    # self.cal_facs = dict({1:-170.3608, 2:-57.2057}) # multiplying by cal_facs converts from ADU/fr to nW m^-2 sr^-1

    def __init__(self, \
                base_path='/Users/luminatech/Documents/ciber2/ciber/', \
                data_path=None, \
                dimx=1024, \
                dimy=1024, \
                n_ps_bin=25):
        
        for attr, valu in locals().items():
            setattr(self, attr, valu)


        if data_path is None:
            self.data_path = self.base_path+'data/fluctuation_data/'
            
        self.Mkk_obj = Mkk_bare(dimx=dimx, dimy=dimy, ell_min=180.*(1024./dimx), nbins=n_ps_bin)
        self.Mkk_obj.precompute_mkk_quantities(precompute_all=True)
        self.B_ell = None
        
        self.g2_facs = dict({1:110./self.Npix, 2:44.2/self.Npix}) # divide by Npix now for correct convention when comparing mean surface brightness (1/31/22)
        self.g1_facs = dict({1:-2.67, 2:-3.04}) # these are updated values derived from flight data, similar to those esimated with focus data

        self.cal_facs = dict({1:self.g1_facs[1]*self.g2_facs[1], 2:self.g1_facs[2]*self.g2_facs[2]})
        self.pixlength_in_deg = self.pixsize/3600. # pixel angle measured in degrees
        self.arcsec_pp_to_radian = self.pixlength_in_deg*(np.pi/180.) # conversion from arcsecond per pixel to radian per pixel
        self.Mkk_obj.delta_ell = self.Mkk_obj.binl[1:]-self.Mkk_obj.binl[:-1]
        self.fsky = self.dimx*self.dimy*self.Mkk_obj.arcsec_pp_to_radian**2 / (4*np.pi)

        self.frame_rate = 1./self.frame_period
        self.powerspec_dat_path = self.base_path+'data/powerspec_dat/'

      
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
        ''' 
        Computes Mkk matrix and its inverse for a given mask.

        Parameters
        ----------

        mask : `~np.array~` of type `float` or `int`, shape (cbps.n_ps_bins, cbps.n_ps_bins).
        nsims : `int`.
            Default is 100.
        n_split : `int`.
            Default is 2. 
        inplace : `bool`. 
            Default is True.

        Returns
        -------

        Mkk_matrix : `~np.array~` of type `float`, shape (cbps.n_ps_bins, cbps.n_ps_bins).
        inv_Mkk : `~np.array~` of type `float`, shape (cbps.n_ps_bins, cbps.n_ps_bins).

        '''
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
    
    def noise_model_realization(self, inst, maplist_split_shape, noise_model=None, fft_obj=None, adu_to_sb=True, chisq=True, div_fac=2.83, \
                               read_noise=True, photon_noise=True, shot_sigma_sb=None, image=None, nfr=None, frame_rate=None):
        
        '''
        Generate noise model realization given some noise model.

        Parameters
        ----------

        inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band.
        

        Returns
        -------


        '''
        self.noise = np.random.normal(0, 1, size=maplist_split_shape)+ 1j*np.random.normal(0, 1, size=maplist_split_shape)

        rnmaps = None
        snmaps = None
        
        if read_noise:

            # print('min/max of readnoise model in noise_model_realization ', np.min(noise_model), np.max(noise_model))
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
                    assert len(self.noise.shape) == 2
                    assert len(chi2_realiz.shape) == 2
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
            snmaps = self.compute_shot_noise_maps(inst, image, nsims, shot_sigma_map=shot_sigma_sb, nfr=nfr) # if not providing shot_sigma_sb, need  to provide nfr/frame_rate

        if adu_to_sb and read_noise:
            rnmaps *= self.cal_facs[inst]/self.arcsec_pp_to_radian

        if len(maplist_split_shape)==2:
            if photon_noise:
                snmaps = snmaps[0]

        return rnmaps, snmaps


    def estimate_noise_power_spectrum(self, inst=None, ifield=None, field_nfr=None,  mask=None, apply_mask=True, noise_model=None, noise_model_fpath=None, verbose=False, inplace=True, \
                                 nsims = 50, n_split=5, simmap_dc = None, show=False, read_noise=True, photon_noise=True, shot_sigma_sb=None, image=None,\
                                  ff_estimate=None, transpose_noise=False, ff_truth=None, mc_ff_estimates = None, gradient_filter=False):

        ''' 
        This function generates realizations of the CIBER read + photon noise model and applies the relevant observational effects that are needed to 
        obtain a minimum variance estimator of the noise power spectrum. 

        This is an updated version of the function self.compute_FW_from_noise_sim(), and makes default the pyfftw package for efficient power spectrum computation. 

            
        Parameters
        ----------

        ifield : `int`. Index of CIBER field.
            Default is None.
        inst (optional): `int`. 1 for 1.1 um band, 2 for 1.8 um band.
        field_nfr (optional, default=None) : 'int'.
        mask (optional, default=None) : `np.array' of type 'float' or 'int' and shape (self.dimx, self.dimy).
        apply_mask (default=True) : 'bool'.
        noise_model (optional, default=None) : `np.array' of type 'float'.
        noise_model_fpath (optional, default=None) : 'str'. 
        verbose (default=False) : 'bool'.



        Returns
        -------

        mean_nl2d : 
        fourier_weights : 

        '''

        verbprint(verbose, 'Computing fourier weight image from noise sims for TM'+str(inst)+', field '+str(ifield)+'..')
        # verbprint(verbose, 'WARNING : This operation can take a lot of memory if caution is thrown to the wind')
    

        if image is None and photon_noise and shot_sigma_sb is None:
            print('Photon noise is set to True, but no shot noise map provided. Setting image to self.image for use in estimating shot_sigma_sb..')
            image = self.image

        if field_nfr is None:
            if self.field_nfrs is not None and ifield is not None:
                print('Using field nfr, ifield='+str(ifield))
                field_nfr = self.field_nfrs[ifield]

            else:
                print('field nfr = None')
            # else:
                # print('Setting field_nfr = self.nfr = '+str(self.nfr))
                # field_nfr = self.nfr

        verbprint(verbose, 'Field nfr used here for ifield '+str(ifield)+', inst '+str(inst)+' is '+str(field_nfr))

        if noise_model is None and read_noise:
            if noise_model_fpath is None:
                if ifield is not None and inst is not None:
                    noise_model_fpath = self.data_path + 'TM' + str(inst) + '/noise_model/noise_model_Cl2D'+str(ifield)+'.fits'
                else:
                    print('Noise model not provided, noise_model_fpath not provided, and ifield/inst not provided, need more information..')
                    return
            # WEIRDDDDDDDDD fix this?
            noise_model = self.load_noise_Cl2D(noise_fpath='data/fluctuation_data/TM'+str(inst)+'/noiseCl2D/field'+str(ifield)+'_noiseCl2D_110421.fits', inplace=False, transpose=transpose_noise)

        if mask is None and apply_mask:
            verbprint(verbose, 'No mask provided but apply_mask=True, using union of self.strmask and self.maskInst_clean..')
            mask = self.strmask*self.maskInst

        # compute the pixel-wise shot noise estimate given the image (which gets converted to e-/s) and the number of frames used in the integration
        if photon_noise and shot_sigma_sb is None:
            print('photon noise set to True but no shot noise sigma map provided')
            if image is not None:
                print('getting sigma map from input image')
                shot_sigma_sb = self.compute_shot_sigma_map(inst, image=image, nfr=field_nfr)
            

        # allocate memory for Welfords online algorithm computing mean and variance
        # now we want to allocate the FFT objects in memory so that we can quickly compute the Fourier transforms for many sims at once.
        # first direction (fft_objs[0]->fft_objs[1]) is inverse DFT, second (fft_objs[1]->fft_objs[2]) is regular DFT

        maplist_split_shape = (nsims//n_split, self.dimx, self.dimy)
        empty_aligned_objs, fft_objs = construct_pyfftw_objs(3, maplist_split_shape)
        simmaps = np.zeros(maplist_split_shape)

        sterad_per_pix = (self.pixsize/3600/180*np.pi)**2
        V = self.dimx*self.dimy*sterad_per_pix
        count = 0

        mean_nl2d = np.zeros((self.dimx, self.dimy))
        M2_nl2d = np.zeros((self.dimx, self.dimy))

        for i in range(n_split):
            print('Split '+str(i+1)+' of '+str(n_split)+'..')

            if photon_noise or read_noise:
                rnmaps, snmaps = self.noise_model_realization(inst, maplist_split_shape, noise_model, fft_obj=fft_objs[0],\
                                                      read_noise=read_noise, photon_noise=photon_noise, shot_sigma_sb=shot_sigma_sb)


            simmaps = np.zeros(maplist_split_shape)
            print('simmaps has shape', simmaps.shape)


            if simmap_dc is not None:
                print('adding constant sky background..')
                simmaps += simmap_dc

            if photon_noise:
                print('adding photon noise')
                simmaps += snmaps 

            if ff_truth is not None:
                print('multiplying simmaps by ff_truth..')
                simmaps *= ff_truth

            if read_noise:
                print('adding read noise..')
                simmaps += rnmaps

            if mc_ff_estimates is not None:
                print('using MC flat field estimate')

                simmaps /= mc_ff_estimates[i]

            elif ff_estimate is not None:

                if i==0:
                    print('std on simmaps before ff estimate is ', np.std(simmaps))
                    print('mean, std on ff_estimate are ', np.mean(ff_estimate), np.std(ff_estimate))
                simmaps /= ff_estimate

                if i==0:
                    print('std on simmaps after ff estimate is ', np.std(simmaps))


            if gradient_filter:
                verbprint(True, 'Gradient filtering image in the noiiiise bias..')
                # print('mask has shape ', mask.shape)

                for s in range(len(simmaps)):

                    theta, plane = fit_gradient_to_map(simmaps[s], mask=mask)
                    simmaps[s] -= plane

                print([np.mean(simmap) for simmap in simmaps])

            if apply_mask and mask is not None:
                simmaps *= mask
                unmasked_means = [np.mean(simmap[mask==1]) for simmap in simmaps]
                simmaps -= np.array([mask*unmasked_mean for unmasked_mean in unmasked_means])
                print('simmaps have means : ', [np.mean(simmap) for simmap in simmaps])
            else:
                simmaps -= np.array([np.full((self.dimx, self.dimy), np.mean(simmap)) for simmap in simmaps])
                # print('simmaps have means : ', [np.mean(simmap) for simmap in simmaps])

            fft_objs[1](simmaps*sterad_per_pix)

            cl2ds = np.array([fftshift(dentry*np.conj(dentry)).real for dentry in empty_aligned_objs[2]])

            count, mean_nl2d, M2_nl2d = update_meanvar(count, mean_nl2d, M2_nl2d, cl2ds/V/self.cal_facs[inst]**2)

        mean_nl2d, var_nl2d, svar_nl2d = finalize_meanvar(count, mean_nl2d, M2_nl2d)

        if show:
            plot_map(mean_nl2d, title='mean Nl2d')
            plot_map(1./var_nl2d, title='inverse variance of Nl2d')
            plot_map(mean_nl2d/var_nl2d, title='Fourier-weighted Nl2d')


        fourier_weights = 1./var_nl2d
        
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
            return mean_nl2d
        else:
            return fourier_weights, mean_nl2d

       
    def load_flight_image(self, ifield, inst, flight_fpath=None, verbose=False, inplace=True):
        ''' 
        Loads flight image from data release.

        Parameters
        ----------
        ifield : `int`. Index of CIBER field.
        inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band.

        '''
        verbprint(verbose, 'Loading flight image from TM'+str(inst)+', field '+str(ifield)+'..')

        if flight_fpath is None:
            flight_fpath = self.data_path + 'TM'+str(inst)+'/flight/field'+str(ifield)+'_flight.fits'
        
        if inplace:
            self.image = fits.open(flight_fpath)[0].data
        else:
            return fits.open(flight_fpath)[0].data

        
    def load_noise_Cl2D(self, ifield=None, inst=None, noise_model=None, noise_fpath=None, verbose=False, inplace=True, transpose=False, mode=None, use_abs=False):
        
        ''' Loads 2D noise power spectrum from data release
        
        Parameters
        ----------

        ifield : `int`. Index of CIBER field.
        inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band.
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
                    noise_fpath = self.data_path+'/TM'+str(inst)+'/noiseCl2D/field'+str(ifield)+'_noiseCl2D_110421.fits'                    
                else:
                    print('Out of luck, need more information. Returning None')
                    return None

            noise_Cl2D = fits.open(noise_fpath)['noise_model_'+str(ifield)].data

            if transpose:
                print('using the transpose here (why)')
                noise_Cl2D = noise_Cl2D.transpose()


        # remove any NaNs/infs from power spectrum
        noise_Cl2D[np.isnan(noise_Cl2D)] = 0.
        noise_Cl2D[np.isinf(noise_Cl2D)] = 0.

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
        ''' 
        Loads flat field estimate from data release 
        
        Parameters
        ----------

        ifield : `int`. Index of CIBER field.
        inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band.
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
        ''' 
        Loads Fourier weight image estimate from data release.

        Parameters
        ----------
        ifield : `int`. Index of CIBER field.
        inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band. 


        '''
        verbprint(verbose, 'Loading fourier weight image estimate from TM'+str(inst)+', field '+str(ifield)+'..')

        if fw_fpath is None:
            fw_fpath = self.data_path + 'TM'+str(inst)+'/FW/field'+str(ifield)+'_FW.fits'
        
        if inplace:
            self.FW_image = fits.open(fw_fpath)[0].data
        else:
            return fits.open(fw_fpath)[0].data
           
            
    def load_mkk_mats(self, ifield, inst, mkk_fpath=None, verbose=False, inplace=True):
        ''' 
        Loads Mkk and inverse Mkk matrices.

        Parameters
        ----------
        ifield : `int`. Index of CIBER field.
        inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band.

        '''
        verbprint(verbose, 'Loading Mkk and inverse Mkk matrix estimates from TM'+str(inst)+', field '+str(ifield)+'..')
         
        if mkk_fpath is None:
            mkk_fpath = self.data_path + 'TM'+str(inst)+'/mkk/field'+str(ifield)+'_mkk_mats.npz'
        
        mkkfile = np.load(mkk_fpath)
        if inplace:
            self.Mkk_matrix = mkkfile['Mkk']
            self.inv_Mkk = mkkfile['inv_Mkk']
        else:
            return mkkfile['Mkk'], mkkfile['inv_Mkk']
        
    def load_mask(self, ifield, inst, masktype=None, mask_fpath=None, verbose=False, inplace=True):
        ''' 
        Loads mask.

        Parameters
        ----------
        ifield : `int`. 
            Index of CIBER field.
        inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band.
        masktype : 'str'. 
            Type of mask to load. 

        '''
        
        verbprint(verbose, 'Loading mask image from TM'+str(inst)+', field '+str(ifield))

        if mask_fpath is None:
            mask_fpath = self.data_path + 'TM'+str(inst)+'/mask/field'+str(ifield)+'_'+masktype+'.fits'
        
        if inplace:
            if masktype=='maskInst_clean':
                self.maskInst_clean = fits.open(mask_fpath)[0].data 
            elif masktype=='maskInst':
                self.maskInst = fits.open(mask_fpath)[0].data
            elif masktype=='strmask':
                self.strmask = fits.open(mask_fpath)[0].data
            elif masktype=='bigmask':
                self.bigmask = fits.open(mask_fpath)[0].data
        else:
            return fits.open(mask_fpath)[0].data
    
        
    def load_dark_current_template(self, ifield, inst, dc_fpath=None, verbose=False, inplace=True):
        ''' 
        Loads dark current template from data release.
        
        Parameters
        ----------
        ifield : `int`. Index of CIBER field.
        inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band.
        
        '''
        
        verbprint(verbose, 'Loading dark current template from TM'+str(inst)+'..')

        if dc_fpath is None:
            dc_fpath = self.data_path + 'TM'+str(inst)+'/DCtemplate.fits'
        if inplace:
            self.dc_template = fits.open(dc_fpath)[0].data
        else:
            return fits.open(dc_fpath)[0].data
        
    def load_psf(self, ifield, inst, psf_fpath=None, verbose=False, inplace=True):
        ''' 
        Loads PSF estimate from data release.

        Parameters
        ----------
        ifield : `int`. Index of CIBER field.
        inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band.

        '''
        
        verbprint(verbose, 'Loading PSF template from TM'+str(inst)+', field '+str(ifield)+'..')

        if psf_fpath is None:
            psf_fpath = self.data_path + 'TM'+str(inst)+'/beam_effect/field'+str(ifield)+'_psf.fits'
        
        # place PSF in CIBER sized image array, some measured PSFs are evaluated over smaller region 
        psf = fits.open(psf_fpath)[0].data
        psf_template = np.zeros(shape=(self.dimx, self.dimy))
        psf_template[:psf.shape[0], :psf.shape[1]] = psf
        
        if inplace:
            self.psf_template = psf_template
        else:
            return psf_template


    def load_bl(self, ifield, inst, inplace=True, verbose=False):

        ''' 
        Loads beam transfer function. 

        Parameters
        ----------
        ifield : `int`. Index of CIBER field.
        inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band.

        '''

        verbprint(verbose, 'Loading Bl estimates for TM'+str(inst)+', field '+str(ifield)+'..')

        B_ell_path = self.powerspec_dat_path+'B_ells/beam_correction_ifield'+str(ifield)+'_inst'+str(inst)+'.npz'
        
        beamdat = np.load(B_ell_path)

        # make sure multipole bins in saved file match those desired
        if not all(self.Mkk_obj.midbin_ell==beamdat['lb']):
            print('loaded multipole bin centers do not match bins required')
            print(beamdat['lb'])
            print(self.Mkk_obj.midbin_ell)
            if not inplace:
                return None
        else:
            B_ell = beamdat['mean_bl']
        
            if inplace:
                self.B_ell = B_ell
                # print('self B_ell is ', self.B_ell)
            else:
                return B_ell
        
    def load_data_products(self, ifield, inst, load_all=True, flight_image=False, dark_current=False, psf=False, \
                           maskInst=False, strmask=False, mkk_mats=False, mkk_fpath=None, \
                           FW_image=False, FF_image=False, noise_Cl2D=False, beam_correction=False, verbose=True, ifield_psf=None, transpose=False):
        
        '''
        Parameters
        ----------
        ifield : `int`. Index of CIBER field.
        inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band.

        '''

        if flight_image or load_all:
            self.load_flight_image(ifield, inst, verbose=verbose)
        if dark_current or load_all:
            self.load_dark_current_template(ifield, inst, verbose=verbose)
        if noise_Cl2D or load_all:
            self.load_noise_Cl2D(ifield, inst, verbose=verbose, transpose=transpose)
        if beam_correction or load_all:
            self.load_bl(ifield, inst, verbose=verbose)
        if mkk_mats or load_all:
            self.load_mkk_mats(ifield, inst, mkk_fpath=mkk_fpath)
        if psf or load_all:
            if ifield_psf is None:
                ifield_psf = ifield
            self.beta, self.rc, self.norm = load_psf_params_dict(inst, ifield=ifield, verbose=verbose)
        if maskInst or load_all:
            self.load_mask(ifield, inst, 'maskInst', verbose=verbose)
        if strmask or load_all:
            self.load_mask(ifield, inst, 'strmask', verbose=verbose)
        if FW_image or load_all:
            self.load_FW_image(ifield, inst, verbose=verbose)
        if FF_image or load_all:
            self.load_FF_image(ifield, inst, verbose=verbose)
            
    def compute_shot_noise_maps(self, inst, image, nsims, shot_sigma_map=None, nfr=None, frame_rate=None, ifield=None, g2_correct=True):

        ''' 
        If no shot_sigma_map provided, uses image to generate estimate of shot noise and subsequent photon noise realizations.

        Parameters
        ----------
        inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band.

        image : 
        nsims : (int)
        shot_sigma_map (optional) : 
        nfr :
        frame_rate : 
        ifield : 
        g2_correct :


        Returns
        -------

        shot_noise
        

        '''
        if nsims*self.dimx*self.dimy > 1e8:
            print('WARNING -- using a lot of memory at once')
        if nsims*self.dimx*self.dimy > 1e9:
            print('probably too much memory, exiting')
            return None

        if nfr is None:
            if ifield is not None:
                nfr = self.field_nfrs[ifield]

        if shot_sigma_map is None:
            print('computing shot sigma map from within compute_shot_noise_maps')
            shot_sigma_map = self.compute_shot_sigma_map(inst, image, nfr=nfr, frame_rate=frame_rate, g2_correct=g2_correct)

        shot_noise = np.random.normal(0, 1, size=(nsims, self.dimx, self.dimy))*shot_sigma_map

        return shot_noise

    def compute_shot_sigma_map(self, inst, image, nfr=None, frame_rate=None, verbose=False, g2_correct=True):

        ''' 
        Assume image is in surface brightness units, returns shot noise map in surface brightness units. 

        Parameters
        ----------
        
        inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band.
        image : `~np.array~` of type `float` and shape (dimx, dimy).
        nfr : `int`. Number of frames in exposure.
            Default is None.
        frame_rate : Number of frames per second.
            Default is None.
        verbose : `bool`. 
            Default is False.
        g2_correct : `bool`. If True, convert map from units of surface brightness to units of photocurrent ini e-/s.
            Default is True. 
        
        Returns
        -------
        shot_sigma_map : `~np.array~` of type `float` and shape (dimx, dimy).

        '''

        if nfr is None:
            nfr = self.nfr
        if frame_rate is None:
            frame_rate = self.frame_rate
        if verbose:
            print('nfr, frame_rate:', nfr, frame_rate)
        
        flight_signal = image.copy()
        if g2_correct:
            flight_signal /= self.g2_facs[inst]

        shot_sigma = np.sqrt((np.abs(flight_signal)/(nfr/frame_rate))*((nfr**2+1.)/(nfr**2 - 1.)))

        shot_sigma_map = shot_sigma.copy()
        if g2_correct:
            shot_sigma_map *= self.g2_facs[inst]

        return shot_sigma_map
        
    def compute_noise_power_spectrum(self, inst, noise_Cl2D=None, apply_FW=True, weights=None, verbose=False, inplace=True):
        
        ''' 
        For Fourier weighting, need apply_FW to be True, and then you need to specify the weights otherwise it grabs
        self.FW_image.
        
        Parameters
        ----------

        inst : `int`. 1 for 1.1 um band, 2 for 1.8 um band.
        noise_Cl2D : 
            Default is None.
        apply_FW : `bool`. 
            Default is True.
        weights : 
            Default is None.
        verbose : `bool`. 
            Default is False.
        inplace : `bool`. If True, saves noise power spectrum to class object
            Default is True.

        Returns
        -------

        Cl_noise : `np.array` of type 'float'. If not inplace, function returns 1-D noise power spectrum.

        '''

        verbprint(verbose, 'Computing 1D noise power spectrum from 2D power spectrum..')
        
        if noise_Cl2D is None:
            verbprint(verbose, 'No Cl2D explicitly provided, using self.noise_Cl2D..')
                
            noise_Cl2D = self.noise_Cl2D.copy()
            
        if apply_FW and weights is None:
            weights = self.FW_image
        elif not apply_FW:
            print('weights provided but apply_FW = False, setting weights to None')
            weights = None
        
        l2d = get_l2d(self.dimx, self.dimy, self.pixsize)

        lbins, Cl_noise, Clerr = azim_average_cl2d(noise_Cl2D*self.cal_facs[inst]**2, l2d, weights=weights, lbinedges=self.Mkk_obj.binl, lbins=self.Mkk_obj.midbin_ell, verbose=verbose)
        
        # lbins, Cl_noise, Clerr = azim_average_cl2d(noise_Cl2D*self.cal_facs[inst]**2, l2d, weights=weights, lbinedges=self.Mkk_obj.binl, lbins=self.Mkk_obj.midbin_ell)
        # Cl_noise has units of sb squared
        if inplace:
            self.N_ell = Cl_noise
            
            # verbprint(verbose, "self.N_ell is "self.N_ell)
            if verbose:
                print('ell^2 N_ell / 2pi = ', lbins**2*self.N_ell/(2*np.pi))
        else:
            return Cl_noise
        
        
    def compute_beam_correction_posts(self, ifield, inst, nbins=25, n_fine_bin=10, \
                                         psf_postage_stamps=None, beta=None, rc=None, norm=None,\
                                        tail_path='data/psf_model_dict_updated_081121_ciber.npz', \
                                        ndim=1):
    
        ''' Computes an average beam correction <B_ell> as an average over PSF templates uniformly sampled across the pixel function.

        Inputs
        ------
        
        ifield : type 'int'. CIBER field index (4, 5, 6, 7, 8 for science fields)
        band : type 'int'. CIBER band index (inst variable = band + 1)
        nbins (optional): type 'int'. Number of bins in power spectrum. Default = 25. 
        psf_postage_stamps (optional) : 'np.array' of floats with size (fac_upsample, fac_upsample, postage_stamp_dimx, postage_stamp_dimy).
                                        Grid of precomputed PSF templates.

        beta, rc, norm (optional): type 'float'. Parameters of PSF beta model. Default is None for all three.

        tail_path (optional): file path pointing to PSF beta model parameter dictionary file. 
                            Default is 'data/psf_model_dict_updated_081121_ciber.npz'.

        ndim (optional, default=1): type 'int'. Specifies dimension of power spectrum returned (1- or 2-d). Sometimes the 2D Bl is useful 


        Returns
        -------

        lb : 'np.array' of type 'float'. Multipole bins of power spectrum
        mean_bl : 'np.array' of size 'nbins'. Beam correction averaged over all PSF postage stamps.
        bls : 'np.array' of size (fac_upsample**2, nbins). List of beam corrections corresponding to individual PSF postage stamps.

        '''
        if psf_postage_stamps is None:
            if beta is None:
                beta, rc, norm = load_psf_params_dict(inst, ifield=ifield, tail_path=self.base_path+tail_path)
            psf_postage_stamps, subpixel_dists = generate_psf_template_bank(beta, rc, norm, n_fine_bin=n_fine_bin)
        
        bls = []
        if ndim==2:
            sum_bl2d = np.zeros((self.dimx, self.dimy))

        psf_stamp_dim = psf_postage_stamps[0,0].shape[0]
        
        psf_large = np.zeros((self.dimx, self.dimy))

        for p in range(psf_postage_stamps.shape[0]):
            for q in range(psf_postage_stamps.shape[1]):
     
                # place postage stamp in middle of image
                psf_large[self.dimx//2-psf_stamp_dim:self.dimx//2, self.dimy//2-psf_stamp_dim:self.dimy//2] = psf_postage_stamps[p,q]

                if ndim==2:
                    
                    l2d, bl2d = get_power_spectrum_2d(psf_large, pixsize=self.Mkk_obj.pixsize)
                    sum_bl2d += bl2d

                lb, clb, clberr = get_power_spec(psf_large - np.mean(psf_large), lbinedges=self.Mkk_obj.binl, lbins=self.Mkk_obj.midbin_ell)    
                bls.append(np.sqrt(clb/np.max(clb)))

        mean_bl = np.mean(bls, axis=0)
        bls = np.array(bls)

        if ndim==2:
            mean_bl2d = sum_bl2d/n_fine_bin**2
            return lb, mean_bl, bls, mean_bl2d
            
        return lb, mean_bl, bls

    def compute_beam_correction(self, psf=None, verbose=False, inplace=True):
        
        verbprint(verbose, 'Computing beam correction..')

        if psf is None:
            verbprint(verbose, 'No template explicitly provided, using self.psf_template to compute Bl..')

            psf = self.psf_template.copy()
        
        lb, Bl, Bl_std = get_power_spec(psf-np.mean(psf), lbinedges=self.Mkk_obj.binl, lbins=self.Mkk_obj.midbin_ell, return_Dl=False)
        
        B_ell = np.sqrt(Bl)/np.max(np.sqrt(Bl))
        B_ell_std = (Bl_std/(2*np.sqrt(Bl)))/np.max(np.sqrt(Bl))
        
        if inplace:
            self.B_ell = B_ell
            self.B_ell_std = B_ell_std
        else:
            return B_ell, B_ell_std

    def compute_ff_weights(self, inst, mean_norms, ifield_list, photon_noise=True, read_noise_models=None, nread=5, additional_rms=None):
        ''' 
        Compute flat field weights based on relative photon noise, read noise and mean normalization across the set of off-fields.
        This weighting is fairly optimal, ignoring the presence of signal fluctuations contributing to flat field error. 

        Inputs
        ------

        inst : `int`. 1 == 1.1 um, 2 == 1.8 um.
        mean_norms : `list` of `floats`. Mean surface brightness levels across fields.
        ifield_list : `list` of `ints`. Field indices.
        photon_noise : `bool`. If True, include photon noise in noise estimate.
            Default is True.
        read_noise_models : `list` of `~np.arrays~` of type `float`, shape (self.dimx,self.dimy). Read noise models
            Default is None.
        additional_rms : array_like
            Default is None.
    
        Returns
        -------

        weights : `list` of `floats`. Final weights, normalized to sum to unity.

        '''

        if photon_noise is False and read_noise_models is None:
            print("Neither photon noise nor read noise models provided, weighting fields by mean brightness only..")
            weights = mean_norms
            return weights/np.sum(weights)

        rms_read = np.zeros_like(mean_norms)
        rms_phot = np.zeros_like(mean_norms)
        if additional_rms is None:
            additional_rms = np.zeros_like(mean_norms)

        if photon_noise:
            for i, ifield in enumerate(ifield_list):
                shot_sigma_sb = self.compute_shot_sigma_map(inst, image=mean_norms[i]*np.ones((10, 10)), nfr=self.field_nfrs[ifield])         
                rms_phot[i] = shot_sigma_sb[0,0]

        if read_noise_models is not None:
            # compute some read noise realizations and measure mean rms. probably don't need that many realizations
            for i, ifield in enumerate(ifield_list):

                rnmaps, _ = self.noise_model_realization(inst, (nread, self.dimx, self.dimy), read_noise_models[i],\
                                                  read_noise=True, photon_noise=False)

                rms_read[i] = np.std(rnmaps)

        weights = (mean_norms/(np.sqrt(rms_phot**2 + rms_read**2+additional_rms**2)))**2
        print('ff weights are computed to be ', weights)

        weights /= np.sum(weights)

        return weights



    def calculate_transfer_function(self, inst, nsims, load_ptsrc_cib=False, niter_grad_ff=1, dgl_scale_fac=5., indiv_ifield=6, apply_FF=False, FF_image=None, FF_fpath=None, plot=False):

        cls_orig = np.zeros((nsims, self.n_ps_bin))
        cls_filt = np.zeros((nsims, self.n_ps_bin))
        t_ells = np.zeros((nsims, self.n_ps_bin))


        if apply_FF:
            if FF_image is None:
                if FF_fpath is not None:
                    FF_image = fits.open(FF_fpath)[0].data 
                else:
                    print('need to either provide file path to FF or FF image..')

            processed_ims, ff_estimates, final_planes, _, coeffs_vs_niter, _ = iterative_gradient_ff_solve(input_signals, niter=niter)
        else:

            for i in range(nsims):

                diff_realization = generate_custom_dgl_clustering(self, dgl_scale_fac=dgl_scale_fac, gen_ifield=indiv_ifield)
                if not load_ptsrc_cib:        
                    _, _, shotcomp = generate_diffuse_realization(self.dimx, self.dimy, power_law_idx=0.0, scale_fac=3e8)
                else:
                    print('need to load CIB here')

                diff_realization += shotcomp 

                if plot:
                    plot_map(diff_realization, title='diff realization '+str(i))

                lb, cl_orig, clerr_orig = get_power_spec(diff_realization - np.mean(diff_realization), lbinedges=self.Mkk_obj.binl, lbins=self.Mkk_obj.midbin_ell)


                theta, plane = fit_gradient_to_map(diff_realization)
                diff_realization -= plane

                lb, cl_filt, clerr_filt = get_power_spec(diff_realization - np.mean(diff_realization), lbinedges=self.Mkk_obj.binl, lbins=self.Mkk_obj.midbin_ell)
                cls_orig[i] = cl_orig
                cls_filt[i] = cl_filt
                t_ell_indiv = cl_filt/cl_orig 

                t_ells[i] = t_ell_indiv

        t_ell_av = np.median(t_ells, axis=0)
        t_ell_stderr = np.std(t_ells, axis=0)/np.sqrt(t_ells.shape[0])


        return lb, t_ell_av, t_ell_stderr, t_ells, cls_orig, cls_filt

  
    def compute_processed_power_spectrum(self, inst, bare_bones=False, mask=None, apply_mask=True, N_ell=None, B_ell=None, inv_Mkk=None,\
                                         image=None, mkk_correct=True, beam_correct=True, ff_bias_correct=None, transfer_function_correct=False, t_ell = None, \
                                         FF_correct=True, FF_image=None, gradient_filter=False, apply_FW=True, noise_debias=True, verbose=False, \
                                        convert_adufr_sb=True, save_intermediate_cls=True):
        

        self.masked_Cl_pre_Nl_correct = None
        self.masked_Cl_post_Nl_correct = None
        self.cl_post_mkk_pre_Bl = None

        if image is None:
            image = self.image.copy()
        else:
            xim = image.copy()
            image = xim.copy()

        # image = image.copy()
        
        if convert_adufr_sb:
            image *= self.cal_facs[inst] # convert from ADU/fr to nW m^-2 sr^-1
        
        if FF_correct and not bare_bones:
            verbprint(True, 'Applying flat field correction..')

            if FF_image is not None:
                verbprint(verbose, 'Mean of FF_image is '+str(np.mean(FF_image))+' with standard deviation '+str(np.std(FF_image)))

                image = image / FF_image

        if mask is None and apply_mask:
            verbprint(verbose, 'Getting mask from maskinst and strmask')
            mask = self.maskInst*self.strmask # load mask


        if gradient_filter: 
            verbprint(True, 'Gradient filtering image..')
            theta, plane = fit_gradient_to_map(image, mask=mask)
            image -= plane

        if apply_mask and not bare_bones:

            verbprint(True, 'Applying mask..')
            masked_image = mean_sub_masked_image(image, mask) # apply mask and mean subtract unmasked pixels
        else:
            verbprint(True, 'No masking, subtracting image by its mean..')
            masked_image = image - np.mean(image)
            mask = None

        verbprint(verbose, 'Mean of masked image is '+str(np.mean(masked_image)))
         
        weights=None
        if apply_FW and not bare_bones:
            verbprint(True, 'Using Fourier weights..')
            weights = self.FW_image # this feeds Fourier weights into get_power_spec, which uses them with Cl2D
            
        lbins, cl_proc, cl_proc_err = get_power_spec(masked_image, mask=None, weights=weights, lbinedges=self.Mkk_obj.binl, lbins=self.Mkk_obj.midbin_ell)
        
        if save_intermediate_cls:
            self.masked_Cl_pre_Nl_correct = cl_proc.copy()

        verbprint(verbose, 'cl_proc after get_power spec is ')
        verbprint(verbose, cl_proc)
            
        if noise_debias and not bare_bones: # subtract noise bias
            verbprint(True, 'Applying noise bias..')

            if N_ell is None:
                cl_proc -= self.N_ell
            else:
                cl_proc -= N_ell     
                
            verbprint(verbose, 'after noise subtraction, ')
            verbprint(verbose, cl_proc)

            if save_intermediate_cls:
                self.masked_Cl_post_Nl_correct = cl_proc.copy()

  
        if mkk_correct and not bare_bones: # apply inverse mode coupling matrix to masked power spectrum
            verbprint(True, 'Applying Mkk correction..')

            if inv_Mkk is None:
                cl_proc = np.dot(self.inv_Mkk.transpose(), cl_proc)
                cl_proc_err= np.dot(np.abs(self.inv_Mkk.transpose()), cl_proc_err)
            else:
                cl_proc = np.dot(inv_Mkk.transpose(), cl_proc)
                cl_proc_err= np.sqrt(np.dot(np.abs(inv_Mkk.transpose()), cl_proc_err**2))

            verbprint(verbose, 'After Mkk correction, cl_proc is ')
            verbprint(verbose, cl_proc)
            verbprint(verbose, 'After Mkk correction, cl_proc_err is ')
            verbprint(verbose, cl_proc_err)

            if save_intermediate_cls:
                self.cl_post_mkk_pre_Bl = cl_proc.copy()
                            
        if beam_correct and not bare_bones: # undo the effects of the PSF by dividing power spectrum by B_ell
            verbprint(True, 'Applying beam correction..')

            if B_ell is None:
                B_ell = self.B_ell
            assert len(B_ell)==len(cl_proc)
            cl_proc /= B_ell**2
            cl_proc_err /= B_ell**2

        verbprint(verbose, 'Processed angular power spectrum is ')
        verbprint(verbose, cl_proc)
        verbprint(verbose, 'Processed angular power spectrum error is ')
        verbprint(verbose, cl_proc_err)


        return lbins, cl_proc, cl_proc_err, masked_image
            


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



def generate_synthetic_mock_test_set(test_set_fpath, trilegal_sim_idx=1, inst=1, cmock=None, cbps=None, n_ps_bin=25, ciberdir='/Users/luminatech/Documents/ciber2/ciber/',\
                                     dimx=1024, dimy=1024, ifield_list=None, generate_diffuse_realization=False, diffuse_realizations=None, power_law_idx=-4.5, scale_fac=2e-1, bkg_val=250.,\
                                     apply_flat_field = False, include_inst_noise=True, include_photon_noise=True, ff_truth=None, ff_truth_fpath='dr40030_TM2_200613_p2.pkl',\
                                     generate_masks=False, mask_galaxies=False, mask_stars=True, intercept_mag=16.0, a1=220., b1=3.632, c1=8.52, param_combo=None, Vega_to_AB=0.91, mag_lim_Vega = 18.0, \
                                     n_split_mkk=1, verbose=False, \
                                    show_plots=True, ifield_psf=None, use_field_nfrs=True, mock_trilegal_path=None, load_bls=True, compute_bls=False, B_ells=None, dx_mcat=0., dy_mcat=0., \
                                    transpose_noise=False):

    if cmock is None:
        cmock = ciber_mock(ciberdir='/Users/luminatech/Documents/ciber2/ciber/')
        
    if cbps is None:
        cbps = CIBER_PS_pipeline(n_ps_bin=n_ps_bin, dimx=dimx, dimy=dimy)
        if not use_field_nfrs:
            cbps.field_nfrs = None

    if ifield_list is None:
        ifield_list = [4, 5, 6, 7, 8]
    nfields = len(ifield_list)
    
    if 'fits' in test_set_fpath:
        mock_cib_file = fits.open(test_set_fpath)
        mock_cib_ims = np.array([mock_cib_file['map'+str(ifield)].data for ifield in ifield_list])
    elif 'npz' in test_set_fpath:
        mock_cib_ims = np.load(test_set_fpath)['full_maps']
    else:
        print('unrecognized file type for '+str(test_set_fpath))
        return None

    midxdict = dict({'x':0, 'y':1, 'redshift':2, 'm_app':3, 'M_abs':4, 'Mh':5, 'Rvir':6})
    

    # if no mean levels provided, use ZL mean levels from file in kelsall folder
    if bkg_val is None:
        bkg_val_dict = np.load(cbps.data_path+'kelsall/mean_zl_level_kelsall.npz', allow_pickle=True)['mean_zl_level'][inst]
        bkg_val = [bkg_val_dict[cmock.ciber_field_dict[ifield]] for ifield in ifield_list]
    elif not type(bkg_val)==list:
        bkg_val = [bkg_val for x in range(len(ifield_list))]

    if verbose:
        print('bkg vals are ', bkg_val)

    
    observed_ims, total_signals, \
        noise_models, shot_sigma_sb_maps, rnmaps, joint_masks, diff_realizations = instantiate_dat_arrays_fftest(cbps.dimx, cbps.dimy, nfields)

    cls_gt, cls_diffuse, cls_cib, cls_postff, B_ells_new = instantiate_cl_arrays_fftest(nfields, cbps.n_ps_bin)

    mag_lim_AB = mag_lim_Vega + Vega_to_AB
    intercept_mag_AB = intercept_mag + Vega_to_AB

    for imidx in range(nfields):
        ifield = ifield_list[imidx]

        if cbps.field_nfrs is not None:
            field_nfr = cbps.field_nfrs[ifield]
        else:
            field_nfr = cbps.nfr

        field_name_trilegal = cmock.ciber_field_dict[ifield]
    
        
        print('image ', imidx+1, ' of 5')
    
        # ---------------- load 2d noise model ---------------------------
        
        if ifield_psf is None:
            ifield_psf = ifield
        
        cbps.load_data_products(ifield, inst, verbose=verbose, ifield_psf=ifield_psf)
        cmock.get_psf(ifield=ifield_psf)
    
        if compute_bls:
            lb, B_ell, B_ell_list = cbps.compute_beam_correction_posts(ifield, inst, nbins=cbps.n_ps_bin, n_fine_bin=10, tail_path='data/psf_model_dict_updated_081121_ciber.npz')
            if verbose:
                print('B_ell is ', B_ell)
                print('B_ell_list has length', len(B_ell_list))
            B_ells_new[imidx] = B_ell
        elif B_ells is not None:
            B_ell = B_ells[imidx]
        else:
            B_ell = cbps.B_ell
        

        # newnoisefpath = 'data/fluctuation_data/dr20210120/TM'+str(inst)+'/readCl2D/field'+str(ifield)+'_readCl2D.fits'
        if include_photon_noise or include_inst_noise:
            # noise_model = cbps.load_noise_Cl2D(ifield, inst, noise_fpath='data/fluctuation_data/TM'+str(inst)+'/noiseCl2D/dr20211007_rmf/field'+str(ifield)+'_noiseCl2D_rmf_highellmask.fits', inplace=False, transpose=transpose_noise
            noise_model = cbps.load_noise_Cl2D(ifield, inst, noise_fpath='data/fluctuation_data/TM'+str(inst)+'/noiseCl2D/field'+str(ifield)+'_noiseCl2D_110421.fits', inplace=False, transpose=transpose_noise)
            # noise_model = cbps.load_noise_Cl2D(noise_fpath='data/fluctuation_data/TM'+str(inst)+'/readCl_input_dr20210611/field'+str(ifield)+'_readCl2d_input.fits', inplace=False, transpose=True)
            # noise_model = cbps.load_noise_Cl2D(ifield, inst, noise_fpath=newnoisefpath, inplace=False, transpose=True)

            if verbose:
                print('min/max of noise model are ', np.min(noise_model), np.max(noise_model))
            noise_models[imidx] = noise_model
        instrument_mask = cbps.maskInst
        

        # if mock_trilegal_path is None:
            # mock_trilegal_path = 'data/mock_trilegal_realizations_081621/'+field_name_trilegal+'/mock_trilegal_'+field_name_trilegal+'_idx'+str(trilegal_sim_idx+1)+'_081621.npz'
            # mock_trilegal_path = 'data/mock_trilegal_realizations_051321/'+field_name_trilegal+'/mock_trilegal_'+field_name_trilegal+'_idx'+str(trilegal_sim_idx)+'_051321.npz'
        if mock_trilegal_path is not None:
            if '.npz' in mock_trilegal_path:
                mock_trilegal = np.load(mock_trilegal_path)
                mock_trilegal_im = mock_trilegal['srcmaps'][inst-1,:,:]

                if verbose:
                    print('mock_trilegal_im has shape ', mock_trilegal_im.shape)

                mock_trilegal_cat = mock_trilegal['cat']
            elif '.fits' in mock_trilegal_path:
                print('loading trilegal fits file..')
                mock_trilegal = fits.open(mock_trilegal_path)
                mock_trilegal_im = mock_trilegal['trilegal_'+str(cbps.inst_to_band[inst])+'_'+str(ifield)].data.transpose()
        
        #  ------------- CIB realizations ----------------

        lbins, cl_cib, cl_cib_err, _ = cbps.compute_processed_power_spectrum(inst, image=mock_cib_ims[imidx]-np.mean(mock_cib_ims[imidx]), bare_bones=True, convert_adufr_sb=False, verbose=verbose)
        cls_cib[imidx] = cl_cib/B_ell**2
    
    
        # ------------- diffuse dgl realization -----------------------
        
        if diffuse_realizations is not None:
            diff_realization = diffuse_realizations[imidx]
        elif generate_diffuse_realization:
            cl_dgl_iras = np.load('data/fluctuation_data/TM'+str(inst)+'/dgl_sim/dgl_from_iris_model_TM'+str(inst)+'_'+field_name_trilegal+'.npz')['cl']
            cl_pivot_fac = cl_dgl_iras[0]*cbps.dimx*cbps.dimy 
            _, _, diff_realization = generate_diffuse_realization(cbps.dimx, cbps.dimy, power_law_idx=-3.0, scale_fac=cl_pivot_fac)
        else:
            diff_realization = None 

        if diff_realization is not None:
            lbins, cl_diffuse,cl_diffuse_err,  _ = cbps.compute_processed_power_spectrum(inst, image=diff_realization, bare_bones=True, convert_adufr_sb=False, verbose=verbose)
            cls_diffuse[imidx] = cl_diffuse/B_ell**2
            diff_realizations[imidx] = diff_realization

        # ------------- combine individual components into total_signal --------------------

        total_signal = mock_cib_ims[imidx].copy()

        if mock_trilegal_path is not None:
            total_signal += mock_trilegal_im

        # if include_diffuse_comp:
        if diff_realization is not None:
            total_signal += diff_realization
        
        total_signal += bkg_val[imidx]
            
        lbins, cl_tot, cl_tot_err, _ = cbps.compute_processed_power_spectrum(inst, image=total_signal-np.mean(total_signal), bare_bones=True, convert_adufr_sb=False, verbose=False)
        
        if verbose:
            print('cl_tot is ', cl_tot)

        gt_sig = mock_cib_ims[imidx]+diff_realizations[imidx]
        lbins, cl_gt, cl_gt_err, _ = cbps.compute_processed_power_spectrum(inst, image=gt_sig-np.mean(gt_sig), bare_bones=True, convert_adufr_sb=False, verbose=False)
        cls_gt[imidx] = cl_gt/B_ell**2
        
        #  ----------------- generate read and shot noise realizations -------------------
        
        shot_sigma_sb = cbps.compute_shot_sigma_map(inst, image=total_signal, nfr=field_nfr)
        
        if include_photon_noise:
            shot_sigma_sb_maps[imidx] = shot_sigma_sb
            
        if include_photon_noise or include_inst_noise:
            rnmap, snmap = cbps.noise_model_realization(inst, (cbps.dimx, cbps.dimy), noise_models[imidx], read_noise=include_inst_noise, shot_sigma_sb=shot_sigma_sb, image=total_signal)
            
        # ------------------- add noise to signal and multiply by the flat field to get "observed" images -----------------------

        if apply_flat_field:
            if ff_truth is None:
                with open(ff_truth_fpath, "rb") as f:
                    dr = pickle.load(f)
                ff_truth = dr[ifield]['FF']
                ff_truth[np.isnan(ff_truth)] = 1.0
                ff_truth[ff_truth==0] = 1.0
        else:
            ff_truth = np.ones_like(total_signal)
            
        
        sum_mock_noise = ff_truth*total_signal
        
        if include_photon_noise:
            sum_mock_noise += ff_truth*snmap
                
        if include_inst_noise:
            sum_mock_noise += rnmap

        total_signals[imidx] = total_signal
        
        if include_inst_noise:
            rnmaps[imidx] = rnmap

        observed_ims[imidx] = sum_mock_noise
        
        if show_plots:
            
            # for plotting image limits
            x0, x1 = 0, 1024
            y0, y1 = 0, 1024
            
            fm = plot_map(gaussian_filter(mock_cib_ims[imidx], sigma=20), title='CIB (smoothed)', x0=x0, x1=x1, y0=y0, y1=y1)
            fm = plot_map(mock_cib_ims[imidx], title='CIB', x0=x0, x1=x1, y0=y0, y1=y1)

            if mock_trilegal_path is not None:
                f = plot_map(mock_trilegal_im, title='trilegal', x0=x0, x1=x1, y0=y0, y1=y1)
            if diff_realization is not None:
                f = plot_map(diff_realization, title='diff realization', x0=x0, x1=x1, y0=y0, y1=y1)
            if include_photon_noise:
                f = plot_map(snmap, title='shot noise', x0=x0, x1=x1, y0=y0, y1=y1)
            if include_inst_noise:
                f = plot_map(rnmap, title='read noise', x0=x0, x1=x1, y0=y0, y1=y1)
            f = plot_map(sum_mock_noise, title='CIB+zodi+shot noise+read noise', x0=x0, x1=x1, y0=y0, y1=y1)
            f = plot_map(ff_truth, title='flat field', x0=x0, x1=x1, y0=y0, y1=y1)
            f = plot_map(observed_ims[imidx], title='post ff image', x0=x0, x1=x1, y0=y0, y1=y1)


        if generate_masks:

            if param_combo is None:
                param_combo = [a1, b1, c1]

            # dx_mcat, dy_mcat = 2.5, 1.0 # 0.0, 0.0
            
            if mask_galaxies:
                gal_cat = {'j_m': mock_cat[:,midxdict['m_app']],'x'+str(inst): mock_cat[:,midxdict['x']]+dx_mcat, 'y'+str(inst): mock_cat[:,midxdict['y']]+dy_mcat}
                gal_cat_df = pd.DataFrame(gal_cat, columns = ['j_m', 'x'+str(inst), 'y'+str(inst)]) # check magnitude system of Helgason model

            if mask_stars:
                star_cat = {'j_m:':mock_trilegal_cat[:,3], 'x'+str(inst):mock_trilegal_cat[:,1]-dy_mcat, 'y'+str(inst): mock_trilegal_cat[:,0]+dx_mcat}
                star_cat_df = pd.DataFrame(star_cat)
                star_cat_df.columns = ['j_m', 'x'+str(inst), 'y'+str(inst)]

            # --------------- mask construction --------------------

            joint_mask, radii_stars_simon, radii_stars_Z14 = get_masks(star_cat_df, param_combo, intercept_mag_AB, mag_lim_AB, inst=inst, instrument_mask=instrument_mask, dimx=cbps.dimx, dimy=cbps.dimy, verbose=True)
            joint_masks[imidx] = joint_mask 

            if show_plots:
                plot_srcmap_mask(joint_masks[imidx], 'joint mask', len(radii_stars_simon)+len(radii_stars_Z14))
                plot_map(observed_ims[imidx]*joint_masks[imidx], title='postff observed image with full mask')

    if B_ells is None and compute_bls:
        B_ells = B_ells_new

    return joint_masks, observed_ims, total_signals, rnmaps, shot_sigma_sb_maps, noise_models, cls_gt, cls_diffuse, cls_cib, cls_postff, B_ells, ff_truth, diff_realizations, ifield_list
        
    
    
    
def calculate_powerspec_quantities(cbps, observed_ims, joint_masks, shot_sigma_sb_maps, noise_models, ifield_list, include_inst_noise=False, include_photon_noise=True, inst=1, ff_estimates=None, ff_truth=None, use_true_ff=False, inv_var_weight=True, infill_smooth_scale=3.,\
                                  ff_stack_min=1, ff_min=0.2, apply_masking=True, apply_fourier_mask=False, inverse_Mkks=None, mkk_fpath=None, n_mkk_sims=100, n_split_mkk=2, \
                                           apply_FW = True, fourier_masks=None, noise_debias=True, mkk_correct=True, beam_correct=True, n_FW_sims = 500, n_FW_split = 10, B_ells=None, show_plots=False, verbose=False, convert_adufr_sb=False, \
                                           stdpower=1):
    
    ''' 
    This is the (monster) function which processes the relevant data products to calculate the underlying sky power spectrum 

    Parameters
    ----------


    Returns
    -------


    '''

    compute_ff = False
    if ff_estimates is None:
        ff_estimates = np.zeros((len(ifield_list), cbps.dimx, cbps.dimy))

        if not use_true_ff:
            print('No ff estimates provided, will compute here using stacking estimator')
            compute_ff = True
    ff_mask = None

    nfields = len(observed_ims)
    
    compute_mkk = False
    if inverse_Mkks is None:
        inverse_Mkks = np.zeros((nfields, cbps.n_ps_bin, cbps.n_ps_bin))
        compute_mkk = True
        
    recovered_cls = np.zeros((nfields, cbps.n_ps_bin))
    recovered_dcls = np.zeros((nfields, cbps.n_ps_bin))
    masked_Nls = np.zeros((nfields, cbps.n_ps_bin))
    # masked_Nls_noff = np.zeros((nfields, cbps.n_ps_bin))
    cls_intermediate = []

    masked_images = np.zeros((nfields, cbps.dimx, cbps.dimy))


    if verbose:
        print('cbps g1 facs are ', cbps.g1_facs)
        print('cbps g2 facs are ', cbps.g2_facs)
    
    for imidx, obs in enumerate(observed_ims):

        # plot_map(obs*joint_masks[imidx], title='obs*mask')

        if use_true_ff:
            ff_estimates[imidx] = ff_truth
        elif compute_ff:
            stack_obs = list(observed_ims.copy())
            stack_mask = list(joint_masks.copy().astype(np.bool))

            if cbps.field_nfrs is not None:
                field_nfrs = list(cbps.field_nfrs.copy())
                del(field_nfrs[imidx])
            else:
                field_nfrs = None
    
            del(stack_obs[imidx])
            del(stack_mask[imidx])

            # for i in range(len(observed_ims)-1):
                # plot_map(stack_obs[i]*stack_mask[i], title='check i = '+str(i))
            # plot_map(obs*joint_masks[imidx], title='obs*mask before stack ff')


            ff_estimate, ff_mask, ff_weights = compute_stack_ff_estimate(stack_obs, target_mask=joint_masks[imidx], masks=stack_mask, inv_var_weight=inv_var_weight, \
                                                                        field_nfrs=field_nfrs)
            
            ff_estimates[imidx] = ff_estimate

            sum_stack_mask = np.sum(stack_mask, axis=0)

            if show_plots:
                plot_map(obs*joint_masks[imidx]*ff_mask, title='obs*mask after stack ff')

                plot_map(ff_estimate*joint_masks[imidx]*ff_mask, title='ff estimate * joint masks[imidx]')
                sumstackfig = plot_map(sum_stack_mask, title='sum stack mask')

        if ff_truth is not None:
            ff_truth_masked = ff_truth.copy()
            ff_truth_masked[joint_masks[imidx]==0] = 1.0
            ff_truth_masked[ff_mask==0] = 1.0
            

        if show_plots and compute_ff:
            ff_f = plot_map(ff_estimates[imidx], title='FF estimate', nanpct=True)
            if ff_truth is not None:
                ff_resid = plot_map(ff_truth_masked, title='FF true', nanpct=True)
                kernel = Gaussian2DKernel(infill_smooth_scale)
                ff_resid = plot_map(convolve(ff_truth_masked-ff_estimates[imidx], kernel), title='FF true - FF estimate', nanpct=True, cmap='bwr')

        joint_mask_with_ff = None
        if apply_masking:
            if show_plots:
                plot_srcmap_mask(joint_masks[imidx], 'joint mask', 0.)
                if ff_mask is not None:
                    plot_srcmap_mask(np.array(ff_mask).astype(np.int), 'ff_mask', 0.)
                    plot_srcmap_mask(joint_masks[imidx]*np.array(ff_mask).astype(np.int), 'joint mask x ff_mask', 0.)

            joint_mask_with_ff = joint_masks[imidx].copy()
            
            # print('before, joint mask frac is ', float(np.sum(joint_masks[imidx]))/float(cbps.dimx*cbps.dimy))

            if not use_true_ff and ff_mask is not None:
                print('multiplying joint mask by ff mask')
                joint_mask_with_ff *= ff_mask
                
            # print('after ff_mask and ff_inf_mask, joint mask frac is ', float(np.sum(joint_mask_with_ff))/float(cbps.dimx*cbps.dimy))
            # print('computing Mkk matrices..')

            if compute_mkk:
                av_Mkk = cbps.Mkk_obj.get_mkk_sim(joint_mask_with_ff, n_mkk_sims, n_split=n_split_mkk, store_Mkks=False)
                inverse_Mkk = compute_inverse_mkk(av_Mkk)
                inverse_Mkks[imidx] = inverse_Mkk
                plot_mkk_matrix(inverse_Mkk, inverse=True, symlogscale=True)
        
            else:
                if inverse_Mkks is not None:
                    inv_Mkk = inverse_Mkks[imidx]
                else:
                    cbps.load_data_products(ifield_list[imidx], inst, load_all=False, load_mkk_mats=True, mkk_fpath=mkk_fpath)
                    inv_Mkk = cbps.inv_Mkk


        # --------------- fourier weights with full mask and noise sims ------------------------------
        

        if include_photon_noise or include_inst_noise:
        

            # fourier_weights, mean_cl2d = cbps.compute_FW_from_noise_sim(nsims=n_FW_sims, n_split=n_FW_split, apply_mask=apply_masking, \
            #                                mask=joint_mask_with_ff, noise_model=noise_models[imidx], inst=inst, show=False,\
            #                                 read_noise=include_inst_noise, photon_noise=include_photon_noise, shot_sigma_sb=shot_sigma_sb_maps[imidx], inplace=False, \
            #                                 ff_estimate=ff_estimates[imidx], stdpower=stdpower)

            # fourier_weights, mean_cl2d = cbps.compute_FW_from_noise_sim(nsims=n_FW_sims, n_split=n_FW_split, apply_mask=apply_masking, \
            #                                mask=joint_mask_with_ff, noise_model=noise_models[imidx], inst=inst, show=False,\
            #                                 read_noise=include_inst_noise, photon_noise=include_photon_noise, shot_sigma_sb=shot_sigma_sb_maps[imidx], inplace=False, \
            #                                 ff_estimate=None, stdpower=stdpower)

            if not use_true_ff:
                unmasked_mean_level = np.mean(obs[joint_mask_with_ff==1])
                print('unmasked mean level is ', unmasked_mean_level)
            else:
                unmasked_mean_level = None

            fourier_weights, mean_cl2d = cbps.estimate_noise_power_spectrum(ifield=ifield_list[imidx], nsims=n_FW_sims, n_split=n_FW_split, apply_mask=apply_masking, \
               mask = joint_mask_with_ff, noise_model=noise_models[imidx], inst=inst, show=False, read_noise=include_inst_noise, \
                photon_noise=include_photon_noise, ff_estimate=ff_estimates[imidx], shot_sigma_sb=shot_sigma_sb_maps[imidx], inplace=False, \
                simmap_dc = unmasked_mean_level)
            


            if apply_fourier_mask:

                if fourier_masks is not None:
                    fourier_weights *= fourier_masks[imidx]
                else:
                    ybandtop = cbps.dimy//2 + 100
                    ybandbottom = cbps.dimy//2 - 100
                    noband_noise = np.zeros_like(fourier_weights)
                    noband_noise[ybandtop:,:] = mean_cl2d[ybandtop:,:]
                    noband_noise[:ybandbottom,:] = mean_cl2d[:ybandbottom,:]
                    itermask = iter_sigma_clip_mask(noband_noise, sig=5, nitermax=10, initial_mask=np.ones_like(noband_noise.astype(np.int)))
                    itermask[ybandbottom:ybandtop,:] = 1
                    plot_map(mean_cl2d, title='mean cl2d')
                    plot_map(itermask, title='itermask')
                    fourier_weights *= itermask
            
            if apply_FW:
                print('setting cbps.FW_image to fourier_weights..')
                cbps.FW_image = fourier_weights

                if fourier_masks is not None:
                    print('fourier_masks is not None, multiplying fourier weights by mask..')
                    cbps.FW_image *= fourier_masks[imidx]

            # want the power spectrum bias due to read + photon noise, passed through 
            cbps.compute_noise_power_spectrum(inst, noise_Cl2D=mean_cl2d, inplace=True, apply_FW=apply_FW, weights=fourier_weights)

            masked_Nls[imidx] = cbps.N_ell
        
        
        if not include_inst_noise and not include_photon_noise:
            print('Neither instrument noise nor photon noise provided, so setting noise_debias to False and apply_FW to False..')
            noise_debias = False
            apply_FW = False

        # if applying beam correction, use input B_ells or load from file if not present
        if beam_correct:
            if B_ells is not None:
                B_ell = B_ells[imidx]
            else:
                cbps.load_data_products(ifield_list[imidx], inst, load_all=False, beam_correction=True)
                B_ell = cbps.B_ell

        if show_plots and joint_mask_with_ff is not None:
            plot_map(obs, title='obs')

            plt.figure()
            plt.hist((obs*joint_mask_with_ff).ravel(), bins=100)
            plt.yscale('log')
            plt.show()

            plot_map(obs*joint_mask_with_ff, title='obs x joint mask')



        lb, cl_proc, cl_proc_err, masked_image = cbps.compute_processed_power_spectrum(inst, mask=joint_mask_with_ff, apply_mask=apply_masking, \
                                                                 image=obs, convert_adufr_sb=convert_adufr_sb, \
                                                                mkk_correct=mkk_correct, beam_correct=beam_correct, B_ell=B_ell, \
                                                                apply_FW=apply_FW, verbose=verbose, noise_debias=noise_debias, \
                                                             FF_correct=True, FF_image=ff_estimates[imidx], inv_Mkk=inverse_Mkks[imidx], save_intermediate_cls=True)

        cls_intermediate.append([cbps.masked_Cl_pre_Nl_correct, cbps.masked_Cl_post_Nl_correct, cbps.cl_post_mkk_pre_Bl])
        

        print('recovered cl is ', cl_proc)
        recovered_cls[imidx] = cl_proc
        recovered_dcls[imidx] = cl_proc_err
        masked_images[imidx] = masked_image


    
    return ff_estimates, inverse_Mkks, lb, recovered_cls, recovered_dcls, masked_images, masked_Nls, None, cls_intermediate


def compute_powerspectra_realdat(inst, n_ps_bin=25, J_mag_lim=17.5, ifield_list=[4, 5, 6, 7, 8], \
                                n_mkk_sims=50, n_split_mkk=1, n_FW_sims=50, n_FW_split=1, mask_fpaths=None, fourier_masks=None, ff_stack_min=1, \
                                show_plots=False, g1_fac_dict=None, g2_fac_dict=None, inverse_Mkks=None, \
                                base_fluc_path='/Volumes/Seagate Backup Plus Drive/Toolkit/Mirror/Richard/ciber_fluctuation_data/', \
                                use_ff_est=True, ff_estimates=None, apply_fourier_mask=False, transpose_noise=False, verbose=False):
    nfields = len(ifield_list)
    B_ells = np.zeros((nfields, n_ps_bin))
    readnoise_cls = np.zeros((nfields, n_ps_bin))
    observed_ims, _, noise_models, shot_sigma_sb_maps, _, joint_masks, _ = instantiate_dat_arrays_fftest(1024, 1024, 5)

    cbps = CIBER_PS_pipeline(n_ps_bin=n_ps_bin)
    

    if g1_fac_dict is not None:
        cbps.g1_facs = g1_fac_dict
   
    if g2_fac_dict is not None:
        cbps.g2_facs = g2_fac_dict
        cbps.cal_facs = dict({1:cbps.g1_facs[1]*cbps.g2_facs[1], 2:cbps.g1_facs[2]*cbps.g2_facs[2]}) # updated 

    print('cbps cal facs are ', cbps.cal_facs)
    print('cbps g1 facs are ', cbps.g1_facs)

    for i, ifield in enumerate(ifield_list):

        cbps.load_data_products(ifield, inst, verbose=True)

        print('cbps.B_ell is ', cbps.B_ell)
        cbps.load_noise_Cl2D(ifield, inst, noise_fpath='data/fluctuation_data/TM'+str(inst)+'/noiseCl2D/field'+str(ifield)+'_noiseCl2D_110421.fits', inplace=True, transpose=transpose_noise)
        
        # cbps.load_noise_Cl2D(ifield, inst, noise_fpath='data/fluctuation_data/TM'+str(inst)+'/noiseCl2D/dr20211007_rmf/field'+str(ifield)+'_noiseCl2D_rmf_highellmask.fits', inplace=True, transpose=transpose_noise)
        # cbps.load_noise_Cl2D(noise_fpath='data/fluctuation_data/TM'+str(inst)+'/readCl_input_dr20210611/field'+str(ifield)+'_readCl2d_input.fits', transpose=True)
        observed_ims[i] = cbps.image*cbps.cal_facs[inst]

        if show_plots:
            plot_map(cbps.noise_Cl2D, title='loaded Noise CL2d (transpose is '+str(transpose_noise)+')')
            plot_map(observed_ims[i], title='obsreved ims i='+str(i))
        
        noise_models[i] = cbps.noise_Cl2D

        shot_sigma_sb_maps[i] = cbps.compute_shot_sigma_map(inst, image=observed_ims[i], nfr=cbps.field_nfrs[ifield])
        # shot_sigma_sb_maps[i] = np.median(shot_sigma_sb_maps[i])*np.ones_like(observed_ims[i])
        
#         mask_fpathprev = cbps.data_path+'/TM'+str(inst)+'/mask/field'+str(ifield)+'_TM'+str(inst)+'_strmask_Jlim='+str(J_mag_lim)+'_040521.fits'
#         instm = cbps.load_mask(ifield, inst, mask_fpath=mask_fpathprev, masktype='strmask', inplace=False)
#         jm = np.array(cbps.maskInst_clean*instm)
#         joint_masks[i] = np.array(cbps.maskInst_clean*cbps.strmask)
        
        if mask_fpaths is None:
            mask_fpath = base_fluc_path+'/TM'+str(inst)+'/masks/joint_mask_with_ffmask_ifield'+str(ifield)+'_inst'+str(inst)+'_abc110821.fits'
            # mask_fpath = base_fluc_path+'/TM'+str(inst)+'/masks/joint_mask_with_ffmask_ifield'+str(ifield)+'_inst'+str(inst)+'.fits'
        else:
            mask_fpath = mask_fpaths[i]
        joint_masks[i] = fits.open(mask_fpath)['joint_mask_'+str(ifield)].data
        
        # if ff_estimates is None:
        #     ff_fpath = base_fluc_path+'/TM'+str(inst)+'/stackFF/ff_estimate_ifield'+str(ifield)+'_inst'+str(inst)+'_Jlim.fits'
        #     ff_estimates[i] = fits.open(ff_fpath)['ff_'+str(ifield)].data
        
        if show_plots:
            plot_map(joint_masks[i]*observed_ims[i], title='joint masks with ff')

            if ff_estimates is not None:
                plot_map(ff_estimates[i], title='ff estimate')
            

        B_ells[i] = cbps.B_ell
    
    if show_plots:
        plt.figure()
        for i, ifield in enumerate(ifield_list):
            plt.plot(cbps.Mkk_obj.midbin_ell, B_ells[i], label=cbps.ciber_field_dict[ifield], color='C'+str(i))
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
    
    if not use_ff_est:
        ff_estimates = None


    for i in range(len(ifield_list)):
            
        sigclip = sigma_clip_maskonly(observed_ims[i], previous_mask=joint_masks[i], sig=5)
        joint_masks[i] *= sigclip

    ff_estimates, inverse_Mkks, lb, \
        recovered_cls,recovered_dcls, masked_images,\
         masked_nls, masked_nls_noff, cls_intermediate = calculate_powerspec_quantities(cbps, observed_ims, joint_masks,\
                                                                                        shot_sigma_sb_maps, noise_models, ifield_list,\
                                                                                        include_inst_noise=True, include_photon_noise=True,\
                                                                                        inst=inst, inverse_Mkks=inverse_Mkks, \
                                                                                        n_mkk_sims=n_mkk_sims, n_split_mkk=n_split_mkk,\
                                                                                        n_FW_sims=n_FW_sims, n_FW_split=n_FW_split,\
                                                                                        ff_estimates=ff_estimates, use_true_ff=False, ff_stack_min=2, \
                                                                                        apply_masking=True, mkk_correct=True, \
                                                                                        beam_correct=True, B_ells=B_ells, show_plots=show_plots)


    return ff_estimates, inverse_Mkks, lb, recovered_cls, recovered_dcls, masked_images, masked_nls, masked_nls_noff, cls_intermediate, lb
    
   





