import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ciber_mocks import *
from plotting_fns import *
from mock_galaxy_catalogs import *
from integrate_cl_wtheta import *
from angular_2pcf import *
import pandas as pd
# import pyfftw
from numpy.fft import fftshift as fftshift
from numpy.fft import ifftshift as ifftshift
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2

def azimuthalAverage(image, ell_min=90, center=None, logbins=True, nbins=60, sterad_term=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    ell_min - the minimum multipole used to set range of multipoles
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    logbins - boolean True if log bins else uniform bins
    nbins - number of bins to use

             
    code adapted from https://www.astrobetter.com/blog/2010/03/03/fourier-transforms-of-images-in-python/
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if center is None:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]
    if sterad_term is not None:
        i_sorted /= sterad_term


    ell_max = ell_min*image.shape[0]/np.sqrt(0.5)
    
    if logbins:
        radbins = 10**(np.linspace(np.log10(ell_min), np.log10(ell_max), nbins+1))
    else:
        radbins = np.linspace(ell_min, ell_max, nbins+1)
    
    # convert multipole bins into pixel values
    radbins /= np.min(radbins)
    rbin_idxs = get_bin_idxs(r_sorted, radbins)

    rad_avg = np.zeros(nbins)
    rad_std = np.zeros(nbins)
    
    for i in range(len(rbin_idxs)-1):
        nmodes= len(i_sorted[rbin_idxs[i]:rbin_idxs[i+1]])
        rad_avg[i] = np.mean(i_sorted[rbin_idxs[i]:rbin_idxs[i+1]])
        rad_std[i] = np.std(i_sorted[rbin_idxs[i]:rbin_idxs[i+1]])/np.sqrt(nmodes)
        
    av_rbins = (radbins[:-1]+radbins[1:])/2

    return av_rbins, np.array(rad_avg), np.array(rad_std)


def azim_average_cl2d(ps2d, l2d, nbins=29, lbinedges=None, lbins=None, weights=None, logbin=False, verbose=False):
    
    if lbinedges is None:
        lmin = np.min(l2d[l2d!=0])
        lmax = np.max(l2d[l2d!=0])
        if logbin:
            lbinedges = np.logspace(np.log10(lmin), np.log10(lmax), nbins)
            lbins = np.sqrt(lbinedges[:-1] * lbinedges[1:])
        else:
            lbinedges = np.linspace(lmin, lmax, nbins)
            lbins = (lbinedges[:-1] + lbinedges[1:]) / 2

        lbinedges[-1] = lbinedges[-1]*(1.01)
        
    if weights is None:
        weights = np.ones(ps2d.shape)
        
    Cl = np.zeros(len(lbins))
    Clerr = np.zeros(len(lbins))
    Nmodes = np.zeros(len(lbins),dtype=int)
    Neffs = np.zeros(len(lbins))

    for i,(lmin, lmax) in enumerate(zip(lbinedges[:-1], lbinedges[1:])):
        sp = np.where((l2d>=lmin) & (l2d<lmax))
        p = ps2d[sp]
        w = weights[sp]

        Neff = compute_Neff(w)

        Cl[i] = np.sum(p*w) / np.sum(w)

        if verbose:
            print('sum weights:', np.sum(w))
        Clerr[i] = np.std(p) / np.sqrt(len(p))
        Nmodes[i] = len(p)
        Neffs[i] = Neff

    # print('Neffs: ', Neffs)
    # print('Nmodes', Nmodes)

    # print('Neff/Nmodes:', Neffs/Nmodes)
        
    return lbins, Cl, Clerr

def compute_Neff(weights):
    N_eff = np.sum(weights)**2/np.sum(weights**2)

    return N_eff

def compute_beam_correction_Mkk(psf, Mkk_obj):
    ''' This function computes the power spectrum of a beam as provided through a PSF template. The beam power spectrum
    can then be used to correct the autospectrum of a mock intensity map which is convolved with an instrument PSF''' 
    rb, Bl, Bl_std = get_power_spec(psf-np.mean(psf), lbinedges=Mkk_obj.binl, lbins=Mkk_obj.midbin_ell, return_Dl=False)
    B_ell = np.sqrt(Bl)/np.max(np.sqrt(Bl))
    return rb, B_ell

def compute_beam_correction(psf, nbins=60):
    ''' This function computes the power spectrum of a beam as provided through a PSF template. The beam power spectrum
    can then be used to correct the autospectrum of a mock intensity map which is convolved with an instrument PSF''' 
    rb, radprof_Bl, radstd_bl = compute_cl(psf, psf, nbins=nbins)
    B_ell = np.sqrt(radprof_Bl)/np.max(np.sqrt(radprof_Bl))
    return rb, B_ell

def compute_cross_spectrum(map_a, map_b, n_deg_across=2.0, sterad_a=True, sterad_b=True):
    ''' This function takes in two maps and computes their cross spectrum, with some optional unit conversions. If map_a=map_b, this is 
    just the autospectrum of the map.'''

    sterad_per_pix = (n_deg_across*(np.pi/180.)/map_a.shape[0])**2
    if sterad_a:
        ffta = fft2(map_a*sterad_per_pix)
    else:
        ffta = fft2(map_a)
    if sterad_b:
        fftb = fft2(map_b*sterad_per_pix)
    else:
        fftb = fft2(map_b)

    xspectrum = np.abs(ffta*np.conj(fftb)+fftb*np.conj(ffta))
    
    return np.fft.fftshift(xspectrum)

def compute_mkk_bl_corrected_powerspectra(srcmap, mask, Mkk_obj, inv_Mkk, B_ell, threshcut=None, verbose=False):
    lbins, cl_unmasked, cl_std_unmasked = get_power_spec(srcmap - np.mean(srcmap), lbinedges=Mkk_obj.binl, lbins=Mkk_obj.midbin_ell, return_Dl=False)
    unmasked_pix_mean = np.mean(srcmap[mask==1])
    masked_srcmap = srcmap*mask
    masked_srcmap[mask==1] -= unmasked_pix_mean
    if verbose:
        print('mean map value after subtraction is ', np.mean(masked_srcmap))
    
    if threshcut is not None:
        masked_srcmap[masked_srcmap > threshcut] = 0.
    
    lbins, cl_masked, cl_std_unmasked = get_power_spec(masked_srcmap, lbinedges=Mkk_obj.binl, lbins=Mkk_obj.midbin_ell, return_Dl=False)
    rectified_cl = np.dot(inv_Mkk.transpose(), cl_masked)

    rectified_cl /= B_ell**2

    
    return lbins, rectified_cl, masked_srcmap


def compare_2pcf_estimators(maps, imsize=256, sigma=None, nmaps=None, pixsize=7., n_ft_bins=200, ft_logbins=False, tc_logbins=False, n_tc_bins=64, log_tc_size=0.01, n_int_steps=200):
    
    if nmaps is None:
        nmaps = len(maps)
        
    Mkk = Mkk_bare(ell_min=150.*(1024./imsize), logbin=False, nbins=nbins, dimx=imsize, dimy=imsize, pixsize=pixsize)
    Mkk.check_precomputed_values(precompute_all=True, shift=True)
    
    zc_maps = np.array([x[:imsize, :imsize]-np.nanmean(x[:imsize,:imsize]) for x in maps])
    
    dft_cls = np.zeros((len(zc_maps), n_ft_bins))
    for i, zc_map in enumerate(zc_maps):
        
        l2b, cl_dft, clerr_dft = get_power_spec(zc_map, lbinedges=Mkk.binl, lbins=Mkk.midbin_ell)
    
        dft_cls[i] = cl_dft
        
        
    if tc_logbins:
        wthetas, var_wthetas, th_bins, covs, fz = compute_2pcf(binning='log', thetabin_size=log_tc_size, \
                                        skymaps=rdns, mask=None, nmaps=nmaps, \
                                        var_method='shot', npatch=1, verbose=False, \
                                        pix_units='deg', plot=True, thetamax=thetamax, pixel_size=pixsize, bin_slop=0.01)
    else:
        wthetas, var_wthetas, th_bins, covs, fz = compute_2pcf(binning='linear', nlinbin=n_tc_bins, \
                                        skymaps=rdns, mask=None, nmaps=nmaps, \
                                        var_method='shot', npatch=1, verbose=False, \
                                        pix_units='deg', plot=True, thetamax=thetamax, pixel_size=pixsize, bin_slop=0.01)


        
    
    cls_from_wthetas, wthetas_from_cls = [], []
        
    thetas = np.linspace(1e-4, thetamax, 100)
    
    for i in range(nmaps):
        wth_from_cl = c_ell_to_wtheta_hb(Mkk.midbin_ell, dft_cls[i], thetas, a=np.min(Mkk.binl), b=np.max(Mkk.binl), N=n_int_steps)
        wthetas_from_cls.append(np.abs(wth_from_cl))
        
        cl_from_wth = wtheta_to_c_ell_hb_sp(th_bins, wthetas[i], Mkk.midbin_ell, a=np.min(thetas)*np.pi/180., b=np.max(thetas)*np.pi/180., N=n_int_steps)
        cls_from_wthetas.append(cl_from_wth)
        
        
    return Mkk.midbin_ell, th_bins, cls_from_wthetas, dft_cls, wthetas, wthetas_from_cls


def allocate_fftw_memory(data_shape, n_blocks=2):

    ''' This function allocates empty aligned memory to be used for declaring FFT objects
    by pyfftw. The shape of the data, i.e. data_shape (including number of realizations as a dimension),
    can be arbitrarily defined depending on the task at hand.'''

    empty_aligned = []
    for i in range(n_blocks):
        empty_aligned.append(pyfftw.empty_aligned(data_shape, dtype='complex64'))

    return empty_aligned

def set_fftw_objects(empty_aligned=None, directions=['FFTW_BACKWARD'], threads=1, axes=(1,2), data_shape=None):
    
    if empty_aligned is None:
        empty_aligned = allocate_fftw_memory(data_shape, n_blocks=len(directions)+1)

    fftw_object_dict = dict({})

    for i in range(len(empty_aligned)-1):
        fftw_obj = pyfftw.FFTW(empty_aligned[i], empty_aligned[i+1], axes=axes, threads=threads, direction=directions[i], flags=('FFTW_MEASURE',))
        fftw_object_dict[directions[i]] = fftw_obj

    return fftw_object_dict, empty_aligned



def compute_cl(map_a, map_b=None, ell_min=90., nbins=60, sterad_term=None, sterad_a=True, sterad_b=True):

    ''' This function computes the angular power spectrum C_ell by 1) computing the cross spectrum of the images and
    2) radially averaging over multipole bins. If only one map is specified, then this calculates the auto power spectrum.

    Parameters
    ----------
    map_a : '~numpy.ndarray' of shape (Nx, Ny)
        One of potentially two input map for which cross spectrum is computed.
    map_b : `~numpy.ndarray' of shape (Nx, Ny), optional
        Second input map for which to compute cross spectrum with map_a. If left unspecified,
        the function computes auto spectrum of map_a. Default is 'None'.

    ell_min (float, default=90.): minimum mulitpole for which to compute angular cross spectrum, corresponds to largest mode of FOV
    
    nbins (int, default=60): number of bins to compute cross spectrum for. 
    
    sterad_term (float, optional, default=None): unit conversion term that is used to divide power spectrum in azimuthalAverage()
    
    sterad_a/sterad_b (bool, default=True): if True, multiplies map to convert from map proportional to 1/steradian to 1/pixel

    Returns
    -------
    
    rbins (np.array): radial bins for computed cross spectrum. note that these are multipole bins, not physical/comoving radius bins
    
    radprof (np.array): profile of cross power spectrum
    
    radstd (np.array): uncertainties on each binned power spectrum value determined by the number of available spatial modes. 
            In other words, this is the uncertainty from cosmic variance
    ''' 

    n_deg_across = 180./ell_min
    if map_b is None:

        xcorrs = compute_cross_spectrum(map_a, map_a, n_deg_across=n_deg_across, sterad_a=sterad_a, sterad_b=sterad_b)
    else:
        xcorrs = compute_cross_spectrum(map_a, map_b, n_deg_across=n_deg_across, sterad_a=sterad_a, sterad_b=sterad_b)
        
    rbins, radprof, radstd = azimuthalAverage(xcorrs, ell_min=ell_min, nbins=nbins, sterad_term=sterad_term)
    
    return rbins, radprof, radstd


def get_power_spectrum_2d(map_a, map_b=None, pixsize=7., verbose=False):
    '''
    calculate 2d cross power spectrum Cl
    
    Inputs:
    =======
    map_a: 
    map_b:map_b=map_a if no input, i.e. map_a auto
    pixsize:[arcsec]
    
    Outputs:
    ========
    l2d: corresponding ell modes
    ps2d: 2D Cl
    '''
    
    if map_b is None:
        map_b = map_a.copy()
        
    dimx, dimy = map_a.shape
    sterad_per_pix = (pixsize/3600/180*np.pi)**2
    V = dimx * dimy * sterad_per_pix
    
    ffta = np.fft.fftn(map_a*sterad_per_pix)
    fftb = np.fft.fftn(map_b*sterad_per_pix)
    ps2d = np.real(ffta * np.conj(fftb)) / V 
    ps2d = np.fft.ifftshift(ps2d)
    
    l2d = get_l2d(dimx, dimy, pixsize)

    return l2d, ps2d

def get_l2d(dimx, dimy, pixsize):
    lx = np.fft.fftfreq(dimx)*2
    ly = np.fft.fftfreq(dimy)*2
    lx = np.fft.ifftshift(lx)*(180*3600./pixsize)
    ly = np.fft.ifftshift(ly)*(180*3600./pixsize)
    ly, lx = np.meshgrid(ly, lx)
    l2d = np.sqrt(lx**2 + ly**2)
    
    return l2d


def get_power_spec(map_a, map_b=None, mask=None, pixsize=7., 
                   lbinedges=None, lbins=None, nbins=29, 
                   logbin=True, weights=None, return_full=False, return_Dl=False, verbose=False):
    '''
    calculate 1d cross power spectrum Cl
    
    Inputs:
    =======
    map_a: 
    map_b:map_b=map_a if no input, i.e. map_a auto
    mask: common mask for both map
    pixsize:[arcsec]
    lbinedges: predefined lbinedges
    lbins: predefined lbinedges
    nbins: number of ell bins
    logbin: use log or linear ell bin
    weights: Fourier weight
    return_full: return full output or not
    return_Dl: return Dl=Cl*l*(l+1)/2pi or Cl
    
    Outputs:
    ========
    lbins: 1d ell bins
    ps2d: 2D Cl
    Clerr: Cl error, calculate from std(Cl2d(bins))/sqrt(Nmode)
    Nmodes: # of ell modes per ell bin
    lbinedges: 1d ell binedges
    l2d: 2D ell modes
    ps2d: 2D Cl before radial binning
    '''

    if map_b is None:
        map_b = map_a.copy()

    if mask is not None:
        map_a = map_a*mask - np.mean(map_a[mask==1])
        map_b = map_b*mask - np.mean(map_b[mask==1])
    else:
        map_a = map_a - np.mean(map_a)
        map_b = map_b - np.mean(map_b)
        
    l2d, ps2d = get_power_spectrum_2d(map_a, map_b=map_b, pixsize=pixsize)
            
    lbins, Cl, Clerr = azim_average_cl2d(ps2d, l2d, nbins=nbins, lbinedges=lbinedges, lbins=lbins, weights=weights, logbin=logbin, verbose=verbose)
    
    if return_Dl:
        Cl = Cl * lbins * (lbins+1) / 2 / np.pi
        
    if return_full:
        return lbins, Cl, Clerr, Nmodes, lbinedges, l2d, ps2d
    else:
        return lbins, Cl, Clerr


def compute_snr(average, errors):
    ''' Given some array of values and errors on those values, this function computes the signal to noise ratio (SNR) in each bin as well as
        the summed SNR over elements''' 

    snrs_per_bin = average / errors
    total_sq = np.sum(snrs_per_bin**2)
    return np.sqrt(total_sq), snrs_per_bin

def cross_correlate_galcat_ciber(cibermap, galaxy_catalog, m_min=14, m_max=30, band='J', \
                         ihl_frac=0.0, magidx=5, zmin=-10, zmax=100, zidx=3):
    # convert galaxy catalog to binary map
    gal_map = make_galaxy_cts_map(galaxy_catalog, cibermap.shape, m_min=m_min, m_max=m_max, magidx=magidx, zmin=zmin, zmax=zmax, zidx=zidx)
    xcorr = compute_cross_spectrum(cibermap, gal_map)
    rbins, radprof, radstd = azimuthalAverage(xcorr)
    return rbins, radprof, radstd, xcorr

def dist(NAXIS):

    axis = np.linspace(-NAXIS/2+1, NAXIS/2, NAXIS)
    result = np.sqrt(axis**2 + axis[:,np.newaxis]**2)
    return np.roll(result, NAXIS/2+1, axis=(0,1))


def ensemble_power_spectra(beam_correction, full_maps=None, catalogs=None, mode='auto', \
                           zmin=0.0, zmax=1.0, nz_bins=3, nbins=30, savefig=False, m_lim_tracer=24, \
                          ncatalogs = 50, Nside = 1024., sterad_per_pix=True, pixscale=7.):
    

    ''' This function takes in maps and catalogs, as well as a beam correction, and calculates either map-map auto spectra, map-galaxy cross spectra,
    or galaxy-galaxy auto spectra (determined by mode).'''
    radprofs, radstds, labels = [], [], []


    sterad_per_pix = (pixscale/3600.)*np.pi/180.


    if mode == 'auto':
        print('Computing auto spectra..')
        profs = []
        for c in range(ncatalogs):
            print('c=', c)
            full_map = full_maps[c]

            # nW m^-2 sr^-1 --> nW m^-2 pix^-1 -- compute FT of maps, multiply by side length --> nW m^-2 Nside pix^-1
            # --- compute conjugate product --> nW^2 m^-4 Npix pix^-2
        
            # --- divide by sr/pix --> nW^2 m^-4 sr^-1 (Npix/pix) 
            # --- divide by Npix --> nW^2 m^-4 sr^-1. -- multiply by ell*(ell+1)/2*pi --> nW^2 m^-4 sr^-2
            
            rbin, radprof, radstd = compute_cl(full_map-np.mean(full_map), sterad_term=sterad_per_pix, nbins=nbins)

            profs.append(radprof[:-1])

        radprofs.append(np.mean(profs, axis=0)/beam_correction[:-1]**2 / Nside**2)
        radstds.append(np.std(profs, axis=0)/beam_correction[:-1]**2 / Nside**2)

        labels.append('CIBER x CIBER')
    
        f, yv, ystd = plot_radavg_xspectrum(rbin[:-1], radprofs=radprofs, raderrs=radstds, labels=labels, \
                           shotnoise=[True], \
                           add_shot_noise=[False], mode='auto',\
                           sn_npoints=10, titlestring='Auto Spectrum (with instrument noise + background)')
        if savefig:
            f.savefig('../figures/power_spectra/auto_spectrum_50_realizations_ciber_full.png', dpi=300, bbox_inches='tight')

    
    elif mode == 'cross':
        
        ngs = []
        print('Computing cross spectra..')
        zrange = np.linspace(zmin, zmax, nz_bins+1)

        for i in range(len(zrange)-1):
            print(zrange[i], zrange[i+1])
            profs = []
            cts_per_steradian = np.zeros(ncatalogs)
            for c in range(ncatalogs):
                print('c=', c)
                full_map = full_maps[c]
                catalog = catalogs[c]
                galcts = make_galaxy_cts_map(catalog, full_map.shape, 1, magidx=3, m_max=m_lim_tracer, zmin=zrange[i], zmax=zrange[i+1], zidx=2, normalize=False)
                cts_per_steradian[c] = np.sum(galcts)/(4*3.046e-4)
                # inputs have units of (nW m^-2 sr^-1, count sr^-1) --> (nW m^-2 pix^-1, count pix^-1) -- compute FT of maps
                # multiply by side length --> (nW m^-2 Nside pix^-1, count Nside pix^-1) --> compute conjugate product
                # --> nW m^-2 Npix pix^-2 

                # -- divide by sr/pix --> nW m^-2 sr^-1 (Npix / pix) -- divide by Npix ->
                # nW m^-2 sr^-1.
                rbin, radprof, radstd = compute_cl(full_map-np.mean(full_map), (galcts-np.mean(galcts))/np.mean(galcts), nbins=nbins, sterad_term=sterad_per_pix)
                rbin, radprof, radstd = compute_cl(full_map-np.mean(full_map), (galcts-np.mean(galcts))/np.mean(galcts), nbins=nbins, sterad_term=sterad_per_pix)

                profs.append(radprof[:-1])
            print(np.mean(cts_per_steradian))
            ngs.append(np.mean(cts_per_steradian))
            profs = np.array(profs)
            radprofs.append(np.mean(profs, axis=0)/beam_correction[:-1] / Nside**2)
            radstds.append(np.std(profs, axis=0)/beam_correction[:-1] / Nside**2)
            labels.append('CIBER x Gal ('+str(np.round(zrange[i], 3))+'<z<'+str(np.round(zrange[i+1], 3))+')')

        f, yv, ystd = plot_radavg_xspectrum(rbin[:-1], radprofs=radprofs, raderrs=radstds, labels=labels, \
                           shotnoise=[True for x in range(len(zrange)-1)], \
                           add_shot_noise=[False for x in range(len(zrange)-1)], mode='cross',\
                           sn_npoints=10, titlestring='Cross Spectra, $m_{tracer}^{lim}=$'+str(m_lim_tracer)+' (with noise/background)')
    
        if savefig:
            f.savefig('../figures/power_spectra/cross_spectrum_50_realizations_ciber_sourcemap_mlim='+str(m_lim_tracer)+'_full.png', dpi=300, bbox_inches='tight')
        
        
    elif mode=='auto_gal':
        print('Computing auto spectra for counts map..')
        zrange = np.linspace(zmin, zmax, nz_bins+1)
        ngs = []
        for i in range(len(zrange)-1):
            print(zrange[i], zrange[i+1])
            profs = []
            cts_per_steradian = np.zeros(ncatalogs)

            for c in range(ncatalogs):
                print('c=', c)

                full_map = full_maps[c]
                catalog = catalogs[c]
                
                # (counts / pix) / (mean counts / pix) --> unitless 
                # (pix) sr^-1 -->  unitless -- compute FT of maps, multiply by side length --> Nside
                # --- compute conjugate product --> Npix --- divide by sr/pix --> sr^-1 (Npix/pix) 
                # --- divide by Npix --> sr^-1. 
                galcts = make_galaxy_cts_map(catalog, full_map.shape, 1, magidx=3, m_max=m_lim_tracer, zmin=zrange[i], zmax=zrange[i+1], zidx=2, normalize=False)
                cts_per_steradian[c] = np.sum(galcts)/(4*3.04617e-4)
#                 print('galcts (counts per steradian):', np.sum(galcts)/(4*3.04617e-4))
                # (counts / pix, counts/pix) --> (counts Nside / pix, counts Nside / pix) --> (counts^2 Npix / pix^2)
                # --> counts^2 sr^-1 (Npix/pix) --- divide by Npix --> counts^2 sr^-1
                # --> should have units of Npix after xcorr 

                rbin, radprof, radstd = compute_cl((galcts-np.mean(galcts))/np.mean(galcts), nbins=nbins, sterad_term=sterad_per_pix)
            
                profs.append(radprof[:-1])
            print(np.mean(cts_per_steradian))
            ngs.append(np.mean(cts_per_steradian))
            radprofs.append(np.mean(profs, axis=0)/Nside**2)
            radstds.append(np.std(profs, axis=0)/Nside**2)

            labels.append('Gal x Gal ('+str(np.round(zrange[i], 3))+'<z<'+str(np.round(zrange[i+1], 3))+')')
    
        f, yv, ystd = plot_radavg_xspectrum(rbin[:-1], radprofs=radprofs, raderrs=radstds, labels=labels, \
                           shotnoise=[True for x in range(len(zrange)-1)], \
                           add_shot_noise=[False for x in range(len(zrange)-1)], mode='auto',\
                           sn_npoints=10, titlestring='Galaxy Counts Auto Spectrum, $m_{lim}$='+str(m_lim_tracer))
        if savefig:
            f.savefig('../figures/power_spectra/gal_counts_auto_spectrum_50_realizations_ciber.png', dpi=300, bbox_inches='tight')

        

    if mode=='cross' or mode=='auto_gal':
        return f, yv, ystd, radprofs, radstds, rbin, ngs
    else:
        return f, yv, ystd, radprofs, radstds, rbin

def get_bin_idxs(arr, bins):
    i=0
    maxval = np.max(arr)
    idxs = [0]
    for ind, val in enumerate(arr):
        if val-bins[i+1]>=0:
            idxs.append(ind)
            if i==len(bins)-1:
                return idxs
            else:
                i+=1
        elif val == maxval:
            idxs.append(ind)
            return idxs


def integrated_xcorr_multiple_redshifts(ihl_frac=0.1, \
    gal_maxmag=22, 
    zmin=0.0, 
    zmax=5, 
    nbin=10, 
    nsrc=100,
    ifield=4,
    m_min=10, 
    m_max=27,
    inst=1, 
    lmin=90):
    
    wints = []
    cmock = ciber_mock()
    zrange = np.linspace(zmin, zmax, nbin)
    if ihl_frac > 0:
        full, srcs, noise, ihl, gal_cat = cmock.make_ciber_map(ifield, m_min, gal_maxmag, band=inst, nsrc=nsrc, ihl_frac=ihl_frac)
    else:
        full, srcs, noise, gal_cat = cmock.make_ciber_map(ifield, m_min, gal_maxmag, band=inst, nsrc=nsrc, ihl_frac=ihl_frac)
        
    for i in range(len(zrange)-1):
        rb, radprof, radstd, xcorr = cross_correlate_galcat_ciber(full, gal_cat, m_max=gal_maxmag, zmin=zrange[i], zmax=zrange[i+1], zidx=3)
        wints.append(integrate_C_l(rb*lmin, radprof))
    
    zs = 0.5*(zrange[:-1]+zrange[1:])
    
    return wints, zs

def integrate_w_theta(ls, w, weights=None):
    ''' Integrate potentially weighted angular correlation function. If no weights are provided, then inverse theta weighting is used.'''

    thetas = np.pi/ls
    dthetas = thetas[:-1]-thetas[1:]
    w_integrand = 0.5*(w[:-1]+w[1:])
    if weights is None: # then use inverse theta weighting
        avthetas = 0.5*(thetas[:-1]+thetas[1:])
        weights = 1./avthetas
        
    w_integrand *= weights
    w_integrand *= dthetas
    return np.sum(w_integrand)


def integrate_C_l(ls, C, weights=None):
    ''' Integrate potentially weighted angular correlation function. If no weights are provided, then inverse multipole weighting is used.'''

    dls = ls[:-1]-ls[1:]
    C_integrand = 0.5*(C[:-1]+C[1:])
    if weights is None:
        weights = 0.5*(ls[1:]+ls[:-1])
        
    C_integrand *= weights
    C_integrand *= dls
    return np.sum(C_integrand)


def knox_spectra(radprofs_auto, radprofs_cross=None, radprofs_gal=None, \
                 ngs=None, fsky=0.0000969, lmin=90., Nside=1024, mode='auto'):
    
    ''' 
    Compute dC_ell and corresponding cross terms/total signal to noise for a given auto or cross power spectrum. This uses
    the Knox formula from Knox (1995). I think in general this will underestimate the total error, because it assumes a gaussian 
    beam and that errors from different terms are uncorrelated and have uniform variance.
    '''

    sb_intensity_unit = u.nW/u.m**2/u.steradian
    npix = Nside**2
    pixel_solidangle = 49*u.arcsecond**2
    print(len(rbin))
    ells = rbin[:-1]*lmin
    d_ells = ells[1:]-ells[:-1]
    
    mode_counting_term = 2./(fsky*(2*ells+1))
    sb_sens_perpix = [33.1, 17.5]*sb_intensity_unit
    
    beam_term = np.exp((pixel_solidangle.to(u.steradian).value)*ells**2)
    noise = (4*np.pi*fsky*u.steradian)*sb_sens_perpix[band]**2/npix
    
    cl_noise = noise*beam_term
    
    if mode=='auto':
        dCl_sq = mode_counting_term*((radprofs_auto[0]) + cl_noise.value)**2
        snr_sq = (radprofs_auto[0])**2 / dCl_sq
        
        return snr_sq, dCl_sq, ells
    
    elif mode=='cross':
        
        snr_sq_cross_list, list_of_crossterms = [], []

         
        for i in range(len(radprofs_cross)):
            print(len(radprofs_auto[0]), len(cl_noise.value), len(radprofs_gal[i]), len(mode_counting_term))
            dCl_sq = mode_counting_term*((radprofs_cross[i])**2 +(radprofs_auto[0] + cl_noise.value)*(radprofs_gal[i] +ngs[i]**(-1)))
            snr_sq_cross = (radprofs_cross[i])**2 / dCl_sq
            snr_sq_cross_list.append(snr_sq_cross)

            cross_terms = [radprofs_cross[i]**2, radprofs_auto[0]*radprofs_gal[i], radprofs_auto[0]*ngs[i]**(-1), cl_noise.value*radprofs_gal[i], cl_noise.value*ngs[i]**(-1)]
            list_of_crossterms.append(cross_terms)

        return snr_sq_cross_list, dCl_sq, ells, list_of_crossterms



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


''' The bottom two functions in principle can be used to calculate effect of IHL/completeness on observable cross power spectrum, but
they haven't really been used to date and likely aren't fully consistent with the current code. To be updated in the future '''  
def xcorr_varying_ihl(ihl_min_frac=0.0, ihl_max_frac=0.5, nbins=10, nsrc=100, m_min=14, m_max=20, gal_comp=21, ifield=4, inst=1):
    radprofs = []
    radstds = []
    cmock = ciber_mock()
    ihl_range = np.linspace(ihl_min_frac, ihl_max_frac, nbins)
    for i, ihlfrac in enumerate(ihl_range):
        if i==0:
            full, srcs, noise, cat = cmock.make_ciber_map(ifield, m_min, m_max, band=inst, nsrc=nsrc, ihl_frac=ihlfrac)
            gal_map = make_galaxy_cts_map(cat, full.shape, m_min, m_max=gal_comp, magidx=5) # cut off galaxy catalog at 20th mag
        else:
            full, srcs, noise, ihl, cat = cmock.make_ciber_map(ifield, m_min, m_max, mock_cat=cat, band=inst, nsrc=nsrc, ihl_frac=ihlfrac)
         
        xcorr = compute_cross_spectrum(full, gal_map)
        rb, radprof, radstd = azimuthalAverage(xcorr)
        radprofs.append(radprof)
        radstds.append(radstd)
        
    return ihl_range, rb, radprofs, radstds


def xcorr_varying_galcat_completeness(ihl_frac=0.1, compmin=18, compmax=22, nbin=10, nsrc=100, ifield=4, m_min=10, inst=1):
    radprofs, radstds = [], []
    cmock = ciber_mock()
    comp_range = np.linspace(compmin, compmax, nbin)
    full, srcs, noise, ihl, gal_cat = cmock.make_ciber_map(ifield, m_min, 25, band=inst, nsrc=nsrc, ihl_frac=ihl_frac)

    for i, comp in enumerate(comp_range):
        gal_map = make_galaxy_cts_map(gal_cat, full.shape, m_min, m_max=comp)        
        xcorr = compute_cross_spectrum(full, gal_map)
        rb, radprof, radstd = azimuthalAverage(xcorr)
        radprofs.append(radprof)
        radstds.append(radstd)
        
    return comp_range, rb, radprofs, radstds

