import numpy as np
from ciber_mocks import *


flight=40030 
inst=1  #inst=1-I band magnitude,2-H band magnitude
ifield=4  #use the PSF in which CIBER field? 4-elat10,5-elat30,6-BootesB,7-BootesA,8-SWIRE
field='XMM' # which HSC field?
m_min=10 
m_max=27 # magnitude limits
usePSF=False # if false, don't convolve with PSF, just put all the light in the central pixel
nbin = 1

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


def compute_cross_spectrum(map_a, map_b):
    ffta = np.fft.fft2(map_a)
    fftb = np.fft.fft2(map_b)
    xspectrum = np.abs(ffta*np.conj(fftb)+fftb*np.conj(ffta))
    return np.fft.fftshift(xspectrum)

def azimuthalAverage(image, lmin=90, center=None, logbins=True, nbins=60):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    lmin - the minimum multipole used to set range of multipoles
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    logbins - boolean True if log bins else uniform bins
    nbins - number of bins to use
             
    code adapted from https://www.astrobetter.com/blog/2010/03/03/fourier-transforms-of-images-in-python/
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]
    
    lmax = lmin*np.sqrt(0.5*image.shape[0]**2)
    
    if logbins:
        radbins = 10**(np.linspace(np.log10(lmin), np.log10(lmax), nbins+1))
    else:
        radbins = np.linspace(lmin, lmax, nbins+1)
    
    # convert multipole bins into pixel values
    radbins /= np.min(radbins)
        
    
    rbin_idxs = get_bin_idxs(r_sorted, radbins)
    rad_avg = []
    rad_std = []
    for i in xrange(len(rbin_idxs)-1):
        nmodes= len(i_sorted[rbin_idxs[i]:rbin_idxs[i+1]])
        rad_avg.append(np.mean(i_sorted[rbin_idxs[i]:rbin_idxs[i+1]]))
        rad_std.append(np.std(i_sorted[rbin_idxs[i]:rbin_idxs[i+1]])/np.sqrt(nmodes))
        
        
    av_rbins = (radbins[:-1]+radbins[1:])/2

    return av_rbins, np.array(rad_avg), np.array(rad_std)


def cross_correlate_galcat_ciber(cibermap, galaxy_catalog, m_min=14, m_max=30, band='J', \
                         ihl_frac=0.0, magidx=5, zmin=-10, zmax=100, zidx=3):
    # convert galaxy catalog to binary map
    gal_map = make_galaxy_binary_map(galaxy_catalog, cibermap, m_min=m_min, m_max=m_max, magidx=magidx, zmin=zmin, zmax=zmax, zidx=zidx)
    xcorr = compute_cross_spectrum(cibermap, gal_map)
    rbins, radprof, radstd = azimuthalAverage(xcorr)
    return rbins, radprof, radstd, xcorr




def xcorr_varying_ihl(ihl_min_frac=0.0, ihl_max_frac=0.5, nbins=10, nsrc=100, m_min=14, m_max=20, gal_comp=21):
    radprofs = []
    radstds = []
    cmock = ciber_mock()
    ihl_range = np.linspace(ihl_min_frac, ihl_max_frac, nbins)
    for i, ihlfrac in enumerate(ihl_range):
        if i==0:
            full, srcs, noise, cat = cmock.make_ciber_map(ifield, m_min, m_max, band=inst, nsrc=nsrc, ihl_frac=ihlfrac)
            gal_map = make_galaxy_binary_map(cat, full, m_min, m_max=gal_comp, magidx=5) # cut off galaxy catalog at 20th mag
        else:
            full, srcs, noise, ihl, cat = cmock.make_ciber_map(ifield, m_min, m_max, mock_cat=cat, band=inst, nsrc=nsrc, ihl_frac=ihlfrac)
         
        xcorr = compute_cross_spectrum(full, gal_map)
        rb, radprof, radstd = azimuthalAverage(xcorr)
        radprofs.append(radprof)
        radstds.append(radstd)
        
    return ihl_range, rb, radprofs, radstds


def xcorr_varying_galcat_completeness(ihl_frac=0.1, compmin=18, compmax=22, nbin=10, nsrc=100):
    radprofs, radstds = [], []
    cmock = ciber_mock()
    comp_range = np.linspace(compmin, compmax, nbin)
    full, srcs, noise, ihl, gal_cat = cmock.make_ciber_map(ifield, m_min, 25, band=inst, nsrc=nsrc, ihl_frac=ihl_frac)

    for i, comp in enumerate(comp_range):
        gal_map = make_galaxy_binary_map(gal_cat, full, m_min, m_max=comp)        
        xcorr = compute_cross_spectrum(full, gal_map)
        rb, radprof, radstd = azimuthalAverage(xcorr)
        radprofs.append(radprof)
        radstds.append(radstd)
        
    return comp_range, rb, radprofs, radstds


def integrated_xcorr_multiple_redshifts(ihl_frac=0.1, \
    gal_maxmag=22, \
    zmin=0.0, \
    zmax=5, \
    nbin=10, \
    nsrc=100):
    
    wints = []
    cmock = ciber_mock()
    zrange = np.linspace(zmin, zmax, nbin)
    if ihl_frac > 0:
        full, srcs, noise, ihl, gal_cat = cmock.make_ciber_map(ifield, m_min, 25, band=inst, nsrc=nsrc, ihl_frac=ihl_frac)
    else:
        full, srcs, noise, gal_cat = cmock.make_ciber_map(ifield, m_min, 25, band=inst, nsrc=nsrc, ihl_frac=ihl_frac)
        
    for i in xrange(len(zrange)-1):
        rb, radprof, radstd, xcorr = cross_correlate_galcat_ciber(full, gal_cat, m_max=gal_maxmag, zmin=zrange[i], zmax=zrange[i+1], zidx=3)
        # wints.append(integrate_w_theta(rb, radprof))
        wints.append(integrate_C_l(rb*90., radprof))
    
    zs = 0.5*(zrange[:-1]+zrange[1:])
    
    return wints, zs

def integrate_w_theta(ls, w, weights=None):
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
    dls = ls[:-1]-ls[1:]
    C_integrand = 0.5*(C[:-1]+C[1:])
    if weights is None: # then use inverse theta weighting
        weights = 0.5*(ls[1:]+ls[:-1])
        
    C_integrand *= weights
    C_integrand *= dls
    return np.sum(C_integrand)





