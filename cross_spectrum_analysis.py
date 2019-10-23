import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ciber_mocks import *


def azimuthalAverage(image, lmin=90, center=None, logbins=True, nbins=60, sterad_term=None):
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
    if sterad_term is not None:
        i_sorted /= sterad_term


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

def compute_beam_correction(psf):
    ''' This function computes the power spectrum of a beam as provided through a PSF template. The beam power spectrum
    can then be used to correct the autospectrum of a mock intensity map which is convolved with an instrument PSF''' 
    rb, radprof_Bl, radstd_bl = compute_cl(psf, psf)
    B_ell = np.sqrt(radprof_Bl)/np.max(np.sqrt(radprof_Bl))
    return rb, B_ell

def compute_cross_spectrum(map_a, map_b, n_deg_across=2.0, sterad_a=True, sterad_b=True):
    sterad_per_pix = (n_deg_across*(np.pi/180.)/map_a.shape[0])**2
    if sterad_a:
        ffta = np.fft.fft2(map_a*sterad_per_pix)
    else:
        ffta = np.fft.fft2(map_a)
    if sterad_b:
        fftb = np.fft.fft2(map_b*sterad_per_pix)
    else:
        fftb = np.fft.fft2(map_b)

    xspectrum = np.abs(ffta*np.conj(fftb)+fftb*np.conj(ffta))
    
    return np.fft.fftshift(xspectrum)

def compute_cl(mapa, mapb=None, lmin=90.):

    ''' This function computes the angular power spectrum C_ell by 1) computing the cross spectrum of the images and
    2) radially averaging over multipole bins ''' 
    n_deg_across = 180./lmin
    if mapb is None:
        xcorr = compute_cross_spectrum(mapa, mapa, n_deg_across=n_deg_across)
    else:
        xcorr = compute_cross_spectrum(mapa, mapb, n_deg_across=n_deg_across)
        
    rbins, radprof, radstd = azimuthalAverage(xcorr, lmin=lmin)
    
    return rbins, radprof, radstd


def compute_mode_coupling(mask, ell_min=90., nphases=50, logbins=True, nbins=60, ps_amplitude=100.0):
    ''' If there is masking done on a given observation due to incomplete coverage or bright sources, 
    this function will compute the mode coupling matrix, which estimates how that masking impacts the 
    computed power spectrum. ''' 
    ell_max = ell_min*np.sqrt(2*(mask.shape[0]/2)**2)
    
    if logbins:
        radbins = 10**(np.linspace(np.log10(ell_min), np.log10(ell_max), nbins))
    else:
        radbins = np.linspace(lmin, lmax, nbins+1)
        
    print 'ell bins:'
    print radbins
    
    Mkk = np.zeros((radbins.shape[0], radbins.shape[0]))
    sigma_Mkk = np.zeros((radbins.shape[0], radbins.shape[0]))
        
    for i, radbin in enumerate(radbins):
        ps = np.zeros_like(radbins)
        ps[i] = ps_amplitude
        grfs, _ = grf_mkk(nphases, size=mask.shape[0], ps=ps, ell_sampled=radbins)

        masked_grfs = grfs*mask
        masked_ps = compute_cross_spectrum(masked_grfs, masked_grfs)
        
        norm_radavs = []
        for j, spec in enumerate(masked_ps):
            _, norm_radav, _ = azimuthalAverage(spec, nbins=nbins)
            norm_radavs.append(norm_radav)
        norm_radavs = np.array(norm_radavs)   
                
        Mkk[i,:] = np.mean(norm_radavs, axis=0)
        sigma_Mkk[i,:] = np.std(norm_radavs, axis=0)
        
        plt.figure()
        plt.title('$\\ell=$'+str(np.round(radbin, 2)))
        plt.imshow(grfs[0]*mask)
        plt.colorbar()
        plt.show()
        
    return Mkk, sigma_Mkk


def cross_correlate_galcat_ciber(cibermap, galaxy_catalog, m_min=14, m_max=30, band='J', \
                         ihl_frac=0.0, magidx=5, zmin=-10, zmax=100, zidx=3):
    # convert galaxy catalog to binary map
    gal_map = make_galaxy_binary_map(galaxy_catalog, cibermap, m_min=m_min, m_max=m_max, magidx=magidx, zmin=zmin, zmax=zmax, zidx=zidx)
    xcorr = compute_cross_spectrum(cibermap, gal_map)
    rbins, radprof, radstd = azimuthalAverage(xcorr)
    return rbins, radprof, radstd, xcorr

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

def grf_mkk(n_samples, size = 100, ps=None, ell_sampled=None):

    grfs = np.zeros((n_samples, size, size))
    noise = np.fft.fft2(np.random.normal(size = (n_samples, size, size)))
    amplitude = np.zeros((size, size))
    for i, sx in enumerate(fftIndgen(size)):
        amplitude[i,:] = Pk2_mkk(sx, np.array(fftIndgen(size)), ps=ps, ell_sampled=ell_sampled, size=size)
    grfs = np.fft.ifft2(noise * amplitude, axes=(-2,-1))
    
    return grfs.real, np.array(noise*amplitude)


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
        
    for i in xrange(len(zrange)-1):
        rb, radprof, radstd, xcorr = cross_correlate_galcat_ciber(full, gal_cat, m_max=gal_maxmag, zmin=zrange[i], zmax=zrange[i+1], zidx=3)
        wints.append(integrate_C_l(rb*lmin, radprof))
    
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


def make_galaxy_binary_map(cat, refmap, inst, m_min=14, m_max=30, magidx=2, zmin=0, zmax=100, zidx=None, normalize=True):
    gal_map = np.zeros_like(refmap)

    if isinstance(cat, pd.DataFrame): # real catalogs read in as pandas dataframes
    
        catalog = cat.loc[(cat['x'+str(inst)]>0)&(cat['x'+str(inst)]<refmap.shape[0])&(cat['y'+str(inst)]>0)&(cat['y'+str(inst)]<refmap.shape[0]) &\
                         (cat[magidx]<m_max)&(cat[magidx]>m_min)&(cat[zidx]>zmin)&(cat[zidx]<zmax)]

        for index, src in catalog.iterrows():
            gal_map[int(src['x'+str(inst)]), int(src['y'+str(inst)])] += 1
   
    else:
        if zidx is not None:
            if magidx is None:
                cat = np.array([src for src in cat if src[0]<refmap.shape[0] and src[1]<refmap.shape[1] and src[zidx]>zmin and src[zidx]<zmax])
            else:
                cat = np.array([src for src in cat if src[0]<refmap.shape[0] and src[1]<refmap.shape[1]\
                and src[magidx]>m_min and src[magidx]<m_max and src[zidx]>zmin and src[zidx]<zmax])
        else:
            if magidx is None:
                cat = np.array([src for src in cat if src[0]<refmap.shape[0] and src[1]<refmap.shape[1]])
            else:
                cat = np.array([src for src in cat if src[0]<refmap.shape[0] and src[1]<refmap.shape[1]\
                and src[magidx]>m_min and src[magidx]<m_max])

        for src in cat:
            gal_map[int(src[0]),int(src[1])] +=1.

    if normalize:
        gal_map /= np.mean(gal_map)
        gal_map -= 1.
    
    return gal_map

def Pk2_mkk(sx, sy, ps, ell_sampled=None, pixsize=3.39e-5, size=512.0, lmin=90):
    ells = np.sqrt((sx**2+sy**2))*lmin
    idx1 = np.array([np.abs(ell_sampled-ell).argmin() for ell in ells])
    return ps[idx1]

def psf_large(psf_template, mapdim=1024):
    ''' all this does is place the 40x40 PSF template at the center of a full map 
    so one can compute the FT and get the beam correction for appropriate ell values
    this assumes that beyond the original template, the beam correction will be 
    exactly unity, which is fair enough for our purposes'''
    psf_temp = np.zeros((mapdim, mapdim))
    psf_temp[mapdim/2 - 20:mapdim/2+21, mapdim/2-20:mapdim/2+21] = psf_template
    psf_temp /= np.sum(psf_temp)
    return psf_temp
    
def xcorr_varying_ihl(ihl_min_frac=0.0, ihl_max_frac=0.5, nbins=10, nsrc=100, m_min=14, m_max=20, gal_comp=21, ifield=4, inst=1):
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


def xcorr_varying_galcat_completeness(ihl_frac=0.1, compmin=18, compmax=22, nbin=10, nsrc=100, ifield=4, m_min=10, inst=1):
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

def xspec_gal_intensity_map(intensity_map, galaxy_cat, beam_correction=None, m_max=None, inst=0, magidx=3):
    gmap = make_galaxy_binary_map(galaxy_cat, intensity_map, inst=inst, magidx=magidx, m_max=m_max)
    if (gmap>100).any():
        gmap[gmap > 100] = 0
    rbins, radprof, radstd = compute_cl(intensity_map, gmap)
    if beam_correction is not None:
        radprof /= np.sqrt(beam_correction)
    
    return rbins, radprof, radstd, gmap



