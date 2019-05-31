import numpy as np


def get_bin_idxs(arr, bins):
    i=0
    maxval = np.max(arr)
    idxs = [0]
    for ind, val in enumerate(arr):
        if val-bins[i+1]>=0:
            idxs.append(ind)
            if i==len(bins)-1:
                print 'i='+str(i)
                return idxs
            else:
                i+=1
        elif val == maxval:
            print('reached end of array')
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
    print 'Binning between l='+str(lmin)+' and l='+str(lmax)
    
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
        rad_avg.append(np.mean(i_sorted[rbin_idxs[i]:rbin_idxs[i+1]]))
        rad_std.append(np.std(i_sorted[rbin_idxs[i]:rbin_idxs[i+1]]))
        
        
    av_rbins = (radbins[:-1]+radbins[1:])/2

    return av_rbins, np.array(rad_avg), np.array(rad_std)
