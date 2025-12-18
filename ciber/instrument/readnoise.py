from numpy.fft import fftshift as fftshift
from numpy.fft import ifftshift as ifftshift
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
import pyfftw
import numpy as np


def generate_readnoise_realizations(c_ell_2d, cal=None, ifield=4, n_realization=1, imdim=1024, ndof=2, pixscale=7., fftw_object_dict=None, empty_aligned=None):


    assert c_ell_2d.shape[0] == imdim # need these to be the same size

    cl2din = c_ell_2d*np.random.chisquare(ndof, (imdim, imdim))/ndof

    if fftw_object_dict is not None:

        noise = fftw_object_dict['FFTW_FORWARD'](np.random.normal(0, 1, (n_realization, imdim, imdim)))
        maps = fftw_object_dict['FFTW_BACKWARD'](fftshift(np.sqrt(cl2din))*noise)

    else:

        maps = ifft2(np.sqrt(cl2din)*fft2(np.random.normal(0, 1, (n_realization, imdim, imdim))))

        
    div_fac = pixscale/3600.0*np.pi/180.0

    maps /= div_fac

    readnoise_realizations = maps.real/(np.abs(maps.real))*np.abs(maps)
    
    # read noise realizations output in units of ADU/frame, so calibration factor needs to be applied to get 
    # into intensity units
    
    if cal is not None:
        print('using calibration factor from ifield', ifield)
        print(cal[ifield])
        readnoise_realizations /= cal[ifield]
        
    return readnoise_realizations
