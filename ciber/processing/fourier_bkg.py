import numpy as np
import matplotlib 
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import gaussian_filter


def precomp_fourier_templates(dimx, dimy, mask=None, n_terms=1,\
                                    quad_offset=False, x0s=None, x1s=None, y0s=None, y1s=None, \
                                    with_gradient=False):
    
    X1, X2 = np.mgrid[:dimx, :dimy]
    
    templates = make_fourier_templates(dimx, dimy, n_terms)
    
    for nx in range(n_terms):
        
        for ny in range(n_terms):
        
            for k in range(4):
                
                reshape_temp = np.reshape(templates[nx, ny, k], (dimx*dimy, 1))
                if nx==0 and ny==0 and k==0:
                    X = reshape_temp
                else:
                    X = np.hstack( (reshape_temp, X))
                    
    if with_gradient:
        print('Adding gradient parameters..')
        X = np.hstack( (np.reshape(X1, (dimx*dimy, 1)) , np.hstack( (np.reshape(X2, (dimx*dimy, 1)), X)) ))


    if quad_offset:
        print('Adding quad offsets..')
        if x0s is None:
            x0s = [0, 0, 512, 512]
            x1s = [512, 512, 1024, 1024]
            y0s = [0, 512, 0, 512]
            y1s = [512, 1024, 512, 1024]
        for q in range(4):
            mquad = np.zeros((dimx, dimy))
            mquad[x0s[q]:x1s[q], y0s[q]:y1s[q]] = 1.
            mquad_rav = np.expand_dims(mquad.ravel(), axis=1)
            X = np.hstack(   ( mquad_rav , X ))
                    
    if mask is not None:
        mask_rav = np.reshape(mask, (dimx*dimy)).astype(bool)
        X_cut = X[mask_rav,:]
        dot1 = np.dot( np.linalg.pinv(np.dot(X_cut.transpose(), X_cut)), X_cut.transpose())
        return dot1, X, mask_rav
    
    else:
        dot1 = np.dot( np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose())
                    
        return dot1, X


def make_fourier_templates(N, M, n_terms, show_templates=False, psf_fwhm=None, shift=False, x_max_pivot=None, scale_fac=None, return_idxs=False):
        
    """
    
    Given image dimensions and order of the series expansion, generates a set of 2D fourier templates.

    Parameters
    ----------

    N : int
        length of image
    M : int
        width of image
    n_terms : int
        Order of Fourier expansion for templates. the number of templates (currently) scales as 2*n_terms^2
    
    show_templates : bool, optional
        if True, plots the array of templates. Default is False.
    
    psf_fwhm : float, optional
        Observation PSF full width at half maximum (FWHM). This can be used to pre-convolve templates for background modeling 
        Default is 'None'.

    x_max_pivot : float, optional
        Indicating pixel coordinate for boundary of FOV in each dimension. Default is 'None'.

    return_idxs : bool, optional
        If True, returns mesh grids of Fourier component indices for x and y. 
        Default is False.

    Returns
    -------
    
    templates : `numpy.ndarray' of shape (n_terms, n_terms, 4, N, M)
        Contains 2D Fourier templates for truncated series

    """

    print(n_terms, N, M)

    # if type(N)==float:
    #     print('Changing N = ', N, 'to an int..')
    N = int(N)
    # if type(M)==float:
    #     print('Changing M = ', M, 'to an int..')
    M = int(M)
    # if type(n_terms)==float:
    #     print('Changing n_terms = ', n_terms, 'to an int..')
    n_terms = int(n_terms)


    templates = np.zeros((n_terms, n_terms, 4, N, M))
    if scale_fac is None:
        scale_fac = 1.

    x = np.arange(N)
    y = np.arange(M)

    meshkx, meshky = np.meshgrid(np.arange(n_terms), np.arange(n_terms))
    
    meshx, meshy = np.meshgrid(x, y)
        
    xtemps_cos, ytemps_cos, xtemps_sin, ytemps_sin = [np.zeros((n_terms, N, M)) for x in range(4)]

    N_denom = N
    M_denom = M

    if x_max_pivot is not None:
        N_denom = x_max_pivot
        M_denom = x_max_pivot

    for n in range(n_terms):

        # modified series
        if shift:
            xtemps_sin[n] = np.sin((n+1-0.5)*np.pi*meshx/N_denom)
            ytemps_sin[n] = np.sin((n+1-0.5)*np.pi*meshy/M_denom)
        else:
            xtemps_sin[n] = np.sin((n+1)*np.pi*meshx/N_denom)
            ytemps_sin[n] = np.sin((n+1)*np.pi*meshy/M_denom)
        
        xtemps_cos[n] = np.cos((n+1)*np.pi*meshx/N_denom)
        ytemps_cos[n] = np.cos((n+1)*np.pi*meshy/M_denom)
    
    for i in range(n_terms):
        for j in range(n_terms):

            if psf_fwhm is not None: # if beam size given, convolve with PSF assumed to be Gaussian
                templates[i,j,0,:,:] = gaussian_filter(xtemps_sin[i]*ytemps_sin[j], sigma=psf_fwhm/2.355)
                templates[i,j,1,:,:] = gaussian_filter(xtemps_sin[i]*ytemps_cos[j], sigma=psf_fwhm/2.355)
                templates[i,j,2,:,:] = gaussian_filter(xtemps_cos[i]*ytemps_sin[j], sigma=psf_fwhm/2.355)
                templates[i,j,3,:,:] = gaussian_filter(xtemps_cos[i]*ytemps_cos[j], sigma=psf_fwhm/2.355)
            else:
                templates[i,j,0,:,:] = xtemps_sin[i]*ytemps_sin[j]
                templates[i,j,1,:,:] = xtemps_sin[i]*ytemps_cos[j]
                templates[i,j,2,:,:] = xtemps_cos[i]*ytemps_sin[j]
                templates[i,j,3,:,:] = xtemps_cos[i]*ytemps_cos[j]
     
    templates *= scale_fac

    if show_templates:
        for k in range(4):
            counter = 1
            plt.figure(figsize=(8,8))
            for i in range(n_terms):
                for j in range(n_terms):           
                    plt.subplot(n_terms, n_terms, counter)
                    plt.title('i = '+ str(i)+', j = '+str(j))
                    plt.imshow(templates[i,j,k,:,:])
                    plt.colorbar()
                    counter +=1
            plt.tight_layout()
            plt.show()

    if return_idxs:
        return templates, meshkx, meshky

    return templates



def fc_fit_precomp(image, dot1, X, mask_rav=None):
    
    dimx, dimy = image.shape[0], image.shape[1]
    YY = np.reshape(image, (dimx*dimy, 1)) # data vector

    if mask_rav is not None:
        YY_cut = YY[mask_rav]
        theta = np.dot(dot1, YY_cut)
    else:
        theta = np.dot(dot1, YY)

    fcs_withquad = np.reshape(np.dot(X, theta), (dimx, dimy))

    quadoffsets = np.reshape(np.dot(X[:,-4:], theta[-4:]), (dimx, dimy))
    
    return theta, fcs_withquad, quadoffsets



