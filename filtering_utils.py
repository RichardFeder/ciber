import numpy as np
from flat_field_est import *
# from cross_spectrum_analysis import get_power_spec, get_power_spectrum_2d, azim_average_cl2d
from plotting_fns import plot_map
# from cross_spectrum_analysis import *


def calculate_plane(theta, dimx=1024, dimy=1024, X=None):
    
    if X is None:
        X1, X2 = np.mgrid[:dimx, :dimy]

        X = np.hstack(   ( np.reshape(X1, (dimx*dimy, 1)) , np.reshape(X2, (dimx*dimy, 1)) ) )
        X = np.hstack(   ( np.ones((dimx*dimy, 1)) , X ))
    
    plane = np.reshape(np.dot(X, theta), (dimx, dimy))
        
    return plane


def precomp_gradient_dat(dimx, dimy, mask=None):
    X1, X2 = np.mgrid[:dimx, :dimy]
    X = np.hstack(   ( np.reshape(X1, (dimx*dimy, 1)) , np.reshape(X2, (dimx*dimy, 1)) ) )
    X = np.hstack(   ( np.ones((dimx*dimy, 1)) , X ))

    if mask is not None:
        mask_rav = np.reshape(mask, (dimx*dimy)).astype(bool)
        X_cut = X[mask_rav,:]

        dot1 = np.dot( np.linalg.pinv(np.dot(X_cut.transpose(), X_cut)), X_cut.transpose())
        return dot1, X, mask_rav

    else:
        dot1 = np.dot( np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose())
        return dot1, X

def fit_gradient_to_map_precomp(image, dot1, X, mask_rav=None):
    dimx, dimy = image.shape[0], image.shape[1]

    YY = np.reshape(image, (dimx*dimy, 1)) # data vector
    if mask_rav is not None:
        # print('mask rav:', mask_rav)
        YY_cut = YY[mask_rav]
        
        theta = np.dot(dot1, YY_cut)
    else:
        theta = np.dot(dot1, YY)

    plane = np.reshape(np.dot(X, theta), (dimx, dimy))
    
    return theta, plane


def fit_gradient_to_map(image, mask=None):
    ''' 
    Fit image with a gradient. If mask provided the normal equation is truncated for all masked values. 
    
    Inputs
    ------

    image : `~np.array~` of shape (dimx, dimy), type `float`.
    mask (optional) : `~np.array~` of shape (dimx, dimy), type `float`. Optional mask.
        Default is None.

    Returns
    -------

    theta : `~np.array~` of shape (3), type `float`. Best fit gradient parameters
    plane : `~np.array~` of shape (dimx, dimy), type `float`. Best-fit plane. 

    '''
    
    dimx = image.shape[0]
    dimy = image.shape[1]

    X1, X2 = np.mgrid[:dimx, :dimy]
    
    X = np.hstack(   ( np.reshape(X1, (dimx*dimy, 1)) , np.reshape(X2, (dimx*dimy, 1)) ) )
    X = np.hstack(   ( np.ones((dimx*dimy, 1)) , X ))
    
    # additional shift parameter
    # X = np.hstack( (np.ones((dimx*dimy))))

    YY = np.reshape(image, (dimx*dimy, 1)) # data vector
    
    if mask is not None:
        mask_rav = np.reshape(mask, (dimx*dimy)).astype(bool)
        YY_cut = YY[mask_rav]
        X_cut = X[mask_rav,:]
        # print('using mask.. YY_cut has length', YY_cut.shape)
        theta = np.dot(np.dot( np.linalg.pinv(np.dot(X_cut.transpose(), X_cut)), X_cut.transpose()), YY_cut)
    else:
        theta = np.dot(np.dot( np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)
        
    # print('theta = ', theta)
    plane = np.reshape(np.dot(X, theta), (dimx, dimy))
        
    return theta, plane


def precomp_offset_gradient(dimx, dimy, mask=None, x0s=None, x1s=None, y0s=None, y1s=None):
    
    if x0s is None:
        x0s = [0, 0, 512, 512]
        x1s = [512, 512, 1024, 1024]
        y0s = [0, 512, 0, 512]
        y1s = [512, 1024, 512, 1024]

    X1, X2 = np.mgrid[:dimx, :dimy]

    X = np.hstack(   ( np.reshape(X1, (dimx*dimy, 1)) , np.reshape(X2, (dimx*dimy, 1)) ) )
    
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

def offset_gradient_fit_precomp(image, dot1, X, mask_rav=None):
    
    dimx, dimy = image.shape[0], image.shape[1]
    YY = np.reshape(image, (dimx*dimy, 1)) # data vector

    if mask_rav is not None:
        YY_cut = YY[mask_rav]
        theta = np.dot(dot1, YY_cut)
    else:
        theta = np.dot(dot1, YY)

    plane_offsets = np.reshape(np.dot(X, theta), (dimx, dimy))
    
    return theta, plane_offsets



