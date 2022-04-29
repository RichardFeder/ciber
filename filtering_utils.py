import numpy as np


def calculate_plane(theta, dimx=1024, dimy=1024, X=None):
    
    if X is None:
        X1, X2 = np.mgrid[:dimx, :dimy]

        X = np.hstack(   ( np.reshape(X1, (dimx*dimy, 1)) , np.reshape(X2, (dimx*dimy, 1)) ) )
        X = np.hstack(   ( np.ones((dimx*dimy, 1)) , X ))
    
    plane = np.reshape(np.dot(X, theta), (dimx, dimy))
        
    return plane


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

    plane : `~np.array~` of shape (dimx, dimy), type `float`. Best-fit plane. 

    '''
    
    dimx = image.shape[0]
    dimy = image.shape[1]

    X1, X2 = np.mgrid[:dimx, :dimy]
    
    X = np.hstack(   ( np.reshape(X1, (dimx*dimy, 1)) , np.reshape(X2, (dimx*dimy, 1)) ) )
    X = np.hstack(   ( np.ones((dimx*dimy, 1)) , X ))
    
    YY = np.reshape(image, (dimx*dimy, 1)) # data vector
    
    if mask is not None:
        mask_rav = np.reshape(mask, (dimx*dimy)).astype(np.bool)
        YY_cut = YY[mask_rav]
        X_cut = X[mask_rav,:]
        # print('using mask.. YY_cut has length', YY_cut.shape)
        theta = np.dot(np.dot( np.linalg.pinv(np.dot(X_cut.transpose(), X_cut)), X_cut.transpose()), YY_cut)
    else:
        theta = np.dot(np.dot( np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)
        
    # print('theta = ', theta)
    plane = np.reshape(np.dot(X, theta), (dimx, dimy))
        
    return theta, plane