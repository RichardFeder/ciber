import numpy as np
import astropy.io.fits as fits
import astropy.wcs as wcs
import pandas as pd
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
from astropy.table import Table

def make_psf_template(path, field, band, nx=1024, pad=30, nwide=20, nbin=0, multfac=7., large=True):
    
    ''' This function loads PSF parameters and generates a normalized template over some array. 
    The PSF is parameterized as a power law'''

    beta, rc, norm = find_psf_params(path, tm=band+1, field=field)
    beta, rc, norm = load_psf_params_dict()
    Nlarge = nx+pad+pad
    radmap = make_radius_map(2*Nlarge+nbin, 2*Nlarge+nbin, Nlarge+nbin, Nlarge+nbin, rc)*multfac
    Imap_large = norm*np.power(1.+radmap, -3*beta/2.)
    Imap_large /= np.sum(Imap_large)
    psf_template = Imap_large[Nlarge-nwide:Nlarge+nwide+1, Nlarge-nwide:Nlarge+nwide+1]
    
    if large:
        psf = psf_large(psf_template)
        return psf, psf_template
    else:
        psf_template


def find_psf_params(path, tm=1, field='elat10'):
    ''' This function loads/parses PSF parameters given a path, field, and choice of tm (forget what this means)'''
    arr = np.genfromtxt(path, dtype='str')
    for entry in arr:
        if entry[0]=='TM'+str(tm) and entry[1]==field:
            beta, rc, norm = float(entry[2]), float(entry[3]), float(entry[4])
            return beta, rc, norm
    return False

def load_psf_params_dict(inst, field=None, ifield=None, tail_path='data/psf_model_dict_updated_081121_ciber.npz', verbose=True):
    
    ''' Parses PSF parameter file, which is two sets of dictionaries with the best fit beta model parameters. '''

    psf_mod = np.load(tail_path, allow_pickle=True)['PSF_model_dict'].item()

    if ifield is None:
        ifield_title_dict = dict({'Elat 10':4,'Elat 30':5, 'Bootes B':6, 'Bootes A':7, 'SWIRE':8})
        ifield = ifield_title_dict[field]

    params = psf_mod[inst][ifield]

    beta = params[0]
    rc = params[1]
    norm = params[2]

    if verbose:
        print('beta = ', beta)
        print('rc = ', rc)
        print('norm = ', norm)

    return beta, rc, norm


def compute_meshgrids(dimx, dimy, sparse=True, compute_dots=False):
    x = np.arange(dimx)
    y = np.arange(dimy)
    xx, yy = np.meshgrid(x, y, sparse=sparse)
    
    if compute_dots:
        xx_xx = xx*xx
        yy_yy = yy*yy
        return xx, yy, xx_xx, yy_yy
    else:
        return xx, yy

def compute_radmap_full(cenx, ceny, xx, yy):
    nsrc = len(cenx)
    for i in range(nsrc):
        radmap = (cenx[i] - xx)**2 + (ceny[i] - yy)**2
        if i==0:
            radmap_full = radmap.copy()
        else:
            radmap_full = np.minimum(radmap_full, radmap)
            
    return radmap_full

def make_radius_map(dimx, dimy, cenx, ceny, rc=1., sqrt=False):
    ''' This function calculates a map, where the value of each pixel is its distance from the central pixel.
    Useful for making PSF templates and other map making functions'''
    x = np.arange(dimx)
    y = np.arange(dimy)
    xx, yy = np.meshgrid(x, y, sparse=True)
    if sqrt:
        return np.sqrt(((cenx - xx)/rc)**2 + ((ceny - yy)/rc)**2)
    else:
        return ((cenx - xx)/rc)**2 + ((ceny - yy)/rc)**2

def make_radius_map_precomp(cenx, ceny, dimx=None, dimy=None, xx=None, yy=None, xx_xx=None, yy_yy=None, rc=None, sqrt=False):
    ''' This function calculates a map, where the value of each pixel is its distance from the central pixel.
    Useful for making PSF templates and other map making functions'''
    if xx is None or yy is None:
        if dimx is not None and dimy is not None:
            xx, yy, xx_xx, yy_yy = compute_meshgrids(dimx, dimy)
    if sqrt:
        return np.sqrt(((cenx - xx)/rc)**2 + ((ceny - yy)/rc)**2)
    else:
        if rc is None:
            return (cenx - xx)**2 + (ceny - yy)**2
        
        return ((cenx - xx)/rc)**2 + ((ceny - yy)/rc)**2

def make_radius_map_yt(mapin, cenx, ceny):
    '''
    return radmap of size mapin.shape. 
    radmap[i,j] = distance between (i,j) and (cenx, ceny)
    '''
    Nx, Ny = mapin.shape
    xx, yy = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    radmap = np.sqrt((xx - cenx)**2 + (yy - ceny)**2)
    return radmap


def grab_src_stamps(observed_image, cat_xs, cat_ys, nwide=50):
    
    nsrc = len(cat_xs)
    
    stamps = np.zeros((nsrc, nwide, nwide))
    x0s, y0s = np.zeros_like(cat_xs), np.zeros_like(cat_ys)
    
    for n in range(nsrc):
        
        x0, y0 = int(np.floor(cat_xs[n])), int(np.floor(cat_ys[n]))
        
        print(cat_xs[n], cat_ys[n], x0, y0)
        
        ylow, yhigh = max(y0-nwide//2, 0), min(y0+nwide//2, observed_image.shape[0])
        xlow, xhigh = max(x0-nwide//2, 0), min(x0+nwide//2, observed_image.shape[1])

        print(ylow, yhigh, xlow, xhigh)
        im_stamp = observed_image[ylow:yhigh, xlow:xhigh]
        
        stamps[n,:im_stamp.shape[0], :im_stamp.shape[1]] = im_stamp
        
        x0s[n] = x0
        y0s[n] = y0
        
    return stamps, x0s, y0s

def psf_large(psf_template, mapdim=1024):
    ''' all this does is place the 40x40 PSF template at the center of a full map 
    so one can compute the FT and get the beam correction for appropriate ell values
    this assumes that beyond the original template, the beam correction will be 
    exactly unity, which is fair enough for our purposes'''
    psf_temp = np.zeros((mapdim, mapdim))
    psf_temp[mapdim/2 - 20:mapdim/2+21, mapdim/2-20:mapdim/2+21] = psf_template
    psf_temp /= np.sum(psf_temp)
    return psf_temp