import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from ciber_data_helpers import make_radius_map, compute_radmap_full
from plotting_fns import plot_map
# from ciber_source_mask_construction_pipeline import find_alpha_beta

''' TO DO : CAMB does not compile with Python 3 at the moment -- need to update Fortran compiler '''
import sys

if sys.version_info[0]==2:
    from ciber_mocks import *

# Yun-Ting's code for masking is here https://github.com/yuntingcheng/python_ciber/blob/master/stack_modelfit/mask.py


def make_fpaths(fpaths):
    for fpath in fpaths:
        if not os.path.isdir(fpath):
            print('making directory path for ', fpath)
            os.makedirs(fpath)
        else:
            print(fpath, 'already exists')

def find_alpha_beta(intercept, minrad=10, dm=3, pivot=16.):
    
    alpha_m = -(intercept - minrad)/dm
    beta_m = intercept - pivot*alpha_m
    
    return alpha_m, beta_m


def filter_trilegal_cat(trilegal_cat, m_min=4, m_max=17, filter_band_idx=16):
    
    filtered_trilegal_cat = np.array([x for x in trilegal_cat if x[filter_band_idx]<m_max and x[filter_band_idx]>m_min])

    return filtered_trilegal_cat

def radius_vs_mag_gaussian(mags, a1, b1, c1):
    return a1*np.exp(-((mags-b1)/c1)**2)

def magnitude_to_radius_linear(magnitudes, alpha_m=-6.25, beta_m=110.):
    ''' Masking radius function as given by Zemcov+2014. alpha_m has units of arcsec mag^{-1}, while beta_m
    has units of arcseconds.

    Parameters
    ----------
    
    magnitudes : float, or list of floats
        Source magnitudes used to find masking radii. 

    alpha_m : float, optional
        Slope of radius/magnitude equation. Default is -6.25. 

    beta_m : float, optional
        Zero point of radius/magnitude equation. Default from Zemcov+14 is 110.

    Returns
    -------

    r : float, or list of floats
        masking radii for input sources 

    '''
    
    r = alpha_m*magnitudes + beta_m

    return r

def make_synthetic_trilegal_cat(trilegal_path, J_band_idx=16, H_band_idx=17, imdim=1024.):
    trilegal = np.loadtxt(trilegal_path)
    nsrc = trilegal.shape[0]
    synthetic_cat = np.random.uniform(0, imdim, size=(nsrc, 2))
    synthetic_cat = np.array([synthetic_cat[:,0], synthetic_cat[:,1], trilegal[:,J_band_idx], trilegal[:,H_band_idx]]).transpose()
    
    print('synthetic cat has shape ', synthetic_cat.shape)
    
    return synthetic_cat



def mask_from_cat(xs=None, ys=None, mags=None, cat_df=None, dimx=1024, dimy=1024, pixsize=7.,\
                    interp_maskfn=None, mode='Zemcov+14', magstr='zMeanPSFMag', alpha_m=-6.25, beta_m=110, a1=252.8, b1=3.632, c1=8.52,\
                     Vega_to_AB = 0., mag_lim_min=0, mag_lim=None, fixed_radius=None, radii=None, compute_radii=True, inst=1, \
                    radmap_full=None, rc=1., plot=True, interp_max_mag=None, interp_min_mag=None, m_min_thresh=None, radcap=200.):
    
    if fixed_radius is not None or radii is not None:
        compute_radii = False
        
    mask = np.ones([dimx,dimy], dtype=int)
    
    if compute_radii:
        if mags is None and cat_df is None:
            print('Need magnitudes one way or another to compute radii, please specify with mags parameter or cat_df..')
            return
        if cat_df is None:
            if xs is None or ys is None or mags is None:
                print('cat_df=None, but no input information for xs, ys, mags..')
                return

    if mag_lim is not None:
        if cat_df is not None:
            mag_lim_mask = np.where((cat_df[magstr] < mag_lim)&(cat_df[magstr] > mag_lim_min))[0]
            cat_df = cat_df.iloc[mag_lim_mask]
            xs = np.array(cat_df['x'+str(inst)])
            ys = np.array(cat_df['y'+str(inst)])

            print('length after cutting on '+magstr+'< '+str(mag_lim)+' is '+str(len(xs)))

            
        elif mags is not None:
            mag_lim_mask = (mags < mag_lim)*(mags > mag_lim_min)
            mags = mags[mag_lim_mask]
            xs = xs[mag_lim_mask]
            ys = ys[mag_lim_mask]

    if interp_maskfn is not None:
        # magnitudes need to be in Vega system as this is how interp_maskfn is defined!! 
        print("Using interpolated function to get masking radii..")
        if cat_df is not None and len(cat_df) > 0:
            mags = cat_df[magstr]

            print('max mag is ', np.max(mags), np.nanmax(mags))
            print('interp max mag is ', interp_max_mag)

            if interp_max_mag is not None:
                mags[mags > interp_max_mag] = interp_max_mag
            if interp_min_mag is not None:
                mags[mags < interp_min_mag] = interp_min_mag

            radii = interp_maskfn(np.array(mags))

            if m_min_thresh is not None:
                radii[np.array(mags) < m_min_thresh] = radcap

            if plot:
                plt.figure()
                plt.scatter(mags, radii, s=3, color='k')
                plt.xlabel('Vega mags')
                plt.ylabel('radii [arcsec]')
                plt.show()
        else:
            return None, []

    if compute_radii and radii is None:
        print('Computing radii based on magnitudes..')
        if cat_df is not None:
            mags = cat_df[magstr]
        if mode=='Zemcov+14':
            radii = magnitude_to_radius_linear(mags, alpha_m=alpha_m, beta_m=beta_m)
        elif mode=='Simon':
            AB_mags = np.array(cat_df[magstr]) + Vega_to_AB
            radii = radius_vs_mag_gaussian(mags, a1=a1, b1=b1, c1=c1)

    if radii is not None:
        for i, r in enumerate(radii):
            radmap = make_radius_map(dimx=dimx, dimy=dimy, cenx=xs[i], ceny=ys[i], sqrt=False)
            mask[radmap<r**2/pixsize**2] = 0.
            
        return mask, radii

    else:
        if radmap_full is None:
            xx, yy = compute_meshgrids(dimx, dimy)
            radmap_full = compute_radmap_full(xs, ys, xx, yy)
        
        if fixed_radius is not None:
            thresh = fixed_radius**2/(pixsize*pixsize)
            mask[(radmap_full < thresh*rc**2)] = 0.
 
        return mask, radmap_full, thresh
    

def get_masks(star_cat_df, mask_fn_param_combo, intercept_mag_AB, mag_lim_AB, inst=1, instrument_mask=None, minrad=14., dm=3, dimx=1024, dimy=1024, verbose=True, Vega_to_AB=0., magstr='j_m'):
    
    '''
    Computes astronomical source mask for catalog

    '''


    intercept = radius_vs_mag_gaussian(intercept_mag_AB, a1=mask_fn_param_combo[0], b1=mask_fn_param_combo[1],\
                                       c1=mask_fn_param_combo[2])
    
    alpha_m, beta_m = find_alpha_beta(intercept, minrad=minrad, dm=dm, pivot=intercept_mag_AB)
    
    if verbose:
        print('param_combo is ', mask_fn_param_combo)
        print('making bright star mask..')
        
    mask_stars_simon, radii_stars_simon = mask_from_cat(cat_df = star_cat_df, mag_lim_min=0, inst=inst,\
                                                                    mag_lim=intercept_mag_AB, mode='Simon', a1=mask_fn_param_combo[0], b1=mask_fn_param_combo[1], c1=mask_fn_param_combo[2], magstr=magstr, Vega_to_AB=Vega_to_AB, dimx=dimx, dimy=dimy)


    if verbose:
        print('alpha, beta are ', alpha_m, beta_m)
        print('making faint source mask..')
    mask_stars_Z14, radii_stars_Z14 = mask_from_cat(cat_df = star_cat_df, inst=inst, mag_lim_min=intercept_mag_AB, mag_lim=mag_lim_AB, mode='Zemcov+14', alpha_m=alpha_m, beta_m=beta_m, magstr=magstr, Vega_to_AB=Vega_to_AB, dimx=dimx, dimy=dimy)

    joint_mask = mask_stars_simon*mask_stars_Z14
    
    print('joint mask is type ', type(joint_mask))
    joint_mask = joint_mask.astype(np.int)
    
    if instrument_mask is not None:
        if verbose:
            print('instrument mask being applied as well')
        joint_mask *= instrument_mask.astype(np.int)
        
    return joint_mask, radii_stars_simon, radii_stars_Z14, alpha_m, beta_m

def get_mask_radius_th_rf(m_arr, beta, rc, norm, band=0, Ith=1., fac=0.7, plot=False):

    m_arr = np.array(m_arr)
    Nlarge = 100
    radmap = make_radius_map_yt(np.zeros([2*Nlarge+1, 2*Nlarge+1]), Nlarge, Nlarge)
    radmap *= fac
    Imap_large = norm * (1. + (radmap/rc)**2)**(-3.*beta/2.)
    
    Imap_large /= np.sum(Imap_large)
    
    # get Imap of PSF, multiply by flux, see where 
    
    lam_effs = np.array([1.05, 1.79]) # effective wavelength for bands in micron

    lambdaeff = lam_effs[band]

    sr = ((7./3600.0)*(np.pi/180.0))**2

    I_arr=3631*10**(-m_arr/2.5)*(3./lambdaeff)*1e6/(sr*1e9) # 1e6 is to convert from microns
    r_arr = np.zeros_like(m_arr, dtype=float)
    for i, I in enumerate(I_arr):
        sp = np.where(Imap_large*I > Ith)
        if len(sp[0])>0:
            r_arr[i] = np.max(radmap[sp])
            
    
    if plot:
        f = plt.figure(figsize=(10,10))

        plt.suptitle('I_th = '+str(Ith)+', multiplicative fac = '+str(fac), fontsize=20)
        plt.subplot(2,2,1)
        plt.title('radmap')
        plt.imshow(radmap)
        plt.colorbar()

        plt.subplot(2,2,2)
        plt.title('Imap_large')
        plt.imshow(Imap_large)
        plt.colorbar()

        plt.subplot(2,2,3)
        plt.scatter(m_arr, I_arr)
        plt.xlabel('$m_I$', fontsize=16)
        plt.ylabel('I_arr', fontsize=16)

        plt.subplot(2,2,4)
        plt.scatter(m_arr, r_arr)
        plt.xlabel('$m_I$', fontsize=16)
        plt.ylabel('$r$ [arcsec]', fontsize=16)
        plt.show()
        
        return r_arr, f

    return r_arr


def I_threshold_mask_simple(xs, ys, ms,psf=None, beta=None, rc=None, norm=None, ciber_mock=None, ifield=None, band=0, dimx=1024, dimy=1024, m_min=-np.inf, m_max=20, \
                    I_thresh=1., method='radmap', nwide=12, fac=0.7, plot=False):
    
    # get the CIBER PSF for a given ifield if desired

    '''
    Parameters
    ----------


    Returns
    -------


    '''
    beta, rc, norm = ciber_mock.get_psf(ifield=ifield, band=band, nx=dimx, ny=dimy, poly_fit=False, nwide=nwide)

    # beta, rc, norm = 1.593e+00, 4.781e+00, 9.477e-03
    
    print('beta, rc, norm:', beta, rc, norm)
    mask = np.ones([dimx, dimy], dtype=int)

    mag_mask = np.where((ms > m_min) & (ms < m_max))[0]
    
    xs, ys, ms = xs[mag_mask], ys[mag_mask], ms[mag_mask]
    
    if plot:
        rs, f = get_mask_radius_th_rf(ms, beta, rc, norm, band=band, Ith=I_thresh, fac=fac, plot=True)
    else:
        rs = get_mask_radius_th_rf(ms, beta, rc, norm, band=band, Ith=I_thresh, fac=fac, plot=False)

        
    for i,(x,y,r) in enumerate(zip(xs, ys, rs)):
        radmap = make_radius_map_yt(np.zeros(shape=(dimx, dimy)), x, y)

        mask[radmap < r/7.] = 0

        if i%1000==0:
            print('i='+str(i)+' of '+str(len(xs)))        
        
    return mask, rs, beta, rc, norm


def I_threshold_image_th(xs, ys, ms, psf=None, beta=None, rc=None, norm=None, ciber_mock=None, ifield=None, band=0, dimx=1024, dimy=1024, m_min=-np.inf, m_max=20, \
                    I_thresh=1., nwide=17, nc=35):

    # get the CIBER PSF for a given ifield if desired
    if ifield is not None:
        if ciber_mock is None:
            print('Need ciber_mock class to obtain PSF for ifield '+str(ifield))
            return None
        else:
            ciber_mock.get_psf(poly_fit=True, nwide=17)
            psf = ciber_mock.psf_template

    mask = np.ones([dimx, dimy], dtype=int)    

    mag_mask = np.where((ms > m_min) & (ms < m_max))[0]
    
    xs, ys, ms = xs[mag_mask], ys[mag_mask], ms[mag_mask]
    
    print('len xs is ', len(xs))

    if ciber_mock is not None:
        Is = ciber_mock.mag_2_nu_Inu(ms, band)
    else:
        lam_effs = np.array([1.05, 1.79])*1e-6 # effective wavelength for bands
        sr = ((7./3600.0)*(np.pi/180.0))**2
        Is=3631*10**(-ms/2.5)*(3/lam_effs[band])*1e6/(sr*1e9)

    print("Using bright source map threshold..")
    srcmap = image_model_eval(np.array(xs).astype(np.float32)+2.5, np.array(ys).astype(np.float32)+1.0 ,np.array(Is.value).astype(np.float32),0., (dimx, dimy), nc, ciber_mock.cf, lib=ciber_mock.libmmult.pcat_model_eval)
    
    mask[srcmap > I_thresh] = 0

    return mask, srcmap


def I_threshold_mask(xs, ys, ms,psf=None, beta=None, rc=None, norm=None, ciber_mock=None, ifield=None, band=0, dimx=1024, dimy=1024, m_min=-np.inf, m_max=20, \
                    I_thresh=1., method='radmap'):
    
    # get the CIBER PSF for a given ifield if desired
    if ifield is not None:
        if ciber_mock is None:
            print('Need ciber_mock class to obtain PSF for ifield '+str(ifield))
            return None
        else:
            beta, rc, norm = ciber_mock.get_psf(ifield=ifield, band=band, nx=dimx, ny=dimy, poly_fit=True)
            psf = ciber_mock.psf_template

    mask = np.ones([dimx, dimy], dtype=int)

    num = np.zeros([dimx, dimy], dtype=int)

    mag_mask = np.where((ms > m_min) & (ms < m_max))[0]
    
    xs, ys, ms = xs[mag_mask], ys[mag_mask], ms[mag_mask]
    
    if ciber_mock is not None:
        Is = ciber_mock.mag_2_nu_Inu(ms, band)
        
        print(Is)
    else:
        lam_effs = np.array([1.05, 1.79])*1e-6 # effective wavelength for bands
        sr = ((7./3600.0)*(np.pi/180.0))**2
        Is=3631*10**(-ms/2.5)*(3/lam_effs[band])*1e6/(sr*1e9)
     
    
    rs = rc*((I_thresh/(Is.value*norm))**(-2/(3.*beta)) - 1.)

    if method=='radmap':
        print('Using radmap method..')
        for i,(x,y,r) in enumerate(zip(xs, ys, rs)):

            radmap = make_radius_map(dimx, dimy, x, y, 1.)

            mask[radmap < r] = 0
            num[radmap < r] += 1

            if i%1000==0:
                print('i='+str(i)+' of '+str(len(xs)))
                    
    else:
        print("Using bright source map threshold..")
        nc = 25
        srcmap = image_model_eval(np.array(xs).astype(np.float32), np.array(ys).astype(np.float32) ,np.array(Is.value).astype(np.float32),0., (dimx, dimy), nc, ciber_mock.cf, lib=ciber_mock.libmmult.pcat_model_eval)
        mask[srcmap > I_thresh] = 0
        
        
    return mask, num, rs, beta, rc, norm



def simon_r_m(mags, a1=252.8, b1=3.632, c1=8.52, Vega_to_AB=0.):
    '''
    Masking radius formula based on best fit from Simon's analysis.

    Parameters
    ----------

    mags : `numpy.ndarray' of shape (Nsrc,)
        Source magnitudes

    a1 : `float', optional
        Normalization coefficient. Default is 252.8.
    b1 : `float', optional
        mean of fit Gaussian. Default is 3.632.
    c1 : `float', optional
        scale radius of Gaussian. Default is 8.52.

    Returns
    -------
    radii : `numpy.ndarray' of shape (Nsrc,)
        Masking radii, given in arcseconds. 


    '''
    radii = a1*np.exp(-((mags-b1)/c1)**2)

    return radii



