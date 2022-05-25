import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ciber_data_helpers import make_radius_map
from ciber_source_mask_construction_pipeline import find_alpha_beta

''' TO DO : CAMB does not compile with Python 3 at the moment -- need to update Fortran compiler '''
import sys

if sys.version_info[0]==2:
    from ciber_mocks import *

# Yun-Ting's code for masking is here https://github.com/yuntingcheng/python_ciber/blob/master/stack_modelfit/mask.py

def filter_trilegal_cat(trilegal_cat, m_min=4, m_max=17, I_band_idx=16):
    
    filtered_trilegal_cat = np.array([x for x in trilegal_cat if x[I_band_idx]<m_max and x[I_band_idx]>m_min])

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

def make_synthetic_trilegal_cat(trilegal_path, I_band_idx=16, H_band_idx=17, imdim=1024.):
    trilegal = np.loadtxt(trilegal_path)
    nsrc = trilegal.shape[0]
    synthetic_cat = np.random.uniform(0, imdim, size=(nsrc, 2))
    synthetic_cat = np.array([synthetic_cat[:,0], synthetic_cat[:,1], trilegal[:,I_band_idx], trilegal[:,H_band_idx]]).transpose()
    
    print('synthetic cat has shape ', synthetic_cat.shape)
    
    return synthetic_cat


def mask_from_cat(catalog, xidx=0, yidx=1, mag_idx=3, dimx=1024, dimy=1024, pixsize=7., mode='Zemcov+14', ciber_mock=None, ifield=None, I_thresh=1., thresh_method='radmap'):
    
    '''This function should take a catalog of bright sources and output a mask around those sources, 
    according to some masking criteria. The steps should be
        - convert magnitudes to radii with predefined function
        - use radii to construct mask
    '''
    
    print('Minimum source magnitude is ', np.min(catalog[:, mag_idx]))
    mask = np.ones([dimx,dimy], dtype=int)

    if mode=='Zemcov+14':
        radii = magnitude_to_radius_linear(catalog[:, mag_idx]-0.91) # vega to AB factor generalized for different bands?
        
        for i, r in enumerate(radii):
            radmap = make_radius_map(dimx=dimx, dimy=dimy, cenx=catalog[i,0], ceny=catalog[i,1], rc=1.)
            mask[radmap<r/pixsize] = 0.

    elif mode=='I_thresh':
        mask, num, rs, _, _, _ = I_threshold_mask(catalog[:,0], catalog[:,1], catalog[:, mag_idx], \
                                                    ciber_mock=ciber_mock, ifield=ifield, dimx=dimx, dimy=dimy, I_thresh=I_thresh, method=thresh_method)
        
    return mask 


def mask_from_df_cat(xs=None, ys=None, mags=None, cat_df=None, dimx=1024, dimy=1024, pixsize=7., mode='Zemcov+14', magstr='zMeanPSFMag', alpha_m=-6.25, beta_m=110, \
    a1=252.8, b1=3.632, c1=8.52, Vega_to_AB = 0., mag_lim_min=0, mag_lim=None, inst=1):
    
    # can take mean color between PanSTARRS band and J band as zeroth order approx. ideally would regress, 
    # but probably doesn't make big difference
    
    if mags is None and cat_df is None:
        print('Need magnitudes one way or another, please specify with mags parameter or cat_df')
        return

    if cat_df is None:
        if xs is None or ys is None or mags is None:
            print('cat_df=None, but no input information for xs, ys, mags')
            return

    if mag_lim is not None:

        if cat_df is not None:
            # print(cat_df[magstr])
            mag_lim_mask = np.where((cat_df[magstr] < mag_lim)&(cat_df[magstr] > mag_lim_min))[0]
            cat_df = cat_df.iloc[mag_lim_mask]
            xs = np.array(cat_df['x'+str(inst)])
            ys = np.array(cat_df['y'+str(inst)])

        elif mags is not None:
            mag_lim_mask = (mags < mag_lim)*(mags > mag_lim_min)
            mags = mags[mag_lim_mask]
            xs = xs[mag_lim_mask]
            ys = ys[mag_lim_mask]

    mask = np.ones([dimx,dimy], dtype=int)

    if cat_df is not None:
        mags = cat_df[magstr]

    ''' is there an error in Simon mode? I need to account for all catalog magnitude systems and their conversions. '''

    if mode=='Zemcov+14':
        radii = magnitude_to_radius_linear(mags, alpha_m=alpha_m, beta_m=beta_m)


    elif mode=='Simon':
        AB_mags = np.array(cat_df[magstr]) + Vega_to_AB
        radii = radius_vs_mag_gaussian(mags, a1=a1, b1=b1, c1=c1)


    for i, r in enumerate(radii):
        radmap = make_radius_map(dimx=dimx, dimy=dimy, cenx=xs[i], ceny=ys[i], rc=1., sqrt=True)
        mask[radmap<r/pixsize] = 0.
            
    return mask, radii

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
        
    mask_stars_simon, radii_stars_simon = mask_from_df_cat(cat_df = star_cat_df, mag_lim_min=0, inst=inst,\
                                                                    mag_lim=intercept_mag_AB, mode='Simon', a1=mask_fn_param_combo[0], b1=mask_fn_param_combo[1], c1=mask_fn_param_combo[2], magstr=magstr, Vega_to_AB=Vega_to_AB, dimx=dimx, dimy=dimy)


    if verbose:
        print('alpha, beta are ', alpha_m, beta_m)
        print('making faint source mask..')
    mask_stars_Z14, radii_stars_Z14 = mask_from_df_cat(cat_df = star_cat_df, inst=inst, mag_lim_min=intercept_mag_AB, mag_lim=mag_lim_AB, mode='Zemcov+14', alpha_m=alpha_m, beta_m=beta_m, magstr=magstr, Vega_to_AB=Vega_to_AB, dimx=dimx, dimy=dimy)

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

    



