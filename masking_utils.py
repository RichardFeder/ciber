import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ciber_data_helpers import make_radius_map
from ciber_mocks import *

# Yun-Ting's code for this stuff is here https://github.com/yuntingcheng/python_ciber/blob/master/stack_modelfit/mask.py


def compute_star_gal_mask(stellar_cat, galaxy_cat, star_mag_idx=2, gal_mag_idx=3, m_max=18.4, return_indiv_masks=False):
    
    '''star and galaxy catalogs reported with AB magnitudes. 
    Default indices assume stellar catalog already passed through make_synthetic_trilegal_cat()'''
    
    filt_star_cat = filter_trilegal_cat(stellar_cat, I_band_idx=star_mag_idx, m_max=m_max)
    
    cmock = ciber_mock()
    filt_gal_cat = cmock.catalog_mag_cut(galaxy_cat, galaxy_cat[:, gal_mag_idx], m_min=0., m_max=m_max)
    
    mask_filt_star = mask_from_cat(filt_x, mag_idx=star_mag_idx)
    mask_gal = mask_from_cat(bright_cat, mag_idx=gal_mag_idx)
    
    joined_mask = mask_filt_star*mask_gal
    
    if return_indiv_masks:
        return mask_filt_star, mask_gal
    
    return joined_mask


def filter_trilegal_cat(trilegal_cat, m_max=17, I_band_idx=16):
    
    filtered_trilegal_cat = np.array([x for x in trilegal_cat if x[I_band_idx]<m_max])
    
    return filtered_trilegal_cat


def make_radius_map_yt(mapin, cenx, ceny):
    '''
    return radmap of size mapin.shape. 
    radmap[i,j] = distance between (i,j) and (cenx, ceny)
    '''
    Nx, Ny = mapin.shape
    xx, yy = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    radmap = np.sqrt((xx - cenx)**2 + (yy - ceny)**2)
    return radmap


def magnitude_to_radius_linear(magnitudes, alpha_m=-6.25, beta_m=110.):
    ''' Masking radius function as given by Zemcov+2014. alpha_m has units of arcsec mag^{-1}, while beta_m
    has units of arcseconds.'''
    
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



def get_mask_radius_th_rf(m_arr, beta, rc, norm, band=0, Ith=1., fac=0.7, plot=False):
    '''
    r_arr: arcsec
    '''
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
                    I_thresh=1., method='radmap', nwide=12, fac=0.7):
    
    # get the CIBER PSF for a given ifield if desired

    beta, rc, norm = ciber_mock.get_psf(ifield=ifield, band=band, nx=dimx, ny=dimy, poly_fit=False, nwide=nwide)

    beta, rc, norm = 1.593e+00, 4.781e+00, 9.477e-03
    
    print('beta, rc, norm:', beta, rc, norm)
    mask = np.ones([dimx, dimy], dtype=int)

    mag_mask = np.where((ms > m_min) & (ms < m_max))[0]
    
    xs, ys, ms = xs[mag_mask], ys[mag_mask], ms[mag_mask]
    
    rs, f = get_mask_radius_th_rf(ms, beta, rc, norm, band=band, Ith=I_thresh, fac=fac)
        
    for i,(x,y,r) in enumerate(zip(xs, ys, rs)):
        radmap = make_radius_map_yt(np.zeros(shape=(dimx, dimy)), x, y)

        mask[radmap < r/7.] = 0

        if i%1000==0:
            print('i='+str(i)+' of '+str(len(xs)))        
        
    return mask, rs, beta, rc, norm



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







