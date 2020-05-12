import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ciber_data_helpers import make_radius_map
from ciber_mocks import *


def make_synthetic_trilegal_cat(trilegal_path, I_band_idx=16, H_band_idx=17, imdim=1024.):
    trilegal = np.loadtxt(trilegal_path)
    nsrc = trilegal.shape[0]
    synthetic_cat = np.random.uniform(0, imdim, size=(nsrc, 2))
    synthetic_cat = np.array([synthetic_cat[:,0], synthetic_cat[:,1], trilegal[:,I_band_idx], trilegal[:,H_band_idx]]).transpose()
    
    print('synthetic cat has shape ', synthetic_cat.shape)
    
    return synthetic_cat


def filter_trilegal_cat(trilegal_cat, m_max=17, I_band_idx=16):
    
    filtered_trilegal_cat = np.array([x for x in trilegal_cat if x[I_band_idx]<m_max])
    
    return filtered_trilegal_cat


def magnitude_to_radius_linear(magnitudes, alpha_m=-6.25, beta_m=110.):
    ''' Masking radius function as given by Zemcov+2014. alpha_m has units of arcsec mag^{-1}, while beta_m
    has units of arcseconds.'''
    
    r = alpha_m*magnitudes + beta_m

    return r

def mask_from_cat(catalog, xidx=0, yidx=1, mag_idx=3, dimx=1024, dimy=1024, pixsize=7., mode='Zemcov+14'):
    
    '''This function should take a catalog of bright sources and output a mask around those sources, 
    according to some masking criteria. The steps should be
        - convert magnitudes to radii with predefined function
        - use radii to construct mask
    '''
    
    print('Minimum source magnitude is ', np.min(catalog[:, mag_idx]))
    mask = np.ones([dimx,dimy], dtype=int)

    if mode=='Zemcov+14':
        radii = magnitude_to_radius_linear(catalog[:, mag_idx]-0.91)
        
    for i, r in enumerate(radii):
        radmap = make_radius_map(dimx=dimx, dimy=dimy, cenx=catalog[i,0], ceny=catalog[i,1], rc=1.)
        mask[radmap<r/pixsize] = 0.
        
    return mask 


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


