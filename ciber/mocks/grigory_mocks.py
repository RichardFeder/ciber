import matplotlib
import matplotlib.pyplot as plt
from ciber.mocks.cib_mocks import *

import numpy as np
from scipy import interpolate
import os
import astropy
import astropy.wcs as wcs

import config
from ciber.core.powerspec_pipeline import *
from ciber.core.ps_pipeline_go import *
from ciber.theory.cl_predictions import *

# %matplotlib inline


cmock = ciber_mock()
cbps = CIBER_PS_pipeline()
lb = cbps.Mkk_obj.midbin_ell

# mocks for Grigory

inst = 1
ifield = 8
npix = 512

cat_fpaths = glob.glob('data/mocks_catalogs_1x1_for_Richard/*.csv')

for f in range(len(cat_fpaths)):
    
    cat = pd.read_csv(cat_fpaths[f])
    
    strsel = cat_fpaths[f].split('/')[-1].replace('.csv', '').replace('cat', 'catmap')
    
    print('strsel:', strsel)
    
    ra, dec, mag = [np.array(cat[key]) for key in ['RA', 'DEC', 'AB_MAG']]
    
    mask = (~np.isinf(mag))*(mag < 28)*(mag > 16)
    
    ra_sel, dec_sel, mag_sel = ra[mask], dec[mask], mag[mask]
    
    x_sel = npix*(ra_sel-np.min(ra_sel))/(np.max(ra_sel)-np.min(ra_sel))
    y_sel = npix*(dec_sel-np.min(dec_sel))/(np.max(dec_sel)-np.min(dec_sel))
    
#     plt.figure(figsize=(5, 5))
#     plt.scatter(x_sel, y_sel, color='k', s=2, alpha=0.1)
#     plt.show()
    
    print('len of mag:', len(mag_sel))
    
    cat_full = np.array([x_sel, y_sel, mag_sel]).transpose()
    
        
    I_arr_full = cmock.mag_2_nu_Inu(cat_full[:,2], lam_eff=0.8*1e-6*u.m)
    cat_full = np.hstack([cat_full, np.expand_dims(I_arr_full.value, axis=1)])
    print(cat_full.shape)
    
    srcmap_full = cmock.make_srcmap_temp_bank(8, 1, cat_full, flux_idx=-1, n_fine_bin=10, nwide=17, load_precomp_tempbank=True, \
                            tempbank_dirpath='data/subpixel_psfs_TM'+str(inst)+'/')

    srcmap = srcmap_full[:npix, :npix]
    
    plot_map(srcmap, figsize=(8, 8))
    plot_map(gaussian_filter(srcmap, sigma=5), figsize=(8, 8))
    
    lb, cl, clerr = get_power_spec(srcmap-np.mean(srcmap), nbins=26)
    
    plt.figure()
    plt.errorbar(lb, (lb**2/(2*np.pi))*cl, yerr=(lb**2/(2*np.pi))*clerr, fmt='o', color='k')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(alpha=0.5)
    plt.xlabel('$\\ell$')
    plt.ylabel('$D_{\\ell}$')
    plt.show()

    np.savez('data/mocks_catalogs_1x1_for_Richard/maps/'+strsel+'_16_ltzlt_28.npz', srcmap=srcmap)
#     np.savez('data/mocks_catalogs_1x1_for_Richard/cats_ciber/'+strsel+'_catalog.npz', x=x_sel, y=y_sel, inst=inst, ifield=ifield, \
#             mag_sel=mag_sel)
    


