import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
from astropy.table import Table
import pandas as pd


def dataframe_to_fits(df, fits_path):
    table = Table.from_pandas(df)
    table.write(fits_path)

def fits_to_dataframe(fits_path):
    table = Table.read(fits_path)
    df = table.to_pandas()
    return df

def hsc_positions(hsc_cat, nchoose, z, dz, ra_min=241.5, ra_max=243.5, dec_min=54.0, dec_max=56.0, rmag_max=29.0):
    restricted_cat = hsc_cat[(hsc_cat['ra']>ra_min)&(hsc_cat['ra']<ra_max)&(hsc_cat['dec']>dec_min)&(hsc_cat['dec']<dec_max)
        &(hsc_cat['photoz_best']< z+0.5*dz)&(hsc_cat['photoz_best']>z-0.5*dz)&(hsc_cat['rmag_psf']<rmag_max)]
    
    dx1 = np.max(restricted_cat['x1'])-np.min(restricted_cat['x1'])
    dy1 = np.max(restricted_cat['y1'])-np.min(restricted_cat['y1'])
    restricted_cat['x1'] -= np.min(restricted_cat['x1'])
    restricted_cat['x1'] *= (float(size-1.)/dx1)
    restricted_cat['y1'] -= np.min(restricted_cat['y1'])
    restricted_cat['y1'] *= (float(size-1.)/dy1)
            
    n_hsc = len(hsc_cat[(hsc_cat['ra']>ra_min)&(hsc_cat['ra']<ra_max)&(hsc_cat['dec']>dec_min)&(hsc_cat['dec']<dec_max)
                  &(hsc_cat['photoz_best']< z+0.5*dz)&(hsc_cat['photoz_best']>z-0.5*dz)&(hsc_cat['rmag_psf']<rmag_max)])
    print('input number_counts:', np.sum(number_counts[i])*4, 'choosing from ', n_hsc, 'galaxies')

    idxs = np.random.choice(np.arange(n_hsc), nchoose, replace=True)
    tx = restricted_cat.iloc[idxs]['x1']+np.random.uniform(-0.5, 0.5, size=len(idxs))
    ty = restricted_cat.iloc[idxs]['y1']+np.random.uniform(-0.5, 0.5, size=len(idxs))
    
    return tx, ty

def sdss_preprocess(sdss_path, redshift_keyword='redshift', class_cut=False, object_class_cut=False, warning_cut=False):
    sdss_df = fits_to_dataframe(sdss_path)
    print('before cuts:', len(sdss_df.index))
    sdss_df = sdss_df.drop(['objid', 'run', 'rerun', 'camcol', 'field'], axis=1)
    
    sdss_df.loc[(sdss_df[redshift_keyword]>-1.0)&(sdss_df[redshift_keyword]<4.0)]
    print('after z cuts:', len(sdss_df.index))

    if class_cut:
        sdss_df = sdss_df.loc[(sdss_df['photoz_class']==1)]
        print('after photoz class cut:', len(sdss_df.index))
    
    if object_class_cut:
        sdss_df = sdss_df.loc[(sdss_df['object_class']=='GALAXY')]
        print('after object class cut:', len(sdss_df.index))


    if warning_cut:
        sdss_df = sdss_df.loc[(sdss_df['warning']==0)]
        print('after warning cut:', len(sdss_df.index))


    print('after cuts:', len(sdss_df.index))
    return sdss_df

