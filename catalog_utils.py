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

def hsc_positions(hsc_cat, nchoose, z, dz=10, ra_min=241.5, ra_max=243.5, dec_min=54.0, dec_max=56.0, rmag_max=29.0, scatter=True):

    ''' This function filters a catalog from HSC given boudns on RA/DEC, as well as redshift and r-band magnitude.
    With this, one can sample HSC catalog positions that can be used for mock realizations that need realistic 
    clustering/selection effects. Note: this was made before I had lognormal catalog generation working.

    Input:
        hsc_cat: HSC catalog
        
        z (float): central redshift for HSC catalog selection
        
        nchoose (int): number of source positions to sample from HSC catalog 
        
        dz (float, optional, default=10): width of redshift selection bin, centered on z.
        
        ra_min/ra_max/dec_min/dec_max (float): bounds on RA/DEC for catalog selection
        
        rmag_max (float, optional, default=29.0): limiting HSC r-band magnitude for catalog selection 
        
        scatter (bool, default=True): if True, add random uniform scatter about catalog pixel coordinates. I think this is because 
                                        catalog positions in x/y are given as integers
    
    Output:
        tx/ty: (float array): array of source x/y-coordinates for catalog 

    '''

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

    idxs = np.random.choice(np.arange(n_hsc), nchoose, replace=True)
    
    tx = restricted_cat.iloc[idxs]['x1']
    ty = restricted_cat.iloc[idxs]['y1']

    if scatter:
        tx += np.random.uniform(-0.5, 0.5, size=(len(idxs)))
        ty += np.random.uniform(-0.5, 0.5, size=(len(idxs)))
    
    return tx, ty

def sdss_preprocess(sdss_path, redshift_keyword='redshift', class_cut=False, object_class_cut=None, warning_cut=False):
    
    ''' 

    Given a file path, this function preprocesses a SDSS catalog as a pandas Dataframe, with optional cuts on catalog properties.
    Most of the cuts are for quality assurance on things SDSS flags as obviously problematic, but some of them are more specific,
    and this function could really be expanded depending on future use.

    Inputs:
        sdss_path (string): File path for desired SDSS catalog. Right now function assumes catalog will be in FITS file.
        
        redshift_keyword (string, default='redshift'): keyword in SDSS catalog that is used in SDSS catalog. One can use different keywords
            when querying a SDSS catalog on SkyServer, so this adds flexibility in those cases where different keys are used.
        
        class_cut (bool, default=False): if True, filter catalog on SDSS reported photometric redshift class. Only leaves sources with class=1 (best photo-zs)  
        
        object_class_cut (string, default=None): if not None, object_class_cut filters for catalog sources with specific SDSS class. 
            For example, one could filter for galaxies (object_class_cut='GALAXY') or stars (object_class_cut='STAR')

        warning_cut (bool, default=False): if True, filter catalog on SDSS warning flags. Only leaves sources with zero warning flags, 
            which may not be perfect (I think there's one warning flag that isn't particularly bad) but filters out obvious failures.

    Output:

        sdss_df (pd.DataFrame): Returns a dataframe with filtered catalog.

    '''

    sdss_df = fits_to_dataframe(sdss_path)
    print('before cuts:', len(sdss_df.index))
    sdss_df = sdss_df.drop(['objid', 'run', 'rerun', 'camcol', 'field'], axis=1)
    
    sdss_df.loc[(sdss_df[redshift_keyword]>-1.0)&(sdss_df[redshift_keyword]<4.0)]
    print('after z cuts:', len(sdss_df.index))

    if class_cut:
        sdss_df = sdss_df.loc[(sdss_df['photoz_class']==1)]
        print('after photoz class cut:', len(sdss_df.index))
    
    if object_class_cut is not None:
        sdss_df = sdss_df.loc[(sdss_df['object_class']==object_class_cut)]
        print('after object class cut:', len(sdss_df.index))


    if warning_cut:
        sdss_df = sdss_df.loc[(sdss_df['warning']==0)]
        print('after warning cut:', len(sdss_df.index))


    print('after cuts:', len(sdss_df.index))
    return sdss_df

