import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
from astropy.table import Table
from astropy.io import fits
import astropy.wcs as wcs
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky
import pandas as pd
from scipy.ndimage import gaussian_filter



def catalog_df_add_xy(field, df, datadir='/Users/luminatech/Documents/ciber2/ciber/data/', imcut=True):
    order = [c for c in df.columns]
    # find the x, y solution with all quad
    for inst in [1,2]:
        print('TM'+str(inst))
        hdrdir = datadir+'astroutputs/inst'+str(inst)+'/'
        xoff = [0,0,512,512]
        yoff = [0,512,0,512]
        for iquad,quad in enumerate(['A','B','C','D']):
            print('quad '+quad)
            hdulist = fits.open(hdrdir + field + '_' + quad + '_astr.fits')
            wcs_hdr=wcs.WCS(hdulist[('primary',1)].header, hdulist)
            hdulist.close()
            src_coord = SkyCoord(ra=df['ra']*u.degree, dec=df['dec']*u.degree, frame='icrs')

            x_arr, y_arr = wcs_hdr.all_world2pix(df['ra'],df['dec'],0)
            df['x' + quad] = x_arr + xoff[iquad]
            df['y' + quad] = y_arr + yoff[iquad]

        df['meanx'] = (df['xA'] + df['xB'] + df['xC'] + df['xD']) / 4
        df['meany'] = (df['yA'] + df['yB'] + df['yC'] + df['yD']) / 4
        
        # assign the x, y with the nearest quad solution
        df['x'+str(inst)] = df['xA'].copy()
        df['y'+str(inst)] = df['yA'].copy()
        bound = 511.5
        df.loc[ (df['meanx'] < bound) & (df['meany'] > bound),'x'+str(inst)] = df['xB']
        df.loc[ (df['meanx'] < bound) & (df['meany'] > bound),'y'+str(inst)] = df['yB']
        
        df.loc[ (df['meanx'] > bound) & (df['meany'] < bound),'x'+str(inst)] = df['xC']
        df.loc[ (df['meanx'] > bound) & (df['meany'] < bound),'y'+str(inst)] = df['yC']

        df.loc[ (df['meanx'] > bound) & (df['meany'] > bound),'x'+str(inst)] = df['xD']
        df.loc[ (df['meanx'] > bound) & (df['meany'] > bound),'y'+str(inst)] = df['yD']

    if imcut:
        print('making cut on image bounds')
        inst1_mask = (df['x1'] > 0)&(df['x1'] < 1023)&(df['y1'] > 0)&(df['y1'] < 1023)
        inst2_mask = (df['x2'] > 0)&(df['x2'] < 1023)&(df['y2'] > 0)&(df['y2'] < 1023)
        imcut_idx = np.where(inst1_mask | inst2_mask)
        df = df.iloc[imcut_idx].copy()
        df = df.reset_index(drop=True)
        
    # write x, y to df
    order = order[:2] + ['x1','y1','x2','y2'] + order[2:]
    dfout = df[order].copy()
    
    return dfout

def check_for_catalog_duplicates(cat, cat2=None, match_thresh=0.1, nthneighbor=2, ra_errors=None, dec_errors=None, zscore=1):
    
    cat_src_coord = SkyCoord(ra=cat['ra']*u.degree, dec=cat['dec']*u.degree, frame='icrs')
    
    if cat2 is None:
        cat2_src_coord = cat_src_coord
    else:
        cat2_src_coord = SkyCoord(ra=cat2['ra']*u.degree, dec=cat['dec']*u.degree, frame='icrs')

    # choose nthneighbor=2 to not just include the same source
    idx, d2d, _ = match_coordinates_sky(cat_src_coord, cat2_src_coord, nthneighbor=nthneighbor) # there is an order specific element to this
    
    if ra_errors is not None:
        match_thresh = zscore*np.sqrt(ra_errors**2 + dec_errors**2)
        print('match threshes is ', match_thresh)
       
    no_dup_mask = np.where(d2d.arcsec > match_thresh)
    no_dup_cat = cat.iloc[no_dup_mask].copy()

    return no_dup_cat, d2d, idx


def compute_tpr_fpr(predictions, labels):
    tpr = np.array(predictions[labels.astype(np.bool)] == 1)
    tpr = np.sum(tpr)/len(tpr)
    
    fpr = np.array(predictions[~labels.astype(np.bool)] == 1)
    fpr = np.sum(fpr)/len(fpr)
    
    return tpr, fpr


def dataframe_to_fits(df, fits_path):
    table = Table.from_pandas(df)
    table.write(fits_path)

def fits_to_dataframe(fits_path):
    table = Table.read(fits_path)
    df = table.to_pandas()
    return df

def get_smoothed_count_field(x_all, y_all, imdim=1024, smooth_sig=20):
    
    H, xedge, yedge = np.histogram2d(x_all, y_all, [np.arange(imdim), np.arange(imdim)])
    gf = gaussian_filter(H.transpose(), sigma=smooth_sig)
    gf -= np.median(gf)
    
    return gf

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


def crossmatch_unWISE_PanSTARRS(base_cat='unWISE', fieldstr_tail_PS='BootesA_DR1_richard_feder', fieldstr='BootesA', nsig=2, \
    datdir = '/Users/luminatech/Documents/ciber2/ciber/data/cats/'):
    
    '''
    Parameters
    ----------
    
    base_cat : 'string', optional
        specifies which catalog is the reference catalog for cross matching
        Default is 'unWISE'.
        
    nsig : 'float', optional
        Positional cross matching criterion for sources is for 
        matching radius to be less than nsig x positional error.
        Default is 2.
    
    '''
    unWISE_cat_wxy = pd.read_csv('data/cats/unWISE/'+fieldstr+'/unWISE_'+fieldstr+'_filt_xy_dpos.csv')
    unWISE_src_coord = SkyCoord(ra=unWISE_cat_wxy['ra']*u.degree, dec=unWISE_cat_wxy['dec']*u.degree, frame='icrs')

    PS_cat_wxy = pd.read_csv(datdir+'PanSTARRS/filt/'+fieldstr_tail_PS+'_filt_any_band_detect.csv')
    PS_src_coord = SkyCoord(ra=PS_cat_wxy['ra']*u.degree, dec=PS_cat_wxy['dec']*u.degree, frame='icrs')

    if base_cat=='unWISE':
        idx, d2d, _ = match_coordinates_sky(unWISE_src_coord, PS_src_coord) # the order of the input catalogs matters!
        
        # Vega magnitudes
        unWISE_cat_wxy['gMeanPSFMag'] = np.array(PS_cat_wxy.iloc[idx].gMeanPSFMag) - 0.08
        unWISE_cat_wxy['rMeanPSFMag'] = np.array(PS_cat_wxy.iloc[idx].rMeanPSFMag) - 0.16
        unWISE_cat_wxy['iMeanPSFMag'] = np.array(PS_cat_wxy.iloc[idx].iMeanPSFMag) - 0.37
        unWISE_cat_wxy['zMeanPSFMag'] = np.array(PS_cat_wxy.iloc[idx].zMeanPSFMag) - 0.54
        unWISE_cat_wxy['yMeanPSFMag'] = np.array(PS_cat_wxy.iloc[idx].yMeanPSFMag) - 0.634

        unWISE_cat_wxy['mag_W1'] = np.array(22.5 - 2.5*np.log10(unWISE_cat_wxy['flux_W1']))
        unWISE_cat_wxy['mag_W2'] = np.array(22.5 - 2.5*np.log10(unWISE_cat_wxy['flux_W2']))

        unWISE_cat_wxy['x1_PS'] = np.array(PS_cat_wxy.iloc[idx].x1)
        unWISE_cat_wxy['y1_PS'] = np.array(PS_cat_wxy.iloc[idx].y1)
        unWISE_cat_wxy['ra_PS'] = np.array(PS_cat_wxy.iloc[idx].ra)
        unWISE_cat_wxy['dec_PS'] = np.array(PS_cat_wxy.iloc[idx].dec)

        unWISE_cat_wxy['dmatch_arcsec'] = d2d.arcsec
    
        unWISE_PS_xmatch_mask = np.where(np.array(unWISE_cat_wxy['dmatch_arcsec']) <= nsig*3600*np.array(unWISE_cat_wxy['dpos']))[0]
        
        crossmatch_PS_unWISE = unWISE_cat_wxy.iloc[unWISE_PS_xmatch_mask].copy()

    elif base_cat=='PanSTARRS':
        idx, d2d, _ = match_coordinates_sky(PS_src_coord, unWISE_src_coord) # the order of the input catalogs matters!
        
        PS_cat_wxy['dpos'] = np.array(unWISE_cat_wxy.iloc[idx].dpos)
        PS_cat_wxy['primary'] = np.array(unWISE_cat_wxy.iloc[idx].primary)
        PS_cat_wxy['flux_W1'] = np.array(unWISE_cat_wxy.iloc[idx].flux_W1)
        PS_cat_wxy['flux_W2'] = np.array(unWISE_cat_wxy.iloc[idx].flux_W2)
        
        # Vega magnitudes
        PS_cat_wxy['mag_W1'] = np.array(22.5 - 2.5*np.log10(PS_cat_wxy['flux_W1']))
        PS_cat_wxy['mag_W2'] = np.array(22.5 - 2.5*np.log10(PS_cat_wxy['flux_W2']))

        PS_cat_wxy['x1_unWISE'] = np.array(unWISE_cat_wxy.iloc[idx].x1)
        PS_cat_wxy['y1_unWISE'] = np.array(unWISE_cat_wxy.iloc[idx].y1)
        PS_cat_wxy['ra_unWISE'] = np.array(unWISE_cat_wxy.iloc[idx].ra)
        PS_cat_wxy['dec_unWISE'] = np.array(unWISE_cat_wxy.iloc[idx].dec)

        PS_cat_wxy['dmatch_arcsec'] = d2d.arcsec

        PS_unWISE_xmatch_mask = np.where(np.array(PS_cat_wxy['dmatch_arcsec']) <= nsig*3600*np.array(PS_cat_wxy['dpos']))[0]
        
        crossmatch_PS_unWISE = PS_cat_wxy.iloc[PS_unWISE_xmatch_mask].copy()


    return crossmatch_PS_unWISE

def panstarrs_preprocess(fieldstr, datadir='/Users/luminatech/Documents/ciber2/ciber/data/cats/', \
                        detect_any=True, detect_y=True, cat_keys=None, apply_flags=True):

    if cat_keys is None:
        cat_keys = ['ra', 'dec', 'gMeanPSFMag', 'rMeanPSFMag', 'iMeanPSFMag', 'zMeanPSFMag', 'yMeanPSFMag']
    df = pd.read_csv(datadir+'PanSTARRS/raw/'+fieldstr+'.csv')
    
    # combine the data
    df_raw = df.drop(['ObjID'],axis=1)
    
    # select flag 8 and flag 16 column
    flag_arr = df_raw.qualityFlag
    bin_values = np.flip(2 ** np.arange(11),0)
    flag_bin_arr = np.zeros([len(flag_arr),len(bin_values)],dtype=int)

    for i, flag in enumerate(flag_arr):
        bin_str = "{:011b}".format(flag)
        bin_int = list(map(int,list(bin_str)))
        flag_bin_arr[i,:] = bin_int

    flag_bin_arr8 = flag_bin_arr[:,int(np.where(bin_values==8)[0])]
    flag_bin_arr16 = flag_bin_arr[:,int(np.where(bin_values==16)[0])]

    for i,bin_value in enumerate(bin_values):
            df_raw['flag{}'.format(bin_value)] = flag_bin_arr[:,i]

    if apply_flags:
        df = df_raw[(df_raw.flag8 == 1) | (df_raw.flag16 == 1)].copy()
    else:
        df = df_raw.copy()
    # set the default value to -99
    df.replace([-999],-99,inplace=True)
    # select source having band y magnitude
    if detect_any:
        print('selecting sources detected in any of the five bands')
        sp = np.where((df['gMeanPSFMag']!=-99)|(df['rMeanPSFMag']!=-99)|\
                   (df['iMeanPSFMag']!=-99)|(df['zMeanPSFMag']!=-99)|(df['yMeanPSFMag']!=-99))[0]
        df = df.iloc[sp].copy()
        df = df.reset_index(drop=True) 
    elif detect_y:
        print('selecting sources with PSF mags in y band')
        sp = np.where((df['yMeanPSFMag']!=-99))[0]
        df = df.iloc[sp].copy()
        df = df.reset_index(drop=True)
    else:
        print('no cuts on PSF mags in grizy or y')
        
    df = df[cat_keys].copy()

    return df

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


def unWISE_flag_filter(cat, flags_unwise_val=0, flags_info_val=0, primary=True, band_merged_idx=0):
    mask = np.bool((cat.shape[0]))
    if primary:
        mask *= (cat['primary']==1)
    if flags_unwise_val is not None:
        mask *= (cat['flags_unwise'][:,band_merged_idx]==0)
    if flags_info_val is not None:
        mask *= (cat['flags_info'][:,band_merged_idx]==0)
        
    return mask

def unWISE_fluxes_to_mags(fluxes, mode='AB'):
    Vega = 22.5-2.5*np.log10(fluxes)
    
    if mode=='Vega':
        return Vega
    else:
        AB = Vega.copy()
        AB[:,0] + 2.699
        AB[:,1] + 3.339
        
        return AB
