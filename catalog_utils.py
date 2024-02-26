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

            ra_arr = np.array(df['ra']).astype(float)
            dec_arr = np.array(df['dec']).astype(float)

            src_coord = SkyCoord(ra=ra_arr*u.degree, dec=dec_arr*u.degree, frame='icrs', unit=u.degree)

            # src_coord = SkyCoord(ra=df['ra']*u.degree, dec=df['dec']*u.degree, frame='icrs', unit=u.degree)

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
        imcut_idx = np.where(inst1_mask | inst2_mask) # keep sources that fall onto at least one imager
        df = df.iloc[imcut_idx].copy()
        df = df.reset_index(drop=True)
        
    # write x, y to df
    order = order[:2] + ['x1','y1','x2','y2'] + order[2:]
    dfout = df[order].copy()
    
    return dfout

def check_for_catalog_duplicates(cat, cat2=None, match_thresh=0.1, nthneighbor=2, ra_errors=None, dec_errors=None, zscore=1):
    
    ra_arr = np.array(cat['ra']).astype(float)
    dec_arr = np.array(cat['dec']).astype(float)

    print('ra arr has length ', len(ra_arr))
    print('dec arr has length ', len(dec_arr))

    cat_src_coord = SkyCoord(ra=ra_arr*u.degree, dec=dec_arr*u.degree, frame='icrs', unit=u.degree)

    # cat_src_coord = SkyCoord(ra=cat['ra']*u.degree, dec=cat['dec']*u.degree, frame='icrs', unit=u.degree)
    
    if cat2 is None:
        cat2_src_coord = cat_src_coord
    else:
        ra_arr2 = np.array(cat2['ra']).astype(float)
        dec_arr2 = np.array(cat2['dec']).astype(float)

        cat2_src_coord = SkyCoord(ra=ra_arr2*u.degree, dec=dec_arr2*u.degree, frame='icrs', unit=u.deg)

        # cat2_src_coord = SkyCoord(ra=cat2['ra']*u.degree, dec=cat['dec']*u.degree, frame='icrs', unit=u.deg)

    # choose nthneighbor=2 to not just include the same source
    idx, d2d, _ = match_coordinates_sky(cat_src_coord, cat2_src_coord, nthneighbor=nthneighbor) # there is an order specific element to this
    
    if ra_errors is not None:
        match_thresh = zscore*np.sqrt(ra_errors**2 + dec_errors**2)
        print('match threshes is ', match_thresh)
       
    no_dup_mask = np.where(d2d.arcsec > match_thresh)
    no_dup_cat = cat.iloc[no_dup_mask].copy()

    return no_dup_cat, d2d, idx


def combine_catalog_dfs_no_duplicates(catalog_dfs, match_threshes=None, verbose=True):
    ''' Make sure catalog_dfs is ordered where first catalog df is union crossmatch across all catalogs'''
    
    src_coords = []
    if match_threshes is None:
        match_threshes = [0.1 for x in range(len(catalog_dfs)-1)]
    
    for cat_df in catalog_dfs:
        cat_src_coord = SkyCoord(ra=cat_df['ra']*u.degree, dec=cat_df['dec']*u.degree, frame='icrs', unit=u.deg)
        src_coords.append(cat_src_coord)
    
    if verbose:
        print('initial df has length ', len(catalog_dfs[0]))
    
    df_list = [catalog_dfs[0]] # this is where order matters
    
    for j in range(len(catalog_dfs)-1):
        
        if j==0:
            idx_xmatch, d2d_xmatch, _ = match_coordinates_sky(src_coords[j+1], src_coords[0])
        else:
            idx_xmatch, d2d_xmatch, _ = match_coordinates_sky(src_coords[j+1], df_merge_coords)
        
        nodup_mask = np.where(d2d_xmatch.arcsec > match_threshes[j])[0] # find all non-duplicates
        nodup_xmatch = catalog_dfs[j+1].iloc[nodup_mask].copy()
        if verbose:
            print('nodup xmatch has length ', len(nodup_xmatch))
        df_list.append(nodup_xmatch)
        
        df_merge = pd.concat(df_list, ignore_index=True)
        df_merge_coords = SkyCoord(ra=df_merge['ra']*u.degree, dec=df_merge['dec']*u.degree, frame='icrs', unit=u.deg)
        if verbose:
            print('df merge has length ', len(df_merge))
        
    return df_merge


def detect_cat_conditions_Jband(catalog_df, j_key='j_mag_best'):
    Jcondition = (np.abs(catalog_df[j_key]) < 90.)
    return Jcondition

def detect_cat_conditions_unWISE(catalog_df, W1key='mag_W1', W2key='mag_W2'):
    W1_condition = (~np.isinf(catalog_df[W1key]))&(~np.isnan(catalog_df[W1key]))
    W2_condition = (~np.isinf(catalog_df[W2key]))&(~np.isnan(catalog_df[W2key]))
    unWISE_detect_condition = (W1_condition | W2_condition)
    
    return unWISE_detect_condition

def detect_cat_conditions_PanSTARRS(catalog_df, tailmagstr='MeanPSFMag'):
    
    PS_g_condition = ((~np.isinf(catalog_df['g'+tailmagstr]))&(~np.isnan(catalog_df['g'+tailmagstr]))&(np.abs(catalog_df['g'+tailmagstr])<30.))
    PS_r_condition = ((~np.isinf(catalog_df['r'+tailmagstr]))&(~np.isnan(catalog_df['r'+tailmagstr]))&(np.abs(catalog_df['r'+tailmagstr])<30.))
    PS_i_condition = ((~np.isinf(catalog_df['i'+tailmagstr]))&(~np.isnan(catalog_df['i'+tailmagstr]))&(np.abs(catalog_df['i'+tailmagstr])<30.))
    PS_z_condition = ((~np.isinf(catalog_df['z'+tailmagstr]))&(~np.isnan(catalog_df['z'+tailmagstr]))&(np.abs(catalog_df['z'+tailmagstr])<30.))
    PS_y_condition = ((~np.isinf(catalog_df['y'+tailmagstr]))&(~np.isnan(catalog_df['y'+tailmagstr]))&(np.abs(catalog_df['y'+tailmagstr])<30.))
    PanSTARRS_detect_condition = (PS_g_condition | PS_r_condition | PS_i_condition | PS_z_condition | PS_y_condition)

    return PanSTARRS_detect_condition

def detect_cat_conditions_J_unWISE_PanSTARRS(catalog_df, j_key='j_mag_best', W1key='mag_W1', W2key='mag_W2'):
    
    Jcondition = detect_cat_conditions_Jband(catalog_df, j_key=j_key)
    unWISE_detect_condition = detect_cat_conditions_unWISE(catalog_df, W1key=W1key, W2key=W2key)
    PanSTARRS_detect_condition = detect_cat_conditions_PanSTARRS(catalog_df)
    
    return Jcondition, unWISE_detect_condition, PanSTARRS_detect_condition


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


def crossmatch_unWISE_PanSTARRS_112822(unWISE_cat_wxy, PS_cat_wxy,\
                                base_cat='unWISE', fieldstr_tail_PS='BootesA_DR1_richard_feder', fieldstr='BootesA',\
                                dpos_max = 1.0, nsig=2, datdir = 'data/catalogs/'):
    
    '''
    Parameters
    ----------
    
    base_cat : 'string', optional
        specifies which catalog is the reference catalog for cross matching
        Default is 'unWISE'.
        
    
    '''

    Vega_to_AB = dict({'g':-0.08, 'r':0.16, 'i':0.37, 'z':0.54, 'y':0.634, 'J':0.91, 'H':1.39})
    PS_bands = ['g', 'r', 'i', 'z', 'y']
    
    unWISE_src_coord = SkyCoord(ra=unWISE_cat_wxy['ra']*u.degree, dec=unWISE_cat_wxy['dec']*u.degree, frame='icrs', unit=u.deg)
    PS_src_coord = SkyCoord(ra=PS_cat_wxy['ra']*u.degree, dec=PS_cat_wxy['dec']*u.degree, frame='icrs', unit=u.deg)

    if base_cat=='unWISE':
        idx, d2d, _ = match_coordinates_sky(unWISE_src_coord, PS_src_coord) # the order of the input catalogs matters!
        
        PS_cat_match = PS_cat_wxy.iloc[idx]
        
        # Vega magnitudes
        for PS_band in PS_bands:
            unWISE_cat_wxy[PS_band+'MeanPSFMag'] = np.array(PS_cat_match[PS_band+'MeanPSFMag']) - Vega_to_AB[PS_band]

        if fieldstr != 'UDS':
            unWISE_cat_wxy['x1_PS'] = np.array(PS_cat_match['x1'])
            unWISE_cat_wxy['y1_PS'] = np.array(PS_cat_match['y1'])
        unWISE_cat_wxy['ra_PS'] = np.array(PS_cat_match['ra'])
        unWISE_cat_wxy['dec_PS'] = np.array(PS_cat_match['dec'])

        unWISE_cat_wxy['dmatch_arcsec'] = d2d.arcsec
    
        unWISE_PS_xmatch_mask = np.where(np.array(unWISE_cat_wxy['dmatch_arcsec']) <= dpos_max)[0]
        
        crossmatch_PS_unWISE = unWISE_cat_wxy.iloc[unWISE_PS_xmatch_mask].copy()
        return crossmatch_PS_unWISE, unWISE_cat_wxy, unWISE_PS_xmatch_mask

    elif base_cat=='PanSTARRS':
        idx, d2d, _ = match_coordinates_sky(PS_src_coord, unWISE_src_coord) # the order of the input catalogs matters!
        
        PS_cat_wxy['primary'] = np.array(unWISE_cat_wxy.iloc[idx].primary)
        # PS_cat_wxy['flux_W1'] = np.array(unWISE_cat_wxy.iloc[idx].flux_W1)
        # PS_cat_wxy['flux_W2'] = np.array(unWISE_cat_wxy.iloc[idx].flux_W2)
        
        # Vega magnitudes
        # PS_cat_wxy['mag_W1'] = np.array(22.5 - 2.5*np.log10(PS_cat_wxy['flux_W1']))
        # PS_cat_wxy['mag_W2'] = np.array(22.5 - 2.5*np.log10(PS_cat_wxy['flux_W2']))

        if fieldstr != 'UDS':
            PS_cat_wxy['x1_unWISE'] = np.array(unWISE_cat_wxy.iloc[idx]['x1'])
            PS_cat_wxy['y1_unWISE'] = np.array(unWISE_cat_wxy.iloc[idx]['y1'])
            PS_cat_wxy['x2_unWISE'] = np.array(unWISE_cat_wxy.iloc[idx]['x2'])
            PS_cat_wxy['y2_unWISE'] = np.array(unWISE_cat_wxy.iloc[idx]['y2'])

        PS_cat_wxy['ra_unWISE'] = np.array(unWISE_cat_wxy.iloc[idx]['ra'])
        PS_cat_wxy['dec_unWISE'] = np.array(unWISE_cat_wxy.iloc[idx]['dec'])
        PS_cat_wxy['dmatch_arcsec'] = d2d.arcsec

        PS_unWISE_xmatch_mask = np.where(np.array(PS_cat_wxy['dmatch_arcsec']) <= dpos_max)[0]
        
        crossmatch_PS_unWISE = PS_cat_wxy.iloc[PS_unWISE_xmatch_mask].copy()

        return crossmatch_PS_unWISE, PS_cat_wxy, PS_unWISE_xmatch_mask


def crossmatch_unWISE_PanSTARRS(unWISE_cat_wxy=None, PS_cat_wxy=None, base_cat='unWISE', fieldstr_tail_PS='BootesA_DR1_richard_feder', fieldstr='BootesA', nsig=2, \
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

    Vega_to_AB = dict({'g':-0.08, 'r':0.16, 'i':0.37, 'z':0.54, 'y':0.634, 'J':0.91, 'H':1.39})
    
    if unWISE_cat_wxy is None:
        unWISE_cat_wxy = pd.read_csv('data/cats/unWISE/'+fieldstr+'/unWISE_'+fieldstr+'_filt_xy_dpos.csv')
    
    if PS_cat_wxy is None:
        PS_cat_wxy = pd.read_csv(datdir+'PanSTARRS/filt/'+fieldstr_tail_PS+'_filt_any_band_detect.csv')
    
    unWISE_src_coord = SkyCoord(ra=unWISE_cat_wxy['ra']*u.degree, dec=unWISE_cat_wxy['dec']*u.degree, frame='icrs', unit=u.deg)
    PS_src_coord = SkyCoord(ra=PS_cat_wxy['ra']*u.degree, dec=PS_cat_wxy['dec']*u.degree, frame='icrs', unit=u.deg)

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
        # unWISE_PS_xmatch_mask = np.where(np.array(unWISE_cat_wxy['dmatch_arcsec']) <= 1.0)[0]
        
        crossmatch_PS_unWISE = unWISE_cat_wxy.iloc[unWISE_PS_xmatch_mask].copy()

        return crossmatch_PS_unWISE, unWISE_cat_wxy, unWISE_PS_xmatch_mask

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


def crossmatch_IBIS_unWISE_PS(field, df_IBIS_wxy=None, crossmatch_unWISE_PS=None, \
        datdir='/Users/luminatech/Documents/ciber2/ciber/data/cats/', xmatch_cut=False, xmatch_radius=0.5, \
        compute_WISE_mags=False, PS_mAB_to_mVega=False):

    '''

    Parameters
    ----------

    field : 'string'
        Field to perform crossmatching on. 

    datdir : 'string', optional

    xmatch_cut : 'bool', optional
        if True, only return crossmatches with source positions within xmatch_radius of each other. 
        Default is False.
    xmatch_radius : 'float', optional
        maximum cross matching radius permitted for merged catalogs if xmatch_cut is False.
        Default is 0.5 [arcseconds].

    Returns
    -------

    crossmatch_unWISE_PS_IBIS : class 'pandas.core.frame.DataFrame'
        Dataframe containing crossmatched unWISE + PanSTARRS + IBIS catalog.

    '''

    if df_IBIS_wxy is None:
        print('loading IBIS catalog because it was not provided')
        df_IBIS_wxy = pd.read_csv(datdir+'IBIS_'+field+'_filt_wxy_ML_selected_best.csv')
    if crossmatch_unWISE_PS is None:
        print('loading crossmatch catalog because it was not provided')
        crossmatch_unWISE_PS = pd.read_csv(datdir+'unWISE/'+field+'/unWISE_PS_crossmatch_2sig_dpos_'+field+'.csv')

    IBIS_src_coord = SkyCoord(ra=df_IBIS_wxy['ra']*u.degree, dec=df_IBIS_wxy['dec']*u.degree, frame='icrs')

    crossmatch_unWISE_PS_src_coord = SkyCoord(ra=crossmatch_unWISE_PS['ra']*u.degree, dec=crossmatch_unWISE_PS['dec']*u.degree, frame='icrs')

    idx_IBIS_unWISE_PS, d2d_IBIS, _ = match_coordinates_sky(crossmatch_unWISE_PS_src_coord, IBIS_src_coord) # there is an order specific element to this

    # IBIS sources are already in Vega magnitude system
    crossmatch_unWISE_PS['j_mag_best'] = np.array(df_IBIS_wxy.iloc[idx_IBIS_unWISE_PS].j_mag_best)
    crossmatch_unWISE_PS['h_mag_best'] = np.array(df_IBIS_wxy.iloc[idx_IBIS_unWISE_PS].h_mag_best)
    crossmatch_unWISE_PS['k_mag_best'] = np.array(df_IBIS_wxy.iloc[idx_IBIS_unWISE_PS].k_mag_best)

    crossmatch_unWISE_PS['j_magerr_best'] = np.array(df_IBIS_wxy.iloc[idx_IBIS_unWISE_PS].j_magerr_best)
    crossmatch_unWISE_PS['h_magerr_best'] = np.array(df_IBIS_wxy.iloc[idx_IBIS_unWISE_PS].h_magerr_best)
    crossmatch_unWISE_PS['k_magerr_best'] = np.array(df_IBIS_wxy.iloc[idx_IBIS_unWISE_PS].k_magerr_best)

    crossmatch_unWISE_PS['dmatch_arcsec_IBIS'] = d2d_IBIS.arcsec

    # converting WISE fluxes to Vega magnitudes
    if compute_WISE_mags:
        print('computing WISE mags (Vega)..')
        crossmatch_unWISE_PS['mag_W2'] = np.array(22.5 - 2.5*np.log10(crossmatch_unWISE_PS['flux_W2']))
        crossmatch_unWISE_PS['mag_W1'] = np.array(22.5 - 2.5*np.log10(crossmatch_unWISE_PS['flux_W1']))

    # PanSTARRS catalog PSF magnitudes are given w.r.t. the AB system, correcting to Vega for consistent calibration
    if PS_mAB_to_mVega:
        print('converting PanSTARRS AB magnitudes to Vega..')
        crossmatch_unWISE_PS['gMeanPSFMag'] -= -0.08
        crossmatch_unWISE_PS['rMeanPSFMag'] -= 0.16
        crossmatch_unWISE_PS['iMeanPSFMag'] -= 0.37
        crossmatch_unWISE_PS['zMeanPSFMag'] -= 0.54
        crossmatch_unWISE_PS['yMeanPSFMag'] -= 0.634

    if xmatch_cut:
        IBIS_PS_unWISE_xmatch_mask = np.where(np.array(crossmatch_unWISE_PS['dmatch_arcsec_IBIS']) <= xmatch_radius)[0]

        crossmatch_unWISE_PS_IBIS = crossmatch_unWISE_PS.iloc[IBIS_PS_unWISE_xmatch_mask].copy()
    else:
        crossmatch_unWISE_PS_IBIS = crossmatch_unWISE_PS.copy()

    return crossmatch_unWISE_PS_IBIS

def filter_trilegal_cat(trilegal_cat, m_min=4, m_max=17, I_band_idx=16):
    
    filtered_trilegal_cat = np.array([x for x in trilegal_cat if x[I_band_idx]<m_max and x[I_band_idx]>m_min])

    return filtered_trilegal_cat

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


def return_color_df(cat_df, band1, band2):
    return np.array(cat_df[band1])-np.array(cat_df[band2])

def return_several_colors_df(cat_df, list_of_bands):
    colors = []
    for band_pair in list_of_bands:
        colors.append(return_color_df(cat_df, band_pair[0], band_pair[1]))
        
    return colors

def read_in_sdwfs_cat(cbps, catalog_basepath='data/Spitzer/sdwfs_catalogs/', catalog_fname='SDWFS_ch1_stack.v34.txt', bootes_ifield_list=[6,7], bootes_cen_ras=[217.2, 218.4], bootes_cen_decs=[33.2, 34.8]):
    
    sdwfs_cat = np.loadtxt(catalog_basepath+catalog_fname, skiprows=21)

    sdwfs_ra = sdwfs_cat[:,0]
    sdwfs_dec = sdwfs_cat[:,1]

    sdwfs_ch1_magauto = sdwfs_cat[:,18]
    sdwfs_ch2_magauto = sdwfs_cat[:,19]

    sdwfs_ch1_magauto_err = sdwfs_cat[:,22]
    sdwfs_ch2_magauto_err = sdwfs_cat[:,23]
    
    remerge_cat = np.array([sdwfs_ra, sdwfs_dec, sdwfs_ch1_magauto, sdwfs_ch2_magauto, sdwfs_ch1_magauto_err, sdwfs_ch2_magauto_err])
    
    print('remerge cat has shape', remerge_cat.shape)
    
    for i, bootes_ifield in enumerate(bootes_ifield_list):
        
        radecmask = (sdwfs_ra > bootes_cen_ras[i]-2)*(sdwfs_ra < bootes_cen_ras[i]+2)*(sdwfs_dec > bootes_cen_decs[i]-2)*(sdwfs_dec < bootes_cen_decs[i]+2)
        
        ciber_fov_cat = remerge_cat[:,np.where(radecmask)[0]]
        
        print('ciber fov cat has shape', ciber_fov_cat.shape)
        
        sdwfs_ciber_df = pd.DataFrame(ciber_fov_cat.transpose(), columns=['ra', 'dec', 'CH1_mag_auto', 'CH2_mag_auto', 'CH1_mag_auto_err', 'CH2_mag_auto_err'])
        
        print(sdwfs_ciber_df)
        sdwfs_ciber_filt = catalog_df_add_xy(cbps.ciber_field_dict[bootes_ifield], sdwfs_ciber_df, datadir=config.ciber_basepath+'data/')

        sdwfs_ciber_filt, _, _ = check_for_catalog_duplicates(sdwfs_ciber_filt)
        
        plt.figure()
        plt.scatter(sdwfs_ciber_filt['x1'], sdwfs_ciber_filt['y1'], s=1, color='k')
        plt.xlim(0, 1024)
        plt.ylim(0, 1024)
        plt.show()
        print('saving to ', catalog_basepath+'sdwfs_wxy_CIBER_ifield'+str(bootes_ifield)+'.csv')
        sdwfs_ciber_filt.to_csv(catalog_basepath+'sdwfs_wxy_CIBER_ifield'+str(bootes_ifield)+'.csv')


def read_in_decals_cat(cbps, ifield_list=[4, 5, 6, 7, 8], catalog_basepath=None, convert_to_Vega=True, with_photz=True, \
                      quadstr='A'):
    
    if catalog_basepath is None:
        catalog_basepath = config.ciber_basepath+'data/catalogs/'
        
    decals_basepath = catalog_basepath+'DECaLS/'
    
    Vega_to_AB = dict({'mag_g':-0.08, 'mag_r':0.16, 'mag_i':0.37, 'mag_z':0.54, 'J':0.91, 'H':1.39, 'K':1.85, 'mag_W1':2.699, 'mag_W2':3.339})


    if with_photz: # DR9 for now
        
        mag_keys = ['mag_g', 'mag_r', 'mag_z', 'mag_W1', 'mag_W2']

        catnames = ('ra', 'dec', 'mag_g', 'mag_r', \
            'mag_z', 'mag_W1', 'mag_W2', 'allmask_g',\
            'allmask_r', 'allmask_z', 'type', 'z_phot_mean', 'z_phot_std', 'z_spec')
        
        dtypes = (float, float, float, float, \
              float, float, float, float, \
              float, float, '|S15', float, float, float)
        
    else:
        mag_keys = ['mag_g', 'mag_r', 'mag_i', 'mag_z', 'mag_W1', 'mag_W2']

        catnames = ('ra', 'dec', 'mag_g', 'mag_r', 'mag_i',\
                    'mag_z', 'mag_W1', 'mag_W2', 'allmask_g',\
                    'allmask_r', 'allmask_i', 'allmask_z', 'type')
    
        dtypes = (float, float, float, float, float, \
              float, float, float, float, \
              float, float, float,'|S15')

    
    for fieldidx, ifield in enumerate(ifield_list):
        
        fieldname = cbps.ciber_field_dict[ifield]
        
        if ifield=='UDS':
            decals_fpath = decals_basepath+'DECaLS_uds.txt'
        else:
            
            if with_photz:
                decals_fpath = decals_basepath+fieldname+'/dr9_'+cbps.ciber_field_dict[ifield]+'_photz_'+quadstr+'.txt'
            else:
                decals_fpath = decals_basepath+fieldname+'/DECaLS_'+cbps.ciber_field_dict[ifield]+'_deep_v2_'+quadstr+'.txt'
        
#         decals_cat = np.loadtxt(decals_fpath, delimiter=',', skiprows=1)
        decals_cat = np.loadtxt(decals_fpath, delimiter=',', skiprows=1, \
                               dtype={'names': catnames,
                          'formats': dtypes})

#         decals_cat = np.genfromtxt(decals_fpath, delimiter=',', dtype=None)
        
        decals_df = pd.DataFrame(decals_cat, columns=list(catnames))

        print(np.array(decals_df['type']))
        print(np.array(decals_df['ra']))
        
        plt.figure()
        plt.hist(np.array(decals_df['z_phot_mean']), bins=np.linspace(0, 2, 20))
        plt.xlabel('z')
        plt.show()
            
        if convert_to_Vega:
            for key in mag_keys:
                print('subtracting ', key, 'by ', Vega_to_AB[key])
                decals_df[key] -= Vega_to_AB[key]
        
        
        plt.figure(figsize=(6,5))
        for key in mag_keys:
            plt.hist(np.array(decals_df[key]), bins=np.linspace(10, 25, 20), histtype='step', label=key)
            
        plt.yscale('log')
        plt.legend()
        plt.show()
            
        if ifield != 'UDS':
            decals_filt = catalog_df_add_xy(cbps.ciber_field_dict[ifield], decals_df, datadir=config.ciber_basepath+'data/')

            decals_filt, _, _ = check_for_catalog_duplicates(decals_filt)

            plt.figure()
            plt.scatter(decals_filt['x1'], decals_filt['y1'], s=1, color='k')
            plt.xlim(0, 1024)
            plt.ylim(0, 1024)
            plt.show()
            
        else:
            decals_filt, _, _ = check_for_catalog_duplicates(decals_df)

        if with_photz:
            save_fpath = decals_basepath+'filt/dr9_CIBER_ifield'+str(ifield)+'_photz_'+quadstr+'.csv'
        else:
            save_fpath = decals_basepath+'filt/decals_CIBER_ifield'+str(ifield)+'_photz_'+quadstr+'.csv'
        print('saving to ', save_fpath)
        decals_filt.to_csv(save_fpath)

def read_in_hsc_swire_cat(cbps, catalog_basepath=None, convert_to_Vega=True):

    if catalog_basepath is None:
        catalog_basepath = config.ciber_basepath+'data/catalogs/'

    hsc_fpath = catalog_basepath+'HSC/SWIRE_full.csv'
    hsc_df = pd.read_csv(hsc_fpath)
    hsc_df = hsc_df.rename(columns={"g_ra":"ra", "g_dec":"dec"})
    
    # hsc_df
    hsc_df['ra'] = np.array(hsc_df['ra']).astype(float)
    hsc_df['dec'] = np.array(hsc_df['dec']).astype(float)

    plt.figure(figsize=(10, 10))
    plt.scatter(hsc_df['ra'], hsc_df['dec'], s=2, alpha=0.01, color='k')
    plt.show()
    
    hsc_filt = catalog_df_add_xy('SWIRE', hsc_df, datadir=config.ciber_basepath+'data/')
    hsc_filt, _, _ = check_for_catalog_duplicates(hsc_filt)
        
    save_fpath =  catalog_basepath+'HSC/HSC_deep_CIBER_ifield8_photz.csv'
    print('saving to ', save_fpath)
    hsc_filt.to_csv(save_fpath)

    return hsc_filt

def read_in_ukidss_cat(catalog_basepath, ifield):
    uk_path_dict = dict({'train':'ukidss_dr11_plus_UDS_12_7_20.csv', 4:'ukidss_dr11_plus_elat10_0_102220.csv', 5:'ukidss_dr11_plus_elat30_0_102220.csv', 8:'ukidss_dr11_plus_SWIRE_0_102220.csv'})
    Vega_to_AB = dict({'g':-0.08, 'r':0.16, 'i':0.37, 'z':0.54, 'y':0.634, 'J':0.91, 'H':1.39, 'K':1.85})

    if ifield=='train':
        ukidss_uds = pd.read_csv(catalog_basepath+'UKIDSS/'+uk_path_dict[ifield], skiprows=9)
        ukidss_uds['ra'] = np.array(ukidss_uds['# ra']).astype(float)
    elif ifield in [4, 5]:
        ukidss_uds = pd.read_csv(catalog_basepath+'UKIDSS/'+uk_path_dict[ifield], skiprows=10)
    elif ifield==8:
        ukidss_uds = pd.read_csv(catalog_basepath+'UKIDSS/'+uk_path_dict[ifield], skiprows=8, header=1)

    ukidss_uds['J_Vega'] = ukidss_uds['jAB'] - Vega_to_AB['J']
    ukidss_uds['H_Vega'] = ukidss_uds['hAB'] - Vega_to_AB['H']
    ukidss_uds['K_Vega'] = ukidss_uds['kAB'] - Vega_to_AB['K']
    
    return ukidss_uds

def read_in_flamingos_cat(cbps, bootes_ifield_list=[6,7], catalog_basepath=None):

    if catalog_basepath is None:
        catalog_basepath = config.ciber_basepath+'data/catalogs/bootes_dr1_flamingos/'
    
    flam_cat_fpath = catalog_basepath+'BOOTES_j_V1.0.cat'
    
    flam_cat = np.loadtxt(flam_cat_fpath, skiprows=74, dtype=str)

    flam_cat_J = flam_cat[:,17].astype(float)
    flam_cat_ra = flam_cat[:,7].astype(float)
    flam_cat_dec = flam_cat[:,8].astype(float)
    
    remerge_cat = np.array([flam_cat_ra, flam_cat_dec, flam_cat_J])
    
    flam_df = pd.DataFrame(remerge_cat.transpose(), columns=['ra', 'dec', 'J'])

    
    print(flam_df['ra'])
    for i, bootes_ifield in enumerate(bootes_ifield_list):
        

        flam_df_filt = catalog_df_add_xy(cbps.ciber_field_dict[bootes_ifield], flam_df, datadir=config.ciber_basepath+'data/')

        flam_df_filt, _, _ = check_for_catalog_duplicates(flam_df_filt)
        
        plt.figure()
        plt.scatter(flam_df_filt['x1'], flam_df_filt['y1'], s=1, color='k')
        plt.xlim(0, 1024)
        plt.ylim(0, 1024)
        plt.show()
        
        save_fpath = catalog_basepath+'flamingos_J_wxy_CIBER_ifield'+str(bootes_ifield)+'.csv'
        print('saving to ', save_fpath)
        flam_df_filt.to_csv(save_fpath)

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



def twomass_srcmap_masking_cat_prep(twomass_df, mean_color_correct, ciber_mock_obj, ifield, twomass_max_mag = 16., nx=1024, ny=1024, inst=1):
    twomass_mag_str_dict = dict({1:'j_m', 2:'h_m'})

    magstr = twomass_mag_str_dict[inst]

    twomass_bright_mask = np.where(twomass_df[magstr] < twomass_max_mag)[0]
    twomass_df_filt = twomass_df.iloc[twomass_bright_mask].copy()
    twomass_df_filt['zband_mask'] = twomass_df_filt[magstr]+mean_color_correct

    return twomass_df_filt


def unWISE_flag_filter(cat, flags_unwise_val=0, flags_info_val=0, primary=True, band_merged_idx=0):
    mask = np.bool((cat.shape[0]))
    if primary:
        mask *= (cat['primary']==1)
    if flags_unwise_val is not None:
        mask *= (cat['flags_unwise'][:,band_merged_idx]==flags_unwise_val)
    if flags_info_val is not None:
        mask *= (cat['flags_info'][:,band_merged_idx]==flags_info_val)
        
    return mask

def unWISE_flag_filter_112822(cat, flags_unwise_val=0, flags_info_val=0, primary=True, band_merged_idx=0):
    mask = np.bool((cat.shape[0]))
    if primary:
        mask *= (cat['primary']==1)
    if flags_unwise_val is not None:
        mask *= (cat['flags_unwise_1']==flags_unwise_val)
        mask *= (cat['flags_unwise_2']==flags_unwise_val)

    if flags_info_val is not None:
        mask *= (cat['flags_info_1']==flags_info_val)
        mask *= (cat['flags_info_2']==flags_info_val)
        
    return mask

def unWISE_fluxes_to_mags_112822(fluxes, mode='AB'):
    Vega = 22.5-2.5*np.log10(fluxes)
    
    if mode=='Vega':
        return Vega
    else:
        print("Converting to AB magnitudes..")
        AB = Vega.copy()
        AB[0,:] += 2.699
        AB[1,:] += 3.339
        
        return AB

def compute_number_counts(inst, mag_min=17.0, mag_max=30.0, dm=0.5, apply_dm=True, plot=False):

    
    if inst==1:
        counts_match_measurements = np.loadtxt('data/Jband_number_counts_vs_mag.csv', delimiter=',', dtype=float)
        counts_match_helgason = np.loadtxt('data/on_curve_Jband.csv', delimiter=',', dtype=float)
        
        bandstr = 'J'
#         lam = 1.25
        lam = 1.05
        Vega_to_AB = 0.91
    else:
        counts_match_measurements = np.loadtxt('data/hband_counts_permag.csv', delimiter=',', dtype=float)
        counts_match_helgason = np.loadtxt('data/on_curve_Hband.csv', delimiter=',', dtype=float)
        bandstr = 'H'
#         lam =1.63
        lam= 1.79
        Vega_to_AB = 1.35
    
    mags_raw_meas = counts_match_measurements[:,0]
    mags_raw_helgason = counts_match_helgason[:,0]

    log_counts_meas = np.log10(counts_match_measurements[:,1])
    log_counts_helgason = np.log10(counts_match_helgason[:,1])
    

    mags_sample = np.arange(mag_min, mag_max, dm) 
    mags_sample += Vega_to_AB
    
    print('mags sample:', mags_sample)

    interp_fn_meas = scipy.interpolate.interp1d(mags_raw_meas, log_counts_meas)
    interp_fn_helgason = scipy.interpolate.interp1d(mags_raw_helgason, log_counts_helgason)

    interp_log_counts_meas = interp_fn_meas(mags_sample)
    interp_log_counts_helgason = interp_fn_helgason(mags_sample)
    
    if plot:
        
        mags_fine = np.linspace(np.min(mags_sample), np.max(mags_sample), 100)
        
        log_counts_fine_meas = interp_fn_meas(mags_fine)
        log_counts_fine_helgason = interp_fn_helgason(mags_fine)

        plt.figure(figsize=(5, 4))        
        plt.plot(mags_fine, 10**log_counts_fine_meas, color='k', label='Direct galaxy counts (HFE)', linestyle='dashed')
        plt.plot(mags_fine, 10**log_counts_fine_helgason, color='k', label='Helgason model best fit')
        plt.xlabel(bandstr+'-band magnitude (AB)', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.5)
        plt.xlim(15, 22)
        plt.ylabel('Number mag$^{-1}$ deg$^{-2}$', fontsize=14)
        plt.yscale('log')
#         plt.savefig('/Users/richardfeder/Downloads/direct_vs_helgason_TM'+str(inst)+'_counts.png', bbox_inches='tight', dpi=200)
        plt.show()
    
    
    nu_Inu = cmock.mag_2_nu_Inu(mags_sample, lam_eff=lam*1e-6*u.m)*cmock.pix_sr

    nm_dm_meas = 10**(interp_log_counts_meas)
    nm_dm_helgason = 10**(interp_log_counts_helgason)

    nm_dm_persr_meas = nm_dm_meas/3.046e-4 # convert to sr-1
    nm_dm_persr_helgason = nm_dm_helgason/3.046e-4 # convert to sr-1

    if apply_dm:
        nm_dm_persr_meas *= dm
        nm_dm_persr_helgason *= dm

    poisson_var_meas = np.sum(nm_dm_persr_meas*(nu_Inu**2))
    poisson_var_helgason = np.sum(nm_dm_persr_helgason*(nu_Inu**2))

    return poisson_var_meas.value, poisson_var_helgason.value



def get_2MASS_slopes(inst, ifield):
    ''' was used in early versions of surface brightness calibration to assess color dependence of results '''
    base_fluc_path = config.exthdpath+'ciber_fluctuation_data/'
    catalog_basepath = base_fluc_path+'catalogs/'
    field_name = cbps.ciber_field_dict[ifield]
    twomass_cat = pd.read_csv(catalog_basepath+'2MASS/filt/2MASS_filt_rdflag_wxy_'+field_name+'_Jlt17.5.csv')
    
    twomass_J = np.array(twomass_cat['j_m'])
    twomass_H = np.array(twomass_cat['h_m'])
    twomass_K = np.array(twomass_cat['k_m'])
    
    # convert to AB mags
    
    twomass_J_AB = twomass_J + cbps.Vega_to_AB[1]
    twomass_H_AB = twomass_H + cbps.Vega_to_AB[2]
    twomass_K_AB = twomass_K + 1.85
    
    twomass_J_mask = (twomass_J_AB < 15)*(twomass_J_AB > 12)
    
    twomass_J_flux = 10**(-0.4*(twomass_J_AB-23.9))
    twomass_H_flux = 10**(-0.4*(twomass_H_AB-23.9))
    twomass_K_flux = 10**(-0.4*(twomass_K_AB-23.9))
    
    print(np.sum(twomass_J_mask))
    
    plt.figure(figsize=(10, 5))
    for x in range(len(twomass_J_AB)):
        if twomass_J_mask[x]:
            plt.plot([1.2, 1.6, 2.2], [twomass_J_flux[x], twomass_H_flux[x], twomass_K_flux[x]], marker='.', color='k', alpha=0.01)
    plt.yscale('log')

    plt.show()
    
    plt.figure()
    plt.scatter((twomass_J_AB-twomass_H_AB)[twomass_J_mask], (twomass_H_AB-twomass_K_AB)[twomass_J_mask], s=10, c=twomass_J_AB[twomass_J_mask])
    cbar = plt.colorbar(label='$J_{AB}$')
#     cbar.ax.tick_params()
#     cbar.('$J_{AB}$')
    plt.xlim(-0.3, 0.3)
    plt.ylim(-0.7, 0.0)
    plt.xlabel('J-H', fontsize=16)
    plt.ylabel('H-K', fontsize=16)
    plt.grid(alpha=0.5)
    plt.show()

