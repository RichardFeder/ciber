import numpy as np
import config
from astropy.io import fits
import pyfftw
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from ciber.plotting.plot_utils import plot_map

from ciber.core.powerspec_pipeline import *
from ciber.plotting.galaxy_plots import *


def completeness_model(m, m_lim=25.0, sigma_m=0.5):
    
    zsm = (m - m_lim)/sigma_m
    
    c = 0.5*(1 - scipy.special.erf(zsm))
    
    return c

def return_default_gal_cat_dict():
    
    gal_cat_dict = dict({})
    
    # file paths
    gal_cat_dict['catalog_basepath'] = config.ciber_basepath+'data/catalogs/'
    
    # CIBER parameters
    gal_cat_dict['ifield_list'] = [4, 5, 6, 7, 8]
    gal_cat_dict['ciber_dimx'] = 1024
    
    # unWISE
    gal_cat_dict['rgb_mode'] = None
    gal_cat_dict['w1_mag_cut'] = None
    gal_cat_dict['wise_bands'] = ['mag_W1', 'mag_W2']
    
    # DECaLS
    gal_cat_dict['decals_gal'] = False
    gal_cat_dict['decals_redshift_key'] = 'z_phot_mean'

    
    # HSC
    gal_cat_dict['hsc_mag_max'] = None
    gal_cat_dict['hsc_mag_min'] = None
    gal_cat_dict['hsc_bands'] = ['g_cmodel_mag', 'r_cmodel_mag']
    gal_cat_dict['hsc_redshift_key'] = 'z_phot_mean'

    # photo-z parameters (when available)
    
    gal_cat_dict['zmin'] = None
    gal_cat_dict['zmax'] = None
    
    return gal_cat_dict


class cat_select():
    
    def __init__(self, gal_dict=None):
        
        if gal_dict is not None:
            print('Loading galaxy parameter dict into cat_select()')
            self.gal_dict = gal_dict
                
    
    def decals_mask(self):

        if self.gal_dict['decals_gal']:
            mask = (self.cat_type != "b'PSF'")
        else:
            mask = (self.cat_type == "b'PSF'")
        return mask
    
    
    def wise_cat_mask(self):
        
        mask = np.ones_like(self.cat_x).astype(int)
        
        if self.gal_dict['rgb_mode'] is not None:
            print('RGB mode is ', self.gal_dict['rgb_mode'])
            mask *= wise_rgb_cuts(self.cat_W1, self.cat_W2, self.gal_dict['rgb_mode'])

        if self.gal_dict['w1_mag_cut'] is not None:
            print('Cutting WISE sources with W1 > '+str(self.gal_dict['w1_mag_cut']))
            mask *= (self.cat_W1 < self.gal_dict['w1_mag_cut'])

        return mask
        
    
    def footprint_mask(self):
        # in FOV
        mask = (self.cat_x > 0)*(self.cat_x < self.gal_dict['ciber_dimx'])*(self.cat_y > 0)*(self.cat_y < self.gal_dict['ciber_dimx'])

        # in redshift range if specified
        if 'redshift' in self.cat_stack_labs:
            if self.gal_dict['zmin'] is not None:
                print('Removing z < '+str(self.gal_dict['zmin'])+' sources')
                mask *= (self.cat_z > gal_dict['zmin'])
            if self.gal_dict['zmax'] is not None:
                print('Removing z > '+str(self.gal_dict['zmax'])+' sources')
                mask *= (self.cat_z < self.gal_dict['zmax'])

        return mask.astype(int)
    
    
    def hsc_cat_mask(self):
        
#         g_r = all_cat_mag - all_cat_rmag
#         g_r_mask = (~np.isnan(g_r))*(~np.isinf(g_r))*(np.abs(g_r) < 2.)*(g_r > -1.)
#         med_gr = np.median(g_r[g_r_mask])

        mask = np.ones_like(self.cat_x).astype(int)
        
        if self.gal_dict['hsc_mag_max'] is not None:
            print('Removing sources with g > '+str(self.gal_dict['hsc_mag_max']))
            mask *= (self.cat_g < self.gal_dict['hsc_mag_max'])

        if self.gal_dict['hsc_mag_min'] is not None:
            print('Removing sources with g < '+str(self.gal_dict['hsc_mag_min']))
            mask *= (self.cat_g < self.gal_dict['hsc_mag_min'])

        
        return mask
    
    
    def apply_cat_select(self, catname):
        
        mask = self.footprint_mask()       
        print(mask)
        if catname=='DECaLS':
            cat_mask = self.decals_cat_mask()
        
        elif catname=='WISE':
            cat_mask = self.wise_cat_mask()    
            
        elif catname=='HSC':
            cat_mask  = self.hsc_cat_mask()

        else:
            cat_mask = np.ones_like(mask)
            
        mask *= cat_mask
        
        return mask
            
            
    def load_cat(self, fpath, ciber_inst, catname):
        
        cat_df = pd.read_csv(fpath)
        
        self.cat_x, self.cat_y = np.array(cat_df['x'+str(ciber_inst)]), np.array(cat_df['y'+str(ciber_inst)])
        
        self.cat_stack = [self.cat_x, self.cat_y]
        self.cat_stack_labs = ['x', 'y']
            
        if catname=='DECaLS':
            self.cat_z, self.cat_type = np.array(cat_df['z_phot_mean']), np.array(cat_df['type'])
            self.cat_stack.extend([self.cat_z, self.cat_type])
            self.cat_stack_labs.extend(['redshift', 'type'])
            
        elif catname=='HSC':
            self.cat_z = np.array(cat_df['photoz_mean'])
            self.cat_g, self.cat_r = np.array(cat_df['g_cmodel_mag']), np.array(cat_df['r_cmodel_mag'])
                
            self.cat_stack.extend([self.cat_z, self.cat_g, self.cat_r])
            self.cat_stack_labs.extend(['redshift', 'mag_g', 'mag_r'])

        elif catname=='WISE':
            self.cat_W1, self.cat_W2 = np.array(cat_df['mag_W1']), np.array(cat_df['mag_W2'])
            
            self.cat_stack.extend([self.cat_W1, self.cat_W2])
            self.cat_stack_labs.extend(['mag_W1', 'mag_W2'])
            
        print('cat labels:', self.cat_stack_labs)





def gen_wget_unWISE_command(tile_list, basedir='raw_neo8/', \
                           base_url='https://portal.nersc.gov/project/cosmo/data/unwise/neo8/unwise-catalog/', \
                           save_fname=None):
    
    ''' Generates a shell script to grab catalogs from WISE tiles indicated by tile_list. '''
    all_commands = []
    
    for tile in tile_list:
        command = 'wget -I '+basedir+' '+base_url+'/objcat/'+tile+'.cat.fits'
        all_commands.append(command) 
        
#         for band in [1, 2]:
#             command = 'wget -I '+basedir+' '+base_url+'/cat/'+tile+'.'+str(band)+'.cat.fits'
#             all_commands.append(command)

    if save_fname is not None:
        filename = basedir+save_fname
        print('Writing wget commands to ', filename)
        with open(filename, 'w') as f:
            for command in all_commands:
                f.write(command + "\n")
        os.chmod(filename, 0o755)    
        
    return all_commands

def get_count_field(x_all, y_all, imdim=1024, smooth=False, smooth_sig=20, mean_sub=False):
    
    H, xedge, yedge = np.histogram2d(x_all, y_all, [np.arange(imdim+1)-0.5, np.arange(imdim+1)-0.5])
    
    if smooth:
        cf = gaussian_filter(H.transpose(), sigma=smooth_sig)
    else:
        cf = H
        
    if mean_sub:
        cf -= np.mean(cf)
    
    return cf


def compute_weighted_cl(all_cl, all_clerr):
    
    all_cl = np.array(all_cl)
    all_clerr = np.array(all_clerr)
    
    variance = all_clerr**2
    
    weights = 1./variance
    
    cl_sumweights = np.sum(weights, axis=0)
    
    weighted_variance = 1./cl_sumweights
    
    field_averaged_std = np.sqrt(weighted_variance)
    
    field_averaged_cl = np.nansum(weights*all_cl, axis=0)/cl_sumweights
    
    return field_averaged_cl, field_averaged_std

def save_gal_density(inst, ifield_list, gal_densities, catname, basepath=None, addstr=None):
    
    if basepath is None:
        basepath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/gal_density/'
        
        
    prim = fits.PrimaryHDU()
    
    hdul = [prim]
    
    for fieldidx, ifield in enumerate(ifield_list):
        imhdu = fits.ImageHDU(gal_densities[fieldidx], name='ifield'+str(ifield))
        
        hdul.append(imhdu)
    
    
    hdul = fits.HDUList(hdul)
    
    save_fpath = basepath+catname+'/gal_density_'+catname+'_TM'+str(inst)

    if addstr is not None:
        save_fpath += '_'+addstr
    
    save_fpath += '.fits'
    print('Saving to ', save_fpath)
    hdul.writeto(save_fpath, overwrite=True)
    
    return save_fpath


def separate_ls_catalog_by_z(zbinedges=None, ifield_list = [4, 5, 6, 7, 8], ls_cat_names = ['ra', 'dec', 'zphot'], mode='data', plot=False):
    
    ls_basepath = config.ciber_basepath+'data/ciber_x_gal/'

    cbps = CIBER_PS_pipeline()
    
    if zbinedges is None:
        zbinedges = np.array(list(np.arange(0, 1.4, 0.2))+[1.5, 2.0])


    for zidx in range(len(zbinedges)-1):

        print(str(zbinedges[zidx])+'<z<'+str(zbinedges[zidx+1]))
        
        catbasepath = ls_basepath+mode+'_catalogs/'

        ls_fpath = catbasepath+'LS_Dr8_z22_'+str(np.round(zbinedges[zidx], 1))+'_zphot_'+str(np.round(zbinedges[zidx+1], 1))
        
        if mode=='random':
            ls_fpath += '_random'
        ls_cat_full = fits.open(ls_fpath+'.fits')[1].data
            
        ls_cat_ra = ls_cat_full['ra']
        ls_cat_dec = ls_cat_full['dec']
        ls_cat_zphot = ls_cat_full['zphot']

        for fieldidx, ifield in enumerate(ifield_list):

            ra_cen, dec_cen = cbps.ra_cen_ciber_fields[ifield], cbps.dec_cen_ciber_fields[ifield]
            near_ciber_fp = (ls_cat_ra > ra_cen - 3.)*(ls_cat_ra < ra_cen + 3)*(ls_cat_dec > dec_cen - 3.)*(ls_cat_dec < dec_cen+3.)

            ls_ra_cut = ls_cat_ra[near_ciber_fp]
            ls_dec_cut = ls_cat_dec[near_ciber_fp]
            ls_zphot_cut = ls_cat_zphot[near_ciber_fp]

            ls_df = pd.DataFrame(np.array([ls_ra_cut, ls_dec_cut, ls_zphot_cut]).transpose(), columns=ls_cat_names)
            ls_filt = catalog_df_add_xy(cbps.ciber_field_dict[ifield], ls_df, datadir=config.ciber_basepath+'data/')
            ls_filt, _, _ = check_for_catalog_duplicates(ls_filt)

            if plot:
                plt.figure()
                plt.scatter(ls_filt['x1'], ls_filt['y1'], s=1, color='k')
                plt.xlim(0, 1024)
                plt.ylim(0, 1024)
                plt.show()
                
            print('for field '+str(ifield)+', there are ', len(ls_ra_cut))

            ls_save_fpath = catbasepath+'ciber_cut/ls_'+str(np.round(zbinedges[zidx], 1))+'_zphot_'+str(np.round(zbinedges[zidx+1], 1))+'_wxy_CIBER_ifield'+str(ifield)
            if mode=='random':
                ls_save_fpath += '_random'
            ls_save_fpath += '.csv'

            print('Saving catalog to ', ls_save_fpath)
            ls_filt.to_csv(ls_save_fpath)


def preprocess_gal_density_maps(inst, ifield_list, catname, save=False, cat_fpath_list=None,\
                                 show=True, addstr=None, **kwargs):
    
    
    gal_dict = return_default_gal_cat_dict()
    gal_dict = update_dicts([gal_dict], kwargs)[0]
       
    ciber_field_dict = dict({4:'elat10', 5:'elat30', 6:'Bootes B', 7:'Bootes A', 8:'SWIRE'})
    
    gal_counts = np.zeros((len(ifield_list), gal_dict['ciber_dimx'], gal_dict['ciber_dimx']))
    
    for fieldidx, ifield in enumerate(ifield_list):
        
        if cat_fpath_list is None:
            cat_fpath = catalog_basepath+catname+'/filt/'+catname+'_CIBER_ifield'+str(ifield)+'.csv'
        else:
            cat_fpath = cat_fpath_list[fieldidx]
            
        # instantiate for each field separately
        cat_sel_obj = cat_select(gal_dict)
        cat_sel_obj.load_cat(cat_fpath, inst, catname)
        
        mask = cat_sel_obj.apply_cat_select(catname)

        cat_x_sel, cat_y_sel = cat_sel_obj.cat_x[np.where(mask)[0]], cat_sel_obj.cat_y[np.where(mask)[0]]
        
        print('After down-selections, the '+str(catname)+' catalog for '+str(ifield)+' has '+str(len(cat_x_sel))+' sources.')
        
        counts = get_count_field(cat_x_sel, cat_y_sel, imdim=gal_dict['ciber_dimx'])
        
        if show:
            plot_map(counts, title=catname+' ifield '+str(ifield))
        
        gal_counts[fieldidx] = counts
        
    if save:
        
        save_fpath = save_gal_density(inst, ifield_list, gal_counts, catname, addstr=addstr)
    else:
        save_fpath = None
    
    return save_fpath


def preprocess_ls_density_maps(inst, zbinedges, ifield_list, 
                            save=False, imdim=1024, plot=False,
                            mode='data', catname='LS', remove_wen_cmgs=False):
    
    ls_basepath = config.ciber_basepath+'data/ciber_x_gal/'
    
    ciber_field_dict = dict({4:'elat10', 5:'elat30', 6:'Bootes B', 7:'Bootes A', 8:'SWIRE'})

    gal_counts = np.zeros((len(ifield_list), imdim, imdim))
    
    all_gal_counts = []
    
    ngal_perz_perfield = np.zeros((len(zbinedges)-1, len(ifield_list)))

    for zidx, z0 in enumerate(zbinedges[:-1]):
        
        z1 = zbinedges[zidx+1]
        
        addstr = str(np.round(z0, 1))+'_z_'+str(np.round(z1, 1))

        if mode=='random':
            addstr += '_random'
        
        for fieldidx, ifield in enumerate(ifield_list):

            cat_fpath = ls_basepath+mode+'_catalogs/ciber_cut/ls_'+str(np.round(z0, 1))+'_zphot_'+str(np.round(z1, 1))+'_wxy_CIBER_ifield'+str(ifield)

            if mode=='random':
                cat_fpath+= '_random'
            cat_fpath += '.csv'

            print('reading from ', cat_fpath)
            cat_df = pd.read_csv(cat_fpath)


            if remove_wen_cmgs:
                wen_basepath = 'data/catalogs/wen_cluster_gals/'
                wen_fpath = wen_basepath+'wen_cluster_member_gals_CIBER_ifield'+str(ifield)+'_wxy.csv'
                wen_df = pd.read_csv(wen_fpath)

                ls_src_coord = SkyCoord(ra=cat_df['ra']*u.degree, dec=cat_df['dec']*u.degree, frame='icrs', unit=u.deg)
                
                wen_src_coord = SkyCoord(ra=wen_df['ra']*u.degree, dec=wen_df['dec']*u.degree, frame='icrs', unit=u.deg)

                idx_xmatch, d2d_xmatch, _ = match_coordinates_sky(ls_src_coord, wen_src_coord)
                nodup_mask = np.where(d2d_xmatch.arcsec > 0.2)[0] # find all non-duplicates

                print('cat df before removing wen cmgs is ', len(cat_df))
                print('wen has ', len(wen_df), 'sources')
                cat_df = cat_df.iloc[nodup_mask].copy()
                print('cat df after removing wen cmgs is ', len(cat_df))


            cat_x = np.array(cat_df['x'+str(inst)])
            cat_y = np.array(cat_df['y'+str(inst)])
            cat_zphot = np.array(cat_df['zphot'])
        
            mask = (cat_x > 0)*(cat_x < imdim)*(cat_y > 0)*(cat_y < imdim)



            cat_x = cat_x[mask]
            cat_y = cat_y[mask]
            
            ngal_perz_perfield[zidx, fieldidx] = len(cat_x)

            counts = get_count_field(cat_x, cat_y, imdim=imdim)

            # gal_density = (counts - np.nanmean(counts))/np.nanmean(counts)
            
            if plot:
                plt.figure()
                plt.title(str(z0)+'$<z_{phot}<$'+str(z1), fontsize=14)
                plt.hist(cat_zphot[mask], bins=30, histtype='step')
                plt.xlabel('zphot')
                plt.ylabel('$N_g$')
                plt.show()
                
                plot_map(counts, title='LS ifield '+str(ifield))
                # plot_map(gal_density, title='Gal density LS ifield '+str(ifield))
                
            gal_counts[fieldidx] = counts
                        
        all_gal_counts.append(gal_counts)
        
        if save:
        
            save_fpath = save_gal_density(inst, ifield_list, gal_counts, catname, addstr=addstr)
        else:
            save_fpath = None
        
    return all_gal_counts, ngal_perz_perfield


def wise_rgb_cuts(mag_W1, mag_W2, rgb_mode):
    
    color_W1W2 = mag_W1-mag_W2
    
    if rgb_mode=='blue':
        w2mask = (mag_W2 > 16.7)
        w1w2mask = (color_W1W2 < 0.3+((17.-mag_W2)/4))
        
    elif rgb_mode=='green':
        w2mask = (mag_W2 > 16.7)
        w1w2mask = (color_W1W2 < 0.8+((17.-mag_W2)/4))
        w1w2mask *= (color_W1W2 > 0.3+((17.-mag_W2)/4))

        
    elif rgb_mode=='red':
        w2mask = (mag_W2 > 16.2)
        w1w2mask = (color_W1W2 < 0.8+((17.-mag_W2)/4))

    rgb_mask = w2mask*w1w2mask
        
    return rgb_mask


# def plot_gal_ps_vs_redshift(inst, zbinedges, catname='LS', figsize=(5, 4), startidx=0, endidx=-1, \
#                            xlim=[150, 1.1e5], ylim=[1e-4, 2e2], colors=['b', 'r'], \
#                              textstr=None, textxpos=200, textypos=5e1, text_fs=16, alph=0.6, \
#                              bbox_to_anchor=[-0.05, 1.25], legend_fs=10, capsize=3, markersize=3, \
#                             addstrs=None, headstr=None):


# def plot_cross_ps_vs_redshift(inst, zbinedges, lb, all_fieldav_cl_cross, all_fieldav_clerr_cross, catname='LS', figsize=(5, 4), startidx=2, endidx=-1, \
#                              xlim=[150, 1.1e5], ylim=[1e-4, 2e2], legend_fs=16, capsize=3, markersize=3, alph=0.8, \
#                              textxpos=280, textypos=1e2, text_fs=12, color=None, color_inst=['b', 'C3'], bbox_to_anchor=[2.0, 1.4]):
    

# def plot_fieldav_ciber_gal_ps(inst_list, catname, addstr=None, labels=None, \
#                              figsize=(6, 5), capsize=3, markersize=3, plot_perfield=False, \
#                              startidx=0, endidx=-1, xlim=[150, 1.1e5], ylim=[1e-4, 2e2], colors=['b', 'r'], \
#                              textstr=None, textxpos=200, textypos=5e1, text_fs=16, alph=0.6, \
#                              bbox_to_anchor=[-0.05, 1.25], legend_fs=10, mask_frac=0.7):
    

# def plot_perfield_gal_auto(catname, inst, addstr=None, figsize=(5, 4), capsize=3, markersize=3, startidx=2, endidx=-1, \
#                           xlim=[300, 1.05e5], legend_fs=10, ifield_list=[4, 5, 6, 7, 8], alph=0.7, \
#                           ylim=[1e-4, 2e2]):
    

