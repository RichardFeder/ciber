import numpy as np


def get_count_field(x_all, y_all, imdim=1024, smooth=False, smooth_sig=20, mean_sub=False):
    
    H, xedge, yedge = np.histogram2d(x_all, y_all, [np.arange(imdim+1)-0.5, np.arange(imdim+1)-0.5])
    
    if smooth:
        cf = gaussian_filter(H.transpose(), sigma=smooth_sig)
    else:
        cf = H
        
    if mean_sub:
        cf -= np.mean(cf)
    
    return cf

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
    hdul.writeto(save_fpath, overwrite=True)
    
    return save_fpath

def preprocess_gal_density_maps(inst, ifield_list, catname, save=False, imdim=1024, cat_fpath_list=None, show=True, \
                               addstr=None):
    
    catalog_basepath = config.ciber_basepath+'data/catalogs/'
    
    gal_densities = np.zeros((len(ifield_list), imdim, imdim))
    
    for fieldidx, ifield in enumerate(ifield_list):
        
        if cat_fpath_list is None:
            cat_fpath = catalog_basepath+catname+'/filt/'+catname+'_CIBER_ifield'+str(ifield)+'.csv'
        else:
            cat_fpath = cat_fpath_list[fieldidx]
            
            
        if type(cat_fpath)==str:
            cat_fpath = [cat_fpath]
            
        print('CAT fpath is ', cat_fpath)
        all_cat_x, all_cat_y, all_cat_mag, all_cat_rmag,\
            all_cat_type, all_cat_z, all_cat_W1W2 = [[] for x in range(7)]
        for fpath in cat_fpath:
            
            cat_df = pd.read_csv(fpath)
            cat_x = np.array(cat_df['x'+str(inst)])
            cat_y = np.array(cat_df['y'+str(inst)])
            
            all_cat_x.extend(cat_x)
            all_cat_y.extend(cat_y)
            
            if catname=='DECaLS':
                cat_W1W2 = np.array(cat_df['mag_W1']) - np.array(cat_df['mag_W2'])
                all_cat_W1W2.extend(cat_W1W2)
                
#                 cat_zmag = np.array(cat_df['mag_z'])-np.array(cat_df['mag_W1'])
                
                cat_type= np.array(cat_df['type'])
                all_cat_type.extend(cat_type)
        
        
                cat_redshift= np.array(cat_df['z_phot_mean'])
                all_cat_z.extend(cat_redshift)
    
            elif catname=='HSC':
            
                
                cat_mag = np.array(cat_df['g_cmodel_mag'])
                all_cat_mag.extend(cat_mag)

                cat_rmag = np.array(cat_df['r_cmodel_mag'])
                all_cat_rmag.extend(cat_rmag)
                
                all_cat_z.extend(np.array(cat_df['photoz_mean']))
        all_cat_x = np.array(all_cat_x)
        all_cat_y = np.array(all_cat_y)
#         all_cat_zmag = np.array(all_cat_zmag)
#         print(np.unique(all_cat_zmag))
        if catname=='DECaLS' or catname=='HSC':
            all_cat_z = np.array(all_cat_z)
            
            if catname=='DECaLS':
                all_cat_type = np.array(all_cat_type)
                all_cat_mag = np.array(all_cat_mag)
                all_cat_W1W2 = np.array(all_cat_W1W2)
                
                plt.figure()
                plt.hist(all_cat_W1W2, bins=30)
                plt.yscale('log')
                plt.show()
                
            if catname=='HSC':
                all_cat_rmag = np.array(all_cat_rmag)

        
        mask = (all_cat_x > 0)*(all_cat_x < imdim)*(all_cat_y > 0)*(all_cat_y < imdim)
    
        print('sum of mask is ', np.sum(mask))
    
        if catname=='DECaLS':
#             addmask = (all_cat_W1W2 < 0.5)*(~np.isnan(all_cat_W1W2))*(~np.isinf(all_cat_W1W2))*(np.abs(all_cat_W1W2)<5)
#             cat_z = np.array(cat_df['mag_z'])
#             addmask = (all_cat_z > 0.5)*(all_cat_z < 0.8)
            addmask = (all_cat_type == "b'PSF'")
            print(np.sum(addmask), len(addmask))
            mask *= addmask

        
        elif catname=='HSC':
            g_r = all_cat_mag - all_cat_rmag
            g_r_mask = (~np.isnan(g_r))*(~np.isinf(g_r))*(np.abs(g_r) < 2.)*(g_r > -1.)
            med_gr = np.median(g_r[g_r_mask])

            plt.figure()
            plt.axvline(med_gr, linestyle='dashed')
            plt.scatter(g_r[g_r_mask], all_cat_mag[g_r_mask], alpha=0.05, s=2)
            plt.ylim(16, 28)
            plt.show()

            addmask = (all_cat_mag < 24.0)*(all_cat_mag > 17)*(all_cat_z > 0.05)*g_r_mask

            mask *= addmask
#             mask *= (all_cat_zmag > -2.5)
#             mask *= (all_cat_zmag > np.median(all_cat_zmag[np.abs(all_cat_zmag < 10)]))
#             mask *= (all_cat_zmag < 23)*(all_cat_zmag > 20)
#             cat_r = np.array(cat_df['mag_r'])
        

        all_cat_x = all_cat_x[mask]
        all_cat_y = all_cat_y[mask]

        counts = get_count_field(all_cat_x, all_cat_y, imdim=imdim)
        
        if show:
            plot_map(counts, title=catname+' ifield '+str(ifield))
        
        gal_densities[fieldidx] = counts
        
    
    if save:
        save_fpath = save_gal_density(inst, ifield_list, gal_densities, catname, addstr=addstr)
    else:
        save_fpath = None
    
    return save_fpath
