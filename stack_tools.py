import matplotlib
import matplotlib.pyplot as plt
from ciber_mocks import *
import numpy as np
from scipy import interpolate
import os
import astropy
import astropy.wcs as wcs
import config
from ciber_powerspec_pipeline import *
from ps_pipeline_go import *
from ciber_sb_calibration_tools import *
from scipy.optimize import curve_fit

class stack_obj():
    
    def __init__(self, mmin_range=None, dms=None):
        
        self.all_mean_fluxes = []
        self.all_mean_post = []
        self.all_nsrc = []
        self.all_posx = []
        self.all_posy = []
        self.all_aper_flux = []
        self.all_aper_flux_unc = []
        # self.all_mean_predmag_Vega = []
        self.all_mean_predmag_AB = []
        self.mmin_range = mmin_range
        self.dms = dms
        
        
    def append_results(self, aper_flux, aper_flux_unc, mean_post, posx, posy, mean_flux, nsrc, mean_predmag_AB):
        
        self.all_aper_flux.append(aper_flux)
        self.all_aper_flux_unc.append(aper_flux_unc)
        self.all_mean_post.append(mean_post)
        self.all_posx.append(posx)
        self.all_posy.append(posy)
        self.all_mean_fluxes.append(mean_flux)
        self.all_nsrc.append(nsrc)
        # self.all_mean_predmag_Vega.append(mean_predmag_Vega)
        self.all_mean_predmag_AB.append(mean_predmag_AB)

def stack_in_mag_bins_pred_flam(bootes_ifield, mask_base_path=None, catalog_basepath=None, \
                               mmin_range = [16.0, 16.5, 17.0, 17.5, 18.0], dx=4, mag_mask=15.0):
    
    cbps = CIBER_PS_pipeline()

    # can only do for 1.1 micron
    inst = 1
    
    bootes_fieldidx=bootes_ifield-4
    fieldname = cbps.ciber_field_dict[bootes_ifield]
    
    mmin_range = np.array(mmin_range)

    
    if mask_base_path is None:
        mask_base_path = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/masks/'

    if catalog_basepath is None:
        catalog_basepath = config.ciber_basepath+'data/catalogs/'
        
    
    catalog_fpath_pred = catalog_basepath + 'mask_predict/mask_predict_unWISE_PS_fullmerge_'+fieldname+'_ukdebias.csv'
    catalog_fpath_flam = catalog_basepath + 'bootes_dr1_flamingos/flamingos_J_wxy_CIBER_ifield'+str(bootes_ifield)+'.csv'
    
    if bootes_ifield==6:
        xlim = [550, 900]
        ylim = [150, 900]

    else:
        xlim = [50, 250]
        ylim = [150, 800]
    
    
    dms = list(mmin_range[1:]-mmin_range[:-1])
    dms.append(dms[-1])
    dms = np.array(dms)
    
    stack_obj_pred = stack_obj(mmin_range=mmin_range, dms=dms)
    stack_obj_flam = stack_obj(mmin_range=mmin_range, dms=dms)
    
    mag_mask = min(mag_mask, np.min(mmin_range))
    print("mag mask is ", mag_mask)
    print('dms is ', dms)
    
    for m, m_min in enumerate(mmin_range):
        m_max = m_min + dms[m]

        mean_post_pred, sum_post_pred, std_post_pred, sum_counts_pred,\
            nsrc_pred, posx_pred, posy_pred, weighted_aper_flux_pred, weighted_aper_unc_pred = calc_stacked_fluxes_new(cbps, inst, bootes_fieldidx, mask_base_path, catalog_fpath_pred, m_min, m_max, cat_type='predict', \
                                                                                       xlim=xlim, ylim=ylim, mag_mask=mag_mask, dx=dx, mask_frac_min=0.9)
        mean_post_flam, sum_post_flam, std_post_flam, sum_counts_flam, nsrc_flam,\
            posx_flam, posy_flam, weighted_aper_flux_flam, weighted_aper_unc_flam = calc_stacked_fluxes_new(cbps, inst, bootes_fieldidx, mask_base_path, catalog_fpath_flam, m_min, m_max, cat_type='flamingos', \
                                                                                       xlim=xlim, ylim=ylim, mag_mask=mag_mask, dx=dx)

        std_post_pred[std_post_pred==0] = np.inf
        std_post_flam[std_post_flam==0] = np.inf


        plt.figure(figsize=(6, 6))
        plt.title(fieldname+', '+str(m_min)+'$<J<$'+str(m_min+dms[m]), fontsize=14)
        plt.scatter(posx_pred, posy_pred, color='k', marker='+', label='Predicted catalog')
        plt.scatter(posx_flam, posy_flam, color='r', marker='x', alpha=0.8, label='FLAMINGOS catalog')
        plt.legend()
        plt.grid()
    #     plt.savefig('/Users/richardfeder/Downloads/flamingos_vs_pred_catpos_mmin='+str(m_min)+'_mmax='+str(m_min+dm)+'_'+fieldname+'.png', bbox_inches='tight', \
    #                dpi=300)
        plt.show()

        invvar_pred = 1./std_post_pred**2
        invvar_flam = 1./std_post_flam**2

        sumweights_pred = np.nansum(invvar_pred)
        sumweights_flam = np.nansum(invvar_flam)

        mean_flux_pred = np.nansum(invvar_pred*mean_post_pred)/sumweights_pred
        mean_flux_flam = np.nansum(mean_post_flam*invvar_flam)/sumweights_flam

        stack_obj_pred.append_results(weighted_aper_flux_pred, weighted_aper_unc_pred, mean_post_pred, posx_pred, posy_pred, mean_flux_pred, nsrc_pred)
        stack_obj_flam.append_results(weighted_aper_flux_flam, weighted_aper_unc_flam, mean_post_flam, posx_flam, posy_flam, mean_flux_flam, nsrc_flam)

    return stack_obj_pred, stack_obj_flam
    
def grab_flam_only_sources(bootes_ifield, catalog_basepath=None, mask_base_path=None, trim_edge=50, m_max_pred=18.5, m_max_flam=18.5, min_dr=2, \
                          mmin_range=None, m_min_start=15.0, dx=4, mask_frac_min=0.9):
    
    cbps = CIBER_PS_pipeline()
    
    inst = 1
    if catalog_basepath is None:
        catalog_basepath = config.ciber_basepath+'data/catalogs/'
        
    if mask_base_path is None:
        mask_base_path = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/masks/'

            
    if bootes_ifield==6:
        xlim = [550, 900]
        ylim = [150, 900]

    else:
        xlim = [50, 250]
        ylim = [150, 800]
    
    fieldname = cbps.ciber_field_dict[bootes_ifield]
    bootes_fieldidx = bootes_ifield - 4
    
    catalog_fpath_pred = catalog_basepath + 'mask_predict/mask_predict_unWISE_PS_fullmerge_'+fieldname+'_ukdebias.csv'
    catalog_fpath_flam = catalog_basepath + 'bootes_dr1_flamingos/flamingos_J_wxy_CIBER_ifield'+str(bootes_ifield)+'.csv'

    flam_df = pd.read_csv(catalog_fpath_flam)
    flam_x = np.array(flam_df['x'+str(inst)])
    flam_y = np.array(flam_df['y'+str(inst)])
    flam_J = np.array(flam_df['J'])

    pred_df = pd.read_csv(catalog_fpath_pred)
    pred_x = np.array(pred_df['x'+str(inst)])
    pred_y = np.array(pred_df['y'+str(inst)])
    pred_J = np.array(pred_df['J_Vega_predict'])
    
    pred_mask = (pred_J < m_max_pred)*(pred_x > trim_edge)*(pred_x < 1023-trim_edge)*(pred_y > trim_edge)*(pred_y < 1023-trim_edge)
    pred_cat_sel = np.array([pred_x, pred_y, pred_J]).transpose()
    pred_cat_sel = pred_cat_sel[pred_mask,:]
    
    flam_mask = (flam_J < m_max_flam)*(flam_x > trim_edge)*(flam_x < 1023-trim_edge)*(flam_y > trim_edge)*(flam_y < 1023-trim_edge)
    flam_cat_sel = np.array([flam_x, flam_y, flam_J]).transpose()
    flam_cat_sel = flam_cat_sel[flam_mask,:]
    
    flam_only_mask = np.zeros_like(flam_cat_sel[:,0]).astype(int)

    plt.figure(figsize=(11,6))
    plt.subplot(1,2,1)
    plt.title('Predicted catalog, '+fieldname, fontsize=14)
    plt.scatter(pred_cat_sel[:,0], pred_cat_sel[:,1], color='k', marker='+', alpha=0.1)
    plt.xlim(0, 1023)
    plt.ylim(0, 1023)
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
    
    plt.subplot(1,2,2)
    plt.title('FLAMINGOS catalog, '+fieldname, fontsize=14)
    plt.scatter(flam_cat_sel[:,0], flam_cat_sel[:,1], color='k', marker='+', alpha=0.1)

    plt.xlim(0, 1023)
    plt.ylim(0, 1023)
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
    plt.tight_layout()
#     plt.savefig('figures/flam_vs_pred_positions_Jlt18_ifield'+str(bootes_ifield)+'.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    mindr_dist = []
    for k in range(flam_cat_sel.shape[0]):

        dxpos = flam_cat_sel[k,0] - pred_cat_sel[:,0]
        dypos = flam_cat_sel[k,1] - pred_cat_sel[:,1]
        dr = np.sqrt(dxpos**2 + dypos**2)

        mindr_dist.append(np.min(dr))

    mindr_dist = np.array(mindr_dist)

    flam_only_mask = (mindr_dist > min_dr) # pixels
    
    plt.figure(figsize=(6,5))
    plt.title('FLAMINGOS x Predicted catalog', fontsize=14)
    plt.hist(mindr_dist[flam_only_mask], bins=np.logspace(-1, 1, 20), histtype='step')
    plt.hist(mindr_dist[~flam_only_mask], bins=np.logspace(-1, 1, 20), histtype='step')
    plt.xscale('log')
    plt.xlabel('Nearest-neighbor distance [pixels]', fontsize=14)
    plt.ylabel('$N_{src}$', fontsize=14)
    plt.show()
    
    flam_only_cat = flam_cat_sel[flam_only_mask,:]
    flam_match_cat = flam_cat_sel[~flam_only_mask,:]
    print('flam only:', flam_only_cat.shape)

    plt.figure(figsize=(11,6))
    plt.subplot(1,2,1)
    plt.title('FLAMINGOS with match, $J<$'+str(m_max_pred), fontsize=16)
    plt.scatter(flam_match_cat[:,0], flam_match_cat[:,1], color='k', marker='x', alpha=0.2)
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
    plt.subplot(1,2,2)

    plt.title('FLAMINGOS only sources, $J<$'+str(m_max_pred), fontsize=16)
    plt.scatter(flam_only_cat[:,0], flam_only_cat[:,1], color='k', marker='x', alpha=0.2)
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
    plt.tight_layout()
#     plt.savefig('figures/flam_only_sources_dist_bootes_ifield'+str(bootes_ifield)+'.png', bbox_inches='tight', dpi=300)
    plt.show()

    plt.figure()
    plt.hist(flam_only_cat[:,2], bins=np.linspace(m_min_start, m_max_flam, 10))
    plt.xlabel('$J$ magnitude [Vega]', fontsize=14)
    plt.ylabel('$N_{src}$', fontsize=14)
    plt.show()
    
    if mmin_range is None:
        mmin_range = np.array([m_min_start])
        dms = np.array([m_max_flam - m_min_range])
    
    else:
        mmin_range = np.array(mmin_range)
        dms = list(mmin_range[1:]-mmin_range[:-1])
        dms.append(dms[-1])
        dms = np.array(dms)

    stack_obj_flamonly = stack_obj(mmin_range=mmin_range, dms=dms)
    
    for m, m_min in enumerate(mmin_range):
        
        print('shape of flam only cat is ', flam_only_cat.shape)
        mean_post_flamonly, sum_post_flamonly, std_post_flamonly,\
        sum_counts_flamonly, nsrc_flamonly, posx_flamonly, posy_flamonly,\
            weighted_aper_flux_flamonly, weighted_aper_unc_flamonly = calc_stacked_fluxes_new(cbps, inst, bootes_fieldidx, mask_base_path, catalog_fpath_pred, m_min, m_min+dms[m], cat_type='predict', \
                                                                                   xlim=xlim, ylim=ylim, mag_mask=m_min_start, dx=dx, mask_frac_min=mask_frac_min, \
                                                                                cat_x=flam_only_cat[:,0], cat_y=flam_only_cat[:,1], cat_mag=flam_only_cat[:,2], \
                                                                                         skip_nn=False)
        
        
        std_post_flamonly[std_post_flamonly==0] = np.inf
        invvar_flamonly = 1./std_post_flamonly**2
        sumweights_flamonly = np.nansum(invvar_flamonly)
        mean_flux_flamonly = np.nansum(mean_post_flamonly*invvar_flamonly)/sumweights_flamonly

        stack_obj_flamonly.append_results(weighted_aper_flux_flamonly, weighted_aper_unc_flamonly, mean_post_flamonly, posx_flamonly, posy_flamonly,\
                                          mean_flux_flamonly, nsrc_flamonly)
        
    return stack_obj_flamonly


def calc_stacked_fluxes_new(cbps, inst, fieldidx_choose, mask_base_path, catalog_fpath, m_min, m_max, cat_type='predict', \
                       dx=5, trim_edge=50, mag_mask=15.0, bkg_rad=0.35, xlim=None, ylim=None, \
                       mask_tail = 'maglim_J_Vega_17.5_111323_ukdebias', mask_frac_min=0.9, \
                       cat_x=None, cat_y=None, cat_mag=None, skip_nn=True, plot=False):

    
    # process ciber maps first
    ifield_list = [4, 5, 6, 7, 8]
    
    bandstr_dict = dict({1:'J', 2:'H'})
    bandstr = bandstr_dict[inst]
    
    data_type='observed'
    config_dict, pscb_dict, float_param_dict, fpath_dict = return_default_cbps_dicts()
    ciber_mock_fpath = config.exthdpath+'ciber_mocks/'
    fpath_dict, list_of_dirpaths, base_path, trilegal_base_path = set_up_filepaths_cbps(fpath_dict, inst, 'test', '112022',\
                                                                                    datestr_trilegal='112022', data_type=data_type, \
                                                                                   save_fpaths=True)

    
    ifield_choose = ifield_list[fieldidx_choose]
    ciber_maps, masks = [np.zeros((len(ifield_list), 1024, 1024)) for x in range(2)]
    dc_template = cbps.load_dark_current_template(inst, verbose=True, inplace=False)
    
    
    flight_im = cbps.load_flight_image(ifield_choose, inst, inplace=False)
    flight_im *= cbps.cal_facs[inst]
    flight_im -= dc_template*cbps.cal_facs[inst]

    bkg_mask = gen_bkg_mask(dx, bkg_rad=bkg_rad)
    if plot:
        plot_map(bkg_mask, figsize=(4,4))
    # load mask
    mask_fpath = mask_base_path+'maglim_'+bandstr+'_Vega_'+str(mag_mask)+'_111323_ukdebias/joint_mask_ifield'+str(ifield_choose)+'_inst'+str(inst)+'_observed_maglim_'+bandstr+'_Vega_'+str(mag_mask)+'_111323_ukdebias.fits'
    print('opening mask from ', mask_fpath)
    mask = fits.open(mask_fpath)[1].data

    # load catalog
    
    print('mmin, mmax = ', m_min, m_max)
    if cat_x is None:

        cat_df = pd.read_csv(catalog_fpath)
        cat_x = np.array(cat_df['x'+str(inst)])
        cat_y = np.array(cat_df['y'+str(inst)])
        if cat_type=='predict':
            cat_mag = np.array(cat_df[bandstr+'_Vega_predict'])
        elif cat_type=='flamingos':
            cat_mag = np.array(cat_df[bandstr])
                
        magmask = (cat_mag > m_min)*(cat_mag < m_max)*(cat_x > trim_edge)*(cat_x < 1023-trim_edge)*(cat_y > trim_edge)*(cat_y < 1023-trim_edge)
    
    else:
        magmask = (cat_mag > m_min)*(cat_mag < m_max)
        
    if xlim is not None:
        xmask = (cat_x > xlim[0])*(cat_x < xlim[1])
        magmask *= xmask
        
    if ylim is not None:
        ymask = (cat_y > ylim[0])*(cat_y < ylim[1])
        magmask *= ymask
    
    cal_src_posx = cat_x[magmask]
    cal_src_posy = cat_y[magmask]
    cal_src_predmag = cat_mag[magmask]
    
    print('cal src posx has length ', len(cal_src_posx))
    neighbor_mask = (cat_mag < m_max+0.2)
    all_postage_stamps, all_postage_stamps_mask, post_bool = grab_postage_stamps(inst, cal_src_posx, cal_src_posy, flight_im, mask, dx, mask_frac_min=mask_frac_min, \
                                                                                skip_nn=skip_nn, neighbor_src_posx=cat_x[neighbor_mask], neighbor_src_posy=cat_y[neighbor_mask])        
        
    sum_post = np.zeros_like(all_postage_stamps[0])
    
    sum_counts = np.zeros_like(sum_post)
    
    all_aper_flux, all_aper_var, all_aper_post, all_cat_predmags = [[] for x in range(4)]

    for n in range(len(all_postage_stamps)):
        if post_bool[n]==1:
            
            aper_flux, aper_flux_var, bkg_level, bkg_var, aper_post = aper_phot(all_postage_stamps[n], all_postage_stamps_mask[n], bkg_mask, mode='mean', plot=False)
            
            all_aper_post.append(aper_post)
            all_aper_flux.append(aper_flux)
            all_aper_var.append(aper_flux_var)
            
            all_cat_predmags.append(cal_src_predmag[n])
            
            indiv_post = all_postage_stamps[n]*all_postage_stamps_mask[n]
            sum_post += aper_post            
            sum_counts += all_postage_stamps_mask[n]
                
    all_aper_post = np.array(all_aper_post)
    
    mean_post = sum_post / sum_counts
    
    std_post = np.nanstd(all_aper_post, axis=0)/np.sqrt(sum_counts)
    
    aper_flux_weights = 1./np.array(all_aper_var)
    
    weighted_aper_flux = np.nansum(np.array(all_aper_flux)*aper_flux_weights)/np.nansum(aper_flux_weights)
    weighted_aper_unc = np.sqrt(1./np.nansum(aper_flux_weights))
    
    if plot:
        plot_map(mean_post, figsize=(6,6), title='mean postage')
        plot_map(std_post, figsize=(6,6), title='std_post')
        
    return mean_post, sum_post, std_post, sum_counts, np.sum(post_bool),\
            cal_src_posx, cal_src_posy, weighted_aper_flux, weighted_aper_unc, all_cat_predmags


def stack_in_mag_bins_pred(inst, ifield, mask_base_path=None, catalog_basepath=None, \
                               mmin_range = [16.0, 16.5, 17.0, 17.5, 18.0], dx=4, mag_mask=15.0, \
                          trim_edge= 50, mask_frac_min=0.95):
    
    cbps = CIBER_PS_pipeline()
    fieldidx = ifield - 4
    fieldname = cbps.ciber_field_dict[ifield]
    mmin_range = np.array(mmin_range)
    
    xlim = [trim_edge, 1023-trim_edge]
    ylim = [trim_edge, 1023-trim_edge]
    
    if mask_base_path is None:
        mask_base_path = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/masks/'

    if catalog_basepath is None:
        catalog_basepath = config.ciber_basepath+'data/catalogs/'
        
    catalog_fpath_pred = catalog_basepath + 'mask_predict/mask_predict_unWISE_PS_fullmerge_'+fieldname+'_ukdebias.csv'

    dms = list(mmin_range[1:]-mmin_range[:-1])
    dms.append(dms[-1])
    dms = np.array(dms)
    
    stack_obj_pred = stack_obj(mmin_range=mmin_range, dms=dms)
    
    mag_mask = min(mag_mask, np.min(mmin_range))

    for m, m_min in enumerate(mmin_range):
        m_max = m_min + dms[m]

        mean_post_pred, sum_post_pred, std_post_pred, sum_counts_pred,\
            nsrc_pred, posx_pred, posy_pred, weighted_aper_flux_pred, weighted_aper_unc_pred, \
                all_cat_predmags = calc_stacked_fluxes_new(cbps, inst, fieldidx, mask_base_path, catalog_fpath_pred, m_min, m_max, cat_type='predict', \
                                                                                       xlim=xlim, ylim=ylim, mag_mask=mag_mask, dx=dx, mask_frac_min=mask_frac_min)

    
        # compute mean AB flux expected from catalog
        all_cat_predmags_AB = np.array(all_cat_predmags)+cbps.Vega_to_AB[inst]
        all_cat_predfluxdens = 10**(-0.4*(all_cat_predmags_AB-23.9))

        
        mean_predfluxdens = np.mean(all_cat_predfluxdens)
        mean_predmag_AB = -2.5*np.log10(mean_predfluxdens)+23.9
    
        std_post_pred[std_post_pred==0] = np.inf
        
        
        invvar_pred = 1./std_post_pred**2
        sumweights_pred = np.nansum(invvar_pred)
        
        mean_flux_pred = np.nansum(mean_post_pred)

        stack_obj_pred.append_results(weighted_aper_flux_pred, weighted_aper_unc_pred, mean_post_pred, posx_pred, posy_pred, mean_flux_pred, nsrc_pred, \
                                     mean_predmag_AB)

    return stack_obj_pred

def ciber_stack_flux_prediction_test(inst, ifield_list = [4, 5, 6, 7, 8])

    mmin_range_Vega = np.array([16.0, 16.5, 17.0, 17.5, 18.0])+0.1

    if inst==2:
        mmin_range -= 0.5

    bandstr_dict = dict({1:'J', 2:'H'})
    bandstr = bandstr_dict[inst]
    lam_effs = np.array([1.05, 1.79])*1e-6*u.m # effective wavelength for bands

    nu_effs = const.c/lam_effs

    print('nu efs:', nu_effs)

    all_mean_post_fluxes, all_mean_post_fluxdens, \ 
        all_stack_mag_Vega_eff, all_stack_mag_AB_eff, \ 
            allfield_meanpredmag_AB = [[] for x in range(5)]

    

    # defined for specific magnitude bins used in default
    color_corr_dict = dict({1:np.array([0.84530903, 0.84347368, 0.84613712, 0.853497, 0.85937481]), \
                            2:np.array([1.0951691,  1.11242438, 1.14006154, 1.16856026, 1.19206676])})

    std_color_corr_dict = dict({1:np.array([0.07766909, 0.0359452,  0.04860506, 0.03304105, 0.03791485]), \
                            2:np.array([0.03891492, 0.04843235, 0.06048847, 0.06128753, 0.07497544])})

    for fieldidx, ifield in enumerate(ifield_list):

        sopred = stack_in_mag_bins_pred(inst, ifield, mmin_range=mmin_range_Vega, dx=3, trim_edge=50, mask_frac_min=0.9)

        print('all pred mag AB is ', sopred.all_mean_predmag_AB)

        mean_predflux = 10**(-0.4*(np.array(sopred.all_mean_predmag_AB)-23.9))

        mean_ciber_predflux = mean_predflux*color_corr_dict[inst]
        mean_ciber_predmag_AB = -2.5*np.log10(mean_ciber_predflux)+23.9
        allfield_meanpredmag_AB.append(np.array(sopred.all_mean_predmag_AB))
        
        postfig = plot_mean_posts_magbins(sopred, bandstr)
    #     postfig.savefig('figures/stack_pred/mean_posts_in_magbins_TM'+str(inst)+'_ifield'+str(ifield)+'.png', bbox_inches='tight', dpi=300)
        # convert fluxes from nW m-2 to W m-2 Hz-1
        specific_flux = 1e-9*np.array(sopred.all_mean_fluxes)*cbps.pix_sr / nu_effs[inst-1].value

        # convert to Jansky
        specific_flux *= 1e26

        # then to microJansky
        specific_flux_uJy = specific_flux * 1e6

        # color correction to UVISTA wavelength
        specific_flux_uJy /= color_corr_dict[inst]

        # AB magnitude
        stack_mag_eff = -2.5*np.log10(specific_flux_uJy)+23.9

        all_stack_mag_AB_eff.append(stack_mag_eff)
        
    return mmin_range_Vega, allfield_meanpredmag_AB, all_stack_mag_AB_eff



