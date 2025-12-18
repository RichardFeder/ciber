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
from ps_pipeline_go import *
from ciber.instrument.calibration import *
from scipy.optimize import curve_fit
from ciber.cross_correlation.spitzer_cross import *

class stack_obj():
    
    def __init__(self, mmin_range=None, dms=None):
        
        self.all_mean_fluxes = []
        self.all_mean_post = []
        self.all_nsrc = []
        self.all_posx = []
        self.all_posy = []
        self.all_aper_flux = []
        self.all_aper_flux_unc = []
        self.all_indiv_aper_flux = []
        self.all_indiv_aper_flux_var = []
        self.all_cat_pred_flux = []
        # self.all_mean_predmag_Vega = []
        self.all_mean_predmag_AB = []
        self.mmin_range = mmin_range
        self.dms = dms
        
        
    def append_results(self, aper_flux=None, aper_flux_unc=None, mean_post=None, posx=None, posy=None, mean_flux=None,\
                         nsrc=None, mean_predmag_AB=None, cat_pred_flux=None, indiv_aper_flux=None, indiv_aper_flux_var=None):
        
        if aper_flux is not None:
            self.all_aper_flux.append(aper_flux)
        if aper_flux_unc is not None:
            self.all_aper_flux_unc.append(aper_flux_unc)

        if mean_post is not None:
            self.all_mean_post.append(mean_post)
        if posx is not None:
            self.all_posx.append(posx)

        if posy is not None:
            self.all_posy.append(posy)

        if mean_flux is not None:
            self.all_mean_fluxes.append(mean_flux)

        if nsrc is not None:
            self.all_nsrc.append(nsrc)

        if mean_predmag_AB is not None:
            self.all_mean_predmag_AB.append(mean_predmag_AB)

        if indiv_aper_flux is not None:
            self.all_indiv_aper_flux.append(indiv_aper_flux)

        if indiv_aper_flux_var is not None:
            self.all_indiv_aper_flux_var.append(indiv_aper_flux_var)

        if cat_pred_flux is not None:
            self.all_cat_pred_flux.append(cat_pred_flux)



def calc_aperture_fluxes_sdwfs_mosaic(cat_df, irac_ch, ifield, m_min, m_max, dx=5, bkg_rad=0.35, mask_frac_min=0.9, \
                                    sdwfs_basepath=None, epochidx=0, plot=False, version=3):

    if sdwfs_basepath is None:
        sdwfs_basepath = config.ciber_basepath+'data/Spitzer/sdwfs/'

    bkg_mask = gen_bkg_mask(dx, bkg_rad=bkg_rad)

    # initial maps are in MJy sr-1
    sdwfs_fpath = sdwfs_basepath+'I'+str(irac_ch)+'_bootes_epoch'+str(epochidx+1)+'.v'+str(version)+'.fits'
    print('Opening ', sdwfs_fpath)
    spitz_map = fits.open(sdwfs_fpath)[0].data
    print('spitz map has shape ', spitz_map.shape)

    irac_lam_dict = dict({1:3.6, 2:4.5})

    lam = irac_lam_dict[irac_ch]
    fac = convert_MJysr_to_nWm2sr(lam)
    print("fac for lambda = "+str(lam)+" is "+str(fac))

    spitz_map /= fac

    cat_mag = np.array(cat_df['CH'+str(irac_ch)+'_mag_auto'])
    cat_x = np.array(cat_df['x'])
    cat_y = np.array(cat_df['y'])
    magmask = (cat_mag > m_min)*(cat_mag < m_max)

    cal_src_posx = cat_x[magmask]
    cal_src_posy = cat_y[magmask]
    cal_src_predmag = cat_mag[magmask]

    all_postage_stamps, _ = grab_postage_stamps_sdwfs(cal_src_posx, cal_src_posy, spitz_map, dx, bkg_mask, mask_frac_min=mask_frac_min, plot=plot)        
    

    all_aper_flux, all_aper_var, all_aper_post, all_cat_predmags = [[] for x in range(4)]

    apermask = np.ones_like(all_postage_stamps[0])

    for n in range(len(all_postage_stamps)):
            
        aper_flux, aper_flux_var, bkg_level, bkg_var, aper_post = aper_phot(all_postage_stamps[n], apermask, bkg_mask, mode='mean', plot=False)
        all_aper_post.append(aper_post)
        all_aper_flux.append(aper_flux)
        all_aper_var.append(aper_flux_var)
        
        all_cat_predmags.append(cal_src_predmag[n])
        
    return cal_src_posx, cal_src_posy, all_cat_predmags, all_aper_flux, all_aper_var



def calc_stacked_fluxes_new(cbps, inst, fieldidx_choose, mask_base_path, catalog_fpath, m_min, m_max, cat_type='predict', \
                       dx=5, trim_edge=50, mag_mask=15.0, bkg_rad=0.35, xlim=None, ylim=None, \
                       mask_tail = None, mask_frac_min=0.9, \
                       cat_x=None, cat_y=None, cat_mag=None, skip_nn=True, plot=False, \
                       sdwfs=False, irac_ch=None, epoch_av=True, epochidx=None, map_mode='gradsub_perep'):

    
    # process ciber maps first
    ifield_list = [4, 5, 6, 7, 8]
    
    bandstr_dict = dict({1:'J', 2:'H'})
    bandstr = bandstr_dict[inst]

    if mask_tail is None:

        mask_tail = 'maglim_'+bandstr+'_Vega_'+str(mag_mask)+'_111323_ukdebias'
    
    data_type='observed'
    config_dict, pscb_dict, float_param_dict, fpath_dict = return_default_cbps_dicts()
    ciber_mock_fpath = config.exthdpath+'ciber_mocks/'
    fpath_dict, list_of_dirpaths, base_path, trilegal_base_path = set_up_filepaths_cbps(fpath_dict, inst, 'test', '112022',\
                                                                                    datestr_trilegal='112022', data_type=data_type, \
                                                                                   save_fpaths=True)

    
    ifield_choose = ifield_list[fieldidx_choose]

    if sdwfs:

        spitzer_regrid_maps, spitzer_regrid_masks, \
            all_diff1, all_diff2, all_spitz_by_epoch, all_coverage_maps = load_spitzer_bootes_maps(inst, irac_ch, [ifield_choose], mask_tail, map_mode=map_mode)

        if epoch_av:
            print('Loading epoch averaged IRAC map..')
            sb_map = spitzer_regrid_maps[0]
        else:
            print('Loading epoch '+str(epochidx+1)+' IRAC map..')
            sb_map = all_spitz_by_epoch[0][epochidx]

        if plot:
            plot_map(sb_map, figsize=(5, 5), title='IRAC map')

    else:
        dc_template = cbps.load_dark_current_template(inst, verbose=True, inplace=False)
        flight_im = cbps.load_flight_image(ifield_choose, inst, inplace=False)
        sb_map = (flight_im - dc_template)*cbps.cal_facs[inst]
        # flight_im -= dc_template*cbps.cal_facs[inst]

    bkg_mask = gen_bkg_mask(dx, bkg_rad=bkg_rad)
    if plot:
        plot_map(bkg_mask, figsize=(4,4))
    # load mask

    mask_fpath = mask_base_path+mask_tail+'/joint_mask_ifield'+str(ifield_choose)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'
    print('opening mask from ', mask_fpath)
    mask = fits.open(mask_fpath)[1].data

    # load catalog
    
    print('mmin, mmax = ', m_min, m_max)
    if cat_x is None:

        cat_df = pd.read_csv(catalog_fpath)
        cat_x = np.array(cat_df['x'+str(inst)])
        cat_y = np.array(cat_df['y'+str(inst)])
        if sdwfs:
            cat_mag = np.array(cat_df['CH'+str(irac_ch)+'_mag_auto'])
            cat_flag = np.array(cat_df['sdwfs_flag'])
        else:
            if cat_type=='predict':
                cat_mag = np.array(cat_df[bandstr+'_Vega_predict'])
            elif cat_type=='flamingos':
                cat_mag = np.array(cat_df[bandstr])
                
        magmask = (cat_mag > m_min)*(cat_mag < m_max)*(cat_x > trim_edge)*(cat_x < 1023-trim_edge)*(cat_y > trim_edge)*(cat_y < 1023-trim_edge)
    else:
        magmask = (cat_mag > m_min)*(cat_mag < m_max)
        
    if sdwfs:
        magmask *= (cat_flag < 3)

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
    neighbor_mask = (cat_mag < m_max+0.5)

    # all_postage_stamps, all_postage_stamps_mask, post_bool = grab_postage_stamps(inst, cal_src_posx, cal_src_posy, sb_map, mask, dx, mask_frac_min=mask_frac_min, \
    #                                                                             skip_nn=skip_nn, neighbor_src_posx=None, neighbor_src_posy=None)        
        
    all_postage_stamps, all_postage_stamps_mask, post_bool = grab_postage_stamps(inst, cal_src_posx, cal_src_posy, sb_map, mask, dx, mask_frac_min=mask_frac_min, \
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
            cal_src_posx, cal_src_posy, weighted_aper_flux, weighted_aper_unc, all_cat_predmags, all_aper_flux, all_aper_var


def stack_in_mag_bins_pred(inst, ifield, mask_base_path=None, catalog_basepath=None, \
                               mmin_range = [16.0, 16.5, 17.0, 17.5, 18.0], dx=4, mag_mask=15.0, \
                          trim_edge= 50, mask_frac_min=0.95, sdwfs=False, irac_ch=None, epoch_av=True, epochidx=None, plot=False, \
                              map_mode='gradsub_perep_conserve_flux'):
    
    cbps = CIBER_PS_pipeline()
    fieldidx = ifield - 4
    fieldname = cbps.ciber_field_dict[ifield]
    mmin_range = np.array(mmin_range)
    
    Vega_to_AB_irac = dict({1:2.79, 2:3.26})
    
    xlim = [trim_edge, 1023-trim_edge]
    ylim = [trim_edge, 1023-trim_edge]
    
    if mask_base_path is None:
        mask_base_path = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/masks/'

    if catalog_basepath is None:
        catalog_basepath = config.ciber_basepath+'data/catalogs/'
        
    if sdwfs:
        catalog_fpath_pred = catalog_basepath + '/sdwfs/sdwfs_wxy_CIBER_ifield'+str(ifield)+'.csv'
    else:
        catalog_fpath_pred = catalog_basepath + 'mask_predict/mask_predict_unWISE_PS_fullmerge_'+fieldname+'_ukdebias.csv'

    dms = list(mmin_range[1:]-mmin_range[:-1])
    dms.append(dms[-1])
    dms = np.array(dms)
    
    stack_obj_pred = stack_obj(mmin_range=mmin_range, dms=dms)
    
    mag_mask = min(mag_mask, np.min(mmin_range))

    print('looping through magnitudes..')
    for m, m_min in enumerate(mmin_range):
        m_max = m_min + dms[m]

        mean_post_pred, sum_post_pred, std_post_pred, sum_counts_pred,\
            nsrc_pred, posx_pred, posy_pred, weighted_aper_flux_pred, weighted_aper_unc_pred, \
                all_cat_predmags, indiv_aper_flux, indiv_aper_flux_var = calc_stacked_fluxes_new(cbps, inst, fieldidx, mask_base_path, catalog_fpath_pred, m_min, m_max, cat_type='predict', \
                                                            xlim=xlim, ylim=ylim, mag_mask=mag_mask, dx=dx, mask_frac_min=mask_frac_min, \
                                                                sdwfs=sdwfs, irac_ch=irac_ch, epoch_av=epoch_av, epochidx=epochidx, plot=plot, \
                                                                                                map_mode=map_mode)

        # compute mean AB flux expected from catalog
        if sdwfs:
            Vega_to_AB_fac = Vega_to_AB_irac[irac_ch]
        else:
            Vega_to_AB_fac = cbps.Vega_to_AB[inst]

        all_cat_predmags_AB = np.array(all_cat_predmags)+Vega_to_AB_fac
        all_cat_predfluxdens = 10**(-0.4*(all_cat_predmags_AB-23.9))
        
        mean_predfluxdens = np.mean(all_cat_predfluxdens)
        mean_predmag_AB = -2.5*np.log10(mean_predfluxdens)+23.9
    
        std_post_pred[std_post_pred==0] = np.inf
        
        invvar_pred = 1./std_post_pred**2
        sumweights_pred = np.nansum(invvar_pred)
        
        mean_flux_pred = np.nansum(mean_post_pred)

        stack_obj_pred.append_results(weighted_aper_flux_pred, weighted_aper_unc_pred, mean_post_pred, posx_pred, posy_pred, mean_flux_pred, nsrc_pred, \
                                     mean_predmag_AB, all_cat_predfluxdens, indiv_aper_flux, indiv_aper_flux_var)

    return stack_obj_pred


# def ciber_stack_flux_prediction_test(inst, ifield_list = [4, 5, 6, 7, 8]):

#     mmin_range_Vega = np.array([16.0, 16.5, 17.0, 17.5, 18.0])+0.1

#     if inst==2:
#         mmin_range -= 0.5

#     bandstr_dict = dict({1:'J', 2:'H'})
#     bandstr = bandstr_dict[inst]
#     lam_effs = np.array([1.05, 1.79])*1e-6*u.m # effective wavelength for bands

#     nu_effs = const.c/lam_effs

#     print('nu efs:', nu_effs)

#     all_mean_post_fluxes, all_mean_post_fluxdens, \
#         all_stack_mag_Vega_eff, all_stack_mag_AB_eff, \
#             allfield_meanpredmag_AB = [[] for x in range(5)]

    

#     # defined for specific magnitude bins used in default
#     color_corr_dict = dict({1:np.array([0.84530903, 0.84347368, 0.84613712, 0.853497, 0.85937481]), \
#                             2:np.array([1.0951691,  1.11242438, 1.14006154, 1.16856026, 1.19206676])})

#     std_color_corr_dict = dict({1:np.array([0.07766909, 0.0359452,  0.04860506, 0.03304105, 0.03791485]), \
#                             2:np.array([0.03891492, 0.04843235, 0.06048847, 0.06128753, 0.07497544])})

#     for fieldidx, ifield in enumerate(ifield_list):

#         sopred = stack_in_mag_bins_pred(inst, ifield, mmin_range=mmin_range_Vega, dx=3, trim_edge=50, mask_frac_min=0.9)

#         print('all pred mag AB is ', sopred.all_mean_predmag_AB)

#         mean_predflux = 10**(-0.4*(np.array(sopred.all_mean_predmag_AB)-23.9))

#         mean_ciber_predflux = mean_predflux*color_corr_dict[inst]
#         mean_ciber_predmag_AB = -2.5*np.log10(mean_ciber_predflux)+23.9
#         allfield_meanpredmag_AB.append(np.array(sopred.all_mean_predmag_AB))
        
#         postfig = plot_mean_posts_magbins(sopred, bandstr)
#     #     postfig.savefig('figures/stack_pred/mean_posts_in_magbins_TM'+str(inst)+'_ifield'+str(ifield)+'.png', bbox_inches='tight', dpi=300)
#         # convert fluxes from nW m-2 to W m-2 Hz-1
#         specific_flux = 1e-9*np.array(sopred.all_mean_fluxes)*cbps.pix_sr / nu_effs[inst-1].value

#         # convert to Jansky
#         specific_flux *= 1e26

#         # then to microJansky
#         specific_flux_uJy = specific_flux * 1e6

#         # color correction to UVISTA wavelength
#         specific_flux_uJy /= color_corr_dict[inst]

#         # AB magnitude
#         stack_mag_eff = -2.5*np.log10(specific_flux_uJy)+23.9

#         all_stack_mag_AB_eff.append(stack_mag_eff)
        
#     return mmin_range_Vega, allfield_meanpredmag_AB, all_stack_mag_AB_eff


def ciber_stack_flux_prediction_test(inst, ifield_list = [4, 5, 6, 7, 8],\
                                     sdwfs=False, irac_ch=None, epoch_av=True,\
                                     epochidx=None, map_mode='gradsub_perep_conserve_flux', \
                                    neff=11.2, bbox_to_anchor=[1.0, 1.3], dx=4, pixsize=7., plot=False, \
                                    mask_frac_min=0.9, trim_edge=50):


    cbps = CIBER_PS_pipeline()
    if sdwfs:
        mmin_range_Vega = np.array([13.0, 13.5, 14.0, 14.5, 15.0, 16.0])
        
        spitzer_pix_sr = (pixsize*pixsize/(3600**2))*(np.pi/180)**2
        lams = [3.6, 4.5]
        lam_um = lams[irac_ch-1]*1e-6*u.m
        nu_eff = (const.c/lam_um).value

        mag_mask = 11.0
        
    else:
        mmin_range_Vega = np.array([16.0, 16.5, 17.0, 17.5, 18.0])+0.1

        mag_mask = 15.0

        if inst==2:
            mmin_range_Vega -= 0.5

    bandstr_dict = dict({1:'J', 2:'H'})
    bandstr = bandstr_dict[inst]
    lam_effs = np.array([1.05, 1.79])*1e-6*u.m # effective wavelength for bands
    
    

    nu_effs = const.c/lam_effs

    print('nu effs:', nu_effs)

    all_mean_post_fluxes, all_mean_post_fluxdens, \
        all_stack_mag_Vega_eff, all_stack_mag_AB_eff, \
            allfield_meanpredmag_AB = [[] for x in range(5)]

    
    color_corr_dict = dict({1:np.array([0.84530903, 0.84347368, 0.84613712, 0.853497, 0.85937481]), \
                            2:np.array([1.0951691,  1.11242438, 1.14006154, 1.16856026, 1.19206676])})

    std_color_corr_dict = dict({1:np.array([0.07766909, 0.0359452,  0.04860506, 0.03304105, 0.03791485]), \
                            2:np.array([0.03891492, 0.04843235, 0.06048847, 0.06128753, 0.07497544])})


    # 0.07766909 0.0359452  0.04860506 0.03304105 0.03791485

    for fieldidx, ifield in enumerate(ifield_list):

        sopred = stack_in_mag_bins_pred(inst, ifield, mmin_range=mmin_range_Vega, dx=dx, trim_edge=trim_edge, mask_frac_min=mask_frac_min, \
                                           sdwfs=sdwfs, irac_ch=irac_ch, epoch_av=epoch_av, epochidx=epochidx, mag_mask=mag_mask, plot=plot, \
                                           map_mode=map_mode)

        
        if sdwfs:
            print('length is ', len(sopred.all_indiv_aper_flux))
            
            lams = [3.6, 4.5]
        
            plt.figure(figsize=(5, 4))
            
#             for x in range(4):
#             plt.title('IRAC CH'+str(irac_ch)+', ifield '+str(ifield), fontsize=14)
            for x in range(len(sopred.all_indiv_aper_flux)-1):
            
                pred_fluxes_plot = sopred.all_cat_pred_flux[x]

                indiv_fluxes_plot = sopred.all_indiv_aper_flux[x]

                indiv_flux_vars_plot = sopred.all_indiv_aper_flux_var[x]

                indiv_flux_snrs = indiv_fluxes_plot/np.sqrt(indiv_flux_vars_plot)
                
                print('initial length is ', len(indiv_flux_snrs))

                selmask = (indiv_flux_snrs > 3)*(~np.isinf(indiv_flux_snrs))*(~np.isnan(indiv_flux_snrs))
                
                specific_flux = 1e-9*np.array(indiv_fluxes_plot)*spitzer_pix_sr / nu_eff
                # convert to Jansky
                specific_flux *= 1e26
                # then to microJansky
                specific_flux_uJy = specific_flux * 1e6


                print('length after selection is ', len(pred_fluxes_plot[selmask]))
                plt.errorbar(pred_fluxes_plot[selmask], (specific_flux_uJy/pred_fluxes_plot)[selmask], yerr=(np.sqrt(indiv_flux_vars_plot)/pred_fluxes_plot)[selmask], \
                            color='C'+str(x), fmt='o', alpha=0.2, label=str(mmin_range_Vega[x])+'$<CH'+str(irac_ch)+'<$'+str(mmin_range_Vega[x+1]))
                        
                print(np.mean((specific_flux_uJy/pred_fluxes_plot)[selmask]))
#             plt.axhline(1./neff, color='r', linestyle='dashed', label='1/Neff')
#             plt.axhline(1./(neff**2), color='b', linestyle='dashed', label='1/Neff^2')

#             plt.yscale('log')
            plt.xscale('log')
            plt.xlabel('Predicted flux densities [$\\mu$Jy]', fontsize=14)
            plt.ylabel('Measured/SDWFS flux densities', fontsize=14)
            if irac_ch==1:
                textxpos = 110
            else:
                textxpos = 70
            textstr = 'IRAC '+str(lams[irac_ch-1])+' $\\mu$m, '+cbps.ciber_field_dict[ifield]+'\nReproject w/ conserved flux'
#             textstr = 'IRAC '+str(lams[irac_ch-1])+' $\\mu$m, '+cbps.ciber_field_dict[ifield]+'\nReproject order 1'

            plt.text(textxpos, 1.5, textstr, fontsize=14)
            plt.ylim(0, 2.0)
#             plt.ylim(0, 1.5)
    
            plt.legend(ncol=2, bbox_to_anchor=bbox_to_anchor, loc=2)

            plt.axhline(1., color='k', linestyle='dashed')

            plt.grid()
            plt.savefig('figures/spitzer_aperphot/spitzer_aperphot_IRAC_CH'+str(irac_ch)+'_ifield'+str(ifield)+'_pred_vs_measured_fluxdens_'+map_mode+'.png', bbox_inches='tight')
            plt.show()

        else:
        
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
        
    if sdwfs:
        return None
    
    return mmin_range_Vega, allfield_meanpredmag_AB, all_stack_mag_AB_eff


''' I tried a comparison with the FLAMINGOS catalog, but the photometry quality is very bad so we abandoned this. '''
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
    


def stack_in_mag_bins_sdwfs_mosaic(irac_ch, ifield, mmin_range, dx=4, bkg_rad=0.4, mask_frac_min=0.9, \
                                  epoch_av=True, epochidx=None, plot=True, catalog_basepath=None, \
                                  sdwfs_basepath=None, version=3):
    
    Vega_to_AB_irac = dict({1:2.79, 2:3.26})
    
    if catalog_basepath is None:
        catalog_basepath = config.ciber_basepath+'data/catalogs/'

    catalog_fpath_pred = catalog_basepath + '/sdwfs/SDWFS_ch'+str(irac_ch)+'_ifield'+str(ifield)+'_with_mosaic_xy.csv'
    catalog_df = pd.read_csv(catalog_fpath_pred)
    
    dms = list(mmin_range[1:]-mmin_range[:-1])
    dms.append(dms[-1])
    dms = np.array(dms)
    
    stack_obj_pred = stack_obj(mmin_range=mmin_range, dms=dms)
    mag_mask = np.min(mmin_range)
    
    Vega_to_AB_fac = Vega_to_AB_irac[irac_ch]
    
    print('looping through magnitudes..')
    for m, m_min in enumerate(mmin_range):
        m_max = m_min + dms[m]
        
        cal_src_posx, cal_src_posy,\
            all_cat_predmags, all_aper_flux, all_aper_var = calc_aperture_fluxes_sdwfs_mosaic(catalog_df, irac_ch, ifield,\
                                                                                              m_min, m_max, dx=dx, bkg_rad=bkg_rad,\
                                                                                              mask_frac_min=mask_frac_min, epochidx=epochidx, \
                                                                                             plot=True, sdwfs_basepath=sdwfs_basepath, \
                                                                                             version=version)
        
        all_cat_predmags_AB = np.array(all_cat_predmags)+Vega_to_AB_fac
        all_cat_predfluxdens = 10**(-0.4*(all_cat_predmags_AB-23.9))
        
        stack_obj_pred.append_results(posx=cal_src_posx, posy=cal_src_posy, \
                                     cat_pred_flux=all_cat_predfluxdens, indiv_aper_flux=all_aper_flux, indiv_aper_flux_var=all_aper_var)

    return stack_obj_pred


def sdwfs_mosaic_flux_prediction_test(irac_ch, dx=6, epochidx=0, ifield_list=[6, 7], pixsize=0.86, sdwfs_basepath=None, \
                                     version=3):
    
    spitzer_pix_sr = (pixsize*pixsize/(3600**2))*(np.pi/180)**2
    mmin_range_Vega = np.array([13.0, 13.5, 14.0, 14.5, 15.0, 16.0])
    lams = [3.6, 4.5]
    
    lam_um = lams[irac_ch-1]*1e-6*u.m
    nu_eff = (const.c/lam_um).value

    all_mean_post_fluxes, all_mean_post_fluxdens, \
        all_stack_mag_Vega_eff, all_stack_mag_AB_eff, \
            allfield_meanpredmag_AB = [[] for x in range(5)]
    
    for fieldidx, ifield in enumerate(ifield_list):

        sopred = stack_in_mag_bins_sdwfs_mosaic(irac_ch, ifield, mmin_range=mmin_range_Vega, dx=dx, mask_frac_min=None, \
                                   epochidx=epochidx, plot=False, sdwfs_basepath=sdwfs_basepath, version=version)
        
        print('length is ', len(sopred.all_indiv_aper_flux))
        
            
        plt.figure(figsize=(5, 4))
        for x in range(len(sopred.all_indiv_aper_flux)-1):

            pred_fluxes_plot = sopred.all_cat_pred_flux[x]
            
            indiv_fluxes_plot = sopred.all_indiv_aper_flux[x]
            
            specific_flux = 1e-9*np.array(indiv_fluxes_plot)*spitzer_pix_sr / nu_eff
            # convert to Jansky
            specific_flux *= 1e26
            # then to microJansky
            specific_flux_uJy = specific_flux * 1e6
            
            specific_flux_uJy *= 11.2
            
            indiv_flux_vars_plot = sopred.all_indiv_aper_flux_var[x]
            indiv_flux_snrs = indiv_fluxes_plot/np.sqrt(indiv_flux_vars_plot)
            selmask = (indiv_flux_snrs > 3)*(~np.isinf(indiv_flux_snrs))*(~np.isnan(indiv_flux_snrs))
            print('length after selection is ', len(pred_fluxes_plot[selmask]))
            
            ratio = specific_flux_uJy/indiv_fluxes_plot
            
            plt.errorbar(pred_fluxes_plot[selmask], (specific_flux_uJy/pred_fluxes_plot)[selmask], yerr=(ratio*np.sqrt(indiv_flux_vars_plot)/pred_fluxes_plot)[selmask], \
                        color='C'+str(x), fmt='o', alpha=0.2, label=str(mmin_range_Vega[x])+'$<CH'+str(irac_ch)+'<$'+str(mmin_range_Vega[x+1]))

        plt.xscale('log')
        plt.xlabel('Predicted flux densities [$\\mu$Jy]', fontsize=14)
        plt.ylabel('Measured/SDWFS flux densities', fontsize=14)
        if irac_ch==1:
            textxpos = 110
        else:
            textxpos = 70
        textstr = 'IRAC '+str(lams[irac_ch-1])+' $\\mu$m, '+cbps.ciber_field_dict[ifield]+'\nInitial SDWFS mosaic'
        plt.text(textxpos, 1.5, textstr, fontsize=14)
        plt.ylim(0, 2.0)

        plt.legend(ncol=2, bbox_to_anchor=[0.0, 1.3], loc=2)
        plt.axhline(1., color='k', linestyle='dashed')
        plt.grid()
        plt.show()


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


