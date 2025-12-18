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
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

def linear_func_nointercept(x, m):
    return m*x

def linear_func(x, m, b):
    return m*x + b

def get_vega_zp_linfit():
    
    lambda2mass = [1.235, 1.662, 2.159]
    vega2mass = [1594, 1024, 666.7]

    a, b = np.polyfit(lambda2mass, vega2mass, 1)
    
    return a, b

def extrap_vega_zp(lam):
    a, b = get_vega_zp_linfit()
    return a*lam + b

def calc_stacked_fluxes(cbps, inst, fieldidx_choose, mask_base_path, catalog_fpath, m_min, m_max, cat_type='predict', \
                       dx=5, trim_edge=50, mag_mask=15.0, bkg_rad=0.35, xlim=None, ylim=None, \
                       mask_tail = 'maglim_J_Vega_17.5_111323_ukdebias'):

    
    # process ciber maps first
    ifield_list = [4, 5, 6, 7, 8]
    
    data_type='observed'
    config_dict, pscb_dict, float_param_dict, fpath_dict = return_default_cbps_dicts()
    ciber_mock_fpath = config.exthdpath+'ciber_mocks/'
    fpath_dict, list_of_dirpaths, base_path, trilegal_base_path = set_up_filepaths_cbps(fpath_dict, inst, 'test', '112022',\
                                                                                    datestr_trilegal='112022', data_type=data_type, \
                                                                                   save_fpaths=True)

    
    ifield_choose = ifield_list[fieldidx_choose]
    
    ciber_maps, masks = [np.zeros((len(ifield_list), 1024, 1024)) for x in range(2)]

    dc_template = cbps.load_dark_current_template(inst, verbose=True, inplace=False)
    
    for fieldidx, ifield in enumerate(ifield_list):
        flight_im = cbps.load_flight_image(ifield, inst, inplace=False)
        flight_im *= cbps.cal_facs[inst]
        flight_im -= dc_template*cbps.cal_facs[inst]
        mask_fpath = fpath_dict['mask_base_path']+'/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'
        mask = fits.open(mask_fpath)[1].data    
        sigclip_mask = iter_sigma_clip_mask(flight_im, sig=5, nitermax=5, mask=mask.astype(int))
        mask *= sigclip_mask
        ciber_maps[fieldidx] = flight_im
        masks[fieldidx] = mask
        
    processed_ciber_maps, ff_estimates,\
            final_planes, stack_masks, ff_weights = process_ciber_maps(cbps, ifield_list, inst, ciber_maps, masks,\
                                                                 clip_sigma=5, nitermax=5, \
                                                                    niter=5, ff_stack_min=1)
    
    
    flight_im = processed_ciber_maps[fieldidx_choose]
    bkg_mask = gen_bkg_mask(dx, bkg_rad=bkg_rad)
    
    # load mask
#     mask_fpath = mask_base_path+'/maskInst_aducorr/field'+str(ifield)+'_TM'+str(inst)+'_maskInst_aducorr.fits'
#     mask = fits.open(mask_fpath)[1].data
    
    mask_fpath = mask_base_path+'maglim_J_Vega_'+str(mag_mask)+'_111323_ukdebias/joint_mask_ifield'+str(ifield_choose)+'_inst'+str(inst)+'_observed_maglim_J_Vega_'+str(mag_mask)+'_111323_ukdebias.fits'
    mask = fits.open(mask_fpath)[1].data
    
    plot_map(flight_im*mask)

    
    # load catalog
    cat_df = pd.read_csv(catalog_fpath)

    cat_x = np.array(cat_df['x'+str(inst)])
    cat_y = np.array(cat_df['y'+str(inst)])
    if cat_type=='predict':
        cat_mag = np.array(cat_df['J_Vega_predict'])
    elif cat_type=='flamingos':
        cat_mag = np.array(cat_df['J'])
                
    magmask = (cat_mag > m_min)*(cat_mag < m_max)*(cat_x > trim_edge)*(cat_x < 1023-trim_edge)*(cat_y > trim_edge)*(cat_y < 1023-trim_edge)
    
    if xlim is not None:
        xmask = (cat_x > xlim[0])*(cat_x < xlim[1])
        magmask *= xmask
        
    if ylim is not None:
        ymask = (cat_y > ylim[0])*(cat_y < ylim[1])
        magmask *= ymask
    
    cal_src_posx = cat_x[magmask]
    cal_src_posy = cat_y[magmask]
    
    all_postage_stamps, all_postage_stamps_mask, post_bool = grab_postage_stamps(inst, cal_src_posx, cal_src_posy, flight_im, mask, dx, mask_frac_min=0.95)        
        
    sum_post = np.zeros_like(all_postage_stamps[0])
    
    sum_counts = np.zeros_like(sum_post)
    
    all_aper_flux, all_aper_var, all_aper_post = [[] for x in range(3)]

    for n in range(len(all_postage_stamps)):
        if post_bool[n]==1:
            
            aper_flux, aper_flux_var, bkg_level, bkg_var, aper_post = aper_phot(all_postage_stamps[n], all_postage_stamps_mask[n], bkg_mask, mode='mean', plot=False)
            
            all_aper_post.append(aper_post)
            all_aper_flux.append(aper_flux)
            all_aper_var.append(aper_flux_var)
            
            sum_post += aper_post
#             sum_post += all_postage_stamps[n]*all_postage_stamps_mask[n]
            sum_counts += all_postage_stamps_mask[n]
                
    all_aper_post = np.array(all_aper_post)
    
    mean_post = sum_post / sum_counts
    
    std_post = np.nanstd(all_aper_post, axis=0)/np.sqrt(sum_counts)
    
    aper_flux_weights = 1./np.array(all_aper_var)
    
    weighted_aper_flux = np.nansum(np.array(all_aper_flux)*aper_flux_weights)/np.nansum(aper_flux_weights)
    weighted_aper_unc = np.sqrt(1./np.nansum(aper_flux_weights))
        
    plot_map(mean_post, figsize=(6,6), title='mean postage')
    plot_map(std_post, figsize=(6,6), title='std_post')
        
    return mean_post, sum_post, std_post, sum_counts, np.sum(post_bool),\
            cal_src_posx, cal_src_posy, weighted_aper_flux, weighted_aper_unc

def compare_with_zl_predictions_slope(inst, ifield_list, mask_base_path, mask_tail, dx=6, nregion=5, dimx=1024, average_g1g2_over_fields=False, \
                                     zl_vals=None, iter_clip=False, nsig=5, niter=2, with_jack=False, savefig=False, include_ffcorr=False):
    
    cbps = CIBER_PS_pipeline()
    xspace = np.linspace(0, dimx, nregion+1).astype(int)
    yspace = np.linspace(0, dimx, nregion+1).astype(int)
    
    all_flight_im, all_masks = [np.zeros((len(ifield_list), 1024, 1024)) for x in range(2)]
    
    ciber_mean_adu_coarse, g1g2_vals_coarse, g1g2_unc_vals_coarse, \
        zl_vals_coarse, nsrc_perbin_coarse, mean_ff_coarse = [np.zeros((len(ifield_list), nregion, nregion)) for x in range(6)]
    
    comp_slope_coarse, comp_intercept_coarse = [np.zeros((nregion, nregion)) for x in range(2)]
    g1g2_coarse_fieldav, comp_slope_unc_coarse, comp_intercept_unc_coarse = [np.zeros((nregion, nregion)) for x in range(3)]
    
    dc_map = cbps.load_dark_current_template(inst, inplace=False)
    calib_result_basepath = config.ciber_basepath+'data/calib/TM'+str(inst)+'/'
    
    for fieldidx, ifield in enumerate(ifield_list):
        
        # load flight images and masks, calculate mean photocurrent in sub-regions
        
        if ifield==5:
            flight_im = fits.open(config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/slope_fits/ciber_flight_TM'+str(inst)+'_ifield'+str(ifield)+'_aducorr.fits')[1].data
        else:
            flight_im = cbps.load_flight_image(ifield, inst, inplace=False)
        flight_im -= dc_map
            
        all_flight_im[fieldidx] = flight_im
            
        mask_fpath = mask_base_path+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'
        mask = fits.open(mask_fpath)[1].data
        
        all_masks[fieldidx] = mask
        
        if ifield==5:
            plotreg = False
        else:
            plotreg=False
            
            
        if include_ffcorr:
            ff_est = fits.open(config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/ff_est/ff_ests_TM'+str(inst)+'_'+mask_tail+'.fits')['ifield'+str(ifield)].data
            
            mean_ff_coarse[fieldidx] = calc_mean_adu_coarse(ff_est, mask, nregion, xspace, yspace)
            
#             ffcoarse = plot_map(mean_ff_coarse[fieldidx], xlabel=None, ylabel=None, title='Binned FF estimate, ifield '+str(ifield), figsize=(5,5), \
#                     return_fig=True)
#             ffcoarse.savefig(config.ciber_basepath+'figures/calibration_results/ffest_byregion_TM'+str(inst)+'_ifield'+str(ifield)+'.png', bbox_inches='tight')

        ciber_mean_adu_coarse[fieldidx] = calc_mean_adu_coarse(flight_im, mask, nregion, xspace, yspace, plot=plotreg)
        plot_map(ciber_mean_adu_coarse[fieldidx], figsize=(5,5), title='mean adu coarse ifield '+str(ifield))
        # load pos and pred/measured fluxes
        
        calib_fpath = calib_result_basepath+'ciber_sbcal_results_TM'+str(inst)+'_ifield'+str(ifield)+'_dx='+str(dx)+'.npz'
        calresult = np.load(calib_fpath)
        cal_src_posx = calresult['cal_src_posx']
        cal_src_posy = calresult['cal_src_posy']
        measured_fluxes = calresult['all_measured_fluxes']
        pred_fluxes = calresult['all_pred_fluxes']
        var_measured_fluxes = calresult['all_var_measured_fluxes']
        
        g1g2_coarsebin, g1g2_unc_coarsebin, nsrc_perbin = calc_g1g2_coarsebin(cal_src_posx, cal_src_posy, pred_fluxes, measured_fluxes, var_measured_fluxes, nregion, \
                                                                             iter_clip=iter_clip, nsig=nsig, niter=niter, with_jack=False)
        g1g2_vals_coarse[fieldidx] = g1g2_coarsebin
        g1g2_unc_vals_coarse[fieldidx] = g1g2_unc_coarsebin
        nsrc_perbin_coarse[fieldidx] = nsrc_perbin
        
#         nsrcfig = plot_map(nsrc_perbin_coarse[fieldidx], vmax=90, vmin=40, cmap='bwr', xlabel=None, ylabel=None, title='Number of calibration sources\n(TM'+str(inst)+', ifield '+str(ifield)+')', figsize=(5,5), return_fig=True)
#         nsrcfig.savefig(config.ciber_basepath+'figures/calibration_results/nsrc_byregion_TM'+str(inst)+'_ifield'+str(ifield)+'.png', bbox_inches='tight')
        
        if inst==1:
            vmin, vmax = -300, -250
        else:
            vmin, vmax = -135, -100
#         g1g2fig = plot_map(g1g2_coarsebin, vmin=vmin, vmax=vmax, xlabel=None, ylabel=None, title='G1G2 by region (TM'+str(inst)+', ifield '+str(ifield)+')\n[nW m$^{-2}$ sr$^{-1}$/ADU fr$^{-1}$]', figsize=(5,5), return_fig=True)
#         if savefig:
#             g1g2fig.savefig(config.ciber_basepath+'figures/calibration_results/g1g2_byregion_TM'+str(inst)+'_ifield'+str(ifield)+'.png', bbox_inches='tight')

#         ciber_mean_sb_coarse = ciber_mean_adu_coarse[fieldidx]*g1g2_coarsebin
#         sbfig = plot_map(ciber_mean_sb_coarse, xlabel=None, ylabel=None, title='CIBER sky SB (TM'+str(inst)+', ifield '+str(ifield)+')\n[nW m$^{-2}$ sr$^{-1}$]', figsize=(5,5), return_fig=True)
#         sbfig.savefig(config.ciber_basepath+'figures/calibration_results/ciber_sb_byregion_TM'+str(inst)+'_ifield'+str(ifield)+'.png', bbox_inches='tight')

        if zl_vals is not None:
            corr_fac = [0.97, 1.0]
            zl_vals_coarse[fieldidx] = zl_vals[fieldidx]
#             zl_vals_coarse[fieldidx] *= corr_fac[inst-1]

#     ffcoarse = plot_map(np.mean(mean_ff_coarse, axis=0), xlabel=None, ylabel=None, title='Binned FF estimate, TM'+str(inst), figsize=(5,5), \
#             return_fig=True)
#     ffcoarse.savefig(config.ciber_basepath+'figures/calibration_results/ffest_byregion_TM'+str(inst)+'_av.png', bbox_inches='tight')

    if nregion==3:
        fig, ax = plt.subplots(nrows=nregion, ncols=nregion, figsize=(8,6))
    elif nregion==2:
        fig, ax = plt.subplots(nrows=nregion, ncols=nregion, figsize=(6,5))
    elif nregion==4:
        fig, ax = plt.subplots(nrows=nregion, ncols=nregion, figsize=(9,7))
    elif nregion==1:
        fig, ax = plt.subplots(nrows=nregion, ncols=nregion, figsize=(5,4))
    
    all_ciber_pred_sb_coarse = np.ones_like(ciber_mean_adu_coarse)
    i=1
    for nx in range(nregion):
        for ny in range(nregion):
            
            plt.subplot(nregion, nregion, i)
            i+=1

            if average_g1g2_over_fields:

                which_use = (g1g2_vals_coarse[:,nx,ny] != 0)
                var_use = g1g2_unc_vals_coarse[which_use,nx,ny]**2
                
                print('var use:', var_use)

                weights_use = 1./var_use
                g1g2_use = g1g2_vals_coarse[which_use,nx,ny]
                g1g2_apply = np.sum(g1g2_use*weights_use)/np.sum(weights_use)
                
                if include_ffcorr:
                    g1g2_apply /= np.mean(mean_ff_coarse[:,nx,ny])

                g1g2_unc_apply = np.sqrt(1./np.sum(weights_use))
                g1g2_coarse_fieldav[nx,ny] = g1g2_apply

            else:
                
                g1g2_apply = g1g2_vals_coarse[:,nx,ny]
                if include_ffcorr:
                    g1g2_apply /= mean_ff_coarse[:,nx,ny]
                    
                g1g2_unc_apply = g1g2_unc_vals_coarse[:,nx,ny]
                g1g2_coarse_fieldav[nx,ny] = np.mean(g1g2_apply)
            

            ciber_pred_zl_sb = ciber_mean_adu_coarse[:,nx,ny]*g1g2_apply
            ciber_pred_zl_sb_unc = np.abs(ciber_mean_adu_coarse[:,nx,ny]*g1g2_unc_apply)

            all_ciber_pred_sb_coarse[:, nx,ny] = ciber_pred_zl_sb
            # fit line to pred zl vs model zl
            
            popt, pcov = curve_fit(linear_func, zl_vals_coarse[:,nx,ny], ciber_pred_zl_sb, sigma=ciber_pred_zl_sb_unc, absolute_sigma=True)

            if with_jack:
                nf = len(ciber_pred_zl_sb)
                all_slopes, all_intercepts = [], []
                for x in np.arange(nf):
                    which = (np.arange(nf) != x)
                    popt_jack, pcov_jack = curve_fit(linear_func, zl_vals_coarse[which,nx,ny], ciber_pred_zl_sb[which], sigma=ciber_pred_zl_sb_unc[which], absolute_sigma=True)
                    all_slopes.append(popt_jack[0])
                    all_intercepts.append(popt_jack[1])
                    
                comp_slope_coarse[nx,ny] = np.mean(all_slopes)
                comp_intercept_coarse[nx,ny] = np.mean(all_intercepts)
                comp_slope_unc_coarse[nx,ny] = np.sqrt((np.std(all_slopes)/np.sqrt(len(ciber_pred_zl_sb)-1))**2 + pcov[0,0])
                comp_intercept_unc_coarse[nx,ny] = np.std(all_intercepts)/np.sqrt(len(ciber_pred_zl_sb)-1)
                
            else:
                comp_slope_coarse[nx,ny] = popt[0]
                comp_intercept_coarse[nx,ny] = popt[1]
                
                comp_slope_unc_coarse[nx,ny] = np.sqrt(pcov[0,0])
                comp_intercept_unc_coarse[nx,ny] = np.sqrt(pcov[1,1])
            
            if inst==1:
                sbmax = 1200
            elif inst==2:
                sbmax = 800
                
            if nregion > 1:
                ax[nregion-nx-1,ny].errorbar(zl_vals_coarse[:,nx,ny], ciber_pred_zl_sb, yerr=ciber_pred_zl_sb_unc, markersize=5, fmt='o', color='r')
    #             ax[nregion-nx-1,ny].plot(np.linspace(0, sbmax, 100), popt[0]*np.linspace(0, sbmax, 100)+popt[1], color='k', linestyle='dashed')
                ax[nregion-nx-1,ny].plot(np.linspace(0, sbmax, 100), comp_slope_coarse[nx,ny]*np.linspace(0, sbmax, 100)+comp_intercept_coarse[nx,ny], color='k', linestyle='dashed')
            else:
                plt.errorbar(zl_vals_coarse[:,nx,ny], ciber_pred_zl_sb, yerr=ciber_pred_zl_sb_unc, markersize=5, fmt='o', color='r')
                plt.plot(np.linspace(0, sbmax, 100), comp_slope_coarse[nx,ny]*np.linspace(0, sbmax, 100)+comp_intercept_coarse[nx,ny], color='k', linestyle='dashed')
                
            chisq = (ciber_pred_zl_sb-(comp_slope_coarse[nx,ny]*zl_vals_coarse[:,nx,ny]+comp_intercept_coarse[nx,ny]))**2/ciber_pred_zl_sb_unc**2
        
            chisq = np.sum(chisq)
            
            if nx==nregion-1:
                plt.xlabel('Kelsall ZL\n[nW m$^{-2}$ sr$^{-1}$]', fontsize=14)
            if ny==0:
                plt.ylabel('CIBER sky SB\n[nW m$^{-2}$ sr$^{-1}$]', fontsize=14)

            fitstr = 'Slope='+str(np.round(comp_slope_coarse[nx,ny], 2))+'$\\pm$'+str(np.round(comp_slope_unc_coarse[nx,ny], 2))
            fitstr += '\nIntercept='+str(int(comp_intercept_coarse[nx,ny]))+'$\\pm$'+str(int(comp_intercept_unc_coarse[nx,ny]))
#             fitstr += '\n$\\chi^2=$'+str(np.round(chisq, 2))
#             fitstr = 'Slope='+str(np.round(popt[0], 2))+'$\\pm$'+str(np.round(np.sqrt(pcov[0,0]), 2))
#             fitstr += '\nIntercept='+str(int(popt[1]))+'$\\pm$'+str(int(np.sqrt(pcov[1,1])))
            if nregion==3:
                fs = 10
            elif nregion==4:
                fs = 9
                
            if nregion > 1:
                ax[nregion-nx-1,ny].text(0.25*sbmax, 0.15*sbmax, fitstr, fontsize=10)

                ax[nregion-nx-1,ny].set_xlim(0, sbmax)
                ax[nregion-nx-1,ny].set_ylim(0, sbmax)
                ax[nregion-nx-1,ny].grid(alpha=0.5)
            else:
                plt.text(0.25*sbmax, 0.15*sbmax, fitstr, fontsize=14)
                plt.xlim(0, sbmax)
                plt.ylim(0, sbmax)
                plt.grid(alpha=0.5)
                plt.tick_params(labelsize=14)
                
    plt.tight_layout()
    if savefig:
        plt.savefig(config.ciber_basepath+'figures/calibration_results/ciber_kelsall_meansb_compare_slopes_perregion_TM'+str(inst)+'_indivg1g2_nregion='+str(nregion)+'_sigclip_nojack.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    g1g2fig = plot_map(g1g2_coarse_fieldav, vmin=vmin, vmax=vmax, xlabel=None, ylabel=None, title='G1G2 (TM'+str(inst)+' field average)\n[nW m$^{-2}$ sr$^{-1}$/ADU fr$^{-1}$]', figsize=(5,5), return_fig=True)

    for fieldidx, ifield in enumerate(ifield_list):
        
        fig_meansb = plot_map(all_ciber_pred_sb_coarse[fieldidx], xlabel=None, ylabel=None, title='Binned Mean SB\nField '+str(ifield)+' [nW m$^{-2}$ sr$^{-1}$]', figsize=(5,5), return_fig=True)
        plot_map(comp_intercept_coarse, title='Intercept', figsize=(5,5))
        fig_meansb_inter = plot_map(all_ciber_pred_sb_coarse[fieldidx]-comp_intercept_coarse, xlabel=None, ylabel=None, title='Mean SB - Intercept\nField '+str(ifield)+' [nW m$^{-2}$ sr$^{-1}$]', figsize=(5,5), return_fig=True)
        fig_meansb_inter_slope = plot_map((all_ciber_pred_sb_coarse[fieldidx]-comp_intercept_coarse)/comp_slope_coarse, xlabel=None, ylabel=None, title='(Mean SB - intercept)/Slope\nField '+str(ifield)+' [nW m$^{-2}$ sr$^{-1}$]', figsize=(5,5), return_fig=True)

#         fig_meansb.savefig(config.ciber_basepath+'figures/calibration_results/meansb/meansb_perregion_TM'+str(inst)+'_ifield'+str(ifield)+'.png', bbox_inches='tight')
#         fig_meansb_inter.savefig(config.ciber_basepath+'figures/calibration_results/meansb_intercept/meansb_intercept_perregion_TM'+str(inst)+'_ifield'+str(ifield)+'.png', bbox_inches='tight')
#         fig_meansb_inter_slope.savefig(config.ciber_basepath+'figures/calibration_results/meansb_intercept_slope/meansb_intercept_slope_perregion_TM'+str(inst)+'_ifield'+str(ifield)+'.png', bbox_inches='tight')
        
        
#         masked_im = all_flight_im[fieldidx]*all_masks[fieldidx]
#         masked_im[all_masks[fieldidx]==1] -= np.mean(masked_im[all_masks[fieldidx]==1])
        
#         plot_map(all_flight_im[fieldidx]*all_masks[fieldidx], title='masked image, field '+str(ifield))
#         plot_map(masked_im, title='mean subtracted, field '+str(ifield))


    slopef = plot_map(comp_slope_coarse, xlabel=None,ylabel=None, title='CIBER/Kelsall slopes, TM'+str(inst), figsize=(5,5), return_fig=True)
    interf = plot_map(comp_intercept_coarse, xlabel=None,ylabel=None, title='Intercepts, TM'+str(inst), figsize=(5,5), return_fig=True)
    
#     if savefig:
#         slopef.savefig(config.ciber_basepath+'figures/calibration_results/slope_perregion_ciber_kelsall_TM'+str(inst)+'_nojack_withelat30.png', bbox_inches='tight')
#         interf.savefig(config.ciber_basepath+'figures/calibration_results/intercept_perregion_ciber_kelsall_TM'+str(inst)+'_nojack_withelat30.png', bbox_inches='tight')

    return comp_slope_coarse, comp_intercept_coarse


def calc_g1g2_slope_iterclip(pred_fluxes, measured_fluxes, var_measured_fluxes, nsig=5, niter=2, with_intercept=False, plot=False):
    
#     print('NSIG = ', nsig)
    pred_fluxes_clip = pred_fluxes.copy()
    measured_fluxes_clip = measured_fluxes.copy()
    var_measured_fluxes_clip = var_measured_fluxes.copy()
    
    for n in range(niter):
        if with_intercept:
            popt, pcov = curve_fit(linear_func, pred_fluxes_clip, measured_fluxes_clip, sigma=np.sqrt(var_measured_fluxes_clip), absolute_sigma=True)
        else:
            popt, pcov = curve_fit(linear_func_nointercept, pred_fluxes_clip, measured_fluxes_clip, sigma=np.sqrt(var_measured_fluxes_clip), absolute_sigma=True)
        
        if n==0:
            popt_init = popt
            g1g2_init = 1./popt[0]
            
        meanpred_measured_fluxes = popt[0]*pred_fluxes_clip
        if with_intercept:
            meanpred_measured_fluxes += popt[1]
        
        dpred_measured_fluxes = measured_fluxes_clip - meanpred_measured_fluxes
        zscore = dpred_measured_fluxes/np.sqrt(var_measured_fluxes_clip)
        which_keep = (np.abs(zscore) < nsig)
        
        pred_fluxes_clip = pred_fluxes_clip[which_keep]
        measured_fluxes_clip = measured_fluxes_clip[which_keep]
        var_measured_fluxes_clip = var_measured_fluxes_clip[which_keep]
      
    # calculate g1g2 from final clipped catalogs
    popt, pcov = curve_fit(linear_func, pred_fluxes_clip, measured_fluxes_clip, sigma=np.sqrt(var_measured_fluxes_clip), absolute_sigma=True)
    
    g1g2 = 1./popt[0]
    g1g2_unc = g1g2*(np.sqrt(pcov[0,0])/popt[0])
    
    fineflux = np.linspace(0.5*np.min(pred_fluxes), 1.2*np.max(pred_fluxes), 100)
    
    if plot:
        plt.figure(figsize=(6,5))
        plt.errorbar(pred_fluxes, -1*measured_fluxes, yerr=np.sqrt(var_measured_fluxes), color='k', markersize=5, fmt='o', label='Before clip')
        plt.errorbar(pred_fluxes_clip, -1*measured_fluxes_clip, yerr=np.sqrt(var_measured_fluxes_clip), markersize=5, color='r', fmt='o', label='After clip')

        init_adupred = popt_init[0]*fineflux
        final_adupred = popt[0]*fineflux

        if with_intercept:
            init_adupred += popt_init[1]
            final_adupred += popt[1]

        plt.plot(fineflux, -1*(init_adupred), label='Before clip', linestyle='dashed', color='k', zorder=10)
        plt.plot(fineflux, -1*(final_adupred), label='After clip', linestyle='dashed', color='r', zorder=10)

        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('Predicted flux [nW m$^{-2}$]', fontsize=14)
        plt.ylabel('Measured flux [-1$\\times$ADU fr$^{-1}$]', fontsize=14)
        plt.grid(alpha=0.5)
        plt.ylim(5, 1e3)
        plt.xlim(0.9*np.min(pred_fluxes), 1.1*np.max(pred_fluxes))
        plt.text(0.1*np.max(pred_fluxes), 20, '$g_ag_1g_2=$'+str(np.round(g1g2_init, 1)), color='k', fontsize=16)
        plt.text(0.1*np.max(pred_fluxes), 10, '$g_ag_1g_2=$'+str(np.round(g1g2, 1)), color='r' ,fontsize=16)
        plt.show()
    
#     print('initial/final number of sources:', len(pred_fluxes), len(pred_fluxes_clip))
#     print('Initial/final g1g2:', g1g2_init, g1g2)
    
    return g1g2, g1g2_unc, pred_fluxes_clip, measured_fluxes_clip, var_measured_fluxes_clip

def calc_g1g2_slope(pred_fluxes, measured_fluxes, var_measured_fluxes):
    
    popt, pcov = curve_fit(linear_func, pred_fluxes, measured_fluxes, sigma=np.sqrt(var_measured_fluxes), absolute_sigma=True)
    
    g1g2 = 1./popt[0]
    
    g1g2_unc = g1g2*(np.sqrt(pcov[0,0])/popt[0])
        
    return g1g2, g1g2_unc

def calc_g1g2_coarsebin(cal_src_posx, cal_src_posy, pred_fluxes, measured_fluxes, var_measured_fluxes, nregion=5, dimx=1024, \
                       iter_clip=False, nsig=5, niter=2, with_jack=False):
    
    xspace = np.linspace(0, dimx, nregion+1).astype(int)
    yspace = np.linspace(0, dimx, nregion+1).astype(int)
    
    nsrc_perbin, g1g2_coarsebin, g1g2_unc_coarsebin = [np.zeros((nregion, nregion)) for x in range(3)]
    
    for nx in range(nregion):
        for ny in range(nregion):
            
            inregion = (cal_src_posx > xspace[nx])*(cal_src_posx < xspace[nx+1])*(cal_src_posy > yspace[ny])*(cal_src_posy < yspace[ny+1])
                        
            nsrc_perbin[nx, ny] = np.sum(inregion)
            
            posx_sel = cal_src_posx[inregion]
            posy_sel = cal_src_posy[inregion]
            pred_fluxes_sel = pred_fluxes[inregion]
            measured_fluxes_sel = measured_fluxes[inregion]
            var_measured_fluxes_sel = var_measured_fluxes[inregion]
            
            if nsrc_perbin[nx,ny] < 2:
                continue
                
            if iter_clip:
                g1g2, g1g2_unc, pred_fluxes_sel, measured_fluxes_sel, var_measured_fluxes_sel = calc_g1g2_slope_iterclip(pred_fluxes_sel, measured_fluxes_sel, var_measured_fluxes_sel, nsig=nsig, niter=niter)
            else:
                g1g2, g1g2_unc = calc_g1g2_slope(pred_fluxes_sel, measured_fluxes_sel, var_measured_fluxes_sel)
            
            if with_jack:
                all_g1g2 = jackknife_g1g2_slope(pred_fluxes_sel, measured_fluxes_sel, var_measured_fluxes_sel, with_intercept=False, plot=True)
            
            g1g2_coarsebin[nx, ny] = g1g2
            g1g2_unc_coarsebin[nx,ny] = g1g2_unc
            
    return g1g2_coarsebin, g1g2_unc_coarsebin, nsrc_perbin

def calc_mean_adu_coarse(flight_im, mask, nregion, xspace, yspace, plot=False, mode='mean'):
    
    mean_adu_coarse = np.zeros((nregion, nregion))
    
    for nx in range(nregion):
        for ny in range(nregion):
            
            flight_reg = flight_im[xspace[nx]:xspace[nx+1], yspace[ny]:yspace[ny+1]]
            mask_reg = mask[xspace[nx]:xspace[nx+1], yspace[ny]:yspace[ny+1]]
            
            if plot:
                plot_map(flight_reg*mask_reg, figsize=(6,6), title='flight*mask region')
            
            if mode=='mean':
                mean_adu_coarse[nx,ny] = np.mean(flight_reg[mask_reg==1])
            elif mode=='median':
                mean_adu_coarse[nx,ny] = np.median(flight_reg[mask_reg==1])
            
    return mean_adu_coarse

def load_calib_catalog(inst, ifield, m_min=11, m_max=14.5, spline_k=3, trim_edge=50, interp_mode='flux', plot=False, \
                        cross_inst=None, m_min_cross=None, m_max_cross=None):
    

    calibration_cat = np.load(config.ciber_basepath+'data/catalogs/calibration_catalogs/calibration_src_catalog_ifield'+str(ifield)+'.npz')['calibration_cat']
    
    magdict = dict({1:6, 2:7})
    Vega_to_AB = dict({1:0.91, 2:1.39})

    ciber_lam = [1.05, 1.79]
    if inst==1:
        # magidx = 6
        xidx, yidx = 2, 3
        # Vega_to_AB = 0.91
    elif inst==2:
        # magidx = 7
        xidx, yidx = 4, 5
        # Vega_to_AB = 1.39

    magidx = magdict[inst]    
    calib_mag_ref = calibration_cat[:,magidx]-Vega_to_AB[inst]

    if cross_inst is not None:
        magidx_cross = magdict[cross_inst]
        calib_mag_cross_ref = calibration_cat[:,magidx_cross]-Vega_to_AB[cross_inst]
        
    cal_posmask = (calibration_cat[:,xidx] > trim_edge)*(calibration_cat[:,xidx] < 1024-trim_edge)*(calibration_cat[:,yidx] > trim_edge)*(calibration_cat[:,yidx] < 1024-trim_edge)
    cal_magmask = (calib_mag_ref > m_min)*(calib_mag_ref < m_max)

    if cross_inst is not None:
        cal_magmask *= (calib_mag_cross_ref > m_min_cross)*(calib_mag_cross_ref < m_max_cross)

    which_sel = np.where(cal_posmask*cal_magmask)[0]
    
    calmags = calib_mag_ref[which_sel]

    if cross_inst is not None:
        calmags_cross = calib_mag_cross_ref[which_sel]
    else:
        calmags_cross = None

    calibration_cat_sel = calibration_cat[which_sel, :]
    cal_x = calibration_cat_sel[:,xidx]
    cal_y = calibration_cat_sel[:,yidx]
    
    if plot:
        plt.figure()
        plt.hist(calibration_cat_sel[:,magidx], bins=np.linspace(10, 15, 20))
        plt.yscale('log')
        plt.show()
        
        plt.figure()
        plt.scatter(cal_x, cal_y)
        plt.show()
    
#     print('cal sel has shape ', calibration_cat_sel.shape)
    
    ciber_fluxes, ciber_twom_flux_ratio = [np.zeros((len(cal_x), 2)) for x in range(2)]
    nbands = np.zeros(len(cal_x))
#     lamvec = np.array([1.25, 1.63, 2.2]) # grizy, JHK
#     lamvec = np.array([0.49, 0.62, 0.75, 0.87, 0.96, 1.25, 1.63]) # grizy, JHK
    
    lamvec = np.array([0.49, 0.62, 0.75, 0.87, 0.96, 1.25, 1.63, 2.2]) # grizy, JHK
    finelam = np.linspace(0.3, 2.5, 500)
    
    all_splines = []
    
    for i in range(calibration_cat_sel.shape[0]):
        magvec = np.array(list(calibration_cat_sel[i,9:])+list(calibration_cat_sel[i,6:9]))
        fluxvec = 10**(-0.4*(magvec-23.9))
        whichgood = (~np.isinf(fluxvec))*(~np.isnan(fluxvec))*(magvec >5.)

        if interp_mode=='flux':
            datavec = fluxvec[whichgood]
        elif interp_mode=='mag':
            datavec = magvec[whichgood]
            
        nbands[i] = len(datavec)
        
        if nbands[i] < 2:
            all_splines.append(None)
            continue

        if nbands[i] <= spline_k:
            sed_spline = scipy.interpolate.UnivariateSpline(lamvec[whichgood], datavec, k=min(spline_k, nbands[i]-1))
        else:
            sed_spline = scipy.interpolate.UnivariateSpline(lamvec[whichgood], datavec, k=spline_k)
              
#         if len(fluxvec[whichgood])<4:
#             if len(fluxvec[whichgood])==3:
#                 print(lamvec[whichgood])
#                 sed_spline = scipy.interpolate.UnivariateSpline(lamvec[whichgood], fluxvec[whichgood], k=1)
#             elif len(fluxvec[whichgood])==2:
#                 sed_spline = scipy.interpolate.UnivariateSpline(lamvec[whichgood], fluxvec[whichgood], k=1)
#             else:
#                 all_splines.append(None)
#                 continue
#         else:
#     #         print(lamvec[whichgood], fluxvec[whichgood])
#             sed_spline = scipy.interpolate.UnivariateSpline(lamvec[whichgood], fluxvec[whichgood], k=spline_k)

        all_splines.append(sed_spline)
    
    
        if interp_mode=='flux':
            ciber_fluxes[i,:] = sed_spline(ciber_lam)
        elif interp_mode=='mag':
            ciber_fluxes[i,:] = 10**(-0.4*(sed_spline(ciber_lam)-23.9))

        ciber_twom_flux_ratio[i,inst-1] = ciber_fluxes[i,inst-1]/fluxvec[magidx-1]
            
        if i < 3 and plot:
            plt.figure()
            plt.scatter(lamvec[whichgood], datavec, color='k')
            plt.plot(finelam, sed_spline(finelam), color='r')

            if interp_mode=='flux':
                plt.ylabel('Flux density [$\\mu$Jy]')

            else:
                plt.ylabel('AB mag')
                plt.gca().invert_yaxis()

#                 plt.plot(finelam, 10**(-0.4*(sed_spline(finelam)-23.9)), color='r')
                
            plt.scatter(ciber_lam, sed_spline(ciber_lam), color='b')
            plt.xlabel('$\\lambda$ [um]')
            plt.show()
            
    
    return cal_x, cal_y, ciber_fluxes, all_splines, calmags, calmags_cross, nbands, ciber_twom_flux_ratio


def calc_pred_flux(cbps, flux_dens, lam_eff, bandpass_corr=None):
    
    cmock = ciber_mock()
    
    flux_dens_Jy = 1e-6*flux_dens
    
    nunaught = 3e8/lam_eff
    
    pred_flux = flux_dens_Jy*1e-17*nunaught

    pred_flux /= cmock.pix_sr    
    
    if bandpass_corr is not None:
        
        pred_flux /= bandpass_corr
        
    return pred_flux.value

def compute_bandpass_correction(filt_lam, filt_T, sed_modl, lam_eff, filter_lam_width=None, interp_mode='flux'):
    
    if filter_lam_width is None:
        filt_mask = (filt_T > 1e-3*np.max(filt_T))
        filter_lam_width = np.max(filt_lam[filt_mask]) - np.min(filt_lam[filt_mask])
    
    if interp_mode=='flux':
        fluxdens_lameff = sed_modl(lam_eff)
        sed_modl_at_filt = sed_modl(filt_lam[:-1])
    elif interp_mode=='mag':
        fluxdens_lameff = 10**(-0.4*(sed_modl(lam_eff)-23.9))
        sed_modl_at_filt = 10**(-0.4*(sed_modl(filt_lam[:-1])-23.9))
    
    denom = fluxdens_lameff*filter_lam_width
    
    dlam= filt_lam[1:]-filt_lam[:-1]
    
    
    intflux = np.sum(sed_modl_at_filt*filt_lam[:-1]*filt_T[:-1]*dlam)/np.sum(dlam)
    
    bandpass_corr = intflux / denom
    
    return bandpass_corr


def gen_bkg_mask(dx, bkg_rad=0.3):
    
    nx = 2*dx+1
    bkg_mask = np.zeros((nx, nx))    
    meshx, meshy = np.meshgrid(np.arange(nx), np.arange(nx))
    
    deltax = meshx - nx//2
    deltay = meshy - nx//2   
    dr = np.sqrt(deltax**2+deltay**2)
        
    bkg_mask[dr > bkg_rad*nx] = 1.
    
    return bkg_mask.astype(int)

def grab_postage_stamps_sdwfs(cal_src_posx, cal_src_posy, mosaic_im, dx, bkg_mask, mask_frac_min=None, plot=False):
    
    nx = 2*dx+1
    npix_post = nx**2
        
    aper_mask = np.zeros_like(bkg_mask)
    aper_mask[bkg_mask==0] = 1
    all_postage_stamps, all_postage_stamps_mask = [np.zeros((len(cal_src_posx), nx, nx)) for x in range(2)]
    post_bool = np.zeros_like(cal_src_posx)

    for n in range(len(cal_src_posx)):
        caty, catx = int(cal_src_posx[n]), int(cal_src_posy[n])
        x0, x1, y0, y1 = catx-dx, catx+dx+1, caty-dx, caty+dx+1

        sdwfs_post = mosaic_im[x0:x1, y0:y1]

        if n < 5 and plot:
            plot_map(sdwfs_post, title='flight post, (catx, caty)=('+str(catx)+','+str(caty)+')', figsize=(5,5))

        all_postage_stamps[n] = sdwfs_post
    
    return all_postage_stamps, aper_mask


def grab_postage_stamps(inst, cal_src_posx, cal_src_posy, flight_im, mask, dx, neighbor_cat=None, mask_frac_min=None, bkg_mask=None, plot=False, \
                       skip_nn=True, neighbor_src_posx=None, neighbor_src_posy=None):
    
    nx = 2*dx+1
    npix_post = nx**2
        
    aper_mask = np.zeros_like(bkg_mask)
    aper_mask[bkg_mask==0] = 1
    
    all_postage_stamps, all_postage_stamps_mask = [np.zeros((len(cal_src_posx), nx, nx)) for x in range(2)]
    post_bool = np.zeros_like(cal_src_posx)
    
    if neighbor_cat is not None:
        neighbor_x = neighbor_cat['x'+str(inst)]
        neighbor_y = neighbor_cat['y'+str(inst)]
        
    
    for n in range(len(cal_src_posx)):
        
        caty, catx = int(cal_src_posx[n]), int(cal_src_posy[n])
        x0, x1, y0, y1 = catx-dx, catx+dx+1, caty-dx, caty+dx+1
        
        mask_post = mask[x0:x1, y0:y1]
        
        if skip_nn:

            if neighbor_src_posx is not None and neighbor_src_posy is not None:
                dxp = neighbor_src_posx-cal_src_posx[n]
                dyp = neighbor_src_posy-cal_src_posy[n]        
            else:        
                dxp = cal_src_posx-cal_src_posx[n]
                dyp = cal_src_posy-cal_src_posy[n]

            dr = np.sqrt(dxp**2+dyp**2)

            which_nearby = (dr < np.sqrt(2)*dx)

            sum_in_post = np.sum(which_nearby)


            if sum_in_post > 1:
                continue
        
        if mask_frac_min is not None:
            mask_frac = float(np.nansum(mask_post))/float(nx**2)            
            if mask_frac < mask_frac_min:
                continue
                
        flight_post = flight_im[x0:x1, y0:y1]
   
    
        mask_in_aper = np.array((aper_mask==1)*(mask_post==0)).astype(int)

#         if n < 10 and plot:
#             plot_map(flight_post, title='flight post, (catx, caty)=('+str(catx)+','+str(caty)+')', figsize=(5,5))
#             plot_map(mask_post, title='mask post', figsize=(5,5))

        all_postage_stamps[n] = flight_post
        all_postage_stamps_mask[n] = mask_post
        post_bool[n] = 1
        
    print('sum post bool:', np.sum(post_bool))
        
    return all_postage_stamps, all_postage_stamps_mask, post_bool

def aper_phot(flight_post, mask_post, bkg_mask, mode='median', plot=False):
    
    # perform bkg subtraction
    
    bkg_mask_use = (mask_post*bkg_mask).astype(bool)    
    aper_mask_use = mask_post*(~bkg_mask_use)

    NA = np.sum(aper_mask_use)
    NB = np.sum(bkg_mask_use)

    if mode=='mean':
        bkg_level = np.nanmean(flight_post[bkg_mask_use])
        k = 1.
    elif mode=='median':
        bkg_level = np.nanmedian(flight_post[bkg_mask_use])
        k = np.pi/2.
        
    bkg_var = np.nanvar(flight_post[bkg_mask_use])
    
    aper_post = (flight_post-bkg_level)*aper_mask_use
    
    if plot:
        plot_map(bkg_mask_use, title='bkg mask use')
        plot_map(aper_mask_use, title='aper mask use')
        plot_map(aper_post, title='aper post')
        
    aper_flux = np.nansum(aper_post)
    aper_flux_var = np.sum(np.abs(flight_post-bkg_level)) + (NA+k*NA**2/NB)*bkg_var
    
    return aper_flux, aper_flux_var, bkg_level, bkg_var, aper_post

def ciber_sb_cal(inst, ifield, mask_base_path, mask_tail, dx=6, bkg_rad=0.3, m_min=11.0, m_max=14.5, load_neighbor_catalog=True, \
                mask_frac_min=0.9, ff_corr=False, plot=False, spline_k=3, interp_mode='flux'):
    
    cbps = CIBER_PS_pipeline()
    
    if inst==1:
        bandstr = 'J'
        lam_eff = 1.05
        jband_transmittance = np.loadtxt('data/bands/iband_transmittance.txt', skiprows=1)
        ciber_filt_lam = jband_transmittance[:,0]*1e-3
        ciber_filt_T = jband_transmittance[:,1]
    elif inst==2:
        bandstr = 'H'
        lam_eff = 1.79
        hband_transmittance = np.loadtxt('data/bands/hband_transmittance.txt', skiprows=1)
        ciber_filt_lam = hband_transmittance[:,0]*1e-3
        ciber_filt_T = hband_transmittance[:,1]
        
    ciber_filt_T /= np.max(ciber_filt_T)
        
    bkg_mask = gen_bkg_mask(dx, bkg_rad=bkg_rad)
    
    aper_mask = ~bkg_mask
    plot_map(aper_mask.astype(int))
    
    # load flight image
    flight_im = cbps.load_flight_image(ifield, inst, inplace=False)
    if plot:
        plot_map(flight_im)

    # load mask
    mask_fpath = mask_base_path+'/maskInst_aducorr/field'+str(ifield)+'_TM'+str(inst)+'_maskInst_aducorr.fits'
#     mask_fpath = mask_base_path+'/maskInst_102422/field'+str(ifield)+'_TM'+str(inst)+'_maskInst_102422.fits'
#     mask_fpath = mask_base_path+'/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_' + mask_tail+'.fits'
    mask = fits.open(mask_fpath)[1].data
    
    plot_map(flight_im*mask, title='flight im ')
    
    mask *= (flight_im!=0)
    mask *= (flight_im < 0) # negative adu/fr
    mask *= ~np.isnan(flight_im)
    mask *= ~np.isinf(flight_im)
    plot_map(mask)
    
    # load flat field estimate
    if ff_corr:
        ff_fpath = 'ff_estimate_TM'+str(inst)+'_ifield'+str(ifield)
        ff_est = fits.open(ff_fpath)[1].data
        plot_map(ff_est)
    else:
        ff_est = None
    
    # dark current subtraction
    dark_current = cbps.load_dark_current_template(inst, inplace=False)
    if plot:
        plot_map(dark_current)
    print('subtracting dark current template..')
    flight_im -= dark_current
    
    if load_neighbor_catalog:
        neighbor_cat_fpath = config.ciber_basepath+'data/catalogs/2MASS/filt/2MASS_filt_rdflag_wxy_'+cbps.ciber_field_dict[ifield]+'_Jlt17.5.csv'
        neighbor_cat = pd.read_csv(neighbor_cat_fpath)
    else:
        neighbor_cat = None
        
    cal_src_posx, cal_src_posy, pred_ciber_flux_densities, sed_modls, calmags, _, nbands, _ = load_calib_catalog(inst, ifield, m_min=m_min, m_max=m_max, spline_k=spline_k, \
                                                                                                          interp_mode=interp_mode)
        
    print('nbands has shape ', nbands.shape)
        
    all_pred_fluxes, all_measured_fluxes,\
        all_var_measured_fluxes, all_bp_corr,\
            all_g1g2, all_bkg_level, all_bkg_var = [np.zeros_like(cal_src_posx) for x in range(7)]
    
    all_postage_stamps, all_postage_stamps_mask, post_bool = grab_postage_stamps(inst, cal_src_posx, cal_src_posy, flight_im, mask, dx, neighbor_cat=neighbor_cat, mask_frac_min=mask_frac_min, \
                                                                                bkg_mask=bkg_mask)
    
    
#     all_H_flux, all_K_flux
    # aperture photometry on postage stamps, compare with predicted fluxes
    
    for n in range(len(post_bool)):
        
        if post_bool[n]==1:
                
            aper_flux, aper_flux_var, bkg_level, bkg_var, _ = aper_phot(all_postage_stamps[n], all_postage_stamps_mask[n], bkg_mask, mode='mean', plot=False)

            bandpass_corr = compute_bandpass_correction(ciber_filt_lam, ciber_filt_T, sed_modls[n], lam_eff, interp_mode=interp_mode)
            
            pred_flux_corr = calc_pred_flux(cbps, pred_ciber_flux_densities[n,inst-1], lam_eff*1e-6*u.m, bandpass_corr=bandpass_corr)
            
            all_bkg_level[n] = bkg_level
            all_bkg_var[n] = bkg_var
            all_g1g2[n] = pred_flux_corr / aper_flux
            all_bp_corr[n] = bandpass_corr
            all_pred_fluxes[n] = pred_flux_corr
            all_measured_fluxes[n] = aper_flux
            all_var_measured_fluxes[n] = aper_flux_var
            
            plotstr = 'g1g2='+str(np.round(all_g1g2[n], 1))+', (x,y)=('+str(int(cal_src_posx[n]))+','+str(int(cal_src_posy[n]))+')'
            plotstr += '\n'+bandstr+'='+str(np.round(calmags[n], 1))
            plotstr += '\nPred: '+str(np.round(all_pred_fluxes[n], 1))+', Meas: '+str(np.round(all_measured_fluxes[n], 1))

#             if all_pred_fluxes[n] > 7e4:
#                 fig = plot_map(all_postage_stamps[n]*all_postage_stamps_mask[n], title=plotstr, figsize=(5,5), return_fig=True, show=False)
                
            
#             if plot:
#                 fig = plot_map(all_postage_stamps[n], title=plotstr, figsize=(5,5), return_fig=True, show=True)
#                 fig.savefig(config.ciber_basepath+'figures/calibration_post_stamps/TM'+str(inst)+'/ifield'+str(ifield)+'/calibration_post_'+str(n)+'.png', bbox_inches='tight')
#                 plot_map(all_postage_stamps_mask[n], title=plotstr, figsize=(5,5), show=True)
            
            
    aper_snr = np.abs(all_measured_fluxes)/np.sqrt(all_var_measured_fluxes)
    
    if inst==1:
        varmax = 50
        calmin, calmax = -450, -150
    else:
        varmax = 50
        calmin, calmax = -200, 0
        
#     if ifield==5:
#         varmax = 100
        
    if inst==1:
        predmin = 4500
    elif inst==2:
        predmin = 1500
        
    snr_mask = (np.abs(all_bkg_level-np.median(all_bkg_level)) < np.abs(np.median(all_bkg_level)))
    snr_mask *= (np.sqrt(all_var_measured_fluxes) < varmax)
    snr_mask *= (aper_snr > 2.0)
    
    snr_mask *= (all_pred_fluxes > predmin)
    
    use_mask = (all_g1g2<0)*snr_mask*(all_measured_fluxes != 0)
    if ifield==5:
        use_mask *=(all_bkg_var < 0.5)
    else:
        use_mask *= (all_bkg_var < 0.3)

        
    sel_pred_fluxes = all_pred_fluxes[use_mask]
    sel_measured_fluxes = all_measured_fluxes[use_mask]
    sel_var_measured_fluxes = all_var_measured_fluxes[use_mask]
    
    sel_bp_corr = all_bp_corr[use_mask]
    sel_g1g2 = all_g1g2[use_mask]
    sel_bkg_level = all_bkg_level[use_mask]
    sel_nbands = nbands[use_mask]
    sel_aper_snr = aper_snr[use_mask]
    sel_cal_src_posx = cal_src_posx[use_mask]
    sel_cal_src_posy = cal_src_posy[use_mask]
    
    sel_postage_stamps = all_postage_stamps[use_mask,:,:] 
    sel_postage_stamps_mask = all_postage_stamps_mask[use_mask,:,:]
    sel_calmags = calmags[use_mask]
    sel_bkg_var = all_bkg_var[use_mask]
    
    plt.figure()
#     plt.scatter(np.abs(all_measured_fluxes[all_measured_fluxes!=0]), aper_snr[all_measured_fluxes!=0])
    plt.scatter(np.abs(sel_measured_fluxes), np.sqrt(sel_var_measured_fluxes))
    
    plt.xscale('log')
    plt.xlim(1e0, 1e3)
    plt.yscale('log')
    plt.show()
    plt.figure(figsize=(10, 5))
    plt.subplot(2,3,1)
    plt.scatter(sel_bkg_var, sel_g1g2, color='k', s=5, alpha=0.5)
    plt.xlabel('Background Var [ADU/fr]$^2$')
    plt.ylabel('$G_1G_2$ [nW m$^{-2}$ sr$^{-1}$/ADU fr$^{-1}$]')
    plt.ylim(calmin, calmax)
    plt.xlim(0, np.percentile(sel_bkg_var, 84))
    
    plt.subplot(2,3,2)
    plt.scatter(sel_aper_snr, sel_g1g2, color='k', s=5, alpha=0.5)
    plt.xlabel('Flux SNR')
    plt.ylabel('$G_1G_2$ [nW m$^{-2}$ sr$^{-1}$/ADU fr$^{-1}$]')
    plt.xlim(0.5, 50)
    plt.xscale('log')
    plt.ylim(calmin, calmax)

    plt.subplot(2,3,3)
    plt.scatter(sel_bp_corr, sel_g1g2, color='k', s=5, alpha=0.5)
    plt.xlabel('Bandpass correction')
    plt.xlim(np.percentile(sel_bp_corr, 5), np.max(sel_bp_corr))
    plt.ylabel('$G_1G_2$ [nW m$^{-2}$ sr$^{-1}$/ADU fr$^{-1}$]')
    plt.ylim(calmin, calmax)

    plt.subplot(2,3,4)
    plt.scatter(sel_pred_fluxes, -1.*sel_measured_fluxes, c=sel_bkg_var, s=2, alpha=0.8, vmin=np.percentile(sel_bkg_var, 1), vmax=np.percentile(sel_bkg_var, 90))
    plt.colorbar()
#     plt.xlabel('Background [ADU/fr]')
#     plt.ylabel('$G_1G_2$ [nW m$^{-2}$ sr$^{-1}$/ADU fr$^{-1}$]')
#     plt.ylim(calmin, calmax)
    plt.ylim(1e1, 1e3)
    plt.xlim(2e3, 1e5)
    plt.xscale('log')
    plt.yscale('log')
    plt.subplot(2,3,5)
    plt.scatter(sel_pred_fluxes, -1.*sel_measured_fluxes, c=sel_aper_snr, s=5, alpha=0.5)
    plt.colorbar()
#     plt.xlabel('Flux SNR')
#     plt.ylabel('$G_1G_2$ [nW m$^{-2}$ sr$^{-1}$/ADU fr$^{-1}$]')
    plt.xlim(0.5, 50)
    plt.xscale('log')
#     plt.ylim(calmin, calmax)
    plt.ylim(1e1, 1e3)
    plt.xlim(2e3, 1e5)
    plt.xscale('log')
    plt.yscale('log')
    plt.subplot(2,3,6)
    plt.scatter(sel_pred_fluxes, -1.*sel_measured_fluxes, c=sel_bp_corr, s=2, vmin=0.95, vmax=1.05)
#     plt.xlabel('Bandpass correction')
    plt.colorbar()

#     plt.xlim(np.percentile(sel_bp_corr, 5), np.max(sel_bp_corr))
#     plt.ylabel('$G_1G_2$ [nW m$^{-2}$ sr$^{-1}$/ADU fr$^{-1}$]')
#     plt.ylim(calmin, calmax)

    
    plt.ylim(1e1, 1e3)
    plt.xlim(2e3, 1e5)
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
    
#     if inst==1:
#         varmax = 30
#         calmin, calmax = -450, -150
#     else:
#         varmax = 40
#         calmin, calmax = -200, 0
    
#     snr_mask = (np.abs(all_bkg_level-np.median(all_bkg_level)) < np.abs(np.median(all_bkg_level)))
    
#     snr_mask *= (np.sqrt(all_var_measured_fluxes) < varmax)
#     snr_mask *= (aper_snr > 0.5)
    
    
#     for n in range(len(sel_postage_stamps)):

#         plotstr = 'g1g2='+str(np.round(sel_g1g2[n], 1))+', (x,y)=('+str(int(sel_cal_src_posx[n]))+','+str(int(sel_cal_src_posy[n]))+')'
#         plotstr += '\n'+bandstr+'='+str(np.round(sel_calmags[n], 1))

#         plotstr += '\nPred: '+str(np.round(sel_pred_fluxes[n], 1))+', Meas: '+str(np.round(sel_measured_fluxes[n], 1))


#         fig = plot_map(sel_postage_stamps[n], title=plotstr, figsize=(5,5), return_fig=True, show=True, lopct=1)
#         fig.savefig(config.ciber_basepath+'figures/calibration_post_stamps/TM'+str(inst)+'/ifield'+str(ifield)+'/calibration_post_'+str(n)+'.png', bbox_inches='tight')
# #         plot_map(all_postage_stamps_mask[n], title=plotstr, figsize=(5,5), show=True)
    
    plt.figure()
    plt.hist(sel_g1g2, bins=np.linspace(calmin, calmax, 20), label='sel')
#     plt.hist(all_g1g2, bins=np.linspace(calmin, calmax, 20), label='all')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.scatter(sel_pred_fluxes, 1./np.sqrt(sel_var_measured_fluxes))
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('pred fluxes')
    plt.ylabel('weights')
    plt.show()
        

    fig = plt.figure(figsize=(5,4))
    plt.title('$\\lambda=$'+str(lam_eff)+' $\\mu$m, '+cbps.ciber_field_dict[ifield], fontsize=13)
    # plt.title('TM'+str(inst)+', ifield '+str(ifield))
    plt.errorbar(sel_pred_fluxes, -1.*sel_measured_fluxes, yerr=np.sqrt(sel_var_measured_fluxes), color='k', fmt='o', markersize=2, alpha=0.4)

    if inst==1:
        whichfmin = [5000, 10000, 30000]
    else:
        whichfmin = [2000, 5000, 10000]
        
    fineflux = np.logspace(3, 6, 100)
    for fidx, fmin in enumerate(whichfmin):
        
        which_above_fmin = (sel_pred_fluxes > fmin)
        popt, pcov = curve_fit(linear_func_nointercept, sel_pred_fluxes[which_above_fmin], sel_measured_fluxes[which_above_fmin],  sigma=np.sqrt(sel_var_measured_fluxes[which_above_fmin]), absolute_sigma=True)
        plt.axvline(fmin, color='C'+str(fidx), linestyle='solid', linewidth=2)
        slopeunc = (1./popt[0])*(np.sqrt(pcov[0,0])/popt[0])
        plt.plot(fineflux, -1.*popt[0]*fineflux, label='$g_ag_1g_2=$'+str(np.round(1./popt[0], 1))+'$\\pm$'+str(np.round(slopeunc, 1)), linestyle='dashed', color='C'+str(fidx), zorder=10)
        
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('2MASS flux [nW m$^{-2}$]', fontsize=13)
    plt.ylabel('Measured signal [-1$\\times$ADU fr$^{-1}$]', fontsize=13)
    plt.grid(alpha=0.5)
    plt.ylim(1e0, 1e3)
    plt.xlim(predmin, 2e5)
#     plt.savefig(config.ciber_basepath+'figures/calibration_results/pred_vs_measured_flux_cal_TM'+str(inst)+'_ifield'+str(ifield)+'_dx='+str(dx)+'.png', bbox_inches='tight', dpi=200)
    plt.show()
    

    return sel_cal_src_posx, sel_cal_src_posy, sel_pred_fluxes, sel_measured_fluxes, sel_var_measured_fluxes,\
                sel_bp_corr, sel_g1g2, sel_bkg_level, fig

def jackknife_g1g2_slope(pred_fluxes, measured_fluxes, var_measured_fluxes, with_intercept=False, plot=False):
    
    nsrc = len(pred_fluxes)
    idxrange = np.arange(nsrc)
    
    all_g1g2 = np.zeros((nsrc))
    
    if with_intercept:
        popt, pcov = curve_fit(linear_func, pred_fluxes, measured_fluxes, sigma=np.sqrt(var_measured_fluxes), absolute_sigma=True)
    else:
        popt, pcov = curve_fit(linear_func_nointercept, pred_fluxes, measured_fluxes, sigma=np.sqrt(var_measured_fluxes), absolute_sigma=True)
    
    g1g2_full = 1./popt[0]
    g1g2_unc_full = g1g2_full*(np.sqrt(pcov[0,0])/popt[0])
    
    for x in range(nsrc):
        which_idx = (idxrange != x)
        pred_fluxes_jack = pred_fluxes[which_idx]
        measured_fluxes_jack = measured_fluxes[which_idx]
        var_measured_fluxes_jack = var_measured_fluxes[which_idx]
        
        if with_intercept:
            popt, pcov = curve_fit(linear_func, pred_fluxes_jack, measured_fluxes_jack, sigma=np.sqrt(var_measured_fluxes_jack), absolute_sigma=True)
        else:
            popt, pcov = curve_fit(linear_func_nointercept, pred_fluxes_jack, measured_fluxes_jack, sigma=np.sqrt(var_measured_fluxes_jack), absolute_sigma=True)
        
        all_g1g2[x] = 1./popt[0]
        
    if plot:
        plt.figure()
        plt.hist(all_g1g2)
        plt.axvline(np.median(all_g1g2), linestyle='dashed', label='Median')
        plt.axvline(np.mean(all_g1g2), linestyle='dashed', label='Mean')
        plt.axvline(g1g2_full, linestyle='dashed', label='Full set')
        plt.legend()
        plt.xlabel('G1G2 [nW m$^{-2}$ sr$^{-1}$/ADU fr$^{-1}]')
        plt.show()
    
    print('g1g2_full, np.median(all_g1g2), np.mean(all_g1g2), np.std(all_g1g2), g1g2_unc_full')
    print(g1g2_full, np.median(all_g1g2), np.mean(all_g1g2), np.std(all_g1g2), g1g2_unc_full)
    
    return all_g1g2

def make_calibration_catalog(ifield, catalog_basepath=None, save=False):
    
    cbps = CIBER_PS_pipeline()
    fieldname = cbps.ciber_field_dict[ifield]

    if catalog_basepath is None:
        catalog_basepath = config.exthdpath+'ciber/ciber1/data/catalogs/'

    # 2MASS
    twomass_cat = pd.read_csv(catalog_basepath+'2MASS/filt/2MASS_filt_rdflag_wxy_'+fieldname+'_Jlt16.0.csv')
    # convert to AB
    twomass_J = twomass_cat['j_m'] + 0.91
    twomass_H = twomass_cat['h_m'] + 1.39
    twomass_K = twomass_cat['k_m'] + 1.83
    
    twomass_ra = np.array(twomass_cat['ra'])
    twomass_dec = np.array(twomass_cat['dec'])

    twomass_x1 = np.array(twomass_cat['x1'])
    twomass_x2 = np.array(twomass_cat['x2'])

    twomass_y1 = np.array(twomass_cat['y1'])
    twomass_y2 = np.array(twomass_cat['y2'])
    
    # PanSTARRS
    panstarrs_cat = pd.read_csv(catalog_basepath+'PanSTARRS/filt/'+fieldname+'_0_102120_filt_any_band_detect.csv')
    panstarrs_ra = panstarrs_cat['ra']
    panstarrs_dec = panstarrs_cat['dec']

    ps_g = np.array(panstarrs_cat['gMeanPSFMag'])
    ps_r = np.array(panstarrs_cat['rMeanPSFMag'])
    ps_i = np.array(panstarrs_cat['iMeanPSFMag'])
    ps_z = np.array(panstarrs_cat['zMeanPSFMag'])
    ps_y = np.array(panstarrs_cat['yMeanPSFMag'])
    
    
    calibration_cat = np.zeros((len(twomass_ra), 14))

    calibration_cat[:,0] = twomass_ra
    calibration_cat[:,1] = twomass_dec

    calibration_cat[:,2] = twomass_x1
    calibration_cat[:,3] = twomass_y1

    calibration_cat[:,4] = twomass_x2
    calibration_cat[:,5] = twomass_y2

    calibration_cat[:,6] = twomass_J
    calibration_cat[:,7] = twomass_H
    calibration_cat[:,8] = twomass_K
    
    for x in range(len(twomass_ra)):
    
        dra = twomass_ra[x]-panstarrs_ra
        ddec = twomass_dec[x]-panstarrs_dec
        
        dr = np.sqrt(dra**2+ddec**2)*3600
        
        whichmin = np.where((dr<2.0))[0]
        
        if len(whichmin) == 0:
            continue
        
        if len(whichmin) > 1:

            mags_g = ps_g[whichmin]
            mags_r = ps_r[whichmin]
            mags_i = ps_i[whichmin]
            mags_z = ps_g[whichmin]
            mags_y = ps_y[whichmin]
            
            flux_g = 10**(-0.4*(mags_g-23.9))
            flux_r = 10**(-0.4*(mags_r-23.9))    
            flux_i = 10**(-0.4*(mags_i-23.9))
            flux_z = 10**(-0.4*(mags_z-23.9))
            flux_y = 10**(-0.4*(mags_y-23.9))
            
            sumflux_g = np.sum(flux_g[mags_g != -99.])
            sumflux_r = np.sum(flux_r[mags_r != -99.])
            sumflux_i = np.sum(flux_i[mags_i != -99.])
            sumflux_z = np.sum(flux_z[mags_z != -99.])
            sumflux_y = np.sum(flux_y[mags_y != -99.])
            
            calibration_cat[x,9] = -2.5*np.log10(sumflux_g)+23.9
            calibration_cat[x,10] = -2.5*np.log10(sumflux_r)+23.9
            calibration_cat[x,11] = -2.5*np.log10(sumflux_i)+23.9
            calibration_cat[x,12] = -2.5*np.log10(sumflux_z)+23.9
            calibration_cat[x,13] = -2.5*np.log10(sumflux_y)+23.9
            
        else:
            
            calibration_cat[x,9] = ps_g[whichmin[0]]
            calibration_cat[x,10] = ps_r[whichmin[0]]
            calibration_cat[x,11] = ps_i[whichmin[0]]
            calibration_cat[x,12] = ps_z[whichmin[0]]
            calibration_cat[x,13] = ps_y[whichmin[0]]
            
        bins = 20
            
    plt.figure()
    plt.hist(calibration_cat[:,6], bins=bins, histtype='step', label='J')
    plt.hist(calibration_cat[:,7], bins=bins, histtype='step', label='H')
    plt.hist(calibration_cat[:,8], bins=bins, histtype='step', label='K')

    plt.hist(calibration_cat[:,9], bins=bins, histtype='step', label='g')
    plt.hist(calibration_cat[:,10], bins=bins, histtype='step', label='r')
    plt.hist(calibration_cat[:,11], bins=bins, histtype='step', label='i')
    plt.hist(calibration_cat[:,12], bins=bins, histtype='step', label='z')
    plt.hist(calibration_cat[:,13], bins=bins, histtype='step', label='y')

    plt.legend()
    plt.yscale('log')
    plt.show()
    
    if save:
        save_fpath = catalog_basepath+'calibration_catalogs/calibration_src_catalog_ifield'+str(ifield)+'.npz'
        print('save fpath is ', save_fpath)
        np.savez(save_fpath, calibration_cat=calibration_cat, \
            columns=['ra', 'dec', 'x1', 'y1', 'x2', 'y2', 'j_m', 'h_m', 'k_m', 'g', 'r', 'i', 'z', 'y'])
        
    return calibration_cat


''' 
These functions were made for reproducing the Z14 calibration as closely as possible, are not the final scripts used in 
the 4th flight gain calibration
'''
def stack_srcs_simp(ciber_map, tracer_cat, mask, dx=5, min_mask_frac=0.9):
    
    dimx = 2*dx + 1
    all_postage_stamps, all_mask_fractions = [], []
    
    for x in range(len(tracer_cat)):
        x0, x1, y0, y1 = int(tracer_cat[x,1])-dx, int(tracer_cat[x,1])+dx+1,\
                                int(tracer_cat[x,0])-dx, int(tracer_cat[x,0])+dx+1

        maskcutout = mask[x0:x1, y0:y1]
        mask_frac = float(np.nansum(maskcutout))/float(dimx**2)
        if mask_frac > min_mask_frac:
            ciber_cutout = ciber_map[x0:x1, y0:y1]                

            all_postage_stamps.append(ciber_cutout*maskcutout)

    return np.array(all_postage_stamps), np.array(all_mask_fractions)

def repr_mike_calibration(inst, ifield, mask_tail, dm=0.1, m_min=13.5, m_max=15.0, trim_edge=50, dx=3):
    
    # ------------ instantiate cbps object and file paths -------------------
    config_dict, pscb_dict, float_param_dict, fpath_dict = return_default_cbps_dicts()
    ciber_mock_fpath = config.exthdpath+'ciber_mocks/'
    cbps = CIBER_PS_pipeline()
    
    datestr = '112022'
    datestr_trilegal = datestr
    fpath_dict, list_of_dirpaths, base_path, trilegal_base_path = set_up_filepaths_cbps(fpath_dict, inst, 'test', datestr,\
                                                                                        datestr_trilegal=datestr, data_type='observed', \
                                                                                       save_fpaths=True)
    
    base_fluc_path = config.exthdpath+'ciber_fluctuation_data/'
    tempbank_dirpath = base_fluc_path+'/TM'+str(inst)+'/subpixel_psfs/'
    catalog_basepath = base_fluc_path+'catalogs/'
    magkey_dict = dict({1:'j_m', 2:'h_m'})
    lam_dict = dict({1:1.05, 2:1.75})
    nunaught_dict = dict({1:2.867e14, 2:1.6802e14})
    fwhm_dict = dict({1:10.3, 2:10.6})
    
    nx = 2*dx+1
    meshgrid_x, meshgrid_y = np.meshgrid(np.arange(nx) - nx//2, np.arange(nx) - nx//2)
    
    r = np.sqrt(meshgrid_x**2 + meshgrid_y**2)
    
    bkg_mask = np.zeros((nx, nx))
    
    bkg_mask[(r > 0.4*nx)] = 1.
    
    plot_map(bkg_mask, title='rmask')

    # --------------
    mag_range = np.arange(m_min, m_max+dm, dm)
    print('mag range = ', mag_range)
    n_mag_bins = len(mag_range)-1
    field_name = cbps.ciber_field_dict[ifield]

    # load mask
    mask_fpath = fpath_dict['mask_base_path']+'/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'
    mask = fits.open(mask_fpath)[1].data

    # load flight image
#     flight_im = fits.open('/Users/richardfeder/Downloads/ciber_TM'+str(inst)+'_ifield'+str(ifield)+'_proc_lin_short_081023.fits')[1].data

    flight_im = fits.open('/Users/richardfeder/Downloads/ciber_TM'+str(inst)+'_ifield'+str(ifield)+'_proc_080423.fits')[1].data
    ciber_photocurrent_map = -1.*flight_im / cbps.cal_facs[inst]
    
    median_flight_im = np.median(ciber_photocurrent_map[mask==1])

    ciber_photocurrent_map[mask==1] -= median_flight_im

    # load 2MASS catalog and grab all sources with m_Vega between m_min and m_max that are sufficiently far from edge
    twomass_cat = pd.read_csv(catalog_basepath+'2MASS/filt/2MASS_filt_rdflag_wxy_'+field_name+'_Jlt17.5.csv')
    twomass_x = np.array(twomass_cat['x'+str(inst)])
    twomass_y = np.array(twomass_cat['y'+str(inst)])
    twomass_mag = np.array(twomass_cat[magkey_dict[inst]]) # magkey   
    cal_src_mask = (twomass_mag >= m_min)*(twomass_mag <= m_max)*(twomass_x > trim_edge)*(twomass_x+trim_edge < 1024)*(twomass_y > trim_edge)*(twomass_y+trim_edge < 1024)

    twomass_cal_src_cat = np.array([twomass_x[cal_src_mask], twomass_y[cal_src_mask], twomass_mag[cal_src_mask]]).transpose()
    print('Number of calibration sources for '+field_name+' is '+str(twomass_cal_src_cat.shape[0]))        
        
    # ------------- for each Vega magnitude bin, calculate corresponding flux and stack sources in bin ----------
    
    sb_bin_predict, sb_cal, nsrc_perstack = [np.zeros((n_mag_bins)) for x in range(3)]
    all_med_postage_stamps = []
    all_sb_cal_srcs = []
    for b in range(n_mag_bins):
        
        # extrapolate Vega zero point to CIBER effective wavelength 
        # and calculate flux density
        zp_ciber = extrap_vega_zp(lam_dict[inst]) 
        ciber_flux_density_pred = zp_ciber*10**(-0.4*mag_range[b])
        
        # convert from Jy to nW m-2 Hz-1
        ciber_nW_m2_Hz_pred = ciber_flux_density_pred*1e-17 
        nunaught = 3e8 / (lam_dict[inst]*1e-6)
        
        # use effective frequency to convert to nW m-2
        ciber_nW_m2_pred = ciber_nW_m2_Hz_pred*nunaught
        
        # this grabs the values used in get_quick_psf.m, 
        # 10.3" for TM1 and 10.6" for TM2
        fwhm = fwhm_dict[inst] 
        
        # area of beam in steradians
        omega_beam = (1.133*(fwhm/3600.)**2*(np.pi/180.)**2) 
#         ciber_nW_m2_sr_pred = ciber_nW_m2_pred / omega_beam
        
        
        ciber_nW_m2_sr_pred = ciber_nW_m2_pred / cmock.pix_sr.value # step in question
        

        sb_bin_predict[b] = ciber_nW_m2_sr_pred
        stack_dm_mask = np.where((twomass_cal_src_cat[:,2] > mag_range[b])*(twomass_cal_src_cat[:,2] <= mag_range[b+1]))[0]
        twomass_dm_stackcat = twomass_cal_src_cat[stack_dm_mask, :]
                
        
        all_postage_stamps, all_mask_fractions = stack_srcs_simp(ciber_photocurrent_map, twomass_dm_stackcat, mask, dx=dx, min_mask_frac=0.9)
        
        nsrc_perstack[b] = len(all_postage_stamps)
        
        all_postage_stamps[np.isinf(all_postage_stamps)] = np.nan
        
        med_postage_stamps = np.nanmedian(all_postage_stamps, axis=0)
        sb_cal[b] = sb_bin_predict[b] / (np.nansum(med_postage_stamps)) 
        
        sb_cal_all_srcs = [sb_bin_predict[b]/np.nansum(post) for post in all_postage_stamps]

        all_sb_cal_srcs.append(sb_cal_all_srcs)
        all_med_postage_stamps.append(med_postage_stamps)
    
    print('sb cals are ', sb_cal)
    
    print('nsrc per stack is ', nsrc_perstack)
    
    return all_med_postage_stamps, mag_range, sb_cal, nsrc_perstack, all_sb_cal_srcs
    

        