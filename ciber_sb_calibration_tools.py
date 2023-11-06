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



def load_calib_catalog(inst, ifield, m_min=11, m_max=14.5, spline_k=2, trim_edge=50):
    
    calibration_cat = np.load('data/calibration_src_catalog_ifield'+str(ifield)+'.npz')['calibration_cat']
    
    ciber_lam = [1.05, 1.79]
    if inst==1:
        magidx = 6
        xidx, yidx = 2, 3
        Vega_to_AB = 0.91
    elif inst==2:
        magidx = 7
        xidx, yidx = 4, 5
        Vega_to_AB = 1.39
        
    calib_mag_ref = calibration_cat[:,magidx]-Vega_to_AB
    cal_posmask = (calibration_cat[:,xidx] > trim_edge)*(calibration_cat[:,xidx] < 1024-trim_edge)*(calibration_cat[:,yidx] > trim_edge)*(calibration_cat[:,yidx] < 1024-trim_edge)
    cal_magmask = (calib_mag_ref > m_min)*(calib_mag_ref < m_max)

    which_sel = np.where(cal_posmask*cal_magmask)[0]
    calmags = calib_mag_ref[which_sel]

    calibration_cat_sel = calibration_cat[which_sel, :]
    cal_x = calibration_cat_sel[:,xidx]
    cal_y = calibration_cat_sel[:,yidx]
    ciber_fluxes = np.zeros((len(cal_x), 2))
#     lamvec = np.array([1.25, 1.63, 2.2]) # grizy, JHK
    lamvec = np.array([0.49, 0.62, 0.75, 0.87, 0.96, 1.25, 1.63, 2.2]) # grizy, JHK
    finelam = np.linspace(0.3, 2.5, 500)
    
    all_splines = []
    
    for i in range(calibration_cat_sel.shape[0]):
        magvec = np.array(list(calibration_cat_sel[i,9:])+list(calibration_cat_sel[i,6:9]))
        fluxvec = 10**(-0.4*(magvec-23.9))
        whichgood = (~np.isinf(fluxvec))*(~np.isnan(fluxvec))*(magvec >5.)
        
        if len(fluxvec[whichgood])<3:
            all_splines.append(None)
            continue
        
        sed_spline = scipy.interpolate.UnivariateSpline(lamvec[whichgood], fluxvec[whichgood], k=spline_k)
        
        all_splines.append(sed_spline)
        ciber_fluxes[i,:] = sed_spline(ciber_lam)
        if i < 2:
            plt.figure()
            plt.scatter(lamvec[whichgood], fluxvec[whichgood], color='k')
            plt.plot(finelam, sed_spline(finelam), color='r')
            plt.scatter(ciber_lam, sed_spline(ciber_lam), color='b')
            plt.xlabel('$\\lambda$ [um]')
            plt.ylabel('Flux density [$\\mu$Jy]')
            plt.show()
            
    
    return cal_x, cal_y, ciber_fluxes, all_splines, calmags


def calc_pred_flux(cbps, flux_dens, lam_eff, bandpass_corr=None):
    
    cmock = ciber_mock()
    
    flux_dens_Jy = 1e-6*flux_dens
    
    nunaught = 3e8/lam_eff
    
    pred_flux = flux_dens_Jy*1e-17*nunaught

    pred_flux /= cmock.pix_sr    
    
    if bandpass_corr is not None:
        
        pred_flux /= bandpass_corr
        
    return pred_flux.value

def compute_bandpass_correction(filt_lam, filt_T, sed_modl, lam_eff, filter_lam_width=None):
    
    if filter_lam_width is None:
        filt_mask = (filt_T > 1e-3*np.max(filt_T))
        filter_lam_width = np.max(filt_lam[filt_mask]) - np.min(filt_lam[filt_mask])
    
    fluxdens_lameff = sed_modl(lam_eff)
    denom = fluxdens_lameff*filter_lam_width
    
    dlam= filt_lam[1:]-filt_lam[:-1]
    
    
    sed_modl_at_filt = sed_modl(filt_lam[:-1])
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

def grab_postage_stamps(inst, cal_src_posx, cal_src_posy, flight_im, mask, dx, neighbor_cat=None, mask_frac_min=None, bkg_mask=None, plot=False):
    
    nx = 2*dx+1
    npix_post = nx**2
        
    aper_mask = (~bkg_mask).astype(int)
    
    all_postage_stamps, all_postage_stamps_mask = [np.zeros((len(cal_src_posx), nx, nx)) for x in range(2)]
    post_bool = np.zeros_like(cal_src_posx)
    
    if neighbor_cat is not None:
        neighbor_x = neighbor_cat['x'+str(inst)]
        neighbor_y = neighbor_cat['y'+str(inst)]
        
    
    for n in range(len(cal_src_posx)):
        
        caty, catx = int(cal_src_posx[n]), int(cal_src_posy[n])
        
        x0, x1, y0, y1 = catx-dx, catx+dx+1, caty-dx, caty+dx+1
        
#         print(x0, x1, y0, y1)
        mask_post = mask[x0:x1, y0:y1]
        
        
        dxp = cal_src_posx-cal_src_posx[n]
        dyp = cal_src_posy-cal_src_posy[n]
        
        dr = np.sqrt(dxp**2+dyp**2)
        
        which_nearby = (dr < np.sqrt(2)*dx)
        
        # check for neighbors in stamp
        sum_in_post = np.sum(which_nearby)
        
        if sum_in_post > 1:
#             print('sum in post = ', sum_in_post)
            continue
        
        if mask_frac_min is not None:
            mask_frac = float(np.nansum(mask_post))/float(nx**2)            
            if mask_frac < mask_frac_min:
                continue
                
        flight_post = flight_im[x0:x1, y0:y1]
   
        mask_in_aper = (aper_mask==1)*(mask_post==0)
        if np.sum(mask_in_aper) > 0:
            
            plot_map(flight_post*aper_mask, title='bad pix skip', figsize=(5,5))
            continue

        all_postage_stamps[n] = flight_post
        all_postage_stamps_mask[n] = mask_post
        post_bool[n] = 1
        
    print('sum post bool:', np.sum(post_bool))
        
    return all_postage_stamps, all_postage_stamps_mask, post_bool

def aper_phot(flight_post, mask_post, bkg_mask, mode='median', plot=False):
    
    # perform bkg subtraction
    
    bkg_mask_use = bkg_mask.astype(bool)
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
    flight_post -= bkg_level
    aper_post = flight_post*aper_mask_use

    
    if plot:
        plot_map(bkg_mask_use, title='bkg mask use')
        plot_map(aper_mask_use, title='aper mask use')
        plot_map(aper_post, title='aper post')
        
    aper_flux = np.nansum(aper_post)
    
    aper_flux_var = np.sum(aper_post-bkg_level) + (NA+k*NA**2/NB)*bkg_var
    
    
    return aper_flux, aper_flux_var, bkg_level

def ciber_sb_cal(inst, ifield, mask_base_path, mask_tail, dx=6, bkg_rad=0.3, m_min=11.0, m_max=14.5, load_neighbor_catalog=True, \
                mask_frac_min=0.9, ff_corr=False, plot=False):
    
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
        
        
    cal_src_posx, cal_src_posy, pred_ciber_flux_densities, sed_modls, calmags = load_calib_catalog(inst, ifield, m_min=m_min, m_max=m_max)
        
    all_pred_fluxes, all_measured_fluxes,\
        all_var_measured_fluxes, all_bp_corr,\
            all_g1g2, all_bkg_level = [np.zeros_like(cal_src_posx) for x in range(6)]
    
    all_postage_stamps, all_postage_stamps_mask, post_bool = grab_postage_stamps(inst, cal_src_posx, cal_src_posy, flight_im, mask, dx, neighbor_cat=neighbor_cat, mask_frac_min=mask_frac_min, \
                                                                                bkg_mask=bkg_mask)
    
    # aperture photometry on postage stamps, compare with predicted fluxes
    
    for n in range(len(post_bool)):
        
        if post_bool[n]==1:

            if np.sum(all_postage_stamps[n])==0:
                continue
                
            aper_flux, aper_flux_var, bkg_level = aper_phot(all_postage_stamps[n], all_postage_stamps_mask[n], bkg_mask, mode='mean', plot=False)

            bandpass_corr = compute_bandpass_correction(ciber_filt_lam, ciber_filt_T, sed_modls[n], lam_eff)
            
            pred_flux_corr = calc_pred_flux(cbps, pred_ciber_flux_densities[n,inst-1], lam_eff*1e-6*u.m, bandpass_corr=bandpass_corr)
            
            all_bkg_level[n] = bkg_level
            all_g1g2[n] = pred_flux_corr / aper_flux
            
            all_pred_fluxes[n] = pred_flux_corr
            all_measured_fluxes[n] = aper_flux
            all_var_measured_fluxes[n] = aper_flux_var
            
            plotstr = 'g1g2='+str(np.round(all_g1g2[n], 1))+', (x,y)=('+str(int(cal_src_posx[n]))+','+str(int(cal_src_posy[n]))+')'
            plotstr += '\n'+bandstr+'='+str(np.round(calmags[n], 1))
            
            plotstr += '\nPred: '+str(np.round(all_pred_fluxes[n], 1))+', Meas: '+str(np.round(all_measured_fluxes[n], 1))

#             if all_pred_fluxes[n] > 7e4:
#                 fig = plot_map(all_postage_stamps[n]*all_postage_stamps_mask[n], title=plotstr, figsize=(5,5), return_fig=True, show=False)
                
            
            if plot:
                fig = plot_map(all_postage_stamps[n], title=plotstr, figsize=(5,5), return_fig=True, show=True)
                fig.savefig(config.ciber_basepath+'figures/calibration_post_stamps/TM'+str(inst)+'/ifield'+str(ifield)+'/calibration_post_'+str(n)+'.png', bbox_inches='tight')
                plot_map(all_postage_stamps_mask[n], title=plotstr, figsize=(5,5), show=True)
            
    # cut sources without measurement
    
    use_mask = (all_g1g2!=0)
    
    all_pred_fluxes = all_pred_fluxes[use_mask]
    all_measured_fluxes = all_measured_fluxes[use_mask]
    all_var_measured_fluxes = all_var_measured_fluxes[use_mask]
    
    all_bp_corr = all_bp_corr[use_mask]
    all_g1g2 = all_g1g2[use_mask]
    all_bkg_level = all_bkg_level[use_mask]
    
    
    aper_snr = np.abs(all_measured_fluxes)/np.sqrt(all_var_measured_fluxes)
    plt.figure()
#     plt.scatter(np.abs(all_measured_fluxes[all_measured_fluxes!=0]), aper_snr[all_measured_fluxes!=0])
    plt.scatter(np.abs(all_measured_fluxes[all_measured_fluxes!=0]), np.sqrt(all_var_measured_fluxes)[all_measured_fluxes!=0])
    
    plt.xscale('log')
    plt.xlim(1e0, 1e3)
    plt.yscale('log')
    plt.show()
    
    plt.figure()
    plt.hist(aper_snr, bins=np.logspace(-1, 2, 30))
    plt.xscale('log')
    plt.show()
    
    plt.figure()
    plt.hist(aper_snr, bins=50)
    plt.yscale('log')
    plt.show()
    
    plt.figure()
    plt.hist(all_bkg_level, bins=30)
    plt.show()


    if inst==1:
        varmax = 30
        calmin, calmax = -400, -200
    else:
        varmax = 40
        calmin, calmax = -200, 0
    
    snr_mask = (np.abs(all_bkg_level-np.median(all_bkg_level)) < np.abs(np.median(all_bkg_level)))
    
    snr_mask *= (np.sqrt(all_var_measured_fluxes) < varmax)
    snr_mask *= (aper_snr > 0.5)

    
    plt.figure()
    plt.hist(all_g1g2, bins=np.linspace(calmin, calmax, 20), label='all')
    plt.hist(all_g1g2[snr_mask], bins=np.linspace(calmin, calmax, 20), label='SNR > 2')
    plt.legend()
    plt.show()
    
    
    plt.figure(figsize=(6,4))
#     plt.errorbar(all_pred_fluxes[~snr_mask], -1.*all_measured_fluxes[~snr_mask], yerr=np.sqrt(all_var_measured_fluxes)[~snr_mask], color='r', fmt='o', markersize=3, alpha=0.3)
    
    plt.errorbar(all_pred_fluxes[snr_mask], -1.*all_measured_fluxes[snr_mask], yerr=np.sqrt(all_var_measured_fluxes)[snr_mask], color='k', fmt='o', markersize=3, alpha=0.1)
#     plt.errorbar(all_pred_fluxes, -1.*all_measured_fluxes, yerr=np.sqrt(all_var_measured_fluxes), color='k', fmt='o', markersize=3, alpha=0.1)
    
    plt.xscale('log')
    plt.yscale('log')
    if inst==1:
        plt.plot(np.linspace(1e3, 3e5, 100), np.linspace(1e3, 3e5, 100)/250, linestyle='dashed', color='r', zorder=10)
        plt.plot(np.linspace(1e3, 3e5, 100), np.linspace(1e3, 3e5, 100)/350, linestyle='dashed', color='b', zorder=10)
    elif inst==2:
        plt.plot(np.linspace(1e3, 3e5, 100), np.linspace(1e3, 3e5, 100)/80, linestyle='dashed', color='r', zorder=10)
        plt.plot(np.linspace(1e3, 3e5, 100), np.linspace(1e3, 3e5, 100)/120, linestyle='dashed', color='b', zorder=10)
        plt.plot(np.linspace(1e3, 3e5, 100), np.linspace(1e3, 3e5, 100)/100, linestyle='dashed', color='limegreen', zorder=10)

    
    plt.ylim(1e0, 1e3)
    plt.xlim(1e3, 3e5)
    plt.show()
    
    cal_src_posx = cal_src_posx[use_mask]
    cal_src_posy = cal_src_posy[use_mask]
    
    return cal_src_posx, cal_src_posy, all_pred_fluxes, all_measured_fluxes, all_var_measured_fluxes, all_bp_corr, all_g1g2, all_bkg_level
            
    

# def ciber_sb_calibration(inst, ifield_list, mask=None, mask_tail=None, mag_min_AB=8.0, mag_lim_AB=16, dx=5):
    

#     config_dict, pscb_dict, float_param_dict, fpath_dict = return_default_cbps_dicts()
#     ciber_mock_fpath = config.exthdpath+'ciber_mocks/'
    
#     cbps = CIBER_PS_pipeline()
    
#     datestr = '112022'
#     datestr_trilegal = datestr
#     fpath_dict, list_of_dirpaths, base_path, trilegal_base_path = set_up_filepaths_cbps(fpath_dict, inst, 'test', datestr,\
#                                                                                         datestr_trilegal=datestr, data_type='observed', \
#                                                                                        save_fpaths=True)
    
    
#     base_fluc_path = config.exthdpath+'ciber_fluctuation_data/'
#     tempbank_dirpath = base_fluc_path+'/TM'+str(inst)+'/subpixel_psfs/'
#     catalog_basepath = base_fluc_path+'catalogs/'
#     magkey_dict = dict({1:'j_m', 2:'h_m'})
# #     magkey_dict = dict({1:'J_Vega_predict', 2:'H_Vega_predict'})
    
#     all_median_cal_facs, all_std_cal_facs = [], []
#     all_sel_x, all_sel_y = [], []
#     all_cal_fac_sel = []
    
#     for fieldidx, ifield in enumerate(ifield_list):
        
#         if mask_tail is not None:
#             mask_fpath = fpath_dict['mask_base_path']+'/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'
#             mask = fits.open(mask_fpath)[1].data
            
# #             plot_map(mask, title='mask')
        
#         field_name = cbps.ciber_field_dict[ifield]
        
# #         twomass_cat = pd.read_csv(catalog_basepath+'mask_predict/merged_cat_2MASSbright_unWISE_PS_pred_TM'+str(inst)+'_ifield'+str(ifield)+'_080723.csv')
# #         print('twomass cat keys:', twomass_cat.keys())
        
#         twomass_cat = pd.read_csv(catalog_basepath+'2MASS/filt/2MASS_filt_rdflag_wxy_'+field_name+'_Jlt17.5.csv')

#         twomass_x = np.array(twomass_cat['x'+str(inst)])
#         twomass_y = np.array(twomass_cat['y'+str(inst)])
#         twomass_mag = np.array(twomass_cat[magkey_dict[inst]]) # magkey    
        
#         twomass_mag_AB = twomass_mag + cbps.Vega_to_AB[inst]
        
#         plt.figure()
#         plt.hist(twomass_mag_AB, bins=np.linspace(10, 20, 20))
#         plt.yscale('log')
#         plt.show()
#         twomass_magcut = (twomass_mag_AB < mag_lim_AB)*(twomass_mag_AB > mag_min_AB)

#         twomass_sim_magcut = (twomass_mag_AB > mag_min_AB)
#         twomass_x_sel = twomass_x[twomass_magcut]
#         twomass_y_sel = twomass_y[twomass_magcut]
#         twomass_mag_AB_sel = twomass_mag_AB[twomass_magcut]        
        
#         # load in flight image in ADU/fr
        
# #         flight_im = fits.open('/Users/richardfeder/Downloads/ciber_TM'+str(inst)+'_ifield'+str(ifield)+'_proc_080423.fits')[1].data
# #         flight_im /= cbps.cal_facs[inst]
        
        

#         flight_im = fits.open('data/CIBER1_dr_ytc/slope_fits/TM'+str(inst)+'/ciber_flight_TM'+str(inst)+'_ifield'+str(ifield)+'_test.fits')[1].data
#         maskInst_fpath = config.exthdpath +'ciber_fluctuation_data/TM'+str(inst)+'/masks/maskInst_102422/field'+str(ifield)+'_TM'+str(inst)+'_maskInst_102422.fits'

# #         flight_im = fits.open('data/CIBER1_dr_ytc/slope_fits/TM'+str(inst)+'/ciber_flight_TM'+str(inst)+'_ifield'+str(ifield)+'_aducorr.fits')[1].data
# #         maskInst_fpath = config.exthdpath +'ciber_fluctuation_data/TM'+str(inst)+'/masks/maskInst_080423/field'+str(ifield)+'_TM'+str(inst)+'_maskInst_080423.fits'
#         mask_inst = fits.open(maskInst_fpath)['maskInst'].data # 10/24/22
        
        
#         I_arr_full_sel = cmock.mag_2_nu_Inu(twomass_mag_AB_sel, inst-1)
#         I_arr_full = cmock.mag_2_nu_Inu(twomass_mag_AB, inst-1)
    
#         full_tracer_cat_sim = np.array([twomass_x[twomass_sim_magcut], twomass_y[twomass_sim_magcut], twomass_mag_AB[twomass_sim_magcut], I_arr_full[twomass_sim_magcut]])

#         full_tracer_cat = np.array([twomass_x_sel, twomass_y_sel, twomass_mag_AB_sel, I_arr_full_sel])
        
#         print('full tracer has shape', full_tracer_cat.shape)

#         bright_src_map = cmock.make_srcmap_temp_bank(ifield, inst, full_tracer_cat_sim.transpose(), flux_idx=-1, load_precomp_tempbank=True, \
#                                                     tempbank_dirpath=tempbank_dirpath)

        
#         if mask is not None:
            
#             mask *= (flight_im!=0)
            
#             bright_src_map_masked = bright_src_map*mask
            
#             median_bright_src_map = np.median(bright_src_map[mask==1])
            
            
#             bright_src_map_masked[mask==1] -= median_bright_src_map
            
#             flight_im_masked = flight_im*mask
            
#             median_flight_im = np.median(flight_im_masked[mask==1])

#             flight_im_masked[mask==1] -= median_flight_im
            
            
            
# #         plot_map(bright_src_map_masked, title='bright_src_map_masked')
# #         plot_map(flight_im_masked, title='flight_im_masked')
        
# #         twomass_mag_mask = (twomass_mag_AB_sel > mag_min_AB)*(twomass_mag_AB_sel < mag_lim+7.+cbps.Vega_to_AB[inst])
        
#         # remove ones near the edge
#         twomass_pos_mask = (twomass_x_sel > dx)*(twomass_x_sel +dx+1 < 1024)*(twomass_y_sel > dx)*(twomass_y_sel +dx+1 < 1024)
        
        
#         brightest_srcs = np.where(twomass_pos_mask)[0]
        
        
#         brightest_tracer_cat = full_tracer_cat[:,brightest_srcs]
        
#         all_mock_cutouts = []
#         all_obs_cutouts = []
#         all_brightest_mag = []
        
        
#         all_sum_sb_mock, all_sum_adu_obs, all_mask_fractions = [], [], []

#         sel_x, sel_y = [], []
#     #     for x in range(50):
#         counter = 0
#         for x in range(len(brightest_srcs)):



#             try:


#                 maskcutout = mask[int(brightest_tracer_cat[1,x])-dx:int(brightest_tracer_cat[1,x])+dx+1, int(brightest_tracer_cat[0,x])-dx:int(brightest_tracer_cat[0,x])+dx+1]

#                 assert maskcutout.shape[0]==maskcutout.shape[1]

#                 mask_frac = float(np.nansum(maskcutout))/float(maskcutout.shape[0]*maskcutout.shape[1])

# #                 print('mask frac = ', mask_frac)

#                 if mask_frac > 0.7:

            
#                     mockcutout = bright_src_map_masked.transpose()[int(np.floor(brightest_tracer_cat[1,x]))-dx:int(np.floor(brightest_tracer_cat[1,x]))+dx+1, int(np.floor(brightest_tracer_cat[0,x]))-dx:int(np.floor(brightest_tracer_cat[0,x]))+dx+1]

#                     mean_sb_mock_masked = np.nansum(mockcutout[maskcutout==1])/np.nansum(maskcutout)

#                     obscutout = flight_im_masked[int(np.floor(brightest_tracer_cat[1,x]))-dx:int(np.floor(brightest_tracer_cat[1,x]))+dx+1, int(np.floor(brightest_tracer_cat[0,x]))-dx:int(np.floor(brightest_tracer_cat[0,x]))+dx+1]
                    
# #                     nearby_bright_srcs = np.where((dr_gtr_src))
# #                     if len(nearby_bright_srcs)==0:
                        
    
                    
#                     which_nanmin = np.where((obscutout==np.nanmin(obscutout)))
# #                     print('which nanmin = ', which_nanmin, np.nanmin(obscutout))
                    
# #                     if np.abs(which_nanmin[0]-dx)<1.5 and np.abs(which_nanmin[1]-dx)<1.5:
#                     if np.abs(which_nanmin[0]-dx)<5.5 and np.abs(which_nanmin[1]-dx)<5.5:

# #                     center_pixel_x = brightest_tracer_cat[0,x]+ which_nanmin[0]
# #                     center_pixel_y = brightest_tracer_cat[1,x]+ which_nanmin[1]

# #                     obscutout = flight_im_masked[int(np.floor(center_pixel_y))-dx:int(np.floor(center_pixel_y))+dx, int(np.floor(center_pixel_x))-dx:int(np.floor(center_pixel_x))+dx]
#                         counter += 1

#                         sel_x.append(brightest_tracer_cat[0,x])
#                         sel_y.append(brightest_tracer_cat[1,x])

#                         mean_adu_obs_masked = np.nansum(obscutout[maskcutout==1])/np.nansum(maskcutout)

#                         all_mask_fractions.append(mask_frac)
#                         all_sum_sb_mock.append(mean_sb_mock_masked)
#                         all_sum_adu_obs.append(mean_adu_obs_masked)

#                         all_mock_cutouts.append(mockcutout)
#                         all_obs_cutouts.append(obscutout)
#                         all_brightest_mag.append(brightest_tracer_cat[2,x])


#             except:
#                 continue

                
#         all_sum_adu_obs = np.array(all_sum_adu_obs)   
        
#         all_sum_sb_mock = np.array(all_sum_sb_mock)
        
#         sel_x = np.array(sel_x)
#         sel_y = np.array(sel_y)
        
        
# #         plt.figure(figsize=(8,4))
# #         plt.subplot(1,2,1)
# #         plt.hist(all_sum_adu_obs, bins=50)
# #         plt.yscale('log')
# #         plt.title('observed ADU/fr')
        
# #         plt.subplot(1,2,2)
# #         plt.hist(all_sum_sb_mock, bins=50)
# #         plt.title('mock sb')
# #         plt.yscale('log')
        
# #         plt.show()
        
        
# #         mock_sb_mask = (all_sum_sb_mock)
# #         plt.figure()
# #         plt.scatter(all_sum_sb_mock[all_sum_sb_mock != 0], -all_sum_adu_obs[all_sum_sb_mock != 0])
# #         plt.xlim(1e1, 5e3)

# #         plt.ylim(1e-1, 5e1)

# #         plt.yscale('log')
# #         plt.xscale('log')
# #         plt.show()
        
#         cal_fac = all_sum_sb_mock/all_sum_adu_obs

# #         plt.figure()
# #         plt.scatter(all_mask_fractions, cal_fac, alpha=0.2)
# #         plt.xlabel('mask fraction')
# #         plt.ylabel('cal fac')
# #         plt.ylim(-1000, 1000)
# #         plt.show()
        
#         sb_mask = (all_sum_sb_mock > 300)*(all_sum_sb_mock < 5000)*(all_sum_adu_obs < 0)
        
#         all_brightest_mag = np.array(all_brightest_mag)
        
#         all_brightest_mag_sel = all_brightest_mag[sb_mask]
        
# #         plt.figure()
# #         plt.scatter(all_sum_sb_mock[sb_mask], cal_fac[sb_mask], c=np.array(all_mask_fractions)[sb_mask])
# #         plt.colorbar()
# #         plt.xlabel('mean mock SB')
# #         plt.ylabel('Cal fac')
# #         plt.ylim(-1000, 1000)
# #         plt.show()

        
#         cal_fac_sel = cal_fac[sb_mask]
        
        

        
#         clipped_cal_fac, lo, hi = scipy.stats.sigmaclip(cal_fac_sel, low=5.0, high=5.0)
        
#         sel_x = sel_x[sb_mask]
#         sel_y = sel_y[sb_mask]
        
#         cal_fac_mask = (cal_fac_sel > lo)*(cal_fac_sel < hi)
#         all_sel_x.append(sel_x[cal_fac_mask])
#         all_sel_y.append(sel_y[cal_fac_mask])
        
#         all_cal_fac_sel.append(cal_fac_sel[cal_fac_mask])

        
#         sig68 = np.std(clipped_cal_fac)/np.sqrt(len(clipped_cal_fac))
        
# #         sig68 = 0.5*(np.percentile(clipped_cal_fac, 84)-np.percentile(clipped_cal_fac, 16))/np.sqrt(len(clipped_cal_fac))
        
# #         sigma_clip_mask
        
        
        
# #         plt.figure()
# #         plt.title('Cal = '+str(np.round(np.median(clipped_cal_fac), 1))+'$\\pm$'+str(np.round(sig68, 1)))
# #         plt.hist(clipped_cal_fac, bins=30)
# #         plt.show()
        
# #         plt.figure()
# #         plt.scatter(all_brightest_mag_sel, cal_fac_sel)
# #         plt.xlim(10, 20)
# #         plt.ylim(-500, 0)
# #         plt.show()
        
#         median_cal_fac = np.median(clipped_cal_fac)
#         all_std_cal_facs.append(sig68)
        
#         all_median_cal_facs.append(median_cal_fac)

        
        

#     return all_median_cal_facs, all_std_cal_facs, all_sel_x, all_sel_y, all_cal_fac_sel
        