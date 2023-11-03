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

def ciber_sb_calibration(inst, ifield_list, mask=None, mask_tail=None, mag_min_AB=8.0, mag_lim_AB=16, dx=5):
    

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
#     magkey_dict = dict({1:'J_Vega_predict', 2:'H_Vega_predict'})
    
    all_median_cal_facs, all_std_cal_facs = [], []
    all_sel_x, all_sel_y = [], []
    all_cal_fac_sel = []
    
    for fieldidx, ifield in enumerate(ifield_list):
        
        if mask_tail is not None:
            mask_fpath = fpath_dict['mask_base_path']+'/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'
            mask = fits.open(mask_fpath)[1].data
            
#             plot_map(mask, title='mask')
        
        field_name = cbps.ciber_field_dict[ifield]
        
#         twomass_cat = pd.read_csv(catalog_basepath+'mask_predict/merged_cat_2MASSbright_unWISE_PS_pred_TM'+str(inst)+'_ifield'+str(ifield)+'_080723.csv')
#         print('twomass cat keys:', twomass_cat.keys())
        
        twomass_cat = pd.read_csv(catalog_basepath+'2MASS/filt/2MASS_filt_rdflag_wxy_'+field_name+'_Jlt17.5.csv')

        twomass_x = np.array(twomass_cat['x'+str(inst)])
        twomass_y = np.array(twomass_cat['y'+str(inst)])
        twomass_mag = np.array(twomass_cat[magkey_dict[inst]]) # magkey    
        
        twomass_mag_AB = twomass_mag + cbps.Vega_to_AB[inst]
        
        plt.figure()
        plt.hist(twomass_mag_AB, bins=np.linspace(10, 20, 20))
        plt.yscale('log')
        plt.show()
        twomass_magcut = (twomass_mag_AB < mag_lim_AB)*(twomass_mag_AB > mag_min_AB)

        twomass_sim_magcut = (twomass_mag_AB > mag_min_AB)
        twomass_x_sel = twomass_x[twomass_magcut]
        twomass_y_sel = twomass_y[twomass_magcut]
        twomass_mag_AB_sel = twomass_mag_AB[twomass_magcut]        
        
        # load in flight image in ADU/fr
        
#         flight_im = fits.open('/Users/richardfeder/Downloads/ciber_TM'+str(inst)+'_ifield'+str(ifield)+'_proc_080423.fits')[1].data
#         flight_im /= cbps.cal_facs[inst]
        
        

        flight_im = fits.open('data/CIBER1_dr_ytc/slope_fits/TM'+str(inst)+'/ciber_flight_TM'+str(inst)+'_ifield'+str(ifield)+'_test.fits')[1].data
        maskInst_fpath = config.exthdpath +'ciber_fluctuation_data/TM'+str(inst)+'/masks/maskInst_102422/field'+str(ifield)+'_TM'+str(inst)+'_maskInst_102422.fits'

#         flight_im = fits.open('data/CIBER1_dr_ytc/slope_fits/TM'+str(inst)+'/ciber_flight_TM'+str(inst)+'_ifield'+str(ifield)+'_aducorr.fits')[1].data
#         maskInst_fpath = config.exthdpath +'ciber_fluctuation_data/TM'+str(inst)+'/masks/maskInst_080423/field'+str(ifield)+'_TM'+str(inst)+'_maskInst_080423.fits'
        mask_inst = fits.open(maskInst_fpath)['maskInst'].data # 10/24/22
        
        
        I_arr_full_sel = cmock.mag_2_nu_Inu(twomass_mag_AB_sel, inst-1)
        I_arr_full = cmock.mag_2_nu_Inu(twomass_mag_AB, inst-1)
    
        full_tracer_cat_sim = np.array([twomass_x[twomass_sim_magcut], twomass_y[twomass_sim_magcut], twomass_mag_AB[twomass_sim_magcut], I_arr_full[twomass_sim_magcut]])

        full_tracer_cat = np.array([twomass_x_sel, twomass_y_sel, twomass_mag_AB_sel, I_arr_full_sel])
        
        print('full tracer has shape', full_tracer_cat.shape)

        bright_src_map = cmock.make_srcmap_temp_bank(ifield, inst, full_tracer_cat_sim.transpose(), flux_idx=-1, load_precomp_tempbank=True, \
                                                    tempbank_dirpath=tempbank_dirpath)

        
        if mask is not None:
            
            mask *= (flight_im!=0)
            
            bright_src_map_masked = bright_src_map*mask
            
            median_bright_src_map = np.median(bright_src_map[mask==1])
            
            
            bright_src_map_masked[mask==1] -= median_bright_src_map
            
            flight_im_masked = flight_im*mask
            
            median_flight_im = np.median(flight_im_masked[mask==1])

            flight_im_masked[mask==1] -= median_flight_im
            
            
            
#         plot_map(bright_src_map_masked, title='bright_src_map_masked')
#         plot_map(flight_im_masked, title='flight_im_masked')
        
#         twomass_mag_mask = (twomass_mag_AB_sel > mag_min_AB)*(twomass_mag_AB_sel < mag_lim+7.+cbps.Vega_to_AB[inst])
        
        # remove ones near the edge
        twomass_pos_mask = (twomass_x_sel > dx)*(twomass_x_sel +dx+1 < 1024)*(twomass_y_sel > dx)*(twomass_y_sel +dx+1 < 1024)
        
        
        brightest_srcs = np.where(twomass_pos_mask)[0]
        
        
        brightest_tracer_cat = full_tracer_cat[:,brightest_srcs]
        
        all_mock_cutouts = []
        all_obs_cutouts = []
        all_brightest_mag = []
        
        
        all_sum_sb_mock, all_sum_adu_obs, all_mask_fractions = [], [], []

        sel_x, sel_y = [], []
    #     for x in range(50):
        counter = 0
        for x in range(len(brightest_srcs)):



            try:


                maskcutout = mask[int(brightest_tracer_cat[1,x])-dx:int(brightest_tracer_cat[1,x])+dx+1, int(brightest_tracer_cat[0,x])-dx:int(brightest_tracer_cat[0,x])+dx+1]

                assert maskcutout.shape[0]==maskcutout.shape[1]

                mask_frac = float(np.nansum(maskcutout))/float(maskcutout.shape[0]*maskcutout.shape[1])

#                 print('mask frac = ', mask_frac)

                if mask_frac > 0.7:

            
                    mockcutout = bright_src_map_masked.transpose()[int(np.floor(brightest_tracer_cat[1,x]))-dx:int(np.floor(brightest_tracer_cat[1,x]))+dx+1, int(np.floor(brightest_tracer_cat[0,x]))-dx:int(np.floor(brightest_tracer_cat[0,x]))+dx+1]

                    mean_sb_mock_masked = np.nansum(mockcutout[maskcutout==1])/np.nansum(maskcutout)

                    obscutout = flight_im_masked[int(np.floor(brightest_tracer_cat[1,x]))-dx:int(np.floor(brightest_tracer_cat[1,x]))+dx+1, int(np.floor(brightest_tracer_cat[0,x]))-dx:int(np.floor(brightest_tracer_cat[0,x]))+dx+1]
                    
#                     nearby_bright_srcs = np.where((dr_gtr_src))
#                     if len(nearby_bright_srcs)==0:
                        
    
                    
                    which_nanmin = np.where((obscutout==np.nanmin(obscutout)))
#                     print('which nanmin = ', which_nanmin, np.nanmin(obscutout))
                    
#                     if np.abs(which_nanmin[0]-dx)<1.5 and np.abs(which_nanmin[1]-dx)<1.5:
                    if np.abs(which_nanmin[0]-dx)<5.5 and np.abs(which_nanmin[1]-dx)<5.5:

#                     center_pixel_x = brightest_tracer_cat[0,x]+ which_nanmin[0]
#                     center_pixel_y = brightest_tracer_cat[1,x]+ which_nanmin[1]

#                     obscutout = flight_im_masked[int(np.floor(center_pixel_y))-dx:int(np.floor(center_pixel_y))+dx, int(np.floor(center_pixel_x))-dx:int(np.floor(center_pixel_x))+dx]
                        counter += 1

                        sel_x.append(brightest_tracer_cat[0,x])
                        sel_y.append(brightest_tracer_cat[1,x])

                        mean_adu_obs_masked = np.nansum(obscutout[maskcutout==1])/np.nansum(maskcutout)

                        all_mask_fractions.append(mask_frac)
                        all_sum_sb_mock.append(mean_sb_mock_masked)
                        all_sum_adu_obs.append(mean_adu_obs_masked)

                        all_mock_cutouts.append(mockcutout)
                        all_obs_cutouts.append(obscutout)
                        all_brightest_mag.append(brightest_tracer_cat[2,x])


            except:
                continue

                
        all_sum_adu_obs = np.array(all_sum_adu_obs)   
        
        all_sum_sb_mock = np.array(all_sum_sb_mock)
        
        sel_x = np.array(sel_x)
        sel_y = np.array(sel_y)
        
        
#         plt.figure(figsize=(8,4))
#         plt.subplot(1,2,1)
#         plt.hist(all_sum_adu_obs, bins=50)
#         plt.yscale('log')
#         plt.title('observed ADU/fr')
        
#         plt.subplot(1,2,2)
#         plt.hist(all_sum_sb_mock, bins=50)
#         plt.title('mock sb')
#         plt.yscale('log')
        
#         plt.show()
        
        
#         mock_sb_mask = (all_sum_sb_mock)
#         plt.figure()
#         plt.scatter(all_sum_sb_mock[all_sum_sb_mock != 0], -all_sum_adu_obs[all_sum_sb_mock != 0])
#         plt.xlim(1e1, 5e3)

#         plt.ylim(1e-1, 5e1)

#         plt.yscale('log')
#         plt.xscale('log')
#         plt.show()
        
        cal_fac = all_sum_sb_mock/all_sum_adu_obs

#         plt.figure()
#         plt.scatter(all_mask_fractions, cal_fac, alpha=0.2)
#         plt.xlabel('mask fraction')
#         plt.ylabel('cal fac')
#         plt.ylim(-1000, 1000)
#         plt.show()
        
        sb_mask = (all_sum_sb_mock > 300)*(all_sum_sb_mock < 5000)*(all_sum_adu_obs < 0)
        
        all_brightest_mag = np.array(all_brightest_mag)
        
        all_brightest_mag_sel = all_brightest_mag[sb_mask]
        
#         plt.figure()
#         plt.scatter(all_sum_sb_mock[sb_mask], cal_fac[sb_mask], c=np.array(all_mask_fractions)[sb_mask])
#         plt.colorbar()
#         plt.xlabel('mean mock SB')
#         plt.ylabel('Cal fac')
#         plt.ylim(-1000, 1000)
#         plt.show()

        
        cal_fac_sel = cal_fac[sb_mask]
        
        

        
        clipped_cal_fac, lo, hi = scipy.stats.sigmaclip(cal_fac_sel, low=5.0, high=5.0)
        
        sel_x = sel_x[sb_mask]
        sel_y = sel_y[sb_mask]
        
        cal_fac_mask = (cal_fac_sel > lo)*(cal_fac_sel < hi)
        all_sel_x.append(sel_x[cal_fac_mask])
        all_sel_y.append(sel_y[cal_fac_mask])
        
        all_cal_fac_sel.append(cal_fac_sel[cal_fac_mask])

        
        sig68 = np.std(clipped_cal_fac)/np.sqrt(len(clipped_cal_fac))
        
#         sig68 = 0.5*(np.percentile(clipped_cal_fac, 84)-np.percentile(clipped_cal_fac, 16))/np.sqrt(len(clipped_cal_fac))
        
#         sigma_clip_mask
        
        
        
#         plt.figure()
#         plt.title('Cal = '+str(np.round(np.median(clipped_cal_fac), 1))+'$\\pm$'+str(np.round(sig68, 1)))
#         plt.hist(clipped_cal_fac, bins=30)
#         plt.show()
        
#         plt.figure()
#         plt.scatter(all_brightest_mag_sel, cal_fac_sel)
#         plt.xlim(10, 20)
#         plt.ylim(-500, 0)
#         plt.show()
        
        median_cal_fac = np.median(clipped_cal_fac)
        all_std_cal_facs.append(sig68)
        
        all_median_cal_facs.append(median_cal_fac)

        
        

    return all_median_cal_facs, all_std_cal_facs, all_sel_x, all_sel_y, all_cal_fac_sel
        