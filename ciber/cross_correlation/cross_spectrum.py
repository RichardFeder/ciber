import numpy as np
from astropy.io import fits
from plotting_fns import plot_map
from reproject import reproject_interp

import config
from ciber_powerspec_pipeline import *
from ciber_mocks import *
from mock_galaxy_catalogs import *
from lognormal_counts import *
# from ciber_data_helpers import *
from helgason import *
from ps_pipeline_go import *
from noise_model import *
# from cross_spectrum_analysis import *
from mkk_parallel import *
from mkk_diagnostics import *
from flat_field_est import *
from plotting_fns import *
from powerspec_utils import *
from astropy.coordinates import SkyCoord
import healpy as hp
import scipy.io


def calc_reproj_tl(inst, ifield_list, mask_tail=None, cross_inst=2, order=None, conserve_flux=False, reproj_mode='observed', \
                    nsims=100, ifield_mock=4, nsetprint=10, plot=False, use_cib_sims=False, n_cib_sims=10):
    cbps = CIBER_PS_pipeline()

    # load masked maps before and after regridding
    
    if reproj_mode=='mock':
        tl_all = np.zeros((nsims, cbps.n_ps_bin))
        tl_regrid = np.zeros((cbps.n_ps_bin))
    else:
        tl_regrid = np.zeros((len(ifield_list), cbps.n_ps_bin))
    

    if reproj_mode=='mock':
        for setidx in range(nsims):

            if use_cib_sims:

                cib_mock_basepath = config.ciber_basepath+'data/ciber_mocks/112022/TM1/cib_realiz/cib_with_tracer_with_dpoint_5field_set'+str(setidx%n_cib_sims)+'_112022_TM1.fits'
                ciber_im = fits.open(cib_mock_basepath)['CIB_J_4'].data
                if setidx==0:
                    plot_map(ciber_im, title='ciber im')
                # ciber_im = None
            else:
                _, _, ciber_im = generate_diffuse_realization(cbps.dimx, cbps.dimy, power_law_idx=0.0, scale_fac=3e8)
            lb, cl_orig, clerr_orig = get_power_spec(ciber_im-np.mean(ciber_im), lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
            
            if setidx%nsetprint==0:
                print('setidx = ', setidx)

            if setidx==0:
                ciber_im_regrid, ciber_fp_regrid, astr_map0_hdrs, astr_map1_hdrs = regrid_arrays_by_quadrant(ciber_im, ifield_mock, inst0=inst, inst1=cross_inst, \
                                                                                                            astr_dir=config.ciber_basepath+'data/', plot=False,\
                                                                                                        order=order, conserve_flux=conserve_flux, return_hdrs=True)
            else:
                ciber_im_regrid, ciber_fp_regrid = regrid_arrays_by_quadrant(ciber_im, ifield_mock, inst0=inst, inst1=cross_inst, \
                                                                                                            astr_dir=config.ciber_basepath+'data/', plot=False,\
                                                                                                        order=order, conserve_flux=conserve_flux, astr_map0_hdrs=astr_map0_hdrs, \
                                                                                                        astr_map1_hdrs=astr_map1_hdrs)

            ciber_im_regrid[np.isnan(ciber_im_regrid)] = 0.
            ciber_im_regrid[np.isinf(ciber_im_regrid)] = 0.
            ciber_im_regrid[ciber_im_regrid!=0] -= np.mean(ciber_im_regrid[ciber_im_regrid!=0])
            lb, cl_regrid, _ = get_power_spec(ciber_im_regrid, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)

            tl_all[setidx] = cl_regrid/cl_orig

        tl_regrid = np.mean(tl_all, axis=0)

    elif reproj_mode=='observed':

        proc_regrid_basepath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'_TM'+str(cross_inst)+'_cross/proc_regrid/'
    
        proc_regrid_basepath += mask_tail+'/'

        for fieldidx, ifield in enumerate(ifield_list):
            
            proc_fpath = proc_regrid_basepath+'proc_regrid_TM'+str(cross_inst)+'_to_TM'+str(inst)+'_ifield'+str(ifield)+'_'+mask_tail
            
            if order is not None:
                proc_fpath += '_order='+str(order)
            if conserve_flux:
                proc_fpath += '_conserve_flux'
            proc_fpath +='.fits'
            
            mask_base_path = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/masks/'
            mask_fpath = mask_base_path+'/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'
            mask_native = fits.open(mask_fpath)[1].data
            
            proc = fits.open(proc_fpath)
            masked_ciber_orig = proc['proc_orig_'+str(ifield)].data
            masked_ciber_regrid = proc['proc_regrid_'+str(ifield)].data
            
            # ciber_regrid_to_orig, _ = regrid_arrays_by_quadrant(masked_ciber_regrid, ifield, inst0=cross_inst, inst1=inst, \
            #                                              plot=False, astr_dir=config.ciber_basepath+'data/', order=1, conserve_flux=conserve_flux)
            # plot_map(mask_native, title='mask original')
            # plot_map(ciber_regrid_to_orig, title='regrid')
            # plot_map(masked_ciber_orig, title='before regridding')
            # plot_map(ciber_regrid_to_orig-masked_ciber_orig)

            plot_map(masked_ciber_orig, title='orig ifield '+str(ifield))
            plot_map(masked_ciber_regrid, title='orig ifield '+str(ifield))

            lb, cl_orig, _ = get_power_spec(masked_ciber_orig, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
            lb, cl_regrid_tm2_to_tm1, _ = get_power_spec(masked_ciber_regrid, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
            # lb, cl_regrid_backto_tm2, _ = get_power_spec(ciber_regrid_to_orig, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
                    
            tl_regrid[fieldidx] = cl_regrid_tm2_to_tm1/cl_orig
        
        
    if plot:
        plt.figure()

        if reproj_mode=='observed':
            for fieldidx, ifield in enumerate(ifield_list):
                plt.plot(lb, tl_regrid[fieldidx], label=cbps.ciber_field_dict[ifield])
                
            plt.plot(lb, np.mean(tl_regrid, axis=0), color='k', label='Field average')
        else:
            plt.plot(lb, tl_regrid, color='k', label='Sims')
        # plt.plot(lb, np.mean(tl_regrid_orig, axis=0), color='r', label='back to orig')
            
        plt.xscale('log')
        plt.ylim(-0.2, 1.1)
        plt.legend(loc=3)
        plt.show()
    
    return lb, tl_regrid


def calculate_transfer_function_regridding(ifield_list, inst_orig=2, inst_map=1, include_dgl=True, nsims=10):
    
    cbps = CIBER_PS_pipeline()

#     if flight_dat_base_path is None:
#         flight_dat_base_path=config.exthdpath+'noise_model_validation_data/'
    
    # load original maps without regridding
    
    orig_masks, regrid_masks = [], []
    t_ell_all = np.zeros((len(ifield_list), nsims, cbps.n_ps_bin))
    t_ell_avs = np.zeros((len(ifield_list), cbps.n_ps_bin))

    cl_pivot_fac_gen = grab_cl_pivot_fac(4, inst_orig, dimx=cbps.dimx, dimy=cbps.dimy)

    for fieldidx, ifield in enumerate(ifield_list):

        # load original maps without regridding

        for setidx in range(nsims):
            
            _, _, ciber_im = generate_diffuse_realization(cbps.dimx, cbps.dimy, power_law_idx=0.0, scale_fac=3e8)
            
            lb, cl_orig, clerr_orig = get_power_spec(ciber_im-np.mean(ciber_im), lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
            
            if include_dgl:
                dgl_realization = cbps.generate_custom_sky_clustering(inst_orig, dgl_scale_fac=50, gen_ifield=5, cl_pivot_fac_gen=cl_pivot_fac_gen)
                ciber_im += dgl_realization 
            
            ciber_im_regrid, ciber_fp_regrid = regrid_arrays_by_quadrant(ciber_im, ifield, inst0=1, inst1=2, astr_dir='../../ciber/data/', plot=False)
            
#             plot_map(ciber_im_regrid, title='regridded clustering realization')
            ciber_im_regrid[np.isnan(ciber_im_regrid)] = 0.
            ciber_im_regrid[np.isinf(ciber_im_regrid)] = 0.
            
            ciber_im_regrid[ciber_im_regrid!=0] -= np.mean(ciber_im_regrid[ciber_im_regrid!=0])
            
            lb, cl_regrid, clerr_regrid = get_power_spec(ciber_im_regrid, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
            t_ell_indiv =  cl_regrid/cl_orig
            if setidx < 2:
                plot_map(ciber_fp_regrid, title='footprint')
                plot_map(gaussian_filter(ciber_im, sigma=20), title='clustering realization')
                plot_map(gaussian_filter(ciber_im_regrid, sigma=20), title='regridded clustering realization')
                print('t_ell indiv:', t_ell_indiv)
            t_ell_all[fieldidx, setidx, :] =t_ell_indiv
            
        t_ell_avs[fieldidx] = np.mean(t_ell_all, axis=0)
        
    print('t_ell_avs:', t_ell_avs)
    
    plt.figure()
    for fieldidx, ifield in enumerate(ifield_list):
        plt.plot(lb, t_ell_avs[fieldidx])
    plt.xscale('log')
    plt.ylim(-0.05, 1.05)
    
    plt.show()
    
    return t_ell_avs
      

def convert_MJysr_to_nWm2sr(lam_micron):
    
    lam_angstrom = lam_micron*1e4
    print(lam_angstrom)
    c = 2.9979e18 #A/s

    fac = 1e6 # to get to Jy from MJy

    # I_nu to I_lambda
    fac *= 1e-26
    fac *= c/(lam_angstrom*lam_angstrom)
    
    fac *= 1e9 # W m-2 sr-1 to nW m-2 sr-1
        
    return fac


def beam_correction_gaussian(lb, theta_fwhm, unit='arcmin'):

    ''' Used for DGL cross spectrum correction '''
    if unit=='arcmin':
        theta_fwhm_rad = theta_fwhm*np.pi/(60*180)
    elif unit=='arcsec':
        theta_fwhm_rad = theta_fwhm*np.pi/(3600*180)

    sigma_fwhm = theta_fwhm_rad/np.sqrt(8*np.log(2))
    
    print(theta_fwhm_rad, sigma_fwhm)
    return np.exp(-(lb*sigma_fwhm)**2/2)


def regrid_iris_ciber_science_fields(ifield_list=[4,5,6,7,8], inst=1, tail_name=None, plot=False, \
            save_fpath=config.exthdpath+'ciber_fluctuation_data/'):
    
    ciber_field_dict = dict({4:'elat10', 5:'elat30',6:'BootesB', 7:'BootesA', 8:'SWIRE'})

    save_paths = []
    for fieldidx, ifield in enumerate(ifield_list):
        print('Loading ifield', ifield)
        fieldname = ciber_field_dict[ifield]
        print(fieldname)
        irismap = regrid_iris_by_quadrant(fieldname, inst=inst)

        # make fits file saving data
    
        hduim = fits.ImageHDU(irismap, name='TM2_regrid')
        
        hdup = fits.PrimaryHDU()
        hdup.header['ifield'] = ifield
        hdup.header['dat_type'] = 'iris_interp'
        hdul = fits.HDUList([hdup, hduim])
        save_fpath_full = save_fpath+'TM'+str(inst)+'/iris_regrid/iris_regrid_ifield'+str(ifield)+'_TM'+str(inst)
        if tail_name is not None:
            save_fpath_full += '_'+tail_name
        hdul.writeto(save_fpath_full+'.fits', overwrite=True)
        
        save_paths.append(save_fpath_full+'.fits')
        
    return save_paths


def regrid_iris_by_quadrant(fieldname, inst=1, quad_list=['A', 'B', 'C', 'D'], \
                             xoff=[0,0,512,512], yoff=[0,512,0,512], astr_dir='../data/astroutputs/', \
                             plot=True, dimx=1024, dimy=1024):
    
    ''' 
    Used for regridding maps from IRIS to CIBER. For the CIBER1 imagers the 
    astrometric solution is computed for each quadrant separately, so this function iterates
    through the quadrants when constructing the full regridded images. 
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    
    iris_regrid = np.zeros((dimx, dimy))    
    astr_hdrs = [fits.open(astr_dir+'inst'+str(inst)+'/'+fieldname+'_'+quad+'_astr.fits')[0].header for quad in quad_list]

    # loop over quadrants of first imager
    
    for iquad, quad in enumerate(quad_list):
        
        arrays, footprints = [], []
        
        iris_interp = mosaic(astr_hdrs[iquad])
        
        iris_regrid[yoff[iquad]:yoff[iquad]+512, xoff[iquad]:xoff[iquad]+512] = iris_interp
        if plot:
            plot_map(iris_interp, title='IRIS map interpolated to CIBER, quadrant '+quad)
    
    if plot:
        plot_map(iris_regrid, title='IRIS map interpolated to CIBER')

    return iris_regrid


def process_ciber_cross_spectra(maglim_J_list, inst=1, cross_inst=2, datestr='111323', ifield_list=[4, 5, 6, 7, 8], \
                               include_legends = [True, False], run_names=None, flatidx = 9, save=False, plot_cl_errs=False, \
                               textxpos=300, xlim=[200, 1e5], include_mirocha_model=False, include_isl_pred=False, include_c15_pred=True, datestr_pred='091624', tailstr='test', \
                               bbox_to_anchor=[1.0, 1.3], ncol=3, order=None, include_swire_pred=False, **kwargs):
    cbps = CIBER_PS_pipeline()
    lb = cbps.Mkk_obj.midbin_ell
    
    fig_list = []

    
    snpred = np.load(config.ciber_basepath+'data/cl_predictions/snpred_color_corr_vs_mag_JH_cosmos15_nuInu_union_magcut.npz')
    all_pv_JH, mmin_J_pv = snpred['all_pv_JH'], snpred['mmin_J_list']
    
    for m, maglim_J in enumerate(maglim_J_list):
        maglim_H = maglim_J - 0.5
        
        if run_names is None:
            run_name = 'ciber_cross_ciber_Jlt'+str(maglim_J)+'_Hlt'+str(maglim_H)+'_052124_interporder2_fcsub_order2'
        else:
            run_name = run_names[m]
            
        all_nl1d_err_weighted, all_cl1d_obs, all_dcl1d_obs  = [[] for x in range(3)]

        cross_cl_file = np.load(config.ciber_basepath+'data/input_recovered_ps/'+datestr+'/TM1/'+run_name+'/input_recovered_ps_estFF_simidx0.npz')

        # cl1d_obs = np.mean(cross_cl_file['recovered_ps_est_nofluc'], axis=0)

        mean_cl1d_obs = np.mean(cross_cl_file['recovered_ps_est_nofluc'], axis=0)


        for fieldidx, ifield in enumerate(ifield_list):
            cl1d_obs = cross_cl_file['recovered_ps_est_nofluc'][fieldidx]
            dcl1d_obs = cross_cl_file['recovered_dcl'][fieldidx]

            frac_knox_errors = return_frac_knox_error(lb, cbps.Mkk_obj.delta_ell)
            knox_err = frac_knox_errors*mean_cl1d_obs

            dcl1d_obs = np.sqrt(dcl1d_obs**2+knox_err**2)
            all_cl1d_obs.append(cl1d_obs)
            all_dcl1d_obs.append(dcl1d_obs)

            noisemodl_run_name = 'ciber_cross_ciber_Jlt'+str(maglim_J)+'_Hlt'+str(maglim_H)+'_020724'
            nl_file = np.load(config.ciber_basepath+'data/noise_models_sim/111323/TM1_TM2_cross/'+noisemodl_run_name+'/noise_bias_fieldidx'+str(fieldidx)+'.npz')
            nl2d, fourier_weights = nl_file['mean_cl2d_nofluc'], nl_file['fourier_weights_nofluc']

            l2d = get_l2d(cbps.dimx, cbps.dimy, cbps.pixsize)
            lb, nl1d_weighted, nl1d_err_weighted = azim_average_cl2d(nl2d*cbps.cal_facs[inst]*cbps.cal_facs[cross_inst], l2d, weights=None, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)

            all_nl1d_err_weighted.append(np.sqrt(dcl1d_obs**2+nl1d_err_weighted**2))

        all_cl1d_obs = np.array(all_cl1d_obs)
        all_dcl1d_obs = np.array(all_dcl1d_obs)

        mean_cl_cross = np.mean(all_cl1d_obs, axis=0)
        std_cl_cross = np.std(all_cl1d_obs, axis=0)/np.sqrt(5)
        
        
        field_weights = 1./all_dcl1d_obs**2
    
        for n in range(field_weights.shape[1]):
            field_weights[:,n] /= np.sum(field_weights[:,n])

        for x in range(flatidx):
            field_weights[:,x] = np.mean(field_weights[:, -10:], axis=1)
            
        if maglim_J < 16.0:
            ylim = None 
        else:
            ylim = [1e-2, 1e4]
        #     pv_cross = None
        # else:
        which_pv = np.where((mmin_J_pv==maglim_J))[0][0]
        pv_cross = all_pv_JH[which_pv]

        if not include_c15_pred:
            pv_cross = None
            
        f, weighted_cross_average_cl, weighted_cross_average_dcl = plot_ciber_x_ciber_ps(cbps, ifield_list, lb, all_cl1d_obs, all_dcl1d_obs, field_weights, maglim_J, \
                                                                            plot_cl_errs=plot_cl_errs, startidx=2, snpred_cross=pv_cross, flatidx=0, ylim=ylim, \
                                                                                    xlim=[250, 1e5], textxpos=textxpos, include_legend=True, \
                                                                                    include_dgl_ul=True, markersize_alpha=0.4, include_mirocha_model=include_mirocha_model, \
                                                                                    include_isl_pred=include_isl_pred, datestr=datestr_pred, tailstr=tailstr, \
                                                                                    bbox_to_anchor=bbox_to_anchor, ncol=ncol, order=order, include_swire_pred=include_swire_pred)


        fig_list.append(f)
        
        if save:
            cl_fpath = save_weighted_cl_file(lb, inst, run_name, all_cl1d_obs, all_dcl1d_obs, weighted_cross_average_cl, weighted_cross_average_dcl, None, cross_inst=cross_inst)
            print('Saved to ', cl_fpath)
            
    return fig_list


def process_ciber_maps_for_cross_wrapper(mag_lim_J_list, mask_tail_list=None, inst=2, regrid_to_inst=1, ifield_list=[4, 5, 6, 7, 8], \
                                        astr_dir = config.ciber_basepath+'data/', niter=1, nitermax=10, datestr='111323', \
                                        ff_min_TM2 = 0.6, ff_max_TM2 = 1.5, save=True, regrid_order=1, conserve_flux=False, \
                                        fc_sub=True, fc_sub_quad_offset=True, fc_sub_n_terms=2, grad_sub=True, sigma_clip_ffest=5, sigma_clip_dict=None, \
                                        use_proc_sigclip=False):

    cbps = CIBER_PS_pipeline()
    regrid_masked_proc_list, regrid_fpaths_list = [], []

    if sigma_clip_dict is None:
        sigma_clip_dict = dict({11.0:None, 12.0:None, 13.0:None, 14.0:30, 15.0:10, 16.0:10.0, 16.5:5, 17.0:4, \
               17.5:4, 18.0:4, 18.5:4})

    
    for m, mag_lim_J in enumerate(mag_lim_J_list):

        if use_proc_sigclip:

            sigma_clip = sigma_clip_dict[mag_lim_J]
        else:
            sigma_clip = None 

        mag_lim_H = mag_lim_J-0.5
        
        if mask_tail_list is not None:
            mask_tail = mask_tail_list[m]
        else:
            mask_tail = 'Jlim_Vega_'+str(mag_lim_J)+'_Hlim_Vega_'+str(mag_lim_H)+'_ukdebias_111323'

        mag_lim_J_ffest = max(mag_lim_J, 17.5)
        mag_lim_H_ffest = mag_lim_J_ffest - 0.5

        mask_tail_ffest = 'Jlim_Vega_'+str(mag_lim_J_ffest)+'_Hlim_Vega_'+str(mag_lim_H_ffest)+'_ukdebias_111323'
        
        regrid_masked_proc, regrid_fpaths = proc_cibermap_regrid(cbps, inst, regrid_to_inst, mask_tail, mask_tail_ffest=mask_tail_ffest, ifield_list=ifield_list, datestr=datestr, \
                                niter=niter, nitermax=nitermax, sig=sigma_clip, sig_ffest=sigma_clip_ffest, ff_min=ff_min_TM2, ff_max=ff_max_TM2, astr_dir=astr_dir, \
                                                     save=save, order=regrid_order, conserve_flux=conserve_flux,\
                                                      fc_sub=fc_sub, fc_sub_quad_offset=fc_sub_quad_offset, fc_sub_n_terms=fc_sub_n_terms, grad_sub=grad_sub)

        regrid_masked_proc_list.append(regrid_masked_proc)
        regrid_fpaths_list.append(regrid_fpaths)
        
    return regrid_masked_proc_list, regrid_fpaths_list

def proc_cibermap_regrid(cbps, inst, regrid_to_inst, mask_tail, ifield_list=[4, 5, 6, 7, 8], datestr='112022', \
                        niter=5, nitermax=1, sig=None, sig_ffest=5, ff_min=0.5, ff_max=1.5, astr_dir='../../ciber/data/', \
                        save=True, mask_tail_ffest=None, order=0, conserve_flux=False, fc_sub=False, \
                        fc_sub_quad_offset=False, fc_sub_n_terms=2, grad_sub=True):
    
    config_dict, pscb_dict, float_param_dict, fpath_dict = return_default_cbps_dicts()

    fpath_dict, list_of_dirpaths, base_path, trilegal_base_path = set_up_filepaths_cbps(fpath_dict, inst, 'test', datestr,\
                                                                                    datestr_trilegal=datestr, data_type='observed', \
                                                                                   save_fpaths=True)
    
    
    bandstr_dict = dict({1:'J',2:'H'})
    band = bandstr_dict[inst]
    observed_ims, masks = [np.zeros((len(ifield_list), cbps.dimx, cbps.dimy)) for x in range(2)]
    processed_ims = np.zeros_like(observed_ims)
    ff_estimates = np.zeros_like(observed_ims)
    
    dc_template = cbps.load_dark_current_template(inst, verbose=True, inplace=False)

    masks_ffest = None
    if mask_tail_ffest is not None:
        masks_ffest = np.zeros((len(ifield_list), cbps.dimx, cbps.dimy))
    
    for fieldidx, ifield in enumerate(ifield_list):
        cbps.load_flight_image(ifield, inst, verbose=True, ytmap=False)
        observed_ims[fieldidx] = cbps.image*cbps.cal_facs[inst]
        observed_ims[fieldidx] -= dc_template*cbps.cal_facs[inst]

        mask_fpath = fpath_dict['mask_base_path']+'/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'
        masks[fieldidx] = fits.open(mask_fpath)[1].data

        if mask_tail_ffest is not None:
            mask_fpath_ffest = fpath_dict['mask_base_path']+'/'+mask_tail_ffest+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail_ffest+'.fits'

            masks_ffest[fieldidx] = fits.open(mask_fpath_ffest)[1].data

        # sigclip = iter_sigma_clip_mask(observed_ims[fieldidx], sig=sig, nitermax=nitermax, mask=masks[fieldidx].astype(int))
        # masks[fieldidx] *= sigclip
        plot_map(masks[fieldidx]*observed_ims[fieldidx], title='masked map')

        if mask_tail_ffest is not None:
            plot_map(masks_ffest[fieldidx]*observed_ims[fieldidx], title='map for ff estimation')

    ciber_maps_byquad = [observed_ims[:, cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]] for q in range(4)]

    if masks_ffest is not None:
        masks_use = masks_ffest.copy()
    else:
        masks_use = masks.copy()

    if fc_sub:

        all_dot1, all_X, all_mask_rav = [], [], []
        mean_norms = [cbps.zl_levels_ciber_fields[inst][cbps.ciber_field_dict[ifield]] for ifield in ifield_list]
        weights_ff = cbps.compute_ff_weights(inst, mean_norms, ifield_list, photon_noise=True)

        print('weights ff:', weights_ff)

        if sig_ffest is not None:
            print('applying sigma clipping')
            for fieldidx, ifield in enumerate(ifield_list):

                sigclip = iter_sigma_clip_mask(observed_ims[fieldidx], sig=sig_ffest, nitermax=nitermax, mask=masks_use[fieldidx].astype(int))
                masks_use[fieldidx] *= sigclip

        processed_ims, ff_estimates, final_planes, stack_masks, all_coeffs = iterative_gradient_ff_solve(observed_ims, niter=niter, masks=masks_use, weights_ff=weights_ff, \
                                                                            ff_stack_min=1, plot=False, fc_sub=fc_sub, fc_sub_quad_offset=fc_sub_quad_offset, fc_sub_n_terms=fc_sub_n_terms, \
                                                                            with_gradient=grad_sub) # masks already accounting for ff_stack_min previously

        masks *= stack_masks
    else:

        for q, quad in enumerate(ciber_maps_byquad):

            if masks_ffest is not None:
                masks_quad = masks_ffest[:,cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]]
                clip_sigma_ff=5
            else:
                masks_quad = masks[:,cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]]
                clip_sigma_ff=None

            processed_ciber_maps_quad, ff_estimates_quad,\
                final_planes, stack_masks, ff_weights = process_ciber_maps(cbps, ifield_list, inst, ciber_maps_byquad[q], masks_quad, nitermax=nitermax, niter=niter, \
                                                                            clip_sigma=clip_sigma_ff)

            processed_ims[:,cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]] = processed_ciber_maps_quad

            print('Multiplying total masks by stack masks..')

            masks[:,cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]] *= stack_masks
            ff_estimates[:,cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]] = ff_estimates_quad

    ff_masks = (ff_estimates > ff_min)*(ff_estimates < ff_max)

    masks *= ff_masks
    
    all_masked_proc = []
    all_regrid_masked_proc = []
    all_regrid_fpath = []


    if sig is not None:
        print('Applying sigma clip of ', sig, 'over ', nitermax, 'iterations')
        for fieldidx, ifield in enumerate(ifield_list):
            sigclip = iter_sigma_clip_mask(processed_ims[fieldidx], sig=sig_ffest, nitermax=nitermax, mask=masks[fieldidx].astype(int))
            masks[fieldidx] *= sigclip

    for fieldidx, ifield in enumerate(ifield_list):
        
        obs_level = np.mean(processed_ims[fieldidx][masks[fieldidx]==1])
        print('obs level:', obs_level)

        masked_obs = processed_ims[fieldidx]*masks[fieldidx]

        plot_map(masked_obs, title='Masked map, target mask, ifield '+str(ifield))

        for q in range(4):

            obs_quad = masked_obs[cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]]
            processed_ims[fieldidx][cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]][obs_quad!=0] -= np.mean(obs_quad[obs_quad!=0])

        masked_proc = processed_ims[fieldidx]*masks[fieldidx]
        plot_map(masked_proc, title='Masked, mean subtracted maps, ifield '+str(ifield))

        all_masked_proc.append(masked_proc)
    
        ciber_regrid, ciber_fp_regrid = regrid_arrays_by_quadrant(masked_proc, ifield, inst0=regrid_to_inst, inst1=inst, \
                                                             plot=False, astr_dir=astr_dir, order=order, conserve_flux=conserve_flux)
    
        plot_map(ciber_regrid, title='CIBER regrid, ifield '+str(ifield))
        plot_map(masked_proc, title='CIBER map before regridding')
        
        print('mean of regrid is ', np.mean(ciber_regrid))
        
        all_regrid_masked_proc.append(ciber_regrid)
        
        if save:
            if fieldidx==0:
                proc_regrid_basepath = fpath_dict['observed_base_path']+'/TM'+str(regrid_to_inst)+'_TM'+str(inst)+'_cross/proc_regrid/'
                proc_regrid_basepath += mask_tail+'/'
                make_fpaths([proc_regrid_basepath])
            
            regrid_fpath = proc_regrid_basepath+'proc_regrid_TM'+str(inst)+'_to_TM'+str(regrid_to_inst)+'_ifield'+str(ifield)+'_'+mask_tail
            if conserve_flux:
                regrid_fpath += '_conserve_flux'
            else:
                regrid_fpath += '_order='+str(order)
            if fc_sub:
                regrid_fpath += '_quadoff_grad_fcsub_order'+str(fc_sub_n_terms)
            regrid_fpath += '.fits'

            hdul = write_regrid_proc_file(ciber_regrid, ifield, inst, regrid_to_inst, mask_tail=mask_tail, \
                                         obs_level=obs_level, masked_proc_orig=masked_proc)
            print('saving to ', regrid_fpath)
            hdul.writeto(regrid_fpath, overwrite=True)

            all_regrid_fpath.append(regrid_fpath)
        
    return all_regrid_masked_proc, all_regrid_fpath

def regrid_tm2_to_tm1_science_fields(ifield_list=[4,5,6,7,8], inst0=1, inst1=2, \
                                    inst0_maps=None, inst1_maps=None, flight_dat_base_path=config.exthdpath+'noise_model_validation_data/', \
                                    save_fpath=config.exthdpath+'/ciber_fluctuation_data/', \
                                    tail_name=None, plot=False, cal_facs=None, astr_dir=None):
    
    ciber_field_dict = dict({4:'elat10', 5:'elat30',6:'BootesB', 7:'BootesA', 8:'SWIRE'})

    save_paths, regrid_ims = [], []
    for fieldidx, ifield in enumerate(ifield_list):
        print('Loading ifield', ifield)
        fieldname = ciber_field_dict[ifield]
        print(fieldname)
        
        if inst0_maps is not None and inst1_maps is not None:
            print('taking directly from inst0_maps and inst1_maps..')
            tm1 = inst0_maps[fieldidx]
            tm2 = inst1_maps[fieldidx]
        else:
            tm1 = fits.open(flight_dat_base_path+'TM1/validationHalfExp/field'+str(ifield)+'/flightMap.FITS')[0].data
            tm2 = fits.open(flight_dat_base_path+'TM2/validationHalfExp/field'+str(ifield)+'/flightMap.FITS')[0].data       


        tm2_regrid, tm2_fp_regrid = regrid_arrays_by_quadrant(tm2, ifield, inst0=inst0, inst1=inst1, \
                                                             plot=plot, astr_dir=astr_dir)

        if cal_facs is not None:
            plot_map(cal_facs[1]*tm2, title='TM2 data')
            plot_map(cal_facs[1]*tm1, title='TM1 data')
            plot_map(cal_facs[2]*tm2_regrid, title='TM2 regrid')
            plot_map(cal_facs[1]*(tm1+tm2_regrid), title='TM1+TM2 data')
        regrid_ims.append(tm2_regrid)
    
        # make fits file saving data
    
        hduim = fits.ImageHDU(tm2_regrid, name='TM2_regrid')
        hdufp = fits.ImageHDU(tm2_fp_regrid, name='TM2_footprint')
        
        hdup = fits.PrimaryHDU()
        hdup.header['ifield'] = ifield
        hdup.header['dat_type'] = 'observed'
        hdul = fits.HDUList([hdup, hduim, hdufp])
        save_fpath_full = save_fpath+'TM'+str(inst1)+'/ciber_regrid/flightMap_ifield'+str(ifield)+'_TM'+str(inst1)+'_regrid_to_TM'+str(inst0)
        if tail_name is not None:
            save_fpath_full += '_'+tail_name
        hdul.writeto(save_fpath_full+'.fits', overwrite=True)
        
        save_paths.append(save_fpath_full+'.fits')
        
    return regrid_ims, save_paths

# def regrid_arrays_by_quadrant(map1, ifield, inst0=1, inst1=2, quad_list=['A', 'B', 'C', 'D'], \
#                              xoff=[0,0,512,512], yoff=[0,512,0,512], astr_map0_hdrs=None, astr_map1_hdrs=None, indiv_map0_hdr=None, indiv_map1_hdr=None, astr_dir=None, \
#                              plot=True, order=0):
    
#     ''' 
#     Used for regridding maps from one imager to another. For the CIBER1 imagers the 
#     astrometric solution is computed for each quadrant separately, so this function iterates
#     through the quadrants when constructing the full regridded images. 
    
#     Parameters
#     ----------
    
#     Returns
#     -------
    
#     '''

#     if astr_dir is None:
#         astr_dir = '../../ciber/data/'

#     ciber_field_dict = dict({4:'elat10', 5:'elat30',6:'BootesB', 7:'BootesA', 8:'SWIRE', 'train':'UDS'})

#     map1_regrid, map1_fp_regrid = [np.zeros_like(map1) for x in range(2)]

#     fieldname = ciber_field_dict[ifield]
    
#     map1_quads = [map1[yoff[iquad]:yoff[iquad]+512, xoff[iquad]:xoff[iquad]+512] for iquad in range(len(quad_list))]
    
#     if astr_map0_hdrs is None and indiv_map0_hdr is None:
#         print('loading WCS for inst = ', inst0)
#         astr_map0_hdrs = load_quad_hdrs(ifield, inst0, base_path=astr_dir, halves=False)
#     if astr_map1_hdrs is None and indiv_map1_hdr is None:
#         print('loading WCS for inst = ', inst1)
#         astr_map1_hdrs = load_quad_hdrs(ifield, inst1, base_path=astr_dir, halves=False)

#     # if astr_map0_hdrs is None and indiv_map0_hdr is None:
#     #     astr_map0_hdrs = [fits.open(astr_dir+'inst'+str(inst0)+'/'+fieldname+'_'+quad+'_astr.fits')[0].header for quad in quad_list]
#     # if astr_map1_hdrs is None and indiv_map1_hdr is None:
#     #     astr_map1_hdrs = [fits.open(astr_dir+'inst'+str(inst1)+'/'+fieldname+'_'+quad+'_astr.fits')[0].header for quad in quad_list]
#     # loop over quadrants of first imager
    
#     for iquad, quad in enumerate(quad_list):
        
#         # arrays, footprints = [], []
#         run_sum_footprint, sum_array = [np.zeros_like(map1_quads[0]) for x in range(2)]

#         # reproject each quadrant of second imager onto first imager
#         if indiv_map0_hdr is None:
#             for iquad2, quad2 in enumerate(quad_list):
#                 input_data = (map1_quads[iquad2], astr_map1_hdrs[iquad2])
#                 array, footprint = reproject_interp(input_data, astr_map0_hdrs[iquad], (512, 512), order=order)

#                 array[np.isnan(array)] = 0.
#                 footprint[np.isnan(footprint)] = 0.

#                 run_sum_footprint += footprint 
#                 sum_array[run_sum_footprint < 2] += array[run_sum_footprint < 2]
#                 run_sum_footprint[run_sum_footprint > 1] = 1

#                 # arrays.append(array)
#                 # footprints.append(footprint)
       
#         # sumarray = np.nansum(arrays, axis=0)
#         # sumfootprints = np.nansum(footprints, axis=0)

#         if plot:
#             plot_map(sum_array, title='sum array')
#             plot_map(run_sum_footprint, title='sum footprints')
        
#         # print('number of pixels with > 1 footprint', np.sum((sumfootprints==2)))
#         # map1_regrid[yoff[iquad]:yoff[iquad]+512, xoff[iquad]:xoff[iquad]+512] = sumarray
#         # map1_fp_regrid[yoff[iquad]:yoff[iquad]+512, xoff[iquad]:xoff[iquad]+512] = sumfootprints

#         map1_regrid[yoff[iquad]:yoff[iquad]+512, xoff[iquad]:xoff[iquad]+512] = sum_array
#         map1_fp_regrid[yoff[iquad]:yoff[iquad]+512, xoff[iquad]:xoff[iquad]+512] = run_sum_footprint

#     return map1_regrid, map1_fp_regrid

def ciber_x_ciber_regrid_all_masks(cbps, inst, cross_inst, cross_union_mask_tail, mask_tail, mask_tail_cross, ifield_list=[4,5,6,7,8], astr_base_path='data/', \
                                  base_fluc_path = 'data/fluctuation_data/', save=True, cross_mask=None, cross_inst_only=False, plot=False, plot_quad=False):
    
    ''' Function to regrid masks from one imager to another and return/save union of masks. 12/20/22'''
    mask_base_path = base_fluc_path+'TM'+str(inst)+'/masks/'
    mask_base_path_cross_inst = base_fluc_path+'TM'+str(cross_inst)+'/masks/'
    mask_base_path_cross_union = base_fluc_path+'TM'+str(inst)+'_TM'+str(cross_inst)+'_cross/masks/'

    masks, masks_cross, masks_cross_regrid, masks_union = [np.zeros((len(ifield_list), cbps.dimx, cbps.dimy)) for x in range(4)]
    union_savefpaths = []

    for fieldidx, ifield in enumerate(ifield_list):
        print('ifield ', ifield)
        
        astr_map0_hdrs = load_quad_hdrs(ifield, inst, base_path=astr_base_path, halves=False)
        astr_map1_hdrs = load_quad_hdrs(ifield, cross_inst, base_path=astr_base_path, halves=False)
        make_fpaths([mask_base_path_cross_union+cross_union_mask_tail])
        
        masks[fieldidx] = fits.open(mask_base_path+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits')['joint_mask_'+str(ifield)].data

        if cross_mask is None:
            if cross_inst_only:
                print('Only using instrument mask of cross CIBER map..')
                inst_mask_fpath = config.exthdpath+'/ciber_fluctuation_data/TM'+str(cross_inst)+'/masks/maskInst_102422/field'+str(ifield)+'_TM'+str(cross_inst)+'_maskInst_102422.fits'
                masks_cross[fieldidx] = cbps.load_mask(ifield, inst, mask_fpath=inst_mask_fpath, instkey='maskinst', inplace=False)
            else:
                print('Using full cross map mask..')
                masks_cross[fieldidx] = fits.open(mask_base_path_cross_inst+mask_tail_cross+'/joint_mask_ifield'+str(ifield)+'_inst'+str(cross_inst)+'_observed_'+mask_tail_cross+'.fits')['joint_mask_'+str(ifield)].data

            mask_cross_regrid, mask_cross_fp_regrid = regrid_arrays_by_quadrant(masks_cross[fieldidx], ifield, inst0=inst, inst1=cross_inst, \
                                                     astr_map0_hdrs=astr_map0_hdrs, astr_map1_hdrs=astr_map1_hdrs, plot=plot_quad)

            masks_cross_regrid[fieldidx] = mask_cross_regrid
        masks_union[fieldidx] = masks[fieldidx]*mask_cross_regrid

        if plot:
            plot_map(masks[fieldidx], title='mask fieldidx = '+str(fieldidx))
            plot_map(masks_cross[fieldidx], title='mask fieldidx = '+str(fieldidx))

        masks_union[fieldidx][masks_union[fieldidx] > 1] = 1.0
        
        print('Mask fraction for mask TM'+str(inst)+' is ', np.sum(masks[fieldidx])/float(1024**2))
        print('Mask fraction for mask TM'+str(cross_inst)+' is ', np.sum(masks_cross[fieldidx])/float(1024**2))
        print('Mask fraction for union mask is ', np.sum(masks_union[fieldidx])/float(1024**2))
        if plot:
            plot_map(masks_union[fieldidx], title='mask x mask cross')

        if save:
            hdul = write_mask_file(masks_union[fieldidx], ifield, inst, dat_type='cross_observed', cross_inst=cross_inst)
            union_savefpath = mask_base_path_cross_union+cross_union_mask_tail+'/joint_mask_ifield'+str(ifield)+'_TM'+str(inst)+'_TM'+str(cross_inst)+'_observed_'+cross_union_mask_tail+'.fits'
            print('Saving cross mask to ', union_savefpath)

            hdul.writeto(union_savefpath, overwrite=True)
            union_savefpaths.append(union_savefpath)
    if save:
        return masks, masks_cross, masks_cross_regrid, masks_union, union_savefpaths
    
    return masks, masks_cross, masks_cross_regrid, masks_union

def calculate_ciber_cross_noise_uncertainty(inst, ifield, mask, cross_map, mask_cross=None, noise_model=None,\
                                            nsims=200, n_split=4, cbps=None, plot=False, verbose=False):
    ''' 
    This function is for evaluating the CIBER noise x [IRIS/Spitzer] maps by simulating a large 
    number of CIBER read noise realizations, masking them and then computing the cross power spectrum. 
    '''
    if cbps is None:
        cbps = CIBER_PS_pipeline()
    all_cl1ds_noise, all_cl1ds_cross_noise = [], []
    
    maplist_split_shape = (nsims//n_split, cbps.dimx, cbps.dimy)


    if noise_model is None:


        nmfile = np.load('data/noise_models_sim/noise_model_fpaths_TM'+str(inst)+'_021523.npz')
        noise_model_fpaths_quad = nmfile['noise_model_fpaths_quad']
        noise_model = fits.open(noise_model_fpaths_quad[ifield-4])[1].data
    
    if mask_cross is not None:
        mask *= mask_cross
        
    if plot:
        plot_map(mask, title='Combined mask')
        plot_map(cross_map*mask, title='cross map x mask')

    empty_aligned_objs, fft_objs = construct_pyfftw_objs(3, maplist_split_shape)
    for i in range(n_split):
        print('Split '+str(i+1)+' of '+str(n_split)+'..')

        rnmaps, snmaps = cbps.noise_model_realization(inst, maplist_split_shape, noise_model, fft_obj=fft_objs[0],\
                                              read_noise=True, photon_noise=False, shot_sigma_sb=None)


        plot_map(rnmaps[0])

        cross_map_meansub = cross_map*mask
        cross_map_meansub[mask==1] -= np.mean(cross_map_meansub[mask==1])

        if plot:
            plot_map(cross_map_meansub, title='cross map meansub')

        cl1ds_cross = [get_power_spec(mask*indiv_map, map_b=cross_map_meansub, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)[1] for indiv_map in rnmaps]

        all_cl1ds_cross_noise.extend(cl1ds_cross)
        
    lb = get_power_spec(mask, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)[0]

        
    return lb, all_cl1ds_cross_noise


def get_ciber_dgl_powerspec2(dgl_fpath, inst, iras_color_facs=None, mapkey='iris_map', pixsize=7., dgl_bins=10):

    '''

    Parameters
    ----------

    iras_color_facs: dictionary of scalar conversion factors between mean surface brightness at 100 micron vs. CIBER bands

    Returns
    -------

    lb: multipoles used
    cl: angular 1d power spectrum
    dgl_map: dgl map obtained from dgl_fpath


    '''

    if iras_color_facs is None:
        iras_color_facs = dict({1:6.4, 2:2.6}) # from MIRIS observations, Onishi++2018

    dgl = fits.open(dgl_fpath)[1].data

    dgl_map = dgl*iras_color_facs[inst]
    dgl_map -= np.mean(dgl_map)

    lb, cl, cl_err = get_power_spec(dgl_map, pixsize=pixsize, nbins=dgl_bins)

    return lb, cl, dgl_map


def interpolate_iris_maps_hp(iris_hp, cbps, inst, ncoarse_samp, nside_upsample=None,\
                             ifield_list=[4, 5, 6, 7, 8], use_ciber_wcs=True, nside_deg=4, \
                            plot=True, nside=2048, nest=False):
    ''' 
    Interpolate IRIS maps from Healpix format to cartesian grid 
    If nside_upsample provided, maps upsampled to desired resolution, 
        otherwise they are kept at resolution of ncoarse_samp.
    '''
    
    all_maps = []
    
    field_center_ras = dict({'elat10':191.5, 'elat30':193.943, 'BootesB':218.109, 'BootesA':219.249, 'SWIRE':241.53})
    field_center_decs = dict({'elat10':8.25, 'elat30':27.998, 'BootesB':33.175, 'BootesA':34.832, 'SWIRE':54.767})

    
    for fieldidx, ifield in enumerate(ifield_list):
        
        iris_coarse_sample = np.zeros((ncoarse_samp, ncoarse_samp))
        
        
        if use_ciber_wcs:
            field = cbps.ciber_field_dict[ifield]
            wcs_hdrs = load_all_ciber_quad_wcs_hdrs(inst, field)
            print('TM'+str(inst), 'ifield '+str(ifield))
            
            for ix in range(ncoarse_samp):
                if ix %10 == 0:
                    print('ix = ', ix)
                for iy in range(ncoarse_samp):

                    x0, x1 = ix*npix_persamp, (ix+1)*npix_persamp - 1
                    y0, y1 = iy*npix_persamp, (iy+1)*npix_persamp - 1


                    xs, ys = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
                    ciber_ra_av_all, ciber_dec_av_all = wcs_hdrs[0].all_pix2world(xs, ys, 0)
                    map_ra_av_new = np.mean(ciber_ra_av_all)
                    map_dec_av_new = np.mean(ciber_dec_av_all)
                    
                    c = SkyCoord(ra=map_ra_av_new*u.degree, dec=map_decs[ix,iy]*u.degree, frame='icrs')
                    ipix = hp.pixelfunc.ang2pix(nside=nside, theta=c.galactic.l.degree, phi=c.galactic.b.degree, lonlat=True, nest=nest)
                    iris_coarse_sample[ix, iy] = iris_hp[ipix]

        else:
            ra_cen = field_center_ras[cbps.ciber_field_dict[ifield]]
            dec_cen = field_center_decs[cbps.ciber_field_dict[ifield]]

            print('ra/dec:', ra_cen, dec_cen)
            map_ras, map_decs = generate_map_meshgrid(ra_cen, dec_cen, nside_deg, dimx=ncoarse_samp, dimy=ncoarse_samp)
            if plot:
                plot_map(map_ras, title='RA')
                plot_map(map_decs, title='DEC')

            for ix in range(ncoarse_samp):
                for iy in range(ncoarse_samp):
                    c = SkyCoord(ra=map_ras[ix,iy]*u.degree, dec=map_decs[ix,iy]*u.degree, frame='icrs')
                    ipix = hp.pixelfunc.ang2pix(nside=nside, theta=c.galactic.l.degree, phi=c.galactic.b.degree, lonlat=True, nest=nest)
                    iris_coarse_sample[ix, iy] = iris_hp[ipix]

        if nside_upsample is not None:
            iris_resize = np.array(Image.fromarray(iris_coarse_sample).resize((nside_upsample, nside_upsample))).transpose()
            
            plot_map(iris_resize)

            all_maps.append(iris_resize)
        else:
            plot_map(np.array(iris_coarse_sample).transpose())
                    
            all_maps.append(np.array(iris_coarse_sample).transpose())
        
    return all_maps
        

def func_powerlaw_fixgamma_steep(ell, c):
    return c*(ell**(-3.0))

def func_powerlaw_fixgamma(ell, c):
    return c*(ell**(-2.6))

def func_powerlaw_noaddnorm(ell, m, c):
    return c*(ell**m)

def func_powerlaw(ell, m, c, c0):
    return c0 + ell**m * c


# 3.4 arcminute resolution for healpix, 7" CIBER pixels --> 30 x 30 pixel size sampling
# so maybe oversample at 16x16 pixel averages
def which_quad(x, y):
    
    bound = 511.5
    if (x < bound)&(y > bound):
        quad = 1
        
    elif (x > bound)*(y < bound):
        quad = 2
    
    elif (x > bound)*(y > bound):
        quad = 3
    
    else:
        quad = 0
    
    return quad

def func_powerlaw_fixgamma_lss(ell, c):
    return c*(ell**(-2.7))
def func_powerlaw_fixgamma_clean(ell, c):
    return c*(ell**(-3.2))


def ciber_dgl_cross_correlations(ifield_list=[4, 5, 6, 7, 8], inst_list=[1,2], dgl_modes=['IRIS', 'sfd', 'sfd_clean'],\
                                 plot_alpha=0.4, datestr='111323', lmax=2500, text_xpos=120, save=False, \
                                show=True, colors = ['C3', 'k'], apply_beam_correction=False, include_irac=False, \
                                figsize=(9, 10), mask_frac=None):
    
    
    cbps = CIBER_PS_pipeline()
    
    gamma_dict = dict({'IRIS':-2.7, 'sfd_clean':-3.2, 'mf15':-2.7, 'sfd_clean_plus_LSS':-2.7})
    theta_fwhm_dict = dict({'IRIS':4.0, 'sfd_clean':6.1, 'mf15':6.1, 'sfd_clean_plus_LSS':6.1})
    fsky = 2*2/(41253.)  

    if mask_frac is not None:
        print('multiplying fsky by mask frac = ', mask_frac)
        fsky *= mask_frac

    all_AC_A1, all_dAC_sq, all_best_ps_fit_av, all_fieldav_ps = [[[] for x in range(2)] for y in range(4)]
        

    nrow = len(dgl_modes)
    ncol = len(inst_list)
    if include_irac:
        ncol += 1

    fig = plt.figure(figsize=figsize)


    for dgl_idx, dgl_mode in enumerate(dgl_modes):  

        bl_dgl = beam_correction_gaussian(lb, theta_fwhm_dict[dgl_mode], unit='arcmin')

        if dgl_mode=='IRIS' or dgl_mode=='mf15' or dgl_mode=='sfd_clean_plus_LSS':
            func_ps = func_powerlaw_fixgamma_lss
            modl_fits_dgl_auto = np.load(config.ciber_basepath+'data/fluctuation_data/TM1/dgl_tracer_maps/IRIS/IRIS_large_best_modl_fits_TM1_I100.npz')

        elif dgl_mode=='sfd_clean':
            func_ps = func_powerlaw_fixgamma_clean
            modl_fits_dgl_auto = np.load(config.ciber_basepath+'data/fluctuation_data/TM1/dgl_tracer_maps/sfd_clean/sfd_clean_large_best_modl_fits_TM1_I100.npz')

        lb_modl, all_ps_fits, best_ps_fit_av = modl_fits_dgl_auto['lb_modl'], np.array(modl_fits_dgl_auto['all_best_ps_fits']), np.array(modl_fits_dgl_auto['best_ps_fit_av'])
        gamma_val = gamma_dict[dgl_mode]
        
        observed_run_name, cross_text_lab = grab_dgl_config(dgl_mode, addstr='_I100_120423')

        for inst in inst_list:
                
            # plt.subplot(3,2,2*dgl_idx+inst)
            plt.subplot(nrow,ncol,2*dgl_idx+inst)
            
            sim_test_fpath = config.ciber_basepath+'data/input_recovered_ps/'+datestr+'/TM'+str(inst)+'/'
            obs_clfile = np.load(sim_test_fpath+observed_run_name+'/input_recovered_ps_estFF_simidx0.npz')
            lb, observed_recov_ps, observed_recov_dcl = obs_clfile['lb'], obs_clfile['recovered_ps_est_nofluc'], obs_clfile['recovered_dcl']

            observed_recov_ps /= bl
            observed_recov_dcl /= bl

            prefac = lb*(lb+1)/(2*np.pi)
            lb_mask = (lb < lmax)*(lb > lb[0])

            for fieldidx, ifield in enumerate(ifield_list):

                frac_knox_errors = return_frac_knox_error(lb, cbps.Mkk_obj.delta_ell)
                knox_errors_full = frac_knox_errors*np.abs(observed_recov_ps[fieldidx])

                dgl_mode_noise = 'sfd_clean'
                dgl_nl_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/dgl_tracer_maps/'
                dgl_nl_fpath += dgl_mode_noise+'/nl1ds_TM'+str(inst)+'_ifield'+str(ifield)+'_cibernoise_'+dgl_mode_noise+'_cross.npz'
                
                cibernoise_iris_cross = np.load(dgl_nl_fpath)['all_cl1ds_cross_noise']
                mean_nl_cross = np.std(cibernoise_iris_cross, axis=0)
                dcl_err_combined_full = np.sqrt(knox_errors_full**2 + observed_recov_dcl[fieldidx]**2 + mean_nl_cross**2)

                plt.errorbar(lb[lb_mask], prefac[lb_mask]*observed_recov_ps[fieldidx,lb_mask], yerr=prefac[lb_mask]*dcl_err_combined_full[lb_mask], alpha=plot_alpha, fmt='o', capsize=3, color='C'+str(fieldidx), markersize=4, label=cbps.ciber_field_dict[ifield])

                
            meancl = np.mean(observed_recov_ps, axis=0)
            std_meancl = np.std(observed_recov_ps, axis=0)/np.sqrt(5)
            lbpivot = np.min(lb[lb_mask])

            ps_fit_ciber_iris_fg, cov_fg = scipy.optimize.curve_fit(func_ps, lb[lb_mask]/lbpivot, meancl[lb_mask], sigma=std_meancl[lb_mask], absolute_sigma=True)

        
            pivfac = lbpivot*(lbpivot+1)/(2*np.pi)
            amp_scaled = ps_fit_ciber_iris_fg[0]*pivfac
            print('amp scaled for tm'+str(inst)+' is ', amp_scaled)
            print('amplitude for TM'+str(inst)+' (fixed gamma) is '+str(ps_fit_ciber_iris_fg[0])+'+-'+str(np.sqrt(cov_fg[0,0])))

            # rescale 6x6 deg model by best fit amplitude of cross spectrum fit
            fac = 3
            mkk_large_obj = Mkk_bare(dimx=1024*fac, dimy=1024*fac, ell_min=180./fac, pixsize=7., nbins=25, precompute=True)                
            lb_iris_large = mkk_large_obj.midbin_ell
            prefac_irislb = lb_iris_large*(lb_iris_large+1)/(2*np.pi)
            best_ps_fit_fg = ps_fit_ciber_iris_fg[0]*(lb_iris_large.astype(float)/lbpivot)**(gamma_val)
            unc_ps_fit_fg = np.sqrt(cov_fg[0,0])*(lb_iris_large.astype(float)/lbpivot)**(gamma_val)

            print('cross / best_ps_fit_av:', (prefac_irislb*best_ps_fit_fg)[:13]/best_ps_fit_av)
            AC_A1 = np.mean((prefac_irislb*best_ps_fit_fg)[:13]/best_ps_fit_av)
            sigma_AC_A1 = np.mean((prefac_irislb*unc_ps_fit_fg)[:13]/best_ps_fit_av)

            print('AC_A1 = ', AC_A1)
            print('sigma AC_A1 = ', sigma_AC_A1)

            dAC_sq = 2*sigma_AC_A1*AC_A1

            all_AC_A1[inst-1].append(AC_A1)
            all_dAC_sq[inst-1].append(dAC_sq)
            all_best_ps_fit_av[inst-1].append(best_ps_fit_av)

            upper_ps_fit_fg = (ps_fit_ciber_iris_fg[0]+np.sqrt(cov_fg[0,0]))*(lb_iris_large.astype(float)/lbpivot)**(gamma_val)
            lower_ps_fit_fg = (ps_fit_ciber_iris_fg[0]-np.sqrt(cov_fg[0,0]))*(lb_iris_large.astype(float)/lbpivot)**(gamma_val)

            plt.plot(lb_iris_large, prefac_irislb*best_ps_fit_fg, color=colors[dgl_idx], label='Power law cross PS fit\n(Fixed $\\gamma$)')

            plt.fill_between(lb_iris_large, prefac_irislb*lower_ps_fit_fg, prefac_irislb*upper_ps_fit_fg, color=colors[dgl_idx], alpha=0.3)

            snr_av = np.mean(observed_recov_ps, axis=0)/(np.std(observed_recov_ps, axis=0)*np.sqrt(1.25)/np.sqrt(5))
            print('snr av is ', snr_av[lb_mask])
            total_snr = np.sqrt(np.sum(snr_av[lb_mask]**2))
            print('total snr is ', total_snr)

            av_ps = np.mean(observed_recov_ps, axis=0)
            std_ps = np.std(observed_recov_ps, axis=0)[lb_mask]*np.sqrt(1.25)/np.sqrt(5)
            
            if save:
                dgl_save_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/dgl_tracer_maps/'+dgl_mode+'/dgl_auto_constraints_TM'+str(inst)+'_'+dgl_mode+'_010924.npz'
                print('Saving to ', dgl_save_fpath)
                
                np.savez(dgl_save_fpath, \
                        lb_modl=lb_modl, best_ps_fit_av=best_ps_fit_av, AC_A1=AC_A1, dAC_sq=dAC_sq, \
                        av_ps=av_ps, std_ps=std_ps, prefac=prefac, lb=lb, lb_mask=lb_mask, \
                        best_ps_fit_fg=best_ps_fit_fg, lb_iris_large=lb_iris_large, prefac_irislb=prefac_irislb, \
                        lower_ps_fit_fg=lower_ps_fit_fg, upper_ps_fit_fg=upper_ps_fit_fg)
            
            plt.errorbar(lb[lb_mask], prefac[lb_mask]*np.mean(observed_recov_ps, axis=0)[lb_mask], fmt='o', alpha=0.8, capsize=3, markersize=4, zorder=10, yerr=prefac[lb_mask]*np.std(observed_recov_ps, axis=0)[lb_mask]*np.sqrt(1.25)/np.sqrt(5), color='k', label='Field average')
            plt.xscale('log')
            plt.xlim(100, lmax)
            
            if dgl_idx==1:
                plt.xlabel('$\\ell$', fontsize=16)
            if inst==1:
                plt.ylabel('$D_{\\ell}$ [(nW m$^{-2}$ sr$^{-1}$)(MJy sr$^{-1}$)]', fontsize=11)

            plt.grid(alpha=0.8)
                
            text_ypos_list = [1.5, 0.6]

            if inst==1:
                plt.yscale('log')
                plt.ylim(1e-3, 4e0)
                plt.text(text_xpos, text_ypos_list[0], 'CIBER 1.1 $\\mu$m $\\times$ '+cross_text_lab, fontsize=13, color=colors[dgl_idx], bbox=dict({'facecolor':'white', 'edgecolor':'None', 'alpha':0.9}))           
                plt.text(text_xpos, text_ypos_list[1], '$\\gamma='+str(gamma_val)+'$', fontsize=13, color=colors[dgl_idx])


            elif inst==2:
                plt.yscale('log')
                plt.ylim(1e-3, 4e0)
                plt.text(text_xpos, text_ypos_list[0], 'CIBER 1.8 $\\mu$m $\\times$ '+cross_text_lab, fontsize=13, color=colors[dgl_idx], bbox=dict({'facecolor':'white', 'edgecolor':'None', 'alpha':0.9}))
                plt.text(text_xpos, text_ypos_list[1], '$\\gamma='+str(gamma_val)+'$', fontsize=13, color=colors[dgl_idx])

            if inst==2 and dgl_idx==0:
                plt.legend(loc=4, ncol=4, bbox_to_anchor=[0.85, 1.05])
    




    if show:
        plt.show()

    return fig, all_best_ps_fit_av, all_AC_A1, all_dAC_sq, lb_iris_large


class dgl_meas():
        
    def __init__(self, name):
        self.name = name
        self.lam_meas = []
        self.dgl_color = []
        self.dgl_color_unc = []
        self.lam_width = []
    
    def add_measurements(self, lam, dgl_col, dgl_col_unc, lam_width=None):
        
        if type(lam)==list:
            print('extending')
            self.lam_meas.extend(lam)
            self.dgl_color.extend(dgl_col)
            self.dgl_color_unc.extend(dgl_col_unc)
            
            if lam_width is not None:
                self.lam_width.extend(lam_width)
            
        else:
            self.lam_meas.append(lam)
            self.dgl_color.append(dgl_col)
            self.dgl_color_unc.append(dgl_col_unc)

def compile_dgl_measurements(dgl_basepath = 'figures/dgl_measurements/'):
    
    ciber_dgl_csfd = dgl_meas('CIBER $\\times$ CSFD (this work)')
    # ciber_dgl_csfd.add_measurements([1.05, 1.79], [8.2, 8.6], [2.2, 2.1], lam_width=[0.15, 0.3])
    ciber_dgl_csfd.add_measurements([1.05, 1.79], [8.2, 8.6], [2.2, 2.1], lam_width=[0.15, 0.3])

    irac_dgl_csfd = dgl_meas('IRAC $\\times$ CSFD (this work)')
    irac_dgl_csfd.add_measurements([3.6], [0.58], [0.6], lam_width=[0.4])

    # ciber_dgl_sfd = dgl_meas('CIBER $\\times$ SFD (this work)')
    # ciber_dgl_sfd.add_measurements([1.05, 1.79], [9.1, 7.7], [1.5, 0.6], lam_width=[0.15, 0.3])

    # Onishi 2018
    onishi = dgl_meas('Onishi et al. (2018)')
    onishi.add_measurements([1.1, 1.6], [6.4, 2.6], [0.9, 0.7])
    
    # arai 2015
    arai_wav, arai_mean, arai_std = grab_wav_dgl_col_from_csv(dgl_basepath+'arai_mean.csv', dgl_basepath+'arai_upper.csv')
    arai_dgl = dgl_meas('Arai et al. (2015)')
    arai_dgl.add_measurements(list(arai_wav), list(arai_mean), list(arai_std))

    # tsumura 2013
    tsumura_wav, tsumura_mean, tsumura_std = grab_wav_dgl_col_from_csv(dgl_basepath+'tsumura_2013.csv', dgl_basepath+'tsumura_upper.csv')
    tsumura_dgl = dgl_meas('Tsumura et al. (2013b)')
    tsumura_dgl.add_measurements(list(tsumura_wav), list(tsumura_mean), list(tsumura_std))

    # sano

    sano_wav, sano_mean, sano_std = grab_wav_dgl_col_from_csv(dgl_basepath+'sano_mean.csv', dgl_basepath+'sano_upper.csv')
    print('sano wav:', sano_wav)
    sano_dgl = dgl_meas('Sano et al. (2015, 2016a)')
    sano_dgl.add_measurements(list(sano_wav), list(sano_mean), list(sano_std))

    # witt 2008

    witt_wav, witt_mean, witt_std = grab_wav_dgl_col_from_csv(dgl_basepath+'witt_mean.csv', dgl_basepath+'witt_upper.csv')
    witt_dgl = dgl_meas('Witt et al. (2008)')
    witt_dgl.add_measurements(list(witt_wav), list(witt_mean), list(witt_std))


    # paley 1991

    paley_wav, paley_mean, paley_std = grab_wav_dgl_col_from_csv(dgl_basepath+'paley_mean.csv', dgl_basepath+'paley_upper.csv')
    paley_dgl = dgl_meas('Paley et al. (1991)')
    paley_dgl.add_measurements(list(paley_wav), list(paley_mean), list(paley_std))
    

    # Symons+2023
    symons_dgl = dgl_meas('Symons et al. (2023)')
    symons_dgl.add_measurements([0.62], [2.74], [0.75], lam_width=[0.2])

    # Postman+2024
    postman_dgl = dgl_meas('Postman et al. (2024)')
    postman_dgl.add_measurements([0.62], [10.09], [0.59], lam_width=[0.3])


    dgl_meas_dicts = [ciber_dgl_csfd, irac_dgl_csfd, onishi, arai_dgl, tsumura_dgl, sano_dgl, witt_dgl, paley_dgl, symons_dgl, \
                        postman_dgl]

    return dgl_meas_dicts

# powerspec_utils.py
# def ciber_ciber_rl_coefficient(obs_name_A, obs_name_B, obs_name_AB, startidx=1, endidx=-1):


