import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from scipy import interpolate
import os
import astropy
import astropy.wcs as wcs
import config

from ciber.core.powerspec_pipeline import *
from ciber.io.catalog_utils import *
from ciber.core.ps_pipeline_go import *
from ciber.mocks.cib_mocks import *
from ciber.theory.cl_predictions import *
from ciber.pseudo_cl.mkk_compute import *


def compute_ciber_hsc_mkk(cbps, inst, mag_lim, quadoff_grad=True, grad_sub=False, ifield=8, mask_tail=None, datestr='111323', \
                         n_mkk_sim=200, n_mkk_split=4, \
                          hscmask_basepath='data/20250731_manualmasks_hsc_swire/'):
    
    # cbps = CIBER_PS_pipeline()
    config_dict, pscb_dict, float_param_dict, fpath_dict = return_default_cbps_dicts()
    ciber_mock_fpath = config.exthdpath+'ciber_mocks/'
    datestr_trilegal = datestr
    fpath_dict, list_of_dirpaths, base_path, trilegal_base_path = set_up_filepaths_cbps(fpath_dict, inst, 'test', datestr,\
                                                                                        datestr_trilegal=datestr, data_type='observed', \
                                                                                       save_fpaths=True)
    
    
    if quadoff_grad:
        mkk_mask_type = 'quadoff_grad'
    elif grad_sub:
        mkk_mask_type = 'gradsub'
        
    else:
        mkk_mask_type = 'maskonly'
        
    print('MKK type is ', mkk_mask_type)
    
    bandstr = cbps.bandstr_dict[inst]
    
    save_fpath_list = []
    
    if mask_tail is None:
        mask_tail_obs = 'maglim_'+bandstr+'_Vega_'+str(mag_lim)+'_111323_ukdebias'
    else:
        mask_tail_obs = mask_tail
        
    mask_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/masks/'+mask_tail_obs+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail_obs+'.fits'  
    
    mask = fits.open(mask_fpath)[1].data   
    # load hsc mask
    
    hscmask = np.load(hscmask_basepath+'TM'+str(inst)+'/hsc_mask_TM'+str(inst)+'_073125.npz')['hscmask']
    mask *= hscmask

    if inst==1:
        mask[-120:, -250:] = 0.
        mask[:120, -250:] = 0.
    elif inst==2:
        mask[:250, :120] = 0.
        mask[:250, -120:] = 0. 
        
        
    plot_map(mask)
    
    av_Mkk = cbps.Mkk_obj.get_mkk_sim(mask, n_mkk_sim, n_split=n_mkk_split, quadoff_grad=quadoff_grad, grad_sub=grad_sub)

    
    
    inv_Mkk = compute_inverse_mkk(av_Mkk)

    mkk_hdul = write_Mkk_fits(av_Mkk, inv_Mkk, ifield, inst, dat_type='observed', mag_lim_AB=mag_lim+cbps.Vega_to_AB[inst])    

    mkkonly_basepath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/mkk/'+mask_tail_obs+'/'
    mkk_savepath = mkkonly_basepath+'mkk_'+mkk_mask_type+'_estimate_ifield'+str(ifield)+'_observed_'+mask_tail_obs+'_hsc.fits'

    print('Writing Mkk file to ', mkk_savepath)
    mkk_hdul.writeto(mkk_savepath, overwrite=True)
    
    return mkk_savepath

def compute_mkk_inst_mask_wrapper(inst, ifield_list=[4, 5, 6, 7, 8], nsims=500, n_split=10, mask_base_path=None, \
                                 maskinst_tail='maskInst_102422'):
    
    if mask_base_path is None:
        mask_base_path = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/masks/'
    
    mkk_base_path = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/mkk/'+maskinst_tail+'/'
    
    mkk_savepaths = []
    
    for fieldidx, ifield in enumerate(ifield_list):
        maskInst_fpath = mask_base_path+maskinst_tail+'/field'+str(ifield)+'_TM'+str(inst)+'_maskInst_102422.fits'
        mask_inst = fits.open(maskInst_fpath)['maskInst'].data # 10/24/22

        plot_map(mask_inst, title='Instrument mask, TM'+str(inst)+', ifield '+str(ifield))
        av_Mkk = cbps_nm.cbps.Mkk_obj.get_mkk_sim(mask_inst, nsims, n_split=n_split)
        inv_Mkk = compute_inverse_mkk(av_Mkk)

        mkk_hdul = write_Mkk_fits(av_Mkk, inv_Mkk, ifield, inst, dat_type='inst')
        
        mkk_savepath = mkk_base_path+'mkk_maskonly_estimate_ifield'+str(ifield)+'_inst'+str(inst)+'_'+maskinst_tail+'.fits'
        
        plot_mkk_matrix(av_Mkk)

        print('Saving field ', ifield, 'to', mkk_savepath)
        mkk_savepaths.append(mkk_savepath)
        
        mkk_hdul.writeto(mkk_savepath, overwrite=True)
        
    return mkk_savepaths


def compute_mkk_ciber_irac_observed_wrapper(ifield_list=[6, 7], inst_list=[1, 2], irac_ch_list=[1, 2], \
                                           mag_lim_wise=16.0, maglim_ciber_dict=None, n_mkk_sim=200, n_mkk_split=4, \
                                           fc_sub=True, fc_sub_quad_offset=True, fc_sub_n_terms=2, grad_sub=True, \
                                           mkk_type=None, ff_est=False, \
                                           mkk_mask_type='quadoff_grad_fcsub_order2', mkk_ffest_type='ffest_quadoff_grad_fcsub_order2'):
    
    cbps = CIBER_PS_pipeline()
    
    bandstr_dict = dict({1:'J', 2:'H'})
    
    if maglim_ciber_dict is None:
        maglim_ciber_dict = dict({1:17.5, 2:17.0})

    mkk_save_fpaths = []

    for irac_ch in irac_ch_list:
        for inst in inst_list:
            
            mean_norms = [cbps.zl_levels_ciber_fields[inst][cbps.ciber_field_dict[ifield]] for ifield in ifield_list]
            ff_weights = cbps.compute_ff_weights(inst, mean_norms, ifield_list, photon_noise=True, read_noise_models=None)

            
            print('On IRAC CH '+str(irac_ch)+', CIBER inst '+str(inst))
        
            bandstr = bandstr_dict[inst]
            maglim = maglim_ciber_dict[inst]
            
            mask_tail = 'maglim_'+bandstr+'_Vega_'+str(maglim)+'_maglim_IRAC_Vega_'+str(mag_lim_wise)+'_111323_take2'
            
            if not ff_est:
                mask_tail += '_IRAC_CH'+str(irac_ch)
        
            mask_fpaths = [config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/masks/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits' for ifield in ifield_list]
            
            masks = np.array([fits.open(mask_fpaths[fieldidx])['joint_mask_'+str(ifield)].data for fieldidx, ifield in enumerate(ifield_list)])
            
            if ff_est:
                mkk_type = mkk_ffest_type
                
                mkk_basepath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/mkk_ffest/'+mask_tail+'/'
                make_fpaths([mkk_basepath])
                
                average_Mkks_perfield, Mkks_per_field = estimate_mkk_ffest_quadoff(cbps, n_mkk_sim, masks, n_split=n_mkk_split, ifield_list=ifield_list, mean_normalizations=mean_norms, ff_weights=ff_weights, \
                                               verbose=False, fc_sub=fc_sub, fc_sub_quad_offset=fc_sub_quad_offset, fc_sub_n_terms=fc_sub_n_terms, niter=1, grad_sub=grad_sub)

            else:
                mkk_type = mkk_mask_type
                mkk_basepath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/mkk/'+mask_tail+'/'

                average_Mkks_perfield = []

                for fieldidx, ifield in enumerate(ifield_list):
                    mask_save_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/masks/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'  
                    mask = fits.open(mask_save_fpath)['joint_mask_'+str(ifield)].data
                    plot_map(mask)

                    av_Mkk = cbps.Mkk_obj.get_mkk_sim(masks[fieldidx], n_mkk_sim, n_split=n_mkk_split, fc_sub=fc_sub,\
                                                        fc_sub_quad_offset=fc_sub_quad_offset, fc_sub_n_terms=fc_sub_n_terms, grad_sub=grad_sub)

                    average_Mkks_perfield.append(av_Mkk)
                    
            make_fpaths([mkk_basepath])

                
            for fieldidx, ifield in enumerate(ifield_list):
                
                inv_Mkk = compute_inverse_mkk(average_Mkks_perfield[fieldidx])

                mkk_hdul = write_Mkk_fits(average_Mkks_perfield[fieldidx], inv_Mkk, ifield, inst, dat_type='real')

                mkk_savepath = mkk_basepath+'mkk_'+mkk_type+'_ifield'+str(ifield)+'_observed_'+mask_tail+'.fits'

                plot_mkk_matrix(average_Mkks_perfield[fieldidx])

                print('saving field ', ifield, 'to', mkk_savepath)
                mkk_hdul.writeto(mkk_savepath, overwrite=True)
                
                mkk_save_fpaths.append(mkk_savepath)
                
    return mkk_save_fpaths


def compute_mkk_fielddiff_wrapper(inst, mag_lim_list, mask_tail_list=None, ifieldA = 6, ifieldB = 7, save_comb_mask=True, \
                                 n_mkk_sim=200, n_mkk_split=4, mkk_type='quadoff_grad_fcsub_order2', fc_sub=True, fc_sub_quad_offset=True, fc_sub_n_terms=2, grad_sub=True, \
                                 plot=True):
        
    if inst==1:
        bandstr = 'J'
    else:
        bandstr = 'H'
        
    mkk_savepath_list = []
    mask_savepath_list = []

    for m, mag_lim in enumerate(mag_lim_list):
        if mask_tail_list is None:
            mask_tail = 'maglim_'+bandstr+'_Vega_'+str(mag_lim)+'_111323_ukdebias'
        else:
            mask_tail = mask_tail_list[m]

        mask_save_fpathA = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/masks/'+mask_tail+'/joint_mask_ifield'+str(ifieldA)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'  
        mask_save_fpathB = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/masks/'+mask_tail+'/joint_mask_ifield'+str(ifieldB)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'  

        maskA = fits.open(mask_save_fpathA)['joint_mask_'+str(ifieldA)].data
        maskB = fits.open(mask_save_fpathB)['joint_mask_'+str(ifieldB)].data

        if plot:
            plot_map(maskA*maskB, title='Combined mask')
        print('Unmasked fraction:', np.sum(maskA*maskB)/float(maskA.shape[0]*maskA.shape[1]))

        av_Mkk = cbps.Mkk_obj.get_mkk_sim(maskA*maskB, n_mkk_sim, n_split=n_mkk_split, fc_sub=fc_sub, fc_sub_quad_offset=fc_sub_quad_offset, \
                                         fc_sub_n_terms=fc_sub_n_terms, grad_sub=grad_sub)
        
        inv_Mkk = compute_inverse_mkk(av_Mkk)

        mkk_hdul = write_Mkk_fits(av_Mkk, inv_Mkk, ifieldA, inst, dat_type='real')
        mkk_savepath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/mkk/'+mask_tail+'/mkk_'+mkk_type+'_ifield'+str(ifieldA)+'_ifield'+str(ifieldB)+'_observed_'+mask_tail+'.fits'
        
        print('Saving mkk to ', mkk_savepath)
        mkk_hdul.writeto(mkk_savepath, overwrite=True)
        
        mkk_savepath_list.append(mkk_savepath)

        if save_comb_mask:
            hdul = write_mask_file(np.array(maskA*maskB), ifield=ifieldA, inst=inst, generate_galmask=True, \
                                  generate_starmask=True, use_inst_mask=True, dat_type='observed', mag_lim_AB=mag_lim)

            mask_save_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/masks/'+mask_tail+'/joint_mask_ifield'+str(ifieldA)+'_ifield'+str(ifieldB)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'

            print('Saving mask to ', mask_save_fpath)
            hdul.writeto(mask_save_fpath, overwrite=True)
            mask_savepath_list.append(mask_save_fpath)
            
    return mkk_savepath_list, mask_savepath_list


def compute_mkk_fielddiff_previous_flight_wrapper(inst, n_mkk_sim=100, n_split=2, flight_nos = [36265, 36277, 36277, 36277], ifield_target_list = [7, 6, 6, 8], \
                                                 ifield_diff_list = [3, 3, 8, 3], fc_sub=False, fc_sub_quad_offset=False, fc_sub_n_terms=2, grad_sub=True):
    
    
    mkk_save_fpaths = []
    
    for diffidx, flight in enumerate(flight_nos):

        diff_mask = fits.open(config.ciber_basepath+'data/previous_flights/'+str(flight)+'/field_diff_mask/diff_mask_regrid_TM'+str(inst)+'_ifield'+str(ifield_target_list[diffidx])+'_ifield'+str(ifield_diff_list[diffidx])+'_'+str(flight)+'.FITS')[1].data

        plot_map(diff_mask)

        print('Fraction of unmasked pixels:', np.sum(diff_mask)/1024**2)
        av_Mkk = cbps.Mkk_obj.get_mkk_sim(diff_mask, n_mkk_sim, n_split=n_split, fc_sub=fc_sub, fc_sub_quad_offset=fc_sub_quad_offset, \
                                         fc_sub_n_terms=fc_sub_n_terms, grad_sub=grad_sub)
        
        inv_Mkk = compute_inverse_mkk(av_Mkk)

        mkk_hdul = write_Mkk_fits(av_Mkk, inv_Mkk, ifield_target_list[diffidx], inst, dat_type='real')

        mkk_basepath = config.ciber_basepath+'data/previous_flights/'+str(flight)+'/mkk/'
        mkk_savepath = mkk_basepath+'mkk_estimate_flight_'+str(flight)+'_ifield'+str(ifield_target_list[diffidx])+'_ifield_'+str(ifield_diff_list[diffidx])+'_diff.fits'

        plot_mkk_matrix(av_Mkk)
        mkk_hdul.writeto(mkk_savepath, overwrite=True)
        
        mkk_save_fpaths.append(mkk_savepath)
        
    return mkk_save_fpaths



def compute_mkk_mats_observed_dat_wrapper(cbps, inst, mag_lim_list, mkk_mask_type, mkk_ffest_type, ifield_list = [4, 5, 6, 7, 8], mask_tail_list=None, datestr='111323',\
                                          mag_lim_wise=16.0, plot=True, \
                                         n_mkk_sim=200, n_mkk_split=20, niter=1,\
                                          quadoff_grad=False, fc_sub=False, fc_sub_quad_offset=False, fc_sub_n_terms=2, grad_sub=True, \
                                         ff_transition_mag=15.0, mode='auto', cross_inst=None):
    
    # cbps = CIBER_PS_pipeline()
    config_dict, pscb_dict, float_param_dict, fpath_dict = return_default_cbps_dicts()
    ciber_mock_fpath = config.exthdpath+'ciber_mocks/'
    datestr_trilegal = datestr
    fpath_dict, list_of_dirpaths, base_path, trilegal_base_path = set_up_filepaths_cbps(fpath_dict, inst, 'test', datestr,\
                                                                                        datestr_trilegal=datestr, data_type='observed', \
                                                                                       save_fpaths=True)
    
    bandstr = cbps.bandstr_dict[inst]
    mean_norms = [cbps.zl_levels_ciber_fields[inst][cbps.ciber_field_dict[ifield]] for ifield in ifield_list]
    ff_weights = cbps.compute_ff_weights(inst, mean_norms, ifield_list, photon_noise=True, read_noise_models=None)

    
    save_fpath_list = []
    
    for m, mag_lim in enumerate(mag_lim_list):
        
        if mask_tail_list is None:
            mask_tail_obs = 'maglim_'+bandstr+'_Vega_'+str(mag_lim)+'_111323_ukdebias'
        else:
            mask_tail_obs = mask_tail_list[m]
            
        make_fpaths([fpath_dict['mkk_ffest_base_path']+'/'+mask_tail_obs])   

        masks = np.zeros((len(ifield_list), cbps.dimx, cbps.dimy))

        for fieldidx, ifield in enumerate(ifield_list):

            mask_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/masks/'+mask_tail_obs+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail_obs+'.fits'  
            fullmask = fits.open(mask_fpath)[1].data    
            print('full mask has fraction ', np.count_nonzero(fullmask)/1024**2)
            masks[fieldidx] = fullmask
            if plot:
                plot_map(masks[fieldidx], title='mask')
                
        mask_fractions = np.array([float(np.sum(masks[fieldidx]))/float(cbps.dimx**2) for fieldidx in range(len(ifield_list))])
        print('mask fractions are ', mask_fractions)
        
        if mag_lim <= ff_transition_mag:
            print('for maglim '+str(mag_lim)+', calculating mask+filtering only matrix')
            average_Mkks_perfield = []
            for fieldidx, ifield in enumerate(ifield_list):
                
                av_Mkk = cbps.Mkk_obj.get_mkk_sim(masks[fieldidx], n_mkk_sim, n_split=n_mkk_sim//50, fc_sub=fc_sub, fc_sub_quad_offset=fc_sub_quad_offset, fc_sub_n_terms=fc_sub_n_terms, \
                                                 grad_sub=grad_sub)
                average_Mkks_perfield.append(av_Mkk)
        else:
            print('for maglim '+str(mag_lim)+', calculating mask+FF+filtering matrix')

            average_Mkks_perfield, Mkks_per_field = estimate_mkk_ffest_quadoff(cbps, n_mkk_sim, masks, n_split=n_mkk_split, ifield_list=ifield_list, mean_normalizations=mean_norms, ff_weights=ff_weights, \
                                   verbose=False, fc_sub=fc_sub, quadoff_grad=quadoff_grad, fc_sub_quad_offset=fc_sub_quad_offset, fc_sub_n_terms=fc_sub_n_terms, niter=niter, grad_sub=grad_sub)


        save_fpath_list_indiv_mag = []
        
        for fieldidx, ifield in enumerate(ifield_list):

            inv_Mkk = compute_inverse_mkk(average_Mkks_perfield[fieldidx])
            mkk_hdul = write_Mkk_fits(average_Mkks_perfield[fieldidx], inv_Mkk, ifield, inst, dat_type='observed', mag_lim_AB=mag_lim+cbps.Vega_to_AB[inst])    
            
            if mag_lim <= ff_transition_mag:
                mkkonly_basepath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/mkk/'+mask_tail_obs+'/'
                mkk_savepath = mkkonly_basepath+'mkk_'+mkk_mask_type+'_estimate_ifield'+str(ifield)+'_observed_'+mask_tail_obs+'.fits'
            else:
                mkk_savepath = fpath_dict['mkk_ffest_base_path']+'/'+mask_tail_obs+'/mkk_'+mkk_ffest_type+'_ifield'+str(ifield)+'_observed_'+mask_tail_obs+'.fits'
            
            print('saving field ', ifield, 'to', mkk_savepath)
            mkk_hdul.writeto(mkk_savepath, overwrite=True)
            
            save_fpath_list_indiv_mag.append(mkk_savepath)
            
        save_fpath_list.append(save_fpath_list_indiv_mag)
        
    return save_fpath_list



def compute_cross_mkk_wrapper(inst, cross_inst, mag_lim_list, mask_tail_list=None, n_mkk_sims=200, n_mkk_split=20, ifield_list=[4, 5, 6, 7, 8], \
                             ff_transition_mag=15.0, mkk_mask_type='quadoff_grad_fcsub_order2', mkk_ffest_type='ffest_quadoff_grad_fcsub_order2', \
                             fc_sub=True, fc_sub_quad_offset=True, fc_sub_n_terms=2, grad_sub=True):
    
    cbps = CIBER_PS_pipeline()
    
    base_fluc_path = config.ciber_basepath+'data/fluctuation_data/'
    mask_base_path = base_fluc_path+'TM'+str(inst)+'/masks/'
    mask_base_path_cross_inst = base_fluc_path+'TM'+str(cross_inst)+'/masks/'
    mask_base_path_cross_union = base_fluc_path+'TM'+str(inst)+'_TM'+str(cross_inst)+'_cross/masks/'
    
    mean_norms = [cbps.zl_levels_ciber_fields[inst][cbps.ciber_field_dict[ifield]] for ifield in ifield_list]
    mean_norms_cross = [cbps.zl_levels_ciber_fields[cross_inst][cbps.ciber_field_dict[ifield]] for ifield in ifield_list]

    ff_weights = cbps.compute_ff_weights(inst, mean_norms, ifield_list)
    ff_weights_cross = cbps.compute_ff_weights(cross_inst, mean_norms_cross, ifield_list)
    
    all_mkk_savepath = []

    for m, mag_lim in enumerate(mag_lim_list):
        
        if mask_tail_list is None:
            cross_union_mask_tail = 'maglim_Vega_J='+str(mag_lim)+'_H='+str(mag_lim-0.5)+'_ukdebias_111323'
        else:
            cross_union_mask_tail = mask_tail_list[m]
        
        if mag_lim <= ff_transition_mag:
            mkkmode = 'mkk'
            mkk_savetype = mkk_mask_type
        else:
            mkkmode = 'mkk_ffest'
            mkk_savetype = mkk_ffest_type
    
        mkk_base_path = config.ciber_basepath+'data/fluctuation_data/TM1_TM2_cross/'+mkkmode+'/'+cross_union_mask_tail
        
        make_fpaths([mkk_base_path])

        cross_union_masks = np.zeros((len(ifield_list), cbps.dimx, cbps.dimy))
    
        for fieldidx, ifield in enumerate(ifield_list):
            
            union_mask_fpath = mask_base_path_cross_union+cross_union_mask_tail+'/joint_mask_ifield'+str(ifield)+'_TM'+str(inst)+'_TM'+str(cross_inst)+'_observed_'+cross_union_mask_tail+'.fits'    
            cross_union_masks[fieldidx] = fits.open(union_mask_fpath)[1].data

            plot_map(cross_union_masks[fieldidx], title='cross mask fieldidx = '+str(fieldidx))


        average_mkk_cross_list = []
        
        if mkkmode=='mkk_ffest':
            
            average_mkk_cross_list, mkkffests_tm1tm2_perf = estimate_mkk_ffest_cross(cbps, n_mkk_sims, cross_union_masks, n_split=n_mkk_split, mean_normalizations=mean_norms, ff_weights=ff_weights, \
                                         mean_normalizations_cross=mean_norms_cross, ff_weights_cross=ff_weights_cross, \
                                       ifield_list=ifield_list, verbose=False, plot=False, fc_sub=fc_sub, fc_sub_quad_offset=fc_sub_quad_offset, fc_sub_n_terms=fc_sub_n_terms, \
                                                                                    grad_sub=grad_sub)
        else:
            for fieldidx, ifield in enumerate(ifield_list):
                av_Mkk = cbps.Mkk_obj.get_mkk_sim(cross_union_masks[fieldidx], n_mkk_sims, n_split=n_mkk_sims//50, fc_sub=fc_sub, fc_sub_quad_offset=fc_sub_quad_offset, fc_sub_n_terms=fc_sub_n_terms, grad_sub=grad_sub)
                average_mkk_cross_list.append(av_Mkk)
                
        
        for fieldidx, ifield in enumerate(ifield_list):

            inv_Mkk = compute_inverse_mkk(average_mkk_cross_list[fieldidx])
            mkk_hdul = write_Mkk_fits(average_mkk_cross_list[fieldidx], inv_Mkk, ifield, inst, cross_inst=cross_inst, dat_type='observed_cross')
            mkk_savepath = mkk_base_path+'/mkk_'+mkk_savetype+'_ifield'+str(ifield)+'_observed_'+cross_union_mask_tail+'.fits'

            print('saving field ', ifield, 'to', mkk_savepath)
            mkk_hdul.writeto(mkk_savepath, overwrite=True)
            
            all_mkk_savepath.append(mkk_savepath)
            
    return all_mkk_savepath
