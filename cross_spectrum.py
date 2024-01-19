import numpy as np
from astropy.io import fits
from plotting_fns import plot_map
from reproject import reproject_interp

import config
from ciber_powerspec_pipeline import *
from ciber_mocks import *
from mock_galaxy_catalogs import *
from lognormal_counts import *
from ciber_data_helpers import *
from helgason import *
from ps_pipeline_go import *
from noise_model import *
from cross_spectrum_analysis import *
from mkk_parallel import *
from mkk_diagnostics import *
from flat_field_est import *
from plotting_fns import *
from powerspec_utils import *
from astropy.coordinates import SkyCoord
import healpy as hp
import scipy.io


def convert_MJysr_to_nWm2sr(lam_micron):
    
    lam_angstrom = lam_micron*1e4
    print(lam_angstrom)
    c = 2.9979e18 #A/s

    fac = 1e6 # to get to Jy from MJy
    fac *= 1e-26
    fac *= c/(lam_angstrom*lam_angstrom)
    
    fac *= 1e9
        
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

def proc_cibermap_regrid(cbps, inst, regrid_to_inst, mask_tail, ifield_list=[4, 5, 6, 7, 8], datestr='112022', \
                        niter=5, nitermax=1, sig=5, ff_min=0.5, ff_max=1.5, astr_dir='../../ciber/data/', \
                        save=True, mask_tail_ffest=None):
    
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
            plot_map(masks[fieldidx]*observed_ims[fieldidx], title='map for ff estimation')

        
      
    ciber_maps_byquad = [observed_ims[:, cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]] for q in range(4)]

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
    
    for fieldidx, ifield in enumerate(ifield_list):
        
        obs_level = np.mean(processed_ims[fieldidx][masks[fieldidx]==1])
        print('obs level:', obs_level)
        
        for q in range(4):
            mquad = masks[fieldidx][cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]]
            processed_ims[fieldidx][cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]][mquad==1] -= np.mean(processed_ims[fieldidx][cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]][mquad==1])

        masked_proc = processed_ims[fieldidx]*masks[fieldidx]
        plot_map(masked_proc, cmap='bwr')

        all_masked_proc.append(masked_proc)
    
        ciber_regrid, ciber_fp_regrid = regrid_arrays_by_quadrant(masked_proc, ifield, inst0=regrid_to_inst, inst1=inst, \
                                                             plot=False, astr_dir=astr_dir)
    
        plot_map(ciber_regrid, title='regrid')
        plot_map(masked_proc, title='before regridding')
        
        print('mean of regrid is ', np.mean(ciber_regrid))
        
        all_regrid_masked_proc.append(ciber_regrid)
        
        
        if save:
            if fieldidx==0:
                proc_regrid_basepath = fpath_dict['observed_base_path']+'/TM'+str(regrid_to_inst)+'_TM'+str(inst)+'_cross/proc_regrid/'
                proc_regrid_basepath += mask_tail+'/'
                make_fpaths([proc_regrid_basepath])
            
            regrid_fpath = proc_regrid_basepath+'proc_regrid_TM'+str(inst)+'_to_TM'+str(regrid_to_inst)+'_ifield'+str(ifield)+'_'+mask_tail+'.fits'
            hdul = write_regrid_proc_file(ciber_regrid, ifield, inst, regrid_to_inst, mask_tail=mask_tail, \
                                         obs_level=obs_level)
            print('saving to ', regrid_fpath)
            hdul.writeto(regrid_fpath, overwrite=True)
        
    return all_regrid_masked_proc

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

def regrid_arrays_by_quadrant(map1, ifield, inst0=1, inst1=2, quad_list=['A', 'B', 'C', 'D'], \
                             xoff=[0,0,512,512], yoff=[0,512,0,512], astr_map0_hdrs=None, astr_map1_hdrs=None, indiv_map0_hdr=None, indiv_map1_hdr=None, astr_dir=None, \
                             plot=True, order=0):
    
    ''' 
    Used for regridding maps from one imager to another. For the CIBER1 imagers the 
    astrometric solution is computed for each quadrant separately, so this function iterates
    through the quadrants when constructing the full regridded images. 
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''

    if astr_dir is None:
        astr_dir = '../../ciber/data/'

    ciber_field_dict = dict({4:'elat10', 5:'elat30',6:'BootesB', 7:'BootesA', 8:'SWIRE', 'train':'UDS'})

    map1_regrid, map1_fp_regrid = [np.zeros_like(map1) for x in range(2)]

    fieldname = ciber_field_dict[ifield]
    
    map1_quads = [map1[yoff[iquad]:yoff[iquad]+512, xoff[iquad]:xoff[iquad]+512] for iquad in range(len(quad_list))]
    
    if astr_map0_hdrs is None and indiv_map0_hdr is None:
        print('loading WCS for inst = ', inst0)
        astr_map0_hdrs = load_quad_hdrs(ifield, inst0, base_path=astr_dir, halves=False)
    if astr_map1_hdrs is None and indiv_map1_hdr is None:
        print('loading WCS for inst = ', inst1)
        astr_map1_hdrs = load_quad_hdrs(ifield, inst1, base_path=astr_dir, halves=False)

    # if astr_map0_hdrs is None and indiv_map0_hdr is None:
    #     astr_map0_hdrs = [fits.open(astr_dir+'inst'+str(inst0)+'/'+fieldname+'_'+quad+'_astr.fits')[0].header for quad in quad_list]
    # if astr_map1_hdrs is None and indiv_map1_hdr is None:
    #     astr_map1_hdrs = [fits.open(astr_dir+'inst'+str(inst1)+'/'+fieldname+'_'+quad+'_astr.fits')[0].header for quad in quad_list]
    # loop over quadrants of first imager
    
    for iquad, quad in enumerate(quad_list):
        
        # arrays, footprints = [], []
        run_sum_footprint, sum_array = [np.zeros_like(map1_quads[0]) for x in range(2)]

        # reproject each quadrant of second imager onto first imager
        if indiv_map0_hdr is None:
            for iquad2, quad2 in enumerate(quad_list):
                input_data = (map1_quads[iquad2], astr_map1_hdrs[iquad2])
                array, footprint = reproject_interp(input_data, astr_map0_hdrs[iquad], (512, 512), order=order)

                array[np.isnan(array)] = 0.
                footprint[np.isnan(footprint)] = 0.

                run_sum_footprint += footprint 
                sum_array[run_sum_footprint < 2] += array[run_sum_footprint < 2]
                run_sum_footprint[run_sum_footprint > 1] = 1

                # arrays.append(array)
                # footprints.append(footprint)
       
        # sumarray = np.nansum(arrays, axis=0)
        # sumfootprints = np.nansum(footprints, axis=0)

        if plot:
            plot_map(sum_array, title='sum array')
            plot_map(run_sum_footprint, title='sum footprints')
        
        # print('number of pixels with > 1 footprint', np.sum((sumfootprints==2)))
        # map1_regrid[yoff[iquad]:yoff[iquad]+512, xoff[iquad]:xoff[iquad]+512] = sumarray
        # map1_fp_regrid[yoff[iquad]:yoff[iquad]+512, xoff[iquad]:xoff[iquad]+512] = sumfootprints

        map1_regrid[yoff[iquad]:yoff[iquad]+512, xoff[iquad]:xoff[iquad]+512] = sum_array
        map1_fp_regrid[yoff[iquad]:yoff[iquad]+512, xoff[iquad]:xoff[iquad]+512] = run_sum_footprint

    return map1_regrid, map1_fp_regrid

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

# powerspec_utils.py
# def ciber_ciber_rl_coefficient(obs_name_A, obs_name_B, obs_name_AB, startidx=1, endidx=-1):


