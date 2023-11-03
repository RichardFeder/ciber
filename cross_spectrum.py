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

def proc_cibermap_regrid(cbps, inst, regrid_to_inst, mask_tail, ifield_list=[4, 5, 6, 7, 8], datestr='112022', \
                        niter=5, nitermax=1, sig=5, ff_min=0.5, ff_max=1.5, astr_dir='../../ciber/data/', \
                        save=True):
    
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
    
    for fieldidx, ifield in enumerate(ifield_list):
        cbps.load_flight_image(ifield, inst, verbose=True, ytmap=False)
        observed_ims[fieldidx] = cbps.image*cbps.cal_facs[inst]
        observed_ims[fieldidx] -= dc_template*cbps.cal_facs[inst]

        mask_fpath = fpath_dict['mask_base_path']+'/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'.fits'
        masks[fieldidx] = fits.open(mask_fpath)[1].data

        sigclip = iter_sigma_clip_mask(observed_ims[fieldidx], sig=sig, nitermax=nitermax, mask=masks[fieldidx].astype(np.int))
        masks[fieldidx] *= sigclip
        plot_map(masks[fieldidx]*observed_ims[fieldidx])
        
      
    ciber_maps_byquad = [observed_ims[:, cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]] for q in range(4)]

    for q, quad in enumerate(ciber_maps_byquad):
        masks_quad = masks[:,cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]]

        processed_ciber_maps_quad, ff_estimates_quad,\
            final_planes, stack_masks, ff_weights = process_ciber_maps(cbps, ifield_list, inst, ciber_maps_byquad[q], masks_quad, nitermax=nitermax, niter=niter)

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


# _ecliptic = np.radians(23.43927944)
    
# def equatorial_to_ecliptic(ra, dec):

#     ec_latitude = np.arcsin(np.sin(dec) * np.cos(_ecliptic))
#     ec_latitude = ec_latitude - (np.cos(dec) * np.sin(_ecliptic) * np.sin(ra))
#     _num, _den = np.sin(ra) * np.cos(_ecliptic) + np.tan(dec) * np.sin(_ecliptic), np.cos(ra)
#     ec_longitude = np.arcsin(_num / _den)
#     print('ec long:', ec_longitude)
#     ec_latitude, ec_longitude = np.degrees([ec_latitude, ec_longitude])
#     return ec_latitude, ec_longitude


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


def ciber_ciber_rl_coefficient(obs_name_A, obs_name_B, obs_name_AB, startidx=1, endidx=-1):

    
    obs_fieldav_cls, obs_fieldav_dcls = [], []
    inst_list = [1, 2]
    for clidx, obs_name in enumerate([obs_name_A, obs_name_B, obs_name_AB]):
        if clidx<2:
            cl_fpath_obs = 'data/input_recovered_ps/cl_files/TM'+str(inst_list[clidx])+'/cl_'+obs_name+'.npz'
            lb, observed_recov_ps, observed_recov_dcl_perfield,\
            observed_field_average_cl, observed_field_average_dcl,\
                mock_all_field_cl_weights = load_weighted_cl_file(cl_fpath_obs)     
        else:
            cl_fpath_obs = 'data/input_recovered_ps/cl_files/TM1_TM2_cross/cl_'+obs_name+'.npz'
            lb, observed_recov_ps, observed_recov_dcl_perfield,\
            observed_field_average_cl, observed_field_average_dcl,\
                mock_all_field_cl_weights = load_weighted_cl_file_cross(cl_fpath_obs)

        obs_fieldav_cls.append(observed_field_average_cl)
        obs_fieldav_dcls.append(observed_field_average_dcl)

    r_TM = obs_fieldav_cls[2]/np.sqrt(obs_fieldav_cls[0]*obs_fieldav_cls[1])

    sigma_r_TM = np.sqrt((1./(obs_fieldav_cls[0]*obs_fieldav_cls[1]))*(obs_fieldav_dcls[2]**2 + (obs_fieldav_cls[2]*obs_fieldav_dcls[0]/(2*obs_fieldav_cls[0]))**2+(obs_fieldav_cls[2]*obs_fieldav_dcls[1]/(2*obs_fieldav_cls[1]))**2))

    return lb, r_TM, sigma_r_TM

def simulate_deep_cats_correlation_coeff_cosmos(inst=1, ifield_choose = 4, include_IRAC_mask=False, maglim_IRAC=18., m_max=28, \
                                               inv_Mkk=None, mkk_correct=True, coverage_mask=None):

    Vega_to_AB = dict({'g':-0.08, 'r':0.16, 'i':0.37, 'z':0.54, 'y':0.634, 'J':0.91, 'H':1.39, 'K':1.85, \
                      'CH1':2.699, 'CH2':3.339})
   
    nx, ny = 650, 650
    startidx, endidx = 1, -1
    
    cbps = CIBER_PS_pipeline(dimx=nx, dimy=ny)

    subpixel_psf_dirpath = config.exthdpath+'ciber_fluctuation_data/TM'+str(inst)+'/subpixel_psfs/'
    lb, mean_bl, bls = cbps.compute_beam_correction_posts(ifield_choose, inst)    
    
    cmock = ciber_mock(nx=nx, ny=ny)

    m_min_J_list = [17.5, 18.5, 19.5, 20.5, 21.5]
    m_min_H_list = [17.0, 18.0, 19.0, 20.0, 21.0]
    m_min_CH1_list = [16.0, 17.0, 18.0, 19.0]
    m_min_CH2_list = [16.0, 17.0, 18.0, 19.0]

    cosmos_catalog = np.load('data/cosmos/cosmos20_farmer_catalog_wxy_073123.npz')

    # these are all AB mags from Farmer catalog
    cosmos_xpos = cosmos_catalog['cosmos_xpos']
    cosmos_ypos = cosmos_catalog['cosmos_ypos']
    cosmos_J_mag = cosmos_catalog['cosmos_J_mag']
    cosmos_H_mag = cosmos_catalog['cosmos_H_mag']
    cosmos_CH1_mag = cosmos_catalog['cosmos_CH1_mag']
    cosmos_CH2_mag = cosmos_catalog['cosmos_CH2_mag']
    
    plt.figure()
    plt.hist(cosmos_J_mag, bins=np.linspace(16, 26, 30), histtype='step', label='J')
    plt.hist(cosmos_H_mag, bins=np.linspace(16, 26, 30), histtype='step', label='H')
    plt.hist(cosmos_CH1_mag, bins=np.linspace(16, 26, 30), histtype='step', label='CH1')
    plt.hist(cosmos_CH2_mag, bins=np.linspace(16, 26, 30), histtype='step', label='CH2')
    plt.yscale('log')
    plt.legend()
    plt.show()

    cosmos_bordermask = (cosmos_xpos < nx)*(cosmos_ypos < ny)*(cosmos_xpos > 0)*(cosmos_ypos > 0)
    
    if inv_Mkk is None and mkk_correct:
        H, xedges, yedges = np.histogram2d(cosmos_xpos, cosmos_ypos, bins=[np.linspace(0, nx, nx+1), np.linspace(0, ny, ny+1)])

        coverage_mask = (H != 0)

        H_mask = H*coverage_mask
        plt.figure()
        plt.imshow(H_mask, origin='lower')
        plt.colorbar()
        plt.show()

        print('coverage_mask has shape ', coverage_mask.shape)
        plt.figure()
        plt.imshow(coverage_mask, origin='lower')
        plt.colorbar()
        plt.show()
        
        av_Mkk = cbps.Mkk_obj.get_mkk_sim(coverage_mask, 100, n_split=1)
        
        inv_Mkk = save_mkks('data/cosmos/Mkk_file_C20_coverage.npz', av_Mkk=av_Mkk, return_inv_Mkk=True, mask=coverage_mask)
    
    
#     coverage_mask = H_mask
    
    all_clauto_J, all_clauto_H, all_clauto_CH1, all_clauto_CH2 = [[] for x in range(4)]
    all_clx_JH, all_clx_J_CH1, all_clx_J_CH2, all_clx_H_CH1, all_clx_H_CH2, all_clx_CH1_CH2 = [[] for x in range(6)]
    
    all_clautos = []
    all_clcrosses = []

    for magidx in range(len(m_min_J_list)):

        magmask_J = (cosmos_J_mag-Vega_to_AB['J'] > m_min_J_list[magidx])*(cosmos_J_mag-Vega_to_AB['J'] < m_max)
        magmask_H = (cosmos_H_mag-Vega_to_AB['H'] > m_min_H_list[magidx])*(cosmos_H_mag-Vega_to_AB['H'] < m_max)
        
        magmask = magmask_J*magmask_H
        
        if include_IRAC_mask:
            print('adding IRAC mask L < '+str(maglim_IRAC))
            magmask *= (cosmos_CH1_mag-Vega_to_AB['CH1'] > maglim_IRAC)
        
        mask = cosmos_bordermask*magmask
        
    
        I_arr_J = cmock.mag_2_nu_Inu(cosmos_J_mag, 0)
        I_arr_H = cmock.mag_2_nu_Inu(cosmos_H_mag, 1)
        I_arr_CH1 = cmock.mag_2_nu_Inu(cosmos_CH1_mag, band=None, lam_eff=3.6*1e-6*u.m)
        I_arr_CH2 = cmock.mag_2_nu_Inu(cosmos_CH2_mag, band=None, lam_eff=4.5*1e-6*u.m)
        
        I_arr_J[np.isnan(I_arr_J)] = 0.
        I_arr_H[np.isnan(I_arr_H)] = 0.
        I_arr_CH1[np.isnan(I_arr_CH1)] = 0.
        I_arr_CH2[np.isnan(I_arr_CH2)] = 0.
        
        mock_cat_J = np.array([cosmos_xpos[mask], cosmos_ypos[mask], cosmos_J_mag[mask], I_arr_J[mask]]).transpose()
        mock_cat_H = np.array([cosmos_xpos[mask], cosmos_ypos[mask], cosmos_H_mag[mask], I_arr_H[mask]]).transpose()
        mock_cat_CH1 = np.array([cosmos_xpos[mask], cosmos_ypos[mask], cosmos_CH1_mag[mask], I_arr_CH1[mask]]).transpose()
        mock_cat_CH2 = np.array([cosmos_xpos[mask], cosmos_ypos[mask], cosmos_CH2_mag[mask], I_arr_CH2[mask]]).transpose()
        
        sourcemap_J = cmock.make_srcmap_temp_bank(ifield_choose, inst, mock_cat_J, flux_idx=-1, n_fine_bin=10, nwide=17,tempbank_dirpath=subpixel_psf_dirpath, load_precomp_tempbank=True)
        sourcemap_H = cmock.make_srcmap_temp_bank(ifield_choose, inst, mock_cat_H, flux_idx=-1, n_fine_bin=10, nwide=17,tempbank_dirpath=subpixel_psf_dirpath, load_precomp_tempbank=True)
        sourcemap_CH1 = cmock.make_srcmap_temp_bank(ifield_choose, inst, mock_cat_CH1, flux_idx=-1, n_fine_bin=10, nwide=17,tempbank_dirpath=subpixel_psf_dirpath, load_precomp_tempbank=True)
        sourcemap_CH2 = cmock.make_srcmap_temp_bank(ifield_choose, inst, mock_cat_CH2, flux_idx=-1, n_fine_bin=10, nwide=17,tempbank_dirpath=subpixel_psf_dirpath, load_precomp_tempbank=True)        
        
        sourcemap_J *= coverage_mask
        sourcemap_H *= coverage_mask
        sourcemap_CH1 *= coverage_mask
        sourcemap_CH2 *= coverage_mask
        
        sourcemap_J[coverage_mask != 0] -= np.mean(sourcemap_J[coverage_mask != 0])
        sourcemap_H[coverage_mask != 0] -= np.mean(sourcemap_H[coverage_mask != 0])
        sourcemap_CH1[coverage_mask != 0] -= np.mean(sourcemap_CH1[coverage_mask != 0])
        sourcemap_CH2[coverage_mask != 0] -= np.mean(sourcemap_CH2[coverage_mask != 0])
        
        
        # autos
        lb, clauto_J, clerr_auto_J = get_power_spec(sourcemap_J, \
                                                 lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
        lb, clauto_H, clerr_auto_H = get_power_spec(sourcemap_H, \
                                                 lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
        lb, clauto_CH1, clerr_auto_CH1 = get_power_spec(sourcemap_CH1, \
                                                 lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
        lb, clauto_CH2, clerr_auto_CH2 = get_power_spec(sourcemap_CH2, \
                                                 lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
        
        cl_autos = [clauto_J, clauto_H, clauto_CH1, clauto_CH2]
        
        # now compute crosses
        lb, clx_J_H, clerr_x_J_H = get_power_spec(sourcemap_J, map_b=sourcemap_H, \
                                 lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
        lb, clx_J_CH1, clerr_x_J_CH1 = get_power_spec(sourcemap_J, map_b=sourcemap_CH1, \
                                 lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
        lb, clx_J_CH2, clerr_x_J_CH2 = get_power_spec(sourcemap_J, map_b=sourcemap_CH2, \
                                 lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
        lb, clx_H_CH1, clerr_x_H_CH1 = get_power_spec(sourcemap_H, map_b=sourcemap_CH1, \
                         lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
        lb, clx_H_CH2, clerr_x_H_CH2 = get_power_spec(sourcemap_H, map_b=sourcemap_CH2, \
                 lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
        lb, clx_CH1_CH2, clerr_x_CH1_CH2 = get_power_spec(sourcemap_CH1, map_b=sourcemap_CH2, \
                 lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
        
        cl_crosses = [clx_J_H, clx_J_CH1, clx_J_CH2, clx_H_CH1, clx_H_CH2, clx_CH1_CH2]

        for clidx, cl in enumerate(cl_autos):
            if mkk_correct:
                cl_autos[clidx] = np.dot(inv_Mkk.transpose(), cl)
            cl_autos[clidx] = cl_autos[clidx] / mean_bl**2
            
            
        for clidx, clx in enumerate(cl_crosses):
            if mkk_correct:
                cl_crosses[clidx] = np.dot(inv_Mkk.transpose(), clx)
            cl_crosses[clidx] = cl_crosses[clidx] / mean_bl**2
    
    
        r_ell_J_H = cl_crosses[0]/np.sqrt(cl_autos[0]*cl_autos[1])
        r_ell_J_CH1 = cl_crosses[1]/np.sqrt(cl_autos[0]*cl_autos[2])
        r_ell_J_CH2 = cl_crosses[2]/np.sqrt(cl_autos[0]*cl_autos[3])
        
        r_ell_H_CH1 = cl_crosses[3]/np.sqrt(cl_autos[1]*cl_autos[2])
        r_ell_H_CH2 = cl_crosses[4]/np.sqrt(cl_autos[1]*cl_autos[3])
        r_ell_CH1_CH2 = cl_crosses[5]/np.sqrt(cl_autos[2]*cl_autos[3])
        
        plt.figure()
        plt.plot(lb[startidx:endidx], r_ell_J_H[startidx:endidx], label='J x H')
        plt.plot(lb[startidx:endidx], r_ell_J_CH1[startidx:endidx], label='J x CH1')
        plt.plot(lb[startidx:endidx], r_ell_J_CH2[startidx:endidx], label='J x CH2')
        
        plt.plot(lb[startidx:endidx], r_ell_H_CH1[startidx:endidx], label='H x CH1')
        plt.plot(lb[startidx:endidx], r_ell_H_CH2[startidx:endidx], label='H x CH2')
        plt.plot(lb[startidx:endidx], r_ell_CH1_CH2[startidx:endidx], label='CH1 x CH2')
        plt.xscale('log')
        plt.ylim(0, 1.3)
        plt.xlabel('$\\ell$', fontsize=16)
        plt.ylabel('$r_{\\ell}$', fontsize=16)
        plt.legend(fontsize=12, ncol=2, loc=3)
        
        textstr = 'COSMOS field\nMask $J<$'+str(m_min_J_list[magidx])+' $\\cup$ $H<$'+str(m_min_H_list[magidx])
        
        if include_IRAC_mask:
            textstr += ' $\\cup$ $L<$'+str(maglim_IRAC)
            
        plt.text(400, 1.05, textstr, color='k', fontsize=16)
        plt.grid()
        
        if include_IRAC_mask:
            plt.savefig('/Users/richardfeder/Downloads/r_ell_cosmos_JHCH1CH2_mask_Jlt'+str(m_min_J_list[magidx])+'_Hlt'+str(m_min_H_list[magidx])+'_CH1lt'+str(maglim_IRAC)+'.png', bbox_inches='tight', dpi=200)
        else:
            plt.savefig('/Users/richardfeder/Downloads/r_ell_cosmos_JHCH1CH2_mask_Jlt'+str(m_min_J_list[magidx])+'_Hlt'+str(m_min_H_list[magidx])+'.png', bbox_inches='tight', dpi=200)
        plt.show()
        
        prefac = lb*(lb+1)/(2*np.pi)
        
        plt.figure()
        plt.plot(lb, prefac*cl_autos[0], label='J')
        plt.plot(lb, prefac*cl_autos[1], label='H')
        plt.plot(lb, prefac*cl_autos[2], label='CH1')
        plt.plot(lb, prefac*cl_autos[3], label='CH2')
        plt.legend(loc=3, ncol=2)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        
        all_clautos.append(cl_autos)
        all_clcrosses.append(cl_crosses)

        all_clauto_J.append(cl_autos[0])
        all_clauto_H.append(cl_autos[1])
        all_clauto_CH1.append(cl_autos[2])
        all_clauto_CH2.append(cl_autos[3])
        
        all_clx_JH.append(cl_crosses[0])
        all_clx_J_CH1.append(cl_crosses[1])
        all_clx_J_CH2.append(cl_crosses[2])
        all_clx_H_CH1.append(cl_crosses[3])
        all_clx_H_CH2.append(cl_crosses[4])
        all_clx_CH1_CH2.append(cl_crosses[5])
        
    return lb, all_clautos, all_clcrosses
