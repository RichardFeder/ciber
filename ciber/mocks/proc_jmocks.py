import matplotlib
import matplotlib.pyplot as plt
from ciber.mocks.cib_mocks import *

import numpy as np
from scipy import interpolate
import os
import astropy
import astropy.wcs as wcs
from astropy.io import fits


import config
from ciber.core.powerspec_pipeline import *
from ciber.core.ps_pipeline_go import *
from ciber.theory.cl_predictions import *


from ciber.io.catalog_utils import *
from ciber.core.powerspec_pipeline import *
from ciber.cross_correlation.galaxy_cross import *


def grab_mock_signal_components(rlz, inst=1, galstr='hsc_i_lt_25.0', jmock_basedir = 'data/jordan_mocks/v2/', \
                                load_tracer_cat=True, counts_from_xy=True, zstr=None, load_gal_counts=True, tailstr='CIBERfidmask', 
                                with_redshift=True):
    
    # file paths
    # gal_map_fpath = jmock_basedir+'mock_maps/galaxy/TM'+str(inst)+'/rlz'+str(rlz)+'_TM'+str(inst)+'_'+galstr+'_CIBERfidmask_zmax=1.0_galaxy.npz'
    # intensity_map_fpath = jmock_basedir+'mock_maps/intensity/TM'+str(inst)+'/rlz'+str(rlz)+'_TM'+str(inst)+'_CIBERfidmask_full_intensity.npz'
    # true_cross_fpath = jmock_basedir+'mock_ps_pred/TM'+str(inst)+'/indiv/rlz'+str(rlz)+'_TM'+str(inst)+'_auto_cross_pred_'+galstr+'_CIBERfidmask_zmax=1.0.npz'

    intensity_map_fpath = jmock_basedir+'mock_maps/intensity/TM'+str(inst)+'/rlz'+str(rlz)+'_TM'+str(inst)+'_full_intensity.npz'

    # intensity_map_fpath = jmock_basedir+'mock_maps/intensity/TM'+str(inst)+'/rlz'+str(rlz)+'_TM'+str(inst)+'_CIBERfidmask_full_intensity.npz'
    intensity_dat = np.load(intensity_map_fpath)
    intensity_map = intensity_dat['ciber_map']

    # tailstr = 'CIBERfidmask'

    if zstr is not None:

        if tailstr == '':
            tailstr = zstr
        else:
            tailstr += '_'+zstr

    gal_map_fpath = jmock_basedir+'mock_maps/galaxy/TM'+str(inst)+'/rlz'+str(rlz)+'_TM'+str(inst)+'_'+galstr+'_'+tailstr+'_galaxy.npz'
    true_cross_fpath = jmock_basedir+'mock_ps_pred/TM'+str(inst)+'/indiv/rlz'+str(rlz)+'_TM'+str(inst)+'_auto_cross_pred_'+galstr+'_'+tailstr+'.npz'

    print('Loading gal_map data from ', gal_map_fpath)

    if load_gal_counts:
        galdat = np.load(gal_map_fpath)

        if not counts_from_xy:
            galmap = galdat['gal_counts']

        if load_tracer_cat:
            if with_redshift:
                tracer_z = galdat['tracer_z']

            tracer_x, tracer_y = [galdat[tkey] for tkey in ['tracer_x', 'tracer_y']]

            if counts_from_xy:
                galmap = get_count_field(tracer_x, tracer_y, imdim=1024)
        else:
            tracer_x, tracer_y, tracer_z = None, None, None
    else:
        galmap = None
    
    true_cross = np.load(true_cross_fpath)
    gal_auto_pred = true_cross['clg_comb']
    cross_pred = true_cross['clx_comb']
    lb_pred = true_cross['lb']

    
    # dat = {'galmap':galmap, 'intensity_map':intensity_map, 'mask':mask, 
    #       'lb':lb_pred, 'cross_pred':cross_pred, 'gal_auto_pred':gal_auto_pred, \
    #       'tracer_x':tracer_x, 'tracer_y':tracer_y}
    dat = {'intensity_map':intensity_map,
          'lb':lb_pred, 'cross_pred':cross_pred, 'gal_auto_pred':gal_auto_pred}

    if load_gal_counts:
        dat['galmap'] = galmap
        dat['tracer_x'] = tracer_x
        dat['tracer_y'] = tracer_y
        if with_redshift:
            dat['tracer_z'] = tracer_z
    
    return dat


def gen_predictions_from_jmock(addstr, nmock=10, inst_list=[1, 2], \
                              jmock_basedir = 'data/jordan_mocks/v2/', plot=True, \
                              pixel_corr=False, beam_corr=True, 
                              ifield_map=8, nbins=26, \
                              add_stars=False, maglim_stars=16.,
                              cl_cross_poisson=True):
    
    mock_idxs = np.arange(1, 1+nmock)
    
    for idx, inst in enumerate(inst_list):
        
        if beam_corr:

            bls_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/beam_correction/bl_est_postage_stamps_TM'+str(inst)+'_081121.npz'
            B_ell = np.load(bls_fpath)['B_ells_post'][ifield_map-4]
        
    
        gal_auto_cls, intensity_auto_cls_tracer, \
            intensity_auto_cls_full, cross_cls, \
                rlx_tracer_full = [np.zeros((nmock, nbins-1)) for x in range(5)]

        all_cl_cross_poisson = np.zeros((nmock,))

        for m, mock_idx in enumerate(mock_idxs):
            
            mock_ps_fpath = f"{jmock_basedir}mock_ps_pred/TM{inst}/indiv/rlz{mock_idx}_TM{inst}_auto_cross_pred_{addstr}.npz"
            mock_ps = np.load(mock_ps_fpath)
            
            mock_ps_fpath_full = f"{jmock_basedir}mock_ps_pred/TM{inst}/indiv/rlz{mock_idx}_TM{inst}_auto_fullCIBER.npz"
            mock_ps_full = np.load(mock_ps_fpath_full)

            if m==0:
                lb = mock_ps['lb']
                if add_stars:
                    pf = lb*(lb+1)/(2*np.pi)
                    if 'sdss' in addstr:
                        ifield_list_stars = [4, 5, 6, 7, 8]
                    else:
                        ifield_list_stars = [ifield_map]
                    
                    isl_dl_pred = generate_auto_dl_pred_trilegal(lb, inst, maglim_stars, ifield_list=ifield_list_stars)
                    isl_cl_pred = isl_dl_pred / pf
                    
                    
                    print('isl cl pred:', isl_cl_pred)


            gal_auto_cls[m] = mock_ps['clg_comb']
            intensity_auto_cls_tracer[m] = mock_ps['clI_comb']
            
            cross_cls[m] = mock_ps['clx_comb']
            intensity_auto_cls_full[m] = mock_ps_full['clI_comb']
            
            if add_stars:
                intensity_auto_cls_full[m] += np.mean(isl_cl_pred, axis=0)*B_ell**2
            

            rlx_tracer_full[m] = cross_cls[m] / np.sqrt(intensity_auto_cls_full[m] * gal_auto_cls[m])

            if cl_cross_poisson:
                all_cl_cross_poisson[m] = mock_ps['cl_cross_poisson']


        res = {}

        allcls = [gal_auto_cls, intensity_auto_cls_tracer, intensity_auto_cls_full, cross_cls]
        keys = ['gal_auto', 'intensity_auto_tracer', 'intensity_auto_full', 'cross']
        res['lb'] = lb
        
        
        if pixel_corr:
            pixel_fn = get_pixel_window_function(lb, 7.)

        
        for k, key in enumerate(keys):

            mean_cl, sem_cl = np.mean(allcls[k], axis=0), scipy.stats.sem(allcls[k], axis=0)

            if beam_corr:
                if key=='cross':
                    
                    mean_cl /= B_ell
                    sem_cl /= B_ell
                
                elif 'intensity' in key:
                    mean_cl /= B_ell**2
                    sem_cl /= B_ell**2
                    
#             if key=='gal_auto' and pixel_corr:
                
#                 mean_cl /= np.sqrt(pixel_fn)
#                 sem_cl /= np.sqrt(pixel_fn)
            
#             elif key=='cross' and pixel_corr:
#                 mean_cl /= np.sqrt(pixel_fn)
#                 sem_cl /= np.sqrt(pixel_fn)

            
            res[key] = mean_cl
            res[key+'_err'] = sem_cl
            
            
        mean_rl, sem_rl = np.mean(rlx_tracer_full, axis=0), scipy.stats.sem(rlx_tracer_full, axis=0)
        res['rlx_tracer_full'] = mean_rl
        res['rlx_err_tracer_full'] = sem_rl


        if cl_cross_poisson:

            res['cross_poisson'] = np.mean(all_cl_cross_poisson)

        
        
        if plot:
            pf = lb*(lb+1)/(2*np.pi)
            plt.figure(figsize=(5, 4))
            
            for k, key in enumerate(keys):
                plt.errorbar(lb, pf*res[key], yerr=pf*res[key+'_err'], fmt='o', capsize=2, label=key)
                
            plt.legend()
            
            plt.ylabel('$D_{\\ell}$', fontsize=16)
            plt.xlabel('$\\ell$', fontsize=16)
            plt.yscale('log')
            plt.xscale('log')
            plt.grid(alpha=0.3)
            plt.show()

            if cl_cross_poisson:

                plt.figure(figsize=(5, 4))

                plt.errorbar(lb, pf*res['cross'], yerr=pf*res['cross_err'], fmt='o', capsize=3, label='Cross power spectrum')
                plt.plot(lb, pf*res['cross_poisson'], linestyle='dashed', label='Cross-shot noise')

                plt.legend()
                
                plt.ylabel('$D_{\\ell}$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=16)
                plt.xlabel('$\\ell$', fontsize=16)
                plt.yscale('log')
                plt.xscale('log')
                plt.grid(alpha=0.3)
                plt.show()

                
            plt.figure(figsize=(5, 4))
            plt.errorbar(lb, res['gal_auto'], yerr=res['gal_auto_err'], fmt='o', capsize=2, label='gal auto')
            plt.errorbar(lb, res['cross'], yerr=res['cross_err'], fmt='o', capsize=2, label='cross')
                            
            plt.ylabel('$C_{\\ell}$', fontsize=16)
            plt.xlabel('$\\ell$', fontsize=16)
            plt.yscale('log')
            plt.xscale('log')
            plt.grid(alpha=0.3)
            plt.show()

            plt.figure(figsize=(5, 4))
            plt.errorbar(lb, res['rlx_tracer_full'], yerr=res['rlx_err_tracer_full'], fmt='o', capsize=2, label=key)            
            
            plt.ylabel('$r_{\\ell}^{\\times}$', fontsize=16)
            plt.xlabel('$\\ell$', fontsize=16)
#             plt.yscale('log')
            plt.ylim(0, 1)
            plt.xscale('log')
            plt.grid(alpha=0.3)
            plt.show()
            

        pred_fpath = jmock_basedir+'mock_ps_pred/TM'+str(inst)+'/field_average/pred_cls_TM'+str(inst)+'_'+addstr+'.npz'
        print('saving predictions to ', pred_fpath)
        np.savez(pred_fpath, **res)


def grab_detector_pos(ra, dec, fov_deg=2.0, imdim=1024, center_ra=None, center_dec=None):
    """
    Project sky coordinates to detector pixels for a square field.

    Parameters
    ----------
    ra, dec : array-like
        Sky coordinates in degrees.
    fov_deg : float
        Angular size of square map in degrees.
    imdim : int
        Pixel dimension of square map.
    center_ra, center_dec : float or None
        Optional map center in degrees. If None, uses catalog mean.
    """
    if center_ra is None:
        center_ra = np.mean(ra)
    if center_dec is None:
        center_dec = np.mean(dec)

    deg_per_pix = fov_deg / float(imdim)
    arcsec_per_pix = deg_per_pix * 3600.0

    delta_ra = (ra - center_ra) * np.cos(np.deg2rad(center_dec)) * 3600.0
    delta_dec = (dec - center_dec) * 3600.0

    xpix = imdim / 2.0 + delta_ra / arcsec_per_pix
    ypix = imdim / 2.0 + delta_dec / arcsec_per_pix

    valid = (xpix >= 0) & (xpix < imdim) & (ypix >= 0) & (ypix < imdim)
    return xpix, ypix, valid



def catalog_to_sb_map(x, y, mag_AB, ciber_inst, ifield=8, imdim=1024):
    
    cmock = ciber_mock(nx=imdim, ny=imdim)
    
    lam_dict = dict({1:1.1, 2:1.8})

    cat_full = np.array([x, y, mag_AB]).transpose()

    I_arr_full = cmock.mag_2_nu_Inu(cat_full[:,2], lam_eff=lam_dict[ciber_inst]*1e-6*u.m)
    cat_full = np.hstack([cat_full, np.expand_dims(I_arr_full.value, axis=1)])
    print(cat_full.shape)
    
    srcmap_full = cmock.make_srcmap_temp_bank(ifield, ciber_inst, cat_full, flux_idx=-1, n_fine_bin=10, nwide=17, load_precomp_tempbank=True, \
                            tempbank_dirpath='data/subpixel_psfs_TM'+str(ciber_inst)+'/')
    
    return srcmap_full
        
    
    
def load_ra_dec_mags_redshift_jmock(ciber_inst, pop, galstr, jmock_dir):
    

    zstr_dict = dict({1:'0.900_1.200_um', 2:'1.350_2.100_um'})

    ra = fits.open(os.path.join(jmock_dir, 'cat_ra_pop_'+str(pop)+'.fits'))[1].data['ra']
    dec = fits.open(os.path.join(jmock_dir, 'cat_dec_pop_'+str(pop)+'.fits'))[1].data['dec']

    gal_flux = fits.open(os.path.join(jmock_dir, 'cat_'+galstr+'_pop_'+str(pop)+'.fits'))[1].data[galstr]
    gal_mag_AB = 23.9-2.5*np.log10(gal_flux)

    ciber_flux = fits.open(os.path.join(jmock_dir, 'cat_'+zstr_dict[ciber_inst]+'_pop_'+str(pop)+'.fits'))[1].data['flux']
    ciber_mag_AB = 23.9-2.5*np.log10(ciber_flux)

    gal_redshifts = fits.open(os.path.join(jmock_dir, 'cat_z_pop_'+str(pop)+'.fits'))[1].data['z']
    
    return ra, dec, gal_mag_AB, ciber_mag_AB, gal_redshifts



def grab_mask_radius_interp_fn(ciber_inst, ifield=8, plot=False):
    
    rm_basepath = config.ciber_basepath+'data/masking_radii_binwiseopt/'
    interp_mask_fn_fpath = rm_basepath+'rad_vs_mag_fullrange_TM'+str(ciber_inst)+'_ifield'+str(ifield)+'_a=180_120722.npz'

    interp_mask_file = np.load(interp_mask_fn_fpath)
    cent_mags, rad_vs_cent_mags = interp_mask_file['cent_mags'], interp_mask_file['radii']
    min_mag, max_mag = np.min(cent_mags), np.max(cent_mags)
    
    interp_maskfn = scipy.interpolate.interp1d(cent_mags[rad_vs_cent_mags!= 0], rad_vs_cent_mags[rad_vs_cent_mags != 0])

    if plot:
        plt.figure()
        plt.scatter(cent_mags, rad_vs_cent_mags, marker='.', color='k')
        cent_mags_fine = np.linspace(np.min(cent_mags), np.max(cent_mags), 1000)
        plt.plot(cent_mags_fine, interp_maskfn(cent_mags_fine), color='r')
        plt.yscale('log')
        plt.ylabel('masking radius [arcsec]')
        plt.xlabel('magnitude (Vega)')
        plt.show()
        
    return interp_maskfn, min_mag, max_mag
    

def generate_mask_for_jmock(xmask, ymask, mags_AB, ciber_inst, mag_lim=16.0, imdim=1024):
    
    cbps = CIBER_PS_pipeline()
    
    interp_maskfn, min_mag, max_mag = grab_mask_radius_interp_fn(ciber_inst, plot=False)
            
    mags_Vega = mags_AB - cbps.Vega_to_AB[ciber_inst]

    if mags_Vega.size == 0:
        return np.ones((imdim, imdim))

    bright = (mags_Vega < mag_lim) & (mags_Vega > 0)
    if np.sum(bright) == 0:
        # Nothing to mask at this mag limit; return pass-through mask.
        return np.ones((imdim, imdim))

    source_mask, radii = mask_from_cat(xs=xmask, ys=ymask, mags=mags_Vega, mag_lim_min=0, inst=ciber_inst,\
                                        mag_lim=mag_lim, interp_maskfn=interp_maskfn,\
                                          Vega_to_AB=None, dimx=imdim, dimy=imdim, plot=False, \
                                     interp_max_mag = max_mag, interp_min_mag=min_mag, m_min_thresh=None, \
                                    mask_transition_mag=14.0, mode=None)
    
    print('source mask')
    
    return source_mask


def process_jmock(ciber_inst, mock_rlz=1, pop_list=[0, 1], galstr='hsc_i', 
                    gen_sb_map=True, 
                    calc_cross_poisson=False,
                    regen_full_map=False,
                    m_min_gal=10.0, m_max_gal=25.0,
                    m_min_ciber=10.0, m_max_ciber=28.0,
                    masking_maglim=16.0,
                    imdim=1024, jmock_basedir='data/jordan_mocks/v2/',
                    jmock_catalog_dir=None,
                    save_maps=True, save_ps=True, save_intensity=True, save_galaxy=True, \
                    redshift_min=None, redshift_max=None, nbins_ps=26, 
                    save_counts=True, gen_sb_pred_from_map=True, return_tracer_z=False, \
                    addstr=None, compute_zsb=False,
                    fov_deg=2.0, tile_bounds_deg=None, tile_label=None, ifield_map=8,
                    skip_missing_pop=False,
                    apply_map_mask=True,
                    save_diagnostic_figs=False, diagnostic_dir=None,
                    save_magcut_diagnostic_figs=False,
                    intensity_units='nW m^-2 sr^-1', save_histograms=True,
                    interactive_plots=False,
                    save_ps_diagnostic_figs=False, ps_figure_dir=None):

    lam_dict = {1: 1.1, 2: 1.8}
    zstr_dict = {1: '0.900_1.200_um', 2: '1.350_2.100_um'}

    s = f"{mock_rlz:03d}"
    mockstr = 'rlz_' + s
    pixsize_arcsec = fov_deg * 3600.0 / float(imdim)
    

    if addstr is None:
        ciber_maskstr = 'CIBERfidmask'
        gal_selstr = galstr + '_lt_' + str(m_max_gal)
        addstr = gal_selstr + '_' + ciber_maskstr
    
    if redshift_min is not None and ('zmin=' not in addstr):
        addstr += '_zmin='+str(redshift_min)

    if redshift_max is not None and ('zmax=' not in addstr):
        addstr += '_zmax='+str(redshift_max)
    
    
    cbps = CIBER_PS_pipeline()
    if jmock_catalog_dir is None:
        jmock_dir = f"{jmock_basedir}{mockstr}/zmin_0.050/zmax_4.573/m_10.00_15.00/"
    else:
        jmock_dir = jmock_catalog_dir

    # Accumulators
    
    total_counts, total_srcmap_full, total_srcmap_pred = [np.zeros((imdim, imdim)) for _ in range(3)]
    total_mask = np.ones((imdim, imdim), dtype=int)

    # PS lists
    all_clI_full, all_clI_full_errs = [], []
    all_clI_pred, all_clI_pred_errs = [], []
    all_clg, all_clg_errs, all_clx, all_clx_errs = [], [], [], []

    tracer_x, tracer_y, tracer_z = [], [], []

    all_mags_pred = []
    all_gal_mags_all = []
    all_gal_mags_sel = []
    all_ciber_mags_all = []
    all_ciber_mags_sel = []

    if gen_sb_pred_from_map and compute_zsb:
        zsb_pred_tot = np.zeros((imdim, imdim))

    n_valid_pop = 0
    for pop in pop_list:

        zstr_dict = {1: '0.900_1.200_um', 2: '1.350_2.100_um'}
        required_fpaths = [
            os.path.join(jmock_dir, f'cat_ra_pop_{pop}.fits'),
            os.path.join(jmock_dir, f'cat_dec_pop_{pop}.fits'),
            os.path.join(jmock_dir, f'cat_{galstr}_pop_{pop}.fits'),
            os.path.join(jmock_dir, f'cat_{zstr_dict[ciber_inst]}_pop_{pop}.fits'),
            os.path.join(jmock_dir, f'cat_z_pop_{pop}.fits'),
        ]
        missing = [fp for fp in required_fpaths if not os.path.exists(fp)]
        if missing:
            if skip_missing_pop:
                print(f'[process_jmock] skipping incomplete population pop={pop} in {jmock_dir}')
                for miss in missing:
                    print(f'  missing: {miss}')
                continue
            raise FileNotFoundError(f'Missing required mock files for pop={pop}: {missing}')
        n_valid_pop += 1

        ra, dec, gal_mag_AB,\
            ciber_mag_AB, gal_redshift = load_ra_dec_mags_redshift_jmock(ciber_inst, pop, galstr, jmock_dir)

        if tile_bounds_deg is not None:
            ra_min, ra_max, dec_min, dec_max = tile_bounds_deg
            tile_sel = (ra >= ra_min) & (ra < ra_max) & (dec >= dec_min) & (dec < dec_max)
            ra, dec = ra[tile_sel], dec[tile_sel]
            gal_mag_AB = gal_mag_AB[tile_sel]
            ciber_mag_AB = ciber_mag_AB[tile_sel]
            gal_redshift = gal_redshift[tile_sel]

            if ra.size == 0:
                print(f'[process_jmock] tile {tile_label} has zero sources for pop={pop}; skipping')
                continue
        
        center_ra = None
        center_dec = None
        if tile_bounds_deg is not None:
            ra_min, ra_max, dec_min, dec_max = tile_bounds_deg
            center_ra = 0.5 * (ra_min + ra_max)
            center_dec = 0.5 * (dec_min + dec_max)

        xpix, ypix, idx_infov = grab_detector_pos(
            ra,
            dec,
            fov_deg=fov_deg,
            imdim=imdim,
            center_ra=center_ra,
            center_dec=center_dec,
        )
    
        xpix, ypix = xpix[idx_infov], ypix[idx_infov]
        gal_mag_AB, ciber_mag_AB = gal_mag_AB[idx_infov], ciber_mag_AB[idx_infov]
        gal_redshift = gal_redshift[idx_infov]

        if xpix.size == 0 or gal_mag_AB.size == 0 or ciber_mag_AB.size == 0:
            print(f'[process_jmock] no in-FOV sources for pop={pop}; skipping')
            continue

        all_gal_mags_all.extend(gal_mag_AB)
        all_ciber_mags_all.extend(ciber_mag_AB)

        print('grabbing both CIBER mags for mask prediction purposes')
        all_ciber_mags_AB = []
        for inst in [1, 2]:
            cbmag = load_ra_dec_mags_redshift_jmock(inst, pop, galstr, jmock_dir)[3]
            if tile_bounds_deg is not None:
                cbmag = cbmag[tile_sel]
            all_ciber_mags_AB.append(cbmag[idx_infov])


        # Mask
        if apply_map_mask and masking_maglim > 15.0:
            mask = generate_mask_for_jmock(
                xpix,
                ypix,
                gal_mag_AB,
                ciber_inst,
                mag_lim=masking_maglim,
                imdim=imdim,
            )
            total_mask *= mask.astype(int)

        
        # tracer-only prediction map
        # mask_pred = ((ciber_mag_AB > masking_maglim + cbps.Vega_to_AB[ciber_inst]) &
        #              (ciber_mag_AB < m_max_ciber) &
        #              (gal_mag_AB < m_max_gal))


        mask_pred = ((all_ciber_mags_AB[0] > masking_maglim + cbps.Vega_to_AB[1]) &
                    (all_ciber_mags_AB[1] > masking_maglim + cbps.Vega_to_AB[2]) &
                     (ciber_mag_AB < m_max_ciber) &
                     (gal_mag_AB < m_max_gal))


        if redshift_min is not None:
            mask_pred = mask_pred & (gal_redshift > redshift_min)
        if redshift_max is not None:
            mask_pred = mask_pred & (gal_redshift < redshift_max)


        if calc_cross_poisson:

            all_mags_pred.extend(ciber_mag_AB[mask_pred])



        # Intensity maps
        if gen_sb_map:
            if regen_full_map:
                # include fiducial mask in full map, assume brightest galaxies covered by mask in map
                mask_full = (ciber_mag_AB > m_min_ciber) & (ciber_mag_AB < m_max_ciber) & (ciber_mag_AB > masking_maglim + cbps.Vega_to_AB[ciber_inst])
                srcmap_full = catalog_to_sb_map(xpix[mask_full], ypix[mask_full],
                                                ciber_mag_AB[mask_full], ciber_inst, ifield=ifield_map, imdim=imdim)
                total_srcmap_full += srcmap_full

                lb, clI_full, clI_full_err = get_power_spec(
                    srcmap_full - np.mean(srcmap_full),
                    nbins=nbins_ps,
                    pixsize=pixsize_arcsec,
                )
                all_clI_full.append(clI_full)
                all_clI_full_errs.append(clI_full_err)


            srcmap_pred = catalog_to_sb_map(xpix[mask_pred], ypix[mask_pred],
                                            ciber_mag_AB[mask_pred], ciber_inst, ifield=ifield_map, imdim=imdim)


            if gen_sb_pred_from_map and compute_zsb:
                zsb_pred = (ciber_mag_AB > masking_maglim + cbps.Vega_to_AB[ciber_inst])

                if redshift_min is not None and redshift_max is not None:

                    zsb_pred *= (gal_redshift > redshift_min)*(gal_redshift < redshift_max)

                zsb_pred_tot += catalog_to_sb_map(xpix[zsb_pred], ypix[zsb_pred],
                                                ciber_mag_AB[zsb_pred], ciber_inst, ifield=ifield_map, imdim=imdim)

            total_srcmap_pred += srcmap_pred

            lb, clI_pred, clI_pred_err = get_power_spec(
                srcmap_pred - np.mean(srcmap_pred),
                nbins=nbins_ps,
                pixsize=pixsize_arcsec,
            )
            all_clI_pred.append(clI_pred)
            all_clI_pred_errs.append(clI_pred_err)

        #  --------------- Galaxy catalog selection ---------------
        # include galaxies not masked by CIBER, less than galaxy catalog magnitude cut, with optional redshift selection
        gal_mask = ((gal_mag_AB > m_min_gal) & (gal_mag_AB < m_max_gal) &
                    (ciber_mag_AB > masking_maglim + cbps.Vega_to_AB[ciber_inst]))
    
        if redshift_min is not None:
            gal_mask = gal_mask & (gal_redshift > redshift_min)
        if redshift_max is not None:
            gal_mask = gal_mask & (gal_redshift < redshift_max)

        tracer_x.extend(xpix[gal_mask])
        tracer_y.extend(ypix[gal_mask])
        tracer_z.extend(gal_redshift[gal_mask])
        all_gal_mags_sel.extend(gal_mag_AB[gal_mask])
        all_ciber_mags_sel.extend(ciber_mag_AB[gal_mask])
        # tracer_xypos = [xpix[gal_mask], ypix[gal_mask]]
        
        # ---------------- make galaxy count/overdensity fields and compute auto spectrum ------------------
        counts = get_count_field(xpix[gal_mask], ypix[gal_mask], imdim=imdim)
        gal_density = (counts - np.mean(counts)) / np.mean(counts)
        total_counts += counts

        lb, clg, clgerr = get_power_spec(gal_density, nbins=nbins_ps, pixsize=pixsize_arcsec)
        all_clg.append(clg)
        all_clg_errs.append(clgerr)

        # Cross-spectrum uses full CIBER if regen_full_map=True
        if regen_full_map:
            meansub_ciber = srcmap_full - np.mean(srcmap_full)
            lb, clx, clxerr = get_power_spec(
                meansub_ciber,
                map_b=gal_density,
                nbins=nbins_ps,
                pixsize=pixsize_arcsec,
            )
            all_clx.append(clx)
            all_clx_errs.append(clxerr)

    if n_valid_pop == 0:
        print('[process_jmock] no valid populations found after filtering; skipping this call')
        return None

    # Combined maps
    if np.mean(total_counts) <= 0:
        print('[process_jmock] zero galaxies selected in this configuration; skipping this call')
        return None

    gal_overdens = (total_counts - np.mean(total_counts)) / np.mean(total_counts)

    if regen_full_map:
        meansub_full_tot = total_srcmap_full - np.mean(total_srcmap_full)
        lb, clI_full_comb, clI_full_err_comb = get_power_spec(
            meansub_full_tot,
            nbins=nbins_ps,
            pixsize=pixsize_arcsec,
        )
    else:
        clI_full_comb, clI_full_err_comb = None, None

    meansub_pred_tot = total_srcmap_pred - np.mean(total_srcmap_pred)
    lb, clI_pred_comb, clI_pred_err_comb = get_power_spec(
        meansub_pred_tot,
        nbins=nbins_ps,
        pixsize=pixsize_arcsec,
    )
    lb, clg_comb, clg_err_comb = get_power_spec(gal_overdens, nbins=nbins_ps, pixsize=pixsize_arcsec)
    lb, clx_comb, clx_err_comb = get_power_spec(
        meansub_pred_tot,
        map_b=gal_overdens,
        nbins=nbins_ps,
        pixsize=pixsize_arcsec,
    )
    rlx_comb = clx_comb / np.sqrt(clI_pred_comb * clg_comb)


    if calc_cross_poisson:
        print('Calculating cross-shot noise from catalog')
        clx_shot_pred = calc_cross_cl_shot_noise(np.array(all_mags_pred), lam_dict[ciber_inst], aeff=4.0)
        print('clx shot pred:', clx_shot_pred)
    else:
        clx_shot_pred = None


    # Save
    if save_ps:
        if regen_full_map:
            full_tag = 'auto_fullCIBER'
            if redshift_min is not None:
                full_tag += f'_zmin={redshift_min}'
            if redshift_max is not None:
                full_tag += f'_zmax={redshift_max}'
            if tile_label is not None:
                full_tag += '_' + tile_label
            savepath_full = f"{jmock_basedir}mock_ps_pred/TM{ciber_inst}/indiv/rlz{mock_rlz}_TM{ciber_inst}_{full_tag}.npz"
            save_jmock_ps(savepath_full, lb, clI_full_comb, clI_full_err_comb,
                          None, None, None, None, None,
                          all_clI_full, all_clI_full_errs, None, None, None, None)

            if save_ps_diagnostic_figs:
                if ps_figure_dir is None:
                    ps_figure_dir = os.path.join(jmock_basedir, 'diagnostics', f'TM{ciber_inst}', 'power_spectra')
                os.makedirs(ps_figure_dir, exist_ok=True)
                _save_ps_diagnostics(
                    fig_basepath=os.path.join(ps_figure_dir, f'rlz{mock_rlz}_TM{ciber_inst}_{full_tag}'),
                    lb=lb,
                    clI=clI_full_comb,
                    clI_err=clI_full_err_comb,
                    clg=None,
                    clg_err=None,
                    clx=None,
                    clx_err=None,
                    cl_cross_poisson=None,
                    title_suffix='(Full CIB map)',
                )

        pred_tag = addstr
        if tile_label is not None:
            pred_tag += '_' + tile_label
        savepath_pred = f"{jmock_basedir}mock_ps_pred/TM{ciber_inst}/indiv/rlz{mock_rlz}_TM{ciber_inst}_auto_cross_pred_{pred_tag}.npz"
        save_jmock_ps(savepath_pred, lb, clI_pred_comb, clI_pred_err_comb,
                      clg_comb, clg_err_comb, clx_comb, clx_err_comb, rlx_comb,
                      all_clI_pred, all_clI_pred_errs, all_clg, all_clg_errs, all_clx, all_clx_errs, 
                      cl_cross_poisson=clx_shot_pred)

        if save_ps_diagnostic_figs:
            if ps_figure_dir is None:
                ps_figure_dir = os.path.join(jmock_basedir, 'diagnostics', f'TM{ciber_inst}', 'power_spectra')
            os.makedirs(ps_figure_dir, exist_ok=True)
            _save_ps_diagnostics(
                fig_basepath=os.path.join(ps_figure_dir, f'rlz{mock_rlz}_TM{ciber_inst}_{pred_tag}'),
                lb=lb,
                clI=clI_pred_comb,
                clI_err=clI_pred_err_comb,
                clg=clg_comb,
                clg_err=clg_err_comb,
                clx=clx_comb,
                clx_err=clx_err_comb,
                cl_cross_poisson=clx_shot_pred,
                title_suffix='(Tracer-selected)',
            )

    if save_maps:
        if save_intensity:
            if regen_full_map:
                full_addstr = 'full'
                if redshift_min is not None:
                    full_addstr += f'_zmin={redshift_min}'
                if redshift_max is not None:
                    full_addstr += f'_zmax={redshift_max}'
                if tile_label is not None:
                    full_addstr += '_' + tile_label
                save_intensity_map(
                    jmock_basedir,
                    mock_rlz,
                    ciber_inst,
                    full_addstr,
                    total_srcmap_full,
                    total_mask if apply_map_mask else None,
                    plot=interactive_plots,
                    save_figures=save_diagnostic_figs,
                    figure_dir=diagnostic_dir,
                    intensity_units=intensity_units,
                    save_histograms=save_histograms,
                )
                # save_intensity_map(jmock_basedir, mock_rlz, ciber_inst, ciber_maskstr + '_full', total_srcmap_full, total_mask, plot=True)
            addstr_pred = addstr + '_pred'
            if tile_label is not None:
                addstr_pred += '_' + tile_label
            save_intensity_map(
                jmock_basedir,
                mock_rlz,
                ciber_inst,
                addstr_pred,
                total_srcmap_pred,
                None,
                plot=interactive_plots,
                save_figures=save_diagnostic_figs,
                figure_dir=diagnostic_dir,
                intensity_units=intensity_units,
                save_histograms=save_histograms,
            )
        if save_galaxy:
            print('saving galaxy map, tracer x has length = ', len(tracer_x))
            if not save_counts:
                total_counts = None

            addstr_gal = addstr
            if tile_label is not None:
                addstr_gal += '_' + tile_label
            save_galaxy_maps(jmock_basedir, mock_rlz, ciber_inst, addstr_gal, gal_overdens, total_counts, plot=False, \
                            tracer_x=tracer_x, tracer_y=tracer_y, tracer_z=tracer_z)

    if save_magcut_diagnostic_figs:
        if diagnostic_dir is None:
            diagnostic_dir = os.path.join(jmock_basedir, 'diagnostics', f'TM{ciber_inst}')
        mag_dir = os.path.join(diagnostic_dir, 'magcuts')
        os.makedirs(mag_dir, exist_ok=True)

        base_tag = addstr
        if tile_label is not None:
            base_tag += '_' + tile_label
        fig_base = os.path.join(mag_dir, f'rlz{mock_rlz}_TM{ciber_inst}_{base_tag}')
        ciber_cut_ab = masking_maglim + cbps.Vega_to_AB[ciber_inst]

        _save_magnitude_cut_diagnostics(
            fig_basepath=fig_base,
            gal_mags_all=np.asarray(all_gal_mags_all),
            gal_mags_sel=np.asarray(all_gal_mags_sel),
            ciber_mags_all=np.asarray(all_ciber_mags_all),
            ciber_mags_sel=np.asarray(all_ciber_mags_sel),
            m_min_gal=m_min_gal,
            m_max_gal=m_max_gal,
            ciber_cut_ab=ciber_cut_ab,
            gal_label=galstr,
        )

    if gen_sb_pred_from_map and compute_zsb:
        mean_sb = np.mean(zsb_pred_tot)

        return mean_sb, tracer_z

    elif return_tracer_z:
        return tracer_z
    
def save_jmock_maps(jmock_basedir, mock_rlz, ciber_inst, addstr,\
                    ciber_map, gal_overdens, gal_counts, mask, plot=False):
    
    if plot:
        plot_map(ciber_map, title='CIBER mock map', figsize=(7, 7))
        plot_map(gal_overdens, title='Galaxy overdensity', figsize=(7, 7))
        plot_map(gal_counts, title='Galaxy counts', figsize=(7, 7))
        plot_map(ciber_map*(mask.transpose()), title='CIBER mock map (masked)', figsize=(7, 7))

    save_fpath = jmock_basedir+'mock_maps/rlz'+str(mock_rlz)+'_TM'+str(ciber_inst)+'_cibermap_gal_'+addstr+'.npz'
    print('Saving maps to ', save_fpath)
    np.savez(save_fpath, mock_rlz=mock_rlz, ciber_map=ciber_map, gal_overdens=gal_overdens, gal_counts=gal_counts, \
            mask=mask.transpose())
    

def _save_intensity_diagnostics(fig_basepath, ciber_map, mask=None, intensity_units='nW m^-2 sr^-1', save_histograms=True):
    os.makedirs(os.path.dirname(fig_basepath), exist_ok=True)

    finite = np.isfinite(ciber_map)
    if not np.any(finite):
        return

    cmap_clip = np.nanpercentile(ciber_map[finite], 99.5)
    cmap_clip = max(cmap_clip, np.nanmax(ciber_map[finite]) * 1e-3)

    plt.figure(figsize=(6, 5))
    plt.imshow(ciber_map, origin='lower', cmap='magma', vmin=0.0, vmax=cmap_clip)
    cbar = plt.colorbar()
    cbar.set_label(f'Intensity [{intensity_units}]')
    plt.title('CIB Intensity Map')
    plt.tight_layout()
    plt.savefig(fig_basepath + '_map.png', dpi=150)
    plt.close()

    if mask is not None:
        masked_map = ciber_map * mask.T
        plt.figure(figsize=(6, 5))
        plt.imshow(masked_map, origin='lower', cmap='magma', vmin=0.0, vmax=cmap_clip)
        cbar = plt.colorbar()
        cbar.set_label(f'Intensity [{intensity_units}]')
        plt.title('CIB Intensity Map (Masked)')
        plt.tight_layout()
        plt.savefig(fig_basepath + '_map_masked.png', dpi=150)
        plt.close()

    if save_histograms:
        pix = ciber_map[finite]
        pix = pix[pix > 0]
        if pix.size > 0:
            plt.figure(figsize=(6, 5))
            plt.hist(pix, bins=200, histtype='stepfilled', alpha=0.8)
            plt.yscale('log')
            plt.xlabel(f'Intensity [{intensity_units}]')
            plt.ylabel('N pixels')
            plt.title('Per-pixel Intensity Histogram')
            plt.tight_layout()
            plt.savefig(fig_basepath + '_hist.png', dpi=150)
            plt.close()

            plt.figure(figsize=(6, 5))
            logbins = np.logspace(np.log10(np.nanmin(pix)), np.log10(np.nanmax(pix)), 160)
            plt.hist(pix, bins=logbins, histtype='stepfilled', alpha=0.8)
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(f'Intensity [{intensity_units}]')
            plt.ylabel('N pixels')
            plt.title('Per-pixel Intensity Histogram (Log bins)')
            plt.tight_layout()
            plt.savefig(fig_basepath + '_hist_logbins.png', dpi=150)
            plt.close()

            hist_counts, hist_edges = np.histogram(pix, bins=200)
            np.savez(
                fig_basepath + '_hist_data.npz',
                counts=hist_counts,
                bin_edges=hist_edges,
                intensity_units=intensity_units,
            )


def _save_ps_diagnostics(fig_basepath, lb, clI, clI_err, clg, clg_err, clx, clx_err,
                         cl_cross_poisson=None, title_suffix=''):
    finite_lb = np.isfinite(lb) & (lb > 0)
    if not np.any(finite_lb):
        return

    ell = lb[finite_lb]
    pf = ell * (ell + 1.0) / (2.0 * np.pi)

    plt.figure(figsize=(5, 4))
    if clI is not None:
        plt.errorbar(ell, pf * np.asarray(clI)[finite_lb], yerr=pf * np.asarray(clI_err)[finite_lb],
                     fmt='o', capsize=2, label='intensity auto')
    if clg is not None:
        plt.errorbar(ell, pf * np.asarray(clg)[finite_lb], yerr=pf * np.asarray(clg_err)[finite_lb],
                     fmt='o', capsize=2, label='galaxy auto')
    if clx is not None:
        plt.errorbar(ell, pf * np.asarray(clx)[finite_lb], yerr=pf * np.asarray(clx_err)[finite_lb],
                     fmt='o', capsize=2, label='cross')
    if cl_cross_poisson is not None:
        plt.plot(ell, pf * cl_cross_poisson * np.ones_like(ell), linestyle='dashed', label='cross-shot noise')

    plt.ylabel('$D_{\\ell}$', fontsize=16)
    plt.xlabel('$\\ell$', fontsize=16)
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(alpha=0.3)
    if title_suffix:
        plt.title(f'Power Spectra {title_suffix}')
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_basepath + '_Dell.png', dpi=160)
    plt.close()

    if clx is not None:
        plt.figure(figsize=(5, 4))
        plt.errorbar(ell, pf * np.asarray(clx)[finite_lb], yerr=pf * np.asarray(clx_err)[finite_lb],
                     fmt='o', capsize=3, label='Cross power spectrum')
        if cl_cross_poisson is not None:
            plt.plot(ell, pf * cl_cross_poisson * np.ones_like(ell), linestyle='dashed', label='Cross-shot noise')

        plt.ylabel('$D_{\\ell}$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=16)
        plt.xlabel('$\\ell$', fontsize=16)
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(fig_basepath + '_Dell_cross.png', dpi=160)
        plt.close()


def _save_magnitude_cut_diagnostics(
    fig_basepath,
    gal_mags_all,
    gal_mags_sel,
    ciber_mags_all,
    ciber_mags_sel,
    m_min_gal,
    m_max_gal,
    ciber_cut_ab,
    gal_label='gal',
):
    if gal_mags_all.size == 0 or ciber_mags_all.size == 0:
        return

    finite_gal_all = np.isfinite(gal_mags_all)
    finite_gal_sel = np.isfinite(gal_mags_sel)
    finite_cib_all = np.isfinite(ciber_mags_all)
    finite_cib_sel = np.isfinite(ciber_mags_sel)

    gal_all = gal_mags_all[finite_gal_all]
    gal_sel = gal_mags_sel[finite_gal_sel]
    cib_all = ciber_mags_all[finite_cib_all]
    cib_sel = ciber_mags_sel[finite_cib_sel]

    if gal_all.size == 0 or cib_all.size == 0:
        return

    gmin = np.nanmin(gal_all)
    gmax = np.nanmax(gal_all)
    cmin = np.nanmin(cib_all)
    cmax = np.nanmax(cib_all)

    gbins = np.linspace(gmin, gmax, 80)
    cbins = np.linspace(cmin, cmax, 80)

    plt.figure(figsize=(9, 4.5))

    ax1 = plt.subplot(1, 2, 1)
    ax1.hist(gal_all, bins=gbins, histtype='step', linewidth=1.8, label='all in FOV')
    if gal_sel.size > 0:
        ax1.hist(gal_sel, bins=gbins, histtype='step', linewidth=1.8, label='selected')
    ax1.axvline(m_min_gal, linestyle='--', linewidth=1.2, color='k', alpha=0.8)
    ax1.axvline(m_max_gal, linestyle='--', linewidth=1.2, color='k', alpha=0.8)
    ax1.set_yscale('log')
    ax1.set_xlabel(f'{gal_label} mag (AB)')
    ax1.set_ylabel('counts')
    ax1.set_title('Galaxy-band Magnitude Counts')
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=9)

    ax2 = plt.subplot(1, 2, 2)
    ax2.hist(cib_all, bins=cbins, histtype='step', linewidth=1.8, label='all in FOV')
    if cib_sel.size > 0:
        ax2.hist(cib_sel, bins=cbins, histtype='step', linewidth=1.8, label='selected')
    ax2.axvline(ciber_cut_ab, linestyle='--', linewidth=1.2, color='k', alpha=0.8)
    ax2.set_yscale('log')
    ax2.set_xlabel('CIBER mag (AB)')
    ax2.set_ylabel('counts')
    ax2.set_title('CIBER-band Magnitude Counts')
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(fig_basepath + '_magcuts.png', dpi=160)
    plt.close()


def save_intensity_map(jmock_basedir, mock_rlz, ciber_inst, addstr, ciber_map, mask, plot=False,
                       save_figures=False, figure_dir=None, intensity_units='nW m^-2 sr^-1',
                       save_histograms=True):
    if plot:
        plot_map(ciber_map, title='CIBER mock map', figsize=(7, 7))

        if mask is not None:
            plot_map(ciber_map * mask.T, title='CIBER mock map (masked)', figsize=(7, 7))

    save_fpath = f"{jmock_basedir}mock_maps/intensity/TM{ciber_inst}/rlz{mock_rlz}_TM{ciber_inst}_{addstr}_intensity.npz"
    print('Saving intensity map to', save_fpath)

    if mask is not None:
        mask_save = mask.T 
    else:
        mask_save = None

    if save_figures:
        if figure_dir is None:
            figure_dir = os.path.join(jmock_basedir, 'diagnostics', f'TM{ciber_inst}')
        fig_base = os.path.join(figure_dir, f'rlz{mock_rlz}_TM{ciber_inst}_{addstr}')
        _save_intensity_diagnostics(
            fig_base,
            ciber_map,
            mask=mask,
            intensity_units=intensity_units,
            save_histograms=save_histograms,
        )

    np.savez(
        save_fpath,
        mock_rlz=mock_rlz,
        ciber_map=ciber_map,
        mask=mask_save,
        intensity_units=intensity_units,
    )


def save_galaxy_maps(jmock_basedir, mock_rlz, ciber_inst,
                     addstr, gal_overdens, gal_counts,
                    tracer_x=None, tracer_y=None, plot=False, tracer_z=None):
    if plot:
        plot_map(gal_overdens, title='Galaxy overdensity', figsize=(7, 7))
        plot_map(gal_counts, title='Galaxy counts', figsize=(7, 7))

    save_fpath = f"{jmock_basedir}mock_maps/galaxy/TM{ciber_inst}/rlz{mock_rlz}_TM{ciber_inst}_{addstr}_galaxy.npz"
    print('Saving galaxy maps to', save_fpath)
    # np.savez(save_fpath, mock_rlz=mock_rlz, gal_overdens=gal_overdens, gal_counts=gal_counts,
    #         tracer_x=tracer_x, tracer_y=tracer_y, tracer_z=None)
    np.savez(save_fpath, mock_rlz=mock_rlz, gal_counts=gal_counts,
            tracer_x=tracer_x, tracer_y=tracer_y, tracer_z=tracer_z)

    
def save_jmock_ps(save_fpath, lb, \
                  clI_comb, clI_err_comb, clg_comb, clg_err_comb, clx_comb, clx_err_comb, rlx_comb, \
                  all_clI=None, all_clI_errs=None, all_clg=None, all_clg_errs=None, all_clx=None, all_clx_errs=None, 
                  cl_cross_poisson=None):
    
    # Build dictionary of arrays to save
    save_dict = {
        'lb': lb,
        'clI_comb': clI_comb,
        'clI_err_comb': clI_err_comb,
        'clg_comb': clg_comb,
        'clg_err_comb': clg_err_comb,
        'clx_comb': clx_comb,
        'clx_err_comb': clx_err_comb, \
        'rlx_comb':rlx_comb
    }
    
    # Add optional arrays if they’re not None
    if all_clI is not None: save_dict['all_clI'] = all_clI
    if all_clI_errs is not None: save_dict['all_clI_errs'] = all_clI_errs
    if all_clg is not None: save_dict['all_clg'] = all_clg
    if all_clg_errs is not None: save_dict['all_clg_errs'] = all_clg_errs
    if all_clx is not None: save_dict['all_clx'] = all_clx
    if all_clx_errs is not None: save_dict['all_clx_errs'] = all_clx_errs
    if cl_cross_poisson is not None: save_dict['cl_cross_poisson'] = cl_cross_poisson
    
    print('Saving to ', save_fpath)
    np.savez(save_fpath, **save_dict)

    # np.savez(save_fpath, lb=lb, clI_comb=clI_comb, clI_err_comb=clI_err_comb)

        
    