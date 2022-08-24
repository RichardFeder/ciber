import numpy as np
import numpy.ctypeslib as npct
import ctypes
from ctypes import c_int, c_double
import astropy.units as u
from astropy import constants as const
from astropy.io import fits
import scipy.signal
from mock_galaxy_catalogs import *
from helgason import *
from ciber_data_helpers import *
from cross_spectrum_analysis import *
from filtering_utils import calculate_plane, fit_gradient_to_map

from PIL import Image
from image_eval import psf_poly_fit, image_model_eval
import sys

from numpy.fft import fftshift as fftshift
from numpy.fft import ifftshift as ifftshift
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2



def compute_star_gal_mask(stellar_cat, galaxy_cat, star_mag_idx=2, gal_mag_idx=3, m_max=18.4, return_indiv_masks=False):
    
    '''star and galaxy catalogs reported with AB magnitudes. 
    Default indices assume stellar catalog already passed through make_synthetic_trilegal_cat()'''
    
    filt_star_cat = filter_trilegal_cat(stellar_cat, I_band_idx=star_mag_idx, m_max=m_max)
    
    cmock = ciber_mock()
    filt_gal_cat = cmock.catalog_mag_cut(galaxy_cat, galaxy_cat[:, gal_mag_idx], m_min=0., m_max=m_max)
    
    mask_filt_star = mask_from_cat(filt_x, mag_idx=star_mag_idx)
    mask_gal = mask_from_cat(bright_cat, mag_idx=gal_mag_idx)
    
    joined_mask = mask_filt_star*mask_gal
    
    if return_indiv_masks:
        return mask_filt_star, mask_gal
    
    return joined_mask

def get_ciber_dgl_powerspec(dgl_fpath, inst, iras_color_facs=None, mapkey='iris_map', pixsize=7., dgl_bins=10):
    
    '''
    
    Parameterss
    ----------
    
    iras_color_facs: dictionary of scalar conversion factors between mean surface brightness at 100 micron vs. CIBER bands
    
    Returns
    -------
    
    lb: multipoles used
    cl: angular 1d power spectrum
    dgl_map: dgl map obtained from dgl_fpath
    
    
    '''
    if iras_color_facs is None:
#         iras_color_facs = dict({1:15., 2:8.}) # nW m^-2 sr^-1 (MJy sr^-1)^-1
        iras_color_facs = dict({1:6.4, 2:2.6}) # from MIRIS observations, Onishi++2018
        
    dgl = np.load(dgl_fpath)[mapkey]
    dgl_map = dgl*iras_color_facs[inst]
    dgl_map -= np.mean(dgl_map)
    
    lb, cl, cl_err = get_power_spec(dgl_map, pixsize=pixsize, nbins=dgl_bins)
    
    return lb, cl, dgl_map


def zl_grad_generator(theta_mag, nsim, dc=0, dimx=1024, dimy=1024):
    
    random_angles = np.random.uniform(0, 2*np.pi, nsim)
    
    if type(dc) != list:
        dc = [dc for x in range(nsim)]
    
    thetas = [[dc[r], theta_mag*np.cos(random_angle), theta_mag*np.sin(random_angle)] for r, random_angle in enumerate(random_angles)]
    planes = np.array([calculate_plane(theta, dimx=dimx, dimy=dimy) for theta in thetas])
    
    return planes, thetas

def generate_zl_realization(zl_level, apply_zl_gradient, theta_mag=0.01, dimx=1024, dimy=1024):

    if apply_zl_gradient:
        zl_realiz, theta_gen = zl_grad_generator(theta_mag, 1, dc=zl_level)
    else:
        zl_realiz = zl_level*np.ones((dimx, dimy))

    return zl_realiz


def grab_cl_pivot_fac(field_name, inst=1, dimx=1024, dimy=1024):
    cl_dgl_iras = np.load('data/fluctuation_data/TM'+str(inst)+'/dgl_sim/dgl_from_iris_model_TM'+str(inst)+'_'+field_name+'.npz')['cl']
    cl_pivot_fac = cl_dgl_iras[0]*dimx*dimy
    return cl_pivot_fac

def generate_custom_dgl_clustering(cbps, dgl_scale_fac=1, gen_ifield=6, ifield=None):
    
    
    field_name_gen = cbps.ciber_field_dict[gen_ifield]
    cl_pivot_fac_gen = grab_cl_pivot_fac(field_name_gen, inst=1, dimx=cbps.dimx, dimy=cbps.dimy)
    
    diff_realization = np.zeros((cbps.dimx, cbps.dimy))
    
    if ifield is not None:
        
        cl_pivot_fac_gen *= (dgl_scale_fac - 1)
        field_name = cbps.ciber_field_dict[ifield]
        
        cl_pivot_fac = grab_cl_pivot_fac(field_name, inst=1, dimx=cbps.dimx, dimy=cbps.dimy)
        
        _, _, diff_realization_varydgl = generate_diffuse_realization(cbps.dimx, cbps.dimy, power_law_idx=-3.0, scale_fac=cl_pivot_fac)
        diff_realization += diff_realization_varydgl

    else:
        cl_pivot_fac_gen *= dgl_scale_fac
    
    if cl_pivot_fac_gen > 0:
        _, _, diff_realization_gen = generate_diffuse_realization(cbps.dimx, cbps.dimy, power_law_idx=-3.0, scale_fac=cl_pivot_fac_gen)
        diff_realization += diff_realization_gen
        
    return diff_realization

def generate_diffuse_realization(N, M, power_law_idx=-3.0, scale_fac=1., B_ell_2d=None):

    ''' 
    This function takes a power law index and an amplitude and generates a Gaussian random field with power law PS. 
    Additionally, it (should) apply a beam correction in Fourier space to get a beam-convolved realization of the diffuse emission.

    Inputs
    ------
    
    N, M : `ints`. Dimension of image in pixels.
    power_law_idx : `float`. Power law index of diffuse component power spectrum.
        Default is -3.0 (DGL).
    scale_fac : `float`, optional. Scales amplitude of input power spectrum.
        Default is 1. 

    Returns
    -------

    ell_map : `np.array` of type `float`, shape (N, M).
    ps : `np.array` of type `float`, shape (N, M).
    diffuse_realiz : `np.array` of type `float`, shape (N, M).

    '''

    freq_x = fftshift(np.fft.fftfreq(N, d=1.0))
    freq_y = fftshift(np.fft.fftfreq(M, d=1.0))

    ell_x,ell_y = np.meshgrid(freq_x,freq_y)
    ell_x = ifftshift(ell_x)
    ell_y = ifftshift(ell_y)

    ell_map = np.sqrt(ell_x**2 + ell_y**2)

    ps = ell_map**power_law_idx
    ps[0,0] = 0.

    if B_ell_2d is None:
        B_ell_2d = np.ones_like(ps)

    diffuse_realiz = ifft2(np.sqrt(scale_fac*ps*B_ell_2d**2)*(np.random.normal(0, 1, size=(N, M)) + 1j*np.random.normal(0, 1, size=(N, M)))).real

    return ell_map, ps, diffuse_realiz

    

def generate_psf_template_bank(beta, rc, norm, n_fine_bin=10, nwide=17, pix_to_arcsec=7.):
    
    # instantiate the postage stamp
    Nsmall_post = 2*nwide+1
    Nlarge_post = Nsmall_post*n_fine_bin
    
    downsampled_psf_posts = np.zeros((n_fine_bin, n_fine_bin, Nsmall_post, Nsmall_post))
    dists = np.zeros((n_fine_bin, n_fine_bin))

    # for each subpixel, generate a PSF with beta/rc/norm parameters centered on the subpixel
    for upx in range(n_fine_bin):
        for upy in range(n_fine_bin):
            
            finex, finey = n_fine_bin*nwide+upx, n_fine_bin*nwide+upy

            # get radius from source center for each sub-pixel in PSF postage stamp
            radmap = make_radius_map(Nlarge_post, Nlarge_post, finex, finey, rc=rc)

            # convert pixels to arcseconds
            radmap *= pix_to_arcsec/float(n_fine_bin) 
            Imap_post = norm*np.power(1+radmap, -3*beta/2.)

            # downsample to native CIBER resolution
            downsamp_Imap = rebin_map_coarse(Imap_post, n_fine_bin)
            downsampled_psf_posts[upx, upy, :, :] = downsamp_Imap/np.sum(downsamp_Imap)
            dists[upx, upy] = (upx-n_fine_bin//2)**2+(upy-n_fine_bin//2)**2
            
    return downsampled_psf_posts, dists


def get_ciber_dgl_powerspec(dgl_fpath, inst, iras_color_facs=None, mapkey='iris_map', pixsize=7., dgl_bins=10):
    
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
        iras_color_facs = dict({1:15., 2:8.}) # nW m^-2 sr^-1 (MJy sr^-1)^-1
    
    dgl = np.load(dgl_fpath)[mapkey]
    dgl_map = dgl*iras_color_facs[inst]
    dgl_map -= np.mean(dgl_map)
    
    lb, cl, cl_err = get_power_spec(dgl_map, pixsize=pixsize, nbins=dgl_bins)
    
    return lb, cl, dgl_map
    
def get_q0_post(q, nwide):
    q0 = int(np.floor(q)-nwide)
    if q - np.floor(q) >= 0.5:
        q0 += 1
    return q0

def make_synthetic_trilegal_cat(trilegal_path, I_band_idx=16, H_band_idx=17, imdim=1024.):
    ''' 
    Generate synthetic catalog realization from TRILEGAL catalog. All this function does is draw uniformly random positions and 
    the TRILEGAL catalog magnitudes to make a new catalog, at the moment.
    '''
    trilegal = np.loadtxt(trilegal_path)
    nsrc = trilegal.shape[0]
    synthetic_cat = np.random.uniform(0, imdim, size=(nsrc, 2))
    synthetic_cat = np.array([synthetic_cat[:,0], synthetic_cat[:,1], trilegal[:,I_band_idx], trilegal[:,H_band_idx]]).transpose()
    print('synthetic cat has shape ', synthetic_cat.shape)
    return synthetic_cat

def ihl_conv_templates(psf=None, rvir_min=1, rvir_max=50, dimx=150, dimy=150):
    
    ''' 
    This function precomputes a range of IHL templates that can then be queried quickly when making mocks, rather than generating a
    separate IHL template for each source. This template is convolved with the mock PSF.

    Inputs:
        psf (np.array, default=None): point spread function used to convolve IHL template
        rvir_min/rvir_max (int, default=1/50): these set range of virial radii in pixels for convolved IHL templates.
        dimx, dimy (int, default=150): dimension of IHL template in x/y.

    Output:
        ihl_conv_temps (list of np.arrays): list of PSF-convolved IHL templates. 

    '''

    ihl_conv_temps = []
    rvir_range = np.arange(rvir_min, rvir_max).astype(np.float)
    for rvir in rvir_range:
        ihl = normalized_ihl_template(R_vir=rvir, dimx=dimx, dimy=dimy)
        if psf is not None:
            conv = scipy.signal.convolve2d(ihl, psf, 'same')
            ihl_conv_temps.append(conv)
        else:
            ihl_conv_temps.append(ihl)
    return ihl_conv_temps


def initialize_cblas_ciber(libmmult):

    print('initializing c routines and data structs')

    array_2d_float = npct.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")
    array_1d_int = npct.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
    array_2d_double = npct.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
    array_2d_int = npct.ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS")

    libmmult.pcat_model_eval.restype = None
    libmmult.pcat_model_eval.argtypes = [c_int, c_int, c_int, c_int, c_int, array_2d_float, array_2d_float, array_2d_float, array_1d_int, array_1d_int, array_2d_float, array_2d_float, array_2d_float, array_2d_double, c_int, c_int, c_int, c_int]


def normalized_ihl_template(dimx=50, dimy=50, R_vir=None):

    ''' This function generates a normalized template for intrahalo light, assuming a spherical projected profile.

    Inputs:
        dimx/dimy (int, default=50): dimension of template in x/y.
        R_vir (float, default=None): the virial radius for the IHL template in units of pixels.

    Output:
        ihl_map (np.array): IHL template normalized to unity
    
    '''

    if R_vir is None:
        R_vir = np.sqrt((dimx/2)**2+(dimy/2)**2)
    xx, yy = np.meshgrid(np.arange(dimx), np.arange(dimy), sparse=True)
    ihl_map = np.sqrt(R_vir**2 - (xx-(dimx/2))**2 - (yy-(dimy/2))**2) # assumes a spherical projected profile for IHL
    ihl_map[np.isnan(ihl_map)]=0
    ihl_map /= np.sum(ihl_map)
    return ihl_map

def rebin_map_coarse(original_map, Nsub):
    ''' Downsample map, taking average of downsampled pixels '''

    m, n = np.array(original_map.shape)//(Nsub, Nsub)
    
    return original_map.reshape(m, Nsub, n, Nsub).mean((1,3))


def save_mock_items_to_npz(filepath, catalog=None, srcmap_full=None, srcmap_nb=None, \
                           conv_noise=None, m_min=None, m_min_nb=None, ihl_map=None, m_lim=None):
    ''' Convenience file for saving mock observation files. '''
    np.savez_compressed(filepath, catalog=catalog, srcmap_full=srcmap_full, \
                        srcmap_nb=srcmap_nb, conv_noise=conv_noise, \
                        ihl_map=ihl_map, m_lim=m_lim, m_min=m_min, m_min_nb=m_min_nb)


def save_mock_to_fits(full_maps, cats, tail_name, full_maps_band2=None, m_tracer_max=None, m_min=None, m_max=None, inst=None, \
                     data_path='/Users/luminatech/Documents/ciber2/ciber/data/mock_cib_fftest/082321/', \
                     ifield_list=None, map_names=None, names=['x', 'y', 'redshift', 'm_app', 'M_abs', 'Mh', 'Rvir']):
    ''' This function is dedicated to converting mocks from ciber_mock.make_mock_ciber_map() to a fits file where they can be accessed.'''
    hdul = []
    hdr = fits.Header()

    if m_tracer_max is not None:
        hdr['m_tracer_max'] = m_tracer_max
    if m_min is not None:
        hdr['m_min'] = m_min
    if m_max is not None:
        hdr['m_max'] = m_max
    if inst is not None:
        hdr['inst'] = inst
        
    primary_hdu = fits.PrimaryHDU(header=hdr)
    hdul.append(primary_hdu)

    for c, cat in enumerate(cats):
        print('cat shape here is ', cat.shape)
        tab = Table([cat[:,i] for i in range(len(names))], names=names)
        cib_idx = c
        if ifield_list is not None:
            cib_idx = ifield_list[c]
            
        if map_names is not None:
            map_name = map_names[0]
            map_name2 = map_names[1]
        else:
            map_name = 'map'
            map_name2 = 'map2'
            
        hdu = fits.BinTableHDU(tab, name='tracer_cat_'+str(cib_idx))
        hdul.append(hdu)

        im_hdu = fits.ImageHDU(full_maps[c], name=map_name+'_'+str(cib_idx))
        hdul.append(im_hdu)
        
        if full_maps_band2 is not None:
            im_hdu2 = fits.ImageHDU(full_maps_band2[c], name=map_name2+'_'+str(cib_idx))
            hdul.append(im_hdu2)

    hdulist = fits.HDUList(hdul)
            
    hdulist.writeto(data_path+tail_name+'.fits', overwrite=True)


def virial_radius_2_reff(r_vir, zs, theta_fov_deg=2.0, npix_sidelength=1024.):
    ''' Converts virial radius to an effective size in pixels. Given radii in Mpc and associated redshifts,
    one can convert to an effective radius in pixels.

    Inputs: 
        r_vir (float, unit=[Mpc]): Virial radii
        zs (float): redshifts
        theta_fov_deg (float, unit=[degree], default=2.0): the maximum angle subtended by the FOV of mock image, in degrees
        npix_sidelength (int, unit=[pixel], default=1024): dimension of mock image in unit of pixels

    Outputs:
        Virial radius size in units of mock CIBER pixels

    '''
    d = cosmo.angular_diameter_distance(zs)*theta_fov_deg*np.pi/180.
    return (r_vir*u.Mpc/d)*npix_sidelength


def generate_full_mask_and_mll(cbps, ifield_list, sim_idxs, inst = 1, dat_type='mock', generate_starmask = True, generate_galmask = True,\
                              use_inst_mask = True, mag_lim_Vega=17.5, save_Mkk = True, save_mask = True, n_mkk_sims = 100,\
                              datestr = '062322', datestr_trilegal='062422', masktail = 'maglim18p0Vega', convert_AB_to_Vega = True, \
                             dm = 3., a1=195., b1=3.632, c1=8.0, \
                             load_preff_joint_mask = False, include_ff_mask = False, interp_mask_fn_fpaths=None):
    
    Vega_to_AB = dict({1:0.91 , 2: 1.39}) # add 0.91 to get AB magnitude in J band
    
    mag_lim_AB = mag_lim_Vega + Vega_to_AB[inst]

    print('mag_lim_AB is ', mag_lim_AB)
    
    param_combo = [a1, b1, c1]
    magkey_dict = dict({1:'j_m', 2:'h_m'})
    magkey = magkey_dict[inst]
    
    imarray_shape = (len(ifield_list), cbps.dimx, cbps.dimy)
    
    base_path = '/Volumes/Seagate Backup Plus Drive/Toolkit/Mirror/Richard/ciber_mocks/'

    mask_fpath = base_path+datestr+'/TM'+str(inst)+'/masks/'
    ff_joint_mask_fpath = mask_fpath+'ff_joint_masks/'
    mkk_fpath = base_path+datestr+'/TM'+str(inst)+'/mkk/'
    mkk_ff_fpath = mkk_fpath+'ff_joint_masks/'

    fpaths = [base_path, mask_fpath, ff_joint_mask_fpath, mkk_fpath, mkk_ff_fpath]

    make_fpaths(fpaths)
    
    interp_maskfn = None
    
    for s, sim_idx in enumerate(sim_idxs):
        
        if s==0:
            plot = True
        else:
            plot = False
        joint_masks = np.zeros(imarray_shape)
        for i, ifield in enumerate(ifield_list):
            
            field_name = cmock.ciber_field_dict[ifield]
            print(i, ifield, field_name)
            
            if interp_mask_fn_fpaths is not None:
                print('Loading masking radii from observed catalogs to get interpolated masking function..')
                interp_mask_file = np.load(interp_mask_fn_fpaths[i])
                cent_mags, rad_vs_cent_mags = interp_mask_file['cent_mags'], interp_mask_file['radii']
                max_mag = np.max(cent_mags)
                interp_maskfn = scipy.interpolate.interp1d(cent_mags[rad_vs_cent_mags!= 0], rad_vs_cent_mags[rad_vs_cent_mags != 0])

                a1, b1, c1, dm, alpha_m, beta_m = [None for x in range(6)]

            # instrument mask
            if use_inst_mask:
                cbps.load_mask(ifield, inst, masktype='maskInst_clean')
                joint_mask = cbps.maskInst_clean
            else:
                joint_mask = np.ones((cbps.dimx, cbps.dimy))
                
            if generate_starmask: 
                mock_trilegal_path = base_path+datestr_trilegal+'/trilegal/mock_trilegal_simidx'+str(sim_idx)+'_'+datestr_trilegal+'.fits'
                mock_trilegal = fits.open(mock_trilegal_path)
                mock_trilegal_cat = mock_trilegal['tracer_cat_'+str(ifield)].data

                if convert_AB_to_Vega:
                    print('adding to mock trilegal cat ', Vega_to_AB[inst])
                    mock_trilegal_cat['j_m'] -= Vega_to_AB[inst]

                mock_trilegal_map = mock_trilegal['trilegal_'+str(cbps.inst_to_band[inst])+'_'+str(ifield)].data
                print('mock trilegal cat has shape', mock_trilegal_cat.shape)
                
                star_cat = {magkey:mock_trilegal_cat[magkey].byteswap().newbyteorder(), 'x'+str(inst):mock_trilegal_cat['x'].byteswap().newbyteorder(), 'y'+str(inst):mock_trilegal_cat['y'].byteswap().newbyteorder()}
                star_cat_df = pd.DataFrame(star_cat)

                star_cat_df.columns = [magkey, 'x'+str(inst), 'y'+str(inst)]
                
                if interp_mask_fn_fpaths is not None:

                    # simulated sources changed to Vega magnitudes with convert_AB_to_Vega, masking function magnitudes in Vega units
                    starmask, radii_stars = mask_from_cat(cat_df = star_cat_df, mag_lim_min=0, inst=inst,\
                                                                mag_lim=mag_lim_Vega, interp_maskfn=interp_maskfn,\
                                                          magstr=magkey, Vega_to_AB=0., dimx=cbps.dimx, dimy=cbps.dimy, plot=False, \
                                                         interp_max_mag = max_mag)
                
                else:
                    starmask, radii_stars_simon, radii_stars_Z14, alpha_m, beta_m = get_masks(star_cat_df, param_combo, intercept_mag, mag_lim_Vega, dm=dm, magstr=magkey, inst=inst, dimx=cbps.dimx, dimy=cbps.dimy, verbose=True)
                
                joint_mask *= starmask

            if generate_galmask:
                midxdict = dict({'x':0, 'y':1, 'redshift':2, 'm_app':3, 'M_abs':4, 'Mh':5, 'Rvir':6})
                mock_gal = fits.open(base_path+datestr+'/TM'+str(inst)+'/cib_with_tracer_5field_set'+str(sim_idx)+'_'+datestr+'_TM'+str(inst)+'.fits')
                mock_gal_cat = mock_gal['tracer_cat_'+str(ifield)].data

                if convert_AB_to_Vega:
                    mock_gal_cat['m_app'] -= Vega_to_AB[inst]

                gal_cat = {'m_app':mock_gal_cat['m_app'].byteswap().newbyteorder(), 'x'+str(inst):mock_gal_cat['x'].byteswap().newbyteorder(), 'y'+str(inst):mock_gal_cat['y'].byteswap().newbyteorder()}
                gal_cat_df = pd.DataFrame(gal_cat, columns = ['m_app', 'x'+str(inst), 'y'+str(inst)]) # check magnitude system of Helgason model
                
                if interp_mask_fn_fpaths is not None:                    
                    galmask, radii_gals = mask_from_cat(cat_df = gal_cat_df, mag_lim_min=0, inst=inst,\
                                            mag_lim=mag_lim_Vega, interp_maskfn=interp_maskfn, magstr='m_app', Vega_to_AB=0., dimx=cbps.dimx, dimy=cbps.dimy, plot=False, \
                                                       interp_max_mag = max_mag)

                else:
                    galmask, radii_stars_simon, radii_stars_Z14, alpha_m, beta_m = get_masks(gal_cat_df, param_combo, intercept_mag, mag_lim_Vega, dm=dm, magstr='m_app', inst=inst, dimx=cbps.dimx, dimy=cbps.dimy, verbose=True)
                    
                    
                if len(radii_gals) > 0:
                    print('len radii gals is ', len(radii_gals))
                    joint_mask *= galmask
                  

            print(float(np.sum(joint_mask))/float(1024**2))

            joint_masks[i] = joint_mask.copy()
            
#             if plot:
#                 plot_map(joint_masks[i], title='joint mask i='+str(i))
            
        for i, ifield in enumerate(ifield_list):

            if save_mask:
                if plot and i==0:
                    plot_map(joint_masks[i], title='joint mask, simidx = '+str(sim_idx))
                hdul = write_mask_file(np.array(joint_masks[i]), ifield=ifield, inst=inst, sim_idx=sim_idx, generate_galmask=generate_galmask, \
                                      generate_starmask=generate_starmask, use_inst_mask=use_inst_mask, dat_type=dat_type, mag_lim_AB=mag_lim_AB, \
                                      a1=a1, b1=b1, c1=c1, dm=dm, alpha_m=alpha_m, beta_m=beta_m)

                hdul.writeto(mask_fpath+'joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(sim_idx)+'_'+masktail+'.fits', overwrite=True)

            if save_Mkk:

                Mkk = cbps.Mkk_obj.get_mkk_sim(joint_masks[i], n_mkk_sims, n_max_persplit=50, store_Mkks=False)
                inv_Mkk = compute_inverse_mkk(Mkk)
                if plot:
                    plot_mkk_matrix(inv_Mkk, inverse=True, symlogscale=True)

                hdul = write_Mkk_fits(Mkk, inv_Mkk, ifield, inst, sim_idx=sim_idx, generate_starmask=generate_starmask, generate_galmask=generate_galmask, \
                                     use_inst_mask=use_inst_mask, dat_type=dat_type, mag_lim_AB=mag_lim_Vega)


                hdul.writeto(mkk_fpath+'mkk_estimate_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(sim_idx)+'_'+masktail+'.fits', overwrite=True)




class ciber_mock():
    sb_intensity_unit = u.nW/u.m**2/u.steradian # helpful unit to have on hand

    # ciberdir = '/Users/richardfeder/Documents/caltech/ciber2'
    darktime_name_dict = dict({36265:['NEP', 'BootesA'],36277:['SWIRE', 'NEP', 'BootesB'], \
    40030:['DGL', 'NEP', 'Lockman', 'elat10', 'elat30', 'BootesB', 'BootesA', 'SWIRE']})
    ciber_field_dict = dict({4:'elat10', 5:'elat30',6:'BootesB', 7:'BootesA', 8:'SWIRE'})

    pix_width = 7.*u.arcsec
    pix_sr = ((pix_width.to(u.degree))*(np.pi/180.0))**2*u.steradian / u.degree # pixel solid angle in steradians
    lam_effs = np.array([1.05, 1.79])*1e-6*u.m # effective wavelength for bands

    sky_brightness = np.array([300., 370.])*sb_intensity_unit
    instrument_noise = np.array([33.1, 17.5])*sb_intensity_unit

    helgason_to_ciber_rough = dict({1:'J', 2:'H'})

    fac_upsample = 10 # upsampling factor for PSF interpolation

    def __init__(self, pcat_model_eval=False, ciberdir='/Users/richardfeder/Documents/caltech/ciber2/ciber/', \
                nx=1024, ny=1024, psf_template=None, psf_temp_bank=None):

        for attr, valu in locals().items():
            if '__' not in attr and attr != 'gdat':
                setattr(self, attr, valu)

        if pcat_model_eval:
            try:
                self.libmmult = npct.load_library('pcat-lion', '.')
                # if sys.version_info[0] <3:
                #     self.libmmult = npct.load_library('pcat-lion', '.')
                # else:
                #     self.libmmult = npct.load_library('pcat-lion.so', self.ciberdir)
            except:
                print('pcat-lion.so not loading, trying blas.so now..')
                self.libmmult = ctypes.cdll['./blas.so'] # not sure how stable this is, trying to find a good Python 3 fix to deal with path configuration


            initialize_cblas_ciber(self.libmmult)


    def catalog_mag_cut(self, cat, m_arr, m_min, m_max):
        ''' Given a catalog (cat), magnitudes, and a magnitude cut range, return the filtered catalog ''' 
        magnitude_mask_idxs = np.array([i for i in range(len(m_arr)) if m_arr[i]>=m_min and m_arr[i] <= m_max])
        if len(magnitude_mask_idxs) > 0:
            catalog = cat[magnitude_mask_idxs,:]
        else:
            catalog = cat
        return catalog   

    def get_catalog(self, catname):
        ''' Load catalog from .txt file ''' 
        cat = np.loadtxt(self.ciberdir+'/data/'+catname)
        x_arr = cat[0,:]
        y_arr = cat[1,:] 
        m_arr = cat[2,:]
        return x_arr, y_arr, m_arr
        
    def get_psf(self, ifield=4, band=0, multfac=7.0, nbin=0., fac_upsample=10, make_ds_temp_bank=False, poly_fit=True, nwide=17, tail_path='data/psf_model_dict_updated_081121_ciber.npz', \
                verbose=False):
        
        ''' Function loads the relevant quantities for the CIBER point spread function 

        Inputs
        ------
        
        ifield : type 'int'. Index of CIBER field. Default is 4.

        band : type 'int'. Zero-indexed instrument ID (band 0 = TM1, band 1 = TM2)

        multfac : type 'float'. Converts radius maps from pixels to arcseconds.

        nbin : type 'int'.

        fac_upsample : type 'int'. Upsampling factor relative to CIBER native pixel resolution. Default is 10.

        make_ds_temp_bank : type 'boolean'. If True, function computes upsampled PSFs evaluated at (n_fine_bin x n_fine_bin) pixel locations, 
                            and then downsamples the interpolated PSFs to CIBER resolution.

        nwide : type 'int'. PSF postage templates have shape (nwide + 1) x (nwide + 1). Also index for central pixel of postage templates

        Returns
        -------



        '''


        # beta, rc, norm = find_psf_params(self.ciberdir+tail_path, tm=band+1, field=self.ciber_field_dict[ifield])
        # new psf test
        beta, rc, norm = load_psf_params_dict(band+1, ifield=ifield, tail_path=self.ciberdir+tail_path, verbose=verbose)
        Nlarge = self.nx+30+30 

        if make_ds_temp_bank:
            print('Generating sub-pixel interpolated PSF template bank')
            psf_postage_stamps, subpixel_dists = generate_psf_template_bank(beta, rc, norm, fac_upsample=fac_upsample)

        radmap = make_radius_map(2*Nlarge+nbin, 2*Nlarge+nbin, Nlarge+nbin, Nlarge+nbin, rc)*multfac # is the input supposed to be 2d?
        
        self.psf_full = norm * np.power(1 + radmap, -3*beta/2.)
        self.psf_full /= np.sum(self.psf_full)     
        self.psf_template = self.psf_full[Nlarge-nwide:Nlarge+nwide+1, Nlarge-nwide:Nlarge+nwide+1]
        # self.psf_template = self.psf_full[Nlarge-nwide:Nlarge+nwide, Nlarge-nwide:Nlarge+nwide]
        
        if verbose:
            print('imap center has shape', self.psf_template.shape)

        if poly_fit:

            psf = np.zeros((70,70))
            psf[0:35,0:35] = self.psf_template

            psf = np.array(Image.fromarray(psf).resize((350, 350), resample=Image.LANCZOS))
            psfnew = np.array(psf[0:175, 0:175])
            psfnew[0:173,0:173] = psf[2:175,2:175]  # shift due to lanczos kernel
            psfnew[0:173,0:173] = psf[2:175,2:175]  # shift due to lanczos kernel
            self.cf = psf_poly_fit(psfnew, nbin=5)

        return beta, rc, norm
        

    def get_darktime_name(self, flight, field):
        return self.darktime_name_dict[flight][field-1]

    def mag_2_jansky(self, mags):
        ''' unit conversion from monochromatic AB magnitudes to units of Jansky. In the presence of a continuous bandpass this needs to change ''' 
        return 3631*u.Jansky*10**(-0.4*mags)

    def mag_2_nu_Inu(self, mags, band):
        ''' unit conversion from magnitudes to intensity at specific wavelength ''' 
        jansky_arr = self.mag_2_jansky(mags)
        return jansky_arr.to(u.nW*u.s/u.m**2)*const.c/(self.pix_sr*self.lam_effs[band])

    def make_srcmap_temp_bank(self, ifield, inst, cat,\
                             flux_idx=-1, n_fine_bin=10, nwide=17,  beta=None, rc=None, norm=None, \
                             tail_path='data/psf_model_dict_updated_081121_ciber.npz', verbose=False):

        if beta is None:
            beta, rc, norm = load_psf_params_dict(inst, ifield=ifield, tail_path=self.ciberdir+tail_path, verbose=verbose)


        if self.psf_temp_bank is None:
            self.psf_temp_bank, dists = generate_psf_template_bank(beta, rc, norm, n_fine_bin=n_fine_bin, nwide=nwide)

        dim_psf_post = nwide*2 + 1
        srcmap = np.zeros((self.nx*2, self.ny*2))
        
        for n in range(len(cat)):
            
            # find subpixel corresponding to source. Note indices are swapped for iy_sub/ix_sub w.r.t. rest of function, but I think this is due to the template bank
            # convention being flipped and this 'undoes' that
            iy_sub = int(((n_fine_bin//2) + np.floor(n_fine_bin*(cat[n, 0]-np.floor(cat[n, 0]))))%n_fine_bin)
            ix_sub = int(((n_fine_bin//2) + np.floor(n_fine_bin*(cat[n, 1]-np.floor(cat[n, 1]))))%n_fine_bin)
                        
            # get boundary index for PSF postage stamp
            x0 = get_q0_post(cat[n, 0], nwide)
            y0 = get_q0_post(cat[n, 1], nwide)
            # insert into map  
            srcmap[self.nx//2 + x0:self.nx//2+x0+dim_psf_post, self.ny//2 + y0:self.ny//2+y0+dim_psf_post] += cat[n, flux_idx]*self.psf_temp_bank[ix_sub, iy_sub]
            
        # downscale map
        return srcmap[self.nx//2:3*self.nx//2, self.ny//2:3*self.ny//2]


    def make_srcmap(self, ifield, cat, flux_idx=-1, band=0, nbin=0., nwide=17, multfac=7.0, \
                    pcat_model_eval=False, libmmult=None, dx=2.5, dy=-1.0, finePSF=False, n_fine_bin=10, \
                    make_ds_temp_bank=False):
        
        ''' 
        This function takes in a catalog, finds the PSF for the specific ifield, makes a PSF template and then populates an image with 
        model sources. When we use galaxies for mocks we can do this because CIBER's angular resolution is large enough that galaxies are
        well modeled as point sources. 

        Note (5/21/20): This function currently only places sources at integer locations, which should probably change
        for any real implementation used in analysis.
        
        Note (5/22/20): with the integration of PCAT's model evaluation routines, we can now do sub-pixel source placement
        and it is a factor of 6 faster than the original implementation when at scale (e.g. catalog sources down to 25th magnitude)
        
        dx and dy are offsets meant to be used when comparing PCAT's model evaluation with the original, but is not strictly 
        necessary for actual model evaluation. In other words, these can be set to zero when not doing comparisons.
        '''

        Nsrc = cat.shape[0]

        srcmap = np.zeros((self.nx*2, self.ny*2))
        
        Nlarge = self.nx+30+30 

        if self.psf_template is None:

            print('generating psf template because it was none')
            # new psf test
            beta, rc, norm = self.get_psf(ifield=ifield, band=band, multfac=multfac, poly_fit=pcat_model_eval, nwide=nwide, \
                                        tail_path='data/psf_model_dict_updated_081121_ciber.npz', make_ds_temp_bank=make_ds_temp_bank)
            print('beta/rc/norm: ', beta, rc, norm)
        if pcat_model_eval:

            srcmap = image_model_eval(np.array(cat[:,0]).astype(np.float32)+dx, np.array(cat[:,1]).astype(np.float32)-dy ,np.array(cat[:, flux_idx]).astype(np.float32),0., (self.nx, self.ny), 35, self.cf, lib=self.libmmult.pcat_model_eval)

            return srcmap
        else:

            if finePSF:

                Nlarge_post  = (2*nwide+1)*n_fine_bin

                for i in range(Nsrc):
                    xrel = n_fine_bin*(nwide-(np.ceil(cat[i,0]) - cat[i,0]))
                    yrel = n_fine_bin*(nwide-(np.ceil(cat[i,1]) - cat[i,1]))

                    radmap = make_radius_map(Nlarge_post, Nlarge_post, xrel, xrel, rc)*multfac # is the input supposed to be 2d?

                Imap_post = norm*np.power(1+radmap, -3*beta/2.)

            else:
                xs = np.round(cat[:,0]).astype(np.int32)
                ys = np.round(cat[:,1]).astype(np.int32)

                for i in range(Nsrc):
                    srcmap[Nlarge//2+2+xs[i]-nwide:Nlarge//2+2+xs[i]+nwide+1, Nlarge//2-1+ys[i]-nwide:Nlarge//2-1+ys[i]+nwide+1] += self.psf_template*cat[i, flux_idx]
        
            return srcmap[self.nx//2+30:3*self.nx//2+30, self.ny//2+30:3*self.ny//2+30]

   
    def make_ihl_map(self, map_shape, cat, ihl_frac, flux_idx=-1, dimx=150, dimy=150, psf=None, extra_trim=20):
        
        ''' 
        Given a catalog amnd a fractional ihl contribution, this function precomputes an array of IHL templates and then 
        uses them to populate a source map image.
        '''

        rvirs = virial_radius_2_reff(r_vir=cat[:,6], zs=cat[:,2])
        rvirs = rvirs.value

        ihl_temps = ihl_conv_templates(psf=psf)

        ihl_map = np.zeros((map_shape[0]+dimx+extra_trim, map_shape[1]+dimy+extra_trim))

        for i, src in enumerate(cat):
            x0 = np.floor(src[0]+extra_trim/2)
            y0 = np.floor(src[1]+extra_trim/2)
            ihl_map[int(x0):int(x0+ihl_temps[0].shape[0]), int(y0):int(y0 + ihl_temps[0].shape[1])] += ihl_temps[int(np.ceil(rvirs[i])-1)]*ihl_frac*src[flux_idx]

        return ihl_map[(ihl_temps[0].shape[0] + extra_trim)/2:-(ihl_temps[0].shape[0] + extra_trim)/2, (ihl_temps[0].shape[0] + extra_trim)/2:-(ihl_temps[0].shape[0] + extra_trim)/2]
     

    def mocks_from_catalogs(self, catalog_list, ncatalog, mock_data_directory=None, m_min=9., m_max=30., m_tracer_max=25., \
                        ihl_frac=0.0, ifield=4, ifield_list=None, band=0, inst=None, save=False, extra_name='', pcat_model_eval=False, add_noise=False, cat_return='tracer', \
                        temp_bank=True, n_fine_bin=10, convert_tracer_to_Vega=False, Vega_to_AB=0.91):
    
        if inst is None:
            inst = band+1
        srcmaps_full, catalogs, noise_realizations, ihl_maps = [[] for x in range(4)]
        
        print('m_min = ', m_min)
        print('m_max = ', m_max)
        print('m_tracer_max = ', m_tracer_max)
        if extra_name != '':
            extra_name = '_'+extra_name

        if save and mock_data_directory is None:
            print('Please provide a mock data directory to save to..')
            return
        
        if ifield_list is None:
            ifield_list = [ifield for c in range(ncatalog)]

        for c in range(ncatalog):

            cat_full = self.catalog_mag_cut(catalog_list[c], catalog_list[c][:,3], m_min, m_max)

            if convert_tracer_to_Vega:
                tracer_cat = self.catalog_mag_cut(catalog_list[c], catalog_list[c][:,3], m_min, m_tracer_max+Vega_to_AB)
            else:
                tracer_cat = self.catalog_mag_cut(catalog_list[c], catalog_list[c][:,3], m_min, m_tracer_max)

            I_arr_full = self.mag_2_nu_Inu(cat_full[:,3], band)
            cat_full = np.hstack([cat_full, np.expand_dims(I_arr_full.value, axis=1)])

            if temp_bank:
                srcmap_full = self.make_srcmap_temp_bank(ifield_list[c], inst, cat_full, flux_idx=-1, n_fine_bin=n_fine_bin, nwide=17)
            else:
                srcmap_full = self.make_srcmap(ifield_list[c], cat_full, band=band, pcat_model_eval=pcat_model_eval, nwide=17)

            
            if cat_return=='tracer':
                catalogs.append(tracer_cat)
            else:
                catalogs.append(cat_full)

            srcmaps_full.append(srcmap_full)
            
            if ihl_frac > 0:
                print('Making IHL map..')
                ihl_map = self.make_ihl_map(srcmap_full.shape, cat_full, ihl_frac, psf=self.psf_template)
                ihl_maps.append(ihl_map)
                if save:
                    print('Saving results..')
                    np.savez_compressed(mock_data_directory+'ciber_mock_'+str(c)+'_mmin='+str(m_min)+extra_name+'.npz', \
                                        catalog=tracer_cat, srcmap_full=srcmap_full, conv_noise=conv_noise,\
                                        ihl_map=ihl_map)
            else:
                if save:
                    print('Saving results..')
                    np.savez_compressed(mock_data_directory+'ciber_mock_'+str(c)+'_mmin='+str(m_min)+extra_name+'.npz', \
                                        catalog=tracer_cat, srcmap_full=srcmap_full, convert_tracer_to_Vega=convert_tracer_to_Vega, Vega_to_AB=Vega_to_AB)
                   

        if ihl_frac > 0: 
            return srcmaps_full, catalogs, noise_realizations, ihl_maps
        else:
            return srcmaps_full, catalogs, noise_realizations

    def make_mock_ciber_map(self, ifield, m_min, m_max, ifield_list=None, mock_cat=None, band=0, ihl_frac=0., ng_bins=8,\
                            zmin=0.0, zmax=2.0, pcat_model_eval=False, ncatalog=1, add_noise=False, cat_return='tracer', m_tracer_max=20., \
                            temp_bank=True, convert_tracer_to_Vega=False):
        """ 
        This is the parent function that uses other functions in the class to generate a full mock catalog/CIBER image. If there is 
        no mock catalog input, the function draws a galaxy catalog from the Helgason model with the galaxy_catalog() class. With a catalog 
        in hand, the function then imposes any cuts on magnitude, computes mock source intensities and then generates the corresponding 
        source maps/ihl maps/noise realizations that go into the final CIBER mock.

        Inputs
        ------
        
        ifield (int): field from which to get PSF parameters
        m_min/m_max (float): minimum and maximum source fluxes to use from mock catalog in image generation
            
        ifield_list (list, default=None): this can be used for making sets of CIBER observations from realistic fields, 
                                                e.g., ifield_list = [4, 5, 6, 7, 8] 
        mock_cat (np.array, default=None): this can be used to specify a catalog to generate beforehand rather than sampling a random one
        band (int, default=0): CIBER band of mock image, either 0 (band J) or 1 (band H)
        ihl_frac (float, default=0.0): determines amplitude of IHL around each source as fraction of source flux If ihl_frac=0.2, it means
                you place a source in the map with flux f and then add a template with amplitude 0.2*f. It's an additive feature, not dividing 
                source flux into 80/20 or anything like that.

        ng_bins (int, default=5): number of redshift bins to use when making mock map/catalog. Each redshift bin has its own generated 
                clustering field drawn with the lognormal technique. 
        zmin/zmax (float, default=0.01/5.0): form redshift range from which to draw galaxies from Helgason model.

        add_noise (boolean, default=False): determines whether to make gaussian noise realization based on sky brightnesses. in most cases
                                                this will be outdated since we use a more detailed read noise/photon noise model

        temp_bank (boolean, default=True): if True, image generation is done using a precomputed template bank of PSFs that are interpolated 
                                                and downsampled to CIBER resolution.

        Returns
        -------
        
        full_map/srcmap/noise/ihl_map (np.array): individual and combined components of mock CIBER map
        cat (np.array): galaxy catalog for mock CIBER map
        psf_template (np.array): psf template used to generate mock CIBER sources
            
        """

        inst = band+1
        band_helgason = self.helgason_to_ciber_rough[inst]
        print('helgason band is ', band_helgason)
        if mock_cat is not None:
            m_arr = []
            cat = mock_cat
        else:
            mock_galaxy = galaxy_catalog()
            cat = mock_galaxy.generate_galaxy_catalogs(band=band_helgason, ng_bins=ng_bins, zmin=zmin, zmax=zmax, n_catalogs=ncatalog, m_min=m_min, m_max=m_max)
        
        # tracer catalogs are selected using cuts on Vega mag, but actual magnitudes in tracer catalog are still in AB magnitude system (6/23/22)
        if ihl_frac > 0:
            srcmaps, cats, noise_realizations, ihl_maps = self.mocks_from_catalogs(cat, ncatalog, m_min=m_min,\
                                                                                   m_max=m_max,ifield=ifield, ifield_list=ifield_list, band=band,pcat_model_eval=pcat_model_eval, ihl_frac=ihl_frac, \
                                                                                   add_noise=add_noise, cat_return=cat_return, m_tracer_max=m_tracer_max, temp_bank=temp_bank, convert_tracer_to_Vega=convert_tracer_to_Vega)
        else:
            srcmaps, cats, noise_realizations = self.mocks_from_catalogs(cat, ncatalog, m_min=m_min,\
                                                                                   m_max=m_max,ifield=ifield, ifield_list=ifield_list, band=band,pcat_model_eval=pcat_model_eval, \
                                                                                   add_noise=add_noise, cat_return=cat_return, m_tracer_max=m_tracer_max, \
                                                                                   temp_bank=temp_bank, convert_tracer_to_Vega=convert_tracer_to_Vega)

        full_maps = np.zeros((ncatalog, self.nx, self.ny))
    
        for c in range(ncatalog):
            full_maps[c] = srcmaps[c]

            if ihl_frac > 0:
                full_maps[c] += ihl_maps[c]

        if ihl_frac > 0:
            return full_maps, srcmaps, noise_realizations, ihl_maps, cats
        else:
            return full_maps, srcmaps, noise_realizations, cats



