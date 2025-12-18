import numpy as np
import numpy.ctypeslib as npct
import ctypes
from ctypes import c_int, c_double
import astropy.units as u
from astropy import constants as const
from astropy.io import fits
import scipy.signal
# from mock_galaxy_catalogs import *
# from helgason import *
# from cross_spectrum_analysis import *
# from ciber_beam import *
# from noise_model import CIBER_NoiseModel
from masking_utils import *
from numerical_routines import *
import config
# from powerspec_utils import write_mask_file, write_Mkk_fits
from mkk_parallel import compute_inverse_mkk, plot_mkk_matrix

from filtering_utils import calculate_plane, fit_gradient_to_map
import os 
from PIL import Image
# from image_eval import psf_poly_fit, image_model_eval
import sys
import time

from numpy.fft import fftshift as fftshift
from numpy.fft import ifftshift as ifftshift
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2

from ciber_data_file_utils import *

# numerical_routines.py
# def get_q0_post(q, nwide):

# ciber_data_file_utils.py
# def save_mock_items_to_npz(filepath, catalog=None, srcmap_full=None, srcmap_nb=None, \
# def save_mock_to_fits(full_maps, cats, tail_name=None, full_maps_band2=None, m_tracer_max=None, m_min=None, m_max=None, inst=None, \
# def make_fpaths(fpaths):

def get_ciber_dgl_powerspec(dgl_fpath, inst, iras_color_facs=None, mapkey='iris_map', pixsize=7., dgl_bins=10, mode='Onishi'):
    
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
        if mode=='Onishi':
            iras_color_facs = dict({1:6.4, 2:2.6}) # from MIRIS observations, Onishi++2018
        elif mode=='Z14':
            iras_color_facs = dict({1:15., 2:8.}) # nW m^-2 sr^-1 (MJy sr^-1)^-1

    dgl = fits.open(dgl_fpath)[1].data
        
    dgl_map = dgl*iras_color_facs[inst]
    dgl_map -= np.mean(dgl_map)
    
    lb, cl, cl_err = get_power_spec(dgl_map, pixsize=pixsize, nbins=dgl_bins)
    
    return lb, cl, dgl_map


def zl_grad_generator(theta_mag=0.01, nsim=1, thetas=None, dc=0, dimx=1024, dimy=1024):
    
    if thetas is None:
        random_angles = np.random.uniform(0, 2*np.pi, nsim)
        
        if type(dc) != list:
            dc = [dc for x in range(nsim)]
        
        thetas = [[dc[r], theta_mag*np.cos(random_angle), theta_mag*np.sin(random_angle)] for r, random_angle in enumerate(random_angles)]
    
    planes = np.array([calculate_plane(theta, dimx=dimx, dimy=dimy) for theta in thetas])
    
    return planes, thetas

def generate_zl_realization(zl_level, apply_zl_gradient, theta_mag=0.01, thetas=None, dimx=1024, dimy=1024):

    if apply_zl_gradient:
        if thetas is None:
            zl_realiz, theta_gen = zl_grad_generator(theta_mag=theta_mag, nsim=1, dc=zl_level, dimx=dimx, dimy=dimy)
        else:
            zl_realiz, theta_gen = zl_grad_generator(thetas=thetas, dimx=dimx, dimy=dimy)

    else:
        zl_realiz = zl_level*np.ones((dimx, dimy))

    return zl_realiz


def grab_cl_pivot_fac(ifield, inst, dimx=1024, dimy=1024):
    iris_basepath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/iris_regrid/'

    # iris_basepath = config.exthdpath+'ciber_fluctuation_data/TM'+str(inst)+'/iris_regrid/'
    cl_dgl_iras = np.load(iris_basepath+'dgl_from_iris_model_TM'+str(inst)+'_ifield'+str(ifield)+'.npz')['cl']

    # cl_dgl_iras = np.load('data/fluctuation_data/TM'+str(inst)+'/dgl_sim/dgl_from_iris_model_TM'+str(inst)+'_'+field_name+'.npz')['cl']
    cl_pivot_fac = cl_dgl_iras[0]*dimx*dimy
    return cl_pivot_fac


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


def generate_subthresh_ptsrc_maps(nsim, inst, m_min=17.5, m_max=23, save=True, nsim0=0, \
                                     ifield_list = [4, 5, 6, 7, 8], m_tracer_max = 18.0):
    
    ''' Make mocks specifically for estimating the additional bias from pointing offset in half exposure differences '''


    band = inst - 1
    datestr = '111323'

    config_dict, pscb_dict, float_param_dict, fpath_dict = return_default_cbps_dicts()
    ciber_mock_fpath = config.ciber_basepath+'data/ciber_mocks/'

    fpath_dict, list_of_dirpaths, base_path, trilegal_base_path = set_up_filepaths_cbps(fpath_dict, inst, 'ciber_lam_obs', datestr,\
                                                                                        datestr_trilegal=datestr, data_type='observed', \
                                                                                       save_fpaths=True)

    cbps = CIBER_PS_pipeline()
    cmock = ciber_mock()

    for cib_setidx in np.arange(nsim0, nsim):
        print('CIB set idx = ', cib_setidx)
        ncatalog = 5

        srcmaps, cats = cmock.make_mock_ciber_map(m_min, m_max, ifield_list=ifield_list, \
                                                        band=band, load_cl_limber_file=True, plot=False, \
                                                        ncatalog=ncatalog, temp_bank=True, \
                                                        simulate_pointing_offset=False, \
                                                        m_tracer_max=m_tracer_max, use_ciber_lam_eff=True, \
                                                        mode=None)
        
        data_path = test_set_fpath = fpath_dict['ciber_mock_fpath']+datestr+'/TM'+str(inst)+'/cib_dffs/'

        tail_name = 'cib_dffs_default_5field_set'+str(cib_setidx)+'_'+datestr+'_TM'+str(inst)

        print(data_path+tail_name)
        print(os.path.isdir(data_path))

        save_mock_to_fits(srcmaps, cats, tail_name, data_path=data_path, m_tracer_max=m_tracer_max, m_min=m_min, m_max=m_max,\
                          ifield_list=ifield_list, map_names=['cib_'+cbps.inst_to_band[inst], 'dummy'])
    
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

def compute_extended_psf_mocks(inst, ifield_list, masking_maglim, datestr, convert_Vega_to_AB=True, \
                              simidx0=0, nsims=100, fpath_dict=None, tailstr='with_dpoint', simstr=None, \
                              sigma_xy=None):
    
    trilegal_base_path = fpath_dict['trilegal_base_path']
    cib_realiz_path = fpath_dict['cib_realiz_path']
    bls_fpath = fpath_dict['bls_base_path']+'/bl_est_postage_stamps_TM'+str(inst)+'_081121.npz'
    tempbank_dirpath = fpath_dict['subpixel_psfs_path']+'/'
    mask_base_path = fpath_dict['mask_base_path']
    mkk_base_path = fpath_dict['mkk_base_path']

    cbps = CIBER_PS_pipeline()
    cmock = ciber_mock()
    
    trilegal_band_dict = dict({1:'j_m', 2:'h_m'})
    trilegal_fnu_dict = dict({1:'j_nuInu', 2:'h_nuInu'})
    bandstr, Iarr_str = trilegal_band_dict[inst], trilegal_fnu_dict[inst]
    print('bandstr, Iarr_str = ', bandstr, Iarr_str)
    
    cib_bandstr = 'm_app'
    
    if convert_Vega_to_AB:
        mag_limit_match = masking_maglim + cmock.Vega_to_AB[inst]
    else:
        mag_limit_match = masking_maglim

    
    B_ells = np.load(bls_fpath)['B_ells_post']
    
    src_map_cls = np.zeros((len(ifield_list), nsims, cbps.n_ps_bin))
    
    mask_tail = 'maglim_'+str(masking_maglim)+'_Vega_test'

    for setidx in np.arange(simidx0, nsims):
        
        print('On set ', setidx, 'of ', nsims)
        trilegal_path = trilegal_base_path+'/mock_trilegal_simidx'+str(setidx)+'_'+datestr+'.fits'
        mock_trilegal = fits.open(trilegal_path)
        
        mock_gal = fits.open(cib_realiz_path+'/cib_with_tracer_'+simstr+'_5field_set'+str(setidx)+'_'+datestr+'_TM'+str(inst)+'.fits')
        
        for fieldidx, ifield in enumerate(ifield_list):
            
            mask_fpath = mask_base_path+'/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(setidx)+'_'+mask_tail+'.fits'
            mask = fits.open(mask_fpath)['joint_mask_'+str(ifield)].data
        
            inv_Mkk = fits.open(mkk_base_path+'/'+mask_tail+'/mkk_maskonly_estimate_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(setidx)+'_'+mask_tail+'.fits')['inv_Mkk_'+str(ifield)].data

            mock_trilegal_cat = mock_trilegal['tracer_cat_'+str(ifield)].data
            
            if sigma_xy is not None:
                mock_trilegal_cat['x'] += np.random.normal(0, sigma_xy, len(mock_trilegal_cat['x']))
                mock_trilegal_cat['y'] += np.random.normal(0, sigma_xy, len(mock_trilegal_cat['y']))
            
            full_tracer_cat_star = np.array([mock_trilegal_cat['x'], mock_trilegal_cat['y'], mock_trilegal_cat[bandstr], mock_trilegal_cat[Iarr_str]])

            mock_gal_cat = mock_gal['tracer_cat_'+str(ifield)].data
            I_arr_full = cmock.mag_2_nu_Inu(mock_gal_cat[cib_bandstr], inst-1)
            
            if sigma_xy is not None:
                mock_gal_cat['x'] += np.random.normal(0, sigma_xy, len(mock_gal_cat['x']))
                mock_gal_cat['y'] += np.random.normal(0, sigma_xy, len(mock_gal_cat['y']))
            
            full_tracer_cat_gal = np.array([mock_gal_cat['x'], mock_gal_cat['y'], mock_gal_cat[cib_bandstr], I_arr_full])

            
            bright_cut_star = np.where(full_tracer_cat_star[2,:] < mag_limit_match)[0]
            bright_cut_gal = np.where(full_tracer_cat_gal[2,:] < mag_limit_match)[0]
            
            cut_cat_star = full_tracer_cat_star[:, bright_cut_star].transpose()
            cut_cat_gal = full_tracer_cat_gal[:, bright_cut_gal].transpose()
            
            bright_star_map = cmock.make_srcmap_temp_bank(ifield, inst, cut_cat_star, flux_idx=-1, load_precomp_tempbank=True, \
                                                        tempbank_dirpath=tempbank_dirpath)
            
            bright_gal_map = cmock.make_srcmap_temp_bank(ifield, inst, cut_cat_gal, flux_idx=-1, load_precomp_tempbank=False, \
                                            tempbank_dirpath=tempbank_dirpath)
            
            bright_src_map = bright_star_map + bright_gal_map
            
            masked_bright_src_map = mask*bright_src_map
            
            masked_bright_src_map[masked_bright_src_map != 0] -= np.mean(masked_bright_src_map[masked_bright_src_map != 0])
            
            if setidx==0:
                plot_map(masked_bright_src_map, title='field '+str(ifield))
                
            lb, cl_outskirts_masked, clerr_outskirts_masked = get_power_spec(masked_bright_src_map, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)

            cl_proc = np.dot(inv_Mkk.transpose(), cl_outskirts_masked)
            
            cl_proc /= B_ells[fieldidx]**2
            
            src_map_cls[fieldidx, setidx] = cl_proc
            
            
    return lb, src_map_cls

def make_synthetic_trilegal_cat_2MSW(trilegal_path, J_band_idx=11, H_band_idx=12, IRAC_CH1_idx=14, IRAC_CH2_idx=15, \
                                     W1_idx=21, W2_idx=22, imdim=1024.):
    ''' 
    Generate synthetic catalog realization from TRILEGAL catalog. All this function does is draw uniformly random positions and 
    the TRILEGAL catalog magnitudes to make a new catalog, at the moment.
    '''

    trilegal = np.loadtxt(trilegal_path)
    nsrc = trilegal.shape[0]
    synthetic_cat = np.random.uniform(0, imdim, size=(nsrc, 2))
    synthetic_cat = np.array([synthetic_cat[:,0], synthetic_cat[:,1],\
                              trilegal[:,J_band_idx], trilegal[:,H_band_idx], \
                             trilegal[:,IRAC_CH1_idx], trilegal[:,IRAC_CH2_idx], \
                             trilegal[:,W1_idx], trilegal[:,W2_idx]]).transpose()

    print('synthetic cat has shape ', synthetic_cat.shape)
    return synthetic_cat

def make_synthetic_trilegal_cat(trilegal_path, J_band_idx=16, H_band_idx=17, imdim=1024.):
    ''' 
    Generate synthetic catalog realization from TRILEGAL catalog. All this function does is draw uniformly random positions and 
    the TRILEGAL catalog magnitudes to make a new catalog, at the moment.
    '''
    trilegal = np.loadtxt(trilegal_path)
    nsrc = trilegal.shape[0]
    synthetic_cat = np.random.uniform(0, imdim, size=(nsrc, 2))
    synthetic_cat = np.array([synthetic_cat[:,0], synthetic_cat[:,1], trilegal[:,J_band_idx], trilegal[:,H_band_idx]]).transpose()
    print('synthetic cat has shape ', synthetic_cat.shape)
    return synthetic_cat

def make_offset_grad_image(offset_range=10, zl_level=500, dimx=1024, dimy=1024, theta_mag=0.01):
    
    cbps = CIBER_PS_pipeline()
    
    image = np.zeros((dimx, dimy))
    
    offsets = []
    for q in range(4):
        
        offset = np.random.uniform(-offset_range, offset_range)
        
        offsets.append(offset)
        
        image[cbps.x0s[q]:cbps.x1s[q], cbps.y0s[q]:cbps.y1s[q]] += offset
        
    # add gradient
    zl_realiz = generate_zl_realization(zl_level, True, theta_mag=theta_mag, dimx=dimx, dimy=dimy)[0]

    image += zl_realiz
    
    plot_map(image)
    
    return image, offsets

def filter_trilegal_cat(trilegal_cat, m_min=4, m_max=17, filter_band_idx=16):
    
    filtered_trilegal_cat = np.array([x for x in trilegal_cat if x[filter_band_idx]<m_max and x[filter_band_idx]>m_min])

    return filtered_trilegal_cat

def rebin_map_coarse(original_map, Nsub):
    ''' Downsample map, taking average of downsampled pixels '''

    m, n = np.array(original_map.shape)//(Nsub, Nsub)
    
    return original_map.reshape(m, Nsub, n, Nsub).mean((1,3))


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


class ciber_mock():
    sb_intensity_unit = u.nW/u.m**2/u.steradian # helpful unit to have on hand

    # ciberdir = '/Users/richardfeder/Documents/caltech/ciber2'
    darktime_name_dict = dict({36265:['NEP', 'BootesA'],36277:['SWIRE', 'NEP', 'BootesB'], \
    40030:['DGL', 'NEP', 'Lockman', 'elat10', 'elat30', 'BootesB', 'BootesA', 'SWIRE']})
    ciber_field_dict = dict({4:'elat10', 5:'elat30',6:'BootesB', 7:'BootesA', 8:'SWIRE'})

    pix_width = 7.*u.arcsec
    pix_sr = ((pix_width.to(u.degree))*(np.pi/180.0))**2*(u.steradian / u.degree) # pixel solid angle in steradians
    lam_effs = np.array([1.05, 1.79])*1e-6*u.m # effective wavelength for bands

    # sky_brightness = np.array([300., 370.])*sb_intensity_unit
    # instrument_noise = np.array([33.1, 17.5])*sb_intensity_unit

    helgason_to_ciber_rough = dict({1:'J', 2:'H'})
    Vega_to_AB = dict({1:0.91 , 2: 1.39}) # add these numbers to go from Vega to AB magnitude

    # ciber_lam_dict
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
        ''' unit conversion from monochromatic AB magnitudes to units of Jansky. In the presence of a realistic bandpass this needs to change ''' 
        return 3631*u.Jansky*10**(-0.4*mags)

    def mag_2_nu_Inu(self, mags, band=None, lam_eff=None):
        ''' conversion from magnitudes to intensity at specific wavelength ''' 
        jansky_arr = self.mag_2_jansky(mags)
        if lam_eff is None:
            lam_eff = self.lam_effs[band]
        return jansky_arr.to(u.nW*u.s/u.m**2)*const.c/(self.pix_sr*lam_eff)

        # return jansky_arr.to(u.nW*u.s/u.m**2)*const.c/(self.pix_sr*self.lam_effs[band])

    def make_srcmap_temp_bank(self, ifield, inst, cat,\
                             flux_idx=-1, n_fine_bin=10, nwide=17, beta=None, rc=None, norm=None, \
                             tail_path='data/psf_model_dict_updated_081121_ciber.npz', verbose=False, \
                             tempbank_dirpath='data/subpixel_psf_temp_bank/', load_precomp_tempbank=False, compute_temp_bank=False):


        # todo load psf temps using inst and ifield

        if load_precomp_tempbank:
            print('Loading PSF template bank for inst '+str(inst)+', ifield '+str(ifield)+'..')
            tempbank_fpath = tempbank_dirpath+'psf_temp_bank_inst'+str(inst)+'_ifield'+str(ifield)+'.npz'
            self.psf_temp_bank = np.load(tempbank_fpath)['psf_temp_bank']
        elif compute_temp_bank:
            if beta is None:
                beta, rc, norm = load_psf_params_dict(inst, ifield=ifield, tail_path=tail_path, verbose=verbose)
            print('Generating PSF sub-pixel template bank for inst '+str(inst)+', ifield '+str(ifield)+'..')
            self.psf_temp_bank, dists = generate_psf_template_bank(beta, rc, norm, n_fine_bin=n_fine_bin, nwide=nwide)
        else:
            if self.psf_temp_bank is None:
                print('no psf template bank in ciber_mock object..')

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

            srcmap = image_model_eval(np.array(cat[:,0]).astype(float)+dx, np.array(cat[:,1]).astype(float)-dy ,np.array(cat[:, flux_idx]).astype(float),0., (self.nx, self.ny), 35, self.cf, lib=self.libmmult.pcat_model_eval)

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
                        temp_bank=True, n_fine_bin=10, convert_tracer_to_Vega=False, Vega_to_AB=0.91, load_precomp_tempbank=True, \
                        dx_pointing=None, dy_pointing=None):
    
        if inst is None:
            inst = band+1
        srcmaps_full, catalogs, ihl_maps = [[] for x in range(3)]
        
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

        if dx_pointing is not None:
            print('Adding offset dx_pointing = ', dx_pointing)
        if dy_pointing is not None:
            print('Adding offset dy_pointing = ', dy_pointing)

        for c in range(ncatalog):

            cat_full = self.catalog_mag_cut(catalog_list[c], catalog_list[c][:,3], m_min, m_max)

            if convert_tracer_to_Vega:
                tracer_cat = self.catalog_mag_cut(catalog_list[c], catalog_list[c][:,3], m_min, m_tracer_max+Vega_to_AB)
            else:
                tracer_cat = self.catalog_mag_cut(catalog_list[c], catalog_list[c][:,3], m_min, m_tracer_max)

            I_arr_full = self.mag_2_nu_Inu(cat_full[:,3], band)
            cat_full = np.hstack([cat_full, np.expand_dims(I_arr_full.value, axis=1)])

            if temp_bank:
                if not load_precomp_tempbank:
                    srcmap_full = self.make_srcmap_temp_bank(ifield_list[c], inst, cat_full, flux_idx=-1, n_fine_bin=n_fine_bin, nwide=17, load_precomp_tempbank=False, \
                                                tail_path=config.exthdpath+'ciber_fluctuation_data/psf_model_dict_updated_081121_ciber.npz', compute_temp_bank=True)
                
                else:
                    print('loading precomputed template bank in tempbank_dirpath')
                    srcmap_full = self.make_srcmap_temp_bank(ifield_list[c], inst, cat_full, flux_idx=-1, n_fine_bin=n_fine_bin, nwide=17, load_precomp_tempbank=True, \
                                                    tempbank_dirpath=config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/subpixel_psfs/')
                self.psf_temp_bank = None

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
                   

        return srcmaps_full, catalogs, ihl_maps


    def make_mock_ciber_map(self,  m_min, m_max, ifield=None, ifield_list=None, mock_cat=None, band=0, ihl_frac=0., ng_bins=8,\
                            zmin=0.0, zmax=2.0, pcat_model_eval=False, ncatalog=1, add_noise=False, cat_return='tracer', m_tracer_max=20., \
                            temp_bank=True, convert_tracer_to_Vega=False, load_cl_limber_file=False, plot=False, \
                            simulate_pointing_offset=False, dx_pointing=0.5, dy_pointing=0.5, wcs_hdrs_first=None, wcs_hdrs_second=None, \
                            use_ciber_lam_eff=False, mode=None):
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

        mode (string, default=None): can specify high faint end ('hfe') and low faint end ('lfe') galaxy LF model from Helgason

        Returns
        -------
        
        full_map/srcmap/noise/ihl_map (np.array): individual and combined components of mock CIBER map
        cat (np.array): galaxy catalog for mock CIBER map
        psf_template (np.array): psf template used to generate mock CIBER sources
            
        """

        inst = band+1
        band_helgason = self.helgason_to_ciber_rough[inst]

        if use_ciber_lam_eff:
            ciber_lam_obs = 1e6*self.lam_effs[band].value
            print('ciber lam obs:', ciber_lam_obs)
        else:
            ciber_lam_obs = None

        if mode is not None:
            print('Mode is ', mode)
        else:
            print('Default H12 model')

        print('helgason band is ', band_helgason)
        if mock_cat is not None:
            m_arr = []
            cat = mock_cat
        else:
            mock_galaxy = galaxy_catalog()
            cat = mock_galaxy.generate_galaxy_catalogs(band=band_helgason, ng_bins=ng_bins, zmin=zmin, zmax=zmax, n_catalogs=ncatalog, m_min=m_min, m_max=m_max, load_cl_limber_file=load_cl_limber_file, plot=plot, \
                                                        lam_obs=ciber_lam_obs, mode=mode)
        
        # tracer catalogs are selected using cuts on Vega mag, but actual magnitudes in tracer catalog are still in AB magnitude system (6/23/22)
        print('len cat is ', len(cat))
        print('while ncatalog is ', ncatalog)

        srcmaps_full, catalogs, _ = self.mocks_from_catalogs(cat, ncatalog, m_min=m_min, \
                                                                                   m_max=m_max,ifield=ifield, ifield_list=ifield_list, band=band,pcat_model_eval=pcat_model_eval, \
                                                                                   add_noise=add_noise, cat_return=cat_return, m_tracer_max=m_tracer_max,\
                                                                                   temp_bank=temp_bank, convert_tracer_to_Vega=convert_tracer_to_Vega)
        
        if simulate_pointing_offset:


            print('Generating set of maps with slight pointing offset')

            cat_full_dpoint = cat.copy()

            all_dpoint = []
            # dpoint_xy = np.zeros((len(cat_full_dpoint), catalogs[0].shape[0], 2))

            # print('cat full dpoint length:', len(cat_full_dpoint))
            # print('cat full dpoint has shape', cat_full_dpoint.shape)

            if wcs_hdrs_first is not None and wcs_hdrs_second is not None:
                print('Using WCS headers of first and second halves to simulate pointing offset')
                print('Since we are using WCS headers, setting dx_pointing, dy_pointing to zero..')
                # dx_pointing, dy_pointing = 0., 0.
                
                for catidx in range(len(cat_full_dpoint)):

                    dpoint_xy = np.zeros((catalogs[catidx].shape[0], 2))
                    ra, dec = wcs_hdrs_first[catidx].all_pix2world(cat_full_dpoint[catidx][:,0], cat_full_dpoint[catidx][:,1], 0)
                    newx, newy = wcs_hdrs_second[catidx].all_world2pix(ra, dec, 0)

                    cat_full_dpoint[catidx][:,0] = newx 
                    cat_full_dpoint[catidx][:,1] = newy

                    # print(len(catalogs[catidx][:,0]), len(catalogs[catidx][:,1]))
                    ra_tracer, dec_tracer = wcs_hdrs_first[catidx].all_pix2world(catalogs[catidx][:,0], catalogs[catidx][:,1], 0)
                    # print(ra_tracer, dec_tracer)

                    newx_tracer, newy_tracer = wcs_hdrs_second[catidx].all_world2pix(ra_tracer, dec_tracer, 0)
                    # print(newx_tracer, newy_tracer)

                    dpoint_xy[:,0] = newx_tracer 
                    dpoint_xy[:,1] = newy_tracer

                    all_dpoint.append(dpoint_xy)
            else:
                print('adding dx_pointing, dy_pointing = ', dx_pointing, dy_pointing)
                cat_full_dpoint[:,0] += dx_pointing
                cat_full_dpoint[:,1] += dy_pointing

            # cat = np.column_stack([cat, cat_full_dpoint[:,0]])
            # cat = np.column_stack([cat, cat_full_dpoint[:,1]])

            if wcs_hdrs_first is not None and wcs_hdrs_second is not None:
                print('Using WCS headers of first and second halves to simulate pointing offset')
                print('Since we are using WCS headers, setting dx_pointing, dy_pointing to zero..')
                dx_pointing, dy_pointing = 0., 0.


            srcmaps_dpoint, _, _ = self.mocks_from_catalogs(cat, ncatalog, m_min=m_min,\
                                                                       m_max=m_max,ifield=ifield, ifield_list=ifield_list, band=band,pcat_model_eval=pcat_model_eval, \
                                                                       add_noise=add_noise, cat_return=cat_return, m_tracer_max=m_tracer_max, \
                                                                       temp_bank=temp_bank, convert_tracer_to_Vega=convert_tracer_to_Vega)
            
            # srcmaps_dpoint, _, _ = self.mocks_from_catalogs(cat, ncatalog, m_min=m_min,\
            #                                                          m_max=m_max,ifield=ifield, ifield_list=ifield_list, band=band,pcat_model_eval=pcat_model_eval, \
            #                                                          add_noise=add_noise, cat_return=cat_return, m_tracer_max=m_tracer_max, \
            #                                                          temp_bank=temp_bank, convert_tracer_to_Vega=convert_tracer_to_Vega, \
            #                                                          dx_pointing=dx_pointing, dy_pointing=dy_pointing, wcs_hdrs_first=wcs_hdrs_first, wcs_hdrs_second=wcs_hdrs_second)

            return srcmaps_full, srcmaps_dpoint, catalogs, all_dpoint

        return srcmaps_full, catalogs

    def generate_trilegal_realizations(self, ifield_list, datestr, nsims, convert_Vega_to_AB=True, \
                                       simidx0=0, m_min=None, m_max=28, m_tracer_max=19.5, save=True, trilegal_base_path=None, \
                                     ciber_mock_dirpath = config.exthdpath+'ciber_mocks/', simulate_pointing_offset=False, \
                                     dx_pointing=0.5, dy_pointing=0.5, wcs_hdrs_tm1_first=None, wcs_hdrs_tm1_second=None, \
                                     wcs_hdrs_tm2_first=None, wcs_hdrs_tm2_second=None, trilegal_path=None, save_fpath=None, tailstr=None, \
                                     trilo_tailstr='trilegal'):
        
        if m_min is not None:
            m_min_perfield = dict({4:m_min, 5: m_min, 6: m_min, 7:m_min, 8:m_min})
        else:
            m_min_perfield = dict({4: 5.69, 5: 4.52, 6: 7.82, 7:7.20, 8:6.63}) # J band

        data_path= ciber_mock_dirpath+datestr

        trilo_path = data_path+'/'+trilo_tailstr+'/'

        if not os.path.isdir(trilo_path):
            os.makedirs(trilo_path)
        
        number_ct_lims = np.array([17.5, 18.0, 18.5, 19.0])
        
        number_counts_array = np.zeros((nsims, len(ifield_list), len(number_ct_lims)))
        
        all_save_fpaths = []

        make_fpaths([data_path, trilo_path])
        
        for setidx in np.arange(simidx0, nsims):
            print('on setidx '+str(setidx))
            full_maps_J, full_maps_H, cats = [], [], [] 
            
            save_fpaths = []

            if simulate_pointing_offset:
                full_maps_J_dpoint, full_maps_H_dpoint = [], []

            for fieldidx, ifield in enumerate(ifield_list):
                field_name = self.ciber_field_dict[ifield]
                print('field name is ', field_name)

                if trilegal_path is None:

                    if trilegal_base_path is None:
                        trilegal_base_path = config.exthdpath+'ciber_fluctuation_data/mock_trilegal_realizations_111722/'
                    trilegal_path = trilegal_base_path+'trilegal_'+field_name+'.dat'
                    # trilegal_path = 'data/mock_trilegal_realizations_051321/trilegal_'+field_name+'.dat'

                x = make_synthetic_trilegal_cat(trilegal_path, J_band_idx=26, H_band_idx=27)
                filt_x = filter_trilegal_cat(x, filter_band_idx=2, m_min=m_min_perfield[ifield], m_max=m_max)

                star_Iarr_full_J = self.mag_2_nu_Inu(filt_x[:,2], band=0)
                star_Iarr_full_H = self.mag_2_nu_Inu(filt_x[:,3], band=1)
                
                cat_full = np.hstack([filt_x, np.expand_dims(star_Iarr_full_J.value, axis=1), np.expand_dims(star_Iarr_full_H.value, axis=1)])
                
                # ------------ TM2 -----------------

                if simulate_pointing_offset:
                    print('generating separate image with pointing offset')

                    cat_full_dpoint = cat_full.copy()

                    if wcs_hdrs_tm1_first is not None and wcs_hdrs_tm1_second is not None:
                        print('Using WCS headers of first and second halves to simulate pointing offset')
                        print('Since we are using WCS headers, setting dx_pointing, dy_pointing to zero..')
                        # dx_pointing, dy_pointing = 0., 0.
                        ra, dec = wcs_hdrs_tm1_first[fieldidx].all_pix2world(cat_full_dpoint[:,0], cat_full_dpoint[:,1], 0)
                        newx, newy = wcs_hdrs_tm1_second[fieldidx].all_world2pix(ra, dec, 0)

                        cat_full_dpoint[:,0] = newx 
                        cat_full_dpoint[:,1] = newy

                    else:
                        cat_full_dpoint[:,0] += dx_pointing
                        cat_full_dpoint[:,1] += dy_pointing

                srcmap_stars_J = self.make_srcmap_temp_bank(ifield, 1, cat_full, flux_idx=4, n_fine_bin=10, nwide=17, load_precomp_tempbank=True, \
                                                            tempbank_dirpath=config.ciber_basepath+'data/fluctuation_data/TM1/subpixel_psfs/')
                
                if simulate_pointing_offset:

                    srcmap_stars_J_dpoint = self.make_srcmap_temp_bank(ifield, 1, cat_full_dpoint, flux_idx=4, n_fine_bin=10, nwide=17, load_precomp_tempbank=True, \
                                                tempbank_dirpath=config.ciber_basepath+'data/fluctuation_data/TM1/subpixel_psfs/')
                    full_maps_J_dpoint.append(srcmap_stars_J_dpoint)

                self.psf_temp_bank = None
                
                # ------------ TM2 -----------------
                srcmap_stars_H = self.make_srcmap_temp_bank(ifield, 2, cat_full, flux_idx=5, n_fine_bin=10, nwide=17, load_precomp_tempbank=True, \
                                                            tempbank_dirpath=config.ciber_basepath+'data/fluctuation_data/TM2/subpixel_psfs/')

                if simulate_pointing_offset:

                    cat_full_dpoint = cat_full.copy()

                    if wcs_hdrs_tm2_first is not None and wcs_hdrs_tm2_second is not None:
                        print('Using WCS headers of first and second halves to simulate pointing offset')
                        print('Since we are using WCS headers, setting dx_pointing, dy_pointing to zero..')
                        # dx_pointing, dy_pointing = 0., 0.
                        ra, dec = wcs_hdrs_tm2_first[fieldidx].all_pix2world(cat_full_dpoint[:,0], cat_full_dpoint[:,1], 0)
                        newx, newy = wcs_hdrs_tm2_second[fieldidx].all_world2pix(ra, dec, 0)

                        cat_full_dpoint[:,0] = newx 
                        cat_full_dpoint[:,1] = newy

                    srcmap_stars_H_dpoint = self.make_srcmap_temp_bank(ifield, 2, cat_full_dpoint, flux_idx=4, n_fine_bin=10, nwide=17, load_precomp_tempbank=True, \
                                                tempbank_dirpath=config.ciber_basepath+'data/fluctuation_data/TM2/subpixel_psfs/')
                    full_maps_H_dpoint.append(srcmap_stars_H_dpoint)

                full_maps_J.append(srcmap_stars_J)
                full_maps_H.append(srcmap_stars_H)
                
                # filter full tracer catalog with respect to J band, but make it deep enough that we can make cuts later on J or H
        
                if convert_Vega_to_AB:
                    if setidx==0:
                        print("Converting tracer magnitude limit in J band from Vega to AB magnitudes..")
                    m_tracer_max_cut = m_tracer_max + 0.91
                else:
                    if setidx==0:
                        print('Keeping tracer magnitude limit in Vega magnitudes, beware!')
                    m_tracer_max_cut = m_tracer_max

                cat_tracer = filter_trilegal_cat(cat_full, filter_band_idx=2, m_max=m_tracer_max_cut)

                for j, Jthresh in enumerate(number_ct_lims):
                    if convert_Vega_to_AB:
                        Jthresh += 0.91
                    cat_thresh = filter_trilegal_cat(cat_full, filter_band_idx=2, m_max=Jthresh)
                    
                    print('there are ', cat_thresh.shape, 'sources less than ', Jthresh, ' for ifield ', ifield)
                    number_counts_array[setidx, fieldidx, j] = cat_thresh.shape[0]
                
                cats.append(cat_tracer)

                if setidx == 0:
                    plot_map(srcmap_stars_J, title='source map J', x0=200, x1=300, y0=200, y1=300, lopct=5, hipct=95, cmap='jet')
                    plot_map(srcmap_stars_H, title='source map H', x0=200, x1=300, y0=200, y1=300, lopct=5, hipct=95, cmap='jet')
                    if simulate_pointing_offset:
                        plot_map(srcmap_stars_J_dpoint, title='source map J with pointing offset', x0=200, x1=300, y0=200, y1=300, lopct=5, hipct=95, cmap='jet')
                        plot_map(srcmap_stars_H_dpoint, title='source map H with pointing offset', x0=200, x1=300, y0=200, y1=300, lopct=5, hipct=95, cmap='jet')
                        plot_map(srcmap_stars_J-srcmap_stars_J_dpoint, title='J difference with dpoint', x0=200, x1=300, y0=200, y1=300, lopct=5, hipct=95, cmap='jet')
                        plot_map(srcmap_stars_H-srcmap_stars_H_dpoint, title='H difference with dpoint', x0=200, x1=300, y0=200, y1=300, lopct=5, hipct=95, cmap='jet')

            names = ['x', 'y', 'j_m', 'h_m', 'j_nuInu', 'h_nuInu']
            tail_name = 'mock_trilegal_simidx'+str(setidx)+'_'+datestr

            if tailstr is not None:
                tail_name+= '_'+tailstr

            if save:
                print('Saving mock set..')
                save_mock_fpath = save_mock_to_fits(full_maps_J, cats, tail_name, names=names, full_maps_band2=full_maps_H, data_path=trilo_path, m_tracer_max=m_tracer_max, m_min=m_min_perfield[ifield], m_max=m_max,\
                                  ifield_list=ifield_list, map_names=['trilegal_J', 'trilegal_H'], return_save_fpath=True, save_fpath=save_fpath)

                save_fpaths.append(save_mock_fpath)
                if simulate_pointing_offset:
                    tail_name_dpoint = 'mock_trilegal_with_dpoint_simidx'+str(setidx)+'_'+datestr

                    save_mock_to_fits(full_maps_J_dpoint, cats, tail_name_dpoint, names=names, full_maps_band2=full_maps_H_dpoint, data_path=trilo_path, m_tracer_max=m_tracer_max, m_min=m_min_perfield[ifield], m_max=m_max,\
                                      ifield_list=ifield_list, map_names=['trilegal_J_dpoint', 'trilegal_H_dpoint'])
                
            all_save_fpaths.append(save_fpaths)


        print('Mean number of stars per field:', np.mean(number_counts_array, axis=0))
        
        return number_counts_array, all_save_fpaths

# cl_predictions.py
# def cl_predictions_vs_magcut(inst, ifield_list=[4, 5, 6, 7, 8], mag_lims=None, mag_cut_cat = 15.0, nsim=10, ifield_choose=4):


''' below here is largely deprecated code '''

def compute_star_gal_mask(stellar_cat, galaxy_cat, star_mag_idx=2, gal_mag_idx=3, m_max=18.4, return_indiv_masks=False):
    
    '''star and galaxy catalogs reported with AB magnitudes. 
    Default indices assume stellar catalog already passed through make_synthetic_trilegal_cat()'''
    
    filt_star_cat = filter_trilegal_cat(stellar_cat, J_band_idx=star_mag_idx, m_max=m_max)
    
    cmock = ciber_mock()
    filt_gal_cat = cmock.catalog_mag_cut(galaxy_cat, galaxy_cat[:, gal_mag_idx], m_min=0., m_max=m_max)
    
    mask_filt_star = mask_from_cat(filt_x, mag_idx=star_mag_idx)
    mask_gal = mask_from_cat(bright_cat, mag_idx=gal_mag_idx)
    
    joined_mask = mask_filt_star*mask_gal
    
    if return_indiv_masks:
        return mask_filt_star, mask_gal
    
    return joined_mask


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
    rvir_range = np.arange(rvir_min, rvir_max).astype(float)
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

    array_2d_float = npct.ndpointer(dtype=float, ndim=2, flags="C_CONTIGUOUS")
    array_1d_int = npct.ndpointer(dtype=int, ndim=1, flags="C_CONTIGUOUS")
    array_2d_double = npct.ndpointer(dtype=float, ndim=2, flags="C_CONTIGUOUS")
    array_2d_int = npct.ndpointer(dtype=int, ndim=2, flags="C_CONTIGUOUS")

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
