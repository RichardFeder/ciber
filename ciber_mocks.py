import numpy as np
import numpy.ctypeslib as npct
import ctypes
from ctypes import c_int, c_double

import astropy.units as u
from astropy import constants as const
import scipy.signal
from mock_galaxy_catalogs import *
from helgason import *
from ciber_data_helpers import *
from image_eval import psf_poly_fit, image_model_eval
import sys


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

def save_mock_items_to_npz(filepath, catalog=None, srcmap_full=None, srcmap_nb=None, \
                           conv_noise=None, m_min=None, m_min_nb=None, ihl_map=None, m_lim=None):
    ''' Convenience file for saving mock observation files. '''
    np.savez_compressed(filepath, catalog=catalog, srcmap_full=srcmap_full, \
                        srcmap_nb=srcmap_nb, conv_noise=conv_noise, \
                        ihl_map=ihl_map, m_lim=m_lim, m_min=m_min, m_min_nb=m_min_nb)



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

    ciberdir = '/Users/richardfeder/Documents/caltech/ciber2'
    darktime_name_dict = dict({36265:['NEP', 'BootesA'],36277:['SWIRE', 'NEP', 'BootesB'], \
    40030:['DGL', 'NEP', 'Lockman', 'elat10', 'elat30', 'BootesB', 'BootesA', 'SWIRE']})
    ciber_field_dict = dict({4:'elat10', 5:'elat30',6:'BootesB', 7:'BootesA', 8:'SWIRE'})

    pix_width = 7.*u.arcsec
    pix_sr = ((pix_width.to(u.degree))*(np.pi/180.0))**2*u.steradian / u.degree # pixel solid angle in steradians
    lam_effs = np.array([1.05, 1.79])*1e-6*u.m # effective wavelength for bands
    sky_brightness = np.array([300., 370.])*sb_intensity_unit
    instrument_noise = np.array([33.1, 17.5])*sb_intensity_unit

    def __init__(self, pcat_model_eval=True):
        self.psf_template=None
        
        if pcat_model_eval:

            if sys.version_info[0] < 3:
                self.libmmult = npct.load_library('pcat-lion', '.')
            else:
                self.libmmult = npct.load_library('pcat-lion.so', '/Users/richardfeder/Documents/caltech/ciber2/ciber/')

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
        
    def get_psf(self, ifield=4, band=0, nx=1024, ny=1024, multfac=7.0, nbin=0., poly_fit=True, nwide=12):
        
        beta, rc, norm = find_psf_params(self.ciberdir+'/data/psfparams.txt', tm=band+1, field=self.ciber_field_dict[ifield])

        Nlarge = nx+30+30 

        radmap = make_radius_map(2*Nlarge+nbin, 2*Nlarge+nbin, Nlarge+nbin, Nlarge+nbin, rc)*multfac # is the input supposed to be 2d?
        
        
        self.psf_full = norm * np.power(1 + radmap, -3*beta/2.)
        self.psf_full /= np.sum(self.psf_full)     
        self.psf_template = self.psf_full[Nlarge-nwide:Nlarge+nwide+1, Nlarge-nwide:Nlarge+nwide+1]
        
        print('imap center has shape', self.psf_template.shape)

        
        if poly_fit:
            psf = np.zeros((50,50))
            psf[0:25,0:25] = self.psf_template
            psf = scipy.misc.imresize(psf, (250, 250), interp='lanczos', mode='F')
            psfnew = np.array(psf[0:125, 0:125])
            psfnew[0:123,0:123] = psf[2:125,2:125]  # shift due to lanczos kernel
            self.cf = psf_poly_fit(psfnew, nbin=5)
        

    def get_darktime_name(self, flight, field):
        return self.darktime_name_dict[flight][field-1]

    def mag_2_jansky(self, mags):
        ''' unit conversion from magnitudes to Jansky ''' 
        return 3631*u.Jansky*10**(-0.4*mags)

    def mag_2_nu_Inu(self, mags, band):
        ''' unit conversion from magnitudes to intensity at specific wavelength ''' 
        jansky_arr = self.mag_2_jansky(mags)
        return jansky_arr.to(u.nW*u.s/u.m**2)*const.c/(self.pix_sr*self.lam_effs[band])

    def make_srcmap(self, ifield, cat, flux_idx=-1, band=0, nbin=0., nx=1024, ny=1024, nwide=12, multfac=7.0, \
                    pcat_model_eval=False, libmmult=None, dx=2.5, dy=-1.0):
        
        ''' This function takes in a catalog, finds the PSF for the specific ifield, makes a PSF template and then populates an image with 
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

        srcmap = np.zeros((nx*2, ny*2))
        
        Nlarge = nx+30+30 


        if self.psf_template is None:
            
            self.get_psf(ifield=ifield, band=band, nx=nx, ny=ny, multfac=multfac, poly_fit=pcat_model_eval, nwide=nwide)
            
    
        print('Making source map TM, mrange=(%d,%d), %d sources'%(np.min(cat[:,3]),np.max(cat[:,3]),Nsrc))
        

        if pcat_model_eval:
            
            srcmap = image_model_eval(np.array(cat[:,0]).astype(np.float32)+2.5, np.array(cat[:,1]).astype(np.float32)-1.0 ,np.array(cat[:, flux_idx]).astype(np.float32),0., (nx, ny), 25, self.cf, lib=self.libmmult.pcat_model_eval)
            
            return srcmap
        else:
            xs = np.round(cat[:,0]).astype(np.int32)
            ys = np.round(cat[:,1]).astype(np.int32)

            for i in range(Nsrc):
                srcmap[Nlarge/2+2+xs[i]-nwide:Nlarge/2+2+xs[i]+nwide+1, Nlarge/2-1+ys[i]-nwide:Nlarge/2-1+ys[i]+nwide+1] += self.psf_template*cat[i, flux_idx]
        
            return srcmap[nx/2+30:3*nx/2+30, ny/2+30:3*ny/2+30]

   
    def make_ihl_map(self, map_shape, cat, ihl_frac, flux_idx=-1, dimx=150, dimy=150, psf=None, extra_trim=20):
        
        ''' Given a catalog amnd a fractional ihl contribution, this function precomputes an array of IHL templates and then 
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
        # return ihl_map[(norm_ihl.shape[0] + extra_trim)/2:-(norm_ihl.shape[0] + extra_trim)/2, (norm_ihl.shape[0] + extra_trim)/2:-(norm_ihl.shape[0] + extra_trim)/2]
     

    def mocks_from_catalogs(self, catalog_list, ncatalog, mock_data_directory=None, m_min=9., m_max=30., m_tracer_max=25., \
                        ihl_frac=0.0, ifield=4, band=0, save=False, extra_name='', pcat_model_eval=False):
    
        srcmaps_full, catalogs, noise_realizations, ihl_maps = [[] for x in range(4)]
        
        print('m_min = ', m_min)
        print('m_max = ', m_max)
        print('m_tracer_max = ', m_tracer_max)
        if extra_name != '':
            extra_name = '_'+extra_name

        if save and mock_data_directory is None:
            print('Please provide a mock data directory to save to..')
            return

        for c in range(ncatalog):

            cat_full = self.catalog_mag_cut(catalog_list[c], catalog_list[c][:,3], m_min, m_max)
            tracer_cat = self.catalog_mag_cut(catalog_list[c], catalog_list[c][:,3], m_min, m_tracer_max)
            I_arr_full = self.mag_2_nu_Inu(cat_full[:,3], band)
            cat_full = np.hstack([cat_full, np.expand_dims(I_arr_full.value, axis=1)])
            srcmap_full = self.make_srcmap(ifield, cat_full, band=band, pcat_model_eval=pcat_model_eval)
            noise = np.random.normal(self.sky_brightness[band].value, self.instrument_noise[band].value, size=srcmap_full.shape)
            conv_noise = scipy.signal.convolve2d(noise, self.psf_template, 'same')
            
            catalogs.append(cat_full)
            noise_realizations.append(conv_noise)
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
                                        catalog=tracer_cat, srcmap_full=srcmap_full, conv_noise=conv_noise)
                   
        if ihl_frac > 0: 
            return srcmaps_full, catalogs, noise_realizations, ihl_maps
        else:
            return srcmaps_full, catalogs, noise_realizations

    def make_mock_ciber_map(self, ifield, m_min, m_max, mock_cat=None, band=0, ihl_frac=0., ng_bins=8,\
                            zmin=0.0, zmax=2.0, pcat_model_eval=True, ncatalog=1):
        ''' This is the parent function that uses other functions in the class to generate a full mock catalog/CIBER image. If there is 
        no mock catalog input, the function draws a galaxy catalog from the Helgason model with the galaxy_catalog() class. With a catalog 
        in hand, the function then imposes any cuts on magnitude, computes mock source intensities and then generates the corresponding 
        source maps/ihl maps/noise realizations that go into the final CIBER mock.

        Inputs:
            ifield (int): field from which to get PSF parameters
            m_min/m_max (float): minimum and maximum source fluxes to use from mock catalog in image generation
            mock_cat (np.array, default=None): this can be used to specify a catalog to generate beforehand rather than sampling a random one
            band (int, default=0): CIBER band of mock image, either 0 (band J) or 1 (band H)
            ihl_frac (float, default=0.0): determines amplitude of IHL around each source as fraction of source flux If ihl_frac=0.2, it means
                you place a source in the map with flux f and then add a template with amplitude 0.2*f. It's an additive feature, not dividing 
                source flux into 80/20 or anything like that.

            ng_bins (int, default=5): number of redshift bins to use when making mock map/catalog. Each redshift bin has its own generated 
                clustering field drawn with the lognormal technique. 
            zmin/zmax (float, default=0.01/5.0): form redshift range from which to draw galaxies from Helgason model.

        Outputs:

            full_map/srcmap/noise/ihl_map (np.array): individual and combined components of mock CIBER map
            cat (np.array): galaxy catalog for mock CIBER map
            psf_template (np.array): psf template used to generate mock CIBER sources
            
        '''
        if mock_cat is not None:
            m_arr = []
            cat = mock_cat
        else:
            mock_galaxy = galaxy_catalog()
            cat = mock_galaxy.generate_galaxy_catalogs(ng_bins=ng_bins, zmin=zmin, zmax=zmax, n_catalogs=ncatalog, m_min=m_min, m_max=m_max)
        
        if ihl_frac > 0:
            srcmaps, cats, noise_realizations, ihl_maps = self.mocks_from_catalogs(cat, ncatalog, m_min=m_min,\
                                                                                   m_max=m_max,ifield=ifield,band=band,pcat_model_eval=pcat_model_eval, ihl_frac=ihl_frac)
        else:
            srcmaps, cats, noise_realizations = self.mocks_from_catalogs(cat, ncatalog, m_min=m_min,\
                                                                                   m_max=m_max,ifield=ifield,band=band,pcat_model_eval=pcat_model_eval)

        full_maps = []
    
        for c in range(ncatalog):
            if ihl_frac > 0:
                full_maps.append(srcmaps[c]+noise_realizations[c]+ihl_maps[c])
            else:
                full_maps.append(srcmaps[c]+noise_realizations[c])
            
        if ihl_frac > 0:
        
            return full_maps, srcmaps, noise_realizations, ihl_maps, cats
        else:
            return full_maps, srcmaps, noise_realizations, cats