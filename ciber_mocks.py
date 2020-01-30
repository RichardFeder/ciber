import numpy as np
# import pandas as pd
import astropy.units as u
from astropy import constants as const
# import time
import scipy.signal
from mock_galaxy_catalogs import *
from helgason import *
from ciber_data_helpers import *

''' Given an input map and a specified center, this will
% create a map with each pixels value its distance from
% the specified pixel. '''



def normalized_ihl_template(dimx=50, dimy=50, R_vir=None):
    if R_vir is None:
        R_vir = np.sqrt((dimx/2)**2+(dimy/2)**2)
    xx, yy = np.meshgrid(np.arange(dimx), np.arange(dimy), sparse=True)
    ihl_map = np.sqrt(R_vir**2 - (xx-(dimx/2))**2 - (yy-(dimy/2))**2) # assumes a spherical projected profile for IHL
    ihl_map[np.isnan(ihl_map)]=0
    ihl_map /= np.sum(ihl_map)
    return ihl_map

''' Downsample map, taking average of downsampled pixels '''
def rebin_map_coarse(original_map, Nsub):
    m, n = np.array(original_map.shape)//(Nsub, Nsub)
    return original_map.reshape(m, Nsub, n, Nsub).mean((1,3))

def save_ihl_conv_templates(psf, rvir_min=1, rvir_max=50, dimx=150, dimy=150):
    ihl_conv_temps = []
    rvir_range = np.arange(rvir_min, rvir_max).astype(np.float)
    for rvir in rvir_range:
        ihl = normalized_ihl_template(R_vir=rvir, dimx=dimx, dimy=dimy)
        conv = scipy.signal.convolve2d(ihl, psf, 'same')
        ihl_conv_temps.append(conv)
    return ihl_conv_temps

def virial_radius_2_reff(r_vir, zs, theta_fov_deg=2.0, npix_sidelength=1024.):
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

    def __init__(self):
        pass

    def catalog_mag_cut(self, cat, m_arr, m_min, m_max):
        magnitude_mask_idxs = np.array([i for i in xrange(len(m_arr)) if m_arr[i]>=m_min and m_arr[i] <= m_max])
        if len(magnitude_mask_idxs) > 0:
            catalog = cat[magnitude_mask_idxs,:]
        else:
            catalog = cat
        return catalog   

    def get_catalog(self, catname):
        cat = np.loadtxt(self.ciberdir+'/data/'+catname)
        x_arr = cat[0,:]
        y_arr = cat[1,:] 
        m_arr = cat[2,:]
        return x_arr, y_arr, m_arr

    def get_darktime_name(self, flight, field):
        return self.darktime_name_dict[flight][field-1]

    def mag_2_jansky(self, mags):
        return 3631*u.Jansky*10**(-0.4*mags)

    def mag_2_nu_Inu(self, mags, band):
        jansky_arr = self.mag_2_jansky(mags)
        return jansky_arr.to(u.nW*u.s/u.m**2)*const.c/(self.pix_sr*self.lam_effs[band])

    def make_srcmap(self, ifield, cat, flux_idx=-1, band=0, nbin=0., nx=1024, ny=1024, nwide=20, multfac=7.0):
        
        srcmap = np.zeros((nx*2, ny*2))

        # get psf params
        beta, rc, norm = find_psf_params(self.ciberdir+'/data/psfparams.txt', tm=band+1, field=self.ciber_field_dict[ifield])
        
        Nlarge = nx+30+30 
        Nsrc = cat.shape[0]

        radmap = make_radius_map(2*Nlarge+nbin, 2*Nlarge+nbin, Nlarge+nbin, Nlarge+nbin, rc)*multfac # is the input supposed to be 2d?
        
        print('Making source map TM, mrange=(%d,%d), %d sources'%(np.min(cat[:,3]),np.max(cat[:,3]),Nsrc))
        
        Imap_large = norm * np.power(1 + radmap, -3*beta/2.)
        Imap_large /= np.sum(Imap_large)     
        Imap_center = Imap_large[Nlarge-nwide:Nlarge+nwide+1, Nlarge-nwide:Nlarge+nwide+1]

        xs = np.round(cat[:,0]).astype(np.int32)
        ys = np.round(cat[:,1]).astype(np.int32)

        for i in xrange(Nsrc):
            srcmap[Nlarge/2+2+xs[i]-nwide:Nlarge/2+2+xs[i]+nwide+1, Nlarge/2-1+ys[i]-nwide:Nlarge/2-1+ys[i]+nwide+1] += Imap_center*cat[i, flux_idx]
        
        return srcmap[nx/2+30:3*nx/2+30, ny/2+30:3*ny/2+30], Imap_center, Imap_large

   
    def make_ihl_map(self, map_shape, cat, ihl_frac, flux_idx=-1, dimx=150, dimy=150, psf=None, extra_trim=20):
        

        rvirs = virial_radius_2_reff(r_vir=cat[:,6], zs=cat[:,2])
        rvirs = rvirs.value

        if psf is not None:
            ihl_temps = save_ihl_conv_templates(psf=psf)

        ihl_map = np.zeros((map_shape[0]+dimx+extra_trim, map_shape[1]+dimy+extra_trim))

        for i, src in enumerate(cat):
            x0 = np.floor(src[0]+extra_trim/2)
            y0 = np.floor(src[1]+extra_trim/2)
            ihl_map[int(x0):int(x0+ihl_temps[0].shape[0]), int(y0):int(y0 + ihl_temps[0].shape[1])] += ihl_temps[int(np.ceil(rvirs[i])-1)]*ihl_frac*src[flux_idx]
            # ihl_map[int(x0):int(x0+ norm_ihl.shape[0]), int(y0):int(y0 + norm_ihl.shape[1])] += norm_ihl*ihl_frac*src[2]

        return ihl_map[(ihl_temps[0].shape[0] + extra_trim)/2:-(ihl_temps[0].shape[0] + extra_trim)/2, (ihl_temps[0].shape[0] + extra_trim)/2:-(ihl_temps[0].shape[0] + extra_trim)/2]
        # return ihl_map[(norm_ihl.shape[0] + extra_trim)/2:-(norm_ihl.shape[0] + extra_trim)/2, (norm_ihl.shape[0] + extra_trim)/2:-(norm_ihl.shape[0] + extra_trim)/2]
     

    def mocks_from_catalogs(self, catalog_list, ncatalog, mock_data_directory, m_min=9., m_max=30., m_tracer_max=25., \
                        ihl_frac=0.2, ifield=4, band=0, save=False):
    
        srcmaps_full, catalogs, noise_realizations, ihl_maps = [[] for x in range(4)]
        
        print('m_min = ', m_min)
        print('m_max = ', m_max)
        print('m_tracer_max = ', m_tracer_max)
        for c in range(ncatalog):

            cat_full = self.catalog_mag_cut(catalog_list[c], catalog_list[c][:,3], m_min, m_max)
            tracer_cat = self.catalog_mag_cut(catalog_list[c], catalog_list[c][:,3], m_min, m_tracer_max)
            I_arr_full = self.mag_2_nu_Inu(cat_full[:,3], band)
            cat_full = np.hstack([cat_full, np.expand_dims(I_arr_full.value, axis=1)])
            srcmap_full, psf_template, psf_full = self.make_srcmap(ifield, cat_full, band=band)
            noise = np.random.normal(self.sky_brightness[band].value, self.instrument_noise[band].value, size=srcmap_full.shape)
            conv_noise = scipy.signal.convolve2d(noise, psf_template, 'same')
            
            noise_realizations.append(conv_noise)
            srcmaps_full.append(srcmap_full)
            
            if ihl_frac > 0:
                print('Making IHL map..')
                ihl_map = self.make_ihl_map(srcmap_full.shape, cat_full, ihl_frac, psf=psf_template)
                ihl_maps.append(ihl_map)
                if save:
                    print('Saving results..')
                    np.savez_compressed(mock_data_directory+'ciber_mock_'+str(c)+'_mmin='+str(m_min)+'.npz', \
                                        catalog=tracer_cat, srcmap_full=srcmap_full, conv_noise=conv_noise,\
                                        ihl_map=ihl_map)
            else:
                if save:
                    print('Saving results..')
                    np.savez_compressed(mock_data_directory+'ciber_mock_'+str(c)+'_mmin='+str(m_min)+'.npz', \
                                        catalog=tracer_cat, srcmap_full=srcmap_full, conv_noise=conv_noise)
                   
        if ihl_frac > 0: 
            return srcmaps_full, catalogs, noise_realizations, ihl_maps
        else:
            return srcmaps_full, catalogs, noise_realizations

    def make_mock_ciber_map(self, ifield, m_min, m_max, mock_cat=None, band=0, catname=None, ihl_frac=0., ng_bins=5, zmin=0.01, zmax=5.0):
        if mock_cat is not None:
            m_arr = []
            cat_arr = mock_cat
        else:
            mock_galaxy = galaxy_catalog()
            cat = mock_galaxy.generate_galaxy_catalog(ng_bins=ng_bins, zmin=zmin, zmax=zmax)
        
        cat = self.catalog_mag_cut(cat, cat[:,3], m_min, m_max) 
        I_arr = self.mag_2_nu_Inu(cat[:,3], band)
        cat = np.hstack([cat, np.expand_dims(I_arr.value, axis=1)])

        srcmap, psf_template, psf_full = self.make_srcmap(ifield, cat, band=band)
        full_map = np.zeros_like(srcmap)
        noise = np.random.normal(self.sky_brightness[band].value, self.instrument_noise[band].value, size=srcmap.shape)
        
        full_map = srcmap + noise
        if ihl_frac > 0:
            ihl_map = self.make_ihl_map(srcmap.shape, cat, ihl_frac, psf=psf_template)
            full_map += ihl_map
            return full_map, srcmap, noise, ihl_map, cat, psf_template
        else:
            return full_map, srcmap, noise, cat, psf_template

