import numpy as np
import pandas as pd
import astropy.units as u
from astropy import constants as const
import time
from helgason import *

''' Downsample map, taking average of downsampled pixels '''
def rebin_map_coarse(original_map, Nsub):
    m, n = np.array(original_map.shape)//(Nsub, Nsub)
    return original_map.reshape(m, Nsub, n, Nsub).mean((1,3))


''' Given an input map and a specified center, this will
% create a map with each pixels value its distance from
% the specified pixel. '''

def make_radius_map(dimx, dimy, cenx, ceny, rc):
    x = np.arange(dimx)
    y = np.arange(dimy)
    xx, yy = np.meshgrid(x, y, sparse=True)
    return (((cenx - xx)/rc)**2 + ((ceny - yy)/rc)**2)


def normalized_ihl_template(dimx=50, dimy=50, R_vir=None):
    if R_vir is None:
        R_vir = np.sqrt((dimx/2)**2+(dimy/2)**2)
    xx, yy = np.meshgrid(np.arange(dimx), np.arange(dimy), sparse=True)
    ihl_map = np.sqrt(R_vir**2 - (xx-(dimx/2))**2 - (yy-(dimy/2))**2) # assumes a spherical projected profile for IHL
    ihl_map[np.isnan(ihl_map)]=0
    ihl_map /= np.sum(ihl_map)
    return ihl_map

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
    pix_sr = ((pix_width.to(u.degree))*(np.pi/180.0))**2*u.steradian # pixel solid angle in steradians
    lam_effs = np.array([1.05, 1.79])*1e-6*u.m # effective wavelength for bands
    sky_brightness = np.array([300., 370.])*sb_intensity_unit
    instrument_noise = np.array([33.1, 17.5])*sb_intensity_unit

    def __init__(self):
        pass

    def get_darktime_name(self, flight, field):
        return self.darktime_name_dict[flight][field-1]

    def find_psf_params(self, path, tm=1, field='elat10'):
        arr = np.genfromtxt(path, dtype='str')
        for entry in arr:
            if entry[0]=='TM'+str(tm) and entry[1]==field:
                beta, rc, norm = float(entry[2]), float(entry[3]), float(entry[4])
                return beta, rc, norm
        return False

    def mag_2_jansky(self, mags):
        return 3631*u.Jansky*10**(-0.4*mags)

    def mag_2_nu_Inu(self, mags, band):
        jansky_arr = self.mag_2_jansky(mags)
        return jansky_arr.to(u.nW*u.s/u.m**2)*const.c/(self.pix_sr*self.lam_effs[band])

    def get_catalog(self, catname):
        cat = np.loadtxt(self.ciberdir+'/data/'+catname)
        x_arr = cat[0,:] # + 0.5 (matlab)
        y_arr = cat[1,:] 
        m_arr = cat[2,:]
        return x_arr, y_arr, m_arr


    '''
    Produce the simulated source map

    Input:
    (Required)
     - flight: flight # (40030 for 4th flight)
     - ifield: 4,5,6,7,8 
     - m_min: min masking magnitude
     - m_max: max masking magnitude
    (Optional)
     - band (default=0): 0 or 1 (I/H) (note different indexing from MATLAB)
     - PSF (default=False): True-use full PSF, False-just put source in the center pix.
     - nbin (default=10.): if finePSF=True, nbin is the upsampling factor

     '''



    def make_srcmap(self, ifield, cat, band=0, finePSF=False, nbin=10., nx=1024, ny=1024):
        loaddir = self.ciberdir + '/psf_analytic/TM'+str(band+1)+'/'
        Npix_x = nx
        Npix_y = ny

        Nsrc = cat.shape[0]

        # get psf params
        beta, rc, norm = self.find_psf_params(self.ciberdir+'/data/psfparams.txt', tm=band+1, field=self.ciber_field_dict[ifield])
        # print 'Beta:', beta, 'rc:', rc, 'norm:', norm
        
        multfac = 7.
        Nlarge = nx+30+30 # not sure what 30 + 30 helps with 
        nwide = 100

        if finePSF:
            print('using fine PSF')
            Nlarge = int(nbin*Nlarge)
            multfac /= nbin
            nwide = int(nbin*nwide)
        t0 = time.clock()
        radmap = make_radius_map(2*Nlarge+nbin, 2*Nlarge+nbin, Nlarge+nbin, Nlarge+nbin, rc)*multfac # is the input supposed to be 2d?
        
        Imap_large = norm * np.power(1 + radmap, -3*beta/2.)
        print('Making source map TM, mrange=(%d,%d), %d sources'%(np.min(cat[:,2]),np.max(cat[:,2]),Nsrc))
        if finePSF:
            fine_srcmap = np.zeros((int(Npix_x*nbin*2), int(Npix_y*nbin*2)))
            Imap_center = Imap_large[Nlarge-nwide:Nlarge+nwide, Nlarge-nwide:Nlarge+nwide]

            xs = np.round(cat[:,0]*nbin + 4.5).astype(np.int32)
            ys = np.round(cat[:,1]*nbin + 4.5).astype(np.int32)
            for i in xrange(Nsrc):
                fine_srcmap[Nlarge/2+2+xs[i]-nwide:Nlarge/2+2+xs[i]+nwide, Nlarge/2-1+ys[i]-nwide:Nlarge/2-1+ys[i]+nwide] += Imap_center*cat[i,2]

            srcmap = rebin_map_coarse(fine_srcmap, nbin)
            srcmap *= 2*nbin**2 # needed since rebin function takes the average over nbin x nbin


        else:

            srcmap = np.zeros((Npix_x*2, Npix_y*2))
            Imap_large /= np.sum(Imap_large)     
            Imap_center = Imap_large[Nlarge-nwide:Nlarge+nwide, Nlarge-nwide:Nlarge+nwide]
   
            xs = np.round(cat[:,0]).astype(np.int32)
            ys = np.round(cat[:,1]).astype(np.int32)
            
            for i in xrange(Nsrc):
                srcmap[Nlarge/2+2+xs[i]-nwide:Nlarge/2+2+xs[i]+nwide, Nlarge/2-1+ys[i]-nwide:Nlarge/2-1+ys[i]+nwide] += Imap_center*cat[i,2]
        
        return srcmap[Npix_x/2+30:3*Npix_x/2+30, Npix_y/2+30:3*Npix_y/2+30]

   
    def make_ihl_map(self, map_shape, cat, ihl_frac, dimx=50, dimy=50):
        extra_trim = 20
        if len(cat[0])<4:
            norm_ihl = normalized_ihl_template(R_vir=10., dimx=dimx, dimy=dimy)
        else:
            rvirs = virial_radius_2_reff(cat[:,4], cat[:,3])
        ihl_map = np.zeros((map_shape[0]+dimx+extra_trim, map_shape[1]+dimy+extra_trim))

        for i, src in enumerate(cat):
            x0 = np.floor(src[0]+extra_trim/2)
            y0 = np.floor(src[1]+extra_trim/2)
            if len(src)>3:
                norm_ihl = normalized_ihl_template(R_vir=np.ceil(rvirs[i]))


            ihl_map[int(x0):int(x0+ norm_ihl.shape[0]), int(y0):int(y0 + norm_ihl.shape[1])] += norm_ihl*ihl_frac*src[2]

        return ihl_map[(norm_ihl.shape[0] + extra_trim)/2:-(norm_ihl.shape[0] + extra_trim)/2, (norm_ihl.shape[0] + extra_trim)/2:-(norm_ihl.shape[0] + extra_trim)/2]




    def make_ciber_map(self, ifield, m_min, m_max, band=0, catname=None, nsrc=0, ihl_frac=0.):
        if catname is not None:
            x_arr, y_arr, m_arr = self.get_catalog(catname)
            I_arr = self.mag_2_nu_Inu(m_arr, band)
            xyI = np.array([x_arr, y_arr, I_arr]).transpose()
            magnitude_mask_idxs = np.array([i for i in xrange(len(m_arr)) if m_arr[i]>=m_min and m_arr[i] <= m_max])
            if len(magnitude_mask_idxs) > 0:
                cat = xyI[magnitude_mask_idxs,:]
            else:
                cat = xyI
        else:
            mock_galaxy = galaxy_catalog()
            cat = mock_galaxy.generate_galaxy_catalog(nsrc)
            print('catalog has shape:', cat.shape)
            x_arr = cat[:,0]
            y_arr = cat[:,1]
            m_arr = cat[:,3] # apparent magnitude
            I_arr = self.mag_2_nu_Inu(m_arr, band)

            xyIzR = np.array([x_arr, y_arr, I_arr, cat[:,2], cat[:,6]]).transpose()
            magnitude_mask_idxs = np.array([i for i in xrange(len(m_arr)) if m_arr[i]>=m_min and m_arr[i] <= m_max])
            if len(magnitude_mask_idxs) > 0:
                cat = xyIzR[magnitude_mask_idxs,:]
            else:
                cat = xyIzR


        srcmap = self.make_srcmap(ifield, cat, band=band)
        full_map = np.zeros_like(srcmap)

        noise = np.random.normal(self.sky_brightness[band].value, self.instrument_noise[band].value, size=srcmap.shape)
        
        full_map = srcmap + noise
        if ihl_frac > 0:
            ihl_map = self.make_ihl_map(srcmap.shape, cat, ihl_frac)
            full_map += ihl_map
            return full_map, srcmap, noise, ihl_map
        else:
            return full_map, srcmap, noise


def make_galaxy_binary_map(cat, refmap):
    cat = np.array([src for src in cat if src[0]<refmap.shape[0] and src[1]<refmap.shape[1]])
    gal_map = np.zeros_like(refmap)
    for src in cat:
        gal_map[int(src[0]),int(src[1])] +=1.
    return gal_map





