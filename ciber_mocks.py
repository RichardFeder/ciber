import numpy as np
import pandas as pd
import astropy.units as u
from astropy import constants as const


''' Downsample map, taking average of downsampled pixels '''
def rebin_map_coarse(original_map, Nsub):
    m, n = np.array(original_map.shape)//(Nsub, Nsub)
    return original_map.reshape(m, Nsub, n, Nsub).mean((1,3))

''' Given an input map and a specified center, this will
% create a map with each pixels value its distance from
% the specified pixel. '''

def make_radius_map(map_array, cenx, ceny):
    rad = np.zeros_like(map_array)
    xpos = np.array([np.full(rad.shape[1], i) for i in xrange(rad.shape[0])]).astype(np.float32)
    ypos = np.array([np.arange(0, rad.shape[0]) for i in xrange(rad.shape[1])]).astype(np.float32)
    rad = np.sqrt((cenx - xpos)**2 + (ceny- ypos)**2)
    return rad

''' revisit this soon '''
def IHL_spherical_proj_profile(x, y, xcenter, ycenter):
    return np.sqrt(xcenter**2 + ycenter**2 - x**2 - y**2)


class ciber_mock():
    sb_intensity_unit = u.nW/u.m**2/u.steradian # helpful unit to have on hand

    ciberdir = '/Users/richardfeder/Documents/caltech/ciber2'
    darktime_name_dict = dict({36265:['NEP', 'BootesA'],36277:['SWIRE', 'NEP', 'BootesB'], \
    40030:['DGL', 'NEP', 'Lockman', 'elat10', 'elat30', 'BootesB', 'BootesA', 'SWIRE']})
    ciber_field_dict = dict({4:'elat10', 5:'elat30',6:'BootesB', 7:'BootesA', 8:'SWIRE'})

    pix_width = 7.*u.arcsec
    pix_sr = ((pix_width.to(u.degree))*(np.pi/180.0))**2*u.steradian # pixel solid angle in steradians
    lam_effs = np.array([1.05, 1.79])*1e-6*u.m # effective wavelength for bands

    def __init__():
        pass

    def get_darktime_name(self, flight, field):
        return darktime_name_dict[flight][field-1]

    def find_psf_params(self, path, tm=1, field='elat10'):
        arr = np.genfromtxt(path, dtype='str')
        for entry in arr:
            if entry[0]=='TM'+str(tm) and entry[1]==field:
                beta, rc, norm = float(entry[2]), float(entry[3]), float(entry[4])
                return beta, rc, norm
        return False

    def mag_2_jansky(mags):
        return 3631*u.Jansky*10**(-0.4*mags)

    def mag_2_nu_Inu(mags, band):
        jansky_arr = mag_2_jansky(mags)
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
    def make_srcmap_hsc(self, flight, band, ifield, m_min, m_max, band=0, finePSF=False, nbin=10.):

        catdir = ciberdir + '/data/20170617_Stacking/maps/catcoord/HSC/' # need to change path
        loaddir = ciberdir + '/psf_analytic/TM'+str(inst+1)+'/'

        dtname = get_darktime_name(flight, ifield)
        cat = np.loadtxt(ciberdir+'/data/test_hsc_catalog.txt')
        # cat_df = pd.read_csv(ciberdir+'/data/hsc_xmm_below23.csv')
        # cat_df = pd.read_csv(catdir+str(field)+'.csv', skiprows=1)


        x_arr, y_arr, m_arr = self.get_catalog('test_hsc_catalog.txt')

        Npix_x = int(np.max(x_arr)-np.min(x_arr))
        Npix_y = int(np.max(y_arr)-np.min(y_arr))
        I_arr = mag_2_nu_Inu(m_arr, inst, pix_sr)
        
        xyI = np.array([x_arr, y_arr, I_arr]).transpose()
        # select catalog data
        magnitude_mask_idxs = np.array([i for i in xrange(len(m_arr)) if m_arr[i]>=m_min and m_arr[i] <= m_max])
        xyI = xyI[magnitude_mask_idxs,:]
        Nsrc = xyI.shape[0]

        # get psf params
        beta, rc, norm = find_psf_params(ciberdir+'/data/psfparams.txt', tm=band+1, field=ciber_field_dict[ifield])
        print 'Beta:', beta, 'rc:', rc, 'norm:', norm
        
        multfac = 0.7
        Nlarge = 1024+30+30 # what is this?
        nwide = 100
        if finePSF:
            print('using fine PSF')
            Nlarge = int(nbin*Nlarge)
            multfac *= nbin
            nwide *= nbin

        radmap = make_radius_map(np.zeros((2*Nlarge+1, 2*Nlarge+1)), Nlarge+1, Nlarge+1)*multfac # is the input supposed to be 2d?
        Imap_large = norm * (1+(radmap/rc)**2)**(-3*beta/2)

        print('Making source map TM%d %s, mrange=(%d,%d), %d sources'%(inst,dtname, m_min,m_max,Nsrc))

        if finePSF:
            xy_small_arr = xyI[:,:-1]*nbin + 4.5 # matlab convention 1 index gives -4.5, with 0 index I believe +4.5

            fine_srcmap = np.zeros((Npix_x*nbin, Npix_y*nbin))

            dx = Nlarge - np.round(xy_small_arr[:,0])
            dy = Nlarge - np.round(xy_small_arr[:,1])
            print(np.min(dx), np.max(dx), np.min(dy), np.max(dy))

            for i in xrange(Nsrc):
                Imap = Imap_large[ dx[i]:Npix_x*nbin+dx[i], dy[i]:Npix_y*nbin+dy[i]]*sub_xyI[i,2]
                fine_srcmap += Imap

            srcmap = rebin_map_coarse(fine_srcmap, nbin)
            srcmap *= nbin**2 # needed since rebin function takes the average over nbin x nbin

            return srcmap

        else:
            srcmap = np.zeros((Npix_x*2, Npix_y*2))
            Imap_large /= np.sum(Imap_large)     
            Imap_center = Imap_large[Nlarge-nwide:Nlarge+nwide, Nlarge-nwide:Nlarge+nwide]
   
            xs = np.round(xyI[:,0]).astype(np.int32)
            ys = np.round(xyI[:,1]).astype(np.int32)
            
            for i in xrange(Nsrc):
                srcmap[Nlarge/2+2+xs[i]-nwide:Nlarge/2+2+xs[i]+nwide, Nlarge/2-1+ys[i]-nwide:Nlarge/2-1+ys[i]+nwide] += Imap_center*xyI[i,2]
        
            return bigsrcmap[Npix_x/2+30:3*Npix_x/2+30, Npix_y/2+30:3*Npix_y/2+30]



