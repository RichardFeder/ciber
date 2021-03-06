import numpy as np
import astropy.io.fits as fits
import astropy.wcs as wcs
import pandas as pd
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
from astropy.table import Table

def find_psf_params(path, tm=1, field='elat10'):
    ''' This function loads/parses PSF parameters given a path, field, and choice of tm (forget what this means)'''
    arr = np.genfromtxt(path, dtype='str')
    for entry in arr:
        if entry[0]=='TM'+str(tm) and entry[1]==field:
            beta, rc, norm = float(entry[2]), float(entry[3]), float(entry[4])
            return beta, rc, norm
    return False

def load_ciber_srcmap(idx, with_noise=True, datapath='/Users/richardfeder/Documents/ciber2/ciber/mock_data/',tail_name='mmin=18.4_mmax=25'):
    load_srcmap = np.load(datapath+'/ciber_mock_'+str(idx)+tail_name+'.npz')['srcmap']
    if with_noise:
        noise = np.load(datapath+'/ciber_mock_'+str(idx)+tail_name+'.npz')['conv_noise']
        load_srcmap += noise
        
    return load_srcmap

def make_psf_template(path, field, band, nx=1024, pad=30, nwide=20, nbin=0, multfac=7., large=True):
    
    ''' This function loads PSF parameters and generates a normalized template over some array. 
    The PSF is parameterized as a power law'''

    beta, rc, norm = find_psf_params(path, tm=band+1, field=field)
    Nlarge = nx+pad+pad
    radmap = make_radius_map(2*Nlarge+nbin, 2*Nlarge+nbin, Nlarge+nbin, Nlarge+nbin, rc)*multfac
    Imap_large = norm*np.power(1.+radmap, -3*beta/2.)
    Imap_large /= np.sum(Imap_large)
    psf_template = Imap_large[Nlarge-nwide:Nlarge+nwide+1, Nlarge-nwide:Nlarge+nwide+1]
    
    if large:
        psf = psf_large(psf_template)
        return psf, psf_template
    else:
        psf_template

def make_radius_map(dimx, dimy, cenx, ceny, rc=1.):
    ''' This function calculates a map, where the value of each pixel is its distance from the central pixel.
    Useful for making PSF templates and other map making functions'''
    x = np.arange(dimx)
    y = np.arange(dimy)
    xx, yy = np.meshgrid(x, y, sparse=True)
    return (((cenx - xx)/rc)**2 + ((ceny - yy)/rc)**2)

def make_radius_map_yt(mapin, cenx, ceny):
    '''
    return radmap of size mapin.shape. 
    radmap[i,j] = distance between (i,j) and (cenx, ceny)
    '''
    Nx, Ny = mapin.shape
    xx, yy = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    radmap = np.sqrt((xx - cenx)**2 + (yy - ceny)**2)
    return radmap

def psf_large(psf_template, mapdim=1024):
    ''' all this does is place the 40x40 PSF template at the center of a full map 
    so one can compute the FT and get the beam correction for appropriate ell values
    this assumes that beyond the original template, the beam correction will be 
    exactly unity, which is fair enough for our purposes'''
    psf_temp = np.zeros((mapdim, mapdim))
    psf_temp[mapdim/2 - 20:mapdim/2+21, mapdim/2-20:mapdim/2+21] = psf_template
    psf_temp /= np.sum(psf_temp)
    return psf_temp


def read_ciber_powerspectra(filename):
    ''' Given some file path name, this function loads/parses previously measured CIBER power spectra'''
    array = np.loadtxt(filename, skiprows=8)
    ells = array[:,0]
    norm_cl = array[:,1]
    norm_dcl_lower = array[:,2]
    norm_dcl_upper = array[:,3]
    return np.array([ells, norm_cl, norm_dcl_lower, norm_dcl_upper])



''' The classes/functions below have not been fully developed or used yet, so I will defer documentation until they are.'''

class ciber_data():
    
    basepath = '/Users/richardfeder/Documents/caltech/ciber2/'
    datapath = basepath+'data/'
    
    mypaths = dict({'ciber_data':datapath+'ciber_data/', \
                    'alldat':datapath+'/ciber_data/alldat/', \
                    'srcmapdir':datapath+'/ciber_data/srcmaps/', \
                   'figuredir':basepath+'/figures/', 
                   'catdir':datapath+'/catalogs/SDSScats'})
    
    ifield_title_dict = dict({4:'Elat 10',5:'Elat 30', 6:'Bootes B', 7:'Bootes A', 8:'SWIRE'})
    ifield_file_dict = dict({4:'elat10', 5:'elat30', 6:'BootesB', 7:'BootesA', 8:'SWIRE'})
    radec_fields = dict({'SWIRE':[242.77, 54.61], 'NEP':[270.52, 66.43], 'Bootes A':[218.28, 34.71], 'Bootes B':[218.28,34.71], \
                        'elat10':[191.50, 8.25], 'elat30':[193.943, 27.998]})
    
    catalog_names = ['hsc_pdr1_deep.forced (w/photo-zs)', 'hsc_pdr2_dud_specz', 'sdss_dr15 (w/photo-zs)']
    
    
    def __init__(self, ifield, inst, nbins=20, psf=None):
            
        self.ifield = ifield
        self.inst = inst
        self.nbins=nbins
        self.psf=psf


    def get_cal_apf2nWpm2ps(self, inst):
        
        if inst==1:
            cal_apf2nWpm2ps = dict({4:-347.92, 5:-305.81, 6:-369.32, 7:-333.67, 8:-314.33})
        else:
            cal_apf2nWpm2ps = dict({4:-117.69, 5:-116.20, 6:-118.79, 7:-127.43, 8:-117.96})
        
        return cal_apf2nWpm2ps
        

    def load_darkstat_mat(self, tail_path='darkstat.mat'):
        try:
            self.darkstat = scipy.io.loadmat(self.datapath+tail_path)
        except:
            print('No file found, file path is currently ', self.datapath+tail_path)

    def list_available_catalogs(self, field):
        if field=='ELAIS-N1':
            print(self.catalog_names[0])
            print(self.catalog_names[1])
            print(self.catalog_names[2])
            print('HSC_PDF1_deep.forced (w/ photo-zs), HSC PDR2_dud_specz')
        elif field=='NEP':
            print('SDSS')
    
    def radec_2_thetaxy(self, ra, dec, field=None, ra0=0, dec0=0):
        if field is not None:
            ra0 = self.radec_fields[field][0]
            dec0 = self.radec_fields[field][1]

        print('Projecting from ('+str(ra0)+','+str(dec0)+')')
        ra_transf = (ra - ra0)/np.cos(dec*np.pi/180)
        dec_transf = dec - dec0
        return ra_transf, dec_transf        
    
    def load_map_from_stack(self, maptype):
        self.maptype=maptype
        return self.stackmapdat[maptype][0][self.ifield-1]
    
    def load_cbmap(self): 
        return self.stackmapdat['cbmap'][0][self.ifield-1]
        
    def load_pointsrc_map(self):
        return self.stackmapdat['psmap'][0][self.ifield-1]
    
    def load_mask_inst(self):
        return self.stackmapdat['mask_inst_clip'][0][self.ifield-1]
    
    def load_maps(self):
        loaddir = self.mypaths['alldat'] + 'TM'+str(self.inst)+'/'
        mask_path = loaddir+'maskdat'
        self.maskdat = scipy.io.loadmat(mask_path+'.mat')
        self.stackmapdat = scipy.io.loadmat(loaddir+'/stackmapdat.mat')['stackmapdat']  
        
    def load_catalog(self, tailname, datatype='fits'):
        
        filepath = self.mypaths['catdir']+'/ifield'+str(self.ifield)+'/'+tailname
        
        if datatype=='csv':
            self.catalog = pd.read_csv(filepath)
        
        elif datatype=='fits':
            self.catalog = fits_to_dataframe(filepath)

        self.cat_keys = self.catalog.keys()
        print('Keys:', self.cat_keys)
        

    def load_psf(self):
        print(self.ifield_file_dict[self.ifield])
        self.psf = make_psf_template(self.datapath+'psfparams.txt', self.ifield_file_dict[self.ifield], 1)[0]
        print(self.psf)
        
        
    def load_beam_correction(self):
        if self.psf is None:
            self.load_psf()
        
        rb, bc = compute_beam_correction(self.psf, nbins=self.nbins)
        self.rb = rb
        self.bc = bc
        print('bc length', bc.shape)
    
    def show_map(self, input_map, lowpct=5, highpct=95, xlim=None, ylim=None, colorbar=True, cat=None,cat_mlim=None, \
                 title_string='', cat_info=None, save=False, return_fig=True, mag_key='r'):
        f = plt.figure(figsize=(10,10))
        title = title_string+' (ifield='+str(self.ifield)+', inst='+str(self.inst)+')'
        if cat_info is not None:
            title += ' ('+cat_info+')'
        plt.title(title, fontsize=16)
        plt.imshow(input_map, vmin=np.percentile(input_map, lowpct), vmax=np.percentile(input_map, highpct))
        
        if xlim is not None:
            plt.xlim(xlim)
            plt.ylim(ylim)

            if xlim[1]-xlim[0] < input_map.shape[0]:
                title_string+='_zoomed'

        plt.xlabel('$\\theta_x$', fontsize=16)
        plt.ylabel('$\\theta_y$', fontsize=16)
        if colorbar:
            plt.colorbar()

        if cat is not None:
            if cat_mlim is not None:
                cat = cat.loc[(cat[mag_key] < cat_mlim)]
                
            plt.scatter(cat['x'+str(self.inst)], cat['y'+str(self.inst)], label='Catalog', marker='x', color='r', alpha=0.5)
            plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(self.mypaths['figuredir']+title_string+'_ifield='+str(self.ifield)+'_inst='+str(self.inst)+'.png', bbox_inches='tight')
        plt.show() 
        
        if return_fig:
            return f


class CIBER_preprocess():
    
    basepath = '/Users/richardfeder/Documents/caltech/ciber2/'
    mypaths = dict({'ciber_data':basepath+'/data/ciber_data/', \
                    'alldat':basepath+'/data/ciber_data/alldat/', \
                    'srcmapdir':basepath+'/data/ciber_data/srcmaps/'})
    
    def __init__():
        pass
    
    def get_linearized_map(self, flight, inst):
        
        return 
    
    ''' This function retrieves the correct calibration factor,
    going from ADU/fr to nW/m2/sr given the instrument.
    
    The values are the results from fit_calfac.m
    '''
    def get_cal_apf2nWpm2ps(self, inst):
        if inst==1:
            cal_apf2nWpm2ps = dict({4:-347.92, 5:-305.81, 6:-369.32, 7:-333.67, 8:-314.33})
        else:
            cal_apf2nWpm2ps = dict({4:-117.69, 5:-116.20, 6:-118.79, 7:-127.43, 8:-117.96})
        return cal_apf2nWpm2ps
    
    def load_maps(self, inst):
        loaddir = self.mypaths['alldat'] + 'TM'+str(inst)+'/'
        mask_path = loaddir+'maskdat'
        self.maskdat = scipy.io.loadmat(mask_path+'.mat')['mask']
        self.stackmapdat = scipy.io.loadmat(loaddir+'/stackmapdat.mat')['stackmapdat']  

        
    def sigma_clip_mask(self, input_map, mask, a=3., b=5.):
        # what are 3 and 5? 
        pass
    
    def compute_quantile(self, input_map, mask, quantile):
        nonzero_vals = input_map[mask > 0]
        return np.percentile(nonzero_vals, quantile*100.)
    
        
    def stack_preprocess(self, flight, inst, varargin, rmin=None, ifields=[4, 5, 6, 7, 8]):
        loaddir = self.mypaths['alldat'] + 'TM'+str(inst)+'/'
        mask_path = loaddir+'maskdat'
        self.maskdat = scipy.io.loadmat(mask_path+'.mat')['mask']
        self.stackmapdat = scipy.io.loadmat(loaddir+'/stackmapdat.mat')['stackmapdat']  
        
        cal = get_cal_apf2nWpm2ps(inst)
        
        for ifield in ifields: # 4, 5, 6, 7, 8 default
            
            dt = get_dark_times(flight, inst, ifield)
            ifield_idx = ifield-1 # matlab 2 python lol
            if ifield==5:
                cbmap_raw = self.stackmapdat['map_last10'][0][ifield_idx]*cal[ifield_idx]
            else:
                cbmap_raw = self.stackmapdat['map'][0][ifield_idx] * cal[ifield_idx]
                
            psmap_raw = fits.open(self.mypaths['srcmapdir']+'TM'+str(inst)+'/'+dt['name']+'_srcmap_ps_all.fits')
            
            # masks
            mask_inst = self.stackmapdat['mask'][0][ifield_idx]
            if rmin==2:
                strmask = self.maskdat['strmask_stack_rmin2'][0][ifield_idx]
            else:
                strmask = self.maskdat['strmask_stack'][0][ifield_idx]
                
            totmask = mask_inst['strmask']
            
            Q1 = self.compute_quantile(cbmap_raw, totmask, 0.25)
            Q3 = self.compute_quantile(cbmap_raw, totmask, 0.75)
            IQR = Q3-Q1
            clipmin = Q1 - 3*IQR
            clipmax = Q3 + 3*IQR
            sigmask = totmask.copy()
            sigmask[np.logical_or(cbmap_raw > clipmax, cbmap_raw < clipmin)] = 0
            
            cbmean = np.mean(cbmap_raw[sigmask > 0])
            psmean = np.mean(psmap_raw[sigmask > 0])
            print('CB mean:', cbmean)
            print('PS mean:', psmean)
            
            mask_inst_clip1 = mask_inst_clip.copy()
            
            

            

