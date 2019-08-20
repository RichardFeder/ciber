import numpy as np
import astropy.io.fits as fits
import astropy.wcs as wcs
import pandas as pd
import scipy.io

def fits_to_dataframe(fits_path):
    table = Table.read(fits_path)
    df = table.to_pandas()
    return df

def dataframe_to_fits(df, fits_path):
    table = Table.from_pandas(df)
    table.write(fits_path)


class ciber_data():
    
    basepath = '/Users/richardfeder/Documents/caltech/ciber2/'
    mypaths = dict({'ciber_data':basepath+'/data/ciber_data/', \
                    'alldat':basepath+'/data/ciber_data/alldat/', \
                    'srcmapdir':basepath+'/data/ciber_data/srcmaps/', \
                   'figuredir':basepath+'/figures/', 
                   'catdir':basepath+'/data/catalogs/'})
    
    ifield_dict = dict({4:'elat10',5:'elat30', 6:'Bootes B', 7:'Bootes A', 8:'SWIRE'})
    radec_fields = dict({'SWIRE':[242.77, 54.61], 'NEP':[270.52, 66.43], 'Bootes A':[218.28, 34.71], 'Bootes B':[218.28,34.71], \
                        'elat10':[191.50, 8.25], 'elat30':[193.943, 27.998]})
    
    catalog_names = ['hsc_pdr1_deep.forced (w/photo-zs)', 'hsc_pdr2_dud_specz', 'sdss_dr15 (w/photo-zs)']
    
    
    def __init__(self, ifield, inst):
        self.ifield = ifield
        self.inst = inst
        
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
        self.ifield=ifield
        return self.stackmapdat['mask_inst_clip'][0][ifield-1]
    
    def load_maps(self):
        loaddir = self.mypaths['alldat'] + 'TM'+str(self.inst)+'/'
        mask_path = loaddir+'maskdat'
        self.maskdat = scipy.io.loadmat(mask_path+'.mat')
        self.stackmapdat = scipy.io.loadmat(loaddir+'/stackmapdat.mat')['stackmapdat']  
        
    def load_catalog(self, catpath):
        
        self.catalog = pd.read_csv(self.mypaths['catdir']+catpath)
        self.cat_keys = self.catalog.keys()
        print('Keys:', self.cat_keys)
        
    def show_map(self, input_map, lowpct=5, highpct=95, xmin=None, xmax=None, ymin=None, ymax=None, colorbar=True, cat=None, title_string='', cat_info=None, save=False):
        plt.figure(figsize=(10,10))
        title = title_string+' (ifield='+str(self.ifield)+', inst='+str(self.inst)+')'
        if cat_info is not None:
            title += ' ('+cat_info+')'
        plt.title(title, fontsize=16)
        plt.imshow(input_map, vmin=np.percentile(input_map, lowpct), vmax=np.percentile(input_map, highpct))
        if xmin is not None:
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        if xmax-xmin < input_map.shape[0]:
            title_string+='_zoomed'
        plt.xlabel('$\\theta_x$', fontsize=16)
        plt.ylabel('$\\theta_y$', fontsize=16)
        if colorbar:
            plt.colorbar()

        if cat is not None:
            plt.scatter(cat['x'+str(self.inst)], cat['y'+str(self.inst)], label='Catalog', marker='x', color='r', alpha=0.5)
            plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(self.mypaths['figuredir']+title_string+'_ifield='+str(self.ifield)+'_inst='+str(self.inst)+'.png', bbox_inches='tight')
        plt.show() 

