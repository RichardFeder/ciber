import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
from astropy.table import Table
import pandas as pd
from ciber_mocks import *
from cross_spectrum_analysis import *


def compute_discrete_pdf_grid(cat_df, range1, range2, minz=0.0, maxz=6.0, nbin=201, key1='r', key2='redshift', target='spec_redshift',\
                              minpts=2, savepath=None, extra_name=''):
    
    counts = np.zeros((len(range1)-1, len(range2)-1))
    
    z_bins = np.linspace(minz, maxz, nbin)
    bin_centers = 0.5*(z_bins[1:]+z_bins[:-1])
    all_pdfs = np.zeros((len(range1)-1, len(range2)-1, nbin-1))
    print('PDF grid has shape', all_pdfs.shape)
        
    for i in range(len(range1)-1):
        pdf_list = []
        for j in range(len(range2)-1):
            
            cut = cat_df.loc[(cat_df[key1]>range1[i]) & (cat_df[key1]<range1[i+1])\
                            &(big_df[key2]>range2[j])&(cat_df[key2]<range2[j+1])]
            
            counts[i, j] = len(cut[target])
            
            all_pdfs[i, j, :] = np.histogram(cut[target], bins=z_bins)[0]
    
    if savepath is not None:
        np.savez(savepath+'/pdf_grid_counts_'+key1+'_'+key2+'_'+target+extra_name+'.npz', pdfs=np.array(all_pdfs), counts=counts, bin_centers=bin_centers, z_bins=z_bins, range1=range1, range2=range2)
        
    return all_pdfs, counts, bin_centers


def compute_kde_grid(cat_df, range1, range2, minz=0.0, maxz=6.0, nbin=201, key1='r', key2='redshift', target='spec_redshift', minpts=2, savepath=None):
    all_kdes = []
    
    counts = np.zeros((len(range1)-1, len(range2)-1))
        
    for i in range(len(range1)-1):
        ke_list = []
        for j in range(len(range2)-1):
            
            cut = cat_df.loc[(cat_df[key1]>range1[i]) & (cat_df[key1]<range1[i+1])\
                            &(big_df[key2]>range2[j])&(cat_df[key2]<range2[j+1])]
            
            counts[i, j] = len(cut['spec_redshift'])
            
            if len(cut[target]) >= minpts:
                kde_pdf = scipy.stats.gaussian_kde(cut[target])
            else:
                kde_pdf = None
            kde_list.append(kde_pdf)

         
        all_kdes.append(kde_list)
    
    return all_pdfs, counts

def compute_colors(spec_cat, bands, keyword='spec_redshift'):

    colors = np.zeros(shape=(len(bands)-1, len(spec_cat[keyword])))

    for b in range(len(bands)-1):
        colors[b] = spec_cat[bands[b]]-spec_cat[bands[b+1]]
        
    return colors

def compute_color_errors(cat, bands, keyword='u'):
    color_errs = np.zeros(shape=(len(bands)-1, len(cat[keyword])))

    for b in range(len(bands)-1):
        color_errs[b] = np.sqrt(cat[bands[b]]**2+cat[bands[b+1]]**2)
        
    return color_errs


# lets start by making a function that computes the chi squared Mahalonobis distance between 
# one set of colors and the colors of the full dataset

def mahalonobis_distance_colors(test_colors, test_colors_errors, all_train_colors):
    dm = np.zeros(shape=(all_train_colors.shape[0],))
    for i in range(all_train_colors.shape[0]):
        dm[i] = np.nansum(((test_colors - all_train_colors[i,:])/(test_colors_errors))**2)

    return dm

# this determines the nearest neighbors according to the chi squared Mahalonobis distance, 
# as determined by the percent point function of the distribution, where the number of degrees
# of freedom is set by the number of colors

def nearest_neighbors_ppf(all_train_colors, Dm, z_train=None, df=4, q=0.68, zmax=3.0, nbins=30):
    cutoff = scipy.stats.chi2.ppf(q, df)
    mask = (Dm < cutoff)
    nn_colors = all_train_colors[mask,:]
    nn_Dm = Dm[mask]
    if z_train is not None:
        # compute redshift PDF
        nn_z = z_train[mask]
        
        return nn_colors, nn_Dm, nn_z
        
    return nn_colors, nn_Dm




class redobj():
    
    degree_per_steradian = 3.046e-4
    
    def __init__(self, zidx=2, magidx=3, ifield=4, nbins=22, zmin=0.0, zmax=1.0, ng_bins=3, m_lim_tracer=None, ndeg=4.):
        self.zidx = zidx
        self.magidx = magidx
        self.ifield=ifield
        self.cmock = ciber_mock()
        self.psf = None
        self.nbins = nbins
        self.ng_bins = ng_bins
        self.m_lim_tracer = m_lim_tracer
        self.xcorr_zbins = np.linspace(zmin, zmax, ng_bins+1)
        self.ndeg = ndeg
        
    
    def read_in_tracer_cat_and_map(self, mappath, with_noise=True):
        
        map_and_cat = np.load(mappath)
        self.map = map_and_cat['srcmap_full']
        if with_noise:
            self.map += map_and_cat['conv_noise']
        
        self.sterad_per_pix =  (2*np.pi/180./self.map.shape[0])**2

        if self.m_lim_tracer is not None:
            tracer = map_and_cat['catalog']
            mask = (tracer[:, self.magidx] < self.m_lim_tracer)
            self.cat = tracer[mask, :]
        else:
            self.cat = map_and_cat['catalog']
            
        print('cat has shape ', self.cat.shape)
            
    
    def load_psf(self):
        self.psf = make_psf_template(self.cmock.ciberdir+'/data/psfparams.txt', self.cmock.ciber_field_dict[self.ifield], 1, large=True)[0]
        
        
    def load_beam_correction(self):
        if self.psf is None:
            self.load_psf()
        
        rb, bc = compute_beam_correction(self.psf, nbins=self.nbins)
        self.rb = rb
        self.bc = bc
        print('bc length', bc.shape)
    
        
    def read_in_redshift_pdf_obj(self, path, pdf_key='pdfs', zbinkey='bin_centers'):
        redshift_pdf_obj = np.load(path)
        self.redshift_pdfs = redshift_pdf_obj[pdf_key]
        self.redshift_pdf_bins = redshift_pdf_obj[zbinkey]
        self.zrange = redshift_pdf_obj['range1']
        self.magrange = redshift_pdf_obj['range2']
        
        print('self.zrange:', self.zrange)
        print('self.magrange:', self.magrange)
        

    def get_redshift_grid_idxs(self, color_sigma=0):
        self.z_pdf_idxs = np.digitize(self.cat[:, self.zidx], self.zrange)-1
        self.mag_pdf_idxs = np.digitize(self.cat[:, self.magidx]+np.random.normal(0, scale=color_sigma, size=self.cat.shape[0]), self.magrange)-1
    

    def get_cat_number_count_grid(self, cat=None, plot=False, band='J'):
        counts_map = np.zeros((len(self.magrange)-1, len(self.zrange)-1))

        for j in range(len(self.zrange)-1):
            for i in range(len(self.magrange)-1):
                if cat is None:
                    mask = (self.cat[:, self.magidx]>self.magrange[i]) & (self.cat[:, self.magidx]<self.magrange[i+1])\
                                       &(self.cat[:, self.zidx]>self.zrange[j])&(self.cat[:, self.zidx]<self.zrange[j+1])
                else:
                    mask = (cat[:, self.magidx]>self.magrange[i]) & (cat[:, self.magidx]<self.magrange[i+1])\
                                       &(cat[:, self.zidx]>self.zrange[j])&(cat[:, self.zidx]<self.zrange[j+1])

                counts_map[i,j] = np.sum(mask)
                
        if plot:       
            f = plt.figure(figsize=(8, 8))
            plt.imshow(counts_map, aspect=0.15, norm=matplotlib.colors.LogNorm(), extent=[self.zrange[0], self.zrange[-1],self.magrange[-1], self.magrange[0]])
            plt.colorbar()
            plt.xlabel('Redshift $z$', fontsize=16)
            plt.ylabel('Simulated '+band+' band magnitude', fontsize=16)
            plt.show()
            
            return counts_map, f
                
        return counts_map
        

    def plot_z_differences(self, newcat):
        dz = newcat[:, self.zidx] - self.cat[:,self.zidx]
        print('dz:', dz[np.nonzero(dz)], np.sum(np.nonzero(dz)))

        plt.figure()
        plt.hist(dz, bins=50)
        plt.yscale('symlog')
        plt.xlabel('$\\Delta z$', fontsize=16)
        plt.ylabel('$N$', fontsize=16)
        plt.show()
        
        
    def sample_sdss_selection_function(self):
        # TODO
        pass
        
    def sample_new_cat(self):
        total_changed = 0
        catalog = self.cat.copy()
        for a in range(len(self.zrange)-1):
            for b in range(len(self.magrange)-1):
                mask = (self.z_pdf_idxs==a)&(self.mag_pdf_idxs==b)
                total_changed += np.sum(mask)
                pdf = self.redshift_pdfs[a,b]
                pdf /= np.sum(pdf)
                pdf[np.isnan(pdf)] = 0.
                if np.sum(pdf)==0:
                    pdf[:] = 1./pdf.shape[0]
                newz = np.random.choice(self.redshift_pdf_bins, p=pdf, size=np.sum(mask))
                catalog[mask, self.zidx] = newz
                
        return catalog
    
    
    def sample_redshift_pdf_cross_spectrum(self, nsamp=50, plot=False):
        
        print('xcorr redshift bins:', self.xcorr_zbins)
        
        Nside = self.map.shape[0]
        ps_samps = np.zeros((nsamp, len(self.xcorr_zbins)-1, self.nbins))
        ng_samps = np.zeros((nsamp, len(self.xcorr_zbins)-1))
        
        for n in range(nsamp):
            print('n=', n)
            cts_per_steradian = np.zeros(self.ng_bins)
            nsrc = self.cat.shape[0]
            print('nsrc=', nsrc)
            
            catalog = self.sample_new_cat()
            
            if plot:
                self.plot_z_differences(catalog)
            
            for i in range(self.ng_bins):
                galcts = make_galaxy_cts_map(catalog, self.map.shape, 1, \
                                             magidx=self.magidx,m_max=self.m_lim_tracer, zmin=self.xcorr_zbins[i],\
                                             zmax=self.xcorr_zbins[i+1], zidx=self.zidx, normalize=False)
                
                cts_per_steradian[i] = np.sum(galcts)/(self.ndeg*self.degree_per_steradian)
                rbin, radprof, radstd = compute_cl(self.map-np.mean(self.map), (galcts-np.mean(galcts))/np.mean(galcts), nbins=self.nbins, sterad_term=self.sterad_per_pix)
                
                ps_samps[n, i, :] = radprof /self.bc / Nside**2
                ng_samps[n, i] = np.sum(galcts)
        
        return rbin, ps_samps, ng_samps