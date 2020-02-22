import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
from astropy.table import Table
import pandas as pd
from ciber_mocks import *
from cross_spectrum_analysis import *


def compute_discrete_pdf_grid(cat_df, range1, range2, minz=0.0, maxz=6.0, nbin=201, key1='r', key2='redshift', target='spec_redshift',\
                              savepath=None, extra_name=''):

    '''
    
    This function takes in a catalog and constructs a two dimensional grid of PDFs over one target parameter given two other parameters.
    After this grid is calculated it is saved as a .npz file if a filepath is given.

    Note: It should be fairly straightforward to generalize this code to n-dimensional grids, though its probably not that helpful past ndim=3

    Inputs:

        cat_df (pd.DataFrame): catalog input dataframe
        range1/range2 (np.array): ranges of two parameters on grid
        minz/maxz (float): minimum/maximum redshift to evaluate PDF over
        nbin (int, default=201): how many bins used to discretize PDF
        key1/key2 (string, default='r'/'redshift'): keywords in catalog dataframe corresponding to parameters in grid
        target (string, default='spec_redshift'): target keyword in catalog indicating redshift
        savepath (string, optional, default=None): filepath to save grid of PDFs
        extra_name (string, default=''): extra string to append to saved grid file that might have useful descriptors

    Outputs:

        all_pdfs (np.array(float)): grid of redshift PDFs
        counts (np.array(float)): array indicating number of catalog sources used to calculate PDF of each grid element
        bin_centers (np.array(float)): centers of redshift bins from redshift PDF. Can be helpful when sampling from PDF

    '''
    
    counts = np.zeros((len(range1)-1, len(range2)-1))
    z_bins = np.linspace(minz, maxz, nbin)
    bin_centers = 0.5*(z_bins[1:]+z_bins[:-1])
    all_pdfs = np.zeros((len(range1)-1, len(range2)-1, nbin-1))
    print('PDF grid has shape', all_pdfs.shape)
        
    # iterate over grid
    for i in range(len(range1)-1):
        pdf_list = []
        for j in range(len(range2)-1):
            
            # filter catalog to grid element specifications
            cut = cat_df.loc[(cat_df[key1]>range1[i]) & (cat_df[key1]<range1[i+1])\
                            &(big_df[key2]>range2[j])&(cat_df[key2]<range2[j+1])]
            
            counts[i, j] = len(cut[target])
            
            all_pdfs[i, j, :] = np.histogram(cut[target], bins=z_bins)[0]
    
    if savepath is not None:
        np.savez(savepath+'/pdf_grid_counts_'+key1+'_'+key2+'_'+target+extra_name+'.npz', pdfs=np.array(all_pdfs), counts=counts, bin_centers=bin_centers, z_bins=z_bins, range1=range1, range2=range2)
        
    return all_pdfs, counts, bin_centers


def compute_kde_grid(cat_df, range1, range2, minz=0.0, maxz=6.0, nbin=201, key1='r', key2='redshift', target='spec_redshift', minpts=2, savepath=None):
    
    ''' 
    This function is effectively the same as compute_discrete_pdf_grid(), but instead returns a list of list of Kernel Density Estimates (KDEs)
    for parameter PDFs of a catalog given two parameters. Not really using this at the moment.

    Note: this function can probably be integrated with compute_discrete_pdf_grid() in the future if needed.
    '''

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

def compute_colors(cat, bands):

    ''' Given a catalog and a list of bands, compute catalog colors. Colors are computed in sequential order, so if
    bands = ['r', 'i', 'u', 'g'], the returned colors will be 'r-i', 'i-u' and 'u-g'.

    Inputs:

        cat (pd.DataFrame or np.array): input catalog to get magnitudes for colors
        bands (list of strings or list of integers): photometric bands used to compute colors

    Output:

        colors (np.array): photometric catalog colors
    '''

    colors = np.zeros(shape=(len(bands)-1, len(cat[bands[0]])))

    for b in range(len(bands)-1):
        colors[b] = cat[bands[b]]-cat[bands[b+1]]
        
    return colors

def compute_color_errors(cat, bands, keyword='u'):

    ''' Given catalog and list of photometric bands, this function computes uncertainty on color given uncertainty on magnitude.
    Note: I'm not fully sure this is correct, but is good enough when given magnitude uncertainties.'''

    color_errs = np.zeros(shape=(len(bands)-1, len(cat[keyword])))

    for b in range(len(bands)-1):
        color_errs[b] = np.sqrt(cat[bands[b]]**2+cat[bands[b+1]]**2)
        
    return color_errs


def mahalonobis_distance_colors(test_colors, test_colors_errors, all_train_colors):

    ''' This function computes the chi squared Mahalonobis distance between one set of colors and the colors of the full dataset'''

    dm = np.zeros(shape=(all_train_colors.shape[0],))
    for i in range(all_train_colors.shape[0]):
        dm[i] = np.nansum(((test_colors - all_train_colors[i,:])/(test_colors_errors))**2)

    return dm



def nearest_neighbors_ppf(all_train_colors, Dm, z_train=None, df=4, q=0.68):

    ''' 

    This determines the nearest neighbors according to the chi squared Mahalonobis distance, as determined by 
    the percent point function of the distribution, where the number of degrees of freedom is set by the number of colors.

    Inputs:
        all_train_colors (np.array): training set colors

        Dm (np.array): With reference to some catalog source, this is an array of Mahalonobis distances (see mahalonobis_distance_colors()) 
            corresonding to training set catalog sources.

        z_train (np.array, optional, default=None): training set redshifts, for use if one wants to have nearest neighbor spectroscopic 
            redshifts returned.

        df (int, default=4): number of degrees of freedom in percentile point function. df=4 corresponds to five bands.

        q (float, default=0.68): percentile used in percentile point function when making nearest neighbor cut.
    
    Outputs:
        nn_colors/nn_Dm/nn_z (np.array(float)): nearest neighbor colors, Mahalonobis distances, and redshifts.

    '''

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

    ''' This class was made to help organize/consolidate functions used for the impact of photometric redshift uncertainties on
        cross spectrum analysis. I think this class can be expanded moving forward in order to help understand/quantify uncertainties on
        various nuisance parameters. 

        Class parameters:

            zidx (int, default=2): redshift index in mock catalog
            magidx (int, default=3): magnitude index in mock catalog
            ifield (int, default=4): field index used when obtaining PSF
            nbins (int, default=22): number of bins to use for power spectra
            zmin/zmax (float, default=0.0/1.0): bounds on redshift for calculating bins used to evaluate cross spectra
            ng_bins (int, default=3): number of redshift bins in which to calculate cross spectra
            m_lim_tracer (float, optional, default=None): specifies limiting magnitude for tracer catalog in cross correlation
            ndeg (float, default=4.): number of square degrees in mock observation field of view.

        Functions:

            read_in_tracer_cat_and_map(mappath, with_noise): parses mock map and catalog to redobj class from mappath location.
            If with_noise is True, it also reads in a noise realization from the same mappath (if it exists) 

            
            load_psf(): takes path to PSF parameters from ciber_mock() class

            load_beam_correction(): if there is no PSF to compute beam correction, it loads it and then computes the correction on 
            C_ell, which gets saved as bc (beam correction) with bins rb (radius bins)

            read_in_redshift_pdf_obj(): this function parses the grid of photometric redshift PDFs that is constructed with compute_discrete_pdf_grid().

            get_redshift_bin_idxs(): for each source in a given catalog (self.cat), get the indices of the corresponding grid element it is part of.
                One can use color_sigma > 0 to sample around the catalog source magnitude/color

            get_cat_number_count_grid(): for a given catalog and grid, get catalog counts evaluated over the grid. If plot=True this plots the 2D grid.
    
            plot_z_differences(newcat): when sampling tracer catalogs that are perturbed from some initial catalog (self.cat), this function shows distribution of 
                delta-zs for catalog sources.

            sample_new_cat(): samples new realization of catalog photometric redshifts according to photometric redshift PDF grid

            sample_redshift_pdf_cross_spectrum(nsamp, plot=False): samples nsamp tracer catalog realizations from photometric redshift PDF grid,
                and then computes the resulting cross spectra. This function also makes beam corrections, and can plot intermediate diagnostics if plot=True. 
                This function returns spatial bins (rbins), cross spectrum samples, and galaxy number counts in each redshift bin, for each sample realization.


        '''
    
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
        
        try:
            map_and_cat = np.load(mappath)
        except:
            print('Failed to load map/catalog from given path')
            pass

        self.map = map_and_cat['srcmap_full']
        if with_noise:
            try:
                self.map += map_and_cat['conv_noise']
            except:
                print('No noise file found..')
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