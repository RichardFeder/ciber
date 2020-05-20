import treecorr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def compute_2pcf(skymaps=None, mask=None, nmaps=50, npatch=10, plot=False, verbose=False):

    xis, varxis = [], []
    deg_per_pix = (7./3600.)
    

    xind = np.indices((1024,1024))[0].ravel()*deg_per_pix
    yind = np.indices((1024,1024))[1].ravel()*deg_per_pix
    
    
    for i in range(nmaps):
    	if verbose:
    		print('i = ', i)
        if skymaps is None:
            load_srcmap = np.load('data/nobright_ciber_mock_sims/ciber_mock_'+str(i)+'mmin=18.4_mmax=25.npz')['srcmap']
        else:
            load_srcmap = skymaps[i]
        
        skymap = load_srcmap - np.mean(load_srcmap)
        
        if mask is not None:
            weights = mask.ravel()
        else:
            weights = None
            
        if npatch > 1:
            cat = treecorr.Catalog(ra=xind, dec=yind, k=skymap.ravel(), w=weights, ra_units='deg', dec_units='deg', npatch=npatch)
            kk = treecorr.KKCorrelation(min_sep= 2*deg_per_pix, max_sep=2., bin_size=0.3, sep_units='deg', var_method='sample')

        else:
            cat = treecorr.Catalog(ra=xind, dec=yind, k=skymap.ravel(), w=weights, ra_units='deg', dec_units='deg')
            kk = treecorr.KKCorrelation(min_sep= 2*deg_per_pix, max_sep=2., bin_size=0.3, sep_units='deg')
        
        kk.process(cat, metric='Euclidean')
        
        bins = 10**(np.linspace(np.log10(2.*deg_per_pix), np.log10(2.), len(kk.xi)))
            
        if plot:
            plt.figure()
            plt.errorbar(10**(np.linspace(np.log10(2.*deg_per_pix), np.log10(2.), len(kk.xi))), kk.xi, yerr=np.sqrt(kk.varxi), marker='.')
            plt.xlabel('$\\theta$ (deg)', fontsize=16)
            plt.ylabel('$\\xi(r)$ (arbitrary units)', fontsize=16)
            plt.xscale('log')
            plt.show()
            
        xis.append(kk.xi)
        varxis.append(kk.varxi)
        
    if len(xis)==1:
        return kk.xi, kk.varxi, bins
        
    return xis, varxis, bins
    

def compute_ensemble_offsets(mask=None, skymaps=None, mode='white', nmaps=10, imdim=1024, npatch=10, plot_indiv=False):
    
    frac_delta_xis = []
    
    
    if mode=='white' and skymaps is None:
        skymaps = [np.random.normal(size=(imdim, imdim)) for i in range(nmaps)]
        
    if mask is None:
        mask = np.ones(size=(imdim, imdim))
    
    masked_xis, varxis, bins = compute_2pcf(skymaps, mask=mask, nmaps=nmaps, npatch=npatch, plot=plot_indiv)
    print('done with masked, now unmasked')
    unmasked_xis, unmasked_varxis, bins = compute_2pcf(skymaps, nmaps=nmaps, npatch=npatch, plot=plot_indiv)
    
    for i in range(nmaps):
        
        frac_delta_xis.append((unmasked_xis[i]-masked_xis[i])/unmasked_xis[i])
        

    return bins, frac_delta_xis, masked_xis, unmasked_xis
        
