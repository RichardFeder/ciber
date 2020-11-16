import treecorr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from plotting_fns import *


# ------------------------------------------ main TreeCorr stuff below --------------------------------------------

def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


def get_treecorr_catalog_vals(xind, yind, skymap, mask=None, prefilter=False):
    
    ks = skymap.ravel()
    
    if mask is not None:
        weights = mask.ravel()
    else:
        weights = None

    if prefilter and weights is not None:
        xind_cat = xind[weights.astype(np.bool)]
        yind_cat = yind[weights.astype(np.bool)]
        ks = ks[weights.astype(np.bool)]
        weights = np.ones_like(xind_cat)
    else:
        xind_cat = xind
        yind_cat = yind
            
    return xind_cat, yind_cat, ks, weights

def wth2d_to_cl(wthetas, ell_bins, binl, pixsize=7., dim=1024, fac=1.):
    
    c_ells = np.zeros((len(wthetas), len(binl)-1))

    cl2s = []
    sterad_per_pix = (pixsize*fac/3600./180*np.pi)**2

    for j, wthet in enumerate(wthetas):
        cl2 = np.sqrt(np.real(fft2(wthet*sterad_per_pix)*np.conj(fft2(wthet*sterad_per_pix))))

        V = dim * dim * sterad_per_pix

        l2d = get_l2d(dim, dim, 7.*fac)
        
        cl2s.append(np.fft.ifftshift(cl2 / V))

        lb, Cl, Clerr =  azim_average_cl2d(np.fft.ifftshift(cl2 / V), l2d, lbinedges=binl, lbins=ell_bins)

        c_ells[j] = Cl

    return c_ells, cl2s

def compute_2pcf(skymaps=None, mask=None, nmaps=50, binning='log',nlinbin=100, pix_units='deg',\
                      npatch=10, var_method='shot', return_cov=True, prefilter=False, verbose=False,\
                      plot=False, thetamin_fac=1.0, mask_list=None,\
                     thetamax=2., thetabin_size=0.3, pixel_size=7., imdim=1024, plot_title='realization', bin_slop=0.1):
    
    ''' Inputs:
    
        thetamax (deg): Maximum theta desired. If using radian pixel units this will get converted to radians in the code.
        nlinbin (int, default: 100): number of linear bins to use if doing linear binning
    '''

    
    fs = []
    wthetas, varwthetas, covs = [], [], []
    deg_per_pix = (pixel_size/3600.) # arcseconds to degrees
    thetamin = thetamin_fac*deg_per_pix # minimum separation is twice the pixel size
    
    print('thetamin is ', thetamin)
    print('in arcseconds this is ', thetamin*3600.)

    if skymaps is not None:
        imdim = skymaps[0].shape[0]
        if nmaps > len(skymaps):
            nmaps = len(skymaps)
        
    deg_per_rad = np.pi/180.
        
    xind = np.indices((imdim,imdim))[0].ravel()*deg_per_pix
    yind = np.indices((imdim,imdim))[1].ravel()*deg_per_pix
    
    # I checked and having the pix_units in radians or degrees does not change the result so long as 
    # they are consistently used within the Catalog object.
    
    if pix_units=='rad':
        xind *= deg_per_rad
        yind *= deg_per_rad
        thetamax *= deg_per_rad
        thetamin *= deg_per_rad
    if verbose:
        print('xind:', xind)
        
    if npatch>1 and var_method=='shot':
        print('because npatch='+str(npatch)+' and var_method is shot, changing var_method to jackknife')
        var_method = 'jackknife'
    elif npatch==1 and var_method=='jackknife':
        print('because npatch=1 and var_method is set to jackknife, changing var_method to shot')
        var_method = 'shot'

    for i in range(nmaps):
        if verbose:
            print('i = ', i)

        load_srcmap = skymaps[i]

        skymap = load_srcmap.copy()
        
        if mask_list is not None:
            xind_cat, yind_cat, ks, weights = get_treecorr_catalog_vals(xind, yind, skymap, mask=mask_list[i], prefilter=prefilter)
            
        else:
            xind_cat, yind_cat, ks, weights = get_treecorr_catalog_vals(xind, yind, skymap, mask=mask, prefilter=prefilter)
            
        print('sum of weights is ', np.sum(weights))

        cat = treecorr.Catalog(ra=xind_cat, dec=yind_cat, k=ks, w=weights, ra_units=pix_units, dec_units=pix_units, npatch=npatch)
        
        if binning=='linear':
            kk = treecorr.KKCorrelation(min_sep=thetamin, max_sep=thetamax, bin_type='Linear', nbins=nlinbin, sep_units=pix_units, bin_slop=bin_slop)
        elif binning=='log':
            kk = treecorr.KKCorrelation(min_sep=thetamin, max_sep=thetamax, bin_size=thetabin_size, sep_units=pix_units, var_method=var_method, bin_slop=bin_slop)

        kk.process(cat, metric='Euclidean')
        
        if verbose:
            print('var method is ', kk.var_method)
            print('number of pairs used is ', kk.npairs)
            print('kk.r_nom is ', kk.rnom*3600)
            print('kk.meanr is ', kk.meanr*3600)

        npair_mask = (kk.npairs > 0.)

        if plot:
            if binning=='linear':
                plot_xscale=None
            else:
                plot_xscale='log'
            f = plot_wtheta(kk.rnom[npair_mask], kk.xi[npair_mask], kk.varxi[npair_mask], skymap, pix_units=pix_units, mask=mask, return_fig=True, plot_xscale=plot_xscale, title=plot_title)
            fs.append(f)

        wthetas.append(kk.xi[npair_mask])
        varwthetas.append(kk.varxi[npair_mask])
        if return_cov:
            covs.append(kk.cov[npair_mask])
            

    if plot:
        if return_cov:
            return wthetas, varwthetas, kk.rnom[npair_mask], covs, fs
        else:
            return wthetas, varwthetas, kk.rnom[npair_mask], fs
    else:
        if return_cov:
            return wthetas, varwthetas, kk.rnom[npair_mask], covs
        else:
            return wthetas, varwthetas, kk.rnom[npair_mask]


def compute_2pcf_2d(skymaps=None, mask=None, nmaps=20, nbins=64, imdim=1024, pixel_size=7., thetamax=2., \
                   with_noise=True, plot=True, datapath='/Users/richardfeder/Documents/ciber2/ciber/data/', tail_name='mmin=18.4_mmax=25', brute=False, \
                    bin_slop=0.05):
        
    
    # for some reason, TreeCorr only works with radians as the units, so this always assumes radians unlike the 1d version

    if skymaps is not None:
        imdim = skymaps[0].shape[0]
        
        if nmaps > len(skymaps):
            nmaps = len(skymaps)
            
    fs, wthetas_2d, varwthetas_2d, covs =[[] for x in range(4)]
    deg_per_pix = (pixel_size/3600.) # arcseconds to degrees
    deg_per_rad = np.pi/180.    
    
    xind = np.indices((imdim,imdim))[0].ravel()*deg_per_pix*deg_per_rad
    yind = np.indices((imdim,imdim))[1].ravel()*deg_per_pix*deg_per_rad
    
    print('min/max xind:', np.min(xind), np.max(xind))
    print('thetamax:', imdim*deg_per_pix*deg_per_rad)
    
    for i in range(nmaps):
        
        if skymaps is None:
            load_srcmap = load_ciber_srcmap(i, with_noise=with_noise, datapath=datapath, tail_name=tail_name)
        else:
            print('skymaps is not None')
            load_srcmap = skymaps[i]

        skymap = load_srcmap - np.mean(load_srcmap)       
        
        
        xind_cat, yind_cat, ks, weights = get_treecorr_catalog_vals(xind, yind, skymap, mask=mask, prefilter=False)
        
        cat = treecorr.Catalog(x=xind_cat, y=yind_cat, k=ks, w=weights, npatch=1)
        if brute:
            kk = treecorr.KKCorrelation(nbins=nbins, bin_type='TwoD', max_sep=imdim*deg_per_pix*deg_per_rad, sep_units='rad', brute=True)
        else:
            kk = treecorr.KKCorrelation(nbins=nbins, bin_type='TwoD', max_sep=imdim*deg_per_pix*deg_per_rad, sep_units='rad', bin_slop=bin_slop)
        kk.process(cat, metric='Euclidean')
        print('bin slop is ', kk.bin_slop)
        
        if plot:
            plt.figure()
            plt.title('npairs, 256x256 image, 256^2 bins')
            plt.imshow(kk.npairs, vmin=0, vmax=10)
            plt.colorbar()
            plt.show()
            
            plt.figure()
            plt.hist(kk.npairs.ravel(), bins=np.linspace(-1, 10, 10))
            plt.show()
            
            plt.figure()
            plt.imshow(kk.xi, vmin=np.percentile(kk.xi, 1), vmax=np.percentile(kk.xi, 99))
            plt.colorbar()
            plt.show()
        
        wthetas_2d.append(kk.xi)
        
    wthetas_2d = np.array(wthetas_2d)
    
    if plot:
        f = plot_w_thetax_thetay(wthetas_2d)
        return wthetas_2d, f
    
    return wthetas_2d
	

def compute_ensemble_offsets(mask=None, skymaps=None, mode='white', nmaps=10, imdim=1024, npatch=10, verbose=False, \
						 plot_indiv=False, with_noise=False, prefilter=False, var_method='jackknife', datapath='/Users/richardfeder/Documents/ciber2/ciber/data/', tail_name='mmin=18.4_mmax=25'):
	
	frac_delta_xis = []

	if skymaps is not None and nmaps > len(skymaps):
		nmaps = len(skymaps)
	
	if mode=='white' and skymaps is None:
		skymaps = [np.random.normal(size=(imdim, imdim)) for i in range(nmaps)]
		
	if mask is None:
		mask = np.ones(size=(imdim, imdim))
	
	masked_xis, masked_varxis, bins = compute_2pcf(skymaps, mask=mask, nmaps=nmaps, npatch=npatch, plot=plot_indiv, datapath=datapath, verbose=verbose, with_noise=with_noise, var_method=var_method, tail_name=tail_name, prefilter=prefilter)
	print('done with masked, now unmasked')
	unmasked_xis, unmasked_varxis, bins = compute_2pcf(skymaps, nmaps=nmaps, npatch=npatch, plot=plot_indiv, datapath=datapath, verbose=verbose, with_noise=with_noise, var_method=var_method, tail_name=tail_name, prefilter=prefilter)
	
	for i in range(nmaps):
		
		frac_delta_xis.append((unmasked_xis[i]-masked_xis[i])/unmasked_xis[i])
		

	return bins, frac_delta_xis, masked_xis, unmasked_xis, masked_varxis, unmasked_varxis
		

