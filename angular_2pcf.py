import treecorr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

def compute_2pcf(skymaps=None, mask=None, nmaps=50, npatch=10, thetamax=2., thetabin_size=0.3, pixel_size=7., plot=False, \
				verbose=False, datapath='/Users/richardfeder/Documents/ciber2/ciber/data/', imdim=1024, return_cov=False, \
				var_method='sample'):

	xis, varxis, covs = [], [], []
	deg_per_pix = (pixel_size/3600.) # arcseconds to degrees
	

	xind = np.indices((imdim,imdim))[0].ravel()*deg_per_pix
	yind = np.indices((imdim,imdim))[1].ravel()*deg_per_pix
	
	
	for i in range(nmaps):
		if verbose:
			print('i = ', i)
		
		if skymaps is None:
			load_srcmap = np.load(datapath+'/ciber_mock_'+str(i)+'mmin=18.4_mmax=25.npz')['srcmap']
			# load_srcmap = np.load('data/nobright_ciber_mock_sims/ciber_mock_'+str(i)+'mmin=18.4_mmax=25.npz')['srcmap']
		else:
			load_srcmap = skymaps[i]
		
		skymap = load_srcmap - np.mean(load_srcmap)
		
		if mask is not None:
			weights = mask.ravel()
		else:
			weights = None
			
		if npatch > 1:
			if verbose:
				print('number of patches is ', npatch)
			cat = treecorr.Catalog(ra=xind, dec=yind, k=skymap.ravel(), w=weights, ra_units='deg', dec_units='deg', npatch=npatch)
			kk = treecorr.KKCorrelation(min_sep= 2*deg_per_pix, max_sep=thetamax, bin_size=thetabin_size, sep_units='deg', var_method=var_method)

		else:
			cat = treecorr.Catalog(ra=xind, dec=yind, k=skymap.ravel(), w=weights, ra_units='deg', dec_units='deg')
			kk = treecorr.KKCorrelation(min_sep= 2*deg_per_pix, max_sep=thetamax, bin_size=thetabin_size, sep_units='deg')
		
		kk.process(cat, metric='Euclidean')
		if verbose:
			print('var method is ', kk.var_method)


		
		bins = 10**(np.linspace(np.log10(2.*deg_per_pix), np.log10(thetamax), len(kk.xi)))
			
		if plot:
			plt.figure()
			plt.errorbar(10**(np.linspace(np.log10(2.*deg_per_pix), np.log10(2.), len(kk.xi))), kk.xi, yerr=np.sqrt(kk.varxi), marker='.')
			plt.xlabel('$\\theta$ (deg)', fontsize=16)
			plt.ylabel('$\\xi(r)$ (arbitrary units)', fontsize=16)
			plt.xscale('log')
			plt.show()
			
		xis.append(kk.xi)
		varxis.append(kk.varxi)
		if return_cov:
			covs.append(kk.cov)
		
	if len(xis)==1:

		if return_cov:
			return kk.xi, kk.varxi, bins, kk.cov
		
		return kk.xi, kk.varxi, bins
		
	if return_cov:
		return xis, varxis, bins, covs

	return xis, varxis, bins
	

def compute_ensemble_offsets(mask=None, skymaps=None, mode='white', nmaps=10, imdim=1024, npatch=10, verbose=False, \
						 plot_indiv=False, datapath='/Users/richardfeder/Documents/ciber2/ciber/data/'):
	
	frac_delta_xis = []
	
	
	if mode=='white' and skymaps is None:
		skymaps = [np.random.normal(size=(imdim, imdim)) for i in range(nmaps)]
		
	if mask is None:
		mask = np.ones(size=(imdim, imdim))
	
	masked_xis, masked_varxis, bins = compute_2pcf(skymaps, mask=mask, nmaps=nmaps, npatch=npatch, plot=plot_indiv, datapath=datapath, verbose=verbose)
	print('done with masked, now unmasked')
	unmasked_xis, unmasked_varxis, bins = compute_2pcf(skymaps, nmaps=nmaps, npatch=npatch, plot=plot_indiv, datapath=datapath, verbose=verbose)
	
	for i in range(nmaps):
		
		frac_delta_xis.append((unmasked_xis[i]-masked_xis[i])/unmasked_xis[i])
		

	return bins, frac_delta_xis, masked_xis, unmasked_xis, masked_varxis, unmasked_varxis
		
