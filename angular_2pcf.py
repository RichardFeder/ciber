import treecorr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


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


def compute_2pcf(skymaps=None, mask=None, nmaps=50, binning='log',nlinbin=100, pix_units='deg',\
                      npatch=10, var_method='shot', return_cov=True, prefilter=False, verbose=False, plot=False,\
                     thetamax=2., thetabin_size=0.3, pixel_size=7., imdim=1024, plot_title='Read noise realization',\
                    datapath='/Users/richardfeder/Documents/ciber2/ciber/data/', tail_name='mmin=18.4_mmax=25', with_noise=False):
    
    ''' Inputs:
    
        thetamax (deg): Maximum theta desired. If using radian pixel units this will get converted to radians in the code.
        nlinbin (int, default: 100): number of linear bins to use if doing linear binning
    '''

    
    fs = []
    wthetas, varwthetas, covs = [], [], []
    deg_per_pix = (pixel_size/3600.) # arcseconds to degrees
    thetamin = 2*deg_per_pix # minimum separation is twice the pixel size

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

        if skymaps is None:
            load_srcmap = load_ciber_srcmap(i, with_noise=with_noise, datapath=datapath, tail_name=tail_name)
        else:
            load_srcmap = skymaps[i]

        skymap = load_srcmap - np.mean(load_srcmap)
        
        
        xind_cat, yind_cat, ks, weights = get_treecorr_catalog_vals(xind, yind, skymap, mask=mask, prefilter=prefilter)
            
        cat = treecorr.Catalog(ra=xind_cat, dec=yind_cat, k=ks, w=weights, ra_units=pix_units, dec_units=pix_units, npatch=npatch)
        
        if binning=='linear':
            kk = treecorr.KKCorrelation(min_sep=thetamin, max_sep=thetamax, bin_type='Linear', nbins=nlinbin, sep_units=pix_units)
        elif binning=='log':
            kk = treecorr.KKCorrelation(min_sep=thetamin, max_sep=thetamax, bin_size=thetabin_size, sep_units=pix_units, var_method=var_method)

        kk.process(cat, metric='Euclidean')
        
        
        if verbose:
            print('var method is ', kk.var_method)
            print('number of pairs used is ', kk.npairs)
            print('kk.r_nom is ', kk.rnom)
            print('kk.meanr is ', kk.meanr)

            
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
		

def spline_wtheta(thetas, wtheta):
    
    xs = thetas
    ys = wtheta
    
    cs = CubicSpline(xs, ys)
    
    return cs

def density_simpsons(func, a, b, N):

    h = (b - a) / (N) 
    integral = func(a) + func(b)
    for i in np.arange(1, N, 2):
        add = func(a + i*h)
        integral += 4. * add
    for i in np.arange(2, N-1, 2):
        add = func(a + i*h)
        integral += 2 * add
    
    return (1/3)*h*integral

def make_integrand_ell(spline, ell):
    def integrand_func(theta):
        function = (theta * spline(theta) * np.sin(theta*ell) / (theta * ell))
        return function
    return integrand_func

def integrate_spline_cell(integrand, ell, N=10, a=2e-3, b=1e-2):
    return density_simpsons(integrand, a=a, b=b, N=N)

def wtheta_to_c_ell_hb(thetabins, wtheta, ells, N=1000, a=2e-3, b=1e-2):
    bins_rad = thetabins*np.pi/180.
    cs = spline_wtheta(bins_rad, wtheta)
    c_ell = []
    for i in ells:
        integrand = make_integrand_ell(cs, i)
        c_ell_i = 2*np.pi*integrate_spline_cell(integrand, ell=i, N = N, a=a, b=b)
        c_ell.append(c_ell_i)
        
    return c_ell


def wtheta_to_cell(bins, wthetas, multipole_bins=None, thetamax=2., pixsize=7., ndim=2, N = 2000, h = 0.003):
    
    c_ells = []
    
    deg_per_pix=pixsize/3600. # pixsize in arcseconds convert to deg
    
    # get thetas, fine_thetas into units of radians across angular range
    thetas = bins*(np.pi/180.)
    ells = np.pi/thetas
        
    ft = SymmetricFourierTransform(ndim = ndim, N = N, h = h)
    
    for i in range(len(wthetas)):
    
        spline_w_theta = interpolate.InterpolatedUnivariateSpline(thetas, wthetas[i])    

        plt.figure()
        plt.plot(thetas, spline_w_theta(thetas))
        finet = np.linspace(np.min(thetas), np.max(thetas), 1000)
        plt.plot(finet, spline_w_theta(finet))
        plt.show()
        
        g = lambda theta: spline_w_theta(theta)

        if multipole_bins is not None:
            c_ell = ft.transform(g ,multipole_bins, ret_err=False, inverse=False)
        else:
            c_ell = ft.transform(g, ells, ret_err=False, inverse=False)
            
        c_ells.append(np.abs(c_ell))
                
    if multipole_bins is None:
        return ells, c_ells
        
    return multipole_bins, c_ells

