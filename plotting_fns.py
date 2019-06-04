import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import seaborn as sns
sns.set()



figure_directory = '/Users/richardfeder/Documents/caltech/ciber2/figures/'

def plot_radavg_xspectrum(rbins, radprofs=[], raderrs=None, labels=[], lmin=90., save=False, snspalette=None):
	if snspalette is None:
		sns.set_palette(sns.color_palette("Blues"))
	else:
		sns.set_palette(sns.color_palette(snspalette))
	plt.figure()
	plt.title('Cross Spectrum', fontsize=16)
	for i, prof in enumerate(radprofs):
		if raderrs is not None:
			plt.errorbar(rbins*lmin, (rbins*lmin)**2*prof/(2*np.pi), yerr=(rbins*lmin)**2*raderrs[i]/(2*np.pi), marker='.', label=labels[i])
		else:
			plt.plot(rbins*lmin, (rbins*lmin)**2*prof/(2*np.pi), marker='.', label=labels[i])

	plt.legend()
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel('$\\ell$', fontsize=14)
	plt.ylabel('$\\ell(\\ell+1) C_{\\ell}/2\\pi$  $(nW m^{-2} sr^{-1})$', fontsize=14)
	if save:
		plt.savefig(figure_directory+'radavg_xspectrum.pdf', bbox_inches='tight')
	plt.show()


def plot_2d_xspectrum(xspec, minpercentile=5, maxpercentile=95, save=False):
	plt.figure()
	plt.title('2D Power Spectrum')
	plt.imshow(xspec, vmin=np.percentile(xspec, minpercentile), vmax=np.percentile(xspec, maxpercentile))
	plt.tick_params( labelbottom='off', labelleft='off')
	plt.xlabel('$1/\\theta_x$', fontsize=14)
	plt.ylabel('$1/\\theta_y$', fontsize=14)
	plt.colorbar()
	if save:
		plt.savefig(figure_directory+'2d_xspectrum.pdf', bbox_inches='tight')
	plt.show()

def cross_correlate_galcat_ciber(cibermap, galaxy_catalog, m_min=14, m_max=20, band='J', ihl_frac=0.0, magidx=5):
    # convert galaxy catalog to binary map
    gal_map = make_galaxy_binary_map(cat, cibermap, m_min=m_min, m_max=m_max, magidx=magidx)
    xcorr = compute_cross_spectrum(cibermap, gal_map)
    rbins, radprof, radstd = azimuthalAverage(xcorr)
    return rbins, radprof, radstd


def xcorr_varying_ihl(ihl_min_frac=0.0, ihl_max_frac=0.5, nbins=10, nsrc=100, m_min=14, m_max=20):
    radprofs = []
    radstds = []
    ihl_range = np.linspace(ihl_min_frac, ihl_max_frac, nbins)
    for i, ihlfrac in enumerate(ihl_range):
        if i==0:
            full, srcs, noise, cat = cmock.make_ciber_map(ifield, m_min, m_max, band=inst, nsrc=nsrc, ihl_frac=ihlfrac)
            gal_map = make_galaxy_binary_map(cat, full, m_min, m_max=20, magidx=5) # cut off galaxy catalog at 20th mag
        else:
            full, srcs, noise, ihl, cat = cmock.make_ciber_map(ifield, m_min, m_max, mock_cat=cat, band=inst, nsrc=nsrc, ihl_frac=ihlfrac)
         
        xcorr = compute_cross_spectrum(full, gal_map)
        rb, radprof, radstd = azimuthalAverage(xcorr)
        radprofs.append(radprof)
        radstds.append(radstd)
        
    return ihl_range, rb, radprofs, radstds


def xcorr_varying_galcat_completeness(ihl_frac=0.1, compmin=18, compmax=22, nbin=10, nsrc=100):
    radprofs, radstds = [], []
    comp_range = np.linspace(compmin, compmax, nbin)
    full, srcs, noise, ihl, gal_cat = cmock.make_ciber_map(ifield, m_min, 25, band=inst, nsrc=nsrc, ihl_frac=ihl_frac)

    for i, comp in enumerate(comp_range):
        gal_map = make_galaxy_binary_map(gal_cat, full, m_min, m_max=comp)        
        xcorr = compute_cross_spectrum(full, gal_map)
        rb, radprof, radstd = azimuthalAverage(xcorr)
        radprofs.append(radprof)
        radstds.append(radstd)
        
    return comp_range, rb, radprofs, radstds


