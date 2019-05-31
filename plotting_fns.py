import matplotlib
import matplotlib.pyplot as plt
import numpy as np


figure_directory = '/Users/richardfeder/Documents/caltech/ciber2/figures/'

def plot_radavg_xspectrum(rbins, radprof, raderr, radprof2=None, raderr2=None, lmin=90., save=False):
	plt.figure()
	plt.title('Cross Spectrum', fontsize=16)
	plt.errorbar(rbins*lmin, (rbins*lmin)**2*radprof/(2*np.pi), color='k', yerr=(rbins*lmin)**2*raderr/(2*np.pi), marker='.')
	if radprof2 is not None:
		plt.errorbar(rbins*lmin, (rbins*lmin)**2*radprof2/(2*np.pi), color='g', yerr=(rbins*lmin)**2*raderr2/(2*np.pi), marker='.')
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


