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

