import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
# import seaborn as sns
# sns.set()


figure_directory = '/Users/richardfeder/Documents/caltech/ciber2/figures/'

def plot_radavg_xspectrum(rbins, radprofs=[], raderrs=None, labels=[], \
                          lmin=90., save=False, snspalette=None, pdf_or_png='png', \
                         image_dim=1024, mode='cross', zbounds=None, shotnoise=True, \
                         add_shot_noise=None, Ntot=160000, sn_npoints=10, titlestring=None, \
                          ylims=None):
    
    steradperpixel = ((np.pi/lmin)/image_dim)**2 

    fig = plt.figure(figsize=(8,6))
    xvals = rbins*lmin

    yerr = None # this gets changed from None to actual errors if they are fed in
    if mode=='cross':
        if titlestring is None:
            titlestring = 'Cross Spectrum'
        yvals = np.array([(rbins*lmin)**2*prof/(2*np.pi) for prof in radprofs])
        ylabel = '$\\ell(\\ell+1) C_{\\ell}/2\\pi$'
        if raderrs is not None:
            yerr = np.array([(rbins*lmin)**2*raderr/(2*np.pi) for raderr in raderrs]) 
    elif mode=='auto':
        if titlestring is None:
            titlestring = 'Auto Spectrum'
        yvals = np.array([np.sqrt((rbins*lmin)**2*prof/(2*np.pi)) for prof in radprofs])
        ylabel = '($\\ell(\\ell+1) C_{\\ell}/2\\pi)^{1/2}$'
        if raderrs is not None:
            yerr = np.array([np.sqrt((rbins*lmin)**2/(2*np.pi))*raderr/(2*np.sqrt(prof)) for raderr in raderrs]) # I don't think this is right, modify at some point
            print('yerr:', yerr)
            mask = yerr > yvals
            
            print(mask)
            yerr2 = np.array(yerr)
            yerr2[yerr >= yvals] = yvals[yerr>=yvals]*0.99999


    if zbounds is not None:
        titlestring +=', '+str(zmin)+'$<z<$'+str(zmax)
        
    plt.title(titlestring, fontsize=16)
    ax = plt.gca()
    for i, prof in enumerate(radprofs):
        color = next(ax._get_lines.prop_cycler)['color']

        if shotnoise is not None:
            if shotnoise[i]:
                if mode=='auto':
                    shot_noise_fit = np.poly1d(np.polyfit(np.log10(xvals[-sn_npoints:]), np.log10(np.sqrt(xvals[-sn_npoints:]**2*prof[-sn_npoints:]/(2*np.pi))), 1))
                else:
                    shot_noise_fit = np.poly1d(np.polyfit(np.log10(xvals[-sn_npoints:]), np.log10(xvals[-sn_npoints:]**2*prof[-sn_npoints:]/(2*np.pi)), 1))
                sn_vals = 10**(shot_noise_fit(np.log10(xvals)))
                if mode=='auto':
                    sn_level = 2*np.pi*sn_vals[0]**2/lmin**2
                elif mode=='cross':
                    sn_level = 2*np.pi*sn_vals[0]/lmin**2 
                print('sn level:', sn_level)
                if i==0:
                    label='Best fit Poisson fluctuations'
                else:
                    label=None
                plt.plot(xvals, sn_vals, label=label,linestyle='dashed', color=color)

            if add_shot_noise is not None:
                if add_shot_noise[i]:
                    sn = surface_area*Ntot*steradperpixel/(image_dim**2)
                    print(surface_area, sn, Ntot)
                    yvals[i] += np.sqrt((rbin*lmin)**2*(surface_area*Ntot*steradperpixel/(image_dim**2))/(2*np.pi))
        
        plt.errorbar(xvals, yvals[i], yerr=yerr[i], marker='.', label=labels[i], color=color)
    if ylims is not None:
        plt.ylim(ylims[0], ylims[1])
    plt.legend(loc=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$\\ell$', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    if save:
        plt.savefig(figure_directory+'radavg_xspectrum.'+pdf_or_png, bbox_inches='tight')
    plt.show()

    
    return fig, yvals, yerr


def plot_2d_xspectrum(xspec, minpercentile=5, maxpercentile=95, save=False, pdf_or_png='png'):
	plt.figure()
	ax = plt.subplot(1,1,1)
	ax.grid(False)
	plt.title('2D Power Spectrum')
	plt.imshow(xspec, vmin=np.percentile(xspec, minpercentile), vmax=np.percentile(xspec, maxpercentile))
	plt.tick_params( labelbottom='off', labelleft='off')
	plt.xlabel('$1/\\theta_x$', fontsize=14)
	plt.ylabel('$1/\\theta_y$', fontsize=14)
	plt.colorbar()
	if save:
		plt.savefig(figure_directory+'2d_xspectrum.'+pdf_or_png, bbox_inches='tight')
	plt.show()

def plot_beam_correction(rb, beam_correction, lmin=90.):
    fig = plt.figure()
    plt.plot(rb*lmin, beam_correction, marker='.')
    plt.xlabel('$\\ell$', fontsize=14)
    plt.ylabel('$B_{\\ell}$', fontsize=14)
    plt.title('Beam Correction', fontsize=16)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    
    return fig


def plot_catalog_properties(cat, zidx=2, mapp_idx=3, mabs_idx=4, mhalo_idx=5, rvir_idx=6, Inu_idx=7, nbin=10, save=False):

    plt.figure(figsize=(15, 10))
    plt.suptitle('Mock Galaxy Catalog', fontsize=16, y=1.02)
    plt.subplot(2,3,1)
    plt.hist(cat[:,zidx], histtype='step', bins=nbin)
    plt.xlabel('Redshift', fontsize=14)
    plt.subplot(2,3,2)

    plt.hist(cat[:,mapp_idx], histtype='step', bins=nbin)
    plt.xlabel('Apparent Magnitude', fontsize=14)

    plt.subplot(2,3,3)
    plt.hist(cat[:,mabs_idx], histtype='step', bins=nbin)
    plt.xlabel('Absolute Magnitude', fontsize=14)

    plt.subplot(2,3,4)
    plt.hist(np.log10(cat[:,mhalo_idx]), histtype='step', bins=nbin)
    plt.xlabel('$\\log_{10}(M_{halo}/M_{\\odot})$', fontsize=14)
    plt.yscale('log')

    plt.subplot(2,3,5)
    plt.hist(cat[:,rvir_idx]*1000, histtype='step', bins=np.linspace(0, 0.15, 20)*1000)
    plt.yscale('log')
    plt.xlabel('$R_{vir}$ (kpc)', fontsize=14)

    plt.subplot(2,3,6)
    plt.hist(np.log10(cat[:,Inu_idx]), histtype='step', bins=nbin)
    plt.xlabel('$\\log_{10}(I_{\\nu})$ (nW $m^{-2}sr^{-1}$)', fontsize=14)
    plt.tight_layout()
    if save:
        plt.savefig('../figures/mock_galaxy_catalog.png', bbox_inches='tight')
    plt.show()


def plot_mock_images(full, srcs, ihl=None, fullminp=1., fullmaxp=99.9, \
                    srcminp=1., srcmaxp=99.9, ihlminp=1., ihlmaxp=99.9, \
                    save=False, show=True, xmin=None, xmax=None, ymin=None, \
                    ymax=None):
    if ihl is None:
        fig = plt.figure(figsize=(10, 5))
        plt.subplot(1,2,1)
    else:
        fig = plt.figure(figsize=(15, 5))
        plt.subplot(1,3,1)
    
    plt.title('Full Map (with noise)', fontsize=14)
    plt.imshow(full, vmin=np.percentile(full, fullminp), vmax=np.percentile(full, fullmaxp))
    if xmin is not None:
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

    plt.colorbar()
    
    if ihl is None:
        plt.subplot(1,2,2)
    else:
        plt.subplot(1,3,2)
    
    plt.title('Galaxies', fontsize=14)
    plt.imshow(srcs, vmin=np.percentile(srcs, srcminp), vmax=np.percentile(srcs, srcmaxp))
    plt.colorbar()
    if xmin is not None:
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
    
    if ihl is not None:
        plt.subplot(1,3,3)
        plt.title('IHL ($f_{IHL}=0.1$)', fontsize=14)
        plt.imshow(ihl, vmin=np.percentile(ihl, ihlminp), vmax=np.percentile(ihl, ihlmaxp))
        plt.colorbar()
        if xmin is not None:
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
    if save:
        plt.savefig('../figures/mock_obs.png', bbox_inches='tight')
    if show:
        plt.show()
    
    return fig


