import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
# import seaborn as sns
# sns.set()
from PIL import Image
# from Pillow import Image
import glob



figure_directory = '/Users/richardfeder/Documents/caltech/ciber2/figures/'

def plot_radavg_xspectrum(rbins, radprofs=[], raderrs=None, labels=[], \
						  lmin=90., save=False, snspalette=None, pdf_or_png='png', \
						 image_dim=1024, mode='cross', zbounds=None, shotnoise=None, \
						 add_shot_noise=None, Ntot=160000, sn_npoints=10, titlestring=None, \
						  ylims=None):
	
	steradperpixel = ((np.pi/lmin)/image_dim)**2 

	f = plt.figure(figsize=(8,6))
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

	
	return f, yvals, yerr


def plot_2d_xspectrum(xspec, minpercentile=5, maxpercentile=95, save=False, pdf_or_png='png'):
	f = plt.figure()
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
	return f

def plot_beam_correction(rb, beam_correction, lmin=90.):
	f = plt.figure()
	plt.plot(rb*lmin, beam_correction, marker='.')
	plt.xlabel('$\\ell$', fontsize=14)
	plt.ylabel('$B_{\\ell}$', fontsize=14)
	plt.title('Beam Correction', fontsize=16)
	plt.xscale('log')
	plt.yscale('log')
	plt.show()
	
	return f


def plot_catalog_properties(cat, zidx=2, mapp_idx=3, mabs_idx=4, mhalo_idx=5, rvir_idx=6, Inu_idx=7, nbin=10, save=False):

	f = plt.figure(figsize=(15, 10))
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
	return f

def plot_cumulative_numbercounts(cat, mag_idx, field=None, df=True, m_min=18, m_max=30, \
								nbin=100, label=None, pdf_or_png='pdf', save=False, show=True):
	f = plt.figure()
	title = 'Cumulative number count distribution'
	if field is not None:
		title += ' -- '+str(field)
	plt.title(title)
	magspace = np.linspace(m_min, m_max, nbin)
	if df:
		cdf_nm = np.array([len(cat.loc[cat[mag_idx]<x]) for x in magspace])
	else:
		cdf_nm = np.array([len(cat[cat[mag_idx] < x]) for x in magspace])
	plt.plot(magspace, cdf_nm, label=label)
	plt.yscale('log')
	plt.ylabel('N[r < magnitude]', fontsize=14)
	plt.xlabel('magnitude', fontsize=14)
	plt.legend()
	if save and field is not None:
		plt.savefig('../figures/catalog_map_figures/number_cts_'+str(field)+'_catalog.'+pdf_or_png, bbox_inches='tight')
	if show:
		plt.show()
		
	return f

def plot_dm_powerspec(show=True):
    
    mass_function = MassFunction(z=0., dlog10m=0.02)
    
    f = plt.figure()
    plt.title('Halo Model Dark Matter Power Spectrum, z=0')
    # plt.plot(mass_function.k, mass_function.nonlinear_delta_k, label='halofit')
    plt.plot(mass_function.k, mass_function.nonlinear_power, label='halofit')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.ylabel('$P(k)$ [$h^{-3}$ $Mpc^3$]', fontsize=14)
    plt.xlabel('$k$ [h $Mpc^{-1}$]', fontsize=14)
    plt.ylim(1e-4, 1e5)
    plt.xlim(1e-3, 1e3)
    if show:
        plt.show()
    
    return f

def plot_mock_images(full, srcs, ihl=None, fullminp=1., fullmaxp=99.9, \
					srcminp=1., srcmaxp=99.9, ihlminp=1., ihlmaxp=99.9, \
					save=False, show=True, xmin=None, xmax=None, ymin=None, \
					ymax=None):
	if ihl is None:
		f = plt.figure(figsize=(10, 5))
		plt.subplot(1,2,1)
	else:
		f = plt.figure(figsize=(15, 5))
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
	
	return f

def plot_cross_terms_knox(list_of_crossterms, ells, mlim=24, save=False, show=True, title=None, titlesize=18):
#     colors = ['b', 'C1', 'green']
	colors = ['b', 'green', 'black']
	linestyles = [(0, (5, 10)), 'solid', 'dotted', 'dashed', 'dashdot']
	labels = np.array(['$(C_{\\ell}^{cg})^2$', '$C_{\\ell}^c C_{\ell}^g$', '$C_{\\ell}^c n_g^{-1}$', '$C_{\\ell}^{noise}C_{\\ell}^g$', '$C_{\\ell}^{noise}n_g^{-1}$'])

	f = plt.figure(figsize=(10, 8))
	if title is not None:
		plt.title(title, fontsize=titlesize)
	else:
		plt.title('Blue --> $0<z<0.33$, Orange --> $0.33<z<0.67$, Green --> $0.67<z<1.0$', fontsize=14)
	for i, crossterms in enumerate(list_of_crossterms):
		for j, crossterm in enumerate(crossterms):
			if i==2:
				plt.plot(ells, crossterm,  linestyle=linestyles[j], color=colors[i], label=labels[j])

			else:
#                 plt.plot(ells, crossterm,  linestyle=linestyles[j], color=colors[i])
				continue
	plt.legend(fontsize=15, loc=1)
	plt.yscale('log')
	plt.xscale('log')
	plt.xlabel('$\ell$', fontsize=16)
	plt.ylabel('Noise Level ($nW^2$ $m^{-4}$)', fontsize=18)
	plt.ylim(1e-14, 2e-9)
	if save:
#         plt.savefig('../figures/power_spectra/cross_noise_term_contributions_ciber_mlim='+str(mlim)+'_0.0<z<0.33.png', bbox_inches='tight')
#         plt.savefig('../figures/power_spectra/cross_noise_term_contributions_ciber_mlim='+str(mlim)+'_0.33<z<0.66.png', bbox_inches='tight')
		plt.savefig('../figures/power_spectra/cross_noise_term_contributions_ciber_mlim='+str(mlim)+'_0.67<z<1.0.png', bbox_inches='tight')
	if show:
		plt.show()
		
	return f

def plot_limber_project_powerspec(show=True, ellsquared=False):
    f = plt.figure()
    plt.title('Limber projected angular DM power spectrum', fontsize=14)
    if ellsquared:
        plt.loglog(ells, ells*(ells+1)*cl/(2*np.pi))
        plt.ylabel('$\\ell(\\ell+1) C_{\\ell}$', fontsize=16)
    else:
        plt.loglog(ells, cl)
        plt.ylabel('$C_{\\ell}$', fontsize=16)

    plt.xlabel('$\\ell$', fontsize=16)
    
    if show:
        plt.show()
    
    return f

def plot_snr_vs_multipole(snr_sq, ells, save=False, show=True, title=None, titlesize=18, mlim=24):

#     zlabels = ['Knox formula ($0<z<0.33$)', 'Knox formula ($0.33<z<0.67$)', 'Knox formula ($0.67<z<1.0$)']
	zlabels = ['0<z<0.2', '0.2<z<0.4', '0.4<z<0.6', '0.6<z<0.8', '0.8<z<1.0']
	d_ells = ells[1:]-ells[:-1]
	colors = ['b', 'green', 'black']
	plt.figure(figsize=(8, 6))
	if title is not None:
		plt.title(title, fontsize=titlesize)
	else:
		plt.title('CIBER 1.1$\mu$m-Galaxy Cross Spectrum, 50s frame', fontsize=16)
	
	for i, prof in enumerate(snr_sq):
		print('here')
#         plt.plot(ells[:-1], np.sqrt(d_ells*snr_sq[i][:-1]), color=colors[i], marker='.', label=zlabels[i]+' $(m_{AB}<$'+str(mlim)+')')
		plt.plot(ells[:-1], np.sqrt(d_ells*snr_sq[i][:-1]),  marker='.', label=zlabels[i]+' $(m_{AB}<$'+str(mlim)+')')

		
	plt.ylabel('SNR per bin', fontsize=18)
	plt.legend()
	plt.yscale('log')
	plt.xscale('log')
	plt.xlabel('$\ell$', fontsize=18)
	if save:
		plt.savefig('../figures/power_spectra/cross_spectrum_snr_comparison_mlim='+str(mlim)+'.png', bbox_inches='tight')
	if show:
		plt.show()



def plot_bkg_galaxy_contribution(z_bins, sum_rates, title=True, show=True):
    f = plt.figure()
    if title:
        plt.title('Background galaxy contribution', fontsize=14)
    plt.plot(np.array(z_bins)[:-1], sum_rates[0], marker='.',  label='$m_{AB} > 27$', markersize=14)
    plt.plot(np.array(z_bins)[:-1], sum_rates[1], marker='.',  label='$m_{AB} < 27$', markersize=14)
    plt.legend(fontsize='large')
    plt.xlabel('$z$', fontsize=14)
    plt.ylabel('$\\nu I_{\\nu}$ ($nW/m^2/sr$)', fontsize=14)
    plt.ylim(-0.1, 3.0)
    if show:
        plt.show()
    return f

def plot_flux_production_rates(z_bins, all_flux_prod_rates, show=True):
    f = plt.figure()
    for i in xrange(len(z_bins)-1):
        zrange = np.linspace(z_bins[i], z_bins[i+1], nbin)
        plt.plot(zrange, zrange*all_flux_prod_rates[i])
    plt.ylabel('$(dF/dz)_{\\nu}$ ($nW/m^2/sr$)', fontsize=14)
    plt.xlabel('$z$', fontsize=14)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(5e-4, 4)
    
    if show:
        plt.show()
    
    return f

def plot_halomass_virial_radius(halo_masses, virial_radii, show=True):
    f = plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1)
    plt.hist(np.log10(halo_masses), bins=30)
    plt.yscale('symlog')
    plt.xlabel('$\log_{10}M_{halo}$ [$M_{\\odot}$]', fontsize=16)
    plt.ylabel('$N$', fontsize=16)
    plt.subplot(1,2,2)
    plt.hist(np.log10(virial_radii), bins=30)
    plt.yscale('symlog')
    plt.xlabel('$\\log_{10}R_{vir}$ [Mpc]', fontsize=16)
    plt.ylabel('$N$', fontsize=16)
    plt.savefig('../figures/halo_mass_virial_radius_mock_catalog.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()
        
    return f


def plot_student_t_dz_sampling(dzs, t_draws, dzs_tails, show=True):
    f = plt.figure(figsize=(10, 4))
    plt.subplot(1,3,1)
    plt.hist(dzs, bins=50, edgecolor='k')
    plt.title('$\\delta z$ (from KDE)', fontsize=16)
    plt.ylabel('$N_{src}$', fontsize=16)
    plt.subplot(1,3,2)
    plt.title('Student-t ($\\nu=$'+str(nu)+')', fontsize=16)
    plt.hist(t_draws, bins=50,  edgecolor='k')
    plt.yscale('symlog')
    plt.subplot(1,3,3)
    plt.title('Sampled $\\Delta z$', fontsize=16)
    plt.hist(dzs_tails, bins=50, edgecolor='k')
    plt.yscale('symlog')
    if show:
        plt.show()
    return f

def gaussian_vs_student_ts(show=True):
    nus = np.arange(1, 5)
    sigma = 1.0
    nsamp = 1000000
    f = plt.figure()
    binz = np.linspace(-8, 8, 50)
    for nu in nus:
        x = np.random.standard_t(nu, nsamp)*sigma
        plt.hist(x, bins=binz, histtype='step', density=True, label='Student-t ($\\nu=$'+str(nu)+')')
    plt.hist(np.random.normal(scale=sigma, size=nsamp),color='royalblue', bins=binz, histtype='step', density=True, label='Gaussian')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.ylabel('Probability density')
    if show:
        plt.show()
    return f

def convert_pngs_to_gif(filenames, gifdir='../../M_ll_correction/', name='mkk', duration=1000, loop=0):

    # Create the frames
    frames = []
    for i in range(len(filenames)):
        new_frame = Image.open(gifdir+filenames[i])
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(gifdir+name+'.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=duration, loop=loop)


def plot_ensemble_offset_stats(thetas, masked_xis, unmasked_xis, masked_varxi, unmasked_varxi, return_fig=True, logscale=True, \
                              suptitle='Mock CIBER source map (no noise), SWIRE flight mask (50 realizations)'):

    delta_xis = np.array([masked_xis[i]-unmasked_xis[i] for i in range(len(masked_xis))])


    print(delta_xis.shape)
    f = plt.figure(figsize=(15, 5))
    plt.suptitle(suptitle, fontsize=20, y=1.04)
    plt.subplot(1,4,2)

    for i in range(len(masked_varxi)):
        if i==0:
            masked_label = 'Masked'
        else:
            masked_label = None

        plt.errorbar(thetas, np.abs(masked_xis[i]), yerr=np.sqrt(masked_varxi[i]), capsize=5, marker='.', alpha=0.05, c='b', label=masked_label)
    plt.xscale('log')
    if logscale:
        plt.yscale('log', nonposy='clip')
    plt.xlabel('$\\theta$ [deg]', fontsize=18)
    plt.ylabel('$w(\\theta)$', fontsize=18)
    plt.legend(fontsize=16)
    plt.subplot(1,4,1)
    for i in range(len(masked_varxi)):
        if i==0:
            unmasked_label = 'True'
        else:
            unmasked_label = None

        plt.errorbar(thetas, np.abs(unmasked_xis[i]), yerr=np.sqrt(unmasked_varxi[i]), capsize=5, marker='.', alpha=0.05, c='r', label=unmasked_label)
    plt.xscale('log')
    if logscale:
        plt.yscale('log', nonposy='clip')
    plt.xlabel('$\\theta$ [deg]', fontsize=18)
    plt.ylabel('$w(\\theta)$', fontsize=18)

    plt.legend(fontsize=16)

    plt.subplot(1,4,3)
    plt.errorbar(thetas, np.mean(delta_xis, axis=0), yerr=np.std(delta_xis, axis=0),capsize=5,capthick=2, marker='.', alpha=0.8, label='masked')
    plt.xlabel('$\\theta$ [deg]', fontsize=18)
    plt.ylabel('$w(\\theta)_{masked} - w(\\theta)_{true}$', fontsize=18)
    plt.xscale('log')
    plt.subplot(1,4,4)
#     plt.plot(thetas, np.mean(delta_xis, axis=0)/np.std(unmasked_xis, axis=0), marker='.', alpha=0.8, label='Standard dev. of $w(\\theta)_{true}$ across realizations')
#     plt.plot(thetas, np.std(delta_xis, axis=0)/np.sqrt(np.mean(unmasked_varxi, axis=0)), marker='.', alpha=0.8, label='Mean of sample variance on $w(\\theta)$')
#     plt.plot(thetas, np.mean(delta_xis/np.sqrt(unmasked_varxi), axis=0), marker='.', alpha=0.8, label='Mean of sample variance on $w(\\theta)$')
    plt.plot(thetas, np.mean(delta_xis, axis=0)/np.sqrt(np.mean(unmasked_varxi, axis=0)), marker='.', alpha=0.8, label='Mean of sample variance on $w(\\theta)$')

    # plt.errorbar(thetas, np.mean(frac_deltaxi, axis=0), yerr=np.std(frac_deltaxi),capsize=5,capthick=2, marker='.', alpha=0.8, label='masked')
    plt.xlabel('$\\theta$ [deg]', fontsize=18)
    plt.ylabel('$\\frac{\\langle w(\\theta)_{masked} - w(\\theta)_{true} \\rangle}{\\langle \\sigma(w(\\theta)_{true})_{SV}\\rangle}$', fontsize=20)
    # plt.ylabel('$\\langle \\frac{w(\\theta)_{masked} - w(\\theta)_{true}}{\\sigma}{sample variance}}} \\rangle$', fontsize=18)

    plt.xscale('log')
    plt.ylim(-0.5, 0.5)
    plt.tight_layout()
    plt.show()
    
    if return_fig:
        return f

def plot_correlation_matrices_masked_unmasked(masked_corr, unmasked_corr, return_fig=True, show=True,\
                                              corr_lims=[-0.8, 1.0], dcorr_lims=[-0.2, 0.2]):


    f = plt.figure(figsize=(15, 5))
    plt.suptitle('Correlation Matrices', fontsize=20)
    plt.subplot(1,3,1)
    plt.title('Masked', fontsize=16)
    plt.imshow(masked_corr, origin='lower', vmin=corr_lims[0], vmax=corr_lims[1])
    plt.colorbar()

    plt.xticks(np.arange(0, masked_corr.shape[0], 2), [str(i) for i in np.arange(0, masked_corr.shape[0], 2)])
    plt.yticks(np.arange(0, masked_corr.shape[0], 2), [str(i) for i in np.arange(0, masked_corr.shape[0], 2)])
    plt.xlabel('Bin index')
    plt.ylabel('Bin index')
    plt.subplot(1,3,2)
    plt.title('Unmasked', fontsize=16)
    plt.imshow(unmasked_corr, origin='lower', vmin=corr_lims[0], vmax=corr_lims[1])

    plt.colorbar()
    plt.xticks(np.arange(0, masked_corr.shape[0], 2), [str(i) for i in np.arange(0, masked_corr.shape[0], 2)])
    plt.yticks(np.arange(0, masked_corr.shape[0], 2), [str(i) for i in np.arange(0, masked_corr.shape[0], 2)])
    plt.xlabel('Bin index')
    plt.ylabel('Bin index')
    plt.subplot(1,3,3)
    plt.title('Unmasked - Masked', fontsize=16)
    plt.imshow(unmasked_corr-masked_corr, origin='lower', vmin=dcorr_lims[0], vmax=dcorr_lims[1])

    plt.colorbar()
    plt.xticks(np.arange(0, masked_corr.shape[0], 2), [str(i) for i in np.arange(0, masked_corr.shape[0], 2)])
    plt.yticks(np.arange(0, masked_corr.shape[0], 2), [str(i) for i in np.arange(0, masked_corr.shape[0], 2)])

    plt.xlabel('Bin index')
    plt.ylabel('Bin index')
    plt.tight_layout()
    if show:
        plt.show()
    
    if return_fig:
        return f


def plot_diff_estimators_var_wtheta(vars_sv, vars_jn, vars_bs, return_fig=True, show=True,\
                                    title='Fractional std dev. of sample variance estimators ($n_{trial}=20$)'):

    f = plt.figure(figsize=(8,6))
    plt.title(title, fontsize=14)
    plt.plot(bins, np.std(np.array(vars_sv), axis=0)/np.mean(np.array(vars_sv), axis=0), label='Sample variance')
    plt.plot(bins, np.std(np.array(vars_jn), axis=0)/np.mean(np.array(vars_jn), axis=0), label='Jackknife')
    plt.plot(bins, np.std(np.array(vars_bs), axis=0)/np.mean(np.array(vars_bs), axis=0), label='Bootstrap')
    plt.legend(fontsize=14)
    plt.ylabel('$\\sigma(\\sigma_{w(\\theta)})/\\sigma_{w(\\theta)}$', fontsize=20)
    plt.xlabel('$\\theta$ [deg]', fontsize=20)
    plt.xscale('log')
    plt.tight_layout()
    if show:
        plt.show()
    if return_fig:
        return f
    

def ratio_corr_variances_masked_unmasked(bins, masked_varxi, unmasked_varxi, mask=None, ratio_fsky_unmasked_masked=None, \
                                        return_fig=True, show=True, title='Ratio of sample variances, mock source maps'):
    f = plt.figure(figsize=(8, 6))
    plt.title(title, fontsize=18)
    plt.errorbar(bins, np.mean(np.array(masked_varxi)/np.array(unmasked_varxi), axis=0), yerr=np.std(np.array(masked_varxi)/np.array(unmasked_varxi), axis=0), marker='.')
    
    if ratio_fsky_unmasked_masked is None and mask is not None:
        ratio_fsky_unmasked_masked = mask.shape[0]*mask.shape[1]/np.sum(mask)
    
    if ratio_fsky_unmasked_masked is not None:
    
        plt.axhline(ratio_fsky_unmasked_masked, linestyle='dashed', label='$f_{sky}/f_{sky}^{masked}$', color='k')
        plt.axhline(np.sqrt(ratio_fsky_unmasked_masked), linestyle='dashed', label='$\\sqrt{f_{sky}/f_{sky}^{masked}}$', color='k')

    plt.legend(fontsize=16)
    plt.ylabel('$\\sigma^2_{w(\\theta),masked}/\\sigma^2_{w(\\theta)}$', fontsize=18)
    plt.xscale('log')
    plt.xlabel('$\\theta$ [deg]', fontsize=18)
    plt.ylim(0, 5)
    if show:
        plt.show()
    if return_fig:
        return f



