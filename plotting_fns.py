import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
# import seaborn as sns
# sns.set()
from PIL import Image
# from Pillow import Image
import glob

from integrate_cl_wtheta import *
# from ciber_powerspec_pipeline import compute_knox_errors



figure_directory = '/Users/richardfeder/Documents/caltech/ciber2/figures/'



def compare_c_ells_diff_estimators(midbin_ells, midbin_ell_fourier=None, cls_2pcf=None, cls_fourier=None, B_ell=None,\
                                   plot_d_ell=False, return_fig=True, show=True, title=None, sqrt=False, ylim=None, plot_ytc=False, fudge_fac=1.):
    

    if plot_d_ell:
        prefactor = midbin_ells*(midbin_ells+1.)/(2*np.pi)
    else:
        prefactor = 1.
    
    f = plt.figure(figsize=(8, 6))
    
    if title is not None:
        plt.title(title, fontsize=16)
    
    
    if cls_2pcf is not None:
        if len(midbin_ells) > 50:
            plt.plot(midbin_ells, fudge_fac*prefactor*np.median(cls_2pcf, axis=0), marker='.', color='r', label='$w(\\theta)\\rightarrow C_{\\ell}$')
            plt.fill_between(midbin_ells, prefactor*np.percentile(cls_2pcf, 16, axis=0), prefactor*np.percentile(cls_2pcf, 84, axis=0), alpha=0.3, color='r')
        else:
            plt.errorbar(midbin_ells, fudge_fac*prefactor*np.median(cls_2pcf, axis=0), yerr=fudge_fac*prefactor*np.std(cls_2pcf, axis=0), marker='.', capsize=5, color='r', label='$w(\\theta)\\rightarrow C_{\\ell}$')
            if B_ell is not None:
                plt.errorbar(midbin_ells, fudge_fac*prefactor*np.median(cls_2pcf, axis=0)/np.sqrt(B_ell), yerr=fudge_fac*prefactor*np.std(cls_2pcf, axis=0)/np.sqrt(B_ell), marker='.', capsize=5, color='r', linestyle='dashed', label='$w(\\theta)\\rightarrow C_{\\ell}$ (Beam corrected)')

    if cls_fourier is not None:
        if midbin_ell_fourier is not None:
            ells = midbin_ell_fourier
            if plot_d_ell:
                prefactor = ells*(ells+1.)/(2*np.pi)
        else:
            ells = midbin_ells
        
        
        print('ells has length', len(ells))
        print('cls_fourier has shape', cls_fourier.shape)
        if len(midbin_ells) > 50:
            plt.plot(ells, prefactor*np.median(cls_fourier, axis=0), marker='.', color='C0', label='$C_{\\ell}$ (DFT)')
            plt.fill_between(ells, prefactor*np.percentile(cls_fourier, 16, axis=0), prefactor*np.percentile(cls_fourier, 84, axis=0), alpha=0.3, color='C0')
        else:
            plt.errorbar(ells, prefactor*np.median(cls_fourier, axis=0), yerr=prefactor*np.std(cls_fourier, axis=0), marker='.', capsize=5, color='C0', label='$C_{\\ell}$ (DFT)')
            if B_ell is not None:
                plt.errorbar(ells, prefactor*np.median(cls_fourier, axis=0)/B_ell, yerr=prefactor*np.std(cls_fourier, axis=0)/np.sqrt(B_ell),linestyle='dashed', marker='.', capsize=5, color='C0', label='$C_{\\ell}$ (DFT, beam corrected)')

    
    if plot_ytc:
        cls_dat = np.load('/Users/richardfeder/Downloads/Cls_data.npy')
        ratio = cls_dat[-1][0]/midbin_ells[-1]    
        prefy = (cls_dat[-1]/ratio)**2/(2*np.pi)
        plt.errorbar(cls_dat[-1]/ratio, prefy*np.median(cls_dat[:-1], axis=0), yerr=prefy*np.std(cls_dat[:-1], axis=0), capsize=5, marker='.', label='Yun-Ting $C_{\\ell}$', color='g')

    plt.yscale('log')
    plt.xscale('log')
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.xlabel('Multipole $\\ell$', fontsize=16)
    if plot_d_ell:
        plt.ylabel('$\\frac{\\ell(\\ell+1)C_{\\ell}}{2\\pi}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=16)
    else:
        plt.ylabel('$C_{\\ell}$ [nW$^2$ m$^{-4}$ sr$^{-1}$]', fontsize=16)
    plt.legend(fontsize=14)
    
    if show:
        plt.show()
    
    if return_fig:
        return f

def filter_fns_config(bandpower_edges, theta_window, thetas, return_fig=True, mode='analytic', apodize=False, lndth=0.5):
    '''
    bandpower_edges -- list specifying edges of multipole bins for bandpowers
    theta_window -- length 2 list with minimum and maximum separation measured in 2PCF (in radians)
    thetas -- list specifying range to compute filter function over
    
    '''
    n_bandpower = len(bandpower_edges)
    print("assuming 20 band powers")
    
    if apodize:
        mint = theta_window[0]
        maxt = theta_window[1]
        hann_window_vals = np.array([hann_window(np.log10(theta), 0.5, np.log10(mint), np.log10(maxt)) for theta in thetas])

        plt.figure()
        plt.title('Hann window, $\\Delta\\ln x=0.5$')
        plt.plot(thetas*(180./np.pi), hann_window_vals)
        plt.xscale('log')
        plt.xlabel('$\\theta$ [deg]')
        plt.ylabel('$T(\\theta)$')
        plt.show()
    
    
    fig = plt.figure(figsize=(15,8))
    plt.suptitle('Full (black) vs. windowed (blue) bandpower filter functions', fontsize=20, y=1.02)
    for b in range(min(20, n_bandpower-1)):
        
        if apodize:
            tophat_vals = g_theta_tophat_bandpass(thetas, bandpower_edges[b], bandpower_edges[b+1])
            tophat_vals *= hann_window_vals

        else:
            tophat_vals = tophat_bp_window_config_space(thetas, bandpower_edges[b], bandpower_edges[b+1],\
                                                             thetamin=theta_window[0], thetamax=theta_window[1])

        plt.subplot(4, 5, b+1)
        plt.title('Bandpower '+str(b+1))
        # x axis has unit of degrees so multiply thetas by 180/np.pi
        plt.plot(thetas*(180./np.pi), g_theta_tophat_bandpass(thetas, bandpower_edges[b], bandpower_edges[b+1]), color='k', label='full')
        plt.plot(thetas*(180./np.pi), tophat_vals, label='apodized', color='b')
        if theta_window[0] > np.min(thetas):
            plt.axvline(theta_window[0]*180./np.pi, color='g', linestyle='dashed')
        if theta_window[1] < np.max(thetas):
            plt.axvline(theta_window[1]*180./np.pi, color='g', linestyle='dashed')
        
        if apodize:
            plt.axvline(180./np.pi*theta_window[0]*10**(-0.5*lndth), color='b', linestyle='dashed')
            plt.axvline(180./np.pi*theta_window[1]*10**(0.5*lndth), color='b', linestyle='dashed')
        plt.xscale('log')
        if b > 14:
            plt.xlabel('$\\theta$ [deg]', fontsize=18)
        if b % 5 ==0:
            plt.ylabel('$g_b(\\theta)$', fontsize=18)
    plt.tight_layout()
    plt.show()
    
    if return_fig:
        return fig


def multi_panel_mc_power_spectrum_vs_Jlim_plot(lbins, all_cl_proc_vs_Jlim, ground_truth_cl_vs_Jlim,Jmags, \
                                             delta_ell=None, B_ell=None, fsky=None, unmasked_N_ell=None, masked_N_ells=None,\
                                               return_fig=True, show=True, inst=None, suptitle=None, suptitlefontsize=20, supy=1.04):
    
    
    prefac = (lbins*(lbins+1)/(2*np.pi))
    
    if unmasked_N_ell is not None:
        f = plt.figure(figsize=(12, 10))
        plt.subplot(2,2,1)
    else:
        f = plt.figure(figsize=(15, 5))
        plt.subplot(1,3,1)
        
    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=suptitlefontsize, y=supy)
        
    plt.title('True vs. recovered power spectra', fontsize=18)
    plt.xscale("log")
    plt.yscale('log')
    
    for j, all_cl_proc in enumerate(all_cl_proc_vs_Jlim):

        yerrs = [prefac*(np.mean(all_cl_proc, axis=0)-np.percentile(all_cl_proc, 16, axis=0)), prefac*(np.percentile(all_cl_proc, 84, axis=0)-np.mean(all_cl_proc, axis=0))]
        tlabel = None
        rlabel = None
        nlabel = None
        nmlabel = None
        if j==0:
            tlabel = 'Truth'
            rlabel = 'Recovered'
            nlabel = 'Noise (unmasked, unweighted)'
            nmlabel = 'Noise (masked, weighted)'
            
        plt.errorbar(lbins, prefac*np.mean(all_cl_proc, axis=0), yerr=yerrs, label=rlabel, color='C'+str(j), capsize=3, linewidth=2)
        plt.plot(lbins, prefac*ground_truth_cl_vs_Jlim[j], label=tlabel, marker='.',linewidth=2, color='C'+str(j), linestyle='dashdot')    

        if masked_N_ells is not None:
            plt.plot(lbins, prefac*masked_N_ells[j], linestyle='solid', alpha=0.2, color='k', label=nmlabel)

            
    plt.legend(fontsize=12)
    plt.xlabel('Multipole $\\ell$', fontsize=18)
    plt.ylabel('$\\ell(\\ell+1)C_{\\ell}/2\\pi$ [(nW m$^{-2}$ sr$^{-1})$$^2$]', fontsize=18)
    plt.ylim(2e-3, 4e4)
    knoxes_dcl_cl = []
    mc_dcl_cl = []
    plt.tick_params(labelsize=14)
    
    if unmasked_N_ell is not None:
        plt.subplot(2,2,2)
    else:
        plt.subplot(1,3,2)
    plt.title('Fractional bias', fontsize=18)
    for j, all_cl_proc in enumerate(all_cl_proc_vs_Jlim):

        fractional_biases = np.array([(cl_proc - ground_truth_cl_vs_Jlim[j])/ground_truth_cl_vs_Jlim[j] for cl_proc in all_cl_proc])
        yerrs = [(np.mean(fractional_biases, axis=0)-np.percentile(fractional_biases, 16, axis=0))/np.sqrt(len(all_cl_proc)), (np.percentile(fractional_biases, 84, axis=0)-np.mean(fractional_biases, axis=0))/np.sqrt(len(all_cl_proc))]
        plt.errorbar(lbins, np.mean(fractional_biases, axis=0), yerr=yerrs, capsize=5, color='C'+str(j), marker='.')

    plt.ylabel('$(\\langle C_{\\ell} \\rangle^{M.C.} - C_{\\ell}^{True})/C_{\\ell}^{True}$', fontsize=18)
    plt.xlabel('Multipole $\\ell$', fontsize=18)
    plt.ylim(-1.5, 1.5)
    plt.xscale('log')
    plt.tick_params(labelsize=14)
    

    if unmasked_N_ell is not None:
        knoxes_cl_dcl, mc_cl_dcl = [], []
        plt.subplot(2,2,3)
    else:
        plt.subplot(1,3,3)
    
    for j, all_cl_proc in enumerate(all_cl_proc_vs_Jlim):
        if j==0:
            mlabel = 'Masked, $J_{lim}=$'+str(Jmags[j])
        else:
            mlabel = '$J_{lim}=$'+str(Jmags[j])
    
    
        plt.plot(lbins, 2*np.median(all_cl_proc, axis=0)/(np.percentile(all_cl_proc, 84, axis=0)-np.percentile(all_cl_proc, 16, axis=0)), label=mlabel, color='C'+str(j))
        
        if unmasked_N_ell is not None:

            knox_cl_dcl = compute_knox_errors(lbins, ground_truth_cl_vs_Jlim[j], unmasked_N_ell, delta_ell, fsky=fsky, \
                                         B_ell=B_ell, snr=True)
            
            knoxes_cl_dcl.append(knox_cl_dcl)
            mc_cl_dcl.append(np.abs(2*np.median(all_cl_proc, axis=0)/(np.percentile(all_cl_proc, 84, axis=0)-np.percentile(all_cl_proc, 16, axis=0))))

        
            if j == 0:
                klabel = 'Knox errors'
            else:
                klabel = None
            plt.plot(lbins, knox_cl_dcl, label=klabel, color='C'+str(j), linestyle='dashed')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Multipole $\\ell$', fontsize=18)
    plt.ylim(1e-2, 2e2)
    plt.ylabel('$C_{\\ell}/\\delta C_{\\ell}$', fontsize=18)
    plt.legend(fontsize=12, loc='best', ncol=2)
    plt.title('Signal to noise', fontsize=18)
    plt.tick_params(labelsize=14)

    if unmasked_N_ell is not None:
        plt.subplot(2,2,4)

        plt.xscale('log')

        for j, all_cl_proc in enumerate(all_cl_proc_vs_Jlim):
            plt.plot(lbins, knoxes_cl_dcl[j]/mc_cl_dcl[j], label='Masked, $J_{lim}=$'+str(Jmags[j]), marker='.', color='C'+str(j))

        plt.ylabel('$\\delta C_{\\ell}^{M.C.}/\\delta C_{\\ell}^{Knox}$', fontsize=18)
        plt.xlabel('Multipole $\\ell$', fontsize=18)
        plt.title('Deviation from Knox errors', fontsize=18)
        plt.tight_layout()
        plt.ylim(0, 10)
        plt.tick_params(labelsize=14)
        plt.axhline(1, linestyle='dashed', color='k')
    
    
    if show:
        plt.show()
    if return_fig:
        if unmasked_N_ell is not None:
            return f, knoxes_cl_dcl, mc_cl_dcl
        
        return f
        

def plot_stacked_smoothed_expmap(sum_im_smoothed, lightsrcmask, minpct=1, maxpct=99, show=True, return_fig=True):
    f = plt.figure(figsize=(10, 4))
    plt.suptitle('Stacked/smoothed exposure map, $\\sigma=5$ pix', fontsize=20, y=1.03)
    plt.subplot(1,2,1)
    plt.imshow(sum_im_smoothed, vmin=np.percentile(sum_im_smoothed, 1), vmax=np.percentile(sum_im_smoothed, 99), origin='lower')
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('[ADU/fr]', fontsize=16)
    plt.xlabel('x [pix]', fontsize=16)
    plt.ylabel('y [pix]', fontsize=16)
    plt.subplot(1,2,2)
    plt.imshow(lightsrcmask, origin='lower')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlabel('x [pix]', fontsize=16)
    plt.ylabel('y [pix]', fontsize=16)
    plt.tight_layout()

    if show:
        plt.show()
    if return_fig:
        return f


def plot_g1_hist(g1facs, mask=None, return_fig=True, show=True, title=None):
    if mask is not None:
        g1facs = g1facs[mask]
        
    f = plt.figure(figsize=(6,5))
    if title is not None:
        plt.title(title, fontsize=18)
    plt.hist(g1facs)
    plt.axvline(np.median(g1facs), label='Median $G_1$ = '+str(np.round(np.median(g1facs), 3)), linestyle='dashed', color='k', linewidth=2)
    plt.axvline(np.mean(g1facs), label='Mean $G_1$ = '+str(np.round(np.mean(g1facs), 3)), linestyle='dashed', color='r', linewidth=2)
    plt.xlabel('$G_1$ [(e-/s)/(ADU/fr)]', fontsize=16)
    plt.ylabel('N', fontsize=16)
    plt.legend(fontsize=14)
    plt.tick_params(labelsize=14)
    
    if show:
        plt.show()
    
    if return_fig:
        return f
        

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
			yerr = np.array([np.sqrt((rbins*lmin)**2/(2*np.pi))*raderrs[i]/(2*np.sqrt(radprofs[i])) for i in range(len(raderrs))]) # I don't think this is right, modify at some point
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


def plot_radavg_xspectrum2(rbins, radprofs=[], raderrs=None, labels=None, \
                          lmin=90., save=False, snspalette=None, pdf_or_png='png', \
                         image_dim=1024, mode='cross', zbounds=None, shotnoise=None, \
                         add_shot_noise=None, Ntot=160000, sn_npoints=10, titlestring=None, \
                          ylims=None):

    steradperpixel = ((np.pi/lmin)/image_dim)**2 

    f = plt.figure(figsize=(8,6))
    xvals = rbins
    
    if labels is None:
        labels = [None for x in range(len(radprofs))]

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
        
        yvals = np.array([np.sqrt((rbins)**2*prof/(2*np.pi)) for prof in radprofs])

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
        
        if raderrs is not None:
            plt.errorbar(xvals, yvals[i], yerr=yerr[i], marker='.', label=labels[i], color=color)
        else:
            plt.plot(xvals, yvals[i], marker='.', label=labels[i], color=color)

    if ylims is not None:
        plt.ylim(ylims[0], ylims[1])
    plt.legend(fontsize=14)
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

def plot_cumulative_numbercounts(cat, mag_idx, field=None, df=True, m_min=18, m_max=30, vlines=None, hlines=None, \
                                nbin=100, label=None, pdf_or_png='pdf', band_label='W1', show=True, \
                                  return_fig=True, magsys='AB'):
    f = plt.figure()
    title = 'Cumulative number count distribution'
    if field is not None:
        title += ' -- '+str(field)
    plt.title(title, fontsize=14)
    magspace = np.linspace(m_min, m_max, nbin)
    if df:
        cdf_nm = np.array([len(cat.loc[cat[mag_idx]<x]) for x in magspace])
    else:
        print(len(cat))
        cdf_nm = np.array([len(cat[(cat[:,mag_idx] < x)]) for x in magspace])
    plt.plot(magspace, cdf_nm, label=label)
    if vlines is not None:
        for vline in vlines:
            plt.axvline(vline, linestyle='dashed')
    if hlines is not None:
        for hline in hlines:
            plt.axhline(hline, linestyle='dashed')
    plt.yscale('log')
    plt.ylabel('N['+band_label+' < magnitude]', fontsize=14)
    plt.xlabel(band_label+' magnitude ('+magsys+')', fontsize=14)
    plt.legend()
    if show:
        plt.show()
    if return_fig:
        return f

def plot_number_counts_uk(df, binedges=None, bands=['yAB', 'jAB', 'hAB', 'kAB'], magstr='', add_title_str='', return_fig=False):
    if binedges is None:
        binedges = np.arange(7,23,0.5)

    f = plt.figure()
    plt.title(magstr+add_title_str, fontsize=16)
    bins = (binedges[:-1] + binedges[1:])/2
    
    for band in bands:
        d = df[band+magstr]
        shist,_ = np.histogram(d, bins = binedges)
        plt.plot(bins,shist / (binedges[1]-binedges[0]) / 4,'o-',label=band)

    plt.yscale('log')
    plt.xlabel('$m_{AB}$', fontsize=14)
    plt.ylabel('counts / mag / deg$^2$', fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()
    if return_fig:
        return f

def plot_number_counts(df, binedges=None, bands=['g', 'r', 'i', 'z', 'y'], magstr='MeanPSFMag', add_title_str='', return_fig=False):
    if binedges is None:
        binedges = np.arange(7,23,0.5)

    f = plt.figure()
    plt.title(magstr+add_title_str, fontsize=16)
    bins = (binedges[:-1] + binedges[1:])/2
    
    for band in bands:
        d = df[band+magstr]
        shist,_ = np.histogram(d, bins = binedges)
        plt.plot(bins,shist / (binedges[1]-binedges[0]) / 4,'o-',label=band)

    plt.yscale('log')
    plt.xlabel('$m_{AB}$', fontsize=14)
    plt.ylabel('counts / mag / deg$^2$', fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()
    if return_fig:
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
                              suptitle='Mock CIBER source map (no noise), SWIRE flight mask (50 realizations)', \
                              nsim_list=None, ylim=[-0.5, 0.5]):

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
    if nsim_list is not None:
        for nsim in nsim_list:
            plt.plot(thetas, np.median(delta_xis[:nsim], axis=0)/np.sqrt(np.median(unmasked_varxi[:nsim], axis=0)), marker='.', alpha=0.8, label='$n_{sim}=$'+str(nsim))
    else:  
        plt.plot(thetas, np.mean(delta_xis, axis=0)/np.sqrt(np.mean(unmasked_varxi, axis=0)), marker='.', alpha=0.8)
    plt.xlabel('$\\theta$ [deg]', fontsize=18)
    plt.ylabel('$\\frac{\\langle w(\\theta)_{masked} - w(\\theta)_{true} \\rangle}{\\langle \\sigma(w(\\theta)_{true})_{SV}\\rangle}$', fontsize=20)
    plt.axhline(0.1, linestyle='dashed', color='k')
    plt.axhline(-0.1, linestyle='dashed', color='k')
    plt.legend()
    plt.xscale('log')
    
    plt.ylim(ylim)
    plt.tight_layout()
    plt.show()
    
    if return_fig:
        return f


def plot_exposure_pair_diffs_means(pair_diff, pair_mean, pair_mask, diffidx, minpct=1, maxpct=99, show=True, return_fig=True):
    f = plt.figure(figsize=(16,8))
    plt.subplot(1,2,2)
    plt.title('Masked pair difference, i = '+str(diffidx), fontsize=20)
    plt.imshow(pair_diff, vmin=np.nanpercentile(pair_diff, minpct), vmax=np.nanpercentile(pair_diff, maxpct), origin='lower')
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('ADU/fr', fontsize=14)
    plt.xlabel('x [pix]', fontsize=16)
    plt.ylabel('y [pix]', fontsize=16)

    plt.subplot(1,2,1)
    plt.title('Masked pair average, i = '+str(diffidx), fontsize=20)
    plt.imshow(pair_mean*pair_mask, vmin=np.nanpercentile(pair_mean*pair_mask, minpct), vmax=np.nanpercentile(pair_mean*pair_mask, maxpct), origin='lower')
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('ADU/fr', fontsize=14)
    plt.xlabel('x [pix]', fontsize=16)
    plt.ylabel('y [pix]', fontsize=16)

    if show:
        plt.show()
        
    if return_fig:
        return f


def plot_srcmap_mask(mask, titlestr, len_cat, return_fig=True, show=True):
    f = plt.figure(figsize=(10, 10))
    
    plt.title(titlestr + ' \n N='+str(len_cat)+', '+str(np.round(100*np.sum(mask)/(mask.shape[0]**2), 2))+'% of pixels unmasked', fontsize=18)
    plt.imshow(mask, cmap='Greys_r', origin='lower')
    plt.xlabel('x [pix]', fontsize=18)
    plt.ylabel('y [pix]', fontsize=18)
    cbar = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08)
    
    if show:
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



def plot_readnoise_realization(rdn_real, cal_fac, show=True, return_fig=True, adupframe_range=[-0.75, 0.75]):
    f = plt.figure(figsize=(15, 6))


    plt.subplot(1,3,1)
    plt.title('Read noise realization', fontsize=16)
    plt.imshow(rdn_real*cal_fac, cmap='Greys')
    plt.colorbar()
    
    plt.subplot(1,3,2)
    plt.hist(rdn_real.ravel(), bins=np.linspace(adupframe_range[0], adupframe_range[1], 100))
    plt.xlabel('Read noise [ADU/frame]', fontsize=16)
    plt.ylabel('$N_{pix}$', fontsize=16)

    plt.subplot(1,3,3)
    plt.hist(rdn_real.ravel()*cal_fac, bins=100)
    plt.xlabel('Read noise [nW m$^{-2}$ sr$^{-1}$]', fontsize=16)
    plt.ylabel('$N_{pix}$', fontsize=16)

    plt.tight_layout()
    
    if show:
        plt.show()
    
    if return_fig:
        return f


def plot_means_vs_vars(m_o_m, v_o_d, timestr_cut=None, var_frac_thresh=None, xlim=None, ylim=None, all_set_numbers_list=None, all_timestr_list=None,\
                      nfr=5, inst=1, fit_line=True, itersigma=4.0, niter=5):
    
    mask = [True for x in range(len(m_o_m))]
    
    from astropy.stats import sigma_clip
    from astropy.modeling import models, fitting
    
    if timestr_cut is not None:
        mask *= np.array([t==timestr_cut for t in all_timestr_list])
    
    photocurrent = m_o_m[mask]
    varcurrent = v_o_d[mask]
    
    if fit_line:
        # initialize a linear fitter
        fit = fitting.LinearLSQFitter()
        # initialize the outlier removal fitter
        or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=niter, sigma=itersigma)
        # initialize a linear model
        line_init = models.Linear1D()
        # fit the data with the fitter
        sigmask, fitted_line = or_fit(line_init, photocurrent, varcurrent)
        
        g1_iter = get_g1_from_slope_T_N(0.5*fitted_line.slope.value, N=nfr)

    
    if var_frac_thresh is not None:
        median_var = np.median(varcurrent)
        print('median variance is ', median_var)
        mask *= (np.abs(v_o_d-median_var) < var_frac_thresh*median_var) 
        
        print(np.sum(mask), len(v_o_d))
        
    min_x_val, max_x_val = np.min(photocurrent), np.max(photocurrent)
    
        
    if all_set_numbers_list is not None:
        colors = np.array(all_set_numbers_list)[mask]
    else:
        colors = np.arange(len(m_o_m))[mask]
        
    f = plt.figure(figsize=(12, 6))
    title = 'TM'+str(inst)
    if timestr_cut is not None:
        title += ', '+timestr_cut
        
    plt.title(title, fontsize=18)
        
    timestrs = ['03-13-2013', '04-25-2013', '05-13-2013']
    markers = ['o', 'x', '*']
    set_color_idxs = []
    for t, target_timestr in enumerate(timestrs):
        tstrmask = np.array([tstr==target_timestr for tstr in all_timestr_list])
        
        tstr_colors = None
        if all_set_numbers_list is not None:
            tstr_colors = np.array(all_set_numbers_list)[mask*tstrmask]
        
        if fit_line:
            if t==0:
                if inst==1:
                    xvals = np.linspace(-5, -18, 100)
                else:
                    xvals = np.linspace(np.min(photocurrent), np.max(photocurrent), 100)
                    print('xvals:', xvals)
                plt.plot(xvals, fitted_line(xvals), label='Iterative sigma clip, $G_1$='+str(np.round(g1_iter, 3)))
            plt.scatter(photocurrent[tstrmask], sigmask[tstrmask], s=100, marker=markers[t], c=tstr_colors, label=target_timestr)
            plt.xlim(xlim)

        else:
            plt.scatter(photocurrent[tstrmask], varcurrent[tstrmask], s=100, marker=markers[t], c=tstr_colors, label=target_timestr)
            plt.xlim(xlim)
            plt.ylim(ylim)            

    plt.legend(fontsize=16)
    plt.xlabel('mean [ADU/fr]', fontsize=18)
    plt.ylabel('$\\sigma^2$ [(ADU/fr)$^2$]', fontsize=18)
    plt.tick_params(labelsize=14)
    plt.show()
    
    return m, f

def plot_median_std_2d_2pcf(wthetas_2d, suptitle='Read noise realizations, 256x256 pixel cutouts', suptitlefontsize=20, return_fig=True, show=True, pixsize=7., n_tick=5):

    f = plt.figure(figsize=(10, 5))
    plt.suptitle(suptitle, fontsize=suptitlefontsize)
    plt.subplot(1,2,1)
    plt.title('Median $w(\\theta_x, \\theta_y)$', fontsize=16)
    median_wtheta2d = np.median(wthetas_2d, axis=0)
    plt.imshow(median_wtheta2d, cmap='Greys', vmin=np.percentile(median_wtheta2d, 5), vmax=np.percentile(median_wtheta2d, 99.9), origin='lower')
    plt.xlabel('$\\theta_x$ [deg]', fontsize=14)
    plt.ylabel('$\\theta_y$ [deg]', fontsize=14)
    nx, ny = median_wtheta2d.shape[0], median_wtheta2d.shape[1]
    plt.xticks(np.arange(0, nx, nx//n_tick), np.round(2*(np.arange(0, nx, nx//n_tick)-nx//2)*(pixsize/3600.), 1))
    plt.yticks(np.arange(0, ny, ny//n_tick), np.round(2*(np.arange(0, ny, ny//n_tick)-ny//2)*(pixsize/3600.), 1))
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.title('Variance of $w(\\theta_x, \\theta_y)$', fontsize=16)
    std_wtheta_2d = np.std(wthetas_2d, axis=0)
    plt.imshow(std_wtheta_2d, vmax=np.percentile(std_wtheta_2d, 99), vmin=np.percentile(std_wtheta_2d, 1), cmap='Greys', origin='lower')
    plt.colorbar()
    plt.xlabel('$\\theta_x$ [deg]', fontsize=14)
    plt.ylabel('$\\theta_y$ [deg]', fontsize=14)
    
    plt.xticks(np.arange(0, nx, nx//n_tick), np.round(2*(np.arange(0, nx, nx//n_tick)-nx//2)*(pixsize/3600.), 1))
    plt.yticks(np.arange(0, ny, ny//n_tick), np.round(2*(np.arange(0, ny, ny//n_tick)-ny//2)*(pixsize/3600.), 1))

    if show:
        plt.show()

    if return_fig:
        return f
        

def plot_w_thetax_thetay(wthetas_2d, return_fig=True, show=True, title='Read noise realizations, 256x256 pixel cutouts'):
    
    nx, ny = np.median(wthetas_2d, axis=0).shape[0], np.median(wthetas_2d, axis=0).shape[1]
    pixsize = 7.
    n_tick = 6
    
    f = plt.figure(figsize=(10, 4))
    plt.suptitle(title, fontsize=20, y=1.03)
    plt.subplot(1,2,1)
    plt.title('Median $w(\\theta_x, \\theta_y)$', fontsize=16)
    plt.imshow(np.median(wthetas_2d, axis=0), cmap='Greys', vmin=np.percentile(np.median(wthetas_2d, axis=0), 5), vmax=np.percentile(np.median(wthetas_2d, axis=0), 95))
    plt.xticks(np.arange(0, nx, nx//n_tick), np.round(1*(np.arange(0, nx, nx//n_tick)-nx//2)*(pixsize/3600.), 1))
    plt.yticks(np.arange(0, ny, ny//n_tick), np.round(1*(np.arange(0, ny, ny//n_tick)-ny//2)*(pixsize/3600.), 1))
    plt.xlabel('$\\delta x$ [deg]', fontsize=16)
    plt.ylabel('$\\delta y$ [deg]', fontsize=16)
    
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.title('Standard deviation of $w(\\theta_x, \\theta_y)$', fontsize=16)
    plt.imshow(np.std(wthetas_2d, axis=0), cmap='Greys', vmin=np.percentile(np.std(wthetas_2d, axis=0), 5), vmax=np.percentile(np.std(wthetas_2d, axis=0), 95))

    plt.xticks(np.arange(0, nx, nx//n_tick), np.round(1*(np.arange(0, nx, nx//n_tick)-nx//2)*(pixsize/3600.), 1))
    plt.yticks(np.arange(0, ny, ny//n_tick), np.round(1*(np.arange(0, ny, ny//n_tick)-ny//2)*(pixsize/3600.), 1))
   
    plt.xlabel('$\\delta x$ [deg]', fontsize=16)
    plt.ylabel('$\\delta y$ [deg]', fontsize=16)

    plt.colorbar()
    if show:
        plt.show()
    if return_fig:
        return f


def plot_readnoise_2pcfs(bins, masked_wtheta=None, masked_var_wtheta=None, unmasked_wtheta=None, unmasked_var_wtheta=None,
                         masked=True, ylim=[-0.00003, 0.00015], xscale='log'):
    
    if masked:
        suptitle = 'Masked read noise realizations'
        wtheta = masked_wtheta
        var_wtheta = masked_var_wtheta
    else:
        suptitle = 'Unmasked read noise realizations'
        wtheta = unmasked_wtheta
        var_wtheta = unmasked_var_wtheta
        
    n_realization = len(wtheta)


    f = plt.figure(figsize=(15,5))
    plt.suptitle(suptitle, y=1.05, fontsize=20)
    plt.subplot(1,3,1)
    
    for i in range(n_realization):
        plt.errorbar(bins, wtheta[i], yerr=np.sqrt(var_wtheta[i]), alpha=0.3, color='C3', capsize=5)

    plt.title('Individual realizations (+jackknife variance)')    
    plt.xscale(xscale)
    plt.xlabel('$\\theta$ [deg]', fontsize=16)
    plt.ylabel('$w(\\theta)$', fontsize=16)

    plt.ylim(ylim)
    plt.subplot(1,3,2)
    plt.title('Mean of realizations (+std. deviation)')   
    plt.fill_between(bins, np.percentile(wtheta, 16, axis=0), np.percentile(wtheta, 84, axis=0), alpha=0.3)
    plt.plot(bins, np.mean(wtheta, axis=0), marker='.')
    plt.xscale(xscale)
    plt.xlabel('$\\theta$ [deg]', fontsize=16)
    plt.ylabel('$w(\\theta)$', fontsize=16)
    plt.ylim(ylim)

    plt.subplot(1,3,3)
    plt.title('Standard deviation across realizations')
    plt.errorbar(bins, np.std(wtheta, axis=0),  marker='.', capsize=5)

    plt.xscale(xscale)
    plt.yscale('log')
    plt.xlabel('$\\theta$ [deg]', fontsize=16)
    plt.ylabel('$\\sigma(w(\\theta))$', fontsize=16)
    
    plt.tight_layout()
    
    plt.show()
    
    return f


def plot_wtheta(rnom, xi, varxi, skymap, pix_units='rad', return_fig=False, mask=None, title='Read noise realization', plot_xscale='log'):
    
    f = plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    if mask is not None:
        title += ' (masked)'
    plt.title(title, fontsize=16)
    if mask is None:
        plt.imshow(skymap, cmap='Greys')
    else:
        plt.imshow(skymap*mask, cmap='Greys')
    plt.colorbar()
    plt.xlabel('x [pixel]', fontsize=16)
    plt.ylabel('y [pixel]', fontsize=16)
    plt.subplot(1,2,2)
    plt.errorbar(rnom, xi, yerr=np.sqrt(varxi), marker='.')
    plt.xlabel('$\\theta$ ('+str(pix_units)+')', fontsize=16)
    plt.ylabel('$w(\\theta)$ [$nW^2 m^{-4} sr^{-2}$]', fontsize=16)
    if plot_xscale=='log':
        plt.xscale('log')
    plt.tight_layout()
    plt.show()
    
    if return_fig:
        return f


def threepanel_wthetaxy_to_cl2d_vs_dftcl2d(wthetaxy, cl2d_dft, zoom=False, return_fig=True, show=True, \
                                          vrange1=[1e3, 1e8], vrange2=[1e3, 1e8]):

    nx, ny = wthetaxy.shape[0], wthetaxy.shape[1]

    f = plt.figure(figsize=(15, 5))
    plt.subplot(1,3,1)
    plt.title('$w(\\theta_x, \\theta_y)$', fontsize=20)
    plt.imshow(wthetaxy.transpose(), cmap='Greys', vmin=-100, vmax=100)
    plt.xticks(np.arange(0, nx, nx//5), np.round(4*(np.arange(0, nx, nx//5)-nx//2)*(7./3600.), 1))
    plt.yticks(np.arange(0, ny, ny//5), np.round(4*(np.arange(0, ny, ny//5)-ny//2)*(7./3600.), 1))

    plt.colorbar()
    plt.xlabel('$\\theta_x$ [deg]', fontsize=16)
    plt.ylabel('$\\theta_y$ [deg]', fontsize=16)

    
    cl2d = np.fft.ifftshift(np.sqrt(np.real(fft2(median_wth)*np.conj(fft2(median_wth)))))

    plt.subplot(1,3,2)
    plt.title('$C_{\\ell_x,\\ell_y} \\approx \\mathcal{F}(\\langle w(\\theta_x, \\theta_y) \\rangle)$', fontsize=20)
    plt.imshow(cl2d.transpose(), norm=LogNorm(), cmap='Greys', vmax=vrange1[1], vmin=vrange1[0])
    plt.colorbar()
    plt.xlabel('$\\ell_x$', fontsize=16)
    plt.ylabel('$\\ell_y$', fontsize=16)
    
    
    plt.xticks([], [])
    plt.yticks([], [])
    # plt.xticks(np.arange(0, nx, nx//6), np.round(180*(np.arange(0, nx, nx//6)-nx//4 + 1), 2))
    # plt.yticks(np.arange(0, ny, ny//6), np.round(180*(np.arange(0, nx, nx//6)-nx//4 +1), 2))
    if zoom:
        plt.xlim(nx//2-24, nx//2+24)
        plt.ylim(ny//2-24, ny//2+24)        
        
        
    nx2, ny2 = cl2d_dft.shape[0], cl2d_dft.shape[1]

    plt.subplot(1,3,3)
    plt.title('$C_{\\ell_x,\\ell_y}$ (full)', fontsize=20)


    plt.imshow(cl2d_dft, norm=LogNorm(), cmap='Greys', vmax=vrange2[1], vmin=vrange2[0])
    plt.colorbar()
    plt.xlabel('$\\ell_x$', fontsize=16)
    plt.ylabel('$\\ell_y$', fontsize=16)

    if zoom:
        plt.xlim(500, 524)
        plt.ylim(500, 524)
        plt.xticks([], [])
        plt.yticks([], [])
    else:
        plt.xticks(np.arange(0, nx2, nx2//6), np.round(180*(np.arange(0, nx2, nx2//6)-nx2//2 + 2), 2))
        plt.yticks(np.arange(0, ny2, ny2//6), np.round(180*(np.arange(0, nx2, nx2//6)-nx2//2 + 2), 2))
    
    plt.tight_layout()

    if show:
        plt.show()
        
    if return_fig:
        return f


def plot_hankel_integrand(bins_rad, fine_bins, wtheta, spline_wtheta, ell, integration_max):
    
    f = plt.figure(figsize=(10, 8))
    plt.subplot(2,2,3)
    plt.title('Discrete Integrand', fontsize=16)
    plt.plot(bins_rad, bins_rad*wtheta*scipy.special.j0(ell*bins_rad), label='$\\ell = $'+str(ell))
    plt.ylabel('$\\theta w(\\theta)J_0(\\ell\\theta)$', fontsize=18)
    plt.xlabel('$\\theta$ [rad]', fontsize=18)
    plt.legend(fontsize=14)
    plt.xscale('log')
    plt.subplot(2,2,4)
    plt.title('Spline Integrand', fontsize=16)
    plt.plot(fine_bins, fine_bins*spline_wtheta(fine_bins)*scipy.special.j0(ell*fine_bins), label='$\\ell = $'+str(ell))
    plt.ylabel('$\\theta w(\\theta)J_0(\\ell\\theta)$', fontsize=18)
    plt.xlabel('$\\theta$ [rad]', fontsize=18)
    plt.axvline(integration_max, linestyle='dashed', color='k', label='$\\theta_{max}$='+str(np.round(integration_max*(180./np.pi), 2))+' deg.')
    plt.legend(fontsize=14)
    plt.xscale('log')
    plt.subplot(2,2,1)
    plt.title('Angular 2PCF', fontsize=16)
    plt.plot(bins_rad, wtheta, marker='.', color='k')
    plt.plot(np.linspace(np.min(bins_rad), np.max(bins_rad), 1000), spline_wtheta(np.linspace(np.min(bins_rad), np.max(bins_rad), 1000)), color='b', linestyle='dashed')
    plt.xscale('log')
    plt.yscale('symlog')
    plt.xlabel('$\\theta$ [rad]', fontsize=18)
    plt.ylabel('$w(\\theta)$', fontsize=18)
    plt.subplot(2,2,2)
    plt.title('Bessel function, $\\ell = $'+str(ell), fontsize=16)
    plt.plot(fine_bins, scipy.special.j0(ell*fine_bins), label='$\\ell = $'+str(ell))
    plt.ylabel('$J_0(\\ell\\theta)$', fontsize=18)
    plt.xlabel('$\\theta$ [rad]', fontsize=18)
    plt.xscale('log')
    plt.tight_layout()
    plt.show()
    
    return f

def show_masked_srcmap(masked_srcmap, headstr='IBIS + 2MASS masked source map', vmin=None, vmax=None, return_fig=True, jmaglim=18.5):
    f = plt.figure(figsize=(8,8))
    plt.title(headstr+'\n Bootes A, J < '+str(jmaglim), fontsize=20)
    plt.imshow(masked_srcmap,vmin=vmin, vmax=vmax, cmap='bwr')
    cbar = plt.colorbar(orientation='vertical', fraction=0.046, pad=0.08)
    cbar.set_label('$\\nu I_{\\nu}$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=16)
    plt.tight_layout()
    plt.xlabel('x [pix]', fontsize=18)
    plt.xlabel('y [pix]', fontsize=18)
    plt.show()
    if return_fig:
        return f


