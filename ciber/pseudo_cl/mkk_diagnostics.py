import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fftshift as fftshift
from numpy.fft import ifftshift as ifftshift
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
from ciber.pseudo_cl.mkk_compute import *
# from ciber_data_helpers import *
import pyfftw


class Mkk_diag():
    ''' This is a class that will contain the various diagnostic tests we want to run on the Mkk method. It uses the Mkk_bare class object
    fairly extensively, i.e. it generally assumes you've already used Mkk_bare() to construct an Mkk matrix. This includes some of the fast FFT
    objects that are constructed in Mkk_bare, though there are options to evaluate the power spectrum of maps one by one. '''
    
    
    mkk_inv = None
    
    def __init__(self, Mkk=None, load_mkks=True):
        
        if Mkk is not None:
            self.Mkk = Mkk
            if load_mkks:
                self.all_Mkks = self.Mkk.all_Mkks
                self.av_Mkk = self.Mkk.av_Mkk
                self.mkk_inv = compute_inverse_mkk(self.Mkk.av_Mkk)
            
    def load_mkk_inv(self, mkk_inv):

        self.mkk_inv = mkk_inv

        
    def compute_true_minus_rectified_c_ell(self, allmaps=None, generate_white_noise=True, n_test_realizations=None, mask=None, \
                                            ensemble_fft=False, n_split=2, sub_bins=False, return_cls=False, fractional_dcl=True, \
                                            n_phase_realizations=None, mean_or_median='median', logclslop=None, A=1.0):
        
        delta_cl = []
        all_cl_true, all_cl_rect, all_cl_masked = [], [], []
        
        # if generating white noise as a test, which seems like a test we might do, specify n_test_realizations
        
        if generate_white_noise:
            if n_test_realizations > 10:
                print('Using ensemble fft to handle '+str(n_test_realizations)+' white noise realizations')
                ensemble_fft=True
            else:
                print('Generating '+str(n_test_realizations)+' white noise realizations..')
        
        else:
            if n_test_realizations is None:
                n_test_realizations = len(allmaps)

        # if n_phase_realizations is specified, only use that many individual Mkk realizations in evaluation.
        # furthermore, one can choose the mean or the median per element for phase realizations
        
        if n_phase_realizations is not None:
            print('using ', n_phase_realizations, 'phase realizations for the Mkk matrix')
            mkk_mat = self.all_Mkks[:n_phase_realizations]
            
            if mean_or_median=='mean':
                mkk_inv_mat = compute_inverse_mkk(np.mean(mkk_mat, axis=0))
            elif mean_or_median=='median':
                mkk_inv_mat = compute_inverse_mkk(np.median(mkk_mat, axis=0))

        else:
            mkk_inv_mat = self.mkk_inv
                    
        # carry out the tests. If not using the pyfftw memory structs for a large number of test realizations,
        # go through them individually, but otherwise get the fftw objects set up and evaluate them over n_split
        # iterations
        
        if not ensemble_fft:
            for i in range(n_test_realizations):
                single_map = allmaps[i]-np.mean(allmaps[i])
                cl_true = self.Mkk.compute_cl_indiv(single_map)
                cl_masked = self.Mkk.compute_cl_indiv(single_map*self.Mkk.mask)
                cl_rect = np.dot(mkk_inv_mat.transpose(), cl_masked)

                if fractional_dcl:
                    delta_cl.append((cl_true-cl_rect)/cl_true)
                else:
                    delta_cl.append(cl_true-cl_rect)
                

            if return_cls:
                all_cl_true.append(cl_true)
                all_cl_rect.append(cl_rect)
                all_cl_masked.append(cl_masked)
                
        else:
            
            # if there is no fftw object in place already, instantiate one
            if self.Mkk.fft_objs is None:
                
                maplist_split_shape = (n_test_realizations//n_split, self.Mkk.dimx, self.Mkk.dimy)
        
                self.Mkk.empty_aligned_objs, self.Mkk.fft_objs = construct_pyfftw_objs(3, maplist_split_shape)
            
            
            for n in range(n_split):

                maplist_split_shape = (n_test_realizations//n_split, self.Mkk.dimx, self.Mkk.dimy)

                if allmaps is None:
                    noise = np.random.normal(size=maplist_split_shape) + 1j*np.random.normal(size=maplist_split_shape)
                
                if generate_white_noise:
                    
                    print('generating '+str(n_test_realizations//n_split)+' test realizations..')
                    self.Mkk.empty_aligned_objs[1][:n_test_realizations//n_split, :, :] = noise

                elif logclslop is not None:

                    # make radius map in ell space
                    self.Mkk.get_ell_bins(shift=False)
                    
                    # multiply by spectrum
                    cl_2d = np.sqrt(A*self.Mkk.ell_map**(logclslop))

                    self.Mkk.empty_aligned_objs[0][:n_test_realizations//n_split, :, :] = np.array([cl_2d*n for n in noise])
                    
                    # remove inf values (can happen with negative log slope sometimes)
                    self.Mkk.empty_aligned_objs[0][np.isinf(self.Mkk.empty_aligned_objs[0])] = 0.
                    
                    
                    self.Mkk.fft_objs[0]()
                    
                    plt.figure()
                    plt.imshow(self.Mkk.empty_aligned_objs[1][0].real) 
                    plt.colorbar()
                    plt.show()

                else:
                    self.Mkk.empty_aligned_objs[1][:n_test_realizations//n_split, :, :] = allmaps[n*n_test_realizations//n_split :(n+1)*n_test_realizations//n_split ]
                

                cl_true = self.Mkk.get_ensemble_angular_autospec(nsims=n_test_realizations//n_split, sub_bins=sub_bins, apply_mask=False)
                cl_masked = self.Mkk.get_ensemble_angular_autospec(nsims=n_test_realizations//n_split, sub_bins=sub_bins, apply_mask=True)
                
                for i in range(len(cl_masked)):
                    cl_rect = np.dot(mkk_inv_mat.transpose(), cl_masked[i])
                    
                    if sub_bins:
                        cl_rect = np.array([np.mean(cl_rect[self.Mkk.correspond_bins[k]]) for k in range(len(self.Mkk.correspond_bins))])
   
                    if fractional_dcl:
                        delta_cl.append((cl_true[i]-cl_rect)/cl_true[i])
                    else:
                        delta_cl.append(cl_true[i]-cl_rect)
                        
        
                    if return_cls:
                        all_cl_true.append(cl_true[i])
                        all_cl_rect.append(cl_rect)
                        all_cl_masked.append(cl_masked[i])


        if return_cls:
            return delta_cl, all_cl_true, all_cl_masked, all_cl_rect
        
        
        
        return delta_cl
    
    
    def compute_knox_error_dcl(self, c_ell_rect=None, logclslop=None, use_fmask = False, f_fac=1024**2):
        
        if c_ell_rect is not None:
            
            cl_analytic = self.Mkk.midbin_ell**(logclslop)
                    
            fac = self.Mkk.arcsec_pp_to_radian**2

            cl_analytic *= fac
            
            dcl_cl = np.abs(c_ell_rect - cl_analytic)/cl_analytic
            
            return dcl_cl
        
        fmask = 1.
        if use_fmask:
            fmask = np.sum(self.Mkk.mask)/self.Mkk.mask.shape[0]**2   
        dcl_cl = np.sqrt(2./((2.*self.Mkk.midbin_ell+1.)*self.Mkk.delta_ell*fmask*self.Mkk.fsky))

        return dcl_cl
    
    def plot_distribution_of_Mkk_elements(self, elements='diag', all_Mkks=None, nbin=20, suptitlefontsize=20, suptitle_y=1.03, titlefontsize=16, labelfontsize=12,\
                                           return_fig=False, show=True):
        
        M_string = '$M_{\\ell \\ell^{\\prime}}$'
        
        if elements=='diag':
            suptitle='Diagonal Elements of '+M_string+' (bin indices i,j)'
            shift_ix=0
            shift_iy=0
            
        elif elements=='upper':
            suptitle='Upper Diagonal Elements of '+M_string+' (bin indices i,j)'
            shift_iy=1
            shift_ix=0
            
        elif elements=='lower':
            suptitle='Lower Diagonal Elements of '+M_string+' (bin indices i,j)'
            shift_iy=0
            shift_ix=1
            
        f = plt.figure(figsize=(12,6))
        plt.suptitle(suptitle, fontsize=suptitlefontsize, y=suptitle_y)
    
        
        for i in range(10):
            
            plt.subplot(2,5,i+1)
            plt.title('$i='+str(i+shift_ix+1)+', j='+str(i+shift_iy+1)+'$', fontsize=16)
            if all_Mkks is None:
                plt.hist(self.Mkk.all_Mkks[:,i+shift_ix,i+shift_iy], histtype='step', bins=nbin)
            else:
                plt.hist(all_Mkks[:,i+shift_ix, i+shift_iy], histtype='step', bins=nbin)

            if i==0:
                mean_label = 'Mean'
                median_label = 'Median'
            else:
                mean_label = None
                median_label = None

            if all_Mkks is None:
                plt.axvline(np.mean(self.Mkk.all_Mkks[:,i+shift_ix,i+shift_iy]), linestyle='solid', label=mean_label, color='C1')
                plt.axvline(np.median(self.Mkk.all_Mkks[:,i+shift_ix,i+shift_iy]), linestyle='dashed', label=median_label, color='C1')
            
            else:
                plt.axvline(np.mean(all_Mkks[:,i+shift_ix,i+shift_iy]), linestyle='solid', label=mean_label, color='C1')
                plt.axvline(np.median(all_Mkks[:,i+shift_ix,i+shift_iy]), linestyle='dashed', label=median_label, color='C1')
   
            plt.ylabel('Number of realizations')
            if i==0:
                plt.legend(frameon=False, fontsize=labelfontsize)

        
        plt.tight_layout()
        
        if show:
            plt.show()
            
        if return_fig:
            return f
            
        
    
    def plot_true_masked_rect_cls(self, cl_true, cl_masked, cl_rect, ell_bins=None, title=None, show=True, return_fig=False, \
                                 titlefontsize=16, ylogscale=False, labelfontsize=18, show_dcl=False):
        
        f = plt.figure(figsize=(8,6))
        if title is not None:
            plt.title(title, fontsize=titlefontisze)
            
        if ell_bins is None:
            xvals = self.Mkk.midbin_ell
        else:
            xvals = ell_bins
            
        plt.plot(xvals, cl_true, label='True', marker='.', color='k', linewidth=2)
        plt.plot(xvals, cl_masked, label='Masked', marker='.', color='g', linewidth=2)
        plt.plot(xvals, cl_rect, label='Rectified', marker='.', color='b', linewidth=2)
        
        if show_dcl:
            plt.plot(xvals, cl_true-cl_rect, label='True-Rectified', marker='.',linestyle='dashed', color='b', linewidth=2)
            
        
        plt.legend(fontsize=14)
        plt.xscale('log')
        
        plt.ylabel('$C_{\\ell}$', fontsize=labelfontsize)
        plt.xlabel('Multipole $\\ell$', fontsize=labelfontsize)
        
        if ylogscale:
            plt.yscale('log')
        
            
        if show:
            plt.show()
            
        if return_fig:
            return f
        
    
    def plot_delta_cls(self, delta_cls, c_ell_rect=None, logclslop=None, ell_bins=None, return_fig=False, analytic_dcl=None, fractional_dcl=True, absolute_deviation=False,\
                       title=None, titlefontsize=16, labelfontsize=18, show=True, labels=None, ylogscale=False, xlogscale=True, nphase_realizations=None, median_or_mean='median', \
                       knox_fudge=1., ylim=None, ylabel=None):
        
        
        f = plt.figure(figsize=(8,6))
        if title is not None:
            plt.title(title, fontsize=titlefontsize)
            
        
        if ell_bins is None:
            xvals = self.Mkk.midbin_ell
        else:
            xvals = ell_bins
            
        if len(delta_cls)<10:
            for i, delta_clz in enumerate(delta_cls):
                if absolute_deviation:
                    delta_clz = np.abs(delta_clz)
                    
                if median_or_mean=='median':
                    median_dcl = np.median(delta_clz, axis=0)
                elif median_or_mean=='mean':
                    median_dcl = np.mean(delta_clz, axis=0)

                plt.errorbar(xvals, median_dcl, yerr=[median_dcl-np.percentile(delta_clz, 16, axis=0), np.percentile(delta_clz, 84, axis=0)-median_dcl], label=labels[i], marker='.', alpha=np.sqrt(1./len(delta_cls)), capsize=5, capthick=2, linewidth=2, markersize=10)
                    
        else:
            if absolute_deviation:
                delta_cls = np.abs(delta_cls)

            if median_or_mean=='median':
                median_dcl = np.median(delta_cls, axis=0)
            elif median_or_mean=='mean':
                median_dcl = np.mean(delta_cls, axis=0)
                
            
            if len(xvals) > 50:
                plt.fill_between(xvals, np.percentile(delta_cls, 16, axis=0), np.percentile(delta_cls, 84, axis=0), alpha=0.3, color='b')
                plt.plot(xvals, median_dcl, marker='.', label=median_or_mean, linewidth=2, markersize=10)
            else:
                plt.errorbar(xvals, median_dcl, yerr=[median_dcl-np.percentile(delta_cls, 16, axis=0), np.percentile(delta_cls, 84, axis=0)-median_dcl], label=median_or_mean, marker='.', capsize=5, capthick=2, linewidth=2, markersize=10)
        
        
        if c_ell_rect is not None and logclslop is not None:
            print('Computing Knox errors..')
            
            analytic_dcl = self.compute_knox_error_dcl(logclslop=logclslop, c_ell_rect=c_ell_rect)
            plt.plot(xvals, np.median(analytic_dcl, axis=0)/knox_fudge, label='Knox', color='k', linewidth=3)
        
        if ylabel is None:
            
            if fractional_dcl:
                if absolute_deviation:
                    ylabel = '$\\langle \\left| \\frac{C_{\ell} - C_{\ell}^{recon}}{C_{\ell}} \\right| \\rangle $'
                else:
                    ylabel = '$\\langle \\frac{C_{\ell} - C_{\ell}^{recon}}{C_{\ell}} \\rangle$'
            else:
                if absolute_deviation:
                    ylabel = '$\\langle \\left| C_{\ell} - C_{\ell}^{recon} \\right| \\rangle$'
                else:
                    ylabel = '$ \\langle C_{\\ell}^{unmasked} - \\ell^{1.0} \\rangle$'

        plt.legend(fontsize=14)  
        plt.ylabel(ylabel, fontsize=labelfontsize+4)
        plt.xlabel('Multipole $\\ell$', fontsize=labelfontsize)
        
        if xlogscale:
            plt.xscale('log')
        
        if ylogscale:
            plt.yscale('log')
        
        plt.ylim(ylim)
        
        if show:
            plt.show()
            
        if return_fig:
            return f

def get_Mkk_kdes(all_Mkks, nbin=50, nfine=1000):
    
    bin_minmax = np.zeros(shape=(all_Mkks.shape[1], all_Mkks.shape[2], 2))
    Mkk_kdes = [[[] for i in range(all_Mkks.shape[1])] for j in range(all_Mkks.shape[2])]
    
    for i in range(all_Mkks.shape[1]):
        
        
        for j in range(all_Mkks.shape[2]):
            
            bin_minmax[i,j,:] = [0.5*np.min(all_Mkks[:,i,j]), np.max(all_Mkks[:,i,j])]
            Mkk_kdes[i][j] = scipy.stats.gaussian_kde(all_Mkks[:,i,j], bw_method=0.2)
                            
    return Mkk_kdes, bin_minmax


def kdes_to_fine_pdf_grid(Mkk_kdes, bin_minmax, nfine=100):

    fine_bins = np.zeros(shape=(len(Mkk_kdes), len(Mkk_kdes[0]), nfine))

    fine_pdfs = np.zeros(shape=(len(Mkk_kdes), len(Mkk_kdes[0]), nfine))
    
    
    for i in range(len(Mkk_kdes)):
        for j in range(len(Mkk_kdes[0])):
            
            fine_bins[i,j] = np.linspace(bin_minmax[i,j,0], bin_minmax[i,j,1], nfine)
            fine_pdfs[i,j] = Mkk_kdes[i][j](fine_bins[i,j])
            fine_pdfs[i,j] /= np.sum(fine_pdfs[i,j])
            
            
        
    return fine_bins, fine_pdfs

def sample_Mkk(Mkk_kdes, bin_minmax, nsamp=10, nfine=100):
    

    sampled_mkks = []
    fine_bins, fine_pdfs = kdes_to_fine_pdf_grid(Mkk_kdes, bin_minmax, nfine=nfine)

    for n in np.arange(nsamp):
        
        mkk = np.zeros((bin_minmax.shape[0], bin_minmax.shape[1]))
        
        for i in range(bin_minmax.shape[0]):
            for j in range(bin_minmax.shape[1]):
                
                mkk[i,j] = np.random.choice(fine_bins[i,j], p=fine_pdfs[i,j])
          
        sampled_mkks.append(mkk)
        
        plot_mkk_matrix(mkk, inverse=False, logscale=True)

    return sampled_mkks
