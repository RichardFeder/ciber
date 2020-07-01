import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fftshift as fftshift
from numpy.fft import ifftshift as ifftshift
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
from mkk_parallel import *
import pyfftw


class Mkk_diag():
    ''' This is a class that will contain the various diagnostic tests we want to run on the Mkk method. It uses the Mkk_bare class object
    fairly extensively, i.e. it generally assumes you've already used Mkk_bare() to construct an Mkk matrix. This includes some of the fast FFT
    objects that are constructed in Mkk_bare, though there are options to evaluate the power spectrum of maps one by one. '''
    
    
    mkk_inv = None
    
    def __init__(self, Mkk=None):
        
        if Mkk is not None:
            self.Mkk = Mkk
            self.all_Mkks = self.Mkk.all_Mkks
            self.av_Mkk = self.Mkk.av_Mkk
            self.mkk_inv = compute_inverse_mkk(self.Mkk.av_Mkk)
            
    def load_mkk_inv(self, mkk_inv):

        self.mkk_inv = mkk_inv


             
    def compute_true_minus_rectified_c_ell(self, allmaps, mask=None, ensemble_fft=False, n_split=2, sub_bins=False, return_cls=False, fractional_dcl=True, \
                                        n_phase_realizations=None, mean_or_median='median'):
        
        delta_cl = []
        all_cl_true, all_cl_rect, all_cl_masked = [], [], []
        
        n_realizations = len(allmaps)
        print("n_realizations is ", n_realizations)

        if n_phase_realizations is not None:
            print('using ', n_phase_realizations, 'phase realizations for the Mkk matrix')
            mkk_mat = self.all_Mkks[:n_phase_realizations]
            
            if mean_or_median=='mean':
                mkk_inv_mat = compute_inverse_mkk(np.mean(mkk_mat, axis=0))
            elif mean_or_median=='median':
                mkk_inv_mat = compute_inverse_mkk(np.median(mkk_mat, axis=0))

        else:
            mkk_inv_mat = self.mkk_inv
        
        if not ensemble_fft:
            for i in range(n_realizations):
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
            
            if self.Mkk.fft_objs is None:
                
                maplist_split_shape = (n_realizations//n_split, self.Mkk.dimx, self.Mkk.dimy)
        
                self.Mkk.empty_aligned_objs, self.Mkk.fft_objs = construct_pyfftw_objs(3, maplist_split_shape)
            
            for n in range(n_split):
                self.Mkk.empty_aligned_objs[1][:n_realizations//n_split, :, :] = allmaps[n*n_realizations//n_split :(n+1)*n_realizations//n_split ]
                
                cl_true = self.Mkk.get_ensemble_angular_autospec(nsims=n_realizations//n_split, sub_bins=sub_bins, apply_mask=False)
                cl_masked = self.Mkk.get_ensemble_angular_autospec(nsims=n_realizations//n_split, sub_bins=sub_bins, apply_mask=True)
                
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
    
    
    def compute_knox_error_dcl(self):
        fmask = np.sum(self.Mkk.mask)/self.Mkk.mask.shape[0]**2    
        dcl_cl = np.sqrt(2./((2.*self.Mkk.midbin_ell+1.)*self.Mkk.delta_ell*fmask*self.Mkk.fsky))
        return dcl_cl
    
    def plot_distribution_of_Mkk_elements(self, elements='diag', nbin=20, suptitlefontsize=20, suptitle_y=1.03, titlefontsize=16, labelfontsize=12,\
                                           return_fig=False, show=True):
        
#         M_string = '$M_{i,j}$'
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
            
        plt.figure(figsize=(12,6))
        plt.suptitle(suptitle, fontsize=suptitlefontsize, y=suptitle_y)
    
        
        for i in range(10):
            
            plt.subplot(2,5,i+1)
            plt.title('$i='+str(i+shift_ix+1)+', j='+str(i+shift_iy+1)+'$', fontsize=16)
            plt.hist(self.Mkk.all_Mkks[:,i+shift_ix,i+shift_iy], histtype='step', bins=nbin)

            if i==0:
                mean_label = 'Mean'
                median_label = 'Median'
            else:
                mean_label = None
                median_label = None

            plt.axvline(np.mean(self.Mkk.all_Mkks[:,i+shift_ix,i+shift_iy]), linestyle='solid', label=mean_label, color='C1')
            plt.axvline(np.median(self.Mkk.all_Mkks[:,i+shift_ix,i+shift_iy]), linestyle='dashed', label=median_label, color='C1')
            plt.ylabel('Number of realizations')

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
        
    
    def plot_delta_cls(self, delta_cls, ell_bins=None, return_fig=False, analytic_dcl=None, fractional_dcl=True, absolute_deviation=False,\
                       title=None, titlefontsize=16, labelfontsize=18, show=True, labels=None, ylogscale=False, nphase_realizations=None):
        
        
        f = plt.figure(figsize=(8,6))
        if title is not None:
            plt.title(title, fontsize=titlefontsize)
            
        
        if ell_bins is None:
            xvals = self.Mkk.midbin_ell
        else:
            xvals = ell_bins
            
        if isinstance(delta_cls, list):
            for i, delta_clz in enumerate(delta_cls):
                
                if absolute_deviation:
                    delta_clz = np.abs(delta_clz)
                    
                median_dcl = np.median(delta_clz, axis=0)
                plt.errorbar(xvals, median_dcl, yerr=[median_dcl-np.percentile(delta_clz, 16, axis=0), np.percentile(delta_clz, 84, axis=0)-median_dcl], label=labels[i], marker='.', alpha=np.sqrt(1./len(delta_cls)), capsize=5, capthick=2, linewidth=2, markersize=10)
                    
        else:
            if absolute_deviation:
                delta_cls = np.abs(delta_cls)
                
            median_dcl = np.median(delta_cls, axis=0)
            plt.errorbar(xvals, median_dcl, yerr=[median_dcl-np.percentile(delta_cls, 16, axis=0), np.percentile(delta_cls, 84, axis=0)-median_dcl], marker='.', capsize=5, capthick=2, linewidth=2, markersize=10)
        
        
        if fractional_dcl and absolute_deviation and nphase_realizations is not None:
            print('Computing Knox errors..')
            analytic_dcl = self.compute_knox_error_dcl()
            plt.plot(xvals, analytic_dcl, label='Knox', color='k', linewidth=3)
        
        
        if fractional_dcl:
            if absolute_deviation:
                ylabel = '$\\left|\\langle \\frac{C_{\ell} - C_{\ell}^{recon}}{C_{\ell}} \\rangle \\right|$'
            else:
                ylabel = '$\\langle \\frac{C_{\ell} - C_{\ell}^{recon}}{C_{\ell}} \\rangle$'
        else:
            if absolute_deviation:
                ylabel = '$\\left| \\langle C_{\ell} - C_{\ell}^{recon} \\rangle \\right|$'
            else:
                ylabel = '$ \\langle C_{\ell} - C_{\ell}^{recon} \\rangle$'

        plt.legend(fontsize=14)  
        plt.ylabel(ylabel, fontsize=labelfontsize)
        plt.xlabel('Multipole $\\ell$', fontsize=labelfontsize)
        plt.xscale('log')
        
        if ylogscale:
            plt.yscale('log')
        
        if show:
            plt.show()
            
        if return_fig:
            return f
    