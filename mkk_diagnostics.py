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
    
    def __init__(Mkk=None):
        
        if Mkk is not None:
            self.Mkk = Mkk
            
            self.mkk_inv = compute_inverse_mkk(self.Mkk.av_Mkk)
            
    def load_mkk_inv(self, mkk_inv):

        self.mkk_inv = mkk_inv

             
    def compute_true_minus_rectified_c_ell(self, allmaps, mask=None, ensemble_fft=False, n_split=2, sub_bins=False, return_cls=False):
        
        delta_cl = []
        all_cl_true, all_cl_rect, all_cl_masked = [], [], []
        
        n_realizations = len(allmaps)
        
        if not ensemble_fft:
            for i in range(n_realizations):
                single_map = allmaps[i]-np.mean(allmaps[i])
                cl_true = self.Mkk.compute_cl_indiv(single_map)
                cl_masked = self.Mkk.compute_cl_indiv(single_map*self.Mkk.mask)
                c_ell_rectified = np.dot(self.mkk_inv.transpose(), cl_masked)

                delta_cl.append((cl_true-c_ell_rectified)/cl_true)
                
        else:
            
            if self.Mkk.fft_objs is None:
                
                maplist_split_shape = (n_realizations//n_split, self.Mkk.dimx, self.Mkk.dimy)
        
                self.Mkk.empty_aligned_objs, self.Mkk.fft_objs = construct_pyfftw_objs(3, maplist_split_shape)
            
            for n in range(n_split):
                
                self.Mkk.empty_aligned_objs[1][:n_realizations//n_split, :, :] = allmaps[n*n_realizations//n_split :(n+1)*n_realizations//n_split ]
                
                cl_true = self.Mkk.get_ensemble_angular_autospec(nsims=n_realizations//n_split, sub_bins=sub_bins, apply_mask=False)
                cl_masked = self.Mkk.get_ensemble_angular_autospec(nsims=n_realizations//n_split, sub_bins=sub_bins, apply_mask=True)
                
                for i in range(len(cl_masked)):
                    cl_rect = np.dot(self.mkk_inv.transpose(), cl_masked[i])
                    
                    if sub_bins:
                        cl_rect = np.array([np.mean(cl_rect[self.Mkk.correspond_bins[k]]) for k in range(len(self.Mkk.correspond_bins))])
   
                    delta_cl.append((cl_true[i]-cl_rect[i])/cl_true[i])
        
                    if return_cls:
                        all_cl_true.append(cl_true[i])
                        all_cl_rect.append(cl_rect[i])
                        all_cl_masked.append(cl_masked[i])


        if return_cls:
            return delta_cl, all_cl_true, all_cl_masked, all_cl_rect
        
        return delta_cl
    