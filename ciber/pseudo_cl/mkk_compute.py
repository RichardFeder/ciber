import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fftshift as fftshift
from numpy.fft import ifftshift as ifftshift
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
import multiprocessing
from multiprocessing import Pool
import pyfftw
import time
from ciber.processing.filtering import fit_gradient_to_map, precomp_gradient_dat, fit_gradient_to_map_precomp, precomp_offset_gradient, offset_gradient_fit_precomp
# from ciber.plotting.plot_utils import plot_map
from ciber.processing.numerical import *
from ciber.processing.fourier_bkg import *


def allocate_fftw_memory(data_shape, n_blocks=2):

    ''' This function allocates empty aligned memory to be used for declaring FFT objects
    by pyfftw. The shape of the data, i.e. data_shape (including number of realizations as a dimension),
    can be arbitrarily defined depending on the task at hand.'''

    empty_aligned = []
    for i in range(n_blocks):
        empty_aligned.append(pyfftw.empty_aligned(data_shape, dtype='complex64'))

    return empty_aligned

# def set_fftw_objects(empty_aligned=None, directions=['FFTW_BACKWARD'], threads=1, axes=(1,2), data_shape=None):
    
#     if empty_aligned is None:
#         empty_aligned = allocate_fftw_memory(data_shape, n_blocks=len(directions)+1)

#     fftw_object_dict = dict({})

#     for i in range(len(empty_aligned)-1):
#         fftw_obj = pyfftw.FFTW(empty_aligned[i], empty_aligned[i+1], axes=axes, threads=threads, direction=directions[i], flags=('FFTW_MEASURE',))
#         fftw_object_dict[directions[i]] = fftw_obj

#     return fftw_object_dict, empty_aligned

def set_fftw_objects(empty_aligned=None, directions=['FFTW_BACKWARD'], threads=None, axes=(1,2), data_shape=None):
    import multiprocessing
    if threads is None:
        threads = multiprocessing.cpu_count()  # all cores

    if empty_aligned is None:
        empty_aligned = allocate_fftw_memory(data_shape, n_blocks=len(directions)+1)

    fftw_object_dict = {}
    for direction in directions:
        fft_obj = pyfftw.FFTW(empty_aligned[0], empty_aligned[1],
                              direction=direction,
                              axes=axes,
                              flags=('FFTW_MEASURE',),  # <-- better planning
                              threads=threads)
        fftw_object_dict[direction] = fft_obj

        # Warm-up
        fft_obj()

    return empty_aligned, fftw_object_dict


def allocate_empty_aligned_pyfftw(dat_shape, dtype='complex64'):
    return pyfftw.empty_aligned(dat_shape, dtype=dtype)

def apply_norm_mask(empty_aligned_objs, obj_idx, mean_normalizations, masks, plot=False):
    # I think this works as an inplace operation
    # print('adding normalization and applying mask to phase realization.. obj_idx = ', obj_idx)
    if plot:
        plot_map(np.array(empty_aligned_objs[obj_idx][0,0].real), title='before mean norm/mask..')

    for k in range(len(mean_normalizations)):
        empty_aligned_objs[obj_idx][k].real += mean_normalizations[k]
        empty_aligned_objs[obj_idx][k].real *= masks[k]
    if plot:
        plot_map(np.array(empty_aligned_objs[obj_idx][0,0].real), title='np.array(empty_aligned_objs[obj_idx][0,0].real)')
    
    return empty_aligned_objs
    
def apply_weighted_ffnorm(k, all_notkranges, empty_aligned_objs, sum_ff_weight_ims,\
                          obj_idx, ffnorm_facs):
    if k==0:
        for kp in all_notkranges[k]:
            empty_aligned_objs[obj_idx][kp] *= ffnorm_facs[kp]
    else:
        empty_aligned_objs[obj_idx][k-1] *= ffnorm_facs[k-1]

    ff_estimate = np.sum(empty_aligned_objs[obj_idx][all_notkranges[k]].real, axis=0)/sum_ff_weight_ims
    ff_estimate[np.isnan(ff_estimate)] = 1.     
    return ff_estimate
    
def clock_log(message=None, t0=0., verbose=False):
    t1 = time.time()
    if message is not None and verbose:
        print(message, t1 - t0)

    return t1

def construct_pyfftw_objs(n_obj, dat_shape, dtype='complex64', axes=(1,2), threads=1, directions=['FFTW_BACKWARD', 'FFTW_FORWARD']):
    
    fft_objs = []
    empty_aligned_objs = []
    
    for i in range(n_obj):
        empty_aligned_objs.append(allocate_empty_aligned_pyfftw(dat_shape, dtype=dtype))
    
    for j in range(n_obj-1):
        
        fft_obj = pyfftw.FFTW(empty_aligned_objs[j], empty_aligned_objs[j+1], axes=axes, threads=threads, direction=directions[j], flags=('FFTW_MEASURE', ))
    
        fft_objs.append(fft_obj)
         
    return empty_aligned_objs, fft_objs


import multiprocessing

# def construct_pyfftw_objs_gen(ndim, shape, axes, n_fft_obj=2,
#                               fft_idx0_list=None, fft_idx1_list=None):
#     n_threads = multiprocessing.cpu_count()  # or tune manually

#     empty_aligned_objs = []
#     fft_objs = []

#     for i in range(n_fft_obj):
#         # Allocate aligned arrays
#         a = pyfftw.empty_aligned(shape, dtype='complex128')
#         b = pyfftw.empty_aligned(shape, dtype='complex128')

#         # Create FFT plan with MEASURE and threading
#         fft_forward = pyfftw.FFTW(a, b, direction='FFTW_FORWARD',
#                                   axes=axes,
#                                   flags=('FFTW_MEASURE',),
#                                   threads=n_threads)

#         fft_inverse = pyfftw.FFTW(b, a, direction='FFTW_BACKWARD',
#                                   axes=axes,
#                                   flags=('FFTW_MEASURE',),
#                                   threads=n_threads)

#         # Warm-up
#         fft_forward()
#         fft_inverse()

#         empty_aligned_objs.append((a, b))
#         fft_objs.append((fft_forward, fft_inverse))

#     return empty_aligned_objs, fft_objs

import multiprocessing

def construct_pyfftw_objs_gen(n_obj, dat_shape, dtype='complex64',
             axes=(1,2), threads=1, directions=['FFTW_BACKWARD', 'FFTW_FORWARD'], \
    n_fft_obj=None, fft_idx0_list=None, fft_idx1_list=None):
    
    fft_objs, empty_aligned_objs = [], []

    # If threads not specified, use all available CPU cores
    if threads is None:
        threads = multiprocessing.cpu_count()

    if n_fft_obj is None:
        n_fft_obj = n_obj-1
    if fft_idx0_list is None:
        fft_idx0_list = np.arange(n_obj-1)
    if fft_idx1_list is None:
        fft_idx1_list = np.arange(n_obj-1)+1
    n_fft_obj = len(fft_idx0_list)

    for i in range(n_obj):
        empty_aligned_objs.append(allocate_empty_aligned_pyfftw(dat_shape, dtype=dtype))
    
    for j in range(n_fft_obj):
        fft_obj = pyfftw.FFTW(empty_aligned_objs[fft_idx0_list[j]],
                             empty_aligned_objs[fft_idx1_list[j]],
                              axes=axes,
                               threads=threads,
                                direction=directions[j],
                                 flags=('FFTW_MEASURE', ))
        fft_objs.append(fft_obj)

        fft_obj()

    return empty_aligned_objs, fft_objs


def compute_inverse_mkk(mkk):
    
    inverse_mkk = np.linalg.inv(mkk)
    
    return inverse_mkk


def group_idxs(idx_list, imdim=1024):
    
    grouped_idx_list = []    
    used_list = []
    
    for idx in idx_list:
        for idx2 in idx_list:
            if idx2[0]%imdim ==(imdim-idx[0])%imdim and idx2[1]%imdim==(imdim-idx[1])%imdim and idx2 not in used_list:
                grouped_idx_list.append([idx, idx2])
                used_list.append(idx)
                used_list.append(idx2)

    return grouped_idx_list


def init_fftobj_info(cross_mkk=False):
    fft_idx0_list, fft_idx1_list, directions  = [0, 2], [1, 3], ['FFTW_BACKWARD', 'FFTW_FORWARD']
    n_obj, n_fft_obj = 4, 2
    if cross_mkk:
        fft_idx0_list.append(5)
        fft_idx1_list.append(6)
        directions.append('FFTW_FORWARD')
        n_obj += 3 # [4, 5, 6]
        n_fft_obj += 1
        
    info_list = [fft_idx0_list, fft_idx1_list, directions, n_obj, n_fft_obj]
    print(info_list)
    return info_list

# numerical_routines.py
# def precomp_gradsub(masks, dimx, dimy):
# def perform_grad_sub(k, empty_aligned_objs, mask, obj_idx, dot1s, Xgrad, mask_rav, nsims_perf):



class Mkk_bare():
    
    def __init__(self, pixsize=7., dimx=1024, dimy=1024, ell_min=180., logbin=True, nbins=20, n_fine_bins=0, precompute=False):
        
        for attr, valu in locals().items():
            setattr(self, attr, valu)
            
        self.pixlength_in_deg = self.pixsize/3600. # pixel angle measured in degrees
        self.ell_max = 2*180./self.pixlength_in_deg # maximum multipole determined by pixel size
        
        self.arcsec_pp_to_radian = self.pixlength_in_deg*(np.pi/180.) # conversion from arcsecond per pixel to radian per pixel
        
        self.sqdeg_in_map = self.dimx*self.dimy*self.pixlength_in_deg**2 # area of map in square degrees
        self.fsky = self.sqdeg_in_map/(4*np.pi*(180./np.pi)**2) # fraction of sky covered by map
        
        self.ell_map = None # used to compute ring masks in fourier space
        self.binl = None # radially averaged multipole bins
        self.weights = None # fourier weights for angular power spectrum calculation
        self.ringmasks = None # 2D multipole masks for each bin
        self.all_Mkks = None # this can be used if one wants all phase realizations of the Mkk matrix that are usually averaged down
        self.av_Mkk = None # average Mkk matrix for many realizations
        self.empty_aligned_objs = None # used for fast fourier transforms through pyfftw
        self.fft_objs = None # used for fast fourier transforms through pyfftw

        if precompute:
            self.precompute_mkk_quantities(precompute_all=True)



    def set_bin_edges(self, binl):
        self.binl = binl

    def get_mkk_sim(self, mask, nsims, show_tone_map=False, mode='auto', n_split=1, threads=1, \
                    print_timestats=False, return_all_Mkks=False, n_fine_bins=None, sub_bins=False, store_Mkks=True, precompute_all=False, \
                    input_cl_func=None, quadoff_grad=False, fc_sub=False, fc_sub_quad_offset=False, fc_sub_n_terms=2, grad_sub=False):
        
        ''' This is the main function that computes an estimate of the mode coupling matrix Mkk from nsims generated tone maps. 
        Knox errors are also computed based on the mask/number of modes/etc. (note: should this also include a beam factor?) This implementation is more memory intensive,
        which makes it ultimately faster. Because we are obtaining a Monte Carlo estimate from randomly drawn phases, the task is also fairly parallelizable. We pre-compute a number of quantities, 
        including the noise realizations. Rather than compute (nsims x number of multipole bins) number of noise realizations, we take advantage of the ring masks being disjoint
        in image space. This means that we can draw just (nsims) number of realizations, save them, and apply all masks to the same one set of realizations. This implementation also attempts to 
        be efficient in terms of memory allocation, restricting the flow of FFTs to certain regions of memory. The memory usage for 20 x 100 x 1024 x 1024 dtype complex64 arrays 
        that are needed throughout is 4 Gb, which can lead to a siginficant slowdown in speed because of the lack of available RAM in a machine. For this reason, the user can specify
        how many sequential tasks to spread out the memory usage over different iterations. The amount of memory needed using this implementation scales linearly with nsims, dimx and dimy. 
        
        Fast version on 2-cores intel i5 (4 threads)
        20 bins:
            10 sims -> 13 s
            20 sims -> 26 s
            30 sims -> 33 s
            40 sims -> 45 s
            50 sims -> 56 s
            80 sims -> 180 s

        With splitting:
        100 @ 50 each -> 100 s

        Fast version on 6-cores intel i7 (12 threads)

        20 bins:
            10 sims -> 11.6 s
            20 sims -> 18 s
            30 sims -> 25 s
            40 sims -> 33 s
            50 sims -> 41 s
            80 sims -> 63 s
            100 sims -> 80 s

        '''
        print('Number of CPU threads available:', multiprocessing.cpu_count())


        self.print_timestats = print_timestats
        
        if n_fine_bins is not None:
            self.n_fine_bins = n_fine_bins
            
        if n_split > 1:
            print('Splitting up computation of '+str(nsims)+' simulations into '+str(n_split)+' chunks..')
            assert (nsims % n_split) == 0

        self.mask = mask
        
        maplist_split_shape = (nsims//n_split, self.dimx, self.dimy)
        
        self.empty_aligned_objs, self.fft_objs = construct_pyfftw_objs(3, maplist_split_shape, threads=threads)
       
        Mkks, dC_ell_list = [], []
        
        # lets precompute a few things that are used later..
        if precompute_all:
            self.precompute_mkk_quantities(precompute_all=True, shift=True, sub_bins=sub_bins, input_cl_func=input_cl_func)
        
        # this is for measuring individual 2D fourier modes for some number of bins 
        if sub_bins and self.n_fine_bins > 0:
            n_total_bins = len(self.unshifted_sub_ringmasks)
        else:
            n_total_bins = len(self.unshifted_ringmasks)
        
        self.n_total_bins = n_total_bins

        if return_all_Mkks or store_Mkks:
            all_Mkks = np.zeros((nsims, n_total_bins, n_total_bins))


        if quadoff_grad:

            dot1, X, mask_rav = precomp_offset_gradient(self.dimx, self.dimy, mask=mask)

        elif fc_sub:
            dot1, X, mask_rav = precomp_fourier_templates(self.dimx, self.dimy, mask=mask, quad_offset=fc_sub_quad_offset, \
                                                         n_terms=fc_sub_n_terms, with_gradient=grad_sub)   
            

        for i in range(n_split):
            print('Split '+str(i+1)+' of '+str(n_split))

            # compute noise realizations beforehand so time isn't wasted generating a whole new set each time. We can do this because the ring masks
            # that we multiply by the noise realizations are spatially disjoint, i.e. the union of all of the Fourier masks is the null set. 
            self.noise = np.random.normal(size=maplist_split_shape)+ 1j*np.random.normal(size=maplist_split_shape)
            
            Mkk = np.zeros((n_total_bins, n_total_bins))
            
            for j in range(n_total_bins):
                print('band ', j)
                
                self.map_from_phase_realization(j, nsims=nsims//n_split, sub_bins=sub_bins)
                
                if show_tone_map:
                    for idx, idx_list in enumerate(self.correspond_bins):
                        if j in idx_list:
                            title_idx = idx
                            break
                                                
                    self.plot_tone_map_realization(self.empty_aligned_objs[1][0].real, idx=title_idx)


                if quadoff_grad:
                    print('subtracting quadrant offsets and gradient')
                    recov_offset_grads = np.array([offset_gradient_fit_precomp(self.empty_aligned_objs[1][s].real, dot1, X, mask_rav)[1] for s in range(maplist_split_shape[0])])
                    self.empty_aligned_objs[1].real -= np.array([mask*recov_offset_grads[s] for s in range(maplist_split_shape[0])])
                elif fc_sub:
                    recov_offset_fcs = np.array([fc_fit_precomp(self.empty_aligned_objs[1][s].real, dot1, X, mask_rav)[1] for s in range(maplist_split_shape[0])])
                    self.empty_aligned_objs[1].real -= np.array([mask*recov_offset_fcs[s] for s in range(maplist_split_shape[0])])

                if mode=='auto':
                    masked_Cl = self.get_ensemble_angular_autospec(nsims=nsims//n_split, sub_bins=sub_bins, apply_mask=True)

                    if return_all_Mkks or store_Mkks:
                        all_Mkks[i*nsims//n_split :(i+1)*nsims//n_split ,j,:] = np.array(masked_Cl)

                    # row entry for Mkk is average over realizations
                    Mkk[j,:] = np.mean(np.array(masked_Cl), axis=0)

            Mkks.append(Mkk)

        if store_Mkks:
            self.all_Mkks = all_Mkks

        if return_all_Mkks:
            return all_Mkks

        if n_split == 1:
            av_Mkk = Mkks[0]
        else:
            av_Mkk = np.mean(np.array(Mkks), axis=0)

        if store_Mkks:
            self.av_Mkk = av_Mkk

        return av_Mkk
    
    
    def map_from_phase_realization(self, idx=0, shift=True, nsims=1, mode=None, sub_bins=False, pix_size=7.):

        ''' This makes maps very quickly given some precomputed quantities. The unshifted ringmasks are used to generate
        the pure tones within each band, so then this gets multiplied by the precalculated noise and fourier transformed to real space. These
        maps are then stored in the variable self.c (self.fft_object_b goes from self.b --> self.c)'''
    
        if self.ell_map is None:
            self.get_ell_map(shift=shift, pix_size=pix_size)
            
        if self.binl is None:
            self.compute_multipole_bins()
         
        if mode == 'white':
            self.fft_objs[0](self.noise)
        else:
            if sub_bins and self.n_fine_bins > 0:
                self.fft_objs[0](self.unshifted_sub_ringmasks[idx]*self.noise)
            else:
                self.fft_objs[0](self.unshifted_ringmasks[idx]*self.noise)


    def get_ensemble_angular_autospec_ndim(self, nsims=2, sub_bins=False, apply_mask=False, mask=None, fft_obj_idx=1, obj_idx0=None, obj_idx1=None, \
                                            obj_idx0_cross=None, obj_idx1_cross=None, fft_obj_idx_cross=None):
        
        if obj_idx0 is None:
            obj_idx0 = 1
        if obj_idx1 is None:
            obj_idx1 = 2

        cross_Mkk=False
        if obj_idx0_cross is not None and obj_idx1_cross is not None and fft_obj_idx_cross is not None:
            cross_Mkk = True

        if apply_mask:
            if mask is None:
                mask = self.mask
            self.empty_aligned_objs[obj_idx0] *= mask

        eao_shape = self.empty_aligned_objs[obj_idx0].shape

        ndim_eao = len(eao_shape)

        self.fft_objs[fft_obj_idx](self.empty_aligned_objs[obj_idx0].real)
        if cross_Mkk:
            self.fft_objs[fft_obj_idx_cross](self.empty_aligned_objs[obj_idx0_cross].real)
        
        if ndim_eao==3:
            c_ell_shape = (nsims, len(self.binl)-1)
            if not cross_Mkk:
                fftsq = [(dentry*np.conj(dentry)).real for dentry in self.empty_aligned_objs[obj_idx1]]
            else:
                fftsq = [(dentry*np.conj(dentry_cross)).real for dentry, dentry_cross in zip(self.empty_aligned_objs[obj_idx1], self.empty_aligned_objs[obj_idx1_cross])]

        else:

            c_ell_shape = (eao_shape[0]*eao_shape[1], len(self.binl)-1)
            fftsq = []

            if cross_Mkk:
                for perf_dentry, perf_dentry_cross in zip(self.empty_aligned_objs[obj_idx1], self.empty_aligned_objs[obj_idx1_cross]):
                    fftsq.extend([(dentry*np.conj(dentry_cross)).real for dentry, dentry_cross in zip(perf_dentry, perf_dentry_cross)])
            else:
                for perf_dentry in self.empty_aligned_objs[obj_idx1]:
                    fftsq.extend([(dentry*np.conj(dentry)).real for dentry in perf_dentry])
            

        if sub_bins and self.n_fine_bins > 0:
            C_ell = np.zeros((nsims, self.n_total_bins))
            for i in range(self.n_total_bins):
                if self.sub_ringmask_sums[i] > 0:
                    C_ell[:,i] = np.array([np.sum(self.sub_masked_weights[i]*fftsq[j][self.sub_ringmasks[i]])/self.sub_masked_weight_sums[i] for j in range(nsims)])

        else:
            
            C_ell = self.bandpower_average_powerspec(c_ell_shape, fftsq)
            # C_ell = np.zeros(c_ell_shape)
            # for i in range(len(self.binl)-1):
            #     if self.ringmask_sums[i] > 0:
            #         C_ell[:,i] = np.array([np.sum(self.masked_weights[i]*fftsq[j][self.ringmasks[i]])/self.masked_weight_sums[i] for j in range(nsims)])

            # return C_ell, 
        return C_ell
    

    def bandpower_average_powerspec(self, c_ell_shape, fftsq):
        C_ell = np.zeros(c_ell_shape)
        nsims = len(fftsq)
        for i in range(len(self.binl)-1):
            if self.ringmask_sums[i] > 0:
                C_ell[:,i] = np.array([np.sum(self.masked_weights[i]*fftsq[j][self.ringmasks[i]])/self.masked_weight_sums[i] for j in range(nsims)])
        return C_ell

    def get_ensemble_angular_autospec(self, nsims=2, sub_bins=False, apply_mask=False, obj_idx0=None, obj_idx1=None):
                 
        if obj_idx0 is None:
            obj_idx0 = 1
        if obj_idx1 is None:
            obj_idx1 = 2

        if apply_mask:
            self.empty_aligned_objs[obj_idx0] *= self.mask

        self.fft_objs[1](self.empty_aligned_objs[obj_idx0].real)
        
        fftsq = [(dentry*np.conj(dentry)).real for dentry in self.empty_aligned_objs[obj_idx1]]
            
        if sub_bins and self.n_fine_bins > 0:
            C_ell = np.zeros((nsims, self.n_total_bins))
            for i in range(self.n_total_bins):
                if self.sub_ringmask_sums[i] > 0:
                    C_ell[:,i] = np.array([np.sum(self.sub_masked_weights[i]*fftsq[j][self.sub_ringmasks[i]])/self.sub_masked_weight_sums[i] for j in range(nsims)])

        else:
            C_ell = np.zeros((nsims, len(self.binl)-1))
            for i in range(len(self.binl)-1):
                if self.ringmask_sums[i] > 0:
                    C_ell[:,i] = np.array([np.sum(self.masked_weights[i]*fftsq[j][self.ringmasks[i]])/self.masked_weight_sums[i] for j in range(nsims)])

        return C_ell
    
    
    def compute_ringmasks(self, sub_bins=False, verbose=False, input_cl_func=None):    
        if verbose:    
            print('Minimum ell_map value is '+str(np.min(self.ell_map)))
        
        self.correspond_bins = []
        
        unshifted_ringmasks, ringmasks = [], []

        if input_cl_func is not None:

            cl2d = input_cl_func(self.ell_map)
        else:
            cl2d = None
                 
        for i in range(len(self.binl)-1):
            
            sub_bins_list = []
            
            if i < self.n_fine_bins and sub_bins:
                
                # here we'll make rows that are M_{(kx,ky), (kx',ky')}
                full_ringmask = (fftshift(self.ell_map) >= self.binl[i])*(fftshift(self.ell_map) <= self.binl[i+1])
                full_unshifted_ringmask = fftshift(np.array((self.ell_map >= self.binl[i])*(self.ell_map <= self.binl[i+1])).astype(float)/self.arcsec_pp_to_radian)
                
                nonzero_us_idxs = np.nonzero(full_unshifted_ringmask)
                pair_us_idxs = [[nonzero_us_idxs[0][i], nonzero_us_idxs[1][i]] for i in range(len(nonzero_us_idxs[0]))]
                grouped_us_idxs = group_idxs(pair_us_idxs, imdim=self.dimx)

                
                for j in range(len(grouped_us_idxs)):
                    
                    sub_bins_list.append(len(unshifted_ringmasks))
                    indiv_us_ptmask = np.zeros_like(self.ell_map)
                
                    for idx in grouped_us_idxs[j]:

                        indiv_us_ptmask[idx[0], idx[1]] = full_unshifted_ringmask[idx[0],idx[1]]

                    unshifted_ringmasks.append(indiv_us_ptmask)
                
        
                nonzero_idxs = np.nonzero(full_ringmask)
            
                pair_idxs = [[nonzero_idxs[0][i], nonzero_idxs[1][i]] for i in range(len(nonzero_idxs[0]))]

                grouped_idxs = group_idxs(pair_idxs, imdim=self.dimx)
                                
                for k in range(len(grouped_idxs)):
                    indiv_ptmask = np.zeros_like(self.ell_map)
                    for idx in grouped_idxs[k]:
                        indiv_ptmask[idx[0], idx[1]] = 1
                    ringmasks.append(indiv_ptmask.astype(np.bool))
        
            else:
                
                sub_bins_list.append(len(unshifted_ringmasks))

                ell_map_mask = np.array((self.ell_map >= self.binl[i])*(self.ell_map <= self.binl[i+1]))
                if cl2d is not None:
                    sum_ell_map_mask = np.sum(ell_map_mask)

                    ell_map_mask *= cl2d
                    ell_map_mask /= sum_ell_map_mask

                    plot_map(ell_map_mask, title='cl2d map with input spectrum')

                unshifted_ringmasks.append(fftshift(ell_map_mask.astype(float)/self.arcsec_pp_to_radian))

                # unshifted_ringmasks.append(fftshift(np.array((self.ell_map >= self.binl[i])*(self.ell_map <= self.binl[i+1])).astype(float)/self.arcsec_pp_to_radian))
                ringmasks.append((fftshift(self.ell_map) >= self.binl[i])*(fftshift(self.ell_map) <= self.binl[i+1]))
 
            self.correspond_bins.append(sub_bins_list)
        
        
        if sub_bins and self.n_fine_bins > 0:
            self.unshifted_sub_ringmasks = unshifted_ringmasks
            self.sub_ringmasks = ringmasks
            self.sub_ringmask_sums = np.array([np.sum(ringmask) for ringmask in self.sub_ringmasks])

        else:
            self.unshifted_ringmasks = unshifted_ringmasks
            self.ringmasks = ringmasks
            self.ringmask_sums = np.array([np.sum(ringmask) for ringmask in self.ringmasks])
        
        if verbose:
            print('self.correspond_bins is ', self.correspond_bins)


    def compute_midbin_delta_ells(self):
        self.midbin_ell = 0.5*(self.binl[1:]+self.binl[:-1])
        self.delta_ell = self.binl[1:]-self.binl[:-1]
        

    def compute_masked_weights(self, sub_bins=False):
        ''' This takes the Fourier weights and produces a  list of masked weights according to each multipole bandpass. The prefactor just converts the units
        appropriately. This is precomputed to make things faster during FFT time''' 
        fac = self.arcsec_pp_to_radian**2

        if sub_bins and self.n_fine_bins > 0:
            self.sub_masked_weights = [fac*self.weights[ringmask] for ringmask in self.sub_ringmasks]
        else:
            self.masked_weights = [fac*self.weights[ringmask] for ringmask in self.ringmasks]
    
    def compute_masked_weight_sums(self, sub_bins=False, verbose=False):
        if sub_bins and self.n_fine_bins > 0:
            self.sub_masked_weight_sums = np.array([np.sum(self.weights[ringmask]) for ringmask in self.sub_ringmasks])
        else:   
            self.masked_weight_sums = np.array([np.sum(self.weights[ringmask]) for ringmask in self.ringmasks])
            if verbose:
                print('self.masked weight sums:', self.masked_weight_sums)
            
    def load_fourier_weights(self, weights):
        ''' Loads the fourier weights!'''
        self.weights = weights
        
    def compute_multipole_bins(self):
        ''' This computes the multipole bins of the desired power spectrum, which depends on the pre-defined 
        number of bins, minimum and maximum multipole (which come from the image pixel size and FOV). Setting 
        self.logbin to True makes bins that are spaced equally in log space, rather than just linearly.'''

        if self.logbin:
            self.binl = 10**(np.linspace(np.log10(self.ell_min), np.log10(self.ell_max), self.nbins+1))
        else:
            self.binl = np.linspace(self.ell_min, self.ell_max, self.nbins+1)
                
    def get_ell_map(self, shift=True, pix_size=7., verbose=False):

        ''' this will take input map dimensions along with a pixel scale (in arcseconds) and compute the 
        corresponding multipole bins for the 2d map.'''
        
        self.ell_max = np.sqrt(2)*180./self.pixlength_in_deg # maximum multipole determined by pixel size

        freq_x = fftshift(np.fft.fftfreq(self.dimx, d=1.0))*2*(180*3600./pix_size)
        freq_y = fftshift(np.fft.fftfreq(self.dimy, d=1.0))*2*(180*3600./pix_size)
        
        ell_x,ell_y = np.meshgrid(freq_x,freq_y)
        ell_x = ifftshift(ell_x)
        ell_y = ifftshift(ell_y)

        self.ell_map = np.sqrt(ell_x**2 + ell_y**2)
        if verbose:
            print('minimum/maximum ell is ', np.min(self.ell_map[np.nonzero(self.ell_map)]), np.max(self.ell_map))

        if shift:
            self.ell_map = fftshift(self.ell_map)
            

    def precompute_mkk_quantities(self, precompute_all=False, ell_map=False, binl=False, weights=False, ringmasks=False,\
                                masked_weight_sums=False, masked_weights=False, midbin_delta_ells=False, shift=True, sub_bins=False, \
                                input_cl_func=None, verbose=False):
        
        
        if ell_map or precompute_all:
            if verbose:
                print('Generating 2D ell map..')
            self.get_ell_map(shift=shift, pix_size=self.pixsize)

                 
        if binl or precompute_all:
            self.compute_multipole_bins()
            if verbose:
                print('Generating multipole bins..')
                print('Multipole bin edges:', self.binl)
                
        if weights or precompute_all:
            # if no weights provided by user, weights are unity across image
            if self.weights is None:
                if verbose:
                    print('No Fourier weights provided, setting to unity..')
                self.weights = np.ones((self.dimx, self.dimy))   
            
        if ringmasks or precompute_all:
            if self.ringmasks is None or sub_bins:
                if verbose:
                    print('Computing Fourier ring masks..')
                self.compute_ringmasks(sub_bins=sub_bins, input_cl_func=input_cl_func)
              
        if masked_weights or precompute_all:
            self.compute_masked_weights(sub_bins=sub_bins)

        if masked_weight_sums or precompute_all:
            self.compute_masked_weight_sums(sub_bins=sub_bins)
            
        if midbin_delta_ells or precompute_all:
            self.compute_midbin_delta_ells()
        
        
    def compute_cl_indiv(self, map1, map2=None, weights=None, precompute=False, sub_bins=False, rebin=False):
        ''' this is for individual map realizations and should be more of a standalone, compared to Mkk.get_ensemble_angular_spec(),
        which is optimized for computing many realizations in parallel.'''
        
        if precompute:
            self.precompute_mkk_quantities(precompute_all=True, shift=True)

        fac = self.arcsec_pp_to_radian**2

        if map2 is None:
            map2 = map1
        
        fftsq = self.weights*((fft2(map1.real)*np.conj(fft2(map2.real))).real)
    
        C_ell = np.zeros(len(self.binl)-1)
        
        if sub_bins and self.n_fine_bins > 0:
            C_ell_star = np.zeros(self.n_total_bins)

            for i in range(self.n_total_bins):
                if self.sub_ringmask_sums[i] > 0:
                    # print(np.array(self.unshifted_sub_ringmasks[i]))
                    # print(fftsq[np.array(self.unshifted_sub_ringmasks[i])])
                    C_ell_star[i] = np.sum(self.sub_masked_weights[i]*fftsq[np.array(self.sub_ringmasks[i])])/self.sub_masked_weight_sums[i]

            if rebin:
                C_ell_star_rb = np.array([np.mean(C_ell_star[self.correspond_bins[k]]) for k in range(len(self.correspond_bins))])
                return C_ell_star_rb

            return C_ell_star
            
        else:
            C_ell = np.zeros(len(self.binl)-1)
            for i in range(len(self.binl)-1):
                if self.ringmask_sums[i] > 0:
                    C_ell[i] = np.sum(self.masked_weights[i]*fftsq[np.array(self.unshifted_ringmasks[i])])/self.masked_weight_sums[i]

            return C_ell

    def compute_error_vs_nsims(self, all_mkks, mask, mode='pure', n_realizations=200, tone_idxs=[0, 1, 2], nsims=[1,5,10,50,100, 200], \
                          analytic_estimate=False):

        inverse_mkks = []

        for nsim in nsims:
            avmkk = np.average(all_mkks[:nsim, :,:], axis=0)
            print('avmkk has shape', avmkk.shape)
            inverse_mkks.append(compute_inverse_mkk(avmkk))


        abs_errors = np.zeros((len(tone_idxs), n_realizations, len(nsims)))

        dcl_cl_list = []

        for j, idx in enumerate(tone_idxs):

            if analytic_estimate:
                ell = 0.5*(self.binl[idx]+self.binl[idx+1])
                d_ell = self.binl[idx+1]-self.binl[idx]

                dcl_cl = np.array([np.sqrt(2/(ell*d_ell*nsim)) for nsim in nsims])
                dcl_cl_list.append(dcl_cl)
                print('dcl/cl=', dcl_cl)

            print('tone index ', idx)
            abs_error = []

            for k in range(n_realizations):
                if mode=='pure':
                    testmap = ifft2(self.unshifted_ringmasks[idx]*np.array(np.random.normal(size=(self.dimx, self.dimy))+1j*np.random.normal(size=(self.dimx, self.dimy)))).real
                elif mode=='white':
                    testmap = ifft2(np.array(np.random.normal(size=(self.dimx, self.dimy))+1j*np.random.normal(size=(self.dimx, self.dimy)))).real

                c_ell, dc_ell = self.compute_cl_indiv(testmap)
                c_ell_masked, dc_ell_masked = self.compute_cl_indiv(testmap*mask)

                for n, nsim in enumerate(nsims):

                    c_ell_rectified = np.dot(np.array(inverse_mkks[n]).transpose(), np.array(c_ell_masked))

                    abs_errors[j,k,n] = np.abs((c_ell[idx] - c_ell_rectified[idx])/c_ell[idx])


        if analytic_estimate:
            return abs_errors, np.array(dcl_cl_list)
        return abs_errors
    
    def plot_tone_map_realization(self, tonemap, idx=None, return_fig=False):

        ''' produces an image of one tone map generated with map_from_phase_realization()'''

        f = plt.figure()
        if idx is not None:
            plt.title('Tone map ('+str(int(self.binl[idx]))+'<$\\ell$<'+str(int(self.binl[idx+1]))+')', fontsize=14)
        else:
            plt.title('Tone map', fontsize=14)
        plt.imshow(tonemap)
        plt.colorbar()
        plt.show()    
        
        if return_fig:
            return f

    def plot_twopanel_Mkk(self, signal_map, mask, inverse_mkk, xlims=None, ylims=None, colorbar=True, return_fig=False, logbin=False):

        c_ell, dc_ell = self.compute_cl_indiv(signal_map)
        c_ell_masked, dc_ell_masked = self.compute_cl_indiv(signal_map*mask)

        f = plt.figure(figsize=(10,5))
        plt.suptitle(str(np.round(self.binl[i],0))+'<$\\ell$<'+str(np.round(self.binl[i+1],0)), fontsize=20)
        plt.subplot(1,2,1)
        plt.imshow(signal_map*mask)

        if xlims is not None:
            plt.xlim(xlims[0], xlims[1])
        if ylims is not None:
            plt.ylim(ylims[0], ylims[1])
        if colorbar:
            plt.colorbar()

        plt.subplot(1,2,2)

        plt.errorbar(self.midbin_ell, c_ell, yerr=dc_ell, label='Unmasked', fmt='x', capsize=5, markersize=10)
        plt.scatter(self.midbin_ell, c_ell_masked, label='Masked, uncorrected')
        plt.errorbar(self.midbin_ell, np.dot(np.array(inverse_mkk), np.array(c_ell_masked)), yerr=dc_ell_masked, label='Masked, corrected',markersize=10, alpha=0.8, fmt='*', capsize=5)
        if logbin:
            plt.xscale('log')
        plt.legend(loc='best')
        plt.show()

        if return_fig:
            return f

    def show_mask(self, return_fig=False):

        f = plt.figure()
        plt.title('Input mask', fontsize=18)
        plt.imshow(self.mask, interpolation=None)
        plt.xlabel('$x$ [pixels]', fontsize=16)
        plt.ylabel('$y$ [pixels]', fontsize=16)
        plt.colorbar()
        plt.show()

        if return_fig:
            return f

                    
def wrapped_mkk(mask, nsims, nbins=20, show_tone_map=False, n_split=1):
    x = Mkk_bare(nbins=nbins)
    mkk, dc_ell = x.get_mkk_sim(mask, nsims, show_tone_map=show_tone_map, n_split=n_split)
    return mkk

def multithreaded_mkk(n_process=4, nsims=100):
    n_process = n_process
    p = Pool(processes=n_process)
    chunks = np.array_split(np.arange(nsims), n_process)
    print([len(chunk) for chunk in chunks])
    mkks = p.map(wrapped_mkk, [len(chunk) for chunk in chunks])
    print('Mkks is ', np.array(mkks).shape)
    mkk = np.average(mkks, axis=0, weights=[len(chunk) for chunk in chunks])
    return mkk


def plot_mkk_matrix(mkk, inverse=False, logscale=False, symlogscale=False, symlinthresh=1e-6, title=None, vmin=None, vmax=None, return_fig=False):
    if title is None:
        if inverse:
            title = '$M_{\\ell \\ell^\\prime}^{-1}$'
        else:
            title = '$M_{\\ell \\ell^\\prime}$'

    f = plt.figure(figsize=(6,6))
    plt.title(title, fontsize=16)
    if logscale:
        plt.imshow(mkk, norm=matplotlib.colors.LogNorm(), vmin=vmin, vmax=vmax, origin='upper')
    elif symlogscale:
        plt.imshow(mkk, norm=matplotlib.colors.SymLogNorm(symlinthresh), vmin=vmin, vmax=vmax, origin='upper')
    else:
        plt.imshow(mkk, origin='upper')
    plt.colorbar()
    plt.xticks(np.arange(0, mkk.shape[0], mkk.shape[0]//5))
    plt.yticks(np.arange(0, mkk.shape[1], mkk.shape[1]//5))
    plt.xlabel('Bin index', fontsize='large')
    plt.ylabel('Bin index', fontsize='large')
    plt.tight_layout()
    plt.show()
    
    if return_fig:
        return f


def save_mkks(filepath, all_Mkks=None, av_Mkk=None, inverse_Mkk=None, bins=None, midbin_ell=None, return_inv_Mkk=False, mask=None):
    
    if av_Mkk is not None and inverse_Mkk is None:
        inverse_Mkk = compute_inverse_mkk(av_Mkk)
        
    np.savez(filepath, all_Mkks=all_Mkks, av_Mkk=av_Mkk, inverse_Mkk=inverse_Mkk, bins=bins, midbin_ell=midbin_ell, mask=mask)
    print("saved mkk matrices to ", filepath)

    if return_inv_Mkk:
        return inverse_Mkk

def estimate_mkk_ffest_quadoff(cbps, nsims, masks, ifield_list=None, n_split=1, mean_normalizations=None, ff_weights=None, \
                      verbose=False, grad_sub=False, niter=1, quadoff_grad=False, poly_order=1, \
                              fc_sub=False, fc_sub_quad_offset=False, fc_sub_n_terms=2):
    
    if ifield_list is None:
        ifield_list = [4, 5, 6, 7, 8]
        
    masks = masks.astype(float)
        
    nfield = len(ifield_list)
    nsims_perf = nsims//n_split
    nstack = nsims_perf*nfield
    
    if mean_normalizations is None:
        mean_normalizations = np.zeros_like(ifield_list)
            
    maplist_split_shape = (nstack, cbps.dimx, cbps.dimy)
    maplist_split_shapenew = (nfield,nsims_perf, cbps.dimx, cbps.dimy)
    
    # empty aligned object will have nfield, nsims perfield shape. so fftsq iterates over fields and extends by nsims per field.
    if verbose:
        print('maplist split shape is ', maplist_split_shape)
    
    cbps.Mkk_obj.precompute_mkk_quantities(precompute_all=True)

    cbps.Mkk_obj.empty_aligned_objs, cbps.Mkk_obj.fft_objs = construct_pyfftw_objs_gen(4, maplist_split_shapenew, axes=(2,3), \
                                                                                  n_fft_obj=2, fft_idx0_list=[0, 2], fft_idx1_list=[1,3])

    n_total_bins = len(cbps.Mkk_obj.unshifted_ringmasks)
    Mkks_per_field = np.zeros((nfield, nsims, n_total_bins, n_total_bins))
    
    all_dot1, all_X, all_mask_rav = [], [], []
    if quadoff_grad:
        print('Precomputing quantities for quad offset + gradient..')
        for m in range(len(masks)):
            dot1, X, mask_rav = precomp_offset_gradient(cbps.dimx, cbps.dimy, mask=masks[m], order=poly_order)   
            all_dot1.append(dot1)
            all_X.append(X)
            all_mask_rav.append(mask_rav)
            
    elif fc_sub:
        print('Precomputing Fourier components')
        if fc_sub_quad_offset:
            print('.. with quadrant offsets')
        if grad_sub:
            print('.. with gradient removal as well')
        for m in range(len(masks)):
            dot1, X, mask_rav = precomp_fourier_templates(cbps.dimx, cbps.dimy, mask=masks[m], quad_offset=fc_sub_quad_offset, \
                                                         n_terms=fc_sub_n_terms, with_gradient=grad_sub)   
            all_dot1.append(dot1)
            all_X.append(X)
            all_mask_rav.append(mask_rav)
            
        
    
    # only need to compute ff estimates for one field at a time, so nstack instead of nstack*nfield
    perf_mapshape = (nsims_perf, cbps.dimx, cbps.dimy)
    
    ff_estimates = np.zeros(perf_mapshape)
    
    if ff_weights is None:
        ff_weights = np.ones((nfield))
        
    ff_weight_ims = np.array([ff_weights[m]*mask for m, mask in enumerate(masks)])

    for i in range(n_split):
        print('Split '+str(i+1)+' of '+str(n_split))
        
        cbps.Mkk_obj.noise = np.random.normal(size=maplist_split_shapenew)+ 1j*np.random.normal(size=maplist_split_shapenew)
        
        for j in range(n_total_bins):
            print('band ', j)
            cbps.Mkk_obj.map_from_phase_realization(j, nsims=nstack)
            
            # now we have 5 x nsims/nsplit maps. add mean normalization to each and apply mask.
            # I think I just need to get the relative mean normalizations correct.
            # does it matter what the relative fluctuation amplitude is? 
            
            for k in range(nfield):

                cbps.Mkk_obj.empty_aligned_objs[1][k].real += mean_normalizations[k]
                cbps.Mkk_obj.empty_aligned_objs[1][k].real *= masks[k]
                
            # for each source estimate the flat field from the other four fields. 
            for k in range(nfield):
                if verbose:
                    print('now computing for k = ', k)
                notkrange = [kp for kp in range(nfield) if kp != k]

                ff_weight_ims = np.array([ff_weights[kp]*masks[kp] for kp in notkrange])

                sum_ff_weight_ims = np.sum(ff_weight_ims, axis=0)

                for kp in notkrange:
                    cbps.Mkk_obj.empty_aligned_objs[1][kp] *= ff_weights[kp]/mean_normalizations[kp]

                if verbose:
                    print('shaaape', np.sum(cbps.Mkk_obj.empty_aligned_objs[1][notkrange,:,:,:].real, axis=0).shape)

                ff_estimates = np.sum(cbps.Mkk_obj.empty_aligned_objs[1][notkrange].real, axis=0)/sum_ff_weight_ims
                ff_estimates[np.isnan(ff_estimates)] = 1.   

                cbps.Mkk_obj.empty_aligned_objs[2][k].real = cbps.Mkk_obj.empty_aligned_objs[1][k].real/ff_estimates

                if quadoff_grad:
                    recov_offset_grads = np.array([offset_gradient_fit_precomp(cbps.Mkk_obj.empty_aligned_objs[2][k,s].real, all_dot1[k], all_X[k], all_mask_rav[k])[1] for s in range(nsims_perf)])
                    cbps.Mkk_obj.empty_aligned_objs[2][k].real -= np.array([masks[k]*recov_offset_grads[s] for s in range(nsims_perf)])
                elif fc_sub:
                    recov_offset_fcs = np.array([fc_fit_precomp(cbps.Mkk_obj.empty_aligned_objs[2][k,s].real, all_dot1[k], all_X[k], all_mask_rav[k])[1] for s in range(nsims_perf)])
                    cbps.Mkk_obj.empty_aligned_objs[2][k].real -= np.array([masks[k]*recov_offset_fcs[s] for s in range(nsims_perf)])

                elif grad_sub:
                    planes = np.array([fit_gradient_to_map(cbps.Mkk_obj.empty_aligned_objs[2][k,s].real, masks[k])[1] for s in range(nsims_perf)])
                    cbps.Mkk_obj.empty_aligned_objs[2][k].real -= np.array([masks[k]*planes[s] for s in range(nsims_perf)])

                unmasked_means = np.array([np.mean(cbps.Mkk_obj.empty_aligned_objs[2][k,s].real[masks[k]==1.]) for s in range(nsims_perf)])                
                    
                cbps.Mkk_obj.empty_aligned_objs[2][k] -= np.array([masks[k]*unmasked_means[s] for s in range(nsims_perf)])
                unmasked_means_sub = np.array([np.mean(cbps.Mkk_obj.empty_aligned_objs[2][k,s].real[masks[k]==1.]) for s in range(nsims_perf)])

                # undo scaling of off fields to ff_estimate
                for kp in notkrange:
                    cbps.Mkk_obj.empty_aligned_objs[1][kp] /= ff_weights[kp]/mean_normalizations[kp]

                if verbose:
                    print('unmasked means sub:', unmasked_means_sub)
                    print('ff scatter for field'+str(k)+':'+str(np.std([ff_estimate[masks[k]==1.] for ff_estimate in ff_estimates])))
    
            masked_Cl = cbps.Mkk_obj.get_ensemble_angular_autospec_ndim(nsims=nstack, apply_mask=False, obj_idx0=2, obj_idx1=3)
            for k in range(nfield):
                if verbose and i==0:
                    print('masked CL field '+str(k)+':')
                    print(np.mean(masked_Cl[k*nsims_perf:(k+1)*nsims_perf], axis=0))

                Mkks_per_field[k, i*nsims_perf:(i+1)*nsims_perf, j, :] = np.array(masked_Cl)[k*nsims_perf:(k+1)*nsims_perf]
            
        average_Mkks_perfield = np.mean(Mkks_per_field, axis=1)
        
        if i==n_split-1:
            for k in range(nfield):
                plot_mkk_matrix(average_Mkks_perfield[k])            


    return average_Mkks_perfield, Mkks_per_field



def estimate_mkk_ffest_cross(cbps, nsims, masks, n_split=1, grad_sub=False, niter=1, \
                       mean_normalizations=None, use_ff=True, ff_weights=None, mean_normalizations_cross=None, ff_weights_cross=None, \
                       ifield_list=[4, 5, 6, 7, 8], verbose=False, plot=False, threads=1, dtype='complex64', clock=False, \
                       quadoff_grad=False, order=1, fc_sub=False, fc_sub_quad_offset=False, fc_sub_n_terms=2):
    
    ''' 
    Compute mode coupling matrices for some number of fields for deconvolving observed power spectra.
    This function uses both the mask and the stacked FF estimator to estimate full mode coupling. In the unmasked case, 
    flat field errors can be 
    from the set of stacked fields couple with 
    
    
    '''
    
    t0 = time.time()
    nfield = len(ifield_list)

    cross_mkk = False
    if use_ff:
        ffnorm_facs = np.array(ff_weights)/np.array(mean_normalizations)
        print('ffnorm facs = ', ffnorm_facs)
    
    if mean_normalizations_cross is not None and ff_weights_cross is not None and use_ff:
        ffnorm_facs_cross = np.array(ff_weights_cross)/np.array(mean_normalizations_cross)
        cross_mkk = True
        
    masks = masks.astype(float)
    nsims_perf = nsims//n_split
    nstack = nsims_perf*nfield
    all_notkranges = [[kp for kp in range(nfield) if kp != k] for k in range(nfield)]

    maplist_split_shape = (nstack, cbps.dimx, cbps.dimy)
    maplist_split_shapenew = (nfield,nsims_perf, cbps.dimx, cbps.dimy)
    perf_mapshape = (nsims_perf, cbps.dimx, cbps.dimy)



    all_dot1, all_X, all_mask_rav = [], [], []
    if quadoff_grad or fc_sub:
        grad_sub=False
        for m in range(len(masks)):

            if quadoff_grad:
                if m==0:
                    print('Precomputing quantities for quad offset + gradient..')
                dot1, X, mask_rav = precomp_offset_gradient(cbps.dimx, cbps.dimy, mask=masks[m], order=order)   

            elif fc_sub:

                if m==0:
                    print('Precomputing FC + quad offsets')

                dot1, X, mask_rav = precomp_fourier_templates(cbps.dimx, cbps.dimy, mask=masks[m], quad_offset=fc_sub_quad_offset, \
                                            n_terms=fc_sub_n_terms, with_gradient=grad_sub)   

            all_dot1.append(dot1)
            all_X.append(X)
            all_mask_rav.append(mask_rav)

    # empty aligned object will have nfield, nsims perfield shape. so fftsq iterates over fields and extends by nsims per field.

    print('maplist split shape is ', maplist_split_shape)
    cbps.Mkk_obj.precompute_mkk_quantities(precompute_all=True)
    n_total_bins = len(cbps.Mkk_obj.unshifted_ringmasks)
    
    fft_idx0_list, fft_idx1_list,\
            directions, n_obj, n_fft_obj = init_fftobj_info(cross_mkk=cross_mkk)
        
    cbps.Mkk_obj.empty_aligned_objs, cbps.Mkk_obj.fft_objs = construct_pyfftw_objs_gen(n_obj, maplist_split_shapenew, axes=(2,3), \
                                                                                  n_fft_obj=n_fft_obj, fft_idx0_list=fft_idx0_list, fft_idx1_list=fft_idx1_list, \
                                                                                  threads=threads, dtype=dtype, directions=directions)

    # precompute vectors for gradient estimation

    # if grad_sub:
    #     dot1s, mask_ravs = precomp_gradsub(masks, cbps.dimx, cbps.dimy)
        
    Mkks_per_field = np.zeros((nfield, nsims, n_total_bins, n_total_bins))
    
    # only need to compute ff estimates for one field at a time, so nstack instead of nstack*nfield
    all_ff_estimates = np.zeros(maplist_split_shapenew)
    
    all_ff_estimates_cross = None
    if cross_mkk:
        all_ff_estimates_cross = np.zeros(maplist_split_shapenew)

    t1 = clock_log(message='time for initializing pyfftw objs and other data:', t0=t0, verbose=clock)
        
    ff_weight_ims = np.array([ff_weights[m]*mask for m, mask in enumerate(masks)])
    if cross_mkk:
        ff_weight_ims_cross = np.array([ff_weights_cross[m]*mask for m, mask in enumerate(masks)])

    for i in range(n_split):
        print('Split '+str(i+1)+' of '+str(n_split))
        
        t2 = time.time()
        # compute noise realizations beforehand so time isn't wasted generating a whole new set each time. We can do this because the ring masks
        # that we multiply by the noise realizations are spatially disjoint, i.e. the union of all of the Fourier masks is the null set. 
        cbps.Mkk_obj.noise = np.random.normal(size=maplist_split_shapenew)+ 1j*np.random.normal(size=maplist_split_shapenew)
        
        t3 = clock_log(message='time to make noise realizations:', t0=t2, verbose=clock)

        for j in range(n_total_bins):
            print('band ', j)
            cbps.Mkk_obj.map_from_phase_realization(j, nsims=nstack)
           
            t4 = time.time()
            
            if cross_mkk:
                if verbose:
                    print('mean normalizations cross:', mean_normalizations_cross)
                
                cbps.Mkk_obj.empty_aligned_objs[4] = cbps.Mkk_obj.empty_aligned_objs[1].copy()
                apply_norm_mask(cbps.Mkk_obj.empty_aligned_objs, 4, mean_normalizations_cross, masks, plot=False)

            
            apply_norm_mask(cbps.Mkk_obj.empty_aligned_objs, 1, mean_normalizations, masks, plot=False)

            t5 = clock_log(message='time to add mean normalizations and multiply by mask:', t0=t4, verbose=clock)

            for n in range(niter):
                
                tff0 = time.time()
                # for each source estimate the flat field from the other four fields. 
                for k in range(nfield):

                    tnf0 = time.time()
                    if verbose:
                        print('For k = '+str(k)+', notkrange = '+str(all_notkranges[k]))
                                        
                    sum_ff_weight_ims = np.sum([ff_weights[kp]*masks[kp] for kp in all_notkranges[k]], axis=0)
                    if cross_mkk:
                        sum_ff_weight_ims_cross = np.sum([ff_weights_cross[kp]*masks[kp] for kp in all_notkranges[k]], axis=0)
                    
                    tnf1 = clock_log(message='tnf1 - tnf0 =', t0=tnf0, verbose=clock)
                            
                    all_ff_estimates[k] = apply_weighted_ffnorm(k, all_notkranges, cbps.Mkk_obj.empty_aligned_objs,\
                                                                sum_ff_weight_ims, 1, ffnorm_facs)
                    
                    cbps.Mkk_obj.empty_aligned_objs[2][k].real = cbps.Mkk_obj.empty_aligned_objs[1][k].real/all_ff_estimates[k]

                    if cross_mkk:
                        all_ff_estimates_cross[k] = apply_weighted_ffnorm(k, all_notkranges, cbps.Mkk_obj.empty_aligned_objs,\
                                                                          sum_ff_weight_ims_cross, 4, ffnorm_facs_cross)
                        
                        cbps.Mkk_obj.empty_aligned_objs[5][k].real = cbps.Mkk_obj.empty_aligned_objs[4][k].real/all_ff_estimates_cross[k]
        
                    tnf3 = clock_log(message='tnf3 - tnf2:', verbose=clock)

                    # undo scaling of off fields to ff_estimate
                    if k < nfield-1:
                        cbps.Mkk_obj.empty_aligned_objs[1][k+1] /= ffnorm_facs[k+1]
                        if cross_mkk:
                            cbps.Mkk_obj.empty_aligned_objs[4][k+1] /= ffnorm_facs_cross[k+1]

                    tnf4 = clock_log(message='tnf4 - tnf3', t0=tnf3, verbose=clock)


                    if quadoff_grad:
                        print('quad off grad beginning')
                        recov_offset_grads = np.array([offset_gradient_fit_precomp(cbps.Mkk_obj.empty_aligned_objs[2][k,s].real, all_dot1[k], all_X[k], all_mask_rav[k])[1] for s in range(nsims_perf)])
                        cbps.Mkk_obj.empty_aligned_objs[2][k].real -= np.array([masks[k]*recov_offset_grads[s] for s in range(nsims_perf)])
                        if cross_mkk:
                            print('doing quad off grad on cross')
                            recov_offset_grads_cross = np.array([offset_gradient_fit_precomp(cbps.Mkk_obj.empty_aligned_objs[5][k,s].real, all_dot1[k], all_X[k], all_mask_rav[k])[1] for s in range(nsims_perf)])
                            cbps.Mkk_obj.empty_aligned_objs[5][k].real -= np.array([masks[k]*recov_offset_grads_cross[s] for s in range(nsims_perf)])

                    elif fc_sub:

                        recov_offset_fcs = np.array([fc_fit_precomp(cbps.Mkk_obj.empty_aligned_objs[2][k,s].real, all_dot1[k], all_X[k], all_mask_rav[k])[1] for s in range(nsims_perf)])
                        cbps.Mkk_obj.empty_aligned_objs[2][k].real -= np.array([masks[k]*recov_offset_fcs[s] for s in range(nsims_perf)])

                        if cross_mkk:
                            recov_offset_fcs = np.array([fc_fit_precomp(cbps.Mkk_obj.empty_aligned_objs[5][k,s].real, all_dot1[k], all_X[k], all_mask_rav[k])[1] for s in range(nsims_perf)])
                            cbps.Mkk_obj.empty_aligned_objs[5][k].real -= np.array([masks[k]*recov_offset_fcs[s] for s in range(nsims_perf)])


                    # if grad_sub: # probably skip this for mocks
                    #     planes = perform_grad_sub(k, cbps.Mkk_obj.empty_aligned_objs, masks[k], 2, dot1s, Xgrad, mask_ravs[k], nsims_perf)
                    #     if cross_mkk:
                    #         planes_cross = perform_grad_sub(k, cbps.Mkk_obj.empty_aligned_objs, masks[k], 5, dot1s, Xgrad, mask_ravs[k], nsims_perf)

                    tnf5 = clock_log(message='grad sub:', t0=tnf4, verbose=clock)

                    if n==niter-1:
                        unmasked_means = np.array([np.mean(cbps.Mkk_obj.empty_aligned_objs[2][k,s].real[masks[k]==1.]) for s in range(nsims_perf)])                
                        cbps.Mkk_obj.empty_aligned_objs[2][k] -= np.array([masks[k]*unmasked_means[s] for s in range(nsims_perf)])
                        
                        if cross_mkk:
                            unmasked_means_cross = np.array([np.mean(cbps.Mkk_obj.empty_aligned_objs[5][k,s].real[masks[k]==1.]) for s in range(nsims_perf)])                
                            cbps.Mkk_obj.empty_aligned_objs[5][k] -= np.array([masks[k]*unmasked_means_cross[s] for s in range(nsims_perf)])
                        
                        if k==0 and i==0 and verbose:
                            print('unmasked mean pre sub:', unmasked_means)
                            if cross_mkk:
                                print('unmasked mean pre sub:', unmasked_means_cross)

                            
                    tnf6 = clock_log(message='tnf6-tnf5 =', t0=tnf5, verbose=clock)

                tff1 = clock_log(message='time for full iteration:', t0=tff0, verbose=clock)
                    
            if j<2 and plot:
                for kc in range(nfield):
                    plot_map(all_ff_estimates[kc,0], title='final ff estimate')
                    plot_map(cbps.Mkk_obj.empty_aligned_objs[1][kc,0].real, title='cbps.Mkk_obj.empty_aligned_objs[1]['+str(kc)+',0]')
                    print(np.mean(cbps.Mkk_obj.empty_aligned_objs[1][kc,0].real[masks[kc]==1]), mean_normalizations[kc])

                    
            obj_idx0_cross, obj_idx1_cross, fft_obj_idx_cross = [None for x in range(3)]
            if cross_mkk:
                obj_idx0_cross, obj_idx1_cross, fft_obj_idx_cross = 5, 6, 2
            
            masked_Cl = cbps.Mkk_obj.get_ensemble_angular_autospec_ndim(nsims=nstack, apply_mask=False, obj_idx0=2, obj_idx1=3, \
                                                                       obj_idx0_cross=obj_idx0_cross, obj_idx1_cross=obj_idx1_cross, fft_obj_idx_cross=fft_obj_idx_cross)
                                
            for k in range(nfield):
                if verbose and i==0:
                    print('masked CL field '+str(k)+':')
                    print(np.mean(masked_Cl[k*nsims_perf:(k+1)*nsims_perf], axis=0))

                Mkks_per_field[k, i*nsims_perf:(i+1)*nsims_perf, j, :] = np.array(masked_Cl)[k*nsims_perf:(k+1)*nsims_perf]
            
        average_Mkks_perfield = np.mean(Mkks_per_field[:,:(i+1)*nsims_perf], axis=1)
        
        if plot:
            for k in range(nfield):
                plot_mkk_matrix(average_Mkks_perfield[k])            


    return average_Mkks_perfield, Mkks_per_field


def estimate_mkk_ffest(cbps, nsims, masks, ifield_list=None, n_split=1, mean_normalizations=None, ff_weights=None, \
                      verbose=False, grad_sub=False, niter=1):
    
    if ifield_list is None:
        ifield_list = [4, 5, 6, 7, 8]
        
    masks = masks.astype(float)
        
    nfield = len(ifield_list)
    nsims_perf = nsims//n_split
    nstack = nsims_perf*nfield
    
    if mean_normalizations is None:
        mean_normalizations = np.zeros_like(ifield_list)
            
    maplist_split_shape = (nstack, cbps.dimx, cbps.dimy)
    maplist_split_shapenew = (nfield,nsims_perf, cbps.dimx, cbps.dimy)
    
    # empty aligned object will have nfield, nsims perfield shape. so fftsq iterates over fields and extends by nsims per field.

    print('maplist split shape is ', maplist_split_shape)
    
    cbps.Mkk_obj.precompute_mkk_quantities(precompute_all=True)

    cbps.Mkk_obj.empty_aligned_objs, cbps.Mkk_obj.fft_objs = construct_pyfftw_objs_gen(4, maplist_split_shapenew, axes=(2,3), \
                                                                                  n_fft_obj=2, fft_idx0_list=[0, 2], fft_idx1_list=[1,3])

    n_total_bins = len(cbps.Mkk_obj.unshifted_ringmasks)
    Mkks_per_field = np.zeros((nfield, nsims, n_total_bins, n_total_bins))
    
    # only need to compute ff estimates for one field at a time, so nstack instead of nstack*nfield
    perf_mapshape = (nsims_perf, cbps.dimx, cbps.dimy)
    
    ff_estimates = np.zeros(perf_mapshape)
    
    if ff_weights is None:
        ff_weights = np.ones((nfield))
        
    ff_weight_ims = np.array([ff_weights[m]*mask for m, mask in enumerate(masks)])

    for i in range(n_split):
        print('Split '+str(i+1)+' of '+str(n_split))
        
        # compute noise realizations beforehand so time isn't wasted generating a whole new set each time. We can do this because the ring masks
        # that we multiply by the noise realizations are spatially disjoint, i.e. the union of all of the Fourier masks is the null set. 
        cbps.Mkk_obj.noise = np.random.normal(size=maplist_split_shapenew)+ 1j*np.random.normal(size=maplist_split_shapenew)
        
        for j in range(n_total_bins):
            print('band ', j)
            cbps.Mkk_obj.map_from_phase_realization(j, nsims=nstack)
            
            # now we have 5 x nsims/nsplit maps. add mean normalization to each and apply mask.
            # I think I just need to get the relative mean normalizations correct.
            # does it matter what the relative fluctuation amplitude is? 
            
            for k in range(nfield):

                cbps.Mkk_obj.empty_aligned_objs[1][k].real += mean_normalizations[k]
                cbps.Mkk_obj.empty_aligned_objs[1][k].real *= masks[k]
                
            # for each source estimate the flat field from the other four fields. 
            for k in range(nfield):
                if verbose:
                    print('now computing for k = ', k)
                notkrange = [kp for kp in range(nfield) if kp != k]

                ff_weight_ims = np.array([ff_weights[kp]*masks[kp] for kp in notkrange])

                sum_ff_weight_ims = np.sum(ff_weight_ims, axis=0)

                for kp in notkrange:
                    cbps.Mkk_obj.empty_aligned_objs[1][kp] *= ff_weights[kp]/mean_normalizations[kp]

                if verbose:
                    print('shaaape', np.sum(cbps.Mkk_obj.empty_aligned_objs[1][notkrange,:,:,:].real, axis=0).shape)

                ff_estimates = np.sum(cbps.Mkk_obj.empty_aligned_objs[1][notkrange].real, axis=0)/sum_ff_weight_ims
                ff_estimates[np.isnan(ff_estimates)] = 1.   

                cbps.Mkk_obj.empty_aligned_objs[2][k].real = cbps.Mkk_obj.empty_aligned_objs[1][k].real/ff_estimates

                if grad_sub:

                    planes = np.array([fit_gradient_to_map(cbps.Mkk_obj.empty_aligned_objs[2][k,s].real, masks[k])[1] for s in range(nsims_perf)])
#                     plot_map(planes[0], title='plane kp = '+str(kp))
                    cbps.Mkk_obj.empty_aligned_objs[2][k].real -= np.array([masks[k]*planes[s] for s in range(nsims_perf)])
#                     plot_map(cbps.Mkk_obj.empty_aligned_objs[2][k,0].real, title='post plane sub kp = '+str(kp))

                unmasked_means = np.array([np.mean(cbps.Mkk_obj.empty_aligned_objs[2][k,s].real[masks[k]==1.]) for s in range(nsims_perf)])                
                cbps.Mkk_obj.empty_aligned_objs[2][k] -= np.array([masks[k]*unmasked_means[s] for s in range(nsims_perf)])
                unmasked_means_sub = np.array([np.mean(cbps.Mkk_obj.empty_aligned_objs[2][k,s].real[masks[k]==1.]) for s in range(nsims_perf)])

                # undo scaling of off fields to ff_estimate
                for kp in notkrange:
                    cbps.Mkk_obj.empty_aligned_objs[1][kp] /= ff_weights[kp]/mean_normalizations[kp]

                if verbose:
                    print('unmasked means sub:', unmasked_means_sub)
                    print('ff scatter for field'+str(k)+':'+str(np.std([ff_estimate[masks[k]==1.] for ff_estimate in ff_estimates])))
    
            masked_Cl = cbps.Mkk_obj.get_ensemble_angular_autospec_ndim(nsims=nstack, apply_mask=False, obj_idx0=2, obj_idx1=3)
            for k in range(nfield):
                if verbose and i==0:
                    print('masked CL field '+str(k)+':')
                    print(np.mean(masked_Cl[k*nsims_perf:(k+1)*nsims_perf], axis=0))

                Mkks_per_field[k, i*nsims_perf:(i+1)*nsims_perf, j, :] = np.array(masked_Cl)[k*nsims_perf:(k+1)*nsims_perf]
            
        average_Mkks_perfield = np.mean(Mkks_per_field, axis=1)
        
        for k in range(nfield):
            plot_mkk_matrix(average_Mkks_perfield[k])            


    return average_Mkks_perfield, Mkks_per_field


def truncate_invert_mkkmat(mkk_matrix, skipidx=1, vmin=1e-5, vmax=3, symlinthresh=1e1, plot=False):
    
    ''' Note this is specifically for 2nd order Fourier component model, could generalize '''
    
    truncated_mkk_matrix = np.zeros((mkk_matrix.shape[0]-1, mkk_matrix.shape[1]-1))
    
    truncated_mkk_matrix[1:,1:] = mkk_matrix[2:,2:]
    truncated_mkk_matrix[0,0] = mkk_matrix[0,0]
    truncated_mkk_matrix[0, 1:] = mkk_matrix[0, 2:]
    truncated_mkk_matrix[1:, 0] = mkk_matrix[2:, 0]
    
    cond = np.linalg.cond(truncated_mkk_matrix)
    
    inv_mkk_truncated = np.linalg.inv(truncated_mkk_matrix)
    inv_mkk_truncated_23 = np.linalg.inv(mkk_matrix[2:,2:])
    inv_mkk_truncated_match_23 = inv_mkk_truncated[1:,1:]

    print('condition number is ', cond)
    
    if plot:
        plot_mkk_matrix(inv_mkk_truncated, inverse=True, symlogscale=True, symlinthresh=symlinthresh)
        plot_mkk_matrix(inv_mkk_truncated_23, inverse=True, symlogscale=True, symlinthresh=symlinthresh)
        plot_map(inv_mkk_truncated_match_23/inv_mkk_truncated_23, title='ratio of matching matrix elements', cmap='bwr', vmin=0, vmax=2, figsize=(5,5), origin='lower')

        plt.figure(figsize=(5, 5))
        plt.imshow((inv_mkk_truncated_match_23/inv_mkk_truncated_23).transpose(), cmap='bwr', vmin=-1, vmax=3)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.ylabel('Bandpower $b^{\\prime}$', fontsize=12)
        plt.xlabel('Bandpower $b$', fontsize=12)
        # plt.text(1, 21, 'A', fontsize=16, bbox=dict({'facecolor':'white', 'edgecolor':'None', 'alpha':alpha_text}))
        plt.yticks([], [])
        plt.xticks([], [])
        plt.show()
    
        plt.figure(figsize=(5, 5))
        plt.imshow(truncated_mkk_matrix, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax), cmap='jet', origin='lower')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.ylabel('Bandpower $b^{\\prime}$', fontsize=12)
        plt.xlabel('Bandpower $b$', fontsize=12)
        # plt.text(1, 21, 'A', fontsize=16, bbox=dict({'facecolor':'white', 'edgecolor':'None', 'alpha':alpha_text}))
        plt.yticks([], [])
        plt.xticks([], [])
        plt.show()
    
    return truncated_mkk_matrix, inv_mkk_truncated_match_23

