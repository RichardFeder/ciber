import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fftshift as fftshift
from numpy.fft import ifftshift as ifftshift
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
import multiprocessing
from multiprocessing import Pool
import pyfftw

def allocate_empty_aligned_pyfftw(dat_shape, dtype='complex64'):
    return pyfftw.empty_aligned(dat_shape, dtype=dtype)
    
    
def construct_pyfftw_objs(n_obj, dat_shape, dtype='complex64', axes=(1,2), n_threads=1, directions=['FFTW_BACKWARD', 'FFTW_FORWARD']):
    
    fft_objs = []
    empty_aligned_objs = []
    
    for i in range(n_obj):
        empty_aligned_objs.append(allocate_empty_aligned_pyfftw(dat_shape, dtype=dtype))
    
    for j in range(n_obj-1):
        
        fft_obj = pyfftw.FFTW(empty_aligned_objs[j], empty_aligned_objs[j+1], axes=axes, threads=n_threads, direction=directions[j], flags=('FFTW_MEASURE', ))
    
        fft_objs.append(fft_obj)
         
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


class Mkk_bare():
    
    def __init__(self, pixsize=7., dimx=1024, dimy=1024, ell_min=180., logbin=True, nbins=20, n_fine_bins=0):
        
        for attr, valu in locals().items():
            setattr(self, attr, valu)
            
        self.pixlength_in_deg = self.pixsize/3600. # pixel angle measured in degrees
        self.ell_max = 2*180./self.pixlength_in_deg # maximum multipole determined by pixel size
        print('ell max is ', self.ell_max)
        
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


    def set_bin_edges(self, binl):
        self.binl = binl

    def get_mkk_sim(self, mask, nsims, show_tone_map=False, mode='auto', n_split=1, \
                    print_timestats=False, return_all_Mkks=False, n_fine_bins=None, sub_bins=False, store_Mkks=True, precompute_all=False):
        
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
        
        self.empty_aligned_objs, self.fft_objs = construct_pyfftw_objs(3, maplist_split_shape)
       
        Mkks, dC_ell_list = [], []
        
        # lets precompute a few things that are used later..
        if precompute_all:
            self.precompute_mkk_quantities(precompute_all=True, shift=True, sub_bins=sub_bins)
        
        # this is for measuring individual 2D fourier modes for some number of bins 
        if sub_bins and self.n_fine_bins > 0:
            n_total_bins = len(self.unshifted_sub_ringmasks)
        else:
            n_total_bins = len(self.unshifted_ringmasks)
        
        self.n_total_bins = n_total_bins

        if return_all_Mkks or store_Mkks:
            all_Mkks = np.zeros((nsims, n_total_bins, n_total_bins))
        
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
    
    
    def map_from_phase_realization(self, idx=0, shift=True,nsims=1, mode=None, sub_bins=False, pix_size=7.):

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


    
    def get_ensemble_angular_autospec(self, nsims=2, sub_bins=False, apply_mask=False):
                 
        if apply_mask:
            self.empty_aligned_objs[1] *= self.mask

        self.fft_objs[1](self.empty_aligned_objs[1].real)
        
        fftsq = [(dentry*np.conj(dentry)).real for dentry in self.empty_aligned_objs[2]]
            
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
    
    
    def compute_ringmasks(self, sub_bins=False):        
        print('Minimum ell_map value is '+str(np.min(self.ell_map)))
        
        self.correspond_bins = []
        
        unshifted_ringmasks, ringmasks = [], []
                 
        for i in range(len(self.binl)-1):
            
            sub_bins_list = []
            
            if i < self.n_fine_bins and sub_bins:
                
                # here we'll make rows that are M_{(kx,ky), (kx',ky')}
                full_ringmask = (fftshift(self.ell_map) >= self.binl[i])*(fftshift(self.ell_map) <= self.binl[i+1])
                full_unshifted_ringmask = fftshift(np.array((self.ell_map >= self.binl[i])*(self.ell_map <= self.binl[i+1])).astype(np.float)/self.arcsec_pp_to_radian)
                
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
                unshifted_ringmasks.append(fftshift(np.array((self.ell_map >= self.binl[i])*(self.ell_map <= self.binl[i+1])).astype(np.float)/self.arcsec_pp_to_radian))
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
    
    def compute_masked_weight_sums(self, sub_bins=False):
        if sub_bins and self.n_fine_bins > 0:
            self.sub_masked_weight_sums = np.array([np.sum(self.weights[ringmask]) for ringmask in self.sub_ringmasks])
        else:   
            self.masked_weight_sums = np.array([np.sum(self.weights[ringmask]) for ringmask in self.ringmasks])
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
                
    def get_ell_map(self, shift=True, pix_size=7.):

        ''' this will take input map dimensions along with a pixel scale (in arcseconds) and compute the 
        corresponding multipole bins for the 2d map.'''
        
        self.ell_max = np.sqrt(2)*180./self.pixlength_in_deg # maximum multipole determined by pixel size

        freq_x = fftshift(np.fft.fftfreq(self.dimx, d=1.0))*2*(180*3600./pix_size)
        freq_y = fftshift(np.fft.fftfreq(self.dimy, d=1.0))*2*(180*3600./pix_size)
        
        ell_x,ell_y = np.meshgrid(freq_x,freq_y)
        ell_x = ifftshift(ell_x)
        ell_y = ifftshift(ell_y)

        self.ell_map = np.sqrt(ell_x**2 + ell_y**2)
        
        print('minimum/maximum ell is ', np.min(self.ell_map[np.nonzero(self.ell_map)]), np.max(self.ell_map))

        if shift:
            self.ell_map = fftshift(self.ell_map)
            

    def precompute_mkk_quantities(self, precompute_all=False, ell_map=False, binl=False, weights=False, ringmasks=False,\
                                masked_weight_sums=False, masked_weights=False, midbin_delta_ells=False, shift=True, sub_bins=False):
        
        
        if ell_map or precompute_all:
            print('Generating 2D ell map..')
            self.get_ell_map(shift=shift, pix_size=self.pixsize)

                 
        if binl or precompute_all:
            print('Generating multipole bins..')
            self.compute_multipole_bins()
            print('Multipole bin edges:', self.binl)
                
        if weights or precompute_all:
            # if no weights provided by user, weights are unity across image
            if self.weights is None:
                print('No Fourier weights provided, setting to unity..')
                self.weights = np.ones((self.dimx, self.dimy))   
            
        if ringmasks or precompute_all:
            if self.ringmasks is None or sub_bins:
                print('Computing Fourier ring masks..')
                self.compute_ringmasks(sub_bins=sub_bins)
              
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


def save_mkks(filepath, all_Mkks=None, av_Mkk=None, inverse_Mkk=None, bins=None, midbin_ell=None):
    
    if av_Mkk is not None and inverse_Mkk is None:
        inverse_Mkk = compute_inverse_mkk(av_Mkk)
        
    np.savez(filepath, all_Mkks=all_Mkks, av_Mkk=av_Mkk, inverse_Mkk=inverse_Mkk, bins=bins, midbin_ell=midbin_ell)
    print("saved mkk matrices to ", filepath)



 
# mkk = multithreaded_mkk(nsims=20, n_process=2)

# this code below is just to do a test run, should ultimately comment out and call routines from another script

# x = Mkk(nbins=20)

# mask = np.ones((x.dimx, x.dimy))

# # # mask[100:700,200:500] = 0.
# # # mask[800:810,600:620] = 0.
# # # mask[800:810,300:320] = 0.
# # # mask[600:660,910:920] = 0.


# # tzero = time.clock()
# mkk, dc_ell = x.get_mkk_sim(mask, 100, show_tone_map=False, n_split=2)


# print('elapsed time is ', time.clock()-tzero)
# plot_mkk_matrix(mkk)

# print([mkk[i,i] for i in range(mkk.shape[0])])

# mkk_inv = compute_inverse_mkk(mkk)
# plot_mkk_matrix(mkk_inv, inverse=True)

# print(mkk)
