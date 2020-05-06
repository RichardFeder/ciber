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
import time


def compute_inverse_mkk(mkk):
    
    inverse_mkk = np.linalg.inv(mkk)
    
    return inverse_mkk


def plot_abs_errors(abs_errors, tone_idxs, nsims, analytic_dcl_cl=None, ell_bins=None, ell_strings=None, return_fig=False, ylims=[5e-4, 1e-1]):
    
    
    if ell_strings is None:
        ell_strings = [str(int(ell_bins[idx]))+'$<\\ell<$'+str(int(ell_bins[idx+1])) for idx in tone_idxs]
    
    f = plt.figure(figsize=(8,6))
    for i, idx in enumerate(tone_idxs):

        plt.errorbar(nsims, np.mean(abs_errors[i], axis=0), np.std(abs_errors[i], axis=0),\
                 label=ell_strings[i], marker='.', capsize=5, capthick=2, linewidth=2, markersize=10)
        
        if analytic_dcl_cl is not None:
            if i==0:
                label='Analytic estimates'
            else:
                label=None
            plt.plot(nsims, analytic_dcl_cl[i],label=label, color='k')
        
    plt.legend()
    plt.xlabel('$N_{sims}$', fontsize=24)
    plt.ylabel('$\\langle \\left| \\frac{C_{\ell} - C_{\ell}^{recon}}{C_{\ell}} \\right| \\rangle$', fontsize=20)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(ylims[0], ylims[1])
    plt.show()
    if return_fig:
        return f
    

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


class Mkk2():
    
    def __init__(self, pixsize=7., dimx=1024, dimy=1024, ell_min=150., logbin=True, nbins=30, n_fine_bins=0):
        
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
        self.ringmasks = None


            
    def map_from_powerspec(self, idx=0, shift=True,nsims=1, mode=None, sub_bins=False):

        ''' This makes maps very quickly given some precomputed quantities. The unshifted ringmasks are used to generate
        the pure tones within each band, so then this gets multiplied by the precalculated noise and fourier transformed to real space. These
        maps are then stored in the variable self.c (self.fft_object_b goes from self.b --> self.c)'''
    
        if self.ell_map is None:
            self.get_ell_bins(shift=shift)
            
        if self.binl is None:
            self.compute_multipole_bins()
         
        t0 = time.clock()
        
        if mode == 'white':
            self.fft_object_b(self.noise)
        else:
            if sub_bins:
                self.fft_object_b(self.unshifted_sub_ringmasks[idx]*self.noise)
            else:
                self.fft_object_b(self.unshifted_ringmasks[idx]*self.noise)

        if self.print_timestats:
            print('fft object b takes ', time.clock()-t0)

    def get_mkk_sim(self, mask, nsims, show_tone_map=False, mode='auto', n_split=1, \
                    print_timestats=False, return_all_Mkks=False, n_fine_bins=None, sub_bins=False):
        
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
        print('number of CPU threads available:', multiprocessing.cpu_count())


        self.print_timestats = print_timestats
        
        if n_fine_bins is not None:
            self.n_fine_bins = n_fine_bins
            
        if n_split > 1:
            print 'Splitting up computation of', nsims, 'simulations into', n_split, 'chunks..'
            assert (nsims % n_split) == 0

        self.mask = mask

        self.b = pyfftw.empty_aligned((nsims//n_split, self.dimx, self.dimy), dtype='complex64')        
        self.c = pyfftw.empty_aligned((nsims//n_split, self.dimx, self.dimy), dtype='complex64')
        self.d = pyfftw.empty_aligned((nsims//n_split, self.dimx, self.dimy), dtype='complex64')

        self.fft_object_b = pyfftw.FFTW(self.b, self.c, axes=(1,2),threads=1, direction='FFTW_BACKWARD', flags=('FFTW_MEASURE',))
        self.fft_object_c = pyfftw.FFTW(self.c, self.d, axes=(1,2),threads=1, direction='FFTW_FORWARD', flags=('FFTW_MEASURE',))
       
        Mkks = []
        dC_ell_list = []
        
        # lets precompute a few things that are used later..
        self.check_precomputed_values(precompute_all=True, shift=True, sub_bins=sub_bins)
        
        if sub_bins:
            n_total_bins = len(self.unshifted_sub_ringmasks)
        else:
            n_total_bins = len(self.unshifted_ringmasks)
        
        self.n_total_bins = n_total_bins

        all_Mkks = np.zeros((nsims, n_total_bins, n_total_bins))
        
        for i in range(n_split):
            print('split ', i+1, 'of', n_split)

            # compute noise realizations beforehand so time isn't wasted generating a whole new set each time. We can do this because the ring masks
            # that we multiply by the noise realizations are spatially disjoint, i.e. the union of all of the masks is the null set. 
            self.noise = np.random.normal(size=(nsims//n_split, self.dimx, self.dimy))+ 1j*np.random.normal(size=(nsims//n_split, self.dimx, self.dimy))
            
            Mkk = np.zeros((n_total_bins, n_total_bins))
            
            for j in range(n_total_bins):
                print('band ', j)
                
                self.map_from_powerspec(j, nsims=nsims//n_split, sub_bins=sub_bins)
                
                if mode=='auto':
                    masked_Cl = self.get_angular_spec(nsims=nsims//n_split, sub_bins=sub_bins, apply_mask=True)

                    if return_all_Mkks:
                        all_Mkks[i*nsims//n_split :(i+1)*nsims//n_split ,j,:] = np.array(masked_Cl)

                    # row entry for Mkk is average over realizations
                    Mkk[j,:] = np.mean(np.array(masked_Cl), axis=0)

            Mkks.append(Mkk)
    
        if n_split == 1:
            if return_all_Mkks:
                return all_Mkks
            return Mkks[0]

        else:
            if return_all_Mkks:
                return all_Mkks
            av_Mkk = np.mean(np.array(Mkks), axis=0)

            return av_Mkk
    
    
    def get_angular_spec(self, map1=None, map2=None, nsims=2, sub_bins=False, apply_mask=False):
        
        fac = self.arcsec_pp_to_radian**2
        
        if map2 is None:
            t0 = time.clock()
            
            if apply_mask:
                self.c *= self.mask

            self.fft_object_c(self.c.real)
            
            if self.print_timestats:
                print('fft_object_c took ', time.clock()-t0)

            t0 = time.clock()
            fftsq = [(dentry*np.conj(dentry)).real for dentry in self.d]
            
            if self.print_timestats:
                print('fftsq took ', time.clock()-t0)
        else:
            fftsq = self.weights*((ifft2(map1)*np.conj(ifft2(map2))).real)*fac

        t0 = time.clock()

        if sub_bins:
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
        print('minimum ell_map value is ', np.min(self.ell_map))
        
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
 

            if sub_bins:
                self.correspond_bins.append(sub_bins_list)
        
        
        if sub_bins:
            self.unshifted_sub_ringmasks = unshifted_ringmasks
            self.sub_ringmasks = ringmasks
            self.sub_ringmask_sums = np.array([np.sum(ringmask) for ringmask in self.sub_ringmasks])
            print('self.correspond_bins is ', self.correspond_bins)

        else:
            self.unshifted_ringmasks = unshifted_ringmasks
            self.ringmasks = ringmasks
            self.ringmask_sums = np.array([np.sum(ringmask) for ringmask in self.ringmasks])

        
    def compute_masked_weight_sums(self, sub_bins=False):
        if sub_bins:
            self.sub_masked_weight_sums = np.array([np.sum(self.weights[ringmask]) for ringmask in self.sub_ringmasks])
        else:   
            self.masked_weight_sums = np.array([np.sum(self.weights[ringmask]) for ringmask in self.ringmasks])
            print('self.masked weight sums:', self.masked_weight_sums)

    def compute_midbin_delta_ells(self):
        self.midbin_ell = 0.5*(self.binl[1:]+self.binl[:-1])
        self.delta_ell = self.binl[1:]-self.binl[:-1]
        

    def compute_masked_weights(self, sub_bins=False):
        ''' This takes the Fourier weights and produces a  list of masked weights according to each multipole bandpass. The prefactor just converts the units
        appropriately. This is precomputed to make things faster during FFT time''' 
        fac = self.arcsec_pp_to_radian**2

        if sub_bins:
            self.sub_masked_weights = [fac*self.weights[ringmask] for ringmask in self.sub_ringmasks]
        else:
            self.masked_weights = [fac*self.weights[ringmask] for ringmask in self.ringmasks]
    
   
    def load_fourier_weights(self, weights):
        ''' Loads the fourier weights!'''
        self.weights = weights
        

    def compute_multipole_bins(self):
        ''' This computes the multipole bins of the desired power spectrum, which depends on the pre-defined 
        number of bins, minimum and maximum multipole (which come from the image pixel size and FOV). Setting 
        self.logbin to True makes bins that are spaced equally in log space, rather than just linearly.'''

        if self.logbin:
            self.binl = 10**(np.linspace(np.log10(self.ell_min), np.log10(self.ell_max), self.nbins))
        else:
            self.binl = np.linspace(0, self.ell_max, self.nbins)
            
            
    def get_ell_bins(self, shift=True):

        ''' this will take input map dimensions along with a pixel scale (in arcseconds) and compute the 
        corresponding multipole bins for the 2d map.'''
        
        self.ell_max = np.sqrt(2)*180./self.pixlength_in_deg # maximum multipole determined by pixel size

        freq_x = fftshift(np.fft.fftfreq(self.dimx, d=1.0))*2*(180*3600./7.)
        freq_y = fftshift(np.fft.fftfreq(self.dimy, d=1.0))*2*(180*3600./7.)
        
        ell_x,ell_y = np.meshgrid(freq_x,freq_y)
        ell_x = ifftshift(ell_x)
        ell_y = ifftshift(ell_y)

        
        self.ell_map = np.sqrt(ell_x**2 + ell_y**2)
        
        print('minimum/maximum ell is ', np.min(self.ell_map[np.nonzero(self.ell_map)]), np.max(self.ell_map))

        if shift:
            self.ell_map = fftshift(self.ell_map)
            


    def check_precomputed_values(self, precompute_all=False, ell_map=False, binl=False, weights=False, ringmasks=False,\
                                masked_weight_sums=False, masked_weights=False, midbin_delta_ells=False, shift=False, sub_bins=False):
        
        
        if ell_map or precompute_all:
            if self.ell_map is None:
                print('Generating ell bins..')
                self.get_ell_bins(shift=shift)
                 
        if binl or precompute_all:
            if self.binl is None:
                print('Generating multipole bins..')
                self.compute_multipole_bins()
                
        if weights or precompute_all:
            # if no weights provided by user, weights are unity across image
            if self.weights is None:
                print('Setting Fourier weights to unity')
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
        
        
    def compute_cl_indiv(self, map1, map2=None, weights=None, mask=None, precompute=False, sub_bins=False):
        ''' this is for individual map realizations and should be more of a standalone, compared to Mkk.get_angular_spec(),
        which is optimized for computing many realizations in parallel.'''
        
        if precompute:
            self.check_precomputed_values(precompute_all=True, shift=True)

        fac = self.arcsec_pp_to_radian**2

        if map2 is None:
            map2 = map1
            fftsq = self.weights*((fft2(map1.real)*np.conj(fft2(map2.real))).real)

            
        C_ell = np.zeros(len(self.binl)-1)
        
        
        if sub_bins:
            C_ell_star = np.zeros(self.n_total_bins)

            for i in range(self.n_total_bins):
                if self.sub_ringmask_sums[i] > 0:
                    C_ell_star[i] = np.sum(self.masked_weights[i]*fftsq[np.array(self.ringmasks[i])])/self.masked_weight_sums[i]

            return C_ell_star
            
        else:
            C_ell = np.zeros(len(self.binl)-1)
            for i in range(len(self.binl)-1):
                if self.ringmask_sums[i] > 0:
                    C_ell[i] = np.sum(self.masked_weights[i]*fftsq[np.array(self.ringmasks[i])])/self.masked_weight_sums[i]

            return C_ell
    
    
    
    def plot_twopanel_Mkk(self, signal_map, mask, inverse_mkk, xlims=None, ylims=None, colorbar=True, return_fig=False, logbin=False):
    
        c_ell, dc_ell = self.compute_cl_indiv(signal_map)
        c_ell_masked, dc_ell_masked = self.compute_cl_indiv(signal_map*mask, mask=mask)

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
        
    def plot_tone_map_realization(self, tonemap, idx=None, return_fig=False):

        ''' produces an image of one tone map generated with map_from_powerspec()'''

        f = plt.figure()
        if idx is not None:
            plt.title('Tone map ('+str(np.round(self.binl[idx], 2))+'<$\\ell$<'+str(np.round(self.binl[idx+1], 2))+')', fontsize=16)
        else:
            plt.title('Tone map')
        plt.imshow(tonemap)
        plt.colorbar()
        plt.show()    
        
        if return_fig:
            return f


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
                c_ell_masked, dc_ell_masked = self.compute_cl_indiv(testmap*mask, mask=mask)

                for n, nsim in enumerate(nsims):

                    c_ell_rectified = np.dot(np.array(inverse_mkks[n]).transpose(), np.array(c_ell_masked))

                    abs_errors[j,k,n] = np.abs((c_ell[idx] - c_ell_rectified[idx])/c_ell[idx])
          
        
        if analytic_estimate:
            return abs_errors, np.array(dcl_cl_list)
        return abs_errors
    
    
    def compute_true_minus_rectified(self, mask, inverse_mkk, n_realizations=100, n_split=2, mode='white', sub_bins=False):

        delta_cl = []
        
            
        self.b = pyfftw.empty_aligned((n_realizations//n_split, self.dimx, self.dimy), dtype='complex64')        
        self.c = pyfftw.empty_aligned((n_realizations//n_split, self.dimx, self.dimy), dtype='complex64')
        self.d = pyfftw.empty_aligned((n_realizations//n_split, self.dimx, self.dimy), dtype='complex64')

        self.fft_object_b = pyfftw.FFTW(self.b, self.c, axes=(1,2),threads=1, direction='FFTW_BACKWARD', flags=('FFTW_MEASURE',))
        self.fft_object_c = pyfftw.FFTW(self.c, self.d, axes=(1,2),threads=1, direction='FFTW_FORWARD', flags=('FFTW_MEASURE',))

        for n in range(n_split):

            self.noise = np.random.normal(size=(n_realizations//n_split, self.dimx, self.dimy))+ 1j*np.random.normal(size=(n_realizations//n_split, self.dimx, self.dimy))

            self.map_from_powerspec(nsims=n_realizations//n_split, sub_bins=sub_bins, mode=mode)

            # c_ells is the "true", azimuthally averaged angular power spectrum
            c_ells = self.get_angular_spec(nsims=n_realizations//n_split, sub_bins=False, apply_mask=False)

            masked_Cls = self.get_angular_spec(nsims=n_realizations//n_split, sub_bins=sub_bins, apply_mask=True)


            for i, masked_Cl in enumerate(masked_Cls):

                c_ell_rect = np.dot(inverse_mkk.transpose(), np.array(masked_Cl))

                if sub_bins:
                    c_ell_rectified = np.array([np.mean(c_ell_rect[self.correspond_bins[k]]) for k in range(len(self.correspond_bins))])
                    delta_cl.append((c_ells[i]-c_ell_rectified)/c_ells[i])
                else:
                    delta_cl.append((c_ells[i]-c_ell_rect)/c_ells[i])

        return delta_cl
                    

def wrapped_mkk(nsims):
    x = Mkk(nbins=20)
    mask = np.ones((x.dimx, x.dimy))
    mkk, dc_ell = x.get_mkk_sim(mask, nsims, show_tone_map=False, n_split=1)
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
