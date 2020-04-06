import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fftshift as fftshift
from numpy.fft import ifftshift as ifftshift
import multiprocessing
import pyfftw
import time

class Mkk():
    
    def __init__(self, pixsize=7., dimx=1024, dimy=1024, ell_min=80., logbin=True, nbins=30):
        
        for attr, valu in locals().items():
            setattr(self, attr, valu)
            
        self.pixlength_in_deg = self.pixsize/3600. # pixel angle measured in degrees
        self.ell_max = 180./self.pixlength_in_deg # maximum multipole determined by pixel size
        self.arcsec_pp_to_radian = self.pixlength_in_deg*(np.pi/180.) # conversion from arcsecond per pixel to radian per pixel
        
        self.sqdeg_in_map = self.dimx*self.dimy*self.pixlength_in_deg**2 # area of map in square degrees
        self.fsky = self.sqdeg_in_map/(4*np.pi*(180./np.pi)**2) # fraction of sky covered by map
        
        self.ell_map = None # used to compute ring masks in fourier space
        self.binl = None # radially averaged multipole bins
        self.weights = None # fourier weights for angular power spectrum calculation
        
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
        
        freq_x = fftshift(np.fft.fftfreq(self.dimx))
        freq_y = fftshift(np.fft.fftfreq(self.dimy))

        ell_x,ell_y = np.meshgrid(freq_x,freq_y)

        ell_x = ifftshift(ell_x)*self.ell_max
        ell_y = ifftshift(ell_y)*self.ell_max

        self.ell_map = np.sqrt(ell_x**2 + ell_y**2)

        if shift:
            self.ell_map = fftshift(self.ell_map)
            
    def map_from_powerspec(self, idx, shift=True,nsims=1):

        ''' This makes maps very quickly given some precomputed quantities. The unshifted ringmasks are used to generate
        the pure tones within each band, so then this gets multiplied by the precalculated noise and fourier transformed to real space. These
        maps are then stored in the variable self.c (self.fft_object_b goes from self.b --> self.c)'''
    
        if self.ell_map is None:
            self.get_ell_bins(shift=shift)
            
        if self.binl is None:
            self.compute_multipole_bins()
         
        t0 = time.clock()
        self.fft_object_b(self.unshifted_ringmasks[idx]*self.noise)

        print('fft object b takes ', time.clock()-t0)


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

    def compute_masked_weights(self):
        ''' This takes the Fourier weights and produces a list of masked weights according to each multipole bandpass. The prefactor just converts the units
        appropriately. This is precomputed to make things faster during FFT time''' 
        fac = self.arcsec_pp_to_radian**2

        self.masked_weights = [fac*self.weights[ringmask] for ringmask in self.ringmasks]

    
    def get_mkk_sim(self, mask, nsims, show_tone_map=False, mode='auto', n_split=1):
        
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

        if n_split > 1:
            print 'Splitting up computation of', nsims, 'simulations into', n_split, 'chunks..'
            assert (nsims % n_split) == 0

        self.mask = mask

        self.b = pyfftw.empty_aligned((nsims//n_split, self.dimx, self.dimy), dtype='complex64')        
        self.c = pyfftw.empty_aligned((nsims//n_split, self.dimx, self.dimy), dtype='complex64')
        self.d = pyfftw.empty_aligned((nsims//n_split, self.dimx, self.dimy), dtype='complex64')

        self.fft_object_b = pyfftw.FFTW(self.b, self.c, axes=(1,2), threads=multiprocessing.cpu_count(), direction='FFTW_BACKWARD', flags=('FFTW_MEASURE',))
        self.fft_object_c = pyfftw.FFTW(self.c, self.d, axes=(1,2), threads=multiprocessing.cpu_count(), direction='FFTW_FORWARD', flags=('FFTW_MEASURE',))
        
        Mkks = []
        dC_ell_list = []
        for i in range(n_split):
            print('split ', i+1, 'of', n_split)

            # compute noise realizations beforehand so time isn't wasted generating a whole new set each time. We can do this because the ring masks
            # that we multiply by the noise realizations are spatially disjoint, i.e. the union of all of the masks is the null set. 
            self.noise = np.random.normal(size=(nsims//n_split, self.dimx, self.dimy))+ 1j*np.random.normal(size=(nsims//n_split, self.dimx, self.dimy))

            if self.ell_map is None:
                self.get_ell_bins(shift=True)
                
            if self.binl is None:
                self.compute_multipole_bins()
            
            self.compute_midbin_delta_ells()

      
            # if no weights provided by user, weights are unity across image
            if self.weights is None:
                self.weights = np.ones((self.dimx, self.dimy))
                
            Mkk = np.zeros((self.nbins-1, self.nbins-1))
            
            # lets precompute a few quantities for later
            self.compute_ringmasks()
            self.compute_masked_weight_sums()
            self.compute_masked_weights()

                    
            for j in range(self.nbins-1):
                print('band ', j)
                
                self.map_from_powerspec(j, nsims=nsims//n_split)
                
                # self.plot_tone_map_realization(self.c[0].real*mask)
                # self.plot_tone_map_realization(tonemaps[1]*mask)
                
                if mode=='auto':
                    
                    masked_Cl, dC_ells = self.get_angular_spec(nsims=nsims//n_split)

                    # row entry for Mkk is average over realizations
                    Mkk[j,:] = np.mean(np.array(masked_Cl), axis=0)

            for dC_ell in dC_ells:
                dC_ell_list.append(dC_ell)

            Mkks.append(Mkk)

    
        if n_split == 1:
            return Mkks[0], dC_ells

        else:
            av_Mkk = np.mean(np.array(Mkks), axis=0)
            print('av_Mkk has shape ', av_Mkk.shape)

            print('dc_ell list has shape ', np.array(dC_ell_list).shape)

            return av_Mkk, np.array(dC_ell_list)
    
    
    def get_angular_spec(self, map1=None, map2=None, nsims=2):
        
        fac = self.arcsec_pp_to_radian**2
        
        if map2 is None:
            t0 = time.clock()

            self.c *= self.mask

            self.fft_object_c(self.c.real)
            print('fft_object_c took ', time.clock()-t0)

            t0 = time.clock()
            fftsq = [(dentry*np.conj(dentry)).real for dentry in self.d]
            print('fftsq took ', time.clock()-t0)
        else:
            # this works, but why ifft2 here and not fft2? 
            fftsq = self.weights*((ifft2(map1)*np.conj(ifft2(map2))).real)*fac

        # t0 = time.clock()
        # C_ell = np.array([[np.sum(self.masked_weights[i]*fftsq[j][self.ringmasks[i]])/self.masked_weight_sums[i] for j in range(nsims)] for i in range(len(self.binl)-1)]).transpose()
        # # C_ell = np.array([[np.sum(self.masked_weights[i]*fftsq[j][self.ringmasks[i]])/self.masked_weight_sums[i] for i in range(len(self.binl)-1)] for j in range(nsims)])

        t0 = time.clock()
        C_ell = np.zeros((nsims, len(self.binl)-1))
        for i in range(len(self.binl)-1):
            if self.ringmask_sums[i] > 0:
                C_ell[:,i] = np.array([np.sum(self.masked_weights[i]*fftsq[j][self.ringmasks[i]])/self.masked_weight_sums[i] for j in range(nsims)])

        # lets return cosmic variance error as well

        dC_ell = C_ell*np.sqrt(2./((2*self.midbin_ell + 1)*(self.delta_ell*self.fsky)))
        print('dC_ell has shape ', dC_ell.shape)


        return C_ell, dC_ell
    
    
    def compute_ringmasks(self):
        
        self.ringmasks = [(fftshift(self.ell_map) >= self.binl[i])*(fftshift(self.ell_map) <= self.binl[i+1]) for i in range(len(self.binl)-1)]
        self.ringmask_sums = np.array([np.sum(ringmask) for ringmask in self.ringmasks])
        self.unshifted_ringmasks = [fftshift(np.array((self.ell_map >= self.binl[i])*(self.ell_map <= self.binl[i+1])).astype(np.float)/self.arcsec_pp_to_radian) for i in range(len(self.binl)-1)]
        
        # print([np.sum(self.unshifted_ringmasks[i]*self.unshifted_ringmasks[i+1]) for i in range(10)]) 
        print('ringmask sums:')
        print(self.ringmask_sums)
        
    def compute_masked_weight_sums(self):
        self.masked_weight_sums = np.array([np.sum(self.weights[ringmask]) for ringmask in self.ringmasks])

    def compute_midbin_delta_ells(self):
        self.midbin_ell = 0.5*(self.binl[1:]+self.binl[:-1])
        self.delta_ell = self.binl[1:]-self.binl[:-1]
        

def compute_inverse_mkk(mkk):
    
    inverse_mkk = np.linalg.inv(mkk)
    
    return inverse_mkk


def plot_mkk_matrix(mkk, inverse=False, logscale=False):
    if inverse:
        title = '$M_{\\ell \\ell^\\prime}^{-1}$'
    else:
        title = '$M_{\\ell \\ell^\\prime}$'
    
    plt.figure(figsize=(6,6))
    plt.title(title, fontsize=16)
    plt.imshow(mkk)
    plt.colorbar()
    plt.xticks(np.arange(mkk.shape[0]))
    plt.yticks(np.arange(mkk.shape[1]))

    plt.show()



# this code below is just to do a test run, should ultimately comment out and call routines from another script

# x = Mkk(nbins=20)

# mask = np.ones((x.dimx, x.dimy))

# # mask[100:700,200:500] = 0.
# # mask[800:810,600:620] = 0.
# # mask[800:810,300:320] = 0.
# # mask[600:660,910:920] = 0.


# tzero = time.clock()
# mkk, dc_ell = x.get_mkk_sim(mask, 200, show_tone_map=False, n_split=4)


# print('elapsed time is ', time.clock()-tzero)
# plot_mkk_matrix(mkk)

# print([mkk[i,i] for i in range(mkk.shape[0])])

# mkk_inv = compute_inverse_mkk(mkk)
# plot_mkk_matrix(mkk_inv, inverse=True)

# print(mkk)
