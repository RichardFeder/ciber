import torch

from torch.fft import fft2, ifft2, fftshift, fftfreq

class Mkk_torch():
    
    def __init__(self, pixsize=7., dimx=1024, dimy=1024, ell_min=180., logbin=True, nbins=25, precompute=False):
        
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
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")  # Use the default CUDA device
        else:
            self.device = torch.device("cpu")
        
        print('Using device ', self.device)

        if precompute:
            self.precompute_mkk_quantities(precompute_all=True)


    def set_bin_edges(self, binl):
        self.binl = binl

    def compute_multipole_bins(self):
        ''' This computes the multipole bins of the desired power spectrum, which depends on the pre-defined 
        number of bins, minimum and maximum multipole (which come from the image pixel size and FOV). Setting 
        self.logbin to True makes bins that are spaced equally in log space, rather than just linearly.'''

        if self.logbin:
            self.binl = 10**(torch.linspace(np.log10(self.ell_min), np.log10(self.ell_max), self.nbins+1))
        else:
            self.binl = torch.linspace(self.ell_min, self.ell_max, self.nbins+1)
            
        self.binl = self.binl.to(self.device)
    
    def compute_midbin_delta_ells(self):
        self.midbin_ell = torch.tensor(0.5*(self.binl[1:]+self.binl[:-1]), device=self.device)
        self.delta_ell = torch.tensor(self.binl[1:]-self.binl[:-1], device=self.device)
            
    def get_ell_map(self, shift=True, verbose=False):

        ''' this will take input map dimensions along with a pixel scale (in arcseconds) and compute the 
        corresponding multipole bins for the 2d map.'''
        
        self.ell_max = np.sqrt(2)*180./self.pixlength_in_deg # maximum multipole determined by pixel size

        freq_x = fftshift(fftfreq(self.dimx, d=1.0))*2*(180*3600./self.pixsize)
        freq_y = fftshift(fftfreq(self.dimy, d=1.0))*2*(180*3600./self.pixsize)
        
        ell_x,ell_y = np.meshgrid(freq_x,freq_y)
        ell_x = ifftshift(ell_x)
        ell_y = ifftshift(ell_y)

        self.ell_map = torch.tensor(np.sqrt(ell_x**2 + ell_y**2))
        if verbose:
            print('minimum/maximum ell is ', torch.min(self.ell_map[torch.nonzero(self.ell_map)]), torch.max(self.ell_map))

        if shift:
            self.ell_map = fftshift(self.ell_map)
            
        print(self.ell_map.shape)
            
        self.ell_map = self.ell_map.to(self.device)
                
    def compute_ringmasks(self, verbose=False):    
        if verbose:    
            print('Minimum ell_map value is '+str(torch.min(self.ell_map)))

        self.unshifted_ringmasks, self.ringmasks = [torch.zeros((len(self.binl)-1, self.dimx, self.dimy), device=self.device) for x in range(2)]

        for i in range(len(self.binl)-1):

            ell_map_mask = (self.ell_map >= self.binl[i])*(self.ell_map <= self.binl[i+1])
            self.unshifted_ringmasks[i] = fftshift(ell_map_mask.float()/self.arcsec_pp_to_radian)
            self.ringmasks[i] = (fftshift(self.ell_map) >= self.binl[i])*(fftshift(self.ell_map) <= self.binl[i+1])

        self.ringmask_sums = torch.tensor([torch.sum(ringmask) for ringmask in self.ringmasks], device=self.device)
        
    def compute_masked_weights(self, sub_bins=False, device=None):
        ''' This takes the Fourier weights and produces a  list of masked weights according to each multipole bandpass. The prefactor just converts the units
        appropriately. This is precomputed to make things faster during FFT time''' 
        fac = self.arcsec_pp_to_radian**2

        
        self.masked_weights = [fac*self.weights[ringmask.bool()] for ringmask in self.ringmasks]
        
        # print(self.masked_weights)
        # self.masked_weights = torch.tensor(self.masked_weights.to(self.device)
            
    def compute_masked_weight_sums(self,  verbose=False):
        self.masked_weight_sums = [torch.sum(self.weights[ringmask.bool()]) for ringmask in self.ringmasks]
        if verbose:
            print('self.masked weight sums:', self.masked_weight_sums)
        
    def precompute_mkk_quantities(self, precompute_all=False, weights=False, ell_map=False, binl=False, ringmasks=False,\
                                masked_weight_sums=False, masked_weights=False, midbin_delta_ells=False, shift=True, sub_bins=False, \
                                input_cl_func=None, verbose=False):
        
        
        if ell_map or precompute_all:
            if verbose:
                print('Generating 2D ell map..')
            self.get_ell_map(shift=shift)

        if weights or precompute_all:
            # if no weights provided by user, weights are unity across image
            if self.weights is None:
                self.weights = torch.ones((self.dimx, self.dimy), device=self.device)   
                
        if binl or precompute_all:
            self.compute_multipole_bins()
            if verbose:
                print('Generating multipole bins..')
                print('Multipole bin edges:', self.binl)
                
        if ringmasks or precompute_all:
            if self.ringmasks is None or sub_bins:
                if verbose:
                    print('Computing Fourier ring masks..')
                self.compute_ringmasks(verbose=verbose)
              
        if masked_weights or precompute_all:
            self.compute_masked_weights()

        if masked_weight_sums or precompute_all:
            self.compute_masked_weight_sums()
            
        if midbin_delta_ells or precompute_all:
            self.compute_midbin_delta_ells()
            
    def get_ensemble_angular_autospec(self, nsims=2, apply_mask=False):
                 

        self.fft_objs[1](self.empty_aligned_objs[obj_idx0].real)
        
        fftsq = [(dentry*np.conj(dentry)).real for dentry in self.empty_aligned_objs[obj_idx1]]
            
        C_ell = np.zeros((nsims, len(self.binl)-1))
        for i in range(len(self.binl)-1):
            if self.ringmask_sums[i] > 0:
                C_ell[:,i] = np.array([np.sum(self.masked_weights[i]*fftsq[j][self.ringmasks[i]])/self.masked_weight_sums[i] for j in range(nsims)])

        return C_ell
        
    def calc_mkk(self, mask, nsims, n_split=1, precompute_all=False, show_tone_map=False):

        nrealiz = nsims//n_split
        
        fac = self.arcsec_pp_to_radian**2

        self.mask = torch.tensor(mask).to(self.device)

        maplist_split_shape = (nrealiz, self.dimx, self.dimy)
        Mkks, dC_ell_list = [], []

        Mkks = torch.zeros((nsims, self.nbins, self.nbins), device=self.device)

        if precompute_all:
            self.precompute_mkk_quantities(precompute_all=True, shift=True)

        for i in range(n_split):
            
            print('on split ', i, 'of ', n_split)

            gauss = torch.randn(maplist_split_shape)+1j*torch.randn(maplist_split_shape)
            gauss = gauss.to(self.device)

            for j in range(self.nbins):

                im = ifft2(gauss*self.unshifted_ringmasks[j]).real

                if show_tone_map:
                    self.plot_tone_map_realization(im[0].cpu().numpy(), idx=title_idx)

                fftim = fft2(torch.mul(im, self.mask))

                fftsq = (fftim*torch.conj(fftim)).real

                masked_Cl = torch.zeros((nrealiz, len(self.binl)-1))

                for k in range(len(self.binl)-1):
                    
                    mul = torch.mul(fftsq, self.ringmasks[k])/self.masked_weight_sums[k]
                    masked_Cl[:,k] = fac*torch.sum(mul, axis=(1, 2))
                    
                Mkks[i*nrealiz:(i+1)*nrealiz ,j,:] = masked_Cl

        av_Mkk = torch.mean(Mkks, axis=0)

        return av_Mkk



