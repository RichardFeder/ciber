import numpy as np
from hankel import SymmetricFourierTransform
from scipy import interpolate
from scipy import stats
import sys


def counts_from_density_2d(overdensity_fields, Ntot = 200000):
    
    ''' This function takes in a collection of number density fields and
    for each generates a poisson realization to get galaxy counts in each cell of the field.
    When a Poisson realization is taken and a specific number of galaxies is desired, a small 
    correction is needed to either remove or add sources. Sources are added/removed with uniform probabilities.

    Inputs:
        overdensity_fields (np.array): array of overdensity fields, which are usually obtained with gaussian_random_field_2d() and then exponentiated.
        
        Ntot (int, default=200000): number of source positions to sample from density field
    
    Output:
        count_maps (np.array): array of counts maps. This has the same shape as overdensity_fields

    '''
    counts_map = np.zeros_like(overdensity_fields)
    
    # compute the mean number density per cell 
    N_mean = float(Ntot) / float(counts_map.shape[-2]*counts_map.shape[-1])
    
    # calculate the expected number of galaxies per cell
    expectation_ngal = (overdensity_fields+1.)*N_mean 
    # generate Poisson realization of 2D field
    count_maps = np.random.poisson(expectation_ngal)
    
    dcounts = np.array([int(np.sum(ct_map)-Ntot) for ct_map in count_maps])
    # if more counts in the poisson realization than Helgason number counts, subtract counts
    # from non-zero elements of array.
    # The difference in counts is usually at the sub-percent level, so adding counts will only slightly
    # increase shot noise, while subtracting will slightly decrease power at all scales
    for i in np.arange(count_maps.shape[0]):
        if dcounts[i] > 0:
            nonzero_x, nonzero_y = np.nonzero(count_maps[i])
            rand_idxs = np.random.choice(np.arange(len(nonzero_x)), dcounts[i], replace=True)
            count_maps[i][nonzero_x[rand_idxs], nonzero_y[rand_idxs]] -= 1
        elif dcounts[i] < 0:
            randx, randy = np.random.choice(np.arange(count_maps[i].shape[-1]), size=(2, np.abs(dcounts[i])))
            for j in range(np.abs(dcounts[i])):
                counts_map[i][randx[j], randy[j]] += 1

    return count_maps

def ell_to_k(ell, comoving_dist):
    ''' This converts multipoles to wavenumbers given some comoving distance (fixed redshift). '''

    theta = np.pi/ell
    k = 1./(comoving_dist*theta)
    return k


def fftIndgen2(n):
    ''' This function generates the indices needed when generating G(k), which gets Fourier transformed to a gaussian
    random field with some power spectrum.'''
    
    a = range(0, n//2+1)
    b = range(1, n//2)
    b.reverse()
    b = [-i for i in b]
    return a + b

def fftIndgen3(n):
    ''' This function generates the indices needed when generating G(k), which gets Fourier transformed to a gaussian
    random field with some power spectrum.'''
    
    a = list(range(0, n//2+1))
    b = reversed(range(1, n//2))
    b = [-i for i in b]
    return a + b

def gaussian_random_field_3d(n_samples, alpha=None, size=128, ell_min=90., ps=None, ksampled=None, z=None, fac=1, plot=False):
    ''' This is not fully working as of now, maybe some day in the future I'll need this! '''

    k_ind = np.mgrid[:size*fac, :size*fac, :size*fac] - int( (size*fac + 1)/2 )
    k_ind = scipy.fftpack.fftshift(k_ind)
    k_idx = ( k_ind )
    ks = np.sqrt(k_idx[0]**2 + k_idx[1]**2 + k_idx[2]**2)
        
    r, xi = P2xi(ksampled)(ps)

    # transform the two point correlation function to the corresponding 2pt in the gaussian random field
    xi_g = np.log(1.+xi)
    # fourier transform back to get P_G(k)
    ksamp, p_G = xi2P(r)(xi_g)
    
    if z is not None:
        k_min = ell_min / cosmo.angular_diameter_distance(z)
        k_min /= float(fac)
        ks *= k_min
        
    vol = (1./k_min.value)**3
    
    spline_G = interpolate.InterpolatedUnivariateSpline(np.log10(ksamp[:-50]), np.log10(p_G[:-50]))
    amplitude = 10**(spline_G(np.log10(ks)))
    amplitude[0,0,0] = 0.
    
    noise = np.random.normal(size = (n_samples, size*fac, size*fac, size*fac)) + 1j * np.random.normal(size = (n_samples, size*fac, size*fac, size*fac))
    gfield = np.array([np.fft.ifftn(n * np.sqrt(amplitude*vol/2)).real for n in noise])
    
    return gfield, amplitude, ks


def gaussian_random_field_2d(n_samples, cl, ell_sampled,  size=128, ell_min=90., fac=1):
    ''' Generate a number of independent 2D gaussian random fields given some input angular power spectrum C_ell.
    To account for the impact of larger spatial modes than the desired FOV, this function generates a field "fac" times as large as specified, 
    after which one can crop for the desired central region. This also means that the angular power spectrum input should have 
    sufficient multipole coverage "fac" times larger than the FOV. This function works by computing a spline interpolation of the input 
    power spectrum, which is then used in lognormal calculations specified by Carron et al. 2014 (arxiv:1406.6072v2). 

    Inputs:
        n_samples (int): number of desired GRF samples
        
        cl (np.array): input angular power spectrum
        
        ell_sampled (np.array): multipole bins corresponding to input angular power spectrum
        
        size (int, default=128): side dimension of desired GRF field in pixels
        
        ell_min (float, default=90.): minimum multipole of desired **scaled** FOV. This should be (ell_min of desired FOV)/fac
        
        fac (int, default=1): to account for larger than FOV modes, this makes this function increase the angular coverage of the 
            GRF in each dimension by a desired factor. fac=2 generally works sufficiently well.

    Outputs:
        gfield (array of np.arrays): the output gaussian random fields, which are not cropped from upscaled size determined by fac. 
            These fields should each have mean zero
        
        amplitude (np.array): 2D angular power spectrum used to generate GRFs
        
        ells (np.array): 2D grid of multipoles used as input when evaluating C_ell over grid that yields amplitude
    '''
    up_size = int(size*fac)
    steradperpixel = ((np.pi/ell_min)/up_size)**2
    surface_area = (np.pi/ell_min)**2

    cl_g, spline_cl_g = hankel_spline_lognormal_cl(ell_sampled, cl)
    ells = make_ell_grid(up_size, ell_min=ell_min)
    amplitude = 10**spline_cl_g(np.log10(ells))
    amplitude /= surface_area
    
    amplitude[0,0] = 0.
    
    noise = np.random.normal(size = (n_samples, up_size,up_size)) + 1j * np.random.normal(size = (n_samples, up_size, up_size))
#     gfield = np.array([np.fft.ifft2(n * amplitude * up_size**2 ).real for n in noise])
    gfield = np.array([np.fft.ifft2(n * amplitude).real for n in noise])
    gfield /= steradperpixel
    
    # up to this point, the gaussian random fields have mean zero
    return gfield, amplitude, ells

def generate_count_map_2d(n_samples, ell_min=90., size=128, Ntot=2000000, cl=None, ell_sampled=None,  fac=2.):
    ''' 

    This combines two of the other functions in this script to generate a number of indepedent source count maps. 
    First, a number of gaussian random fields are generated, after which they are cropped to the desired FOV, exponentiated,
    and then sampled to obtain source count maps. 

    Inputs:
        n_samples (int): number of sample realizations
        ell_min (float, default=90.): minimum mulitpole for desired **final** FOV. Note this is slightly different than ell_min in gaussian_random_field_2d().
        size (int, default=128): side dimension of counts maps in pixels
        Ntot (int, default=2000000): number of sources to sample in count maps. This is usually set by Helgason number counts in practice
        fac (float, default=2): factor by which to scale up generated fields before cropping to desired FOV size. This allows one to account for 
            spatial modes larger than the FOV that contribute to observable clustering. 

    Outputs:
        counts (array of np.arrays): source count maps
        overdensity_fields (array of np.arrays): corresponding overdensity fields for count maps

    '''

    realgrf, amp, k = gaussian_random_field_2d(n_samples, size=size, cl=cl, ell_sampled=ell_sampled, ell_min=ell_min/fac, fac=fac)
    
    cropped_grf = realgrf[:, int(0.5*(realgrf.shape[1]-size)):int(0.5*(realgrf.shape[1]+size)), int(0.5*(realgrf.shape[2]-size)):int(0.5*(realgrf.shape[2]+size))]
    # realgrf = realgrf[:,int(0.5*size):int(1.5*size),int(0.5*size):int(1.5*size)]
    
    overdensity_fields = np.array([np.exp(grf-np.std(grf)**2)-1. for grf in cropped_grf]) # assuming a log-normal density distribution
    
    counts = counts_from_density_2d(overdensity_fields, Ntot=Ntot)

    return counts, overdensity_fields




def hankel_spline_lognormal_cl(ells, cl, plot=False, ell_min=90, ell_max=1e5):
    ''' This function takes an angular power spectrum and computes the corresponding power spectrum for the log 
    of the density field with that power spectrum. This involves converting the power spectrum to a correlation function, 
    computing the lognormal correlation function, and then transforming back to ell space to get C^G_ell.
    '''
    
    ft = SymmetricFourierTransform(ndim=2, N = 200, h = 0.03)
    # transform to angular correlation function with inverse hankel transform and spline interpolated C_ell
    spline_cl = interpolate.InterpolatedUnivariateSpline(np.log10(ells), np.log10(cl))
    f = lambda ell: 10**(spline_cl(np.log10(ell)))
    thetas = np.pi/ells
    w_theta = ft.transform(f ,thetas, ret_err=False, inverse=True)
    # compute lognormal angular correlation function
    w_theta_g = np.log(1.+w_theta)
    # convert back to multipole space
    spline_w_theta_g = interpolate.InterpolatedUnivariateSpline(np.log10(np.flip(thetas)), np.flip(w_theta_g))
    g = lambda theta: spline_w_theta_g(np.log10(theta))
    cl_g = ft.transform(g ,ells, ret_err=False)
    spline_cl_g = interpolate.InterpolatedUnivariateSpline(np.log10(ells), np.log10(np.abs(cl_g)))
    
    # plotting is just for validation if ever unsure
    if plot:
        plt.figure()
        plt.loglog(ells, cl, label='$C_{\\ell}$', marker='.')
        plt.loglog(ells, cl_g, label='$C_{\\ell}^G$', marker='.')
        plt.loglog(np.linspace(ell_min, ell_max, 1000), 10**spline_cl_g(np.log10(np.linspace(ell_min, ell_max, 1000))), label='spline of $C_{\\ell}^G$')
        plt.legend(fontsize=14)
        plt.xlabel('$\\ell$', fontsize=16)
        plt.title('Angular power spectra', fontsize=16)
        plt.show()
    
    return cl_g, spline_cl_g

''' This is used when making gaussian random fields '''
def make_ell_grid(size, ell_min=90.):
    ell_grid = np.zeros(shape=(size,size))

    if sys.version_info[0] < 3:
        size_ind = np.array(fftIndgen2(int(size)))
    else:
        size_ind = np.array(fftIndgen3(int(size)))

    for i, sx in enumerate(size_ind):
        ell_grid[i,:] = np.sqrt(sx**2+size_ind**2)*ell_min
    return ell_grid







