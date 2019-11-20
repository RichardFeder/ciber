import numpy as np
from hankel import SymmetricFourierTransform
from scipy import interpolate
from scipy import stats


def counts_from_density_2d(density_fields, Ntot = 200000, ell_min=90.):
    counts_map = np.zeros_like(density_fields)
    
    surface_area = (np.pi/ell_min)**2

    # compute the mean number density per cell 
    N_mean = float(Ntot) / float(counts_map.shape[-2]*counts_map.shape[-1])
    
    # calculate the expected number of galaxies per cell
    expectation_ngal = (density_fields+1.)*N_mean 
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
            for j in xrange(np.abs(dcounts[i])):
                counts_map[i][randx[j], randy[j]] += 1

    return count_maps

''' This converts multipoles to wavenumbers given some comoving distance (fixed redshift). '''
def ell_to_k(ell, comoving_dist):
    theta = np.pi/ell
    k = 1./(comoving_dist*theta)
    return k

''' This function generates the indices needed when generating G(k), which gets Fourier transformed to a gaussian
random field with some power spectrum.'''
def fftIndgen(n):
    a = range(0, n/2+1)
    b = range(1, n/2)
    b.reverse()
    b = [-i for i in b]
    return a + b

''' This is not fully working as of now, maybe some day in the future I'll need this! '''
def gaussian_random_field_3d(n_samples, alpha=None, size=128, ell_min=90., ps=None, ksampled=None, z=None, fac=1, plot=False):
    
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


def gaussian_random_field_2d(n_samples, alpha=None, size=128, ell_min=90., cl=None, ell_sampled=None, fac=1, plot=False):
    
    up_size = size*fac
    steradperpixel = ((np.pi/ell_min)/up_size)**2
    surface_area = (np.pi/ell_min)**2

    cl_g, spline_cl_g = hankel_spline_lognormal_cl(ell_sampled, cl)
    ls = make_ell_grid(up_size, ell_min=ell_min)
    amplitude = 10**spline_cl_g(np.log10(ls))
    amplitude /= surface_area
    
    amplitude[0,0] = 0.
    
    noise = np.random.normal(size = (n_samples, up_size,up_size)) + 1j * np.random.normal(size = (n_samples, up_size, up_size))
#     gfield = np.array([np.fft.ifft2(n * amplitude * up_size**2 ).real for n in noise])
    gfield = np.array([np.fft.ifft2(n * amplitude).real for n in noise])
    gfield /= steradperpixel
    
    # up to this point, the gaussian random fields have mean zero
    return gfield, amplitude, ls

def generate_count_map_2d(n_samples, ell_min=90., size=128, Ntot=2000000, cl=None, ell_sampled=None, plot=False, save=False):
    # we want to generate a GRF twice the size of our final GRF so we include larger scale modes
    realgrf, amp, k = gaussian_random_field_2d(n_samples, size=size, cl=cl, ell_sampled=ell_sampled, ell_min=ell_min/2.,fac=2)
    realgrf = realgrf[:,int(0.5*size):int(1.5*size),int(0.5*size):int(1.5*size)]
    
    density_fields = np.array([np.exp(grf-np.std(grf)**2)-1. for grf in realgrf]) # assuming a log-normal density distribution
    density_fields = np.array([df-np.mean(df) for df in density_fields]) # this ensures that each field has mean zero
    
    counts = counts_from_density_2d(density_fields, Ntot=Ntot)

    return counts, density_fields



''' This function takes an angular power spectrum and computes the corresponding power spectrum for the log 
of the density field with that power spectrum. This involves converting the power spectrum to a correlation function, 
computing the lognormal correlation function, and then transforming back to ell space to get C^G_ell.'''
def hankel_spline_lognormal_cl(ells, cl, plot=False, ell_min=90, ell_max=1e5):
    
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
    size_ind = np.array(fftIndgen(size))
    for i, sx in enumerate(size_ind):
        ell_grid[i,:] = np.sqrt(sx**2+size_ind**2)*ell_min
    return ell_grid





