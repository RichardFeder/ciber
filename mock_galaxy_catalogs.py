import numpy as np
# from compos import matterps
# from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
# from astropy import constants as const
from scipy import stats
# import time
from halo_model import *
from helgason import *
import camb
# from camb import model, initialpower
from hankel import SymmetricFourierTransform
from scipy import interpolate
from hmf import MassFunction

pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(ns=0.965)

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
    
    if plot:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.title('Gaussian random field $A$')
        plt.imshow(realgrf[0])
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.title('Dimensionless fluctuations $\\rho = e^A$')
        plt.imshow(density_fields[0])
        plt.colorbar()
        if save:
            plt.savefig('../figures/grf_rho.png', bbox_inches='tight')
        plt.show()

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


def hsc_positions(hsc_cat, nchoose, z, dz, ra_min=241.5, ra_max=243.5, dec_min=54.0, dec_max=56.0, rmag_max=29.0):
    restricted_cat = hsc_cat[(hsc_cat['ra']>ra_min)&(hsc_cat['ra']<ra_max)&(hsc_cat['dec']>dec_min)&(hsc_cat['dec']<dec_max)
        &(hsc_cat['photoz_best']< z+0.5*dz)&(hsc_cat['photoz_best']>z-0.5*dz)&(hsc_cat['rmag_psf']<rmag_max)]
    
    dx1 = np.max(restricted_cat['x1'])-np.min(restricted_cat['x1'])
    dy1 = np.max(restricted_cat['y1'])-np.min(restricted_cat['y1'])
    restricted_cat['x1'] -= np.min(restricted_cat['x1'])
    restricted_cat['x1'] *= (float(size-1.)/dx1)
    restricted_cat['y1'] -= np.min(restricted_cat['y1'])
    restricted_cat['y1'] *= (float(size-1.)/dy1)
            
    n_hsc = len(hsc_cat[(hsc_cat['ra']>ra_min)&(hsc_cat['ra']<ra_max)&(hsc_cat['dec']>dec_min)&(hsc_cat['dec']<dec_max)
                  &(hsc_cat['photoz_best']< z+0.5*dz)&(hsc_cat['photoz_best']>z-0.5*dz)&(hsc_cat['rmag_psf']<rmag_max)])
    print('input number_counts:', np.sum(number_counts[i])*4, 'choosing from ', n_hsc, 'galaxies')

    idxs = np.random.choice(np.arange(n_hsc), nchoose, replace=True)
    tx = restricted_cat.iloc[idxs]['x1']+np.random.uniform(-0.5, 0.5, size=len(idxs))
    ty = restricted_cat.iloc[idxs]['y1']+np.random.uniform(-0.5, 0.5, size=len(idxs))
    
    return tx, ty

''' Convert wavenumber to multipole for fixed redshift '''
def k_to_ell(k, comoving_dist):
    theta = 1./(comoving_dist*k)
    ell = np.pi/theta
    return ell

''' This is used when making gaussian random fields'''
def make_ell_grid(size, ell_min=90.):
    ell_grid = np.zeros(shape=(size,size))
    size_ind = np.array(fftIndgen(size))
    for i, sx in enumerate(size_ind):
        ell_grid[i,:] = np.sqrt(sx**2+size_ind**2)*ell_min
    return ell_grid

''' this is just a convenience function for when I want to convert a bunch of lists to numpy arrays'''
def make_lists_arrays(list_of_lists):
    list_of_arrays = []
    for l in list_of_lists:
        list_of_arrays.append(np.array(l))
    return list_of_arrays

''' Given a counts map, generate source catalog positions consistent with those counts. 
Not doing any subpixel position assignment or anything like that.''' 
def positions_from_counts(counts_map, cat_len=None):
    thetax, thetay = [], []
    for i in np.arange(np.max(counts_map)):
        pos = np.where(counts_map > i)
        thetax.extend(pos[0].astype(np.float))
        thetay.extend(pos[1].astype(np.float))

    if cat_len is not None:
        idxs = np.random.choice(np.arange(len(thetax)), cat_len)
        thetax = np.array(thetax)[idxs]
        thetay = np.array(thetay)[idxs]

    return np.array(thetax), np.array(thetay)


def w_theta(theta, A=0.2, gamma=1.8):
    return A*theta**(1.-gamma)

def two_pt_r(r, r0=6., gamma=1.8):
    ''' Returns the two point correlation function at a given comoving distance r'''
    return (r/r0)**(-gamma)


''' 

I want to construct random catalogs drawn from the galaxy distribution specified by Helgason et al.
This involves:
- drawing redshifts from dN/dz - done
- drawing absolute magnitudes from Phi(M|z) - done
- converting to apparent magnitudes, and then to flux units from AB - done
- computing an estimated mass for the galaxy - done
- get virial radius for IHL component - done
- generate positions consistent with two point correlation function of galaxies - done!

'''

class galaxy_catalog():
    
    lf = Luminosity_Function()
    halomod = halo_model_class()

    mass_function = MassFunction(z=0., dlog10m=0.02)

    def __init__(self, band='J', ell_min=90., ell_max=1e5):
        self.band = band
        self.ell_min = ell_min
        self.ell_max = ell_max 
    
    def ab_match_percentiles(self, mabs, mabs_largesamp):
        mass_range, dndm = self.load_halo_mass_function('../data/halo_mass_function_hmfcalc.txt')
        many_halo_masses = np.random.choice(mass_range, len(mabs_largesamp), p=dndm)
        halo_masses = []
        # compute percentile of absolute mag relative to large sample, then use that to compute halo mass given percentile        
        for m in mabs:
            halo_masses.append(np.percentile(many_halo_masses, stats.percentileofscore(mabs_largesamp, m)))

        halo_masses = np.sort(halo_masses)[::-1] # maybe not?

        virial_radii = self.halomod.mass_2_virial_radius(halo_masses) # in Mpc

        return halo_masses, virial_radii  

    def abundance_match_ms_given_mags(self, zs):
        ''' here we want to sort these in descending order, since abs mags are ordered and most neg absolute
        magnitude corresponds to most massive halo'''
        ngal = len(zs)
        mass_range, dndm = self.load_halo_mass_function('../data/halo_mass_function_hmfcalc.txt')
        halo_masses = np.sort(np.random.choice(mass_range, ngal,p=dndm))[::-1]
        halo_masses *= u.solMass
        virial_radii = self.halomod.mass_2_virial_radius(halo_masses, zs).to(u.Mpc) # in Mpc
        
        return halo_masses, virial_radii.value

    def draw_mags_given_zs(self, Mabs, gal_zs, ngal_per_z, pdfs, zs):
        '''we are going to order these by absolute magnitude, which makes things easier when abundance matching to 
        halo mass'''
        gal_app_mags = np.array([])
        gal_abs_mags = np.array([])
        for i in xrange(len(ngal_per_z)):
            absolute_mags = np.random.choice(Mabs, ngal_per_z[i], p=pdfs[i])
            apparent_mags = apparent_mag_from_absolute(absolute_mags, zs[i])
            gal_app_mags = np.append(gal_app_mags, apparent_mags)
            gal_abs_mags = np.append(gal_abs_mags, absolute_mags)

        print(len([m for m in gal_app_mags if m < 20]), 'under 20')
        print(len([m for m in gal_app_mags if m < 22]), 'under 22')
        print(len([m for m in gal_app_mags if m < 24]), 'under 24')
                        
        arr = np.array([gal_zs, gal_app_mags, gal_abs_mags]).transpose()
        cat = arr[np.argsort(arr[:,2])] # sort apparent and absolute mags by absolute mags

        return cat

    def draw_redshifts(self, Nsrc, zmin, zmax, Mabs, band='H'):
        zfine = np.linspace(zmin, zmax, 20)[:-1]
        dndz = []
        for zed in zfine:
            dndz.append(np.sum(self.lf.schechter_lf_dm(Mabs, zed, band)*(10**(-3) * self.lf.schechter_units)*(np.max(Mabs)-np.min(Mabs))/len(Mabs)).value)
        dndz = np.array(dndz)/np.sum(dndz)    
        zeds = np.random.choice(zfine, Nsrc, p=dndz)
        return zeds, zfine

    def get_schechter_m_given_zs(self, zs, Mabs):
        pdfs = []
        for z in zs:
            pdf = self.lf.schechter_lf_dm(Mabs, z, self.band)
            pdf /= np.sum(pdf)
            pdfs.append(pdf)
        return pdfs

    def generate_positions(self, Nsrc, size, nmaps, all_counts_array, ell_min=90., random_positions=False, hsc=False, cl=None, ells=None):
        
        if random_positions:
            txs = np.random.uniform(0, size, (nmaps, Nsrc))
            tys = np.random.uniform(0, size, (nmaps, Nsrc))

        elif hsc:
            txs, tys = hsc_positions(hsc_cat, np.sum(number_counts[i])*4, z, midz_dz)

        else:
            txs, tys = [], []
            # draw galaxy positions from GRF with given power spectrum, number_counts in deg^-2
            counts, grfs = generate_count_map_2d(nmaps, cl=cl, size=size, ell_min=ell_min, Ntot=Nsrc, ell_sampled=ells)

            self.total_counts += counts

            for i in range(counts.shape[0]):
                tx, ty = positions_from_counts(counts[i])
                #shuffle is used because x positions are ordered and then apparent magnitudes are assigned starting brightest first a few lines down

                randomize = np.arange(len(tx))
                np.random.shuffle(randomize)
                tx = tx[randomize]
                ty = ty[randomize]
                
                txs.append(tx)
                tys.append(ty)
        
        return txs, tys

    def generate_galaxy_catalogs(self, ng_bins=5, zmin=0.01, zmax=5.0, ndeg=4.0, m_min=13, m_max=28, hsc=False, \
                               ell_min=90., Mabs_min=-30.0, Mabs_max=-15., size=1024, random_positions=False, \
                                Mabs_nbin=100, band='H', cl=None, ells=None, n_catalogs=1):
        
        self.total_counts = 0
        n_deg_across = 180./ell_min
        n_square_deg = n_deg_across**2
        
        if hsc:
            hsc_cat = fits_to_dataframe('../data/catalogs/hsc/hsc_pdr1_deep_forced_swire_photoz_cleaned_w_xy.fits')

        zrange_grf = np.linspace(zmin, zmax, ng_bins+1) # ng_bins+1 since we take midpoints
        midzs = 0.5*(zrange_grf[:-1]+zrange_grf[1:])
        dzs = zrange_grf[1:]-zrange_grf[:-1] 

        print('midzs:', midzs)
        print('dzs:', dzs)

        Mapps = np.linspace(m_min, m_max, m_max-m_min + 1)
        Mabs = np.linspace(Mabs_min, Mabs_max, Mabs_nbin)
        
        number_counts = np.array(self.lf.number_counts(midzs, dzs, Mapps, band)[1]).astype(np.int)

        thetax, thetay, gal_app_mags, gal_abs_mags, gal_zs, all_finezs, all_counts_array = [[] for x in xrange(7)]
        
        thetax_list = [[] for x in range(n_catalogs)]
        thetay_list = [[] for x in range(n_catalogs)]
        gal_zs_list = [[] for x in range(n_catalogs)]
        mags_list = [[] for x in range(n_catalogs)]
        gal_app_mag_list = [[] for x in range(n_catalogs)]

        for i, z in enumerate(midzs):
            
            if cl is None:
                ells, cl = limber_project(self.mass_function, zrange_grf[i], zrange_grf[i+1], ell_min=30, ell_max=3e5)

            txs, tys = self.generate_positions(np.sum(number_counts[i])*n_square_deg, size, n_catalogs, \
                                             all_counts_array, ell_min=ell_min, random_positions=random_positions, hsc=hsc, \
                                            cl=cl, ells=ells)
            
            for cat in range(n_catalogs):
                print(len(txs[cat]), len(thetax_list[cat]), type(txs[cat]))
                thetax_list[cat].extend(txs[cat])
                thetay_list[cat].extend(tys[cat])
                
                # draw redshifts within each bin from pdf
                zeds, zfine = self.draw_redshifts(len(txs[cat]), zrange_grf[i], zrange_grf[i+1], Mabs, band=band)
                gal_zs_list[cat].extend(zeds)

            all_finezs.extend(zfine)
            mags = []

            # draw apparent magnitudes based on Helgason number counts N(m)
            for j, M in enumerate(Mapps):

                finer_mapp = np.linspace(M, M+1.0, 100)
                finer_number_counts = self.lf.number_counts([z], [dzs[i]], finer_mapp, band)[1][0]# band doesn't really affect number counts for CIBER
                fine_mapp_pdf = finer_number_counts/np.sum(finer_number_counts)
                n = int(number_counts[i][j]*n_deg_across**2)
                
                if n > 0:
                    for cat in range(n_catalogs):
                        mags_list[cat].extend(np.random.choice(finer_mapp, size=n, p=fine_mapp_pdf))
            
            for cat in range(n_catalogs):
                gal_app_mag_list[cat].extend(mags_list[cat])
                print('len of gal app mag list is ', len(gal_app_mag_list[cat]))
                print('len of thetax list is ', len(thetax_list[cat]))
  
        # gal_zs, thetax, thetay, gal_app_mags, gal_abs_mags, all_finezs = make_lists_arrays([gal_zs, thetax, thetay, gal_app_mags, gal_abs_mags, all_finezs])
        print('len app mags vs thetax:', len(gal_app_mag_list[0]), len(thetax_list[cat]))
        
        cosmo_dist_mods = cosmo.distmod(all_finezs)
        distmod_spline = interpolate.InterpolatedUnivariateSpline(all_finezs, cosmo_dist_mods)

        # poisson realization of galaxy counts not exactly equal to Helgason number counts, so remove extra sources
        array_list = []
        self.catalogs = []
        for cat in range(n_catalogs):
            if len(gal_app_mag_list[cat]) > len(thetax_list[cat]):
                idx_choice = np.sort(np.random.choice(np.arange(len(gal_app_mag_list[cat])), len(thetax_list[cat]), replace=False))
                gal_app_mag_list[cat] = np.array(gal_app_mag_list[cat])[idx_choice]
            dist_mods = distmod_spline(gal_zs_list[cat])
            print('distmods has length:', len(dist_mods))
            print(len(gal_app_mag_list[cat]), len(gal_zs_list[cat]))

            gal_abs_mags = np.array(gal_app_mag_list[cat]) - dist_mods - 2.5*np.log10(1.0+np.array(gal_zs_list[cat]))
            
            array = np.array([thetax_list[cat], thetay_list[cat], gal_zs_list[cat], gal_app_mag_list[cat], gal_abs_mags]).transpose()
            partial_cat = array[np.argsort(array[:,4])]
            halo_masses, virial_radii = self.abundance_match_ms_given_mags(partial_cat[:,2])
            self.catalogs.append(np.hstack([partial_cat, np.array([halo_masses, virial_radii]).transpose()]))
            
        if hsc:
            return self.catalogs
        else:
#             return self.catalog, total_counts, all_counts_array
            return self.catalogs

    def load_halo_mass_function(self, filename):
        hmf = np.loadtxt(filename, skiprows=12)
        ms = hmf[:,0] # [M_sun/h]
        dndm = hmf[:,5]/np.sum(hmf[:,5]) # [h^4/(Mpc^3*M_sun)]
        return ms, dndm




            