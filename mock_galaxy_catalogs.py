import numpy as np
from compos import const, matterps
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy import constants as const
from scipy import stats
from halo_model import *

# ell_min = 90.
# ell_max = ell_min*np.sqrt(2*512**2)

# pars = camb.CAMBparams()
# pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
# pars.InitPower.set_params(ns=0.965)
# kh_nonlin, _, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-3, maxkh=2.0, npoints = 2000)

const.initializecosmo()

def fftIndgen(n):
    a = range(0, n/2+1)
    b = range(1, n/2)
    b.reverse()
    b = [-i for i in b]
    return a + b

def power_spec(k):
    ps = matterps.normalizedmp(k)
    ps[k==0] = 0.
    return ps

def gaussian_random_field(n_samples, alpha=None, size = 100):
    def Pk2(kx, ky, alph=None):
        if alph is not None:
            return np.sqrt(np.sqrt(kx**2+ky**2)**alph)
        return power_spec(np.array([np.sqrt(kx**2+ky**2)]))

    grfs = np.zeros((n_samples, size, size))
    
    noise = np.fft.fft2(np.random.normal(size = (n_samples, size, size)))

    amplitude = np.zeros((size, size))
    for i, kx in enumerate(fftIndgen(size)):
        amplitude[i,:] = Pk2(kx, np.array(fftIndgen(size)))

    grfs = np.fft.ifft2(noise * amplitude, axes=(-2,-1))

    return grfs.real, np.array(noise*amplitude)


def counts_from_density(density_field, Ntot = 20000):
    counts_map = np.zeros_like(density_field)
    N_mean = Ntot / (counts_map.shape[-2]*counts_map.shape[-1])
    counts_map = np.random.poisson(density_field*N_mean)
    return counts_map
    
def generate_galaxy_count_map(n_samples, size=1024, Ntot=2000000)
    realgrf, _ = gaussian_random_field(n_samples, size=size)
    density_fields = np.exp(realgrf) # assuming a log-normal density distribution
    counts = counts_from_density(density_fields, Ntot=Ntot)
    return counts



''' 

I want to construct random catalogs drawn from the galaxy distribution specified by Helgason et al.
This involves:
- drawing redshifts from dN/dz - done
- drawing absolute magnitudes from Phi(M|z) - done
- converting to apparent magnitudes, and then to flux units from AB - done
- computing an estimated mass for the galaxy - done
- get virial radius for IHL component - done
- generate positions consistent with two point correlation function of galaxies (TODO)

'''

class galaxy_catalog():
    
    catalog = []
    lf = Luminosity_Function()
    halomod = halo_model()

    
    def __init__(self, band='J'):
        self.band = band

    def pdf_cdf_dndz(self, zmin=0.01, zmax=5, nbins=100, mabs_min=-30., mabs_max=-15., band='J'):
        zs = np.linspace(zmin, zmax, nbins)
        # Mapp = np.linspace(mapp_min, mapp_max, nbins)
        Mabs = np.linspace(mabs_min, mabs_max, nbins)
        dndz = []
        for zed in zs:
            val = np.sum(self.lf.schechter_lf_dm(Mabs, zed, band)*(10**(-3) * self.lf.schechter_units)*(np.max(Mabs)-np.min(Mabs))/len(Mabs))
            dndz.append(val.value)
        dndz = np.array(dndz)/np.sum(dndz)
        cdf_dndz = np.cumsum(dndz)

        return dndz, cdf_dndz, Mabs, zs


    def draw_gal_redshifts(self, ngal, limiting_mag=20):
        ''' returns redshifts sorted '''
        dndz, cdf_dndz, Mabs, zs = self.pdf_cdf_dndz()
        ndraw_total = int(ngal)
        
        gal_zs = np.random.choice(zs, ndraw_total, p=dndz)
        ngal_per_z = np.array([len([zg for zg in gal_zs if zg==z]) for z in zs])
        return np.sort(gal_zs), ngal_per_z

    def get_schechter_m_given_zs(self, zs, Mabs):
        pdfs = []
        for z in zs:
            pdf = self.lf.schechter_lf_dm(Mabs, z, self.band)
            pdf /= np.sum(pdf)
            pdfs.append(pdf)
        return pdfs

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
                        
        arr = np.array([gal_zs, gal_app_mags, gal_abs_mags]).transpose()
        cat = arr[np.argsort(arr[:,2])] # sort apparent and absolute mags by absolute mags

        return cat

    def abundance_match_ms_given_mags(self, zs):
        ''' here we want to sort these in descending order, since abs mags are ordered and most neg absolute
        magnitude corresponds to most massive halo'''
        ngal = len(zs)
        mass_range, dndm = self.load_halo_mass_function('../data/halo_mass_function_hmfcalc.txt')
        halo_masses = np.sort(np.random.choice(mass_range, ngal,p=dndm))[::-1]
        virial_radii = self.halomod.mass_2_virial_radius(halo_masses, zs).to(u.Mpc) # in Mpc
        
        return halo_masses, virial_radii.value
        
    def load_halo_mass_function(self, filename):
        hmf = np.loadtxt(filename, skiprows=12)
        ms = hmf[:,0] # [M_sun/h]
        dndm = hmf[:,5]/np.sum(hmf[:,5]) # [h^4/(Mpc^3*M_sun)]
        return ms, dndm

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

    def generate_galaxy_catalog(self, ngal, limiting_mag=22):
        self.catalog = []


        gal_zs, ng_perz = self.draw_gal_redshifts(ngal, limiting_mag=limiting_mag)
        dndz, cdf_dndz, mabs, zs = self.pdf_cdf_dndz()
        pdfs = self.get_schechter_m_given_zs(zs, mabs)
        zmm = self.draw_mags_given_zs(mabs, gal_zs, ng_perz, pdfs, zs)

        # here is where we want to generate clustered galaxy positions !! uniform for now
        thetax = np.random.uniform(0, 1024, len(gal_zs))
        thetay = np.random.uniform(0, 1024, len(gal_zs))
        


        halo_masses, virial_radii = self.abundance_match_ms_given_mags(gal_zs)        
        
        self.catalog = np.array([thetax, thetay, zmm[:,0], zmm[:,1],zmm[:,2],halo_masses,virial_radii]).transpose()
        
        return self.catalog


            