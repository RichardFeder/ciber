import numpy as np
from compos import matterps

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
# from astropy import constants as const
from scipy import stats
import time
from halo_model import *
from helgason import *
import camb
from camb import model, initialpower


def ell_to_k(ell, comoving_dist):
    theta = np.pi/ell
    k = 1./(comoving_dist*theta)
    return k

def fftIndgen(n):
    a = range(0, n/2+1)
    b = range(1, n/2)
    b.reverse()
    b = [-i for i in b]
    return a + b

def find_nearest_2_idxs(array, vals):
    idxs = np.array([np.abs(array-val).argmin() for val in vals])
    dk = vals - array[idxs]
    idxs_2 = idxs + np.sign(dk)
    idxs_2[idxs_2<0]=0
    return idxs, idxs_2, dk

def k_to_ell(k, comoving_dist):
    theta = 1./(comoving_dist*k)
    ell = np.pi/theta
    return ell

def counts_from_density(density_field, Ntot = 200000):
    counts_map = np.zeros_like(density_field)
    N_mean = Ntot / (counts_map.shape[-2]*counts_map.shape[-1])
    counts_map = np.random.poisson(density_field*N_mean)
    print(counts_map)
    return counts_map

def gaussian_random_field(n_samples, alpha=None, size = 100, ps=None, ksampled=None, comoving_dist=None):

    grfs = np.zeros((n_samples, size, size))
    noise = np.fft.fft2(np.random.normal(size = (n_samples, size, size)))
    amplitude = np.zeros((size, size))
    for i, sx in enumerate(fftIndgen(size)):
        amplitude[i,:] = Pk2(sx, np.array(fftIndgen(size)), ps=ps, ksampled=ksampled, comoving_dist=comoving_dist)
    grfs = np.fft.ifft2(noise * amplitude, axes=(-2,-1))

    return grfs.real, np.array(noise*amplitude)
    
def generate_count_map(n_samples, size=1024, Ntot=2000000, ps=None, ksampled=None, comoving_dist=None):

    realgrf, _ = gaussian_random_field(n_samples, size=size, ps=ps, ksampled=ksampled, comoving_dist=comoving_dist)
    realgrf /= np.max(np.abs(realgrf))
    density_fields = np.exp(realgrf-np.std(realgrf)**2) # assuming a log-normal density distribution
    counts = counts_from_density(density_fields, Ntot=Ntot)

    return counts, density_fields



def Pk2(sx, sy, alph=None, ps=None, ksampled=None, comoving_dist=None, pixsize=3.39e-5):
    if alph is not None:
        return np.sqrt(np.sqrt(sx**2+sy**2)**alph)

    elif ps is not None:

        thetas = np.sqrt(sx**2+sy**2)*pixsize
        k = ell_to_k(np.pi/thetas, comoving_dist)
        idx1, idx2, dk = find_nearest_2_idxs(ksampled, k)
        k1 = ksampled[idx1]
        # k2 = ksampled[idx2]
        ps1 = ps[idx1]
        ps1[k1==0]=0
        # ps2 = ps(np.array([k2]))
        # ps_firstdiff = ps1 + np.abs(dk)*(ps2-ps1)/(k2-k1)
        # return ps_firstdiff
        return ps1

    return power_spec(np.array([np.sqrt(sx**2+sy**2)]))

def power_spec(k):
    ps = matterps.normalizedmp(k)
    ps[k==0] = 0.
    return ps

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

    return thetax, thetay


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

    ell_min = 90.
    ell_max = 1e5

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(ns=0.965)

        
    def __init__(self, band='J'):
        self.band = band

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
        virial_radii = self.halomod.mass_2_virial_radius(halo_masses, zs).to(u.Mpc) # in Mpc
        
        return halo_masses, virial_radii.value

    def draw_gal_redshifts(self, ngal, limiting_mag=20):
        ''' returns redshifts sorted '''
        dndz, cdf_dndz, Mabs, zs = self.pdf_cdf_dndz()
        ndraw_total = int(ngal)
        
        gal_zs = np.random.choice(zs, ndraw_total, p=dndz)
        ngal_per_z = np.array([len([zg for zg in gal_zs if zg==z]) for z in zs])
        return np.sort(gal_zs), ngal_per_z


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

    def get_schechter_m_given_zs(self, zs, Mabs):
        pdfs = []
        for z in zs:
            pdf = self.lf.schechter_lf_dm(Mabs, z, self.band)
            pdf /= np.sum(pdf)
            pdfs.append(pdf)
        return pdfs

    def generate_galaxy_catalog(self, ngal, limiting_mag=22, ng_bins=5):
        self.catalog = []


        gal_zs, ng_perz = self.draw_gal_redshifts(ngal, limiting_mag=limiting_mag)
        dndz, cdf_dndz, mabs, zs = self.pdf_cdf_dndz()
        pdfs = self.get_schechter_m_given_zs(zs, mabs)
        zmm = self.draw_mags_given_zs(mabs, gal_zs, ng_perz, pdfs, zs)

        # generate clustered galaxy positions
        zrange_grf = np.linspace(self.zmin, self.zmax, ng_bins)
        midzs = 0.5*(zrange_grf[:-1]+zrange_grf[1:])
        # eventually should take this stuff out and make external since it takes a little while
        # only thing that really needs to be specified is redshifts to evaluate GRF
        self.pars.set_matter_power(redshifts=midzs, kmax=5.0)
        self.pars.NonLinear = model.NonLinear_both
        results = camb.get_results(self.pars)
        kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=3e-3, maxkh=10, npoints=200)
        comoving_dists = results.comoving_radial_distance(midzs)/(0.01*self.pars.H0)

        thetax = np.zeros_like(gal_zs)
        thetay = np.zeros_like(gal_zs)
        for i, z in enumerate(midzs):
            counts, grfs = generate_count_map(1, ps=pk_nonlin[i,:], ksampled=kh_nonlin, comoving_dist=comoving_dists[i])
            tx, ty = positions_from_counts(counts[0], cat_len=len(gal_zs))
            thetax[gal_zs < zrange_grf[1+i]] = tx
            thetay[gal_zs < zrange_grf[1+i]] = ty
        
        # counts, grfs = generate_count_map(1, ps=pk_nonlin, ksampled=kh_nonlin, comoving_dist=comoving_dists[i])
        # thetax, thetay = positions_from_counts(counts[0], cat_len=len(gal_zs))


        halo_masses, virial_radii = self.abundance_match_ms_given_mags(gal_zs)        
        
        self.catalog = np.array([thetax, thetay, zmm[:,0], zmm[:,1],zmm[:,2],halo_masses,virial_radii]).transpose()
        
        return self.catalog

    def load_halo_mass_function(self, filename):
        hmf = np.loadtxt(filename, skiprows=12)
        ms = hmf[:,0] # [M_sun/h]
        dndm = hmf[:,5]/np.sum(hmf[:,5]) # [h^4/(Mpc^3*M_sun)]
        return ms, dndm

    def pdf_cdf_dndz(self, zmin=0.01, zmax=5, nbins=100, mabs_min=-30., mabs_max=-15., band='J'):
        zs = np.linspace(zmin, zmax, nbins)
        self.zmin = zmin
        self.zmax = zmax
        # Mapp = np.linspace(mapp_min, mapp_max, nbins)
        Mabs = np.linspace(mabs_min, mabs_max, nbins)
        dndz = []
        for zed in zs:
            val = np.sum(self.lf.schechter_lf_dm(Mabs, zed, band)*(10**(-3) * self.lf.schechter_units)*(np.max(Mabs)-np.min(Mabs))/len(Mabs))
            dndz.append(val.value)
        dndz = np.array(dndz)/np.sum(dndz)
        cdf_dndz = np.cumsum(dndz)

        return dndz, cdf_dndz, Mabs, zs


            