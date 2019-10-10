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


def counts_from_density(density_field, Ntot = 200000):
    counts_map = np.zeros_like(density_field)
    N_mean = float(Ntot) / float(counts_map.shape[-3]*counts_map.shape[-2]*counts_map.shape[-1])
    counts_map = np.random.poisson(density_field*N_mean)
    counts_map = np.sum(counts_map, axis=3)
    print(counts_map.shape)
    return counts_map

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

def gaussian_random_field(n_samples, alpha=None, size = 100, ps=None, ksampled=None, comoving_dist=None):

    grfs = np.zeros((n_samples, size, size))
    noise = np.fft.fft2(np.random.normal(size = (n_samples, size, size)))
    amplitude = np.zeros((size, size))
    for i, sx in enumerate(fftIndgen(size)):
        amplitude[i,:] = Pk2(sx, np.array(fftIndgen(size)), ps=ps, ksampled=ksampled, comoving_dist=comoving_dist)
    grfs = np.fft.ifft2(noise * amplitude, axes=(-2,-1))

    return grfs.real, np.array(noise*amplitude)

def gaussian_random_field_3d(n_samples, alpha=None, size=100, ps=None, ksampled=None, comoving_dist=None):
    
    k_ind = np.mgrid[:size, :size, :size] - int( (size + 1)/2 )
    k_ind = scipy.fftpack.fftshift(k_ind)
    k_idx = ( k_ind )
    ks = np.sqrt(k_idx[0]**2 + k_idx[1]**2 + k_idx[2]**2 + 1e-10)
    logfit = np.poly1d(np.polyfit(np.log10(ksampled), np.log10(ps), 30))
    amplitude = 10**(logfit(np.log10(ks)))
    print(amplitude)
    amplitude[0,0,0] = 0.
    noise = np.random.normal(size = (n_samples, size, size, size)) + 1j * np.random.normal(size = (n_samples, size, size, size))
    gfield = np.array([np.fft.ifftn(n * amplitude).real for n in noise])

    return gfield


def generate_count_map(n_samples, size=1024, Ntot=2000000, ps=None, ksampled=None, comoving_dist=None):

    realgrf, _ = gaussian_random_field_3d(n_samples, size=size, ps=ps, ksampled=ksampled, comoving_dist=comoving_dist)
    # realgrf /= np.max(np.abs(realgrf))
    density_fields = np.exp(realgrf-np.std(realgrf)**2) # assuming a log-normal density distribution
    counts = counts_from_density(density_fields, Ntot=Ntot)

    return counts, density_fields

def k_to_ell(k, comoving_dist):
    theta = 1./(comoving_dist*k)
    ell = np.pi/theta
    return ell

def Pk2(sx, sy, alph=None, ps=None, ksampled=None, comoving_dist=None, pixsize=3.39e-5):
    if alph is not None:
        return np.sqrt(np.sqrt(sx**2+sy**2)**alph)

    if ps is not None:
        ell_sampled = k_to_ell(ksampled, comoving_dist)
        ells = np.pi/(np.sqrt(sx**2+sy**2)*pixsize)
        idx1 = np.array([np.abs(ell_sampled-ell).argmin() for ell in ells])
        ps1 = ps[idx1]
        return ps1
    return True

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

    return np.array(thetax), np.array(thetay)


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
    
    lf = Luminosity_Function()
    halomod = halo_model_class()
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(ns=0.965)

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

    def get_schechter_m_given_zs(self, zs, Mabs):
        pdfs = []
        for z in zs:
            pdf = self.lf.schechter_lf_dm(Mabs, z, self.band)
            pdf /= np.sum(pdf)
            pdfs.append(pdf)
        return pdfs

    def matter_power_spectrum(self, zs, minkh=3e-3, maxkh=5.0, npoints=200):
        self.pars.set_matter_power(redshifts=zs, kmax=maxkh)
        self.pars.NonLinear = model.NonLinear_both
        results = camb.get_results(self.pars)
        kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=minkh, maxkh=maxkh, npoints=npoints)
        comoving_dists = results.comoving_radial_distance(zs)/(0.01*self.pars.H0)

        return kh_nonlin, z_nonlin, pk_nonlin, comoving_dists


    def generate_galaxy_catalog(self, ng_bins=5, zmin=0.01, zmax=5.0, ndeg=4.0, m_min=13, m_max=28):
                
        zrange_grf = np.linspace(zmin, zmax, ng_bins+1) # ng_bins+1 since we take midpoints
        midzs = 0.5*(zrange_grf[:-1]+zrange_grf[1:])
        print('midzs:', midzs)

        kh_nonlin, z_nonlin, pk_nonlin, comoving_dists = self.matter_power_spectrum(midzs)
        Mapps = np.linspace(m_min, m_max, m_max-m_min + 1)
        Mabs = np.linspace(-30.0, -15., 100)
        number_counts = self.lf.number_counts(midzs, Mapps, 'H')[1] # band doesn't really affect number counts for CIBER
        thetax, thetay, gal_app_mags, gal_abs_mags, gal_zs, all_finezs = [[] for x in xrange(6)]

        for i, z in enumerate(midzs):
    
            # draw galaxy positions from GRF with given power spectrum
            counts, grfs = generate_count_map(1, ps=pk_nonlin[i,:], Ntot=int(np.sum(number_counts[i])*4), ksampled=kh_nonlin, comoving_dist=comoving_dists[i])
            tx, ty = positions_from_counts(counts[0])

            # shuffle is used because x positions are ordered and then apparent magnitudes are assigned starting brightest first a few lines down
            randomize = np.arange(len(tx))
            np.random.shuffle(randomize)
            tx = tx[randomize]
            ty = ty[randomize]
            thetax.extend(tx)
            thetay.extend(ty)

             # draw redshifts within each bin from pdf
            zfine = np.linspace(zrange_grf[i], zrange_grf[i+1], 20)[:-1]
            all_finezs.extend(zfine)
            dndz = []
            for zed in zfine:
                dndz.append(np.sum(self.lf.schechter_lf_dm(Mabs, zed, 'H')*(10**(-3) * self.lf.schechter_units)*(np.max(Mabs)-np.min(Mabs))/len(Mabs)).value)
            dndz = np.array(dndz)/np.sum(dndz)    
            zeds = np.random.choice(zfine, len(tx), p=dndz)
            gal_zs.extend(zeds)

            # draw apparent magnitudes based on Helgason number counts N(m)
            mags = []

            if i == len(midzs)-1:
                dz = midzs[i]-midzs[i-1]
            else:
                dz = midzs[i+1]-midzs[i]

            for j, M in enumerate(Mapps):

                finer_mapp = np.linspace(M, M+1.0, 100)
                finer_number_counts = self.lf.number_counts([z], finer_mapp, 'H', dz=dz)[1][0]# band doesn't really affect number counts for CIBER
                fine_mapp_pdf = finer_number_counts/np.sum(finer_number_counts)

                n = int(number_counts[i][j]*4)
                if n > 0:
                    mags.extend(np.random.choice(finer_mapp, size=n, p=fine_mapp_pdf))
            
            gal_app_mags.extend(mags)


        gal_zs = np.array(gal_zs)
        thetax = np.array(thetax)
        thetay = np.array(thetay)
        gal_app_mags = np.array(gal_app_mags)
        gal_abs_mags = np.array(gal_abs_mags)

        # poisson realization of galaxy counts not exactly equal to Helgason number counts, so remove extra sources
        # miiight need a corner case if counts realization gets more sources than number counts, but this probably wont happen
        idx_choice = np.sort(np.random.choice(np.arange(len(gal_app_mags)), len(thetax), replace=False))
        gal_app_mags = np.array(gal_app_mags)[idx_choice]

        all_finezs = np.array(all_finezs)
        cosmo_dist_mods = cosmo.distmod(np.array(all_finezs))
       
        gal_tile = np.tile(gal_zs, (len(all_finezs), 1)).transpose()
        dist_mods = np.array(cosmo_dist_mods[np.argmin(np.abs(np.subtract(gal_tile, all_finezs)), axis=1)].value)
        gal_abs_mags = gal_app_mags - dist_mods - 2.5*np.log10(1.0+gal_zs)

        array = np.array([thetax, thetay, gal_zs, gal_app_mags, gal_abs_mags]).transpose()
        partial_cat = array[np.argsort(array[:,4])]

        halo_masses, virial_radii = self.abundance_match_ms_given_mags(partial_cat[:,2]) # use redshifts as input        
        self.catalog = np.hstack([partial_cat, np.array([halo_masses, virial_radii]).transpose()])

        return self.catalog

    def load_halo_mass_function(self, filename):
        hmf = np.loadtxt(filename, skiprows=12)
        ms = hmf[:,0] # [M_sun/h]
        dndm = hmf[:,5]/np.sum(hmf[:,5]) # [h^4/(Mpc^3*M_sun)]
        return ms, dndm



            