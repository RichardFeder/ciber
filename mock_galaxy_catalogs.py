import numpy as np
import astropy.units as u
from scipy import stats
from helgason import *
from lognormal_counts import *
from catalog_utils import *

import sys
import camb

if sys.version_info[0] == 3:
    from halo_model import *
    from hmf import MassFunction
    import hmf

pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(ns=0.965)

def k_to_ell(k, comoving_dist):
    ''' Convert wavenumber to multipole for fixed redshift '''
    theta = 1./(comoving_dist*k)
    ell = np.pi/theta
    return ell

def make_lists_arrays(list_of_lists):
    ''' this is just a convenience function for when I want to convert a bunch of lists to numpy arrays '''

    list_of_arrays = []
    for l in list_of_lists:
        list_of_arrays.append(np.array(l))
    return list_of_arrays


def positions_from_counts(counts_map, cat_len=None):

    ''' Given a counts map, generate source catalog positions consistent with those counts. 
    Not doing any subpixel position assignment or anything like that.''' 

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
    ''' Simple power law function for angular correlation function'''
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
    # mass_function = MassFunction(z=0., dlog10m=0.02)
    cosmo = FlatLambdaCDM(H0=70, Om0=0.28)


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

        virial_radii = self.mass_2_virial_radius(halo_masses) # in Mpc

        return halo_masses, virial_radii  

    def abundance_match_ms_given_mags(self, zs, Mmin=10.0, library=True):
        ''' here we want to sort these in descending order, since abs mags are ordered and most neg absolute
        magnitude corresponds to most massive halo'''
        ngal = len(zs)

        if library:
            hmasses, mf = hmf.sample.sample_mf(ngal, 10.0)
            halo_masses = np.sort(hmasses)[::-1]
        else:
            mass_range, dndm = self.load_halo_mass_function('../data/halo_mass_function_hmfcalc.txt')
            halo_masses = np.sort(np.random.choice(mass_range, ngal,p=dndm))[::-1]
        halo_masses *= u.solMass
        virial_radii = self.mass_2_virial_radius(halo_masses, zs).to(u.Mpc) # in Mpc
        
        return halo_masses, virial_radii.value

    def draw_mags_given_zs(self, Mabs, gal_zs, ngal_per_z, pdfs, zs):
        '''we are going to order these by absolute magnitude, which makes things easier when abundance matching to 
        halo mass'''
        gal_app_mags = np.array([])
        gal_abs_mags = np.array([])
        for i in range(len(ngal_per_z)):
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
        ''' Given some redshift range and observing band, this function uses the luminosity function from Helgason to sample Nsrc redshifts.'''
        zfine = np.linspace(zmin, zmax, 20)[:-1]
        dndz = []
        for zed in zfine:
            dndz.append(np.sum(self.lf.schechter_lf_dm(Mabs, zed, band)*(10**(-3) * self.lf.schechter_units)*(np.max(Mabs)-np.min(Mabs))/len(Mabs)).value)
        dndz = np.array(dndz)/np.sum(dndz)    
        zeds = np.random.choice(zfine, Nsrc, p=dndz)
        return zeds, zfine

    def get_schechter_m_given_zs(self, zs, Mabs):
        ''' this function computes collection of apparent magnitude PDFs evaluated at different redshifts as determined by the Schechter luminosity function 
        from Helgason'''
        pdfs = []
        for z in zs:
            pdf = self.lf.schechter_lf_dm(Mabs, z, self.band)
            pdf /= np.sum(pdf)
            pdfs.append(pdf)
        return pdfs

    def generate_positions(self, Nsrc, size, nmaps, ell_min=90., random_positions=False, hsc=False, cl=None, ells=None):
        
        ''' If random_positions is True, this function just draws random source positions, but otherwise a clustering realization is generated and
        positions are sampled from this field. Can also take in a preselected HSC catalog'''

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
                                Mabs_nbin=100, band='H', cl=None, ells=None, n_catalogs=1, n_bin_Mapp=200):
        
        ''' This function puts together other functions in the galaxy_catalog() class as full pipeline to generate galaxy catalog realizations,
        given some angular power spectrum and Helgason model'''

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

        Mapps = np.linspace(m_min, m_max, int(m_max-m_min) + 1)

        Mabs = np.linspace(Mabs_min, Mabs_max, Mabs_nbin)

        # First, I need to figure out how many sources are within a given redshift bin
        Mapps = np.linspace(m_min, m_max, n_bin_Mapp)
        dMapp = Mapps[1]-Mapps[0]
        number_counts = np.array(self.lf.number_counts(zrange_grf, Mapps, band, dzs=dzs)[1]).astype(np.int)
        print('number counts has shape ', number_counts.shape, 'while Mapps has nbins=', len(Mapps))
        # number_counts = np.array(self.lf.number_counts(midzs, dzs, Mapps, band)[1]).astype(np.int)
        # this should be a 2d array with len(Mapps) rows and len(midzs) columns

        thetax, thetay, gal_app_mags, gal_abs_mags, gal_zs, all_finezs, all_counts_array = [[] for x in range(7)]
        
        thetax_list = [[] for x in range(n_catalogs)]
        thetay_list = [[] for x in range(n_catalogs)]
        gal_zs_list = [[] for x in range(n_catalogs)]
        mags_list = [[] for x in range(n_catalogs)]
        gal_app_mag_list = [[] for x in range(n_catalogs)]

        for i, z in enumerate(midzs):
            
            if cl is None:
                ells, cl = limber_project(self.mass_function, zrange_grf[i], zrange_grf[i+1], ell_min=30, ell_max=3e5)
            
            print('number counts here are:', np.sum(number_counts[i])*n_square_deg, n_square_deg)
            txs, tys = self.generate_positions(np.sum(number_counts[i])*n_square_deg, size, n_catalogs, \
                                              ell_min=ell_min, random_positions=random_positions, hsc=hsc, \
                                            cl=cl, ells=ells)
            
            for cat in range(n_catalogs):
                thetax_list[cat].extend(txs[cat])
                thetay_list[cat].extend(tys[cat])

                # draw redshifts within each bin from pdf
                zeds, zfine = self.draw_redshifts(len(txs[cat]), zrange_grf[i], zrange_grf[i+1], Mabs, band=band)
                gal_zs_list[cat].extend(zeds)

            all_finezs.extend(zfine)
            mags = []

            # draw apparent magnitudes based on Helgason number counts N(m)
            mapp_pdf = number_counts[i].astype(np.float32)/float(np.sum(number_counts[i]))
            n = int(np.sum(number_counts[i])*n_deg_across**2)
            if n > 0:
                for cat in range(n_catalogs):
                    mag_draw = np.random.choice(Mapps, size=n, p=mapp_pdf)

                    mags_list[cat].extend(mag_draw)
            
        cosmo_dist_mods = cosmo.distmod(all_finezs)
        distmod_spline = interpolate.InterpolatedUnivariateSpline(all_finezs, cosmo_dist_mods)

        # poisson realization of galaxy counts not exactly equal to Helgason number counts, so remove extra sources
        array_list = []
        self.catalogs = []
        for cat in range(n_catalogs):
            if len(mags_list[cat]) > len(thetax_list[cat]):
                idx_choice = np.sort(np.random.choice(np.arange(len(mags_list[cat])), len(thetax_list[cat]), replace=False))
                mags_list[cat] = np.array(mags_list[cat])[idx_choice]
                
            dist_mods = distmod_spline(gal_zs_list[cat])
            print('distmods has length:', len(dist_mods))
            gal_abs_mags = np.array(mags_list[cat]) - dist_mods - 2.5*np.log10(1.0+np.array(gal_zs_list[cat]))
            array = np.array([thetax_list[cat], thetay_list[cat], gal_zs_list[cat], mags_list[cat], gal_abs_mags]).transpose()
            partial_cat = array[np.argsort(array[:,4])]
            halo_masses, virial_radii = self.abundance_match_ms_given_mags(partial_cat[:,2])
            self.catalogs.append(np.hstack([partial_cat, np.array([halo_masses, virial_radii]).transpose()]))
            
        return self.catalogs

    def load_halo_mass_function(self, filename):
        hmf = np.loadtxt(filename, skiprows=12)
        ms = hmf[:,0] # [M_sun/h]
        dndm = hmf[:,5]/np.sum(hmf[:,5]) # [h^4/(Mpc^3*M_sun)]
        return ms, dndm

    def mass_2_virial_radius(self, halo_mass, z=0):
        ''' This assumes the halo is spherically symmetric at least to determine the virial radius '''

        R_vir_cubed = (3/(4*np.pi))*self.cosmo.Om(z)*halo_mass.to(u.g)/(200*self.cosmo.critical_density(z))
        return R_vir_cubed**(1./3.)



def make_galaxy_cts_map(cat, refmap_shape, inst, m_min=14, m_max=30, magidx=2, zmin=0, zmax=100, zidx=None, normalize=True):
    ''' Given a catalog of sources and some cuts on apparent magnitude/redshift, this function calculates the corresponding counts map, which
    can then be used to compute auto or cross power spectra.''' 

    gal_map = np.zeros(shape=refmap_shape)

    if isinstance(cat, pd.DataFrame): # real catalogs read in as pandas dataframes
    
        catalog = cat.loc[(cat['x'+str(inst)]>0)&(cat['x'+str(inst)]<refmap_shape[0])&(cat['y'+str(inst)]>0)&(cat['y'+str(inst)]<refmap_shape[0]) &\
                         (cat[magidx]<m_max)&(cat[magidx]>m_min)&(cat[zidx]>zmin)&(cat[zidx]<zmax)]

        for index, src in catalog.iterrows():
            gal_map[int(src['x'+str(inst)]), int(src['y'+str(inst)])] += 1
   
    else:
        if zidx is not None:
            if magidx is None:
                cat = np.array([src for src in cat if src[0]<refmap_shape[0] and src[1]<refmap_shape[1] and src[zidx]>zmin and src[zidx]<zmax])
            else:
                cat = np.array([src for src in cat if src[0]<refmap_shape[0] and src[1]<refmap_shape[1]\
                and src[magidx]>m_min and src[magidx]<m_max and src[zidx]>zmin and src[zidx]<zmax])
        else:
            if magidx is None:
                cat = np.array([src for src in cat if src[0]<refmap_shape[0] and src[1]<refmap_shape[1]])
            else:
                cat = np.array([src for src in cat if src[0]<refmap_shape[0] and src[1]<refmap_shape[1]\
                and src[magidx]>m_min and src[magidx]<m_max])

        for src in cat:
            gal_map[int(src[0]),int(src[1])] +=1.

    if normalize:
        gal_map /= np.mean(gal_map)
        gal_map -= 1.
    
    return gal_map


    
            
            