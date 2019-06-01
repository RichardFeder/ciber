import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import time
import numpy as np
import seaborn as sns
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy import constants as const
sns.set()

# initialize cosmology class 
cosmo = FlatLambdaCDM(H0=70, Om0=0.28)

def absolute_mag_from_apparent(Mapp, z):
    Mabs = Mapp - cosmo.distmod(z).value-2.5*np.log10(1+zed)
    return Mabs

def apparent_mag_from_absolute(Mabs, z):
    Mapp = Mabs + cosmo.distmod(z).value-2.5*np.log(1+z)
    return Mapp


class Luminosity_Function():
    
    J_dict = dict({'lambda':1.27, 'zmax':3.2, 'm0':-23.04, 'q':0.4, 'phi0':2.21, 'p':0.6, 'alpha0':-1.00, 'r':0.035})
    H_dict = dict({'lambda':1.63, 'zmax':3.2, 'm0':-23.41, 'q':0.5, 'phi0':1.91, 'p':0.8, 'alpha0':-1.00, 'r':0.035})
    K_dict = dict({'lambda':2.20, 'zmax':3.8, 'm0':-22.97, 'q':0.4, 'phi0':2.74, 'p':0.8, 'alpha0':-1.00, 'r':0.035})
    L_dict = dict({'lambda':3.60, 'zmax':0.7, 'm0':-22.40, 'q':0.2, 'phi0':3.29, 'p':0.8, 'alpha0':-1.00, 'r':0.035})
    
    
    z0alpha = 0.01 # used for alpha(z)
    alpha0 = -1.00
    r = 0.035
    DH = 4000*u.Mpc # hubble distance in Mpc
    schecter_units = u.Mpc**(-3)
    cosmo = FlatLambdaCDM(H0=70, Om0=0.28)

    
    omega_m = 0.28
    
    def __init__(self, z0m = 0.8):
        self.ps = dict({'J':self.J_dict, 'H':self.H_dict, 'K':self.K_dict, 'L':self.L_dict}) # sets parameter dictionary of dictionaries
        self.z0m = z0m
        
    def comoving_vol_element(self, z): # checked
        V = cosmo.differential_comoving_volume(z)*u.steradian*(np.pi/180)**2 # to get in deg^-2
        return V
    
    def get_abs_from_app(self, Mapp, z): # checked
        Mabs = Mapp - self.cosmo.distmod(z).value + 2.5*np.log(1+z)
        return Mabs
        
    def get_app_from_abs(self, Mabs, z): # checked
        Mapp = Mabs + self.cosmo.distmod(z).value - 2.5*np.log(1+z)
        return Mapp
    
    def alpha(self, z): # checked
        return self.alpha0*(z/self.z0alpha)**self.r
    
    def phi_star(self, z, band): # checked, has units of 10^-3 Mpc^-3
        phi0 = self.ps[band]['phi0']
        p = self.ps[band]['p']
        return phi0*np.exp(-p*(z-self.z0m))
    
    def m_star(self, z, band): # checked
        m0 = self.ps[band]['m0']
        q = self.ps[band]['q']
        
        return m0-2.5*np.log((1+(z-self.z0m))**q)

    def schecter_lf_dm(self, M, z, band): # checked
        
        msz = self.m_star(z, band)
        phiz = self.phi_star(z, band)
        alph = self.alpha(z)
        
        phi = 0.4*np.log(10)*phiz*(10**(0.4*(msz-M)))**(alph+1)
        phi *= np.exp(-10**(0.4*(msz-M)))
        
        return phi 
    
    def find_nearest_band(self, lam): # chceked
        optkey = ''
        mindist = None
        for key, value in self.ps.iteritems():
            if mindist is None:
                mindist = np.abs(lam-value['lambda'])
                optkey = key
            elif np.abs(lam-value['lambda']) < mindist:
                mindist = np.abs(lam-value['lambda'])
                optkey = key
        return optkey, mindist
        
    
    def number_counts(self, zs, Mapp, band):
        nm = []
        for z in zs:
            # get redshifted wavelength that will fall in observing band
            redshifted_lambda = self.ps[band]['lambda']*(1+z)
            nearest_band, dist = self.find_nearest_band(redshifted_lambda)
            Mabs = self.get_abs_from_app(Mapp, z)
            schec = self.schecter_lf_dm(Mabs, z, nearest_band)*(10**(-3) * self.schecter_units)*self.comoving_vol_element(z)*(zs[1]-zs[0])
            nm.append(schec)
        return np.sum(nm), nm
    
    def specific_flux(self, m_ab, nu):
        val = nu*10**(-0.4*(m_ab-23.9))*u.microJansky
        return val
        
    
    def total_bkg_light(self, ms, z, band):
        dfdz = 0.
        
        nu = const.c/(self.ps[band]['lambda']*u.um)
        
        for i in xrange(len(ms)-1):
            mabs = self.get_abs_from_app(ms[i], z)
            val = (ms[i+1]-ms[i])*self.specific_flux(ms[i], nu)*self.schecter_lf_dm(mabs, z, band)
            val *= (10**(-3) * self.schecter_units)*cosmo.differential_comoving_volume(z)
            dfdz += val

        return dfdz.to(u.nW*u.m**(-2)*u.steradian**(-1))
    
    def flux_prod_rate(self, zrange, ms, band):
        flux_prod_rates = []
        for z in zrange:
            flux_prod_rates.append(self.total_bkg_light(ms, z, band).value)
        return flux_prod_rates



def pdf_cdf_dndz(zmin=0.1, zmax=5, mapp_min=15, mapp_max=30, nbins=50, band='J'):
    zs = np.linspace(zmin, zmax, nbins)
    Mapp = np.linspace(mapp_min, mapp_max, nbins)
    dndz = []
    for zed in zs:
        val = np.sum(lf.schecter_lf_dm(Mapp, zed, band)*(10**(-3) * lf.schecter_units)*(np.max(Mapp)-np.min(Mapp))/len(Mapp))
        dndz.append(val.value)
    dndz = np.array(dndz)/np.sum(dndz)
    cdf_dndz = np.cumsum(dndz)
    
    return dndz, cdf_dndz, Mapp, zs


def draw_gal_redshifts(ngal, limiting_mag=20):
    ''' returns redshifts sorted '''
    dndz, cdf_dndz, mapp, zs = pdf_cdf_dndz()
    idx = np.argmin(np.abs(mapp-limiting_mag))    
    ratio = cdf_dndz[idx]
    ndraw_total = int(ngal/ratio)
    print 'Drawing '+str(ndraw_total)+' galaxies from dN/dz..'
    gal_zs = np.random.choice(zs, ndraw_total, p=dndz)
    ngal_per_z = np.array([len([zg for zg in gal_zs if zg==z]) for z in zs])
    return np.sort(gal_zs), ngal_per_z
    
def get_schecter_m_given_zs(zs, Mapp, band='J'):
    pdfs = []
    for z in zs:
        pdf = lf.schecter_lf_dm(Mapp, z, band)
        pdf /= np.sum(pdf)
        pdfs.append(pdf)
    return pdfs


def absolute_mag_from_apparent(Mapp, z):
    Mabs = Mapp - cosmo.distmod(z).value-2.5*np.log10(1+zed)
    return Mabs

def apparent_mag_from_absolute(Mabs, z):
    Mapp = Mabs + cosmo.distmod(z).value-2.5*np.log(1+z)
    return Mapp


def draw_mags_given_zs(Mapp, ngal_per_z, pdfs, zs):
    gal_app_mags = np.array([])
    gal_abs_mags = np.array([])
    for i in xrange(len(ngal_per_z)):
        apparent_mags = np.random.choice(Mapp, ngal_per_z[i], p=pdfs[i])
        absolute_mags = absolute_mag_from_apparent(apparent_mags, zs[i])
        gal_app_mags = np.append(gal_app_mags, apparent_mags)
        gal_abs_mags = np.append(gal_abs_mags, absolute_mags)
    
    return gal_app_mags, gal_abs_mags
            
    
def generate_galaxy_catalog(ngal):
    lf = Luminosity_Function()
    gal_zs, ng_perz = draw_gal_redshifts(ngal)
    dndz, cdf_dndz, mapp, zs = pdf_cdf_dndz()
    pdfs = get_schecter_m_given_zs(zs, mapp)
    gal_mapp, gal_mabs = draw_mags_given_zs(mapp, ng_perz, pdfs, zs)
    gal_cat = np.array([gal_zs, gal_mapp, gal_mabs]).transpose()
    
    return gal_cat
            

