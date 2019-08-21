import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy import constants as const

# initialize cosmology class 
cosmo = FlatLambdaCDM(H0=70, Om0=0.28)

def absolute_mag_from_apparent(Mapp, z):
    Mabs = Mapp - cosmo.distmod(z).value-2.5*np.log10(1+z)
    return Mabs

def apparent_mag_from_absolute(Mabs, z):
    Mapp = Mabs + cosmo.distmod(z).value-2.5*np.log10(1+z)
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
    schechter_units = u.Mpc**(-3)
    cosmo = FlatLambdaCDM(H0=70, Om0=0.28)

    
    omega_m = 0.28
    
    def __init__(self, z0m = 0.8):
        self.band_dicts = dict({'J':self.J_dict, 'H':self.H_dict, 'K':self.K_dict, 'L':self.L_dict}) # sets parameter dictionary of dictionaries
        self.z0m = z0m

    def alpha(self, z): # checked
        return self.alpha0*(z/self.z0alpha)**self.r
        
    def comoving_vol_element(self, z): # checked
        V = cosmo.differential_comoving_volume(z)*u.steradian*(np.pi/180)**2 # to get in deg^-2
        return V
    

    def find_nearest_band(self, lam): # chceked
        optkey = ''
        mindist = None
        for key, value in self.band_dicts.iteritems():
            if mindist is None:
                mindist = np.abs(lam-value['lambda'])
                optkey = key
            elif np.abs(lam-value['lambda']) < mindist:
                mindist = np.abs(lam-value['lambda'])
                optkey = key
        return optkey, mindist

    def flux_prod_rate(self, zrange, ms, band):
        flux_prod_rates = []
        for z in zrange:
            flux_prod_rates.append(self.total_bkg_light(ms, z, band).value)
        return flux_prod_rates

    def get_abs_from_app(self, Mapp, z): # checked
        Mabs = Mapp - self.cosmo.distmod(z).value + 2.5*np.log(1+z)
        return Mabs
        
    def get_app_from_abs(self, Mabs, z): # checked
        Mapp = Mabs + self.cosmo.distmod(z).value - 2.5*np.log(1+z)
        return Mapp

    def m_star(self, z, band): # checked
        m0 = self.band_dicts[band]['m0']
        q = self.band_dicts[band]['q']
        
        return m0-2.5*np.log((1+(z-self.z0m))**q)

    ''' This returns the number counts per magnitude per square degree. It does this by, for each redshift,
        1) computing wavelength that would redshift into observing band at z=0
        2) getting absolute magnitude at that redshift
        3) computing Schechter luminosity function in (pre-redshifted) band and converting to counts/mag/deg^2
    '''
    def number_counts(self, zs, Mapp, band, dz=None):
        nm = []
        for z in zs:
            redshifted_lambda = self.band_dicts[band]['lambda']*(1+z)
            nearest_band, dist = self.find_nearest_band(redshifted_lambda)
            Mabs = self.get_abs_from_app(Mapp, z)
            # print(Mabs)
            if dz is None:
                dz = zs[1]-zs[0]
            schec = self.schechter_lf_dm(Mabs, z, nearest_band)*(10**(-3) * self.schechter_units)*self.comoving_vol_element(z)*dz
            nm.append(schec.value)
        return np.sum(nm), nm

    def phi_star(self, z, band): # checked, has units of 10^-3 Mpc^-3
        phi0 = self.band_dicts[band]['phi0']
        p = self.band_dicts[band]['p']
        return phi0*np.exp(-p*(z-self.z0m))
    

    def schechter_lf_dm(self, M, z, band): # in absolute magnitudes
        
        Msz = self.m_star(z, band)
        phiz = self.phi_star(z, band)
        alph = self.alpha(z)
        
        phi = 0.4*np.log(10)*phiz*(10**(0.4*(Msz-M)))**(alph+1)
        phi *= np.exp(-10**(0.4*(Msz-M)))
        
        return phi 
    
    def specific_flux(self, m_ab, nu):
        val = nu*10**(-0.4*(m_ab-23.9))*u.microJansky
        return val
        
    def total_bkg_light(self, ms, z, band):
        dfdz = 0.
        nu = const.c/(self.band_dicts[band]['lambda']*u.um)
        for i in xrange(len(ms)-1):
            mabs = self.get_abs_from_app(ms[i], z)
            val = (ms[i+1]-ms[i])*self.specific_flux(ms[i], nu)*self._lf_dm(mabs, z, band)
            val *= (10**(-3) * self.schechter_units)*cosmo.differential_comoving_volume(z)
            dfdz += val

        return dfdz.to(u.nW*u.m**(-2)*u.steradian**(-1))
    




