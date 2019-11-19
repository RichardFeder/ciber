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
    
    UV_dict = dict({'lambda':0.15,'N_LF':24, 'zmax':8.0, 'm0':-19.62, 'q':1.1, 'phi0':2.43, 'p':0.2, 'alpha0':-1.00, 'r':0.086})
    U_dict = dict({'lambda':0.36,'N_LF':27, 'zmax':4.5, 'm0':-20.20, 'q':1.0, 'phi0':5.46, 'p':0.5, 'alpha0':-1.00, 'r':0.076})
    B_dict = dict({'lambda':0.45,'N_LF':44, 'zmax':4.5, 'm0':-21.35, 'q':0.6, 'phi0':3.41, 'p':0.4, 'alpha0':-1.00, 'r':0.055})
    V_dict = dict({'lambda':0.55,'N_LF':18, 'zmax':3.6, 'm0':-22.13, 'q':0.5, 'phi0':2.42, 'p':0.5, 'alpha0':-1.00, 'r':0.060})
    R_dict = dict({'lambda':0.65,'N_LF':25, 'zmax':3.0, 'm0':-22.40, 'q':0.5, 'phi0':2.25, 'p':0.5, 'alpha0':-1.00, 'r':0.070})
    I_dict = dict({'lambda':0.79,'N_LF':17, 'zmax':3.0, 'm0':-22.80, 'q':0.4, 'phi0':2.05, 'p':0.4, 'alpha0':-1.00, 'r':0.070})
    z_dict = dict({'lambda':0.91,'N_LF':7, 'zmax':2.9, 'm0':-22.86, 'q':0.4, 'phi0':2.55, 'p':0.4, 'alpha0':-1.00, 'r':0.060})
    J_dict = dict({'lambda':1.27, 'N_LF':15, 'zmax':3.2, 'm0':-23.04, 'q':0.4, 'phi0':2.21, 'p':0.6, 'alpha0':-1.00, 'r':0.035})
    H_dict = dict({'lambda':1.63, 'N_LF':6,  'zmax':3.2, 'm0':-23.41, 'q':0.5, 'phi0':1.91, 'p':0.8, 'alpha0':-1.00, 'r':0.035})
    K_dict = dict({'lambda':2.20, 'N_LF':38, 'zmax':3.8, 'm0':-22.97, 'q':0.4, 'phi0':2.74, 'p':0.8, 'alpha0':-1.00, 'r':0.035})
    L_dict = dict({'lambda':3.60, 'N_LF':6,  'zmax':0.7, 'm0':-22.40, 'q':0.2, 'phi0':3.29, 'p':0.8, 'alpha0':-1.00, 'r':0.035})
    
    
    z0alpha = 0.01 # used for alpha(z)
    alpha0 = -1.00
    r = 0.035
    schechter_units = u.Mpc**(-3)
    cosmo = FlatLambdaCDM(H0=70, Om0=0.28)
    
    omega_m = 0.28
    
    def __init__(self, z0m = 0.8):
        self.band_dicts = dict({'UV':self.UV_dict, 'U':self.U_dict, 'B':self.B_dict, 'V':self.V_dict, \
                                'R':self.R_dict, 'I':self.I_dict, 'z':self.z_dict, 'J':self.J_dict, \
                                'H':self.H_dict, 'K':self.K_dict, 'L':self.L_dict}) # sets parameter dictionary of dictionaries
        self.z0m = z0m

    def alpha(self, z): # checked
        return self.alpha0*(z/self.z0alpha)**self.r
        
    def comoving_vol_element(self, z): # checked
        V = cosmo.differential_comoving_volume(z)*u.steradian*(np.pi/180.)**2 # to get in deg^-2
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
        Mabs = Mapp - self.cosmo.distmod(z).value + 2.5*np.log10(1.+z)
        return Mabs
        
    def get_app_from_abs(self, Mabs, z): # checked
        Mapp = Mabs + self.cosmo.distmod(z).value - 2.5*np.log10(1.+z)
        return Mapp

    def m_star(self, z, band): # checked
        m0 = self.band_dicts[band]['m0']
        q = self.band_dicts[band]['q']
        
        return m0-2.5*np.log10((1.+(z-self.z0m))**q)

    ''' This returns the number counts per magnitude per square degree. It does this by, for each redshift,
        1) computing wavelength that would redshift into observing band at z=0
        2) getting absolute magnitude at that redshift
        3) computing Schechter luminosity function in (pre-redshifted) band and converting to counts/mag/deg^2
    '''
    def number_counts(self, zs, dzs, Mapp, band):
        nm = np.zeros(len(zs))
        for i, z in enumerate(zs):
            rest_frame_lambda = self.band_dicts[band]['lambda']/(1.+z)            
            nearest_band, dist = self.find_nearest_band(rest_frame_lambda)
            Mabs = self.get_abs_from_app(Mapp, z)
            schec = self.schechter_lf_dm(Mabs, z, nearest_band)*(10**(-3) * self.schechter_units)*self.comoving_vol_element(z)
            nm[i] = schec.value
            
        trap_list = dzs*np.array([0.5*(nm[i]+nm[i+1]) for i in range(len(zs)-1)])

        return np.sum(trap_list), trap_list

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
        ''' output has units of microJansky '''
        val = nu*10**(-0.4*(m_ab-23.9))*u.microJansky
        return val
        
    def total_bkg_light(self, ms, z, band):
        ''' for a given redshift, this function integrates the luminosity function over apparent
        magnitudes, weighted by the specific flux within each magnitude bin. Resulting quantity has 
        units of intensity, i.e. nW/m^2/sr. '''
        dfdz = 0.
        nu = const.c/(self.band_dicts[band]['lambda']*u.um)
        for i in xrange(len(ms)-1):
            mabs = self.get_abs_from_app(ms[i], z)
            val = (ms[i+1]-ms[i])*self.specific_flux(ms[i], nu)*self.schechter_lf_dm(mabs, z, band)
            val *= (10**(-3) * self.schechter_units)*cosmo.differential_comoving_volume(z)
            dfdz += val

        return dfdz.to(u.nW*u.m**(-2)*u.steradian**(-1))


    # these shot noise levels agree with Fig. 8 of Helgason+ given certain limiting magnitudes

    def auto_shot_noise(self, z0, z1, ms, nzbin=10, band='J'):
        ''' This function computes the shot noise from galaxies as driven by the Schechter luminosity function.
        The shot noise in the auto spectrum is calculated in Helgason with the double integral
        P_{SN} = (integral over redshift) dz (integral over magnitude)dm f^2(m)dN(m|z)/dm. The resulting 
        shot noise has units of intensity squared, i.e. (nW/m^2/sr)^2. '''
        
        nu = const.c/(self.band_dicts[band]['lambda']*u.um)
        shots = []
        zrange = np.linspace(z0, z1, nzbin)
        dz = (z1-z0)/nzbin
        
        for j in xrange(len(zrange)-1):
            
            cl_shot = 0.
            z = 0.5*(zrange[j+1]+zrange[j])
            
            for i in xrange(len(ms)-1):
                m_in_btwn = 0.5*(ms[i+1]+ms[i])
                dm = ms[i+1]-ms[i]
                mabs = self.get_abs_from_app(m_in_btwn, z)
                dndm = self.schechter_lf_dm(mabs, z, band)*(10**(-3) * self.schechter_units)*cosmo.differential_comoving_volume(z)
                f_m = self.specific_flux(m_in_btwn, nu)
                psn = dm*dndm*f_m**2
                cl_shot += psn.to(u.nW**2*u.m**(-4)*u.steradian**(-1))

            shots.append(cl_shot.value*dz) # nW^2 m^-4 sr^-1

        return np.sum(shots)


    def cross_shot_noise(self, z0, z1, ms, nzbin=10, band='J'):
        ''' This computes the shot noise when cross correlating an intensity map with a galaxy counts map.
        This differs from the auto shot noise in two ways:
            1) The integral over number counts is weighted by f(m)n_g^{-1} rather than f^2(m)
            2) The integral over magnitude is evaluated over the tracer catalog magnitude range.
        Cross shot noise has units of intensity, i.e. nW/m^2/sr. 
        '''

        nu = const.c/(self.band_dicts[band]['lambda']*u.um)
        shots = []
        zrange = np.linspace(z0, z1, nzbin)
        dz = (z1-z0)/nzbin
        for j in xrange(len(zrange)-1):
            cl_shot = 0.
            z = 0.5*(zrange[j+1]+zrange[j])

            for i in xrange(len(ms)-1):
                m_in_btwn = 0.5*(ms[i+1]+ms[i])
                dm = ms[i+1]-ms[i]
                mabs = self.get_abs_from_app(m_in_btwn, z)
                dndm = self.schechter_lf_dm(mabs, z, band)*(10**(-3) * self.schechter_units)*cosmo.differential_comoving_volume(z)
                f_m = self.specific_flux(m_in_btwn, nu)
                psn = dm*dndm*f_m
                cl_shot += psn.to(u.nW*u.m**(-2)*u.steradian**(-1))

            shots.append(cl_shot.value*dz) # nW m^-2 sr^-1

        return np.sum(shots)





