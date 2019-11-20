import scipy
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from scipy.special import sici
from scipy import interpolate
import numpy as np
from hmf import MassFunction
import camb
from camb import model
from astropy import constants as const

# initialize cosmology class 
cosmo = FlatLambdaCDM(H0=70, Om0=0.28)

class halo_model_class():
    
    # NFW or M99 profile? might be good to get analytical version of convolved profiles
    h = 0.67 
    R_star = (3.135/h)*u.Mpc
    Delta = 200. # fiducial overdensity for spherical collapse, fixes normalization factors below
    
    linpars = camb.CAMBparams()
    linpars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    linpars.InitPower.set_params(ns=0.965)
    
    def __init__(self, redshifts=None):
        
        self.hmf = MassFunction(dlog10m=0.02)
        self.mrange = self.hmf.m*u.solMass
        self.rho = self.hmf.rho_gtm[0]*u.solMass*u.Mpc**(-3) # mean number of DM halos per Mpc^3
        self.m_star = (4*np.pi*self.R_star**3/3)*self.rho # used as pivot mass for halo concentration
        self.kmax = 1e3/self.h
        
        if redshifts is None:
            self.redshifts = np.array([0.0])
            self.linpars.set_matter_power(kmax=self.kmax)
        else:
            self.redshifts = redshifts
            self.linpars.set_matter_power(redshifts=redshifts, kmax=self.kmax)
        
        self.linpars.NonLinear = model.NonLinear_none
        self.lin_results = camb.get_results(self.linpars)
    
    ''' The concentration is the ratio c_{200} = r_{200}/r_s, where r_s is the scale radius and r_{200} is
    the radius of a halo where the density is 200 times the critical density of the universe. '''
    def concentration(self, m, z=0):
        return (9./(1.+z))*(m/self.m_star)**(-0.13)
    
    ''' Used when computing the NFW profile '''
    def f(self, c):
        return 1./(np.log(1.+c)-1./(1.+c))

    ''' This assumes the halo is spherically symmetric at least to determine the virial radius '''
    def mass_2_virial_radius(self, halo_mass, z=0):
        R_vir_cubed = (3/(4*np.pi))*cosmo.Om(z)*halo_mass.to(u.g)/(200*cosmo.critical_density(z))
        return R_vir_cubed**(1./3.)
            
    def NFW_r(self, rfrac, m, normalize=True):
        R_vir = self.mass_2_virial_radius(m).to(u.Mpc).value
        c = self.concentration(m)   
        u_r = (self.f(c)*c**3/(4*np.pi*(R_vir**3)))/(c*rfrac*(1+c*rfrac)**2)
        if normalize:
            return u_r/np.max(u_r)
        else:
            return u_r
    
    def NFW_k(self, k, m, normalize=True):
        
        ''' This parameterization is from Scoccimarro et al. 2000 
            "How Many Galaxies Fit in a Halo?" '''
        R = self.mass_2_virial_radius(m).to(u.Mpc).value*self.Delta**(1./3.)
        c = self.concentration(m)
        khat = k*self.R_star*self.Delta**(-1./3.)
        y = ((R/self.R_star)/self.h)

        kappa = np.array([khat.value*y[i].value/c[i].value for i in range(len(c))])
        kappa_c = np.array([kappa[i]*c[i].value for i in range(len(c))])
        kappa_1plus_c = np.array([kappa[i]*(1.+c[i].value) for i in range(len(c))])
        
        SI_1, CI_1 = sici(kappa_1plus_c)
        SI_2, CI_2 = sici(kappa)
        
        # no fs given, yet...
        ufunc_no_f = np.sin(kappa)*(SI_1-SI_2)+np.cos(kappa)*(CI_1-CI_2)-(np.sin(kappa_c)/kappa_1plus_c)
        
        fs = self.f(c)

        ufunc = np.array([fs[i]*ufunc_no_f[i] for i in range(len(c))])
                        
        if normalize:
            return np.array([ufunc[i]/np.max(ufunc[i]) for i in range(ufunc.shape[0])])
        else:
            return ufunc
    
    def P_lin(self, ks):
        ''' get_matter_power_spectrum takes in k/h (NOT k!) and outputs P_lin(k/h) and k/h values
        make sure to use k/h values output since the linear power spectrum re-interpolates within the set range'''
        kh, z, pk = self.lin_results.get_matter_power_spectrum(minkh=np.min(ks/self.h), maxkh=np.max(ks/self.h), npoints=len(ks))
        return kh, z, pk
    
    def P_hh(self, k, m1=1, m2=1):
        k_h, z, pk_lin = self.P_lin(k) # computes P_lin(k/h)
        return k_h, z, pk_lin

    def dm_power_spectrum_1h(self, k, integral=True):
        ''' The one halo term appears to work, though it seems slightly higher than the 
        power spectrum from Cooray and Sheth. Returns the integral over halo mass if boolean set to True'''
            
        # dm
        dm = self.mrange[1:]-self.mrange[:-1]
        
        # |u(k|m)|^2
        prof_squared = self.h**2*np.abs(self.NFW_k(k, self.hmf.m[:-1]*u.solMass))**2

        
        # (m/rho)^2
        m_rho_squared = (self.hmf.m[:-1]/self.rho)**2

        # dm*(dN/dm)        
        dm_dndm = dm*self.hmf.dndm[:-1]*(u.solMass**(-1)*u.Mpc**(-3))
        
        pks = np.array([prof_squared[i]*m_rho_squared[i]*dm_dndm[i] for i in range(len(m_rho_squared))])

        if integral:
            return np.sum(pks, axis=0)
        else:
            return pks
        
    def dm_power_spectrum_2h(self, k, bias=1.0):
        
        k_h, z, pk_lin = self.P_hh(k) # outputs k/h array with units Mpc^-1, P(k) [Mpc^3 h^-3]
        
        dm = self.mrange[1:]-self.mrange[:-1]

        # dm*(dN/dm) [M_sol/h * h^4 M_sol^-1 Mpc^-3] = [h^3 Mpc^-3]
        dm_dndm = dm*self.hmf.dndm[:-1]*(u.solMass**(-1)*u.Mpc**(-3))
        
        # u(k|m) [unitless, but I think theres an extra h^-1 around here]
        prof = self.NFW_k(k_h*self.h, self.mrange[:-1])
        
        # (m/rho) [(M_sol/h) / (M_sol h^2 Mpc^-3)] = [h^-3 Mpc^3]
        m_over_rho = self.hmf.m[:-1]/self.rho
        
        # dm*(dN/dm)*(m/rho)*u(k|m) [h^3 Mpc^-3 h^-3 Mpc^3] = [unitless (h^-1)]
        prod = np.array([dm_dndm[i]*m_over_rho[i]*prof[i] for i in range(len(m_over_rho))])

        # \int dm*(dN/dm)*(m/rho)*u(k|m)*P_lin(k,z) [(h^-2) h^-3 Mpc^3] = [Mpc^3 h^-3 (h^-1)]
        inner_int = np.array([np.sum([prod[i]*pk_lin[j] for i in range(prod.shape[0])], axis=0) for j in range(pk_lin.shape[0])])
        
        # [Mpc^3 h^-3]
        p2h = self.h**(-1)*np.array([np.sum([prod[i]*inner_int[j] for i in range(prod.shape[0])], axis=0) for j in range(pk_lin.shape[0])])        
                
        return k_h, p2h

    
    def dm_total_power_spectrum(self, k):
        k_h, pk_2h = self.dm_power_spectrum_2h(k)
        pk_1h = self.dm_power_spectrum_1h(k_h*self.h)
        
        if len(self.redshifts)>1:
            return np.array([pk_1h+pk_2h[j] for j in range(len(self.redshifts))])
        else:
            return pk_1h+pk_2h



# this is for projecting a 3D power spectrum to a 2D angular power spectrum.

def limber_project(halo_ps, zmin, zmax, ng=None, flux_prod_rate=None, nbin=20, ell_min=90, ell_max=1e5, n_ell_bin=30):
    ''' This currently takes in the dark matter halo power spectrum from the hmf package. The only place where this is
    used is when updating the redshift of the power spectrum. Should add option to use array of power spectra as well'''
    cls = np.zeros((nbin, n_ell_bin))    

    ell_space = 10**(np.linspace(np.log10(ell_min), np.log10(ell_max), n_ell_bin))
    zs = np.linspace(zmin, zmax, nbin+1)
    dz = zs[1]-zs[0]
    central_zs = 0.5*(zs[1:]+zs[:-1])
    
    D_A = cosmo.angular_diameter_distance(central_zs)/cosmo.h # has units h^-1 Mpc
    H_z = cosmo.H(central_zs) # has units km/s/Mpc (or km/s/(h^-1 Mpc)?)
    inv_product = (cosmo.h*dz*H_z)/(const.c.to('km/s')*(1.+central_zs)*D_A**2)
    
    for i in range(nbin):
        halo_ps.update(z=central_zs[i])
        ks = ell_space / (cosmo.comoving_distance(central_zs[i]))
        log_spline = interpolate.InterpolatedUnivariateSpline(np.log10(halo_ps.k), np.log10(halo_ps.nonlinear_power))    
        ps = 10**(log_spline(np.log10(ks.value)))
        cls[i] = ps

    ''' One can compute the auto and cross power spectrum for intensity maps/tracer catalogs 
    by specifying flux_prod_rate and/or ng (average number of galaxies per steradian)''' 
    
    if flux_prod_rate is not None:
        if ng is not None:
            cls = np.array([flux_prod_rate[i]*ng**(-1)*inv_product[i]*cls[i] for i in range(inv_product.shape[0])])
        else:
            cls = np.array([flux_prod_rate[i]**2*inv_product[i]*cls[i] for i in range(inv_product.shape[0])])
    else:
        if ng is not None:
            cls = np.array([ng**(-2)*inv_product[i]*cls[i] for i in range(inv_product.shape[0])])
        else:
            cls = np.array([inv_product[i]*cls[i] for i in range(inv_product.shape[0])])

    integral_cl = np.sum(cls, axis=0)

    return ell_space, integral_cl





