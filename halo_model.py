import scipy
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from scipy.special import sici
import numpy as np
from hmf import MassFunction
# initialize cosmology class 
cosmo = FlatLambdaCDM(H0=70, Om0=0.28)


class tinker_hmf():
    
    Delta = 200. # fiducial overdensity for spherical collapse, fixes normalization factors below
    alpha = 0.368
    beta0 = 0.589
    gamma0 = 0.864
    phi0 = -0.729
    eta0 = -0.243
    delta_c = 1.686 # local peak collapse threshold
    def __init__(self):
        pass
    
    def beta(self, z):
        return self.beta0*(1+z)**(0.20)
    
    def phi(self, z):
        return self.phi0*(1+z)**(-0.08)
    
    def eta(self, z):
        return self.eta0*(1+z)**(0.27)
    
    def gamma(self, z):
        return self.gamma0*(1+z)**(-0.01)
    
    def nu_fnu(self, nu, z=0):
        return nu*self.alpha*(1 + (self.beta(z)*nu)**(-2*self.phi(z)))*nu**(2*self.eta(z))*np.exp(-self.gamma(z)*nu**2/2)
    
    def eulerian_bias(self, nu, z=0):
        return 1. + ((self.gamma(z)*nu**2 -(1+2*self.eta(z)))/self.delta_c) + (2*self.phi(z)/self.delta_c)/(1+(self.beta(z)*nu)**(2*self.phi(z)))
    

class halo_model():
    
    # NFW or M99 profile? might be good to get analytical version of convolved profiles
    h = 0.67 
    sig_lnc = 0.25 
    c_mean = 1.
    R_star = (3.135/h)*u.Mpc
    S8 = 0.8 # sigma_8 from LCDM
    ns = 0.96
    omega_m = 0.3
    Delta = 200. # fiducial overdensity for spherical collapse, fixes normalization factors below
    
    linpars = camb.CAMBparams()
    linpars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    linpars.InitPower.set_params(ns=0.965)
    
    def __init__(self, redshifts=None):
        
        self.hmf = MassFunction(dlog10m=0.02)
        self.mrange = self.hmf.m*u.solMass
        self.rho = self.hmf.rho_gtm[0]*u.solMass*u.Mpc**(-3)
        self.m_star = (4*np.pi*self.R_star**3/3)*self.rho
        self.kmax = 1e3/self.h
        print('self.mstar:', self.m_star)
        print('self.rho:', self.rho)

        
        if redshifts is None:
            self.redshifts = np.array([0.0])
            self.linpars.set_matter_power(kmax=self.kmax)
        else:
            self.redshifts = redshifts
            self.linpars.set_matter_power(redshifts=redshifts, kmax=self.kmax)
        
        self.linpars.NonLinear = model.NonLinear_none
        self.lin_results = camb.get_results(self.linpars)
    
    def onehalo_corr(self):
        pass
    
    def twohalo_corr(self):
        pass
    
    def dm_corr(self):
        dm_corr_fn = self.onehalo_corr()+self.twohalo_corr()
        return dm_corr_fn

    def total_cross_power_spectrum(self):
        return self.cross_power_spectrum_1h()+self.cross_power_spectrum_2h()
    
    def concentration(self, m, z=0):
        return (9./(1.+z))*(m/self.m_star)**(-0.13)
    
    def f(self, c):
        return 1./(np.log(1.+c)-1./(1.+c))

    def mass_2_virial_radius(self, halo_mass, z=0):
        term = (3/(4*np.pi))*cosmo.Om(z)*halo_mass.to(u.g)/(200*cosmo.critical_density(z))
        return term**(1./3.)
            
    def NFW_r(self, rfrac, m, normalize=True):
        R_vir = self.mass_2_virial_radius(m).to(u.Mpc).value
        c = self.concentration(m)   
        ur = (self.f(c)*c**3/(4*np.pi*(R_vir**3)))/(c*rfrac*(1+c*rfrac)**2)
        if normalize:
            return ur/np.max(ur)
        else:
            return ur
    
    def NFW_k(self, k, m, normalize=True):
        
        ''' This parameterization is from Scoccimarro et al. 2000 
            "How Many Galaxies Fit in a Halo?" '''
        
        R = self.mass_2_virial_radius(m).to(u.Mpc).value*self.Delta**(1./3.)
        c = self.concentration(m)
        khat = k*self.R_star*self.Delta**(-1./3.)
        y = (R/self.R_star)/0.67
        
        kappa = np.array([khat.value*y[i].value/c[i].value for i in range(len(c))])
        kappa_c = np.array([kappa[i]*c[i].value for i in range(len(c))])
        kappa_1plus_c = np.array([kappa[i]*(1.+c[i].value) for i in range(len(c))])


        SI_1, CI_1 = sici(kappa_1plus_c)
        SI_2, CI_2 = sici(kappa)
        
        ufunc_no_f = np.sin(kappa)*(SI_1-SI_2)+np.cos(kappa)*(CI_1-CI_2)-(np.sin(kappa_c)/kappa_1plus_c)
        fs = self.f(c)
        ufunc = np.array([fs[i]*ufunc_no_f[i] for i in range(len(c))])
        
#         print('R has shape ', R.shape)
#         print('concentration has shape ', c.shape)
#         print('khat has shape', khat.shape)
#         print('y has shape', y.shape)
#         print('kappa has shape', kappa.shape, kappa_c.shape, kappa_1plus_c.shape)
#         print('ufunc has shape ', ufunc.shape)
                        
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
        prof = self.NFW_k(k_h*halomod.h, self.mrange[:-1])
        
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

def limber_project(halo_ps, zmin, zmax, ng=None, flux_prod_rate=None, nbin=10, ell_min=90, ell_max=1e5, n_ell_bin=30):
    ''' This currently takes in the dark matter halo power spectrum from the hmf package. The only place where this is
    used is when updating the redshift of the power spectrum. Should add option to use array of power spectra as well'''
    ell_space = 10**(np.linspace(np.log10(ell_min), np.log10(ell_max), n_ell_bin))
    cls = np.zeros((nbin, n_ell_bin))    
    zs = np.linspace(zmin, zmax, nbin+1)
    dz = zs[1]-zs[0]
    central_zs = 0.5*(zs[1:]+zs[:-1])
    D_A = cosmo.angular_diameter_distance(central_zs)
    H_z = cosmo.H(central_zs)
    inv_product = (const.c.to('km/s')/(16*np.pi**2))*(dz/(H_z*D_A**2*(1+central_zs)**2))
    
    
    for i in range(nbin):
        halo_ps.update(z=central_zs[i])
        ks = ell_space / cosmo.comoving_distance(central_zs[i])        
        logfit = np.poly1d(np.polyfit(np.log10(halo_ps.k), np.log10(halo_ps.nonlinear_power), 15))
        ps = 10**(logfit(np.log10(ks.value)))
        cls[i] = ps
    
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
    

