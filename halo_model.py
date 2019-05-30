import scipy
from scipy.special import *
import astropy.units as u

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
    
    def __init__(self):
        self.rho = (3*1.16*10**12/(4*np.pi))*self.omega_m*u.solMass*self.h**2*u.Mpc**(-3)
        self.m_star = (4*np.pi*self.R_star**3/3)*self.rho
        self.tinker = tinker_hmf()
    
    def onehalo_corr(self):
        pass
    
    def twohalo_corr(self):
        pass
    
    def dm_corr(self):
        dm_corr_fn = self.onehalo_corr()+self.twohalo_corr()
        return dm_corr_fn

    def total_cross_power_spectrum(self):
        return self.cross_power_spectrum_1h()+self.cross_power_spectrum_2h()
    

    def concentration_pdf(self, c):
        if self.sig_lnc != 0:
            # lognormal
            return (1/np.sqrt(2*np.pi*self.sig_lnc**2))*np.exp(-(np.log(c/self.cmean))**2/(2*self.sig_lnc**2))
        else:
            # delta function distribution used sometimes
            if c==self.c_mean:
                return 1.0
            else:
                return 0.0
    
    def concentration(self, m, z=0):
        return (9./(1.+z))*(m/self.m_star)**(-0.13)
    
    def f(self, c):
        return 1./(np.log(1.+c)-1./(1.+c))
    
    def mass_2_virial_radius(self, m):
        rvir = (3*m/(4*np.pi*self.rho*self.Delta)).value**(1./3)
        return rvir*u.Mpc
    
    def mass_variance(self, m):
        #TODO
        return 1.
    
    def mass_2_nu(self, m):
        return (self.tinker.delta_c/self.mass_variance(m))**2
            
    def NFW_r(self, rfrac, m, normalize=True):
        R_vir = self.mass_2_virial_radius(m)
        c = self.concentration(m)   
        ur = (self.f(c)*c**3/(4*np.pi*(R_vir**3)))/(c*rfrac*(1+c*rfrac)**2)

        if normalize:
        	return ur/np.max(ur)
        else:
        	return ur
    
    def NFW_k(self, k, m, normalize=True):
        R = self.mass_2_virial_radius(m)*self.Delta**(-1./3)
        c = self.concentration(m)
    
        khat = k*self.R_star*self.Delta**(-1./3)
        y = R/self.R_star

        kappa = khat*y/c
        
        SI_1, CI_1 = sici(kappa.value*(1+c.value)) # sine/cosine integrals from Scoccimaro et al. (2001)
        SI_2, CI_2 = sici(kappa.value)
        
        u = self.f(c)*(np.sin(kappa.value)*(SI_1-SI_2)+np.cos(kappa.value)*(CI_1-CI_2)-(np.sin(kappa.value*c.value)/(kappa.value*(1+c.value))))
        
        if normalize:
        	return u/np.max(u)
        else:
        	return u
    
    def P_lin(self, k, z):
        return self.S8*k**self.ns
    
    def P_hh(self, k, m1=1, m2=1):
        # figure out how to get biases
#         return self.bias(m1, n=1)*self.bias(m2, n=2)*self.P_lin(k)
        return self.P_lin(k)

    def dm_power_spectrum_1h(self, k):
        pks = np.zeros_like(k)
        dm = self.mrange[1]-self.mrange[0] # maybe should do log bins
        for m in self.mrange:
            R = self.mass_2_virial_radius(m)*self.Delta**(-1./3)
            c = self.concentration(m)
            pk = dm*self.n(m)*(m/self.rho)**2 * np.abs(self.NFW_k(k, R, c))**2
            pks += pk
        return pks
        
    def dm_power_spectrum_2h(self, k):

        pk_2halo = np.zeros_like(k)
        dm = self.mrange[1]-self.mrange[0]
    
        for m1 in self.mrange:
            int2 = np.zeros_like(k)
            for m2 in self.mrange:
                int2 += dm*self.n(m2)*(m2/self.rho)*self.NFW_k(k,m=m2)*self.P_hh(k, m1=m1,m2=m2)
            
            pk_2halo += dm*self.n(m1)*(m1/self.rho)*self.NFW_k(k,m=m1)*int2
            
        return pk_2halo
    
    def dm_total_power_spectrum(self, k):
        return self.dm_power_spectrum_1h(k)+self.dm_power_spectrum_2h(k)
    

    