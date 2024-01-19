import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import scipy
from matplotlib import cm

# from plotting_fns import show_window_functions, show_window_matrix

def compute_window_matrix(bandpower_edges, theta_window, N_ell=1000, Nbelow1000=1000, Nabove1000=2000, Nabove10000=5000, N_within_bp =200, plot=True, \
                         logminell=0, logmaxell=6, apodize=False, lndth=0.5, upper=False, lower=False):
    
    n_bandpower = len(bandpower_edges)-1
    
    window_matrix = np.zeros((n_bandpower, N_ell))

    for b in range(n_bandpower):
        if bandpower_edges[b] < 1000:
            N = Nbelow1000
        elif bandpower_edges[b] > 1000 and bandpower_edges[b] < 10000:
            N = Nabove1000
        else:
            N = Nabove10000
            
        bandpower_ells = np.logspace(logminell, logmaxell, N_ell)
        
        c_ells = np.zeros_like(bandpower_ells)
        
        for i, ell in enumerate(bandpower_ells):
            if apodize:
                integrand = make_integrand_hankel_transform_bp_config_window_apo(tophat_bp_window_config_space_indiv_apodize, ell, theta_window[0], theta_window[1],\
                                                                         bandpower_edges[b], bandpower_edges[b+1], lndth=lndth, upper=upper, lower=lower)
            else:
                integrand = make_integrand_hankel_transform_bp_config_window(tophat_bp_window_config_space_indiv, ell, theta_window[0], theta_window[1],\
                                                                         bandpower_edges[b], bandpower_edges[b+1])
            c_ells[i] = integrate_2pcf(integrand, N=N, a=theta_window[0], b=theta_window[1])
            
        if plot:
            plt.figure()
            plt.title(str(np.round(bandpower_edges[b], 1))+' < $\\ell$ < '+str(np.round(bandpower_edges[b+1], 1)))
            plt.plot(bandpower_ells, c_ells, label='windowed response')
            plt.plot(bandpower_ells, tophat_func(bandpower_ells, bandpower_edges[b], bandpower_edges[b+1]), label='bandpower')
            plt.xscale('log')
            plt.legend()
            plt.show()
            
        window_matrix[b,:] = c_ells
        
    if plot:
        plt.figure()
        plt.imshow(window_matrix)
        plt.colorbar()
        plt.show()
        
    return window_matrix, bandpower_ells



def density_simpsons(func, a, b, N):

    h = (b - a) / (N) 
    integral = func(a) + func(b)
    for i in np.arange(1, N, 2):
        add = func(a + i*h)
        integral += 4. * add
    for i in np.arange(2, N-1, 2):
        add = func(a + i*h)
        integral += 2 * add
    
    return (1/3)*h*integral


def c_ell_to_wtheta_hb_sp(ells, c_ell, thetabins, N=1000, a=1., b=1.):
    
    # first, handle any nans or infs that might be in c_ell
    
    nanmask = np.isnan(c_ell)
    if np.sum(nanmask) > 0:
        c_ell = c_ell[~nanmask]
        ells = ells[~nanmask]
        if b > np.max(ells):
            b = np.max(ells)

    cs = CubicSpline(ells, c_ell)
    thetaz = thetabins*np.pi/180.
    w_theta = np.zeros((len(thetaz)))

    for i, theta in enumerate(thetaz):
        integrand = make_integrand_hankel_transform_sp(cs, theta)
        w_theta[i] = integrate_2pcf(integrand, N=N, a=a, b=b)/(2*np.pi)

    return w_theta

def wtheta_to_c_ell_hb_sp(thetabins, wtheta, ells, N=1000, a=2e-3, b=1e-2):
    
    nanmask = np.isnan(wtheta)
    if np.sum(nanmask) > 0:
        print('weve got a nan up in here')
        wtheta = wtheta[~nanmask]
        thetabins = thetabins[~nanmask]
        if b > np.max(thetabins):
            b = np.max(thetabins)

    bins_rad = thetabins*np.pi/180.
    cs = CubicSpline(bins_rad, wtheta)
    c_ell = np.zeros((len(ells)))
    for i, ell in enumerate(ells):
        integrand = make_integrand_hankel_transform_sp(cs, ell)
        c_ell[i] = 2*np.pi*integrate_2pcf(integrand, N = N, a=a, b=b)
        
    return c_ell


def get_edges(minval, maxval, nbins, logbin=True):
    if logbin:
        bin_edges = np.logspace(np.log10(minval), np.log10(maxval), nbins+1)
    else:
        bin_edges = np.linspace(minval, maxval, nbins+1)
        
    return bin_edges


def get_window_matrix_bp_edge_idx(bandpower_edges, bandpower_ells):
    idxs = []
    
    minlogb = np.log10(np.min(bandpower_ells))
    maxlogb = np.log10(np.max(bandpower_ells))
    for b in bandpower_edges:
        logb = np.log10(b)
        
        idxs.append(len(bandpower_ells)*(logb-minlogb)/(maxlogb-minlogb))
        
    return idxs


def g_theta_tophat_bandpass(thetas, ell_min, ell_max):
    return (ell_max*scipy.special.j1(thetas*ell_max)-ell_min*scipy.special.j1(thetas*ell_min))/thetas
    
def g_theta_tophat_bandpass_apodize(thetas, ell_min, ell_max, theta_window, lndth=0.25):
    filtfunc_full = (ell_max*scipy.special.j1(thetas*ell_max)-ell_min*scipy.special.j1(thetas*ell_min))/thetas
    hann_window_vals = np.array([hann_window(np.log10(theta), lndth, np.log10(theta_window[0]), np.log10(theta_window[1])) for theta in thetas])
    return filtfunc_full*hann_window_vals

def hann_window(x, delta_x, x_lo, x_hi):
    if x < x_lo - 0.5*delta_x:
        return 0.
    elif x < x_lo + 0.5*delta_x:
        return np.cos(0.5*np.pi*(x-(x_lo+0.5*delta_x))/delta_x)**2
    elif x < x_hi-0.5*delta_x:
        return 1.
    elif x < x_hi+0.5*delta_x:
        return np.cos(0.5*np.pi*(x-(x_hi-0.5*delta_x))/delta_x)**2
    else:
        return 0.
    
def lower_hann_window(x, delta_x, x_lo, x_hi):
    if x < x_lo - 0.5*delta_x:
        return 0.
    elif x < x_lo + 0.5*delta_x:
        return np.cos(0.5*np.pi*(x-(x_lo+0.5*delta_x))/delta_x)**2
    elif x < x_hi:
        return 1.
    else:
        return 0.
    
def upper_hann_window(x, delta_x, x_lo, x_hi):
    if x < x_lo:
        return 0.
    elif x < x_hi-0.5*delta_x:
        return 1.
    elif x < x_hi+0.5*delta_x:
        return np.cos(0.5*np.pi*(x-(x_hi-0.5*delta_x))/delta_x)**2
    else:
        return 0.


def integrate_2pcf(integrand, N=100, a=None, b=None, direction='FORWARD'):
    if a is None:
        if direction=='FORWARD':
            a = 2e-3
            b = 1e-2
        else:
            a = 200
            b = 1e5

    return density_simpsons(integrand, a=a, b=b, N=N)

def integrate_spline_cell(integrand, ell, N=10, a=2e-3, b=1e-2):
    return density_simpsons(integrand, a=a, b=b, N=N)

def integrate_spline_wtheta(integrand, theta, N=100, a=200, b=1e5):
    return density_simpsons(integrand, a=a, b=b, N=N)

def integrate_w_theta(ls, w, weights=None):
    ''' Integrate potentially weighted angular correlation function. If no weights are provided, then inverse theta weighting is used.'''

    thetas = np.pi/ls
    dthetas = thetas[:-1]-thetas[1:]
    w_integrand = 0.5*(w[:-1]+w[1:])
    if weights is None: # then use inverse theta weighting
        avthetas = 0.5*(thetas[:-1]+thetas[1:])
        weights = 1./avthetas
        
    w_integrand *= weights
    w_integrand *= dthetas
    return np.sum(w_integrand)

def integrate_C_l(ls, C, weights=None):
    ''' Integrate potentially weighted angular correlation function. If no weights are provided, then inverse multipole weighting is used.'''

    dls = ls[:-1]-ls[1:]
    C_integrand = 0.5*(C[:-1]+C[1:])
    if weights is None:
        weights = 0.5*(ls[1:]+ls[:-1])
        
    C_integrand *= weights
    C_integrand *= dls
    return np.sum(C_integrand)

def make_integrand_theta_sp(spline, theta):
    def integrand_func(ell):
        function = (ell * spline(ell) * scipy.special.j0(ell*theta))
        return function
    return integrand_func

def make_integrand_hankel_transform_sp(spline, a):
    def integrand_func(b):
        function = (b * spline(b) * scipy.special.j0(a*b))
        return function
    return integrand_func

def make_integrand_hankel_transform_window(tophat, a, bin_min, bin_max):
    def integrand_func(b):
        function = (b * tophat(b, bin_min, bin_max) * scipy.special.j0(a*b))
        return function
    return integrand_func

def make_integrand_hankel_transform_bp_config_window(config_tophat, a, bin_min, bin_max, ell_min, ell_max):
    def integrand_func(b):
        function = (b * config_tophat(b, ell_min, ell_max, thetamin=bin_min, thetamax=bin_max) * scipy.special.j0(a*b))
        return function
    return integrand_func

def make_integrand_hankel_transform_bp_config_window_apo(config_tophat, a, bin_min, bin_max, ell_min, ell_max, lndth=0.5, upper=False, lower=False):
    def integrand_func(b):
        function = (b * config_tophat(b, ell_min, ell_max, thetamin=bin_min, thetamax=bin_max, lndth=lndth, upper=upper, lower=lower) * scipy.special.j0(a*b))
        return function
    return integrand_func


def project_cl_window_fn(cl, window_matrix, window_ells=None, dells_window=None, bandpower_ells=None, dells_bandpower=None):
    if dells_window is None:
        dells_window = window_ells[1:]-window_ells[:-1]
    if dells_bandpower is None:
        dells_bandpower = bandpower_ells[1:]-bandpower_ells[:-1]
        
    cl_window_project = np.dot(window_matrix, cl*dells_window)/dells_bandpower
    
    return cl_window_project

def wtheta_to_c_ell_hb_sp_9_10(thetabins, wtheta, ells, N=1000, a=2e-3, b=1e-2, preconvolve_sig=None, plot_integrands=False, \
                             integrand_ells = [200., 2000., 20000., 100000.], Nbelow1000=1000, Nabove1000=5000, Nabove10000=20000):
    
        
    nanmask = np.isnan(wtheta)
    if np.sum(nanmask) > 0:
        print('weve got a nan up in here')
        wtheta = wtheta[~nanmask]
        thetabins = thetabins[~nanmask]
        if b > np.max(thetabins):
            b = np.max(thetabins)
            
    dth = np.abs(thetabins[1]-thetabins[0])
    
    bins_rad = thetabins*np.pi/180.

    if preconvolve_sig is not None:
        conv_wtheta = scipy.ndimage.gaussian_filter1d(wtheta, preconvolve_sig/dth)
        cs = CubicSpline(bins_rad, conv_wtheta)
    else:
        cs = CubicSpline(bins_rad, wtheta)
    
    if plot_integrands:
        # just for show
        figs = []
        fine_b = np.linspace(a, np.max(thetabins)*np.pi/180., N)
        for ello in integrand_ells:
            figs.append(plot_hankel_integrand(bins_rad, fine_b, wtheta, cs, ello, b))
            
    
    c_ell = np.zeros((len(ells)))
    for i, ell in enumerate(ells):
        if ell < 1000:
            N = Nbelow1000
        elif ell > 1000 and ell < 10000:
            N = Nabove1000
        elif ell > 10000:
            N = Nabove10000
            
        integrand = make_integrand_hankel_transform_sp(cs, ell)
        c_ell[i] = 2*np.pi*integrate_2pcf(integrand, N = N, a=a, b=b)
        
    if plot_integrands:
        return c_ell, fine_b, cs(fine_b), figs
        
    return c_ell


def tophat_func(x, binmin, binmax):
    arr = np.zeros_like(x)
    arr[(x >= binmin)*(x < binmax)] = 1.
    return arr

def tophat_bp_window_config_space_indiv(theta, ell_min, ell_max, thetamin=0., thetamax=np.inf):
    if theta >= thetamin and theta < thetamax:
        return g_theta_tophat_bandpass(theta, ell_min, ell_max)
    return 0.

def tophat_bp_window_config_space(thetas, ell_min, ell_max, thetamin=0., thetamax=np.inf):
    
    tophat_vals = np.zeros_like(thetas)
    mask = (thetas >= thetamin)*(thetas < thetamax)    
    tophat_vals[mask] = g_theta_tophat_bandpass(thetas[mask], ell_min, ell_max)
    
    return tophat_vals

def tophat_bp_window_config_space_indiv_apodize(theta, ell_min, ell_max, thetamin=0., thetamax=np.inf, lndth=0.5, upper=False, lower=False):
    if upper:
        return g_theta_tophat_bandpass(theta, ell_min, ell_max)*upper_hann_window(np.log10(theta), lndth, np.log10(thetamin), np.log10(thetamax))
    if lower:
        return g_theta_tophat_bandpass(theta, ell_min, ell_max)*lower_hann_window(np.log10(theta), lndth, np.log10(thetamin), np.log10(thetamax))
    else:
        return g_theta_tophat_bandpass(theta, ell_min, ell_max)*hann_window(np.log10(theta), lndth, np.log10(thetamin), np.log10(thetamax))



def show_window_functions(bandpower_edges, bp_ells, window_matrix, return_fig=True):
    cmap = cm.get_cmap('viridis', 20)
    cmap_colors = cmap(np.arange(len(bandpower_edges)-1))

    f = plt.figure(figsize=(15, 4))
    for b, bp_edge in enumerate(bandpower_edges[:-1]):
        if b==0:
            label = 'Bandpowers'
        else:
            label = None
        plt.plot(bp_ells, tophat_func(bp_ells, bandpower_edges[b], bandpower_edges[b+1]), color='k', label=label)

    for b, bp_edge in enumerate(bandpower_edges[:-1]):
        if b==0:
            label = 'Window functions'
        else:
            label = None
        plt.plot(bp_ells, window_matrix[b], color=cmap_colors[b], label=label)
    plt.xscale('log')
    plt.xlabel('Multipole $\\ell$', fontsize=18)
    plt.ylabel('Window functions $W_b(\\ell)$', fontsize=18)
    plt.ylim(-0.5, 1.25)
    plt.legend(fontsize=14)
    plt.show()

    if return_fig:
        return f
    
def show_window_matrix(window_matrix, theta_window, bp_edge_idxs=None, return_fig=True, title=None):
    fig = plt.figure(figsize=(15, 3))
    if title is None:
        plt.title('Window matrix, $\\theta_{min}=$'+str(np.round(theta_window[0]*(180./np.pi)*3600.))+'\", $\\theta_{max}=$'+str(np.round(theta_window[1]*(180./np.pi), 1))+' deg', \
             fontsize=18)
    else:
        plt.title(title, fontsize=18)
    plt.imshow(window_matrix, cmap='Greys', aspect='auto', vmin=-1, vmax=1)
    if bp_edge_idxs is not None:
        for i, idx in enumerate(bp_edge_idxs):
            if i==0:
                label = 'Bandpower edges'
            else:
                label = None
            plt.axvline(idx, color='r', linewidth=1, label=label)
    plt.legend(loc=1)
    tickmark = np.linspace(0, window_matrix.shape[1], 7)
    exp_marks = ['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$', '$10^6$']
    plt.xticks(tickmark, exp_marks)
    plt.yticks(np.arange(0, window_matrix.shape[0], 4), np.arange(0, window_matrix.shape[0], 4))
    plt.xlabel('Multipole $\\ell$', fontsize=18)
    plt.ylabel('Bandpower index', fontsize=18)
    plt.colorbar()
    plt.show()
    
    if return_fig:
        return fig



