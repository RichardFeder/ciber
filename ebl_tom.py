import numpy as np
import camb
from camb import get_matter_power_interpolator, model
import config
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
from scipy.optimize import minimize
from scipy import sparse
from scipy.stats import norm

from ciber_data_file_utils import *
from gal_plotting_fns import *



# from scipy.interpolate import interp1d

def deconvolve_with_regularization(bI_I_measured, bI_I_err, K_tilde, 
                                   alpha=0.0, regularization='tikhonov', 
                                   verbose=True):
    """
    Deconvolve with optional regularization to reduce noise amplification.
    
    Parameters:
    -----------
    alpha : float
        Regularization strength (0 = no regularization)
    regularization : str
        'tikhonov' - L2 regularization (smoothness)
        'tikhonov_1st' - First derivative regularization (smoothness)
        'tikhonov_2nd' - Second derivative regularization (extra smooth)
    """
    
    N_photo, N_true = K_tilde.shape
    
    if verbose:
        print(f"Regularized deconvolution:")
        print(f"  Method: {regularization}")
        print(f"  Alpha: {alpha}")
    
    # Build regularization matrix
    if alpha > 0:
        if regularization == 'tikhonov':
            # Standard L2 regularization (Tikhonov)
            L = np.eye(N_true)
            
        elif regularization == 'tikhonov_1st':
            # First derivative regularization
            L = np.zeros((N_true-1, N_true))
            for i in range(N_true-1):
                L[i, i] = -1
                L[i, i+1] = 1
                
        elif regularization == 'tikhonov_2nd':
            # Second derivative regularization  
            L = np.zeros((N_true-2, N_true))
            for i in range(N_true-2):
                L[i, i] = 1
                L[i, i+1] = -2
                L[i, i+2] = 1
        else:
            raise ValueError(f"Unknown regularization: {regularization}")
    
    # Set up the regularized system
    # min ||K*x - b||² + alpha²*||L*x||²
    
    if alpha > 0:
        # Augmented system
        A_aug = np.vstack([K_tilde / bI_I_err[:, np.newaxis], 
                          alpha * L])
        b_aug = np.concatenate([bI_I_measured / bI_I_err, 
                               np.zeros(L.shape[0])])
    else:
        # No regularization
        A_aug = K_tilde / bI_I_err[:, np.newaxis]
        b_aug = bI_I_measured / bI_I_err
    
    # Solve
    bI_I_true, residuals, rank, s = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    
    if verbose:
        print(f"\nSolution statistics:")
        print(f"  Range: [{bI_I_true.min():.2e}, {bI_I_true.max():.2e}]")
        print(f"  Standard deviation: {np.std(bI_I_true):.2e}")
        
        # Compute roughness (measure of jumpiness)
        roughness = np.sum(np.diff(bI_I_true)**2)
        print(f"  Roughness (sum of squared differences): {roughness:.2e}")
    
    # Error estimation (approximate for regularized case)
    try:
        if alpha > 0:
            # For regularized solution, this is approximate
            W_inv = np.diag(1.0 / bI_I_err**2)
            # Use the generalized inverse for error estimation
            K_reg = K_tilde.T @ W_inv @ K_tilde + alpha**2 * (L.T @ L)
            cov_matrix = np.linalg.inv(K_reg)
            bI_I_true_err = np.sqrt(np.diag(cov_matrix))
        else:
            W_inv = np.diag(1.0 / bI_I_err**2)
            cov_matrix = np.linalg.inv(K_tilde.T @ W_inv @ K_tilde)
            bI_I_true_err = np.sqrt(np.diag(cov_matrix))
    except:
        bI_I_true_err = np.abs(bI_I_true) * 0.3
    
    # Validation
    reconstructed = K_tilde @ bI_I_true
    residuals = reconstructed - bI_I_measured
    chi2 = np.sum((residuals / bI_I_err)**2)
    dof = max(1, len(bI_I_measured) - len(bI_I_true))
    
    if verbose:
        print(f"\nFit quality:")
        print(f"  χ²/dof = {chi2:.2f}/{dof} = {chi2/dof:.2f}")
    
    return bI_I_true, bI_I_true_err, reconstructed, chi2/dof


def proc_ebl_tomography(inst_list=[1, 2], catname='LS', maskstr='JHlt16',
                        zbinedges=None, zbinedges_fine=None, 
                       ell_max = 2000., ell_min=None, deconvolve=False, ell_effective=1000., alpha=0.001, subtract_sn=True, \
                       avg_method="weighted", ell_selection=None, method='nnls', 
                       load_wg=True, headstr=None, with_ff_err=False, subtract_randoms=True,
                       ifield_list=[4, 5, 6, 7, 8]):
    

    # Your existing code for loading data
    
    if zbinedges is None:
        zbinedges = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    if zbinedges_fine is None:
        zbinedges_fine = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    if load_wg:
        Wg_binned, z_true_centers = load_binned_Wg(zbinedges, zbinedges_fine)

    # Load power spectra
    lb, full_cl_gI, full_cl_gI_err, full_cl_gg, full_cl_gg_err = collect_ciber_gal_vs_redshift(catname, 
                                                                    subtract_randoms=subtract_randoms, 
                                                                    maskstr=maskstr, 
                                                                    zbinedges=zbinedges, \
                                                                    subtract_sn=False, 
                                                                    headstr=headstr, 
                                                                    with_ff_err=with_ff_err, 
                                                                    ifield_list=ifield_list)
    
    ng_tot = collect_ngtot(catname, zbinedges=zbinedges,headstr=headstr, ifield_list=ifield_list)
            
    z_fine, all_dNdz_b, all_dNdz_b_err = load_norm_bg_dndz(zbinedges_fine, ng_tot=None)
    
    
    if subtract_sn:
        deg2_to_sr = (np.pi/180)**2  # approximately 3.05e-4 sr/deg^2
        shot_noise_gal = 1./(ng_tot/4./deg2_to_sr)
        print(shot_noise_gal)


    # Measure b_I*dI/dz in photometric bins    
    all_bI_I, all_bI_I_err, \
        all_ratio_mean, all_ratio_mean_err = [np.zeros((len(inst_list), len(zbinedges)-1)) for x in range(4)]
            
    if deconvolve:
        deconvolved_bI_I, deconvolved_bI_I_err = [np.zeros_like(all_bI_I) for x in range(2)]

        print('computing K_tilde')
        K_tilde = compute_K_tilde(zbinedges, zbinedges_fine, ng_tot)

        plot_map(K_tilde, figsize=(5, 5), title='K tilde')
    
    
    for inst in inst_list:
        
        print('zfine has shape', z_fine.shape)
        print('all_dNdz_b has shape', all_dNdz_b.shape)

        z_coarse_mid, bI_I, bI_I_err,\
            ratio_mean, ratio_mean_err = estimate_intensity_bias(
                                            z_fine, all_dNdz_b, zbinedges,
                                            full_cl_gI[inst-1], full_cl_gg[inst-1], lb,
                                            Cl_gI_err=full_cl_gI_err[inst-1], 
                                            Cl_gg_err=full_cl_gg_err[inst-1],
                                            bg_ng_fine_err=all_dNdz_b_err,
                                            ell_max=ell_max, 
                                            ell_min=ell_min,
                                            shot_noise_gal=shot_noise_gal, 
                                            avg_method=avg_method,
                                            ell_selection=ell_selection
                                        )

            
        all_bI_I[inst-1] = bI_I
        all_bI_I_err[inst-1] = bI_I_err
        
        all_ratio_mean[inst-1] = ratio_mean
        all_ratio_mean_err[inst-1] = ratio_mean_err
        
        if deconvolve:

            bI_I_true, bI_I_true_err,\
                 reconstructed = deconvolve_with_normalized_K(
                                                            all_bI_I[inst-1], 
                                                            all_bI_I_err[inst-1],
                                                            K_tilde, 
                                                            method=method,
                                                            verbose=True)

            # print('ng tot:', ng_tot)
            # bI_I_true, bI_I_true_err, reconstructed = deconvolve_intensity_bias_normalized(
            #     all_bI_I[inst-1], 
            #     all_bI_I_err[inst-1],
            #     z_true_centers,
            #     Wg_binned,
            #     ng_tot,
            #     ell_effective=ell_effective
            # )

            # results = deconvolve_comparison(all_bI_I[inst-1], all_bI_I_err[inst-1], z_true_centers, Wg_binned, alpha=alpha)
            # plot_deconvolution_comparison(results, figsize=(7, 6))
        
    
            # Compare original and reconstructed measurements
            # print("\nComparison of original vs reconstructed measurements:")
            # for i in range(len(all_bI_I[inst-1])):
            #     print(f"Bin {i}: Original={all_bI_I[inst-1][i]:.4f}, Reconstructed={reconstructed[i]:.4f}")

                
            # print('norm result:', bI_I_true)
            # print('compare result:', results['constrained']['solution'])
            
            # print('norm uncertainty:', bI_I_true_err)
            # print('compare unc:', results['constrained']['errors'])
                

            deconvolved_bI_I[inst-1] = bI_I_true
            deconvolved_bI_I_err[inst-1] = bI_I_true_err

            
    res = {'z_coarse_mid':z_coarse_mid, 'all_bI_I':all_bI_I, 'all_bI_I_err':all_bI_I_err, \
          'all_ratio_mean':all_ratio_mean, 'all_ratio_mean_err':all_ratio_mean_err}
    
    if deconvolve:
        res['deconvolved_bI_I'] = deconvolved_bI_I
        res['deconvolved_bI_I_err'] = deconvolved_bI_I_err
        res['K_tilde'] = K_tilde
        
    
    return res



def get_pmk_interpolator():

    pars = model.CAMBparams()
    pars.set_cosmology(H0=67.66, ombh2=0.02242, omch2=0.11933)
    pars.InitPower.set_params(As=2.105e-9, ns=0.9665)
    pars.set_matter_power(redshifts=[0, 0.2, 0.4, 0.6, 0.8, 1.0], kmax=10.0)
    pars.NonLinear = model.NonLinear_none
    # results = camb.get_results(pars)

    Pk_interp = get_matter_power_interpolator(pars, hubble_units=False, k_hunit=False)
    
    return Pk_interp

def compute_scattering_matrix_at_ell(ell, z_true, Wg_binned, Pk_interp_fn):
    """
    Compute scattering matrix K_ij at fixed ell using binned galaxy kernels and interpolated P(k, z).

    Parameters
    ----------
    ell : float
        Multipole value at which to compute the scattering matrix.
    z_true : (N_j,) array
        Central values of true redshift bins.
    Wg_binned : (N_i, N_j) array
        Binned galaxy kernels Wg_i^b(z_j) of shape (N_photo_bins, N_true_bins).
    Pkz_interp2d : function
        Function of (k, z) that returns the linear power spectrum P(k, z).
    cosmology : astropy.cosmology instance
        Cosmology for computing H(z) and chi(z).

    Returns
    -------
    K : (N_i, N_j) array
        Scattering matrix K_ij(ell), with i indexing photo-z bins and j true-z bins.
    """
    
    c = 3e5  # speed of light in km/s

    chi = cosmo.comoving_distance(z_true).value  # Mpc
    H = cosmo.H(z_true).to("km/s/Mpc").value     # H(z) in km/s/Mpc
    k = ell / chi                                     # shape (N_j,)
    
    def Pkz_interp2d(k, z):
        return Pk_interp_fn.P(z, k)
    

    # Evaluate P(k, z) along diagonal (ell / chi(z), z)
    Pkz = np.array([Pkz_interp2d(k_val, z_val) for k_val, z_val in zip(k, z_true)])

    # Pre-factor in the kernel definition
    prefactor = H / (c * chi**2)                     # shape (N_j,)

    # Apply prefactor and multiply by each row of Wg_binned
    K = Wg_binned * prefactor[np.newaxis, :] * Pkz[np.newaxis, :]  # shape (N_i, N_j)

    return K


def load_binned_Wg_ana(zbin_edges=None, zfine_edges=None, 
                   photoz_sigma=None, ntrue_funcs=None, bias_func=None):
    """
    Loads or generates Wg = b_g * dN/dz for each photo-z bin into shared true-z bins.

    Parameters
    ----------
    zbin_edges : array-like
        Photometric redshift bin edges.
    zfine_edges : array-like
        True redshift bin edges.
    photoz_sigma : float or None
        If None, read from file (default behavior).
        If float, use Gaussian photo-z error with sigma_z/(1+z) = photoz_sigma.
    ntrue_func : callable or None
        Function ntrue_func(z) giving the *true* redshift distribution (unnormalized).
        Only used if photoz_sigma is not None. Default: uniform in z.
    bias_func : callable or None
        Function bias_func(z) giving galaxy bias b_g(z). Default: 1.0.

    Returns
    -------
    Wg_binned : (N_photo_z, N_true_z) array
        Binned galaxy kernels.
    z_true : (N_true_z,) array
        Centers of the true-z bins.
    """
    if zbin_edges is None:
        zbin_edges = np.arange(0.0, 1.1, 0.1)
    if zfine_edges is None:
        zfine_edges = zbin_edges

    N_photo = len(zbin_edges) - 1
    N_true = len(zfine_edges) - 1
    z_true_centers = 0.5 * (zfine_edges[:-1] + zfine_edges[1:])
    dz_true = np.diff(zfine_edges)

    Wg_binned = np.zeros((N_photo, N_true))

    if photoz_sigma is None:
        # Default: read from file
        for i, (z0, z1) in enumerate(zip(zbin_edges[:-1], zbin_edges[1:])):
            fname = f"{config.ciber_basepath}data/ciber_x_gal/tomographer2_dNdzb/dNdzb_{np.round(z0,1)}_zphot_{np.round(z1,1)}.fit"
            with fits.open(fname) as hdul:
                data = hdul[1].data
                z_fine = data['z']
                dndz_b = data['dNdz_b']  # This is b_g * dN/dz

            for j in range(N_true):
                mask = (z_fine >= zfine_edges[j]) & (z_fine < zfine_edges[j+1])
                if np.any(mask):
                    dz = np.diff(z_fine)[0]
                    Wg_binned[i, j] = np.sum(dndz_b[mask]) * dz / dz_true[j]
    else:
        # Synthetic Gaussian photo-z model

        if bias_func is None:
            bias_func = lambda z: np.ones_like(z)  # b_g = 1

        for i, (zphot_min, zphot_max) in enumerate(zip(zbin_edges[:-1], zbin_edges[1:])):
            
            fname = f"{config.ciber_basepath}data/ciber_x_gal/tomographer2_dNdzb/dNdzb_{np.round(zphot_min,1)}_zphot_{np.round(zphot_max,1)}.fit"
            with fits.open(fname) as hdul:
                data = hdul[1].data
                z_fine = data['z']
                dndz_b = data['dNdz_b']  # This is b_g * dN/dz
            
                ntrue_func = lambda z: np.sum(dndz_b)
#             if ntrue_funcs is None:
#                 ntrue_func = lambda z: np.ones_like(z)  # uniform
#             else:
#                 ntrue_func = ntrue_funcs[i]
            
            for j, zc in enumerate(z_true_centers):
                # Integrate the Gaussian photo-z PDF over the photo-z bin
                sigma = photoz_sigma * (1.0 + zc)
                p_bin = norm.cdf(zphot_max, loc=zc, scale=sigma) - norm.cdf(zphot_min, loc=zc, scale=sigma)
                Wg_binned[i, j] = bias_func(zc) * ntrue_func(zc) * p_bin
                
                
#             Normalize each photo-z bin so sum over true z gives the total
            Wg_binned[i, :] *= dz_true

    return Wg_binned, z_true_centers

def load_binned_Wg(zbin_edges=None, zfine_edges=None):
    """
    Loads and rebins Wg = b_g * dN/dz for each photo-z bin into shared true-z bins.

    Parameters
    ----------
    config : object
        Configuration object with path `config.ciber_basepath`.
    zbin_edges : array-like, optional
        Photometric redshift bin edges. Default is np.arange(0, 1.1, 0.1).
    zfine_edges : array-like, optional
        True redshift bin edges. Default is same as zbin_edges.

    Returns
    -------
    Wg_binned : (N_photo_z, N_true_z) array
        Binned galaxy kernels.
    z_true : (N_true_z,) array
        Centers of the true-z bins.
    """
    if zbin_edges is None:
        zbin_edges = np.arange(0.0, 1.1, 0.1)
    if zfine_edges is None:
        zfine_edges = zbin_edges

    N_photo = len(zbin_edges) - 1
    N_true = len(zfine_edges) - 1
    Wg_binned = np.zeros((N_photo, N_true))

    for i, (z0, z1) in enumerate(zip(zbin_edges[:-1], zbin_edges[1:])):
        fname = f"{config.ciber_basepath}data/ciber_x_gal/tomographer2_dNdzb/dNdzb_{np.round(z0,1)}_zphot_{np.round(z1,1)}.fit"
        with fits.open(fname) as hdul:
            data = hdul[1].data
            z_fine = data['z']
            dndz_b = data['dNdz_b']  # This is b_g * dN/dz

        # Re-bin dndz_b into zfine_edges
        for j in range(N_true):
            mask = (z_fine >= zfine_edges[j]) & (z_fine < zfine_edges[j+1])
            if np.any(mask):
                dz = np.diff(z_fine)[0]  # assumes uniform spacing
                Wg_binned[i, j] = np.sum(dndz_b[mask]) * dz / (zfine_edges[j+1] - zfine_edges[j])

    z_true = 0.5 * (zfine_edges[:-1] + zfine_edges[1:])
    return Wg_binned, z_true

def compute_K_tilde_ana(zbinedges, zbinedges_fine, ng_tot, ell_effective = 1000., 
                       photoz_sigma=None, ntrue_funcs=None, bias_func=lambda z: 1.0):

    Pk_interp = get_pmk_interpolator()

    Wg_binned, z_true_centers = load_binned_Wg(zbinedges, zbinedges_fine)

    Wg_binned[Wg_binned < 0] = 0.

    # Compute scattering matrix at effective ell
    K_raw = compute_scattering_matrix_at_ell(ell_effective, z_true_centers, Wg_binned, Pk_interp)
    
    Wg_binned, z_true_centers = load_binned_Wg_ana(zbinedges, zbinedges_fine, 
                                                   photoz_sigma=photoz_sigma, ntrue_funcs=ntrue_funcs, bias_func=bias_func)
    
    
    plot_map(Wg_binned, figsize=(6, 6), title='Wg binned')
    
    Wg_normalized = np.zeros_like(Wg_binned)
    N_photo, N_true = K_raw.shape
    
    
    for i in range(N_photo):
        
        Wg_normalized[i, :] = Wg_binned[i, :] / ng_tot[i]

    denominators = np.zeros(N_photo)

    for i in range(N_photo):
        # For photo-z bin i, sum over all true-z bins j
        denominators[i] = np.sum(K_raw[i, :] * Wg_normalized[i, :])


    K_tilde = np.zeros_like(K_raw)

    for i in range(N_photo):
        if denominators[i] > 0:
            K_tilde[i, :] = K_raw[i, :] / denominators[i]
            
    
    return K_tilde

def compute_K_tilde(zbinedges, zbinedges_fine, ng_tot, ell_effective = 1000.):

    Pk_interp = get_pmk_interpolator()

    Wg_binned, z_true_centers = load_binned_Wg(zbinedges, zbinedges_fine)

    Wg_binned[Wg_binned < 0] = 0.

    # Compute scattering matrix at effective ell
    K_raw = compute_scattering_matrix_at_ell(ell_effective, z_true_centers, Wg_binned, Pk_interp)
    
    Wg_binned, z_true_centers = load_binned_Wg(zbinedges, zbinedges_fine)
    Wg_normalized = np.zeros_like(Wg_binned)
    N_photo, N_true = K_raw.shape
    
    for i in range(N_photo):
        Wg_normalized[i, :] = Wg_binned[i, :] / ng_tot[i]

    denominators = np.zeros(N_photo)

    for i in range(N_photo):
        # For photo-z bin i, sum over all true-z bins j
        denominators[i] = np.sum(K_raw[i, :] * Wg_normalized[i, :])


    K_tilde = np.zeros_like(K_raw)

    for i in range(N_photo):
        if denominators[i] > 0:
            K_tilde[i, :] = K_raw[i, :] / denominators[i]
            
    
    return K_tilde


def collect_ngtot(catname, inst=1, ifield_list=[4, 5, 6, 7, 8], zbinedges=None, 
                    headstr=None):

    if zbinedges is None:
        zbinedges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    ng_tot = np.zeros((len(zbinedges)-1))

    for zidx in range(len(zbinedges)-1):

        addstr = str(zbinedges[zidx])+'_z_'+str(zbinedges[zidx+1])

        if headstr is not None:
            addstr = headstr + '_' + addstr

        gal_counts, gal_fpath, noise_base_path = load_delta_g_maps(catname, inst, addstr)
        
        perf = [np.sum(gal_counts['ifield'+str(ifield)].data) for ifield in ifield_list]
        print(perf)
        ngsum = np.sum(perf)

        ng_tot[zidx] = ngsum / len(ifield_list)

    return ng_tot



def load_norm_bg_dndz(zbinedges, ng_tot=None, Atot=4.):
    ''' Loads tomographic redshift distribution'''
    cluster_bgdndz_fpath = config.ciber_basepath + 'data/ciber_x_gal/tomographer2_dNdzb/'
    ntom = len(zbinedges)-1
    
    for zidx in range(ntom):
        
        fpath = cluster_bgdndz_fpath+'dNdzb_'+str(zbinedges[zidx])+'_zphot_'+str(zbinedges[zidx+1])+'.fit'
        cluster_bgdndz = fits.open(fpath)[1].data
        
        z_fine, dNdz_b, dNdz_b_err = [cluster_bgdndz[key] for key in ['z', 'dNdz_b', 'dNdz_b_err']]

        if zidx==0:
            all_norm_dNdz_b, all_norm_dNdz_b_err = [np.zeros((ntom, len(z_fine))) for x in range(2)]
        
        # integrates 
        area = simps(dNdz_b, z_fine)

        if ng_tot is not None:
            print('Integrated within redshift bin int (dNdz_b)dz, N:', np.round(area, 1), int(ng_tot[zidx]))
            
            # fac = (ng_tot[zidx]/area)

            fac = (ng_tot[zidx]/area)

            print('fac = ', fac)
            all_norm_dNdz_b[zidx] = dNdz_b*fac
            all_norm_dNdz_b_err[zidx] = dNdz_b_err*fac        
            
        else:
            all_norm_dNdz_b[zidx] = dNdz_b * Atot / area
            all_norm_dNdz_b_err[zidx] = dNdz_b_err * Atot / area
            
    return z_fine, all_norm_dNdz_b, all_norm_dNdz_b_err


def compute_mean_ratio(ratio, ratio_err, method="weighted", ell_indices=None):
    """
    Compute mean ratio with uncertainty.
    
    method:
        "weighted"   - inverse variance weighted mean
        "unweighted" - simple mean
        "median"     - median with MAD-based error
        "select"     - weighted mean on selected bandpower indices
    ell_indices:
        list/array of integer indices (relative to the masked ell_array in the loop)
    """
    # Apply manual index selection if requested
    if method == "select":
        if ell_indices is None:
            raise ValueError("ell_indices must be provided for 'select' method.")
        ratio = ratio[ell_indices]
        ratio_err = ratio_err[ell_indices]
        method = "weighted"  # after selection, just do weighted mean

    if len(ratio) == 0:
        return np.nan, np.nan

    if method == "weighted":
        weights = 1.0 / ratio_err**2
        mean_ratio = np.average(ratio, weights=weights)
        mean_err = 1.0 / np.sqrt(np.sum(weights))

    elif method == "unweighted":
        mean_ratio = np.mean(ratio)
        mean_err = np.std(ratio, ddof=1) / np.sqrt(len(ratio))

    elif method == "median":
        mean_ratio = np.median(ratio)
        mad = np.median(np.abs(ratio - mean_ratio))
        mean_err = 1.4826 * mad / np.sqrt(len(ratio))  # MAD → σ

    else:
        raise ValueError(f"Unknown averaging method: {method}")

    return mean_ratio, mean_err


def estimate_intensity_bias(
    z_fine,
    bg_ng_fine,
    coarse_bin_edges,
    Cl_gI,
    Cl_gg,
    ell_array,
    Cl_gI_err,
    Cl_gg_err,
    bg_ng_fine_err,
    ell_max=2000, 
    ell_min=300,
    verbose=True,
    shot_noise_gal=None,
    plot=False, 
    avg_method="weighted", 
    ell_selection=None
):
    """
    Estimate b_I(z) * Ī(z) in coarse redshift bins using Cl_gI / Cl_gg ratio below ell_max,
    with error propagation and optional normalization of b_g * dN/dz.

    Returns:
        z_coarse_mid: bin centers
        bI_I: estimated intensity bias × mean intensity
        bI_I_err: uncertainty on estimate
    """

    N_coarse = len(coarse_bin_edges) - 1
    z_coarse_mid = 0.5 * (coarse_bin_edges[:-1] + coarse_bin_edges[1:])
    ell_array = np.asarray(ell_array)
    
    print('N_coarse = ', N_coarse)
    
    binned_bg_ng, binned_bg_ng_err, \
        ratio_mean, ratio_mean_err = [np.zeros((N_coarse)) for x in range(4)]

    ell_max_array = np.full(N_coarse, ell_max)


    for i in range(N_coarse):
        zmin, zmax = coarse_bin_edges[i], coarse_bin_edges[i+1]
        mask = (z_fine >= zmin) & (z_fine < zmax)
        if np.any(mask):
            z_slice = z_fine[mask]
            y_slice = bg_ng_fine[i,mask]
            
            binned_bg_ng[i] = np.trapz(y_slice, z_slice)
            err_slice = bg_ng_fine_err[i,mask]
            binned_bg_ng_err[i] = np.sqrt(np.trapz(err_slice**2, z_slice))

    for i in range(N_coarse):

        ell_mask = (ell_array >= ell_min) & (ell_array <= ell_max_array[i])

        # if verbose:
            # print('Cl_gI has shape', Cl_gI.shape)
            # print('ell mask:', ell_mask)

        if not np.any(ell_mask):
            # print('setting ratio means to nan..')
            ratio_mean[i] = np.nan
            ratio_mean_err[i] = np.nan
            continue

        ClgI = Cl_gI[i, ell_mask]
        Clgg = Cl_gg[i, ell_mask]

        # print('ClgI:', ClgI)
        # print('Clgg:', Clgg)

        if shot_noise_gal is not None:
            Clgg -= shot_noise_gal[i]

        ratio = ClgI / Clgg

        ClgI_err = Cl_gI_err[i, ell_mask]
        Clgg_err = Cl_gg_err[i, ell_mask]
        frac_err = np.sqrt((ClgI_err / ClgI)**2 + (Clgg_err / Clgg)**2)
        ratio_err = ratio * frac_err

        # print('ClgI_err:', ClgI_err)
        # print('Clgg_err:', Clgg_err)

        mean_ratio, mean_err = compute_mean_ratio(
                                                    ratio, ratio_err,
                                                    method=avg_method,
                                                    ell_indices=ell_selection)


        # print('mean ratio, mean err:', mean_ratio, mean_err)
            
        ratio_mean[i] = mean_ratio
        ratio_mean_err[i] = mean_err


    if verbose:
        print('ratio mean:', ratio_mean.shape)
        print('binned bg_ng:', binned_bg_ng.shape)
        print('bg ng:', binned_bg_ng)
    
    bI_I = ratio_mean * binned_bg_ng
    bI_I_err = np.abs(bI_I * np.sqrt(
        (ratio_mean_err / ratio_mean)**2 +
        (binned_bg_ng_err / binned_bg_ng)**2
    ))

    if verbose:
        print('bI_I:', bI_I)
        print('bI_I_err:', bI_I_err)

    mask_zero = (ratio_mean == 0) | (binned_bg_ng == 0)
    bI_I_err[mask_zero] = 0.0

    return z_coarse_mid, bI_I, bI_I_err, ratio_mean, ratio_mean_err



def deconvolve_with_normalized_K(bI_I_measured, bI_I_err, K_tilde, 
                                  method='nnls', verbose=True):
    """
    Deconvolve intensity measurements using the properly normalized K_tilde.
    
    Parameters:
    -----------
    bI_I_measured : array of shape (N_photo,)
        Measured values from ratio method (estimates of b_I × I)
    bI_I_err : array of shape (N_photo,)
        Errors on measured values
    K_tilde : array of shape (N_photo, N_true)
        Normalized scattering matrix from normalize_step_by_step()
    method : str
        'nnls' for non-negative least squares
        'lstsq' for standard least squares
        'inverse' for direct matrix inversion
    """
    
    N_photo, N_true = K_tilde.shape
    
    if verbose:
        print(f"Input measurements:")
        print(f"  bI_I_measured: {bI_I_measured}")
        print(f"  Range: [{bI_I_measured.min():.2e}, {bI_I_measured.max():.2e}]")
        print(f"\nK_tilde shape: {K_tilde.shape}")
        print(f"K_tilde range: [{K_tilde.min():.2e}, {K_tilde.max():.2e}]")
        print(f"K_tilde condition number: {np.linalg.cond(K_tilde):.2e}")
    
    # The linear system is: bI_I_measured = K_tilde @ bI_I_true
    # where bI_I_true is what we want to solve for
    
    if method == 'inverse':
        # Direct matrix inversion
        # This requires K_tilde to be square
        try:
            K_inv = np.linalg.inv(K_tilde)
            method_name = "Direct inverse"
        except np.linalg.LinAlgError:
            print("Warning: Matrix is singular, using pseudo-inverse")
            K_inv = np.linalg.pinv(K_tilde)
            method_name = "Pseudo-inverse"
        
        # Apply inverse
        bI_I_true = K_inv @ bI_I_measured
        
        # Compute residual for consistency with other methods
        residual = np.linalg.norm(K_tilde @ bI_I_true - bI_I_measured)
        
        if verbose:
            print(f"\n{method_name} solution:")
            print(f"  Residual norm: {residual:.2e}")
            print(f"  Solution: {bI_I_true}")
            print(f"  Range: [{bI_I_true.min():.2e}, {bI_I_true.max():.2e}]")
            if np.any(bI_I_true < 0):
                print(f"  Warning: {np.sum(bI_I_true < 0)} negative values")
    
    elif method == 'nnls':
        from scipy.optimize import nnls
        
        # Weight by measurement errors
        A_weighted = K_tilde / bI_I_err[:, np.newaxis]
        b_weighted = bI_I_measured / bI_I_err
        
        # Solve with non-negativity constraint
        bI_I_true, residual = nnls(A_weighted, b_weighted)
        
        if verbose:
            print(f"\nNNLS solution:")
            print(f"  Residual: {residual:.2e}")
            print(f"  Solution: {bI_I_true}")
            print(f"  Range: [{bI_I_true.min():.2e}, {bI_I_true.max():.2e}]")
            print(f"  Number of zero values: {np.sum(bI_I_true == 0)}")
    
    elif method == 'lstsq':
        # Weight by measurement errors
        A_weighted = K_tilde / bI_I_err[:, np.newaxis]
        b_weighted = bI_I_measured / bI_I_err
        
        # Standard least squares (can give negative values)
        bI_I_true, residuals, rank, s = np.linalg.lstsq(A_weighted, b_weighted, rcond=None)
        
        if verbose:
            print(f"\nLeast squares solution:")
            if len(residuals) > 0:
                print(f"  Residual: {residuals[0]:.2e}")
            print(f"  Matrix rank: {rank}/{min(K_tilde.shape)}")
            print(f"  Solution: {bI_I_true}")
            print(f"  Range: [{bI_I_true.min():.2e}, {bI_I_true.max():.2e}]")
            if np.any(bI_I_true < 0):
                print(f"  Warning: {np.sum(bI_I_true < 0)} negative values")
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'inverse', 'nnls', or 'lstsq'")
    
    # Error estimation via error propagation
    try:
        if method == 'inverse':
            # For direct inversion, propagate errors through K_inv
            # Error propagation: σ_out^2 = K_inv @ Σ @ K_inv^T
            cov_in = np.diag(bI_I_err**2)
            cov_out = K_inv @ cov_in @ K_inv.T
            bI_I_true_err = np.sqrt(np.diag(cov_out))
        else:
            # For optimization methods, use Fisher matrix approach
            W_inv = np.diag(1.0 / bI_I_err**2)
            cov_matrix = np.linalg.inv(K_tilde.T @ W_inv @ K_tilde)
            bI_I_true_err = np.sqrt(np.diag(cov_matrix))
        
        if verbose:
            print(f"\nError estimates:")
            print(f"  Errors: {bI_I_true_err}")
            print(f"  Relative errors: {bI_I_true_err / (np.abs(bI_I_true) + 1e-10)}")
    
    except np.linalg.LinAlgError:
        if verbose:
            print("\nWarning: Error calculation failed, using approximate errors")
        bI_I_true_err = np.abs(bI_I_true) * 0.3
    
    # Validation: reconstruct the measurements
    reconstructed = K_tilde @ bI_I_true
    residuals = reconstructed - bI_I_measured
    chi2 = np.sum((residuals / bI_I_err)**2)
    dof = max(1, len(bI_I_measured) - len(bI_I_true))
    
    if verbose:
        print(f"\nValidation:")
        print(f"  Reconstructed: {reconstructed}")
        print(f"  Original: {bI_I_measured}")
        print(f"  Residuals: {residuals}")
        print(f"  Residuals/error: {residuals/bI_I_err}")
        print(f"  χ²/dof = {chi2:.2f}/{dof} = {chi2/dof:.2f}")
    
    return bI_I_true, bI_I_true_err, reconstructed


# def deconvolve_intensity_bias_normalized(bI_I_measured, bI_I_err, z_true_centers, Wg_binned, 
#                                         Ng_tot_per_bin, ell_effective=1000, verbose=True, method='nnls'):
#     """
#     Deconvolution with proper galaxy count normalization.
    
#     Parameters:
#     -----------
#     bI_I_measured : array
#         Measured values from ratio method (should be ~b_I × intensity)
#     bI_I_err : array
#         Errors on measured values
#     z_true_centers : array
#         Centers of true redshift bins
#     Wg_binned : array of shape (N_photo, N_true)
#         Binned b_g × dN/dz from tomography
#     Ng_tot_per_bin : array of shape (N_photo,)
#         Total galaxy count in each photometric redshift bin
#     """
    
#     if verbose:
#         print('Input measurements:')
#         print(f'  bI_I_measured: {bI_I_measured}')
#         print(f'  Ng_tot per bin: {Ng_tot_per_bin}')
    
#     # Step 1: Fix negatives in Wg
#     Wg_fixed = np.maximum(Wg_binned, 0)
    
#     # Step 2: Normalize to get n_i(z) and effective bias
#     N_photo, N_true = Wg_fixed.shape
#     dz = z_true_centers[1] - z_true_centers[0] if len(z_true_centers) > 1 else 0.1
    
#     # For each photo-z bin, extract normalized distribution and effective bias
#     n_iz = np.zeros_like(Wg_fixed)  # Normalized probability distribution
#     b_eff = np.zeros(N_photo)       # Effective bias for each photo-z bin
    
#     for i in range(N_photo):
#         # Current integral of b_g × dN/dz
#         current_integral = np.sum(Wg_fixed[i, :]) * dz
        
#         if current_integral > 0:
#             # The effective bias-weighted galaxy count
#             # We know ∫ dN/dz dz = Ng_tot, so:
#             # ∫ b_g × dN/dz dz = Ng_tot × <b_g>
#             b_eff[i] = current_integral / Ng_tot_per_bin[i]
            
#             # Normalized distribution (integrates to 1)
#             n_iz[i, :] = Wg_fixed[i, :] / current_integral
#         else:
#             if verbose:
#                 print(f'Warning: Photo-z bin {i} has zero integral')
#             b_eff[i] = 1.0  # Default
    
#     if verbose:
#         print(f'\nNormalization results:')
#         print(f'  Effective bias per bin: {b_eff}')
#         print(f'  n(z) integrals (should be 1): {[np.sum(n_iz[i,:]*dz) for i in range(min(3,N_photo))]}')
    
#     # Step 3: Build scattering matrix with proper normalization
#     # We work with the bias-weighted probability: b_g(z) × n_i(z)
#     Wg_norm = np.zeros_like(Wg_fixed)
#     for i in range(N_photo):
#         Wg_norm[i, :] = b_eff[i] * n_iz[i, :]
    
#     # Get geometric factors
#     Pk_interp = get_pmk_interpolator()
#     K_no_Wg = compute_scattering_matrix_without_Wg(ell_effective, z_true_centers, Pk_interp)
    
#     # Build scattering matrix
#     K = np.zeros((N_photo, N_true))
#     for i in range(N_photo):
#         # K_ij represents how true-z bin j contributes to photo-z bin i
#         K[i, :] = K_no_Wg * Wg_norm[i, :]
    
#     # Normalize K by rows for numerical stability
#     K_row_sums = np.sum(K, axis=1)
#     K_normalized = K / K_row_sums[:, np.newaxis]
    
#     if verbose:
#         print(f'\nScattering matrix:')
#         print(f'  K condition number: {np.linalg.cond(K_normalized):.2e}')
#         print(f'  K_normalized range: [{K_normalized.min():.2e}, {K_normalized.max():.2e}]')
    
#     # Step 4: Adjust measurements
#     # The measurements are ratios × some normalization
#     # We need to account for the 1/Ng_tot factor in the ratio
#     measurements_adjusted = bI_I_measured * Ng_tot_per_bin / K_row_sums
#     errors_adjusted = bI_I_err * Ng_tot_per_bin / K_row_sums
    
#     if verbose:
#         print(f'\nAdjusted measurements:')
#         print(f'  Range: [{measurements_adjusted.min():.2e}, {measurements_adjusted.max():.2e}]')
#         print(f'  Compare to raw: [{bI_I_measured.min():.2e}, {bI_I_measured.max():.2e}]')
    
#     # Step 5: Solve the linear system
#     if method == 'nnls':
#         from scipy.optimize import nnls
        
#         A_weighted = K_normalized / errors_adjusted[:, np.newaxis]
#         b_weighted = measurements_adjusted / errors_adjusted
        
#         W_I_solution, residual = nnls(A_weighted, b_weighted)
        
#         if verbose:
#             print(f'\nNNLS solution:')
#             print(f'  Residual: {residual:.2e}')
            
#     else:  # lstsq
#         A_weighted = K_normalized / errors_adjusted[:, np.newaxis]
#         b_weighted = measurements_adjusted / errors_adjusted
        
#         W_I_solution, residuals, rank, s = np.linalg.lstsq(A_weighted, b_weighted, rcond=None)
        
#         if verbose:
#             print(f'\nLeast squares solution:')
#             if len(residuals) > 0:
#                 print(f'  Residual: {residuals[0]:.2e}')
    
#     # The solution W_I_solution is now b_I × (intensity kernel) in consistent units
#     bI_I_true = W_I_solution
    
#     if verbose:
#         print(f'\nFinal solution (b_I × intensity):')
#         print(f'  Values: {bI_I_true}')
#         print(f'  Range: [{bI_I_true.min():.2e}, {bI_I_true.max():.2e}]')
    
#     # Step 6: Error propagation
#     try:
#         Sigma_inv = np.diag(1.0 / errors_adjusted**2)
#         cov_matrix = np.linalg.inv(K_normalized.T @ Sigma_inv @ K_normalized)
#         bI_I_true_err = np.sqrt(np.diag(cov_matrix))
#     except np.linalg.LinAlgError:
#         if verbose:
#             print("Warning: Using approximate errors")
#         bI_I_true_err = np.abs(bI_I_true) * 0.3
    
#     # Step 7: Validation - reconstruct the measurements
#     reconstructed_norm = K_normalized @ bI_I_true
#     reconstructed = reconstructed_norm * K_row_sums / Ng_tot_per_bin
    
#     chi2 = np.sum(((reconstructed - bI_I_measured) / bI_I_err)**2)
#     dof = max(1, len(bI_I_measured) - len(bI_I_true))
    
#     if verbose:
#         print(f'\nValidation:')
#         print(f'  χ²/dof = {chi2:.2f}/{dof} = {chi2/dof:.2f}')
#         print(f'  Reconstructed: {reconstructed}')
#         print(f'  Original: {bI_I_measured}')
#         relative_diff = (reconstructed - bI_I_measured) / bI_I_measured
#         print(f'  Relative differences: {relative_diff}')
    
#     return bI_I_true, bI_I_true_err, reconstructed


def compute_scattering_matrix_without_Wg(ell, z_true, Pk_interp_fn):
    """
    Compute the scattering matrix prefactor WITHOUT the Wg term.
    This computes: H(z_j)/(c*chi^2(z_j)) * P(k=ell/chi(z_j), z_j)
    
    Parameters:
    -----------
    ell : float
        Multipole
    z_true : array
        True redshift bin centers
    Pk_interp_fn : function
        Power spectrum interpolator
        
    Returns:
    --------
    K_no_Wg : array of shape (N_true,)
        Geometric/cosmological factors for each true-z bin
    """
    c = 3e5  # speed of light in km/s
    
    chi = cosmo.comoving_distance(z_true).value  # Mpc
    H = cosmo.H(z_true).to("km/s/Mpc").value     # H(z) in km/s/Mpc
    k = ell / chi
    
    def Pkz_interp2d(k, z):
        return Pk_interp_fn.P(z, k)
    
    # Evaluate P(k, z) along diagonal
    Pkz = np.array([Pkz_interp2d(k_val, z_val) for k_val, z_val in zip(k, z_true)])
    
    # Pre-factor WITHOUT Wg
    K_no_Wg = H / (c * chi**2) * Pkz  # shape (N_true,)
    
    return K_no_Wg


# def deconvolve_intensity_bias_normalized(bI_I_measured, bI_I_err, z_true_centers, Wg_binned, 
#                                        ell_effective=1000, verbose=True):
    
#     """Deconvolution with proper normalization and physical constraints"""
    
#     if verbose:
#         print('bI_I_measured:', bI_I_measured)
#         print('bI_I_err:', bI_I_err)

#     # Get matter power spectrum interpolator
#     Pk_interp = get_pmk_interpolator()
    
#     # Compute scattering matrix at effective ell
#     K_raw = compute_scattering_matrix_at_ell(ell_effective, z_true_centers, Wg_binned, Pk_interp)

#     # Normalize the matrix
#     K, norm_factor = normalize_scattering_matrix(K_raw)
        
#     plot_map(K, figsize=(5, 5), xlabel='ztrue', ylabel='zphot')

    
#     # Define constrained optimization with non-negativity
#     def objective(x):
#         return np.sum(((K @ x - bI_I_measured) / bI_I_err)**2)
    
#     # Initial guess (all positive, matching magnitude of measurements)
#     x0 = np.abs(bI_I_measured).mean() * np.ones(K.shape[1])
    
#     if verbose:
#         print('x0:', x0)
    
#     # Bounds to enforce non-negativity
#     bounds = [(0, None) for _ in range(K.shape[1])]
    
#     # Solve constrained optimization
    
#     bI_I_true = np.dot(np.linalg.inv(K), bI_I_measured)
    
# #     result = minimize(objective, x0, bounds=bounds)
# #     bI_I_true = result.x
    
#     # Estimate errors (approximate)
#     J = K  # Jacobian is just K for linear problem
#     try:
#         cov = np.linalg.inv(J.T @ np.diag(1./bI_I_err**2) @ J)
#         bI_I_true_err = np.sqrt(np.diag(cov))
#     except np.linalg.LinAlgError:
#         print("Warning: Error calculation failed, using approximate errors")
#         bI_I_true_err = np.abs(bI_I_true) * 0.2  # Rough estimate
    
#     # Verify reconstruction matches measurements
#     reconstructed = K @ bI_I_true
#     chi2 = np.sum(((reconstructed - bI_I_measured)/bI_I_err)**2)
#     print(f"Reconstruction chi2: {chi2:.2f} for {len(bI_I_measured)} measurements")
        
#     return bI_I_true, bI_I_true_err, reconstructed



def deconvolve_comparison(bI_I_measured, bI_I_err, z_true_centers, Wg_binned, 
                          ell_effective=1000, alpha=0.001):
    """
    Compare direct matrix inversion vs. non-negative constrained optimization.
    
    Parameters:
    -----------
    bI_I_measured : array
        Measured intensity bias in photometric redshift bins
    bI_I_err : array
        Measurement errors
    z_true_centers : array
        Centers of true redshift bins
    Wg_binned : array
        Binned galaxy window functions
    ell_effective : float
        Single effective multipole value to use
    alpha : float
        Regularization strength for both methods
        
    Returns:
    --------
    Dictionary containing results from both methods
    """
#     import numpy as np
#     from scipy.optimize import minimize
    
    # Get matter power spectrum interpolator
    Pk_interp = get_pmk_interpolator()
    
    # Compute scattering matrix at effective ell
    K = compute_scattering_matrix_at_ell(ell_effective, z_true_centers, Wg_binned, Pk_interp)
    
    # Check condition number
    cond_num = np.linalg.cond(K)
    print(f"Condition number of scattering matrix: {cond_num:.1f}")
    
    # Normalize matrix if needed
    K_norm, norm_factor = normalize_scattering_matrix(K)
    
    # 1. DIRECT MATRIX INVERSION with Tikhonov regularization
    n_true_bins = K_norm.shape[1]
    
    # Create regularization matrix (second derivative for smoothness)
    D = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(n_true_bins-2, n_true_bins)).toarray()
    
    # Combined system with regularization
    A = np.vstack([K_norm, alpha * D])
    b = np.hstack([bI_I_measured, np.zeros(n_true_bins-2)])
    
    # Solve the regularized system
    direct_solution, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    # Error estimation for direct solution
    try:
        cov_direct = np.linalg.inv(K_norm.T @ np.diag(1/bI_I_err**2) @ K_norm + alpha**2 * D.T @ D)
        direct_err = np.sqrt(np.diag(cov_direct))
    except np.linalg.LinAlgError:
        print("Warning: Error calculation failed for direct solution")
        direct_err = np.abs(direct_solution) * 0.2  # Rough estimate
    
    # 2. NON-NEGATIVE CONSTRAINED OPTIMIZATION (your current approach)
    def objective(x):
        return np.sum(((K_norm @ x - bI_I_measured) / bI_I_err)**2) + alpha**2 * np.sum((D @ x)**2)
    
    # Initial guess (all positive)
    x0 = np.abs(bI_I_measured).mean() * np.ones(K_norm.shape[1])
    
    # Bounds to enforce non-negativity
    bounds = [(0, None) for _ in range(K_norm.shape[1])]
    
    # Solve constrained optimization
    result = minimize(objective, x0, bounds=bounds)
    constrained_solution = result.x
    
    # Error estimation for constrained solution (approximate)
    # This is more complex for constrained problems, so we use a rough approximation
    active_constraints = constrained_solution < 1e-6  # Almost zero
    free_indices = ~active_constraints
    
    if np.any(free_indices):
        # For unconstrained dimensions, use linearized covariance
        K_free = K_norm[:, free_indices]
        try:
            cov_free = np.linalg.inv(K_free.T @ np.diag(1/bI_I_err**2) @ K_free)
            constrained_err = np.zeros_like(constrained_solution)
            constrained_err[free_indices] = np.sqrt(np.diag(cov_free))
        except:
            constrained_err = np.abs(constrained_solution) * 0.2
    else:
        constrained_err = np.abs(constrained_solution) * 0.2
    
    # Forward modeling to check reconstruction
    direct_reconstructed = K_norm @ direct_solution
    constrained_reconstructed = K_norm @ constrained_solution
    
    # Calculate chi-squared
    chi2_direct = np.sum(((direct_reconstructed - bI_I_measured)/bI_I_err)**2)
    chi2_constrained = np.sum(((constrained_reconstructed - bI_I_measured)/bI_I_err)**2)
    
    print(f"Direct inversion χ²: {chi2_direct:.2f}")
    print(f"Constrained optimization χ²: {chi2_constrained:.2f}")
    
    # Package results
    return {
        'direct': {
            'solution': direct_solution,
            'errors': direct_err,
            'reconstructed': direct_reconstructed
        },
        'constrained': {
            'solution': constrained_solution, 
            'errors': constrained_err,
            'reconstructed': constrained_reconstructed
        },
        'z_true': z_true_centers,
        'original_measurements': bI_I_measured,
        'original_errors': bI_I_err
    }


def compute_scattering_matrices(z_fine, zbinedges_fine, zbinedges, ell_array):
    """
    Compute scattering matrices for relevant ells. (not using currently 8/20/25)
    
    Parameters:
    -----------
    z_fine : array
        Fine redshift grid
    zbinedges_fine : array
        Edges of fine redshift bins for true z
    zbinedges : array
        Edges of photometric redshift bins
    ell_array : array
        Multipole values to compute matrices for
        
    Returns:
    --------
    K_matrices : dict
        Dictionary mapping ells to scattering matrices
    z_true_centers : array
        Centers of true redshift bins
    """
    # Get matter power spectrum interpolator
    Pk_interp = get_pmk_interpolator()
    
    # Load binned galaxy window functions
    Wg_binned, z_true_centers = load_binned_Wg(zbinedges, zbinedges_fine)
    
    # Compute scattering matrix at each ell
    K_matrices = {}
    for ell in ell_array:
        K_matrices[ell] = compute_scattering_matrix_at_ell(ell, z_true_centers, Wg_binned, Pk_interp)
    
    return K_matrices, z_true_centers



import numpy as np
from scipy.stats import norm

def load_binned_Wg_ana(zbin_edges=None, zfine_edges=None, 
                   photoz_sigma=None, ntrue_funcs=None, bias_func=None):
    """
    Loads or generates Wg = b_g * dN/dz for each photo-z bin into shared true-z bins.

    Parameters
    ----------
    zbin_edges : array-like
        Photometric redshift bin edges.
    zfine_edges : array-like
        True redshift bin edges.
    photoz_sigma : float or None
        If None, read from file (default behavior).
        If float, use Gaussian photo-z error with sigma_z/(1+z) = photoz_sigma.
    ntrue_func : callable or None
        Function ntrue_func(z) giving the *true* redshift distribution (unnormalized).
        Only used if photoz_sigma is not None. Default: uniform in z.
    bias_func : callable or None
        Function bias_func(z) giving galaxy bias b_g(z). Default: 1.0.

    Returns
    -------
    Wg_binned : (N_photo_z, N_true_z) array
        Binned galaxy kernels.
    z_true : (N_true_z,) array
        Centers of the true-z bins.
    """
    if zbin_edges is None:
        zbin_edges = np.arange(0.0, 1.1, 0.1)
    if zfine_edges is None:
        zfine_edges = zbin_edges

    N_photo = len(zbin_edges) - 1
    N_true = len(zfine_edges) - 1
    z_true_centers = 0.5 * (zfine_edges[:-1] + zfine_edges[1:])
    dz_true = np.diff(zfine_edges)

    Wg_binned = np.zeros((N_photo, N_true))

    if photoz_sigma is None:
        # Default: read from file
        for i, (z0, z1) in enumerate(zip(zbin_edges[:-1], zbin_edges[1:])):
            fname = f"{config.ciber_basepath}data/ciber_x_gal/tomographer2_dNdzb/dNdzb_{np.round(z0,1)}_zphot_{np.round(z1,1)}.fit"
            with fits.open(fname) as hdul:
                data = hdul[1].data
                z_fine = data['z']
                dndz_b = data['dNdz_b']  # This is b_g * dN/dz

            for j in range(N_true):
                mask = (z_fine >= zfine_edges[j]) & (z_fine < zfine_edges[j+1])
                if np.any(mask):
                    dz = np.diff(z_fine)[0]
                    Wg_binned[i, j] = np.sum(dndz_b[mask]) * dz / dz_true[j]
    else:
        # Synthetic Gaussian photo-z model

        if bias_func is None:
            bias_func = lambda z: np.ones_like(z)  # b_g = 1

        for i, (zphot_min, zphot_max) in enumerate(zip(zbin_edges[:-1], zbin_edges[1:])):
            
            fname = f"{config.ciber_basepath}data/ciber_x_gal/tomographer2_dNdzb/dNdzb_{np.round(zphot_min,1)}_zphot_{np.round(zphot_max,1)}.fit"
            with fits.open(fname) as hdul:
                data = hdul[1].data
                z_fine = data['z']
                dndz_b = data['dNdz_b']  # This is b_g * dN/dz
            
                ntrue_func = lambda z: np.sum(dndz_b)
#             if ntrue_funcs is None:
#                 ntrue_func = lambda z: np.ones_like(z)  # uniform
#             else:
#                 ntrue_func = ntrue_funcs[i]
            
            for j, zc in enumerate(z_true_centers):
                # Integrate the Gaussian photo-z PDF over the photo-z bin
                sigma = photoz_sigma * (1.0 + zc)
                p_bin = norm.cdf(zphot_max, loc=zc, scale=sigma) - norm.cdf(zphot_min, loc=zc, scale=sigma)
                Wg_binned[i, j] = bias_func(zc) * ntrue_func(zc) * p_bin
                
                
#             Normalize each photo-z bin so sum over true z gives the total
            Wg_binned[i, :] *= dz_true

    return Wg_binned, z_true_centers


def compute_K_tilde_ana(zbinedges, zbinedges_fine, ng_tot, ell_effective = 1000., 
                       photoz_sigma=None, ntrue_funcs=None, bias_func=lambda z: 1.0):

    Pk_interp = get_pmk_interpolator()

    Wg_binned, z_true_centers = load_binned_Wg(zbinedges, zbinedges_fine)

    Wg_binned[Wg_binned < 0] = 0.

    # Compute scattering matrix at effective ell
    K_raw = compute_scattering_matrix_at_ell(ell_effective, z_true_centers, Wg_binned, Pk_interp)
    
    Wg_binned, z_true_centers = load_binned_Wg_ana(zbinedges, zbinedges_fine, 
                                                   photoz_sigma=photoz_sigma, ntrue_funcs=ntrue_funcs, bias_func=bias_func)
    
    
    plot_map(Wg_binned, figsize=(6, 6), title='Wg binned')
    
    Wg_normalized = np.zeros_like(Wg_binned)
    N_photo, N_true = K_raw.shape
    
    
    for i in range(N_photo):
        
        Wg_normalized[i, :] = Wg_binned[i, :] / ng_tot[i]

    denominators = np.zeros(N_photo)

    for i in range(N_photo):
        # For photo-z bin i, sum over all true-z bins j
        denominators[i] = np.sum(K_raw[i, :] * Wg_normalized[i, :])


    K_tilde = np.zeros_like(K_raw)

    for i in range(N_photo):
        if denominators[i] > 0:
            K_tilde[i, :] = K_raw[i, :] / denominators[i]
            
    
    return K_tilde


