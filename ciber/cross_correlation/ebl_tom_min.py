import numpy as np
from astropy.io import fits
from camb import get_matter_power_interpolator, model
import camb
from astropy.cosmology import Planck18 as cosmo
from scipy.optimize import minimize

from ciber.io.ciber_data_utils import *
from ciber.plotting.galaxy_plots import *
import config
from scipy.optimize import minimize
from scipy import sparse



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


def normalize_scattering_matrix(K):
    """
    Normalize the scattering matrix to make it physically meaningful.
    
    Parameters:
    -----------
    K : ndarray
        Original scattering matrix
        
    Returns:
    --------
    K_norm : ndarray
        Normalized scattering matrix
    norm_factor : float
        Normalization factor applied (to scale results back if needed)
    """
    # First, handle any negative values (set to zero or small positive)
    K_fixed = np.maximum(K, 0)
    
    # Column-normalize (each true z bin contributes total probability of 1 across photo-z)
    # This preserves the physical relationship between true and observed distributions
    col_sums = K_fixed.sum(axis=0)
    
    # Avoid division by zero
    col_sums[col_sums == 0] = 1.0
    
    # Store the normalization factor (mean of column sums)
    norm_factor = np.mean(col_sums)
    
    # Normalize columns
    K_norm = K_fixed / col_sums[np.newaxis, :]
    
    print(f"Applied normalization factor: {norm_factor:.4e}")
    print(f"New matrix sum: {K_norm.sum():.4f}")
    print(f"New matrix min/max: {K_norm.min():.4e}, {K_norm.max():.4e}")
    
    return K_norm, norm_factor

def collect_ngtot(catname, inst=1, ifield_list=[4, 5, 6, 7, 8], zbinedges=None):

    if zbinedges is None:
        zbinedges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    ng_tot = np.zeros((len(zbinedges)-1))

    for zidx in range(len(zbinedges)-1):

        addstr = str(zbinedges[zidx])+'_z_'+str(zbinedges[zidx+1])

        gal_counts, noise_base_path = load_delta_g_maps(catname, inst, addstr)
        
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

        if verbose:
            print('Cl_gI has shape', Cl_gI.shape)
            print('ell mask:', ell_mask)

        if not np.any(ell_mask):
            print('setting ratio means to nan..')
            ratio_mean[i] = np.nan
            ratio_mean_err[i] = np.nan
            continue

        ClgI = Cl_gI[i, ell_mask]
        Clgg = Cl_gg[i, ell_mask]

        print('ClgI:', ClgI)
        print('Clgg:', Clgg)

        if shot_noise_gal is not None:
            Clgg -= shot_noise_gal[i]

        ratio = ClgI / Clgg

        ClgI_err = Cl_gI_err[i, ell_mask]
        Clgg_err = Cl_gg_err[i, ell_mask]
        frac_err = np.sqrt((ClgI_err / ClgI)**2 + (Clgg_err / Clgg)**2)
        ratio_err = ratio * frac_err

        print('ClgI_err:', ClgI_err)
        print('Clgg_err:', Clgg_err)

        mean_ratio, mean_err = compute_mean_ratio(
                                                    ratio, ratio_err,
                                                    method=avg_method,
                                                    ell_indices=ell_selection)


        print('mean ratio, mean err:', mean_ratio, mean_err)
            
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


def deconvolve_intensity_bias_normalized(bI_I_measured, bI_I_err, z_true_centers, Wg_binned, 
                                       ell_effective=1000, verbose=True):
    
    """Deconvolution with proper normalization and physical constraints"""
    
    if verbose:
        print('bI_I_measured:', bI_I_measured)
        print('bI_I_err:', bI_I_err)

    # Get matter power spectrum interpolator
    Pk_interp = get_pmk_interpolator()
    
    # Compute scattering matrix at effective ell
    K_raw = compute_scattering_matrix_at_ell(ell_effective, z_true_centers, Wg_binned, Pk_interp)

    # Normalize the matrix
    K, norm_factor = normalize_scattering_matrix(K_raw)
        
    plot_map(K, figsize=(5, 5), xlabel='ztrue', ylabel='zphot')

    
    # Define constrained optimization with non-negativity
    def objective(x):
        return np.sum(((K @ x - bI_I_measured) / bI_I_err)**2)
    
    # Initial guess (all positive, matching magnitude of measurements)
    x0 = np.abs(bI_I_measured).mean() * np.ones(K.shape[1])
    
    if verbose:
        print('x0:', x0)
    
    # Bounds to enforce non-negativity
    bounds = [(0, None) for _ in range(K.shape[1])]
    
    # Solve constrained optimization
    
    bI_I_true = np.dot(np.linalg.inv(K), bI_I_measured)
    
#     result = minimize(objective, x0, bounds=bounds)
#     bI_I_true = result.x
    
    # Estimate errors (approximate)
    J = K  # Jacobian is just K for linear problem
    try:
        cov = np.linalg.inv(J.T @ np.diag(1./bI_I_err**2) @ J)
        bI_I_true_err = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        print("Warning: Error calculation failed, using approximate errors")
        bI_I_true_err = np.abs(bI_I_true) * 0.2  # Rough estimate
    
    # Verify reconstruction matches measurements
    reconstructed = K @ bI_I_true
    chi2 = np.sum(((reconstructed - bI_I_measured)/bI_I_err)**2)
    print(f"Reconstruction chi2: {chi2:.2f} for {len(bI_I_measured)} measurements")
        
    return bI_I_true, bI_I_true_err, reconstructed
