import numpy as np
import config
from astropy.io import fits
import pyfftw
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from ciber.plotting.plotting_fns import plot_map

from ciber.core.powerspec_pipeline import *
from ciber.plotting.gal_plotting_fns import *
from ciber.io.catalog_utils import *


def completeness_model(m, m_lim=25.0, sigma_m=0.5):
    
    zsm = (m - m_lim)/sigma_m
    
    c = 0.5*(1 - scipy.special.erf(zsm))
    
    return c

def return_default_gal_cat_dict():
    
    gal_cat_dict = dict({})
    
    # file paths
    gal_cat_dict['catalog_basepath'] = config.ciber_basepath+'data/catalogs/'
    
    # CIBER parameters
    gal_cat_dict['ifield_list'] = [4, 5, 6, 7, 8]
    gal_cat_dict['ciber_dimx'] = 1024
    
    # unWISE
    gal_cat_dict['rgb_mode'] = None
    gal_cat_dict['w1_mag_max'] = None
    gal_cat_dict['w1_mag_min'] = None
    gal_cat_dict['wise_bands'] = ['mag_W1', 'mag_W2']
    
    # DECaLS
    gal_cat_dict['decals_gal'] = False
    gal_cat_dict['decals_redshift_key'] = 'z_phot_mean'

    
    # HSC
    gal_cat_dict['hsc_mag_max'] = None
    gal_cat_dict['hsc_mag_min'] = None
    gal_cat_dict['hsc_bands'] = ['g_cmodel_mag', 'r_cmodel_mag', 'z_cmodel_mag', 'i_cmodel_mag', 'y_cmodel_mag']
    gal_cat_dict['hsc_redshift_key'] = 'z_phot_mean'

    # photo-z parameters (when available)
    
    gal_cat_dict['zmin'] = None
    gal_cat_dict['zmax'] = None
    
    return gal_cat_dict


class cat_select():
    
    def __init__(self, gal_dict=None, which_hsc_band='i'):
        
        if gal_dict is not None:
            print('Loading galaxy parameter dict into cat_select()')
            self.gal_dict = gal_dict
            
        self.which_hsc_band = which_hsc_band
                
    
    def decals_mask(self):

        if self.gal_dict['decals_gal']:
            mask = (self.cat_type != "b'PSF'")
        else:
            mask = (self.cat_type == "b'PSF'")
        return mask
    
    
    def wise_cat_mask(self):
        
        mask = np.ones_like(self.cat_x).astype(int)
        
        if self.gal_dict['rgb_mode'] is not None:
            print('RGB mode is ', self.gal_dict['rgb_mode'])
            mask *= wise_rgb_cuts(self.cat_W1, self.cat_W2, self.gal_dict['rgb_mode'])

        if self.gal_dict['w1_mag_max'] is not None:
            print('Cutting WISE sources with W1 > '+str(self.gal_dict['w1_mag_max']))
            mask *= (self.cat_W1 < self.gal_dict['w1_mag_max'])
            
        if self.gal_dict['w1_mag_min'] is not None:
            print('Cutting WISE sources with W1 < '+str(self.gal_dict['w1_mag_min']))
            mask *= (self.cat_W1 > self.gal_dict['w1_mag_min'])
            

        return mask
        
    
    def footprint_mask(self):
        # in FOV
        mask = (self.cat_x > 0)*(self.cat_x < self.gal_dict['ciber_dimx'])*(self.cat_y > 0)*(self.cat_y < self.gal_dict['ciber_dimx'])

        # in redshift range if specified
        if 'redshift' in self.cat_stack_labs:
            if self.gal_dict['zmin'] is not None:
                print('Removing z < '+str(self.gal_dict['zmin'])+' sources')
                mask *= (self.cat_redshift > self.gal_dict['zmin'])
            if self.gal_dict['zmax'] is not None:
                print('Removing z > '+str(self.gal_dict['zmax'])+' sources')
                mask *= (self.cat_redshift < self.gal_dict['zmax'])

        return mask.astype(int)
    
    
    def gaia_cat_mask(self):
        
        mask = np.ones_like(self.cat_x).astype(int)
        
        return mask
    
    def hsc_cat_mask(self):
        
        mask = np.ones_like(self.cat_x).astype(int)
                    
        if self.gal_dict['hsc_mag_max'] is not None:
            if self.which_hsc_band=='i':
                print('Removing sources with i > '+str(self.gal_dict['hsc_mag_max']))

                mask *= (self.cat_i < self.gal_dict['hsc_mag_max'])
            elif self.which_hsc_band=='z':
                
                print('Removing sources with zAB > '+str(self.gal_dict['hsc_mag_max']))

                mask *= (self.cat_z < self.gal_dict['hsc_mag_max'])
                
            
        if self.gal_dict['hsc_mag_min'] is not None:
            if self.which_hsc_band=='i':

                print('Removing sources with i < '+str(self.gal_dict['hsc_mag_min']))
                mask *= (self.cat_i > self.gal_dict['hsc_mag_min'])
            elif self.which_hsc_band=='z':
                print('Removing sources with zAB < '+str(self.gal_dict['hsc_mag_min']))
                mask *= (self.cat_z > self.gal_dict['hsc_mag_min'])
        
        return mask
    
    
    def apply_cat_select(self, catname):
        
        mask = self.footprint_mask()       
        print(mask)

        cat_mask = None
        if catname=='DECaLS':
            cat_mask = self.decals_mask()
        
        elif catname=='WISE':
            cat_mask = self.wise_cat_mask()    
            
        elif catname=='HSC':
            cat_mask  = self.hsc_cat_mask()
            
        elif catname=='gaia':
            cat_mask = self.gaia_cat_mask()
            
        if cat_mask is not None:
            mask *= cat_mask
        
        return mask
            
            
    def load_cat(self, fpath, ciber_inst, catname):
        
        cat_df = pd.read_csv(fpath)
        
        print(cat_df.keys())
        
        self.cat_x, self.cat_y = np.array(cat_df['x'+str(ciber_inst)]), np.array(cat_df['y'+str(ciber_inst)])
        
        self.cat_stack = [self.cat_x, self.cat_y]
        self.cat_stack_labs = ['x', 'y']
            
        if catname=='LS':
            self.cat_redshift, self.cat_type = np.array(cat_df['z_phot_mean']), np.array(cat_df['type'])
            self.cat_stack.extend([self.cat_redshift, self.cat_type])
            self.cat_stack_labs.extend(['redshift', 'type'])
            
        elif catname=='HSC':
            self.cat_redshift = np.array(cat_df['photoz_mean'])
            self.cat_redshift_unc = np.array(cat_df['photoz_std_mean'])
            self.cat_g, self.cat_r, self.cat_i, self.cat_z, self.cat_ymag = [np.array(cat_df[bandstr]) for bandstr in ['g_cmodel_mag', 'r_cmodel_mag', 'i_cmodel_mag', 'z_cmodel_mag', 'y_cmodel_mag']]
            
            self.cat_stack.extend([self.cat_redshift, self.cat_g, self.cat_r, self.cat_i, self.cat_z, self.cat_ymag])
            self.cat_stack_labs.extend(['redshift', 'mag_g', 'mag_r', 'mag_i', 'mag_z', 'mag_y'])

        elif catname=='WISE':
            self.cat_W1, self.cat_W2 = np.array(cat_df['mag_W1']), np.array(cat_df['mag_W2'])
            
            self.cat_stack.extend([self.cat_W1, self.cat_W2])
            self.cat_stack_labs.extend(['mag_W1', 'mag_W2'])
            
        print('cat labels:', self.cat_stack_labs)
        




def gen_wget_unWISE_command(tile_list, basedir='raw_neo8/', \
                           base_url='https://portal.nersc.gov/project/cosmo/data/unwise/neo8/unwise-catalog/', \
                           save_fname=None):
    
    ''' Generates a shell script to grab catalogs from WISE tiles indicated by tile_list. '''
    all_commands = []
    
    for tile in tile_list:
        command = 'wget -I '+basedir+' '+base_url+'/objcat/'+tile+'.cat.fits'
        all_commands.append(command) 
        
#         for band in [1, 2]:
#             command = 'wget -I '+basedir+' '+base_url+'/cat/'+tile+'.'+str(band)+'.cat.fits'
#             all_commands.append(command)

    if save_fname is not None:
        filename = basedir+save_fname
        print('Writing wget commands to ', filename)
        with open(filename, 'w') as f:
            for command in all_commands:
                f.write(command + "\n")
        os.chmod(filename, 0o755)    
        
    return all_commands

def get_count_field(x_all, y_all, imdim=1024, smooth=False, smooth_sig=20, mean_sub=False):
    
    H, xedge, yedge = np.histogram2d(x_all, y_all, [np.arange(imdim+1)-0.5, np.arange(imdim+1)-0.5])
    
    if smooth:
        cf = gaussian_filter(H.transpose(), sigma=smooth_sig)
    else:
        cf = H
        
    if mean_sub:
        cf -= np.mean(cf)
    
    return cf


def compute_weighted_cl(all_cl, all_clerr):
    
    all_cl = np.array(all_cl)
    all_clerr = np.array(all_clerr)
    
    variance = all_clerr**2
    
    weights = 1./variance
    
    cl_sumweights = np.sum(weights, axis=0)
    
    weighted_variance = 1./cl_sumweights
    
    field_averaged_std = np.sqrt(weighted_variance)
    
    field_averaged_cl = np.nansum(weights*all_cl, axis=0)/cl_sumweights
    
    return field_averaged_cl, field_averaged_std


def compute_effective_bias_ls(zbinedges, dz=0.1,
                               dNdzb_basepath=None, cat_basepath=None,
                               survey_area_deg2=19400.0):
    """Compute effective galaxy bias b_g for each broad redshift bin from
    tomographer b_g*dN/dz estimates.

    The tomographer dNdz_b column stores b_g(z) * dN/dz(z) with units of
    deg^-2 per unit redshift (i.e. galaxies per deg^2 per dz), normalized by
    the LS DR8 survey area (19400 deg^2).

    For each dz=0.1 photo-z slice the per-slice effective bias is:

        b_eff_i = integral_allz(b*dN/dz dz) / (N_i / survey_area_deg2)

    where the integral is computed over the full fine-z grid in the file
    (which spans 0–2 to capture photo-z scatter) and N_i is the galaxy count
    from the catalog for that slice.

    For a broad bin covering multiple slices the bias is weighted by
    N_i / survey_area_deg2 (i.e. by the surface density of galaxies):

        b_eff = sum_i( integral_i ) / sum_i( N_i / survey_area_deg2 )

    Parameters
    ----------
    zbinedges : array_like
        Edges of the broad redshift bins (e.g. [0.0, 0.2, 0.4, ...])
    dz : float, optional
        Width of fine slices (default 0.1, matching the .fit files)
    dNdzb_basepath : str, optional
        Directory containing dNdzb_*.fit files. Defaults to config path.
    cat_basepath : str, optional
        Directory containing LS_Dr8_z22_*.fits catalogs. Defaults to config path.
    survey_area_deg2 : float, optional
        Survey area in deg^2 used to normalize dNdz_b (default 19400.0 for LS DR8)

    Returns
    -------
    b_eff : ndarray
        Effective bias for each broad bin, shape [len(zbinedges)-1]
    b_eff_err : ndarray
        Propagated uncertainty on b_eff, shape [len(zbinedges)-1]
    z_centers : ndarray
        Bin center redshifts
    """
    if dNdzb_basepath is None:
        dNdzb_basepath = config.ciber_basepath + 'data/ciber_x_gal/tomographer2_dNdzb/'
    if cat_basepath is None:
        cat_basepath = config.ciber_basepath + 'data/ciber_x_gal/data_catalogs/'

    # Build list of fine slices covering the full range
    z_fine_edges = np.round(np.arange(zbinedges[0], zbinedges[-1] + dz/2, dz), 1)

    # For each fine slice: integral of b*dN/dz dz (in deg^-2) and N_allsky
    slice_integral     = {}   # z0 -> sum(dNdz_b * dz)  [deg^-2]
    slice_integral_var = {}   # z0 -> sum((dNdz_b_err * dz)^2)
    slice_N            = {}   # z0 -> all-sky galaxy count

    for zidx in range(len(z_fine_edges) - 1):
        z0 = z_fine_edges[zidx]
        z1 = z_fine_edges[zidx + 1]

        dNdzb_fpath = dNdzb_basepath + f'dNdzb_{z0:.1f}_zphot_{z1:.1f}.fit'
        try:
            hdul = fits.open(dNdzb_fpath)
            data = hdul[1].data
            hdul.close()
            slice_integral[z0]     = np.sum(data['dNdz_b'] * data['dz'])
            slice_integral_var[z0] = np.sum((data['dNdz_b_err'] * data['dz'])**2)
        except FileNotFoundError:
            print(f'[effective_bias] Missing dNdzb file for z={z0:.1f}-{z1:.1f}, skipping')
            slice_integral[z0]     = np.nan
            slice_integral_var[z0] = np.nan

        cat_fpath = cat_basepath + f'LS_Dr8_z22_{z0:.1f}_zphot_{z1:.1f}.fits'
        try:
            hdul = fits.open(cat_fpath)
            slice_N[z0] = len(hdul[1].data)
            hdul.close()
        except FileNotFoundError:
            print(f'[effective_bias] Missing catalog for z={z0:.1f}-{z1:.1f}, skipping')
            slice_N[z0] = 0

    # Compute effective bias per broad bin
    n_bins = len(zbinedges) - 1
    b_eff     = np.zeros(n_bins)
    b_eff_err = np.zeros(n_bins)
    z_centers = 0.5 * (zbinedges[:-1] + zbinedges[1:])

    for bidx in range(n_bins):
        z0_bin = zbinedges[bidx]
        z1_bin = zbinedges[bidx + 1]

        slices_in_bin = [z for z in slice_integral
                         if z >= z0_bin - dz/10 and z < z1_bin - dz/10]

        # numerator = sum_i integral_i  [deg^-2]
        # denominator = sum_i N_i / fullsky_deg2  [deg^-2]
        # b_eff = numerator / denominator  (dimensionless)
        numerator     = 0.0
        numerator_var = 0.0
        denominator   = 0.0

        for z0s in slices_in_bin:
            integ = slice_integral[z0s]
            ivar  = slice_integral_var[z0s]
            N     = slice_N[z0s]

            if np.isnan(integ) or N == 0:
                continue

            numerator     += integ
            numerator_var += ivar
            denominator   += N / survey_area_deg2

        if denominator > 0:
            b_eff[bidx]     = numerator / denominator
            b_eff_err[bidx] = np.sqrt(numerator_var) / denominator
        else:
            b_eff[bidx]     = np.nan
            b_eff_err[bidx] = np.nan

    return b_eff, b_eff_err, z_centers


def save_gal_density(inst, ifield_list, gal_densities, catname, basepath=None, addstr=None):
    
    if basepath is None:
        basepath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/gal_density/'
        
        
    prim = fits.PrimaryHDU()
    
    hdul = [prim]
    
    for fieldidx, ifield in enumerate(ifield_list):
        imhdu = fits.ImageHDU(gal_densities[fieldidx], name='ifield'+str(ifield))
        
        hdul.append(imhdu)
    
    
    hdul = fits.HDUList(hdul)
    
    save_fpath = basepath+catname+'/gal_density_'+catname+'_TM'+str(inst)

    if addstr is not None:
        save_fpath += '_'+addstr
    
    save_fpath += '.fits'
    print('Saving to ', save_fpath)
    hdul.writeto(save_fpath, overwrite=True)
    
    return save_fpath


def combine_gal_density_zbins(inst_list, ifield_list, zbinedges, catname='LS', mode='data',
                               basepath=None, field_size=2.0, plot=False):
    """Combine galaxy density maps across redshift slices into a single map.

    Loads density maps for individual redshift bins and combines them (e.g., to get
    z < 1.0 total counts). Works with both standard (2×2°) and larger footprints.

    Parameters
    ----------
    inst_list : list or int
        CIBER instrument(s) (1, 2, or [1, 2])
    ifield_list : list
        List of CIBER field indices
    zbinedges : array_like
        Redshift bin edges to combine
    catname : str, optional
        Catalog name (default 'LS')
    mode : str, optional
        'data' or 'random' catalog mode
    basepath : str, optional
        Base path for loading density maps. If None, uses default.
    field_size : float, optional
        Field size in degrees (2.0 for standard 2×2°, 4.0 for 4×4°, etc.)
    plot : bool, optional
        Whether to plot combined maps

    Returns
    -------
    combined_counts : dict
        Dictionary with keys 'inst{inst}' containing combined count arrays
        Shape: [len(ifield_list), imdim, imdim]
    """
    if isinstance(inst_list, int):
        inst_list = [inst_list]

    if basepath is None:
        basepath = config.ciber_basepath+'data/fluctuation_data/'

    # Determine output map size based on field_size
    base_imdim = 1024
    imdim = int(base_imdim * (field_size / 2.0))

    combined_counts = {}

    for inst in inst_list:
        inst_basepath = basepath + f'TM{inst}/gal_density/'
        tot_counts = np.zeros((len(ifield_list), imdim, imdim))

        for zidx, z0 in enumerate(zbinedges[:-1]):
            z1 = zbinedges[zidx+1]
            addstr = str(np.round(z0, 1))+'_z_'+str(np.round(z1, 1))

            if mode == 'random':
                addstr += '_random'

            # Add field size to filename if not standard 2×2°
            if field_size != 2.0:
                addstr += f'_{field_size:.1f}deg'

            fpath = inst_basepath + f'{catname}/gal_density_{catname}_TM{inst}_{addstr}.fits'

            print(f'[combine] Loading {inst} z={z0:.1f}-{z1:.1f} from {fpath}')
            hdul = fits.open(fpath)

            for fieldidx, ifield in enumerate(ifield_list):
                tot_counts[fieldidx] += hdul[f'ifield{ifield}'].data

            hdul.close()

        addstr = 'zlt1.0_random' if mode == 'random' else 'zlt1.0'
        save_gal_density(inst, ifield_list, tot_counts, catname, basepath=inst_basepath, addstr=addstr)

        if plot:
            for fieldidx, ifield in enumerate(ifield_list):
                print(f'[combine] TM{inst} ifield {ifield}: {np.sum(tot_counts[fieldidx])} total counts')
                plot_map(tot_counts[fieldidx], title=f'{catname} z<{zbinedges[-1]} ifield {ifield} TM{inst}',
                         figsize=(6, 6))

    return tot_counts


def compute_gal_auto_spectrum_large(inst, ifield_list, zbinedges, field_size=4.0,
                                     catname='LS', subtract_randoms=True,
                                     save=False, plot=False):
    """Compute galaxy auto-spectrum from larger footprint density maps.

    Computes power spectra from the density maps extracted with
    preprocess_ls_density_maps_large() or combined with combine_gal_density_zbins().
    Processes maps similarly to ciber_gal_cross(): subtracts randoms (scaled to match
    data), normalizes to galaxy overdensity, and computes 1D power spectrum.

    Parameters
    ----------
    inst : int
        CIBER instrument (1 or 2)
    ifield_list : list
        List of CIBER field indices
    zbinedges : array_like
        Redshift bin edges
    field_size : float, optional
        Field size in degrees (default 4.0 for 4×4°)
    catname : str, optional
        Catalog name (default 'LS')
    subtract_randoms : bool, optional
        Whether to subtract random catalog contribution (default True)
    save : bool, optional
        Whether to save power spectra to .npz file
    plot : bool, optional
        Whether to plot power spectra and maps

    Returns
    -------
    dict
        Dictionary containing:
        - 'lb': multipole bin centers
        - 'all_cl_gal': galaxy auto C_ell [n_field, n_ell]
        - 'all_clerr_gal': galaxy auto uncertainties [n_field, n_ell]
        - 'ifield_list_use': list of fields with valid data
    """
    from ciber.core.powerspec_pipeline import CIBER_PS_pipeline
    from ciber.core.powerspec_utils import get_power_spec

    cbps = CIBER_PS_pipeline(dimx=int(1024*(field_size/2.0)), dimy=int(1024*(field_size/2.0)))
    basepath = config.ciber_basepath + f'data/fluctuation_data/TM{inst}/gal_density/'

    # Prepare addstr for the first redshift bin
    z0 = zbinedges[0]
    z1 = zbinedges[1]
    addstr = str(np.round(z0, 1)) + '_z_' + str(np.round(z1, 1))

    if field_size != 2.0:
        field_size_tag = f'_{field_size:.1f}deg'
    else:
        field_size_tag = ''

    # Get lbinedges for power spectrum calculation
    lbinedges = cbps.Mkk_obj.binl
    lbins = cbps.Mkk_obj.midbin_ell

    # Initialize power spectrum arrays
    all_cl_gal = []
    all_clerr_gal = []
    ifield_list_use = []

    for ifield in ifield_list:
        # Load data density map
        data_fpath = basepath + f'{catname}/gal_density_{catname}_TM{inst}_{addstr}{field_size_tag}.fits'
        print('Loading from ', data_fpath)
        try:
            hdul_data = fits.open(data_fpath)
            gal_map = hdul_data[f'ifield{ifield}'].data.transpose()
            hdul_data.close()
        except (FileNotFoundError, KeyError):
            print(f'[gal_auto_large] Missing data map for ifield {ifield}, skipping')
            continue

        # Subtract randoms if requested
        if subtract_randoms:
            rand_fpath = basepath + f'{catname}/gal_density_{catname}_TM{inst}_{addstr}_random{field_size_tag}.fits'
            print('random fpath is ', rand_fpath)
            try:
                hdul_rand = fits.open(rand_fpath)
                rand_map = hdul_rand[f'ifield{ifield}'].data.transpose()
                hdul_rand.close()

                # Scale random to match data (following ciber_gal_cross)
                gal_sum = gal_map.sum()
                rand_sum = rand_map.sum()
                scale = gal_sum / rand_sum

                print(f'[gal_auto_large] ifield {ifield} random scale factor: {scale:.4f}')

                # Subtract scaled randoms and normalize
                gal_map_masked = gal_map - scale * rand_map
                mean_rand = np.mean(scale * rand_map)
                gal_map_masked /= mean_rand

                if plot:
                    plt.figure(figsize=(10, 4))
                    plt.subplot(1, 3, 1)
                    plt.hist(gal_map.ravel(), bins=50)
                    plt.yscale('log')
                    plt.xlabel('Data counts')
                    plt.title(f'ifield {ifield} data')
                    plt.subplot(1, 3, 2)
                    plt.hist((scale * rand_map).ravel(), bins=50)
                    plt.yscale('log')
                    plt.xlabel('Scaled random counts')
                    plt.title('Scaled randoms')
                    plt.subplot(1, 3, 3)
                    plt.hist(gal_map_masked.ravel(), bins=50)
                    plt.yscale('log')
                    plt.xlabel('$\\delta_g$')
                    plt.title('Galaxy overdensity')
                    plt.tight_layout()
                    plt.show()

            except (FileNotFoundError, KeyError):
                print(f'[gal_auto_large] Missing random map for ifield {ifield}, using data only')
                # Normalize to overdensity
                meandens = np.mean(gal_map)
                gal_map_masked = gal_map.copy()
                gal_map_masked -= meandens
                gal_map_masked /= meandens
        else:
            # No random subtraction: just normalize to overdensity
            meandens = np.mean(gal_map)
            gal_map_masked = gal_map.copy()
            gal_map_masked -= meandens
            gal_map_masked /= meandens

        # Compute power spectrum using get_power_spec
        lb, cl_gal, clerr_gal = get_power_spec(gal_map_masked, map_b=None, mask=None, pixsize=7.,
                                                lbinedges=lbinedges, lbins=lbins)

        all_cl_gal.append(cl_gal)
        all_clerr_gal.append(clerr_gal)
        ifield_list_use.append(ifield)

        if plot:
            pf = lb*(lb+1)/(2*np.pi)
            plt.figure(figsize=(5, 4))
            plt.loglog(lb, pf*cl_gal, 'o-', label=f'ifield {ifield}')
            plt.errorbar(lb, pf*cl_gal, yerr=pf*clerr_gal, fmt='none', alpha=0.5)
            plt.xlabel('$\\ell$', fontsize=14)
            plt.ylabel('$D_\\ell^{gg}$', fontsize=14)
            plt.title(f'{catname} galaxy auto TM{inst} ifield {ifield} ({field_size:.1f}°)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xlim(30, 1e5)
            plt.ylim(1e-4, 5e2)
            plt.show()

    if save and len(ifield_list_use) > 0:
        save_addstr = addstr
        if subtract_randoms:
            save_addstr += '_wrandsub'
        save_addstr += field_size_tag

        save_dict = {
            'lb': lb,
            'all_cl_gal': np.array(all_cl_gal),
            'all_clerr_gal': np.array(all_clerr_gal),
            'ifield_list_use': ifield_list_use,
            'field_size': field_size,
            'catname': catname,
        }

        save_fpath = basepath + f'{catname}/gal_auto_{catname}_TM{inst}_{save_addstr}.npz'
        print(f'[gal_auto_large] Saving to {save_fpath}')
        np.savez(save_fpath, **save_dict)

    return {
        'lb': lb,
        'all_cl_gal': np.array(all_cl_gal),
        'all_clerr_gal': np.array(all_clerr_gal),
        'ifield_list_use': ifield_list_use,
    }


def collect_gal_auto_large_vs_redshift(inst_list, zbinedges, ifield_list,
                                        field_size=4.0, catname='LS',
                                        subtract_randoms=True, fmask=0.67):
    """Load large-footprint galaxy auto-spectra and return field-averaged arrays.

    Loads per-field npz files from compute_gal_auto_spectrum_large() for each
    redshift bin and instrument, field-averages them, and returns a dict in the
    same format expected by run_gal_auto_fits_two_stage() via its gal_ps_dict
    bypass parameter.

    Parameters
    ----------
    inst_list : list
        CIBER instrument indices (e.g. [1, 2])
    zbinedges : array_like
        Redshift bin edges
    ifield_list : list
        List of CIBER field indices
    field_size : float, optional
        Field size in degrees (default 4.0)
    catname : str, optional
        Catalog name (default 'LS')
    subtract_randoms : bool, optional
        Whether randoms were subtracted when computing spectra
    fmask : float, optional
        Mask fraction per field for Knox error calculation

    Returns
    -------
    dict
        Dictionary with keys:
        - 'lb': multipole bin centers
        - 'full_cl_gal': field-averaged galaxy auto C_ell [n_inst, n_zbin, n_ell]
        - 'full_clerr_gal': field-averaged uncertainties with Knox [n_inst, n_zbin, n_ell]
    """
    from ciber.core.powerspec_pipeline import CIBER_PS_pipeline
    from ciber.core.powerspec_utils import compute_field_averaged_power_spectrum

    cbps = CIBER_PS_pipeline(dimx=int(1024*(field_size/2.0)), dimy=int(1024*(field_size/2.0)))
    lb = cbps.Mkk_obj.midbin_ell
    n_ell = len(lb)
    nzbin = len(zbinedges) - 1

    if field_size != 2.0:
        field_size_tag = f'_{field_size:.1f}deg'
    else:
        field_size_tag = ''

    # Knox prefactor using correct field_size² (not hardcoded 2×2 deg²)
    full_sky_deg2 = 41253.
    field_area_deg2 = field_size ** 2

    full_cl_gal = np.zeros((len(inst_list), nzbin, n_ell))
    full_clerr_gal = np.zeros((len(inst_list), nzbin, n_ell))

    for idx, inst in enumerate(inst_list):
        basepath = config.ciber_basepath + f'data/fluctuation_data/TM{inst}/gal_density/{catname}/'

        for zidx in range(nzbin):
            z0, z1 = zbinedges[zidx], zbinedges[zidx + 1]
            addstr = str(np.round(z0, 1)) + '_z_' + str(np.round(z1, 1))
            if subtract_randoms:
                addstr += '_wrandsub'
            addstr += field_size_tag

            fpath = basepath + f'gal_auto_{catname}_TM{inst}_{addstr}.npz'
            print(f'[collect_large] Loading {fpath}')

            try:
                dat = np.load(fpath, allow_pickle=True)
            except FileNotFoundError:
                print(f'[collect_large] Missing {fpath}, skipping zbin {zidx}')
                continue

            all_cl_gal = dat['all_cl_gal']     # [n_field, n_ell]
            all_clerr_gal = dat['all_clerr_gal']
            nfield = len(all_cl_gal)

            # Field-average with inverse-variance weighting
            if nfield > 1:
                fieldav_cl_gal, fieldav_clerr_gal, _, _ = compute_field_averaged_power_spectrum(
                    all_cl_gal.copy(), per_field_dcls=all_clerr_gal.copy()
                )
            else:
                fieldav_cl_gal = all_cl_gal[0]
                fieldav_clerr_gal = all_clerr_gal[0]

            # Knox cosmic variance with correct field_size
            fsky = nfield * fmask * field_area_deg2 / full_sky_deg2
            gal_knox_errors = np.sqrt(2. / ((2 * lb + 1) * cbps.Mkk_obj.delta_ell))
            gal_knox_errors /= np.sqrt(fsky)
            gal_knox_errors *= np.abs(fieldav_cl_gal)

            fieldav_clerr_gal = np.sqrt(gal_knox_errors**2 + fieldav_clerr_gal**2)

            full_cl_gal[idx, zidx] = fieldav_cl_gal
            full_clerr_gal[idx, zidx] = fieldav_clerr_gal

    return {
        'lb': lb,
        'full_cl_gal': full_cl_gal,
        'full_clerr_gal': full_clerr_gal,
    }


def separate_ls_catalog_by_z(zbinedges=None, ifield_list = [4, 5, 6, 7, 8], ls_cat_names = ['ra', 'dec', 'zphot'], mode='data', plot=False, 
                             ):
    
    ls_basepath = config.ciber_basepath+'data/ciber_x_gal/'

    cbps = CIBER_PS_pipeline()
    
    if zbinedges is None:
        zbinedges = np.array(list(np.arange(0, 1.4, 0.2))+[1.5, 2.0])


    for zidx in range(len(zbinedges)-1):

        print(str(zbinedges[zidx])+'<z<'+str(zbinedges[zidx+1]))
        
        catbasepath = ls_basepath+mode+'_catalogs/'

        ls_fpath = catbasepath+'LS_Dr8_z22_'+str(np.round(zbinedges[zidx], 1))+'_zphot_'+str(np.round(zbinedges[zidx+1], 1))
        
        if mode=='random':
            ls_fpath += '_random'
        ls_cat_full = fits.open(ls_fpath+'.fits')[1].data
            
        ls_cat_ra = ls_cat_full['ra']
        ls_cat_dec = ls_cat_full['dec']
        ls_cat_zphot = ls_cat_full['zphot']

        for fieldidx, ifield in enumerate(ifield_list):

            ra_cen, dec_cen = cbps.ra_cen_ciber_fields[ifield], cbps.dec_cen_ciber_fields[ifield]
            near_ciber_fp = (ls_cat_ra > ra_cen - 3.)*(ls_cat_ra < ra_cen + 3)*(ls_cat_dec > dec_cen - 3.)*(ls_cat_dec < dec_cen+3.)

            ls_ra_cut = ls_cat_ra[near_ciber_fp]
            ls_dec_cut = ls_cat_dec[near_ciber_fp]
            ls_zphot_cut = ls_cat_zphot[near_ciber_fp]

            ls_df = pd.DataFrame(np.array([ls_ra_cut, ls_dec_cut, ls_zphot_cut]).transpose(), columns=ls_cat_names)
            ls_filt = catalog_df_add_xy(cbps.ciber_field_dict[ifield], ls_df, datadir=config.ciber_basepath+'data/')
            ls_filt, _, _ = check_for_catalog_duplicates(ls_filt)

            if plot:
                plt.figure()
                plt.scatter(ls_filt['x1'], ls_filt['y1'], s=1, color='k')
                plt.xlim(0, 1024)
                plt.ylim(0, 1024)
                plt.show()
                
            print('for field '+str(ifield)+', there are ', len(ls_ra_cut))

            ls_save_fpath = catbasepath+'ciber_cut/ls_'+str(np.round(zbinedges[zidx], 1))+'_zphot_'+str(np.round(zbinedges[zidx+1], 1))+'_wxy_CIBER_ifield'+str(ifield)
            if mode=='random':
                ls_save_fpath += '_random'
            ls_save_fpath += '.csv'

            print('Saving catalog to ', ls_save_fpath)
            ls_filt.to_csv(ls_save_fpath)


def preprocess_gal_density_maps(inst, ifield_list, catname, save=False, cat_fpath_list=None,\
                                 show=True, addstr=None, which_hsc_band='i', **kwargs):
    
    
    gal_dict = return_default_gal_cat_dict()
    gal_dict = update_dicts([gal_dict], kwargs)[0]
       
    ciber_field_dict = dict({4:'elat10', 5:'elat30', 6:'Bootes B', 7:'Bootes A', 8:'SWIRE'})
    
    gal_densities = np.zeros((len(ifield_list), gal_dict['ciber_dimx'], gal_dict['ciber_dimx']))
    
    all_w1 = []
    for fieldidx, ifield in enumerate(ifield_list):
        
        if cat_fpath_list is None:
            cat_fpath = catalog_basepath+catname+'/filt/'+catname+'_CIBER_ifield'+str(ifield)+'.csv'
        else:
            cat_fpath = cat_fpath_list[fieldidx]
            
        # instantiate for each field separately
        cat_sel_obj = cat_select(gal_dict, which_hsc_band=which_hsc_band)
        cat_sel_obj.load_cat(cat_fpath, inst, catname)
        
        mask = cat_sel_obj.apply_cat_select(catname)

        cat_x_sel, cat_y_sel = cat_sel_obj.cat_x[np.where(mask)[0]], cat_sel_obj.cat_y[np.where(mask)[0]]
        
        print(cat_x_sel)
        
        print('After down-selections, the '+str(catname)+' catalog for '+str(ifield)+' has '+str(len(cat_x_sel))+' sources.')
        
        counts = get_count_field(cat_x_sel, cat_y_sel, imdim=gal_dict['ciber_dimx'])
        
        if catname=='WISE':
            cat_w1_sel = cat_sel_obj.cat_W1[np.where(mask)[0]]
            all_w1.append(cat_w1_sel)
        
        if show:
            plot_map(counts, title=catname+' ifield '+str(ifield))
        
        gal_densities[fieldidx] = counts
        
    
    if catname=='WISE':
        plt.figure(figsize=(5, 4))
        for fieldidx, ifield in enumerate(ifield_list):

            nbar = len(all_w1[fieldidx][(all_w1[fieldidx]<18.0)])/4.

            label = cbps.ciber_field_dict[ifield]+': $\\overline{n}=$'+str(int(nbar))+' deg$^{-2}$'

            if fieldidx==0:
                label +='\n(with W1 cut)'

            plt.hist(all_w1[fieldidx], bins=np.linspace(15, 18.5, 20), histtype='step', label=label)
        plt.yscale('log')
        plt.xlabel('W1 magnitude [Vega]', fontsize=14)
        plt.axvline(18.0, linestyle='dashed', color='k')
        plt.legend(fontsize=9, loc=4)
        plt.ylabel('$N_{src}$', fontsize=14)
        plt.title('unWISE neo8 catalog', fontsize=16)
        plt.savefig('figures/unWISE_neo8_W1counts_perfield_'+addstr+'.png', bbox_inches='tight')
        plt.show()

        
    if save:
        
        save_fpath = save_gal_density(inst, ifield_list, gal_densities, catname, addstr=addstr)
    else:
        save_fpath = None
    
    return save_fpath

# def preprocess_gal_density_maps(inst, ifield_list, catname, save=False, cat_fpath_list=None,\
#                                  show=True, addstr=None, **kwargs):
    
    
#     gal_dict = return_default_gal_cat_dict()
#     gal_dict = update_dicts([gal_dict], kwargs)[0]
       
#     ciber_field_dict = dict({4:'elat10', 5:'elat30', 6:'Bootes B', 7:'Bootes A', 8:'SWIRE'})
    
#     gal_counts = np.zeros((len(ifield_list), gal_dict['ciber_dimx'], gal_dict['ciber_dimx']))
    
#     for fieldidx, ifield in enumerate(ifield_list):
        
#         if cat_fpath_list is None:
#             cat_fpath = gal_dict['catalog_basepath']+catname+'/filt/'+catname+'_CIBER_ifield'+str(ifield)+'.csv'
#         else:
#             cat_fpath = cat_fpath_list[fieldidx]
            
#         # instantiate for each field separately
#         cat_sel_obj = cat_select(gal_dict)
#         cat_sel_obj.load_cat(cat_fpath, inst, catname)
        
#         mask = cat_sel_obj.apply_cat_select(catname)

#         cat_x_sel, cat_y_sel = cat_sel_obj.cat_x[np.where(mask)[0]], cat_sel_obj.cat_y[np.where(mask)[0]]
        
#         print('After down-selections, the '+str(catname)+' catalog for '+str(ifield)+' has '+str(len(cat_x_sel))+' sources.')
        
#         counts = get_count_field(cat_x_sel, cat_y_sel, imdim=gal_dict['ciber_dimx'])
        
#         if show:
#             plot_map(counts, title=catname+' ifield '+str(ifield))
        
#         gal_counts[fieldidx] = counts
        
#     if save:
        
#         save_fpath = save_gal_density(inst, ifield_list, gal_counts, catname, addstr=addstr)
#     else:
#         save_fpath = None
    
#     return save_fpath


def preprocess_intensity_maps(inst, ifield_list, catname='HSC', save=False, cat_fpath_list=None,
                              show=False, addstr=None, hsc_mag_column='i_cmodel_mag',
                              hsc_mag_min=None, hsc_mag_max=None, zmin=None, zmax=None,
                              imdim=1024, **kwargs):
    """Thin wrapper so intensity preprocessing is discoverable from galaxy_cross APIs."""
    from ciber.cross_correlation.intensity_recon_cross import preprocess_intensity_maps as _preprocess

    intensity_maps, save_fpath = _preprocess(
        inst=inst,
        ifield_list=ifield_list,
        catname=catname,
        save=save,
        cat_fpath_list=cat_fpath_list,
        addstr=addstr,
        hsc_mag_column=hsc_mag_column,
        hsc_mag_min=hsc_mag_min,
        hsc_mag_max=hsc_mag_max,
        zmin=zmin,
        zmax=zmax,
        show=show,
        imdim=imdim,
        **kwargs,
    )

    return intensity_maps, save_fpath


def preprocess_ls_density_maps(inst, zbinedges, ifield_list, 
                            save=False, imdim=1024, plot=False,
                            mode='data', catname='LS', remove_wen_cmgs=False, tailstr=None):
    
    ls_basepath = config.ciber_basepath+'data/ciber_x_gal/'
    
    ciber_field_dict = dict({4:'elat10', 5:'elat30', 6:'Bootes B', 7:'Bootes A', 8:'SWIRE'})

    gal_counts = np.zeros((len(ifield_list), imdim, imdim))
    
    all_gal_counts = []
    
    ngal_perz_perfield = np.zeros((len(zbinedges)-1, len(ifield_list)))

    for zidx, z0 in enumerate(zbinedges[:-1]):
        
        z1 = zbinedges[zidx+1]
        
        addstr = str(np.round(z0, 1))+'_z_'+str(np.round(z1, 1))

        if mode=='random':
            addstr += '_random'
        
        if tailstr is not None:
            addstr += '_'+tailstr
        
        for fieldidx, ifield in enumerate(ifield_list):

            cat_fpath = ls_basepath+mode+'_catalogs/ciber_cut/ls_'+str(np.round(z0, 1))+'_zphot_'+str(np.round(z1, 1))+'_wxy_CIBER_ifield'+str(ifield)

            if mode=='random':
                cat_fpath+= '_random'
            cat_fpath += '.csv'

            print('reading from ', cat_fpath)
            cat_df = pd.read_csv(cat_fpath)


            if remove_wen_cmgs:
                wen_basepath = 'data/catalogs/wen_cluster_gals/'
                wen_fpath = wen_basepath+'wen_cluster_member_gals_CIBER_ifield'+str(ifield)+'_wxy.csv'
                wen_df = pd.read_csv(wen_fpath)

                ls_src_coord = SkyCoord(ra=cat_df['ra']*u.degree, dec=cat_df['dec']*u.degree, frame='icrs', unit=u.deg)
                
                wen_src_coord = SkyCoord(ra=wen_df['ra']*u.degree, dec=wen_df['dec']*u.degree, frame='icrs', unit=u.deg)

                idx_xmatch, d2d_xmatch, _ = match_coordinates_sky(ls_src_coord, wen_src_coord)
                nodup_mask = np.where(d2d_xmatch.arcsec > 7.0)[0] # find all non-duplicates

                print('cat df before removing wen cmgs is ', len(cat_df))
                print('wen has ', len(wen_df), 'sources')
                cat_df = cat_df.iloc[nodup_mask].copy()
                print('cat df after removing wen cmgs is ', len(cat_df))


            cat_x = np.array(cat_df['x'+str(inst)])
            cat_y = np.array(cat_df['y'+str(inst)])
            cat_zphot = np.array(cat_df['zphot'])
        
            mask = (cat_x > 0)*(cat_x < imdim)*(cat_y > 0)*(cat_y < imdim)

            cat_x = cat_x[mask]
            cat_y = cat_y[mask]
            
            ngal_perz_perfield[zidx, fieldidx] = len(cat_x)

            counts = get_count_field(cat_x, cat_y, imdim=imdim)

            # gal_density = (counts - np.nanmean(counts))/np.nanmean(counts)
            
            if plot:
                plt.figure()
                plt.title(str(z0)+'$<z_{phot}<$'+str(z1), fontsize=14)
                plt.hist(cat_zphot[mask], bins=30, histtype='step')
                plt.xlabel('zphot')
                plt.ylabel('$N_g$')
                plt.show()
                
                plot_map(counts, title='LS ifield '+str(ifield))
                # plot_map(gal_density, title='Gal density LS ifield '+str(ifield))
                
            gal_counts[fieldidx] = counts
                        
        all_gal_counts.append(gal_counts)
        
        if save:
        
            save_fpath = save_gal_density(inst, ifield_list, gal_counts, catname, addstr=addstr)
        else:
            save_fpath = None
        
    return all_gal_counts, ngal_perz_perfield


def preprocess_ls_density_maps_large(inst, zbinedges, ifield_list, field_size=4.0,
                                      save=False, plot=False,
                                      mode='data', catname='LS', remove_wen_cmgs=False, tailstr=None):
    """Extract LS galaxy density maps for larger field regions.

    Loads from the full-sky LS catalogs and extracts a user-specified sky region
    around each CIBER field. This allows extraction of regions larger than the
    standard 2×2° CIBER FOV.

    Parameters
    ----------
    inst : int
        CIBER instrument (1 or 2)
    zbinedges : array_like
        Redshift bin edges
    ifield_list : list
        List of CIBER field indices
    field_size : float, optional
        Field size in degrees (default 4.0 for 4×4°). Scales output map size.
        1024 pixels = 2° → output_pixels = 1024 * (field_size / 2.0)
    save : bool, optional
        Whether to save density maps
    plot : bool, optional
        Whether to plot maps
    mode : str, optional
        'data' or 'random' catalog mode
    catname : str, optional
        Catalog name for labels
    remove_wen_cmgs : bool, optional
        Remove Wen cluster member galaxies
    tailstr : str, optional
        Tail string for filenames

    Returns
    -------
    all_gal_counts : list
        List of galaxy count arrays for each redshift bin
    ngal_perz_perfield : ndarray
        Number of galaxies per redshift bin per field
    """
    ls_basepath = config.ciber_basepath+'data/ciber_x_gal/'
    cbps = CIBER_PS_pipeline()

    # Scale output map size based on field_size
    # 1024 pixels = 2.0 degrees → scale by field_size/2.0
    base_imdim = 1024
    imdim = int(base_imdim * (field_size / 2.0))

    # Sky region extent (±degrees from field center)
    sky_extent = field_size

    gal_counts = np.zeros((len(ifield_list), imdim, imdim))

    all_gal_counts = []

    ngal_perz_perfield = np.zeros((len(zbinedges)-1, len(ifield_list)))

    for zidx, z0 in enumerate(zbinedges[:-1]):

        z1 = zbinedges[zidx+1]

        addstr = str(np.round(z0, 1))+'_z_'+str(np.round(z1, 1))

        if mode=='random':
            addstr += '_random'

        if tailstr is not None:
            addstr += '_'+tailstr

        # Load full-sky catalog for this redshift bin
        cat_basepath = ls_basepath+mode+'_catalogs/'
        ls_fpath = cat_basepath+'LS_Dr8_z22_'+str(np.round(z0, 1))+'_zphot_'+str(np.round(z1, 1))

        if mode=='random':
            ls_fpath += '_random'
        ls_fpath += '.fits'

        print(f'[large] reading full catalog from {ls_fpath}')
        ls_cat_full = fits.open(ls_fpath)[1].data

        ls_cat_ra = ls_cat_full['ra']
        ls_cat_dec = ls_cat_full['dec']
        ls_cat_zphot = ls_cat_full['zphot']

        for fieldidx, ifield in enumerate(ifield_list):

            ra_cen = cbps.ra_cen_ciber_fields[ifield]
            dec_cen = cbps.dec_cen_ciber_fields[ifield]

            # Extract larger sky region
            sky_mask = ((ls_cat_ra > ra_cen - sky_extent) *
                        (ls_cat_ra < ra_cen + sky_extent) *
                        (ls_cat_dec > dec_cen - sky_extent) *
                        (ls_cat_dec < dec_cen + sky_extent))

            ls_ra_cut = ls_cat_ra[sky_mask]
            ls_dec_cut = ls_cat_dec[sky_mask]
            ls_zphot_cut = ls_cat_zphot[sky_mask]

            # Convert to CIBER pixel coordinates
            ls_cat_names = ['ra', 'dec', 'zphot']
            ls_df = pd.DataFrame(np.array([ls_ra_cut, ls_dec_cut, ls_zphot_cut]).transpose(),
                                 columns=ls_cat_names)
            ls_filt = catalog_df_add_xy(cbps.ciber_field_dict[ifield], ls_df,
                                        datadir=config.ciber_basepath+'data/', imcut=False)
            ls_filt, _, _ = check_for_catalog_duplicates(ls_filt)

            if remove_wen_cmgs:
                wen_basepath = 'data/catalogs/wen_cluster_gals/'
                wen_fpath = wen_basepath+'wen_cluster_member_gals_CIBER_ifield'+str(ifield)+'_wxy.csv'
                wen_df = pd.read_csv(wen_fpath)

                ls_src_coord = SkyCoord(ra=ls_filt['ra']*u.degree, dec=ls_filt['dec']*u.degree,
                                        frame='icrs', unit=u.deg)
                wen_src_coord = SkyCoord(ra=wen_df['ra']*u.degree, dec=wen_df['dec']*u.degree,
                                         frame='icrs', unit=u.deg)

                idx_xmatch, d2d_xmatch, _ = match_coordinates_sky(ls_src_coord, wen_src_coord)
                nodup_mask = np.where(d2d_xmatch.arcsec > 7.0)[0]

                print(f'[large] field {ifield} before removing wen cmgs: {len(ls_filt)}')
                print(f'[large] wen has {len(wen_df)} sources')
                ls_filt = ls_filt.iloc[nodup_mask].copy()
                print(f'[large] field {ifield} after removing wen cmgs: {len(ls_filt)}')

            cat_x = np.array(ls_filt['x'+str(inst)])
            cat_y = np.array(ls_filt['y'+str(inst)])
            cat_zphot = np.array(ls_filt['zphot'])

            # For larger fields, calculate offset from field center
            # The 2×2° CIBER FOV is centered at (512, 512) in 1024×1024 space
            # For larger field, center is at (imdim/2, imdim/2)
            offset = (imdim - base_imdim) / 2.0
            x_min = -offset
            x_max = base_imdim + offset
            y_min = -offset
            y_max = base_imdim + offset

            mask = (cat_x > x_min) * (cat_x < x_max) * (cat_y > y_min) * (cat_y < y_max)

            cat_x = cat_x[mask]
            cat_y = cat_y[mask]

            # Shift coordinates to pixel space [0, imdim)
            cat_x_shifted = cat_x + offset
            cat_y_shifted = cat_y + offset

            ngal_perz_perfield[zidx, fieldidx] = len(cat_x_shifted)

            counts = get_count_field(cat_x_shifted, cat_y_shifted, imdim=imdim)

            if plot:
                plt.figure()
                plt.title(str(z0)+'$<z_{phot}<$'+str(z1), fontsize=14)
                plt.hist(cat_zphot[mask], bins=30, histtype='step')
                plt.xlabel('zphot')
                plt.ylabel('$N_g$')
                plt.show()

                plot_map(counts, title=f'LS ifield {ifield} ({field_size}°)')

            gal_counts[fieldidx] = counts

        all_gal_counts.append(gal_counts)

        if save:
            # Append field size to addstr for saved filenames
            save_addstr = addstr + f'_{field_size:.1f}deg'
            save_fpath = save_gal_density(inst, ifield_list, gal_counts, catname, addstr=save_addstr)
        else:
            save_fpath = None

    return all_gal_counts, ngal_perz_perfield


def wise_rgb_cuts(mag_W1, mag_W2, rgb_mode):
    
    color_W1W2 = mag_W1-mag_W2
    
    if rgb_mode=='blue':
        w2mask = (mag_W2 > 16.7)
        w1w2mask = (color_W1W2 < 0.3+((17.-mag_W2)/4))
        
    elif rgb_mode=='green':
        w2mask = (mag_W2 > 16.7)
        w1w2mask = (color_W1W2 < 0.8+((17.-mag_W2)/4))
        w1w2mask *= (color_W1W2 > 0.3+((17.-mag_W2)/4))

        
    elif rgb_mode=='red':
        w2mask = (mag_W2 > 16.2)
        w1w2mask = (color_W1W2 < 0.8+((17.-mag_W2)/4))

    rgb_mask = w2mask*w1w2mask
        
    return rgb_mask


