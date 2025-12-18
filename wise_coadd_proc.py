import os
import numpy as np
import requests
from tqdm import tqdm
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import block_reduce
from astropy.coordinates import SkyCoord
from astropy import units as u
from reproject import reproject_interp
# from reproject.exceptions import InvalidTransformError

from astroquery.ipac.irsa import Irsa
Irsa.ROW_LIMIT = -1

# Directory to store cached downloads
# CACHE_DIR = "unwise_neo8_cache"
# os.makedirs(CACHE_DIR, exist_ok=True)

def ra_dec_to_tile(ra, dec):
	# unWISE tile centers are spaced every 1.56 deg in RA and Dec
	ra_index = int(np.floor(ra / 1.56))
	dec_index = int(np.floor(dec / 1.56))
	ra_str = f"{ra_index:04d}"
	dec_sign = 'p' if dec_index >= 0 else 'm'
	dec_str = f"{abs(dec_index):03d}"
	return f"{ra_str}{dec_sign}{dec_str}"

def get_tile_dir(tile_name):
	# Extract the tile RA portion and convert to directory name
	return tile_name[:3]
	
def download_unwise_file(band, tile_name, filetype, CACHE_DIR='data/unWISE_coadds/unwise_neo8_cache/'):
	tile_dir = get_tile_dir(tile_name)
	base_url = "https://portal.nersc.gov/project/cosmo/data/unwise/neo8/unwise-coadds/fulldepth"
	
	
	if filetype=='msk':
		filename = f"unwise-{tile_name}-{filetype}.fits.gz"
	else:
		filename = f"unwise-{tile_name}-w{band}-{filetype}.fits.gz"

	url = f"{base_url}/{tile_dir}/{tile_name}/{filename}"
	local_path = os.path.join(CACHE_DIR, filename)

	if os.path.exists(local_path):
		print('File available locally, ', local_path)
		return local_path

	try:
		r = requests.get(url, stream=True, timeout=10)
		r.raise_for_status()
		with open(local_path, "wb") as f:
			for chunk in r.iter_content(chunk_size=8192):
				f.write(chunk)
		return local_path
	except Exception as e:
		print(f"Warning: could not download {url}: {e}")
		return None


def get_unwise_map(ra, dec, tiles=None, field_size_deg=2.0, band=1, pixel_scale=2.75, filetype='msk', \
				  check_if_exists=True, CACHE_DIR='data/unWISE_coadds/unwise_neo8_cache/'):
	"""
	Returns:
		- mask (np.ndarray)
		- depth map (np.ndarray)
		- WCS object
	"""
	from astropy.wcs.utils import proj_plane_pixel_scales

	field_size_pix = int(field_size_deg * 3600 / pixel_scale)
	size = (field_size_pix, field_size_pix)

	# Define WCS for the target field
	wcs = WCS(naxis=2)
	wcs.wcs.crpix = [field_size_pix // 2, field_size_pix // 2]
	wcs.wcs.cdelt = np.array([-pixel_scale / 3600.0, pixel_scale / 3600.0])
	wcs.wcs.crval = [ra, dec]
	wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

	# Determine surrounding tiles (within ~2 deg)
	coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
	radius = np.sqrt(2) * field_size_deg / 2.0 + 0.5
	ras = np.arange(ra - radius, ra + radius, 1.56)
	decs = np.arange(dec - radius, dec + radius, 1.56)
	
	if tiles is None:
		tiles = set(ra_dec_to_tile(r, d) for r in ras for d in decs)
	
	print('tiles:', tiles)

	if filetype=='msk':
		dtype_map = np.uint16
	else:
		dtype_map = np.float32

	combined_map = np.zeros(size, dtype=dtype_map)

	for tile in tqdm(tiles, desc="Processing tiles"):
		
		map_fpath = download_unwise_file(band, tile, filetype, CACHE_DIR=CACHE_DIR)
		
#         depth_path = download_unwise_file(band, tile, "depth")

		if map_fpath:
			try:
				with fits.open(map_fpath) as hdul:
					data = hdul[0].data
					tile_wcs = WCS(hdul[0].header)
					reproj_map, _ = reproject_interp((data, tile_wcs), wcs, shape_out=size)

					if filetype=='msk':
						combined_map |= np.nan_to_num(reproj_map.astype(np.uint16), nan=0)
					elif 'invvar' in filetype:
						combined_map = np.nansum([combined_map, reproj_map], axis=0)
					elif 'std' in filetype:
						inv_variance_map = 1./reproj_map**2 
						combined_map = np.nansum([combined_map, inv_variance_map], axis=0)

			except (OSError, ValueError) as e:
				print(f"Warning: could not reproject mask for {tile}: {e}")
				
	if 'std' in filetype:
		combined_map = 1./np.sqrt(combined_map)

	return combined_map, wcs


def grab_wise_info_maps(ifield_list, filetype='msk',\
						CACHE_DIR='data/unWISE_coadds/unwise_neo8_cache/', \
						plot=True, save=False, save_basepath='data/unWISE_coadds/', addstr=None):
	
	cbps = CIBER_PS_pipeline()
	
	all_maps, all_wcs = [], []
	
	tile_lists = dict({'elat10': {'1906p075', '1922p075', '1914p090', '1899p090', '1891p075', '1808p060', '1913p060'}, \
					  'elat30':{'1926p272', '1943p272', '1933p257', '1916p257', '1909p272', '1920p287', '1937p287'}, \
					  'BootesA':{'2201p363', '2182p363', '2163p363', '2156p348', '2174p348', '2192p348', '2211p348', '2195p333', '2177p333', '2159p333', '2188p318', '2170p318', '2152p318', '2205p318', '2213p333'}, \
					'BootesB':{'2201p363', '2182p363', '2163p363', '2156p348', '2174p348', '2192p348', '2211p348', '2195p333', '2177p333', '2159p333', '2188p318', '2170p318', '2152p318', '2205p318', '2213p333'}, \
						'SWIRE':{'2425p545', '2451p545', '2453p560', '2426p560', '2399p560', '2400p545','2391p530', '2416p530', '2441p530'}})
	
	for fieldidx, ifield in enumerate(ifield_list):
		
		fieldname = cbps.ciber_field_dict[ifield]
		ra, dec = cbps.ra_cen_ciber_fields[ifield], cbps.dec_cen_ciber_fields[ifield]
		tile_list = tile_lists[fieldname]
					
		wise_map, wcs = get_unwise_map(ra=ra, dec=dec, tiles=tile_list, field_size_deg=2.5, filetype=filetype, CACHE_DIR=CACHE_DIR)
		
		if plot:
			plot_map(wise_map, title=filetype+' for ifield '+str(ifield), figsize=(6, 6))
		

		all_wcs.append(wcs)
		all_maps.append(wise_map)
		
		if save:
			filename_save = save_basepath+filetype+'_and_wcs_ifield'+str(ifield)
			if addstr is not None:
				filename_save+='_'+addstr
			filename_save += '.fits'
			
			print('saving map to ', filename_save)
			save_mask_to_fits(wise_map, wcs, filename_save)
		

	return all_maps, all_wcs

def convert_invvar_to_depth_map(ifield_list=[4, 5, 6, 7, 8], save_basepath='data/unWISE_coadds/combined_invvar/', \
							   wise_channel=1, plot=False):
		
	
	zp_Vega = 22.5 # Check unWISE documentation to confirm this is the correct ZP for your data

	all_depth_maps, all_headers = [], []
	
	for fieldidx, ifield in enumerate(ifield_list):
		
		std_fpath = save_basepath+'std-m_and_wcs_ifield'+str(ifield)+'.fits'
		std_m_file = fits.open(std_fpath)
		
		sigma_map = std_m_file[0].data

		depth_map = zp_Vega - 2.5 * np.log10(5*sigma_map)
	
		if plot:
			plt.figure()
			plt.hist(depth_map.ravel(), bins=np.linspace(19.5, 21.5, 50))
			plt.xlabel('depth map [Vega]', fontsize=14)
			plt.show()
			
			plot_map(depth_map, title='depth map ifield '+str(ifield), figsize=(6, 6), vmin=20, vmax=21.5)
			
		all_depth_maps.append(depth_map)
		all_headers.append(std_m_file[0].header)
			
			
	return all_depth_maps, all_headers

def regrid_wise_depth_maps(ciber_inst, wise_maps, wise_wcs_headers, save_basepath='data/unWISE_coadds/combined_invvar/',\
						   ifield_list=[4, 5, 6, 7, 8], quad_list=['A', 'B', 'C', 'D'], interp_order=0, \
						  mask_depth_cut_pct=1, save_depth_cut_mask=True):
	
	cbps = CIBER_PS_pipeline()
	x0, x1, y0, y1 = 0, 1024, 0, 1024
	wise_map_regrid = np.zeros((len(ifield_list), x1, y1))

	filenames = []
	for fieldidx, ifield in enumerate(ifield_list):

		astr_map_hdrs = load_quad_hdrs(ifield, ciber_inst, base_path='data/', halves=False)
		
		wcs_hdr = wise_wcs_headers[fieldidx]
		
#         wcs_hdr = WCS(maskfile[0].header)
#         mask = maskfile[0].data
		
		
		for iquad, quad in enumerate(quad_list):
			print('iquad = ', iquad)
			wise_dat = (wise_maps[fieldidx], wcs_hdr)
			print('Using reproject interp, order '+str(interp_order))
			
			map_regrid, footprint = reproject_interp(wise_dat, astr_map_hdrs[iquad], (512, 512), order=interp_order)

			wise_map_regrid[fieldidx, cbps.y0s[iquad]:cbps.y1s[iquad], cbps.x0s[iquad]:cbps.x1s[iquad]] = map_regrid

		plot_map(wise_map_regrid[fieldidx], title='regrid map', figsize=(6, 6), x0=x0, x1=x1, y0=y0, y1=y1)
		
		plt.figure()
		plt.hist(wise_map_regrid[fieldidx].ravel(), bins=np.linspace(10, 22, 50))
		plt.yscale('log')
		plt.show()
		
		minpctval = np.nanpercentile(wise_map_regrid[fieldidx], mask_depth_cut_pct)
		depth_cut_mask = (wise_map_regrid[fieldidx] < minpctval).astype(int)
		
		plot_map(depth_cut_mask, title='depth cut mask = '+str(np.round(minpctval, 1)))
		
#         print('1st percentile of map regrid:', np.nanpercentile(wise_map_regrid[fieldidx], 1))
		
		
		filename = save_basepath+'unWISE_5sig_pointsrc_depth_regrid_CIBER_TM'+str(ciber_inst)+'_ifield'+str(ifield)+'.fits'
		hdu = fits.PrimaryHDU(data=wise_map_regrid[fieldidx])
		# Write to file
		print('Saving to ', filename)
		hdu.writeto(filename, overwrite=True)
		
		
		if save_depth_cut_mask:
			
			filename_mask = save_basepath+'unWISE_depth_mask_regrid_pctcut='+str(mask_depth_cut_pct)+'_CIBER_TM'+str(ciber_inst)+'_ifield'+str(ifield)+'.fits'
			hdum = fits.PrimaryHDU(data=depth_cut_mask)
			# Write to file
			print('Saving to ', filename_mask)
			hdum.writeto(filename_mask, overwrite=True)
		
		filenames.append(filename)
	
	return wise_map_regrid


def save_mask_to_fits(mask, wcs, filename):
	# Convert WCS to FITS header
	header = wcs.to_header()

	# Create PrimaryHDU with the mask data and WCS header
	hdu = fits.PrimaryHDU(data=mask, header=header)

	# Write to file
	hdu.writeto(filename, overwrite=True)

# def depth_to_noise(depth_map):
# 	"""
# 	Convert depth map (inverse variance) to noise map (standard deviation).
# 	"""
# 	return 1.0 / np.sqrt(depth_map)

# def snr_from_flux_and_noise(flux_map, noise_map):
# 	"""
# 	Calculate the SNR from the flux and noise maps.
# 	"""
# 	return flux_map / noise_map

# def flux_from_snr_and_noise(snr_map, noise_map):
# 	"""
# 	Calculate flux from SNR and noise.
# 	"""
# 	return snr_map * noise_map

# def magnitude_from_flux(flux_map, zero_point=25.0):
# 	"""
# 	Convert flux to magnitude. Zero point is typically 25 for unWISE in the near-infrared bands.
# 	"""
# 	return -2.5 * np.log10(flux_map) + zero_point

# def effective_magnitude_limit(depth_map, flux_map, exposure_count, snr_threshold=5.0):
# 	"""
# 	Calculate the effective magnitude limit based on the depth map, flux map,
# 	and number of exposures.
# 	"""
# 	# Convert depth to noise
# 	noise_map = depth_to_noise(depth_map)

# 	# Adjust depth for multiple exposures (assuming added inverse variance)
# 	combined_depth = depth_map / exposure_count
# 	combined_noise = depth_to_noise(combined_depth)

# 	# Calculate SNR for the given flux
# 	snr_map = snr_from_flux_and_noise(flux_map, combined_noise)

# 	# Calculate the flux corresponding to the SNR threshold
# 	flux_at_limit = flux_from_snr_and_noise(np.full_like(snr_map, snr_threshold), combined_noise)

# 	# Convert flux at limit to magnitude
# 	mag_limit = magnitude_from_flux(flux_at_limit)

	return mag_limit