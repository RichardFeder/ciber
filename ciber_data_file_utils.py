import numpy as np
import astropy.io.fits as fits
import astropy.wcs as wcs
import pandas as pd
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
from astropy.table import Table
from PIL import Image
import os


'''---------------------- loading functions ----------------------'''
def load_all_ciber_quad_wcs_hdrs(inst, field, hdrdir=None):
    
    if hdrdir is None:
        hdrdir = 'data/astroutputs/inst'+str(inst)+'/'

    xoff = [0,0,512,512]
    yoff = [0,512,0,512]

    wcs_hdrs = []
    for iquad,quad in enumerate(['A','B','C','D']):
        print('quad '+quad)
        hdulist = fits.open(hdrdir + field + '_' + quad + '_astr.fits')
        wcs_hdr=wcs.WCS(hdulist[('primary',1)].header, hdulist)
        wcs_hdrs.append(wcs_hdr)

    return wcs_hdrs


def load_focus_mat(filename, nfr):
    x = loadmat(filename, struct_as_record=True)
    nfr_arr = x['framedat'][0,0]['nfr_arr'][0]
    if nfr in nfr_arr:
        nfr_idx = list(nfr_arr).index(nfr)

        if np.max(nfr_arr) < maxframe:
            return x['framedat'][0,0]['map_arr'][nfr_idx]
        else:
            return None

def load_regrid_dgl_map(inst, ifield, dgl_mode, base_path=None, p = 0.0184):

	if base_path is None:
		base_path = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/dgl_tracer_maps/'+dgl_mode+'/'

	regrid_fpath = base_path + dgl_mode+'_regrid_ifield'+str(ifield)+'_TM'+str(inst)+'_fromhp_nside=2048_120423.fits'

	print('regrid fpath is ', regrid_fpath)
	regrid_map = fits.open(regrid_fpath)[1].data

	if dgl_mode != 'IRIS':
		regrid_map /= p # from E(B-V) to 100 um intensity

	return regrid_map


def load_read_noise_modl_filedat(fw_basepath, inst, ifield, tailstr=None, apply_FW=False, load_cl1d=False, \
								mean_nl2d_key='mean_nl2d', fw_key='fourier_weights'):
	fw_fpath = fw_basepath+'/fourier_weights_TM'+str(inst)+'_ifield'+str(ifield)
	if tailstr is not None:
		fw_fpath += '_'+tailstr
	print('loading fourier weights and mean nl2d from ', fw_fpath+'.npz')
	
	fourier_weights=None 
	nm_file = np.load(fw_fpath+'.npz')

	if apply_FW:
		fourier_weights = nm_file[fw_key]
	
	mean_nl2d = nm_file[mean_nl2d_key]  
	cl1d_unweighted = None 
	if load_cl1d:
		cl1d_unweighted = nm_file['cls_1d_unweighted']

	return mean_nl2d, fourier_weights, cl1d_unweighted

def load_isl_rms(isl_rms_fpath, masking_maglim, nfield):
	
	isl_rms = np.load(isl_rms_fpath)['isl_sb_rms']
	isl_maglims = np.load(isl_rms_fpath)['mag_lim_list']

	matchidx = np.where(isl_maglims == masking_maglim)[0]
	if len(matchidx)==0:
		print('Cannot find masking magnitude limit of ', masking_maglim, ' in ', isl_rms_fpath)
		mean_isl_rms = np.zeros((nfield))
	else:
		mean_isl_rms = np.mean(isl_rms[:, :, matchidx[0]], axis=0)
		print("mean isl rms of each field is ", mean_isl_rms)
		
	return mean_isl_rms

def load_quad_hdrs(ifield, inst, base_path='/Users/richardfeder/Downloads/ciber_flight/', quad_list=['A', 'B', 'C', 'D'], halves=True):
    
    if halves:
        fpaths_first = [base_path+'/TM'+str(inst)+'/firsthalf/ifield'+str(ifield)+'/ciber_wcs_ifield'+str(ifield)+'_TM'+str(inst)+'_quad'+quad_str+'_firsthalf.fits' for quad_str in quad_list]
        fpaths_second = [base_path+'/TM'+str(inst)+'/secondhalf/ifield'+str(ifield)+'/ciber_wcs_ifield'+str(ifield)+'_TM'+str(inst)+'_quad'+quad_str+'_secondhalf.fits' for quad_str in quad_list]
        
        all_wcs_first = [wcs.WCS(fits.open(fpath_first)[0].header) for fpath_first in fpaths_first]
        all_wcs_second = [wcs.WCS(fits.open(fpath_second)[0].header) for fpath_second in fpaths_second]
        
        return all_wcs_first, all_wcs_second
    else:
        ciber_field_dict = dict({4:'elat10', 5:'elat30',6:'BootesB', 7:'BootesA', 8:'SWIRE', 'train':'UDS'})

        fpaths = [base_path+'/astroutputs/inst'+str(inst)+'/'+ciber_field_dict[ifield]+'_'+quad_str+'_astr.fits' for quad_str in quad_list]
        all_wcs = [wcs.WCS(fits.open(fpath)[0].header) for fpath in fpaths]
        return all_wcs


def load_weighted_cl_file_cross(cl_fpath, mode='observed'):

    clfile = np.load(cl_fpath)
    
    if mode=='observed':
        observed_recov_ps = clfile['observed_recov_ps']
        observed_recov_dcl_perfield = clfile['observed_recov_dcl_perfield']
        observed_field_average_cl = clfile['observed_field_average_cl']
        observed_field_average_dcl = clfile['observed_field_average_dcl']
        lb = clfile['lb']
    
        return lb, observed_recov_ps, observed_recov_dcl_perfield, observed_field_average_cl, observed_field_average_dcl, None
    
    elif mode=='mock':
        mock_mean_input_ps = clfile['mock_mean_input_ps']
        mock_all_field_averaged_cls = clfile['mock_all_field_averaged_cls']
        mock_all_field_cl_weights = clfile['mock_all_field_cl_weights']
        all_mock_recov_ps = clfile['all_mock_recov_ps']
        all_mock_signal_ps = clfile['all_mock_signal_ps']
        lb = clfile['lb']
    
        return lb, mock_mean_input_ps, mock_all_field_averaged_cls, mock_all_field_cl_weights, all_mock_recov_ps, all_mock_signal_ps

def read_ciber_powerspectra(filename):
	''' Given some file path name, this function loads/parses previously measured CIBER power spectra'''
	array = np.loadtxt(filename, skiprows=8)
	ells = array[:,0]
	norm_cl = array[:,1]
	norm_dcl_lower = array[:,2]
	norm_dcl_upper = array[:,3]
	return np.array([ells, norm_cl, norm_dcl_lower, norm_dcl_upper])


def load_regrid_lens_map(inst, ifield, cmblens_mode, base_path=None):
	if base_path is None:
		base_path = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/cmblens_maps/'+cmblens_mode+'/'

	regrid_fpath = base_path + cmblens_mode+'_regrid_ifield'+str(ifield)+'_TM'+str(inst)+'_fromhp_nside=2048.fits'
	print('regrid fpath is ', regrid_fpath)
	regrid_map = fits.open(regrid_fpath)[1].data

	return regrid_map


''' File directory structure '''

def init_mocktest_fpaths(ciber_mock_fpath, run_name):
    ff_fpath = ciber_mock_fpath+'030122/ff_realizations/'+run_name+'/'
    noisemod_fpath = ciber_mock_fpath +'030122/noise_models_sim/'+run_name+'/'
    input_recov_ps_fpath = 'data/input_recovered_ps/sim_tests_030122/'+run_name+'/'

    fpaths = [input_recov_ps_fpath, ff_fpath, noisemod_fpath]
    for fpath in fpaths:
        if not os.path.isdir(fpath):
            print('making directory path for ', fpath)
            os.makedirs(fpath)
        else:
            print(fpath, 'already exists')

    return ff_fpath, noisemod_fpath, input_recov_ps_fpath

def make_fpaths(fpaths):
    for fpath in fpaths:
        if not os.path.isdir(fpath):
            print('making directory path for ', fpath)
            os.makedirs(fpath)
        else:
            print(fpath, 'already exists')

'''---------------------- Data file formatting and saving ----------------------'''



def write_ff_file(ff_estimate, ifield, inst, sim_idx=None, dat_type=None, mag_lim_AB=None, ff_stack_min=None):
    hdum = fits.ImageHDU(ff_estimate, name='ff_'+str(ifield))
    hdup = fits.PrimaryHDU()
    hdup.header['ifield'] = ifield
    hdup.header['inst'] = inst
    
    if sim_idx is not None:
        hdup.header['sim_idx'] = sim_idx
    if dat_type is not None:
        hdup.header['dat_type'] = dat_type
    if mag_lim_AB is not None:
        hdup.header['mag_lim_AB'] = mag_lim_AB
    if ff_stack_min is not None:
        hdup.header['ff_stack_min'] = ff_stack_min

    hdul = fits.HDUList([hdup, hdum])
    
    return hdul

def write_Mkk_fits(Mkk, inv_Mkk, ifield, inst, cross_inst=None, sim_idx=None, generate_starmask=True, generate_galmask=True, \
                  use_inst_mask=True, dat_type=None, mag_lim_AB=None):
    hduim = fits.ImageHDU(inv_Mkk, name='inv_Mkk_'+str(ifield))        
    hdum = fits.ImageHDU(Mkk, name='Mkk_'+str(ifield))

    hdup = fits.PrimaryHDU()
    hdup.header['ifield'] = ifield
    hdup.header['inst'] = inst

    if cross_inst is not None:
        hdup.header['cross_inst'] = cross_inst
    if sim_idx is not None:
        hdup.header['sim_idx'] = sim_idx
    hdup.header['generate_galmask'] = generate_galmask
    hdup.header['generate_starmask'] = generate_starmask
    hdup.header['use_inst_mask'] = use_inst_mask
    if dat_type is not None:
        hdup.header['dat_type'] = dat_type
    if mag_lim_AB is not None:
        hdup.header['mag_lim_AB'] = mag_lim_AB
        
    hdup.header['n_ps_bin'] = Mkk.shape[0]

    hdul = fits.HDUList([hdup, hdum, hduim])
    return hdul

def write_regrid_proc_file(masked_proc, ifield, inst, regrid_to_inst, mask_tail=None,\
                           dat_type='observed', mag_lim=None, mag_lim_cross=None, obs_level=None):
    hdum = fits.ImageHDU(masked_proc, name='proc_regrid_'+str(ifield))
    hdup = fits.PrimaryHDU()
    hdup.header['ifield'] = ifield
    hdup.header['inst'] = inst
    hdup.header['regrid_to_inst'] = regrid_to_inst
    
    if mask_tail is not None:
        hdup.header['mask_tail'] = mask_tail
    if dat_type is not None:
        hdup.header['dat_type'] = dat_type
    if mag_lim is not None:
        hdup.header['mag_lim'] = mag_lim
    if mag_lim_cross is not None:
        hdup.header['mag_lim_cross'] = mag_lim_cross
    if obs_level is not None:
        hdup.header['obs_level'] = obs_level

    hdul = fits.HDUList([hdup, hdum])
    
    return hdul

def write_noise_model_fits(noise_model, ifield, inst, pixsize=7., photon_eps=None, \
                  dat_type=None):

    ''' writes noise model to FITS file with some metadata. '''

    hduim = fits.ImageHDU(noise_model, name='noise_model_'+str(ifield))        

    hdup = fits.PrimaryHDU()
    hdup.header['ifield'] = ifield
    hdup.header['inst'] = inst
    hdup.header['pixsize'] = pixsize
    if photon_eps is not None:
        hdup.header['photon_eps'] = photon_eps
    if dat_type is not None:
        hdup.header['dat_type'] = dat_type        

    hdul = fits.HDUList([hdup, hduim])
    return hdul

def write_mask_file(mask, ifield, inst, cross_inst=None, sim_idx=None, generate_galmask=None, generate_starmask=None, use_inst_mask=None, \
                   dat_type=None, mag_lim_AB=None, with_ff_mask=None, name=None, a1=None, b1=None, c1=None, dm=None, alpha_m=None, beta_m=None):

    if name is None:
        name = 'joint_mask_'+str(ifield)
    hdum = fits.ImageHDU(mask, name=name)
    hdup = fits.PrimaryHDU()
    hdup.header['ifield'] = ifield
    hdup.header['inst'] = inst

    if cross_inst is not None:
        hdup.header['cross_inst'] = cross_inst
    if sim_idx is not None:
        hdup.header['sim_idx'] = sim_idx
    if generate_galmask is not None:
        hdup.header['generate_galmask'] = generate_galmask
    if generate_starmask is not None:
        hdup.header['generate_starmask'] = generate_starmask
    if use_inst_mask is not None:
        hdup.header['use_inst_mask'] = use_inst_mask
    if dat_type is not None:
        hdup.header['dat_type'] = dat_type
    if mag_lim_AB is not None:
        hdup.header['mag_lim_AB'] = mag_lim_AB
    if with_ff_mask is not None:
        hdup.header['with_ff_mask'] = with_ff_mask

    if a1 is not None:
        hdup.header['a1'] = a1
    if b1 is not None:
        hdup.header['b1'] = b1
    if c1 is not None:
        hdup.header['c1'] = c1
    if dm is not None:
        hdup.header['dm'] = dm

    if alpha_m is not None:
        hdup.header['alpha_m'] = alpha_m
    if beta_m is not None:
        hdup.header['beta_m'] = beta_m
        
    hdul = fits.HDUList([hdup, hdum])
    
    return hdul


def make_fits_files_by_quadrant(images, ifield_list=[4, 5, 6, 7, 8], use_maskinst=True, tail_name='', \
                               save_path_base = '/Users/richardfeder/Downloads/ciber_flight/'):
    
    xs = [0, 0, 512, 512]
    ys = [0, 512, 0, 512]
    quadlist = ['A', 'B', 'C', 'D']
    
    maskinst_basepath = config.exthdpath+'ciber_fluctuation_data/TM1/masks/maskInst_102422'
    for i in range(len(images)):
        
        mask_inst = fits.open(maskinst_basepath+'/field'+str(ifield_list[i])+'_TM'+str(inst)+'_maskInst_102422.fits')[1].data
        image = images[i].copy()
        
        if use_maskinst:
            image *= mask_inst
        
        for q in range(4):
            quad = image[ys[q]:ys[q]+512, xs[q]:xs[q]+512]
            plot_map(quad, title='quad '+str(q+1))
            
            hdul = fits.HDUList([fits.PrimaryHDU(quad), fits.ImageHDU(quad)])
            
            save_fpath = save_path_base+'TM'+str(inst)+'/secondhalf/ifield'+str(ifield_list[i])+'/ciber_flight_ifield'+str(ifield_list[i])+'_TM'+str(inst)+'_quad'+quadlist[q]
            if tail_name is not None:
                save_fpath += '_'+tail_name
            hdul.writeto(save_fpath+'.fits', overwrite=True)

def save_mock_to_fits(full_maps, cats, tail_name=None, full_maps_band2=None, m_tracer_max=None, m_min=None, m_max=None, inst=None, \
					 data_path='/Users/luminatech/Documents/ciber2/ciber/data/mock_cib_fftest/082321/', \
					 ifield_list=None, map_names=None, names=['x', 'y', 'redshift', 'm_app', 'M_abs', 'Mh', 'Rvir'], save_fpath=None, return_save_fpath=False, **kwargs):
	''' This function is dedicated to converting mocks from ciber_mock.make_mock_ciber_map() to a fits file where they can be accessed.'''
	hdul = []
	hdr = fits.Header()

	for key, value in kwargs.items():
		hdr[key] = value

	if m_tracer_max is not None:
		hdr['m_tracer_max'] = m_tracer_max
	if m_min is not None:
		hdr['m_min'] = m_min
	if m_max is not None:
		hdr['m_max'] = m_max
	if inst is not None:
		hdr['inst'] = inst
		
	primary_hdu = fits.PrimaryHDU(header=hdr)
	hdul.append(primary_hdu)

	for c, cat in enumerate(cats):
		print('cat shape here is ', cat.shape)
		tab = Table([cat[:,i] for i in range(len(names))], names=names)
		cib_idx = c
		if ifield_list is not None:
			cib_idx = ifield_list[c]
			
		if map_names is not None:
			map_name = map_names[0]

			if len(map_names)==2:
				map_name2 = map_names[1]
		else:
			map_name = 'map'
			map_name2 = 'map2'
			
		hdu = fits.BinTableHDU(tab, name='tracer_cat_'+str(cib_idx))
		hdul.append(hdu)

		im_hdu = fits.ImageHDU(full_maps[c], name=map_name+'_'+str(cib_idx))
		hdul.append(im_hdu)
		
		if full_maps_band2 is not None:
			im_hdu2 = fits.ImageHDU(full_maps_band2[c], name=map_name2+'_'+str(cib_idx))
			hdul.append(im_hdu2)

	hdulist = fits.HDUList(hdul)
	
	if save_fpath is None:
		save_fpath = data_path+tail_name+'.fits'

	hdulist.writeto(save_fpath, overwrite=True)

	if return_save_fpath:
		return save_fpath

def save_resid_cl_file(cl_table, names, mode='isl', return_hdul=False, save=True, cl_save_fpath=None, **kwargs):
    tab = Table(cl_table, names=tuple(names))
    hdu_cl = fits.BinTableHDU(tab, name='cls_'+mode)
    hdr = fits.Header()

    for key, value in kwargs.items():
        hdr[key] = value
    prim = fits.PrimaryHDU(header=hdr)
    hdul = fits.HDUList([prim, hdu_cl])
    if save:
        if cl_save_fpath is None:
            print("No cl_save_fpath provided..")
        else:
            hdul.writeto(cl_save_fpath, overwrite=True)

    if return_hdul:
        return hdul

def save_mock_items_to_npz(filepath, catalog=None, srcmap_full=None, srcmap_nb=None, \
						   conv_noise=None, m_min=None, m_min_nb=None, ihl_map=None, m_lim=None):
	''' Convenience file for saving mock observation files. '''
	np.savez_compressed(filepath, catalog=catalog, srcmap_full=srcmap_full, \
						srcmap_nb=srcmap_nb, conv_noise=conv_noise, \
						ihl_map=ihl_map, m_lim=m_lim, m_min=m_min, m_min_nb=m_min_nb)


def convert_pngs_to_gif(filenames, gifdir='../../M_ll_correction/', name='mkk', duration=1000, loop=0):

    # Create the frames
    frames = []
    for i in range(len(filenames)):
        new_frame = Image.open(gifdir+filenames[i])
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(gifdir+name+'.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=duration, loop=loop)


