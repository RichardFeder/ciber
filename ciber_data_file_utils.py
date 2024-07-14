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
import config

'''---------------------- loading functions ----------------------'''

    
def grab_wav_dgl_col_from_csv(mean_fpath, upper_fpath):
    meanfile = np.array(pd.read_csv(mean_fpath, header=None))
    upper = np.array(pd.read_csv(upper_fpath, header=None))[:,1]
    
    wav = meanfile[:,0]
    mean = meanfile[:,1]
    std = upper-mean
    
    return wav, mean, std

def grab_dgl_config(dgl_mode, addstr=''):

    
    observed_run_name = 'ciber_'+dgl_mode+'_cross_nside=2048_ptsrcsubtract_full_gradfilter_I100_120423'

    if dgl_mode=='sfd_clean':
        observed_run_name = 'ciber_csfd_cross_nside=2048_ptsrcsubtract_full_gradfilter_I100_120423'
        cross_text_lab = 'CSFD (-LSS)'
    elif dgl_mode=='sfd_clean_plus_LSS':
        cross_text_lab='SFD'
    elif dgl_mode=='IRIS':
        cross_text_lab = 'IRIS 100 $\\mu$m'
    elif dgl_mode=='mf15':
        cross_text_lab = 'SFD (MF15)'
    
    return observed_run_name, cross_text_lab

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

def load_clx_dgl(dgl_mode='sfd_clean', verbose=False):
    
    dgl_auto_TM1 = np.load(config.ciber_basepath+'data/fluctuation_data/TM1/dgl_tracer_maps/'+dgl_mode+'/dgl_auto_constraints_TM1_'+dgl_mode+'_053023.npz')
    lb_modl = dgl_auto_TM1['lb_modl']
    best_ps_fit_av_TM1 = dgl_auto_TM1['best_ps_fit_av']

    dgl_auto_TM2 = np.load(config.ciber_basepath+'data/fluctuation_data/TM2/dgl_tracer_maps/'+dgl_mode+'/dgl_auto_constraints_TM2_'+dgl_mode+'_053023.npz')
    lb_modl = dgl_auto_TM2['lb_modl']
    best_ps_fit_av_TM2 = dgl_auto_TM2['best_ps_fit_av']

    AC_A1_TM1 = dgl_auto_TM1['AC_A1']
    dAC_sq_TM1 = dgl_auto_TM1['dAC_sq']

    AC_A1_TM2 = dgl_auto_TM2['AC_A1']
    dAC_sq_TM2 = dgl_auto_TM2['dAC_sq']

    if verbose:
        print('AC A1 values:', AC_A1_TM1, AC_A1_TM2)
        print('dAC A1 values:', np.sqrt(dAC_sq_TM1), np.sqrt(dAC_sq_TM2))
        
    AC_A1_cross = (AC_A1_TM1*AC_A1_TM2)

    best_ps_fit_av = np.sqrt(best_ps_fit_av_TM1*best_ps_fit_av_TM2)*AC_A1_cross
    
    return lb_modl, best_ps_fit_av, AC_A1_cross, dAC_sq_TM1, dAC_sq_TM2

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

	if dgl_mode=='wise_12micron':
		regrid_fpath = base_path + dgl_mode+'_reproj_TM'+str(inst)+'_ifield'+str(ifield)+'.fits'
	else:
		regrid_fpath = base_path + dgl_mode+'_regrid_ifield'+str(ifield)+'_TM'+str(inst)+'_fromhp_nside=2048_120423.fits'

	print('regrid fpath is ', regrid_fpath)
	regrid_map = fits.open(regrid_fpath)[1].data

	if dgl_mode != 'IRIS' and dgl_mode != 'wise_12micron':
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


def load_weighted_cl_file(cl_fpath, mode='observed', cltype='auto'):

	clfile = np.load(cl_fpath, allow_pickle=True)
	
	if mode=='observed':
		observed_recov_ps = clfile['observed_recov_ps']
		observed_recov_dcl_perfield = clfile['observed_recov_dcl_perfield']
		observed_field_average_cl = clfile['observed_field_average_cl']
		observed_field_average_dcl = clfile['observed_field_average_dcl']

		if cltype=='auto':
			field_average_error = clfile['field_average_error']
		else:
			field_average_error = None 
		lb = clfile['lb']
	
		return lb, observed_recov_ps, observed_recov_dcl_perfield, observed_field_average_cl, observed_field_average_dcl, None, field_average_error
	
	elif mode=='mock':
		mock_mean_input_ps = clfile['mock_mean_input_ps']
		mock_all_field_averaged_cls = clfile['mock_all_field_averaged_cls']
		mock_all_field_cl_weights = clfile['mock_all_field_cl_weights']
		all_mock_recov_ps = clfile['all_mock_recov_ps']
		all_mock_signal_ps = clfile['all_mock_signal_ps']
		lb = clfile['lb']
	
		return lb, mock_mean_input_ps, mock_all_field_averaged_cls, mock_all_field_cl_weights, all_mock_recov_ps, all_mock_signal_ps



def load_ciber_spitzer_ebl_fluc_results(maglims = [17.5, 17.0], inst_list = [1, 2], bandstrs = ['J', 'H'], \
                                       irac_ch_list = [1, 2], tailstr='newcal_conserve_flux', observed_run_name_cross=None):
    all_compsub_cl, all_compsub_dcl = [], []
    
    cl_base_path = config.ciber_basepath+'data/input_recovered_ps/cl_files/'
    cl_spitz_basepath = config.ciber_basepath+'data/input_recovered_ps/ciber_spitzer/'


    for idx, inst in enumerate(inst_list):
    
        clfile = np.load(cl_base_path+'TM'+str(inst)+'/cl_observed_'+bandstrs[idx]+'lt'+str(maglims[idx])+'_012524_ukdebias.npz')

        lb = clfile['lb']
        fieldav_cl = clfile['observed_field_average_cl']
        fieldav_dcl = clfile['observed_field_average_dcl']
        fieldav_cl_sub, mean_cl_sn = remove_igl_dgl_clus(lb, fieldav_cl)

        all_compsub_cl.append(fieldav_cl_sub)
        all_compsub_dcl.append(fieldav_dcl)
        
    # CIBER x CIBER cross
    all_compsub_cl_cross, all_compsub_dcl_cross = [], []
    maglim_J = maglims[0]

    if observed_run_name_cross is None:
        observed_run_name_cross = 'ciber_cross_ciber_Jlt'+str(maglim_J)+'_Hlt'+str(maglim_J-0.5)+'_020724_interporder2'
    

    cl_fpath_obs = cl_base_path+'TM1_TM2_cross/cl_'+observed_run_name_cross+'.npz'
    lb, observed_recov_ps, observed_recov_dcl_perfield,\
        observed_field_average_cl, observed_field_average_dcl,\
            _, _ = load_weighted_cl_file(cl_fpath_obs, cltype='cross')

    fieldav_clx_sub, mean_cl_sn_x = remove_igl_dgl_clus(lb, observed_field_average_cl)
    all_compsub_cl_cross.append(fieldav_clx_sub)
    all_compsub_dcl_cross.append(observed_field_average_dcl)     
    
    for spitzidx, irac_ch in enumerate(irac_ch_list):
        auto_file = np.load(cl_spitz_basepath+'ciber_spitzer_cross_auto_cl_TM1_IRACCH'+str(irac_ch)+'_'+tailstr+'.npz')
        lb = auto_file['lb']
        pf = lb*(lb+1)/(2*np.pi)

        fieldav_cl_spitzer = np.array(auto_file['fieldav_cl_crossep'])
        fieldav_dcl_spitzer = np.array(auto_file['fieldav_clerr_crossep'])

        fieldav_cl_spitz_sub, mean_cl_sn_spitz = remove_igl_dgl_clus(lb, fieldav_cl_spitzer)


        all_compsub_cl.append(fieldav_cl_spitz_sub)
        all_compsub_dcl.append(fieldav_dcl_spitzer)

        # Spitzer x CIBER points
        for idx, inst in enumerate(inst_list):
            cross_file = np.load(cl_spitz_basepath+'ciber_spitzer_cross_auto_cl_TM'+str(inst)+'_IRACCH'+str(irac_ch)+'_'+tailstr+'.npz')
            fieldav_cl_cross = cross_file['fieldav_cl_cross']
            fieldav_clerr_cross = cross_file['fieldav_clerr_cross']

            fieldav_cl_spitz_cross_sub, mean_cl_sn_spitz_cross = remove_igl_dgl_clus(lb, fieldav_cl_cross)

            all_compsub_cl_cross.append(fieldav_cl_spitz_cross_sub)
            all_compsub_dcl_cross.append(fieldav_clerr_cross)
            
    return lb, all_compsub_cl, all_compsub_dcl, all_compsub_cl_cross, all_compsub_dcl_cross 


def remove_igl_dgl_clus(lb, fieldav_cl, lmin_sn=50000, subtract_dgl=False, endidx=-1):
    
    # load power spectra
    
    # fit to the shot noise

    lbsnmask = (lb >= lmin_sn)*(lb < lb[endidx])
    mean_cl_sn = np.mean(fieldav_cl[lbsnmask])
    
    fieldav_cl_sub = fieldav_cl-mean_cl_sn
    
    # subtract DGL contribution if included
    
    if subtract_dgl:
        dgl_modl = ''
        fieldav_cl_sub -= dgl_modl
    
    return fieldav_cl_sub, mean_cl_sn

def load_mkk_per_quadrant(mode_couple_base_dir, mask_tail, mkk_type, ifield, inst, q):
	inv_Mkk_fpaths_per_quadrant, inv_Mkks_indiv = [], []
	for q in range(4):
		inv_Mkk_fpaths_per_quadrant.append(mode_couple_base_dir+'/'+mask_tail+'/mkk_'+mkk_type+'_ifield'+str(ifield)+'_inst'+str(inst)+'_observed_'+mask_tail+'_quad'+str(q)+'.fits')
		inv_Mkks_indiv.append(fits.open(inv_Mkk_fpaths_per_quadrant[q])['inv_Mkk_'+str(ifield)].data)
	inv_Mkks_per_quadrant.append(inv_Mkks_indiv)

	return inv_Mkks_per_quadrant


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


def load_noisemodl_validation_dat(inst, ifield, nvdir, tailstr):

	clfile = np.load(nvdir+'powerspec/cls_expdiff_lab_flight_TM'+str(inst)+'_ifield'+str(ifield)+'_'+tailstr+'.npz')

	lb = clfile['lb']
	cl_diff_flight = clfile['cl_diff_flight']
	clerr_diff_flight = clfile['clerr_diff_flight']
	cl_diffs_lab = clfile['cl_diffs_darkexp']
	cl_diff_noisemodl = clfile['cl_diff_noisemodl']
	clerr_diff_noisemodl = clfile['clerr_diff_noisemodl']
	cl_diffs_phot = clfile['cl_diffs_phot']
	nl1ds_diff_weighted = clfile['nl1ds_diff_weighted']/2. # from noise model
	nl1ds_diff = clfile['nl1ds_diff'] # already divided by two in compute_lab_flight_exp_differences()
	fourier_weights = clfile['fourier_weights']

	nv_dict = dict({'lb':lb, 'cl_diff_flight':cl_diff_flight, 'clerr_diff_flight':clerr_diff_flight, 'cl_diffs_lab':cl_diffs_lab, \
					'cl_diff_noisemodl':cl_diff_noisemodl, 'clerr_diff_noisemodl':clerr_diff_noisemodl, 'cl_diffs_phot':cl_diffs_phot, \
					'nl1ds_diff_weighted':nl1ds_diff_weighted, 'nl1ds_diff':nl1ds_diff, 'fourier_weights':fourier_weights})

	return nv_dict



def load_cosmos_snpred(band):
	# COSMOS 15 predictions and color corrections
	cl_pred_basepath = config.ciber_basepath+'data/cl_predictions/'
	c15_auto_pred = np.load(cl_pred_basepath+'COSMOS15/auto/cl_predictions_auto_JH_CH1CH2.npz')
	c15_auto_cl = c15_auto_pred['all_clautos']
	c15_m_min_list = c15_auto_pred['m_min_'+band+'_list']
	c15_lb=c15_auto_pred['lb']
	c15_pf=c15_auto_pred['pf']/c15_lb
	
	c15_ccorr = np.load(cl_pred_basepath+'color_corr_vs_'+band+'_min_cosmos15_nuInu.npz')
	mmin_range_cc = c15_ccorr['mmin_list']
	all_pv_c15 = c15_ccorr['all_pv_c15']
	
	return c15_lb, cl15_m_min_list, all_pv_c15, mmin_range_cc, c15_ccorr

def load_2MASS_snpred(inst):
	
	snpred_ciber_bright = np.load(config.ciber_basepath+'data/cl_predictions/snpred_vs_mag_TM'+str(inst)+'_2MASS_nuInu.npz')
	mmin_range_bright = snpred_ciber_bright['mmin_list']
	av_pv_bright = snpred_ciber_bright['av_pv']

	all_pv_bright = np.array(snpred_ciber_bright['all_pv'])
	
	return mmin_range_bright, all_pv_bright, av_pv_bright


def load_filter_curves(banddir = 'data/bands/', psbands = ['g', 'r', 'i', 'z', 'y'], twomass_bands=['J', 'H', 'Ks']):
    

    all_filter_lam, all_filter_T = [], []
    # CIBER filters

    for inst in [1, 2]:
        if inst==1:
            bandstr = 'J'
            lam_eff = 1.05
            jband_transmittance = np.loadtxt(banddir+'iband_transmittance.txt', skiprows=1)
            ciber_filt_lam = jband_transmittance[:,0]*1e-3
            ciber_filt_T = jband_transmittance[:,1]

        elif inst==2:
            bandstr = 'H'
            lam_eff = 1.79
            hband_transmittance = np.loadtxt(banddir+'/hband_transmittance.txt', skiprows=1)
            ciber_filt_lam = hband_transmittance[:,0]*1e-3
            ciber_filt_T = hband_transmittance[:,1]

        ciber_filt_T /= np.max(ciber_filt_T)

        all_filter_lam.append(ciber_filt_lam)
        all_filter_T.append(ciber_filt_T)

        if inst==1:
              # add H-band filter from flights 2/3
            hband_earlier = np.loadtxt('data/bands/hband_transmittance_flights_2_3.csv', delimiter=',', dtype=float)
            hband_wav = hband_earlier[:,0]
            hband_T = hband_earlier[:,1]
            hband_T /= np.max(hband_T)

            all_filter_lam.append(hband_wav*1e-3)
            all_filter_T.append(hband_T)


    for psband in psbands:
        ps = np.loadtxt(banddir+'PS_'+psband+'_filter.txt')

        all_filter_lam.append(ps[:,0]*1e-4)
        all_filter_T.append(ps[:,1]/np.max(ps[:,1]))


    # 2MASS filters

    for bandstr in twomass_bands:

        twom = np.loadtxt(banddir+'2MASS_'+bandstr+'_transmission.txt')

        all_filter_lam.append(twom[:,0])
        all_filter_T.append(twom[:,1]/np.max(twom[:,1]))


    # IRAC channels

    for ch in [1, 2]:
        irac = np.loadtxt(banddir+'IRAC_CH'+str(ch)+'_transmission.txt')

        all_filter_lam.append(irac[:,0])
        all_filter_T.append(irac[:,1]/np.max(irac[:,1]))
        
    return all_filter_lam, all_filter_T


''' File directory structure '''

def init_mocktest_fpaths(ciber_mock_fpath, run_name, verbose=False):
	ff_fpath = ciber_mock_fpath+'030122/ff_realizations/'+run_name+'/'
	noisemod_fpath = ciber_mock_fpath +'030122/noise_models_sim/'+run_name+'/'
	input_recov_ps_fpath = 'data/input_recovered_ps/sim_tests_030122/'+run_name+'/'

	fpaths = [input_recov_ps_fpath, ff_fpath, noisemod_fpath]
	for fpath in fpaths:
		if not os.path.isdir(fpath):
			print('making directory path for ', fpath)
			os.makedirs(fpath)
		else:
			if verbose:
				print(fpath, 'already exists')

	return ff_fpath, noisemod_fpath, input_recov_ps_fpath

def make_fpaths(fpaths, verbose=False):
	for fpath in fpaths:
		if not os.path.isdir(fpath):
			print('making directory path for ', fpath)
			os.makedirs(fpath)
		else:
			if verbose:
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


def write_regrid_spitzer_proc_file(ifield, inst, irac_ch, mask_tail=None, sdwfs_epoch_diffs=None, sdwfs_epoch_av=None, sdwfs_per_epoch=None, \
									 grad_sub=True, save=True, tailstr=None, spitzer_basepath=None, coverage_map=None):
	

	if spitzer_basepath is None and save:
		spitzer_basepath = config.ciber_basepath+'data/Spitzer/spitzer_regrid/proc/TM'+str(inst)+'/IRAC_CH'+str(irac_ch)+'/'
	
	if mask_tail is not None:
		spitzer_basepath += mask_tail+'/'

		if not os.path.exists(spitzer_basepath):
			print('making directory ', spitzer_basepath)
			os.makedirs(spitzer_basepath)

	hdup = fits.PrimaryHDU()
	hdup.header['ifield'] = ifield
	hdup.header['inst'] = inst
	hdup.header['irac_ch'] = irac_ch
	hdup.header['grad_sub'] = grad_sub
	hdulist = [hdup]

	if sdwfs_epoch_diffs is not None:

		for diffidx in np.arange(2):
			hdu_epoch_diffs = fits.ImageHDU(sdwfs_epoch_diffs[diffidx], name='diff'+str(diffidx))
			hdulist.append(hdu_epoch_diffs)

	if sdwfs_per_epoch is not None:
		for epochidx in range(len(sdwfs_per_epoch)):
			hdu_per_epoch = fits.ImageHDU(sdwfs_per_epoch[epochidx], name='epoch_'+str(epochidx))
			hdulist.append(hdu_per_epoch)

	if sdwfs_epoch_av is not None:
		hdum = fits.ImageHDU(sdwfs_epoch_av, name='epoch_av')
		hdulist.append(hdum)

	if coverage_map is not None:
		hdum = fits.ImageHDU(coverage_map, name='coverage_map')
		hdulist.append(hdum)

	hdul = fits.HDUList(hdulist)

	if save:
		save_fpath = spitzer_basepath + 'spitzer_regrid_TM'+str(inst)+'_IRAC_CH'+str(irac_ch)+'_ifield'+str(ifield)
		if tailstr is not None:
			save_fpath += '_'+tailstr
		hdul.writeto(save_fpath+'.fits', overwrite=True)

	return save_fpath


def write_regrid_spitzer_raw_file(ifield, inst, irac_ch, spitzer_regrid_per_epoch=None, spitzer_regrid_epoch_av=None, \
							spitzer_basepath=None, save=True, tailstr=None):

	if spitzer_basepath is None and save:
		spitzer_basepath = config.ciber_basepath+'data/Spitzer/spitzer_regrid/raw/TM'+str(inst)+'/IRAC_CH'+str(irac_ch)+'/'
	
	hdup = fits.PrimaryHDU()
	hdup.header['ifield'] = ifield
	hdup.header['inst'] = inst
	hdup.header['irac_ch'] = irac_ch

	hdulist = [hdup]

	if spitzer_regrid_per_epoch is not None:
		for epochidx in range(4):
			hdum = fits.ImageHDU(spitzer_regrid_per_epoch[epochidx], name='regrid_epoch_'+str(epochidx))
			hdulist.append(hdum)

	if spitzer_regrid_epoch_av is not None:
		hdum = fits.ImageHDU(spitzer_regrid_epoch_av, name='regrid_epoch_av')
		hdulist.append(hdum)

	hdul = fits.HDUList(hdulist)

	if save:
		save_fpath = spitzer_basepath + 'spitzer_regrid_TM'+str(inst)+'_IRAC_CH'+str(irac_ch)+'_ifield'+str(ifield)
		if tailstr is not None:
			save_fpath += '_'+tailstr
		hdul.writeto(save_fpath+'.fits', overwrite=True)

		return save_fpath, hdul
	else:
		return hdul

def write_regrid_proc_file(masked_proc, ifield, inst, regrid_to_inst, mask_tail=None,\
						   dat_type='observed', mag_lim=None, mag_lim_cross=None, obs_level=None, masked_proc_orig=None):
	hdum = fits.ImageHDU(masked_proc, name='proc_regrid_'+str(ifield))

	if masked_proc_orig is not None:
		hdu_orig = fits.ImageHDU(masked_proc_orig, name='proc_orig_'+str(ifield))
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

	hdulist = [hdup, hdum]
	if masked_proc_orig is not None:
		hdulist.append(hdu_orig)

	hdul = fits.HDUList(hdulist)
	
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

# def write_map_file(map, ifield, inst, dat_type=None):


def write_mask_file(mask, ifield, inst, cross_inst=None, sim_idx=None, generate_galmask=None, generate_starmask=None, use_inst_mask=None, \
				   dat_type=None, mag_lim_AB=None, with_ff_mask=None, name=None, a1=None, b1=None, c1=None, dm=None, alpha_m=None, beta_m=None, \
				   include_keywords=True):

	if name is None:
		name = 'joint_mask_'+str(ifield)
	hdum = fits.ImageHDU(mask, name=name)

	hdup = fits.PrimaryHDU()

	if include_keywords:
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

	# else:
	#     hdul = fits.HDUList([hdum])
	
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



def save_weighted_cl_file(lb, inst, observed_run_name, observed_recov_ps, observed_recov_dcl_perfield, observed_field_average_cl, observed_field_average_dcl, \
				mock_all_field_cl_weights, cl_base_path=config.ciber_basepath+'data/input_recovered_ps/cl_files/', cross_inst=None, field_average_error=None):
	
	if cross_inst is None:
		cl_fpath = cl_base_path+'TM'+str(inst)+'/cl_'+observed_run_name+'.npz'
	else:
		cl_fpath = cl_base_path+'TM'+str(inst)+'_TM'+str(cross_inst)+'_cross/cl_'+observed_run_name+'.npz'

	print('saving to ', cl_fpath)
	np.savez(cl_fpath, \
			lb=lb, inst=inst, observed_recov_ps=observed_recov_ps, observed_recov_dcl_perfield=observed_recov_dcl_perfield, \
			observed_field_average_cl=observed_field_average_cl, observed_field_average_dcl=observed_field_average_dcl, \
			mock_all_field_cl_weights=mock_all_field_cl_weights, field_average_error=field_average_error)
	
	return cl_fpath

def save_weighted_mock_cl_file(lb, inst, mock_run_name, mock_mean_input_ps, mock_all_field_averaged_cls, mock_all_field_cl_weights, \
							  all_mock_recov_ps, all_mock_signal_ps, cl_base_path=config.ciber_basepath+'data/input_recovered_ps/cl_files/'):
	
	
	cl_fpath = cl_base_path+'TM'+str(inst)+'/cl_'+mock_run_name+'.npz'

	print('saving to ', cl_fpath)
	np.savez(cl_fpath, \
			lb=lb, inst=inst, mock_mean_input_ps=mock_mean_input_ps, mock_all_field_averaged_cls=mock_all_field_averaged_cls, \
			mock_all_field_cl_weights=mock_all_field_cl_weights, all_mock_recov_ps=all_mock_recov_ps, \
			all_mock_signal_ps=all_mock_signal_ps)
	
	return cl_fpath

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


