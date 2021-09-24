import numpy as np
from mkk_parallel import *



# class powerspec_plot():


# 	def __init__(self, powerspec_obj=None, n_ps_bin=25, dimx=1024, \
#                 dimy=1024):
# 		if powerspec_obj is None:
# 			powerspec_obj = powerspec(n_ps_bin=n_ps_bin, dimx=dimx, dimy=dimy)

# 		self.powerspec_obj = powerspec_obj

# 	def plot_mock_truth_vs_recovered(self, return_fig=True, show=False):


# 		f = plt.figure()


# 		if show:
# 			plt.show()
# 		if return_fig:
# 			return f


# 	def plot_observed_ps(self, xlim=None, ylim=None, xscale='log', yscale='log', ff_correct=False, return_fig=True, text=None, show=False):


# 		f = plt.figure()



# 		if ff_correct:


# 		if text is not None:


# 		plt.xscale(xscale)
# 		plt.yscale(yscale)
# 		plt.xlim(xlim)
# 		plt.ylim(ylim)

# 		if show:
# 			plt.show()
# 		if return_fig:
# 			return f



def compute_sqrt_cl_errors(cl, dcl):
    sqrt_cl_errors = 0.5*(dcl*cl**(-0.5))
    return sqrt_cl_errors



class powerspec():

	def __init__(self, n_ps_bin=25, dimx=1024, \
				dimy=1024):

		self.Mkk_obj = Mkk_bare(dimx=dimx, dimy=dimy, ell_min=180.*(1024./dimx), nbins=n_ps_bin)


	def load_in_powerspec(self, powerspec_fpath, mode=None, inplace=False):

		# if inplace:
		# 	if mode==''
		# 	self.ps = powerspec

		# else:
		# 	return powerspec

		pass


	def compute_est_true_ratio(self, ps_measured=None, ps_ground_truth=None, inplace=True):

		''' Computes ratio between power spectra 

		Inputs
		------
		
		ps_measured (optional): np.array of shape (nfields, n_ps_bin), (nsims, n_ps_bin) or (nsims, nfields, n_ps_bin). 
								Default is None.

		ps_ground_truth (optional): np.array of shape (nfields, n_ps_bin), (nsims, n_ps_bin) or (nsims, nfields, n_ps_bin). 
								Default is None.	

		inplace : type 'boolean'. If True, stores power spectrum ratio in object. Default is True. 	

		Returns
		-------

		est_true_ratio : np.array of shape (nfields, n_ps_bin), (nsims, n_ps_bin) or (nsims, nfields, n_ps_bin). 
						Ratio of (sets of) power spectra.

		'''
		if ps_measured is None:
			ps_measured = self.ps_measured
		if ps_ground_truth is None:
			ps_ground_truth = self.ps_ground_truth

		est_true_ratio = ps_measured/ps_ground_truth

		if inplace:
			self.est_true_ratio = est_true_ratio
		else:
			return est_true_ratio


	def grab_all_simidx_dat(self, fpaths, mean_or_median='mean', inter_idx=None, per_field=True, bls=None, inplace=True):

		nsims = len(fpaths)

		# here, "mean" power spectrum refers to average over fields
		est_true_ratios = np.zeros((nsims, nfield, self.n_ps_bin))
		all_mean_true_cls = np.zeros((nsims, self.n_ps_bin))
		all_true_cls = np.zeros((nsims, nfield, self.n_ps_bin))
		all_recovered_cls = np.zeros((nsims, nfield, self.n_ps_bin))
		all_mean_recovered_cls = np.zeros((nsims, self.n_ps_bin))


		for f, fpath in enumerate(fpaths):
			lb, recovered_cls, true_cls, mean_recovered_cls, mean_true_cl, est_true_ratio = grab_recovered_cl_dat(fpath, mean_or_median=mean_or_median, inter_idx=inter_idx, \
																												  per_field=per_field, bls=bls)

			est_true_ratios[f] = est_true_ratio
			all_recovered_cls[f] = recovered_cls
			all_true_cls[f] = true_cls
			all_mean_recovered_cls[f] = mean_recovered_cls

		if inplace:
			self.est_true_ratios = est_true_ratios
			self.all_mean_true_cls = all_mean_true_cls
			self.all_recovered_cls = all_recovered_cls
			self.all_true_cls = all_true_cls
			self.lb = lb 

		else:
			return lb, est_true_ratios, all_mean_true_cls, all_recovered_cls, all_recovered_mean_cls, all_true_cls


	# def compute_field_averaged_power_spectrum(self, per_field_cls, per_field_dcls=None, per_field_cl_weights=None, weight_mode='invvar', obj_verbose=True, verbose=True):
		
	# 	'''
	# 	Inputs
	# 	------
		
	# 	lb : np.array or list of multipole bin centers
	# 	per_field_cls : np.array of shape (nfields, nbins)
	# 	per_field_cls : np.array of shape (nfields, nbins)
		
	# 	Returns
	# 	-------
		
	# 	field_averaged_cl :
	# 	field_averaged_std :
	# 	cl_sumweights : 
	# 	per_field_cl_weights : 
		
	# 	'''
		
	# 	if per_field_cl_weights is None:
	# 		per_field_cl_weights = np.zeros_like(per_field_cls)
		
	# 		if weight_mode=='invvar':
	# 			print(per_field_cl_weights.shape, per_field_cls.shape)
	# 			per_field_cl_weights = per_field_dcls**(-2)
	# 			per_field_cl_weights[per_field_cls < 0.] = 0.



	# 	cl_sumweights = np.sum(per_field_cl_weights, axis=0)

	# 	weighted_variance = 1./cl_sumweights
	# 	field_averaged_std = np.sqrt(weighted_variance) # weighted mean has uncertainty equal to 1/sqrt(sum of weights)
		
	# 	per_field_cls[per_field_cls < 0] = 0
		
	# 	nfields_per_bin = np.count_nonzero(per_field_cls, axis=0)
		
	# 	per_field_cls[per_field_cls == 0] = np.nan
				
	# 	field_averaged_cl = np.nansum(per_field_cl_weights*per_field_cls, axis=0)/cl_sumweights

	# 	if obj_verbose:
	# 		print('per_field_cl_weights has shape ', per_field_cl_weights.shape)
	# 		print('per_field_cls has shape ', per_field_cls.shape)
	# 		print('cl_sumweights has shape ', cl_sumweights.shape)
	# 		print('field averaged std has shape ', field_averaged_std.shape)
	# 		print('field averaged cl has shape ', field_averaged_cl.shape)

	# 	if verbose:
	# 		print("per field cl weights ", per_field_cl_weights)
	# 		print('cl_sumweights:', cl_sumweights)
	# 		print(nfields_per_bin)
	# 		print('field_averaged_cl is ', field_averaged_cl)
	# 		print('field averaged std is', field_averaged_std)

	# 	if inplace:
	# 		self.field_averaged_cl = field_averaged_cl
	# 		self.field_averaged_std = field_averaged_std
	# 		self.cl_sumweights = cl_sumweights
	# 		self.per_field_cl_weights = per_field_cl_weights
	# 	else:
	# 		return field_averaged_cl, field_averaged_std, cl_sumweights, per_field_cl_weights


def compute_powerspectra_realdat(inst, n_ps_bin=25, J_mag_lim=17.5, ifield_list=[4, 5, 6, 7, 8], \
                                n_mkk_sims=50, n_split_mkk=1, n_FW_sims=50, n_FW_split=1, ff_stack_min=1, \
                                show_plots=False, g1_fac_dict=None, g2_fac_dict=None, inverse_Mkks=None, \
                                base_fluc_path='/Volumes/Seagate Backup Plus Drive/Toolkit/Mirror/Richard/ciber_fluctuation_data/', \
                                use_ff_est=True, stdpower=2):
    nfields = len(ifield_list)
    B_ells = np.zeros((nfields, n_ps_bin))
    readnoise_cls = np.zeros((nfields, n_ps_bin))
    observed_ims, _, noise_models, shot_sigma_sb_maps, _, joint_masks, _ = instantiate_dat_arrays_fftest(1024, 1024, 5)

    cbps = CIBER_PS_pipeline(n_ps_bin=n_ps_bin)

    
    ff_estimates = np.zeros((nfields, cbps.dimx, cbps.dimy))


    if g1_fac_dict is not None:
        cbps.g1_facs = g1_fac_dict
   
    if g2_fac_dict is not None:
        cbps.g2_facs = g2_fac_dict
        cbps.cal_facs = dict({1:cbps.g1_facs[1]*cbps.g2_facs[1], 2:cbps.g1_facs[2]*cbps.g2_facs[2]}) # updated 


    print('cbps cal facs are ', cbps.cal_facs)
    print('cbps cal facs are ', cbps.g1_facs)


    for i, ifield in enumerate(ifield_list):
        

        cbps.load_data_products(ifield, inst, verbose=True)

        print('cbps.B_ell is ', cbps.B_ell)
        
        cbps.load_noise_Cl2D(noise_fpath='data/fluctuation_data/TM'+str(inst)+'/readCl_input_dr20210611/field'+str(ifield)+'_readCl2d_input.fits')
        observed_ims[i] = cbps.image*cbps.cal_facs[inst]
        
        plot_map(observed_ims[i], title='obsreved ims i='+str(i))
        noise_models[i] = cbps.noise_Cl2D
        shot_sigma_sb_maps[i] = cbps.compute_shot_sigma_map(inst, image=observed_ims[i], nfr=cbps.field_nfrs[ifield])

        shot_sigma_sb_maps[i] = np.median(shot_sigma_sb_maps[i])*np.ones_like(observed_ims[i])
        
        
        plt.figure(figsize=(6, 5))
        plt.title(cbps.ciber_field_dict[ifield], fontsize=18)
        plt.hist(shot_sigma_sb_maps[i].ravel(), bins=30)
        plt.yscale('log')
        plt.show()
#         mask_fpathprev = cbps.data_path+'/TM'+str(inst)+'/mask/field'+str(ifield)+'_TM'+str(inst)+'_strmask_Jlim='+str(J_mag_lim)+'_040521.fits'
#         instm = cbps.load_mask(ifield, inst, mask_fpath=mask_fpathprev, masktype='strmask', inplace=False)
#         jm = np.array(cbps.maskInst_clean*instm)
#         joint_masks[i] = np.array(cbps.maskInst_clean*cbps.strmask)

    
        mask_fpath = base_fluc_path+'/TM'+str(inst)+'/masks/joint_mask_with_ffmask_ifield'+str(ifield)+'_inst'+str(inst)+'.fits'
        joint_masks[i] = fits.open(mask_fpath)['joint_mask_'+str(ifield)].data
        
        ff_fpath = base_fluc_path+'/TM'+str(inst)+'/stackFF/ff_estimate_ifield'+str(ifield)+'_inst'+str(inst)+'_Jlim.fits'
        ff_estimates[i] = fits.open(ff_fpath)['ff_'+str(ifield)].data
        
        plot_map(joint_masks[i], title='joinjt masks with ff')
        plot_map(ff_estimates[i], title='ff estimate')
            

        B_ells[i] = cbps.B_ell
    
        
    plt.figure()
    for i, ifield in enumerate(ifield_list):
        plt.plot(cbps.Mkk_obj.midbin_ell, B_ells[i], label=cbps.ciber_field_dict[ifield], color='C'+str(i))
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    
    if not use_ff_est:
        ff_estimates = None
    ff_estimates, inverse_Mkks, lb, recovered_cls, masked_images, masked_nls, masked_nls_noff, cls_intermediate = calculate_powerspec_quantities(cbps, observed_ims, joint_masks, shot_sigma_sb_maps, noise_models, ifield_list, include_inst_noise=True, \
                                                                                              n_mkk_sims=n_mkk_sims, n_split_mkk=n_split_mkk, n_FW_sims=n_FW_sims, n_FW_split=n_FW_split, \
                                                                                                 ff_stack_min=1, B_ells=B_ells, show_plots=show_plots, inverse_Mkks=inverse_Mkks, ff_estimates=ff_estimates, stdpower=stdpower)


    return ff_estimates, inverse_Mkks, lb, recovered_cls, masked_images, masked_nls, masked_nls_noff, cls_intermediate, lb
    
   


def compute_field_averaged_power_spectrum(per_field_cls, per_field_dcls=None, per_field_cl_weights=None, weight_mode='invvar', \
                                         which_idxs=None, verbose=False):
    
    '''
    Inputs
    ------
    
    lb : np.array or list of multipole bin centers
    per_field_cls : np.array of shape (nfields, nbins)
    per_field_dcls (optional) : np.array of shape (nfields, nbins)
    per_field_cl_weights (optional) : np.array of shape (nfields, nbins)
    weight_mode (optional) : 'str'. If set to 'invvar', computes inverse variance weights using per_field_dcls.
            Default is 'invvar'.
    which_idxs (optional) : 
    verbose (optional) : 'bool'. 
    
    Returns
    -------
    
    field_averaged_cl :
    field_averaged_std :
    cl_sumweights : 
    per_field_cl_weights : 
    
    '''
    
    if which_idxs is not None:
        which_idxs = np.array(which_idxs)
        per_field_cls = per_field_cls[which_idxs,:]
        if per_field_dcls is not None:
            per_field_dcls = per_field_dcls[which_idxs,:]
    
    if per_field_cl_weights is None:
        per_field_cl_weights = np.zeros_like(per_field_cls)
    
        if weight_mode=='invvar':
            if verbose:
                print(per_field_cl_weights.shape, per_field_cls.shape)
            per_field_cl_weights = per_field_dcls**(-2)
            per_field_cl_weights[per_field_cls < 0.] = 0.
                
    else:
        if which_idxs is not None:
            per_field_cl_weights = per_field_cl_weights[which_idxs,:]
        
    cl_sumweights = np.sum(per_field_cl_weights, axis=0)
    if verbose:
        print("per field cl weights ", per_field_cl_weights)
        print('cl_sumweights has shape', cl_sumweights.shape)
        print('cl_sumweights:', cl_sumweights)
    
    weighted_variance = 1./cl_sumweights
    
    field_averaged_std = np.sqrt(weighted_variance)
    
    per_field_cls[per_field_cls < 0] = 0
    
    nfields_per_bin = np.count_nonzero(per_field_cls, axis=0)
    
    per_field_cls[per_field_cls == 0] = np.nan
    field_averaged_cl = np.nansum(per_field_cl_weights*per_field_cls, axis=0)/cl_sumweights

    if verbose:
        print('at this step')
        print(per_field_cl_weights*per_field_cls)
        print('field_averaged_cl is ', field_averaged_cl)
        print('field averaged std is', field_averaged_std)
    
    return field_averaged_cl, field_averaged_std, cl_sumweights, per_field_cl_weights



def get_observed_power_spectra(est_fpath=None, recov_obs_fpath=None, recov_mock_fpath=None,\
                               recovered_cls_obs=None, est_true_ratios=None, \
                               all_recovered_cls=None, mean_recovered_cls=None, std_recovered_cls=None, \
                               apply_correction=True, skip_elat30=False):
    if est_fpath is not None:
        est_true_ratios = np.load(est_fpath)['est_true_ratios']
    if recov_obs_fpath is not None:
        recov_observed = np.load(recov_obs_fpath)
        recovered_cls_obs = recov_observed['recovered_cls']
        
    if recov_mock_fpath is not None:
        recov_full = np.load(recov_mock_fpath)
        all_recovered_cls = recov_full['all_recovered_cls']
        mean_recovered_cls = recov_full['mean_recovered_cls']
        std_recovered_cls = recov_full['std_recovered_cls']
    
    recovered_cls_obs_corrected = np.zeros_like(recovered_cls_obs)
    recovered_dcls_obs_corrected = np.zeros_like(recovered_cls_obs)

    if skip_elat30:
        which_idxs = [0, 2, 3, 4]
    else:
        which_idxs = np.arange(5)

    print('which idxs is ', which_idxs)
        
    for i in range(5):
                
        recovered_cls_obs_corrected[i] = recovered_cls_obs[i]
        recovered_dcls_obs_corrected[i] = std_recovered_cls[i]
        if apply_correction:
            print(np.mean(est_true_ratios[:,i,:], axis=0))
            recovered_cls_obs_corrected[i] /= np.mean(est_true_ratios[:,i,:], axis=0)
            recovered_dcls_obs_corrected[i] /= np.mean(est_true_ratios[:,i,:], axis=0)
            

    field_averaged_cl, field_averaged_std, cl_sumweights, per_field_cl_weights = compute_field_averaged_power_spectrum(recovered_cls_obs_corrected, per_field_dcls=recovered_dcls_obs_corrected, \
                                                                                                                      which_idxs=which_idxs)
    recovered_cls_obs_corrected[recovered_cls_obs_corrected < 0] = np.nan
    std_cls_basic = np.nanstd(recovered_cls_obs_corrected, axis=0)
    
    return field_averaged_cl, field_averaged_std, cl_sumweights, per_field_cl_weights, recovered_cls_obs_corrected, recovered_dcls_obs_corrected, std_cls_basic
        
 

def compute_flatfield_prods(ifield_list, inst, observed_ims, joint_masks, cbps, show_plots=False, ff_stack_min=1, \
                           inv_var_weight=True, field_nfrs=None):

    
    ff_estimates = np.zeros((len(ifield_list), cbps.dimx, cbps.dimy))
    ff_joint_masks = np.zeros((len(ifield_list), cbps.dimx, cbps.dimy))
    
    
    for i, ifield in enumerate(ifield_list):
        stack_obs = list(observed_ims.copy())
        stack_mask = list(joint_masks.copy().astype(np.bool))

        if field_nfrs is not None:
            ff_field_nfrs = list(np.array(field_nfrs).copy())
        elif cbps.field_nfrs is not None:
            ff_field_nfrs = list(cbps.field_nfrs.copy())
            del(ff_field_nfrs[imidx])
        else:
            ff_field_nfrs = None

        del(stack_obs[i])
        del(stack_mask[i])

        ff_estimate, ff_mask, ff_weights = compute_stack_ff_estimate(stack_obs, target_mask=joint_masks[i], masks=stack_mask, means=None, inv_var_weight=inv_var_weight, ff_stack_min=ff_stack_min, \
                                                                    field_nfrs=ff_field_nfrs)
        ff_estimates[i] = ff_estimate

        sum_stack_mask = np.sum(stack_mask, axis=0)
        
        
        ff_joint_masks[i] = joint_masks[i]*ff_mask

        if show_plots:
            plot_map(ff_estimate)
            sumstackfig = plot_map(sum_stack_mask, title='sum stack mask')
            
    return ff_estimates, ff_joint_masks



def grab_recovered_cl_dat(fpath, mean_or_median='mean', inter_idx=None, per_field=True, bls=None):
    
    ff_test_dat = np.load(fpath, allow_pickle=True)
    
    lb = ff_test_dat['lb']
    
    if inter_idx is not None:
        recovered_cls = [ff_test_dat['cls_intermediate'][j][inter_idx] for j in range(len(ff_test_dat['cls_intermediate']))]
        if recovered_cls[0] is None:
            recovered_cls = [ff_test_dat['cls_intermediate'][j][inter_idx-1] for j in range(len(ff_test_dat['cls_intermediate']))]

    else:
        recovered_cls = ff_test_dat['recovered_cls']
            
    true_cls = ff_test_dat['true_cls']
    
    if recovered_cls is None:
        mean_recovered_cls, mean_true_cls, est_true_ratio = None, None, None                
    else:

        if per_field:   
            if bls is not None:
                true_cls = [true_cls[field_idx]/bls[field_idx]**2 for field_idx in range(recovered_cls.shape[0])]
            est_true_ratio = np.array([recovered_cls[field_idx]/true_cls[field_idx] for field_idx in range(recovered_cls.shape[0])])
            mean_recovered_cls, mean_recovered_cls, mean_true_cls = None, None, None
        else:
        
            if mean_or_median=='median':
                mean_recovered_cls = np.median(recovered_cls, axis=0)
                mean_true_cls = np.median(true_cls, axis=0)
            else:
                mean_recovered_cls = np.mean(recovered_cls, axis=0)
                mean_true_cls = np.mean(true_cls, axis=0)

            est_true_ratio = mean_recovered_cls/mean_true_cls
        

    return lb, recovered_cls, true_cls, mean_recovered_cls, mean_true_cls, est_true_ratio



def grab_all_simidx_dat(fpaths, mean_or_median='mean', inter_idx=None, per_field=True, bls=None):
    est_true_ratios, all_mean_true_cls, all_true_cls, all_recovered_cls, all_recovered_mean_cls = [], [], [], [], []
    
    for fpath in fpaths:
        lb, recovered_cls, true_cls, mean_recovered_cls, mean_true_cl, est_true_ratio = grab_recovered_cl_dat(fpath, mean_or_median=mean_or_median, inter_idx=inter_idx, \
                                                                                                              per_field=per_field, bls=bls)
        est_true_ratios.append(est_true_ratio)
        all_recovered_cls.append(recovered_cls)
        all_true_cls.append(true_cls)
        all_recovered_mean_cls.append(mean_recovered_cls)
        all_mean_true_cls.append(mean_true_cl)

    est_true_ratios = np.array(est_true_ratios)
    all_mean_true_cls = np.array(all_mean_true_cls)
    all_recovered_cls = np.array(all_recovered_cls)
    all_true_cls = np.array(all_true_cls)
    
    return lb, est_true_ratios, all_mean_true_cls, all_recovered_cls, all_recovered_mean_cls, all_true_cls


def instantiate_dat_arrays_fftest(dimx, dimy, nfields):
	''' this function initializes the arrays for the mock pipeline test '''
	observed_ims = np.zeros((nfields, dimx, dimy))
	total_signals = np.zeros((nfields, dimx, dimy))
	noise_models = np.zeros((nfields, dimx, dimy))
	diff_realizations = np.zeros((nfields, dimx, dimy))
	shot_sigma_sb_maps = np.zeros((nfields, dimx, dimy))
	rn_maps = np.zeros((nfields, dimx, dimy))
	joint_masks = np.zeros((nfields, dimx, dimy))
	
	return observed_ims, total_signals, noise_models, shot_sigma_sb_maps, rn_maps, joint_masks, diff_realizations
	
	
def instantiate_cl_arrays_fftest(nfields, n_ps_bin):
	''' this function initializes the arrays for the power spcetra computed at various points of the mock pipeline test'''
	cls_gt = np.zeros((nfields, n_ps_bin))
	cls_diffuse = np.zeros((nfields, n_ps_bin))
	cls_cib = np.zeros((nfields, n_ps_bin))
	cls_postff = np.zeros((nfields, n_ps_bin))
	B_ells = np.zeros((nfields, n_ps_bin))
	
	return cls_gt, cls_diffuse, cls_cib, cls_postff, B_ells

def lin_interp_powerspec(lb_orig, lb_interp, powerspec_orig):
	''' this function uses linear interpolation between one set of multipoles and another tot regrid an input tpower spectrum '''
	powerspec_interp = np.zeros_like(lb_interp)
	for l, lb_int in enumerate(lb_interp):
		# find nearest bins
		lb_diff = np.abs(lb_orig - lb_int)
		arglb_nearest = np.argmin(lb_diff)
		lb_nearest = lb_orig[arglb_nearest]
		
		if lb_nearest > lb_int:
			lb_upper = lb_nearest
			lb_lower = lb_orig[arglb_nearest-1]
			
			cl_lower = powerspec_orig[arglb_nearest-1]
			cl_upper = powerspec_orig[arglb_nearest]

		else:
			lb_lower = lb_nearest
			if arglb_nearest == len(lb_orig)-1:
				continue
			else:
				lb_upper = lb_orig[arglb_nearest+1]
			cl_lower = powerspec_orig[arglb_nearest]
			cl_upper = powerspec_orig[arglb_nearest+1]
			
		slope = (cl_upper-cl_lower)/(lb_upper-lb_lower)

		powerspec_interp[l] = cl_lower + slope*(lb_int-lb_lower)

	return powerspec_interp




def compute_knox_errors(lbins, C_ell, N_ell, delta_ell, fsky=None, B_ell=None, snr=False):
	
	knox_errors = np.sqrt(2./((2*lbins+1)*delta_ell))
	if fsky is not None:
		knox_errors /= np.sqrt(fsky)

	beam_fac = 1.
	if B_ell is not None:
		beam_fac = 1./B_ell**2
		
	knox_errors *= (C_ell + N_ell*beam_fac)

	if snr:
		snr = C_ell / knox_errors
		return snr

	return knox_errors



def compute_sqrt_cl_errors(cl, dcl):
	sqrt_cl_errors = 0.5*(dcl*cl**(-0.5))
	return sqrt_cl_errors



def read_ciber_powerspectra(filename):
	''' Given some file path name, this function loads/parses previously measured CIBER power spectra'''
	array = np.loadtxt(filename, skiprows=8)
	ells = array[:,0]
	norm_cl = array[:,1]
	norm_dcl_lower = array[:,2]
	norm_dcl_upper = array[:,3]
	return np.array([ells, norm_cl, norm_dcl_lower, norm_dcl_upper])


def write_Mkk_fits(Mkk, inv_Mkk, ifield, inst, sim_idx=None, generate_starmask=True, generate_galmask=True, \
                  use_inst_mask=True, dat_type=None, mag_lim_AB=None):
    hduim = fits.ImageHDU(inv_Mkk, name='inv_Mkk_'+str(ifield))        
    hdum = fits.ImageHDU(Mkk, name='Mkk_'+str(ifield))

    hdup = fits.PrimaryHDU()
    hdup.header['ifield'] = ifield
    hdup.header['inst'] = inst
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


def write_mask_file(joint_mask, ifield, inst, sim_idx=None, generate_galmask=None, generate_starmask=None, use_inst_mask=None, \
                   dat_type=None, mag_lim_AB=None, with_ff_mask=None):

    hdum = fits.ImageHDU(joint_mask, name='joint_mask_'+str(ifield))
    hdup = fits.PrimaryHDU()
    hdup.header['ifield'] = ifield
    hdup.header['inst'] = inst
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


    hdul = fits.HDUList([hdup, hdum])
    
    return hdul




    














