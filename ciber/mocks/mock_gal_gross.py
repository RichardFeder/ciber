import numpy as np
import config
from ciber_powerspec_pipeline import *
from powerspec_utils import *
from proc_jmocks import *
from ebl_tom import load_norm_bg_dndz


def gen_mock_ciber_obs(mock_rlz, inst, ifield,
					   apply_zl_gradient=False,
					  with_read_noise=False,
					  read_noise_model=None,
					  with_photon_noise=False,
					  mask_tailstr=None, 
					  galstr='hsc_i_lt_25.0', plot=False, 
					  add_weird_ciber_comp=False, dgl_scale_fac=10., 
					  load_gal_counts=True, perturb_gal_xy=False, dx=0.5, dy=0.5, 
					  with_redshift=False, make_photo_z=True, sigma_zphot=0.05, dndz_phot=None, 
					  load_mask=False, mask_rlz=None, masking_maglim=16.0, ifield_mask=8, \
					  zstr=None, tailstr='CIBERfidmask'):
	
	cbps = CIBER_PS_pipeline()
	
	mockdat = grab_mock_signal_components(mock_rlz, inst=inst, galstr=galstr, load_tracer_cat=perturb_gal_xy, zstr=zstr, load_gal_counts=load_gal_counts, 
										tailstr=tailstr, with_redshift=with_redshift)

	bls_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/beam_correction/bl_est_postage_stamps_TM'+str(inst)+'_081121.npz'
	B_ell = np.load(bls_fpath)['B_ells_post'][ifield-4]

	mockdat['cross_pred'] /= B_ell

	zl_level = cbps.zl_levels_ciber_fields[inst][cbps.ciber_field_dict[ifield]]
	field_nfr = cbps.field_nfrs[ifield]
	
	zl_realiz = generate_zl_realization(zl_level, apply_zl_gradient)
	
	# add to mock map and compute photon noise
	ciber_mock_map = mockdat['intensity_map'] + zl_realiz    
	ciber_mock_obs = ciber_mock_map.copy()

	if perturb_gal_xy:
		tracer_x, tracer_y = mockdat['tracer_x'], mockdat['tracer_y']
		tracer_x_perturb = tracer_x+np.random.normal(0, dx, len(tracer_x))
		tracer_y_perturb = tracer_y+np.random.normal(0, dy, len(tracer_y))

		mockdat['galmap_perturb'] = get_count_field(tracer_x_perturb, tracer_y_perturb, imdim=cbps.dimx)

		if plot:
			print('sum galmap perturb:', np.sum(mockdat['galmap_perturb']))
			print('sum galmap:', np.sum(mockdat['galmap']))
			plot_map(gaussian_filter(mockdat['galmap_perturb'], 20), title='perturbed map')
			plot_map(gaussian_filter(mockdat['galmap'], 20), title='original map')

	maplist_split_shape = (1, cbps.dimx, cbps.dimy)
	empty_aligned_objs, fft_objs = construct_pyfftw_objs(3, maplist_split_shape)

	# add ell^-3 fluctuation component
	if add_weird_ciber_comp:
		
		weird_ciber_comp = cbps.generate_custom_sky_clustering(inst, dgl_scale_fac=dgl_scale_fac)

		if plot:
			plot_map(weird_ciber_comp, title='ell-3 clustering comp')
		ciber_mock_map += weird_ciber_comp
		ciber_mock_obs += weird_ciber_comp

	if with_photon_noise:
		shot_sigma_sb = cbps.compute_shot_sigma_map(inst, image=ciber_mock_map, nfr=field_nfr)
		photon_noise = np.random.normal(0, 1, size=(cbps.dimx, cbps.dimy))*shot_sigma_sb
		ciber_mock_obs += photon_noise

	# add read noise from generated realization
	if with_read_noise:
		
		if read_noise_model is None:
			read_noise_model = cbps.grab_noise_model_set([ifield], inst, noise_modl_type='quadsub_021523')[0]
			
		read_noise, snmaps = cbps.noise_model_realization(inst, maplist_split_shape, read_noise_model, fft_obj=fft_objs[0],\
												  read_noise=True, photon_noise=False, shot_sigma_sb=None)

		ciber_mock_obs += read_noise[0]

	if load_mask:

		mock_mask_dir = config.ciber_basepath+'data/ciber_mocks/112022/TM'+str(inst)+'/'
		mask_basepath = mock_mask_dir+'masks/maglim_'+str(masking_maglim)+'_Vega_081323/'
		maskstr = 'maglim_'+str(masking_maglim)+'_Vega'

		mask_fpath = mask_basepath+'joint_mask_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(mask_rlz)+'_'+maskstr+'_081323.fits'		
		mockdat['mask'] = fits.open(mask_fpath)[1].data
		
		mkk_basepath = mock_mask_dir+'mkk/'+maskstr+'_081323/mkk_maskonly_estimate_ifield'+str(ifield)+'_inst'+str(inst)+'_simidx'+str(mask_rlz)+'_'+maskstr+'_081323.fits'
		mockdat['invmkk'] = fits.open(mkk_basepath)['inv_Mkk_'+str(ifield)].data

		
	mockdat['ciber_mock_obs'] = ciber_mock_obs
	
	return mockdat


def powerspec_from_mockdat(cbps, mockdat, inst, ifield,
							 with_mask, perturb_gal_xy, gradient_filter, 
							 plot=False):

	if with_mask:
	
		if gradient_filter:
			dot1, X, mask_rav = precomp_filter_general(cbps.dimx, cbps.dimy, mask=mockdat['mask'], gradient_filter=True, quadoff_grad=False,  \
																fc_sub=False, fc_sub_quad_offset=False, fc_sub_n_terms=2)

	else:
		dot1, X = precomp_filter_general(cbps.dimx, cbps.dimy, gradient_filter=False, quadoff_grad=False,  \
													fc_sub=False, fc_sub_quad_offset=True, fc_sub_n_terms=2)
		mask_rav = None


	intensity_map = mockdat['intensity_map']

	if perturb_gal_xy:
		galmap = mockdat['galmap_perturb']
	else:
		galmap = mockdat['galmap']


	true_clg, true_clx = mockdat['gal_auto_pred'], mockdat['cross_pred']
	
	if with_mask:
		mask = mockdat['mask']
	else:
		mask = np.ones_like(intensity_map)

	intensity_map *= mask
	intensity_map[intensity_map != 0] -= np.mean(intensity_map[intensity_map != 0])
	
	galmap[(mask!=0)] = (galmap[(mask != 0)]-np.mean(galmap[(mask != 0)]))/np.mean(galmap[(mask != 0)])
	galmap *= mask

	if gradient_filter:
	
		print('separating gradients..')
		_, filter_comp_gal = apply_filter_to_map_precomp(galmap, dot1, X, mask_rav=mask_rav)
		galmap[mask != 0] -= filter_comp_gal[mask != 0]
		
		if plot:
			plot_map(galmap, title='gal map after filter')

		_, filter_comp = apply_filter_to_map_precomp(intensity_map, dot1, X, mask_rav=mask_rav)
		intensity_map[mask != 0] -= filter_comp[mask != 0]
		
		if plot:
			plot_map(intensity_map, title='intensity map after filter', figsize=(6,6))
		
	# print('intensity map has mean ', np.mean(intensity_map))
	lb, clgg, clggerr = get_power_spec(galmap, nbins=26)
	lb, clx, clxerr = get_power_spec(intensity_map, map_b=galmap, nbins=26)
	
	clx = np.dot(mockdat['invmkk'].T, clx)
	clgg = np.dot(mockdat['invmkk'].T, clgg)

	bls_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/beam_correction/bl_est_postage_stamps_TM'+str(inst)+'_081121.npz'
	B_ell = np.load(bls_fpath)['B_ells_post'][ifield-4]

	clx /= B_ell

	return lb, true_clg, true_clx, clgg, clx


def test_mock_gal_auto_cross_recovery(nrealiz_test, inst, nrealiz_jmock=10, ifield=8,
									  apply_zl_gradient=False, 
									  with_read_noise=True,
									  with_photon_noise=True,
									  add_weird_ciber_comp=True,
									 nbins=25, with_mask=False, \
									 gradient_filter=False, 
									 fc_sub=False, 
									 ifield_mask=8, 
									 nrealiz_mask=10, 
									 masking_maglim=16.0, 
									 galstr='hsc_i_lt_25.0', \
									 dgl_scale_fac=10., \
									 perturb_gal_xy=False,
									 dx=0.5, dy=0.5, 
									 with_redshift=True):
	
	
	cbps = CIBER_PS_pipeline()
	
	mock_mask_dir = config.ciber_basepath+'data/ciber_mocks/112022/TM'+str(inst)+'/'
	
	
	bandstr_dict = dict({1:'J', 2:'H'})
	
	if with_mask:
		mask_basepath = mock_mask_dir+'masks/maglim_'+str(masking_maglim)+'_Vega_081323/'
		maskstr = 'maglim_'+str(masking_maglim)+'_Vega'
		
	
	all_true_clg, all_true_clx, all_est_clg, all_est_clx = [np.zeros((nrealiz_test, nbins)) for x in range(4)]
	
	
	if gradient_filter and not with_mask:
		dot1, X = precomp_filter_general(cbps.dimx, cbps.dimy, gradient_filter=gradient_filter, quadoff_grad=False,  \
															fc_sub=fc_sub, fc_sub_quad_offset=True, fc_sub_n_terms=2)
		mask_rav = None

	
	for n in np.arange(nrealiz_test):
		
		
		if n==0:
			plot=True
		else:
			plot=False
			
		mock_rlz = n%nrealiz_jmock + 1

		mask_rlz = n%nrealiz_mask
		
		print('on mock realiz ', n, 'of', nrealiz_test)
		
		mockdat = gen_mock_ciber_obs(mock_rlz, inst, ifield,
									 apply_zl_gradient=apply_zl_gradient,
									 with_read_noise=with_read_noise,
									 with_photon_noise=with_photon_noise,
									 galstr=galstr, 
									add_weird_ciber_comp=add_weird_ciber_comp, \
									plot=plot, dgl_scale_fac=dgl_scale_fac, \
									perturb_gal_xy=perturb_gal_xy, dx=dx, dy=dy, 
									load_mask=with_mask, mask_rlz=mask_rlz, masking_maglim=masking_maglim, 
									with_redshift=with_redshift)
		
		lb, true_clg, true_clx, clgg, clx = powerspec_from_mockdat(cbps, mockdat, inst, ifield,
																 with_mask, perturb_gal_xy, gradient_filter, 
																 plot=plot)


		all_true_clg[n] = true_clg
		all_true_clx[n] = true_clx
		all_est_clg[n] = clgg 
		all_est_clx[n] = clx

		
# 		if with_mask:
		
# 			if gradient_filter:
# 				dot1, X, mask_rav = precomp_filter_general(cbps.dimx, cbps.dimy, mask=mask_indiv, gradient_filter=True, quadoff_grad=False,  \
# 																	fc_sub=False, fc_sub_quad_offset=False, fc_sub_n_terms=2)
# #             plot_map(mask_indiv, title='mask', figsize=(6,6))

# 		intensity_map = mockdat['intensity_map']

# 		if perturb_gal_xy:
# 			galmap = mockdat['galmap_perturb']
# 		else:
# 			galmap = mockdat['galmap']

# 		# plot_map(mockdat['galmap_perturb']-mockdat['galmap'], title='difference between perturbed and original maps')
		
# 		true_clg[n] = mockdat['gal_auto_pred']
# 		true_clx[n] = mockdat['cross_pred']
		
# 		if with_mask:
# 			mask = mockdat['mask']
# 		else:
# 			mask = np.ones_like(intensity_map)
			
			
# 		intensity_map *= mask
# 		intensity_map[intensity_map != 0] -= np.mean(intensity_map[intensity_map != 0])
# 		galmap[(mask!=0)] = (galmap[(mask != 0)]-np.mean(galmap[(mask != 0)]))/np.mean(galmap[(mask != 0)])
# 		galmap *= mask
		
# 		if gradient_filter:
			
# 			print('separating gradients..')
# 			_, filter_comp_gal = apply_filter_to_map_precomp(galmap, dot1, X, mask_rav=mask_rav)
# 			galmap[mask != 0] -= filter_comp_gal[mask != 0]
			
# 			if n==0:
# 				plot_map(galmap, title='gal map after filter')


# 			_, filter_comp = apply_filter_to_map_precomp(intensity_map, dot1, X, mask_rav=mask_rav)
# 			intensity_map[mask != 0] -= filter_comp[mask != 0]
			
# 			if n==0:
# 				plot_map(intensity_map, title='intensity map after filter', figsize=(6,6))
		
		
# 		print('intensity map has mean ', np.mean(intensity_map))
# 		lb, clgg, clggerr = get_power_spec(galmap, nbins=26)
# 		lb, clx, clxerr = get_power_spec(intensity_map, map_b=galmap, nbins=26)
		
		
# 		clx = np.dot(invmkk.T, clx)
# 		clgg = np.dot(invmkk.T, clgg)

# 		bls_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/beam_correction/bl_est_postage_stamps_TM'+str(inst)+'_081121.npz'
# 		B_ell = np.load(bls_fpath)['B_ells_post'][ifield-4]
	
# 		clx /= B_ell

# 		est_clg[n] = clgg
# 		est_clx[n] = clx
		
		
		
	ps_res = {'lb':lb, 'true_clg':all_true_clg, 'true_clx':all_true_clx, 'est_clg':all_est_clg, 'est_clx':all_est_clx}
		
	return ps_res


	
def test_mock_redshift_tomography(nrealiz, inst, ifield=8, gradient_filter=False, with_mask=True,
								 galstr='sdss_z_lt_22.0', nrealiz_jmock=24, apply_zl_gradient=True,
								  with_read_noise=True, with_photon_noise=True, add_weird_ciber_comp=True,
								  dgl_scale_fac=10., perturb_gal_xy=False, dx=0.5, dy=0.5,
								  with_photoz_error=False, sigma_z_phot=0.05, load_bg_dndz=False, 
								 n_ps_bins=25, nrealiz_mask=10, 
								 zbinedges=None):
	

	if zbinedges is None:
		zbinedges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

	cbps = CIBER_PS_pipeline()
	jmock_basedir = 'data/jordan_mocks/v2/'
	
	all_true_clg, all_true_clx,\
		all_est_clg, all_est_clx = [np.zeros((nrealiz, len(zbinedges[:-1]), n_ps_bins)) for x in range(4)]
	
	bls_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/beam_correction/bl_est_postage_stamps_TM'+str(inst)+'_081121.npz'
	B_ell = np.load(bls_fpath)['B_ells_post'][ifield-4]
	load_gal_counts = False
	
	for n in range(nrealiz):
		
		if n==0:
			plot=True
		else:
			plot=False

		mock_rlz = n%nrealiz_jmock + 1
		
		mask_rlz = np.random.choice(np.arange(nrealiz_mask))
#         mask_rlz = n%nrealiz_mask

		print('on mock realiz ', n, 'of', nrealiz)

		zstr = 'zmin='+str(zbinedges[0])+'_zmax='+str(zbinedges[1])

		mockdat = gen_mock_ciber_obs(mock_rlz, inst, ifield,
									 apply_zl_gradient=apply_zl_gradient,
									 with_read_noise=with_read_noise,
									 with_photon_noise=with_photon_noise,
									 galstr=galstr, 
									add_weird_ciber_comp=add_weird_ciber_comp, \
									plot=plot, dgl_scale_fac=dgl_scale_fac, \
									perturb_gal_xy=perturb_gal_xy, dx=dx, dy=dy,
									 load_gal_counts=load_gal_counts, zstr=zstr, 
									load_mask=True, mask_rlz=mask_rlz, tailstr='')

		
		if with_photoz_error:
			print('Making photo-z catalogs..')
			phot_z_catalogs = gen_photoz(galstr, inst, mock_rlz, zbinedges=zbinedges, sigma_z_phot=sigma_z_phot, load_bg_dndz=load_bg_dndz, 
										plot=plot)

		for i in range(len(zbinedges[:-1])):

#             zstr = 'CIBERfidmask_zmin='+str(zbinedges[i])+'_zmax='+str(zbinedges[i+1])
			zstr = 'zmin='+str(zbinedges[i])+'_zmax='+str(zbinedges[i+1])

			tracer_x_photz, tracer_y_photz = phot_z_catalogs[i][:,0], phot_z_catalogs[i][:,1]
			
			galcount_photoz = get_count_field(tracer_x_photz, tracer_y_photz, imdim=cbps.dimx)
			
			if i==0:
				print('first bin tracer x has length ', len(tracer_x_photz))
				if plot:
					plot_map(galcount_photoz, title='gal count photoz bin', figsize=(6,6))
			
			mockdat['galmap'] = galcount_photoz
			
			lb, _, _, est_clg, est_clx = powerspec_from_mockdat(cbps, mockdat, inst, ifield,
																		 with_mask, perturb_gal_xy, gradient_filter, 
																		 plot=plot)

			
			true_cross_fpath = jmock_basedir+'mock_ps_pred/TM'+str(inst)+'/indiv/rlz'+str(mock_rlz)+'_TM'+str(inst)+'_auto_cross_pred_'+galstr+'_'+zstr+'.npz'
			true_cross = np.load(true_cross_fpath)
			lb_pred, true_clg, true_clx = [true_cross[key] for key in ['lb', 'clg_comb', 'clx_comb']]
			
			
			all_true_clg[n, i] = true_clg
			all_true_clx[n, i] = true_clx/B_ell
			all_est_clg[n, i] = est_clg
			all_est_clx[n, i] = est_clx
			
			
	res = {'lb':lb_pred, 'all_true_clg':all_true_clg, 'all_true_clx':all_true_clx, 'all_est_clg':all_est_clg, 'all_est_clx':all_est_clx}
	
	
	
	return res


def gen_photoz(galstr, inst, rlz, zbinedges=None, sigma_z_phot=0.05, load_bg_dndz=False, plot=False,
			  jmock_basedir = 'data/jordan_mocks/v2/'):

	if zbinedges is None:
		zbinedges = [0.0, 0.1, 0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	
	if load_bg_dndz:
		
		z_fine, all_dNdz_b, all_dNdz_b_err = load_norm_bg_dndz(zbinedges, ng_tot=None)

		print('all dndzb has shape', np.array(all_dNdz_b).shape)
#         sampler = RedshiftSampler(z_fine, np.array(all_dNdz_b))
	
	all_tracer_x, all_tracer_y, all_z_true, all_z_phot = [[] for x in range(4)]
	
	for i in range(len(zbinedges[:-1])):
		addstr = 'zmin='+str(zbinedges[i])+'_zmax='+str(zbinedges[i+1])
	

#         gal_map_fpath = jmock_basedir+'mock_maps/galaxy/TM'+str(inst)+'/rlz'+str(rlz)+'_TM'+str(inst)+'_'+galstr+'_CIBERfidmask_'+addstr+'_galaxy.npz'
		gal_map_fpath = jmock_basedir+'mock_maps/galaxy/TM'+str(inst)+'/rlz'+str(rlz)+'_TM'+str(inst)+'_'+galstr+'_'+addstr+'_galaxy.npz'

		galdat = np.load(gal_map_fpath)
		
		tracer_x, tracer_y, tracer_z = [galdat[tkey] for tkey in ['tracer_x', 'tracer_y', 'tracer_z']]
		
		all_tracer_x.extend(tracer_x)
		all_tracer_y.extend(tracer_y)
		all_z_true.extend(tracer_z)
		
		
		if load_bg_dndz:
			zphot = sample_redshift(z_fine, all_dNdz_b[i], N=len(tracer_z))       
		else:
			zphot = np.abs(tracer_z + np.random.normal(0, sigma_z_phot, size=len(tracer_z)))
		
#         print('zphot = ', zphot)
		all_z_phot.extend(zphot)
		
		
	if plot:
		plt.figure(figsize=(7, 3))
		plt.hist(all_z_true, bins=np.linspace(0, 1.2, 40), label='ztrue', histtype='step')
		plt.hist(all_z_phot, bins=np.linspace(0, 1.2, 40), label='zphot', histtype='step')
		plt.xlabel('Redshift')
		plt.legend()
		plt.show()
		
		
	# bin new redshifts and make 
				
	data = np.array([all_tracer_x, all_tracer_y, all_z_true, all_z_phot]).transpose()
	
	binned_catalogs = split_into_photoz_bins(data, zbinedges, photoz_col=3)
	
	
	return binned_catalogs


def split_into_photoz_bins(data, zbinedges, photoz_col=3):
	"""
	Split a 2D array into a list of arrays based on photometric redshift bins.

	Parameters
	----------
	data : ndarray, shape (N, M)
		Input data array. One column must contain photometric redshifts.
	zbinedges : array-like
		Bin edges for photometric redshift.
	photoz_col : int, optional (default=3)
		Column index for photometric redshift in `data`.

	Returns
	-------
	binned_data : list of ndarrays
		List where each element is a sub-array of `data` for one photo-z bin.
	"""
	binned_data = []
	for i in range(len(zbinedges) - 1):
		zmin, zmax = zbinedges[i], zbinedges[i+1]
		mask = (data[:, photoz_col] >= zmin) & (data[:, photoz_col] < zmax)
		binned_data.append(data[mask])
	return binned_data


def sample_redshift(z_grid, dndz_unnorm, N=1):
	"""
	Draw redshift(s) from an unnormalized dN/dz.

	Parameters
	----------
	z_grid : array, shape (nz,)
		Redshift grid.
	dndz_unnorm : array, shape (nz,)
		Unnormalized dN/dz values.
	N : int, optional (default=1)
		Number of samples to draw.

	Returns
	-------
	z_samples : float or array
		A single redshift if N=1, otherwise an array of redshifts.
	"""
	dz = np.gradient(z_grid)

	# Normalize to PDF
	pdf = dndz_unnorm / np.sum(dndz_unnorm * dz)

	# Build CDF
	cdf = np.cumsum(pdf * dz)
	cdf /= cdf[-1]

	# Draw uniform random numbers
	u = np.random.rand(N)

	# Inverse transform sampling
	samples = np.interp(u, cdf, z_grid)

	if N == 1:
		return samples[0]  # return scalar if only one sample
	return samples
