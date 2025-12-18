import numpy as np
import matplotlib 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import astropy.wcs as wcs
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky

from sklearn import tree
import sys

from sklearn import tree
from sklearn.tree import DecisionTreeRegressor

clf = DecisionTreeRegressor(max_depth=max_depth)
if max_predict is not None:
	fig = clf.fit(train_features[mags_train < max_predict], mags_train[mags_train < max_predict])
else:
	fig = clf.fit(train_features, mags_train)

if sys.version_info[0] == 3:
	from sklearn.tree import plot_tree

from ciber.io.catalog_utils import *
from ciber.masking.mask_utils import *


def compute_tpr_fpr(predictions, labels):
	
	'''
	Computes true positive rate and false positive rate for collection of predictions and labels.

	Parameters
	----------

	predictions : 'list' of length [Nsources]
		list of predictions

	labels : 'list' of length [Nsources]
		classification labels 

	Returns
	-------

	tpr : 'float'
		True positive rate

	fpr : 'float'
		False positive rate

	'''
	
	tpr = np.array(predictions[labels.astype(np.bool)] == 1)
	true_positive_rate = float(np.sum(tpr))/float(len(tpr))
	
	fpr = np.array(predictions[~labels.astype(np.bool)] == 1)
	false_positive_rate = float(np.sum(fpr))/float(len(fpr))
	
	return true_positive_rate, false_positive_rate

def compute_tpr_fpr_curves(predictions, labels, vals, nbin=30, minval=13, boundary=17.5):
	
	below_vals = vals[labels.astype(np.bool)]
	above_vals = vals[~labels.astype(np.bool)]
	
	val_bins = np.arange(14, 23, 0.5)

	tpr = np.array(predictions[labels.astype(np.bool)] == 1)
	fpr = np.array(predictions[~labels.astype(np.bool)] == 1)
	fnr = np.array(predictions[labels.astype(np.bool)] == 0)

	
	tpr_bin, fpr_bin, fnr_bin = [], [], []
	tpr_stds, fpr_stds, fnr_stds = [], [], []

	for i in range(len(val_bins[:-1])):
		magbin_tpr_mask = np.where((below_vals > val_bins[i])*(below_vals < val_bins[i+1]))[0]
		magbin_fpr_mask = np.where((above_vals > val_bins[i])*(above_vals < val_bins[i+1]))[0]

		if len(tpr[magbin_tpr_mask])==0:
			magbin_tpr, magbin_fnr = None, None
			tpr_std, fnr_std = None, None
		else:
			magbin_tpr = np.mean(tpr[magbin_tpr_mask])
			magbin_fnr = np.mean(fnr[magbin_tpr_mask])
			tpr_std = np.std(tpr[magbin_tpr_mask])/np.sqrt(len(tpr[magbin_tpr_mask]))
			fnr_std = np.std(fnr[magbin_tpr_mask])/np.sqrt(len(fnr[magbin_tpr_mask]))
		
		if len(fpr[magbin_fpr_mask]) == 0:
			fpr_std = None
			magbin_fpr = None
		else:
			magbin_fpr = np.mean(fpr[magbin_fpr_mask])
			fpr_std = np.std(fpr[magbin_fpr_mask])/np.sqrt(len(fpr[magbin_fpr_mask]))

		tpr_bin.append(magbin_tpr)
		fpr_bin.append(magbin_fpr)
		fnr_bin.append(magbin_fnr)
	
		tpr_stds.append(tpr_std)
		fpr_stds.append(fpr_std)
		fnr_stds.append(fnr_std)

		
	return np.array(tpr_bin), np.array(fpr_bin), np.array(fnr_bin), val_bins, np.array(tpr_stds), np.array(fpr_stds), np.array(fnr_stds)



def calculate_mag_err_rms(inst, valid_features, mags_valid, predicted_valid_mags, compute_by_subsample=True, \
						 mmin=4, mmax=21, nbins=41):
	
	bandstr_dict = dict({1:'J', 2:'H'})
	labels = ['unWISE+PS', 'PS only', 'unWISE only']
	mags = np.linspace(mmin, mmax, nbins)
	mid_mags = 0.5*(mags[1:]+mags[:-1])
		
	mag_errs_vs_rms_unPS, mag_errs_vs_rms_noWISE, mag_errs_vs_rms_noPS = [[] for x in range(3)]
	mag_errs_vs_rms_list = [mag_errs_vs_rms_unPS, mag_errs_vs_rms_noWISE, mag_errs_vs_rms_noPS]

	noPSmask = (valid_features[:,-2] != 30.)*(valid_features[:,0]==30.)
	noWISEmask = (valid_features[:,-2] == 30.)*(valid_features[:,0]!=30.)
	unPSmask = (valid_features[:,-2] != 30.)*(valid_features[:,0] !=30.)
	mags_valid_list = [mags_valid[unPSmask], mags_valid[noWISEmask], mags_valid[noPSmask]]

	mag_errs_valid_unPS = predicted_valid_mags[unPSmask]-mags_valid[unPSmask]
	mag_errs_valid_noWISE = predicted_valid_mags[noWISEmask]-mags_valid[noWISEmask]
	mag_errs_valid_noPS = predicted_valid_mags[noPSmask]-mags_valid[noPSmask]
	mag_errs_valid_list = [mag_errs_valid_unPS, mag_errs_valid_noWISE, mag_errs_valid_noPS]

	all_nmag_bysel = []
	for m, mag in enumerate(mags[:-1]):
		nmag = []
		for midx, mag_errs_vs_rms_indiv in enumerate(mag_errs_vs_rms_list):

			magmask = (mags_valid_list[midx] > mag)*(mags_valid_list[midx] <= mags[m+1])
			nmag.append(float(np.sum(magmask)))

			if np.sum(magmask)==0:
				mag_errs_vs_rms_list[midx].append(0.)
			else:
				mag_errs_inmask = np.array(mag_errs_valid_list[midx])[magmask]
				mag_errs_vs_rms_list[midx].append(0.5*(np.nanpercentile(mag_errs_inmask, 84)-np.nanpercentile(mag_errs_inmask, 16)))

		all_nmag_bysel.append(nmag)
	
	all_nmag_bysel = np.array(all_nmag_bysel)
	type_weights_bysel = np.zeros_like(all_nmag_bysel)
	
	for m, mag in enumerate(mags[:-1]):
		type_weights_bysel[m,:] = all_nmag_bysel[m,:]/np.sum(all_nmag_bysel[m,:])
		

	plt.figure(figsize=(7, 4))
	plt.subplot(1,2,1)
	for lidx, lab in enumerate(labels):
		plt.plot(mid_mags, type_weights_bysel[:,lidx], label=lab)
		
	plt.legend()
	plt.xlabel(bandstr_dict[inst]+' magnitude', fontsize=12)
	plt.ylabel('Catalog fraction', fontsize=12)
	plt.subplot(1,2,2)
	for lidx, lab in enumerate(labels):
		plt.plot(mid_mags, mag_errs_vs_rms_list[lidx], label=lab)
		
	plt.xlabel(bandstr_dict[inst]+' magnitude', fontsize=12)
	plt.ylabel('Magnitude error RMS', fontsize=12)
	plt.legend()
	plt.tight_layout()
	plt.xlim(12, 22)
	plt.show()
		

	return labels, mags, mid_mags, mag_errs_vs_rms_list, type_weights_bysel



def draw_train_validation_idxs(data, trainfrac=0.8):
	if trainfrac==1:
		ntrain = int(data.shape[0])
		nvalid = 0
		trainidx = np.arange(ntrain)
		valididx = []
	else:
		ntrain  = int(data.shape[0] * trainfrac)
		nvalid = data.shape[0] - ntrain
		permutation = np.random.permutation(data.shape[0])
		np.random.seed()
		trainidx = permutation[0:ntrain]
		valididx = permutation[-1-nvalid:-1]  
		
	return trainidx, valididx


def train_random_forest(training_catalog, feature_names, outlablstr='J_Vega', max_depth=5, max_predict=None, \
					   trainfrac=0.7, debias=True, mag_train_max=21, mag_train_min=0):

	
	training_catalog = training_catalog[(training_catalog[outlablstr] > mag_train_min)*(training_catalog[outlablstr] < mag_train_max)] # only train on detected NIR sources
	
	if feature_names is None:
		feature_names=['rMeanPSFMag', 'iMeanPSFMag', 'gMeanPSFMag', 'zMeanPSFMag', 'yMeanPSFMag', 'mag_W1', 'mag_W2']

	all_mags = np.array(training_catalog[outlablstr])
	
	plt.figure()
	plt.hist(all_mags, bins=np.linspace(10, 25, 30))
	plt.yscale('log')
	plt.show()
	
	if debias:
		print('Subtracting UKIDSS magnitudes by 0.2')
		all_mags -= 0.2
	
	nsrc_tot = len(all_mags)
	
	print('nsrc tot is ', nsrc_tot)
	all_features = feature_matrix_from_df(training_catalog, feature_names, filter_nans=True)
	print('min/max feature matrix is ', np.min(all_features), np.max(all_features))
	
	trainidx, valididx = draw_train_validation_idxs(all_mags, trainfrac=trainfrac)
	train_features = all_features[trainidx,:]
	valid_features = all_features[valididx,:]
	mags_train = all_mags[trainidx]
	mags_valid = all_mags[valididx]
	
	print('train/valid features have shapes:', train_features.shape, valid_features.shape)
		
	clf = DecisionTreeRegressor(max_depth=max_depth)
	if max_predict is not None:
		fig = clf.fit(train_features[mags_train < max_predict], mags_train[mags_train < max_predict])
	else:
		fig = clf.fit(train_features, mags_train)
		
	predicted_train_mags = fig.predict(train_features)
	predicted_valid_mags = fig.predict(valid_features)
	

	return fig, mags_train, mags_valid, train_features, valid_features, predicted_train_mags, predicted_valid_mags



def train_decision_tree(training_catalog, feature_names, \
							   extra_features=None, extra_feature_names=None,\
							   outlablstr='j_mag_best', mag_lim=18.5, max_depth=5, \
							  mode='regress', max_predict=None):

	classes_train = np.array(training_catalog[outlablstr] < mag_lim).astype(np.int)
	
	if mode=='regress':
		vals_train = training_catalog[outlablstr]
		print('min/max vals train', np.min(vals_train), np.max(vals_train))
		
	train_features = feature_matrix_from_df(training_catalog, feature_names, filter_nans=True)
	print('min/max training feature is ', np.min(train_features), np.max(train_features))
	if mode=='classify':
		clf = tree.DecisionTreeClassifier(max_depth=max_depth)
		fig = clf.fit(train_features, classes_train)

	elif mode=='regress':
		clf = DecisionTreeRegressor(max_depth=max_depth)
		if max_predict is not None:
			fig = clf.fit(train_features[vals_train < max_predict], vals_train[vals_train < max_predict])
		else:
			fig = clf.fit(train_features, vals_train)

	return fig, classes_train, train_features

	

def test_prediction(cat, decision_tree, feature_names, extra_features=None, extra_feature_names=None,\
						   outlablstr='j_mag_best', mag_lim=18.5, mode='classify', plot=False):
	
	classes_cat = np.array(cat[outlablstr] < mag_lim).astype(np.int)
	if mode=='regress':
		vals_cat = np.array(cat[outlablstr])
		
	features_cat = feature_matrix_from_df(cat, feature_names=feature_names)
			
	if mode=='regress':
		predictions_cat = decision_tree.predict(features_cat)
		predictions_classify = (predictions_cat < mag_lim)
		
		if plot:
			plt.figure()
			plt.hist(predictions_cat - vals_cat, bins=30)
			plt.xlabel('Predicted - True')
			plt.show()
		
	elif mode=='classify':
		predictions_classify = decision_tree.predict(features_cat)

	tpr, fpr = compute_tpr_fpr(predictions_classify, classes_cat)
	
	if mode=='classify':
		predictions_cat, vals_cat = None, None
	
	return tpr, fpr, predictions_classify, classes_cat, predictions_cat, vals_cat  

 
def evaluate_decision_tree_performance(decision_tree, test_catalog, extra_features=None,extra_feature_names=None, feature_names=None, feature_bands=None, J_mag_lim=18.5, \
									  outlablstr='j_mag_best', mode='classify'):
	
	if feature_names is None:
		feature_names = ['rMeanPSFMag', 'iMeanPSFMag', 'gMeanPSFMag', 'zMeanPSFMag', 'yMeanPSFMag', 'mag_W1', 'mag_W2']
	if feature_bands is None:
		feature_bands = ['r', 'i', 'g', 'z', 'y', 'W1']

	
	tpr, fpr, predictions, J_mask_bool_true, predictions_Jmag, J_mag_true = test_prediction(test_catalog, decision_tree, feature_names, extra_features=extra_features, extra_feature_names=extra_feature_names, maglim=J_mag_lim, \
														 outlablstr=outlablstr, mode=mode)
	tpr_curve, fpr_curve, fnr_curve, \
			mag_bins, tpr_stds, fpr_stds, fnr_stds = compute_tpr_fpr_curves(predictions, J_mask_bool_true, \
																			np.array(test_catalog[outlablstr]), \
																		   nbin=25, minval=14, boundary=J_mag_lim)
	
	return tpr_curve, fpr_curve, mag_bins, predictions, J_mask_bool_true, predictions_Jmag, J_mag_true



def parse_extra_features(xmatch_cat):
	g_r, r_i, i_z, W1_W2 = return_several_colors_df(xmatch_cat,\
													[['gMeanPSFMag', 'rMeanPSFMag'], ['rMeanPSFMag', 'iMeanPSFMag'], \
													['iMeanPSFMag', 'zMeanPSFMag'], ['mag_W1', 'mag_W2']])

	extra_features = [g_r, r_i, i_z, W1_W2]
	extra_feature_names = ['g-r', 'r-i', 'i-z', 'W1-W2']
	
	return g_r, r_i, i_z, W1_W2, extra_features, extra_feature_names
	

def decision_tree_train_and_test(training_catalog, feature_names, feature_bands=None, test_catalog=None, outlablstr='j_mag_best', J_mag_lim=18.5, max_depth=5, show_tree=False):

	'''
	Main function to train decision tree on crossmatched catalog data.

	Parameters
	----------

	training_catalog

	feature_names : `list' of strings with length [Nfeatures]
		list containing names of features used for classification

	feature_bands : `list' of strings with length [Nfeatures], optional
		list of features, same as above, but abbreviated for decision tree visualization.
		Default is 'None'.

	test_catalog : 

	outlablstr : 'string', optional
		output label key to be used with training_catalog and test_catalog.
		Default is 'j_mag_best'.

	J_mag_lim : `float', optional
		J magnitude boundary used for some classifications (CIBER specific). 
		Default is 18.5.

	max_depth : 'int', optional
		Maximum depth permitted for input decision tree.
		Default is 5.

	show_tree : 'bool', optional
		boolean for whether to produce visualization of decision tree, which is done by calling plot_decision_tree().
		Default is 'False'.

	
	Returns
	-------

	train_features, test_features : `~numpy.ndarray' of shape (Nsources, Nfeatures)
		feature matrices for training and test sets.

	tpr_train, fpr_train : 'float'
		true positive rate and false positive rate of training set

	tpr_test, fpr_test : 'float'
		true positive rate and false positive rate of test set

	classes_train, classes_test : 'list' of length [Nsources]
		training and test labels for classification task.

	predictions_train, predictions_test : `~numpy.ndarray' of shape (Nsources,)
		predictions made by decision tree on the training/test set.

	dt_fig : `matplotlib.figure.Figure', optional
		Figure containing decision tree visualization.

	'''

	decision_tree, classes_train, train_features = train_decision_tree(training_catalog, feature_names, outlablstr=outlablstr, J_mag_lim=J_mag_lim, max_depth=max_depth)

	if show_tree:
		above_label = 'J > '+str(J_mag_lim)
		below_label = 'J < '+str(J_mag_lim)
		if feature_bands is None:
			print('cant show decision tree without feature labels!')
			pass
		dt_fig = plot_decision_tree(decision_tree, feature_bands, class_names=[above_label, below_label], max_depth=max_depth)

	predictions_train = decision_tree.predict(train_features)

	tpr_train, fpr_train = compute_tpr_fpr(predictions_train, classes_train)

	if test_catalog is not None:

		classes_test = np.array(test_catalog[outlablstr] < J_mag_lim).astype(np.int)

		test_features = feature_matrix_from_df(test_catalog, feature_names, filter_nans=True)

		predictions_test = decision_tree.predict(test_features)

		tpr_test, fpr_test = compute_tpr_fpr(predictions_test, classes_test)

		if show_tree:
			return train_features, tpr_train, fpr_train, classes_train, predictions_train,\
					 test_features, tpr_test, fpr_test, classes_test, predictions_test, dt_fig


		return train_features, tpr_train, fpr_train, classes_train, predictions_train, \
				test_features, tpr_test, fpr_test, classes_test, predictions_test

	if show_tree:
		return train_features, tpr_train, fpr_train, classes_train, predictions_train, dt_fig


	return train_features, tpr_train, fpr_train, classes_train, predictions_train

def feature_matrix_from_df(df, feature_names, filter_nans=True, nan_replace_val = 25., min_replace_val=25, verbose=True):

	'''
	Helper function for loading catalog values into feature matrix.

	Parameters
	----------

	df : class 'pandas.core.frame.DataFrame'
		Dataframe containing catalog data.

	feature_names : `list' of strings with length [Nfeatures]
		list containing names of features used for classification

	filter_nans : 'bool', optional
		boolean determining whether feature matrix is filtered for infinite/NaN values.
		Default is 'True'.


	Returns
	-------

	feature_matrix : `numpy.ndarray' of shape (Nsources, Nfeatures)
		The compiled feature matrix.

	'''

	feature_matrix = []
	for f, feature_name in enumerate(feature_names):
		feature_matrix.append(df[feature_name])
		
		# print(feature_name)
		# print(feature_matrix[f])

	feature_matrix = np.array(feature_matrix).transpose()

	if filter_nans:

		feature_matrix[np.isinf(feature_matrix)] = nan_replace_val
		feature_matrix[np.isnan(feature_matrix)] = nan_replace_val
		feature_matrix[feature_matrix<-50] = min_replace_val
		feature_matrix[feature_matrix>50] = nan_replace_val


	return feature_matrix


def filter_mask_cat_dt(input_cat, decision_tree, feature_names, mag_lim=17.5, mode='regress'):

	cat_feature_matrix = feature_matrix_from_df(input_cat, feature_names)

	mask_predict = decision_tree.predict(cat_feature_matrix)

	if mode=='classify':
		mask_src_bool = np.where(mask_predict==1)[0]
	else:
		mask_src_bool = np.where(mask_predict < mag_lim)[0]

	filt_cat = input_cat.iloc[mask_src_bool].copy()

	return filt_cat


def magnitude_to_radius_linear(magnitudes, alpha_m=-6.25, beta_m=110.):
	''' Masking radius function as given by Zemcov+2014. alpha_m has units of arcsec mag^{-1}, while beta_m
	has units of arcseconds.'''
	
	r = alpha_m*magnitudes + beta_m

	return r

def predict_masking_magnitude_z_W1(mask_cat, J_mag_lim=17.5):
	# we will use the Zemcov+14 masking radius formula based on z-band magnitudes when available, and 
	# when z band is not available for a source we will use W1 + mean(z - W1) for the effective magnitude
	zs_mask = np.array(mask_cat['zMeanPSFMag'])
	W1_mask = np.array(mask_cat['mag_W1'])
	colormask = ((~np.isinf(zs_mask))&(~np.isinf(W1_mask))&(~np.isnan(zs_mask))&(~np.isnan(W1_mask))&(np.abs(W1_mask) < 50)&(np.abs(zs_mask) < 50))
	median_z_W1_color = np.median(zs_mask[colormask]-W1_mask[colormask])

	print('median z - W1 is ', median_z_W1_color)
	# find any non-detections in z band and replace with W1 + mean z-W1 
	nanzs = ((np.isnan(zs_mask))|(np.abs(zs_mask) > 50)|(np.isinf(zs_mask)))
	zs_mask[nanzs] = W1_mask[nanzs]+median_z_W1_color

	# anything that is neither detected in z or W1 (~10 sources) set to z=18.5.
	still_nanz = ((np.isnan(zs_mask))|(np.isinf(zs_mask)))
	zs_mask[still_nanz] = J_mag_lim
	mask_cat['zMeanPSFMag_mask'] = zs_mask + 0.5
	
	return zs_mask, mask_cat, W1_mask, colormask, median_z_W1_color


def predict_masking_catalogs(ifield_list, catalog_basepath, tailstr, feature_names=None, max_depth=8, save=False, debias=False, \
								   color_list=None, mag_train_max=21, mag_train_min=0., dpos_max=1.0, trainstr=None, \
							apply_to_science_cats=False, test_cosmos15=False, catalog_fpath=None, save_training_sets=False, \
							trainfrac=0.7, mmin_plot=14, mmax_plot=20, plot_s=5, alpha=0.5, mag_cut_rms=None, rms_textfs=12, rms_xpos=16., rms_ypos=15., \
							alpha_cosmos=0.1):
	
	if feature_names is None:
		feature_names=['rMeanPSFMag', 'iMeanPSFMag', 'gMeanPSFMag', 'zMeanPSFMag', 'yMeanPSFMag', 'mag_W1', 'mag_W2']

	ciber_field_dict = dict({4:'elat10', 5:'elat30',6:'BootesB', 7:'BootesA', 8:'SWIRE', 'train':'UDS'})

	# load NIR + unWISE + PanSTARRS merged catalog for training
	
	if catalog_fpath is None:
		catalog_fpath = catalog_basepath+'crossmatch/UKIDSS_unWISE_PS_fullmerge/UKIDSS_unWISE_PS_fullmerge_dpos='+str(dpos_max)+'_UDS.csv'
		
	merged_crossmatch_catalog = pd.read_csv(catalog_fpath)

	plt.figure()
	plt.hist(merged_crossmatch_catalog['rMeanPSFMag'], bins=np.linspace(15, 27, 30), histtype='step', label='r')
	plt.hist(merged_crossmatch_catalog['iMeanPSFMag'], bins=np.linspace(15, 27, 30), histtype='step', label='i')
	plt.hist(merged_crossmatch_catalog['zMeanPSFMag'], bins=np.linspace(15, 27, 30), histtype='step', label='z')
	plt.hist(merged_crossmatch_catalog['yMeanPSFMag'], bins=np.linspace(15, 27, 30), histtype='step', label='y')
	plt.hist(merged_crossmatch_catalog['mag_W1'], bins=np.linspace(15, 27, 30), histtype='step', label='mag W1')
	plt.hist(merged_crossmatch_catalog['mag_W2'], bins=np.linspace(15, 27, 30), histtype='step', label='mag W2')
	plt.yscale('log')
	plt.legend()
	plt.xlabel('Vega mag')
	plt.ylabel('Nsrc')
	plt.show()



	if color_list is not None:
		for c, comb in enumerate(color_list):
			color_label = comb[0]+'_'+comb[1]
			color = merged_crossmatch_catalog[comb[0]]-merged_crossmatch_catalog[comb[1]]
			merged_crossmatch_catalog.insert(c+1, color_label, color, True)
			feature_names.append(color_label)

	random_forest_J, mags_train_J, mags_valid_J, train_features_J,\
			valid_features_J, predicted_train_mags_J,\
			predicted_valid_mags_J = train_random_forest(merged_crossmatch_catalog, feature_names=feature_names,\
													   trainfrac=trainfrac, max_depth=max_depth, outlablstr='J_Vega', debias=debias, mag_train_max=mag_train_max, mag_train_min=mag_train_min)
	


	random_forest_H, mags_train_H, mags_valid_H, train_features_H,\
		valid_features_H, predicted_train_mags_H,\
		predicted_valid_mags_H = train_random_forest(merged_crossmatch_catalog, feature_names=feature_names,\
												   trainfrac=trainfrac, max_depth=max_depth, outlablstr='H_Vega', debias=debias, mag_train_max=mag_train_max, mag_train_min=mag_train_min)

	print('maximum feature values:', np.max(train_features_J), np.max(train_features_H))

	if save_training_sets:
		train_result_fpath = config.ciber_basepath+'data/catalogs/mask_predict/train_validation_set_J'
		if trainstr is not None:
			train_result_fpath += '_'+trainstr

		np.savez(train_result_fpath+'.npz', mags_train_J=mags_train_J, mags_valid_J=mags_valid_J, \
				train_features_J=train_features_J, valid_features_J=valid_features_J, predicted_train_mags_J=predicted_train_mags_J, \
				predicted_valid_mags_J=predicted_valid_mags_J)

		train_result_fpath = config.ciber_basepath+'data/catalogs/mask_predict/train_validation_set_H'
		if trainstr is not None:
			train_result_fpath += '_'+trainstr

		np.savez(train_result_fpath+'.npz', mags_train_H=mags_train_H, mags_valid_H=mags_valid_H, \
				train_features_H=train_features_H, valid_features_H=valid_features_H, predicted_train_mags_H=predicted_train_mags_H, \
				predicted_valid_mags_H=predicted_valid_mags_H)

	print('feature names:', feature_names)
	# now load science catalogs and make predictions
	predicted_catalogs = []
	
	if test_cosmos15:
		print('Applying model to predict COSMOS 15 sources')
		c15_cat = pd.read_csv(catalog_basepath+'COSMOS15/COSMOS15_hjmcc_rizy_JH_CH1CH2_AB.csv')
		
#         color_list_c15 = [['z_AB', 'y_AB']] # labels are AB but subtracting after converting to Vega magnitudes
		color_list_c15 = None
		# plt.figure()
		# plt.hist(c15_cat['J'], bins=np.linspace(14, 23, 20), histtype='step', label='J')
		# plt.hist(c15_cat['H'], bins=np.linspace(14, 23, 20), histtype='step', label='H')
		# plt.legend()
		# plt.yscale('log')
		# plt.show()
		
		print('c15 cat is originally', len(c15_cat))

		c15_cat_cut = c15_cat.iloc[np.where((c15_cat['J'] < 22))]
		# c15_cat_cut = c15_cat.iloc[np.where((c15_cat['J'] < mag_train_max))]

		Vega_to_AB_c15 = dict({'r_AB':0.16, 'i_AB':0.37, 'z_AB':0.54, 'y_AB':0.634, 'J':0.91, 'H':1.39, 'K':1.85, 'mag_CH1':2.699, 'mag_CH2':3.339})


		print('after J cut c15 cat now has length ', len(c15_cat))
		
		feature_names_c15 = ['r_AB', 'i_AB', 'z_AB', 'y_AB', 'mag_CH1', 'mag_CH2']

		mag_lims_training = [22, 22, 21.5, 21.0, 18.0, 17.0]
#         feature_names_c15 = ['r_AB', 'i_AB', 'z_AB', 'y_AB', 'mag_CH1']
		
		test_bands = ['J', 'H']
		

		for n, name in enumerate(feature_names_c15):
			c15_cat_cut[name] -= Vega_to_AB_c15[name]

			# if name in ['mag_CH1', 'mag_CH2']:
			# 	c15_cat_cut[name].iloc[(c15_cat_cut[name] > mag_lims_training[n])] = np.nan
		
		for name in test_bands:
			c15_cat_cut[name] -= Vega_to_AB_c15[name] 
			

		plt.figure()
		plt.title('COSMOS 2015')
		plt.hist(c15_cat_cut['r_AB'], bins=np.linspace(15, 27, 30), histtype='step', label='r')
		plt.hist(c15_cat_cut['i_AB'], bins=np.linspace(15, 27, 30), histtype='step', label='i')
		plt.hist(c15_cat_cut['z_AB'], bins=np.linspace(15, 27, 30), histtype='step', label='z')
		plt.hist(c15_cat_cut['y_AB'], bins=np.linspace(15, 27, 30), histtype='step', label='y')
		plt.hist(c15_cat_cut['mag_CH1'], bins=np.linspace(15, 27, 30), histtype='step', label='mag W1')
		plt.hist(c15_cat_cut['mag_CH2'], bins=np.linspace(15, 27, 30), histtype='step', label='mag W2')
		plt.yscale('log')
		plt.legend()
		plt.xlabel('Vega mag')
		plt.ylabel('Nsrc')
		plt.show()
				
		if color_list_c15 is not None:
			for c, comb in enumerate(color_list_c15):
				color_label = comb[0]+'_'+comb[1]
				color = c15_cat_cut[comb[0]]-c15_cat_cut[comb[1]]
				c15_cat_cut.insert(c+1, color_label, color, True)
				feature_names_c15.append(color_label)
			
			
		plt.figure()
		plt.hist(c15_cat['J'], bins=np.linspace(14, 23, 20), histtype='step', label='J')
		plt.hist(c15_cat['H'], bins=np.linspace(14, 23, 20), histtype='step', label='H')
		
		plt.hist(c15_cat_cut['J'], bins=np.linspace(14, 23, 20), histtype='step', label='J Vega')
		plt.hist(c15_cat_cut['H'], bins=np.linspace(14, 23, 20), histtype='step', label='H Vega')
		
		plt.legend()
		plt.yscale('log')
		plt.show()
		
		features_c15_test = feature_matrix_from_df(c15_cat_cut, feature_names=feature_names_c15, filter_nans=True)
		
		predictions_J_C15 = random_forest_J.predict(features_c15_test)
		c15_cat_cut['J_Vega_predict'] = predictions_J_C15
		
		predictions_H_C15 = random_forest_H.predict(features_c15_test)
		c15_cat_cut['H_Vega_predict'] = predictions_H_C15
		
		# mmin, mmax = 12, 20


		fig_J = plt.figure(figsize=(4, 4))

		if mag_cut_rms is not None:
			mag_cut_J = mag_cut_rms[0]

			mag_err = np.array(c15_cat_cut['J_Vega_predict'])-np.array(c15_cat_cut['J'])

			mag_err_cut = mag_err[(np.array(c15_cat_cut['J']) < mag_cut_J)]

			rms = np.std(mag_err_cut)
			print('rms:', rms)

			plt.text(rms_xpos, rms_ypos,'RMS ($J<$'+str(mag_cut_J)+'): '+str(np.round(rms, 2)), color='k', fontsize=rms_textfs)

		plt.scatter(c15_cat_cut['J'], c15_cat_cut['J_Vega_predict'], alpha=alpha_cosmos, color='k', s=plot_s)
		# plt.scatter(c15_cat_cut['J'], c15_cat_cut['J_Vega_predict'], c=c15_cat_cut['y_AB'], alpha=0.2, s=5, vmax=22)


		# plt.plot(np.linspace(mmin, mmax, 100), np.linspace(mmin, mmax, 100), linestyle='dashed', color='r')
		plt.text(mmin_plot+0.3, mmax_plot-0.7, 'COSMOS15 test set', fontsize=14, bbox=dict({'facecolor':'white', 'alpha':0.8, 'edgecolor':'None'}))
		
		# plt.text(mmin+0.25, mmax-1.0, 'UDS training set\nCOSMOS15 test set', fontsize=12, bbox=dict({'facecolor':'white', 'alpha':0.5, 'edgecolor':'None'}))
		# plt.xlabel('$J_{true}$ [COSMOS 15]', fontsize=12)
		# plt.ylabel('$J_{predict}$', fontsize=12)
		plt.xlabel('UVISTA J [COSMOS 15]', fontsize=14)
		plt.ylabel('Predicted J magnitude', fontsize=14)

		plt.xlim(mmin_plot, mmax_plot)
		plt.ylim(mmin_plot, mmax_plot)
		plt.plot(np.linspace(mmin_plot, mmax_plot, 100), np.linspace(mmin_plot, mmax_plot, 100), linestyle='dashed', color='grey')

		plt.axvspan(mmin_plot, 15.5, color='b', alpha=0.1)
		plt.text(13.7, 12.5, 'Optical photometry saturated', rotation=90, fontsize=12, color='b')

		plt.grid(alpha=0.5)
		plt.tight_layout()
		# plt.savefig('figures/train_uds_test_C15_meas_vs_predicted_J_032524.png', bbox_inches='tight', dpi=300)
		plt.show()

		fig_H = plt.figure(figsize=(4, 4))

		if mag_cut_rms is not None:
			mag_cut_H = mag_cut_rms[1]

			mag_err = np.array(c15_cat_cut['H_Vega_predict'])-np.array(c15_cat_cut['H'])

			mag_err_cut = mag_err[(np.array(c15_cat_cut['H']) < mag_cut_H)]

			rms = np.std(mag_err_cut)

			plt.text(rms_xpos, rms_ypos, 'RMS ($H<$'+str(mag_cut_H)+'): '+str(np.round(rms, 2)), color='k', fontsize=rms_textfs)

		plt.text(mmin_plot+0.3, mmax_plot-0.7, 'COSMOS15 test set', fontsize=14, bbox=dict({'facecolor':'white', 'alpha':0.8, 'edgecolor':'None'}))

		# plt.scatter(c15_cat_cut['H'], c15_cat_cut['H_Vega_predict'], c=c15_cat_cut['y_AB'], alpha=0.2, s=5, vmax=22)

		plt.scatter(c15_cat_cut['H'], c15_cat_cut['H_Vega_predict'], alpha=alpha_cosmos, color='k', s=plot_s)
	
		plt.xlim(mmin_plot, mmax_plot)
		plt.ylim(mmin_plot, mmax_plot)
		plt.plot(np.linspace(mmin_plot, mmax_plot, 100), np.linspace(mmin_plot, mmax_plot, 100), linestyle='dashed', color='grey')
		plt.grid(alpha=0.5)

		plt.axvspan(mmin_plot, 15.5, color='b', alpha=0.1)

		plt.text(13.7, 12.5, 'Optical photometry saturated', rotation=90, fontsize=12, color='b')

		plt.xlabel('UVISTA H [COSMOS 15]', fontsize=14)
		# plt.ylabel('$H_{predict}$', fontsize=12)
		plt.ylabel('Predicted H magnitude', fontsize=14)
		plt.tight_layout()
		# plt.savefig('figures/train_uds_test_C15_meas_vs_predicted_H_032524.png', bbox_inches='tight', dpi=300)
		plt.show()
		
		# convert back to AB magnitudes
		print('converting back to AB magnitudes')
		for name in feature_names_c15:
			c15_cat_cut[name] += Vega_to_AB_c15[name]
		
		for name in test_bands:
			c15_cat_cut[name] += Vega_to_AB_c15[name] 
			
		c15_cat_cut['J_Vega_predict'] += Vega_to_AB_c15['J']
		c15_cat_cut['H_Vega_predict'] += Vega_to_AB_c15['H']
		
		# save predicted COSMOS catalogs
		
		c15_cat_cut.to_csv(catalog_basepath+'mask_predict/mask_predict_C15test_'+tailstr+'_AB.csv')
	
	if apply_to_science_cats:
		for fieldidx, ifield in enumerate(ifield_list):

			fieldname = ciber_field_dict[ifield]

			merged_science_field_cat = pd.read_csv(catalog_basepath+'crossmatch/unWISE_PS_fullmerge/unWISE_PS_fullmerge_dpos=1.0_'+fieldname+'.csv')
	#         merged_crossmatch_unWISE_PS = pd.read_csv(catalog_basepath+'crossmatch/unWISE_PS_fullmerge/unWISE_PS_fullmerge_dpos=1.0_'+fieldname+'.csv')
	#         merged_crossmatch_unWISE_PS = pd.read_csv(catalog_basepath+'DECaLS/filt/decals_CIBER_ifield'+str(ifield)+'.csv')

			if color_list is not None:
				for c, comb in enumerate(color_list):
					color_label = comb[0]+'_'+comb[1]
					color = merged_science_field_cat[comb[0]]-merged_science_field_cat[comb[1]]
					merged_science_field_cat.insert(c+1, color_label, color, True)

			features_merged_cat = feature_matrix_from_df(merged_science_field_cat, feature_names=feature_names, filter_nans=True)
			predictions_J_CIBER_field = random_forest_J.predict(features_merged_cat)
			merged_science_field_cat['J_Vega_predict'] = predictions_J_CIBER_field

			predictions_H_CIBER_field = random_forest_H.predict(features_merged_cat)
			merged_science_field_cat['H_Vega_predict'] = predictions_H_CIBER_field

			if save:
				print('saving merged catalog..')
				merged_science_field_cat.to_csv(catalog_basepath+'mask_predict/mask_predict_LS_fullmerge_'+fieldname+'_'+tailstr+'.csv')

			predicted_catalogs.append(merged_science_field_cat)
		
	return predicted_catalogs, fig_J, fig_H



def mask_cat_predict_rf(ifield, inst, cmock, mask_cat_unWISE_PS=None, fieldstr_train = 'UDS',\
						mag_lim=17.5, feature_names=None, max_depth=8, zkey = 'zMeanPSFMag', W1key='mag_W1', mean_z_J_color_all = 1.0925, \
							 mask_cat_directory='data/cats/masking_cats/', twomass_cat_directory='data/cats/2MASS/filt/', \
							mode='regress'):

	''' Train model on UDS field and use to predict J and H band magnitudes '''

	fieldstr_mask = cmock.ciber_field_dict[ifield]
	magstr = cmock.helgason_to_ciber_rough[inst]
	Vega_dict = dict({1:'j_Vega', 2:'h_Vega'})
	
	magstr_predict = cmock.helgason_to_ciber_rough[inst]+'_predict'

	if feature_names is None:
		feature_names=['rMeanPSFMag', 'iMeanPSFMag', 'gMeanPSFMag', 'zMeanPSFMag', 'yMeanPSFMag', 'mag_W1', 'mag_W2']

	if mask_cat_unWISE_PS is None:

		if fieldstr_train == 'UDS': # default is to use UDS field as training set

			print(magstr+'_mag_lim = ', mag_lim)

			# this catalog is in Vega magnitudes, 
			nodup_crossmatch_unWISE_PS_uk_UDS = pd.read_csv('data/cats/masking_cats/UDS/unWISE_PanSTARRS_UKIDSS_full_xmatch_merge_UDS.csv')
			nodup_crossmatch_unWISE_PS_uk_UDS['gMeanPSFMag'] += 0.16
			mag_condition_uds, unWISE_condition_uds, PanSTARRS_condition_uds = detect_cat_conditions_J_unWISE_PanSTARRS(nodup_crossmatch_unWISE_PS_uk_UDS, \
																													j_key=Vega_dict[inst])
			unPSuk_mask = np.where(mag_condition_uds&unWISE_condition_uds&PanSTARRS_condition_uds)
			unWISE_PS_uk_xmatch = nodup_crossmatch_unWISE_PS_uk_UDS.iloc[unPSuk_mask].copy()

			print('Training decision tree..')

			decision_tree, classes_train, train_features = train_decision_tree(unWISE_PS_uk_xmatch, feature_names=feature_names, mag_lim=mag_lim, \
																  max_depth=max_depth, outlablstr=Vega_dict[inst], mode=mode)


		full_merged_cat_unWISE_PS = pd.read_csv('data/cats/masking_cats/'+fieldstr_mask+'/unWISE_PanSTARRS_full_xmatch_merge_'+fieldstr_mask+'_121620.csv')
		full_merged_cat_unWISE_PS['gMeanPSFMag'] += 0.16 # correcting error in processed PS catalog

		# use decision tree to identify sources that need masking
		featuresd_cat_unWISE_PS = feature_matrix_from_df(full_merged_cat_unWISE_PS, feature_names=feature_names, filter_nans=True)
		predictions_CIBER_field_unWISE_PS = decision_tree.predict(features_merged_cat_unWISE_PS)
		
		if mode=='regress':
			full_merged_cat_unWISE_PS[magstr_predict] = predictions_CIBER_field_unWISE_PS

		mask_cat_unWISE_PS = filter_mask_cat_dt(full_merged_cat_unWISE_PS, decision_tree, feature_names, mag_lim=mag_lim, mode=mode)

	if mode=='classify':
		zs_mask, mask_cat, W1_mask, colormask, median_z_W1_color = predict_masking_magnitude_z_W1(mask_cat_unWISE_PS, J_mag_lim=mag_lim)
		mask_cat_unWISE_PS[zkey_mask] = zs_mask + 0.5
		magstr_predict = zkey +'_mask'
	print('magstr predict is ', magstr_predict)

	twomass = pd.read_csv(twomass_cat_directory+'2MASS_'+fieldstr_mask+'_filtxy.csv') # these are in Vega
	twomass_max_mag = min(mag_lim, 16.)
	print('2MASS maximum is ', twomass_max_mag)

	if mode=='regress':
		mean_z_J_color_all = 0.
		
	twomass_lt_max_mag = twomass_srcmap_masking_cat_prep(twomass, mean_z_J_color_all, cmock, ifield, inst=inst, twomass_max_mag=twomass_max_mag)
	
	return mask_cat_unWISE_PS, twomass_lt_max_mag
	

def source_mask_construct_dt(ifield, inst, cmock, mask_cat_unWISE_PS=None, fieldstr_train = 'UDS',\
							 J_mag_lim=19.0, feature_names=None, max_depth=8, \
							zkey = 'zMeanPSFMag', W1key='mag_W1', mean_z_J_color_all = 1.0925, \
							 mask_cat_directory='data/cats/masking_cats/', twomass_cat_directory='data/cats/2MASS/filt/', \
							nx=1024, ny=1024, pixsize=7., \
							# linear fit parameters
							beta_m=125., alpha_m=-5.5, \
							# Gaussian fit parameters
							a1=252.8, b1=3.632, c1=8.52, intercept_mag=16.0, minrad=10.5, deltamag=3, \
							 mode='regress', plot=False, make_mask=False):

	''' Train model on UDS field and use to predict J and H band magnitudes '''
	
	fieldstr_mask = cmock.ciber_field_dict[ifield]
	magstr = cmock.helgason_to_ciber_rough[inst]

	if feature_names is None:
		feature_names=['rMeanPSFMag', 'iMeanPSFMag', 'gMeanPSFMag', 'zMeanPSFMag', 'yMeanPSFMag', 'mag_W1', 'mag_W2']
	
	if mask_cat_unWISE_PS is None:

		if fieldstr_train == 'UDS': # default is to use UDS field as training set

			print('J_mag_lim = ', J_mag_lim)
			
			# this catalog is in Vega magnitudes, 
			nodup_crossmatch_unWISE_PS_uk_UDS = pd.read_csv('data/cats/masking_cats/UDS/unWISE_PanSTARRS_UKIDSS_full_xmatch_merge_UDS.csv')
			nodup_crossmatch_unWISE_PS_uk_UDS['gMeanPSFMag'] += 0.16
			Jcondition_uds, unWISE_condition_uds, PanSTARRS_condition_uds = detect_cat_conditions_J_unWISE_PanSTARRS(nodup_crossmatch_unWISE_PS_uk_UDS, \
																													j_key='j_Vega')
			unPSuk_mask = np.where(Jcondition_uds&unWISE_condition_uds&PanSTARRS_condition_uds)
			unWISE_PS_uk_xmatch = nodup_crossmatch_unWISE_PS_uk_UDS.iloc[unPSuk_mask].copy()
			
			print('Training decision tree..')
						
			decision_tree, classes_train, train_features = train_decision_tree(unWISE_PS_uk_xmatch, feature_names=feature_names, J_mag_lim=J_mag_lim, \
																  max_depth=max_depth, outlablstr='j_Vega', \
																			mode=mode)

			
		full_merged_cat_unWISE_PS = pd.read_csv('data/cats/masking_cats/'+fieldstr_mask+'/unWISE_PanSTARRS_full_xmatch_merge_'+fieldstr_mask+'_121620.csv')
		full_merged_cat_unWISE_PS['gMeanPSFMag'] += 0.16 # correcting error in processed PS catalog

		# use decision tree to identify sources that need masking
		features_merged_cat_unWISE_PS = feature_matrix_from_df(full_merged_cat_unWISE_PS, feature_names=feature_names, filter_nans=True)
		predictions_CIBER_field_unWISE_PS = decision_tree.predict(features_merged_cat_unWISE_PS)
		if mode=='regress':
			full_merged_cat_unWISE_PS['J_predict'] = predictions_CIBER_field_unWISE_PS

		mask_cat_unWISE_PS = filter_mask_cat_dt(full_merged_cat_unWISE_PS, decision_tree, feature_names, J_mag_lim=J_mag_lim, mode=mode)

	
	if mode=='classify':
		zs_mask, mask_cat, W1_mask, colormask, median_z_W1_color = predict_masking_magnitude_z_W1(mask_cat_unWISE_PS, J_mag_lim=J_mag_lim)
		mask_cat_unWISE_PS[zkey_mask] = zs_mask + 0.5
		magstr_predict = zkey +'_mask'
	else:
		magstr_predict = cmock.helgason_to_ciber_rough[inst]+'_predict'

	if make_mask:
		# find best alpha, beta parameters for a1, b1, c1
		intercept = radius_vs_mag_gaussian(intercept_mag, a1=a1, b1=b1, c1=c1)
		alpha_m, beta_m = find_alpha_beta(intercept, minrad=minrad, dm=deltamag, pivot=intercept_mag)

		if plot:
			plt.figure()
			full_range = np.linspace(12, 20, 100)
			plt.plot(full_range, radius_vs_mag_gaussian(full_range, a1=a1, b1=b1, c1=c1))
			plt.plot(full_range, magnitude_to_radius_linear(full_range, beta_m=beta_m, alpha_m=alpha_m))
			plt.show()

		# using the effective masking magnitudes, compute the source mask
		print('Masking catalog has length ', len(mask_cat_unWISE_PS))
		mask_unWISE_PS, radii_mask_cat_unWISE_PS = mask_from_cat(cat_df=mask_cat_unWISE_PS, magstr=magstr_predict,\
																		 beta_m=beta_m, a1=a1, b1=b1, c1=c1, mag_lim=J_mag_lim,\
																	alpha_m=alpha_m, pixsize=pixsize, inst=inst, dimx=nx, dimy=ny)

	print('Now creating mask for 2MASS..')
	twomass = pd.read_csv(twomass_cat_directory+'2MASS_'+fieldstr_mask+'_filtxy.csv') # these are in Vega

	if J_mag_lim <= 16.:
		twom_Jmax = J_mag_lim
	else:
		twom_Jmax = 16.
	print('2MASS maximum is ', twom_Jmax)

	if mode=='regress':
		mean_z_J_color_all = 0.
	twomass_lt_16 = twomass_srcmap_masking_cat_prep(twomass, mean_z_J_color_all, cmock, twomass_max_mag=twom_Jmax)
	
	if make_mask:
		mask_twomass_simon, radii_mask_cat_twomass_simon = mask_from_cat(cat_df=twomass_lt_16, mag_lim=J_mag_lim, mode='Simon', magstr='j_m', Vega_to_AB=0., inst=inst, \
																				a1=a1, b1=b1, c1=c1, dimx=nx, dimy=ny)

		print('2MASS catalog has length ', len(radii_mask_cat_twomass_simon))
	
		return mask_unWISE_PS, mask_twomass_simon, mask_cat_unWISE_PS, twomass_lt_16
	else:
		return mask_cat_unWISE_PS, twomass_lt_16



def plot_decision_tree(dt, feature_names, line1='Left branches = condition True; Right branches = condition False',\
					   line2='Orange = J > 18.5; Blue = J < 18.5', class_names=['J>18.5', 'J<18.5'], return_fig=True, show=True, max_depth=4):
	'''
	This function produces a plot of a given decision tree through the sklearn module function sklearn.plot_tree().

	Parameters
	----------

	dt : class 'sklearn.tree._classes.DecisionTreeClassifier'
		Decision tree to be visualized.

	feature_names : `list' of strings with length [Nfeatures]
		list containing names of features used for classification

	line1, line2 : 'string', optional
		text to be placed in lines 1 and 2 of the figure's supertitle.

	class_names : 'list', optional
		list of names of decision tree output classes.
		Default is ['J>18.5', 'J<18.5'].

	max_depth : 'int', optional
		Maximum depth permitted for input decision tree.
		Default is 4.

	return_fig : `bool'
		boolean for whether to return figure at end of function. 
		Default is 'True'.

	Returns
	-------
	
	figure : `matplotlib.figure.Figure', optional
		Figure containing decision tree visualization.


	'''

	if max_depth==2:
		box_fs = 18
	elif max_depth==3:
		box_fs = 16
	elif max_depth==4:
		box_fs = 13
	else:
		box_fs = 10
	figure = plt.figure(figsize=(16,12))
	plt.suptitle(line1+' \n '+ line2, fontsize=20)
	ax = plt.subplot()
	f = tree.plot_tree(dt, fontsize=box_fs, label='root',impurity=True, ax=ax, filled=True, precision=2, proportion=True, feature_names=feature_names, class_names=class_names)

	if show:
		plt.show()
	if return_fig:
		return figure

def project_decision_tree_classification(train_features, train_labels, predictions_train, feature_names, train_field=None, tpr_train=None, fpr_train=None,\
										 J_mag_lim=18.5, project_idx0=0, project_idx1=1, \
										 test_features=None, test_labels=None, test_field=None, predictions_test=None, tpr_test=None, fpr_test=None, \
										xlims=None, ylims=None, s=2, alpha=0.5, show=True, return_fig=True):
	'''
	This function can be used to visualize decision tree performance on training and validation (optional) data.

	Parameters
	----------

	train_features : `~numpy.ndarray' of shape (Nsources, Nfeatures)
		feature matrix for training set. 

	train_labels : `~numpy.ndarray' of shape (Nsources,)
		classification labels for training set.

	predictions_train : `~numpy.ndarray' of shape (Nsources,)
		predictions made by decision tree on training set.

	feature_names : `list' of strings with length [Nfeatures]
		list containing names of features used for classification

	train_field : 'string', optional
		name of field used for training set. Default is 'None'.

	tpr_train, fpr_train : `float', optional
		true positive rate/false positve rate of training set, which can be shown along with classifications.
		Default is 'None'.

	J_mag_lim : `float', optional
		J magnitude boundary used for some classifications (CIBER specific). 
		Default is 18.5.

	project_idx0, project_idx1 : `int', optional
		indices specify features on which to project classifications.
		Defaults are 0 and 1, respectively.

	test_features : `~numpy.ndarray' of shape (Nsources, Nfeatures), optional
		feature matrix for test set. Default is 'None'.

	test_labels : `~numpy.ndarray' of shape (Nsources,), optional
		classification labels for test set.

	test_field : 'string', optional
		name of field used for test set. Default is 'None'

	predictions_test : `~numpy.ndarray' of shape (Nsources,), optional
		predictions made by decision tree on training set.
		Default is 'None'.

	tpr_test, fpr_test : `float', optional
		true positive rate/false positve rate of test set, which can be shown along with classifications.
		Default is 'None'.

	return_fig : `bool'
		boolean for whether to return figure at end of function. 
		Default is 'True'.

	Returns
	-------

	f : `matplotlib.figure.Figure', optional
		Figure containing classification for training/test set.


	'''

	nsubplots=2
	if test_features is not None:
		nsubplots = 3
		
	above_label = 'J > '+str(J_mag_lim)
	below_label = 'J < '+str(J_mag_lim)
	
	f = plt.figure(figsize=(6, int(4*nsubplots)))
	plt.subplot(nsubplots,1, 1)
	titlestr = 'Training set labels'
	if train_field is not None:
		titlestr += '\n ('+train_field+')'
	plt.title(titlestr, fontsize=14)
	plt.scatter(train_features[~train_labels.astype(np.bool),project_idx0], train_features[~train_labels.astype(np.bool),project_idx1], label=above_label, color='C1', s=s, alpha=alpha)
	plt.scatter(train_features[train_labels.astype(np.bool),project_idx0], train_features[train_labels.astype(np.bool),project_idx1], label=below_label, color='C0', s=s, alpha=alpha)
	plt.xlim(xlims)
	plt.ylim(ylims)
	plt.xlabel(feature_names[project_idx0], fontsize=14)
	plt.ylabel(feature_names[project_idx1], fontsize=14)
	plt.legend()
	
	plt.subplot(nsubplots,1, 2)
	titlestr_trainpred = 'Predictions on Training Set'
	if train_field is not None:
		titlestr_trainpred += '\n ('+train_field+')'
	plt.title(titlestr_trainpred, fontsize=14)
	plt.scatter(train_features[~predictions_train.astype(np.bool),project_idx0], train_features[~predictions_train.astype(np.bool),project_idx1], label=above_label, color='C1', s=s, alpha=alpha)
	plt.scatter(train_features[predictions_train.astype(np.bool),project_idx0], train_features[predictions_train.astype(np.bool),project_idx1], label=below_label, color='C0', s=s, alpha=alpha)
	
	plt.xlim(xlims)
	plt.ylim(ylims)
	plt.xlabel(feature_names[project_idx0], fontsize=14)
	plt.ylabel(feature_names[project_idx1], fontsize=14)
	plt.legend()
	if tpr_train is not None:
		text_val = 'TPR = '+str(np.round(tpr_train, 3))+'\nFPR = '+str(np.round(fpr_train, 3))
		plt.text(18.5, 13.5, text_val, fontsize=14)
	
	if test_features is not None:
		plt.subplot(nsubplots,1, nsubplots)
		
		titlestr_testpred = 'Predictions on Test Set'
		if test_field is not None:
			titlestr_testpred += ' \n ('+test_field+')'
		plt.title(titlestr_testpred, fontsize=14)
		
		plt.scatter(test_features[~predictions_test.astype(np.bool), project_idx0], test_features[~predictions_test.astype(np.bool), project_idx1], label=above_label, color='C1', s=s, alpha=alpha)
		plt.scatter(test_features[predictions_test.astype(np.bool), project_idx0], test_features[predictions_test.astype(np.bool), project_idx1], label=below_label, color='C0', s=s, alpha=alpha)

		plt.xlim(xlims)
		plt.ylim(ylims)
		plt.xlabel(feature_names[project_idx0], fontsize=14)
		plt.ylabel(feature_names[project_idx1], fontsize=14)
		plt.legend()
		if tpr_test is not None:
			text_val = 'TPR = '+str(np.round(tpr_test, 3))+'\nFPR = '+str(np.round(fpr_test, 3))
			plt.text(18.5, 13.5, text_val, fontsize=14)

	plt.tight_layout()
	if show:
		plt.show()
	if return_fig:
		return f


def plot_completeness_fdr_curves_decision_tree(mag_bins, J_mag_lim, tpr_binned, fpr_binned, tpr_binned_std=None, fpr_binned_std=None, \
											  show=True, return_fig=True, field=None, list_labels=None):
	
	midbin = 0.5*(mag_bins[1:]+mag_bins[:-1])
	
	if type(tpr_binned)==list:
		f = plt.figure(figsize=(8, 5))
	else:
		f = plt.figure(figsize=(6,5))
	titlestr = ''
	if field is not None:
		titlestr += field + ', '
	condition_str = 'J < '+str(J_mag_lim)
	titlestr += condition_str
	plt.title(titlestr, fontsize=16)
	
	plt.axvline(J_mag_lim, linestyle='solid', alpha=0.5, linewidth=3, label='J = '+str(J_mag_lim), color='k')

	if type(tpr_binned)==list:
		for i, tpr_bin in enumerate(tpr_binned):
			
			compstr = ''
			fdrstr = None
			if i==0:
				compstr += 'Completeness \n'
				fdrstr = 'False discovery rate \n'
				fdrstr += list_labels[i]

			compstr += list_labels[i]

			if tpr_binned_std is not None:
				plt.errorbar(midbin, tpr_bin, yerr=tpr_binned_std[i], label=compstr, marker='.', linestyle='solid', color='C'+str(i))
				plt.errorbar(midbin, fpr_binned[i], yerr=fpr_binned_std[i], label=fdrstr, marker='.', linestyle='dashed', color='C'+str(i))
			else:
				plt.plot(midbin, tpr_bin, label=compstr, marker='x', color='C'+str(i), linestyle='solid', markersize=15, linewidth=2, markeredgewidth=2, alpha=0.7)
				plt.plot(midbin, fpr_binned[i], label=fdrstr, marker='x', color='C'+str(i), linestyle='dashed', markersize=15, linewidth=2, markeredgewidth=2, alpha=0.7)
			
	else:
		if tpr_binned_std is not None:
			plt.errorbar(midbin, tpr_binned, yerr=tpr_binned_std, label='Completeness', marker='x')
			plt.errorbar(midbin, fpr_binned, yerr=fpr_binned_std, label='False positive rate', marker='x')
		else:
			plt.plot(midbin, tpr_binned, label='Completeness', marker='x')
			plt.plot(midbin, fpr_binned, label='False positive rate', marker='x')

	plt.legend(fontsize=14)
	plt.xlabel('J band magnitude', fontsize=16)
	plt.tick_params(labelsize=14)
	
	if show:
		plt.show()
	if return_fig:
		return f


