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

if sys.version_info[0] == 3:
	from sklearn.tree import plot_tree

from catalog_utils import *


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
	tpr = float(np.sum(tpr))/float(len(tpr))
	
	fpr = np.array(predictions[~labels.astype(np.bool)] == 1)
	fpr = float(np.sum(fpr))/float(len(fpr))
	
	return tpr, fpr

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



def crossmatch_IBIS_unWISE_PS(field, df_IBIS_wxy=None, crossmatch_unWISE_PS=None, \
        datdir='/Users/luminatech/Documents/ciber2/ciber/data/cats/', xmatch_cut=False, xmatch_radius=0.5, \
        compute_WISE_mags=False, PS_mAB_to_mVega=False):

    '''

    Parameters
    ----------

    field : 'string'
        Field to perform crossmatching on. 

    datdir : 'string', optional

    xmatch_cut : 'bool', optional
        if True, only return crossmatches with source positions within xmatch_radius of each other. 
        Default is False.
    xmatch_radius : 'float', optional
        maximum cross matching radius permitted for merged catalogs if xmatch_cut is False.
        Default is 0.5 [arcseconds].

    Returns
    -------

    crossmatch_unWISE_PS_IBIS : class 'pandas.core.frame.DataFrame'
        Dataframe containing crossmatched unWISE + PanSTARRS + IBIS catalog.

    '''

    if df_IBIS_wxy is None:
    	print('loading IBIS catalog because it was not provided')
        df_IBIS_wxy = pd.read_csv(datdir+'IBIS_'+field+'_filt_wxy_ML_selected_best.csv')
    if crossmatch_unWISE_PS is None:
    	print('loading crossmatch catalog because it was not provided')
        crossmatch_unWISE_PS = pd.read_csv(datdir+'unWISE/'+field+'/unWISE_PS_crossmatch_2sig_dpos_'+field+'.csv')

    IBIS_src_coord = SkyCoord(ra=df_IBIS_wxy['ra']*u.degree, dec=df_IBIS_wxy['dec']*u.degree, frame='icrs')

    crossmatch_unWISE_PS_src_coord = SkyCoord(ra=crossmatch_unWISE_PS['ra']*u.degree, dec=crossmatch_unWISE_PS['dec']*u.degree, frame='icrs')

    idx_IBIS_unWISE_PS, d2d_IBIS, _ = match_coordinates_sky(crossmatch_unWISE_PS_src_coord, IBIS_src_coord) # there is an order specific element to this

    # IBIS sources are already in Vega magnitude system
    crossmatch_unWISE_PS['j_mag_best'] = np.array(df_IBIS_wxy.iloc[idx_IBIS_unWISE_PS].j_mag_best)
    crossmatch_unWISE_PS['h_mag_best'] = np.array(df_IBIS_wxy.iloc[idx_IBIS_unWISE_PS].h_mag_best)
    crossmatch_unWISE_PS['k_mag_best'] = np.array(df_IBIS_wxy.iloc[idx_IBIS_unWISE_PS].k_mag_best)

    crossmatch_unWISE_PS['j_magerr_best'] = np.array(df_IBIS_wxy.iloc[idx_IBIS_unWISE_PS].j_magerr_best)
    crossmatch_unWISE_PS['h_magerr_best'] = np.array(df_IBIS_wxy.iloc[idx_IBIS_unWISE_PS].h_magerr_best)
    crossmatch_unWISE_PS['k_magerr_best'] = np.array(df_IBIS_wxy.iloc[idx_IBIS_unWISE_PS].k_magerr_best)



    crossmatch_unWISE_PS['dmatch_arcsec_IBIS'] = d2d_IBIS.arcsec

    # converting WISE fluxes to Vega magnitudes
    if compute_WISE_mags:
        print('computing WISE mags (Vega)..')
        crossmatch_unWISE_PS['mag_W2'] = np.array(22.5 - 2.5*np.log10(crossmatch_unWISE_PS['flux_W2']))
        crossmatch_unWISE_PS['mag_W1'] = np.array(22.5 - 2.5*np.log10(crossmatch_unWISE_PS['flux_W1']))

    # PanSTARRS catalog PSF magnitudes are given w.r.t. the AB system, correcting to Vega for consistent calibration
    if PS_mAB_to_mVega:
        print('converting PanSTARRS AB magnitudes to Vega..')
        crossmatch_unWISE_PS['gMeanPSFMag'] -= -0.08
        crossmatch_unWISE_PS['rMeanPSFMag'] -= 0.16
        crossmatch_unWISE_PS['iMeanPSFMag'] -= 0.37
        crossmatch_unWISE_PS['zMeanPSFMag'] -= 0.54
        crossmatch_unWISE_PS['yMeanPSFMag'] -= 0.634

    if xmatch_cut:
        IBIS_PS_unWISE_xmatch_mask = np.where(np.array(crossmatch_unWISE_PS['dmatch_arcsec_IBIS']) <= xmatch_radius)[0]

        crossmatch_unWISE_PS_IBIS = crossmatch_unWISE_PS.iloc[IBIS_PS_unWISE_xmatch_mask].copy()
    else:
        crossmatch_unWISE_PS_IBIS = crossmatch_unWISE_PS.copy()

    return crossmatch_unWISE_PS_IBIS


def train_decision_tree(training_catalog, feature_names, outlablstr='j_mag_best', J_mag_lim=18.5, max_depth=5):
	
	classes_train = np.array(training_catalog[outlablstr] < J_mag_lim).astype(np.int)

	train_features = feature_matrix_from_df(training_catalog, feature_names, filter_nans=True)

	clf = tree.DecisionTreeClassifier(max_depth=max_depth)
	fig = clf.fit(train_features, classes_train)
	
	return fig, classes_train, train_features
	
	

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

  

def feature_matrix_from_df(df, feature_names, filter_nans=True, verbose=True):

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
    for feature_name in feature_names:
        if feature_name in list(df.columns):
            feature_matrix.append(df[feature_name])
        else:
        	if verbose:
            	print(feature_name+' not in input dataframe, adding nans to column')
            feature_matrix.append([np.nan for x in range(len(df))])

    feature_matrix = np.array(feature_matrix).transpose()

    if filter_nans:

        feature_matrix[np.isinf(feature_matrix)] = 30.
        feature_matrix[np.isnan(feature_matrix)] = 30.
        feature_matrix[feature_matrix<0.0] = 30.

    return feature_matrix


def filter_mask_cat_dt(input_cat, decision_tree, feature_names):
	
	cat_feature_matrix = feature_matrix_from_df(input_cat, feature_names)
	
	mask_predict = decision_tree.predict(cat_feature_matrix)
	
	mask_src_bool = np.where(mask_predict==1)[0]
	
	filt_cat = input_cat.iloc[mask_src_bool].copy()
	
	return filt_cat

def magnitude_to_radius_linear(magnitudes, alpha_m=-6.25, beta_m=110.):
	''' Masking radius function as given by Zemcov+2014. alpha_m has units of arcsec mag^{-1}, while beta_m
	has units of arcseconds.'''
	
	r = alpha_m*magnitudes + beta_m

	return r


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
		f = plt.figure(figsize=(10, 5))
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

	plt.legend()
	plt.xlabel('J band magnitude', fontsize=16)
	
	if show:
		plt.show()
	if return_fig:
		return f


