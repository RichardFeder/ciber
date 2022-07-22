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

if sys.version_info[0] == 3:
	from sklearn.tree import plot_tree

from catalog_utils import *
from masking_utils import *


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


def train_decision_tree(training_catalog, feature_names, \
                               extra_features=None, extra_feature_names=None,\
                               outlablstr='j_mag_best', J_mag_lim=18.5, max_depth=5, \
                              mode='classify', max_predict=None):

    classes_train = np.array(training_catalog[outlablstr] < J_mag_lim).astype(np.int)
    if mode=='regress':
        vals_train = training_catalog[outlablstr]
        print('min/max vals train', np.min(vals_train), np.max(vals_train))
        
    train_features = feature_matrix_from_df(training_catalog, feature_names, filter_nans=True)
        
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
                           outlablstr='j_mag_best', maglim=18.5, mode='classify'):
    
    classes_cat = np.array(cat[outlablstr] < maglim).astype(np.int)
    if mode=='regress':
        vals_cat = np.array(cat[outlablstr])
        
    features_cat = feature_matrix_from_df(cat, feature_names=feature_names)
            
    if mode=='regress':
        predictions_cat = decision_tree.predict(features_cat)
        predictions_classify = (predictions_cat < maglim)
        
        plt.figure()
        plt.hist(predictions_cat - vals_cat, bins=30)
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

    
    tpr, fpr, predictions, J_mask_bool_true, predictions_Jmag, J_mag_true = test_prediction_062222(test_catalog, decision_tree, feature_names, extra_features=extra_features, extra_feature_names=extra_feature_names, maglim=J_mag_lim, \
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

  

def feature_matrix_from_df(df, feature_names, filter_nans=True, nan_replace_val = 30., verbose=True):

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

		feature_matrix[np.isinf(feature_matrix)] = nan_replace_val
		feature_matrix[np.isnan(feature_matrix)] = nan_replace_val
		feature_matrix[feature_matrix<0.0] = nan_replace_val


	return feature_matrix


def filter_mask_cat_dt(input_cat, decision_tree, feature_names, J_mag_lim=17.5, mode='classify'):

    cat_feature_matrix = feature_matrix_from_df(input_cat, feature_names)

    mask_predict = decision_tree.predict(cat_feature_matrix)

    if mode=='classify':
        mask_src_bool = np.where(mask_predict==1)[0]
    else:
        mask_src_bool = np.where(mask_predict < J_mag_lim)[0]

    filt_cat = input_cat.iloc[mask_src_bool].copy()

    return filt_cat


def magnitude_to_radius_linear(magnitudes, alpha_m=-6.25, beta_m=110.):
	''' Masking radius function as given by Zemcov+2014. alpha_m has units of arcsec mag^{-1}, while beta_m
	has units of arcseconds.'''
	
	r = alpha_m*magnitudes + beta_m

	return r

def predict_masking_magnitude_z_W1(mask_cat):
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


def source_mask_construct_dt(ifield, inst, cmock, mask_cat_unWISE_PS=None, fieldstr_train = 'UDS',\
                             J_mag_lim=19.0, feature_names=None, max_depth=8, \
                            zkey = 'zMeanPSFMag', W1key='mag_W1', mean_z_J_color_all = 1.0925, \
                             mask_cat_directory='data/cats/masking_cats/', twomass_cat_directory='data/cats/2MASS/filt/', \
                            nx=1024, ny=1024, pixsize=7., \
                            # linear fit parameters
                            beta_m=125., alpha_m=-5.5, \
                            # Gaussian fit parameters
                            a1=252.8, b1=3.632, c1=8.52, intercept_mag=16.0, minrad=10.5, deltamag=3, \
                             mode='regress', plot=False):
    
    fieldstr_mask = cmock.ciber_field_dict[ifield]
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

        mask_cat_unWISE_PS = filter_mask_cat_dt_062222(full_merged_cat_unWISE_PS, decision_tree, feature_names, J_mag_lim=J_mag_lim, mode=mode)

    
    if mode=='classify':
        zs_mask, mask_cat, W1_mask, colormask, median_z_W1_color = predict_masking_magnitude_z_W1(mask_cat_unWISE_PS)
        mask_cat_unWISE_PS[zkey_mask] = zs_mask + 0.5
        magstr = zkey +'_mask'
    else:
        magstr = 'J_predict'

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
    mask_unWISE_PS, radii_mask_cat_unWISE_PS = mask_from_df_cat(cat_df=mask_cat_unWISE_PS, magstr=magstr,\
                                                                     beta_m=beta_m, a1=a1, b1=b1, c1=c1, mag_lim=J_mag_lim,\
                                                                alpha_m=alpha_m, pixsize=pixsize, inst=inst, dimx=nx, dimy=ny)

    print('Now creating mask for 2MASS..')
    twomass = pd.read_csv(twomass_cat_directory+'2MASS_'+fieldstr_mask+'_filtxy.csv') # these are in Vega
    print('field is ', fieldstr_mask)

    if J_mag_lim <= 16.:
        twom_Jmax = J_mag_lim
    else:
        twom_Jmax = 16.
        
    print('2MASS maximum is ', twom_Jmax)
    if mode=='regress':
        mean_z_J_color_all = 0.
    twomass_lt_16, srcmap_twomass_J_lt_16 = twomass_srcmap_masking_cat_prep(twomass, mean_z_J_color_all, cmock, ifield, nx=nx, ny=ny, twomass_Jmax=twom_Jmax)
    mask_twomass_simon, radii_mask_cat_twomass_simon = mask_from_df_cat(cat_df=twomass_lt_16, mag_lim=J_mag_lim, mode='Simon', magstr='j_m', Vega_to_AB=0., inst=inst, \
                                                                            a1=a1, b1=b1, c1=c1, dimx=nx, dimy=ny)

    print('2MASS catalog has length ', len(radii_mask_cat_twomass_simon))
    
    return mask_unWISE_PS, mask_twomass_simon, mask_cat_unWISE_PS




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


