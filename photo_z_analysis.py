import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
from astropy.table import Table
import pandas as pd


def compute_kde_grid(cat_df, range1, range2, key1='r', key2='redshift', target='spec_redshift', minpts=2):
    all_kdes = []
    for i in range(len(range1)-1):
        kde_list = []
        for j in range(len(range2)-1):
            
            cut = cat_df.loc[(cat_df[key1]>range1[i]) & (cat_df[key1]<range1[i+1])\
                            &(big_df[key2]>range2[j])&(cat_df[key2]<range2[j+1])]
        
            if len(cut[target]) >= minpts :
                kde = scipy.stats.gaussian_kde(cut[target])
            else:
                kde = None
            kde_list.append(kde)
            

        all_kdes.append(kde_list)
        
    return all_kdes

def compute_colors(spec_cat, bands, keyword='spec_redshift'):

    colors = np.zeros(shape=(len(bands)-1, len(spec_cat[keyword])))

    for b in range(len(bands)-1):
        colors[b] = spec_cat[bands[b]]-spec_cat[bands[b+1]]
        
    return colors

def compute_color_errors(cat, bands, keyword='u'):
    color_errs = np.zeros(shape=(len(bands)-1, len(cat[keyword])))

    for b in range(len(bands)-1):
        color_errs[b] = np.sqrt(cat[bands[b]]**2+cat[bands[b+1]]**2)
        
    return color_errs


# lets start by making a function that computes the chi squared Mahalonobis distance between 
# one set of colors and the colors of the full dataset

def mahalonobis_distance_colors(test_colors, test_colors_errors, all_train_colors):
    dm = np.zeros(shape=(all_train_colors.shape[0],))
    for i in range(all_train_colors.shape[0]):
        dm[i] = np.nansum(((test_colors - all_train_colors[i,:])/(test_colors_errors))**2)

    return dm

# this determines the nearest neighbors according to the chi squared Mahalonobis distance, 
# as determined by the percent point function of the distribution, where the number of degrees
# of freedom is set by the number of colors

def nearest_neighbors_ppf(all_train_colors, Dm, z_train=None, df=4, q=0.68, zmax=3.0, nbins=30):
    cutoff = scipy.stats.chi2.ppf(q, df)
    mask = (Dm < cutoff)
    nn_colors = all_train_colors[mask,:]
    nn_Dm = Dm[mask]
    if z_train is not None:
        # compute redshift PDF
        nn_z = z_train[mask]
        
        return nn_colors, nn_Dm, nn_z
        
    return nn_colors, nn_Dm

