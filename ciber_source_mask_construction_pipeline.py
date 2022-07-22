import numpy as np
import matplotlib 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from catalog_utils import *
if sys.version_info[0]==2:
    from ciber_mocks import *
from cross_spectrum_analysis import *
from mask_source_classification import *
from masking_utils import *
from mkk_parallel import *


def find_alpha_beta(intercept, minrad=10, dm=3, pivot=16.):
    
    alpha_m = -(intercept - minrad)/dm
    beta_m = intercept - pivot*alpha_m
    
    return alpha_m, beta_m
    
def srcmask_predict_PanSTARRS_unWISE_UKIDSS(cat_unPSuk_train=None, cat_unPSuk_train_fpath=None, decision_tree=None, \
                                            feature_names=['rMeanPSFMag', 'iMeanPSFMag', 'gMeanPSFMag', 'zMeanPSFMag', 'yMeanPSFMag', 'mag_W1', 'mag_W2'], \
                                            cat_unPS=None, cat_unPS_fpath=None, maskmagstr='zMeanPSFMag_mask', max_depth_dt = 8,  \
                                             J_mag_lim=19.0, dimx=512, dimy=512, pixsize=7., beta_m=125, alpha_m=-5.5, \
                                           return_mask=True, return_mask_cat=True, verbose=False):
    
    
    '''
     
    Parameters
    ----------
    
    cat_unPSuk_train, unPS_cat : 
    
    cat_unPSuk_train_fpath, cat_unPS_fpath : 'str'
    
    decision_tree : , optional
        Default is 'None'.
    
    max_depth_dt : 'int', optional
        Maximum depth permitted for decision tree. Default is 8.
        
    feature_names : , optional
    
    maskmagstr : 'str', optional
        Default is 'zMeanPSFMag_mask'. 
        
    dimx, dimy : 'int', optional
        Default is 512.
        
    pixsize : 'float', optional
        Default is 7 (arcseconds).
    
    beta_m, alpha_m : 'float', optional
        Defaults are 125 and -5.5, respectively.
        
    return_mask, return_mask_cat : 'booleans'
        Default is 'True'.
    
    Returns
    -------
    
    mask_cat_unWISE_PS : 
    
    mask_unWISE_PS : '~numpy.ndarray'
    
    '''
    
    # if decision tree loaded in, make sure feature_names corresponds to same features trained on
    if decision_tree is None:
        
        if cat_unPSuk_train is None:
            if cat_unPSuk_train_fpath is not None:
                cat_unPSuk_train = pd.read_csv(cat_unPSuk_train_fpath)
            else:
                print('User needs to either provide a trained decision tree, a training catalog, or a path to the training catalog. Exiting..')
                return None
            

        decision_tree, classes_train, train_features = train_decision_tree(cat_unPSuk_train, feature_names=feature_names, J_mag_lim=J_mag_lim, \
                                                                  max_depth=max_depth_dt, outlablstr='j_Vega')
        
        
    # -------------------- load the unWISE + PanSTARRS merged catalog --------------------
    
    if cat_unPS is None:
        if cat_unPS_fpath is not None:
            
            cat_unPS = pd.read_csv(cat_unPS_fpath)
        else:
            print('User needs to either provide a catalog dataframe or a path to the catalog used for prediction. Exiting..')
            return None
    
    mask_cat_unWISE_PS = filter_mask_cat_dt(cat_unPS, decision_tree, feature_names)
    
    print('mask_cat_unWISE_PS has length ', len(mask_cat_unWISE_PS))
        
    zs_mask = np.array(mask_cat_unWISE_PS['zMeanPSFMag'])
    W1_mask = np.array(mask_cat_unWISE_PS['mag_W1'])
    
    colormask = ((~np.isinf(zs_mask))&(~np.isinf(W1_mask))&(~np.isnan(zs_mask))&(~np.isnan(W1_mask))&(np.abs(W1_mask) < 50)&(np.abs(zs_mask) < 50))

    median_z_W1_color = np.median(zs_mask[colormask]-W1_mask[colormask])
    
    # find any non-detections in z band and replace with W1 + mean z-W1 
    nanzs = ((np.isnan(zs_mask))|(np.abs(zs_mask) > 50)|(np.isinf(zs_mask)))
    zs_mask[nanzs] = W1_mask[nanzs]+median_z_W1_color

    # anything that is neither detected in z or W1 (~10 sources) set to z=J_mag_lim.
    still_nanz = ((np.isnan(zs_mask))|(np.isinf(zs_mask)))
    zs_mask[still_nanz] = J_mag_lim
    mask_cat_unWISE_PS[maskmagstr] = zs_mask

    if return_mask_cat and not return_mask:
        return mask_cat_unWISE_PS
    
    mask_unWISE_PS, radii_mask_cat_unWISE_PS = mask_from_df_cat(mask_cat_unWISE_PS, magstr=maskmagstr,\
                                                                beta_m=beta_m, alpha_m=alpha_m, pixsize=pixsize, dimx=dimx, dimy=dimy)

    if return_mask and not return_mask_cat:
        return mask_unWISE_PS
    
    return mask_unWISE_PS, mask_cat_unWISE_PS



def source_mask_construct_dt(ifield, inst, cmock, mask_cat_unWISE_PS=None, fieldstr_train = 'UDS',\
                             J_mag_lim=19.0, feature_names=None, max_depth=8, \
                            zkey = 'zMeanPSFMag', W1key='mag_W1', mean_z_J_color_all = 1.0925, \
                             mask_cat_directory='data/cats/masking_cats/', twomass_cat_directory='data/cats/2MASS/filt/', \
                            nx=1024, ny=1024, pixsize=7., \
                            # linear fit parameters
                            beta_m=125., alpha_m=-5.5, \
                            # Gaussian fit parameters
                            a1=252.8, b1=3.632, c1=8.52):


    ''' This is used in the CIBER 4th flight data analysis to construct bright source masks '''
    
    fieldstr_mask = cmock.ciber_field_dict[ifield]
    
    if mask_cat_unWISE_PS is None:
        if feature_names is None:
            feature_names=['rMeanPSFMag', 'iMeanPSFMag', 'gMeanPSFMag', 'zMeanPSFMag', 'yMeanPSFMag', 'mag_W1', 'mag_W2']

        if fieldstr_train == 'UDS': # default is to use UDS field as training set

            print('J_mag_lim heere is ', J_mag_lim)
            full_merged_cat_unPSuk_train = pd.read_csv(mask_cat_directory+'UDS/unWISE_PanSTARRS_UKIDSS_full_xmatch_merge_UDS.csv')

            decision_tree, classes_train, train_features = train_decision_tree(full_merged_cat_unPSuk_train, feature_names=feature_names, J_mag_lim=J_mag_lim, \
                                                                          max_depth=max_depth, outlablstr='j_Vega')

        else: # if not UDS, use the cleaned IBIS catalog with unWISE/PS for the training field

            full_merged_cat_unPSIB_train = pd.read_csv(mask_cat_directory+fieldstr_train+'/unWISE_PanSTARRS_IBIS_full_xmatch_merge_'+fieldstr_train+'.csv')

            decision_tree, classes_train, train_features = train_decision_tree(full_merged_cat_unPSIB_train, feature_names=feature_names, J_mag_lim=J_mag_lim, \
                                                                      max_depth=max_depth)


        full_merged_cat_unWISE_PS = pd.read_csv(mask_cat_directory+fieldstr_mask+'/unWISE_PanSTARRS_full_xmatch_merge_'+fieldstr_mask+'_121620.csv')
        features_merged_cat_unWISE_PS = feature_matrix_from_df(full_merged_cat_unWISE_PS, feature_names=feature_names)

        predictions_CIBER_field_unWISE_PS = decision_tree.predict(features_merged_cat_unWISE_PS)
        mask_cat_unWISE_PS = filter_mask_cat_dt(full_merged_cat_unWISE_PS, decision_tree, feature_names)


    # we will use the Zemcov+14 masking radius formula based on z-band magnitudes when available, and 
    # when z band is not available for a source we will use W1 + mean(z - W1) for the effective magnitude
    zs_mask = np.array(mask_cat_unWISE_PS[zkey])
    W1_mask = np.array(mask_cat_unWISE_PS[W1key])
    colormask = ((~np.isinf(zs_mask))&(~np.isinf(W1_mask))&(~np.isnan(zs_mask))&(~np.isnan(W1_mask))&(np.abs(W1_mask) < 50)&(np.abs(zs_mask) < 50))
    median_z_W1_color = np.median(zs_mask[colormask]-W1_mask[colormask])

    print('median z - W1 is ', median_z_W1_color)

    # find any non-detections in z band and replace with W1 + mean z-W1 
    nanzs = ((np.isnan(zs_mask))|(np.abs(zs_mask) > 50)|(np.isinf(zs_mask)))
    zs_mask[nanzs] = W1_mask[nanzs]+median_z_W1_color

    # anything that is neither detected in z or W1 (~10 sources) set to z=18.5.
    still_nanz = ((np.isnan(zs_mask))|(np.isinf(zs_mask)))
    zs_mask[still_nanz] = J_mag_lim
    zkey_mask = zkey+'_mask'
    
    mask_cat_unWISE_PS[zkey_mask] = zs_mask + 0.5

    # using the effective masking magnitudes, compute the source mask
    print('masking catalog has length ', len(mask_cat_unWISE_PS))
    mask_unWISE_PS, radii_mask_cat_unWISE_PS = mask_from_df_cat(mask_cat_unWISE_PS, magstr=zkey_mask,\
                                                                     beta_m=beta_m, alpha_m=alpha_m, pixsize=pixsize, inst=inst)

    print('now creating mask for 2MASS..')
    twomass = pd.read_csv(twomass_cat_directory+'2MASS_'+fieldstr_mask+'_filtxy.csv')
    print('field is ', fieldstr_mask)

    twomass_lt_16, srcmap_twomass_J_lt_16 = twomass_srcmap_masking_cat_prep(twomass, mean_z_J_color_all, cmock, ifield, nx=nx, ny=ny)
    mask_twomass_simon, radii_mask_cat_twomass_simon = mask_from_df_cat(twomass_lt_16, mode='Simon', magstr='j_m', Vega_to_AB=0.91, inst=inst, \
                                                                            a1=a1, b1=b1, c1=c1)

    return mask_unWISE_PS, mask_twomass_simon, mask_cat_unWISE_PS


def srcmask_binwise_opt(cbps, m_min, m_max, d_mag, ifield=4, inst=1, rad_min=7., rad_max=150., n_rad_bin=None, cat_df=None, cat_xs=None, cat_ys=None, cat_mags=None, cat_nuInu=None, pixsize=7., \
                       xkey='x', ykey='y', magkey='j_m', cmock=None, n_fine_bin=10, nwide=17, \
                       use_running_mask=False, running_mask=None, make_bright_mask=False, bright_mask_thresh=16, psthresh=1e-8, plot=False, \
                       observed_image=None, bright_nong_sub=False, bright_radmin=50., bright_drad=7., nong_mmax=7, nwide_cutout=100, inst_mask=None, \
                       a1=220., b1=3.632, c1=8.52):
    ''' 
    Radii are in units of arcseconds 
    
    PSFs are generated on first call of make_srcmap_temp_bank, and then grabbed directly from cmock.psf_temp_bank, 
    so if you want to generate a new set of PSFs from different field, need to instantiate ciber_mock again. 
    
    '''
    
    if cmock is None:
        cmock = ciber_mock(ciberdir='/Users/luminatech/Documents/ciber2/ciber/')

    if use_running_mask and running_mask is None:
        running_mask = np.ones((cbps.dimx, cbps.dimy))
        running_slice = np.zeros_like(running_mask)
        
    bright_mask = None
    if make_bright_mask:
        bright_mask = np.ones((cbps.dimx, cbps.dimy))
            
        print('Making bright source mask for anything brighter than '+str(bright_mask_thresh)+'..')

    mag_range = np.linspace(m_max, m_min, (m_max-m_min)*2+1)
    print('mag range :', mag_range)
    
    xx, yy = compute_meshgrids(cbps.dimx, cbps.dimy, compute_dots=False)
    
    if n_rad_bin is None:
        rad_range = np.arange(rad_min, rad_max) # enumerate all radii
        n_rad_bin = len(rad_range)
    else:
        rad_range = np.linspace(rad_min, rad_max, n_rad_bin)
        
    if bright_nong_sub:
        rad_range_bright = np.arange(rad_min, rad_max, bright_drad)
        rad_range_bright = rad_range_bright[rad_range_bright > bright_radmin]
        
    all_ps_permagbin = np.zeros((len(mag_range), len(rad_range), cbps.n_ps_bin))
    min_rads = np.zeros_like(mag_range[:-1])

    if cat_df is not None:
        cat_xs = cat_df[xkey]
        cat_ys = cat_df[ykey]
        cat_mags = cat_df[magkey]
        
    if cat_nuInu is None:
        cat_nuInu = cmock.mag_2_nu_Inu(cat_mags, inst-1)
        
    cat_full = np.array([cat_xs, cat_ys, cat_mags, cat_nuInu]).transpose()
    cent_mag = 0.5*(mag_range[1:]+mag_range[:-1])
    all_good_ps = []
    
    for m, mag in enumerate(mag_range[:-1]):
        magmask = (cat_mags < mag)*(cat_mags >= mag_range[m+1])
        print(mag, mag_range[m+1])
        # if we have sources in the bin, do the optimization
        if np.sum(magmask)==0:
            print('No sources in this mag bin, moving on..')
        
        else:
            mask_cat = cat_full[magmask,:]
            mask_cat_mag = cat_mags[magmask]
            
            # generate source image from slice
            
            srcmap_slice = cmock.make_srcmap_temp_bank(ifield, inst, mask_cat, flux_idx=-1, n_fine_bin=10, nwide=17)
            radmap_slice = compute_radmap_full(cat_ys[magmask], cat_xs[magmask], xx, yy)

            if mag < nong_mmax and len(cat_xs[magmask])>0:
                
                if bright_nong_sub and observed_image is not None:
                    cl_masked = None
                    print('we are going to look at the bright sources in the observed image..')

                    obs_image_bright = observed_image.copy()

                    if running_mask is not None:
                        obs_image_bright *= running_mask
                    if inst_mask is not None:
                        obs_image_bright *= inst_mask

                    obs_stamps, x0s, y0s = grab_src_stamps(obs_image_bright, cat_xs[magmask], cat_ys[magmask], nwide=nwide_cutout)

                    print('rad range bright:', rad_range_bright)

                    bright_radii = []

                    for o, obs_stamp in enumerate(obs_stamps):   
                        meansub_levels = []
                        nmasks = []
                        count = 0
                        for r, rad_bright in enumerate(rad_range_bright):

                            mask_bright, _, thresh_bright = mask_from_cat([int(nwide_cutout//2)], [int(nwide_cutout//2)], dimx=nwide_cutout, dimy=nwide_cutout, fixed_radius=rad_bright)
                            nmask = np.count_nonzero(mask_bright)
                            nmasks.append(nmask)

                            meansub_im = obs_stamp*mask_bright
                            meansub_levels.append(np.mean(meansub_im[meansub_im != 0.]))
                            dms=meansub_levels[r-1]-meansub_levels[r]

                            if r > 0 and nmasks[r] < nmasks[r-1] and dms < 0.3 and dms != 0.:
                                count += 1
                                continue

                            if count > 0:
                                print(meansub_levels)
                                print(nmasks)
                                print(meansub_levels[r-1], meansub_levels[r], rad_bright, 'moving on')
                                bright_radii.append(rad_bright)

                                plot_map(obs_stamp*mask_bright, title='obs stamp '+str(o)+', mean = '+str(np.round(meansub_levels[r], 2)), hipct=99)
                                break

                    print('bright radii are', bright_radii)
                    if len(bright_radii)==0:
                        bright_radii = [rad_max for x in range(len(obs_stamps))]

                else:
                    bright_radii = radius_vs_mag_gaussian(mask_cat_mag, a1=a1, b1=b1, c1=c1)
                    
                mask, _ = mask_from_cat(cat_xs[magmask], cat_ys[magmask], radii=bright_radii, compute_radii=False)
                rad = np.mean(bright_radii)
        
                
            
            else:
                for r, rad in enumerate(rad_range):

                    mask, _, thresh = mask_from_cat(radmap_full=radmap_slice, fixed_radius=rad)
                    if inst_mask is not None:
                        mask *= inst_mask

                    masked_im_meansub = srcmap_slice*mask

                    if use_running_mask:
                        masked_im_meansub *= running_mask
                        masked_im_meansub[running_mask*mask != 0] -= np.mean(masked_im_meansub[running_mask*mask != 0])
                    else:
                        masked_im_meansub[mask != 0] -= np.mean(masked_im_meansub[mask != 0])

                    
                        
                    # compute power spectrum
                    lb, cl_masked, clerr_masked = get_power_spec(masked_im_meansub, lbinedges=cbps.Mkk_obj.binl, lbins=cbps.Mkk_obj.midbin_ell)
                    mask_frac = np.count_nonzero(mask)/float(cbps.dimx**2)
                    cl_masked /= mask_frac # correct ps due to mask to first order
                    all_ps_permagbin[m, r] = cl_masked

                    if mag < 12:
                        fac = 0.2
                    else:
                        fac = 1.

                    if (cl_masked < fac*psthresh).all():
                        if inst_mask is not None: # then recompute the source mask without inst mask included
                            mask, _, thresh = mask_from_cat(radmap_full=radmap_slice, fixed_radius=rad)

                        print('all cls are below ps thresh! moving on')
                        plot_map(srcmap_slice, title='srcmap slic')
                        
#                         plot_map(masked_im_meansub, title='masked im meansub')
                        break
            
            all_good_ps.append(cl_masked)
            
            min_rads[m] = rad
            

            if inst_mask is not None and running_mask is not None:
                mask_frac = np.count_nonzero(mask*running_mask*inst_mask)/float(cbps.dimx**2)
                print('mask frac (with running mask and instrument mask) is', mask_frac)

            
            if use_running_mask:
                running_mask *= mask
                running_slice += srcmap_slice
                
                if make_bright_mask and mag < bright_mask_thresh:
                    bright_mask *= mask

                if plot:
                    if inst_mask is not None:
                        plot_map(running_slice*running_mask*inst_mask, title='running mask x slice x inst mask', hipct=99.)
                    else:
                        plot_map(running_slice*running_mask, title='running mask x slice', hipct=99.)

            if plot:
                prefac = lb*(lb+1)/(2*np.pi)
                plt.figure(figsize=(6, 5))
                plt.title(str(mag)+'< J < '+str(mag_range[m+1]), fontsize=18)
                plt.plot(lb, prefac*all_ps_permagbin[m, r], label='r='+str(np.round(rad, 1)))
                plt.plot(lb, prefac*psthresh, color='k', linestyle='dashed', label='Point source residual threshold')
                plt.legend(fontsize=14)
                plt.tick_params(labelsize=14)
                plt.xscale('log')
                plt.yscale('log')
                plt.xlabel('Multipole $\\ell$', fontsize=18)
                plt.ylabel('$D_{\\ell}$', fontsize=18)
                plt.ylim(1e-5, 1e3)
                plt.show()

        
    if use_running_mask:
        return all_ps_permagbin, min_rads, running_mask, running_slice, cent_mag, all_good_ps, bright_mask
       
    return all_ps_permagbin, min_rads, cent_mag, all_good_ps, bright_mask


def generate_grid_of_masks_dtmethod_2MASS_PanSTARRS_unWISE(twomass_mask_param_combos, train_fpath, unWISE_PS_mask_cat_fpath, twomass_cat_fpath, dimx, dimy, ifield=4, \
                                                        cmock=None, compute_mkk_mats=True, Mkk_obj=None, n_sims_Mkk=200, n_split=2, J_mag_lim = 19.0, intercept_mag=16.0, ciberdir = '/Users/luminatech/Documents/ciber2/ciber/', n_ps_bin=15, \
                                                          mean_z_J_color_all=1.0925, show=False, max_depth_dt=8, \
                                                    feature_names=['rMeanPSFMag', 'iMeanPSFMag', 'gMeanPSFMag', 'zMeanPSFMag', 'yMeanPSFMag', 'mag_W1', 'mag_W2'], \
                                                    srcmap_plot_str='UDS'):

    '''
    
    This function 
    Parameters
    ----------
    
    twomass_mask_param_combos : list of tuples
    
    train_fpath : string
    
    unWISE_PS_mask_cat_fpath : string
    
    twomass_cat_fpath : string
    
    dimx, dimy : ints
    
    ifield : int, optional
        Default is 4. 
        
    cmock : ciber_mock() object, optional
        Default is 'None'.
        
    Mkk_obj : Mkk_bare() object, optional
        Default is 'None'.
        
    n_sims_Mkk : int, optional
        Default is 200.
    
    n_split : int, optional
        Default is 2.
        
    J_mag_lim : float, optional
        Default is 19.0

    intercept_mag : float, optional
        Magnitude cutoff for 2MASS catalog. Default is 16.0
        
    n_ps_bin : int, optional
        Number of power spectrum bins between l_min and l_max. Default is 15. 
        
    mean_z_J_color_all : float, optional
        Mean z-J color used for consistent source masking when we don't necessarily have J band data. 
        Default is 1.0925. 
    
    
    Returns
    -------
    
    list_of_masks : list of length len(twomass_mask_param_combos)
    
    list_of_mkks : list of length len(twomass_mask_param_combos)
    
    list_of_inv_mkks : list of length len(twomass_mask_param_combos)
    
    
    '''
    
    if cmock is None:
        cmock = ciber_mock(ciberdir=ciberdir, pcat_model_eval=True)
    cmock.get_psf(nx=dimx, ny=dimy, ifield=ifield)
    
    print('field is ', cmock.ciber_field_dict[ifield])
    
    # if no Mkk object provided, generate one using dimx, dimy and n_ps_bin
    if Mkk_obj is None:
        Mkk_obj = Mkk_bare(dimx=dimx, dimy=dimy, ell_min=180*(1024./dimx), nbins=n_ps_bin)
    else:
        print('Using inputted Mkk object with n_ps_bin = '+str(Mkk_obj.nbins))
    Mkk_obj.precompute_mkk_quantities(precompute_all=True)
    
    # load 2MASS catalog and mean color estimate z-J to get bright catalog sources
    twomass_full_cat = pd.read_csv(twomass_cat_fpath)
    twomass_bright, srcmap_twomass_bright = twomass_srcmap_masking_cat_prep(twomass_full_cat, mean_z_J_color_all, cmock, ifield, nx=dimx, ny=dimy, twomass_Jmax=intercept_mag)
    
    # load training set, train decision tree, and then apply same decision tree to all of the mask combinations
    cat_train = pd.read_csv(train_fpath)
    decision_tree, classes_train, train_features = train_decision_tree(cat_train, feature_names=feature_names, J_mag_lim=J_mag_lim, \
                                                              max_depth=max_depth_dt, outlablstr='j_Vega')
    
    list_of_mkks, list_of_inv_mkks, list_of_masks = [], [], []
    
    # for each parameter combination: calculate piecewise masking function; predict masking catalog; generate mask; compute Mkk matrices
    for p, param_combo in enumerate(twomass_mask_param_combos):
        print('parameter combination is '+str(param_combo))
            
        mask_twomass_simon, radii = mask_from_df_cat(twomass_bright, mode='Simon', magstr='j_m', Vega_to_AB=0.91, dimx=dimx, dimy=dimy, \
                                                  a1=param_combo[0], b1=param_combo[1], c1=param_combo[2])
        
            
        intercept = radius_vs_mag_gaussian(intercept_mag, a1=param_combo[0], b1=param_combo[1], c1=param_combo[2])
        if len(param_combo)==4:
            alpha_m, beta_m = find_alpha_beta(intercept, minrad=param_combo[3], dm=3, pivot=intercept_mag)
        else:
            alpha_m, beta_m = find_alpha_beta(intercept, minrad=14., dm=3, pivot=intercept_mag)
        
        print('intercept, alpha, beta = ', intercept, alpha_m, beta_m)

        if show:
            plt.figure()
            full_range = np.linspace(np.min(twomass_bright['j_m']), 20, 100)
            plt.plot(full_range, radius_vs_mag_gaussian(full_range, a1=param_combo[0], b1=param_combo[1], c1=param_combo[2]))
            plt.plot(full_range, magnitude_to_radius_linear(full_range, beta_m=beta_m, alpha_m=alpha_m))
            plt.show()
        
        mask_unWISE_PS, mask_cat_unWISE_PS_test = srcmask_predict_PanSTARRS_unWISE_UKIDSS(decision_tree=decision_tree, \
                                                                                 cat_unPS_fpath = unWISE_PS_mask_cat_fpath, dimx=dimx, dimy=dimy, \
                                                                                 J_mag_lim=J_mag_lim, alpha_m=alpha_m, beta_m=beta_m)
        
        if show:
            
            # these are just to confirm that the mask is properly oriented/aligned with the observed CIBER images
            f = plot_srcmap_mask(mask_unWISE_PS*mask_twomass_simon, 'UDS', len(mask_cat_unWISE_PS_test)+len(radii))
            plt.figure(figsize=(10, 10))
            field_flight = fits.open('data/fluctuation_data/TM'+str(inst)+'/flight/field'+str(ifield)+'_flight.fits')[0].data
            plt.imshow(field_flight*mask_unWISE_PS*mask_twomass_simon*(-170.3608), vmin=np.percentile(field_flight*mask_unWISE_PS*mask_twomass_simon*(-170.3608), 5), vmax=np.percentile((-170.3608)*field_flight*mask_unWISE_PS*mask_twomass_simon, 95))
            plt.colorbar()
            plt.show()
            
            plt.figure(figsize=(10, 10))
            field_flight = fits.open('data/fluctuation_data/TM'+str(inst)+'/flight/field'+str(ifield)+'_flight.fits')[0].data
            plt.imshow(field_flight*(-170.3608), vmin=np.percentile((-170.3608)*field_flight, 5), vmax=np.percentile((-170.3608)*field_flight, 95))
            plt.colorbar()
            plt.show() 
            
        list_of_masks.append(mask_twomass_simon*mask_unWISE_PS)
        
        if compute_mkk_mats:
            print('Computing Mkk matrices..')
            Mkk_matrix = Mkk_obj.get_mkk_sim(list_of_masks[p], n_sims_Mkk, n_split=n_split)
            inv_Mkk_matrix = compute_inverse_mkk(Mkk_matrix)

            list_of_mkks.append(Mkk_matrix)
            list_of_inv_mkks.append(inv_Mkk_matrix)
    
            if show:
                plot_mkk_matrix(Mkk_matrix, logscale=True)
                plot_mkk_matrix(inv_Mkk_matrix, symlogscale=True, inverse=True)
        
        else:
            print('skipping mkk matrices, another time!')
        
    return list_of_masks, list_of_mkks, list_of_inv_mkks, Mkk_obj, cmock
        
    
def tuple_list_from_ranges(a_range, b_range, c_range, minrad_range=None):
    
    list_of_tuples = []
    
    for a in a_range:
        for b in b_range:
            for c in c_range:
                if minrad_range is not None:
                    for m in minrad_range:
                        list_of_tuples.append([a, b, c, m])
                else:
                    list_of_tuples.append([a,b,c])
                
                
    return list_of_tuples


