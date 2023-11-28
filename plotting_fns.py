import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits
# from mpl_toolkits.axes_grid.inset_locator import inset_axes
import numpy as np
from matplotlib import cm
from PIL import Image
import glob
import matplotlib.patches as mpatches


def plot_field_weights_ciber_bands(mock_field_cl_basepath, return_fig=True, show=True):
    

    fig = plt.figure(figsize=(10,4))

    linestyles = ['solid', 'dashed']
    textstrs = ['CIBER 1.1$\\mu$m\nMask $J<17.5$', 'CIBER 1.8$\\mu$m\nMask $H<17.0$']
    lams = [1.1, 1.8]
    for i, inst in enumerate([1, 2]):
        plt.subplot(1,3,inst)
        mock_all_field_cl_weights = np.load(mock_field_cl_basepath+'mock_all_field_cl_weights_TM'+str(inst)+'.npz')['mock_all_field_cl_weights']
        cl_sumweights = np.sum(mock_all_field_cl_weights, axis=0)
        lam = lams[i]
        for k in range(5):
            if k==0:
                lab = 'CIBER '+str(lam)+' $\\mu$m'
            else:
                lab = None
            plt.plot(lb, mock_all_field_cl_weights[k]/cl_sumweights, color='C'+str(k), linestyle=linestyles[i], label=lab)
        plt.text(200, 0.35, textstrs[i], fontsize=14)
        plt.axhline(0.2, alpha=0.5, linestyle='dashdot', color='k')
        plt.xscale('log')
        plt.xlabel('$\\ell$', fontsize=14)
        plt.ylabel('Field weights', fontsize=14)
        plt.tick_params(labelsize=11)
        plt.ylim(0, 0.45)
        plt.grid(alpha=0.3)

    plt.subplot(1,3,3)
    textstrs_plot = ['CIBER 1.1$\\mu$m', 'CIBER 1.8$\\mu$m']
    marker_list = ['*', 'o']
    for i, inst in enumerate([1, 2]):
        neff_frac = np.load(mock_field_cl_basepath+'mode_fraction_TM'+str(inst)+'.npz')['mode_fraction']
        plt.scatter(lb, neff_frac, marker=marker_list[i], color='k', label=textstrs_plot[i])
        
    plt.legend()
    plt.xscale('log')
    plt.xlabel('$\\ell$', fontsize=14)
    plt.ylim(0, 1)
    plt.tick_params(labelsize=11)
    plt.grid(alpha=0.5)
    plt.ylabel('$N_{eff}^{weighted}/N_{eff}^{unweighted}$', fontsize=14)
    plt.tight_layout()
    plt.tight_layout()
    if show:
        plt.show()
    
    if return_fig:
        return fig


def single_panel_observed_ps_results(inst, masking_maglim, observed_field_average_cl, field_average_error, all_mock_recov_ps, \
                                     include_dgl_ul=True, include_igl_helgason=True,\
                                     include_z14=False, include_perfield=True, startidx=0, endidx=-1, show=True, \
                                    return_fig=True, zcolor='grey', xlim=[1.5e2, 1e5], ylim=[1e-2, 4e3], textpos=[2e2, 2e2], \
                                    obs_labels=['4th flight field average'], obs_colors=['k'], figsize=(6,5), zorders=None):
    
    ifield_list = [4, 5, 6, 7, 8]
    
    lam_dict = dict({1:1.1, 2:1.8})
    lam_dict_z14 = dict({1:1.1, 2:1.6})
    bandstr_dict = dict({1:'J', 2:'H'})
        
    prefac = lb[startidx:endidx]*(lb[startidx:endidx]+1)/(2*np.pi)
       
    fig = plt.figure(figsize=figsize)
    
    if include_perfield:
        for fieldidx, ifield in enumerate(ifield_list):
            std_recov_mock_ps = 0.5*(np.percentile(all_mock_recov_ps[:,fieldidx,:], 84, axis=0)-np.percentile(all_mock_recov_ps[:,fieldidx,:], 16, axis=0))

            plt.errorbar(lb[startidx:endidx], prefac*observed_recov_ps[fieldidx,startidx:endidx], yerr=prefac*std_recov_mock_ps[startidx:endidx], label=cbps.ciber_field_dict[ifield], fmt='.', color='C'+str(fieldidx), alpha=0.3, capsize=4, markersize=10)
            
    if len(np.array(observed_field_average_cl).shape)==2:
        for obs_idx, obs_cl in enumerate(observed_field_average_cl):
            plt.errorbar(lb[startidx:endidx], prefac*observed_field_average_cl[obs_idx][startidx:endidx], yerr=(prefac)*np.median(np.abs(field_average_error), axis=0)[startidx:endidx],\
                         label=obs_labels[obs_idx], fmt='o', capthick=1.5, color=obs_colors[obs_idx], capsize=3, markersize=4, linewidth=2., \
                        zorder=zorders[obs_idx])
    else:     
        plt.errorbar(lb[startidx:endidx], prefac*observed_field_average_cl[startidx:endidx], yerr=(prefac)*np.median(np.abs(field_average_error), axis=0)[startidx:endidx], label=obs_labels[0], fmt='o', capthick=1.5, color='k', capsize=3, markersize=4, linewidth=2.)

    if include_dgl_ul:
    
        dgl_auto_pred = np.load(config.ciber_basepath+'data/dgl_auto_constraints_TM'+str(inst)+'_sfd_clean_053023.npz')
        lb_modl = dgl_auto_pred['lb_modl']
        best_ps_fit_av = dgl_auto_pred['best_ps_fit_av']
        AC_A1 = dgl_auto_pred['AC_A1']
        dAC_sq = dgl_auto_pred['dAC_sq']

        plt.plot(lb_modl, best_ps_fit_av*AC_A1**2, color='k', label='DGL', linestyle='solid')
        plt.fill_between(lb_modl, best_ps_fit_av*(AC_A1**2-dAC_sq), best_ps_fit_av*(AC_A1**2+dAC_sq), color='k', alpha=0.2)
        plt.fill_between(lb_modl, best_ps_fit_av*(AC_A1**2-2*dAC_sq), best_ps_fit_av*(AC_A1**2+2*dAC_sq), color='k', alpha=0.1)
       
    if include_igl_helgason:
        print('inst = ', inst)
        config_dict, pscb_dict, float_param_dict, fpath_dict = return_default_cbps_dicts()
        ciber_mock_fpath = config.ciber_basepath+'data/ciber_mocks/'
        fpath_dict, list_of_dirpaths, base_path, trilegal_base_path = set_up_filepaths_cbps(fpath_dict, inst, 'test', '112022',\
                                                                                        datestr_trilegal='112022', data_type=data_type, \
                                                                                       save_fpaths=True)

        ifield_choose = 4

        all_cl = []
        prefac_full = lb*(lb+1)/(2*np.pi)

        for cib_setidx in range(50):

            cib_cl_file = fits.open(fpath_dict['cib_resid_ps_path']+'/cls_cib_vs_maglim_ifield'+str(ifield_choose)+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_Vega_magcut.fits')
            isl_cl_file = fits.open(fpath_dict['isl_resid_ps_path']+'/cls_isl_vs_maglim_ifield'+str(ifield_choose)+'_inst'+str(inst)+'_simidx'+str(cib_setidx)+'_Vega_magcut.fits')

            all_cl.append(cib_cl_file['cls_cib'].data['cl_maglim_'+str(masking_maglim)] + isl_cl_file['cls_isl'].data['cl_maglim_'+str(masking_maglim)])

        plt.plot(lb, prefac_full*(np.mean(all_cl, axis=0)), linestyle='dashdot', color='grey', label='IGL+ISL')
       
    if include_z14:
        zemcov_auto = np.loadtxt(config.ciber_basepath+'/data/zemcov14_ps/ciber_'+str(lam_dict_z14[inst])+'x'+str(lam_dict_z14[inst])+'_dCl.txt', skiprows=8)
        zemcov_lb = zemcov_auto[:,0]
        plt.plot(zemcov_lb, zemcov_auto[:,1], label='Zemcov+14', marker='.', color=zcolor, alpha=0.3)
        plt.fill_between(zemcov_lb, zemcov_auto[:,1]-zemcov_auto[:,2], zemcov_auto[:,1]+zemcov_auto[:,3], color=zcolor, alpha=0.15)


    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.yscale('log')
    plt.xscale('log')
    plt.tick_params(labelsize=14)
    plt.xlabel('$\\ell$', fontsize=16)
    plt.ylabel('$D_{\\ell}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=16)
    plt.grid(alpha=0.5, color='grey')

    plt.text(textpos[0], textpos[1], 'CIBER '+str(lam_dict[inst])+'$\\mu$m\nObserved data\nMask '+bandstr_dict[inst]+'$<'+str(masking_maglim)+'$', fontsize=18)
    
    plt.legend(fontsize=10, loc=4, ncol=2)
    plt.tight_layout()
    
    if show:
        plt.show()
        
    if return_fig:
        return fig


def make_figure_cross_spec_vs_masking_magnitude(inst=1, cross_inst=2, run_name='ciber_cross_ciber_perquad_regrid_070523', maglim_J=[17.5, 18.0, 18.5, 19.0], \
                                               maglim_H=[17.0, 17.5, 18.0, 18.5], return_fig=True, show=True, startidx=1, endidx=-1):
    observed_run_names_cross = ['ciber_cross_ciber_perquad_regrid_071023_Jlt'+str(maglim_J[j])+'_Hlt'+str(maglim_H[j])+'_withcrossnoise' for j in range(len(maglim_J))]
    obs_labels = ['$J<'+str(maglim_J[m])+'\\times H<'+str(maglim_H[m])+'$' for m in range(len(maglim_J))]

    obs_fieldav_cross_cl, obs_fieldav_cross_dcl = [], []    
    obs_colors = ['indigo', 'darkviolet', 'mediumorchid', 'plum']
    
    for obs_name in observed_run_names_cross:
        cl_fpath_obs = 'data/input_recovered_ps/cl_files/TM'+str(inst)+'_TM'+str(cross_inst)+'_cross/cl_'+obs_name+'.npz'
        lb, observed_recov_ps, observed_recov_dcl_perfield,\
            observed_field_average_cl, observed_field_average_dcl,\
                mock_all_field_cl_weights = load_weighted_cl_file_cross(cl_fpath_obs)

        obs_fieldav_cross_cl.append(observed_field_average_cl)
        obs_fieldav_cross_dcl.append(observed_field_average_dcl)
        
        
    fig = plt.figure(figsize=(6,5))
    
    prefac = lb*(lb+1)/(2*np.pi)

    for m, maglim in enumerate(maglim_J):
        plt.errorbar(lb[startidx:endidx], (prefac*obs_fieldav_cross_cl[m])[startidx:endidx], yerr=(prefac*obs_fieldav_cross_dcl[m])[startidx:endidx], label=obs_labels[m], fmt='o', capthick=1.5, color=obs_colors[m], capsize=3, markersize=4, linewidth=2.)

    plt.xlim(2e2, 1e5)
    plt.yscale('log')
    plt.xscale('log')
    plt.tick_params(labelsize=14)
    plt.xlabel('$\\ell$', fontsize=16)
    plt.ylabel('$D_{\\ell}$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=18)
    plt.grid(alpha=0.5, color='grey')
    plt.ylim(1e-2, 1e3)
    plt.text(250, 150, 'CIBER 1.1$\\mu$m $\\times 1.8\\mu$m\nObserved data', fontsize=18)

    bbox_dict = dict({'facecolor':'white', 'alpha':0.8, 'edgecolor':'None', 'pad':0.})

    plt.legend(fontsize=12, loc=4, ncol=1, framealpha=1.)
    plt.tight_layout()

    if show:
        plt.show()
    if return_fig:
        return fig

def make_figure_cross_corrcoeff_ciber_ciber_vs_mag(maglim_J = [17.5, 18.0, 18.5, 19.0], maglim_H = [17.0, 17.5, 18.0, 18.5], show=True, return_fig=True, \
                                                  verbose=False, observed_run_names_cross=None):
    
    if observed_run_names_cross is None:
        observed_run_names_cross = ['ciber_cross_ciber_perquad_regrid_071023_Jlt'+str(maglim_J[j])+'_Hlt'+str(maglim_H[j])+'_withcrossnoise' for j in range(len(maglim_J))]
    obs_colors = ['indigo', 'darkviolet', 'mediumorchid', 'plum']
    all_r_TM, all_sigma_r_TM = [], []

    for o, obs_name_AB in enumerate(observed_run_names_cross):
    
        obs_name_A = 'observed_Jlim_Vega_'+str(maglim_J[o])+'_Hlim_Vega_'+str(maglim_H[o])+'_070723'
        obs_name_B = obs_name_A
        lb, r_TM, sigma_r_TM = ciber_ciber_rl_coefficient(obs_name_A, obs_name_B, obs_name_AB)
        all_r_TM.append(r_TM)
        all_sigma_r_TM.append(sigma_r_TM)
        
        if verbose:
            print('r_TM:', r_TM)
            print('sigma_r_TM:', sigma_r_TM)
        
    fig = plt.figure(figsize=(6,5))

    for obs_idx in range(len(all_r_TM)):

        lbmask_science = (lb < 2000)
        weights = 1./(all_sigma_r_TM[obs_idx][lbmask_science])**2
        weighted_average = np.sum(weights*all_r_TM[obs_idx][lbmask_science])/np.sum(weights)
        weighted_variance = 1./np.sum(weights)
        
        if verbose:
            print('weighted variance for lb < 2000:', weighted_variance)

        plt.subplot(2,2,obs_idx+1)
        plt.errorbar(lb[1:-1], all_r_TM[obs_idx][1:-1], yerr=all_sigma_r_TM[obs_idx][1:-1], fmt='o', capsize=3, color=obs_colors[obs_idx], \
                    label='$J<'+str(maglim_J[obs_idx])+'\\times H<$'+str(maglim_H[obs_idx]), markersize=4, capthick=1.5, alpha=0.8)
        plt.xscale('log')
        plt.ylim(-0.1, 1.25)
        bbox_dict = dict({'facecolor':'white', 'alpha':0.9, 'edgecolor':'k', 'linewidth':0.5})
        plt.text(1000, 1.05, '$J<'+str(maglim_J[obs_idx])+'\\times H<$'+str(maglim_H[obs_idx]), fontsize=12, color=obs_colors[obs_idx], bbox=bbox_dict)
        if obs_idx > 1:
            plt.xlabel('$\\ell$', fontsize=14)
        if obs_idx==0 or obs_idx==2:
            plt.ylabel('$r_{\\ell} = C_{\\ell}^{1.1\\times1.8}/\\sqrt{C_{\\ell}^{1.1}C_{\\ell}^{1.8}}$', fontsize=12)
        plt.tick_params(labelsize=11)

        if obs_idx==1 or obs_idx==3:
            plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25], ['', '', '', '', '', ''])
        plt.grid()
    plt.tight_layout()
    if show:
        plt.show()
    if return_fig:
        return fig

def plot_ciber_field_consistency_test(inst, lb, observed_recov_ps, all_mock_signal_ps, mock_all_field_cl_weights, \
                                      lmax=10000, startidx=1, endidx=None, mode='chi2', \
                                    ybound=5, observed_run_name=None):
    
    ciber_field_dict = dict({4:'elat10', 5:'elat30',6:'BootesB', 7:'BootesA', 8:'SWIRE', 'train':'UDS'})


    ifield_list = [4, 5, 6, 7, 8]
    all_chistat = []
    
    lbmask_chistat = (lb < lmax)*(lb >= lb[startidx])

    if mode=='chi2':
        observed_chi2_red = []
    
    f = plt.figure(figsize=(10, 7))
    prefac = lb*(lb+1)/(2*np.pi)
    
    if observed_run_name is not None:
        all_resid_data_matrices = np.load('/Users/richardfeder/Downloads/ciber_field_consistency_cov_matrices_TM'+str(inst)+'_'+observed_run_name+'.npz')['all_resid_data_matrices']
        resid_joint_data_matrix = np.load('/Users/richardfeder/Downloads/ciber_field_consistency_cov_matrices_TM'+str(inst)+'_'+observed_run_name+'.npz')['resid_joint_data_matrix']
        
        cov_joint = np.cov(resid_joint_data_matrix.transpose())
        
        all_cov_indiv_full = [np.cov(all_resid_data_matrices[fieldidx,:,lbmask_chistat]) for fieldidx in range(len(ifield_list))]
        all_cov_indiv_full = np.array(all_cov_indiv_full)
        
    else:
        all_cov_indiv_full = None
        cov_joint = None
        
    all_inv_cov_indiv_lbmask = np.zeros_like(all_cov_indiv_full)
    if all_cov_indiv_full is not None:
        for fieldidx, ifield in enumerate(ifield_list):
            all_inv_cov_indiv_lbmask[fieldidx] = np.linalg.inv(all_cov_indiv_full[fieldidx])
        
    chistat_perfield_mock, pte_perfield_mock, all_chistat_largescale = mock_consistency_chistat(lb, all_mock_recov_ps, mock_all_field_cl_weights, lmax=lmax, mode=mode, all_cov_indiv_full=all_cov_indiv_full, \
                                                                                               cov_joint=None)

    for fieldidx, ifield in enumerate(ifield_list):
        ax = f.add_subplot(2,3,fieldidx+1)
            
            
        mean_cl_obs = observed_recov_ps[fieldidx]
        std_recov_mock_ps = np.std(all_mock_recov_ps[:,fieldidx], axis=0)
        tot_std_ps = np.sqrt(std_recov_mock_ps**2)
        plt.axhline(0, linestyle='dashed', color='k', linewidth=2)
        plt.axvline(lmax, color='k', linestyle='dashed', linewidth=1.5)

        plt.errorbar(lb[startidx:endidx], ((mean_cl_obs-observed_field_average_cl)/observed_field_average_cl)[startidx:endidx], yerr=(tot_std_ps/observed_field_average_cl)[startidx:endidx],\
                     label=ciber_field_dict[ifield], fmt='.', color='C'+str(fieldidx), zorder=2, alpha=1.0,\
                     capthick=2, capsize=3, linewidth=2, markersize=5)

        resid = observed_field_average_cl-mean_cl_obs

        if mode=='chi2':
            if all_cov_indiv_full is not None:
            
                chistat_mean_cl_mockstd = np.multiply(resid[lbmask_chistat].transpose(), np.dot(all_inv_cov_indiv_lbmask[fieldidx], resid[lbmask_chistat]))
            else:
            
                chistat_mean_cl_mockstd = resid**2/(tot_std_ps**2)
        elif mode=='chi':
            chistat_mean_cl_mockstd = resid/tot_std_ps
        
        all_chistat.append(chistat_mean_cl_mockstd)

        if all_cov_indiv_full is not None:
            
            chistat_largescale = np.array(all_chistat[fieldidx])
            print('chistat large scale:', chistat_largescale)

        else:
            chistat_largescale = np.array(all_chistat[fieldidx])[lbmask_chistat]
            
        ndof = len(lb[lbmask_chistat])
        if mode=='chi2':
            chi2_red = np.sum(chistat_largescale)/ndof
            observed_chi2_red.append(chi2_red)
            chistat_info_string = '$\\chi^2:$'+str(np.round(np.sum(chistat_largescale), 1))+'/'+str(ndof)+' ('+str(np.round(chi2_red, 2))+')'
        elif mode=='chi':
            chistat_info_string = '$\\chi:$'+str(np.round(np.sum(chistat_largescale), 1))+'/'+str(ndof)
            
            
        if startidx > 0:
            plt.axvline(0.5*(lb[startidx]+lb[startidx-1]), color='k', linestyle='dashed', linewidth=1.5)
        else:
            plt.axvline(0.7*lb[0], color='k', linestyle='dashed', linewidth=1.5)

        bbox_dict = dict({'facecolor':'white', 'alpha':0.8, 'edgecolor':'None', 'pad':0.})
        plt.text(3e2, ybound*0.5, 'TM'+str(inst)+', '+ciber_field_dict[ifield]+'\nObserved data\n'+chistat_info_string, color='C'+str(fieldidx), fontsize=11, \
            bbox=bbox_dict)
            
        plt.tick_params(labelsize=12)
        if fieldidx==0 or fieldidx==3:
            plt.ylabel('Fractional deviation $\\Delta C_{\\ell}/\\langle \\hat{C}_{\\ell}\\rangle$', fontsize=12)
        plt.xlabel('$\\ell$', fontsize=16)
        plt.grid(alpha=0.2, color='grey')

        plt.xscale('log')
        plt.ylim(-ybound, ybound)
        
        
        axin = inset_axes(ax, 
                width="45%", # width = 30% of parent_bbox
                height=0.75, # height : 1 inch
                loc=4, borderpad=1.6)
            
        
        chistat_mock = all_chistat_largescale[fieldidx]
        chistat_mock /= ndof
            
        chistat_order_idx = np.argsort(chistat_mock)
        if mode=='chi2':
            bins = np.linspace(0, 4, 30)
        elif mode=='chi':
            bins = np.linspace(-2, 2, 30)
            
        plt.hist(chistat_mock, bins=bins, linewidth=1, histtype='stepfilled', color='k', alpha=0.2, label=ciber_field_dict[ifield])
        plt.axvline(np.median(chistat_mock), color='k', alpha=0.5)
        print(np.median(chistat_mock), 'median chi squared')

        if mode=='chi2':
            pte = 1.-(np.digitize(observed_chi2_red[fieldidx], chistat_mock[chistat_order_idx])/len(chistat_mock))
            plt.axvline(chi2_red, color='C'+str(fieldidx), linestyle='solid', linewidth=2, label='Observed data')
            axin.set_xlabel('$\\chi^2_{red}$', fontsize=10)

        elif mode=='chi':
            pte = 1.-(np.digitize(np.sum(chistat_largescale)/ndof, chistat_mock[chistat_order_idx])/len(chistat_mock))
            plt.axvline(np.sum(chistat_largescale)/ndof, color='C'+str(fieldidx), linestyle='solid', linewidth=2, label='Observed data')
            axin.set_xlabel('$\\chi_{red}$', fontsize=10)

        axin.xaxis.set_label_position('top')
        
        if mode=='chi2':
            plt.xticks([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
            plt.xlim(0, 4)
            xpos_inset_text = 1.5

        else:
            plt.xticks([-1, 0, 1], [-1, 0, 1])
            plt.xlim(-1.5, 1.5)
            xpos_inset_text = -1.4

        
        axin.tick_params(labelsize=8,bottom=True, top=True, labelbottom=True, labeltop=False)

        plt.yticks([], [])
        
        hist = np.histogram(chistat_mock, bins=bins)
        
        plt.text(xpos_inset_text, 0.8*np.max(hist[0]), 'Mocks', color='grey', fontsize=9, bbox=bbox_dict)
        plt.text(xpos_inset_text, 0.6*np.max(hist[0]), 'PTE='+str(np.round(pte, 2)), color='C'+str(fieldidx), fontsize=9, bbox=bbox_dict)
            
    plt.tight_layout()
    plt.show()
    
    return f


def plot_nframe_difference(cl_diffs_flight, cl_diffs_shotnoise, ifield_list, cl_elat30=None, ifield_elat30=5, nframe=10, show=True, return_fig=True, \
                          titlefontsize=20):
    
    labfs = 18
    legendfs = 12
    tickfs = 14
    
    cl_diffs_flight = np.array(cl_diffs_flight)
    cl_diffs_shotnoise = np.array(cl_diffs_shotnoise)
    clav = np.mean(cl_diffs_flight-cl_diffs_shotnoise, axis=0)
    
    f = plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    
    # show Dl power spectra
    plt.title(str(nframe)+'-frame differences', fontsize=titlefontsize)
    plt.errorbar(lb, prefac*np.mean(cl_diffs_flight-cl_diffs_shotnoise, axis=0), yerr=prefac*np.std(cl_diffs_flight-cl_diffs_shotnoise, axis=0),\
            linewidth=1.5, color='k', linestyle='solid', capsize=3, capthick=2, label='Field average\n(no elat30)')
    
    if cl_elat30 is not None:
        plt.errorbar(lb, prefac*cl_elat30, label='elat30 noise model', linestyle='dashed', color='C1', linewidth=2, zorder=2)
    
    plot_colors = ['C0', 'C2', 'C3', 'C4']

    for i, ifield in enumerate(ifield_list):
        snlab, fsnlab = None, ''
        if i==0:
            snlab = str(nframe)+'-frame photon noise'
            fsnlab = 'Flight diff. - $N_{\\ell}^{phot}$\n'

        fsnlab += cbps.ciber_field_dict[ifield]
        zorder=0
    
        if ifield==5:
            zorder=10
            
        plt.plot(lb, prefac*(cl_diffs_flight[i]-cl_diffs_shotnoise[i]), zorder=zorder, linewidth=2, color=plot_colors[i], linestyle='solid', label=fsnlab)

    plt.yscale('log')
    plt.xscale('log')
    plt.legend(fontsize=legendfs, loc=4)
    plt.xlabel('Multipole $\\ell$', fontsize=labfs)
    plt.ylabel('$\\ell(\\ell+1)C_{\\ell}/2\\pi$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=labfs)
    plt.tick_params(labelsize=tickfs)
    
    # show ratio with respect to non-elat30 fields (nframe=10) or full five-field average (nframe=5)

    plt.subplot(1,2,2)
    plt.title(str(nframe)+'-frame noise spectra', fontsize=titlefontsize)
    for i, ifield in enumerate(ifield_list):

        plt.plot(lb, (cl_diffs_flight[i]-cl_diffs_shotnoise[i])/clav, label=cbps.ciber_field_dict[ifield], color=plot_colors[i], linewidth=2)
    
    if cl_elat30 is not None:
        plt.plot(lb, cl_elat30/clav, linewidth=2, label='elat30', linestyle='dashed', color='C1', zorder=2)
    plt.axhline(1.0, linestyle='dashed', color='k', label='Field average\n(no elat30)')
        
    plt.ylabel('$N_{\\ell}^{read}/\\langle N_{\\ell}^{read}\\rangle$', fontsize=labfs)
    plt.xscale('log')
    plt.xlabel('Multipole $\\ell$', fontsize=labfs)
    plt.tick_params(labelsize=tickfs)
    
    plt.tight_layout()

    
    if show:
        plt.show()
    if return_fig:
        return f
        


def plot_g1_hist(g1facs, mask=None, return_fig=True, show=True, title=None):
    if mask is not None:
        g1facs = g1facs[mask]
        
    std_g1 = np.std(g1facs)/np.sqrt(len(g1facs))
    f = plt.figure(figsize=(6,5))
    if title is not None:
        plt.title(title, fontsize=18)
    plt.hist(g1facs)
    plt.axvline(np.median(g1facs), label='Median $G_1$ = '+str(np.round(np.median(g1facs), 3))+'$\\pm$'+str(np.round(std_g1, 3)), linestyle='dashed', color='k', linewidth=2)
    plt.axvline(np.mean(g1facs), label='Mean $G_1$ = '+str(np.round(np.mean(g1facs), 3))+'$\\pm$'+str(np.round(std_g1, 3)), linestyle='dashed', color='r', linewidth=2)
    plt.xlabel('$G_1$ [(e-/s)/(ADU/fr)]', fontsize=16)
    plt.ylabel('N', fontsize=16)
    plt.legend(fontsize=14)
    plt.tick_params(labelsize=14)
    
    if show:
        plt.show()
    
    if return_fig:
        return f
        

def plot_noise_modl_val_flightdiff(lb, N_ells_modl, N_ell_flight, shot_noise=None, title=None, ifield=None, inst=None, show=True, return_fig=True):
    
    ''' 
    
    Parameters
    ----------
    
    lb : 
    N_ells_modl : 'np.array' of type 'float' and shape (nsims, n_ps_bins). Noise model power spectra
    N_ell_flight : 'np.array' of type 'float' and shape (n_ps_bins). Flight half difference power spectrum
    ifield (optional) 'str'.
        Default is None.
    inst (optional) : 'str'.
        Default is None.
        
    show (optional) : 'bool'. 
        Default is True.
    return_fig (optional) : 'bool'.
        Default is True.
    
    
    Returns
    -------
    
    f (optional): Matplotlib figure
    
    
    '''
    ciber_field_dict = dict({4:'elat10', 5:'elat30',6:'BootesB', 7:'BootesA', 8:'SWIRE'})

    # compute the mean and scatter of noise model power spectra realizations
    
    mean_N_ell_modl = np.mean(N_ells_modl, axis=0)
    std_N_ell_modl = np.std(N_ells_modl, axis=0)
    
    print('mean Nell shape', mean_N_ell_modl.shape)
    print('std Nell shape', std_N_ell_modl.shape)
    print(shot_noise)
    print(N_ell_flight.shape)

    
    prefac = lb*(lb+1)/(2*np.pi)
    
    f = plt.figure(figsize=(8,6))
    if title is None:
        if inst is not None and ifield is not None:
            title = 'TM'+str(inst)+' , '+ciber_field_dict[ifield]
            
    plt.title(title, fontsize=18)
    
    plt.plot(lb, prefac*N_ell_flight, label='Flight difference', color='r', linewidth=2, marker='.') 
    if shot_noise is not None:
        plt.plot(lb, prefac*shot_noise, label='Photon noise', color='C1', linestyle='dashed')
    plt.errorbar(lb, prefac*(mean_N_ell_modl), yerr=prefac*std_N_ell_modl, label='Read noise model', color='k', capsize=5, linewidth=2)
    
    plt.legend(fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.tick_params(labelsize=14)
    
    if show:
        plt.show()
        
    if return_fig:
        return f

def plot_cumulative_numbercounts(cat, mag_idx, field=None, df=True, m_min=18, m_max=30, vlines=None, hlines=None, \
                                nbin=100, label=None, pdf_or_png='pdf', band_label='W1', show=True, \
                                  return_fig=True, magsys='AB'):
    f = plt.figure()
    title = 'Cumulative number count distribution'
    if field is not None:
        title += ' -- '+str(field)
    plt.title(title, fontsize=14)
    magspace = np.linspace(m_min, m_max, nbin)
    if df:
        cdf_nm = np.array([len(cat.loc[cat[mag_idx]<x]) for x in magspace])
    else:
        print(len(cat))
        cdf_nm = np.array([len(cat[(cat[:,mag_idx] < x)]) for x in magspace])
    plt.plot(magspace, cdf_nm, label=label)
    if vlines is not None:
        for vline in vlines:
            plt.axvline(vline, linestyle='dashed')
    if hlines is not None:
        for hline in hlines:
            plt.axhline(hline, linestyle='dashed')
    plt.yscale('log')
    plt.ylabel('N['+band_label+' < magnitude]', fontsize=14)
    plt.xlabel(band_label+' magnitude ('+magsys+')', fontsize=14)
    plt.legend()
    if show:
        plt.show()
    if return_fig:
        return f

def plot_number_counts_uk(df, binedges=None, bands=['yAB', 'jAB', 'hAB', 'kAB'], magstr='', add_title_str='', return_fig=False):
    if binedges is None:
        binedges = np.arange(7,23,0.5)

    f = plt.figure()
    plt.title(magstr+add_title_str, fontsize=16)
    bins = (binedges[:-1] + binedges[1:])/2
    
    for band in bands:
        d = df[band+magstr]
        shist,_ = np.histogram(d, bins = binedges)
        plt.plot(bins,shist / (binedges[1]-binedges[0]) / 4,'o-',label=band)

    plt.yscale('log')
    plt.xlabel('$m_{AB}$', fontsize=14)
    plt.ylabel('counts / mag / deg$^2$', fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()
    if return_fig:
        return f

def plot_number_counts(df, binedges=None, bands=['g', 'r', 'i', 'z', 'y'], magstr='MeanPSFMag', add_title_str='', return_fig=False):
    if binedges is None:
        binedges = np.arange(7,23,0.5)

    f = plt.figure()
    plt.title(magstr+add_title_str, fontsize=16)
    bins = (binedges[:-1] + binedges[1:])/2
    
    for band in bands:
        d = df[band+magstr]
        shist,_ = np.histogram(d, bins = binedges)
        plt.plot(bins,shist / (binedges[1]-binedges[0]) / 4,'o-',label=band)

    plt.yscale('log')
    plt.xlabel('$m_{AB}$', fontsize=14)
    plt.ylabel('counts / mag / deg$^2$', fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()
    if return_fig:
        return f

def plot_dm_powerspec(show=True):
    
    mass_function = MassFunction(z=0., dlog10m=0.02)
    
    f = plt.figure()
    plt.title('Halo Model Dark Matter Power Spectrum, z=0')
    # plt.plot(mass_function.k, mass_function.nonlinear_delta_k, label='halofit')
    plt.plot(mass_function.k, mass_function.nonlinear_power, label='halofit')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.ylabel('$P(k)$ [$h^{-3}$ $Mpc^3$]', fontsize=14)
    plt.xlabel('$k$ [h $Mpc^{-1}$]', fontsize=14)
    plt.ylim(1e-4, 1e5)
    plt.xlim(1e-3, 1e3)
    if show:
        plt.show()
    
    return f

def plot_map(image, figsize=(8,8), title=None, titlefontsize=16, xlabel='x [pix]', ylabel='y [pix]',\
             x0=None, x1=None, y0=None, y1=None, lopct=5, hipct=99,\
             return_fig=False, show=True, nanpct=True, cl2d=False, cmap='viridis', noxticks=False, noyticks=False, \
             cbar_label=None, norm=None, vmin=None, vmax=None, scatter_xs=None, scatter_ys=None, scatter_marker='x', scatter_color='r', \
             interpolation='none', cbar_fontsize=14, xylabel_fontsize=16, tick_fontsize=14, \
             textstr=None, text_xpos=None, text_ypos=None, bbox_dict=None, text_fontsize=16, origin='lower'):

    f = plt.figure(figsize=figsize)



    if vmin is None:
        vmin = np.nanpercentile(image, lopct)
    if vmax is None:
        vmax = np.nanpercentile(image, hipct)

    if title is not None:
    
        plt.title(title, fontsize=titlefontsize)

    print('min max of image in plot map are ', np.min(image), np.max(image))
    plt.imshow(image, vmin=vmin, vmax=vmax, cmap=cmap, interpolation=interpolation, origin=origin, norm=norm)

    # if nanpct:
    #     plt.imshow(image, vmin=np.nanpercentile(image, lopct), vmax=np.nanpercentile(image, hipct), cmap=cmap, interpolation='None', origin='lower', norm=norm)
    # else:
    #     plt.imshow(image, cmap=cmap, origin='lower', interpolation='none', norm=norm)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14)
    if cbar_label is not None:
        cbar.set_label(cbar_label, fontsize=cbar_fontsize)

    if scatter_xs is not None and scatter_ys is not None:
        plt.scatter(scatter_xs, scatter_ys, marker=scatter_marker, color=scatter_color)
    if x0 is not None and x1 is not None:
        plt.xlim(x0, x1)
        plt.ylim(y0, y1)
        
    if cl2d:
        plt.xlabel('$\\ell_x$', fontsize=xylabel_fontsize)
        plt.ylabel('$\\ell_y$', fontsize=xylabel_fontsize)
    else:
        plt.xlabel(xlabel, fontsize=xylabel_fontsize)
        plt.ylabel(ylabel, fontsize=xylabel_fontsize)

    if noxticks:
        plt.xticks([], [])
    if noyticks:
        plt.yticks([], [])

    if textstr is not None:
        if bbox_dict is None:
            bbox_dict = dict({'facecolor':'white', 'edgecolor':'None', 'alpha':0.7})
        plt.text(text_xpos, text_ypos, textstr, fontsize=text_fontsize, bbox=bbox_dict)
        
    plt.tick_params(labelsize=tick_fontsize)

    if show:
        plt.show()
    if return_fig:
        return f

def compute_all_intermediate_power_spectra(lb, cls_inter, inter_labels, signal_ps=None, fieldidx=0, show=True, return_fig=True):
    
    f = plt.figure(figsize=(8, 6))

    for interidx, inter_label in enumerate(inter_labels):
        prefac = lb*(lb+1)/(2*np.pi)
        plt.plot(lb, prefac*cls_inter[interidx][fieldidx], marker='.',  label=inter_labels[interidx], color='C'+str(interidx%10))

    if signal_ps is not None:
        plt.plot(lb, prefac*signal_ps[fieldidx], marker='*', label='Signal PS', color='k', zorder=2)
    plt.legend(fontsize=14, bbox_to_anchor=[1.01, 1.3], ncol=3, loc=1)
    plt.ylabel('$D_{\\ell}$', fontsize=18)
    plt.xlabel('Multipole $\\ell$', fontsize=18)
    plt.tick_params(labelsize=16)
    plt.text(200, 3e3, 'ifield='+str(fieldidx+4)+' (TM2)', color='C2', fontsize=24)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(alpha=0.5, color='grey')
    plt.ylim(1e-1, 1e4)
    # plt.savefig('/Users/luminatech/Downloads/ps_step_by_step_bootesB_tm1_072122.png', bbox_inches='tight')
    if show:
        plt.show()
    if return_fig:
        return f

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


def plot_means_vs_vars(m_o_m, v_o_d, timestr_cut=None, var_frac_thresh=None, xlim=None, ylim=None, all_set_numbers_list=None, all_timestr_list=None,\
                      nfr=5, inst=1, fit_line=True, itersigma=4.0, niter=5):
    
    mask = [True for x in range(len(m_o_m))]
    
    from astropy.stats import sigma_clip
    from astropy.modeling import models, fitting
    
    if timestr_cut is not None:
        mask *= np.array([t==timestr_cut for t in all_timestr_list])
    
    photocurrent = m_o_m[mask]
    varcurrent = v_o_d[mask]
    
    if fit_line:
        # initialize a linear fitter
        fit = fitting.LinearLSQFitter()
        # initialize the outlier removal fitter
        or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=niter, sigma=itersigma)
        # initialize a linear model
        line_init = models.Linear1D()
        # fit the data with the fitter
        sigmask, fitted_line = or_fit(line_init, photocurrent, varcurrent)
        
        g1_iter = get_g1_from_slope_T_N(0.5*fitted_line.slope.value, N=nfr)

    
    if var_frac_thresh is not None:
        median_var = np.median(varcurrent)
        print('median variance is ', median_var)
        mask *= (np.abs(v_o_d-median_var) < var_frac_thresh*median_var) 
        
        print(np.sum(mask), len(v_o_d))
        
    min_x_val, max_x_val = np.min(photocurrent), np.max(photocurrent)
    
        
    if all_set_numbers_list is not None:
        colors = np.array(all_set_numbers_list)[mask]
    else:
        colors = np.arange(len(m_o_m))[mask]
        
    f = plt.figure(figsize=(12, 6))
    title = 'TM'+str(inst)
    if timestr_cut is not None:
        title += ', '+timestr_cut
        
    plt.title(title, fontsize=18)
        
    timestrs = ['03-13-2013', '04-25-2013', '05-13-2013']
    markers = ['o', 'x', '*']
    set_color_idxs = []
    for t, target_timestr in enumerate(timestrs):
        tstrmask = np.array([tstr==target_timestr for tstr in all_timestr_list])
        
        tstr_colors = None
        if all_set_numbers_list is not None:
            tstr_colors = np.array(all_set_numbers_list)[mask*tstrmask]
        
        if fit_line:
            if t==0:
                if inst==1:
                    xvals = np.linspace(-5, -18, 100)
                else:
                    xvals = np.linspace(np.min(photocurrent), np.max(photocurrent), 100)
                    print('xvals:', xvals)
                plt.plot(xvals, fitted_line(xvals), label='Iterative sigma clip, $G_1$='+str(np.round(g1_iter, 3)))
            plt.scatter(photocurrent[tstrmask], sigmask[tstrmask], s=100, marker=markers[t], c=tstr_colors, label=target_timestr)
            plt.xlim(xlim)

        else:
            plt.scatter(photocurrent[tstrmask], varcurrent[tstrmask], s=100, marker=markers[t], c=tstr_colors, label=target_timestr)
            plt.xlim(xlim)
            plt.ylim(ylim)            

    plt.legend(fontsize=16)
    plt.xlabel('mean [ADU/fr]', fontsize=18)
    plt.ylabel('$\\sigma^2$ [(ADU/fr)$^2$]', fontsize=18)
    plt.tick_params(labelsize=14)
    plt.show()
    
    return m, f


def plot_hankel_integrand(bins_rad, fine_bins, wtheta, spline_wtheta, ell, integration_max):
    
    f = plt.figure(figsize=(10, 8))
    plt.subplot(2,2,3)
    plt.title('Discrete Integrand', fontsize=16)
    plt.plot(bins_rad, bins_rad*wtheta*scipy.special.j0(ell*bins_rad), label='$\\ell = $'+str(ell))
    plt.ylabel('$\\theta w(\\theta)J_0(\\ell\\theta)$', fontsize=18)
    plt.xlabel('$\\theta$ [rad]', fontsize=18)
    plt.legend(fontsize=14)
    plt.xscale('log')
    plt.subplot(2,2,4)
    plt.title('Spline Integrand', fontsize=16)
    plt.plot(fine_bins, fine_bins*spline_wtheta(fine_bins)*scipy.special.j0(ell*fine_bins), label='$\\ell = $'+str(ell))
    plt.ylabel('$\\theta w(\\theta)J_0(\\ell\\theta)$', fontsize=18)
    plt.xlabel('$\\theta$ [rad]', fontsize=18)
    plt.axvline(integration_max, linestyle='dashed', color='k', label='$\\theta_{max}$='+str(np.round(integration_max*(180./np.pi), 2))+' deg.')
    plt.legend(fontsize=14)
    plt.xscale('log')
    plt.subplot(2,2,1)
    plt.title('Angular 2PCF', fontsize=16)
    plt.plot(bins_rad, wtheta, marker='.', color='k')
    plt.plot(np.linspace(np.min(bins_rad), np.max(bins_rad), 1000), spline_wtheta(np.linspace(np.min(bins_rad), np.max(bins_rad), 1000)), color='b', linestyle='dashed')
    plt.xscale('log')
    plt.yscale('symlog')
    plt.xlabel('$\\theta$ [rad]', fontsize=18)
    plt.ylabel('$w(\\theta)$', fontsize=18)
    plt.subplot(2,2,2)
    plt.title('Bessel function, $\\ell = $'+str(ell), fontsize=16)
    plt.plot(fine_bins, scipy.special.j0(ell*fine_bins), label='$\\ell = $'+str(ell))
    plt.ylabel('$J_0(\\ell\\theta)$', fontsize=18)
    plt.xlabel('$\\theta$ [rad]', fontsize=18)
    plt.xscale('log')
    plt.tight_layout()
    plt.show()
    
    return f

def create_multipanel_figure(images, names, colormap):
    num_images = len(images)
    num_cols = min(num_images, 3)  # Maximum 3 columns for better visualization
    
    num_rows = num_images // num_cols
    if num_images % num_cols != 0:
        num_rows += 1

    fig, axes = plt.subplots(3, 2, figsize=(8,10))
    axes = axes.flatten()  # Flatten axes into a 1D array for easier indexing
    
    if type(colormap)==str:
        colormap = [colormap for x in range(len(axes))]
    
    for i in range(num_images):
        ax = axes[i]
        image = images[i]
        name = names[i]
        
        # Calculate vmin and vmax based on the image data
        vmin = np.percentile(image, 5)  # Adjust percentile values as needed
        
        if i==1:
            vmax= np.percentile(image, 99)
        elif i==4:
            vmax = np.percentile(image, 84)
            vmin = np.percentile(image, 16)
        else:
            vmax = np.percentile(image, 95)  # Adjust percentile values as needed
        
        im = ax.imshow(image, cmap=colormap[i], vmin=vmin, vmax=vmax)
#         ax.set_title(name)
        if i==0 or i==2:
            ax.set_ylabel("x [pixels]")
        if i>1:
            ax.set_xlabel("y [pixels]")
#         ax.axis('off')
        
        ax.text(0.95, 0.95, name, transform=ax.transAxes,
                fontsize=12, ha='right', va='top', bbox=dict({'facecolor':'white', 'alpha':0.8, 'edgecolor':None}))
        
            
        # Add colorbar with units
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.04, fraction=0.046)
        
        if i%2==1:
            cbar.set_label("nW m$^{-2}$ sr$^{-1}$", fontsize=12)
    
    # Hide unused subplots
    for j in range(num_images, num_rows * num_cols):
        fig.delaxes(axes[j])
    
    plt.subplots_adjust(hspace=0, wspace=0)  # Remove blank space between subplots

    plt.tight_layout()
    plt.show()
    
    return fig


