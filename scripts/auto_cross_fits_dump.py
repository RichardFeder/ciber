import matplotlib
import matplotlib.pyplot as plt
from ciber.mocks.cib_mocks import *

import numpy as np
from scipy import interpolate
import os
import astropy

import astropy.wcs as wcs
# import fitsio
# from mask_source_classification import *
import config
from ciber.core.powerspec_pipeline import * 
from ciber.core.ps_pipeline_go import *
from ciber.cross_correlation.galaxy_cross import *
from ciber.plotting.plotting_fns import *

from ciber.plotting.gal_plotting_fns import *
from ciber.cross_correlation.cl_forecast import *

from ciber.cross_correlation.ebl_tom import *


# %matplotlib inline

from ciber.theory.cross_ps_parametric_model import *


''' GALAXY AUTO FITS 1/29/26 '''

print("\n\n" + "="*80)
print("Example 2: DESI-LS galaxy auto-spectra with two-stage fitting")
print("="*80)


from ciber.theory.cross_ps_parametric_model import run_gal_auto_fits_two_stage

fitstr = 'two_stage_fixed_1h'

headstr= 'hsc_ilt25.0'
# headstr = None

# lMax = 50000

for lMax in [20000]:

    fit_results_hsc = run_gal_auto_fits_two_stage(
        inst_list=[1, 2],
        cat='HSC',
        ifield_list=[8],
        zbinedges=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        lMax_fit=lMax,
        chi2_eval_max=lMax,
        fitstr=fitstr,
        figbasedir='figures/gal_auto_fits_two_stage/',
        save_figs=True,
        save_results=True,
        file_fpath='HSC_coarsez_gal_auto_fits_'+fitstr+'_lMax='+str(lMax)+'.npz',
        ihl_1h_params_path='ihl_templates/ihl_1h_param_fit_v0.npz',
        nwalkers=32,
        nsteps_stage1=2000,
        nsteps_stage2=4000,
        nburn_stage1=500,
        nburn_stage2=1000,
        headstr=headstr,
        fmask=0.7
    )
    
    fit_results_ls = run_gal_auto_fits_two_stage(
        inst_list=[1, 2],
        cat='LS',
        ifield_list=[4, 5, 6, 7, 8],
        zbinedges=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        lMax_fit=lMax,
        chi2_eval_max=lMax,
        fitstr=fitstr,
        figbasedir='figures/gal_auto_fits_two_stage/',
        save_figs=True,
        save_results=True,
        file_fpath='DESILS_coarsez_gal_auto_fits_'+fitstr+'_lMax='+str(lMax)+'.npz',
        ihl_1h_params_path='ihl_templates/ihl_1h_param_fit_v0.npz',
        nwalkers=32,
        nsteps_stage1=2000,
        nsteps_stage2=4000,
        nburn_stage1=500,
        nburn_stage2=1000,
        headstr=headstr,
        fmask=0.7,
        chi2_lim=[-6, 6]
    )


''' PLOT GALAXY AUTO PARAMETER FITS 1/29/26 '''

# Load galaxy auto fit results for different lMax values
all_res_auto = []

catname = 'DESILS'
fitstr = 'two_stage_fixed_1h'

for lMax in [20000, 30000, 50000, 70000, 90000]:
    fpath = 'data/gal_auto_fits/DESILS_coarsez_gal_auto_fits_'+fitstr+'_lMax='+str(lMax)+'.npz'
    print('fpath = ', fpath)
    all_res_auto.append(load_fit_results_npz(fpath))
#     print(all_res_auto[lidx].keys())
#     print(all_res_auto[lidx]['params'])

lams = [1.1, 1.8]
cmap_name = 'Greens'

for inst in [1, 2]:
    ylim_2h = [-0.005, 0.02]
    
    configs = [
        {'results': all_res_auto[0], 'inst': inst, 'label': catname+' Auto ($\\ell<20000$)', 'color':'C0'},
        {'results': all_res_auto[1], 'inst': inst, 'label': '($\\ell<30000$)', 'color':'C1'},
        {'results': all_res_auto[2], 'inst': inst, 'label': '($\\ell<50000$)', 'color':'C2'},
        {'results': all_res_auto[3], 'inst': inst, 'label': '($\\ell<70000$)', 'color':'C3'},
        {'results': all_res_auto[4], 'inst': inst, 'label': '($\\ell<90000$)', 'color':'C4'},
    ]
    
    # Plot amplitude comparison
    fig = plot_amplitude_comparison(configs, save_path=None, ylim_2h=ylim_2h, ylim_ihl=[-0.02, 1.5], 
                                   legend_ncol=2, figsize=(6, 6), bbox_to_anchor=[0.02, 1.45], 
                                   use_cmap=True, cmap_name=cmap_name)
    fig.savefig(f'figures/gal_auto_fit_{fitstr}_{catname}_CIBERTM{inst}_vs_lMax_fit.png', 
                bbox_inches='tight')
    
    # Plot chi2 comparison
    fig = plot_chi2_comparison(configs, save_path=None, figsize=(5.5, 4), legend_ncol=2, 
                              bbox_to_anchor=[0.0, 1.45], ylim_chi2=[0.0, 10], 
                              use_cmap=True, cmap_name=cmap_name)
    fig.savefig(f'figures/chi2_reduced_gal_auto_{fitstr}_{catname}_CIBERTM{inst}_vs_lMax_fit.png', 
                bbox_inches='tight')
    



''' CIBER x GALAXY CROSS FITS 1/29/26'''
lower_bounds = np.array([0., 0., 1.5, 0.])
upper_bounds = np.array([10., 100., 5.0, 10.])
prior_bounds = np.array([lower_bounds, upper_bounds])
ln_ell_peak_relation = (8.5, 7.4)  # intercept, slope


# fitstr = 'IHL1hfit_fixshape_ffbias_weighted_wdamp'
# fitstr = 'IHL1hfit_fixshape_ffbias_mcnoisefmaskcorr_wdamp'
# fitstr = 'IHL1hfit_fixshape_newcl_thetacut'
fitstr = 'no1h_thetacut'


# for lMax in [20000, 30000, 50000, 70000, 90000]:
# for lMax in [90000, 20000, 30000, 50000, 70000]:
# for lMax in [20000, 30000]:
for lMax in [20000, 30000, 50000, 70000, 90000]:
    
    
#     all_res_mcmc = run_gal_cross_fits(cat='DESILS',
#                                  save_figs=False,
#                                  save_results=True,
#                                 file_fpath='DESILS_coarsez_cross_cl_fits_lognormal1h_unifprior_fixlnellpeaklin_lMax='+str(lMax)+'.npz', 
#                                 use_ihl_templates=False, 
#                                  prior_bounds=prior_bounds, 
#                                  lMax_fit=lMax, 
#                                  ln_ell_peak_relation=ln_ell_peak_relation)


    all_res_mcmc = run_gal_cross_fits(cat='HSC',
                                      save_results=True, 
                                      file_fpath='HSC_coarsez_ilt25.0_cross_cl_fits_'+fitstr+'_lMax='+str(lMax)+'.npz',
                                      lMax_fit=lMax,
                                      use_ihl_templates=False,
                                      use_ihl_1h_params=True,
                                      fix_ihl_1h_shape=True,
                                      ifield_list=[8],
                                     ihl_1h_params_path='ihl_templates/ihl_1h_param_fit_v0.npz', 
                                     fitstr=fitstr, 
                                     save_figs=True,
                                     use_astrometry_damping=True, 
                                     chi2_lim=[-5, 5], 
                                     headstr='hsc_ilt25.0', 
                                     use_one_halo=False)
    
    
#     all_res_mcmc = run_gal_cross_fits(cat='DESILS',
#                                       save_results=True, 
#                                       file_fpath='DESILS_coarsez_cross_cl_fits_'+fitstr+'_lMax='+str(lMax)+'.npz',
#                                       lMax_fit=lMax,
#                                       use_ihl_templates=False,
#                                       use_ihl_1h_params=True,
#                                       fix_ihl_1h_shape=True,
#                                       ifield_list=[4, 5, 6, 7, 8],
#                                      ihl_1h_params_path='ihl_templates/ihl_1h_param_fit_v0.npz', 
#                                      fitstr=fitstr, 
#                                      save_figs=True,
#                                      use_astrometry_damping=True, 
#                                      chi2_lim=[-5, 5], 
#                                      use_one_halo=False)



''' CIBER x GALAXY CROSS PLOT PARAMETER FITS 1/29/26 '''

# all_res_ls = load_fit_results_npz('data/cross_cl_fits/DESILS_coarsez_cross_cl_fits_IHLtemp_lMax=50000.npz')
# all_res_ls_80k = load_fit_results_npz('data/cross_cl_fits/DESILS_coarsez_cross_cl_fits_IHLtemp_lMax=80000.npz')
# all_res_ls_100k = load_fit_results_npz('data/cross_cl_fits/DESILS_coarsez_cross_cl_fits_IHLtemp_lMax=100000.npz')


all_res = []

catname = 'HSC'
# fitstr = 'IHL1hfit_fixshape_ffbias_weighted_wdamp'
# fitstr = 'IHL1hfit_fixshape_ffbias_wdamp'

# fitstr = 'IHL1hfit_fixshape_wdamp'
# fitstr = 'IHL1hfit_fixshape_ffbias_mcnoisefmaskcorr_wdamp'
# fitstr = 'IHL1hfit_fixshape_ffbias_mcnoisefmaskcorrfilt_wdamp'

# fitstr = 'IHL1hfit_fixshape_newcl_thetacut'
fitstr = 'no1h_thetacut'


for lMax in [20000, 30000, 50000, 70000, 90000]:

#     all_res.append(load_fit_results_npz('data/cross_cl_fits/DESILS_coarsez_cross_cl_fits_IHLtemp_lMax='+str(lMax)+'_v2.npz'))
#     all_res.append(load_fit_results_npz('data/cross_cl_fits/'+catname+'_coarsez_cross_cl_fits_'+fitstr+'_lMax='+str(lMax)+'.npz'))
    all_res.append(load_fit_results_npz('data/cross_cl_fits/'+catname+'_coarsez_ilt25.0_cross_cl_fits_'+fitstr+'_lMax='+str(lMax)+'.npz'))

#     all_res.append(load_fit_results_npz('data/cross_cl_fits/DESILS_coarsez_cross_cl_fits_lognormal1h_unifprior_fixlnellpeaklin_lMax='+str(lMax)+'.npz'))

lams = [1.1, 1.8]

if catname =='HSC':
    cmap_name = 'Oranges'
else:
    cmap_name = 'Blues'

for inst in [1, 2]:
    
    if inst==2:
        ylim_2h = [-0.02, 0.5]
    else:
        ylim_2h = [-0.02, 0.5]
    configs = [
        {'results': all_res[0], 'inst': inst, 'label': catname+' x CIBER '+str(lams[inst-1])+' μm ($\\ell<20000$)', 'color':'C0'},
        {'results': all_res[1], 'inst': inst, 'label': '($\\ell<30000$)', 'color':'C1'},
        {'results': all_res[2], 'inst': inst, 'label': '($\\ell<50000$)', 'color':'C2'},

        {'results': all_res[3], 'inst': inst, 'label': '($\\ell<70000$)', 'color':'C3'},
        {'results': all_res[4], 'inst': inst, 'label': '($\\ell<90000$)', 'color':'C4'},
    ]
    
    fig = plot_amplitude_comparison(configs, save_path=None, ylim_2h=ylim_2h, ylim_ihl=[-0.02, 1.5], 
                                   legend_ncol=2, figsize=(6, 6), bbox_to_anchor=[0.02, 1.45], use_cmap=True, cmap_name=cmap_name)
    fig.savefig('figures/cl_fit_'+fitstr+'_'+catname+'_CIBERTM'+str(inst)+'_vs_lMax_fit_012329.png', bbox_inches='tight')
#     fig.savefig('figures/cl_fit_'+fitstr+'_'+catname+'_ilt25_CIBERTM'+str(inst)+'_vs_lMax_fit_012526.png', bbox_inches='tight')

    fig = plot_chi2_comparison(configs, save_path=None, figsize=(5.5, 4), legend_ncol=2, bbox_to_anchor=[0.0, 1.45], 
                              ylim_chi2=[0.0, 4], use_cmap=True, cmap_name=cmap_name)
    
    fig.savefig('figures/chi2_reduced_'+fitstr+'_'+catname+'_CIBERTM'+str(inst)+'_vs_lMax_fit_012329.png', bbox_inches='tight')
#     fig.savefig('figures/chi2_reduced_'+fitstr+'_'+catname+'_ilt25_CIBERTM'+str(inst)+'_vs_lMax_fit_012526.png', bbox_inches='tight')

    

''' PLOT CROSS-CORRELATION FIT COMPONENTS 1/29/26 '''

lMax = 50000
catname = 'DESILS'
zbinedges=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
inst_list=[1, 2]

lams = [1.1, 1.8]
fitstr_cross = 'IHL1hfit_fixshape_newcl_thetacut'

fpath = 'data/cross_cl_fits/'+catname+'_coarsez_cross_cl_fits_'+fitstr_cross+'_lMax='+str(lMax)+'.npz'
# fpath = 'data/cross_cl_fits/'+catname+'_coarsez_ilt25.0_cross_cl_fits_'+fitstr_cross+'_lMax='+str(lMax)+'.npz'

# _ilt25.0
# cross_results = load_fit_results_npz('data/cross_cl_fits/'+catname+'_coarsez_cross_cl_fits_'+fitstr_cross+'_lMax='+str(lMax)+'.npz')



# Load and plot in one shot
fig, axes = plot_cross_fit_components_from_file(
    fpath,
    zbinedges=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    inst_list=[1, 2],
    cat=catname,
    save_path='figures/'+catname+'_cross_components_'+fitstr_cross+'_012926.png',
    figsize=(7, 8)
)
