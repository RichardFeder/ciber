import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy.stats import norm
# from ciber2_noise_utils import *
# from cross_spectrum_analysis import *
from ciber_powerspec_pipeline import *
from noise_model import *
import pickle
import os
from plotting_fns import *
from scipy import stats
import re

class CIBER1_flight_frames():
    
    names = ['raildark', 'BootesA', 'elat10', 'DGL', 'NEP', 'elat30', 'BootesB', \
            'SWIRE', 'Lockman', 'vega1', 'vega2', 'vega3']
    
    timeup_dict_dt0 = dict({'raildark':-101., 'BootesA':636., 'elat10':436., \
                           'DGL':220., 'NEP':299., 'elat30':500., 'BootesB':569., \
                           'SWIRE':705., 'Lockman':370., 'vega1':727., 'vega2':746., 'vega3':775.})
    timedown_dict_dt0 = dict({'raildark':-167., 'BootesA':581., 'elat10':387., \
                             'DGL':148., 'NEP':233., 'elat30':450., 'BootesB':513., \
                             'SWIRE':655., 'Lockman':316., 'vega1':717., 'vega2':734., 'vega3':755.})
    
    
    datadir = '/Users/richardfeder/Documents/ciber_temp/110422/data/'
    # framedir = datadir+'CIBER1_dr_ytc/framedata/'

    # tof_basepath = datadir+'CIBER1_dr_ytc/time_ordered_frames_080123/'
    tof_basepath = config.ciber_basepath+'/data/time_ordered_frames/'
    framedir = config.ciber_basepath+'/data/framedata/'

    tframe = 1.78
    
    def __init__(self, t0=11100.75000, framedir=None, dimx=1024, dimy=1024):
        self.t0 = t0
        if framedir is not None:
            self.framedir = framedir
        self.cbps = CIBER_PS_pipeline(dimx=dimx, dimy=dimy)
        self.cbps_nm = CIBER_NoiseModel()
        
    def grab_frame_fpaths(self, inst, inplace=True):
        
        self.fpaths_all = np.array(glob.glob(self.framedir+'TM'+str(inst)+'*.mat'))
        
        self.all_tframe = np.zeros((len(self.fpaths_all)))
        for fidx, fpath in enumerate(self.fpaths_all):
            tframe = float(fpath.split('/')[-1].split('TM'+str(inst)+'_')[-1].split('.mat')[0])
            
            self.all_tframe[fidx] = tframe

        if not inplace:
            return self.all_tframe
    
    def separate_fpaths_by_field(self, inst, inplace=True):
        
        self.fpath_by_field_dict = dict({})
        self.tframe_by_field_dict = dict({})

        self.grab_frame_fpaths(inst)
        
        for fieldname in self.names:
            timedown, timeup = self.timedown_dict_dt0[fieldname]+self.t0, self.timeup_dict_dt0[fieldname]+self.t0 
            which_in_field = (self.all_tframe > timedown)*(self.all_tframe < timeup)
            
            fpaths_indiv_field = self.fpaths_all[which_in_field]
            tframe_indiv_field = self.all_tframe[which_in_field]
            
            self.fpath_by_field_dict[fieldname] = fpaths_indiv_field
            self.tframe_by_field_dict[fieldname] = tframe_indiv_field
            
        if not inplace:
            return self.fpath_by_field_dict, self.tframe_by_field_dict
            
    def package_time_ordered_frames(self, inst):
        
        for fieldname in self.names:
            
            order_frames = np.argsort(self.tframe_by_field_dict[fieldname])
            
            ordered_frames_indiv_field = np.zeros((len(order_frames), self.cbps.dimx, self.cbps.dimy))
            
            for frameidx, order in enumerate(order_frames):
                
                frame_fpath = self.fpath_by_field_dict[fieldname][order]
                
                frame_file = scipy.io.loadmat(frame_fpath)
        
                frame_array = frame_file['arraymap']
            
                ordered_frames_indiv_field[frameidx] = frame_array
            
            np.savez(self.tof_basepath+'TM'+str(inst)+'/CIBER1_40030_TM'+str(inst)+'_timeordered_frames_'+fieldname+'.npz', \
                    ordered_frames_indiv_field=ordered_frames_indiv_field,\
                     ordered_fpaths=self.fpath_by_field_dict[fieldname][order_frames], \
                    ordered_tframe=self.tframe_by_field_dict[fieldname][order_frames])
            
            
    def ravel_frames_to_timestreams(self, inst, fieldname, skipfr=2):
        
        # open .npz ordered frame file
        ordered_frame_file = np.load(self.tof_basepath+'TM'+str(inst)+'/CIBER1_40030_TM'+str(inst)+'_timeordered_frames_'+fieldname+'.npz')
        ordered_frames_indiv_field = ordered_frame_file['ordered_frames_indiv_field'][skipfr:]
        ordered_tframe = ordered_frame_file['ordered_tframe'][skipfr:]
        
        print('There are ', len(ordered_tframe), ' frames for '+fieldname)
        for frameidx, ordered_frame in enumerate(ordered_frames_indiv_field[:-1]):
            
            plot_map(ordered_frames_indiv_field[frameidx+1]-ordered_frame)
        
    def load_ordered_frame_file(self, inst, fieldname, skipfr=2, nframe=None):
        
        ordered_frame_file = np.load(self.tof_basepath+'TM'+str(inst)+'/CIBER1_40030_TM'+str(inst)+'_timeordered_frames_'+fieldname+'.npz')
        if nframe is not None:
                
            ordered_frames_indiv_field = ordered_frame_file['ordered_frames_indiv_field'][skipfr:skipfr+nframe+1]
            ordered_tframe = ordered_frame_file['ordered_tframe'][skipfr:skipfr+nframe+1]
        else:
            if fieldname=='elat30':
                # use last ten frames from integration
                ordered_frames_indiv_field = ordered_frame_file['ordered_frames_indiv_field'][-11:]
                ordered_tframe = ordered_frame_file['ordered_tframe'][-11:]
            else:
                ordered_frames_indiv_field = ordered_frame_file['ordered_frames_indiv_field'][skipfr:]
                ordered_tframe = ordered_frame_file['ordered_tframe'][skipfr:]
                
        return ordered_frames_indiv_field, ordered_tframe
        
    
    def slope_fits(self, inst, fieldname, ordered_frames_indiv_field=None, nframe=None, skipfr=2, nsplit=1):
               
        
        if ordered_frames_indiv_field is None:
            
            ordered_frames_indiv_field, ordered_tframe = self.load_ordered_frame_file(inst, fieldname, skipfr=skipfr, nframe=nframe)
        
        nMB = len(ordered_frames_indiv_field)*self.cbps.dimx*self.cbps.dimy*8//1024**2
        ncol_perfit = self.cbps.dimy//nsplit 
        best_fit_slope = np.zeros((self.cbps.dimx, self.cbps.dimy))
        
        print('nMB = ', nMB)
        nframe = len(ordered_frames_indiv_field)
        print('number of frame files = ', nframe)
        obs_rav = np.zeros((len(ordered_frames_indiv_field), self.cbps.dimx*self.cbps.dimy//nsplit))
        
        for n in range(nsplit):
            for i, frame_idx in enumerate(ordered_frames_indiv_field):
                obs_rav[i,:] = ordered_frames_indiv_field[i][:,n*ncol_perfit:(n+1)*ncol_perfit].ravel()
            x_matrix = np.array([[1 for x in range(nframe)], np.arange(nframe)]).transpose()
            inv_xprod = np.linalg.inv(np.dot(x_matrix.transpose(), x_matrix))
            second_prod = np.dot(inv_xprod, x_matrix.transpose())
            best_fit = np.dot(second_prod, obs_rav)

            best_fit_slope[:,n*ncol_perfit:(n+1)*ncol_perfit] = np.reshape(best_fit[1,:], (self.cbps.dimx, self.cbps.dimy//nsplit)) # returns in units of electrons

        return best_fit_slope
    
    
    def plot_slope_maps_brightpixels(self, inst, fieldname, skipfr=2, nframe_short=4, maskinst=None):
        ordered_frames_indiv_field, ordered_tframe = self.load_ordered_frame_file(inst, fieldname, skipfr=skipfr)
        
        slope_map, slope_map_corr, aducorr_mask, ordered_frames_aducorr = self.calculate_adu_fr_maps(inst, fieldname, ifield, ordered_frames_indiv_field=ordered_frames_indiv_field, linearize_map=False)

        init_adu = ordered_frames_indiv_field[0]
        final_adu = ordered_frames_indiv_field[-1]
        
        init_adu_rav = init_adu.ravel()
    
        plt.figure(figsize=(5,4))
        plt.title('TM'+str(inst)+', '+fieldname, fontsize=14)
        plt.hist(init_adu_rav[init_adu_rav != 0], bins=100)
        median_init_adu = np.median(init_adu_rav[init_adu_rav != 0])
        plt.axvline(median_init_adu, label='Median = '+str(median_init_adu), linestyle='dashed', color='r')
        plt.yscale('log')
        plt.legend()
        plt.ylabel('$N_{pix}$', fontsize=14)
        plt.xlabel('Initial ADU', fontsize=14)
#         plt.savefig('/Users/richardfeder/Downloads/init_adu_hist_TM'+str(inst)+'_'+fieldname+'.png', bbox_inches='tight')
        plt.show()
    
        plt.figure(figsize=(5,4))
        slope_map_corr_rav = slope_map_corr.ravel()*self.cbps.cal_facs[inst]
        plt.hist(slope_map_corr_rav, bins=np.logspace(1, 4, 20))
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        
        plt.figure()
        which_large_slope = (slope_map_corr_rav > 1000)
        plt.hist(init_adu_rav[which_large_slope], bins=np.arange(0, 65536, 100))
        plt.yscale('log')
        plt.show()
        
        
        fadu = plot_map(init_adu, title='Initial ADU values TM'+str(inst)+', '+fieldname, lopct=0, hipct=95, return_fig=True)
#         fadu.savefig('/Users/richardfeder/Downloads/init_adu_map_TM'+str(inst)+'_'+fieldname+'.png', dpi=300, bbox_inches='tight')
        
        
        e_mask = aducorr_mask.copy() 
        

        nrows, ncols = 3, 3
    
        fig, ax = plt.subplots(nrows=nrows, ncols=nrows, figsize=(8,6))
        ax = ax.flatten()
        
        
#         plt.figure(figsize=(15, 5))
        ctr = 0
        
        for nrow in np.arange(nrows):
            for ncol in np.arange(ncols):
                
                if ncol==0:
                    ax[3*nrow+ncol].set_ylabel('ADU values', fontsize=14)
                if nrow==2:
                    ax[3*nrow+ncol].set_xlabel('$i_{frame}$', fontsize=14)
                    
                ax[3*nrow+ncol].axhline(0, color='grey', linestyle='dashed', alpha=0.5)
#                 ax[3*nrow+ncol].set_ylim(-30000, 5000)
#                 ax[3*nrow+ncol].text(1, -28000, '$i_{flip}=$'+str(3*nrow+ncol), fontsize=14)
#                 ax[3*nrow+ncol].text(1, 8000, '$i_{flip}=$'+str(3*nrow+ncol), fontsize=14)

#                 ax[3*nrow+ncol].axhline(65536, color='grey', linestyle='dashed', alpha=0.5)

#         viridis = matplotlib.colormaps['jet']

        for x in range(e_mask.shape[0]):
            for y in range(e_mask.shape[1]):

                if e_mask[x,y]==1:
                    aduvals = ordered_frames_indiv_field[:,x,y]
            
                    aduvals_corr = ordered_frames_aducorr[:,x,y]
                    
                    which_frame_reset = np.where((aduvals[:-1] < 5000)*(aduvals[1:] > 50000))[0]
                    
                    if len(which_frame_reset) > 0:
                        
                        which_frame_reset = np.min(which_frame_reset)
                    
#                         plt.scatter(np.arange(len(ordered_frames_indiv_field)), aduvals, s=5, marker='.', c=viridis(which_frame_reset/ordered_frames_indiv_field.shape[0]), alpha=0.1)

                    
#                         print('which frame reset:', which_frame_reset)
#                         print(which_frame_reset//ncols, which_frame_reset%nrows)
                        if which_frame_reset < 9:

                            nrow, ncol = which_frame_reset//ncols, which_frame_reset%nrows

#                             ax[3*nrow+ncol].scatter(np.arange(len(ordered_frames_indiv_field)), aduvals_corr-aduvals_corr[0], s=2, c='C'+str(int(which_frame_reset)), alpha=0.5)

                            ax[3*nrow+ncol].scatter(np.arange(len(ordered_frames_indiv_field)), aduvals_corr-aduvals_corr[0], s=2, c='C'+str(int(which_frame_reset)), alpha=0.5)
#                         plt.plot(np.arange(len(ordered_frames_indiv_field)), aduvals, marker='.', color='k', alpha=0.1)

                
                

                    ctr +=1 
                    
                    if ctr > 1000:
                        break
#         plt.axhline(0, color='r')
#         plt.xlabel('$i_{frame}$')
#         plt.ylabel('Integrated charge [e-]')
#         plt.ylim(0, 1.25e5)
#         plt.colorbar()
        plt.tight_layout()
#         plt.savefig('/Users/richardfeder/Downloads/adu_vs_frame_uncorrected_TM'+str(inst)+'_'+fieldname+'_byiflip.png', bbox_inches='tight', dpi=300)
        plt.show()
                    
                    
#         for i in range(len(e_mask)):
#             if e_mask[i]:
#                 adu_vals = 
                
#                 plt.plot(np.)
        
        
    
    def calculate_adu_fr_maps(self, inst, fieldname, ifield, ordered_frames_indiv_field=None,\
                              ts_filt=False, skipfr=2, nframe=4, n_electron_max = 3.5e4, plot=True):
        ''' wrapper script for processing field '''
        
                
        if ordered_frames_indiv_field is None:
            ordered_frames_indiv_field, ordered_tframe = self.load_ordered_frame_file(inst, fieldname, skipfr=skipfr)
            
#         plot_map(ordered_frames_indiv_field[0], title='initial ADU')
        slope_map_fullexp = self.slope_fits(inst, fieldname, skipfr=skipfr)
    
        ordered_frames_aducorr, aducorr_mask = self.correct_aduflips(inst, fieldname, ifield, ordered_frames_indiv_field=ordered_frames_indiv_field, skipfr=skipfr)

        slope_map_corrected = self.slope_fits(inst, fieldname, skipfr=skipfr, ordered_frames_indiv_field=ordered_frames_aducorr)

        which_max_array, max_adu_jump_array = self.compute_max_adu_jumps(ordered_frames_aducorr, aducorr_mask=aducorr_mask)
        

        max_adu_jump_slope_ratio = max_adu_jump_array/slope_map_corrected
        
        bad_jump_mask = (np.abs(max_adu_jump_slope_ratio) < 5.)*(np.abs(max_adu_jump_array) < 1000)
        
        plot_map(bad_jump_mask.astype(int), title='bad jump mask')
        aducorr_mask *= bad_jump_mask
        
        max_adu_jump_rav = max_adu_jump_array.ravel()
        aducorr_mask_rav = aducorr_mask.ravel().astype(int)
        slope_map_corrected_rav = slope_map_corrected.ravel()
        
        plt.figure()
        plt.scatter(slope_map_corrected_rav[aducorr_mask_rav==1], max_adu_jump_rav[aducorr_mask_rav==1]/slope_map_corrected_rav[aducorr_mask_rav==1])
        plt.ylabel('Maximum ADU jump between samples / slope fit')
        plt.xlabel('slope fit ADU/fr')
        plt.show()
        
        slope_map_corrected[~bad_jump_mask] = slope_map_fullexp[~bad_jump_mask]
                   
        plot_map(slope_map_corrected, title='slope map corrected')
        
        return slope_map_fullexp, slope_map_corrected, aducorr_mask, ordered_frames_aducorr

    def correct_aduflips(self, inst, fieldname, ifield, ordered_frames_indiv_field=None, skipfr=2):
        if ordered_frames_indiv_field is None:
            ordered_frames_indiv_field, ordered_tframe = self.load_ordered_frame_file(inst, fieldname, skipfr=skipfr)
           
        ordered_frames_aducorr = ordered_frames_indiv_field.copy()
        aducorr_mask = np.zeros_like(ordered_frames_aducorr[0])
        
        for nx in range(ordered_frames_indiv_field.shape[1]):
            for ny in range(ordered_frames_indiv_field.shape[2]):
                    
                which_frame_reset = np.where((ordered_frames_indiv_field[:-1, nx, ny] < 5000)*(ordered_frames_indiv_field[1:,nx,ny] > 50000))[0]
                if len(which_frame_reset)==1:
                    aducorr_mask[nx, ny] = 1.
                    ordered_frames_aducorr[which_frame_reset[0]+1:, nx, ny] -= 65536
                    
        return ordered_frames_aducorr, aducorr_mask


    def compute_max_adu_jumps(self, observed_frames_aducorr, aducorr_mask=None):

        frame_diffs = np.array([observed_frames_aducorr[i+1]-observed_frames_aducorr[i] for i in range(len(observed_frames_aducorr)-1)])
        
        abs_frame_diffs = np.abs(frame_diffs)
        
        which_max_array = np.zeros_like(frame_diffs[0])
        max_adu_jump_array = np.zeros_like(frame_diffs[0])
        
        if aducorr_mask is None:
            aducorr_mask = np.ones_like(which_max_array)
        
        for nx in range(frame_diffs[0].shape[0]):
            for ny in range(frame_diffs[0].shape[1]):
                
                if aducorr_mask[nx,ny]:

                    which_max = np.where((abs_frame_diffs[:,nx,ny]==np.max(abs_frame_diffs[:,nx,ny])))[0][0]
                    which_max_array[nx, ny] = which_max
                    max_adu_jump_array[nx, ny] = frame_diffs[which_max, nx, ny]


        plot_map(which_max_array, title='which_max_array')
        plot_map(max_adu_jump_array, title='max_adu_jump_array')
        
        return which_max_array, max_adu_jump_array
        

        
    def process_all_slopefits(self, inst, ifield_list, add_str=None, plot=False, ts_filt=False, nframe=None, \
                             rectify_maskinst=False, include_dark_mask=True, ifield_dark=8, save=False):
        all_save_fpaths = []
        all_adu_corr_masks = []
        all_aducorr_masks = []
        
        which_large_photocurrent_tot = np.zeros((1024, 1024))
        if include_dark_mask:
            labexp_fpaths = self.cbps_nm.grab_all_exposures(inst, ifield_dark, key='full')
            for fpath in labexp_fpaths:
                exp_eps = fits.open(fpath)[0].data*self.cbps_nm.cbps.g1_facs[inst]
                which_large_photocurrent = (np.abs(exp_eps) > 10)
                which_large_photocurrent_tot *= which_large_photocurrent.astype(int)
            
            # which_large_photocurrent_tot[which_large_photocurrent_tot > 1] = 1

            for fieldidx, ifield in enumerate(ifield_list):

                ciber_dat_path = config.ciber_basepath+'/data/'
                # save_fpath = config.exthdpath +'ciber_fluctuation_data/'
                maskInst_fpath = ciber_dat_path+'fluctuation_data/TM'+str(inst)+'/masks/maskInst_102422/field'+str(ifield)+'_TM'+str(inst)+'_maskInst_102422.fits'
                # maskInst_fpath = save_fpath+'TM'+str(inst)+'/masks/maskInst_102422/field'+str(ifield)+'_TM'+str(inst)+'_maskInst_102422.fits'
                maskinst_dr = fits.open(maskInst_fpath)[1].data

                maskinst_frac = float(np.sum(maskinst_dr))/float(1024**2)

                maskinst_dr_photmask = maskinst_dr*(which_large_photocurrent_tot==0)

                maskinst_photmask_frac = float(np.sum(maskinst_dr_photmask))/float(1024**2)

                print('for ifield ', ifield, ', maskinst_frac, maskinst_photmask_frac = ', maskinst_frac, maskinst_photmask_frac)


        
        for fieldidx, ifield in enumerate(ifield_list):
            
            fieldname = self.cbps.ciber_field_dict[ifield]
            slope_map, slope_map_corr, aducorr_mask, _ = self.calculate_adu_fr_maps(inst, fieldname, ifield, nframe=nframe)
        
            plot_map(aducorr_mask*slope_map_corr)
            
            all_aducorr_masks.append(aducorr_mask)
            if fieldidx==0:
                prod_aducorr_mask = aducorr_mask.astype(int)
            else:
                prod_aducorr_mask *= aducorr_mask.astype(int)
#             all_adu_corr_masks.append(aducorr_mask)
            
            plt.figure()
            slope_map_corr_adumask = slope_map_corr.ravel()[aducorr_mask.ravel()==1]
            
            plt.hist(self.cbps.cal_facs[inst]*slope_map_corr_adumask)
            plt.show()
            
            if plot:
                plot_map(self.cbps.cal_facs[inst]*slope_map_corr, title='SB map TM'+str(inst)+'_ifield'+str(ifield))
            # save_fpath = self.datadir+'CIBER1_dr_ytc/slope_fits/TM'+str(inst)+'/ciber_flight_TM'+str(inst)+'_ifield'+str(ifield)

            save_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/slope_fits/ciber_flight_TM'+str(inst)+'_ifield'+str(ifield)

            if add_str is not None:
                save_fpath += '_'+add_str
            
            if save:
                all_save_fpaths.append(save_fpath)
                self.save_slopefits(save_fpath+'.fits', slope_map_corr, inst, ifield, ts_filt=ts_filt, nframe=nframe)

        # identify pixels with high photocurrent in all exposures
        
        print('sum prod aducorr mask :', np.sum(prod_aducorr_mask))
        plot_map(prod_aducorr_mask, cmap='Greys_r')
        plot_map(which_large_photocurrent_tot, title='which large photocurrent tot')
        # if not include_dark_mask:
        #     prod_aducorr_mask *= which_large_photocurrent_tot
        #     plot_map(prod_aducorr_mask, cmap='Greys_r')
        
                
        for fieldidx, ifield in enumerate(ifield_list):
            if rectify_maskinst:
                # maskinst_aducorr = self.rectify_instrument_mask(inst, ifield, all_aducorr_masks[fieldidx]*(prod_aducorr_mask==0)*(which_large_photocurrent_tot==0), save=save)
            
                maskinst_aducorr = self.rectify_instrument_mask(inst, ifield, all_aducorr_masks[fieldidx]*(which_large_photocurrent_tot==0), save=save)
            
                
        return all_save_fpaths
        
    def save_slopefits(self, save_fpath, slope_fit, inst, ifield, ts_filt=False, nframe=None):
        
        image_hdu = fits.ImageHDU(slope_fit, name='slope_fit_TM'+str(inst)+'_ifield'+str(ifield))
        
        prim = fits.PrimaryHDU()
        prim.header['ts_filt'] = ts_filt
        prim.header['nframe'] = nframe
        prim.header['inst'] = inst
        prim.header['ifield'] = ifield
        
        hdul = fits.HDUList([prim, image_hdu])
        
        print('Saving to ', save_fpath)
        hdul.writeto(save_fpath, overwrite=True)
        
    def rectify_instrument_mask(self, inst, ifield, aducorr_mask, maskinst_dr=None, mask_save_dir=None, save=False):
        
        if maskinst_dr is None:
            save_fpath = config.ciber_basepath+'/data/fluctuation_data/'

            # save_fpath = config.exthdpath +'ciber_fluctuation_data/'
            maskInst_fpath = save_fpath+'TM'+str(inst)+'/masks/maskInst_102422/field'+str(ifield)+'_TM'+str(inst)+'_maskInst_102422.fits'
            maskinst_dr = fits.open(maskInst_fpath)[1].data
            print(np.sum(maskinst_dr))
            plot_map(maskinst_dr, title='maskinst_dr')
        
        maskinst_aducorr = maskinst_dr.copy()
        
        maskinst_aducorr[aducorr_mask==1] = 1.
        print(np.sum(maskinst_aducorr))
        plot_map(maskinst_aducorr, title='maskinst_aducorr')
        
                
        if save:
            
            if mask_save_dir is None:
                mask_save_dir = save_fpath+'TM'+str(inst)+'/masks/maskInst_aducorr/'
                
            mask_save_fpath = mask_save_dir+'field'+str(ifield)+'_TM'+str(inst)+'_maskInst_aducorr.fits'
            
            
            hdup = fits.PrimaryHDU()
            hdup.header['inst'] = inst
            hdup.header['ifield'] = ifield
            hdup.header['aducorr'] = True
            
            hduim = fits.ImageHDU(maskinst_aducorr, name='maskinst')
            hdul = fits.HDUList([hdup, hduim])
            print('saving to ', mask_save_fpath)
            hdul.writeto(mask_save_fpath, overwrite=True)
            
            
        return maskinst_aducorr