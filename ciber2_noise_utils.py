import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy.stats import norm
import os
import sys
import re

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def read_in_run_info(path):
    if path.lower().endswith('.xlsx'):
        run_df = pd.read_excel(path)
    return run_df

def img_from_fits(fits_file, dat_key=None):
    if dat_key is None:
        dat_key = 0
    return fits_file[dat_key].data

def make_frame_diff_str(idx1, idx2):
    return 'Frame '+str(idx1)+' - Frame '+str(idx2)

def make_exp_diff_str(idx1, idx2):
    return 'Exposure '+str(idx1)+' - Exposure '+str(idx2)

def plot_median_std_signals(frame_idxs, median_values, sigma_values, mode='frame', frame_or_exp='exposure', show=True, channel=1):
    
    f = plt.figure(figsize=(8, 4))
    if mode=='difference':
        plt.suptitle(frame_or_exp+' differences (Channel '+str(channel)+')', y=1.01, fontsize=16)
        inpt = frame_idxs[1:]
    else:
        inpt = frame_idxs
    plt.subplot(1,2,1)
    plt.plot(inpt, median_values, marker='.')
    plt.xlabel(frame_or_exp+' number')
    if mode=='difference':
        plt.ylabel('Median difference signal [e-/s/pixel]')
    elif mode=='frame':
        plt.ylabel('Median '+frame_or_exp+' signal [e-/s/pixel]')

    plt.xticks(inpt)
    plt.subplot(1,2,2)
    plt.plot(inpt, sigma_values, marker='.')
    plt.xlabel(frame_or_exp+' number')
    
    if mode=='difference':
        plt.ylabel('Std. deviation of difference signal [e-/s/pixel]')
    elif mode=='frame':
        plt.ylabel('Std. deviation of exposure signal [e-/s/pixel]')
    
    plt.tight_layout()
    plt.xticks(inpt)
    if show:
        plt.show()
        
    return f

def sigma_clip_maskonly(vals, previous_mask=None, sig=5):
    
    valcopy = vals.copy()
    if previous_mask is not None:
        valcopy[previous_mask==0] = np.nan
        sigma_val = np.nanstd(valcopy)
    else:
        sigma_val = np.nanstd(valcopy)
    
    abs_dev = np.abs(vals-np.nanmedian(valcopy))
    mask = (abs_dev < sig*sigma_val).astype(np.int)

    return mask

def iter_sigma_clip_mask(image, sig=5, nitermax=10, initial_mask=None):
    # image assumed to be 2d
    iteridx = 0
    
    summask = image.shape[0]*image.shape[1]
    running_mask = (image != 0).astype(np.int)
    if initial_mask is not None:
        running_mask *= initial_mask
        
    while iteridx < nitermax:
        
        new_mask = sigma_clip_maskonly(image, previous_mask=running_mask, sig=sig)
        
        if np.sum(running_mask*new_mask) < summask:
            running_mask *= new_mask
            summask = np.sum(running_mask)
        else:
            return running_mask

        iteridx += 1
        
    return running_mask

def view_img(img, vmin=None, vmax=None, minpercentile=5, maxpercentile=95, xlabel='x [pixels]', ylabel='y [pixels]', title=None, xticks=None, yticks=None, return_fig=False, \
            xlims=None, ylims=None, labelfontsize=16, titlefontsize=16, extent=None, cmap='Greys', figdim=10):
    if vmin is None and vmax is None:
        vmin=np.percentile(img, minpercentile)
        vmax=np.percentile(img, maxpercentile)
    
    print('vmin, vmax = ', vmin, vmax)
    f = plt.figure(figsize=(figdim, figdim))
    if title is not None:
        plt.title(title, fontsize=titlefontsize)
    if extent:
        plt.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap, extent=extent)
    else:
        plt.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14)
    if xticks is not None:
        plt.xticks(xticks)
        plt.yticks(xticks)
    plt.tick_params(labelsize=14)

    plt.xlabel(xlabel, fontsize=labelfontsize)
    plt.ylabel(ylabel, fontsize=labelfontsize)
    if xlims is not None:
        plt.xlim(xlims[0], xlims[1])
    if ylims is not None:
        plt.ylim(ylims[0], ylims[1])
        
    plt.show()
    
    if return_fig:
        return f
    
def view_hist(img, nbins=60, mask=None, filter_hot_pix=False, maxp=99.9, histmin=None, histmax=None, bins=None, \
          minpercentile=None, maxpercentile=None, yscale=None, title=None, return_fig=False, plot_median=True, titlefontsize=16, \
              label_unit='e-/s/pixel', xlabel=None):

    img_rav = img.ravel()

        
    if mask is not None:
        imv_rav = img_rav[mask.ravel()]

    else:
        if filter_hot_pix:
            mask= img_rav < np.percentile(img_rav, maxp)
            img_rav = img_rav[mask]
        
    if bins is None:
        if histmin is not None:
            bins = np.linspace(histmin, histmax, nbins)
        if minpercentile is not None:
            bins = np.linspace(np.percentile(img_rav, minpercentile), np.percentile(img_rav, maxpercentile), nbins)

    f = plt.figure(figsize=(6,5))
    if title is not None:
        plt.title(title, fontsize=titlefontsize)
    vals = plt.hist(img_rav, bins=bins, histtype='step')
    
    if plot_median:
        median = np.median(img_rav)
        onesig = 0.5*(np.percentile(img_rav, 84)-np.percentile(img_rav, 16))
        plt.axvline(median, linestyle='dashed', label='Median = '+str(np.round(median, 4))+' $\\pm$'+str(np.round(onesig, 4))+' '+label_unit)

    if yscale is not None:
        plt.yscale(yscale)

    plt.ylabel('$N_{pix}$', fontsize=16)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=16)
    plt.legend(fontsize=14)
    plt.tick_params(labelsize=14)
    plt.show()
    
    if return_fig:
        return f

    
def compute_cross_spectrum(map_a, map_b, n_deg_across=2.0, sterad_a=False, sterad_b=False):
    
    if sterad_a or sterad_b:
        sterad_per_pix = compute_sterad_per_pix(n_deg_across, map_a.shape[0])
    
    if sterad_a:
        # from electron per second per pixel to electron per second per steradian
        ffta = np.fft.fft2(map_a/sterad_per_pix)
    else:
        ffta = np.fft.fft2(map_a)
    if sterad_b:
        # from electron per second per pixel to electron per second per steradian
        fftb = np.fft.fft2(map_b/sterad_per_pix)
    else:
        fftb = np.fft.fft2(map_b)

    xspectrum = np.abs(ffta*np.conj(fftb)+fftb*np.conj(ffta))
    
    return np.fft.fftshift(xspectrum)
  
def compute_sterad_per_pix(n_deg_across, nside):
    sterad_per_pix = (n_deg_across*(np.pi/180.)/nside)**2
    return sterad_per_pix


class ciber2_noisedata():
    
    V_to_e = 4e-6 # this converts individual frames from Volts to electrons
    
    def __init__(self, timestr_head='RUN02102020', base_path='../data/20200210_noise_data', indiv_exposure_path=None, \
                 shutter_key='Shutter', exp_time_key='Exp Time [s]', idx_no_key='Image', power_key='Power', \
                notes_key='Notes', channel=1, figdim=8, dimx=2048, dimy=2048):
        
        for attr, valu in locals().items():
            setattr(self, attr, valu)

        if indiv_exposure_path is None:
            self.indiv_exposure_path = base_path+'/indiv_exposures/'

    def img_difference(self, exp_number_idx1=None, exp_number_idx2=None, img1=None, img2=None, idx1=None, idx2=None, frame_or_exp='exposure', exp_number=None, dat_key=None, verbose=False):
        
        if img1 is None:
            
            if frame_or_exp=='exposure':

                if exp_number_idx1 is None:
                    exp_number_idx1 = self.exp_numbers[idx1]
                if exp_number_idx2 is None:
                    exp_number_idx2 = self.exp_numbers[idx2]

                img1 = self.load_fits_from_timestr(exp_number_idx1, fits_or_img='img', frame_or_exp=frame_or_exp, verbose=verbose, dat_key=dat_key)
                img2 = self.load_fits_from_timestr(exp_number_idx2, fits_or_img='img', frame_or_exp=frame_or_exp, verbose=verbose, dat_key=dat_key)

            elif frame_or_exp=='frame':

                img1 = self.load_fits_from_timestr(exp_number, fits_or_img='img', frame_or_exp=frame_or_exp, frame_idx=idx1, verbose=verbose, dat_key=dat_key)
                img2 = self.load_fits_from_timestr(exp_number, fits_or_img='img', frame_or_exp=frame_or_exp, frame_idx=idx2, verbose=verbose, dat_key=dat_key)

        return img1 - img2

    def load_fits_from_timestr(self, timestr, fits_or_img='fits', dat_key=None, frame_or_exp='exposure', frame_idx=None, verbose=False):
        
        if len(str(timestr)) < 6:
            timestr = '0'+str(timestr)
        
        if verbose:
            print('timestr = ', timestr)
        if frame_or_exp=='exposure':
            path = self.base_path+'/ch'+str(self.channel)+'/'+self.timestr_head+'_'+str(timestr)+'.fits'
            print('path is ', path)
        
        elif frame_or_exp=='frame':
            path = self.indiv_exposure_path+'/ch'+str(self.channel)+'/'+self.timestr_head+'_'+str(timestr)+'/FRM'+str(frame_idx)+'_PIX.fts'


        if verbose:
            print('path:', path)
        f = fits.open(path)

        if fits_or_img=='fits':
            return f
        else:
            return img_from_fits(f,dat_key=dat_key)
        
        
    def parse_run_df(self, run_df, verbose=False):
        
        n_entries = len(run_df)
        
        self.shutter = np.array([1 if run_df[self.shutter_key][i]=='Open' else 0 for i in range(n_entries)])
        if verbose:
            print('Shutter bools (1 = Open, 0 = Closed):', self.shutter)
        if self.exp_time_key is not None:
            self.exp_times = np.array([run_df[self.exp_time_key][i] for i in range(n_entries)])
        if verbose:
            print('exp times (s):', self.exp_times)
        self.exp_numbers = np.array([int(np.nan_to_num(run_df[self.idx_no_key][i])) for i in range(n_entries)])
        if verbose:
            print('exp numbers:', self.exp_numbers)
        if self.power_key is not None:
            self.power_src = np.array([run_df[self.power_key][i] for i in range(n_entries)])
        if self.notes_key is not None:
            self.notes = np.array([run_df[self.notes_key][i] for i in range(n_entries)])

    def get_available_idxs(self, missing_idx_dict=None, frame_or_exp='exposure', exp_number=None):

        if missing_idx_dict is None:
            missing_idx_dict = dict({1:np.array([]),\
                                   2:np.array([]), 
                                  3:np.array([])})

        missing_idxs = missing_idx_dict[self.channel]
        if frame_or_exp == 'exposure':
            exp_idxs = np.setdiff1d(np.arange(exp_min_idx, exp_max_idx), missing_idxs)
            print('exp_idxs:', exp_idxs)

            return exp_idxs

        elif frame_or_exp == 'frame':

            frame_idxs = []

            exp_number_str = str(exp_number)
            if exp_number < 1e5:
                exp_number_str = exp_number_str.zfill(6)

            for file in os.listdir(self.indiv_exposure_path+'ch'+str(self.channel)+'/'+self.timestr_head+'_'+exp_number_str):
                if file.endswith('PIX.fts'):
                    frame_digits = [int(i) for i in str(file).split()[0] if i.isdigit()]
                    frame_idx = map(str, frame_digits)
                    frame_idx = int(''.join(frame_idx))
                    frame_idxs.append(frame_idx)

            frame_idxs = np.setdiff1d(np.array(frame_idxs), missing_idxs)
            print('frame_idxs:', frame_idxs)

            return frame_idxs
        
    def compute_cds_noise(self, diffs, mask=None, plot=True, filter_hot_pix=True, maxp=99.9, bins=100, imlopct=5, imhipct=95, yscale=None, vmin=None, vmax=None, figdim=None):
        if figdim is None:
            figdim = self.figdim
        cds = np.std(diffs, axis=0)/self.V_to_e # this is a 2048x2048 image converted to electrons
        
        if plot:
            title='CDS Noise [e-] (Channel '+str(self.channel)+')'
            cds_fig = view_img(cds, title=title, titlefontsize=20, return_fig=True, vmin=vmin, vmax=vmax, figdim=figdim)
            cds_hist_fig = view_hist2(cds, title=title, filter_hot_pix=filter_hot_pix, xlabel='$\\langle (frame_i - frame_j)^2 \\rangle^{1/2}$', mask=mask, maxp=maxp, yscale='symlog', bins=bins, return_fig=True, label_unit='e-')
        
        return cds, cds_fig, cds_hist_fig

    def compute_diff_images(self, missing_exposures_dict=None, exp_min_idx=0, exp_max_idx=30, \
                       show=True, histmin=-100, histmax=100, frame_or_exp='exposure', exp_number=133255, figdim=None, dat_key=None):
        median_values, width_values, ims, load_idxs, diffs, diff_figs, hist_figs = [[] for x in range(7)]
        
        if figdim is None:
            figdim = self.figdim
            
        exp_idxs = self.get_available_idxs(missing_idx_dict=missing_exposures_dict, frame_or_exp=frame_or_exp, exp_number=exp_number)

        print('exp_idxs:', exp_idxs)
        
        for i, exp_no in enumerate(exp_idxs):
        
            if exp_no==exp_idxs[0]:
                print('exp_no = ', exp_no, 'pass')
                continue

            try:
                im = self.img_difference(idx1=exp_no, idx2=exp_idxs[i-1], frame_or_exp=frame_or_exp, exp_number=exp_number, dat_key=dat_key)
                title = make_exp_diff_str(idx1=exp_no, idx2=exp_idxs[i-1])+' [e-/s] (Channel '+str(self.channel)+')'
                im_path = 'diff_image/exp'+str(exp_no)+'_'+str(exp_idxs[i-1])
                hist_path = 'diff_hist/exp'+str(exp_no)+'_'+str(exp_idxs[i-1])
                diffs.append(im)
                load_idxs.append(exp_no)

                median_values.append(np.median(im))
                width_values.append(np.std(im))
            except:
                print('no luck here on', exp_idxs[i], exp_idxs[i-1], ', moving on')
                continue
                        
            if show:
                if frame_or_exp=='frame':
                    diff_fig = view_img(im/self.V_to_e, title=title, titlefontsize=20, return_fig=True, figdim=figdim)
                    hist_fig = view_hist2(im/self.V_to_e, histmin=histmin, histmax=histmax, nbins=150, return_fig=True, title=title, titlefontsize=16, yscale='symlog')
                else:
                    diff_fig = view_img(im/self.V_to_e, title=title, titlefontsize=20, return_fig=True, figdim=figdim)
                    hist_fig = view_hist(im/self.V_to_e, histmin=histmin, histmax=histmax, nbins=150, return_fig=True, title=title, titlefontsize=16, yscale='symlog')
     
                diff_figs.append(diff_fig)
                hist_figs.append(hist_fig)
            
            
        return median_values, width_values, ims, load_idxs, diffs, diff_figs, hist_figs
    


class detector_readout():
    
    
    def __init__(self, exp_number=None, dimx=2048, dimy=2048, timestr_head='RUN02102020', base_path='../data/20200210_noise_data', \
                channel=1, power_key=None, exp_time_key=None, notes_key=None, figdim=8, ncol_per_channel=64, tframe = 1.35168):
        
        for attr, valu in locals().items():
            setattr(self, attr, valu)
        
        self.nd_obj = ciber2_noisedata(timestr_head=timestr_head, base_path=base_path, channel=channel, dimx=dimx, dimy=dimy,\
                                       exp_time_key=self.exp_time_key, power_key=self.power_key, notes_key=self.notes_key, figdim=self.figdim)
        
    def grab_timestreams(self):
        pass
        
    def grab_ref_pixels(self, image, inplace=True, refwide=4):
        
        ref_pixels_top = image[-refwide:,:]
        ref_pixels_bottom = image[:refwide,:]
        ref_pixels_left = image[:,:refwide]
        ref_pixels_right = image[:,-refwide:]
                
        if inplace:
            self.ref_pixels_top = ref_pixels_top
            self.ref_pixels_bottom = ref_pixels_bottom
            self.ref_pixels_left = ref_pixels_left
            self.ref_pixels_right = ref_pixels_right
            
        else:
            return ref_pixels_top, ref_pixels_bottom, ref_pixels_left, ref_pixels_right
        
        
    def subtract_ref_pixels(self, image, refwide=4, per_column=True, refwide_per_indiv=1, refwide_list=None, side_list=['bottom']):
        
        n_col_split = self.dimx//self.ncol_per_channel # typically periodic readout channel behavior
        
        self.grab_ref_pixels(image, refwide=refwide, inplace=True)
        
        ref_corrected_arr = image.copy()
        ref_vals_all_cols = []

        for j in range(n_col_split):
        
            if refwide_list is None:
                refwide_list = [refwide_per_indiv for x in range(self.ncol_per_channel//refwide_per_indiv)]

            ref_vals = []
            running_idx = 0
        
            for k in range(len(refwide_list)):
                combine_ref_vals = []
                
                for side in side_list:
                    if side=='bottom':
                        combine_ref_vals.extend(self.ref_pixels_bottom[:,j*self.ncol_per_channel+running_idx:j*self.ncol_per_channel+running_idx+refwide_list[k]].ravel())
                    elif side=='top':
                        combine_ref_vals.extend(self.ref_pixels_top[:,j*self.ncol_per_channel+running_idx:j*self.ncol_per_channel+running_idx+refwide_list[k]].ravel())
                    
                    # not sure how left/right pixels will be used, fill in later TODO
#                     elif side=='left':
#                         combine_ref_vals.extend(ref_pixels_left[])
#                     elif side=='right':
#                         combine_ref_vals.extend(ref_pixels_left[])
                        
                ref_val = np.median(combine_ref_vals)
                
                ref_corrected_arr[:,j*self.ncol_per_channel+running_idx:j*self.ncol_per_channel+running_idx+refwide_list[k]] -= ref_val
                running_idx += refwide_list[k]
                ref_vals.append(ref_val)


            ref_vals_all_cols.append(ref_vals)

        return ref_corrected_arr, ref_vals_all_cols
                
    def line_fit(self, exp_idxs, nskip=1, postrfr=True, frame0idx=0, parallel=False, plot=False, nparallel=None, verbose=False, memmax=2048):
        
        ''' nskip with postrfr False will skip first nskip PIX frames, even if that includes ones before a reset.
            with postrfr True, will find reset frames and choose frame index nskip after those. '''
        
        if postrfr:
            
            exp_str = str(self.exp_number)
            if len(exp_str) < 6:
                exp_str = '0'+str(exp_str)
                
            path = self.nd_obj.base_path + '/ch'+str(self.channel)+'/'+self.nd_obj.timestr_head+'_'+exp_str

            print('path is ', path)
            filenames = os.listdir(path)
            print('filenames are ', filenames)
            
            rfr_fnames = [fname for fname in filenames if 'RFR' in fname]
            print('rfr fnames are ', rfr_fnames)
            
            if len(rfr_fnames) > 0:
                rfr_digs = [int(re.sub("[^0-9]", "", rfr_fname)) for rfr_fname in rfr_fnames]
    
                max_rfr_dig = np.max(rfr_digs)
            
                exp_idxs = [e for e in exp_idxs if e > max_rfr_dig]
            
                print('exp idxs is now ', exp_idxs)
        
        
        ''' I think this will assume the detector array is divisible by a power of 2, which does hold for CIBER/CIBER2.'''
        nMB = (len(exp_idxs)-nskip-1)*self.dimx*self.dimy*8//1024**2
    
        nsplit = 1
        if nMB > memmax:
            print('Splitting up into sub-chunks, memory required for arrays in full (' +str(int(nMB))+' MB) is larger than memmax='+str(memmax)+' MB')
            
            while nMB/nsplit > memmax:
                nsplit *= 2
                
            print('Splitting into ', nsplit, ' chunks..')
        
        volt_rav = np.zeros((len(exp_idxs)-nskip-1, self.dimx*self.dimy//nsplit))
        ncol_perfit = self.dimy//nsplit 
        best_fit_slope = np.zeros((self.dimx, self.dimy))
        
        
        for n in range(nsplit):
            
            nframe=len(volt_rav)

            if verbose:
                print('volt_rav has shape ', volt_rav.shape, ' nframe = ', nframe)

            for i, frame_idx in enumerate(exp_idxs[nskip:-1]):
                
                
                with HiddenPrints():
                    img1 = self.nd_obj.load_fits_from_timestr(self.exp_number, fits_or_img='img', frame_or_exp='frame', frame_idx=frame_idx)    
                        
                    
                    if i > 0:
                        
                        if plot:
                            plt.figure(figsize=(8,8))
                            plt.subplot(1,2,1)
                            plt.title('frame '+str(i)+' - frame '+str(i-1), fontsize=18)
                            plt.imshow(img1-img0, cmap='Greys', vmin=np.nanpercentile(img1-img0, 16), vmax=np.nanpercentile(img1-img0, 84))
                            cbar = plt.colorbar(fraction=0.046, pad=0.04)
                            cbar.set_label('e-/s', fontsize=16)
                            plt.tick_params(labelsize=14)
                            plt.show()

                    img0 = img1
                    
                    volt_rav[i,:] = img1[:,n*ncol_perfit:(n+1)*ncol_perfit].ravel()

            x_matrix = np.array([[1 for x in range(nframe)], [self.tframe*i for i in range(nframe)]]).transpose()
            inv_xprod = np.linalg.inv(np.dot(x_matrix.transpose(), x_matrix))
            second_prod = np.dot(inv_xprod, x_matrix.transpose())
            best_fit = np.dot(second_prod, volt_rav)

            best_fit_slope[:,n*ncol_perfit:(n+1)*ncol_perfit] = (np.reshape(best_fit[1,:], (self.dimx, self.dimy//nsplit)))/self.nd_obj.V_to_e # returns in units of electrons

        return best_fit_slope
        
    def fourier_transform_timestreams(self, timestreams, plot=True):
        
        pass
    


def badpixmask_construct(exp_idxs, channel, nd_obj=None, mask_venn='union', timestr_head=None, base_path=None, mode='difference_exp',\
                         plot_indiv=False, plot_mask=True, save_figs=False, figpath=None, dimx=2048, dimy=2048, \
                        sig_clip_sigma=5.0, nitermax=50, png_or_pdf='png', figdim=8, verbose=False, dat_key='photmap', tailfigstr=None):
    
    median_values, width_values, load_idxs, fracs_bad_indiv = [[] for x in range(4)]
    diffs = []
    
    if nd_obj is None:
        if timestr_head is None:
            print("need to specify timestr_head")
            return
        nd_obj = ciber2_noisedata(timestr_head=timestr_head, base_path=base_path, channel=channel, dimx=dimx, dimy=dimy)
        
    print('file path is ', nd_obj.base_path)
    npix = nd_obj.dimx*nd_obj.dimy
    sum_badpixmask = np.zeros((nd_obj.dimx, nd_obj.dimy))

    if figpath is None:
        figpath = base_path
        
    for i, exp_no in enumerate(exp_idxs):
        if exp_no==exp_idxs[0] and mode=='difference_exp':
            print('exp_no = ', exp_no, 'pass')
            continue
    
        if mode=='single_exp':

            im = nd_obj.load_fits_from_timestr(nd_obj.frame_numbers[exp_no], fits_or_img='img', dat_key=dat_key)
            title = 'Frame '+str(exp_no)+' (Channel '+str(nd_obj.channel)+')'
            im_path = 'frame_image/exp'+str(exp_no)
            hist_path = 'frame_hist/exp'+str(exp_no)
            
            ims.append(im)
            load_idxs.append(exp_no)

        elif mode=='difference_exp':
            try:
                im = nd_obj.img_difference(exp_number_idx1=exp_no, exp_number_idx2=exp_idxs[i-1], dat_key=dat_key, verbose=True)
                title = make_exp_diff_str(idx1=exp_no, idx2=exp_idxs[i-1])+' [e-/s] (Channel '+str(nd_obj.channel)+')'
            
                im_path = 'diff_image/exp'+str(exp_no)+'_'+str(exp_idxs[i-1])
                hist_path = 'diff_hist/exp'+str(exp_no)+'_'+str(exp_idxs[i-1])
                
                if verbose:
                    print('impath:', im_path, ', histpath:', hist_path)

                diffs.append(im)
                load_idxs.append(exp_no)

            except:
                print('no luck here on', exp_idxs[i], exp_idxs[i-1], ', moving on')
                continue

        if plot_indiv:
            framefig = view_img(im, title=title, titlefontsize=20, return_fig=True)
            histfig = view_hist(im, histmin=-100.0, histmax=100.0, nbins=150, return_fig=True, title=title, titlefontsize=16, yscale='symlog')
            if save_figs:
                if figpath is not None:
                    framefig.savefig(figpath+'/'+im_path+'_image_'+str(exposure_time)+'s_exposure_ch'+str(channel)+'.'+png_or_pdf, bbox_inches='tight')
                    histfig.savefig(figpath+'/'+hist_path+'_hist_'+str(exposure_time)+'s_exposure_ch'+str(channel)+'.'+png_or_pdf, bbox_inches='tight')
                else:
                    print('need figpath specified to save, continuing..')

        badpixmask_indiv = iter_sigma_clip_mask(im, sig=sig_clip_sigma, nitermax=nitermax)
        sum_badpixmask += badpixmask_indiv
        fracs_bad_indiv.append(float(npix - np.sum(badpixmask_indiv))/float(npix))

        
    if mask_venn == 'intersect':
        badpixmask = (sum_badpixmask > 0)
        frac_bad_mask = float(npix - np.sum(badpixmask))/float(npix)

    elif mask_venn == 'union':
        badpixmask = (sum_badpixmask == len(diffs))
        frac_bad_mask = float(npix - np.sum(badpixmask))/float(npix)
        
    print('frac_bad_mask is ', frac_bad_mask)
    if plot_mask:
        badpixmask_fig = view_img(~badpixmask, figdim=figdim, vmax=1., vmin=0., title=str(np.round(100*frac_bad_mask, 4))+'% $\\sigma$-clipped pixels', return_fig=True)
        
        if tailfigstr is None:
            tailfigstr = ''
        else:
            tailfigstr = '_'+tailfigstr
        if save_figs:
            fname = figpath+'/ch'+str(channel)+'_badpixmask_'+mask_venn+'_'+mode+'_'+str(sig_clip_sigma)+'sig'+tailfigstr+'.'+png_or_pdf
            print('saving to ', fname)
            badpixmask_fig.savefig(fname, bbox_inches='tight')

    return badpixmask, frac_bad_mask, fracs_bad_indiv



def construct_ciber_photmap_fits(image_list, card_names=['photmap'], wcs_header=None, primary_header=None, header_keys=None, header_vals=None):
    
    # primary header info
    hdu = fits.PrimaryHDU(None)
    if primary_header is not None:
        hdu.header = primary_header

        if header_keys is not None:
            for h, header_key in enumerate(header_keys):
                hdu.header.set(header_key, header_vals[h])
                
    temphdu = None
    cards = [hdu]

    # add image cards
    for e, card_name in enumerate(card_names):
        card_hdu = fits.ImageHDU(image_list[e], name=card_name)
        cards.append(card_hdu)

    if wcs_header is not None:
        for card_hdu in cards:
            card_hdu.header.update(wcs_header.to_header())

    hdulist = fits.HDUList(cards)
    
    return hdulist


def convert_integrations_to_photmaps(channel, timestr_head, base_path, run_info_fname=None, run_df=None, memmax=1024, verbose=False, \
                                    ref_pix_corr=False, save_fits=False, fits_base_path=None, refwide_per_indiv=64, side_list=['bottom'], \
                                    wcs_header=None, clobber=True, nskip=1, plot=False, header_keys=None, header_vals=None):
    
    dr_obj = detector_readout(channel=channel, timestr_head=timestr_head, base_path=base_path)
    
    if run_df is None:
        if run_info_fname is not None:
            run_df = read_in_run_info(base_path+'/'+run_info_fname)
        else:
            print('No run info spreadsheet specified and no dataframe provided, exiting')
            return
    
    dr_obj.nd_obj.parse_run_df(run_df)
    
    dr_obj.nd_obj.indiv_exposure_path = dr_obj.nd_obj.base_path

    exp_numbers = dr_obj.nd_obj.exp_numbers
    
    for i, exp_number in enumerate(exp_numbers):
        
        if exp_number==0:
            continue
        
        print('exp number is ', exp_number)
        dr_obj.exp_number=exp_number
        try:
            exp_idxs = dr_obj.nd_obj.get_available_idxs(frame_or_exp='frame', exp_number=exp_number)
        except:
            exp_number += 1
            try:
                exp_idxs = dr_obj.nd_obj.get_available_idxs(frame_or_exp='frame', exp_number=exp_number)
            except:
                exp_number -= 2
                try:
                    exp_idxs = dr_obj.nd_obj.get_available_idxs(frame_or_exp='frame', exp_number=exp_number)
                except:
                    print('no dice')
                    continue
        
        dr_obj.exp_number=exp_number
            
        bestfitslop = dr_obj.line_fit(exp_idxs[nskip:], verbose=verbose, memmax=memmax)
        card_names=['photmap']

        if plot:
            view_img(bestfitslop)
        
        if ref_pix_corr:
            ref_pix_corr, _ = dr_obj.subtract_ref_pixels(bestfitslop, refwide_per_indiv=refwide_per_indiv, side_list=side_list)
            card_names.append('photmap_refcorr')
        
        if save_fits:
            if verbose:
                print('saving files, exp name ', exp_number)
                
            if fits_base_path is None:
                photpath = base_path+'/photmaps'
            else:
                photpath = fits_base_path+'/photmaps'
                
            if not os.path.isdir(photpath):
                os.mkdir(photpath)
            
            photpath +='/ch'+str(channel)
            if not os.path.isdir(photpath):
                os.mkdir(photpath)
                
            photpath += '/'+timestr_head+'_'+str(exp_number).zfill(6)+'.fits'
            
            ciber_fits = construct_ciber_photmap_fits([bestfitslop], card_names=card_names, wcs_header=wcs_header, header_keys=header_keys, header_vals=header_vals)
            
            if verbose:
                print('saving photmap to ', photpath)
            try:
                ciber_fits.writeto(photpath, clobber=clobber)
            except:
                if verbose:
                    print('clobber set to '+str(clobber)+' and file exists, continuing to next one')
                continue
                
    
                
            


    

    