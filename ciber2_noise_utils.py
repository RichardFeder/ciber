import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy.stats import norm


def read_in_run_info(path):
    if path.lower().endswith('.xlsx'):
        run_df = pd.read_excel(path)
    return run_df

def img_from_fits(fits_file, dat_idx=0):
    return fits_file[dat_idx].data

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
    
    def __init__(self, timestr_head='RUN02102020', base_path='../data/20200210_noise_data',\
                 shutter_key='Shutter', exp_time_key='Exp Time [s]', idx_no_key='Image', power_key='Power', \
                notes_key='Notes', channel=1, figdim=8, dimx=2048, dimy=2048):
        
        for attr, valu in locals().items():
            setattr(self, attr, valu)

        self.indiv_exposure_path = base_path+'/indiv_exposures/'

    def frame_difference(self, img1=None, img2=None, idx1=None, idx2=None, frame_or_exp='exposure', exp_number=None):
        
        if img1 is None:
            
            if frame_or_exp=='exposure':

                img1 = self.load_fits_from_timestr(self.exp_numbers[idx1], fits_or_img='img', frame_or_exp=frame_or_exp)
                img2 = self.load_fits_from_timestr(self.exp_numbers[idx2], fits_or_img='img', frame_or_exp=frame_or_exp)

            elif frame_or_exp=='frame':

                img1 = self.load_fits_from_timestr(exp_number, fits_or_img='img', frame_or_exp=frame_or_exp, frame_idx=idx1)
                img2 = self.load_fits_from_timestr(exp_number, fits_or_img='img', frame_or_exp=frame_or_exp, frame_idx=idx2)

        return img1 - img2

    def load_fits_from_timestr(self, timestr, fits_or_img='fits', dat_idx=0, frame_or_exp='exposure', frame_idx=None):
        
        if len(str(timestr)) < 6:
            timestr = '0'+str(timestr)
            
        if frame_or_exp=='exposure':
            path = self.base_path+'/ch'+str(self.channel)+'/'+self.timestr_head+'_'+str(timestr)+'.fits'
        
        elif frame_or_exp=='frame':
            path = self.indiv_exposure_path+'/ch'+str(self.channel)+'/'+self.timestr_head+'_'+str(timestr)+'/FRM'+str(frame_idx)+'_PIX.fts'

        print('path:', path)
        f = fits.open(path)

        if fits_or_img=='fits':
            return f
        else:
            return img_from_fits(f,dat_idx=dat_idx)
        
        
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
            for file in os.listdir(self.indiv_exposure_path+'ch'+str(self.channel)+'/'+self.timestr_head+'_'+str(exp_number)):
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
                       show=True, histmin=-100, histmax=100, frame_or_exp='exposure', exp_number=133255, figdim=None):
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
                im = self.frame_difference(idx1=exp_no, idx2=exp_idxs[i-1], frame_or_exp=frame_or_exp, exp_number=exp_number)
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
    
   
    