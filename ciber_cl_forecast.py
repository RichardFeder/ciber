from scipy.interpolate import interp1d
import matplotlib
from scipy.ndimage import gaussian_filter1d
import numpy as np
import sys
import os
import config

# Get the parent directory
# parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
# # Add the parent directory to sys.path
# sys.path.append(parent_dir)
from ciber_powerspec_pipeline import *
from ciber_data_file_utils import *


def forecast_vs_nbar(ciber_inst, ifield, nbar_fid=2e4, Adeg=20., mask_frac=0.7, \
                    nbar_list=[1000, 5000, 20000, 100000], plot=True):

    clf = ciber_cl_forecast(nbar=nbar_fid, Adeg=Adeg, mask_frac=mask_frac)

    clf.load_bl(ciber_inst, ifield)

    clf.load_clg(ciber_inst, plot=plot)

    clf.load_clI_auto(ciber_inst, plot=plot)
    clf.load_nlI_auto(ciber_inst, ifield, beam_correct=True, plot=plot)
    
    lb, dclsq, dcl_bp, dcl_terms_bp, terms_fid, inverse_nl, fig, _ = clf.forecast_ciber_gal_cross(nbar_fid, ylim=[1e-3, 1e0], plot=plot)

    all_dcl_bp = []
    for nbar in nbar_list:
        lb, dclsq, dcl_bandpowers, _, terms, inverse_nl, fig, xerr = clf.forecast_ciber_gal_cross(nbar, ylim=[1e-3, 1e0], plot=plot)

        all_dcl_bp.append(dcl_bandpowers)


    return clf, lb, all_dcl_bp, dcl_terms_bp, terms_fid, xerr, nbar_list
    

class ciber_cl_forecast():
    
    
    band_dict = dict({1:'J', 2:'H'})

    def __init__(self, ell_min=300, ell_max=1e5, Adeg=20, nbar=None, noisemodl_datestr='111323', \
                mask_frac=None):
        
        
        self.cbps = CIBER_PS_pipeline()
        
        self.ell_min = ell_min
        self.ell_max = ell_max
        
        self.lrange = np.arange(self.ell_min, self.ell_max)
        self.pf = self.lrange*(self.lrange+1)/(2*np.pi)
        self.Adeg = Adeg
        
        self.nbar = nbar # put in sr-1
        
        self.fsky = self.Adeg/41253.

        if mask_frac is not None:
            self.fsky *= mask_frac

        self.noisemodl_datestr = noisemodl_datestr

        self.mask_frac = mask_frac
        
        print('fsky = ', self.fsky)


    def load_bl(self, ciber_inst, ifield, inplace=True, plot=False):
        
        # load files
                
        data_dir = '/Users/richardfeder/Documents/ciber/data/fluctuation_data/TM'+str(ciber_inst)+'/'
        bls_fpath = data_dir+'/beam_correction/bl_est_postage_stamps_TM'+str(ciber_inst)+'_081121.npz'

        beamdat = np.load(bls_fpath)
        
        blval = beamdat['B_ells_post'][ifield-4,:]
        
        lb = self.cbps.Mkk_obj.midbin_ell
        
        bl = interp1d(lb, blval, bounds_error=False, fill_value=1.0)
        
        lbindiv = np.arange(np.min(lb), np.max(lb))

        if plot:
            plt.figure(figsize=(4, 3))
            plt.scatter(lb, blval, color='b')
            plt.plot(lbindiv, bl(lbindiv), color='r')
            plt.yscale('log')
            plt.xscale('log')
            plt.show()
        
        if inplace:
            self.bl = bl
            
        else:
            return bl
        
        
    def load_clx(self, ciber_inst, catname='WISE', addstr='unWISE_neo8'):
    
        clx_file = np.load('output/fieldav_clx_'+catname+'_'+addstr+'_kappa_TM'+str(ciber_inst)+'.npz')
        
        lC, clx = clx_file['lC'], clx_file['clx']
        
        clx_interp = interp1d(lC, clx)
        
        self.clx = gaussian_smooth(np.abs(clx_interp(self.lrange)), 50)
        
    
    def plot_all_cls(self, ylim=[1e-12, 1e-6], bbox_to_anchor=[0.0, 1.2], ncol=3):
        
        fig = plt.figure(figsize=(5, 4))
        plt.plot(self.lrange, self.clg_clus, label='$C_{\\ell}^{g}$')
        plt.axhline(self.clg_sn, label='$1/\\overline{n}$', color='k', linestyle='dashed')

        plt.plot(self.lrange, self.clI_auto, label='$C_{\\ell}^{I}$')
        plt.plot(self.lrange, self.clx, label='$C_{\\ell}^{\\kappa g}$')
        plt.legend(ncol=ncol, bbox_to_anchor=bbox_to_anchor)

        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel('$C_{\\ell}$', fontsize=14)
        plt.xlabel('$\\ell$', fontsize=14)
        plt.ylim(ylim)
        plt.grid(alpha=0.3)
        plt.show()  
        return fig  
        
    def load_clg(self, ciber_inst, catname='WISE', addstr='unWISE_neo8', plot=False, basepath=None):
        
        # load autos from unWISE
        
        cgps_file = load_ciber_gal_ps(ciber_inst, catname, addstr=addstr, basepath=basepath)

        lb, all_cl_gal, all_clerr_gal, ifield_list_use = [cgps_file[key] for key in ['lb', 'all_cl_gal', 'all_clerr_gal', 'ifield_list_use']]  

        clg = np.mean(all_cl_gal, axis=0) # used for sample variance estimate
                
        # separate shot noise from clustering
        
        clg_sn = clg[-1]
        # clg_sn = np.mean(clg[lb > 1e5])
        
        clg_clus = clg - clg_sn
                
        clg_tot_interp = interp1d(lb, clg)
        clg_interp = interp1d(lb, clg_clus)


        self.clg_clus = clg_interp(self.lrange)
        self.clg_sn = clg_sn

        if plot:
        
            plt.figure(figsize=(5, 4))
            plt.plot(self.lrange, clg_tot_interp(self.lrange), label='Total $C_{\\ell}^g$')
            plt.plot(self.lrange, self.clg_clus, linestyle='dashed', color='r', label='clus')
            plt.axhline(self.clg_sn, color='k', linestyle='dashed', label='1/$\\overline{n}$')
            plt.legend()
            plt.yscale('log')
            plt.xscale('log')
            plt.ylabel('Clg')
            plt.show()        
        

    def load_nlI_auto(self, ciber_inst, ifield, run_name=None, apply_FW=False, beam_correct=False, plot=False):

        noisemodl_basepath = config.ciber_basepath+'data/noise_models_sim/'+self.noisemodl_datestr+'/TM'+str(ciber_inst)+'/'

        if run_name is None:
            run_name = 'observed_'+str(self.band_dict[ciber_inst])+'lt17.0_042624_quadoff_grad'

        noisemodl_tailpath = '/noise_bias_fieldidx'+str(ifield-4)+'.npz'

        noisemodl_fpath = noisemodl_basepath + run_name +'/'+ noisemodl_tailpath

        print('Loading noise bias from ', noisemodl_fpath)

        noisemodl_file = np.load(noisemodl_fpath)

        if apply_FW:
            fourier_weights = noisemodl_file['fourier_weights_nofluc']
        else:
            fourier_weights = None 

        mean_cl2d_nofluc = noisemodl_file['mean_cl2d_nofluc']

        if plot:
            plot_map(mean_cl2d_nofluc, title='nl2d', figsize=(6, 6))
        nl_dict = self.cbps.compute_noise_power_spectrum(ciber_inst, noise_Cl2D=mean_cl2d_nofluc.copy(), inplace=False, apply_FW=apply_FW, weights=fourier_weights)

        lb = nl_dict['lbins']
        N_ell_est = nl_dict['Cl_noise']

        if beam_correct:
            print('De-beaming noise bias..')
            N_ell_est /= self.bl(lb)**2

        nlI = interp1d(lb, N_ell_est, kind='linear', bounds_error=False, fill_value="extrapolate")

        self.nlI_auto = nlI(self.lrange)

        if plot:
            plt.figure()
            plt.scatter(lb, N_ell_est, color='k')
            plt.plot(self.lrange, self.nlI_auto, color='r')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('$\\ell$')
            plt.ylabel('$C_{\\ell}$')
            plt.xlim(2e2, 1e5)
            plt.show()

        
    def load_clI_auto(self, ciber_inst, plot=False):
        
        ''' Load prediction for CIBER auto power spectrum '''
        
        ciber_cl = np.load('data/input_recovered_ps/ciber_auto_'+self.band_dict[ciber_inst]+'lt16.0_F25B.npz')
        lb, clauto = ciber_cl['lb'], ciber_cl['fieldav_cl']   

        ciber_auto = interp1d(lb, clauto, kind='linear', bounds_error=False, fill_value='extrapolate')
 
        # # to do load noise bias from fiducial auto run
        # self.nlI_auto = None

        if plot:
            plt.figure(figsize=(5, 4))
            plt.plot(self.lrange, ciber_auto(self.lrange))
            plt.yscale('log')
            plt.xscale('log')
            plt.ylabel('$C_{\\ell}^I$', fontsize=14)
            plt.xlabel('$\\ell$', fontsize=14)
            plt.show()
        
        self.clI_auto = ciber_auto(self.lrange)


    def perl_to_bandpowers(self, dcl, lEdges=None):

        if lEdges is None:
            lEdges = self.cbps.Mkk_obj.binl

        lcen = np.sqrt(lEdges[:-1]*lEdges[1:])
        dcl_bandpowers = np.zeros_like(lcen)

        for x in range(len(lEdges)-1):
            delta_ell = lEdges[x+1]-lEdges[x]

            dcl_bandpowers[x] = np.mean(dcl[(self.lrange > lEdges[x])*(self.lrange < lEdges[x+1])])/np.sqrt(delta_ell)

        return dcl_bandpowers, lEdges


    def forecast_ciber_gal_cross(self, nbar, lab_fs=14, ylim=[1e-11, 1e-6], figsize=(5, 4), plot=True):

        
        nbar_sr = nbar*(180/np.pi)**2
        def n_ell_inv(l):
            return 1./(2*l+1)
        
        inverse_nl = n_ell_inv(self.lrange)   
        
        inverse_nl /= self.fsky
                
        term1 = (5*self.clg_clus)**2
        term2 = self.clI_auto*self.clg_clus
        term3 = self.nlI_auto*self.clg_clus
        
        term4 = self.clI_auto/nbar_sr
        term5 = self.nlI_auto*self.clg_sn
    
#         terms = [term1, term2, term3]
        terms = np.array([term1, term2, term3, term4, term5])
        # terms = [term1, term2, term4]
        
        dclsq = inverse_nl*(term1+term2+term3+term4+term5)

        dclsq_terms = np.array([inverse_nl*term for term in terms])

        # dclsq = inverse_nl*(term1+term2+term4)
        
        labels = ['$\\propto (C_{\\ell}^{I\\times g})^2$', '$\\propto C_{\\ell}^{I}C_{\\ell}^g$', '$\\propto N_{\\ell}^{I}C_{\\ell}^g$', \
                 '$\\propto C_{\\ell}^{I} \\overline{n}^{-1}$', '$\\propto N_{\\ell}^{I}\\overline{n}^{-1}$']

        # labels = ['$(C_{\\ell}^{I\\times g})^2$', '$C_{\\ell}^{I}C_{\\ell}^g$', \
        #          '$C_{\\ell}^{I} \\overline{n}^{-1}$']


        dcl_bandpowers, lEdges = self.perl_to_bandpowers(np.sqrt(dclsq))

        dcl_term_bandpowers = np.zeros((5, dcl_bandpowers.shape[0]))

        for t, term in enumerate(terms):

            dcl_term_bandpowers[t] = self.perl_to_bandpowers(np.sqrt(dclsq_terms[t]))[0]


        lbtemp = self.cbps.Mkk_obj.midbin_ell
        xerr = [self.cbps.Mkk_obj.midbin_ell-lEdges[:-1], lEdges[1:]-self.cbps.Mkk_obj.midbin_ell]
        
        if plot:
            pf = lbtemp*(lbtemp+1)/(2*np.pi)

            fig = plt.figure(figsize=figsize)
            for x in range(len(terms)):

                plt.errorbar(self.cbps.Mkk_obj.midbin_ell, pf*dcl_term_bandpowers[x], xerr=xerr, fmt='o',  label=labels[x])

                # plt.plot(self.lrange, self.pf*np.sqrt(inverse_nl*terms[x]), label=labels[x])
            # plt.plot(self.lrange, self.pf*np.sqrt(dclsq), label='Total', color='k', linewidth=2, linestyle='dashed', alpha=0.)


            plt.errorbar(self.cbps.Mkk_obj.midbin_ell, pf*dcl_bandpowers,\
                             xerr=xerr, \
                             fmt='o', color='k', label='Bandpower sensitivity')


            plt.yscale('log')
            plt.xscale('log')
            plt.legend(ncol=2)
            
            plt.ylim(ylim)
            plt.xlim(250, 1e5)
            plt.ylabel('$D_{\\ell}$ [nW m$^{-2}$ sr$^{-1}$]', fontsize=lab_fs)
            plt.xlabel('$\\ell$', fontsize=lab_fs)
            
            plt.grid(alpha=0.3)
            plt.show()




        else:
            fig = None
                
        return lbtemp, dclsq, dcl_bandpowers, dcl_term_bandpowers, terms, inverse_nl, fig, xerr
