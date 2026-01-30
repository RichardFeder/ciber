"""
Compare mock-derived uncertainties with per-field uncertainties from real data

Author: GitHub Copilot
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from ciber.plotting.gal_plotting_fns import collect_ciber_gal_vs_redshift


def compare_mock_vs_data_uncertainties(inst, zbinedges, zbin_idx, sigz_phot, 
                                       catname='DESILS', 
                                       ifield_list=[4, 5, 6, 7, 8],
                                       mock_basedir='data/',
                                       galstr='sdss_z_lt_22.0',
                                       subtract_randoms=True,
                                       maskstr='JHlt16_wFFerr',
                                       startidx=2, endidx=-1,
                                       figsize=(10, 4),
                                       xlim=[250, 1e5],
                                       colors=None,
                                       show_ratio=True):
    """
    Compare mock-derived uncertainties with per-field data uncertainties for a given redshift bin.
    
    Parameters
    ----------
    inst : int
        CIBER instrument (1 or 2)
    zbinedges : array_like
        Redshift bin edges
    zbin_idx : int
        Index of redshift bin to compare
    sigz_phot : float
        Photo-z error used in mocks
    catname : str, optional
        Catalog name for real data (default 'DESILS')
    ifield_list : list, optional
        List of field indices
    mock_basedir : str, optional
        Base directory for mock files
    galstr : str, optional
        Galaxy sample string for mocks
    subtract_randoms : bool, optional
        Whether randoms were subtracted in real data
    maskstr : str, optional
        Mask string for real data
    startidx : int, optional
        Starting bandpower index
    endidx : int, optional
        Ending bandpower index
    figsize : tuple, optional
        Figure size
    xlim : tuple, optional
        x-axis limits
    colors : list, optional
        Colors for each field
    show_ratio : bool, optional
        Show ratio panel
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    results : dict
        Dictionary with comparison results
    """
    
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(ifield_list)))
    
    # Load real data uncertainties
    print(f"Loading real data for {catname}...")
    data_res = collect_ciber_gal_vs_redshift(
        catname, 
        subtract_randoms=subtract_randoms,
        inst_list=[inst],
        zbinedges=zbinedges,
        maskstr=maskstr,
        ifield_list=ifield_list,
        startidx=startidx,
        endidx=endidx,
        with_ff_err=('wFFerr' in maskstr)
    )
    
    lb = data_res['lb']
    
    # Extract per-field uncertainties from real data
    # Shape: (n_inst, n_zbins, n_fields, n_ell)
    data_clerr_perf = data_res['full_clerr_cross_perf'][0, zbin_idx]  # [0] for inst index
    
    # Load and process mock results for each field
    mock_clerr_perf = []
    
    for fieldidx, ifield in enumerate(ifield_list):
        mock_fpath = f"{mock_basedir}mock_redshift_tom_res_sigmazphot={sigz_phot}_dz=0.2_TM{inst}_ifield{ifield}.npz"
        
        try:
            mock_res = np.load(mock_fpath, allow_pickle=True)
            all_est_clx = mock_res['all_est_clx']
            
            # Compute uncertainty from 16-84 percentile
            # all_est_clx shape: (n_realizations, n_zbins, n_ell)
            std_clx = 0.5 * (np.percentile(all_est_clx, 84, axis=0)[zbin_idx, :] - 
                            np.percentile(all_est_clx, 16, axis=0)[zbin_idx, :])
            
            mock_clerr_perf.append(std_clx)
            
        except FileNotFoundError:
            print(f"Warning: Mock file not found for ifield={ifield}, skipping")
            mock_clerr_perf.append(np.full_like(lb, np.nan))
    
    mock_clerr_perf = np.array(mock_clerr_perf)
    
    # Create figure
    if show_ratio:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                       gridspec_kw={'height_ratios': [3, 1]}, 
                                       sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    
    # Plot uncertainties for each field
    field_names = {4: 'elat10', 5: 'elat30', 6: 'Bootes B', 7: 'Bootes A', 8: 'SWIRE'}
    
    for fieldidx, ifield in enumerate(ifield_list):
        field_label = field_names.get(ifield, f'Field {ifield}')
        
        # Mock uncertainties
        ax1.plot(lb[startidx:endidx], mock_clerr_perf[fieldidx, startidx:endidx], 
                '-', color=colors[fieldidx], linewidth=2, 
                label=f'{field_label} (mock)', alpha=0.8)
        
        # Data uncertainties
        ax1.plot(lb[startidx:endidx], data_clerr_perf[fieldidx, startidx:endidx], 
                '--', color=colors[fieldidx], linewidth=2, 
                label=f'{field_label} (data)', alpha=0.8)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel(r'$\sigma(C_\ell)$ [nW$^2$ m$^{-4}$ sr$^{-2}$]', fontsize=14)
    ax1.grid(alpha=0.3)
    ax1.legend(ncol=2, fontsize=10, loc='best')
    
    z0, z1 = zbinedges[zbin_idx], zbinedges[zbin_idx + 1]
    title = f'TM{inst}, ${z0:.1f} < z < {z1:.1f}$, $\sigma_z={sigz_phot}$'
    ax1.set_title(title, fontsize=14)
    
    if xlim is not None:
        ax1.set_xlim(xlim)
    
    # Ratio panel
    if show_ratio:
        for fieldidx, ifield in enumerate(ifield_list):
            ratio = data_clerr_perf[fieldidx, startidx:endidx] / mock_clerr_perf[fieldidx, startidx:endidx]
            ax2.plot(lb[startidx:endidx], ratio, '-', color=colors[fieldidx], 
                    linewidth=2, alpha=0.8)
        
        ax2.axhline(1.0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xscale('log')
        ax2.set_xlabel(r'$\ell$', fontsize=14)
        ax2.set_ylabel('Data / Mock', fontsize=12)
        ax2.set_ylim([0.5, 2.0])
        ax2.grid(alpha=0.3)
        
        if xlim is not None:
            ax2.set_xlim(xlim)
        
        plt.subplots_adjust(hspace=0.05)
    else:
        ax1.set_xlabel(r'$\ell$', fontsize=14)
    
    plt.tight_layout()
    
    # Compile results
    results = {
        'lb': lb,
        'mock_clerr_perf': mock_clerr_perf,
        'data_clerr_perf': data_clerr_perf,
        'ratio': data_clerr_perf / mock_clerr_perf,
        'ifield_list': ifield_list,
        'zbinedges': zbinedges,
        'zbin_idx': zbin_idx
    }
    
    return fig, results


def compare_all_zbins_mock_vs_data(inst, zbinedges, sigz_phot,
                                   catname='DESILS',
                                   ifield_list=[4, 5, 6, 7, 8],
                                   mock_basedir='data/',
                                   subtract_randoms=True,
                                   maskstr='JHlt16_wFFerr',
                                   startidx=2, endidx=-1,
                                   figsize=(12, 8),
                                   xlim=[250, 1e5]):
    """
    Create multi-panel comparison for all redshift bins.
    
    Parameters
    ----------
    inst : int
        CIBER instrument
    zbinedges : array_like
        Redshift bin edges
    sigz_phot : float
        Photo-z error
    (other parameters same as compare_mock_vs_data_uncertainties)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    all_results : list of dict
        Results for each redshift bin
    """
    
    n_zbins = len(zbinedges) - 1
    ncols = min(3, n_zbins)
    nrows = int(np.ceil(n_zbins / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    if n_zbins == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    all_results = []
    colors = plt.cm.tab10(np.linspace(0, 1, len(ifield_list)))
    
    # Load real data once
    print(f"Loading real data for {catname}...")
    # data_res = collect_ciber_gal_vs_redshift(
    #     catname,
    #     subtract_randoms=subtract_randoms,
    #     inst_list=[inst],
    #     zbinedges=zbinedges,
    #     maskstr=maskstr,
    #     ifield_list=ifield_list,
    #     startidx=startidx,
    #     endidx=endidx,
    #     with_ff_err=('wFFerr' in maskstr)
    # )

    if catname=='DESILS':

        data_res = collect_ciber_gal_vs_redshift('LS', subtract_randoms=True, \
                                        inst_list=[inst], zbinedges=zbinedges, \
                                        maskstr=maskstr, subtract_sn=False, 
                                        tl_pix_correct=True)
        
    elif catname=='HSC':
        data_res = collect_ciber_gal_vs_redshift('HSC', subtract_randoms=True, \
                                    inst_list=[inst], zbinedges=zbinedges, \
                                    maskstr=None, subtract_sn=False, 
                                    tl_pix_correct=True, 
                                    ifield_list=[8], 
                                    with_ff_err=True, 
                                    headstr='hsc_ilt24.0')
    
    lb = data_res['lb']
    field_names = {4: 'elat10', 5: 'elat30', 6: 'Bootes B', 7: 'Bootes A', 8: 'SWIRE'}
    
    for zbin_idx in range(n_zbins):
        ax = axes[zbin_idx]
        
        # Extract data uncertainties
        data_clerr_perf = data_res['full_clerr_cross_perf'][0, zbin_idx]
        
        # Load mock uncertainties
        mock_clerr_perf = []
        for fieldidx, ifield in enumerate(ifield_list):
            mock_fpath = f"{mock_basedir}mock_redshift_tom_res_sigmazphot={sigz_phot}_dz=0.2_TM{inst}_ifield{ifield}.npz"
            
            try:
                mock_res = np.load(mock_fpath, allow_pickle=True)

                all_true_clx = mock_res['all_true_clx']

                all_est_clx = mock_res['all_est_clx']
                # std_clx = 0.5 * (np.percentile(all_est_clx, 84, axis=0)[zbin_idx, :] - 
                #                 np.percentile(all_est_clx, 16, axis=0)[zbin_idx, :])
                
                resid = all_est_clx - all_true_clx
                std_clx = 0.5*(np.percentile(resid,84,axis=0)[zbin_idx,:] - np.percentile(resid,16,axis=0)[zbin_idx,:])

                mock_clerr_perf.append(std_clx)
            except FileNotFoundError:
                mock_clerr_perf.append(np.full_like(lb, np.nan))
        
        mock_clerr_perf = np.array(mock_clerr_perf)
        
        # Plot
        for fieldidx, ifield in enumerate(ifield_list):
            show_label = (zbin_idx == 0)  # Only show legend in first panel
            field_label = field_names.get(ifield, f'Field {ifield}')
            
            ax.plot(lb[startidx:endidx], mock_clerr_perf[fieldidx, startidx:endidx],
                   '-', color=colors[fieldidx], linewidth=1.5, 
                   label=f'{field_label} (mock)' if show_label else '', alpha=0.8)
            
            ax.plot(lb[startidx:endidx], data_clerr_perf[fieldidx, startidx:endidx],
                   '--', color=colors[fieldidx], linewidth=1.5,
                   label=f'{field_label} (data)' if show_label else '', alpha=0.8)
        
        z0, z1 = zbinedges[zbin_idx], zbinedges[zbin_idx + 1]
        ax.set_title(f'${z0:.1f} < z < {z1:.1f}$', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        
        if xlim is not None:
            ax.set_xlim(xlim)
        
        # Store results
        all_results.append({
            'lb': lb,
            'mock_clerr_perf': mock_clerr_perf,
            'data_clerr_perf': data_clerr_perf,
            'ratio': data_clerr_perf / mock_clerr_perf,
            'zbin_idx': zbin_idx
        })
    
    # Add legend to first panel
    axes[0].legend(ncol=2, fontsize=8, loc='best')
    
    # Labels
    for i in range(n_zbins, len(axes)):
        axes[i].axis('off')
    
    for i in range(nrows):
        axes[i * ncols].set_ylabel(r'$\sigma(C_\ell)$', fontsize=12)
    
    for i in range(ncols):
        axes[(nrows - 1) * ncols + i].set_xlabel(r'$\ell$', fontsize=12)
    
    plt.suptitle(f'TM{inst}, $\sigma_z={sigz_phot}$: Mock vs Data Uncertainties', 
                fontsize=14, y=0.995)
    plt.tight_layout()
    
    return fig, all_results
