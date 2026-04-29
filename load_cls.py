import numpy as np
import glob
import os
import matplotlib.pyplot as plt


def plot_cl_matrix_components(input_dir, file_prefix='feder25_cl', 
                              figsize_per_subplot=(4, 3), show_errors=True,
                              ylim=[1e-3, 1e4], xlim=[800, 1e5], colors=None,
                              separate_figures=False, verbose=True):
    """
    Load and plot the saved power spectrum components in log-log plots.
    
    Parameters
    ----------
    input_dir : str
        Directory containing the saved .npz files
    file_prefix : str, optional
        Prefix of the saved files (default 'cl_component')
    figsize_per_subplot : tuple, optional
        Size of each subplot (default (4, 3))
    show_errors : bool, optional
        Whether to show error bars (default True)
    ylim : tuple, optional
        Y-axis limits for all plots (auto if None)
    xlim : tuple, optional  
        X-axis limits for all plots (auto if None)
    colors : dict, optional
        Color mapping for different component types
    separate_figures : bool, optional
        If True, create separate figure for each component (default False)
    verbose : bool, optional
        Print loading info (default True)
    
    Returns
    -------
    figures : list
        List of matplotlib figure objects
    """
    # import os
    # import glob
    # import matplotlib.pyplot as plt
    
    # Find all matching files
    pattern = os.path.join(input_dir, f"{file_prefix}_*.npz")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return []
    
    # Default colors
    if colors is None:
        colors = {
            'auto': {'1.1': 'blue', '1.8': 'red', '3.6': 'green', '4.5': 'orange'},
            'cross': 'gray'
        }
    


    # Load data from all files
    data_dict = {}
    for file_path in files:
        filename = os.path.basename(file_path)
    
        # Load data
        data = np.load(file_path)
        lb = data['lb']
        cl = data['cl'] 
        dcl = data['dcl']
        component_type = str(data['component_type'])
        band_name_i = data['band_name_i']
        band_name_j = data['band_name_j']

        print('cl:', cl)
        
        # Create component label
        if component_type == 'auto':
            comp_label = f"{band_name_i}μm auto"
            color = colors['auto'].get(f"{band_name_i:g}", 'black')
        else:
            comp_label = f"{band_name_i}μm×{band_name_j}μm cross"
            color = colors['cross']
        

        data_dict[comp_label] = {
            'lb': lb, 'cl': cl, 'dcl': dcl, 
            'color': color, 'type': component_type,
            'filename': filename
        }
        
        if verbose:
            print(f"Loaded: {comp_label} from {filename}")
    
    figures = []
    
    if separate_figures:
        # Create separate figure for each component
        for comp_label, data in data_dict.items():
            fig, ax = plt.subplots(1, 1, figsize=figsize_per_subplot)
            
            # Plot data

            pf = data['lb']*(data['lb']+1)/(2*np.pi)
            if show_errors and np.any(data['dcl'] > 0):
                ax.errorbar(data['lb'], pf*data['cl'], yerr=pf*data['dcl'], 
                           fmt='o-', color=data['color'], capsize=3, 
                           markersize=4, linewidth=1.5, label=comp_label)
            else:
                ax.plot(data['lb'], pf*data['cl'], 'o-', color=data['color'],
                       markersize=4, linewidth=1.5, label=comp_label)
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('ℓ')
            ax.set_ylabel('C_ℓ')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_title(comp_label)
            
            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)
                
            plt.tight_layout()
            figures.append(fig)
    
    else:
        # Create single figure with subplots
        n_components = len(data_dict)
        n_cols = min(4, n_components)  # Max 4 columns
        n_rows = int(np.ceil(n_components / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(figsize_per_subplot[0]*n_cols, 
                                        figsize_per_subplot[1]*n_rows))
        
        # Handle single subplot case
        if n_components == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each component
        for idx, (comp_label, data) in enumerate(data_dict.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Plot data
            if show_errors and np.any(data['dcl'] > 0):
                ax.errorbar(data['lb'], data['cl'], yerr=data['dcl'], 
                           fmt='o-', color=data['color'], capsize=3,
                           markersize=4, linewidth=1.5)
            else:
                ax.plot(data['lb'], data['cl'], 'o-', color=data['color'],
                       markersize=4, linewidth=1.5)
            
            ax.set_xscale('log')
            ax.set_yscale('log') 
            ax.set_xlabel('ℓ')
            ax.set_ylabel('C_ℓ')
            ax.grid(True, alpha=0.3)
            ax.set_title(comp_label)
            
            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)
        
        # Hide empty subplots
        for idx in range(n_components, n_rows * n_cols):
            row = idx // n_cols  
            col = idx % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            elif n_cols > 1:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        figures.append(fig)
    
    return figures