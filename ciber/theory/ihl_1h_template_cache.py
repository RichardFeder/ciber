"""
Cache and manage one-halo templates from IHL decomposition.

This module provides utilities to:
1. Cache individual and effective one-halo templates
2. Load cached templates directly without recomputing
3. Integrate with power spectrum fitting workflows

Author: Richard Feder
Date: May 2026
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import json


class OneHaloTemplateCache:
    """
    Cache for one-halo templates from IHL decomposition.

    Stores individual z-bin templates and effective (summed) templates
    for quick retrieval during power spectrum fitting.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize cache.

        Parameters
        ----------
        cache_dir : str, optional
            Directory for cache files. Default: data/1h_template_cache/
        """
        if cache_dir is None:
            # cache_dir = 'data/1h_template_cache'
            cache_dir = 'data/1h_template_cache_corrected'

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.cache_dir / 'cache_metadata.json'

    def save_cache(self, effective_1h: Dict, individual_1h: Dict,
                   zbinedges: np.ndarray, slopes: list,
                   description: str = "") -> None:
        """
        Save one-halo templates to cache.

        Parameters
        ----------
        effective_1h : dict
            Effective templates from compute_effective_1h_template()
        individual_1h : dict
            Individual z-bin templates from compute_effective_1h_template()
        zbinedges : array_like
            Redshift bin edges
        slopes : list
            Slope values included
        description : str
            Optional description of cache contents
        """
        print("\n" + "="*70)
        print("SAVING ONE-HALO TEMPLATE CACHE")
        print("="*70)

        zbinedges = np.asarray(zbinedges)
        nzbins = len(zbinedges) - 1

        metadata = {
            'description': description,
            'zbinedges': zbinedges.tolist(),
            'slopes': slopes,
            'nzbins': nzbins,
            'cached_slopes': {}
        }

        for slope in slopes:
            slope_str = f"slope_{slope}"

            # Save effective template
            eff_fname = self.cache_dir / f'effective_1h_{slope_str}.npz'
            np.savez(
                eff_fname,
                ell=effective_1h[slope]['ell'],
                one_halo_sum=effective_1h[slope]['one_halo_sum'],
                one_halo_avg=effective_1h[slope]['one_halo_avg'],
                one_halo_norm=effective_1h[slope]['one_halo_norm'],
            )
            print(f"  ✓ Saved effective template: {eff_fname.name}")

            # Save individual z-bin templates
            individual_fname = self.cache_dir / f'individual_1h_{slope_str}.npz'
            individual_data = {}

            for zidx in range(nzbins):
                if zidx not in individual_1h[slope]:
                    continue

                z_info = individual_1h[slope][zidx]
                prefix = f'zbin_{zidx}_'

                individual_data[prefix + 'ell'] = z_info['ell']
                individual_data[prefix + 'one_halo'] = z_info['one_halo']
                individual_data[prefix + 'z_range'] = np.array(z_info['z_range'])
                individual_data[prefix + 'z_mid'] = z_info['z_mid']
                individual_data[prefix + 'A_1h'] = z_info['A_1h']
                individual_data[prefix + 'mu_1h'] = z_info['mu_1h']
                individual_data[prefix + 'sigma_1h'] = z_info['sigma_1h']

            np.savez(individual_fname, **individual_data)
            print(f"  ✓ Saved individual templates: {individual_fname.name}")

            metadata['cached_slopes'][slope] = {
                'effective_file': eff_fname.name,
                'individual_file': individual_fname.name
            }

        # Save metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✓ Saved metadata: {self.metadata_file.name}")

        print("\n" + "="*70)
        print(f"Cache saved to: {self.cache_dir}")
        print("="*70)

    def load_cache(self, slope: float = 1.0) -> Tuple[Dict, Dict, np.ndarray]:
        """
        Load one-halo templates from cache.

        Parameters
        ----------
        slope : float, optional
            Which slope to load (default 1.0)

        Returns
        -------
        effective_1h : dict
            Effective template data
        individual_1h : dict
            Individual z-bin templates
        zbinedges : array
            Redshift bin edges
        """
        # Load metadata
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Cache metadata not found: {self.metadata_file}")

        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)

        zbinedges = np.array(metadata['zbinedges'])
        nzbins = metadata['nzbins']

        # Convert slope to string key for JSON lookup
        slope_str = str(float(slope))

        if slope_str not in metadata['cached_slopes']:
            raise ValueError(f"Slope {slope} not in cache. "
                           f"Available slopes: {list(metadata['cached_slopes'].keys())}")

        slope_meta = metadata['cached_slopes'][slope_str]

        # Load effective template
        eff_fname = self.cache_dir / slope_meta['effective_file']
        eff_data = np.load(eff_fname)

        effective_1h = {
            slope: {
                'ell': eff_data['ell'] / np.pi,
                'one_halo_sum': eff_data['one_halo_sum'],
                'one_halo_avg': eff_data['one_halo_avg'],
                'one_halo_norm': eff_data['one_halo_norm'],
                'n_bins_summed': nzbins,
            }
        }

        # Load individual z-bin templates
        ind_fname = self.cache_dir / slope_meta['individual_file']
        ind_data = np.load(ind_fname)

        individual_1h = {slope: {}}

        for zidx in range(nzbins):
            prefix = f'zbin_{zidx}_'

            if prefix + 'ell' not in ind_data:
                continue

            individual_1h[slope][zidx] = {
                'ell': ind_data[prefix + 'ell'] / np.pi,
                'one_halo': ind_data[prefix + 'one_halo'],
                'z_range': tuple(ind_data[prefix + 'z_range']),
                'z_mid': float(ind_data[prefix + 'z_mid']),
                'A_1h': float(ind_data[prefix + 'A_1h']),
                'mu_1h': float(ind_data[prefix + 'mu_1h']),
                'sigma_1h': float(ind_data[prefix + 'sigma_1h']),
            }

        print(f"\n✓ Loaded cached templates from: {self.cache_dir}")
        print(f"  Slope: {slope}")
        print(f"  Redshift bins: {zbinedges}")
        print(f"  Individual templates: {len(individual_1h[slope])}")

        return effective_1h, individual_1h, zbinedges

    def get_effective_1h_shape(self, slope: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and return just the effective 1h normalized shape.

        Useful for quick access to the template shape without loading all data.

        Parameters
        ----------
        slope : float, optional
            Which slope to load (default 1.0)

        Returns
        -------
        ell : array
            Multipole values
        one_halo_norm : array
            Normalized effective 1h template
        """
        slope_str = f"slope_{slope}"
        eff_fname = self.cache_dir / f'effective_1h_{slope_str}.npz'

        if not eff_fname.exists():
            raise FileNotFoundError(f"Effective template not found: {eff_fname}")

        data = np.load(eff_fname)
        return data['ell'] / np.pi, data['one_halo_norm']

    def get_effective_1h_sum(self, slope: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and return the summed (unnormalized) effective 1h template.

        Parameters
        ----------
        slope : float, optional
            Which slope to load (default 1.0)

        Returns
        -------
        ell : array
            Multipole values
        one_halo_sum : array
            Summed unnormalized 1h template
        """
        slope_str = f"slope_{slope}"
        eff_fname = self.cache_dir / f'effective_1h_{slope_str}.npz'

        if not eff_fname.exists():
            raise FileNotFoundError(f"Effective template not found: {eff_fname}")

        data = np.load(eff_fname)
        return data['ell'] / np.pi, data['one_halo_sum']

    def list_cached_slopes(self) -> list:
        """
        List all cached slope values.

        Returns
        -------
        slopes : list
            Available slope values in cache
        """
        if not self.metadata_file.exists():
            return []

        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)

        return list(metadata['cached_slopes'].keys())

    def get_effective_lognormal_params(self, slope: float = 1.0) -> tuple:
        """
        Return the effective (mu_1h, sigma_1h) fit to the combined z<1 template.

        These are stored in cache_metadata.json by cache_effective_1h_templates.py.

        Returns
        -------
        mu_1h, sigma_1h : float
        """
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Cache metadata not found: {self.metadata_file}")
        with open(self.metadata_file) as f:
            metadata = json.load(f)
        slope_str = str(float(slope))
        slope_meta = metadata['cached_slopes'].get(slope_str, {})
        if 'effective_mu_1h' not in slope_meta:
            raise KeyError(
                f"effective_mu_1h not found in cache for slope={slope}. "
                "Re-run scripts/cache_effective_1h_templates.py to update the cache."
            )
        return float(slope_meta['effective_mu_1h']), float(slope_meta['effective_sigma_1h'])

    def cache_exists(self) -> bool:
        """
        Check if cache exists and is valid.

        Returns
        -------
        bool
            True if cache metadata and at least one template exists
        """
        return self.metadata_file.exists()


def create_and_cache_effective_1h_template(
    template_dir: str = 'data/ihl_templates',
    zbinedges: Optional[np.ndarray] = None,
    slopes: list = [1.0],
    cache_dir: Optional[str] = None,
    description: str = "",
    plot: bool = True,
    ell_scale: float = 1.0
) -> Tuple[Dict, Dict, np.ndarray]:
    """
    Convenience function to compute effective 1h template and save to cache.

    This combines compute_effective_1h_template() + cache saving in one call.

    Parameters
    ----------
    template_dir : str
        Directory containing IHL template files
    zbinedges : array_like, optional
        Redshift bin edges. Default: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    slopes : list, optional
        Slope values to process (default [1.0])
    cache_dir : str, optional
        Cache directory. Default: data/1h_template_cache/
    description : str, optional
        Description to store in cache metadata
    plot : bool, optional
        Whether to create comparison plots
    ell_scale : float, optional
        Scaling factor for ell values from templates (default 1.0)

    Returns
    -------
    effective_1h, individual_1h, zbinedges
        Same as compute_effective_1h_template()
    """
    from compute_effective_1h_template import compute_effective_1h_template

    if zbinedges is None:
        zbinedges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Compute effective template
    print("\n" + "="*70)
    print("COMPUTING EFFECTIVE ONE-HALO TEMPLATE FOR CACHING")
    print("="*70)

    effective_1h, individual_1h, fit_results = compute_effective_1h_template(
        template_dir=template_dir,
        zbinedges=zbinedges,
        slopes=slopes,
        plot=plot,
        figsize=(14, 10),
        ell_scale=ell_scale
    )

    # Save to cache
    cache = OneHaloTemplateCache(cache_dir=cache_dir)
    cache.save_cache(
        effective_1h=effective_1h,
        individual_1h=individual_1h,
        zbinedges=zbinedges,
        slopes=slopes,
        description=description
    )

    return effective_1h, individual_1h, zbinedges


# Convenience wrapper for fitting pipeline
def load_effective_1h_for_fitting(slope: float = 1.0,
                                   cache_dir: Optional[str] = None) -> np.ndarray:
    """
    Quick load of effective 1h template normalized shape for fitting.

    Returns just the normalized shape array, ready to use as a prior/template.

    Parameters
    ----------
    slope : float
        Slope value to load
    cache_dir : str, optional
        Cache directory

    Returns
    -------
    one_halo_norm : array
        Normalized effective 1h template shape
    """
    cache = OneHaloTemplateCache(cache_dir=cache_dir)
    ell, one_halo_norm = cache.get_effective_1h_shape(slope=slope)
    return one_halo_norm
