"""
Cross-correlation analysis.

This module contains functions for CIBER × galaxy, CIBER × Spitzer, 
EBL tomography, and other cross-correlation analyses.
"""

# Don't import submodules automatically to avoid circular imports
# Users should explicitly import what they need

__all__ = ['galaxy_cross', 'cross_spectrum', 'spitzer_cross', 'angular_corr', 
           'ebl_tomography', 'ebl_tom_min', 'cl_forecast']
