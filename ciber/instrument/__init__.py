"""
Instrument characterization and calibration.

This module contains functions for CIBER beam/PSF, noise modeling, flat field estimation,
and surface brightness calibration.
"""

# Don't import submodules automatically to avoid circular imports
# Users should explicitly import what they need

__all__ = ['beam', 'noise_model', 'noise_data_utils', 'readnoise', 
           'flat_field', 'calibration', 'frame_processing']
