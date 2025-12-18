"""
External dataset processing.

This module contains functions for processing data from WISE, photo-z analysis,
and other external datasets.
"""

from .wise_processing import *
try:
    from .photo_z import *
except ImportError:
    pass  # photo_z module optional

__all__ = ['wise_processing']
