"""
CIBER Power Spectrum Analysis Package

A comprehensive package for analyzing CIBER (Cosmic Infrared Background ExpeRiment) data,
including power spectrum estimation, mock generation, masking, and cross-correlation analysis.
"""

__version__ = "1.0.0"
__author__ = "Richard Feder"

# Top-level imports for convenient access
from . import core
from . import instrument
from . import mocks
from . import pseudo_cl
from . import masking
from . import cross_correlation
from . import theory
from . import processing
from . import io
from . import plotting

__all__ = [
    'core',
    'instrument',
    'mocks',
    'pseudo_cl',
    'masking',
    'cross_correlation',
    'theory',
    'processing',
    'io',
    'plotting',
]
