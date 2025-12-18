"""
Theoretical predictions and models.

This module contains functions for computing theoretical power spectra,
halo models, and EBL/CIB predictions.
"""

from .cl_predictions import *
from .halo_model import *
from .helgason_model import *
from .cl_wtheta import *

__all__ = ['cl_predictions', 'halo_model', 'helgason_model', 'cl_wtheta']
