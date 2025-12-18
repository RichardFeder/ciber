"""
Mock generation and simulations.

This module contains functions for generating CIB mocks, galaxy catalogs,
and simulated observations.
"""

# Import only cib_mocks by default, others need explicit import to avoid circular dependencies
from .cib_mocks import *

__all__ = ['cib_mocks', 'galaxy_catalogs', 'grigory_mocks', 'lognormal', 
           'mock_gal_cross', 'proc_jmocks']
