"""
Test configuration and fixtures for pytest.
"""

import pytest
import numpy as np


@pytest.fixture
def sample_map():
    """Generate a simple test map."""
    return np.random.randn(256, 256)


@pytest.fixture
def sample_mask():
    """Generate a simple test mask."""
    mask = np.ones((256, 256))
    # Mask out some regions
    mask[120:136, 120:136] = 0
    return mask


@pytest.fixture
def sample_power_spectrum():
    """Generate a sample power spectrum."""
    ell = np.logspace(2, 4, 20)
    cl = 1000 * (ell / 1000)**(-3)
    return ell, cl
