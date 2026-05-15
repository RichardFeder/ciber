#!/usr/bin/env python
"""
Quick validation script for Gaia stellar density cross fit refactor.
Tests data loading and model setup without running expensive MCMC.
"""

import numpy as np
import sys

def test_data_loading():
    """Test that we can load the Gaia cross spectrum data."""
    try:
        from ciber.plotting.gal_plotting_fns import load_ciber_gal_ps
        print("✓ Loading gal_plotting_fns...")
        
        # Try loading TM1 (J-band)
        for inst in [1, 2]:
            try:
                cgps_file = load_ciber_gal_ps(inst, "gaia", addstr="stars_glt20p5_JHlt14_wFFerr")
                lb = cgps_file["lb"]
                all_cl_cross = cgps_file["all_cl_cross"]
                print(f"✓ Loaded Gaia cross data for TM{inst}")
                print(f"  - lb shape: {lb.shape}")
                print(f"  - all_cl_cross shape: {all_cl_cross.shape}")
                return True
            except FileNotFoundError as e:
                print(f"⚠ Data file not found for TM{inst}: {e}")
                return False
            except Exception as e:
                print(f"✗ Error loading TM{inst}: {e}")
                import traceback
                traceback.print_exc()
                return False
    except Exception as e:
        print(f"✗ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """Test that the refactored model can be created."""
    try:
        from ciber.theory.cross_ps_parametric_model import CrossPowerSpectrumModel
        print("✓ Importing CrossPowerSpectrumModel...")
        
        # Create a test model
        lb_test = np.array([100., 200., 500., 1000., 2000., 5000., 10000.])
        model = CrossPowerSpectrumModel(
            lb=lb_test,
            use_powerlaw_2h=True,
            use_astrometry_damping=True,
            use_one_halo=False,
            use_two_halo=False,
        )
        print("✓ Created CrossPowerSpectrumModel with Poisson-only config")
        
        # Test model evaluation
        params = np.array([0.0, 0.0, 0.0, 0.0, 1e-4, 2.0])
        dl = model.model_dl(lb_test, *params)
        print(f"✓ Model evaluation successful")
        print(f"  - Output shape: {dl.shape}")
        print(f"  - Sample values: {dl[:3]}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_function_imports():
    """Test that the refactored function imports correctly."""
    try:
        from scripts.generate_gal_cross_paper_figures import (
            fit_gaia_cross_poisson_damping,
            run_gaia_cross,
            _load_gaia_cross_fit,
        )
        print("✓ All functions imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Gaia Stellar Density Fit Refactoring")
    print("=" * 60)
    
    all_pass = True
    
    print("\n1. Testing function imports...")
    all_pass &= test_function_imports()
    
    print("\n2. Testing model creation...")
    all_pass &= test_model_creation()
    
    print("\n3. Testing data loading...")
    all_pass &= test_data_loading()
    
    print("\n" + "=" * 60)
    if all_pass:
        print("✓ All validation tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed - see above for details")
        sys.exit(1)
