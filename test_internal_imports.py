#!/usr/bin/env python3
"""
Test script to verify all ciber module imports after internal import updates.
This tests that modules can import each other using the new ciber.* structure.
"""

import sys
import importlib

def test_module_import(module_name, description=""):
    """Test if a module can be imported successfully."""
    try:
        importlib.import_module(module_name)
        print(f"✓ {module_name:50s} {description}")
        return True
    except Exception as e:
        print(f"✗ {module_name:50s} FAILED: {str(e)[:80]}")
        return False

def main():
    print("Testing ciber package imports with updated internal references...\n")
    
    results = []
    
    # Test main package
    print("=" * 80)
    print("MAIN PACKAGE")
    print("=" * 80)
    results.append(test_module_import("ciber", "Main package"))
    print()
    
    # Test core modules
    print("=" * 80)
    print("CORE MODULES")
    print("=" * 80)
    results.append(test_module_import("ciber.core", "Core module init"))
    results.append(test_module_import("ciber.core.powerspec_pipeline", "Power spectrum pipeline"))
    results.append(test_module_import("ciber.core.powerspec_utils", "Power spectrum utilities"))
    results.append(test_module_import("ciber.core.ps_pipeline_go", "Pipeline runner"))
    results.append(test_module_import("ciber.core.ps_tests", "Pipeline tests"))
    print()
    
    # Test instrument modules
    print("=" * 80)
    print("INSTRUMENT MODULES")
    print("=" * 80)
    results.append(test_module_import("ciber.instrument", "Instrument module init"))
    results.append(test_module_import("ciber.instrument.beam", "Beam model"))
    results.append(test_module_import("ciber.instrument.calibration", "Calibration tools"))
    results.append(test_module_import("ciber.instrument.flat_field", "Flat field estimation"))
    results.append(test_module_import("ciber.instrument.frame_processing", "Frame processing"))
    results.append(test_module_import("ciber.instrument.noise_data_utils", "Noise data utilities"))
    results.append(test_module_import("ciber.instrument.noise_model", "Noise model"))
    results.append(test_module_import("ciber.instrument.readnoise", "Read noise"))
    print()
    
    # Test mocks modules
    print("=" * 80)
    print("MOCKS MODULES")
    print("=" * 80)
    results.append(test_module_import("ciber.mocks", "Mocks module init"))
    results.append(test_module_import("ciber.mocks.cib_mocks", "CIB mocks"))
    results.append(test_module_import("ciber.mocks.galaxy_catalogs", "Galaxy catalogs"))
    results.append(test_module_import("ciber.mocks.grigory_mocks", "Grigory mocks"))
    results.append(test_module_import("ciber.mocks.j_band_proc", "J-band processing"))
    results.append(test_module_import("ciber.mocks.lognormal", "Lognormal mocks"))
    results.append(test_module_import("ciber.mocks.mock_gal_gross", "Mock galaxy gross"))
    print()
    
    # Test pseudo_cl modules
    print("=" * 80)
    print("PSEUDO-CL MODULES")
    print("=" * 80)
    results.append(test_module_import("ciber.pseudo_cl", "Pseudo-Cl module init"))
    results.append(test_module_import("ciber.pseudo_cl.mkk_compute", "Mkk computation"))
    results.append(test_module_import("ciber.pseudo_cl.mkk_diagnostics", "Mkk diagnostics"))
    results.append(test_module_import("ciber.pseudo_cl.mkk_torch", "Mkk torch"))
    results.append(test_module_import("ciber.pseudo_cl.mkk_wrappers", "Mkk wrappers"))
    print()
    
    # Test masking modules
    print("=" * 80)
    print("MASKING MODULES")
    print("=" * 80)
    results.append(test_module_import("ciber.masking", "Masking module init"))
    results.append(test_module_import("ciber.masking.mask_utils", "Mask utilities"))
    results.append(test_module_import("ciber.masking.source_classification", "Source classification"))
    results.append(test_module_import("ciber.masking.mask_pipeline", "Mask pipeline"))
    print()
    
    # Test cross_correlation modules
    print("=" * 80)
    print("CROSS-CORRELATION MODULES")
    print("=" * 80)
    results.append(test_module_import("ciber.cross_correlation", "Cross-correlation module init"))
    results.append(test_module_import("ciber.cross_correlation.angular_corr", "Angular correlation"))
    results.append(test_module_import("ciber.cross_correlation.cross_spectrum", "Cross spectrum"))
    results.append(test_module_import("ciber.cross_correlation.ebl_tom_min", "EBL tomography min"))
    results.append(test_module_import("ciber.cross_correlation.ebl_tomography", "EBL tomography"))
    results.append(test_module_import("ciber.cross_correlation.galaxy_cross", "Galaxy cross"))
    results.append(test_module_import("ciber.cross_correlation.spitzer_cross", "Spitzer cross"))
    print()
    
    # Test theory modules
    print("=" * 80)
    print("THEORY MODULES")
    print("=" * 80)
    results.append(test_module_import("ciber.theory", "Theory module init"))
    results.append(test_module_import("ciber.theory.cl_predictions", "Cl predictions"))
    results.append(test_module_import("ciber.theory.cl_wtheta", "Cl to w(theta)"))
    results.append(test_module_import("ciber.theory.halo_model", "Halo model"))
    results.append(test_module_import("ciber.theory.helgason_model", "Helgason model"))
    print()
    
    # Test processing modules
    print("=" * 80)
    print("PROCESSING MODULES")
    print("=" * 80)
    results.append(test_module_import("ciber.processing", "Processing module init"))
    results.append(test_module_import("ciber.processing.filtering", "Filtering"))
    results.append(test_module_import("ciber.processing.fourier_bkg", "Fourier background"))
    results.append(test_module_import("ciber.processing.numerical", "Numerical routines"))
    print()
    
    # Test io modules
    print("=" * 80)
    print("I/O MODULES")
    print("=" * 80)
    results.append(test_module_import("ciber.io", "I/O module init"))
    results.append(test_module_import("ciber.io.catalog_utils", "Catalog utilities"))
    results.append(test_module_import("ciber.io.ciber_data_utils", "CIBER data utilities"))
    print()
    
    # Test external modules
    print("=" * 80)
    print("EXTERNAL MODULES")
    print("=" * 80)
    results.append(test_module_import("ciber.external", "External module init"))
    results.append(test_module_import("ciber.external.photo_z", "Photo-z analysis"))
    results.append(test_module_import("ciber.external.wise_processing", "WISE processing"))
    print()
    
    # Test plotting modules
    print("=" * 80)
    print("PLOTTING MODULES")
    print("=" * 80)
    results.append(test_module_import("ciber.plotting", "Plotting module init"))
    results.append(test_module_import("ciber.plotting.galaxy_plots", "Galaxy plots"))
    results.append(test_module_import("ciber.plotting.plot_utils", "Plot utilities"))
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n✓ All imports successful!")
        return 0
    else:
        print(f"\n✗ {total - passed} imports failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
