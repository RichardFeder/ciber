# CIBER Package Migration Guide

This document tracks the migration of files from the flat structure to the new organized structure.

## Migration Status

### Phase 1: Preparation ✅ COMPLETE
- [x] Created directory structure
- [x] Added __init__.py files
- [x] Set up setup.py
- [x] Created test framework
- [x] Documented structure

### Phase 2: Module Migration 🔄 IN PROGRESS

#### Processing (Leaf Dependencies - Start Here)
- [ ] `numerical_routines.py` → `ciber/processing/numerical.py`
- [ ] `filtering_utils.py` → `ciber/processing/filtering.py`
- [ ] `fourier_bkg_modl_ciber.py` → `ciber/processing/fourier_bkg.py`

#### I/O
- [ ] `catalog_utils.py` → `ciber/io/catalog_utils.py`
- [ ] `ciber_data_file_utils.py` → `ciber/io/ciber_data_utils.py`

#### Plotting
- [ ] `plotting_fns.py` → `ciber/plotting/plot_utils.py`
- [ ] `gal_plotting_fns.py` → `ciber/plotting/galaxy_plots.py`

#### Instrument
- [ ] `ciber_beam.py` → `ciber/instrument/beam.py`
- [ ] `noise_model.py` → `ciber/instrument/noise_model.py`
- [ ] `ciber_noise_data_utils.py` → `ciber/instrument/noise_data_utils.py`
- [ ] `readnoise_realization.py` → `ciber/instrument/readnoise.py`
- [ ] `flat_field_est.py` → `ciber/instrument/flat_field.py`
- [ ] `ciber_sb_calibration_tools.py` → `ciber/instrument/calibration.py`
- [ ] `frame_proc_filter.py` → `ciber/instrument/frame_processing.py`

#### Theory
- [ ] `helgason.py` → `ciber/theory/helgason_model.py`
- [ ] `halo_model.py` → `ciber/theory/halo_model.py`
- [ ] `integrate_cl_wtheta.py` → `ciber/theory/cl_wtheta.py`
- [ ] `cl_predictions.py` → `ciber/theory/cl_predictions.py`

#### Mocks
- [ ] `ciber_mocks.py` → `ciber/mocks/cib_mocks.py`
- [ ] `mock_galaxy_catalogs.py` → `ciber/mocks/galaxy_catalogs.py`
- [ ] `grigory_gal_mocks.py` → `ciber/mocks/grigory_mocks.py`
- [ ] `lognormal_counts.py` → `ciber/mocks/lognormal.py`
- [ ] `mock_gal_gross.py` → `ciber/mocks/mock_gal_gross.py`
- [ ] `proc_jmocks.py` → `ciber/mocks/j_band_proc.py`

#### Pseudo-Cl
- [ ] `mkk_parallel.py` → `ciber/pseudo_cl/mkk_compute.py`
- [ ] `mkk_diagnostics.py` → `ciber/pseudo_cl/mkk_diagnostics.py`
- [ ] `mkk_wrappers.py` → `ciber/pseudo_cl/mkk_wrappers.py`
- [ ] `mkk_torch_dev.py` → `ciber/pseudo_cl/mkk_torch.py`

#### Masking
- [ ] `masking_utils.py` → `ciber/masking/mask_utils.py`
- [ ] `mask_source_classification.py` → `ciber/masking/source_classification.py`
- [ ] `ciber_source_mask_construction_pipeline.py` → `ciber/masking/mask_pipeline.py`

#### Core (High Priority)
- [ ] `powerspec_utils.py` → `ciber/core/powerspec_utils.py`
- [ ] `ciber_powerspec_pipeline.py` → `ciber/core/powerspec_pipeline.py`
- [ ] `ps_pipeline_go.py` → `ciber/core/pipeline_runner.py`
- [ ] `ps_tests.py` → `ciber/core/pipeline_tests.py`

#### Cross-Correlation
- [ ] `galaxy_cross.py` → `ciber/cross_correlation/galaxy_cross.py`
- [ ] `cross_spectrum.py` → `ciber/cross_correlation/cross_spectrum.py`
- [ ] `spitzer_auto_cross.py` → `ciber/cross_correlation/spitzer_cross.py`
- [ ] `angular_2pcf.py` → `ciber/cross_correlation/angular_corr.py`
- [ ] `ebl_tom.py` → `ciber/cross_correlation/ebl_tomography.py`
- [ ] `ebl_tom_min.py` → `ciber/cross_correlation/ebl_tom_min.py` (or merge)

#### Lensing
- [ ] Integration with FlatSkyQE (already in FlatSkyQE/ subdirectory)

#### External
- [ ] `wise_coadd_proc.py` → `ciber/external/wise_processing.py`
- [ ] `photo_z_analysis.py` → `ciber/external/photo_z.py`

### Phase 3: Update Imports ⏳ PENDING
- [ ] Create compatibility shims in root directory
- [ ] Update import statements in migrated modules
- [ ] Test all imports work

### Phase 4: Cleanup ⏳ PENDING
- [ ] Remove compatibility shims
- [ ] Move deprecated scripts
- [ ] Update documentation
- [ ] Final testing

## Migration Commands

### For each file:
```bash
# 1. Copy file to new location
cp old_file.py ciber/module/new_name.py

# 2. Update imports in the new file
# Change: from old_module import *
# To: from ciber.other_module import specific_function

# 3. Update __init__.py
# Add: from .new_name import *

# 4. Test the new module
python -c "from ciber.module import new_name"

# 5. Create compatibility shim (temporary)
# In root: echo "from ciber.module.new_name import *" > old_file.py
```

## Testing Strategy

After each migration batch:
1. Run `pytest tests/`
2. Try importing in Python REPL
3. Run a simple analysis notebook
4. Check for missing dependencies

## Notes

- Start with leaf dependencies (no local imports)
- Test after each migration
- Keep compatibility shims until all imports updated
- Document any issues in this file
