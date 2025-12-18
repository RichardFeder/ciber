# CIBER Repository Reorganization Proposal

## Current Issues
1. **Flat structure**: All ~80 Python files and notebooks in root directory
2. **Unclear dependencies**: Heavy use of `from X import *` makes dependency tracking difficult
3. **Mixed concerns**: Mocks, data processing, analysis, and utilities all intermixed
4. **Duplicate/deprecated code**: Scripts in `scripts/depr/` suggest cleanup needed
5. **Tight coupling**: Many circular-ish dependencies between modules

## Analysis Summary

### Core Module Categories Identified:

1. **Power Spectrum Pipeline** (Core data processing)
   - `ciber_powerspec_pipeline.py` (3000+ lines - main pipeline class)
   - `powerspec_utils.py` (utilities for PS calculations)
   - `ps_pipeline_go.py` (pipeline execution/wrappers)
   - `ps_tests.py` (pipeline testing)

2. **Mock & Simulation Generation**
   - `ciber_mocks.py` (CIB mock generation)
   - `mock_galaxy_catalogs.py` (galaxy catalog mocks)
   - `grigory_gal_mocks.py` (specific mock implementation)
   - `proc_jmocks.py` (process J-band mocks)
   - `lognormal_counts.py` (lognormal field generation)
   - `mock_gal_gross.py` (galaxy mocks)

3. **Data File I/O & Utilities**
   - `ciber_data_file_utils.py` (CIBER-specific file loading)
   - `catalog_utils.py` (catalog operations)
   - `lens_data_file_utils.py` (lensing data I/O)

4. **Pseudo-Cl Analysis (Mkk)**
   - `mkk_parallel.py` (mode coupling matrix computation)
   - `mkk_diagnostics.py` (Mkk validation)
   - `mkk_torch_dev.py` (GPU acceleration attempts)
   - `mkk_wrappers.py` (higher-level Mkk functions)

5. **Masking & Source Extraction**
   - `masking_utils.py` (mask generation/manipulation)
   - `mask_source_classification.py` (ML-based source classification)
   - `ciber_source_mask_construction_pipeline.py` (automated masking)

6. **Instrument Characterization**
   - `ciber_beam.py` (PSF/beam functions)
   - `ciber_noise_data_utils.py` (noise characterization)
   - `noise_model.py` (noise modeling)
   - `readnoise_realization.py` (read noise simulations)
   - `flat_field_est.py` (flat field estimation)
   - `frame_proc_filter.py` (frame processing)

7. **Cross-Correlation Analysis**
   - `galaxy_cross.py` (CIBER × galaxy cross-correlation)
   - `cross_spectrum.py` (general cross-spectrum analysis)
   - `spitzer_auto_cross.py` (Spitzer cross-correlations)
   - `angular_2pcf.py` (2-point correlation functions)

8. **CMB Lensing (FlatSkyQE)**
   - `FlatSkyQE/` subdirectory (already organized!)
   - Integration with CIBER for lensing reconstruction

9. **Theory & Predictions**
   - `cl_predictions.py` (theoretical C_ℓ predictions)
   - `halo_model.py` (halo model calculations)
   - `helgason.py` (EBL/CIB models)
   - `integrate_cl_wtheta.py` (Cl ↔ w(θ) transforms)

10. **Calibration & Processing**
    - `ciber_sb_calibration_tools.py` (surface brightness calibration)
    - `fourier_bkg_modl_ciber.py` (Fourier-space background modeling)
    - `filtering_utils.py` (filtering operations)

11. **Plotting & Visualization**
    - `plotting_fns.py` (general plotting utilities)
    - `gal_plotting_fns.py` (galaxy-specific plots)
    - `kappa_plotting_fns.py` (lensing-specific plots)

12. **External Data Processing**
    - `wise_coadd_proc.py` (WISE data processing)
    - `photo_z_analysis.py` (photo-z analysis)

13. **Configuration**
    - `config.py` (global configuration, paths)

## Proposed Directory Structure

```
ciber/
├── README.md
├── setup.py  # New: make package installable
├── environment.yml  # New: conda environment spec
├── config.py  # Keep at root for easy import
│
├── ciber/  # Main package directory
│   ├── __init__.py
│   │
│   ├── core/  # Core pipeline functionality
│   │   ├── __init__.py
│   │   ├── powerspec_pipeline.py  # Renamed from ciber_powerspec_pipeline.py
│   │   ├── powerspec_utils.py
│   │   ├── pipeline_runner.py  # Renamed from ps_pipeline_go.py
│   │   └── pipeline_tests.py  # Renamed from ps_tests.py
│   │
│   ├── instrument/  # Instrument-specific characterization
│   │   ├── __init__.py
│   │   ├── beam.py  # Renamed from ciber_beam.py
│   │   ├── noise_model.py
│   │   ├── noise_data_utils.py  # Renamed from ciber_noise_data_utils.py
│   │   ├── readnoise.py  # Renamed from readnoise_realization.py
│   │   ├── flat_field.py  # Renamed from flat_field_est.py
│   │   ├── calibration.py  # Renamed from ciber_sb_calibration_tools.py
│   │   └── frame_processing.py  # Renamed from frame_proc_filter.py
│   │
│   ├── mocks/  # Simulation and mock generation
│   │   ├── __init__.py
│   │   ├── cib_mocks.py  # Renamed from ciber_mocks.py
│   │   ├── galaxy_catalogs.py  # Renamed from mock_galaxy_catalogs.py
│   │   ├── grigory_mocks.py  # Renamed from grigory_gal_mocks.py
│   │   ├── lognormal.py  # Renamed from lognormal_counts.py
│   │   └── j_band_proc.py  # Renamed from proc_jmocks.py
│   │
│   ├── pseudo_cl/  # Pseudo-Cl analysis (mode coupling)
│   │   ├── __init__.py
│   │   ├── mkk_compute.py  # Renamed from mkk_parallel.py
│   │   ├── mkk_diagnostics.py
│   │   ├── mkk_wrappers.py
│   │   └── mkk_torch.py  # Renamed from mkk_torch_dev.py
│   │
│   ├── masking/  # Source masking and classification
│   │   ├── __init__.py
│   │   ├── mask_utils.py  # Renamed from masking_utils.py
│   │   ├── source_classification.py  # Renamed from mask_source_classification.py
│   │   └── mask_pipeline.py  # Renamed from ciber_source_mask_construction_pipeline.py
│   │
│   ├── cross_correlation/  # Cross-correlation analysis
│   │   ├── __init__.py
│   │   ├── galaxy_cross.py
│   │   ├── cross_spectrum.py
│   │   ├── spitzer_cross.py  # Renamed from spitzer_auto_cross.py
│   │   └── angular_corr.py  # Renamed from angular_2pcf.py
│   │
│   ├── lensing/  # CMB lensing-specific (keep FlatSkyQE separate)
│   │   ├── __init__.py
│   │   ├── prep_ciber_data.py  # Renamed from prep_ciber_dat_lens.py
│   │   └── ebl_tomography.py  # Renamed from ebl_tom.py
│   │
│   ├── theory/  # Theoretical predictions
│   │   ├── __init__.py
│   │   ├── cl_predictions.py
│   │   ├── halo_model.py
│   │   ├── helgason_model.py  # Renamed from helgason.py
│   │   └── cl_wtheta.py  # Renamed from integrate_cl_wtheta.py
│   │
│   ├── processing/  # Data processing utilities
│   │   ├── __init__.py
│   │   ├── filtering.py  # Renamed from filtering_utils.py
│   │   ├── fourier_bkg.py  # Renamed from fourier_bkg_modl_ciber.py
│   │   └── numerical.py  # Renamed from numerical_routines.py
│   │
│   ├── io/  # Data I/O
│   │   ├── __init__.py
│   │   ├── ciber_data_utils.py  # Renamed from ciber_data_file_utils.py
│   │   ├── catalog_utils.py
│   │   └── lens_data_utils.py  # Renamed from lens_data_file_utils.py
│   │
│   ├── external/  # External dataset processing
│   │   ├── __init__.py
│   │   ├── wise_processing.py  # Renamed from wise_coadd_proc.py
│   │   └── photo_z.py  # Renamed from photo_z_analysis.py
│   │
│   └── plotting/  # Visualization
│       ├── __init__.py
│       ├── plot_utils.py  # Renamed from plotting_fns.py
│       ├── galaxy_plots.py  # Renamed from gal_plotting_fns.py
│       └── lensing_plots.py  # Renamed from kappa_plotting_fns.py
│
├── scripts/  # Analysis scripts and executables
│   ├── run_pipeline.py  # Main pipeline runner
│   ├── generate_mocks.py
│   ├── compute_mkk.py
│   ├── cross_correlation_analysis.py
│   └── deprecated/  # Archive old scripts
│       └── ...
│
├── notebooks/  # Jupyter notebooks (already exists)
│   ├── analysis/
│   │   ├── power_spectrum_analysis.ipynb
│   │   ├── galaxy_cross_analysis.ipynb
│   │   └── ...
│   ├── mocks/
│   │   ├── ciber_mocks.ipynb
│   │   ├── grigory_gal_mocks.ipynb
│   │   └── ...
│   ├── calibration/
│   │   ├── noise_model.ipynb
│   │   ├── flat_field.ipynb
│   │   └── ...
│   └── lensing/
│       ├── ebl_tomography.ipynb
│       └── ...
│
├── FlatSkyQE/  # Keep as separate submodule
│   └── ...
│
├── LensQuEst/  # Keep as separate submodule
│   └── ...
│
├── tests/  # New: unit tests
│   ├── __init__.py
│   ├── test_powerspec.py
│   ├── test_mocks.py
│   ├── test_mkk.py
│   └── ...
│
├── data/  # Data directory (already exists)
│   └── ...
│
└── docs/  # New: documentation
    ├── api/
    ├── tutorials/
    └── pipeline_guide.md
```

## Key Dependency Reduction Strategies

### 1. Replace Wildcard Imports
Change from:
```python
from ciber_powerspec_pipeline import *
from plotting_fns import *
```

To:
```python
from ciber.core import powerspec_pipeline as cbps
from ciber.plotting import plot_utils
```

### 2. Create Clear Module Interfaces
Each `__init__.py` should expose only public API:
```python
# ciber/core/__init__.py
from .powerspec_pipeline import CIBER_PS_pipeline
from .powerspec_utils import compute_cl, azimuthal_average

__all__ = ['CIBER_PS_pipeline', 'compute_cl', 'azimuthal_average']
```

### 3. Break Circular Dependencies
- `config.py` stays at root (configuration layer)
- Core modules should not import from analysis modules
- Flow: config → io → core → analysis

### 4. Consolidate Duplicate Functionality
Multiple files have similar functions:
- `compute_cl` appears in multiple places → consolidate to `core/powerspec_utils.py`
- Beam correction functions scattered → consolidate to `instrument/beam.py`
- Mkk computation split across files → unify in `pseudo_cl/`

## Migration Strategy

### Phase 1: Preparation (No Breaking Changes)
1. Create new directory structure (empty)
2. Add `__init__.py` files
3. Set up `setup.py` for package installation
4. Write tests for critical functions

### Phase 2: Module Migration (One at a time)
1. Start with leaf dependencies (no imports from other local modules)
   - `numerical_routines.py` → `ciber/processing/numerical.py`
   - `plotting_fns.py` → `ciber/plotting/plot_utils.py`
2. Move to mid-level modules
   - `powerspec_utils.py` → `ciber/core/powerspec_utils.py`
   - `ciber_beam.py` → `ciber/instrument/beam.py`
3. Finally move high-level modules
   - `ciber_powerspec_pipeline.py` → `ciber/core/powerspec_pipeline.py`

### Phase 3: Update Imports
1. Create compatibility shim at root:
```python
# ciber_powerspec_pipeline.py (compatibility)
import warnings
warnings.warn("Direct import deprecated. Use: from ciber.core import powerspec_pipeline", 
              DeprecationWarning)
from ciber.core.powerspec_pipeline import *
```
2. Update notebooks to use new imports
3. Remove compatibility shims after notebooks updated

### Phase 4: Cleanup
1. Move deprecated scripts to `scripts/deprecated/`
2. Remove unused functions
3. Consolidate duplicate code
4. Update documentation

## Benefits

1. **Clearer Organization**: Related code grouped together
2. **Easier Navigation**: Find files by functionality
3. **Better Testing**: Isolated modules easier to test
4. **Reduced Dependencies**: Explicit imports show true dependencies
5. **Onboarding**: New users can understand structure
6. **Maintenance**: Changes isolated to relevant modules
7. **Reusability**: Clean modules can be used independently

## Dependency Graph (Simplified)

```
config
  ↓
io (data loading)
  ↓
instrument (beam, noise)  →  processing (filtering, numerical)
  ↓                             ↓
mocks  →  core (powerspec)  →  pseudo_cl (Mkk)
  ↓           ↓                    ↓
theory   cross_correlation  →  lensing
  ↓           ↓
     plotting
```

## Recommended Next Steps

1. **Review this proposal** - Adjust categories/structure as needed
2. **Pilot migration** - Move 1-2 simple modules to test workflow
3. **Update 1-2 notebooks** - Verify new imports work
4. **Create migration script** - Automate bulk of the work
5. **Full migration** - Move all modules systematically
6. **Documentation** - Update README, add API docs

## Files Requiring Special Attention

### High-Priority (Many dependencies)
- `ciber_powerspec_pipeline.py` (3000+ lines, core functionality)
- `ps_pipeline_go.py` (many imports)
- `plotting_fns.py` (widely imported)
- `config.py` (imported everywhere)

### Medium-Priority (Moderate complexity)
- `cross_spectrum.py` (complex analysis)
- `mkk_parallel.py` (performance-critical)
- `ciber_mocks.py` (simulation foundation)

### Low-Priority (Simpler/less used)
- Utility scripts
- Plotting helpers
- Deprecated code

## Questions for Discussion

1. Should we keep both old and new import paths temporarily?
2. How to handle notebooks during migration?
3. Should FlatSkyQE be better integrated or stay separate?
4. Any additional categorization needed?
5. Priority order for migration?
