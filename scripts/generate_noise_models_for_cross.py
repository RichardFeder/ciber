#!/usr/bin/env python3
"""Generate noise model files for use in cross-spectrum pipeline.

This script computes noise bias files (Fourier weights + 2D/1D power spectra)
for each instrument and field, using the same masking/filtering as the cross-spectrum.
These are then used by ciber_gal_cross() to compute in-situ CIBER auto-spectra.
"""

import numpy as np
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from ciber.core.powerspec_pipeline import CIBER_PS_pipeline, return_default_cbps_dicts
from ciber.core.ps_pipeline_go import run_cbps_pipeline

def generate_noise_models_observed(inst_list=[1, 2],
                                  mag_lim_list=[16.0, 15.5],
                                  datestr='111323',
                                  ifield_list=[4, 5, 6, 7, 8]):
    """
    Generate noise model files for observed data using gradient filtering + quadoff.

    Parameters
    ----------
    inst_list : list
        Instrument indices [1, 2]
    mag_lim_list : list
        Magnitude limits for each instrument
    datestr : str
        Date string for file organization (default: '111323')
    ifield_list : list
        Field indices to process
    """

    bandstr_list = ['J', 'H']

    # Create output directory
    noise_models_dir = Path(config.ciber_basepath) / 'data' / 'noise_models_sim' / datestr
    noise_models_dir.mkdir(parents=True, exist_ok=True)

    for inst, bandstr, mag_lim in zip(inst_list, bandstr_list, mag_lim_list):
        print(f"\n{'='*80}")
        print(f"Generating noise models for TM{inst} ({bandstr})")
        print(f"{'='*80}")

        # Set up run name matching cross-spectrum pipeline convention
        run_name = f"observed_{bandstr}lt{mag_lim}_072424_quadoff_grad_fcsub_order2"

        # Initialize pipeline
        cbps = CIBER_PS_pipeline()

        # Get default configuration dicts
        config_dict, pscb_dict, float_param_dict, fpath_dict = return_default_cbps_dicts()

        # Configure for noise model computation
        pscb_dict['save'] = True  # Force save
        pscb_dict['ff_estimate_correct'] = False  # Skip flat-field estimation
        pscb_dict['load_noise_bias'] = False  # Compute fresh noise models

        # Noise computation parameters
        float_param_dict['n_FW_sims'] = 500  # Number of noise simulations
        float_param_dict['n_FW_split'] = 10  # Splits for error estimation

        # Set paths
        fpath_dict['noisemodl_basepath'] = str(noise_models_dir / f'TM{inst}' / '')
        fpath_dict['noisemodl_run_name'] = run_name

        # Masking parameters matching cross-spectrum pipeline
        mask_tail = 'ilt25.0' if inst == 1 else None

        print(f"Run name: {run_name}")
        print(f"Output directory: {fpath_dict['noisemodl_basepath']}")

        try:
            # Run pipeline to compute and save noise models
            lb, signal_power_spectra, \
                recovered_power_spectra, recovered_dcl, nls_estFF_nofluc,\
                cls_inter, inter_labels, ff_estimates, final_masked_images = run_cbps_pipeline(
                cbps, inst, nsims=1, run_name=run_name,
                ifield_list=ifield_list,
                datestr=datestr,
                show_plots=False,
                mask_tail=mask_tail,
                mask_tail_ffest=None,
                mkk_base_path=None,  # Use defaults
                mkk_ffest_base_path=None,
                noisemodl_basepath=fpath_dict['noisemodl_basepath'],
                noisemodl_run_name=fpath_dict['noisemodl_run_name'],
                load_noise_bias=False,
                save=True,
                verbose=True,
                per_quadrant=False,
                quadoff_grad=False,
                save_fourier_planes=False,
                mkk_type=None,
                fc_sub=True,
                fc_sub_quad_offset=True,
                fc_sub_n_terms=2,
                compute_mkk_pinv=False,
                nitermax=10,
                **pscb_dict,
                **float_param_dict,
            )

            print(f"✓ Noise models generated for TM{inst}")

            # Verify files were created
            tm_dir = noise_models_dir / f'TM{inst}' / run_name
            if tm_dir.exists():
                files = list(tm_dir.glob('noise_bias_fieldidx*.npz'))
                print(f"  Files created: {len(files)}")
                for f in sorted(files)[:3]:
                    print(f"    - {f.name}")

        except Exception as e:
            print(f"✗ Error generating noise models for TM{inst}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("Noise model generation complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    generate_noise_models_observed()
