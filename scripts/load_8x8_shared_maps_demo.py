#!/usr/bin/env python3
"""Simple loader demo for shared 8x8 mock map products.

Reads intensity and galaxy files for HSC/SDSS samples and prints basic
sanity stats so a collaborator can verify they are using the same inputs.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List

import numpy as np


def _sample_tag(sample: str) -> str:
    if sample == "hsc":
        return "hsc_i_lt_25.0_CIBERfidmask"
    if sample == "sdss":
        return "sdss_z_lt_22.0_CIBERfidmask"
    raise ValueError(f"Unsupported sample: {sample}")


def _collect_files(base_dir: str, tm: int, sample: str) -> Dict[str, List[str]]:
    tag = _sample_tag(sample)

    intensity_glob = os.path.join(
        base_dir,
        f"mock_maps/intensity/TM{tm}/rlz1_TM{tm}_{tag}_zmin=*_*_pred_tile000_8.0deg_intensity.npz",
    )
    galaxy_glob = os.path.join(
        base_dir,
        f"mock_maps/galaxy/TM{tm}/rlz1_TM{tm}_{tag}_zmin=*_*_tile000_8.0deg_galaxy.npz",
    )

    return {
        "intensity": sorted(glob.glob(intensity_glob)),
        "galaxy": sorted(glob.glob(galaxy_glob)),
    }


def _summarize_pair(intensity_path: str, galaxy_path: str) -> None:
    i_dat = np.load(intensity_path, allow_pickle=True)
    g_dat = np.load(galaxy_path, allow_pickle=True)

    ciber_map = i_dat["ciber_map"]
    gal_counts = g_dat["gal_counts"]
    tracer_x = g_dat["tracer_x"]

    print(f"  intensity: {os.path.basename(intensity_path)}")
    print(f"    ciber_map shape={ciber_map.shape}, mean={np.mean(ciber_map):.6g}, std={np.std(ciber_map):.6g}")
    print(f"  galaxy:    {os.path.basename(galaxy_path)}")
    print(f"    gal_counts shape={gal_counts.shape}, total_counts={np.sum(gal_counts):.6g}, n_tracers={len(tracer_x)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load shared 8x8 map products and print quick stats")
    parser.add_argument(
        "--base-dir",
        default="/Users/richardfeder/Documents/ciber/data/jordan_mocks/v3_boxed_outputs/tiles_8p0deg",
        help="Base directory containing mock_maps",
    )
    parser.add_argument(
        "--sample",
        choices=["hsc", "sdss"],
        default="hsc",
        help="Sample selection",
    )
    parser.add_argument(
        "--tm",
        type=int,
        choices=[1, 2],
        default=1,
        help="Instrument/TM index",
    )
    args = parser.parse_args()

    files = _collect_files(args.base_dir, args.tm, args.sample)
    n_int = len(files["intensity"])
    n_gal = len(files["galaxy"])

    print(f"base_dir={args.base_dir}")
    print(f"sample={args.sample}, TM{args.tm}")
    print(f"found intensity files={n_int}, galaxy files={n_gal}")

    if n_int == 0 or n_gal == 0:
        raise RuntimeError("No files found for the requested selection")
    if n_int != n_gal:
        raise RuntimeError("Intensity/Galaxy file count mismatch")

    for intensity_path, galaxy_path in zip(files["intensity"], files["galaxy"]):
        _summarize_pair(intensity_path, galaxy_path)


if __name__ == "__main__":
    main()
