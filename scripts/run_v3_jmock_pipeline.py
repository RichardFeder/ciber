#!/usr/bin/env python3
"""Stage-based runner for Jordan v3 mock processing.

This script discovers box-split redshift bins in data/jordan_mocks/v3/fov_10.0,
generates spatially complete tile definitions (2x2 deg by default, optional 5x5),
and runs `process_jmock` in stages so full-map generation can be reused.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from ciber.core.powerspec_pipeline import get_power_spec
from ciber.mocks.proc_jmocks import process_jmock


ZBIN_BOX_MAP = {
    (0.0, 0.2): "box_128",
    (0.2, 0.4): "box_256",
    (0.4, 0.6): "box_512",
    (0.6, 0.8): "box_512",
    (0.8, 1.0): "box_512",
}

DEFAULT_SAMPLES = {
    "legacy": {"galstr": "sdss_z", "m_min_gal": 14.0, "m_max_gal": 22.0},
    "hsc": {"galstr": "hsc_i", "m_min_gal": 14.0, "m_max_gal": 25.0},
}


@dataclass
class ZBinCatalog:
    box: str
    dim: str
    zmin: float
    zmax: float
    catalog_dir: str
    populations: List[int]


@dataclass
class TileDef:
    tile_size_deg: float
    tile_idx: int
    ra_min: float
    ra_max: float
    dec_min: float
    dec_max: float

    @property
    def label(self) -> str:
        return f"tile{self.tile_idx:03d}_{self.tile_size_deg:.1f}deg"

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (self.ra_min, self.ra_max, self.dec_min, self.dec_max)


def _parse_val(path_name: str, prefix: str) -> float:
    token = path_name.replace(prefix, "")
    return float(token)


def _extract_populations(catalog_dir: str) -> List[int]:
    pops = []
    for fpath in glob.glob(os.path.join(catalog_dir, "cat_ra_pop_*.fits")):
        m = re.search(r"cat_ra_pop_(\d+)\.fits$", os.path.basename(fpath))
        if m is not None:
            pops.append(int(m.group(1)))
    return sorted(set(pops))


def discover_v3_zbins(base_dir: str, rlz: int = 1) -> Tuple[List[ZBinCatalog], List[str]]:
    warnings: List[str] = []
    results: List[ZBinCatalog] = []

    for (zmin_expected, zmax_expected), box in ZBIN_BOX_MAP.items():
        box_dir = os.path.join(base_dir, box)
        if not os.path.isdir(box_dir):
            warnings.append(f"Missing box directory: {box_dir}")
            continue

        dim_dirs = sorted(glob.glob(os.path.join(box_dir, "dim_*")))
        if not dim_dirs:
            warnings.append(f"No dim_* directory found in {box_dir}")
            continue
        dim_dir = dim_dirs[0]

        rlz_dir = os.path.join(dim_dir, f"rlz_{rlz:03d}")
        if not os.path.isdir(rlz_dir):
            warnings.append(f"Missing realization directory: {rlz_dir}")
            continue

        zmin_dirs = sorted(glob.glob(os.path.join(rlz_dir, "zmin_*")))
        found_match = False
        for zmin_dir in zmin_dirs:
            zmin_val = _parse_val(os.path.basename(zmin_dir), "zmin_")
            if not np.isclose(zmin_val, zmin_expected):
                continue

            zmax_dirs = sorted(glob.glob(os.path.join(zmin_dir, "zmax_*")))
            for zmax_dir in zmax_dirs:
                zmax_val = _parse_val(os.path.basename(zmax_dir), "zmax_")
                if not np.isclose(zmax_val, zmax_expected):
                    continue

                cat_dir = os.path.join(zmax_dir, "m_10.00_15.00")
                if not os.path.isdir(cat_dir):
                    warnings.append(f"Missing m_10.00_15.00 catalog directory: {cat_dir}")
                    continue

                populations = _extract_populations(cat_dir)
                if not populations:
                    warnings.append(f"No cat_ra_pop_*.fits found in {cat_dir}")
                    continue

                results.append(
                    ZBinCatalog(
                        box=box,
                        dim=os.path.basename(dim_dir),
                        zmin=zmin_val,
                        zmax=zmax_val,
                        catalog_dir=cat_dir,
                        populations=populations,
                    )
                )
                found_match = True

        if not found_match:
            warnings.append(
                f"Could not find expected z-bin {zmin_expected:.1f}-{zmax_expected:.1f} in {rlz_dir}"
            )

    results = sorted(results, key=lambda x: (x.zmin, x.zmax))
    return results, warnings


def _load_ra_dec(catalog_dir: str, pop: int) -> Tuple[np.ndarray, np.ndarray]:
    ra = fits.open(os.path.join(catalog_dir, f"cat_ra_pop_{pop}.fits"))[1].data["ra"]
    dec = fits.open(os.path.join(catalog_dir, f"cat_dec_pop_{pop}.fits"))[1].data["dec"]
    return ra, dec


def build_complete_tiles(
    ra: np.ndarray,
    dec: np.ndarray,
    tile_size_deg: float,
) -> List[TileDef]:
    ra_min, ra_max = float(np.min(ra)), float(np.max(ra))
    dec_min, dec_max = float(np.min(dec)), float(np.max(dec))

    n_ra = int((ra_max - ra_min) // tile_size_deg)
    n_dec = int((dec_max - dec_min) // tile_size_deg)

    tiles: List[TileDef] = []
    tidx = 0
    for i in range(n_ra):
        for j in range(n_dec):
            cur_ra_min = ra_min + i * tile_size_deg
            cur_ra_max = cur_ra_min + tile_size_deg
            cur_dec_min = dec_min + j * tile_size_deg
            cur_dec_max = cur_dec_min + tile_size_deg
            if cur_ra_max <= ra_max and cur_dec_max <= dec_max:
                tiles.append(
                    TileDef(
                        tile_size_deg=tile_size_deg,
                        tile_idx=tidx,
                        ra_min=cur_ra_min,
                        ra_max=cur_ra_max,
                        dec_min=cur_dec_min,
                        dec_max=cur_dec_max,
                    )
                )
                tidx += 1
    return tiles


def build_centered_tile(ra: np.ndarray, dec: np.ndarray, tile_size_deg: float) -> TileDef:
    rac = float(np.mean(ra))
    decc = float(np.mean(dec))
    half = tile_size_deg / 2.0
    return TileDef(
        tile_size_deg=tile_size_deg,
        tile_idx=0,
        ra_min=rac - half,
        ra_max=rac + half,
        dec_min=decc - half,
        dec_max=decc + half,
    )


def _tile_quality_metrics(
    tiles: Sequence[TileDef],
    ra: np.ndarray,
    dec: np.ndarray,
    grid_n: int = 8,
) -> Dict[int, Dict[str, float]]:
    metrics: Dict[int, Dict[str, float]] = {}
    for tile in tiles:
        m = (
            (ra >= tile.ra_min)
            & (ra < tile.ra_max)
            & (dec >= tile.dec_min)
            & (dec < tile.dec_max)
        )
        n_obj = int(np.sum(m))
        if n_obj == 0:
            metrics[tile.tile_idx] = {"n_obj": 0, "coverage": 0.0, "score": 0.0}
            continue

        h, _, _ = np.histogram2d(
            ra[m],
            dec[m],
            bins=grid_n,
            range=[[tile.ra_min, tile.ra_max], [tile.dec_min, tile.dec_max]],
        )
        coverage = float(np.mean(h > 0))
        score = n_obj * coverage
        metrics[tile.tile_idx] = {"n_obj": n_obj, "coverage": coverage, "score": score}
    return metrics


def _select_complete_tiles(
    tiles: Sequence[TileDef],
    metrics: Dict[int, Dict[str, float]],
    min_objects_per_tile: int,
    min_coverage_frac: float,
    target_n_tiles: int,
) -> Tuple[List[TileDef], List[TileDef]]:
    complete = []
    rejected = []
    for tile in tiles:
        mm = metrics[tile.tile_idx]
        if mm["n_obj"] >= min_objects_per_tile and mm["coverage"] >= min_coverage_frac:
            complete.append(tile)
        else:
            rejected.append(tile)

    complete = sorted(complete, key=lambda t: metrics[t.tile_idx]["score"], reverse=True)
    if len(complete) < target_n_tiles:
        fallback = sorted(rejected, key=lambda t: metrics[t.tile_idx]["score"], reverse=True)
        need = target_n_tiles - len(complete)
        complete.extend(fallback[:need])
        rejected = [t for t in rejected if t not in fallback[:need]]

    return complete[:target_n_tiles], rejected


def _imdim_from_tile_size(tile_size_deg: float) -> int:
    # Keep 2 deg at 1024 px and scale linearly for larger maps.
    return int(round(1024.0 * (tile_size_deg / 2.0)))


def _tile_source_catalog(zbins: Sequence[ZBinCatalog], prefer_pop: int = 1) -> ZBinCatalog:
    for zbin in zbins:
        if prefer_pop in zbin.populations:
            return zbin
    return zbins[0]


def _run_maps_stage(
    zbins: Sequence[ZBinCatalog],
    tiles: Sequence[TileDef],
    rlz: int,
    inst_list: Sequence[int],
    jmock_outdir: str,
    ifield_map: int,
    masking_maglim: float,
    apply_map_mask: bool,
    save_magcut_diagnostics: bool,
    overwrite: bool = False,
) -> None:
    for inst in inst_list:
        os.makedirs(os.path.join(jmock_outdir, f"mock_maps/intensity/TM{inst}"), exist_ok=True)
        os.makedirs(os.path.join(jmock_outdir, f"mock_maps/galaxy/TM{inst}"), exist_ok=True)
        os.makedirs(os.path.join(jmock_outdir, f"mock_ps_pred/TM{inst}/indiv"), exist_ok=True)
        os.makedirs(os.path.join(jmock_outdir, f"mock_ps_pred/TM{inst}/field_average"), exist_ok=True)

    for zbin in zbins:
        for tile in tiles:
            for inst in inst_list:
                addstr = "fullmapcache"
                ps_pred = os.path.join(
                    jmock_outdir,
                    f"mock_ps_pred/TM{inst}/indiv/rlz{rlz}_TM{inst}_auto_cross_pred_{addstr}_zmin={zbin.zmin}_zmax={zbin.zmax}_{tile.label}.npz",
                )
                if (not overwrite) and os.path.exists(ps_pred):
                    continue

                process_jmock(
                    ciber_inst=inst,
                    mock_rlz=rlz,
                    pop_list=zbin.populations,
                    galstr="hsc_i",
                    regen_full_map=True,
                    m_min_gal=10.0,
                    m_max_gal=30.0,
                    masking_maglim=masking_maglim,
                    imdim=_imdim_from_tile_size(tile.tile_size_deg),
                    jmock_basedir=jmock_outdir,
                    jmock_catalog_dir=zbin.catalog_dir,
                    save_maps=True,
                    save_ps=True,
                    save_intensity=True,
                    save_galaxy=False,
                    redshift_min=zbin.zmin,
                    redshift_max=zbin.zmax,
                    save_counts=False,
                    addstr=addstr,
                    fov_deg=tile.tile_size_deg,
                    tile_bounds_deg=tile.bounds,
                    tile_label=tile.label,
                    ifield_map=ifield_map,
                    skip_missing_pop=True,
                    apply_map_mask=apply_map_mask,
                    save_diagnostic_figs=True,
                    diagnostic_dir=os.path.join(jmock_outdir, "diagnostics", f"TM{inst}"),
                    save_magcut_diagnostic_figs=save_magcut_diagnostics,
                    intensity_units='nW m^-2 sr^-1',
                    save_histograms=True,
                    interactive_plots=False,
                    save_ps_diagnostic_figs=True,
                    ps_figure_dir=os.path.join(jmock_outdir, "diagnostics", f"TM{inst}", "power_spectra"),
                )


def _run_selection_spectra_stage(
    zbins: Sequence[ZBinCatalog],
    tiles: Sequence[TileDef],
    rlz: int,
    inst_list: Sequence[int],
    jmock_outdir: str,
    ifield_map: int,
    masking_maglim: float,
    apply_map_mask: bool,
    save_magcut_diagnostics: bool,
    run_samples: Sequence[str],
    overwrite: bool = False,
) -> None:
    for inst in inst_list:
        os.makedirs(os.path.join(jmock_outdir, f"mock_maps/intensity/TM{inst}"), exist_ok=True)
        os.makedirs(os.path.join(jmock_outdir, f"mock_maps/galaxy/TM{inst}"), exist_ok=True)
        os.makedirs(os.path.join(jmock_outdir, f"mock_ps_pred/TM{inst}/indiv"), exist_ok=True)
        os.makedirs(os.path.join(jmock_outdir, f"mock_ps_pred/TM{inst}/field_average"), exist_ok=True)

    for sample_name in run_samples:
        sample_cfg = DEFAULT_SAMPLES[sample_name]
        for zbin in zbins:
            addstr = (
                f"{sample_cfg['galstr']}_lt_{sample_cfg['m_max_gal']:.1f}_CIBERfidmask"
            )
            for tile in tiles:
                for inst in inst_list:
                    ps_pred = os.path.join(
                        jmock_outdir,
                        f"mock_ps_pred/TM{inst}/indiv/rlz{rlz}_TM{inst}_auto_cross_pred_{addstr}_zmin={zbin.zmin}_zmax={zbin.zmax}_{tile.label}.npz",
                    )
                    if (not overwrite) and os.path.exists(ps_pred):
                        continue

                    process_jmock(
                        ciber_inst=inst,
                        mock_rlz=rlz,
                        pop_list=zbin.populations,
                        galstr=sample_cfg["galstr"],
                        regen_full_map=False,
                        m_min_gal=sample_cfg["m_min_gal"],
                        m_max_gal=sample_cfg["m_max_gal"],
                        masking_maglim=masking_maglim,
                        imdim=_imdim_from_tile_size(tile.tile_size_deg),
                        jmock_basedir=jmock_outdir,
                        jmock_catalog_dir=zbin.catalog_dir,
                        save_maps=True,
                        save_ps=True,
                        save_intensity=True,
                        save_galaxy=True,
                        redshift_min=zbin.zmin,
                        redshift_max=zbin.zmax,
                        save_counts=True,
                        addstr=addstr,
                        fov_deg=tile.tile_size_deg,
                        tile_bounds_deg=tile.bounds,
                        tile_label=tile.label,
                        ifield_map=ifield_map,
                        skip_missing_pop=True,
                        apply_map_mask=apply_map_mask,
                        save_diagnostic_figs=False,
                        diagnostic_dir=os.path.join(jmock_outdir, "diagnostics", f"TM{inst}"),
                        save_magcut_diagnostic_figs=save_magcut_diagnostics,
                        intensity_units='nW m^-2 sr^-1',
                        save_histograms=True,
                        interactive_plots=False,
                        save_ps_diagnostic_figs=True,
                        ps_figure_dir=os.path.join(jmock_outdir, "diagnostics", f"TM{inst}", "power_spectra"),
                    )

        _run_sample_zlt1_aggregate_stage(
            zbins=zbins,
            tiles=tiles,
            rlz=rlz,
            inst_list=inst_list,
            jmock_outdir=jmock_outdir,
            sample_cfg=sample_cfg,
            overwrite=overwrite,
        )


def _run_sample_zlt1_aggregate_stage(
    zbins: Sequence[ZBinCatalog],
    tiles: Sequence[TileDef],
    rlz: int,
    inst_list: Sequence[int],
    jmock_outdir: str,
    sample_cfg: Dict[str, float | str],
    overwrite: bool = False,
) -> None:
    """Compute and save z<1 spectra from summed maps for one tracer sample.

    Saves files named like:
    rlzX_TM{inst}_auto_cross_pred_{addstr}_zmin=0.0_zmax=1.0_{tile}.npz
    """
    addstr = f"{sample_cfg['galstr']}_lt_{sample_cfg['m_max_gal']:.1f}_CIBERfidmask"

    # Use discovered z-bins in order and keep only bins strictly below z=1.
    zkeys = sorted([(z.zmin, z.zmax) for z in zbins if z.zmin >= 0.0 and z.zmax <= 1.0], key=lambda x: x[0])

    if len(zkeys) == 0:
        print(f"[warning] No z-bins available for z<1 aggregation for sample '{addstr}'")
        return

    for inst in inst_list:
        outdir_pred = os.path.join(jmock_outdir, f"mock_ps_pred/TM{inst}/indiv")
        os.makedirs(outdir_pred, exist_ok=True)

        for tile in tiles:
            outpath = os.path.join(
                outdir_pred,
                f"rlz{rlz}_TM{inst}_auto_cross_pred_{addstr}_zmin=0.0_zmax=1.0_{tile.label}.npz",
            )
            if (not overwrite) and os.path.exists(outpath):
                continue

            sum_intensity = None
            sum_counts = None
            n_loaded = 0

            for zmin, zmax in zkeys:
                int_path = os.path.join(
                    jmock_outdir,
                    f"mock_maps/intensity/TM{inst}/rlz{rlz}_TM{inst}_{addstr}_zmin={zmin}_zmax={zmax}_pred_{tile.label}_intensity.npz",
                )
                gal_path = os.path.join(
                    jmock_outdir,
                    f"mock_maps/galaxy/TM{inst}/rlz{rlz}_TM{inst}_{addstr}_zmin={zmin}_zmax={zmax}_{tile.label}_galaxy.npz",
                )
                if (not os.path.exists(int_path)) or (not os.path.exists(gal_path)):
                    continue

                int_map = np.load(int_path)["ciber_map"]
                gal_counts = np.load(gal_path)["gal_counts"]
                if sum_intensity is None:
                    sum_intensity = np.array(int_map, copy=True)
                    sum_counts = np.array(gal_counts, copy=True)
                else:
                    sum_intensity += int_map
                    sum_counts += gal_counts
                n_loaded += 1

            if n_loaded == 0 or sum_intensity is None or sum_counts is None:
                print(
                    f"[warning] Could not build z<1 aggregate for TM{inst}, {addstr}, {tile.label}: missing per-z maps"
                )
                continue

            mean_counts = float(np.mean(sum_counts))
            if mean_counts <= 0:
                print(
                    f"[warning] Mean counts <= 0 for z<1 aggregate TM{inst}, {addstr}, {tile.label}; skipping"
                )
                continue

            gal_overdens = (sum_counts - mean_counts) / mean_counts
            meansub_intensity = sum_intensity - np.mean(sum_intensity)

            pixsize_arcsec = tile.tile_size_deg * 3600.0 / float(sum_intensity.shape[0])

            # Match current z<1 summary plotting choice for consistency.
            nbins_ps = 26
            lb, clI_comb, clI_err_comb = get_power_spec(
                meansub_intensity,
                nbins=nbins_ps,
                pixsize=pixsize_arcsec,
            )
            _, clg_comb, clg_err_comb = get_power_spec(
                gal_overdens,
                nbins=nbins_ps,
                pixsize=pixsize_arcsec,
            )
            _, clx_comb, clx_err_comb = get_power_spec(
                meansub_intensity,
                map_b=gal_overdens,
                nbins=nbins_ps,
                pixsize=pixsize_arcsec,
            )

            np.savez(
                outpath,
                lb=lb,
                clI_comb=clI_comb,
                clI_err_comb=clI_err_comb,
                clg_comb=clg_comb,
                clg_err_comb=clg_err_comb,
                clx_comb=clx_comb,
                clx_err_comb=clx_err_comb,
                zmin=0.0,
                zmax=1.0,
                n_zbins_aggregated=n_loaded,
                sample_addstr=addstr,
                tile_label=tile.label,
            )
            print(f"[z<1 aggregate] saved {outpath}")


def _plot_tiling_overview(
    zbin: ZBinCatalog,
    accepted_tiles: Sequence[TileDef],
    rejected_tiles: Sequence[TileDef],
    metrics: Dict[int, Dict[str, float]],
    outdir: str,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    all_ra = []
    all_dec = []
    all_w = []
    for pop in zbin.populations:
        ra = fits.open(os.path.join(zbin.catalog_dir, f"cat_ra_pop_{pop}.fits"))[1].data["ra"]
        dec = fits.open(os.path.join(zbin.catalog_dir, f"cat_dec_pop_{pop}.fits"))[1].data["dec"]
        flux = fits.open(os.path.join(zbin.catalog_dir, f"cat_0.900_1.200_um_pop_{pop}.fits"))[1].data["flux"]
        m = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(flux) & (flux > 0)
        all_ra.append(ra[m])
        all_dec.append(dec[m])
        all_w.append(flux[m])

    ra = np.concatenate(all_ra)
    dec = np.concatenate(all_dec)
    w = np.log10(np.concatenate(all_w))

    nb = 300
    h, xedges, yedges = np.histogram2d(ra, dec, bins=nb, weights=w)

    plt.figure(figsize=(7, 6))
    plt.imshow(
        h.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="magma",
        aspect="auto",
    )
    for tile in rejected_tiles:
        rect = plt.Rectangle(
            (tile.ra_min, tile.dec_min),
            tile.ra_max - tile.ra_min,
            tile.dec_max - tile.dec_min,
            fill=False,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.35,
        )
        plt.gca().add_patch(rect)

    for tile in accepted_tiles:
        rect = plt.Rectangle(
            (tile.ra_min, tile.dec_min),
            tile.ra_max - tile.ra_min,
            tile.dec_max - tile.dec_min,
            fill=False,
            edgecolor="cyan",
            linewidth=1.7,
            alpha=0.95,
        )
        plt.gca().add_patch(rect)
        mm = metrics.get(tile.tile_idx, None)
        if mm is not None:
            plt.text(
                tile.ra_min,
                tile.dec_min,
                f"{tile.tile_idx}: N={mm['n_obj']}, cov={mm['coverage']:.2f}",
                fontsize=6,
                color="cyan",
                alpha=0.9,
            )
    plt.xlabel("RA [deg]")
    plt.ylabel("Dec [deg]")
    plt.title("Tiling Overlay on Full Mock Density")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "tiling_overlay_TM1_proxy.png"), dpi=180)
    plt.close()


def _plot_field_average_spectra(
    jmock_outdir: str,
    rlz: int,
    inst: int,
    outdir: str,
    selected_tiles: Sequence[TileDef],
    masking_maglim: float,
) -> None:
    tm_outdir = os.path.join(outdir, f"TM{inst}")
    os.makedirs(tm_outdir, exist_ok=True)
    fpaths = glob.glob(os.path.join(jmock_outdir, f"mock_ps_pred/TM{inst}/indiv/rlz{rlz}_TM{inst}_auto_cross_pred_*.npz"))
    by_zbin = {}
    by_group = {}
    allowed_tiles = {tile.label for tile in selected_tiles}

    zpat = re.compile(r"zmin=([0-9.]+)_zmax=([0-9.]+)")
    tpat = re.compile(r"_tile\d+_[0-9.]+deg\.npz$")

    def _group_display_label(group: str) -> str:
        if "hsc_i_lt_25.0" in group:
            return "HSC i < 25"
        if "sdss_z_lt_22.0" in group:
            return "SDSS z < 22"
        if "fullmapcache" in group:
            return "Full-map cache"
        return group

    def _group_short_tag(group: str) -> str:
        if "hsc_i_lt_25.0" in group:
            return "hsc_i_lt_25.0"
        if "sdss_z_lt_22.0" in group:
            return "sdss_z_lt_22.0"
        if "fullmapcache" in group:
            return "fullmapcache"
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", group)

    for fp in fpaths:
        name = os.path.basename(fp)
        if allowed_tiles and not any(lbl in name for lbl in allowed_tiles):
            continue
        zm = zpat.search(name)
        if zm is None:
            continue
        zkey = (float(zm.group(1)), float(zm.group(2)))
        core = name.split("auto_cross_pred_")[-1]
        core = tpat.sub("", core)
        core = zpat.sub("", core).replace("__", "_").strip("_")

        dat = np.load(fp)
        lb = dat["lb"]
        entry = {
            "lb": lb,
            "clI": dat["clI_comb"],
            "clg": dat["clg_comb"],
            "clx": dat["clx_comb"],
        }
        by_zbin.setdefault((core, zkey), []).append(entry)
        by_group.setdefault(core, []).append(entry)

    def _plot_group(entries, title, savebase):
        if not entries:
            return None
        lb = entries[0]["lb"]
        pf = lb * (lb + 1) / (2.0 * np.pi)
        clx_stack = np.array([e["clx"] for e in entries])
        clg_stack = np.array([e["clg"] for e in entries])
        cli_stack = np.array([e["clI"] for e in entries])

        plt.figure(figsize=(6.0, 5.3))
        mean_clx = np.mean(clx_stack, axis=0)
        mean_clg = np.mean(clg_stack, axis=0)
        mean_cli = np.mean(cli_stack, axis=0)
        plt.plot(lb, pf * mean_clx, color="tab:blue", linewidth=2.2, label="cross (field avg)")
        plt.plot(lb, pf * mean_clg, color="tab:orange", linewidth=2.0, label="gal auto (field avg)")
        plt.plot(lb, pf * mean_cli, color="tab:green", linewidth=2.0, label="intensity auto (field avg)")

        plt.xscale("log")
        plt.yscale("log")
        plt.grid(alpha=0.3)
        plt.xlabel(r"$\ell$", fontsize=15)
        plt.ylabel(r"$D_{\ell}$", fontsize=15)
        cut_line1 = f"CIB map mask cut: m_Vega > {masking_maglim:.1f}"
        if "fullmapcache" in title:
            cut_line2 = "Tracer cut: broad cache selection (m_gal < 30, CIBER-unmasked)"
        elif "sdss_z_lt" in title:
            cut_line2 = "Tracer cut: Legacy (sdss_z) mag limit + CIBER-unmasked"
        elif "hsc_i_lt" in title:
            cut_line2 = "Tracer cut: HSC (hsc_i) mag limit + CIBER-unmasked"
        else:
            cut_line2 = "Tracer cut: see filename tag"

        plt.title(title)
        plt.figtext(0.12, 0.02, cut_line1 + "\n" + cut_line2, fontsize=9)
        plt.legend(fontsize=9)
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.savefig(savebase, dpi=170)
        plt.close()

        return {
            "lb": lb,
            "pf": pf,
            "mean_clx": mean_clx,
            "mean_clg": mean_clg,
            "mean_cli": mean_cli,
        }

    def _compute_zlt1_from_maps(group: str, zkeys: Sequence[Tuple[float, float]]):
        per_tile = []
        for tile in selected_tiles:
            sum_intensity = None
            sum_counts = None
            n_loaded = 0
            for zmin, zmax in zkeys:
                int_path = os.path.join(
                    jmock_outdir,
                    f"mock_maps/intensity/TM{inst}/rlz{rlz}_TM{inst}_{group}_zmin={zmin}_zmax={zmax}_pred_{tile.label}_intensity.npz",
                )
                gal_path = os.path.join(
                    jmock_outdir,
                    f"mock_maps/galaxy/TM{inst}/rlz{rlz}_TM{inst}_{group}_zmin={zmin}_zmax={zmax}_{tile.label}_galaxy.npz",
                )
                if (not os.path.exists(int_path)) or (not os.path.exists(gal_path)):
                    continue

                int_map = np.load(int_path)["ciber_map"]
                gal_counts = np.load(gal_path)["gal_counts"]
                if sum_intensity is None:
                    sum_intensity = np.array(int_map, copy=True)
                    sum_counts = np.array(gal_counts, copy=True)
                else:
                    sum_intensity += int_map
                    sum_counts += gal_counts
                n_loaded += 1

            if n_loaded == 0 or sum_intensity is None or sum_counts is None:
                continue
            if np.mean(sum_counts) <= 0:
                continue

            gal_overdens = (sum_counts - np.mean(sum_counts)) / np.mean(sum_counts)
            meansub_intensity = sum_intensity - np.mean(sum_intensity)

            pixsize_arcsec = tile.tile_size_deg * 3600.0 / float(sum_intensity.shape[0])
            nbins_ps = 26
            lb_zlt1, cli_zlt1, _ = get_power_spec(meansub_intensity, nbins=nbins_ps, pixsize=pixsize_arcsec)
            _, clg_zlt1, _ = get_power_spec(gal_overdens, nbins=nbins_ps, pixsize=pixsize_arcsec)
            _, clx_zlt1, _ = get_power_spec(meansub_intensity, map_b=gal_overdens, nbins=nbins_ps, pixsize=pixsize_arcsec)

            per_tile.append((lb_zlt1, cli_zlt1, clg_zlt1, clx_zlt1))

        if len(per_tile) == 0:
            return None

        lb_zlt1 = per_tile[0][0]
        cli_stack = np.array([x[1] for x in per_tile])
        clg_stack = np.array([x[2] for x in per_tile])
        clx_stack = np.array([x[3] for x in per_tile])

        return {
            "lb": lb_zlt1,
            "mean_cli": np.mean(cli_stack, axis=0),
            "mean_clg": np.mean(clg_stack, axis=0),
            "mean_clx": np.mean(clx_stack, axis=0),
        }

    group_totals = {}
    group_zmeans = {}

    for group, entries in by_group.items():
        full_dir = os.path.join(tm_outdir, group, "full_sample")
        os.makedirs(full_dir, exist_ok=True)
        group_label = _group_display_label(group)
        stats = _plot_group(
            entries,
            f"TM{inst}, {group_label} (all z bins)",
            os.path.join(full_dir, f"TM{inst}_{group}_fieldavg_Dell.png"),
        )
        if stats is not None:
            group_totals[group] = stats

    for (group, zkey), entries in by_zbin.items():
        zmin, zmax = zkey
        zdir = os.path.join(tm_outdir, group, f"zbin_{zmin:.1f}_{zmax:.1f}")
        os.makedirs(zdir, exist_ok=True)
        group_label = _group_display_label(group)
        stats = _plot_group(
            entries,
            f"TM{inst}, {group_label}, z=[{zmin:.1f},{zmax:.1f}]",
            os.path.join(zdir, f"TM{inst}_{group}_zmin={zmin}_zmax={zmax}_tiles_plus_fieldavg_Dell.png"),
        )
        if stats is not None:
            group_zmeans.setdefault(group, []).append((zkey, stats))

    # Requested multi-panel summaries: one panel per z bin for each sample/group.
    for group, zlist in group_zmeans.items():
        zlist = sorted(zlist, key=lambda x: x[0][0])
        if len(zlist) == 0:
            continue

        group_label = _group_display_label(group)
        group_tag = _group_short_tag(group)
        outdir_group = os.path.join(tm_outdir, group, "full_sample")
        os.makedirs(outdir_group, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(10, 8), sharey=True)
        axes_flat = axes.ravel()

        n_show = min(5, len(zlist))
        for idx in range(5):
            ax = axes_flat[idx]
            if idx >= n_show:
                ax.set_visible(False)
                continue
            zkey, st = zlist[idx]
            zmin, zmax = zkey
            lb = st["lb"]
            pf = lb * (lb + 1) / (2.0 * np.pi)
            ax.plot(lb, pf * st["mean_clx"], color="tab:blue", linewidth=1.8, label="cross")
            ax.plot(lb, pf * st["mean_clg"], color="tab:orange", linewidth=1.8, label="gal auto")
            ax.plot(lb, pf * st["mean_cli"], color="tab:green", linewidth=1.8, label="intensity auto")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(alpha=0.3)
            ax.set_xlabel(r"$\ell$", fontsize=11)
            ax.set_title(f"z=[{zmin:.1f},{zmax:.1f}]", fontsize=10)

        # Bottom-right panel: recompute z<1 directly from summed maps and summed counts.
        zkeys = [z for z, _ in zlist]
        zlt1_stats = _compute_zlt1_from_maps(group, zkeys)
        if zlt1_stats is not None:
            lb = zlt1_stats["lb"]
            mean_clx_zlt1 = zlt1_stats["mean_clx"]
            mean_clg_zlt1 = zlt1_stats["mean_clg"]
            mean_cli_zlt1 = zlt1_stats["mean_cli"]
        elif group in group_totals:
            lb = group_totals[group]["lb"]
            mean_clx_zlt1 = group_totals[group]["mean_clx"]
            mean_clg_zlt1 = group_totals[group]["mean_clg"]
            mean_cli_zlt1 = group_totals[group]["mean_cli"]
        else:
            lb = zlist[0][1]["lb"]
            mean_clx_zlt1 = np.mean([st["mean_clx"] for _, st in zlist], axis=0)
            mean_clg_zlt1 = np.mean([st["mean_clg"] for _, st in zlist], axis=0)
            mean_cli_zlt1 = np.mean([st["mean_cli"] for _, st in zlist], axis=0)

        pf = lb * (lb + 1) / (2.0 * np.pi)

        ax_zlt1 = axes_flat[5]
        ax_zlt1.plot(lb, pf * mean_clx_zlt1, color="tab:blue", linewidth=2.0, label="cross")
        ax_zlt1.plot(lb, pf * mean_clg_zlt1, color="tab:orange", linewidth=2.0, label="gal auto")
        ax_zlt1.plot(lb, pf * mean_cli_zlt1, color="tab:green", linewidth=2.0, label="intensity auto")
        ax_zlt1.set_xscale("log")
        ax_zlt1.set_yscale("log")
        ax_zlt1.grid(alpha=0.3)
        ax_zlt1.set_xlabel(r"$\ell$", fontsize=11)
        ax_zlt1.set_title("z < 1", fontsize=10)
        if n_show > 0:
            axes_flat[0].legend(fontsize=12, loc="best")

        axes_flat[0].set_ylabel(r"$D_{\ell}$", fontsize=11)
        axes_flat[3].set_ylabel(r"$D_{\ell}$", fontsize=11)
        if inst == 1 and "hsc_i_lt_25.0" in group:
            fig.suptitle("TM1, HSC i < 25 (8 x 8 deg$^2$)", fontsize=16)
        elif inst == 2 and "hsc_i_lt_25.0" in group:
            fig.suptitle("TM2, HSC i < 25 (8 x 8 deg$^2$)", fontsize=16)
        else:
            fig.suptitle(f"TM{inst}, {group_label}", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(outdir_group, f"TM{inst}_{group_tag}_zbin_multipanel_Dell.png"), dpi=180)
        plt.close(fig)

    for group, zlist in group_zmeans.items():
        if group not in group_totals:
            continue

        zlist = sorted(zlist, key=lambda x: x[0][0])
        lb = group_totals[group]["lb"]
        pf = lb * (lb + 1) / (2.0 * np.pi)
        idx = np.argmin(np.abs(lb - 1000.0))

        zcen = np.array([0.5 * (z[0] + z[1]) for z, _ in zlist])
        total_cross = pf[idx] * group_totals[group]["mean_clx"][idx]
        total_gal = pf[idx] * group_totals[group]["mean_clg"][idx]
        total_int = pf[idx] * group_totals[group]["mean_cli"][idx]

        z_cross = np.array([pf[idx] * st["mean_clx"][idx] for _, st in zlist])
        z_gal = np.array([pf[idx] * st["mean_clg"][idx] for _, st in zlist])
        z_int = np.array([pf[idx] * st["mean_cli"][idx] for _, st in zlist])

        cmp_dir = os.path.join(tm_outdir, group, "full_sample")
        os.makedirs(cmp_dir, exist_ok=True)
        plt.figure(figsize=(5.8, 4.8))
        plt.plot(zcen, z_cross, "o-", label="cross by z-bin")
        plt.plot(zcen, z_gal, "o-", label="gal auto by z-bin")
        plt.plot(zcen, z_int, "o-", label="intensity auto by z-bin")
        plt.axhline(total_cross, linestyle="--", linewidth=1.3, label="cross total")
        plt.axhline(total_gal, linestyle="--", linewidth=1.3, label="gal auto total")
        plt.axhline(total_int, linestyle="--", linewidth=1.3, label="intensity auto total")
        plt.yscale("log")
        plt.grid(alpha=0.3)
        plt.xlabel("Redshift bin center", fontsize=13)
        plt.ylabel(r"$D_{\ell=1000}$", fontsize=13)
        plt.title(f"TM{inst} {group}: z-slice field avg vs total")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(cmp_dir, f"TM{inst}_{group}_zslice_vs_total_Dell_l1000.png"), dpi=180)
        plt.close()


def _regenerate_fullmap_tile_figures(
    jmock_outdir: str,
    rlz: int,
    inst_list: Sequence[int],
    zbins: Sequence[ZBinCatalog],
    selected_tiles: Sequence[TileDef],
) -> None:
    for inst in inst_list:
        outdir = os.path.join(jmock_outdir, "diagnostics", f"TM{inst}", "full_tile_checks")
        os.makedirs(outdir, exist_ok=True)
        for zbin in zbins:
            for tile in selected_tiles:
                map_path = os.path.join(
                    jmock_outdir,
                    f"mock_maps/intensity/TM{inst}/rlz{rlz}_TM{inst}_full_zmin={zbin.zmin}_zmax={zbin.zmax}_{tile.label}_intensity.npz",
                )
                if not os.path.exists(map_path):
                    continue

                dat = np.load(map_path)
                cmap = dat["ciber_map"]
                units = str(dat["intensity_units"]) if "intensity_units" in dat else "nW m^-2 sr^-1"

                finite = np.isfinite(cmap)
                if not np.any(finite):
                    continue

                clip = np.nanpercentile(cmap[finite], 99.5)
                clip = max(clip, np.nanmax(cmap[finite]) * 1e-3)
                figbase = os.path.join(
                    outdir,
                    f"rlz{rlz}_TM{inst}_zmin={zbin.zmin}_zmax={zbin.zmax}_{tile.label}",
                )

                plt.figure(figsize=(6, 5))
                plt.imshow(cmap, origin="lower", cmap="magma", vmin=0.0, vmax=clip)
                cb = plt.colorbar()
                cb.set_label(f"Intensity [{units}]")
                plt.title(f"TM{inst} Full CIB Map z=[{zbin.zmin:.1f},{zbin.zmax:.1f}] {tile.label}")
                plt.tight_layout()
                plt.savefig(figbase + "_map.png", dpi=160)
                plt.close()

                pix = cmap[finite]
                pix = pix[pix > 0]
                if pix.size > 0:
                    plt.figure(figsize=(6, 5))
                    plt.hist(pix, bins=200, histtype="stepfilled", alpha=0.8)
                    plt.yscale("log")
                    plt.xlabel(f"Intensity [{units}]")
                    plt.ylabel("N pixels")
                    plt.title("Per-pixel Intensity Histogram")
                    plt.tight_layout()
                    plt.savefig(figbase + "_hist.png", dpi=160)
                    plt.close()


def _write_manifest(
    outpath: str,
    base_dir: str,
    rlz: int,
    zbins: Sequence[ZBinCatalog],
    tiles: Sequence[TileDef],
    warnings: Sequence[str],
    args: argparse.Namespace,
) -> None:
    payload = {
        "base_dir": base_dir,
        "realization": rlz,
        "ifield_map": args.ifield_map,
        "stages": args.stages,
        "samples": args.samples,
        "tile_sizes_deg": args.tile_sizes,
        "zbins": [asdict(z) for z in zbins],
        "tiles": [asdict(t) for t in tiles],
        "warnings": list(warnings),
    }
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v3 Jordan mock pipeline with tiled outputs")
    parser.add_argument(
        "--base-dir",
        default=os.path.join(config.ciber_basepath, "data/jordan_mocks/v3/fov_10.0"),
        help="Input v3 mock base directory",
    )
    parser.add_argument(
        "--jmock-outdir",
        default=os.path.join(config.ciber_basepath, "data/jordan_mocks/v3_boxed_outputs/"),
        help="Output base directory for generated products",
    )
    parser.add_argument("--rlz", type=int, default=1, help="Realization index (default: 1)")
    parser.add_argument(
        "--tile-sizes",
        type=float,
        nargs="+",
        default=[2.0],
        help="Tile sizes in degrees. Use '--tile-sizes 2 5' to generate both 2x2 and 5x5 products.",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Optional cap on number of discovered tiles to process (use 1 for a single cutout test)",
    )
    parser.add_argument(
        "--centered-single-tile",
        action="store_true",
        help="Use one tile centered on the catalog centroid for each tile size",
    )
    parser.add_argument(
        "--target-n-tiles",
        type=int,
        default=3,
        help="Number of complete tiles to keep for processing/summary",
    )
    parser.add_argument(
        "--min-objects-per-tile",
        type=int,
        default=5000,
        help="Minimum number of sources in a tile to be considered complete",
    )
    parser.add_argument(
        "--min-coverage-frac",
        type=float,
        default=0.75,
        help="Minimum occupancy fraction (0-1) on an 8x8 grid for tile completeness",
    )
    parser.add_argument(
        "--ifield-map",
        type=int,
        default=8,
        help="CIBER beam/PSF reference field (default: 8)",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["discover", "maps", "samples", "summary", "all"],
        default=["all"],
        help="Stages to run",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        choices=list(DEFAULT_SAMPLES.keys()),
        default=["legacy", "hsc"],
        help="Tracer samples to generate",
    )
    parser.add_argument(
        "--masking-maglim",
        type=float,
        default=16.0,
        help="Masking magnitude limit in Vega for CIBER masking",
    )
    parser.add_argument(
        "--skip-map-mask",
        action="store_true",
        help="Skip applying spatial source masks to CIB maps; keep only catalog-level magnitude/redshift cuts",
    )
    parser.add_argument(
        "--save-magcut-diagnostics",
        action="store_true",
        help="Save magnitude-count histograms before/after cuts for each processing call",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest output path. Defaults to <jmock-outdir>/manifests/rlzXXX_manifest.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute products even if matching output files already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stages = set(args.stages)
    if "all" in stages:
        stages = {"discover", "maps", "samples", "summary"}

    if len(args.tile_sizes) == 1:
        tile_tag = str(args.tile_sizes[0]).replace(".", "p")
        run_outdir = os.path.join(args.jmock_outdir, f"tiles_{tile_tag}deg")
    else:
        run_outdir = args.jmock_outdir

    if not run_outdir.endswith(os.sep):
        run_outdir = run_outdir + os.sep

    zbins, warnings = discover_v3_zbins(args.base_dir, rlz=args.rlz)
    if not zbins:
        raise RuntimeError(f"No valid z-bin catalogs discovered under {args.base_dir}")

    tile_catalog = _tile_source_catalog(zbins)
    ra, dec = _load_ra_dec(tile_catalog.catalog_dir, tile_catalog.populations[0])

    all_tiles: List[TileDef] = []
    if args.centered_single_tile:
        for tile_size in args.tile_sizes:
            all_tiles.append(build_centered_tile(ra, dec, tile_size_deg=tile_size))
        metrics = _tile_quality_metrics(all_tiles, ra, dec, grid_n=8)
        selected_tiles = all_tiles
        rejected_tiles: List[TileDef] = []
    else:
        for tile_size in args.tile_sizes:
            all_tiles.extend(build_complete_tiles(ra, dec, tile_size_deg=tile_size))

        metrics = _tile_quality_metrics(all_tiles, ra, dec, grid_n=8)
        selected_tiles, rejected_tiles = _select_complete_tiles(
            all_tiles,
            metrics,
            min_objects_per_tile=args.min_objects_per_tile,
            min_coverage_frac=args.min_coverage_frac,
            target_n_tiles=args.target_n_tiles,
        )

    if args.max_tiles is not None:
        selected_tiles = selected_tiles[: args.max_tiles]

    if not selected_tiles:
        raise RuntimeError("No spatially complete tiles found for requested tile sizes")

    print(f"Discovered {len(zbins)} z-bins and {len(all_tiles)} candidate tiles")
    print(f"Selected {len(selected_tiles)} complete tiles for processing")
    for w in warnings:
        print(f"[warning] {w}")

    if "maps" in stages:
        _run_maps_stage(
            zbins=zbins,
            tiles=selected_tiles,
            rlz=args.rlz,
            inst_list=[1, 2],
            jmock_outdir=run_outdir,
            ifield_map=args.ifield_map,
            masking_maglim=args.masking_maglim,
            apply_map_mask=(not args.skip_map_mask),
            save_magcut_diagnostics=args.save_magcut_diagnostics,
            overwrite=args.overwrite,
        )

    if "samples" in stages:
        _run_selection_spectra_stage(
            zbins=zbins,
            tiles=selected_tiles,
            rlz=args.rlz,
            inst_list=[1, 2],
            jmock_outdir=run_outdir,
            ifield_map=args.ifield_map,
            masking_maglim=args.masking_maglim,
            apply_map_mask=(not args.skip_map_mask),
            save_magcut_diagnostics=args.save_magcut_diagnostics,
            run_samples=args.samples,
            overwrite=args.overwrite,
        )

    if "summary" in stages:
        summary_dir = os.path.join(run_outdir, "diagnostics", "summary")
        _plot_tiling_overview(zbins[0], selected_tiles, rejected_tiles, metrics, summary_dir)
        _regenerate_fullmap_tile_figures(run_outdir, args.rlz, [1, 2], zbins, selected_tiles)
        for inst in [1, 2]:
            _plot_field_average_spectra(
                run_outdir,
                args.rlz,
                inst,
                summary_dir,
                selected_tiles,
                args.masking_maglim,
            )

    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = os.path.join(
            run_outdir,
            "manifests",
            f"rlz{args.rlz:03d}_manifest.json",
        )
    _write_manifest(
        outpath=manifest_path,
        base_dir=args.base_dir,
        rlz=args.rlz,
        zbins=zbins,
        tiles=selected_tiles,
        warnings=warnings,
        args=args,
    )
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
