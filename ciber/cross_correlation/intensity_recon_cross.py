import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import fits


import config


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

    
from ciber.core.powerspec_pipeline import CIBER_PS_pipeline
from ciber.cross_correlation.galaxy_cross import (
    cat_select,
    return_default_gal_cat_dict,
    save_gal_density,
)
from ciber.io.catalog_utils import catalog_df_add_xy

from ciber.mocks.cib_mocks import ciber_mock


SUPPORTED_BANDS = [
    "i_cmodel_mag",
    "z_cmodel_mag",
    "r_cmodel_mag",
    "g_cmodel_mag",
    "z_mag",
]


def _resolve_catalog_path(catalog_basepath, catname, ifield):
    return os.path.join(
        catalog_basepath,
        catname,
        "filt",
        f"{catname}_CIBER_ifield{ifield}.csv",
    )


def _ensure_mag_column(cat_df, mag_column):
    if mag_column not in cat_df.columns:
        supported = [col for col in SUPPORTED_BANDS if col in cat_df.columns]
        raise KeyError(
            f"Requested magnitude column '{mag_column}' is missing. "
            f"Available mag columns in file: {supported}"
        )


def inspect_hsc_catalog_fields(ifield_list, catalog_basepath=None, catname="HSC"):
    """Inspect available HSC columns and simple statistics for quick verification."""
    if catalog_basepath is None:
        catalog_basepath = config.ciber_basepath + "data/catalogs/"

    summary = {}
    for ifield in ifield_list:

        cat_fpath = catalog_basepath+'HSC/HSC_deep_CIBER_ifield'+str(ifield)+'_photz.csv'
        cat_df = pd.read_csv(cat_fpath)

        mag_cols = [
            col
            for col in ["g_cmodel_mag", "r_cmodel_mag", "i_cmodel_mag", "z_cmodel_mag"]
            if col in cat_df.columns
        ]
        stats = {}
        for col in mag_cols:
            vals = np.asarray(cat_df[col], dtype=float)
            finite = np.isfinite(vals)
            if np.any(finite):
                stats[col] = {
                    "n_finite": int(np.sum(finite)),
                    "p16": float(np.nanpercentile(vals[finite], 16)),
                    "p50": float(np.nanpercentile(vals[finite], 50)),
                    "p84": float(np.nanpercentile(vals[finite], 84)),
                }
            else:
                stats[col] = {"n_finite": 0, "p16": np.nan, "p50": np.nan, "p84": np.nan}

        summary[ifield] = {
            "catalog_path": cat_fpath,
            "n_rows": int(len(cat_df)),
            "mag_columns": mag_cols,
            "stats": stats,
        }

    return summary


def generate_intensity_map_from_catalog(
    x,
    y,
    mag_ab,
    ciber_inst,
    mask=None,
    imdim=1024,
    mean_subtract=True,
):
    """Generate a surface-brightness map by binning flux-weighted sources.

    The output map is in nW m^-2 sr^-1 per pixel and can be fed directly into
    the pseudo-Cl pipeline after masking/filtering.
    """
    cmock = ciber_mock(nx=imdim, ny=imdim)
    lam_dict = {1: 1.1, 2: 1.8}

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mag_ab = np.asarray(mag_ab, dtype=float)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(mag_ab)
    finite &= (x >= 0) & (x < imdim) & (y >= 0) & (y < imdim)

    if not np.any(finite):
        return np.zeros((imdim, imdim), dtype=float)

    flux = cmock.mag_2_nu_Inu(mag_ab[finite], lam_eff=lam_dict[ciber_inst] * 1e-6 * u.m).value

    # Histogram uses x first; transpose to match map orientation used elsewhere.
    H, _, _ = np.histogram2d(
        x[finite],
        y[finite],
        bins=[np.arange(imdim + 1) - 0.5, np.arange(imdim + 1) - 0.5],
        weights=flux,
    )
    intensity_map = H

    if mask is not None:
        mask_bool = np.asarray(mask) != 0
        intensity_map *= mask_bool
        if mean_subtract and np.any(mask_bool):
            mu = np.mean(intensity_map[mask_bool])
            intensity_map[mask_bool] -= mu
            intensity_map[~mask_bool] = 0.0
    elif mean_subtract:
        intensity_map -= np.mean(intensity_map)

    return intensity_map


def preprocess_intensity_maps(
    inst,
    ifield_list,
    catname="HSC",
    save=False,
    cat_fpath_list=None,
    addstr=None,

    mag_column_recon="i_cmodel_mag",
    mag_select_column="i_cmodel_mag",
    mag_min=18.0,
    mag_max=25.0,
    zmin=None,
    zmax=None,
    show=False,
    imdim=1024,
    **kwargs,
):
    """Build per-field flux-weighted intensity maps from catalog photometry."""
    gal_dict = return_default_gal_cat_dict()
    gal_dict = {**gal_dict, **kwargs}
    gal_dict["zmin"] = zmin
    gal_dict["zmax"] = zmax

    if catname == "LS":
        mag_column_recon = 'mag_z'
        mag_select_column = 'mag_z'

    if catname=='HSC':
        gal_dict['hsc_mag_max'] = mag_max
        gal_dict['hsc_mag_min'] = mag_min


    cbps = CIBER_PS_pipeline()
    intensity_maps = np.zeros((len(ifield_list), imdim, imdim), dtype=float)

    for fieldidx, ifield in enumerate(ifield_list):
        if cat_fpath_list is None:
            if catname == "HSC":
                cat_fpath = gal_dict["catalog_basepath"] + f"HSC/HSC_deep_CIBER_ifield{ifield}_photz.csv"
            elif catname == "LS":
                cat_fpath = f'data/catalogs/LS/ciber_ifield{ifield}_RADEC_ZMAG_ZPHOT.fits'
            # cat_fpath = _resolve_catalog_path(gal_dict["catalog_basepath"], catname, ifield)
        else:
            cat_fpath = cat_fpath_list[fieldidx]

        if catname == "HSC":
            cat_df = pd.read_csv(cat_fpath)
            _ensure_mag_column(cat_df, mag_column_recon)
            _ensure_mag_column(cat_df, mag_select_column)
            
            cat_sel_obj = cat_select(gal_dict=gal_dict, which_hsc_band=mag_select_column[0])
            cat_sel_obj.load_cat(cat_fpath, inst, catname)
            base_mask = cat_sel_obj.apply_cat_select(catname).astype(bool)

        elif catname == "LS":
            from astropy.table import Table
            # Use astropy Table which handles byte order automatically
            cat_table = Table.read(cat_fpath)
            cat_df = cat_table.to_pandas()
            print('HERE THE COLUMNS ARE', cat_df.columns)
            _ensure_mag_column(cat_df, mag_column_recon)
            _ensure_mag_column(cat_df, mag_select_column)

            print('min/max ra is ', np.min(cat_df['ra']), np.max(cat_df['ra']))
            print('min/max dec is ', np.min(cat_df['dec']), np.max(cat_df['dec']))
            cat_df = catalog_df_add_xy(cbps.ciber_field_dict[ifield], cat_df, datadir=config.ciber_basepath+'data/')

            # Manually populate cat_select with loaded LS data
            cat_sel_obj = cat_select(gal_dict)
            cat_sel_obj.cat_x = np.array(cat_df[f'x{inst}'])
            cat_sel_obj.cat_y = np.array(cat_df[f'y{inst}'])
            cat_sel_obj.cat_redshift = np.array(cat_df['zphot'])
            cat_sel_obj.cat_stack = [cat_sel_obj.cat_x, cat_sel_obj.cat_y, cat_sel_obj.cat_redshift]
            cat_sel_obj.cat_stack_labs = ['x', 'y', 'redshift', 'type']
            
            base_mask = cat_sel_obj.apply_cat_select(catname).astype(bool)

        mag = np.asarray(cat_df[mag_select_column], dtype=float)

        mag_recon = np.asarray(cat_df[mag_column_recon], dtype=float)
        mag_mask = np.isfinite(mag)

        if mag_min is not None:
            mag_mask &= mag >= mag_min
        if mag_max is not None:
            mag_mask &= mag <= mag_max

        print('sum of basemask is ', np.sum(base_mask), 'sum of mag_mask is ', np.sum(mag_mask))
        sel = base_mask & mag_mask
        print('sum of sel is ', np.sum(sel), 'for ifield ', ifield, 'and inst ', inst)

        x = np.asarray(cat_df[f"x{inst}"], dtype=float)[sel]
        y = np.asarray(cat_df[f"y{inst}"], dtype=float)[sel]
        mag_sel = mag_recon[sel]

        print(
            f"[intensity_recon] ifield={ifield}, selected_sources={len(x)}, "
            f"mag_col={mag_select_column}, mag_min={mag_min}, mag_max={mag_max}"
        )

        intensity_map = generate_intensity_map_from_catalog(
            x,
            y,
            mag_sel,
            ciber_inst=inst,
            mask=np.ones((imdim, imdim), dtype=bool),
            imdim=imdim,
            mean_subtract=True,
        )

        intensity_maps[fieldidx] = intensity_map

        if show:
            from ciber.plotting.plotting_fns import plot_map

            plot_map(intensity_map, title=f"Intensity recon {catname} ifield {ifield}")

    save_fpath = None
    if save:
        save_fpath = save_gal_density(
            inst,
            ifield_list,
            intensity_maps,
            catname,
            addstr=addstr,
        )

    return intensity_maps, save_fpath
