#!/usr/bin/env python3
"""
Extract RA/DEC/Z_MAG/Z_PHOT subsets from a large FITS catalog for CIBER fields.

Default behavior:
  - Process all fields: 4,5,6,7,8
Optional:
  - Restrict fields with --fields (e.g. --fields 4 7)

For each field, selects rows within:
  RA  in [ra_center - dra, ra_center + dra]  (with wrap-around handling)
  DEC in [dec_center - ddec, dec_center + ddec]

Writes one FITS file per field in --outdir.
"""

import os
import argparse
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack

FITS_PATH_DEFAULT = "/scratch/06224/rfederst/data/good_zphot_no_dup_z22_5.fits"
REQ_COLS = ["ra", "dec", "zphot", "mag_z"]

ra_cen_ciber_fields = {4: 190.5, 5: 193.1, 6: 217.2, 7: 218.4, 8: 242.8}
dec_cen_ciber_fields = {4: 8.0,   5: 28.0,  6: 33.2,  7: 34.8,  8: 54.8}


def normalize_ra_deg(x):
    """Map RA to [0, 360)."""
    return x % 360.0


def print_columns_and_find_hdu(fits_path, required_cols):
    with fits.open(fits_path, memmap=True) as hdul:
        chosen_hdu = None
        print(f"\nOpened: {fits_path}")
        print(f"Number of HDUs: {len(hdul)}\n")

        for i, hdu in enumerate(hdul):
            data = hdu.data
            if data is None or not hasattr(data, "names") or data.names is None:
                continue

            names = list(data.names)
            print(f"HDU {i} ({hdu.name}) has {len(names)} columns:")
            print(", ".join(names))
            print()

            if chosen_hdu is None and all(c in names for c in required_cols):
                chosen_hdu = i

        if chosen_hdu is None:
            raise RuntimeError(f"No table HDU contains required columns: {required_cols}")

        print(f"Using HDU {chosen_hdu} (contains {required_cols})\n")
        return chosen_hdu


def iter_filtered_chunks(fits_path, hdu, ra_min, ra_max, dec_min, dec_max, chunk_size):
    """Yield filtered structured-array chunks with REQ_COLS only."""
    ra_min = normalize_ra_deg(ra_min)
    ra_max = normalize_ra_deg(ra_max)

    with fits.open(fits_path, memmap=True) as hdul:
        data = hdul[hdu].data
        n = len(data)
        out_dtype = [(c, data[c].dtype) for c in REQ_COLS]

        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)

            ra = np.asarray(data["ra"][start:stop], dtype=np.float64)
            dec = np.asarray(data["dec"][start:stop], dtype=np.float64)
            ra = normalize_ra_deg(ra)

            # RA wrap-aware mask
            if ra_min <= ra_max:
                ra_mask = (ra >= ra_min) & (ra <= ra_max)
            else:
                ra_mask = (ra >= ra_min) | (ra <= ra_max)

            dec_mask = (dec >= dec_min) & (dec <= dec_max)
            mask = ra_mask & dec_mask

            if not np.any(mask):
                continue

            idx = np.flatnonzero(mask) + start
            out = np.empty(len(idx), dtype=out_dtype)
            for c in REQ_COLS:
                out[c] = data[c][idx]

            yield out


def save_field_subset(
    fits_path,
    hdu,
    outdir,
    ifield,
    ra_center,
    dec_center,
    dra,
    ddec,
    chunk_size=1_000_000,
):
    os.makedirs(outdir, exist_ok=True)

    ra_min = ra_center - dra
    ra_max = ra_center + dra
    dec_min = dec_center - ddec
    dec_max = dec_center + ddec

    print(
        f"Field {ifield}: center=(RA={ra_center}, DEC={dec_center}), "
        f"box: RA[{ra_min}, {ra_max}], DEC[{dec_min}, {dec_max}]"
    )

    tables = []
    total = 0

    for chunk in iter_filtered_chunks(
        fits_path=fits_path,
        hdu=hdu,
        ra_min=ra_min,
        ra_max=ra_max,
        dec_min=dec_min,
        dec_max=dec_max,
        chunk_size=chunk_size,
    ):
        tables.append(Table(chunk))
        total += len(chunk)

    outname = f"ciber_ifield{ifield}_RADEC_ZMAG_ZPHOT.fits"
    outpath = os.path.join(outdir, outname)

    if total == 0:
        # Preserve expected column names; use float fallback dtypes for empty output
        empty = Table({c: np.array([], dtype=np.float64) for c in REQ_COLS})
        empty.write(outpath, format="fits", overwrite=True)
    else:
        out_table = vstack(tables, metadata_conflicts="silent")
        out_table.meta["IFIELD"] = ifield
        out_table.meta["RA_CEN"] = ra_center
        out_table.meta["DEC_CEN"] = dec_center
        out_table.meta["DRA"] = dra
        out_table.meta["DDEC"] = ddec
        out_table.write(outpath, format="fits", overwrite=True)

    print(f"  -> Saved {total} rows to {outpath}\n")


def parse_fields_arg(fields_arg):
    available = sorted(ra_cen_ciber_fields.keys())
    if fields_arg is None or len(fields_arg) == 0:
        return available

    requested = sorted(set(fields_arg))
    bad = [f for f in requested if f not in ra_cen_ciber_fields or f not in dec_cen_ciber_fields]
    if bad:
        raise ValueError(f"Invalid field(s): {bad}. Valid fields are {available}")
    return requested


def main():
    parser = argparse.ArgumentParser(
        description="Extract RA/DEC/Z_MAG/Z_PHOT FITS subsets for selected CIBER fields."
    )
    parser.add_argument("--fits", default=FITS_PATH_DEFAULT, help="Input FITS path")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument(
        "--fields",
        nargs="*",
        type=int,
        default=None,
        help="Field IDs to run (default: all fields 4 5 6 7 8). Example: --fields 4 7",
    )
    parser.add_argument("--dra", type=float, default=1.0, help="Half-width in RA (deg)")
    parser.add_argument("--ddec", type=float, default=1.0, help="Half-width in DEC (deg)")
    parser.add_argument("--chunk-size", type=int, default=1_000_000, help="Rows per chunk")
    args = parser.parse_args()

    fields_to_run = parse_fields_arg(args.fields)
    hdu = print_columns_and_find_hdu(args.fits, REQ_COLS)

    print(f"Running fields: {fields_to_run}\n")
    for ifield in fields_to_run:
        save_field_subset(
            fits_path=args.fits,
            hdu=hdu,
            outdir=args.outdir,
            ifield=ifield,
            ra_center=ra_cen_ciber_fields[ifield],
            dec_center=dec_cen_ciber_fields[ifield],
            dra=args.dra,
            ddec=args.ddec,
            chunk_size=args.chunk_size,
        )


if __name__ == "__main__":
    main()