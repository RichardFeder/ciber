#!/usr/bin/env python3
"""Compute A_2h from mock power spectrum prediction files.

Procedure:
1) Estimate and remove shot noise using a high-ell window.
2) Convert residual to D_ell if inputs are C_ell.
3) Compute mean D_ell over a low-ell window as A_2h.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


def _load_component(npz_path: Path, component: str) -> Tuple[np.ndarray, np.ndarray]:
    pred = np.load(npz_path, allow_pickle=True)
    if "lb" not in pred:
        raise KeyError(f"Missing 'lb' in {npz_path}")
    lb = np.asarray(pred["lb"], dtype=float)

    key_candidates = {
        "cross": ("cross", "clx", "clx_comb", "cross_pred"),
        "gal_auto": ("gal_auto", "clg", "clg_comb", "auto"),
        "intensity_auto_tracer": ("intensity_auto_tracer",),
        "intensity_auto_full": ("intensity_auto_full",),
    }

    if component not in key_candidates:
        raise KeyError(f"Unsupported component: {component}")

    spec = None
    for key in key_candidates[component]:
        if key in pred:
            spec = np.asarray(pred[key], dtype=float)
            break
    if spec is None:
        raise KeyError(f"No spectrum key found for component '{component}' in {npz_path}")

    return lb, spec


def _estimate_shot_level(lb: np.ndarray, spec: np.ndarray, shot_ell_range: Tuple[float, float]) -> float:
    mask = np.isfinite(lb) & np.isfinite(spec) & (lb >= shot_ell_range[0]) & (lb <= shot_ell_range[1])
    if not np.any(mask):
        mask = np.isfinite(lb) & np.isfinite(spec)
    return float(np.nanmean(spec[mask]))


def _to_dell(lb: np.ndarray, cl: np.ndarray) -> np.ndarray:
    pf = lb * (lb + 1.0) / (2.0 * np.pi)
    return pf * cl


def compute_a2h(npz_path: Path, component: str, assume_dl: bool,
                shot_ell_range: Tuple[float, float], signal_ell_max: float) -> dict:
    lb, spec = _load_component(npz_path, component)

    if assume_dl:
        shot_level = _estimate_shot_level(lb, spec, shot_ell_range)
        dl = spec - shot_level
    else:
        shot_level = _estimate_shot_level(lb, spec, shot_ell_range)
        dl = _to_dell(lb, spec - shot_level)

    signal_mask = np.isfinite(lb) & np.isfinite(dl) & (lb <= signal_ell_max)
    if not np.any(signal_mask):
        signal_mask = np.isfinite(lb) & np.isfinite(dl)

    a2h = float(np.nanmean(dl[signal_mask]))
    return {
        "path": str(npz_path),
        "component": component,
        "assume_dl": assume_dl,
        "shot_level": shot_level,
        "a2h": a2h,
        "n_signal": int(np.sum(signal_mask)),
    }


def _iter_inputs(paths: Iterable[str], glob: str | None) -> Iterable[Path]:
    for p in paths:
        yield Path(p)
    if glob:
        for p in sorted(Path(".").glob(glob)):
            if p.is_file():
                yield p


def _standard_heads(cat: str) -> List[str]:
    if cat == "DESILS":
        return ["sdss_z_lt_22.0_CIBERfidmask", "sdss_z_lt_22.0"]
    if cat == "HSC":
        return ["hsc_i_lt_25.0_CIBERfidmask", "hsc_i_lt_25.0", "hsc_ilt25.0"]
    raise ValueError(f"Unsupported catalog: {cat}")


def _find_pred_file(basepath: Path, cat: str, inst: int, zlo: float | None, zhi: float) -> Path | None:
    base = basepath / "mock_ps_pred" / f"TM{inst}" / "field_average"
    for head in _standard_heads(cat):
        if zlo is None:
            name = f"pred_cls_TM{inst}_{head}_zmax={zhi}.npz"
            fp = base / name
            if fp.exists():
                return fp
        else:
            name = f"pred_cls_TM{inst}_{head}_zmin={zlo}_zmax={zhi}.npz"
            fp = base / name
            if fp.exists():
                return fp
    return None


def _collect_standard_inputs(basepath: Path, dz: float = 0.2) -> List[Path]:
    zbins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] if dz == 0.2 else None
    if zbins is None:
        raise ValueError("Only dz=0.2 is supported in the standard preset")

    inputs: List[Path] = []
    for cat in ["HSC", "DESILS"]:
        for inst in [1, 2]:
            for zlo, zhi in zip(zbins[:-1], zbins[1:]):
                fp = _find_pred_file(basepath, cat, inst, zlo, zhi)
                if fp is not None:
                    inputs.append(fp)
            fp = _find_pred_file(basepath, cat, inst, 0.0, 1.0)
            if fp is not None:
                inputs.append(fp)
            else:
                fp = _find_pred_file(basepath, cat, inst, None, 1.0)
                if fp is not None:
                    inputs.append(fp)
    return inputs


def _default_cache_path(args, basepath: Path | None) -> Path | None:
    if args.cache_path is not None:
        return Path(args.cache_path)
    if args.glob:
        return Path(args.glob).parent / "a2h_cache.json"
    if basepath is not None:
        return basepath / "a2h_cache.json"
    return None


def _write_cache(cache_path: Path, results: List[dict]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_results": len(results),
        "results": results,
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute A_2h from mock power spectra")
    parser.add_argument("--input", nargs="+", default=[], help="Input .npz file(s)")
    parser.add_argument("--glob", default=None, help="Glob for input files (relative to cwd)")
    parser.add_argument("--run-standard", action="store_true",
                        help="Run standard HSC i<25 and DESILS z<22 samples for dz=0.2 and z<1")
    parser.add_argument("--mock-version", default="v2",
                        help="Mock version under data/jordan_mocks (used with --run-standard)")
    parser.add_argument("--mock-basepath", default=None,
                        help="Override mock basepath (used with --run-standard)")
    parser.add_argument("--component", default="cross",
                        choices=["cross", "gal_auto", "intensity_auto_tracer", "intensity_auto_full"],
                        help="Spectrum component to analyze")
    parser.add_argument("--assume-dl", action="store_true",
                        help="Assume inputs are already D_ell (skip C_ell to D_ell conversion)")
    parser.add_argument("--shot-ell-range", nargs=2, type=float, default=[50000.0, 80000.0],
                        metavar=("ELL_MIN", "ELL_MAX"), help="High-ell range for shot-noise estimation")
    parser.add_argument("--signal-ell-max", type=float, default=2000.0,
                        help="Maximum ell for averaging residual D_ell as A_2h")
    parser.add_argument("--cache-path", default=None,
                        help="Write results to this JSON cache path")

    args = parser.parse_args()

    basepath = None
    if args.run_standard:
        basepath = Path(args.mock_basepath) if args.mock_basepath else Path("data") / "jordan_mocks" / args.mock_version
        inputs = _collect_standard_inputs(basepath)
    else:
        inputs = list(_iter_inputs(args.input, args.glob))

    if not inputs:
        raise SystemExit("No input files. Use --input or --glob.")

    shot_range = (float(args.shot_ell_range[0]), float(args.shot_ell_range[1]))
    results: List[dict] = []
    for path in inputs:
        summary = compute_a2h(path, args.component, args.assume_dl, shot_range, args.signal_ell_max)
        results.append(summary)
        print(
            f"{summary['path']} :: A2h={summary['a2h']:.6e}  "
            f"shot={summary['shot_level']:.6e}  "
            f"assume_dl={summary['assume_dl']}  n_signal={summary['n_signal']}"
        )

    cache_path = _default_cache_path(args, basepath)
    if cache_path is not None:
        _write_cache(cache_path, results)
        print(f"Cached {len(results)} results to {cache_path}")


if __name__ == "__main__":
    main()
