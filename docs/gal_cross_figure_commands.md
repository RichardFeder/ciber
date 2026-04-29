# z<1 reconstructed auto figure command reference

This file records the exact commands used to regenerate:

- figures/gal_cross_paper/ciber_auto_vs_predicted_from_zlt1_cross1h_alt2h_lmax50000.pdf

## Standard refresh (reuse cached fit outputs)

From repo root:

conda run --no-capture-output -n ciber python scripts/generate_gal_cross_paper_figures.py --overwrite --outdir figures/gal_cross_paper zlt1-cross1h-refit-update --fit-tag zlt1_cross1h_alt2h --ell-max-list 50000

Notes:
- This is the normal fast command for figure-only styling updates.
- It reuses cached cross/auto fit NPZ files when present.

## Force full refit then regenerate figure

conda run --no-capture-output -n ciber python scripts/generate_gal_cross_paper_figures.py --overwrite --outdir figures/gal_cross_paper zlt1-cross1h-refit-update --fit-tag zlt1_cross1h_alt2h --ell-max-list 50000 --force-refit

Use this only when model/data settings changed and cached fits should not be reused.

## Optional explicit cache flag (same behavior as standard)

conda run --no-capture-output -n ciber python scripts/generate_gal_cross_paper_figures.py --overwrite --outdir figures/gal_cross_paper zlt1-cross1h-refit-update --fit-tag zlt1_cross1h_alt2h --ell-max-list 50000 --reuse-fit-cache

## Check output timestamp

ls -l figures/gal_cross_paper/ciber_auto_vs_predicted_from_zlt1_cross1h_alt2h_lmax50000.pdf
