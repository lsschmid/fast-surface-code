# Workshop paper (QCE26 / CEVNAC 2026)

IEEE conference-format draft. **Target: 4 pages including references**
(currently exactly 4 pages).

Status: full prose draft — all sections written, figures and tables in place,
references complete.

## Files
- `main.tex` — the paper (IEEEtran `conference` class).
- `references.bib` — bibliography (journal names IEEE-abbreviated, long
  author lists as `and others`).
- `figures/` — exported PDF figures (committed, so the build is
  self-contained).

## Figures
Regenerate from the evaluation scripts, then copy here:
```bash
cp ../evaluation/patch_overview.pdf figures/patch_overview.pdf   # plot_patch_overview.py
cp ../evaluation/aod_moves.pdf      figures/aod_moves.pdf        # plot_aod_moves.py
cp ../evaluation/error_prop_X.pdf   figures/error_prop_X.pdf     # plot_error_prop_cnots.py
cp ../evaluation/error_prop_Z.pdf   figures/error_prop_Z.pdf     # plot_error_prop_cnots.py
cp ../evaluation/plot.pdf           figures/benchmark.pdf        # plot_results.py
# stab_slices is a vector SVG -> convert to PDF (needs librsvg / rsvg-convert):
rsvg-convert ../evaluation/stab_slices.svg -f pdf -o figures/stab_slices.pdf
```
The stabilizer figure falls back to a placeholder box if `figures/stab_slices.pdf`
is missing, so the document still compiles without the conversion.

## Build
```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```
(Needs the `IEEEtran` class, standard in TeX Live / MacTeX.)

## Notes
- IEEE/QCE26 requires disclosing AI-generated text/figures/code; the
  disclosure is folded into the Acknowledgment section.
- A preprint on arXiv does not count as prior work for QCE26.
