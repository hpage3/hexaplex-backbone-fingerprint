# Hexaplex Backbone Fingerprint

This repository analyzes Hexaflex/Hexaplex polypeptide backbone geometry.
The initial focus is peptide-plane fingerprints derived from backbone atoms in
structural models.

The biological and structural question is whether recurring backbone distances,
peptide-plane orientations, and plane distortions map onto the experimental
powder/fiber diffraction bands currently called:

- C band target: 5.6 Angstrom
- D band target: 7.3 Angstrom

Initial outputs include plane features, adjacent-plane features, and candidate
C/D distance pairs. This repository is separate from diffraction simulation and
scoring repositories; it is meant to provide backbone geometry fingerprints that
can later be compared with diffraction-derived hypotheses.

## Layout

- `input_data/`: raw coordinate files for analysis.
- `outputs/`: generated analysis outputs.
- `scripts/`: command-line entry points.
- `src/hexaplex_backbone_fingerprint/`: importable Python package.
- `legacy/peptide_box/`: preserved reference code from the earlier
  `peptide_box` workflow.

Large or raw coordinate files should be placed in `input_data/` but are not
automatically tracked. Generated files belong in `outputs/`.

## Quick Start

```powershell
python -m pip install -e .
python scripts/analyze_backbone_fingerprint.py input_data/model.pdb --outdir outputs/model_test --c-target 5.6 --d-target 7.3 --tol 0.25
```

Run the smoke tests:

```powershell
python -m pytest
```
