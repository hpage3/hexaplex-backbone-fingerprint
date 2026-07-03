# Hexaplex Backbone Fingerprint — Agent Instructions

This repository supports Hexaplex/Hexaflex structural diagnostics and diffraction-model interpretation.

## Project rules

- Do not modify raw input files.
- Do not commit unless explicitly asked.
- Keep generated outputs under `outputs/`.
- Prefer small, deterministic scripts.
- Prefer reading PDB, CSV, YAML, JSON, and report files from disk rather than pasting their contents into prompts.
- Use small synthetic fixtures in tests.
- Keep tests fast and deterministic.
- Use PowerShell-compatible commands in reports and handoff notes.

## Output discipline

Avoid large terminal output. Do not print or paste:

- full PDB files;
- full XYZ files;
- full CSV tables;
- full radial profiles;
- full DataFrames;
- long coordinate arrays;
- large git diffs;
- long traceback logs unless needed to diagnose a failure.

When reporting results, prefer only:

- pytest result;
- files changed;
- files created;
- top-level metrics;
- a short interpretation;
- `git status --short`.

## Scientific context

Current best diagnostic structure:

- `parameterized_rise_0p9750`
- C = 5.6422 A
- D = 7.2756 A
- combined C/D absolute error = 0.0667 A

Current interpretation:

- Local C-alpha-anchored torsion perturbations did not meaningfully move C/D.
- D is radial/inter-strand-distance sensitive.
- C is axial/rise-sensitive.
- The current best model should be described as an effective computational z-layer or rise-like compression, not as a final physically minimized structure.
- The next scientific goal is to map this diagnostic axial/rise compression onto a chemically/register-defined structural model.

## Important cautions

Do not claim:

- the structure is solved;
- the current diagnostic coordinate transform is a minimized physical model;
- the 45 computational z-slices are validated physical hexad layers;
- a literal uniform 2.5 percent compression is the final structure.

Preferred wording:

"The current diagnostics identify an axial/rise degree of freedom that controls the C band while preserving D over a useful range. The next step is to express that displacement in chemically meaningful structural parameters rather than as a coordinate transform."

## Useful current files

Parent baseline PDB:

`outputs/coordinates/ideal_hexaflex_variants/ideal_hexaflex_backbone_plus_carboxylate.pdb`

Best current diagnostic PDB:

`outputs/coordinates/parameterized_rise_variants/parameterized_rise_0p9750.pdb`

Generic rise-like diagnostic PDB:

`outputs/coordinates/rise_like_variants/rise_like_0p9700.pdb`

Over-compressed diagnostic PDB:

`outputs/coordinates/parameterized_rise_variants/parameterized_rise_0p9600.pdb`

Global/rise summary:

`outputs/reports/global_rise_diagnostic_phase_summary.md`

Key structure visualization report:

`outputs/reports/key_structure_variant_visualization.md`
