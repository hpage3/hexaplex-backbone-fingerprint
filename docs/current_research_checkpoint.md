# Current Research Checkpoint

Last updated after commit:

`e6f80ad Improve axial displacement visualization`

## Purpose

This file is a compact checkpoint for Codex and future analysis sessions. It is intended to reduce repeated prompt context and avoid large copy/paste handoffs.

## Current best diagnostic result

Best current diagnostic structure:

- structure id: `parameterized_rise_0p9750`
- C peak: 5.6422 A
- D peak: 7.2756 A
- combined C/D absolute error: 0.0667 A

Reference structures:

- parent baseline:
  `outputs/coordinates/ideal_hexaflex_variants/ideal_hexaflex_backbone_plus_carboxylate.pdb`
- best parameterized rise diagnostic:
  `outputs/coordinates/parameterized_rise_variants/parameterized_rise_0p9750.pdb`
- generic rise-like diagnostic:
  `outputs/coordinates/rise_like_variants/rise_like_0p9700.pdb`
- over-compressed parameterized diagnostic:
  `outputs/coordinates/parameterized_rise_variants/parameterized_rise_0p9600.pdb`

## Main scientific interpretation

The current diagnostics indicate:

- C is axial/rise-sensitive.
- D is radial/inter-strand-distance sensitive.
- Small local C-alpha-anchored backbone torsion perturbations did not meaningfully move C/D peak positions.
- Controlled global/rise-like diagnostics did move C.
- Controlled radial diagnostics moved D.
- Small twist and anisotropic xy deformations did not meaningfully move C or D.

The current best model should be described as a diagnostic structural direction, not a solved physical structure.

Preferred wording:

"The current diagnostics identify an axial/rise degree of freedom that controls the C band while preserving D over a useful range. The next step is to express that displacement in chemically meaningful structural parameters rather than as a coordinate transform."

## Important caution

Do not claim:

- the structure is solved;
- the current diagnostic coordinate transform is a physically minimized model;
- the 45 computational z-slices are validated physical hexad layers;
- a literal uniform 2.5 percent compression is the final structure.

## Key completed phases

### Local constrained torsion phase

C-alpha anchored CYP->GLU, GLU->MEP, and coupled CYP->GLU plus GLU->MEP variants were generated, geometry-audited, and C/D-scored.

Result:

Geometry-safe local torsion variants did not meaningfully move C/D peak positions. Profile diagnostics supported that this was not only a peak-picking artifact.

### Global/rise diagnostic phase

Controlled deformation diagnostics showed:

- radial xy scaling strongly affects D;
- axial/rise-like compression affects C;
- twist and anisotropic xy modes were comparatively flat for C/D.

Best generic rise-like diagnostic:

- `rise_like_0p9700`
- C = 5.6422 A
- D = 7.2756 A
- combined C/D absolute error = 0.0667 A

### Parameterized rise phase

Layer-aware parameterized rise diagnostics inferred 45 computational z-layers and preserved within-layer z offsets while changing layer spacing.

Best parameterized result:

- `parameterized_rise_0p9750`
- C = 5.6422 A
- D = 7.2756 A
- combined C/D absolute error = 0.0667 A

Layer/register audit showed these 45 z-layers are computational slices and should not be treated as validated physical hexad layers.

### Visualization phase

Key structure visualization workflow now produces:

- `outputs/figures/key_structure_variant_overview.png`
- `outputs/figures/key_structure_variant_displacements.png`
- `outputs/figures/key_structure_variant_axial_profiles.png`
- `outputs/figures/key_structure_variant_geometry_summary.png`
- `outputs/reports/key_structure_variant_visualization.md`
- `outputs/metrics/key_structure_variant_visual_summary.csv`

The corrected axial displacement plot shows inward axial displacement relative to the parent: opposite ends of the structure move toward the center. Stronger compression gives larger inward displacement while the mean C-alpha radius remains essentially unchanged.

## Best next scientific task

Move from diagnostic coordinate transform to physically parameterized modeling.

Immediate next task:

Audit whether source files, YAMLs, manifests, pNAB inputs, or generation scripts contain explicit helical/generative parameters such as:

- helical rise;
- helical twist;
- chain radius;
- inter-strand radius;
- register offset;
- hexad-to-hexad spacing;
- repeat spacing;
- pNAB settings.

If explicit source parameters exist, build the next branch around those parameters.

If source parameters do not exist, derive an approximate coordinate-based helical/rise model from the parent PDB and use that as the physical parameterization bridge.
