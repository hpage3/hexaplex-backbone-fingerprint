# Global/Rise Diagnostic Phase Summary

## 1. Why This Phase Was Run

The constrained local torsion branch did not move C/D peak positions meaningfully. The next question was whether larger global geometric dimensions could move C and D in a controlled way. The goal here was diagnostic sensitivity, not finalized structural modeling.

## 2. Starting Point From Constrained-Backbone Phase

Local C-alpha anchored torsion variants were geometry-safe in narrow basins, but C/D peak positions were robust. CYP->GLU-only, GLU->MEP-only, and coupled CYP->GLU + GLU->MEP variants did not move C/D peak positions, and coupled profile diagnostics showed only tiny sub-bin movement. This motivated the global deformation tests.

## 3. Global Deformation Pilot

The global deformation pilot generated 12 variants and all 12 were geometry-interpretable. Radial mode moved D, axial mode moved C, and twist/anisotropic modes were flat. The best early global variant was `axial_m1`, with C near 5.6934 A, D near 7.2756 A, and combined error about 0.1178 A. Largest RMSD was about 0.1240 A and largest max displacement was about 0.1664 A.

## 4. Focused Radial/Axial Refinement

The focused grid generated 25 variants, with 25/25 geometry-interpretable and 25 scored. Axial tuning mainly controls C, radial tuning mainly controls D, and cross-coupling appears near peak-picking/bin thresholds. The best representative was `radial_1p0000__axial_0p9900`, with C near 5.6934 A, D near 7.2756 A, and combined error about 0.1178 A.

## 5. Axial-Only Extension

The axial-only branch compressed axial scale with radial scale fixed at 1.0000. The best variant was `axial_only_0p9700`: C 5.6422 A, D 7.2756 A, combined error 0.0667 A. D stayed stable across 0.9700 to 1.0000, while the picked C response was discretized by peak-picking/binning.

## 6. Fine Axial Profile Diagnostics

Fine axial diagnostics showed that the C profile moves smoothly underneath discretized picked peaks, while D remains picked-position stable. Max profile shifts versus 0.9700 were C: centroid 0.00406336 A, parabolic 0.0559768 A; D: centroid 0.000481402 A, parabolic 0.0249334 A. This supports real C-profile movement rather than random peak-picking noise.

## 7. Rise-Like Diagnostic Branch

The rise-like branch tested axial_rise_scale from 0.9600 to 1.0000. The best combined result was `rise_like_0p9700`: C 5.6422 A, D 7.2756 A, combined error 0.0667 A. C reaches about 5.5920 A at 0.9600/0.9650, but D drops to about 7.1923 A there. Thus 0.9700 is the best combined diagnostic compromise: it improves C while preserving D. Rise-like profile shifts versus baseline were C: centroid 0.0103378 A, parabolic 0.155359 A; D: centroid 0.00116501 A, parabolic 0.0616218 A.

## Parameterized rise diagnostic branch

This branch was run because the generic rise_like branch was useful but still used continuous global z-scaling. The parameterized branch inferred 45 axial layers from C-alpha z positions, estimated a mean parent layer rise of about 1.1829 A, preserved each atom's local offset from its assigned layer center, and then moved only the layer centers according to the rise scale.

The best parameterized result was `parameterized_rise_0p9750`: C 5.6422 A, D 7.2756 A, combined error 0.0667 A. It ties the generic `rise_like_0p9700` C/D score, but is more interpretable because the same improvement appears in a layer/repeat-aware model rather than only uniform z-scaling. C still moves toward 5.6 A with compression. D remains stable at 7.2756 A from 0.9750 through 1.0000, but drops to 7.1923 A at 0.9600, 0.9650, and 0.9700. Preserving within-layer z offsets changes the D threshold behavior slightly. Parameterized profile shifts versus baseline were C: centroid 0.0103134 A, parabolic 0.155171 A; D: centroid 0.00118915 A, parabolic 0.062937 A.

## Register-defined layer model audit

This audit was run to validate whether the 45 z-slice layers used by the parameterized rise branch are chemically/register meaningful. It compared five models: `z_slice_layer`, `residue_index_layer`, `ca_register_layer`, `repeat_pair_layer`, and `peptide_plane_layer`.

Main contrast: z-slice layers are geometrically regular but split most residues; register-defined layers avoid residue splitting and are chemically cleaner but geometrically diffuse in this antiparallel/offset parent structure.

- `z_slice_layer`: 45 layers, 174/180 split residues, median spacing about 1.1768 A.
- `residue_index_layer`: 30 layers, 0 split residues, median thickness about 27.0555 A.
- `ca_register_layer`: 30 layers, 0 split residues, median thickness about 26.0625 A.
- `repeat_pair_layer`: 15 layers, 0 split residues, median thickness about 27.2300 A.
- `peptide_plane_layer`: 30 layers, 0 split residues, median thickness about 24.8489 A.

Mean z-slice overlap counts were: residue_index_layer 5.73, ca_register_layer 1.97, repeat_pair_layer 7.60, and peptide_plane_layer 5.73.

Conservative conclusion: the z-slice model is useful as a computational deformation coordinate; register-defined layers are chemically cleaner but geometrically diffuse. Do not call the 45 z-slices physical hexad layers yet.

## 8. Overall Interpretation

Local C-alpha anchored torsion basin did not move C/D. Global/rise diagnostics did move C/D. D is mainly radial/inter-strand-distance sensitive. C is mainly axial/rise-like sensitive. The best current diagnostic result is `parameterized_rise_0p9750`: C 5.6422 A, D 7.2756 A, combined absolute error 0.0667 A. It corresponds to about 2.5% effective layer-rise compression in the diagnostic layer model. The register-defined audit shows the current layer model should be described as effective computational z-layer compression, pending mapping to a chemically/register-defined structural model; it is not chemically definitive. Stronger compression still improves C, but begins to damage D.

## 9. What Not To Overclaim

- Treat these as diagnostic variants, not minimized physical structures.
- do not claim the final structure requires literal uniform 3% z-scaling.
- Treat the layer-aware parameterized rise model as not fully physical or minimized.
- Do not claim the inferred 45 layers are uniquely defined structural layers without further validation.
- Do not claim the optimal scale is exact; peak-picking/binning discretization still matters.
- Do not claim the 45 z-slices are physical hexad layers.
- Do not claim `parameterized_rise_0p9750` is a validated physical rise-per-hexad model.
- Do not claim register-defined residue/peptide-plane layers reproduce the z-slice rise result.
- Do not claim the model is chemically minimized.
- Do not claim backbone is irrelevant.
- Do not claim C/D sensitivity is fully solved.
- Do not treat loose global geometry gates as chemical validation.

## 10. Recommended Next Scientific Branches

- Option A: chemically annotated register model. Use known/validated hexad or repeat-register annotations rather than purely z-derived layers.
- Option B: physically rebuilt helical/rise model. Generate coordinates from helical/rise parameters instead of transforming the parent coordinates.
- Option C: repeat-pair or peptide-plane constrained rise model. Even though diffuse, these are chemically cleaner and may be useful if paired with additional structural constraints.
- Option D: rise + radial compensation. Only after a more physical rise model is defined, test whether radial compensation can preserve D while targeting C.
- Option E: external minimization/refinement. Relax diagnostic candidates under structural restraints to test whether the effective rise compression is chemically feasible.

## 11. Current Best Diagnostic Result

- Variant: `parameterized_rise_0p9750`
- C peak: 5.6422 A
- D peak: 7.2756 A
- Combined absolute error: 0.0667 A
- Diagnostic interpretation: `parameterized_rise_0p9750` is the best current diagnostic result, but after the register-defined layer audit it should be described as effective computational z-layer compression, pending mapping to a chemically/register-defined structural model.

## Summary Table

| phase | variants_generated | geometry_interpretable | variants_scored | best_variant | best_C_peak_A | best_D_peak_A | best_combined_abs_error_A |
| --- | --- | --- | --- | --- | --- | --- | --- |
| constrained_backbone_context |  |  |  |  |  |  |  |
| global_deformation | 12 | 12 | 12 | axial_m1 | 5.69336 | 7.27558 | 0.117776 |
| radial_axial_refinement | 25 | 25 | 25 | radial_1p0000__axial_0p9900 | 5.69336 | 7.27558 | 0.117776 |
| axial_only_extension | 7 | 7 | 7 | axial_only_0p9700 | 5.64223 | 7.27558 | 0.0666505 |
| fine_axial_profile_diagnostic | 7 | 7 | 7 | fine_axial_0p9700 | 5.64223 | 7.27558 | 0.0666505 |
| rise_like_diagnostic | 9 | 9 | 9 | rise_like_0p9700 | 5.64223 | 7.27558 | 0.0666505 |
| parameterized_rise_diagnostic | 9 | 9 | 9 | parameterized_rise_0p9750 | 5.64223 | 7.27558 | 0.0666505 |
| updated_overall_best | 9 | 9 | 9 | parameterized_rise_0p9750 | 5.64223 | 7.27558 | 0.0666505 |
| register_defined_layer_audit | 0 | not applicable | 0 | not applicable |  |  |  |
| final_current_interpretation | 9 | 9 | 9 | parameterized_rise_0p9750 | 5.6422 | 7.2756 | 0.0667 |

## Missing Optional Inputs

_None._
