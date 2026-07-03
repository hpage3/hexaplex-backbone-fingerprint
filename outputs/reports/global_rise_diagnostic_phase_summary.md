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

## 8. Overall Interpretation

C is mainly axial/rise-like sensitive. D is mainly radial/inter-strand-distance sensitive. Moderate rise-like compression improves C while preserving D. Stronger rise compression reaches the C target more closely but worsens D. Local torsion perturbations did not change the larger structural length scales enough to move C/D. The best diagnostic scale is around 0.9700, corresponding to about 3% effective rise compression.

## 9. What Not To Overclaim

- Treat these as diagnostic variants, not minimized physical structures.
- do not claim the final structure requires literal uniform 3% z-scaling.
- Do not claim backbone is irrelevant.
- Do not claim C/D sensitivity is fully solved.
- Do not treat loose global geometry gates as chemical validation.

## 10. Recommended Next Scientific Branches

- Option A: physically parameterized rise/rise-per-repeat model. Build or regenerate coordinates by changing helical rise/repeat spacing rather than globally scaling z.
- Option B: combined rise + radial compensation. Test whether mild radial adjustment can preserve D while stronger rise compression targets C.
- Option C: map diagnostic deformation back to backbone/hexad parameters. Ask what backbone/stack parameters produce an effective 3% rise compression.
- Option D: minimized/refined structural candidates. Use an external minimizer or physically constrained coordinate builder to relax the best diagnostic variants.
- Option E: prepare concise Nick/team update. Summarize the local torsion negative result plus the global/rise positive result.

## 11. Current Best Diagnostic Result

- Variant: `rise_like_0p9700`
- C peak: 5.6422 A
- D peak: 7.2756 A
- Combined absolute error: 0.0667 A
- Diagnostic interpretation: moderate effective rise compression improves C while preserving D.

## Summary Table

| phase | variants_generated | geometry_interpretable | variants_scored | best_variant | best_C_peak_A | best_D_peak_A | best_combined_abs_error_A |
| --- | --- | --- | --- | --- | --- | --- | --- |
| constrained_backbone_context |  |  |  |  |  |  |  |
| global_deformation | 12 | 12 | 12 | axial_m1 | 5.69336 | 7.27558 | 0.117776 |
| radial_axial_refinement | 25 | 25 | 25 | radial_1p0000__axial_0p9900 | 5.69336 | 7.27558 | 0.117776 |
| axial_only_extension | 7 | 7 | 7 | axial_only_0p9700 | 5.64223 | 7.27558 | 0.0666505 |
| fine_axial_profile_diagnostic | 7 | 7 | 7 | fine_axial_0p9700 | 5.64223 | 7.27558 | 0.0666505 |
| rise_like_diagnostic | 9 | 9 | 9 | rise_like_0p9700 | 5.64223 | 7.27558 | 0.0666505 |
| overall_best | 9 | 9 | 9 | rise_like_0p9700 | 5.64223 | 7.27558 | 0.0666505 |

## Missing Optional Inputs

_None._
