# Constrained Backbone Search Phase Summary

## 1. Why this phase was run

Nick proposed a constrained backbone search rather than an unconstrained phi/psi sweep. The key idea was to keep the C-alpha anchor positions fixed as a first-order proxy for compatibility with the hexad stack, while exploring small backbone torsion changes.

This phase tested that idea in a controlled way:

- keep C-alpha anchors fixed;
- first use Nick's fixed/trans omega policy;
- generate only geometrically closed candidates;
- audit geometry before diffraction scoring;
- score only geometry-safe candidates;
- defer broad omega sensitivity until the fixed-omega branch was understood.

The goal was not to prove the full C/D mechanism yet. The goal was to determine whether a constrained, systematic backbone-search workflow is feasible and whether small compatible phi/psi perturbations move the C and D bands.

## 2. Starting model

The constrained search used the ideal Hexaflex-derived `backbone_plus_carboxylate` coordinate model as the source geometry.

Earlier rich-coordinate/add-back work identified this as the best clean C/D compromise:

- C peak near 5.745 A
- D peak near 7.276 A
- C/D combined error about 0.1698 A

This model is not being treated as experimental truth. It is being used as a controlled coordinate parent for asking which backbone changes are compatible with the stack and whether those changes move the C/D bands.

## 3. Feasibility and repeat structure

The torsion/repeat audit found that the parent coordinate model supports a constrained torsion search:

- standard N/CA/C/O atoms are present;
- phi/psi/omega torsions are extractable directly from atom names;
- C-alpha anchors are available;
- omega is trans-like, but not always exactly 180 degrees;
- two main repeat families were identified:
  - CYP->GLU
  - GLU->MEP

Important indexing note:

Raw residue IDs are not safe as repeat/register indices because chains are numbered in offset blocks. Future repeat/register work should use per-chain coordinate-order indexing, with anti-parallel metadata where needed.

## 4. Local constrained candidate pilot

A first local candidate-generation pass created 10 constrained phi/psi candidates.

Geometry audit result:

- candidates audited: 10
- safe for diffraction scoring: 7/10
- C-alpha anchors preserved
- unsafe candidates were excluded before scoring

C/D scoring result for the 7 safe local candidates:

- C peak: 5.7454 A for all scored candidates
- D peak: 7.2756 A for all scored candidates
- combined absolute C/D error: 0.1698 A for all scored candidates

Interpretation:

A single local repeat perturbation is too diluted to move global C/D peak positions in the powder/Debye scoring. This motivated coherent repeated-variant generation.

## 5. CYP->GLU fixed_180 branch

The coherent CYP->GLU branch applied the same perturbation across all equivalent CYP->GLU windows.

Generation result:

- equivalent CYP->GLU windows found: 45
- variants generated for deltas: -2, -1, 0, +1, +2, +3
- per variant: 45 attempted / 45 applied / 0 skipped
- C-alpha anchor shift: 0.0 A
- omega policy: fixed_180

Geometry audit result:

- repeated variants audited: 6
- safe for diffraction scoring: 3/6
- safe deltas: -1, 0, +1
- failed deltas: -2, +2, +3
- failures were driven by backbone bond/angle thresholds, not C-alpha anchor movement

C/D scoring result for safe variants:

- delta -1: C 5.7454 A, D 7.2756 A
- delta  0: C 5.7454 A, D 7.2756 A
- delta +1: C 5.7454 A, D 7.2756 A

Intensity sensitivity:

- C intensity span across -1, 0, +1: about 0.3147%
- D intensity span across -1, 0, +1: about 0.0290%

Interpretation:

Under fixed C-alpha anchors and fixed/trans omega, coherent CYP->GLU phi perturbations are geometry-safe only in a narrow +/-1 degree range. Within that safe range, C/D peak positions are flat, and intensity changes are tiny.

## 6. GLU->MEP fixed_180 branch

The GLU->MEP fixed-omega refinement tested whether GLU->MEP could be made geometry-safe under strict omega = 180 degrees.

Result:

- GLU->MEP windows found: 42
- chains containing GLU->MEP windows: B, D, F
- solve modes attempted: one_torsion, two_torsion, local_refine
- attempts: 21
- geometry-safe attempts: 0/21
- nonzero geometry-safe attempts: 0

Best endpoint-closed case:

- delta +1, one_torsion
- endpoint error: 0.0137 A
- failed geometry because backbone angle delta was 10.3783 degrees, above the 5 degree threshold

Interpretation:

Endpoint closure alone is not enough. Under strict fixed omega = 180 degrees, GLU->MEP does not look geometry-feasible in this refinement prototype.

## 7. GLU->MEP omega-mode comparison

Because GLU->MEP failed under strict fixed_180 omega, we tested whether fixed omega = 180 degrees was overconstraining the repeat.

Omega modes compared:

- fixed_180
- baseline_parent

Baseline parent omega values for the representative GLU->MEP window:

- omega0: -167.4556 degrees, trans deviation 12.5444 degrees
- omega1: -179.9850 degrees, trans deviation 0.0150 degrees

Result:

- fixed_180: 0/21 geometry-safe attempts
- baseline_parent: 8/21 geometry-safe attempts

Safe nonzero baseline_parent examples included:

- delta -2, one_torsion
- delta -1, one_torsion
- delta +1, one_torsion
- delta +1, local_refine
- delta +2, one_torsion
- delta +2, two_torsion
- delta +2, local_refine

Interpretation:

Strict omega = 180 degrees is too restrictive for GLU->MEP. Holding omega to the parent baseline, especially the omega0 geometry near -167.46 degrees, allows multiple GLU->MEP perturbations to pass the same geometry thresholds.

This is not an unconstrained omega scan. The result says that GLU->MEP may require parent-like, trans-but-nonideal omega geometry.

## 8. GLU->MEP baseline_parent omega branch

Using baseline_parent omega, coherent repeated GLU->MEP variants were generated.

Generation result:

- GLU->MEP windows found: 42
- chains containing GLU->MEP windows: B, D, F
- variants generated for deltas: -2, -1, 0, +1, +2
- solve mode: one_torsion
- per variant: 42 attempted / 42 applied / 0 skipped
- C-alpha anchor shift: 0.0 A
- omega mode: baseline_parent

Geometry audit result:

- variants audited: 5
- safe for diffraction scoring: 2/5
- safe deltas: -1, 0
- failed deltas: -2, +1, +2

Per-delta geometry result:

- delta -2: fail; bond delta 0.0711 A; omega trans deviation 15.0045 degrees
- delta -1: pass; bond delta 0.0354 A; angle delta 2.1784 degrees; omega deviation 13.7661 degrees
- delta  0: pass; baseline/control
- delta +1: fail; omega trans deviation 15.4154 degrees
- delta +2: fail; bond delta 0.0718 A; omega trans deviation 18.3102 degrees

C/D scoring result for safe variants:

- delta -1: C 5.7454 A, D 7.2756 A
- delta  0: C 5.7454 A, D 7.2756 A

Relative intensity/score for delta -1 versus baseline:

- C relative score: 0.9970, about -0.30%
- D relative score: 1.0022, about +0.22%

Interpretation:

Baseline-parent omega rescues a small geometry-safe GLU->MEP region that fixed_180 could not. However, within the safe region, C/D peak positions still do not move, and intensity/score changes are tiny.

## 9. Overall interpretation

This constrained-search phase supports Nick's core intuition that compatible backbone space is narrow.

The results also add an important refinement:

- fixed omega = 180 degrees is useful as a first control;
- but strict ideal omega = 180 degrees overconstrains GLU->MEP;
- parent-like trans-but-nonideal omega is important for GLU->MEP feasibility.

Across both repeat families, the geometry-safe perturbation basins are small:

- CYP->GLU fixed_180: safe only around -1, 0, +1
- GLU->MEP fixed_180: no safe nonzero variants
- GLU->MEP baseline_parent: safe only around -1 and 0

Within those safe basins, C/D peak positions remain flat at the current scoring resolution:

- C peak: 5.7454 A
- D peak: 7.2756 A

Intensity changes are detectable but very small and should not be overinterpreted.

## 10. What not to overclaim

Do not claim that backbone structure is irrelevant.

Do not claim that C/D positions cannot move under any backbone change.

Do not claim that omega must vary freely.

Do not claim that the baseline_parent omega result is an unconstrained omega scan.

Do not claim that this proves the final C/D structural mechanism.

The careful statement is:

The C-alpha anchored, geometry-audited pilot shows that the compatible backbone basin is narrow. Within the small safe perturbations tested so far, C/D peak positions are robust. GLU->MEP specifically requires parent-like nonideal trans omega to become geometrically feasible.

## 11. Recommended next branches

Option A: Coupled CYP->GLU plus GLU->MEP baseline-parent omega perturbations.

This is the closest continuation of Nick's idea. Single-repeat-family perturbations did not move C/D; coupled changes may be needed.

Option B: Small baseline-parent omega sensitivity for both repeat families.

This would test whether CYP->GLU also behaves differently when parent omega values are retained rather than idealized.

Option C: Increase peak-position/profile resolution.

The current scoring may not detect sub-bin shifts. Before expanding the chemical search too far, it may be worth checking whether higher-resolution radial profiles reveal small C/D shifts.

Option D: Better closure/minimization.

The current deterministic grid and geometry reconstruction are useful prototypes, but an external minimizer or stronger internal-coordinate optimizer may be needed before exploring wider torsion space.

## 12. Working conclusion

The constrained-backbone workflow is now viable and internally gated:

1. generate constrained candidates;
2. audit geometry;
3. score only safe structures;
4. interpret peak movement only after geometry filtering.

The current pilot suggests that small geometry-safe torsion changes do not move C/D positions, but the omega-mode comparison shows that exact omega handling matters for whether candidate backbones are feasible at all.
