 To run, type
 
 ./boxed.sh --all


What the script does (in plain English)
Input: a standard PDB file containing protein backbone atoms.

Goal: for each adjacent residue pair (i, i+1) in a chain, estimate the peptide “plane” using CA(i), C(i), O(i), N(i+1), CA(i+1) and (if present) HN(i+1).

How:
Fit a best-fit plane by SVD.
Build orthonormal in-plane axes (u, v).
Project N (and HN if present) into the plane and compute an axis-aligned rectangle that tightly bounds the anchors {CA(i), O(i), CA(i+1), projected N/HN}, with optional padding.
Emit a PDB of HETATM pseudo-atoms (resn PLN) that outline each rectangle:
PV1–PV4 = rectangle corners (and CONECT bonds for the perimeter),
optional interior “horizontal” lines (PHkA/PHkB) along the long axis,
optional central point + normal line (PCEN, PN1, PN2; ±1 Å along the normal).
TER after each rectangle, CONECT records (no diagonals).

Outputs (optional):
CSVs for plane normals and adjacent signed angles; optional SVG plot of the angles.
PML to color rectangles by secondary structure (HELIX/SHEET from the input PDB) and force sticks.
PML to force sticks even without coloring.
The code is robust to missing HN: it always includes projected N in the rectangle extents so N/HN won’t sit outside.

Files it writes
If your output is out_boxes.pdb, you may also get:
out_boxes_normals.csv — plane index + (nx, ny, nz) + RMS.
out_boxes_adjacent_angles.csv — signed angle between consecutive planes (within chain).
out_boxes_angles_signed.svg — line plot (if --plot).
out_boxes_chain_<X>_normals.csv, out_boxes_chain_<X>_adjacent_angles.csv — per-chain (if --csv-per-chain).
out_boxes_color_ss.pml — color by secondary structure (and show sticks) if --color-ss.
out_boxes_as_sticks.pml — force sticks (optionally set radius) if --as-sticks.
Key implementation details (useful to know)
Proper PDB columns (altLoc/iCode included) so resn PLN and chain selections work in PyMOL.
All PLN records use the input chain ID, or a forced one via --force-chain.
PV1–PV4: corners; PHkA/PHkB: interior lines; PCEN/PN1/PN2: center/normal.
CONECT includes perimeter and interior lines (no cross-box diagonals).
Line endings are LF so tools like PyMOL are happy.

What this setup does
Script: boxed.sh
Behavior:
Looks for input PDBs only in ./input_data/
Writes all outputs to ./output/
Always runs with: --csv --plot --color-ss --force-chain A --as-sticks --stick-radius 0.15
Outputs (per input):
<name>_boxes.pdb, _normals.csv, _adjacent_angles.csv, _angles_signed.svg, _color_ss.pml, _as_sticks.pml

Folder layout
peptide_box/
├── planes_from_backbone_ortho_boxes.py
├── boxed.sh
├── input_data/        # put your .pdb files here
└── output/            # results appear here (auto-created)

One-time setup
cd /Users/darwin/code/peptide_box
chmod +x boxed.sh

Run on one or more files (no paths needed)
./boxed.sh 8oys_tim
./boxed.sh 8oys_tim 1a6m_myoglobin 3mi4_trypsin

Run in batch (everything in input_data/)
./boxed.sh --all


Check results
ls -1 output | sed 's/^/output\//'

Notes & tips
The script skips files in input_data/ that already look like outputs (names containing _boxes).
If you get “file not found,” make sure your PDBs live in input_data/:
ls -l input_data/*.pdb

Inputs / outputs
input (positional): path to an input PDB (omit if using --from-input-data).
--outdir OUTDIR — write all outputs into this directory (auto-created).
--output PATH — explicitly set the output PDB path (overrides the default <stem>_boxes.pdb inside --outdir).

Batch
--from-input-data — process every *.pdb in ./input_data/ (script-dir), one by one.
Chain / adjacency
--chain CHAIN_ID — restrict processing to a single chain (e.g., A).
--min-sep INT — reserved; keep at default.

Padding (in-plane extents)
--pad FLOAT — symmetric padding on both axes.
--pad-u FLOAT — symmetric padding along ±u (overrides --pad for u).
--pad-v FLOAT — symmetric padding along ±v (overrides --pad for v).
--pad-u-min FLOAT — padding toward u-min only.
--pad-u-max FLOAT — padding toward u-max only.
--pad-v-min FLOAT — padding toward v-min only.
--pad-v-max FLOAT — padding toward v-max only.

Drawing / geometry
--normal — add a center point and ±1 Å normal line to each rectangle.
--horiz-lines, -hl INT — number of interior “horizontal” lines (0..9).

CSV / plots
--csv — write normals and adjacent-angle CSVs.
--csv-per-chain — also write per-chain CSVs.
--plot — write an SVG plot of signed adjacent angles (title includes input file name).

Coloring / PML
--color-ss — write a PML that colors rectangles by HELIX/SHEET records (cyan/magenta; other=orange) and shows sticks.
--force-chain CHAIN_ID — override chain ID stamped in PLN output (e.g., A).
Sticks PML
--as-sticks — write a PML that loads the boxes and sets sticks.
--stick-radius FLOAT — stick radius (e.g., 0.15).

Notes:
In your setup, the wrapper always passes --csv --plot --color-ss --force-chain A --as-sticks --stick-radius 0.15.
Without --outdir, the script defaults to writing next to the input. The wrapper ensures --outdir output/.

boxed.sh (wrapper)
./boxed.sh <name1[.pdb]> [name2[.pdb] ...]
Looks for inputs at input_data/<name>.pdb, writes all outputs to output/.
./boxed.sh --all
Processes all *.pdb in input_data/ (skips any file whose name already contains _boxes).