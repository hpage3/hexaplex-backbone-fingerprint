#!/usr/bin/env python
import os
import glob
import gemmi

input_dir = "input_data"
for cif_file in glob.glob(os.path.join(input_dir, "*.cif")):
    pdb_file = cif_file.replace(".cif", ".pdb")
    if os.path.exists(pdb_file):
        continue

    print(f"[convert] {cif_file} -> {pdb_file}")
    structure = gemmi.read_structure(cif_file)
    structure.remove_empty_chains()
    structure.write_pdb(pdb_file)
