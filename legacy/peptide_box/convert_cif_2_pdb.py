#!/usr/bin/env python
import os
import glob
import gemmi
import string

input_dir = "input_data"
letters = iter(string.ascii_uppercase + string.ascii_lowercase + string.digits)

for cif_file in glob.glob(os.path.join(input_dir, "*.cif")):
    pdb_file = cif_file.replace(".cif", ".pdb")
    map_file = cif_file.replace(".cif", ".chain_map.txt")

    if os.path.exists(pdb_file):
        continue

    with open(cif_file, "r") as f:
        first_line = f.readline().strip()

    if first_line.lower().startswith("data_"):
        print(f"[convert] Valid mmCIF -> {pdb_file}")
        try:
            structure = gemmi.read_structure(cif_file)
            structure.remove_empty_chains()

            chain_map = {}
            for model in structure:
                for chain in model:
                    if len(chain.name) > 1:  # too long for PDB
                        if chain.name not in chain_map:
                            try:
                                new_id = next(letters)
                            except StopIteration:
                                raise ValueError("Ran out of unique chain IDs!")
                            chain_map[chain.name] = new_id
                        chain.name = chain_map[chain.name]

            # Save mapping if any chains were renamed
            if chain_map:
                with open(map_file, "w") as mf:
                    for old, new in chain_map.items():
                        mf.write(f"{old} -> {new}\n")
                print(f"[map] Chain remap written to {map_file}")

            structure.write_pdb(pdb_file)

        except Exception as e:
            print(f"[error] Failed to convert {cif_file}: {e}")

    elif first_line.startswith("HEADER") or first_line.startswith("COMPND"):
        print(f"[rename] {cif_file} is actually a PDB format — renaming")
        if os.path.exists(pdb_file):
            # If the PDB already exists, just remove the .cif copy
            os.remove(cif_file)
            print(f"[skip] {pdb_file} already exists, deleted {cif_file}")
        else:
            os.rename(cif_file, pdb_file)