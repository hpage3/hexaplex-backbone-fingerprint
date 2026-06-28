@echo off
if not exist qc mkdir qc
echo [info] Generating PyMOL overlays for systematic high-mean set...
for /f %%P in (systematic_ids_new.txt) do (
  if exist output\%%P_boxes.pdb if exist input_data\%%P.pdb (
    echo [gen] %%P
    python generate_normals_pml.py --boxes output\%%P_boxes.pdb --src input_data\%%P.pdb --out qc\%%P_overlay.pml
  ) else (
    echo [skip] %%P: missing output\%%P_boxes.pdb or input_data\%%P.pdb 1>&2
  )
)